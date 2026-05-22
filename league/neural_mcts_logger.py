import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detective_engine import DetectiveEngine
from game import Game
from gnn_detective_engine import GNNDetectiveEngine
from gnn_mrx_engine import RELATIONS, build_dense_adj, build_node_static_features, is_reveal_turn
from neural_mrx_mcts_engine import NeuralMrXMCTSEngine
from utility import _min_detective_distance

from league.common import best_model, ensure_project_root, now_tag, seed_everything, write_json


N_NODES = 199
N_REL = len(RELATIONS)
ACTION_DIM = N_NODES * N_REL
GAMMA = 0.95
R_TIMEOUT = 10.0
R_CAPTURE = -10.0
R_STEP = 0.05
R_DIST_COEF = 0.10


def _write_dataframe(df, parquet_path):
    try:
        df.to_parquet(parquet_path, index=False)
        return Path(parquet_path)
    except ImportError:
        pickle_path = Path(parquet_path).with_suffix(".pkl")
        df.to_pickle(pickle_path)
        return pickle_path


def _read_dataframe(path):
    path = Path(path)
    if path.suffix == ".pkl":
        return pd.read_pickle(path)
    return pd.read_parquet(path)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _dumps(obj):
    return json.dumps(obj, default=_json_default, separators=(",", ":"))


def _flat_action(destination, ticket):
    if ticket not in RELATIONS:
        return -1
    return (int(destination) - 1) * N_REL + RELATIONS.index(ticket)


def _discounted_returns(rewards, gamma):
    returns = [0.0] * len(rewards)
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running = float(rewards[idx]) + gamma * running
        returns[idx] = running
    return returns


def _play_detective_phase(game, engine, policy):
    for detective_id in range(game.num_detectives):
        policy.play_detective_turn(game, engine.belief_state, detective_id)
        if game.check_victory(silent=True):
            return True
        engine.kalman_filter()
    game.detectives_moves.append(game.detectives_pos[:])
    return False


def _target_policy_dense(search_info, target_action):
    dense = np.zeros(ACTION_DIM, dtype=np.float16)
    actions = search_info.get("actions") or []
    total_visits = sum(int(a.get("visits", 0)) for a in actions)
    if total_visits > 0:
        for action in actions:
            flat = int(action["flat_action"])
            visits = int(action.get("visits", 0))
            if 0 <= flat < ACTION_DIM and visits > 0:
                dense[flat] = np.float16(visits / total_visits)
    elif 0 <= target_action < ACTION_DIM:
        dense[target_action] = np.float16(1.0)
    return dense


def _policy_entropy(search_info):
    actions = search_info.get("actions") or []
    total_visits = sum(int(a.get("visits", 0)) for a in actions)
    if total_visits <= 0:
        return 0.0
    entropy = 0.0
    for action in actions:
        visits = int(action.get("visits", 0))
        if visits <= 0:
            continue
        p = visits / total_visits
        entropy -= p * math.log(p)
    return entropy


class PartWriter:
    def __init__(self, out_dir, games_per_part):
        self.out_dir = Path(out_dir)
        self.games_per_part = int(games_per_part)
        self.part_id = 0
        self.parts = []
        self.records = []
        self.node_dyn = []
        self.glob = []
        self.legal_mask = []
        self.target_action = []
        self.target_policy = []

    def append(self, record, sample, target_policy):
        self.records.append(record)
        self.node_dyn.append(sample["node_dyn"].astype(np.float16))
        self.glob.append(sample["glob"].astype(np.float32))
        self.legal_mask.append(sample["legal_mask"].astype(np.bool_))
        self.target_action.append(int(record["target_action"]))
        self.target_policy.append(target_policy.astype(np.float16))

    def flush(self):
        if not self.records:
            return
        base = f"part_{self.part_id:03d}"
        samples_path = self.out_dir / f"samples_{base}.parquet"
        tensors_path = self.out_dir / f"tensors_{base}.npz"
        samples_path = _write_dataframe(pd.DataFrame(self.records), samples_path)
        np.savez_compressed(
            tensors_path,
            node_dyn=np.asarray(self.node_dyn, dtype=np.float16),
            glob=np.asarray(self.glob, dtype=np.float32),
            legal_action_mask=np.asarray(self.legal_mask, dtype=np.bool_),
            target_action=np.asarray(self.target_action, dtype=np.int64),
            target_policy=np.asarray(self.target_policy, dtype=np.float16),
        )
        self.parts.append(
            {
                "samples": os.fspath(samples_path),
                "tensors": os.fspath(tensors_path),
                "rows": int(len(self.records)),
            }
        )
        print(f"Saved {base}: {len(self.records)} rows")
        self.part_id += 1
        self.records = []
        self.node_dyn = []
        self.glob = []
        self.legal_mask = []
        self.target_action = []
        self.target_policy = []


def run_logged_game(game_id, seed, teacher, detective_policy, writer):
    seed_everything(seed)
    teacher.reset()
    detective_policy.reset()

    game = Game()
    engine = DetectiveEngine(game.detectives_pos)
    local_indices = []
    rewards = []
    blocked = 0
    decisions = 0

    while True:
        if game.check_victory(silent=True):
            break

        if _play_detective_phase(game, engine, detective_policy):
            if rewards:
                rewards[-1] += R_CAPTURE
            break

        sample = teacher.mrx_policy.build_input(game, engine.belief_state)
        if not sample["legal_mask"].any():
            blocked += 1
            game.x_automated_turn(game.mrx_pos, None)
            if game.check_victory(silent=True):
                if rewards:
                    rewards[-1] += R_TIMEOUT if game.winner == 1 else R_CAPTURE
                break
            engine.update_belief_after_mrx_move(None)
            if is_reveal_turn(game.turn):
                engine.mrx_is_spotted(game.mrx_pos)
            detective_policy.observe_mrx_move(game, None)
            teacher.observe_mrx_move(game, None)
            continue

        destination, ticket = teacher.choose_action(game, engine.belief_state)
        search_info = teacher.last_search_info or {}
        target = _flat_action(destination, ticket)
        if target < 0 or not sample["legal_mask"][target]:
            raise AssertionError(
                f"Neural MCTS target outside legal mask: game={game_id} "
                f"turn={game.turn} action=({destination}, {ticket})"
            )

        game.x_automated_turn(destination, ticket)
        min_dist = _min_detective_distance(game.detectives_pos, game.mrx_pos)
        reward = R_STEP + R_DIST_COEF * float(min_dist)
        done = False
        if game.check_victory(silent=True):
            reward += R_TIMEOUT if game.winner == 1 else R_CAPTURE
            done = True

        selected = search_info.get("selected") or {}
        target_policy = _target_policy_dense(search_info, target)
        visit_policy = [
            {"flat": int(i), "prob": float(p)}
            for i, p in enumerate(target_policy)
            if float(p) > 0.0
        ]
        record = {
            "game_id": int(game_id),
            "decision_id": int(decisions),
            "turn": int(game.turn),
            "mrx_pos_after": str(game.mrx_pos),
            "detectives_pos": _dumps(game.detectives_pos),
            "target_destination": str(destination),
            "target_ticket": ticket,
            "target_action": int(target),
            "target_in_legal_mask": True,
            "visit_policy_json": _dumps(visit_policy),
            "visit_entropy": float(_policy_entropy(search_info)),
            "selected_visits": int(selected.get("visits", 0)),
            "root_actions": int(len(search_info.get("actions") or [])),
            "root_value": float(search_info.get("root_value", 0.0)),
            "selected_q": float(selected.get("q_value", 0.0)),
            "selected_prior": float(selected.get("prior", 0.0)),
            "reward": float(reward),
            "done": bool(done),
            "min_dist_after_mrx": int(min_dist),
        }
        writer.append(record, sample, target_policy)
        local_indices.append(len(writer.records) - 1)
        rewards.append(float(reward))
        decisions += 1

        if done:
            break

        engine.update_belief_after_mrx_move(ticket)
        if is_reveal_turn(game.turn):
            engine.mrx_is_spotted(game.mrx_pos)
        detective_policy.observe_mrx_move(game, ticket)
        teacher.observe_mrx_move(game, ticket)

    returns = _discounted_returns(rewards, GAMMA)
    start = len(writer.records) - len(returns)
    for offset, ret in enumerate(returns):
        writer.records[start + offset]["return_to_go"] = float(ret)

    return {
        "game_id": int(game_id),
        "seed": int(seed),
        "winner": int(game.winner),
        "mrx_win": int(game.winner == 1),
        "final_turn": int(game.turn),
        "logged_decisions": int(decisions),
        "blocked_decisions": int(blocked),
    }


def run(args):
    ensure_project_root()
    seed_everything(args.seed_base)
    run_tag = args.run_tag or now_tag()
    mrx = best_model("mrx", root=args.root)
    detectives = best_model("detectives", root=args.root)
    mrx_checkpoint = args.mrx_checkpoint or mrx["path"]
    detective_checkpoint = args.detective_checkpoint or detectives["path"]

    out_dir = Path(args.output_dir) / f"neural_mcts_logs_{run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "static_graph.npz",
        dense_adj=build_dense_adj().astype(np.float32),
        node_static=build_node_static_features().astype(np.float32),
        relations=np.array(RELATIONS),
        node_dyn_dim=np.array([14], dtype=np.int64),
        global_dim=np.array([28], dtype=np.int64),
    )

    teacher = NeuralMrXMCTSEngine(
        mrx_checkpoint_path=mrx_checkpoint,
        detective_checkpoint_path=detective_checkpoint,
        device=args.device,
        num_simulations=args.simulations,
        root_dirichlet=args.root_dirichlet,
        temperature=args.temperature,
    )
    detective_policy = GNNDetectiveEngine(
        checkpoint_path=detective_checkpoint,
        device=args.device,
    )
    writer = PartWriter(out_dir, games_per_part=args.games_per_part)
    stats = []
    started = time.time()

    print("Output:", out_dir)
    print("Mr.X teacher base:", mrx_checkpoint)
    print("Detective opponent:", detective_checkpoint)
    print("Neural MCTS simulations:", args.simulations)

    for game_id in range(args.games):
        stats.append(
            run_logged_game(
                game_id=game_id,
                seed=args.seed_base + game_id,
                teacher=teacher,
                detective_policy=detective_policy,
                writer=writer,
            )
        )
        if (game_id + 1) % args.games_per_part == 0:
            writer.flush()
        if (game_id + 1) % args.log_every == 0:
            recent = stats[-args.log_every :]
            wr = np.mean([s["mrx_win"] for s in recent])
            turn = np.mean([s["final_turn"] for s in recent])
            print(f"games={game_id + 1}/{args.games} recent_wr={wr*100:.1f}% turn={turn:.1f}")

    writer.flush()
    stats_df = pd.DataFrame(stats)
    stats_path = out_dir / "game_stats.parquet"
    stats_path = _write_dataframe(stats_df, stats_path)
    sample_parts = [_read_dataframe(p["samples"]) for p in writer.parts]
    samples_df = pd.concat(sample_parts, ignore_index=True) if sample_parts else pd.DataFrame()
    sanity = {
        "n_games": int(args.games),
        "n_samples": int(len(samples_df)),
        "mrx_winrate": float(stats_df["mrx_win"].mean()) if len(stats_df) else None,
        "avg_final_turn": float(stats_df["final_turn"].mean()) if len(stats_df) else None,
        "target_in_legal_mask_rate": (
            float(samples_df["target_in_legal_mask"].mean()) if len(samples_df) else None
        ),
        "ticket_counts": (
            {str(k): int(v) for k, v in samples_df["target_ticket"].value_counts().items()}
            if len(samples_df)
            else {}
        ),
    }
    manifest = {
        "schema_version": 1,
        "run_tag": run_tag,
        "generated_at": now_tag(),
        "teacher": "neural_mcts",
        "mrx_base_model_id": mrx["id"],
        "mrx_checkpoint": os.fspath(mrx_checkpoint),
        "detective_model_id": detectives["id"],
        "detective_checkpoint": os.fspath(detective_checkpoint),
        "config": vars(args),
        "parts": writer.parts,
        "game_stats": os.fspath(stats_path),
        "sanity": sanity,
        "elapsed_seconds": time.time() - started,
    }
    write_json(out_dir / "manifest.json", manifest)
    print(json.dumps(sanity, indent=2))
    print("Manifest:", out_dir / "manifest.json")
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description="Log Neural MCTS teacher games for Mr.X SL.")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--mrx-checkpoint", default=None)
    parser.add_argument("--detective-checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="/kaggle/working/neural_mcts_logs")
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--seed-base", type=int, default=20260522)
    parser.add_argument("--games-per-part", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--root-dirichlet", action="store_true")
    return parser


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
