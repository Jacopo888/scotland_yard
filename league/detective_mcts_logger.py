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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detective_engine import DetectiveEngine
from game import Game
from gnn_detective_engine import build_dense_adj, build_node_static_features
from league.common import best_model, ensure_project_root, now_tag, seed_everything, write_json
from league.detective_belief_mcts import (
    BeliefMCTSConfig,
    BeliefMCTSDetectiveTeacher,
    _apply_detective_move,
    _legal_destinations,
)
from league.train_detective_rl_vs_latest_mrx import (
    _opponent_mix_summary,
    _policy_for_plan,
    _training_opponent_plans,
)
from utility import _min_detective_distance


N_NODES = 199
GAMMA = 0.95
R_CAPTURE = 10.0
R_TIMEOUT = -10.0
R_STEP = -0.05
R_DIST_COEF = -0.10


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


def _discounted_returns(rewards, gamma):
    returns = [0.0] * len(rewards)
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running = float(rewards[idx]) + gamma * running
        returns[idx] = running
    return returns


def _target_policy_dense(search_info, target_action):
    dense = np.zeros(N_NODES, dtype=np.float16)
    actions = search_info.get("actions") or []
    total_visits = sum(int(action.get("visits", 0)) for action in actions)
    if total_visits > 0:
        for action in actions:
            flat = int(action["target_action"])
            visits = int(action.get("visits", 0))
            if 0 <= flat < N_NODES and visits > 0:
                dense[flat] = np.float16(visits / total_visits)
    elif 0 <= target_action < N_NODES:
        dense[target_action] = np.float16(1.0)
    return dense


def _policy_entropy(search_info):
    actions = search_info.get("actions") or []
    total_visits = sum(int(action.get("visits", 0)) for action in actions)
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


def _rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _value_diagnostics(samples_df):
    if samples_df.empty or "return_to_go" not in samples_df:
        return {}
    returns = pd.to_numeric(samples_df["return_to_go"], errors="coerce").to_numpy()
    diagnostics = {}
    for col in (
        "value_before_raw",
        "value_after_joint_raw",
        "root_value",
        "selected_q",
        "joint_root_value",
        "joint_selected_q",
    ):
        if col not in samples_df:
            continue
        values = pd.to_numeric(samples_df[col], errors="coerce").to_numpy()
        mask = np.isfinite(values) & np.isfinite(returns)
        if int(mask.sum()) < 2:
            continue
        entry = {
            "n": int(mask.sum()),
            "pearson_vs_return": _corr(values[mask], returns[mask]),
            "spearman_vs_return": _corr(_rankdata(values[mask]), _rankdata(returns[mask])),
            "mean": float(np.mean(values[mask])),
            "std": float(np.std(values[mask])),
        }
        if col.startswith("value_"):
            entry["mae_vs_return"] = float(np.mean(np.abs(values[mask] - returns[mask])))
        diagnostics[col] = entry
    return diagnostics


def _is_reveal_turn(turn):
    return (int(turn) - 3) % 5 == 0


class PartWriter:
    def __init__(self, out_dir, games_per_part):
        self.out_dir = Path(out_dir)
        self.games_per_part = int(games_per_part)
        self.part_id = 0
        self.parts = []
        self.records = []
        self.node_dyn = []
        self.glob = []
        self.det_id = []
        self.ego_pos = []
        self.legal_mask = []
        self.target_action = []
        self.target_policy = []

    def append(self, record, sample, target_policy):
        self.records.append(record)
        self.node_dyn.append(sample["node_dyn"].astype(np.float16))
        self.glob.append(sample["glob"].astype(np.float32))
        self.det_id.append(int(sample["det_id"]))
        self.ego_pos.append(int(sample["ego_pos"]))
        self.legal_mask.append(sample["legal_neighbors_mask"].astype(np.bool_))
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
            det_id=np.asarray(self.det_id, dtype=np.int64),
            ego_pos=np.asarray(self.ego_pos, dtype=np.int64),
            legal_neighbors_mask=np.asarray(self.legal_mask, dtype=np.bool_),
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
        self.det_id = []
        self.ego_pos = []
        self.legal_mask = []
        self.target_action = []
        self.target_policy = []


def _make_teacher(args, detective_checkpoint, mrx_policy, seed):
    config = BeliefMCTSConfig(
        simulations=args.simulations,
        rollout_turns=args.rollout_turns,
        exploration_c=args.exploration_c,
        policy_temperature=args.temperature,
        seed=seed,
        teacher_mode=args.teacher_mode,
        joint_top_k=args.joint_top_k,
        joint_random_actions=args.joint_random_actions,
        value_leaf_weight=args.value_leaf_weight,
    )
    return BeliefMCTSDetectiveTeacher(
        detective_checkpoint=detective_checkpoint,
        mrx_policy=mrx_policy,
        device=args.device,
        config=config,
    )


def run_logged_game(game_id, seed, args, writer, detective_checkpoint, mrx_plan, mrx_policy):
    seed_everything(seed)
    teacher = _make_teacher(args, detective_checkpoint, mrx_policy, seed)
    teacher.reset()

    game = Game()
    engine = DetectiveEngine(game.detectives_pos)
    rewards = []
    decisions = 0
    blocked = 0

    while True:
        if game.check_victory(silent=True):
            break

        captured = False
        phase_record_indices = []
        joint_info = None
        selected_joint = None
        if args.teacher_mode == "joint":
            selected_joint, joint_info = teacher.search_joint(game, engine)

        for detective_id in range(game.num_detectives):
            evaluated = teacher.default_policy.evaluate(
                game,
                engine.belief_state,
                detective_id,
            )
            sample = evaluated["sample"]
            if not sample["legal_neighbors_mask"].any():
                blocked += 1
                continue

            if args.teacher_mode == "joint":
                legal_destinations = _legal_destinations(game, detective_id)
                destination = (
                    selected_joint[detective_id]
                    if selected_joint is not None and detective_id < len(selected_joint)
                    else None
                )
                search_info = teacher.decision_info_from_joint(
                    joint_info,
                    detective_id,
                    legal_destinations,
                )
                selected = search_info.get("selected") or {}
                if destination is None or str(destination) not in set(legal_destinations):
                    destination = selected.get("destination")
                if destination is None or str(destination) not in set(legal_destinations):
                    blocked += 1
                    continue
                destination = str(destination)
                vehicle = _apply_detective_move(game, detective_id, destination)
            else:
                result, search_info = teacher.play_detective_turn(game, engine, detective_id)
                if result is None:
                    blocked += 1
                    continue
                destination, vehicle = result
                selected = search_info.get("selected") or {}

            target = int(destination) - 1
            if target < 0 or not sample["legal_neighbors_mask"][target]:
                raise AssertionError(
                    f"Detective MCTS target outside legal mask: game={game_id} "
                    f"turn={game.turn} det={detective_id} destination={destination}"
                )

            reward = R_STEP + R_DIST_COEF * float(
                _min_detective_distance(game.detectives_pos, game.mrx_pos)
            )
            done = False
            if game.check_victory(silent=True):
                reward += R_CAPTURE
                done = True
                captured = True

            target_policy = _target_policy_dense(search_info, target)
            visit_policy = [
                {"node": int(i), "prob": float(p)}
                for i, p in enumerate(target_policy)
                if float(p) > 0.0
            ]
            record = {
                "game_id": int(game_id),
                "decision_id": int(decisions),
                "turn": int(game.turn),
                "detective_id": int(detective_id),
                "teacher_mode": args.teacher_mode,
                "mrx_pos_hidden": str(game.mrx_pos),
                "detectives_pos": _dumps(game.detectives_pos),
                "belief_entropy": float(
                    -np.sum(
                        np.asarray(engine.belief_state)
                        * np.log(np.maximum(np.asarray(engine.belief_state), 1e-12))
                    )
                ),
                "target_destination": str(destination),
                "target_vehicle": vehicle,
                "target_action": int(target),
                "target_in_legal_mask": True,
                "visit_policy_json": _dumps(visit_policy),
                "visit_entropy": float(_policy_entropy(search_info)),
                "selected_visits": int(selected.get("visits", 0)),
                "root_actions": int(len(search_info.get("actions") or [])),
                "root_value": float(search_info.get("root_value", 0.0)),
                "selected_q": float(selected.get("q_value", 0.0)),
                "joint_root_value": float(search_info.get("joint_root_value", np.nan)),
                "joint_selected_q": float(search_info.get("joint_selected_q", np.nan)),
                "joint_action_count": int(search_info.get("joint_action_count", 0)),
                "value_before_norm": float(evaluated["value_norm"]),
                "value_before_raw": float(evaluated["value_raw"]),
                "reward": float(reward),
                "done": bool(done),
                "min_dist_after_detective": int(
                    _min_detective_distance(game.detectives_pos, game.mrx_pos)
                ),
                "mrx_opponent_id": mrx_plan["opponent_id"],
                "mrx_opponent_strategy": mrx_plan["strategy"],
            }
            writer.append(record, sample, target_policy)
            phase_record_indices.append(len(writer.records) - 1)
            rewards.append(float(reward))
            decisions += 1

            if captured:
                break
            engine.kalman_filter()

        game.detectives_moves.append(game.detectives_pos[:])
        if phase_record_indices:
            if captured:
                value_after = {"value_norm": np.nan, "value_raw": R_CAPTURE}
            else:
                value_after = teacher.default_policy.evaluate_value(
                    game,
                    engine.belief_state,
                    detective_id=0,
                )
            for record_idx in phase_record_indices:
                writer.records[record_idx]["value_after_joint_norm"] = float(
                    value_after["value_norm"]
                )
                writer.records[record_idx]["value_after_joint_raw"] = float(
                    value_after["value_raw"]
                )
        if captured:
            break

        ticket = mrx_policy.play_mrx_turn(game, engine.belief_state)
        terminal = game.check_victory(silent=True)
        if terminal and rewards:
            rewards[-1] += R_TIMEOUT if game.winner == 1 else R_CAPTURE
            break

        engine.update_belief_after_mrx_move(ticket)
        if _is_reveal_turn(game.turn):
            engine.mrx_is_spotted(game.mrx_pos)
        teacher.observe_mrx_move(game, ticket)

    returns = _discounted_returns(rewards, GAMMA)
    start = len(writer.records) - len(returns)
    for offset, ret in enumerate(returns):
        writer.records[start + offset]["return_to_go"] = float(ret)

    return {
        "game_id": int(game_id),
        "seed": int(seed),
        "winner": int(game.winner),
        "detective_win": int(game.winner == 0),
        "final_turn": int(game.turn),
        "logged_decisions": int(decisions),
        "blocked_decisions": int(blocked),
        "mrx_opponent_id": mrx_plan["opponent_id"],
    }


def run(args):
    ensure_project_root()
    seed_everything(args.seed_base)
    run_tag = args.run_tag or now_tag()
    mrx = best_model("mrx", root=args.root)
    detectives = best_model("detectives", root=args.root)
    mrx_checkpoint = args.mrx_checkpoint or mrx["path"]
    detective_checkpoint = args.detective_checkpoint or detectives["path"]
    args.mrx_opponent_id = args.mrx_opponent_id or mrx["id"]

    out_dir = Path(args.output_dir) / f"detective_mcts_logs_{run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "static_graph.npz",
        dense_adj=build_dense_adj().astype(np.float32),
        node_static=build_node_static_features().astype(np.float32),
        node_dyn_dim=np.array([14], dtype=np.int64),
        global_dim=np.array([28], dtype=np.int64),
    )

    plans, weights = _training_opponent_plans(args, mrx_checkpoint, mrx)
    opponent_mix = _opponent_mix_summary(plans, weights)
    opponent_rng = np.random.default_rng(args.seed_base + 909)
    policy_cache = {}
    writer = PartWriter(out_dir, games_per_part=args.games_per_part)
    stats = []
    started = time.time()

    print("Output:", out_dir)
    print("Detective teacher base:", detective_checkpoint)
    print("Primary Mr.X opponent:", args.mrx_opponent_id, mrx_checkpoint)
    print("Training opponent mix:", json.dumps(opponent_mix, indent=2))
    print("Belief MCTS simulations:", args.simulations)

    for game_id in range(args.games):
        idx = int(opponent_rng.choice(len(plans), p=weights))
        mrx_plan = plans[idx]
        mrx_policy = _policy_for_plan(mrx_plan, policy_cache, device=args.device)
        stats.append(
            run_logged_game(
                game_id=game_id,
                seed=args.seed_base + game_id,
                args=args,
                writer=writer,
                detective_checkpoint=detective_checkpoint,
                mrx_plan=mrx_plan,
                mrx_policy=mrx_policy,
            )
        )
        if (game_id + 1) % args.games_per_part == 0:
            writer.flush()
        if (game_id + 1) % args.log_every == 0:
            recent = stats[-args.log_every :]
            wr = np.mean([s["detective_win"] for s in recent])
            turn = np.mean([s["final_turn"] for s in recent])
            print(f"games={game_id + 1}/{args.games} recent_wr={wr*100:.1f}% turn={turn:.1f}")

    writer.flush()
    stats_df = pd.DataFrame(stats)
    stats_path = _write_dataframe(stats_df, out_dir / "game_stats.parquet")
    sample_parts = [_read_dataframe(p["samples"]) for p in writer.parts]
    samples_df = pd.concat(sample_parts, ignore_index=True) if sample_parts else pd.DataFrame()
    opponent_counts = Counter(s["mrx_opponent_id"] for s in stats)
    sanity = {
        "n_games": int(args.games),
        "n_samples": int(len(samples_df)),
        "detective_winrate": float(stats_df["detective_win"].mean()) if len(stats_df) else None,
        "avg_final_turn": float(stats_df["final_turn"].mean()) if len(stats_df) else None,
        "target_in_legal_mask_rate": (
            float(samples_df["target_in_legal_mask"].mean()) if len(samples_df) else None
        ),
        "opponent_counts": {str(k): int(v) for k, v in opponent_counts.items()},
    }
    value_diagnostics = _value_diagnostics(samples_df)
    manifest = {
        "schema_version": 1,
        "run_tag": run_tag,
        "generated_at": now_tag(),
        "teacher": "belief_mcts_detective",
        "detective_base_model_id": detectives["id"],
        "detective_checkpoint": os.fspath(detective_checkpoint),
        "mrx_base_model_id": mrx["id"],
        "mrx_checkpoint": os.fspath(mrx_checkpoint),
        "opponent_mix": opponent_mix,
        "config": vars(args),
        "parts": writer.parts,
        "game_stats": os.fspath(stats_path),
        "sanity": sanity,
        "value_diagnostics": value_diagnostics,
        "elapsed_seconds": time.time() - started,
    }
    write_json(out_dir / "manifest.json", manifest)
    print(json.dumps(sanity, indent=2))
    if value_diagnostics:
        print("Value diagnostics:", json.dumps(value_diagnostics, indent=2))
    print("Manifest:", out_dir / "manifest.json")
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description="Log Belief MCTS teacher data for detectives.")
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--rollout-turns", type=int, default=3)
    parser.add_argument("--exploration-c", type=float, default=1.35)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--teacher-mode",
        choices=("sequential", "joint"),
        default="joint",
        help="Use independent per-detective search or one synchronized joint-action search.",
    )
    parser.add_argument("--joint-top-k", type=int, default=3)
    parser.add_argument("--joint-random-actions", type=int, default=1)
    parser.add_argument(
        "--value-leaf-weight",
        type=float,
        default=0.0,
        help="Optional raw value bootstrap weight inside the teacher search; 0 logs diagnostics only.",
    )
    parser.add_argument("--mrx-checkpoint", default=None)
    parser.add_argument("--mrx-opponent-id", default=None)
    parser.add_argument("--detective-checkpoint", default=None)
    parser.add_argument("--opponent-pool", default="detective_training_mrx_pool_v1")
    parser.add_argument("--primary-mrx-weight", type=float, default=0.5)
    parser.add_argument("--pool-mrx-weight", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="/kaggle/working/detective_mcts_logs")
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--seed-base", type=int, default=20260522)
    parser.add_argument("--games-per-part", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=10)
    return parser


def main():
    args = build_parser().parse_args()
    if args.primary_mrx_weight < 0 or args.pool_mrx_weight < 0:
        raise SystemExit("--primary-mrx-weight and --pool-mrx-weight must be >= 0")
    run(args)


if __name__ == "__main__":
    main()
