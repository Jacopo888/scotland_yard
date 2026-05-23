import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.distributions import Categorical

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mrx_engine
from detective_engine import DetectiveEngine
from game import Game
from gnn_detective_engine import GNNDetectiveEngine
from gnn_mrx_engine import GNNMrXEngine
from league.common import (
    best_model,
    ensure_project_root,
    load_pool,
    next_candidate,
    now_tag,
    seed_everything,
    write_candidate_update,
    write_json,
)
from model_registry import get_entry, resolve_path
from mrx_engine import MrxEngine
from utility import _min_detective_distance


GAMMA = 0.95
R_CAPTURE = 10.0
R_TIMEOUT = -10.0
R_STEP = -0.05
R_DIST_COEF = -0.10


class RandomMrXPolicy:
    def reset(self):
        return None

    def observe_mrx_move(self, game, ticket):
        return None

    def play_mrx_turn(self, game, belief_state):
        return game.x_random_turn()


class MCTSMrXPolicy:
    def __init__(self, explorations, simulations):
        self.explorations = int(explorations)
        self.simulations = int(simulations)

    def reset(self):
        return None

    def observe_mrx_move(self, game, ticket):
        return None

    def play_mrx_turn(self, game, belief_state):
        old_explorations = mrx_engine.NUM_EXPLORATIONS
        old_simulations = mrx_engine.NUM_SIMULATIONS
        try:
            mrx_engine.NUM_EXPLORATIONS = self.explorations
            mrx_engine.NUM_SIMULATIONS = self.simulations
            engine = DetectiveEngine(
                game.detectives_pos,
                belief_state.copy(),
                skip_filter=True,
            )
            destination, ticket = MrxEngine(game, engine).search()
            game.x_automated_turn(destination, ticket)
            return "blocked" if ticket is None else ticket
        finally:
            mrx_engine.NUM_EXPLORATIONS = old_explorations
            mrx_engine.NUM_SIMULATIONS = old_simulations


def _candidate_id_from_checkpoint(path):
    stem = Path(path).stem
    return stem[:-5] if stem.endswith("_last") else stem


def _registry_mrx_plan(opponent_id, root):
    entry = get_entry(opponent_id, root=root)
    kind = entry.get("kind")
    if opponent_id == "mrx_random":
        return {"opponent_id": opponent_id, "strategy": "random"}
    if kind == "mcts":
        config = entry.get("config", {})
        return {
            "opponent_id": opponent_id,
            "strategy": "mcts",
            "explorations": int(config["explorations"]),
            "simulations": int(config["simulations"]),
        }
    return {
        "opponent_id": opponent_id,
        "strategy": "gnn",
        "checkpoint": resolve_path(opponent_id, root=root),
    }


def _training_opponent_plans(args, mrx_checkpoint, mrx_opponent):
    weighted = []
    if args.primary_mrx_weight > 0:
        opponent_id = (
            args.mrx_opponent_id
            or (_candidate_id_from_checkpoint(mrx_checkpoint) if args.mrx_checkpoint else mrx_opponent["id"])
        )
        weighted.append(
            (
                {
                    "opponent_id": opponent_id,
                    "strategy": "gnn",
                    "checkpoint": mrx_checkpoint,
                    "role": "primary",
                },
                float(args.primary_mrx_weight),
            )
        )

    if args.opponent_pool and args.pool_mrx_weight > 0:
        pool = load_pool(args.opponent_pool, root=args.root)
        for row in pool.get("sample_weights", []):
            plan = _registry_mrx_plan(row["opponent_id"], root=args.root)
            plan["role"] = args.opponent_pool
            weighted.append((plan, float(row["weight"]) * float(args.pool_mrx_weight)))

    weighted = [(plan, weight) for plan, weight in weighted if weight > 0]
    if not weighted:
        weighted.append(
            (
                {
                    "opponent_id": mrx_opponent["id"],
                    "strategy": "gnn",
                    "checkpoint": mrx_checkpoint,
                    "role": "fallback",
                },
                1.0,
            )
        )
    total = sum(weight for _, weight in weighted)
    plans = [plan for plan, _ in weighted]
    weights = np.array([weight / total for _, weight in weighted], dtype=np.float64)
    return plans, weights


def _make_mrx_policy(plan, device):
    strategy = plan["strategy"]
    if strategy == "gnn":
        return GNNMrXEngine(checkpoint_path=plan["checkpoint"], device=device)
    if strategy == "random":
        return RandomMrXPolicy()
    if strategy == "mcts":
        return MCTSMrXPolicy(
            explorations=plan["explorations"],
            simulations=plan["simulations"],
        )
    raise ValueError(f"Unsupported Mr.X opponent strategy: {strategy}")


def _policy_for_plan(plan, cache, device):
    key = json.dumps(plan, sort_keys=True)
    if key not in cache:
        cache[key] = _make_mrx_policy(plan, device=device)
    return cache[key]


def _opponent_mix_summary(plans, weights):
    return {
        plan["opponent_id"]: {
            "weight": float(weight),
            "strategy": plan["strategy"],
            "role": plan.get("role"),
        }
        for plan, weight in zip(plans, weights)
    }


def _rollout_opponent_stats(episode_stats):
    counts = Counter(stat["opponent_id"] for stat in episode_stats)
    wins = Counter(
        stat["opponent_id"] for stat in episode_stats if int(stat.get("detective_win", 0))
    )
    return {
        opponent_id: {
            "games": int(count),
            "detective_winrate": float(wins[opponent_id] / max(count, 1)),
        }
        for opponent_id, count in sorted(counts.items())
    }


def _play_mrx_phase(game, engine, mrx_policy):
    ticket = mrx_policy.play_mrx_turn(game, engine.belief_state)
    if game.check_victory(silent=True):
        return ticket, True
    engine.update_belief_after_mrx_move(ticket)
    if (game.turn - 3) % 5 == 0:
        engine.mrx_is_spotted(game.mrx_pos)
    return ticket, False


def _sample_detective_action(policy, game, engine, detective_id, sample=True):
    sample_dict = policy.build_input(game, engine.belief_state, detective_id)
    batch = policy._collate_one(sample_dict)
    value, logits_list, cand_list = policy.model(batch, policy.dense_adj, policy.node_static)
    logits = logits_list[0]
    cand = cand_list[0]
    if not cand:
        return sample_dict, None, None, float(value[0].item()), None, None
    if sample:
        dist = Categorical(logits=logits)
        action_idx = dist.sample()
    else:
        dist = Categorical(logits=logits)
        action_idx = torch.argmax(logits)
    log_prob = dist.log_prob(action_idx)
    entropy = dist.entropy()
    dest_idx = int(cand[int(action_idx.item())])
    return (
        sample_dict,
        dest_idx,
        str(dest_idx + 1),
        float(value[0].item()),
        float(log_prob.item()),
        float(entropy.item()),
    )


def _apply_detective_move(game, detective_id, destination):
    origin = game.detectives_pos[detective_id]
    vehicle = game.use_ticket(detective_id, origin, destination)
    game.detectives_pos[detective_id] = destination
    return vehicle


def collect_episode(seed, detective_policy, mrx_policy, rtg_mean, rtg_std):
    seed_everything(seed)
    detective_policy.reset()
    mrx_policy.reset()
    game = Game()
    engine = DetectiveEngine(game.detectives_pos)
    transitions = []
    turn_start = 0
    tickets = []

    while True:
        if game.check_victory(silent=True):
            break

        turn_start = len(transitions)
        captured = False
        for detective_id in range(game.num_detectives):
            sample_dict, dest_idx, destination, value, old_log_prob, entropy = (
                _sample_detective_action(
                    detective_policy,
                    game,
                    engine,
                    detective_id,
                    sample=True,
                )
            )
            if destination is None:
                continue
            _apply_detective_move(game, detective_id, destination)
            transition = {
                "sample": sample_dict,
                "action_dest_idx": int(dest_idx),
                "old_log_prob": float(old_log_prob),
                "value": float(value),
                "entropy": float(entropy),
                "reward": 0.0,
                "done": False,
                "detective_id": int(detective_id),
            }
            transitions.append(transition)
            if game.check_victory(silent=True):
                transition["reward"] += R_CAPTURE
                transition["done"] = True
                captured = True
                break
            engine.kalman_filter()

        game.detectives_moves.append(game.detectives_pos[:])
        if captured:
            break

        ticket, terminal_after_mrx = _play_mrx_phase(game, engine, mrx_policy)
        tickets.append(ticket if ticket is not None else "blocked")
        if terminal_after_mrx:
            reward = R_TIMEOUT if game.winner == 1 else R_CAPTURE
            _assign_turn_reward(transitions, turn_start, reward)
            break

        min_dist = _min_detective_distance(game.detectives_pos, game.mrx_pos)
        reward = R_STEP + R_DIST_COEF * float(min_dist)
        _assign_turn_reward(transitions, turn_start, reward)
        detective_policy.observe_mrx_move(game, ticket)
        mrx_policy.observe_mrx_move(game, ticket)

    _add_returns(transitions, rtg_mean, rtg_std)
    return transitions, {
        "winner": int(game.winner),
        "detective_win": int(game.winner == 0),
        "final_turn": int(game.turn),
        "tickets": tickets,
    }


def _assign_turn_reward(transitions, turn_start, reward):
    active = transitions[turn_start:]
    if not active:
        return
    split = float(reward) / len(active)
    for transition in active:
        transition["reward"] += split


def _add_returns(transitions, rtg_mean, rtg_std):
    running = 0.0
    for transition in reversed(transitions):
        if transition["done"]:
            running = 0.0
        running = float(transition["reward"]) + GAMMA * running
        transition["return_raw"] = running
        transition["return_norm"] = (running - rtg_mean) / max(rtg_std, 1e-6)
        transition["advantage"] = transition["return_norm"] - transition["value"]


def _logprob_value(policy, transition):
    batch = policy._collate_one(transition["sample"])
    value, logits_list, cand_list = policy.model(batch, policy.dense_adj, policy.node_static)
    logits = logits_list[0]
    cand = cand_list[0]
    try:
        action_idx = cand.index(int(transition["action_dest_idx"]))
    except ValueError as exc:
        raise RuntimeError("Stored detective action is no longer legal for its sample.") from exc
    dist = Categorical(logits=logits)
    action_t = torch.tensor(action_idx, dtype=torch.long, device=logits.device)
    return dist.log_prob(action_t), dist.entropy(), value[0]


def ppo_update(policy, optimizer, transitions, args):
    indices = np.arange(len(transitions))
    metrics = defaultdict(float)
    steps = 0
    advantages = np.array([t["advantage"] for t in transitions], dtype=np.float32)
    adv_mean = float(advantages.mean())
    adv_std = float(advantages.std() + 1e-6)
    for transition in transitions:
        transition["advantage_norm"] = (transition["advantage"] - adv_mean) / adv_std

    for _ in range(args.ppo_epochs):
        np.random.shuffle(indices)
        for start in range(0, len(indices), args.minibatch_size):
            minibatch = [transitions[int(i)] for i in indices[start : start + args.minibatch_size]]
            log_probs = []
            entropies = []
            values = []
            old_log_probs = []
            returns = []
            advs = []
            for transition in minibatch:
                log_prob, entropy, value = _logprob_value(policy, transition)
                log_probs.append(log_prob)
                entropies.append(entropy)
                values.append(value)
                old_log_probs.append(transition["old_log_prob"])
                returns.append(transition["return_norm"])
                advs.append(transition["advantage_norm"])

            log_probs_t = torch.stack(log_probs)
            entropies_t = torch.stack(entropies)
            values_t = torch.stack(values)
            old_log_probs_t = torch.tensor(old_log_probs, dtype=torch.float32, device=values_t.device)
            returns_t = torch.tensor(returns, dtype=torch.float32, device=values_t.device)
            advs_t = torch.tensor(advs, dtype=torch.float32, device=values_t.device)

            ratio = torch.exp(log_probs_t - old_log_probs_t)
            surr1 = ratio * advs_t
            surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * advs_t
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = torch.nn.functional.mse_loss(values_t, returns_t)
            entropy = entropies_t.mean()
            loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.model.parameters(), args.grad_clip)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (old_log_probs_t - log_probs_t).mean().abs()
                clip_frac = ((ratio - 1.0).abs() > args.clip_eps).float().mean()
            metrics["loss"] += float(loss.item())
            metrics["policy_loss"] += float(policy_loss.item())
            metrics["value_loss"] += float(value_loss.item())
            metrics["entropy"] += float(entropy.item())
            metrics["approx_kl"] += float(approx_kl.item())
            metrics["clip_frac"] += float(clip_frac.item())
            steps += 1
    return {key: value / max(steps, 1) for key, value in metrics.items()}


@torch.no_grad()
def evaluate_argmax(policy, mrx_checkpoint, device, n_games, seed_offset):
    wins = 0
    turns = []
    ticket_counter = Counter()
    for game_idx in range(n_games):
        seed_everything(seed_offset + game_idx)
        policy.reset()
        mrx_policy = GNNMrXEngine(checkpoint_path=mrx_checkpoint, device=device)
        game = Game()
        engine = DetectiveEngine(game.detectives_pos)
        while True:
            if game.check_victory(silent=True):
                break
            captured = False
            for detective_id in range(game.num_detectives):
                _, _, destination, _, _, _ = _sample_detective_action(
                    policy,
                    game,
                    engine,
                    detective_id,
                    sample=False,
                )
                if destination is not None:
                    _apply_detective_move(game, detective_id, destination)
                if game.check_victory(silent=True):
                    captured = True
                    break
                engine.kalman_filter()
            game.detectives_moves.append(game.detectives_pos[:])
            if captured:
                break
            ticket, terminal = _play_mrx_phase(game, engine, mrx_policy)
            ticket_counter.update([ticket if ticket is not None else "blocked"])
            if terminal:
                break
            policy.observe_mrx_move(game, ticket)
            mrx_policy.observe_mrx_move(game, ticket)
        wins += int(game.winner == 0)
        turns.append(int(game.turn))
    return {
        "n_games": int(n_games),
        "detective_wins": int(wins),
        "detective_winrate": float(wins / max(n_games, 1)),
        "avg_final_turn": float(np.mean(turns)) if turns else None,
        "ticket_counts": {str(k): int(v) for k, v in ticket_counter.items()},
    }


def run(args):
    ensure_project_root()
    seed_everything(args.seed)
    run_tag = args.run_tag or now_tag()
    out_dir = Path(args.output_dir) / f"detective_rl_{run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    detective_parent = best_model("detectives", root=args.root)
    mrx_opponent = best_model("mrx", root=args.root)
    detective_checkpoint = args.detective_checkpoint or detective_parent["path"]
    mrx_checkpoint = args.mrx_checkpoint or mrx_opponent["path"]
    candidate_id = next_candidate(
        "detectives",
        "ppo",
        explicit_id=args.candidate_id,
        root=args.root,
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    policy = GNNDetectiveEngine(checkpoint_path=detective_checkpoint, device=device)
    ckpt = torch.load(detective_checkpoint, map_location=device, weights_only=False)
    rtg_mean = float(ckpt.get("rtg_mean", 0.0))
    rtg_std = float(ckpt.get("rtg_std", 1.0))
    optimizer = torch.optim.AdamW(policy.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opponent_plans, opponent_weights = _training_opponent_plans(
        args,
        mrx_checkpoint=mrx_checkpoint,
        mrx_opponent=mrx_opponent,
    )
    opponent_mix = _opponent_mix_summary(opponent_plans, opponent_weights)
    opponent_rng = np.random.default_rng(args.seed + 404)
    opponent_policy_cache = {}
    primary_mrx_id = (
        args.mrx_opponent_id
        or (_candidate_id_from_checkpoint(mrx_checkpoint) if args.mrx_checkpoint else mrx_opponent["id"])
    )

    print("Detective parent:", detective_parent["id"], detective_checkpoint)
    print("Primary Mr.X opponent:", primary_mrx_id, mrx_checkpoint)
    print("Training opponent mix:", json.dumps(opponent_mix, indent=2))
    print("Candidate:", candidate_id)

    baseline_eval = evaluate_argmax(
        policy,
        mrx_checkpoint=mrx_checkpoint,
        device=device,
        n_games=args.eval_games,
        seed_offset=args.seed + 500000,
    )
    best_score = baseline_eval["detective_winrate"]
    best_path = out_dir / f"{candidate_id}.pt"
    history = [{"update": 0, "eval": baseline_eval}]
    started = time.time()
    print("Initial eval:", json.dumps(baseline_eval, indent=2))

    for update in range(1, args.updates + 1):
        transitions = []
        episode_stats = []
        for game_idx in range(args.games_per_update):
            opponent_idx = int(
                opponent_rng.choice(len(opponent_plans), p=opponent_weights)
            )
            opponent_plan = opponent_plans[opponent_idx]
            mrx_policy = _policy_for_plan(
                opponent_plan,
                cache=opponent_policy_cache,
                device=device,
            )
            ts, stats = collect_episode(
                seed=args.seed + update * 10000 + game_idx,
                detective_policy=policy,
                mrx_policy=mrx_policy,
                rtg_mean=rtg_mean,
                rtg_std=rtg_std,
            )
            stats["opponent_id"] = opponent_plan["opponent_id"]
            stats["opponent_strategy"] = opponent_plan["strategy"]
            transitions.extend(ts)
            episode_stats.append(stats)
        if not transitions:
            continue
        train_metrics = ppo_update(policy, optimizer, transitions, args)
        rollout_opponents = _rollout_opponent_stats(episode_stats)
        row = {
            "update": int(update),
            "n_transitions": int(len(transitions)),
            "rollout_detective_winrate": float(np.mean([s["detective_win"] for s in episode_stats])),
            "rollout_avg_final_turn": float(np.mean([s["final_turn"] for s in episode_stats])),
            "rollout_opponents": rollout_opponents,
            **{f"train_{k}": v for k, v in train_metrics.items()},
        }
        if update % args.eval_every == 0 or update == args.updates:
            eval_result = evaluate_argmax(
                policy,
                mrx_checkpoint=mrx_checkpoint,
                device=device,
                n_games=args.eval_games,
                seed_offset=args.seed + 700000 + update * 1000,
            )
            row["eval"] = eval_result
            score = eval_result["detective_winrate"]
            if score > best_score:
                best_score = score
                torch.save(
                    _checkpoint_payload(
                        policy,
                        ckpt,
                        candidate_id,
                        detective_parent["id"],
                        detective_checkpoint,
                        mrx_opponent["id"],
                        mrx_checkpoint,
                        run_tag,
                        update,
                        eval_result,
                        history,
                        args,
                        opponent_mix,
                    ),
                    best_path,
                )
                print("saved best:", best_path)
            if args.stop_on_improvement and (
                score >= baseline_eval["detective_winrate"] + args.target_improvement_pp / 100.0
            ):
                history.append(row)
                print("Target improvement reached; stopping training.")
                break
        history.append(row)
        pd.DataFrame(_flatten_history(history)).to_csv(out_dir / "history.csv", index=False)
        print(
            f"update={update:03d} trans={len(transitions)} "
            f"rollout_wr={row['rollout_detective_winrate']*100:.1f}% "
            f"loss={train_metrics.get('loss', 0):.4f} "
            f"opponents={json.dumps(rollout_opponents, sort_keys=True)}"
        )

    final_eval = evaluate_argmax(
        policy,
        mrx_checkpoint=mrx_checkpoint,
        device=device,
        n_games=args.eval_games,
        seed_offset=args.seed + 900000,
    )
    last_path = out_dir / f"{candidate_id}_last.pt"
    torch.save(
        _checkpoint_payload(
            policy,
            ckpt,
            candidate_id,
            detective_parent["id"],
            detective_checkpoint,
            mrx_opponent["id"],
            mrx_checkpoint,
            run_tag,
            len(history),
            final_eval,
            history,
            args,
            opponent_mix,
        ),
        last_path,
    )
    metrics = {
        "baseline_detective_winrate": float(baseline_eval["detective_winrate"]),
        "best_detective_winrate": float(best_score),
        "final_detective_winrate": float(final_eval["detective_winrate"]),
        "elapsed_seconds": float(time.time() - started),
        "training_opponent_mix": opponent_mix,
    }
    update_path = write_candidate_update(
        out_dir=out_dir,
        candidate_id=candidate_id,
        side="detectives",
        kind="ppo",
        checkpoint_path=best_path if best_path.exists() else last_path,
        parent=detective_parent["id"],
        trained_against=list(opponent_mix.keys()),
        metrics=metrics,
    )
    manifest = {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "best_checkpoint": os.fspath(best_path) if best_path.exists() else None,
        "last_checkpoint": os.fspath(last_path),
        "registry_candidate_update": os.fspath(update_path),
        "training_opponent_mix": opponent_mix,
        "metrics": metrics,
    }
    write_json(out_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return manifest


def _flatten_history(history):
    rows = []
    for item in history:
        row = {}
        for key, value in item.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if not isinstance(sub_value, dict):
                        row[f"{key}_{sub_key}"] = sub_value
            else:
                row[key] = value
        rows.append(row)
    return rows


def _checkpoint_payload(
    policy,
    parent_ckpt,
    candidate_id,
    parent_id,
    parent_checkpoint,
    mrx_id,
    mrx_checkpoint,
    run_tag,
    update,
    eval_result,
    history,
    args,
    opponent_mix,
):
    return {
        "model_state_dict": policy.model.state_dict(),
        "config": parent_ckpt.get("config", {}),
        "rtg_mean": parent_ckpt.get("rtg_mean", 0.0),
        "rtg_std": parent_ckpt.get("rtg_std", 1.0),
        "update": int(update),
        "source_kind": "detective_ppo_league",
        "source_checkpoint": os.fspath(parent_checkpoint),
        "parent_model_id": parent_id,
        "mrx_opponent_id": mrx_id,
        "mrx_opponent_checkpoint": os.fspath(mrx_checkpoint),
        "candidate_id": candidate_id,
        "run_tag": run_tag,
        "eval": eval_result,
        "delta_pp": float(eval_result["detective_winrate"] * 100.0),
        "training_opponent_mix": opponent_mix,
        "history_tail": history[-10:],
        "training_args": vars(args),
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Train detective PPO against latest Mr.X GNN.")
    parser.add_argument("--detective-checkpoint", default=None)
    parser.add_argument("--mrx-checkpoint", default=None)
    parser.add_argument("--mrx-opponent-id", default=None)
    parser.add_argument("--opponent-pool", default="detective_training_mrx_pool_v1")
    parser.add_argument("--primary-mrx-weight", type=float, default=0.5)
    parser.add_argument("--pool-mrx-weight", type=float, default=0.5)
    parser.add_argument("--candidate-id", default=None)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="/kaggle/working/detective_rl_checkpoints")
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--updates", type=int, default=60)
    parser.add_argument("--games-per-update", type=int, default=32)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--clip-eps", type=float, default=0.15)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--target-improvement-pp", type=float, default=3.0)
    parser.add_argument("--stop-on-improvement", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.primary_mrx_weight < 0 or args.pool_mrx_weight < 0:
        parser.error("--primary-mrx-weight and --pool-mrx-weight must be >= 0")
    run(args)


if __name__ == "__main__":
    main()
