import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np

from gnn_detective_engine import GNNDetectiveEngine
from gnn_mrx_engine import GNNMrXEngine
from model_registry import get_best, get_entry, get_pool, resolve_path
from validation.validate_gnn_mrx import evaluate_suite


POOL_BY_SIDE = {
    "detectives": "detective_training_mrx_pool_v1",
    "mrx": "mrx_training_detective_pool_v1",
}


def _abs(path):
    return os.path.abspath(os.fspath(path))


def _resolve_registry_path(model_id, root):
    return resolve_path(model_id, root=root)


def _model_or_baseline(model_id, root):
    return get_entry(model_id, root=root)


def _games(base_games, games_scale, max_games_per_suite):
    games = max(1, int(round(int(base_games) * float(games_scale))))
    if max_games_per_suite is not None:
        games = min(games, int(max_games_per_suite))
    return games


def _next_output_path(candidate_id):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("Notebook") / "Promotion_Evals" / f"{candidate_id}_promotion_{timestamp}.json"


def _opponent_plan(row, root):
    opponent_id = row["opponent_id"]
    entry = _model_or_baseline(opponent_id, root=root)
    kind = entry.get("kind")
    side = entry.get("side")

    if opponent_id == "detective_heuristic":
        return {
            "opponent_id": opponent_id,
            "side": side,
            "strategy": "heuristic",
            "checkpoint": None,
        }
    if opponent_id == "mrx_random":
        return {
            "opponent_id": opponent_id,
            "side": side,
            "strategy": "random",
            "checkpoint": None,
        }
    if kind == "mcts":
        config = entry.get("config", {})
        return {
            "opponent_id": opponent_id,
            "side": side,
            "strategy": "mcts",
            "checkpoint": None,
            "explorations": int(config["explorations"]),
            "simulations": int(config["simulations"]),
        }

    checkpoint = _resolve_registry_path(opponent_id, root=root)
    return {
        "opponent_id": opponent_id,
        "side": side,
        "strategy": "gnn",
        "checkpoint": checkpoint,
    }


def _run_mrx_suite(
    suite_name,
    games,
    opponent,
    candidate_checkpoint,
    device,
    seed_offset,
    quiet,
):
    if opponent["strategy"] == "heuristic":
        detective_strategy = "heuristic"
        detective_policy = None
    else:
        detective_strategy = "gnn"
        detective_policy = GNNDetectiveEngine(
            checkpoint_path=opponent["checkpoint"], device=device
        )

    return evaluate_suite(
        name=suite_name,
        detective_strategy=detective_strategy,
        mrx_strategy="gnn",
        n_games=games,
        detective_policy=detective_policy,
        gnn_mrx_policy=GNNMrXEngine(checkpoint_path=candidate_checkpoint, device=device),
        seed_offset=seed_offset,
        quiet=quiet,
    )


def _run_detective_suite(
    suite_name,
    games,
    opponent,
    candidate_checkpoint,
    device,
    seed_offset,
    quiet,
):
    mrx_strategy = opponent["strategy"]
    mrx_policy = None
    explorations = None
    simulations = None
    if mrx_strategy == "gnn":
        mrx_policy = GNNMrXEngine(checkpoint_path=opponent["checkpoint"], device=device)
    elif mrx_strategy == "mcts":
        explorations = opponent["explorations"]
        simulations = opponent["simulations"]

    return evaluate_suite(
        name=suite_name,
        detective_strategy="gnn",
        mrx_strategy=mrx_strategy,
        n_games=games,
        detective_policy=GNNDetectiveEngine(checkpoint_path=candidate_checkpoint, device=device),
        gnn_mrx_policy=mrx_policy,
        mrx_explorations=explorations,
        mrx_simulations=simulations,
        seed_offset=seed_offset,
        quiet=quiet,
    )


def _score(side, result):
    if side == "mrx":
        return float(result["mrx_winrate"])
    if side == "detectives":
        return float(result["detective_wins"] / max(result["n_games"], 1))
    raise ValueError(f"Unsupported side: {side}")


def _check_gate(side, pool, suite_rows):
    gate = pool.get("promotion_gate", {})
    min_improvement = float(gate.get("minimum_improvement_pp", 0.0))
    max_regression = float(gate.get("max_allowed_regression_pp_any_core_suite", 100.0))
    core_suites = set(gate.get("core_suites", []))
    hard = gate.get("hard_requirements", {})

    candidate_scores = [row["candidate_score"] for row in suite_rows]
    baseline_scores = [row["baseline_score"] for row in suite_rows]
    candidate_mean = float(np.mean(candidate_scores)) if candidate_scores else 0.0
    baseline_mean = float(np.mean(baseline_scores)) if baseline_scores else 0.0
    improvement_pp = (candidate_mean - baseline_mean) * 100.0

    failed = []
    if improvement_pp < min_improvement:
        failed.append(
            f"mean improvement {improvement_pp:.2f}pp < required {min_improvement:.2f}pp"
        )

    core_regressions = {}
    for row in suite_rows:
        if row["suite"] not in core_suites:
            continue
        regression_pp = row["delta_pp"]
        core_regressions[row["suite"]] = regression_pp
        if regression_pp < -max_regression:
            failed.append(
                f"{row['suite']} regression {regression_pp:.2f}pp exceeds "
                f"{max_regression:.2f}pp limit"
            )

    candidate_illegal = sum(
        int(row["candidate_result"].get("illegal_actions", 0)) for row in suite_rows
    )
    illegal_required = hard.get("illegal_actions", None)
    illegal_enforced = side == "mrx"
    if illegal_required is not None and illegal_enforced and candidate_illegal != int(illegal_required):
        failed.append(
            f"candidate illegal actions {candidate_illegal} != required {illegal_required}"
        )

    if side == "mrx" and "mrx_winrate_vs_heuristic_min" in hard:
        row = next((r for r in suite_rows if r["opponent_id"] == "detective_heuristic"), None)
        if row is None:
            failed.append("missing detective_heuristic suite for hard requirement")
        elif row["candidate_score"] < float(hard["mrx_winrate_vs_heuristic_min"]):
            failed.append(
                f"Mr.X winrate vs heuristic {row['candidate_score']:.3f} < "
                f"{float(hard['mrx_winrate_vs_heuristic_min']):.3f}"
            )

    if side == "detectives" and "detective_winrate_vs_random_min" in hard:
        row = next((r for r in suite_rows if r["opponent_id"] == "mrx_random"), None)
        if row is None:
            failed.append("missing mrx_random suite for hard requirement")
        elif row["candidate_score"] < float(hard["detective_winrate_vs_random_min"]):
            failed.append(
                f"detective winrate vs random {row['candidate_score']:.3f} < "
                f"{float(hard['detective_winrate_vs_random_min']):.3f}"
            )

    return {
        "passed": not failed,
        "failed_reasons": failed,
        "candidate_mean_score": candidate_mean,
        "baseline_mean_score": baseline_mean,
        "improvement_pp": improvement_pp,
        "core_regressions_pp": core_regressions,
        "candidate_illegal_actions_reported": candidate_illegal,
        "candidate_illegal_actions_enforced": illegal_enforced,
    }


def build_plan(args):
    side = args.side
    root = args.root
    pool_name = args.pool or POOL_BY_SIDE[side]
    pool = get_pool(pool_name, root=root)
    gate = pool.get("promotion_gate", {})

    candidate_id = args.candidate_id or Path(args.candidate_checkpoint).stem
    baseline_id = args.baseline_id or gate.get("must_improve_over")
    if baseline_id is None:
        baseline_id = get_best(side, root=root)["id"]
    baseline_checkpoint = args.baseline_checkpoint or _resolve_registry_path(baseline_id, root=root)

    matrix = []
    for row in pool.get("eval_matrix", []):
        planned = {
            "suite": row["suite"],
            "opponent_id": row["opponent_id"],
            "base_games": int(row["games"]),
            "games": _games(row["games"], args.games_scale, args.max_games_per_suite),
            "opponent": _opponent_plan(row, root=root),
        }
        matrix.append(planned)

    return {
        "side": side,
        "pool_name": pool_name,
        "pool": pool,
        "candidate_id": candidate_id,
        "candidate_checkpoint": args.candidate_checkpoint,
        "baseline_id": baseline_id,
        "baseline_checkpoint": baseline_checkpoint,
        "matrix": matrix,
    }


def run_promotion_validation(args):
    plan = build_plan(args)
    started = time.time()
    suite_rows = []

    if args.dry_run:
        return {
            "schema_version": 1,
            "generated_at": datetime.now().isoformat(),
            "dry_run": True,
            "plan": plan,
        }

    print(f"Promotion validation side: {plan['side']}")
    print(f"Candidate: {plan['candidate_id']} -> {plan['candidate_checkpoint']}")
    print(f"Baseline:  {plan['baseline_id']} -> {plan['baseline_checkpoint']}")
    print(f"Pool:      {plan['pool_name']}")

    for idx, suite in enumerate(plan["matrix"]):
        seed = args.seed_offset + idx * 100000
        opponent = suite["opponent"]
        print(
            f"\nSuite {suite['suite']}: games={suite['games']}, "
            f"opponent={suite['opponent_id']}"
        )

        if plan["side"] == "mrx":
            runner = _run_mrx_suite
        else:
            runner = _run_detective_suite

        candidate_result = runner(
            suite_name=f"{suite['suite']}_candidate",
            games=suite["games"],
            opponent=opponent,
            candidate_checkpoint=plan["candidate_checkpoint"],
            device=args.device,
            seed_offset=seed,
            quiet=args.quiet,
        )
        baseline_result = runner(
            suite_name=f"{suite['suite']}_baseline",
            games=suite["games"],
            opponent=opponent,
            candidate_checkpoint=plan["baseline_checkpoint"],
            device=args.device,
            seed_offset=seed,
            quiet=args.quiet,
        )

        candidate_score = _score(plan["side"], candidate_result)
        baseline_score = _score(plan["side"], baseline_result)
        delta_pp = (candidate_score - baseline_score) * 100.0
        print(
            f"  candidate={candidate_score*100:.1f}% "
            f"baseline={baseline_score*100:.1f}% delta={delta_pp:+.1f}pp"
        )

        suite_rows.append(
            {
                "suite": suite["suite"],
                "opponent_id": suite["opponent_id"],
                "games": suite["games"],
                "base_games": suite["base_games"],
                "metric": "mrx_winrate" if plan["side"] == "mrx" else "detective_winrate",
                "candidate_score": candidate_score,
                "baseline_score": baseline_score,
                "delta_pp": delta_pp,
                "opponent": opponent,
                "candidate_result": candidate_result,
                "baseline_result": baseline_result,
            }
        )

    summary = _check_gate(plan["side"], plan["pool"], suite_rows)
    return {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(),
        "dry_run": False,
        "side": plan["side"],
        "candidate_id": plan["candidate_id"],
        "candidate_checkpoint": _abs(plan["candidate_checkpoint"]),
        "baseline_id": plan["baseline_id"],
        "baseline_checkpoint": _abs(plan["baseline_checkpoint"]),
        "pool_name": plan["pool_name"],
        "promotion_gate": plan["pool"].get("promotion_gate", {}),
        "suites": {row["suite"]: row for row in suite_rows},
        "summary": summary,
        "elapsed_seconds": time.time() - started,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run registry-based large validation and promotion gate checks."
    )
    parser.add_argument("--side", choices=("detectives", "mrx"), required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--candidate-id", default=None)
    parser.add_argument("--baseline-id", default=None)
    parser.add_argument("--baseline-checkpoint", default=None)
    parser.add_argument("--pool", default=None)
    parser.add_argument("--root", default=".")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--games-scale", type=float, default=1.0)
    parser.add_argument("--max-games-per-suite", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()
    if args.games_scale <= 0:
        parser.error("--games-scale must be > 0")

    result = run_promotion_validation(args)
    output = Path(args.output) if args.output else _next_output_path(result["plan"]["candidate_id"] if args.dry_run else result["candidate_id"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved promotion validation: {output}")
    if not args.dry_run:
        summary = result["summary"]
        print(
            f"Gate passed: {summary['passed']} | "
            f"improvement={summary['improvement_pp']:+.2f}pp"
        )
        if summary["failed_reasons"]:
            print("Failed reasons:")
            for reason in summary["failed_reasons"]:
                print(f"  - {reason}")
        if args.fail_on_gate and not summary["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
