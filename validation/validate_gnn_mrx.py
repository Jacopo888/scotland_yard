import argparse
import contextlib
import json
import os
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np

import mrx_engine
from detective_engine import DetectiveEngine
from game import Game
from gnn_detective_engine import (
    GNNDetectiveEngine,
    find_latest_checkpoint as find_latest_detective_checkpoint,
)
from gnn_mrx_engine import (
    GNNMrXEngine,
    find_latest_checkpoint as find_latest_mrx_checkpoint,
)
from mrx_engine import MrxEngine
from utility import _min_detective_distance


DETECTIVE_STRATEGIES = ("heuristic", "gnn")
MRX_STRATEGIES = ("random", "mcts", "gnn")


def set_mrx_strength(explorations, simulations):
    if explorations is not None:
        mrx_engine.NUM_EXPLORATIONS = int(explorations)
    if simulations is not None:
        mrx_engine.NUM_SIMULATIONS = int(simulations)


def _play_detective_phase(game, engine, detective_strategy, detective_policy):
    game_over = False
    for detective_id in range(game.num_detectives):
        if detective_strategy == "heuristic":
            game.detective_automated_turn(engine.belief_state, detective_id)
        elif detective_strategy == "gnn":
            detective_policy.play_detective_turn(game, engine.belief_state, detective_id)
        else:
            raise ValueError(f"Unsupported detective strategy: {detective_strategy}")

        if game.check_victory(silent=True):
            game_over = True
            break
        engine.kalman_filter()

    game.detectives_moves.append(game.detectives_pos[:])
    return game_over


def _choose_mrx_action(game, engine, mrx_strategy, mcts_policy, gnn_mrx_policy):
    if mrx_strategy == "random":
        legal_before = game.find_legal_moves_x()
        ticket = game.x_random_turn()
        if ticket == "blocked":
            return game.mrx_pos, ticket, True
        return game.mrx_pos, ticket, game.mrx_pos in legal_before.get(ticket, [])

    if mrx_strategy == "mcts":
        destination, ticket = mcts_policy.search()
        legal_before = game.find_legal_moves_x()
        legal = ticket is None or destination in legal_before.get(ticket, [])
        game.x_automated_turn(destination, ticket)
        return destination, ticket, legal

    if mrx_strategy == "gnn":
        destination, ticket = gnn_mrx_policy.choose_action(game, engine.belief_state)
        legal_before = game.find_legal_moves_x()
        legal = ticket is None or destination in legal_before.get(ticket, [])
        game.x_automated_turn(destination, ticket)
        return destination, ticket, legal

    raise ValueError(f"Unsupported Mr.X strategy: {mrx_strategy}")


def play_game(
    detective_strategy,
    mrx_strategy,
    detective_policy=None,
    gnn_mrx_policy=None,
    mrx_explorations=None,
    mrx_simulations=None,
):
    set_mrx_strength(mrx_explorations, mrx_simulations)
    if detective_policy is not None:
        detective_policy.reset()
    if gnn_mrx_policy is not None:
        gnn_mrx_policy.reset()

    game = Game()
    engine = DetectiveEngine(game.detectives_pos)
    mcts_policy = MrxEngine(game, engine) if mrx_strategy == "mcts" else None

    tickets_used = []
    illegal_actions = 0
    min_dist_after_mrx = []
    gnn_values = []

    while True:
        if game.check_victory(silent=True):
            break

        if _play_detective_phase(game, engine, detective_strategy, detective_policy):
            break

        destination, ticket, legal = _choose_mrx_action(
            game, engine, mrx_strategy, mcts_policy, gnn_mrx_policy
        )
        tickets_used.append(ticket if ticket is not None else "blocked")
        if not legal:
            illegal_actions += 1
        if mrx_strategy == "gnn" and gnn_mrx_policy.last_value is not None:
            gnn_values.append(float(gnn_mrx_policy.last_value))
        min_dist_after_mrx.append(_min_detective_distance(game.detectives_pos, game.mrx_pos))

        if game.check_victory(silent=True):
            break

        engine.update_belief_after_mrx_move(ticket)
        if (game.turn - 3) % 5 == 0:
            engine.mrx_is_spotted(game.mrx_pos)

        if detective_policy is not None:
            detective_policy.observe_mrx_move(game, ticket)
        if gnn_mrx_policy is not None:
            gnn_mrx_policy.observe_mrx_move(game, ticket)

    return {
        "winner": int(game.winner),
        "mrx_win": int(game.winner == 1),
        "final_turn": int(game.turn),
        "mrx_pos": game.mrx_pos,
        "detectives_pos": list(game.detectives_pos),
        "tickets_used": tickets_used,
        "illegal_actions": int(illegal_actions),
        "avg_min_dist_after_mrx": (
            float(np.mean(min_dist_after_mrx)) if min_dist_after_mrx else None
        ),
        "avg_gnn_value": float(np.mean(gnn_values)) if gnn_values else None,
    }


def evaluate_suite(
    name,
    detective_strategy,
    mrx_strategy,
    n_games,
    detective_policy=None,
    gnn_mrx_policy=None,
    mrx_explorations=None,
    mrx_simulations=None,
    seed_offset=0,
    quiet=False,
):
    results = []

    def run_games():
        for i in range(n_games):
            seed = seed_offset + i
            random.seed(seed)
            np.random.seed(seed)
            results.append(
                play_game(
                    detective_strategy=detective_strategy,
                    mrx_strategy=mrx_strategy,
                    detective_policy=detective_policy,
                    gnn_mrx_policy=gnn_mrx_policy,
                    mrx_explorations=mrx_explorations,
                    mrx_simulations=mrx_simulations,
                )
            )

    if quiet:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stdout(devnull):
                run_games()
    else:
        run_games()

    ticket_counter = Counter()
    for result in results:
        ticket_counter.update(result["tickets_used"])
    mrx_wins = sum(r["mrx_win"] for r in results)
    illegal_actions = sum(r["illegal_actions"] for r in results)
    final_turns = [r["final_turn"] for r in results]
    distances = [
        r["avg_min_dist_after_mrx"]
        for r in results
        if r["avg_min_dist_after_mrx"] is not None
    ]
    values = [r["avg_gnn_value"] for r in results if r["avg_gnn_value"] is not None]

    return {
        "name": name,
        "n_games": int(n_games),
        "detectives": detective_strategy,
        "mrx": mrx_strategy,
        "mrx_explorations": None if mrx_explorations is None else int(mrx_explorations),
        "mrx_simulations": None if mrx_simulations is None else int(mrx_simulations),
        "mrx_wins": int(mrx_wins),
        "detective_wins": int(n_games - mrx_wins),
        "mrx_winrate": float(mrx_wins / n_games) if n_games else 0.0,
        "avg_final_turn": float(np.mean(final_turns)) if final_turns else None,
        "illegal_actions": int(illegal_actions),
        "ticket_counts": {str(k): int(v) for k, v in ticket_counter.items()},
        "avg_min_dist_after_mrx": float(np.mean(distances)) if distances else None,
        "avg_gnn_value": float(np.mean(values)) if values else None,
    }


def parse_suite(raw):
    parts = raw.split(":")
    if len(parts) not in (4, 6):
        raise argparse.ArgumentTypeError(
            "Suite must be name:detectives:mrx:games[:explorations:simulations]"
        )
    name, detectives, mrx, games = parts[:4]
    if detectives not in DETECTIVE_STRATEGIES:
        raise argparse.ArgumentTypeError(f"Unsupported detective strategy: {detectives}")
    if mrx not in MRX_STRATEGIES:
        raise argparse.ArgumentTypeError(f"Unsupported Mr.X strategy: {mrx}")
    explorations = simulations = None
    if len(parts) == 6:
        explorations, simulations = int(parts[4]), int(parts[5])
    elif mrx == "mcts":
        explorations, simulations = 15, 25
    return name, detectives, mrx, int(games), explorations, simulations


def main():
    parser = argparse.ArgumentParser(description="Validate Mr.X BC GNN checkpoint.")
    parser.add_argument("--mrx-checkpoint", default=None, help="Path to Mr.X BC checkpoint")
    parser.add_argument(
        "--detective-checkpoint", default=None, help="Path to detective GNN checkpoint"
    )
    parser.add_argument("--device", default=None, help="torch device, e.g. cpu or cuda")
    parser.add_argument("--suite", action="append", type=parse_suite, default=None)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    parser.add_argument("--quiet", action="store_true", help="Suppress rollout prints")
    args = parser.parse_args()

    suites = args.suite or [
        ("heuristic_vs_random", "heuristic", "random", 100, None, None),
        ("heuristic_vs_mrx_gnn_bc", "heuristic", "gnn", 100, None, None),
        ("gnn_detectives_vs_mrx_gnn_bc", "gnn", "gnn", 100, None, None),
        ("gnn_detectives_vs_mcts_fixed", "gnn", "mcts", 50, 15, 25),
    ]

    needs_detective_gnn = any(s[1] == "gnn" for s in suites)
    needs_mrx_gnn = any(s[2] == "gnn" for s in suites)

    detective_checkpoint = args.detective_checkpoint
    if needs_detective_gnn:
        detective_checkpoint = detective_checkpoint or find_latest_detective_checkpoint()
        if detective_checkpoint is None:
            raise FileNotFoundError("No detective GNN checkpoint found.")
        detective_policy = GNNDetectiveEngine(
            checkpoint_path=detective_checkpoint, device=args.device
        )
    else:
        detective_policy = None

    mrx_checkpoint = args.mrx_checkpoint
    if needs_mrx_gnn:
        mrx_checkpoint = mrx_checkpoint or find_latest_mrx_checkpoint()
        if mrx_checkpoint is None:
            raise FileNotFoundError("No Mr.X GNN checkpoint found.")
        gnn_mrx_policy = GNNMrXEngine(checkpoint_path=mrx_checkpoint, device=args.device)
    else:
        gnn_mrx_policy = None

    started = time.time()
    results = {
        "generated_at": datetime.now().isoformat(),
        "mrx_checkpoint": os.path.abspath(mrx_checkpoint) if mrx_checkpoint else None,
        "mrx_checkpoint_metadata": (
            gnn_mrx_policy.metadata if gnn_mrx_policy is not None else {}
        ),
        "detective_checkpoint": (
            os.path.abspath(detective_checkpoint) if detective_checkpoint else None
        ),
        "detective_checkpoint_metadata": (
            detective_policy.metadata if detective_policy is not None else {}
        ),
        "suites": {},
    }

    print(f"Mr.X checkpoint: {mrx_checkpoint or 'not used'}")
    if gnn_mrx_policy is not None and gnn_mrx_policy.metadata:
        print("Mr.X checkpoint metadata:")
        for key, value in gnn_mrx_policy.metadata.items():
            print(f"  {key}: {value}")
    print(f"Detective checkpoint: {detective_checkpoint or 'not used'}")

    for idx, suite in enumerate(suites):
        name, detectives, mrx, games, explorations, simulations = suite
        print(
            f"\nEvaluating {name}: games={games}, detectives={detectives}, mrx={mrx}"
        )
        if mrx == "mcts":
            print(f"  MCTS explorations={explorations}, simulations={simulations}")
        suite_result = evaluate_suite(
            name=name,
            detective_strategy=detectives,
            mrx_strategy=mrx,
            n_games=games,
            detective_policy=detective_policy,
            gnn_mrx_policy=gnn_mrx_policy,
            mrx_explorations=explorations,
            mrx_simulations=simulations,
            seed_offset=args.seed_offset + idx * 100000,
            quiet=args.quiet,
        )
        results["suites"][name] = suite_result
        print(
            f"  Mr.X WR={suite_result['mrx_winrate']*100:.1f}% "
            f"avg_turn={suite_result['avg_final_turn']:.1f} "
            f"illegal={suite_result['illegal_actions']}"
        )
        print(f"  tickets={suite_result['ticket_counts']}")

    results["elapsed_seconds"] = time.time() - started
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results: {args.output}")
    print(f"\nElapsed: {results['elapsed_seconds'] / 60:.1f} min")


if __name__ == "__main__":
    main()
