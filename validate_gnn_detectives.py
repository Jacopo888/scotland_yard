import argparse
import contextlib
import json
import os
import random
import time
from datetime import datetime

import numpy as np

import mrx_engine
from detective_engine import DetectiveEngine
from game import Game
from gnn_detective_engine import GNNDetectiveEngine, find_latest_checkpoint
from mrx_engine import MrxEngine


def set_mrx_strength(explorations, simulations):
    mrx_engine.NUM_EXPLORATIONS = int(explorations)
    mrx_engine.NUM_SIMULATIONS = int(simulations)


def play_heuristic(mrx_explo=15, mrx_sims=25):
    set_mrx_strength(mrx_explo, mrx_sims)
    game = Game()
    engine = DetectiveEngine(game.detectives_pos)
    mrx = MrxEngine(game, engine)

    while True:
        if game.check_victory(silent=True):
            break

        game_over = False
        for detective_id in range(game.num_detectives):
            game.detective_automated_turn(engine.belief_state, detective_id)
            if game.check_victory(silent=True):
                game_over = True
                break
            engine.kalman_filter()
        game.detectives_moves.append(game.detectives_pos[:])
        if game_over:
            break

        best_pos, best_ticket = mrx.search()
        game.x_automated_turn(best_pos, best_ticket)
        if game.check_victory(silent=True):
            break

        engine.update_belief_after_mrx_move(best_ticket)
        if (game.turn - 3) % 5 == 0:
            engine.mrx_is_spotted(game.mrx_pos)

    return {
        "winner": int(game.winner),
        "final_turn": int(game.turn),
        "mrx_pos": game.mrx_pos,
        "detectives_pos": list(game.detectives_pos),
    }


def play_gnn(policy, mrx_explo=15, mrx_sims=25):
    set_mrx_strength(mrx_explo, mrx_sims)
    policy.reset()

    game = Game()
    engine = DetectiveEngine(game.detectives_pos)
    mrx = MrxEngine(game, engine)

    while True:
        if game.check_victory(silent=True):
            break

        game_over = False
        for detective_id in range(game.num_detectives):
            policy.play_detective_turn(game, engine.belief_state, detective_id)
            if game.check_victory(silent=True):
                game_over = True
                break
            engine.kalman_filter()
        game.detectives_moves.append(game.detectives_pos[:])
        if game_over:
            break

        best_pos, best_ticket = mrx.search()
        game.x_automated_turn(best_pos, best_ticket)
        if game.check_victory(silent=True):
            break

        engine.update_belief_after_mrx_move(best_ticket)
        if (game.turn - 3) % 5 == 0:
            engine.mrx_is_spotted(game.mrx_pos)
        policy.observe_mrx_move(game, best_ticket)

    return {
        "winner": int(game.winner),
        "final_turn": int(game.turn),
        "mrx_pos": game.mrx_pos,
        "detectives_pos": list(game.detectives_pos),
    }


def paired_eval(policy, n_games, mrx_explo, mrx_sims, seed_offset=0, quiet=False):
    gnn_wins = 0
    heuristic_wins = 0
    gnn_turns = []
    heuristic_turns = []

    def run_games():
        nonlocal gnn_wins, heuristic_wins
        for i in range(n_games):
            seed = seed_offset + i
            random.seed(seed)
            np.random.seed(seed)
            h = play_heuristic(mrx_explo=mrx_explo, mrx_sims=mrx_sims)

            random.seed(seed)
            np.random.seed(seed)
            g = play_gnn(policy, mrx_explo=mrx_explo, mrx_sims=mrx_sims)

            if h["winner"] == 0:
                heuristic_wins += 1
            if g["winner"] == 0:
                gnn_wins += 1
            heuristic_turns.append(h["final_turn"])
            gnn_turns.append(g["final_turn"])

    if quiet:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stdout(devnull):
                run_games()
    else:
        run_games()

    gnn_wr = gnn_wins / n_games if n_games else 0.0
    heuristic_wr = heuristic_wins / n_games if n_games else 0.0
    return {
        "n_games": int(n_games),
        "mrx_explorations": int(mrx_explo),
        "mrx_simulations": int(mrx_sims),
        "gnn_wins": int(gnn_wins),
        "heuristic_wins": int(heuristic_wins),
        "gnn_winrate": float(gnn_wr),
        "heuristic_winrate": float(heuristic_wr),
        "delta_pp": float((gnn_wr - heuristic_wr) * 100.0),
        "gnn_avg_final_turn": float(np.mean(gnn_turns)) if gnn_turns else None,
        "heuristic_avg_final_turn": float(np.mean(heuristic_turns)) if heuristic_turns else None,
    }


def parse_suite(raw):
    try:
        name, explorations, simulations, games = raw.split(":")
        return name, int(explorations), int(simulations), int(games)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Suite must be formatted as name:explorations:simulations:games"
        ) from exc


def main():
    parser = argparse.ArgumentParser(description="Validate final GNN detective checkpoint.")
    parser.add_argument("--checkpoint", default=None, help="Path to rgnn checkpoint .pt")
    parser.add_argument("--device", default=None, help="torch device, e.g. cpu or cuda")
    parser.add_argument("--suite", action="append", type=parse_suite, default=None)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    parser.add_argument("--quiet", action="store_true", help="Suppress noisy rollout prints")
    args = parser.parse_args()

    checkpoint = args.checkpoint or find_latest_checkpoint()
    if checkpoint is None:
        raise FileNotFoundError("No checkpoint found. Pass --checkpoint explicitly.")

    suites = args.suite or [
        ("fixed", 15, 25, 100),
        ("hard", 30, 50, 50),
        ("very_hard", 50, 80, 20),
    ]

    started = time.time()
    policy = GNNDetectiveEngine(checkpoint_path=checkpoint, device=args.device)
    print(f"Loaded checkpoint: {checkpoint}")
    if policy.metadata:
        print("Checkpoint metadata:")
        for key, value in policy.metadata.items():
            print(f"  {key}: {value}")

    results = {
        "checkpoint": os.path.abspath(checkpoint),
        "checkpoint_metadata": policy.metadata,
        "generated_at": datetime.now().isoformat(),
        "suites": {},
    }
    for idx, (name, explorations, simulations, games) in enumerate(suites):
        print(
            f"\nEvaluating {name}: games={games}, "
            f"Mr.X explorations={explorations}, simulations={simulations}"
        )
        suite_result = paired_eval(
            policy,
            n_games=games,
            mrx_explo=explorations,
            mrx_sims=simulations,
            seed_offset=args.seed_offset + idx * 100000,
            quiet=args.quiet,
        )
        results["suites"][name] = suite_result
        print(
            f"  GNN={suite_result['gnn_winrate']*100:.1f}% "
            f"heuristic={suite_result['heuristic_winrate']*100:.1f}% "
            f"delta={suite_result['delta_pp']:+.1f}pp"
        )

    results["elapsed_seconds"] = time.time() - started
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results: {args.output}")

    print(f"\nElapsed: {results['elapsed_seconds'] / 60:.1f} min")


if __name__ == "__main__":
    main()
