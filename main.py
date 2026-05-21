import argparse
from datetime import datetime
from time import sleep

import mrx_engine
from detective_engine import DetectiveEngine
from game import Game
from mrx_engine import MrxEngine


DETECTIVE_STRATEGIES = ("heuristic", "gnn")
MRX_STRATEGIES = ("mcts", "random", "gnn", "neural_mcts")


def configure_mrx_strength(explorations=None, simulations=None):
    if explorations is not None:
        mrx_engine.NUM_EXPLORATIONS = int(explorations)
    if simulations is not None:
        mrx_engine.NUM_SIMULATIONS = int(simulations)


def _load_gnn_policy(checkpoint=None, device=None):
    from gnn_detective_engine import GNNDetectiveEngine

    return GNNDetectiveEngine(checkpoint_path=checkpoint, device=device)


def _load_gnn_mrx_policy(checkpoint=None, device=None):
    from gnn_mrx_engine import GNNMrXEngine

    return GNNMrXEngine(checkpoint_path=checkpoint, device=device)


def _load_neural_mcts_policy(
    mrx_checkpoint=None,
    detective_checkpoint=None,
    device=None,
    simulations=None,
    c_puct=None,
    temperature=None,
):
    from neural_mrx_mcts_engine import NeuralMrXMCTSEngine

    kwargs = {}
    if simulations is not None:
        kwargs["num_simulations"] = simulations
    if c_puct is not None:
        kwargs["c_puct"] = c_puct
    if temperature is not None:
        kwargs["temperature"] = temperature
    return NeuralMrXMCTSEngine(
        mrx_checkpoint_path=mrx_checkpoint,
        detective_checkpoint_path=detective_checkpoint,
        device=device,
        **kwargs,
    )


def _play_detective_phase(game, belief_engine, detective_strategy, gnn_policy=None):
    game_over = False
    for detective_id in range(game.num_detectives):
        if detective_strategy == "heuristic":
            game.detective_automated_turn(belief_engine.belief_state, detective_id)
        elif detective_strategy == "gnn":
            gnn_policy.play_detective_turn(
                game, belief_engine.belief_state, detective_id
            )
        else:
            raise ValueError(f"Unsupported detective strategy: {detective_strategy}")

        if game.check_victory(silent=True):
            game_over = True
            break
        belief_engine.kalman_filter()

    game.detectives_moves.append(game.detectives_pos[:])
    return game_over


def _play_mrx_phase(game, belief_engine, mrx_strategy, mrx_policy=None):
    if mrx_strategy == "mcts":
        best_pos, best_ticket = mrx_policy.search()
        game.x_automated_turn(best_pos, best_ticket)
        return best_ticket
    if mrx_strategy == "random":
        return game.x_random_turn()
    if mrx_strategy == "gnn":
        return mrx_policy.play_mrx_turn(game, belief_engine.belief_state)
    if mrx_strategy == "neural_mcts":
        return mrx_policy.play_mrx_turn(game, belief_engine.belief_state)
    raise ValueError(f"Unsupported Mr.X strategy: {mrx_strategy}")


def play(
    detective_strategy="heuristic",
    mrx_strategy="mcts",
    checkpoint=None,
    mrx_checkpoint=None,
    device=None,
    mrx_explorations=None,
    mrx_simulations=None,
    neural_mcts_simulations=None,
    neural_mcts_c_puct=None,
    neural_mcts_temperature=None,
    silent=True,
):
    """Play one game.

    detective_strategy:
        "heuristic" keeps the original belief/shortest-path detective policy.
        "gnn" uses GNNDetectiveEngine with the selected checkpoint.
    mrx_strategy:
        "mcts" keeps the current Mr.X search engine.
        "random" is a lightweight baseline/debug opponent.
        "gnn" uses GNNMrXEngine with the selected checkpoint.
        "neural_mcts" uses Mr.X GNN priors/value with frozen GNN detective rollouts.
    """
    configure_mrx_strength(mrx_explorations, mrx_simulations)

    game = Game()
    belief_engine = DetectiveEngine(game.detectives_pos)
    if mrx_strategy == "mcts":
        mrx_policy = MrxEngine(game, belief_engine)
    elif mrx_strategy == "gnn":
        mrx_policy = _load_gnn_mrx_policy(checkpoint=mrx_checkpoint, device=device)
    elif mrx_strategy == "neural_mcts":
        mrx_policy = _load_neural_mcts_policy(
            mrx_checkpoint=mrx_checkpoint,
            detective_checkpoint=checkpoint,
            device=device,
            simulations=neural_mcts_simulations,
            c_puct=neural_mcts_c_puct,
            temperature=neural_mcts_temperature,
        )
    else:
        mrx_policy = None
    gnn_policy = (
        _load_gnn_policy(checkpoint=checkpoint, device=device)
        if detective_strategy == "gnn"
        else None
    )

    while True:
        if game.check_victory(silent=silent):
            break

        if _play_detective_phase(game, belief_engine, detective_strategy, gnn_policy):
            break

        best_ticket = _play_mrx_phase(game, belief_engine, mrx_strategy, mrx_policy)
        if game.check_victory(silent=silent):
            break

        belief_engine.update_belief_after_mrx_move(best_ticket)
        if (game.turn - 3) % 5 == 0:
            belief_engine.mrx_is_spotted(game.mrx_pos)

        if gnn_policy is not None:
            gnn_policy.observe_mrx_move(game, best_ticket)
        if mrx_strategy in ("gnn", "neural_mcts"):
            mrx_policy.observe_mrx_move(game, best_ticket)

    return game.winner


def play_visual():
    from belief_state_visualizer import BeliefStateVisualizer
    from board_generation import Board

    board = Board()
    game = Game()
    engine = DetectiveEngine(game.detectives_pos)
    mrx = MrxEngine(game, engine)
    visualizer = BeliefStateVisualizer(board)

    board.update_detectives_pos(game.detectives_pos)
    board.update_mrx_position(game.mrx_pos)
    visualizer.show(engine.belief_state)

    while True:
        if game.check_victory():
            break

        game_over = False
        for i in range(game.num_detectives):
            game.detective_automated_turn(engine.belief_state, i)
            if game.check_victory():
                game_over = True
                break
            engine.kalman_filter()
        game.detectives_moves.append(game.detectives_pos[:])
        if game_over:
            break

        visualizer.show(engine.belief_state)
        board.update_detectives_pos(game.detectives_pos)

        best_pos, best_ticket = mrx.search()
        game.x_automated_turn(best_pos, best_ticket)

        if game.check_victory():
            break

        engine.update_belief_after_mrx_move(best_ticket)
        visualizer.show(engine.belief_state)
        board.update_mrx_position(game.mrx_pos)

        if (game.turn - 3) % 5 == 0:
            engine.mrx_is_spotted(game.mrx_pos)
            visualizer.show(engine.belief_state)

        sleep(1.5)

    return game.winner


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run Scotland Yard engine matchups.")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--detectives", choices=DETECTIVE_STRATEGIES, default="heuristic")
    parser.add_argument("--mrx", choices=MRX_STRATEGIES, default="mcts")
    parser.add_argument("--checkpoint", default=None, help="GNN detective checkpoint path")
    parser.add_argument("--mrx-checkpoint", default=None, help="GNN Mr.X checkpoint path")
    parser.add_argument("--device", default=None, help="Torch device for GNN engines")
    parser.add_argument("--mrx-explorations", type=int, default=None)
    parser.add_argument("--mrx-simulations", type=int, default=None)
    parser.add_argument("--neural-mcts-simulations", type=int, default=None)
    parser.add_argument("--neural-mcts-c-puct", type=float, default=None)
    parser.add_argument("--neural-mcts-temperature", type=float, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    start = datetime.now()
    mrx_wins = sum(
        play(
            detective_strategy=args.detectives,
            mrx_strategy=args.mrx,
            checkpoint=args.checkpoint,
            mrx_checkpoint=args.mrx_checkpoint,
            device=args.device,
            mrx_explorations=args.mrx_explorations,
            mrx_simulations=args.mrx_simulations,
            neural_mcts_simulations=args.neural_mcts_simulations,
            neural_mcts_c_puct=args.neural_mcts_c_puct,
            neural_mcts_temperature=args.neural_mcts_temperature,
            silent=not args.verbose,
        )
        for _ in range(args.games)
    )
    elapsed = datetime.now() - start

    print(f"Detectives: {args.detectives}")
    print(f"Mr.X: {args.mrx}")
    if args.detectives == "gnn" or args.mrx == "neural_mcts":
        print(f"Detective checkpoint: {args.checkpoint or 'auto'}")
    if args.mrx in ("gnn", "neural_mcts"):
        print(f"Mr.X checkpoint: {args.mrx_checkpoint or 'auto'}")
    if args.mrx == "neural_mcts":
        print(f"Neural MCTS simulations: {args.neural_mcts_simulations or 'default'}")
    print(f"Time: {elapsed}")
    print(f"Mr. X wins: {mrx_wins}/{args.games}")
    print(f"Detectives wins: {args.games - mrx_wins}/{args.games}")


if __name__ == "__main__":
    main()
