from datetime import datetime
from time import sleep

from game import Game
from detective_engine import DetectiveEngine
from mrx_engine import MrxEngine


def play_visual():
    from board_generation import Board
    from belief_state_visualizer import BeliefStateVisualizer

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

        if game.turn % 5 == 0:
            engine.mrx_is_spotted(game.mrx_pos)
            visualizer.show(engine.belief_state)

        sleep(1.5)

    return game.winner


def play():
    game = Game()
    engine = DetectiveEngine(game.detectives_pos)
    mrx = MrxEngine(game, engine)

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

        best_pos, best_ticket = mrx.search()
        game.x_automated_turn(best_pos, best_ticket)

        if game.check_victory():
            break

        engine.update_belief_after_mrx_move(best_ticket)

        if game.turn % 5 == 0:
            engine.mrx_is_spotted(game.mrx_pos)

    return game.winner


if __name__ == "__main__":
    NUM_GAMES = 10

    start = datetime.now()
    mrx_wins = sum(play() for _ in range(NUM_GAMES))
    elapsed = datetime.now() - start

    print(f"Time: {elapsed}")
    print(f"Mr. X wins: {mrx_wins}/{NUM_GAMES}")
