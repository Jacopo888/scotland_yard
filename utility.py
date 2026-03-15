from copy import deepcopy

from game import SHORTEST_PATH_TENSOR


def play_mrx_turn(starting_pos, game_status, detective_engine, ticket):
    game = deepcopy(game_status)
    engine = detective_engine.copy(game.detectives_pos)

    game.mrx_pos = starting_pos
    game.turn += 1
    game.mrx_moves.append(game.mrx_pos)

    if game.check_victory(silent=True):
        return game, engine

    engine.update_belief_after_mrx_move(ticket)

    if game.turn % 5 == 0:
        engine.mrx_is_spotted(game.mrx_pos)

    return game, engine


def play_detectives_turn(game_status, detective_engine):
    game = deepcopy(game_status)
    engine = detective_engine.copy(game.detectives_pos)

    if game.check_victory(silent=True):
        return game, engine

    for i in range(game.num_detectives):
        game.detective_automated_turn(engine.belief_state, i)
        if game.check_victory(silent=True):
            return game, engine
        engine.kalman_filter()

    return game, engine


def _simulate_turns(game_status, detective_engine, max_turns=None):
    game = deepcopy(game_status)
    engine = detective_engine.copy(game.detectives_pos)
    turns_played = 0

    while max_turns is None or turns_played < max_turns:
        if game.check_victory(silent=True):
            break

        game_over = False
        for i in range(game.num_detectives):
            game.detective_automated_turn(engine.belief_state, i)
            if game.check_victory(silent=True):
                game_over = True
                break
            engine.kalman_filter()
        if game_over:
            break

        ticket = game.x_random_turn()
        if game.check_victory(silent=True):
            break

        engine.update_belief_after_mrx_move(ticket)
        if game.turn % 5 == 0:
            engine.mrx_is_spotted(game.mrx_pos)

        turns_played += 1

    return game


def play_one_game(game_status, detective_engine):
    game = _simulate_turns(game_status, detective_engine)
    return game.winner


def play_five_turns(game_status, detective_engine):
    game = _simulate_turns(game_status, detective_engine, max_turns=5)
    return _min_detective_distance(game.detectives_pos, game.mrx_pos)


def _min_detective_distance(detectives_pos, mrx_pos):
    mrx_idx = int(mrx_pos) - 1
    return min(
        int(SHORTEST_PATH_TENSOR[7][int(pos) - 1][mrx_idx])
        for pos in detectives_pos
    )
