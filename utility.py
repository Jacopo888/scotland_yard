from copy import deepcopy
from detective_engine import Detective_Engine


def play_mrx_turn(starting_pos, game_status, detective_engine, ticket):
        game_on_the_next_status=deepcopy(game_status)
        detective_engine_on_the_next_status=Detective_Engine(game_on_the_next_status.detectives_pos, belief_state=deepcopy(detective_engine.belief_state))
        
        game_on_the_next_status.mrx_pos=starting_pos
        game_on_the_next_status.turn += 1
        game_on_the_next_status.mrx_moves.append(game_on_the_next_status.mrx_pos)
        if game_on_the_next_status.check_victory(silent=True):
            return game_on_the_next_status, detective_engine_on_the_next_status
        
        detective_engine_on_the_next_status.update_belief_after_mrx_move(ticket)
        
        if game_on_the_next_status.turn%5==0:
            detective_engine_on_the_next_status.mrx_is_spotted(game_on_the_next_status.mrx_pos)    

        return game_on_the_next_status, detective_engine_on_the_next_status


def play_detectives_turn(game, detective_engine):
    game_on_the_next_status=deepcopy(game)
    detective_engine=Detective_Engine(game_on_the_next_status.detectives_pos, belief_state=deepcopy(detective_engine.belief_state))
    if game_on_the_next_status.check_victory(silent=True):
                return game_on_the_next_status, detective_engine
    for id in range(game_on_the_next_status.num_detectives):
        game_on_the_next_status.detective_automated_turn(detective_engine.belief_state, id)
        if game_on_the_next_status.check_victory(silent=True):
            return game_on_the_next_status, detective_engine
        detective_engine.kalman_filter()
    return game_on_the_next_status, detective_engine

     
     

def play_one_game( game_status, detective_engine):
        #create a copy of the game status after the move that we want to test
        game=deepcopy(game_status)
        curr_detective_engine=Detective_Engine(game.detectives_pos, deepcopy(detective_engine.belief_state) )
        #simulate the game till endgame. -1 if detectives win, 1 if mrx win
        game_over=False
        while True:
            if game.check_victory(silent=True):
                break
            for id in range(game.num_detectives):
                game.detective_automated_turn(curr_detective_engine.belief_state, id)
                if game.check_victory(silent=True):
                    game_over=True
                    break
                curr_detective_engine.kalman_filter()
            if game_over:
                break
            
            ticket=game.x_turn()
            
            if game.check_victory(silent=True):
                break

            curr_detective_engine.update_belief_after_mrx_move(ticket)
            
            if game.turn%5==0:
                curr_detective_engine.mrx_is_spotted(game.mrx_pos)

        return game.winner

