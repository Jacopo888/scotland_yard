from board_generation import Board
from game import Game
from detective_engine import Detective_Engine
#from belief_state_visualizer import BeliefStateVisualizer
from random import randint
from datetime import datetime
NUM_DETECTIVES=3


def play():
    game=Game(board)
    detective_engine=Detective_Engine(board, game)
    #belief_state=BeliefStateVisualizer(board)
    #board.update_detectives_pos(game.detectives_pos)
    #board.update_mrx_position(game.mrx_pos)
    #belief_state.show(detective_engine.belief_state)
    game_over=False
    while True:
        #print(f"turn{n}\nmrxpos={game.mrx_pos}\ndetectives_pos={game.detectives_pos}")
        if game.check_victory():
            break
        for id in range(NUM_DETECTIVES):
            game.detective_automated_turn(detective_engine.belief_state, id)
            if game.check_victory():
                game_over=True
                break
            detective_engine.kalman_filter()
        if game_over:
            break
        
        #belief_state.show(detective_engine.belief_state)
        #board.update_detectives_pos(game.detectives_pos)

        ticket=game.x_turn()
        if game.check_victory():
            break

        detective_engine.update_belief_after_mrx_move(ticket)

        
        #belief_state.show(detective_engine.belief_state)
        #board.update_mrx_position(game.mrx_pos)
        if game.turn%5==0:
            detective_engine.mrx_is_spotted(game.mrx_pos)
            #belief_state.show(detective_engine.belief_state)
            #print(f"{game.mrx_moves}")
    return game.winner
        

n=0
time1=datetime.now()
board=Board()
i=0
while i<100:
    print(f"partita {i}")
    n += play()
    i +=1
    
time2=datetime.now()
print(time2-time1)
print(n) 