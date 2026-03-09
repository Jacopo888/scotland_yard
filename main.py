from board_generation import Board
from game import Game
from detective_engine import Detective_Engine
from belief_state_visualizer import BeliefStateVisualizer
from datetime import datetime
from mrx_engine import Mrx_Engine


def play():
    game=Game()
    detective_engine=Detective_Engine(game.detectives_pos)
    mrx_engine=Mrx_Engine(game,detective_engine)
    #belief_state=BeliefStateVisualizer(board)
    #board.update_detectives_pos(game.detectives_pos)
    #board.update_mrx_position(game.mrx_pos)
    #belief_state.show(detective_engine.belief_state)
    game_over=False
    while True:
        #print(f"turn{n}\nmrxpos={game.mrx_pos}\ndetectives_pos={game.detectives_pos}")
        if game.check_victory():
            break
        for id in range(game.num_detectives):
            game.detective_automated_turn(detective_engine.belief_state, id)
            if game.check_victory():
                game_over=True
                break
            detective_engine.kalman_filter()
        game.detectives_moves.append(game.detectives_pos[:])
        if game_over:
            break
        
        #belief_state.show(detective_engine.belief_state)
        #board.update_detectives_pos(game.detectives_pos)
        best_node, best_veichle= mrx_engine.MontecarloTreeSearch()
        ticket=game.x_automated_turn(best_node, best_veichle)

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
#board=Board()
i=0
while i<10 :
    #print(f"partita {i}")
    n += play()
    i +=1

time2=datetime.now()
print(time2-time1)
print(n) 