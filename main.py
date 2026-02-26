from board_generation import Board
from game import Game
from detective_engine import Detective_Engine
from belief_state_visualizer import BeliefStateVisualizer
board=Board()
game=Game(board)
detective_engine=Detective_Engine(board, game.detectives_pos)
belief_state=BeliefStateVisualizer(board)


board.update_detectives_pos(game.detectives_pos)
board.update_mrx_position(game.mrx_pos)

belief_state.show(detective_engine.belief_state)


while True:
    game.detectives_turn()
    board.update_detectives_pos(game.detectives_pos)
    if game.check_victory():
        break
    ticket=game.x_turn()
    detective_engine.update_belief_after_mrx_move(ticket)
    belief_state.show(detective_engine.belief_state)
    board.update_mrx_position(game.mrx_pos)
    if game.check_victory():
        break
    if game.turn%2==0:
        detective_engine.mrx_is_spotted(game.mrx_pos)
        belief_state.show(detective_engine.belief_state)
