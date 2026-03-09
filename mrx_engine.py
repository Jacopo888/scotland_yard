from copy import deepcopy
from detective_engine import Detective_Engine
NUM_SIMULATIONS=25
class Mrx_Engine():
    def __init__(self, game, detective_engine):
        self.game_status=game
        self.detective_engine=detective_engine

    def simulations(self, starting_pos, ticket):
        #I do the move that I want to test 
        game_on_the_next_status=deepcopy(self.game_status)
        detective_engine_on_the_next_status=Detective_Engine(game_on_the_next_status.detectives_pos, belief_state=deepcopy(self.detective_engine.belief_state))
        
        game_on_the_next_status.mrx_pos=starting_pos
        game_on_the_next_status.turn += 1
        game_on_the_next_status.mrx_moves.append(game_on_the_next_status.mrx_pos)
        if game_on_the_next_status.check_victory(silent=True):
            return game_on_the_next_status.winner*100
        
        detective_engine_on_the_next_status.update_belief_after_mrx_move(ticket)
        
        if game_on_the_next_status.turn%5==0:
            detective_engine_on_the_next_status.mrx_is_spotted(game_on_the_next_status.mrx_pos)        
        
        #From this position, I simulate 100 games and evaluate the score
        iterations=0
        score=0
        while iterations<NUM_SIMULATIONS:
            iterations +=1
            score += self.play_one_game(game_on_the_next_status, detective_engine_on_the_next_status)
        #print(f"nodo {starting_pos} ha punteggio {score}")
        return score

    def MontecarloTreeSearch(self):
        avaiable_moves=self.game_status.find_legal_moves_x(duble_tickets=False)
        leafs=[]
        for veichle,nodes in avaiable_moves.items():
            for node in nodes:
                score=self.simulations(node,veichle)
                leafs.append((score, node, veichle))
        best_score, best_node, best_veichle = max(leafs, key=lambda x: x[0])
        #print(f"scelgo nodo {best_node}")
        return best_node, best_veichle


    def play_one_game(self, game_status, detective_engine):
        #create a copy of the game status after the move that we want to test
        game=deepcopy(game_status)
        detective_engine=Detective_Engine(game.detectives_pos, deepcopy(detective_engine.belief_state) )
        #simulate the game till endgame. -1 if detectives win, 1 if mrx win
        game_over=False
        while True:
            if game.check_victory(silent=True):
                break
            for id in range(game.num_detectives):
                game.detective_automated_turn(detective_engine.belief_state, id)
                if game.check_victory(silent=True):
                    game_over=True
                    break
                detective_engine.kalman_filter()
            if game_over:
                break
            
            ticket=game.x_turn()
            if game.check_victory(silent=True):
                break

            detective_engine.update_belief_after_mrx_move(ticket)

            if game.turn%5==0:
                detective_engine.mrx_is_spotted(game.mrx_pos)

        return game.winner
        
        
        


