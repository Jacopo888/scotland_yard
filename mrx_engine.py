from copy import deepcopy
import math
from MCTS_Node import MCTSNode
from utility import play_mrx_turn, play_one_game
NUM_SIMULATIONS=75
EXPLORATIONS=25
#I wanto to have 10 explorations




class Mrx_Engine():
    def __init__(self, game, detective_engine):
        self.root=MCTSNode(game, detective_engine, None)
        self.nodes=[]


    def rollout(self, game_on_the_next_status, detective_engine_on_the_next_status):
        #I simulate the move that I want to test 
        #game_on_the_next_status , detective_engine_on_the_next_status = self.play_mrx_turn(starting_pos,self.game_status, self.detective_engine, ticket)
        #From this position, I simulate 100 games and evaluate the score
        iterations=0
        score=0
        while iterations<NUM_SIMULATIONS:
            iterations +=1
            score += play_one_game(game_on_the_next_status, detective_engine_on_the_next_status)
        #print(f"nodo {starting_pos} ha punteggio {score}")
        return score

    def MontecarloTreeSearch(self, current_status=None, iterations=EXPLORATIONS):
        
        if current_status is None:
            current_status = self.root
            self.i=1
            self.nodes += current_status.gen_child()
            nodes=self.nodes[:]
        else:
            nodes=current_status.child[:]

        while self.i<iterations:
            best_node=self.UCB1(nodes)
            if best_node.visits==0:
                score = self.rollout(deepcopy(best_node.game_status), deepcopy(best_node.detective_engine))
                best_node.update_visits(score)
                best_node.check_terminal()
                #print(f"nodo scelto:{best_node.game_status.mrx_pos}, {best_node.ticket}")
            
            elif best_node.visits == 1:
                #print(f"nodo scelto ha figli :{best_node.game_status.mrx_pos}, {best_node.ticket}")
                nodes += best_node.gen_child(is_root=False)
            
            elif best_node.visits > 1:
                #print(f"nodo scelto già esplorato:{best_node.game_status.mrx_pos}, {best_node.ticket}")
                self.MontecarloTreeSearch(current_status=best_node, iterations=self.i+1)
                self.i -= 1
            self.i +=1
        
        if current_status == self.root:
            nodes=[[node.score, node.game_status.mrx_pos, node.parent.game_status.mrx_pos] for node in nodes]
            #print(nodes)
            optimal_move=self.UCB1(self.root.child, final=True)
            #print(f"best move: {optimal_move.game_status.mrx_pos}")
            self.i=1
            self.nodes=[]
            self.root.child=[]
            return optimal_move.game_status.mrx_pos, optimal_move.ticket
        return
    
    def UCB1(self, nodes, final=False):
        best_score = float("-inf")
        best_node = None
        for node in nodes:
            if node.is_terminal and not final:
                continue
            if node.visits == 0:
                ucb1_score = float("inf")
            else:
                ucb1_score = node.score + 1.42*math.sqrt((math.log(self.i))/node.visits)
            
            if ucb1_score > best_score:
                best_score = ucb1_score
                best_node = node
        for node in nodes:
            if node.score==25:
                best_node=node
        if best_node==None:
            best_node=nodes[0]
        return best_node
        


