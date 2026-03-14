from utility import play_mrx_turn, play_detectives_turn
from copy import deepcopy

class MCTSNode():
    def __init__(self, game_status, detective_engine, ticket):
        self.game_status=game_status
        self.detective_engine=detective_engine
        self.parent=None
        self.child=[]
        self.score=0
        self.visits=0
        self.is_terminal=0
        self.ticket=ticket

    def gen_child(self, is_root=True):
        if is_root:
            current_game_status=self.game_status
            current_detective_engine=self.detective_engine            
        else:
            current_game_status, current_detective_engine = play_detectives_turn((self.game_status), (self.detective_engine))
        
        available_moves=current_game_status.find_legal_moves_x(double_tickets=False)
        
        for vehicle,nodes in available_moves.items():
            for node in nodes:
                game_status, detective_engine_status = play_mrx_turn(node, (current_game_status), (current_detective_engine), vehicle)
                child_node=MCTSNode(game_status, detective_engine_status, vehicle)
                child_node.parent = self
                self.child.append(child_node)
        return self.child[:]
    
    def update_visits(self, score):
        self.visits += 1
        self.score += score
        if self.parent is not None:
            self.parent.update_visits(score)
    
    def check_terminal(self):
        game, _ =play_detectives_turn((self.game_status), (self.detective_engine))
        end=game.check_victory(silent=True)
        if end:
            self.is_terminal=1
        return end