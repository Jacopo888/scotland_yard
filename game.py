NUM_DETECTIVES=5
import pickle
import numpy as np
from random import choice
import random
from copy import deepcopy

SHORTEST_PATH_TENSOR=np.load("./Matrix_generation/distanze_scotland_yard_3d.npy")
STARTING_POS=['13', '26', '29', '34', '50', '53', '91', '94', '103', '112', '117', '132', '138', '141', '155', '174', '197', '198']
with open("./Matrix_generation/board_graph.pkl", "rb") as f:
    BOARD_GRAPH = pickle.load(f)

def _build_adjacency(graph):
    adj = {}
    for node in graph.nodes():
        adj[node] = {}
    for u, v, data in graph.edges(data=True):
        t = data["type"]
        adj[u].setdefault(v, set()).add(t)
        adj[v].setdefault(u, set()).add(t)
    return adj

ADJ = _build_adjacency(BOARD_GRAPH)


def random_key_value(dictionary):
    # Filter keys that have at least one associated value (not empty/None)
    valid_keys = [k for k, v in dictionary.items() if v]
 
    if not valid_keys:
        return None
    
    key = random.choice(valid_keys)
    value = random.choice(dictionary[key]) if isinstance(dictionary[key], (list, tuple)) else dictionary[key]
    
    return value

def ticket_bitmask(tickets):
    idx = 0
    if tickets.get("taxi", 0) > 0:
        idx |= 1
    if tickets.get("bus", 0) > 0:
        idx |= 2
    if tickets.get("underground", 0) > 0:
        idx |= 4
    return idx

class Game():
    def __init__(self):
        self.num_detectives=NUM_DETECTIVES
        self.starting_pos=['13', '26', '29', '34', '50', '53', '91', '94', '103', '112', '117', '132', '138', '141', '155', '174', '197', '198']
        self.winner=0

        self.mrx_pos=(choice(self.starting_pos))
        self.starting_pos.remove(self.mrx_pos)
        self.mrx_moves=[self.mrx_pos]

        self.detectives_pos=random.sample(self.starting_pos, NUM_DETECTIVES)
        self.detectives_moves=[self.detectives_pos[:]]

        self.turn=0
        self.mrx_tickets={
            "taxi":4,
            "bus":3,
            "underground":3,
            "water":5,
            "double":2

        }
        self.detective_tickets=[{
            "taxi":10,
            "bus":8,
            "underground":4
        }for i in range(self.num_detectives)]
        

    def x_turn(self, double_tickets=True ):

        available_moves=self.find_legal_moves_x(double_tickets)
        new_pos=str(random_key_value(available_moves))

        if new_pos=="double":
            self.mrx_moves.append("double")
            self.mrx_tickets["double"] -=1
            ticket1=self.x_turn(double_tickets=False)
            if self.check_victory(silent=True):
                return ticket1
            ticket2=self.x_turn(double_tickets=False)
            return [ticket1, ticket2]
        
        elif not any(new_pos in dests for dests in available_moves.values()):
                if self.mrx_tickets["taxi"]==0:
                    self.turn += 1
                    return str("blocked")
                else:
                    print("invalid move")
        else:
            self.mrx_pos=new_pos
            self.mrx_moves.append(self.mrx_pos)
            ticket = next((vehicle for vehicle, node in available_moves.items() if new_pos in node ), None)
            self.turn += 1
            self.mrx_tickets[ticket] -= 1
            return str(ticket)
    
    def x_automated_turn(self, proposed_mrx_pos, ticket):
        available_moves=self.find_legal_moves_x(double_tickets=False)
        if not any(proposed_mrx_pos in dests for dests in available_moves.values()):
            print("invalid move")
        else:
            self.mrx_pos=proposed_mrx_pos
            self.mrx_moves.append(self.mrx_pos)
            self.turn += 1
            self.mrx_tickets[ticket] -= 1
            return str(ticket)


    def find_legal_moves_x(self, double_tickets=True):
        moves = {}
        for neighbor, types in ADJ[self.mrx_pos].items():
            for t in types:
                if self.mrx_tickets.get(t, 0) > 0:
                    moves.setdefault(t, []).append(neighbor)
        if self.mrx_tickets["double"] and double_tickets:
            moves["double"]="double"
        return moves
    
    def check_victory(self, silent=False):
        if str(self.mrx_pos) in self.detectives_pos:
            if silent==False:
                #print("detectives WON!")
                pass
            self.winner=0
            return 1
        if self.turn>=22:
            if silent==False:
                #print("Mr. X WON!")
                pass
            self.winner=1
            return 1
        return 0

    def has_tickets(self, detective_id, node):
        pos = self.detectives_pos[detective_id]
        types = ADJ[pos].get(node, set())
        tickets = self.detective_tickets[detective_id]
        for t in types:
            if t != "water" and tickets[t] > 0:
                return True
        return False
    
    def use_tickets(self, detective_id, origin, destination):
        types = ADJ[origin].get(destination, set())
        for vehicle in ["taxi", "bus", "underground"]:
            if self.detective_tickets[detective_id][vehicle] > 0 and vehicle in types:
                self.detective_tickets[detective_id][vehicle] -= 1
                self.mrx_tickets[vehicle] +=1
                return vehicle
        
        return None

    def detective_automated_turn(self, belief_state, id):
        #find legal moves
        occupied_nodes = self.detectives_pos[:id] + self.detectives_pos[id+1:]
        reachable_nodes = [
            node for node in ADJ[self.detectives_pos[id]]
            if node not in occupied_nodes and self.has_tickets(id, node)
        ] 
        target=belief_state.argmax()
        try:
            matrix_idx = ticket_bitmask(self.detective_tickets[id])
            candidates = [(node, SHORTEST_PATH_TENSOR[matrix_idx][int(node)-1][target]) for node in reachable_nodes]
            node, distance = min(candidates, key=lambda x: x[1]) 
            
            #valid_nodes = [n for n in self.board.board.nodes() if n not in nodes_to_avoid]
            #limited_graph = self.board.board.subgraph(valid_nodes)
            #evaluated_node=shortest_path(limited_graph, self.detectives_pos[id], str(np.argmax(belief_state)+1))[1]

            #if evaluated_node != node:
                #print("evaluated_node != node")          
            self.use_tickets(id,self.detectives_pos[id], node)
            self.detectives_pos[id]=node
        except ValueError: 
            #print(f"detective {id} is blocked")
            #print(f"{self.detectives_pos}")
            pass
    
    def __deepcopy__(self, memo): #
        cls = self.__class__.__new__(self.__class__)
        cls.num_detectives = self.num_detectives
        cls.winner = self.winner
        cls.mrx_pos = self.mrx_pos
        cls.mrx_moves = self.mrx_moves[:]
        cls.detectives_pos = self.detectives_pos[:]
        cls.detectives_moves = [m[:] for m in self.detectives_moves]
        cls.turn = self.turn
        cls.starting_pos = self.starting_pos[:]
        cls.mrx_tickets = self.mrx_tickets.copy()
        cls.detective_tickets = [t.copy() for t in self.detective_tickets]
        return cls


    
