NUM_DETECTIVES=3
from board_generation import Board
from random import randint
from networkx import shortest_path, NetworkXNoPath
import numpy as np
from random import choice
import random

SHORTEST_PATH_MATRIX=np.load("./Matrix_generation/distanze_scotland_yard.npy")
STARTING_POS=['13', '26', '29', '34', '50', '53', '91', '94', '103', '112', '117', '132', '138', '141', '155', '174', '197', '198']

def random_key_value(dictionary):
    # Filtra le chiavi che hanno almeno un valore associato (non vuoto/None)
    valid_keys = [k for k, v in dictionary.items() if v]
 
    if not valid_keys:
        return None
    
    key = random.choice(valid_keys)
    value = random.choice(dictionary[key]) if isinstance(dictionary[key], (list, tuple)) else dictionary[key]
    
    return value


class Game():
    def __init__(self, game_board):
        self.starting_pos=['13', '26', '29', '34', '50', '53', '91', '94', '103', '112', '117', '132', '138', '141', '155', '174', '197', '198']
        self.board=game_board
        self.winner=0

        self.mrx_pos=(choice(self.starting_pos))
        self.starting_pos.remove(self.mrx_pos)
        self.mrx_moves=[self.mrx_pos]

        self.detectives_pos=random.sample(self.starting_pos, NUM_DETECTIVES)
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
        }for i in range(NUM_DETECTIVES)]
        

    def x_turn(self, duble_tickets=True ):

        avaiable_moves=self.find_legal_moves_x(duble_tickets)
        new_pos=str(random_key_value(avaiable_moves))

        if new_pos=="double":
            ticket1=self.x_turn(duble_tickets=False)
            if self.check_victory(silent=True):
                return ticket1
            ticket2=self.x_turn(duble_tickets=False)
            return [ticket1, ticket2]
        
        elif not any(new_pos in dests for dests in avaiable_moves.values()):
                print("mossa non valida")
                return self.x_turn()
        else:
            self.mrx_pos=new_pos
            self.mrx_moves.append(self.mrx_pos)
            ticket = next((k for k, v in avaiable_moves.items() if new_pos in v ), None)
            self.turn += 1
            return str(ticket)
    
    def find_legal_moves_x(self, duble_tickets=True):
        moves = {}
        
        for vehicle, tickets in self.mrx_tickets.items():
            if tickets:
                moves[vehicle] = [str(v) for _, v, data in self.board.board.edges(self.mrx_pos, data=True) if data["type"] == vehicle]
        if self.mrx_tickets["double"] and duble_tickets:
            moves["double"]="double"
        return moves
    
    def check_victory(self, silent=False):
        if str(self.mrx_pos) in self.detectives_pos:
            if silent==False:
                print("detectives WON!")
            self.winner=-1
            return 1
        if self.turn==22:
            if silent==False:
                print("Mr. X WON!")
            self.winner=1
            return 1
        return 0

    def has_tickets(self, detective_id, nodo):
        pos = self.detectives_pos[detective_id]
        return any(
            self.detective_tickets[detective_id][d["type"]]
            for d in self.board.board[pos][nodo].values() if d["type"] != "water"
        )
    
    def use_tickets(self, detective_id, origin, destination):
        for veichle in ["taxi", "bus", "underground"]:
            if self.detective_tickets[detective_id][veichle] > 0:
                edges = self.board.board[origin][destination]
                if any(d["type"] == veichle for d in edges.values()):
                    self.detective_tickets[detective_id][veichle] -= 1
                    return veichle
        return None

    def detective_automated_turn(self, belief_state, id):
        #find legal moves
        nodi_occupati = self.detectives_pos[:id] + self.detectives_pos[id+1:]
        nodi_accessibili = [
            node for node in self.board.board.neighbors(self.detectives_pos[id])
            if node not in nodi_occupati and self.has_tickets(id, node)
        ]        
        try:
            candidates = [(node, SHORTEST_PATH_MATRIX[int(node)-1][np.argmax(belief_state)]) for node in nodi_accessibili]
            node, distance = min(candidates, key=lambda x: x[1])            
            self.use_tickets(id,self.detectives_pos[id], node)
            self.detectives_pos[id]=node

        except ValueError: 
            #print(f"detective {id} is blocked")
            #print(f"{self.detectives_pos}")
            pass
                    




    
