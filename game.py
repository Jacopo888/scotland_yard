NUM_DETECTIVES=3
from board_generation import Board
from random import randint
class Game():
    def __init__(self, game_board):
        self.board=game_board
        self.mrx_pos=str(46)
        self.detectives_pos=[str(randint(1,200)) for detective in range(NUM_DETECTIVES)]
        self.turn=0
        self.mrx_tickets={
            "taxi":3,
            "bus":4,
            "underground":5
        }
        self.detective_tickets={f"detective_{i}":{
            "taxi":3,
            "bus":4,
            "underground":5
        } for i in range(NUM_DETECTIVES)}
        

    def x_turn(self):
        avaiable_moves=self.find_legal_moves_x()
        print(f"mr_x si trova in{self.mrx_pos}, dove si vuole andare?")
        print(f"ecco gli spostamenti disponibili: {avaiable_moves}\n")
        new_pos=input("inserisci il numero della casella dove vuoi andare: ")
        if not any(new_pos in dests for dests in avaiable_moves.values()):
                print("mossa non valida")
                return self.x_turn()
        else:
            self.mrx_pos=new_pos
            ticket = next((k for k, v in avaiable_moves.items() if new_pos in v ), None)
            self.turn += 1
            return str(ticket)
    
    def detectives_turn(self):
        for i in range(NUM_DETECTIVES):
            avaiable_moves = self.find_legal_moves_detective(i)
            current_detective_team_pos = self.detectives_pos[:i] + self.detectives_pos[i+1:]
            legal_moves = [v for _, v in avaiable_moves if v not in current_detective_team_pos]

            print(f"detective{i} is in {self.detectives_pos[i]}, dove vuole andare?")
            while True:
                print(f"ecco gli spostamenti disponibili: {legal_moves}\n")
                new_pos = input("inserisci il numero della casella dove vuoi andare: ")
                if new_pos in legal_moves:
                    self.detectives_pos[i] = new_pos
                    break
                else:
                    print("mossa non valida")

    def find_legal_moves_x(self):
        moves = {}
        for vehicle, tickets in self.mrx_tickets.items():
            if tickets:
                moves[vehicle] = [str(v) for _, v, data in self.board.board.edges(self.mrx_pos, data=True) if data["type"] == vehicle]
        return moves
    
    def find_legal_moves_detective(self, id): #(id = detective id)
        all_edges=[]

        for veichles in self.detective_tickets[f"detective_{id}"]:
            
            if self.detective_tickets[f"detective_{id}"][veichles]:
                edges=[(u,v) for u,v,data in self.board.board.edges(self.detectives_pos[id], data=True)
                if data["type"] == veichles]
                all_edges += edges
        return all_edges
    def check_victory(self):
        if self.mrx_pos in self.detectives_pos:
            print("detectives WON!")
            return 1
        if self.turn==22:
            print("Mr. X WON!")
            return 1

        

        
