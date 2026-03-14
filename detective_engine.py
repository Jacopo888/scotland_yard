import numpy as np
_TAXI_MATRIX = np.load("./Matrix_generation/taxi_matrix.npy").T
_BUS_MATRIX = np.load("./Matrix_generation/bus_matrix.npy").T
_UNDERGROUND_MATRIX = np.load("./Matrix_generation/underground_matrix.npy").T
_UNKNOWN_MATRIX = np.load("./Matrix_generation/unknown_matrix.npy").T
class Detective_Engine():
    def __init__(self, detectives_pos, belief_state=np.ones(199)*0.005):
        self.detectives_pos=detectives_pos
        self.belief_state=belief_state
        self.kalman_filter()
        self.taxi_matrix = _TAXI_MATRIX
        self.bus_matrix = _BUS_MATRIX
        self.underground_matrix = _UNDERGROUND_MATRIX
        self.unknown_matrix = _UNKNOWN_MATRIX
                

    def update_belief_after_mrx_move(self, tickets):
        if isinstance(tickets, str):
            tickets = [tickets]
        for ticket in tickets:
            if ticket=="taxi":
                self.belief_state = self.taxi_matrix @ self.belief_state
            elif ticket=="bus":
                self.belief_state = self.bus_matrix @ self.belief_state
            elif ticket=="underground":
                self.belief_state = self.underground_matrix @ self.belief_state
            elif ticket=="water":
                self.belief_state = self.unknown_matrix @ self.belief_state 
            elif ticket=="blocked":
                self.belief_state=self.belief_state

            if self.belief_state.sum() == 0:
                    print("belief state 0")
            self.kalman_filter()
        
        return 
    
    def mrx_is_spotted(self, mrx_pos):
        self.belief_state[:]=0
        self.belief_state[int(mrx_pos)-1]=1
        return
    
    def kalman_filter(self):
        if self.belief_state.sum() == 0:
            print(f"Somma zero: belief_state={self.belief_state}, detectives={self.detectives_pos}")
        prec_belief_state=self.belief_state[:]
        #print(prec_belief_state, self.belief_state)
        for pos in self.detectives_pos:
            self.belief_state[int(pos) - 1] = 0        
        s = self.belief_state.sum()
        if s != 0:
            self.belief_state /= s
        else:
            print(prec_belief_state)
            print(f"Somma zero: belief_state={self.belief_state}, detectives={self.detectives_pos}")
        return
