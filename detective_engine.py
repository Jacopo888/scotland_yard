import numpy as np

_TAXI_MATRIX = np.load("./Matrix_generation/taxi_matrix.npy").T
_BUS_MATRIX = np.load("./Matrix_generation/bus_matrix.npy").T
_UNDERGROUND_MATRIX = np.load("./Matrix_generation/underground_matrix.npy").T
_UNKNOWN_MATRIX = np.load("./Matrix_generation/unknown_matrix.npy").T

TICKET_TO_MATRIX = {
    "taxi": _TAXI_MATRIX,
    "bus": _BUS_MATRIX,
    "underground": _UNDERGROUND_MATRIX,
    "water": _UNKNOWN_MATRIX,
}


class DetectiveEngine:
    def __init__(self, detectives_pos, belief_state=None, skip_filter=False):
        self.detectives_pos = detectives_pos
        self.belief_state = (
            belief_state if belief_state is not None
            else np.ones(199) * 0.005
        )
        if not skip_filter:
            self.kalman_filter()

    def copy(self, new_detectives_pos=None):
        return DetectiveEngine(
            new_detectives_pos if new_detectives_pos is not None else self.detectives_pos,
            self.belief_state.copy(),
            skip_filter=True,
        )

    def update_belief_after_mrx_move(self, ticket):
            matrix = TICKET_TO_MATRIX.get(ticket)
            if matrix is not None:
                self.belief_state = matrix @ self.belief_state
            self.kalman_filter()
            if sum(self.belief_state)==0:
                print("somma nulla")
            if ticket == "blocked":
                print("bloccato")

    def mrx_is_spotted(self, mrx_pos):
        self.belief_state[:] = 0
        self.belief_state[int(mrx_pos) - 1] = 1

    def kalman_filter(self):
        for pos in self.detectives_pos:
            self.belief_state[int(pos) - 1] = 0
        total = self.belief_state.sum()
        if total > 0:
            self.belief_state /= total
