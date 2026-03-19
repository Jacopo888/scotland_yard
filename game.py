import pickle
import random

import numpy as np

NUM_DETECTIVES = 5

STARTING_POSITIONS = [
    '13', '26', '29', '34', '50', '53', '91', '94',
    '103', '112', '117', '132', '138', '141', '155', '174', '197', '198',
]

SHORTEST_PATH_TENSOR = np.load("./Matrix_generation/distanze_scotland_yard_3d.npy")

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


def ticket_bitmask(tickets):
    idx = 0
    if tickets.get("taxi", 0) > 0:
        idx |= 1
    if tickets.get("bus", 0) > 0:
        idx |= 2
    if tickets.get("underground", 0) > 0:
        idx |= 4
    return idx


class Game:
    def __init__(self):
        self.num_detectives = NUM_DETECTIVES
        self.winner = 0

        available = STARTING_POSITIONS[:]
        self.mrx_pos = random.choice(available)
        available.remove(self.mrx_pos)
        self.mrx_moves = [self.mrx_pos]

        self.detectives_pos = random.sample(available, NUM_DETECTIVES)
        self.detectives_moves = [self.detectives_pos[:]]

        self.turn = 0
        self.mrx_tickets = {
            "taxi": 4,
            "bus": 3,
            "underground": 3,
            "water": 5,
            "double": 2,
        }
        self.detective_tickets = [
            {"taxi": 10, "bus": 8, "underground": 4}
            for _ in range(self.num_detectives)
        ]

    def x_random_turn(self, allow_double=True):
        available_moves = self.find_legal_moves_x(allow_double)
        chosen = _random_move(available_moves)

        if chosen == "double":
            self.mrx_moves.append("double")
            self.mrx_tickets["double"] -= 1
            ticket1 = self.x_random_turn(allow_double=False)
            if self.check_victory(silent=True):
                return ticket1
            ticket2 = self.x_random_turn(allow_double=False)
            return [ticket1, ticket2]

        if chosen is None:
            self.turn += 1
            return "blocked"

        position, ticket = chosen
        self.mrx_pos = position
        self.mrx_moves.append(self.mrx_pos)
        self.turn += 1
        self.mrx_tickets[ticket] -= 1
        return ticket

    def x_automated_turn(self, proposed_pos, ticket):
        self.mrx_pos = proposed_pos
        self.mrx_moves.append(self.mrx_pos)
        self.turn += 1
        self.mrx_tickets[ticket] -= 1
        return ticket

    def find_legal_moves_x(self, allow_double=True):
        moves = {}
        for neighbor, types in ADJ[self.mrx_pos].items():
            for t in types:
                if self.mrx_tickets.get(t, 0) > 0:
                    moves.setdefault(t, []).append(neighbor)
        if allow_double and self.mrx_tickets["double"] > 0:
            moves["double"] = "double"
        return moves

    def check_victory(self, silent=False):
        if self.mrx_pos in self.detectives_pos:
            if not silent:
                print("Detectives won!")
            self.winner = 0
            return True
        if self.turn >= 22:
            if not silent:
                print("Mr. X won!")
            self.winner = 1
            return True
        return False

    def has_tickets(self, detective_id, node):
        pos = self.detectives_pos[detective_id]
        types = ADJ[pos].get(node, set())
        tickets = self.detective_tickets[detective_id]
        return any(t != "water" and tickets.get(t, 0) > 0 for t in types)

    def use_ticket(self, detective_id, origin, destination):
        types = ADJ[origin].get(destination, set())
        for vehicle in ("taxi", "bus", "underground"):
            if self.detective_tickets[detective_id][vehicle] > 0 and vehicle in types:
                self.detective_tickets[detective_id][vehicle] -= 1
                self.mrx_tickets[vehicle] += 1
                return vehicle
        return None

    def detective_automated_turn(self, belief_state, detective_id):
        occupied = (
            self.detectives_pos[:detective_id]
            + self.detectives_pos[detective_id + 1:]
        )
        reachable = [
            node for node in ADJ[self.detectives_pos[detective_id]]
            if node not in occupied and self.has_tickets(detective_id, node)
        ]
        if not reachable:
            return

        target = belief_state.argmax()
        matrix_idx = ticket_bitmask(self.detective_tickets[detective_id])
        candidates = [
            (node, SHORTEST_PATH_TENSOR[matrix_idx][int(node) - 1][target])
            for node in reachable
        ]
        best_node, _ = min(candidates, key=lambda x: x[1])
        self.use_ticket(detective_id, self.detectives_pos[detective_id], best_node)
        self.detectives_pos[detective_id] = best_node

    def __deepcopy__(self, memo):
        clone = self.__class__.__new__(self.__class__)
        clone.num_detectives = self.num_detectives
        clone.winner = self.winner
        clone.mrx_pos = self.mrx_pos
        clone.mrx_moves = self.mrx_moves[:]
        clone.detectives_pos = self.detectives_pos[:]
        clone.detectives_moves = [m[:] for m in self.detectives_moves]
        clone.turn = self.turn
        clone.mrx_tickets = self.mrx_tickets.copy()
        clone.detective_tickets = [t.copy() for t in self.detective_tickets]
        return clone
    def copy(self):
        clone = Game.__new__(Game)
        clone.num_detectives = self.num_detectives
        clone.winner = self.winner
        clone.mrx_pos = self.mrx_pos
        clone.mrx_moves = self.mrx_moves[:]
        clone.detectives_pos = self.detectives_pos[:]
        clone.detectives_moves = [m[:] for m in self.detectives_moves]
        clone.turn = self.turn
        clone.mrx_tickets = self.mrx_tickets.copy()
        clone.detective_tickets = [t.copy() for t in self.detective_tickets]
        return clone
    
    def snapshot(self):
        return (self.winner, self.mrx_pos, self.mrx_moves[:],
                self.detectives_pos[:], [m[:] for m in self.detectives_moves],
                self.turn, self.mrx_tickets.copy(),
                [t.copy() for t in self.detective_tickets])

    def restore(self, snap):
        (self.winner, self.mrx_pos, self.mrx_moves, self.detectives_pos,
        self.detectives_moves, self.turn, self.mrx_tickets,
        self.detective_tickets) = (
            snap[0], snap[1], snap[2][:], snap[3][:],
            [m[:] for m in snap[4]], snap[5], snap[6].copy(),
            [t.copy() for t in snap[7]])


def _random_move(available_moves):
    valid_vehicles = [v for v, nodes in available_moves.items() if nodes]
    if not valid_vehicles:
        return None
    vehicle = random.choice(valid_vehicles)
    if vehicle == "double":
        return "double"
    destination = random.choice(available_moves[vehicle])
    return destination, vehicle
