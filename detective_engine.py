import numpy as np
import networkx as nx


def stochastic_matrix(A):
    A = np.array(A, dtype=float)
    row_sums = A.sum(axis=1)
    inv_row_sums = np.divide(1, row_sums, out=np.zeros_like(row_sums), where=row_sums != 0)
    T = A * inv_row_sums[:, np.newaxis]
    return T


class Detective_Engine():
    def __init__(self, board, detectives_pos):
        self.board=board
        self.belief_state=np.ones(199)*0.005
        self.gen_adjacent_matrix()
        self.detectives_pos=detectives_pos

    def update_belief_after_mrx_move(self, ticket):
        if ticket=="taxi":
            new_belief_state = self.taxi_matrix.T @ self.belief_state
        elif ticket=="bus":
            new_belief_state = self.bus_matrix.T @ self.belief_state
        elif ticket=="underground":
            new_belief_state = self.underground_matrix.T @ self.belief_state
        
        filtered_belief_state = self.kalman_filter(new_belief_state)

        self.belief_state=filtered_belief_state
        return 
    
    def mrx_is_spotted(self, mrx_pos):
        self.belief_state[:]=0
        self.belief_state[int(mrx_pos)]=1
        return



    def gen_adjacent_matrix(self):
        all_nodes = sorted(self.board.board.nodes())
        # Taxi
        taxi_edges = [(u, v) for u, v, d in self.board.board.edges(data=True) if d['type'] == 'taxi']
        taxi_graph = nx.Graph()
        taxi_graph.add_nodes_from(all_nodes)
        taxi_graph.add_edges_from(taxi_edges)
        self.taxi_matrix = nx.to_numpy_array(taxi_graph, nodelist=all_nodes)
        self.taxi_matrix = stochastic_matrix(self.taxi_matrix)


        # Bus
        bus_edges = [(u, v) for u, v, d in self.board.board.edges(data=True) if d['type'] == 'bus']
        bus_graph = nx.Graph()
        bus_graph.add_nodes_from(all_nodes)
        bus_graph.add_edges_from(bus_edges)
        self.bus_matrix = nx.to_numpy_array(bus_graph, nodelist=all_nodes)
        self.bus_matrix = stochastic_matrix(self.bus_matrix)

        
        # Underground
        underground_edges = [(u, v) for u, v, d in self.board.board.edges(data=True) if d['type'] == 'underground']
        underground_graph = nx.Graph()
        underground_graph.add_nodes_from(all_nodes)
        underground_graph.add_edges_from(underground_edges)
        self.underground_matrix = nx.to_numpy_array(underground_graph, nodelist=all_nodes)
        self.underground_matrix=stochastic_matrix(self.underground_matrix)

        return 
    
    
    def kalman_filter(self, b_old):
        # mask di lunghezza del belief vector: 1 = libero, 0 = occupato da detective
        mask = np.ones_like(b_old, dtype=float)
        for pos in self.detectives_pos:
            try:
                idx = int(pos)
                if 0 <= idx < len(mask):
                    mask[idx] = 0.0
            except:
                continue

        b_new = b_old * mask
        s = b_new.sum()
        if s > 0:
            b_new /= s
        else:
            b_new = np.ones_like(b_old) / len(b_old)
        return b_new