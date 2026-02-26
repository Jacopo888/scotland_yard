import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

class BeliefStateVisualizer:
    def __init__(self, board):
        self.board = board
        self.pos = board.pos
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.ion()
        plt.show()

    def show(self, belief_state):
        self.ax.clear()
        G = self.board.board
        nodes = sorted(G.nodes())  # stesso ordine usato altrove
        b = np.array(belief_state)
        # costruisci lista di probabilità corrispondente ai nodi
        if len(b) == len(nodes):
            probs = b
        else:
            probs = []
            for n in nodes:
                try:
                    idx = int(n)
                except:
                    idx = None
                probs.append(float(b[idx]) if idx is not None and idx < len(b) else 0.0)
            probs = np.array(probs)

        # normalizza per mappatura colori
        if probs.max() > 0:
            norm = probs / probs.max()
        else:
            norm = np.zeros_like(probs)

        # colore: verde (low) -> rosso (high)
        node_colors = [(float(p), 1.0 - float(p), 0.0) for p in norm]

        nx.draw(
            G,
            pos=self.pos,
            with_labels=True,
            node_color=node_colors,
            font_color="black",
            font_size=8,
            font_weight='bold',
            edge_color='gray',
            width=1.5,
            node_size=300,
            ax=self.ax
        )

        sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap("RdYlGn_r"), norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        self.fig.colorbar(sm, ax=self.ax, fraction=0.03, pad=0.04).set_label("Probability of Mr. X's Location", fontsize=10)
        plt.title("Belief State Visualization")
        plt.tight_layout()
        plt.pause(0.1)