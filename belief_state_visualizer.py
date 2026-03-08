import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

class BeliefStateVisualizer:
    def __init__(self, board):
        self.board = board
        self.pos = board.pos
        self.fig = plt.figure(figsize=(10, 8))
        self.fig.canvas.manager.set_window_title("Belief State")

        plt.ion()
        plt.show()

    def show(self, belief_state):
        plt.figure(self.fig.number)
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        G = self.board.board
        b = np.array(belief_state)
        probs=b

        max_prob = probs.max()
        lista=[str(i) for i in list(range(1, 200))]
        nx.draw(
            G,
            pos=self.pos,
            with_labels=True,
            nodelist= lista,
            node_color=probs,           
            cmap=plt.cm.Reds,           
            vmin=0.0,                   
            vmax=max_prob,              
            font_color="black",
            font_size=8,
            font_weight='bold',
            edge_color='gray',
            width=1.5,
            node_size=300,
            ax=self.ax
        )

        #sidebar info 
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=max_prob))
        sm.set_array([])
        self.fig.colorbar(sm, ax=self.ax, fraction=0.03, pad=0.04).set_label("Probability of Mr. X's Location", fontsize=10)
        
        self.ax.set_title("Belief State Visualization")
        plt.tight_layout()
        plt.pause(0.1)