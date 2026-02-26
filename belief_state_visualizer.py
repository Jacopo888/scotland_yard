import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

class BeliefStateVisualizer:
    def __init__(self, board):
        self.board = board
        self.pos = board.pos
        # Creiamo una figura dedicata e salviamo il riferimento
        self.fig = plt.figure(figsize=(10, 8))
        self.fig.canvas.manager.set_window_title("Belief State")

        plt.ion()
        plt.show()

    def show(self, belief_state):
        # 1. Rendiamo attiva questa figura specifica per non sovrascrivere l'altra
        plt.figure(self.fig.number)
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        
        G = self.board.board
        #nodes = [int(i) for i in G.nodes()].sort()

        b = np.array(belief_state)
        probs=b
        # Costruisci lista di probabilità corrispondente ai nodi
        """         if len(b) == len(nodes):
            probs = b
        else:
            probs = []
            for n in nodes:
                try:
                    idx = int(n)
                except ValueError:
                    idx = None
                probs.append(float(b[idx]) if idx is not None and idx < len(b) else 0.0)
            probs = np.array(probs) """

        # 2. Definiamo il massimo per la scala di colore delle sfumature
        max_prob = probs.max()*1.2
        lista=[str(i) for i in list(range(1, 200))]
        # 3. Disegniamo usando una mappa di colori nativa 'Reds'
        nx.draw(
            G,
            pos=self.pos,
            with_labels=True,
            nodelist= lista,
            node_color=probs,           # passiamo direttamente le probabilità
            cmap=plt.cm.Reds,           # usiamo le sfumature di rosso (bianco=0, rosso scuro=max)
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

        # Aggiungi la colorbar laterale
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=max_prob))
        sm.set_array([])
        self.fig.colorbar(sm, ax=self.ax, fraction=0.03, pad=0.04).set_label("Probability of Mr. X's Location", fontsize=10)
        
        self.ax.set_title("Belief State Visualization")
        plt.tight_layout()
        plt.pause(0.1)