import networkx as nx
import matplotlib.pyplot as plt

stations=list(range(1,200))
stations=[str(i) for i in stations]


def add_road(board,station1, station2, veichle):
    board.add_edge(station1,station2, type=veichle)
    return

class Board():
    def __init__(self, ):
        self.board = nx.MultiGraph()
        with open("connections.txt", "r") as file:
            connections = [line.strip().split(" ") for line in file]
            connections = [(c[0], c[1], {"type": c[2]}) for c in connections]
        self.board.add_nodes_from(stations)
        self.board.add_edges_from(connections)
        self.type_to_color = {
            "taxi": "yellow",
            "bus": "blue",
            "underground": "red"
        }
        self.detectives_pos = []
        self.mrx_pos = None
        self.pos = nx.spring_layout(self.board, seed=42)  # fixed layout for consistency
        self.fig = plt.figure(figsize=(10, 8))
        self.fig.canvas.manager.set_window_title("Scotland Yard Board")
        self._draw_board()

    def _draw_board(self):
        plt.figure(self.fig.number)
        plt.clf()
        edge_colors = [
            self.type_to_color.get(data["type"], "black")
            for _, _, data in self.board.edges(data=True)
        ]
        # Draw all nodes as gray by default
        nx.draw(
            self.board,
            pos=self.pos,
            with_labels=True,
            node_color="#222222",
            font_color="white",
            font_size=8,
            font_weight='bold',
            edge_color=edge_colors,
            width=2.5,
            node_size=300
        )
        # Draw detectives (blue, larger)
        if self.detectives_pos:
            for i, dpos in enumerate(self.detectives_pos):
                if str(dpos) in self.pos:
                    nx.draw_networkx_nodes(
                        self.board, self.pos, nodelist=[str(dpos)], node_color='deepskyblue', node_size=500, label=f"Detective {i+1}")
        # Draw Mr. X (red, largest)
        if self.mrx_pos and str(self.mrx_pos) in self.pos:
            nx.draw_networkx_nodes(
                self.board, self.pos, nodelist=[str(self.mrx_pos)], node_color='crimson', node_size=600, label="Mr. X")
        # Only show legend once
        handles = [plt.Line2D([0], [0], marker='o', color='w', label='Detective', markerfacecolor='deepskyblue', markersize=10),
                   plt.Line2D([0], [0], marker='o', color='w', label='Mr. X', markerfacecolor='crimson', markersize=12)]
        plt.legend(handles=handles, loc='upper left')
        plt.tight_layout()
        plt.pause(0.1)
        plt.draw()

    def update_detectives_pos(self, detectives_pos):
        self.detectives_pos = detectives_pos
        self._draw_board()

    def update_mrx_position(self, mrx_pos):
        self.mrx_pos = mrx_pos
        self._draw_board()

        
        



