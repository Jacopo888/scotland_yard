import tkinter as tk
import networkx as nx


class Board:
    def __init__(self, width=1200, height=900):
        self.root = tk.Tk()
        self.root.title("Scotland Yard Board")
        self.canvas = tk.Canvas(self.root, width=width, height=height, bg="#1e1e1e")
        self.canvas.pack()
        self.width = width
        self.height = height

        # Load the graph
        self.board = nx.MultiGraph()
        with open("./Matrix_generation/connections.txt", "r") as file:
            connections = [line.strip().split(" ") for line in file]
            connections = [(c[0], c[1], {"type": c[2]}) for c in connections]
        stations = [str(i) for i in range(1, 200)]
        self.board.add_nodes_from(stations)
        self.board.add_edges_from(connections)

        # Fixed layout
        pos = nx.spring_layout(self.board, seed=42)
        margin = 50
        self.pos = {
            node: (
                int(margin + (x + 1) / 2 * (width - 2 * margin)),
                int(margin + (y + 1) / 2 * (height - 2 * margin)),
            )
            for node, (x, y) in pos.items()
        }

        self.type_to_color = {
            "taxi": "#ffff00",
            "bus": "#0064ff",
            "underground": "#ff0000",
        }
        self.detectives_pos = []
        self.mrx_pos = None

        # Draw edges once (they don't change)
        self._draw_edges()
        # Placeholder for nodes (will be redrawn)
        self._node_items = []
        self._draw_nodes()
        self.root.update()

    def _draw_edges(self):
        for u, v, data in self.board.edges(data=True):
            color = self.type_to_color.get(data["type"], "#969696")
            x1, y1 = self.pos[u]
            x2, y2 = self.pos[v]
            self.canvas.create_line(x1, y1, x2, y2, fill=color, width=1, tags="edge")

    def _draw_nodes(self):
        # Clear previous nodes
        for item_id in self._node_items:
            self.canvas.delete(item_id)
        self._node_items.clear()

        # Convert detectives_pos to a set of strings for consistent comparison
        det_set = set(str(d) for d in self.detectives_pos)
        mrx_str = str(self.mrx_pos) if self.mrx_pos is not None else None

        for node, (x, y) in self.pos.items():
            color = "#505050"
            r = 10

            if node in det_set:
                color = "#00beff"
                r = 16
            if mrx_str is not None and node == mrx_str:
                color = "#dc143c"
                r = 18

            oval = self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=color, outline="", tags="node")
            text = self.canvas.create_text(x, y, text=node, fill="white", font=("Arial", 7), tags="node")
            self._node_items.append(oval)
            self._node_items.append(text)

        # Ensure nodes are drawn above edges
        self.canvas.tag_raise("node", "edge")

    def update_detectives_pos(self, detectives_pos):
        self.detectives_pos = detectives_pos
        self._draw_nodes()
        self.root.update()

    def update_mrx_position(self, mrx_pos):
        self.mrx_pos = mrx_pos
        self._draw_nodes()
        self.root.update()