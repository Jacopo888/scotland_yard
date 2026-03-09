import tkinter as tk
import numpy as np


class BeliefStateVisualizer:
    def __init__(self, board, width=1200, height=900):
        self.board = board
        self.pos = board.pos
        self.width = width
        self.height = height

        # Finestra separata (Toplevel) così non interferisce con la board
        self.win = tk.Toplevel(board.root)
        self.win.title("Belief State Visualization")
        self.canvas = tk.Canvas(self.win, width=width, height=height, bg="#1e1e1e")
        self.canvas.pack()

        self._node_items = []

        # Disegna archi una sola volta
        self._draw_edges()
        self.win.update()

    def _draw_edges(self):
        type_to_color = {
            "taxi": "#555500",
            "bus": "#003366",
            "underground": "#550000",
        }
        for u, v, data in self.board.board.edges(data=True):
            color = type_to_color.get(data["type"], "#333333")
            x1, y1 = self.pos[u]
            x2, y2 = self.pos[v]
            self.canvas.create_line(x1, y1, x2, y2, fill=color, width=1, tags="edge")

    def _prob_to_color(self, prob, max_prob):
        """Mappa una probabilità a un colore dal nero al rosso intenso."""
        if max_prob <= 0:
            return "#1e1e1e"
        t = min(prob / max_prob, 1.0)
        # Gradiente: nero -> rosso scuro -> rosso -> arancione/bianco
        r = int(40 + 215 * t)
        g = int(30 * (1 - t))
        b = int(30 * (1 - t))
        return f"#{r:02x}{g:02x}{b:02x}"

    def show(self, belief_state):
        # Cancella nodi precedenti
        for item_id in self._node_items:
            self.canvas.delete(item_id)
        self._node_items.clear()

        b = np.array(belief_state)
        max_prob = b.max() if b.max() > 0 else 1.0

        for node, (x, y) in self.pos.items():
            idx = int(node) - 1  # nodi numerati da 1
            if idx < 0 or idx >= len(b):
                continue

            prob = b[idx]
            color = self._prob_to_color(prob, max_prob)
            r = 10

            # Nodi con probabilità alta sono più grandi
            if prob > 0:
                r = int(10 + 8 * (prob / max_prob))

            oval = self.canvas.create_oval(
                x - r, y - r, x + r, y + r,
                fill=color, outline="", tags="bnode"
            )
            self._node_items.append(oval)

            # Mostra label solo se probabilità significativa
            if prob > max_prob * 0.1:
                label = f"{node}\n{prob:.2f}"
                text = self.canvas.create_text(
                    x, y, text=label, fill="white",
                    font=("Arial", 6, "bold"), tags="bnode"
                )
                self._node_items.append(text)
            else:
                text = self.canvas.create_text(
                    x, y, text=node, fill="#888888",
                    font=("Arial", 5), tags="bnode"
                )
                self._node_items.append(text)

        # Nodi sopra archi
        self.canvas.tag_raise("bnode", "edge")

        # Disegna legenda/colorbar
        self._draw_colorbar(max_prob)

        self.win.update()

    def _draw_colorbar(self, max_prob):
        # Cancella colorbar precedente
        self.canvas.delete("colorbar")

        bar_x = self.width - 60
        bar_y_top = 50
        bar_y_bottom = 300
        bar_width = 20
        steps = 50

        for i in range(steps):
            t = i / steps
            prob = t * max_prob
            color = self._prob_to_color(prob, max_prob)
            y1 = bar_y_bottom - (i / steps) * (bar_y_bottom - bar_y_top)
            y2 = bar_y_bottom - ((i + 1) / steps) * (bar_y_bottom - bar_y_top)
            rect = self.canvas.create_rectangle(
                bar_x, y2, bar_x + bar_width, y1,
                fill=color, outline="", tags="colorbar"
            )
            self._node_items.append(rect)

        # Labels
        for val, label in [(0, "0.00"), (max_prob / 2, f"{max_prob / 2:.3f}"), (max_prob, f"{max_prob:.3f}")]:
            t = val / max_prob if max_prob > 0 else 0
            y = bar_y_bottom - t * (bar_y_bottom - bar_y_top)
            txt = self.canvas.create_text(
                bar_x + bar_width + 25, y, text=label,
                fill="white", font=("Arial", 8), tags="colorbar"
            )
            self._node_items.append(txt)

        title = self.canvas.create_text(
            bar_x + bar_width // 2, bar_y_top - 20,
            text="P(Mr. X)", fill="white",
            font=("Arial", 10, "bold"), tags="colorbar"
        )
        self._node_items.append(title)