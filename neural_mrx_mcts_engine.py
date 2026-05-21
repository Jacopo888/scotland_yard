import math
from dataclasses import dataclass, field

import numpy as np
import torch

from detective_engine import DetectiveEngine
from game import Game
from gnn_detective_engine import (
    GNNDetectiveEngine,
    find_latest_checkpoint as find_latest_detective_checkpoint,
)
from gnn_mrx_engine import (
    IDX2REL,
    RELATIONS,
    GNNMrXEngine,
    find_latest_checkpoint as find_latest_mrx_checkpoint,
    is_reveal_turn,
)


NUM_SIMULATIONS = 64
C_PUCT = 1.5
VALUE_SCALE = 10.0
TEMPERATURE = 0.0
ROOT_DIRICHLET = False
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPS = 0.25


@dataclass
class NeuralMCTSEdge:
    destination: str
    ticket: str | None
    flat_action: int
    prior: float
    visits: int = 0
    total_value: float = 0.0
    q_value: float = 0.0
    child: "NeuralMCTSNode | None" = None

    def backup(self, value):
        self.visits += 1
        self.total_value += float(value)
        self.q_value = self.total_value / self.visits


@dataclass
class NeuralMCTSNode:
    game: Game
    detective_engine: DetectiveEngine
    last_mrx_ticket: str
    revealed_positions: list[int] = field(default_factory=list)
    edges: list[NeuralMCTSEdge] = field(default_factory=list)
    expanded: bool = False
    terminal: bool = False


class NeuralMrXMCTSEngine:
    """AlphaZero-style inference search for Mr.X.

    The tree contains only Mr.X decision nodes. Each edge applies one Mr.X
    action and then advances the game through the frozen detective GNN phase,
    so every backed-up value remains from Mr.X's perspective.
    """

    def __init__(
        self,
        mrx_checkpoint_path=None,
        detective_checkpoint_path=None,
        device=None,
        num_simulations=NUM_SIMULATIONS,
        c_puct=C_PUCT,
        value_scale=VALUE_SCALE,
        temperature=TEMPERATURE,
        root_dirichlet=ROOT_DIRICHLET,
        dirichlet_alpha=DIRICHLET_ALPHA,
        dirichlet_eps=DIRICHLET_EPS,
    ):
        mrx_checkpoint_path = mrx_checkpoint_path or find_latest_mrx_checkpoint()
        detective_checkpoint_path = (
            detective_checkpoint_path or find_latest_detective_checkpoint()
        )
        if mrx_checkpoint_path is None:
            raise FileNotFoundError("No Mr.X GNN checkpoint found.")
        if detective_checkpoint_path is None:
            raise FileNotFoundError("No detective GNN checkpoint found.")

        self.mrx_policy = GNNMrXEngine(
            checkpoint_path=mrx_checkpoint_path,
            device=device,
        )
        self.detective_policy = GNNDetectiveEngine(
            checkpoint_path=detective_checkpoint_path,
            device=device,
        )
        self.num_simulations = int(num_simulations)
        self.c_puct = float(c_puct)
        self.value_scale = float(value_scale)
        self.temperature = float(temperature)
        self.root_dirichlet = bool(root_dirichlet)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.dirichlet_eps = float(dirichlet_eps)
        self.last_search_info = None
        self.reset()

    @property
    def checkpoint_path(self):
        return self.mrx_policy.checkpoint_path

    @property
    def detective_checkpoint_path(self):
        return self.detective_policy.checkpoint_path

    @property
    def metadata(self):
        return {
            "mrx": self.mrx_policy.metadata,
            "detective": self.detective_policy.metadata,
            "num_simulations": self.num_simulations,
            "c_puct": self.c_puct,
            "value_scale": self.value_scale,
            "temperature": self.temperature,
            "root_dirichlet": self.root_dirichlet,
        }

    def reset(self):
        self.last_mrx_ticket = ""
        self.revealed_positions = []
        self.last_search_info = None
        self.mrx_policy.reset()
        self.detective_policy.reset()

    def observe_mrx_move(self, game, ticket):
        if ticket is not None:
            self.last_mrx_ticket = ticket
        if is_reveal_turn(game.turn):
            self.revealed_positions.append(int(game.mrx_pos))
        self._sync_trackers(self.last_mrx_ticket, self.revealed_positions)

    def choose_action(self, game, belief_state):
        root_engine = DetectiveEngine(
            game.detectives_pos,
            belief_state=belief_state.copy(),
            skip_filter=True,
        )
        root = NeuralMCTSNode(
            game=game.copy(),
            detective_engine=root_engine,
            last_mrx_ticket=self.last_mrx_ticket,
            revealed_positions=self.revealed_positions[:],
        )
        root_value = self._expand(root)
        if not root.edges:
            self.last_search_info = {
                "root_value": root_value,
                "selected": {
                    "destination": game.mrx_pos,
                    "ticket": None,
                    "flat_action": -1,
                    "visits": 0,
                    "prior": 1.0,
                    "q_value": root_value,
                },
                "actions": [],
            }
            self._sync_trackers(self.last_mrx_ticket, self.revealed_positions)
            return game.mrx_pos, None

        if self.root_dirichlet:
            self._apply_root_dirichlet(root)

        for _ in range(self.num_simulations):
            self._simulate(root)

        selected = self._select_search_output(root)
        self.last_search_info = {
            "root_value": root_value,
            "selected": self._edge_summary(selected),
            "actions": [self._edge_summary(edge) for edge in root.edges],
        }
        self._sync_trackers(self.last_mrx_ticket, self.revealed_positions)
        return selected.destination, selected.ticket

    def play_mrx_turn(self, game, belief_state):
        destination, ticket = self.choose_action(game, belief_state)
        game.x_automated_turn(destination, ticket)
        return ticket

    def visit_distribution(self):
        if not self.last_search_info:
            return []
        total = sum(edge["visits"] for edge in self.last_search_info["actions"])
        if total <= 0:
            return []
        return [
            {"flat": edge["flat_action"], "prob": edge["visits"] / total}
            for edge in self.last_search_info["actions"]
            if edge["flat_action"] >= 0 and edge["visits"] > 0
        ]

    def _simulate(self, node):
        if node.terminal or node.game.check_victory(silent=True):
            node.terminal = True
            return self._terminal_value(node.game)

        if not node.expanded:
            return self._expand(node)

        edge = self._select_edge(node)
        if edge.child is None:
            edge.child = self._transition(node, edge)

        value = self._simulate(edge.child)
        edge.backup(value)
        return value

    @torch.no_grad()
    def _expand(self, node):
        if node.game.check_victory(silent=True):
            node.terminal = True
            node.expanded = True
            return self._terminal_value(node.game)

        self._set_mrx_tracker(node.last_mrx_ticket, node.revealed_positions)
        sample = self.mrx_policy.build_input(
            node.game,
            node.detective_engine.belief_state,
        )
        legal_mask = sample["legal_mask"].astype(bool)

        if not legal_mask.any():
            node.edges = [
                NeuralMCTSEdge(
                    destination=node.game.mrx_pos,
                    ticket=None,
                    flat_action=-1,
                    prior=1.0,
                )
            ]
            node.expanded = True
            return self._blocked_value(node)

        batch = self.mrx_policy._collate_one(sample)
        logits, value = self.mrx_policy.model(
            batch,
            self.mrx_policy.dense_adj,
            self.mrx_policy.node_static,
        )
        legal_t = batch["legal_mask"].bool()
        masked_logits = logits.masked_fill(~legal_t, -1e9)
        legal_indices = np.flatnonzero(legal_mask)
        priors_t = torch.softmax(masked_logits[0, legal_t[0]], dim=0)
        priors = priors_t.detach().cpu().numpy()

        edges = []
        for flat, prior in zip(legal_indices, priors):
            destination_idx = int(flat) // len(RELATIONS)
            ticket_idx = int(flat) % len(RELATIONS)
            edges.append(
                NeuralMCTSEdge(
                    destination=str(destination_idx + 1),
                    ticket=IDX2REL[ticket_idx],
                    flat_action=int(flat),
                    prior=float(prior),
                )
            )

        node.edges = edges
        node.expanded = True
        value_norm = float(value[0].item())
        raw_return = value_norm * self.mrx_policy.return_std + self.mrx_policy.return_mean
        return self._normalize_value(raw_return)

    def _transition(self, node, edge):
        child_game = node.game.copy()
        child_engine = node.detective_engine.copy(child_game.detectives_pos)

        child_game.x_automated_turn(edge.destination, edge.ticket)
        next_last_ticket = node.last_mrx_ticket
        next_revealed = node.revealed_positions[:]
        if edge.ticket is not None:
            next_last_ticket = edge.ticket

        if child_game.check_victory(silent=True):
            return NeuralMCTSNode(
                game=child_game,
                detective_engine=child_engine,
                last_mrx_ticket=next_last_ticket,
                revealed_positions=next_revealed,
                terminal=True,
            )

        child_engine.update_belief_after_mrx_move(edge.ticket)
        if is_reveal_turn(child_game.turn):
            child_engine.mrx_is_spotted(child_game.mrx_pos)
            next_revealed.append(int(child_game.mrx_pos))

        self._play_frozen_detective_phase(
            child_game,
            child_engine,
            next_last_ticket,
            next_revealed,
        )

        return NeuralMCTSNode(
            game=child_game,
            detective_engine=child_engine,
            last_mrx_ticket=next_last_ticket,
            revealed_positions=next_revealed,
            terminal=child_game.check_victory(silent=True),
        )

    def _play_frozen_detective_phase(
        self,
        game,
        detective_engine,
        last_mrx_ticket,
        revealed_positions,
    ):
        self._set_detective_tracker(last_mrx_ticket, revealed_positions)
        for detective_id in range(game.num_detectives):
            self.detective_policy.play_detective_turn(
                game,
                detective_engine.belief_state,
                detective_id,
            )
            if game.check_victory(silent=True):
                break
            detective_engine.kalman_filter()
        game.detectives_moves.append(game.detectives_pos[:])

    def _select_edge(self, node):
        total_visits = sum(edge.visits for edge in node.edges)
        sqrt_total = math.sqrt(max(1, total_visits))
        best_score = float("-inf")
        best_edge = node.edges[0]
        for edge in node.edges:
            score = (
                edge.q_value
                + self.c_puct
                * edge.prior
                * sqrt_total
                / (1 + edge.visits)
            )
            if score > best_score:
                best_score = score
                best_edge = edge
        return best_edge

    def _select_search_output(self, root):
        if self.temperature <= 0:
            return max(root.edges, key=lambda edge: (edge.visits, edge.q_value, edge.prior))

        visits = np.array([edge.visits for edge in root.edges], dtype=np.float64)
        if visits.sum() <= 0:
            return max(root.edges, key=lambda edge: edge.prior)
        weights = visits ** (1.0 / self.temperature)
        if weights.sum() <= 0:
            return max(root.edges, key=lambda edge: edge.prior)
        weights = weights / weights.sum()
        idx = int(np.random.choice(len(root.edges), p=weights))
        return root.edges[idx]

    def _apply_root_dirichlet(self, root):
        if not root.edges:
            return
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(root.edges))
        for edge, noise_value in zip(root.edges, noise):
            edge.prior = (
                (1.0 - self.dirichlet_eps) * edge.prior
                + self.dirichlet_eps * float(noise_value)
            )

    def _blocked_value(self, node):
        blocked_game = node.game.copy()
        blocked_engine = node.detective_engine.copy(blocked_game.detectives_pos)
        blocked_game.x_automated_turn(blocked_game.mrx_pos, None)
        if blocked_game.check_victory(silent=True):
            return self._terminal_value(blocked_game)
        blocked_engine.update_belief_after_mrx_move(None)
        if is_reveal_turn(blocked_game.turn):
            blocked_engine.mrx_is_spotted(blocked_game.mrx_pos)
        return -0.25

    def _terminal_value(self, game):
        if game.mrx_pos in game.detectives_pos:
            return -1.0
        if game.turn >= 22:
            return 1.0
        if game.winner == 1:
            return 1.0
        return -1.0

    def _normalize_value(self, raw_return):
        return float(np.tanh(float(raw_return) / self.value_scale))

    def _sync_trackers(self, last_mrx_ticket, revealed_positions):
        self._set_mrx_tracker(last_mrx_ticket, revealed_positions)
        self._set_detective_tracker(last_mrx_ticket, revealed_positions)

    def _set_mrx_tracker(self, last_mrx_ticket, revealed_positions):
        self.mrx_policy.last_mrx_ticket = last_mrx_ticket
        self.mrx_policy.revealed_positions = list(revealed_positions)

    def _set_detective_tracker(self, last_mrx_ticket, revealed_positions):
        self.detective_policy.last_mrx_ticket = last_mrx_ticket
        self.detective_policy.revealed_positions = list(revealed_positions)

    def _edge_summary(self, edge):
        return {
            "destination": edge.destination,
            "ticket": edge.ticket,
            "flat_action": int(edge.flat_action),
            "prior": float(edge.prior),
            "visits": int(edge.visits),
            "q_value": float(edge.q_value),
        }
