import math
from copy import deepcopy

from mcts_node import MCTSNode
from utility import _simulate_turns_inplace, _min_detective_distance

NUM_SIMULATIONS = 25
NUM_EXPLORATIONS = 15
UCB1_CONSTANT = 1.42


class MrxEngine:
    def __init__(self, game, detective_engine):
        self.root = MCTSNode(game, detective_engine, None)
        self.iteration = 1

    def rollout(self, game_status, detective_engine):
        scratch = game_status.copy()
        snap = scratch.snapshot()
        saved_belief = detective_engine.belief_state.copy()
        scratch_engine = detective_engine.copy(scratch.detectives_pos)
        score = 0
        for _ in range(NUM_SIMULATIONS):
            scratch.restore(snap)
            scratch_engine.detectives_pos = scratch.detectives_pos
            scratch_engine.belief_state[:] = saved_belief
            _simulate_turns_inplace(scratch, scratch_engine, max_turns=5)
            score += _min_detective_distance(scratch.detectives_pos, scratch.mrx_pos)
        return score

    def search(self):
        self.iteration = 1
        self.root.children = []

        root_children = self.root.expand()
        if not root_children:
            # Mr.X completamente bloccato: nessuna mossa, nessun ticket.
            return self.root.game_status.mrx_pos, None

        self._explore(self.root, root_children, NUM_EXPLORATIONS)

        best = self._best_child(self.root.children, exploit_only=True)
        if best is None:
            # Tutti i children erano terminali e _best_child non ne ha
            # selezionato nessuno: prendiamo comunque il primo a disposizione.
            best = self.root.children[0]
        return best.game_status.mrx_pos, best.ticket

    def _explore(self, node, nodes, max_iterations):
        while self.iteration < max_iterations:
            best = self._best_child(nodes)
            if best is None:
                return
            if best.visits == 0:
                score = self.rollout(best.game_status, best.detective_engine)
                best.backpropagate(score)
                best.check_terminal()
            elif best.visits == 1:
                nodes += best.expand(is_root=False)
            else:
                self._explore(best, best.children, self.iteration + 1)
                self.iteration -= 1
            self.iteration += 1

    def _best_child(self, nodes, exploit_only=False):
        if not nodes:
            return None
        best_score = float("-inf")
        best_node = None
        for node in nodes:
            if node.is_terminal and not exploit_only:
                continue
            if node.visits == 0:
                ucb1 = float("inf")
            else:
                ucb1 = node.score + UCB1_CONSTANT * math.sqrt(
                    math.log(self.iteration) / node.visits
                )
            if ucb1 > best_score:
                best_score = ucb1
                best_node = node
        if best_node is None and not exploit_only:
            return self._best_child(nodes, exploit_only=True)
        return best_node
