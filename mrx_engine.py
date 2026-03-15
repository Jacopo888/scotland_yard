import math
from copy import deepcopy

from mcts_node import MCTSNode
from utility import play_five_turns

NUM_SIMULATIONS = 25
NUM_EXPLORATIONS = 15
UCB1_CONSTANT = 1.42


class MrxEngine:
    def __init__(self, game, detective_engine):
        self.root = MCTSNode(game, detective_engine, None)
        self.iteration = 1

    def rollout(self, game, detective_engine):
        score = 0
        for _ in range(NUM_SIMULATIONS):
            score += play_five_turns(game, detective_engine)
        return score

    def search(self):
        self.iteration = 1
        self.root.children = []

        root_children = self.root.expand()
        self._explore(self.root, root_children, NUM_EXPLORATIONS)

        best = self._best_child(self.root.children, exploit_only=True)
        return best.game_status.mrx_pos, best.ticket

    def _explore(self, node, nodes, max_iterations):
        while self.iteration < max_iterations:
            best = self._best_child(nodes)
            if best.visits == 0:
                score = self.rollout(deepcopy(best.game_status), deepcopy(best.detective_engine))
                best.backpropagate(score)
                best.check_terminal()
            elif best.visits == 1:
                nodes += best.expand(is_root=False)
            else:
                self._explore(best, best.children, self.iteration + 1)
                self.iteration -= 1
            self.iteration += 1

    def _best_child(self, nodes, exploit_only=False):
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
        if best_node is None:
            return self._best_child(nodes, exploit_only=True)
        return best_node
