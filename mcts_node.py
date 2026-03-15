from utility import play_mrx_turn, play_detectives_turn


class MCTSNode:
    def __init__(self, game_status, detective_engine, ticket):
        self.game_status = game_status
        self.detective_engine = detective_engine
        self.parent = None
        self.children = []
        self.score = 0
        self.visits = 0
        self.is_terminal = False
        self.ticket = ticket

    def expand(self, is_root=True):
        if is_root:
            game = self.game_status
            engine = self.detective_engine
        else:
            game, engine = play_detectives_turn(self.game_status, self.detective_engine)

        available_moves = game.find_legal_moves_x(allow_double=False)
        for vehicle, nodes in available_moves.items():
            for node in nodes:
                child_game, child_engine = play_mrx_turn(node, game, engine, vehicle)
                child = MCTSNode(child_game, child_engine, vehicle)
                child.parent = self
                self.children.append(child)
        return self.children[:]

    def backpropagate(self, score):
        self.visits += 1
        self.score += score
        if self.parent is not None:
            self.parent.backpropagate(score)

    def check_terminal(self):
        game, _ = play_detectives_turn(self.game_status, self.detective_engine)
        if game.check_victory(silent=True):
            self.is_terminal = True
            return True
        return False
