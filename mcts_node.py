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

        # Se entriamo in stato terminale (cattura o turn>=22) non espandiamo
        # nulla: marchiamo il nodo come terminale e basta. Senza questo
        # check, find_legal_moves_x verrebbe chiamato su uno stato di
        # cattura e produrrebbe pseudo-figli con belief stale.
        if game.check_victory(silent=True):
            self.is_terminal = True
            return []

        available_moves = game.find_legal_moves_x()
        detective_set = set(game.detectives_pos)

        # Le mosse di Mr.X che lo portano su una casella di detective sono
        # catture immediate: in play_mrx_turn fanno early-return SALTANDO
        # update_belief_after_mrx_move. Il MCTS, espandendo i loro
        # discendenti, propagherebbe la belief precedente (stale): qualunque
        # successivo M_t @ belief darebbe sum=0 quando il vehicle non e'
        # compatibile con il nodo congelato (caso A) oppure quando i suoi
        # vicini coincidono con detective (caso B). Da qui le cascade
        # documentate dei "somma nulla".
        capture_moves = []
        for vehicle, nodes in available_moves.items():
            for node in nodes:
                if node in detective_set:
                    capture_moves.append((vehicle, node))
                    continue
                child_game, child_engine = play_mrx_turn(node, game, engine, vehicle)
                child = MCTSNode(child_game, child_engine, vehicle)
                child.parent = self
                self.children.append(child)

        # Se nessuna mossa "viva" e' rimasta (Mr.X circondato), creiamo
        # comunque i nodi cattura ma marcati gia' terminali, in modo che
        # search() abbia almeno qualcosa da restituire.
        if not self.children and capture_moves:
            for vehicle, node in capture_moves:
                child_game, child_engine = play_mrx_turn(node, game, engine, vehicle)
                child = MCTSNode(child_game, child_engine, vehicle)
                child.is_terminal = True
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
