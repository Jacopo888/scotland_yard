import json
import math
from dataclasses import dataclass

import numpy as np

from detective_engine import DetectiveEngine
from game import ADJ
from gnn_detective_engine import GNNDetectiveEngine
from utility import _min_detective_distance


R_CAPTURE = 10.0
R_TIMEOUT = -10.0
R_STEP = -0.05
R_DIST_COEF = -0.10


def _snapshot_policy(policy):
    state = {}
    for name in ("last_mrx_ticket", "revealed_positions", "last_value", "last_action_info"):
        if hasattr(policy, name):
            value = getattr(policy, name)
            state[name] = value[:] if isinstance(value, list) else value
    return state


def _restore_policy(policy, state):
    for name, value in state.items():
        setattr(policy, name, value[:] if isinstance(value, list) else value)


def _maybe_reset(policy):
    reset = getattr(policy, "reset", None)
    if reset is not None:
        reset()


def _maybe_observe_mrx(policy, game, ticket):
    observe = getattr(policy, "observe_mrx_move", None)
    if observe is not None:
        observe(game, ticket)


def _is_reveal_turn(turn):
    return (int(turn) - 3) % 5 == 0


def sample_mrx_position_from_belief(belief_state, rng, forbidden_positions=()):
    probs = np.asarray(belief_state, dtype=np.float64).copy()
    probs[~np.isfinite(probs)] = 0.0
    probs = np.maximum(probs, 0.0)
    for pos in forbidden_positions:
        idx = int(pos) - 1
        if 0 <= idx < len(probs):
            probs[idx] = 0.0
    total = probs.sum()
    if total <= 0:
        probs[:] = 1.0
        for pos in forbidden_positions:
            idx = int(pos) - 1
            if 0 <= idx < len(probs):
                probs[idx] = 0.0
        probs /= probs.sum()
    else:
        probs /= total
    return str(int(rng.choice(len(probs), p=probs)) + 1)


def _apply_detective_move(game, detective_id, destination):
    origin = game.detectives_pos[detective_id]
    vehicle = game.use_ticket(detective_id, origin, destination)
    game.detectives_pos[detective_id] = destination
    return vehicle


def _legal_destinations(game, detective_id):
    occupied = set(
        game.detectives_pos[:detective_id] + game.detectives_pos[detective_id + 1 :]
    )
    tickets = game.detective_tickets[detective_id]
    out = []
    for destination, types in ADJ[game.detectives_pos[detective_id]].items():
        if destination in occupied:
            continue
        if any(vehicle != "water" and tickets.get(vehicle, 0) > 0 for vehicle in types):
            out.append(str(destination))
    return sorted(out, key=lambda x: int(x))


@dataclass
class BeliefMCTSConfig:
    simulations: int = 32
    rollout_turns: int = 3
    exploration_c: float = 1.35
    policy_temperature: float = 1.0
    seed: int = 20260522


class BeliefMCTSDetectiveTeacher:
    def __init__(
        self,
        detective_checkpoint,
        mrx_policy,
        device=None,
        config=None,
    ):
        self.default_policy = GNNDetectiveEngine(
            checkpoint_path=detective_checkpoint,
            device=device,
        )
        self.mrx_policy = mrx_policy
        self.config = config or BeliefMCTSConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.last_search_info = None

    def reset(self):
        self.default_policy.reset()
        _maybe_reset(self.mrx_policy)
        self.last_search_info = None

    def observe_mrx_move(self, game, ticket):
        self.default_policy.observe_mrx_move(game, ticket)
        _maybe_observe_mrx(self.mrx_policy, game, ticket)

    def _play_default_detectives_from(self, game, engine, start_detective_id):
        for det_id in range(start_detective_id, game.num_detectives):
            self.default_policy.play_detective_turn(game, engine.belief_state, det_id)
            if game.check_victory(silent=True):
                return True
            engine.kalman_filter()
        game.detectives_moves.append(game.detectives_pos[:])
        return False

    def _play_mrx_phase(self, game, engine):
        ticket = self.mrx_policy.play_mrx_turn(game, engine.belief_state)
        terminal = game.check_victory(silent=True)
        if not terminal:
            engine.update_belief_after_mrx_move(ticket)
            if _is_reveal_turn(game.turn):
                engine.mrx_is_spotted(game.mrx_pos)
            self.default_policy.observe_mrx_move(game, ticket)
            _maybe_observe_mrx(self.mrx_policy, game, ticket)
        return ticket, terminal

    def _rollout_score(self, game, engine, detective_id, destination):
        game = game.copy()
        sampled_mrx_pos = sample_mrx_position_from_belief(
            engine.belief_state,
            self.rng,
            forbidden_positions=game.detectives_pos,
        )
        game.mrx_pos = sampled_mrx_pos
        engine = DetectiveEngine(
            game.detectives_pos,
            engine.belief_state.copy(),
            skip_filter=True,
        )
        default_state = _snapshot_policy(self.default_policy)
        mrx_state = _snapshot_policy(self.mrx_policy)
        score = 0.0
        discount = 1.0
        try:
            _apply_detective_move(game, detective_id, destination)
            if game.check_victory(silent=True):
                return R_CAPTURE
            engine.kalman_filter()

            if self._play_default_detectives_from(game, engine, detective_id + 1):
                return R_CAPTURE

            score += discount * (
                R_STEP + R_DIST_COEF * float(_min_detective_distance(game.detectives_pos, game.mrx_pos))
            )
            for _ in range(self.config.rollout_turns):
                _, terminal = self._play_mrx_phase(game, engine)
                if terminal:
                    terminal_reward = R_TIMEOUT if game.winner == 1 else R_CAPTURE
                    score += discount * terminal_reward
                    break

                if self._play_default_detectives_from(game, engine, 0):
                    score += discount * R_CAPTURE
                    break

                score += discount * (
                    R_STEP
                    + R_DIST_COEF
                    * float(_min_detective_distance(game.detectives_pos, game.mrx_pos))
                )
                discount *= 0.95
        finally:
            _restore_policy(self.default_policy, default_state)
            _restore_policy(self.mrx_policy, mrx_state)
        return float(score)

    def search(self, game, engine, detective_id):
        candidates = _legal_destinations(game, detective_id)
        if not candidates:
            self.last_search_info = {
                "detective_id": int(detective_id),
                "actions": [],
                "selected": None,
                "root_value": 0.0,
            }
            return None, self.last_search_info

        visits = {destination: 0 for destination in candidates}
        values = {destination: 0.0 for destination in candidates}
        total_visits = 0
        for sim_idx in range(max(1, int(self.config.simulations))):
            unvisited = [dst for dst in candidates if visits[dst] == 0]
            if unvisited:
                destination = unvisited[sim_idx % len(unvisited)]
            else:
                log_total = math.log(max(total_visits, 1))
                destination = max(
                    candidates,
                    key=lambda dst: values[dst] / visits[dst]
                    + self.config.exploration_c * math.sqrt(log_total / visits[dst]),
                )
            score = self._rollout_score(game, engine, detective_id, destination)
            visits[destination] += 1
            values[destination] += score
            total_visits += 1

        actions = []
        for destination in candidates:
            n = visits[destination]
            q_value = values[destination] / max(n, 1)
            actions.append(
                {
                    "destination": destination,
                    "target_action": int(destination) - 1,
                    "visits": int(n),
                    "q_value": float(q_value),
                }
            )

        selected = max(actions, key=lambda row: (row["q_value"], row["visits"]))
        root_value = sum(row["q_value"] * row["visits"] for row in actions) / max(
            total_visits, 1
        )
        self.last_search_info = {
            "detective_id": int(detective_id),
            "actions": actions,
            "selected": selected,
            "root_value": float(root_value),
            "config": {
                **self.config.__dict__,
                "belief_sampling": "categorical_full_belief",
            },
        }
        return selected["destination"], self.last_search_info

    def play_detective_turn(self, game, engine, detective_id):
        destination, info = self.search(game, engine, detective_id)
        if destination is None:
            return None, info
        vehicle = _apply_detective_move(game, detective_id, destination)
        return (destination, vehicle), info


def dumps_search_info(info):
    return json.dumps(info, separators=(",", ":"), sort_keys=True)
