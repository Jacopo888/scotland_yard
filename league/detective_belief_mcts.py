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
    teacher_mode: str = "joint"
    joint_top_k: int = 3
    joint_random_actions: int = 1
    value_leaf_weight: float = 0.0


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

    def _candidate_rows(self, game, engine, detective_id):
        legal = _legal_destinations(game, detective_id)
        if not legal:
            return []
        legal_set = set(legal)
        rows_by_destination = {}
        try:
            evaluated = self.default_policy.evaluate(
                game,
                engine.belief_state,
                detective_id,
            )
            for row in evaluated["policy"]:
                destination = str(row["destination"])
                if destination in legal_set:
                    rows_by_destination[destination] = {
                        "destination": destination,
                        "target_action": int(row["target_action"]),
                        "logit": float(row["logit"]),
                        "prob": float(row["prob"]),
                    }
        except Exception:
            rows_by_destination = {}

        if rows_by_destination:
            floor_logit = min(row["logit"] for row in rows_by_destination.values()) - 1.0
        else:
            floor_logit = 0.0
        for destination in legal:
            rows_by_destination.setdefault(
                destination,
                {
                    "destination": destination,
                    "target_action": int(destination) - 1,
                    "logit": float(floor_logit),
                    "prob": 0.0,
                },
            )

        ranked = sorted(
            rows_by_destination.values(),
            key=lambda row: (row["logit"], row["prob"]),
            reverse=True,
        )
        top_k = max(1, int(self.config.joint_top_k))
        chosen = list(ranked[:top_k])
        chosen_destinations = {row["destination"] for row in chosen}
        tail = [row for row in ranked[top_k:] if row["destination"] not in chosen_destinations]
        random_count = min(max(0, int(self.config.joint_random_actions)), len(tail))
        if random_count:
            indices = self.rng.choice(len(tail), size=random_count, replace=False)
            for idx in np.atleast_1d(indices):
                chosen.append(tail[int(idx)])

        total_prob = sum(max(float(row.get("prob", 0.0)), 0.0) for row in chosen)
        if total_prob <= 0:
            uniform = 1.0 / max(len(chosen), 1)
            for row in chosen:
                row["sample_prob"] = uniform
        else:
            floor = 0.15 / max(len(chosen), 1)
            for row in chosen:
                base = max(float(row.get("prob", 0.0)), 0.0) / total_prob
                row["sample_prob"] = float(0.85 * base + floor)
        return chosen

    def _select_candidate_destination(self, rows, greedy=False):
        if not rows:
            return None
        if greedy:
            return max(rows, key=lambda row: (row["logit"], row.get("prob", 0.0)))[
                "destination"
            ]
        probs = np.asarray([max(float(row.get("sample_prob", 0.0)), 0.0) for row in rows])
        if float(probs.sum()) <= 0:
            probs[:] = 1.0 / len(rows)
        else:
            temperature = max(float(self.config.policy_temperature), 1e-6)
            if temperature != 1.0:
                probs = np.power(probs, 1.0 / temperature)
            probs = probs / float(probs.sum())
        return rows[int(self.rng.choice(len(rows), p=probs))]["destination"]

    def _sample_joint_action(self, game, engine, greedy=False):
        draft_game = game.copy()
        draft_engine = DetectiveEngine(
            draft_game.detectives_pos,
            engine.belief_state.copy(),
            skip_filter=True,
        )
        moves = []
        for det_id in range(draft_game.num_detectives):
            rows = self._candidate_rows(draft_game, draft_engine, det_id)
            destination = self._select_candidate_destination(rows, greedy=greedy)
            moves.append(destination)
            if destination is None:
                continue
            _apply_detective_move(draft_game, det_id, destination)
            draft_engine.kalman_filter()
        return tuple(moves)

    def _apply_joint_action(self, game, engine, joint_action):
        captured = False
        for det_id, destination in enumerate(joint_action):
            if destination is None:
                continue
            if destination not in _legal_destinations(game, det_id):
                return False
            _apply_detective_move(game, det_id, destination)
            if game.check_victory(silent=True):
                captured = True
                break
            engine.kalman_filter()
        game.detectives_moves.append(game.detectives_pos[:])
        return captured

    def _rollout_score_joint(self, game, engine, joint_action):
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
            if self._apply_joint_action(game, engine, joint_action):
                return R_CAPTURE

            score += discount * (
                R_STEP + R_DIST_COEF * float(_min_detective_distance(game.detectives_pos, game.mrx_pos))
            )
            if self.config.value_leaf_weight:
                try:
                    value_eval = self.default_policy.evaluate_value(
                        game,
                        engine.belief_state,
                        detective_id=0,
                    )
                    score += float(self.config.value_leaf_weight) * float(
                        value_eval["value_raw"]
                    )
                except Exception:
                    pass

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

    def search_joint(self, game, engine):
        simulations = max(1, int(self.config.simulations))
        explore_budget = min(simulations, max(4, simulations // 2))
        visits = {}
        values = {}
        total_visits = 0
        for sim_idx in range(simulations):
            if sim_idx == 0 or total_visits < explore_budget or not visits:
                joint_action = self._sample_joint_action(game, engine, greedy=(sim_idx == 0))
            else:
                log_total = math.log(max(total_visits, 1))
                joint_action = max(
                    visits,
                    key=lambda action: values[action] / visits[action]
                    + self.config.exploration_c
                    * math.sqrt(log_total / max(visits[action], 1)),
                )
            score = self._rollout_score_joint(game, engine, joint_action)
            visits[joint_action] = visits.get(joint_action, 0) + 1
            values[joint_action] = values.get(joint_action, 0.0) + score
            total_visits += 1

        joint_actions = []
        for action, n in visits.items():
            q_value = values[action] / max(n, 1)
            joint_actions.append(
                {
                    "moves": [None if dst is None else str(dst) for dst in action],
                    "visits": int(n),
                    "q_value": float(q_value),
                }
            )
        selected = max(joint_actions, key=lambda row: (row["q_value"], row["visits"]))
        root_value = sum(row["q_value"] * row["visits"] for row in joint_actions) / max(
            total_visits,
            1,
        )
        self.last_search_info = {
            "teacher_mode": "joint",
            "joint_actions": joint_actions,
            "selected_joint": selected,
            "root_value": float(root_value),
            "joint_root_value": float(root_value),
            "config": {
                **self.config.__dict__,
                "belief_sampling": "categorical_full_belief",
                "joint_action_sampling": "policy_top_k_plus_random_tail",
            },
        }
        return selected["moves"], self.last_search_info

    def decision_info_from_joint(self, joint_info, detective_id, legal_destinations):
        selected_joint = joint_info.get("selected_joint") or {}
        selected_moves = selected_joint.get("moves") or []
        selected_destination = (
            selected_moves[detective_id] if detective_id < len(selected_moves) else None
        )
        legal_set = {str(dst) for dst in legal_destinations}
        prefix = tuple(selected_moves[:detective_id])
        aggregates = {}
        for joint_action in joint_info.get("joint_actions") or []:
            moves = joint_action.get("moves") or []
            if tuple(moves[:detective_id]) != prefix or detective_id >= len(moves):
                continue
            destination = moves[detective_id]
            if destination is None or str(destination) not in legal_set:
                continue
            destination = str(destination)
            visits = int(joint_action.get("visits", 0))
            q_value = float(joint_action.get("q_value", 0.0))
            row = aggregates.setdefault(
                destination,
                {"visits": 0, "value_sum": 0.0},
            )
            row["visits"] += visits
            row["value_sum"] += q_value * visits

        if selected_destination is not None and str(selected_destination) in legal_set:
            aggregates.setdefault(str(selected_destination), {"visits": 1, "value_sum": 0.0})

        actions = []
        for destination, row in sorted(aggregates.items(), key=lambda item: int(item[0])):
            visits = int(row["visits"])
            actions.append(
                {
                    "destination": destination,
                    "target_action": int(destination) - 1,
                    "visits": visits,
                    "q_value": float(row["value_sum"] / max(visits, 1)),
                }
            )

        selected = None
        for row in actions:
            if row["destination"] == str(selected_destination):
                selected = row
                break
        if selected is None and actions:
            selected = max(actions, key=lambda row: (row["visits"], row["q_value"]))

        return {
            "teacher_mode": "joint",
            "detective_id": int(detective_id),
            "actions": actions,
            "selected": selected,
            "root_value": float(joint_info.get("root_value", 0.0)),
            "joint_root_value": float(joint_info.get("joint_root_value", 0.0)),
            "joint_selected_q": float(selected_joint.get("q_value", 0.0)),
            "joint_selected_visits": int(selected_joint.get("visits", 0)),
            "joint_action_count": int(len(joint_info.get("joint_actions") or [])),
            "selected_joint": selected_joint,
            "config": joint_info.get("config", {}),
        }

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
