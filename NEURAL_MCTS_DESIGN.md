# Neural Mr.X MCTS Design

Last updated: 2026-05-21.

This document specifies the proposed AlphaGo-style Neural MCTS for Mr.X.
It is a future experimental branch after `mrx_ppo_v001`, not a replacement for
the current PPO/self-play league.

## Goal

Use the trained Mr.X R-GNN as both:

- policy prior over legal `(destination, ticket)` actions;
- value estimate for leaf states.

Then use MCTS to improve action selection at inference time and, if successful,
to generate stronger policy targets for later training.

Current starting network:

```
Notebook/Models/mrx/mrx_ppo_v001.pt
```

Current frozen detective opponent for first experiments:

```
Notebook/Models/detectives/detective_ppo_v001.pt
```

## Why This Is Different From Current MCTS

Current `MrxEngine`:

- expands Mr.X moves;
- evaluates children through short heuristic/random rollouts;
- uses heuristic detectives inside rollout;
- does not use the learned Mr.X policy/value.

Neural MCTS:

- expands Mr.X moves using neural priors;
- evaluates leaves with the Mr.X value head;
- can simulate the real frozen detective GNN between Mr.X decision nodes;
- returns either argmax visit action for play or visit distribution for training.

This makes search closer to the actual matchup against GNN detectives.

## State Representation

Use **Mr.X decision nodes only**.

A node represents a state where detectives have already moved and Mr.X is about
to act. The node must contain enough information to rebuild the exact model
input:

- `game` copy;
- `DetectiveEngine` copy, including belief state;
- Mr.X policy tracker:
  - `last_mrx_ticket`;
  - `revealed_positions`;
- terminal status;
- edge statistics for legal Mr.X actions.

Important invariant:

```
game.mrx_pos and game.detectives_pos must remain string node ids.
```

## Action Space

Action is ticket-specific:

```
(destination, ticket)
```

Flattening matches `gnn_mrx_engine.py`:

```
flat_action = (int(destination) - 1) * 4 + relation_idx
relation order = taxi, bus, underground, water
```

Legal actions:

- adjacent to current Mr.X position;
- Mr.X has the ticket;
- water allowed only for Mr.X;
- moving onto a detective is legal in game code but terminal bad. Do not
  silently remove it unless an explicit no-suicide ablation is being tested.

## Node Statistics

For each edge/action from a node store:

```python
N[s, a]  # visits
W[s, a]  # total backed-up value
Q[s, a]  # W / N
P[s, a]  # policy prior from Mr.X R-GNN
```

Value is always from Mr.X's perspective:

- terminal Mr.X timeout/survival: `+1`;
- terminal detective capture: `-1`;
- non-terminal neural value: normalized to roughly `[-1, 1]`.

Because the current value head predicts normalized return-to-go, convert it for
MCTS with:

```python
raw_return = value_norm * return_std + return_mean
mcts_value = tanh(raw_return / 10.0)
```

Do not back up unbounded raw returns into PUCT.

## PUCT Selection

Use AlphaZero-style PUCT:

```text
score(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))
```

Recommended first settings:

```python
NUM_SIMULATIONS = 64
C_PUCT = 1.5
VALUE_SCALE = 10.0
TEMPERATURE = 0.0   # argmax visits for evaluation
ROOT_DIRICHLET = False
```

For training data generation later:

```python
ROOT_DIRICHLET = True
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPS = 0.25
TEMPERATURE = 1.0 for early turns, then 0.25 or 0.0
```

## Expansion

When expanding a leaf:

1. Build Mr.X input with `GNNMrXEngine.build_input(game, belief_state)`.
2. Run the model:

   ```python
   logits, value = model(batch, dense_adj, node_static)
   ```

3. Mask illegal actions.
4. Compute priors:

   ```python
   priors = softmax(masked_logits over legal actions)
   ```

5. Create child edge records for legal actions.
6. Return transformed value for backup.

If no legal actions exist:

- apply blocked turn semantics;
- usually treat as bad/non-winning unless timeout is reached;
- be careful to keep current game rules aligned with `Game.x_automated_turn`.

## Transition Function

For each selected action `(destination, ticket)`:

1. Copy game and belief engine.
2. Apply Mr.X move:

   ```python
   game.x_automated_turn(destination, ticket)
   ```

3. If terminal, child is terminal.
4. Update belief:

   ```python
   engine.update_belief_after_mrx_move(ticket)
   if reveal turn:
       engine.mrx_is_spotted(game.mrx_pos)
   ```

5. Update Mr.X tracker:

   ```python
   last_mrx_ticket = ticket
   if reveal turn:
       revealed_positions.append(int(game.mrx_pos))
   ```

6. Simulate the detective phase using frozen `GNNDetectiveEngine`, not the
   heuristic policy:

   ```python
   for detective_id in 0..4:
       detective_policy.play_detective_turn(game, engine.belief_state, detective_id)
       if capture: terminal loss
       engine.kalman_filter()
   game.detectives_moves.append(game.detectives_pos[:])
   ```

7. The resulting state is the next Mr.X decision node.

This "Mr.X decision node only" design keeps the tree smaller and matches the
turn contract used by the existing engines.

## Backup

Back up the leaf value through all selected edges:

```python
edge.N += 1
edge.W += value
edge.Q = edge.W / edge.N
```

No sign flip is needed because every decision node is Mr.X's turn and every
value is from Mr.X's perspective.

## Search Output

For evaluation:

```python
best_action = argmax_a N(root, a)
```

For training data:

```python
pi(a) = N(root, a) / sum_b N(root, b)
```

Store sparse policy targets:

```json
[
  {"flat": 628, "prob": 0.52},
  {"flat": 629, "prob": 0.31}
]
```

## First Implementation Files

Proposed files:

```text
neural_mrx_mcts_engine.py
validation/validate_neural_mrx_mcts.py
Notebook/MrX GNN Notebooks/kaggle_neural_mrx_mcts_validate.ipynb
```

Optional later:

```text
Notebook/MrX GNN Notebooks/kaggle_neural_mcts_logger.ipynb
```

## Validation Gate

First validate Neural MCTS as inference only.

Baseline matrix:

```text
MrX_PPO_V1 argmax vs detective_ppo_v001
NeuralMCTS(MrX_PPO_V1, 32 sims) vs detective_ppo_v001
NeuralMCTS(MrX_PPO_V1, 64 sims) vs detective_ppo_v001
MCTS_15_25 vs detective_ppo_v001
MCTS_30_50 vs detective_ppo_v001
```

Promotion as a teacher candidate only if:

- illegal actions = 0;
- winrate beats MrX_PPO_V1 argmax by at least 3 percentage points on a
  500+ game eval, or clearly improves average final turn at similar winrate;
- runtime is acceptable for teacher logging;
- no obvious ticket collapse, e.g. not all taxi unless game state forces it.

Current reference numbers:

| Engine | Opponent | Mr.X WR |
|---|---|---:|
| MrX_PPO_V1 argmax | detective_ppo_v001 | 44.7 % |
| MCTS `15/25` | detective_ppo_v001 | 18.7 % |
| MCTS `30/50` | detective_ppo_v001 | 34.0 % |

## Training Use After Validation

If Neural MCTS improves inference:

1. Use it as a teacher/logger.
2. Collect rows with:
   - current Mr.X input tensors;
   - legal mask;
   - selected action;
   - visit distribution;
   - terminal outcome/return.
3. Train Mr.X next version with:

   ```text
   policy loss = cross_entropy(student_logits, visit_distribution)
   value loss  = MSE(value, outcome/return)
   ```

4. Optionally continue PPO from that MCTS-improved checkpoint.

This is the AlphaZero-style policy improvement loop adapted to Scotland Yard.

## Known Risks

1. Value calibration can be poor.

   If value is poorly calibrated, search can amplify errors. Always compare
   against argmax policy and classical MCTS.

2. Runtime can be high.

   64 simulations means 64 repeated detective GNN phases per Mr.X move. Start
   with 16/32/64 simulation sweeps.

3. Tree state must include tracker state.

   `last_mrx_ticket` and `revealed_positions` are part of the model input.
   Forgetting them makes tree features differ from training/inference.

4. Do not mix string and integer node ids.

   `ADJ` is keyed by strings. Keep game state node ids as strings.

5. Do not make this the only opponent.

   Even if Neural MCTS is strong, self-play should still train against pools
   containing random, MCTS, BC, PPO, and historical checkpoints.
