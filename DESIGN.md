# Scotland Yard AI Design Document

This document is the source of truth for the AI architecture in this
repository. It describes the game loop, the engines that currently coexist,
the trained detective R-GNN, the warm-start Mr.X R-GNN, validation results,
and the intended path toward PPO/self-play.

Last updated: 2026-05-21.

---

## 1. Current State

The repository now supports two detective policy families and two Mr.X policy
families:

| Side | Engine | Status | Main file |
|---|---|---|---|
| Detectives | belief heuristic | Done, legacy baseline | `game.py`, `detective_engine.py` |
| Detectives | PPO R-GNN | Done, integrated | `gnn_detective_engine.py` |
| Mr.X | random | Done, debug baseline | `game.py` |
| Mr.X | MCTS | Done, current strong opponent | `mrx_engine.py`, `mcts_node.py` |
| Mr.X | BC/PPO R-GNN policy/value | Done, current neural best is MrX_PPO_V1 | `gnn_mrx_engine.py` |

Important operational answer:

```
python main.py
```

still runs the legacy setup by default:

```
detectives = heuristic
Mr.X       = MCTS
```

To run the trained GNN detectives:

```
python main.py --detectives gnn --mrx mcts --checkpoint Notebook/Models/detectives/detective_ppo_v001.pt
```

To run the current best Mr.X GNN:

```
python main.py --detectives gnn --mrx gnn \
  --checkpoint Notebook/Models/detectives/detective_ppo_v001.pt \
  --mrx-checkpoint Notebook/Models/mrx/mrx_ppo_v001.pt
```

The default remains heuristic for backward compatibility and for quick baseline
runs. The GNN engine is fully integrated, but it is opt-in through the CLI.

---

## 2. Repository Entry Points

### 2.1 Game core

- `game.py`
  - Owns mutable game state: Mr.X position, detective positions, tickets,
    turn counter, move history, and victory checks.
  - Node identifiers are strings (`"154"`, not `154`). This is a hard invariant
    because `ADJ` is keyed by string node ids.
  - Exposes:
    - `Game.find_legal_moves_x()`
    - `Game.x_random_turn()`
    - `Game.x_automated_turn(proposed_pos, ticket)`
    - `Game.detective_automated_turn(belief_state, detective_id)`
    - `Game.use_ticket(detective_id, origin, destination)`

- `detective_engine.py`
  - Maintains the detectives' belief distribution over Mr.X's position.
  - Updates belief after observed Mr.X tickets.
  - Removes detective-occupied nodes from possible Mr.X positions.
  - Handles reveal observations through `mrx_is_spotted`.

- `utility.py`
  - Contains simulation helpers used by MCTS rollouts.
  - Currently those rollouts use the legacy heuristic detectives, even when
    the real game loop uses GNN detectives. This is intentional for now:
    the existing MCTS model of detective behavior is heuristic-based.

### 2.2 Policy engines

- `mrx_engine.py`
  - MCTS engine for Mr.X.
  - Uses `MCTSNode` and short heuristic rollouts.
  - Tuned by module constants:
    - `NUM_EXPLORATIONS`
    - `NUM_SIMULATIONS`

- `gnn_mrx_engine.py`
  - Loads the warm-start behavioural-cloning R-GNN checkpoint for Mr.X.
  - Rebuilds the same features produced by the Mr.X BC logger.
  - Scores legal `(destination, ticket)` actions and applies argmax.

- `gnn_detective_engine.py`
  - Loads the trained dense R-GNN detective checkpoint.
  - Rebuilds the same features used during BC/PPO training.
  - Exposes `GNNDetectiveEngine.play_detective_turn(...)`.

- `validation/validate_gnn_detectives.py`
  - Runs paired validation:
    - same seed
    - same initial game setup
    - heuristic detectives vs GNN detectives
    - same Mr.X strength
  - Saves JSON validation summaries.

- `validation/validate_gnn_mrx.py`
  - Validates the Mr.X BC GNN against heuristic and frozen GNN detectives.
  - Compares against random and MCTS Mr.X baselines.
  - Saves JSON validation summaries.

- `validation/promotion_validate.py`
  - Convenience wrapper for future self-play notebooks.
  - Reads `Notebook/Registry/opponent_pools.json`.
  - Runs candidate and baseline on the same evaluation matrix and same seeds.
  - Writes one promotion JSON with suite deltas, hard-requirement checks and
    `summary.passed`.

- `main.py`
  - CLI runner for engine combinations.
  - Current supported strategies:
    - `--detectives heuristic`
    - `--detectives gnn`
    - `--mrx mcts`
    - `--mrx random`
    - `--mrx gnn`

Example commands:

```
python main.py --games 100 --detectives heuristic --mrx mcts
python main.py --games 100 --detectives gnn --mrx mcts --checkpoint Notebook/Models/detectives/detective_ppo_v001.pt
python main.py --games 20 --detectives gnn --mrx random
python main.py --games 20 --detectives gnn --mrx gnn --checkpoint Notebook/Models/detectives/detective_ppo_v001.pt --mrx-checkpoint Notebook/Models/mrx/mrx_ppo_v001.pt
```

---

## 3. Game Loop Contract

The game loop is:

1. Check terminal state.
2. Detectives act sequentially, id `0..4`.
3. After each detective move:
   - check capture;
   - update belief with `DetectiveEngine.kalman_filter()`.
4. Append detective positions to `game.detectives_moves`.
5. Mr.X acts.
6. Check terminal state.
7. Update belief from Mr.X's observed ticket.
8. If the turn is a reveal turn, set belief to the real Mr.X position.
9. Notify any neural detective policy of Mr.X observations.

Reveal schedule:

```
REVEAL_TURNS = (3, 8, 13, 18)
```

The code checks this as:

```
(game.turn - 3) % 5 == 0
```

Victory:

- Detectives win when `game.mrx_pos in game.detectives_pos`.
- Mr.X wins when `game.turn >= 22`.

Ticket constraints:

- Mr.X tickets: taxi, bus, underground, water.
- Detectives: taxi, bus, underground.
- Detectives cannot move onto another detective's occupied node.
- Detectives transfer used tickets to Mr.X via `Game.use_ticket`.

Hard invariant:

```
game.detectives_pos and game.mrx_pos must contain string node ids.
```

The earlier eval bug came from writing integer node ids into
`game.detectives_pos`. That produced errors like `ADJ[154]` because `ADJ`
expects `ADJ["154"]`.

---

## 4. Belief Heuristic Detectives

The legacy detective policy is implemented in:

```
Game.detective_automated_turn(belief_state, detective_id)
```

Behavior:

1. Compute legal reachable neighboring nodes for the acting detective.
2. Target `belief_state.argmax()`.
3. Choose the legal neighbor minimizing shortest-path distance to that target.

Strengths:

- Very fast.
- Good baseline.
- Simple and deterministic.

Weaknesses:

- Each detective independently chases the same belief argmax.
- No learned coordination.
- No long-term ticket economy.
- No anticipation of Mr.X's future options.

This policy remains important because:

- it is the baseline in validation;
- it is still used inside the current Mr.X MCTS rollout model;
- it was the teacher for Stage 1 behavioural cloning.

---

## 5. Detective R-GNN Engine

### 5.1 Goal

The detective R-GNN replaces the action-selection part of the heuristic while
still consuming the same observable belief state maintained by
`DetectiveEngine`.

It is a cooperative multi-agent policy:

- one shared network;
- queried once per detective action;
- acting detective is identified by:
  - an `is_ego` node feature;
  - a learned detective-id embedding.

Execution is decentralized and sequential:

```
for detective_id in 0..4:
    query same model with current game state
    apply chosen move
    update belief
```

The state seen by detective `i` includes moves already made by detectives
`0..i-1` in the same turn.

### 5.2 Checkpoint

Current selected checkpoint:

```
Notebook/Models/detectives/detective_ppo_v001.pt
```

Checkpoint metadata:

| Metric | Value |
|---|---:|
| `validation_score` | +99.0 |
| fixed GNN winrate (`15/25`) | 87 % |
| fixed heuristic winrate (`15/25`) | 24 % |
| fixed delta | +63 pp |
| hard GNN winrate (`30/50`) | 74 % |
| hard heuristic winrate (`30/50`) | 2 % |
| hard delta | +72 pp |
| PPO continuation update | 10 |

The checkpoint was produced by the Stage 2b continuation run on Kaggle and
uploaded into `Notebook/`.

### 5.3 Node features

Each of the 199 graph nodes has 19 scalar features.

| # | Feature | Meaning |
|---|---|---|
| 1 | `belief[v]` | Probability mass from `DetectiveEngine.belief_state` |
| 2-6 | `is_detective_k` | One-hot occupancy for each detective |
| 7 | `is_ego` | Acting detective position |
| 8 | `min_dist_to_detectives` | Minimum shortest-path distance to any detective |
| 9-13 | `dist_to_detective_k` | Per-detective shortest-path distances |
| 14-17 | relation degrees | taxi, bus, underground, water degree |
| 18 | `has_underground` | Node touches an underground edge |
| 19 | `mrx_visited` | Revealed Mr.X positions so far |

`mrx_visited` contains only revealed positions, not the hidden real trajectory.
Using the full trajectory would be information leakage.

### 5.4 Global features

The global feature vector has 26 scalars:

| Feature | Shape |
|---|---:|
| normalized turn number | 1 |
| turns to next reveal | 1 |
| Mr.X tickets | 4 |
| detective tickets | 15 |
| last Mr.X ticket one-hot | 5 |

The acting detective id is represented separately through an embedding.

### 5.5 Architecture

The model is a dense relational GCN:

- 199 fixed nodes;
- 4 relation types: taxi, bus, underground, water;
- row-normalized dense adjacency tensor `(R, N, N)`;
- 3 R-GCN layers;
- hidden size 64;
- residual connections where dimensions match;
- edge-conditioned policy head;
- graph-level value head.

The dense implementation is intentionally used instead of PyG because the graph
is fixed and tiny. Dense relation matmuls are faster and simpler here.

### 5.6 Action head

The GNN does not produce logits over all 199 nodes directly. Instead, for the
acting detective at node `v`, it scores only legal neighboring destinations:

```
logit(v -> u) = MLP([h_v, h_u, edge_type_embedding, global_features, detective_id_embedding])
```

Illegal actions are excluded by construction:

- no ticket for that edge type;
- destination occupied by another detective;
- non-detective vehicle type such as water.

At evaluation and in `main.py`, the policy uses argmax.

During PPO rollout, the policy sampled from the categorical distribution.

---

## 6. Training History

### 6.1 Stage 1: behavioural cloning

Notebook:

```
Notebook/colab_warmstart_logger.ipynb
Notebook/colab_warmstart_train.ipynb
```

Teacher:

```
Game.detective_automated_turn(...)
```

Dataset:

- generated from games against weakened MCTS Mr.X;
- Mr.X training distribution: `NUM_EXPLORATIONS=8`, `NUM_SIMULATIONS=8`;
- about 38k detective decisions in the documented 600-game run.

Targets:

- policy target: heuristic chosen destination;
- value target: cooperative return-to-go.

Reward:

| Component | Value |
|---|---:|
| capture | +10 |
| timeout | -10 |
| step penalty | -0.05 |
| distance shaping | `-0.1 * min_dist_to_real_mrx` |

The value target uses true Mr.X position during training only. The policy input
does not receive true hidden Mr.X position at inference.

### 6.2 Stage 1 evaluation

Notebook:

```
Notebook/colab_warmstart_eval.ipynb
```

Result:

| Mr.X strength | GNN improvement over heuristic |
|---|---:|
| weakened (`8/8`) | +10 pp |
| full (`15/25`) | +44 pp |

This validated that the GNN learned more than just copying the belief argmax.

### 6.3 Stage 2: PPO

Notebook:

```
Notebook/colab_ppo.ipynb
```

Warm start:

```
Notebook/Models/detectives/detective_bc_v001.pt
```

PPO setup:

- opponent: full MCTS Mr.X (`15/25`);
- rollout policy: categorical sampling;
- eval policy: argmax;
- discount: `gamma=0.95`;
- GAE lambda: `0.95`;
- clip epsilon: `0.2`;
- entropy coefficient: `0.01`;
- value coefficient: `0.5`.

Reward attribution:

The team reward for a game turn is split across detective decisions collected in
that turn. This keeps the summed reward per game turn close to the BC
return-to-go scale while still assigning credit to sequential detective actions.

Initial PPO result:

- BC baseline: +44 pp over heuristic;
- PPO best: +66 pp over heuristic;
- later updates drifted down, so the best intermediate checkpoint was used.

### 6.4 Stage 2b: PPO continuation

Notebook:

```
Notebook/colab_ppo_continue.ipynb
```

Purpose:

- start from the best PPO checkpoint;
- reduce PPO drift;
- randomize Mr.X strength;
- validate on both fixed and hard suites.

Continuation settings:

| Hyperparameter | Value |
|---|---:|
| learning rate | `1e-4` |
| PPO epochs | `2` |
| games per update | `16` |
| entropy coefficient | `0.005` |

Rollout Mr.X pool:

| Mr.X strength | Weight |
|---|---:|
| `8/8` | 0.15 |
| `15/25` | 0.45 |
| `25/40` | 0.25 |
| `30/50` | 0.15 |

Validation suites:

- fixed: `15/25`;
- hard: `30/50`.

Final selected checkpoint:

```
Notebook/Models/detectives/detective_ppo_v001.pt
```

---

## 7. Validation

### 7.1 Existing final checkpoint validation

The final checkpoint metadata records:

| Suite | GNN winrate | Heuristic winrate | Delta |
|---|---:|---:|---:|
| fixed `15/25` | 87 % | 24 % | +63 pp |
| hard `30/50` | 74 % | 2 % | +72 pp |

These are the primary current validation numbers.

### 7.2 Repo validation script

Use:

```
python validation/validate_gnn_detectives.py \
  --checkpoint Notebook/Models/detectives/detective_ppo_v001.pt \
  --suite fixed:15:25:500 \
  --suite hard:30:50:300 \
  --suite very_hard:50:80:100 \
  --output final_gnn_validation.json \
  --quiet
```

The script runs paired comparisons:

- same seed;
- same initial positions;
- heuristic detectives vs GNN detectives;
- same Mr.X strength.

Local smoke validation passed on 2026-05-20:

| Suite | Games | GNN winrate | Heuristic winrate | Delta |
|---|---:|---:|---:|---:|
| smoke fixed `15/25` | 3 | 100 % | 66.7 % | +33.3 pp |
| smoke hard `30/50` | 1 | 100 % | 0 % | +100 pp |

This smoke test verifies integration, not statistical strength. Use the longer
command above for a final publishable estimate.

---

## 8. Engine Coexistence Plan

The repository should keep engines separate and composable.

### 8.1 Detective side

Current:

- `DetectiveEngine`
  - observation/belief engine;
  - not a policy by itself except that the heuristic consumes its belief.

- heuristic detective policy
  - currently a method on `Game`;
  - should eventually be wrapped in a `HeuristicDetectivePolicy` class for
    symmetry, but it is stable enough for now.

- `GNNDetectiveEngine`
  - learned policy;
  - consumes `DetectiveEngine.belief_state`;
  - tracks revealed Mr.X positions and last Mr.X ticket;
  - chooses one detective action at a time.

Future cleanup:

```
detective_policies/
  heuristic.py
  gnn.py
```

or:

```
engines/
  belief.py
  detective_heuristic.py
  detective_gnn.py
  mrx_mcts.py
  mrx_gnn.py
```

Do this when adding Mr.X neural code, so the abstraction is informed by both
sides' needs.

### 8.2 Mr.X side

Current:

- random Mr.X:
  - debug baseline;
  - implemented in `Game.x_random_turn()`.

- MCTS Mr.X:
  - current strong opponent;
  - implemented in `MrxEngine`;
  - still models future detective behavior with heuristic rollouts.

Future:

- GNN/NN Mr.X:
  - should be implemented as a separate engine, not as a replacement for MCTS;
  - MCTS remains teacher, baseline, and evaluation opponent.

The intended final matrix is:

| Detectives | Mr.X | Use case |
|---|---|---|
| heuristic | random | debug |
| heuristic | MCTS | legacy baseline |
| GNN | MCTS | current final detective benchmark |
| heuristic | GNN Mr.X | measure Mr.X NN imitation strength |
| GNN | GNN Mr.X | self-play experiments |
| GNN | checkpoint pool Mr.X | robust self-play |

---

## 9. Known Pitfalls

1. Node ids must stay strings in `Game`.

   Good:

   ```
   game.detectives_pos[i] = "154"
   ```

   Bad:

   ```
   game.detectives_pos[i] = 154
   ```

2. Do not give detective GNN true hidden Mr.X position.

   Allowed:

   - belief state;
   - observed last ticket;
   - reveal history.

   Not allowed:

   - full `game.mrx_moves`;
   - `game.mrx_pos` except when computing training reward/labels.

3. Mr.X MCTS rollouts still use heuristic detectives.

   This means that in GNN-vs-MCTS games, Mr.X may be planning against a weaker
   internal model than the real GNN detectives. This is acceptable for current
   benchmarking, but a future stronger Mr.X should either:

   - roll out against GNN detectives;
   - learn a neural policy/value from MCTS;
   - use self-play checkpoint pools.

4. Checkpoint file names are timestamps, not scores.

   To compare checkpoints, read checkpoint metadata:

   - `validation_score`;
   - `fixed_eval`;
   - `hard_eval`;
   - `delta_pp`.

5. Small validation runs are noisy.

   A 50-game eval can move by several percentage points. Use hundreds of paired
   games before declaring a new final model.

---

## 10. Mr.X Neural Policy

The Mr.X neural path has started with behavioural cloning from MCTS. The
recommended path remains staged: BC first, then PPO against frozen detectives,
then alternating self-play only after both sides are stable.

### 10.1 Stage X1: Mr.X behavioural cloning from MCTS

Goal:

Train a Mr.X policy/value network to imitate `MrxEngine.search()`.

Teacher:

```
MrxEngine.search()
```

Students:

- policy head: choose legal destination/ticket action;
- value head: estimate escape strength / future return.

Why BC first:

- MCTS is already strong;
- BC gives the neural Mr.X a stable starting point;
- PPO from scratch against strong detectives would be unstable and sample
  inefficient.

Current BC checkpoint:

```
Notebook/Models/mrx/mrx_bc_v001.pt
```

Current PPO checkpoint:

```
Notebook/Models/mrx/mrx_ppo_v001.pt
```

Training data:

- logger output: 5,000 games;
- decisions: 47,382 Mr.X actions;
- target mask correctness: 100%;
- teacher mix: `8/8`, `15/25`, `30/50`, `50/80`.

Warm-start metrics:

| Metric | Value |
|---|---:|
| validation loss | 1.4132 |
| exact action accuracy | 43.1 % |
| top-3 action accuracy | 85.8 % |
| ticket accuracy | 84.5 % |

The exact action accuracy is only a hard-target imitation metric. Many MCTS
moves are near-equivalent, so game-level validation is required before judging
whether the checkpoint is strong enough for PPO.

BC game-level validation against frozen detective GNN:

| Opponent setup | Mr.X winrate | Avg final turn | Illegal actions |
|---|---:|---:|---:|
| random Mr.X vs heuristic detectives | 0.0 % | 5.0 | 0 |
| BC Mr.X vs heuristic detectives | 47.5 % | 15.4 | 0 |
| BC Mr.X vs GNN detectives | 42.0 % | 13.6 | 0 |
| MCTS `15/25` Mr.X vs GNN detectives | 12.0 % | 8.7 | 0 |

### 10.2 Mr.X observation model

Unlike detectives, Mr.X knows:

- its true position;
- its tickets;
- detective positions;
- detective ticket counts;
- turn and reveal schedule;
- its own hidden/revealed move history.

Mr.X also knows what the detectives can observe:

- last ticket used;
- reveal positions;
- likely belief distribution.

Candidate node features:

| Feature | Meaning |
|---|---|
| `is_mrx` | true Mr.X position |
| `detective_one_hot_k` | detective positions |
| `belief[v]` | current detective belief over Mr.X |
| `is_reveal_node` | revealed Mr.X positions so far |
| `dist_to_detective_k` | shortest-path threat distance |
| `min_dist_to_detectives` | immediate danger |
| relation degrees | mobility by transport |
| ticket-reachable masks | optional legal-action context |

Candidate global features:

| Feature | Meaning |
|---|---|
| turn | current turn |
| turns to reveal | pressure to hide before surfacing |
| Mr.X tickets | own resources |
| detective tickets | opponent resources |
| last ticket | public observation emitted last turn |
| reveal flag | whether current/next turn is reveal-sensitive |

### 10.3 Mr.X action space

Action should be edge-conditioned, like detectives, but with ticket-specific
actions:

```
(destination, ticket)
```

This matters because multiple transport types can connect the same two nodes,
and the ticket choice changes what detectives observe.

Legal actions:

- destination adjacent to current Mr.X position;
- Mr.X has the corresponding ticket;
- water allowed for Mr.X if graph edge type is water;
- moving onto a detective is legal in game code but immediately loses, so it
  should be labelled as terminal/bad during training rather than silently
  removed unless a deliberate no-suicide mask is chosen.

### 10.4 Mr.X logger design

The implemented logger collects one row per Mr.X decision:

Parquet row:

- `game_id`;
- `turn`;
- `mrx_pos`;
- `detectives_pos`;
- `mrx_tickets`;
- `detective_tickets`;
- `belief_state`;
- `last_mrx_ticket`;
- `revealed_positions`;
- `legal_actions`;
- MCTS chosen `(destination, ticket)`;
- MCTS diagnostics if available:
  - visits per child;
  - mean score per child;
  - selected child score;
- game outcome;
- return-to-go.

Tensor arrays:

- node features;
- global features;
- legal action masks;
- target action index;
- optional target policy distribution from MCTS visits.

The best target is not just the selected action. If available, use the MCTS
visit distribution as a soft policy target, AlphaZero-style:

```
cross_entropy(student_policy, mcts_visit_distribution)
```

The current BC warm start trains on the hard selected MCTS action. The logger
also stores sparse visit distributions where available, so a later training
iteration can add a soft policy target.

### 10.5 Stage X2: Mr.X PPO

After BC:

1. Freeze detective PPO checkpoint.
2. Train Mr.X NN with PPO against fixed GNN detectives.
3. Validate Mr.X NN against:
   - heuristic detectives;
   - GNN detectives;
   - MCTS Mr.X baseline.

Reward should be Mr.X-centric:

| Event | Reward |
|---|---:|
| survive to timeout | positive |
| capture | large negative |
| each turn survived | small positive |
| distance from nearest detective | shaping positive |
| reveal danger | optional penalty near reveal turns |

Current PPO result:

| Metric | Value |
|---|---:|
| PPO best update | 18 |
| eval games | 100 |
| Mr.X winrate vs frozen GNN detectives | 54.0 % |
| average final turn | 16.54 |
| illegal actions | 0 |
| validation score | 70.54 |

Large validation:

| Suite | Games | Mr.X winrate | Avg final turn | Illegal actions |
|---|---:|---:|---:|---:|
| PPO Mr.X vs heuristic detectives | 1,000 | 51.0 % | 16.6 | 0 |
| PPO Mr.X vs GNN detectives `132159` | 1,000 | 44.7 % | 14.9 | 0 |
| MCTS `15/25` vs GNN detectives `132159` | 300 | 18.7 % | 9.5 | 0 |
| MCTS `30/50` vs GNN detectives `132159` | 150 | 34.0 % | 12.6 | 0 |

Canonical eval file:

```
Notebook/Eval_MrX_V1/mrx_ppo_validation_large.json
```

The large validation number, not the 100-game internal eval, should be treated
as the publishable MrX_PPO_V1 strength.

### 10.6 Stage X3: alternating self-play

Only after both sides have stable neural policies:

1. Freeze Mr.X, train detectives.
2. Freeze detectives, train Mr.X.
3. Evaluate against a pool of historical checkpoints.

Do not train both networks freely at the same time at first. That usually
creates unstable exploit cycles.

Recommended pool:

```
detective_pool = [heuristic, BC, PPO_v1, PPO_continue_best]
mrx_pool = [random, MCTS_8_8, MCTS_15_25, MCTS_30_50, MrX_BC, MrX_PPO]
```

Self-play training should sample opponents from the pool, not only from the
latest checkpoint.

### 10.7 League registry

The project now uses an explicit local registry under:

```
Notebook/Registry/
```

Promoted checkpoint files live under:

```
Notebook/Models/detectives/
Notebook/Models/mrx/
```

Files:

| File | Purpose |
|---|---|
| `model_registry.json` | All checkpoint ids, paths, lineage, status, metrics, eval files |
| `best_detective.json` | Current best detective alias |
| `best_mrx.json` | Current best Mr.X alias |
| `opponent_pools.json` | Training pools, eval matrices, promotion gates |
| `README.md` | Registry workflow |

Current model ids:

| Id | Side | Kind | Status | Path |
|---|---|---|---|---|
| `detective_bc_v001` | detectives | BC | historical | `Notebook/Models/detectives/detective_bc_v001.pt` |
| `detective_ppo_v001` | detectives | PPO | best | `Notebook/Models/detectives/detective_ppo_v001.pt` |
| `mrx_bc_v001` | Mr.X | BC | historical | `Notebook/Models/mrx/mrx_bc_v001.pt` |
| `mrx_ppo_v001` | Mr.X | PPO | best | `Notebook/Models/mrx/mrx_ppo_v001.pt` |

Virtual baseline ids:

| Id | Meaning |
|---|---|
| `detective_heuristic` | `Game.detective_automated_turn` |
| `mrx_random` | `Game.x_random_turn` |
| `mrx_mcts_8_8` | MCTS with 8 explorations, 8 simulations |
| `mrx_mcts_15_25` | MCTS with 15 explorations, 25 simulations |
| `mrx_mcts_25_40` | MCTS with 25 explorations, 40 simulations |
| `mrx_mcts_30_50` | MCTS with 30 explorations, 50 simulations |
| `mrx_mcts_50_80` | MCTS with 50 explorations, 80 simulations |

Promotion rule:

- never overwrite an existing promoted checkpoint;
- copy accepted candidates to the next versioned filename, for example
  `Notebook/Models/mrx/mrx_ppo_v002.pt`;
- update `model_registry.json` with lineage, metrics and eval files;
- update `best_mrx.json` or `best_detective.json` only after the promotion
  gate passes;
- keep historical promoted checkpoints because they become useful league
  opponents and regression anchors.

Useful registry commands:

```
python model_registry.py --verify-paths
python model_registry.py --next-id --side mrx --kind ppo
python model_registry.py --next-id --side detectives --kind ppo
```

### 10.8 Two self-play notebooks

The intended repeated workflow is two dedicated notebooks:

```
Notebook/Detective GNN Notebooks/kaggle_detective_selfplay.ipynb
Notebook/MrX GNN Notebooks/kaggle_mrx_selfplay.ipynb
```

They should be symmetric in structure.

Each notebook must:

1. Load `Notebook/Registry/model_registry.json`.
2. Load the side's current best from `best_detective.json` or `best_mrx.json`.
3. Build the opponent pool from `opponent_pools.json`.
4. Run an initial evaluation matrix for the current best.
5. Train a candidate checkpoint from the current best.
6. Run periodic eval during training.
7. Run a final large evaluation matrix.
8. Save:
   - candidate checkpoint;
   - history CSV;
   - curves PNG;
   - final eval JSON;
   - `registry_candidate_update.json`.
9. Never overwrite `best_detective.json` or `best_mrx.json` inside Kaggle.
   Kaggle inputs are read-only and promotion should be reviewed locally.

Detective self-play notebook:

```
warm start = best_detective.json
frozen opponent pool = detective_training_mrx_pool_v1
candidate id prefix = detective_ppo_v002
```

Initial opponent pool:

| Opponent | Weight |
|---|---:|
| `mrx_random` | 0.05 |
| `mrx_mcts_8_8` | 0.10 |
| `mrx_mcts_15_25` | 0.20 |
| `mrx_mcts_30_50` | 0.20 |
| `mrx_bc_v001` | 0.20 |
| `mrx_ppo_v001` | 0.25 |

Mr.X self-play notebook:

```
warm start = best_mrx.json
frozen opponent pool = mrx_training_detective_pool_v1
candidate id prefix = mrx_ppo_v002
```

Initial opponent pool before Detective_PPO_V2 exists:

| Opponent | Weight |
|---|---:|
| `detective_heuristic` | 0.15 |
| `detective_bc_v001` | 0.15 |
| `detective_ppo_v001` | 0.70 |

After Detective_PPO_V2 is promoted, update the pool:

| Opponent | Weight |
|---|---:|
| `detective_heuristic` | 0.10 |
| `detective_bc_v001` | 0.10 |
| `detective_ppo_v001` | 0.30 |
| `detective_ppo_v002` | 0.50 |

### 10.9 Promotion gates

A model is promoted only if it improves the evaluation matrix and does not
collapse on baselines.

Detective promotion gate:

- primary metric: mean detective winrate over the eval matrix;
- must improve over `detective_ppo_v001`;
- target improvement: at least +3 percentage points;
- max regression on any core suite: 5 percentage points;
- core suites:
  - `mrx_mcts_15_25`;
  - `mrx_mcts_30_50`;
  - `mrx_ppo_v001`;
- hard requirements:
  - illegal actions = 0;
  - detective winrate vs `mrx_random` >= 98%.

Mr.X promotion gate:

- primary metric: mean Mr.X winrate over the eval matrix;
- must improve over `mrx_ppo_v001`;
- target improvement: at least +3 percentage points;
- max regression on any core suite: 5 percentage points;
- core suites:
  - `detective_ppo_v001`;
  - latest promoted detective checkpoint when available;
- hard requirements:
  - illegal actions = 0;
  - Mr.X winrate vs `detective_heuristic` >= 45%.

Promotion should create a new version id, not overwrite old ids:

```
detective_ppo_v002
mrx_ppo_v002
```

### 10.10 Promotion validation wrapper

The low-level validators remain useful:

```
validation/validate_gnn_detectives.py
validation/validate_gnn_mrx.py
```

Use them when manually probing one matchup or publishing a specific diagnostic.
For actual promotion decisions, use:

```
validation/promotion_validate.py
```

This wrapper is the preferred entry point for future Kaggle self-play notebooks.
It reads the registry pool, compares the candidate against the current baseline
on the same suite matrix, applies the promotion gate, and saves a single JSON.

Mr.X candidate example:

```
python validation/promotion_validate.py \
  --side mrx \
  --candidate-id mrx_ppo_v002 \
  --candidate-checkpoint Notebook/Models/mrx/mrx_ppo_v002.pt \
  --device cpu \
  --output Notebook/Eval_MrX_V2/mrx_ppo_v002_promotion.json \
  --quiet
```

Detective candidate example:

```
python validation/promotion_validate.py \
  --side detectives \
  --candidate-id detective_ppo_v002 \
  --candidate-checkpoint Notebook/Models/detectives/detective_ppo_v002.pt \
  --device cpu \
  --output Notebook/Eval_detective_V2/detective_ppo_v002_promotion.json \
  --quiet
```

Notebook smoke-test mode:

```
python validation/promotion_validate.py \
  --side mrx \
  --candidate-checkpoint /kaggle/working/mrx_ppo_checkpoints/candidate.pt \
  --candidate-id mrx_ppo_v002_candidate \
  --device cpu \
  --max-games-per-suite 2 \
  --output /kaggle/working/mrx_promotion_smoke.json \
  --quiet
```

Plan-only mode:

```
python validation/promotion_validate.py \
  --side detectives \
  --candidate-checkpoint /kaggle/working/detective_candidate.pt \
  --candidate-id detective_ppo_v002_candidate \
  --dry-run
```

Interpretation:

- `summary.passed = true` means the candidate may be promoted locally.
- `summary.passed = false` means keep it as a Kaggle artifact or archive it,
  but do not update `best_mrx.json` or `best_detective.json`.
- `summary.improvement_pp` is candidate mean matrix score minus baseline mean
  matrix score.
- Each suite stores candidate score, baseline score and `delta_pp`.

For Mr.X, higher score means higher Mr.X winrate.
For detectives, higher score means higher detective winrate, computed as
`1 - mrx_winrate` when the underlying game runner reports from Mr.X's side.

### 10.11 Anti-overfitting rule

Do not train only against the newest strongest opponent. Always include:

- latest best opponent;
- historical neural checkpoints;
- MCTS baselines;
- noob/random baseline with small weight.

This follows the same practical idea used by AlphaGo/AlphaZero-style systems:
use self-play to improve, but use gating and historical opponents to avoid
fragile exploit cycles. For this project, the league pool is deliberately
simpler than AlphaStar/OpenAI Five, but it serves the same purpose.

### 10.12 Neural MCTS branch

Neural MCTS is a separate experimental branch, documented in:

```
NEURAL_MCTS_DESIGN.md
```

First target:

```
MrX_PPO_V1 -> NeuralMCTS inference -> validation vs detective_ppo_v001
```

Do not use Neural MCTS as the only training opponent. If it validates stronger
than MrX_PPO_V1 argmax, it can become a teacher/logger for soft visit-policy
targets, AlphaZero-style.

### 10.13 Automated league cycle

The repo now has a restartable league workflow under:

```
league/
Notebook/League Notebooks/
```

The cycle is split into three dedicated stages:

1. Neural MCTS teacher logging.

   ```
   league/neural_mcts_logger.py
   Notebook/League Notebooks/01_kaggle_log_neural_mcts_teacher.ipynb
   ```

   This reads the current best Mr.X and detective checkpoints from the registry,
   runs the accepted Neural MCTS teacher, and writes:

   - `samples_part_*.parquet` or `.pkl`;
   - `tensors_part_*.npz`;
   - dense soft visit-policy targets;
   - `manifest.json`;
   - `game_stats`.

2. Mr.X supervised learning from Neural MCTS logs.

   ```
   league/train_mrx_sl.py
   Notebook/League Notebooks/02_kaggle_train_mrx_sl_from_neural_mcts.ipynb
   ```

   This warm-starts from the current best Mr.X checkpoint, trains on the latest
   Neural MCTS logs, and emits a candidate such as `mrx_sl_v001`. It also
   writes `registry_candidate_update.json`.

3. Detective PPO against the latest promoted Mr.X.

   ```
   league/train_detective_rl_vs_latest_mrx.py
   Notebook/League Notebooks/03_kaggle_train_detectives_rl_vs_latest_mrx.ipynb
   ```

   This starts from the current best detective checkpoint and trains against
   the current best Mr.X alias. If the candidate improves enough during eval,
   training can stop early and write a detective PPO candidate update.

Restart behavior:

- each stage reads `best_mrx.json` and/or `best_detective.json`;
- promoted checkpoints become the starting point for the next run;
- failed candidates do not become automatic opponents;
- rerunning a notebook repeats the correct next cycle after the registry is
  updated locally.

---

## 11. Mr.X Logging Status

Logging and first BC training are done. The logger notebook is:

```
Notebook/MrX GNN Notebooks/colab_mrx_bc_logger.ipynb
```

The first training notebook is:

```
Notebook/MrX GNN Notebooks/kaggle_mrx_warmstart_train.ipynb
```

The validation entry points are:

```
validation/validate_gnn_mrx.py
validation/promotion_validate.py
Notebook/MrX GNN Notebooks/kaggle_mrx_bc_validate.ipynb
```

Use `validation/validate_gnn_mrx.py` for manual matchup diagnostics. Use
`validation/promotion_validate.py` for registry-based large promotion gates.

The PPO continuation notebook is:

```
Notebook/MrX GNN Notebooks/kaggle_mrx_ppo.ipynb
```

The logger has answered:

1. Can we capture all legal `(destination, ticket)` actions?
2. Can we expose MCTS child scores/visits?
3. Does the logged action always belong to the legal mask?
4. Can we reconstruct the same features offline?
5. Is the dataset balanced, or does MCTS almost always win/lose against the
   selected detective opponent?

Recommended first logging opponent:

```
detectives = GNN PPO final checkpoint
Mr.X teacher = MCTS with randomized budgets
```

Suggested MCTS teacher mix:

| Budget | Purpose |
|---|---|
| `8/8` | cheaper, diverse weaker teacher |
| `15/25` | current standard |
| `30/50` | stronger target |
| `50/80` | expensive hard target for smaller sample |

- legal mask correctness: passed;
- target action coverage: passed;
- first serious dataset: 5,000 games;
- first BC model: trained and integrated in `gnn_mrx_engine.py`;
- first PPO model: trained and auto-discovered by `GNNMrXEngine`.

---

## 12. Immediate TODO

1. Use the registry and promotion wrapper as the common notebook interface.

   Implemented files:

   ```
   model_registry.py
   validation/promotion_validate.py
   ```

   Future notebooks should:

   - read model paths and pools from `Notebook/Registry/`;
   - train a candidate checkpoint;
   - call `validation/promotion_validate.py --dry-run` before the expensive eval;
   - call `validation/promotion_validate.py` for the final large gate;
   - write `registry_candidate_update.json` only after the candidate has a
     promotion JSON.

2. Build the detective self-play notebook first.

   Suggested file:

   ```
   Notebook/Detective GNN Notebooks/kaggle_detective_selfplay.ipynb
   ```

   Goal:

   - warm-start from `detective_ppo_v001`;
   - train candidate `detective_ppo_v002`;
   - opponent pool = `detective_training_mrx_pool_v1`;
   - freeze all Mr.X opponents;
   - output checkpoint, history, eval matrix, candidate registry update.

3. Promote `detective_ppo_v002` only if the promotion gate passes.

   After promotion:

   - update `model_registry.json`;
   - update `best_detective.json`;
   - add `detective_ppo_v002` to `mrx_training_detective_pool_v1`.

4. Build the Mr.X self-play notebook second.

   Suggested file:

   ```
   Notebook/MrX GNN Notebooks/kaggle_mrx_selfplay.ipynb
   ```

   Goal:

   - warm-start from `mrx_ppo_v001`;
   - train candidate `mrx_ppo_v002`;
   - opponent pool = `mrx_training_detective_pool_v1`;
   - freeze all detective opponents;
   - output checkpoint, history, eval matrix, candidate registry update.

5. Implement Neural MCTS as a separate experimental branch only after the
   league registry workflow is stable.

   Suggested files:

   ```
   neural_mrx_mcts_engine.py
   validation/validate_neural_mrx_mcts.py
   Notebook/MrX GNN Notebooks/kaggle_neural_mrx_mcts_validate.ipynb
   ```

   Specification:

   ```
   NEURAL_MCTS_DESIGN.md
   ```

6. Optional cleanup: wrap the heuristic detective policy in its own class.
