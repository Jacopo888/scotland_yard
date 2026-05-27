# Scotland Yard AI

A Python implementation of the **Scotland Yard** board game, featuring fully automated AI players for both Mr. X and the detectives.

By default, the project still runs the original baseline: **Mr. X MCTS** against **belief-state detectives**. The codebase now also includes trained **R-GNN policy/value engines**, model validation, promotion gates, Kaggle league automation, and AlphaGo-style search teachers for both Mr. X and the detectives.

## Game Rules

- **Board**: 199 stations connected by taxi, bus, underground (and water routes for Mr. X)
- **Players**: 1 Mr. X vs 5 detectives
- **Turns**: detectives move first, then Mr. X. The game lasts up to 22 turns
- **Detective tickets**: taxi (10), bus (8), underground (4)
- **Mr. X tickets**: taxi (4), bus (3), underground (3), water (5)
- **Visibility**: Mr. X is hidden, but his position is revealed every 5 turns. Detectives only see the ticket type he uses
- **Detectives win**: by landing on Mr. X's position
- **Mr. X wins**: by surviving until turn 22

## Current State

The project has grown from a playable AI baseline into a small neural self-play lab:

- **Legacy baseline**: Mr. X MCTS vs belief-state/Kalman detectives remains the default and the main regression baseline
- **Neural policies**: `gnn_mrx_engine.py` and `gnn_detective_engine.py` load trained relational GNN policy/value checkpoints
- **Current best models**: tracked in `Notebook/Registry` (`mrx_sl_v003` for Mr. X, `detective_ppo_v004` for detectives)
- **League loop**: `league/` logs teacher games, trains candidates, validates them, and promotes only through registry gates
- **Next step implemented**: AlphaGo-style GNN + MCTS teachers for Mr. X and detectives, used to generate stronger soft policy targets

## How the AI Works

### Mr. X — Monte Carlo Tree Search

The Mr. X engine (`mrx_engine.py`) builds a tree of possible moves and evaluates them through simulation:

1. **Selection**: picks the most promising node using UCB1 (`score + 1.42 * sqrt(ln(N) / n)`)
2. **Expansion**: generates legal moves from the selected node
3. **Simulation (rollout)**: runs N simulations per node to evaluate each candidate move
4. **Backpropagation**: updates scores back up the tree

Parameters: 15 expansion iterations, 25 simulations per rollout.

#### MCTS Rollout Value Function

Two approaches were implemented and tested for evaluating positions during MCTS rollouts:

1. **Distance-based evaluation (active)**: simulates 5 turns from the candidate position, then returns the shortest-path distance between Mr. X and the nearest detective. Higher distance = better position for Mr. X. This acts as a heuristic that rewards moves leading to positions where Mr. X is far from all detectives.

2. **Win/loss simulation**: simulates the game to completion from the candidate position and returns 1 if Mr. X wins, 0 if the detectives win. This gives a binary outcome based on full game playouts.

The **distance-based evaluation** is currently used because it produced better results in practice. The shorter simulation horizon (5 turns vs full game) provides a more reliable signal, as full-game random rollouts introduce too much noise to effectively distinguish between moves.

This engine is still useful as a baseline, opponent, and teacher, even though the strongest current models are neural.

### Detectives — Belief State + Kalman Filter

The detective engine (`detective_engine.py`) maintains a probability distribution over all 199 stations:

- **Update With a Markov Chain Model**: after each Mr. X move, the belief state is multiplied by the transition matrix of the vehicle used
- **Kalman filter**: zeroes out probability at detective positions and renormalizes
- **Spotting**: when Mr. X is spotted (every 5 turns), the belief becomes 100% at his real position

Each detective moves toward the station with the highest probability, using a precomputed shortest-path tensor.

### Neural R-GNN Policies

The neural engines are opt-in through `main.py`:

- `gnn_detective_engine.py`: a dense relational GNN that reads belief state, detective positions, tickets, reveal history, and legal moves; it outputs a policy over legal detective destinations plus a value estimate
- `gnn_mrx_engine.py`: a relational GNN for Mr. X that scores legal `(destination, ticket)` actions and predicts value from Mr. X's perspective
- `model_registry.py` and `Notebook/Registry/`: keep stable model IDs, current best aliases, lineage, metrics, and opponent pools

### AlphaGo-Style GNN Search

The latest step is search-improved neural training, inspired by AlphaGo/AlphaZero:

- **Mr. X**: `neural_mrx_mcts_engine.py` uses the Mr. X GNN as policy prior and value head, runs PUCT search, and simulates frozen GNN detectives between Mr. X decision nodes
- **Detectives**: `league/detective_belief_mcts.py` uses belief-sampled MCTS, including a joint mode that searches coordinated moves for all five detectives
- **Training data**: `league/neural_mcts_logger.py` and `league/detective_mcts_logger.py` save visit distributions as soft policy targets
- **Student updates**: `league/train_mrx_sl.py` and `league/train_detective_sl.py` train the next GNN candidates from those search targets

In short: the GNNs do fast policy/value inference; MCTS improves decisions; the visit distributions become better training labels for the next model.

### Belief State Visualizer

The belief state visualizer (`belief_state_visualizer.py`) opens a second Tkinter window that displays a real-time heatmap of where the detectives think Mr. X might be.

- **Heatmap**: each of the 199 stations is drawn as a circle whose color and size reflect the current probability of Mr. X being there. The gradient goes from black (probability = 0) to bright red (highest probability).
- **Labels**: stations with probability above 10% of the maximum show their node number and numeric probability; the rest show only a dimmed node number.
- **Color bar**: a legend on the right side maps the color gradient to actual probability values (0 -> max P(Mr. X)).
- **Live updates**: the visualizer refreshes after every detective turn, every Mr. X move, and every spotting event (turns 5, 10, 15, 20), so you can watch the belief state sharpen and spread in real time.

## Visual Mode

To visualize the board and the belief state heatmap during gameplay, edit `main.py`:

1. In the `if __name__ == "__main__"` block, replace `play()` with `play_visual()`
2. The `play_visual()` function will open two Tkinter windows: the game board and the belief state heatmap

```python
if __name__ == "__main__":
    play_visual()
```

## AI usage
I used Claude to help implement the visual components (board visualizer and belief state representation).

## Project Structure

```
scotland_yard/
├── main.py                     # Entry point — runs the games
├── game.py                     # Game state, moves, rules
├── detective_engine.py         # Detective AI (belief state + Kalman)
├── gnn_detective_engine.py     # Detective R-GNN policy/value engine
├── mrx_engine.py               # Mr. X AI (MCTS)
├── gnn_mrx_engine.py           # Mr. X R-GNN policy/value engine
├── neural_mrx_mcts_engine.py   # AlphaGo-style Mr. X neural MCTS
├── mcts_node.py                # MCTS tree node
├── model_registry.py           # Registry helpers for promoted models
├── utility.py                  # Turn simulation helpers
├── board_generation.py         # Board visualization (Tkinter)
├── belief_state_visualizer.py  # Belief state heatmap
├── league/                     # Self-play logging, training, Kaggle cycle
├── validation/                 # GNN validation and promotion gates
├── Notebook/Registry/          # Current bests, lineage, opponent pools
├── Notebook/Models/            # Promoted neural checkpoints
└── Matrix_generation/
    ├── connections.txt                # Graph: node1 node2 vehicle
    ├── board_graph.pkl                # Serialized NetworkX graph
    ├── taxi_matrix.npy                # Taxi transition matrix (199x199)
    ├── bus_matrix.npy                 # Bus transition matrix
    ├── underground_matrix.npy         # Underground transition matrix
    ├── unknown_matrix.npy             # Water/unknown transition matrix
    ├── distanze_scotland_yard_3d.npy  # Shortest path tensor (8x199x199)
    ├── Gen_board.py                   # Generates board_graph.pkl
    ├── Stochastic_matrix_Gen.py       # Generates transition matrices
    └── shortest_path-matrix.py        # Generates shortest path tensor
```

## Requirements

- Python 3.8+
- [NetworkX](https://networkx.org/)
- [NumPy](https://numpy.org/)
- [PyTorch](https://pytorch.org/) for GNN engines
- [pandas](https://pandas.pydata.org/) for league logging/training

```bash
pip install networkx numpy torch pandas
```

## Running

```bash
python main.py
```

Runs 100 automated games by default (Mr. X MCTS vs belief-state detectives) and prints:
- Total execution time
- Number of Mr. X and detective wins

Useful neural matchups:

```bash
python main.py --games 100 --detectives gnn --mrx gnn
python main.py --games 20 --detectives gnn --mrx neural_mcts --neural-mcts-simulations 32
```

If no checkpoint is passed, the GNN engines resolve the current best model from `Notebook/Registry`.

## Training and Validation

Neural training is handled outside the main game loop:

- `league/kaggle_cycle.py`: orchestrates logging, training, validation, and promotion stages
- `validation/promotion_validate.py`: checks a candidate against registry baselines before promotion
- `Notebook/Registry/`: records which checkpoints are historical, current best, or baseline opponents

## Regenerating Precomputed Data

To regenerate the matrices in `Matrix_generation/`:

```bash
python Matrix_generation/Gen_board.py               # Build the graph
python Matrix_generation/Stochastic_matrix_Gen.py    # Generate transition matrices
python Matrix_generation/shortest_path-matrix.py     # Generate shortest path tensor
```
