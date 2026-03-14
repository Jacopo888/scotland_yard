# Scotland Yard AI

A Python implementation of the **Scotland Yard** board game, featuring fully automated AI players for both Mr. X and the detectives.

Mr. X uses **Monte Carlo Tree Search (MCTS)** to choose optimal moves, while the detectives rely on a **belief state** updated via Kalman filtering to estimate Mr. X's position.

## Game Rules

- **Board**: 199 stations connected by taxi, bus, underground (and water routes for Mr. X)
- **Players**: 1 Mr. X vs 5 detectives
- **Turns**: detectives move first, then Mr. X. The game lasts up to 22 turns
- **Detective tickets**: taxi (10), bus (8), underground (4)
- **Mr. X tickets**: taxi (4), bus (3), underground (3), water (5), double move (2)
- **Visibility**: Mr. X is hidden, but his position is revealed every 5 turns. Detectives only see the ticket type he uses
- **Detectives win**: by landing on Mr. X's position
- **Mr. X wins**: by surviving until turn 22

## How the AI Works

### Mr. X — Monte Carlo Tree Search

The Mr. X engine (`mrx_engine.py`) builds a tree of possible moves and evaluates them by simulating random games:

1. **Selection**: picks the most promising node using UCB1 (`score + 1.42 * sqrt(ln(N) / n)`)
2. **Expansion**: generates legal moves from the selected node
3. **Simulation (rollout)**: plays 75 random games to completion
4. **Backpropagation**: updates scores back up the tree

Parameters: 25 expansion iterations, 75 simulations per node.

### Detectives — Belief State + Kalman Filter

The detective engine (`detective_engine.py`) maintains a probability distribution over all 199 stations:

- **Update**: after each Mr. X move, the belief state is multiplied by the transition matrix of the vehicle used
- **Kalman filter**: zeroes out probability at detective positions and renormalizes
- **Spotting**: when Mr. X is spotted (every 5 turns), the belief becomes 100% at his real position

Each detective moves toward the station with the highest probability, using a precomputed shortest-path tensor.

## Project Structure

```
scotland_yard/
├── main.py                     # Entry point — runs the games
├── game.py                     # Game state, moves, rules
├── detective_engine.py         # Detective AI (belief state + Kalman)
├── mrx_engine.py               # Mr. X AI (MCTS)
├── MCTS_Node.py                # MCTS tree node
├── utility.py                  # Turn simulation helpers
├── board_generation.py         # Board visualization (Tkinter)
├── belief_state_visualizer.py  # Belief state heatmap
└── Matrix_generation/
    ├── connections.txt                # Graph: node1 node2 vehicle
    ├── board_graph.pkl                # Serialized NetworkX graph
    ├── taxi_matrix.npy                # Taxi transition matrix (199×199)
    ├── bus_matrix.npy                 # Bus transition matrix
    ├── underground_matrix.npy         # Underground transition matrix
    ├── unknown_matrix.npy             # Water/unknown transition matrix
    ├── distanze_scotland_yard_3d.npy  # Shortest path tensor (8×199×199)
    ├── Gen_board.py                   # Generates board_graph.pkl
    ├── Stochastic_matrix_Gen.py       # Generates transition matrices
    └── shortest_path-matrix.py        # Generates shortest path tensor
```

## Requirements

- Python 3.8+
- [NetworkX](https://networkx.org/)
- [NumPy](https://numpy.org/)

```bash
pip install networkx numpy
```

## Running

```bash
python main.py
```

Runs 10 automated games (Mr. X with MCTS vs detectives with belief state) and prints:
- Total execution time
- Number of Mr. X wins out of 10

## Regenerating Precomputed Data

To regenerate the matrices in `Matrix_generation/`:

```bash
python Matrix_generation/Gen_board.py               # Build the graph
python Matrix_generation/Stochastic_matrix_Gen.py    # Generate transition matrices
python Matrix_generation/shortest_path-matrix.py     # Generate shortest path tensor
```
