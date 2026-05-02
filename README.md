# Scotland Yard AI

A Python implementation of the **Scotland Yard** board game, featuring fully automated AI players for both Mr. X and the detectives.

Mr. X uses **Monte Carlo Tree Search (MCTS)** to choose optimal moves, while the detectives rely on a **belief state** updated via Markov Chains and Kalman filtering to estimate Mr. X's position.

## Game Rules

- **Board**: 199 stations connected by taxi, bus, underground (and water routes for Mr. X)
- **Players**: 1 Mr. X vs 5 detectives
- **Turns**: detectives move first, then Mr. X. The game lasts up to 22 turns
- **Detective tickets**: taxi (10), bus (8), underground (4)
- **Mr. X tickets**: taxi (4), bus (3), underground (3), water (5)
- **Visibility**: Mr. X is hidden, but his position is revealed every 5 turns. Detectives only see the ticket type he uses
- **Detectives win**: by landing on Mr. X's position
- **Mr. X wins**: by surviving until turn 22

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

### Detectives — Belief State + Kalman Filter

The detective engine (`detective_engine.py`) maintains a probability distribution over all 199 stations:

- **Update With a Markov Chain Model**: after each Mr. X move, the belief state is multiplied by the transition matrix of the vehicle used
- **Kalman filter**: zeroes out probability at detective positions and renormalizes
- **Spotting**: when Mr. X is spotted (every 5 turns), the belief becomes 100% at his real position

Each detective moves toward the station with the highest probability, using a precomputed shortest-path tensor.

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
├── mrx_engine.py               # Mr. X AI (MCTS)
├── mcts_node.py                # MCTS tree node
├── utility.py                  # Turn simulation helpers
├── board_generation.py         # Board visualization (Tkinter)
├── belief_state_visualizer.py  # Belief state heatmap
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
