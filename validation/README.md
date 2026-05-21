# Validation Entry Points

This folder contains model validation and promotion-gate scripts.

- `validate_gnn_detectives.py`: paired detective GNN validation against MCTS Mr.X.
- `validate_gnn_mrx.py`: Mr.X GNN matchup validation.
- `validate_neural_mrx_mcts.py`: Neural MCTS matchup validation.
- `promotion_validate.py`: registry-based promotion gate wrapper.

Run from the repository root, for example:

```powershell
python validation/promotion_validate.py --side mrx --candidate-checkpoint <path> --dry-run
```
