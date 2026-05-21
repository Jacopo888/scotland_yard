# Model Checkpoints

This folder contains promoted neural checkpoints only.

Layout:

- `detectives/`: promoted detective GNN checkpoints.
- `mrx/`: promoted Mr.X GNN checkpoints.

Naming:

- Detectives: `detective_<kind>_vXXX.pt`
- Mr.X: `mrx_<kind>_vXXX.pt`

Examples:

- `detectives/detective_ppo_v001.pt`
- `mrx/mrx_ppo_v001.pt`

Rules:

1. Do not overwrite an existing `vXXX` file.
2. A newly trained Kaggle artifact is a candidate until it passes validation.
3. Accepted candidates get the next version number and a new entry in
   `Notebook/Registry/model_registry.json`.
4. `Notebook/Registry/best_detective.json` and `best_mrx.json` are aliases:
   they point to the current best version, while older versions stay available
   for opponent pools and regression checks.

Promotion check:

- Use `python validation/promotion_validate.py --side mrx --candidate-checkpoint <path>`
  for Mr.X candidates.
- Use `python validation/promotion_validate.py --side detectives --candidate-checkpoint <path>`
  for detective candidates.
- Promote only when the output JSON has `summary.passed: true`.
