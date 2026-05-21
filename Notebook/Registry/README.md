# Scotland Yard Model Registry

This folder is the local source of truth for neural checkpoint identity,
lineage, best-model aliases, opponent pools, and promotion gates.

Current files:

- `model_registry.json`: all known neural checkpoints and virtual baselines.
- `best_detective.json`: current best detective checkpoint alias.
- `best_mrx.json`: current best Mr.X checkpoint alias.
- `opponent_pools.json`: self-play opponent pools and promotion criteria.

Current bests:

- Detectives: `detective_ppo_v001`
  - path: `Notebook/Models/detectives/detective_ppo_v001.pt`
- Mr.X: `mrx_ppo_v001`
  - path: `Notebook/Models/mrx/mrx_ppo_v001.pt`

Workflow:

1. Kaggle self-play notebooks read these registry files as input context.
2. A notebook trains a candidate and writes:
   - checkpoint file;
   - evaluation JSON;
   - `registry_candidate_update.json`.
3. Locally inspect the matrix and promotion gate.
4. If accepted, copy the checkpoint into the versioned model folder:
   - detectives: `Notebook/Models/detectives/detective_<kind>_vXXX.pt`
   - Mr.X: `Notebook/Models/mrx/mrx_<kind>_vXXX.pt`
5. Update `model_registry.json`, `best_*.json`, and `opponent_pools.json` if
   the new model should enter a pool.

Do not promote a model because it beats only the latest opponent. Promotion is
based on matrix strength plus no-regression constraints against baselines.

Never overwrite an existing promoted checkpoint. Keep historical promoted
versions because they are useful opponents for self-play, regression checks, and
lineage. Candidate artifacts that fail the promotion gate can stay in Kaggle
output history or in a local scratch/archive folder, but they should not become
`best_*` aliases.

Useful local checks:

- `python model_registry.py --verify-paths`
- `python model_registry.py --next-id --side mrx --kind ppo`
- `python model_registry.py --next-id --side detectives --kind ppo`
- `python validation/promotion_validate.py --side mrx --candidate-checkpoint <path> --dry-run`
- `python validation/promotion_validate.py --side detectives --candidate-checkpoint <path> --dry-run`
