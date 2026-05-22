# League Notebooks

These Kaggle-oriented notebooks are thin wrappers around the `league/` scripts.
They are intended to be restarted safely: each run reads the registry aliases,
chooses the current best checkpoints, writes candidate artifacts, and emits a
`registry_candidate_update.json` only for review.

Order:

1. `01_kaggle_log_neural_mcts_teacher.ipynb`
2. `02_kaggle_train_mrx_sl_from_neural_mcts.ipynb`
3. `03_kaggle_train_detectives_rl_vs_latest_mrx.ipynb`

