# League Automation

This package contains restartable command-line entry points for the self-play
loop.

## Cycle

1. `neural_mcts_logger.py`
   - reads current best Mr.X and detective checkpoints from `Notebook/Registry`;
   - logs Neural MCTS teacher decisions;
   - writes tensors, metadata, visit-policy targets, and a manifest.

2. `train_mrx_sl.py`
   - reads the latest Neural MCTS logs;
   - warm-starts from the current best Mr.X checkpoint;
   - trains a supervised Mr.X policy on soft visit targets;
   - writes a candidate checkpoint and `registry_candidate_update.json`.

3. `train_detective_rl_vs_latest_mrx.py`
   - reads current best detective and Mr.X checkpoints from the registry;
   - continues detective PPO against the current best Mr.X policy;
   - stops early if the configured improvement threshold is reached;
   - writes a candidate checkpoint and `registry_candidate_update.json`.

Promotion remains explicit: run `validation/promotion_validate.py`, inspect the
gate result, then update the registry locally if the candidate is accepted.

## Kaggle Notebooks

Thin notebook wrappers live in:

```text
Notebook/League Notebooks/
```

The notebooks intentionally call these scripts instead of duplicating large
training code.

## Local Kaggle Cycle

`kaggle_cycle.py` runs separate Kaggle kernels for CPU and GPU stages:

```text
CPU  01 logging
GPU  02 Mr.X SL training
CPU  03 Mr.X promotion validation
GPU  04 detective PPO training
CPU  05 detective promotion validation
```

Example:

```powershell
python league/kaggle_cycle.py --once --gpu-accelerator NvidiaTeslaT4
```

Useful quick smoke:

```powershell
python league/kaggle_cycle.py --once --log-games 5 --mrx-sl-epochs 1 --detective-updates 1 --promotion-games-scale 0.01 --promotion-max-games 2
```

The local process must stay alive while it is orchestrating the loop. Kaggle
kernels continue running after upload, but the local process is what polls
status, downloads outputs, applies passed promotions, and launches the next
stage.
