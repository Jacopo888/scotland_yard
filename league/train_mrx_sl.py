import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gnn_mrx_engine import GNNMrXEngine, MrXRGNN, RELATIONS
from league.common import (
    best_model,
    ensure_project_root,
    next_candidate,
    now_tag,
    seed_everything,
    write_candidate_update,
    write_json,
)


class MrXSoftPolicyDataset(Dataset):
    def __init__(
        self,
        node_dyn,
        glob,
        legal_mask,
        target_action,
        target_policy,
        return_norm,
        mrx_pos_idx,
    ):
        self.node_dyn = node_dyn
        self.glob = glob
        self.legal_mask = legal_mask
        self.target_action = target_action
        self.target_policy = target_policy
        self.return_norm = return_norm
        self.mrx_pos_idx = mrx_pos_idx

    def __len__(self):
        return len(self.target_action)

    def __getitem__(self, idx):
        return {
            "node_dyn": torch.from_numpy(self.node_dyn[idx].astype(np.float32, copy=False)),
            "glob": torch.from_numpy(self.glob[idx].astype(np.float32, copy=False)),
            "legal_mask": torch.from_numpy(self.legal_mask[idx].astype(np.bool_, copy=False)),
            "target_action": torch.tensor(int(self.target_action[idx]), dtype=torch.long),
            "target_policy": torch.from_numpy(
                self.target_policy[idx].astype(np.float32, copy=False)
            ),
            "return_to_go": torch.tensor(float(self.return_norm[idx]), dtype=torch.float32),
            "mrx_pos_idx": torch.tensor(int(self.mrx_pos_idx[idx]), dtype=torch.long),
        }


def _find_log_dir(path):
    if path is not None:
        return Path(path)
    candidates = sorted(Path("/kaggle/working/neural_mcts_logs").glob("neural_mcts_logs_*"))
    candidates += sorted(Path("/kaggle/input").glob("**/neural_mcts_logs_*"))
    candidates += sorted(Path("Notebook").glob("Neural_MCTS_Logs/neural_mcts_logs_*"))
    if not candidates:
        raise FileNotFoundError("No Neural MCTS log directory found. Pass --data-dir.")
    return candidates[-1]


def _load_data(data_dir):
    data_dir = Path(data_dir)
    static = np.load(data_dir / "static_graph.npz", allow_pickle=True)
    sample_paths = sorted(data_dir.glob("samples_part_*.parquet"))
    sample_paths += sorted(data_dir.glob("samples_part_*.pkl"))
    tensor_paths = sorted(data_dir.glob("tensors_part_*.npz"))
    if not sample_paths or not tensor_paths:
        raise FileNotFoundError(f"Missing samples/tensors parts in {data_dir}")

    frames = []
    node_dyn_parts = []
    glob_parts = []
    legal_parts = []
    target_parts = []
    target_policy_parts = []
    for sample_path, tensor_path in zip(sample_paths, tensor_paths):
        if sample_path.suffix == ".pkl":
            frames.append(pd.read_pickle(sample_path))
        else:
            frames.append(pd.read_parquet(sample_path))
        arr = np.load(tensor_path)
        node_dyn_parts.append(arr["node_dyn"])
        glob_parts.append(arr["glob"])
        legal_parts.append(arr["legal_action_mask"])
        target_parts.append(arr["target_action"])
        if "target_policy" in arr:
            target_policy_parts.append(arr["target_policy"])
        else:
            target = arr["target_action"]
            action_dim = arr["legal_action_mask"].shape[1]
            dense = np.zeros((len(target), action_dim), dtype=np.float16)
            dense[np.arange(len(target)), target] = np.float16(1.0)
            target_policy_parts.append(dense)

    samples = pd.concat(frames, ignore_index=True)
    node_dyn = np.concatenate(node_dyn_parts, axis=0)
    glob = np.concatenate(glob_parts, axis=0).astype(np.float32)
    legal_mask = np.concatenate(legal_parts, axis=0).astype(np.bool_)
    target_action = np.concatenate(target_parts, axis=0).astype(np.int64)
    target_policy = np.concatenate(target_policy_parts, axis=0).astype(np.float16)
    returns_raw = samples["return_to_go"].astype(np.float32).to_numpy()
    mrx_pos_idx = node_dyn[:, :, 0].argmax(axis=1).astype(np.int64)

    n = len(target_action)
    assert node_dyn.shape[0] == n
    assert glob.shape[0] == n
    assert legal_mask.shape[0] == n
    assert target_policy.shape == legal_mask.shape
    assert np.all(legal_mask[np.arange(n), target_action])
    return static, samples, node_dyn, glob, legal_mask, target_action, target_policy, returns_raw, mrx_pos_idx


def _move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _run_epoch(model, loader, dense_adj, node_static, optimizer, device, value_coef, entropy_coef, train):
    model.train(train)
    totals = defaultdict(float)
    count = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = _move_batch(batch, device)
            logits, value = model(batch, dense_adj, node_static)
            legal = batch["legal_mask"].bool()
            target_policy = batch["target_policy"]
            target_action = batch["target_action"]
            returns = batch["return_to_go"]

            masked_logits = logits.masked_fill(~legal, -1e9)
            logp = F.log_softmax(masked_logits, dim=-1)
            policy_loss = -(target_policy * logp).sum(dim=-1).mean()
            value_loss = F.mse_loss(value, returns)
            probs = logp.exp()
            entropy = -(probs * logp).sum(dim=-1).mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            bs = target_action.shape[0]
            pred = masked_logits.argmax(dim=-1)
            top3 = masked_logits.topk(3, dim=-1).indices
            count += bs
            totals["loss"] += float(loss.item()) * bs
            totals["policy_loss"] += float(policy_loss.item()) * bs
            totals["value_loss"] += float(value_loss.item()) * bs
            totals["entropy"] += float(entropy.item()) * bs
            totals["acc"] += float((pred == target_action).float().sum().item())
            totals["top3_acc"] += float(
                (top3 == target_action[:, None]).any(dim=1).float().sum().item()
            )
            totals["ticket_acc"] += float(
                ((pred % len(RELATIONS)) == (target_action % len(RELATIONS)))
                .float()
                .sum()
                .item()
            )
    return {key: value / max(count, 1) for key, value in totals.items()}


def run(args):
    ensure_project_root()
    seed_everything(args.seed)
    data_dir = _find_log_dir(args.data_dir)
    run_tag = args.run_tag or now_tag()
    out_dir = Path(args.output_dir) / f"mrx_sl_{run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    parent = best_model("mrx", root=args.root)
    parent_checkpoint = args.parent_checkpoint or parent["path"]
    candidate_id = next_candidate("mrx", "sl", explicit_id=args.candidate_id, root=args.root)

    (
        static,
        samples,
        node_dyn,
        glob,
        legal_mask,
        target_action,
        target_policy,
        returns_raw,
        mrx_pos_idx,
    ) = _load_data(data_dir)

    indices = np.arange(len(target_action))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(indices)
    val_size = max(1, int(len(indices) * args.val_frac))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    return_mean = float(returns_raw[train_idx].mean())
    return_std = float(returns_raw[train_idx].std() + 1e-6)
    returns_norm = ((returns_raw - return_mean) / return_std).astype(np.float32)

    dataset = MrXSoftPolicyDataset(
        node_dyn=node_dyn,
        glob=glob,
        legal_mask=legal_mask,
        target_action=target_action,
        target_policy=target_policy,
        return_norm=returns_norm,
        mrx_pos_idx=mrx_pos_idx,
    )
    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dense_adj = torch.from_numpy(static["dense_adj"].astype(np.float32)).to(device)
    node_static = torch.from_numpy(static["node_static"].astype(np.float32)).to(device)
    parent_engine = GNNMrXEngine(checkpoint_path=parent_checkpoint, device=device)
    cfg = dict(parent_engine.model.__dict__.get("config", {}))
    ckpt = torch.load(parent_checkpoint, map_location=device, weights_only=False)
    base_config = dict(ckpt.get("config", {}))

    model = MrXRGNN(
        node_dyn_dim=int(node_dyn.shape[-1]),
        node_static_dim=int(node_static.shape[1]),
        global_dim=int(glob.shape[-1]),
        hidden=base_config.get("hidden", 96),
        n_layers=base_config.get("n_layers", 3),
        n_relations=base_config.get("n_relations", len(RELATIONS)),
        edge_emb_dim=base_config.get("edge_emb_dim", 12),
        dropout=args.dropout,
    ).to(device)
    model.load_state_dict(parent_engine.model.state_dict(), strict=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_metric = float("inf")
    best_path = out_dir / f"{candidate_id}.pt"
    last_path = out_dir / f"{candidate_id}_last.pt"
    history = []
    started = time.time()

    print("Data:", data_dir)
    print("Samples:", len(dataset), "train:", len(train_idx), "val:", len(val_idx))
    print("Parent:", parent["id"], parent_checkpoint)
    print("Candidate:", candidate_id)
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            dense_adj,
            node_static,
            optimizer,
            device,
            args.value_coef,
            args.entropy_coef,
            train=True,
        )
        val_metrics = _run_epoch(
            model,
            val_loader,
            dense_adj,
            node_static,
            None,
            device,
            args.value_coef,
            args.entropy_coef,
            train=False,
        )
        row = {
            "epoch": int(epoch),
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.3f} "
            f"val_top3={val_metrics['top3_acc']:.3f}"
        )
        if val_metrics["loss"] < best_metric:
            best_metric = val_metrics["loss"]
            torch.save(
                _checkpoint_payload(
                    model,
                    base_config,
                    candidate_id,
                    parent["id"],
                    parent_checkpoint,
                    data_dir,
                    run_tag,
                    return_mean,
                    return_std,
                    epoch,
                    val_metrics,
                    history,
                    args,
                ),
                best_path,
            )

    val_metrics = _run_epoch(
        model,
        val_loader,
        dense_adj,
        node_static,
        None,
        device,
        args.value_coef,
        args.entropy_coef,
        train=False,
    )
    torch.save(
        _checkpoint_payload(
            model,
            base_config,
            candidate_id,
            parent["id"],
            parent_checkpoint,
            data_dir,
            run_tag,
            return_mean,
            return_std,
            args.epochs,
            val_metrics,
            history,
            args,
        ),
        last_path,
    )
    metrics = {
        "val_loss": float(best_metric),
        "val_acc": float(max(row["val_acc"] for row in history)),
        "val_top3_acc": float(max(row["val_top3_acc"] for row in history)),
        "n_samples": int(len(dataset)),
        "elapsed_seconds": float(time.time() - started),
    }
    update_path = write_candidate_update(
        out_dir=out_dir,
        candidate_id=candidate_id,
        side="mrx",
        kind="sl",
        checkpoint_path=best_path,
        parent=parent["id"],
        trained_against=["neural_mcts_teacher", "detective_best"],
        metrics=metrics,
    )
    manifest = {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "best_checkpoint": os.fspath(best_path),
        "last_checkpoint": os.fspath(last_path),
        "registry_candidate_update": os.fspath(update_path),
        "metrics": metrics,
    }
    write_json(out_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return manifest


def _checkpoint_payload(
    model,
    base_config,
    candidate_id,
    parent_id,
    parent_checkpoint,
    data_dir,
    run_tag,
    return_mean,
    return_std,
    epoch,
    val_metrics,
    history,
    args,
):
    cfg = dict(base_config)
    cfg.update(
        {
            "node_dyn_dim": 14,
            "global_dim": 28,
            "n_relations": len(RELATIONS),
            "relations": RELATIONS,
            "return_mean": float(return_mean),
            "return_std": float(return_std),
            "source_kind": "neural_mcts_sl",
            "teacher": "neural_mcts",
        }
    )
    return {
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "epoch": int(epoch),
        "val_loss": float(val_metrics["loss"]),
        "val_acc": float(val_metrics["acc"]),
        "val_top3_acc": float(val_metrics["top3_acc"]),
        "source_kind": "mrx_neural_mcts_sl",
        "source_checkpoint": os.fspath(parent_checkpoint),
        "parent_model_id": parent_id,
        "candidate_id": candidate_id,
        "data_dir": os.fspath(data_dir),
        "run_tag": run_tag,
        "history_tail": history[-10:],
        "training_args": vars(args),
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Train Mr.X GNN from Neural MCTS logs.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--parent-checkpoint", default=None)
    parser.add_argument("--candidate-id", default=None)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="/kaggle/working/mrx_sl_checkpoints")
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--value-coef", type=float, default=0.25)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    return parser


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
