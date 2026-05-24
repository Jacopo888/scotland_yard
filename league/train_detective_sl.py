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

from gnn_detective_engine import DetectiveRGNN, GNNDetectiveEngine
from league.common import (
    best_model,
    ensure_project_root,
    next_candidate,
    now_tag,
    seed_everything,
    write_candidate_update,
    write_json,
)


class DetectiveSoftPolicyDataset(Dataset):
    def __init__(
        self,
        node_dyn,
        glob,
        det_id,
        ego_pos,
        legal_mask,
        target_action,
        target_policy,
        return_norm,
    ):
        self.node_dyn = node_dyn
        self.glob = glob
        self.det_id = det_id
        self.ego_pos = ego_pos
        self.legal_mask = legal_mask
        self.target_action = target_action
        self.target_policy = target_policy
        self.return_norm = return_norm

    def __len__(self):
        return len(self.target_action)

    def __getitem__(self, idx):
        return {
            "node_dyn": torch.from_numpy(self.node_dyn[idx].astype(np.float32, copy=False)),
            "glob": torch.from_numpy(self.glob[idx].astype(np.float32, copy=False)),
            "det_id": torch.tensor(int(self.det_id[idx]), dtype=torch.long),
            "ego_pos": torch.tensor(int(self.ego_pos[idx]), dtype=torch.long),
            "legal_neighbors_mask": torch.from_numpy(
                self.legal_mask[idx].astype(np.bool_, copy=False)
            ),
            "target_action": torch.tensor(int(self.target_action[idx]), dtype=torch.long),
            "target_policy": torch.from_numpy(
                self.target_policy[idx].astype(np.float32, copy=False)
            ),
            "return_to_go": torch.tensor(float(self.return_norm[idx]), dtype=torch.float32),
        }


def _expand_log_dir(path):
    path = Path(path)
    if path.name.startswith("detective_mcts_logs_"):
        return [path]
    children = sorted(path.glob("detective_mcts_logs_*"))
    return children or [path]


def _find_log_dirs(data_dir=None, data_dirs=None):
    candidates = []
    for path in data_dirs or []:
        candidates.extend(_expand_log_dir(path))
    if data_dir is not None:
        candidates.extend(_expand_log_dir(data_dir))
    if not candidates:
        candidates = sorted(Path("/kaggle/working/detective_mcts_logs").glob("detective_mcts_logs_*"))
        candidates += sorted(Path("/kaggle/input").glob("**/detective_mcts_logs_*"))
        candidates += sorted(Path("Notebook").glob("Detective_MCTS_Logs/detective_mcts_logs_*"))
    candidates = sorted({Path(path) for path in candidates})
    if not candidates:
        raise FileNotFoundError("No Detective MCTS log directory found. Pass --data-dir or --data-dirs.")
    return candidates


def _load_data(data_dirs):
    data_dirs = [Path(path) for path in data_dirs]
    static = np.load(data_dirs[0] / "static_graph.npz", allow_pickle=True)

    frames = []
    node_dyn_parts = []
    glob_parts = []
    det_id_parts = []
    ego_pos_parts = []
    legal_parts = []
    target_parts = []
    target_policy_parts = []
    for data_dir in data_dirs:
        sample_paths = sorted(data_dir.glob("samples_part_*.parquet"))
        sample_paths += sorted(data_dir.glob("samples_part_*.pkl"))
        tensor_paths = sorted(data_dir.glob("tensors_part_*.npz"))
        if not sample_paths or not tensor_paths:
            raise FileNotFoundError(f"Missing samples/tensors parts in {data_dir}")
        if len(sample_paths) != len(tensor_paths):
            raise FileNotFoundError(
                f"Samples/tensors part count mismatch in {data_dir}: "
                f"{len(sample_paths)} samples vs {len(tensor_paths)} tensors"
            )
        for sample_path, tensor_path in zip(sample_paths, tensor_paths):
            if sample_path.suffix == ".pkl":
                frames.append(pd.read_pickle(sample_path))
            else:
                frames.append(pd.read_parquet(sample_path))
            arr = np.load(tensor_path)
            node_dyn_parts.append(arr["node_dyn"])
            glob_parts.append(arr["glob"])
            det_id_parts.append(arr["det_id"])
            ego_pos_parts.append(arr["ego_pos"])
            legal_parts.append(arr["legal_neighbors_mask"])
            target_parts.append(arr["target_action"])
            if "target_policy" in arr:
                target_policy_parts.append(arr["target_policy"])
            else:
                target = arr["target_action"]
                dense = np.zeros((len(target), 199), dtype=np.float16)
                dense[np.arange(len(target)), target] = np.float16(1.0)
                target_policy_parts.append(dense)

    samples = pd.concat(frames, ignore_index=True)
    node_dyn = np.concatenate(node_dyn_parts, axis=0)
    glob = np.concatenate(glob_parts, axis=0).astype(np.float32)
    det_id = np.concatenate(det_id_parts, axis=0).astype(np.int64)
    ego_pos = np.concatenate(ego_pos_parts, axis=0).astype(np.int64)
    legal_mask = np.concatenate(legal_parts, axis=0).astype(np.bool_)
    target_action = np.concatenate(target_parts, axis=0).astype(np.int64)
    target_policy = np.concatenate(target_policy_parts, axis=0).astype(np.float16)
    returns_raw = samples["return_to_go"].astype(np.float32).to_numpy()

    n = len(target_action)
    assert node_dyn.shape[0] == n
    assert glob.shape[0] == n
    assert det_id.shape[0] == n
    assert ego_pos.shape[0] == n
    assert legal_mask.shape == target_policy.shape
    assert np.all(legal_mask[np.arange(n), target_action])
    return static, samples, node_dyn, glob, det_id, ego_pos, legal_mask, target_action, target_policy, returns_raw


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
            value, logits_list, cand_list = model(batch, dense_adj, node_static)
            returns = batch["return_to_go"]
            target_actions = batch["target_action"]
            target_policy_dense = batch["target_policy"]

            losses = []
            policy_losses = []
            entropies = []
            correct = 0
            top3_correct = 0
            active = 0
            for idx, (logits, candidates) in enumerate(zip(logits_list, cand_list)):
                if len(candidates) == 0:
                    continue
                cand_t = torch.tensor(candidates, dtype=torch.long, device=device)
                target_probs = target_policy_dense[idx, cand_t]
                prob_sum = target_probs.sum()
                if prob_sum <= 0:
                    target = int(target_actions[idx].item())
                    target_probs = (cand_t == target).float()
                    prob_sum = target_probs.sum()
                target_probs = target_probs / prob_sum.clamp_min(1e-8)
                logp = F.log_softmax(logits, dim=-1)
                policy_loss = -(target_probs * logp).sum()
                probs = logp.exp()
                entropy = -(probs * logp).sum()
                losses.append(policy_loss)
                policy_losses.append(policy_loss)
                entropies.append(entropy)

                pred_idx = int(torch.argmax(logits).item())
                pred = int(candidates[pred_idx])
                target = int(target_actions[idx].item())
                correct += int(pred == target)
                topk = min(3, len(candidates))
                top_idx = torch.topk(logits, topk).indices.detach().cpu().tolist()
                top_nodes = {int(candidates[i]) for i in top_idx}
                top3_correct += int(target in top_nodes)
                active += 1

            if active == 0:
                continue
            policy_loss = torch.stack(policy_losses).mean()
            entropy = torch.stack(entropies).mean()
            value_loss = F.mse_loss(value, returns)
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            bs = int(active)
            count += bs
            totals["loss"] += float(loss.item()) * bs
            totals["policy_loss"] += float(policy_loss.item()) * bs
            totals["value_loss"] += float(value_loss.item()) * bs
            totals["entropy"] += float(entropy.item()) * bs
            totals["acc"] += float(correct)
            totals["top3_acc"] += float(top3_correct)
    return {key: value / max(count, 1) for key, value in totals.items()}


def run(args):
    ensure_project_root()
    seed_everything(args.seed)
    data_dirs = _find_log_dirs(args.data_dir, args.data_dirs)
    run_tag = args.run_tag or now_tag()
    out_dir = Path(args.output_dir) / f"detective_sl_{run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    parent = best_model("detectives", root=args.root)
    parent_checkpoint = args.parent_checkpoint or parent["path"]
    candidate_id = next_candidate(
        "detectives",
        "sl",
        explicit_id=args.candidate_id,
        root=args.root,
    )

    (
        static,
        samples,
        node_dyn,
        glob,
        det_id,
        ego_pos,
        legal_mask,
        target_action,
        target_policy,
        returns_raw,
    ) = _load_data(data_dirs)

    indices = np.arange(len(target_action))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(indices)
    val_size = max(1, int(len(indices) * args.val_frac))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    return_mean = float(returns_raw[train_idx].mean())
    return_std = float(returns_raw[train_idx].std() + 1e-6)
    returns_norm = ((returns_raw - return_mean) / return_std).astype(np.float32)

    dataset = DetectiveSoftPolicyDataset(
        node_dyn=node_dyn,
        glob=glob,
        det_id=det_id,
        ego_pos=ego_pos,
        legal_mask=legal_mask,
        target_action=target_action,
        target_policy=target_policy,
        return_norm=returns_norm,
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
    parent_engine = GNNDetectiveEngine(checkpoint_path=parent_checkpoint, device=device)
    parent_ckpt = torch.load(parent_checkpoint, map_location=device, weights_only=False)
    base_config = dict(parent_ckpt.get("config", {}))
    model = DetectiveRGNN(
        node_dyn_dim=int(node_dyn.shape[-1]),
        node_static_dim=int(node_static.shape[1]),
        global_dim=int(glob.shape[-1]),
        hidden=base_config.get("hidden", 64),
        n_layers=base_config.get("n_layers", 3),
        n_relations=base_config.get("n_relations", 4),
        dropout=args.dropout,
    ).to(device)
    model.load_state_dict(parent_engine.model.state_dict(), strict=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_metric = float("inf")
    best_path = out_dir / f"{candidate_id}.pt"
    last_path = out_dir / f"{candidate_id}_last.pt"
    history = []
    started = time.time()

    print("Data dirs:", json.dumps([os.fspath(path) for path in data_dirs], indent=2))
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
                    data_dirs,
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
            data_dirs,
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
        side="detectives",
        kind="sl",
        checkpoint_path=best_path,
        parent=parent["id"],
        trained_against=["belief_mcts_teacher"],
        metrics=metrics,
    )
    manifest = {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "best_checkpoint": os.fspath(best_path),
        "last_checkpoint": os.fspath(last_path),
        "registry_candidate_update": os.fspath(update_path),
        "data_dirs": [os.fspath(path) for path in data_dirs],
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
    data_dirs,
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
            "return_mean": float(return_mean),
            "return_std": float(return_std),
            "source_kind": "detective_belief_mcts_sl",
            "teacher": "belief_mcts_detective",
        }
    )
    return {
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "epoch": int(epoch),
        "val_loss": float(val_metrics["loss"]),
        "val_acc": float(val_metrics["acc"]),
        "val_top3_acc": float(val_metrics["top3_acc"]),
        "source_kind": "detective_belief_mcts_sl",
        "source_checkpoint": os.fspath(parent_checkpoint),
        "parent_model_id": parent_id,
        "candidate_id": candidate_id,
        "data_dirs": [os.fspath(path) for path in data_dirs],
        "run_tag": run_tag,
        "history_tail": history[-10:],
        "training_args": vars(args),
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Train detective GNN from Belief MCTS logs.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--data-dirs", nargs="*", default=None)
    parser.add_argument("--parent-checkpoint", default=None)
    parser.add_argument("--candidate-id", default=None)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="/kaggle/working/detective_sl_checkpoints")
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--value-coef", type=float, default=0.25)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    return parser


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
