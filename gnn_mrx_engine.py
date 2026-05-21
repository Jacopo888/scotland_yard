import glob
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from game import ADJ, SHORTEST_PATH_TENSOR


N_NODES = 199
RELATIONS = ("taxi", "bus", "underground", "water")
REL2IDX = {r: i for i, r in enumerate(RELATIONS)}
IDX2REL = {i: r for r, i in REL2IDX.items()}
TICKET_VOCAB = {"": 0, "taxi": 1, "bus": 2, "underground": 3, "water": 4, "blocked": 0}
MAX_TURN = 22
REVEAL_TURNS = (3, 8, 13, 18)
NODE_DYN_DIM = 14
GLOBAL_FEATURE_DIM = 28
DEFAULT_CHECKPOINT_PATTERNS = (
    "Notebook/mrx_rgnn_ppo_best_*.pt",
    "mrx_rgnn_ppo_best_*.pt",
    "mrx_ppo_checkpoints/mrx_rgnn_ppo_best_*.pt",
    "Notebook/mrx_rgnn_bc_best_*.pt",
    "mrx_rgnn_bc_best_*.pt",
    "mrx_bc_checkpoints/mrx_rgnn_bc_best_*.pt",
)

_SP_ANY_NP = SHORTEST_PATH_TENSOR[7, :N_NODES, :N_NODES].astype(np.float32)


def resolve_device(device=None):
    if device is not None:
        requested = str(device)
        if requested.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA requested but unavailable in this Torch build; falling back to CPU.")
            return torch.device("cpu")
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_reveal_turn(turn):
    return int(turn) in REVEAL_TURNS


def turns_to_next_reveal(turn):
    for reveal_turn in REVEAL_TURNS:
        if reveal_turn > turn:
            return reveal_turn - turn
    return -1


def find_latest_checkpoint(root="."):
    root = Path(root)
    paths = []
    for pattern in DEFAULT_CHECKPOINT_PATTERNS:
        paths.extend(glob.glob(str(root / pattern)))
    if not paths:
        return None
    return sorted(set(paths))[-1]


def build_dense_adj():
    adj = np.zeros((len(RELATIONS), N_NODES, N_NODES), dtype=np.float32)
    for v_str, neigh in ADJ.items():
        v = int(v_str) - 1
        for u_str, types in neigh.items():
            u = int(u_str) - 1
            for vehicle in types:
                if vehicle in REL2IDX:
                    adj[REL2IDX[vehicle], v, u] = 1.0
    deg = adj.sum(axis=2, keepdims=True)
    deg[deg == 0] = 1.0
    return adj / deg


def build_node_static_features():
    deg = np.zeros((N_NODES, len(RELATIONS)), dtype=np.float32)
    for v_str, neigh in ADJ.items():
        v = int(v_str) - 1
        for _, types in neigh.items():
            for vehicle in types:
                if vehicle in REL2IDX:
                    deg[v, REL2IDX[vehicle]] += 1.0
    has_underground = (deg[:, REL2IDX["underground"]] > 0).astype(np.float32)
    return np.concatenate([deg, has_underground[:, None]], axis=1)


def flat_action(destination, ticket):
    if ticket not in REL2IDX:
        return -1
    return (int(destination) - 1) * len(RELATIONS) + REL2IDX[ticket]


class DenseRGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_relations, dropout=0.1):
        super().__init__()
        self.W0 = nn.Linear(in_dim, out_dim, bias=True)
        self.Wr = nn.Parameter(torch.empty(n_relations, in_dim, out_dim))
        nn.init.xavier_uniform_(self.Wr)
        self.dropout = nn.Dropout(dropout)
        self.residual = in_dim == out_dim

    def forward(self, h, adj):
        self_msg = self.W0(h)
        ah = torch.einsum("rmn,bnk->rbmk", adj, h)
        ahw = torch.einsum("rbmk,rkj->rbmj", ah, self.Wr)
        out = F.relu(self_msg + ahw.sum(dim=0))
        out = self.dropout(out)
        if self.residual:
            out = out + h
        return out


class MrXRGNN(nn.Module):
    def __init__(
        self,
        node_dyn_dim,
        node_static_dim,
        global_dim,
        hidden=96,
        n_layers=3,
        n_relations=4,
        edge_emb_dim=12,
        dropout=0.1,
    ):
        super().__init__()
        self.hidden = hidden
        self.n_relations = n_relations
        self.input_proj = nn.Linear(node_dyn_dim + node_static_dim, hidden)
        self.layers = nn.ModuleList(
            [
                DenseRGCNLayer(hidden, hidden, n_relations, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.edge_type_emb = nn.Embedding(n_relations, edge_emb_dim)

        policy_in = 2 * hidden + edge_emb_dim + global_dim
        self.policy_mlp = nn.Sequential(
            nn.Linear(policy_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        value_in = 2 * hidden + global_dim
        self.value_mlp = nn.Sequential(
            nn.Linear(value_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def encode(self, node_dyn, node_static_b, adj):
        h = self.input_proj(torch.cat([node_dyn, node_static_b], dim=-1))
        for layer in self.layers:
            h = layer(h, adj)
        return h

    def forward(self, batch, adj, node_static):
        node_dyn = batch["node_dyn"]
        glob_features = batch["glob"]
        mrx_pos_idx = batch["mrx_pos_idx"]
        batch_size = node_dyn.shape[0]
        n_nodes = node_dyn.shape[1]

        node_static_b = node_static.unsqueeze(0).expand(batch_size, -1, -1)
        h = self.encode(node_dyn, node_static_b, adj)

        b_idx = torch.arange(batch_size, device=h.device)
        h_origin = h[b_idx, mrx_pos_idx]
        h_origin = h_origin[:, None, None, :].expand(
            batch_size, n_nodes, self.n_relations, -1
        )
        h_dest = h[:, :, None, :].expand(batch_size, n_nodes, self.n_relations, -1)
        edge_ids = torch.arange(self.n_relations, device=h.device)
        edge_emb = self.edge_type_emb(edge_ids)[None, None, :, :].expand(
            batch_size, n_nodes, -1, -1
        )
        glob_exp = glob_features[:, None, None, :].expand(
            batch_size, n_nodes, self.n_relations, -1
        )

        feat = torch.cat([h_origin, h_dest, edge_emb, glob_exp], dim=-1)
        logits = self.policy_mlp(feat).squeeze(-1)
        logits = logits.reshape(batch_size, n_nodes * self.n_relations)

        h_mean = h.mean(dim=1)
        h_max = h.max(dim=1).values
        value = self.value_mlp(torch.cat([h_mean, h_max, glob_features], dim=-1))
        return logits, value.squeeze(-1)


class GNNMrXEngine:
    def __init__(self, checkpoint_path=None, device=None):
        if checkpoint_path is None:
            checkpoint_path = find_latest_checkpoint()
        if checkpoint_path is None:
            raise FileNotFoundError("No Mr.X GNN checkpoint found.")

        self.checkpoint_path = os.fspath(checkpoint_path)
        self.device = resolve_device(device)
        self.dense_adj = torch.from_numpy(build_dense_adj()).to(self.device)
        self.node_static = torch.from_numpy(build_node_static_features()).to(self.device)

        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        cfg = ckpt.get("config", {})
        self.model = MrXRGNN(
            node_dyn_dim=cfg.get("node_dyn_dim", NODE_DYN_DIM),
            node_static_dim=cfg.get("node_static_dim", int(self.node_static.shape[1])),
            global_dim=cfg.get("global_dim", GLOBAL_FEATURE_DIM),
            hidden=cfg.get("hidden", 96),
            n_layers=cfg.get("n_layers", 3),
            n_relations=cfg.get("n_relations", len(RELATIONS)),
            edge_emb_dim=cfg.get("edge_emb_dim", 12),
            dropout=cfg.get("dropout", 0.1),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.return_mean = float(cfg.get("return_mean", 0.0))
        self.return_std = float(cfg.get("return_std", 1.0))
        self.metadata = {
            key: ckpt.get(key)
            for key in (
                "epoch",
                "val_loss",
                "val_acc",
                "val_top3_acc",
                "train_loss",
                "update",
                "gnn_wr",
                "mrx_wr",
                "validation_score",
                "eval",
                "source_kind",
                "source_checkpoint",
                "run_tag",
                "data_dir",
            )
            if key in ckpt
        }
        self.reset()

    def reset(self):
        self.last_mrx_ticket = ""
        self.revealed_positions = []
        self.last_value = None
        self.last_action_info = None

    def observe_mrx_move(self, game, ticket):
        if ticket is not None:
            self.last_mrx_ticket = ticket
        if is_reveal_turn(game.turn):
            self.revealed_positions.append(int(game.mrx_pos))

    def build_legal_action_mask(self, game):
        mask = np.zeros((N_NODES, len(RELATIONS)), dtype=bool)
        for ticket, nodes in game.find_legal_moves_x().items():
            if ticket not in REL2IDX:
                continue
            rel_idx = REL2IDX[ticket]
            for destination in nodes:
                mask[int(destination) - 1, rel_idx] = True
        return mask.reshape(-1)

    def build_input(self, game, belief_state):
        belief = belief_state.astype(np.float32)
        mrx_idx = int(game.mrx_pos) - 1
        det_pos_arr = np.array([int(p) - 1 for p in game.detectives_pos], dtype=np.int64)

        is_mrx = np.zeros((N_NODES, 1), dtype=np.float32)
        is_mrx[mrx_idx, 0] = 1.0

        is_det = np.zeros((5, N_NODES), dtype=np.float32)
        for k, p in enumerate(det_pos_arr):
            is_det[k, p] = 1.0

        revealed = np.zeros((N_NODES, 1), dtype=np.float32)
        for p in self.revealed_positions:
            revealed[int(p) - 1, 0] = 1.0

        dist_per_det = np.stack([_SP_ANY_NP[p] for p in det_pos_arr], axis=1)
        min_dist = dist_per_det.min(axis=1, keepdims=True)
        node_dyn = np.concatenate(
            [
                is_mrx,
                is_det.T,
                belief[:, None],
                revealed,
                min_dist.astype(np.float32),
                dist_per_det.astype(np.float32),
            ],
            axis=1,
        ).astype(np.float32)

        last_ticket_oh = np.zeros(5, dtype=np.float32)
        last_ticket_oh[TICKET_VOCAB.get(self.last_mrx_ticket, 0)] = 1.0
        all_det_tickets = np.array(
            [
                [t["taxi"], t["bus"], t["underground"]]
                for t in game.detective_tickets
            ],
            dtype=np.float32,
        ) / 10.0
        glob_features = np.concatenate(
            [
                np.array(
                    [
                        game.turn / MAX_TURN,
                        turns_to_next_reveal(game.turn) / 5.0,
                        float(is_reveal_turn(game.turn)),
                        float(is_reveal_turn(game.turn + 1)),
                    ],
                    dtype=np.float32,
                ),
                np.array([game.mrx_tickets[v] for v in RELATIONS], dtype=np.float32)
                / 10.0,
                all_det_tickets.reshape(-1),
                last_ticket_oh,
            ]
        ).astype(np.float32)

        return {
            "node_dyn": node_dyn,
            "glob": glob_features,
            "mrx_pos_idx": int(mrx_idx),
            "legal_mask": self.build_legal_action_mask(game),
        }

    def _collate_one(self, sample):
        return {
            "node_dyn": torch.from_numpy(sample["node_dyn"]).unsqueeze(0).to(self.device),
            "glob": torch.from_numpy(sample["glob"]).unsqueeze(0).to(self.device),
            "mrx_pos_idx": torch.tensor(
                [sample["mrx_pos_idx"]], dtype=torch.long, device=self.device
            ),
            "legal_mask": torch.from_numpy(sample["legal_mask"])
            .unsqueeze(0)
            .to(self.device),
        }

    @torch.no_grad()
    def choose_action(self, game, belief_state):
        sample = self.build_input(game, belief_state)
        legal_mask = sample["legal_mask"]
        if not legal_mask.any():
            self.last_value = None
            self.last_action_info = {
                "destination": game.mrx_pos,
                "ticket": None,
                "flat_action": -1,
                "value": None,
            }
            return game.mrx_pos, None

        batch = self._collate_one(sample)
        logits, value = self.model(batch, self.dense_adj, self.node_static)
        legal_t = batch["legal_mask"].bool()
        masked_logits = logits.masked_fill(~legal_t, -1e9)
        action = int(torch.argmax(masked_logits[0]).item())
        destination_idx = action // len(RELATIONS)
        ticket_idx = action % len(RELATIONS)
        destination = str(destination_idx + 1)
        ticket = IDX2REL[ticket_idx]

        raw_value = float(value[0].item())
        denorm_value = raw_value * self.return_std + self.return_mean
        self.last_value = denorm_value
        self.last_action_info = {
            "destination": destination,
            "ticket": ticket,
            "flat_action": action,
            "value": denorm_value,
            "raw_value": raw_value,
            "logit": float(masked_logits[0, action].item()),
        }
        return destination, ticket

    def play_mrx_turn(self, game, belief_state):
        destination, ticket = self.choose_action(game, belief_state)
        game.x_automated_turn(destination, ticket)
        return ticket
