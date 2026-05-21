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
VEHICLES = ("taxi", "bus", "underground")
VEH2IDX = {v: i for i, v in enumerate(VEHICLES)}
TICKET_VOCAB = {"": 0, "taxi": 1, "bus": 2, "underground": 3, "water": 4}
MAX_TURN = 22
REVEAL_TURNS = (3, 8, 13, 18)
NODE_DYN_DIM = 1 + 5 + 1 + 1 + 5 + 1
GLOBAL_FEATURE_DIM = 1 + 1 + 4 + 15 + 5
DEFAULT_CHECKPOINT_PATTERNS = (
    "Notebook/Models/detectives/detective_ppo_*.pt",
    "Notebook/Models/detectives/rgnn_ppo_best_*.pt",
    "Notebook/rgnn_ppo_best_*.pt",
    "rgnn_ppo_best_*.pt",
    "ppo_checkpoints/rgnn_ppo_best_*.pt",
    "Notebook/Models/detectives/detective_bc_*.pt",
    "Notebook/Models/detectives/rgnn_bc_best_*.pt",
    "Notebook/rgnn_bc_best_*.pt",
    "rgnn_bc_best_*.pt",
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


def turns_to_next_reveal(turn):
    for reveal_turn in REVEAL_TURNS:
        if reveal_turn > turn:
            return reveal_turn - turn
    return -1


def find_latest_checkpoint(root="."):
    root = Path(root)
    for pattern in DEFAULT_CHECKPOINT_PATTERNS:
        paths = sorted(set(glob.glob(str(root / pattern))))
        if paths:
            return paths[-1]
    return None


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


def build_edge_lookup():
    lookup = {}
    for v_str, neigh in ADJ.items():
        v = int(v_str) - 1
        out = []
        for u_str, types in neigh.items():
            u = int(u_str) - 1
            rel_idx = sorted({REL2IDX[t] for t in types if t in REL2IDX})
            out.append((u, rel_idx))
        lookup[v] = out
    return lookup


EDGE_LOOKUP = build_edge_lookup()


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


class DetectiveRGNN(nn.Module):
    def __init__(
        self,
        node_dyn_dim,
        node_static_dim,
        global_dim,
        hidden=64,
        n_layers=3,
        n_relations=4,
        det_emb_dim=8,
        edge_emb_dim=8,
        dropout=0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(node_dyn_dim + node_static_dim, hidden)
        self.layers = nn.ModuleList(
            [
                DenseRGCNLayer(hidden, hidden, n_relations, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.det_id_emb = nn.Embedding(5, det_emb_dim)
        self.edge_type_emb = nn.Embedding(n_relations, edge_emb_dim)

        policy_in = 2 * hidden + edge_emb_dim + global_dim + det_emb_dim
        self.policy_mlp = nn.Sequential(
            nn.Linear(policy_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        value_in = 2 * hidden + global_dim + det_emb_dim
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
        glob = batch["glob"]
        det_id = batch["det_id"]
        ego_pos = batch["ego_pos"]
        legal_neigh = batch["legal_neighbors_mask"]

        batch_size = node_dyn.shape[0]
        node_static_b = node_static.unsqueeze(0).expand(batch_size, -1, -1)
        h = self.encode(node_dyn, node_static_b, adj)

        det_emb = self.det_id_emb(det_id)
        glob_full = torch.cat([glob, det_emb], dim=-1)
        h_mean = h.mean(dim=1)
        h_max = h.max(dim=1).values
        value = self.value_mlp(torch.cat([h_mean, h_max, glob_full], dim=-1))
        value = value.squeeze(-1)

        all_logits = []
        all_cand = []
        for b in range(batch_size):
            v = int(ego_pos[b].item())
            mask = legal_neigh[b].detach().cpu().numpy()
            cand = [
                (u, rels)
                for (u, rels) in EDGE_LOOKUP[v]
                if u < N_NODES and mask[u]
            ]
            if not cand:
                all_logits.append(torch.zeros(0, device=h.device))
                all_cand.append([])
                continue

            us = torch.tensor([u for (u, _) in cand], device=h.device, dtype=torch.long)
            edge_emb = torch.stack(
                [
                    self.edge_type_emb(
                        torch.tensor(rels, device=h.device, dtype=torch.long)
                    ).sum(dim=0)
                    for (_, rels) in cand
                ],
                dim=0,
            )
            h_v = h[b, v].unsqueeze(0).expand(len(cand), -1)
            h_u = h[b, us]
            g_b = glob_full[b].unsqueeze(0).expand(len(cand), -1)
            logits = self.policy_mlp(torch.cat([h_v, h_u, edge_emb, g_b], dim=-1))
            all_logits.append(logits.squeeze(-1))
            all_cand.append([u for (u, _) in cand])

        return value, all_logits, all_cand


class GNNDetectiveEngine:
    def __init__(self, checkpoint_path=None, device=None):
        if checkpoint_path is None:
            checkpoint_path = find_latest_checkpoint()
        if checkpoint_path is None:
            raise FileNotFoundError("No GNN checkpoint found.")

        self.checkpoint_path = os.fspath(checkpoint_path)
        self.device = resolve_device(device)
        self.dense_adj = torch.from_numpy(build_dense_adj()).to(self.device)
        self.node_static = torch.from_numpy(build_node_static_features()).to(self.device)

        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        cfg = ckpt.get("config", {})
        self.model = DetectiveRGNN(
            node_dyn_dim=cfg.get("node_dyn_dim", NODE_DYN_DIM),
            node_static_dim=cfg.get("node_static_dim", int(self.node_static.shape[1])),
            global_dim=cfg.get("global_dim", GLOBAL_FEATURE_DIM),
            hidden=cfg.get("hidden", 64),
            n_layers=cfg.get("n_layers", 3),
            n_relations=cfg.get("n_relations", 4),
            dropout=0.1,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.metadata = {
            key: ckpt.get(key)
            for key in (
                "update",
                "gnn_wr",
                "heu_wr",
                "delta_pp",
                "validation_score",
                "fixed_eval",
                "hard_eval",
                "source_kind",
                "source_checkpoint",
            )
            if key in ckpt
        }
        self.reset()

    def reset(self):
        self.revealed_positions = []
        self.last_mrx_ticket = ""

    def observe_mrx_move(self, game, ticket):
        if ticket is not None:
            self.last_mrx_ticket = ticket
        if (game.turn - 3) % 5 == 0:
            self.revealed_positions.append(int(game.mrx_pos))

    def build_input(self, game, belief_state, detective_id):
        mrx_visited_mask = np.zeros(N_NODES, dtype=np.float32)
        for p in self.revealed_positions:
            mrx_visited_mask[int(p) - 1] = 1.0

        det_pos_arr = np.array([int(p) - 1 for p in game.detectives_pos], dtype=np.int64)
        ego = det_pos_arr[detective_id]
        is_det = np.zeros((5, N_NODES), dtype=np.float32)
        for k, p in enumerate(det_pos_arr):
            is_det[k, p] = 1.0
        is_ego = np.zeros(N_NODES, dtype=np.float32)
        is_ego[ego] = 1.0
        dist_per_det = np.stack([_SP_ANY_NP[p] for p in det_pos_arr], axis=1)
        min_dist = dist_per_det.min(axis=1, keepdims=True)
        node_dyn = np.concatenate(
            [
                belief_state.astype(np.float32)[:, None],
                is_det.T,
                is_ego[:, None],
                min_dist.astype(np.float32),
                dist_per_det.astype(np.float32),
                mrx_visited_mask[:, None],
            ],
            axis=1,
        ).astype(np.float32)

        legal = np.zeros((N_NODES, 3), dtype=bool)
        occupied = set(
            game.detectives_pos[:detective_id]
            + game.detectives_pos[detective_id + 1 :]
        )
        tickets = game.detective_tickets[detective_id]
        for nb, types in ADJ[game.detectives_pos[detective_id]].items():
            if nb in occupied:
                continue
            for vehicle in types:
                if vehicle in VEH2IDX and tickets.get(vehicle, 0) > 0:
                    legal[int(nb) - 1, VEH2IDX[vehicle]] = True

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
                    [game.turn / MAX_TURN, turns_to_next_reveal(game.turn) / 5.0],
                    dtype=np.float32,
                ),
                np.array(
                    [game.mrx_tickets[v] for v in ("taxi", "bus", "underground", "water")],
                    dtype=np.float32,
                )
                / 10.0,
                all_det_tickets.reshape(-1),
                last_ticket_oh,
            ]
        ).astype(np.float32)

        return {
            "node_dyn": node_dyn,
            "glob": glob_features,
            "det_id": int(detective_id),
            "ego_pos": int(ego),
            "legal_neighbors_mask": legal.any(axis=1),
        }

    def _collate_one(self, sample):
        return {
            "node_dyn": torch.from_numpy(sample["node_dyn"]).unsqueeze(0).to(self.device),
            "glob": torch.from_numpy(sample["glob"]).unsqueeze(0).to(self.device),
            "det_id": torch.tensor([sample["det_id"]], dtype=torch.long, device=self.device),
            "ego_pos": torch.tensor([sample["ego_pos"]], dtype=torch.long, device=self.device),
            "legal_neighbors_mask": torch.from_numpy(sample["legal_neighbors_mask"])
            .unsqueeze(0)
            .to(self.device),
        }

    @torch.no_grad()
    def choose_destination(self, game, belief_state, detective_id):
        sample = self.build_input(game, belief_state, detective_id)
        _, all_logits, all_cand = self.model(
            self._collate_one(sample), self.dense_adj, self.node_static
        )
        cand = all_cand[0]
        if not cand:
            return None
        best_idx = int(torch.argmax(all_logits[0]).item())
        return str(int(cand[best_idx]) + 1)

    def play_detective_turn(self, game, belief_state, detective_id):
        destination = self.choose_destination(game, belief_state, detective_id)
        if destination is None:
            return None
        origin = game.detectives_pos[detective_id]
        vehicle = game.use_ticket(detective_id, origin, destination)
        game.detectives_pos[detective_id] = destination
        return destination, vehicle
