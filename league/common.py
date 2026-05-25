import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np

from model_registry import (
    get_best,
    get_entry,
    get_pool,
    make_candidate_update,
    next_model_id,
    resolve_path,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ensure_project_root():
    os.chdir(PROJECT_ROOT)
    return PROJECT_ROOT


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def now_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def best_model(side, root="."):
    entry = get_best(side, root=root)
    return {
        "id": entry["id"],
        "path": resolve_path(entry["id"], root=root),
        "entry": entry,
    }


def model_path(model_id, root="."):
    return resolve_path(model_id, root=root)


def next_candidate(side, kind, explicit_id=None, root="."):
    return explicit_id or next_model_id(side=side, kind=kind, root=root)


def latest_existing_model(side, preferred_kind=None, root="."):
    best = best_model(side, root=root)
    if preferred_kind is None:
        return best

    registry_id = best["id"]
    if get_entry(registry_id, root=root).get("kind") == preferred_kind:
        return best
    return best


def candidate_update_path(out_dir):
    return Path(out_dir) / "registry_candidate_update.json"


def write_candidate_update(
    out_dir,
    candidate_id,
    side,
    kind,
    checkpoint_path,
    parent,
    trained_against,
    eval_file=None,
    metrics=None,
):
    payload = make_candidate_update(
        candidate_id=candidate_id,
        side=side,
        kind=kind,
        path=os.fspath(checkpoint_path),
        parent=parent,
        trained_against=trained_against,
        eval_file=eval_file,
        metrics=metrics or {},
    )
    path = candidate_update_path(out_dir)
    write_json(path, payload)
    return path


def load_pool(pool_name, root="."):
    return get_pool(pool_name, root=root)


def relative_to_root(path):
    path = Path(path)
    try:
        return os.fspath(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return os.fspath(path)
