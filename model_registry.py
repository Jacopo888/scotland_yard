import json
from pathlib import Path


DEFAULT_REGISTRY_DIR = Path("Notebook") / "Registry"


def _read_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def registry_dir(root="."):
    return Path(root) / DEFAULT_REGISTRY_DIR


def load_model_registry(root="."):
    return _read_json(registry_dir(root) / "model_registry.json")


def load_best_detective(root="."):
    return _read_json(registry_dir(root) / "best_detective.json")


def load_best_mrx(root="."):
    return _read_json(registry_dir(root) / "best_mrx.json")


def load_opponent_pools(root="."):
    return _read_json(registry_dir(root) / "opponent_pools.json")


def _all_entries(registry):
    return list(registry.get("models", [])) + list(registry.get("virtual_baselines", []))


def _checkpoint_path(path, root="."):
    checkpoint_path = Path(path)
    if checkpoint_path.is_absolute():
        return checkpoint_path
    return Path(root) / checkpoint_path


def list_models(side=None, kind=None, include_virtual=False, root="."):
    registry = load_model_registry(root)
    entries = _all_entries(registry) if include_virtual else list(registry.get("models", []))
    if side is not None:
        entries = [entry for entry in entries if entry.get("side") == side]
    if kind is not None:
        entries = [entry for entry in entries if entry.get("kind") == kind]
    return entries


def get_entry(model_id, root="."):
    registry = load_model_registry(root)
    for entry in _all_entries(registry):
        if entry.get("id") == model_id:
            return entry
    raise KeyError(f"Unknown registry id: {model_id}")


def get_best(side, root="."):
    if side == "detectives":
        best = load_best_detective(root)
    elif side == "mrx":
        best = load_best_mrx(root)
    else:
        raise ValueError("side must be 'detectives' or 'mrx'")
    entry = get_entry(best["best_model_id"], root=root)
    return {**entry, "best_alias": best}


def resolve_path(model_id, root="."):
    entry = get_entry(model_id, root=root)
    path = entry.get("path")
    if path is None:
        raise KeyError(f"Registry entry has no path: {model_id}")
    return str(_checkpoint_path(path, root=root))


def next_model_id(side, kind, root="."):
    if side not in ("detectives", "mrx"):
        raise ValueError("side must be 'detectives' or 'mrx'")
    prefix = "detective" if side == "detectives" else "mrx"
    stem = f"{prefix}_{kind}_v"
    max_version = 0
    for entry in list_models(side=side, kind=kind, root=root):
        model_id = entry.get("id", "")
        if not model_id.startswith(stem):
            continue
        suffix = model_id[len(stem):]
        if suffix.isdigit():
            max_version = max(max_version, int(suffix))
    return f"{stem}{max_version + 1:03d}"


def verify_model_paths(root="."):
    rows = []
    for entry in list_models(root=root):
        path = entry.get("path")
        resolved = _checkpoint_path(path, root=root) if path else None
        rows.append({
            "id": entry.get("id"),
            "path": path,
            "resolved_path": str(resolved) if resolved else None,
            "exists": bool(resolved and resolved.exists()),
        })
    return rows


def get_pool(pool_name, root="."):
    pools = load_opponent_pools(root)
    if pool_name not in pools:
        raise KeyError(f"Unknown pool: {pool_name}")
    return pools[pool_name]


def make_candidate_update(
    candidate_id,
    side,
    kind,
    path,
    parent,
    trained_against,
    eval_file,
    metrics,
    status="candidate",
):
    return {
        "schema_version": 1,
        "candidate": {
            "id": candidate_id,
            "side": side,
            "kind": kind,
            "status": status,
            "path": path,
            "parent": parent,
            "trained_against": trained_against,
            "eval_files": [eval_file] if eval_file else [],
            "metrics": metrics,
        },
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Inspect Scotland Yard model registry.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--best", choices=("detectives", "mrx"), default=None)
    parser.add_argument("--pool", default=None)
    parser.add_argument("--id", default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--side", choices=("detectives", "mrx"), default=None)
    parser.add_argument("--kind", default=None)
    parser.add_argument("--next-id", action="store_true")
    parser.add_argument("--verify-paths", action="store_true")
    args = parser.parse_args()

    if args.next_id:
        if not args.side or not args.kind:
            parser.error("--next-id requires --side and --kind")
        print(next_model_id(args.side, args.kind, root=args.root))
    elif args.verify_paths:
        rows = verify_model_paths(root=args.root)
        print(json.dumps(rows, indent=2))
        if not all(row["exists"] for row in rows):
            raise SystemExit(1)
    elif args.list:
        print(json.dumps(list_models(side=args.side, kind=args.kind, root=args.root), indent=2))
    elif args.best:
        print(json.dumps(get_best(args.best, root=args.root), indent=2))
    elif args.pool:
        print(json.dumps(get_pool(args.pool, root=args.root), indent=2))
    elif args.id:
        print(json.dumps(get_entry(args.id, root=args.root), indent=2))
    else:
        print(json.dumps(load_model_registry(args.root), indent=2))


if __name__ == "__main__":
    main()
