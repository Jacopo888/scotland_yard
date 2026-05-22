import argparse
import json
import os
import shutil
from datetime import date
from pathlib import Path

from league.common import ensure_project_root, relative_to_root, write_json
from model_registry import load_model_registry, registry_dir


def _read_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def _model_folder(side):
    if side == "mrx":
        return Path("Notebook") / "Models" / "mrx"
    if side == "detectives":
        return Path("Notebook") / "Models" / "detectives"
    raise ValueError(f"Unsupported side: {side}")


def _best_alias_path(side):
    if side == "mrx":
        return registry_dir(".") / "best_mrx.json"
    if side == "detectives":
        return registry_dir(".") / "best_detective.json"
    raise ValueError(f"Unsupported side: {side}")


def _promotion_eval_from_result(result):
    summary = result.get("summary", {})
    out = {
        "promotion_result": bool(summary.get("passed", False)),
        "improvement_pp": summary.get("improvement_pp"),
        "candidate_mean_score": summary.get("candidate_mean_score"),
        "baseline_mean_score": summary.get("baseline_mean_score"),
        "candidate_illegal_actions_reported": summary.get(
            "candidate_illegal_actions_reported"
        ),
    }
    for suite, row in (result.get("suites") or {}).items():
        metric = row.get("metric", "score")
        out[f"{suite}_{metric}"] = row.get("candidate_score")
        out[f"{suite}_baseline_{metric}"] = row.get("baseline_score")
        out[f"{suite}_delta_pp"] = row.get("delta_pp")
    return out


def apply_promotion(candidate_update_path, promotion_result_path, force=False):
    ensure_project_root()
    candidate_update = _read_json(candidate_update_path)
    promotion_result = _read_json(promotion_result_path)
    summary = promotion_result.get("summary", {})
    if not summary.get("passed", False):
        raise RuntimeError(
            "Promotion gate did not pass; refusing to update registry. "
            f"Reasons: {summary.get('failed_reasons', [])}"
        )

    candidate = candidate_update["candidate"]
    candidate_id = candidate["id"]
    side = candidate["side"]
    src_checkpoint = Path(candidate["path"])
    if not src_checkpoint.exists():
        src_checkpoint = Path(promotion_result.get("candidate_checkpoint", ""))
    if not src_checkpoint.exists():
        wanted = Path(candidate["path"]).name
        search_roots = [
            Path(candidate_update_path).parent,
            Path(candidate_update_path).parent.parent,
            Path(promotion_result_path).parent,
            Path(promotion_result_path).parent.parent,
        ]
        matches = []
        for root in search_roots:
            if root.exists():
                matches.extend(root.rglob(wanted))
        if matches:
            src_checkpoint = sorted(set(matches))[-1]
    if not src_checkpoint.exists():
        raise FileNotFoundError(f"Candidate checkpoint not found: {candidate['path']}")

    dst_folder = _model_folder(side)
    dst_folder.mkdir(parents=True, exist_ok=True)
    dst_checkpoint = dst_folder / f"{candidate_id}.pt"
    if dst_checkpoint.exists() and not force:
        raise FileExistsError(f"Promoted checkpoint already exists: {dst_checkpoint}")
    shutil.copy2(src_checkpoint, dst_checkpoint)

    registry = load_model_registry(".")
    for entry in registry.get("models", []):
        if entry.get("id") == candidate_id:
            raise ValueError(f"Registry already contains model id: {candidate_id}")
        if entry.get("side") == side and entry.get("status") == "best":
            entry["status"] = "historical"

    eval_files = list(candidate.get("eval_files", []))
    eval_files.append(relative_to_root(promotion_result_path))
    metrics = dict(candidate.get("metrics", {}))
    metrics.update(_promotion_eval_from_result(promotion_result))
    promoted_entry = {
        **candidate,
        "status": "best",
        "path": relative_to_root(dst_checkpoint),
        "eval_files": eval_files,
        "metrics": metrics,
        "promoted_at": date.today().isoformat(),
        "promotion_source_checkpoint": os.fspath(src_checkpoint),
    }
    registry.setdefault("models", []).append(promoted_entry)
    registry["updated_at"] = date.today().isoformat()
    write_json(registry_dir(".") / "model_registry.json", registry)

    alias = {
        "schema_version": 1,
        "updated_at": date.today().isoformat(),
        "best_model_id": candidate_id,
        "side": side,
        "path": relative_to_root(dst_checkpoint),
        "status": "best",
        "promotion_eval": _promotion_eval_from_result(promotion_result),
        "notes": "Updated automatically by league/promotion_apply.py after a passed promotion gate.",
    }
    write_json(_best_alias_path(side), alias)
    return {
        "candidate_id": candidate_id,
        "side": side,
        "checkpoint": os.fspath(dst_checkpoint),
        "best_alias": os.fspath(_best_alias_path(side)),
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Apply a passed promotion gate to the local registry."
    )
    parser.add_argument("--candidate-update", required=True)
    parser.add_argument("--promotion-result", required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    result = apply_promotion(
        candidate_update_path=args.candidate_update,
        promotion_result_path=args.promotion_result,
        force=args.force,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
