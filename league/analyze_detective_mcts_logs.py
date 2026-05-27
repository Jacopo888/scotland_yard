import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


VALUE_COLUMNS = (
    "value_before_raw",
    "value_after_joint_raw",
    "root_value",
    "selected_q",
    "joint_root_value",
    "joint_selected_q",
)


def _read_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def _read_dataframe(path):
    path = Path(path)
    if path.suffix == ".pkl":
        return pd.read_pickle(path)
    return pd.read_parquet(path)


def _rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _safe_mean(series):
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    return float(np.mean(values))


def _safe_sum(series):
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    return float(np.sum(values))


def _value_diagnostics(samples_df):
    if samples_df.empty or "return_to_go" not in samples_df:
        return {}
    returns = pd.to_numeric(samples_df["return_to_go"], errors="coerce").to_numpy()
    diagnostics = {}
    for col in VALUE_COLUMNS:
        if col not in samples_df:
            continue
        values = pd.to_numeric(samples_df[col], errors="coerce").to_numpy()
        mask = np.isfinite(values) & np.isfinite(returns)
        if int(mask.sum()) < 2:
            continue
        entry = {
            "n": int(mask.sum()),
            "pearson_vs_return": _corr(values[mask], returns[mask]),
            "spearman_vs_return": _corr(_rankdata(values[mask]), _rankdata(returns[mask])),
            "mean": float(np.mean(values[mask])),
            "std": float(np.std(values[mask])),
        }
        if col.startswith("value_"):
            entry["mae_vs_return"] = float(np.mean(np.abs(values[mask] - returns[mask])))
        diagnostics[col] = entry
    return diagnostics


def _discover_manifests(paths):
    manifests = []
    for raw in paths:
        path = Path(raw)
        if path.is_file() and path.name == "manifest.json":
            manifests.append(path)
        elif path.is_dir():
            manifests.extend(path.glob("**/detective_mcts_logs_*/manifest.json"))
            if path.name.startswith("detective_mcts_logs_") and (path / "manifest.json").exists():
                manifests.append(path / "manifest.json")
        else:
            manifests.extend(Path(".").glob(raw))
    return sorted(set(manifests))


def _read_dataframes(paths, read_errors):
    frames = []
    for path in paths:
        try:
            frames.append(_read_dataframe(path))
        except ImportError as exc:
            read_errors.append(
                {
                    "path": str(path),
                    "error": str(exc).splitlines()[0],
                    "hint": "Install pyarrow or fastparquet for exact sample-level aggregation.",
                }
            )
        except Exception as exc:
            read_errors.append({"path": str(path), "error": str(exc)})
    return frames


def _load_log_dir(manifest_path):
    log_dir = manifest_path.parent
    manifest = _read_json(manifest_path)
    sample_paths = sorted(log_dir.glob("samples_part_*.parquet"))
    sample_paths += sorted(log_dir.glob("samples_part_*.pkl"))
    game_stats_paths = sorted(log_dir.glob("game_stats.parquet"))
    game_stats_paths += sorted(log_dir.glob("game_stats.pkl"))

    read_errors = []
    samples = _read_dataframes(sample_paths, read_errors)
    stats = _read_dataframes(game_stats_paths, read_errors)
    return {
        "manifest_path": manifest_path,
        "log_dir": log_dir,
        "manifest": manifest,
        "samples": pd.concat(samples, ignore_index=True) if samples else pd.DataFrame(),
        "game_stats": pd.concat(stats, ignore_index=True) if stats else pd.DataFrame(),
        "read_errors": read_errors,
    }


def analyze(paths):
    manifests = _discover_manifests(paths)
    if not manifests:
        raise FileNotFoundError("No detective_mcts_logs_*/manifest.json files found.")

    loaded = [_load_log_dir(path) for path in manifests]
    samples_df = pd.concat([item["samples"] for item in loaded], ignore_index=True)
    stats_df = pd.concat([item["game_stats"] for item in loaded], ignore_index=True)
    read_errors = [error for item in loaded for error in item["read_errors"]]

    manifest_rows = []
    manifest_value_diagnostics = {}
    for item in loaded:
        manifest = item["manifest"]
        sanity = manifest.get("sanity", {})
        config = manifest.get("config", {})
        for col, diag in (manifest.get("value_diagnostics") or {}).items():
            manifest_value_diagnostics.setdefault(col, []).append(
                {
                    "log_dir": item["log_dir"].name,
                    **diag,
                }
            )
        manifest_rows.append(
            {
                "log_dir": item["log_dir"].name,
                "games": sanity.get("n_games"),
                "samples": sanity.get("n_samples"),
                "detective_winrate": sanity.get("detective_winrate"),
                "avg_final_turn": sanity.get("avg_final_turn"),
                "elapsed_seconds": manifest.get("elapsed_seconds"),
                "teacher_mode": config.get("teacher_mode"),
                "simulations": config.get("simulations"),
                "rollout_turns": config.get("rollout_turns"),
                "joint_top_k": config.get("joint_top_k"),
                "joint_random_actions": config.get("joint_random_actions"),
                "value_leaf_weight": config.get("value_leaf_weight"),
                "detective_base_model_id": manifest.get("detective_base_model_id"),
                "mrx_base_model_id": manifest.get("mrx_base_model_id"),
            }
        )

    total_games = int(len(stats_df)) if not stats_df.empty else int(
        sum(row.get("games") or 0 for row in manifest_rows)
    )
    if not stats_df.empty and "detective_win" in stats_df:
        detective_winrate = float(pd.to_numeric(stats_df["detective_win"]).mean())
        avg_final_turn = _safe_mean(stats_df["final_turn"])
    else:
        games = np.asarray([row.get("games") or 0 for row in manifest_rows], dtype=np.float64)
        wr = np.asarray(
            [row.get("detective_winrate") or np.nan for row in manifest_rows],
            dtype=np.float64,
        )
        turns = np.asarray(
            [row.get("avg_final_turn") or np.nan for row in manifest_rows],
            dtype=np.float64,
        )
        valid_wr = np.isfinite(wr) & (games > 0)
        valid_turns = np.isfinite(turns) & (games > 0)
        detective_winrate = (
            float(np.average(wr[valid_wr], weights=games[valid_wr])) if valid_wr.any() else None
        )
        avg_final_turn = (
            float(np.average(turns[valid_turns], weights=games[valid_turns]))
            if valid_turns.any()
            else None
        )

    samples_summary = {}
    if not samples_df.empty:
        for col in (
            "joint_action_count",
            "root_actions",
            "visit_entropy",
            "selected_visits",
            "joint_selected_q",
            "root_value",
            "return_to_go",
        ):
            if col in samples_df:
                samples_summary[f"mean_{col}"] = _safe_mean(samples_df[col])
        if "target_in_legal_mask" in samples_df:
            samples_summary["target_in_legal_mask_rate"] = float(
                pd.to_numeric(samples_df["target_in_legal_mask"]).mean()
            )
        if "detective_id" in samples_df:
            counts = samples_df["detective_id"].value_counts().sort_index()
            samples_summary["samples_by_detective_id"] = {
                str(int(k)): int(v) for k, v in counts.items()
            }

    elapsed = [row.get("elapsed_seconds") for row in manifest_rows]
    elapsed = [float(x) for x in elapsed if x is not None and math.isfinite(float(x))]
    summary = {
        "log_dirs": [row["log_dir"] for row in manifest_rows],
        "n_shards": int(len(manifest_rows)),
        "n_games": total_games,
        "n_samples": int(len(samples_df)),
        "detective_winrate": detective_winrate,
        "avg_final_turn": avg_final_turn,
        "elapsed_seconds_max": max(elapsed) if elapsed else None,
        "seconds_per_game_max": (max(elapsed) / max(row.get("games") or 1 for row in manifest_rows))
        if elapsed
        else None,
        "configs": manifest_rows,
        "samples_summary": samples_summary,
        "value_diagnostics": _value_diagnostics(samples_df),
        "manifest_value_diagnostics": manifest_value_diagnostics,
        "read_errors": read_errors,
    }
    return summary


def build_parser():
    parser = argparse.ArgumentParser(
        description="Aggregate detective joint/Belief-MCTS log diagnostics."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["kaggle_outputs"],
        help="Log dirs, manifest files, roots to scan, or glob patterns.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON summary path.")
    return parser


def main():
    args = build_parser().parse_args()
    summary = analyze(args.paths)
    text = json.dumps(summary, indent=2)
    print(text)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
