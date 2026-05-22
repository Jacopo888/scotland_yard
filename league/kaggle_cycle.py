import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from league.promotion_apply import apply_promotion


DEFAULT_GPU_ACCELERATOR = "NvidiaTeslaT4"
TERMINAL_STATUSES = {"complete", "error", "cancel_acknowledged", "cancelled"}


def _run(cmd, cwd=None, check=True):
    print("+", " ".join(os.fspath(c) for c in cmd))
    result = subprocess.run(
        [os.fspath(c) for c in cmd],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.stdout:
        print(result.stdout.rstrip())
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout)
    return result


def _slugify(value):
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    return value[:63].strip("-")


def _read_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _kaggle_username():
    result = _run(["kaggle", "config", "view"], check=True)
    match = re.search(r"- username:\s*([^\s]+)", result.stdout or "")
    if not match:
        raise RuntimeError("Could not infer Kaggle username. Pass --owner.")
    return match.group(1)


def _load_state(path):
    path = Path(path)
    if path.exists():
        return _read_json(path)
    return {
        "schema_version": 1,
        "best_mrx_kernel": None,
        "best_detective_kernel": None,
        "runs": [],
    }


def _save_state(path, state):
    state["updated_at"] = datetime.now().isoformat()
    _write_json(path, state)


def _kernel_folder(work_dir, stage):
    folder = Path(work_dir) / "kernels" / stage
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _metadata(owner, slug, title, code_file, kernel_sources=None, enable_gpu=False):
    return {
        "id": f"{owner}/{slug}",
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true" if enable_gpu else "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": kernel_sources or [],
        "model_sources": [],
    }


def _base_bootstrap(repo_url, repo_ref):
    checkout = ""
    if repo_ref:
        checkout = f"subprocess.run(['git', 'checkout', {repo_ref!r}], cwd=repo, check=True)\n"
    return f"""
import glob
import os
import subprocess
import sys
from pathlib import Path

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'networkx', 'numpy', 'pandas', 'pyarrow', 'tqdm', 'scipy', 'matplotlib', 'torch'], check=False)
repo = Path('/tmp/scotland_yard')
if not repo.exists():
    subprocess.run(['git', 'clone', '--depth', '1', {repo_url!r}, str(repo)], check=True)
{checkout}os.chdir(repo)

def newest(patterns, exclude=()):
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern, recursive=True))
    out = []
    for path in paths:
        name = os.path.basename(path)
        if any(token in name for token in exclude):
            continue
        out.append(path)
    return sorted(set(out))[-1] if out else None
"""


def _write_kernel(folder, metadata, source):
    code_file = metadata["code_file"]
    _write_json(folder / "kernel-metadata.json", metadata)
    (folder / code_file).write_text(source, encoding="utf-8")


def _push_kernel(folder, accelerator=None, timeout=None):
    cmd = ["kaggle", "kernels", "push", "-p", folder]
    if timeout is not None:
        cmd += ["--timeout", str(timeout)]
    if accelerator:
        cmd += ["--accelerator", accelerator]
    _run(cmd)


def _status(kernel):
    result = _run(["kaggle", "kernels", "status", kernel], check=False)
    match = re.search(r'has status "([^"]+)"', result.stdout or "")
    if not match:
        return "unknown"
    status = match.group(1).lower()
    return status.rsplit(".", 1)[-1]


def _wait(kernel, poll_seconds):
    while True:
        status = _status(kernel)
        print(f"{kernel}: {status}")
        if status in TERMINAL_STATUSES:
            if status != "complete":
                raise RuntimeError(f"Kaggle kernel failed: {kernel} status={status}")
            return status
        time.sleep(poll_seconds)


def _download(kernel, output_root, force=True, attempts=3):
    target = Path(output_root) / kernel.replace("/", "__")
    target.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "kernels", "output", kernel, "-p", target]
    if force:
        cmd.append("--force")
    for attempt in range(1, attempts + 1):
        try:
            _run(cmd)
            break
        except subprocess.CalledProcessError:
            if attempt == attempts:
                raise
            print(f"Download failed for {kernel}; retrying ({attempt + 1}/{attempts})...")
            time.sleep(10 * attempt)
    return target


def _find_one(root, patterns):
    hits = []
    root = Path(root)
    for pattern in patterns:
        hits.extend(root.glob(pattern))
    hits = sorted(set(hits))
    if not hits:
        return None
    return hits[-1]


def _promotion_passed(path):
    payload = _read_json(path)
    return bool(payload.get("summary", {}).get("passed", False))


def prepare_log_kernel(args, owner, slug, kernel_sources):
    folder = _kernel_folder(args.work_dir, "01_log")
    metadata = _metadata(
        owner=owner,
        slug=slug,
        title=slug,
        code_file="run_log.py",
        kernel_sources=kernel_sources,
        enable_gpu=bool(args.log_accelerator),
    )
    device = "cuda" if args.log_accelerator else "cpu"
    source = _base_bootstrap(args.repo_url, args.repo_ref) + f"""
mrx_ckpt = newest(['/kaggle/input/**/*.pt'], exclude=('detective_', 'last'))
det_ckpt = newest(['/kaggle/input/**/detective_*.pt'], exclude=('last',))
cmd = [
    sys.executable, 'league/neural_mcts_logger.py',
    '--games', {args.log_games!r},
    '--simulations', {args.log_simulations!r},
    '--device', {device!r},
    '--output-dir', '/kaggle/working/neural_mcts_logs',
    '--games-per-part', {args.log_games_per_part!r},
    '--log-every', {args.log_every!r},
]
if mrx_ckpt:
    cmd += ['--mrx-checkpoint', mrx_ckpt]
if det_ckpt:
    cmd += ['--detective-checkpoint', det_ckpt]
cmd = [str(x) for x in cmd]
print('Running:', ' '.join(cmd))
subprocess.run(cmd, check=True)
"""
    _write_kernel(folder, metadata, source)
    return folder


def prepare_train_mrx_kernel(args, owner, slug, kernel_sources):
    folder = _kernel_folder(args.work_dir, "02_train_mrx")
    metadata = _metadata(
        owner=owner,
        slug=slug,
        title=slug,
        code_file="run_train_mrx.py",
        kernel_sources=kernel_sources,
        enable_gpu=True,
    )
    source = _base_bootstrap(args.repo_url, args.repo_ref) + f"""
data_dir = newest(['/kaggle/input/**/neural_mcts_logs_*'])
parent = newest(['/kaggle/input/**/mrx_*.pt'], exclude=('last',))
cmd = [
    sys.executable, 'league/train_mrx_sl.py',
    '--epochs', {args.mrx_sl_epochs!r},
    '--batch-size', {args.mrx_sl_batch_size!r},
    '--device', 'cuda',
    '--output-dir', '/kaggle/working/mrx_sl_checkpoints',
]
if data_dir:
    cmd += ['--data-dir', data_dir]
if parent:
    cmd += ['--parent-checkpoint', parent]
cmd = [str(x) for x in cmd]
print('Running:', ' '.join(cmd))
subprocess.run(cmd, check=True)
"""
    _write_kernel(folder, metadata, source)
    return folder


def prepare_promote_mrx_kernel(args, owner, slug, kernel_sources):
    folder = _kernel_folder(args.work_dir, "03_promote_mrx")
    metadata = _metadata(
        owner=owner,
        slug=slug,
        title=slug,
        code_file="run_promote_mrx.py",
        kernel_sources=kernel_sources,
        enable_gpu=bool(args.promotion_accelerator),
    )
    device = "cuda" if args.promotion_accelerator else "cpu"
    max_games_line = (
        f"cmd += ['--max-games-per-suite', {args.promotion_max_games!r}]\n"
        if args.promotion_max_games is not None
        else ""
    )
    source = _base_bootstrap(args.repo_url, args.repo_ref) + f"""
candidate = newest(['/kaggle/input/**/mrx_sl_*.pt', '/kaggle/input/**/mrx_ppo_*.pt'], exclude=('last',))
baseline = newest(['/kaggle/input/**/mrx_*.pt'], exclude=('sl_', 'last'))
if not candidate:
    raise SystemExit('Missing Mr.X candidate checkpoint in kernel sources')
candidate_id = Path(candidate).stem
out_dir = Path('/kaggle/working/promotion_mrx')
out_dir.mkdir(parents=True, exist_ok=True)
cmd = [
    sys.executable, 'validation/promotion_validate.py',
    '--side', 'mrx',
    '--candidate-checkpoint', candidate,
    '--candidate-id', candidate_id,
    '--games-scale', {args.promotion_games_scale!r},
    '--device', {device!r},
    '--quiet',
    '--output', str(out_dir / 'promotion_result.json'),
]
if baseline:
    cmd += ['--baseline-checkpoint', baseline]
{max_games_line}cmd = [str(x) for x in cmd]
print('Running:', ' '.join(cmd))
subprocess.run(cmd, check=True)
"""
    _write_kernel(folder, metadata, source)
    return folder


def prepare_train_detective_kernel(args, owner, slug, kernel_sources):
    folder = _kernel_folder(args.work_dir, "04_train_detective")
    metadata = _metadata(
        owner=owner,
        slug=slug,
        title=slug,
        code_file="run_train_detective.py",
        kernel_sources=kernel_sources,
        enable_gpu=True,
    )
    source = _base_bootstrap(args.repo_url, args.repo_ref) + f"""
mrx_candidate = newest(['/kaggle/input/**/mrx_sl_*.pt', '/kaggle/input/**/mrx_ppo_*.pt'], exclude=('last',))
det_parent = newest(['/kaggle/input/**/detective_*.pt'], exclude=('last',))
cmd = [
    sys.executable, 'league/train_detective_rl_vs_latest_mrx.py',
    '--updates', {args.detective_updates!r},
    '--games-per-update', {args.detective_games_per_update!r},
    '--eval-games', {args.detective_eval_games!r},
    '--device', 'cuda',
    '--output-dir', '/kaggle/working/detective_rl_checkpoints',
    '--stop-on-improvement',
]
if mrx_candidate:
    cmd += ['--mrx-checkpoint', mrx_candidate]
if det_parent:
    cmd += ['--detective-checkpoint', det_parent]
cmd = [str(x) for x in cmd]
print('Running:', ' '.join(cmd))
subprocess.run(cmd, check=True)
"""
    _write_kernel(folder, metadata, source)
    return folder


def prepare_promote_detective_kernel(args, owner, slug, kernel_sources):
    folder = _kernel_folder(args.work_dir, "05_promote_detective")
    metadata = _metadata(
        owner=owner,
        slug=slug,
        title=slug,
        code_file="run_promote_detective.py",
        kernel_sources=kernel_sources,
        enable_gpu=bool(args.promotion_accelerator),
    )
    device = "cuda" if args.promotion_accelerator else "cpu"
    max_games_line = (
        f"cmd += ['--max-games-per-suite', {args.promotion_max_games!r}]\n"
        if args.promotion_max_games is not None
        else ""
    )
    source = _base_bootstrap(args.repo_url, args.repo_ref) + f"""
candidate = newest(['/kaggle/input/**/detective_ppo_*.pt'], exclude=('last',))
baseline = newest(['/kaggle/input/**/detective_*.pt'], exclude=('last',))
if not candidate:
    raise SystemExit('Missing detective candidate checkpoint in kernel sources')
candidate_id = Path(candidate).stem
out_dir = Path('/kaggle/working/promotion_detective')
out_dir.mkdir(parents=True, exist_ok=True)
cmd = [
    sys.executable, 'validation/promotion_validate.py',
    '--side', 'detectives',
    '--candidate-checkpoint', candidate,
    '--candidate-id', candidate_id,
    '--games-scale', {args.promotion_games_scale!r},
    '--device', {device!r},
    '--quiet',
    '--output', str(out_dir / 'promotion_result.json'),
]
if baseline and baseline != candidate:
    cmd += ['--baseline-checkpoint', baseline]
{max_games_line}cmd = [str(x) for x in cmd]
print('Running:', ' '.join(cmd))
subprocess.run(cmd, check=True)
"""
    _write_kernel(folder, metadata, source)
    return folder


def _run_stage(args, stage_name, slug, folder, accelerator, timeout):
    kernel = f"{args.owner}/{slug}"
    if args.prepare_only:
        print(f"Prepared {stage_name}: {folder}")
        return kernel, None
    _push_kernel(folder, accelerator=accelerator, timeout=timeout)
    _wait(kernel, poll_seconds=args.poll_seconds)
    output = _download(kernel, args.output_dir)
    return kernel, output


def _apply_if_passed(side, train_output, promotion_output, force=False):
    candidate_update = _find_one(train_output, ["**/registry_candidate_update.json"])
    promotion_result = _find_one(promotion_output, ["**/promotion_result.json"])
    if not candidate_update:
        raise FileNotFoundError(f"Missing registry_candidate_update.json in {train_output}")
    if not promotion_result:
        raise FileNotFoundError(f"Missing promotion_result.json in {promotion_output}")
    if not _promotion_passed(promotion_result):
        print(f"{side} promotion did not pass; not applying registry update.")
        return None
    result = apply_promotion(candidate_update, promotion_result, force=force)
    print(json.dumps(result, indent=2))
    return result


def run_once(args):
    args.owner = args.owner or _kaggle_username()
    state = _load_state(args.state)
    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d%H%M%S")
    prefix = _slugify(f"{args.slug_prefix}-{run_tag}")
    run_record = {"run_tag": run_tag, "stages": {}}

    source_kernels = [
        k
        for k in (state.get("best_mrx_kernel"), state.get("best_detective_kernel"))
        if k
    ]

    log_slug = _slugify(f"{prefix}-01-log")
    log_folder = prepare_log_kernel(args, args.owner, log_slug, source_kernels)
    log_kernel, log_output = _run_stage(
        args,
        "log",
        log_slug,
        log_folder,
        accelerator=args.log_accelerator,
        timeout=args.log_timeout,
    )
    run_record["stages"]["log"] = log_kernel

    train_mrx_slug = _slugify(f"{prefix}-02-train-mrx")
    train_mrx_sources = [log_kernel] + source_kernels
    train_mrx_folder = prepare_train_mrx_kernel(
        args,
        args.owner,
        train_mrx_slug,
        train_mrx_sources,
    )
    train_mrx_kernel, train_mrx_output = _run_stage(
        args,
        "train_mrx",
        train_mrx_slug,
        train_mrx_folder,
        accelerator=args.gpu_accelerator,
        timeout=args.train_timeout,
    )
    run_record["stages"]["train_mrx"] = train_mrx_kernel

    promote_mrx_slug = _slugify(f"{prefix}-03-promote-mrx")
    promote_mrx_sources = [train_mrx_kernel] + source_kernels
    promote_mrx_folder = prepare_promote_mrx_kernel(
        args,
        args.owner,
        promote_mrx_slug,
        promote_mrx_sources,
    )
    promote_mrx_kernel, promote_mrx_output = _run_stage(
        args,
        "promote_mrx",
        promote_mrx_slug,
        promote_mrx_folder,
        accelerator=args.promotion_accelerator,
        timeout=args.promotion_timeout,
    )
    run_record["stages"]["promote_mrx"] = promote_mrx_kernel

    if args.prepare_only:
        state["runs"].append(run_record)
        _save_state(args.state, state)
        return state

    mrx_apply = _apply_if_passed(
        "mrx",
        train_output=train_mrx_output,
        promotion_output=promote_mrx_output,
        force=args.force_apply,
    )
    if not mrx_apply:
        state["runs"].append(run_record)
        _save_state(args.state, state)
        print("Stopping cycle because Mr.X candidate was not promoted.")
        return state
    state["best_mrx_kernel"] = train_mrx_kernel
    _save_state(args.state, state)

    train_det_slug = _slugify(f"{prefix}-04-train-det")
    train_det_sources = [train_mrx_kernel]
    if state.get("best_detective_kernel"):
        train_det_sources.append(state["best_detective_kernel"])
    train_det_folder = prepare_train_detective_kernel(
        args,
        args.owner,
        train_det_slug,
        train_det_sources,
    )
    train_det_kernel, train_det_output = _run_stage(
        args,
        "train_detective",
        train_det_slug,
        train_det_folder,
        accelerator=args.gpu_accelerator,
        timeout=args.train_timeout,
    )
    run_record["stages"]["train_detective"] = train_det_kernel

    promote_det_slug = _slugify(f"{prefix}-05-promote-det")
    promote_det_sources = [train_det_kernel]
    if state.get("best_detective_kernel"):
        promote_det_sources.append(state["best_detective_kernel"])
    promote_det_folder = prepare_promote_detective_kernel(
        args,
        args.owner,
        promote_det_slug,
        promote_det_sources,
    )
    promote_det_kernel, promote_det_output = _run_stage(
        args,
        "promote_detective",
        promote_det_slug,
        promote_det_folder,
        accelerator=args.promotion_accelerator,
        timeout=args.promotion_timeout,
    )
    run_record["stages"]["promote_detective"] = promote_det_kernel

    det_apply = _apply_if_passed(
        "detectives",
        train_output=train_det_output,
        promotion_output=promote_det_output,
        force=args.force_apply,
    )
    if det_apply:
        state["best_detective_kernel"] = train_det_kernel
    else:
        print("Detective candidate was not promoted.")

    state["runs"].append(run_record)
    _save_state(args.state, state)
    return state


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run the Kaggle CPU/GPU league cycle from the local machine."
    )
    parser.add_argument("--owner", default=None, help="Kaggle username; defaults to kaggle config.")
    parser.add_argument("--repo-url", default="https://github.com/Jacopo888/scotland_yard.git")
    parser.add_argument("--repo-ref", default=None, help="Optional git branch/tag/commit to checkout on Kaggle.")
    parser.add_argument("--slug-prefix", default="sy-league")
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--work-dir", default=".kaggle_cycle")
    parser.add_argument("--output-dir", default="kaggle_outputs")
    parser.add_argument("--state", default=".kaggle_cycle/state.json")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force-apply", action="store_true")

    parser.add_argument("--gpu-accelerator", default=DEFAULT_GPU_ACCELERATOR)
    parser.add_argument("--log-accelerator", default=None)
    parser.add_argument("--promotion-accelerator", default=None)
    parser.add_argument("--log-timeout", type=int, default=None)
    parser.add_argument("--train-timeout", type=int, default=None)
    parser.add_argument("--promotion-timeout", type=int, default=None)

    parser.add_argument("--log-games", type=int, default=2000)
    parser.add_argument("--log-simulations", type=int, default=32)
    parser.add_argument("--log-games-per-part", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=25)

    parser.add_argument("--mrx-sl-epochs", type=int, default=12)
    parser.add_argument("--mrx-sl-batch-size", type=int, default=256)
    parser.add_argument("--detective-updates", type=int, default=20)
    parser.add_argument("--detective-games-per-update", type=int, default=16)
    parser.add_argument("--detective-eval-games", type=int, default=50)

    parser.add_argument("--promotion-games-scale", type=float, default=1.0)
    parser.add_argument("--promotion-max-games", type=int, default=None)
    parser.add_argument("--once", action="store_true", help="Run one full league cycle.")
    parser.add_argument("--loop", action="store_true", help="Repeat cycles until a promotion fails or interrupted.")
    return parser


def main():
    args = build_parser().parse_args()
    if not args.once and not args.loop and not args.prepare_only:
        raise SystemExit("Pass --once, --loop, or --prepare-only.")
    while True:
        run_once(args)
        if not args.loop:
            break


if __name__ == "__main__":
    main()
