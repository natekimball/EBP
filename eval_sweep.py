#!/usr/bin/env python3
"""
eval_sweep.py

Evaluates every checkpoint in a training output directory on math benchmarks
(AIME24 by default) and logs the results to the same W&B run that the model
was trained under, using the checkpoint step as the x-axis so eval curves
align with training loss curves.

The script:
  1. Discovers step_N checkpoint directories inside the output_dir.
  2. Finds the corresponding W&B run by matching output_dir in ./wandb/ configs.
  3. Resumes that W&B run with a custom eval_step x-axis metric.
  4. Evaluates each checkpoint, caching results in <output_dir>/eval_cache/.
  5. Logs pass@1, pass@4 (if applicable), and majority-vote accuracy per step.

Usage
-----
    # All checkpoints in cpt_output, AIME 2024, 32 samples each:
    python eval_sweep.py ./cpt_output

    # Every 5th checkpoint, 8 samples, two benchmarks:
    python eval_sweep.py ./cpt_output --every_n 5 --num_samples 8 \\
        --benchmarks aime24 aime23

    # Specific step range:
    python eval_sweep.py ./cpt_output --min_step 100000 --max_step 500000

    # Dry run — list which checkpoints would be evaluated without running:
    python eval_sweep.py ./cpt_output --dry_run

    # Force re-evaluation even if cached results exist:
    python eval_sweep.py ./cpt_output --no_cache
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import yaml
import wandb

from eval_benchmark import BENCHMARK_CONFIGS, evaluate as eval_checkpoint


# ---------------------------------------------------------------------------
# W&B run discovery
# ---------------------------------------------------------------------------

def _normalize_path(p: str) -> Path:
    """Resolve to absolute Path, following symlinks."""
    return Path(p).resolve()


def find_wandb_run(
    output_dir: str,
    wandb_dir: str = "./wandb",
) -> Optional[dict]:
    """Search local ./wandb/ for the run whose output_dir matches *output_dir*.

    Returns a dict with keys: run_id, run_name, project, wandb_dir_path.
    When multiple runs match, the most recent one (by directory name) is used.
    """
    target = _normalize_path(output_dir)
    wandb_root = Path(wandb_dir)
    if not wandb_root.exists():
        return None

    matches = []
    for run_dir in sorted(wandb_root.glob("run-*")):
        config_path = run_dir / "files" / "config.yaml"
        if not config_path.exists():
            continue
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
        except Exception:
            continue

        raw_output_dir = cfg.get("output_dir", {})
        if isinstance(raw_output_dir, dict):
            raw_output_dir = raw_output_dir.get("value", "")
        if not raw_output_dir:
            continue

        if _normalize_path(raw_output_dir) == target:
            run_id = run_dir.name.split("-")[-1]
            run_name = (cfg.get("wandb_run_name") or {}).get("value") or run_id
            project = (cfg.get("wandb_project") or {}).get("value") or "EBP"
            matches.append(
                dict(
                    run_id=run_id,
                    run_name=run_name,
                    project=project,
                    wandb_dir_path=str(run_dir),
                    dir_name=run_dir.name,
                )
            )

    if not matches:
        return None
    # Most recent match (directory names sort chronologically)
    return matches[-1]


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def discover_checkpoints(output_dir: str) -> list[tuple[int, str]]:
    """Return sorted [(step, path)] for all step_N dirs in output_dir.

    'final' is mapped to max(step) + 1 so it sorts last.
    """
    root = Path(output_dir)
    results = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        m = re.fullmatch(r"step_(\d+)", entry.name)
        if m:
            results.append((int(m.group(1)), str(entry)))
    results.sort()

    # Append 'final' if present
    final = root / "final"
    if final.is_dir():
        max_step = results[-1][0] if results else 0
        results.append((max_step + 1, str(final)))

    return results


# ---------------------------------------------------------------------------
# Per-checkpoint eval with caching
# ---------------------------------------------------------------------------

def cache_path(output_dir: str, benchmark: str) -> Path:
    return Path(output_dir) / "eval_cache" / f"{benchmark}.jsonl"


def load_cache(output_dir: str, benchmark: str) -> dict[int, dict]:
    """Load cached results keyed by step."""
    p = cache_path(output_dir, benchmark)
    if not p.exists():
        return {}
    cached = {}
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                cached[entry["step"]] = entry
            except (json.JSONDecodeError, KeyError):
                pass
    return cached


def append_cache(output_dir: str, benchmark: str, step: int, metrics: dict) -> None:
    p = cache_path(output_dir, benchmark)
    p.parent.mkdir(parents=True, exist_ok=True)
    entry = {"step": step, **metrics}
    with open(p, "a") as f:
        f.write(json.dumps(entry) + "\n")


def run_eval(
    checkpoint_path: str,
    benchmark: str,
    num_samples: int,
    batch_size: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    use_chat_template: bool,
) -> dict:
    """Run eval_benchmark.evaluate and return the metrics dict."""
    args = SimpleNamespace(
        checkpoint=checkpoint_path,
        benchmark=benchmark,
        num_samples=num_samples,
        num_problems=None,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        use_chat_template=use_chat_template,
        save_texts=False,
        output_dir=None,  # we handle caching ourselves
    )
    return eval_checkpoint(args)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def sweep(args: argparse.Namespace) -> None:
    output_dir = str(Path(args.output_dir).resolve())

    # --- Discover checkpoints ---
    all_checkpoints = discover_checkpoints(output_dir)
    if not all_checkpoints:
        print(f"No step_N checkpoints found in {output_dir}.")
        sys.exit(1)

    # Apply filters
    checkpoints = [
        (step, path)
        for step, path in all_checkpoints
        if (args.min_step is None or step >= args.min_step)
        and (args.max_step is None or step <= args.max_step)
    ]
    if args.every_n > 1:
        checkpoints = checkpoints[:: args.every_n]

    print(
        f"Found {len(all_checkpoints)} checkpoints in {output_dir}, "
        f"will evaluate {len(checkpoints)} after filtering."
    )

    if args.dry_run:
        print("\nDry run — checkpoints that would be evaluated:")
        for step, path in checkpoints:
            print(f"  step={step:>8}  {path}")
        return

    # --- Find W&B run ---
    wandb_info = find_wandb_run(output_dir, wandb_dir=args.wandb_dir)
    if wandb_info is None:
        print(
            f"[!] Could not find a W&B run whose output_dir matches {output_dir}.\n"
            f"    Searched in: {args.wandb_dir}\n"
            f"    Results will still be cached locally. Pass --wandb_run_id and\n"
            f"    --wandb_project to upload manually."
        )
        wandb_run_id = args.wandb_run_id
        wandb_project = args.wandb_project or "EBP"
        wandb_run_name = args.wandb_run_name
    else:
        wandb_run_id = args.wandb_run_id or wandb_info["run_id"]
        wandb_project = args.wandb_project or wandb_info["project"]
        wandb_run_name = args.wandb_run_name or wandb_info["run_name"]
        print(
            f"Found W&B run: {wandb_run_name!r} (id={wandb_run_id}) "
            f"in project {wandb_project!r}"
        )

    # --- Init W&B ---
    run = wandb.init(
        project=wandb_project,
        id=wandb_run_id,
        name=wandb_run_name,
        resume="allow",
        settings=wandb.Settings(init_timeout=300),
    )

    # Set up custom x-axis so eval metrics align with training steps
    run.define_metric("eval_step")
    for bench in args.benchmarks:
        run.define_metric(f"eval/{bench}/*", step_metric="eval_step")

    print(
        f"W&B run initialized: {run.url}\n"
        f"Benchmarks: {args.benchmarks}\n"
        f"Samples per problem: {args.num_samples}\n"
    )

    # --- Evaluation loop ---
    for i, (step, ckpt_path) in enumerate(checkpoints):
        print(
            f"\n{'='*60}\n"
            f"[{i+1}/{len(checkpoints)}] step={step}  path={ckpt_path}\n"
            f"{'='*60}"
        )

        log_payload: dict = {"eval_step": step}

        for bench in args.benchmarks:
            # Check cache
            cache = {} if args.no_cache else load_cache(output_dir, bench)
            if step in cache and not args.no_cache:
                metrics = cache[step]
                print(f"  [{bench}] Using cached result: pass@1={metrics.get('pass_at_1'):.4f}")
            else:
                print(f"  [{bench}] Running evaluation...")
                metrics = run_eval(
                    checkpoint_path=ckpt_path,
                    benchmark=bench,
                    num_samples=args.num_samples,
                    batch_size=args.batch_size,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    use_chat_template=args.use_chat_template,
                )
                append_cache(output_dir, bench, step, metrics)

            # Collect W&B log payload for this benchmark
            for metric_key in ("pass_at_1", "pass_at_4", "pass_at_8", "pass_at_16",
                               "majority_vote_accuracy"):
                if metric_key in metrics:
                    log_payload[f"eval/{bench}/{metric_key}"] = metrics[metric_key]

        run.log(log_payload)
        print(f"  Logged to W&B at eval_step={step}")

    wandb.finish()
    print(f"\nSweep complete. Results at: {run.url}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all checkpoints in a training output dir and log to W&B.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "output_dir",
        help="Training output directory containing step_N checkpoint subdirectories.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["aime24"],
        choices=list(BENCHMARK_CONFIGS),
        help="One or more benchmarks to evaluate on.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=32,
        help="Samples per problem for pass@k estimation.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Sequences generated in a single forward pass.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Token budget per generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature. 0 = greedy.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Nucleus sampling top-p.",
    )
    parser.add_argument(
        "--use_chat_template",
        action="store_true",
        help="Apply the tokenizer's chat template. Use for instruction-tuned checkpoints.",
    )
    parser.add_argument(
        "--every_n",
        type=int,
        default=1,
        help="Evaluate every Nth checkpoint (e.g. --every_n 5 skips 4 out of 5).",
    )
    parser.add_argument(
        "--min_step",
        type=int,
        default=None,
        help="Skip checkpoints below this step number.",
    )
    parser.add_argument(
        "--max_step",
        type=int,
        default=None,
        help="Skip checkpoints above this step number.",
    )
    parser.add_argument(
        "--wandb_dir",
        type=str,
        default="./wandb",
        help="Local W&B run directory to search for the matching run.",
    )
    # Manual W&B overrides (used when auto-discovery fails)
    parser.add_argument("--wandb_run_id", type=str, default=None,
                        help="Override W&B run ID (auto-detected from ./wandb/ if not set).")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="Override W&B project name.")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Override W&B run name.")
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Re-evaluate every checkpoint even if cached results exist.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the list of checkpoints that would be evaluated, then exit.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    sweep(parse_args())
