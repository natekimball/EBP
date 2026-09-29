"""
Benchmark script for comparing EBP training throughput under different configurations.

This script measures tokens/sec throughput for:
  * ``online + gamma=0`` (rollout-policy only, single forward pass per rollout)
  * ``online + gamma=X`` (mixed objective, two forward passes per rollout)
  * ``ema + gamma=X`` (EMA variant, two passes for EMA+gen)

Run with ``--help`` for full usage.

Example::

    # Benchmark online variant with gamma=0 (max rollout throughput)
    python benchmark.py \\
        --model_name Qwen/Qwen3-0.6B \\
        --model_type online \\
        --gamma 0.0 \\
        --num_steps 100

    # Compare with mixed objective (gamma=0.1)
    python benchmark.py \\
        --model_name Qwen/Qwen3-0.6B \\
        --model_type online \\
        --gamma 0.1 \\
        --num_steps 100
"""

import argparse
import time
from typing import Union

import torch
from transformers import AutoTokenizer

from ebp.data import PretrainingDataset, collate_fn
from ebp.model import EMAEBPModel, OnlineEBPModel
from ebp.rewards import compute_feature_matching_terms_batched, compute_rloo_baseline_batched
from train import training_step, memory_constrained_training_step, GPUPrefetcher
import os
from functools import partial
from torch.utils.data import DataLoader


def parse_benchmark_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark EBP training throughput under different configurations."
    )
    # Model
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace model identifier.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="online",
        choices=["ema", "online"],
    )
    # Data
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="allenai/dolma",
    )
    parser.add_argument("--dataset_config", type=str, default="v1_7")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument(
        "--context_length", type=int, default=128,
    )
    parser.add_argument(
        "--generation_length", type=int, default=8,
    )
    # EBP hyperparameters
    parser.add_argument("--num_rollouts", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    # Benchmark settings
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "bfloat16", "float16"],
    )
    parser.add_argument(
        "--memory_constrained",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--pin_memory",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
    )
    return parser.parse_args()


def benchmark():
    args = parse_benchmark_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.dtype == "auto":
        if device.type == "cuda":
            torch_dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
        else:
            torch_dtype = torch.float32
    else:
        dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        torch_dtype = dtype_map[args.dtype]

    print(f"Using dtype: {torch_dtype}")

    # Tokeniser
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    # Model
    print(f"Loading model: {args.model_name} (variant: {args.model_type})...")
    from transformers import AutoModelForCausalLM

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2" if device.type == "cuda" else None,
    ).to(device)

    if args.model_type == "ema":
        model: Union[EMAEBPModel, OnlineEBPModel] = EMAEBPModel(
            model=base_model,
            ema_decay=args.ema_decay,
        )
    else:
        model = OnlineEBPModel(model=base_model)

    # Dataset & DataLoader
    print(f"Loading dataset: {args.dataset_name}/{args.dataset_config}...")
    dataset = PretrainingDataset(
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.dataset_split,
        context_length=args.context_length,
        completion_length=args.generation_length,
        streaming=True,
        max_documents=None,
        max_tokens=None,
        max_examples=None,
    )

    pin_memory = args.pin_memory if args.pin_memory is not None else device.type == "cuda"
    if args.num_workers is None:
        num_workers = 0 if device.type == "cpu" else max(1, min(4, (os.cpu_count() or 1) // 2))
    else:
        num_workers = max(0, args.num_workers)

    dataloader_kwargs = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": False,  # Don't need shuffling for benchmark
        "collate_fn": partial(collate_fn, pad_token_id=pad_id),
        "drop_last": True,
        "pin_memory": pin_memory,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = 2

    dataloader = DataLoader(**dataloader_kwargs)
    prefetched_loader = GPUPrefetcher(dataloader, device)

    # Optimizer (for making updates, though we won't track loss for benchmark)
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-5)

    step_fn = memory_constrained_training_step if args.memory_constrained else training_step

    print(
        f"\nBenchmarking configuration:"
        f"\n  Model type: {args.model_type}"
        f"\n  Gamma (CE weight): {args.gamma}"
        f"\n  Batch size: {args.batch_size}"
        f"\n  Num rollouts: {args.num_rollouts}"
        f"\n  Generation length: {args.generation_length}"
        f"\n  Context length: {args.context_length}"
        f"\n  Training steps: {args.num_steps}"
        f"\n  Warmup steps: {args.warmup_steps}"
    )

    # Benchmark loop
    times = []
    tokens_generated = 0

    for step in range(args.num_steps):
        for batch in prefetched_loader:
            optimizer.zero_grad(set_to_none=True)

            torch.cuda.synchronize(device) if device.type == "cuda" else None
            step_start = time.perf_counter()

            result = step_fn(
                model=model,
                batch=batch,
                num_rollouts=args.num_rollouts,
                generation_length=args.generation_length,
                gamma=args.gamma,
                temperature=args.temperature,
                device=device,
                log_cuda_memory=False,
            )

            torch.cuda.synchronize(device) if device.type == "cuda" else None
            step_time = time.perf_counter() - step_start

            # Track time (skip warmup steps)
            if step >= args.warmup_steps:
                times.append(step_time)

            # Count tokens generated per step
            # tokens = batch_size * num_rollouts * generation_length
            batch_tokens = (
                args.batch_size * args.num_rollouts * args.generation_length
            )
            tokens_generated += batch_tokens

            optimizer.step()

            if (step + 1) % max(1, args.num_steps // 5) == 0:
                print(
                    f"Step {step + 1:3d}/{args.num_steps} | "
                    f"step_time={step_time:.3f}s | "
                    f"loss={result['loss']:.4f}"
                )

            break  # Only one batch per step for benchmarking

    # Compute statistics
    if times:
        avg_time = sum(times) / len(times)
        throughput = args.batch_size * args.num_rollouts * args.generation_length / avg_time

        print(
            f"\n{'='*60}"
            f"\nBenchmark Results (after {args.warmup_steps} warmup steps):"
            f"\n{'='*60}"
            f"\n  Avg step time: {avg_time:.3f}s"
            f"\n  Tokens/sec: {throughput:.1f}"
            f"\n  Total tokens generated: {tokens_generated}"
            f"\n{'='*60}"
        )


if __name__ == "__main__":
    benchmark()
