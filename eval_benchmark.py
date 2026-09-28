#!/usr/bin/env python3
"""
Evaluate a model checkpoint on math competition benchmarks (AIME, AMC).

Loads a HuggingFace-format checkpoint (or model name), generates responses for
each problem, extracts integer answers via regex, and reports majority-vote
accuracy and pass@k estimates.

Usage
-----
    # Quick greedy check on AIME 2024 (1 sample per problem):
    python eval_benchmark.py ./cpt_output/step_500000 --num_samples 1

    # Full pass@32 evaluation on AIME 2024:
    python eval_benchmark.py ./cpt_output/step_500000 --num_samples 32

    # Compare against the base model:
    python eval_benchmark.py Qwen/Qwen3-0.6B --num_samples 32

    # Subset of problems for quick sanity check:
    python eval_benchmark.py ./cpt_output/step_500000 --num_problems 5

    # Evaluate on AMC:
    python eval_benchmark.py ./cpt_output/final --benchmark amc --num_samples 8
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from datetime import datetime
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Benchmark registry
# ---------------------------------------------------------------------------

def _parse_int(x) -> int:
    return int(str(x).strip())

def _parse_float_int(x) -> int:
    return int(float(str(x).strip()))

def _parse_gsm8k(x) -> int:
    return int(x.split("####")[-1].strip().replace(",", ""))


BENCHMARK_CONFIGS: dict[str, dict] = {
    "aime24": {
        "dataset": "AI-MO/aimo-validation-aime",
        "split": "train",
        "filter_fn": lambda ex: "2024" in ex.get("url", ""),
        "problem_key": "problem",
        "answer_key": "answer",
        "answer_parse_fn": _parse_int,
        "answer_min": 0,
        "answer_max": 999,
        "prompt_prefix": "Solve the following math competition problem step by step.",
        "description": "AIME 2024 (I + II, 30 problems)",
    },
    "aime23": {
        "dataset": "AI-MO/aimo-validation-aime",
        "split": "train",
        "filter_fn": lambda ex: "2023" in ex.get("url", ""),
        "problem_key": "problem",
        "answer_key": "answer",
        "answer_parse_fn": _parse_int,
        "answer_min": 0,
        "answer_max": 999,
        "prompt_prefix": "Solve the following math competition problem step by step.",
        "description": "AIME 2023 (I + II, 30 problems)",
    },
    "aime_all": {
        "dataset": "AI-MO/aimo-validation-aime",
        "split": "train",
        "filter_fn": None,
        "problem_key": "problem",
        "answer_key": "answer",
        "answer_parse_fn": _parse_int,
        "answer_min": 0,
        "answer_max": 999,
        "prompt_prefix": "Solve the following math competition problem step by step.",
        "description": "AIME 2022–2024 (all available)",
    },
    "amc": {
        "dataset": "AI-MO/aimo-validation-amc",
        "split": "train",
        "filter_fn": None,
        "problem_key": "problem",
        "answer_key": "answer",
        "answer_parse_fn": _parse_float_int,
        "answer_min": -10_000,
        "answer_max": 10_000,
        "prompt_prefix": "Solve the following math competition problem step by step.",
        "description": "AMC validation set (83 problems)",
    },
    "gsm8k": {
        "dataset": "openai/gsm8k",
        "dataset_config": "main",
        "split": "test",
        "filter_fn": None,
        "problem_key": "question",
        "answer_key": "answer",
        "answer_parse_fn": _parse_gsm8k,
        "answer_min": -1_000_000,
        "answer_max": 10_000_000,
        "prompt_prefix": "Solve the following math problem step by step.",
        "max_problems": 500,  # cap to keep eval time reasonable
        "description": "GSM8K (500 problems, grade-school math)",
    },
}


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(
    text: str,
    min_val: int = 0,
    max_val: int = 999,
) -> Optional[int]:
    """Parse an integer answer from generated text.

    Priority order:
      1. Last \\boxed{...} expression.
      2. Explicit "the answer is N" / "answer: N" phrase.
      3. Last integer within [min_val, max_val] appearing after an '=' sign.
    """
    def in_range(v: int) -> bool:
        return min_val <= v <= max_val

    # 1. \boxed{...} — highest confidence
    boxed_matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed_matches:
        raw = boxed_matches[-1].strip().replace(",", "").replace(" ", "")
        try:
            val = int(raw)
            if in_range(val):
                return val
        except ValueError:
            pass

    # 2. Explicit answer declaration
    decl = re.findall(
        r"(?:the\s+answer\s+is|answer\s*[:=])\s*\**(-?\d[\d,]*)\**",
        text,
        re.IGNORECASE,
    )
    if decl:
        try:
            val = int(decl[-1].replace(",", ""))
            if in_range(val):
                return val
        except ValueError:
            pass

    # 3. Last "= N" within range
    eq_matches = re.findall(r"=\s*(-?\d[\d,]*)\b", text)
    for raw in reversed(eq_matches):
        try:
            val = int(raw.replace(",", ""))
            if in_range(val):
                return val
        except ValueError:
            pass

    return None


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021).

    Args:
        n: Total samples generated.
        c: Number of correct samples.
        k: Threshold (report probability that at least one of k is correct).
    """
    if k > n:
        return 1.0 if c > 0 else 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def majority_vote(answers: list[Optional[int]]) -> Optional[int]:
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert mathematician. Solve competition math problems carefully, "
    "showing your reasoning step by step. Always end with your final integer answer "
    "inside \\boxed{}, for example \\boxed{42}."
)


def build_prompt(
    problem: str,
    tokenizer: AutoTokenizer,
    use_chat_template: bool,
    prompt_prefix: str = "Solve the following math problem step by step.",
) -> str:
    user_msg = (
        f"{prompt_prefix} "
        f"Put your final integer answer inside \\boxed{{}}.\n\n"
        f"Problem: {problem}"
    )
    if use_chat_template and getattr(tokenizer, "chat_template", None) is not None:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    # Base model: simple completion prefix
    return f"{SYSTEM_PROMPT}\n\n{user_msg}\n\nSolution:\n"


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> list[str]:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature if args.temperature > 0 else 1.0,
            top_p=args.top_p,
            do_sample=args.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    return tokenizer.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load model ---
    print(f"Loading checkpoint: {args.checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Required for batched left-padded generation

    torch_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint,
            dtype=torch_dtype,
            device_map=device,
            attn_implementation="flash_attention_2",
        )
    except (ValueError, ImportError):
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint,
            dtype=torch_dtype,
            device_map=device,
        )
    model.eval()
    print(f"Model loaded ({torch_dtype})")

    # --- Load benchmark ---
    cfg = BENCHMARK_CONFIGS[args.benchmark]
    answer_parse_fn = cfg.get("answer_parse_fn", _parse_int)
    answer_min = cfg.get("answer_min", 0)
    answer_max = cfg.get("answer_max", 999)
    prompt_prefix = cfg.get("prompt_prefix", "Solve the following math problem step by step.")

    print(f"Loading benchmark: {cfg['description']}")
    load_kwargs = {"split": cfg["split"], "trust_remote_code": True}
    if "dataset_config" in cfg:
        load_kwargs["name"] = cfg["dataset_config"]
    raw = load_dataset(cfg["dataset"], **load_kwargs)
    if cfg["filter_fn"] is not None:
        raw = raw.filter(cfg["filter_fn"])
    problems = list(raw)

    # Config-level cap (e.g. GSM8K), then CLI --num_problems
    if cfg.get("max_problems") is not None:
        problems = problems[: cfg["max_problems"]]
    if args.num_problems is not None:
        problems = problems[: args.num_problems]

    print(f"  {len(problems)} problems to evaluate")
    print(
        f"  {args.num_samples} sample(s) per problem | "
        f"batch_size={args.batch_size} | "
        f"temp={args.temperature} | "
        f"max_new_tokens={args.max_new_tokens}"
    )

    # --- Per-problem evaluation ---
    results = []

    for i, ex in enumerate(problems):
        problem_text: str = ex[cfg["problem_key"]]
        ground_truth = answer_parse_fn(ex[cfg["answer_key"]])
        source = ex.get("source") or ex.get("url", f"problem {i+1}")
        prompt = build_prompt(problem_text, tokenizer, args.use_chat_template, prompt_prefix)

        all_answers: list[Optional[int]] = []
        all_texts: list[str] = []
        remaining = args.num_samples

        while remaining > 0:
            batch_n = min(args.batch_size, remaining)
            texts = generate_batch(model, tokenizer, [prompt] * batch_n, args, device)
            for text in texts:
                all_answers.append(extract_answer(text, answer_min, answer_max))
                all_texts.append(text)
            remaining -= batch_n

        correct_count = sum(1 for a in all_answers if a == ground_truth)
        mv = majority_vote(all_answers)
        mv_correct = mv == ground_truth

        result: dict = {
            "index": i,
            "source": source,
            "ground_truth": ground_truth,
            "extracted_answers": all_answers,
            "majority_vote": mv,
            "majority_vote_correct": mv_correct,
            "correct_count": correct_count,
            "pass_at_1": pass_at_k(args.num_samples, correct_count, 1),
        }
        if args.num_samples >= 4:
            result["pass_at_4"] = pass_at_k(args.num_samples, correct_count, 4)
        if args.num_samples >= 8:
            result["pass_at_8"] = pass_at_k(args.num_samples, correct_count, 8)
        if args.num_samples >= 16:
            result["pass_at_16"] = pass_at_k(args.num_samples, correct_count, 16)
        if args.save_texts:
            result["generated_texts"] = all_texts

        results.append(result)

        tick = "+" if mv_correct else "-"
        parse_rate = sum(1 for a in all_answers if a is not None) / len(all_answers)
        print(
            f"  [{tick}] {i+1:2d}/{len(problems)}  gt={ground_truth:3d}  "
            f"mv={str(mv):>3}  correct={correct_count}/{args.num_samples}  "
            f"parsed={parse_rate:.0%}  ({source})"
        )

    # --- Aggregate metrics ---
    n_probs = len(results)
    if n_probs == 0:
        print(f"[!] No problems matched benchmark '{args.benchmark}'. Check filter / dataset.")
        return {"benchmark": args.benchmark, "checkpoint": args.checkpoint, "num_problems": 0}
    mv_acc = sum(r["majority_vote_correct"] for r in results) / n_probs
    avg_pass1 = sum(r["pass_at_1"] for r in results) / n_probs

    metrics: dict = {
        "benchmark": args.benchmark,
        "checkpoint": args.checkpoint,
        "num_problems": n_probs,
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "majority_vote_accuracy": round(mv_acc, 4),
        "pass_at_1": round(avg_pass1, 4),
    }
    if args.num_samples >= 4:
        metrics["pass_at_4"] = round(
            sum(r.get("pass_at_4", 0) for r in results) / n_probs, 4
        )
    if args.num_samples >= 8:
        metrics["pass_at_8"] = round(
            sum(r.get("pass_at_8", 0) for r in results) / n_probs, 4
        )
    if args.num_samples >= 16:
        metrics["pass_at_16"] = round(
            sum(r.get("pass_at_16", 0) for r in results) / n_probs, 4
        )

    print("\n" + "=" * 60)
    print(f"  Benchmark : {cfg['description']}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Problems  : {n_probs}")
    print(f"  Majority vote accuracy: {mv_acc:.1%}  ({int(mv_acc * n_probs)}/{n_probs})")
    print(f"  Pass@1    : {avg_pass1:.4f}")
    for k in (4, 8, 16):
        key = f"pass_at_{k}"
        if key in metrics:
            print(f"  Pass@{k:<4}: {metrics[key]:.4f}")
    print("=" * 60)

    # --- Save results ---
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_name = os.path.basename(args.checkpoint.rstrip("/")) or "model"
        fname = f"{args.benchmark}_{ckpt_name}_{ts}.json"
        out_path = os.path.join(args.output_dir, fname)
        with open(out_path, "w") as f:
            json.dump({"metrics": metrics, "results": results}, f, indent=2)
        print(f"Results saved to {out_path}")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a model checkpoint on math competition benchmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "checkpoint",
        help="Local checkpoint directory (e.g. ./cpt_output/step_500000) "
             "or HuggingFace model name (e.g. Qwen/Qwen3-0.6B).",
    )
    parser.add_argument(
        "--benchmark",
        default="aime24",
        choices=list(BENCHMARK_CONFIGS),
        help="Which benchmark to evaluate on.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=32,
        help="Samples per problem. Larger values give better pass@k estimates.",
    )
    parser.add_argument(
        "--num_problems",
        type=int,
        default=None,
        help="Evaluate only the first N problems (useful for quick sanity checks).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Sequences to generate in a single forward pass.",
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
        help="Sampling temperature. Set to 0 for greedy decoding.",
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
        help="Wrap prompts in the tokenizer's chat template if available. "
             "Use this for instruction-tuned checkpoints.",
    )
    parser.add_argument(
        "--save_texts",
        action="store_true",
        help="Include raw generated texts in the output JSON (increases file size).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Directory for JSON result files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
