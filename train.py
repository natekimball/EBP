"""
Energy-Based Pre-training (EBP) training script.

Trains Qwen/Qwen3-0.6B (or any compatible causal LM) with a mixed objective:

    L(theta) = L_FM(theta)  +  gamma * L_CE(theta)

where:
  * L_FM  is the feature-matching loss, optimised via REINFORCE with RLOO
           baseline using rollouts sampled from the generator.
  * L_CE  is the standard next-token cross-entropy loss (teacher forcing).

Two model variants are available via ``--model_type``:

  * ``ema``    (:class:`~ebp.model.EMAEBPModel`) – uses an Exponential Moving
    Average of the generator as the feature network.  Requires two forward
    passes per training step (EMA for features, generator for log-probs).

  * ``online`` (:class:`~ebp.model.OnlineEBPModel`) – uses the live generator
    as the feature network.  A single forward pass extracts features (detached)
    and log-probs simultaneously, halving rollout-processing cost.

Both variants use a prefix KV cache during rollout generation so that the
context is processed only once regardless of ``--num_rollouts``.

Metrics
-------
Every step the following scalars are logged to stdout and accumulated in a
list of dicts that is pickled to ``<output_dir>/metrics.pkl`` (override with
``--metrics_file``)::

    step          – global training step
    loss          – total loss (reinforce + gamma * ce)
    reinforce_loss – REINFORCE policy-gradient term
    ce_loss       – cross-entropy term (0 when --gamma 0)
    mean_reward   – mean feature-matching reward across all rollouts
    entropy       – mean per-token negative log-prob of rollout sequences
                    (proxy for policy entropy; higher = more exploratory)

To plot the metrics after training::

    import pickle, matplotlib.pyplot as plt
    history = pickle.load(open("output/metrics.pkl", "rb"))
    steps   = [m["step"] for m in history]
    plt.plot(steps, [m["mean_reward"] for m in history], label="reward")
    plt.plot(steps, [m["entropy"]     for m in history], label="entropy")
    plt.legend(); plt.show()

Usage
-----
    # EMA variant (feature network = EMA of generator):
    python train.py \\
        --model_name Qwen/Qwen3-0.6B \\
        --model_type ema \\
        --num_rollouts 4 \\
        --generation_length 8 \\
        --gamma 0.1 \\
        --max_steps 10000

    # Online variant (feature network = live generator, one forward pass):
    python train.py \\
        --model_name Qwen/Qwen3-0.6B \\
        --model_type online \\
        --num_rollouts 4 \\
        --generation_length 8 \\
        --gamma 0.1 \\
        --max_steps 10000

See ``python train.py --help`` for the full list of arguments.
"""

from __future__ import annotations

import argparse
import os
import pickle
from functools import partial
from typing import List, Union

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ebp.data import PretrainingDataset, collate_fn
from ebp.model import EMAEBPModel, OnlineEBPModel
from ebp.rewards import compute_feature_matching_rewards, compute_rloo_baseline


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Energy-Based Pre-training (EBP) for causal language models."
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
        help=(
            "Feature-network variant: 'ema' uses an EMA copy of the generator; "
            "'online' uses the live generator itself (one forward pass per step, "
            "lower VRAM, usually higher throughput)."
        ),
    )
    # Data
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--dataset_split", type=str, default="train")
    # Sequence lengths
    parser.add_argument(
        "--context_length",
        type=int,
        default=128,
        help="Number of context tokens fed to the generator.",
    )
    parser.add_argument(
        "--generation_length",
        type=int,
        default=8,
        help="Number of tokens generated per rollout.",
    )
    # EBP hyperparameters
    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=4,
        help="Number of rollouts sampled per context.",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.999,
        help="EMA decay factor (only used when --model_type ema).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="Weight of the cross-entropy term in the mixed objective.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for rollout generation.",
    )
    # Optimiser
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument(
        "--grad_clip", type=float, default=1.0, help="Max gradient norm."
    )
    # Training schedule
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=10_000)
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=1_000)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument(
        "--metrics_file",
        type=str,
        default=None,
        help=(
            "Path to save the metrics history as a pickle file. "
            "Defaults to <output_dir>/metrics.pkl."
        ),
    )
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "bfloat16", "float16"],
        help=(
            "Model parameter dtype. 'auto' picks bf16 on supported CUDA GPUs, "
            "else fp16 on CUDA, else fp32 on CPU."
        ),
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help=(
            "Enable gradient checkpointing on the trainable generator to reduce "
            "activation memory (slight speed tradeoff)."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def training_step(
    model: Union[EMAEBPModel, OnlineEBPModel],
    batch: dict,
    num_rollouts: int,
    generation_length: int,
    gamma: float,
    temperature: float,
    device: torch.device,
) -> dict:
    """Execute one EBP training step.

    1. Extract reference features from the feature network (EMA or live model).
    2. Generate ``num_rollouts`` completions per context (with prefix KV cache).
    3. Compute rollout features + log-probs via ``model.compute_rollout_data``
       (one forward pass for OnlineEBPModel; two for EMAEBPModel).
    4. Compute REINFORCE loss with RLOO advantages.
    5. Optionally add the cross-entropy term.

    Args:
        model: Either :class:`EMAEBPModel` or :class:`OnlineEBPModel`.
        batch: Dict of tensors from :func:`ebp.data.collate_fn`.
        num_rollouts: Number of completions to generate per context.
        generation_length: Number of tokens to generate.
        gamma: Weight of the cross-entropy term (0 -> feature-matching only).
        temperature: Sampling temperature.
        device: Target device.

    Returns:
        Dict with keys ``loss``, ``reinforce_loss``, ``ce_loss``,
        ``mean_reward``, and ``entropy`` (mean per-token negative log-prob of
        rollout sequences, a proxy for the policy entropy).
    """
    context_ids = batch["context_ids"].to(device)
    context_mask = batch["context_mask"].to(device)
    completion_ids = batch["completion_ids"].to(device)
    completion_mask = batch["completion_mask"].to(device)

    batch_size = context_ids.shape[0]
    context_len = context_ids.shape[1]

    # ------------------------------------------------------------------
    # 1. Reference features  phi(c:y)
    # ------------------------------------------------------------------
    full_ids = torch.cat([context_ids, completion_ids], dim=1)
    full_mask = torch.cat([context_mask, completion_mask], dim=1)

    ref_features = model.extract_features(
        full_ids, full_mask, completion_start=context_len
    )  # (B, feat_dim)

    # ------------------------------------------------------------------
    # 2. Generate rollouts  y_hat_j ~ p_theta(.|c)
    # ------------------------------------------------------------------
    rollout_ids, rollout_masks = model.generate_rollouts(
        context_ids=context_ids,
        context_attention_mask=context_mask,
        num_rollouts=num_rollouts,
        generation_length=generation_length,
        temperature=temperature,
    )
    # rollout_ids:   (B * n, context_len + generation_length)
    # rollout_masks: (B * n, context_len + generation_length)

    # ------------------------------------------------------------------
    # 3. Rollout features + log-probs
    #    EMAEBPModel:    two forward passes (EMA features + generator log-probs)
    #    OnlineEBPModel: one forward pass (features detached + log-probs with grad)
    # ------------------------------------------------------------------
    rollout_features, rollout_log_probs = model.compute_rollout_data(
        rollout_ids, rollout_masks, completion_start=context_len
    )
    # rollout_features:  (B * n, feat_dim)  - detached
    # rollout_log_probs: (B * n,)           - differentiable

    # ------------------------------------------------------------------
    # 4. Feature-matching rewards and REINFORCE loss
    # ------------------------------------------------------------------
    reinforce_loss = torch.tensor(0.0, device=device)
    all_rewards: List[torch.Tensor] = []

    for i in range(batch_size):
        item_feat = rollout_features[i * num_rollouts : (i + 1) * num_rollouts]
        item_ref = ref_features[i]  # (feat_dim,)
        item_lp = rollout_log_probs[i * num_rollouts : (i + 1) * num_rollouts]

        rewards = compute_feature_matching_rewards(item_feat, item_ref)
        all_rewards.append(rewards.detach())
        baselines = compute_rloo_baseline(rewards)
        advantages = (rewards - baselines).detach()  # stop-gradient on advantages

        # REINFORCE: maximise E[advantage * log p]  ->  minimise negation
        reinforce_loss = reinforce_loss - (advantages * item_lp).mean()

    reinforce_loss = reinforce_loss / batch_size

    # Mean reward across all rollouts (diagnostic)
    mean_reward = torch.cat(all_rewards).mean().item()

    # Per-token NLL of rollout sequences (proxy for policy entropy).
    # NLL = -mean(log p) / gen_len.  By Jensen's inequality, H(p) ≥ NLL,
    # so this is a lower bound on entropy: higher = more exploratory policy.
    policy_nll = (-rollout_log_probs.detach().mean() / max(generation_length, 1)).item()

    # ------------------------------------------------------------------
    # 5. Optional cross-entropy term
    # ------------------------------------------------------------------
    total_loss = reinforce_loss
    ce_loss_val = 0.0

    if gamma > 0.0:
        ce_out = model(input_ids=full_ids, attention_mask=full_mask, labels=full_ids)
        ce_loss = ce_out.loss
        total_loss = total_loss + gamma * ce_loss
        ce_loss_val = ce_loss.item()

    return {
        "loss": total_loss,
        "reinforce_loss": reinforce_loss.item(),
        "ce_loss": ce_loss_val,
        "mean_reward": mean_reward,
        # Logged as "entropy"; computed as per-token NLL (lower bound on H(p))
        "entropy": policy_nll,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
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

    # ------------------------------------------------------------------
    # Tokeniser
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print(f"Loading model: {args.model_name} (variant: {args.model_type}) ...")
    from transformers import AutoModelForCausalLM

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch_dtype
    ).to(device)

    if args.model_type == "ema":
        model: Union[EMAEBPModel, OnlineEBPModel] = EMAEBPModel(
            model=base_model,
            ema_decay=args.ema_decay,
        )
    else:
        model = OnlineEBPModel(model=base_model)

    if args.gradient_checkpointing:
        model.model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled on generator model")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    print(f"Loading dataset: {args.dataset_name}/{args.dataset_config} ...")
    dataset = PretrainingDataset(
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.dataset_split,
        context_length=args.context_length,
        completion_length=args.generation_length,
    )
    print(f"Dataset size: {len(dataset)} examples")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, pad_token_id=pad_id),
        drop_last=True,
    )

    # ------------------------------------------------------------------
    # Optimiser (only the trainable generator parameters)
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_file = args.metrics_file or os.path.join(args.output_dir, "metrics.pkl")

    step = 0
    model.train()
    metrics_history: List[dict] = []

    while step < args.max_steps:
        for batch in dataloader:
            if step >= args.max_steps:
                break

            result = training_step(
                model=model,
                batch=batch,
                num_rollouts=args.num_rollouts,
                generation_length=args.generation_length,
                gamma=args.gamma,
                temperature=args.temperature,
                device=device,
            )

            optimizer.zero_grad(set_to_none=True)
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), args.grad_clip)
            optimizer.step()

            # Update EMA feature network (only for EMAEBPModel)
            if isinstance(model, EMAEBPModel):
                model.update_ema()

            step += 1

            # Convert the loss tensor to a plain float now that backward() has
            # been called; all other result values are already Python floats.
            loss_val = result["loss"].item()

            # Record metrics for every step (floats only, very lightweight)
            metrics_history.append(
                {
                    "step": step,
                    "loss": loss_val,
                    "reinforce_loss": result["reinforce_loss"],
                    "ce_loss": result["ce_loss"],
                    "mean_reward": result["mean_reward"],
                    "entropy": result["entropy"],
                }
            )

            if step % args.log_steps == 0:
                print(
                    f"Step {step:6d} | loss={loss_val:.4f} "
                    f"reinforce={result['reinforce_loss']:.4f} "
                    f"ce={result['ce_loss']:.4f} "
                    f"reward={result['mean_reward']:.4f} "
                    f"entropy={result['entropy']:.4f}"
                )

            if step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"step_{step}")
                model.model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"Saved checkpoint to {save_path}")
                # Also persist metrics so far
                with open(metrics_file, "wb") as fh:
                    pickle.dump(metrics_history, fh)
                print(f"Metrics saved to {metrics_file}")

    # ------------------------------------------------------------------
    # Final save
    # ------------------------------------------------------------------
    final_path = os.path.join(args.output_dir, "final")
    model.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Training complete. Final model saved to {final_path}")

    with open(metrics_file, "wb") as fh:
        pickle.dump(metrics_history, fh)
    print(f"Metrics history saved to {metrics_file}")


if __name__ == "__main__":
    train(parse_args())
