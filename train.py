"""
Energy-Based Pre-training (EBP) training script.

Trains Qwen/Qwen3-0.6B (or any compatible causal LM) with a mixed objective:

    L(θ) = L_FM(θ)  +  γ · L_CE(θ)

where:
  * L_FM  is the feature-matching loss, optimised via REINFORCE with RLOO
           baseline using rollouts sampled from the generator.
  * L_CE  is the standard next-token cross-entropy loss (teacher forcing).

The feature network ϕ is an Exponential Moving Average (EMA) of the
generator: as the generator improves, so do the features used to evaluate
rollout quality.  EMA weights are updated with stop-gradient after each
optimiser step.

Usage
-----
    python train.py \\
        --model_name Qwen/Qwen3-0.6B \\
        --num_rollouts 4 \\
        --generation_length 8 \\
        --gamma 0.1 \\
        --max_steps 10000

See ``python train.py --help`` for the full list of arguments.
"""

from __future__ import annotations

import argparse
import os
from functools import partial

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ebp.data import PretrainingDataset, collate_fn
from ebp.model import EBPModel
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
        help="EMA decay factor τ (closer to 1 → slower EMA update).",
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
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "bfloat16", "float16"],
        help="Model parameter dtype.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def training_step(
    model: EBPModel,
    batch: dict,
    num_rollouts: int,
    generation_length: int,
    gamma: float,
    temperature: float,
    device: torch.device,
) -> dict:
    """Execute one EBP training step.

    1. Extract reference features from the EMA model.
    2. Generate ``num_rollouts`` completions per context.
    3. Extract rollout features from the EMA model.
    4. Compute per-rollout feature-matching rewards + RLOO advantages.
    5. Compute REINFORCE loss and (optionally) cross-entropy loss.

    Args:
        model: :class:`EBPModel` with trainable generator and EMA network.
        batch: Dict of tensors from :func:`ebp.data.collate_fn`.
        num_rollouts: Number of completions to generate per context.
        generation_length: Number of tokens to generate.
        gamma: Weight of the cross-entropy term (0 → feature-matching only).
        temperature: Sampling temperature.
        device: Target device.

    Returns:
        Dict with ``loss``, ``reinforce_loss``, and (if ``gamma > 0``)
        ``ce_loss`` as Python floats for logging.
    """
    context_ids = batch["context_ids"].to(device)
    context_mask = batch["context_mask"].to(device)
    completion_ids = batch["completion_ids"].to(device)
    completion_mask = batch["completion_mask"].to(device)

    batch_size = context_ids.shape[0]
    context_len = context_ids.shape[1]

    # ------------------------------------------------------------------
    # 1. Reference features  φ(c:y)  from EMA model
    # ------------------------------------------------------------------
    full_ids = torch.cat([context_ids, completion_ids], dim=1)
    full_mask = torch.cat([context_mask, completion_mask], dim=1)

    ref_features = model.extract_features(
        full_ids, full_mask, completion_start=context_len
    )  # (B, feat_dim)

    # ------------------------------------------------------------------
    # 2. Generate rollouts  ŷ_j ~ p_θ(·|c)
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
    # 3. Rollout features  φ(c:ŷ_j)  from EMA model
    # ------------------------------------------------------------------
    rollout_features = model.extract_features(
        rollout_ids, rollout_masks, completion_start=context_len
    )  # (B * n, feat_dim)

    # ------------------------------------------------------------------
    # 4. Compute rewards and REINFORCE loss
    # ------------------------------------------------------------------
    reinforce_loss = torch.tensor(0.0, device=device)

    for i in range(batch_size):
        # Features for the i-th context's rollouts
        item_rollout_feat = rollout_features[i * num_rollouts : (i + 1) * num_rollouts]
        item_ref_feat = ref_features[i]  # (feat_dim,)

        rewards = compute_feature_matching_rewards(item_rollout_feat, item_ref_feat)
        baselines = compute_rloo_baseline(rewards)
        advantages = (rewards - baselines).detach()  # stop-gradient on advantages

        # Log-probabilities under the trainable generator
        item_rollout_ids = rollout_ids[i * num_rollouts : (i + 1) * num_rollouts]
        item_rollout_masks = rollout_masks[i * num_rollouts : (i + 1) * num_rollouts]

        log_probs = model.compute_log_probs(
            item_rollout_ids, item_rollout_masks, completion_start=context_len
        )  # (n,)

        # REINFORCE: maximise E[advantage * log p]  →  minimise negation
        reinforce_loss = reinforce_loss - (advantages * log_probs).mean()

    reinforce_loss = reinforce_loss / batch_size

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
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    torch_dtype = dtype_map[args.dtype]

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
    print(f"Loading model: {args.model_name} ...")
    from transformers import AutoModelForCausalLM

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch_dtype
    ).to(device)

    model = EBPModel(
        model=base_model,
        ema_decay=args.ema_decay,
    )

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

    step = 0
    model.train()

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

            optimizer.zero_grad()
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), args.grad_clip)
            optimizer.step()

            # Update EMA feature network (stop-gradient)
            model.update_ema()

            step += 1

            if step % args.log_steps == 0:
                print(
                    f"Step {step:6d} | loss={result['loss'].item():.4f} "
                    f"reinforce={result['reinforce_loss']:.4f} "
                    f"ce={result['ce_loss']:.4f}"
                )

            if step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"step_{step}")
                model.model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"Saved checkpoint to {save_path}")

    # ------------------------------------------------------------------
    # Final save
    # ------------------------------------------------------------------
    final_path = os.path.join(args.output_dir, "final")
    model.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Training complete. Final model saved to {final_path}")


if __name__ == "__main__":
    train(parse_args())
