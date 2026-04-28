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
import time
import wandb
from functools import partial
from typing import Iterator, List, Union

import torch
import torch._dynamo
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ebp.data import PretrainingDataset, collate_fn
from ebp.model import EMAEBPModel, OnlineEBPModel
from ebp.rewards import (
    compute_feature_matching_terms_batched,
    compute_rloo_baseline_batched,
    compute_whitened_feature_matching_terms_batched,
)


# ---------------------------------------------------------------------------
# GPU Prefetcher for asynchronous data movement
# ---------------------------------------------------------------------------


class GPUPrefetcher:
    """Overlaps CPU->GPU data transfer with model training.

    Fetches batch i+1 while batch i is being processed, reducing GPU idle time.
    Uses non-blocking transfers to pipeline data movement with computation.
    """

    def __init__(self, dataloader: DataLoader, device: torch.device):
        """Initialize prefetcher.

        Args:
            dataloader: PyTorch DataLoader to prefetch from.
            device: Target device (GPU).
        """
        self.dataloader = dataloader
        self.device = device
        self.stream = torch.cuda.Stream() if device.type == "cuda" else None

    def __iter__(self) -> Iterator[dict]:
        """Yield batches with prefetching."""
        dataloader_iter = iter(self.dataloader)

        # Fetch first batch in background
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            return

        if self.device.type == "cuda":
            # Pre-load first batch in prefetch stream
            with torch.cuda.stream(self.stream):
                batch = self._move_to_device_async(batch)

        for next_batch in dataloader_iter:
            # Synchronize prefetch stream if on GPU (ensures batch is ready)
            if self.device.type == "cuda":
                self.stream.synchronize()

            yield batch

            # Prefetch next batch in background while current batch trains
            if self.device.type == "cuda":
                with torch.cuda.stream(self.stream):
                    batch = self._move_to_device_async(next_batch)
            else:
                batch = self._move_to_device(next_batch)

        # Final batch synchronization
        if self.device.type == "cuda":
            self.stream.synchronize()
        yield batch

    def _move_to_device_async(self, batch: dict) -> dict:
        """Move batch to device asynchronously (in prefetch stream)."""
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def _move_to_device(self, batch: dict) -> dict:
        """Move batch to device synchronously (CPU only)."""
        return {k: v.to(self.device, non_blocking=False) for k, v in batch.items()}



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
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="allenai/dolma",
        help="Dataset name or path to a local directory containing a saved dataset.",
    )
    parser.add_argument(
        "--tokenized",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Whether the dataset is already tokenized. If not specified, "
            "it is automatically enabled if --dataset_name is a directory."
        ),
    )
    parser.add_argument("--dataset_config", type=str, default="v1_7")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument(
        "--val_split",
        type=str,
        default=None,
        help=(
            "Dataset split for validation (e.g. 'validation'). If None, no "
            "validation is performed. If set to the same as --dataset_split, "
            "a validation set is automatically carved out from the start."
        ),
    )
    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream dataset samples instead of loading split materialization upfront.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Optional cap on total concatenated tokens before chunking.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Optional cap on number of fixed-length training examples to build.",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=None,
        help="Batch size for validation. Defaults to 2x train batch_size if not specified.",
    )
    parser.add_argument(
        "--max_val_docs",
        type=int,
        default=500,
        help="Number of documents to use for validation when carving out from training split.",
    )
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
    parser.add_argument(
        "--whitening",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use whitened feature matching (Eq. 9 of the EBFT paper).",
    )
    parser.add_argument(
        "--pool_type",
        type=str,
        default="last",
        choices=["last", "mean"],
        help="Pooling strategy for hidden-state features.",
    )
    # Optimiser
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument(
        "--grad_clip", type=float, default=1.0, help="Max gradient norm."
    )
    # Training schedule
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=10_000)
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument(
        "--val_steps",
        type=int,
        default=2_000,
        help="Interval for validation steps.",
    )
    parser.add_argument(
        "--max_val_batches",
        type=int,
        default=100,
        help="Number of batches to evaluate during validation.",
    )
    parser.add_argument("--save_steps", type=int, default=10_000)
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
    # Logging
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="EBP",
        help="W&B project name.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B run name.",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B logging mode.",
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
    parser.add_argument(
        "--pin_memory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Whether to use DataLoader pinned host memory for faster CPU->GPU "
            "transfer. Defaults to enabled on CUDA and disabled on CPU."
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help=(
            "DataLoader worker count. Defaults to 0 on CPU, and to a small "
            "CPU-dependent value on CUDA for better pinned-memory reuse."
        ),
    )
    parser.add_argument(
        "--persistent_workers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Keep DataLoader workers alive between iterations. Defaults to "
            "enabled when num_workers > 0 to re-use pinned buffers."
        ),
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="Number of prefetched batches per worker when num_workers > 0.",
    )
    parser.add_argument(
        "--use_fused_adamw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use CUDA fused AdamW kernels when available (falls back "
            "automatically if unsupported)."
        ),
    )
    parser.add_argument(
        "--use_flash_attention",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Request FlashAttention v2 when loading supported models on CUDA "
            "(falls back automatically if unavailable)."
        ),
    )
    parser.add_argument(
        "--compile_model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compile the generator with torch.compile when available.",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default=None,
        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
        help=(
            "Compilation mode for torch.compile. If not specified, defaults to "
            "'default' when --memory_constrained is on, otherwise 'reduce-overhead'."
        ),
    )
    parser.add_argument(
        "--compile_fullgraph",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass fullgraph=True to torch.compile.",
    )
    parser.add_argument(
        "--log_cuda_memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Log per-stage CUDA peak memory (allocated and reserved) at the "
            "same cadence as --log_steps."
        ),
    )
    parser.add_argument(
        "--memory_constrained",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the memory-constrained training step that backpropagates CE "
            "early before rollout processing. Disable for the regular single-"
            "backward step (typically higher throughput when memory allows)."
        ),
    )
    return parser.parse_args()


def _configure_tensor_cores(device: torch.device) -> None:
    """Enable Tensor Core-friendly settings on Ampere+ CUDA GPUs."""
    if device.type != "cuda":
        return

    major, minor = torch.cuda.get_device_capability(device)
    ampere_or_newer = major >= 8
    if ampere_or_newer:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        print(
            "Enabled Tensor Core acceleration (TF32 matmul/cudnn) "
            f"for compute capability {major}.{minor}"
        )
    else:
        print(
            "Tensor Core TF32 acceleration not enabled "
            f"(compute capability {major}.{minor} is pre-Ampere)"
        )


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
    log_cuda_memory: bool = False,
    whitening: bool = True,
) -> dict:
    """Regular EBP training step with a single backward pass.

    This path is throughput-oriented: CE and REINFORCE are combined into one
    total loss and backpropagated once. For the online model with ``gamma > 0``,
    the CE/ref combined forward is intentionally delayed until *after* rollout
    generation and rollout processing to match the requested ordering.

    Args:
        model: Either :class:`EMAEBPModel` or :class:`OnlineEBPModel`.
        batch: Dict of tensors from :func:`ebp.data.collate_fn` (already on device).
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
    # Batch is already on device via GPUPrefetcher, no need for explicit transfer
    context_ids = batch["context_ids"]
    context_mask = batch["context_mask"]
    completion_ids = batch["completion_ids"]
    completion_mask = batch["completion_mask"]

    batch_size = context_ids.shape[0]
    context_len = context_ids.shape[1]

    cuda_mem = {}

    def _reset_cuda_peak(tag: str) -> None:
        if log_cuda_memory and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    def _record_cuda_peak(tag: str) -> None:
        if log_cuda_memory and device.type == "cuda":
            cuda_mem[f"{tag}_alloc_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            cuda_mem[f"{tag}_reserved_mb"] = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

    # ------------------------------------------------------------------
    # 1. Build full-sequence tensors used by ref/CE paths
    # ------------------------------------------------------------------
    full_ids = torch.cat([context_ids, completion_ids], dim=1)
    full_mask = torch.cat([context_mask, completion_mask], dim=1)

    ce_loss: torch.Tensor | None = None
    ref_features: torch.Tensor

    # ------------------------------------------------------------------
    # 2. Generate rollouts  y_hat_j ~ p_theta(.|c)
    # ------------------------------------------------------------------
    model.eval()
    _reset_cuda_peak("gen")
    rollout_ids, rollout_masks = model.generate_rollouts(
        context_ids=context_ids,
        context_attention_mask=context_mask,
        num_rollouts=num_rollouts,
        generation_length=generation_length,
        temperature=temperature,
    )
    _record_cuda_peak("gen")
    # rollout_ids:   (B * n, context_len + generation_length)
    # rollout_masks: (B * n, context_len + generation_length)

    # ------------------------------------------------------------------
    # 3. Rollout features + log-probs
    #    EMAEBPModel:    two forward passes (EMA features + generator log-probs)
    #    OnlineEBPModel: one forward pass (features detached + log-probs with grad)
    # ------------------------------------------------------------------
    model.train()
    _reset_cuda_peak("rollout_fwd")
    rollout_features, rollout_log_probs = model.compute_rollout_data(
        rollout_ids, rollout_masks, completion_start=context_len
    )
    _record_cuda_peak("rollout_fwd")
    # rollout_features:  (B * n, feat_dim)  - detached
    # rollout_log_probs: (B * n,)           - differentiable

    # ------------------------------------------------------------------
    # 4. Reference features and CE
    # ------------------------------------------------------------------
    _reset_cuda_peak("ref")
    ce_loss, ref_features = model.forward_ce_and_ref_features(
        full_ids,
        full_mask,
        completion_start=context_len,
    )
    _record_cuda_peak("ref")
    ce_loss_val = ce_loss.item()

    # ------------------------------------------------------------------
    # 5. Feature-matching rewards and REINFORCE loss
    # ------------------------------------------------------------------
    if whitening:
        alignment, diversity = compute_whitened_feature_matching_terms_batched(
            rollout_features, ref_features, num_rollouts
        )
    else:
        alignment, diversity = compute_feature_matching_terms_batched(
            rollout_features, ref_features, num_rollouts
        )
    rewards = alignment - diversity  # (B * n,)

    baselines = compute_rloo_baseline_batched(rewards, num_rollouts)
    advantages = (rewards - baselines).detach()  # stop-gradient on advantages

    # REINFORCE: maximise E[advantage * log p]  ->  minimise negation
    reinforce_loss = -(advantages * rollout_log_probs).mean()

    # Mean reward / alignment / diversity across all rollouts (diagnostics)
    mean_reward = rewards.detach().mean().item()
    mean_alignment = alignment.detach().mean().item()
    mean_diversity = diversity.detach().mean().item()

    # Per-token NLL of rollout sequences (proxy for policy entropy).
    # NLL = -mean(log p) / gen_len.  By Jensen's inequality, H(p) ≥ NLL,
    # so this is a lower bound on entropy: higher = more exploratory policy.
    policy_nll = (-rollout_log_probs.detach().mean() / max(generation_length, 1)).item()

    # ------------------------------------------------------------------
    # 6. Optional cross-entropy term
    # ------------------------------------------------------------------
    total_loss = reinforce_loss

    if gamma > 0.0:
        total_loss = total_loss + gamma * ce_loss

    _reset_cuda_peak("backward")
    total_loss.backward()
    _record_cuda_peak("backward")

    result = {
        "loss": total_loss.item(),
        "reinforce_loss": reinforce_loss.item(),
        "ce_loss": ce_loss_val,
        "mean_reward": mean_reward,
        "mean_alignment": mean_alignment,
        "mean_diversity": mean_diversity,
        # Logged as "entropy"; computed as per-token NLL (lower bound on H(p))
        "entropy": policy_nll,
    }
    if cuda_mem:
        result["cuda_mem"] = cuda_mem
    return result


@torch.no_grad()
def validation_epoch(
    model: Union[EMAEBPModel, OnlineEBPModel],
    dataloader: DataLoader,
    num_rollouts: int,
    generation_length: int,
    gamma: float,
    temperature: float,
    device: torch.device,
    max_batches: int = 50,
    whitening: bool = True,
) -> dict:
    """Computes mean metrics over validation batches.

    This function sets the model to eval() mode and iterates through the
    provided dataloader for a fixed number of batches, computing the same
    metrics as training_step but without gradient updates or backpropagation.
    """
    model.eval()
    all_results = []

    count = 0
    for batch in dataloader:
        if count >= max_batches:
            break

        # Move batch to device
        batch_device = {k: v.to(device, non_blocking=False) for k, v in batch.items()}

        context_ids = batch_device["context_ids"]
        context_mask = batch_device["context_mask"]
        completion_ids = batch_device["completion_ids"]
        completion_mask = batch_device["completion_mask"]
        context_len = context_ids.shape[1]

        full_ids = torch.cat([context_ids, completion_ids], dim=1)
        full_mask = torch.cat([context_mask, completion_mask], dim=1)

        # 1. Generate rollouts
        rollout_ids, rollout_masks = model.generate_rollouts(
            context_ids=context_ids,
            context_attention_mask=context_mask,
            num_rollouts=num_rollouts,
            generation_length=generation_length,
            temperature=temperature,
        )

        # 2. Rollout features + log-probs
        rollout_features, rollout_log_probs = model.compute_rollout_data(
            rollout_ids, rollout_masks, completion_start=context_len
        )

        # 3. Reference features and CE
        ce_loss, ref_features = model.forward_ce_and_ref_features(
            full_ids,
            full_mask,
            completion_start=context_len,
        )

        # 4. Feature-matching rewards
        if whitening:
            alignment, diversity = compute_whitened_feature_matching_terms_batched(
                rollout_features, ref_features, num_rollouts
            )
        else:
            alignment, diversity = compute_feature_matching_terms_batched(
                rollout_features, ref_features, num_rollouts
            )
        rewards = alignment - diversity

        baselines = compute_rloo_baseline_batched(rewards, num_rollouts)
        advantages = (rewards - baselines)
        reinforce_loss = -(advantages * rollout_log_probs).mean()

        total_loss = reinforce_loss
        if gamma > 0.0:
            total_loss = total_loss + gamma * ce_loss

        policy_nll = (-rollout_log_probs.mean() / max(generation_length, 1)).item()

        result = {
            "loss": total_loss.item(),
            "reinforce_loss": reinforce_loss.item(),
            "ce_loss": ce_loss.item(),
            "mean_reward": rewards.mean().item(),
            "mean_alignment": alignment.mean().item(),
            "mean_diversity": diversity.mean().item(),
            "entropy": policy_nll,
        }
        all_results.append(result)
        count += 1

    if not all_results:
        return {}

    # Aggregate results
    avg_res = {
        k: sum(r[k] for r in all_results) / len(all_results)
        for k in all_results[0].keys()
    }
    return avg_res


def memory_constrained_training_step(
    model: Union[EMAEBPModel, OnlineEBPModel],
    batch: dict,
    num_rollouts: int,
    generation_length: int,
    gamma: float,
    temperature: float,
    device: torch.device,
    log_cuda_memory: bool = False,
    whitening: bool = True,
) -> dict:
    """Memory-constrained EBP step with early CE backward.

    This path reduces peak VRAM by backpropagating the CE term before rollout
    tensors/activations are created.
    """
    # Batch is already on device via GPUPrefetcher, no need for explicit transfer
    context_ids = batch["context_ids"]
    context_mask = batch["context_mask"]
    completion_ids = batch["completion_ids"]
    completion_mask = batch["completion_mask"]

    context_len = context_ids.shape[1]
    cuda_mem = {}

    def _reset_cuda_peak(tag: str) -> None:
        if log_cuda_memory and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    def _record_cuda_peak(tag: str) -> None:
        if log_cuda_memory and device.type == "cuda":
            cuda_mem[f"{tag}_alloc_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            cuda_mem[f"{tag}_reserved_mb"] = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

    full_ids = torch.cat([context_ids, completion_ids], dim=1)
    full_mask = torch.cat([context_mask, completion_mask], dim=1)

    # ------------------------------------------------------------------
    # 1. Reference features and CE
    # ------------------------------------------------------------------
    # If using memory-constrained step, we compute CE here.
    model.train()
    _reset_cuda_peak("ref")
    ce_loss, ref_features = model.forward_ce_and_ref_features(
        full_ids,
        full_mask,
        completion_start=context_len,
    )
    _record_cuda_peak("ref")
    ce_loss_val = ce_loss.item()

    if gamma > 0.0:
        _reset_cuda_peak("ce_bwd_early")
        (gamma * ce_loss).backward()
        _record_cuda_peak("ce_bwd_early")

    # ------------------------------------------------------------------
    # 2. Generate rollouts
    # ------------------------------------------------------------------
    model.eval()
    rollout_ids, rollout_masks = model.generate_rollouts(
        context_ids=context_ids,
        context_attention_mask=context_mask,
        num_rollouts=num_rollouts,
        generation_length=generation_length,
        temperature=temperature,
    )
    _record_cuda_peak("gen")

    model.train()
    _reset_cuda_peak("rollout_fwd")
    rollout_features, rollout_log_probs = model.compute_rollout_data(
        rollout_ids, rollout_masks, completion_start=context_len
    )
    _record_cuda_peak("rollout_fwd")

    if whitening:
        alignment, diversity = compute_whitened_feature_matching_terms_batched(
            rollout_features, ref_features, num_rollouts
        )
    else:
        alignment, diversity = compute_feature_matching_terms_batched(
            rollout_features, ref_features, num_rollouts
        )
    rewards = alignment - diversity  # (B * n,)

    baselines = compute_rloo_baseline_batched(rewards, num_rollouts)
    advantages = (rewards - baselines).detach()  # stop-gradient on advantages

    # REINFORCE: maximise E[advantage * log p]  ->  minimise negation
    reinforce_loss = -(advantages * rollout_log_probs).mean()

    # Mean reward / alignment / diversity across all rollouts (diagnostics)
    mean_reward = rewards.detach().mean().item()
    mean_alignment = alignment.detach().mean().item()
    mean_diversity = diversity.detach().mean().item()
    policy_nll = (-rollout_log_probs.detach().mean() / max(generation_length, 1)).item()

    total_loss = reinforce_loss.item() + gamma * ce_loss_val

    _reset_cuda_peak("backward")
    reinforce_loss.backward()
    _record_cuda_peak("backward")

    result = {
        "loss": total_loss,
        "reinforce_loss": reinforce_loss.item(),
        "ce_loss": ce_loss_val,
        "mean_reward": mean_reward,
        "mean_alignment": mean_alignment,
        "mean_diversity": mean_diversity,
        "entropy": policy_nll,
    }
    if cuda_mem:
        result["cuda_mem"] = cuda_mem
    return result


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)

    # Optimization: Increase dynamo recompile limit to handle HF cache initialization
    # allow unspecized integers (like layer indices) on nn.Module to be dynamic.
    torch._dynamo.config.allow_unspec_int_on_nn_module = True
    torch._dynamo.config.recompile_limit = 32

    # Initialize W&B
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
        mode=args.wandb_mode,
        settings=wandb.Settings(silent=True)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    _configure_tensor_cores(device)

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
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print(f"Loading model: {args.model_name} (variant: {args.model_type}) ...")
    from transformers import AutoModelForCausalLM

    model_load_kwargs = {"torch_dtype": torch_dtype}
    if device.type == "cuda" and args.use_flash_attention:
        model_load_kwargs["attn_implementation"] = "flash_attention_2"
        print("Requesting FlashAttention v2 kernels for model attention")

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name, **model_load_kwargs
        ).to(device)
    except Exception as exc:
        if model_load_kwargs.get("attn_implementation") == "flash_attention_2":
            print(
                "FlashAttention request failed; falling back to model default "
                f"attention implementation. Reason: {exc}"
            )
            model_load_kwargs.pop("attn_implementation", None)
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model_name, **model_load_kwargs
            ).to(device)
        else:
            raise

    if args.model_type == "ema":
        model = EMAEBPModel(
            model=base_model,
            ema_decay=args.ema_decay,
            pool_type=args.pool_type,
        )
    else:
        model = OnlineEBPModel(model=base_model, pool_type=args.pool_type)

    if args.gradient_checkpointing:
        model.model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled on generator model")

    if args.compile_model and hasattr(torch, "compile"):
        compile_mode = args.compile_mode
        if compile_mode is None:
            compile_mode = "default" if args.memory_constrained else "reduce-overhead"

        try:
            model.model = torch.compile(
                model.model,
                mode=compile_mode,
                fullgraph=args.compile_fullgraph,
            )
            print(
                "Enabled torch.compile for generator "
                f"(mode={compile_mode}, fullgraph={args.compile_fullgraph})"
            )
        except Exception as exc:
            print(f"torch.compile unavailable for this setup; continuing without it: {exc}")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    val_bs = args.val_batch_size if args.val_batch_size is not None else args.batch_size * 2
    val_skip_docs = 0
    if args.val_split == args.dataset_split:
        val_skip_docs = args.max_val_docs
        print(f"Carving out {val_skip_docs} documents from {args.dataset_split} split for validation")

    print(f"Loading dataset: {args.dataset_name} ...")
    dataset = PretrainingDataset(
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.dataset_split,
        context_length=args.context_length,
        completion_length=args.generation_length,
        streaming=args.streaming,
        max_tokens=args.max_tokens,
        max_examples=args.max_examples,
        skip_documents=val_skip_docs,
        tokenized=args.tokenized,
        text_column="tokens" if args.tokenized else "text",
    )
    print("Train dataset loaded")

    val_dataloader = None
    if args.val_split:
        if val_skip_docs > 0:
            print(f"Creating validation set from first {val_skip_docs} documents of {args.dataset_split} split...")
            val_dataset = PretrainingDataset(
                tokenizer=tokenizer,
                dataset_name=args.dataset_name,
                dataset_config=args.dataset_config,
                split=args.dataset_split,
                context_length=args.context_length,
                completion_length=args.generation_length,
                streaming=args.streaming,
                max_documents=val_skip_docs,
                max_examples=args.max_val_batches * val_bs,
                tokenized=args.tokenized,
                text_column="tokens" if args.tokenized else "text",
            )
        else:
            print(f"Loading validation dataset split: {args.val_split} ...")
            val_dataset = PretrainingDataset(
                tokenizer=tokenizer,
                dataset_name=args.dataset_name,
                dataset_config=args.dataset_config,
                split=args.val_split,
                context_length=args.context_length,
                completion_length=args.generation_length,
                streaming=args.streaming,
                # Use a small subset for validation to avoid long pauses
                max_examples=args.max_val_batches * val_bs,
                tokenized=args.tokenized,
                text_column="tokens" if args.tokenized else "text",
            )
        print("Validation dataset loaded")
        print(f"Validation batch size: {val_bs}")
        
        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=val_bs,
            shuffle=False,
            collate_fn=partial(collate_fn, pad_token_id=pad_id),
            drop_last=False,
            pin_memory=pin_memory if 'pin_memory' in locals() else (device.type == "cuda"),
            num_workers=0,
        )
    else:
        print("No validation split provided; skipping periodic validation.")

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    pin_memory = args.pin_memory if args.pin_memory is not None else device.type == "cuda"
    if args.num_workers is None:
        if device.type == "cuda":
            cpu_count = os.cpu_count() or 1
            num_workers = max(1, min(8, cpu_count // 2))
        else:
            num_workers = 0
    else:
        num_workers = max(0, args.num_workers)

    persistent_workers = (
        args.persistent_workers
        if args.persistent_workers is not None
        else num_workers > 0
    )

    dataloader_kwargs = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": False,  # IterableDataset handles shuffling internal to the stream
        "collate_fn": partial(collate_fn, pad_token_id=pad_id),
        "drop_last": True,
        "pin_memory": pin_memory,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = persistent_workers
        dataloader_kwargs["prefetch_factor"] = args.prefetch_factor

    print(
        "DataLoader settings: "
        f"pin_memory={pin_memory}, num_workers={num_workers}, "
        f"persistent_workers={persistent_workers if num_workers > 0 else False}, "
        f"prefetch_factor={args.prefetch_factor if num_workers > 0 else 'n/a'}"
    )

    dataloader = DataLoader(**dataloader_kwargs)

    # ------------------------------------------------------------------
    # Optimiser (only the trainable generator parameters)
    # ------------------------------------------------------------------
    optimizer_kwargs = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }
    if args.use_fused_adamw and device.type == "cuda":
        optimizer_kwargs["fused"] = True

    try:
        optimizer = torch.optim.AdamW(
            model.model.parameters(),
            **optimizer_kwargs,
        )
        if optimizer_kwargs.get("fused"):
            print("Using fused AdamW CUDA kernels")
    except TypeError:
        optimizer_kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(
            model.model.parameters(),
            **optimizer_kwargs,
        )
        print("Fused AdamW unsupported in this torch build; using standard AdamW")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_file = args.metrics_file or os.path.join(args.output_dir, "metrics.pkl")

    step = 0
    model.train()
    metrics_history: List[dict] = []

    # Memory-constrained fallback: backpropagate CE early to release its graph
    # before rollout tensors/activations are created. Alternative (often faster
    # when memory headroom exists): keep CE and REINFORCE in one combined loss
    # and call backward once.
    if args.memory_constrained:
        print("Using memory-constrained training step (early CE backward)")
        step_fn = memory_constrained_training_step
    else:
        print("Using regular training step (single backward, CE/ref after rollouts)")
        step_fn = training_step

    # Wrap dataloader with GPU prefetcher for asynchronous H2D transfer
    prefetched_loader = GPUPrefetcher(dataloader, device)

    # ------------------------------------------------------------------
    # Baseline validation
    # ------------------------------------------------------------------
    if val_dataloader is not None:
        print("Running baseline validation (step 0)...")
        baseline_val_start = time.perf_counter()
        val_res = validation_epoch(
            model=model,
            dataloader=val_dataloader,
            num_rollouts=args.num_rollouts,
            generation_length=args.generation_length,
            gamma=args.gamma,
            temperature=args.temperature,
            device=device,
            max_batches=args.max_val_batches,
            whitening=args.whitening,
        )
        baseline_val_duration = time.perf_counter() - baseline_val_start
        print(f"Baseline validation duration: {baseline_val_duration:.2f}s")
        if val_res:
            print(
                f"Step      0 [VAL] | loss={val_res['loss']:.4f} "
                f"reinforce={val_res['reinforce_loss']:.4f} "
                f"ce={val_res['ce_loss']:.4f} "
                f"reward={val_res['mean_reward']:.4f} "
                f"entropy={val_res['entropy']:.4f}"
            )
            val_metrics = {f"val_{k}": v for k, v in val_res.items()}
            val_metrics["val_duration_sec"] = baseline_val_duration
            val_metrics["step"] = 0
            wandb.log(val_metrics)

    while step < args.max_steps:
        for batch in prefetched_loader:
            if step >= args.max_steps:
                break

            optimizer.zero_grad(set_to_none=True)

            result = step_fn(
                model=model,
                batch=batch,
                num_rollouts=args.num_rollouts,
                generation_length=args.generation_length,
                gamma=args.gamma,
                temperature=args.temperature,
                device=device,
                log_cuda_memory=args.log_cuda_memory,
                whitening=args.whitening,
            )
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), args.grad_clip)
            if args.log_cuda_memory and device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            optimizer.step()
            if args.log_cuda_memory and device.type == "cuda":
                result.setdefault("cuda_mem", {})["opt_step_alloc_mb"] = (
                    torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                )
                result.setdefault("cuda_mem", {})["opt_step_reserved_mb"] = (
                    torch.cuda.max_memory_reserved(device) / (1024 ** 2)
                )

            # Update EMA feature network (only for EMAEBPModel)
            if isinstance(model, EMAEBPModel):
                model.update_ema()

            step += 1

            # Loss is already stored as a detached scalar for logging.
            loss_val = result["loss"]

            # Record metrics for every step (floats only, very lightweight)
            metrics_history.append(
                {
                    "step": step,
                    "loss": loss_val,
                    "reinforce_loss": result["reinforce_loss"],
                    "ce_loss": result["ce_loss"],
                    "mean_reward": result["mean_reward"],
                    "mean_alignment": result["mean_alignment"],
                    "mean_diversity": result["mean_diversity"],
                    "entropy": result["entropy"],
                }
            )
            # Log to W&B
            wandb.log(metrics_history[-1])

            if step % args.log_steps == 0:
                print(
                    f"Step {step:6d} | loss={loss_val:.4f} "
                    f"reinforce={result['reinforce_loss']:.4f} "
                    f"ce={result['ce_loss']:.4f} "
                    f"reward={result['mean_reward']:.4f} "
                    f"align={result['mean_alignment']:.4f} "
                    f"div={result['mean_diversity']:.4f} "
                    f"entropy={result['entropy']:.4f}"
                )
                if args.log_cuda_memory and device.type == "cuda" and "cuda_mem" in result:
                    mem = result["cuda_mem"]
                    mem_line = (
                        "CUDA peak MB | "
                        f"ref={mem.get('ref_alloc_mb', 0.0):.0f}/{mem.get('ref_reserved_mb', 0.0):.0f} "
                        f"gen={mem.get('gen_alloc_mb', 0.0):.0f}/{mem.get('gen_reserved_mb', 0.0):.0f} "
                        f"rollout_fwd={mem.get('rollout_fwd_alloc_mb', 0.0):.0f}/{mem.get('rollout_fwd_reserved_mb', 0.0):.0f} "
                        f"backward={mem.get('backward_alloc_mb', 0.0):.0f}/{mem.get('backward_reserved_mb', 0.0):.0f} "
                        f"opt={mem.get('opt_step_alloc_mb', 0.0):.0f}/{mem.get('opt_step_reserved_mb', 0.0):.0f}"
                    )
                    print(mem_line)

            if step % args.val_steps == 0 and val_dataloader is not None:
                print(f"Running validation at step {step}...")
                val_res = validation_epoch(
                    model=model,
                    dataloader=val_dataloader,
                    num_rollouts=args.num_rollouts,
                    generation_length=args.generation_length,
                    gamma=args.gamma,
                    temperature=args.temperature,
                    device=device,
                    max_batches=args.max_val_batches,
                    whitening=args.whitening,
                )
                if val_res:
                    print(
                        f"Step {step:6d} [VAL] | loss={val_res['loss']:.4f} "
                        f"reinforce={val_res['reinforce_loss']:.4f} "
                        f"ce={val_res['ce_loss']:.4f} "
                        f"reward={val_res['mean_reward']:.4f} "
                        f"entropy={val_res['entropy']:.4f}"
                    )
                    # Store validation metrics with a prefix
                    val_metrics = {f"val_{k}": v for k, v in val_res.items()}
                    # Ensure step info is included for W&B x-axis consistency if logged separately
                    val_metrics["step"] = step
                    metrics_history[-1].update(val_metrics)
                    # Log validation to W&B
                    wandb.log(val_metrics)
                
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
    
    wandb.finish()


if __name__ == "__main__":
    train(parse_args())
