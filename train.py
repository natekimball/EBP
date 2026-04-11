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
from ebp.rewards import compute_feature_matching_terms_batched, compute_rloo_baseline_batched


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
    )
    parser.add_argument("--dataset_config", type=str, default="v1_7")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream dataset samples instead of loading split materialization upfront.",
    )
    parser.add_argument(
        "--max_documents",
        type=int,
        default=None,
        help="Optional cap on number of raw documents tokenised from the dataset.",
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
    parser.add_argument("--batch_size", type=int, default=4)
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
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="Compilation mode used by torch.compile.",
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
    early_ce_backward: bool = False,
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
    non_blocking = device.type == "cuda"
    context_ids = batch["context_ids"].to(device, non_blocking=non_blocking)
    context_mask = batch["context_mask"].to(device, non_blocking=non_blocking)
    completion_ids = batch["completion_ids"].to(device, non_blocking=non_blocking)
    completion_mask = batch["completion_mask"].to(device, non_blocking=non_blocking)

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
    # 1. Reference features  phi(c:y)
    # ------------------------------------------------------------------
    full_ids = torch.cat([context_ids, completion_ids], dim=1)
    full_mask = torch.cat([context_mask, completion_mask], dim=1)

    ce_loss: torch.Tensor | None = None
    ce_loss_val = 0.0
    used_combined_ref_ce = False

    # In online mode with gamma > 0, compute CE and reference features
    # together to avoid a redundant full-sequence forward pass.
    if isinstance(model, OnlineEBPModel) and gamma > 0.0:
        model.train()
        _reset_cuda_peak("ref")
        ce_loss, ref_features = model.forward_ce_and_ref_features(
            full_ids,
            full_mask,
            completion_start=context_len,
        )
        _record_cuda_peak("ref")
        used_combined_ref_ce = True
        ce_loss_val = ce_loss.item()

        if early_ce_backward:
            _reset_cuda_peak("ce_bwd_early")
            (gamma * ce_loss).backward()
            _record_cuda_peak("ce_bwd_early")
    else:
        model.eval()
        _reset_cuda_peak("ref")
        ref_features = model.extract_features(
            full_ids, full_mask, completion_start=context_len
        )  # (B, feat_dim)
        _record_cuda_peak("ref")

        if early_ce_backward and gamma > 0.0:
            model.train()
            _reset_cuda_peak("ce_fwd")
            ce_out = model(input_ids=full_ids, attention_mask=full_mask, labels=full_ids)
            ce_loss = ce_out.loss
            ce_loss_val = ce_loss.item()
            _record_cuda_peak("ce_fwd")

            _reset_cuda_peak("ce_bwd_early")
            (gamma * ce_loss).backward()
            _record_cuda_peak("ce_bwd_early")

    # ------------------------------------------------------------------
    # 2. Generate rollouts  y_hat_j ~ p_theta(.|c)
    # ------------------------------------------------------------------
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
    # 4. Feature-matching rewards and REINFORCE loss
    # ------------------------------------------------------------------
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
    # 5. Optional cross-entropy term
    # ------------------------------------------------------------------
    total_loss = reinforce_loss

    if gamma > 0.0:
        if ce_loss is None and not early_ce_backward:
            _reset_cuda_peak("ce_fwd")
            ce_out = model(input_ids=full_ids, attention_mask=full_mask, labels=full_ids)
            ce_loss = ce_out.loss
            _record_cuda_peak("ce_fwd")
            ce_loss_val = ce_loss.item()
        elif used_combined_ref_ce and log_cuda_memory and device.type == "cuda":
            # Combined pass already measured in the "ref" stage.
            cuda_mem["ce_fwd_alloc_mb"] = cuda_mem.get("ref_alloc_mb", 0.0)
            cuda_mem["ce_fwd_reserved_mb"] = cuda_mem.get("ref_reserved_mb", 0.0)

        if early_ce_backward:
            # CE term was already backpropagated to reduce peak VRAM.
            total_loss = reinforce_loss + reinforce_loss.new_tensor(gamma * ce_loss_val)
        else:
            total_loss = total_loss + gamma * ce_loss

    # Backward schedule lives in this function for a simpler outer loop.
    backward_target = reinforce_loss if early_ce_backward else total_loss
    _reset_cuda_peak("backward")
    backward_target.backward()
    _record_cuda_peak("backward")

    result = {
        "loss": total_loss,
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


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)

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
        model: Union[EMAEBPModel, OnlineEBPModel] = EMAEBPModel(
            model=base_model,
            ema_decay=args.ema_decay,
        )
    else:
        model = OnlineEBPModel(model=base_model)

    if args.gradient_checkpointing:
        model.model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled on generator model")

    if args.compile_model and hasattr(torch, "compile"):
        try:
            model.model = torch.compile(
                model.model,
                mode=args.compile_mode,
                fullgraph=args.compile_fullgraph,
            )
            print(
                "Enabled torch.compile for generator "
                f"(mode={args.compile_mode}, fullgraph={args.compile_fullgraph})"
            )
        except Exception as exc:
            print(f"torch.compile unavailable for this setup; continuing without it: {exc}")

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
        streaming=args.streaming,
        max_documents=args.max_documents,
        max_tokens=args.max_tokens,
        max_examples=args.max_examples,
    )
    print(f"Dataset size: {len(dataset)} examples")

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
        "shuffle": True,
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
    use_memory_constrained_backward = True

    while step < args.max_steps:
        for batch in dataloader:
            if step >= args.max_steps:
                break

            optimizer.zero_grad(set_to_none=True)

            result = training_step(
                model=model,
                batch=batch,
                num_rollouts=args.num_rollouts,
                generation_length=args.generation_length,
                gamma=args.gamma,
                temperature=args.temperature,
                device=device,
                log_cuda_memory=args.log_cuda_memory,
                early_ce_backward=use_memory_constrained_backward,
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
                    "mean_alignment": result["mean_alignment"],
                    "mean_diversity": result["mean_diversity"],
                    "entropy": result["entropy"],
                }
            )

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
                        f"ce_fwd={mem.get('ce_fwd_alloc_mb', 0.0):.0f}/{mem.get('ce_fwd_reserved_mb', 0.0):.0f} "
                        f"backward={mem.get('backward_alloc_mb', 0.0):.0f}/{mem.get('backward_reserved_mb', 0.0):.0f} "
                        f"opt={mem.get('opt_step_alloc_mb', 0.0):.0f}/{mem.get('opt_step_reserved_mb', 0.0):.0f}"
                    )
                    print(mem_line)

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
