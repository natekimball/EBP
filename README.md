# EBP – Energy-Based Pre-Training

Pre-trains a causal language model (default: **Qwen/Qwen3-0.6B**) with a
feature-matching objective inspired by
[Energy-Based Fine-Tuning (EBFT)](https://arxiv.org/abs/2503.xxxxx).

EBP uses a **stable feature network** to evaluate rollout quality. This repository supports two variants:
- **EMA (Default)**: Uses an Exponential Moving Average of the generator weights.
- **Online**: Uses the live generator weights directly (lower memory, higher throughput).

In both cases, as the model improves, the features used for evaluation evolve automatically — no separate reference model or frozen backbone is required.

---

## Method overview

For each training example *(context c, reference completion y)*:

1. **Generate rollouts** – sample *n* completions `ŷ_j ~ p_θ(·|c)`.
2. **Extract features** – run the feature network (EMA or Online weights) to obtain `ϕ(c:y)` and
   `ϕ(c:ŷ_j)` at layers placed at depths 25 %, 50 %, and 75 % of the network.
   Each per-layer vector is mean-pooled over the completion positions and
   L2-normalised.
3. **Compute rewards** (EBFT Eq. 7):

   ```
   r_j = 2 ϕ(c:ŷ_j)ᵀ ϕ(c:y)  −  2/(n-1) Σ_{j'≠j} ϕ(c:ŷ_j)ᵀ ϕ(c:ŷ_{j'})
         ↑ alignment                ↑ diversity
   ```

4. **REINFORCE with RLOO baseline** – update `p_θ` to maximise
   `E[r_j · log p_θ(ŷ_j|c)]`.
5. **Cross-entropy term** (optional, weight `γ`) – standard teacher-forcing
   loss for stable training.
6. **Update EMA** – `ema ← τ·ema + (1-τ)·θ` (stop-gradient).

For a direct continued-pretraining baseline, pass `--ce_only` to disable the
feature-matching / REINFORCE objective and optimize only the standard
cross-entropy loss.

---

## Performance & Memory

This repository includes several optimizations for large-scale pre-training on consumer and data-center GPUs.

### Training Strategies
EBP supports two main execution paths to balance throughput and memory:

- **Standard (`--no-memory_constrained`)**: Uses `torch.compile(mode="reduce-overhead")` with CUDA Graphs. This offers the highest throughput (~6% faster) but has a higher "Reserved VRAM" floor because graph buffers are pre-allocated.
- **Memory-Constrained (`--memory_constrained`)**: Automatically used as the default. It performs the Cross-Entropy backward pass *before* rollout generation, allowing the GPU to reuse activation buffers. This reduces peak reserved VRAM by ~30% for small models like Qwen3-0.6B.

See [benchmark_report.md](benchmark_report.md) for a detailed comparison of VRAM utilization and iteration speeds.

### Key Optimizations
- **Custom Sampling Loop**: Rollout generation uses a specialized token sampling loop in `ebp/model.py` instead of the general `model.generate()`, reducing overhead by 5-15%.
- **GPU Prefetcher**: Overlaps CPU-to-GPU data transfers with model computation to hide I/O latency.
- **Online Variant**: Use `--model_type online` to use the live generator as its own reward feature extractor, saving the memory cost of a second EMA model.

Refer to [OPTIMIZATIONS.md](OPTIMIZATIONS.md) for technical implementation details.

---

## Repository layout

```
EBP/
├── train.py          # Main training script (generator training + RLOO)
├── chat.py           # Interactive inference CLI for checkpoints
├── benchmark.py      # Throughput and memory profiling utility
├── plot_metrics.py   # Metrics visualization
├── tokenize_dataset.py # Standalone dataset pre-tokenization utility
├── requirements.txt
├── OPTIMIZATIONS.md   # Detailed performance implementation notes
├── benchmark_report.md # Comparative study of memory strategies
└── ebp/
    ├── model.py      # EBPModel: generator + EMA/Online feature networks
    ├── rewards.py    # Feature-matching rewards & RLOO baseline
    └── data.py       # PretrainingDataset and collate_fn
tests/
└── test_ebp.py       # Unit + integration tests (purely synthetic inputs)
```

---

## Quick start

### 1. Pre-tokenize the dataset (Recommended)
Pre-tokenizing saves significant CPU time during training, especially for large datasets.

```bash
python tokenize_dataset.py \
    --dataset_name allenai/dolma \
    --dataset_config v1_7 \
    --output_dir ./data/dolma_tokenized
```

### 2. Launch Training
Train with the hybrid EBP + Cross-Entropy objective.

```bash
python train.py \
   --model_name Qwen/Qwen3-0.6B \
   --dataset_name ./data/dolma_tokenized \
   --context_length 128 \
   --generation_length 8 \
   --num_rollouts 4 \
   --gamma 0.1 \
   --batch_size 4 \
   --compile_model \
   --lr 1e-5 \
   --output_dir ./output
```

*Note: For a CE-only baseline, add `--ce_only` to the command above.*

### 3. Analyze & Chat
Visualize training metrics and interact with the resulting model.

```bash
# Plot loss curves and reward distribution
python plot_metrics.py --metrics_file ./output/metrics.pkl

# Interactive CLI chat with a checkpoint
python chat.py ./output/step_10000
```

---

## Utility Tools

### Benchmarking
Measure the throughput of your configuration without launching a full training run:
```bash
python benchmark.py --model_type online --gamma 0.1 --batch_size 4
```

### Key arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--model_name` | `Qwen/Qwen3-0.6B` | HuggingFace model identifier. |
| `--model_type` | `ema` | `ema` (dual network) or `online` (single network). |
| `--memory_constrained` | `True` | Prioritize memory reuse over peak throughput (recommended). |
| `--gamma` | `0.1` | Weight of the Cross-Entropy loss term. |
| `--num_rollouts` | `4` | Number of completions sampled per context. |
| `--generation_length` | `8` | Tokens generated per rollout. |
| `--ema_decay` | `0.999` | EMA decay factor $\tau$ (only used with `--model_type ema`). |
| `--ce_only` | `false` | Run standard continued-pretraining with CE loss only. |
| `--temperature` | `1.0` | Sampling temperature for rollouts. |
| `--dtype` | `auto` | `float32` / `bfloat16` / `float16` / `auto`. |
| `--compile_model` | `false` | Compile generator with `torch.compile`. |
| `--compile_mode` | `default` | `torch.compile` mode (`default` or `reduce-overhead`). |
| `--whitening` | `false` | Use whitened feature matching (EBFT Eq. 9). |
| `--use_fused_adamw` | `true` | Use fused CUDA AdamW kernels when available. |
| `--use_flash_attention` | `true` | Request FlashAttention v2 on supported CUDA setups. |
| `--pin_memory` | auto | Enable pinned host memory for faster GPU transfer. |
| `--num_workers` | auto | DataLoader worker count. |
| `--persistent_workers` | auto | Reuse workers across batches. |
| `--max_examples` | `None` | Optional cap on total training examples. |
| `--max_tokens` | `None` | Optional cap on total raw tokens before chunking. |

---

## Running tests

```bash
pip install pytest
python -m pytest tests/ -v
```
