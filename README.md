# EBP – Energy-Based Pre-Training

Pre-trains a causal language model (default: **Qwen/Qwen3-0.6B**) with a
feature-matching objective inspired by
[Energy-Based Fine-Tuning (EBFT)](https://arxiv.org/abs/2503.xxxxx).

Instead of a fully frozen feature network, EBP uses an **Exponential Moving
Average (EMA)** of the generator itself.  As the generator improves, so do
the features used to evaluate rollout quality — no separate reference model
is required.

---

## Method overview

For each training example *(context c, reference completion y)*:

1. **Generate rollouts** – sample *n* completions `ŷ_j ~ p_θ(·|c)`.
2. **Extract features** – run the EMA model to obtain `ϕ(c:y)` and
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

---

## Repository layout

```
EBP/
├── train.py          # Main training script
├── requirements.txt
└── ebp/
    ├── model.py      # EBPModel: generator + EMA feature network
    ├── rewards.py    # Feature-matching rewards & RLOO baseline
    └── data.py       # PretrainingDataset and collate_fn
tests/
└── test_ebp.py       # Unit + integration tests (no model download needed)
```

---

## Quick start

```bash
pip install -r requirements.txt

python train.py \
   --model_name Qwen/Qwen3-0.6B \
   --dataset_name allenai/dolma \
   --dataset_config v1_7 \
   --max_documents 200000 \
   --context_length 128 \
   --generation_length 8 \
   --num_rollouts 4 \
   --ema_decay 0.999 \
   --gamma 0.1 \
   --batch_size 4 \
   --num_workers 4 \
   --persistent_workers \
   --pin_memory \
   --use_fused_adamw \
   --use_flash_attention \
   --compile_model \
   --lr 1e-5 \
   --max_steps 10000 \
   --output_dir ./output
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--model_name` | `Qwen/Qwen3-0.6B` | HuggingFace model id |
| `--num_rollouts` | `4` | Completions sampled per context |
| `--generation_length` | `8` | Tokens generated per rollout |
| `--ema_decay` | `0.999` | EMA decay factor τ |
| `--gamma` | `0.1` | Weight of the CE term |
| `--temperature` | `1.0` | Sampling temperature |
| `--dtype` | `float32` | `float32` / `bfloat16` / `float16` |
| `--use_fused_adamw` | `true` | Use fused CUDA AdamW kernels when available |
| `--use_flash_attention` | `true` | Request FlashAttention v2 on supported CUDA setups |
| `--compile_model` | `true` | Compile generator with `torch.compile` |
| `--compile_mode` | `reduce-overhead` | `torch.compile` mode |
| `--pin_memory` | auto (`true` on CUDA, `false` on CPU) | Enable pinned host memory in the DataLoader for faster GPU transfer |
| `--num_workers` | auto | DataLoader worker count (auto-tuned by device) |
| `--persistent_workers` | auto | Reuse workers (and pinned buffers) across batches |
| `--max_documents` | `None` | Optional cap on raw documents tokenized before chunking |
| `--max_tokens` | `None` | Optional cap on total tokens accumulated before chunking |

---

## Running tests

```bash
pip install pytest
python -m pytest tests/ -v
```
