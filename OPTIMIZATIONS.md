# Performance Optimizations for EBP Training

This document describes three key performance optimizations implemented to improve training throughput:

## 1. Custom Fixed-Length Sampling Loop (model.py)

**Location:** `ebp/model.py:_generate_with_kv_cache()` and new `_sample_tokens()`

**Overview:**
Replaced `model.generate()` with a custom fixed-length sampling loop to eliminate general-purpose generation overhead. The `generate()` method is designed for flexible decoding scenarios (variable length, beam search, etc.); for training rollouts with fixed length and simple sampling policy, a tight loop is faster.

**Implementation:**
- New `_sample_tokens()` helper: Efficient temperature-scaled sampling from logits using `torch.multinomial()` 
- Modified `_generate_with_kv_cache()`: Uses a simple loop instead of `model.generate()`
  - Maintains prefix KV cache (already in place)
  - Each iteration: forward → sample → update cache → repeat
  - No beam search overhead, no stopping criteria checks, no extra bookkeeping

**Performance Impact:**
- Reduces per-token generation overhead
- Most effective for shorter rollouts (8-16 tokens) where loop overhead dominates generate() overhead
- Expected speedup: **5-15% faster rollout generation** depending on model size and generation length

**Usage:**
No API changes—this optimization is transparent. The `generate_rollouts()` method automatically uses the new implementation.

---

## 2. GPU Prefetcher for I/O Overlap (train.py)

**Location:** `train.py:GPUPrefetcher` class and integration in main training loop

**Overview:**
Overlaps CPU→GPU data transfer for batch i+1 with GPU computation on batch i. Reduces GPU idle time while waiting for data.

**Implementation:**
- `GPUPrefetcher` class wraps the DataLoader
  - Uses CUDA streams (when available) for asynchronous H2D transfer
  - Non-blocking transfers (`non_blocking=True`) allow computation and transfer to overlap
  - Synchronizes before yielding batches to ensure data is ready
- Training loop wraps dataloader: `prefetched_loader = GPUPrefetcher(dataloader, device)`
- Removed explicit `.to(device)` calls in `training_step()` and `memory_constrained_training_step()` since batches arrive pre-loaded

**Performance Impact:**
- Hides H2D transfer latency when CPU pipeline is borderline
- Most effective for small batch sizes where transfer is a bottleneck
- Expected speedup: **2-8% depending on batch size and data loading speed**
  - Higher impact with small batches (B=1-4)
  - Diminishing returns with very large batches (where transfer is negligible vs compute)

**Usage:**
Automatic—integrated into the training loop. Enable with command line as usual:
```bash
python train.py --batch_size 4 --pin_memory --num_workers 4
```

The prefetcher works best when `pin_memory=True` and `num_workers > 0` are already enabled (which they are by default on CUDA).

---

## 3. Benchmark Script for Throughput Comparison (benchmark.py)

**Location:** New `benchmark.py` script

**Overview:**
Measures tokens/sec throughput under different configurations, especially `online + gamma=0` vs `online + gamma=X` to characterize the quality/speed tradeoff.

**Motivation:**
- CE forwarding is expensive
- `gamma=0` removes the CE forward pass entirely
- Can show the maximum achievable rollout-policy throughput
- Helps quantify the compute cost of the CE term

**Measurements:**
- Tokens/sec (primary metric)
- Per-step time (in seconds)
- Memory usage (if `--log_cuda_memory` supported)
- Warmup steps to stabilize timing

**Usage:**

```bash
# Benchmark online variant at maximum speed (gamma=0, no CE)
ENV/bin/python3 benchmark.py \
    --model_name Qwen/Qwen3-0.6B \
    --model_type online \
    --gamma 0.0 \
    --num_steps 100 \
    --warmup_steps 5

# Compare with mixed objective (gamma=0.1)
ENV/bin/python3 benchmark.py \
    --model_name Qwen/Qwen3-0.6B \
    --model_type online \
    --gamma 0.1 \
    --num_steps 100
```

**Example Output:**
```
============================================================
Benchmark Results (after 5 warmup steps):
============================================================
  Avg step time: 0.287s
  Tokens/sec: 446
  Total tokens generated: 150000
============================================================
```

**Command-line Arguments:**
- `--model_name`: HuggingFace model ID
- `--model_type`: "online" or "ema"
- `--gamma`: CE weight (0.0 = rollout-policy only, 0.1+ = mixed)
- `--batch_size`: Batch size (default 4)
- `--num_rollouts`: Rollouts per example (default 4)
- `--generation_length`: Tokens generated per rollout
- `--num_steps`: Total benchmark steps
- `--warmup_steps`: Steps to skip before timing (default 5)
- `--memory_constrained`: Use memory-constrained training step

---

## Synergistic Effects

These three optimizations are complementary:

1. **Sampling loop** + **prefetcher**: The sampling loop is very tight, so H2D overlap becomes more critical
2. **gamma ≈ 0.0** + **sampling loop**: Maximizes rollout throughput (no CE overhead)
3. All three together: Expected **10-25% end-to-end speedup** for rollout-heavy workloads

---

## Recommended Testing Plan

1. **Establish baseline:**
   ```bash
   ENV/bin/python3 benchmark.py --gamma 0.1 --model_type online --num_steps 50
   ```

2. **Check max rollout throughput:**
   ```bash
   ENV/bin/python3 benchmark.py --gamma 0.0 --model_type online --num_steps 50
   ```

3. **Compare model types:**
   ```bash
   ENV/bin/python3 benchmark.py --gamma 0.1 --model_type ema --num_steps 50
   ENV/bin/python3 benchmark.py --gamma 0.1 --model_type online --num_steps 50
   ```

4. **Run full training and log metrics:**
   ```bash
   ENV/bin/python3 train.py \
       --model_type online \
       --gamma 0.1 \
       --max_steps 10000 \
       --log_steps 100
   ```

---

## API Compatibility

All three optimizations are **backward compatible**:
- No changes to public API
- Existing scripts continue to work unchanged
- Optimizations are transparent to user code

## Notes

- **Custom sampling loop:** Works best with fixed generation length. For variable-length decoding, consider reverting to `model.generate()`
- **GPU prefetcher:** Requires CUDA for overlapped streaming; falls back gracefully on CPU
- **Benchmark script:** Uses same models/configs as training; measurements may differ slightly due to one-batch-per-step vs streaming

