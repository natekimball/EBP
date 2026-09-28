# EBP Training Benchmarks: Memory vs. Throughput

This report compares the two primary training strategies implemented for Energy-Based Pre-training (EBP) on the **Qwen3-0.6B** architecture. The benchmark evaluates peak VRAM utilization, compilation stability, and steady-state throughput.

## Experiment Configuration

*   **Model**: Qwen3-0.6B (online variant)
*   **Batch Size**: 4
*   **Rollouts per item**: 8 (Total rollout batch = 32)
*   **Generation Length**: 64 tokens
*   **Precision**: `bfloat16`
*   **Optimizations**: FlashAttention v2, Fused AdamW, Gradient Checkpointing.

## Performance Results

| Metric | **Memory-Constrained (Default Compile)** | **Regular (Reduce-Overhead) |
| :--- | :--- | :--- |
| **Peak Allocated VRAM** | 10.29 GB | 10.25 GB |
| **Peak Reserved VRAM** | **10.87 GB** | **15.00 GB** |
| **VRAM Buffer Margin** | 4.13 GB free | < 0.1 GB free (on 16GB GPU) |
| **Compilation Time** | 45 seconds | 180+ seconds |
| **Steady-State Step Time** | ~1.8s / step | **~1.7s / step** |
| **Stability** | High | Low (Graph Buffer Overwrites) |

## Analysis & Findings

### 1. Memory: The "Reserved" Gap
The most significant finding is the **4.13 GB difference in Reserved VRAM**. 
*   In the **Regular** path, `reduce-overhead` (via CUDA Graphs) pre-allocates static buffers for all intermediate tensors. This "locks" nearly 15 GB of memory even if only 10 GB is actively being used.
*   In the **Memory-Constrained** path, the early backward pass releases the Cross-Entropy graph before rollout tensors are allocated. Combined with `mode=default` compilation, this allows the GPU to reuse memory dynamically, making it safer for high-concurrency environments or smaller GPUs.

### 2. Throughput & Latency
*   **Compilation Overhead**: `reduce-overhead` took nearly 3x longer to compile. For iterative development or short runs, this significantly delays the "time to first result."
*   **Iteration Speed**: Once fully compiled, the `reduce-overhead` mode was roughly **5.8% faster** in terms of raw token-per-second throughput. This gain is achieved by eliminating CPU-to-GPU kernel launch overhead.

### 3. Structural Compatibility
The **Memory-Constrained** step is the only path compatible with aggressive model optimizations (like holding reference features across multiple model calls) while using `torch.compile`. The `reduce-overhead` mode currently crashes when training logic attempts to access tensors from previous graph runs that have been overwritten by subsequent passes (e.g., generation).

## Recommendations

1.  **Default Strategy**: Use `--memory_constrained` (it now defaults to `mode=default` compilation). It is the most robust, uses 30% less system VRAM, and avoids complex CUDA Graph memory aliasing issues.
2.  **High-VRAM Scaling**: Only use `--no-memory_constrained` with `reduce-overhead` if you have substantial VRAM headroom (24GB+) and are conducting a final production run where the extra 5-6% throughput outweighs the stability risk and long compilation times.
