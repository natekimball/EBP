#!/bin/bash
# GSM8K evals at a matched reduced config (200 problems x 4 samples x 512 tokens).
# All three models are re-run at the SAME config so they are mutually comparable;
# the existing eval_results/*.json used 500x8x2048 and are NOT comparable to these.
set -u
PY=/home/natekimball/Projects/EBP/ENV/bin/python
ARGS="--benchmark gsm8k --num_problems 200 --num_samples 4 --max_new_tokens 512 --batch_size 32 --temperature 0.7"

for ckpt in output/step_110000 cpt_output/final Qwen/Qwen3-0.6B-Base; do
  echo "=== $ckpt  ($(date +%H:%M:%S)) ==="
  $PY eval_benchmark.py "$ckpt" $ARGS --output_dir ./eval_results || echo "FAILED: $ckpt"
done
echo "=== done ($(date +%H:%M:%S)) ==="
