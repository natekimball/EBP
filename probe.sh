#!/usr/bin/env bash
# Quick memory/timing probe for a single config.
# Usage: bash probe.sh <batch_size> <val_batch_size> <chunk_size> <grad_accum>
# Runs 2 steps with 1 val batch and reports outcome.
set -euo pipefail

BS=${1:-4}
VBS=${2:-2}
CHUNK=${3:-32}
ACCUM=${4:-2}
OUTDIR="/tmp/probe_b${BS}_v${VBS}_c${CHUNK}_a${ACCUM}"
LOG="/tmp/probe_b${BS}_v${VBS}_c${CHUNK}_a${ACCUM}.log"

echo "=== PROBE: batch_size=$BS val_batch=$VBS chunk=$CHUNK grad_accum=$ACCUM ==="

CHUNK_ARG=""
if [ "$CHUNK" != "none" ]; then
    CHUNK_ARG="--rollout_chunk_size $CHUNK"
fi

START=$(date +%s)

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/natekimball/Projects/EBP/ENV/bin/python /home/natekimball/Projects/EBP/train.py \
    --model_name Qwen/Qwen3-0.6B-Base \
    --model_type online \
    --dataset_name data/Nemotron-CC-Math-v1_4plus \
    --tokenized \
    --context_length 1024 \
    --generation_length 32 \
    --num_rollouts 32 \
    --batch_size "$BS" \
    --grad_accum_steps "$ACCUM" \
    --log_steps 1 \
    --save_steps 1_000_000 \
    --max_steps 2 \
    --use_fused_adamw \
    --use_flash_attention \
    --num_workers 4 \
    --compile_model \
    --compile_fullgraph \
    --memory_constrained \
    $CHUNK_ARG \
    --val_split test \
    --val_steps 1000 \
    --val_batch_size "$VBS" \
    --max_val_batches 1 \
    --wandb_mode disabled \
    --output_dir "$OUTDIR" \
    2>&1 | tee "$LOG"

END=$(date +%s)
ELAPSED=$((END - START))

echo ""
echo "=== RESULT: batch_size=$BS val_batch=$VBS chunk=$CHUNK grad_accum=$ACCUM ==="
echo "Exit code: $?"
echo "Wall time: ${ELAPSED}s"
grep -E "^Step\s+[0-9]|OOM|Killed|Error|cuda out" "$LOG" | tail -5 || true
