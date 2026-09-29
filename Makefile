PYTHON = /home/natekimball/Projects/EBP/ENV/bin/python

train:
	$(PYTHON) train.py \
		--model_name Qwen/Qwen3-0.6B-Base \
		--model_type online \
		--dataset_name data/Nemotron-CC-Math-v1_4plus \
		--tokenized \
		--context_length 512 \
		--generation_length 8 \
		--num_rollouts 8 \
		--batch_size 8 \
		--gradient_checkpointing \
		--log_steps 500 \
		--save_steps 10_000 \
		--max_steps 1_000_000 \
		--gradient_checkpointing \
		--use_fused_adamw \
		--use_flash_attention \
		--num_workers 4 \
		--compile_model \
		--memory_constrained \
		--val_split train \
		--val_steps 1000 \
		--val_batch_size 8 \
		--max_val_batches 100 \
		--wandb_run_name EBP

cpt:
	$(PYTHON) train.py \
	    --ce_only \
		--model_name Qwen/Qwen3-0.6B-Base \
		--model_type online \
		--dataset_name data/Nemotron-CC-Math-v1_4plus \
		--tokenized \
		--context_length 512 \
		--generation_length 8 \
		--num_rollouts 8 \
		--batch_size 8 \
		--gradient_checkpointing \
		--log_steps 500 \
		--save_steps 10_000 \
		--max_steps 1_000_000 \
		--gradient_checkpointing \
		--use_fused_adamw \
		--use_flash_attention \
		--num_workers 4 \
		--compile_model \
		--memory_constrained \
		--val_split train \
		--val_steps 1000 \
		--val_batch_size 8 \
		--max_val_batches 100 \
		--wandb_run_name CPT \
		--output_dir cpt_output

LARGE_ARGS = \
		--model_name Qwen/Qwen3-0.6B-Base \
		--model_type online \
		--dataset_name data/Nemotron-CC-Math-v1_4plus \
		--tokenized \
		--context_length 512 \
		--generation_length 16 \
		--num_rollouts 8 \
		--batch_size 4 \
		--grad_accum_steps 1 \
		--log_steps 10 \
		--save_steps 1_000 \
		--max_steps 1_000_000 \
		--use_fused_adamw \
		--use_flash_attention \
		--gradient_checkpointing \
		--num_workers 4 \
		--compile_model \
		--compile_fullgraph \
		--memory_constrained \
		--rollout_chunk_size 32 \
		--val_split train \
		--val_steps 200 \
		--val_batch_size 8 \
		--max_val_batches 10 \
		--wandb_run_name EBP_large \
		--output_dir output_large

large:
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $(PYTHON) train.py $(LARGE_ARGS)

resume:
	@CKPT=$$(ls -d output_large/step_* 2>/dev/null | sort -t_ -k2 -n | tail -1); \
	if [ -z "$$CKPT" ]; then echo "No checkpoint found in output_large/"; exit 1; fi; \
	echo "Resuming from $$CKPT"; \
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $(PYTHON) train.py $(LARGE_ARGS) \
		--resume_from_checkpoint $$CKPT

# ---------------------------------------------------------------------------
# Additivity experiment: is the feature-matching term additive with CE, or does
# it detract?  AdamW is invariant to the overall loss scale, so GAMMA (the CE
# weight relative to REINFORCE) is the only knob that changes the balance.
#
#   make cpt                      <- the control (CE only, no rollouts)
#   make additive GAMMA=1         <- CE at parity with REINFORCE
#   make additive GAMMA=10        <- CE-dominant
#
# Read reward_std / frac_active_groups from the logs before setting
# MIN_REWARD_STD: it should sit below the bulk of the reward_std distribution.
# ---------------------------------------------------------------------------
GAMMA ?= 1.0
MIN_REWARD_STD ?= 0.0

additive:
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $(PYTHON) train.py \
		--model_name Qwen/Qwen3-0.6B-Base \
		--model_type online \
		--dataset_name data/Nemotron-CC-Math-v1_4plus \
		--tokenized \
		--context_length 512 \
		--generation_length 16 \
		--num_rollouts 8 \
		--batch_size 4 \
		--gamma $(GAMMA) \
		--min_reward_std $(MIN_REWARD_STD) \
		--loss_agg token \
		--no-whitening \
		--warmup_steps 100 \
		--log_steps 100 \
		--save_steps 10_000 \
		--max_steps 50_000 \
		--gradient_checkpointing \
		--use_fused_adamw \
		--use_flash_attention \
		--num_workers 4 \
		--compile_model \
		--memory_constrained \
		--rollout_chunk_size 32 \
		--val_split train \
		--val_steps 500 \
		--val_batch_size 8 \
		--max_val_batches 20 \
		--wandb_run_name EBP_g$(GAMMA) \
		--output_dir output_g$(GAMMA)

small:
	$(PYTHON) train.py \
		--model_name Qwen/Qwen3-0.6B-Base \
		--model_type "online" \
		--dataset_name data/Nemotron-CC-Math-v1_4plus \
		--tokenized \
		--generation_length 4 \
		--num_rollouts 2 \
		--batch_size 2 \
		--gradient_checkpointing \
		--log_steps 10 \
		--save_steps 10_000 \
		--max_steps 1000 \
		--pin_memory \
		--num_workers 4 \
		--persistent_workers \
		--prefetch_factor 2 \
		--use_fused_adamw \
		--use_flash_attention \
		--compile_model \
		--compile_fullgraph \
		--wandb_run_name "Test_Run" \
		--output_dir "test_output"



# 		--memory_constrained

# --pin_memory --num_workers 4 --persistent_workers --prefetch_factor 2

# 		--dtype float16 \

# 		--dataset_name nvidia/Nemotron-CC-Math-v1 \
# 		--dataset_config 4plus \