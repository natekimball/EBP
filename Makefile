train:
	python train.py \
		--model_type online \
		--dataset_name data/Nemotron-CC-Math-v1_4plus \
		--tokenized \
		--context_length 1024 \
		--generation_length 32 \
		--num_rollouts 16 \
		--batch_size 16 \
		--dtype float16 \
		--gradient_checkpointing \
		--log_steps 1_000 \
		--save_steps 10_000 \
		--max_steps 1_000_000 \
		--gradient_checkpointing \
		--use_fused_adamw \
		--use_flash_attention \
		--compile_model \
		--memory_constrained \
		--num_workers 4 \
		--val_split test \
		--val_steps 1000 \
		--max_val_batches 100

small:
	python train.py \
		--model_type "online" \
		--dataset_name data/Nemotron-CC-Math-v1_4plus \
		--tokenized \
		--generation_length 4 \
		--num_rollouts 2 \
		--batch_size 2 \
		--gradient_checkpointing \
		--max_steps 10 \
		--output_dir "test_output" \
		--pin_memory \
		--num_workers 4 \
		--persistent_workers \
		--prefetch_factor 2 \
		--use_fused_adamw \
		--use_flash_attention \
		--compile_model \



# 		--memory_constrained

# --pin_memory --num_workers 4 --persistent_workers --prefetch_factor 2

test:
	python train.py \
		--model_type online \
		--dataset_name data/Nemotron-CC-Math-v1_4plus \
		--tokenized \
		--context_length 1024 \
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
		--val_split test \
		--val_steps 1000 \
		--val_batch_size 8 \
		--max_val_batches 100
		
# 		--log_steps 500 \
# 		--val_steps 1000 \

# 	 --val_split train --val_steps 100 --val_batch_size 8
# 		--num_workers 4 \

# 		--dtype float16 \

# 		--dataset_name nvidia/Nemotron-CC-Math-v1 \
# 		--dataset_config 4plus \