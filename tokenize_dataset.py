"""
Unified utility script to tokenize a dataset and save it as a Hugging Face Dataset (Arrow)
for memory-mapped local loading during training.

Supports streaming (for large remote datasets) and non-streaming (for local or cacheable datasets)
modes, with batched tokenization for performance.
"""

import argparse
import os
import fsspec
from aiohttp import ClientTimeout

# Optimize fsspec timeout for large downloads
fsspec.config.conf["http"] = {"client_kwargs": {"timeout": ClientTimeout(total=600000)}}

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize a dataset for EBP training.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace model identifier (to get the tokenizer).",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="allenai/dolma",
    )
    parser.add_argument("--dataset_config", type=str, default="v1_7")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
    )
    parser.add_argument(
        "--min_doc_chars",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--max_documents",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to save the tokenized HF Dataset folder.",
    )
    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to stream the dataset. Use --no-streaming to download fully first.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Batch size for mapping/tokenization.",
    )
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print(f"Loading dataset: {args.dataset_name}/{args.dataset_config} ({args.dataset_split})...")
    load_kwargs = {
        "path": args.dataset_name,
        "name": args.dataset_config,
        "split": args.dataset_split,
        "trust_remote_code": True,
        "streaming": args.streaming,
    }
    
    # Non-streaming specific optimizations
    if not args.streaming:
        load_kwargs["num_proc"] = os.cpu_count()
    else:
        # Stream-specific: disable block size for some backends
        load_kwargs["storage_options"] = {"block_size": 0}

    raw = load_dataset(**load_kwargs)

    # Apply document cap if specified
    if args.max_documents:
        raw = raw.take(args.max_documents)

    def tokenize_function(examples):
        return {"tokens": tokenizer(examples[args.text_column], add_special_tokens=False)["input_ids"]}

    print(f"Filtering and tokenizing (streaming={args.streaming})...")
    
    # Filter for min_doc_chars
    if args.min_doc_chars > 0:
        raw = raw.filter(lambda x: len(x[args.text_column].strip()) >= args.min_doc_chars)

    # Use map with batched=True for speed
    tokenized_dataset = raw.map(
        tokenize_function,
        batched=True,
        batch_size=args.batch_size,
        remove_columns=raw.column_names if not args.streaming else None,
        num_proc=os.cpu_count() if not args.streaming else None,
    )

    # For streaming datasets, removing columns usually happens after map if not supported inside
    if args.streaming:
        tokenized_dataset = tokenized_dataset.remove_columns([c for c in raw.column_names])

    print(f"Saving to {args.output_dir}...")
    
    if args.streaming:
        # For streaming, we need to consume the generator
        if args.max_tokens is not None:
            def token_capped_generator():
                total_tokens = 0
                for item in tokenized_dataset:
                    if total_tokens >= args.max_tokens:
                        break
                    yield item
                    total_tokens += len(item["tokens"])
            ds = Dataset.from_generator(token_capped_generator)
        else:
            ds = Dataset.from_generator(lambda: tokenized_dataset)
        ds.save_to_disk(args.output_dir, num_proc=os.cpu_count())
    else:
        # Non-streaming already has the full dataset object
        raw = tokenized_dataset
        if args.max_tokens:
            # Note: max_tokens is slightly harder to apply efficiently in non-streaming 
            # without iterating, but we can do it post-hoc or via filter.
            # Simplified version:
            print("Warning: --max_tokens is applied by iterating for non-streaming mode.")
            def token_capped_generator():
                total_tokens = 0
                for item in tokenized_dataset:
                    if total_tokens >= args.max_tokens:
                        break
                    yield item
                    total_tokens += len(item["tokens"])
            ds = Dataset.from_generator(token_capped_generator)
            ds.save_to_disk(args.output_dir, num_proc=os.cpu_count())
        else:
            tokenized_dataset.save_to_disk(args.output_dir, num_proc=os.cpu_count())

    print("Done.")

if __name__ == "__main__":
    main()
