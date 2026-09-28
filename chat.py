#!/usr/bin/env python3
"""
Interactive text completion script for EBP-trained models.
Loads a model from a checkpoint and provides a simple CLI for completions.
"""

import argparse
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


def main():
    parser = argparse.ArgumentParser(description="Chat with a fine-tuned EBP model.")
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to the model checkpoint (e.g., ./output/step_10000)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling top_p.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda or cpu).",
    )
    args = parser.parse_args()

    print(f"[*] Loading tokenizer from {args.checkpoint_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
    except Exception as e:
        print(f"[!] Error loading tokenizer: {e}")
        sys.exit(1)

    print(f"[*] Loading model from {args.checkpoint_path} onto {args.device}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint_path,
            torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
            device_map=args.device,
        )
    except Exception as e:
        print(f"[!] Error loading model: {e}")
        sys.exit(1)

    model.eval()

    print("\n" + "="*50)
    print(" EBP Model Interactive Completion")
    print(" Type 'exit' or 'quit' to stop.")
    print("="*50 + "\n")

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    while True:
        try:
            prompt = input(">>> ")
        except EOFError:
            break

        if prompt.lower() in ["exit", "quit"]:
            break

        if not prompt.strip():
            continue

        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
        
        print("\nCompletion:", end=" ", flush=True)
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.temperature > 0,
                streamer=streamer,
                pad_token_id=tokenizer.eos_token_id,
            )
        print("\n")


if __name__ == "__main__":
    main()
