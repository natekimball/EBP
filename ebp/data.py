"""
Pretraining dataset: chunks text into (context, completion) pairs.

Each example consists of a fixed-length context prefix and a continuation
of length ``completion_length``.  The context is fed to the model as a
conditioning prefix; the continuation is used both as the ground-truth
completion for feature matching *and* as the supervised target for the
cross-entropy term.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class PretrainingDataset(Dataset):
    """Sliding-window text dataset for EBP pretraining.

    Loads a HuggingFace ``datasets`` dataset, tokenises all documents, and
    creates fixed-length examples by sliding a window over the token stream.

    Args:
        tokenizer: HuggingFace tokeniser used for encoding.
        dataset_name: Dataset name passed to ``datasets.load_dataset``.
        dataset_config: Dataset configuration / subset name.
        split: Dataset split (e.g. ``"train"``).
        context_length: Number of tokens used as conditioning context.
        completion_length: Number of tokens used as the ground-truth
            completion (and generation target).
        stride: Step size for the sliding window.  Defaults to
            ``context_length`` (non-overlapping windows).
        min_doc_chars: Minimum character length required for a document to
            be included.
        text_column: Name of the text column in the dataset.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-2-raw-v1",
        split: str = "train",
        context_length: int = 128,
        completion_length: int = 32,
        stride: Optional[int] = None,
        min_doc_chars: int = 50,
        text_column: str = "text",
    ) -> None:
        from datasets import load_dataset  # local import to keep top-level fast

        self.context_length = context_length
        self.completion_length = completion_length
        chunk_size = context_length + completion_length
        stride = stride if stride is not None else chunk_size

        raw = load_dataset(dataset_name, dataset_config, split=split)

        # Concatenate all document tokens into one long token stream
        all_tokens: List[int] = []
        for item in raw:
            text = item[text_column]
            if len(text.strip()) < min_doc_chars:
                continue
            tokens = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)

        # Create fixed-length chunks via sliding window
        self.examples: List[List[int]] = []
        for start in range(0, len(all_tokens) - chunk_size + 1, stride):
            self.examples.append(all_tokens[start : start + chunk_size])

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        tokens = self.examples[idx]
        return {
            "context_ids": tokens[: self.context_length],
            "completion_ids": tokens[self.context_length :],
        }


def collate_fn(
    batch: List[Dict[str, List[int]]],
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    """Collate a list of dataset items into padded tensors.

    Context sequences are left-padded; completion sequences are right-padded.
    Padding positions are indicated by a zero in the corresponding mask.

    Args:
        batch: List of dicts with keys ``context_ids`` and ``completion_ids``.
        pad_token_id: Token id used for padding.

    Returns:
        Dict with keys:
            * ``context_ids``    – ``(B, max_ctx_len)``
            * ``context_mask``   – ``(B, max_ctx_len)``
            * ``completion_ids`` – ``(B, max_comp_len)``
            * ``completion_mask``– ``(B, max_comp_len)``
    """
    max_ctx = max(len(b["context_ids"]) for b in batch)
    max_comp = max(len(b["completion_ids"]) for b in batch)

    ctx_ids_list, ctx_mask_list = [], []
    comp_ids_list, comp_mask_list = [], []

    for item in batch:
        ctx = item["context_ids"]
        comp = item["completion_ids"]

        # Left-pad context
        pad_len = max_ctx - len(ctx)
        ctx_ids_list.append([pad_token_id] * pad_len + ctx)
        ctx_mask_list.append([0] * pad_len + [1] * len(ctx))

        # Right-pad completion
        pad_len = max_comp - len(comp)
        comp_ids_list.append(comp + [pad_token_id] * pad_len)
        comp_mask_list.append([1] * len(comp) + [0] * pad_len)

    return {
        "context_ids": torch.tensor(ctx_ids_list, dtype=torch.long),
        "context_mask": torch.tensor(ctx_mask_list, dtype=torch.long),
        "completion_ids": torch.tensor(comp_ids_list, dtype=torch.long),
        "completion_mask": torch.tensor(comp_mask_list, dtype=torch.long),
    }
