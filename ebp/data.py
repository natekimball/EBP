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


def load_dataset(*args, **kwargs):
    """Thin wrapper around ``datasets.load_dataset``.

    Defined at module level so tests can patch ``ebp.data.load_dataset``
    without importing the heavy ``datasets`` library at import time.
    """
    from datasets import load_dataset as _load  # type: ignore[import]

    return _load(*args, **kwargs)


class PretrainingDataset(Dataset):
    """Sliding-window text dataset for EBP pretraining.

    Loads a HuggingFace ``datasets`` dataset, tokenises all documents, and
    creates fixed-length examples by sliding a window over the token stream.
    Documents are separated by the tokeniser's EOS token so that no chunk
    spans two documents, preventing the model from learning spurious
    cross-document dependencies.

    Args:
        tokenizer: HuggingFace tokeniser used for encoding.
        dataset_name: Dataset name passed to ``datasets.load_dataset``.
        dataset_config: Dataset configuration / subset name, or ``None`` if the
            dataset has no named configuration.
        split: Dataset split (e.g. ``"train"``).
        context_length: Number of tokens used as conditioning context.
        completion_length: Number of tokens used as the ground-truth
            completion (and generation target).
        stride: Step size for the sliding window.  Defaults to
            ``context_length + completion_length`` (non-overlapping windows).
        min_doc_chars: Minimum character length required for a document to
            be included.
        text_column: Name of the text column in the dataset.
        streaming: Whether to stream documents from the dataset backend.
        max_documents: Optional cap on number of documents to tokenise.
        max_tokens: Optional cap on total concatenated tokens before chunking.
        max_examples: Optional cap on number of produced training chunks.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset_name: str = "allenai/dolma",
        dataset_config: Optional[str] = "v1_7",
        split: str = "train",
        context_length: int = 128,
        completion_length: int = 32,
        stride: Optional[int] = None,
        min_doc_chars: int = 50,
        text_column: str = "text",
        streaming: bool = True,
        max_documents: Optional[int] = None,
        max_tokens: Optional[int] = None,
        max_examples: Optional[int] = None,
        skip_documents: int = 0,
    ) -> None:
        self.context_length = context_length
        self.completion_length = completion_length
        chunk_size = context_length + completion_length
        stride = stride if stride is not None else chunk_size

        raw = load_dataset(
            dataset_name,
            dataset_config,
            split=split,
            trust_remote_code=True,
            streaming=streaming,
        )

        # EOS token used as a document separator so that no chunk spans
        # two documents.  When eos_token_id is None (rare) we skip separators.
        eos_id: Optional[int] = tokenizer.eos_token_id

        # Stream documents and incrementally build chunks so startup does not
        # require tokenising the entire corpus before creating examples.
        token_buffer: List[int] = []
        buffer_start = 0
        accepted_docs = 0
        total_tokens = 0
        have_previous_doc = False
        docs_skipped = 0
        self.examples: List[List[int]] = []

        for item in raw:
            if docs_skipped < skip_documents:
                docs_skipped += 1
                continue

            if max_documents is not None and accepted_docs >= max_documents:
                break
            if max_tokens is not None and total_tokens >= max_tokens:
                break
            if max_examples is not None and len(self.examples) >= max_examples:
                break

            text = item[text_column]
            if len(text.strip()) < min_doc_chars:
                continue

            tokens = tokenizer.encode(text, add_special_tokens=False)
            doc_tokens: List[int] = tokens
            if have_previous_doc and eos_id is not None:
                doc_tokens = [eos_id] + doc_tokens

            if max_tokens is not None:
                remaining = max_tokens - total_tokens
                if remaining <= 0:
                    break
                doc_tokens = doc_tokens[:remaining]

            if not doc_tokens:
                continue

            token_buffer.extend(doc_tokens)
            total_tokens += len(doc_tokens)
            accepted_docs += 1
            have_previous_doc = True

            while len(token_buffer) - buffer_start >= chunk_size:
                end = buffer_start + chunk_size
                self.examples.append(token_buffer[buffer_start:end])

                if max_examples is not None and len(self.examples) >= max_examples:
                    break

                buffer_start += stride

            if max_examples is not None and len(self.examples) >= max_examples:
                break

            # Periodically trim consumed prefix to keep memory bounded.
            if buffer_start > 4096:
                token_buffer = token_buffer[buffer_start:]
                buffer_start = 0

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
            * ``context_ids``    - ``(B, max_ctx_len)``
            * ``context_mask``   - ``(B, max_ctx_len)``
            * ``completion_ids`` - ``(B, max_comp_len)``
            * ``completion_mask``- ``(B, max_comp_len)``
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
