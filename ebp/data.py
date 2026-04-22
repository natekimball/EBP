"""
Pretraining dataset: chunks text into (context, completion) pairs.

Each example consists of a fixed-length context prefix and a continuation
of length ``completion_length``.  The context is fed to the model as a
conditioning prefix; the continuation is used both as the ground-truth
completion for feature matching *and* as the supervised target for the
cross-entropy term.
"""

from __future__ import annotations

import os
from typing import Dict, Iterator, List, Optional, Union

import torch
from torch.utils.data import IterableDataset, get_worker_info
from transformers import PreTrainedTokenizerBase


def load_dataset(*args, **kwargs):
    """Thin wrapper around ``datasets.load_dataset``.

    Defined at module level so tests can patch ``ebp.data.load_dataset``
    without importing the heavy ``datasets`` library at import time.
    """
    from datasets import load_dataset as _load  # type: ignore[import]

    return _load(*args, **kwargs)


def load_from_disk(*args, **kwargs):
    """Thin wrapper around ``datasets.load_from_disk``.

    Defined at module level so tests can patch ``ebp.data.load_from_disk``
    without importing the heavy ``datasets`` library at import time.
    """
    from datasets import load_from_disk as _load  # type: ignore[import]

    return _load(*args, **kwargs)


class PretrainingDataset(IterableDataset):
    """Sliding-window text dataset for EBP pretraining.

    Loads a HuggingFace ``datasets`` dataset, tokenises all documents, and
    creates fixed-length examples by sliding a window over the token stream.
    Documents are separated by the tokeniser's EOS token so that no chunk
    spans two documents, preventing the model from learning spurious
    cross-document dependencies.

    Args:
        tokenizer: HuggingFace tokeniser used for encoding.
        dataset_name: Dataset name passed to ``datasets.load_dataset``, or 
            path to a local directory containing a saved dataset.
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
        max_tokens: Optional cap on total concatenated tokens before chunking.
        max_examples: Optional cap on number of produced training chunks.
        tokenized: Whether the dataset already contains token IDs in the 
            ``text_column``. If True, skips the tokenizer.
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
        tokenized: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.context_length = context_length
        self.completion_length = completion_length
        self.chunk_size = context_length + completion_length
        self.stride = stride if stride is not None else self.chunk_size
        self.min_doc_chars = min_doc_chars
        self.text_column = text_column
        self.streaming = streaming
        self.max_documents = max_documents
        self.max_tokens = max_tokens
        self.max_examples = max_examples
        self.skip_documents = skip_documents
        self.tokenized = tokenized

        # Auto-detect if loading from local disk
        self.is_cached = os.path.isdir(dataset_name)

    def __iter__(self) -> Iterator[Dict[str, List[int]]]:
        worker_info = get_worker_info()

        if self.is_cached:
            raw = load_from_disk(self.dataset_name)
            # If a split is specified and it's a DatasetDict, pick it
            from datasets import DatasetDict
            if isinstance(raw, DatasetDict) and self.split in raw:
                raw = raw[self.split]
        else:
            raw = load_dataset(
                self.dataset_name,
                self.dataset_config,
                split=self.split,
                trust_remote_code=True,
                streaming=self.streaming,
            )

        # Multi-worker sharding
        if worker_info is not None:
            try:
                # HuggingFace streaming datasets often support .shard()
                raw = raw.shard(num_shards=worker_info.num_workers, index=worker_info.id)
            except Exception:
                # Fallback: simple modulo skip if .shard() isn't supported or fails
                import itertools

                raw = itertools.islice(raw, worker_info.id, None, worker_info.num_workers)

        eos_id: Optional[int] = self.tokenizer.eos_token_id

        token_buffer: List[int] = []
        buffer_start = 0
        accepted_docs = 0
        total_tokens = 0
        examples_yielded = 0
        docs_skipped = 0
        have_previous_doc = False

        for item in raw:
            if docs_skipped < self.skip_documents:
                docs_skipped += 1
                continue

            if self.max_documents is not None and accepted_docs >= self.max_documents:
                break
            if self.max_tokens is not None and total_tokens >= self.max_tokens:
                break

            data = item[self.text_column]
            if self.tokenized:
                # Expecting a list/tensor of token IDs
                doc_tokens = list(data)
            else:
                # Expecting text, needs tokenization
                if len(data.strip()) < self.min_doc_chars:
                    continue
                doc_tokens = self.tokenizer.encode(data, add_special_tokens=False)

            if have_previous_doc and eos_id is not None:
                doc_tokens = [eos_id] + doc_tokens

            if self.max_tokens is not None:
                remaining = self.max_tokens - total_tokens
                if remaining <= 0:
                    break
                doc_tokens = doc_tokens[:remaining]

            if not doc_tokens:
                continue

            token_buffer.extend(doc_tokens)
            total_tokens += len(doc_tokens)
            accepted_docs += 1
            have_previous_doc = True

            while len(token_buffer) - buffer_start >= self.chunk_size:
                end = buffer_start + self.chunk_size
                tokens_out = token_buffer[buffer_start:end]

                yield {
                    "context_ids": tokens_out[: self.context_length],
                    "completion_ids": tokens_out[self.context_length :],
                }

                examples_yielded += 1
                if self.max_examples is not None and examples_yielded >= self.max_examples:
                    return

                buffer_start += self.stride

            # Periodically trim consumed prefix to keep memory bounded.
            if buffer_start > 4096:
                token_buffer = token_buffer[buffer_start:]
                buffer_start = 0


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
