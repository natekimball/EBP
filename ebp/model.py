"""
EBP Model variants.

Two model classes are provided for experimentation:

* :class:`EMAEBPModel` – the **EMA** variant.  Maintains an Exponential
  Moving Average (EMA) copy of the generator.  The EMA model is used as a
  stable, slowly-evolving feature network.  Features are extracted with a
  dedicated ``@torch.no_grad()`` forward pass through the EMA model, and
  log-probabilities are computed in a separate forward pass through the
  trainable generator.

* :class:`OnlineEBPModel` – the **online** variant.  No EMA copy is kept.
  The generator itself acts as the feature network.  A single forward pass
  extracts hidden-state features (detached via hooks) *and* computes
  differentiable log-probabilities at the same time, halving the number of
  forward passes needed per training step relative to ``EMAEBPModel``.  As
  the generator improves so do the features; the diversity term of the
  feature-matching reward prevents representational collapse.

Both classes expose a common :meth:`compute_rollout_data` interface that
returns ``(rollout_features, rollout_log_probs)`` so that :mod:`train` does
not need to branch on the model type.

Both classes also optimise rollout generation by pre-computing a prefix KV
cache once per batch item and reusing it across all ``num_rollouts`` for that
item, avoiding redundant context re-computation.
"""

from __future__ import annotations

import copy
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


# ---------------------------------------------------------------------------
# Module-level helpers (shared between both model classes)
# ---------------------------------------------------------------------------


def _pool_hidden_state(
    h: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    completion_start: Optional[int],
    pool_type: str = "last",
) -> torch.Tensor:
    """Pool hidden states and L2-normalise.

    Args:
        h: ``(B, L, D)`` hidden state tensor.
        attention_mask: ``(B, L)`` binary mask (1 = real token).
        completion_start: If given, pool only over ``[completion_start, L)``.
        pool_type: Pooling strategy: "last" (default) or "mean".

    Returns:
        ``(B, D)`` L2-normalised pooled vector.
    """
    if pool_type == "mean":
        if completion_start is not None:
            comp_h = h[:, completion_start:, :]  # (B, comp_len, D)
            if attention_mask is not None:
                mask = attention_mask[:, completion_start:].float().unsqueeze(-1)
                pooled = (comp_h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
            else:
                pooled = comp_h.mean(dim=1)
        elif attention_mask is not None:
            mask = attention_mask.float().unsqueeze(-1)
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
        else:
            pooled = h.mean(dim=1)
    else:
        # Default: last-token pooling
        if attention_mask is not None:
            # Get index of the last non-masked token for each sequence
            # (B, L) -> (B,)
            last_indices = attention_mask.sum(dim=1).long() - 1
            # B-sized range: [0, 1, ..., B-1]
            batch_range = torch.arange(h.size(0), device=h.device)
            # Gather (B, D)
            pooled = h[batch_range, last_indices]
        else:
            # Fallback: physical last token
            pooled = h[:, -1, :]

    return F.normalize(pooled, p=2, dim=-1)


def _sum_completion_log_probs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    completion_start: Optional[int],
) -> torch.Tensor:
    """Compute summed log-probabilities of completion tokens from ``logits``.

    Args:
        logits: ``(B, L, V)`` model logits.
        input_ids: ``(B, L)`` token ids (same sequence).
        attention_mask: ``(B, L)`` binary mask.
        completion_start: Index of first completion token.

    Returns:
        ``(B,)`` summed per-token log-probabilities for the completion.
    """
    # Shift: logits[t] predicts input_ids[t+1]
    shift_logits = logits[:, :-1, :]  # (B, L-1, V)
    shift_labels = input_ids[:, 1:]   # (B, L-1)

    log_probs_all = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs_all.gather(
        dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)  # (B, L-1)

    if completion_start is not None:
        # After the left-shift, `token_log_probs[t]` is the log-prob of
        # `input_ids[t+1]`.  To get the log-prob of the first completion
        # token (at index `completion_start`) we therefore need index
        # `completion_start - 1` in the shifted array.
        start = max(0, completion_start - 1)
        token_log_probs = token_log_probs[:, start:]  # (B, comp_len)

        if attention_mask is not None:
            comp_mask = attention_mask[:, completion_start:]
            min_len = min(token_log_probs.shape[1], comp_mask.shape[1])
            token_log_probs = (
                token_log_probs[:, :min_len] * comp_mask[:, :min_len].float()
            )

    return token_log_probs.sum(dim=-1)  # (B,)


def _expand_kv_cache(past_key_values, num_rollouts: int):
    """Expand a KV cache along the batch dimension by ``num_rollouts``.

    Handles three common formats:

    * **transformers >= 5.x** :class:`~transformers.cache_utils.DynamicCache`
      with a ``batch_repeat_interleave`` method (in-place expansion).
    * **transformers 4.38–4.x** :class:`~transformers.cache_utils.DynamicCache`
      with ``key_cache`` / ``value_cache`` list attributes.
    * **Legacy** tuple-of-tuples ``((k0, v0), (k1, v1), ...)``.
    """
    try:
        from transformers.cache_utils import DynamicCache  # type: ignore[import]

        if isinstance(past_key_values, DynamicCache):
            if hasattr(past_key_values, "batch_repeat_interleave"):
                # transformers >= 5.x: in-place API
                past_key_values.batch_repeat_interleave(num_rollouts)
                return past_key_values
            # transformers 4.38–4.x: list-of-tensors API
            new_cache = DynamicCache()
            new_cache.key_cache = [
                k.repeat_interleave(num_rollouts, dim=0)
                for k in past_key_values.key_cache
            ]
            new_cache.value_cache = [
                v.repeat_interleave(num_rollouts, dim=0)
                for v in past_key_values.value_cache
            ]
            return new_cache
    except (ImportError, AttributeError):
        pass

    # Legacy tuple-of-tuples format
    return tuple(
        tuple(t.repeat_interleave(num_rollouts, dim=0) for t in layer)
        for layer in past_key_values
    )


# ---------------------------------------------------------------------------
# Shared architecture helper
# ---------------------------------------------------------------------------


def _get_transformer_layers(model: nn.Module) -> nn.ModuleList:
    """Return the ``nn.ModuleList`` of transformer decoder blocks.

    Supports the most common HuggingFace causal-LM architectures:

    * Qwen / LLaMA / Mistral / Gemma -> ``model.model.layers``
    * GPT-2 -> ``model.transformer.h``
    * GPT-NeoX / Pythia -> ``model.gpt_neox.layers``
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise ValueError(
        f"Cannot locate transformer layer list for {type(model).__name__}. "
        "Pass a supported architecture (Qwen/LLaMA/Mistral/GPT-2/GPT-NeoX)."
    )


def _features_from_hidden_states(
    hidden_states: Tuple[torch.Tensor, ...],
    feature_layer_indices: Sequence[int],
    attention_mask: Optional[torch.Tensor],
    completion_start: Optional[int],
    detach: bool,
    pool_type: str = "last",
) -> torch.Tensor:
    """Build concatenated pooled features from model hidden states.

    HuggingFace models return ``hidden_states`` as a tuple where index 0 is
    the embedding output and index ``i + 1`` corresponds to transformer block
    ``i``.  ``feature_layer_indices`` are block indices.
    """
    blocks = []
    for idx in feature_layer_indices:
        h = hidden_states[idx + 1]
        if detach:
            h = h.detach()
        blocks.append(_pool_hidden_state(h, attention_mask, completion_start, pool_type=pool_type))
    return torch.cat(blocks, dim=-1)


# ---------------------------------------------------------------------------
# EMAEBPModel
# ---------------------------------------------------------------------------


class EMAEBPModel(nn.Module):
    """Energy-Based Pre-training with an EMA feature network.

    Holds two copies of a causal LM:

    * ``model`` - the trainable generator ``p_theta``.
    * ``ema_model`` - an Exponential Moving Average of ``model``, used as the
      feature network ``phi``.  Updated via :meth:`update_ema` after every
      gradient step; no gradient ever flows through it.

    Feature extraction follows the EBFT paper: hidden states at layers placed
    at depths ``feature_layer_fractions`` of the network are pooled (last-token
    or mean), then L2-normalised per layer, and finally concatenated into a
    single feature vector.

    Rollout generation uses a **prefix KV cache**: the KV cache is computed
    once for each batch item's context and reused across all ``num_rollouts``
    for that item, avoiding redundant context re-computation.

    Args:
        model_name: HuggingFace model identifier (default: ``"Qwen/Qwen3-0.6B"``).
        ema_decay: EMA decay factor (``ema <- decay*ema + (1-decay)*theta``).
        feature_layer_fractions: Relative depths at which to capture hidden
            states, e.g. ``(0.25, 0.50, 0.75)``.
        pool_type: Pooling strategy for hidden-state features (default: ``"last"``,
            or ``"mean"``).
        model: Optional pre-instantiated model; if supplied, *model_name* is
            ignored.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        ema_decay: float = 0.999,
        feature_layer_fractions: Sequence[float] = (0.25, 0.50, 0.75),
        pool_type: str = "last",
        model: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        if model is not None:
            self.model: nn.Module = model
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # EMA copy - no gradients ever flow through this
        self.ema_model: nn.Module = copy.deepcopy(self.model)
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

        self.ema_decay = ema_decay
        self.feature_layer_fractions = tuple(feature_layer_fractions)
        self.pool_type = pool_type

        num_layers = len(_get_transformer_layers(self.model))
        self.feature_layer_indices: List[int] = [
            max(0, min(round(f * num_layers) - 1, num_layers - 1))
            for f in self.feature_layer_fractions
        ]
        self._ema_layers = _get_transformer_layers(self.ema_model)

    # ------------------------------------------------------------------
    # Forward (trainable generator)
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ):
        """Standard HuggingFace forward through the trainable generator."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=use_cache,
        )

    # ------------------------------------------------------------------
    # Feature extraction (EMA model, no grad)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        completion_start: Optional[int] = None,
    ) -> torch.Tensor:
        """Extract features from the EMA model (no gradient).

        Runs a single forward pass through ``ema_model`` with
        ``output_hidden_states=True`` and returns the concatenation of the
        per-layer mean-pooled, L2-normalised hidden states.

        Args:
            input_ids: ``(B, L)`` token ids.
            attention_mask: ``(B, L)`` binary mask.
            completion_start: If given, pool only over ``[completion_start, L)``.

        Returns:
            ``(B, D * num_feature_layers)`` feature tensor (detached).
        """
        outputs = self.ema_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )

        return _features_from_hidden_states(
            hidden_states=outputs.hidden_states,
            feature_layer_indices=self.feature_layer_indices,
            attention_mask=attention_mask,
            completion_start=completion_start,
            detach=True,
            pool_type=self.pool_type,
        )  # (B, D * K)

    # ------------------------------------------------------------------
    # Log-probability computation (trainable model, with grad)
    # ------------------------------------------------------------------

    def compute_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        completion_start: Optional[int] = None,
    ) -> torch.Tensor:
        """Sum log-probabilities of the completion tokens under the generator.

        Args:
            input_ids: ``(B, L)`` full sequence (context + completion).
            attention_mask: ``(B, L)`` binary mask.
            completion_start: Index of the first completion token.

        Returns:
            ``(B,)`` sum of per-token log-probabilities for the completion
            (with gradients).
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return _sum_completion_log_probs(
            outputs.logits, input_ids, attention_mask, completion_start
        )

    def forward_ce_and_ref_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        completion_start: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return CE loss and detached reference features for EMA variant.

        For :class:`EMAEBPModel`, this intentionally remains a two-forward
        implementation: CE is computed on the trainable generator while
        reference features are computed via ``ema_model``.

        Args:
            input_ids: ``(B, L)`` full sequence (context + completion).
            attention_mask: ``(B, L)`` binary mask.
            completion_start: Index of first completion token.

        Returns:
            ce_loss: Scalar CE loss tensor with gradients.
            ref_features: ``(B, D * K)`` detached EMA reference features.
        """
        ce_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
            use_cache=False,
        ).loss
        ref_features = self.extract_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            completion_start=completion_start,
        )
        return ce_loss, ref_features

    # ------------------------------------------------------------------
    # Combined rollout data (features from EMA + log probs from generator)
    # ------------------------------------------------------------------

    def compute_rollout_data(
        self,
        rollout_ids: torch.Tensor,
        rollout_masks: torch.Tensor,
        completion_start: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return features and log-probs for all rollouts.

        Performs two forward passes: one through the EMA model for features
        (no grad), and one through the trainable generator for log-probs
        (with grad).  Both passes process all ``B * n`` rollouts at once.

        Args:
            rollout_ids: ``(B * n, context_len + gen_len)`` rollout token ids.
            rollout_masks: ``(B * n, context_len + gen_len)`` attention masks.
            completion_start: Index of the first generated token (= context_len).

        Returns:
            features: ``(B * n, feat_dim)`` - detached feature vectors.
            log_probs: ``(B * n,)`` - differentiable summed log-probabilities.
        """
        features = self.extract_features(rollout_ids, rollout_masks, completion_start)
        log_probs = self.compute_log_probs(rollout_ids, rollout_masks, completion_start)
        return features, log_probs

    # ------------------------------------------------------------------
    # Rollout generation with prefix KV cache (no grad)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_rollouts(
        self,
        context_ids: torch.Tensor,
        context_attention_mask: torch.Tensor,
        num_rollouts: int,
        generation_length: int,
        temperature: float = 1.0,
        **generate_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample ``num_rollouts`` completions of length ``generation_length``.

        The context prefix KV cache is computed **once per batch item** and
        then expanded for ``num_rollouts``, avoiding redundant re-computation
        of the context for each rollout.

        Args:
            context_ids: ``(B, context_len)`` context token ids.
            context_attention_mask: ``(B, context_len)`` binary mask.
            num_rollouts: Number of completions to sample per context.
            generation_length: Number of new tokens to generate.
            temperature: Sampling temperature.
            **generate_kwargs: Extra keyword arguments for ``model.generate``.

        Returns:
            rollout_ids:   ``(B * num_rollouts, context_len + generation_length)``
            rollout_masks: ``(B * num_rollouts, context_len + generation_length)``
        """
        return _generate_with_kv_cache(
            self.model,
            context_ids,
            context_attention_mask,
            num_rollouts,
            generation_length,
            temperature,
            **generate_kwargs,
        )

    # ------------------------------------------------------------------
    # EMA update (stop-gradient)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_ema(self) -> None:
        """Update EMA model parameters after a gradient step.

        ``ema_param <- decay * ema_param + (1 - decay) * param``

        Non-parameter buffers (e.g. layer-norm running stats) are copied
        directly so they stay in sync with the generator.
        """
        for ema_p, p in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_p.data.mul_(self.ema_decay).add_(p.data, alpha=1.0 - self.ema_decay)

        for ema_buf, buf in zip(self.ema_model.buffers(), self.model.buffers()):
            ema_buf.data.copy_(buf.data)


# ---------------------------------------------------------------------------
# OnlineEBPModel
# ---------------------------------------------------------------------------


class OnlineEBPModel(nn.Module):
    """Energy-Based Pre-training using the live model as the feature network.

    Unlike :class:`EMAEBPModel`, no EMA copy is maintained.  The trainable
    generator itself serves as the feature network, so features naturally
    improve as training progresses.  The **diversity term** in the
    feature-matching reward prevents representational collapse.

    The key efficiency advantage over ``EMAEBPModel`` is
    :meth:`extract_features_and_log_probs`: a **single forward pass** through
    the generator simultaneously captures hidden-state features (detached via
    hooks) and computes differentiable log-probabilities from the output
    logits.  This halves the number of rollout-processing forward passes
    compared to ``EMAEBPModel``.

    Args:
        model_name: HuggingFace model identifier (default: ``"Qwen/Qwen3-0.6B"``).
        feature_layer_fractions: Relative depths at which to capture hidden
            states.
        pool_type: Pooling strategy for hidden-state features (default: ``"last"``,
            or ``"mean"``).
        model: Optional pre-instantiated model; if supplied, *model_name* is
            ignored.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        feature_layer_fractions: Sequence[float] = (0.25, 0.50, 0.75),
        pool_type: str = "last",
        model: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        if model is not None:
            self.model: nn.Module = model
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.feature_layer_fractions = tuple(feature_layer_fractions)
        self.pool_type = pool_type

        num_layers = len(_get_transformer_layers(self.model))
        self.feature_layer_indices: List[int] = [
            max(0, min(round(f * num_layers) - 1, num_layers - 1))
            for f in self.feature_layer_fractions
        ]
        # Kept for compatibility with tests that inspect selected layer range.
        self._model_layers = _get_transformer_layers(self.model)

    # ------------------------------------------------------------------
    # Forward (trainable generator)
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ):
        """Standard HuggingFace forward through the trainable generator."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=use_cache,
        )

    # ------------------------------------------------------------------
    # Feature extraction (live model, no grad - for reference features)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        completion_start: Optional[int] = None,
    ) -> torch.Tensor:
        """Extract features from the live model without gradients.

        Used to compute the **reference** feature vector ``phi(c:y)`` for the
        ground-truth completion.  Gradients are not needed here because the
        reference features are used only as fixed targets.

        Args:
            input_ids: ``(B, L)`` token ids.
            attention_mask: ``(B, L)`` binary mask.
            completion_start: If given, pool only over ``[completion_start, L)``.

        Returns:
            ``(B, D * num_feature_layers)`` feature tensor (detached).
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )

        return _features_from_hidden_states(
            hidden_states=outputs.hidden_states,
            feature_layer_indices=self.feature_layer_indices,
            attention_mask=attention_mask,
            completion_start=completion_start,
            detach=True,
            pool_type=self.pool_type,
        )

    # ------------------------------------------------------------------
    # Combined feature extraction + log-prob computation (with grad)
    # ------------------------------------------------------------------

    def extract_features_and_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        completion_start: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single forward pass returning both features and log-probabilities.

        A single forward call returns both hidden states and logits. Features
        are detached immediately so no gradient flows through them, while
        logits retain the graph for REINFORCE gradients.

        This is the core efficiency gain of :class:`OnlineEBPModel`: instead
        of two separate forward passes (EMA for features + generator for
        log-probs), a single pass provides both.

        Args:
            input_ids: ``(B, L)`` full sequence (context + completion).
            attention_mask: ``(B, L)`` binary mask.
            completion_start: Index of the first completion token.

        Returns:
            features:  ``(B, D * K)`` - detached feature vectors (no grad).
            log_probs: ``(B,)`` - differentiable summed log-probabilities.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )

        features = _features_from_hidden_states(
            hidden_states=outputs.hidden_states,
            feature_layer_indices=self.feature_layer_indices,
            attention_mask=attention_mask,
            completion_start=completion_start,
            detach=True,
            pool_type=self.pool_type,
        )  # (B, D * K), detached

        log_probs = _sum_completion_log_probs(
            outputs.logits, input_ids, attention_mask, completion_start
        )  # (B,), with grad

        return features, log_probs

    def forward_ce_and_ref_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        completion_start: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single forward pass returning CE loss and detached reference features.

        Used by the memory-constrained training schedule to compute the CE
        term and online reference features together in one full-sequence pass.

        Args:
            input_ids: ``(B, L)`` full sequence (context + completion).
            attention_mask: ``(B, L)`` binary mask.
            completion_start: Index of the first completion token.

        Returns:
            ce_loss: Scalar CE loss tensor with gradients.
            ref_features: ``(B, D * K)`` detached reference features.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )

        ref_features = _features_from_hidden_states(
            hidden_states=outputs.hidden_states,
            feature_layer_indices=self.feature_layer_indices,
            attention_mask=attention_mask,
            completion_start=completion_start,
            detach=True,
            pool_type=self.pool_type,
        )
        return outputs.loss, ref_features

    # ------------------------------------------------------------------
    # Combined rollout data (features + log probs in one forward pass)
    # ------------------------------------------------------------------

    def compute_rollout_data(
        self,
        rollout_ids: torch.Tensor,
        rollout_masks: torch.Tensor,
        completion_start: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return features and log-probs for all rollouts in a single pass.

        Delegates to :meth:`extract_features_and_log_probs`, combining what
        would be two separate forward passes in :class:`EMAEBPModel` into one.

        Args:
            rollout_ids: ``(B * n, context_len + gen_len)`` rollout token ids.
            rollout_masks: ``(B * n, context_len + gen_len)`` attention masks.
            completion_start: Index of the first generated token.

        Returns:
            features: ``(B * n, feat_dim)`` - detached feature vectors.
            log_probs: ``(B * n,)`` - differentiable summed log-probabilities.
        """
        return self.extract_features_and_log_probs(
            rollout_ids, rollout_masks, completion_start
        )

    # ------------------------------------------------------------------
    # Rollout generation with prefix KV cache (no grad)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_rollouts(
        self,
        context_ids: torch.Tensor,
        context_attention_mask: torch.Tensor,
        num_rollouts: int,
        generation_length: int,
        temperature: float = 1.0,
        **generate_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample ``num_rollouts`` completions of length ``generation_length``.

        Uses prefix KV caching identical to :class:`EMAEBPModel`.

        Args:
            context_ids: ``(B, context_len)`` context token ids.
            context_attention_mask: ``(B, context_len)`` binary mask.
            num_rollouts: Number of completions per context.
            generation_length: Number of new tokens to generate.
            temperature: Sampling temperature.
            **generate_kwargs: Extra keyword arguments for ``model.generate``.

        Returns:
            rollout_ids:   ``(B * num_rollouts, context_len + generation_length)``
            rollout_masks: ``(B * num_rollouts, context_len + generation_length)``
        """
        return _generate_with_kv_cache(
            self.model,
            context_ids,
            context_attention_mask,
            num_rollouts,
            generation_length,
            temperature,
            **generate_kwargs,
        )


# ---------------------------------------------------------------------------
# Shared rollout generation with prefix KV cache
# ---------------------------------------------------------------------------


@torch.no_grad()
def _generate_with_kv_cache(
    model: nn.Module,
    context_ids: torch.Tensor,
    context_mask: torch.Tensor,
    num_rollouts: int,
    generation_length: int,
    temperature: float = 1.0,
    **generate_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate rollouts reusing a shared prefix KV cache.

    The context prefix KV cache is computed **once per batch item** and then
    replicated ``num_rollouts`` times, so the context self-attention is
    computed only once regardless of how many rollouts are requested.

    When ``context_len <= 1`` (degenerate case), falls back to the naive
    approach (no prefix caching).

    Implementation details:

    1. Run a forward pass on ``context_ids[:, :-1]`` to populate the KV cache
       for positions ``0 ... context_len - 2``.
    2. Replicate the cache ``num_rollouts`` times along the batch dimension.
    3. Call ``model.generate()`` with ``input_ids = context_ids[:, -1:]``
       (the last context token) and ``past_key_values = expanded_cache``.
       Generation then starts at position ``context_len - 1`` and produces
       ``generation_length`` new tokens.
    4. Reconstruct the full ``(B * n, context_len + generation_length)``
       sequences by prepending the original context.

    Args:
        model: The causal LM whose ``generate`` method will be called.
        context_ids: ``(B, context_len)``
        context_mask: ``(B, context_len)``
        num_rollouts: Rollouts per context item.
        generation_length: Tokens to generate per rollout.
        temperature: Sampling temperature.
        **generate_kwargs: Forwarded to ``model.generate``.

    Returns:
        full_ids:      ``(B * num_rollouts, context_len + generation_length)``
        rollout_masks: ``(B * num_rollouts, context_len + generation_length)``
    """
    context_len = context_ids.shape[1]

    # Expand the attention mask once; used in both branches.
    expanded_mask = context_mask.repeat_interleave(num_rollouts, dim=0)

    if context_len > 1:
        # ----------------------------------------------------------------
        # Optimised path: compute context KV cache once and reuse it.
        # ----------------------------------------------------------------
        prefix_out = model(
            input_ids=context_ids[:, :-1],
            attention_mask=context_mask[:, :-1],
            use_cache=True,
            return_dict=True,
        )

        expanded_pkv = _expand_kv_cache(prefix_out.past_key_values, num_rollouts)

        # Last context token serves as the first input to generate()
        last_tokens = context_ids[:, -1:].repeat_interleave(num_rollouts, dim=0)

        output_ids = model.generate(
            input_ids=last_tokens,
            attention_mask=expanded_mask,
            past_key_values=expanded_pkv,
            max_new_tokens=generation_length,
            do_sample=True,
            temperature=temperature,
            use_cache=True,
            pad_token_id=model.config.eos_token_id,
            **generate_kwargs,
        )
        # output_ids: (B*n, 1 + generation_length)
        # output_ids[:, 0] is the last context token; drop it and prepend
        # the full context to reconstruct the complete sequences.
        context_expanded = context_ids.repeat_interleave(num_rollouts, dim=0)
        gen_tokens = output_ids[:, 1:]  # (B*n, generation_length)
        full_ids = torch.cat([context_expanded, gen_tokens], dim=1)

    else:
        # ----------------------------------------------------------------
        # Fallback: standard generate (no prefix to cache).
        # ----------------------------------------------------------------
        expanded_ids = context_ids.repeat_interleave(num_rollouts, dim=0)

        output_ids = model.generate(
            input_ids=expanded_ids,
            attention_mask=expanded_mask,
            max_new_tokens=generation_length,
            do_sample=True,
            temperature=temperature,
            use_cache=True,
            pad_token_id=model.config.eos_token_id,
            **generate_kwargs,
        )
        full_ids = output_ids

    # Build attention mask for the full (context + generated) sequences
    gen_len = full_ids.shape[1] - context_len
    new_mask = torch.ones(
        full_ids.shape[0], gen_len, dtype=torch.long, device=full_ids.device
    )
    rollout_masks = torch.cat([expanded_mask, new_mask], dim=1)

    return full_ids, rollout_masks
