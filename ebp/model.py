"""
EBP Model variants.

Three classes are provided:

* :class:`BaseEBPModel` – shared base holding the trainable generator,
  feature-layer configuration, and the :meth:`generate_rollouts` sampling
  loop.  Not intended for direct use.

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

Both concrete classes expose a common :meth:`compute_rollout_data` interface
that returns ``(rollout_features, rollout_log_probs)`` so that :mod:`train`
does not need to branch on the model type.

Rollout generation is shared via the base class and uses
:class:`StaticCache`, which pre-allocates fixed-shape KV tensors so
generation stays O(1) per token and CUDA graph capture inside ``generate()``
remains possible under compilation.
"""

from __future__ import annotations

import copy
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, StaticCache


# ---------------------------------------------------------------------------
# Module-level helpers
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
            comp_h = h[:, completion_start:, :]
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
        # last-token pooling
        if attention_mask is not None:
            last_indices = attention_mask.sum(dim=1).long() - 1
            batch_range = torch.arange(h.size(0), device=h.device)
            pooled = h[batch_range, last_indices]
        else:
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
        # token_log_probs[t] is the log-prob of input_ids[t+1]; to get the
        # log-prob of the first completion token (index completion_start) we
        # need offset completion_start - 1 in the shifted array.
        start = max(0, completion_start - 1)
        token_log_probs = token_log_probs[:, start:]

        if attention_mask is not None:
            comp_mask = attention_mask[:, completion_start:]
            min_len = min(token_log_probs.shape[1], comp_mask.shape[1])
            token_log_probs = (
                token_log_probs[:, :min_len] * comp_mask[:, :min_len].float()
            )

    return token_log_probs.sum(dim=-1)  # (B,)


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
# BaseEBPModel
# ---------------------------------------------------------------------------


class BaseEBPModel(nn.Module):
    """Shared base for EMA and Online EBP model variants.

    Holds the trainable generator and common configuration (feature-layer
    fractions, pooling type).  Provides the :meth:`generate_rollouts` sampling
    loop used by both subclasses.

    Args:
        model_name: HuggingFace model identifier (default: ``"Qwen/Qwen3-0.6B"``).
        feature_layer_fractions: Relative depths at which to capture hidden
            states, e.g. ``(0.25, 0.50, 0.75)``.
        pool_type: Pooling strategy for hidden-state features (``"last"`` or
            ``"mean"``).
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

        self.model: nn.Module = (
            model if model is not None
            else AutoModelForCausalLM.from_pretrained(model_name)
        )
        self.feature_layer_fractions = tuple(feature_layer_fractions)
        self.pool_type = pool_type

        num_layers = len(_get_transformer_layers(self.model))
        self.feature_layer_indices: List[int] = [
            max(0, min(round(f * num_layers) - 1, num_layers - 1))
            for f in self.feature_layer_fractions
        ]

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

    @torch.no_grad()
    def generate_rollouts(
        self,
        context_ids: torch.Tensor,
        context_attention_mask: torch.Tensor,
        num_rollouts: int,
        generation_length: int,
        temperature: float = 1.0,
        use_cache: bool = True,
        **generate_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample ``num_rollouts`` completions of length ``generation_length``.

        Args:
            context_ids: ``(B, context_len)`` context token ids.
            context_attention_mask: ``(B, context_len)`` binary mask.
            num_rollouts: Number of completions per context.
            generation_length: Number of new tokens to generate.
            temperature: Sampling temperature.
            use_cache: If True, uses StaticCache for O(1) per-token cost and
                stable tensor shapes compatible with CUDA graph capture.
            **generate_kwargs: Extra keyword arguments for ``model.generate``.

        Returns:
            rollout_ids:   ``(B * num_rollouts, context_len + generation_length)``
            rollout_masks: ``(B * num_rollouts, context_len + generation_length)``
        """
        expanded_ids = context_ids.repeat_interleave(num_rollouts, dim=0)
        expanded_mask = context_attention_mask.repeat_interleave(num_rollouts, dim=0)
        context_len = context_ids.shape[1]

        generate_kwargs_common = dict(
            input_ids=expanded_ids,
            attention_mask=expanded_mask,
            max_new_tokens=generation_length,
            do_sample=True,
            temperature=temperature,
            pad_token_id=self.model.config.eos_token_id,
            **generate_kwargs,
        )

        if use_cache:
            pkv = StaticCache(
                config=self.model.config,
                batch_size=expanded_ids.shape[0],
                max_cache_len=context_len + generation_length,
                device=self.model.device,
                dtype=self.model.dtype,
            )
            output_ids = self.model.generate(
                **generate_kwargs_common, past_key_values=pkv, use_cache=True
            )
        else:
            output_ids = self.model.generate(**generate_kwargs_common, use_cache=False)

        gen_len = output_ids.shape[1] - context_len
        new_mask = torch.ones(
            output_ids.shape[0], gen_len, dtype=torch.long, device=output_ids.device
        )
        rollout_masks = torch.cat([expanded_mask, new_mask], dim=1)
        return output_ids, rollout_masks


# ---------------------------------------------------------------------------
# EMAEBPModel
# ---------------------------------------------------------------------------


class EMAEBPModel(BaseEBPModel):
    """Energy-Based Pre-training with an EMA feature network.

    Extends :class:`BaseEBPModel` with an Exponential Moving Average copy of
    the generator (``ema_model``) that serves as a stable, slowly-evolving
    feature network.  No gradient ever flows through the EMA model.

    Feature extraction follows the EBFT paper: hidden states at layers placed
    at depths ``feature_layer_fractions`` of the network are pooled (last-token
    or mean), then L2-normalised per layer, and finally concatenated into a
    single feature vector.

    Args:
        model_name: HuggingFace model identifier (default: ``"Qwen/Qwen3-0.6B"``).
        ema_decay: EMA decay factor (``ema <- decay*ema + (1-decay)*theta``).
        feature_layer_fractions: Relative depths at which to capture hidden
            states, e.g. ``(0.25, 0.50, 0.75)``.
        pool_type: Pooling strategy for hidden-state features (default: ``"last"``).
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
        super().__init__(model_name, feature_layer_fractions, pool_type, model)

        self.ema_model: nn.Module = copy.deepcopy(self.model)
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

        self.ema_decay = ema_decay
        self._ema_layers = _get_transformer_layers(self.ema_model)

    # ------------------------------------------------------------------
    # Feature extraction (EMA model, no grad)
    # ------------------------------------------------------------------

    @torch.no_grad()
    @torch.compiler.disable
    def extract_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        completion_start: Optional[int] = None,
    ) -> torch.Tensor:
        """Extract hidden-state features from the EMA model (no grad).

        Runs a forward pass through ``ema_model`` with
        ``output_hidden_states=True`` and returns the concatenation of the
        per-layer pooled, L2-normalised hidden states.

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
        )

    # ------------------------------------------------------------------
    # Log-probability computation (trainable model, with grad)
    # ------------------------------------------------------------------

    @torch.compiler.disable
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
            ``(B,)`` sum of per-token log-probabilities (with gradients).
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return _sum_completion_log_probs(
            outputs.logits, input_ids, attention_mask, completion_start
        )

    @torch.compiler.disable
    def forward_ce_and_ref_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        completion_start: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return CE loss and detached reference features.

        CE is computed on the trainable generator; reference features come
        from the EMA model.

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

    @torch.compiler.disable
    def compute_rollout_data(
        self,
        rollout_ids: torch.Tensor,
        rollout_masks: torch.Tensor,
        completion_start: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return features and log-probs for all rollouts.

        Two forward passes: EMA model for features (no grad), generator for
        log-probs (with grad).

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
    # EMA update (stop-gradient)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_ema(self) -> None:
        """Update EMA model parameters: ``ema <- decay*ema + (1-decay)*theta``.

        Uses ``_foreach`` operations to minimise Python overhead.
        """
        ema_params = list(self.ema_model.parameters())
        main_params = list(self.model.parameters())

        if hasattr(torch, "_foreach_lerp_"):
            torch._foreach_lerp_(ema_params, main_params, 1.0 - self.ema_decay)
        else:
            for ema_p, p in zip(ema_params, main_params):
                ema_p.lerp_(p, 1.0 - self.ema_decay)

        ema_bufs = list(self.ema_model.buffers())
        main_bufs = list(self.model.buffers())

        if hasattr(torch, "_foreach_copy_"):
            torch._foreach_copy_(ema_bufs, main_bufs)
        else:
            for ema_buf, buf in zip(ema_bufs, main_bufs):
                ema_buf.copy_(buf)


# ---------------------------------------------------------------------------
# OnlineEBPModel
# ---------------------------------------------------------------------------


class OnlineEBPModel(BaseEBPModel):
    """Energy-Based Pre-training using the live model as the feature network.

    Unlike :class:`EMAEBPModel`, no EMA copy is maintained.  The trainable
    generator itself serves as the feature network, so features naturally
    improve as training progresses.  The **diversity term** in the
    feature-matching reward prevents representational collapse.

    The key efficiency advantage over ``EMAEBPModel`` is
    :meth:`extract_features_and_log_probs`: a **single forward pass** through
    the generator simultaneously captures hidden-state features (detached) *and*
    computes differentiable log-probabilities from the output logits.

    Args:
        model_name: HuggingFace model identifier (default: ``"Qwen/Qwen3-0.6B"``).
        feature_layer_fractions: Relative depths at which to capture hidden
            states.
        pool_type: Pooling strategy for hidden-state features (default: ``"last"``).
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
        super().__init__(model_name, feature_layer_fractions, pool_type, model)
        # Kept for compatibility with tests that inspect selected layer range.
        self._model_layers = _get_transformer_layers(self.model)

    # ------------------------------------------------------------------
    # Feature extraction (live model, no grad - for reference features)
    # ------------------------------------------------------------------

    @torch.no_grad()
    @torch.compiler.disable
    def extract_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        completion_start: Optional[int] = None,
    ) -> torch.Tensor:
        """Extract hidden-state features from the live model (no grad).

        Used to compute the **reference** feature vector ``phi(c:y)`` for the
        ground-truth completion.  Gradients are not needed here because the
        reference features serve as fixed targets.

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

    @torch.compiler.disable
    def extract_features_and_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        completion_start: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single forward pass returning both features and log-probabilities.

        Features are detached immediately so no gradient flows through them,
        while logits retain the graph for REINFORCE gradients.

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
        )

        log_probs = _sum_completion_log_probs(
            outputs.logits, input_ids, attention_mask, completion_start
        )

        return features, log_probs

    @torch.compiler.disable
    def forward_ce_and_ref_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        completion_start: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single forward pass returning CE loss and detached reference features.

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

    @torch.compiler.disable
    def compute_rollout_data(
        self,
        rollout_ids: torch.Tensor,
        rollout_masks: torch.Tensor,
        completion_start: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return features and log-probs for all rollouts in a single pass.

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
