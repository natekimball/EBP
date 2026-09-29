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
        # Last-token pooling: index the final *unmasked* position.  The mask is
        # not necessarily right-aligned — ``collate_fn`` left-pads contexts and
        # right-pads completions, so ``attention_mask.sum(dim=1) - 1`` would
        # land in the middle of the sequence.  Scan from the right instead.
        if attention_mask is not None:
            mask = attention_mask
            if completion_start is not None:
                mask = mask[:, completion_start:]
            seq_len = mask.shape[1]
            # Index of the last 1 in each row (0 if the row is entirely masked).
            rev_first_one = torch.argmax(mask.flip(dims=(1,)).long(), dim=1)
            last_indices = (seq_len - 1 - rev_first_one).clamp(min=0)
            if completion_start is not None:
                last_indices = last_indices + completion_start
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

    # log p(label) = logit[label] - logsumexp(logits).
    # Both ops reduce over V without materializing a (B, L, V) intermediate.
    label_logits = shift_logits.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    log_sum_exp = torch.logsumexp(shift_logits, dim=-1)
    token_log_probs = label_logits - log_sum_exp

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


def _position_features_from_hidden_states(
    hidden_states: Tuple[torch.Tensor, ...],
    feature_layer_indices: Sequence[int],
    attention_mask: Optional[torch.Tensor],
    max_positions: int,
) -> torch.Tensor:
    """Per-position L2-normalised features, **not** detached, for SIGReg.

    The reward pools each sequence down to a single vector, which gives only
    ``B`` samples per batch — far too few to estimate a distributional statistic
    and rank-deficient in the same way that makes the rollout covariance
    unusable.  Regularising the per-token representations instead yields
    ``B * max_positions`` samples from the same layers at no extra forward cost,
    and shapes the representation the pooling reads from.

    Positions are subsampled uniformly and masked positions dropped.

    Args:
        hidden_states: Tuple of ``(B, L, D)`` hidden states from the model.
        feature_layer_indices: Transformer block indices to read.
        attention_mask: ``(B, L)`` binary mask; padded positions are excluded.
        max_positions: Positions to sample per sequence.

    Returns:
        ``(N, D * K)`` features carrying gradient, where ``N <= B * max_positions``.
    """
    ref = hidden_states[feature_layer_indices[0] + 1]
    b, seq_len, _ = ref.shape
    num_pos = min(max_positions, seq_len)

    idx = torch.randperm(seq_len, device=ref.device)[:num_pos]  # (P,)

    blocks = []
    for layer_idx in feature_layer_indices:
        h = hidden_states[layer_idx + 1][:, idx, :]  # (B, P, D)
        blocks.append(F.normalize(h, p=2, dim=-1))
    feats = torch.cat(blocks, dim=-1).reshape(b * num_pos, -1)

    if attention_mask is not None:
        valid = attention_mask[:, idx].reshape(-1).bool()
        if not bool(valid.all()):
            feats = feats[valid]
    return feats


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
        model_name: str = "Qwen/Qwen3-0.6B-Base",
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
            pad_token_id=(self._eos_token_ids() or [None])[0],
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

        # Mask out everything *after* a generated EOS.  ``generate`` right-pads
        # finished sequences with ``pad_token_id``; without this those filler
        # tokens would contribute to the pooled features and would receive
        # REINFORCE credit in the summed log-probability.
        generated = output_ids[:, context_len:]
        new_mask = torch.ones_like(generated, dtype=torch.long)
        eos_ids = self._eos_token_ids()
        if eos_ids:
            is_eos = torch.zeros_like(generated, dtype=torch.bool)
            for eos_id in eos_ids:
                is_eos |= generated == eos_id
            # cumsum - self keeps the first EOS unmasked and drops the rest.
            new_mask = ((is_eos.long().cumsum(dim=1) - is_eos.long()) == 0).long()

        rollout_masks = torch.cat([expanded_mask, new_mask], dim=1)
        return output_ids, rollout_masks

    def _eos_token_ids(self) -> List[int]:
        """Return the model's EOS token id(s) as a flat list (may be empty)."""
        eos = getattr(self.model.config, "eos_token_id", None)
        if eos is None:
            return []
        if isinstance(eos, int):
            return [eos]
        return [int(e) for e in eos]


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
        model_name: str = "Qwen/Qwen3-0.6B-Base",
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
        sigreg_positions: int = 0,
    ) -> Tuple[torch.Tensor, ...]:
        """Return CE loss and detached reference features.

        When ``sigreg_positions > 0`` a third value is returned: per-position
        features from the *trainable* generator that still carry gradient, for
        the SIGReg penalty.  Reference features still come from the EMA model.

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
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
            use_cache=False,
            output_hidden_states=sigreg_positions > 0,
            return_dict=True,
        )
        ref_features = self.extract_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            completion_start=completion_start,
        )
        if sigreg_positions > 0:
            policy_features = _position_features_from_hidden_states(
                hidden_states=outputs.hidden_states,
                feature_layer_indices=self.feature_layer_indices,
                attention_mask=attention_mask,
                max_positions=sigreg_positions,
            )
            return outputs.loss, ref_features, policy_features
        return outputs.loss, ref_features

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
        model_name: str = "Qwen/Qwen3-0.6B-Base",
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
        sigreg_positions: int = 0,
    ) -> Tuple[torch.Tensor, ...]:
        """Single forward pass returning CE loss and detached reference features.

        When ``sigreg_positions > 0`` a third value is returned: per-position
        features that still carry gradient, for the SIGReg penalty.

        Args:
            input_ids: ``(B, L)`` full sequence (context + completion).
            attention_mask: ``(B, L)`` binary mask.
            completion_start: Index of the first completion token.

        Returns:
            ce_loss: Scalar CE loss tensor with gradients.
            ref_features: ``(B, D * K)`` detached reference features.
        """
        raw_model = getattr(self.model, "_orig_mod", self.model)
        outputs = raw_model(
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
        if sigreg_positions > 0:
            policy_features = _position_features_from_hidden_states(
                hidden_states=outputs.hidden_states,
                feature_layer_indices=self.feature_layer_indices,
                attention_mask=attention_mask,
                max_positions=sigreg_positions,
            )
            return outputs.loss, ref_features, policy_features
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
        chunk_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return features and log-probs for all rollouts.

        When ``chunk_size`` is set, processes sequences in chunks under
        ``torch.no_grad()`` using the uncompiled model to avoid OOM on large
        rollout batches (e.g. during validation).

        Args:
            rollout_ids: ``(B * n, context_len + gen_len)`` rollout token ids.
            rollout_masks: ``(B * n, context_len + gen_len)`` attention masks.
            completion_start: Index of the first generated token.
            chunk_size: Sequences per forward pass. ``None`` = all at once.

        Returns:
            features: ``(B * n, feat_dim)`` - detached feature vectors.
            log_probs: ``(B * n,)`` - summed log-probabilities.
        """
        if chunk_size is None:
            return self.extract_features_and_log_probs(
                rollout_ids, rollout_masks, completion_start
            )
        raw_model = getattr(self.model, "_orig_mod", self.model)
        N = rollout_ids.shape[0]
        all_features: list[torch.Tensor] = []
        all_log_probs: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, N, chunk_size):
                chunk_ids = rollout_ids[start : start + chunk_size]
                chunk_masks = rollout_masks[start : start + chunk_size]
                outputs = raw_model(
                    input_ids=chunk_ids,
                    attention_mask=chunk_masks,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                features = _features_from_hidden_states(
                    hidden_states=outputs.hidden_states,
                    feature_layer_indices=self.feature_layer_indices,
                    attention_mask=chunk_masks,
                    completion_start=completion_start,
                    detach=True,
                    pool_type=self.pool_type,
                )
                log_probs = _sum_completion_log_probs(
                    outputs.logits, chunk_ids, chunk_masks, completion_start
                )
                all_features.append(features)
                all_log_probs.append(log_probs)
        return torch.cat(all_features, dim=0), torch.cat(all_log_probs, dim=0)

    @torch.compiler.disable
    def compute_rollout_features(
        self,
        rollout_ids: torch.Tensor,
        rollout_masks: torch.Tensor,
        completion_start: int,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute detached rollout features in chunks to avoid OOM.

        Runs under ``torch.no_grad()``; logits are freed after each chunk.

        Args:
            rollout_ids: ``(N, L)`` rollout token ids.
            rollout_masks: ``(N, L)`` attention masks.
            completion_start: Index of the first completion token.
            chunk_size: Sequences per forward pass. ``None`` = all at once.

        Returns:
            ``(N, feat_dim)`` detached feature vectors.
        """
        # Use the uncompiled model to avoid triggering Inductor kernel generation
        # for the chunked (batch_size=chunk, full_seq_len) shapes, which are different
        # from the compiled paths (generation: batch*K×1, CE: batch×seq).
        raw_model = getattr(self.model, "_orig_mod", self.model)
        N = rollout_ids.shape[0]
        step = chunk_size if chunk_size is not None else N
        all_features: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, N, step):
                chunk_ids = rollout_ids[start : start + step]
                chunk_masks = rollout_masks[start : start + step]
                outputs = raw_model(
                    input_ids=chunk_ids,
                    attention_mask=chunk_masks,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                feat = _features_from_hidden_states(
                    hidden_states=outputs.hidden_states,
                    feature_layer_indices=self.feature_layer_indices,
                    attention_mask=chunk_masks,
                    completion_start=completion_start,
                    detach=True,
                    pool_type=self.pool_type,
                )
                all_features.append(feat)
                # outputs (logits + hidden_states) freed here
        return torch.cat(all_features, dim=0)

    def compute_log_probs_chunk(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        completion_start: int,
    ) -> torch.Tensor:
        """Forward pass returning only differentiable log-probs (no hidden states).

        Uses the compiled model directly — no hidden states needed, so the
        output_capturing hooks are not required and fullgraph=True is compatible.
        Dynamo compiles a cached trace for the chunk shape on first call.

        Args:
            input_ids: ``(B, L)`` token ids.
            attention_mask: ``(B, L)`` mask.
            completion_start: Index of the first completion token.

        Returns:
            ``(B,)`` summed log-probs with gradient.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        return _sum_completion_log_probs(
            outputs.logits, input_ids, attention_mask, completion_start
        )
