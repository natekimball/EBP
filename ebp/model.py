"""
EBP Model: wraps a causal LM with an EMA feature network.

The EMA model provides intermediate-layer features used to compute the
feature-matching reward.  Because EMA weights are updated with stop-gradient,
the feature space improves along with the generator without introducing
gradient coupling between the two copies.
"""

from __future__ import annotations

import copy
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


class EBPModel(nn.Module):
    """
    Energy-Based Pre-training model wrapper.

    Holds two copies of a causal LM:
      - ``model``: the trainable generator (``pθ``).
      - ``ema_model``: an Exponential Moving Average of ``model``, used as the
        frozen feature network ``ϕ``.  Updated via :meth:`update_ema` after
        every gradient step; no gradient flows through it.

    Feature extraction follows the EBFT paper: hidden states at layers placed
    at depths ``feature_layer_fractions`` of the network are mean-pooled over
    the completion token positions, then L2-normalised per layer, and finally
    concatenated into a single feature vector.

    Args:
        model_name: HuggingFace model identifier (default: ``"Qwen/Qwen3-0.6B"``).
        ema_decay: EMA decay factor ``τ`` (``ema ← τ·ema + (1-τ)·θ``).
        feature_layer_fractions: Relative depths at which to capture hidden
            states, e.g. ``(0.25, 0.50, 0.75)``.
        model: optional pre-instantiated model (useful for tests / fine-
            grained control); if supplied, *model_name* is ignored.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        ema_decay: float = 0.999,
        feature_layer_fractions: Sequence[float] = (0.25, 0.50, 0.75),
        model: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        if model is not None:
            self.model: nn.Module = model
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # EMA copy — no gradients ever flow through this
        self.ema_model: nn.Module = copy.deepcopy(self.model)
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

        self.ema_decay = ema_decay
        self.feature_layer_fractions = tuple(feature_layer_fractions)

        # Resolve layer lists once so we don't traverse attributes each call
        transformer_layers = self._get_transformer_layers(self.model)
        num_layers = len(transformer_layers)
        self.feature_layer_indices: List[int] = [
            max(0, min(round(f * num_layers) - 1, num_layers - 1))
            for f in self.feature_layer_fractions
        ]
        # Expose EMA layers for hooks
        self._ema_layers = self._get_transformer_layers(self.ema_model)

    # ------------------------------------------------------------------
    # Architecture helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_transformer_layers(model: nn.Module) -> nn.ModuleList:
        """Return the ``nn.ModuleList`` of transformer decoder blocks.

        Supports the most common HuggingFace causal-LM architectures:
        * Qwen / LLaMA / Mistral / Gemma → ``model.model.layers``
        * GPT-2 → ``model.transformer.h``
        * GPT-NeoX / Pythia → ``model.gpt_neox.layers``
        """
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h
        if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
            return model.gpt_neox.layers
        raise ValueError(
            f"Cannot locate transformer layer list for {type(model).__name__}. "
            "Override _get_transformer_layers or pass a supported architecture."
        )

    # ------------------------------------------------------------------
    # Forward (trainable generator)
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """Standard HuggingFace forward through the trainable generator."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
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
        """Extract features from the EMA model.

        Registers forward hooks on the target layers, runs a single forward
        pass through ``ema_model``, and returns the concatenation of the
        per-layer mean-pooled, L2-normalised hidden states.

        Args:
            input_ids: ``(B, L)`` token ids.
            attention_mask: ``(B, L)`` binary mask (1 = real token).
            completion_start: If given, pool only over positions
                ``[completion_start, L)``, which correspond to completion
                (generated) tokens.  Context positions are excluded from
                pooling so the feature represents *completions* given context.

        Returns:
            ``(B, D * num_feature_layers)`` feature tensor.
        """
        captured: dict[int, torch.Tensor] = {}
        hooks = []

        def make_hook(key: int):
            def hook_fn(module, inp, output):
                h = output[0] if isinstance(output, tuple) else output
                captured[key] = h.detach()

            return hook_fn

        for idx in self.feature_layer_indices:
            hooks.append(self._ema_layers[idx].register_forward_hook(make_hook(idx)))

        try:
            self.ema_model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            for hook in hooks:
                hook.remove()

        layer_features: List[torch.Tensor] = []
        for idx in self.feature_layer_indices:
            h = captured[idx]  # (B, L, D)

            if completion_start is not None:
                # Pool over completion positions only
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
                # Fallback: use last token position
                pooled = h[:, -1, :]

            # Normalise each layer's representation to unit L2 norm
            pooled = F.normalize(pooled, p=2, dim=-1)
            layer_features.append(pooled)

        return torch.cat(layer_features, dim=-1)  # (B, D * K)

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
            completion_start: Index of the first completion token.  If
                ``None``, sums log-probs over the entire sequence.

        Returns:
            ``(B,)`` sum of per-token log-probabilities for the completion.
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (B, L, V)

        # Shift by one position: logits[t] predicts token at position t+1
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
                    token_log_probs[:, :min_len]
                    * comp_mask[:, :min_len].float()
                )

        return token_log_probs.sum(dim=-1)  # (B,)

    # ------------------------------------------------------------------
    # Rollout generation (no grad)
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

        Each context in the batch is independently expanded to produce
        ``num_rollouts`` rollouts, so the returned tensors have batch size
        ``B * num_rollouts``.

        Args:
            context_ids: ``(B, context_len)`` context token ids.
            context_attention_mask: ``(B, context_len)`` binary mask.
            num_rollouts: Number of completions to sample per context.
            generation_length: Number of new tokens to generate.
            temperature: Sampling temperature.
            **generate_kwargs: Additional keyword arguments forwarded to
                ``model.generate``.

        Returns:
            rollout_ids: ``(B * num_rollouts, context_len + generation_length)``
            rollout_masks: ``(B * num_rollouts, context_len + generation_length)``
        """
        context_len = context_ids.shape[1]

        expanded_ids = context_ids.repeat_interleave(num_rollouts, dim=0)
        expanded_mask = context_attention_mask.repeat_interleave(num_rollouts, dim=0)

        output_ids = self.model.generate(
            input_ids=expanded_ids,
            attention_mask=expanded_mask,
            max_new_tokens=generation_length,
            do_sample=True,
            temperature=temperature,
            pad_token_id=self.model.config.eos_token_id,
            **generate_kwargs,
        )

        # Construct attention mask for the newly generated tokens
        generated_len = output_ids.shape[1] - context_len
        new_mask = torch.ones(
            output_ids.shape[0],
            generated_len,
            dtype=torch.long,
            device=output_ids.device,
        )
        rollout_masks = torch.cat([expanded_mask, new_mask], dim=1)

        return output_ids, rollout_masks

    # ------------------------------------------------------------------
    # EMA update (stop-gradient)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_ema(self) -> None:
        """Update EMA model parameters after a gradient step.

        Implements the stop-gradient update:
            ``ema_param ← decay * ema_param + (1 - decay) * param``

        Non-parameter buffers (e.g. layer-norm running stats) are copied
        directly so they stay in sync with the generator.
        """
        for ema_p, p in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_p.data.mul_(self.ema_decay).add_(p.data, alpha=1.0 - self.ema_decay)

        for ema_buf, buf in zip(self.ema_model.buffers(), self.model.buffers()):
            ema_buf.data.copy_(buf.data)
