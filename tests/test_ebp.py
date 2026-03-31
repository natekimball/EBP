"""
Unit tests for the EBP package.

All tests use a tiny Qwen2-style model built from a small configuration so
that they run quickly without downloading any pretrained weights.
"""

from __future__ import annotations

import copy
import unittest

import torch
import torch.nn as nn
from transformers import Qwen2Config, Qwen2ForCausalLM

from ebp.model import EBPModel
from ebp.rewards import compute_feature_matching_rewards, compute_rloo_baseline
from ebp.data import collate_fn


# ---------------------------------------------------------------------------
# Tiny model factory – no downloads required
# ---------------------------------------------------------------------------

TINY_CONFIG = Qwen2Config(
    vocab_size=256,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=4,
    max_position_embeddings=128,
    tie_word_embeddings=False,
)


def make_tiny_model() -> Qwen2ForCausalLM:
    """Return a randomly-initialised, untrained Qwen2 model with tiny dims."""
    model = Qwen2ForCausalLM(TINY_CONFIG)
    model.eval()
    return model


def make_ebp_model() -> EBPModel:
    """Return an :class:`EBPModel` wrapping the tiny Qwen2 model."""
    return EBPModel(model=make_tiny_model())


# ---------------------------------------------------------------------------
# Tests: EBPModel
# ---------------------------------------------------------------------------


class TestEBPModelInit(unittest.TestCase):
    def test_ema_is_deepcopy(self):
        """EMA and generator share no parameters."""
        ebp = make_ebp_model()
        for p, ema_p in zip(ebp.model.parameters(), ebp.ema_model.parameters()):
            self.assertIsNot(p, ema_p)

    def test_ema_no_grad(self):
        """EMA parameters must not require gradients."""
        ebp = make_ebp_model()
        for p in ebp.ema_model.parameters():
            self.assertFalse(p.requires_grad)

    def test_feature_layer_indices_in_range(self):
        """Feature-layer indices must be valid for the chosen model."""
        ebp = make_ebp_model()
        num_layers = len(ebp._ema_layers)
        for idx in ebp.feature_layer_indices:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, num_layers)

    def test_feature_layer_indices_count(self):
        """Number of feature-layer indices equals number of fractions."""
        ebp = make_ebp_model()
        self.assertEqual(
            len(ebp.feature_layer_indices), len(ebp.feature_layer_fractions)
        )


class TestExtractFeatures(unittest.TestCase):
    def setUp(self):
        self.ebp = make_ebp_model()
        self.batch_size = 2
        self.seq_len = 12
        self.ids = torch.randint(0, TINY_CONFIG.vocab_size, (self.batch_size, self.seq_len))
        self.mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.long)

    def test_output_shape(self):
        """Feature tensor should be (B, hidden_size * num_feature_layers)."""
        features = self.ebp.extract_features(self.ids, self.mask)
        expected_dim = TINY_CONFIG.hidden_size * len(self.ebp.feature_layer_fractions)
        self.assertEqual(features.shape, (self.batch_size, expected_dim))

    def test_unit_norm_per_layer_block(self):
        """Each per-layer block of the feature vector should be unit L2 norm."""
        features = self.ebp.extract_features(self.ids, self.mask)
        d = TINY_CONFIG.hidden_size
        num_layers = len(self.ebp.feature_layer_fractions)
        for k in range(num_layers):
            block = features[:, k * d : (k + 1) * d]  # (B, d)
            norms = block.norm(dim=-1)
            for norm in norms:
                self.assertAlmostEqual(norm.item(), 1.0, places=5)

    def test_no_grad_through_ema(self):
        """Extracted features must not carry gradients (stop-gradient)."""
        features = self.ebp.extract_features(self.ids, self.mask)
        self.assertFalse(features.requires_grad)

    def test_completion_start_changes_features(self):
        """Using completion_start should change the pooled feature."""
        f_full = self.ebp.extract_features(self.ids, self.mask)
        f_comp = self.ebp.extract_features(self.ids, self.mask, completion_start=6)
        # They should differ because different positions are pooled
        self.assertFalse(torch.allclose(f_full, f_comp))


class TestComputeLogProbs(unittest.TestCase):
    def setUp(self):
        self.ebp = make_ebp_model()
        self.batch_size = 2
        self.context_len = 6
        self.gen_len = 4
        self.seq_len = self.context_len + self.gen_len
        self.ids = torch.randint(0, TINY_CONFIG.vocab_size, (self.batch_size, self.seq_len))
        self.mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.long)

    def test_output_shape(self):
        log_probs = self.ebp.compute_log_probs(self.ids, self.mask)
        self.assertEqual(log_probs.shape, (self.batch_size,))

    def test_output_shape_with_completion_start(self):
        log_probs = self.ebp.compute_log_probs(
            self.ids, self.mask, completion_start=self.context_len
        )
        self.assertEqual(log_probs.shape, (self.batch_size,))

    def test_log_probs_are_negative(self):
        """Log-probabilities of random tokens should be ≤ 0."""
        log_probs = self.ebp.compute_log_probs(self.ids, self.mask)
        self.assertTrue((log_probs <= 0).all())

    def test_completion_log_probs_differ_from_full(self):
        """Restricting to completion tokens must change the sum."""
        lp_full = self.ebp.compute_log_probs(self.ids, self.mask)
        lp_comp = self.ebp.compute_log_probs(
            self.ids, self.mask, completion_start=self.context_len
        )
        # Full sequence covers more tokens → larger absolute value
        self.assertTrue((lp_full.abs() >= lp_comp.abs()).all())

    def test_gradients_flow(self):
        """Gradients should flow from log_probs back through the generator."""
        self.ebp.model.train()
        log_probs = self.ebp.compute_log_probs(self.ids, self.mask)
        loss = log_probs.sum()
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in self.ebp.model.parameters()
        )
        self.assertTrue(has_grad)


class TestGenerateRollouts(unittest.TestCase):
    def setUp(self):
        self.ebp = make_ebp_model()
        self.batch_size = 2
        self.context_len = 8
        self.num_rollouts = 3
        self.gen_len = 4
        self.context_ids = torch.randint(
            0, TINY_CONFIG.vocab_size, (self.batch_size, self.context_len)
        )
        self.context_mask = torch.ones(
            self.batch_size, self.context_len, dtype=torch.long
        )

    def test_output_shapes(self):
        ids, masks = self.ebp.generate_rollouts(
            self.context_ids, self.context_mask, self.num_rollouts, self.gen_len
        )
        expected_batch = self.batch_size * self.num_rollouts
        expected_len = self.context_len + self.gen_len
        self.assertEqual(ids.shape, (expected_batch, expected_len))
        self.assertEqual(masks.shape, (expected_batch, expected_len))

    def test_context_preserved(self):
        """The context prefix should be identical in every rollout."""
        ids, _ = self.ebp.generate_rollouts(
            self.context_ids, self.context_mask, self.num_rollouts, self.gen_len
        )
        for i in range(self.batch_size):
            for j in range(self.num_rollouts):
                rollout_ctx = ids[i * self.num_rollouts + j, : self.context_len]
                torch.testing.assert_close(rollout_ctx, self.context_ids[i])

    def test_mask_values(self):
        """All attention-mask values should be 0 or 1."""
        _, masks = self.ebp.generate_rollouts(
            self.context_ids, self.context_mask, self.num_rollouts, self.gen_len
        )
        self.assertTrue(((masks == 0) | (masks == 1)).all())


class TestUpdateEMA(unittest.TestCase):
    def test_ema_weights_change_after_update(self):
        """EMA weights should change after calling update_ema."""
        ebp = make_ebp_model()
        # Record initial EMA state
        initial_ema = {
            name: param.clone() for name, param in ebp.ema_model.named_parameters()
        }
        # Perturb generator weights
        with torch.no_grad():
            for p in ebp.model.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        ebp.update_ema()

        for name, new_param in ebp.ema_model.named_parameters():
            old_param = initial_ema[name]
            self.assertFalse(
                torch.equal(old_param, new_param),
                f"EMA parameter '{name}' did not change after update.",
            )

    def test_ema_decay_formula(self):
        """Verify the EMA update formula: ema_new = decay*ema_old + (1-decay)*param."""
        ebp = make_ebp_model()
        decay = ebp.ema_decay

        initial_ema = {
            name: param.clone() for name, param in ebp.ema_model.named_parameters()
        }
        generator_params = {
            name: param.clone() for name, param in ebp.model.named_parameters()
        }

        ebp.update_ema()

        for (name, new_ema), (_, gen_p) in zip(
            ebp.ema_model.named_parameters(), ebp.model.named_parameters()
        ):
            expected = decay * initial_ema[name] + (1 - decay) * generator_params[name]
            torch.testing.assert_close(
                new_ema.data, expected, msg=f"EMA formula mismatch for '{name}'"
            )

    def test_generator_gradients_not_affected(self):
        """update_ema must not introduce gradients on the generator."""
        ebp = make_ebp_model()
        ebp.update_ema()
        for p in ebp.model.parameters():
            self.assertIsNone(p.grad)


# ---------------------------------------------------------------------------
# Tests: rewards
# ---------------------------------------------------------------------------


class TestComputeFeatureMatchingRewards(unittest.TestCase):
    def _make_inputs(self, n: int = 4, d: int = 16):
        rollout_feat = torch.randn(n, d)
        ref_feat = torch.randn(d)
        return rollout_feat, ref_feat

    def test_output_shape(self):
        rollout_feat, ref_feat = self._make_inputs()
        rewards = compute_feature_matching_rewards(rollout_feat, ref_feat)
        self.assertEqual(rewards.shape, (4,))

    def test_single_rollout(self):
        """With n=1 the diversity term is zero."""
        rollout_feat = torch.tensor([[1.0, 0.0]])
        ref_feat = torch.tensor([1.0, 0.0])
        rewards = compute_feature_matching_rewards(rollout_feat, ref_feat)
        # alignment = 2 * (1*1 + 0*0) = 2, diversity = 0
        self.assertAlmostEqual(rewards[0].item(), 2.0, places=5)

    def test_alignment_term_sign(self):
        """Identical rollout and reference should give maximum alignment."""
        d = 16
        feat = torch.randn(d)
        feat_norm = feat / feat.norm()
        rollout_feat = feat_norm.unsqueeze(0)  # (1, d)
        rewards_same = compute_feature_matching_rewards(rollout_feat, feat_norm)
        # alignment = 2 * (feat_norm . feat_norm) = 2 * 1 = 2
        self.assertAlmostEqual(rewards_same[0].item(), 2.0, places=5)

    def test_orthogonal_gives_zero_alignment(self):
        """Orthogonal rollout and reference → alignment term = 0."""
        rollout_feat = torch.tensor([[1.0, 0.0]])
        ref_feat = torch.tensor([0.0, 1.0])
        rewards = compute_feature_matching_rewards(rollout_feat, ref_feat)
        # alignment = 2 * 0 = 0, diversity = 0
        self.assertAlmostEqual(rewards[0].item(), 0.0, places=5)

    def test_diversity_term_reduces_reward(self):
        """Adding a nearly identical second rollout should reduce both rewards."""
        feat = torch.randn(16)
        feat = feat / feat.norm()
        ref = torch.randn(16)
        ref = ref / ref.norm()

        # Single rollout
        r_single = compute_feature_matching_rewards(feat.unsqueeze(0), ref)

        # Two near-identical rollouts
        rollout_feat = feat.unsqueeze(0).repeat(2, 1)
        r_pair = compute_feature_matching_rewards(rollout_feat, ref)

        # Diversity term should lower the reward compared to singleton alignment
        for r in r_pair:
            self.assertLess(r.item(), r_single[0].item() + 1e-6)

    def test_invalid_rollout_dim_raises(self):
        with self.assertRaises(ValueError):
            compute_feature_matching_rewards(torch.randn(4), torch.randn(4))

    def test_invalid_ref_dim_raises(self):
        with self.assertRaises(ValueError):
            compute_feature_matching_rewards(torch.randn(4, 8), torch.randn(2, 8))


class TestComputeRLOOBaseline(unittest.TestCase):
    def test_single_reward_gives_zero(self):
        rewards = torch.tensor([3.0])
        baselines = compute_rloo_baseline(rewards)
        self.assertAlmostEqual(baselines[0].item(), 0.0, places=6)

    def test_two_rewards(self):
        rewards = torch.tensor([2.0, 4.0])
        baselines = compute_rloo_baseline(rewards)
        # b_0 = (4.0 - 2.0) / 1 = 2? No: (total - r_0)/(n-1) = (6-2)/1 = 4
        # b_1 = (6 - 4) / 1 = 2
        self.assertAlmostEqual(baselines[0].item(), 4.0, places=6)
        self.assertAlmostEqual(baselines[1].item(), 2.0, places=6)

    def test_equal_rewards_give_same_baseline(self):
        rewards = torch.ones(5) * 3.0
        baselines = compute_rloo_baseline(rewards)
        for b in baselines:
            self.assertAlmostEqual(b.item(), 3.0, places=6)

    def test_output_shape(self):
        rewards = torch.randn(8)
        baselines = compute_rloo_baseline(rewards)
        self.assertEqual(baselines.shape, rewards.shape)

    def test_advantages_zero_mean(self):
        """RLOO advantages (r - b) should sum to zero."""
        rewards = torch.randn(6)
        baselines = compute_rloo_baseline(rewards)
        advantages = rewards - baselines
        self.assertAlmostEqual(advantages.sum().item(), 0.0, places=5)


# ---------------------------------------------------------------------------
# Tests: data utilities
# ---------------------------------------------------------------------------


class TestCollateFn(unittest.TestCase):
    def _make_batch(self):
        return [
            {"context_ids": [1, 2, 3], "completion_ids": [4, 5]},
            {"context_ids": [6, 7], "completion_ids": [8, 9, 10]},
        ]

    def test_output_keys(self):
        out = collate_fn(self._make_batch(), pad_token_id=0)
        self.assertIn("context_ids", out)
        self.assertIn("context_mask", out)
        self.assertIn("completion_ids", out)
        self.assertIn("completion_mask", out)

    def test_context_padded_to_max_length(self):
        out = collate_fn(self._make_batch(), pad_token_id=0)
        max_ctx = 3
        self.assertEqual(out["context_ids"].shape[1], max_ctx)
        self.assertEqual(out["context_mask"].shape[1], max_ctx)

    def test_completion_padded_to_max_length(self):
        out = collate_fn(self._make_batch(), pad_token_id=0)
        max_comp = 3
        self.assertEqual(out["completion_ids"].shape[1], max_comp)
        self.assertEqual(out["completion_mask"].shape[1], max_comp)

    def test_mask_correctly_marks_padding(self):
        out = collate_fn(self._make_batch(), pad_token_id=0)
        # Second batch item's context has length 2, padded to 3 → first position is 0
        self.assertEqual(out["context_mask"][1, 0].item(), 0)
        self.assertEqual(out["context_mask"][1, 1].item(), 1)
        self.assertEqual(out["context_mask"][1, 2].item(), 1)

    def test_completion_mask_marks_padding(self):
        out = collate_fn(self._make_batch(), pad_token_id=0)
        # First item completion length 2, padded to 3 → last position is 0
        self.assertEqual(out["completion_mask"][0, 0].item(), 1)
        self.assertEqual(out["completion_mask"][0, 1].item(), 1)
        self.assertEqual(out["completion_mask"][0, 2].item(), 0)

    def test_left_padding_for_context(self):
        """Shorter context should be left-padded."""
        out = collate_fn(self._make_batch(), pad_token_id=0)
        # Second item context [6, 7] padded to length 3 → [0, 6, 7]
        self.assertEqual(out["context_ids"][1, 0].item(), 0)
        self.assertEqual(out["context_ids"][1, 1].item(), 6)
        self.assertEqual(out["context_ids"][1, 2].item(), 7)

    def test_batch_size(self):
        out = collate_fn(self._make_batch(), pad_token_id=0)
        self.assertEqual(out["context_ids"].shape[0], 2)


# ---------------------------------------------------------------------------
# Integration smoke test
# ---------------------------------------------------------------------------


class TestEndToEndStep(unittest.TestCase):
    """Smoke test: one training step should not crash and produce a scalar loss."""

    def test_forward_backward_smoke(self):
        ebp = make_ebp_model()
        ebp.model.train()

        batch_size = 2
        context_len = 8
        gen_len = 4

        context_ids = torch.randint(0, TINY_CONFIG.vocab_size, (batch_size, context_len))
        context_mask = torch.ones(batch_size, context_len, dtype=torch.long)
        completion_ids = torch.randint(0, TINY_CONFIG.vocab_size, (batch_size, gen_len))
        completion_mask = torch.ones(batch_size, gen_len, dtype=torch.long)

        # Reference features
        full_ids = torch.cat([context_ids, completion_ids], dim=1)
        full_mask = torch.cat([context_mask, completion_mask], dim=1)
        ref_features = ebp.extract_features(full_ids, full_mask, completion_start=context_len)

        # Rollouts
        rollout_ids, rollout_masks = ebp.generate_rollouts(
            context_ids, context_mask, num_rollouts=3, generation_length=gen_len
        )

        rollout_features = ebp.extract_features(
            rollout_ids, rollout_masks, completion_start=context_len
        )

        # REINFORCE loss
        total_loss = torch.tensor(0.0)
        for i in range(batch_size):
            item_rf = rollout_features[i * 3 : (i + 1) * 3]
            item_ref = ref_features[i]
            rewards = compute_feature_matching_rewards(item_rf, item_ref)
            baselines = compute_rloo_baseline(rewards)
            advantages = (rewards - baselines).detach()
            lp = ebp.compute_log_probs(
                rollout_ids[i * 3 : (i + 1) * 3],
                rollout_masks[i * 3 : (i + 1) * 3],
                completion_start=context_len,
            )
            total_loss = total_loss - (advantages * lp).mean()

        total_loss = total_loss / batch_size
        total_loss.backward()

        # Check that at least one generator parameter received a gradient
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in ebp.model.parameters()
        )
        self.assertTrue(has_grad, "No gradients flowed into the generator.")

        # EMA update must not crash
        ebp.update_ema()


if __name__ == "__main__":
    unittest.main()
