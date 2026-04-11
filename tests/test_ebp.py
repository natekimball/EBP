"""
Unit tests for the EBP package.

All tests use a tiny Qwen2-style model built from a small configuration so
that they run quickly without downloading any pretrained weights.
"""

from __future__ import annotations

import unittest

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from ebp.model import EMAEBPModel, OnlineEBPModel
from ebp.rewards import (
    compute_feature_matching_rewards,
    compute_feature_matching_rewards_batched,
    compute_rloo_baseline,
    compute_rloo_baseline_batched,
)
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


def make_ema_model() -> EMAEBPModel:
    return EMAEBPModel(model=make_tiny_model())


def make_online_model() -> OnlineEBPModel:
    return OnlineEBPModel(model=make_tiny_model())


# ---------------------------------------------------------------------------
# Tests: EMAEBPModel
# ---------------------------------------------------------------------------


class TestEMAEBPModelInit(unittest.TestCase):
    def test_ema_is_deepcopy(self):
        ebp = make_ema_model()
        for p, ema_p in zip(ebp.model.parameters(), ebp.ema_model.parameters()):
            self.assertIsNot(p, ema_p)

    def test_ema_no_grad(self):
        ebp = make_ema_model()
        for p in ebp.ema_model.parameters():
            self.assertFalse(p.requires_grad)

    def test_feature_layer_indices_in_range(self):
        ebp = make_ema_model()
        num_layers = len(ebp._ema_layers)
        for idx in ebp.feature_layer_indices:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, num_layers)

    def test_feature_layer_indices_count(self):
        ebp = make_ema_model()
        self.assertEqual(
            len(ebp.feature_layer_indices), len(ebp.feature_layer_fractions)
        )


class TestEMAExtractFeatures(unittest.TestCase):
    def setUp(self):
        self.ebp = make_ema_model()
        self.batch_size = 2
        self.seq_len = 12
        self.ids = torch.randint(0, TINY_CONFIG.vocab_size, (self.batch_size, self.seq_len))
        self.mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.long)

    def test_output_shape(self):
        features = self.ebp.extract_features(self.ids, self.mask)
        expected_dim = TINY_CONFIG.hidden_size * len(self.ebp.feature_layer_fractions)
        self.assertEqual(features.shape, (self.batch_size, expected_dim))

    def test_unit_norm_per_layer_block(self):
        features = self.ebp.extract_features(self.ids, self.mask)
        d = TINY_CONFIG.hidden_size
        for k in range(len(self.ebp.feature_layer_fractions)):
            block = features[:, k * d : (k + 1) * d]
            norms = block.norm(dim=-1)
            for norm in norms:
                self.assertAlmostEqual(norm.item(), 1.0, places=5)

    def test_no_grad_through_ema(self):
        features = self.ebp.extract_features(self.ids, self.mask)
        self.assertFalse(features.requires_grad)

    def test_completion_start_changes_features(self):
        f_full = self.ebp.extract_features(self.ids, self.mask)
        f_comp = self.ebp.extract_features(self.ids, self.mask, completion_start=6)
        self.assertFalse(torch.allclose(f_full, f_comp))


class TestEMAComputeLogProbs(unittest.TestCase):
    def setUp(self):
        self.ebp = make_ema_model()
        self.batch_size = 2
        self.context_len = 6
        self.gen_len = 4
        self.seq_len = self.context_len + self.gen_len
        self.ids = torch.randint(0, TINY_CONFIG.vocab_size, (self.batch_size, self.seq_len))
        self.mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.long)

    def test_output_shape(self):
        lp = self.ebp.compute_log_probs(self.ids, self.mask)
        self.assertEqual(lp.shape, (self.batch_size,))

    def test_output_shape_with_completion_start(self):
        lp = self.ebp.compute_log_probs(self.ids, self.mask, completion_start=self.context_len)
        self.assertEqual(lp.shape, (self.batch_size,))

    def test_log_probs_are_negative(self):
        lp = self.ebp.compute_log_probs(self.ids, self.mask)
        self.assertTrue((lp <= 0).all())

    def test_gradients_flow(self):
        self.ebp.model.train()
        lp = self.ebp.compute_log_probs(self.ids, self.mask)
        lp.sum().backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in self.ebp.model.parameters()
        )
        self.assertTrue(has_grad)


class TestEMAComputeRolloutData(unittest.TestCase):
    def test_shapes_and_grad(self):
        ebp = make_ema_model()
        ebp.model.train()
        ids = torch.randint(0, TINY_CONFIG.vocab_size, (6, 12))
        mask = torch.ones(6, 12, dtype=torch.long)
        features, log_probs = ebp.compute_rollout_data(ids, mask, completion_start=8)
        self.assertEqual(features.shape[0], 6)
        self.assertEqual(log_probs.shape, (6,))
        self.assertFalse(features.requires_grad)
        self.assertTrue(log_probs.requires_grad)


class TestEMAGenerateRollouts(unittest.TestCase):
    def setUp(self):
        self.ebp = make_ema_model()
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
        ids, _ = self.ebp.generate_rollouts(
            self.context_ids, self.context_mask, self.num_rollouts, self.gen_len
        )
        for i in range(self.batch_size):
            for j in range(self.num_rollouts):
                rollout_ctx = ids[i * self.num_rollouts + j, : self.context_len]
                torch.testing.assert_close(rollout_ctx, self.context_ids[i])

    def test_mask_values(self):
        _, masks = self.ebp.generate_rollouts(
            self.context_ids, self.context_mask, self.num_rollouts, self.gen_len
        )
        self.assertTrue(((masks == 0) | (masks == 1)).all())


class TestUpdateEMA(unittest.TestCase):
    def test_ema_weights_change_after_update(self):
        ebp = make_ema_model()
        initial_ema = {n: p.clone() for n, p in ebp.ema_model.named_parameters()}
        with torch.no_grad():
            for p in ebp.model.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        ebp.update_ema()
        for name, new_p in ebp.ema_model.named_parameters():
            self.assertFalse(torch.equal(initial_ema[name], new_p))

    def test_ema_decay_formula(self):
        ebp = make_ema_model()
        decay = ebp.ema_decay
        initial_ema = {n: p.clone() for n, p in ebp.ema_model.named_parameters()}
        gen_params = {n: p.clone() for n, p in ebp.model.named_parameters()}
        ebp.update_ema()
        for (name, new_ema), (_, gen_p) in zip(
            ebp.ema_model.named_parameters(), ebp.model.named_parameters()
        ):
            expected = decay * initial_ema[name] + (1 - decay) * gen_params[name]
            torch.testing.assert_close(new_ema.data, expected)

    def test_generator_gradients_not_affected(self):
        ebp = make_ema_model()
        ebp.update_ema()
        for p in ebp.model.parameters():
            self.assertIsNone(p.grad)


# ---------------------------------------------------------------------------
# Tests: OnlineEBPModel
# ---------------------------------------------------------------------------


class TestOnlineEBPModelInit(unittest.TestCase):
    def test_no_ema_model(self):
        ebp = make_online_model()
        self.assertFalse(hasattr(ebp, "ema_model"))

    def test_feature_layer_indices_in_range(self):
        ebp = make_online_model()
        num_layers = len(ebp._model_layers)
        for idx in ebp.feature_layer_indices:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, num_layers)

    def test_feature_layer_indices_count(self):
        ebp = make_online_model()
        self.assertEqual(
            len(ebp.feature_layer_indices), len(ebp.feature_layer_fractions)
        )


class TestOnlineExtractFeatures(unittest.TestCase):
    def setUp(self):
        self.ebp = make_online_model()
        self.batch_size = 2
        self.seq_len = 12
        self.ids = torch.randint(0, TINY_CONFIG.vocab_size, (self.batch_size, self.seq_len))
        self.mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.long)

    def test_output_shape(self):
        features = self.ebp.extract_features(self.ids, self.mask)
        expected_dim = TINY_CONFIG.hidden_size * len(self.ebp.feature_layer_fractions)
        self.assertEqual(features.shape, (self.batch_size, expected_dim))

    def test_unit_norm_per_layer_block(self):
        features = self.ebp.extract_features(self.ids, self.mask)
        d = TINY_CONFIG.hidden_size
        for k in range(len(self.ebp.feature_layer_fractions)):
            block = features[:, k * d : (k + 1) * d]
            norms = block.norm(dim=-1)
            for norm in norms:
                self.assertAlmostEqual(norm.item(), 1.0, places=5)

    def test_no_grad_through_extract_features(self):
        features = self.ebp.extract_features(self.ids, self.mask)
        self.assertFalse(features.requires_grad)


class TestOnlineExtractFeaturesAndLogProbs(unittest.TestCase):
    def setUp(self):
        self.ebp = make_online_model()
        self.ebp.model.train()
        self.batch_size = 2
        self.context_len = 8
        self.gen_len = 4
        self.seq_len = self.context_len + self.gen_len
        self.ids = torch.randint(0, TINY_CONFIG.vocab_size, (self.batch_size, self.seq_len))
        self.mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.long)

    def test_output_shapes(self):
        features, log_probs = self.ebp.extract_features_and_log_probs(
            self.ids, self.mask, completion_start=self.context_len
        )
        expected_feat_dim = TINY_CONFIG.hidden_size * len(self.ebp.feature_layer_fractions)
        self.assertEqual(features.shape, (self.batch_size, expected_feat_dim))
        self.assertEqual(log_probs.shape, (self.batch_size,))

    def test_features_are_detached(self):
        features, _ = self.ebp.extract_features_and_log_probs(
            self.ids, self.mask, completion_start=self.context_len
        )
        self.assertFalse(features.requires_grad)

    def test_log_probs_have_grad(self):
        _, log_probs = self.ebp.extract_features_and_log_probs(
            self.ids, self.mask, completion_start=self.context_len
        )
        self.assertTrue(log_probs.requires_grad)

    def test_backward_through_log_probs(self):
        _, log_probs = self.ebp.extract_features_and_log_probs(
            self.ids, self.mask, completion_start=self.context_len
        )
        log_probs.sum().backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in self.ebp.model.parameters()
        )
        self.assertTrue(has_grad, "No gradients flowed into the generator.")

    def test_features_same_as_extract_features(self):
        """Features from the combined method should match extract_features."""
        self.ebp.eval()
        with torch.no_grad():
            feats_combined, _ = self.ebp.extract_features_and_log_probs(
                self.ids, self.mask, completion_start=self.context_len
            )
        feats_separate = self.ebp.extract_features(
            self.ids, self.mask, completion_start=self.context_len
        )
        torch.testing.assert_close(feats_combined, feats_separate)


class TestOnlineComputeRolloutData(unittest.TestCase):
    def test_shapes_and_grad(self):
        ebp = make_online_model()
        ebp.model.train()
        ids = torch.randint(0, TINY_CONFIG.vocab_size, (6, 12))
        mask = torch.ones(6, 12, dtype=torch.long)
        features, log_probs = ebp.compute_rollout_data(ids, mask, completion_start=8)
        self.assertEqual(features.shape[0], 6)
        self.assertEqual(log_probs.shape, (6,))
        self.assertFalse(features.requires_grad)
        self.assertTrue(log_probs.requires_grad)


class TestOnlineGenerateRollouts(unittest.TestCase):
    def setUp(self):
        self.ebp = make_online_model()
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
        self.assertEqual(ids.shape, (self.batch_size * self.num_rollouts,
                                     self.context_len + self.gen_len))
        self.assertEqual(masks.shape, ids.shape)

    def test_context_preserved(self):
        ids, _ = self.ebp.generate_rollouts(
            self.context_ids, self.context_mask, self.num_rollouts, self.gen_len
        )
        for i in range(self.batch_size):
            for j in range(self.num_rollouts):
                ctx = ids[i * self.num_rollouts + j, : self.context_len]
                torch.testing.assert_close(ctx, self.context_ids[i])


# ---------------------------------------------------------------------------
# Tests: rewards
# ---------------------------------------------------------------------------


class TestComputeFeatureMatchingRewards(unittest.TestCase):
    def _make_inputs(self, n: int = 4, d: int = 16):
        return torch.randn(n, d), torch.randn(d)

    def test_output_shape(self):
        r = compute_feature_matching_rewards(*self._make_inputs())
        self.assertEqual(r.shape, (4,))

    def test_single_rollout(self):
        rollout_feat = torch.tensor([[1.0, 0.0]])
        ref_feat = torch.tensor([1.0, 0.0])
        r = compute_feature_matching_rewards(rollout_feat, ref_feat)
        self.assertAlmostEqual(r[0].item(), 2.0, places=5)

    def test_alignment_term_sign(self):
        d = 16
        feat = torch.randn(d)
        feat_norm = feat / feat.norm()
        r = compute_feature_matching_rewards(feat_norm.unsqueeze(0), feat_norm)
        self.assertAlmostEqual(r[0].item(), 2.0, places=5)

    def test_orthogonal_gives_zero_alignment(self):
        r = compute_feature_matching_rewards(
            torch.tensor([[1.0, 0.0]]), torch.tensor([0.0, 1.0])
        )
        self.assertAlmostEqual(r[0].item(), 0.0, places=5)

    def test_diversity_term_reduces_reward(self):
        feat = torch.randn(16)
        feat = feat / feat.norm()
        ref = torch.randn(16)
        ref = ref / ref.norm()
        r_single = compute_feature_matching_rewards(feat.unsqueeze(0), ref)
        r_pair = compute_feature_matching_rewards(feat.unsqueeze(0).repeat(2, 1), ref)
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
        b = compute_rloo_baseline(torch.tensor([3.0]))
        self.assertAlmostEqual(b[0].item(), 0.0, places=6)

    def test_two_rewards(self):
        b = compute_rloo_baseline(torch.tensor([2.0, 4.0]))
        self.assertAlmostEqual(b[0].item(), 4.0, places=6)
        self.assertAlmostEqual(b[1].item(), 2.0, places=6)

    def test_equal_rewards_give_same_baseline(self):
        b = compute_rloo_baseline(torch.ones(5) * 3.0)
        for bi in b:
            self.assertAlmostEqual(bi.item(), 3.0, places=6)

    def test_output_shape(self):
        self.assertEqual(compute_rloo_baseline(torch.randn(8)).shape, (8,))

    def test_advantages_zero_mean(self):
        rewards = torch.randn(6)
        adv = rewards - compute_rloo_baseline(rewards)
        self.assertAlmostEqual(adv.sum().item(), 0.0, places=5)


# ---------------------------------------------------------------------------
# Tests: batched reward and baseline computation
# ---------------------------------------------------------------------------


class TestComputeFeatureMatchingRewardsBatched(unittest.TestCase):
    """Tests for the vectorized batch variants of reward computation."""

    def _make_inputs(self, batch_size: int = 3, num_rollouts: int = 4, d: int = 16):
        rollout_features = torch.randn(batch_size * num_rollouts, d)
        ref_features = torch.randn(batch_size, d)
        return rollout_features, ref_features

    def test_output_shape(self):
        rf, ref = self._make_inputs()
        rewards = compute_feature_matching_rewards_batched(rf, ref, num_rollouts=4)
        self.assertEqual(rewards.shape, (3 * 4,))

    def test_matches_per_item_loop(self):
        """Batched result must match the per-item result for every rollout."""
        batch_size, n, d = 3, 4, 16
        rf, ref = self._make_inputs(batch_size, n, d)

        rewards_batched = compute_feature_matching_rewards_batched(rf, ref, num_rollouts=n)

        for i in range(batch_size):
            item_rf = rf[i * n : (i + 1) * n]
            item_ref = ref[i]
            rewards_item = compute_feature_matching_rewards(item_rf, item_ref)
            torch.testing.assert_close(
                rewards_batched[i * n : (i + 1) * n], rewards_item
            )

    def test_single_rollout_no_diversity(self):
        """With n=1 the diversity term is zero so reward = alignment."""
        batch_size, d = 2, 8
        rf = torch.randn(batch_size, d)
        ref = torch.randn(batch_size, d)
        rewards = compute_feature_matching_rewards_batched(rf, ref, num_rollouts=1)
        # alignment = 2 * dot(rf[i], ref[i]) for each i
        for i in range(batch_size):
            expected = 2.0 * (rf[i] * ref[i]).sum()
            self.assertAlmostEqual(rewards[i].item(), expected.item(), places=5)

    def test_invalid_rollout_dim_raises(self):
        with self.assertRaises(ValueError):
            compute_feature_matching_rewards_batched(
                torch.randn(12), torch.randn(3, 4), num_rollouts=4
            )

    def test_invalid_ref_dim_raises(self):
        with self.assertRaises(ValueError):
            compute_feature_matching_rewards_batched(
                torch.randn(12, 8), torch.randn(8), num_rollouts=4
            )

    def test_mismatch_batch_size_raises(self):
        with self.assertRaises(ValueError):
            compute_feature_matching_rewards_batched(
                torch.randn(12, 8), torch.randn(4, 8), num_rollouts=4
            )

    def test_non_divisible_rollouts_raises(self):
        with self.assertRaises(ValueError):
            compute_feature_matching_rewards_batched(
                torch.randn(10, 8), torch.randn(3, 8), num_rollouts=4
            )


class TestComputeRLOOBaselineBatched(unittest.TestCase):
    """Tests for the vectorized batch variant of the RLOO baseline."""

    def test_output_shape(self):
        b = compute_rloo_baseline_batched(torch.randn(3 * 4), num_rollouts=4)
        self.assertEqual(b.shape, (3 * 4,))

    def test_matches_per_item_loop(self):
        """Batched result must match computing per-item baselines independently."""
        batch_size, n = 3, 4
        rewards = torch.randn(batch_size * n)

        baselines_batched = compute_rloo_baseline_batched(rewards, num_rollouts=n)

        for i in range(batch_size):
            item_r = rewards[i * n : (i + 1) * n]
            item_b = compute_rloo_baseline(item_r)
            torch.testing.assert_close(baselines_batched[i * n : (i + 1) * n], item_b)

    def test_single_rollout_gives_zero(self):
        rewards = torch.tensor([1.0, 2.0, 3.0])  # B=3, n=1
        b = compute_rloo_baseline_batched(rewards, num_rollouts=1)
        torch.testing.assert_close(b, torch.zeros(3))

    def test_advantages_zero_sum_per_context(self):
        """Advantages (reward - baseline) must sum to zero within each context."""
        batch_size, n = 4, 5
        rewards = torch.randn(batch_size * n)
        baselines = compute_rloo_baseline_batched(rewards, num_rollouts=n)
        adv = (rewards - baselines).reshape(batch_size, n)
        for i in range(batch_size):
            self.assertAlmostEqual(adv[i].sum().item(), 0.0, places=5)

    def test_non_divisible_rewards_raises(self):
        with self.assertRaises(ValueError):
            compute_rloo_baseline_batched(torch.randn(10), num_rollouts=4)


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
        for key in ("context_ids", "context_mask", "completion_ids", "completion_mask"):
            self.assertIn(key, out)

    def test_context_padded_to_max_length(self):
        out = collate_fn(self._make_batch(), pad_token_id=0)
        self.assertEqual(out["context_ids"].shape[1], 3)

    def test_completion_padded_to_max_length(self):
        out = collate_fn(self._make_batch(), pad_token_id=0)
        self.assertEqual(out["completion_ids"].shape[1], 3)

    def test_mask_correctly_marks_padding(self):
        out = collate_fn(self._make_batch(), pad_token_id=0)
        self.assertEqual(out["context_mask"][1, 0].item(), 0)
        self.assertEqual(out["context_mask"][1, 1].item(), 1)
        self.assertEqual(out["context_mask"][1, 2].item(), 1)

    def test_completion_mask_marks_padding(self):
        out = collate_fn(self._make_batch(), pad_token_id=0)
        self.assertEqual(out["completion_mask"][0, 0].item(), 1)
        self.assertEqual(out["completion_mask"][0, 1].item(), 1)
        self.assertEqual(out["completion_mask"][0, 2].item(), 0)

    def test_left_padding_for_context(self):
        out = collate_fn(self._make_batch(), pad_token_id=0)
        self.assertEqual(out["context_ids"][1, 0].item(), 0)
        self.assertEqual(out["context_ids"][1, 1].item(), 6)
        self.assertEqual(out["context_ids"][1, 2].item(), 7)

    def test_batch_size(self):
        self.assertEqual(collate_fn(self._make_batch(), pad_token_id=0)["context_ids"].shape[0], 2)


class TestPretrainingDatasetSeparator(unittest.TestCase):
    """Verify that documents are separated by EOS tokens."""

    def test_eos_appears_between_documents(self):
        """A synthetic run through the dataset builder should insert the EOS id."""
        from ebp.data import PretrainingDataset
        from unittest.mock import patch, MagicMock

        EOS_ID = 99

        # Two documents each yielding distinct token ranges
        doc1_tokens = list(range(1, 21))    # 20 tokens
        doc2_tokens = list(range(101, 121)) # 20 tokens

        # Minimal tokenizer-like object
        class _MockTokenizer:
            eos_token_id = EOS_ID
            _calls = iter([doc1_tokens, doc2_tokens])

            def encode(self, text, add_special_tokens=False):
                return next(self._calls)

        tokenizer = _MockTokenizer()

        mock_dataset = [
            {"text": "a " * 20},
            {"text": "b " * 20},
        ]

        with patch("ebp.data.load_dataset", return_value=mock_dataset):
            dataset = PretrainingDataset(
                tokenizer=tokenizer,
                dataset_name="dummy",
                context_length=10,
                completion_length=5,
                stride=15,
                min_doc_chars=1,
            )

        # Flatten all stored tokens and check EOS is present between docs
        all_stored = []
        for ex in dataset.examples:
            all_stored.extend(ex)

        self.assertIn(EOS_ID, all_stored, "EOS separator not found between documents")

        # All tokens should be from doc1, EOS, or doc2 — nothing else
        valid = set(doc1_tokens) | {EOS_ID} | set(doc2_tokens)
        self.assertTrue(
            all(t in valid for t in all_stored),
            "Unexpected tokens found in stored examples",
        )


# ---------------------------------------------------------------------------
# Integration smoke tests
# ---------------------------------------------------------------------------


class TestEMAEndToEndStep(unittest.TestCase):
    def test_forward_backward_smoke(self):
        ebp = make_ema_model()
        ebp.model.train()
        batch_size, context_len, gen_len, n = 2, 8, 4, 3

        context_ids = torch.randint(0, TINY_CONFIG.vocab_size, (batch_size, context_len))
        context_mask = torch.ones(batch_size, context_len, dtype=torch.long)
        completion_ids = torch.randint(0, TINY_CONFIG.vocab_size, (batch_size, gen_len))
        completion_mask = torch.ones(batch_size, gen_len, dtype=torch.long)

        full_ids = torch.cat([context_ids, completion_ids], dim=1)
        full_mask = torch.cat([context_mask, completion_mask], dim=1)
        ref_features = ebp.extract_features(full_ids, full_mask, completion_start=context_len)

        rollout_ids, rollout_masks = ebp.generate_rollouts(
            context_ids, context_mask, num_rollouts=n, generation_length=gen_len
        )
        rollout_features, rollout_log_probs = ebp.compute_rollout_data(
            rollout_ids, rollout_masks, completion_start=context_len
        )

        total_loss = torch.tensor(0.0)
        for i in range(batch_size):
            rewards = compute_feature_matching_rewards(
                rollout_features[i * n : (i + 1) * n], ref_features[i]
            )
            advantages = (rewards - compute_rloo_baseline(rewards)).detach()
            total_loss = total_loss - (advantages * rollout_log_probs[i * n : (i + 1) * n]).mean()

        (total_loss / batch_size).backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in ebp.model.parameters()
        )
        self.assertTrue(has_grad)
        ebp.update_ema()


class TestOnlineEndToEndStep(unittest.TestCase):
    def test_forward_backward_smoke(self):
        ebp = make_online_model()
        ebp.model.train()
        batch_size, context_len, gen_len, n = 2, 8, 4, 3

        context_ids = torch.randint(0, TINY_CONFIG.vocab_size, (batch_size, context_len))
        context_mask = torch.ones(batch_size, context_len, dtype=torch.long)
        completion_ids = torch.randint(0, TINY_CONFIG.vocab_size, (batch_size, gen_len))
        completion_mask = torch.ones(batch_size, gen_len, dtype=torch.long)

        full_ids = torch.cat([context_ids, completion_ids], dim=1)
        full_mask = torch.cat([context_mask, completion_mask], dim=1)
        ref_features = ebp.extract_features(full_ids, full_mask, completion_start=context_len)

        rollout_ids, rollout_masks = ebp.generate_rollouts(
            context_ids, context_mask, num_rollouts=n, generation_length=gen_len
        )
        rollout_features, rollout_log_probs = ebp.compute_rollout_data(
            rollout_ids, rollout_masks, completion_start=context_len
        )

        total_loss = torch.tensor(0.0)
        for i in range(batch_size):
            rewards = compute_feature_matching_rewards(
                rollout_features[i * n : (i + 1) * n], ref_features[i]
            )
            advantages = (rewards - compute_rloo_baseline(rewards)).detach()
            total_loss = total_loss - (advantages * rollout_log_probs[i * n : (i + 1) * n]).mean()

        (total_loss / batch_size).backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in ebp.model.parameters()
        )
        self.assertTrue(has_grad)


if __name__ == "__main__":
    unittest.main()
