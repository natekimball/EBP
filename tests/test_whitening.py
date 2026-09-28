import torch
import unittest
from ebp.rewards import (
    _whiten_via_gram,
    compute_whitened_feature_matching_terms_batched,
    compute_whitened_feature_matching_rewards,
    compute_whitened_feature_matching_rewards_batched,
    compute_feature_matching_rewards,
    compute_feature_matching_rewards_batched
)

class TestWhitenedRewards(unittest.TestCase):
    def test_whitening_shape(self):
        n, d = 5, 10
        rollout_features = torch.randn(n, d)
        ref_feature = torch.randn(d)
        rewards = compute_whitened_feature_matching_rewards(rollout_features, ref_feature)
        self.assertEqual(rewards.shape, (n,))

    def test_batched_whitening_shape(self):
        b, n, d = 3, 4, 8
        rollout_features = torch.randn(b * n, d)
        ref_features = torch.randn(b, d)
        rewards = compute_whitened_feature_matching_rewards_batched(rollout_features, ref_features, num_rollouts=n)
        self.assertEqual(rewards.shape, (b * n,))

    def test_batched_matches_single(self):
        b, n, d = 2, 4, 8
        rollout_features = torch.randn(b * n, d)
        ref_features = torch.randn(b, d)
        
        rewards_batched = compute_whitened_feature_matching_rewards_batched(
            rollout_features, ref_features, num_rollouts=n
        )
        
        for i in range(b):
            item_rf = rollout_features[i * n : (i + 1) * n]
            item_ref = ref_features[i]
            rewards_single = compute_whitened_feature_matching_rewards(item_rf, item_ref)
            torch.testing.assert_close(
                rewards_batched[i * n : (i + 1) * n], 
                rewards_single
            )

    def test_alignment_is_normalized(self):
        # In whitened space, the alignment term uses unit vectors.
        # r = 2 * (phi_w / |phi_w|) . (phi_y_w / |phi_y_w|) - diversity
        # If we have only 1 rollout, diversity is 0.
        # So reward should be 2 * cos_sim(phi_w, phi_y_w).
        # If phi_w and phi_y_w are collinear, reward should be 2.0.
        
        n, d = 1, 4
        # Even with 1 rollout, whitening still happens (though sigma is rank 1)
        rf = torch.randn(n, d)
        ref = torch.randn(d)
        
        # We need more than 1 rollout for a non-singular sigma if d > 1, 
        # but our code handles small eigenvalues with eps.
        rewards = compute_whitened_feature_matching_rewards(rf, ref)
        # For n=1, diversity=0. Alignment uses unit vectors.
        # 2 * dot(u1, u2) is in [-2, 2].
        self.assertTrue(torch.all(rewards <= 2.0001))
        self.assertTrue(torch.all(rewards >= -2.0001))

    def test_whitening_matches_explicit_pseudoinverse(self):
        """Gram-based whitening must equal (Sigma^dagger)^(1/2) applied directly.

        Sigma is rank <= n, so a jittered inverse of the full (D, D) matrix
        amplifies the null space by 1/sqrt(jitter) instead of projecting it
        out.  Comparing against an explicit truncated pseudo-inverse pins the
        correct behaviour.
        """
        torch.manual_seed(0)
        b, n, d = 3, 4, 32
        rf = torch.nn.functional.normalize(torch.randn(b, n, d).double(), dim=-1)
        ref = torch.nn.functional.normalize(torch.randn(b, d).double(), dim=-1)

        rf_w, ref_w = _whiten_via_gram(rf, ref)

        for i in range(b):
            sigma = rf[i].T @ rf[i] / n
            lam, q = torch.linalg.eigh(sigma)
            keep = lam > 1e-6 * lam.max()
            inv_sqrt = torch.where(keep, lam.clamp(min=1e-30).rsqrt(), torch.zeros_like(lam))
            w = (q * inv_sqrt) @ q.T
            torch.testing.assert_close(rf_w[i].double(), rf[i] @ w, atol=1e-8, rtol=1e-6)
            torch.testing.assert_close(ref_w[i].double(), ref[i] @ w, atol=1e-8, rtol=1e-6)

    def test_alignment_recovers_matching_rollout(self):
        """A rollout identical to the reference must score alignment ~= 2.

        Feature dim is far larger than the rollout count, the regime the model
        actually runs in (D = 3072, n = 4 for Qwen3-0.6B).
        """
        torch.manual_seed(0)
        b, n, d = 2, 4, 512
        rf = torch.nn.functional.normalize(torch.randn(b, n, d), dim=-1)
        ref = rf[:, 0].clone()  # reference == first rollout

        alignment, _ = compute_whitened_feature_matching_terms_batched(
            rf.reshape(b * n, d), ref, num_rollouts=n
        )
        alignment = alignment.reshape(b, n)

        # The matching rollout scores ~2; the others are near-orthogonal.
        torch.testing.assert_close(
            alignment[:, 0], torch.full((b,), 2.0), atol=1e-3, rtol=1e-3
        )
        self.assertTrue(torch.all(alignment[:, 1:].abs() < 1.0))


if __name__ == "__main__":
    unittest.main()
