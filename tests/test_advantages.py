import torch
import unittest

from ebp.rewards import (
    compute_advantages_batched,
    compute_rloo_baseline_batched,
    reinforce_loss_from_advantages,
)


class TestAdvantages(unittest.TestCase):
    def test_advantages_are_rloo_and_sum_to_zero_per_group(self):
        n, b = 4, 3
        rewards = torch.randn(b * n)
        adv, _, _ = compute_advantages_batched(rewards, n)
        torch.testing.assert_close(
            adv, rewards - compute_rloo_baseline_batched(rewards, n)
        )
        # RLOO advantages sum to zero within each group.
        torch.testing.assert_close(
            adv.reshape(b, n).sum(dim=1), torch.zeros(b), atol=1e-5, rtol=0
        )

    def test_no_per_group_std_normalisation(self):
        """Groups with more reward spread must produce larger advantages.

        Dividing by the per-group std would flatten these to equal magnitude,
        promoting uninformative contexts to full gradient weight.
        """
        n = 4
        wide = torch.tensor([2.0, -2.0, 1.0, -1.0])
        narrow = wide * 0.01
        adv, std, _ = compute_advantages_batched(torch.cat([wide, narrow]), n)
        wide_mag, narrow_mag = adv[:n].abs().mean(), adv[n:].abs().mean()
        self.assertGreater(wide_mag / narrow_mag, 50.0)
        self.assertGreater(std[0], std[1])

    def test_min_reward_std_zeroes_flat_groups(self):
        n = 4
        wide = torch.tensor([2.0, -2.0, 1.0, -1.0])
        flat = torch.tensor([0.5, 0.5, 0.5, 0.5])
        adv, _, active = compute_advantages_batched(
            torch.cat([wide, flat]), n, min_reward_std=0.01
        )
        self.assertTrue(bool(active[0]))
        self.assertFalse(bool(active[1]))
        torch.testing.assert_close(adv[n:], torch.zeros(n))
        self.assertGreater(adv[:n].abs().sum(), 0.0)

    def test_min_reward_std_disabled_by_default(self):
        n = 4
        flat = torch.tensor([0.5, 0.5, 0.5, 0.5])
        _, _, active = compute_advantages_batched(flat, n)
        self.assertTrue(bool(active[0]))

    def test_token_aggregation_is_length_unbiased(self):
        """Two rollouts, one twice as long, same per-token advantage-weighted
        log-prob: token aggregation weights them by length, sequence does not."""
        adv = torch.tensor([1.0, 1.0])
        log_probs = torch.tensor([-2.0, -4.0])   # -1/token over 2 and 4 tokens
        counts = torch.tensor([2.0, 4.0])

        tok = reinforce_loss_from_advantages(adv, log_probs, counts, "token")
        seq = reinforce_loss_from_advantages(adv, log_probs, counts, "sequence")
        # token: (2 + 4) / 6 = 1.0 -- every token weighted equally
        torch.testing.assert_close(tok, torch.tensor(1.0))
        # sequence: (2 + 4) / 2 = 3.0 -- the long rollout is over-weighted
        torch.testing.assert_close(seq, torch.tensor(3.0))

    def test_unknown_aggregation_rejected(self):
        with self.assertRaises(ValueError):
            reinforce_loss_from_advantages(
                torch.zeros(2), torch.zeros(2), torch.ones(2), "bogus"
            )


if __name__ == "__main__":
    unittest.main()
