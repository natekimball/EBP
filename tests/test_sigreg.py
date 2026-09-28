import math
import unittest

import torch

from ebp.sigreg import (
    full_sigreg,
    mean_pairwise_cosine,
    sigreg_loss,
    weak_sigreg,
)


def _sphere(n, d, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(n, d, generator=g), dim=1)


def _collapsed(n, d, jitter=0.01, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(1, d, generator=g).repeat(n, 1)
    return torch.nn.functional.normalize(
        base + jitter * torch.randn(n, d, generator=g), dim=1
    )


class TestSIGReg(unittest.TestCase):
    def test_isotropic_gaussian_is_near_zero(self):
        torch.manual_seed(0)
        self.assertLess(full_sigreg(torch.randn(512, 64)).item(), 0.02)

    def test_penalises_collapse(self):
        torch.manual_seed(0)
        iso = torch.randn(512, 64)
        collapsed = torch.randn(1, 64).repeat(512, 1) + 0.01 * torch.randn(512, 64)
        self.assertGreater(full_sigreg(collapsed).item(), 10 * full_sigreg(iso).item())

    def test_detects_dimensional_collapse_that_cosine_misses(self):
        """Rank-deficient features are collapsed but pairwise-orthogonal.

        Mean cosine similarity cannot see this; SIGReg must.
        """
        torch.manual_seed(0)
        low_rank = torch.randn(512, 4) @ torch.randn(4, 64)
        iso = torch.randn(512, 64)
        self.assertLess(abs(mean_pairwise_cosine(low_rank).item()), 0.05)
        self.assertGreater(full_sigreg(low_rank).item(), 10 * full_sigreg(iso).item())

    def test_sphere_target_uniform_is_near_zero(self):
        """sqrt(d)-scaled uniform directions are the zero-loss solution."""
        feats = _sphere(512, 64).repeat(1, 3)
        self.assertLess(sigreg_loss(feats, num_layers=3, mode="full").item(), 0.02)

    def test_sphere_target_collapsed_is_penalised(self):
        good = sigreg_loss(_sphere(512, 64).repeat(1, 3), 3, "full").item()
        bad = sigreg_loss(_collapsed(512, 64).repeat(1, 3), 3, "full").item()
        self.assertGreater(bad, 10 * good)

    def test_gradient_descent_undoes_collapse(self):
        z = _collapsed(64, 64, jitter=0.05).repeat(1, 3).clone().requires_grad_(True)
        opt = torch.optim.Adam([z], lr=0.05)
        start = sigreg_loss(z, 3, "full").item()
        for _ in range(60):
            opt.zero_grad()
            loss = sigreg_loss(z, 3, "full")
            loss.backward()
            opt.step()
        self.assertLess(loss.item(), start)
        self.assertLess(mean_pairwise_cosine(z.detach()).item(), 0.5)

    def test_weak_sketch_dim_clamped_below_sample_count(self):
        """A (k, k) covariance from N < k samples is rank-deficient; the sketch
        width must clamp rather than silently produce a degenerate estimate."""
        z = torch.randn(8, 64)
        self.assertTrue(torch.isfinite(weak_sigreg(z, sketch_dim=64)))

    def test_mean_pairwise_cosine_extremes(self):
        d = 32
        same = torch.ones(4, d)
        self.assertAlmostEqual(mean_pairwise_cosine(same).item(), 1.0, places=4)
        self.assertEqual(mean_pairwise_cosine(torch.randn(1, d)).item(), 0.0)

    def test_rejects_bad_shapes_and_modes(self):
        with self.assertRaises(ValueError):
            full_sigreg(torch.randn(4, 4, 4))
        with self.assertRaises(ValueError):
            sigreg_loss(torch.randn(8, 10), num_layers=3)
        with self.assertRaises(ValueError):
            sigreg_loss(torch.randn(8, 9), num_layers=3, mode="bogus")


if __name__ == "__main__":
    unittest.main()
