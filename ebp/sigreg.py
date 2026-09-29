"""Sketched Isotropic Gaussian Regularization (SIGReg).

SIGReg, introduced with LeJEPA, prevents representation collapse by pushing the
embedding distribution toward an isotropic Gaussian.  Rather than working with
the ``(D, D)`` covariance directly it projects embeddings onto random 1-D
directions and enforces that each resulting univariate distribution matches
``N(0, 1)``, which keeps cost linear in the embedding dimension.

Two variants are provided:

* :func:`full_sigreg` – the Epps-Pulley goodness-of-fit statistic on each 1-D
  slice, comparing the empirical characteristic function against that of a
  standard normal.  Matches *all* moments.
* :func:`weak_sigreg` – the cheaper Weak-SIGReg variant, which sketches to
  ``k`` dimensions and penalises ``||Cov - I||_F``.  Matches the second moment
  only.

**Which to use here.** The usual guidance prefers the weak variant for speed,
but that assumes a large batch.  A ``(k, k)`` covariance estimated from ``N``
samples is rank-deficient whenever ``N < k`` — the same degeneracy that makes
the rollout-covariance whitening meaningless (see
:func:`ebp.rewards._whiten_via_gram`).  A 1-D goodness-of-fit test needs far
fewer samples per slice, so :func:`full_sigreg` is the better-conditioned
choice at the batch sizes used here.

**Target distribution.** EBP features are L2-normalised per layer, so they live
on a unit sphere and cannot be Gaussian.  Callers should pass
``sqrt(d) * phi_hat``: if the direction of ``phi`` is uniform on the sphere —
the maximally non-collapsed state, and what an isotropic Gaussian becomes after
normalisation — then each coordinate has variance ``1/d`` and the rescaled
vector matches ``N(0, I)``.  Working with the normalised direction also makes
the penalty independent of the language model's activation scale, so it does
not fight the cross-entropy objective for control of the residual stream.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


def _random_directions(
    d: int,
    num_slices: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample ``num_slices`` unit vectors uniformly from the sphere ``S^{d-1}``.

    Returns:
        ``(d, num_slices)`` matrix of unit-norm columns.
    """
    a = torch.randn(d, num_slices, device=device, dtype=dtype, generator=generator)
    return a / a.norm(dim=0, keepdim=True).clamp(min=1e-12)


def full_sigreg(
    z: torch.Tensor,
    num_slices: int = 8,
    num_quadrature: int = 16,
    domain: float = 6.0,
    bandwidth: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Epps-Pulley SIGReg: match ``N(0, I)`` on random 1-D projections.

    For each random unit direction ``a`` the projected samples ``a^T z`` should
    be standard normal.  The Epps-Pulley statistic measures this by comparing
    the empirical characteristic function ``phi_N(t) = mean(exp(i t a^T z))``
    against the standard-normal characteristic function ``exp(-t^2 / 2)``,
    integrated against a Gaussian window::

        T = \\int | phi_N(t) - exp(-t^2/2) |^2 exp(-t^2 / (2 * bandwidth^2)) dt

    The conventional ``N`` prefactor is omitted so the value does not scale with
    batch size and the loss weight need not be retuned when the batch changes.

    Args:
        z: ``(N, d)`` embeddings, already scaled so the target is ``N(0, I)``.
        num_slices: Number of random 1-D projections (``M``).
        num_quadrature: Trapezoid points across the integration domain.
        domain: Integrate over ``[-domain, domain]``.
        bandwidth: Width of the Gaussian window.
        generator: Optional RNG for reproducible slice directions.

    Returns:
        Scalar loss, zero when the projections are standard normal.
    """
    if z.dim() != 2:
        raise ValueError(f"z must be 2-D (N, d), got shape {tuple(z.shape)}")

    # The characteristic function and its gradients are sensitive to rounding.
    work_dtype = torch.float32 if z.dtype in (torch.float16, torch.bfloat16) else z.dtype
    z = z.to(work_dtype)

    directions = _random_directions(
        z.shape[1], num_slices, z.device, work_dtype, generator
    )
    proj = z @ directions  # (N, M)

    t = torch.linspace(-domain, domain, num_quadrature, device=z.device, dtype=work_dtype)

    # Empirical characteristic function of each slice, evaluated on the grid.
    arg = proj.unsqueeze(-1) * t  # (N, M, P)
    real = arg.cos().mean(dim=0)  # (M, P)
    imag = arg.sin().mean(dim=0)  # (M, P)

    target_real = torch.exp(-0.5 * t.square())  # CF of N(0, 1) is real
    sq_dist = (real - target_real).square() + imag.square()  # (M, P)

    window = torch.exp(-t.square() / (2.0 * bandwidth**2))
    integrand = sq_dist * window
    per_slice = torch.trapezoid(integrand, t, dim=-1)  # (M,)
    return per_slice.mean()


def weak_sigreg(
    z: torch.Tensor,
    sketch_dim: int = 64,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Weak-SIGReg: penalise ``||Cov(sketch) - I||_F`` on a random sketch.

    Cheaper than :func:`full_sigreg` but constrains only the second moment, and
    it needs ``N > sketch_dim`` samples for the covariance to be full rank.  The
    sketch columns are orthonormalised so that an isotropic ``z`` maps to an
    isotropic sketch, making ``I`` the correct target.

    Args:
        z: ``(N, d)`` embeddings, already scaled so the target is ``N(0, I)``.
        sketch_dim: Sketch width ``k``; clamped to ``min(k, d, N - 1)``.
        generator: Optional RNG for reproducible sketch directions.

    Returns:
        Scalar loss, zero when the sketched covariance is the identity.
    """
    if z.dim() != 2:
        raise ValueError(f"z must be 2-D (N, d), got shape {tuple(z.shape)}")

    n, d = z.shape
    work_dtype = torch.float32 if z.dtype in (torch.float16, torch.bfloat16) else z.dtype
    z = z.to(work_dtype)

    k = max(1, min(sketch_dim, d, n - 1))
    s = torch.randn(d, k, device=z.device, dtype=work_dtype, generator=generator)
    # Orthonormal columns: an isotropic z then sketches to an isotropic (k,).
    s, _ = torch.linalg.qr(s)

    centered = z - z.mean(dim=0, keepdim=True)
    sketch = centered @ s  # (N, k)
    cov = (sketch.T @ sketch) / max(n - 1, 1)
    eye = torch.eye(k, device=z.device, dtype=work_dtype)
    return torch.linalg.matrix_norm(cov - eye, ord="fro")


def sigreg_loss(
    features: torch.Tensor,
    num_layers: int,
    mode: str = "full",
    num_slices: int = 8,
    sketch_dim: int = 64,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """SIGReg over per-layer feature blocks of a concatenated feature vector.

    EBP concatenates ``num_layers`` separately L2-normalised blocks, so the
    penalty is applied per block and averaged rather than to the concatenation.
    Each block is rescaled by ``sqrt(d)`` so that a direction uniform on the
    unit sphere is the zero-loss solution.

    Args:
        features: ``(N, num_layers * d)`` L2-normalised per-layer features.
        num_layers: Number of concatenated per-layer blocks.
        mode: ``"full"`` (Epps-Pulley) or ``"weak"`` (covariance sketch).
        num_slices: Random 1-D projections, ``"full"`` mode only.
        sketch_dim: Sketch width, ``"weak"`` mode only.
        generator: Optional RNG for reproducible directions.

    Returns:
        Scalar loss averaged over the per-layer blocks.
    """
    n, total = features.shape
    if total % num_layers != 0:
        raise ValueError(
            f"feature dim ({total}) must be divisible by num_layers ({num_layers})"
        )
    d = total // num_layers
    blocks = features.reshape(n, num_layers, d)

    scale = math.sqrt(d)
    losses = []
    for i in range(num_layers):
        z = blocks[:, i, :] * scale
        if mode == "full":
            losses.append(full_sigreg(z, num_slices=num_slices, generator=generator))
        elif mode == "weak":
            losses.append(weak_sigreg(z, sketch_dim=sketch_dim, generator=generator))
        else:
            raise ValueError(f"unknown SIGReg mode {mode!r}; expected 'full' or 'weak'")
    return torch.stack(losses).mean()


def mean_pairwise_cosine(features: torch.Tensor) -> torch.Tensor:
    """Mean off-diagonal cosine similarity — a direct collapse diagnostic.

    Approaches 1.0 as representations collapse to a single direction, and sits
    near 0 for well-spread features in high dimension.

    Args:
        features: ``(N, D)`` feature vectors.

    Returns:
        Scalar mean off-diagonal cosine similarity (0 when ``N < 2``).
    """
    n = features.shape[0]
    if n < 2:
        return features.new_zeros(())
    unit = features / features.norm(dim=1, keepdim=True).clamp(min=1e-12)
    gram = unit @ unit.T
    off_diag_sum = gram.sum() - gram.diagonal().sum()
    return off_diag_sum / (n * (n - 1))
