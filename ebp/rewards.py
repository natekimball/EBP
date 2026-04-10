"""
Feature-matching rewards and REINFORCE baselines for EBP.

Implements the per-rollout reward from Equation (7) of the EBFT paper:

    r_j = 2 φ(c:ŷ_j)ᵀ φ(c:y)  −  2/(n-1) Σ_{j'≠j} φ(c:ŷ_j)ᵀ φ(c:ŷ_{j'})
         ↑ alignment term             ↑ diversity term

and the REINFORCE Leave-One-Out (RLOO) baseline used to reduce variance in
the policy-gradient update.
"""

from __future__ import annotations

import torch


def compute_feature_matching_rewards(
    rollout_features: torch.Tensor,
    ref_feature: torch.Tensor,
) -> torch.Tensor:
    """Compute the feature-matching reward for each rollout (Eq. 7, EBFT).

    Args:
        rollout_features: ``(n, D)`` feature vectors of the ``n`` sampled
            completions.
        ref_feature: ``(D,)`` feature vector of the ground-truth completion.

    Returns:
        rewards: ``(n,)`` scalar reward for each rollout.
    """
    if rollout_features.dim() != 2:
        raise ValueError(
            f"rollout_features must be 2-D (n, D), got shape {rollout_features.shape}"
        )
    if ref_feature.dim() != 1:
        raise ValueError(
            f"ref_feature must be 1-D (D,), got shape {ref_feature.shape}"
        )

    n = rollout_features.shape[0]

    # Alignment term: 2 φ_j · φ_y  →  (n,)
    alignment = 2.0 * torch.mv(rollout_features, ref_feature)

    # Diversity term: 2/(n-1) Σ_{j'≠j} φ_j · φ_{j'}
    if n > 1:
        pairwise = rollout_features @ rollout_features.T  # (n, n)
        # Sum of off-diagonal entries for each row
        sum_others = pairwise.sum(dim=1) - pairwise.diagonal()  # (n,)
        diversity = (2.0 / (n - 1)) * sum_others
    else:
        diversity = torch.zeros(n, device=rollout_features.device, dtype=rollout_features.dtype)

    return alignment - diversity  # (n,)


def compute_rloo_baseline(rewards: torch.Tensor) -> torch.Tensor:
    """REINFORCE Leave-One-Out (RLOO) baseline.

    For rollout ``j`` the baseline is the mean reward of the remaining
    ``n - 1`` rollouts:

        b_j = (Σ_{j'} r_{j'} − r_j) / (n − 1)

    For a single rollout (``n == 1``) a zero baseline is returned.

    Args:
        rewards: ``(n,)`` reward for each rollout.

    Returns:
        baselines: ``(n,)`` RLOO baseline for each rollout.
    """
    n = rewards.shape[0]
    if n <= 1:
        return torch.zeros_like(rewards)

    total = rewards.sum()
    baselines = (total - rewards) / (n - 1)
    return baselines
