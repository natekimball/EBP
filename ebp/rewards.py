"""
Feature-matching rewards and REINFORCE baselines for EBP.

Implements the per-rollout reward from Equation (7) of the EBFT paper:

    r_j = 2 φ(c:ŷ_j)ᵀ φ(c:y)  −  2/(n-1) Σ_{j'≠j} φ(c:ŷ_j)ᵀ φ(c:ŷ_{j'})
         ↑ alignment term             ↑ diversity term

and the REINFORCE Leave-One-Out (RLOO) baseline used to reduce variance in
the policy-gradient update.

Batched/vectorized variants (``*_batched``) operate over an entire batch of
``B`` contexts each with ``n`` rollouts in a single tensor operation, avoiding
a Python-level loop over batch items.
"""

from __future__ import annotations

import torch


def compute_feature_matching_terms(
    rollout_features: torch.Tensor,
    ref_feature: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute alignment and diversity terms used in Eq. 7 (EBFT).

    Args:
        rollout_features: ``(n, D)`` feature vectors of the ``n`` sampled
            completions.
        ref_feature: ``(D,)`` feature vector of the ground-truth completion.

    Returns:
        alignment: ``(n,)`` term ``2 * phi_j · phi_y``.
        diversity: ``(n,)`` term ``2/(n-1) * sum_{j'!=j} phi_j · phi_j'``.
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

    # Alignment term: 2 phi_j · phi_y  ->  (n,)
    alignment = 2.0 * torch.mv(rollout_features, ref_feature)

    # Diversity term: 2/(n-1) * sum_{j'!=j} phi_j · phi_j'
    if n > 1:
        pairwise = rollout_features @ rollout_features.T  # (n, n)
        sum_others = pairwise.sum(dim=1) - pairwise.diagonal()  # (n,)
        diversity = (2.0 / (n - 1)) * sum_others
    else:
        diversity = torch.zeros(
            n, device=rollout_features.device, dtype=rollout_features.dtype
        )

    return alignment, diversity


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
    alignment, diversity = compute_feature_matching_terms(
        rollout_features=rollout_features,
        ref_feature=ref_feature,
    )

    return alignment - diversity  # (n,)


def compute_whitened_feature_matching_terms(
    rollout_features: torch.Tensor,
    ref_feature: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute whitened alignment and diversity terms (Eq. 9, EBFT).

    Args:
        rollout_features: ``(n, D)`` feature vectors of the ``n`` sampled
            completions.
        ref_feature: ``(D,)`` feature vector of the ground-truth completion.
        eps: Small constant for numerical stability.

    Returns:
        alignment: ``(n,)`` term ``2 * (phi_j / ||phi_j||) · (phi_y / ||phi_y||)``.
        diversity: ``(n,)`` term ``2/(n-1) * sum_{j'!=j} phi_j · phi_j'``.
    """
    if rollout_features.dim() != 2:
        raise ValueError(
            f"rollout_features must be 2-D (n, D), got shape {rollout_features.shape}"
        )
    if ref_feature.dim() != 1:
        raise ValueError(
            f"ref_feature must be 1-D (D,), got shape {ref_feature.shape}"
        )

    n, d = rollout_features.shape

    # 1. Estimate second-moment matrix Σ_c = (1/n) Σ_j φ_j φ_jᵀ
    # (n, d) -> (d, d)
    sigma = (rollout_features.T @ rollout_features) / n

    # 2. Compute whitened features: φ̃ = (Σ_c^†)^1/2 φ
    # We use eigen-decomposition of Σ_c (symmetric positive semi-definite)
    # Σ_c = Q Λ Qᵀ  =>  (Σ_c^†)^1/2 = Q (Λ^†)^1/2 Qᵀ
    # NOTE: Since we apply (Σ_c^†)^1/2 to many vectors, we can just transform
    # everything into the whitened space.
    try:
        # torch.linalg.eigh is not implemented for BFloat16/Float16 on CUDA
        orig_dtype = sigma.dtype
        if sigma.device.type == "cuda" and orig_dtype in (torch.float16, torch.bfloat16):
            sigma = sigma.to(torch.float32)

        # Use symeig-like eigh for symmetric matrices
        L, Q = torch.linalg.eigh(sigma)

        # Avoid division by zero/negative eigenvalues (numerical noise)
        # L_inv_sqrt = 1 / sqrt(L) where L > eps
        mask = L > eps
        L_inv_sqrt = torch.zeros_like(L)
        L_inv_sqrt[mask] = 1.0 / torch.sqrt(L[mask])

        # Whitening matrix W = Q diag(L_inv_sqrt) Qᵀ
        # (d, d) @ (d, d) @ (d, d) -> (d, d)
        whitening_matrix = (Q * L_inv_sqrt) @ Q.T

        # Transform features in Float32 for maximum precision
        # (n, d) @ (d, d) -> (n, d)
        rollout_features_w = rollout_features.to(torch.float32) @ whitening_matrix
        ref_feature_w = ref_feature.to(torch.float32) @ whitening_matrix

        # Cast back to original precision AFTER transformation
        rollout_features_w = rollout_features_w.to(orig_dtype)
        ref_feature_w = ref_feature_w.to(orig_dtype)

    except torch.linalg.LinAlgError:
        # Fallback to unwhitened if decomposition fails
        rollout_features_w = rollout_features
        ref_feature_w = ref_feature

    # 3. Alignment term (normalized): 2 * (phi_tilde_j / ||phi_tilde_j||) · (phi_tilde_y / ||phi_tilde_y||)
    # Norms (n,) and scalar
    rollout_norms = torch.linalg.vector_norm(rollout_features_w, dim=1)
    ref_norm = torch.linalg.vector_norm(ref_feature_w)

    # Unit vectors for alignment (n, d) and (d,)
    # Use eps to avoid div-by-zero if norm is 0
    rollout_unit = rollout_features_w / (rollout_norms[:, None] + eps)
    ref_unit = ref_feature_w / (ref_norm + eps)

    alignment = 2.0 * (rollout_unit @ ref_unit)

    # 4. Diversity term (unnormalized): 2/(n-1) * sum_{j'!=j} phi_tilde_j · phi_tilde_j'
    if n > 1:
        # (n, d) @ (d, n) -> (n, n)
        pairwise = rollout_features_w @ rollout_features_w.T
        sum_others = pairwise.sum(dim=1) - pairwise.diagonal()  # (n,)
        diversity = (2.0 / (n - 1)) * sum_others
    else:
        diversity = torch.zeros(
            n, device=rollout_features.device, dtype=rollout_features.dtype
        )

    return alignment, diversity


def compute_whitened_feature_matching_rewards(
    rollout_features: torch.Tensor,
    ref_feature: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute the whitened feature-matching reward for each rollout (Eq. 9, EBFT).

    Args:
        rollout_features: ``(n, D)`` feature vectors of the ``n`` sampled
            completions.
        ref_feature: ``(D,)`` feature vector of the ground-truth completion.
        eps: Small constant for numerical stability.

    Returns:
        rewards: ``(n,)`` scalar reward for each rollout.
    """
    alignment, diversity = compute_whitened_feature_matching_terms(
        rollout_features=rollout_features,
        ref_feature=ref_feature,
        eps=eps,
    )

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


# ---------------------------------------------------------------------------
# Batched / vectorized variants  (operate over full batch without Python loop)
# ---------------------------------------------------------------------------


def compute_feature_matching_terms_batched(
    rollout_features: torch.Tensor,
    ref_features: torch.Tensor,
    num_rollouts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched alignment and diversity terms for a full batch of contexts.

    Processes all ``B`` contexts and their ``n = num_rollouts`` rollouts in a
    single vectorized operation, avoiding a Python-level loop over batch items.

    The layout of ``rollout_features`` follows the convention used by
    :meth:`~ebp.model.EMAEBPModel.compute_rollout_data`: rollouts for context
    ``i`` occupy rows ``[i * n : (i + 1) * n]``.

    Args:
        rollout_features: ``(B * n, D)`` feature vectors for all rollouts.
        ref_features: ``(B, D)`` feature vector for each ground-truth
            completion.
        num_rollouts: Number of rollouts per context (``n``).

    Returns:
        alignment: ``(B * n,)`` term ``2 * phi_j · phi_y``.
        diversity: ``(B * n,)`` term ``2/(n-1) * sum_{j'!=j} phi_j · phi_j'``.
    """
    if rollout_features.dim() != 2:
        raise ValueError(
            f"rollout_features must be 2-D (B*n, D), got shape {rollout_features.shape}"
        )
    if ref_features.dim() != 2:
        raise ValueError(
            f"ref_features must be 2-D (B, D), got shape {ref_features.shape}"
        )

    n = num_rollouts
    total = rollout_features.shape[0]
    if total % n != 0:
        raise ValueError(
            f"rollout_features first dimension ({total}) must be divisible by "
            f"num_rollouts ({n})"
        )
    batch_size = total // n
    D = rollout_features.shape[1]

    if ref_features.shape[0] != batch_size:
        raise ValueError(
            f"ref_features batch dimension ({ref_features.shape[0]}) must equal "
            f"rollout_features first dimension / num_rollouts ({batch_size})"
        )

    # Reshape to (B, n, D) for batched operations
    rf = rollout_features.reshape(batch_size, n, D)  # (B, n, D)

    # Alignment: 2 * phi_j . phi_y  ->  (B, n)
    # ref_features: (B, D) -> (B, 1, D) for broadcasting
    alignment = 2.0 * (rf * ref_features.unsqueeze(1)).sum(dim=-1)  # (B, n)

    # Diversity: 2/(n-1) * sum_{j'!=j} phi_j . phi_j'  ->  (B, n)
    if n > 1:
        # Batched pairwise dot products: (B, n, n)
        pairwise = torch.bmm(rf, rf.transpose(1, 2))  # (B, n, n)
        # Sum over all j' and subtract the j==j' term
        sum_others = pairwise.sum(dim=2) - pairwise.diagonal(dim1=1, dim2=2)  # (B, n)
        diversity = (2.0 / (n - 1)) * sum_others  # (B, n)
    else:
        diversity = torch.zeros(
            batch_size, n, device=rollout_features.device, dtype=rollout_features.dtype
        )

    return alignment.reshape(total), diversity.reshape(total)


def compute_whitened_feature_matching_terms_batched(
    rollout_features: torch.Tensor,
    ref_features: torch.Tensor,
    num_rollouts: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched whitened alignment and diversity terms (Eq. 9, EBFT).

    Args:
        rollout_features: ``(B * n, D)`` feature vectors for all rollouts.
        ref_features: ``(B, D)`` feature vector for each ground-truth
            completion.
        num_rollouts: Number of rollouts per context (``n``).
        eps: Small constant for numerical stability.

    Returns:
        alignment: ``(B * n,)`` term ``2 * (phi_tilde_j / ||phi_tilde_j||) · (phi_tilde_y / ||phi_tilde_y||)``.
        diversity: ``(B * n,)`` term ``2/(n-1) * sum_{j'!=j} phi_tilde_j · phi_tilde_j'``.
    """
    bn, d = rollout_features.shape
    n = num_rollouts
    b = bn // n

    # Reshape to (B, n, D) for batched operations
    rf = rollout_features.reshape(b, n, d)

    # 1. Estimate second-moment matrix Σ_c for each context
    # (B, D, n) @ (B, n, D) -> (B, D, D)
    sigma = torch.bmm(rf.transpose(1, 2), rf) / n

    # 2. Compute whitened features: φ̃ = (Σ_c^†)^1/2 φ
    try:
        # torch.linalg.eigh is not implemented for BFloat16/Float16 on CUDA
        orig_dtype = sigma.dtype
        if (
            sigma.device.type == "cuda"
            and orig_dtype in (torch.float16, torch.bfloat16)
        ):
            sigma = sigma.to(torch.float32)

        # Batch eigen-decomposition
        L, Q = torch.linalg.eigh(sigma)

        # L: (B, D), Q: (B, D, D)
        mask = L > eps
        L_inv_sqrt = torch.zeros_like(L)
        L_inv_sqrt[mask] = 1.0 / torch.sqrt(L[mask])

        # Whitening matrix W = Q diag(L_inv_sqrt) Qᵀ in FP32
        # ((B, D, D) * (B, 1, D) (B, D, D) -> (B, D, D)
        whitening_matrix = (Q * L_inv_sqrt.unsqueeze(1)) @ Q.transpose(1, 2)

        # Transform features in FP32
        # rf_w: (B, n, D) @ (B, D, D) -> (B, n, D)
        rf_w = torch.bmm(rf.to(torch.float32), whitening_matrix)

        # ref_features_w: (B, 1, D) @ (B, D, D) -> (B, 1, D) -> (B, D)
        ref_features_w = torch.bmm(ref_features.to(torch.float32).unsqueeze(1), whitening_matrix).squeeze(
            1
        )

        # Cast back to original precision AFTER transformation
        rf_w = rf_w.to(orig_dtype)
        ref_features_w = ref_features_w.to(orig_dtype)

    except torch.linalg.LinAlgError:
        rf_w = rf
        ref_features_w = ref_features

    # 3. Alignment term (normalized): 2 * (phi_tilde_j / ||phi_tilde_j||) · (phi_tilde_y / ||phi_tilde_y||)
    # rollout_norms: (B, n), ref_norms: (B,)
    rollout_norms = torch.linalg.vector_norm(rf_w, dim=2)
    ref_norms = torch.linalg.vector_norm(ref_features_w, dim=1)

    # Unit vectors
    rollout_unit = rf_w / (rollout_norms.unsqueeze(2) + eps)
    ref_unit = ref_features_w / (ref_norms.unsqueeze(1) + eps)

    # (B, n, D) * (B, 1, D) -> (B, n, D) -> (B, n)
    alignment = 2.0 * (rollout_unit * ref_unit.unsqueeze(1)).sum(dim=-1)

    # 4. Diversity term (unnormalized): 2/(n-1) * sum_{j'!=j} phi_tilde_j · phi_tilde_j'
    if n > 1:
        # (B, n, D) @ (B, D, n) -> (B, n, n)
        pairwise = torch.bmm(rf_w, rf_w.transpose(1, 2))
        sum_others = pairwise.sum(dim=2) - torch.diagonal(pairwise, dim1=1, dim2=2)
        diversity = (2.0 / (n - 1)) * sum_others
    else:
        diversity = torch.zeros(
            b, n, device=rollout_features.device, dtype=rollout_features.dtype
        )

    return alignment.reshape(bn), diversity.reshape(bn)


def compute_whitened_feature_matching_rewards_batched(
    rollout_features: torch.Tensor,
    ref_features: torch.Tensor,
    num_rollouts: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute whitened feature-matching rewards for a full batch of contexts.

    Args:
        rollout_features: ``(B * n, D)`` feature vectors for all rollouts.
        ref_features: ``(B, D)`` feature vector for each ground-truth
            completion.
        num_rollouts: Number of rollouts per context (``n``).
        eps: Small constant for numerical stability.

    Returns:
        rewards: ``(B * n,)`` scalar rewards for all rollouts.
    """
    alignment, diversity = compute_whitened_feature_matching_terms_batched(
        rollout_features=rollout_features,
        ref_features=ref_features,
        num_rollouts=num_rollouts,
        eps=eps,
    )

    return alignment - diversity  # (B * n,)


def compute_feature_matching_rewards_batched(
    rollout_features: torch.Tensor,
    ref_features: torch.Tensor,
    num_rollouts: int,
) -> torch.Tensor:
    """Batched feature-matching rewards for a full batch of contexts (Eq. 7).

    Vectorized counterpart of :func:`compute_feature_matching_rewards` that
    handles all ``B`` contexts at once.

    Args:
        rollout_features: ``(B * n, D)`` feature vectors for all rollouts.
        ref_features: ``(B, D)`` ground-truth feature vectors.
        num_rollouts: Number of rollouts per context (``n``).

    Returns:
        rewards: ``(B * n,)`` scalar reward for each rollout.
    """
    alignment, diversity = compute_feature_matching_terms_batched(
        rollout_features=rollout_features,
        ref_features=ref_features,
        num_rollouts=num_rollouts,
    )
    return alignment - diversity  # (B * n,)


def compute_rloo_baseline_batched(
    rewards: torch.Tensor,
    num_rollouts: int,
) -> torch.Tensor:
    """Batched REINFORCE Leave-One-Out (RLOO) baseline.

    Vectorized counterpart of :func:`compute_rloo_baseline` that operates over
    all ``B`` contexts simultaneously.

    For rollout ``j`` of context ``i`` the baseline is:

        b_{i,j} = (Σ_{j'} r_{i,j'} − r_{i,j}) / (n − 1)

    For a single rollout per context (``n == 1``) zero baselines are returned.

    Args:
        rewards: ``(B * n,)`` reward for each rollout, ordered so that rollouts
            for context ``i`` occupy positions ``[i * n : (i + 1) * n]``.
        num_rollouts: Number of rollouts per context (``n``).

    Returns:
        baselines: ``(B * n,)`` RLOO baseline for each rollout.
    """
    n = num_rollouts
    total = rewards.shape[0]
    if total % n != 0:
        raise ValueError(
            f"rewards length ({total}) must be divisible by num_rollouts ({n})"
        )

    if n <= 1:
        return torch.zeros_like(rewards)

    batch_size = total // n
    r = rewards.reshape(batch_size, n)  # (B, n)
    total_per_ctx = r.sum(dim=1, keepdim=True)  # (B, 1)
    baselines = (total_per_ctx - r) / (n - 1)  # (B, n)
    return baselines.reshape(total)  # (B * n,)
