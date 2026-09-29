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


# ---------------------------------------------------------------------------
# Whitening core
# ---------------------------------------------------------------------------


def _whiten_via_gram(
    rf: torch.Tensor,
    ref: torch.Tensor,
    rtol: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply ``(Sigma_c^dagger)^{1/2}`` to rollout and reference features.

    ``Sigma_c = (1/n) Phi^T Phi`` is the ``(D, D)`` second-moment matrix of the
    ``n`` rollout features.  Because it is built from only ``n`` vectors its
    rank is at most ``n``, which is far below ``D`` in practice (``n = 4`` vs
    ``D = 3072`` for Qwen3-0.6B).  Eigendecomposing the ``(D, D)`` matrix is
    therefore both wasteful (``O(D^3)``) and numerically meaningless: the
    ``D - n`` null-space eigenvalues are pure floating-point noise.

    Instead we work with the ``(n, n)`` Gram matrix ``G = (1/n) Phi Phi^T``,
    which shares the non-zero spectrum of ``Sigma_c``.  Writing
    ``G = U S^2 U^T`` gives the exact identities::

        Phi_whitened = U S^-1 U^T Phi                       (n, D)
        phi_y_whitened = (1/n) Phi^T U S^-3 U^T (Phi phi_y)  (D,)

    Eigenvalues are truncated at ``rtol * lambda_max`` (the standard
    ``pinv`` convention) so directions outside the rollout span are projected
    out rather than amplified by ``1/sqrt(jitter)``.

    Args:
        rf: ``(B, n, D)`` rollout features.
        ref: ``(B, D)`` reference features.
        rtol: Relative eigenvalue cutoff for the pseudo-inverse.

    Returns:
        rf_w: ``(B, n, D)`` whitened rollout features.
        ref_w: ``(B, D)`` whitened reference features.
    """
    n = rf.shape[1]
    orig_dtype = rf.dtype

    # eigh is not implemented for half precision on CUDA, and the inverse
    # square root is sensitive to rounding, so decompose in at least float32.
    work_dtype = (
        torch.float32
        if orig_dtype in (torch.float16, torch.bfloat16)
        else orig_dtype
    )
    rf32 = rf.to(work_dtype)
    ref32 = ref.to(work_dtype)

    gram = torch.bmm(rf32, rf32.transpose(1, 2)) / n  # (B, n, n)
    lam, u = torch.linalg.eigh(gram)  # lam = S^2, ascending

    # Pseudo-inverse truncation relative to the largest eigenvalue.
    cutoff = rtol * lam[:, -1:].clamp(min=0.0)
    keep = lam > cutoff
    safe_lam = lam.clamp(min=torch.finfo(work_dtype).tiny)
    inv_sqrt = torch.where(keep, safe_lam.rsqrt(), torch.zeros_like(lam))  # S^-1
    inv_cube = torch.where(keep, safe_lam.pow(-1.5), torch.zeros_like(lam))  # S^-3

    # Rollouts: U S^-1 U^T Phi
    g_inv_sqrt = (u * inv_sqrt.unsqueeze(1)) @ u.transpose(1, 2)  # (B, n, n)
    rf_w = torch.bmm(g_inv_sqrt, rf32)  # (B, n, D)

    # Reference: (1/n) Phi^T U S^-3 U^T (Phi phi_y)
    proj = torch.bmm(rf32, ref32.unsqueeze(2))  # (B, n, 1)
    tmp = torch.bmm(u.transpose(1, 2), proj) * inv_cube.unsqueeze(2)
    tmp = torch.bmm(u, tmp)  # (B, n, 1)
    ref_w = torch.bmm(rf32.transpose(1, 2), tmp).squeeze(2) / n  # (B, D)

    return rf_w.to(orig_dtype), ref_w.to(orig_dtype)


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

    n = rollout_features.shape[0]

    rf_w, ref_w = _whiten_via_gram(
        rollout_features.unsqueeze(0), ref_feature.unsqueeze(0), rtol=eps
    )
    rollout_features_w = rf_w.squeeze(0)  # (n, D)
    ref_feature_w = ref_w.squeeze(0)      # (D,)

    # Alignment term (normalised): 2 * (phi_tilde_j / ||.||) . (phi_tilde_y / ||.||)
    rollout_norms = torch.linalg.vector_norm(rollout_features_w, dim=1)
    ref_norm = torch.linalg.vector_norm(ref_feature_w)

    rollout_unit = rollout_features_w / rollout_norms[:, None].clamp(min=eps)
    ref_unit = ref_feature_w / ref_norm.clamp(min=eps)

    alignment = 2.0 * (rollout_unit @ ref_unit)

    # Diversity term (unnormalised): 2/(n-1) * sum_{j'!=j} phi_tilde_j . phi_tilde_j'
    if n > 1:
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


@torch.compiler.disable
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

    Note:
        With ``Sigma_c`` estimated from *only* the ``n`` rollouts it whitens,
        the diversity term is identically zero.  Writing ``Phi / sqrt(n) =
        U S V^T``, the whitened Gram matrix is::

            Phi_tilde Phi_tilde^T = Phi Sigma_c^dagger Phi^T = n U U^T = n I

        because ``U`` is square orthogonal when ``rank(Phi) = n``.  Every
        off-diagonal inner product is therefore exactly 0.

        This is a property of the *estimator*, not of whitening in general.
        It holds whenever the sample set behind ``Sigma`` has full row rank
        (i.e. no more samples than feature dimensions) and contains the
        rollouts.  Here that is badly the case: ``n = 4`` samples in
        ``D = 3072`` dimensions.  With a full-rank ``Sigma`` — estimated over
        ``N >> D`` samples, e.g. a running covariance over the dataset — the
        whitened features are *not* mutually orthogonal and the diversity term
        carries signal again.

        So the anti-collapse penalty is inactive under ``--whitening`` as
        currently implemented.  Fixing it means changing how ``Sigma`` is
        estimated, not removing either term.
    """
    bn, d = rollout_features.shape
    n = num_rollouts
    b = bn // n

    rf = rollout_features.reshape(b, n, d)
    rf_w, ref_w = _whiten_via_gram(rf, ref_features, rtol=eps)

    # Alignment term (normalised)
    rollout_norms = torch.linalg.vector_norm(rf_w, dim=2)          # (B, n)
    ref_norms = torch.linalg.vector_norm(ref_w, dim=1)             # (B,)

    rollout_unit = rf_w / rollout_norms.unsqueeze(2).clamp(min=eps)
    ref_unit = ref_w / ref_norms.unsqueeze(1).clamp(min=eps)

    alignment = 2.0 * (rollout_unit * ref_unit.unsqueeze(1)).sum(dim=-1)  # (B, n)

    # Diversity term (unnormalised)
    if n > 1:
        pairwise = torch.bmm(rf_w, rf_w.transpose(1, 2))           # (B, n, n)
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


# ---------------------------------------------------------------------------
# Advantages
# ---------------------------------------------------------------------------


def compute_advantages_batched(
    rewards: torch.Tensor,
    num_rollouts: int,
    min_reward_std: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RLOO advantages, optionally dropping groups with no reward spread.

    Deliberately does **not** divide by the per-group standard deviation.  The
    leave-one-out baseline already makes this an unbiased estimator of the
    policy gradient; dividing by a std computed from the same ``n`` samples
    reintroduces bias and rescales every context to unit gradient magnitude
    regardless of whether its rollouts differed for a reason.  In an RLVR
    setting with binary rewards a group where every rollout scores the same has
    ``std == 0`` and drops out on its own.  Feature-matching rewards are
    continuous, so such groups have *small* rather than zero spread and
    normalisation would promote pure noise to full weight.

    ``min_reward_std`` is the continuous analogue of DAPO's dynamic sampling:
    groups whose reward spread falls below the threshold contribute no
    gradient, instead of being amplified.

    Args:
        rewards: ``(B * n,)`` reward per rollout, context ``i`` occupying
            positions ``[i * n : (i + 1) * n]``.
        num_rollouts: Number of rollouts per context (``n``).
        min_reward_std: Groups with a reward std at or below this value are
            zeroed out.  ``0.0`` disables filtering.

    Returns:
        advantages: ``(B * n,)`` advantages, zeroed for filtered groups.
        group_std: ``(B,)`` per-context reward standard deviation.
        active: ``(B,)`` bool mask of groups that survived filtering.
    """
    n = num_rollouts
    total = rewards.shape[0]
    if total % n != 0:
        raise ValueError(
            f"rewards length ({total}) must be divisible by num_rollouts ({n})"
        )
    batch_size = total // n

    advantages = rewards - compute_rloo_baseline_batched(rewards, n)

    if n > 1:
        group_std = rewards.reshape(batch_size, n).std(dim=1)
    else:
        group_std = torch.zeros(
            batch_size, device=rewards.device, dtype=rewards.dtype
        )

    if min_reward_std > 0.0:
        active = group_std > min_reward_std
        advantages = advantages * active.repeat_interleave(n).to(advantages.dtype)
    else:
        active = torch.ones(batch_size, device=rewards.device, dtype=torch.bool)

    return advantages, group_std, active


def reinforce_loss_from_advantages(
    advantages: torch.Tensor,
    log_probs: torch.Tensor,
    completion_token_counts: torch.Tensor,
    loss_agg: str = "token",
) -> torch.Tensor:
    """REINFORCE loss with a choice of aggregation.

    ``log_probs`` are *summed* over each completion, so averaging them over
    sequences (``loss_agg="sequence"``) weights every rollout equally
    regardless of length — the per-sample aggregation that Dr. GRPO identifies
    as length-biased.  ``loss_agg="token"`` divides by the total number of
    completion tokens in the batch instead (DAPO-style), so each *token*
    carries equal weight.

    Args:
        advantages: ``(B * n,)`` advantages (already detached).
        log_probs: ``(B * n,)`` summed completion log-probabilities.
        completion_token_counts: ``(B * n,)`` unmasked completion tokens per
            rollout.
        loss_agg: ``"token"`` or ``"sequence"``.

    Returns:
        Scalar REINFORCE loss.
    """
    weighted = -(advantages * log_probs)
    if loss_agg == "token":
        return weighted.sum() / completion_token_counts.sum().clamp(min=1.0)
    if loss_agg == "sequence":
        return weighted.mean()
    raise ValueError(f"unknown loss_agg {loss_agg!r}; expected 'token' or 'sequence'")
