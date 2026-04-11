"""
Energy-Based Pre-training (EBP) package.

Implements a pre-training objective inspired by Energy-Based Fine-Tuning (EBFT),
using either an Exponential Moving Average (EMA) or the live model itself as
the feature network.
"""

from ebp.model import EMAEBPModel, OnlineEBPModel
from ebp.rewards import (
    compute_feature_matching_rewards,
    compute_feature_matching_rewards_batched,
    compute_feature_matching_terms_batched,
    compute_rloo_baseline,
    compute_rloo_baseline_batched,
)
from ebp.data import PretrainingDataset

__all__ = [
    "EMAEBPModel",
    "OnlineEBPModel",
    "compute_feature_matching_rewards",
    "compute_feature_matching_rewards_batched",
    "compute_feature_matching_terms_batched",
    "compute_rloo_baseline",
    "compute_rloo_baseline_batched",
    "PretrainingDataset",
]
