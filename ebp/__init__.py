"""
Energy-Based Pre-training (EBP) package.

Implements a pre-training objective inspired by Energy-Based Fine-Tuning (EBFT),
using an Exponential Moving Average (EMA) of the model as a dynamic feature
network instead of a fully frozen copy.
"""

from ebp.model import EBPModel
from ebp.rewards import compute_feature_matching_rewards, compute_rloo_baseline
from ebp.data import PretrainingDataset

__all__ = [
    "EBPModel",
    "compute_feature_matching_rewards",
    "compute_rloo_baseline",
    "PretrainingDataset",
]
