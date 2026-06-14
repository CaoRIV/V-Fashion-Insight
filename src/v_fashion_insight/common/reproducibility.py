"""Utilities for deterministic pseudo-random behavior."""

import random

import numpy as np

from v_fashion_insight.common.constants import DEFAULT_RANDOM_SEED


def seed_everything(seed: int = DEFAULT_RANDOM_SEED) -> int:
    """Seed Python and NumPy pseudo-random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    return seed
