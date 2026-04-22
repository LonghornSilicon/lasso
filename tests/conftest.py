"""
Shared pytest fixtures for LASSO golden model tests.
"""

import numpy as np
import pytest


@pytest.fixture
def sample_kv_group():
    """
    Return a 32-element INT16 numpy array with values in [-1000, 1000].
    Reproducible via fixed seed.
    """
    rng = np.random.default_rng(0xDEADBEEF)
    values = rng.integers(-1000, 1001, size=32, dtype=np.int16)
    return values


@pytest.fixture
def sample_attn_weights():
    """
    Return shape [4, 16] float array (4 heads, 16 tokens).
    Each row is a valid softmax distribution (sums to 1).
    """
    rng = np.random.default_rng(0xCAFEBABE)
    logits = rng.standard_normal((4, 16))
    # Stable softmax per row
    logits -= logits.max(axis=1, keepdims=True)
    exp_l = np.exp(logits)
    weights = exp_l / exp_l.sum(axis=1, keepdims=True)
    return weights.astype(np.float64)


@pytest.fixture
def beta_star_smollm_1_7b():
    """
    Return β* = 1.59 for SmolLM-1.7B from PRD calibration data.
    Derived from gap_mean = 0.424 / 0.267 ≈ 1.5880...
    """
    return 1.59
