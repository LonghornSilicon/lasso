"""
LACU (Lightweight Attention Compute Unit) golden model for LASSO.

Implements FlashAttention-style tiling:
  - Never materializes full N×N attention matrix
  - Running max and sum for numerically stable softmax
  - Tile-by-tile accumulation of output

Reference: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact
Attention with IO-Awareness", NeurIPS 2022.
"""

import math
import numpy as np
from typing import Tuple

TILE_SIZE = 64  # Default tile size


INT32_MAX = 2_147_483_647
INT64_MAX = 9_223_372_036_854_775_807


def fixed_point_dot_product_int64(a: np.ndarray, b: np.ndarray) -> int:
    """
    Integer dot product using INT64 accumulator — matches RTL requirement.

    For head_dim=64, INT16 inputs: max single dot product = 64 × 32767²
    = 6.87×10¹⁰, which overflows INT32 (max 2.15×10⁹) but fits INT64.

    On DSP48E2 (ZCU102/ZCU104): cascade two DSPs or use 48-bit accumulator
    with carry chain to reach INT64. The 48-bit accumulator of DSP48E2 covers
    up to 2.81×10¹⁴, which is sufficient for any single INT16×INT16 dot
    product up to head_dim=4096 (4096 × 32767² = 4.40×10¹²).

    Returns
    -------
    int : accumulated dot product (INT64)
    """
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    result = int(np.dot(a, b))
    if abs(result) > INT64_MAX:
        raise OverflowError(f"INT64 overflow: {result}")
    return result


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """
    Simple integer/float dot product of two 1-D arrays.

    Examples
    --------
    >>> dot_product([1, 0, 0], [1, 2, 3])
    1
    >>> dot_product([1, 2, 3], [1, 2, 3])
    14
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.dot(a, b))


def softmax_update(
    running_max: float,
    running_sum: float,
    running_output: np.ndarray,
    scores: np.ndarray,
    v_tile: np.ndarray,
) -> Tuple[float, float, np.ndarray]:
    """
    Update running FlashAttention statistics with one new tile.

    This is the online softmax update step. Given:
      - (m_old, l_old, acc_old): state before this tile
      - scores: raw dot-product scores for this tile, shape (tile_len,)
      - v_tile: value vectors for this tile, shape (tile_len, head_dim)

    Computes:
      m_new = max(m_old, max(scores))
      l_new = exp(m_old - m_new) * l_old + sum(exp(scores - m_new))
      acc_new = exp(m_old - m_new) * acc_old + sum_k(exp(score_k - m_new) * V_k)

    Parameters
    ----------
    running_max : float
        Current running maximum (m_old). Initialize to -inf.
    running_sum : float
        Current running normalization sum (l_old). Initialize to 0.
    running_output : np.ndarray, shape (head_dim,)
        Current accumulated output (acc_old). Initialize to zeros.
    scores : np.ndarray, shape (tile_len,)
        Raw QK^T scores for this tile (already divided by sqrt(d) by caller).
    v_tile : np.ndarray, shape (tile_len, head_dim)
        Value vectors for this tile.

    Returns
    -------
    (new_max, new_sum, new_output)
    """
    scores = np.asarray(scores, dtype=np.float64)
    v_tile = np.asarray(v_tile, dtype=np.float64)
    running_output = np.asarray(running_output, dtype=np.float64)

    tile_max = float(scores.max()) if len(scores) > 0 else -math.inf
    new_max = max(running_max, tile_max)

    # Rescale old state
    scale_old = math.exp(running_max - new_max) if not math.isinf(running_max) else 0.0

    # Compute exp weights for this tile
    exp_scores = np.exp(scores - new_max)  # shape (tile_len,)

    new_sum = scale_old * running_sum + float(exp_scores.sum())
    new_output = scale_old * running_output + np.dot(exp_scores, v_tile)

    return new_max, new_sum, new_output


def flash_attention_tile(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    tile_size: int = TILE_SIZE,
) -> np.ndarray:
    """
    FlashAttention tiled computation for one query vector.

    Never materializes the full N×N attention matrix. Uses running
    max/sum for numerically stable softmax across tiles.

    Parameters
    ----------
    Q : np.ndarray, shape (head_dim,)
        Query vector for this token position.
    K : np.ndarray, shape (seq_len, head_dim)
        Key matrix.
    V : np.ndarray, shape (seq_len, head_dim)
        Value matrix.
    tile_size : int
        Tile size for K/V (default 64).

    Returns
    -------
    output : np.ndarray, shape (head_dim,)
        Attention output vector.
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    head_dim = Q.shape[0]
    seq_len = K.shape[0]

    if seq_len == 0:
        return np.zeros(head_dim, dtype=np.float64)

    scale = 1.0 / math.sqrt(head_dim)

    # Initialize running state
    running_max = -math.inf
    running_sum = 0.0
    running_output = np.zeros(head_dim, dtype=np.float64)

    # Process K/V in tiles
    for tile_start in range(0, seq_len, tile_size):
        tile_end = min(tile_start + tile_size, seq_len)
        k_tile = K[tile_start:tile_end]   # shape (tile_len, head_dim)
        v_tile = V[tile_start:tile_end]   # shape (tile_len, head_dim)

        # Compute raw scores for this tile: Q @ K_tile^T
        scores = (k_tile @ Q) * scale    # shape (tile_len,)

        running_max, running_sum, running_output = softmax_update(
            running_max, running_sum, running_output, scores, v_tile
        )

    # Normalize
    if running_sum == 0.0:
        return np.zeros(head_dim, dtype=np.float64)

    output = running_output / running_sum
    return output


def attention_reference(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
) -> np.ndarray:
    """
    Non-tiled reference attention implementation.

    Materializes the full N-length attention weight vector for one query.
    Used for correctness comparison against flash_attention_tile.

    Parameters
    ----------
    Q : np.ndarray, shape (head_dim,)
    K : np.ndarray, shape (seq_len, head_dim)
    V : np.ndarray, shape (seq_len, head_dim)

    Returns
    -------
    output : np.ndarray, shape (head_dim,)
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    seq_len = K.shape[0]
    head_dim = Q.shape[0]

    if seq_len == 0:
        return np.zeros(head_dim, dtype=np.float64)

    scale = 1.0 / math.sqrt(head_dim)
    scores = (K @ Q) * scale   # shape (seq_len,)

    # Numerically stable softmax
    scores_shifted = scores - scores.max()
    exp_scores = np.exp(scores_shifted)
    attn_weights = exp_scores / exp_scores.sum()

    output = attn_weights @ V   # shape (head_dim,)
    return output
