"""
Token Importance Unit (TIU) golden model for LASSO.

Implements the scoring pipeline:
  C_t = max(softmax(QK^T / sqrt(d))) over all heads/tokens
  H_t = -sum(p * log2(p)) averaged over heads (entropy)
  score = w_C * C_t + w_H * (1 - H_t_norm)

Tokens 0..(sink_count-1) are always RETAIN (attention sinks).
Tokens with score >= threshold are RETAIN; others are EVICT.
"""

import math
import numpy as np
from typing import Tuple


def compute_softmax(scores: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax (subtract max first).

    Parameters
    ----------
    scores : np.ndarray, shape (N,)
        Raw attention logits or any 1-D array.

    Returns
    -------
    probs : np.ndarray, shape (N,), dtype float64
        Probability distribution summing to 1.
    """
    scores = np.asarray(scores, dtype=np.float64)
    shifted = scores - scores.max()
    exp_s = np.exp(shifted)
    return exp_s / exp_s.sum()


def compute_ct(attn_weights: np.ndarray) -> float:
    """
    Compute C_t = maximum softmax weight over all heads and tokens.

    Parameters
    ----------
    attn_weights : np.ndarray, shape (n_heads, n_tokens)
        Each row should be a valid softmax distribution over tokens.
        If rows are raw logits, softmax is applied per head.

    Returns
    -------
    float in [0, 1]
    """
    attn_weights = np.asarray(attn_weights, dtype=np.float64)
    if attn_weights.ndim == 1:
        attn_weights = attn_weights[np.newaxis, :]

    n_heads, n_tokens = attn_weights.shape
    max_weight = 0.0
    for h in range(n_heads):
        row = attn_weights[h]
        # Normalise: if row sums to ~1 already, no-op; otherwise apply softmax
        row_sum = row.sum()
        if abs(row_sum - 1.0) < 1e-6:
            probs = row
        else:
            probs = compute_softmax(row)
        max_weight = max(max_weight, float(probs.max()))

    return float(np.clip(max_weight, 0.0, 1.0))


def compute_ht(attn_weights: np.ndarray) -> float:
    """
    Compute H_t = -sum(p * log2(p)) averaged over heads.

    Uses safe log: 0 * log(0) = 0.

    Parameters
    ----------
    attn_weights : np.ndarray, shape (n_heads, n_tokens)

    Returns
    -------
    float >= 0
    """
    attn_weights = np.asarray(attn_weights, dtype=np.float64)
    if attn_weights.ndim == 1:
        attn_weights = attn_weights[np.newaxis, :]

    n_heads, _ = attn_weights.shape
    head_entropies = []
    for h in range(n_heads):
        row = attn_weights[h]
        row_sum = row.sum()
        if abs(row_sum - 1.0) < 1e-6:
            probs = row
        else:
            probs = compute_softmax(row)
        # Safe entropy
        with np.errstate(divide="ignore", invalid="ignore"):
            log_p = np.where(probs > 0, np.log2(probs), 0.0)
        entropy = float(-np.sum(probs * log_p))
        head_entropies.append(entropy)

    return float(np.mean(head_entropies))


def normalize_entropy(ht: float, seq_len: int) -> float:
    """
    Normalize entropy by log2(seq_len).

    Uses floor of log2(max(seq_len, 2)) to avoid divide-by-zero for seq_len<=1.

    Parameters
    ----------
    ht : float
        Raw entropy value.
    seq_len : int
        Sequence length.

    Returns
    -------
    float
    """
    denom = math.log2(max(seq_len, 2))
    return float(ht / denom)


def compute_importance_score(
    ct: float,
    ht: float,
    seq_len: int,
    w_c: float = 0.6,
    w_h: float = 0.4,
) -> float:
    """
    Composite importance score.

    score = w_C * C_t + w_H * (1 - H_t_norm)

    Parameters
    ----------
    ct : float
        C_t value in [0, 1].
    ht : float
        H_t entropy value.
    seq_len : int
        Sequence length for normalizing entropy.
    w_c, w_h : float
        Weights (need not sum to 1).

    Returns
    -------
    float
    """
    ht_norm = normalize_entropy(ht, seq_len)
    return float(w_c * ct + w_h * (1.0 - ht_norm))


def should_retain(
    token_idx: int,
    score: float,
    threshold: float = 0x4000 / 0xFFFF,
    sink_count: int = 4,
) -> bool:
    """
    Retention decision for a token.

    Always RETAIN if token_idx < sink_count (attention sinks).
    Otherwise RETAIN if score >= threshold.

    Parameters
    ----------
    token_idx : int
        0-based token position in sequence.
    score : float
        Importance score from compute_importance_score.
    threshold : float
        Eviction threshold (default ≈ 0.2500, i.e. 0x4000/0xFFFF).
    sink_count : int
        Number of attention sink tokens always retained.

    Returns
    -------
    bool : True = RETAIN, False = EVICT
    """
    if token_idx < sink_count:
        return True
    return bool(score >= threshold)


def score_token(
    token_idx: int,
    attn_weights: np.ndarray,
    threshold: float = 0x4000 / 0xFFFF,
    w_c: float = 0.6,
    w_h: float = 0.4,
    sink_count: int = 4,
) -> Tuple[str, float]:
    """
    Full TIU pipeline for one token position.

    Parameters
    ----------
    token_idx : int
        0-based position of this token.
    attn_weights : np.ndarray, shape (n_heads, n_tokens)
        Attention weights for this token position.
    threshold : float
        Eviction threshold.
    w_c, w_h : float
        Importance score weights.
    sink_count : int
        Number of attention sink tokens.

    Returns
    -------
    (tag, score) : (str, float)
        tag is 'RETAIN' or 'EVICT'.
    """
    attn_weights = np.asarray(attn_weights, dtype=np.float64)

    # Attention sinks bypass scoring entirely
    if token_idx < sink_count:
        return ("RETAIN", 1.0)

    seq_len = attn_weights.shape[-1] if attn_weights.ndim > 1 else len(attn_weights)

    ct = compute_ct(attn_weights)
    ht = compute_ht(attn_weights)
    score = compute_importance_score(ct, ht, seq_len, w_c, w_h)

    tag = "RETAIN" if should_retain(token_idx, score, threshold, sink_count) else "EVICT"
    return (tag, score)
