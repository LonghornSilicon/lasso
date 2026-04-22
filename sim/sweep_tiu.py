"""
TIU (Token Importance Unit) simulation sweep.

Tests:
  1. Eviction rate curve sweep
  2. Attention weight underflow sweep
  3. Entropy edge cases
  4. Weight sensitivity sweep
  5. GQA stress sweep
  6. Sink count boundary sweep
"""

import math
import numpy as np
from sim.logger import SimLogger

from golden_model.tiu import (
    compute_ct,
    compute_ht,
    compute_softmax,
    compute_importance_score,
    should_retain,
    score_token,
    normalize_entropy,
)

BLOCK = "TIU"


def _sweep_eviction_rate_curve(logger: SimLogger, rng: np.random.Generator):
    """Sweep 1: Eviction rate curve."""
    test = "eviction_rate_curve"
    w_c = 0.6
    w_h = 0.4
    seq_len = 64
    n_heads = 4
    n_samples = 100
    sink_count = 4

    for threshold in np.linspace(0, 1, 21):
        threshold = float(threshold)
        n_evicted = 0
        n_non_sink = 0

        for _ in range(n_samples):
            # Random attention weight matrix [n_heads, seq_len]
            raw = rng.random((n_heads, seq_len))
            # Normalize rows to be valid softmax distributions
            attn = raw / raw.sum(axis=1, keepdims=True)

            for tok_idx in range(sink_count, seq_len):
                n_non_sink += 1
                ct = compute_ct(attn)
                ht = compute_ht(attn)
                score = compute_importance_score(ct, ht, seq_len, w_c, w_h)
                if not should_retain(tok_idx, score, threshold, sink_count):
                    n_evicted += 1

        eviction_rate = n_evicted / max(n_non_sink, 1)

        # Log EDGE for every threshold (builds the eviction curve)
        logger.log(
            BLOCK, test, "EDGE",
            {"threshold": round(threshold, 3), "eviction_rate": round(eviction_rate, 4),
             "n_samples": n_samples},
            f"eviction_rate={eviction_rate:.4f} at threshold={threshold:.3f}",
            severity="EDGE",
        )

        # Warn for degenerate rates at non-extreme thresholds
        if 0.1 <= threshold <= 0.9:
            if eviction_rate == 0.0:
                logger.log(
                    BLOCK, test, "WARN",
                    {"threshold": round(threshold, 3), "eviction_rate": 0.0},
                    f"0% eviction at non-extreme threshold {threshold:.3f} — "
                    f"scoring may be too generous",
                    severity="WARN",
                )
            elif eviction_rate == 1.0:
                logger.log(
                    BLOCK, test, "WARN",
                    {"threshold": round(threshold, 3), "eviction_rate": 1.0},
                    f"100% eviction at non-extreme threshold {threshold:.3f} — "
                    f"scoring may be too aggressive",
                    severity="WARN",
                )


def _sweep_attn_underflow(logger: SimLogger, rng: np.random.Generator):
    """Sweep 2: Attention weight underflow."""
    test = "attn_underflow"

    # All very small weights (near zero)
    tiny = np.full((4, 64), 1e-30, dtype=np.float64)
    try:
        # softmax should handle this via numerical stability (subtract max)
        probs = compute_softmax(tiny[0])
        has_nan = bool(np.any(np.isnan(probs)))
        has_inf = bool(np.any(np.isinf(probs)))
        if has_nan or has_inf:
            logger.log(
                BLOCK, test, "FAIL",
                {"input": "all_1e-30"},
                f"softmax on near-zero inputs produced NaN={has_nan} Inf={has_inf}",
                severity="FAIL",
            )
        else:
            logger.log(
                BLOCK, test, "OK",
                {"input": "all_1e-30", "probs_sum": float(probs.sum())},
                "softmax handles near-zero inputs without NaN/Inf",
                severity="PASS",
            )
    except Exception as e:
        logger.log(
            BLOCK, test, "FAIL",
            {"input": "all_1e-30"},
            f"softmax raised {type(e).__name__}: {e}",
            severity="FAIL",
        )

    # Perfectly peaked: one weight = 1.0, rest = 0.0
    peaked = np.zeros((4, 64), dtype=np.float64)
    peaked[:, 0] = 1.0  # all attention on token 0
    try:
        ht = compute_ht(peaked)
        if ht > 0.01:
            logger.log(
                BLOCK, test, "FAIL",
                {"input": "peaked_attn", "H_t": round(ht, 6)},
                f"H_t={ht:.6f} > 0.01 for perfectly peaked attention (expected ~0)",
                severity="FAIL",
            )
        else:
            logger.log(
                BLOCK, test, "OK",
                {"input": "peaked_attn", "H_t": round(ht, 6)},
                f"H_t={ht:.6f} ≈ 0 for perfectly peaked attention",
                severity="PASS",
            )
    except Exception as e:
        logger.log(
            BLOCK, test, "FAIL",
            {"input": "peaked_attn"},
            f"compute_ht raised {type(e).__name__}: {e}",
            severity="FAIL",
        )


def _sweep_entropy_edge_cases(logger: SimLogger, rng: np.random.Generator):
    """Sweep 3: Entropy edge cases."""
    test = "entropy_edge"

    # seq_len = 1: H_t should be 0
    attn1 = np.array([[1.0]])  # 1 head, 1 token
    ht1 = compute_ht(attn1)
    if abs(ht1) > 1e-9:
        logger.log(
            BLOCK, test, "FAIL",
            {"seq_len": 1, "H_t": ht1},
            f"H_t={ht1} for seq_len=1 (expected 0)",
            severity="FAIL",
        )
    else:
        logger.log(
            BLOCK, test, "OK",
            {"seq_len": 1, "H_t": ht1},
            "H_t=0 for seq_len=1 (correct)",
            severity="PASS",
        )

    # seq_len = 2, uniform: H_t should be 1.0
    attn2 = np.array([[0.5, 0.5]])
    ht2 = compute_ht(attn2)
    if abs(ht2 - 1.0) > 0.01:
        logger.log(
            BLOCK, test, "FAIL",
            {"seq_len": 2, "H_t": round(ht2, 6)},
            f"H_t={ht2:.6f} for uniform seq_len=2 (expected 1.0)",
            severity="FAIL",
        )
    else:
        logger.log(
            BLOCK, test, "OK",
            {"seq_len": 2, "H_t": round(ht2, 6)},
            f"H_t={ht2:.6f} ≈ 1.0 for uniform seq_len=2",
            severity="PASS",
        )

    # seq_len = 1024: no overflow
    try:
        attn_large = rng.random((4, 1024))
        attn_large = attn_large / attn_large.sum(axis=1, keepdims=True)
        ht_large = compute_ht(attn_large)
        ct_large = compute_ct(attn_large)
        score_large = compute_importance_score(ct_large, ht_large, 1024)
        logger.log(
            BLOCK, test, "OK",
            {"seq_len": 1024, "H_t": round(ht_large, 4), "C_t": round(ct_large, 4),
             "score": round(score_large, 4)},
            "No overflow for seq_len=1024",
            severity="PASS",
        )
    except Exception as e:
        logger.log(
            BLOCK, test, "FAIL",
            {"seq_len": 1024},
            f"seq_len=1024 raised {type(e).__name__}: {e}",
            severity="FAIL",
        )


def _sweep_weight_sensitivity(logger: SimLogger, rng: np.random.Generator):
    """Sweep 4: Weight sensitivity sweep — find flip point."""
    test = "weight_sensitivity"
    # Fixed C_t and H_t_norm
    ct = 0.7
    # H_t raw: assume seq_len=10, so log2(10) ≈ 3.32; H_t=1.0 gives H_t_norm≈0.3
    seq_len = 10
    ht_raw = 1.0  # H_t_norm = 1.0 / log2(10)
    threshold = 0.5

    flip_w_c = None
    prev_decision = None

    for w_c in np.linspace(0, 1, 1001):
        w_h = 1.0 - float(w_c)
        score = compute_importance_score(ct, ht_raw, seq_len, float(w_c), w_h)
        decision = "RETAIN" if score >= threshold else "EVICT"
        if prev_decision is not None and decision != prev_decision:
            flip_w_c = float(w_c)
            logger.log(
                BLOCK, test, "EDGE",
                {"flip_w_c": round(flip_w_c, 4), "C_t": ct, "H_t": ht_raw,
                 "seq_len": seq_len, "threshold": threshold,
                 "score_at_flip": round(score, 6)},
                f"Retain/evict decision flips at w_C={flip_w_c:.4f} "
                f"(score={score:.6f} vs threshold={threshold})",
                severity="EDGE",
            )
            break
        prev_decision = decision

    if flip_w_c is None:
        logger.log(
            BLOCK, test, "WARN",
            {"C_t": ct, "H_t": ht_raw, "seq_len": seq_len, "threshold": threshold},
            "No flip point found across w_C in [0,1] — decision never changes",
            severity="WARN",
        )


def _sweep_gqa_stress(logger: SimLogger, rng: np.random.Generator):
    """Sweep 5: GQA stress sweep."""
    test = "gqa_stress"
    n_tokens = 16
    default_threshold = 0x4000 / 0xFFFF
    sink_count = 4

    for n_kv_heads in [1, 2, 4, 8]:
        for n_q_heads in [8, 16, 32]:
            attn = rng.random((n_kv_heads, n_tokens))
            attn = attn / attn.sum(axis=1, keepdims=True)

            try:
                for tok_idx in [5, 10]:  # non-sink tokens
                    tag, score = score_token(
                        tok_idx, attn,
                        threshold=default_threshold,
                        sink_count=sink_count,
                    )
                    if score < 0 or score > 2.0:
                        logger.log(
                            BLOCK, test, "WARN",
                            {"n_kv_heads": n_kv_heads, "n_q_heads": n_q_heads,
                             "tok_idx": tok_idx, "score": round(score, 4)},
                            f"Score {score:.4f} outside expected [0, 2.0] range",
                            severity="WARN",
                        )

                logger.log(
                    BLOCK, test, "OK",
                    {"n_kv_heads": n_kv_heads, "n_q_heads": n_q_heads, "n_tokens": n_tokens},
                    f"score_token works for n_kv_heads={n_kv_heads}, n_q_heads={n_q_heads}",
                    severity="PASS",
                )
            except Exception as e:
                logger.log(
                    BLOCK, test, "FAIL",
                    {"n_kv_heads": n_kv_heads, "n_q_heads": n_q_heads},
                    f"score_token raised {type(e).__name__}: {e}",
                    severity="FAIL",
                )


def _sweep_sink_count_boundary(logger: SimLogger, rng: np.random.Generator):
    """Sweep 6: Sink count boundary sweep."""
    test = "sink_count_boundary"
    seq_len = 20
    n_heads = 4
    attn = rng.random((n_heads, seq_len))
    attn = attn / attn.sum(axis=1, keepdims=True)
    default_threshold = 0x4000 / 0xFFFF

    # sink_count = 0: tokens 0-3 can be evicted
    sink_count = 0
    tokens_03_can_evict = True
    for tok_idx in range(4):
        # Score with low score to force eviction path
        tag, score = score_token(tok_idx, attn, threshold=default_threshold, sink_count=sink_count)
        # With sink_count=0, token 0 should use scoring — might RETAIN or EVICT based on score
        # We just verify it doesn't always force RETAIN like sink behavior
    # Actually check: with threshold=1.0 (impossible to meet), all non-sink tokens evicted
    for tok_idx in range(4):
        tag, _ = score_token(tok_idx, attn, threshold=1.1, sink_count=0)
        if tag == "RETAIN":
            tokens_03_can_evict = False

    logger.log(
        BLOCK, test, "EDGE",
        {"sink_count": 0, "tokens_03_evictable": tokens_03_can_evict},
        f"With sink_count=0 and threshold=1.1: tokens 0-3 {'can' if tokens_03_can_evict else 'cannot'} be evicted",
        severity="EDGE",
    )

    # sink_count = 4 (default): tokens 0-3 always RETAIN
    all_retained = True
    for tok_idx in range(4):
        tag, _ = score_token(tok_idx, attn, threshold=1.1, sink_count=4)
        if tag != "RETAIN":
            all_retained = False

    if all_retained:
        logger.log(
            BLOCK, test, "OK",
            {"sink_count": 4},
            "Tokens 0-3 always RETAIN with sink_count=4 (correct)",
            severity="PASS",
        )
    else:
        logger.log(
            BLOCK, test, "FAIL",
            {"sink_count": 4},
            "Some tokens 0-3 were EVICT with sink_count=4 — sink behavior broken",
            severity="FAIL",
        )

    # sink_count = 16: first 16 always RETAIN
    all16_retained = True
    for tok_idx in range(16):
        tag, _ = score_token(tok_idx, attn, threshold=1.1, sink_count=16)
        if tag != "RETAIN":
            all16_retained = False

    if all16_retained:
        logger.log(
            BLOCK, test, "OK",
            {"sink_count": 16},
            "Tokens 0-15 always RETAIN with sink_count=16 (correct)",
            severity="PASS",
        )
    else:
        logger.log(
            BLOCK, test, "FAIL",
            {"sink_count": 16},
            "Some tokens 0-15 were EVICT with sink_count=16 — sink behavior broken",
            severity="FAIL",
        )

    # sink_count > seq_len: sink_count=100, seq_len=10 — should not crash
    attn_small = rng.random((n_heads, 10))
    attn_small = attn_small / attn_small.sum(axis=1, keepdims=True)
    try:
        tag, score = score_token(5, attn_small, threshold=default_threshold, sink_count=100)
        logger.log(
            BLOCK, test, "EDGE",
            {"sink_count": 100, "seq_len": 10, "tok_idx": 5, "tag": tag},
            f"sink_count=100 > seq_len=10 does not crash. tok_idx=5 tag={tag}",
            severity="EDGE",
        )
    except Exception as e:
        logger.log(
            BLOCK, test, "FAIL",
            {"sink_count": 100, "seq_len": 10},
            f"sink_count > seq_len raised {type(e).__name__}: {e}",
            severity="FAIL",
        )


def run_tiu_sweeps(logger: SimLogger):
    """Run all TIU sweeps."""
    rng = np.random.default_rng(42)
    _sweep_eviction_rate_curve(logger, rng)
    _sweep_attn_underflow(logger, rng)
    _sweep_entropy_edge_cases(logger, rng)
    _sweep_weight_sensitivity(logger, rng)
    _sweep_gqa_stress(logger, rng)
    _sweep_sink_count_boundary(logger, rng)
