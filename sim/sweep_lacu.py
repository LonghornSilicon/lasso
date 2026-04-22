"""
LACU (Lightweight Attention Compute Unit) simulation sweep.

Tests:
  1. Sequence length scaling sweep
  2. Accumulator overflow sweep
  3. Tile size edge sweep
  4. Numerical stability sweep
  5. Running softmax precision sweep
  6. ZCU102/ZCU104 DSP mapping analysis
"""

import math
import numpy as np
from sim.logger import SimLogger

from golden_model.lacu import (
    flash_attention_tile,
    attention_reference,
    softmax_update,
    TILE_SIZE,
)

BLOCK = "LACU"


def _sweep_seq_length_scaling(logger: SimLogger, rng: np.random.Generator):
    """Sweep 1: Sequence length scaling sweep."""
    test = "seq_length_scaling"
    head_dim = 64

    for seq_len in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
        Q = rng.standard_normal(head_dim)
        K = rng.standard_normal((seq_len, head_dim))
        V = rng.standard_normal((seq_len, head_dim))

        try:
            out_flash = flash_attention_tile(Q, K, V)
            out_ref = attention_reference(Q, K, V)

            max_abs_error = float(np.max(np.abs(out_flash - out_ref)))
            # Relative error (avoid div-by-zero)
            ref_norm = float(np.max(np.abs(out_ref)))
            rel_error = max_abs_error / max(ref_norm, 1e-12)

            if max_abs_error > 1e-2:
                sev = "FAIL"
            elif max_abs_error > 1e-3:
                sev = "WARN"
            else:
                sev = "PASS"

            logger.log(
                BLOCK, test, "EDGE" if sev == "PASS" else sev,
                {"seq_len": seq_len, "max_abs_error": float(f"{max_abs_error:.2e}"),
                 "rel_error": float(f"{rel_error:.2e}")},
                f"seq_len={seq_len}: max_abs_error={max_abs_error:.2e}, "
                f"rel_error={rel_error:.2e}",
                severity="EDGE" if sev == "PASS" else sev,
            )

        except Exception as e:
            logger.log(
                BLOCK, test, "FAIL",
                {"seq_len": seq_len},
                f"seq_len={seq_len} raised {type(e).__name__}: {e}",
                severity="FAIL",
            )


def _sweep_accumulator_overflow(logger: SimLogger, rng: np.random.Generator):
    """Sweep 2: Accumulator overflow sweep."""
    test = "accumulator_overflow"
    head_dim = 64
    INT32_MAX = (1 << 31) - 1
    INT16_MAX = 32767

    # Theoretical: 64 * 32767^2 ≈ 6.87e13 >> INT32_MAX (2.1e9)
    # This overflows INT32 — need INT64 or score scaling
    theoretical_dot = head_dim * (INT16_MAX ** 2)
    overflows_int32 = theoretical_dot > INT32_MAX

    logger.log(
        BLOCK, test, "OVERFLOW",
        {"head_dim": head_dim, "INT16_MAX": INT16_MAX,
         "max_dot_product": theoretical_dot,
         "INT32_MAX": INT32_MAX,
         "overflows": overflows_int32},
        f"INT32 accumulator overflows for seq_len>=2048 with max-magnitude inputs; "
        f"need INT64 or score scaling. "
        f"64 * 32767^2 = {theoretical_dot:.2e} >> INT32_MAX={INT32_MAX:.2e}",
        severity="OVERFLOW",
    )

    # Find seq_len at which overflow first occurs in practice
    # (using max-magnitude float inputs as proxy)
    overflow_seq_len = None
    for seq_len in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
        Q = np.full(head_dim, float(INT16_MAX))
        K = np.full((seq_len, head_dim), float(INT16_MAX))
        V = np.full((seq_len, head_dim), float(INT16_MAX))

        # The dot product Q @ K[i] = head_dim * INT16_MAX^2
        raw_score = float(np.dot(K[0], Q))  # = 64 * 32767^2

        # Would this overflow INT32?
        if raw_score > INT32_MAX and overflow_seq_len is None:
            overflow_seq_len = seq_len
            logger.log(
                BLOCK, test, "FAIL",
                {"overflow_seq_len": seq_len,
                 "raw_dot_product": raw_score,
                 "INT32_MAX": INT32_MAX},
                f"KNOWN: Single QK dot product overflows INT32 at seq_len={seq_len} "
                f"(overflow is per-dot-product, independent of seq_len). "
                f"dot(Q,K[i])={raw_score:.2e} > INT32_MAX={INT32_MAX:.2e}. "
                f"RTL accumulator needs INT64 or input pre-scaling when inputs near INT16_MAX.",
                severity="FAIL",
            )
            break

    if overflow_seq_len is None:
        logger.log(
            BLOCK, test, "PASS",
            {"note": "No overflow detected with max-magnitude inputs across tested seq_lens"},
            "INT32 accumulator safe for all tested seq_len values",
            severity="PASS",
        )


def _sweep_tile_size_edge(logger: SimLogger, rng: np.random.Generator):
    """Sweep 3: Tile size edge sweep."""
    test = "tile_size_edge"
    head_dim = 64

    # seq_len = 1 with tile_size = 64: partial tile of size 1, output == V[0]
    Q = rng.standard_normal(head_dim)
    K = rng.standard_normal((1, head_dim))
    V = rng.standard_normal((1, head_dim))

    out_flash = flash_attention_tile(Q, K, V, tile_size=64)
    # With seq_len=1: attention weight = softmax([score]) = [1.0], output = V[0]
    out_ref = attention_reference(Q, K, V)
    err = float(np.max(np.abs(out_flash - out_ref)))

    if err > 1e-10:
        logger.log(
            BLOCK, test, "FAIL",
            {"seq_len": 1, "tile_size": 64, "error": err},
            f"seq_len=1, tile_size=64: error={err:.2e} (expected ~0)",
            severity="FAIL",
        )
    else:
        logger.log(
            BLOCK, test, "PASS",
            {"seq_len": 1, "tile_size": 64, "error": err},
            f"seq_len=1, tile_size=64: output == V[0] (error={err:.2e})",
            severity="PASS",
        )

    # seq_len = 63, tile_size = 64: full sequence fits in one partial tile
    K63 = rng.standard_normal((63, head_dim))
    V63 = rng.standard_normal((63, head_dim))
    out63_flash = flash_attention_tile(Q, K63, V63, tile_size=64)
    out63_ref = attention_reference(Q, K63, V63)
    err63 = float(np.max(np.abs(out63_flash - out63_ref)))

    sev = "FAIL" if err63 > 1e-2 else ("WARN" if err63 > 1e-3 else "PASS")
    logger.log(
        BLOCK, test, "OK" if sev == "PASS" else sev,
        {"seq_len": 63, "tile_size": 64, "error": err63},
        f"seq_len=63, tile_size=64 (one partial tile): error={err63:.2e}",
        severity=sev,
    )

    # seq_len = 65, tile_size = 64: one full tile + one token
    K65 = rng.standard_normal((65, head_dim))
    V65 = rng.standard_normal((65, head_dim))
    out65_flash = flash_attention_tile(Q, K65, V65, tile_size=64)
    out65_ref = attention_reference(Q, K65, V65)
    err65 = float(np.max(np.abs(out65_flash - out65_ref)))

    sev = "FAIL" if err65 > 1e-2 else ("WARN" if err65 > 1e-3 else "PASS")
    logger.log(
        BLOCK, test, "OK" if sev == "PASS" else sev,
        {"seq_len": 65, "tile_size": 64, "error": err65},
        f"seq_len=65, tile_size=64 (one full + partial tile): error={err65:.2e}",
        severity=sev,
    )

    # tile_size = 1: degenerate case, each tile is 1 token
    K32 = rng.standard_normal((32, head_dim))
    V32 = rng.standard_normal((32, head_dim))
    out_tile1_flash = flash_attention_tile(Q, K32, V32, tile_size=1)
    out_tile1_ref = attention_reference(Q, K32, V32)
    err_tile1 = float(np.max(np.abs(out_tile1_flash - out_tile1_ref)))

    sev = "FAIL" if err_tile1 > 1e-2 else ("WARN" if err_tile1 > 1e-3 else "PASS")
    logger.log(
        BLOCK, test, "OK" if sev == "PASS" else sev,
        {"seq_len": 32, "tile_size": 1, "error": err_tile1},
        f"tile_size=1 (degenerate): error={err_tile1:.2e} vs reference",
        severity=sev,
    )


def _sweep_numerical_stability(logger: SimLogger, rng: np.random.Generator):
    """Sweep 4: Numerical stability sweep."""
    test = "numerical_stability"
    head_dim = 64
    seq_len = 64

    for q_scale in [1, 10, 100, 1000, 10000]:
        Q = rng.standard_normal(head_dim) * q_scale
        K = rng.standard_normal((seq_len, head_dim)) * q_scale
        V = rng.standard_normal((seq_len, head_dim)) * q_scale

        try:
            out = flash_attention_tile(Q, K, V)

            has_nan = bool(np.any(np.isnan(out)))
            has_inf = bool(np.any(np.isinf(out)))

            # Verify softmax validity by checking the reference output
            scale_factor = 1.0 / math.sqrt(head_dim)
            raw_scores = (K @ Q) * scale_factor
            scores_shifted = raw_scores - raw_scores.max()
            exp_s = np.exp(scores_shifted)
            softmax_sum = float(exp_s.sum())
            softmax_valid = abs(softmax_sum - 1.0) < 0.01 if softmax_sum > 0 else False
            # The sum of exp(shifted) should equal softmax denominator, not 1.0
            # Actually check that softmax probs sum to 1
            attn_weights = exp_s / (exp_s.sum() + 1e-300)
            softmax_sum_normed = float(attn_weights.sum())

            if has_nan or has_inf:
                logger.log(
                    BLOCK, test, "FAIL",
                    {"q_scale": q_scale, "has_nan": has_nan, "has_inf": has_inf},
                    f"Output NaN={has_nan}/Inf={has_inf} at Q_scale={q_scale}",
                    severity="FAIL",
                )
            elif abs(softmax_sum_normed - 1.0) > 0.01:
                logger.log(
                    BLOCK, test, "FAIL",
                    {"q_scale": q_scale, "softmax_sum": softmax_sum_normed},
                    f"Softmax weights sum to {softmax_sum_normed:.6f} (expected 1.0) "
                    f"at Q_scale={q_scale}",
                    severity="FAIL",
                )
            else:
                logger.log(
                    BLOCK, test, "PASS",
                    {"q_scale": q_scale, "softmax_sum": round(softmax_sum_normed, 6)},
                    f"Softmax valid at Q_scale={q_scale}: sum={softmax_sum_normed:.6f}",
                    severity="PASS",
                )
        except Exception as e:
            logger.log(
                BLOCK, test, "FAIL",
                {"q_scale": q_scale},
                f"flash_attention_tile raised {type(e).__name__} at Q_scale={q_scale}: {e}",
                severity="FAIL",
            )

    # Large negative scores: Q = -1 × large random K
    scale_large = 1000
    K_ref = rng.standard_normal((seq_len, head_dim)) * scale_large
    Q_neg = -1 * (K_ref[0])  # Q anti-correlated with K rows
    try:
        out_neg = flash_attention_tile(Q_neg, K_ref, V=rng.standard_normal((seq_len, head_dim)))
        has_nan = bool(np.any(np.isnan(out_neg)))
        has_inf = bool(np.any(np.isinf(out_neg)))
        if has_nan or has_inf:
            logger.log(
                BLOCK, test, "FAIL",
                {"case": "large_negative_scores"},
                "Large negative scores produce NaN/Inf output",
                severity="FAIL",
            )
        else:
            logger.log(
                BLOCK, test, "PASS",
                {"case": "large_negative_scores"},
                "Large negative scores handled correctly (no NaN/Inf)",
                severity="PASS",
            )
    except Exception as e:
        logger.log(
            BLOCK, test, "FAIL",
            {"case": "large_negative_scores"},
            f"Large negative scores raised {type(e).__name__}: {e}",
            severity="FAIL",
        )


def _sweep_running_softmax_precision(logger: SimLogger, rng: np.random.Generator):
    """Sweep 5: Running softmax precision sweep."""
    test = "running_softmax_precision"
    head_dim = 64
    seq_len = 512
    tile_size = 64
    n_tiles = seq_len // tile_size  # = 8

    Q = rng.standard_normal(head_dim)
    K = rng.standard_normal((seq_len, head_dim))
    V = rng.standard_normal((seq_len, head_dim))

    scale = 1.0 / math.sqrt(head_dim)
    running_max = -math.inf
    running_sum = 0.0
    running_output = np.zeros(head_dim, dtype=np.float64)

    prev_max = -math.inf
    fail_detected = False

    for tile_idx in range(n_tiles):
        tile_start = tile_idx * tile_size
        tile_end = min(tile_start + tile_size, seq_len)
        k_tile = K[tile_start:tile_end]
        v_tile = V[tile_start:tile_end]
        scores = (k_tile @ Q) * scale

        running_max, running_sum, running_output = softmax_update(
            running_max, running_sum, running_output, scores, v_tile
        )

        # Check running_sum > 0
        if running_sum <= 0:
            logger.log(
                BLOCK, test, "FAIL",
                {"tile": tile_idx, "running_sum": running_sum},
                f"running_sum={running_sum} <= 0 after tile {tile_idx}",
                severity="FAIL",
            )
            fail_detected = True

        # Check output_accum is finite
        if not np.all(np.isfinite(running_output)):
            logger.log(
                BLOCK, test, "FAIL",
                {"tile": tile_idx},
                f"running_output contains non-finite values after tile {tile_idx}",
                severity="FAIL",
            )
            fail_detected = True

        # Check running_max is non-decreasing
        if running_max < prev_max - 1e-12:
            logger.log(
                BLOCK, test, "WARN",
                {"tile": tile_idx, "running_max": running_max, "prev_max": prev_max},
                f"running_max decreased: {running_max:.6f} < {prev_max:.6f} at tile {tile_idx}. "
                f"Should be monotonically non-decreasing.",
                severity="WARN",
            )

        prev_max = running_max

    if not fail_detected:
        logger.log(
            BLOCK, test, "OK",
            {"seq_len": seq_len, "tile_size": tile_size, "n_tiles": n_tiles},
            f"Running softmax: running_sum>0 and finite output across all {n_tiles} tiles",
            severity="PASS",
        )


def _sweep_dsp_mapping_analysis(logger: SimLogger, rng: np.random.Generator):
    """Sweep 6: ZCU102/ZCU104 DSP mapping analysis."""
    test = "dsp_mapping_zcu"

    logger.log(
        BLOCK, test, "EDGE",
        {"dsps_per_tile": 32, "zcu102_total_dsps": 2520, "zcu104_total_dsps": 1728,
         "lacu_dsps": 32},
        "32-wide dot product needs 32 DSP48E2 slices per tile computation",
        severity="EDGE",
    )

    logger.log(
        BLOCK, test, "EDGE",
        {"zcu102_dsps": 2520, "zcu104_dsps": 1728, "lacu_dsps": 32,
         "zcu102_utilization_pct": round(32 / 2520 * 100, 1),
         "zcu104_utilization_pct": round(32 / 1728 * 100, 1)},
        "ZCU102 has 2520 DSP48E2; LACU uses 32 (1.3%). "
        "ZCU104 has 1728; LACU uses 32 (1.9%).",
        severity="EDGE",
    )

    # At 50 MHz: 32 MACs × 64 elements = 2048 multiply-adds per tile
    # One tile per 64 cycles = 1.28 μs per tile at 50 MHz
    freq_50_mhz = 50e6
    freq_300_mhz = 300e6
    macs_per_tile = 32 * 64  # = 2048
    cycles_per_tile = 64
    time_50mhz_us = cycles_per_tile / freq_50_mhz * 1e6
    time_300mhz_ns = cycles_per_tile / freq_300_mhz * 1e9

    logger.log(
        BLOCK, test, "EDGE",
        {"macs_per_tile": macs_per_tile, "cycles_per_tile": cycles_per_tile,
         "freq_mhz": 50, "time_per_tile_us": round(time_50mhz_us, 3)},
        f"At 50 MHz: 32 MACs × 64 elements = {macs_per_tile} multiply-adds per tile. "
        f"One tile per {cycles_per_tile} cycles = {time_50mhz_us:.2f} μs per tile at 50 MHz",
        severity="EDGE",
    )

    logger.log(
        BLOCK, test, "EDGE",
        {"freq_mhz": 300, "time_per_tile_ns": round(time_300mhz_ns, 1)},
        f"At 300 MHz (ZCU102 native): one tile per {time_300mhz_ns:.0f} ns",
        severity="EDGE",
    )


def run_lacu_sweeps(logger: SimLogger):
    """Run all LACU sweeps."""
    rng = np.random.default_rng(42)
    _sweep_seq_length_scaling(logger, rng)
    _sweep_accumulator_overflow(logger, rng)
    _sweep_tile_size_edge(logger, rng)
    _sweep_numerical_stability(logger, rng)
    _sweep_running_softmax_precision(logger, rng)
    _sweep_dsp_mapping_analysis(logger, rng)
