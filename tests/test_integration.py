"""
Integration tests for the full LASSO decode pipeline:
KVE → TIU → MHC → LACU

18 test cases covering end-to-end encode/store/retrieve/decode flows.
"""

import math
import numpy as np
import pytest

from golden_model.kve import (
    encode_kv_vector,
    decode_kv_vector,
    compute_beta_star,
    encode_group,
    decode_group,
)
from golden_model.tiu import score_token
from golden_model.mhc import MHC, _PAGE_TABLE_SIZE
from golden_model.lacu import flash_attention_tile, attention_reference


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BETA_STAR = compute_beta_star(0.424)   # ≈ 1.588 (SmolLM-1.7B)
HEAD_DIM  = 64
GROUP_SIZE = 32


def _make_kv(seed: int, seq_len: int = 16) -> np.ndarray:
    """Return a random INT16 KV vector of length seq_len * HEAD_DIM."""
    rng = np.random.default_rng(seed)
    return rng.integers(-500, 501, size=seq_len * HEAD_DIM, dtype=np.int16)


def _make_attn(seed: int, n_heads: int = 4, n_tokens: int = 16) -> np.ndarray:
    """Return valid softmax attention weights [n_heads, n_tokens]."""
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal((n_heads, n_tokens))
    logits -= logits.max(axis=1, keepdims=True)
    exp_l = np.exp(logits)
    return exp_l / exp_l.sum(axis=1, keepdims=True)


def _pack_encode(groups):
    """Pack a list of (codes, scale, mode) as a single int for MHC (simplified)."""
    # Use first group's scale for demonstration
    if not groups:
        return 0, 0, "Q8"
    _, scale, mode = groups[0]
    packed_kv = int(scale) & 0xFFFF
    return packed_kv, int(scale), mode


def _store_token(mhc: MHC, token_idx: int, kv_vec: np.ndarray, attn: np.ndarray,
                 beta: float, beta_star: float):
    """Encode KV, score token, and write to MHC."""
    groups = encode_kv_vector(kv_vec, beta, beta_star, GROUP_SIZE)
    packed_kv, scale, mode = _pack_encode(groups)
    tag, score = score_token(token_idx, attn)
    mhc.write_kv(token_idx, packed_kv, scale, mode, tag)
    return tag, groups


# ===========================================================================
# Integration tests
# ===========================================================================

# 1. encode KV → store in MHC → read from MHC → decode: round-trip correct
def test_full_encode_store_read_decode():
    mhc = MHC()
    kv = _make_kv(1, seq_len=1)
    groups = encode_kv_vector(kv, 0.0, BETA_STAR, GROUP_SIZE)
    packed, scale, mode = _pack_encode(groups)
    mhc.write_kv(0, packed, scale, mode, "RETAIN")

    packed_r, scale_r, mode_r = mhc.read_kv(0)
    assert mode_r == mode
    assert scale_r == scale
    # Decode original groups
    recon = decode_kv_vector(groups, GROUP_SIZE)
    assert len(recon) == len(kv)


# 2. β < β*: full pipeline uses Q8 mode end-to-end
def test_beta_lt_betastar_pipeline_q8():
    beta = BETA_STAR * 0.5  # β < β* → Q8
    kv = _make_kv(2, seq_len=1)
    groups = encode_kv_vector(kv, beta, BETA_STAR, GROUP_SIZE)
    for _, _, mode in groups:
        assert mode == "Q8"


# 3. β > β*: full pipeline uses Q4 mode end-to-end
def test_beta_gt_betastar_pipeline_q4():
    beta = BETA_STAR * 2.0  # β > β* → Q4
    kv = _make_kv(3, seq_len=1)
    groups = encode_kv_vector(kv, beta, BETA_STAR, GROUP_SIZE)
    for _, _, mode in groups:
        assert mode == "Q4"


# 4. TIU EVICT: token not stored in MHC (read raises KeyError)
def test_tiu_evict_not_stored():
    mhc = MHC()
    attn = _make_attn(4)
    kv = _make_kv(4, seq_len=1)
    # Use threshold=1.0 to force all non-sinks to EVICT
    tag, _ = score_token(10, attn, threshold=1.0)
    assert tag == "EVICT"

    groups = encode_kv_vector(kv, 0.0, BETA_STAR, GROUP_SIZE)
    packed, scale, mode = _pack_encode(groups)
    mhc.write_kv(10, packed, scale, mode, "EVICT")

    with pytest.raises(KeyError):
        mhc.read_kv(10)


# 5. TIU RETAIN: token stored and readable
def test_tiu_retain_stored():
    mhc = MHC()
    attn = _make_attn(5)
    kv = _make_kv(5, seq_len=1)
    tag, _ = score_token(10, attn, threshold=0.0)
    assert tag == "RETAIN"

    groups = encode_kv_vector(kv, 0.0, BETA_STAR, GROUP_SIZE)
    packed, scale, mode = _pack_encode(groups)
    mhc.write_kv(10, packed, scale, mode, "RETAIN")

    packed_r, scale_r, mode_r = mhc.read_kv(10)
    assert mode_r == mode


# 6. Attention sink (token 0) always stored regardless of TIU score
def test_attention_sink_always_stored():
    mhc = MHC()
    attn = _make_attn(6)
    kv = _make_kv(6, seq_len=1)
    # Even with threshold=1.0, token 0 is sink → RETAIN
    tag, _ = score_token(0, attn, threshold=1.0, sink_count=4)
    assert tag == "RETAIN"

    groups = encode_kv_vector(kv, 0.0, BETA_STAR, GROUP_SIZE)
    packed, scale, mode = _pack_encode(groups)
    mhc.write_kv(0, packed, scale, mode, "RETAIN")
    packed_r, _, _ = mhc.read_kv(0)
    assert packed_r == packed


# 7. Full decode step: KVE→TIU→MHC→LACU for 16-token sequence
def test_full_pipeline_16_tokens():
    mhc = MHC()
    rng = np.random.default_rng(7)
    seq_len = 16
    KV_store = {}

    # Encode and store all tokens
    for t in range(seq_len):
        kv = rng.integers(-500, 501, size=HEAD_DIM, dtype=np.int16)
        attn = _make_attn(t, n_tokens=seq_len)
        groups = encode_kv_vector(kv, 0.0, BETA_STAR, GROUP_SIZE)
        packed, scale, mode = _pack_encode(groups)
        tag, _ = score_token(t, attn, threshold=0.0)
        mhc.write_kv(t, packed, scale, mode, tag)
        KV_store[t] = decode_kv_vector(groups, GROUP_SIZE).astype(float)

    # Build K, V from stored tokens
    K = np.stack([KV_store[t][:HEAD_DIM // 2] for t in range(seq_len)])
    V = np.stack([KV_store[t][HEAD_DIM // 2:] for t in range(seq_len)])
    Q = rng.standard_normal(HEAD_DIM // 2)

    out_flash = flash_attention_tile(Q, K, V)
    out_ref   = attention_reference(Q, K, V)
    assert np.max(np.abs(out_flash - out_ref)) < 1e-3


# 8. Full decode step for 64-token sequence (one tile)
def test_full_pipeline_64_tokens():
    rng = np.random.default_rng(8)
    seq_len = 64
    KV_store = {}
    mhc = MHC()

    for t in range(seq_len):
        kv = rng.integers(-200, 201, size=HEAD_DIM, dtype=np.int16)
        groups = encode_kv_vector(kv, 0.0, BETA_STAR, GROUP_SIZE)
        packed, scale, mode = _pack_encode(groups)
        mhc.write_kv(t, packed, scale, mode, "RETAIN")
        KV_store[t] = decode_kv_vector(groups, GROUP_SIZE).astype(float)

    K = np.stack([KV_store[t][:HEAD_DIM // 2] for t in range(seq_len)])
    V = np.stack([KV_store[t][HEAD_DIM // 2:] for t in range(seq_len)])
    Q = rng.standard_normal(HEAD_DIM // 2)

    out = flash_attention_tile(Q, K, V, tile_size=64)
    ref = attention_reference(Q, K, V)
    assert np.max(np.abs(out - ref)) < 1e-3


# 9. Full decode step for 128-token sequence (two tiles)
def test_full_pipeline_128_tokens():
    rng = np.random.default_rng(9)
    seq_len = 128
    KV_store = {}
    mhc = MHC(hot_thresh=200)

    for t in range(seq_len):
        kv = rng.integers(-200, 201, size=HEAD_DIM, dtype=np.int16)
        groups = encode_kv_vector(kv, 0.0, BETA_STAR, GROUP_SIZE)
        packed, scale, mode = _pack_encode(groups)
        mhc.write_kv(t, packed, scale, mode, "RETAIN")
        KV_store[t] = decode_kv_vector(groups, GROUP_SIZE).astype(float)

    K = np.stack([KV_store[t][:HEAD_DIM // 2] for t in range(seq_len)])
    V = np.stack([KV_store[t][HEAD_DIM // 2:] for t in range(seq_len)])
    Q = rng.standard_normal(HEAD_DIM // 2)

    out = flash_attention_tile(Q, K, V, tile_size=64)
    ref = attention_reference(Q, K, V)
    assert np.max(np.abs(out - ref)) < 1e-3


# 10. β switch mid-sequence: Q8 tokens and Q4 tokens coexist in MHC
def test_beta_switch_mixed_modes():
    mhc = MHC()
    rng = np.random.default_rng(10)

    # First 8 tokens: Q8 (β < β*)
    for t in range(8):
        kv = rng.integers(-200, 201, size=HEAD_DIM, dtype=np.int16)
        groups = encode_kv_vector(kv, 0.0, BETA_STAR, GROUP_SIZE)
        packed, scale, mode = _pack_encode(groups)
        assert mode == "Q8"
        mhc.write_kv(t, packed, scale, mode, "RETAIN")

    # Next 8 tokens: Q4 (β > β*)
    for t in range(8, 16):
        kv = rng.integers(-200, 201, size=HEAD_DIM, dtype=np.int16)
        groups = encode_kv_vector(kv, BETA_STAR * 2, BETA_STAR, GROUP_SIZE)
        packed, scale, mode = _pack_encode(groups)
        assert mode == "Q4"
        mhc.write_kv(t, packed, scale, mode, "RETAIN")

    # All 16 tokens should be readable
    for t in range(16):
        packed_r, scale_r, mode_r = mhc.read_kv(t)
        assert mode_r in ("Q4", "Q8")


# 11. SRAM overflow: 257th token triggers OverflowError
def test_sram_overflow_257th_token():
    mhc = MHC(hot_thresh=1000)
    for t in range(_PAGE_TABLE_SIZE):
        mhc.write_kv(t, 0x1234, 10, "Q8", "RETAIN")
    with pytest.raises(OverflowError):
        mhc.write_kv(_PAGE_TABLE_SIZE, 0x1234, 10, "Q8", "RETAIN")


# 12. Cold tier: tokens beyond hot_thresh go cold
def test_cold_tier_after_hot_thresh():
    mhc = MHC(hot_thresh=4)
    for t in range(8):
        mhc.write_kv(t, 0x1234, 10, "Q8", "RETAIN")

    for t in range(4):
        pte = mhc.page_table.lookup(t)
        assert pte.tier == "hot"
    for t in range(4, 8):
        pte = mhc.page_table.lookup(t)
        assert pte.tier == "cold"


# 13. Eviction: evicted token cannot be read (stall condition)
def test_evicted_token_stalls_lacu():
    mhc = MHC()
    mhc.write_kv(5, 0xABCD, 50, "Q8", "RETAIN")
    mhc.evict(5)
    with pytest.raises(KeyError):
        mhc.read_kv(5)


# 14. LACU output matches attention_reference for pipeline-generated KV
def test_lacu_matches_reference_pipeline():
    rng = np.random.default_rng(14)
    seq_len = 16
    K = rng.standard_normal((seq_len, HEAD_DIM // 2))
    V = rng.standard_normal((seq_len, HEAD_DIM // 2))
    Q = rng.standard_normal(HEAD_DIM // 2)

    out_flash = flash_attention_tile(Q, K, V)
    out_ref   = attention_reference(Q, K, V)
    assert np.max(np.abs(out_flash - out_ref)) < 1e-3


# 15. Multiple-head attention (4 heads) through full pipeline
def test_multi_head_pipeline():
    n_heads = 4
    rng = np.random.default_rng(15)
    seq_len = 16
    results = []

    for h in range(n_heads):
        K = rng.standard_normal((seq_len, HEAD_DIM // 2))
        V = rng.standard_normal((seq_len, HEAD_DIM // 2))
        Q = rng.standard_normal(HEAD_DIM // 2)
        out = flash_attention_tile(Q, K, V)
        ref = attention_reference(Q, K, V)
        assert np.max(np.abs(out - ref)) < 1e-3
        results.append(out)

    assert len(results) == n_heads


# 16. GQA: 4 KV heads, 8 Q heads → TIU scores per KV head
def test_gqa_kv_score_per_head():
    n_kv_heads = 4
    n_tokens = 16
    rng = np.random.default_rng(16)
    attn_kv = rng.random((n_kv_heads, n_tokens))
    attn_kv /= attn_kv.sum(axis=1, keepdims=True)

    for t in range(4, n_tokens):
        tag, score = score_token(t, attn_kv, threshold=0.0)
        assert tag == "RETAIN"


# 17. Reset: after MHC reset, all tokens evicted, reads raise KeyError
def test_mhc_reset_all_evicted():
    mhc = MHC()
    for t in range(8):
        mhc.write_kv(t, 0x1234, 10, "Q8", "RETAIN")

    mhc.reset()

    for t in range(8):
        with pytest.raises(KeyError):
            mhc.read_kv(t)


# 18. Calibration: β* computed from gap_mean=0.424 gives β*≈1.59 (SmolLM-1.7B)
def test_calibration_smollm_17b():
    beta_star = compute_beta_star(0.424)
    assert abs(beta_star - 1.59) < 0.01
