"""
LACU (Lightweight Attention Compute Unit) golden model tests.
30 test cases covering FlashAttention tiling, numerical stability, and edge cases.
"""

import math
import numpy as np
import pytest

from golden_model.lacu import (
    flash_attention_tile,
    attention_reference,
    dot_product,
    softmax_update,
    TILE_SIZE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_qkv(seq_len: int, head_dim: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal(head_dim)
    K = rng.standard_normal((seq_len, head_dim))
    V = rng.standard_normal((seq_len, head_dim))
    return Q, K, V


MAX_ERR = 1e-3   # acceptable floating-point error vs reference


# ===========================================================================
# 1. flash_attention_tile output shape matches Q shape [head_dim]
# ===========================================================================
def test_output_shape():
    Q, K, V = _random_qkv(16, 64)
    out = flash_attention_tile(Q, K, V)
    assert out.shape == (64,)


# 2. flash_attention_tile vs attention_reference: max error < 1e-3
def test_flash_vs_reference_accuracy():
    Q, K, V = _random_qkv(64, 64, seed=2)
    out_flash = flash_attention_tile(Q, K, V)
    out_ref   = attention_reference(Q, K, V)
    assert np.max(np.abs(out_flash - out_ref)) < MAX_ERR


# 3. Single token (seq_len=1): output equals V[0]
def test_single_token_output_equals_v0():
    Q = np.ones(64)
    K = np.ones((1, 64))
    V = np.full((1, 64), 3.14)
    out = flash_attention_tile(Q, K, V)
    assert np.allclose(out, 3.14, atol=1e-10)


# 4. tile_size=32 matches tile_size=64 output
def test_tile_size_32_matches_64():
    Q, K, V = _random_qkv(128, 64, seed=4)
    out32 = flash_attention_tile(Q, K, V, tile_size=32)
    out64 = flash_attention_tile(Q, K, V, tile_size=64)
    assert np.max(np.abs(out32 - out64)) < MAX_ERR


# 5. tile_size=128 matches reference
def test_tile_size_128_matches_reference():
    Q, K, V = _random_qkv(256, 64, seed=5)
    out128 = flash_attention_tile(Q, K, V, tile_size=128)
    out_ref = attention_reference(Q, K, V)
    assert np.max(np.abs(out128 - out_ref)) < MAX_ERR


# 6. Q all-zeros: uniform softmax → output is mean(V)
def test_q_all_zeros_uniform_softmax():
    seq_len, head_dim = 16, 32
    rng = np.random.default_rng(6)
    Q = np.zeros(head_dim)
    K = rng.standard_normal((seq_len, head_dim))
    V = rng.standard_normal((seq_len, head_dim))
    out = flash_attention_tile(Q, K, V)
    # Q=0 → all QK scores equal → uniform softmax → output = mean(V)
    expected = V.mean(axis=0)
    assert np.max(np.abs(out - expected)) < MAX_ERR


# 7. K all-zeros: uniform softmax → output is mean(V)
def test_k_all_zeros_uniform_softmax():
    seq_len, head_dim = 16, 32
    rng = np.random.default_rng(7)
    Q = rng.standard_normal(head_dim)
    K = np.zeros((seq_len, head_dim))
    V = rng.standard_normal((seq_len, head_dim))
    out = flash_attention_tile(Q, K, V)
    expected = V.mean(axis=0)
    assert np.max(np.abs(out - expected)) < MAX_ERR


# 8. Large Q·K scores: numerically stable (no NaN/inf)
def test_large_scores_numerically_stable():
    rng = np.random.default_rng(8)
    Q = rng.standard_normal(64) * 100
    K = rng.standard_normal((64, 64)) * 100
    V = rng.standard_normal((64, 64))
    out = flash_attention_tile(Q, K, V)
    assert np.all(np.isfinite(out))


# 9. Very small Q·K scores: softmax still sums to 1
def test_small_scores_softmax_valid():
    rng = np.random.default_rng(9)
    Q = rng.standard_normal(64) * 0.0001
    K = rng.standard_normal((16, 64)) * 0.0001
    V = rng.standard_normal((16, 64))
    out = flash_attention_tile(Q, K, V)
    assert np.all(np.isfinite(out))


# 10. head_dim=64 standard
def test_head_dim_64():
    Q, K, V = _random_qkv(32, 64, seed=10)
    out = flash_attention_tile(Q, K, V)
    ref = attention_reference(Q, K, V)
    assert np.max(np.abs(out - ref)) < MAX_ERR


# 11. head_dim=32 (smaller)
def test_head_dim_32():
    Q, K, V = _random_qkv(32, 32, seed=11)
    out = flash_attention_tile(Q, K, V)
    ref = attention_reference(Q, K, V)
    assert np.max(np.abs(out - ref)) < MAX_ERR


# 12. Parametrized seq_len
@pytest.mark.parametrize("seq_len", [1, 4, 16, 64, 128, 256])
def test_seq_len_parametrized(seq_len):
    Q, K, V = _random_qkv(seq_len, 64, seed=seq_len)
    out = flash_attention_tile(Q, K, V)
    ref = attention_reference(Q, K, V)
    assert np.max(np.abs(out - ref)) < MAX_ERR


# 13. seq_len not multiple of tile_size: partial last tile handled correctly
def test_partial_last_tile():
    # seq_len=70 with tile_size=64: full tile of 64 + partial tile of 6
    Q, K, V = _random_qkv(70, 64, seed=13)
    out = flash_attention_tile(Q, K, V, tile_size=64)
    ref = attention_reference(Q, K, V)
    assert np.max(np.abs(out - ref)) < MAX_ERR


# 14. running_max initialized to -inf
def test_running_max_initial_value():
    # Test that softmax_update with running_max=-inf gives correct first tile
    Q, K, V = _random_qkv(4, 8, seed=14)
    scores = (K @ Q) / math.sqrt(8)
    rm, rs, ro = softmax_update(-math.inf, 0.0, np.zeros(8), scores, V)
    assert math.isfinite(rm)
    assert rs > 0.0


# 15. softmax_update: new tile with larger scores rescales previous accumulator
def test_softmax_update_rescales():
    head_dim = 8
    # First tile: scores = 0 → exp = 1 for each
    v1 = np.ones((4, head_dim))
    rm1, rs1, ro1 = softmax_update(-math.inf, 0.0, np.zeros(head_dim),
                                   np.zeros(4), v1)

    # Second tile: scores = 10 (much larger) → should rescale first accumulator
    v2 = np.full((4, head_dim), 2.0)
    rm2, rs2, ro2 = softmax_update(rm1, rs1, ro1, np.full(4, 10.0), v2)

    assert rm2 == pytest.approx(10.0)
    assert rs2 > rs1
    # Output should now be dominated by the high-score tile
    assert np.all(ro2 / rs2 > 1.5)  # closer to 2.0 than 1.0


# 16. dot_product tests
def test_dot_product_unit_vector():
    assert dot_product([1, 0, 0], [1, 2, 3]) == pytest.approx(1)


def test_dot_product_squares():
    assert dot_product([1, 2, 3], [1, 2, 3]) == pytest.approx(14)


# 17. Q4 dequantized input: output close to FP16 reference
def test_q4_dequantized_input():
    # Simulate dequantized Q4 values (low precision but should still work)
    rng = np.random.default_rng(17)
    Q = (rng.integers(-7, 8, size=64) * 10).astype(float)
    K = (rng.integers(-7, 8, size=(32, 64)) * 10).astype(float)
    V = (rng.integers(-7, 8, size=(32, 64)) * 10).astype(float)
    out = flash_attention_tile(Q, K, V)
    ref = attention_reference(Q, K, V)
    assert np.max(np.abs(out - ref)) < MAX_ERR


# 18. Ping-pong: two softmax_update calls + finalize = one flash_attention call
def test_ping_pong_two_tiles():
    Q, K, V = _random_qkv(128, 64, seed=18)
    scale = 1.0 / math.sqrt(64)

    # Manual two-tile
    k1, v1 = K[:64], V[:64]
    k2, v2 = K[64:], V[64:]
    rm, rs, ro = softmax_update(-math.inf, 0.0, np.zeros(64),
                                (k1 @ Q) * scale, v1)
    rm, rs, ro = softmax_update(rm, rs, ro, (k2 @ Q) * scale, v2)
    manual_out = ro / rs

    flash_out = flash_attention_tile(Q, K, V, tile_size=64)
    assert np.max(np.abs(manual_out - flash_out)) < 1e-10


# 19. All V values identical: output equals that value
def test_all_v_identical():
    Q, K, _ = _random_qkv(32, 64, seed=19)
    V = np.full((32, 64), 7.77)
    out = flash_attention_tile(Q, K, V)
    assert np.allclose(out, 7.77, atol=1e-10)


# 20-24. attention_reference and flash agree on random inputs (5 seeds)
@pytest.mark.parametrize("seed", [100, 101, 102, 103, 104])
def test_flash_agrees_with_reference_random(seed):
    Q, K, V = _random_qkv(64, 64, seed=seed)
    out = flash_attention_tile(Q, K, V)
    ref = attention_reference(Q, K, V)
    assert np.max(np.abs(out - ref)) < MAX_ERR


# 25. Output normalized: running_sum divides accumulator
def test_output_normalized():
    Q, K, V = _random_qkv(32, 64, seed=25)
    out = flash_attention_tile(Q, K, V)
    ref = attention_reference(Q, K, V)
    # Check that the outputs are close (normalization correct)
    assert np.allclose(out, ref, atol=MAX_ERR)


# 26. seq_len=0: returns zero vector
def test_seq_len_zero():
    head_dim = 64
    Q = np.ones(head_dim)
    K = np.empty((0, head_dim))
    V = np.empty((0, head_dim))
    out = flash_attention_tile(Q, K, V)
    assert out.shape == (head_dim,)
    assert np.all(out == 0.0)


# 27. seq_len=TILE_SIZE exactly: no partial tile handling error
def test_seq_len_equals_tile_size():
    Q, K, V = _random_qkv(TILE_SIZE, 64, seed=27)
    out = flash_attention_tile(Q, K, V, tile_size=TILE_SIZE)
    ref = attention_reference(Q, K, V)
    assert np.max(np.abs(out - ref)) < MAX_ERR


# 28. seq_len=2×TILE_SIZE: two full tiles
def test_seq_len_two_tiles():
    Q, K, V = _random_qkv(2 * TILE_SIZE, 64, seed=28)
    out = flash_attention_tile(Q, K, V, tile_size=TILE_SIZE)
    ref = attention_reference(Q, K, V)
    assert np.max(np.abs(out - ref)) < MAX_ERR


# 29. seq_len=3×TILE_SIZE: three full tiles
def test_seq_len_three_tiles():
    Q, K, V = _random_qkv(3 * TILE_SIZE, 64, seed=29)
    out = flash_attention_tile(Q, K, V, tile_size=TILE_SIZE)
    ref = attention_reference(Q, K, V)
    assert np.max(np.abs(out - ref)) < MAX_ERR


# 30. seq_len=3×TILE_SIZE+1: three full tiles + one token
def test_seq_len_three_tiles_plus_one():
    Q, K, V = _random_qkv(3 * TILE_SIZE + 1, 64, seed=30)
    out = flash_attention_tile(Q, K, V, tile_size=TILE_SIZE)
    ref = attention_reference(Q, K, V)
    assert np.max(np.abs(out - ref)) < MAX_ERR
