"""
KV Cache Engine (KVE) golden model for LASSO.

Implements:
  - 5-stage Walsh-Hadamard Transform (WHT) butterfly on 32-element INT16 groups
  - Lloyd-Max codebook quantization (Q4 / Q8 mode selected by β vs β*)
  - Encode/decode for individual groups and full KV vectors
"""

import math
import numpy as np
from typing import List, Tuple

# Divisors matching hardware scale computation
_Q4_DIV = 7
_Q8_DIV = 127

# Number of WHT stages for group_size=32: log2(32) = 5
_DEFAULT_GROUP_SIZE = 32


def _wht_stages(x: np.ndarray, group_size: int) -> np.ndarray:
    """
    Generic WHT butterfly for a given group_size (must be power of 2).
    Uses integer arithmetic; no division — caller defers scaling.
    """
    n_stages = int(math.log2(group_size))
    x = x.copy().astype(np.int64)  # promote to avoid INT16 overflow mid-butterfly
    for stage in range(n_stages):
        stride = 1 << stage
        for i in range(group_size):
            j = i ^ stride
            if j > i:
                a = x[i]
                b = x[j]
                x[i] = a + b
                x[j] = a - b
    return x


def wht_butterfly(x: np.ndarray) -> np.ndarray:
    """
    5-stage Walsh-Hadamard Transform on 32-element INT16 array.

    Each stage k: for each pair (i, j) where j = i XOR (1 << k),
    replace (a[i], a[j]) with (a[i]+a[j], a[i]-a[j]).

    Returns INT64 array (intermediate values may exceed INT16 range).
    No division — scaling is deferred to dequantization.
    """
    x = np.asarray(x, dtype=np.int64)
    if len(x) != 32:
        raise ValueError(f"wht_butterfly expects 32 elements, got {len(x)}")
    return _wht_stages(x, 32)


def iwht_butterfly(x: np.ndarray) -> np.ndarray:
    """
    Inverse WHT for 32 elements.

    WHT is its own inverse up to a scale factor of N=32.
    So IWHT(x) = WHT(x) >> 5  (divide by 32 via arithmetic right-shift).

    Returns INT16-clamped array.
    """
    x = np.asarray(x, dtype=np.int64)
    if len(x) != 32:
        raise ValueError(f"iwht_butterfly expects 32 elements, got {len(x)}")
    result = _wht_stages(x, 32)
    result = result >> 5  # divide by 32
    return result.astype(np.int16)


def _wht_generic(x: np.ndarray, group_size: int) -> np.ndarray:
    """WHT for arbitrary power-of-2 group size."""
    if group_size & (group_size - 1) != 0:
        raise ValueError(f"group_size must be a power of 2, got {group_size}")
    x = np.asarray(x, dtype=np.int64)
    return _wht_stages(x, group_size)


def _iwht_generic(x: np.ndarray, group_size: int) -> np.ndarray:
    """Inverse WHT for arbitrary power-of-2 group size."""
    if group_size & (group_size - 1) != 0:
        raise ValueError(f"group_size must be a power of 2, got {group_size}")
    x = np.asarray(x, dtype=np.int64)
    result = _wht_stages(x, group_size)
    shift = int(math.log2(group_size))
    result = result >> shift
    return result.astype(np.int16)


def encode_group(
    group: np.ndarray,
    beta: float,
    beta_star: float,
    group_size: int = 32,
) -> Tuple[np.ndarray, int, str]:
    """
    Encode one group of INT16 elements using WHT + quantization.

    Mode selection:
      β < β*  → Q8 (more precise, 8-bit)
      β > β*  → Q4 (more compressed, 4-bit)
      β == β* → Q8 (tie-break: safe default)
      β == 0  → Q8 (reset/power-on default per PRD §5.7 case 8)

    Parameters
    ----------
    group : array-like, shape (group_size,), dtype INT16
        Raw KV cache elements.
    beta : float
        Current distortion metric β.
    beta_star : float
        Threshold β* from calibration.
    group_size : int
        Must be a power of 2 (default 32).

    Returns
    -------
    codes : np.ndarray, dtype int
        Quantization codes (INT4 in [0,15] for Q4, INT8 in [0,255] for Q8,
        or raw INT16 values for bypass).
    scale : int
        Scale factor stored as INT16.
    mode : str
        'Q4', 'Q8', or 'bypass'.
    """
    group = np.asarray(group, dtype=np.int16)
    if len(group) != group_size:
        raise ValueError(
            f"encode_group: group length {len(group)} != group_size {group_size}"
        )

    # Select quantization mode
    if beta > beta_star:
        mode = "Q4"
        divisor = _Q4_DIV
        n_levels = 16
    else:
        # β <= β* (includes β==0 reset default and tie-break)
        mode = "Q8"
        divisor = _Q8_DIV
        n_levels = 256

    # WHT rotation
    x_wht = _wht_generic(group, group_size)

    # Compute scale = max(|x̃|) / divisor, stored as INT16
    max_abs = int(np.max(np.abs(x_wht)))
    if max_abs == 0:
        scale = 0
        codes = np.zeros(group_size, dtype=np.int32)
        return codes, scale, mode

    scale = max_abs // divisor  # integer division to match HW
    if scale == 0:
        scale = 1  # minimum nonzero scale

    # Quantize: code = round(x̃ / scale) + offset so codes are unsigned
    # Q4: offset = 8 (center: [-7,7] → [0,15], midpoint = 8 - but we use
    #     direct rounding clamped to [0, n_levels-1])
    # Q8: offset = 128 (signed [-127,127] → [0,255])
    half = n_levels // 2
    x_norm = x_wht.astype(float) / scale
    raw_codes = np.round(x_norm).astype(np.int32)
    codes = np.clip(raw_codes + half, 0, n_levels - 1).astype(np.int32)

    # Clamp scale to INT16 range
    scale = int(np.clip(scale, 0, 32767))

    return codes, scale, mode


def decode_group(
    codes: np.ndarray,
    scale: int,
    mode: str,
    group_size: int = 32,
) -> np.ndarray:
    """
    Decode one group: dequantize codes then apply IWHT.

    Parameters
    ----------
    codes : np.ndarray
        Quantization codes from encode_group.
    scale : int
        Scale factor (INT16).
    mode : str
        'Q4', 'Q8', or 'bypass'.
    group_size : int
        Must match the value used during encoding.

    Returns
    -------
    x_hat : np.ndarray, dtype INT16
        Reconstructed group elements.
    """
    codes = np.asarray(codes, dtype=np.int32)
    if len(codes) != group_size:
        raise ValueError(
            f"decode_group: codes length {len(codes)} != group_size {group_size}"
        )

    if mode == "bypass":
        return codes.astype(np.int16)

    if mode == "Q4":
        divisor = _Q4_DIV
        n_levels = 16
    elif mode == "Q8":
        divisor = _Q8_DIV
        n_levels = 256
    else:
        raise ValueError(f"Unknown mode: {mode}")

    half = n_levels // 2

    # Dequantize: x̃ = (code - offset) * scale
    signed_codes = codes.astype(np.int64) - half
    x_wht_hat = signed_codes * int(scale)

    # Apply IWHT
    x_hat = _iwht_generic(x_wht_hat, group_size)
    return x_hat


def encode_kv_vector(
    kv_vec: np.ndarray,
    beta: float,
    beta_star: float,
    group_size: int = 32,
) -> List[Tuple[np.ndarray, int, str]]:
    """
    Encode a full KV vector by splitting into groups.

    Parameters
    ----------
    kv_vec : array-like, dtype INT16
        Full KV vector (length must be divisible by group_size).
    beta : float
        Current β value.
    beta_star : float
        β* threshold.
    group_size : int
        Group size (default 32).

    Returns
    -------
    list of (codes, scale, mode) tuples, one per group.
    """
    kv_vec = np.asarray(kv_vec, dtype=np.int16)
    n = len(kv_vec)
    if n % group_size != 0:
        raise ValueError(
            f"kv_vec length {n} is not divisible by group_size {group_size}"
        )

    n_groups = n // group_size
    result = []
    for g in range(n_groups):
        group = kv_vec[g * group_size : (g + 1) * group_size]
        codes, scale, mode = encode_group(group, beta, beta_star, group_size)
        result.append((codes, scale, mode))
    return result


def decode_kv_vector(
    encoded_groups: List[Tuple[np.ndarray, int, str]],
    group_size: int = 32,
) -> np.ndarray:
    """
    Decode all groups and concatenate.

    Parameters
    ----------
    encoded_groups : list of (codes, scale, mode)
    group_size : int

    Returns
    -------
    np.ndarray, dtype INT16
        Reconstructed full KV vector.
    """
    parts = []
    for codes, scale, mode in encoded_groups:
        x_hat = decode_group(codes, scale, mode, group_size)
        parts.append(x_hat)
    return np.concatenate(parts).astype(np.int16)


def compute_beta_star(gap_mean: float) -> float:
    """
    Compute β* from calibration data.

    β* = gap_mean / 0.267

    Per PRD calibration:
      SmolLM-1.7B: gap_mean=0.424 → β*=1.59
      SmolLM-135M: gap_mean=0.330 → β*=1.24
    """
    return gap_mean / 0.267
