"""
Lloyd-Max codebook generator for LASSO KV Cache Compression Coprocessor.

Generates Q4 (16-level) and Q8 (256-level) codebooks for Gaussian-distributed
inputs, matching the hardware codebook lookup table in the KVE block.
"""

import os
import struct
import numpy as np
from scipy.stats import norm


def generate_lloydmax_codebook(n_levels: int, n_iter: int = 100):
    """
    Run Lloyd-Max iteration on Gaussian samples to produce an optimal
    scalar quantizer for a standard normal distribution.

    Parameters
    ----------
    n_levels : int
        Number of quantization levels (16 for Q4, 256 for Q8).
    n_iter : int
        Number of Lloyd-Max iterations.

    Returns
    -------
    boundaries : np.ndarray, shape (n_levels - 1,)
        Decision boundaries between levels.
    centroids : np.ndarray, shape (n_levels,)
        Reconstruction (centroid) values for each level.
    """
    n_boundaries = n_levels - 1

    # Sample Gaussian for initialization
    rng = np.random.default_rng(42)
    samples = rng.standard_normal(200_000)
    samples.sort()

    # Uniform initialization of boundaries in [-3, 3]
    boundaries = np.linspace(-3.0, 3.0, n_boundaries + 2)[1:-1]

    for _ in range(n_iter):
        # Compute centroids as conditional means of the Gaussian
        # Using analytical form: E[X | b_{k-1} < X <= b_k] for N(0,1)
        # = (phi(b_{k-1}) - phi(b_k)) / (Phi(b_k) - Phi(b_{k-1}))
        # where phi = pdf, Phi = cdf

        left_bounds = np.concatenate([[-np.inf], boundaries])
        right_bounds = np.concatenate([boundaries, [np.inf]])

        pdf_left = norm.pdf(left_bounds)
        pdf_right = norm.pdf(right_bounds)
        cdf_left = norm.cdf(left_bounds)
        cdf_right = norm.cdf(right_bounds)

        region_mass = cdf_right - cdf_left
        # Avoid divide-by-zero for empty regions
        region_mass = np.where(region_mass < 1e-15, 1e-15, region_mass)

        centroids = (pdf_left - pdf_right) / region_mass

        # Update boundaries as midpoints of adjacent centroids
        boundaries = 0.5 * (centroids[:-1] + centroids[1:])

    return boundaries, centroids


def quantize_lloydmax(x: np.ndarray, codebook: np.ndarray, scale: float) -> np.ndarray:
    """
    Quantize input array using a Lloyd-Max codebook.

    Parameters
    ----------
    x : np.ndarray
        Input values (after WHT, before scale multiply). Float or int.
    codebook : np.ndarray
        Codebook centroids (in normalized space, i.e. after dividing by scale).
    scale : float
        Scale factor: scale = max(|x|) / divisor.

    Returns
    -------
    codes : np.ndarray, dtype int
        Quantization indices (0-based).
    """
    if scale == 0.0:
        return np.zeros(len(x), dtype=np.int32)

    x_norm = np.asarray(x, dtype=float) / scale
    # For each element, find nearest centroid
    x_col = x_norm.reshape(-1, 1)
    cb_row = codebook.reshape(1, -1)
    distances = np.abs(x_col - cb_row)
    codes = np.argmin(distances, axis=1)
    return codes.astype(np.int32)


def dequantize_lloydmax(codes: np.ndarray, scale: float, n_bits: int) -> np.ndarray:
    """
    Dequantize codes using the stored codebook centroids.

    Parameters
    ----------
    codes : np.ndarray, dtype int
        Quantization indices.
    scale : float
        Scale factor used during quantization.
    n_bits : int
        Bit depth: 4 for Q4, 8 for Q8.

    Returns
    -------
    x_hat : np.ndarray, dtype float
        Reconstructed values (not yet integer-rounded).
    """
    n_levels = 2 ** n_bits
    _, centroids = generate_lloydmax_codebook(n_levels)
    codes = np.asarray(codes, dtype=np.int32)
    x_hat = centroids[codes] * scale
    return x_hat


def _save_q4_codebook_hex(out_path: str):
    """
    Generate Q4 Lloyd-Max codebook and save to hex file.

    Format: 16 INT8 centroids (scaled to [-128, 127]) + 15 INT8 boundaries
    = 31 bytes, padded with one zero byte to 32 bytes.
    Each byte is written as two hex chars, one per line.
    """
    boundaries, centroids = generate_lloydmax_codebook(16)

    # Scale centroids to INT8 range [-128, 127]
    # Max centroid magnitude from Lloyd-Max on N(0,1) is ~2.4
    cb_scale = 127.0 / max(abs(centroids.max()), abs(centroids.min()), 1e-9)
    centroids_int8 = np.clip(np.round(centroids * cb_scale), -128, 127).astype(np.int8)
    boundaries_int8 = np.clip(np.round(boundaries * cb_scale), -128, 127).astype(np.int8)

    data = bytearray()
    for c in centroids_int8:
        data.append(c & 0xFF)
    for b in boundaries_int8:
        data.append(b & 0xFF)
    # Pad to 32 bytes
    data.append(0x00)

    assert len(data) == 32

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for byte in data:
            f.write(f"{byte:02x}\n")


if __name__ == "__main__":
    import pathlib
    repo_root = pathlib.Path(__file__).parent.parent
    out = repo_root / "fixtures" / "codebook_q4.hex"
    _save_q4_codebook_hex(str(out))
    print(f"Saved Q4 codebook to {out}")
