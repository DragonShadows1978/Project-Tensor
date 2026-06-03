"""Quantization-table construction for APA-Quant attention.

The Lloyd-Max codebook and the per-head random rotations are one-time CPU work
(they depend only on (head_dim, bits) and a seed), so they stay in Python and
are cached. The C++/CUDA kernels consume the resulting tensors. This is a
numpy-only port of the relevant pieces of ``tensor_gpu_v2._quant`` so the
extension has no CuPy dependency.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

_CODEBOOK_CACHE: Dict[tuple, np.ndarray] = {}
_ROTATION_CACHE: Dict[tuple, np.ndarray] = {}


def _beta_coordinate_pdf(dimension: int, grid: np.ndarray) -> np.ndarray:
    if dimension < 2:
        raise ValueError("APA quantization requires head_dim >= 2")
    exponent = 0.5 * (dimension - 3)
    log_coeff = (
        math.lgamma(dimension / 2)
        - 0.5 * math.log(math.pi)
        - math.lgamma((dimension - 1) / 2)
    )
    values = np.maximum(0.0, 1.0 - grid * grid) ** exponent
    return np.exp(log_coeff) * values


def _weighted_quantiles(values: np.ndarray, weights: np.ndarray, q: np.ndarray) -> np.ndarray:
    cdf = np.cumsum(weights)
    total = cdf[-1]
    targets = np.clip(q, 0.0, 1.0) * total
    return np.interp(targets, cdf, values)


def build_lloyd_max_codebook(
    dimension: int,
    bits: int,
    *,
    grid_size: int = 16385,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> np.ndarray:
    """Construct the 1D Lloyd-Max codebook for beta-distributed sphere coords."""
    key = (dimension, bits, grid_size, max_iter, tol)
    cached = _CODEBOOK_CACHE.get(key)
    if cached is not None:
        return cached.copy()

    num_centroids = 1 << bits
    grid = np.linspace(-1.0, 1.0, grid_size, dtype=np.float64)
    pdf = _beta_coordinate_pdf(dimension, grid)
    dx = grid[1] - grid[0]
    weights = pdf * dx
    weights[0] *= 0.5
    weights[-1] *= 0.5

    quantiles = (np.arange(num_centroids, dtype=np.float64) + 0.5) / num_centroids
    centroids = _weighted_quantiles(grid, weights, quantiles)
    centroids = np.clip(np.sort(centroids), -1.0, 1.0)

    for _ in range(max_iter):
        old = centroids.copy()
        boundaries = 0.5 * (centroids[:-1] + centroids[1:])
        labels = np.searchsorted(boundaries, grid, side="left")
        for k in range(num_centroids):
            mask = labels == k
            weight_sum = weights[mask].sum()
            if weight_sum > 0:
                centroids[k] = float(np.sum(grid[mask] * weights[mask]) / weight_sum)
            elif k == 0:
                centroids[k] = -1.0
            elif k == num_centroids - 1:
                centroids[k] = 1.0
            else:
                centroids[k] = 0.5 * (centroids[k - 1] + centroids[k + 1])
        centroids = np.clip(np.sort(centroids), -1.0, 1.0)
        if np.max(np.abs(centroids - old)) < tol:
            break

    _CODEBOOK_CACHE[key] = centroids.copy()
    return centroids.astype(np.float32)


def random_orthogonal_matrix(dimension: int, seed: int) -> np.ndarray:
    """A Haar-ish orthogonal matrix via QR, with sign correction (matches ref)."""
    key = (dimension, seed)
    cached = _ROTATION_CACHE.get(key)
    if cached is not None:
        return cached.copy()
    rng = np.random.default_rng(seed)
    gaussian = rng.standard_normal((dimension, dimension))
    q, r = np.linalg.qr(gaussian)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    q = (q * signs).astype(np.float32)
    _ROTATION_CACHE[key] = q.copy()
    return q


def build_tables(head_dim: int, bits: int, num_heads: int, apa_rotation: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (rotations[H,D,D], codebook[2**bits], boundaries[2**bits-1]).

    Per-head seeds mirror the reference: ``h * 1337 + 42`` when rotation is on,
    else a shared seed of 0 for every head.
    """
    codebook = build_lloyd_max_codebook(head_dim, bits)
    boundaries = (0.5 * (codebook[:-1] + codebook[1:])).astype(np.float32)
    rotations = np.empty((num_heads, head_dim, head_dim), dtype=np.float32)
    for h in range(num_heads):
        seed_h = (h * 1337 + 42) if apa_rotation else 0
        rotations[h] = random_orthogonal_matrix(head_dim, seed_h)
    return rotations, codebook, boundaries
