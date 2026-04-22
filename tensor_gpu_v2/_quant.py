"""TurboQuant vector quantization utilities.

Implements paper-inspired online vector quantization primitives for:
- MSE-oriented reconstruction (`TurboQuantMSE`)
- unbiased inner-product estimation (`TurboQuantProd`)

The implementation follows the TurboQuant paper's unit-sphere formulation and
extends it to arbitrary vectors by storing per-vector L2 norms separately, as
explicitly suggested in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional, Tuple

import cupy as cp
import numpy as np

from ._core import Tensor


_CODEBOOK_CACHE = {}


def _is_tensor(obj: Any) -> bool:
    return isinstance(obj, Tensor)


def _to_array(data: Any):
    if _is_tensor(data):
        return data.data
    return data


def _xp_of(data) -> Any:
    if isinstance(data, cp.ndarray):
        return cp
    return np


def _ensure_last_dim(array, dim: int) -> None:
    if array.ndim == 0 or array.shape[-1] != dim:
        raise ValueError(
            f"expected input with trailing dimension {dim}, got shape {array.shape}"
        )


def _flatten_vectors(array, dim: int):
    _ensure_last_dim(array, dim)
    return array.reshape(-1, dim), array.shape[:-1]


def _restore_shape(array, prefix_shape: Tuple[int, ...]) -> Any:
    return array.reshape(prefix_shape + (array.shape[-1],))


def _quant_index_dtype(bits: int):
    if bits <= 8:
        return np.uint8
    if bits <= 16:
        return np.uint16
    return np.int32


def _sign_pm_one(xp, array):
    ones = xp.ones_like(array, dtype=xp.int8)
    neg_ones = -ones
    return xp.where(array >= 0, ones, neg_ones)


def _beta_coordinate_pdf(dimension: int, grid: np.ndarray) -> np.ndarray:
    if dimension < 2:
        raise ValueError("TurboQuant requires dimension >= 2")
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


def _build_lloyd_max_codebook(
    dimension: int,
    bits: int,
    *,
    grid_size: int = 16385,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> np.ndarray:
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
    return centroids


def _random_orthogonal_matrix(dimension: int, seed: Optional[int]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gaussian = rng.standard_normal((dimension, dimension))
    q, r = np.linalg.qr(gaussian)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    q = q * signs
    return q.astype(np.float32)


def _random_projection_matrix(dimension: int, seed: Optional[int]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((dimension, dimension)).astype(np.float32)


@dataclass
class TurboQuantMSEEncoding:
    indices: Any
    norms: Any
    prefix_shape: Tuple[int, ...]


@dataclass
class TurboQuantProdEncoding:
    indices: Any
    qjl: Any
    residual_norms: Any
    norms: Any
    prefix_shape: Tuple[int, ...]


class TurboQuantMSE:
    """TurboQuant variant optimized for MSE reconstruction."""

    def __init__(
        self,
        dimension: int,
        bits: int,
        *,
        seed: Optional[int] = None,
        grid_size: int = 16385,
        max_iter: int = 200,
        rotation: Optional[np.ndarray] = None,
        codebook: Optional[np.ndarray] = None,
    ):
        if dimension < 2:
            raise ValueError("dimension must be >= 2")
        if bits < 1:
            raise ValueError("bits must be >= 1")

        self.dimension = int(dimension)
        self.bits = int(bits)
        self.rotation = (
            np.asarray(rotation, dtype=np.float32)
            if rotation is not None
            else _random_orthogonal_matrix(self.dimension, seed)
        )
        if self.rotation.shape != (self.dimension, self.dimension):
            raise ValueError(
                f"rotation must have shape {(self.dimension, self.dimension)}, got {self.rotation.shape}"
            )

        self.codebook = (
            np.asarray(codebook, dtype=np.float32)
            if codebook is not None
            else _build_lloyd_max_codebook(
                self.dimension, self.bits, grid_size=grid_size, max_iter=max_iter
            ).astype(np.float32)
        )
        if self.codebook.ndim != 1 or self.codebook.shape[0] != (1 << self.bits):
            raise ValueError("codebook shape does not match bit-width")
        self.boundaries = 0.5 * (self.codebook[:-1] + self.codebook[1:])
        self.index_dtype = _quant_index_dtype(self.bits)

    def _rotation_for(self, xp):
        return xp.asarray(self.rotation)

    def _codebook_for(self, xp):
        return xp.asarray(self.codebook)

    def _boundaries_for(self, xp):
        return xp.asarray(self.boundaries)

    def _normalize_vectors(self, flat_vectors):
        xp = _xp_of(flat_vectors)
        norms = xp.linalg.norm(flat_vectors, axis=1)
        safe_norms = xp.where(norms > 0, norms, xp.ones_like(norms))
        unit = flat_vectors / safe_norms[:, None]
        unit = xp.where(norms[:, None] > 0, unit, xp.zeros_like(unit))
        return unit, norms

    def _quantize_unit(self, flat_unit_vectors):
        xp = _xp_of(flat_unit_vectors)
        rotated = flat_unit_vectors @ self._rotation_for(xp).T
        indices = xp.searchsorted(self._boundaries_for(xp), rotated, side="left")
        return indices.astype(self.index_dtype, copy=False)

    def _dequantize_unit(self, indices):
        xp = _xp_of(indices)
        centroids = self._codebook_for(xp)
        rotated = centroids[indices]
        return rotated @ self._rotation_for(xp)

    def quantize(self, vectors: Any) -> TurboQuantMSEEncoding:
        array = _to_array(vectors)
        flat, prefix_shape = _flatten_vectors(array, self.dimension)
        unit, norms = self._normalize_vectors(flat)
        indices = self._quantize_unit(unit)
        return TurboQuantMSEEncoding(indices=indices, norms=norms, prefix_shape=prefix_shape)

    def dequantize(self, encoding: TurboQuantMSEEncoding):
        unit_recon = self._dequantize_unit(encoding.indices)
        recon = unit_recon * encoding.norms[:, None]
        return _restore_shape(recon, encoding.prefix_shape)


class TurboQuantProd:
    """TurboQuant variant optimized for unbiased inner-product estimation."""

    def __init__(
        self,
        dimension: int,
        bits: int,
        *,
        seed: Optional[int] = None,
        grid_size: int = 16385,
        max_iter: int = 200,
        rotation: Optional[np.ndarray] = None,
        projection: Optional[np.ndarray] = None,
    ):
        if bits < 1:
            raise ValueError("bits must be >= 1")
        self.dimension = int(dimension)
        self.bits = int(bits)
        self.mse = TurboQuantMSE(
            dimension,
            max(bits - 1, 1),
            seed=seed,
            grid_size=grid_size,
            max_iter=max_iter,
            rotation=rotation,
        )
        proj_seed = None if seed is None else seed + 1
        self.projection = (
            np.asarray(projection, dtype=np.float32)
            if projection is not None
            else _random_projection_matrix(self.dimension, proj_seed)
        )
        if self.projection.shape != (self.dimension, self.dimension):
            raise ValueError(
                f"projection must have shape {(self.dimension, self.dimension)}, got {self.projection.shape}"
            )

    def _projection_for(self, xp):
        return xp.asarray(self.projection)

    def quantize(self, vectors: Any) -> TurboQuantProdEncoding:
        array = _to_array(vectors)
        flat, prefix_shape = _flatten_vectors(array, self.dimension)
        unit, norms = self.mse._normalize_vectors(flat)
        indices = self.mse._quantize_unit(unit)
        mse_recon = self.mse._dequantize_unit(indices)
        residual = unit - mse_recon
        residual_norms = _xp_of(residual).linalg.norm(residual, axis=1)
        signed = _sign_pm_one(
            _xp_of(residual),
            residual @ self._projection_for(_xp_of(residual)).T,
        )
        return TurboQuantProdEncoding(
            indices=indices,
            qjl=signed,
            residual_norms=residual_norms,
            norms=norms,
            prefix_shape=prefix_shape,
        )

    def _unit_stage1(self, indices):
        return self.mse._dequantize_unit(indices)

    def _unit_stage2(self, qjl, residual_norms):
        xp = _xp_of(qjl)
        correction = (qjl @ self._projection_for(xp)) * (
            math.sqrt(math.pi / 2.0) / self.dimension
        )
        return correction * residual_norms[:, None]

    def dequantize(self, encoding: TurboQuantProdEncoding):
        unit_recon = self._unit_stage1(encoding.indices) + self._unit_stage2(
            encoding.qjl, encoding.residual_norms
        )
        recon = unit_recon * encoding.norms[:, None]
        return _restore_shape(recon, encoding.prefix_shape)

    def attention_score(self, query: Any, encoding: TurboQuantProdEncoding):
        query_array = _to_array(query)
        flat_q, q_prefix = _flatten_vectors(query_array, self.dimension)
        num_codes = encoding.indices.shape[0]
        if flat_q.shape[0] == 1 and num_codes > 1:
            xp = _xp_of(flat_q)
            flat_q = xp.broadcast_to(flat_q, (num_codes, self.dimension))
            q_prefix = encoding.prefix_shape
        elif flat_q.shape[0] != num_codes:
            raise ValueError(
                f"query batch size {flat_q.shape[0]} does not match encoding batch size {num_codes}"
            )

        xp = _xp_of(flat_q)
        stage1 = self._unit_stage1(encoding.indices)
        score_stage1 = xp.sum(flat_q * stage1, axis=1)
        projected_q = flat_q @ self._projection_for(xp).T
        score_stage2 = (
            math.sqrt(math.pi / 2.0)
            / self.dimension
            * encoding.residual_norms
            * xp.sum(projected_q * encoding.qjl, axis=1)
        )
        scores = encoding.norms * (score_stage1 + score_stage2)
        return scores.reshape(q_prefix if flat_q.shape[0] != 1 else encoding.prefix_shape)


def turboquant_mse(vectors: Any, bits: int, *, seed: Optional[int] = None) -> Tuple[TurboQuantMSE, TurboQuantMSEEncoding]:
    """Convenience helper that instantiates and applies `TurboQuantMSE`."""
    array = _to_array(vectors)
    dimension = array.shape[-1]
    quantizer = TurboQuantMSE(dimension, bits, seed=seed)
    return quantizer, quantizer.quantize(vectors)


def turboquant_prod(vectors: Any, bits: int, *, seed: Optional[int] = None) -> Tuple[TurboQuantProd, TurboQuantProdEncoding]:
    """Convenience helper that instantiates and applies `TurboQuantProd`."""
    array = _to_array(vectors)
    dimension = array.shape[-1]
    quantizer = TurboQuantProd(dimension, bits, seed=seed)
    return quantizer, quantizer.quantize(vectors)
