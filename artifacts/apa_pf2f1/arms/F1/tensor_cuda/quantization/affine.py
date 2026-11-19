"""Uniform affine quantization and low-bit packing primitives.

These routines are intentionally NumPy-only reference math. Runtime kernels use
the resulting packed bytes, scales, and zero points; calibration methods can
swap scale rules without touching CUDA bindings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AffineQuantizedWeights:
    """Packed per-group affine weights for a logical (N, K) matrix."""

    packed: np.ndarray
    scales: np.ndarray
    zeros: np.ndarray
    bits: int
    in_features: int
    group_size: int


def qmax(bits: int) -> int:
    """Maximum unsigned code value for a bit width."""
    _validate_bits(bits)
    return (1 << int(bits)) - 1


def symmetric_offset(bits: int) -> float:
    """Unsigned-code offset for signed symmetric grids: q - 2^(bits-1)."""
    _validate_bits(bits)
    return float(1 << (int(bits) - 1))


def packed_width(in_features: int, bits: int) -> int:
    """Bytes needed for one packed row of K codes at `bits` bits/code."""
    _validate_bits(bits)
    if in_features <= 0:
        raise ValueError("in_features must be > 0")
    return (int(in_features) * int(bits) + 7) // 8


def pack_lowbit(codes: np.ndarray, bits: int) -> np.ndarray:
    """Pack a 2D uint code matrix into little-endian bit rows.

    q[k]'s bit b is stored at bit offset k*bits+b. This matches the TensorCUDA
    `intn_*` kernels and keeps INT3 byte padding explicit.
    """
    bits = _validate_bits(bits)
    q = np.asarray(codes)
    if q.ndim != 2:
        raise ValueError(f"codes must be rank-2, got shape {q.shape}")
    if q.size and (q.min() < 0 or q.max() > qmax(bits)):
        raise ValueError(f"codes contain values outside the {bits}-bit range")
    rows, cols = q.shape
    if bits in (1, 2, 4):
        return _pack_even_lowbit(q.astype(np.uint8, copy=False), bits, rows, cols)
    if bits == 3:
        return _pack_int3(q.astype(np.uint8, copy=False), rows, cols)
    return _pack_lowbit_slow(q, bits, rows, cols)


def unpack_lowbit(packed: np.ndarray, bits: int, in_features: int) -> np.ndarray:
    """Unpack little-endian low-bit rows into a uint8 code matrix."""
    bits = _validate_bits(bits)
    p = np.asarray(packed, dtype=np.uint8)
    if p.ndim != 2:
        raise ValueError(f"packed must be rank-2, got shape {p.shape}")
    expected = packed_width(in_features, bits)
    if p.shape[1] != expected:
        raise ValueError(
            f"packed byte width mismatch: got {p.shape[1]}, expected {expected}"
        )
    rows = p.shape[0]
    codes = np.zeros((rows, int(in_features)), dtype=np.uint8)
    for row in range(rows):
        for col in range(int(in_features)):
            bit0 = col * bits
            value = 0
            for b in range(bits):
                bit = bit0 + b
                value |= int((p[row, bit >> 3] >> (bit & 7)) & 1) << b
            codes[row, col] = value
    return codes


def _pack_even_lowbit(q: np.ndarray, bits: int, rows: int, cols: int) -> np.ndarray:
    per_byte = 8 // bits
    width = packed_width(cols, bits)
    padded_cols = ((cols + per_byte - 1) // per_byte) * per_byte
    if padded_cols != cols:
        qp = np.zeros((rows, padded_cols), dtype=np.uint8)
        qp[:, :cols] = q
    else:
        qp = q
    packed = np.zeros((rows, padded_cols // per_byte), dtype=np.uint8)
    mask = (1 << bits) - 1
    for i in range(per_byte):
        packed |= ((qp[:, i::per_byte] & mask) << (i * bits)).astype(np.uint8)
    return packed[:, :width]


def _pack_int3(q: np.ndarray, rows: int, cols: int) -> np.ndarray:
    width = packed_width(cols, 3)
    packed = np.zeros((rows, width), dtype=np.uint8)
    bit0 = np.arange(cols, dtype=np.int64) * 3
    byte_idx = bit0 >> 3
    shift = (bit0 & 7).astype(np.uint16)
    vals = (q.astype(np.uint16, copy=False) & 0x07) << shift[None, :]
    lo = (vals & 0xFF).astype(np.uint8, copy=False)
    hi = (vals >> 8).astype(np.uint8, copy=False)
    spill = hi != 0
    for row in range(rows):
        np.bitwise_or.at(packed[row], byte_idx, lo[row])
        if spill[row].any():
            np.bitwise_or.at(
                packed[row],
                byte_idx[spill[row]] + 1,
                hi[row, spill[row]],
            )
    return packed


def _pack_lowbit_slow(q: np.ndarray, bits: int, rows: int, cols: int) -> np.ndarray:
    packed = np.zeros((rows, packed_width(cols, bits)), dtype=np.uint8)
    q64 = q.astype(np.uint64, copy=False)
    for row in range(rows):
        for col in range(cols):
            value = int(q64[row, col])
            bit0 = col * bits
            for b in range(bits):
                if value & (1 << b):
                    bit = bit0 + b
                    packed[row, bit >> 3] |= np.uint8(1 << (bit & 7))
    return packed


def quantize_affine_per_group(
    weights: np.ndarray,
    bits: int,
    group_size: int = 128,
    *,
    scale_dtype=np.float16,
) -> AffineQuantizedWeights:
    """Uniform min/max affine quantization for a logical (N, K) weight matrix."""
    bits = _validate_bits(bits)
    w = np.asarray(weights, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError(f"weights must be rank-2, got shape {w.shape}")
    out_features, in_features = w.shape
    groups = _num_groups(in_features, group_size)
    grouped = w.reshape(out_features, groups, int(group_size))
    mins = grouped.min(axis=2)
    maxs = grouped.max(axis=2)
    scales = (maxs - mins) / float(qmax(bits))
    scales = np.where(scales == 0, np.ones_like(scales), scales)
    zeros = mins
    codes = np.clip(
        np.round((grouped - zeros[:, :, None]) / scales[:, :, None]),
        0,
        qmax(bits),
    ).astype(np.uint8)
    packed = pack_lowbit(codes.reshape(out_features, in_features), bits)
    return AffineQuantizedWeights(
        packed=packed,
        scales=scales.astype(scale_dtype),
        zeros=zeros.astype(scale_dtype),
        bits=bits,
        in_features=in_features,
        group_size=int(group_size),
    )


def dequantize_affine_per_group(
    packed: np.ndarray,
    scales: np.ndarray,
    zeros: np.ndarray,
    bits: int,
    in_features: int,
    group_size: int = 128,
    *,
    out_dtype=np.float32,
) -> np.ndarray:
    """Dequantize packed affine weights to logical (N, K) layout."""
    bits = _validate_bits(bits)
    q = unpack_lowbit(packed, bits, in_features).astype(np.float32)
    rows = q.shape[0]
    groups = _num_groups(in_features, group_size)
    s = _validate_param_matrix(scales, rows, groups, "scales")
    z = _validate_param_matrix(zeros, rows, groups, "zeros")
    w = q.reshape(rows, groups, int(group_size))
    w = w * s.astype(np.float32)[:, :, None]
    w = w + z.astype(np.float32)[:, :, None]
    return w.reshape(rows, int(in_features)).astype(out_dtype, copy=False)


def dequantize_symmetric_per_group(
    packed: np.ndarray,
    scales: np.ndarray,
    bits: int,
    in_features: int,
    group_size: int = 128,
    *,
    out_dtype=np.float32,
) -> np.ndarray:
    """Dequantize packed weights using q - 2^(bits-1), no zero tensor."""
    bits = _validate_bits(bits)
    q = unpack_lowbit(packed, bits, in_features).astype(np.float32)
    rows = q.shape[0]
    groups = _num_groups(in_features, group_size)
    s = _validate_param_matrix(scales, rows, groups, "scales")
    w = q.reshape(rows, groups, int(group_size)) - symmetric_offset(bits)
    w = w * s.astype(np.float32)[:, :, None]
    return w.reshape(rows, int(in_features)).astype(out_dtype, copy=False)


def _validate_bits(bits: int) -> int:
    bits = int(bits)
    if bits < 1 or bits > 8:
        raise ValueError("bits must be in [1, 8] for uint8 low-bit packing")
    return bits


def _num_groups(in_features: int, group_size: int) -> int:
    in_features = int(in_features)
    group_size = int(group_size)
    if group_size <= 0:
        raise ValueError("group_size must be > 0")
    if in_features <= 0:
        raise ValueError("in_features must be > 0")
    if in_features % group_size != 0:
        raise ValueError("in_features must be divisible by group_size")
    return in_features // group_size


def _validate_param_matrix(values, rows: int, groups: int, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.shape != (rows, groups):
        raise ValueError(f"{name} must have shape {(rows, groups)}, got {arr.shape}")
    return arr
