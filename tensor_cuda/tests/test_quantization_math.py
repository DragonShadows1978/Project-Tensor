"""Pure Python gates for reusable quantization math."""

import numpy as np

from tensor_cuda.quantization import (
    dequantize_affine_per_group,
    dequantize_symmetric_per_group,
    pack_lowbit,
    packed_width,
    qmax,
    quantize_affine_per_group,
    quantize_symmetric_per_group,
    symmetric_offset,
    unpack_lowbit,
)


def test_lowbit_pack_roundtrip_handles_int3_padding():
    rng = np.random.default_rng(101)
    codes = rng.integers(0, 8, size=(7, 257), dtype=np.uint8)
    packed = pack_lowbit(codes, bits=3)

    assert packed.shape == (7, packed_width(257, 3))
    assert packed.shape[1] == 97
    np.testing.assert_array_equal(unpack_lowbit(packed, 3, 257), codes)


def test_affine_per_group_roundtrip_matches_manual_formula():
    rng = np.random.default_rng(102)
    weights = rng.standard_normal((11, 128)).astype(np.float32) * 0.1
    q = quantize_affine_per_group(weights, bits=3, group_size=32)
    deq = dequantize_affine_per_group(
        q.packed, q.scales, q.zeros, q.bits, q.in_features, q.group_size
    )

    assert q.packed.shape == (11, packed_width(128, 3))
    assert q.scales.shape == (11, 4)
    assert q.zeros.shape == (11, 4)
    rel = np.abs(deq - weights).mean() / (np.abs(weights).mean() + 1e-9)
    assert rel < 0.25


def test_symmetric_grid_math_is_explicit():
    codes = np.tile(np.arange(4, dtype=np.uint8), (2, 2))
    packed = pack_lowbit(codes, bits=2)
    scales = np.full((2, 2), 0.5, dtype=np.float16)

    deq = dequantize_symmetric_per_group(packed, scales, 2, 8, group_size=4)
    expected = (codes.astype(np.float32) - symmetric_offset(2)) * 0.5

    np.testing.assert_allclose(deq, expected)
    assert qmax(2) == 3
    assert qmax(3) == 7


def test_int3_eight_values_per_three_bytes_layout():
    codes = np.arange(8, dtype=np.uint8).reshape(1, 8)
    packed = pack_lowbit(codes, bits=3)

    np.testing.assert_array_equal(
        packed, np.array([[0x88, 0xC6, 0xFA]], dtype=np.uint8)
    )
    np.testing.assert_array_equal(unpack_lowbit(packed, 3, 8), codes)


def test_int3_pack_roundtrip_covers_all_codes_and_group_boundaries():
    codes = np.tile(np.arange(8, dtype=np.uint8), (3, 32))
    packed = pack_lowbit(codes, bits=3)

    assert packed.shape == (3, 96)
    np.testing.assert_array_equal(unpack_lowbit(packed, 3, 256), codes)


def test_int3_symmetric_quantizer_returns_empty_zeros():
    rng = np.random.default_rng(103)
    weights = rng.standard_normal((13, 256)).astype(np.float32)
    q = quantize_symmetric_per_group(weights, bits=3, group_size=128)
    deq = dequantize_symmetric_per_group(
        q.packed, q.scales, q.bits, q.in_features, q.group_size
    )

    assert q.packed.shape == (13, 96)
    assert q.scales.shape == (13, 2)
    assert q.scales.dtype == np.float16
    assert q.zeros.shape == (0,)
    rel_fro = np.linalg.norm(deq - weights) / np.linalg.norm(weights)
    assert rel_fro < 0.27


def test_int6_four_values_per_three_bytes_layout():
    codes = np.array([[1, 2, 3, 4]], dtype=np.uint8)
    packed = pack_lowbit(codes, bits=6)

    np.testing.assert_array_equal(
        packed, np.array([[0x81, 0x30, 0x10]], dtype=np.uint8)
    )
    np.testing.assert_array_equal(unpack_lowbit(packed, 6, 4), codes)


def test_int6_pack_roundtrip_covers_all_codes_and_group_boundaries():
    codes = np.tile(np.arange(64, dtype=np.uint8), (3, 4))
    packed = pack_lowbit(codes, bits=6)

    assert packed.shape == (3, 192)
    np.testing.assert_array_equal(unpack_lowbit(packed, 6, 256), codes)


def test_int6_symmetric_quantizer_returns_empty_zeros():
    rng = np.random.default_rng(106)
    weights = rng.standard_normal((13, 256)).astype(np.float32)
    q = quantize_symmetric_per_group(weights, bits=6, group_size=128)
    deq = dequantize_symmetric_per_group(
        q.packed, q.scales, q.bits, q.in_features, q.group_size
    )

    assert q.packed.shape == (13, 192)
    assert q.scales.shape == (13, 2)
    assert q.scales.dtype == np.float16
    assert q.zeros.shape == (0,)
    rel_fro = np.linalg.norm(deq - weights) / np.linalg.norm(weights)
    assert rel_fro < 0.04
