"""Focused correctness coverage for the native group-128 W3A16 path."""

import numpy as np
import pytest

import tensor_cuda as tc
from tensor_cuda.quantization import (
    dequantize_symmetric_per_group,
    pack_lowbit,
)


GROUP = 128


def _empty_zeros():
    return tc.tensor(np.empty((0,), dtype=np.float16), dtype="float16")


def test_int3_dequant_all_codes_scale_signs_and_boundaries_are_exact():
    base = np.tile(np.arange(8, dtype=np.uint8), 16)
    codes = np.stack((np.tile(base, 2), np.tile(base[::-1], 2)))
    scales = np.array(
        [[0.01973, -0.00731], [-0.04321, 0.01117]], dtype=np.float16
    )
    packed = pack_lowbit(codes, bits=3)
    expected = dequantize_symmetric_per_group(
        packed, scales, 3, 256, GROUP
    )

    actual = tc.int3_dequant(
        tc.tensor(packed, dtype="uint8"),
        tc.tensor(scales, dtype="float16"),
        _empty_zeros(),
        GROUP,
        out_dtype="float32",
    ).numpy().T

    np.testing.assert_array_equal(actual, expected)
    for boundary in (127, 128):
        np.testing.assert_array_equal(actual[:, boundary], expected[:, boundary])


def test_int3_explicit_zeros_affine_path_is_exact():
    codes = np.tile(np.arange(8, dtype=np.uint8), (3, 32))
    scales = np.array(
        [[0.5, 0.25], [0.125, -0.5], [-0.25, 0.0625]], dtype=np.float16
    )
    zeros = np.array(
        [[-1.0, 2.0], [0.5, -3.0], [4.0, 0.25]], dtype=np.float16
    )
    packed = pack_lowbit(codes, bits=3)
    expected = (
        codes.reshape(3, 2, GROUP).astype(np.float32)
        * scales.astype(np.float32)[:, :, None]
        + zeros.astype(np.float32)[:, :, None]
    ).reshape(3, 256)

    actual = tc.int3_dequant(
        tc.tensor(packed, dtype="uint8"),
        tc.tensor(scales, dtype="float16"),
        tc.tensor(zeros, dtype="float16"),
        GROUP,
        out_dtype="float32",
    ).numpy().T

    np.testing.assert_array_equal(actual, expected)


def test_int3_gemv_and_fused_tile_match_dequant_matmul():
    rng = np.random.default_rng(303)
    m, n, k = 4, 96, 256
    codes = rng.integers(0, 8, size=(n, k), dtype=np.uint8)
    scales = (
        rng.random((n, k // GROUP), dtype=np.float32) * 0.02 + 2.0**-12
    ).astype(np.float16)
    packed = pack_lowbit(codes, bits=3)
    x = (rng.standard_normal((m, k)).astype(np.float32) * 0.25).astype(
        np.float16
    )

    packed_t = tc.tensor(packed, dtype="uint8")
    scales_t = tc.tensor(scales, dtype="float16")
    zeros_t = _empty_zeros()
    x_t = tc.tensor(x, dtype="float16")
    w_kn = tc.int3_dequant(
        packed_t, scales_t, zeros_t, GROUP, out_dtype="float16"
    )
    reference = tc.matmul(x_t, w_kn).numpy().astype(np.float32)
    direct = tc.int3_linear(x_t, packed_t, scales_t, zeros_t, GROUP).numpy()
    tile = tc.int3_linear_fused(
        x_t, packed_t, scales_t, zeros_t, GROUP
    ).numpy()
    decode = tc.int3_linear_fused(
        tc.tensor(x[:1], dtype="float16"),
        packed_t,
        scales_t,
        zeros_t,
        GROUP,
    ).numpy()

    np.testing.assert_allclose(direct, reference, rtol=3e-3, atol=3e-3)
    np.testing.assert_allclose(tile, reference, rtol=3e-3, atol=3e-3)
    np.testing.assert_allclose(decode, reference[:1], rtol=3e-3, atol=3e-3)


def test_int3_linear_rejects_non_fp16_activations():
    packed = np.zeros((2, 48), dtype=np.uint8)
    scales = np.ones((2, 1), dtype=np.float16)
    with pytest.raises(RuntimeError, match="W3A16"):
        tc.int3_linear(
            tc.tensor(np.zeros((1, 128), dtype=np.float32), dtype="float32"),
            tc.tensor(packed, dtype="uint8"),
            tc.tensor(scales, dtype="float16"),
            _empty_zeros(),
            GROUP,
        )
