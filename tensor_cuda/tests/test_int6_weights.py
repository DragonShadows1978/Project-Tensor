"""Focused correctness coverage for the native group-128 INT6 weight path."""

import numpy as np

import tensor_cuda as tc
from tensor_cuda.quantization import (
    dequantize_symmetric_per_group,
    pack_lowbit,
)


GROUP = 128


def _empty_zeros():
    return tc.tensor(np.empty((0,), dtype=np.float16), dtype="float16")


def test_int6_dequant_all_codes_scale_signs_and_boundaries_are_exact():
    base = np.tile(np.arange(64, dtype=np.uint8), 4)
    codes = np.stack((base, base[::-1])).astype(np.uint8)
    scales = np.array(
        [[0.01973, -0.00731], [-0.04321, 0.01117]],
        dtype=np.float16,
    )
    packed = pack_lowbit(codes, bits=6)
    expected = (
        (codes.reshape(2, 2, GROUP).astype(np.float32) - 32.0)
        * scales.astype(np.float32)[:, :, None]
    ).reshape(2, 256)

    actual = tc.int6_dequant(
        tc.tensor(packed, dtype="uint8"),
        tc.tensor(scales, dtype="float16"),
        _empty_zeros(),
        GROUP,
        out_dtype="float32",
    ).numpy().T

    np.testing.assert_array_equal(actual, expected)
    for boundary in (127, 128):
        np.testing.assert_array_equal(actual[:, boundary], expected[:, boundary])


def test_int6_explicit_zeros_affine_path_is_exact():
    codes = np.tile(np.arange(64, dtype=np.uint8), (3, 4))
    scales = np.array(
        [[0.5, 0.25], [0.125, -0.5], [-0.25, 0.0625]], dtype=np.float16
    )
    zeros = np.array(
        [[-1.0, 2.0], [0.5, -3.0], [4.0, 0.25]], dtype=np.float16
    )
    packed = pack_lowbit(codes, bits=6)
    expected = (
        codes.reshape(3, 2, GROUP).astype(np.float32)
        * scales.astype(np.float32)[:, :, None]
        + zeros.astype(np.float32)[:, :, None]
    ).reshape(3, 256)

    actual = tc.int6_dequant(
        tc.tensor(packed, dtype="uint8"),
        tc.tensor(scales, dtype="float16"),
        tc.tensor(zeros, dtype="float16"),
        GROUP,
        out_dtype="float32",
    ).numpy().T

    np.testing.assert_array_equal(actual, expected)


def test_int6_two_stage_tile_and_gemv_match():
    rng = np.random.default_rng(606)
    m, n, k = 4, 96, 256
    codes = rng.integers(0, 64, size=(n, k), dtype=np.uint8)
    scales = (
        rng.random((n, k // GROUP), dtype=np.float32) * 0.02 + 2.0**-12
    ).astype(np.float16)
    packed = pack_lowbit(codes, bits=6)
    w_ref = dequantize_symmetric_per_group(packed, scales, 6, k, GROUP)
    x = rng.standard_normal((m, k)).astype(np.float32) * 0.25
    y_ref = x @ w_ref.T

    packed_t = tc.tensor(packed, dtype="uint8")
    scales_t = tc.tensor(scales, dtype="float16")
    zeros_t = _empty_zeros()
    x_t = tc.tensor(x, dtype="float32")

    y_two = tc.int6_linear(x_t, packed_t, scales_t, zeros_t, GROUP).numpy()
    y_tile = tc.int6_linear_fused(
        x_t, packed_t, scales_t, zeros_t, GROUP
    ).numpy()
    y_gemv = tc.int6_linear_fused(
        tc.tensor(x[:1], dtype="float32"),
        packed_t,
        scales_t,
        zeros_t,
        GROUP,
    ).numpy()

    np.testing.assert_allclose(y_two, y_ref, rtol=3e-4, atol=3e-4)
    np.testing.assert_allclose(y_tile, y_ref, rtol=3e-4, atol=3e-4)
    np.testing.assert_allclose(y_gemv, y_ref[:1], rtol=3e-4, atol=3e-4)
