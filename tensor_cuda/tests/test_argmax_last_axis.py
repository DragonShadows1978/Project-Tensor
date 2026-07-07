"""Phase 1.1 (KERNEL_OPT_IMPLEMENTATION_PLAN.md): device-side last-axis
argmax for the decode hot path. Correctness only here — timings are a
separate (non-gating) receipt. Tie-break must match numpy.argmax exactly
(lowest index wins on ties)."""

import numpy as np
import pytest

import tensor_cuda as tc

DTYPES = ["float32", "float16", "bfloat16"]


def _to_np_f32(t):
    # bf16 has no numpy dtype; the Tensor's own .astype("float32") round-trip
    # is how the rest of the suite reads bf16 tensors back to host.
    return t.astype("float32").numpy().astype(np.float64)


def _check(x_np, dtype):
    """x_np: float64 host array, any shape, last axis is the reduced one.
    Builds a device tensor at `dtype`, runs argmax_last_axis, compares to
    numpy.argmax(-1) computed on the SAME data as actually stored at that
    dtype (so f16/bf16 rounding can't cause a spurious mismatch)."""
    if dtype == "bfloat16":
        t = tc.tensor(x_np.astype(np.float32), dtype="float32").astype("bfloat16")
    else:
        np_dtype = np.float32 if dtype == "float32" else np.float16
        t = tc.tensor(x_np.astype(np_dtype), dtype=dtype)
    # Reference computed on the value ACTUALLY stored on device (post-cast),
    # not the original float64 input, so f16/bf16 rounding is not a source
    # of disagreement between kernel and reference.
    stored = _to_np_f32(t)
    expected = np.argmax(stored, axis=-1)
    got = tc.argmax_last_axis(t).numpy()
    np.testing.assert_array_equal(got, expected,
        err_msg=f"dtype={dtype} shape={x_np.shape}")


@pytest.mark.parametrize("dtype", DTYPES)
def test_basic_2d(dtype):
    x = np.array([[1.0, 3.0, 2.0, -5.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    _check(x, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_ties_lowest_index_wins(dtype):
    # Exact ties at several positions; numpy.argmax returns the FIRST max.
    x = np.array([
        [5.0, 5.0, 5.0, 5.0],
        [1.0, 2.0, 2.0, 1.0],
        [-1.0, -1.0, 0.0, 0.0],
        [3.0, 1.0, 3.0, 3.0],
    ], dtype=np.float64)
    _check(x, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_negatives_and_zeros(dtype):
    x = np.array([
        [-1.0, -2.0, -3.0, -0.5],
        [-100.0, -100.0, -99.999, -100.0],
        [0.0, -0.0, 0.0, -0.0],
    ], dtype=np.float64)
    _check(x, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_single_element_row(dtype):
    x = np.array([[42.0], [-7.0]], dtype=np.float64)
    _check(x, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_1d_row(dtype):
    x = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], dtype=np.float64)
    _check(x, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_3d_batched(dtype):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 5, 17)).astype(np.float64)
    _check(x, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", [1, 2, 31, 32, 33, 63, 64, 65, 255, 256, 257, 1000, 4096])
def test_random_various_widths(dtype, N):
    rng = np.random.default_rng(N * 7 + 1)
    x = rng.standard_normal((4, N)).astype(np.float64)
    _check(x, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_decode_hot_shape_single_row_vocab(dtype):
    """The actual decode shape this kernel targets: ONE row, ~152k vocab.
    outer=1 must still parallelize across the block (this is the whole
    point vs. the generic one-thread-per-row reduce_arg)."""
    rng = np.random.default_rng(152000)
    N = 152064  # Qwen-class vocab size
    x = rng.standard_normal((1, N)).astype(np.float64)
    # plant an adversarial unique max near the end so a truncated/short
    # grid-stride loop would miss it
    x[0, N - 3] = x.max() + 10.0
    _check(x, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_adversarial_max_at_boundaries(dtype):
    rng = np.random.default_rng(3)
    N = 300
    for pos in [0, 1, N // 2, N - 2, N - 1]:
        x = rng.standard_normal((2, N)).astype(np.float64)
        x[:, pos] = x.max() + 5.0
        _check(x, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_all_equal_row(dtype):
    x = np.full((3, 200), 7.0, dtype=np.float64)
    _check(x, dtype)


def test_matches_generic_argmax_axis_minus1():
    """Cross-check against the existing generic Tensor.argmax(-1) path on
    plain fp32 to confirm the two kernels agree independent of numpy."""
    rng = np.random.default_rng(11)
    x = rng.standard_normal((6, 613)).astype(np.float32)
    t = tc.tensor(x)
    fast = tc.argmax_last_axis(t).numpy()
    generic = t.argmax(-1).numpy().reshape(-1)
    np.testing.assert_array_equal(fast, generic)


def test_output_shape_axis_removed():
    x = np.random.randn(2, 3, 9).astype(np.float32)
    t = tc.tensor(x)
    out = tc.argmax_last_axis(t)
    assert tuple(out.shape) == (2, 3)


def test_empty_last_axis_raises():
    x = np.zeros((2, 0), dtype=np.float32)
    t = tc.tensor(x)
    with pytest.raises(RuntimeError):
        tc.argmax_last_axis(t)
