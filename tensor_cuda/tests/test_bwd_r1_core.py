"""Direct VJPs on the exact fp32 operands widened to NumPy fp64."""

import os

import numpy as np
import pytest

import tensor_cuda as tc
from bwd_r1_support.oracle import CORE, Case, compare, kokoro, real_tail, small

gpu = pytest.mark.skipif(os.environ.get("CUDA_VISIBLE_DEVICES") == "", reason="CUDA disabled for CPU lane")


def require_cuda():
    # Fixture-backed cells resolve and validate their data before this skip.
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        pytest.skip("CUDA disabled for CPU lane")


def forward_tc(case, *xs):
    op, attrs = case.op, case.attrs
    x = xs[0]
    if op == "matmul":
        return tc.matmul(*xs, **attrs)
    if op == "add":
        return x + xs[1]
    if op == "mul":
        return x * xs[1]
    if op == "div":
        return x / xs[1]
    if op == "div_constant":
        return x / tc.tensor(np.float32(attrs["divisor"]))
    if op == "square":
        return x * x
    if op == "leaky":
        return x.maximum(x * attrs["slope"])
    if op == "mean":
        return x.mean(axes=[attrs["axis"]])
    if op == "snake":
        alpha = xs[1]
        sine = (alpha * x).sin()
        return x + alpha.reciprocal() * (sine * sine)
    return getattr(x, op)()


def device_gradients(case):
    require_cuda()
    xs = tuple(tc.from_numpy(a, requires_grad=True) for a in case.operands)
    forward = forward_tc(case, *xs)
    assert tuple(forward.shape) == case.seed.shape
    (forward * tc.from_numpy(case.seed)).sum().backward()
    assert all(x.grad is not None for x in xs)
    return tuple(x.grad.numpy() for x in xs)


@pytest.mark.parametrize("op", [*CORE, "div_constant"])
@pytest.mark.parametrize("size", ["small", "kokoro"])
def test_core_gradients(op, size):
    case = (small if size == "small" else kokoro)(op)
    got = device_gradients(case)
    compare(case, got)
    if op == "add":
        for actual in got:
            np.testing.assert_array_equal(actual.view(np.uint32), case.seed.view(np.uint32))


@pytest.mark.parametrize("op", ["add", "mul", "div"])
@pytest.mark.parametrize("size", ["small", "kokoro"])
def test_core_broadcast_gradients(op, size):
    case = (small if size == "small" else kokoro)(op, "broadcast")
    compare(case, device_gradients(case))


@pytest.mark.parametrize("kind", ["trans", "batch_shared", "batch_shared_trans", "batched", "batched_trans"])
@gpu
def test_core_matmul_geometries(kind):
    case = small("matmul", kind)
    compare(case, device_gradients(case))


@pytest.mark.parametrize("kind", ["0p1", "0p01", "capture"])
def test_core_leaky_additional_shapes(kind):
    case = real_tail() if kind == "capture" else kokoro("leaky", kind)
    compare(case, device_gradients(case))


@pytest.mark.parametrize("slope", [.2, .1, .01])
@gpu
def test_core_leaky_ties(slope):
    x = tc.tensor([-2., -0., 0., 7.], requires_grad=True)
    g = tc.tensor([-3., 4., -5., 6.])
    (x.maximum(x * slope) * g).sum().backward()
    expected = np.array([-3. * np.float32(slope), 4., -5., 6.], dtype=np.float32)
    np.testing.assert_array_equal(x.grad.numpy(), expected)
    print("LeakyReLU +/-0 derivative 1", slope, flush=True)


@gpu
def test_core_maximum_first_slot_on_equality():
    a = tc.tensor([-0., 0., -1., 7.], requires_grad=True)
    b = tc.tensor([0., -0., -1., 7.], requires_grad=True)
    seed = tc.tensor([3., -2., 1., -5.])
    (a.maximum(b) * seed).sum().backward()
    np.testing.assert_array_equal(a.grad.numpy(), seed.numpy())
    np.testing.assert_array_equal(b.grad.numpy(), np.zeros(4, dtype=np.float32))
    print("maximum equality derivatives (1, 0)", flush=True)


@gpu
def test_core_matmul_exact_cancellation_and_zero_taps():
    case = Case("matmul", "exact_taps",
                (np.array([[1, 1], [2, -1]], dtype=np.float32),
                 np.array([[1, 0, 2], [-1, 0, 3]], dtype=np.float32)),
                np.array([[2, 1, 0], [2, -4, 0]], dtype=np.float32),
                {"trans_b": False, "alpha": 1.})
    got = device_gradients(case)
    compare(case, got)
    np.testing.assert_array_equal(got[0], [[2, -2], [2, -2]])
    np.testing.assert_array_equal(got[1], [[6, -7, 0], [0, 5, 0]])
