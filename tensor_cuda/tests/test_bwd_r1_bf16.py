"""The bf16 factory creates leaves, and the BWD-R1 core ops consume them."""

from dataclasses import replace
from functools import cache
import json
import os

import numpy as np
import pytest

import tensor_cuda as tc
from bwd_r1_support.oracle import ART, CORE, classify, numerical, sha, small, vjp
from test_bwd_r1_core import forward_tc


@pytest.mark.parametrize("factory", ["tensor", "zeros", "ones", "rand", "randn"])
def test_host_bf16_factory_keeps_its_gradient(factory):
    args = ([1., -2., 4.],) if factory == "tensor" else (3,)
    x = getattr(tc, factory)(*args, device="cpu", dtype="bfloat16", requires_grad=True)
    assert x.requires_grad and x.dtype == "bfloat16" and x.grad is None
    # Host reshape and the existing custom VJP exercise the same factory and
    # real accumulation as matmul, without entering CUDA-only matmul.
    reshaped = x.reshape([1, 3])
    seed = tc.tensor([[-2., 3., 5.]], device="cpu", dtype="bfloat16")
    value = tc.tensor([[3., 1., -4.]], device="cpu", dtype="bfloat16")
    tc.attach_vjp(value, [reshaped], lambda dy, needs: (seed,)).backward()
    print("bf16 factory backward completed", factory, flush=True)
    assert x.grad is not None, "bf16 factory gradient was dropped"
    assert x.grad.dtype == "bfloat16"
    expected = np.array([-2., 3., 5.], dtype=np.float32)
    actual = x.grad.numpy()
    numerical(actual, expected)
    np.testing.assert_array_equal(actual, expected)


def test_host_bf16_factory_no_grad_and_explicit_cast():
    with tc.no_grad():
        x = tc.tensor([1., -2.], device="cpu", dtype="bfloat16", requires_grad=True)
    x.backward()
    assert x.grad is not None and x.grad.dtype == "bfloat16"

    source = tc.tensor([1., -2.], device="cpu", requires_grad=True)
    cast = source.astype("bfloat16")
    cast.backward()
    assert cast.requires_grad and cast.grad is None
    assert source.grad is not None and source.grad.dtype == "float32"


@cache
def bf16_tolerances():
    path = ART / "TOLERANCES_BF16_R5.json"
    seal = (ART / "TOLERANCES_BF16_SHA256_R5.txt").read_text().split()
    assert sha(path) == seal[0], "bf16 tolerance seal changed"
    return json.loads(path.read_text())


# Prior art: Google Brain bf16 (2018), whose 8-bit significand gives
# round-to-nearest unit roundoff 2^-8. The local budgets apply that precision
# to the inherited fp64 VJP and termwise error scale for these registered cells.
def assert_bf16_gradient(case, slot, data, expected, terms):
    numerical(data, expected)
    np.testing.assert_array_equal(data[expected == 0], 0, err_msg=f"{case.op} slot {slot}: zero taps")
    table = bf16_tolerances()
    op_class = table["class_overrides"].get(case.op, classify(case, slot))
    spec = table["classes"][op_class]
    scale = spec["atol_floor"] if terms is None else np.maximum(spec["atol_floor"], spec["atol_scale"] * terms)
    bound = scale + spec["rtol"] * abs(expected)
    error = abs(data.astype(np.float64) - expected)
    ratio = error / bound
    print(json.dumps(dict(dtype="bf16", op=case.op, slot=slot, tolerance=op_class,
                          max_abs=float(error.max()), max_bound_ratio=float(ratio.max()))), flush=True)
    assert np.all(ratio <= 1), (case.op, slot, float(ratio.max()), np.unravel_index(ratio.argmax(), ratio.shape))


@pytest.mark.skipif(os.environ.get("CUDA_VISIBLE_DEVICES") == "", reason="CUDA required for core bf16 ops")
@pytest.mark.parametrize("op", [*CORE, "div_constant"])
def test_gpu_bf16_core_gradients(op):
    # The source census covers the sealed core set. Each registered op has a
    # bf16 kernel path, so even an explicit refusal is a regression here.
    case = small(op)
    xs = tuple(tc.tensor(a, dtype="bfloat16", requires_grad=True) for a in case.operands)
    if op == "div_constant":
        divisor = tc.tensor(np.float32(case.attrs["divisor"]), dtype="bfloat16")
        out = xs[0] / divisor
        case = replace(case, attrs={"divisor": divisor.numpy().item()})
    else:
        out = forward_tc(case, *xs)
    assert out.requires_grad and out.dtype == "bfloat16"
    seed = tc.tensor(case.seed, dtype="bfloat16")
    assert out.shape == seed.shape
    (out * seed).sum().backward()
    # NumPy receives the stored bf16 values widened exactly to fp32. Widen
    # again before evaluating the analytic VJP, including the upstream seed.
    operands = tuple(x.numpy().astype(np.float64) for x in xs)
    upstream = seed.numpy().astype(np.float64)
    reference, terms = vjp(case, operands, upstream)
    if op == "snake":
        phase = operands[0] * operands[1]
        terms = (abs(upstream) * (1 + abs(np.sin(2 * phase)) + 2 * abs(phase)), terms[1])
    assert len(xs) == len(reference) == len(terms)
    for slot, (x, expected, scale) in enumerate(zip(xs, reference, terms)):
        assert x.grad is not None, f"{op}: requested bf16 gradient is absent"
        assert x.grad.dtype == "bfloat16" and x.grad.shape == x.shape
        assert_bf16_gradient(case, slot, x.grad.numpy(), expected, scale)
