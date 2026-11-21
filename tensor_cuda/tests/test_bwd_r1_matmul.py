"""FP32 precision witness, actual cuBLAS configuration and five-run equality."""

import hashlib
from dataclasses import replace
import os

import numpy as np
import pytest

import tensor_cuda as tc
from bwd_r1_support.oracle import Case, allowed, compare, kokoro, numerical, vjp
from test_bwd_r1_core import device_gradients, gpu


def small_canary():
    # Exact fp32 increments lie below the round-to-nearest TF32 midpoint.
    # Use different powers of two and a signed, nonuniform seed.
    a = (2. ** (np.arange(35).reshape(5, 7) % 3)) * (1. + 3 * 2. ** -13)
    w = (2. ** (np.arange(56).reshape(7, 8) % 3)) * (1. + 3 * 2. ** -13)
    s = np.array([1, 2, 1, -1, 2, 1, -1, 1])[None, :] * (1. + 3 * 2. ** -13)
    g = (np.arange(5).reshape(5, 1) + 1) * s
    return Case("matmul", "TF32_canary", tuple(np.asarray(x, dtype=np.float32) for x in (a, w)),
                g.astype(np.float32), {"trans_b": False, "alpha": 1.})


def canary():
    # Align all dimensions for Ampere/Ada tensor-core dispatch. Use powers
    # of two so every nonzero operand keeps the same TF32 rounding witness.
    size = 128
    i, j = np.indices((size, size))
    increment = np.float32(1. + 3 * 2. ** -13)
    a = np.ascontiguousarray(np.exp2((i + j) % 3) * increment, dtype=np.float32)
    w = np.ascontiguousarray(np.exp2((i + 2 * j) % 3) * increment, dtype=np.float32)
    g = np.ascontiguousarray(np.exp2((2 * i + j) % 3) * increment, dtype=np.float32)
    return Case("matmul", "TF32_canary", (a, w), g, {"trans_b": False, "alpha": 1.})


# Prior art: NVIDIA Ampere TF32 (2020) and cuBLAS strided-batched GEMM
# (CUDA 8, 2016). The second matrix keeps the r3 rounding witness while
# changing the input, weight and seed bytes at each batch stride.
def strided_canary():
    single = canary()
    a, w = single.operands
    g = single.seed
    return replace(
        single, label="TF32_strided_canary",
        operands=(np.stack((a, np.roll(a, 1, axis=0) * .5)),
                  np.stack((w, np.roll(w, 1, axis=1) * 2.))),
        seed=np.stack((g, np.roll(g, 1, axis=0) * 2.)),
    )


# Prior art: NVIDIA Ampere TF32 (2020), fp32 exponent and ten fraction bits.
# Ours: an RNE witness built from the same operand bytes as the candidate case.
def to_tf32(x):
    bits = x.view(np.uint32)
    return ((bits + np.uint32(0xFFF) + ((bits >> 13) & 1)) & np.uint32(0xFFFFE000)).view(np.float32)


def canary_reference(case):
    operands = (*case.operands, case.seed)
    fp64, terms = vjp(case, tuple(x.astype(np.float64) for x in operands[:2]), operands[-1].astype(np.float64))
    tf64 = tuple(to_tf32(x).astype(np.float64) for x in operands)
    rounded, _ = vjp(case, tf64[:2], tf64[-1])
    for i, (a, b, s) in enumerate(zip(fp64, rounded, terms)):
        numerical(a, b)
        ratio = float((abs(a - b) / allowed(case, i, a, s)).max())
        assert ratio >= 4.0, (i, ratio)
        print("TF32 sensitivity bound ratio", i, ratio, flush=True)
    return fp64, terms


def check_canary(case, capfd, unmasked):
    override = os.environ.get("NVIDIA_TF32_OVERRIDE")
    print("canary NVIDIA_TF32_OVERRIDE", repr(override), flush=True)
    if unmasked:
        assert override is None
    fp64, terms = canary_reference(case)
    try:
        got = device_gradients(case)
    finally:
        captured = capfd.readouterr()
        print(captured.out, end="", flush=True)
        print(captured.err, end="", flush=True)
    rows = [line for line in captured.err.splitlines() if line.startswith("bwd_r1_gemm")]
    if os.environ.get("TC_MATMUL_BWD_LOG_CONFIG"):
        expected = ("stream_legacy=1 atomics_disabled=1 pedantic_math=1 host_pointer=1 "
                    "workspace_bytes=33554432 pedantic_compute=1 default_algo=1 ")
        assert len(rows) == 2 and all(expected in row for row in rows), rows
        print("TF32 canary strict configuration observed", flush=True)
    compare(case, got, fp64, terms)


@gpu
def test_gpu_canary_and_configuration(capfd):
    check_canary(canary(), capfd, os.environ.get("TC_BWD_R3_EXPECT_CANARY_UNMASKED") == "1")


@pytest.mark.skipif(os.environ.get("NVIDIA_TF32_OVERRIDE") is not None,
                    reason="strided TF32 numeric witness requires NVIDIA_TF32_OVERRIDE unset")
@gpu
def test_gpu_strided_canary(capfd):
    case = strided_canary()
    assert all(arr.shape == (2, 128, 128) for arr in (*case.operands, case.seed))
    check_canary(case, capfd, unmasked=True)


@pytest.mark.parametrize("cell", ["small", "kokoro_largest", "kokoro_split_k", "strided"])
def test_gpu_matmul_repeat_five(cell, capfd):
    if cell == "small":
        case = small_canary()
    elif cell == "strided":
        case = strided_canary()
    else:
        case = kokoro("matmul")
        # The census's sole Kokoro matmul is itself split-K-prone on dA.
        # Retain its exact capture and repeat that same geometry with the
        # registered grid's negative scale as a separate stress cell.
        assert case.operands[1].shape == (1028, 128) and case.seed.shape == (1, 1028)
        if cell == "kokoro_split_k":
            case = replace(case, attrs={**case.attrs, "alpha": -.625})
    first = None
    for step in range(5):
        got = tuple(a.view(np.uint32) for a in device_gradients(case))
        assert all(a.size and np.ptp(a) for a in got)
        print("repeat", cell, step + 1, *(hashlib.sha256(a.tobytes()).hexdigest() for a in got), flush=True)
        if first is None:
            first = got
        else:
            for actual, expected in zip(got, first):
                np.testing.assert_array_equal(actual, expected)
    captured = capfd.readouterr()
    print(captured.out + captured.err, end="", flush=True)
    if os.environ.get("TC_MATMUL_BWD_LOG_CONFIG"):
        rows = [line for line in captured.err.splitlines() if line.startswith("bwd_r1_gemm")]
        assert len(rows) == 10
        for row in rows:
            values = dict(item.split("=") for item in row.split()[1:])
            assert int(values["cublas_version"]) > 0 and int(values["sm_count"]) > 0


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@gpu
def test_gpu_lower_precision_fallback(dtype):
    a = tc.tensor([[1, -2], [3, 4]], dtype=dtype, requires_grad=True)
    b = tc.tensor([[2, 1], [-1, 3]], dtype=dtype, requires_grad=True)
    tc.matmul(a, b).backward()
    for operand, expected in [(a, [[3, 2], [3, 2]]), (b, [[4, 4], [2, 2]])]:
        assert operand.grad.dtype == dtype
        actual = operand.grad.numpy()
        numerical(actual, np.asarray(expected))
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(("a_batch", "b_batch"), [((), (2,)), ((2, 1), (2,)), ((2,), (1, 2))])
@gpu
def test_gpu_refuses_forward_first_batch_only(a_batch, b_batch):
    a = tc.tensor(np.ones((*a_batch, 2, 3)), requires_grad=True)
    b = tc.tensor(np.ones((*b_batch, 3, 4)), requires_grad=True)
    output = tc.matmul(a, b)
    with pytest.raises(RuntimeError, match="forward uses stride 0 and only the first B matrix"):
        output.backward()
    assert a.grad is None and b.grad is None and output.grad is None


@pytest.mark.parametrize("trans_b", [False, True])
@gpu
def test_gpu_unequal_rank_singleton_b_broadcast(trans_b):
    x = np.arange(1, 13, dtype=np.float32).reshape(2, 2, 3)
    b = np.arange(1, 13, dtype=np.float32).reshape(3, 4)
    if trans_b:
        b = np.ascontiguousarray(b.T)
    shape = (1, 1, *b.shape)
    # A has rank 3 and B rank 4 with only singleton batch dimensions.
    g = np.arange(1, 17, dtype=np.float32).reshape(2, 2, 4)
    case = Case("matmul", "singleton_B", (x, b), g, {"trans_b": trans_b, "alpha": 1.})
    a_dev = tc.tensor(x, requires_grad=True)
    b_dev = tc.tensor(b.reshape(shape), requires_grad=True)
    (tc.matmul(a_dev, b_dev, trans_b=trans_b) * tc.tensor(g)).sum().backward()
    assert b_dev.grad.shape == shape
    compare(case, (a_dev.grad.numpy(), b_dev.grad.numpy().reshape(b.shape)))
