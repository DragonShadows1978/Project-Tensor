"""ENG4: the matmul forward refuses every operand pair it cannot compute, and numpy_dtype refuses unlisted dtypes.

G1 (CPU, CUDA_VISIBLE_DEVICES empty, host operands, public Python API):
  * every M3 class refuses with RuntimeError naming both shapes (test_g1_m3_refuses);
  * every M4 size class refuses (test_g1_m4_size_refuses), built from zero-element operands;
  * the M4 device class refuses (test_g1_m4_device_refuses): its CUDA operand has zero elements, so building it makes
    no CUDA call (Storage and from_host return early for 0 bytes);
  * the accepted controls are NOT refused: with no visible GPU a nonempty, K > 0 pair reaches the forward's cuBLAS
    call and fails there with the CUDA error, as on Build B (test_g1_accepted_controls_reach_the_gemm); since
    ENG4-F1 X2 a zero-size pair returns NumPy's result without a GEMM (test_h1_zero_size_returns_numpy_result);
  * N1: tc._C.tensor (the binding of C++ tensor_from_numpy) and tc.from_numpy refuse float64, int32 and uint16
    (test_g1_n1_refuses_unlisted), and every listed dtype returns the array's own bytes
    (test_g1_n1_listed_dtypes_keep_their_bytes; ENG4-F1 X4: computed here, no file outside this test is read).
ENG4-F1 H1 (CPU): int64/uint8/bool pairs refuse before any CUDA call (test_h1_unsupported_dtype_refuses);
'>f4', '>f2' and '>i8' refuse on tc.from_numpy and _C.tensor (test_h1_byteorder_refuses); tc.tensor of a '>f4' array
keeps its values (test_h1_tensor_casts_byteorder).
GPU tests (skipped with CUDA_VISIBLE_DEVICES empty; staged as GPU lines): the refusals with real CUDA operands, the
mixed host/CUDA device refusal with nonzero operands (m02's GPU line), the zero-size results on the device (m08's GPU
line: K = 0 must give zeros), and no output or gradient after a refusal.

This file is self-contained (ENG4-F1 X4): it imports only os, re, numpy, pytest and tensor_cuda, and reads no file.

Prior art: NumPy's matmul shape rule (numpy.matmul docs) for which pairs are meaningful; the cuBLAS strided-batched
GEMM contract (cuBLAS docs) for why a B batch smaller than A's reads out of bounds; mutation testing (DeMillo,
Lipton and Sayward, 1978) for the kill assertions named in artifacts/eng4/MUTANTS.json and f1/MUTANTS_F1.json;
NumPy's matmul on zero-size operands and dtype.isnative (ENG4-F1); ENG3-F5's self-contained test pattern (this repo).
Ours: the witness table.
"""
import os
import re

import numpy as np
import pytest

import tensor_cuda as tc

CPU_LANE = os.environ.get("CUDA_VISIBLE_DEVICES") == ""
gpu = pytest.mark.skipif(CPU_LANE, reason="CUDA disabled for CPU lane")
cpu_lane = pytest.mark.skipif(not CPU_LANE, reason="host-operand GEMM controls run only with no visible GPU (L58)")

BIG = 2 ** 31       # INT_MAX + 1
MAXI = 2 ** 31 - 1  # INT_MAX

# (id, A shape, B shape, trans_b)
M3_CASES = [
    ("eq_rank_smaller_b_batch_ENG-MM1", (4, 2, 3), (2, 3, 4), False),
    ("eq_rank_smaller_b_batch_trans", (4, 2, 3), (2, 4, 3), True),
    ("eq_rank_larger_b_batch", (2, 2, 3), (4, 3, 4), False),
    ("eq_rank_larger_b_batch_a_ones", (1, 2, 3), (2, 3, 4), False),
    ("eq_count_diff_shape", (2, 3, 2, 3), (3, 2, 3, 4), False),
    ("eq_rank_ones_b_under_larger_a", (2, 2, 3), (1, 3, 4), False),
    ("eq_rank_ones_b_under_larger_a_rank4", (2, 3, 2, 3), (1, 1, 3, 4), False),
    ("uneq_rank_a2_b3_batched", (2, 3), (2, 3, 4), False),
    ("uneq_rank_a4_b3_batched", (2, 1, 2, 3), (2, 3, 4), False),
    ("uneq_rank_a3_b4_partial_ones", (2, 2, 3), (1, 2, 3, 4), False),
    ("uneq_rank_a3_b4_partial_ones_trans", (2, 2, 3), (1, 2, 4, 3), True),
]
M4_SIZE_CASES = [
    ("M_over_intmax", (BIG, 0), (0, 0), False),
    ("N_over_intmax", (0, 0), (0, BIG), False),
    ("N_over_intmax_trans", (0, 0), (BIG, 0), True),
    ("K_over_intmax", (0, BIG), (BIG, 0), False),
    ("K_over_intmax_trans", (0, BIG), (0, BIG), True),
    ("batch_over_intmax_b2d", (BIG, 0, 0), (0, 0), False),
    ("batch_over_intmax_eq_rank", (2 ** 16, 2 ** 15, 0, 0), (2 ** 16, 2 ** 15, 0, 0), False),
]
# (id, A shape, A device, B shape, B device)
M4_DEVICE_CASES = [
    ("host_a_cuda_b", (2, 0), "cpu", (0, 3), "cuda"),
    ("cuda_a_host_b", (2, 0), "cuda", (0, 3), "cpu"),
]
ACCEPTED_CONTROLS = [  # nonempty outputs with K > 0: they reach the GEMM
    ("a_b_rank2", (4, 2, 3), (3, 4), False),
    ("b_equal_batch", (4, 2, 3), (4, 3, 4), False),
    ("b_equal_batch_rank4_trans", (2, 3, 2, 3), (2, 3, 4, 3), True),
    ("b_rank2_rank2", (2, 3), (3, 4), False),
    ("c_ones_b_rank4_a3", (2, 2, 3), (1, 1, 3, 4), False),
    ("c_ones_b_rank3_a2", (2, 3), (1, 3, 4), False),
]
# ENG4-F1 X2: zero-size accepted pairs (incl. the INT_MAX boundary, which ENG4 listed as accepted controls) return
# NumPy's result without a GEMM. (id, class, A shape, B shape, trans_b); class c's shape is A's batch + [M, N] (M2(c)).
ZERO_CASES = [
    ("M_eq_intmax", "a", (MAXI, 0), (0, 0), False),
    ("N_eq_intmax", "a", (0, 0), (0, MAXI), False),
    ("K_eq_intmax", "a", (0, MAXI), (MAXI, 0), False),
    ("batch_eq_intmax", "a", (MAXI, 0, 0), (0, 0), False),
    ("a_M0", "a", (3, 0, 5), (5, 4), False),
    ("a_N0", "a", (3, 2, 5), (5, 0), False),
    ("a_N0_trans", "a", (3, 2, 5), (0, 5), True),
    ("a_K0", "a", (3, 2, 0), (0, 4), False),
    ("a_K0_trans", "a", (3, 2, 0), (4, 0), True),
    ("a_batch0", "a", (0, 2, 5), (5, 4), False),
    ("b_M0", "b", (2, 3, 0, 5), (2, 3, 5, 4), False),
    ("b_N0", "b", (2, 3, 2, 5), (2, 3, 5, 0), False),
    ("b_K0", "b", (2, 3, 2, 0), (2, 3, 0, 4), False),
    ("b_batch0", "b", (0, 3, 2, 5), (0, 3, 5, 4), False),
    ("b_rank2_K0", "b", (2, 0), (0, 4), False),
    ("c_M0", "c", (3, 0, 5), (1, 1, 5, 4), False),
    ("c_N0", "c", (3, 2, 5), (1, 1, 5, 0), False),
    ("c_K0", "c", (3, 2, 0), (1, 1, 0, 4), False),
    ("c_batch0", "c", (0, 2, 5), (1, 1, 5, 4), False),
]
ZERO_DTYPES = ("float32", "float16", "bfloat16")


def shape_text(shape):
    return "[" + ", ".join(str(int(d)) for d in shape) + "]"


def names_both(sa, sb, da="cpu", db="cpu"):
    """The refusal names both shapes (and devices) in the predicate's operand text."""
    return (re.escape(f"A {shape_text(sa)} on ") + r"\S+" + re.escape(", B " + shape_text(sb) + " on ") + r"\S+")


def host(shape, device="cpu"):
    return tc.tensor(np.zeros(shape, np.float32), device=device)


@pytest.mark.parametrize("case", M3_CASES, ids=[c[0] for c in M3_CASES])
def test_g1_m3_refuses(case):
    _, sa, sb, tb = case
    a, b = host(sa), host(sb)
    # Named assertion for m01 (eq_count_diff_shape) and m04 (eq_rank_ones_b_under_larger_a*): MUTANTS.json.
    with pytest.raises(RuntimeError, match=r"^matmul refused: .*batch.*\(" + names_both(sa, sb)):
        tc.matmul(a, b, trans_b=tb)


@pytest.mark.parametrize("case", M4_SIZE_CASES, ids=[c[0] for c in M4_SIZE_CASES])
def test_g1_m4_size_refuses(case):
    _, sa, sb, tb = case
    a, b = host(sa), host(sb)
    assert 0 in a.shape and 0 in b.shape  # zero-element operands: nothing large is allocated
    # Named assertion for m03: MUTANTS.json.
    with pytest.raises(RuntimeError, match=r"^matmul refused: GEMM (dimension|batch count) above INT_MAX.*\("
                       + names_both(sa, sb)):
        tc.matmul(a, b, trans_b=tb)


@pytest.mark.parametrize("case", M4_DEVICE_CASES, ids=[c[0] for c in M4_DEVICE_CASES])
def test_g1_m4_device_refuses(case):
    _, sa, da, sb, db = case
    a, b = host(sa, da), host(sb, db)
    assert str(a.device).startswith(da) and str(b.device).startswith(db)
    # Named assertion for m02 (CPU witness; the GPU line uses nonzero operands): MUTANTS.json.
    with pytest.raises(RuntimeError, match=r"^matmul refused: operands are on different devices \("
                       + names_both(sa, sb, da, db)):
        tc.matmul(a, b)


@cpu_lane
@pytest.mark.parametrize("case", ACCEPTED_CONTROLS, ids=[c[0] for c in ACCEPTED_CONTROLS])
def test_g1_accepted_controls_reach_the_gemm(case):
    """Nonempty K > 0 accepted pairs pass the predicate and reach the forward's cuBLAS call (no GPU: the CUDA error
    after it)."""
    _, sa, sb, tb = case
    a, b = host(sa), host(sb)
    with pytest.raises(RuntimeError) as err:
        tc.matmul(a, b, trans_b=tb)
    assert str(err.value).startswith("CUDA error at matmul: "), str(err.value)


UNLISTED = [np.float64, np.int32, np.uint16]


@pytest.mark.parametrize("entry", ["_C.tensor", "from_numpy"])
@pytest.mark.parametrize("dtype", UNLISTED, ids=[np.dtype(d).name for d in UNLISTED])
def test_g1_n1_refuses_unlisted(dtype, entry):
    arr = (np.arange(6).reshape(2, 3) + 1).astype(dtype)
    name = np.dtype(dtype).name
    call = (lambda: tc._C.tensor(arr, "cpu", False)) if entry == "_C.tensor" else (
        lambda: tc.from_numpy(arr, device="cpu"))
    # Named assertion for m05: MUTANTS.json.
    with pytest.raises(TypeError, match=r"unsupported NumPy dtype " + re.escape(name)):
        call()


def listed_inputs():
    """Seeded arrays of every listed dtype (seed 4400); float rows include NaN, +-inf, -0.0 and a subnormal."""
    rng = np.random.default_rng(4400)
    special = [np.nan, np.inf, -np.inf, -0.0, 0.0, 1.0, -2.5]
    f32 = np.concatenate([rng.standard_normal(17), special, [np.float32(1e-45)]]).astype(np.float32).reshape(5, 5)
    f16 = np.concatenate([rng.standard_normal(17), special, [6e-8]]).astype(np.float16).reshape(5, 5)
    i64 = rng.integers(-2 ** 62, 2 ** 62, size=(3, 4), dtype=np.int64)
    i64[0, 0], i64[0, 1] = np.iinfo(np.int64).min, np.iinfo(np.int64).max
    u8 = rng.integers(0, 256, size=(4, 6), dtype=np.uint8)
    u8[0, 0], u8[0, 1] = 0, 255
    bl = rng.integers(0, 2, size=(2, 7)).astype(np.bool_)
    bl[0, 0], bl[0, 1] = True, False
    return {"float32": f32, "float16": f16, "int64": i64, "uint8": u8, "bool": bl}


@pytest.mark.parametrize("entry", ["_C.tensor", "from_numpy"])
@pytest.mark.parametrize("name", ["float32", "float16", "int64", "uint8", "bool"])
def test_g1_n1_listed_dtypes_keep_their_bytes(name, entry):
    """ENG4-F1 X4: the expected result is the array's own bytes (no recorded file); Build B gave the same bytes
    (ENG4's n1/B_cpu.json record, not read here)."""
    arr = listed_inputs()[name]
    raw = arr.tobytes()
    assert len(raw) and len(set(raw)) > 1  # L19: nonzero, non-constant input bytes
    t = tc._C.tensor(arr, "cpu", False) if entry == "_C.tensor" else tc.from_numpy(arr, device="cpu")
    out = t.numpy()
    assert t.dtype == name and tuple(t.shape) == arr.shape and out.dtype == arr.dtype
    assert np.ascontiguousarray(out).tobytes() == raw


# ------------------------------------------------------------------------------------------- ENG4-F1 H1 (CPU)
UNSUPPORTED = ["int64", "uint8", "bool"]
# (id, A shape, A device, B shape, B device): host pairs (nonempty K > 0, K = 0 with a nonempty output, an empty
# output) and a zero-element CUDA pair with a nonempty output (its output allocation would be a CUDA call)
UNSUPPORTED_CASES = [
    ("host_nonempty", (2, 3), "cpu", (3, 4), "cpu"),
    ("host_k0", (2, 0), "cpu", (0, 4), "cpu"),
    ("host_empty_out", (0, 3), "cpu", (3, 4), "cpu"),
    ("host_batched_k0", (3, 2, 0), "cpu", (0, 4), "cpu"),
    ("cuda_k0", (2, 0), "cuda", (0, 4), "cuda"),
]


@pytest.mark.parametrize("case", UNSUPPORTED_CASES, ids=[c[0] for c in UNSUPPORTED_CASES])
@pytest.mark.parametrize("dtype", UNSUPPORTED)
def test_h1_unsupported_dtype_refuses(dtype, case):
    """X1: an int64, uint8 or bool pair is refused by the operand predicate, before any allocation or GEMM. With no
    visible GPU, any CUDA call would raise a different (CUDA) error, and a K = 0 or empty-output pair would otherwise
    return a result. Named assertion for m06_dtype_after_alloc (f1/MUTANTS_F1.json)."""
    _, sa, da, sb, db = case
    a = tc.tensor(np.zeros(sa), dtype=dtype, device=da)
    b = tc.tensor(np.zeros(sb), dtype=dtype, device=db)
    with pytest.raises(RuntimeError, match=r"^matmul supports float32/float16/bfloat16 only: operand dtype "
                       + dtype + r" \(" + names_both(sa, sb, da, db)):
        tc.matmul(a, b)


BYTEORDER = [">f4", ">f2", ">i8"]


@pytest.mark.parametrize("entry", ["_C.tensor", "from_numpy"])
@pytest.mark.parametrize("dtype", BYTEORDER)
def test_h1_byteorder_refuses(dtype, entry):
    """X3: a non-native byte order of a listed dtype is refused, naming it. Named assertion for m07_byteorder."""
    arr = (np.arange(6).reshape(2, 3) + 1).astype(dtype)
    assert not arr.dtype.isnative
    call = (lambda: tc._C.tensor(arr, "cpu", False)) if entry == "_C.tensor" else (
        lambda: tc.from_numpy(arr, device="cpu"))
    with pytest.raises(TypeError, match=r"non-native byte order NumPy dtype " + re.escape(dtype)):
        call()


def test_h1_tensor_casts_byteorder():
    """X3: the public tc.tensor casts first, so a '>f4' array keeps its values (bitwise, as native float32)."""
    rng = np.random.default_rng(4403)
    native = rng.standard_normal((3, 5)).astype(np.float32)
    big = native.astype(">f4")
    assert not big.dtype.isnative and np.array_equal(big, native)
    out = tc.tensor(big, device="cpu").numpy()
    assert out.dtype == np.float32 and out.tobytes() == native.tobytes() and len(set(native.tobytes())) > 1


def zero_expected(cls, sa, sb, tb):
    """NumPy's matmul shape for classes a and b; A's batch shape + [M, N] for class c (M2(c))."""
    b_logical = tuple(sb[:-2]) + ((sb[-1], sb[-2]) if tb else tuple(sb[-2:]))
    if cls in ("a", "b"):
        return tuple(np.matmul(np.zeros(sa, np.float32), np.zeros(b_logical, np.float32)).shape)
    return tuple(sa[:-1]) + (b_logical[-1],)


@pytest.mark.parametrize("dtype", ZERO_DTYPES)
@pytest.mark.parametrize("case", ZERO_CASES, ids=[c[0] for c in ZERO_CASES])
def test_h1_zero_size_returns_numpy_result(case, dtype):
    """X2 on host operands: no GEMM (with no visible GPU a GEMM would raise the CUDA error). The output has
    NumPy's shape (classes a, b) or A's batch + [M, N] (class c), and K = 0 gives positive zeros. Named assertion for
    m08_k0_gemm (CPU part; its GPU line uses CUDA operands)."""
    _, cls, sa, sb, tb = case
    a, b = host(sa), host(sb)
    if dtype != "float32":
        a, b = a.astype(dtype), b.astype(dtype)
    try:
        r = tc.matmul(a, b, trans_b=tb)
    except RuntimeError as e:
        pytest.fail(f"zero check: a zero-size matmul must return NumPy's result, it raised: {e}")
    want = zero_expected(cls, sa, sb, tb)
    assert tuple(r.shape) == want and r.dtype == dtype
    out = r.numpy()
    assert out.shape == want
    assert out.size == 0 or not np.any(out.view(np.uint16 if out.dtype == np.float16 else np.uint32)), \
        "zero check: K = 0 must give positive zeros"


# ------------------------------------------------------------------------------------------------- GPU lines
def dev(shape, seed, dtype="float32"):
    rng = np.random.default_rng(seed)
    return tc.tensor(rng.standard_normal(shape).astype(np.float32), dtype=dtype, requires_grad=True)


@gpu
@pytest.mark.parametrize("case", M3_CASES, ids=[c[0] for c in M3_CASES])
def test_gpu_m3_refuses_cuda_operands(case):
    _, sa, sb, tb = case
    a, b = dev(sa, 1), dev(sb, 2)
    with pytest.raises(RuntimeError, match=r"^matmul refused: .*batch.*\(" + names_both(sa, sb)):
        tc.matmul(a, b, trans_b=tb)
    assert a.grad is None and b.grad is None


@gpu
@pytest.mark.parametrize("order", ["cuda_a_host_b", "host_a_cuda_b"])
def test_gpu_m4_device_refuses_nonzero_operands(order):
    """m02's GPU line: one host and one CUDA operand, both nonzero."""
    rng = np.random.default_rng(7)
    xa, xb = rng.standard_normal((3, 5)).astype(np.float32), rng.standard_normal((5, 4)).astype(np.float32)
    da, db = ("cuda", "cpu") if order == "cuda_a_host_b" else ("cpu", "cuda")
    a, b = tc.tensor(xa, device=da), tc.tensor(xb, device=db)
    with pytest.raises(RuntimeError, match=r"^matmul refused: operands are on different devices \("
                       + names_both((3, 5), (5, 4), da, db)):
        tc.matmul(a, b)


@gpu
@pytest.mark.parametrize("dtype", ZERO_DTYPES)
@pytest.mark.parametrize("case", ZERO_CASES[4:], ids=[c[0] for c in ZERO_CASES[4:]])
def test_gpu_zero_size_returns_numpy_result(case, dtype):
    """X2 on CUDA operands (m08's GPU line: with the K = 0 early return dropped, the K = 0 cells reach cuBLAS with
    lda = 0 and must fail this zero check)."""
    _, cls, sa, sb, tb = case
    rng = np.random.default_rng(4404)
    a = tc.tensor(rng.standard_normal(sa).astype(np.float32), dtype=dtype)
    b = tc.tensor(rng.standard_normal(sb).astype(np.float32), dtype=dtype)
    try:
        r = tc.matmul(a, b, trans_b=tb)
        out = r.numpy()
    except RuntimeError as e:
        pytest.fail(f"zero check: a zero-size matmul must return NumPy's result, it raised: {e}")
    want = zero_expected(cls, sa, sb, tb)
    assert out.shape == want and r.dtype == dtype
    assert out.size == 0 or not np.any(out.view(np.uint16 if out.dtype == np.float16 else np.uint32)), \
        "zero check: K = 0 must give positive zeros"
