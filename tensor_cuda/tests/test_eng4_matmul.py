"""ENG4: the matmul forward refuses every operand pair it cannot compute, and numpy_dtype refuses unlisted dtypes.

G1 (CPU, CUDA_VISIBLE_DEVICES empty, host operands, public Python API):
  * every M3 class refuses with RuntimeError naming both shapes (test_g1_m3_refuses);
  * every M4 size class refuses (test_g1_m4_size_refuses), built from zero-element operands;
  * the M4 device class refuses (test_g1_m4_device_refuses): its CUDA operand has zero elements, so building it makes
    no CUDA call (Storage and from_host return early for 0 bytes);
  * the accepted controls are NOT refused: with no visible GPU they reach the forward's cuBLAS call and fail there
    with the CUDA error, as on Build B (test_g1_accepted_controls_reach_the_gemm);
  * N1: tc._C.tensor (the binding of C++ tensor_from_numpy) and tc.from_numpy refuse float64, int32 and uint16
    (test_g1_n1_refuses_unlisted), and every listed dtype keeps Build B's result
    (test_g1_n1_listed_dtypes_keep_build_b_result, against artifacts/eng4/n1/B_cpu.json, recorded by
    scripts/eng4_n1_record.py on Build B in its own process).
GPU tests (skipped with CUDA_VISIBLE_DEVICES empty; staged as GPU lines): the refusals with real CUDA operands, the
mixed host/CUDA device refusal with nonzero operands (m02's GPU line), and no output or gradient after a refusal.

Prior art: NumPy's matmul shape rule (numpy.matmul docs) for which pairs are meaningful; the cuBLAS strided-batched
GEMM contract (cuBLAS docs) for why a B batch smaller than A's reads out of bounds; mutation testing (DeMillo,
Lipton and Sayward, 1978) for the kill assertions named in artifacts/eng4/MUTANTS.json. Ours: the witness table.
"""
import json
import os
import pathlib
import re

import numpy as np
import pytest

import tensor_cuda as tc

ROOT = pathlib.Path(__file__).resolve().parents[2]
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
ACCEPTED_CONTROLS = [
    ("a_b_rank2", (4, 2, 3), (3, 4), False),
    ("b_equal_batch", (4, 2, 3), (4, 3, 4), False),
    ("b_equal_batch_rank4_trans", (2, 3, 2, 3), (2, 3, 4, 3), True),
    ("b_rank2_rank2", (2, 3), (3, 4), False),
    ("c_ones_b_rank4_a3", (2, 2, 3), (1, 1, 3, 4), False),
    ("c_ones_b_rank3_a2", (2, 3), (1, 3, 4), False),
    ("M_eq_intmax", (MAXI, 0), (0, 0), False),
    ("N_eq_intmax", (0, 0), (0, MAXI), False),
    ("K_eq_intmax", (0, MAXI), (MAXI, 0), False),
    ("batch_eq_intmax", (MAXI, 0, 0), (0, 0), False),
]


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
    """Accepted pairs pass the predicate and reach the forward's cuBLAS call (no GPU: the CUDA error after it)."""
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


def test_g1_n1_listed_dtypes_keep_build_b_result():
    import importlib.util
    spec = importlib.util.spec_from_file_location("eng4_n1_record", ROOT / "scripts" / "eng4_n1_record.py")
    rec_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rec_mod)
    ref = json.loads((ROOT / "artifacts/eng4/n1/B_cpu.json").read_text())
    assert ref["label"] == "B_cpu" and ref["seed"] == rec_mod.SEED
    got = rec_mod.record(tc)
    assert sorted(got) == sorted(ref["rows"]) == ["bool", "float16", "float32", "int64", "uint8"]
    for name, row in got.items():
        raw = bytes.fromhex(row["out_hex"])
        assert len(raw) and len(set(raw)) > 1, name  # L19: nonzero, non-constant bytes
        assert row == ref["rows"][name], name


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
