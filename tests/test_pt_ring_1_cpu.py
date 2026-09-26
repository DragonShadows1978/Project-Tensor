"""PT-RING-1 CPU author baseline; no CUDA calls or GPU pass substitution.

Prior art: shared C ABI contract testing (ctypes, Python 2006), independent
pairwise interval oracle, PT-DET/PT-TF32 fail-closed receipts (2026).
Taken: test mechanisms; ours: this order's shape/alias and measurement cases.
"""
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import pt_ring_1 as ring


@pytest.fixture(scope="module")
def contract(tmp_path_factory):
    out = tmp_path_factory.mktemp("ring_cpu_contract")/"contract.so"
    cmd = ["c++", "-std=c++17", "-O2", "-shared", "-fPIC", "-I"+str(ROOT/"tensor_cuda/include"),
           str(ROOT/"tests/pt_ring_1_contract.cpp"), "-o", str(out)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES="", TMPDIR=str(out.parent)))
    assert result.returncode == 0, result.stdout+result.stderr
    lib = ctypes.CDLL(str(out))
    lib.ring_bytes.argtypes = [ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t, ctypes.c_int, ctypes.POINTER(ctypes.c_size_t)]
    lib.ring_alias.argtypes = [ctypes.POINTER(ctypes.c_size_t)]*3+[ctypes.c_size_t]
    lib.ring_error.restype = ctypes.c_char_p
    return lib


@pytest.mark.parametrize("dtype,item", [(0, 4), (1, 2), (2, 8), (3, 1), (4, 1), (5, 2)])
@pytest.mark.parametrize("shape,count", [((), 1), ((19,), 19), ((3, 7), 21), ((2, 3, 5), 30), ((2, 3, 4, 7), 168), ((2, 0, 3), 0)])
def test_actual_shared_checked_byte_count(contract, dtype, item, shape, count):
    dims = (ctypes.c_int64*len(shape))(*shape); result = ctypes.c_size_t()
    assert contract.ring_bytes(dims, len(shape), dtype, ctypes.byref(result)) == 0
    assert result.value == count*item


@pytest.mark.parametrize("shape,dtype", [((-1,), 0), ((0, -1), 0), ((1,)*9, 0), ((2**62, 2), 0),
                                        ((0, 2**62, 2), 2), ((1,), 127)])
def test_overflow_negative_rank_dtype_fail_closed(contract, shape, dtype):
    dims = (ctypes.c_int64*len(shape))(*shape); result = ctypes.c_size_t()
    assert contract.ring_bytes(dims, len(shape), dtype, ctypes.byref(result)) == 1


def test_actual_interval_validation_against_independent_quadratic_oracle(contract):
    rng = random.Random(329)
    for trial in range(2000):
        count = rng.randrange(0, 12)
        dst = [rng.randrange(1, 100) for _ in range(count)]
        src = [rng.randrange(1, 100) for _ in range(count)]
        size = [rng.randrange(0, 16) for _ in range(count)]
        if trial % 3 == 0: src = dst[:]
        def overlaps(a, na, b, nb): return na > 0 and nb > 0 and max(a, b) < min(a+na, b+nb)
        expected = any(overlaps(dst[i], size[i], dst[j], size[j]) for i in range(count) for j in range(i))
        expected |= any(overlaps(dst[i], size[i], src[j], size[j]) and not (i == j and dst[i] == src[j])
                        for i in range(count) for j in range(count))
        args = [(ctypes.c_size_t*count)(*a) for a in (dst, src, size)]
        assert bool(contract.ring_alias(*args, count)) == expected, (dst, src, size, expected)


def test_python_api_and_invalid_contracts_without_cuda():
    code = '''from pathlib import Path
import numpy as np
import tensor_cuda as tc
assert Path(tc._C.__file__).resolve().is_relative_to(Path.cwd())
names = ["Stream", "Event", "PinnedBuffer", "copy_many_", "copy_to_host_async", "legacy_stream", "mem_get_info", "empty_like"]
assert all(hasattr(tc, n) and n in tc.__all__ for n in names)
assert hasattr(tc.Tensor, "copy_")
assert tc.pinned_bytes() == 0 and tc.collect_async_copies() == 0
for dtype in ["float32", "float16", "bfloat16", "int64", "bool", "uint8"]:
    b = tc.pinned_empty((3, 0, 2), dtype)
    assert b.shape == (3, 0, 2) and b.nbytes == 0 and b.numpy().nbytes == 0
    assert np.asarray(b).dtype == np.dtype("uint16" if dtype == "bfloat16" else dtype)
for shape in [(-1,), (0, -1), (2**62, 2)]:
    try: tc.pinned_empty(shape, "int64")
    except (ValueError, OverflowError): pass
    else: raise AssertionError("invalid shape accepted")
old = tc.pinned_memory_limit()
tc.set_pinned_memory_limit(0)
try:
    tc.pinned_empty((1,), "uint8")
except RuntimeError as e: assert "no allocation attempted" in str(e)
else: raise AssertionError("budget failed")
tc.set_pinned_memory_limit(old)
try: tc.copy_many_([], [])
except RuntimeError as e: assert "no_grad" in str(e)
else: raise AssertionError("no_grad guard failed")
with tc.no_grad():
    try: tc.copy_many_([], [], method="bad")
    except ValueError: pass
    else: raise AssertionError("invalid method accepted")
    cpu = tc._C.tensor(np.empty((0,), np.float32), "cpu", False)
    try: cpu.copy_(cpu)
    except ValueError as e: assert "CUDA tensor" in str(e)
    else: raise AssertionError("CPU operands accepted")
    a = tc._C._ring_empty((0,), "float32")
    b = tc._C._ring_empty((0, 1), "float32")
    try: a.copy_(b)
    except ValueError as e: assert "shape/dtype" in str(e)
    else: raise AssertionError("mismatched empty shapes accepted")
print("PT_RING_1 API/empty/invalid contracts passed; no CUDA calls")
'''
    r = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=15,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT/"tensor_cuda"), PYTHONDONTWRITEBYTECODE="1"))
    assert r.returncode == 0, r.stdout+r.stderr


def test_original_kernel_bytes_and_unchanged_engine_sources():
    registration = json.loads((ring.ART/"REGISTRATION.json").read_text())
    before = (ring.ART/"baseline/tensor_cuda/src/kernels.cu").read_bytes()
    now = (ROOT/"tensor_cuda/src/kernels.cu").read_bytes()
    assert now[:len(before)] == before
    allowed = {"tensor_cuda/src/kernels.cu", "tensor_cuda/src/bindings.cpp", "tensor_cuda/tensor_cuda/__init__.py"}
    for path, digest in registration["baseline"].items():
        if path not in allowed: assert ring.sha(ROOT/path) == digest, path
    assert ring.sha(ROOT/registration["order"]) == registration["order_sha256"]


def test_exact_checkpoint_schema_is_nonvacuous():
    g = json.loads((ring.ART/"GEOMETRY.json").read_text())
    assert g["count"] == len(g["tensors"]) == 1110
    assert g["bytes"] == sum(r["bytes"] for r in g["tensors"]) == 2_939_091_552
    assert len(g["source_sha256"]) == 64 and g["step"] == 32055


def overlap_fixture():
    return dict(bytes=2_940_000_000, calibrated_compute_ms=500., samples=[dict(solo_wall_ms=500., concurrent_wall_ms=510.,
                compute_start_ms=0., compute_end_ms=500., copy_start_ms=1., copy_end_ms=140.) for _ in range(7)])


@pytest.mark.parametrize("fault", ["size", "empty", "one", "serialized", "zero", "slow", "nan", "missing", "too_short", "boolean"])
def test_overlap_gate_kills_false_positive_receipts(fault):
    r = overlap_fixture(); assert ring.assess_overlap(r) == "GREEN"
    if fault == "size": r["bytes"] = 1
    if fault == "empty": r["samples"] = []
    if fault == "one": r["samples"] = r["samples"][:1]
    if fault == "serialized": r["samples"][0].update(copy_start_ms=501., copy_end_ms=600.)
    if fault == "zero": r["samples"][0]["solo_wall_ms"] = 0
    if fault == "slow":
        for row in r["samples"]: row["concurrent_wall_ms"] = 516.
    if fault == "nan": r["samples"][0]["copy_end_ms"] = float("nan")
    if fault == "missing": del r["samples"][0]["compute_end_ms"]
    if fault == "too_short": r["calibrated_compute_ms"] = 399
    if fault == "boolean": r["samples"][0]["solo_wall_ms"] = True
    assert ring.assess_overlap(r) == "RED"


@pytest.mark.parametrize("fault", ["count", "size", "slots", "slow_host", "slow_device", "missing_method", "nan", "empty", "corrupt", "pageable", "too_short"])
def test_ring_gate_kills_incomplete_or_slow_receipts(fault):
    r = dict(count=1110, bytes=2_939_091_552, slots=3, pinned_bytes=8_817_274_656, compute_ms=500., host_slots_bitwise=True,
             methods={m:dict(host_ms=[20.]*7, legacy_ms=[25.]*7) for m in ("memcpy", "kernel")})
    assert ring.assess_ring(r) == "GREEN"
    if fault == "count": r["count"] = 1109
    if fault == "size": r["bytes"] -= 1
    if fault == "slots": r["slots"] = 1
    if fault == "slow_host": r["methods"]["memcpy"]["host_ms"] = [51.]*7
    if fault == "slow_device": r["methods"]["memcpy"]["legacy_ms"] = [51.]*7
    if fault == "missing_method": del r["methods"]["kernel"]
    if fault == "nan": r["methods"]["kernel"]["legacy_ms"][0] = float("nan")
    if fault == "empty": r["methods"]["kernel"]["host_ms"] = []
    if fault == "corrupt": r["host_slots_bitwise"] = False
    if fault == "pageable": r["pinned_bytes"] = 0
    if fault == "too_short": r["compute_ms"] = 399
    assert ring.assess_ring(r) == "RED"


def test_guard_runs_before_cuda_or_output_access(tmp_path):
    out = tmp_path/"must_not_exist"
    r = subprocess.run([sys.executable, "-B", str(ROOT/"scripts/pt_ring_1.py"), "--lead-gpu", "--out", str(out)],
                       capture_output=True, text=True, timeout=10, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    assert r.returncode == 2 and "BLOCKED" in r.stdout and not out.exists()


def test_receipts_are_create_only(tmp_path):
    out = tmp_path/"receipt.json"
    ring.write_json(out, {"status":"BLOCKED"})
    with pytest.raises(FileExistsError): ring.write_json(out, {"status":"GREEN"})
    assert json.loads(out.read_text())["status"] == "BLOCKED"
