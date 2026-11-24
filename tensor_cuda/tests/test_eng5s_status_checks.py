"""ENG5-S-T (orders/ENG5ST_REGRESSION_TEST.md): a standalone production regression test for ENG5-S's nine status checks.

ENG5-S made nine CUDA calls check their own returned status (kernels.cu: cuda_sync, Storage's allocation retry and its
drain, NDArray::zeros, from_host, to_host, clone, to, axpy_; packed_prefill.cuh's bulk NaN fill). This file fails if any
of them is removed. It is the production counterpart of the fork lane's differential harness (test_eng5s_status.py,
which compares two receipted builds and is not part of production).

Self-contained: it imports os, re, json, shutil, subprocess, sys, numpy, pytest and the installed `tensor_cuda`, and
reads no build directory, receipt, artifact or script. It compiles the committed ENG5-S injection shim
(tests/support/eng5s_cuda_shim.c) at test time into a pytest temporary directory (compiler found as test_eng3_host.py
does: cc, gcc, clang; skip, naming the compiler, when there is none; skip, naming the headers, when the CUDA toolkit
headers the shim includes are missing). Every row runs in its own child interpreter with the shim preloaded, on the
same `tensor_cuda` package this pytest process imported.

Rows (each one asserts a status check from the injection's side: the injection must FIRE, and the call must RAISE
RuntimeError in the S1 form "<entry>: <call> failed: cudaError<Name>: <text>"):

CPU lane (CUDA_VISIBLE_DEVICES empty; skipped on the GPU lane):
- test_cpu_status_rows[synchronize_injected]   site 1: tc.synchronize() with cudaDeviceSynchronize injected to fail
- test_cpu_status_rows[synchronize_no_device]  site 1: tc.synchronize() with no device (the real runtime's failure)
- test_cpu_status_rows[oom_drain]              site 2: allocation fails with cudaErrorMemoryAllocation (cudaMalloc),
                                               the OOM retry's drain (cudaStreamSynchronize) is injected to fail
- test_cpu_status_rows[oom_drain_pooled]       site 2 through cudaMallocAsync (pooling on), the same two injections
GPU lane (a GPU visible; skipped on the CPU lane): each row runs an uninjected success control first (its result equals
a value computed on the host), then the injected call:
- test_gpu_status_rows[synchronize]            site 1: cudaDeviceSynchronize
- test_gpu_status_rows[oom_drain]              site 2: the OOM retry's drain after a cudaMalloc out-of-memory
- test_gpu_status_rows[oom_drain_pooled]       site 2 through cudaMallocAsync
- test_gpu_status_rows[zeros]                  site 3: NDArray::zeros' cudaMemset
- test_gpu_status_rows[from_host]              site 4: NDArray::from_host's cudaMemcpy
- test_gpu_status_rows[to_host]                site 5: NDArray::to_host's cudaMemcpy
- test_gpu_status_rows[clone]                  site 6: NDArray::clone's cudaMemcpy
- test_gpu_status_rows[to_d2h]                 site 7: NDArray::to, device to host
- test_gpu_status_rows[to_h2d]                 site 7: NDArray::to, host to device
- test_gpu_status_rows[axpy_]                  site 8: axpy_'s cudaMemcpy
- test_gpu_status_rows[axpy_empty]             control only: axpy_ on empty CUDA operands (0 bytes) returns, no raise
Site 9 (packed prefill's bulk NaN fill, reachable only with dump=True in a hooks build) is NOT in this production
test: its evidence stays the ENG5-S lead's g3_S and g3_B lines.

Prior art: LD_PRELOAD interposition with dlsym(RTLD_NEXT) and ENG3's call counter (cudart_count.c, this repo) and ENG5's
injection shim (this repo); library-level fault injection (LFI, Marinescu and Candea, 2009); mutation testing (DeMillo,
Lipton and Sayward, 1978) for the kill assertions (a row fails when its site's check is removed); ENG3-F5 and ENG4's
self-contained test pattern (this repo). Copied from ENG5-S (this repo): the S1 regular expression (test_eng5s_status.py),
the row bodies (scripts/eng5s_gpu_step.py G2_CHILD) and the entry/call prefixes (scripts/eng5s_gpu_verdict.py PREFIX).
Ours: the production packaging (compile at test time, one child per row, the lane split) and the assertions' form.
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys

import pytest

import tensor_cuda as tc

HERE = os.path.dirname(os.path.abspath(__file__))
SHIM = os.path.join(HERE, "support", "eng5s_cuda_shim.c")
# The package parent of the `tensor_cuda` this pytest process imported: the children import the same package.
PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(tc.__file__)))
CPU_LANE = os.environ.get("CUDA_VISIBLE_DEVICES") == ""
cpu_lane = pytest.mark.skipif(
    not CPU_LANE, reason="CPU-lane rows run only with CUDA_VISIBLE_DEVICES empty; this is the GPU lane")
gpu_lane = pytest.mark.skipif(
    CPU_LANE, reason="GPU-lane rows need a visible GPU; this is the CPU lane (CUDA_VISIBLE_DEVICES is empty)")
ROW_TIMEOUT = 180
HEADERS = ("cuda_runtime_api.h", "cublas_v2.h", "cublasLt.h")

# S1's form: "<entry>: <call> failed: <cudaGetErrorName>: <cudaGetErrorString>" for the nine sites (copied from
# test_eng5s_status.py's S1_FORM; that file is not imported).
S1_FORM = re.compile(r"(cuda_sync: cudaDeviceSynchronize|Storage \(cudaMalloc(Async)? OOM retry\): cudaStreamSynchronize "
                     r"\(drain\)|NDArray::zeros: cudaMemset|NDArray::from_host: cudaMemcpy|NDArray::to_host: cudaMemcpy|"
                     r"NDArray::clone: cudaMemcpy|NDArray::to: cudaMemcpy|axpy_: cudaMemcpy|apa_spq_packed_prefill: "
                     r"cudaMemset \(bulk NaN fill\)) failed: cudaError\w+: ")
INVALID_NAME = "cudaErrorInvalidValue: "
# The entry and call each row must name (from scripts/eng5s_gpu_verdict.py PREFIX; site 9 is not in this file).
EXPECT = {
    "synchronize": "cuda_sync: cudaDeviceSynchronize failed: ",
    "oom_drain": "Storage (cudaMalloc OOM retry): cudaStreamSynchronize (drain) failed: ",
    "oom_drain_pooled": "Storage (cudaMallocAsync OOM retry): cudaStreamSynchronize (drain) failed: ",
    "zeros": "NDArray::zeros: cudaMemset failed: ",
    "from_host": "NDArray::from_host: cudaMemcpy failed: ",
    "to_host": "NDArray::to_host: cudaMemcpy failed: ",
    "clone": "NDArray::clone: cudaMemcpy failed: ",
    "to_d2h": "NDArray::to: cudaMemcpy failed: ",
    "to_h2d": "NDArray::to: cudaMemcpy failed: ",
    "axpy_": "axpy_: cudaMemcpy failed: ",
}
CPU_ROWS = ["synchronize_injected", "synchronize_no_device", "oom_drain", "oom_drain_pooled"]
GPU_ROWS = ["synchronize", "oom_drain", "oom_drain_pooled", "zeros", "from_host", "to_host", "clone", "to_d2h",
            "to_h2d", "axpy_"]
GPU_CONTROL_ONLY = ["axpy_empty"]

# One row per child. The GPU bodies are copied from scripts/eng5s_gpu_step.py's G2_CHILD (the lease check line is not
# part of a production test: its children run inside whatever lease the caller holds); the cpu_* bodies are new.
ROW_CHILD = r'''
import ctypes, hashlib, json, os, sys
sys.path.insert(0, sys.argv[1])
import numpy as np
import tensor_cuda as tc
C = tc._C
L = ctypes.CDLL(None)
L.eng5_inject_next.argtypes = [ctypes.c_char_p, ctypes.c_long]
L.eng5s_fired.argtypes = [ctypes.c_char_p]
case = sys.argv[2]
INVALID = 1   # cudaErrorInvalidValue
OOM = 2       # cudaErrorMemoryAllocation (reaches the allocator's retry)
rng = np.random.default_rng(20261124)
arr = rng.standard_normal(4096).astype(np.float32)
def informative(a):
    a = np.asarray(a)
    return bool(a.size > 1 and a.any() and len(np.unique(a)) > 1)
def check(a, ref):
    a = np.ascontiguousarray(np.asarray(a))
    return {"shape": list(a.shape), "dtype": str(a.dtype), "sha256": hashlib.sha256(a.tobytes()).hexdigest(),
            "equals_expected": bool(a.shape == ref.shape and a.dtype == ref.dtype and
                                    np.array_equal(a.view(np.uint8), np.ascontiguousarray(ref).view(np.uint8))),
            "informative": informative(a), "expected_informative": informative(ref)}
def attempt(fn, arms):
    L.eng5s_inject_clear()
    for name, code in arms:
        assert L.eng5_inject_next(name.encode(), code) == 0, name
    out = {"armed": [n for n, _ in arms]}
    try:
        out["value"] = fn()
        out["outcome"] = "ok"
    except BaseException as ex:
        out["outcome"] = "raised"
        out["exc_type"] = type(ex).__name__
        out["message"] = str(ex)[:500]
    out["fired"] = {n: L.eng5s_fired(n.encode()) for n, _ in arms}
    L.eng5s_inject_clear()
    return out
three = (arr + np.float32(2.0) * arr).astype(np.float32)
rec = {"case": case, "tc_file": os.path.abspath(tc.__file__), "module_file": os.path.abspath(C.__file__)}
if case == "cpu_synchronize_injected":
    rec["injected"] = attempt(lambda: (tc.synchronize(), None)[1], [("cudaDeviceSynchronize", INVALID)])
elif case == "cpu_synchronize_no_device":
    rec["injected"] = attempt(lambda: (tc.synchronize(), None)[1], [])
elif case in ("cpu_oom_drain", "cpu_oom_drain_pooled"):
    alloc = "cudaMalloc" if case == "cpu_oom_drain" else "cudaMallocAsync"
    if case == "cpu_oom_drain_pooled":
        tc.set_alloc_pooling(True)
    rec["injected"] = attempt(lambda: C.tensor(arr, "cuda", False), [(alloc, OOM), ("cudaStreamSynchronize", INVALID)])
elif case == "synchronize":
    x = C.tensor(arr, "cuda", False)
    rec["control"] = attempt(lambda: (tc.synchronize(), None)[1], [])
    rec["injected"] = attempt(lambda: (tc.synchronize(), None)[1], [("cudaDeviceSynchronize", INVALID)])
elif case in ("oom_drain", "oom_drain_pooled"):
    alloc = "cudaMalloc" if case == "oom_drain" else "cudaMallocAsync"
    if case == "oom_drain_pooled":
        tc.set_alloc_pooling(True)
    tc.synchronize()
    # control: the allocation fails once with cudaErrorMemoryAllocation, the drain is real, the retry succeeds
    rec["control"] = attempt(lambda: check(C.tensor(arr, "cuda", False).numpy(), arr), [(alloc, OOM)])
    rec["injected"] = attempt(lambda: check(C.tensor(arr, "cuda", False).numpy(), arr),
                              [(alloc, OOM), ("cudaStreamSynchronize", INVALID)])
elif case == "zeros":
    z = np.zeros(4096, np.float32)
    rec["control"] = attempt(lambda: check(C._core_zeros([4096], "float32", "cuda").numpy(), z), [])
    rec["injected"] = attempt(lambda: check(C._core_zeros([4096], "float32", "cuda").numpy(), z),
                              [("cudaMemset", INVALID)])
elif case == "from_host":
    tc.synchronize()
    rec["control"] = attempt(lambda: check(C.tensor(arr, "cuda", False).numpy(), arr), [])
    rec["injected"] = attempt(lambda: check(C.tensor(arr, "cuda", False).numpy(), arr), [("cudaMemcpy", INVALID)])
elif case == "to_host":
    x = C.tensor(arr, "cuda", False); tc.synchronize()
    rec["control"] = attempt(lambda: check(x.numpy(), arr), [])
    rec["injected"] = attempt(lambda: check(x.numpy(), arr), [("cudaMemcpy", INVALID)])
elif case == "clone":
    x = C.tensor(arr, "cuda", False); tc.synchronize()
    rec["control"] = attempt(lambda: check(C._core_clone(x).numpy(), arr), [])
    held = []
    def inj():
        r = C._core_clone(x); held.append(r)
        return None
    rec["injected"] = attempt(inj, [("cudaMemcpy", INVALID)])
    if held:
        rec["injected"]["value"] = check(held[0].numpy(), arr)
elif case == "to_d2h":
    x = C.tensor(arr, "cuda", False); tc.synchronize()
    rec["control"] = attempt(lambda: check(C._core_to(x, "cpu").numpy(), arr), [])
    held = []
    def inj():
        r = C._core_to(x, "cpu"); held.append(r)
        return None
    rec["injected"] = attempt(inj, [("cudaMemcpy", INVALID)])
    if held:
        rec["injected"]["value"] = check(held[0].numpy(), arr)
elif case == "to_h2d":
    h = C.tensor(arr, "cpu", False)
    rec["control"] = attempt(lambda: check(C._core_to(h, "cuda").numpy(), arr), [])
    held = []
    def inj():
        r = C._core_to(h, "cuda"); held.append(r)
        return None
    rec["injected"] = attempt(inj, [("cudaMemcpy", INVALID)])
    if held:
        rec["injected"]["value"] = check(held[0].numpy(), arr)
elif case == "axpy_":
    p = C.tensor(arr, "cuda", False); o = C.tensor(arr, "cuda", False); tc.synchronize()
    rec["control"] = attempt(lambda: (C.axpy_(p, o, 2.0), check(p.numpy(), three))[1], [])
    p2 = C.tensor(arr, "cuda", False); o2 = C.tensor(arr, "cuda", False); tc.synchronize()
    rec["injected"] = attempt(lambda: C.axpy_(p2, o2, 2.0), [("cudaMemcpy", INVALID)])
    rec["injected"]["param_after"] = check(p2.numpy(), three)
elif case == "axpy_empty":
    e = np.zeros((0,), np.float32)
    p = C.tensor(e, "cuda", False); o = C.tensor(e, "cuda", False); tc.synchronize()
    rec["control"] = attempt(lambda: (C.axpy_(p, o, 2.0), list(p.shape))[1], [])
else:
    raise SystemExit("unknown case " + case)
print("ROW " + json.dumps(rec), flush=True)
'''


def _compiler():
    return next((path for name in ("cc", "gcc", "clang") if (path := shutil.which(name))), None)


def _cuda_include():
    """The CUDA toolkit include directory holding the headers the shim includes, or None."""
    cands = [os.path.join(os.environ[v], "include") for v in ("CUDA_HOME", "CUDA_PATH") if os.environ.get(v)]
    nvcc = shutil.which("nvcc")
    if nvcc:
        cands.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(nvcc))), "include"))
    cands += ["/usr/local/cuda/include"] + sorted(glob.glob("/usr/local/cuda-*/include"))
    return next((c for c in cands if all(os.path.isfile(os.path.join(c, h)) for h in HEADERS)), None)


@pytest.fixture(scope="session")
def inject_so(tmp_path_factory):
    """ENG5-S's injection shim, compiled at test time (no counter build: the injection build, as the GPU lines use)."""
    compiler = _compiler()
    if compiler is None:
        pytest.skip("No C compiler available (tried cc, gcc, clang) to build the ENG5-S injection shim")
    inc = _cuda_include()
    if inc is None:
        pytest.skip("No CUDA toolkit headers (" + ", ".join(HEADERS) + ") found to build the ENG5-S injection shim")
    out = tmp_path_factory.mktemp("eng5s_shim") / "eng5s_inject.so"
    p = subprocess.run([compiler, "-O1", "-fPIC", "-shared", "-I" + inc, "-o", str(out), SHIM, "-ldl"],
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, "shim build failed:\n" + p.stderr[-3000:]
    return str(out)


def run_row(inject_so, case):
    """Run one row in its own child interpreter with the shim preloaded. The child sees this process's environment
    (CUDA_VISIBLE_DEVICES in particular) unchanged, and imports the same tensor_cuda package."""
    pre = inject_so + (":" + os.environ["LD_PRELOAD"] if os.environ.get("LD_PRELOAD") else "")
    env = dict(os.environ, LD_PRELOAD=pre)
    p = subprocess.run([sys.executable, "-B", "-P", "-c", ROW_CHILD, PKG_PARENT, case], env=env, capture_output=True,
                       text=True, timeout=ROW_TIMEOUT)
    got = [line for line in p.stdout.splitlines() if line.startswith("ROW ")]
    assert p.returncode == 0 and got, f"row child {case!r} failed: rc={p.returncode}\n{p.stderr[-3000:]}"
    rec = json.loads(got[0][4:])
    # The child exercised the very package and module this session imported.
    assert os.path.realpath(rec["tc_file"]) == os.path.realpath(tc.__file__), rec["tc_file"]
    return rec


def assert_raises_s1(att, prefix, name, what):
    """The injected call: every armed injection fired, and the call raised RuntimeError in the S1 form that names the
    entry and the call (`prefix`) and the CUDA error name (`name`)."""
    assert att is not None, what + ": no attempt recorded"
    for arm in att["armed"]:
        assert att["fired"].get(arm) == 1, f"{what}: the armed injection of {arm} did not fire: {att['fired']}"
    assert att["outcome"] == "raised", f"{what}: returned under a fired injection (the failure passed silently): {att}"
    assert att["exc_type"] == "RuntimeError", f"{what}: {att}"
    msg = att["message"]
    assert S1_FORM.match(msg), f"{what}: message is not in the S1 form: {msg!r}"
    assert msg.startswith(prefix + name), f"{what}: expected {prefix + name!r}, got {msg!r}"


@cpu_lane
@pytest.mark.parametrize("row", CPU_ROWS)
def test_cpu_status_rows(row, inject_so):
    rec = run_row(inject_so, "cpu_" + row)
    if row == "synchronize_no_device":
        # No injection: the real runtime fails (no device is visible) and tc.synchronize reports it in the S1 form.
        assert rec["injected"]["armed"] == []
        assert_raises_s1(rec["injected"], EXPECT["synchronize"], "cudaErrorNoDevice: ", row)
    elif row == "synchronize_injected":
        assert_raises_s1(rec["injected"], EXPECT["synchronize"], INVALID_NAME, row)
    else:
        # Both injections fire: the allocation fails with out-of-memory, the drain before the retry fails.
        assert set(rec["injected"]["armed"]) == {"cudaMalloc" if row == "oom_drain" else "cudaMallocAsync",
                                                 "cudaStreamSynchronize"}
        assert_raises_s1(rec["injected"], EXPECT[row], INVALID_NAME, row)


def _control_ok(row, ctl):
    """The uninjected success control: the call returns its value, which equals the value computed on the host."""
    assert ctl is not None and ctl["outcome"] == "ok", f"{row}: the success control did not return: {ctl}"
    v = ctl.get("value")
    if row == "synchronize":
        return
    if row == "axpy_empty":
        assert v == [0], f"{row}: expected an empty parameter, got {v}"
        return
    assert isinstance(v, dict) and v["equals_expected"], f"{row}: the control's value differs from the host value: {v}"
    if row != "zeros":  # all-zero by design: the injected row decides site 3
        assert v["informative"] and v["expected_informative"], f"{row}: control compared on a constant array: {v}"


@gpu_lane
@pytest.mark.parametrize("row", GPU_ROWS + GPU_CONTROL_ONLY)
def test_gpu_status_rows(row, inject_so):
    rec = run_row(inject_so, row)
    _control_ok(row, rec.get("control"))
    if row in GPU_CONTROL_ONLY:
        assert "injected" not in rec
        return
    inj = rec["injected"]
    if row.startswith("oom"):
        assert set(inj["armed"]) == {"cudaMalloc" if row == "oom_drain" else "cudaMallocAsync", "cudaStreamSynchronize"}
        assert rec["control"]["fired"] == {inj["armed"][0]: 1}, rec["control"]  # the control's OOM retry did happen
    assert_raises_s1(inj, EXPECT[row], INVALID_NAME, row)
