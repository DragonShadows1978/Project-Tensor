"""ENG5-S G1 (orders/ENG5S_STATUS_CHECKS.md): the nine status checks change nothing that reaches no CUDA call, and the
change is confined to the nine sites.

CPU lane only (CUDA_VISIBLE_DEVICES empty). Builds B (fbd335ae7) and S (this round) are read from their receipted
packages, tensor_cuda/build-eng5s/<arm>/pkg (each module's sha256 is checked against artifacts/eng5s/build/BUILD_<arm>.json
first). Each arm runs in its own child interpreter with ENG3's CUDA runtime counter (support/cudart_count.c) and
ENG5-S's copy of ENG5's cuBLAS / remaining-runtime counter (support/eng5s_cuda_shim.c, -DENG5_WITH_ENG3_COUNTER)
preloaded, so every row records the CUDA calls it made.

- test_g1_source_map: Build B's and Build S's engine source maps (src/, include/, the Python package) differ in exactly
  kernels.cu, apa_spo/packed_prefill.cuh and include/tc/core.h; the worktree's engine files and installed module are
  Build S's; the hooks arms B_H / S_H have B's / S's sources (only a copied CMakeLists.txt carries the hooks line).
- test_g1_status_rows[<row>]: every public route to the nine sites called with host operands, with empty CUDA
  operands, and (CUDA hidden) with nonempty CUDA operands. Rule (judge): where Build B made no CUDA call, Build S gives
  exactly Build B's outcome, exception type, message, result and per-function call counts; where Build B made CUDA
  calls, Build S gives Build B's outcome or raises RuntimeError in the S1 form (entry: call failed: cudaError...).
- test_g1_eng5_catalogue[<mode>-<key>]: ENG5's whole catalogue of public callables (scripts/eng5s_catalogue.py, a copy)
  in modes host, empty, emptylast and one, under the same rule (the confinement check over every public entry).
- test_g1_oom_drain_cpu: site 2 on the CPU lane. The injection shim (support/eng5s_cuda_shim.c) makes the allocation
  return cudaErrorMemoryAllocation and the drain fail; Build S raises in the S1 form naming the OOM retry and the
  drain, where Build B ignores the drain and retries.

Prior art: LD_PRELOAD + dlsym(RTLD_NEXT) call counting (ENG3's counter, this repo); fault injection by library
interposition (LFI, Marinescu and Candea, 2009 -- unverified, lead to check "LFI library fault injection
interposition"); differential testing of two builds (McKeeman, 1998, "Differential Testing for Software" -- unverified,
lead to check). Ours: the row list and the B-versus-S rule.
"""
import hashlib
import json
import os
import re
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SCRIPTS = os.path.join(ROOT, "scripts")
SUPPORT = os.path.join(HERE, "support")
SO = "_tensor_cuda.cpython-312-x86_64-linux-gnu.so"
SHIMS = "/tmp/eng5s_shim"
TMP = os.environ.get("TMPDIR", "/tmp/eng5s_tmp")
CPU_LANE = os.environ.get("CUDA_VISIBLE_DEVICES") == ""
pytestmark = pytest.mark.skipif(not CPU_LANE, reason="ENG5-S G1 runs with CUDA hidden (L58)")
CHANGED = {"tensor_cuda/src/kernels.cu", "tensor_cuda/src/apa_spo/packed_prefill.cuh", "tensor_cuda/include/tc/core.h"}
# S1's form: "<entry>: <call> failed: <cudaGetErrorName>: <cudaGetErrorString>" for the nine sites.
S1_FORM = re.compile(r"(cuda_sync: cudaDeviceSynchronize|Storage \(cudaMalloc(Async)? OOM retry\): cudaStreamSynchronize "
                     r"\(drain\)|NDArray::zeros: cudaMemset|NDArray::from_host: cudaMemcpy|NDArray::to_host: cudaMemcpy|"
                     r"NDArray::clone: cudaMemcpy|NDArray::to: cudaMemcpy|axpy_: cudaMemcpy|apa_spq_packed_prefill: "
                     r"cudaMemset \(bulk NaN fill\)) failed: cudaError\w+: ")


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def receipt(arm):
    return json.load(open(os.path.join(ROOT, "artifacts/eng5s/build", f"BUILD_{arm}.json")))


def pkg(arm):
    p = os.path.join(ROOT, "tensor_cuda/build-eng5s", arm, "pkg")
    assert sha(os.path.join(p, "tensor_cuda", SO)) == receipt(arm)["module_sha256"], arm  # L103
    return p


def build_shims():
    os.makedirs(SHIMS, exist_ok=True)
    eng3, cnt, inj = (os.path.join(SHIMS, n) for n in ("cudart_count.so", "eng5s_count.so", "eng5s_inject.so"))
    subprocess.run(["gcc", "-O1", "-fPIC", "-shared", "-mgeneral-regs-only", "-o", eng3,
                    os.path.join(SUPPORT, "cudart_count.c"), "-ldl"], check=True, timeout=120)
    subprocess.run(["gcc", "-O1", "-fPIC", "-shared", "-DENG5_WITH_ENG3_COUNTER", "-I/usr/local/cuda-12.6/include",
                    "-o", cnt, os.path.join(SUPPORT, "eng5s_cuda_shim.c"), "-ldl"], check=True, timeout=120)
    subprocess.run(["gcc", "-O1", "-fPIC", "-shared", "-I/usr/local/cuda-12.6/include", "-o", inj,
                    os.path.join(SUPPORT, "eng5s_cuda_shim.c"), "-ldl"], check=True, timeout=120)
    return eng3, cnt, inj


# Rows whose returned tensor is unwritten memory on every build (INT4 fused with K = 0, ENG5's INT4-K0-UNWRITTEN: the
# kernel divides by zero before its return). Build B against itself gives a different result sha256 on each run
# (artifacts/eng5s/g1/UNWRITTEN_REPEAT_B_{1,2}.json), so the bytes are not compared there; everything else is (outcome,
# type, message, shape, dtype, device, call counts). ENG5-F1 listed the same rows (scripts/eng5_f1_boundary.py UNWRITTEN).
UNWRITTEN = {("emptylast", "C.int4_linear_fused")}


def _result(r, unwritten):
    v = r.get("result")
    if unwritten and isinstance(v, dict):
        return {k: x for k, x in v.items() if k != "sha256"}
    return v


def judge(b, s, unwritten=False):
    """G1's rule for one row observed on both builds. Returns (ok, why)."""
    if b is None or s is None:
        return False, "row missing on one build"
    if b.get("outcome") == "child_died" or s.get("outcome") == "child_died":
        return False, "a child died"
    keys = ("outcome", "exc_type", "message")
    same = all(b.get(k) == s.get(k) for k in keys) and _result(b, unwritten) == _result(s, unwritten)
    if b.get("cuda_calls") == 0:
        if same and s.get("cuda_calls") == 0 and b.get("cuda_calls_by_function") == s.get("cuda_calls_by_function"):
            return True, "no CUDA call on Build B: identical"
        return False, "no CUDA call on Build B, but Build S differs"
    if same:
        return True, "CUDA calls on Build B: identical"
    if s.get("outcome") == "raised" and s.get("exc_type") == "RuntimeError" and S1_FORM.search(s.get("message") or ""):
        return True, "CUDA calls on Build B: Build S raises in the S1 form"
    return False, "CUDA calls on Build B: Build S differs outside the S1 form"


ROW_CHILD = r"""
import ctypes, hashlib, json, os, sys
sys.path.insert(0, sys.argv[1])
import numpy as np
import tensor_cuda as tc
C = tc._C
L = ctypes.CDLL(None)
L.eng3_cudart_calls.restype = ctypes.c_ulong
L.eng3_cudart_count.restype = ctypes.c_ulong
L.eng3_cudart_name.restype = ctypes.c_char_p
L.eng5_extra_calls.restype = ctypes.c_ulong
L.eng5_count.restype = ctypes.c_ulong
L.eng5_name.restype = ctypes.c_char_p
def snap():
    d = {}
    for i in range(L.eng3_cudart_n()):
        c = L.eng3_cudart_count(i)
        if c: d[L.eng3_cudart_name(i).decode()] = c
    for i in range(33):
        c = L.eng5_count(i)
        if c: d[L.eng5_name(i).decode()] = c
    return d
def total():
    return L.eng3_cudart_calls() + L.eng5_extra_calls()
def summary(r, depth=0):
    if isinstance(r, np.ndarray):
        return {"ndarray": list(r.shape), "dtype": str(r.dtype), "sha256": hashlib.sha256(r.tobytes()).hexdigest()}
    if isinstance(r, tc.Tensor):
        s = {"shape": list(r.shape), "dtype": r.dtype, "device": r.device}
        if r.device == "cpu":
            a = r.astype("float32").numpy() if r.dtype == "bfloat16" else r.numpy()
            s["sha256"] = hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
        return s
    if isinstance(r, (tuple, list)) and depth < 3:
        return [summary(x, depth + 1) for x in r]
    return {"type": type(r).__name__}
rows = json.loads(sys.argv[2])
for name, setup, call, post in rows:
    g = {"tc": tc, "C": C, "np": np, "os": os}
    print(json.dumps({"key": name, "start": True}), flush=True)
    out = {"key": name}
    try:
        exec(setup, g)
    except BaseException as ex:
        out.update(outcome="setup_failed", message=str(ex)[:400])
        print(json.dumps(out), flush=True)
        continue
    before, b0 = snap(), total()
    try:
        r = eval(call, g)
        b1 = total(); after = snap()
        out["outcome"] = "ok"
        out["result"] = summary(r)
        if post:
            g["r"] = r
            out["result"] = {"value": out["result"], "post": summary(eval(post, g))}
    except BaseException as ex:
        b1 = total(); after = snap()
        out["outcome"] = "raised"
        out["exc_type"] = type(ex).__name__
        out["message"] = str(ex)[:600]
    out["cuda_calls"] = b1 - b0
    out["cuda_calls_by_function"] = {k: after.get(k, 0) - before.get(k, 0) for k in after
                                     if after.get(k, 0) != before.get(k, 0)}
    print(json.dumps(out), flush=True)
"""

A = "A = np.random.default_rng(20261124).standard_normal((4, 3)).astype(np.float32)"
A0 = "A0 = np.zeros((0, 3), np.float32)"
HX = A + "; X = C.tensor(A, 'cpu', False)"
CX0 = A0 + "; X0 = C.tensor(A0, 'cuda', False)"
HX0 = A0 + "; XH0 = C.tensor(A0, 'cpu', False)"
PACK_HOST = ("rng = np.random.default_rng(7); "
             "q = C.tensor(rng.standard_normal((1, 4, 4, 64)).astype(np.float32), 'cpu', False); "
             "k = C.tensor(rng.standard_normal((1, 1, 16, 64)).astype(np.float32), 'cpu', False); "
             "v = C.tensor(rng.standard_normal((1, 1, 16, 64)).astype(np.float32), 'cpu', False); "
             "codes = C.tensor(rng.integers(0, 256, size=(1, 1, 16, 32)).astype(np.uint8), 'cpu', False); "
             "sc = C.tensor(np.ones((1, 1, 16), np.float32), 'cpu', False); "
             "rot = C.tensor(np.eye(64, dtype=np.float32)[None], 'cpu', False); "
             "cb = C.tensor(np.linspace(-1, 1, 16).astype(np.float32), 'cpu', False); "
             "os.environ['TC_APA_SP'] = '1'")
# (row id, setup, call, post-call value read (or "")) -- every public route to the nine sites this lane can drive.
ROWS = [
    # site 1, cuda_sync (reaches cudaDeviceSynchronize: a CUDA call, GPU hidden)
    ("s1_tc_synchronize", "", "tc.synchronize()", ""),
    ("s1_C_synchronize", "", "C.synchronize()", ""),
    # site 2, Storage's allocation (the OOM retry itself is test_g1_oom_drain_cpu)
    ("s2_cuda_alloc_hidden", "", "C._core_zeros([4, 3], 'float32', 'cuda')", ""),
    ("s2_cuda_alloc_empty", "", "C._core_zeros([0, 3], 'float32', 'cuda')", ""),
    # site 3, NDArray::zeros
    ("s3_zeros_host_f32", "", "C._core_zeros([4, 3], 'float32', 'cpu')", ""),
    ("s3_zeros_host_f16", "", "C._core_zeros([4, 3], 'float16', 'cpu')", ""),
    ("s3_zeros_host_bf16", "", "C._core_zeros([4, 3], 'bfloat16', 'cpu')", ""),
    ("s3_zeros_host_empty", "", "C._core_zeros([0, 3], 'float32', 'cpu')", ""),
    ("s3_py_zeros_host", "", "tc.zeros(4, 3, device='cpu')", ""),
    ("s3_py_zeros_cuda_empty", "", "tc.zeros(0, 3, device='cuda')", ""),
    ("s3_py_zeros_cuda_hidden", "", "tc.zeros(4, 3, device='cuda')", ""),
    # site 4, NDArray::from_host
    ("s4_from_host_host", A, "C.tensor(A, 'cpu', False)", ""),
    ("s4_from_host_host_empty", A0, "C.tensor(A0, 'cpu', False)", ""),
    ("s4_from_host_cuda_empty", A0, "C.tensor(A0, 'cuda', False)", ""),
    ("s4_from_host_cuda_hidden", A, "C.tensor(A, 'cuda', False)", ""),
    ("s4_py_tensor_host", A, "tc.tensor(A, device='cpu')", ""),
    ("s4_py_tensor_host_f16", A, "tc.tensor(A, device='cpu', dtype='float16')", ""),
    ("s4_py_tensor_cuda_hidden", A, "tc.tensor(A, device='cuda')", ""),
    # site 5, NDArray::to_host
    ("s5_to_host_host", HX, "X.numpy()", ""),
    ("s5_to_host_host_empty", HX0, "XH0.numpy()", ""),
    ("s5_to_host_cuda_empty", CX0, "X0.numpy()", ""),
    # site 6, NDArray::clone (and its callers astype-same-dtype and detach)
    ("s6_clone_host", HX, "C._core_clone(X)", ""),
    ("s6_clone_host_empty", HX0, "C._core_clone(XH0)", ""),
    ("s6_clone_cuda_empty", CX0, "C._core_clone(X0)", ""),
    ("s6_astype_same_host", HX, "X.astype('float32')", ""),
    ("s6_astype_same_cuda_empty", CX0, "X0.astype('float32')", ""),
    ("s6_detach_host", HX, "X.detach()", ""),
    # site 7, NDArray::to
    ("s7_to_cpu_from_host", HX, "C._core_to(X, 'cpu')", ""),
    ("s7_to_cuda_from_host_hidden", HX, "C._core_to(X, 'cuda')", ""),
    ("s7_to_cuda_from_host_empty", HX0, "C._core_to(XH0, 'cuda')", ""),
    ("s7_to_cpu_from_cuda_empty", CX0, "C._core_to(X0, 'cpu')", ""),
    ("s7_to_cuda_from_cuda_empty", CX0, "C._core_to(X0, 'cuda')", ""),
    # site 8, axpy_ (the value read after the call is the parameter)
    ("s8_axpy_host", HX + "; P = C.tensor(A, 'cpu', False); O = C.tensor(A, 'cpu', False)", "C.axpy_(P, O, 2.0)",
     "P"),
    ("s8_axpy_cuda_empty", CX0 + "; P = C.tensor(A0, 'cuda', False); O = C.tensor(A0, 'cuda', False)",
     "C.axpy_(P, O, 2.0)", ""),
    # site 9, apa_spq packed prefill (the NaN fill runs only with dump=True in a hooks build)
    ("s9_packed_prefill_host", PACK_HOST,
     "tc._spo2.packed_prefill(q, k, v, (codes, sc, rot, cb), 0.125, 6.3, True, None, False, fmt='tq4')", ""),
    ("s9_packed_prefill_host_dump", PACK_HOST,
     "tc._spo2.packed_prefill(q, k, v, (codes, sc, rot, cb), 0.125, 6.3, True, None, False, fmt='tq4', dump=True)",
     ""),
]


def run_rows(arm, eng3, cnt):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_PRELOAD=eng3 + ":" + cnt, TMPDIR=TMP)
    out, todo = {}, list(ROWS)
    while todo:
        p = subprocess.run([sys.executable, "-B", "-P", "-c", ROW_CHILD, pkg(arm), json.dumps(todo)], env=env,
                           capture_output=True, text=True, timeout=600, cwd=TMP)
        started = None
        for line in p.stdout.splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("start"):
                started = d["key"]
            else:
                out[d["key"]] = d
                started = None
        if started is not None:
            out[started] = {"key": started, "outcome": "child_died", "returncode": p.returncode,
                            "stderr": p.stderr[-600:]}
        elif p.returncode != 0:
            raise AssertionError(f"row child failed outside a row: rc={p.returncode} {p.stderr[-2000:]}")
        todo = [r for r in todo if r[0] not in out]
    return out


@pytest.fixture(scope="session")
def shims():
    os.makedirs(TMP, exist_ok=True)
    return build_shims()


@pytest.fixture(scope="session")
def status_rows(shims):
    eng3, cnt, _ = shims
    res = {arm: run_rows(arm, eng3, cnt) for arm in ("B", "S")}
    rec = os.environ.get("ENG5S_G1_RECORD")
    if rec:
        json.dump(res, open(os.path.join(rec, "G1_STATUS_ROWS.json"), "w"), indent=1)
    return res


@pytest.fixture(scope="session")
def catalogue_rows():
    res = {}
    for arm in ("B", "S"):
        out = os.path.join(TMP, f"eng5s_g1_catalogue_{arm}.json")
        p = subprocess.run([sys.executable, "-B", "-P", os.path.join(SCRIPTS, "eng5s_probe.py"), pkg(arm), out],
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES="", TMPDIR=TMP), capture_output=True, text=True,
                           timeout=1500, cwd=TMP)
        assert p.returncode == 0, p.stderr[-3000:]
        res[arm] = {(r["mode"], r["key"]): r for r in json.load(open(out))["rows"]}
    rec = os.environ.get("ENG5S_G1_RECORD")
    if rec:
        json.dump({arm: [dict(v, mode=k[0]) for k, v in sorted(rows.items())] for arm, rows in res.items()},
                  open(os.path.join(rec, "G1_CATALOGUE_ROWS.json"), "w"), indent=1)
    return res


def test_g1_source_map():
    b, s = receipt("B"), receipt("S")
    differ = {f for f in set(b["sources"]) | set(s["sources"]) if b["sources"].get(f) != s["sources"].get(f)}
    assert differ == CHANGED, sorted(differ ^ CHANGED)
    # The worktree's engine files and installed module are Build S's (the suite runner restores them: L128).
    for f, h in s["sources"].items():
        assert sha(os.path.join(ROOT, f)) == h, f
    assert sha(os.path.join(ROOT, "tensor_cuda/tensor_cuda", SO)) == s["module_sha256"]
    # Hooks arms: the same engine sources; only the copied CMakeLists.txt carries the hooks line.
    bh, sh = receipt("B_H"), receipt("S_H")
    assert bh["sources"] == b["sources"] and sh["sources"] == s["sources"]
    assert bh["hooks_line_present"] and sh["hooks_line_present"]
    assert not b["hooks_line_present"] and not s["hooks_line_present"]
    assert bh["cmakelists_sha256"] == sh["cmakelists_sha256"] and b["cmakelists_sha256"] == s["cmakelists_sha256"]


@pytest.mark.parametrize("row", [r[0] for r in ROWS])
def test_g1_status_rows(row, status_rows):
    b, s = status_rows["B"].get(row), status_rows["S"].get(row)
    ok, why = judge(b, s)
    assert ok, (row, why, b, s)


def _catalogue_ids():
    sys.path.insert(0, SCRIPTS)
    import eng5s_catalogue as cat
    ids = []
    for mode in ("host", "empty", "emptylast", "one"):
        for k, r in cat.ROWS.items():
            if r["call"] is None:
                continue
            if mode == "one" and not (r["empty_rule"] == "any" and len(r["ops"]) > 1):
                continue
            ids.append(f"{mode}-{k}")
    return sorted(ids)


@pytest.mark.parametrize("case", _catalogue_ids())
def test_g1_eng5_catalogue(case, catalogue_rows):
    mode, key = case.split("-", 1)
    ok, why = judge(catalogue_rows["B"].get((mode, key)), catalogue_rows["S"].get((mode, key)),
                    unwritten=(mode, key) in UNWRITTEN)
    assert ok, (case, why, catalogue_rows["B"].get((mode, key)), catalogue_rows["S"].get((mode, key)))


OOM_CHILD = r"""
import ctypes, json, sys
sys.path.insert(0, sys.argv[1])
import numpy as np
import tensor_cuda as tc
L = ctypes.CDLL(None)
L.eng5_inject_next.argtypes = [ctypes.c_char_p, ctypes.c_long]
L.eng5s_fired.argtypes = [ctypes.c_char_p]
A = np.arange(12, dtype=np.float32)
rows = {}
for case, arms in (("drain_fails", (("cudaMalloc", 2), ("cudaStreamSynchronize", 1))),
                   ("drain_real", (("cudaMalloc", 2),))):
    L.eng5s_inject_clear()
    for name, code in arms:
        assert L.eng5_inject_next(name.encode(), code) == 0
    rec = {}
    try:
        tc._C.tensor(A, "cuda", False)
        rec["outcome"] = "ok"
    except BaseException as ex:
        rec.update(outcome="raised", exc_type=type(ex).__name__, message=str(ex)[:400])
    rec["fired"] = {name: L.eng5s_fired(name.encode()) for name, _ in arms}
    rows[case] = rec
print("ROWS " + json.dumps(rows))
"""


def test_g1_oom_drain_cpu(shims):
    _, _, inj = shims
    got = {}
    for arm in ("B", "S"):
        p = subprocess.run([sys.executable, "-B", "-P", "-c", OOM_CHILD, pkg(arm)],
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_PRELOAD=inj, TMPDIR=TMP),
                           capture_output=True, text=True, timeout=300, cwd=TMP)
        line = [l for l in p.stdout.splitlines() if l.startswith("ROWS ")]
        assert p.returncode == 0 and line, p.stderr[-2000:]
        got[arm] = json.loads(line[0][5:])
    rec = os.environ.get("ENG5S_G1_RECORD")
    if rec:
        json.dump(got, open(os.path.join(rec, "G1_OOM_DRAIN_CPU.json"), "w"), indent=1)
    for arm in ("B", "S"):
        assert got[arm]["drain_fails"]["fired"] == {"cudaMalloc": 1, "cudaStreamSynchronize": 1}, got[arm]
        assert got[arm]["drain_real"]["fired"] == {"cudaMalloc": 1}, got[arm]
    s = got["S"]["drain_fails"]
    assert s["outcome"] == "raised" and s["exc_type"] == "RuntimeError", s
    assert s["message"].startswith("Storage (cudaMalloc OOM retry): cudaStreamSynchronize (drain) failed: "
                                   "cudaErrorInvalidValue: "), s
    b = got["B"]["drain_fails"]
    # Build B ignores the failed drain and retries the allocation (which fails here only because no GPU is visible).
    assert b["outcome"] == "raised" and b["message"].startswith("cudaMalloc failed: "), b
