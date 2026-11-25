import subprocess
if subprocess.run(["forge-gpu-lease-check", "--require-held"]).returncode != 0: raise SystemExit("not inside a held forge-gpu-run lease")
"""ENG5-S GPU step script: the only script a staged ENG5-S GPU line runs inside forge-gpu-run. Its first statement
(line 2, after `import subprocess`) is the held-lease check (order Machine rules, L125), before anything imports the
engine or reads CUDA_VISIBLE_DEVICES. Every child it starts repeats the check as its first statement.

  python3 -B -P scripts/eng5s_gpu_step.py <line>        (from scripts/eng5s_gpu_line.py's forge command only)

Lines (artifacts/eng5s/GPU_LINES.json):
  g3_<arm>       arms B, S, s01..s04 -- G2's failure injection (ported from ENG5-F2's f2g3 lines,
                 scripts/eng5_f2_gpu_step.py on the eng5 branch). The injection shim
                 (tensor_cuda/tests/support/eng5s_cuda_shim.c) makes each of the nine calls fail in turn, ONE CHILD PER
                 CASE. Each child first runs the same call without injection (the success control: S2) and then with
                 the injection armed. Site 9 (the packed prefill's dump fill) runs on the arm's hooks build (B_H for B,
                 S_H for S); the mutant arms have no hooks build, so that row is recorded not applicable there.
  suite_<arm>_NN arms B, S -- G3's GPU half: production's suite per test ID (scripts/eng5s_suite.py, budgeted chunks,
                 series gpu_<arm>). Each B chunk installs Build B's three engine files and module into the worktree and
                 restores Build S's afterwards, sha-checked (L128).
This script never removes or overrides CUDA_VISIBLE_DEVICES; if the lease's environment hides the GPU it refuses (exit 7).
The post-device code (verdicts) is scripts/eng5s_gpu_verdict.py, driven on CPU with constructed rows (L48).

Prior art: forge-gpu-run lease contract (house tooling); fault injection by library interposition (LFI, Marinescu and
Candea, 2009 -- unverified, lead to check "LFI library fault injection interposition"); process-per-test isolation (as
in pytest-forked, unverified -- lead to check); ENG5-F2's step script (this repo), whose layout this copies. Ours: the
ten injected rows, the success controls and the hooks-arm routing of site 9.
"""
import hashlib
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
SUPPORT = os.path.join(ROOT, "tensor_cuda", "tests", "support")
SHIMS = "/tmp/eng5s_gpu_shim"
TMP = "/tmp/eng5s_tmp"
SO = "_tensor_cuda.cpython-312-x86_64-linux-gnu.so"
ARMS = {"B": "B", "S": "S", "s01": "s01_clone_status_dropped", "s02": "s02_to_status_dropped",
        "s03": "s03_zeros_status_dropped", "s04": "s04_from_host_last_error_only"}
HOOKS = {"B": "B_H", "S": "S_H"}
CHILD_TIMEOUT = 60
# The ten injected rows: the nine sites (site 7, NDArray::to, in both directions; site 2 through both allocators) ...
G2_CASES = ["synchronize", "oom_drain", "oom_drain_pooled", "zeros", "from_host", "to_host", "clone", "to_d2h",
            "to_h2d", "axpy_", "packed_prefill_dump"]
# ... and one control-only row: axpy_ on empty CUDA operands calls the copy with 0 bytes (Build B did too, unchecked).
CONTROL_ONLY = ["axpy_empty"]


def receipt(arm):
    return os.path.join(ROOT, f"artifacts/eng5s/build/BUILD_{arm}.json")


def pkg(arm):
    return os.path.join(ROOT, "tensor_cuda", "build-eng5s", arm, "pkg")


def build_shim():
    os.makedirs(SHIMS, exist_ok=True)
    inj = os.path.join(SHIMS, "eng5s_inject.so")
    subprocess.run(["gcc", "-O1", "-fPIC", "-shared", "-I/usr/local/cuda-12.6/include", "-o", inj,
                    os.path.join(SUPPORT, "eng5s_cuda_shim.c"), "-ldl"], check=True, timeout=120)
    return inj


G2_CHILD = r'''
import subprocess
if subprocess.run(["forge-gpu-lease-check", "--require-held"]).returncode != 0: raise SystemExit("not inside a held forge-gpu-run lease")
import ctypes, hashlib, json, os, sys
sys.path.insert(0, sys.argv[1])
import numpy as np
import tensor_cuda as tc
C = tc._C
L = ctypes.CDLL(None)
L.eng5_inject_next.argtypes = [ctypes.c_char_p, ctypes.c_long]
L.eng5s_fired.argtypes = [ctypes.c_char_p]
case = sys.argv[2]
INVALID = 1   # cudaErrorInvalidValue (the code ENG5-F1/F2's lines injected)
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
rec = {"case": case}
if case == "synchronize":
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
elif case == "packed_prefill_dump":
    os.environ["TC_APA_SP"] = "1"
    q = tc.tensor(rng.standard_normal((1, 4, 4, 64)).astype(np.float32))
    k = tc.tensor(rng.standard_normal((1, 1, 16, 64)).astype(np.float32))
    v = tc.tensor(rng.standard_normal((1, 1, 16, 64)).astype(np.float32))
    pk = tc.PackedSpKeys(k)
    tc.synchronize()
    def pf():
        out, mask, bulk = tc._spo2.packed_prefill(q, k, v, pk._packed_tuple(), 0.125, 6.3, True, None, False,
                                                  fmt="tq4", dump=True)
        o, b = out.numpy(), bulk.numpy()
        return {"out_finite": bool(np.isfinite(o).all()), "out_informative": informative(o),
                "bulk_nan": int(np.isnan(b).sum()), "bulk_finite": int(np.isfinite(b).sum()), "bulk_size": int(b.size),
                "bulk_sha256": hashlib.sha256(np.ascontiguousarray(b).tobytes()).hexdigest()}
    rec["control"] = attempt(pf, [])
    rec["injected"] = attempt(pf, [("cudaMemset", INVALID)])
else:
    raise SystemExit("unknown case " + case)
try:
    tc.synchronize(); rec["synchronize_after"] = "ok"
except BaseException as ex:
    rec["synchronize_after"] = str(ex)[:200]
print("G2ROW " + json.dumps(rec), flush=True)
'''


def module_ok(arm):
    want = json.load(open(receipt(arm)))["module_sha256"]
    got = hashlib.sha256(open(os.path.join(pkg(arm), "tensor_cuda", SO), "rb").read()).hexdigest()
    return got == want, got, want


def child(argv, env, tag, timeout):
    try:
        q = subprocess.run([sys.executable, "-B", "-P", "-c"] + argv, env=env, capture_output=True, text=True,
                           timeout=timeout, cwd=TMP)
    except subprocess.TimeoutExpired as ex:
        err = ex.stderr.decode(errors="replace") if isinstance(ex.stderr, bytes) else str(ex.stderr or "")
        return None, {"returncode": "timeout", "stderr": err[-1500:]}
    got = [l for l in q.stdout.splitlines() if l.startswith(tag + " ")]
    info = {"returncode": q.returncode, "stderr": q.stderr[-1500:]}
    return (json.loads(got[0][len(tag) + 1:]) if got else None), info


def g2_rows(short, env, child_code=None):
    """One child per case on the arm's package (site 9 on its hooks package). Returns the rows."""
    arm = ARMS[short]
    rows = []
    for case in G2_CASES + CONTROL_ONLY:
        if case == "packed_prefill_dump":
            if short not in HOOKS:
                rows.append({"case": case, "outcome": "not_applicable",
                             "why": "site 9 runs only in a hooks build; the mutant arms have none"})
                continue
            p = pkg(HOOKS[short])
        else:
            p = pkg(arm)
        rec, info = child([child_code or G2_CHILD, p, case], env, "G2ROW", CHILD_TIMEOUT)
        if rec is None:
            rows.append({"case": case, "outcome": "child_failed", **info})
        else:
            rec["child_returncode"] = info["returncode"]
            rec["package"] = os.path.relpath(p, ROOT)
            rows.append(rec)
    return rows


def main():
    line = sys.argv[1]
    d = os.path.join(ROOT, "artifacts/eng5s/gpu", line)
    if not os.path.isdir(d):
        raise SystemExit("no line directory: run the line through scripts/eng5s_gpu_line.py run")
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        open(os.path.join(d, "INNER_EXIT"), "w").write("7")
        raise SystemExit("REFUSED: CUDA_VISIBLE_DEVICES is empty in this lease's environment (this script never "
                         "changes it); run the line from an environment with the GPU visible")
    rc = 0
    if line.startswith("suite_"):
        _, arm, nn = line.split("_")
        import eng5s_suite  # imports the engine only inside the held lease (checked on line 2)
        sys.argv = ["eng5s_suite.py", f"gpu_{arm}", arm, "--budget", "330"]
        rc = eng5s_suite.main()
        verdict = {"line": line, "verdict": "suite chunk: compared per ID after the series (scripts/eng5s_compare.py)"}
    else:
        short = line.split("_", 1)[1]
        for a in [ARMS[short]] + ([HOOKS[short]] if short in HOOKS else []):
            ok, got, want = module_ok(a)
            if not ok:
                open(os.path.join(d, "INNER_EXIT"), "w").write("3")
                raise SystemExit(f"REFUSED (L103): {a} package module {got} != receipt {want}")
        inj = build_shim()
        os.makedirs(TMP, exist_ok=True)
        env = dict(os.environ, TMPDIR=TMP, LD_PRELOAD=inj, ENG5S_MUTANT_MARKER=os.path.join(d, "mutant_marker.txt"))
        rows = g2_rows(short, env)
        json.dump(rows, open(os.path.join(d, "g3_rows.json"), "w"), indent=1)
        import eng5s_gpu_verdict
        verdict = eng5s_gpu_verdict.judge_line(line, d)
    json.dump(verdict, open(os.path.join(d, "VERDICT.json"), "w"), indent=1)
    open(os.path.join(d, "INNER_EXIT"), "w").write(str(rc))
    return rc


if __name__ == "__main__":
    sys.exit(main())
