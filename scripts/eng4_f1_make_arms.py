"""ENG4-F1 mutant arms (H4) and the frozen F1 package copies.

  python3 -B -P scripts/eng4_f1_make_arms.py register <killing-test map json>
        -> artifacts/eng4/f1/MUTANTS_F1.json (before any F1 mutant is built; write-once)
  python3 -B -P scripts/eng4_f1_make_arms.py tree <id>...
        -> /tmp/eng4_src_<id>/tensor_cuda: Build F1's sources with one registered hunk; diff in f1/mutants/<id>.diff
  python3 -B -P scripts/eng4_f1_make_arms.py pkg <arm> <package dir with the .py files>
        -> tensor_cuda/build-eng4/<arm>/pkg/tensor_cuda (the .py files + the arm's receipted module; f1/build receipt)

m01_f1..m05_f1 are ENG4's five hunks (imported from scripts/eng4_make_arms.py, unchanged), applied to F1's sources.
m06..m08 are this order's. Each hunk prints "ENG4-MUTANT <id> executed" to stderr every time the mutated line runs
(L41). The killing tests and named assertions come from the order (H4, L47); exact pytest IDs are filled from a
collection of tensor_cuda/tests/test_eng4_matmul.py (the map file passed to `register`).

Prior art: mutation testing (DeMillo, Lipton and Sayward, 1978); ENG3/ENG4 arm makers (this repo). No other prior
art known to me.
"""
import datetime
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TC = os.path.join(ROOT, "tensor_cuda")
SO = "_tensor_cuda.cpython-312-x86_64-linux-gnu.so"
F1 = os.path.join(ROOT, "artifacts/eng4/f1")

_spec = importlib.util.spec_from_file_location("eng4_make_arms", os.path.join(ROOT, "scripts/eng4_make_arms.py"))
_eng4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eng4)
mark = _eng4.mark


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


HUNKS = {f"{k[:3]}_f1": dict(v, eng4_id=k) for k, v in _eng4.MUTANTS.items()}
for k, v in HUNKS.items():  # the marker text names the F1 arm
    eid = v["eng4_id"]
    if isinstance(v["new"], list):
        v["new"] = [n.replace(f"ENG4-MUTANT {eid} executed", f"ENG4-MUTANT {k} executed") for n in v["new"]]
    else:
        v["new"] = v["new"].replace(f"ENG4-MUTANT {eid} executed", f"ENG4-MUTANT {k} executed")
HUNKS["m06_dtype_after_alloc"] = {
    "file": "src/matmul.cu",
    "what": "the supported-dtype check moves back after the output allocation (the dispatch's final else)",
    "old": "  if (a.dtype != DType::Float32 && a.dtype != DType::Float16 && a.dtype != DType::BFloat16)\n"
           "    return refuse(p, MatmulOperandIssue::UnsupportedDtype,",
    "new": "  " + mark("m06_dtype_after_alloc") + "\n  if (false)\n"
           "    return refuse(p, MatmulOperandIssue::UnsupportedDtype,",
    "order_source": "H4 m06 (H1, host int64 pair)",
    "named_assertion": "pytest.raises(RuntimeError, match='^matmul supports float32/float16/bfloat16 only: operand "
                       "dtype <d> (A ..., B ...)') -- host K = 0 / empty-output pairs then return a result ('DID NOT "
                       "RAISE'); the zero-element CUDA pair reaches its output allocation (cudaMalloc error, regex "
                       "mismatch). The nonempty K > 0 host pair is NOT a killing test: there the late refusal follows "
                       "a host malloc (no CUDA call) with the same message, so this mutant cannot be seen through it.",
    "run": "CPU",
}
HUNKS["m07_byteorder"] = {
    "file": "src/bindings.cpp",
    "what": "numpy_dtype's byte-order check is dropped",
    "old": "  if (!dt.attr(\"isnative\").cast<bool>())\n",
    "new": "  " + mark("m07_byteorder") + "\n  if (false)\n",
    "order_source": "H4 m07 (H1)",
    "named_assertion": "pytest.raises(TypeError, match='non-native byte order NumPy dtype <d>') -- 'DID NOT RAISE'",
    "run": "CPU",
}
HUNKS["m08_k0_gemm"] = {
    "file": "src/matmul.cu",
    "what": "the forward's K = 0 early return is dropped (K = 0 reaches the GEMM with lda = 0)",
    "old": "  if (plan.k == 0) return NDArray::zeros(plan.out_shape, a.dtype, a.device);\n",
    "new": "  " + mark("m08_k0_gemm") + "\n  if (false) return NDArray::zeros(plan.out_shape, a.dtype, a.device);\n",
    "order_source": "H4 m08 (a staged GPU line; it must fail on the zero check)",
    "named_assertion": "the zero check of test_gpu_zero_size_returns_numpy_result[<K0 cells>] (GPU line mut_m08): "
                       "pytest.fail('zero check: ...') when the call raises (cuBLAS refuses lda = 0, now checked) or "
                       "'zero check: K = 0 must give positive zeros' if it returns unwritten memory. CPU part: the "
                       "same zero check in test_h1_zero_size_returns_numpy_result[<K0 cells>] (no device: the GEMM "
                       "path raises the CUDA error).",
    "run": "CPU part + staged GPU line mut_m08",
}


def pairs(m):
    return list(zip(m["old"], m["new"])) if isinstance(m["old"], list) else [(m["old"], m["new"])]


def register(map_path):
    out = os.path.join(F1, "MUTANTS_F1.json")
    if os.path.exists(out):
        raise SystemExit("MUTANTS_F1.json exists; registrations are write-once")
    tests = json.load(open(map_path))
    for mid, m in HUNKS.items():
        s = open(os.path.join(TC, m["file"])).read()
        for old, _ in pairs(m):
            assert s.count(old) == 1, (mid, s.count(old))
        m["killing_tests"] = tests[mid]
        assert m["killing_tests"], mid
    rec = {"written_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "rule": "each mutant must fail its named assertion in each killing test; the expected failures come from "
                   "the order (H4, L47); each killing test's captured stderr must show the mutated line ran (L41)",
           "base": {"tree": "Build F1 sources", "matmul.cu": sha(os.path.join(TC, "src/matmul.cu")),
                    "bindings.cpp": sha(os.path.join(TC, "src/bindings.cpp"))},
           "mutants": HUNKS}
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    print("registered", out, sha(out))


def tree(ids):
    os.makedirs(os.path.join(F1, "mutants"), exist_ok=True)
    for mid in ids:
        m = HUNKS[mid]
        dst = f"/tmp/eng4_src_{mid}/tensor_cuda"
        if os.path.exists(dst):
            raise SystemExit(f"{dst} exists; refusing to overwrite (no deletes)")
        os.makedirs(dst)
        shutil.copy2(os.path.join(TC, "CMakeLists.txt"), dst)
        for d in ("src", "include"):
            shutil.copytree(os.path.join(TC, d), os.path.join(dst, d))
        shutil.copytree(os.path.join(TC, "tensor_cuda"), os.path.join(dst, "tensor_cuda"),
                        ignore=shutil.ignore_patterns("*.so", "__pycache__", "*.eng4tmp"))
        path = os.path.join(dst, m["file"])
        s = open(path).read()
        for old, new in pairs(m):
            assert s.count(old) == 1, (mid, s.count(old))
            s = s.replace(old, new)
        open(path, "w").write(s)
        r = subprocess.run(["diff", "-u", os.path.join(TC, m["file"]), path], capture_output=True, text=True)
        with open(os.path.join(F1, "mutants", f"{mid}.diff"), "w") as f:
            f.write(r.stdout)
        print(mid, "->", dst, "diff lines", len(r.stdout.splitlines()))


def pkg(arm, src_pkg):
    dst = os.path.join(TC, "build-eng4", arm, "pkg", "tensor_cuda")
    if os.path.exists(dst):
        raise SystemExit(f"{dst} exists; refusing to overwrite (no deletes)")
    shutil.copytree(src_pkg, dst, ignore=shutil.ignore_patterns("__pycache__", "*.so", "*.eng4tmp"))
    mod = os.path.join(TC, "build-eng4", arm, "module", SO)
    want = json.load(open(os.path.join(F1, "build", f"BUILD_{arm}.json")))["module_sha256"]
    assert sha(mod) == want, (arm, "module copy differs from its build receipt")
    shutil.copy2(mod, os.path.join(dst, SO))
    files = {os.path.relpath(os.path.join(d, n), dst): sha(os.path.join(d, n))
             for d, _, ns in os.walk(dst) for n in sorted(ns)}
    rec = {"arm": arm, "source": src_pkg, "pkg": dst, "module_sha256": sha(os.path.join(dst, SO)), "files": files}
    with open(os.path.join(TC, "build-eng4", arm, "pkg.json"), "w") as f:
        json.dump(rec, f, indent=1)
    print(arm, "pkg", dst, rec["module_sha256"])


if __name__ == "__main__":
    {"register": lambda: register(sys.argv[2]), "tree": lambda: tree(sys.argv[2:]),
     "pkg": lambda: pkg(sys.argv[2], sys.argv[3])}[sys.argv[1]]()
