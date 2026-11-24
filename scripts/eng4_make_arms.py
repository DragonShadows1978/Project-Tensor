"""ENG4 mutant arms (G5) and the frozen package copies the GPU lines import.

  python3 -B -P scripts/eng4_make_arms.py register        -> artifacts/eng4/MUTANTS.json (before any mutant is built)
  python3 -B -P scripts/eng4_make_arms.py tree <id>...    -> /tmp/eng4_src_<id>/tensor_cuda: Build E's sources with one
                                                             registered hunk; diff in artifacts/eng4/mutants/<id>.diff
  python3 -B -P scripts/eng4_make_arms.py pkg <arm> <package dir holding the built .so>
                                                          -> tensor_cuda/build-eng4/<arm>/pkg/tensor_cuda/ (the package
                                                             .py files + that arm's module), sha256 in pkg.json

Each hunk also prints "ENG4-MUTANT <id> executed" to stderr every time the mutated line runs (L41), so the killing
test's captured stderr (junit system-err) shows the mutated line executed in that test.

Prior art: mutation testing (DeMillo, Lipton and Sayward, 1978): one registered syntactic change per mutant; ENG3's
eng3_make_arms.py (this repo) for the tree layout and the executed marker. No other prior art known to me.
"""
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TC = os.path.join(ROOT, "tensor_cuda")
SO = "_tensor_cuda.cpython-312-x86_64-linux-gnu.so"


def mark(i):
    return 'std::fprintf(stderr, "ENG4-MUTANT %s executed\\n"); std::fflush(stderr); ' % i


T = "tensor_cuda/tests/test_eng4_matmul.py::"
MUTANTS = {
    "m01_count_not_shape": {
        "file": "src/matmul.cu",
        "what": "the shared predicate compares batch COUNTS at equal ranks, not shapes",
        "old": ("    for (int d = 0; d < nda - 2; ++d)\n"
                "      if (a.shape[d] != b.shape[d])\n"
                "        return refuse(p, MatmulOperandIssue::BatchShape,\n"),
        "new": ("    int64_t eng4_bcount = 1;\n"
                "    for (int d = 0; d < ndb - 2; ++d) eng4_bcount *= b.shape[d];\n"
                "    " + mark("m01_count_not_shape") + "\n"
                "      if (eng4_bcount != p.batch)\n"
                "        return refuse(p, MatmulOperandIssue::BatchShape,\n"),
        "killing_tests": [T + "test_g1_m3_refuses[eq_count_diff_shape]"],
        "named_assertion": "pytest.raises(RuntimeError, match='^matmul refused: .*batch.*(A [2, 3, 2, 3] on ..., B "
                           "[3, 2, 3, 4] on ...') -- the mutant accepts the equal-count pair, so the call reaches the "
                           "GEMM and raises the no-device CUDA error instead (regex does not match)",
        "order_source": "G5 m01 (G1 equal-count case)",
        "run": "CPU",
    },
    "m02_no_device_check": {
        "file": "src/matmul.cu",
        "what": "the forward's predicate drops the device check",
        "old": ("  if (a.device.type != b.device.type || a.device.index != b.device.index)\n"
                "    return refuse(p, MatmulOperandIssue::Device, \"operands are on different devices\", a, b, "
                "trans_b);\n"),
        "new": "  " + mark("m02_no_device_check") + "\n",
        "killing_tests": [T + "test_g1_m4_device_refuses[host_a_cuda_b]", T + "test_g1_m4_device_refuses[cuda_a_host_b]",
                          T + "test_gpu_m4_device_refuses_nonzero_operands[cuda_a_host_b]",
                          T + "test_gpu_m4_device_refuses_nonzero_operands[host_a_cuda_b]"],
        "named_assertion": "pytest.raises(RuntimeError, match='^matmul refused: operands are on different devices "
                           "(A ... on <dev>, B ... on <dev>') -- CPU: the zero-element CUDA operand lets the call reach "
                           "the output allocation / GEMM (CUDA error); GPU line: nonzero host + CUDA operands reach "
                           "cuBLAS (any outcome other than the refusal fails the regex or 'DID NOT RAISE')",
        "order_source": "G5 m02 (G1, mixed host and CUDA operands, staged GPU line)",
        "run": "CPU witness + staged GPU line mut_m02",
    },
    "m03_no_intmax": {
        "file": "src/matmul.cu",
        "what": "the forward's predicate drops the INT_MAX checks (M, N, K and the batch count)",
        "old": [("  if (p.m > INT_MAX || p.n > INT_MAX || p.k > INT_MAX)\n"
                 "    return refuse(p, MatmulOperandIssue::Dim,\n"),
                ("  if (p.batch > INT_MAX)\n"
                 "    return refuse(p, MatmulOperandIssue::BatchCount,\n")],
        "new": [("  " + mark("m03_no_intmax") + "\n  if (false)\n"
                 "    return refuse(p, MatmulOperandIssue::Dim,\n"),
                ("  if (false)\n"
                 "    return refuse(p, MatmulOperandIssue::BatchCount,\n")],
        "killing_tests": [T + f"test_g1_m4_size_refuses[{c}]" for c in
                          ("M_over_intmax", "N_over_intmax", "N_over_intmax_trans", "K_over_intmax",
                           "K_over_intmax_trans", "batch_over_intmax_b2d", "batch_over_intmax_eq_rank")],
        "named_assertion": "pytest.raises(RuntimeError, match='^matmul refused: GEMM (dimension|batch count) above "
                           "INT_MAX.*(A ..., B ...') -- the mutant lets the call reach the GEMM (no-device CUDA error)",
        "order_source": "G5 m03 (G1)",
        "run": "CPU",
    },
    "m04_ones_any_rank": {
        "file": "src/matmul.cu",
        "what": "the predicate accepts an all-ones B batch at equal ranks (as a stride-0 broadcast)",
        "old": ("    for (int d = 0; d < nda - 2; ++d)\n"
                "      if (a.shape[d] != b.shape[d])\n"
                "        return refuse(p, MatmulOperandIssue::BatchShape,\n"
                "                      \"equal ranks need equal batch shapes, dimension by dimension\", a, b, trans_b);\n"
                "    p.batched_b = true;\n"),
        "new": ("    bool eng4_ones = true;\n"
                "    for (int d = 0; d < ndb - 2; ++d) eng4_ones = eng4_ones && b.shape[d] == 1;\n"
                "    if (eng4_ones) { " + mark("m04_ones_any_rank") + "}\n"
                "    for (int d = 0; d < nda - 2 && !eng4_ones; ++d)\n"
                "      if (a.shape[d] != b.shape[d])\n"
                "        return refuse(p, MatmulOperandIssue::BatchShape,\n"
                "                      \"equal ranks need equal batch shapes, dimension by dimension\", a, b, trans_b);\n"
                "    p.batched_b = !eng4_ones;\n"),
        "killing_tests": [T + "test_g1_m3_refuses[eq_rank_ones_b_under_larger_a]",
                          T + "test_g1_m3_refuses[eq_rank_ones_b_under_larger_a_rank4]"],
        "named_assertion": "pytest.raises(RuntimeError, match='^matmul refused: .*batch.*(A [2, 2, 3] ..., B [1, 3, 4] "
                           "...') (and the rank-4 case) -- the mutant accepts, the call reaches the GEMM",
        "order_source": "G5 m04 (G1, M3 case)",
        "run": "CPU",
    },
    "m05_numpy_dtype_fallback": {
        "file": "src/bindings.cpp",
        "what": "numpy_dtype returns float32 for unlisted dtypes again",
        "old": "  throw py::type_error(\"tensor: unsupported NumPy dtype \"",
        "new": "  " + mark("m05_numpy_dtype_fallback") + "return DType::Float32;\n  throw py::type_error(\"tensor: "
               "unsupported NumPy dtype \"",
        "killing_tests": [T + f"test_g1_n1_refuses_unlisted[{d}-{e}]" for d in ("float64", "int32", "uint16")
                          for e in ("_C.tensor", "from_numpy")],
        "named_assertion": "pytest.raises(TypeError, match='unsupported NumPy dtype <name>') -- the mutant returns a "
                           "float32 tensor: 'DID NOT RAISE'",
        "order_source": "G5 m05 (G1, N1)",
        "run": "CPU",
    },
}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def pairs(m):
    olds, news = m["old"], m["new"]
    return list(zip(olds, news)) if isinstance(olds, list) else [(olds, news)]


def register():
    out = os.path.join(ROOT, "artifacts/eng4/MUTANTS.json")
    if os.path.exists(out):
        raise SystemExit("MUTANTS.json exists; registrations are write-once")
    for mid, m in MUTANTS.items():  # every hunk must apply exactly once to Build E's source
        s = open(os.path.join(TC, m["file"])).read()
        for old, _ in pairs(m):
            assert s.count(old) == 1, (mid, s.count(old))
    rec = {"written_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "rule": "each mutant must fail its named assertion in each killing test (order G5, L41, L47: expected "
                   "failures come from the order, not from this round's runs)",
           "base": {"tree": "Build E sources", "matmul.cu": sha(os.path.join(TC, "src/matmul.cu")),
                    "bindings.cpp": sha(os.path.join(TC, "src/bindings.cpp"))},
           "marker": "stderr line 'ENG4-MUTANT <id> executed' each time the mutated line runs",
           "mutants": MUTANTS}
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    print("registered", out, sha(out))


def tree(ids):
    os.makedirs(os.path.join(ROOT, "artifacts/eng4/mutants"), exist_ok=True)
    for mid in ids:
        m = MUTANTS[mid]
        dst = f"/tmp/eng4_src_{mid}/tensor_cuda"
        if os.path.exists(dst):
            raise SystemExit(f"{dst} exists; refusing to overwrite (no deletes)")
        os.makedirs(dst)
        shutil.copy2(os.path.join(TC, "CMakeLists.txt"), dst)
        for d in ("src", "include"):
            shutil.copytree(os.path.join(TC, d), os.path.join(dst, d))
        shutil.copytree(os.path.join(TC, "tensor_cuda"), os.path.join(dst, "tensor_cuda"),
                        ignore=shutil.ignore_patterns("*.so", "__pycache__"))
        path = os.path.join(dst, m["file"])
        s = open(path).read()
        for old, new in pairs(m):
            assert s.count(old) == 1, (mid, s.count(old))
            s = s.replace(old, new)
        open(path, "w").write(s)
        r = subprocess.run(["diff", "-u", os.path.join(TC, m["file"]), path], capture_output=True, text=True)
        with open(os.path.join(ROOT, f"artifacts/eng4/mutants/{mid}.diff"), "w") as f:
            f.write(r.stdout)
        print(mid, "->", dst, "diff lines", len(r.stdout.splitlines()))


def pkg(arm, src_pkg):
    dst = os.path.join(TC, "build-eng4", arm, "pkg", "tensor_cuda")
    if os.path.exists(dst):
        raise SystemExit(f"{dst} exists; refusing to overwrite (no deletes)")
    # the package's .py files from src_pkg; the module from the arm's receipted copy (never the installed one)
    shutil.copytree(src_pkg, dst, ignore=shutil.ignore_patterns("__pycache__", "*.so", "*.eng4tmp"))
    mod = os.path.join(TC, "build-eng4", arm, "module", SO)
    want = json.load(open(os.path.join(ROOT, f"artifacts/eng4/build/BUILD_{arm}.json")))["module_sha256"]
    assert sha(mod) == want, (arm, "module copy differs from its build receipt")
    shutil.copy2(mod, os.path.join(dst, SO))
    files = {os.path.relpath(os.path.join(d, n), dst): sha(os.path.join(d, n))
             for d, _, ns in os.walk(dst) for n in sorted(ns)}
    rec = {"arm": arm, "source": src_pkg, "pkg": dst, "module_sha256": sha(os.path.join(dst, SO)), "files": files}
    with open(os.path.join(TC, "build-eng4", arm, "pkg.json"), "w") as f:
        json.dump(rec, f, indent=1)
    print(arm, "pkg", dst, rec["module_sha256"])


if __name__ == "__main__":
    if sys.argv[1] == "register":
        register()
    elif sys.argv[1] == "tree":
        tree(sys.argv[2:])
    elif sys.argv[1] == "pkg":
        pkg(sys.argv[2], sys.argv[3])
