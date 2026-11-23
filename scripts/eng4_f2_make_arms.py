"""ENG4-F2 mutant arms (J4) and the frozen F2 package copies.

  python3 -B -P scripts/eng4_f2_make_arms.py register <killing-test map json> -> artifacts/eng4/f2/MUTANTS_F2.json
  python3 -B -P scripts/eng4_f2_make_arms.py tree <id>...   -> /tmp/eng4_src_<id>/tensor_cuda (F2 sources + one hunk)
  python3 -B -P scripts/eng4_f2_make_arms.py pkg <arm> <package dir with the .py files>
                                                            -> tensor_cuda/build-eng4/<arm>/pkg/tensor_cuda

m09_f2_naive_product reverts Y1 to the plain product (ENG4/F1's code); m10_f2_empty_batch_raise reverts Y2 (the
shared-B empty-batch zero gradient). Each hunk prints "ENG4-MUTANT <id> executed" to stderr every time its line runs
(L41). Killing tests and named assertions come from the order (J4, L47).

Prior art: mutation testing (DeMillo, Lipton and Sayward, 1978); ENG3/ENG4/ENG4-F1 arm makers (this repo).
No other prior art known to me.
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
F2 = os.path.join(ROOT, "artifacts/eng4/f2")


def mark(i):
    return 'std::fprintf(stderr, "ENG4-MUTANT %s executed\\n"); std::fflush(stderr); ' % i


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


Y1_OLD = '''  bool zero_leading = false;
  for (int d = 0; d < nda - 2; ++d) zero_leading = zero_leading || a.shape[d] == 0;
  p.batch = zero_leading ? 0 : 1;
  for (int d = 0; d < nda - 2 && !zero_leading; ++d) {
    const int64_t extent = a.shape[d];  // >= 1 here (no zero; NDArray extents are never negative)
    if (extent > INT_MAX / p.batch)
      return refuse(p, MatmulOperandIssue::BatchCount,
                    "GEMM batch count above INT_MAX: the product of A's leading extents exceeds " +
                        std::to_string(INT_MAX) + " at leading dimension " + std::to_string(d) + " (" +
                        std::to_string(p.batch) + " x " + std::to_string(extent) + ")",
                    a, b, trans_b);
    p.batch *= extent;
  }
'''
Y1_NAIVE = '''  ''' + mark("m09_f2_naive_product") + '''
  p.batch = 1;
  for (int d = 0; d < nda - 2; ++d) p.batch *= a.shape[d];
  if (p.batch > INT_MAX)
    return refuse(p, MatmulOperandIssue::BatchCount,
                  "GEMM batch count above INT_MAX: " + std::to_string(p.batch), a, b, trans_b);
'''
Y2_OLD = '''    if (a.ndim() != b.ndim() &&
        std::any_of(a_shape.begin(), a_shape.end() - 2, [](int64_t dim) { return dim == 0; })) {
'''
Y2_NEW = '''    ''' + mark("m10_f2_empty_batch_raise") + '''
    if (false && a.ndim() != b.ndim() &&
        std::any_of(a_shape.begin(), a_shape.end() - 2, [](int64_t dim) { return dim == 0; })) {
'''
HUNKS = {
    "m09_f2_naive_product": {
        "file": "src/matmul.cu", "old": Y1_OLD, "new": Y1_NAIVE,
        "what": "Y1 reverted: the plain signed product of A's leading extents, then the INT_MAX check (ENG4/F1's code)",
        "order_source": "J4 m09 (J1, the 2^32 x 2^32 x 0 witness)",
        "named_assertion": "test_j1_y1_refuses_before_overflow: pytest.raises(RuntimeError, match='^matmul refused: "
                           "GEMM batch count above INT_MAX: the product of A's leading extents exceeds 2147483647 at "
                           "leading dimension <d> ...') -- the naive product wraps (2^64 -> 0; 3 * 2^62 -> negative), "
                           "the mutant accepts and the call stops at the output byte bound (OverflowError): 'Regex "
                           "pattern did not match' or a non-RuntimeError exception (pytest reports the OverflowError "
                           "as the failure).",
        "unkillable_witness": "the order's 2^32 x 2^32 x 0 witness (test_j1_y1_zero_leading_is_batch0): the naive "
                              "product of a list holding a 0 is also 0 (2^32 * 2^32 wraps to 0 in practice, a signed "
                              "overflow, then * 0), so its assertion (accepted as batch 0, stopping at the output "
                              "byte bound) holds for the mutant too. Reading that shows the overflow: the mutant "
                              "multiplies 2^32 * 2^32 in int64 before reaching the 0 (signed overflow, undefined "
                              "behaviour); the over-2^63 witnesses are the ones a test can see.",
    },
    "m10_f2_empty_batch_raise": {
        "file": "src/ops.cpp", "old": Y2_OLD, "new": Y2_NEW,
        "what": "Y2 reverted: a shared B under an empty batch goes back to reduce_to, which raises",
        "order_source": "J4 m10 (a staged GPU line; it must fail on the zero-gradient check)",
        "named_assertion": "the zero-gradient check of test_j1_y2_empty_batch_shared_b_zero_gradient (CPU, host "
                           "float16/bfloat16) and test_gpu_bwd0_zero_gradient (GPU line mut_m10_f2, float32 included): "
                           "pytest.fail('zero-gradient check: ... the backward raised: reshape: element count "
                           "mismatch')",
    },
}


def register(map_path):
    out = os.path.join(F2, "MUTANTS_F2.json")
    if os.path.exists(out):
        raise SystemExit("MUTANTS_F2.json exists; registrations are write-once")
    tests = json.load(open(map_path))
    for mid, m in HUNKS.items():
        assert open(os.path.join(TC, m["file"])).read().count(m["old"]) == 1, mid
        m["killing_tests"] = tests[mid]
        assert m["killing_tests"], mid
    rec = {"written_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "rule": "each mutant must fail its named assertion in each killing test (order J4, L41, L47); each killing "
                   "test's captured output must show the mutated line ran",
           "base": {"tree": "Build F2 sources", "matmul.cu": sha(os.path.join(TC, "src/matmul.cu")),
                    "ops.cpp": sha(os.path.join(TC, "src/ops.cpp"))},
           "mutants": HUNKS}
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    print("registered", out, sha(out))


def tree(ids):
    os.makedirs(os.path.join(F2, "mutants"), exist_ok=True)
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
        assert s.count(m["old"]) == 1, mid
        open(path, "w").write(s.replace(m["old"], m["new"]))
        r = subprocess.run(["diff", "-u", os.path.join(TC, m["file"]), path], capture_output=True, text=True)
        with open(os.path.join(F2, "mutants", f"{mid}.diff"), "w") as f:
            f.write(r.stdout)
        print(mid, "->", dst, "diff lines", len(r.stdout.splitlines()))


def pkg(arm, src_pkg):
    dst = os.path.join(TC, "build-eng4", arm, "pkg", "tensor_cuda")
    if os.path.exists(dst):
        raise SystemExit(f"{dst} exists; refusing to overwrite (no deletes)")
    shutil.copytree(src_pkg, dst, ignore=shutil.ignore_patterns("__pycache__", "*.so", "*.eng4tmp"))
    mod = os.path.join(TC, "build-eng4", arm, "module", SO)
    want = json.load(open(os.path.join(F2, "build", f"BUILD_{arm}.json")))["module_sha256"]
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
