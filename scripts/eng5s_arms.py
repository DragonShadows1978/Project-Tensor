"""ENG5-S arms: mutant registration (artifacts/eng5s/MUTANTS.json), source trees, builds and per-arm packages.

  python3 -B -P scripts/eng5s_arms.py register         # MUTANTS.json, written once, before any mutant builds or runs
  python3 -B -P scripts/eng5s_arms.py build <arm..>    # /tmp/eng5s_src_<arm> + scripts/eng5s_build.py
                                                       #   arms: s01..s04 (mutants of Build S), B_H, S_H (hooks arms)
  python3 -B -P scripts/eng5s_arms.py pkgs             # tensor_cuda/build-eng5s/<arm>/pkg for every built arm
  python3 -B -P scripts/eng5s_arms.py results          # artifacts/eng5s/MUTANT_RESULTS.json from the staged lines

Arms:
- B: fbd335ae7 (`git archive`, read-only git) in /tmp/eng5s_src_B; S: this worktree's tensor_cuda/ (this round).
- B_H / S_H: B and S with the repository's own test-hook line appended to a COPY of CMakeLists.txt
  (`target_compile_definitions(_tensor_cuda PRIVATE TC_APA_P1_TEST_HOOKS=1)`, the line scripts/apa_p1f1_build.py
  appends). Site 9 (apa_spo/packed_prefill.cuh, the dump's NaN fill) runs only when dump=True, and dump is refused
  before it in a build without the hooks, so its G2 row needs these two arms. Engine sources are unchanged.
- s01..s04: the order's four mutants (G4), each one registered hunk applied to a copy of Build S's sources. Each hunk
  writes an executed-marker line (the mutant id) to $ENG5S_MUTANT_MARKER when the mutated statement runs (L41).
  Expectations (the failing G2 row) come from the order (L41, L47), not from runs.

Prior art: mutation testing (DeMillo, Lipton and Sayward, 1978); ENG5-F2's scripts/eng5_f2_mutants.py (eng5 branch),
whose register / tree / package layout this copies. Ours: the four hunks and the hooks arms. No other prior art known
to me.
"""
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SO = "_tensor_cuda.cpython-312-x86_64-linux-gnu.so"
REG = os.path.join(ROOT, "artifacts/eng5s/MUTANTS.json")
RES = os.path.join(ROOT, "artifacts/eng5s/MUTANT_RESULTS.json")
K = "tensor_cuda/src/kernels.cu"
SCR = "/tmp/eng5s"
SRC_B = "/tmp/eng5s_src_B/tensor_cuda"
HOOKS_LINE = ('\n# Test-only hook; never enabled by the shipped CMake default.\n'
              'target_compile_definitions(_tensor_cuda PRIVATE TC_APA_P1_TEST_HOOKS=1)\n')
MARK = ('{ if (const char* p_ = ::getenv("ENG5S_MUTANT_MARKER")) { if (FILE* f_ = ::fopen(p_, "a")) '
        '{ ::fputs("%s\\n", f_); ::fclose(f_); } } }')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def mutants():
    return [
        {"id": "s01_clone_status_dropped", "arm": "s01_clone_status_dropped", "file": K,
         "site": "site 6, NDArray::clone",
         "old": ('    if (device.is_cuda())\n'
                 '      check_cuda_status(cudaMemcpy(out.data_ptr(), data_ptr(), bytes, cudaMemcpyDeviceToDevice), '
                 '"NDArray::clone",\n                        "cudaMemcpy");\n'),
         "new": ('    if (device.is_cuda()) { ' + MARK % "s01_clone_status_dropped" +
                 ' cudaMemcpy(out.data_ptr(), data_ptr(), bytes, cudaMemcpyDeviceToDevice); }\n'),
         "killing_row": "clone", "killing_line": "g3_s01",
         "named_assertion": "G2 row `clone`: with the injected cudaMemcpy failure fired, the call must raise "
                            "RuntimeError starting 'NDArray::clone: cudaMemcpy failed: cudaError'; the mutant returns "
                            "(scripts/eng5s_gpu_verdict.py, row_pass)"},
        {"id": "s02_to_status_dropped", "arm": "s02_to_status_dropped", "file": K,
         "site": "site 7, NDArray::to",
         "old": '  if (bytes) check_cuda_status(cudaMemcpy(out.data_ptr(), data_ptr(), bytes, k), "NDArray::to", '
                '"cudaMemcpy");\n',
         "new": ('  if (bytes) { ' + MARK % "s02_to_status_dropped" +
                 ' cudaMemcpy(out.data_ptr(), data_ptr(), bytes, k); }\n'),
         "killing_row": "to_d2h and to_h2d", "killing_line": "g3_s02",
         "named_assertion": "G2 rows `to_d2h` and `to_h2d`: with the injected cudaMemcpy failure fired, each must raise "
                            "RuntimeError starting 'NDArray::to: cudaMemcpy failed: cudaError'; the mutant returns"},
        {"id": "s03_zeros_status_dropped", "arm": "s03_zeros_status_dropped", "file": K,
         "site": "site 3, NDArray::zeros",
         "old": '      check_cuda_status(cudaMemset(a.data_ptr(), 0, a.numel() * dtype_size(dtype)), "NDArray::zeros", '
                '"cudaMemset");\n',
         "new": ('      { ' + MARK % "s03_zeros_status_dropped" +
                 ' cudaMemset(a.data_ptr(), 0, a.numel() * dtype_size(dtype)); }\n'),
         "killing_row": "zeros", "killing_line": "g3_s03",
         "named_assertion": "G2 row `zeros`: with the injected cudaMemset failure fired, the call must raise "
                            "RuntimeError starting 'NDArray::zeros: cudaMemset failed: cudaError'; the mutant returns"},
        {"id": "s04_from_host_last_error_only", "arm": "s04_from_host_last_error_only", "file": K,
         "site": "site 4, NDArray::from_host",
         "old": ('    check_cuda_status(cudaMemcpy(a.data_ptr(), src, bytes, cudaMemcpyHostToDevice), '
                 '"NDArray::from_host", "cudaMemcpy");\n    cuda_check_last("from_host");\n'),
         "new": ('    ' + MARK % "s04_from_host_last_error_only" +
                 '\n    cudaMemcpy(a.data_ptr(), src, bytes, cudaMemcpyHostToDevice);\n    cuda_check_last("from_host");\n'),
         "killing_row": "from_host", "killing_line": "g3_s04",
         "named_assertion": "G2 row `from_host`: the injected cudaMemcpy failure records nothing in the last-error "
                            "slot, so a slot-only check misses it: the call must raise RuntimeError starting "
                            "'NDArray::from_host: cudaMemcpy failed: cudaError'; the mutant returns"},
    ]


def register():
    if os.path.exists(REG):
        raise SystemExit("MUTANTS.json exists: registrations are written once")
    ms = mutants()
    for m in ms:
        src = open(os.path.join(ROOT, m["file"])).read()
        assert src.count(m["old"]) == 1, (m["id"], src.count(m["old"]))
    doc = {"order": "orders/ENG5S_STATUS_CHECKS.md (G4, L41, L47, L77)", "written": now(),
           "base": "Build S sources (this tree); each hunk is applied to a copy in /tmp/eng5s_src_<arm>",
           "source_sha256": {f: sha(os.path.join(ROOT, f)) for f in sorted({m["file"] for m in ms})},
           "kill_rule": "KILLED when the named G2 row's injection fired and the call returned (outcome ok) on the "
                        "mutant arm, so the row's Build S assertion fails; SURVIVED otherwise. Each mutant is run as "
                        "its staged GPU line (L77); its executed marker is recorded.",
           "mutants": ms}
    json.dump(doc, open(REG, "w"), indent=1)
    print("registered", len(ms), doc["written"])


def reg(mid):
    return dict(next(x for x in json.load(open(REG))["mutants"] if x["id"] == mid or x["id"].startswith(mid + "_")))


def copy_tree(src_tc, dst):
    if os.path.exists(dst):
        raise SystemExit(f"{dst} exists")
    os.makedirs(dst + "/tensor_cuda")
    for sub in ("CMakeLists.txt", "src", "include", "tensor_cuda"):
        s = os.path.join(src_tc, sub)
        d = os.path.join(dst, "tensor_cuda", sub)
        if os.path.isdir(s):
            shutil.copytree(s, d, ignore=shutil.ignore_patterns("*.so", "*.so.*", "__pycache__"))
        else:
            shutil.copy2(s, d)
    return dst + "/tensor_cuda"


def make_tree(arm):
    if arm in ("B_H", "S_H"):
        tree = copy_tree(SRC_B if arm == "B_H" else os.path.join(ROOT, "tensor_cuda"), f"{SCR}_src_{arm}")
        with open(os.path.join(tree, "CMakeLists.txt"), "a") as f:
            f.write(HOOKS_LINE)
        return tree
    m = reg(arm)
    tree = copy_tree(os.path.join(ROOT, "tensor_cuda"), f"{SCR}_src_{m['arm']}")
    p = os.path.join(os.path.dirname(tree), m["file"])
    src = open(p).read()
    assert sha(os.path.join(ROOT, m["file"])) == json.load(open(REG))["source_sha256"][m["file"]], "S source moved"
    assert src.count(m["old"]) == 1, arm
    open(p, "w").write(src.replace(m["old"], m["new"]))
    return tree


def arm_name(a):
    return a if a in ("B", "S", "B_H", "S_H") else reg(a)["arm"]


def build(arm):
    tree = make_tree(arm)
    r = subprocess.run([sys.executable, "-B", "-P", os.path.join(ROOT, "scripts/eng5s_build.py"), arm_name(arm), tree],
                       cwd=ROOT, timeout=3500)
    if r.returncode:
        raise SystemExit(f"build {arm} rc {r.returncode}")


def pkg_source(arm):
    if arm == "B":
        return os.path.join(SRC_B, "tensor_cuda")
    if arm == "S":
        return os.path.join(ROOT, "tensor_cuda/tensor_cuda")
    return f"{SCR}_src_{arm}/tensor_cuda/tensor_cuda"


def pkgs():
    arms = ["B", "S", "B_H", "S_H"] + [m["arm"] for m in json.load(open(REG))["mutants"]]
    for arm in arms:
        rp = os.path.join(ROOT, f"artifacts/eng5s/build/BUILD_{arm}.json")
        if not os.path.exists(rp):
            print(arm, "not built")
            continue
        rec = json.load(open(rp))
        dst = os.path.join(ROOT, "tensor_cuda/build-eng5s", arm, "pkg", "tensor_cuda")
        if os.path.exists(dst):
            print(arm, "pkg exists; kept")
            continue
        shutil.copytree(pkg_source(arm), dst, ignore=shutil.ignore_patterns("*.so", "*.so.*", "__pycache__"))
        shutil.copy2(os.path.join(ROOT, "tensor_cuda/build-eng5s", arm, "module", SO), os.path.join(dst, SO))
        assert sha(os.path.join(dst, SO)) == rec["module_sha256"], arm
        json.dump({"arm": arm, "time": now(), "package_source": pkg_source(arm), "module_sha256": rec["module_sha256"]},
                  open(os.path.join(ROOT, "tensor_cuda/build-eng5s", arm, "pkg.json"), "w"), indent=1)
        print(arm, "pkg", rec["module_sha256"])


def results():
    """MUTANT_RESULTS.json from the staged lines' verdicts (STAGED until the lead has run them)."""
    out = {"written": now(), "registration": "artifacts/eng5s/MUTANTS.json", "mutants": []}
    for m in json.load(open(REG))["mutants"]:
        line = m["killing_line"]
        vp = os.path.join(ROOT, "artifacts/eng5s/gpu", line, "VERDICT.json")
        rec = {"id": m["id"], "killing_line": line, "killing_row": m["killing_row"],
               "module_sha256": json.load(open(os.path.join(ROOT, f"artifacts/eng5s/build/BUILD_{m['arm']}.json")))
               ["module_sha256"]}
        if os.path.exists(vp):
            v = json.load(open(vp))
            rec["status"] = v.get("verdict")
            rec["mutant_assertion"] = v.get("mutant_assertion")
        else:
            rec["status"] = "STAGED (GPU line; the lead runs it)"
        out["mutants"].append(rec)
    json.dump(out, open(RES, "w"), indent=1)
    for r in out["mutants"]:
        print(r["id"], r["status"])


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "register":
        register()
    elif cmd == "build":
        for a in sys.argv[2:]:
            build(a)
    elif cmd == "pkgs":
        pkgs()
    elif cmd == "results":
        results()
    else:
        raise SystemExit(__doc__)
