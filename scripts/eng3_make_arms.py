"""ENG3: source trees for the extra builds, under /tmp/eng3_src_<arm>/ (CMakeLists.txt, src/, include/, the Python
package without its .so), each built out of tree into tensor_cuda/build-eng3/<arm>/.

- Bh: Build B's engine sources (artifacts/eng3/before/) plus ONLY the A1.1 private-hook block of Build E's
  bindings.cpp: the device-path reference for G3's bitwise check of zeros/full/clone/to (Build B has no entry for
  them). No Bool branch, no kernels.cu change.
- m01..m05: Build E's sources with one registered hunk (artifacts/eng3/MUTANTS.json), each also printing
  "ENG3-MUTANT <id> executed" once from the mutated line (L41).
Writes artifacts/eng3/mutants/<arm>.diff (diff -u against the worktree file) and arms.json (sha256 of each tree file).

Prior art: mutation testing (DeMillo, Lipton and Sayward, 1978): one syntactic change per mutant. No other prior art
known to me.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TC = os.path.join(ROOT, "tensor_cuda")


def marker(i):
    return '{ static bool eng3_m = (std::fprintf(stderr, "ENG3-MUTANT %s executed\\n"), true); (void)eng3_m; } ' % i


HUNKS = {
    "m01_from_host_checks_cuda": ("src/kernels.cu",
        "  } else {\n    std::memcpy(a.data_ptr(), src, bytes);\n  }\n  return a;",
        "  } else {\n    std::memcpy(a.data_ptr(), src, bytes);\n    " + marker("m01_from_host_checks_cuda")
        + "cuda_check_last(\"from_host\");\n  }\n  return a;"),
    "m02_host_astype_truncates": ("src/kernels.cu",
        "  const uint32_t lsb = (u >> 16) & 1u;\n  return static_cast<uint16_t>((u + 0x7FFFu + lsb) >> 16);",
        "  " + marker("m02_host_astype_truncates") + "return static_cast<uint16_t>(u >> 16);"),
    "m03_host_clone_cudamemcpy": ("src/kernels.cu",
        "    else std::memcpy(out.data_ptr(), data_ptr(), bytes);  // ENG3 E2: host clone, no CUDA call",
        "    else { " + marker("m03_host_clone_cudamemcpy")
        + "cudaMemcpy(out.data_ptr(), data_ptr(), bytes, cudaMemcpyDeviceToDevice); }"),
    "m04_u8_no_clamp": ("src/kernels.cu",
        "  v = host_rintf(v);\n  v = (v > 0.0f) ? v : 0.0f;\n  v = (v < 255.0f) ? v : 255.0f;\n",
        "  v = host_rintf(v);\n  " + marker("m04_u8_no_clamp") + "\n"),
    "m05_bool_numpy_float": ("src/bindings.cpp",
        "  else if (a.dtype == DType::Bool) out = py::array(py::dtype(\"bool\"), shape);\n",
        "  else if (a.dtype == DType::Bool && ([] { static bool eng3_m = (std::fprintf(stderr, \"ENG3-MUTANT "
        "m05_bool_numpy_float executed\\n\"), true); (void)eng3_m; return false; })()) {}\n"),
}


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def copy_tree(dst):
    if os.path.exists(dst):
        raise SystemExit(f"{dst} exists; refusing to overwrite (no deletes)")
    os.makedirs(dst)
    shutil.copy2(os.path.join(TC, "CMakeLists.txt"), dst)
    shutil.copytree(os.path.join(TC, "src"), os.path.join(dst, "src"))
    shutil.copytree(os.path.join(TC, "include"), os.path.join(dst, "include"))
    shutil.copytree(os.path.join(TC, "tensor_cuda"), os.path.join(dst, "tensor_cuda"),
                    ignore=shutil.ignore_patterns("*.so", "__pycache__"))


def replace_once(path, old, new):
    s = open(path).read()
    assert s.count(old) == 1, (path, s.count(old), old[:60])
    open(path, "w").write(s.replace(old, new))


def main(arms):
    os.makedirs(os.path.join(ROOT, "artifacts/eng3/mutants"), exist_ok=True)
    rec_path = os.path.join(ROOT, "artifacts/eng3/mutants/arms.json")
    rec = json.load(open(rec_path)) if os.path.exists(rec_path) else {}
    for arm in arms:
        dst = f"/tmp/eng3_src_{arm}"
        copy_tree(dst)
        if arm == "Bh":
            before = os.path.join(ROOT, "artifacts/eng3/before")
            shutil.copy2(os.path.join(before, "kernels.cu"), os.path.join(dst, "src/kernels.cu"))
            shutil.copy2(os.path.join(before, "core.h"), os.path.join(dst, "include/tc/core.h"))
            e = open(os.path.join(TC, "src/bindings.cpp")).read()
            start = e.index("  // ENG3 A1.1 (orders/ENG3_R2_RESUME.md)")
            end = e.index("  // P1T-FIX1 (F04/F05): counters of the DLPack import release policy (tests).")
            hooks = e[start:end]
            b = open(os.path.join(before, "bindings.cpp")).read()
            anchor = "  // P1T-FIX1 (F04/F05): counters of the DLPack import release policy (tests)."
            assert b.count(anchor) == 1
            open(os.path.join(dst, "src/bindings.cpp"), "w").write(b.replace(anchor, hooks + anchor))
            diff_against = [("src/bindings.cpp", os.path.join(before, "bindings.cpp")),
                            ("src/kernels.cu", os.path.join(before, "kernels.cu"))]
        else:
            f, old, new = HUNKS[arm]
            replace_once(os.path.join(dst, f), old, new)
            diff_against = [(f, os.path.join(TC, f))]
        with open(os.path.join(ROOT, f"artifacts/eng3/mutants/{arm}.diff"), "w") as out:
            for f, ref in diff_against:
                r = subprocess.run(["diff", "-u", ref, os.path.join(dst, f)], capture_output=True, text=True)
                out.write(r.stdout)
        rec[arm] = {"tree": dst, "files": {os.path.relpath(os.path.join(d, n), dst): sha(os.path.join(d, n))
                                           for d, _, ns in os.walk(dst) for n in ns}}
        print(arm, "->", dst)
    json.dump(rec, open(rec_path, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
