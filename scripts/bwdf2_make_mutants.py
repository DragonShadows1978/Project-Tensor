"""BWD-F2 K4: the six mutants of orders/BWDF2_REV1_FIXES.md. Each mutant's source tree is Build F2's tree
(/tmp/bwdf2_src_F2) plus exactly one edit; MUTANTS_F2.json (written by `spec` before any mutant tree exists or runs)
names each mutant's edit, where its row runs (CPU or a staged GPU line) and the assertion it must fail.

  python3 -B -P scripts/bwdf2_make_mutants.py spec      # artifacts/bwdf1/f2/MUTANTS_F2.json (refuses if it exists)
  python3 -B -P scripts/bwdf2_make_mutants.py trees     # /tmp/bwdf2_src_<mutant> + artifacts/bwdf1/f2/mutants/*.diff

Prior art: mutation testing (DeMillo, Lipton and Sayward, 1978); scripts/bwdf1_make_mutants.py (this repo), copied.
Ours: the edits and their killing assertions.
"""
import datetime
import difflib
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F2 = "/tmp/bwdf2_src_F2"
OUT = os.path.join(ROOT, "artifacts/bwdf1/f2/mutants")
SPEC = os.path.join(ROOT, "artifacts/bwdf1/f2/MUTANTS_F2.json")
K = "tensor_cuda/src/kernels.cu"
T = "tensor_cuda/tests/test_bwdf2_fixes.py::test_cpu_k1_row"

EDITS = {
    "n01_dtype_check_all_members": dict(
        ruling="R1", file=K,
        old="    if (a.numel() > 0 && a.dtype != a0.dtype)\n",
        new="    if (a.dtype != a0.dtype)  // MUTANT n01_dtype_check_all_members\n",
        where="CPU",
        kills=[f"{T}[f03-cat-named-case]", f"{T}[f03-stack-empty-mixed-dtype]",
               f"{T}[f03-cat-empty-mixed-device-dtype]"],
        assertion="verdict(row, r) is None: the row expects the call to pass the host checks and reach the device "
                  "step ('CUDA error at cat: no CUDA-capable device'); the mutant refuses with 'operand 1 has dtype "
                  "float16/bfloat16' (did not reach the device step)",
        gpu_line=None),
    "n02_no_rank_ceiling_stack": dict(
        ruling="R2", file=K,
        old="  check_meta_rank(op, nd + (all_extents ? 1 : 0), \"operand 0 \" + copy_operand_text(a0));\n",
        new="  // MUTANT n02_no_rank_ceiling_stack: the copy ops' rank ceiling (stack's post-unsqueeze rank) is deleted\n",
        where="CPU",
        kills=[f"{T}[f02-stack-rank8-to-9]", f"{T}[f02-cat-rank9-empty]"],
        assertion="verdict(row, r) is None: the row expects RuntimeError 'stack: rank 9 exceeds TC_MAX_DIMS=8' with 0 "
                  "CUDA calls; the mutant writes DimCopySpec slot 8 and reaches cat's status check ('CUDA error at "
                  "cat') or dies",
        gpu_line=None),
    "n03_sin_span_unchecked": dict(
        ruling="R3", file=K,
        old=" || pos0 > sn.shape[0] ||\n      rows > sn.shape[0] - pos0)",
        new=")  // MUTANT n03_sin_span_unchecked",
        where="CPU (and the staged K2 memory line k2_mem_n03 for the GPU read)",
        kills=[f"{T}[f01-rope_apply-short-sin]", f"{T}[f01-swap-short-sin]", f"{T}[f01-arena-short-sin]"],
        assertion="verdict(row, r) is None: the row expects RuntimeError 'rope_apply: position past table end' "
                  "naming 'sin [1, 2]'; the mutant returns",
        gpu_line="k2_mem_n03: F01 rows (swap with sin [1, 2] on [2, 2] inserts) not refused; memcheck reports the "
                 "read past sin"),
    "n04_ring_no_modulo": dict(
        ruling="R4", file=K,
        old="  if (n) start %= CAP;\n",
        new="  // MUTANT n04_ring_no_modulo: start is not reduced modulo CAP on the host\n",
        where="GPU (staged line k2_mem_n04; the kernel arithmetic is device-only)",
        kills=["scripts/bwdf2_step_k2.py mem rows f04-start-int64max and f04-start-cap-plus-2"],
        assertion="the row's verdict: buf after the write equals NumPy's ring write at (start + l) % CAP, and memcheck "
                  "reports 0 errors; the mutant's wrap leaves [0, CAP) (writes outside buf: memcheck errors and/or "
                  "wrong rows)",
        gpu_line="k2_mem_n04"),
    "n05_condition_as_data_dtype": dict(
        ruling="R5", file=K,
        old="  st<T>(out, idx, cond_at<C>(c, co) ? ld<T>(x, xo) : ld<T>(y, yo));\n",
        new="  st<T>(out, idx, ld<T>(reinterpret_cast<const T*>(c), co) != 0.f ? ld<T>(x, xo) : ld<T>(y, yo));  "
            "// MUTANT n05_condition_as_data_dtype\n",
        where="GPU (staged line k2_mem_n05; the condition load is device-only)",
        kills=["scripts/bwdf2_step_k2.py mem rows f07-where-bool-cond and f07-masked_fill-bool-mask"],
        assertion="the row's verdict: the result equals np.where(cond, x, y) bitwise (cond [False, True, True, True] "
                  "bool); the mutant reads the 4-byte bool condition as float32 values (element 0 takes x: bytes "
                  "00 01 01 01 are a nonzero float32), and memcheck reports the read past the condition",
        gpu_line="k2_mem_n05"),
    "n06_axpy_broadcast_accepted": dict(
        ruling="R6", file=K,
        old="    if (strip(upd_shape) != strip(param.shape))\n",
        new="    if (false)  // MUTANT n06_axpy_broadcast_accepted\n",
        where="CPU",
        kills=[f"{T}[f12-axpy-named-case]", f"{T}[f12-axpy-trailing-broadcast]"],
        assertion="verdict(row, r) is None: the row expects RuntimeError 'does not map element for element onto "
                  "param' with 0 CUDA calls; the mutant launches ew_scalar ('CUDA error at ew_scalar')",
        gpu_line=None),
}


def spec():
    if os.path.exists(SPEC):
        raise SystemExit(f"{SPEC} exists")
    doc = {"order": "BWD-F2 K4", "written": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "written_before_any_mutant_tree_or_run": True, "base": "Build F2's tree (/tmp/bwdf2_src_F2)",
           "rule": "each mutant = Build F2 + exactly one edit (the `old` text occurs once); KILLED when every listed "
                   "killing assertion fails on the mutant (CPU rows), or when its staged GPU line's verdict fails",
           "mutants": EDITS}
    json.dump(doc, open(SPEC, "w"), indent=1)
    print("spec", len(EDITS))


def trees():
    os.makedirs(OUT, exist_ok=True)
    for name, e in EDITS.items():
        dst = f"/tmp/bwdf2_src_{name}"
        if os.path.exists(dst):
            raise SystemExit(f"{dst} exists")
        shutil.copytree(F2, dst, symlinks=False)
        p = os.path.join(dst, e["file"])
        src = open(p).read()
        if src.count(e["old"]) != 1:
            raise SystemExit(f"{name}: old text occurs {src.count(e['old'])} times")
        mut = src.replace(e["old"], e["new"])
        open(p, "w").write(mut)
        diff = "".join(difflib.unified_diff(src.splitlines(True), mut.splitlines(True), "a/" + e["file"],
                                            "b/" + e["file"]))
        open(os.path.join(OUT, f"{name}.diff"), "w").write(diff)
        print("TREE", name, len(diff.splitlines()), "diff lines")


if __name__ == "__main__":
    {"spec": spec, "trees": trees}[sys.argv[1]]()
