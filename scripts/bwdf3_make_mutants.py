"""BWD-F3 P4: the four mutants of orders/BWDF3_REV2_FIXES.md. Each mutant's source tree is Build F3's tree
(/tmp/bwdf3_src_F3) plus exactly one edit; MUTANTS_F3.json (written by `spec` before any mutant tree exists or runs)
names each mutant's edit, where its rows run (CPU, and a staged GPU line for p04) and the assertion it must fail.

  python3 -B -P scripts/bwdf3_make_mutants.py spec      # artifacts/bwdf1/f3/MUTANTS_F3.json (refuses if it exists)
  python3 -B -P scripts/bwdf3_make_mutants.py trees     # /tmp/bwdf3_src_<mutant> + artifacts/bwdf1/f3/mutants/*.diff

Prior art: mutation testing (DeMillo, Lipton and Sayward, 1978); scripts/bwdf2_make_mutants.py (this repo), copied.
Ours: the edits and their killing assertions (the order names the four mutants and their rules).
"""
import datetime
import difflib
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F3 = "/tmp/bwdf3_src_F3"
OUT = os.path.join(ROOT, "artifacts/bwdf1/f3/mutants")
SPEC = os.path.join(ROOT, "artifacts/bwdf1/f3/MUTANTS_F3.json")
K = "tensor_cuda/src/kernels.cu"
T = "tensor_cuda/tests/test_bwdf3_fixes.py::test_cpu_p1_row"

EDITS = {
    "p01_empty_span_refused": dict(
        ruling="E1", file=K,
        old="  if (reads && (pos0 < 0 || rows < 0 || pos0 > cs.shape[0] ||",
        new="  if (/* MUTANT p01_empty_span_refused */ (pos0 < 0 || rows < 0 || pos0 > cs.shape[0] ||",
        where="CPU",
        kills=[f"{T}[e1-rope_apply-empty-short-sin]", f"{T}[e1-export_rope_rows-empty-short-sin]",
               f"{T}[e1-swap-empty-short-sin]"],
        assertion="verdict(row, r) is None: the row expects the empty call to return (shape [0, 2, 2] / [0, 2, 2] / a "
                  "tuple) with 0 CUDA calls; the mutant checks the span on every call and refuses with '<op>: "
                  "position past table end' naming 'sin [1, 2]' (the mutated condition's own text)",
        gpu_line=None),
    "p02_trailing_singleton_table_refused": dict(
        ruling="E2", file=K,
        old="  if (t.ndim() < 2 || t.shape[1] != D) return false;\n",
        new="  if (t.ndim() != 2 || t.shape[1] != D) return false;  // MUTANT p02_trailing_singleton_table_refused\n",
        where="CPU",
        kills=[f"{T}[e2-rope_apply-sin-T-D-1]", f"{T}[e2-rope_apply-cos-T-D-1-1-empty]",
               f"{T}[e2-export_rope_rows-sin-T-D-1-empty]"],
        assertion="verdict(row, r) is None: the rows expect a [T, D, 1] (or [T, D, 1, 1]) table to be accepted (the "
                  "reading rope_apply reaches 'CUDA error at rope_apply'; the empty calls return their shapes); the "
                  "mutant refuses with '<op>: tables must be (T, D)'",
        gpu_line=None),
    "p03_duplicate_axis_accepted": dict(
        ruling="E3", file=K,
        old="    if (seen[(size_t)d])\n",
        new="    if (false && seen[(size_t)d])  // MUTANT p03_duplicate_axis_accepted\n",
        where="CPU",
        kills=[f"{T}[e3-sum-duplicate]", f"{T}[e3-mean-duplicate]", f"{T}[e3-sum-empty-duplicate]"],
        assertion="verdict(row, r) is None: the rows expect ValueError '<op>: duplicate value in axis' with 0 CUDA "
                  "calls; the mutant accepts the repeated axis (sum([0, 0]) on [2] reaches the kernel: 'CUDA error at "
                  "reduce_lastdim'; mean counts the axis twice and reaches the kernel; the empty sum returns [0])",
        gpu_line=None),
    "p04_splice_bound_after_work": dict(
        ruling="E4", file=K,
        old="  preflight_out_bound(\"splice_rows\", out_shape, old_cache.dtype, t);\n",
        new="  // MUTANT p04_splice_bound_after_work: the output's byte bound is left to the constructor (after work)\n",
        where="CPU, and the staged GPU line p2_count_p04_splice_bound_after_work (the ordering needs a first pair "
              "that allocates and launches, which needs a device)",
        kills=[f"{T}[e4-splice-byte-bound]", f"{T}[e4-swap-pair1-byte-bound]", f"{T}[e4-arena-layer1-byte-bound]",
               "scripts/bwdf3_step_p2.py count row e4-swap-pair1-bound-after-work (GPU, staged)"],
        assertion="CPU: verdict(row, r) is None: the rows expect OverflowError 'splice_rows: the output NDArray: byte "
                  "size overflows int64' (and '[pair 1 of 2]' for the pair ops) from the preflight; the mutant's "
                  "refusal is the constructor's bare text with no pair named. GPU: the count row expects the refusal "
                  "with 0 allocations, launches and copies during the call; the mutant runs pair 0 (allocates and "
                  "launches) before pair 1's constructor refuses",
        gpu_line="p2_count_p04_splice_bound_after_work"),
}


def spec():
    if os.path.exists(SPEC):
        raise SystemExit(f"{SPEC} exists")
    doc = {"order": "BWD-F3 P4", "written": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "written_before_any_mutant_tree_or_run": True, "base": "Build F3's tree (/tmp/bwdf3_src_F3)",
           "rule": "each mutant = Build F3 + exactly one edit (the `old` text occurs once); KILLED when every listed "
                   "CPU killing assertion fails on the mutant, and (p04) when its staged GPU line's verdict fails",
           "names_from_order": ["p01_empty_span_refused (E1)", "p02_trailing_singleton_table_refused (E2)",
                                "p03_duplicate_axis_accepted (E3)", "p04_splice_bound_after_work (E4)"],
           "mutants": EDITS}
    json.dump(doc, open(SPEC, "w"), indent=1)
    print("spec", len(EDITS))


def trees():
    os.makedirs(OUT, exist_ok=True)
    for name, e in EDITS.items():
        dst = f"/tmp/bwdf3_src_{name}"
        if os.path.exists(dst):
            raise SystemExit(f"{dst} exists")
        shutil.copytree(F3, dst, symlinks=False)
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
