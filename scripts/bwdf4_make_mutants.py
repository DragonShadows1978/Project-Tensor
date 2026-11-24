"""BWD-F4 G3: the two mutants of orders/BWDF4_REV3_HARDENING.md. Each mutant's source tree is Build F4's tree
(/tmp/bwdf4_src_F4) plus exactly one edit; MUTANTS_F4.json (written by `spec` before any mutant tree exists or runs)
names each mutant's edit and the G1 rows that must fail on it.

  python3 -B -P scripts/bwdf4_make_mutants.py spec      # artifacts/bwdf1/f4/MUTANTS_F4.json (refuses if it exists)
  python3 -B -P scripts/bwdf4_make_mutants.py trees     # /tmp/bwdf4_src_<mutant> + artifacts/bwdf1/f4/mutants/*.diff

Prior art: mutation testing (DeMillo, Lipton and Sayward, 1978); scripts/bwdf3_make_mutants.py (this repo), copied.
Ours: the edits and their killing rows (the order names the two mutants and their rules).
"""
import datetime
import difflib
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F4 = "/tmp/bwdf4_src_F4"
OUT = os.path.join(ROOT, "artifacts/bwdf1/f4/mutants")
SPEC = os.path.join(ROOT, "artifacts/bwdf1/f4/MUTANTS_F4.json")
T = "tensor_cuda/tests/test_bwdf4_hardening.py::test_cpu_g1_row"

EDITS = {
    "h01_mean_unchecked_product": dict(
        ruling="H1", file="tensor_cuda/src/ops.cpp",
        old="        if (__builtin_mul_overflow(cnt, a.shape()[d], &cnt)) {\n",
        new="        cnt *= a.shape()[d]; if (false /* MUTANT h01_mean_unchecked_product */) {\n",
        kills=[f"{T}[h1-overflow-2pow32-pair]", f"{T}[h1-overflow-2pow21-triple]", f"{T}[h1-overflow-negative-axes]",
               f"{T}[h1-overflow-edge-2pow62-times-2]"],
        assertion="verdict(row, r) is None: each row expects OverflowError whose message starts at 'mean: the reduced "
                  "element count of [...] overflows int64 (axes [...]); refused before any allocation' with 0 CUDA calls; "
                  "the mutant multiplies the selected extents unchecked (signed overflow, as Build F3), so the refusal "
                  "comes from the reducer with 'reduce: the reduced element count of ...' and the row fails on the text"),
    "h02_table_rank_unbounded": dict(
        ruling="H2", file="tensor_cuda/src/kernels.cu",
        old="  if (t.ndim() < 2 || t.ndim() > TC_MAX_DIMS || t.shape[1] != D) return false;\n",
        new="  if (t.ndim() < 2 || t.shape[1] != D) return false;  // MUTANT h02_table_rank_unbounded\n",
        kills=[f"{T}[h2-rope_apply-sin-rank9-empty]", f"{T}[h2-rope_apply-cos-rank9-empty]",
               f"{T}[h2-rope_apply-sin-rank9-reading]", f"{T}[h2-export_rope_rows-sin-rank9]", f"{T}[h2-swap-sin-rank9]"],
        assertion="verdict(row, r) is None: each row expects RuntimeError 'tables must be (T, D)' with 0 CUDA calls for a "
                  "rank-9 table; the mutant accepts the trailing-singleton table at any rank (empty launches return "
                  "their shapes; the reading launch reaches 'CUDA error at rope_apply')"),
}


def spec():
    if os.path.exists(SPEC):
        raise SystemExit(f"{SPEC} exists")
    doc = {"order": "BWD-F4 G3", "written": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "written_before_any_mutant_tree_or_run": True, "base": "Build F4's tree (/tmp/bwdf4_src_F4)",
           "rule": "each mutant = Build F4 + exactly one edit (the `old` text occurs once); KILLED when every listed "
                   "CPU killing row fails on the mutant (and passes on Build F4)",
           "names_from_order": ["h01_mean_unchecked_product (H1)", "h02_table_rank_unbounded (H2)"], "mutants": EDITS}
    json.dump(doc, open(SPEC, "w"), indent=1)
    print("spec", len(EDITS))


def trees():
    os.makedirs(OUT, exist_ok=True)
    for name, e in EDITS.items():
        dst = f"/tmp/bwdf4_src_{name}"
        if os.path.exists(dst):
            raise SystemExit(f"{dst} exists")
        shutil.copytree(F4, dst, symlinks=False)
        p = os.path.join(dst, e["file"])
        src = open(p).read()
        if src.count(e["old"]) != 1:
            raise SystemExit(f"{name}: old text occurs {src.count(e['old'])} times")
        mut = src.replace(e["old"], e["new"])
        open(p, "w").write(mut)
        diff = "".join(difflib.unified_diff(src.splitlines(True), mut.splitlines(True), "a/" + e["file"], "b/" + e["file"]))
        open(os.path.join(OUT, name + ".diff"), "w").write(diff)
        print("tree", dst, "diff lines", diff.count("\n"))


if __name__ == "__main__":
    {"spec": spec, "trees": trees}[sys.argv[1]]()
