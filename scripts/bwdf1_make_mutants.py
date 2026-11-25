"""BWD-F1 G5: make each mutant's source tree as Build E's tree plus exactly one edit (artifacts/bwdf1/MUTANTS.json).

  python3 -B -P scripts/bwdf1_make_mutants.py [--refresh]

- --refresh (Amendment 1): the trees exist from the first pass; copy Build E's current files over them (overwrite, no
  delete; E removed no file) and apply the edit again to the fresh copy of the edited file.

- Copies /tmp/bwdf1_src_E (tensor_cuda/, tc_signal/) to /tmp/bwdf1_src_<mutant> (refuses if it exists), applies the
  edit (each `old` must occur exactly once), writes the unified diff to artifacts/bwdf1/mutants/<mutant>.diff.
- m04 is Python only: its tree holds Build E's tc_signal package with its built module and the mutated __init__.py.

Prior art: mutation testing (DeMillo, Lipton and Sayward, 1978); ENG4's eng4_make_arms.py (this repo). Ours: the edits.
"""
import difflib
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E = "/tmp/bwdf1_src_E"
OUT = os.path.join(ROOT, "artifacts/bwdf1/mutants")

EDITS = {
    "m01_max_rule": ("tensor_cuda/src/kernels.cu",
                     """    if (ad == bd) out[i] = ad;
    else if (ad == 1) out[i] = bd;
    else if (bd == 1) out[i] = ad;
    else
      throw""",
                     """    if (!(ad != bd && ad != 1 && bd != 1)) out[i] = ad > bd ? ad : bd;  // MUTANT m01_max_rule
    else
      throw"""),
    "m02_no_empty_operand_guard": ("tensor_cuda/src/kernels.cu",
                                   """                               std::initializer_list<const Shape*> shapes) {
  const auto empty""",
                                   """                               std::initializer_list<const Shape*> shapes) {
  return;  // MUTANT m02_no_empty_operand_guard
  const auto empty"""),
    "m03_reduce_gt1": ("tensor_cuda/src/kernels.cu",
                       "    if (tsz == 1 && a.shape[d] != 1) axes.push_back(d);",
                       "    if (tsz == 1 && a.shape[d] > 1) axes.push_back(d);  // MUTANT m03_reduce_gt1"),
    "m05_cat_no_extent_check": ("tensor_cuda/src/kernels.cu",
                                """    for (int k = 0; k < nd; ++k)
      if (k != d && a.shape[k] != a0.shape[k])
        throw std::runtime_error(std::string(op) + ": operand " + std::to_string(i) + " has extent " +
                                 std::to_string(a.shape[k]) + " at dim " + std::to_string(k) + " but operand 0 has " +
                                 std::to_string(a0.shape[k]) +
                                 (all_extents ? "; every shape must be equal" : "; extents must match outside dim " +
                                                                                    std::to_string(d)) + pair);
""",
                                """    // MUTANT m05_cat_no_extent_check: the extent loop is deleted
"""),
    "m06_cat_no_dtype_check": ("tensor_cuda/src/kernels.cu",
                               """    if (a.dtype != a0.dtype)
      throw std::runtime_error(std::string(op) + ": operand " + std::to_string(i) + " has dtype " +
                               dtype_name(a.dtype) + " but operand 0 has " + dtype_name(a0.dtype) + pair);
""",
                               """    // MUTANT m06_cat_no_dtype_check: the dtype check is deleted
"""),
    "m04_h1_dropped": ("tc_signal/tc_signal/__init__.py",
                       """        destinations = out if isinstance(out, (tuple, list)) else (out,)
        if _grad_context(destinations) is not None:
            raise RuntimeError(f"tc_signal: {name}: out cannot require grad")
        return None""",
                       """        return None  # MUTANT m04_h1_dropped"""),
}


def main():
    os.makedirs(OUT, exist_ok=True)
    for m, (rel, old, new) in EDITS.items():
        dst = f"/tmp/bwdf1_src_{m}"
        if os.path.exists(dst) and "--refresh" not in sys.argv:
            raise SystemExit(f"{dst} exists")
        shutil.copytree(E, dst, ignore=shutil.ignore_patterns("__pycache__"), dirs_exist_ok=True)
        p = os.path.join(dst, rel)
        s = open(p).read()
        assert s.count(old) == 1, (m, rel)
        t = s.replace(old, new)
        open(p, "w").write(t)
        diff = "".join(difflib.unified_diff(s.splitlines(True), t.splitlines(True), f"E/{rel}", f"{m}/{rel}"))
        open(os.path.join(OUT, f"{m}.diff"), "w").write(diff)
        print(m, rel, "lines changed:", sum(1 for l in diff.splitlines() if l[:1] in "+-" and l[:3] not in ("+++", "---")))


if __name__ == "__main__":
    sys.exit(main())
