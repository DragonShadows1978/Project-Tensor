"""BWD-F3 L60: BOUNDARY_F3.json, every new or changed refusal (and acceptance) with its terms, the rows that trigger
each, and Build B's behaviour as measured by scripts/bwdf3_p1.py (one child per row, CUDA_VISIBLE_DEVICES empty, ENG3's
counter). D1 rows are labelled "domain refusal" with Build B's answer.

  python3 -B -P scripts/bwdf3_boundary.py      (reads artifacts/bwdf1/f3/P1_RESULTS.json and INTENDED_CHANGES_F3.json)
Prior art: none beyond the tables.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F3 = os.path.join(ROOT, "artifacts/bwdf1/f3")

TERMS = {
    "E1": {"site": "tensor_cuda/src/kernels.cu rope_tables_check (span term)",
           "term": "the span [pos0, pos0 + rows) must lie in both tables only when the launch reads (output has "
                   "elements); an empty launch passes whatever the tables' lengths and pos0",
           "callers": ["rope_apply (rope_apply_check: reads = x.numel() > 0)",
                       "export_rope_rows / export_row_pair / export_row_pairs (export_rope_rows_check: reads = the "
                       "output [.., len, D] has elements)",
                       "swap_row_pairs_with_rope preflight (rope_apply_check per rope insert)",
                       "arena_row_pair_transaction (through the swap preflight)"],
           "message": "<op>: position past table end (<operands>) [pair i of n]"},
    "E2": {"site": "tensor_cuda/src/kernels.cu rope_table_layout_ok / rope_tables_check (layout term)",
           "term": "each table: rank >= 2, shape[1] == D, every extent after the second is 1 (contiguous [T, D] "
                   "storage); otherwise refused",
           "message": "<op>: tables must be (T, D) (<operands>)"},
    "E3": {"site": "tensor_cuda/src/kernels.cu normalize_reduce_axes (reduce_impl, reduce_arg); tensor_cuda/src/ops.cpp "
                   "ops::mean",
           "term": "each axis in [-nd, nd), normalized; a repeated (normalized) axis refused; before any shape read or "
                   "ReduceSpec write",
           "message": "IndexError '<op>: axis <a> is out of bounds for array of dimension <nd> (axes [...], operand "
                      "[...]); refused before any allocation' / ValueError '<op>: duplicate value in axis (...)'",
           "ops": "<op> is reduce (sum, max, min, prod, softmax, log_softmax, every ops.cpp wrapper over reduce_impl), "
                  "mean (mean, var, std), argmax / argmin"},
    "E4": {"site": "tensor_cuda/src/kernels.cu preflight_out_bound (splice_rows_check; also export_rows_check, "
                   "evict_rows_check, export_rope_rows_check, rope_apply_check, where it cannot fire)",
           "term": "the output shape's byte bound (ndarray_checked_nbytes, the NDArray constructor's own function) "
                   "in the preflight, before any pair's work",
           "message": "OverflowError 'splice_rows: the output NDArray: byte size overflows int64 for shape [...] and "
                      "dtype <dt> (nonzero extents count next to zero) (<operands>) [pair i of n]'"},
    "D1": {"site": "where_nd (x/y dtype, every call); check_meta_rank (14 sites, BWD-F2 R2, unchanged); "
                   "rope_tables_check (table/data dtype, every call, unchanged); axpy_ (destination shape, unchanged)",
           "term": "refused before any work even where Build B returned an empty no-op",
           "label": "domain refusal"},
}


def answer(r):
    if r is None:
        return None
    if "child_crash" in r:
        return f"process died (signal {-r['child_crash']})" if r["child_crash"] < 0 else f"exit {r['child_crash']}"
    if r.get("raised"):
        return f"{r['raised']}: {r['message'][:220]} (CUDA calls {r['calls']})"
    return f"returned {r['result']} (CUDA calls {r['calls']})"


def label(row):
    c, kind = row["class"], row["kind"]
    b = row["B"]["result"] or {}
    b_ok = b.get("raised") is None and "child_crash" not in b
    if c.startswith("D1"):
        return "domain refusal"
    if "control" in c:
        return "unchanged from Build F2 (control)"
    if kind in ("accept", "preflight"):
        return ("restored Build B acceptance" if (b_ok or (b.get("message", "").startswith("CUDA error")))
                else "new acceptance (Build B refused)")
    return "new refusal (Build B: " + ("returned" if b_ok else "did not refuse on the host") + ")" if \
        (b_ok or b.get("message", "").startswith("CUDA error") or "child_crash" in b) else \
        "refusal text changed (Build B refused later or with another text)"


def main():
    p1 = json.load(open(os.path.join(F3, "P1_RESULTS.json")))
    ic = json.load(open(os.path.join(F3, "INTENDED_CHANGES_F3.json")))
    rows = {}
    for r in p1["rows"]:
        rows[r["id"]] = {"class": r["class"], "rule": r["class"].split()[0], "label": label(r), "setup": r["setup"],
                         "call": r["call"], "expect_F3": r["expect"], "Build_B": answer(r["B"]["result"]),
                         "Build_F2": answer(r["F2"]["result"]), "Build_F3": answer(r["F3"]["result"]),
                         "F3_holds": r["F3"]["holds"], "F2_holds": r["F2"]["holds"]}
    doc = {"order": "BWD-F3 (orders/BWDF3_REV2_FIXES.md) L60",
           "measured_by": "scripts/bwdf3_p1.py (Build B 70cd1752, Build F2 5ff79dee, Build F3 per P1_RESULTS.json arms)",
           "terms": TERMS, "rows": rows,
           "domain_refusals": {k: v for k, v in rows.items() if v["label"] == "domain refusal"},
           "inherited_rows_changed": ic["existing_test_changes"],
           "notes": ["Build B's E3 rows on the CPU reach the kernel launch ('CUDA error ...') or die (signal 6 / 11: "
                     "the out-of-range axis writes past keep_shape or ReduceSpec on the host); on a GPU they read past "
                     "the operand (scripts/bwdf3_step_p2.py e3, staged).",
                     "E2 accepts cos tables with trailing extents of 1, which Build B refused ('tables must be (T, D)'), "
                     "and still refuses a sin table with a trailing extent other than 1, which Build B accepted on an "
                     "empty launch (e2c-sin-trailing-2; BWD-F2 refused it too).",
                     "E1 accepts empty launches Build B refused (e1-rope_apply-empty-short-cos, e1-rope_apply-zero-D, "
                     "copy_ops export_rope_rows-past-cos): the ruling bounds only the read span."]}
    json.dump(doc, open(os.path.join(F3, "BOUNDARY_F3.json"), "w"), indent=1)
    from collections import Counter
    print("rows", len(rows), Counter(v["label"] for v in rows.values()))


if __name__ == "__main__":
    main()
