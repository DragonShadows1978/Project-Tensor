"""BWD-F2 L60: BOUNDARY_F2.json. Every new or changed refusal (and every R1 restored acceptance) with its terms, the
K1 / R12 rows that trigger it, and Build B's behaviour for the same operands, measured: each row runs on Build B's
frozen package (tensor_cuda/build-bwdf1/B/pkg, sha-checked) in its own child with CUDA_VISIBLE_DEVICES empty and ENG3's
counter (scripts: tensor_cuda/tests/test_bwdf2_fixes.py run_rows), next to Build E's and Build F2's results from
artifacts/bwdf1/f2/K1_RESULTS.json.

  python3 -B -P scripts/bwdf2_boundary.py
Prior art: BWD-F1's BOUNDARY.json layout (this repo). Ours: the terms table.
"""
import hashlib
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tensor_cuda", "tests"))
os.environ.setdefault("BWDF2_PKG_PARENT", os.path.join(ROOT, "tensor_cuda/build-bwdf1/F2/pkg"))
import test_bwdf2_fixes as T  # noqa: E402

K = "tensor_cuda/src/kernels.cu"
TERMS = [
    {"ruling": "R1", "finding": "F03", "where": f"{K} check_copy_operands",
     "term": "cat/stack: an operand with elements must have operand 0's dtype and device (the output's); empty "
             "operands are not dtype/device-checked (rank and extents still are)",
     "kind": "narrowed refusal (BWD-F1 refused every operand)",
     "rows": ["f03-cat-named-case", "f03-cat-empty-later-device", "f03-stack-empty-mixed-dtype",
              "f03-cat-empty-mixed-device-dtype", "f03-cat-copied-dtype-still-refused",
              "f03-cat-copied-device-still-refused"]},
    {"ruling": "R1", "finding": "F03", "where": f"{K} write_rows",
     "term": "write_rows: the device term only when src has elements", "kind": "narrowed refusal",
     "rows": ["f03-write_rows-empty-device"]},
    {"ruling": "R1/R3", "finding": "F03/F01/F05", "where": f"{K} rope_tables_check",
     "term": "RoPE tables on the data's device and a float launch dtype, only when the kernel reads",
     "kind": "narrowed (device) and new (dtype, before allocation)",
     "rows": ["f03-export_rope_rows-empty-host-cos", "f05-rope_apply-uint8-launch-dtype"]},
    {"ruling": "R3", "finding": "F01", "where": f"{K} rope_tables_check",
     "term": "both tables rank 2 with D columns; 0 <= pos0, pos0 <= T and rows <= T - pos0 for cos AND sin "
             "(subtraction-based); used by rope_apply, export_rope_rows, the export/swap preflights and the arena",
     "kind": "new refusal (sin span, negative pos0, sin rank)",
     "rows": ["f01-rope_apply-short-sin", "f01-swap-short-sin", "f01-arena-short-sin", "f01-rope_apply-negative-pos0",
              "f01-rope_apply-tables-cover"]},
    {"ruling": "R3", "finding": "F05", "where": f"{K} export_row_pair",
     "term": "both sides' checks before either side's work", "kind": "reordered refusal",
     "rows": ["f05-export_row_pair-rope-side"]},
    {"ruling": "R2", "finding": "F02", "where": f"{K} check_meta_rank (ew_binary, compare, where_nd, reduce_impl, "
                                                "permute, broadcast_to, check_copy_operands, slice_nd, pad_into, "
                                                "gather_nd, scatter_add_nd, flip_nd, axpy_; ops.cpp masked_fill)",
     "term": "rank > TC_MAX_DIMS (8) refused before any allocation or metadata write, empty operands included; stack "
             "checks its post-unsqueeze rank", "kind": "new refusal",
     "rows": ["f02-stack-rank8-to-9", "f02-cat-rank9-empty", "f02-slice-rank9", "f02-binary-rank9",
              "f02-binary-rank9-empty", "f02-compare-rank9", "f02-where-rank9", "f02-masked_fill-rank9",
              "f02-reduce-rank9", "f02-permute-rank9", "f02-expand-rank9", "f02-rank8-control"]},
    {"ruling": "R5", "finding": "F07", "where": f"{K} where_nd",
     "term": "x and y dtypes must match when the output has elements (before the allocation); the condition is loaded "
             "in its own dtype (no refusal)", "kind": "new refusal (data dtypes) + dispatch change (condition)",
     "rows": ["f07-where-data-dtype", "f07-where-bool-cond", "f07-masked_fill-bool-mask", "f07-where-empty-mixed-data"]},
    {"ruling": "R6", "finding": "F12", "where": f"{K} axpy_",
     "term": "the broadcast of param and other must equal param's shape up to leading size-one axes",
     "kind": "new refusal", "rows": ["f12-axpy-named-case", "f12-axpy-trailing-broadcast", "f12-axpy-leading-one-other",
                                     "f12-axpy-leading-one-param"]},
    {"ruling": "R7", "finding": "F08", "where": f"{K} cat_nd / splice_rows_check / arena_row_pair_transaction / "
                                                "product_dims / reduce_impl",
     "term": "checked sums (cat extent, splice output extent, arena tail_start) and emptiness-first checked products "
             "(row-cache leading / inner counts, reduced count) before allocation or any pair's work",
     "kind": "new refusal (OverflowError) where the sum overflows; same results otherwise",
     "rows": ["f08-cat-sum-overflow", "f08-splice-sum-overflow", "f08-arena-span-overflow",
              "f08-write_rows-empty-leading-product"]},
    {"ruling": "R8", "finding": "F09", "where": f"{K} slice_nd / pad_into; tensor_cuda/src/bindings.cpp getitem",
     "term": "slice / pad_into: the float launch dtype before allocating when a copy will run; __getitem__: the axis "
             "exists before its extent is read (IndexError)", "kind": "reordered refusal / new refusal",
     "rows": ["f09-slice-uint8", "f09-getitem-rank0"]},
    {"ruling": "R10", "finding": "F11", "where": f"{K} normalize_dim / with_index / arena",
     "term": "dim refusals keep '<op>: dim out of range' and add dim, rank, operand position and shape; list scans add "
             "'[pair i of n]'; the arena names the layer", "kind": "changed messages (same refusals)",
     "rows": ["f11-export_rows-dim-context", "f11-evict_row_pairs-pair-index", "f11-swap-pair-index",
              "f11-arena-layer-dim"]},
    {"ruling": "R10", "finding": "F13", "where": "tensor_cuda/tensor_cuda/functional.py einsum",
     "term": "the per-label extent check covers terms with fewer letters than the operand's rank (the extents the "
             "factor construction reads)", "kind": "earlier refusal", "rows": ["f13-einsum-shorter-term"]},
]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    parent = os.path.join(ROOT, "tensor_cuda/build-bwdf1/B/pkg")
    got = sha(os.path.join(parent, "tensor_cuda", "_tensor_cuda.cpython-312-x86_64-linux-gnu.so"))
    want = json.load(open(os.path.join(ROOT, "artifacts/bwdf1/build/BUILD_tc_B.json")))["module_sha256"]
    assert got == want, (got, want)
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", "/tmp/bwdf2_tmp")) as d:
        b = T.run_rows(parent, d)
    k1 = {r["id"]: r for r in json.load(open(os.path.join(ROOT, "artifacts/bwdf1/f2/K1_RESULTS.json")))["rows"]}
    rows = {}
    for row in T.ROWS:
        rid = row[0]
        rows[rid] = {"setup": row[3], "call": row[4], "kind": row[2], "expect_F2": row[5],
                     "build_B": b["results"].get(rid), "build_E": k1[rid]["E"]["result"],
                     "build_F2": k1[rid]["F2"]["result"], "F2_holds": k1[rid]["F2"]["holds"]}
    doc = {"order": "BWD-F2 L60", "build_B_module_sha256": got, "cuda_visible_devices": os.environ.get(
        "CUDA_VISIBLE_DEVICES"), "control_calls_B": b.get("control_calls"), "terms": TERMS, "rows": rows,
        "note": "Build B results come from one child per row (a crash is recorded as child_crash). On the CPU a "
                "'CUDA error at <op>' message means Build B passed its host checks and reached a device step."}
    json.dump(doc, open(os.path.join(ROOT, "artifacts/bwdf1/f2/BOUNDARY_F2.json"), "w"), indent=1)
    for rid, r in rows.items():
        bb = r["build_B"] or {}
        print(rid[:42].ljust(42), "B:", (bb.get("raised") or "returned"), (bb.get("message") or str(bb.get("result")) or
                                                                           str(bb.get("child_crash")))[:90])


if __name__ == "__main__":
    main()
