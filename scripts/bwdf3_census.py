"""BWD-F3 operand census (orders/BWDF3_REV2_FIXES.md "Operand census", L82, L89, L90) and L55 witnesses.

  python3 -B -P scripts/bwdf3_census.py   -> artifacts/bwdf1/f3/CENSUS_F3.json and artifacts/bwdf1/f3/L55_F3.json

The census starts from L82's grep and then finds each class by what the code does, not by name (L89): RoPE paths by
their table reads (`ld<T>(cs, t)` / `ld<T>(sn, t)` with t = (pos0 + l) * D + d), reductions by their ReduceSpec writes
and by every entry that reads an axis's extent, pair preflights by the "every pair's checks before any pair's work"
loops. Line numbers are found in the current sources at run time. Prior art: none beyond grep.
"""
import json
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K = "tensor_cuda/src/kernels.cu"
O = "tensor_cuda/src/ops.cpp"


def lines(path):
    return open(os.path.join(ROOT, path)).read().split("\n")


def find(path, needle, nth=1):
    hits = [i + 1 for i, l in enumerate(lines(path)) if needle in l]
    if len(hits) < nth:
        raise SystemExit(f"{path}: {needle!r} found {len(hits)} times")
    return f"{path}:{hits[nth - 1]}"


def grep_all(pattern, paths):
    out = []
    for p in paths:
        for i, l in enumerate(lines(p)):
            if re.search(pattern, l):
                out.append(f"{p}:{i + 1}: {l.strip()[:140]}")
    return out


def main():
    l82 = subprocess.run(["grep", "-n", r"rope_tables_check\|rope_apply_check\|reduce_impl\|ReduceSpec\|"
                          r"splice_rows_check\|product_dims", K, O, "tensor_cuda/src/bindings.cpp"],
                         cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
    census = {
        "order": "BWD-F3 operand census",
        "L82_grep": l82,
        "rope_paths_by_table_read": {
            "table_reads (the only kernels that read cos/sin rows)": grep_all(r"ld<T>\((cs|sn), t\)", [K]),
            "launchers and their validator": {
                "rope_apply": {"launch": find(K, "rope_kernel<T><<<"), "check": find(K, "  rope_apply_check(x, cs, sn, pos0);  // BWD-F2 R3"),
                               "reads": "x.numel() > 0", "read_span": "[pos0, pos0 + L), L = x.shape[-2]"},
                "export_rope_rows": {"launch": find(K, "export_rope_rows_kernel<T><<<"),
                                     "check": find(K, '  rope_tables_check("export_rope_rows"'),
                                     "reads": "output [.., len, D] has elements", "read_span": "[pos0, pos0 + len)"},
                "export_row_pair": {"check": find(K, "  export_rope_rows_check(rope_cache, cs, sn, rope_dim, rope_start, len, pos0);"),
                                    "note": "both sides checked before the raw export"},
                "export_row_pairs": {"check": find(K, "      export_rope_rows_check(rope_caches[i], cs, sn, rope_dim, rope_starts[i], len, pos0);"),
                                     "note": "every pair checked before any pair's work"},
                "swap_row_pairs_with_rope": {"check": find(K, "      rope_apply_check(x, cs, sn, pos0);"),
                                             "note": "rope_apply's own check on each rope insert, every pair first"},
                "arena_row_pair_transaction": {"check": "delegates to swap_row_pairs_with_rope (or evict_row_pairs) "
                                               + find(K, "  auto out = swap_row_pairs_with_rope(")}},
            "validator": {"rope_tables_check": find(K, "static void rope_tables_check("),
                          "dtype (D1, every call)": find(K, "  if (cs.dtype != xdtype || sn.dtype != xdtype)"),
                          "layout (E2)": find(K, "  if (!rope_table_layout_ok(cs, D) || !rope_table_layout_ok(sn, D))"),
                          "span (E1, reads only)": find(K, "  if (reads && (pos0 < 0 || rows < 0"),
                          "device / launch dtype (reads only, BWD-F2 R1)": find(K, "  if (reads && (!same_device(cs.device")},
            "consumers (report-only)": ["tensor_cuda/tensor_cuda/__init__.py re-exports rope_apply and the row ops",
                                        "tensor_cuda/tensor_cuda/functional.py apply_rotary composes ops (no fused "
                                        "table read)"],
        },
        "reductions": {
            "spec_writers (ReduceSpec)": {"reduce_impl": find(K, "NDArray reduce_impl("),
                                          "axis rule": find(K, '  axes = normalize_reduce_axes("reduce", axes, a.shape);'),
                                          "first spec write after it": find(K, "  ReduceSpec s{};")},
            "entries": {
                "reduce_sum / reduce_max / reduce_min / reduce_prod (kernels.cu)": "reduce_impl",
                "ops::sum, ops::max, ops::min, ops::prod (ops.cpp)": "reduce_impl first; keepdim_shape after it reads only "
                                                                    "validated axes",
                "ops::softmax, ops::log_softmax (ops.cpp)": "reduce_max({axis}) first",
                "ops::mean (ops.cpp)": find(O, '  const std::vector<int> nax = tc::normalize_reduce_axes("mean", axes, a.shape());'),
                "ops::var, ops::std": "ops::mean first",
                "reduce_arg (argmax / argmin)": find(K, '  axis = normalize_reduce_axes(is_max ? "argmax" : "argmin", {axis}, a.shape)[0];'),
                "reduce_to (autograd)": "builds distinct in-range axes itself; reduce_impl re-checks them",
                "cross_entropy, mse_loss": "fixed axes ({-1}, {}) through sum / mean",
            },
            "helper": find(K, "std::vector<int> normalize_reduce_axes("),
            "not reductions (listed, unchanged)": ["cumsum_nd (a scan: axis_split reads shape[axis] unchecked; residual)",
                                                   "flip_nd (FlipSpec axes: the confirm's successor)",
                                                   "gather_nd / scatter_add_nd (the confirm's successor)"],
            "consumers (report-only)": "tensor_cuda/tensor_cuda/nn.py, quant.py, optim.py, functional.py call "
                                       "sum / mean / var with fixed axes; tc_signal does not include the core headers",
        },
        "pair_preflight_byte_bounds": {
            "helper": find(K, "static void preflight_out_bound("),
            "shared bound": find(K, "size_t ndarray_checked_nbytes("),
            "sites": {"export_rows_check": find(K, '  preflight_out_bound("export_rows"'),
                      "splice_rows_check": find(K, '  preflight_out_bound("splice_rows"'),
                      "evict_rows_check": find(K, '  preflight_out_bound("evict_rows"'),
                      "export_rope_rows_check": find(K, '  preflight_out_bound("export_rope_rows"'),
                      "rope_apply_check": find(K, '  preflight_out_bound("rope_apply"')},
            "pair ops and the preflights they run": {
                "export_row_pair": "export_rows_check + export_rope_rows_check",
                "export_row_pairs": "the same, for every pair first",
                "swap_row_pairs_with_rope": "splice_rows_check (raw) + rope_apply_check + splice_rows_check (rope), "
                                            "every pair first",
                "evict_row_pairs": "evict_rows_check x 2, every pair first",
                "arena_row_pair_transaction": "evict_row_pairs or swap_row_pairs_with_rope"},
            "can_fire": "only splice (its extent grows); the others' outputs are no larger than an operand",
        },
        "D1_rows": {
            "rank above TC_MAX_DIMS": {"sites": grep_all(r"check_meta_rank\(", [K, O]),
                                       "note": "BWD-F2 R2, unchanged; every call"},
            "where x / y dtype": find(K, "  if (x.dtype != y.dtype)"),
            "RoPE table dtype": find(K, "  if (cs.dtype != xdtype || sn.dtype != xdtype)"),
            "axpy_ destination shape": find(K, "    if (strip(upd_shape) != strip(param.shape))"),
            "copy-op members (unchanged, BWD-F2 R1)": "dtype / device checked only for members whose bytes are copied",
        },
        "roles": {"product (engine)": 4, "test": 3, "consumer (report-only)": 0, "harness": 15,
                  "detail": {"product": [K, O, "tensor_cuda/include/tc/core.h", "(no change: bindings.cpp, functional.py)"],
                             "test": ["tensor_cuda/tests/test_bwdf3_fixes.py (new)",
                                      "tensor_cuda/tests/test_bwdf2_fixes.py (E6 rows)",
                                      "tensor_cuda/tests/test_bwdf1_copy_ops.py (E6 rows)"],
                             "harness": "scripts/bwdf3_*.py",
                             "product_count_note": "4 = kernels.cu (E1, E2, E3, E4, D1 where), ops.cpp (E3 mean), "
                                                   "core.h (two declarations), and the shared constructor bound"}},
    }
    json.dump(census, open(os.path.join(ROOT, "artifacts/bwdf1/f3/CENSUS_F3.json"), "w"), indent=1)
    p1 = {r["id"]: r for r in json.load(open(os.path.join(ROOT, "artifacts/bwdf1/f3/P1_RESULTS.json")))["rows"]}

    def w(rid, f2_path, f3_pos):
        r = p1[rid]
        short = lambda x: (x or {}).get("message", (x or {}).get("result")) if "child_crash" not in (x or {}) else \
            f"process died ({(x or {})['child_crash']})"
        return {"row": rid, "inputs": r["setup"] + "; " + r["call"], "build_F2_path": f2_path,
                "build_F2_measured": str(short(r["F2"]["result"]))[:240], "build_F3_check_position": f3_pos,
                "build_F3_measured": str(short(r["F3"]["result"]))[:240]}
    l55 = {
        "E1": w("e1-rope_apply-empty-short-sin", "rope_tables_check checked the span on every call (e4ad8381a "
                "kernels.cu:8708) and refused sin [1, 2] < L 2 although x [0, 2, 2] reads nothing",
                find(K, "  if (reads && (pos0 < 0 || rows < 0")),
        "E2": w("e2-rope_apply-sin-T-D-1", "rope_tables_check required rank 2 for both tables (e4ad8381a "
                "kernels.cu:8706)", find(K, "  if (!rope_table_layout_ok(cs, D) || !rope_table_layout_ok(sn, D))")),
        "E3": w("e3-sum-duplicate", "reduce_impl marked axis 0 once but multiplied its extent twice (red_size 4; "
                "e4ad8381a kernels.cu:1410-1416), then the tail kernel read 4 elements (kernels.cu:1336)",
                find(K, '  axes = normalize_reduce_axes("reduce", axes, a.shape);')),
        "E3-mean": w("e3-mean-out-of-range", "ops::mean read a.shape()[5] of a rank-1 tensor before sum (e4ad8381a "
                     "ops.cpp mean)", find(O, '  const std::vector<int> nax = tc::normalize_reduce_axes("mean"')),
        "E4": w("e4-swap-pair1-byte-bound", "splice_rows_check proved the extent fits int64 (e4ad8381a "
                "kernels.cu:8603); pair 0 ran, then pair 1's NDArray constructor refused [0, 2^62, 2] uint8 "
                "(kernels.cu:8615)", find(K, '  preflight_out_bound("splice_rows"')),
        "D1-where": w("d1-where-empty-mixed-dtype", "where_nd checked x/y dtype only when the output had elements "
                      "(e4ad8381a kernels.cu:1615) and returned [0]", find(K, "  if (x.dtype != y.dtype)")),
        "D1-rank": w("d1-rank9-binary-empty", "check_meta_rank refused before any work (BWD-F2 R2, unchanged)",
                     find(K, 'check_meta_rank("ew_binary"')),
        "D1-axpy": w("d1-axpy-empty-destination-shape", "axpy_ refused the destination shape (BWD-F2 R6, unchanged)",
                     find(K, "    if (strip(upd_shape) != strip(param.shape))")),
        "D1-rope-dtype": w("d1-rope-table-dtype-empty", "rope_tables_check's dtype term on every call (unchanged)",
                           find(K, "  if (cs.dtype != xdtype || sn.dtype != xdtype)")),
    }
    json.dump(l55, open(os.path.join(ROOT, "artifacts/bwdf1/f3/L55_F3.json"), "w"), indent=1)
    print("census written; L55 witnesses", len(l55))


if __name__ == "__main__":
    main()
