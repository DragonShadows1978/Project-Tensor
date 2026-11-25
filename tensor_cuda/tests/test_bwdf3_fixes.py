"""BWD-F3 (orders/BWDF3_REV2_FIXES.md) P1: one CPU row per E1-E4 case and per D1 domain refusal, run with
CUDA_VISIBLE_DEVICES empty through the public API, with ENG3's LD_PRELOAD CUDA-call counter. The harness (one child
process per row, per-function counts, the row kinds refuse / accept / preflight and their verdicts) is BWD-F2's
(tests/test_bwdf2_fixes.py run_rows and verdict), imported unchanged.

Row classes (the second field):
- "E1".."E4": the ruled fix. The row fails on Build F2 (5ff79dee) and holds on Build F3 (scripts/bwdf3_p1.py checks
  both). E1 (BWDF1R2-F01): an empty launch reads no table row and passes; E2 (F02): a contiguous (T, D) table, rank
  2 or trailing extents of 1, is accepted; E3 (F06): every reduction entry refuses an out-of-range or repeated axis
  before indexing a shape or writing a spec slot (NumPy's rule and texts); E4 (F07): a splice preflight refuses an
  output whose bytes exceed int64 before any pair's work, naming the pair.
- "<E> control": the rule's other side, unchanged from Build F2 (a reading span still refused, a valid axis still
  reaches the device step, an output that fits still accepted).
- "D1 (F0x)": a domain refusal ruled by D1 (rank above TC_MAX_DIMS, where's x/y dtype, a RoPE table's dtype, axpy_'s
  destination shape). It holds on Build F3; Build F2's and Build B's answers are recorded (BOUNDARY_F3.json).
Every refusal row asserts 0 CUDA calls. The last field is Build F2's expected behaviour (from the confirm's readings,
re-measured by scripts/bwdf3_p1.py).

Operands: H(shape, dtype) a host tensor of ones (a float ones tensor is an all-true condition), C(shape, dtype) a
CUDA tensor with zero elements (built without a device). A host operand with elements lets a row reach the kernel
launch on the CPU ("CUDA error at <op>: no CUDA-capable device"), which is the CPU form of a passing control.

Environment: BWDF3_PKG_PARENT selects the tensor_cuda package under test (Build F2, Build F3, Build B, mutants).
Prior art: NumPy's normalize_axis_tuple ("duplicate value in axis", AxisError "axis 8 is out of bounds for array of
dimension 1"); the RoPE formulation (Su et al., 2021, "RoFormer"); ENG3's LD_PRELOAD counter (ltrace-style
interposition); DeMillo, Lipton and Sayward (1978), mutation testing (rows named in artifacts/bwdf1/f3/MUTANTS_F3.json
kill p01-p04). Ours: the rows.
"""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_bwdf2_fixes as T2  # noqa: E402  (BWD-F2's K1 harness)

PKG_PARENT = os.environ.get("BWDF3_PKG_PARENT", os.path.dirname(HERE))
NG = "no_grad"
P61 = 2 ** 61

# (id, class, kind, setup, call, expectation, flags, Build F2's expected behaviour)
ROWS = [
    # ---- E1 (F01): an empty launch reads no table row and passes, whatever the tables' lengths and pos0
    ("e1-rope_apply-empty-short-sin", "E1", "accept", "x=C((0,2,2)); cs=H((2,2)); sn=H((1,2))",
     "tc.rope_apply(x, cs, sn, 0)", [0, 2, 2], {}, "refuses 'position past table end' (the review's case)"),
    ("e1-rope_apply-empty-short-cos", "E1", "accept", "x=C((0,2,2)); cs=H((1,2)); sn=H((2,2))",
     "tc.rope_apply(x, cs, sn, 0)", [0, 2, 2], {}, "refuses 'position past table end'"),
    ("e1-rope_apply-empty-negative-pos0", "E1", "accept", "x=C((0,2,2)); cs=H((4,2)); sn=H((4,2))",
     "tc.rope_apply(x, cs, sn, -1)", [0, 2, 2], {}, "refuses 'position past table end'"),
    ("e1-rope_apply-zero-D", "E1", "accept", "x=C((2,3,0)); cs=H((1,0)); sn=H((1,0))",
     "tc.rope_apply(x, cs, sn, 0)", [2, 3, 0], {}, "refuses 'position past table end' (L 3 > 1 row)"),
    ("e1-export_rope_rows-empty-short-sin", "E1", "accept", "c=C((0,4,2)); cs=H((2,2)); sn=H((1,2))",
     "tc.export_rope_rows(c, cs, sn, 1, 0, 2, 0)", [0, 2, 2], {NG: 1}, "refuses 'position past table end'"),
    ("e1-export_row_pair-empty-short-sin", "E1", "accept", "r=C((0,4,2)); p=C((0,4,2)); cs=H((2,2)); sn=H((1,2))",
     "tc.export_row_pair(r, p, cs, sn, 1, 1, 0, 0, 2, 0)", "tuple", {NG: 1}, "refuses 'position past table end'"),
    ("e1-export_row_pairs-empty-short-sin", "E1", "accept",
     "r0=C((0,4,2)); p0=C((0,4,2)); cs=H((2,2)); sn=H((1,2))",
     "tc.export_row_pairs([r0], [p0], cs, sn, 1, 1, [0], [0], 2, 0)", "tuple", {NG: 1},
     "refuses 'position past table end' [pair 0 of 1]"),
    ("e1-swap-empty-short-sin", "E1", "accept",
     "r0=C((0,2,2)); p0=C((0,2,2)); i0=C((0,2,2)); j0=C((0,2,2)); cs=H((2,2)); sn=H((1,2))",
     "tc.swap_row_pairs_with_rope([r0], [p0], [i0], [j0], cs, sn, 1, 1, 0, 0, 0)", "tuple", {NG: 1},
     "refuses 'position past table end' [pair 0 of 1]"),
    ("e1-arena-empty-short-sin", "E1", "accept",
     "r0=C((0,2,2)); p0=C((0,2,2)); i0=C((0,2,2)); j0=C((0,2,2)); cs=H((2,2)); sn=H((1,2))",
     "tc.arena_row_pair_transaction([r0], [p0], [i0], [j0], cs, sn, 1, 1, 0, 0, 4)", "tuple", {NG: 1},
     "refuses 'position past table end' (through the swap preflight)"),
    ("e1c-rope_apply-reading-short-sin", "E1 control", "refuse", "x=H((1,2,2)); cs=H((2,2)); sn=H((1,2))",
     "tc.rope_apply(x, cs, sn, 0)", ["rope_apply: position past table end", "sin [1, 2]"], {}, "same"),
    ("e1c-rope_apply-reading-negative-pos0", "E1 control", "refuse", "x=H((1,2,2)); cs=H((4,2)); sn=H((4,2))",
     "tc.rope_apply(x, cs, sn, -1)", ["rope_apply: position past table end", "pos0 -1"], {}, "same"),
    ("e1c-rope_apply-reading-covering", "E1 control", "preflight", "x=H((1,2,2)); cs=H((2,2)); sn=H((2,2))",
     "tc.rope_apply(x, cs, sn, 0)", ["CUDA error at rope_apply"], {}, "same"),
    # ---- E2 (F02): a contiguous (T, D) table: rank 2, or extra (trailing) extents all 1
    ("e2-rope_apply-sin-T-D-1", "E2", "preflight", "x=H((1,2,2)); cs=H((2,2)); sn=H((2,2,1))",
     "tc.rope_apply(x, cs, sn, 0)", ["CUDA error at rope_apply"], {},
     "refuses 'tables must be (T, D)' (the review's case; Build B computes it)"),
    ("e2-rope_apply-cos-T-D-1-1-empty", "E2", "accept", "x=C((0,2,2)); cs=H((2,2,1,1)); sn=H((2,2))",
     "tc.rope_apply(x, cs, sn, 0)", [0, 2, 2], {}, "refuses 'tables must be (T, D)'"),
    ("e2-export_rope_rows-sin-T-D-1-empty", "E2", "accept", "c=C((0,4,2)); cs=H((8,2)); sn=H((8,2,1))",
     "tc.export_rope_rows(c, cs, sn, 1, 0, 1, 0)", [0, 1, 2], {NG: 1}, "refuses 'tables must be (T, D)'"),
    ("e2-swap-sin-T-D-1-empty", "E2", "accept",
     "r0=C((0,2,2)); p0=C((0,2,2)); i0=C((0,2,2)); j0=C((0,2,2)); cs=H((2,2)); sn=H((2,2,1))",
     "tc.swap_row_pairs_with_rope([r0], [p0], [i0], [j0], cs, sn, 1, 1, 0, 0, 0)", "tuple", {NG: 1},
     "refuses 'tables must be (T, D)'"),
    ("e2c-sin-trailing-2", "E2 control", "refuse", "x=C((0,2,2)); cs=H((2,2)); sn=H((2,2,2))",
     "tc.rope_apply(x, cs, sn, 0)", ["rope_apply: tables must be (T, D)", "sin [2, 2, 2]"], {}, "same"),
    ("e2c-sin-rank1", "E2 control", "refuse", "x=C((0,2,2)); cs=H((2,2)); sn=H((2,))",
     "tc.rope_apply(x, cs, sn, 0)", ["rope_apply: tables must be (T, D)", "sin [2]"], {}, "same"),
    ("e2c-cos-leading-1", "E2 control", "refuse", "x=C((0,2,2)); cs=H((1,2,2)); sn=H((2,2))",
     "tc.rope_apply(x, cs, sn, 0)", ["rope_apply: tables must be (T, D)", "cos [1, 2, 2]"], {}, "same"),
    ("e2c-sin-wrong-D", "E2 control", "refuse", "x=H((1,2,2)); cs=H((2,2)); sn=H((2,4,1))",
     "tc.rope_apply(x, cs, sn, 0)", ["rope_apply: tables must be (T, D)", "sin [2, 4, 1]"], {}, "same"),
    # ---- E3 (F06): every reduction entry normalizes, range-checks and de-duplicates its axes first
    ("e3-sum-duplicate", "E3", "refuse", "a=H((2,))", "a.sum([0, 0])",
     ["reduce: duplicate value in axis", "axes [0, 0]", "operand [2]"], {"raises": "ValueError"},
     "reaches the kernel with red_size 4 over a 2-element buffer (the review's case)"),
    ("e3-sum-out-of-range", "E3", "refuse", "a=H((2,))", "a.sum([8])",
     ["reduce: axis 8 is out of bounds for array of dimension 1"], {"raises": "IndexError"},
     "writes ReduceSpec::reduced[8] and reads shape[8] (the review's case)"),
    ("e3-sum-negative-duplicate", "E3", "refuse", "a=H((2,3))", "a.sum([1, -1])",
     ["reduce: duplicate value in axis", "axes [1, -1]"], {"raises": "ValueError"}, "reaches the kernel"),
    ("e3-sum-negative-out-of-range", "E3", "refuse", "a=H((2,3))", "a.sum([-3])",
     ["reduce: axis -3 is out of bounds for array of dimension 2"], {"raises": "IndexError"}, "writes slot -1"),
    ("e3-sum-rank0-axis", "E3", "refuse", "a=H((1,)).reshape([])", "a.sum([0])",
     ["reduce: axis 0 is out of bounds for array of dimension 0"], {"raises": "IndexError"},
     "indexes an empty shape vector"),
    ("e3-sum-empty-duplicate", "E3", "refuse", "a=C((0,3))", "a.sum([1, 1])",
     ["reduce: duplicate value in axis"], {"raises": "ValueError"},
     "returns [0] (axis 1 counted twice; the output is empty, so nothing is allocated)"),
    ("e3-mean-duplicate", "E3", "refuse", "a=H((2,3))", "a.mean([0, 0])",
     ["mean: duplicate value in axis", "axes [0, 0]"], {"raises": "ValueError"}, "counts axis 0 twice, then sums"),
    ("e3-mean-out-of-range", "E3", "refuse", "a=H((2,))", "a.mean([5])",
     ["mean: axis 5 is out of bounds for array of dimension 1"], {"raises": "IndexError"},
     "reads a.shape()[5] before sum"),
    ("e3-max-duplicate", "E3", "refuse", "a=H((2,3))", "a.max([1, 1])",
     ["reduce: duplicate value in axis"], {"raises": "ValueError"}, "reaches the kernel"),
    ("e3-min-out-of-range", "E3", "refuse", "a=H((2,3))", "a.min([2])",
     ["reduce: axis 2 is out of bounds for array of dimension 2"], {"raises": "IndexError"}, "writes slot 2's shape"),
    ("e3-prod-duplicate", "E3", "refuse", "a=H((2,3))", "a.prod([-2, 0])",
     ["reduce: duplicate value in axis"], {"raises": "ValueError"}, "reaches the kernel"),
    ("e3-var-out-of-range", "E3", "refuse", "a=H((2,3))", "a.var([3])",
     ["mean: axis 3 is out of bounds for array of dimension 2"], {"raises": "IndexError"}, "reads shape[3]"),
    ("e3-std-duplicate", "E3", "refuse", "a=H((2,3))", "a.std([1, 1])",
     ["mean: duplicate value in axis"], {"raises": "ValueError"}, "reaches the kernel"),
    ("e3-softmax-out-of-range", "E3", "refuse", "a=H((2,3))", "a.softmax(-3)",
     ["reduce: axis -3 is out of bounds for array of dimension 2"], {"raises": "IndexError"},
     "writes keep_shape[-1] (the process can die)"),
    ("e3-log_softmax-out-of-range", "E3", "refuse", "a=H((2,3))", "a.log_softmax(2)",
     ["reduce: axis 2 is out of bounds for array of dimension 2"], {"raises": "IndexError"}, "writes slot 2"),
    ("e3-argmax-out-of-range", "E3", "refuse", "a=H((2,3))", "a.argmax(2)",
     ["argmax: axis 2 is out of bounds for array of dimension 2"], {"raises": "IndexError"},
     "reads shape[2] in axis_split"),
    ("e3-argmin-out-of-range", "E3", "refuse", "a=H((2,3))", "a.argmin(-3)",
     ["argmin: axis -3 is out of bounds for array of dimension 2"], {"raises": "IndexError"}, "reads shape[-1]"),
    ("e3c-sum-valid-negative", "E3 control", "preflight", "a=H((2,3))", "a.sum([-1])",
     ["CUDA error at reduce_lastdim"], {}, "same"),
    ("e3c-sum-two-distinct", "E3 control", "preflight", "a=H((2,3,4))", "a.sum([2, 0])",
     ["CUDA error at reduce"], {}, "same"),
    ("e3c-mean-valid", "E3 control", "preflight", "a=H((2,3))", "a.mean([0, 1])",
     ["CUDA error at reduce_lastdim"], {}, "same"),
    ("e3c-argmax-valid", "E3 control", "preflight", "a=H((2,3))", "a.argmax(-1)", ["CUDA error at arg"], {}, "same"),
    # ---- E4 (F07): the output's byte bound in every pair preflight, before any pair's work. The rope side uses D = 0
    # (tables (T, 0) covering every span, empty on the CPU), so Build F2's span term passes and its failure is the
    # late byte bound alone.
    ("e4-splice-byte-bound", "E4", "refuse", f"o=C((0,{P61},2),'uint8'); i=C((0,{P61},2),'uint8')",
     "tc.splice_rows(o, i, 1, 0, 0)",
     ["splice_rows: the output NDArray: byte size overflows int64", "[0, 4611686018427387904, 2]",
      "operand 0 old_cache [0, 2305843009213693952, 2]"], {"raises": "OverflowError", NG: 1},
     "the constructor refuses it with its bare text, after the check"),
    ("e4-swap-pair1-byte-bound", "E4", "refuse",
     f"r0=C((0,4,2),'uint8'); i0=C((0,1,2),'uint8'); r1=C((0,{P61},2),'uint8'); i1=C((0,{P61},2),'uint8'); "
     "p0=C((0,4,0)); p1=C((0,4,0)); j0=C((0,1,0)); j1=C((0,1,0)); cs=C((1,0)); sn=C((1,0))",
     "tc.swap_row_pairs_with_rope([r0, r1], [p0, p1], [i0, i1], [j0, j1], cs, sn, 1, 1, 0, 0, 0)",
     ["splice_rows: the output NDArray: byte size overflows int64", "[pair 1 of 2]"],
     {"raises": "OverflowError", NG: 1}, "runs pair 0, then pair 1's constructor refuses (no pair named; the review's case)"),
    ("e4-arena-layer1-byte-bound", "E4", "refuse",
     f"r0=C((0,4,2),'uint8'); r1=C((0,{P61},2),'uint8'); p0=C((0,4,0),'uint8'); p1=C((0,4,0),'uint8'); "
     f"i0=C((0,{P61},2),'uint8'); i1=C((0,{P61},2),'uint8'); j0=C((0,{P61},0),'uint8'); j1=C((0,{P61},0),'uint8'); "
     f"cs=C(({P61},0),'uint8'); sn=C(({P61},0),'uint8')",
     f"tc.arena_row_pair_transaction([r0, r1], [p0, p1], [i0, i1], [j0, j1], cs, sn, 1, 1, 0, 0, {P61})",
     ["splice_rows: the output NDArray: byte size overflows int64", "[pair 1 of 2]"],
     {"raises": "OverflowError", NG: 1}, "runs layer 0, then layer 1's constructor refuses"),
    ("e4c-splice-fits", "E4 control", "accept", f"o=C((0,{P61},2),'uint8'); i=C((0,1,2),'uint8')",
     "tc.splice_rows(o, i, 1, 0, 0)", [0, P61 + 1, 2], {NG: 1}, "same"),
    # ---- D1: domain refusals (refused before any work; Build B's answers in BOUNDARY_F3.json)
    ("d1-where-empty-mixed-dtype", "D1 (F05)", "refuse", "c=C((0,)); x=C((0,)); y=C((0,),'float16')",
     "tc.where(c, x, y)", ["where: x has dtype float32 but y has float16"], {},
     "returns [0] (Build B's empty no-op)"),
    ("d1-where-all-true-mixed-dtype", "D1 (F05)", "refuse", "c=H((2,)); x=H((2,)); y=H((2,),'float16')",
     "tc.where(c, x, y)", ["where: x has dtype float32 but y has float16"], {}, "same (the review's case)"),
    ("d1-rank9-binary-empty", "D1 (F04)", "refuse", "a=C((0,)).reshape([1]*8+[0]); b=C((0,)).reshape([1]*8+[0])",
     "a + b", ["ew_binary: rank 9 exceeds TC_MAX_DIMS=8"], {}, "same (the review's case)"),
    ("d1-rank9-compare-empty", "D1 (F04)", "refuse", "a=C((0,)).reshape([1]*8+[0])", "a.gt(a)",
     ["compare: rank 9 exceeds TC_MAX_DIMS=8"], {}, "same"),
    ("d1-rank9-where-empty", "D1 (F04)", "refuse", "a=C((0,)).reshape([1]*8+[0])", "tc.where(a, a, a)",
     ["where: rank 9 exceeds TC_MAX_DIMS=8"], {}, "same"),
    ("d1-rank9-expand-empty", "D1 (F04)", "refuse", "a=C((0,))", "a.expand([1]*8+[0])",
     ["broadcast_to: rank 9 exceeds TC_MAX_DIMS=8"], {}, "same"),
    ("d1-rank9-masked_fill-empty", "D1 (F04)", "refuse", "a=C((0,)).reshape([1]*8+[0])", "a.masked_fill(a, 0.0)",
     ["masked_fill: rank 9 exceeds TC_MAX_DIMS=8"], {}, "same"),
    ("d1-rank9-axpy-empty", "D1 (F04)", "refuse", "a=C((0,)).reshape([1]*8+[0])", "tc._C.axpy_(a, a, 1.0)",
     ["axpy_: rank 9 exceeds TC_MAX_DIMS=8"], {}, "same"),
    ("d1-rank9-sum-empty", "D1 (F04)", "refuse", "a=C((0,)).reshape([1]*8+[0])", "a.sum()",
     ["reduce: rank 9 exceeds TC_MAX_DIMS=8"], {}, "same"),
    ("d1-axpy-empty-destination-shape", "D1 (F03)", "refuse", "p=C((0,1)); o=C((0,2))", "tc._C.axpy_(p, o, 1.0)",
     ["axpy_: the update of param [0, 1] by other [0, 2] broadcasts to [0, 2]", "does not map element for element"],
     {}, "same (the review's case; Build B's zero-byte no-op)"),
    ("d1-rope-table-dtype-empty", "D1 (F08)", "refuse", "x=C((0,2,2)); cs=H((2,2),'float16'); sn=H((2,2))",
     "tc.rope_apply(x, cs, sn, 0)", ["rope_apply: table/x dtype mismatch", "cos [2, 2] float16"], {},
     "same (the review's case: no table byte is read)"),
    ("d1-rope-table-dtype-swap-empty", "D1 (F08)", "refuse",
     "r0=C((0,2,2)); p0=C((0,2,2)); i0=C((0,2,2)); j0=C((0,2,2)); cs=H((2,2)); sn=H((2,2),'bfloat16')",
     "tc.swap_row_pairs_with_rope([r0], [p0], [i0], [j0], cs, sn, 1, 1, 0, 0, 0)",
     ["rope_apply: table/x dtype mismatch", "[pair 0 of 1]"], {NG: 1}, "same"),
]


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        pytest.skip("CPU lane only (CUDA_VISIBLE_DEVICES must be empty)")
    res = T2.run_rows(PKG_PARENT, str(tmp_path_factory.mktemp("bwdf3_p1")), ROWS)
    assert res["child_exit"] == 0, res
    assert res["control_calls"] > 0, res   # the counter sees CUDA calls in this process
    rec = os.environ.get("BWDF3_RECORD_DIR")
    if rec:
        os.makedirs(rec, exist_ok=True)
        with open(os.path.join(rec, "p1_rows.json"), "w") as f:
            json.dump({"rows": ROWS, **res}, f, indent=1)
    return res["results"]


@pytest.mark.parametrize("row", ROWS, ids=[r[0] for r in ROWS])
def test_cpu_p1_row(results, row):
    why = T2.verdict(row, results.get(row[0]))
    assert why is None, (row[0], why)
