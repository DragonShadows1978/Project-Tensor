"""BWD-F4 (orders/BWDF4_REV3_HARDENING.md) G1: one CPU row per H1 / H2 case and control, run with CUDA_VISIBLE_DEVICES
empty through the public API, with ENG3's LD_PRELOAD CUDA-call counter. The harness (one child process per row,
per-function counts, the row kinds refuse / accept / preflight and their verdicts) is BWD-F2's
(tests/test_bwdf2_fixes.py run_rows and verdict), imported unchanged, as BWD-F3's rows do.

Row classes (the second field):
- "H1": mean's checked count (review BWDF1R3-F01). A selected extent product that overflows int64 is refused by mean
  itself ("mean: the reduced element count ... overflows int64"), before sum runs. Build F3 multiplied the extents in
  signed int64 first (signed overflow, undefined behaviour) and refused only in the reducer, with the reducer's text
  ("reduce: ..."); the row fails on Build F3 on that text and holds on Build F4.
- "H1 (UB only)": the reviewer's own case, an empty tensor reshaped to [2^32, 2^32, 0, 0] with mean over [0, 1, 2]. The
  result is an empty tensor on every build: Build F3's product wraps to 0 (2^32 * 2^32 = 2^64 mod 2^64, then * 0), the
  value emptiness-first gives, so the defect there is the undefined behaviour of the signed overflow, which a CPU run
  without a sanitizer does not observe. The row holds on both builds (it is NOT a failing row on Build F3; see the
  report's Deviation 1 and the replica scripts/bwdf4_count_replica.py, which shows the overflow under UBSan).
- "H1 control" / "H2 control": the rule's other side, unchanged from Build F3 (a product that fits, a valid mean that
  reaches the device step on rank 1 to 4, a rank-8 table).
- "H2": a RoPE table of rank above TC_MAX_DIMS (8) is refused by rope_table_layout_ok (ruling D1), through each of its
  callers (rope_apply, export_rope_rows, the swap preflight); Build F3 accepted it.
- "H3": the corrected comment above preflight_out_bound: export_rows of an empty reshaped view [0, 2^62, 2] (uint8)
  fires the byte bound (holds on Build F3 and Build F4: a comment-only change).
Every refusal row asserts 0 CUDA calls. The last field is Build F3's behaviour (re-measured by scripts/bwdf4_boundary.py).

Operands: H(shape, dtype) a host tensor of ones, C(shape, dtype) a CUDA tensor with zero elements (built without a
device). A host operand with elements reaches the kernel launch on the CPU ("CUDA error at <op>: no CUDA-capable
device"), which is the CPU form of a passing control (the engine has no CPU reduction, so a mean's VALUE is not
observable here).

Environment: BWDF4_PKG_PARENT selects the tensor_cuda package under test (Build F3, Build F4, mutants).
Prior art: GCC's __builtin_mul_overflow (2015); NumPy's rule that an empty reduction axis gives count 0; the RoPE
formulation (Su et al., 2021, "RoFormer"); ENG3's LD_PRELOAD counter (ltrace-style interposition); DeMillo, Lipton and
Sayward (1978), mutation testing (rows named in artifacts/bwdf1/f4/MUTANTS_F4.json kill h01 and h02). Ours: the rows.
"""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_bwdf2_fixes as T2  # noqa: E402  (BWD-F2's K1 harness)

PKG_PARENT = os.environ.get("BWDF4_PKG_PARENT", os.path.dirname(HERE))
NG = "no_grad"
R9 = "[2,2,1,1,1,1,1,1,1]"      # rank 9: [T, D, 1 x 7]
R8 = "[2,2,1,1,1,1,1,1]"        # rank 8: [T, D, 1 x 6]
MEAN_OVF = "mean: the reduced element count of"

# (id, class, kind, setup, call, expectation, flags, Build F3's behaviour)
ROWS = [
    # ---- H1 (F01): mean refuses an overflowing selected-extent product itself, before sum
    ("h1-overflow-2pow32-pair", "H1", "refuse", "a=C((0,)).reshape([2**32,2**32,1,0])", "a.mean([0,1])",
     [MEAN_OVF, "[4294967296, 4294967296, 1, 0] overflows int64", "(axes [0, 1])", "refused before any allocation"],
     {"raises": "OverflowError"}, "refuses, in the reducer: 'reduce: the reduced element count of [...]' (signed overflow first)"),
    ("h1-overflow-2pow21-triple", "H1", "refuse", "a=C((0,)).reshape([2**21,2**21,2**21,0])", "a.mean([0,1,2])",
     [MEAN_OVF, "[2097152, 2097152, 2097152, 0] overflows int64", "(axes [0, 1, 2])"], {"raises": "OverflowError"},
     "refuses, in the reducer (the product 2^63 is one past int64)"),
    ("h1-overflow-negative-axes", "H1", "refuse", "a=C((0,)).reshape([2**32,2**32,1,0])", "a.mean([-4,-3])",
     [MEAN_OVF, "(axes [-4, -3])"], {"raises": "OverflowError"}, "refuses, in the reducer"),
    ("h1-overflow-edge-2pow62-times-2", "H1", "refuse", "a=C((0,)).reshape([2**62,2,1,0])", "a.mean([0,1])",
     [MEAN_OVF, "[4611686018427387904, 2, 1, 0] overflows int64"], {"raises": "OverflowError"},
     "refuses, in the reducer (2^63 is one past int64)"),
    ("h1u-reviewer-empty-case", "H1 (UB only)", "accept", "a=C((0,)).reshape([2**32,2**32,0,0])", "a.mean([0,1,2])", [0], {},
     "same result [0] (the product wraps to 0; the signed overflow is not observable without a sanitizer)"),
    ("h1u-reviewer-empty-case-negative-axes", "H1 (UB only)", "accept", "a=C((0,)).reshape([2**32,2**32,0,0])",
     "a.mean([-4,-3,-2])", [0], {}, "same result [0]"),
    ("h1c-product-fits-2pow31-pair", "H1 control", "accept", "a=C((0,)).reshape([2**31,2**31,1,0])", "a.mean([0,1])",
     [1, 0], {}, "same (the product 2^62 fits)"),
    ("h1c-product-fits-edge-2pow62", "H1 control", "accept", "a=C((0,)).reshape([2**62,1,1,0])", "a.mean([0,1])",
     [1, 0], {}, "same (the product 2^62 fits)"),
    ("h1c-empty-selected-axis-first", "H1 control", "accept", "a=C((0,)).reshape([0,2**32,2**32,0])", "a.mean([0,1,2])",
     [0], {}, "same (a zero extent first: the old product reached 0 before overflowing)"),
    ("h1c-mean-rank1", "H1 control", "preflight", "a=H((3,))", "a.mean([0])", ["CUDA error at reduce"], {}, "same"),
    ("h1c-mean-rank2", "H1 control", "preflight", "a=H((2,3))", "a.mean([0,1])", ["CUDA error at reduce"], {}, "same"),
    ("h1c-mean-rank3-negative", "H1 control", "preflight", "a=H((2,3,4))", "a.mean([-1,0])", ["CUDA error at reduce"], {},
     "same"),
    ("h1c-mean-rank4", "H1 control", "preflight", "a=H((2,3,4,5))", "a.mean([1,3])", ["CUDA error at reduce"], {}, "same"),
    # ---- H2 (F02): a RoPE table of rank above TC_MAX_DIMS (8) is refused (ruling D1)
    ("h2-rope_apply-sin-rank9-empty", "H2", "refuse", f"x=C((0,2,2)); cs=H((2,2)); sn=H((2,2)).reshape({R9})",
     "tc.rope_apply(x, cs, sn, 0)", ["rope_apply: tables must be (T, D)", "sin [2, 2, 1, 1, 1, 1, 1, 1, 1]"], {},
     "accepts, returns [0, 2, 2] (the review's case)"),
    ("h2-rope_apply-cos-rank9-empty", "H2", "refuse", f"x=C((0,2,2)); cs=H((2,2)).reshape({R9}); sn=H((2,2))",
     "tc.rope_apply(x, cs, sn, 0)", ["rope_apply: tables must be (T, D)", "cos [2, 2, 1, 1, 1, 1, 1, 1, 1]"], {},
     "accepts, returns [0, 2, 2]"),
    ("h2-rope_apply-sin-rank9-reading", "H2", "refuse", f"x=H((1,2,2)); cs=H((2,2)); sn=H((2,2)).reshape({R9})",
     "tc.rope_apply(x, cs, sn, 0)", ["rope_apply: tables must be (T, D)", "sin [2, 2, 1, 1, 1, 1, 1, 1, 1]"], {},
     "reaches the kernel ('CUDA error at rope_apply')"),
    ("h2-export_rope_rows-sin-rank9", "H2", "refuse",
     f"c=C((0,4,2)); cs=H((8,2)); sn=H((8,2)).reshape([8,2]+[1]*7)", "tc.export_rope_rows(c, cs, sn, 1, 0, 1, 0)",
     ["export_rope_rows: tables must be (T, D)", "sin [8, 2, 1, 1, 1, 1, 1, 1, 1]"], {NG: 1},
     "accepts, returns [0, 1, 2]"),
    ("h2-swap-sin-rank9", "H2", "refuse",
     f"r0=C((0,2,2)); p0=C((0,2,2)); i0=C((0,2,2)); j0=C((0,2,2)); cs=H((2,2)); sn=H((2,2)).reshape({R9})",
     "tc.swap_row_pairs_with_rope([r0], [p0], [i0], [j0], cs, sn, 1, 1, 0, 0, 0)",
     ["rope_apply: tables must be (T, D)", "[pair 0 of 1]"], {NG: 1}, "accepts, returns a tuple"),
    ("h2c-rope_apply-sin-rank8-empty", "H2 control", "accept", f"x=C((0,2,2)); cs=H((2,2)); sn=H((2,2)).reshape({R8})",
     "tc.rope_apply(x, cs, sn, 0)", [0, 2, 2], {}, "same"),
    ("h2c-rope_apply-both-rank8-reading", "H2 control", "preflight",
     f"x=H((1,2,2)); cs=H((2,2)).reshape({R8}); sn=H((2,2)).reshape({R8})", "tc.rope_apply(x, cs, sn, 0)",
     ["CUDA error at rope_apply"], {}, "same"),
    ("h2c-rope_apply-sin-T-D-1", "H2 control", "preflight", "x=H((1,2,2)); cs=H((2,2)); sn=H((2,2,1))",
     "tc.rope_apply(x, cs, sn, 0)", ["CUDA error at rope_apply"], {}, "same (BWD-F3 E2)"),
    ("h2c-export_rope_rows-sin-rank8", "H2 control", "accept",
     f"c=C((0,4,2)); cs=H((8,2)); sn=H((8,2)).reshape([8,2]+[1]*6)", "tc.export_rope_rows(c, cs, sn, 1, 0, 1, 0)",
     [0, 1, 2], {NG: 1}, "same"),
    ("h2c-rank1-table-still-refused", "H2 control", "refuse", "x=C((0,2,2)); cs=H((2,2)); sn=H((2,))",
     "tc.rope_apply(x, cs, sn, 0)", ["rope_apply: tables must be (T, D)", "sin [2]"], {}, "same"),
    # ---- H3 (F03): the corrected comment's claim, measured: a reshaped empty view fires export_rows' byte bound
    ("h3-export_rows-empty-view-fires-bound", "H3", "refuse", "c=C((0,),'uint8').reshape([0,2**62,2])",
     "tc.export_rows(c, 1, 0, 2**62)",
     ["export_rows: the output NDArray: byte size overflows int64", "[0, 4611686018427387904, 2]"],
     {"raises": "OverflowError", NG: 1}, "same (the comment's claim: the bound fires for export_rows, not only splice)"),
]


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        pytest.skip("CPU lane only (CUDA_VISIBLE_DEVICES must be empty)")
    res = T2.run_rows(PKG_PARENT, str(tmp_path_factory.mktemp("bwdf4_g1")), ROWS)
    assert res["child_exit"] == 0, res
    assert res["control_calls"] > 0, res   # the counter sees CUDA calls in this process
    rec = os.environ.get("BWDF4_RECORD_DIR")
    if rec:
        os.makedirs(rec, exist_ok=True)
        with open(os.path.join(rec, "g1_rows.json"), "w") as f:
            json.dump({"rows": ROWS, **res}, f, indent=1)
    return res["results"]


@pytest.mark.parametrize("row", ROWS, ids=[r[0] for r in ROWS])
def test_cpu_g1_row(results, row):
    why = T2.verdict(row, results.get(row[0]))
    assert why is None, (row[0], why)
