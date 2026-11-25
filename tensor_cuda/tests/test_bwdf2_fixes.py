"""BWD-F2 (orders/BWDF2_REV1_FIXES.md) K1: one row per review finding's named case that the CPU can reach, run with
CUDA_VISIBLE_DEVICES empty through the public API, with ENG3's LD_PRELOAD CUDA-call counter
(tests/support/cudart_count.c, compiled at test time; per-function counts read through eng3_cudart_count).

Row kinds:
- refuse: the call raises (RuntimeError unless the row names another type) with the listed substrings and makes
  0 CUDA calls (refused on the host, before any allocation or launch).
- accept: the call returns (the listed result) with 0 CUDA calls (empty work; nothing to launch).
- preflight: the call passes every host check and reaches its first device step, which fails without a device
  ("CUDA error at <op>: no CUDA-capable device"); with no_alloc the counter also shows 0 allocations and 0 launches
  (only the op's closing status check ran). This is the CPU form of a passing control: a GPU line (scripts/
  bwdf2_step_k2.py) checks the values.
Each row names its finding and Build E's expected behaviour (`e`); scripts/bwdf2_k1.py runs this table on Build E and
on Build F2 and records both (artifacts/bwdf1/f2/K1_RESULTS.json). Operands: H(shape, dtype) a host tensor of ones,
C(shape, dtype) a CUDA tensor with zero elements (built without a device), R(rank9) reshapes.

Environment: BWDF2_PKG_PARENT selects the tensor_cuda package under test (Build E, Build F2, mutants).
Prior art: NumPy's concatenate / stack / where / putmask rules; the RoPE formulation (Su et al., 2021, "RoFormer");
ENG3's LD_PRELOAD counter (ltrace-style interposition); DeMillo, Lipton and Sayward (1978), mutation testing (rows
named in artifacts/bwdf1/f2/MUTANTS_F2.json kill n01, n02, n03, n06). Ours: the rows.
"""
import json
import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_PARENT = os.environ.get("BWDF2_PKG_PARENT", os.path.dirname(HERE))
SHIM = os.path.join(HERE, "support", "cudart_count.c")
NG = "no_grad"
BIG = 2 ** 62

# (id, finding, kind, setup, call, expectation, flags, Build E's expected behaviour)
ROWS = [
    # ---- F01 / R3: both tables' read spans (sin), subtraction-based, negative pos0
    # BWD-F3 E1 (artifacts/bwdf1/f3/INTENDED_CHANGES_F3.json): the empty-launch rows below (x, inserts or the export's
    # output have no elements) read no table row and now pass; the reading cases stay refused (BWD-F3's P1 rows and
    # the GPU rows of scripts/bwdf2_step_k2.py f01 and scripts/bwdf3_step_p2.py e1).
    ("f01-rope_apply-short-sin", "F01 (BWD-F3 E1: accepted)", "accept", "x=C((0,2,2)); cs=H((2,2)); sn=H((1,2))",
     "tc.rope_apply(x, cs, sn, 0)", [0, 2, 2], {},
     "returns (only cos's span was checked)"),
    ("f01-swap-short-sin", "F01 (BWD-F3 E1: accepted)", "accept",
     "r0=C((0,2,2)); p0=C((0,2,2)); i0=C((0,2,2)); j0=C((0,2,2)); cs=H((2,2)); sn=H((1,2))",
     "tc.swap_row_pairs_with_rope([r0], [p0], [i0], [j0], cs, sn, 1, 1, 0, 0, 0)",
     "tuple", {NG: 1},
     "returns (the preflight checked cos only)"),
    ("f01-arena-short-sin", "F01 (BWD-F3 E1: accepted)", "accept",
     "r0=C((0,2,2)); p0=C((0,2,2)); i0=C((0,2,2)); j0=C((0,2,2)); cs=H((2,2)); sn=H((1,2))",
     "tc.arena_row_pair_transaction([r0], [p0], [i0], [j0], cs, sn, 1, 1, 0, 0, 4)",
     "tuple", {NG: 1}, "returns (delegates to the swap preflight)"),
    ("f01-rope_apply-negative-pos0", "F01 (BWD-F3 E1: accepted)", "accept", "x=C((0,2,2)); cs=H((4,2)); sn=H((4,2))",
     "tc.rope_apply(x, cs, sn, -1)", [0, 2, 2], {},
     "returns (pos0 + L <= cos rows passed)"),
    ("f01-rope_apply-tables-cover", "F01 control", "accept", "x=C((0,2,2)); cs=H((2,2)); sn=H((2,2))",
     "tc.rope_apply(x, cs, sn, 0)", [0, 2, 2], {}, "same"),
    # ---- F05 / R3: launch dtype in the shared validator (CPU form: rope_apply on a host uint8 operand)
    ("f05-rope_apply-uint8-launch-dtype", "F05", "refuse", "x=H((1,2,2),'uint8'); cs=H((2,2),'uint8'); sn=H((2,2),'uint8')",
     "tc.rope_apply(x, cs, sn, 0)", ["rope_apply: op supports float32/float16/bfloat16 only", "x [1, 2, 2] uint8"], {},
     "refuses after allocating the output, with the bare dispatch text"),
    ("f05-export_row_pair-rope-side", "F05 (BWD-F3 E1: accepted)", "accept",
     "r=C((0,4,2)); p=C((0,4,2)); cs=C((0,2)); sn=C((0,2))",
     "tc.export_row_pair(r, p, cs, sn, 1, 1, 0, 0, 1, 0)", "tuple", {NG: 1},
     "same on the CPU (the raw export of an empty cache makes no call); the work-before-refusal is a GPU row"),
    # ---- F02 / R2: the rank ceiling of the fixed-size metadata arrays
    ("f02-stack-rank8-to-9", "F02", "refuse", "a=C((1,1,1,1,1,1,1,0)); b=C((1,1,1,1,1,1,1,0))", "tc.stack([a, b], 0)",
     ["stack: rank 9 exceeds TC_MAX_DIMS=8"], {}, "writes DimCopySpec slot 8, then cat's status check fails"),
    ("f02-cat-rank9-empty", "F02", "refuse", "a=C((0,)).reshape([1]*8+[0]); b=C((0,)).reshape([1]*8+[0])",
     "tc.cat([a, b], 0)", ["cat: rank 9 exceeds TC_MAX_DIMS=8"], {}, "writes slot 8 for each operand"),
    ("f02-slice-rank9", "F02", "refuse", "a=C((0,)).reshape([1]*8+[0])", "a.slice(0, 0, 1)",
     ["slice: rank 9 exceeds TC_MAX_DIMS=8"], {}, "writes slot 8"),
    ("f02-binary-rank9", "F02", "refuse", "a=H((1,)).reshape([1]*9); b=H((1,)).reshape([1]*9)", "a + b",
     ["ew_binary: rank 9 exceeds TC_MAX_DIMS=8"], {}, "writes DimSpec slot 8 and launches"),
    ("f02-binary-rank9-empty", "F02", "refuse", "a=C((0,)).reshape([1]*8+[0]); b=C((0,)).reshape([1]*8+[0])", "a + b",
     ["ew_binary: rank 9 exceeds TC_MAX_DIMS=8"], {}, "returns the empty result (R2 refuses empty operands too)"),
    ("f02-compare-rank9", "F02", "refuse", "a=H((1,)).reshape([1]*9); b=H((1,)).reshape([1]*9)", "a.gt(b)",
     ["compare: rank 9 exceeds TC_MAX_DIMS=8"], {}, "writes DimSpec slot 8 and launches"),
    ("f02-where-rank9", "F02", "refuse", "a=H((1,)).reshape([1]*9)", "tc.where(a, a, a)",
     ["where: rank 9 exceeds TC_MAX_DIMS=8"], {}, "writes Dim3Spec slot 8 and launches"),
    ("f02-masked_fill-rank9", "F02", "refuse", "a=H((1,)).reshape([1]*9)", "a.masked_fill(a, 0.0)",
     ["masked_fill: rank 9 exceeds TC_MAX_DIMS=8"], {}, "builds the fill, writes Dim3Spec slot 8, launches"),
    ("f02-reduce-rank9", "F02", "refuse", "a=H((1,)).reshape([1]*9)", "a.sum()",
     ["reduce: rank 9 exceeds TC_MAX_DIMS=8"], {}, "writes ReduceSpec slot 8 and launches"),
    ("f02-permute-rank9", "F02", "refuse", "a=H((1,)).reshape([1]*9)", "a.permute([8,7,6,5,4,3,2,1,0])",
     ["permute: rank 9 exceeds TC_MAX_DIMS=8"], {}, "writes PermSpec slot 8 and launches"),
    ("f02-expand-rank9", "F02", "refuse", "a=H((1,))", "a.expand([1]*9)",
     ["broadcast_to: rank 9 exceeds TC_MAX_DIMS=8"], {}, "allocates the zeros, then ew_binary writes slot 8"),
    ("f02-rank8-control", "F02 control", "preflight", "a=H((1,)).reshape([1]*8); b=H((1,)).reshape([1]*8)", "a + b",
     ["CUDA error at ew_binary"], {}, "same"),
    # ---- F03 / R1: check what is read (dtype and device only for members whose bytes are copied)
    ("f03-cat-named-case", "F03", "preflight", "a=H((2,3)); b=H((0,3),'float16')", "tc.cat([a, b], 0)",
     ["CUDA error at cat"], {}, "refuses: operand 1 has dtype float16"),
    ("f03-cat-empty-later-device", "F03", "preflight", "a=H((2,3)); b=C((0,3))", "tc.cat([a, b], 0)",
     ["CUDA error at cat"], {}, "refuses: operand 1 is on cuda:0"),
    ("f03-stack-empty-mixed-dtype", "F03 / R9", "preflight", "a=C((0,3)); b=C((0,3),'float16')", "tc.stack([a, b], 0)",
     ["CUDA error at cat"], {"no_alloc": 1}, "refuses: operand 1 has dtype float16"),
    ("f03-cat-empty-mixed-device-dtype", "F03 / R9", "preflight", "a=C((0,3)); b=H((0,3),'bfloat16')",
     "tc.cat([a, b], 0)", ["CUDA error at cat"], {"no_alloc": 1}, "refuses: operand 1 has dtype bfloat16"),
    ("f03-write_rows-empty-device", "F03", "accept", "b=C((0,4,2)); s=H((0,1,2))", "tc.write_rows(b, s, 0)", None,
     {NG: 1}, "refuses: device mismatch"),
    ("f03-export_rope_rows-empty-host-cos", "F03", "accept", "c=C((0,4,2)); cs=H((0,2)); sn=C((0,2))",
     "tc.export_rope_rows(c, cs, sn, 1, 0, 0, 0)", [0, 0, 2], {NG: 1}, "refuses: tables must be on the cache's device"),
    ("f03-cat-copied-dtype-still-refused", "F03 control", "refuse", "a=H((0,3),'float16'); b=H((2,3))",
     "tc.cat([a, b], 0)", ["cat: operand 1 has dtype float32 but operand 0 has float16", "[2, 3]"], {}, "same"),
    ("f03-cat-copied-device-still-refused", "F03 control", "refuse", "a=C((0,3)); b=H((2,3))", "tc.cat([a, b], 0)",
     ["cat: operand 1 is on cpu but operand 0 is on cuda:0"], {}, "same"),
    # ---- F07 / R5: the data operands' dtypes; the condition in its own dtype
    ("f07-where-data-dtype", "F07", "refuse", "c=H((2,)); x=H((2,)); y=H((2,),'float16')", "tc.where(c, x, y)",
     ["where: x has dtype float32 but y has float16"], {}, "allocates and launches, reading y as float32"),
    ("f07-where-bool-cond", "F07 control", "preflight", "c=B([0,1,1,1]); x=H((4,)); y=H((4,))", "tc.where(c, x, y)",
     ["CUDA error at where"], {}, "same on the CPU; the condition read is a GPU row"),
    ("f07-masked_fill-bool-mask", "F07 control", "preflight", "m=B([0,1,1,1]); a=H((4,))", "a.masked_fill(m, 2.0)",
     ["CUDA error at where"], {}, "same on the CPU; the mask read is a GPU row"),
    # BWD-F3 D1 (INTENDED_CHANGES_F3.json): where's x and y must match for every call, an empty one included.
    ("f07-where-empty-mixed-data", "F07 control (BWD-F3 D1: refused)", "refuse",
     "c=C((0,)); x=C((0,)); y=C((0,),'float16')",
     "tc.where(c, x, y)", ["where: x has dtype float32 but y has float16"], {}, "same"),
    # ---- F08 / R7: checked sums and products
    ("f08-cat-sum-overflow", "F08", "refuse", f"a=C((0,{BIG}),'uint8'); b=C((0,{BIG}),'uint8')", "tc.cat([a, b], 1)",
     ["cat: the concatenated extent at dim 1 overflows int64"], {"raises": "OverflowError"},
     "the sum wraps negative; the constructor refuses 'negative extent' (ValueError)"),
    ("f08-splice-sum-overflow", "F08", "refuse", f"o=C((0,{BIG},1),'uint8'); i=C((0,{BIG},1),'uint8')",
     f"tc.splice_rows(o, i, 1, {BIG}, {BIG})", ["splice_rows: the output extent", "overflows int64"],
     {"raises": "OverflowError", NG: 1}, "the sum wraps negative; 'negative extent' (ValueError)"),
    ("f08-arena-span-overflow", "F08", "refuse", "r0=C((0,4,2)); p0=C((0,4,2)); i0=C((0,1,2)); j0=C((0,1,2)); "
     "cs=C((0,2)); sn=C((0,2))",
     f"tc.arena_row_pair_transaction([r0], [p0], [i0], [j0], cs, sn, 1, 1, {BIG}, {BIG + 2 ** 61}, 4)",
     ["arena_row_pair_transaction: sink_tokens", "overflows int64"], {"raises": "OverflowError", NG: 1},
     "sink + mount wraps negative; refused as 'invalid head/tail'"),
    ("f08-write_rows-empty-leading-product", "F08", "accept",
     "b=C((0,)).reshape([2**32,2**32,0,3,1]); s=C((0,)).reshape([2**32,2**32,0,2,1])", "tc.write_rows(b, s, 0)",
     None, {NG: 1}, "same result through a wrapped product (signed overflow; not visible without a sanitizer)"),
    # ---- F09 / R8: launch dtype before allocation; __getitem__ axis before extent
    ("f09-slice-uint8", "F09", "refuse", "a=H((2,2),'uint8')", "a.slice(0, 0, 1)",
     ["slice: op supports float32/float16/bfloat16 only", "refused before any allocation"], {},
     "allocates the output, then the dispatch text (no context)"),
    ("f09-getitem-rank0", "F09", "refuse", "a=H((1,)).reshape([])", "a[0]",
     ["getitem: too many indices", "shape []"], {"raises": "IndexError"},
     "reads shape()[0] of an empty shape vector, then slice refuses 'dim 0 out of range'"),
    # ---- F11 / R10: dim-error context, pair and layer index
    ("f11-export_rows-dim-context", "F11", "refuse", "c=C((0,4,2))", "tc.export_rows(c, 3, 0, 1)",
     ["export_rows: dim out of range: dim 3 for rank 3", "cache [0, 4, 2]"], {NG: 1}, "bare 'dim out of range'"),
    ("f11-evict_row_pairs-pair-index", "F11", "refuse", "r0=C((0,4,2)); p0=C((0,4,2)); r1=C((0,4,2)); p1=C((0,2,2))",
     "tc.evict_row_pairs([r0, r1], [p0, p1], 1, 1, 1, 2)", ["evict_rows: drop exceeds live tail", "[pair 1 of 2]"],
     {NG: 1}, "no pair index"),
    ("f11-swap-pair-index", "F11", "refuse",
     "r0=C((0,4,2)); p0=C((1,4,0)); r1=C((0,4,2)); p1=C((1,4,0)); i0=C((0,1,2)); j0=C((1,1,0)); "
     "i1=C((0,1,3)); j1=C((1,1,0)); cs=C((1,0)); sn=C((1,0))",
     "tc.swap_row_pairs_with_rope([r0, r1], [p0, p1], [i0, i1], [j0, j1], cs, sn, 1, 1, 0, 1, 0)",
     ["splice_rows: shape mismatch outside splice dim", "[pair 1 of 2]"], {NG: 1}, "no pair index"),
    ("f11-arena-layer-dim", "F11", "refuse", "r0=C((0,4,2)); p0=C((0,4,2)); i0=C((0,1,2)); j0=C((0,1,2)); "
     "cs=C((0,2)); sn=C((0,2))", "tc.arena_row_pair_transaction([r0], [p0], [i0], [j0], cs, sn, 5, 1, 0, 0, 4)",
     ["arena_row_pair_transaction raw: dim out of range: dim 5 for rank 3", "layer 0 of 1", "[0, 1, 2]"], {NG: 1},
     "bare 'dim out of range'"),
    # ---- F12 / R6: axpy_'s update maps element for element onto param
    ("f12-axpy-named-case", "F12", "refuse", "p=H((2,1)); o=H((2,2))", "tc._C.axpy_(p, o, 1.0)",
     ["axpy_: the update of param [2, 1] by other [2, 2] broadcasts to [2, 2]",
      "does not map element for element onto param"], {}, "launches (installs the wrong rows on a GPU)"),
    ("f12-axpy-trailing-broadcast", "F12", "refuse", "p=H((3,1)); o=H((1,4))", "tc._C.axpy_(p, o, 1.0)",
     ["broadcasts to [3, 4]", "does not map element for element"], {}, "launches"),
    ("f12-axpy-leading-one-other", "F12 control", "preflight", "p=H((3,)); o=H((1,3))", "tc._C.axpy_(p, o, 1.0)",
     ["CUDA error at ew_scalar"], {}, "same"),
    ("f12-axpy-leading-one-param", "F12 control", "preflight", "p=H((1,3)); o=H((3,))", "tc._C.axpy_(p, o, 1.0)",
     ["CUDA error at ew_scalar"], {}, "same"),
    # ---- F13 / R10: einsum preflights the extents its factor construction reads
    ("f13-einsum-shorter-term", "F13", "refuse", "a=H((2,0,1)); b=H((4,3))", "tc.einsum('ij,kj->ik', a, b)",
     ["shapes not broadcastable: einsum index 'j'", "[2, 0, 1]", "[4, 3]"], {}, "permutes b (a launch) first"),
]

_CHILD = r'''
import ctypes, json, os, sys
sys.path.insert(0, os.environ["BWDF2_PKG_PARENT"])
import numpy as np
import tensor_cuda as tc
L = ctypes.CDLL(None)
L.eng3_cudart_calls.restype = ctypes.c_ulong
L.eng3_cudart_count.restype = ctypes.c_ulong
L.eng3_cudart_name.restype = ctypes.c_char_p
names = [L.eng3_cudart_name(i).decode() for i in range(L.eng3_cudart_n())]
def counts():
    return {n: L.eng3_cudart_count(i) for i, n in enumerate(names)}
rows = json.loads(os.environ["BWDF2_ROWS"])
def H(shape, dt="float32"):
    return tc.tensor(np.ones(shape, np.uint8 if dt == "uint8" else np.float32), device="cpu", dtype=dt)
def C(shape, dt="float32"):
    assert int(np.prod(shape)) == 0, shape
    return tc.tensor(np.zeros(shape, np.uint8 if dt == "uint8" else np.float32), device="cuda", dtype=dt)
def B(vals):
    return tc.tensor(np.array(vals, bool), device="cpu", dtype="bool")
out = {"__tensor_cuda__": tc.__file__}
for rid, _f, _k, setup, call, _w, flags, _e in rows:
    env = {"tc": tc, "H": H, "C": C, "B": B}
    try:
        exec(setup, env)
    except BaseException as e:  # noqa: BLE001
        out[rid] = {"setup_error": f"{type(e).__name__}: {e}"}
        continue
    before, c0 = L.eng3_cudart_calls(), counts()
    try:
        if flags.get("no_grad"):
            with tc.no_grad():
                r = eval(call, env)
        else:
            r = eval(call, env)
        res = {"raised": None, "result": None if r is None else (list(r.shape) if hasattr(r, "shape")
                                                                  else type(r).__name__)}
    except BaseException as e:  # noqa: BLE001
        res = {"raised": type(e).__name__, "message": str(e)}
    c1 = counts()
    res["calls"] = L.eng3_cudart_calls() - before
    res["by_function"] = {n: c1[n] - c0[n] for n in names if c1[n] != c0[n]}
    out[rid] = res
base = L.eng3_cudart_calls()
try:
    tc.tensor([1.0], device="cuda")
except RuntimeError:
    pass
print(json.dumps({"results": out, "control_calls": L.eng3_cudart_calls() - base}))
'''


def run_rows(pkg_parent, tmpdir, rows=ROWS):
    """Run the rows in a child with the counter preloaded; returns the child's JSON (also used by scripts/bwdf2_k1.py)."""
    compiler = next((p for n in ("cc", "gcc", "clang") if (p := shutil.which(n))), None)
    if compiler is None:
        pytest.skip("no C compiler for the CUDA-call counter shim")
    shim = os.path.join(tmpdir, "cudart_count.so")
    r = subprocess.run([compiler, "-shared", "-fPIC", "-O2", "-o", shim, SHIM, "-ldl"], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, r.stderr
    # One child per row: a defective build can crash its process (Build E writes past the 8-slot launch specs on
    # rank-9 rows), and one crash must not hide the other rows. The last child runs no row: the counter's positive
    # control (a CUDA tensor with elements makes CUDA calls).
    out = {"results": {}, "child_exits": {}}
    for row in list(rows) + [None]:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_PRELOAD=shim, BWDF2_PKG_PARENT=pkg_parent,
                   BWDF2_ROWS=json.dumps([] if row is None else [row]))
        p = subprocess.run([sys.executable, "-B", "-P", "-c", _CHILD], env=env, capture_output=True, text=True,
                           timeout=120)
        key = "__control__" if row is None else row[0]
        out["child_exits"][key] = p.returncode
        if p.returncode != 0:
            if row is not None:
                out["results"][key] = {"child_crash": p.returncode, "stderr": p.stderr[-1500:]}
            continue
        res = json.loads(p.stdout.strip().splitlines()[-1])
        if row is None:
            out["control_calls"] = res["control_calls"]
            out["tensor_cuda_file"] = res["results"]["__tensor_cuda__"]
        else:
            out["results"][key] = res["results"][key]
    out["child_exit"] = out["child_exits"].get("__control__")
    return out


def verdict(row, r):
    """None when the row's expectation holds on result r, else the reason (used by the tests and by bwdf2_k1.py)."""
    rid, _f, kind, _s, _c, want, flags, _e = row
    if r is None:
        return "no result"
    if "setup_error" in r:
        return "setup error: " + r["setup_error"]
    if "child_crash" in r:
        return f"the process died (exit {r['child_crash']})"
    if kind == "refuse":
        exp = flags.get("raises", "RuntimeError")
        if r["raised"] != exp:
            return f"raised {r['raised']} (want {exp}): {r.get('message')}"
        for w in want:
            if w not in r["message"]:
                return f"message lacks {w!r}: {r['message']}"
        if r["calls"] != 0:
            return f"{r['calls']} CUDA calls before the refusal: {r['by_function']}"
        return None
    if kind == "accept":
        if r["raised"] is not None:
            return f"raised {r['raised']}: {r.get('message')}"
        if r["result"] != want:
            return f"result {r['result']} (want {want})"
        if r["calls"] != 0:
            return f"{r['calls']} CUDA calls: {r['by_function']}"
        return None
    if kind == "preflight":
        if r["raised"] != "RuntimeError" or not r["message"].startswith(want[0]) or \
                "no CUDA-capable device" not in r["message"]:
            return f"did not reach the device step: {r['raised']}: {r.get('message')}"
        if flags.get("no_alloc"):
            bad = {k: v for k, v in r["by_function"].items() if k in ("cudaMalloc", "cudaMallocAsync",
                                                                        "cudaLaunchKernel", "cudaMemcpy")}
            if bad:
                return f"allocated or launched: {bad}"
        return None
    return f"unknown kind {kind}"


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        pytest.skip("CPU lane only (CUDA_VISIBLE_DEVICES must be empty)")
    res = run_rows(PKG_PARENT, str(tmp_path_factory.mktemp("bwdf2_k1")))
    assert res["child_exit"] == 0, res
    assert res["control_calls"] > 0, res   # the counter sees CUDA calls in this process
    rec = os.environ.get("BWDF2_RECORD_DIR")
    if rec:
        os.makedirs(rec, exist_ok=True)
        with open(os.path.join(rec, "k1_rows.json"), "w") as f:
            json.dump({"rows": ROWS, **res}, f, indent=1)
    return res["results"]


@pytest.mark.parametrize("row", ROWS, ids=[r[0] for r in ROWS])
def test_cpu_k1_row(results, row):
    why = verdict(row, results.get(row[0]))
    assert why is None, (row[0], why)
