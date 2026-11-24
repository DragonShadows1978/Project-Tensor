"""BWD-F1 Amendment 1 (Part C, orders/BWDF1_AMENDMENT_1_COPY_OPS.md), G1 rows on the CPU (CUDA_VISIBLE_DEVICES empty).

One refusal row per C2 term per copy op, through the public API, each asserting RuntimeError, the op name, the operand
shapes in the message, and no CUDA call made by the op (ENG3's LD_PRELOAD counter, tests/support/cudart_count.c;
operands are built before the count starts; positive control: a device tensor with elements makes CUDA calls).
Operands are host tensors, or CUDA tensors with zero elements (built with no device), so every term is reachable
without a GPU; the terms that need a CUDA operand with elements (export_rows / export_rope_rows on a rank-0 or
rank-1 cache) are in the staged memcheck line. Accepted results are G4's (Build B against Build E, bitwise).
m05 (cat's extent check removed) fails the cat extent rows; m06 (the dtype check removed) fails the cat dtype row.
BWD-F2 (R1, R9, R12): four rows whose differing operand is empty moved to R1_ROWS (not refused: Build B's behaviour);
the later-pair rows make no ordering claim (the ordering witnesses are GPU rows in scripts/bwdf2_step_k2.py).

Environment: BWDF1_PKG_PARENT selects the tensor_cuda package under test (mutant runs).
Prior art: numpy.concatenate / numpy.stack rules (the reference for cat and stack); ENG3's call counter; DeMillo,
Lipton and Sayward (1978), mutation testing. Ours: the rows.
"""
import json
import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PKG_PARENT = os.environ.get("BWDF1_PKG_PARENT", os.path.dirname(HERE))
SHIM = os.path.join(HERE, "support", "cudart_count.c")

# (id, op, setup code defining the operands, call code, substrings the message must hold). H(shape, dt) is a host
# tensor of ones, C(shape) a CUDA tensor with zero elements, T a table pair.
ROWS = [
    # cat (cat_nd): list, dim range, rank, extents (Voice's two cases + a smaller one), dtype, device, dtype class
    ("cat-empty-list", "cat", "", "tc.cat([], 0)", ["cat: empty input"]),
    ("cat-rank0", "cat", "a=H((1,)).reshape([]); b=H((1,)).reshape([])", "tc.cat([a, b], 0)", ["cat:", "operand 0 has shape []"]),
    ("cat-dim-high", "cat", "a=H((2,3)); b=H((2,3))", "tc.cat([a, b], 2)", ["cat: dim 2 out of range", "[2, 3]"]),
    ("cat-dim-low", "cat", "a=H((2,3)); b=H((2,3))", "tc.cat([a, b], -3)", ["cat: dim -3 out of range", "[2, 3]"]),
    ("cat-rank", "cat", "a=H((2,3)); b=H((2,3,1))", "tc.cat([a, b], 0)",
     ["cat: operand 1 has rank 3", "[2, 3, 1]", "[2, 3]"]),
    ("cat-extent-voice-zero-first", "cat", "a=H((1,512,0)); b=H((1,1,1))", "tc.cat([a, b], 1)",
     ["cat: operand 1 has extent 1 at dim 2", "[1, 1, 1]", "[1, 512, 0]"]),
    ("cat-extent-voice-longer-later", "cat", "a=H((1,3,5)); b=H((1,2,6)); c=H((1,4,6))", "tc.cat([a, b, c], 1)",
     ["cat: operand 1 has extent 6 at dim 2", "[1, 2, 6]", "[1, 3, 5]"]),
    ("cat-extent-shorter-later", "cat", "a=H((4,3)); b=H((2,2))", "tc.cat([a, b], 0)",
     ["cat: operand 1 has extent 2 at dim 1", "[2, 2]", "[4, 3]"]),
    ("cat-dtype", "cat", "a=H((2,3)); b=H((2,3),'float16')", "tc.cat([a, b], 0)",
     ["cat: operand 1 has dtype float16", "[2, 3]"]),
    ("cat-dtype-class", "cat", "a=H((2,3),'uint8'); b=H((2,3),'uint8')", "tc.cat([a, b], 0)",
     ["cat: op supports float32/float16/bfloat16 only", "[2, 3]"]),
    # stack (ops::stack): list, dim range, rank, shape, dtype, device
    ("stack-empty-list", "stack", "", "tc.stack([], 0)", ["stack: empty input"]),
    ("stack-dim-high", "stack", "a=H((2,3)); b=H((2,3))", "tc.stack([a, b], 3)", ["stack: dim 3 out of range", "[2, 3]"]),
    ("stack-dim-low", "stack", "a=H((2,3)); b=H((2,3))", "tc.stack([a, b], -4)", ["stack: dim -4 out of range", "[2, 3]"]),
    ("stack-rank", "stack", "a=H((2,3)); b=H((2,3,1))", "tc.stack([a, b], 0)", ["stack: operand 1 has rank 3", "[2, 3, 1]"]),
    ("stack-shape", "stack", "a=H((2,3)); b=H((2,4))", "tc.stack([a, b], 1)",
     ["stack: operand 1 has extent 4 at dim 1", "every shape must be equal", "[2, 4]", "[2, 3]"]),
    ("stack-dtype", "stack", "a=H((2,3)); b=H((2,3),'bfloat16')", "tc.stack([a, b], 0)",
     ["stack: operand 1 has dtype bfloat16", "[2, 3]"]),
    ("slice-dim-high", "slice", "a=H((4,3))", "a.slice(2, 0, 1)", ["slice: dim 2 out of range", "[4, 3]"]),
    ("slice-dim-low", "slice", "a=H((4,3))", "a.slice(-3, 0, 1)", ["slice: dim -3 out of range", "[4, 3]"]),
    ("slice-start-negative", "slice", "a=H((4,3))", "a.slice(1, -1, 1)", ["slice: start -1 and length 1", "[4, 3]"]),
    ("slice-length-negative", "slice", "a=H((4,3))", "a.slice(0, 1, -1)", ["slice: start 1 and length -1", "[4, 3]"]),
    ("slice-past-end", "slice", "a=H((4,3))", "a.slice(0, 3, 2)", ["slice: start 3 and length 2", "extent 4", "[4, 3]"]),
    ("slice-start-past-extent", "slice", "a=H((4,3))", "a.slice(0, 5, 0)", ["slice: start 5 and length 0", "[4, 3]"]),
    # write_rows: dtype, rank, device, last dim, leading extents, capacity, start, CUDA, zero CAP/D (B divided by 0)
    ("write_rows-dtype", "write_rows", "b=C((0,4,2)); s=C((0,1,2),'float16')", "tc.write_rows(b, s, 0)",
     ["write_rows: dtype mismatch", "[0, 4, 2]", "[0, 1, 2]"]),
    ("write_rows-rank", "write_rows", "b=C((0,)); s=C((0,2))", "tc.write_rows(b, s, 0)",
     ["write_rows: need >=2D buf and src", "buf [0] ", "[0, 2]"]),
    ("write_rows-last-dim", "write_rows", "b=C((0,4,2)); s=C((0,1,3))", "tc.write_rows(b, s, 0)",
     ["write_rows: last-dim mismatch", "[0, 1, 3]"]),
    ("write_rows-leading", "write_rows", "b=C((0,4,2)); s=C((2,0,2))", "tc.write_rows(b, s, 0)",
     ["write_rows: leading-dim mismatch", "[0, 4, 2]", "[2, 0, 2]"]),
    ("write_rows-capacity", "write_rows", "b=C((0,2,2)); s=C((0,3,2))", "tc.write_rows(b, s, 0)",
     ["write_rows: src rows exceed capacity", "[0, 3, 2]"]),
    ("write_rows-start-negative", "write_rows", "b=C((0,4,2)); s=C((0,1,2))", "tc.write_rows(b, s, -1)",
     ["write_rows: start -1 is negative", "[0, 4, 2]"]),
    ("write_rows-host-with-elements", "write_rows", "b=H((1,4,2)); s=H((1,1,2))", "tc.write_rows(b, s, 0)",
     ["write_rows: CUDA tensors required", "[1, 4, 2]"]),
    ("write_rows-zero-capacity", "write_rows", "b=H((0,2)); s=H((1,2))", "tc.write_rows(b, s, 0)",
     ["write_rows: src rows exceed capacity", "[0, 2]", "[1, 2]"]),
    # export_rows: CUDA, dim range, span (start, length, end)
    ("export_rows-host", "export_rows", "c=H((1,4,2))", "tc.export_rows(c, 1, 0, 1)",
     ["export_rows: CUDA tensor required", "[1, 4, 2]"]),
    ("export_rows-dim", "export_rows", "c=C((0,4,2))", "tc.export_rows(c, 3, 0, 1)", ["export_rows: dim out of range"]),
    ("export_rows-start-negative", "export_rows", "c=C((0,4,2))", "tc.export_rows(c, 1, -1, 1)",
     ["export_rows: invalid span", "[0, 4, 2]"]),
    ("export_rows-length-negative", "export_rows", "c=C((0,4,2))", "tc.export_rows(c, 1, 0, -1)",
     ["export_rows: invalid span", "[0, 4, 2]"]),
    ("export_rows-past-end", "export_rows", "c=C((0,4,2))", "tc.export_rows(c, 1, 3, 2)",
     ["export_rows: invalid span", "[0, 4, 2]"]),
    # splice_rows: dtype, device, CUDA, rank, extents outside dim, dim range, head/tail
    ("splice_rows-dtype", "splice_rows", "o=C((0,4,2)); i=C((0,1,2),'float16')", "tc.splice_rows(o, i, 1, 0, 1)",
     ["splice_rows: dtype mismatch", "[0, 4, 2]", "[0, 1, 2]"]),
    ("splice_rows-device", "splice_rows", "o=C((0,4,2)); i=H((0,1,2))", "tc.splice_rows(o, i, 1, 0, 1)",
     ["splice_rows: device mismatch", "[0, 4, 2]", "[0, 1, 2]"]),
    ("splice_rows-host", "splice_rows", "o=H((1,4,2)); i=H((1,1,2))", "tc.splice_rows(o, i, 1, 0, 1)",
     ["splice_rows: CUDA tensors required", "[1, 4, 2]", "[1, 1, 2]"]),
    ("splice_rows-rank", "splice_rows", "o=C((0,4,2)); i=C((0,1))", "tc.splice_rows(o, i, 1, 0, 1)",
     ["splice_rows: rank mismatch", "[0, 4, 2]", "[0, 1]"]),
    ("splice_rows-extent", "splice_rows", "o=C((0,4,2)); i=C((0,1,3))", "tc.splice_rows(o, i, 1, 0, 1)",
     ["splice_rows: shape mismatch outside splice dim", "[0, 4, 2]", "[0, 1, 3]"]),
    ("splice_rows-dim", "splice_rows", "o=C((0,4,2)); i=C((0,1,2))", "tc.splice_rows(o, i, 3, 0, 1)",
     ["splice_rows: dim out of range"]),
    ("splice_rows-head-negative", "splice_rows", "o=C((0,4,2)); i=C((0,1,2))", "tc.splice_rows(o, i, 1, -1, 1)",
     ["splice_rows: invalid head/tail", "[0, 4, 2]"]),
    ("splice_rows-tail-before-head", "splice_rows", "o=C((0,4,2)); i=C((0,1,2))", "tc.splice_rows(o, i, 1, 2, 1)",
     ["splice_rows: invalid head/tail", "[0, 4, 2]"]),
    ("splice_rows-tail-past-end", "splice_rows", "o=C((0,4,2)); i=C((0,1,2))", "tc.splice_rows(o, i, 1, 0, 5)",
     ["splice_rows: invalid head/tail", "[0, 4, 2]"]),
    # evict_rows: CUDA, dim, head, drop
    ("evict_rows-host", "evict_rows", "o=H((1,4,2))", "tc.evict_rows(o, 1, 0, 1)",
     ["evict_rows: CUDA tensor required", "[1, 4, 2]"]),
    ("evict_rows-dim", "evict_rows", "o=C((0,4,2))", "tc.evict_rows(o, -4, 0, 1)", ["evict_rows: dim out of range"]),
    ("evict_rows-head-negative", "evict_rows", "o=C((0,4,2))", "tc.evict_rows(o, 1, -1, 1)",
     ["evict_rows: drop exceeds live tail", "[0, 4, 2]"]),
    ("evict_rows-drop-negative", "evict_rows", "o=C((0,4,2))", "tc.evict_rows(o, 1, 0, -1)",
     ["evict_rows: drop exceeds live tail", "[0, 4, 2]"]),
    ("evict_rows-head-past-end", "evict_rows", "o=C((0,4,2))", "tc.evict_rows(o, 1, 5, 0)",
     ["evict_rows: drop exceeds live tail", "[0, 4, 2]"]),
    ("evict_rows-drop-past-tail", "evict_rows", "o=C((0,4,2))", "tc.evict_rows(o, 1, 3, 2)",
     ["evict_rows: drop exceeds live tail", "[0, 4, 2]"]),
    # export_rope_rows: CUDA, rank, seq dim, odd D, span, table dtype, table shapes (cs, sn rank), table device,
    # positions (pos0 < 0, past cos, past sin)
    ("export_rope_rows-host", "export_rope_rows", "c=H((1,4,2)); cs=H((8,2)); sn=H((8,2))",
     "tc.export_rope_rows(c, cs, sn, 1, 0, 1, 0)", ["export_rope_rows: CUDA tensor required", "[1, 4, 2]"]),
    ("export_rope_rows-seq-dim", "export_rope_rows", "c=C((0,4,2)); cs=C((0,2)); sn=C((0,2))",
     "tc.export_rope_rows(c, cs, sn, 0, 0, 1, 0)", ["export_rope_rows: seq dim must be -2", "[0, 4, 2]"]),
    ("export_rope_rows-odd-D", "export_rope_rows", "c=C((0,4,3)); cs=C((0,3)); sn=C((0,3))",
     "tc.export_rope_rows(c, cs, sn, 1, 0, 1, 0)", ["export_rope_rows: odd head_dim", "[0, 4, 3]"]),
    ("export_rope_rows-span", "export_rope_rows", "c=C((0,4,2)); cs=C((0,2)); sn=C((0,2))",
     "tc.export_rope_rows(c, cs, sn, 1, 3, 2, 0)", ["export_rope_rows: invalid span", "[0, 4, 2]"]),
    ("export_rope_rows-table-dtype", "export_rope_rows", "c=C((0,4,2)); cs=C((0,2),'float16'); sn=C((0,2))",
     "tc.export_rope_rows(c, cs, sn, 1, 0, 0, 0)", ["export_rope_rows: table/cache dtype mismatch", "[0, 2]"]),
    ("export_rope_rows-cos-shape", "export_rope_rows", "c=C((0,4,2)); cs=C((0,3)); sn=C((0,2))",
     "tc.export_rope_rows(c, cs, sn, 1, 0, 0, 0)", ["export_rope_rows: tables must be (T, D)", "[0, 3]"]),
    ("export_rope_rows-sin-rank", "export_rope_rows", "c=C((0,4,2)); cs=C((0,2)); sn=C((0,))",
     "tc.export_rope_rows(c, cs, sn, 1, 0, 0, 0)", ["export_rope_rows: tables must be (T, D)", "[0]"]),
    ("export_rope_rows-pos-negative", "export_rope_rows", "c=C((0,4,2)); cs=C((0,2)); sn=C((0,2))",
     "tc.export_rope_rows(c, cs, sn, 1, 0, 0, -1)", ["export_rope_rows: position past table end", "pos0 -1"]),
    ("export_rope_rows-past-cos", "export_rope_rows", "c=C((0,4,2)); cs=C((0,2)); sn=C((0,2))",
     "tc.export_rope_rows(c, cs, sn, 1, 0, 1, 0)", ["export_rope_rows: position past table end", "[0, 2]"]),
    # the batched pair ops: pair 1 is the bad one. BWD-F2 R9 (review BWDF1R1-F06): these are refusal rows only; they
    # make no ordering claim. Pair 0 here has no elements, so Build B's serial loop also made no CUDA call before
    # refusing pair 1, and on the CPU no pair can be built that would allocate or launch (a CUDA operand with
    # elements needs a device). The ordering witnesses (a first pair that allocates and launches, a bad later pair,
    # 0 CUDA calls; Build B's serial loop as the counter-control) are GPU rows: scripts/bwdf2_step_k2.py (count line).
    ("export_row_pairs-pair1", "export_row_pairs",
     "r0=C((0,4,2)); p0=C((0,4,2)); r1=C((0,4,2)); p1=C((0,4,2)); cs=C((0,2)); sn=C((0,2))",
     "tc.export_row_pairs([r0, r1], [p0, p1], cs, sn, 1, 1, [0, 5], [0, 0], 0, 0)",
     ["export_rows: invalid span", "[0, 4, 2]"]),
    ("swap_row_pairs-pair1-extent", "swap_row_pairs_with_rope",
     "r0=C((0,4,2)); p0=C((1,4,0)); r1=C((0,4,2)); p1=C((1,4,0)); i0=C((0,1,2)); j0=C((1,1,0)); "
     "i1=C((0,1,3)); j1=C((1,1,0)); cs=C((1,0)); sn=C((1,0))",
     "tc.swap_row_pairs_with_rope([r0, r1], [p0, p1], [i0, i1], [j0, j1], cs, sn, 1, 1, 0, 1, 0)",
     ["splice_rows: shape mismatch outside splice dim", "[0, 1, 3]"]),
    ("evict_row_pairs-pair1", "evict_row_pairs", "r0=C((0,4,2)); p0=C((0,4,2)); r1=C((0,4,2)); p1=C((0,2,2))",
     "tc.evict_row_pairs([r0, r1], [p0, p1], 1, 1, 1, 2)", ["evict_rows: drop exceeds live tail", "[0, 2, 2]"]),
    ("arena_row_pair_transaction-pair1", "arena_row_pair_transaction",
     "r0=C((0,4,2)); p0=C((1,4,0)); r1=C((0,4,2)); p1=C((1,4,0)); i0=C((0,1,2)); j0=C((1,1,0)); "
     "i1=C((0,1,2)); j1=C((1,1,0),'float16'); cs=C((1,0)); sn=C((1,0))",
     "tc.arena_row_pair_transaction([r0, r1], [p0, p1], [i0, i1], [j0, j1], cs, sn, 1, 1, 0, 1, 4)",
     ["rope_apply: table/x dtype mismatch"]),
]
GRAD_FREE = {"write_rows", "export_rows", "splice_rows", "evict_rows", "export_rope_rows", "export_row_pairs",
             "swap_row_pairs_with_rope", "evict_row_pairs", "arena_row_pair_transaction"}

_CHILD = r'''
import ctypes, json, os, sys
sys.path.insert(0, os.environ["BWDF1_PKG_PARENT"])
import numpy as np
import tensor_cuda as tc
L = ctypes.CDLL(None)
L.eng3_cudart_calls.restype = ctypes.c_ulong
rows, grad_free = json.loads(os.environ["BWDF1_ROWS"]), set(json.loads(os.environ["BWDF1_GRAD_FREE"]))
def H(shape, dt="float32"):
    return tc.tensor(np.ones(shape, np.float32), device="cpu", dtype=dt) if dt != "uint8" else \
        tc.tensor(np.ones(shape, np.uint8), device="cpu", dtype=dt)
def C(shape, dt="float32"):
    assert int(np.prod(shape)) == 0, shape
    return tc.tensor(np.zeros(shape, np.float32), device="cuda", dtype=dt)
out = {}
for rid, op, setup, call, _ in rows:
    env = {"tc": tc, "H": H, "C": C}
    try:
        exec(setup, env)
    except BaseException as e:  # noqa: BLE001
        out[rid] = {"setup_error": f"{type(e).__name__}: {e}"}
        continue
    before = L.eng3_cudart_calls()
    try:
        if op in grad_free:
            with tc.no_grad():
                eval(call, env)
        else:
            eval(call, env)
        r = {"raised": None}
    except BaseException as e:  # noqa: BLE001
        r = {"raised": type(e).__name__, "message": str(e)}
    r["calls"] = L.eng3_cudart_calls() - before
    out[rid] = r
base = L.eng3_cudart_calls()
try:
    tc.tensor([1.0], device="cuda")
except RuntimeError:
    pass
print(json.dumps({"results": out, "control_calls": L.eng3_cudart_calls() - base}))
'''


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    compiler = next((p for n in ("cc", "gcc", "clang") if (p := shutil.which(n))), None)
    if compiler is None:
        pytest.skip("no C compiler for the CUDA-call counter shim")
    d = tmp_path_factory.mktemp("bwdf1_copy")
    shim = d / "cudart_count.so"
    r = subprocess.run([compiler, "-shared", "-fPIC", "-O2", "-o", str(shim), SHIM, "-ldl"], capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_PRELOAD=str(shim), BWDF1_PKG_PARENT=PKG_PARENT,
               BWDF1_ROWS=json.dumps(ROWS), BWDF1_GRAD_FREE=json.dumps(sorted(GRAD_FREE)))
    p = subprocess.run([sys.executable, "-B", "-P", "-c", _CHILD], env=env, capture_output=True, text=True,
                       timeout=300)
    assert p.returncode == 0, p.stderr[-3000:]
    res = json.loads(p.stdout.strip().splitlines()[-1])
    assert res["control_calls"] > 0, res
    rec = os.environ.get("BWDF1_RECORD_DIR")
    if rec:
        os.makedirs(rec, exist_ok=True)
        with open(os.path.join(rec, "g1_copy_ops.json"), "w") as f:
            json.dump({"rows": ROWS, **res}, f, indent=1)
    return res["results"]


# BWD-F2 R1 / R12 (artifacts/bwdf1/f2/INTENDED_CHANGES_F2.json): four BWD-F1 refusal rows whose differing operand
# has no elements. R1 checks dtype and device only where bytes are read, so each now gets Build B's behaviour:
# (id, op, setup, call, expected): "preflight" = no host refusal, the call reaches cat's closing CUDA status check
# (on the CPU: "CUDA error at cat: no CUDA-capable device") with no allocation and no launch; a tuple = the shape
# returned with 0 CUDA calls; None = returns None with 0 CUDA calls. Their GPU values are rows in bwdf2_step_k2.py.
R1_ROWS = [
    ("cat-device", "cat", "a=H((0,3)); b=C((0,3))", "tc.cat([a, b], 0)", "preflight"),
    ("stack-device", "stack", "a=C((0,3)); b=H((0,3))", "tc.stack([a, b], 0)", "preflight"),
    ("write_rows-device", "write_rows", "b=C((0,4,2)); s=H((0,1,2))", "tc.write_rows(b, s, 0)", None),
    ("export_rope_rows-table-device", "export_rope_rows", "c=C((0,4,2)); cs=H((0,2)); sn=C((0,2))",
     "tc.export_rope_rows(c, cs, sn, 1, 0, 0, 0)", (0, 0, 2)),
]


@pytest.fixture(scope="module")
def r1_results(tmp_path_factory):
    import test_bwdf2_fixes as T2  # the BWD-F2 K1 harness (one child per row, per-function counts)
    rows = [(rid, "R1", "preflight" if want == "preflight" else "accept", setup, call,
             ["CUDA error at cat"] if want == "preflight" else (None if want is None else list(want)),
             {"no_alloc": 1} if want == "preflight" else {"no_grad": 1}, "refuses (BWD-F1)")
            for rid, op, setup, call, want in R1_ROWS]
    res = T2.run_rows(PKG_PARENT, str(tmp_path_factory.mktemp("bwdf1_copy_r1")), rows)
    assert res["control_calls"] > 0, res
    return {r[0]: (r, res["results"].get(r[0])) for r in rows}


@pytest.mark.parametrize("rid", [r[0] for r in R1_ROWS])
def test_cpu_r1_no_op_member_not_refused(r1_results, rid):
    import test_bwdf2_fixes as T2
    row, r = r1_results[rid]
    why = T2.verdict(row, r)
    assert why is None, (rid, why)


@pytest.mark.parametrize("row", ROWS, ids=[r[0] for r in ROWS])
def test_cpu_c3_copy_op_refusal_no_cuda_call(results, row):
    rid, op, _, _, wants = row
    r = results[rid]
    assert "setup_error" not in r, r
    assert r["raised"] == "RuntimeError", r
    for w in wants:
        assert w in r["message"], (w, r["message"])
    assert r["calls"] == 0, r   # refused on the host before any CUDA call
