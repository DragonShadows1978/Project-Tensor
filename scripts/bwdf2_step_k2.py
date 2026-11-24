import subprocess
if subprocess.run(["forge-gpu-lease-check", "--require-held"]).returncode != 0: raise SystemExit("not inside a held forge-gpu-run lease")
# BWD-F2 K2 (staged GPU line): the review's cases that only the GPU can show, one mode per line so a memory fault in
# one hazardous case cannot hide the others. Usage: python3 -B -P scripts/bwdf2_step_k2.py <mode> <arm>; writes only
# artifacts/bwdf1/f2/gpu/k2_<mode>_<arm>/ (refuses to start if it exists). Modes:
#   f01   (memcheck) RoPE: the review's swap case (inserts [2, 2], cos [2, 2], sin [1, 2], pos0 0), rope_apply with a
#         short sin, a negative pos0; control: covering tables (values kept for B vs F2 bitwise).
#   f04   (memcheck) write_rows ring index: buf [3, 1] / src [2, 1] at start 1 (control), CAP + 2 and INT64_MAX, and
#         buf [2, 3, 4] / src [2, 2, 4] at INT64_MAX; buf must equal NumPy's ring write at (start + l) % CAP.
#   f07   (memcheck) where / masked_fill condition dtype: a float condition of the data dtype (control, bitwise B vs
#         F2), then bool conditions [False, True, True, True] (and a [4, 1] bool under [4, 3] float16 data).
#   vals  (memcheck) F12 axpy_ values, F03 / R9 copy results (zero-size cat/stack controls, the named cat case), R1's
#         empty write_rows / export_rope_rows, F02 rank 9 on a CUDA operand with elements (last: Build E may fault).
#   count (no memcheck; ENG3's LD_PRELOAD counter in a child) R9 ordering: a first pair that allocates and launches,
#         a bad later pair, CUDA calls during the call; F05's singular pair and uint8 batch; F09's uint8 slice.
# Every row records its result; the verdicts are scripts/bwdf2_gpu_post.py's (CPU, after the line).
# Prior art: NumPy's where / putmask / concatenate rules and its modular ring arithmetic as the references; the RoPE
# formulation (Su et al., 2021, "RoFormer"); ENG3's call counter (ltrace-style interposition). Ours: the cases.
import json
import os
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bwdf2_gpu_common as G  # noqa: E402

mode, arm = sys.argv[1], sys.argv[2]
assert mode in ("f01", "f04", "f07", "vals", "count"), mode
OUT = G.GPU / f"k2_{mode}_{arm}"
if OUT.exists():
    raise SystemExit(f"{OUT} exists; move it aside before a retry")
OUT.mkdir(parents=True)
import numpy as np  # noqa: E402

INT64_MAX = 2 ** 63 - 1
rows, arrays = [], {}


def record(rid, fn):
    row = {"id": rid}
    try:
        r = fn()
        tc.synchronize()
        row["result"] = "RETURNED"
        if isinstance(r, dict):
            for k, v in r.items():
                arrays[f"{rid}|{k}"] = v
                row.setdefault("arrays", {})[k] = {"shape": list(v.shape), "dtype": str(v.dtype)}
        elif r is not None:
            row["value"] = r
    except Exception as e:  # noqa: BLE001
        row["result"] = "RAISED"
        row["error"] = f"{type(e).__name__}: {e}"
    rows.append(row)
    print(json.dumps(row, default=str), flush=True)


if mode != "count":
    tc, sha = G.load_arm(arm)

    def T(shape, dt="float32", salt=0):
        return tc.tensor(G.data(shape, dt, salt), device="cuda", dtype=dt)

    def Bool(vals):
        return tc.tensor(np.array(vals, bool), device="cuda", dtype="bool")

    def rb(t):
        return np.ascontiguousarray(t.astype("float32").numpy())

if mode == "f01":
    def rope_ok():
        x, cs, sn = T((3, 2, 4)), T((6, 4), salt=1), T((6, 4), salt=2)
        return {"out": rb(tc.rope_apply(x, cs, sn, 1)), "x": rb(x), "cos": rb(cs), "sin": rb(sn)}
    record("f01-control-covering-tables", rope_ok)

    def swap_named():
        with tc.no_grad():
            r = tc.swap_row_pairs_with_rope([T((4, 2), salt=3)], [T((4, 2), salt=4)], [T((2, 2), salt=5)],
                                            [T((2, 2), salt=6)], T((2, 2), salt=7), T((1, 2), salt=8), 0, 0, 0, 0, 0)
        return {"raw": rb(r[0][0]), "rope": rb(r[1][0])}
    record("f01-swap-named-case", swap_named)
    record("f01-rope_apply-short-sin", lambda: {"out": rb(tc.rope_apply(T((2, 2), salt=9), T((2, 2), salt=10),
                                                                          T((1, 2), salt=11), 0))})
    record("f01-rope_apply-negative-pos0", lambda: {"out": rb(tc.rope_apply(T((1, 2), salt=12), T((4, 2), salt=13),
                                                                              T((4, 2), salt=14), -1))})
elif mode == "f04":
    def ring(buf_shape, src_shape, start, salt):
        buf, src = T(buf_shape, salt=salt), T(src_shape, salt=salt + 1)
        b0, s0 = rb(buf), rb(src)
        with tc.no_grad():
            tc.write_rows(buf, src, start)
        cap, L = buf_shape[-2], src_shape[-2]
        want = b0.copy()
        for l in range(L):
            want[..., (start + l) % cap, :] = s0[..., l, :]   # Python integers: no overflow
        return {"buf": rb(buf), "want": want, "src": s0, "before": b0}
    record("f04-control-start-1", lambda: ring((3, 1), (2, 1), 1, 20))
    record("f04-start-cap-plus-2", lambda: ring((3, 1), (2, 1), 5, 22))
    record("f04-start-int64max", lambda: ring((3, 1), (2, 1), INT64_MAX, 24))
    record("f04-start-int64max-batched", lambda: ring((2, 3, 4), (2, 2, 4), INT64_MAX, 26))
elif mode == "f07":
    cond_vals = [False, True, True, True]

    def where_float_cond():
        c = tc.tensor(np.array(cond_vals, np.float32), device="cuda")
        x, y = T((4,), salt=30), T((4,), salt=31)
        return {"out": rb(tc.where(c, x, y)), "want": np.where(np.array(cond_vals), rb(x), rb(y))}
    record("f07-control-float-cond", where_float_cond)

    def where_bool():
        x, y = T((4,), salt=32), T((4,), salt=33)
        return {"out": rb(tc.where(Bool(cond_vals), x, y)), "want": np.where(np.array(cond_vals), rb(x), rb(y))}
    record("f07-where-bool-cond", where_bool)

    def masked_bool():
        a = T((4,), salt=34)
        return {"out": rb(a.masked_fill(Bool(cond_vals), 2.0)), "want": np.where(np.array(cond_vals), 2.0,
                                                                                rb(a)).astype(np.float32)}
    record("f07-masked_fill-bool-mask", masked_bool)

    def where_bool_bcast_f16():
        cv = np.array([[True], [False], [True], [False]])
        x, y = T((4, 3), "float16", 35), T((4, 3), "float16", 36)
        c = tc.tensor(cv, device="cuda", dtype="bool")
        return {"out": rb(tc.where(c, x, y)), "want": np.where(cv, rb(x), rb(y))}
    record("f07-where-bool-cond-broadcast-f16", where_bool_bcast_f16)
elif mode == "vals":
    def axpy(pshape, oshape, salt):
        p, o = T(pshape, salt=salt), T(oshape, salt=salt + 1)
        p0, o0 = rb(p), rb(o)
        tc._C.axpy_(p, o, 1.0)
        want = (p0 + o0).reshape(p0.shape) if (p0 + o0).size == p0.size else None
        return {"param": rb(p), "want": want if want is not None else np.zeros(0, np.float32), "before": p0,
                "other": o0}
    record("f12-axpy-named-case", lambda: axpy((2, 1), (2, 2), 40))
    record("f12-axpy-leading-one-other", lambda: axpy((3,), (1, 3), 42))
    record("f12-axpy-equal-control", lambda: axpy((2, 3), (2, 3), 44))

    def cat_named():
        a = T((2, 3), salt=46)
        out = tc.cat([a, tc.tensor(np.zeros((0, 3), np.float32), device="cuda", dtype="float16")], 0)
        return {"out": rb(out), "want": rb(a), "meta": np.array([out.dtype == "float32"], np.float32)}
    record("f03-cat-named-case", cat_named)

    def empty_ops(kind):
        z32 = tc.tensor(np.zeros((0, 3), np.float32), device="cuda")
        if kind == "cat-mixed-dtype":
            o = tc.cat([z32, tc.tensor(np.zeros((0, 3), np.float32), device="cuda", dtype="float16")], 0)
        elif kind == "stack-mixed-dtype":
            o = tc.stack([z32, tc.tensor(np.zeros((0, 3), np.float32), device="cuda", dtype="bfloat16")], 0)
        elif kind == "cat-host-later":
            a = T((2, 3), salt=48)
            o = tc.cat([a, tc.tensor(np.zeros((0, 3), np.float32), device="cpu")], 0)
            return {"out": rb(o), "want": rb(a)}
        else:
            o = tc.stack([z32, tc.tensor(np.zeros((0, 3), np.float32), device="cpu")], 0)
        return {"shape": np.array(o.shape, np.int64).astype(np.float32),
                "meta": np.array([o.dtype == "float32", str(o.device).startswith("cuda")], np.float32)}
    for k in ("cat-mixed-dtype", "stack-mixed-dtype", "cat-host-later", "stack-host-later"):
        record("r9-zero-size-" + k, lambda k=k: empty_ops(k))

    def r1_write_rows():
        with tc.no_grad():
            r = tc.write_rows(T((0, 4, 2)), tc.tensor(np.zeros((0, 1, 2), np.float32), device="cpu"), 0)
        return {"returned_none": np.array([r is None], np.float32)}
    record("r1-write_rows-empty-host-src", r1_write_rows)

    def r1_export_rope():
        with tc.no_grad():
            r = tc.export_rope_rows(tc.tensor(np.zeros((0, 4, 2), np.float32), device="cuda"),
                                    tc.tensor(np.zeros((0, 2), np.float32), device="cpu"),
                                    tc.tensor(np.zeros((0, 2), np.float32), device="cuda"), 1, 0, 0, 0)
        return {"shape": np.array(r.shape, np.float32)}
    record("r1-export_rope_rows-empty-host-cos", r1_export_rope)
    record("f02-rank9-cuda-with-elements", lambda: {"out": rb(T((1,)).reshape([1] * 9) + T((1,), salt=1).reshape(
        [1] * 9))})
else:  # count
    G.check_arm("tc", arm)
    shim = OUT / "cudart_count.so"
    cc = shutil.which("cc") or shutil.which("gcc")
    r = subprocess.run([cc, "-shared", "-fPIC", "-O2", "-o", str(shim),
                        str(G.ROOT / "tensor_cuda/tests/support/cudart_count.c"), "-ldl"], capture_output=True,
                       text=True, timeout=120)
    if r.returncode != 0:
        raise SystemExit("shim build failed: " + r.stderr[-1000:])
    CHILD = r'''
import subprocess
if subprocess.run(["forge-gpu-lease-check", "--require-held"]).returncode != 0: raise SystemExit("not inside a held forge-gpu-run lease")
import ctypes, json, sys
sys.path.insert(0, sys.argv[1])
import bwdf2_gpu_common as G
tc, sha = G.load_arm(sys.argv[2])
import numpy as np
L = ctypes.CDLL(None)
L.eng3_cudart_calls.restype = ctypes.c_ulong
L.eng3_cudart_count.restype = ctypes.c_ulong
L.eng3_cudart_name.restype = ctypes.c_char_p
names = [L.eng3_cudart_name(i).decode() for i in range(L.eng3_cudart_n())]
def T(shape, dt="float32", salt=0):
    return tc.tensor(G.data(shape, dt, salt) if dt != "uint8" else np.ones(shape, np.uint8), device="cuda", dtype=dt)
def counts():
    return {n: L.eng3_cudart_count(i) for i, n in enumerate(names)}
def F(shape, salt=0):
    return T(shape, "float32", salt)
def U(shape):
    return T(shape, "uint8")
c4 = lambda: [F((1, 4, 2), 0), F((1, 4, 2), 1)]
r4 = lambda: [F((1, 4, 2), 2), F((1, 4, 2), 3)]
tb = lambda: (F((8, 2), 8), F((8, 2), 9))
# (id, operand builder (uncounted), call on the operands (counted))
CASES = [
 ("control-export_row_pairs-all-valid", lambda: (c4(), r4(), *tb()),
  lambda o: tc.export_row_pairs(o[0], o[1], o[2], o[3], 1, 1, [0, 1], [0, 1], 2, 0)),
 ("f06-export_row_pairs-later-bad", lambda: (c4(), r4(), *tb()),
  lambda o: tc.export_row_pairs(o[0], o[1], o[2], o[3], 1, 1, [0, 5], [0, 0], 2, 0)),
 ("control-swap-all-valid", lambda: (c4(), r4(), [F((1, 1, 2), 4), F((1, 1, 2), 5)], [F((1, 1, 2), 6), F((1, 1, 2), 7)], *tb()),
  lambda o: tc.swap_row_pairs_with_rope(o[0], o[1], o[2], o[3], o[4], o[5], 1, 1, 0, 1, 0)),
 ("f06-swap-later-bad", lambda: (c4(), r4(), [F((1, 1, 2), 4), F((1, 1, 3), 5)], [F((1, 1, 2), 6), F((1, 1, 2), 7)], *tb()),
  lambda o: tc.swap_row_pairs_with_rope(o[0], o[1], o[2], o[3], o[4], o[5], 1, 1, 0, 1, 0)),
 ("control-evict-all-valid", lambda: (c4(), r4()), lambda o: tc.evict_row_pairs(o[0], o[1], 1, 1, 1, 2)),
 ("f06-evict-later-bad", lambda: (c4(), [F((1, 4, 2), 2), F((1, 2, 2), 3)]),
  lambda o: tc.evict_row_pairs(o[0], o[1], 1, 1, 1, 2)),
 ("control-arena-all-valid", lambda: (c4(), r4(), [F((1, 1, 2), 4), F((1, 1, 2), 5)], [F((1, 1, 2), 6), F((1, 1, 2), 7)], *tb()),
  lambda o: tc.arena_row_pair_transaction(o[0], o[1], o[2], o[3], o[4], o[5], 1, 1, 0, 1, 4)),
 ("f06-arena-later-bad", lambda: (c4(), r4(), [F((1, 1, 2), 4), F((1, 1, 2), 5)], [F((1, 1, 2), 6), T((1, 1, 2), "float16", 7)], *tb()),
  lambda o: tc.arena_row_pair_transaction(o[0], o[1], o[2], o[3], o[4], o[5], 1, 1, 0, 1, 4)),
 ("control-export_row_pair-valid", lambda: (F((1, 4, 2)), F((1, 4, 2), 1), F((8, 2), 2), F((8, 2), 3)),
  lambda o: tc.export_row_pair(o[0], o[1], o[2], o[3], 1, 1, 0, 0, 2, 0)),
 ("f05-export_row_pair-rope-bad", lambda: (F((1, 4, 2)), F((1, 4, 2), 1), F((1, 2), 2), F((1, 2), 3)),
  lambda o: tc.export_row_pair(o[0], o[1], o[2], o[3], 1, 1, 0, 0, 2, 0)),
 ("f05-swap-uint8-late-dispatch", lambda: ([U((1, 4, 2)), U((1, 4, 2))], [U((1, 4, 2)), U((1, 4, 2))], [U((1, 1, 2)), U((1, 1, 2))],
   [tc.tensor(np.zeros((1, 0, 2), np.uint8), device="cuda", dtype="uint8"), U((1, 1, 2))], U((8, 2)), U((8, 2))),
  lambda o: tc.swap_row_pairs_with_rope(o[0], o[1], o[2], o[3], o[4], o[5], 1, 1, 0, 1, 0)),
 ("control-slice-float32", lambda: (F((2, 2)),), lambda o: o[0].slice(0, 0, 1)),
 ("f09-slice-uint8", lambda: (U((2, 2)),), lambda o: o[0].slice(0, 0, 1)),
]
rows = []
for cid, build, fn in CASES:
    row = {"id": cid}
    with tc.no_grad():
        try:
            ops = build()
            tc.synchronize()
        except Exception as e:
            row["result"] = "SETUP-ERROR"
            row["error"] = f"{type(e).__name__}: {e}"
            rows.append(row)
            continue
        c0 = counts()
        try:
            fn(ops)
            tc.synchronize()
            row["result"] = "RETURNED"
        except Exception as e:
            row["result"] = "RAISED"
            row["error"] = f"{type(e).__name__}: {e}"
        c1 = counts()
    row["by_function"] = {n: c1[n] - c0[n] for n in names if c1[n] != c0[n]}
    row["calls"] = sum(row["by_function"].values())
    rows.append(row)
print("ROWS " + json.dumps({"module_sha256": sha, "rows": rows}))
'''
    # Each case builds its operands first (uncounted, then synchronized) and counts only the call itself; the
    # counted window ends after a synchronize, so "calls" covers the call's allocations, launches and copies. The
    # synchronize after a refusal is itself a CUDA call: the post step's bar counts cudaMalloc*, cudaLaunchKernel and
    # cudaMemcpy* only (see scripts/bwdf2_gpu_post.py k2_count).
    env = dict(os.environ, LD_PRELOAD=str(shim))
    p = subprocess.run([sys.executable, "-B", "-P", "-c", CHILD, str(HERE), arm], env=env, capture_output=True,
                       text=True, timeout=400)
    got = [l[5:] for l in p.stdout.splitlines() if l.startswith("ROWS ")]
    doc = json.loads(got[-1]) if got else {"rows": [], "error": f"child exit {p.returncode}: {p.stderr[-1500:]}"}
    doc["child_exit"] = p.returncode
    rows = doc["rows"]
    sha = doc.get("module_sha256")
    for r_ in rows:
        print(json.dumps(r_), flush=True)
    (OUT / "child_stderr.txt").write_text(p.stderr[-20000:])

np.savez(OUT / "arrays.npz", **{k: v for k, v in arrays.items()})
(OUT / "results.json").write_text(json.dumps({"line": f"k2_{mode}_{arm}", "mode": mode, "arm": arm,
                                              "module_sha256": sha, "rows": rows}, indent=1, default=str))
print("DONE", mode, arm, len(rows), "rows")
