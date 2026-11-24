"""ENG4-F3 g3z: the G3 grid's zero-size cells and the 24 empty-batch shared-B gradient cases on F3, against NumPy and
exact zeros (L09, L25: F3 alone, no cross-build comparison). Run inside the GPU line g3z_F3.

  python3 -B -P scripts/eng4_f3_g3z.py run <outdir>        (GPU line; F3 package on PYTHONPATH)
  python3 -B -P scripts/eng4_f3_g3z.py verdict <outdir>    (CPU, after the line: VERDICT.json)

Cells: every zero_size cell of scripts/eng4_g3.py's grid (M, N, K or batch = 0; classes a, b, c; three dtypes; the
same seeds); gradient cases: scripts/eng4_f2_bwd0.py's 24 cases, recorded with its record_case on "cuda" and judged
with its judge_row (B's gradient: positive zeros of B's shape and dtype; A's: empty). Both helper scripts are
imported unchanged. This script never reads, sets or clears CUDA_VISIBLE_DEVICES.

Zero-size rule: the output's shape is NumPy's for classes a and b and A's batch + [M, N] for class c (M2(c)); an
empty output is checked by shape and dtype, a K = 0 output by value (every element a positive zero, on the unsigned
view; bfloat16 comes back as its float32 widening).

Prior art: NumPy's matmul on zero-size operands (empty result; zeros for an empty contraction); the ENG4 G3 and bwd0
harnesses (this repo). No other prior art known to me.
"""
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


g3 = load("eng4_g3", "scripts/eng4_g3.py")
bwd0 = load("eng4_f2_bwd0", "scripts/eng4_f2_bwd0.py")


def zero_cells():
    return [c for c in g3.cells() if c["zero_size"]]


def expected_shape(c):
    a_s, b_s = c["a"], c["b"]
    b_log = tuple(b_s[:-2]) + ((b_s[-1], b_s[-2]) if c["trans_b"] else tuple(b_s[-2:]))
    if c["class"] in ("a", "b"):
        return list(np.matmul(np.zeros(a_s, np.float32), np.zeros(b_log, np.float32)).shape)
    return list(a_s[:-1]) + [b_log[-1]]


def run_cells(tc, device="cuda"):
    rows = []
    for c in zero_cells():
        rng = np.random.default_rng(c["seed"])
        row = {"id": c["id"], "class": c["class"], "dtype": c["dtype"], "a": list(c["a"]), "b": list(c["b"]),
               "trans_b": c["trans_b"], "seed": c["seed"]}
        try:
            x = rng.standard_normal(c["a"]).astype(np.float32)
            y = rng.standard_normal(c["b"]).astype(np.float32)
            hx = x.astype(np.float16) if c["dtype"] == "float16" else x
            hy = y.astype(np.float16) if c["dtype"] == "float16" else y
            a = tc.tensor(hx, dtype=c["dtype"], device=device)
            b = tc.tensor(hy, dtype=c["dtype"], device=device)
            r = tc.matmul(a, b, alpha=c["alpha"], trans_b=c["trans_b"])
            host = np.ascontiguousarray(r.numpy())
            row.update(tensor_dtype=r.dtype, shape=list(host.shape), host_dtype=str(host.dtype),
                       hex=host.tobytes().hex(), sha256=hashlib.sha256(host.tobytes()).hexdigest())
        except Exception as e:  # noqa: BLE001 (recorded)
            row["raised"] = {"type": type(e).__name__, "message": str(e)}
        rows.append(row)
    return rows


def run(outdir):
    import tensor_cuda as tc
    os.makedirs(outdir)  # write-once
    cells = run_cells(tc)
    grads = []
    for c in bwd0.cases():
        r = dict(c, child_exit=0)
        r.update(bwd0.record_case(tc, c, "cuda"))
        grads.append(r)
    tc.synchronize()
    rec = {"module": tc._C.__file__, "module_sha256": hashlib.sha256(open(tc._C.__file__, "rb").read()).hexdigest(),
           "cells": cells, "gradients": grads}
    json.dump(rec, open(os.path.join(outdir, "g3z.json"), "w"), indent=1)
    return 0


def judge_cell(r):
    if "raised" in r:
        return "FAIL", f"raised: {r['raised']}"
    want = expected_shape(r)
    host = "float16" if r["dtype"] == "float16" else "float32"
    if r["tensor_dtype"] != r["dtype"] or r["host_dtype"] != host or r["shape"] != want:
        return "FAIL", f"dtype/shape {r['tensor_dtype']} {r['host_dtype']} {r['shape']} vs {want}"
    raw = bytes.fromhex(r["hex"])
    if int(np.prod(want)) == 0:
        return ("PASS", "empty, NumPy's shape and dtype") if not raw else ("FAIL", "bytes in an empty output")
    width = np.uint16 if host == "float16" else np.uint32
    u = np.frombuffer(raw, dtype=width)
    if u.size != int(np.prod(want)) or np.any(u):
        return "FAIL", "K = 0 output is not all positive zeros"
    return "PASS", "K = 0: positive zeros of NumPy's shape"


def verdict(outdir):
    rec = json.load(open(os.path.join(outdir, "g3z.json")))
    cells = [{"id": r["id"], "verdict": judge_cell(r)[0], "reason": judge_cell(r)[1]} for r in rec["cells"]]
    grads = [{"id": r["id"], "verdict": bwd0.judge_row(r)[0], "reason": bwd0.judge_row(r)[1]} for r in rec["gradients"]]
    ok = (len(cells) == len(zero_cells()) and len(grads) == 24 and
          all(x["verdict"] == "PASS" for x in cells + grads))
    out = {"module_sha256": rec["module_sha256"], "zero_size_cells": len(cells), "gradient_cases": len(grads),
           "verdict": "PASS" if ok else "FAIL", "cells": cells, "gradients": grads}
    json.dump(out, open(os.path.join(outdir, "VERDICT.json"), "w"), indent=1)
    print("g3z", out["verdict"], len(cells), len(grads))
    return 0 if ok else 1


if __name__ == "__main__":
    {"run": lambda: sys.exit(run(sys.argv[2])), "verdict": lambda: sys.exit(verdict(sys.argv[2]))}[sys.argv[1]]()
