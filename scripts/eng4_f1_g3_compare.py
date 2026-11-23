"""ENG4-F1 g3_F1 compare (CPU, after the GPU lines g3_B and g3_F1).

  python3 -B -P scripts/eng4_f1_g3_compare.py <dir_B> <dir_F1> <out.json>
  (dir_B = artifacts/eng4/gpu/g3_B/g3, the lead's ENG4 line; dir_F1 = artifacts/eng4/f1/gpu/g3_F1/g3)

Both directories are written by scripts/eng4_g3.py `run` (the same seeded grid, each build in its own process).
- Nonzero-size cells: F1's output must be bitwise equal to B's, per element on same-width unsigned views, after
  tensor-dtype, host-dtype and shape checks; each side must be nonzero and non-constant at full width (L19).
- Zero-size cells (M, N, K or batch = 0): F1's output must equal NumPy's result (L09): an empty array compared by
  shape and dtype; K = 0 with a nonempty output compared by value (every element a positive zero). The shape rule is
  NumPy's for classes a and b and A's batch + [M, N] for class c (M2(c)). B's value for the same cell is recorded
  beside it (shape, all-positive-zero, sha256 of its bytes), never gated (X2: B's K = 0 output was unwritten memory).

Prior art: NumPy's matmul on zero-size operands (empty result; zeros for an empty contraction); ENG4's eng4_g3.py
compare (this repo). No other prior art known to me.
"""
import hashlib
import json
import os
import sys

import numpy as np


def uview(x, dtype):
    return x.view(np.uint16 if dtype == "float16" else np.uint32)


def m2_shape(c):
    a_s, b_s = c["a"], c["b"]
    return list(a_s[:-1]) + [b_s[-2] if c["trans_b"] else b_s[-1]]


def main():
    db, df, out = sys.argv[1], sys.argv[2], sys.argv[3]
    mb = json.load(open(os.path.join(db, "manifest.json")))
    mf = json.load(open(os.path.join(df, "manifest.json")))
    ab, af = np.load(os.path.join(db, "outputs.npz")), np.load(os.path.join(df, "outputs.npz"))
    assert [c["id"] for c in mb["cells"]] == [c["id"] for c in mf["cells"]], "grids differ"
    rows, fails = [], []
    for cb, cf in zip(mb["cells"], mf["cells"]):
        cid = cb["id"]
        row = {"id": cid, "seed": cb["seed"], "class": cb["class"], "zero_size": cb["zero_size"],
               "same_inputs": (cb["a_sha256"], cb["b_sha256"]) == (cf["a_sha256"], cf["b_sha256"])}
        if "raised" in cf:
            row.update(verdict="FAIL: F1 raised", raised_F1=cf["raised"], raised_B=cb.get("raised"))
        elif not cb["zero_size"]:
            if "raised" in cb:
                row.update(verdict="FAIL: B raised", raised_B=cb["raised"])
            else:
                xb, xf = ab[cid], af[cid]
                checks = {"tensor_dtype": cb["tensor_dtype"] == cf["tensor_dtype"], "host_dtype": xb.dtype == xf.dtype,
                          "shape": xb.shape == xf.shape}
                row["checks"] = checks
                if all(checks.values()):
                    ub, uf = uview(xb, cb["dtype"]), uview(xf, cb["dtype"])
                    diff = int(np.count_nonzero(ub != uf))
                    l19 = {s: bool(np.any(u != 0) and np.unique(u).size > 1) for s, u in (("B", ub), ("F1", uf))}
                    row.update(elements=int(ub.size), differing_elements=diff, L19_nonzero_nonconstant=l19,
                               verdict="PASS" if diff == 0 and row["same_inputs"] and all(l19.values()) else "FAIL")
                else:
                    row["verdict"] = "FAIL: dtype/shape"
        else:
            xf = af[cid]
            rule = cb["numpy_matmul_shape"] if cb["class"] in ("a", "b") else m2_shape(cb)
            uf = uview(xf, cb["dtype"])
            f1 = {"shape": list(xf.shape), "tensor_dtype": cf["tensor_dtype"], "host_dtype": str(xf.dtype),
                  "elements": int(xf.size), "all_positive_zero": bool(np.all(uf == 0))}
            want_dtype = "float16" if cb["dtype"] == "float16" else "float32"  # bf16 comes back as its fp32 widening
            ok = (f1["shape"] == list(rule) and cf["tensor_dtype"] == cb["dtype"] and str(xf.dtype) == want_dtype
                  and (xf.size == 0 or f1["all_positive_zero"]))
            row.update(numpy_shape_rule=list(rule), F1=f1, verdict="PASS" if ok else "FAIL")
            if "raised" in cb:
                row["B"] = {"raised": cb["raised"]}
            elif cid in ab.files:
                xb = ab[cid]
                ubb = uview(xb, cb["dtype"])
                row["B"] = {"shape": list(xb.shape), "elements": int(xb.size), "all_positive_zero": bool(np.all(ubb == 0)),
                            "sha256": hashlib.sha256(xb.tobytes()).hexdigest(),
                            "nonzero_elements": int(np.count_nonzero(ubb))}
        rows.append(row)
        if row["verdict"] != "PASS":
            fails.append(cid)
    rec = {"B": {"dir": db, "module_sha256": mb["module_sha256"]}, "F1": {"dir": df, "module_sha256": mf["module_sha256"]},
           "cells": len(rows), "nonzero_cells": sum(not r["zero_size"] for r in rows),
           "zero_size_cells": sum(r["zero_size"] for r in rows), "failed": fails,
           "verdict": "PASS" if not fails else "FAIL", "rows": rows}
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    print("g3_F1", rec["verdict"], "cells", len(rows), "failed", len(fails))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
