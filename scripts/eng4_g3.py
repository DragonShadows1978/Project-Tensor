"""ENG4 G3: accepted results unchanged. Each build runs the same seeded grid in its own process (L25); the outputs
are compared afterwards on the CPU, per element on same-width unsigned views, after dtype and shape checks (L09).

  PYTHONPATH=<package parent> python3 -B -P scripts/eng4_g3.py run <outdir>      (GPU; one build per process)
  python3 -B -P scripts/eng4_g3.py compare <dir_B> <dir_E> <out.json>           (CPU)

Grid (order G3): every M2 class (a: B rank 2; b: equal ranks and batch shapes, incl. rank 2 x rank 2; c: unequal
ranks with an all-ones B batch, B rank 4 under A rank 3 and B rank 3 under A rank 2), each dtype (float32, float16,
bfloat16), trans_b false/true, alpha 1 and 0.5; small and medium shapes for a and b. Plus the zero-size cases
(M, N, K or batch = 0) per class and dtype (trans_b false, alpha 1), each recorded with numpy.matmul's shape.
Seeds: cell i uses default_rng(SEED0 + i), recorded per cell with the input sha256s.
L19: every nonzero-size comparison first checks that each side is nonzero and non-constant at full width (the
uint view has more than one distinct value and is not all zero); zero-size cases record the expected value instead
(empty output, or all positive zeros for K = 0, as numpy.matmul gives).

Prior art: bitwise A/B comparison of two builds on the same seeded inputs (ENG2/ENG3 receipts in this repo); NumPy's
matmul shape rule (numpy.matmul docs) for the zero-size shapes. No other prior art known to me.
"""
import hashlib
import json
import os
import sys

import numpy as np

SEED0 = 20261123
SHAPES = {  # class: (A shape, B shape without trans_b; B's last two dims are swapped for trans_b)
    "a_small": ("a", (3, 17, 33), (33, 19)),
    "a_medium": ("a", (4, 64, 128), (128, 96)),
    "b_small": ("b", (2, 3, 17, 33), (2, 3, 33, 19)),
    "b_medium": ("b", (3, 64, 128), (3, 128, 96)),
    "b_rank2": ("b", (17, 33), (33, 19)),
    "c_b4_a3": ("c", (3, 17, 33), (1, 1, 33, 19)),
    "c_b3_a2": ("c", (17, 33), (1, 33, 19)),
}
ZERO = {  # zero-size cases (trans_b false, alpha 1)
    "a_M0": ("a", (3, 0, 33), (33, 19)), "a_N0": ("a", (3, 17, 33), (33, 0)), "a_K0": ("a", (3, 17, 0), (0, 19)),
    "a_batch0": ("a", (0, 17, 33), (33, 19)),
    "b_M0": ("b", (2, 3, 0, 33), (2, 3, 33, 19)), "b_N0": ("b", (2, 3, 17, 33), (2, 3, 33, 0)),
    "b_K0": ("b", (2, 3, 17, 0), (2, 3, 0, 19)), "b_batch0": ("b", (0, 3, 17, 33), (0, 3, 33, 19)),
    "b_rank2_K0": ("b", (17, 0), (0, 19)),
    "c_M0": ("c", (3, 0, 33), (1, 1, 33, 19)), "c_N0": ("c", (3, 17, 33), (1, 1, 33, 0)),
    "c_K0": ("c", (3, 17, 0), (1, 1, 0, 19)), "c_batch0": ("c", (0, 17, 33), (1, 1, 33, 19)),
}
DTYPES = ("float32", "float16", "bfloat16")
UINT = {"float32": np.uint32, "float16": np.uint16}


def cells():
    out = []
    for name, (cls, sa, sb) in SHAPES.items():
        for dt in DTYPES:
            for tb in (False, True):
                for alpha in (1.0, 0.5):
                    out.append({"id": f"{name}-{dt}-{'T' if tb else 'N'}-a{alpha}", "class": cls, "a": sa,
                                "b": sb[:-2] + (sb[-1], sb[-2]) if tb else sb, "dtype": dt, "trans_b": tb,
                                "alpha": alpha, "zero_size": False})
    for name, (cls, sa, sb) in ZERO.items():
        for dt in DTYPES:
            out.append({"id": f"{name}-{dt}", "class": cls, "a": sa, "b": sb, "dtype": dt, "trans_b": False,
                        "alpha": 1.0, "zero_size": True})
    for i, c in enumerate(out):
        c["seed"] = SEED0 + i
    return out


def operand(tc, rng, shape, dtype):
    x = rng.standard_normal(shape).astype(np.float32)
    host = x.astype(np.float16) if dtype == "float16" else x
    t = tc.tensor(host, dtype=dtype)
    return t, hashlib.sha256(host.tobytes()).hexdigest()


def run(outdir):
    import tensor_cuda as tc
    os.makedirs(outdir)  # write-once
    manifest = {"module": tc._C.__file__,
                "module_sha256": hashlib.sha256(open(tc._C.__file__, "rb").read()).hexdigest(), "cells": []}
    arrays = {}
    for c in cells():
        rng = np.random.default_rng(c["seed"])
        a, ha = operand(tc, rng, c["a"], c["dtype"])
        b, hb = operand(tc, rng, c["b"], c["dtype"])
        row = dict(c, a=list(c["a"]), b=list(c["b"]), a_sha256=ha, b_sha256=hb)
        try:
            r = tc.matmul(a, b, alpha=c["alpha"], trans_b=c["trans_b"])
            host = r.numpy()
            row.update(tensor_dtype=r.dtype, out_shape=list(host.shape), host_dtype=str(host.dtype))
            arrays[c["id"]] = host
        except Exception as e:  # noqa: BLE001 (recorded; an accepted cell must not raise)
            row["raised"] = {"type": type(e).__name__, "message": str(e)}
        if c["zero_size"]:
            row["numpy_matmul_shape"] = list(np.matmul(np.zeros(c["a"], np.float32),
                                                       np.zeros(c["b"], np.float32)).shape)
        manifest["cells"].append(row)
    tc.synchronize()
    np.savez(os.path.join(outdir, "outputs.npz"), **arrays)
    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("cells", len(manifest["cells"]), "raised", sum("raised" in r for r in manifest["cells"]))


def uview(x, dtype):
    # bfloat16 outputs come back as their exact float32 widening (bindings.cpp tensor_to_numpy), compared as uint32
    return x.view(UINT.get(str(x.dtype), np.uint32) if dtype != "float16" else np.uint16)


def compare(db, de, out):
    mb = json.load(open(os.path.join(db, "manifest.json")))
    me = json.load(open(os.path.join(de, "manifest.json")))
    ab, ae = np.load(os.path.join(db, "outputs.npz")), np.load(os.path.join(de, "outputs.npz"))
    rows, fails = [], []
    assert [c["id"] for c in mb["cells"]] == [c["id"] for c in me["cells"]]
    for cb, ce in zip(mb["cells"], me["cells"]):
        cid, row = cb["id"], {"id": cb["id"], "seed": cb["seed"]}
        same_inputs = (cb["a_sha256"], cb["b_sha256"]) == (ce["a_sha256"], ce["b_sha256"])
        row["same_inputs"] = same_inputs
        if "raised" in cb or "raised" in ce:
            row["verdict"] = "FAIL: raised"
            row["raised"] = {"B": cb.get("raised"), "E": ce.get("raised")}
        else:
            xb, xe = ab[cid], ae[cid]
            checks = {"tensor_dtype": cb["tensor_dtype"] == ce["tensor_dtype"],
                      "host_dtype": xb.dtype == xe.dtype, "shape": xb.shape == xe.shape}
            row.update(shape=list(xb.shape), tensor_dtype=cb["tensor_dtype"], checks=checks)
            if all(checks.values()):
                ub, ue = uview(xb, cb["dtype"]), uview(xe, cb["dtype"])
                diff = int(np.count_nonzero(ub != ue))
                row["elements"], row["differing_elements"] = int(ub.size), diff
                if cb["zero_size"]:
                    row["all_positive_zero"] = {"B": bool(np.all(ub == 0)), "E": bool(np.all(ue == 0))}
                    row["numpy_matmul_shape"] = cb["numpy_matmul_shape"]
                    # M6 shape rule under M2: classes a and b give numpy.matmul's shape; class c gives A's batch
                    # shape + [M, N] (M2(c): the output shape stays as today, not NumPy's broadcast shape)
                    a_s, b_s = cb["a"], cb["b"]
                    m2_shape = list(a_s[:-1]) + [b_s[-2] if cb["trans_b"] else b_s[-1]]
                    rule = cb["numpy_matmul_shape"] if cb["class"] in ("a", "b") else m2_shape
                    row["m2_shape_rule"] = rule
                    row["shape_rule_ok"] = list(xb.shape) == rule
                    expect_ok = row["shape_rule_ok"] and (
                        ub.size == 0 or (row["all_positive_zero"]["B"] and row["all_positive_zero"]["E"]))
                    row["verdict"] = "PASS" if diff == 0 and same_inputs and expect_ok else "FAIL"
                else:
                    l19 = {s: bool(np.any(u != 0) and np.unique(u).size > 1) for s, u in (("B", ub), ("E", ue))}
                    row["L19_nonzero_nonconstant"] = l19
                    row["verdict"] = "PASS" if diff == 0 and same_inputs and all(l19.values()) else "FAIL"
            else:
                row["verdict"] = "FAIL: dtype/shape"
        rows.append(row)
        if row["verdict"] != "PASS":
            fails.append(cid)
    rec = {"B": {"dir": db, "module_sha256": mb["module_sha256"]}, "E": {"dir": de, "module_sha256": me["module_sha256"]},
           "cells": len(rows), "failed": fails, "verdict": "PASS" if not fails else "FAIL", "rows": rows}
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    print("G3", rec["verdict"], "cells", len(rows), "failed", len(fails))
    return 0 if not fails else 1


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2])
    elif sys.argv[1] == "compare":
        sys.exit(compare(sys.argv[2], sys.argv[3], sys.argv[4]))
    elif sys.argv[1] == "list":
        for c in cells():
            print(c["id"], c["seed"], c["a"], c["b"])
