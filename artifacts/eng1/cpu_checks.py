#!/usr/bin/env python3
"""ENG1 G2 CPU checks (no device; CUDA_VISIBLE_DEVICES must be empty).

1. E2: the float64 reference gradient against float64 central differences.
2. E3: the float64 VJP of every census class and case against float64 central differences of its forward.
3. E2 per-ID prediction: NumPy emulation of the baseline arm (pow, scale, multiply each rounded to the dtype) and of
   the candidate (float32 chain, one rounding), scored with test_eng1.py's own tolerances. powf is emulated by NumPy's
   float32 power; a case is "knife-edge" when its margin is within CUDA powf's 4-ulp band, so the prediction is not
   claimed there.
4. L55: a constructed float32 input on which reordering g * (p * x^(p-1)) as (g * p) * x^(p-1) changes the bits.
Prior art: central differences (classical; Fornberg 1988 for the error analysis, unverified -- lead to check);
Higham (2002) gamma_n; bfloat16 round-to-nearest-even by integer bias (as in TensorFlow's bfloat16 conversion,
unverified -- lead to check). Ours: the emulation and the scoring.
Usage: cpu_checks.py OUT.json
"""
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU checks run with CUDA hidden"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tensor_cuda"))
spec = importlib.util.spec_from_file_location("t_eng1", ROOT / "tensor_cuda" / "tests" / "test_eng1.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)  # imports tensor_cuda (no device call at import)

out = {}


def rnd(a, dtype):
    a = np.asarray(a, dtype=np.float32)
    if dtype == "float16":
        return a.astype(np.float16).astype(np.float32)
    if dtype == "bfloat16":
        b = a.view(np.uint32).astype(np.uint64)
        b = ((b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
        return b.view(np.float32)
    return a


# 1. E2 reference vs central differences
rows = []
for p in T.E2_P:
    x = np.array(T.E2_X)
    w = np.ones_like(x) * 0.75
    h = x * 1e-5
    f = lambda z: w * np.power(z, p)
    cd = (f(x + h) - f(x - h)) / (2 * h)
    ref = T.e2_reference_grad(x, w, p)
    rel = float(np.max(np.abs(cd - ref) / np.abs(ref)))
    rows.append(dict(p=p, max_rel=rel, passed=rel < 1e-8))
out["e2_central_difference"] = rows

# 2. E3 VJP vs central differences of sum(y * g)
rows = []
for cls in T.E3_CLASSES:
    for case in T.E3_CASES:
        x = T.e3_inputs(cls, case)
        shape = x.shape
        C, _ = T._affine_shape(cls, shape)
        w = 1.0 + 0.1 * np.arange(C)
        b = None if cls == "RMSNorm" else 0.05 * np.arange(C) - 0.1
        g = T._e3_pattern(shape, 1.1)
        _, dx = T.e3_reference(cls, x, w, b, g)
        spread = float(np.std(x)) if case == "regular" else math.sqrt(T.E3_EPS[cls])
        h = 1e-5 * spread
        cd = np.zeros(x.size)
        for i in range(x.size):
            e = np.zeros(x.size)
            e[i] = h
            yp, _ = T.e3_reference(cls, x + e.reshape(shape), w, b, g)
            ym, _ = T.e3_reference(cls, x - e.reshape(shape), w, b, g)
            cd[i] = (np.sum(yp * g) - np.sum(ym * g)) / (2 * h)
        err = float(np.max(np.abs(cd - dx.ravel())) / np.max(np.abs(dx)))
        y, _ = T.e3_reference(cls, x, w, b, g)
        rows.append(dict(cls=cls, case=case, max_rel_to_scale=err, passed=err < 1e-6,
                         y_ptp=float(np.ptp(y)), dx_ptp=float(np.ptp(dx))))
out["e3_vjp_central_difference"] = rows

# E3 hand value: LayerNorm of [1, 3] with eps 0, w 1, b 0 is [-1, 1]
y, _ = T.e3_reference("LayerNorm", np.array([[1.0, 3.0]]), np.ones(2), np.zeros(2), np.ones((1, 2)))
out["e3_hand_value"] = dict(value=y.tolist(), passed=bool(np.allclose(y, [[-1.0, 1.0]], atol=1e-5)))

# 3. E2 emulation of both arms, scored with the test's tolerances
pred = {}
for dtype in T.E2_DTYPES:
    edt = T.E2_ENGINE_DTYPE[dtype]
    for p in T.E2_P:
        x_all = rnd(np.array(T.E2_X), edt).astype(np.float64)
        k = T.e2_scale_exponents(x_all, p)
        keep = k <= 24 if edt == "float16" else np.ones(k.shape, dtype=bool)
        x, w = x_all[keep], np.power(2.0, -k[keep])
        ref = T.e2_reference_grad(x, w, p)
        x32, w32 = x.astype(np.float32), rnd(w, edt)
        with np.errstate(over="ignore", invalid="ignore"):
            pw = np.power(x32, np.float32(p - 1.0))
            base = rnd(w32 * rnd(rnd(pw, edt) * np.float32(p), edt), edt).astype(np.float64)
            cand = rnd(w32 * (pw * np.float32(p)), edt).astype(np.float64)
        rtol, atol = T.E2_RTOL[edt], T.E2_ATOL[edt]
        res = {}
        for arm, got in (("baseline", base), ("candidate", cand)):
            finite = bool(np.all(np.isfinite(got)))
            slack = atol + rtol * np.abs(ref) - np.abs(np.where(np.isfinite(got), got, np.inf) - ref)
            band = T.POWF_REL * np.abs(ref) * (1 + 3 * T.U.get(edt, 0))
            passes = finite and bool(np.all(slack >= 0))
            knife = finite and bool(np.any(np.abs(slack) <= band))
            res[arm] = dict(predicted="pass" if passes else "fail", knife_edge=knife, finite=finite,
                            worst_slack_over_tol=float(np.min(slack / (atol + rtol * np.abs(ref)))) if finite else None)
        pred[f"tensor_cuda/tests/test_eng1.py::test_e2_pow_grad[{dtype}-p{p:+g}]"] = res
out["e2_emulated_predictions"] = pred

# 4. L55 witness
xw, gw, p = np.float32(T.E2_WITNESS_X), np.float32(T.E2_WITNESS_G), np.float32(3.0)
x2 = xw * xw
a = gw * (x2 * p)
b = (gw * p) * x2
out["l55_witness"] = dict(x=float(xw), g=float(gw), p=3.0, x_squared_exact=bool(float(x2) == float(xw) ** 2),
                          ordered_bits=hex(int(np.array(a).view(np.uint32))),
                          reordered_bits=hex(int(np.array(b).view(np.uint32))), differ=bool(a != b))

# Tolerance tables
out["tolerances"] = dict(E2_RTOL=T.E2_RTOL, E2_ATOL=T.E2_ATOL, E2_RTOL_F32=T.E2_RTOL_F32, E5_RTOL=T.E5_RTOL,
                         U=T.U, SUB_HALF=T.SUB_HALF, POWF_REL=T.POWF_REL, E3_ROUNDINGS=T.E3_ROUNDINGS)
passed = (all(r["passed"] for r in out["e2_central_difference"])
          and all(r["passed"] for r in out["e3_vjp_central_difference"])
          and out["e3_hand_value"]["passed"] and out["l55_witness"]["differ"] and out["l55_witness"]["x_squared_exact"]
          and all(v["candidate"]["predicted"] == "pass" for v in pred.values()))
out["passed"] = passed
Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(dict(passed=passed,
                      e2_cd=[r["passed"] for r in out["e2_central_difference"]],
                      e3_cd_worst=max(r["max_rel_to_scale"] for r in out["e3_vjp_central_difference"]),
                      l55=out["l55_witness"],
                      baseline_pred={k.split("[")[1][:-1]: (v["baseline"]["predicted"], v["baseline"]["knife_edge"]) for k, v in pred.items()},
                      candidate_fail=[k for k, v in pred.items() if v["candidate"]["predicted"] != "pass"])))
sys.exit(0 if passed else 1)
