#!/usr/bin/env python3
"""ENG2 CPU emulation of the N1 tests (GPU-only) for the baseline, the candidate and every N1 mutant.

Usage: CUDA_VISIBLE_DEVICES= n1_emulation.py OUT.json
It imports E3's tape (_E3Arithmetic: IEEE nearest-even per primitive, float32 reductions in kernels.cu order, dtype
stores) from tensor_cuda/tests/test_eng1.py unchanged. ENG1's G5 census showed that tape bit-exact against the
baseline engine (engine_distance_from_emulation 0.0 on every finite row; equal non-finite patterns), so the baseline
predictions are strong. The candidate and mutant graphs are new compositions of the same primitives (an upcast at the
start, float32 arithmetic, one cast back); their only modelling gap is CUDA powf against a correctly rounded float32
power (<= 4 ulp float32, far below one half-precision ulp of the outputs).
Predicted outcome per test ID:
  test_n1_half_vs_reference: max |emulated - float64 reference| against E3's tolerance (finite required);
  test_n1_half_rule_bits: the emulated layer graph against the emulated registered rule, element by element in the
    stored dtype (the rule's emulation is the candidate graph, so the candidate is equal by construction; baseline and
    mutants are predicted to FAIL when at least one stored element differs);
  test_n1_batchnorm_running_stats: NumPy float32/float16 update emulation (NEP 50 promotion) against the test's
    tolerance;
  test_n1_float32_bits: float32 graphs (baseline sequence vs the arm's float32 path).
Prior art: Wengert (1964) reverse-mode tape; IEEE 754 roundTiesToEven; E3 (ENG1) emulation reused unchanged. Ours:
the candidate/mutant graphs below.
"""
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tensor_cuda"))
import tensor_cuda  # noqa: E402,F401  (test_eng1 imports it; CPU only, no tensor is created here)

spec = importlib.util.spec_from_file_location("eng1_e3", ROOT / "tensor_cuda/tests/test_eng1.py")
E3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E3)
A = E3._E3Arithmetic

CLASSES = ["LayerNorm", "RMSNorm", "BatchNorm1D", "InstanceNorm2D", "GroupNorm"]
SHORT = {"LayerNorm": "LN", "RMSNorm": "RMS", "BatchNorm1D": "BN", "InstanceNorm2D": "IN"}


def stored(values, dtype):
    return E3.e3_round(values, dtype)


def operands(cls, dtype, case):
    shape = E3.E3_SHAPE[cls]
    C, _ = E3._affine_shape(cls, shape)
    x = stored(E3.e3_inputs(cls, case), dtype)
    g = stored(E3._e3_pattern(shape, 1.1), dtype)
    w = stored(1.0 + 0.1 * np.arange(C), dtype)
    b = None if cls == "RMSNorm" else stored(0.05 * np.arange(C) - 0.1, dtype)
    return x, w, b, g


def graph(cls, dtype, x, w, b, g, variant):
    """Forward value and input gradient of sum(layer(x) * g) for `variant`:
    baseline   - bc02d3a1's nn.py (E3.e3_emulation, unchanged);
    candidate  - upcast, float32 graph (the baseline sequence), float32 weights, one cast back;
    MN1        - candidate without the cast back (values; the test fails on dtype regardless);
    MN2        - bfloat16 takes the baseline graph, float16 the candidate's;
    MN3        - the class's statistics in the input dtype, then cast to float32 (the registered hunk);
    f32 / MN4  - float32 inputs: the baseline sequence, and MN4's float16 round trip of xc * inv (LayerNorm)."""
    if variant == "baseline" or (variant == "MN2" and dtype == "bfloat16"):
        return E3.e3_emulation(cls, dtype, x, w, b, g)
    if cls == "GroupNorm" and variant in ("candidate", "MN1", "MN2", "MN3"):
        return E3.e3_emulation(cls, dtype, x, w, b, g)  # GroupNorm is unchanged (already GN1's float32 rule)
    leaf = A(x, dtype)
    work = leaf if dtype == "float32" else leaf.cast("float32")
    shape = work.value.shape
    axes = (-1,) if cls in ("RMSNorm", "LayerNorm") else E3._e3_axes(cls, shape)
    eps = E3.E3_EPS[cls]
    if variant == "MN3":
        h = leaf  # statistics on the stored half values
        if cls == "LayerNorm":
            mean = h.mean(axes); xc = h - mean; var = (xc * xc).mean(axes)
            centered, stat = xc.cast("float32"), var.cast("float32")
        elif cls == "RMSNorm":
            ms = (h * h).mean(axes)
            centered, stat = h.cast("float32"), ms.cast("float32")
        elif cls == "BatchNorm1D":  # x.astype(dtype).mean(...).astype(f32) on the upcast x
            hh = work.cast(dtype)
            mean = hh.mean(axes).cast("float32")
            hh2 = work.cast(dtype)
            d = hh2 - hh2.mean(axes)
            var = (d * d).mean(axes).cast("float32")
            centered, stat = work - mean, var
        else:  # InstanceNorm2D: x, mean, var cast after the half statistics
            mean = h.mean(axes); d = h - h.mean(axes); var = (d * d).mean(axes)
            centered, stat = h.cast("float32") - mean.cast("float32"), var.cast("float32")
    elif cls == "GroupNorm":  # float32 inputs only reach here
        return E3.e3_emulation(cls, dtype, x, w, b, g)
    elif cls == "RMSNorm":
        centered, stat = work, (work * work).mean(axes)
    else:
        mean = work.mean(axes)
        centered = work - mean
        vi = centered if cls == "LayerNorm" else work - work.mean(axes)
        stat = (vi * vi).mean(axes)
    normalized = centered * stat.scalar(eps).pow(-0.5)
    if variant == "MN4" and cls == "LayerNorm":
        normalized = normalized.cast("float16").cast("float32")
    _, bshape = E3._affine_shape(cls, shape)
    weight = A(w.reshape(bshape), "float32", requires_grad=False)
    y = normalized * weight
    if b is not None:
        y = y + A(b.reshape(bshape), "float32", requires_grad=False)
    if dtype != "float32" and variant != "MN1":
        y = y.cast(dtype)
    upstream = A(g, y.dtype, requires_grad=False) if variant != "MN1" else A(g, "float32", requires_grad=False)
    (y * upstream).sum(tuple(range(len(shape)))).backward()
    if variant == "MN1-prerounding":
        raise AssertionError("unreachable")
    if PRE is not None:  # margin analysis: the float32 values before the final cast (forward) / the input cast (VJP)
        PRE[:] = [y.value, work.grad]
    return y.value, leaf.grad


PRE = None


def margins(cls, dtype, case, direction, variant):
    """For a predicted rule-bits failure: at each element where the emulated variant differs from the emulated
    candidate, the candidate's float32 pre-rounding value's distance to the nearest rounding midpoint of `dtype`,
    in float32 ulps (a CUDA powf deviation of a few float32 ulps cannot move a value farther than that)."""
    global PRE
    x, w, b, g = operands(cls, dtype, case)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        got = graph(cls, dtype, x, w, b, g, variant)
        PRE = []
        graph(cls, dtype, x, w, b, g, "MN1")  # the candidate's float32 graph without the final cast
        pre_y, pre_dx = PRE
        PRE = None
        want = graph(cls, dtype, x, w, b, g, "candidate")
    i = 0 if direction == "forward" else 1
    a, c, pre = got[i].ravel(), want[i].ravel(), (pre_y if i == 0 else pre_dx).ravel()
    out = []
    for k in np.nonzero(~((a == c) | (np.isnan(a) & np.isnan(c))))[0]:
        v = np.float32(pre[k])
        lo, hi = np.float32(E3.e3_round(v, dtype)), None
        # the two representable neighbours around v in dtype, and their midpoint nearest to v
        step = np.float32(2.0 ** (math.frexp(float(abs(c[k])) or 1.0)[1] - (11 if dtype == "float16" else 8)))
        mids = [np.float64(c[k]) - step / 2, np.float64(c[k]) + step / 2]
        ulp32 = float(np.spacing(np.float32(abs(v)) or np.float32(1e-30)))
        out.append(min(abs(float(v) - m) for m in mids) / ulp32)
    return dict(differing=len(out), min_margin_f32_ulps=(min(out) if out else None))


def tolerance_row(cls, dtype, case, direction, variant):
    x, w, b, g = operands(cls, dtype, case)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        y, dx = graph(cls, dtype, x, w, b, g, variant)
    ref_y, ref_dx = E3.e3_reference(cls, x, w, b, g)
    got, ref = (y, ref_y) if direction == "forward" else (dx, ref_dx)
    scale = float(np.max(np.abs(ref)))
    tol = E3.e3_tolerance(cls, direction, dtype, scale)
    finite = bool(np.all(np.isfinite(got)))
    err = float(np.max(np.abs(np.where(np.isfinite(got), got, np.inf) - ref)))
    dtype_ok = variant != "MN1" or cls == "GroupNorm"
    return dict(err=err, tol=tol, ratio=err / tol if math.isfinite(err) else None, finite=finite,
                passes=bool(dtype_ok and finite and err <= tol), dtype_ok=dtype_ok)


def rule_row(cls, dtype, case, direction, variant):
    x, w, b, g = operands(cls, dtype, case)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        got = graph(cls, dtype, x, w, b, g, variant)
        want = graph(cls, dtype, x, w, b, g, "candidate")
    i = 0 if direction == "forward" else 1
    a, c = got[i], want[i]
    same_nan = np.isnan(a) & np.isnan(c)
    differ = int(np.count_nonzero(~((a == c) | same_nan)))
    dtype_ok = variant != "MN1" or cls == "GroupNorm"
    return dict(differing_elements=differ, of=int(a.size), passes=bool(dtype_ok and differ == 0), dtype_ok=dtype_ok)


def f32_row(cls, case, direction, variant):
    x, w, b, g = operands(cls, "float32", case)
    got = graph(cls, "float32", x, w, b, g, "MN4" if variant == "MN4" else "f32")
    want = graph(cls, "float32", x, w, b, g, "f32")
    i = 0 if direction == "forward" else 1
    differ = int(np.count_nonzero(got[i] != want[i]))
    return dict(differing_elements=differ, passes=differ == 0)


def running_stats_row(dtype, case, variant):
    """test_n1_batchnorm_running_stats: emulate the batch statistics and nn.py's NumPy update."""
    cls = "BatchNorm1D"
    x, _, _, _ = operands(cls, dtype, case)
    # Precision of the batch statistics, and the NumPy dtype of mean.numpy() feeding the update.
    half_stats = dtype != "float32" and (variant == "baseline" or variant == "MN3"
                                         or (variant == "MN2" and dtype == "bfloat16"))
    sdt = dtype if half_stats else "float32"
    numpy_f16 = half_stats and dtype == "float16" and variant != "MN3"  # MN3 casts the statistics to float32
    h = A(x, sdt, requires_grad=False)
    mean = h.mean((0,))
    d = h - h.mean((0,))
    var = (d * d).mean((0,))
    mvals, vvals = mean.value.ravel(), var.value.ravel()
    if numpy_f16:  # mean.numpy() is a float16 array: momentum * mn stays float16 under NEP 50
        mn, vr = mvals.astype(np.float16), vvals.astype(np.float16)
    else:  # float32 arrays (bfloat16 reads back widened to float32; MN3's statistics are cast to float32)
        mn, vr = mvals.astype(np.float32), vvals.astype(np.float32)
    rm0, rv0 = np.zeros(4, np.float32), np.ones(4, np.float32)
    rm = (1 - 0.1) * rm0 + 0.1 * mn
    rv = (1 - 0.1) * rv0 + 0.1 * vr
    rm, rv = np.asarray(rm, np.float32).astype(np.float64), np.asarray(rv, np.float32).astype(np.float64)
    N, m, u = x.shape[0], 0.1, 2.0 ** -24
    mu, vv = x.mean(axis=0), x.var(axis=0)
    want_m, want_v = (1 - m) * 0 + m * mu, (1 - m) * 1 + m * vv
    d_mean = (N + 1) * u * np.abs(x).mean(axis=0)
    tol_m = m * d_mean + 3 * u * np.abs(want_m) + 2.0 ** -149
    tol_v = m * ((N + 3) * u * vv + d_mean ** 2) + 3 * u * np.abs(want_v) + 2.0 ** -149
    em, ev = np.abs(rm - want_m), np.abs(rv - want_v)
    return dict(err_mean_over_tol=float(np.max(em / tol_m)), err_var_over_tol=float(np.max(ev / tol_v)),
                passes=bool(np.all(em <= tol_m) and np.all(ev <= tol_v)))


def main(out):
    variants = ["baseline", "candidate"] + [f"MN{k}_{SHORT[c]}" for k in (1, 2, 3) for c in SHORT] + ["MN4_LN"]
    P = "tensor_cuda/tests/test_eng2.py::"
    rows = {}
    for v in variants:
        kind, _, short = v.partition("_")
        target = {s: c for c, s in SHORT.items()}.get(short)
        res = {}
        for cls in CLASSES:
            arm = kind if (target is None or cls == target) else "candidate"
            for dtype in ("float16", "bfloat16"):
                for case in ("regular", "near_eps"):
                    for direction in ("forward", "backward"):
                        a = arm if arm in ("baseline", "candidate") else arm  # MN1/MN2/MN3 on the target class
                        res[P + f"test_n1_half_vs_reference[{cls}-{dtype}-{case}-{direction}]"] = tolerance_row(
                            cls, dtype, case, direction, a if a != "MN4" else "candidate")
                        res[P + f"test_n1_half_rule_bits[{cls}-{dtype}-{case}-{direction}]"] = rule_row(
                            cls, dtype, case, direction, a if a != "MN4" else "candidate")
            for case in ("regular", "near_eps"):
                for direction in ("forward", "backward"):
                    fv = "MN4" if (arm == "MN4" and cls == "LayerNorm") else "f32"
                    res[P + f"test_n1_float32_bits[{cls}-{case}-{direction}]"] = f32_row(cls, case, direction, fv)
        bn_arm = kind if (target is None or target == "BatchNorm1D") else "candidate"
        for dtype in ("float32", "float16", "bfloat16"):
            for case in ("regular", "near_eps"):
                rv = bn_arm if bn_arm in ("baseline", "candidate", "MN2", "MN3") else "candidate"
                res[P + f"test_n1_batchnorm_running_stats[{dtype}-{case}]"] = running_stats_row(dtype, case, rv)
        rows[v] = res
    fails = {v: sorted(k for k, r in res.items() if not r["passes"]) for v, res in rows.items()}
    # Margin of every predicted rule-bits failure that rests on value differences (dtype failures need none).
    for v, res in rows.items():
        kind, _, short = v.partition("_")
        for k, r in res.items():
            if "half_rule_bits[" in k and not r["passes"] and r["dtype_ok"]:
                cls, dtype, case, direction = k.split("[")[1][:-1].split("-")
                arm = kind
                r["margin"] = margins(cls, dtype, case, direction, arm)
    Path(out).write_text(json.dumps(dict(predicted_failures=fails, rows=rows), indent=1, default=float) + "\n")
    for v in variants:
        print(v, len(fails[v]), "predicted failing N1 IDs")


if __name__ == "__main__":
    main(sys.argv[1])
