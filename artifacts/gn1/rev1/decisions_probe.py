#!/usr/bin/env python3
"""GN1-REV1 Q1/Q4: uncovered decisions and input validation, on CPU with the
NumPy stand-in engine from shim_trace.py (forward semantics only).

(a) Census of (x dtype, weight dtype, bias dtype) over every registered value ID.
(b) Variant V1: fast-path condition without the bias check; V2: without the
    weight check; V3: upcast predicate `dtype != "float32"` instead of the
    half-precision tuple. Each variant is the candidate nn.py with one edit,
    written under /tmp/gn1_rev1_scratch. Shown: same behaviour on every
    registered value ID's forward (so no test can kill it), different
    behaviour on an accepted input outside the census.
(c) Input validation: GroupNorm(-2, 6), GroupNorm(0, 6), GroupNorm(2.0, 6),
    channel mismatch.
Prior art: DeMillo, Lipton and Sayward (1978), mutation adequacy; RIP model
(unverified - lead to check). Ours: the variants.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
A1 = ROOT / "artifacts/gn1/a1"
sys.path.insert(0, str(HERE))
import shim_trace as S  # noqa: E402
import run_numerics as RN  # noqa: E402

CAND = A1 / "arms/candidate/tensor_cuda/nn.py"
SCR = Path("/tmp/gn1_rev1_scratch/variants")


def variant(name, old, new):
    text = CAND.read_text()
    assert text.count(old) == 1, name
    d = SCR / name / "tensor_cuda"
    d.mkdir(parents=True, exist_ok=True)
    (d / "nn.py").write_text(text.replace(old, new))
    return d / "nn.py"


def forward(nnmod, cs):
    layer = nnmod.GroupNorm(cs["G"], cs["x"].shape[1]) if cs["eps"] == 1e-5 else nnmod.GroupNorm(cs["G"], cs["x"].shape[1], eps=cs["eps"])
    layer.weight = S.Tensor(cs["w"], cs["wd"], True)
    layer.bias = S.Tensor(cs["b"], cs["bd"], True)
    try:
        y = layer(S.Tensor(cs["x"], cs["xd"], True))
        return ("ok", y.dtype, y._d)
    except Exception as exc:
        return ("raise", f"{type(exc).__name__}: {exc}", None)


def main():
    out = {}
    cases = RN.cases()
    out["param_dtype_census"] = sorted({(c["xd"], c["wd"], c["bd"]) for c in cases})
    out["any_weight_bias_dtype_differ"] = any(c["wd"] != c["bd"] for c in cases)
    V = {
        "V1_no_bias_check": variant("V1", '''        if (dtype == "float32" and self.weight.dtype == "float32"
                and self.bias.dtype == "float32"):''', '''        if (dtype == "float32" and self.weight.dtype == "float32"):'''),
        "V2_no_weight_check": variant("V2", '''        if (dtype == "float32" and self.weight.dtype == "float32"
                and self.bias.dtype == "float32"):''', '''        if (dtype == "float32"
                and self.bias.dtype == "float32"):'''),
        "V3_upcast_any_non_fp32": variant("V3", 'if dtype in ("float16", "bfloat16"):', 'if dtype != "float32":'),
    }
    cand = S.load_nn(CAND)
    extra = [dict(id="x f32, w f32, b f16", x=np.linspace(-1, 1, 12).reshape(2, 6), G=3, eps=1e-5,
                  w=np.linspace(.5, 1.5, 6), b=np.linspace(-.5, .5, 6), xd="float32", wd="float32", bd="float16"),
             dict(id="x f32, w f16, b f32", x=np.linspace(-1, 1, 12).reshape(2, 6), G=3, eps=1e-5,
                  w=np.linspace(.5, 1.5, 6), b=np.linspace(-.5, .5, 6), xd="float32", wd="float16", bd="float32")]
    res = {}
    for name, path in V.items():
        mod = S.load_nn(path)
        same = 0; diff = []
        for cs in cases:
            a = forward(cand, cs); b = forward(mod, cs)
            eq = a[0] == b[0] and a[1] == b[1] and (a[2] is None or np.array_equal(a[2], b[2]))
            same += eq
            if not eq:
                diff.append(cs["id"])
        outside = {}
        for cs in extra:
            a = forward(cand, cs); b = forward(mod, cs)
            outside[cs["id"]] = dict(candidate=a[:2], variant=b[:2])
        res[name] = dict(registered_ids_same_forward=same, registered_ids_differing=diff[:5],
                         n_registered=len(cases), outside_census=outside)
    out["variants"] = res
    # V3 on a non-float input: the stand-in has no int dtype path, so state the engine facts instead
    out["V3_note"] = ("Engine: DISPATCH_FLOAT refuses int64/bool/uint8 in reduce/ew ops "
                      "(kernels.cu:567-573), so the candidate refuses an int64 x inside g.mean; "
                      "V3 would cast it to float32 (ops::cast) and return a truncated int64 output. "
                      "No registered test uses a non-float x.")
    val = {}
    for args in ((-2, 6), (0, 6), (2.0, 6), (6, 6)):
        try:
            layer = cand.GroupNorm(*args)
            msg = "constructed"
            try:
                y = layer(S.Tensor(np.arange(6.).reshape(1, 6), "float32"))
                msg += f"; forward ok shape {list(y._d.shape)}"
            except Exception as exc:
                msg += f"; forward: {type(exc).__name__}: {exc}"
        except Exception as exc:
            msg = f"constructor: {type(exc).__name__}: {exc}"
        val[f"GroupNorm{args}"] = msg
    try:
        layer = cand.GroupNorm(2, 6)
        layer(S.Tensor(np.arange(8.).reshape(1, 4, 2), "float32"))
        val["GroupNorm(2,6) on C=4 input"] = "accepted"
    except Exception as exc:
        val["GroupNorm(2,6) on C=4 input"] = f"{type(exc).__name__}: {exc}"
    # numel_of on [1,-2,-3] (kernels.cu:80-88): no zero extent, product of signs = 6 = numel -> reshape accepted
    n = 1
    for s in (1, -2, -3):
        n *= s
    val["engine_numel_of([1,-2,-3])"] = n
    out["input_validation"] = val
    (HERE / "decisions_probe_results.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
