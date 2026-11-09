#!/usr/bin/env python3
"""GN1-REV3 Q4: D5 validation in A3's REAL candidate nn.py (sealed mirror arm)
against A2's nn.py (git show bd30498a, read-only), both under A3's NumPy engine
stand-in (a3/cpu_shim.py). Probes the Python class's decisions, not CUDA numerics.

1. Count domain: every value class; A3 must accept exactly what operator.index
   accepts minus bool and numpy.bool_, and store a builtin int.
2. Parameter shapes: refused with ValueError, and no stand-in tensor op runs in
   forward before the refusal (all Tensor methods are instrumented).
3. Every configuration A2 accepts (3x3x3 dtypes x 3 ranks x 4 group counts, forward
   and backward) is accepted by A3 with bit-identical output and gradients.
PyTorch (CPU, installed) observed for the count classes where it can be.

Prior art: boundary-value / equivalence-partition testing (Myers 1979, unverified -
lead to check); PEP 357 operator.index (2006); REV2 d5_probe.py (2026). Ours: the matrix.
"""
import importlib.util
import itertools
import json
import operator
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
OUT = Path(__file__).resolve().parent
REAL = OUT.parents[2]
MA3 = Path("/tmp/gn1_rev3_m/W/artifacts/gn1/a3")
SCR = Path("/tmp/gn1_rev3_tmp/d5")
SCR.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("a3_cpu_shim", MA3 / "cpu_shim.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
a2_path = SCR / "a2" / "nn.py"
a2_path.parent.mkdir(parents=True, exist_ok=True)
a2_path.write_bytes(subprocess.check_output(["git", "-C", str(REAL), "show", "bd30498a:tensor_cuda/tensor_cuda/nn.py"]))
NN = {"A2": S.install(a2_path).nn}
TC3 = S.install(MA3 / "arms/candidate/tensor_cuda/nn.py")
NN["A3"] = TC3.nn
Tensor = S.Tensor
FLOATS = ("float32", "float16", "bfloat16")
report = dict(counts=[], shapes=[], accepted=dict(), torch=None)
try:
    import torch
    torch.set_num_threads(1)
    report["torch"] = torch.__version__
except ImportError:
    torch = None


class Idx:
    def __init__(self, v):
        self.v = v

    def __index__(self):
        return self.v


def expect_accept(v):
    if isinstance(v, (bool, np.bool_)):
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            operator.index(v)
        return True
    except TypeError:
        return False


VALUES = [("int2", 2), ("zero", 0), ("negative", -2), ("float", 2.0), ("true", True), ("false", False), ("string", "2"),
          ("none", None), ("huge", 2 ** 70), ("np_int8", np.int8(2)), ("np_int16", np.int16(2)), ("np_int32", np.int32(2)),
          ("np_int64", np.int64(2)), ("np_uint8", np.uint8(2)), ("np_uint16", np.uint16(2)), ("np_uint32", np.uint32(2)),
          ("np_uint64", np.uint64(2)), ("np_bool_true", np.bool_(True)), ("np_bool_false", np.bool_(False)),
          ("np_float64", np.float64(2.0)), ("np_0d_int", np.array(2)), ("np_0d_bool", np.array(True)),
          ("np_1d_int", np.array([2])), ("index_obj", Idx(2)), ("index_obj_neg", Idx(-2)),
          ("index_returns_bool", Idx(True)), ("np_int64_zero", np.int64(0))]


def try_(fn):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return dict(outcome="accepted", value=fn())
    except Exception as exc:  # noqa: BLE001
        return dict(outcome="refused", exception_class=type(exc).__name__, message=str(exc)[:140])


for which in ("num_groups", "num_channels"):
    for name, v in VALUES:
        args = dict(num_groups=2, num_channels=6)
        args[which] = v
        row = dict(operand=which, value=name)
        for ver in ("A2", "A3"):
            r = try_(lambda: NN[ver].GroupNorm(**args))
            if r["outcome"] == "accepted":
                layer = r.pop("value")
                r["stored_types"] = [type(layer.G).__name__, type(layer.C).__name__]
                r["stored"] = [layer.G, layer.C]
            row[ver] = r
        exp = expect_accept(v) and int(operator.index(v)) > 0 if expect_accept(v) else False
        if which == "num_channels" and exp:
            exp = int(operator.index(v)) % 2 == 0
        if which == "num_groups" and exp:
            exp = 6 % int(operator.index(v)) == 0
        row["d5_expect_accept"] = exp
        a3 = row["A3"]
        row["a3_ok"] = (a3["outcome"] == "accepted") == exp and (
            a3["outcome"] != "accepted" or a3["stored_types"] == ["int", "int"]) and (
            a3["outcome"] == "accepted" or a3["exception_class"] == "ValueError")
        if torch is not None:
            t = try_(lambda: tuple(torch.nn.GroupNorm(**args)(torch.ones((2, 6 if which == "num_groups" else max(int(operator.index(v)) if expect_accept(v) else 6, 1), 4))).shape))
            row["torch"] = {k: t[k] for k in t if k != "value"}
        report["counts"].append(row)

# 2. Parameter shapes, with every stand-in Tensor method instrumented.
CALLS = []
for meth in ("reshape", "astype", "sum", "mean", "var", "pow", "_binary", "permute", "numpy", "detach", "backward", "_new"):
    orig = getattr(Tensor, meth)
    def wrap(self, *a, _o=orig, _m=meth, **k):
        CALLS.append(_m)
        return _o(self, *a, **k)
    setattr(Tensor, meth, wrap)
for operand, shape, dtype in itertools.product(("weight", "bias"), [(), (2, 3), (6, 1), (1, 6), (1, 6, 1), (5,), (7,), (0,), (6, 0)],
                                               ("float32", "float16")):
    layer = NN["A3"].GroupNorm(2, 6)
    setattr(layer, operand, Tensor(np.ones(shape), dtype))
    x = Tensor(np.ones((2, 6, 3)), "float32")
    CALLS.clear()
    r = try_(lambda: layer(x))
    r.pop("value", None)
    report["shapes"].append(dict(operand=operand, shape=list(shape), dtype=dtype, ops_before_refusal=list(CALLS), **r,
                                 ok=r["outcome"] == "refused" and r.get("exception_class") == "ValueError" and not CALLS
                                 and operand in r.get("message", "")))
for meth in ("reshape", "astype", "sum", "mean", "var", "pow", "_binary", "permute", "numpy", "detach", "backward", "_new"):
    setattr(Tensor, meth, getattr(Tensor, meth).__defaults__ and getattr(Tensor, meth) or getattr(Tensor, meth))
# restore original methods (reload shim classes cleanly for part 3)
spec2 = importlib.util.spec_from_file_location("a3_cpu_shim2", MA3 / "cpu_shim.py")
S = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(S)
NN = {"A2": S.install(a2_path).nn}
NN["A3"] = S.install(MA3 / "arms/candidate/tensor_cuda/nn.py").nn
Tensor = S.Tensor

# 3. Bit-identity on everything A2 accepts.
rng = np.random.default_rng(571)
same = 0
diffs = []
checked = 0
for xd, wd, bd in itertools.product(FLOATS, FLOATS, FLOATS):
    for shape in ((3, 6), (3, 6, 5), (2, 6, 3, 2)):
        for g in (1, 2, 3, 6):
            data = rng.standard_normal(shape) * 2 + 0.5
            wv = rng.standard_normal(6) + 1
            bv = rng.standard_normal(6)
            proj = rng.standard_normal(shape)
            res = {}
            for ver in ("A2", "A3"):
                layer = NN[ver].GroupNorm(g, 6)
                layer.weight = Tensor(wv, wd, requires_grad=True)
                layer.bias = Tensor(bv, bd, requires_grad=True)
                x = Tensor(data, xd, requires_grad=True)
                try:
                    y = layer(x)
                    loss = (y.astype("float32") * Tensor(proj, "float32")).sum()
                    loss.backward()
                    res[ver] = ("ok", y.numpy().tobytes(), y.dtype, x.grad.numpy().tobytes(),
                                layer.weight.grad.numpy().tobytes(), layer.bias.grad.numpy().tobytes())
                except Exception as exc:  # noqa: BLE001
                    res[ver] = ("refused", type(exc).__name__, str(exc)[:100])
            if res["A2"][0] == "ok":
                checked += 1
                if res["A2"] == res["A3"]:
                    same += 1
                else:
                    diffs.append(dict(dtypes=[xd, wd, bd], shape=shape, groups=g, a3=res["A3"][:3] if res["A3"][0] != "ok" else "values differ"))
report["accepted"] = dict(a2_accepted_configs=checked, a3_bit_identical=same, diffs=diffs)
summary = dict(torch=report["torch"],
               counts_rows=len(report["counts"]), counts_ok=sum(r["a3_ok"] for r in report["counts"]),
               counts_bad=[(r["operand"], r["value"], r["A3"]) for r in report["counts"] if not r["a3_ok"]],
               shape_rows=len(report["shapes"]), shapes_ok=sum(r["ok"] for r in report["shapes"]),
               shapes_bad=[(r["operand"], r["shape"], r["dtype"], r.get("exception_class"), r.get("message"), r["ops_before_refusal"]) for r in report["shapes"] if not r["ok"]],
               accepted=report["accepted"])
report["summary"] = summary
(OUT / "d5_probe_rev3.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
print(json.dumps(summary, indent=1, default=str))
