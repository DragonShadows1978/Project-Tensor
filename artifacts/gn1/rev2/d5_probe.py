#!/usr/bin/env python3
"""GN1-REV2 Q2: does A2's GroupNorm refuse exactly the inputs outside its
domain, and still accept everything A1 accepted inside it?

Runs the REAL candidate nn.py (A2 arm, sealed bytes) and A1's nn.py
(git show 0c6a4244, read-only) under A2's own NumPy engine stand-in
(a2/cpu_shim.py, which reuses REV1's gn_emulate rounding). The stand-in is not
the CUDA engine: this probes the Python class's decisions, not device numerics.
Every A1-accepted configuration must be accepted by A2 with bit-identical
outputs and gradients (the D5 checks precede and do not alter computation).
PyTorch (CPU, installed) is the external reference for the domain questions.

Prior art: boundary-value / equivalence-partition testing (Myers, The Art of
Software Testing, 1979; unverified - lead to check); PyTorch GroupNorm argument
checks (observed live). Ours: the matrix.
"""
import importlib.util
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
OUT = Path(__file__).resolve().parent
REAL = OUT.parents[2]
MA2 = Path("/tmp/gn1_rev2_regen/W/artifacts/gn1/a2")
SCR = Path("/tmp/gn1_rev2_tmp/d5")
SCR.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("a2_cpu_shim", MA2 / "cpu_shim.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
a1_path = SCR / "a1" / "nn.py"
a1_path.parent.mkdir(parents=True, exist_ok=True)
a1_path.write_bytes(subprocess.check_output(["git", "-C", str(REAL), "show", "0c6a4244:tensor_cuda/tensor_cuda/nn.py"]))
TC_A1 = S.install(a1_path)
NN = {"A1": TC_A1.nn}
TC_A2 = S.install(MA2 / "arms/candidate/tensor_cuda/nn.py")
NN["A2"] = TC_A2.nn
Tensor = S.Tensor
FLOATS = ("float32", "float16", "bfloat16")
rows = []


def outcome(fn):
    try:
        v = fn()
        return dict(outcome="accepted", value=v)
    except Exception as exc:  # noqa: BLE001 - the class is the observation
        return dict(outcome="refused", exception_class=type(exc).__name__, message=str(exc)[:160])


def ctor(version, g, c):
    return outcome(lambda: (NN[version].GroupNorm(g, c), None)[1])


# 1. Constructor domain
values = [("zero", 0), ("negative", -2), ("float", 2.0), ("float_nonint", 2.5), ("true", True), ("false", False),
          ("string", "2"), ("none", None), ("np_int64", np.int64(2)), ("np_int32", np.int32(2)),
          ("np_uint8", np.uint8(2)), ("huge", 2 ** 70)]
try:
    import torch
    torch.set_num_threads(1)
    TORCH = torch.__version__
except ImportError:
    torch, TORCH = None, None


def torch_ctor_forward(g, c):
    def go():
        layer = torch.nn.GroupNorm(g, c)
        n = int(c) if not isinstance(c, (str, type(None))) else 6
        return tuple(layer(torch.ones((2, n, 4))).shape)
    return outcome(go)


for which in ("num_groups", "num_channels"):
    for label, v in values:
        args = dict(num_groups=2, num_channels=6)
        if which == "num_channels" and label == "huge":
            continue  # would allocate 2**70 parameters if accepted
        args[which] = v
        if which == "num_channels" and label in ("np_int64", "np_int32", "np_uint8"):
            args[which] = type(v)(6)
        if which == "num_groups" and label == "huge":
            args["num_channels"] = 2 ** 70  # divisible; refused or accepted before allocation?
            continue
        row = dict(section="constructor", case=which + "/" + label,
                   A1=ctor("A1", args["num_groups"], args["num_channels"]),
                   A2=ctor("A2", args["num_groups"], args["num_channels"]))
        if torch is not None:
            row["torch_ctor_and_forward"] = torch_ctor_forward(args["num_groups"], args["num_channels"])
        rows.append(row)


# 2. Forward acceptance: every x/weight/bias float dtype combination, forward and backward.
def build(version, xd, wd, bd, shape=(2, 6, 4), groups=3, wgrad=True, bgrad=True, wshape=None):
    rng = np.random.default_rng(7)
    x = rng.uniform(-1, 1, shape) + 0.25 * np.arange(shape[1]).reshape((1, shape[1]) + (1,) * (len(shape) - 2))
    w = np.array([0.5, -0.75, 1.25, -1.5, 0.875, 1.125][:shape[1]] if shape[1] <= 6 else np.ones(shape[1]))
    b = np.array([-0.375, 0.625, 0.25, -0.5, 0.75, -0.125][:shape[1]] if shape[1] <= 6 else np.zeros(shape[1]))
    layer = NN[version].GroupNorm(groups, shape[1])
    layer.weight = Tensor(w.reshape(wshape) if wshape else w, wd, requires_grad=wgrad)
    layer.bias = Tensor(b, bd, requires_grad=bgrad)
    xt = Tensor(x, xd, requires_grad=True)
    return layer, xt


def run_cfg(version, xd, wd, bd, backward, **kw):
    def go():
        layer, xt = build(version, xd, wd, bd, **kw)
        y = layer(xt)
        res = dict(dtype=y.dtype, y=y.numpy().astype(np.float64))
        if backward:
            (y * Tensor(np.full(y.shape, 0.5), y.dtype)).sum().backward()
            res.update({k: (None if t.grad is None else (t.grad.dtype, t.grad.numpy().astype(np.float64)))
                        for k, t in (("x", xt), ("w", layer.weight), ("b", layer.bias))})
        return res
    return outcome(go)


def same(a, b):
    if a["outcome"] != b["outcome"]:
        return False
    if a["outcome"] == "refused":
        return a["exception_class"] == b["exception_class"]
    va, vb = a["value"], b["value"]
    if va.keys() != vb.keys():
        return False
    for k in va:
        x, y = va[k], vb[k]
        if isinstance(x, tuple):
            if x[0] != y[0] or not np.array_equal(x[1], y[1], equal_nan=True):
                return False
        elif isinstance(x, np.ndarray):
            if not np.array_equal(x, y, equal_nan=True):
                return False
        elif x != y:
            return False
    return True


accept_rows = []
for xd, wd, bd in itertools.product(FLOATS, FLOATS, FLOATS):
    for backward in (False, True):
        a1 = run_cfg("A1", xd, wd, bd, backward)
        a2 = run_cfg("A2", xd, wd, bd, backward)
        ok_dtypes = (a2["outcome"] == "accepted" and a2["value"]["dtype"] == xd and
                     (not backward or (a2["value"]["x"][0] == xd and a2["value"]["w"][0] == wd and a2["value"]["b"][0] == bd)))
        accept_rows.append(dict(section="accept", case=f"x={xd},w={wd},b={bd},{'bwd' if backward else 'fwd'}",
                                A1=a1["outcome"], A2=a2["outcome"], A2_dtypes_ok=ok_dtypes,
                                A2_identical_to_A1=same(a1, a2) if a1["outcome"] == "accepted" else None))
for xd in FLOATS:  # parameters that need no gradient
    for wgrad, bgrad in ((False, False), (False, True), (True, False)):
        a1 = run_cfg("A1", xd, "float32", "float32", True, wgrad=wgrad, bgrad=bgrad)
        a2 = run_cfg("A2", xd, "float32", "float32", True, wgrad=wgrad, bgrad=bgrad)
        accept_rows.append(dict(section="accept", case=f"x={xd},w/b float32,no-grad w={not wgrad},b={not bgrad}",
                                A1=a1["outcome"], A2=a2["outcome"], A2_identical_to_A1=same(a1, a2)))
for shape, groups in (((2, 6), 2), ((1, 6), 6), ((2, 6, 2, 3), 2), ((1, 6, 2, 2, 3), 1)):
    for xd in FLOATS:
        a1 = run_cfg("A1", xd, xd, xd, True, shape=shape, groups=groups)
        a2 = run_cfg("A2", xd, xd, xd, True, shape=shape, groups=groups)
        accept_rows.append(dict(section="accept", case=f"shape={shape},G={groups},{xd}", A1=a1["outcome"],
                                A2=a2["outcome"], A2_identical_to_A1=same(a1, a2)))
rows += accept_rows


# 3. Forward refusals (A2 must refuse with the registered class) and parameter shapes.
def meta_refusal(version, x_shape=(2, 6, 4), xd="float32", wd="float32", bd="float32"):
    def go():
        layer, _ = build(version, "float32", wd, bd)
        x = Tensor(np.ones(x_shape), xd, requires_grad=False)
        y = layer(x)
        return dict(dtype=y.dtype, shape=list(y.shape))
    return outcome(go)


for name, kw in (("rank0", dict(x_shape=())), ("rank1", dict(x_shape=(6,))), ("channels_rank2", dict(x_shape=(2, 4))),
                 ("channels_rank3", dict(x_shape=(2, 4, 3))), ("channels_rank4_more", dict(x_shape=(2, 12, 2, 2)))):
    rows.append(dict(section="refuse", case=name, A1=meta_refusal("A1", **kw), A2=meta_refusal("A2", **kw)))
for dt in ("int64", "bool", "uint8"):
    for operand in ("input", "weight", "bias"):
        kw = {"input": dict(xd=dt), "weight": dict(wd=dt), "bias": dict(bd=dt)}[operand]
        rows.append(dict(section="refuse", case=f"{dt}/{operand}", A1=meta_refusal("A1", **kw),
                         A2=meta_refusal("A2", **kw)))
for label, wshape in (("weight_shape_2x3", (2, 3)), ("weight_shape_6x1", (6, 1)), ("weight_shape_1x6x1", (1, 6, 1))):
    a2 = run_cfg("A2", "float32", "float32", "float32", False, wshape=wshape)
    row = dict(section="param_shape", case=label, A2=dict(outcome=a2["outcome"], **(
        {"exception_class": a2.get("exception_class"), "message": a2.get("message")} if a2["outcome"] == "refused" else {})))
    if torch is not None:
        def tgo(wshape=wshape):
            layer = torch.nn.GroupNorm(3, 6)
            layer.weight = torch.nn.Parameter(torch.ones(wshape))
            return tuple(layer(torch.ones((2, 6, 4))).shape)
        t = outcome(tgo)
        row["torch"] = {k: v for k, v in t.items() if k != "value"}
    rows.append(row)


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items() if k != "value"}
    return o


rows = [clean(r) for r in rows]
summary = dict(
    torch_version=TORCH,
    constructor_np_integer_refused_by_A2=[r["case"] for r in rows if r["section"] == "constructor" and "np_" in r["case"]
                                          and r["A2"]["outcome"] == "refused"],
    accept_all_A1_accepted_also_A2=all(r["A2"] == "accepted" for r in rows if r["section"] == "accept" and r["A1"] == "accepted"),
    accept_identical_outputs=all(r.get("A2_identical_to_A1") for r in rows if r["section"] == "accept" and r["A1"] == "accepted"),
    accept_A2_dtypes_ok=all(r.get("A2_dtypes_ok", True) for r in rows if r["section"] == "accept"),
    accept_rows=len(accept_rows),
    A1_refused_in_accept_matrix=[r["case"] for r in rows if r["section"] == "accept" and r["A1"] != "accepted"],
    refusals_A2=[(r["case"], r["A2"].get("exception_class")) for r in rows if r["section"] == "refuse"],
    param_shape=[(r["case"], r["A2"]["outcome"], (r.get("torch") or {}).get("outcome")) for r in rows if r["section"] == "param_shape"],
)
(OUT / "d5_probe.json").write_text(json.dumps(dict(summary=summary, rows=rows), indent=1, default=str) + "\n")
print(json.dumps(summary, indent=1, default=str))
