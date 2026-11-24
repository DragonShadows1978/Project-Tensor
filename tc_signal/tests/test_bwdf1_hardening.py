"""BWD-F1 Part H in tc_signal (orders/BWDF1_ZERO_EXTENT_AND_AUDIT_FIXES.md), CPU only (CUDA_VISIBLE_DEVICES empty,
TensorCUDA host tensors).

H1 (audit BWDA-F06): every op with a tracked path (conv1d, conv_transpose1d, instance_norm1d, upsample_nearest1d,
reflection_pad1d), called under grad mode with frozen inputs and an `out=` TensorCUDA destination that requires
grad, raises RuntimeError "tc_signal: <op>: out cannot require grad" -- the refused ops' rule (rint is the
reference row) -- with no CUDA call (ENG3's counter, tensor_cuda/tests/support/cudart_count.c). Controls: grad mode
off, or a destination that does not require grad, is not refused by H1. m04 (H1's check removed) fails the H1 rows.
H3 (audit BWDA-F08): the private backward entry's shape refusals name the operand and its shape (and the upstream
one the expected shape); the old text stays as the prefix and the type stays ValueError.

Environment: BWDF1_PKG_PARENT (directory holding the tensor_cuda package) and BWDF1_TS_PARENT (directory holding the
tc_signal package) select the builds under test; defaults are this worktree's packages.
Prior art: ENG3's LD_PRELOAD call counter; DeMillo, Lipton and Sayward (1978), mutation testing. Ours: the cases.
"""
import json
import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PKG_PARENT = os.environ.get("BWDF1_PKG_PARENT", os.path.join(ROOT, "tensor_cuda"))
TS_PARENT = os.environ.get("BWDF1_TS_PARENT", os.path.dirname(HERE))
SHIM = os.path.join(ROOT, "tensor_cuda", "tests", "support", "cudart_count.c")

TRACKED = ["conv1d", "conv_transpose1d", "instance_norm1d", "upsample_nearest1d", "reflection_pad1d"]
H1_CASES = [{"id": f"{op}-{mode}", "op": op, "mode": mode} for op in TRACKED + ["rint"]
            for mode in ("grad_out_rg", "nograd_out_rg", "grad_out_frozen")]
H3_CASES = [
    {"id": "conv-dy-shape", "args": [False, [1, 1, 10], [1, 1, 3], [1, 1, 9]],
     "want": "tc_signal: convolution backward: unexpected upstream shape: dy has shape [1, 1, 9], expected [1, 1, 8] "
             "for x [1, 1, 10] and w [1, 1, 3]"},
    {"id": "convT-dy-shape", "args": [True, [1, 1, 10], [1, 1, 3], [1, 1, 13]],
     "want": "tc_signal: convolution backward: unexpected upstream shape: dy has shape [1, 1, 13], expected "
             "[1, 1, 12] for x [1, 1, 10] and w [1, 1, 3]"},
    {"id": "x-rank", "args": [False, [1, 10], [1, 1, 3], [1, 1, 8]],
     "want": "tc_signal: convolution backward: expected positive rank-3 shapes: x has shape [1, 10]"},
    {"id": "w-zero", "args": [False, [1, 1, 10], [1, 0, 3], [1, 1, 8]],
     "want": "tc_signal: convolution backward: expected positive rank-3 shapes: w has shape [1, 0, 3]"},
    {"id": "dy-zero", "args": [False, [1, 1, 10], [1, 1, 3], [1, 0, 8]],
     "want": "tc_signal: convolution backward: expected positive rank-3 shapes: dy has shape [1, 0, 8]"},
]

_CHILD = r'''
import ctypes, json, os, sys
sys.path.insert(0, os.environ["BWDF1_PKG_PARENT"])
sys.path.insert(0, os.environ["BWDF1_TS_PARENT"])
import numpy as np
import tensor_cuda as tc
import tc_signal as ts
L = ctypes.CDLL(None)
L.eng3_cudart_calls.restype = ctypes.c_ulong
h1, h3 = json.loads(os.environ["BWDF1_H1"]), json.loads(os.environ["BWDF1_H3"])

def host(shape, rg=False):
    return tc.tensor(np.ones(shape, np.float32), device="cpu", requires_grad=rg)

CALLS = {
    "conv1d": (lambda out: ts.conv1d(host((1, 2, 8)), host((3, 2, 3)), host((3,)), out=out), (1, 3, 6)),
    "conv_transpose1d": (lambda out: ts.conv_transpose1d(host((1, 2, 8)), host((2, 3, 3)), out=out), (1, 3, 10)),
    "instance_norm1d": (lambda out: ts.instance_norm1d(host((1, 2, 8)), host((2,)), host((2,)), out=out), (1, 2, 8)),
    "upsample_nearest1d": (lambda out: ts.upsample_nearest1d(host((1, 2, 8)), 2, out=out), (1, 2, 16)),
    "reflection_pad1d": (lambda out: ts.reflection_pad1d(host((1, 2, 8)), (1, 2), out=out), (1, 2, 11)),
    "rint": (lambda out: ts.rint(host((1, 2, 8)), out=out), (1, 2, 8)),
}
res = {}
for c in h1:
    fn, oshape = CALLS[c["op"]]
    before = L.eng3_cudart_calls()
    try:
        if c["mode"] == "nograd_out_rg":
            with tc.no_grad():
                fn(host(oshape, True))
        else:
            fn(host(oshape, c["mode"] == "grad_out_rg"))
        r = {"raised": None}
    except BaseException as e:  # noqa: BLE001
        r = {"raised": type(e).__name__, "message": str(e)}
    r["calls"] = L.eng3_cudart_calls() - before
    res[c["id"]] = r
for c in h3:
    try:
        ts._C._conv_backward_scratch(*c["args"], (1, 0, 0, 0, 1, 1))
        res[c["id"]] = {"raised": None}
    except BaseException as e:  # noqa: BLE001
        res[c["id"]] = {"raised": type(e).__name__, "message": str(e)}
base = L.eng3_cudart_calls()
try:
    tc.tensor([1.0], device="cuda")
except RuntimeError:
    pass
print(json.dumps({"results": res, "control_calls": L.eng3_cudart_calls() - base}))
'''


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    compiler = next((p for n in ("cc", "gcc", "clang") if (p := shutil.which(n))), None)
    if compiler is None:
        pytest.skip("no C compiler for the CUDA-call counter shim")
    d = tmp_path_factory.mktemp("bwdf1_ts")
    shim = d / "cudart_count.so"
    r = subprocess.run([compiler, "-shared", "-fPIC", "-O2", "-o", str(shim), SHIM, "-ldl"], capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_PRELOAD=str(shim), BWDF1_PKG_PARENT=PKG_PARENT,
               BWDF1_TS_PARENT=TS_PARENT, BWDF1_H1=json.dumps(H1_CASES), BWDF1_H3=json.dumps(H3_CASES))
    p = subprocess.run([sys.executable, "-B", "-P", "-c", _CHILD], env=env, capture_output=True, text=True,
                       timeout=300)
    assert p.returncode == 0, p.stderr[-3000:]
    out = json.loads(p.stdout.strip().splitlines()[-1])
    assert out["control_calls"] > 0, out
    rec = os.environ.get("BWDF1_RECORD_DIR")
    if rec:
        os.makedirs(rec, exist_ok=True)
        with open(os.path.join(rec, "g1_ts.json"), "w") as f:
            json.dump(out, f, indent=1)
    return out["results"]


@pytest.mark.parametrize("op", TRACKED + ["rint"])
def test_cpu_h1_frozen_inputs_refuse_destination_requiring_grad(results, op):
    r = results[f"{op}-grad_out_rg"]
    assert r["raised"] == "RuntimeError", r
    assert r["message"] == f"tc_signal: {op}: out cannot require grad", r
    assert r["calls"] == 0, r


@pytest.mark.parametrize("op", TRACKED + ["rint"])
@pytest.mark.parametrize("mode", ["nograd_out_rg", "grad_out_frozen"])
def test_cpu_h1_controls_not_refused_by_the_destination_rule(results, op, mode):
    r = results[f"{op}-{mode}"]
    # Without a device the call continues to the DLPack export of a host tensor (BufferError) or further; it is
    # only required not to be the destination refusal.
    assert "out cannot require grad" not in r.get("message", ""), r


@pytest.mark.parametrize("case", H3_CASES, ids=[c["id"] for c in H3_CASES])
def test_cpu_h3_backward_entry_messages_name_operands(results, case):
    r = results[case["id"]]
    assert r["raised"] == "ValueError", r
    assert r["message"] == case["want"], r
