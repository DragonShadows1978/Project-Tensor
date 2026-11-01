#!/usr/bin/env python3
"""GN1-REV3 Q6: independent single-site and call-phase witness-reach check for every A3
mutant (52: V1-V4, M01-M18 and the 21 D5 mutants). Adapted from REV2 q5_mutant_trace.py (2026).

For each mutant: (1) diff the arm's nn.py against the candidate's with
difflib and require exactly one removed and one added line inside class
GroupNorm; (2) run its registered witness test ID from the sealed test file
under pytest 9, with the arm's nn.py loaded under A3's NumPy engine stand-in,
tracing executed lines of that nn.py (sys.settrace); require the edited line
to execute; (3) record the witness outcome on the mutant and on the candidate
(stand-in evidence, not a GPU kill). Then (4) check V3's equivalence claim by
running every engine dtype (the six dtype_name strings in kernels.cu) for x,
weight and bias through the candidate and V3, comparing outcomes exactly.

Prior art: reachability-infection-propagation (RIP) model of mutation
analysis (Offutt; Voas, early 1990s; unverified - lead to check: "RIP model
mutation testing Voas PIE"); DeMillo, Lipton and Sayward (1978). Ours: the
harness.
"""
import difflib
import importlib.util
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
OUT = Path(__file__).resolve().parent
MROOT = Path("/tmp/gn1_rev3_m/W")
MA3 = MROOT / "artifacts/gn1/a3"
import pytest  # noqa: E402

spec = importlib.util.spec_from_file_location("a3_cpu_shim", MA3 / "cpu_shim.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
TEST = MROOT / "tensor_cuda/tests/test_groupnorm_gn1.py"
CAND = MA3 / "arms/candidate/tensor_cuda/nn.py"
cand_lines = CAND.read_text().splitlines(True)
gn_lo = next(i for i, l in enumerate(cand_lines, 1) if l.startswith("class GroupNorm("))
gn_hi = next(i for i, l in enumerate(cand_lines, 1) if l.startswith("class InstanceNorm2D("))
mutants = json.loads((MA3 / "mutants.json").read_text())


class Tracer:
    def __init__(self, path):
        self.path = str(path)
        self.lines = set()

    def global_trace(self, frame, event, arg):
        if frame.f_code.co_filename == self.path:
            return self.local_trace
        return None

    def local_trace(self, frame, event, arg):
        if event == "line":
            self.lines.add(frame.f_lineno)
        return self.local_trace

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item):
        sys.settrace(self.global_trace)
        try:
            yield
        finally:
            sys.settrace(None)


def run(nn_path, node, trace=True):
    S.install(nn_path)
    t = Tracer(nn_path)
    rc = int(pytest.main(["-q", "-p", "no:cacheprovider", "--import-mode=importlib", "--rootdir=" + str(MROOT),
                          "--basetemp=/tmp/gn1_rev3_tmp/q5", str(TEST) + "::" + node], plugins=[t] if trace else []))
    return rc, t.lines


rows = []
for m in mutants:
    nn = MA3 / "arms" / m["id"] / "tensor_cuda/nn.py"
    new = nn.read_text().splitlines(True)
    sm = difflib.SequenceMatcher(a=cand_lines, b=new, autojunk=False)
    ops = [op for op in sm.get_opcodes() if op[0] != "equal"]
    removed = sum(i2 - i1 for _, i1, i2, _, _ in ops)
    added = sum(j2 - j1 for _, _, _, j1, j2 in ops)
    edited = [j1 + 1 for _, _, _, j1, j2 in ops for _ in range(j2 - j1)]  # 1-based lines in the mutant
    inside = all(gn_lo <= i1 + 1 < gn_hi for _, i1, _, _, _ in ops)
    node = m["predicted_witness"].split("::", 1)[1]
    rc_m, lines = run(nn, node)
    rc_c, _ = run(CAND, node, trace=False)
    reached = bool(edited) and all(e in lines for e in edited)
    rows.append(dict(id=m["id"], witness=node, hunks=len(ops), removed_lines=removed, added_lines=added,
                     single_site=len(ops) == 1 and removed == 1 and added == 1, inside_groupnorm=inside,
                     edited_line=edited, witness_reaches_edit=reached, witness_exit_on_mutant=rc_m,
                     witness_exit_on_candidate=rc_c, equivalence=m.get("equivalence"),
                     old=m["old"][:90], new=m["new"][:90]))
    print(m["id"], rows[-1]["single_site"], reached, rc_m, rc_c, flush=True)

# V3 equivalence: every engine dtype string for x, weight, bias (6^3), forward + backward.
DTYPES = ("float32", "float16", "bfloat16", "int64", "bool", "uint8")
tc_c = S.install(CAND)
nn_c = tc_c.nn
tc_v = S.install(MA3 / "arms/V3/tensor_cuda/nn.py")
nn_v = tc_v.nn


def go(nn_mod, xd, wd, bd, backward):
    try:
        layer = nn_mod.GroupNorm(3, 6)
        rng = np.random.default_rng(3)
        layer.weight = S.Tensor(rng.uniform(-1, 1, 6), wd, requires_grad=True)
        layer.bias = S.Tensor(rng.uniform(-1, 1, 6), bd, requires_grad=True)
        x = S.Tensor(rng.uniform(-1, 1, (2, 6, 4)), xd, requires_grad=True)
        y = layer(x)
        out = [y.dtype, y.numpy().astype(np.float64).tobytes()]
        if backward:
            y.sum().backward()
            out += [None if t.grad is None else (t.grad.dtype, t.grad.numpy().astype(np.float64).tobytes())
                    for t in (x, layer.weight, layer.bias)]
        return ("ok", out)
    except Exception as exc:  # noqa: BLE001
        return ("raise", type(exc).__name__, str(exc))


v3 = []
for xd, wd, bd in itertools.product(DTYPES, DTYPES, DTYPES):
    for backward in (False, True):
        a, b = go(nn_c, xd, wd, bd, backward), go(nn_v, xd, wd, bd, backward)
        v3.append(dict(case=f"{xd}/{wd}/{bd}/{'bwd' if backward else 'fwd'}", identical=a == b, candidate=a[0]))
# M14: the edited line is a def line (default evaluated at class definition); check the binding in a call.
tc_m = S.install(MA3 / "arms/M14/tensor_cuda/nn.py")
m14_eps = tc_m.nn.GroupNorm(2, 6).eps
tc_c2 = S.install(CAND)
cand_eps = tc_c2.nn.GroupNorm(2, 6).eps
m14 = dict(mutant_default_eps=m14_eps, candidate_default_eps=cand_eps, binding_differs=m14_eps != cand_eps)
summary = dict(m14_default_binding=m14, mutants=len(rows), single_site_all=all(r["single_site"] for r in rows),
               inside_groupnorm_all=all(r["inside_groupnorm"] for r in rows),
               reach_all=all(r["witness_reaches_edit"] for r in rows),
               not_reached=[r["id"] for r in rows if not r["witness_reaches_edit"]],
               witness_fails_on_mutant=[r["id"] for r in rows if r["witness_exit_on_mutant"] == 1],
               witness_passes_on_mutant=[r["id"] for r in rows if r["witness_exit_on_mutant"] == 0],
               witness_fails_on_candidate=[r["id"] for r in rows if r["witness_exit_on_candidate"] != 0],
               v3_cases=len(v3), v3_all_identical=all(r["identical"] for r in v3),
               v3_accepted_cases=sum(r["candidate"] == "ok" for r in v3))
(OUT / "q5_mutant_trace_rev3.json").write_text(json.dumps(dict(summary=summary, mutants=rows, v3=v3), indent=1) + "\n")
print(json.dumps(summary, indent=1))
