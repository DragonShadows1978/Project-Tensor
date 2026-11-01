#!/usr/bin/env python3
"""GN1-REV1 L41 trace: run each arm's REAL nn.py GroupNorm (candidate and
M01-M18, byte-identical sealed files) on its registered witness operands,
with the engine replaced by a NumPy stand-in that mirrors the engine's
forward semantics used by GroupNorm (reshape element-count check, ew_binary
dtype-mismatch refusal, numpy-style broadcasting refusal, per-op rounding to
the tensor dtype with float32 reductions, cast). Records, with sys.settrace,
which nn.py lines execute, and whether each mutant's edited line runs.

This is a call-path trace of the Python code under a stand-in engine, NOT an
engine run (the engine cannot allocate without a GPU). Forward only; the M15
witness is a backward ID, whose overflow is shown in numerics_results.json.

Prior art: sys.settrace line tracing (CPython); DeMillo, Lipton and Sayward
(1978) and the RIP model (reachability-infection-propagation; Offutt/Voas,
1990s, unverified - lead to check "RIP model mutation testing") for "the
witness must reach the mutated line". Ours: the stand-in and the wiring.
"""
import difflib
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
A1 = ROOT / "artifacts/gn1/a1"
sys.path.insert(0, str(HERE))
import gn_emulate as E  # noqa: E402


class Tensor:
    def __init__(self, data, dtype="float32", requires_grad=False):
        self._d = E.rnd(np.asarray(data, np.float64), dtype)
        self.dtype = dtype
        self.requires_grad = requires_grad

    shape = property(lambda self: list(self._d.shape))

    def numpy(self):
        return self._d.copy()

    def _new(self, d):
        return Tensor(d, self.dtype)

    def reshape(self, s):
        s = [int(v) for v in s]
        prod = 1
        for v in s:
            prod *= v
        if prod != self._d.size:
            raise RuntimeError("reshape: element count mismatch")
        if any(v < 0 for v in s):
            raise RuntimeError("stand-in: negative extent reached the engine (the engine itself would accept it)")
        return self._new(self._d.reshape(s))

    def permute(self, d):
        return self._new(np.transpose(self._d, d))

    def mean(self, axes, keep):
        cnt = int(np.prod([self._d.shape[a] for a in axes]))
        s = self._d
        for a in sorted(axes, reverse=True):
            s = E.rsum(s, a, True, self.dtype)
        if not keep:
            s = s.squeeze(tuple(axes))
        return self._new(E.rnd(s * E.rnd(1.0 / cnt, "float32"), self.dtype))

    def var(self, axes, keep):
        m = self.mean(axes, True)
        d = self - m
        return (d * d).mean(axes, keep)

    def pow(self, p):
        with np.errstate(all="ignore"):
            return self._new(np.power(self._d, p))

    def astype(self, dt):
        return self if dt == self.dtype else Tensor(self._d, dt)

    def _bin(self, o, f):
        if isinstance(o, Tensor):
            if o.dtype != self.dtype:
                raise RuntimeError("ew_binary dtype mismatch")
            try:
                shape = np.broadcast_shapes(self._d.shape, o._d.shape)
            except ValueError as exc:
                raise RuntimeError("broadcast: " + str(exc))
            return self._new(f(self._d, o._d))
        return self._new(f(self._d, E.rnd(float(o), "float32")))

    __add__ = lambda s, o: s._bin(o, np.add)
    __radd__ = __add__
    __sub__ = lambda s, o: s._bin(o, np.subtract)
    __mul__ = lambda s, o: s._bin(o, np.multiply)
    __rmul__ = __mul__


def tensor(array, device="cuda", dtype="float32", requires_grad=False):
    return Tensor(array, dtype, requires_grad)


def load_nn(path):
    shim = types.ModuleType("tensor_cuda")
    shim.Tensor, shim.tensor = Tensor, tensor
    sys.modules["tensor_cuda"] = shim
    spec = importlib.util.spec_from_file_location("nn_under_test_" + path.parent.parent.name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_witness(nnpath, wid, numerics_cases):
    nn = load_nn(nnpath)
    import run_numerics as RN  # noqa: E402  (input census reused; read-only)
    T = RN.T
    executed = set()
    target = str(nnpath)

    def tracer(frame, event, arg):
        if frame.f_code.co_filename == target:
            if event == "line":
                executed.add(frame.f_lineno)
            return tracer
        return None

    short = wid.split("::", 1)[1]
    outcome = {}
    sys.settrace(tracer)
    try:
        if short == "test_nondivisible_refusal":
            try:
                nn.GroupNorm(4, 6)
                outcome = dict(killed=True, how="no ValueError raised")
            except ValueError:
                outcome = dict(killed=False, how="ValueError raised")
        elif short == "test_constructor_defaults":
            layer = nn.GroupNorm(2, 6)
            outcome = dict(killed=layer.eps != 1e-5, how=f"eps={layer.eps}")
        else:
            cs = numerics_cases[short]
            layer = nn.GroupNorm(cs["G"], cs["x"].shape[1]) if cs["eps"] == 1e-5 else nn.GroupNorm(cs["G"], cs["x"].shape[1], eps=cs["eps"])
            if short.startswith(("test_mixed_parameters", "test_low_variance", "test_fp32_input_half")):
                if "half" in short:
                    layer.weight = tensor(cs["w"], dtype="float32", requires_grad=True)
                    layer.bias = tensor(cs["b"], dtype="float32", requires_grad=True)
                    layer.weight = Tensor(cs["w"], "float16", True); layer.bias = Tensor(cs["b"], "float16", True)
            else:
                layer.weight = Tensor(cs["w"], cs["wd"], True); layer.bias = Tensor(cs["b"], cs["bd"], True)
            x = Tensor(cs["x"], cs["xd"], True)
            try:
                out = layer(x)
                ref = T.groupnorm_reference(cs["x"], cs["G"], cs["w"], cs["b"], cs["eps"])
                atol, rtol = T.TOLERANCES[cs["xd"]]["forward"]
                same_shape = list(out._d.shape) == list(ref.shape)
                ok = (out.dtype == cs["xd"] and same_shape and np.all(np.isfinite(out._d))
                      and np.allclose(out._d, ref, atol=atol, rtol=rtol))
                outcome = dict(killed=not ok, how=f"dtype {out.dtype}, shape_ok {same_shape}, allclose {ok}")
            except Exception as exc:
                outcome = dict(killed=True, how=f"{type(exc).__name__}: {exc}")
    finally:
        sys.settrace(None)
    return executed, outcome


def main():
    import run_numerics as RN
    cases = {c["id"]: c for c in RN.cases()}
    muts = json.loads((A1 / "mutants.json").read_text())
    cand = A1 / "arms/candidate/tensor_cuda/nn.py"
    report = {}
    ex, oc = run_witness(cand, "x::test_forward_values[float32-r3-n2-g2-default]", cases)
    report["candidate_on_M04_witness"] = oc
    for m in muts:
        mid, wid = m["id"], m["predicted_witness"]
        path = A1 / "arms" / mid / "tensor_cuda/nn.py"
        a = cand.read_text().splitlines(); b = path.read_text().splitlines()
        edited = sorted({j + 1 for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes()
                         if tag != "equal" for j in range(j1, j2)})
        cand_ex, cand_oc = run_witness(cand, wid, cases)
        executed, outcome = run_witness(path, wid, cases)
        report[mid] = dict(witness=wid, edited_lines=edited,
                           edited_lines_executed=sorted(set(edited) & executed),
                           reached=bool(set(edited) & executed), mutant_outcome=outcome,
                           candidate_outcome_same_witness=cand_oc)
    (HERE / "shim_trace_results.json").write_text(json.dumps(report, indent=2) + "\n")
    for k, v in report.items():
        print(k, json.dumps(v)[:400])


if __name__ == "__main__":
    main()
