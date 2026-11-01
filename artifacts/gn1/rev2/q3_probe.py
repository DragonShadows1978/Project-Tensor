#!/usr/bin/env python3
"""GN1-REV2 Q3: the kill rule as the REAL pipeline produces it.

Runs real test IDs from the sealed test file, the REAL candidate nn.py (A2
arm) under A2's NumPy engine stand-in (a2/cpu_shim.py), and A2's REAL child
plugin (run_gpu_tests.Reports) inside pytest 9; then feeds the plugin's
reports to A2's REAL evidence.acceptable_killers() and DEVICE_PATTERNS veto.
Fault injection is done in the stand-in engine only (never in nn.py or the
tests). CPU only; no CUDA module is imported.

Cases
  P0 honest:   unmodified stand-in; the value test passes.
  P1 generic:  the engine's pow raises a generic RuntimeError inside
               GroupNorm.forward (no device text). G4's control
               Generic_RuntimeError_not_kill claims this is not a kill.
  P2 foreign:  the engine's tensor factory raises TypeError while
               GroupNorm.__init__ allocates its parameters (origin: engine,
               not a GroupNorm refusal); test_constructor_defaults.
  P3 device:   pow raises 'CUDA error at pow_kernel: an illegal memory access
               was encountered'; the wrapper re-raises as AssertionError.
Plus a census of every source file that can appear in a traceback for the 12
device patterns (case-folded), to bound false vetoes of honest kills.

Prior art: fault injection / test stubs (Meszaros 2007); mutation validity
(DeMillo, Lipton, Sayward 1978). Ours: the cases.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
OUT = Path(__file__).resolve().parent
MROOT = Path("/tmp/gn1_rev2_regen/W")
MA2 = MROOT / "artifacts/gn1/a2"
sys.path.insert(0, str(MA2))
import evidence  # noqa: E402  (sealed mirror copy)
import run_gpu_tests  # noqa: E402  (sealed mirror copy; main() not run)
import pytest  # noqa: E402

spec = importlib.util.spec_from_file_location("a2_cpu_shim", MA2 / "cpu_shim.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
TEST = MROOT / "tensor_cuda/tests/test_groupnorm_gn1.py"
VALUE_ID = "test_forward_values[float32-r3-n2-g2-default]"
CTOR_ID = "test_constructor_defaults"
results = {}
orig_pow = S.Tensor.pow


def run_case(name, node, patch=None):
    tc = S.install(MA2 / "arms/candidate/tensor_cuda/nn.py")
    S.Tensor.pow = orig_pow
    undo = patch(tc) if patch else None
    plugin = run_gpu_tests.Reports()
    rc = int(pytest.main(["-q", "-p", "no:cacheprovider", "--import-mode=importlib", "--rootdir=" + str(MROOT),
                          "--basetemp=/tmp/gn1_rev2_tmp/q3_" + name, str(TEST) + "::" + node], plugins=[plugin]))
    if undo:
        undo()
    S.Tensor.pow = orig_pow
    reps = [dict(r) for r in plugin.reports]
    data = dict(reports=reps)
    details = [r["detail"] for r in reps if r["outcome"] == "failed"]
    veto = any(p.casefold() in t.casefold() for t in details for p in evidence.DEVICE_PATTERNS)
    call = next(r for r in reps if r["when"] == "call")
    results[name] = dict(node=node, pytest_exit=rc, call_outcome=call["outcome"],
                         exception_class=call["exception_class"], groupnorm_refusal=call["groupnorm_refusal"],
                         acceptable_killers=evidence.acceptable_killers(data), device_veto=veto,
                         detail_head=call["detail"][-600:])
    print(name, rc, call["outcome"], call["exception_class"], call["groupnorm_refusal"],
          len(results[name]["acceptable_killers"]), veto, flush=True)


def generic_pow(tc):
    def bad(self, p):
        raise RuntimeError("pow: unsupported exponent path")  # a non-device engine RuntimeError
    S.Tensor.pow = bad


def device_pow(tc):
    def bad(self, p):
        raise RuntimeError("CUDA error at pow_kernel: an illegal memory access was encountered")
    S.Tensor.pow = bad


def foreign_factory(tc):
    real = tc.tensor

    def factory(data, device="cuda", dtype="float32", requires_grad=False):
        if requires_grad:
            raise TypeError("tensor(): incompatible function arguments (engine factory)")
        return real(data, device=device, dtype=dtype, requires_grad=requires_grad)
    tc.tensor = factory
    return lambda: setattr(tc, "tensor", real)


run_case("P0_honest", VALUE_ID)
run_case("P1_generic_engine_RuntimeError", VALUE_ID, generic_pow)
run_case("P2_engine_TypeError_under_GroupNorm_init", CTOR_ID, foreign_factory)
run_case("P3_device_fault_in_forward", VALUE_ID, device_pow)

# Census: device-pattern text in any source that a traceback of these suites can print.
sources = sorted(set(list((MA2 / "arms/candidate/tensor_cuda").rglob("*.py")) +
                     list((MA2 / "mutants").rglob("nn.py")) + [TEST, MROOT / "tensor_cuda/tests/test_ext_phase7.py"]))
hits = []
for src in sources:
    for n, line in enumerate(src.read_text(errors="replace").splitlines(), 1):
        for p in evidence.DEVICE_PATTERNS:
            if p.casefold() in line.casefold():
                hits.append(dict(file=str(src.relative_to(MROOT)), line=n, pattern=p, text=line.strip()[:120]))
results["source_pattern_census"] = dict(files=len(sources), hits=hits)
(OUT / "q3_probe.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "detail_head"} if isinstance(v, dict) else v
                  for k, v in results.items()}, indent=1)[:4000])
