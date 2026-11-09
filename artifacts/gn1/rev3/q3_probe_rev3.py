#!/usr/bin/env python3
"""GN1-REV3 Q2: the D2/D3 kill rule as A3's REAL pipeline produces it.

Real sealed test file + A3's REAL child plugin (run_gpu_tests.Reports) inside
pytest 9, A3's candidate (or mutant) nn.py under A3's NumPy engine stand-in
(a3/cpu_shim.py, mirror copy); the plugin's reports go to A3's REAL
evidence.acceptable_killers() and its device veto (same text assembly as
evidence.evaluate()). Fault injection only in the stand-in engine.

REV2 cases P0-P3 re-run, plus new:
  P4 backward:   engine RuntimeError in pow backward-path only (test_backward_values), no device text.
  P5 generic VE: engine ValueError (no device text) inside GroupNorm.forward.
  P6 ctx-device: generic RuntimeError raised while handling a CUDA error (implicit __context__).
  P7 mutant VE:  A3's D5_g_npbool mutant arm, test_index_counts[np_int64-num_groups] (GroupNorm.__init__ raises).
  P8 callee VE:  a stand-in factory ValueError raised under GroupNorm.__init__ (callee frame).
Prior art: fault injection / test stubs (Meszaros 2007); REV2 q3_probe.py (2026). Ours: P4-P8.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
OUT = Path(__file__).resolve().parent
MROOT = Path("/tmp/gn1_rev3_m/W")
MA3 = MROOT / "artifacts/gn1/a3"
sys.path.insert(0, str(MA3))
import evidence  # noqa: E402
import run_gpu_tests  # noqa: E402
import pytest  # noqa: E402

spec = importlib.util.spec_from_file_location("a3_cpu_shim", MA3 / "cpu_shim.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
TEST = MROOT / "tensor_cuda/tests/test_groupnorm_gn1.py"
VALUE_ID = "test_forward_values[float32-r3-n2-g2-default]"
results = {}
orig_pow = S.Tensor.pow


def run_case(name, node, patch=None, arm="candidate"):
    tc = S.install(MA3 / "arms" / arm / "tensor_cuda/nn.py")
    S.Tensor.pow = orig_pow
    undo = patch(tc) if patch else None
    plugin = run_gpu_tests.Reports()
    rc = int(pytest.main(["-q", "-p", "no:cacheprovider", "--import-mode=importlib", "--rootdir=" + str(MROOT),
                          "--basetemp=/tmp/gn1_rev3_tmp/q3_" + name, str(TEST) + "::" + node], plugins=[plugin]))
    if undo:
        undo()
    S.Tensor.pow = orig_pow
    reps = [dict(r) for r in plugin.reports]
    data = dict(reports=reps)
    details = [r["detail"] + " " + str(r.get("original_exception_class", "")) + " " + str(r.get("original_exception_message", ""))
               for r in reps if r["outcome"] == "failed"]
    veto = any(p.casefold() in t.casefold() for t in details for p in evidence.DEVICE_PATTERNS)
    call = next(r for r in reps if r["when"] == "call")
    killers = evidence.acceptable_killers(dict(reports=[dict(r, id="tensor_cuda/tests/test_groupnorm_gn1.py::" + node) for r in reps]))
    results[name] = dict(node=node, arm=arm, pytest_exit=rc, call_outcome=call["outcome"],
                         exception_class=call["exception_class"], groupnorm_refusal=call["groupnorm_refusal"],
                         wrapped=call.get("wrapped_engine_refusal"), engine_call=call.get("engine_refusal"),
                         original=(call.get("original_exception_class"), call.get("original_exception_message")),
                         acceptable_kill=bool(killers), device_veto=veto,
                         counts_as_kill=bool(killers) and not veto, detail_head=call["detail"][-500:])
    print(name, rc, call["outcome"], call["exception_class"], "gn_refusal", call["groupnorm_refusal"], "wrapped",
          call.get("wrapped_engine_refusal"), "engine_call", call.get("engine_refusal"), "kill", bool(killers), "veto", veto,
          flush=True)


def raiser(exc):
    def patch(tc):
        def bad(self, p):
            raise exc
        S.Tensor.pow = bad
    return patch


def ctx_device(tc):
    def bad(self, p):
        try:
            raise RuntimeError("CUDA error at pow_kernel: an illegal memory access was encountered")
        except RuntimeError:
            raise RuntimeError("pow: generic failure after device error")
    S.Tensor.pow = bad


def backward_only(tc):
    real = orig_pow
    def pw(self, p):
        out = real(self, p)
        def bw(*a, **k):
            raise RuntimeError("pow backward: unsupported exponent path")
        out.vjp = bw
        return out
    S.Tensor.pow = pw


def factory(exc_type):
    def patch(tc):
        real = tc.tensor
        def f(data, device="cuda", dtype="float32", requires_grad=False):
            if requires_grad:
                raise exc_type("tensor(): engine factory refusal")
            return real(data, device=device, dtype=dtype, requires_grad=requires_grad)
        tc.tensor = f
        return lambda: setattr(tc, "tensor", real)
    return patch


run_case("P0_honest", VALUE_ID)
run_case("P1_generic_engine_RuntimeError", VALUE_ID, raiser(RuntimeError("pow: unsupported exponent path")))
run_case("P2_engine_TypeError_under_GroupNorm_init", "test_constructor_defaults", factory(TypeError))
run_case("P3_device_fault_in_forward", VALUE_ID, raiser(RuntimeError("CUDA error at pow_kernel: an illegal memory access was encountered")))
run_case("P4_engine_error_backward_only", "test_backward_values[float32-r3-n2-g2-default]", backward_only)
run_case("P5_generic_engine_ValueError", VALUE_ID, raiser(ValueError("pow: exponent domain")))
run_case("P6_device_in_context_chain", VALUE_ID, ctx_device)
run_case("P7_mutant_D5_g_npbool_init_refusal", "test_index_counts[np_int64-num_groups]", arm="D5_g_npbool")
run_case("P8_callee_ValueError_under_init", "test_constructor_defaults", factory(ValueError))
(OUT / "q3_probe_rev3.json").write_text(json.dumps(results, indent=2) + "\n")
