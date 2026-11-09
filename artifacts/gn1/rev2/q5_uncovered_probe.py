#!/usr/bin/env python3
"""GN1-REV2 Q5: the one GroupNorm decision with no registered mutant, the
fast-path conjunct `dtype == "float32"` (V1/V2 cover only the weight and bias
conjuncts). Builds that single-site mutant in /tmp scratch (the arms are not
touched) and runs two real witness IDs under A2's NumPy engine stand-in.
Prior art: DeMillo, Lipton, Sayward (1978), mutation operators. Ours: the site.
"""
import importlib.util, json, os, sys
from pathlib import Path
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
OUT = Path(__file__).resolve().parent
MROOT = Path("/tmp/gn1_rev2_regen/W"); MA2 = MROOT / "artifacts/gn1/a2"
import pytest
spec = importlib.util.spec_from_file_location("a2_cpu_shim", MA2 / "cpu_shim.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
cand = (MA2 / "arms/candidate/tensor_cuda/nn.py").read_text()
old = 'if (dtype == "float32" and self.weight.dtype == "float32"'
assert cand.count(old) == 1
mut = Path("/tmp/gn1_rev2_tmp/V4_fastpath_input_check/nn.py"); mut.parent.mkdir(parents=True, exist_ok=True)
mut.write_text(cand.replace(old, 'if (True and self.weight.dtype == "float32"', 1))
res = {}
for node in ("test_mixed_parameters[forward-default-float16]", "test_forward_values[float16-r3-n2-g2-default]",
             "test_mixed_parameters[forward-default-bfloat16]"):
    out = {}
    for name, path in (("candidate", MA2 / "arms/candidate/tensor_cuda/nn.py"), ("mutant", mut)):
        S.install(path)
        out[name] = int(pytest.main(["-q", "-p", "no:cacheprovider", "--import-mode=importlib", "--rootdir=" + str(MROOT),
                                     "--basetemp=/tmp/gn1_rev2_tmp/q5u", str(MROOT / "tensor_cuda/tests/test_groupnorm_gn1.py") + "::" + node]))
    res[node] = out
(OUT / "q5_uncovered_probe.json").write_text(json.dumps(res, indent=2) + "\n")
print(json.dumps(res))
