#!/usr/bin/env python3
"""GN1-REV3 R2-F09: REV2's uncovered-decision probe nodes run against A3's
registered V4 arm (sealed mirror) and candidate, under A3's NumPy stand-in; plus
a byte check that V4 is exactly REV2's scratch mutant of A3's candidate.
Prior art: REV2 q5_uncovered_probe.py (2026); DeMillo, Lipton, Sayward (1978)."""
import importlib.util, json, os
from pathlib import Path
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
OUT = Path(__file__).resolve().parent
MROOT = Path("/tmp/gn1_rev3_m/W"); MA3 = MROOT / "artifacts/gn1/a3"
import pytest
spec = importlib.util.spec_from_file_location("a3_cpu_shim", MA3 / "cpu_shim.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
cand = (MA3 / "arms/candidate/tensor_cuda/nn.py").read_text()
old = 'if (dtype == "float32" and self.weight.dtype == "float32"'
v4 = (MA3 / "arms/V4/tensor_cuda/nn.py").read_text()
res = dict(v4_equals_rev2_mutant=v4 == cand.replace(old, 'if (True and self.weight.dtype == "float32"', 1))
for node in ("test_mixed_parameters[forward-default-float16]", "test_forward_values[float16-r3-n2-g2-default]",
             "test_mixed_parameters[forward-default-bfloat16]"):
    out = {}
    for name, path in (("candidate", MA3 / "arms/candidate/tensor_cuda/nn.py"), ("V4", MA3 / "arms/V4/tensor_cuda/nn.py")):
        S.install(path)
        out[name] = int(pytest.main(["-q", "-p", "no:cacheprovider", "--import-mode=importlib", "--rootdir=" + str(MROOT),
                                     "--basetemp=/tmp/gn1_rev3_tmp/v4", str(MROOT / "tensor_cuda/tests/test_groupnorm_gn1.py") + "::" + node]))
    res[node] = out
(OUT / "v4_rev2_nodes.json").write_text(json.dumps(res, indent=2) + "\n")
print(json.dumps(res))
