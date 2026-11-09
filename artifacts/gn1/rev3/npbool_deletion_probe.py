#!/usr/bin/env python3
"""GN1-REV3 Q5: amendment 1. Builds A3's original numpy.bool_-clause DELETION
mutants (scratch copies; arms untouched) and runs the WHOLE sealed GN test file
(405 IDs) on each under A3's NumPy stand-in; plus operator.index on numpy.bool_.
Prior art: DeMillo, Lipton, Sayward (1978) equivalent mutants; PEP 357 (2006)."""
import importlib.util, json, operator, os
from pathlib import Path
import numpy as np
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
OUT = Path(__file__).resolve().parent
MROOT = Path("/tmp/gn1_rev3_m/W"); MA3 = MROOT / "artifacts/gn1/a3"
import pytest
spec = importlib.util.spec_from_file_location("a3_cpu_shim", MA3 / "cpu_shim.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
cand = (MA3 / "arms/candidate/tensor_cuda/nn.py").read_text()
res = {"numpy": np.__version__}
for v in (np.True_, np.False_):
    try:
        operator.index(v); res["operator.index(%r)" % v] = "accepted"
    except TypeError as e:
        res["operator.index(%r)" % v] = "TypeError: " + str(e)
class Count:
    def __init__(self): self.failed = []; self.n = 0
    def pytest_runtest_logreport(self, report):
        if report.when == "call": self.n += 1
        if report.failed: self.failed.append(report.nodeid)
for name, old in (("del_groups", "isinstance(num_groups, bool) or isinstance(num_groups, np.bool_)"),
                  ("del_channels", "isinstance(num_channels, bool) or isinstance(num_channels, np.bool_)")):
    assert cand.count(old) == 1
    mut = Path("/tmp/gn1_rev3_tmp/npbool_" + name + "/nn.py"); mut.parent.mkdir(parents=True, exist_ok=True)
    mut.write_text(cand.replace(old, old.split(" or ")[0], 1))
    for which, path in (("candidate", MA3 / "arms/candidate/tensor_cuda/nn.py"), ("mutant", mut)):
        S.install(path)
        c = Count()
        rc = int(pytest.main(["-q", "-p", "no:cacheprovider", "--import-mode=importlib", "--rootdir=" + str(MROOT),
                              "--basetemp=/tmp/gn1_rev3_tmp/npb", str(MROOT / "tensor_cuda/tests/test_groupnorm_gn1.py")], plugins=[c]))
        res[name + "::" + which] = dict(exit=rc, calls=c.n, failed=c.failed)
(OUT / "npbool_deletion_probe.json").write_text(json.dumps(res, indent=2) + "\n")
print(json.dumps(res))
