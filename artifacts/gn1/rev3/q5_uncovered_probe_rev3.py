#!/usr/bin/env python3
"""GN1-REV3 Q6: A3's new D5 decisions that carry no registered mutant: the two
`except TypeError` refusal-class mappings in GroupNorm.__init__ (num_groups and
num_channels). Builds each single-site mutant in /tmp scratch (arms untouched)
and runs candidate witnesses under A3's NumPy engine stand-in. Also V4 re-check.
Prior art: DeMillo, Lipton, Sayward (1978) mutation operators; REV2 q5_uncovered_probe.py (2026). Ours: the sites.
"""
import importlib.util, json, os
from pathlib import Path
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
OUT = Path(__file__).resolve().parent
MROOT = Path("/tmp/gn1_rev3_m/W"); MA3 = MROOT / "artifacts/gn1/a3"
import pytest
spec = importlib.util.spec_from_file_location("a3_cpu_shim", MA3 / "cpu_shim.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
cand = (MA3 / "arms/candidate/tensor_cuda/nn.py").read_text()
sites = {
 "except_groups": ("num_groups = int(operator.index(num_groups))\n        except TypeError as exc:",
                   "num_groups = int(operator.index(num_groups))\n        except ArithmeticError as exc:"),
 "except_channels": ("num_channels = int(operator.index(num_channels))\n        except TypeError as exc:",
                     "num_channels = int(operator.index(num_channels))\n        except ArithmeticError as exc:"),
}
nodes = {"except_groups": ["test_positive_integer_refusal[float-num_groups]", "test_positive_integer_refusal[string-num_groups]",
                           "test_numpy_bool_refusal[true-num_groups]"],
         "except_channels": ["test_positive_integer_refusal[float-num_channels]", "test_positive_integer_refusal[none-num_channels]",
                             "test_numpy_bool_refusal[true-num_channels]"]}
res = {}
for name, (old, new) in sites.items():
    assert cand.count(old) == 1
    mut = Path("/tmp/gn1_rev3_tmp/uncovered_" + name + "/nn.py"); mut.parent.mkdir(parents=True, exist_ok=True)
    mut.write_text(cand.replace(old, new, 1))
    for node in nodes[name]:
        out = {}
        for which, path in (("candidate", MA3 / "arms/candidate/tensor_cuda/nn.py"), ("mutant", mut)):
            S.install(path)
            out[which] = int(pytest.main(["-q", "-p", "no:cacheprovider", "--import-mode=importlib", "--rootdir=" + str(MROOT),
                                          "--basetemp=/tmp/gn1_rev3_tmp/q5u", str(MROOT / "tensor_cuda/tests/test_groupnorm_gn1.py") + "::" + node]))
        res[name + "::" + node] = out
(OUT / "q5_uncovered_probe_rev3.json").write_text(json.dumps(res, indent=2) + "\n")
print(json.dumps(res, indent=1))
