"""CPU import/collection/reference instrument, actual package and pinned engine.
Prior art: round-one GN1 hooks; pytest collection hooks (Krekel et al., 2004+,
unverified lead to check). Preload real arm to avoid namespace-package shadowing.
"""
import json
import os
from pathlib import Path
import sys

TOOL=Path(__file__).resolve().parent
ROOT=TOOL.parents[2]
arm,suite,mode=sys.argv[1:]
assert os.environ.get("CUDA_VISIBLE_DEVICES")==""
from evidence import scratch_guard
scratch_guard(os.environ["TMPDIR"])
package=TOOL/"arms"/arm/"tensor_cuda"
sys.path.insert(0,str(package.parent))
sys.path.insert(1,str(ROOT))
import tensor_cuda as tc
from tensor_cuda import nn
import pytest
assert Path(tc.__file__).resolve()==(package/"__init__.py").resolve()
assert Path(nn.__file__).resolve()==(package/"nn.py").resolve()
pin=json.loads((TOOL/"engine_pin.json").read_text())
assert Path(tc._C.__file__).resolve()==(package/Path(pin["module"]).name).resolve()
import hashlib
assert hashlib.sha256(Path(tc._C.__file__).read_bytes()).hexdigest()==pin["sha256"]
from run_gpu_tests import Reports
plugin=Reports()
files=(["tensor_cuda/tests/test_groupnorm_gn1.py", "tensor_cuda/tests/test_ext_phase7.py"] if suite=="focus" else ["tensor_cuda/tests/test_groupnorm_gn1.py"] if suite=="mutation" else json.loads((TOOL/"nn_import_test_files.json").read_text()))
args=["-q","-p","no:cacheprovider","--import-mode=importlib","--rootdir="+str(ROOT)]
if mode=="collect":args+=["--collect-only"]
else:
    assert arm=="candidate" and suite=="mutation" and mode=="references"
    args += ["-k","reference or nondivisible_refusal or positive_integer_refusal"]
args += [str(ROOT/f) for f in files]
rc=int(pytest.main(args,plugins=[plugin]))
data=dict(arm=arm,suite=suite,mode=mode,CUDA_VISIBLE_DEVICES="",TMPDIR=os.environ["TMPDIR"],engine=str(Path(tc._C.__file__).resolve()),package=str(package),exit=rc,ids=plugin.ids,collection_errors=plugin.collection_errors,reports=plugin.reports)
(TOOL/("cpu_"+arm+"_"+suite+"_"+mode+".json")).write_text(json.dumps(data,indent=2)+"\n")
if mode=="collect":(TOOL/("ids_"+arm+"_"+suite+".txt")).write_text("\n".join(plugin.ids)+"\n")
print(json.dumps(dict(arm=arm,suite=suite,mode=mode,exit=rc,count=len(plugin.ids))))
raise SystemExit(rc)
