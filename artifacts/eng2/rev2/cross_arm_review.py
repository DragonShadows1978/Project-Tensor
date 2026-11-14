"""Use real collection and _record IDs, then execute the exact scorer block with equal constructed payloads.
Prior art: McKeeman 1998 differential testing, pytest collection (Krekel et al. 2004); unverified -- lead to check. No GPU or claimed device receipt.
"""
from pathlib import Path
import ast,json,os,subprocess,sys
ROOT=Path.cwd();OUT=ROOT/"artifacts/eng2/rev2";TOOL=ROOT/"artifacts/eng2/tool"
child=r"""
from pathlib import Path
from types import SimpleNamespace
import json,os,sys
package,root,out,arm=sys.argv[1:];sys.path.insert(0,package);import tensor_cuda;import pytest
p=Path(out);p.mkdir(exist_ok=True);os.environ["ENG2_RECORD_DIR"]=str(p)
class P:
 def pytest_collection_finish(self,session):
  ids=[i.nodeid for i in session.items];(p/"collected.json").write_text(json.dumps(ids,indent=2)+"\n")
  for i in session.items:
   name=i.name.split("[",1)[0]
   kind="p302bits" if name=="test_n1_float32_bits" else "f1bytes" if name in ["test_f1_tensor_supported","test_f1_factory_supported","test_f1_default_float64_data","test_f1_factory_default"] else None
   if kind:i.module._record(kind,SimpleNamespace(node=SimpleNamespace(nodeid=i.nodeid)),p,dict(sha256="1"*64,constructed_same_payload=True))
rc=pytest.main(["-q","-p","no:cacheprovider","--import-mode=importlib","--rootdir="+root,"--collect-only",str(Path(package)/"tests/test_eng2.py")],plugins=[P()]);sys.exit(rc)
"""
for arm in ["baseline","candidate"]:
 r=subprocess.run([sys.executable,"-B","-c",child,str(TOOL/"arms"/arm/"tensor_cuda"),str(ROOT),str(OUT/(arm+"_records")),arm],cwd=TOOL,capture_output=True,text=True,timeout=90)
 (OUT/(arm+"_collection.log")).write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
 assert len(json.loads((OUT/(arm+"_records/collected.json")).read_text()))==298
# Execute precisely the scorer cross-arm loop, no substitute comparison function.
main=next(n for n in ast.parse((TOOL/"score_gpu.py").read_text()).body if isinstance(n,ast.FunctionDef) and n.name=="main")
loop=next(n for n in main.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=="kind")
def rows(label,pattern):return [json.loads(p.read_text()) for p in sorted((OUT/(label.replace("_eng2","")+"_records")).glob(pattern))]
sys.path.insert(0,str(TOOL));import score_gpu
ns=dict(vars(score_gpu),rows=rows,cross={},summary={"constructed_equal_payloads":True},errors=[])
exec(compile(ast.Module(body=[loop],type_ignores=[]),str(TOOL/"score_gpu.py"),"exec"),ns)
result=dict(provenance="Real pytest IDs and actual _record, constructed identical hashes; exact scorer loop",cross=ns["cross"],errors=ns["errors"])
(OUT/"cross_arm_review.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(dict(collected_per_arm=298,cross={k:{q:v for q,v in d.items() if q!="differing"} for k,d in ns["cross"].items()},errors=[e[:350] for e in ns["errors"]]),indent=2))
