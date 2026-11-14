"""Prior art: ENG2 stage.make_arm (2026), SHA-256 file comparison (NIST FIPS 180-4).
Reuses the exact recipe in a new scratch root; no new algorithm; no N1 semantic review.
"""
from pathlib import Path
import ast, difflib, hashlib, json, os, subprocess, sys
ROOT=Path.cwd(); OUT=ROOT/"artifacts/eng2/rev2"; TOOL=ROOT/"artifacts/eng2/tool"
def sha(b):return hashlib.sha256(b).hexdigest()
def git(*args):return subprocess.check_output(["git","--no-optional-locks",*args],cwd=ROOT)
def read(p):return json.loads(Path(p).read_text())
def save(n,v):(OUT/n).write_text(json.dumps(v,indent=2)+"\n")
changed=[line.split("\t") for line in git("diff","--name-status","9452ef61","1a4beed1").decode().splitlines()]
(OUT/"scope_full.diff").write_bytes(git("diff","9452ef61","1a4beed1"))
ag3=read(ROOT/"artifacts/eng2/a1/AG3_SCOPE.json")
modified={path for status,path in changed if status=="M"}
registered=set(ag3["changed_outside_a1"])
for path,arms in ag3["arm_file_changes"].items():
 registered.update("artifacts/eng2/tool/arms/"+arm+"/"+path for arm in arms)
expected_modified=modified==registered
assert expected_modified,(modified-registered,registered-modified)
added={path for status,path in changed if status=="A"}
unexpected_added=[p for p in added if not p.startswith(("artifacts/eng2/a1/","artifacts/eng2/tool/gpu_runs/dry_2/","artifacts/eng2/tool/gpu_runs/audit_2/")) and p!="orders/ENG2_A1_REV1_FIXES.md"]
assert not unexpected_added,unexpected_added
assert all(status in ("A","M") for status,_ in changed)
allowed={"tensor_cuda/tensor_cuda/__init__.py":{"_np_dtype","tensor","_factory"},
         "tensor_cuda/tensor_cuda/functional.py":{"_causal_mask"},
         "artifacts/eng2/tool/evidence.py":{"literal_brackets","matches","regression_reasons","pin_failure_reasons","PIN_EXCEPTION"},
         "artifacts/eng2/tool/score_gpu.py":{"CROSS_ROWS","keyed_rows","main"}}
owners={}
def nodes(b):
 result={}
 for i,n in enumerate(ast.parse(b).body):
  if isinstance(n,(ast.FunctionDef,ast.ClassDef)):key=n.name
  elif isinstance(n,ast.Assign):key=ast.unparse(n.targets[0])
  else:key=type(n).__name__+":"+ast.dump(n,include_attributes=False)
  result[key]=ast.dump(n,include_attributes=False)
 return result
for path,names in allowed.items():
 a=nodes(git("show","9452ef61:"+path));b=nodes(git("show","1a4beed1:"+path))
 diff={k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
 assert diff<=names,(path,diff)
 owners[path]=sorted(diff)
checks={}
for path in list(allowed)+["tensor_cuda/tests/test_eng2.py","artifacts/eng2/dtype_census.py"]:
 checks[path]=dict(live_equals_A1=sha((ROOT/path).read_bytes())==sha(git("show","1a4beed1:"+path)),before_copy_equals_scope_base=(ROOT/"artifacts/eng2/a1/before"/path.replace("/","__")).read_bytes()==git("show","9452ef61:"+path))
 assert all(checks[path].values()),checks[path]
test_prefix=git("show","1a4beed1:tensor_cuda/tests/test_eng2.py").startswith(git("show","9452ef61:tensor_cuda/tests/test_eng2.py"))
assert test_prefix
nn_same=git("show","9452ef61:tensor_cuda/tensor_cuda/nn.py")==git("show","1a4beed1:tensor_cuda/tensor_cuda/nn.py")== (ROOT/"tensor_cuda/tensor_cuda/nn.py").read_bytes()
assert nn_same
save("scope_review.json",dict(tracked_delta=len(changed),modified=len(modified),added=len(added),modified_exactly_AG3=expected_modified,unexpected_added=unexpected_added,dispatch_order_extra="orders/ENG2_A1_REV1_FIXES.md predates A1 before-map",tracked_vs_whole_tree_note="AG3 includes ignored/untracked receipts; git diff only includes committed files. Modified file sets match exactly.",owners=owners,bindings=checks,test_additions_only=test_prefix,N1_bytes_unchanged=nn_same))
# Execute the exact make_arm recipe after rebinding its destination only.
sys.path.insert(0,str(TOOL));import stage,evidence
scratch=Path(os.environ["TMPDIR"])/"recipe_rebuild"
assert not scratch.exists();scratch.mkdir()
assert os.statvfs(scratch).f_bavail*os.statvfs(scratch).f_frsize>2_000_000_000
stage.TOOL=scratch
base=ROOT/"artifacts/eng2/base_pkg/tensor_cuda"
base_binding={str(p.relative_to(base)):sha(p.read_bytes())==sha(git("show","bc02d3a1:tensor_cuda/tensor_cuda/"+str(p.relative_to(base)))) for p in base.rglob("*.py")}
assert all(base_binding.values())
registry={m["id"]:m for m in stage.registry()}
arm_results={}
for arm in ["baseline","candidate","MF1","MF2"]:
 stage.make_arm(arm,base if arm=="baseline" else ROOT/"tensor_cuda/tensor_cuda",stage.BASE_MODULE if arm=="baseline" else stage.CAND_MODULE,edit=registry.get(arm))
 actual=evidence.arm_files(arm)
 recipe={str(p.relative_to(scratch/"arms"/arm)):sha(p.read_bytes()) for p in (scratch/"arms"/arm).rglob("*") if p.is_file()}
 assert actual==recipe,(arm,actual,recipe)
 arm_results[arm]=dict(files=len(actual),exact_equal=True,sha256_map=actual)
seal=evidence.verify_seal()
for arm in arm_results:evidence.verify_arm(arm,seal)
fresh={}
for arm,count in [("candidate",0),("MF1",12),("MF2",48)]:
 d=read(OUT/(arm+"_cpu.json"));failed=sorted(k for k,v in d["outcomes"].items() if v!="passed")
 assert d["n"]==157 and len(failed)==count
 row=dict(collected=157,passed=157-count,failed=count,pytest_rc=d["rc"])
 if arm!="candidate":
  m=registry[arm];killers=sorted(m["killers"])
  traces={k:d["mutant_trace"].get(k,[]) for k in killers}
  assert failed==killers
  assert all(set(v)&set(m["hunk_lines"]) for v in traces.values())
  row.update(failed_exactly_killers=True,hunk_lines=m["hunk_lines"],killers_traced=len(traces),trace=traces)
 fresh[arm]=row
save("arms_review.json",dict(recipe="stage.make_arm, destination rebound to scratch only",scratch=str(scratch),baseline_source_equals_bc02d3a1=base_binding,arms=arm_results,seal_verified=True,fresh_cpu=fresh))
print(json.dumps(dict(scope_modified=len(modified),scope_added=len(added),scope_exact=True,test_additions_only=test_prefix,N1_unchanged=nn_same,arms={k:v["files"] for k,v in arm_results.items()},fresh_cpu={k:{q:v for q,v in r.items() if q!="trace"} for k,r in fresh.items()}),indent=2))
