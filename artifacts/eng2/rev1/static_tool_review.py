"""Read-only identity, mutation and tooling checks. Prior art: differential/mutation testing (McKeeman 1998; DeMillo et al. 1978), SHA-256 (NIST 2001); unverified -- lead to check. No novel method."""
from pathlib import Path
import ast,json,hashlib,sys,subprocess,difflib
ROOT=Path.cwd();OUT=ROOT/"artifacts/eng2/rev1";TOOL=ROOT/"artifacts/eng2/tool"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
sys.path.insert(0,str(TOOL));import evidence
seal=evidence.verify_seal()
for arm in seal["arms"]:evidence.verify_arm(arm,seal)
print("Live seal and",len(seal["arms"]),"arms verified")
deps=[]
for row in json.loads((TOOL.parent/"COPY_DEPS.json").read_text())["files"]:
    n=row["file"];a=ROOT/"artifacts/eng1/tool"/n;b=TOOL/n
    record=dict(file=n,before=sha(a)==row["sha256_before"],after=(not b.exists()) if row["sha256_after"] is None else sha(b)==row["sha256_after"])
    assert record["before"] and record["after"],record
    deps.append(record)
mutants=json.loads((TOOL.parent/"mutants/mutants.json").read_text());checked=[]
for m in mutants:
    rel=Path(m["file"]);name=rel.name;candidate=(ROOT/rel).read_text();arm=TOOL/"arms"/m["id"]
    got=(arm/rel).read_text();prefix="";suffix="";region=candidate
    if m.get("scope"):
        tree=ast.parse(candidate);c=next(c for c in tree.body if isinstance(c,ast.ClassDef) and c.name==m["scope"])
        lines=candidate.splitlines(keepends=True);prefix="".join(lines[:c.lineno-1]);region="".join(lines[c.lineno-1:c.end_lineno]);suffix="".join(lines[c.end_lineno:])
    assert region.count(m["old"])==1,m["id"]
    expected=prefix+region.replace(m["old"],m["new"],1)+suffix
    assert got==expected,m["id"]
    ast.parse(got)
    first,last=m["hunk_lines"];assert got.splitlines()[first-1:last]==m["new"].splitlines()
    changed=[]
    for f in sorted((TOOL/"arms/candidate").rglob("*")):
        if f.is_file() and "__pycache__" not in f.parts:
            r=f.relative_to(TOOL/"arms/candidate")
            if sha(f)!=sha(arm/r):changed.append(str(r))
    assert changed==[m["file"]],(m["id"],changed)
    checked.append(dict(id=m["id"],single_registered_edit=True,hunk=m["hunk_lines"],killers=len(m["killers"]),changed=changed))
# Verify runtime source and E3 against the exact requested commits.
identities=[]
for rev,arm in [("bc02d3a1","baseline"),("2237299b","candidate")]:
    for name in ["__init__.py","nn.py"]:
        old=subprocess.check_output(["git","--no-optional-locks","show",rev+":tensor_cuda/tensor_cuda/"+name])
        assert old==(TOOL/"arms"/arm/"tensor_cuda/tensor_cuda"/name).read_bytes()
        if arm=="candidate":assert old==(ROOT/"tensor_cuda/tensor_cuda"/name).read_bytes()
        identities.append(dict(revision=rev,arm=arm,file=name,sha256=hashlib.sha256(old).hexdigest()))
base_e3=subprocess.check_output(["git","--no-optional-locks","show","bc02d3a1:tensor_cuda/tests/test_eng1.py"])
assert base_e3==(ROOT/"tensor_cuda/tests/test_eng1.py").read_bytes()
for arm in ["baseline","candidate"]:assert base_e3==(TOOL/"arms"/arm/"tensor_cuda/tests/test_eng1.py").read_bytes()
# Device-mode dependency recomputation is read-only; root dry runs cannot satisfy a dependency.
runs=TOOL/"gpu_runs/dry_1";s,start=evidence.verify_start(runs);assert not start["audit"]
dependencies={}
for label in ["candidate_eng2","baseline_regnorm","baseline_regdtype"]:
    r=evidence.recompute_line(runs,label,s,start);assert not r["passed"];dependencies[label]=dict(passed=r["passed"],reasons=r["reasons"])
result=dict(copy_dependencies=deps,mutants=checked,identities=identities,e3_unchanged=True,live_seal=True,verified_arms=len(seal["arms"]),device_dependencies=dependencies)
(OUT/"static_tool_review.json").write_text(json.dumps(result,indent=2)+"\n")
print("COPY_DEPS",len(deps),"matched; mutants",len(checked),"valid single edits; source pins matched; E3 unchanged")
print("Dependencies",dependencies)
