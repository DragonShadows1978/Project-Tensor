"""A3 G1/G3. Prior art: SHA-256/FIPS 180-4, A2 structural scope gate.
Read-only git and explicit frozen/source maps; no traversal outside named trees.
"""
import json,subprocess
from pathlib import Path
from evidence import TOOL,ROOT,read,sha,verify_seal,verify_arm

def git(*args):return subprocess.check_output(["git",*args],cwd=ROOT)
def save(name,obj):(TOOL/name).write_text(json.dumps(obj,indent=2)+"\n")
def outside(s):
    a=s.index("class GroupNorm(");b=s.index("\n\nclass InstanceNorm2D",a)
    return s[:a],s[b:]
candidate=(ROOT/"tensor_cuda/tensor_cuda/nn.py").read_text()
base=git("show","0e1dea3:tensor_cuda/tensor_cuda/nn.py").decode()
a2=git("show","bd30498a:tensor_cuda/tensor_cuda/nn.py").decode()
assert outside(candidate)==outside(base)==outside(a2)
expected=a2
for old,new in read(TOOL/"d5_edits.json"):
    assert expected.count(old)==1
    expected=expected.replace(old,new,1)
assert expected==candidate
files=git("diff","--name-only","0e1dea3","--",".",":(exclude)artifacts/gn1",":(exclude)orders",":(exclude)logs").decode().splitlines()
assert files==["tensor_cuda/tensor_cuda/nn.py","tensor_cuda/tests/test_groupnorm_gn1.py"],files
for ref,name in [("0e1dea3","changes_vs_base.patch"),("bd30498a","changes_vs_a2.patch")]:
    (TOOL/name).write_bytes(git("diff",ref,"--","tensor_cuda/tensor_cuda/nn.py","tensor_cuda/tests/test_groupnorm_gn1.py"))
(TOOL/"nn_vs_bd30498a.diff").write_bytes(git("diff","bd30498a","--","tensor_cuda/tensor_cuda/nn.py"))
frozen={}
for d in ["a2","rev1","rev2"]:
    after={str(p.relative_to(ROOT)):sha(p) for p in (TOOL.parent/d).rglob("*") if p.is_file()}
    save(d+"_frozen_after.json",after)
    assert after==read(TOOL/(d+"_frozen_before.json")),d
    frozen[d]=dict(files=len(after),before_map_sha256=sha(TOOL/(d+"_frozen_before.json")),after_map_sha256=sha(TOOL/(d+"_frozen_after.json")))
# Whole tracked scope relative to the amendment commit, excluding allowed files.
changed=git("diff","--name-only","HEAD").decode().splitlines()
assert all(n in files or n.startswith(("artifacts/gn1/a3/","logs/")) for n in changed),changed
save("G1_scope.json",dict(passed=True,diff_files=files,nn_D5_only_vs_A2=True,nn_outside_GroupNorm_identical=True,frozen=frozen,changed_tracked=changed))
seal=verify_seal()
for arm in seal["arms"]:verify_arm(arm,seal)
pin=read(TOOL/"engine_pin.json")
assert sha(TOOL/"engine_pin.json")==sha(TOOL.parent/"engine_pin.json")
source_map=read(TOOL.parent/"engine_sources_after.json")
actual={n:sha(Path(n) if Path(n).is_absolute() else ROOT/n) for n in source_map}
assert source_map==actual
assert sha(TOOL.parent/"engine_sources_after.json")==pin["source_map_sha256"]
save("engine_sources_verified.json",actual)
save("G3_pin.json",dict(passed=True,module_sha256=pin["sha256"],source_map_sha256=pin["source_map_sha256"],source_files=len(actual),module_copies=len(seal["arms"]),symlinks=0,builds=0))
print("G1 PASS; G3 PASS",len(seal["arms"]),"arms")
