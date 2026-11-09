"""G1/G3 exact scope, historical hashes and pinned-copy checks.
Prior art: SHA-256 NIST (2001), AST structural comparison; GN1/A1 gates.
Read-only git only. Known source-map paths are read directly, never searched.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from evidence import TOOL, ROOT, sha, read, verify_seal, verify_arm


def save(name,data):
    (TOOL/name).write_text(json.dumps(data,indent=2)+"\n")


def git(*args):
    return subprocess.check_output(["git",*args],cwd=ROOT)


baseline=git("show","0e1dea3:tensor_cuda/tensor_cuda/nn.py").decode()
a1=git("show","0c6a4244:tensor_cuda/tensor_cuda/nn.py").decode()
candidate=(ROOT/"tensor_cuda/tensor_cuda/nn.py").read_text()
def outside(s):
    lo=s.index("class GroupNorm(");hi=s.index("\n\nclass InstanceNorm2D")
    return s[:lo],s[hi:]
assert outside(candidate)==outside(baseline)==outside(a1), "non-GroupNorm nn.py change"
d5=a1
for old,new in read(TOOL/"d5_edits.json"):
    assert d5.count(old)==1
    d5=d5.replace(old,new,1)
assert d5==candidate, "nn.py is not D5-only relative to A1"
files=git("diff","--name-only","0e1dea3","--",".",":(exclude)artifacts/gn1",":(exclude)orders",":(exclude)logs").decode().splitlines()
assert files==["tensor_cuda/tensor_cuda/nn.py","tensor_cuda/tests/test_groupnorm_gn1.py"],files
for ref,out in (("0e1dea3","changes_vs_base.patch"),("0c6a4244","changes_vs_a1.patch")):
    (TOOL/out).write_bytes(git("diff",ref,"--","tensor_cuda/tensor_cuda/nn.py","tensor_cuda/tests/test_groupnorm_gn1.py"))
(TOOL/"nn_vs_0c6a4244.diff").write_bytes(git("diff","0c6a4244","--","tensor_cuda/tensor_cuda/nn.py"))
before=read(TOOL/"tracked_before.json")
after={p:sha(ROOT/p) for p in before if (ROOT/p).is_file()}
save("tracked_after.json",after)
changed=[p for p,h in before.items() if after.get(p)!=h]
assert sorted(changed)==files,changed
base=read(TOOL.parent/"a1/base_sha256.json")
mismatches=[p for p,h in base.items() if p not in files and not p.startswith(("orders/","artifacts/gn1/","logs/")) and sha(ROOT/p)!=h]
assert not mismatches,mismatches
frozen_before=read(TOOL/"frozen_before.json")
frozen_after={str(p.relative_to(ROOT)):sha(p) for p in sorted(TOOL.parent.rglob("*")) if p.is_file() and "a2" not in p.relative_to(TOOL.parent).parts and "__pycache__" not in p.parts}
save("frozen_after.json",frozen_after)
assert frozen_before==frozen_after,"frozen GN1 receipt tree changed"
save("G1_scope.json",dict(passed=True,diff_files=files,nn_outside_GroupNorm_identical=True,nn_D5_only_vs_A1=True,
     tracked_count=len(before),tracked_changed=changed,baseline_map_count=len(base),baseline_other_mismatches=mismatches,
     frozen_files=len(frozen_before),frozen_maps_equal=True,frozen_before_map_sha256=sha(TOOL/"frozen_before.json"),
     frozen_after_map_sha256=sha(TOOL/"frozen_after.json")))
seal=verify_seal()
for arm in seal["arms"]:verify_arm(arm,seal)
pin=read(TOOL/"engine_pin.json")
source_file=TOOL.parent/"engine_sources_after.json"
assert sha(source_file)==pin["source_map_sha256"]
source_map=read(source_file)
actual={p:sha(Path(p) if Path(p).is_absolute() else ROOT/p) for p in source_map}
assert actual==source_map,"pinned source map differs"
save("engine_sources_verified.json",actual)
assert sha(TOOL/"engine_pin.json")==sha(TOOL.parent/"engine_pin.json"),"pin JSON changed"
assert sha(TOOL/"engine_sources_after.json")==sha(source_file)
save("G3_pin.json",dict(passed=True,module_sha256=pin["sha256"],pin_file_sha256=sha(TOOL/"engine_pin.json"),
     source_map_sha256=sha(source_file),source_files=len(source_map),source_mismatches=[],module_copies=len(seal["arms"]),symlinks=0,
     arm_maps=read(TOOL/"arm_maps.json"),builds=0))
print(json.dumps(dict(G1="PASS",G3="PASS",frozen_files=len(frozen_before),tracked_count=len(before),arms=len(seal["arms"]))))
