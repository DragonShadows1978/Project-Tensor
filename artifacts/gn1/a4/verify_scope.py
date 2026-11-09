"""A4 G1/G3 exact frozen maps. Prior art: SHA-256 FIPS 180-4 (2015)."""
import json,subprocess
from pathlib import Path
from evidence import TOOL,ROOT,read,sha,verify_seal,verify_arm

def save(n,obj):(TOOL/n).write_text(json.dumps(obj,indent=2)+"\n")
def git(*args):return subprocess.check_output(["git",*args],cwd=ROOT)
frozen={}
for d in ("a3","rev1","rev2","rev3"):
    after={str(p.relative_to(ROOT)):sha(p) for p in (TOOL.parent/d).rglob("*") if p.is_file()}
    save(d+"_frozen_after.json",after)
    assert after==read(TOOL/(d+"_frozen_before.json")),d
    frozen[d]=dict(files=len(after),before_map_sha256=sha(TOOL/(d+"_frozen_before.json")),after_map_sha256=sha(TOOL/(d+"_frozen_after.json")))
for n,h in read(TOOL/"product_before.json").items():
    assert sha(ROOT/n)==h
    import hashlib
    assert hashlib.sha256(git("show","cf922891:"+n)).hexdigest()==h,n
changed=git("diff","--name-only","HEAD").decode().splitlines()
assert all(n.startswith(("artifacts/gn1/a4/","logs/")) for n in changed),changed
save("G1_scope.json",dict(passed=True,product_byte_identical_cf922891=True,frozen=frozen,changed_tracked=changed,product=read(TOOL/"product_before.json")))
seal=verify_seal()
for arm in seal["arms"]:verify_arm(arm,seal)
pin=read(TOOL/"engine_pin.json")
assert sha(TOOL/"engine_pin.json")==sha(TOOL.parent/"engine_pin.json")
source_map=read(TOOL.parent/"engine_sources_after.json")
actual={n:sha(Path(n) if Path(n).is_absolute() else ROOT/n) for n in source_map}
assert source_map==actual
save("engine_sources_verified.json",actual)
save("G3_pin.json",dict(passed=True,module_sha256=pin["sha256"],source_map_sha256=pin["source_map_sha256"],source_files=len(actual),module_copies=len(seal["arms"]),symlinks=0,builds=0))
print("G1 PASS; G3 PASS",len(seal["arms"]),"arms")
