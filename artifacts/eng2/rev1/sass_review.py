"""Module-ID-only SASS differential check. Prior art: reproducible-builds/diffoscope normalization (year unverified -- lead to check); ENG2 G3 recipe. No instruction/whitespace normalization."""
from pathlib import Path
import subprocess,re,hashlib,json,shutil
root=Path.cwd(); out=root/"artifacts/eng2/rev1"; scratch=Path("/tmp/eng2_rev1_bootstrap")
assert shutil.disk_usage(scratch).free>60_000_000_000
rows={}; normalized={}
for arm in ["base","cand"]:
    module=next((root/"tensor_cuda/build-eng2"/arm/"module").glob("*.so"))
    command=["/usr/local/cuda-12.6/bin/cuobjdump","-sass",str(module)]
    r=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=240)
    assert r.returncode==0,r.stderr
    raw=r.stdout; norm,n=re.subn(rb"_[0-9a-f]{8}_[0-9]{7}",b"_MODID",raw)
    (scratch/(arm+".sass")).write_bytes(raw); (scratch/(arm+".normalized.sass")).write_bytes(norm)
    normalized[arm]=norm
    rows[arm]=dict(command=command,module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),raw_sha256=hashlib.sha256(raw).hexdigest(),normalized_sha256=hashlib.sha256(norm).hexdigest(),lines=len(raw.splitlines()),replacements=n,stderr=r.stderr.decode())
rows["equal"]=normalized["base"]==normalized["cand"]
rows["normalization"]="Only regex _[0-9a-f]{8}_[0-9]{7} replaced by _MODID; no path or whitespace changes"
(out/"sass_comparison.json").write_text(json.dumps(rows,indent=2)+"\n"); print(json.dumps(rows,indent=2))
assert rows["equal"]
