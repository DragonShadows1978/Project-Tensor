"""Finalize review receipts and a standard SHA-256 file manifest.
Prior art: SHA-256 (NIST FIPS 180-4, 2015), ordinary file-manifest practice.
No novel algorithm. The manifest excludes itself to avoid a self hash cycle.
"""
from pathlib import Path
import collections,datetime,hashlib,json
root=Path(__file__).resolve().parent
findings=json.loads((root/"findings.json").read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
counts=dict(collections.Counter(x["severity"] for x in findings["findings"]))
message="\n".join(["FINAL CPU REVIEW COMPLETE; GPU REPRODUCTIONS STAGED ONLY", "Command: TMPDIR=/tmp/p1_rev2_review CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 120s nice -n 19 python3 artifacts/apa_p1_rev2/finalize_artifacts.py", "findings.json sha256 "+sha(root/"findings.json"),"REPORT.md sha256 "+sha(root/"REPORT.md"),"active finding counts "+json.dumps(counts,sort_keys=True),"cleanup proposed for lead only: /tmp/p1_rev2_review/", "No deletes performed; no product edits; no builds or GPU runs."])+"\n"
with (root/"LEDGER.jsonl").open("a") as f:f.write(json.dumps({"event":"review_complete","utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"findings_sha256":sha(root/"findings.json"),"counts":counts,"gpu_reproductions":"three staged and compiled only","product_edits":False,"deletes":False,"prior_art":"SHA-256 NIST2015; sources per finding/script/report"})+"\n")
(root/"FINAL_CHECK.log").write_text(message)
files=sorted(p for p in root.rglob("*") if p.is_file() and p.name!="MANIFEST.sha256")
lines=[sha(p)+"  "+str(p.relative_to(root)) for p in files]
(root/"MANIFEST.sha256").write_text("\n".join(lines)+"\n")
for p,line in zip(files,lines):assert sha(p)==line.split("  ",1)[0]
print(message,end="")
print("manifest files",len(files),"manifest sha256",sha(root/"MANIFEST.sha256"))
