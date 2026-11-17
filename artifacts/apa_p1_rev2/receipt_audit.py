"""Recompute stored evidence from arrays, logs and per-ID records.
Prior art: NumPy unsigned-view bit equality (Harris et al., 2020; P1-D 2026),
SHA-256 (NIST, 2015). Identity excludes timing/log text; no new algorithm.
"""
from pathlib import Path
import collections,hashlib,json,re
import numpy as np
root=Path(__file__).resolve().parent
base=Path("/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1d/artifacts/apa_p1f1")
b=base/"gpu";allrows=[];summ=[]
def hashfile(p):
 with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
for label,pa,pb,metaA,metaB in [("public",b/"LF1/l1_B/arm_B.npz",b/"LF1/l1_F/arm_B.npz",b/"LF1/l1_B/arm_B.json",b/"LF1/l1_F/arm_B.json"),("raw",b/"LF1/raw_B.npz",b/"LF1/raw_F.npz",b/"LF1/raw_B.json",b/"LF1/raw_F.json")]:
 ma,mb=[json.loads(p.read_text()) for p in (metaA,metaB)]
 assert hashfile(pa)==ma["npz_sha256"] and hashfile(pb)==mb["npz_sha256"]
 with np.load(pa,allow_pickle=False) as za,np.load(pb,allow_pickle=False) as zb:
  ka,kb=set(za.files),set(zb.files);assert ka==kb
  rows=[]
  for key in sorted(ka):
   a,c=za[key],zb[key]
   shape_dtype=a.shape==c.shape and a.dtype==c.dtype
   assert shape_dtype
   unsigned={1:np.uint8,2:np.uint16,4:np.uint32,8:np.uint64}[a.dtype.itemsize]
   u,v=[np.ascontiguousarray(x).view(unsigned).ravel() for x in (a,c)]
   informative=[bool(x.size and np.any(x!=0) and np.any(x!=x[0])) for x in (u,v)]
   # L19: informative checked before each bitwise comparison.
   timing_excluded=(label=="raw" and "/eb2b/" in key and key.endswith("/10"))
   equal=bool(np.array_equal(u,v)) if all(informative) and not timing_excluded else None
   row=dict(group=label,key=key,shape=list(a.shape),dtype=str(a.dtype),informative=informative,bitwise_equal=equal,timing_excluded=timing_excluded)
   rows.append(row)
  allrows+=rows
  s=dict(group=label,computed_arrays=len(ka),stored_B_arrays=ma["arrays"],stored_F_arrays=mb["arrays"],bitwise_equal=sum(x["bitwise_equal"] is True for x in rows),not_equal=[x["key"] for x in rows if x["bitwise_equal"] is False],not_informative=[x["key"] for x in rows if not all(x["informative"])],timing_excluded=[x["key"] for x in rows if x["timing_excluded"]])
  summ.append(s);print(json.dumps(s))
stored=json.loads((b/"LF1/RESULT.json").read_text())
print("LF1 computed total",len(allrows),"stored",stored["comparisons"],"computed equal",sum(x["bitwise_equal"] is True for x in allrows),"stored",stored["bitwise_equal"])
# Per-check counts from raw row/check lists, not summary counts.
checks=[]
for name in ["LF2/RESULT.json","LF2S/RESULT_F_supp.json","LF3/RESULT.json","LF4/RESULT.json","LF5/RESULT.json"]:
 d=json.loads((b/name).read_text());rr=d.get("checks",d.get("rows",[]))
 x=dict(file=name,stored_status=d.get("status"),computed_count=len(rr),computed_true=sum(v.get("ok") is True for v in rr),computed_false=sum(v.get("ok") is False for v in rr))
 checks.append(x);print(json.dumps(x))
# Log hashes and both exit classes from every line receipt in the top-level GPU folder.
exits=[]
for p in sorted(b.glob("*.exit")):
 d=json.loads(p.read_text());log=p.with_suffix(".log");actual=hashfile(log)
 x=dict(file=p.name,forge_exit=d["forge_exit"],inner_timeout_exit=d["inner_timeout_exit"],actual_log_sha256=actual,stored_log_sha256=d["log_sha256"],matches=actual==d["log_sha256"])
 assert x["matches"];exits.append(x)
print("verified GPU line exit/log receipts",len(exits))
# New CPU tests: exact IDs from this review's raw verbose log.
text=(root/"receipts/025_cpu_pytest.log").read_text()
ids=re.findall(r"^(tensor_cuda/tests/\S+::.*?) PASSED",text,re.M)
counter=collections.Counter(x.split("::")[0] for x in ids)
print("CPU computed per file",dict(counter),"computed total",len(ids),"stored pytest total 55")
result=dict(module_identity="LF1 B vs F; one build per arm, same device as recorded by the original harness; this audit is CPU-only",arrays_summary=summ,arrays=allrows,stored_LF1={k:stored[k] for k in ("comparisons","bitwise_equal","both_sides_informative")},check_counts=checks,gpu_exit_receipts=exits,cpu_ids=ids,cpu_counts=dict(counter))
(root/"RECEIPT_AUDIT.json").write_text(json.dumps(result,indent=2)+"\n")
