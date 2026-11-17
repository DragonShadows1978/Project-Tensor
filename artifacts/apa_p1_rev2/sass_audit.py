"""Stream exact Function blocks, normalizing ONLY nvcc anonymous suffixes.
Prior art: NVIDIA cuobjdump; SHA-256 (NIST 2015); P1-D per-function comparison
(Project-Tensor 2026). Independent implementation, no instruction normalization.
"""
from pathlib import Path
import collections,hashlib,json,re,subprocess
root=Path(__file__).resolve().parent
base=Path("/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1d/artifacts/apa_p1f1")
claim=json.loads((base/"SASS_COMPARISON_F_r2.json").read_text())
suffix=re.compile(r"_cu_[0-9a-fA-F]{8}(?:_[0-9]+)?")
def scan(label,path):
 with path.open("rb") as f: digest=hashlib.file_digest(f,"sha256").hexdigest()
 assert digest==claim["dumps"][label]["dump_sha256"]
 rows={}; raw={}; name=None; h=None; ns={}; matches=collections.Counter()
 with path.open() as f:
  for line in f:
   m=re.search(r"Function : (\S+)",line)
   if m:
    assert name is None
    actual=m[1];name=suffix.sub("_cu_BUILD",actual)
    assert name not in rows,("non-bijective function map",name)
    raw[name]=actual;h=hashlib.sha256();ns[name]=0
   elif name is not None and line.strip()=="..........":
    rows[name]=h.hexdigest();name=None
   elif name is not None:
    # Only permitted suffix substitutions, including any call targets.
    normalized=suffix.sub("_cu_BUILD",line)
    h.update(normalized.encode());ns[name]+=1
 assert name is None
 return rows,raw,digest,ns
A,AN,AS,AL=scan("B",base/"sass/B.sass")
B,BN,BS,BL=scan("F_r2",base/"sass/F_r2.sass")
keys=sorted(set(A)|set(B))
p=subprocess.run(["timeout","30s","c++filt"],input="\n".join([AN.get(k,BN.get(k)) for k in keys])+"\n",capture_output=True,text=True,check=True)
names=dict(zip(keys,p.stdout.splitlines()))
counts=collections.Counter();rows=[]
for k in keys:
 kind="added" if k not in A else "removed" if k not in B else "equal" if A[k]==B[k] else "changed"
 counts[kind]+=1
 rows.append(dict(key=k,demangled=names[k],kind=kind,B_sha256=A.get(k),F_sha256=B.get(k),B_lines=AL.get(k),F_lines=BL.get(k)))
for k in ("equal","changed","removed","added"):counts[k]+=0
# Verify one-to-one suffix mapping with exact source identifiers.
pairs={(AN[k],BN[k]) for k in set(AN)&set(BN) if AN[k]!=BN[k]}
assert len({a for a,b in pairs})==len({b for a,b in pairs})==len(pairs)
result=dict(method=__doc__,B_functions=len(A),F_functions=len(B),union=len(keys),counts=dict(counts),stored_counts=claim["counts"],stored_union=claim["kernels"],suffix_function_renames=len(pairs),stored_suffix_renames=claim["symbol_suffix_renames"],dump_sha256={"B":AS,"F":BS},differing=[x for x in rows if x["kind"]!="equal"],rows=rows)
(root/"SASS_AUDIT.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items() if k not in ("rows","differing")},indent=2))
for row in result["differing"]:print(row["kind"],row["demangled"])
print("retained spq6 variants",*[names[k] for k in keys if k in B and "bulk_mma_tq" in names[k]],sep="\n")
