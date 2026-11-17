"""Independent, streaming Function-block hashes. Prior art: NVIDIA cuobjdump Function blocks;
SHA256 (NIST FIPS180-4, 2015). String canonicalization: no prior art known to me.
No import of F2/lead comparison code. Body bytes are never normalized.
"""
import collections,hashlib,json,pathlib,re,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[2];OUT=ROOT/"artifacts/apa_p1_rev3"
inputs=json.loads((OUT/"SASS_INPUTS.json").read_text())
patterns=[(r"__nv_static_[0-9]+__[0-9a-f]{8}_","__nv_static_LEN__PATH_"),
          (r"[0-9]+_GLOBAL__N__[0-9a-f]{8}_","LEN_GLOBAL__N__PATH_"),
          (r"_cu_[0-9a-f]{8}(?:_[0-9]+)?","_cu_UNIT")]
rewrites={}
def canonical(s):
 for regex,replacement in patterns:
  for m in re.finditer(regex,s):rewrites.setdefault(m[0],replacement)
  s=re.sub(regex,replacement,s)
 return s
allblocks={};raw={}
for label,r in inputs.items():
 p=pathlib.Path(r["path"])
 with p.open("rb") as f:assert hashlib.file_digest(f,"sha256").hexdigest()==r["sha256"],"dump sha mismatch"
 blocks={};raw[label]={};symbol=None;body_hash=None;nlines=0
 with p.open("rb") as f:
  for line in f:
   if b"Function : " in line:
    assert symbol is None,"unterminated function"
    symbol=line.split(b"Function : ",1)[1].strip().decode();body_hash=hashlib.sha256();nlines=0
   elif symbol is not None and line.strip()==b"..........":
    key=canonical(symbol);assert key not in blocks,("non-bijective",label,key)
    blocks[key]={"symbol":symbol,"sha256":body_hash.hexdigest(),"body_lines":nlines}
    raw[label][symbol]=body_hash.hexdigest();symbol=None
   elif symbol is not None:body_hash.update(line);nlines+=1
 assert symbol is None,"unfinished dump"
 allblocks[label]=blocks
A,B=allblocks["F"],allblocks["F2"];assert len(A)==len(B)==781
rows=[]
for key in sorted(set(A)|set(B)):
 a,b=A.get(key),B.get(key)
 kind="added" if a is None else "removed" if b is None else "equal" if a["sha256"]==b["sha256"] else "changed"
 rows.append({"canonical":key,"kind":kind,"F":a,"F2":b})
changed=[r for r in rows if r["kind"]!="equal"]
names=[r["F2"]["symbol"] for r in changed if r["F2"]]
demangled=subprocess.run(["c++filt"],input="\n".join(names)+"\n",text=True,capture_output=True,check=True).stdout.splitlines()
for r,name in zip(changed,demangled):r["demangled"]=name
raw_a,raw_b=raw["F"],raw["F2"];common=set(raw_a)&set(raw_b)
result={"method":"stream sha-verified dumps; exact body-byte SHA256; independent bijective symbol normalization; no body rewriting",
 "normalization_patterns":patterns,"rewrites":rewrites,"input":inputs,"own":{"F":len(A),"F2":len(B),"paired":sum(r["F"] is not None and r["F2"] is not None for r in rows),"counts":dict(collections.Counter(r["kind"] for r in rows)),"paired_symbol_renames":sum(r["F"] and r["F2"] and r["F"]["symbol"]!=r["F2"]["symbol"] for r in rows)},
 "raw":{"paired":len(common),"equal":sum(raw_a[k]==raw_b[k] for k in common),"changed":sum(raw_a[k]!=raw_b[k] for k in common),"only_F":len(set(raw_a)-set(raw_b)),"only_F2":len(set(raw_b)-set(raw_a))},"differing":changed,"rows":rows}
assert result["own"]["counts"]=={"equal":778,"changed":3},result["own"]
assert all("tc::spq1::prepare<" in r["demangled"] for r in changed)
# Read the lead receipt only after computing the independent result.
lead=json.loads((ROOT/"artifacts/apa_p1f2/lead_f2_sass_check.json").read_text())
result["lead_receipt"]=lead
(OUT/"SASS_RECOUNT.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items() if k not in ("rows","rewrites")},indent=2))
