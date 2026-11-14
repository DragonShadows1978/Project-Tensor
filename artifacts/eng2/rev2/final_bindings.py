"""Prior art: exact keyed comparison, SHA-256 (NIST FIPS 180-4); reused ENG2 evidence.norm_id.
No new algorithm. Binds freshly generated review rows and real A1 inputs; no device claims.
"""
from pathlib import Path
import ast, hashlib, json, os, re, subprocess, sys
ROOT=Path.cwd();OUT=ROOT/"artifacts/eng2/rev2";TOOL=ROOT/"artifacts/eng2/tool"
sys.path.insert(0,str(TOOL));import evidence
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
out={}
# Compare fresh constructed row hashes to both earlier generations, by normalized ID.
rowmaps={}
for source,root in [("rev1",ROOT/"artifacts/eng2/rev1"),("a1",ROOT/"artifacts/eng2/a1/rev1_out"),("rev2",OUT)]:
 rowmaps[source]={}
 for arm in ["baseline","candidate"]:
  for kind,count in [("f1bytes",35),("p302bits",20)]:
   records=[read(p) for p in (root/(arm+"_records")).glob(kind+"_*.json")]
   keyed={evidence.norm_id(r["test_id"]):r["sha256"] for r in records}
   assert len(records)==len(keyed)==count,(source,arm,kind)
   rowmaps[source][arm+"/"+kind]=keyed
assert rowmaps["rev1"]==rowmaps["a1"]==rowmaps["rev2"]
out["cross_generation_rows"]=dict(equal=True,counts={k:len(v) for k,v in rowmaps["rev2"].items()},construction="Fresh _record calls from actual collected IDs with constructed identical 64-one hash payloads, as in REV1; not device-output hashes",maps=rowmaps["rev2"])
# Full census row identity, and every newly refused name against baseline outcome.
old=read(OUT/"census_baseline.json")["rows"];new=read(OUT/"census_candidate.json")["rows"]
assert len(old)==len(new)==337
changed=[]
normalize=lambda r:re.sub(r"object at 0x[0-9a-f]+","object at ADDRESS",json.dumps(r,sort_keys=True))
for a,b in zip(old,new):
 assert (a["entry"],a["requested"],a["kind"])==(b["entry"],b["requested"],b["kind"])
 if normalize(a)!=normalize(b):changed.append(dict(entry=a["entry"],requested=a["requested"],kind=a["kind"],baseline=a,candidate=b))
out["census_delta"]=dict(rows_per_arm=337,changed_rows=changed,entry_count=len({r["entry"] for r in new}),known_omission="apa_sp2.RegisteredMargins.lookup; E2R2-F02")
# Registered mask supported names must be served with their existing bias rule.
mask=[r for r in new if r["entry"]=="functional._causal_mask" and r["kind"]=="supported"]
assert len(mask)==6 and all(r["outcome"]=="handled" for r in mask)
out["mask_supported"]=mask
# Preserve input hashes for the real receipts re-scored in this review.
inputs=[ROOT/"artifacts/eng2/a1/AG2_CPU.json",ROOT/"artifacts/eng2/a1/AG3_SCOPE.json",ROOT/"artifacts/eng2/a1/MUTANT_EVAL_CPU.json",ROOT/"artifacts/eng2/a1/TOOL_CHECKS.json",ROOT/"artifacts/eng2/a1/A1_SOURCE.diff",TOOL/"ids_eng2.txt",TOOL/"ids_audit.txt",TOOL/"staging_seal.json"]
for category,labels in [("cpu_mutants",["candidate_eng2","MF1","MF2"]),("cpu_pins",["candidate_regnorm"])]:
 for label in labels:
  base=TOOL/"gpu_runs/audit_2"/category/label/"attempt-0001"
  inputs += [base/n for n in ["child_result.json","junit.xml","forge.log"]]
out["input_sha256"]={str(p.relative_to(ROOT)):sha(p) for p in inputs}
assert len((TOOL/"ids_eng2.txt").read_text().splitlines())==298
assert len((TOOL/"ids_audit.txt").read_text().splitlines())==157
# Commit/load bindings; no working tree mutations by git.
out["review_head"]=subprocess.check_output(["git","--no-optional-locks","rev-parse","HEAD"],text=True).strip()
out["A1_commit"]=subprocess.check_output(["git","--no-optional-locks","rev-parse","1a4beed1"],text=True).strip()
out["all_checks_passed"]=True
(OUT/"final_bindings.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(dict(cross_generation_rows_equal=True,rows_per_arm=337,census_entries=out["census_delta"]["entry_count"],census_changed_rows=len(changed),input_hashes=len(inputs),mask_supported=6,review_head=out["review_head"]),indent=2))
