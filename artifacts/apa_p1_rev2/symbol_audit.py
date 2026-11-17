"""CPU symbol evidence. Prior art: GNU nm/c++filt, NVIDIA cuobjdump, SHA-256 (NIST 2015). No novel algorithm."""
from pathlib import Path
import hashlib,json,re,subprocess
root=Path(__file__).resolve().parent
base=Path("/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1d/artifacts")
mods=[("B","apa_p1d/build_B","8728e67e26027dbafe8a61592decb38e68171fa7b5634239ce0af7897c4e8c4e"),("F","apa_p1f1/build_F_r2","5c17147715ffc422b0aa7ba8fae826c554a4caff63b1231c22ce580867efc243"),("FH","apa_p1f1/build_FH","19b4a1783e981302522e79654c278e0e3c462606c2b3687c11876395756be2b6")]
summary=[]
for label,pin,digest in mods:
 so=base/pin/"lib/_tensor_cuda.cpython-312-x86_64-linux-gnu.so"
 outputs={}
 for kind,argv in [("nm",["nm","-C",str(so)]),("exports",["nm","-D","-C","--defined-only",str(so)]),("device",["cuobjdump","--dump-elf-symbols",str(so)])]:
  with so.open("rb") as f:actual=hashlib.file_digest(f,"sha256").hexdigest()
  assert actual==digest
  cmd=["timeout","120s"]+argv
  p=subprocess.run(cmd,capture_output=True,text=True,check=True)
  (root/"receipts"/("symbols_"+label+"_"+kind+".log")).write_text(p.stdout+p.stderr)
  outputs[kind]=p.stdout
  print("COMMAND",cmd,"exit",p.returncode,"lines",len(p.stdout.splitlines()))
 hook=re.compile(r"_p1_test|EarlyCounterHook|early_counter_|apa_p1f1_merge_tickets")
 d={"module":label,"sha256":digest,"initializers":[s for s in outputs["exports"].splitlines() if "PyInit_" in s],"hooks":{k:[s for s in out.splitlines() if hook.search(s)] for k,out in outputs.items()},"eb2b_symbols":[s for s in outputs["nm"].splitlines() if "apa_eb2b_run(" in s or "apa_p1f1_eb2b_run(" in s],"undefined_eb2b":[s for s in outputs["nm"].splitlines() if " U " in s and "eb2b" in s]}
 summary.append(d)
 print(json.dumps({**d,"hooks":{k:{"count":len(v),"examples":v[:4]} for k,v in d["hooks"].items()}},indent=2))
(root/"SYMBOL_AUDIT.json").write_text(json.dumps(summary,indent=2)+"\n")
