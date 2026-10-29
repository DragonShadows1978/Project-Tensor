"""CPU disassembly comparison against immutable base binary.
Prior art: NVIDIA cuobjdump (CUDA toolkit; year unverified - lead to check).
Ours: specialization-name pairing, exact instruction text/encoding comparison.
This is static evidence, never a substitute for GPU output-byte comparisons.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess

P=Path(__file__).resolve().parent

def read(path):
    functions={};name=None;lines=[]
    with path.open() as f:
        for line in f:
            if 'Function :' in line:
                if name is not None:functions[name]=''.join(lines)
                name=line.split('Function :',1)[1].strip();lines=[]
            elif name is not None and ('/*' in line or '.headerflags' in line):
                lines.append(line.strip()+'\n')
    if name is not None:functions[name]=''.join(lines)
    names=list(functions)
    demangled=subprocess.run(['c++filt'],input='\n'.join(names)+'\n',capture_output=True,text=True,check=True).stdout.splitlines()
    return {d:functions[n] for n,d in zip(names,demangled)}

old=read(P/'scratch/base_all.sass');new=read(P/'scratch/arm_all.sass')
rows=[]
for name,code in new.items():
    family=next((tag for tag in ['apa_selective_bwd_kernel','apa_selective_bwd_bk1_kernel','apa_selective_bwd_bk2_kv_kernel','tc::bk3::backward','tc::pt_tf32::backward'] if tag in name),None)
    if family is None:continue
    args=re.search(r'<([^<>]*)>',name)
    assert args
    if not args[1].endswith(', false'):continue # STE=true is a separate specialization
    baseline=name[:args.start(1)]+args[1][:-7]+name[args.end(1):]
    exists=baseline in old
    rows.append({'name':name,'base_name':baseline,'base_exists':exists,'PASS':exists and code==old[baseline],'instruction_encoding_sha256':hashlib.sha256(code.encode()).hexdigest(),'base_instruction_encoding_sha256':hashlib.sha256(old[baseline].encode()).hexdigest() if exists else None})
forward=[]
for name,code in new.items():
    if any(tag in name for tag in ['apa_selective_fwd_train_kernel','tc::bk4::forward','tc::pt_tf32::forward']):
        forward.append({'name':name,'PASS':name in old and code==old[name]})
result={'evidence_class':'static disassembly against base 69b154cd, sm89','off_specializations':rows,'forward_specializations':forward,'PASS':bool(rows) and bool(forward) and all(x['PASS'] for x in rows+forward),'limits':'does not establish run-to-run atomic ordering or runtime output bytes'}
(P/'receipts/SASS_IDENTITY.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'PASS':result['PASS'],'off':len(rows),'off_identical':sum(x['PASS'] for x in rows),'forward':len(forward),'forward_identical':sum(x['PASS'] for x in forward)}))
