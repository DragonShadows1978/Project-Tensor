# Prior art: Project-Tensor SPQ1/SPQ4/SPQ6/EB2b/EB3/EB4 (2026), unchanged.
# Git (2005) three-way merge supplied survey S5's dry run.
from pathlib import Path
import subprocess, hashlib, json
A=Path('artifacts/apa_p1')
def show(p): return subprocess.check_output(['git','show','aaa9b4ea:'+p])
headers='d1 parallel tensor packed launcher baseline_dump merge packed_launcher spq4 spq6 eb2b_values eb2b eb3_tuning eb3 eb4_tuning eb4'.split()
ident=[]
for n in headers:
    p=Path('tensor_cuda/src/apa_spo')/(n+'.cuh'); p.parent.mkdir(exist_ok=True)
    assert not p.exists(),p
    data=show(str(p)); p.write_bytes(data)
    ident.append(dict(path=str(p),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
hunks=[]
def append(p,data,source,comment='//',guard=False):
    f=Path(p); old=f.read_bytes(); assert old.endswith(b'\n')
    assert old==subprocess.check_output(['git','show','0e1dea3:'+p]),p
    extra='\n'+comment+' APA-P1 port: Git three-way merge (2005), survey S5 dry run; inherited\n'
    extra+=comment+' Project-Tensor SPQ1/SPQ4/SPQ6/EB2b/EB3/EB4 (2026), unchanged numerics.\n'
    if guard: extra+='#ifdef TC_APA_PACKED\n'
    extra+=data
    if guard: extra+='\n#endif  // TC_APA_PACKED\n'
    f.write_bytes(old+extra.encode())
    hunks.append(dict(path=p,production_lines=len(old.splitlines()),added_start=len(old.splitlines())+1,added_end=len(f.read_bytes().splitlines()),source=source,design='D2',prior_art='Git 2005; Project-Tensor SPQ1/SPQ4/SPQ6/EB2b/EB3/EB4 2026'))
def extract(p,begin,end):
    t=show(p).decode(); a=t.index(begin); b=t.index(end,a)+len(end)
    return t[a:b]+'\n'
append('tensor_cuda/src/kernels.cu',extract('tensor_cuda/src/kernels.cu','// APA_SPO2_D1_BEGIN include','// APA_SPO2_D1_END include'),'aaa9b4ea:8191-8197',guard=True)
append('tensor_cuda/include/tc/ops.h',extract('tensor_cuda/include/tc/ops.h','// APA_SPO2_D1_BEGIN declaration','// APA_SPO2_D1_END declaration'),'aaa9b4ea:302-342',guard=True)
append('tensor_cuda/src/bindings.cpp',extract('tensor_cuda/src/bindings.cpp','// APA_SPO2_D1_BEGIN binding','// APA_SPO2_D1_END binding'),'aaa9b4ea:986-1064 (companion initializer only)',guard=True)
append('tensor_cuda/tensor_cuda/__init__.py',extract('tensor_cuda/tensor_cuda/__init__.py','# APA_SPO2_D1_BEGIN binding','# APA_SPO2_D1_END binding'),'aaa9b4ea:1290-1422 (loader/workspace/wrappers)',comment='#')
append('tensor_cuda/CMakeLists.txt','''# Default-off opt-in packed decode; AB-1 main feature definitions are applied
# consistently to host bindings and CUDA. Production's initializer is retained.
option(TC_APA_PACKED "Build opt-in packed APA SP/SP-tk8 decode entries" OFF)
if(TC_APA_PACKED)
  target_compile_definitions(_tensor_cuda PRIVATE
    TC_APA_PACKED=1 TC_APA_EB2B=1 TC_APA_EB3=1 TC_APA_EB4=1
    APA_EB2B_MUTANT=0 APA_FXTC1_VARIANT_DEFAULT=2)
endif()
''','survey S5 step 3 + aaa9b4ea:scripts/apa_ab1_build.sh main defines',comment='#')
(A/'header_identity_seed.json').write_text(json.dumps(ident,indent=2)+'\n')
(A/'hunk_map.json').write_text(json.dumps(hunks,indent=2)+'\n')
print('Copied',len(ident),'headers and appended',len(hunks),'seams.')
