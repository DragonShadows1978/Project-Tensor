"""Evidence-only gate recomputation. Prior art: SHA256/NIST, GNU diff and pytest/JUnit; ENG1/REV1 receipt method.
Taken: exact identities/IDs/completed-child exits. Ours: A1 scope and registered acceptance aggregation.
"""
import hashlib,json,os,subprocess,sys,datetime
from pathlib import Path
R=Path(__file__).resolve().parents[3];A=R/'artifacts/eng1/a1';T=R/'artifacts/eng1/tool'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def git(*args):return subprocess.run(['git',*args],cwd=R,capture_output=True,check=True,text=True).stdout
before=read(A/'tensor_cuda_before.json')
after={str(p.relative_to(R)):sha(p) for p in sorted((R/'tensor_cuda').rglob('*')) if p.is_file() and not any(x.startswith('build-') for x in p.relative_to(R/'tensor_cuda').parts)}
(A/'tensor_cuda_after.json').write_text(json.dumps(after,indent=2)+'\n')
changed=sorted(p for p in set(before)|set(after) if before.get(p)!=after.get(p))
diff=git('diff','5031c765','--','.',':!artifacts/eng1',':!orders',':!logs')
(A/'test_eng1.diff').write_text(diff)
paths=git('diff','--name-only','5031c765','--','.',':!artifacts/eng1',':!orders',':!logs').splitlines()
pin='3e6d5805fb8b89ebfeb2b32c173fd097ef1cfe4a01b3349dc177d4b0c8c0ad53'
module=T/'arms/candidate/tensor_cuda/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so'
module_work=R/'tensor_cuda/build-eng1/module'/module.name
rev_before=read(A/'rev1_before.json');rev_after={str(p.relative_to(R)):sha(p) for p in sorted((R/'artifacts/eng1/rev1').rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
old_tool=read(A/'tool_before.json')
old_seal='artifacts/eng1/tool/staging_seal.json'
G1=dict(passed=paths==['tensor_cuda/tests/test_eng1.py'] and changed==['tensor_cuda/tests/test_eng1.py'] and sha(module)==sha(module_work)==pin and rev_before==rev_after and sha(R/old_seal)==old_tool[old_seal],
 dispatch=git('rev-parse','HEAD').strip(),diff_base='5031c765',diff_paths=paths,map_changed=changed,source_byte_identical=not any(p.startswith('tensor_cuda/src/') for p in changed),candidate_sha256=sha(module),worktree_candidate_sha256=sha(module_work),review_receipts_unchanged=rev_before==rev_after,old_seal_sha256=sha(R/old_seal),old_seal_unchanged=sha(R/old_seal)==old_tool[old_seal])
cpu=read(A/'candidate_attempt2_pytest.json');cp=read(A/'candidate_attempt2_parent.json');mut=read(A/'M16_ow_zero_pytest.json');mp=read(A/'M16_ow_zero_parent.json')
passed_cpu=[r['id'] for r in cpu['reports'] if r['when']=='call' and r['outcome']=='passed']
failed_mut=[r['id'] for r in mut['reports'] if r['when']=='call' and r['outcome']=='failed']
normalize=lambda x:x[x.index('tensor_cuda/tests/test_eng1.py'):] if '/tensor_cuda/tests/test_eng1.py' in x else x
new_ids=['tensor_cuda/tests/test_eng1.py::test_e2_pow_grad_nonpow2['+d+']' for d in ('float16','bfloat16')]+['tensor_cuda/tests/test_eng1.py::test_e5_refuse_cpu['+c+'-ow_zero]' for c in ('im2col','col2im','avg_pool2d','max_pool2d')]
collected=(T/'ids_a1_eng1_candidate.txt').read_text().splitlines();repros=read(A/'review_reproductions.json');rows=read(A/'e3_emulation.json')
killers=[x for x in new_ids if 'ow_zero' in x]
G2=dict(passed=cp['exit']==cpu['exit']==0 and len(passed_cpu)==47 and set(new_ids)<=set(collected) and mp['exit']==mut['exit']==1 and len(failed_mut)==4 and {normalize(x) for x in failed_mut}==set(killers) and set(killers)<={normalize(x) for x in passed_cpu} and len(rows)==40 and all(x['absolute_difference']<=x['reproduction_tolerance'] for x in repros),
 collected=len(collected),cpu_passed=len(passed_cpu),new_ids=new_ids,M16_killers=[normalize(x) for x in failed_mut],reproductions=repros,emulation_rows=len(rows),M15_status='built; CPU arithmetic prediction passes instrument checks; actual engine kill GPU-only G5')
source=read(T/'source_reading_tests.json');base_ids=(T/'ids_a1_regression.txt').read_text().splitlines();cand_ids=(T/'ids_a1_regression_candidate.txt').read_text().splitlines();excluded={x['test_id'] for x in source}
isolation={arm:read(A/('isolation_'+arm+'.json')) for arm in ('baseline','candidate')}
G3=dict(passed=all(x['after']['exit']==0 and x['after']['sha256']==x['pin']==x['origin']['observed']['engine_sha256'] for x in isolation.values()) and base_ids==[i for i in cand_ids if i.split('[',1)[0] not in excluded] and read(A/'origin_refusal.json')['passed'],
 origins={arm:dict(pin=x['pin'],package=x['origin']['observed']['tensor_cuda_file'],engine=x['origin']['observed']['engine_file'],before_exit=x['before']['exit'],after_exit=x['after']['exit']) for arm,x in isolation.items()},excluded_ids=sorted(excluded),baseline_comparison_ids=len(base_ids),candidate_ids=len(cand_ids),wrong_pin_refused=read(A/'origin_refusal.json')['passed'])
sys.path.insert(0,str(T));from evidence import verify_seal,verify_start,recompute_line,lines,SEAL_NAME
seal,start=verify_start(T/'audit_a1_final');audit=recompute_line(T/'audit_a1_final','candidate_eng1',seal,start)
staged=read(T/'gpu_lines.json');syntax=subprocess.run(['bash','-n',str(T/'STAGED_GPU_COMMANDS.sh')],capture_output=True,text=True)
G4=dict(passed=audit['passed'] and audit['mode']=='cpu' and len(staged)==24 and not seal['arms_pending'] and all(x['expected_child_exits'] and x['expected_parent_exit']==0 for x in staged) and syntax.returncode==0,
 line_count=len(staged),sealed_arms=len(seal['arms']),audit=dict(passed=audit['passed'],mode=audit['mode'],selected_attempt=audit['selected_attempt']),seal_file=SEAL_NAME,seal_sha256=sha(T/SEAL_NAME),script_sha256=sha(T/'STAGED_GPU_COMMANDS.sh'),expected_exits={x['label']:x['expected_child_exits'] for x in staged})
gates=dict(G1=G1,G2=G2,G3=G3,G4=G4,accepted=all(x['passed'] for x in (G1,G2,G3,G4)),evidence_class='scope identity, CPU unit runs, arithmetic instrument checks and CPU audit only; no GPU execution')
(A/'gates.json').write_text(json.dumps(gates,indent=2)+'\n')
print(json.dumps({k:v['passed'] for k,v in gates.items() if k in ('G1','G2','G3','G4')}));print('seal',G4['seal_sha256'],'accepted',gates['accepted'])
assert gates['accepted']
