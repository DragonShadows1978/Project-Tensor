import os
os.environ["GN1_CONTROL"] = "1"  # A6 D1: CPU fixture root cannot accept.
"""A3's four supplementary targets through A4's parent/child/scorer.
Prior art: Meszaros (2007) fault injection; A3 G4_extra; SHA-256 FIPS 180-4.
Only A4 receipt/seal copies are perturbed; restored in finally blocks.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
from evidence import TOOL, lines, read
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
ROWS=[]
runs=TOOL/'hostile_runs/Full_valid_synthetic'
env=dict(os.environ,RUNS=str(runs))


def run(args, **extra):
    e=dict(env,**extra)
    with (TOOL/'extra_launches.jsonl').open('a') as f:
        f.write(json.dumps(dict(args=args,RUNS=e['RUNS']))+'\n')
    p=subprocess.run(['timeout','60s',sys.executable,'-B',*args],cwd=TOOL,env=e,capture_output=True,text=True)
    return dict(exit=p.returncode,stdout=p.stdout[-1500:],stderr=p.stderr[-2000:])


p=runs/'candidate_focus/attempt-0001/parent_result.json'
original=p.read_text()
try:
    d=json.loads(original);d['timeout_cause']='outer-timeout';p.write_text(json.dumps(d))
    r=run(['score_gpu.py'])
    s=read(runs/'G5_SUMMARY.json')
    ROWS.append(dict(control='Stored_timeout_cause_tamper',target='stored timeout cause differs from log',
                     passed=r['exit']==1 and 'stored timeout cause differs from log' in str(s),result=r))
finally:
    p.write_text(original)

p=TOOL/'staging_seal.json'
original=p.read_text()
try:
    d=json.loads(original);d['import_modules']['scripts/apa_om135_core.py']='0'*64
    p.write_text(json.dumps(d,indent=2)+'\n')
    child=TOOL/'extra_child_attempt';child.mkdir(exist_ok=False)
    line=lines()['candidate_focus']
    for name,args,extra in (
        ('parent',['run_gpu_line.py','candidate_focus','0',str(line['outer']),sys.executable,
                   'hostile_forge.py','--min',str(line['minimum']),'--max',str(line['maximum']),
                   '--','timeout',str(line['inner'])+'s','python3','run_gpu_tests.py','candidate','focus'],{}),
        ('child',['run_gpu_tests.py','candidate','focus'],dict(GN1_ATTEMPT=str(child),GN1_LINE='candidate_focus')),
        ('scorer',['score_gpu.py'],{})):
        r=run(args,**extra)
        ROWS.append(dict(control='Import_helper_pin_'+name,target='import module seal mismatch: scripts/apa_om135_core.py',
                         passed=r['exit']!=0 and 'import module seal mismatch: scripts/apa_om135_core.py' in str(r),result=r))
finally:
    p.write_text(original)
r=run(['score_gpu.py'])
out=dict(passed=all(x['passed'] for x in ROWS) and r['exit']==1 and all(v['passed'] for v in read(runs/'G5_SUMMARY.json')['lines'].values()),controls=ROWS,restored_full_valid_scorer_exit=r['exit'])
(TOOL/'G4_extra.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(passed=out['passed'],controls=len(ROWS))))
sys.exit(0 if out['passed'] else 1)
