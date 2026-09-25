#!/usr/bin/env python3
"""One sequential, <=1800-second lead-owned GPU slot. Never acquires a lock.

Prior art: POSIX process groups/timeouts and Python subprocess (Python 2003),
PT-TF32-1 create-only receipts (2026), taken. Ours: one global deadline,
per-lane budgets, comprehensive remaining-lane BLOCKED receipts.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import pt_tf32_2 as t


def sequence(out):
    py=[sys.executable,'-B']
    tests=['-m','pytest','-q','tests/test_pt_tf32_gpu.py','tests/test_pt_tf32_2_gpu.py',
           '-p','no:cacheprovider']
    rows=[('gpu_units',75,py+tests+['--basetemp='+str(out/'unit_tmp')])]
    rows.append(('gemm',100,py+['scripts/pt_tf32_2.py','gemm','--lead-gpu','--out',str(out/'gemm')]))
    for i,budget in enumerate((75,75,140,140)):
        rows.append((f'attention_{i}',budget,py+['scripts/pt_tf32_2.py','attention','--case',str(i),'--lead-gpu','--out',str(out/f'attention_{i}')]))
    for tool,budget in [('memcheck',75),('racecheck',100),('synccheck',75)]:
        argv=['/usr/local/cuda-12.6/bin/compute-sanitizer','--tool',tool,'--error-exitcode','86']
        rows.append((tool,budget,argv+py+tests+['--basetemp='+str(out/(tool+'_tmp'))]))
    rows.append(('noise_floor',220,py+['scripts/pt_tf32_2_grapa.py','noise','--lead-gpu','--out',str(out/'noise_floor')]))
    for state in ('onset','healthy','control'):
        argv=py+['scripts/pt_tf32_2_grapa.py','state','--lead-gpu','--state',state,'--out',str(out/state)]
        if state!='onset':argv+=['--noise-floor',str(out/'noise_floor/NOISE_FLOOR.json')]
        rows.append((state,100,argv))
    rows.append(('step_time',300,py+['scripts/pt_tf32_2_grapa.py','timing','--lead-gpu','--out',str(out/'step_time')]))
    return rows


def sanitizer_clean(name,log):
    if name=='racecheck':
        return ('ERROR SUMMARY: 0 errors' in log or 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in log)
    return 'ERROR SUMMARY: 0 errors' in log


def run_lane(argv,log_path,env,seconds):
    start=time.monotonic()
    with log_path.open('x') as log:
        p=subprocess.Popen(argv,cwd=t.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            rc=p.wait(timeout=seconds);status='GREEN' if rc==0 else 'RED'
        except subprocess.TimeoutExpired:
            # This process group was created by this call, including its own
            # model children. Never enumerate or signal unrelated processes.
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=2)
            rc=p.returncode;status='BLOCKED_TIMEOUT'
    return dict(status=status,returncode=rc,elapsed_seconds=time.monotonic()-start,argv=argv,log=str(log_path))


def main():
    start=time.monotonic()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lead-gpu',action='store_true');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--print-sequence',action='store_true');args=p.parse_args()
    out=args.out.resolve()
    if not out.is_relative_to(t.ART):p.error('out must be a fresh PT-TF32-2 artifact directory')
    if args.print_sequence:
        print(json.dumps(sequence(out),indent=2));return 0
    if not args.lead_gpu or not os.environ.get('CUDA_VISIBLE_DEVICES'):p.error('BLOCKED: lead-owned GPU slot required')
    if ',' in os.environ['CUDA_VISIBLE_DEVICES']:p.error('exactly one visible device required')
    t.verify_manifest();out.mkdir(parents=True,exist_ok=False)
    temp=out/'tmp';temp.mkdir()
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PT_TF32_LEAD_GPU='1',TC_TF32_GEMM='0',
             TMPDIR=str(temp),CUDA_CACHE_PATH=str(out/'cuda_cache'),OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
    deadline=start+1800;receipts={}
    for name,budget,argv in sequence(out):
        remaining=deadline-time.monotonic()-5
        if remaining<=0:
            row=dict(status='BLOCKED',reason='global 1800-second deadline reached',argv=argv)
        elif name in ('healthy','control') and receipts.get('noise_floor',{}).get('status')!='GREEN':
            row=dict(status='BLOCKED',reason='noise-floor acquisition did not pass',argv=argv)
        else:
            print('LANE',name,'BUDGET',min(budget,remaining),flush=True)
            try:
                row=run_lane(argv,out/(name+'.log'),env,min(budget,remaining))
                if row['status']=='GREEN' and name in ('memcheck','racecheck','synccheck'):
                    if not sanitizer_clean(name,Path(row['log']).read_text()):
                        row.update(status='RED',reason='missing explicit zero-error sanitizer summary')
            except FileNotFoundError as exc:row=dict(status='BLOCKED',reason='required executable missing',error=repr(exc),argv=argv)
            except Exception as exc:row=dict(status='RED',error=repr(exc),argv=argv)
        receipts[name]=row;t.create_json(out/(name+'_receipt.json'),row)
        print('RESULT',name,row['status'],flush=True)
    ok=all(r['status']=='GREEN' for r in receipts.values())
    summary=dict(verdict='GREEN' if ok else 'RED_OR_BLOCKED',lanes=receipts,
        elapsed_seconds=time.monotonic()-start,registration_sha256=t.sha(t.REG),
        manifest_sha256=t.sha(t.ART/'SOURCE_MANIFEST.json'),
        evidence_class='single sequential lead slot; blind review remains separate')
    t.create_json(out/'SLOT_SUMMARY.json',summary)
    print('SLOT',summary['verdict'],'SECONDS',summary['elapsed_seconds']);return 0 if ok else 1


if __name__=='__main__':sys.exit(main())
