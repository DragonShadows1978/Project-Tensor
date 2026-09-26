#!/usr/bin/env python3
"""One sequential, <=1200-second lead-owned GPU slot. Never acquires a lock.

Prior art: POSIX process groups/timeouts and Python subprocess (Python 2003),
PT-TF32-1 create-only receipts (2026), taken. Ours: one global deadline,
per-lane budgets, comprehensive remaining-lane BLOCKED receipts.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import re
import subprocess
import sys
import time
import pt_tf32_4 as t
import pt_tf32_3_storage as storage
import pt_det_1 as det


def sequence(out,keep_grads=False,include_model=False,det_repro=None):
    py=[sys.executable,'-B']
    tests=['-m','pytest','-q','tests/test_pt_tf32_gpu.py','tests/test_pt_tf32_2_gpu.py','tests/test_pt_tf32_3_gpu.py',
           '-p','no:cacheprovider']
    rules=t.registration()['slot'];budgets=rules['budgets_seconds']
    rows=[('dispatch',budgets['dispatch'],py+['scripts/pt_tf32_4.py','dispatch','--lead-gpu','--out',str(out/'dispatch')]),
          ('gpu_units',budgets['gpu_units'],py+tests+['--basetemp='+str(out/'unit_tmp')])]
    rows.append(('gemm',budgets['gemm'],py+['scripts/pt_tf32_4.py','gemm','--lead-gpu','--out',str(out/'gemm')]))
    for i in range(4):
        budget=budgets[f'attention_{i}']
        rows.append((f'attention_{i}',budget,py+['scripts/pt_tf32_4.py','attention','--case',str(i),'--lead-gpu','--out',str(out/f'attention_{i}')]))
    for tool in ('memcheck','racecheck','synccheck'):
        budget=budgets[tool]
        argv=['/usr/local/cuda-12.6/bin/compute-sanitizer','--tool',tool,'--error-exitcode','86']
        rows.append((tool,budget,argv+py+tests+['--basetemp='+str(out/(tool+'_tmp'))]))
    if include_model:
        mb=rules['optional_model_budgets_seconds']
        rows.append(('noise_floor',mb['noise_floor'],py+['scripts/pt_tf32_4_grapa.py','noise','--lead-gpu','--out',str(out/'noise_floor')]))
        for state in ('onset','healthy','control'):
            argv=py+['scripts/pt_tf32_4_grapa.py','state','--lead-gpu','--state',state,'--out',str(out/state)]
            if state!='onset':argv+=['--noise-floor',str(out/'noise_floor/NOISE_FLOOR.json')]
            rows.append((state,mb[state],argv))
        rows.append(('step_time',mb['step_time'],py+['scripts/pt_tf32_4_grapa.py','timing','--lead-gpu','--out',str(out/'step_time')]))
    if keep_grads:
        for name,_,argv in rows:
            if name in ('noise_floor','onset','healthy','control','step_time'):argv.append('--keep-grads')
    # Prior art: required certification receipts (CC46/PT-TF32, 2026), taken.
    # Replays run in their own <=1500 s slot; this lane revalidates the exact
    # source/build receipt before any GPU work and has no additional GPU budget.
    rows.append(('pt_det_1_repro',0,py+['scripts/pt_det_1.py','verify-repro']+
                 (['--receipt',str(det_repro)] if det_repro is not None else [])))
    return rows


def sanitizer_result(name,log):
    # Prior art: NVIDIA Compute Sanitizer (2024) summary syntax, taken. Ours:
    # independent sanitizer verdict, preserve test rc and reject absent coverage.
    counts=[int(n) for n in re.findall(r'ERROR SUMMARY:\s*(\d+) errors?\b',log)]
    warnings=[]
    if name=='racecheck':
        for err,warn in re.findall(r'RACECHECK SUMMARY:.*?\((\d+) errors?, (\d+) warnings?\)',log):
            counts.append(int(err));warnings.append(int(warn))
    completed=sum(int(n) for n in re.findall(r'\b(\d+) (?:passed|failed|error(?:s)?)\b',log))
    if not counts or completed==0 or 'No attachable process found' in log or 'No CUDA' in log:
        return dict(status='BLOCKED',reason='missing sanitizer summary or executed test coverage',error_counts=counts)
    return dict(status='GREEN' if max(counts)==0 else 'RED',error_counts=counts,
                errors=max(counts),warning_counts=warnings,executed_tests=completed)


def sanitizer_clean(name,log):return sanitizer_result(name,log)['status']=='GREEN'


def run_lane(argv,log_path,env,seconds):
    check=storage.space_status(log_path.parent)
    if check['status']!='GREEN':return dict(check,argv=argv,returncode=None,elapsed_seconds=0.)
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
    text=log_path.read_text()
    if status!='BLOCKED_TIMEOUT' and ('No space left on device' in text or 'free space below 8 GiB' in text):
        status='BLOCKED'
    return dict(status=status,returncode=rc,elapsed_seconds=time.monotonic()-start,argv=argv,
                log=str(log_path),space_check=check)


def dump_estimate(keep):
    # Prior art: ZIP central-directory sizes (PKWARE 1989), taken; estimate
    # full optional payload from existing registered files without loading arrays.
    import zipfile
    reg=t.ROOT/'artifacts/pt_tf32_2/GRAPA_REGISTRATION_002.json'
    r=json.loads(reg.read_text());ref=Path(r['states']['onset']['refs']['fp64']['path'])
    with zipfile.ZipFile(ref) as f:
        # Conservative archive bound: FP64 references can be larger than FP32.
        grad=sum(i.file_size+len(i.filename)*2+256 for i in f.infolist())
    ckpt=Path(r['timing']['source']['path']).stat().st_size
    return dict(keep_grads=keep,default_gradient_checkpoint_dump_bytes=0,
                gradient_archives=10 if keep else 0,checkpoint_writes=2 if keep else 0,
                total_dump_bytes_upper_estimate=10*grad+2*ckpt if keep else 0,
                excludes='small JSON/text/token-batch receipts, CUDA cache; optional checkpoint size estimated from source')


def main():
    start=time.monotonic()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lead-gpu',action='store_true');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--print-sequence',action='store_true')
    p.add_argument('--include-model',action='store_true',help='Rerun unchanged model lanes within the same 1200-second deadline')
    p.add_argument('--det-repro',type=Path,help='Required GREEN PT-DET-1 repro/summary.json for this exact source and engine')
    p.add_argument('--keep-grads',action='store_true');args=p.parse_args()
    out=args.out.resolve()
    if not out.is_relative_to(t.ART):p.error('out must be a fresh PT-TF32-4 artifact directory')
    if args.print_sequence:
        rows=sequence(out,args.keep_grads,args.include_model,args.det_repro)
        print(json.dumps(dict(lanes=rows,lane_budget_seconds=sum(r[1] for r in rows),
            global_deadline_seconds=1200,storage=dump_estimate(args.keep_grads)),indent=2));return 0
    if not args.lead_gpu or not os.environ.get('CUDA_VISIBLE_DEVICES'):p.error('BLOCKED: lead-owned GPU slot required')
    if ',' in os.environ['CUDA_VISIBLE_DEVICES']:p.error('exactly one visible device required')
    out.mkdir(parents=True,exist_ok=False)
    try:
        if args.det_repro is None:raise det.Blocked('required PT_DET_1 repro receipt missing (--det-repro)')
        det.verify_repro(args.det_repro)
        repro_lane=dict(status='GREEN',receipt=str(args.det_repro.resolve()),sha256=det.sha(args.det_repro))
    except Exception as exc:
        lanes={name:dict(status='BLOCKED',reason='required PT_DET_1 repro gate: '+str(exc))
               for name,_,_ in sequence(out,args.keep_grads,args.include_model,args.det_repro)}
        t.create_json(out/'SLOT_SUMMARY.json',dict(verdict='BLOCKED',lanes=lanes,gpu_executed=False))
        print('SLOT BLOCKED: required PT_DET_1 repro gate:',str(exc));return 2
    os.environ['PT_DET_1_CERTIFICATION']='1'
    t.verify_manifest()
    temp=out/'tmp';temp.mkdir()
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PT_TF32_LEAD_GPU='1',PT_TF32_GENERATION='4',TC_TF32_GEMM='0',
             TMPDIR=str(temp),CUDA_CACHE_PATH=str(out/'cuda_cache'),OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
    deadline=start+1200;receipts={}
    for name,budget,argv in sequence(out,args.keep_grads,args.include_model,args.det_repro):
        remaining=deadline-time.monotonic()-5
        space=storage.space_status(out)
        if name=='pt_det_1_repro':row=dict(repro_lane,argv=argv)
        elif space['status']!='GREEN':row=dict(space,argv=argv)
        elif remaining<=0:
            row=dict(status='BLOCKED',reason='global 1200-second deadline reached',argv=argv)
        elif name in ('healthy','control') and receipts.get('noise_floor',{}).get('status')!='GREEN':
            row=dict(status='BLOCKED',reason='noise-floor acquisition did not pass',argv=argv)
        else:
            print('LANE',name,'BUDGET',min(budget,remaining),flush=True)
            try:
                row=run_lane(argv,out/(name+'.log'),env,min(budget,remaining))
                if row['status'] in ('GREEN','RED') and name in ('memcheck','racecheck','synccheck'):
                    row['unit_test_returncode']=row['returncode']
                    row.update(sanitizer_result(name,Path(row['log']).read_text()))
                summary=out/name/'summary.json'
                if summary.exists() and json.loads(summary.read_text()).get('verdict')=='BLOCKED':
                    row.update(status='BLOCKED',reason='lane reports blocked prerequisite')
            except FileNotFoundError as exc:row=dict(status='BLOCKED',reason='required executable missing',error=repr(exc),argv=argv)
            except Exception as exc:row=dict(status='RED',error=repr(exc),argv=argv)
        receipts[name]=row;t.create_json(out/(name+'_receipt.json'),row)
        print('RESULT',name,row['status'],flush=True)
    ok=all(r['status']=='GREEN' for r in receipts.values())
    summary=dict(verdict='GREEN' if ok else 'RED_OR_BLOCKED',lanes=receipts,
        elapsed_seconds=time.monotonic()-start,registration_sha256=t.sha(t.REG),
        manifest_sha256=t.sha(t.ART/'SOURCE_MANIFEST.json'),
        keep_grads=args.keep_grads,include_model=args.include_model,total_dump_bytes=storage.dump_bytes(out),
        evidence_class='single sequential lead slot; blind review remains separate')
    summary['implementation_manifest_sha256']=det.sha(det.ART/det.MANIFEST_NAME)
    summary['required_repro']=repro_lane
    t.create_json(out/'SLOT_SUMMARY.json',summary)
    print('SLOT',summary['verdict'],'SECONDS',summary['elapsed_seconds']);return 0 if ok else 1


if __name__=='__main__':sys.exit(main())
