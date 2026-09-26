#!/usr/bin/env python3
"""Lead-owned sequential PT-DET-1 slot, <=1500 seconds; never opens a lock.

Prior art: POSIX inherited flock and process groups; PT-TF32-4 (2026) global
deadline/blocked receipts, taken. Ours: mandatory embedding+four-arm replay.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import pt_det_1 as t


def sequence(out, lock_fd):
    py = [sys.executable, '-B', 'scripts/pt_det_1.py']
    common = ['--lead-gpu', '--lock-fd', str(lock_fd)]
    return [('embedding', 80, py + ['embedding', '--out', str(out/'embedding')] + common),
            ('pt_det_1_repro', 1400, py + ['repro', '--steps', '30', '--out', str(out/'repro')] + common)]


def main():
    start = time.monotonic()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--lead-gpu', action='store_true')
    p.add_argument('--lock-fd', type=int)
    p.add_argument('--print-sequence', action='store_true')
    args = p.parse_args(); out = args.out.resolve()
    t.registration()
    if not out.is_relative_to(t.ART) or out == t.ART: p.error('out must be a fresh PT-DET-1 artifact subdirectory')
    if args.print_sequence:
        print(json.dumps(dict(lanes=sequence(out,args.lock_fd),deadline_seconds=1500,
                              reserve_seconds=20,lock='inherited exclusive descriptor required'),indent=2)); return 0
    out.mkdir(parents=True,exist_ok=False)
    try:
        lock=t.require_lead(args.lead_gpu,args.lock_fd);m=t.verify_manifest()
    except Exception as e:
        result=dict(verdict='BLOCKED',reason=str(e),gpu_executed=False,
                    lanes={name:dict(status='BLOCKED',reason=str(e)) for name,_,_ in sequence(out,args.lock_fd)})
        t.create_json(out/'SLOT_SUMMARY.json',result)
        print('PT_DET_1 SLOT BLOCKED',str(e));return 2
    (out/'tmp').mkdir()
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(out/'tmp'),
             CUDA_CACHE_PATH=str(out/'cuda_cache'),OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
    deadline=start+1500;rows={}
    for name,budget,argv in sequence(out,args.lock_fd):
        seconds=min(budget,deadline-time.monotonic()-20)
        if seconds<=0:
            row=dict(status='BLOCKED_TIMEOUT',reason='global deadline exhausted')
        else:
            print('PT_DET_1 LANE',name,'budget_seconds',round(seconds,1),flush=True)
            try:
                row=t.run_process(argv,t.ROOT,env,out/(name+'.log'),seconds,pass_fds=(args.lock_fd,))
                summary=out/('repro' if name=='pt_det_1_repro' else name)/'summary.json'
                if summary.exists():
                    result=json.loads(summary.read_text());row['gate_verdict']=result.get('verdict')
                    if result.get('verdict')!='GREEN':row['status']=result.get('verdict','RED')
                    row['summary']=str(summary);row['summary_sha256']=t.sha(summary)
                elif row['status']=='GREEN':row.update(status='RED',reason='missing lane summary')
            except Exception as e:row=dict(status='RED',error=repr(e))
        rows[name]=row;t.create_json(out/(name+'_receipt.json'),row)
        print('PT_DET_1 RESULT',name,row['status'],flush=True)
    ok=all(row['status']=='GREEN' for row in rows.values())
    result=dict(verdict='GREEN' if ok else 'RED_OR_BLOCKED',lanes=rows,lock=lock,
                seconds=time.monotonic()-start,binary=m['binary'],registration_sha256=t.REG_SHA)
    t.create_json(out/'SLOT_SUMMARY.json',result)
    print('PT_DET_1 SLOT',result['verdict'],'seconds',round(result['seconds'],2))
    return 0 if ok else 1


if __name__=='__main__':sys.exit(main())
