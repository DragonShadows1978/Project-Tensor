#!/usr/bin/env python3
"""BP-KERNEL-2: frozen-state FP64 reference, registered gate and bounded cell.

Prior art: Dao et al. FlashAttention (2022) softmax VJP/output-dot; Dao (2023)
FlashAttention-2 key ownership; NumPy (Harris et al. 2020) dense algebra;
NVIDIA CUDA events (2007+), POSIX flock, SHA-256 (NIST 2001), BP-KERNEL-1
(2026) bounded child/fixtures. Taken: these methods. Ours: APA integration and
this order's reference-distance gate. Unverified — lead to check those names.
"""
from __future__ import annotations
import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
import sys
sys.dont_write_bytecode = True
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
import uuid
import numpy as np
import bp_kernel_1 as old

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/'artifacts/bp_kernel_2'
REG = ART/'registration.json'
INPUT = ROOT/'artifacts/bp_kernel_1/inputs.npz'
STATE = ROOT/'artifacts/bp_kernel_1/forward_state.npz'
REFERENCE = ART/'reference.npz'
SHAPE = old.SHAPE
NAMES = ('dQ','dK','dV')
VARIANTS = ('a','b','f','d')
PREDICTION = 'f ≤ 0.35 × a (≤ ~144 ms at a ≈ 411) with dQ, dK, dV all inside the reference tolerance.'
FALSIFIER = 'f green but > 0.35 × a → the scalar dot loops are the next target (tiling / tensor cores); f RED → fix before any timing.'
STEP_PREDICTION = 'whole step with f ≤ 5.0 s (a-arm reproduces ≈ 11.8 s within 10 %).'
STEP_FALSIFIER = '> 5.0 s → the census component table says what remains.'
TOLERANCE = 'For each of dQ/dK/dV and each metric (max-abs, relative-L2), tolerance = 2 × |a − reference|; fresh GPU a; zero distance means zero tolerance, no epsilon.'
BUDGET = dict(cells=2,work_seconds=300,lease_seconds=590,lock='/tmp/forge-gpu.lock',cards=1)
sha, create_json, save_npz, metric = old.sha, old.create_json, old.save_npz, old.metric


def arrays(path):
    with np.load(path,allow_pickle=False) as f: return {k:f[k] for k in f.files}


def reference(x, state, shape, *, output_dot=False):
    """FP64 shipped VJP with frozen lse/thr; never renormalize p.

    Prior art: softmax reverse mode / FlashAttention (Dao et al. 2022), taken;
    NumPy dense products (Harris et al. 2020), taken. Ours: materialized APA
    stop-gradient mask and bounded row chunks. Unverified — lead to check.
    """
    x={k:np.asarray(v,np.float64) for k,v in x.items()}
    b,h,kh,l,s,d,vd=[shape[k] for k in ('B','H','KVH','L','S','D','VD')]
    # The C++ API takes float scale; promote that actual parameter to FP64.
    scale=float(np.float32(shape['scale']))
    dq=np.zeros_like(x['q']); dk=np.zeros_like(x['k']); dv=np.zeros_like(x['v'])
    selection=np.zeros((b,h,l,s),dtype=bool); rowdot=np.zeros((b,h,l),np.float64)
    for bi in range(b):
        for hi in range(h):
            ki=hi//(h//kh); k=x['k'][bi,ki]; kq=x['kq'][bi,ki]; v=x['v'][bi,ki]
            for lo in range(0,l,128):
                end=min(l,lo+128); q=x['q'][bi,hi,lo:end]; do=x['dO'][bi,hi,lo:end]
                bulk=(q@kq.T)*scale
                visible=np.broadcast_to(np.arange(s)[None,:] < (s-l+np.arange(lo,end)[:,None]+1),bulk.shape) if shape['causal'] else np.ones(bulk.shape,bool)
                sel=(np.abs(bulk)>=state['thr'][bi,hi,lo:end,None]) & visible
                selection[bi,hi,lo:end]=sel
                score=np.where(sel,(q@k.T)*scale,bulk)
                p=np.exp(np.where(visible,score-state['lse'][bi,hi,lo:end,None],-np.inf))
                dov=do@v.T
                rd=np.sum(do*state['out'][bi,hi,lo:end],axis=1) if output_dot else np.sum(p*dov,axis=1)
                rowdot[bi,hi,lo:end]=rd
                ds=p*(dov-rd[:,None])*scale
                selected=np.where(sel,ds,0.)
                dq[bi,hi,lo:end]=selected@k+np.where(sel,0.,ds)@kq
                dk[bi,ki]+=selected.T@q
                dv[bi,ki]+=p.T@do
    result=dict(dQ=dq,dK=dk,dV=dv,rowdot=rowdot,selection=selection)
    if not all(np.isfinite(v).all() for v in result.values()): raise ValueError('nonfinite reference')
    return result


def build_reference():
    parent=json.loads((ROOT/'artifacts/bp_kernel_1/registration.json').read_text())
    forward=json.loads((ROOT/'artifacts/bp_kernel_1/forward_state_registration.json').read_text())
    if sha(INPUT)!=parent['input']['sha256'] or sha(STATE)!=forward['sha256']:
        raise ValueError('parent input/state drift')
    create_json(ART/'reference_registration.json',dict(input_sha256=sha(INPUT),state_sha256=sha(STATE),
        source_sha256=sha(__file__),shape=SHAPE,scale='float32 API scalar promoted to FP64',
        math='FP64 bulk selection >= saved thr, exact selected scores; saved lse, no renormalization; bottom-right causal mask',
        phase='before CPU reference construction; immutable'))
    started=time.monotonic()
    ref=reference(arrays(INPUT),arrays(STATE),SHAPE)
    save_npz(REFERENCE,ref)
    create_json(ART/'reference_receipt.json',dict(evidence_class='FP64 CPU reference, not GPU validation',
        elapsed_seconds=time.monotonic()-started,sha256=sha(REFERENCE),
        arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in ref.items()},
        selected=int(ref['selection'].sum())))
    print(f'{REFERENCE} sha256={sha(REFERENCE)}')


def gate(a, ref, candidates):
    # Order-registered 2x reference distance. No prior art known to me for this
    # exact selection rule; it is the user's rule, not a novel error estimator.
    if len(a)!=3 or len(ref)!=3 or any(len(v)!=3 for v in candidates.values()):
        raise ValueError('expected dQ/dK/dV')
    for v in [a,*candidates.values()]:
        if any(np.shape(x)!=np.shape(y) for x,y in zip(v,ref)): raise ValueError('gradient shape mismatch')
    distance={k:metric(x,y) for k,x,y in zip(NAMES,a,ref)}
    finite=all(v['finite'] for v in distance.values())
    tol={k:{m:2*v[m] if v['finite'] else None for m in ('max_abs','relative_L2')} for k,v in distance.items()}
    out={}
    for name,values in candidates.items():
        errors={k:metric(x,y) for k,x,y in zip(NAMES,values,ref)}
        passes={k:finite and v['finite'] and all(v[m]<=tol[k][m] for m in tol[k]) for k,v in errors.items()}
        out[name]=dict(verdict='GREEN' if all(passes.values()) else 'RED',errors=errors,passes=passes)
    return dict(a_distance_from_reference=distance,tolerance=tol,control_finite=finite,candidates=out,
                zero_tolerances=[f'{k}.{m}' for k in tol for m,v in tol[k].items() if v==0])


def verify_registration():
    if sha(REG)!=REG.with_suffix('.sha256').read_text().strip(): raise ValueError('registration drift')
    r=json.loads(REG.read_text())
    for k,v in dict(shape=SHAPE,prediction=PREDICTION,falsifier=FALSIFIER,step_prediction=STEP_PREDICTION,
                    step_falsifier=STEP_FALSIFIER,tolerance_rule=TOLERANCE,budget=BUDGET,warmups=3,samples=10).items():
        if r[k]!=v: raise ValueError('protocol drift: '+k)
    for p,h in r['pins'].items():
        if sha(ROOT/p)!=h: raise ValueError('pin drift: '+p)
    for p,entry in r['sources'].items():
        if sha(ROOT/p)!=entry['after_sha256'] or sha(ROOT/entry['baseline'])!=entry['before_sha256']:
            raise ValueError('source drift: '+p)
    pin=r['variant_a_region']; data=(ROOT/'tensor_cuda/src/kernels.cu').read_bytes()
    if hashlib.sha256(data[pin['start_byte']:pin['end_byte_exclusive']]).hexdigest()!=pin['sha256']:
        raise ValueError('variant a drift')
    return r


class GPU(old.CUDABackend):
    def __init__(self,x,state,r):
        super().__init__(x,SHAPE,r)
        self.out=self.c.tensor(np.ascontiguousarray(state['out']),'cuda',False).astype('bfloat16')
        self.lse=self.c.tensor(np.ascontiguousarray(state['lse']),'cuda',False)
        self.thr=self.c.tensor(np.ascontiguousarray(state['thr']),'cuda',False)
        self.route='f'
    def backward(self,v): return super().backward(self.route if v=='f' else v)


class CPU(old.CPUBackend):
    def __init__(self,x,s):
        super().__init__(x,s); self.state=self.forward(); self.route='f'
    def backward(self,v):
        if v in ('f','f_pass_a'):
            ref=reference(self.x,self.state,self.s,output_dot=(self.route=='f' and v=='f'))
            return tuple(old.bf16(ref[k]) for k in NAMES)
        return super().backward(v)


def empty_receipt(dry):
    return dict(schema_version=1,experiment='BP-KERNEL-2',mode='dry-run' if dry else 'run',
        evidence_class='CPU plumbing only' if dry else 'kernel micro-benchmark',
        registration_sha256=sha(REG),input_sha256=sha(INPUT),forward_state_sha256=sha(STATE),
        reference_sha256=sha(REFERENCE),shape=old.TINY if dry else SHAPE,correctness=None,
        rowdot_route=None,rowdot_probe=None,timings={},launch_order=[],verdict='INCONCLUSIVE',error=None,
        elapsed_seconds=None,budget=BUDGET)


def experiment(backend,ref,r,emit):
    a=backend.host(backend.backward('a'))
    candidates={v:backend.host(backend.backward(v)) for v in ('b','d','f')}
    initial=gate(a,ref,candidates)
    r['rowdot_probe']=initial['candidates']['f']
    if not initial['control_finite']: raise ValueError('nonfinite control reference distance')
    if not initial['candidates']['f']['passes']['dQ']:
        backend.route='f_pass_a'
        candidates['f']=backend.host(backend.backward('f'))
    r['rowdot_route']=backend.route
    r['correctness']=gate(a,ref,candidates)
    emit(dict(kind='gate',correctness=r['correctness'],rowdot_route=r['rowdot_route'],rowdot_probe=r['rowdot_probe']))
    # RED f: no f micro-timing, per registered falsifier. Other variants keep
    # exactly 3+10 rounds; no changed shapes/samples. Cell 2 still runs both arms.
    for v in VARIANTS:
        ok=v=='a' or r['correctness']['candidates'][v]['verdict']=='GREEN'
        r['timings'][v]=dict(samples_ms=[],mean_ms=None,min_ms=None,timing_counts=False,gate_green=ok)
    for phase,count in [('warmup',3),('measured',10)]:
        for i in range(count):
            for v in VARIANTS:
                if v=='f' and not r['timings']['f']['gate_green']: continue
                if phase=='warmup': backend.backward(v);backend.sync()
                else:
                    ms=backend.timed(v)
                    if not math.isfinite(ms) or ms<=0: raise ValueError('invalid event timing')
                    r['timings'][v]['samples_ms'].append(ms)
                r['launch_order'].append(dict(phase=phase,round=i,variant=v))
    for v,t in r['timings'].items():
        if len(t['samples_ms'])==10:
            t['mean_ms']=sum(t['samples_ms'])/10; t['min_ms']=min(t['samples_ms'])
            t['timing_counts']=r['mode']=='run' and t['gate_green']
    if r['mode']=='dry-run': r['verdict']='DRY_RUN'
    elif not r['timings']['f']['gate_green']: r['verdict']='F_RED_FIX_BEFORE_TIMING'
    else: r['verdict']='CONFIRMED' if r['timings']['f']['mean_ms']<=.35*r['timings']['a']['mean_ms'] else 'SCALAR_DOTS_NEXT'


def validate_receipt(r):
    if r['schema_version']!=1 or r['experiment']!='BP-KERNEL-2' or r['mode'] not in ('run','dry-run'):
        raise ValueError('receipt schema')
    if r['verdict'] not in ('INCONCLUSIVE','DRY_RUN','F_RED_FIX_BEFORE_TIMING','CONFIRMED','SCALAR_DOTS_NEXT'):
        raise ValueError('receipt verdict')
    if r['verdict']!='INCONCLUSIVE':
        if set(r['timings'])!=set(VARIANTS) or not r['correctness']: raise ValueError('incomplete receipt')
        if r['rowdot_route'] not in ('f','f_pass_a'): raise ValueError('missing rowdot')
        for v,t in r['timings'].items():
            n=0 if v=='f' and not t['gate_green'] else 10
            if len(t['samples_ms'])!=n: raise ValueError('incomplete samples')
            if t['timing_counts'] and (r['mode']!='run' or not t['gate_green']): raise ValueError('ineligible timing')
        if r['mode']=='dry-run' and r['verdict']!='DRY_RUN': raise ValueError('CPU native verdict')
    json.dumps(r,allow_nan=False)
    return r


def gpu_preflight():
    p=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=5)
    if len(p.stdout.strip().splitlines())!=1: raise ValueError('single physical card required')
    busy=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=5)
    if busy.stdout.strip(): raise ValueError('existing GPU compute process; refusing overlap')
    return p.stdout.strip()


def supervise(script,art,reg_hash,summarize):
    # Prior art: BP-KERNEL-1 / POSIX flock + Python subprocess timeout, taken.
    # Foreground child only; parent holds lock through child termination.
    with open(BUDGET['lock'],'a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (art/'receipt.json').exists(): raise FileExistsError(art/'receipt.json')
        create_json(art/'cell_claim.json',dict(registration_sha256=reg_hash,started=time.time(),budget=BUDGET))
        started=time.monotonic(); error=None; events=[]
        try:
            child=subprocess.run([sys.executable,str(script),'--worker'],capture_output=True,text=True,timeout=300,
                pass_fds=(lock.fileno(),),env=dict(os.environ,BP_KERNEL_2_WORKER='1',CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1'))
            stdout,stderr=child.stdout,child.stderr
            if child.returncode: error=f'worker exit {child.returncode}: {stderr[-8000:]}'
        except subprocess.TimeoutExpired as e:
            # subprocess.run terminates only its own child and waits for it.
            stdout=e.stdout or b''; stderr=e.stderr or b''; error='300 s work cap exceeded; INCONCLUSIVE; samples not shortened'
        if isinstance(stdout,bytes): stdout=stdout.decode(errors='replace')
        if isinstance(stderr,bytes): stderr=stderr.decode(errors='replace')
        for line in stdout.splitlines():
            try: events.append(json.loads(line))
            except json.JSONDecodeError: error=(error or '')+' invalid worker JSON'
        create_json(art/'worker_output.json',dict(events=events,stderr=stderr,error=error))
        r=summarize(events,error,time.monotonic()-started)
        create_json(art/'receipt.json',r)
    print(f'{r["verdict"]} {art/"receipt.json"}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--run',action='store_true');p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.reference:
        if args.run or args.dry_run or args.worker: raise ValueError('reference is separate')
        build_reference();return
    reg=verify_registration()
    if args.dry_run:
        started=time.monotonic(); backend=CPU(old.seed_inputs(old.TINY),old.TINY)
        ref=reference(backend.x,backend.state,old.TINY); r=empty_receipt(True)
        experiment(backend,tuple(ref[k] for k in NAMES),r,lambda e:None)
        r['elapsed_seconds']=time.monotonic()-started
        path=ART/('dry_kernel_'+uuid.uuid4().hex)/'receipt.json';path.parent.mkdir()
        create_json(path,validate_receipt(r)); print(f'DRY_RUN {path}');return
    if args.worker:
        if os.environ.get('BP_KERNEL_2_WORKER')!='1': raise ValueError('supervisor required')
        def emit(e): print(json.dumps(e,allow_nan=False),flush=True)
        emit(dict(kind='gpu',description=gpu_preflight()))
        backend=GPU(arrays(INPUT),arrays(STATE),reg); ref=arrays(REFERENCE); r=empty_receipt(False)
        started=time.monotonic();experiment(backend,tuple(ref[k] for k in NAMES),r,emit)
        r['elapsed_seconds']=time.monotonic()-started;emit(dict(kind='result',receipt=validate_receipt(r)));return
    if not args.run: raise ValueError('choose --reference, --dry-run or --run')
    def summarize(events,error,elapsed):
        done=[e['receipt'] for e in events if e.get('kind')=='result']
        r=done[-1] if done else empty_receipt(False)
        if error or not done:
            r['verdict']='INCONCLUSIVE';r['error']=error or 'missing worker result'
            for e in events:
                if e.get('kind')=='gate':
                    for k in ('correctness','rowdot_route','rowdot_probe'): r[k]=e[k]
            for t in r['timings'].values(): t['timing_counts']=False
        r['elapsed_seconds']=elapsed
        return validate_receipt(r)
    supervise(Path(__file__).resolve(),ART,sha(REG),summarize)

if __name__=='__main__': main()
