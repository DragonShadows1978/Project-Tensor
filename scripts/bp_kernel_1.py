#!/usr/bin/env python3
"""BP-KERNEL-1: one immutable, bounded micro-census; no GPU on --dry-run.

Prior art: Dao et al., FlashAttention (2022), softmax VJP/recomputation and
output-dot identity; NVIDIA CUDA events/atomicAdd (year unverified); NumPy PCG64
(2019; O'Neill PCG 2014) seeded fixtures; IEEE round-to-nearest-even for BF16.
All external attributions unverified — lead to check those names/search terms.
Taken: existing algebra, events, RNG and rounding. Ours: native APA experiment
plumbing and deletion ablation. c is NOT a valid key-parallel backward.
"""
from __future__ import annotations
import argparse
import ctypes
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/bp_kernel_1'
REG = ART / 'registration.json'
VARIANTS = 'abcd'
SHAPE = dict(B=1, H=16, KVH=16, L=2048, S=2048, D=96, VD=64,
             dtype='bfloat16', causal=True, refine=0.15,
             scale=1.0 / math.sqrt(96), zthr=1.0364333894937898)
TINY = dict(SHAPE, H=2, KVH=1, L=5, S=5, D=4, VD=3, scale=0.5)
PREDICTION = 'time(a) − time(c) ≥ 50 % of time(a) — the atomics own the kernel.'
FALSIFIER = '< 50 % promotes the scalar dot loops (tiling / tensor cores, variant e) as the first target.'
TOLERANCE = ('For each of dQ/dK/dV, max_abs and relative_L2(candidate,a1) must '
             'each be <= 2 * the same metric(a2,a1). No additional absolute or '
             'relative tolerance. a1/a2 and their spreads must be finite; '
             'zero spread gives zero tolerance. relative_L2 = ||x-ref||_2 / '
             '||ref||_2; zero/zero=0, nonzero/zero is nonfinite and RED.')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def create_json(path, obj):
    # Exclusive creation is the registration/receipt immutability primitive.
    data = json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n'
    with Path(path).open('x') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def bf16(x):
    # IEEE ties-to-even, represented losslessly as float32 on the CPU.
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    return ((u + np.uint32(0x7fff) + ((u >> 16) & 1)) &
            np.uint32(0xffff0000)).view(np.float32)


def seed_inputs(shape, seed=20260913):
    rng = np.random.default_rng(seed)
    b,h,kh,l,s,d,vd = [shape[k] for k in ('B','H','KVH','L','S','D','VD')]
    q = bf16(rng.standard_normal((b,h,l,d), dtype=np.float32))
    k = bf16(rng.standard_normal((b,kh,s,d), dtype=np.float32))
    # Synthetic correlated detached Kq; not a checkpoint activation capture or
    # a claim to reproduce GRAPA's 4-bit quantizer. No prior art known to me for
    # this exact fixture choice; registered once, never selected using timings.
    kq = bf16(k + np.float32(.125)*rng.standard_normal(k.shape, dtype=np.float32))
    v = bf16(rng.standard_normal((b,kh,s,vd), dtype=np.float32))
    do = bf16(rng.standard_normal((b,h,l,vd), dtype=np.float32))
    return dict(q=q,k=k,kq=kq,v=v,dO=do)


def save_npz(path, arrays):
    with Path(path).open('xb') as f: np.savez(f, **arrays)


def metric(x, ref):
    x, ref = np.asarray(x, np.float64), np.asarray(ref, np.float64)
    if x.shape != ref.shape or not np.isfinite(x).all() or not np.isfinite(ref).all():
        return dict(finite=False, max_abs=None, relative_L2=None)
    delta = x-ref
    den, num = float(np.linalg.norm(ref.ravel())), float(np.linalg.norm(delta.ravel()))
    rel = num/den if den else (0.0 if num == 0 else math.inf)
    ma = float(np.max(np.abs(delta))) if delta.size else 0.0
    finite = math.isfinite(ma) and math.isfinite(rel)
    return dict(finite=finite, max_abs=ma if math.isfinite(ma) else None,
                relative_L2=rel if math.isfinite(rel) else None)


def gate(a1, a2, candidates):
    if len(a1)!=3 or len(a2)!=3 or any(len(x)!=3 for x in candidates.values()):
        raise ValueError('expected exactly dQ/dK/dV')
    spreads = {k:metric(y,x) for k,x,y in zip(('dQ','dK','dV'),a1,a2)}
    finite = all(m['finite'] for m in spreads.values())
    limits = {k:{m:2*v[m] if v['finite'] else None for m in ('max_abs','relative_L2')}
              for k,v in spreads.items()}
    result = {}
    for name, values in candidates.items():
        errors = {k:metric(y,x) for k,x,y in zip(('dQ','dK','dV'),a1,values)}
        green = finite and all(v['finite'] and all(v[m] <= limits[k][m]
            for m in ('max_abs','relative_L2')) for k,v in errors.items())
        result[name] = dict(verdict='GREEN' if green else 'RED', errors=errors)
    return dict(control_finite=finite, spreads=spreads, tolerance=limits, candidates=result)


def decide(timings, correctness, complete=True):
    if not complete or not correctness.get('control_finite'):
        return dict(primary='INCONCLUSIVE', atomic_fraction=None, secondary='INCONCLUSIVE')
    if any(len(timings.get(v,[])) != 10 or
           not all(math.isfinite(t) and t > 0 for t in timings[v]) for v in VARIANTS):
        return dict(primary='INCONCLUSIVE', atomic_fraction=None, secondary='INCONCLUSIVE')
    means = {v:sum(timings[v])/10 for v in VARIANTS}
    fraction = (means['a']-means['c'])/means['a']
    primary = 'ATOMICS_GE_50' if fraction >= .5 else 'SCALAR_DOTS_PROMOTED'
    secondary = 'NOT_APPLICABLE'
    if fraction >= .5 and correctness['candidates']['d']['verdict'] == 'GREEN':
        secondary = 'CONFIRMED' if means['d'] <= .6*means['a'] else 'FALSIFIED'
    return dict(primary=primary, atomic_fraction=fraction, secondary=secondary)


def verify_registration(reg_path=REG, root=ROOT):
    reg_path = Path(reg_path)
    expected_hash = reg_path.with_suffix('.sha256').read_text().strip()
    if sha(reg_path) != expected_hash: raise RuntimeError('registration bytes drift')
    reg = json.loads(reg_path.read_text())
    for rel, pin in reg['pins'].items():
        if sha(root/rel) != pin: raise RuntimeError(f'registration drift: {rel}')
    if reg['shape'] != SHAPE or reg['tolerance_rule'] != TOLERANCE or \
       reg['prediction'] != PREDICTION or reg['falsifier'] != FALSIFIER or \
       reg['warmups'] != 3 or reg['samples'] != 10 or \
       reg['budget'] != dict(work_seconds=300, lease_seconds=590, cells=1, lock='/tmp/forge-gpu.lock'):
        raise RuntimeError('registration protocol drift')
    for name, pin in reg['sources'].items():
        if sha(root/'tensor_cuda/src'/name) != pin['after_sha256']:
            raise RuntimeError(f'source drift: {name}')
        if sha(root/'artifacts/bp_kernel_1/baseline'/name) != pin['before_sha256']:
            raise RuntimeError(f'baseline drift: {name}')
    return reg


class CPUBackend:
    """Tiny scalar NumPy simulator. Tests orchestration, never CUDA parity."""
    def __init__(self, arrays, shape): self.x, self.s = arrays, shape

    def forward(self):
        x,s = self.x,self.s
        out = np.zeros_like(x['dO']); lse = np.zeros(x['q'].shape[:-1],np.float32); thr=lse.copy()
        self.rows = []
        for b in range(s['B']):
            for h in range(s['H']):
                kh=h//(s['H']//s['KVH'])
                for i in range(s['L']):
                    n=s['S']-s['L']+i+1 if s['causal'] else s['S']
                    q=x['q'][b,h,i]; k=x['k'][b,kh,:n]; kq=x['kq'][b,kh,:n]; v=x['v'][b,kh,:n]
                    bulk=(kq@q)*s['scale']; ab=np.abs(bulk)
                    threshold=ab.mean()+s['zthr']*np.sqrt(max(0.,float((ab*ab).mean()-ab.mean()**2)))
                    sel=ab>=threshold; score=np.where(sel,k@q*s['scale'],bulk)
                    p=np.exp(score-score.max()); z=p.sum(); p=p/z
                    out[b,h,i]=bf16(p@v); lse[b,h,i]=score.max()+np.log(z); thr[b,h,i]=threshold
                    self.rows.append((b,h,kh,i,n,sel,p))
        self.out=out
        return dict(out=out,lse=lse,thr=thr)

    def backward(self, variant):
        x,s=self.x,self.s
        dq=np.zeros_like(x['q']);dk=np.zeros_like(x['k']);dv=np.zeros_like(x['v'])
        for b,h,kh,i,n,sel,p in self.rows:
            do=x['dO'][b,h,i];q=x['q'][b,h,i];v=x['v'][b,kh,:n]
            dov=v@do
            rowdot=float(do@self.out[b,h,i]) if variant=='d' else float(p@dov)
            ds=p*(dov-rowdot)*s['scale']
            keys=np.where(sel[:,None],x['k'][b,kh,:n],x['kq'][b,kh,:n])
            dq[b,h,i]=bf16(ds@keys)
            if variant=='c': continue
            for j in range(n):
                av=p[j]*do; ak=ds[j]*q if sel[j] else np.zeros_like(q)
                if variant=='a':
                    dv[b,kh,j]=bf16(dv[b,kh,j]+bf16(av)); dk[b,kh,j]=bf16(dk[b,kh,j]+bf16(ak))
                else:
                    dv[b,kh,j]+=av; dk[b,kh,j]+=ak
        return dq,bf16(dk),bf16(dv)

    def host(self, values): return values
    def sync(self): pass
    def timed(self, variant):
        start=time.perf_counter(); self.backward(variant)
        return (time.perf_counter()-start)*1000


class CUDABackend:
    def __init__(self, arrays, shape, reg):
        # Load the pinned extension by absolute filename, not a global install.
        binary=ROOT/reg['engine_binary']['path']
        if sha(binary)!=reg['engine_binary']['sha256']: raise RuntimeError('engine binary drift')
        spec=importlib.util.spec_from_file_location('_tensor_cuda',binary)
        self.c=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.c)
        self.s=shape
        self.x={k:self.c.tensor(np.ascontiguousarray(v),'cuda',False).astype('bfloat16') for k,v in arrays.items()}
        self.rt=ctypes.CDLL('/usr/local/cuda-12.6/lib64/libcudart.so')
        for name,args in [('cudaEventCreate',[ctypes.POINTER(ctypes.c_void_p)]),
                          ('cudaEventRecord',[ctypes.c_void_p,ctypes.c_void_p]),
                          ('cudaEventSynchronize',[ctypes.c_void_p]),
                          ('cudaEventElapsedTime',[ctypes.POINTER(ctypes.c_float),ctypes.c_void_p,ctypes.c_void_p]),
                          ('cudaEventDestroy',[ctypes.c_void_p])]:
            f=getattr(self.rt,name);f.argtypes=args;f.restype=ctypes.c_int

    def check(self, name, *args):
        rc=getattr(self.rt,name)(*args)
        if rc: raise RuntimeError(f'{name}: CUDA rc={rc}')

    def forward(self):
        self.out,self.lse,self.thr=self.c.apa_selective_fwd_train(
            *[self.x[k] for k in ('q','k','kq','v')],self.s['scale'],self.s['zthr'],self.s['causal'])
        self.sync()
        return {k:v.numpy() for k,v in [('out',self.out),('lse',self.lse),('thr',self.thr)]}

    def backward(self, variant):
        args=[self.x[k] for k in ('q','k','kq','v','dO')]+[self.lse,self.thr]
        if variant=='a': return self.c.apa_selective_bwd(*args,self.s['scale'],self.s['causal'])
        return self.c.apa_selective_bwd_variant(*args,self.out,self.s['scale'],self.s['causal'],variant)

    def host(self, values):
        self.sync()
        return tuple(v.numpy() for v in values)

    def sync(self): self.c.synchronize()

    def timed(self, variant):
        # Prior art: NVIDIA CUDA event elapsed time, default stream. Taken:
        # event bracketing; ours: interleaved bare-op census including zero/cast.
        start,end=ctypes.c_void_p(),ctypes.c_void_p()
        self.check('cudaEventCreate',ctypes.byref(start))
        try:
            self.check('cudaEventCreate',ctypes.byref(end))
            try:
                self.sync()
                self.check('cudaEventRecord',start,None)
                result=self.backward(variant)
                self.check('cudaEventRecord',end,None)
                self.check('cudaEventSynchronize',end)
                self.sync()
                elapsed=ctypes.c_float()
                self.check('cudaEventElapsedTime',ctypes.byref(elapsed),start,end)
                # Keep outputs alive through the end event; no host copy in timing.
                del result
                return float(elapsed.value)
            finally: self.check('cudaEventDestroy',end)
        finally: self.check('cudaEventDestroy',start)


def empty_receipt(mode, shape, registration_hash):
    return dict(schema_version=1, experiment='BP-KERNEL-1', mode=mode,
        evidence_class='CPU simulation' if mode=='dry-run' else 'kernel micro-benchmark',
        registration_sha256=registration_hash, shape=shape, verdict='INCONCLUSIVE',
        correctness=None, timings={v:dict(samples_ms=[],mean_ms=None,min_ms=None,
        timing_counts=False,valid_gradient=v!='c') for v in VARIANTS},
        launch_order=[], warmups=3, samples=10, input_sha256=None,
        forward_state=None, engine_binary=None, result=None, elapsed_seconds=None,
        budget=dict(work_seconds=300,lease_seconds=590,cells=1), error=None,
        residuals=['Synthetic seeded inputs, not checkpoint activations.',
                   'c is timing only; no gradient or training-quality claim.',
                   'd uses saved BF16 output, so output rounding is gated.',
                   'CPU dry-run does not establish CUDA correctness or performance.'])


def experiment(backend, receipt, output_dir, deadline):
    started=time.monotonic()
    def check_time():
        if time.monotonic() >= deadline: raise TimeoutError('300 s work budget exhausted')
    check_time()
    state=backend.forward()
    if not all(np.isfinite(x).all() for x in state.values()): raise RuntimeError('nonfinite forward state')
    state_path=output_dir/'forward_state.npz';save_npz(state_path,state)
    manifest=dict(schema_version=1, registration_sha256=receipt['registration_sha256'],
        input_sha256=receipt['input_sha256'],sha256=sha(state_path),path=str(state_path),
        provenance='CPU simulation' if receipt['mode']=='dry-run' else 'real apa_selective_fwd_train CUDA kernel',
        arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in state.items()},
        phase='created before any backward correctness gate or timing')
    create_json(output_dir/'forward_state_registration.json',manifest)
    receipt['forward_state']=manifest
    check_time()
    a1=backend.host(backend.backward('a'));check_time()
    a2=backend.host(backend.backward('a'));check_time()
    candidates={}
    for v in 'bd':
        candidates[v]=backend.host(backend.backward(v));check_time()
    receipt['correctness']=gate(a1,a2,candidates)
    if not receipt['correctness']['control_finite']: raise RuntimeError('control run-to-run spread is not finite')
    # Correctness ALWAYS precedes warmups/timing. RED candidate timings are
    # retained as ineligible raw observations, never counted as gradient speed.
    for phase, count in [('warmup',3),('measured',10)]:
        for i in range(count):
            for v in VARIANTS:
                check_time()
                if phase=='warmup': backend.backward(v);backend.sync()
                else: receipt['timings'][v]['samples_ms'].append(backend.timed(v))
                receipt['launch_order'].append(dict(phase=phase,round=i,variant=v))
                check_time()
    raw={v:t['samples_ms'] for v,t in receipt['timings'].items()}
    result=decide(raw,receipt['correctness'])
    receipt['result']=result
    for v,t in receipt['timings'].items():
        t['mean_ms']=sum(t['samples_ms'])/10;t['min_ms']=min(t['samples_ms'])
        t['timing_counts']=receipt['mode']=='run' and result['primary']!='INCONCLUSIVE' and (v in 'ac' or receipt['correctness']['candidates'][v]['verdict']=='GREEN')
    receipt['verdict']='DRY_RUN' if receipt['mode']=='dry-run' else result['primary']
    receipt['elapsed_seconds']=time.monotonic()-started


def validate_receipt(r):
    required=set(empty_receipt('dry-run',TINY,'x'))
    if set(r)!=required: raise ValueError('receipt schema keys differ')
    if r['mode'] not in ('run','dry-run') or r['schema_version']!=1: raise ValueError('receipt schema value')
    if r['verdict'] not in ('INCONCLUSIVE','DRY_RUN','ATOMICS_GE_50','SCALAR_DOTS_PROMOTED'): raise ValueError('receipt verdict')
    if r['mode']=='dry-run' and any(t['timing_counts'] for t in r['timings'].values()): raise ValueError('CPU timing counted')
    if r['verdict'] not in ('INCONCLUSIVE',):
        if len(r['launch_order'])!=52 or any(len(t['samples_ms'])!=10 for t in r['timings'].values()): raise ValueError('incomplete receipt')
    json.dumps(r,allow_nan=False)


def worker(output_dir, reg, reg_hash):
    r=empty_receipt('run',SHAPE,reg_hash); started=time.monotonic()
    r['input_sha256']=sha(ART/'inputs.npz');r['engine_binary']=reg['engine_binary']
    try:
        with np.load(ART/'inputs.npz',allow_pickle=False) as f: arrays={k:f[k] for k in f.files}
        backend=CUDABackend(arrays,SHAPE,reg)
        experiment(backend,r,output_dir,started+300)
    except Exception as e:
        r['error']=f'{type(e).__name__}: {e}';r['verdict']='INCONCLUSIVE'
    r['elapsed_seconds']=time.monotonic()-started
    validate_receipt(r);create_json(output_dir/'receipt.json',r)
    print(f"{r['verdict']} receipt={output_dir/'receipt.json'}",flush=True)
    return 0 if r['verdict']!='INCONCLUSIVE' else 2


def run_cell(reg, reg_hash):
    # Prior art: local GraftRepository grm_cmc1_gpu_arms.py gpu_lease (2026),
    # flock exclusion + bounded foreground work. Ours: a synchronous child with
    # subprocess timeout, so a stuck native CUDA call cannot defeat SIGALRM.
    # Only this newly started child may be terminated on timeout. No retries.
    lock=open('/tmp/forge-gpu.lock','a+')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        started=time.monotonic()
        if (ART/'receipt.json').exists(): raise FileExistsError('receipt already exists')
        create_json(ART/'cell_claim.json',dict(registration_sha256=reg_hash,
            pid=os.getpid(),start_unix=time.time(),work_seconds=300,lease_seconds=590))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1')
        cmd=[sys.executable,str(Path(__file__).resolve()),'--worker']
        try:
            child=subprocess.run(cmd,env=env,timeout=300,check=False,pass_fds=(lock.fileno(),))
            rc=child.returncode
        except subprocess.TimeoutExpired:
            rc=124
        if not (ART/'receipt.json').exists():
            r=empty_receipt('run',SHAPE,reg_hash)
            r['input_sha256']=sha(ART/'inputs.npz');r['engine_binary']=reg['engine_binary']
            r['elapsed_seconds']=time.monotonic()-started
            r['error']=f'foreground cell incomplete, child rc={rc}; no retries permitted'
            validate_receipt(r);create_json(ART/'receipt.json',r)
        print(f"cell_rc={rc} receipt={ART/'receipt.json'}",flush=True)
        return rc
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true')
    p.add_argument('--dry-run',action='store_true',help='takes precedence over --run; CPU only')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--output-dir',type=Path,help='dry-run only, new directory')
    args=p.parse_args(argv)
    reg=verify_registration();reg_hash=sha(REG)
    if args.dry_run:
        out=args.output_dir or ART/('dry_run_'+uuid.uuid4().hex)
        out.mkdir(parents=True,exist_ok=False)
        r=empty_receipt('dry-run',TINY,reg_hash);arrays=seed_inputs(TINY)
        save_npz(out/'inputs.npz',arrays);r['input_sha256']=sha(out/'inputs.npz')
        experiment(CPUBackend(arrays,TINY),r,out,time.monotonic()+300)
        validate_receipt(r);create_json(out/'receipt.json',r)
        print(f"DRY_RUN receipt={out/'receipt.json'}")
        return 0
    if args.output_dir: p.error('--output-dir is dry-run only')
    if args.worker:
        # Internal workers need the immutable parent's claim AND inherited lock
        # inode. Direct --worker cannot open a second unleased GPU lane.
        claim=json.loads((ART/'cell_claim.json').read_text())
        if claim['pid']!=os.getppid() or claim['registration_sha256']!=reg_hash:
            raise RuntimeError('worker has no matching parent claim')
        return worker(ART,reg,reg_hash)
    if args.run: return run_cell(reg,reg_hash)
    p.error('choose --dry-run or --run')

if __name__=='__main__':
    try: sys.exit(main())
    except Exception as e:
        print(f'FAIL_CLOSED {type(e).__name__}: {e}',file=sys.stderr)
        sys.exit(2)
