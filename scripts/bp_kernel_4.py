#!/usr/bin/env python3
"""BP-KERNEL-4 CPU preparation and lead-only forward microbenchmark.
Prior art: FlashAttention/FA-2 (Dao 2022/2023), NumPy (Harris 2020),
CUDA events (NVIDIA 2007+), SHA256 (NIST 2001), BP-KERNEL-1/2/3 harness
(2026), taken. Ours: APA statistics reference and this order's gates.
Unverified — lead to check those titles/authors; no network in this seat.
"""
from __future__ import annotations
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import sys
sys.dont_write_bytecode=True
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import uuid
import numpy as np
import bp_kernel_2 as parent

ROOT=parent.ROOT
ART=ROOT/'artifacts/bp_kernel_4'
REG=ART/'registration.json'
REFERENCE=ART/'reference.npz'
INPUT,STATE,BACK_REFERENCE=parent.INPUT,parent.STATE,parent.REFERENCE
SHAPE=parent.SHAPE
BUDGET=parent.BUDGET
sha,create_json,arrays,supervise,gpu_preflight=parent.sha,parent.create_json,parent.arrays,parent.supervise,parent.gpu_preflight
PREDICTION='h ≤ 0.20 × a-forward with all gates green.'
FALSIFIER='> 0.20 × → report which of tiling / MMA / threshold derivation is the residual (per-phase timing inside the kernel or per-half diagnostics REQUIRED in the receipt).'
STEP_PREDICTION='initial forward + checkpoint replay ≤ 0.25 s combined (from 0.90) and whole step ≤ 1.3 s.'
STEP_FALSIFIER='the census component table says what remains (if the APA forward turns out to be less than half of the forward, say so with the numbers — that is a legitimate outcome, not a failure of the seat).'
PHASES=('shared_load','bulk_mma','threshold_statistics','masked_exact','online_softmax','pv_mma_and_rescale','output_store')


def protocol():
    return dict(schema_version=1,experiment='BP-KERNEL-4',shape=SHAPE,budget=BUDGET,
        prediction=PREDICTION,falsifier=FALSIFIER,step_prediction=STEP_PREDICTION,step_falsifier=STEP_FALSIFIER,
        variants=['a','h'],h2_built=False,warmups=3,samples=10,interleaved_order=['a','h'],
        tile=dict(query=16,key=16,width=128,threads=128,wmma=[16,16,16],inputs='BF16',accumulator='FP32'),
        tolerance_rule='For O and lse, each max_abs and relative_L2 tolerance = 2 × |a-forward − FP64 reference|; zero means zero, no epsilon.',
        downstream_tolerance_rule=parent.TOLERANCE,flip_ceiling=.005,
        flip_denominator='all B*H*L*S pairs, masked pairs unselected; also report visible-pair rate',
        threshold_rule='mean(abs(bulk)) + float32(zthr)*sqrt(max(mean(abs(bulk)^2)-mean(abs(bulk))^2,0)); visible keys only, abs(bulk)>=thr',
        downstream_rule='unchanged g1 on h O/lse/thr versus pinned BP-KERNEL-2 FP64 reference; tolerances from fresh a-backward on pinned state',
        diagnostic_protocol=dict(warmups=3,samples=10,phases=list(PHASES),
            scope='Separate h diagnostic specialization; clock64 CTA elapsed cycles, sum/mean/max per phase, not GPU wall milliseconds. Native selection mask replay for shipped a.'),
        red_rule='Forward or downstream RED: stop before timing; no timing counts. Cell 2 requires completed cell 1; RED cell 2 timing-only.',
        budget_enforcement='Two create-only cell claims; exclusive nonblocking flock, foreground child timeout 300 s; lease <=590 s; incomplete => INCONCLUSIVE.')


def verify_registration(require_reference=True):
    if sha(REG)!=REG.with_suffix('.sha256').read_text().strip():raise ValueError('registration drift')
    r=json.loads(REG.read_text())
    for key,value in protocol().items():
        if r.get(key)!=value:raise ValueError('protocol drift: '+key)
    for p,h in r['pins'].items():
        if sha(ROOT/p)!=h:raise ValueError('pin drift: '+p)
    data=(ROOT/'tensor_cuda/src/kernels.cu').read_bytes()
    for name,p in r['regions'].items():
        if hashlib.sha256(data[p['start_byte']:p['end_byte_exclusive']]).hexdigest()!=p['sha256']:raise ValueError('region drift: '+name)
    # Build/reference completion are separate create-only receipts; the immutable
    # preregistration pins source, recipe and inputs before either is executed.
    # Prior art: content-addressed manifests (SHA256 NIST 2001), taken.
    build=arrays_manifest(ART/'engine_build_receipt.json')
    if build['rc']!=0 or build['source_pins']!={p:h for p,h in r['pins'].items() if p.startswith('tensor_cuda/')}:
        raise ValueError('build source drift')
    binary=build['binaries'][0]
    if sha(ROOT/binary['path'])!=binary['sha256']:raise ValueError('binary drift')
    if require_reference:
        rr=arrays_manifest(ART/'reference_receipt.json')
        if rr['registration_sha256']!=sha(REG) or rr['sha256']!=sha(REFERENCE):raise ValueError('reference drift')
    return dict(r,engine_binary=binary,effective_sha256=sha(REG))


def arrays_manifest(p):return json.loads(p.read_text())


def reference(x,s,chunk=128):
    """FP64 exact forward math; float API scalars promoted to FP64.
    Prior art: standard stable softmax and FlashAttention (Dao 2022), taken;
    APA shipped absolute-score z-score rule taken, bounded dense chunks ours.
    The supplied refine percentile already maps to zthr upstream, NOT quantile.
    """
    x={k:np.asarray(v,np.float64) for k,v in x.items()}
    B,H,KVH,L,S,D,VD=[s[n] for n in ('B','H','KVH','L','S','D','VD')]
    if min(B,H,KVH,L,S,D,VD)<=0 or H%KVH or S<L:raise ValueError('geometry')
    out=np.empty((B,H,L,VD));lse=np.empty((B,H,L));thr=np.empty_like(lse)
    selection=np.zeros((B,H,L,S),bool)
    scale,zthr=float(np.float32(s['scale'])),float(np.float32(s['zthr']))
    for b in range(B):
        for h in range(H):
            kh=h//(H//KVH)
            for lo in range(0,L,chunk):
                hi=min(L,lo+chunk);q=x['q'][b,h,lo:hi]
                bulk=(q@x['kq'][b,kh].T)*scale
                visible=(np.arange(S)[None,:]<S-L+np.arange(lo,hi)[:,None]+1) if s['causal'] else np.ones(bulk.shape,bool)
                count=visible.sum(1);ab=np.where(visible,np.abs(bulk),0.)
                mean=ab.sum(1)/count
                threshold=mean+zthr*np.sqrt(np.maximum((ab*ab).sum(1)/count-mean*mean,0.))
                sel=visible & (np.abs(bulk)>=threshold[:,None])
                score=np.where(visible,np.where(sel,(q@x['k'][b,kh].T)*scale,bulk),-np.inf)
                m=score.max(1);p=np.exp(score-m[:,None]);den=p.sum(1)
                out[b,h,lo:hi]=(p@x['v'][b,kh])/den[:,None]
                lse[b,h,lo:hi]=m+np.log(den);thr[b,h,lo:hi]=threshold;selection[b,h,lo:hi]=sel
    result=dict(out=out,lse=lse,thr=thr,selection=selection)
    if not all(np.isfinite(v).all() for v in result.values()):raise ValueError('nonfinite reference')
    return result


def build_reference():
    verify_registration(False)
    start=time.monotonic();r=reference(arrays(INPUT),SHAPE)
    parent.save_npz(REFERENCE,r)
    create_json(ART/'reference_receipt.json',dict(evidence_class='FP64 CPU reference, not GPU parity',
        registration_sha256=sha(REG),sha256=sha(REFERENCE),elapsed_seconds=time.monotonic()-start,
        arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in r.items()},selected=int(r['selection'].sum())))
    print(f'FP64_REFERENCE {REFERENCE} sha256={sha(REFERENCE)}')


def flips(a,h,visible=None):
    # User-registered selection flip fraction; no prior art known to me for
    # this exact 0.5 percent gate. Nonfinite masks fail closed, never cast NaN.
    a,h=np.asarray(a),np.asarray(h)
    if a.shape!=h.shape or a.size==0 or a.dtype!=bool or h.dtype!=bool:raise ValueError('boolean selection shape')
    diff=a!=h
    r=dict(flips=int(diff.sum()),pairs=a.size,rate=float(diff.mean()),ceiling=.005,passes=float(diff.mean())<=.005)
    if visible is not None:
        visible=np.broadcast_to(visible,a.shape)
        if np.any((a|h)&~visible):raise ValueError('masked selection')
        r['visible_pairs']=int(visible.sum());r['visible_rate']=float(diff.sum()/visible.sum())
    return r


def forward_gate(a,h,ref,visible=None):
    errors={};distance={};tol={};passes={}
    for n in ('out','lse'):
        if np.shape(a[n])!=np.shape(ref[n]) or np.shape(h[n])!=np.shape(ref[n]):raise ValueError('forward shape')
        distance[n]=parent.metric(a[n],ref[n]);errors[n]=parent.metric(h[n],ref[n])
        tol[n]={m:2*distance[n][m] if distance[n]['finite'] else None for m in ('max_abs','relative_L2')}
        passes[n]=distance[n]['finite'] and errors[n]['finite'] and all(errors[n][m]<=tol[n][m] for m in tol[n])
    for obj in (a,h,ref):
        if np.shape(obj['thr'])!=np.shape(ref['thr']):raise ValueError('threshold shape')
    th=parent.metric(h['thr'],a['thr']);flip=flips(a['selection'],h['selection'],visible)
    finite=all(np.isfinite(obj['thr']).all() for obj in (a,h,ref))
    return dict(verdict='GREEN' if all(passes.values()) and flip['passes'] and finite else 'RED',
        a_distance=distance,errors=errors,tolerance=tol,passes=passes,threshold_h_vs_a=th,
        threshold_a_vs_reference=parent.metric(a['thr'],ref['thr']),threshold_h_vs_reference=parent.metric(h['thr'],ref['thr']),selection=flip)


class GPU(parent.GPU):
    def forward_op(self,v,diagnostic=False):
        fn=self.c.bp_kernel_4_diagnostic if diagnostic else self.c.apa_selective_fwd_train_variant
        return fn(*[self.x[k] for k in ('q','k','kq','v')],self.s['scale'],self.s['zthr'],self.s['causal'],v)
    def snapshot(self,v):
        data=self.forward_op(v,True);host=self.host(data)
        return dict(zip(('out','lse','thr','selection','cycles'),(*host[:3],host[3].astype(bool),host[4])))
    def downstream(self,v):
        self.out,self.lse,self.thr=self.forward_op(v)
        return self.host(self.backward('g1'))
    def timed(self,v):
        # Reuse the audited CUDA-event lifecycle, swapping only the callable.
        old=self.backward
        try:
            self.backward=self.forward_op
            return parent.old.CUDABackend.timed(self,v)
        finally:self.backward=old
    def phases(self):
        return self.host(self.forward_op('h',True))[4]


class CPU:
    """Plumbing adapter only: FP64 reference rounded to saved-state formats.
    Prior art: BP-KERNEL-1 CPU fake, taken. No simulation of CUDA rounding.
    """
    def __init__(self):
        self.s=dict(parent.old.TINY);self.x=parent.old.seed_inputs(self.s)
        self.ref=reference(self.x,self.s)
    def snapshot(self,v):
        return dict(out=parent.old.bf16(self.ref['out']),lse=self.ref['lse'].astype('float32'),
            thr=self.ref['thr'].astype('float32'),selection=self.ref['selection'].copy())
    def downstream(self,v):return tuple(self.backref[n].astype('float32') for n in parent.NAMES)
    def forward_op(self,v):return self.snapshot(v)
    def sync(self):pass
    def timed(self,v):
        t=time.perf_counter();self.forward_op(v);return max((time.perf_counter()-t)*1000,1e-9)
    def phases(self):return np.ones((1,1,1,7))


def empty_receipt(dry):
    return dict(schema_version=1,experiment='BP-KERNEL-4',mode='dry-run' if dry else 'run',
        evidence_class='CPU plumbing only' if dry else 'kernel micro-benchmark',
        registration_sha256=sha(REG),reference_sha256=sha(REFERENCE),back_reference_sha256=sha(BACK_REFERENCE),
        correctness=None,timings={},launch_order=[],phase_diagnostics=None,verdict='INCONCLUSIVE',error=None,elapsed_seconds=None)


def experiment(b,r,emit):
    dry=r['mode']=='dry-run'
    ref=b.ref if dry else arrays(REFERENCE)
    if dry:
        b.backref=parent.reference(b.x,b.snapshot('a'),b.s)
        backref=tuple(b.backref[n] for n in parent.NAMES)
        control=tuple(v.astype('float32') for v in backref)
    else:
        backref=tuple(arrays(BACK_REFERENCE)[n] for n in parent.NAMES)
        # Backend starts with the pinned state; BP-KERNEL-2 baseline unchanged.
        control=b.host(b.backward('a'))
    a,h=b.snapshot('a'),b.snapshot('h')
    visible=(np.arange(b.s['S'])[None,:]<b.s['S']-b.s['L']+np.arange(b.s['L'])[:,None]+1) if b.s['causal'] else np.ones((b.s['L'],b.s['S']),bool)
    fg=forward_gate(a,h,ref,visible)
    # Gate the normal specialization as well as diagnostic selection replay.
    normal={}
    for v,snap in [('a',a),('h',h)]:
        raw=b.forward_op(v)
        host=raw if dry else dict(zip(('out','lse','thr'),b.host(raw)))
        normal[v]=all(np.array_equal(host[n],snap[n]) for n in ('out','lse','thr'))
    bg=parent.gate(control,backref,dict(h=b.downstream('h')))
    ok=fg['verdict']=='GREEN' and bg['candidates']['h']['verdict']=='GREEN' and all(normal.values())
    r['correctness']=dict(forward=fg,downstream=bg,diagnostic_matches_normal=normal,green=ok)
    emit(dict(kind='gate',correctness=r['correctness']))
    if not ok:
        r['verdict']='DRY_RUN' if dry else 'RED';return
    for v in ('a','h'):r['timings'][v]=dict(samples_ms=[],mean_ms=None,timing_counts=False)
    for phase,count in [('warmup',3),('measured',10)]:
        for i in range(count):
            for v in ('a','h'):
                if phase=='warmup':b.forward_op(v);b.sync()
                else:r['timings'][v]['samples_ms'].append(b.timed(v))
                r['launch_order'].append(dict(phase=phase,round=i,variant=v))
    for t in r['timings'].values():
        t['mean_ms']=sum(t['samples_ms'])/10;t['timing_counts']=not dry
    phases=[]
    for i in range(13):
        p=b.phases()
        if i>=3:phases.append(p)
    data=np.stack(phases).reshape(-1,7)
    r['phase_diagnostics']=dict(evidence_class='CPU fabricated plumbing' if dry else 'clock64 diagnostic specialization',
        samples=10,warmups=3,unit='CTA elapsed cycles',phase={n:dict(sum=float(data[:,i].sum()),mean=float(data[:,i].mean()),max=float(data[:,i].max())) for i,n in enumerate(PHASES)},
        largest_phase=PHASES[int(data.sum(0).argmax())],
        limitation='Phase cycles include synchronization and scheduling; summed CTA cycles are not wall time. shared_load measures tiling, bulk_mma and pv_mma_and_rescale measure MMA plus stores/rescale, threshold_statistics measures threshold. No abstract causal attribution beyond these phase boundaries.')
    r['verdict']='DRY_RUN' if dry else ('CONFIRMED' if r['timings']['h']['mean_ms']<=.2*r['timings']['a']['mean_ms'] else 'PREDICTION_FALSIFIED')


def validate_receipt(r):
    if set(r)!=set(empty_receipt(r.get('mode')=='dry-run')) or r['schema_version']!=1 or r['experiment']!='BP-KERNEL-4':raise ValueError('schema')
    if r['mode'] not in ('run','dry-run') or r['verdict'] not in ('INCONCLUSIVE','DRY_RUN','RED','CONFIRMED','PREDICTION_FALSIFIED'):raise ValueError('verdict')
    if r['mode']=='dry-run' and r['verdict'] not in ('DRY_RUN','INCONCLUSIVE'):raise ValueError('CPU claim')
    counted=any(t['timing_counts'] for t in r['timings'].values())
    if counted and (r['mode']!='run' or r['verdict'] in ('INCONCLUSIVE','RED') or not r['correctness']['green']):raise ValueError('ineligible timing')
    if r['verdict'] in ('CONFIRMED','PREDICTION_FALSIFIED') or (r['verdict']=='DRY_RUN' and r['correctness']['green']):
        expected=[dict(phase=p,round=i,variant=v) for p,n in [('warmup',3),('measured',10)] for i in range(n) for v in ('a','h')]
        if r['launch_order']!=expected or set(r['timings'])!={'a','h'} or not r['phase_diagnostics']:raise ValueError('incomplete')
        for t in r['timings'].values():
            if len(t['samples_ms'])!=10 or any(not math.isfinite(v) or v<=0 for v in t['samples_ms']) or t['mean_ms']!=sum(t['samples_ms'])/10:raise ValueError('samples')
        if r['mode']=='run':
            expected='CONFIRMED' if r['timings']['h']['mean_ms']<=.2*r['timings']['a']['mean_ms'] else 'PREDICTION_FALSIFIED'
            if r['verdict']!=expected or not r['correctness']['green']:raise ValueError('decision')
    if r['verdict']=='RED' and (not r['correctness'] or r['correctness']['green'] or r['timings']):raise ValueError('RED schema')
    json.dumps(r,allow_nan=False);return r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('reference','dry-run','run','worker'):p.add_argument('--'+flag,action='store_true')
    args=p.parse_args()
    if args.reference:
        if args.run or args.dry_run or args.worker:raise ValueError('reference separate')
        build_reference();return
    reg=verify_registration()
    if args.dry_run:
        start=time.monotonic();r=empty_receipt(True);experiment(CPU(),r,lambda e:None);r['elapsed_seconds']=time.monotonic()-start
        path=ART/('dry_kernel_'+uuid.uuid4().hex)/'receipt.json';path.parent.mkdir()
        create_json(path,validate_receipt(r));print(f'DRY_RUN {path}');return
    if args.worker:
        if os.environ.get('BP_KERNEL_2_WORKER')!='1':raise ValueError('supervisor required')
        def emit(e):print(json.dumps(e,allow_nan=False),flush=True)
        emit(dict(kind='gpu',description=gpu_preflight()));b=GPU(arrays(INPUT),arrays(STATE),reg)
        start=time.monotonic();r=empty_receipt(False);experiment(b,r,emit);r['elapsed_seconds']=time.monotonic()-start
        emit(dict(kind='result',receipt=validate_receipt(r)));return
    if not args.run:raise ValueError('choose --reference / --dry-run / --run')
    def summarize(events,error,elapsed):
        done=[e['receipt'] for e in events if e.get('kind')=='result'];r=done[-1] if done else empty_receipt(False)
        if error or not done:
            r['verdict']='INCONCLUSIVE';r['error']=error or 'missing worker result'
            for e in events:
                if e.get('kind')=='gate':r['correctness']=e['correctness']
            for t in r['timings'].values():t['timing_counts']=False
        r['elapsed_seconds']=elapsed;return validate_receipt(r)
    supervise(Path(__file__).resolve(),ART,sha(REG),summarize)

if __name__=='__main__':main()
