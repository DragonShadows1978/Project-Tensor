#!/usr/bin/env python3
"""BP-KERNEL-3: immutable CPU preparation and two lead-only bounded GPU cells.

Prior art: FlashAttention-2 (Dao 2023) tiled ownership/recomputation; CUDA WMMA
(NVIDIA 2017, BF16 2020) and events (2007+); NumPy (Harris et al. 2020);
BP-KERNEL-2 (2026) reference, gate and foreground supervision, all taken.
Ours: APA selective tile integration and this registered comparison.
Unverified — lead to check those names; this seat has no network.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid
import numpy as np
import bp_kernel_2 as parent

ROOT=parent.ROOT
ART=ROOT/'artifacts/bp_kernel_3'
REG=ART/'registration.json'
INPUT,STATE,REFERENCE=parent.INPUT,parent.STATE,parent.REFERENCE
SHAPE=parent.SHAPE
NAMES=parent.NAMES
VARIANTS=('a','f','g1','g2')
G=('g1','g2')
TILE=16
PREDICTION='best green g ≤ 0.35 × f (≤ ~46 ms at f ≈ 131).'
FALSIFIER='> 0.35 × f → report which of tiling / MMA / selection-gather is the residual (per-kernel timing of the query-owned and key-owned halves is REQUIRED in the receipt).'
SECONDARY='g1 ≤ g2 (selection still pays on tensor cores); if g2 < g1, say so plainly.'
STEP_PREDICTION='whole step ≤ 2.5 s with g (f-arm reproduces ≈ 5.0 s within 10 %).'
STEP_FALSIFIER='> 2.5 s → report the census component table; f outside 10 % → reproduction failed.'
BUDGET=parent.BUDGET
TOLERANCE=parent.TOLERANCE
sha,create_json,arrays,gate,reference=parent.sha,parent.create_json,parent.arrays,parent.gate,parent.reference
supervise,gpu_preflight=parent.supervise,parent.gpu_preflight


def protocol():
    return dict(schema_version=1,experiment='BP-KERNEL-3',shape=SHAPE,variants=list(VARIANTS),
        prediction=PREDICTION,falsifier=FALSIFIER,secondary_prediction=SECONDARY,
        step_prediction=STEP_PREDICTION,step_falsifier=STEP_FALSIFIER,budget=BUDGET,
        tolerance_rule=TOLERANCE,warmups=3,samples=10,interleaved_order=list(VARIANTS),
        tile=dict(query=16,key=16,padded_width=128,threads=128,wmma=[16,16,16],
                  input='BF16',accumulate='FP32',max_D=128,max_VD=128),
        half_protocol=dict(variants=['f','g1','g2'],warmups=3,samples=10,
            scope='Separate diagnostic interleaved launches; timers disabled for whole-op timings.',
            names=['query_owned','key_owned']),
        rowdot_route='output-dot, fixed; no fallback',g2_built=True,
        red_rule='Collect all timing samples; RED timings do not count. No green g => cell 2 f vs g1 TIMING-ONLY.',
        residual_rule='Report both halves and g1/g2 comparison; halves alone cannot causally separate tiling from MMA. No attribution beyond these measurements.',
        budget_enforcement='Create-only claims for two cells; inherited foreground 300 s child timeout under exclusive nonblocking flock; <=590 s lease; incomplete => INCONCLUSIVE.')


def registration_chain():
    # Prior art: SHA-256 hash chaining / append-only logs (NIST 2001 hash,
    # Haber and Stornetta 1991 timestamp chains), taken. Ours: source-only
    # amendment records; protocol/predictions cannot be amended here.
    # Unverified — lead to check "Haber Stornetta 1991 hash chain".
    base=sha(REG)
    if base!=REG.with_suffix('.sha256').read_text().strip(): raise ValueError('registration drift')
    r=json.loads(REG.read_text());previous=base
    amendments=sorted(REG.parent.glob('registration_amendment_*.json'))
    for number,path in enumerate(amendments,1):
        a=json.loads(path.read_text())
        if path.name!=f'registration_amendment_{number:03d}.json' or sha(path)!=path.with_suffix('.sha256').read_text().strip(): raise ValueError('amendment drift')
        if set(a)!={'number','previous_sha256','reason','pins','sources','engine_binary'} or a['number']!=number or a['previous_sha256']!=previous: raise ValueError('amendment chain')
        r['pins'].update(a['pins']);r['sources'].update(a['sources'])
        if a['engine_binary'] is not None:r['engine_binary']=a['engine_binary']
        previous=sha(path)
    return r,previous


def verify_registration():
    if sha(REG)!=REG.with_suffix('.sha256').read_text().strip(): raise ValueError('registration drift')
    r,effective_sha=registration_chain()
    for key,value in protocol().items():
        if r.get(key)!=value: raise ValueError('protocol drift: '+key)
    for p,h in r['pins'].items():
        if sha(ROOT/p)!=h: raise ValueError('pin drift: '+p)
    for p,entry in r['sources'].items():
        if sha(ROOT/p)!=entry['after_sha256'] or sha(ROOT/entry['baseline'])!=entry['before_sha256']:
            raise ValueError('source drift: '+p)
    pin=r['variant_a_region'];s=(ROOT/'tensor_cuda/src/kernels.cu').read_bytes()
    if hashlib.sha256(s[pin['start_byte']:pin['end_byte_exclusive']]).hexdigest()!=pin['sha256']:
        raise ValueError('variant a drift')
    if sha(ROOT/r['engine_binary']['path'])!=r['engine_binary']['sha256']: raise ValueError('binary drift')
    return dict(r,effective_sha256=effective_sha)


def tiled_backward(x,state,s,variant,*,round_mma=False):
    """CPU model of BOTH owner traversals (not CUDA execution or validation).

    Prior art: Dao 2023 owner/recompute tiles, NumPy dense algebra (2020),
    taken. Ours: mixed-score coefficients and mirrored padded boundary rules.
    With round_mma=False FP64 algebra checks traversal independently of BF16.
    Unverified — lead to check FlashAttention-2; BF16 via BP-KERNEL-1 helper.
    """
    if variant not in G: raise ValueError('g1/g2 required')
    x={k:np.asarray(v,np.float64) for k,v in x.items()}
    result={n:np.zeros_like(x[k]) for n,k in zip(NAMES,('q','k','v'))}
    selection=np.zeros((s['B'],s['H'],s['L'],s['S']),bool)
    half=[];scale=float(np.float32(s['scale']));group=s['H']//s['KVH']
    def rnd(v): return parent.old.bf16(v).astype(np.float64) if round_mma else v
    def pair(b,h,kh,qi,kj):
        # Explicit zero padding agrees with native fragment load extents.
        ni=min(TILE,s['L']-qi);nj=min(TILE,s['S']-kj)
        q=x['q'][b,h,qi:qi+ni];do=x['dO'][b,h,qi:qi+ni]
        k=x['k'][b,kh,kj:kj+nj];kq=x['kq'][b,kh,kj:kj+nj];v=x['v'][b,kh,kj:kj+nj]
        bulk=q@kq.T*scale
        visible=np.ones((ni,nj),bool) if not s['causal'] else kj+np.arange(nj)[None,:]<=s['S']-s['L']+qi+np.arange(ni)[:,None]
        sel=(np.abs(bulk)>=state['thr'][b,h,qi:qi+ni,None])&visible
        exact=q@k.T if variant=='g2' else np.zeros_like(bulk)
        if variant=='g1':
            for i,j in zip(*np.nonzero(sel)): exact[i,j]=np.dot(q[i],k[j])
        p=np.exp(np.where(visible,np.where(sel,exact*scale,bulk)-state['lse'][b,h,qi:qi+ni,None],-np.inf))
        rd=np.sum(do*state['out'][b,h,qi:qi+ni],axis=1)
        ds=p*(do@v.T-rd[:,None])*scale
        selection[b,h,qi:qi+ni,kj:kj+nj]=sel
        a=np.zeros((TILE,TILE));u=a.copy();pr=a.copy()
        a[:ni,:nj]=rnd(np.where(sel,ds,0));u[:ni,:nj]=rnd(np.where(sel,0,ds));pr[:ni,:nj]=rnd(p)
        def pad(y): return np.pad(y,((0,TILE-len(y)),(0,128-y.shape[1])))
        return a,u,pr,pad(q),pad(do),pad(k),pad(kq),ni,nj
    for key in (False,True):
        start=time.perf_counter()
        for b in range(s['B']):
            for ownerh in range(s['KVH'] if key else s['H']):
                kh=ownerh if key else ownerh//group
                for owner in range(0,s['S'] if key else s['L'],TILE):
                    acc=np.zeros((TILE,128));av=acc.copy()
                    for gh in range(group if key else 1):
                        h=kh*group+gh if key else ownerh
                        first=max(0,owner-(s['S']-s['L']))//TILE*TILE if key and s['causal'] else 0
                        limit=s['L'] if key else (min(s['S'],s['S']-s['L']+owner+TILE) if s['causal'] else s['S'])
                        for other in range(first,limit,TILE):
                            qi,kj=(other,owner) if key else (owner,other)
                            a,u,p,q,do,k,kq,ni,nj=pair(b,h,kh,qi,kj)
                            if key: acc+=a.T@q;av+=p.T@do
                            else: acc+=a@k+u@kq
                    n=min(TILE,(s['S'] if key else s['L'])-owner)
                    if key:
                        result['dK'][b,kh,owner:owner+n]=acc[:n,:s['D']]
                        result['dV'][b,kh,owner:owner+n]=av[:n,:s['VD']]
                    else: result['dQ'][b,ownerh,owner:owner+n]=acc[:n,:s['D']]
        half.append((time.perf_counter()-start)*1000)
    result['selection']=selection
    return result,half


class CPU(parent.CPU):
    def backward(self,v):
        if v in G:
            result,self.halves=tiled_backward(self.x,self.state,self.s,v,round_mma=True)
            return tuple(parent.old.bf16(result[n]) for n in NAMES)
        return super().backward(v)
    def half_timed(self,v):
        if v=='f':
            _,halves=tiled_backward(self.x,self.state,self.s,'g1')
            return halves  # CPU plumbing only; no f-kernel timing claim.
        self.backward(v);return self.halves


class GPU(parent.GPU):
    def half_timed(self,v):
        self.c.bp_kernel_3_profile(True)
        try:
            result=self.backward(v);self.sync()
            return list(self.c.bp_kernel_3_halves())
        finally: self.c.bp_kernel_3_profile(False)


def empty_receipt(dry):
    return dict(schema_version=1,experiment='BP-KERNEL-3',mode='dry-run' if dry else 'run',
        evidence_class='CPU plumbing only' if dry else 'kernel micro-benchmark',
        registration_sha256=sha(REG),registration_chain_sha256=registration_chain()[1],input_sha256=sha(INPUT),forward_state_sha256=sha(STATE),
        reference_sha256=sha(REFERENCE),shape=parent.old.TINY if dry else SHAPE,
        correctness=None,timings={},half_timings={},launch_order=[],half_launch_order=[],
        best_green=None,secondary=None,residual=None,verdict='INCONCLUSIVE',error=None,
        elapsed_seconds=None,budget=BUDGET)


def decision(correctness,timings,dry):
    # The user's registered rule, no prior art known to me for this exact rule.
    green=[v for v in G if correctness['candidates'][v]['verdict']=='GREEN']
    best=min(green,key=lambda v:timings[v]['mean_ms']) if green else None
    secondary=None
    if len(green)==2:
        secondary='g2 < g1: selection does not pay in this implementation' if timings['g2']['mean_ms']<timings['g1']['mean_ms'] else 'g1 ≤ g2: selection pays in this implementation'
    if dry: verdict='DRY_RUN'
    elif not correctness['control_finite']: verdict='INCONCLUSIVE'
    elif correctness['candidates']['f']['verdict']!='GREEN': verdict='F_RED'
    elif best is None: verdict='G_RED'
    else: verdict='CONFIRMED' if timings[best]['mean_ms']<=.35*timings['f']['mean_ms'] else 'PREDICTION_FALSIFIED'
    return best,secondary,verdict


def experiment(backend,ref,r,emit):
    a=backend.host(backend.backward('a'))
    candidates={v:backend.host(backend.backward(v)) for v in VARIANTS[1:]}
    r['correctness']=gate(a,ref,candidates)
    emit(dict(kind='gate',correctness=r['correctness']))
    for v in VARIANTS:
        ok=r['correctness']['control_finite'] if v=='a' else r['correctness']['candidates'][v]['verdict']=='GREEN'
        r['timings'][v]=dict(samples_ms=[],mean_ms=None,gate_green=ok,timing_counts=False)
    for phase,count in [('warmup',3),('measured',10)]:
        for i in range(count):
            for v in VARIANTS:
                if phase=='warmup': backend.backward(v);backend.sync()
                else:
                    ms=backend.timed(v)
                    if not math.isfinite(ms) or ms<=0: raise ValueError('invalid CUDA timing')
                    r['timings'][v]['samples_ms'].append(ms)
                r['launch_order'].append(dict(phase=phase,round=i,variant=v))
    for v,t in r['timings'].items():
        t['mean_ms']=sum(t['samples_ms'])/10
        t['timing_counts']=r['mode']=='run' and t['gate_green']
    for v in ('f',*G):
        r['half_timings'][v]={h:dict(samples_ms=[],mean_ms=None,timing_counts=False) for h in ('query_owned','key_owned')}
    for phase,count in [('warmup',3),('measured',10)]:
        for i in range(count):
            for v in ('f',*G):
                halves=backend.half_timed(v)
                if len(halves)!=2 or any(not math.isfinite(x) or x<=0 for x in halves): raise ValueError('invalid half timings')
                if phase=='measured':
                    for h,x in zip(('query_owned','key_owned'),halves): r['half_timings'][v][h]['samples_ms'].append(x)
                r['half_launch_order'].append(dict(phase=phase,round=i,variant=v))
    for v,halves in r['half_timings'].items():
        for t in halves.values():
            t['mean_ms']=sum(t['samples_ms'])/10;t['timing_counts']=r['timings'][v]['timing_counts']
    r['best_green'],r['secondary'],r['verdict']=decision(r['correctness'],r['timings'],r['mode']=='dry-run')
    r['residual']=dict(evidence_class='reasoning from diagnostic half timings',
        larger_half={v:max(h,key=lambda n:h[n]['mean_ms']) for v,h in r['half_timings'].items()},
        selection_comparison=r['secondary'],
        limitation='g1/g2 isolates masked scalar exact versus dense exact MMA, not an abstract selection benefit. Half timings cannot distinguish shared-memory tiling cost from MMA cost; no causal attribution claimed.')


def validate_receipt(r):
    required=set(empty_receipt(r.get('mode')=='dry-run'))
    if set(r)!=required or r['schema_version']!=1 or r['experiment']!='BP-KERNEL-3' or r['mode'] not in ('run','dry-run'): raise ValueError('receipt schema')
    if r['verdict'] not in ('INCONCLUSIVE','DRY_RUN','F_RED','G_RED','CONFIRMED','PREDICTION_FALSIFIED'): raise ValueError('receipt verdict')
    if r['mode']=='dry-run' and r['verdict'] not in ('DRY_RUN','INCONCLUSIVE'): raise ValueError('CPU native claim')
    if r['verdict']!='INCONCLUSIVE':
        if not r['correctness'] or set(r['timings'])!=set(VARIANTS) or set(r['half_timings'])!={'f',*G}: raise ValueError('incomplete receipt')
        for key,variants in [('launch_order',VARIANTS),('half_launch_order',('f',*G))]:
            expected=[dict(phase=p,round=i,variant=v) for p,n in [('warmup',3),('measured',10)] for i in range(n) for v in variants]
            if r[key]!=expected: raise ValueError('launch protocol')
        for v,t in r['timings'].items():
            ok=r['correctness']['control_finite'] if v=='a' else r['correctness']['candidates'][v]['verdict']=='GREEN'
            if t['gate_green']!=ok: raise ValueError('gate drift')
            for data in [t,*r['half_timings'].get(v,{}).values()]:
                samples=data['samples_ms']
                if len(samples)!=10 or any(not math.isfinite(x) or x<=0 for x in samples): raise ValueError('samples')
                if data['mean_ms']!=sum(samples)/10: raise ValueError('mean')
                if data['timing_counts']!=(r['mode']=='run' and ok): raise ValueError('ineligible timing')
        if (r['best_green'],r['secondary'],r['verdict'])!=decision(r['correctness'],r['timings'],r['mode']=='dry-run'): raise ValueError('decision drift')
    elif any(t.get('timing_counts') for t in r['timings'].values()): raise ValueError('incomplete timing counts')
    json.dumps(r,allow_nan=False);return r


def main():
    p=argparse.ArgumentParser(description=__doc__);group=p.add_mutually_exclusive_group(required=True)
    for flag in ('dry-run','run','worker'): group.add_argument('--'+flag,action='store_true')
    args=p.parse_args();reg=verify_registration()
    if args.dry_run:
        started=time.monotonic();backend=CPU(parent.old.seed_inputs(parent.old.TINY),parent.old.TINY)
        ref=reference(backend.x,backend.state,parent.old.TINY);r=empty_receipt(True)
        experiment(backend,tuple(ref[n] for n in NAMES),r,lambda e:None)
        r['elapsed_seconds']=time.monotonic()-started
        path=ART/('dry_kernel_'+uuid.uuid4().hex)/'receipt.json'
        path.parent.mkdir()
        create_json(path,validate_receipt(r));print(f'DRY_RUN {path}');return
    if args.worker:
        if os.environ.get('BP_KERNEL_2_WORKER')!='1': raise ValueError('supervisor required')
        def emit(e): print(json.dumps(e,allow_nan=False),flush=True)
        emit(dict(kind='gpu',description=gpu_preflight()))
        backend=GPU(arrays(INPUT),arrays(STATE),reg);ref=arrays(REFERENCE);r=empty_receipt(False)
        started=time.monotonic();experiment(backend,tuple(ref[n] for n in NAMES),r,emit)
        r['elapsed_seconds']=time.monotonic()-started;emit(dict(kind='result',receipt=validate_receipt(r)));return
    def summarize(events,error,elapsed):
        done=[e['receipt'] for e in events if e.get('kind')=='result'];r=done[-1] if done else empty_receipt(False)
        if error or not done:
            r['verdict']='INCONCLUSIVE';r['error']=error or 'missing worker result'
            for e in events:
                if e.get('kind')=='gate': r['correctness']=e['correctness']
            for t in r['timings'].values(): t['timing_counts']=False
            for halves in r['half_timings'].values():
                for t in halves.values(): t['timing_counts']=False
        r['elapsed_seconds']=elapsed;return validate_receipt(r)
    supervise(Path(__file__).resolve(),ART,sha(REG),summarize)

if __name__=='__main__': main()
