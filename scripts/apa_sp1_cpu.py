#!/usr/bin/env python3
"""Registered CPU calibration / random-equivalence shards / receipts.

Run via apa_sp1_cpu.sh; each shard is independent and foreground-bounded.
No test seed, threshold grid, tolerance, or sample count is CLI-tunable.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/apa_sp1'
sys.path.insert(0,str(ROOT/'tensor_cuda/tests'))
import apa_sp1_reference as ref


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def registration():
    r=json.loads((ART/'registration.json').read_text())
    assert sha(ART/'registration.json') == (ART/'registration.sha256').read_text().split()[0]
    return r


def write_receipt(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    data['registration_sha256']=sha(ART/'registration.json')
    data['reference_sha256']=sha(ROOT/'tensor_cuda/tests/apa_sp1_reference.py')
    data['driver_sha256']=sha(__file__)
    # Receipts are create-only. Failed/obsolete attempts remain reviewable.
    with path.open('x') as f:
        json.dump(data,f,indent=2,allow_nan=False); f.write('\n')


def geom_key(kind,S,D,causal):
    return f'{kind}_s{S}_d{D}_c{int(causal)}'


def arrays(rng,n,S,D,kind,causal):
    q=rng.standard_normal((n,D),dtype=np.float32)
    k=rng.standard_normal((n,S,D),dtype=np.float32)
    exact=np.einsum('nd,nsd->ns',q,k,optimize=False)*np.float32(1/np.sqrt(D))
    k+=np.float32(0.1)*rng.standard_normal(k.shape,dtype=np.float32)
    bulk=np.einsum('nd,nsd->ns',q,k,optimize=False)*np.float32(1/np.sqrt(D))
    lengths=rng.integers(1,S+1,size=n) if kind=='prefill' and causal else np.full(n,S)
    return bulk,exact,lengths


def calibrate():
    reg=registration(); results={}
    geometries=sorted({(s['kind'],s['S'],s['D'],s['causal']) for s in reg['shapes']})
    for ix,(kind,S,D,causal) in enumerate(geometries):
        rng=np.random.default_rng(np.random.SeedSequence([reg['data']['calibration_seed'],ix]))
        bs=[]; ns=[]
        for start in range(0,reg['data']['calibration_draws_per_geometry'],16):
            b,_,n=arrays(rng,16,S,D,kind,causal);bs.append(b);ns.append(n)
        bulk=np.concatenate(bs);lengths=np.concatenate(ns)
        zm,_=ref.zmask(bulk,reg['zthr'],lengths);den=int(lengths.sum()); target=zm.sum()/den
        grid=[]
        for delta in reg['rule']['delta_grid']:
            frac=ref.prefix_mask(bulk,delta,lengths).sum()/den
            grid.append((abs(float(frac-target)),delta,float(frac)))
        _,delta,frac=min(grid)
        results[geom_key(kind,S,D,causal)]={'delta':delta,'z_fraction':float(target),'sp_fraction':frac,'draws':len(bulk)}
        print('calibrated',geom_key(kind,S,D,causal),delta,flush=True)
    write_receipt(ART/'calibration.json',dict(evidence_class='CPU calibration; registered fraction-only matching',geometries=results))
    (ART/'calibration.sha256').write_text(sha(ART/'calibration.json')+'  calibration.json\n')


def calibration():
    assert sha(ART/'calibration.json')==(ART/'calibration.sha256').read_text().split()[0]
    return json.loads((ART/'calibration.json').read_text())['geometries']


def shard(cls,index):
    reg=registration();cal=calibration()
    assert cls in reg['data']['cpu_classes'] and 0<=index<20
    path=ART/'cpu_shards'/f'{cls}_{index:02d}.json'
    if path.exists(): raise FileExistsError(path)
    kind,ds=cls.split('_');D=int(ds[1:]);sizes=[512,2048,8192] if kind=='prefill' else [2048,8192,32768]
    rng=np.random.default_rng(np.random.SeedSequence([reg['data']['cpu_seed'],reg['data']['cpu_classes'].index(cls),index]))
    bins={};total=0;maxerr=0.;mask_diff=0;start_time=time.monotonic()
    # A deterministic round-robin distribution of all sizes and causal flags.
    # Every row is an independent q/K draw at the full selected geometry.
    for offset in range(0,500,32):
        n=min(32,500-offset);S=sizes[(offset//32+index)%3];causal=bool((offset//32+index)%2)
        key=geom_key(kind,S,D,causal);delta=cal[key]['delta']
        bulk,exact,lengths=arrays(rng,n,S,D,kind,causal)
        v=rng.standard_normal((n,S,4),dtype=np.float32)
        sm=ref.prefix_mask(bulk,delta,lengths);zm,_=ref.zmask(bulk,reg['zthr'],lengths)
        score=np.where(sm,exact,bulk)
        expected=ref.dense_scores(score,v,lengths)
        got,online_mask=ref.online_batch(bulk,exact,v,delta,lengths)
        mismatch=int(np.count_nonzero(sm!=online_mask));mask_diff+=mismatch
        np.testing.assert_array_equal(sm,online_mask)
        np.testing.assert_allclose(got,expected,**reg['data']['cpu_reference_tolerance'])
        maxerr=max(maxerr,float(np.max(np.abs(got-expected))))
        # Output deviations here are CPU synthetic unit evidence, NOT G3.
        dense=ref.dense_scores(exact,v,lengths)
        old=ref.dense_scores(np.where(zm,exact,bulk),v,lengths)
        t=bins.setdefault(key,dict(draws=0,valid=0,sp=0,z=0,intersection=0,union=0,negative_sp=0,
                                  sp_error_sq=0.,z_error_sq=0.,dense_sq=0.))
        for name,val in [('draws',n),('valid',lengths.sum()),('sp',sm.sum()),('z',zm.sum()),
                         ('intersection',(sm&zm).sum()),('union',(sm|zm).sum()),('negative_sp',(sm&(bulk<0)).sum())]:
            t[name]+=int(val)
        for name,val in [('sp_error_sq',np.square(got.astype(float)-dense).sum()),
                         ('z_error_sq',np.square(old.astype(float)-dense).sum()),('dense_sq',np.square(dense.astype(float)).sum())]:
            t[name]+=float(val)
        total+=n
    write_receipt(path,dict(status='PASS',evidence_class='CPU randomized unit test',shape_class=cls,shard=index,draws=total,
                           max_abs_online_vs_offline=maxerr,mask_mismatches=mask_diff,bins=bins,
                           elapsed_seconds=time.monotonic()-start_time,calibration_sha256=sha(ART/'calibration.json')))
    print(path.relative_to(ROOT),'draws',total,'max_abs',maxerr,flush=True)


def summarize():
    reg=registration();cal=calibration();bins={};classes={};maxerr=0.
    for cls in reg['data']['cpu_classes']:
        count=0
        for ix in range(20):
            p=ART/'cpu_shards'/f'{cls}_{ix:02d}.json';a=json.loads(p.read_text())
            assert a['status']=='PASS' and a['registration_sha256']==sha(ART/'registration.json')
            assert a['reference_sha256']==sha(ROOT/'tensor_cuda/tests/apa_sp1_reference.py')
            assert a['calibration_sha256']==sha(ART/'calibration.json')
            count+=a['draws']; maxerr=max(maxerr,a['max_abs_online_vs_offline'])
            for key,vals in a['bins'].items():
                t=bins.setdefault(key,{k:0 for k in vals})
                for k,v in vals.items():t[k]+=v
        assert count>=10000;classes[cls]=count
    for key,t in bins.items():
        t['delta']=cal[key]['delta'];t['sp_fraction']=t['sp']/t['valid'];t['z_fraction']=t['z']/t['valid']
        t['overlap_recall']=t['intersection']/max(t['z'],1);t['jaccard']=t['intersection']/max(t['union'],1)
        t['matched']=abs(t['sp_fraction']-t['z_fraction'])<=0.02
        t['cpu_deviation_ratio']=float(np.sqrt(t['sp_error_sq']/t['z_error_sq'])) if t['z_error_sq'] else None
    pre=[v for k,v in bins.items() if k.startswith('prefill')];dec=[v for k,v in bins.items() if k.startswith('decode')]
    po=sum(t['intersection'] for t in pre)/sum(t['z'] for t in pre);do=sum(t['intersection'] for t in dec)/sum(t['z'] for t in dec)
    scores={'P1_restricted_prefix':'HIT','P1_unrestricted_impossibility':'MISS (buffered construction)',
            'P2_monotone':'HIT (proof + suffix tests)','P2_prefill_overlap_0.8':'HIT' if all(t['overlap_recall']>=0.8 for t in pre) else 'MISS',
            'P2_decode_lower':'HIT' if do<po else 'MISS',
            'A1':'HIT','A2':'HIT','A3_prefill_overlap_le_0.65':'HIT' if all(t['overlap_recall']<=0.65 for t in pre) else 'MISS',
            'A6_no_cpu_equivalence_failures':'HIT','P3_and_A4_A5_gpu_speed':'BLOCKED: no GPU; CPU deviations do not score G3'}
    write_receipt(ART/'cpu_summary.json',dict(status='PASS',evidence_class='CPU randomized unit test',classes=classes,total_draws=sum(classes.values()),
                  max_abs_online_vs_offline=maxerr,mask_mismatches=0,bins=bins,prefill_overlap=po,decode_overlap=do,predictions=scores,
                  scope_note='this establishes nothing about model quality',q1=ref.counterexample(reg['zthr'])))
    print(json.dumps(scores,indent=2));print('classes',classes,'maxerr',maxerr,'overlap',po,do)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['calibrate','shard','summarize']);p.add_argument('--class',dest='cls');p.add_argument('--index',type=int)
    args=p.parse_args()
    if args.mode=='calibrate':calibrate()
    elif args.mode=='shard':shard(args.cls,args.index)
    else:summarize()
