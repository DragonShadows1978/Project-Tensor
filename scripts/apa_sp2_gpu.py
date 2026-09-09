#!/usr/bin/env python3
"""Leased, bounded SP2 G2/G3. No model quality evidence or epsilon recommendation."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from apa_sp2_common import (ROOT, ART, NOTE, sha, registration, load_runtime,
                            fingerprint, write_new, receipt, latest, targets, decode_target)
sys.path.insert(0, str(ROOT/'tensor_cuda/tests'))
import apa_sp1_reference as ref
from apa_sp1_1_reference import partition_mask, cuda_bulk
from apa_sp1_gpu import Events, metrics


def require_lease():
    if os.environ.get('APA_SP2_LEASED') != '1':
        raise RuntimeError('BLOCKED: invoke scripts/apa_sp2_lead_gpu.sh')
    held = os.fstat(9)
    expected = os.stat('/tmp/forge-gpu.lock')
    if (held.st_dev, held.st_ino) != (expected.st_dev, expected.st_ino):
        raise RuntimeError('BLOCKED: inherited fd 9 is not the GPU lease')
    with open('/tmp/forge-gpu.lock', 'a') as probe:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        raise RuntimeError('BLOCKED: GPU lease is not locked')


def data(shape, index, seed, draw=0, sample=False):
    rng = np.random.default_rng(np.random.SeedSequence([seed, index, draw]))
    B,H,L,D = [shape[k] for k in ('B','H','L','D')]
    q = rng.standard_normal((B,H,L,D), dtype=np.float32)
    k = rng.standard_normal((B,shape['KVH'],shape['S'],D), dtype=np.float32)
    v = rng.standard_normal((*k.shape[:-1],shape['VD']), dtype=np.float32)
    indices = np.arange(L)
    if sample:
        indices = np.linspace(0,L-1,min(32,L)).astype(int)
        q = np.ascontiguousarray(q[:,:,indices,:])
    return q,k,v,indices


def quantized(tc, k, bits):
    kt = tc.tensor(k)
    kqt = tc.quantize_sp2_keys(kt, bits)
    return kt,kqt


def stats(values):
    values = np.asarray(values, np.float64)
    if not values.size or not np.isfinite(values).all():
        raise ValueError('empty or nonfinite error population')
    return dict(count=int(values.size), max=float(values.max()),
                p99_9=float(np.quantile(values,.999,method='linear')),
                mean=float(values.mean()))


def measure(tc, bits, shape, index, reg):
    chunks = []
    checked_scores = 0
    scale = np.float32(1/math.sqrt(shape['D']))
    for draw in range(reg['eq_rule']['draws_per_shape']):
        q,k,_,indices = data(shape,index,reg['eq_rule']['measurement_seed'],draw,True)
        qt = tc.tensor(q); kt,kqt = quantized(tc,k,bits)
        bulk, exact = tc._C.apa_sp2_scores(qt,kt,kqt,float(scale),0,len(indices),
                                         shape['kind']=='prefill' or shape['D']>64)
        bulk = bulk.numpy(); exact = exact.numpy()
        reconstructed=kqt.numpy()
        # Independent CPU FMA/shuffle emulator pins the measurement diagnostic
        # on first/last 17 keys and first two sampled queries of every head.
        key_indices=np.unique(np.r_[np.arange(17),np.arange(shape['S']-17,shape['S'])])
        for b in range(shape['B']):
            for h in range(shape['H']):
                kh = h//(shape['H']//shape['KVH'])
                for keys,observed in [(reconstructed,bulk),(k,exact)]:
                    want=cuda_bulk(q[b,h,:2],keys[b,kh,key_indices],float(scale),
                                   wcoop=shape['kind']=='prefill' or shape['D']>64)
                    np.testing.assert_array_equal(observed[b,h,:2][:,key_indices],want)
                    checked_scores+=want.size
                truth = q[b,h].astype(np.float64) @ k[b,kh].astype(np.float64).T * float(scale)
                lengths = shape['S']-shape['L']+indices+1 if shape['causal'] else np.full(len(indices),shape['S'])
                valid = np.arange(shape['S'])[None,:] < lengths[:,None]
                chunks.append(np.abs(bulk[b,h].astype(np.float64)-truth)[valid])
        del qt,kt,kqt
    errors = np.concatenate(chunks)
    result = stats(errors)
    path = ART/'gpu'/f'eq_values_b{bits}_{shape["id"]}.{time.time_ns()}.npy'
    path.parent.mkdir(exist_ok=True,parents=True)
    with path.open('xb') as f:
        np.save(f,errors,allow_pickle=False)
    return dict(status='PASS',bits=bits,shape=shape,statistics=result,
                error_values=str(path.relative_to(ROOT)),error_values_sha256=sha(path),
                evidence_class='GPU quantizer and kernel-order diagnostic bulk vs FP64 exact logits',
                diagnostic_cpu_order_pin=dict(checked_scores=checked_scores,bit_mismatches=0),
                query_draws=reg['eq_rule']['draws_per_shape']*len(indices)*shape['B']*shape['H'])


def freeze():
    reg=registration(); rows=[]
    for target in targets(reg)[0]:
        row=latest(target)
        if not row or row['status']!='PASS':
            raise RuntimeError(f'BLOCKED: G2 missing/failed {target}')
        rows.append(row)
    entries={}
    for bits in reg['bits']:
        for D in (64,128):
            group=[r for r in rows if r['bits']==bits and r['shape']['D']==D]
            arrays=[]
            for r in group:
                path=ROOT/r['error_values']
                assert sha(path)==r['error_values_sha256']
                arrays.append(np.load(path,allow_pickle=False))
            values=np.concatenate(arrays); info=stats(values)
            info['e_q']=float(np.nextafter(np.float32(info['max']),np.float32(np.inf)))
            info['receipts']={r['_receipt']:sha(ROOT/r['_receipt']) for r in group}
            entries[f'{bits}:{D}']=info
    table=dict(status='FROZEN',registration_sha256=sha(ART/'registration.json'),
               quantizer=reg['quantizer']['id'],statistic=reg['eq_rule']['statistic'],
               measurement_classes=len(rows),entries=entries,fingerprint=fingerprint(),
               scope_note=NOTE,validity=reg['eq_rule']['validity'])
    path=ART/'e_q_table.json'
    if path.exists():
        if json.loads(path.read_text())!=table:
            raise RuntimeError('immutable e_q table already exists with different evidence; separate amendment required')
    else:
        write_new(path,table)
    pin=ART/'e_q_table.sha256'
    if not pin.exists():
        with pin.open('x') as f:f.write(sha(path)+'  e_q_table.json\n')
    assert pin.read_text().split()[0]==sha(path)
    print('FROZEN',sha(path))


def load_margins(tc):
    table=ART/'e_q_table.json';pin=ART/'e_q_table.sha256'
    if not table.exists() or not pin.exists():
        raise RuntimeError('BLOCKED: freeze complete G2 margins before any epsilon sweep')
    metadata=json.loads(table.read_text())
    if metadata['fingerprint']!=fingerprint():
        raise RuntimeError('BLOCKED: frozen table implementation fingerprint is stale; amendment required')
    for entry in metadata['entries'].values():
        for path,digest in entry['receipts'].items():
            assert sha(ROOT/path)==digest
    return tc.RegisteredMargins.load(table,expected_sha256=pin.read_text().split()[0])


def sweep(tc,bits,s,index,eindex,reg):
    margins=load_margins(tc)
    epsilon=reg['epsilon_grid'][eindex]; D=s['D'];scale=np.float32(1/math.sqrt(D))
    e_q=margins.lookup(bits,D,'float32',float(scale));delta=tc._C.apa_sp2_delta(epsilon,e_q)
    q,k,v,_=data(s,index,reg['sweep']['seed'])
    qt=tc.tensor(q);kt,kqt=quantized(tc,k,bits);vt=tc.tensor(v)
    args=(qt,kt,kqt,vt,float(scale))
    new=lambda:tc.apa_selective_attention_sp(*args,epsilon,bulk_bits=bits,margins=margins,is_causal=s['causal'])
    old=lambda:tc.apa_selective_attention(*args,reg['baseline']['zthr'],s['causal'])
    with tc.no_grad():
        out,mask=tc.apa_selective_attention_sp(*args,epsilon,bulk_bits=bits,margins=margins,is_causal=s['causal'],diagnostics=True)
        got=out.numpy(); gm=mask.numpy(); del out,mask
        plain=new().numpy()
        direct=tc._C.apa_selective_attention_sp(*args,delta,s['causal']).numpy()
        baseline=old().numpy()
        thresholds=tc._C.apa_sp2_baseline_thresholds(qt,kqt,float(scale),reg['baseline']['zthr'],s['causal']).numpy()
    diag_identical=np.array_equal(got,plain);direct_identical=np.array_equal(plain,direct)
    dense=np.empty_like(got); own=np.empty_like(got); oldref=np.empty_like(got)
    counts=dict(valid=0,sp=0,z=0,mask_disagreements=0,eq_exceedances=0,skip_floor_violations=0)
    max_error=0.; max_skipped=0; max_bound=0.; max_mass=0.;max_rel_mass=0.
    wc=s['kind']=='prefill' or D>64
    for first in range(0,s['L'],reg['sweep']['chunk_queries']):
        n=min(reg['sweep']['chunk_queries'],s['L']-first)
        bdev,edev=tc._C.apa_sp2_scores(qt,kt,kqt,float(scale),first,n,wc)
        bulk=bdev.numpy(); exact=edev.numpy();del bdev,edev
        lengths=s['S']-s['L']+np.arange(first,first+n)+1 if s['causal'] else np.full(n,s['S'])
        valid=np.arange(s['S'])[None,:]<lengths[:,None]
        for b in range(s['B']):
            for h in range(s['H']):
                kh=h//(s['H']//s['KVH']);bs=bulk[b,h];ex=exact[b,h]
                wantmask=(partition_mask(bs,delta,lengths=lengths) if s['kind']=='decode' else ref.prefix_mask(bs,delta,lengths))
                actual=(gm[b,h,first:first+n]!=0)&valid
                zmask=(np.abs(bs)>=thresholds[b,h,first:first+n,None])&valid
                counts['valid']+=int(valid.sum());counts['sp']+=int(actual.sum());counts['z']+=int(zmask.sum())
                counts['mask_disagreements']+=int(np.count_nonzero(actual!=wantmask))
                values=np.broadcast_to(v[b,kh],(n,s['S'],s['VD']))
                exact32=q[b,h,first:first+n] @ k[b,kh].T * scale
                dense[b,h,first:first+n]=ref.dense_scores(exact32,values,lengths)
                own[b,h,first:first+n]=ref.dense_scores(np.where(wantmask,ex,bs),values,lengths)
                oldref[b,h,first:first+n]=ref.dense_scores(np.where(zmask,ex,bs),values,lengths)
                truth=q[b,h,first:first+n].astype(np.float64) @ k[b,kh].astype(np.float64).T * float(scale)
                errors=np.abs(bs.astype(np.float64)-truth)
                counts['eq_exceedances']+=int(np.count_nonzero((errors>e_q)&valid))
                max_error=max(max_error,float(errors[valid].max()))
                star=np.where(valid,truth,-np.inf).max(-1,keepdims=True)
                relative=np.where(valid,np.exp(truth-star),0.)
                skipped=valid&~actual;N=skipped.sum(-1)
                counts['skip_floor_violations']+=int(np.count_nonzero(skipped&(relative>=epsilon)))
                max_skipped=max(max_skipped,int(N.max()));max_bound=max(max_bound,float((N*epsilon).max()))
                mass=(relative*skipped).sum(-1);max_rel_mass=max(max_rel_mass,float(mass.max()))
                max_mass=max(max_mass,float((mass/relative.sum(-1)).max()))
    tol={k:reg['sweep']['gates'][k] for k in ('atol','rtol')}
    parity=dict(diagnostic_bit_identical=diag_identical,direct_delta_bit_identical=direct_identical,
                mask_disagreements=counts['mask_disagreements'],sp_vs_own=metrics(got,own),
                baseline_vs_diagnostic_materialization=metrics(baseline,oldref),tolerance=tol)
    passed=diag_identical and direct_identical and counts['mask_disagreements']==0 and np.allclose(got,own,**tol) and np.allclose(baseline,oldref,**tol)
    row=dict(status='PASS' if passed else 'FAIL',bits=bits,shape=s,epsilon=epsilon,eindex=eindex,
             e_q=e_q,delta=delta,margin_table_sha256=margins.sha256,parity=parity,
             fractions=dict(sp=counts['sp']/counts['valid'],baseline=counts['z']/counts['valid']),counts=counts,
             sp_vs_dense_fp32=metrics(got,dense),baseline_vs_dense_fp32=metrics(baseline,dense),
             guarantee=dict(status='EMPIRICAL_BOUND_HELD' if counts['eq_exceedances']==0 and counts['skip_floor_violations']==0 else 'RED_BOUND_EXCEEDED',
                            heldout_max_error=max_error,max_skipped_in_row=max_skipped,
                            max_N_epsilon=max_bound,max_observed_skipped_mass_over_wstar=max_rel_mass,
                            max_observed_skipped_probability_mass=max_mass,
                            note='conditional real-arithmetic theorem; float refinement/exp/reduction error also remains'),
             scope_note=NOTE)
    if not passed:
        row['timing']={'status':'BLOCKED_BY_PARITY'}
        return row
    events=Events();gpu={'sp':[],'baseline':[]};wall={'sp':[],'baseline':[]}
    try:
        with tc.no_grad():
            for _ in range(reg['sweep']['warmups']):new();old()
            tc.synchronize()
            for rep in range(reg['sweep']['repetitions']):
                pair=[('baseline',old),('sp',new)] if rep%2==0 else [('sp',new),('baseline',old)]
                for label,fn in pair:
                    ms,wms=events.measure(fn);gpu[label].append(ms);wall[label].append(wms)
    finally:
        events.close()
    row['timing']=dict(status='PASS',gpu_ms=gpu,wall_ms=wall,
                       speedup=float(np.median(gpu['baseline'])/np.median(gpu['sp'])),
                       wall_speedup=float(np.median(wall['baseline'])/np.median(wall['sp'])))
    return row


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['list','next','summary','freeze','worker','validate'])
    parser.add_argument('target',nargs='?');args=parser.parse_args();reg=registration()
    if args.mode=='list':
        print('\n'.join(targets(reg)[0]+['freeze']+targets(reg)[1]));return
    if args.mode=='validate':decode_target(args.target,reg);return
    if args.mode=='summary':
        from apa_sp2_summary import summary
        summary();return
    if args.mode=='freeze':freeze();return
    if args.mode=='next':
        for target in targets(reg)[0]+['freeze']+targets(reg)[1]:
            if target=='freeze':
                if not (ART/'e_q_table.json').exists():print(target);return
                continue
            row=latest(target)
            if row is None:print(target);return
            if row['status']!='PASS':print('BLOCKED_FAILED_'+target);return
        print('DONE');return
    kind,bits,shape,index,eindex=decode_target(args.target,reg)
    try:
        require_lease();tc=load_runtime()
        with tc.no_grad():
            result=measure(tc,bits,shape,index,reg) if kind=='eq' else sweep(tc,bits,shape,index,eindex,reg)
    except Exception as exc:
        receipt(args.target,dict(status='ERROR',error=repr(exc),traceback=traceback.format_exc()))
        raise
    receipt(args.target,result)
    if result['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
