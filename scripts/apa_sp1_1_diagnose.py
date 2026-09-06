#!/usr/bin/env python3
"""CPU reconstruction of the immutable SP1 baseline failure, bounded by caller."""
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'tensor_cuda/tests')]
from apa_sp1_cpu import registration, sha
from apa_sp1_gpu import arrays, metrics
import apa_sp1_reference as ref
import apa_sp1_1_reference as ordered


def diagnose():
    reg = registration()
    name = 'prefill_s2048_d64_c1_h4_kv4'
    ix, shape = next((i,s) for i,s in enumerate(reg['shapes']) if s['id']==name)
    q,k,kq,v = arrays(shape,np.random.SeedSequence([reg['data']['gpu_seed'],ix]))
    oldout = np.empty((1,4,2048,64), np.float32)
    orderedout = np.empty_like(oldout)
    witnesses = []
    counters = dict(dot_only=0,stats_only=0,combined=0)
    for h in range(4):
        for first in range(0,2048,32):
            last = first+32
            qr=q[0,h,first:last];lengths=np.arange(first,last)+1
            b=qr@kq[0,h].T*np.float32(0.125)
            e=qr@k[0,h].T*np.float32(0.125)
            cb=ordered.cuda_bulk(qr,kq[0,h],0.125)
            ce=ordered.cuda_bulk(qr,k[0,h],0.125)
            zm,thr=ref.zmask(b,reg['zthr'],lengths)
            cm,cthr=ordered.cuda_zmask(cb,reg['zthr'],lengths)
            dotmask,_=ref.zmask(cb,reg['zthr'],lengths)
            statmask,_=ordered.cuda_zmask(b,reg['zthr'],lengths)
            for key,m in [('dot_only',dotmask),('stats_only',statmask),('combined',cm)]:
                counters[key]+=int(np.count_nonzero(m!=zm))
            vv=np.broadcast_to(v[0,h],(32,2048,64))
            oldout[0,h,first:last]=ref.dense_scores(np.where(zm,e,b),vv,lengths)
            orderedout[0,h,first:last]=ref.dense_scores(np.where(cm,ce,cb),vv,lengths)
            for row,j in np.argwhere(cm!=zm):
                bits=lambda x: hex(int(np.asarray(x,dtype=np.float32).view(np.uint32)))
                witnesses.append(dict(batch=0,head=h,query=int(first+row),key=int(j),valid_keys=int(lengths[row]),
                    original_bulk=float(b[row,j]),ordered_bulk=float(cb[row,j]),
                    original_thr=float(thr[row]),ordered_thr=float(cthr[row]),
                    original_bulk_bits=bits(abs(b[row,j])),ordered_bulk_bits=bits(abs(cb[row,j])),
                    original_thr_bits=bits(thr[row]),ordered_thr_bits=bits(cthr[row]),
                    original_refine=bool(zm[row,j]),ordered_refine=bool(cm[row,j]),
                    dot_only_refine=bool(dotmask[row,j]),stats_only_refine=bool(statmask[row,j]),
                    exact_original=float(e[row,j]),exact_ordered=float(ce[row,j]),
                    row_max_abs_output_shift=float(np.max(np.abs(oldout[0,h,first+row]-orderedout[0,h,first+row])))))
        print('reconstructed head',h,flush=True)
    path=next((ROOT/'artifacts/apa_sp1/gpu').glob(name+'.*.json'))
    observed=json.loads(path.read_text())['G2']['baseline_vs_own_emulator']
    result=dict(evidence_class='CPU arithmetic-order reconstruction against supplied GPU aggregate receipt',
        status='RECONSTRUCTED',verdict='EMULATOR_ORDER_SENSITIVE',shape=shape,shape_index=ix,
        seed=[reg['data']['gpu_seed'],ix],witnesses=witnesses,decision_difference_counts=counters,
        reconstructed_output_difference=metrics(orderedout,oldout),observed_gpu_vs_original_emulator=observed,
        source_receipt=str(path.relative_to(ROOT)),source_receipt_sha256=sha(path),
        sources={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'tensor_cuda/tests/apa_sp1_1_reference.py']},
        limitations='Original receipt saved aggregate errors, not baseline output/mask. Witnesses are CPU reconstructions, not observed GPU key diagnostics. GPU confirmation remains blocked. CUDA expf/online-softmax order is not reproduced by materialized output.',
        change='Original kernel and mathematical emulator unchanged; added arithmetic-order diagnostic reference only; original G2 FAIL retained.',
        causal='S=L=2048: valid keys 0..query inclusive in both; all witnesses inside bounds',
        gqa='H=KVH=4, group=1: query head equals KV head',
        scope_note='this establishes nothing about model quality')
    dest=ROOT/'artifacts/apa_sp1'/f'sp1_1_diagnosis.{time.time_ns()}.json'
    with dest.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(dest.relative_to(ROOT));print(json.dumps(result,indent=2))


if __name__=='__main__':diagnose()
