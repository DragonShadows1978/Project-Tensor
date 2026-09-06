#!/usr/bin/env python3
"""SP1.1 leased GPU gate implementations; CPU imports do not create a context."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/apa_sp1'
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tensor_cuda/tests')]
from apa_sp1_cpu import registration
import apa_sp1_reference as ref
import apa_sp1_1_reference as sp


def splitk_boundary(tc,reg):
    from apa_sp1_gpu import arrays,all_references,metrics
    cases=[]
    for D,VD,B,S in [(33,7,2,2051),(64,32,1,8192),(128,64,2,8193),(512,17,1,4097)]:
        shape=dict(B=B,H=4,KVH=2,L=1,S=S,D=D,VD=VD,causal=True)
        for dtype,key in [('float32','gpu_fp32_tolerance'),('float16','gpu_fp16_tolerance'),('bfloat16','gpu_bf16_tolerance')]:
            aa=arrays(shape,np.random.SeedSequence([reg['data']['gpu_seed'],D,VD]))
            tt=[tc.tensor(a,dtype=dtype) for a in aa]
            q,k,kq,v=[a.numpy().astype(np.float32) for a in tt]
            sinks=tc.tensor(np.array([-10,0,3,10],np.float32),dtype=dtype)
            with tc.no_grad():
                out,mask=tc._C.apa_selective_attention_sp(*tt,1/np.sqrt(D),.5,True,sinks,diagnostics=True)
                got=out.numpy().astype(np.float32);gm=mask.numpy()
                plain=tc._C.apa_selective_attention_sp(*tt,1/np.sqrt(D),.5,True,sinks).numpy().astype(np.float32)
            want,_,_,counts=all_references(q,k,kq,v,shape,.5,reg['zthr'],gm,sinks.numpy().astype(np.float32),splitk=True)
            ok=bool(np.allclose(got,want,**reg['data'][key])) and np.array_equal(got,plain) and counts['cpu_gpu_mask_disagreements']==0
            cases.append(dict(shape=shape,dtype=dtype,pass_gate=bool(ok),metrics=metrics(got,want),counts=counts,
                              diagnostics_bit_identical=bool(np.array_equal(got,plain))))
    return dict(G2_boundary=dict(status='PASS' if all(c['pass_gate'] for c in cases) else 'FAIL',cases=cases),
                scope_note='this establishes nothing about model quality')


def partb_gate(tc,reg):
    from apa_sp1_gpu import arrays,metrics
    shape=reg['shapes'][10]
    assert shape['id']=='prefill_s2048_d64_c1_h4_kv4'
    q,k,kq,v=arrays(shape,np.random.SeedSequence([reg['data']['gpu_seed'],10]))
    tt=[tc.tensor(a) for a in [q,k,kq,v]]
    with tc.no_grad():
        baseline=tc.apa_selective_attention(*tt,.125,reg['zthr'],True).numpy()
        dt,dm=tc._C.apa_sp1_1_baseline_diagnostics(tt[0],tt[2])
        gpu_thr,gpu_mask=dt.numpy(),dm.numpy().astype(bool)
    ordered=np.empty_like(baseline);original=np.empty_like(baseline)
    disagreements=0;thr_max=0.;witnesses=[]
    for h in range(4):
        for first in range(0,2048,32):
            last=first+32;lengths=np.arange(first,last)+1;qr=q[0,h,first:last]
            b=qr@kq[0,h].T*np.float32(.125);e=qr@k[0,h].T*np.float32(.125)
            cb=sp.cuda_bulk(qr,kq[0,h],.125);ce=sp.cuda_bulk(qr,k[0,h],.125)
            om,ot=ref.zmask(b,reg['zthr'],lengths);cm,ct=sp.cuda_zmask(cb,reg['zthr'],lengths)
            gm=gpu_mask[0,h,first:last];gt=gpu_thr[0,h,first:last]
            disagreements+=int(np.count_nonzero(cm!=gm));thr_max=max(thr_max,float(abs(ct-gt).max()))
            vv=np.broadcast_to(v[0,h],(32,2048,64))
            original[0,h,first:last]=ref.dense_scores(np.where(om,e,b),vv,lengths)
            ordered[0,h,first:last]=ref.dense_scores(np.where(cm,ce,cb),vv,lengths)
            for ri,j in np.argwhere(om!=gm):
                witnesses.append(dict(head=h,query=int(first+ri),key=int(j),original_refine=bool(om[ri,j]),
                    gpu_diagnostic_refine=bool(gm[ri,j]),ordered_refine=bool(cm[ri,j]),original_thr=float(ot[ri]),
                    gpu_thr=float(gt[ri]),ordered_thr=float(ct[ri]),original_bulk=float(b[ri,j]),ordered_bulk=float(cb[ri,j])))
    tol=reg['data']['gpu_fp32_tolerance']
    return dict(G2=dict(status='PASS' if np.allclose(baseline,ordered,**tol) and disagreements==0 else 'FAIL',
                    baseline_vs_ordered=metrics(baseline,ordered),baseline_vs_original=metrics(baseline,original),
                    original_emulator_gate_pass=bool(np.allclose(baseline,original,**tol)),
                    tolerance=tol,ordered_gpu_diagnostic_mask_disagreements=disagreements,threshold_max_abs=thr_max),
                verdict='EMULATOR_ORDER_SENSITIVE',witnesses=witnesses,
                note='Diagnostic runs byte-identical baseline stats plus separately reconstructed pass-two dots. Existing fused baseline output is checked in full. Original SP1 G2 FAIL is not overwritten.',
                scope_note='this establishes nothing about model quality')


def splitk_summary():
    from apa_sp1_gpu import fingerprint
    reg=registration();rows=[];missing=[]
    for s in reg['shapes']:
        if s['kind']!='decode':continue
        found=[]
        for path in (ART/'gpu').glob('splitk_'+s['id']+'.*.json'):
            r=json.loads(path.read_text())
            if r.get('status')=='COMPLETE' and r.get('fingerprint')==fingerprint():
                r['_receipt']=str(path.relative_to(ROOT));found.append((path,r))
        if found:rows.append(sorted(found)[-1][1])
        else:missing.append(s['id'])
    boundary=[]
    for path in (ART/'gpu').glob('splitk_boundary.*.json'):
        r=json.loads(path.read_text())
        if r.get('status')=='COMPLETE' and r.get('fingerprint')==fingerprint():boundary.append((path,r))
    boundary_status=sorted(boundary)[-1][1]['G2_boundary']['status'] if boundary else 'BLOCKED'
    scores={}
    for name,min_s,speed,ratio in [('lead_speed_S_ge_8192',8192,.8,None),('lead_speed_S_eq_32768',32768,1.,None),('lead_deviation_at_matched_fraction',2048,None,1.)]:
        group=[r for r in rows if r['shape']['S']>=min_s]
        expected=sum(s['kind']=='decode' and s['S']>=min_s for s in reg['shapes'])
        eligible=[r for r in group if r['G2']['status']=='PASS' and r['tail']['matched'] and r['G3'].get('speedup') is not None]
        hit=len(eligible)==expected and all(r['G3']['speedup']>=speed if speed is not None else r['G3']['deviation_ratio'] is not None and r['G3']['deviation_ratio']<=ratio for r in eligible)
        scores[name]=dict(verdict='BLOCKED' if len(group)<expected else ('HIT' if hit else 'MISS'),eligible=len(eligible),expected=expected,receipts=[r['_receipt'] for r in group])
    report=dict(evidence_class='kernel sweep',scope_note='this establishes nothing about model quality',
        completed=len(rows),missing=missing,boundary_status=boundary_status,predictions=scores,rows=rows)
    (ART/'sp1_1_gpu_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
