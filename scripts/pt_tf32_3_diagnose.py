#!/usr/bin/env python3
"""CPU ablations only, no native import. Prior art: Higham (2002) rounding
analysis, NVIDIA TF32 (2020), Ootomo/Yokota (2022) residual products, taken.
Ours: separate saved-threshold accumulation from score error and expose dV
quantization jumps. CPU arithmetic is not an emulation of native HMMA order.
"""
import json
from pathlib import Path
import numpy as np
import pt_tf32_2 as t


def threshold_ablation(s):
    x=t.fixture(s);counts=dict(fp32_sum=0,fp64_sum=0,fp64_sum_fp32_store=0)
    errors={k:0. for k in counts};examples=[]
    for h in range(s['H']):
        q=x['q'][0,h].astype(np.float64);k=x['kq'][0,h].astype(np.float64)
        for lo in range(0,s['L'],128):
            bulk=(q[lo:lo+128]@k.T)*s['scale'];ab=abs(bulk)
            vis=np.arange(s['S'])[None,:]<=np.arange(lo,lo+len(bulk))[:,None]
            n=vis.sum(1);mean=np.where(vis,ab,0.).sum(1)/n
            var=np.where(vis,(ab-mean[:,None])**2,0.).sum(1)/n
            ref=mean+s['zthr']*np.sqrt(var)
            exact=vis & (ab>=ref[:,None]);exact[n==1]=vis[n==1]
            # Controlled experiment: identical rounded score matrix in all
            # arms. This isolates statistics/storage, not TF32x3 score error.
            a=np.where(vis,ab,0.).astype(np.float32)
            acc=np.zeros(len(a),np.float32);sqr=acc.copy()
            for j in range(s['S']):
                acc+=a[:,j];sqr+=a[:,j]*a[:,j]
            mu=acc/n.astype(np.float32)
            naive=mu+np.float32(s['zthr'])*np.sqrt(np.maximum(sqr/n.astype(np.float32)-mu*mu,0))
            da=a.astype(np.float64);mu64=da.sum(1)/n
            precise=mu64+s['zthr']*np.sqrt(np.maximum((da*da).sum(1)/n-mu64*mu64,0))
            for key,thr in [('fp32_sum',naive),('fp64_sum',precise),('fp64_sum_fp32_store',precise.astype(np.float32))]:
                selected=vis & (a>=thr[:,None]);selected[n==1]=vis[n==1]
                diff=selected!=exact;counts[key]+=int(diff.sum())
                errors[key]=max(errors[key],float(np.max(abs(thr-ref))))
                if key=='fp32_sum':
                    for i,j in np.argwhere(diff)[:2]:
                        if len(examples)<8:examples.append(dict(h=h,i=int(lo+i),j=int(j),score=float(ab[i,j]),reference=float(ref[i]),naive=float(naive[i]),precise=float(precise[i])))
    return dict(shape=s,selection_flips=counts,threshold_max_abs=errors,examples=examples)


def edge_midpoints(L,S):
    s=dict(B=1,H=4,KVH=2,L=L,S=S,D=96,VD=64,causal=True,scale=float(np.float32(1/np.sqrt(96))),zthr=-100.)
    x=t.fixture(s);x['kq']=x['k'].copy();f=t.num.forward_reference(x,s)
    witnesses=[]
    for h in range(4):
        q=x['q'][0,h].astype(np.float64);k=x['k'][0,h//2].astype(np.float64)
        scores=q@k.T*s['scale'];vis=np.arange(S)[None,:]<=S-L+np.arange(L)[:,None]
        p=np.exp(scores-f['lse'][0,h,:,None].astype(np.float32));p=np.where(vis,p,0.)
        rounded=t.old.tf32_round(p)
        bits=p.astype(np.float32).view(np.uint32);low=bits & 0x1fff
        distance=abs(low.astype(np.int64)-4096)
        for i,j in np.argwhere(vis & (distance<=16)):
            # One TF32 bin jump in this p creates Delta dV_j=p_ulp*RN(dO_i).
            exponent=int((bits[i,j]>>23)&255)-127
            jump=float(2.**(float(exponent)-10))
            worst=jump*float(np.max(abs(t.old.tf32_round(x['dO'][0,h,i]))))
            witnesses.append(dict(h=h,i=int(i),j=int(j),fp32_ulps_to_midpoint=int(distance[i,j]),probability=float(p[i,j]),tf32_bin=jump,max_dV_jump=worst))
    return dict(shape=s,witnesses=witnesses)


def main():
    out=Path(__file__).resolve().parents[1]/'artifacts/pt_tf32_3/CPU_DIAGNOSIS.json'
    thresholds=[]
    for s in t.registration()['attention_shapes']:
        r=threshold_ablation(t.old.shape_values(s));thresholds.append(r)
        print('THRESHOLD',r['shape']['L'],r['shape']['D'],r['selection_flips'],r['threshold_max_abs'],flush=True)
    result=dict(evidence_class='CPU controlled arithmetic; not native reproduction',
        thresholds=thresholds,edges=[edge_midpoints(17,31),edge_midpoints(33,35)])
    t.create_json(out,result)
    for r in result['edges']:print('EDGE',r['shape']['L'],len(r['witnesses']),'MIDPOINT_WITNESSES',flush=True)


if __name__=='__main__':main()
