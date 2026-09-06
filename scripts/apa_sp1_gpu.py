#!/usr/bin/env python3
"""Leased G2/G3 driver. Invoke only via apa_sp1_lead_gpu.sh.

Kernel sweep: this establishes nothing about model quality.
No model/corpus is loaded. Diagnostic mask allocations never enter timings.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/apa_sp1'
sys.path.insert(0,str(ROOT/'scripts'))
from apa_sp1_cpu import registration, calibration, geom_key, sha
sys.path.insert(0,str(ROOT/'tensor_cuda/tests'))
import apa_sp1_reference as ref


def fingerprint():
    files=[ART/'registration.json',ART/'calibration.json',ART/'build/manifest.json',
           ROOT/'scripts/apa_sp1_gpu.py',ROOT/'scripts/apa_sp1_lead_gpu.sh',
           ROOT/'tensor_cuda/tests/apa_sp1_reference.py',
           ART/'registration_sp1_1.json',ROOT/'tensor_cuda/tests/apa_sp1_1_reference.py',
           ROOT/'scripts/apa_sp1_1_gpu.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in files}


def load_runtime():
    # Fail closed on module provenance; no installed/main-checkout fallback.
    build=ART/'build'; manifest=json.loads((build/'manifest.json').read_text())
    for p,digest in manifest['sources'].items():
        assert sha(ROOT/p)==digest, f'stale build source: {p}'
    for p,digest in manifest['modules'].items():
        assert sha(build/p)==digest, f'stale module: {p}'
    sys.path[:0]=[str(build),str(ROOT/'tensor_cuda')]
    import tensor_cuda as tc
    assert Path(tc.__file__).resolve().is_relative_to(ROOT/'tensor_cuda')
    assert Path(tc._C.__file__).resolve().parent==build.resolve()
    return tc


class Events:
    def __init__(self):
        self.lib=ctypes.CDLL('/usr/local/cuda-12.6/lib64/libcudart.so')
        self.lib.cudaEventCreate.argtypes=[ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cudaEventRecord.argtypes=[ctypes.c_void_p,ctypes.c_void_p]
        self.lib.cudaEventSynchronize.argtypes=[ctypes.c_void_p]
        self.lib.cudaEventElapsedTime.argtypes=[ctypes.POINTER(ctypes.c_float),ctypes.c_void_p,ctypes.c_void_p]
        self.lib.cudaEventDestroy.argtypes=[ctypes.c_void_p]
        self.a=ctypes.c_void_p();self.b=ctypes.c_void_p()
        self.check(self.lib.cudaEventCreate(ctypes.byref(self.a)))
        self.check(self.lib.cudaEventCreate(ctypes.byref(self.b)))

    @staticmethod
    def check(code):
        if code:raise RuntimeError(f'CUDA event error {code}')

    def measure(self,fn):
        wall=time.perf_counter()
        self.check(self.lib.cudaEventRecord(self.a,None))
        out=fn()
        self.check(self.lib.cudaEventRecord(self.b,None))
        self.check(self.lib.cudaEventSynchronize(self.b))
        wall_ms=(time.perf_counter()-wall)*1000
        ms=ctypes.c_float()
        self.check(self.lib.cudaEventElapsedTime(ctypes.byref(ms),self.a,self.b))
        del out
        return float(ms.value),wall_ms

    def close(self):
        self.check(self.lib.cudaEventDestroy(self.a));self.check(self.lib.cudaEventDestroy(self.b))


def arrays(shape,seed):
    rng=np.random.default_rng(seed)
    B,H,KVH,L,S,D,VD=[shape[x] for x in ['B','H','KVH','L','S','D','VD']]
    q=rng.standard_normal((B,H,L,D),dtype=np.float32)
    k=rng.standard_normal((B,KVH,S,D),dtype=np.float32)
    kq=k+np.float32(0.1)*rng.standard_normal(k.shape,dtype=np.float32)
    v=rng.standard_normal((B,KVH,S,VD),dtype=np.float32)
    return q,k,kq,v


def metrics(got,want):
    bad_got=int(np.count_nonzero(~np.isfinite(got)))
    bad_want=int(np.count_nonzero(~np.isfinite(want)))
    if bad_got or bad_want:
        # Preserve a failed numerical gate as valid JSON, never NaN/Infinity
        # tokens or a serialization failure that loses the failure receipt.
        return dict(max_abs=None,rmse=None,relative_frobenius=None,
                    nonfinite_got=bad_got,nonfinite_reference=bad_want)
    d=got.astype(np.float64)-want.astype(np.float64)
    den=np.square(want.astype(np.float64)).sum()
    return dict(max_abs=float(np.max(np.abs(d))),rmse=float(np.sqrt(np.square(d).mean())),
                relative_frobenius=float(np.sqrt(np.square(d).sum()/max(den,1e-30))))


def all_references(q,k,kq,v,shape,delta,z,newmask,sinks=None,splitk=False):
    # All rows/elements, bounded query chunks; one pair of dot matrices per chunk.
    from apa_sp1_1_reference import partition_mask
    B,H,L,D=q.shape;S=k.shape[2];VD=v.shape[-1];KVH=k.shape[1]
    sp=np.empty((B,H,L,VD),np.float32);old=np.empty_like(sp);dense=np.empty_like(sp)
    counts=dict(valid=0,sp=0,z=0,intersection=0,union=0,negative_sp=0,cpu_gpu_mask_disagreements=0)
    for b in range(B):
        for h in range(H):
            kh=h//(H//KVH)
            for first in range(0,L,32):
                last=min(first+32,L);n=last-first
                lengths=S-L+np.arange(first,last)+1 if shape['causal'] else np.full(n,S)
                bulk=q[b,h,first:last]@kq[b,kh].T*np.float32(1/np.sqrt(D))
                exact=q[b,h,first:last]@k[b,kh].T*np.float32(1/np.sqrt(D))
                sm=(partition_mask(bulk,delta,lengths=lengths) if splitk else ref.prefix_mask(bulk,delta,lengths));zm,_=ref.zmask(bulk,z,lengths)
                gm=newmask[b,h,first:last].astype(bool)
                vv=np.broadcast_to(v[b,kh],(n,S,VD))
                sink=None if sinks is None else np.full(n,sinks[h],dtype=np.float32)
                sp[b,h,first:last]=ref.dense_scores(np.where(sm,exact,bulk),vv,lengths,sink)
                old[b,h,first:last]=ref.dense_scores(np.where(zm,exact,bulk),vv,lengths,sink)
                dense[b,h,first:last]=ref.dense_scores(exact,vv,lengths,sink)
                for key,value in [('valid',lengths.sum()),('sp',gm.sum()),('z',zm.sum()),('intersection',(gm&zm).sum()),
                                  ('union',(gm|zm).sum()),('negative_sp',(gm&(bulk<0)).sum()),
                                  ('cpu_gpu_mask_disagreements',(sm!=gm).sum())]:
                    counts[key]+=int(value)
    counts.update(sp_fraction=counts['sp']/counts['valid'],z_fraction=counts['z']/counts['valid'],
                  overlap_recall=counts['intersection']/max(counts['z'],1),jaccard=counts['intersection']/max(counts['union'],1),
                  source='actual GPU SP diagnostic mask; z-score mask estimated by independent fp32 CPU reference')
    counts['matched']=abs(counts['sp_fraction']-counts['z_fraction'])<=0.02
    return sp,old,dense,counts


def shape_gate(tc,shape,index,reg,splitk=False):
    delta=calibration()[geom_key(shape['kind'],shape['S'],shape['D'],shape['causal'])]['delta']
    q,k,kq,v=arrays(shape,np.random.SeedSequence([reg['data']['gpu_seed'],index]))
    tensors=[tc.tensor(x) for x in [q,k,kq,v]]
    scale=1/np.sqrt(shape['D']);z=reg['zthr'];causal=shape['causal']
    new=lambda:tc._C.apa_selective_attention_sp(*tensors,scale,delta,causal)
    old=lambda:tc.apa_selective_attention(*tensors,scale,z,causal)
    with tc.no_grad():
        diagnostic,mask=tc._C.apa_selective_attention_sp(*tensors,scale,delta,causal,diagnostics=True)
        tc.synchronize();got=diagnostic.numpy();gm=mask.numpy()
        del diagnostic,mask
        # Timing path and diagnostics must have exactly the same output bytes.
        plain=new().numpy();baseline=old().numpy();tc.synchronize()
    diag_identical=np.array_equal(plain,got)
    spref,oldref,dense,tail=all_references(q,k,kq,v,shape,delta,z,gm,splitk=splitk)
    del gm
    tol=reg['data']['gpu_fp32_tolerance']
    sp_ok=bool(np.allclose(got,spref,**tol));old_ok=bool(np.allclose(baseline,oldref,**tol))
    masks_ok=not splitk or tail['cpu_gpu_mask_disagreements']==0
    g2=dict(status='PASS' if sp_ok and old_ok and diag_identical and masks_ok else 'FAIL',
            sp_mask_exact=masks_ok,
            sp_vs_own_emulator=metrics(got,spref),baseline_vs_own_emulator=metrics(baseline,oldref),
            diagnostics_bit_identical=diag_identical,tolerance=tol,checked_elements=int(got.size))
    candidate_error=metrics(got,dense);baseline_error=metrics(baseline,dense)
    result=dict(shape=shape,variant='sp1_1_splitk' if splitk else 'sp1_prefill',delta=delta,G2=g2,tail=tail,sp_vs_dense_fp32=candidate_error,
                baseline_vs_dense_fp32=baseline_error,scope_note='this establishes nothing about model quality')
    if g2['status']!='PASS':
        result['G3']={'status':'BLOCKED_BY_G2'};return result
    events=Events();samples={'sp':[],'baseline':[]};wall={'sp':[],'baseline':[]}
    try:
        with tc.no_grad():
            for _ in range(reg['gpu']['warmups']):
                new();old()
            tc.synchronize()
            for rep in range(reg['gpu']['repetitions']):
                order=[('baseline',old),('sp',new)] if rep%2==0 else [('sp',new),('baseline',old)]
                for key,fn in order:
                    gpu_ms,wall_ms=events.measure(fn);samples[key].append(gpu_ms);wall[key].append(wall_ms)
    finally:events.close()
    speed=float(np.median(samples['baseline'])/np.median(samples['sp']))
    denom=baseline_error['relative_frobenius']
    ratio=candidate_error['relative_frobenius']/denom if denom>0 else None
    rows=shape['B']*shape['H']*shape['L'];VD=shape['VD'];S=shape['S']
    parts=(S+2047)//2048 if shape['L']==1 and rows<112 and S>=4096 else 0
    result['memory']={'evidence_class':'code inspection: structural allocation counts; NOT measured peak VRAM',
                      'resident_input_bytes':sum(a.nbytes for a in [q,k,kq,v]),'output_bytes':int(got.nbytes),
                      'sp_transient_global_bytes_excluding_output':4*rows*((S+2047)//2048)*(VD+2) if splitk else 0,
                      'baseline_transient_global_bytes_excluding_output':4*rows*(1+parts*(VD+2)) if parts else 0,
                      'diagnostic_mask_bytes_excluded_from_timing':rows*S,
                      'sp_block_threads':128 if splitk else 32,'sp_shared_bytes':(5*next(c for c in [64,128,256,512] if c>=max(shape['D'],VD))+260)*4 if splitk else 0,'sp_register_storage_order':'O(D+VD) per row; ptxas receipt reports spills',
                      'actual_peak_vram_bytes':None,'kq_residency':'full floating reconstructed/perturbed K, same for both; no KV compression claim'}
    result['G3']=dict(status='COMPLETE_MATCHED' if tail['matched'] else 'RED_UNMATCHED_FRACTION',
                     evidence_class='kernel sweep',cuda_event_ms=samples,wall_ms=wall,
                     medians_ms={key:float(np.median(vals)) for key,vals in samples.items()},
                     iqr_ms={key:float(np.percentile(vals,75)-np.percentile(vals,25)) for key,vals in samples.items()},
                     speedup=speed,deviation_ratio=ratio,
                     P2_overlap_hit=(tail['overlap_recall']>=0.8) if shape['kind']=='prefill' else None,
                     P3_deviation_hit=(ratio<=2) if ratio is not None and tail['matched'] else None,
                     P3_prefill_speed_hit=(speed>=1.4) if shape['kind']=='prefill' and tail['matched'] else None,
                     A3_prefill_overlap_hit=(tail['overlap_recall']<=0.65) if shape['kind']=='prefill' else None,
                     A4_decode_speed_hit=(speed<=1.0) if shape['kind']=='decode' and tail['matched'] else None,
                     scope_note='this establishes nothing about model quality')
    return result


def boundary_gate(tc,reg):
    cases=[]
    for D,VD,L,S in [(33,7,3,11),(64,32,7,29),(128,64,1,2048),(512,17,3,19)]:
        shape=dict(B=1 if D==512 else 2,H=16 if D==512 else 4,
                   KVH=1 if D==512 else 2,L=L,S=S,D=D,VD=VD,causal=True)
        for dtype,tolkey in [('float32','gpu_fp32_tolerance'),('float16','gpu_fp16_tolerance'),('bfloat16','gpu_bf16_tolerance')]:
            aa=arrays(shape,np.random.SeedSequence([reg['data']['gpu_seed'],D,VD]))
            tt=[tc.tensor(a,dtype=dtype) for a in aa]
            # Reference must see the actual input rounding, including BF16 Kq.
            q,k,kq,v=[x.numpy().astype(np.float32) for x in tt]
            sinks=tc.tensor(np.resize(np.array([-10,0,3,10],dtype=np.float32),shape['H']),dtype=dtype)
            with tc.no_grad():
                got=tc._C.apa_selective_attention_sp(*tt,1/np.sqrt(D),0.5,True,sinks).numpy().astype(np.float32)
            want=ref.tensor_reference(q,k,kq,v,1/np.sqrt(D),0.5,True,sinks.numpy().astype(np.float32),'prefix')
            ok=bool(np.allclose(got,want,**reg['data'][tolkey]))
            cases.append(dict(shape=shape,dtype=dtype,pass_gate=ok,metrics=metrics(got,want)))
    # On actual CUDA inputs, verify untouched legacy routing under the new flag.
    os.environ['TC_APA_SP']='0'
    baseline_off=tc.apa_selective_attention_sink(*tt,sinks,1/np.sqrt(D),reg['zthr'],True).numpy()
    os.environ['TC_APA_SP']='1'
    baseline_on=tc.apa_selective_attention_sink(*tt,sinks,1/np.sqrt(D),reg['zthr'],True).numpy()
    flag_preserves_legacy=bool(np.array_equal(baseline_off,baseline_on))
    return dict(status='PASS' if all(x['pass_gate'] for x in cases) and flag_preserves_legacy else 'FAIL',
                cases=cases,legacy_flag_toggle_bit_identical=flag_preserves_legacy)


def save_attempt(name,result):
    dest=ART/'gpu';dest.mkdir(exist_ok=True)
    result.update(registration_sha256=sha(ART/'registration.json'),fingerprint=fingerprint(),
                  evidence_class='kernel sweep' if name not in ['probe','boundary','legacy','selector'] else 'GPU unit/suite gate',
                  device_selection=os.environ.get('CUDA_VISIBLE_DEVICES'),time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    path=dest/f'{name}.{time.time_ns()}.json'
    payload=json.dumps(result,indent=2,allow_nan=False)+'\n'
    temporary=path.with_suffix('.tmp')
    with temporary.open('x') as f:
        f.write(payload);f.flush();os.fsync(f.fileno())
    # Atomic create-only publication: interrupted workers leave no partial
    # final JSON for resume to mistake for a complete receipt.
    os.link(temporary,path)
    temporary.unlink()
    print(path.relative_to(ROOT),flush=True)


def completed(name):
    fp=fingerprint()
    for path in (ART/'gpu').glob(name+'.*.json'):
        a=json.loads(path.read_text())
        if a.get('status')=='COMPLETE' and a.get('fingerprint')==fp:return True
    return False


def summary():
    from apa_sp1_1_scoring import finalize
    finalize()


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['shape','boundary','probe','legacy','selector','next','list','summary','splitk','splitk_boundary','partb','next-splitk','splitk-summary']);p.add_argument('shape',nargs='?');args=p.parse_args()
    reg=registration()
    if args.mode=='list':
        print('\n'.join(s['id'] for s in reg['shapes']));return
    if args.mode=='summary':summary();return
    if args.mode=='splitk-summary':
        from apa_sp1_1_gpu import splitk_summary
        splitk_summary();return
    if args.mode=='next-splitk':
        for name in ['splitk_boundary']+['splitk_'+s['id'] for s in reg['shapes'] if s['kind']=='decode']:
            if not completed(name):print(name);return
        print('DONE');return
    if args.mode=='next':
        for name in ['boundary','legacy','selector']+[s['id'] for s in reg['shapes']]:
            if not completed(name):print(name);return
        print('DONE');return
    assert os.environ.get('APA_SP1_LEASED')=='1','Use scripts/apa_sp1_lead_gpu.sh'
    # A lease marker alone is not permission to use an installed runtime.
    name=('splitk_'+args.shape) if args.mode=='splitk' else (args.shape if args.mode=='shape' else args.mode)
    try:
        tc=load_runtime()
        probe=tc.tensor(np.zeros(1,dtype=np.float32));tc.synchronize();del probe
        if args.mode=='probe':result={'status':'COMPLETE','cuda_probe':'PASS'}
        elif args.mode in ['shape','splitk']:
            ix=next(i for i,s in enumerate(reg['shapes']) if s['id']==args.shape)
            shape=reg['shapes'][ix]
            if args.mode=='splitk':assert shape['kind']=='decode'
            result=shape_gate(tc,shape,ix,reg,splitk=shape['L']==1);result['status']='COMPLETE'
        elif args.mode in ['splitk_boundary','partb']:
            from apa_sp1_1_gpu import splitk_boundary,partb_gate
            result=(splitk_boundary if args.mode=='splitk_boundary' else partb_gate)(tc,reg)
            result['status']='COMPLETE'
        elif args.mode=='boundary':result={'status':'COMPLETE','G2_boundary':boundary_gate(tc,reg)}
        elif args.mode=='legacy':
            files=['test_apa_selective.py','test_apa_selective_splitk.py','test_apa_selective_int4.py','test_apa_value_dim.py',
                   'test_apa_phase6.py','test_qtile_attention.py','test_apa_int4_sdpa_noncausal.py']
            env=dict(os.environ,PYTHONPATH=f'{ART}/build:{ROOT}/tensor_cuda')
            proc=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider']+[str(ROOT/'tensor_cuda/tests'/f) for f in files],
                                env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=480)
            result={'status':'COMPLETE','suite_status':'PASS' if proc.returncode==0 else 'FAIL','returncode':proc.returncode,'output':proc.stdout}
        else:
            # Existing selector is a standalone script, not a pytest module.
            path=ROOT/'tensor_cuda/tests/test_selector_accuracy.py';source=path.read_text()
            needle='sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")'
            assert source.count(needle)==1
            source=source.replace(needle,'sys.path.insert(0, '+repr(str(ROOT/'tensor_cuda'))+')')
            sys.argv=[str(path),'4']
            exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
            result={'status':'COMPLETE','standalone_selector':'PASS','source_sha256':sha(path),
                    'only_transform':'main-checkout import path redirected in memory to dispatched worktree; assertions unchanged'}
        save_attempt(name,result)
        if result.get('G2',{}).get('status')=='FAIL' or result.get('G2_boundary',{}).get('status')=='FAIL' or result.get('suite_status')=='FAIL':sys.exit(1)
    except Exception as e:
        result={'status':'ERROR','error':repr(e),'traceback':traceback.format_exc()}
        save_attempt(name,result);raise


if __name__=='__main__':main()
