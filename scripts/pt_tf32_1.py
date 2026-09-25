#!/usr/bin/env python3
"""PT-TF32-1 registered lead-only gates and CPU arithmetic tools.

Prior art: NVIDIA Ampere TF32 / CUDA 11 (2020), FA-2 (Dao 2023), and
Project-Tensor BP-KERNEL-2/3/4 (2026), taken formats, references, VJP,
selection and 2x-spread gates. NumPy / Harris et al. (2020), CUDA events
(NVIDIA 2007+) and SHA256 (NIST 2001), taken. Ours: fork-only loading,
FP32 fixtures with MLA split keys, and this order's registered comparisons.
See docs/PT_TF32_1_LEDGER.md for verified primary references and attribution.
No GPU discovery/import/allocation occurs in the blocked or CPU helper paths.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

sys.dont_write_bytecode = True
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
import numpy as np
import bp_kernel_2 as back
import bp_kernel_4 as front

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/pt_tf32_1'
REG = ART / 'REGISTRATION.json'
NAMES = ('dQ', 'dK', 'dV')
GATES = ('gemm_fp64', 'gemm_speed', 'attention_forward_fp64',
         'attention_backward_fp64', 'attention_selection', 'attention_speed',
         'gpu_unit_tests', 'cuda_sanitizers', 'model_onset', 'model_healthy_control',
         'model_step_time', 'blind_lead_verification')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def create_json(path, data):
    # Create-only receipts (BP-KERNEL-2); JSON rejects NaN instead of claiming a pass.
    with Path(path).open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')


def registration():
    if sha(REG) != (ART / 'REGISTRATION.sha256').read_text().strip():
        raise ValueError('registration drift')
    r = json.loads(REG.read_text())
    if sha(ROOT / r['order']) != r['order_sha256']:
        raise ValueError('immutable order drift')
    return r


def seal():
    """Pin delivered source and the exact fork binary after author CPU work."""
    registration()
    paths = list((ROOT / 'tensor_cuda/src').glob('*'))
    paths += list((ROOT / 'tensor_cuda/include/tc').glob('*'))
    paths += list((ROOT / 'scripts').glob('pt_tf32*.py'))
    paths += list((ROOT / 'tests').glob('test_pt_tf32*.py'))
    paths += [ROOT / 'tensor_cuda/CMakeLists.txt', ROOT / 'tests/pt_tf32_host_contract.cpp']
    paths += list(ART.glob('AMENDMENT_*.json'))
    paths += [ART/'GRAPA_REGISTRATION.json', ART/'GRAPA_REGISTRATION.sha256']
    # Pin every transitive local reference module used by the imported harness.
    paths += list((ROOT / 'scripts').glob('bp_*.py'))
    pins = {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths) if p.is_file()}
    binaries = list((ROOT / 'tensor_cuda/tensor_cuda').glob('_tensor_cuda*.so'))
    if len(binaries) != 1:
        raise ValueError('exactly one built fork extension required')
    build = sorted(ART.glob('build_*/receipt.json'))[-1]
    b = json.loads(build.read_text())
    if b['rc'] or not b['sources_unchanged_during_build']:
        raise ValueError('last build failed')
    for p, h in b['source_pins'].items():
        if sha(ROOT / p) != h:
            raise ValueError('source changed after build: ' + p)
    binary = dict(path=str(binaries[0].relative_to(ROOT)), sha256=sha(binaries[0]))
    if binary['sha256'] != b['binaries'][0]['sha256']:
        raise ValueError('binary changed after build')
    manifest = dict(registration_sha256=sha(REG), pins=pins, binary=binary,
                    build_receipt=str(build.relative_to(ROOT)), build_receipt_sha256=sha(build))
    create_json(ART / 'SOURCE_MANIFEST.json', manifest)
    with (ART / 'SOURCE_MANIFEST.sha256').open('x') as f:
        f.write(sha(ART / 'SOURCE_MANIFEST.json') + '\n')
    print('SEALED', ART / 'SOURCE_MANIFEST.json', sha(ART / 'SOURCE_MANIFEST.json'))


def verify_manifest():
    registration()
    path = ART / 'SOURCE_MANIFEST.json'
    if sha(path) != (ART / 'SOURCE_MANIFEST.sha256').read_text().strip():
        raise ValueError('manifest drift')
    m = json.loads(path.read_text())
    if m['registration_sha256'] != sha(REG):
        raise ValueError('manifest registration drift')
    for p, h in m['pins'].items():
        if sha(ROOT / p) != h:
            raise ValueError('source drift: ' + p)
    if sha(ROOT / m['binary']['path']) != m['binary']['sha256']:
        raise ValueError('fork binary drift')
    return m


def load_fork(require_manifest=True):
    """Absolute extension loading prevents any installed/live engine fallback."""
    if require_manifest:
        binary = ROOT / verify_manifest()['binary']['path']
    else:  # CPU contract tests may precede final source sealing.
        binaries = list((ROOT / 'tensor_cuda/tensor_cuda').glob('_tensor_cuda*.so'))
        if len(binaries) != 1:
            raise ValueError('one local binary required')
        binary = binaries[0]
    if not binary.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('extension is outside the authorized fork')
    name = '_tensor_cuda'
    existing=None
    for known in (name,'tensor_cuda._tensor_cuda'):
        if known in sys.modules:
            c = sys.modules[known]
            if Path(c.__file__).resolve() != binary.resolve():
                raise ValueError('a different engine is already imported')
            existing=c
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, binary)
    c = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c)
    sys.modules[name] = c
    return c


def tf32_round(value):
    """CPU model of CUDA 12.6 mma.h cvt.rna.tf32.f32 (ties away).

    Taken: NVIDIA TF32 format and intrinsic rounding, not a new quantizer.
    This models conversion only, not GPU MMA accumulator order/subnormals.
    """
    x = np.asarray(value, dtype=np.float32)
    bits = x.view(np.uint32)
    finite = (bits & np.uint32(0x7f800000)) != np.uint32(0x7f800000)
    rounded = (bits + np.uint32(0x1000)) & np.uint32(0xffffe000)
    return np.where(finite, rounded, bits).view(np.float32)


def mma_model(a, b, rounded=True):
    """Independent dense product of modeled operands (CPU, not native parity)."""
    convert = tf32_round if rounded else np.asarray
    return convert(a).astype(np.float64) @ convert(b).astype(np.float64)


def fixture(shape, seed=250925):
    """Synthetic FP32 MLA-shaped inputs; exact RoPE suffix in KQ.

    Taken: MLA [non-RoPE | RoPE] layout in GRAPA (2026). Ours: reproducible
    normal inputs and quarter-step surrogate KQ; NOT the model quantizer.
    """
    B, H, KH, L, S, D, VD = [shape[n] for n in ('B', 'H', 'KVH', 'L', 'S', 'D', 'VD')]
    if min(B, H, KH, L, S, D, VD) <= 0 or H % KH or S < L or max(D, VD) > 128:
        raise ValueError('invalid fixture geometry')
    rng = np.random.default_rng(seed)
    shapes = dict(q=(B,H,L,D), k=(B,KH,S,D), v=(B,KH,S,VD), dO=(B,H,L,VD))
    x = {n: (rng.standard_normal(s) * .5).astype(np.float32) for n, s in shapes.items()}
    x['kq'] = x['k'].copy()
    rope = shape.get('rope_width', 0)
    if not 0 <= rope <= D:
        raise ValueError('invalid RoPE split')
    x['kq'][..., :D-rope] = np.rint(x['k'][..., :D-rope]*4) / 4
    return x


def shape_values(shape):
    return dict(shape, scale=float(np.float32(1 / math.sqrt(shape['D']))),
                zthr=float(np.float32(statistics.NormalDist().inv_cdf(1-shape['refine']))))


def tiled_forward_model(x, s, *, rounded=True):
    """Author CPU model of h_tf32's two passes, including 16-key P@V tiles.

    Taken: BP-KERNEL-4 / Dao online softmax. Ours: TF32 operand model.
    FP64 accumulator arithmetic deliberately separates traversal from hardware
    rounding; passing this model cannot certify CUDA numerical behavior.
    """
    B,H,KH,L,S,D,VD = [s[n] for n in ('B','H','KVH','L','S','D','VD')]
    out = np.zeros((B,H,L,VD)); lse = np.zeros((B,H,L)); thr = np.zeros_like(lse)
    selection = np.zeros((B,H,L,S), dtype=bool)
    scale,zthr = float(np.float32(s['scale'])),float(np.float32(s['zthr']))
    for b in range(B):
        for h in range(H):
            kh = h // (H//KH)
            for qi in range(0,L,16):
                q = np.asarray(x['q'][b,h,qi:qi+16],np.float64)
                count = (S-L+np.arange(qi,qi+len(q))+1) if s['causal'] else np.full(len(q),S)
                ab = np.zeros((len(q),S))
                for kj in range(0,S,16):
                    score = mma_model(q,x['kq'][b,kh,kj:kj+16].T,rounded)*scale
                    ab[:,kj:kj+16] = np.where(np.arange(kj,kj+score.shape[1])[None,:]<count[:,None],abs(score),0)
                mean=ab.sum(1)/count
                th=mean+zthr*np.sqrt(np.maximum((ab*ab).sum(1)/count-mean*mean,0))
                thr[b,h,qi:qi+len(q)]=th
                acc=np.zeros((len(q),VD)); den=np.zeros(len(q)); m=np.full(len(q),-1e30)
                for kj in range(0,int(count.max()),16):
                    key=np.asarray(x['k'][b,kh,kj:kj+16],np.float64)
                    bulk=mma_model(q,x['kq'][b,kh,kj:kj+16].T,rounded)*scale
                    visible=np.arange(kj,kj+len(key))[None,:]<count[:,None]
                    sel=visible & (abs(bulk)>=th[:,None])
                    selection[b,h,qi:qi+len(q),kj:kj+len(key)]=sel
                    score=np.where(visible,np.where(sel,q@key.T*scale,bulk),-1e30)
                    nm=np.maximum(m,score.max(1)); corr=np.exp(m-nm)
                    p=np.where(visible,np.exp(score-nm[:,None]),0)
                    acc=acc*corr[:,None]+mma_model(p,x['v'][b,kh,kj:kj+16],rounded)
                    den=den*corr+p.sum(1);m=nm
                out[b,h,qi:qi+len(q)]=acc/den[:,None]
                lse[b,h,qi:qi+len(q)]=m+np.log(den)
    return dict(out=out,lse=lse,thr=thr,selection=selection)


def tiled_backward_model(x, state, s, *, rounded=True):
    """CPU author model of both g1_tf32 owners; FA-2/BP-KERNEL-3 taken.

    Ours: TF32 conversion at products only. This tests traversal/padding and
    precision placement, not CUDA's reduction order or its native accuracy.
    """
    B,H,KH,L,S,D,VD=[s[n] for n in ('B','H','KVH','L','S','D','VD')]
    result={n:np.zeros_like(x[v],dtype=np.float64) for n,v in zip(NAMES,('q','k','v'))}
    scale=float(np.float32(s['scale']))
    def pair(b,h,kh,qi,kj):
        q=np.asarray(x['q'][b,h,qi:qi+16],np.float64)
        k=np.asarray(x['k'][b,kh,kj:kj+16],np.float64)
        kq=np.asarray(x['kq'][b,kh,kj:kj+16],np.float64)
        do=np.asarray(x['dO'][b,h,qi:qi+16],np.float64)
        v=np.asarray(x['v'][b,kh,kj:kj+16],np.float64)
        bulk=mma_model(q,kq.T,rounded)*scale
        visible=(kj+np.arange(len(k))[None,:] <= S-L+qi+np.arange(len(q))[:,None]) if s['causal'] else np.ones(bulk.shape,bool)
        selected=visible & (abs(bulk)>=state['thr'][b,h,qi:qi+len(q),None])
        score=np.where(selected,q@k.T*scale,bulk)
        p=np.exp(np.where(visible,score-state['lse'][b,h,qi:qi+len(q),None],-np.inf))
        rd=(do*state['out'][b,h,qi:qi+len(q)]).sum(1)
        ds=p*(mma_model(do,v.T,rounded)-rd[:,None])*scale
        return q,k,kq,do,np.where(selected,ds,0),np.where(selected,0,ds),p
    for key in (False,True):
        for b in range(B):
            for oh in range(KH if key else H):
                kh=oh if key else oh//(H//KH)
                for owner in range(0,S if key else L,16):
                    heads=range(kh*(H//KH),(kh+1)*(H//KH)) if key else (oh,)
                    for h in heads:
                        first=max(0,owner-(S-L))//16*16 if key and s['causal'] else 0
                        limit=L if key else min(S,S-L+owner+16) if s['causal'] else S
                        for other in range(first,limit,16):
                            qi,kj=(other,owner) if key else (owner,other)
                            q,k,kq,do,sel,bulk,p=pair(b,h,kh,qi,kj)
                            if key:
                                result['dK'][b,kh,kj:kj+len(k)]+=mma_model(sel.T,q,rounded)
                                result['dV'][b,kh,kj:kj+len(k)]+=mma_model(p.T,do,rounded)
                            else:
                                result['dQ'][b,h,qi:qi+len(q)]+=mma_model(sel,k,rounded)+mma_model(bulk,kq,rounded)
    return result


class Events:
    """Taken: BP-KERNEL-1 CUDA-event lifecycle; includes native output allocation."""
    def __init__(self, c):
        self.c=c
        self.rt=ctypes.CDLL('/usr/local/cuda-12.6/lib64/libcudart.so')
        for name,args in [('cudaEventCreate',[ctypes.POINTER(ctypes.c_void_p)]),
                          ('cudaEventRecord',[ctypes.c_void_p,ctypes.c_void_p]),
                          ('cudaEventSynchronize',[ctypes.c_void_p]),
                          ('cudaEventElapsedTime',[ctypes.POINTER(ctypes.c_float),ctypes.c_void_p,ctypes.c_void_p]),
                          ('cudaEventDestroy',[ctypes.c_void_p])]:
            f=getattr(self.rt,name);f.argtypes=args;f.restype=ctypes.c_int

    def call(self,name,*args):
        rc=getattr(self.rt,name)(*args)
        if rc:
            raise RuntimeError(f'{name}: CUDA rc={rc}')

    def timed(self,fn):
        a,b=ctypes.c_void_p(),ctypes.c_void_p()
        self.call('cudaEventCreate',ctypes.byref(a))
        try:
            self.call('cudaEventCreate',ctypes.byref(b))
            try:
                self.c.synchronize(); self.call('cudaEventRecord',a,None)
                result=fn()
                self.call('cudaEventRecord',b,None);self.call('cudaEventSynchronize',b)
                ms=ctypes.c_float();self.call('cudaEventElapsedTime',ctypes.byref(ms),a,b)
                del result
                return ms.value
            finally:
                self.call('cudaEventDestroy',b)
        finally:
            self.call('cudaEventDestroy',a)


def interleaved(timer, arms, warmups=3, samples=10):
    results={n:[] for n in arms};order=[]
    for iteration in range(warmups+samples):
        # Alternate order to reduce one-way clock/thermal bias; same session.
        names=list(arms) if iteration%2==0 else list(reversed(arms))
        for name in names:
            ms=timer.timed(arms[name])
            if not math.isfinite(ms) or ms<=0:
                raise ValueError('invalid event time')
            if iteration>=warmups:
                results[name].append(ms);order.append(name)
    return dict(samples_ms=results,median_ms={n:float(np.median(v)) for n,v in results.items()},
                launch_order=order,warmups=warmups,samples=samples)


def gemm_reference(a,b,trans_b=False):
    b64=np.asarray(b,np.float64);b64=b64.T if trans_b else b64
    out=np.empty((a.shape[0],b64.shape[1]),np.float64)
    for lo in range(0,len(a),128):
        out[lo:lo+128]=np.asarray(a[lo:lo+128],np.float64)@b64
    return out


def gemm(c, emit):
    r=registration();timer=Events(c);all_green=True
    if os.environ.get('NVIDIA_TF32_OVERRIDE')=='0':
        raise RuntimeError('NVIDIA_TF32_OVERRIDE=0 disables cuBLAS TF32')
    rng=np.random.default_rng(r['seed'])
    for s in r['gemm_shapes']:
        M,K,N=s['M'],s['K'],s['N']
        a=rng.standard_normal((M,K)).astype(np.float32)*.25
        w=rng.standard_normal((N,K)).astype(np.float32)*.25
        g=rng.standard_normal((M,N)).astype(np.float32)*.25
        for direction,aa,bb,tb in [('forward',a,w,True),('dInput',g,w,False),('dWeight',g.T,a,False)]:
            aa=np.ascontiguousarray(aa);bb=np.ascontiguousarray(bb)
            ref=gemm_reference(aa,bb,tb)
            A=c.tensor(aa,'cuda',False);B=c.tensor(bb,'cuda',False)
            def run(enabled):
                c.set_tf32_gemm(enabled)
                return c.matmul(A,B,1.,tb)
            standard=run(False).numpy();fast=run(True).numpy()
            err=back.metric(fast,ref);baseline=back.metric(standard,ref)
            times=interleaved(timer,{'sgemm':lambda:run(False),'tf32':lambda:run(True)})
            speedup=times['median_ms']['sgemm']/times['median_ms']['tf32']
            correct=err['finite'] and err['relative_L2']<=r['gemm_gate']['relative_L2_max']
            green=bool(correct and speedup>=r['gemm_gate']['min_speedup'])
            all_green &= green
            emit(dict(kind='gemm',shape=s,direction=direction,sgemm_distance=baseline,
                      tf32_distance=err,timing=times,speedup=speedup,
                      timing_counts=bool(correct),verdict='GREEN' if green else 'RED'))
    c.set_tf32_gemm(False)
    return all_green


def attention(c, emit, case_index=None):
    r=registration();timer=Events(c);all_green=True
    specs=r['attention_shapes']
    if case_index is not None:
        specs=[specs[case_index]]
    for spec in specs:
        s=shape_values(spec);x=fixture(s,r['seed'])
        device={n:c.tensor(np.ascontiguousarray(v),'cuda',False) for n,v in x.items()}
        bf={n:v.astype('bfloat16') for n,v in device.items()}
        def fwd(variant,diagnostic=False):
            d=bf if variant=='h' else device
            fn=c.bp_kernel_4_diagnostic if diagnostic else c.apa_selective_fwd_train_variant
            return fn(*[d[n] for n in ('q','k','kq','v')],s['scale'],s['zthr'],s['causal'],variant)
        def bwd(variant,state):
            d=bf if variant=='g1' else device
            out,lse,thr=state[:3]
            return c.apa_selective_bwd_variant(*[d[n] for n in ('q','k','kq','v','dO')],
                                               lse,thr,out,s['scale'],s['causal'],variant)
        def host_state(result):
            vals=[t.numpy() for t in result]
            return dict(out=vals[0],lse=vals[1],thr=vals[2],selection=vals[3].astype(bool))
        a=fwd('a',True);t=fwd('h_tf32',True);b=fwd('h')
        ah,th=host_state(a),host_state(t)
        ref=front.reference(x,s)
        visible=(np.arange(s['S'])[None,:] <= s['S']-s['L']+np.arange(s['L'])[:,None])
        fg=front.forward_gate(ah,th,ref,visible)
        # Frozen a state, no renormalization, FP64 VJP (same BP-KERNEL-2 rule).
        br=back.reference(x,ah,s)
        aa=tuple(v.numpy() for v in bwd('a',a))
        isolated=tuple(v.numpy() for v in bwd('g1_tf32',a))
        downstream=tuple(v.numpy() for v in bwd('g1_tf32',t))
        bg=back.gate(aa,tuple(br[n] for n in NAMES),dict(isolated=isolated,downstream=downstream))
        numerical=fg['verdict']=='GREEN' and all(v['verdict']=='GREEN' for v in bg['candidates'].values())
        # Retain timing even on RED only as explicitly non-counting diagnostics.
        ft=interleaved(timer,{'a_fp32':lambda:fwd('a'),'h_tf32':lambda:fwd('h_tf32'),'h_bf16':lambda:fwd('h')})
        bt=interleaved(timer,{'a_fp32':lambda:bwd('a',a),'f_fp32':lambda:bwd('f',a),
                              'g1_tf32':lambda:bwd('g1_tf32',t),'g1_bf16':lambda:bwd('g1',b)})
        fr=ft['median_ms']['h_tf32']/ft['median_ms']['h_bf16']
        brr=bt['median_ms']['g1_tf32']/bt['median_ms']['g1_bf16']
        green=bool(numerical and max(fr,brr)<=r['attention_gate']['max_time_vs_bf16'])
        all_green &= green
        emit(dict(kind='attention',shape=s,forward_gate=fg,backward_gate=bg,
                  forward_timing=ft,backward_timing=bt,forward_vs_bf16=fr,backward_vs_bf16=brr,
                  timing_counts=bool(numerical),verdict='GREEN' if green else 'RED'))
    return all_green


def blocked():
    registration()
    report=dict(evidence_class='authorization boundary; no GPU measurement',
                registration_sha256=sha(REG),verdict='BLOCKED',
                reason='Seat has no GPU authorization; live training owns device. Lead must run in its authorized idle slot.',
                gates={name:dict(status='BLOCKED',measured=False) for name in GATES},
                not_claimed_fixed=['onset gradient precision','6.5 s/step','kernel correctness/speed'])
    create_json(ART/'BLOCKED_REPORT.json',report)
    print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['gemm','attention','blocked','seal'])
    p.add_argument('--lead-gpu',action='store_true',help='Lead invokes only during its authorized idle GPU slot')
    p.add_argument('--case',type=int,choices=range(4),help='One registered attention geometry per bounded invocation')
    p.add_argument('--out',type=Path,help='New receipt directory under this fork')
    args=p.parse_args()
    if args.command=='blocked':
        blocked();return 0
    if args.command=='seal':
        seal();return 0
    if not args.lead_gpu or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        p.error('BLOCKED: lead-only GPU command; CUDA_VISIBLE_DEVICES must be explicitly nonempty')
    if not args.out or not args.out.resolve().is_relative_to(ROOT.resolve()):
        p.error('--out must be a new directory under this fork')
    manifest=verify_manifest()
    args.out.mkdir(exist_ok=False,parents=True)
    start=time.monotonic();rows=[];error=None;ok=False
    def emit(row):
        create_json(args.out/f'case_{len(rows):02d}.json',row)
        rows.append(row)
        print(row['kind'],row.get('direction',''),row['shape'],row['verdict'],flush=True)
    try:
        c=load_fork()
        ok=gemm(c,emit) if args.command=='gemm' else attention(c,emit,args.case)
    except Exception as exc:
        error=repr(exc)
        print(error,file=sys.stderr)
    summary=dict(evidence_class='GPU micro-benchmark; does not establish model safety',
                 registration_sha256=sha(REG),manifest_sha256=sha(ART/'SOURCE_MANIFEST.json'),
                 binary=manifest['binary'],command=args.command,case=args.case,
                 argv=sys.argv,cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
                 nvidia_tf32_override=os.environ.get('NVIDIA_TF32_OVERRIDE'),
                 elapsed_seconds=time.monotonic()-start,completed_cases=len(rows),error=error,
                 verdict='GREEN' if ok else ('INCONCLUSIVE' if error else 'RED'))
    create_json(args.out/'summary.json',summary)
    print(json.dumps(summary,indent=2))
    return 0 if ok else 1


if __name__=='__main__':
    sys.exit(main())
