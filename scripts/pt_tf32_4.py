#!/usr/bin/env python3
"""PT-TF32-4: registered CPU helpers and lead-only GPU kernel gates.

Prior art: PT-TF32-1/BP-KERNEL-2/4 (2026) harness and CUDA event interleaving,
NumPy (2020), SHA256 (2001), NVIDIA Lt (2024); taken mechanisms. Ours: new
tail registration, immutable historical receipts, and implementation
capability evidence. Numerical prior art is at pt_tf32_4_numerics.py.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import pt_tf32_1 as old
import pt_tf32_4_numerics as num
import numpy as np
import pt_tf32_3_storage as storage

ROOT=old.ROOT
ART=ROOT/'artifacts/pt_tf32_4'
REG=ART/'REGISTRATION.json'
NAMES=old.NAMES
sha,create_json=old.sha,old.create_json
fixture=old.fixture
front,back=old.front,old.back
tiled_backward_model=num.backward_model
GATES=('gemm_fp64','gemm_tensor_core_dispatch','attention_forward_fp64',
       'attention_backward_isolated_fp64','attention_backward_downstream_fp64',
       'attention_backward_same_native_state_fp64',
       'attention_selection','attention_speed','gpu_unit_tests','memcheck','racecheck',
       'synccheck','model_noise_floor','model_onset','model_healthy','model_control',
       'model_step_time','blind_lead_verification')


def registration():
    if sha(REG)!=REG.with_suffix('.sha256').read_text().strip():raise ValueError('registration drift')
    r=json.loads(REG.read_text())
    if sha(ROOT/r['order'])!=r['order_sha256']:raise ValueError('order drift')
    if sha(ROOT/r['prior_registration'])!=r['prior_sha256']:raise ValueError('historical registration drift')
    path=ART/'AMENDMENT_001.json'
    if sha(path)!=path.with_suffix('.sha256').read_text().strip():raise ValueError('amendment drift')
    amendment=json.loads(path.read_text())
    if amendment['registration_sha256']!=sha(REG):raise ValueError('amendment parent drift')
    r['edge_accumulator']=amendment['edge_accumulator']
    return r


def seal():
    registration()
    build=sorted(ART.glob('build_*/receipt.json'))[-1]
    b=json.loads(build.read_text())
    if b['rc'] or not b['sources_unchanged_during_build']:raise ValueError('build failed')
    for p,h in b['source_pins'].items():
        if sha(ROOT/p)!=h:raise ValueError('source changed after build: '+p)
    binaries=list((ROOT/'tensor_cuda/tensor_cuda').glob('_tensor_cuda*.so'))
    if len(binaries)!=1 or sha(binaries[0])!=b['binaries'][0]['sha256']:raise ValueError('binary/build drift')
    paths=list((ROOT/'tensor_cuda/src').glob('*'))+list((ROOT/'tensor_cuda/include/tc').glob('*'))
    paths+=list((ROOT/'scripts').glob('pt_tf32*.py'))+list((ROOT/'scripts').glob('bp_*.py'))
    paths+=list((ROOT/'tests').glob('test_pt_tf32*.py'))+list((ROOT/'tensor_cuda/tensor_cuda').glob('*.py'))
    paths+=[ROOT/'tensor_cuda/CMakeLists.txt',ROOT/'tests/pt_tf32_host_contract.cpp']
    paths+=list(ART.glob('GRAPA_REGISTRATION*.json'))
    paths+=list(ART.glob('AMENDMENT_*.json'))
    paths+=list(ART.glob('AMENDMENT_*.sha256'))
    m=dict(registration_sha256=sha(REG),pins={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths) if p.is_file()},
           binary=dict(path=str(binaries[0].relative_to(ROOT)),sha256=sha(binaries[0])),
           build_receipt=str(build.relative_to(ROOT)),build_receipt_sha256=sha(build))
    create_json(ART/'SOURCE_MANIFEST.json',m)
    with (ART/'SOURCE_MANIFEST.sha256').open('x') as f:f.write(sha(ART/'SOURCE_MANIFEST.json')+'\n')
    print('SEALED',sha(ART/'SOURCE_MANIFEST.json'))


def verify_manifest():
    registration();p=ART/'SOURCE_MANIFEST.json'
    if sha(p)!=p.with_suffix('.sha256').read_text().strip():raise ValueError('manifest drift')
    m=json.loads(p.read_text())
    if m['registration_sha256']!=sha(REG):raise ValueError('manifest registration drift')
    for p,h in m['pins'].items():
        if sha(ROOT/p)!=h:raise ValueError('source drift: '+p)
    if sha(ROOT/m['binary']['path'])!=m['binary']['sha256']:raise ValueError('binary drift')
    if sha(ROOT/m['build_receipt'])!=m['build_receipt_sha256']:raise ValueError('build receipt drift')
    return m


def load_fork(require_manifest=True):
    if require_manifest:verify_manifest()
    return old.load_fork(False)


def dispatch_ok(row,m,n,k,tb,select=True):
    """NVIDIA cuBLASLt (2024) flags/readback; exact local shape binding is ours.

    Flags are algorithm evidence, not profiler counts or a speed guarantee.
    Missing, stale-shape and failed-query receipts cannot satisfy this gate.
    """
    required=dict(M=m,N=n,K=k,batch=1,compute_type=77,fast_tf32=1,
                  transa=int(tb),transb=0,lda=k if tb else n,ldb=k,ldc=n)
    if not row or any(row.get(key)!=v for key,v in required.items()):return False
    if not select:return row.get('device_selected')==0 and row.get('algorithm_id')==-1
    return bool(row.get('device_selected')==1 and row.get('algorithm_id',-1)>=0 and
                row.get('numerical_flags',0)&0x40202==0x40202 and
                row.get('numerical_flags_query_status')==0 and row.get('numerical_flags_bytes')==8 and
                row.get('numerical_flags_attribute')==15 and
                row.get('mathmode_query_attempted')==0 and row.get('mathmode_query_supported')==0 and
                all(row.get('alignment_'+a,0)>=16 for a in ('a','b','c')) and
                0<=row.get('workspace_bytes',-1)<=32*1024*1024)


def gemm_row_ok(distance,readback,m,n,k,tb):
    # NVIDIA TF32/Lt (2020/2024), taken capabilities; accuracy <=1e-3 is
    # inherited. Speed is deliberately absent from the registered predicate.
    return bool(distance.get('finite') and np.isfinite(distance.get('relative_L2',np.nan)) and
                0<=distance['relative_L2']<=registration()['gemm_gate']['relative_L2_max'] and
                dispatch_ok(readback,m,n,k,tb))


def gemm(c,emit):
    r=registration();timer=old.Events(c);green=True;completed=0
    if os.environ.get('NVIDIA_TF32_OVERRIDE')=='0':raise RuntimeError('TF32 override disables required path')
    # Prior art: PT-TF32-1/3 interleaved CUDA-event/FP64 reference harness
    # (2026), NVIDIA stream-ordered allocation (2021), taken unchanged arms.
    # Ours: capture dispatch immediately after this shape's actual fast
    # matmul, before timing changes the last-call state, and gate its flags.
    try:
        for pooled in (False,True):
            c.set_alloc_pooling(pooled);rng=np.random.default_rng(r['seed'])
            for s in r['gemm_shapes']:
                M,K,N=s['M'],s['K'],s['N']
                a=rng.standard_normal((M,K)).astype(np.float32)*.25
                w=rng.standard_normal((N,K)).astype(np.float32)*.25
                g=rng.standard_normal((M,N)).astype(np.float32)*.25
                for direction,aa,bb,tb in [('forward',a,w,True),('dInput',g,w,False),('dWeight',g.T,a,False)]:
                    aa=np.ascontiguousarray(aa);bb=np.ascontiguousarray(bb)
                    m,k=aa.shape;n=bb.shape[0] if tb else bb.shape[1]
                    ref=old.gemm_reference(aa,bb,tb)
                    A=c.tensor(aa,'cuda',False);B=c.tensor(bb,'cuda',False)
                    def run(enabled):
                        c.set_tf32_gemm(enabled);return c.matmul(A,B,1.,tb)
                    standard=run(False).numpy();fast=run(True).numpy()
                    readback=c.get_tf32_gemm_dispatch();info=c.get_tf32_gemm_info()
                    err=back.metric(fast,ref);baseline=back.metric(standard,ref)
                    good=gemm_row_ok(err,readback,m,n,k,tb)
                    good &= tuple(info)==(readback.get('algorithm_id'),readback.get('numerical_flags'),readback.get('workspace_bytes'))
                    good &= readback.get('output_stream_ordered')==1
                    times=old.interleaved(timer,{'sgemm':lambda:run(False),'tf32':lambda:run(True)},
                                          warmups=r['gemm_gate']['warmups'],samples=r['gemm_gate']['samples'])
                    speedup=times['median_ms']['sgemm']/times['median_ms']['tf32']
                    # Prior art: Williams, Waterman & Patterson, Roofline
                    # (2009). Taken FLOP/byte ceiling reasoning; ours: report
                    # compulsory FP32 traffic only, not measured DRAM traffic.
                    flops=2*m*n*k
                    emit(dict(kind='gemm',shape=s,direction=direction,sgemm_distance=baseline,tf32_distance=err,
                        implementation=dict(algorithm_id=info[0],numerical_flags=info[1],workspace_bytes=info[2],dispatch=readback,
                            tf32_hmma=dispatch_ok(readback,m,n,k,tb),evidence='actual Lt algorithm capability; not profiler counters'),
                        timing=times,speedup=speedup,speedup_is_diagnostic=True,timing_counts=bool(err['finite']),
                        sgemm_tflops=flops/(times['median_ms']['sgemm']*1e9),tf32_tflops=flops/(times['median_ms']['tf32']*1e9),
                        ideal_arithmetic_intensity=flops/(4*(m*k+k*n+m*n)),
                        allocation='model_pool_both' if pooled else 'global_pool_off_tf32_output_pool_on',
                        sgemm_output_stream_ordered=pooled,verdict='GREEN' if good else 'RED'))
                    completed+=1;green &= good
    finally:c.set_tf32_gemm(False);c.set_alloc_pooling(False)
    return bool(green and completed==2*len(r['gemm_shapes'])*3)


def dispatch(c,emit,select):
    good=True
    for s in registration()['gemm_shapes']:
        M,K,N=s['M'],s['K'],s['N']
        for direction,m,n,k,tb in [('forward',M,N,K,True),('dInput',M,K,N,False),('dWeight',N,K,M,False)]:
            row=c.tf32_gemm_self_check(m,n,k,tb,select)
            ok=dispatch_ok(row,m,n,k,tb,select)
            good &= ok
            record=dict(kind='dispatch',shape=s,direction=direction,readback=row,
                verdict='GREEN' if ok else 'RED',evidence_class='host descriptor/heuristic API; no matmul, profiler or speed claim')
            emit(record)
            print('DISPATCH',direction,json.dumps(row,sort_keys=True),flush=True)
    return good


def attention(c,emit,case):
    r=registration();s=old.shape_values(r['attention_shapes'][case]);x=fixture(s,r['seed'])
    d={n:c.tensor(np.ascontiguousarray(v),'cuda',False) for n,v in x.items()}
    bf={n:v.astype('bfloat16') for n,v in d.items()}
    def fwd(variant,diag=False):
        a=bf if variant=='h' else d
        fn=c.bp_kernel_4_diagnostic if diag else c.apa_selective_fwd_train_variant
        return fn(*[a[n] for n in ('q','k','kq','v')],s['scale'],s['zthr'],s['causal'],variant)
    def bwd(variant,state):
        a=bf if variant=='g1' else d
        return c.apa_selective_bwd_variant(*[a[n] for n in ('q','k','kq','v','dO')],
                    state[1],state[2],state[0],s['scale'],s['causal'],variant)
    f=fwd('h_tf32',True);a=fwd('a',True);b=fwd('h')
    native=dict(zip(('out','lse','thr','selection'),[v.numpy() for v in f[:4]]))
    native['selection']=native['selection'].astype(bool)
    ref=num.forward_reference(x,s)
    def gate(values,reference,names):
        return {n:num.precision_gate(values[n],reference[n],r,s,n) for n in names}
    fg=gate(native,ref,('out','lse'))
    frozen={n:ref[n].astype(np.float32) for n in ('out','lse','thr')}
    fd=tuple(c.tensor(np.ascontiguousarray(frozen[n]),'cuda',False) for n in ('out','lse','thr'))
    iso=dict(zip(NAMES,[v.numpy() for v in bwd('g1_tf32',fd)]))
    isolated=gate(iso,num.backward_reference(x,frozen,s),NAMES)
    down=dict(zip(NAMES,[v.numpy() for v in bwd('g1_tf32',f)]))
    downstream=gate(down,num.backward_reference(x,ref,s),NAMES)
    saved=gate(down,num.backward_reference(x,native,s),NAMES)
    visible=np.arange(s['S'])[None,:] <= s['S']-s['L']+np.arange(s['L'])[:,None] if s['causal'] else np.ones((s['L'],s['S']),bool)
    flips=front.flips(ref['selection'],native['selection'],visible)
    numerical=flips['passes'] and all(v['verdict']=='GREEN' for arm in (fg,isolated,downstream,saved) for v in arm.values())
    timer=old.Events(c)
    ft=old.interleaved(timer,{'a_fp32':lambda:fwd('a'),'h_tf32':lambda:fwd('h_tf32'),'h_bf16':lambda:fwd('h')})
    bt=old.interleaved(timer,{'a_fp32':lambda:bwd('a',a),'g1_tf32':lambda:bwd('g1_tf32',f),'g1_bf16':lambda:bwd('g1',b)})
    fr=ft['median_ms']['h_tf32']/ft['median_ms']['h_bf16'];br=bt['median_ms']['g1_tf32']/bt['median_ms']['g1_bf16']
    good=numerical and max(fr,br)<=r['attention_gate']['max_time_vs_bf16']
    emit(dict(kind='attention',case=case,shape=s,forward=fg,isolated=isolated,downstream=downstream,
              same_native_state=saved,selection=flips,forward_timing=ft,backward_timing=bt,
              forward_vs_bf16=fr,backward_vs_bf16=br,timing_counts=bool(numerical),verdict='GREEN' if good else 'RED'))
    return bool(good)


def blocked():
    registration();manifest=verify_manifest()
    r=dict(verdict='BLOCKED',evidence_class='authorization boundary, no GPU measurements',
           reason="PT-TF32-4 seat is CPU/build only; CUDA_VISIBLE_DEVICES=''. Lead owns GPU gates.",
           registration_sha256=sha(REG),gates={k:dict(status='BLOCKED',measured=False) for k in GATES},
           amendment_sha256=sha(ART/'AMENDMENT_001.json'),manifest_sha256=sha(ART/'SOURCE_MANIFEST.json'),
           binary=manifest['binary'],historical_reassessment='HISTORICAL_REASSESSMENT.json',
           not_claimed_fixed=['native edge passes on rebuilt binary','new tail/dispatch GPU certification','training stability'])
    create_json(ART/'BLOCKED_REPORT.json',r);print(json.dumps(r,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['seal','blocked','gemm','attention','dispatch','host'])
    p.add_argument('--lead-gpu',action='store_true');p.add_argument('--case',type=int,choices=range(4),default=0)
    p.add_argument('--out',type=Path);args=p.parse_args()
    if args.command=='seal':seal();return 0
    if args.command=='blocked':blocked();return 0
    if args.command=='host':
        print('CPU descriptors only; selected algorithm BLOCKED without a lead GPU slot')
        return 0 if dispatch(load_fork(False),lambda row:None,False) else 1
    if not args.lead_gpu or not os.environ.get('CUDA_VISIBLE_DEVICES'):p.error('BLOCKED: lead-only GPU gate')
    if not args.out or not args.out.resolve().is_relative_to(ART):p.error('--out must be a fresh PT-TF32-4 artifact directory')
    m=verify_manifest();args.out.mkdir(parents=True,exist_ok=False)
    check=storage.space_status(args.out)
    if check['status']!='GREEN':
        create_json(args.out/'summary.json',dict(check,verdict='BLOCKED'));print(json.dumps(check));return 2
    rows=[];start=time.monotonic();error=None;good=False
    def emit(row):
        create_json(args.out/f'case_{len(rows):02d}.json',row);rows.append(row)
        print(row['kind'],row.get('direction',''),row['verdict'],flush=True)
    try:
        c=load_fork();good=gemm(c,emit) if args.command=='gemm' else dispatch(c,emit,True) if args.command=='dispatch' else attention(c,emit,args.case)
    except Exception as exc:
        error=repr(exc);print(error,file=sys.stderr)
    result=dict(verdict='GREEN' if good else 'RED',error=error,completed_cases=len(rows),
                registration_sha256=sha(REG),manifest_sha256=sha(ART/'SOURCE_MANIFEST.json'),binary=m['binary'],
                command=args.command,case=args.case,elapsed_seconds=time.monotonic()-start,
                evidence_class='GPU kernel gate; no model quality claim')
    create_json(args.out/'summary.json',result);print(json.dumps(result,indent=2));return 0 if good else 1


if __name__=='__main__':sys.exit(main())
