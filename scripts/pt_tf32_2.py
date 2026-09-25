#!/usr/bin/env python3
"""PT-TF32-2: registered CPU helpers and lead-only GPU kernel gates.

Prior art: PT-TF32-1/BP-KERNEL-2/4 (2026) harness and CUDA event interleaving,
NumPy (2020), SHA256 (2001), NVIDIA Lt (2024); taken mechanisms. Ours: new
precision registration, immutable historical receipts, and implementation
capability evidence. Numerical prior art is at pt_tf32_2_numerics.py.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import pt_tf32_1 as old
import pt_tf32_2_numerics as num
import numpy as np

ROOT=old.ROOT
ART=ROOT/'artifacts/pt_tf32_2'
REG=ART/'REGISTRATION.json'
NAMES=old.NAMES
sha,create_json=old.sha,old.create_json
fixture=old.fixture
front,back=old.front,old.back
tiled_backward_model=num.backward_model
GATES=('gemm_fp64','gemm_tensor_core_dispatch','gemm_speed','attention_forward_fp64',
       'attention_backward_isolated_fp64','attention_backward_downstream_fp64',
       'attention_selection','attention_speed','gpu_unit_tests','memcheck','racecheck',
       'synccheck','model_noise_floor','model_onset','model_healthy','model_control',
       'model_step_time','blind_lead_verification')


def registration():
    if sha(REG)!=REG.with_suffix('.sha256').read_text().strip():raise ValueError('registration drift')
    r=json.loads(REG.read_text())
    if sha(ROOT/r['order'])!=r['order_sha256']:raise ValueError('order drift')
    if sha(ROOT/r['prior_registration'])!=r['prior_sha256']:raise ValueError('historical registration drift')
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


def gemm(c,emit):
    flags_required=0x40202 # NVIDIA: HMMA | accumulator FP32 | input TF32
    green=True
    def record(row):
        nonlocal green
        algo,flags,workspace=c.get_tf32_gemm_info()
        good=algo>=0 and flags & flags_required == flags_required
        row['implementation']=dict(algorithm_id=algo,numerical_flags=flags,workspace_bytes=workspace,
                                   tf32_hmma=good,evidence='cuBLASLt runtime algorithm capability; not profiler counters')
        if not good:row['verdict']='RED'
        green &= row['verdict']=='GREEN'
        emit(row)
    # Shapes, GEMM numerical and speed rules are identical to PT-TF32-1.
    old.gemm(c,record)
    return green


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
        return {n:dict(num.metric_gate(values[n],reference[n],num.bounds(r,s,n)['bound']),
                       expected=num.bounds(r,s,n)['expected']) for n in names}
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
    registration()
    r=dict(verdict='BLOCKED',evidence_class='authorization boundary, no GPU measurements',
           reason="PT-TF32-2 seat is CPU/build only; CUDA_VISIBLE_DEVICES=''. Lead owns GPU gates.",
           registration_sha256=sha(REG),gates={k:dict(status='BLOCKED',measured=False) for k in GATES},
           not_claimed_fixed=['native dK defect','seven native edge failures','5x GEMM speed','new model gates','training stability'])
    create_json(ART/'BLOCKED_REPORT.json',r);print(json.dumps(r,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['seal','blocked','gemm','attention'])
    p.add_argument('--lead-gpu',action='store_true');p.add_argument('--case',type=int,choices=range(4),default=0)
    p.add_argument('--out',type=Path);args=p.parse_args()
    if args.command=='seal':seal();return 0
    if args.command=='blocked':blocked();return 0
    if not args.lead_gpu or not os.environ.get('CUDA_VISIBLE_DEVICES'):p.error('BLOCKED: lead-only GPU gate')
    if not args.out or not args.out.resolve().is_relative_to(ART):p.error('--out must be a fresh PT-TF32-2 artifact directory')
    m=verify_manifest();args.out.mkdir(parents=True,exist_ok=False)
    rows=[];start=time.monotonic();error=None;good=False
    def emit(row):
        create_json(args.out/f'case_{len(rows):02d}.json',row);rows.append(row)
        print(row['kind'],row.get('direction',''),row['verdict'],flush=True)
    try:
        c=load_fork();good=gemm(c,emit) if args.command=='gemm' else attention(c,emit,args.case)
    except Exception as exc:
        error=repr(exc);print(error,file=sys.stderr)
    result=dict(verdict='GREEN' if good else 'RED',error=error,completed_cases=len(rows),
                registration_sha256=sha(REG),manifest_sha256=sha(ART/'SOURCE_MANIFEST.json'),binary=m['binary'],
                command=args.command,case=args.case,elapsed_seconds=time.monotonic()-start,
                evidence_class='GPU kernel gate; no model quality claim')
    create_json(args.out/'summary.json',result);print(json.dumps(result,indent=2));return 0 if good else 1


if __name__=='__main__':sys.exit(main())
