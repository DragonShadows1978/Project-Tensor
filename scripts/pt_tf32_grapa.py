#!/usr/bin/env python3
"""Fork-only lead adapter for CC39/CC41 exact-state and 20-step gates.

Prior art: CC39/CC41 (Project-Tensor/GRAPA 2026) exact-state harness and
per-block policy, taken unchanged as imported callables. Mixed precision
(Micikevicius et al. 2018), taken scoped compute policy; ours: explicit TF32
mapping/context and separate immutable engine registration. Original GRAPA
files, live engine and existing registrations are never edited. No GPU lock.
"""
from __future__ import annotations
import argparse
import contextlib
import copy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace

sys.dont_write_bytecode=True
import pt_tf32_1 as gates
import numpy as np

ROOT=gates.ROOT
ART=gates.ART
GRAPA=Path('/mnt/ForgeRealm/wt/grapa-cc41')
REG=ART/'GRAPA_REGISTRATION.json'


def streaming_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def install_policy(tc,fp):
    """Apply once before model construction; initial/replay scopes both use it.

    Taken: CC41 context/restoration and BP-KERNEL-2 backward capture. Ours:
    turn on TF32 GEMM only inside an fp32 block; native GEMM backward captures
    that setting, so restoring the caller cannot silently change its gradient.

    Deferred split design (NOT implemented): preserve initial/replay FP32
    forward and saved state, then explicitly choose BF16 operands/temporaries
    inside each VJP, restoring FP32 leaf accumulation at the block boundary.
    Taken: Micikevicius et al., Mixed Precision Training (ICLR 2018),
    https://arxiv.org/abs/1710.03740, and CC39-B's forward/backward ablation.
    Ours: proposed scope at CC41 block boundaries. Requires the TF32 step gate
    to miss before implementation; casting just a block output cannot do this.
    """
    if getattr(fp,'_pt_tf32_installed',False):
        raise RuntimeError('TF32 policy already installed')
    base=fp.fp32_apa_variants
    fp.FP32_BWD_FALLBACK=dict(fp.FP32_BWD_FALLBACK,g1='g1_tf32',g2='g1_tf32')
    fp.FP32_FWD_FALLBACK=dict(fp.FP32_FWD_FALLBACK,h='h_tf32')
    @contextlib.contextmanager
    def block(tc):
        old=tc._C.get_tf32_gemm()
        tc._C.set_tf32_gemm(True)
        try:
            with base(tc):yield
        finally:
            tc._C.set_tf32_gemm(old)
    fp.fp32_apa_variants=block
    fp._pt_tf32_installed=True


def prepare():
    """CPU only: pin the extension, original registration and all local code."""
    gates.registration()
    base=GRAPA/'artifacts/cc41/REGISTRATION.json'
    r=json.loads(base.read_text())
    for rel,h in r['trainer_sources_sha256'].items():
        if gates.sha(GRAPA/rel)!=h:raise ValueError('CC41 trainer source drift: '+rel)
    binary=list((ROOT/'tensor_cuda/tensor_cuda').glob('_tensor_cuda*.so'))
    if len(binary)!=1:raise ValueError('one fork binary required')
    r['engine']=dict(so_path=str(binary[0]),so_sha256=gates.sha(binary[0]),
        py_sha256={str(p):gates.sha(p) for p in sorted((ROOT/'tensor_cuda/tensor_cuda').glob('*.py'))})
    paths=[]
    for folder in ('grapa','corpus','scripts'):
        paths.extend((GRAPA/folder).rglob('*.py'))
    r['pt_tf32_1']=dict(parent_registration=str(base),parent_sha256=gates.sha(base),
        order_registration_sha256=gates.sha(gates.REG),
        source_pins={str(p):gates.sha(p) for p in paths},
        policy='FP32 blocks 0-10: GEMM TF32 and h_tf32/g1_tf32; other blocks original g1/h; initial and replay covered.',
        original_trace_engine=json.loads(base.read_text())['engine'],
        gates=gates.registration()['model_gate'],
        evidence='New fork experiment; not the engine used by the historical trace.')
    gates.create_json(REG,r)
    with REG.with_suffix('.sha256').open('x') as f:f.write(gates.sha(REG)+'\n')
    print('GRAPA_REGISTERED',REG,gates.sha(REG))


def registered():
    if gates.sha(REG)!=REG.with_suffix('.sha256').read_text().strip():raise ValueError('GRAPA registration drift')
    r=json.loads(REG.read_text());meta=r['pt_tf32_1']
    if gates.sha(meta['parent_registration'])!=meta['parent_sha256']:raise ValueError('CC41 registration drift')
    if gates.sha(gates.REG)!=meta['order_registration_sha256']:raise ValueError('order registration drift')
    for p,h in {**meta['source_pins'],**r['engine']['py_sha256']}.items():
        if gates.sha(p)!=h:raise ValueError('source drift: '+p)
    if gates.sha(r['engine']['so_path'])!=r['engine']['so_sha256']:raise ValueError('fork engine drift')
    return r


def load_gv():
    # Pure-Python CC39 import; its GPU entry point is called only by --child.
    sys.path.insert(0,str(GRAPA));sys.path.insert(0,str(GRAPA/'scripts'))
    spec=importlib.util.spec_from_file_location('pt_tf32_cc39',GRAPA/'scripts/cc39_grad_variants.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def engine_and_policy():
    if os.environ.get('NVIDIA_TF32_OVERRIDE')=='0':
        raise RuntimeError('NVIDIA_TF32_OVERRIDE=0 disables the requested TF32 GEMMs')
    sys.path.insert(0,str(ROOT/'tensor_cuda'))
    tc=importlib.import_module('tensor_cuda')
    if Path(tc._C.__file__).resolve()!=Path(registered()['engine']['so_path']).resolve():
        raise RuntimeError('refusing non-fork engine')
    from grapa import fwd_precision as fp
    tc._C.set_tf32_gemm(False)
    install_policy(tc,fp)
    return tc


def state_child(args):
    r=registered();gv=load_gv();st=r['states'][args.state]
    if streaming_sha(st['ckpt'])!=st['sha256']:raise ValueError('checkpoint drift')
    if streaming_sha(r['batches_npz']['path'])!=r['batches_npz']['sha256']:raise ValueError('batches drift')
    data=np.load(r['batches_npz']['path'],allow_pickle=False)
    tokens=data[f'x_{st["batch"]}'].astype(np.int64)
    if gv.sha256_array(tokens)!=st['tokens_sha256']:raise ValueError('tokens drift')
    batch=args.out.parent/f'batch_{args.arm}.npz'
    with batch.open('xb') as f:
        np.savez(f,x=tokens[None,:-1],y=tokens[None,1:],w=np.ones((1,len(tokens)-1),np.float32))
    engine_and_policy()
    # CC39's own trainer/engine/token checks run on a separately registered
    # fork experiment, not by mutating or bypassing the historical registration.
    gv.load_reg=lambda unused:r
    a=SimpleNamespace(child=f'g1:h@{args.arm}',state=st['ckpt'],batch=st['batch'],
        fwd_fp32_blocks='0-10',fixture_registration=None,engine=str(ROOT/'tensor_cuda'),
        batch_npz=str(batch),out=str(args.out))
    return gv.child_main(a)


def state(args):
    r=registered();args.out.mkdir(parents=True,exist_ok=False)
    for arm in ('none','0-10'):
        cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'state',
             '--lead-gpu','--child','--state',args.state,'--arm',arm,'--out',str(args.out/arm)]
        with (args.out/f'{arm}.log').open('x') as log:
            p=subprocess.run(cmd,cwd=ROOT,env=os.environ,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        if p.returncode:raise RuntimeError(f'{arm} child exit {p.returncode}; see log')
    gv=load_gv()
    none=gv.load_grads(args.out/'none/grad.npz')
    candidate=gv.load_grads(args.out/'0-10/grad.npz')
    change=gv.global_cos(candidate,none);rules=r['pt_tf32_1']['gates']
    if args.state=='onset':
        ref=r['states']['onset']['refs']['fp64']
        fp64=gv.load_grads(ref['path'],ref['sha256'])
        comparison=gv.global_cos(candidate,fp64)
        passed=comparison['cos']>=rules['onset_cos_min'] and comparison['rel_err']<=rules['onset_relative_L2_max']
    else:
        comparison=change;passed=comparison['cos']>=rules['healthy_control_cos_min']
    result=dict(state=args.state,tf32_vs_none=change,registered_comparison=comparison,
                verdict='GREEN' if passed else 'RED',registration_sha256=gates.sha(REG),
                evidence_class='one exact-state GPU forward/backward; no training stability claim')
    gates.create_json(args.out/'summary.json',result)
    print(json.dumps(result,indent=2));return 0 if passed else 1


def timing_child(args):
    r=registered();load_gv();tc=engine_and_policy()
    source=r['timing']['source']
    if streaming_sha(source['path'])!=source['sha256']:raise ValueError('timing checkpoint drift')
    # Exact registered CC41 trainer argv, with output paths confined to the fork.
    argv=[s.replace('{ROOT}',str(GRAPA)).replace('{ARM_DIR}',str(args.out))
          for s in r['timing']['arms'][args.arm]]
    argv[0]=str(GRAPA/'grapa/train.py')
    tc._C.bp_kernel_2_set_variant('g1');tc._C.bp_kernel_4_set_variant('h')
    args.out.mkdir(parents=True,exist_ok=False)
    gates.create_json(args.out/'argv.json',argv)
    # Each arm loads its own checkpoint; no reuse of a model updated by another arm.
    sys.argv=argv;os.chdir(GRAPA)
    runpy.run_path(argv[0],run_name='__main__')
    return 0


def timing(args):
    r=registered();args.out.mkdir(parents=True,exist_ok=False)
    gv=load_gv();cc41=importlib.import_module('cc41_fwd_precision')
    summaries={}
    for arm in ('none','0-10'):
        dest=args.out/arm
        cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'timing','--lead-gpu',
             '--child','--arm',arm,'--out',str(dest)]
        with (args.out/f'{arm}.log').open('x') as log:
            p=subprocess.run(cmd,cwd=ROOT,env=os.environ,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        if p.returncode:raise RuntimeError(f'{arm} timing child exit {p.returncode}')
        rows=cc41.timing_rows((dest/'logs/train.log').read_text(),arm,r['timing']['reference_rows_step_sample_i_loss'])
        if len(rows)!=20 or not all(x['cursor_ok'] for x in rows):raise ValueError('timing steps/cursor mismatch')
        if arm=='none' and not all(x['loss_equals_reference'] for x in rows):raise ValueError('none loss reproduction failed')
        med=float(np.median([x['s_per_step'] for x in rows]))
        summaries[arm]=dict(rows=rows,median_s_per_step=med)
    base=summaries['none']['median_s_per_step'];fast=summaries['0-10']['median_s_per_step']
    rules=r['pt_tf32_1']['gates'];passed=fast<=rules['seconds_per_step_max'] and fast/base<=rules['bf16_ratio_max']
    result=dict(arms=summaries,ratio=fast/base,verdict='GREEN' if passed else 'RED',
                registration_sha256=gates.sha(REG),evidence_class='20-step GPU timing, not long-term stability')
    gates.create_json(args.out/'summary.json',result);print(json.dumps(result,indent=2))
    return 0 if passed else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','state','timing'])
    p.add_argument('--lead-gpu',action='store_true')
    p.add_argument('--state',choices=['onset','healthy','control'],default='onset')
    p.add_argument('--arm',choices=['none','0-10'],default='0-10')
    p.add_argument('--child',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--out',type=Path)
    args=p.parse_args()
    if args.command=='prepare':prepare();return 0
    if not args.lead_gpu or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        p.error('BLOCKED: GPU gates are lead-only')
    if not args.out or not args.out.resolve().is_relative_to(ROOT.resolve()):
        p.error('--out must be under the fork')
    args.out=args.out.resolve()
    gates.verify_manifest();registered()
    if args.command=='state':return state_child(args) if args.child else state(args)
    return timing_child(args) if args.child else timing(args)


if __name__=='__main__':
    sys.exit(main())
