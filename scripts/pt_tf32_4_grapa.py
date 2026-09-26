#!/usr/bin/env python3
"""Exact-state lead harness with default zero gradient/checkpoint dumps.

Prior art: CC39/CC41 and PT-TF32-1/2 (2026) exact-state loaders/policy, taken;
POSIX pipes/Python subprocess, NumPy (2020), SHA256 (2001), taken transport.
Ours: scoped output adapters, exact in-memory comparisons, retained sufficient
statistics and source/data hashes. Model/trainer sources are never edited.
"""
import argparse
import contextlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pt_tf32_2_grapa as previous
import pt_tf32_grapa as legacy
import pt_tf32_4 as gates
import pt_tf32_3_storage as storage
from pt_tf32_2_numerics import cosine,noise_floor

ROOT,ART=gates.ROOT,gates.ART
REG=ART/'GRAPA_REGISTRATION_002.json'


def prepare():
    # Reuse exact source/checkpoint/cursor pins, create a NEW engine receipt.
    original=(previous.ART/'GRAPA_REGISTRATION_002.json')
    if gates.sha(original)!=original.with_suffix('.sha256').read_text().strip():
        raise ValueError('previous GRAPA registration drift')
    r=json.loads(original.read_text());meta=r.pop('pt_tf32_2')
    for p,h in meta['source_pins'].items():
        if gates.sha(p)!=h:raise ValueError('model source drift: '+p)
    binary=list((ROOT/'tensor_cuda/tensor_cuda').glob('_tensor_cuda*.so'))
    if len(binary)!=1:raise ValueError('one fork extension required')
    r['engine']=dict(so_path=str(binary[0]),so_sha256=gates.sha(binary[0]),
        py_sha256={str(p):gates.sha(p) for p in sorted((ROOT/'tensor_cuda/tensor_cuda').glob('*.py'))})
    meta.update(previous_experiment=dict(path=str(original),sha256=gates.sha(original)),
        order_registration_sha256=gates.sha(gates.REG),gates=gates.registration()['model_gate'],
        evidence='PT-TF32-4 new fork experiment; historical registrations untouched')
    r['pt_tf32_4']=meta;gates.create_json(REG,r)
    with REG.with_suffix('.sha256').open('x') as f:f.write(gates.sha(REG)+'\n')
    print('GRAPA_REGISTERED',gates.sha(REG))


def registered():
    if gates.sha(REG)!=REG.with_suffix('.sha256').read_text().strip():raise ValueError('GRAPA registration drift')
    r=json.loads(REG.read_text());meta=r['pt_tf32_4']
    if gates.sha(gates.REG)!=meta['order_registration_sha256']:raise ValueError('order registration drift')
    for entry in (dict(path=meta['parent_registration'],sha256=meta['parent_sha256']),meta['previous_experiment']):
        if gates.sha(entry['path'])!=entry['sha256']:raise ValueError('parent registration drift')
    for p,h in {**meta['source_pins'],**r['engine']['py_sha256']}.items():
        if gates.sha(p)!=h:raise ValueError('source drift: '+p)
    path=Path(r['engine']['so_path'])
    if not path.resolve().is_relative_to(ROOT) or gates.sha(path)!=r['engine']['so_sha256']:
        raise ValueError('fork binary drift')
    return r


@contextlib.contextmanager
def gradient_output(gv,dest,stream,keep):
    """Adapt only this isolated CC39 module's output functions, not global NumPy.

    Prior art: dependency injection and Python module proxies, taken. Ours:
    replace the grad.npz output sink with an exact pipe and content receipt.
    All checkpoint, trainer, token, engine and gradient computation checks run.
    """
    original_np,original_sha,original_write=gv.np,gv.sha256_file,gv.write_exclusive
    gradient_path=dest/'grad.npz';captured={}
    class Outputs:
        def __getattr__(self,key):return getattr(original_np,key)
        def savez(self,path,*args,**grads):
            if Path(path)!=gradient_path:return original_np.savez(path,*args,**grads)
            if args or captured:raise ValueError('unexpected gradient output')
            content_sha=storage.canonical_sha(grads)
            captured.update(content_sha256=content_sha,bytes=sum(a.nbytes for a in grads.values()))
            if keep:
                storage.require_space(dest);original_np.savez(path,**grads)
            storage.send_grads(stream,grads)
            # EOF is sent before the small pair receipt; parent then waits for
            # child exit and verifies that receipt against the received arrays.
            stream.close()
    def sha(path):
        if Path(path)==gradient_path and not keep:return captured['content_sha256']
        return original_sha(path)
    def write(path,record):
        if Path(path)==dest/'pair.json':
            record=dict(record,gradient_transport='pipe',gradient_content_sha256=captured['content_sha256'],
                        gradient_bytes=captured['bytes'],keep_grads=keep)
            if not keep:record.update(grad_file=None,grad_sha256=None)
        return original_write(path,record)
    gv.np=Outputs();gv.sha256_file=sha;gv.write_exclusive=write
    try:yield
    finally:gv.np=original_np;gv.sha256_file=original_sha;gv.write_exclusive=original_write


@contextlib.contextmanager
def checkpoint_output(module,dest,keep):
    """Prior art: scoped Python injection, taken; only the output save is adapted.
    Trainer load, updates, cursor and timing remain identical. Save attempts
    remain auditable because the trainer's text still says 'final ckpt'.
    """
    original=module.save;attempts=[]
    def save(path,*args,**kwargs):
        if not Path(path).resolve().is_relative_to(dest.resolve()):raise ValueError('checkpoint outside lane')
        attempts.append(dict(path=str(path),written=keep))
        if keep:
            storage.require_space(dest);return original(path,*args,**kwargs)
    module.save=save
    try:yield
    finally:
        module.save=original
        if dest.exists():gates.create_json(dest/'checkpoint_output.json',dict(keep_grads=keep,attempts=attempts))


def child(args):
    storage.require_space(args.out)
    legacy.registered=registered;legacy.gates=gates
    if args.command=='state':
        if args.gradient_fd is None:raise ValueError('state child requires parent-owned gradient pipe')
        gv=legacy.load_gv();original_loader=legacy.load_gv
        legacy.load_gv=lambda:gv
        try:
            with os.fdopen(args.gradient_fd,'wb') as stream:
                with gradient_output(gv,args.out,stream,args.keep_grads):return legacy.state_child(args)
        finally:legacy.load_gv=original_loader
    legacy.load_gv()
    # checkpoint imports tensor_cuda at module scope. Bind and verify the fork
    # BEFORE that import, then reuse the installed policy exactly once.
    tc=legacy.engine_and_policy();original_engine=legacy.engine_and_policy
    legacy.engine_and_policy=lambda:tc
    try:
        module=importlib.import_module('grapa.checkpoint')
        with checkpoint_output(module,args.out,args.keep_grads):return legacy.timing_child(args)
    finally:legacy.engine_and_policy=original_engine


def run_child(command,state,arm,dest,keep=False):
    storage.require_space(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    argv=[sys.executable,'-B',str(Path(__file__).resolve()),command,'--lead-gpu','--child',
          '--state',state,'--arm',arm,'--out',str(dest)]
    if keep:argv.append('--keep-grads')
    read_fd=write_fd=None
    if command=='state':
        read_fd,write_fd=os.pipe();argv+=['--gradient-fd',str(write_fd)]
    try:
        with dest.with_suffix('.log').open('x') as log:
            # Inherit only our slot's process group. No detached child or GPU lock.
            p=subprocess.Popen(argv,cwd=ROOT,env=os.environ,stdout=log,stderr=subprocess.STDOUT,
                               pass_fds=() if write_fd is None else (write_fd,))
            if write_fd is not None:os.close(write_fd);write_fd=None
            grads=None;error=None
            try:
                if read_fd is not None:
                    with os.fdopen(read_fd,'rb') as stream:
                        read_fd=None;grads=storage.receive_grads(stream)
            except Exception as exc:error=exc
            rc=p.wait()
            if rc:
                text=dest.with_suffix('.log').read_text()
                if 'No space left on device' in text or 'free space below 8 GiB' in text:
                    raise storage.StorageBlocked(dict(status='BLOCKED',reason='child storage failure',returncode=rc))
                raise RuntimeError(f'{command}/{state}/{arm} child rc={rc}; {dest.with_suffix(".log")}')
            if error:raise error
        if grads is not None:
            receipt=json.loads((dest/'pair.json').read_text())
            if storage.canonical_sha(grads)!=receipt['gradient_content_sha256']:raise ValueError('pipe/receipt drift')
        return grads
    finally:
        for fd in (read_fd,write_fd):
            if fd is not None:os.close(fd)


def dot_statistics(a,b):
    # Prior art: exact FP64 cosine dot/norm sufficient statistics, standard
    # linear algebra; NumPy. These are exact full-gradient comparisons, not a sketch.
    cosine(a,b) # enforce matching keys/shapes and finite, nonzero norms
    dot=aa=bb=0.
    for k in sorted(a):
        x,y=a[k].astype(np.float64),b[k].astype(np.float64)
        dot+=float(np.sum(x*y));aa+=float(np.sum(x*x));bb+=float(np.sum(y*y))
    return dict(dot=dot,norm_a_squared=aa,norm_b_squared=bb)


def calibrate(out,keep=False):
    inputs=[];gradients=[]
    for i,arm in enumerate(('none','0-10','0-10','none')):
        dest=out/f'run_{i}'/arm;g=run_child('state','healthy',arm,dest,keep)
        gradients.append(g)
        inputs.append(dict(arm=arm,receipt=str(dest/'pair.json'),receipt_sha256=gates.sha(dest/'pair.json'),
                           content_sha256=storage.canonical_sha(g)))
    floor=noise_floor(gradients)
    for pair in floor['pairs']:pair.update(dot_statistics(gradients[pair['i']],gradients[pair['j']]))
    floor.update(inputs=inputs,state='healthy',schema=3,keep_grads=keep,
        registration_sha256=gates.sha(gates.REG),manifest_sha256=gates.sha(ART/'SOURCE_MANIFEST.json'),
        evidence_class='four fresh exact-state GPU gradients; full FP64 dot/norms; arrays ephemeral by default')
    gates.create_json(out/'NOISE_FLOOR.json',floor)
    with (out/'NOISE_FLOOR.sha256').open('x') as f:f.write(gates.sha(out/'NOISE_FLOOR.json')+'\n')
    return dict(verdict='GREEN',bar=floor['bar'],observed_spread=floor['observed_spread'])


def checked_floor(path):
    if path is None or not path.resolve().is_relative_to(ART):raise ValueError('frozen noise-floor receipt required')
    if gates.sha(path)!=path.with_suffix('.sha256').read_text().strip():raise ValueError('noise-floor seal drift')
    floor=json.loads(path.read_text())
    if floor['registration_sha256']!=gates.sha(gates.REG) or floor['manifest_sha256']!=gates.sha(ART/'SOURCE_MANIFEST.json'):
        raise ValueError('noise-floor experiment drift')
    if [p['arm'] for p in floor['inputs']]!=['none','0-10','0-10','none']:raise ValueError('calibration arm coverage')
    if len({p['receipt'] for p in floor['inputs']})!=4:raise ValueError('separate calibration executions required')
    for p in floor['inputs']:
        receipt=Path(p['receipt'])
        if not receipt.resolve().is_relative_to(path.parent) or gates.sha(receipt)!=p['receipt_sha256']:
            raise ValueError('calibration receipt drift')
        r=json.loads(receipt.read_text())
        if r['gradient_content_sha256']!=p['content_sha256']:raise ValueError('calibration content drift')
        if r['grad_file'] is not None and gates.sha(r['grad_file'])!=r['grad_sha256']:raise ValueError('retained gradient drift')
    expected=[(i,j) for i in range(4) for j in range(i+1,4)]
    if [(p['i'],p['j']) for p in floor['pairs']]!=expected:raise ValueError('six-pair coverage required')
    for pair in floor['pairs']:
        dot,aa,bb=[pair[k] for k in ('dot','norm_a_squared','norm_b_squared')]
        if not all(map(math.isfinite,(dot,aa,bb))) or min(aa,bb)<=0:raise ValueError('invalid norms')
        c=dot/math.sqrt(aa)/math.sqrt(bb)
        if abs(c)>1+8*np.finfo(np.float64).eps:raise ValueError('invalid cosine')
        c=min(1.,max(-1.,c))
        if pair['cos']!=c or pair['distance']!=1-c:raise ValueError('noise-floor computation drift')
    spread=max(p['distance'] for p in floor['pairs'])
    if spread>=1/3 or floor['observed_spread']!=spread or floor['bar']!=1-3*spread:
        raise ValueError('noise-floor computation drift')
    return floor


def state(args):
    r=registered();rules=gates.registration()['model_gate']
    floor=checked_floor(args.noise_floor) if args.state!='onset' else None
    base=run_child('state',args.state,'none',args.out/'none',args.keep_grads)
    candidate=run_child('state',args.state,'0-10',args.out/'0-10',args.keep_grads)
    change=cosine(candidate,base)
    if args.state=='onset':
        ref=r['states']['onset']['refs']['fp64']
        if legacy.streaming_sha(ref['path'])!=ref['sha256']:raise ValueError('FP64 reference drift')
        oracle=previous.load_grads(ref['path']);c=cosine(candidate,oracle)
        diff=sum(float(np.sum((candidate[k].astype(np.float64)-oracle[k])**2)) for k in candidate)
        norm=sum(float(np.sum(oracle[k].astype(np.float64)**2)) for k in candidate)
        relative=(diff/norm)**.5;bar=rules['onset_cos_min']
        good=c>=bar and relative<=rules['onset_relative_L2_max']
    else:c=change;relative=None;bar=floor['bar'];good=c>=bar
    return dict(verdict='GREEN' if good else 'RED',state=args.state,tf32_vs_none_cos=change,
                registered_cos=c,relative_L2=relative,bar=bar,
                noise_floor_sha256=gates.sha(args.noise_floor) if floor else None,
                evidence_class='exact-state full-gradient comparison, no long-term training claim')


def timing(args):
    r=registered();legacy.load_gv();cc41=importlib.import_module('cc41_fwd_precision');summaries={}
    for arm in ('none','0-10'):
        dest=args.out/arm;run_child('timing','healthy',arm,dest,args.keep_grads)
        rows=cc41.timing_rows((dest/'logs/train.log').read_text(),arm,r['timing']['reference_rows_step_sample_i_loss'])
        if len(rows)!=20 or not all(x['cursor_ok'] for x in rows):raise ValueError('timing steps/cursor mismatch')
        if arm=='none' and not all(x['loss_equals_reference'] for x in rows):raise ValueError('none loss reproduction failed')
        summaries[arm]=dict(rows=rows,median_s_per_step=float(np.median([x['s_per_step'] for x in rows])))
    base=summaries['none']['median_s_per_step'];fast=summaries['0-10']['median_s_per_step']
    rules=gates.registration()['model_gate'];good=base>0 and fast<=rules['seconds_per_step_max'] and fast/base<=rules['bf16_ratio_max']
    return dict(verdict='GREEN' if good else 'RED',arms=summaries,ratio=fast/base,
                evidence_class='20-step GPU timing; checkpoint saves disabled unless --keep-grads')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','noise','state','timing'])
    p.add_argument('--lead-gpu',action='store_true');p.add_argument('--child',action='store_true')
    p.add_argument('--state',choices=['onset','healthy','control'],default='onset')
    p.add_argument('--arm',choices=['none','0-10'],default='0-10');p.add_argument('--out',type=Path)
    p.add_argument('--noise-floor',type=Path);p.add_argument('--keep-grads',action='store_true')
    p.add_argument('--gradient-fd',type=int,help=argparse.SUPPRESS);args=p.parse_args()
    if args.command=='prepare':prepare();return 0
    if not args.lead_gpu or not os.environ.get('CUDA_VISIBLE_DEVICES'):p.error('BLOCKED: lead-only GPU model gate')
    if not args.out or not args.out.resolve().is_relative_to(ART):p.error('output must stay in PT-TF32-4 artifacts')
    args.out=args.out.resolve();gates.verify_manifest();registered()
    check=storage.space_status(args.out)
    if check['status']!='GREEN':
        args.out.mkdir(parents=True,exist_ok=False);gates.create_json(args.out/'summary.json',dict(check,verdict='BLOCKED'))
        print(json.dumps(check));return 2
    if args.child:return child(args)
    args.out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    try:result=calibrate(args.out,args.keep_grads) if args.command=='noise' else state(args) if args.command=='state' else timing(args)
    except storage.StorageBlocked as exc:result=dict(exc.receipt,verdict='BLOCKED')
    except OSError as exc:
        result=dict(verdict='BLOCKED' if exc.errno==28 else 'RED',error=repr(exc))
    except Exception as exc:result=dict(verdict='RED',error=repr(exc));print(repr(exc),file=sys.stderr)
    result.update(registration_sha256=gates.sha(gates.REG),manifest_sha256=gates.sha(ART/'SOURCE_MANIFEST.json'),
                  elapsed_seconds=time.monotonic()-start,keep_grads=args.keep_grads,dump_bytes=storage.dump_bytes(args.out))
    gates.create_json(args.out/'summary.json',result);print(json.dumps(result,indent=2))
    return 0 if result['verdict']=='GREEN' else 2 if result['verdict']=='BLOCKED' else 1


if __name__=='__main__':sys.exit(main())
