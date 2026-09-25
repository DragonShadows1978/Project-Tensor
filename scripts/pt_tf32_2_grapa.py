#!/usr/bin/env python3
"""Lead-only exact-state calibration, holdouts and step timing.

Prior art: CC39/CC41 and PT-TF32-1 (2026), taken read-only model loaders and
scoped policy; Micikevicius et al. (2018), taken mixed precision principle.
Ours: four-process healthy calibration, frozen bar and fresh holdouts. The
specific cosine rule is the user's order; no prior art known to me.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
import pt_tf32_grapa as legacy
import pt_tf32_2 as gates
from pt_tf32_2_numerics import cosine,noise_floor

ROOT,ART=gates.ROOT,gates.ART
REG=ART/'GRAPA_REGISTRATION_002.json'


def prepare():
    gates.registration()
    old=legacy.ART/'GRAPA_REGISTRATION.json'
    if gates.sha(old)!=old.with_suffix('.sha256').read_text().strip():raise ValueError('historical GRAPA registration drift')
    r=json.loads(old.read_text());meta=r.pop('pt_tf32_1')
    # Retain all exact checkpoint/token/cursor pins. Only this new experiment's
    # binary and rules change, in a new file; never repin historical results.
    for p,h in meta['source_pins'].items():
        if gates.sha(p)!=h:raise ValueError('GRAPA source drift: '+p)
    binary=list((ROOT/'tensor_cuda/tensor_cuda').glob('_tensor_cuda*.so'))
    if len(binary)!=1:raise ValueError('one fork extension required')
    r['engine']=dict(so_path=str(binary[0]),so_sha256=gates.sha(binary[0]),
        py_sha256={str(p):gates.sha(p) for p in sorted((ROOT/'tensor_cuda/tensor_cuda').glob('*.py'))})
    meta['previous_experiment']=dict(path=str(old),sha256=gates.sha(old))
    meta['order_registration_sha256']=gates.sha(gates.REG)
    meta['gates']=gates.registration()['model_gate']
    meta['evidence']='PT-TF32-2 fork experiment; no replacement of historical trace or PT-TF32-1 receipts'
    r['pt_tf32_2']=meta
    gates.create_json(REG,r)
    with REG.with_suffix('.sha256').open('x') as f:f.write(gates.sha(REG)+'\n')
    print('GRAPA_REGISTERED',gates.sha(REG))


def registered():
    if gates.sha(REG)!=REG.with_suffix('.sha256').read_text().strip():raise ValueError('GRAPA registration drift')
    r=json.loads(REG.read_text());m=r['pt_tf32_2']
    if gates.sha(gates.REG)!=m['order_registration_sha256']:raise ValueError('order registration drift')
    for entry in (dict(path=m['parent_registration'],sha256=m['parent_sha256']),m['previous_experiment']):
        if gates.sha(entry['path'])!=entry['sha256']:raise ValueError('parent registration drift')
    for p,h in {**m['source_pins'],**r['engine']['py_sha256']}.items():
        if gates.sha(p)!=h:raise ValueError('source drift: '+p)
    path=Path(r['engine']['so_path'])
    if not path.resolve().is_relative_to(ROOT) or gates.sha(path)!=r['engine']['so_sha256']:
        raise ValueError('fork binary drift')
    return r


def configure_child_adapter():
    # Explicit dependency injection into read-only PT-TF32-1 child functions.
    # Every imported child still enforces the new registered fork binary,
    # trainer sources, checkpoint and token pins. No historical files change.
    legacy.registered=registered
    legacy.gates=gates


def run_child(command,state,arm,dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    argv=[sys.executable,'-B',str(Path(__file__).resolve()),command,'--lead-gpu','--child',
          '--state',state,'--arm',arm,'--out',str(dest)]
    with dest.with_suffix('.log').open('x') as log:
        # Inherit the slot process group: the parent kills ONLY its own group
        # on deadline, including this child. No detached grandchildren/waits.
        p=subprocess.run(argv,cwd=ROOT,env=os.environ,stdout=log,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError(f'{command}/{state}/{arm} child rc={p.returncode}; {dest.with_suffix(".log")}')


def load_grads(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}


def calibrate(out):
    inputs=[];gradients=[]
    for i,arm in enumerate(('none','0-10','0-10','none')):
        dest=out/f'run_{i}'/arm;run_child('state','healthy',arm,dest)
        path=dest/'grad.npz';inputs.append(dict(arm=arm,path=str(path),sha256=gates.sha(path)))
        gradients.append(load_grads(path))
    floor=dict(noise_floor(gradients),inputs=inputs,state='healthy',
               registration_sha256=gates.sha(gates.REG),manifest_sha256=gates.sha(ART/'SOURCE_MANIFEST.json'),
               evidence_class='four exact-state GPU forwards/backwards, followed by CPU cosine computation')
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
    if len({p['path'] for p in floor['inputs']})!=4:raise ValueError('calibration inputs must be separate executions')
    gradients=[]
    for p in floor['inputs']:
        if gates.sha(p['path'])!=p['sha256']:raise ValueError('calibration input drift')
        gradients.append(load_grads(p['path']))
    recomputed=noise_floor(gradients)
    if any(floor[k]!=recomputed[k] for k in ('pairs','observed_spread','bar')):raise ValueError('noise-floor computation drift')
    return floor


def state(args):
    r=registered();rules=gates.registration()['model_gate']
    # Freeze/check calibration before running either holdout arm, not afterward.
    floor=checked_floor(args.noise_floor) if args.state!='onset' else None
    for arm in ('none','0-10'):run_child('state',args.state,arm,args.out/arm)
    base=load_grads(args.out/'none/grad.npz');candidate=load_grads(args.out/'0-10/grad.npz')
    change=cosine(candidate,base)
    if args.state=='onset':
        ref=r['states']['onset']['refs']['fp64']
        if gates.sha(ref['path'])!=ref['sha256']:raise ValueError('FP64 reference drift')
        oracle=load_grads(ref['path']);c=cosine(candidate,oracle)
        diff=sum(float(np.sum((candidate[k].astype(np.float64)-oracle[k])**2)) for k in candidate)
        norm=sum(float(np.sum(oracle[k].astype(np.float64)**2)) for k in candidate)
        relative=(diff/norm)**.5;bar=rules['onset_cos_min']
        good=c>=bar and relative<=rules['onset_relative_L2_max']
    else:
        c=change;relative=None;bar=floor['bar'];good=c>=bar
    return dict(verdict='GREEN' if good else 'RED',state=args.state,tf32_vs_none_cos=change,
                registered_cos=c,relative_L2=relative,bar=bar,
                noise_floor_sha256=gates.sha(args.noise_floor) if floor else None,
                evidence_class='one exact-state comparison, no long-term training claim')


def timing(args):
    r=registered();legacy.load_gv();cc41=importlib.import_module('cc41_fwd_precision')
    summaries={}
    for arm in ('none','0-10'):
        dest=args.out/arm;run_child('timing','healthy',arm,dest)
        rows=cc41.timing_rows((dest/'logs/train.log').read_text(),arm,r['timing']['reference_rows_step_sample_i_loss'])
        if len(rows)!=20 or not all(x['cursor_ok'] for x in rows):raise ValueError('timing steps/cursor mismatch')
        if arm=='none' and not all(x['loss_equals_reference'] for x in rows):raise ValueError('none loss reproduction failed')
        summaries[arm]=dict(rows=rows,median_s_per_step=float(np.median([x['s_per_step'] for x in rows])))
    base=summaries['none']['median_s_per_step'];fast=summaries['0-10']['median_s_per_step']
    rules=gates.registration()['model_gate'];good=base>0 and fast<=rules['seconds_per_step_max'] and fast/base<=rules['bf16_ratio_max']
    return dict(verdict='GREEN' if good else 'RED',arms=summaries,ratio=fast/base,
                evidence_class='20-step GPU timing, no long-term training stability claim')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','noise','state','timing'])
    p.add_argument('--lead-gpu',action='store_true');p.add_argument('--child',action='store_true')
    p.add_argument('--state',choices=['onset','healthy','control'],default='onset')
    p.add_argument('--arm',choices=['none','0-10'],default='0-10');p.add_argument('--out',type=Path)
    p.add_argument('--noise-floor',type=Path);args=p.parse_args()
    if args.command=='prepare':prepare();return 0
    if not args.lead_gpu or not os.environ.get('CUDA_VISIBLE_DEVICES'):p.error('BLOCKED: lead-only GPU model gate')
    if not args.out or not args.out.resolve().is_relative_to(ART):p.error('output must stay in PT-TF32-2 artifacts')
    gates.verify_manifest();registered();configure_child_adapter()
    if args.child:
        if args.command not in ('state','timing'):p.error('invalid child command')
        return legacy.state_child(args) if args.command=='state' else legacy.timing_child(args)
    args.out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    try:
        result=calibrate(args.out) if args.command=='noise' else state(args) if args.command=='state' else timing(args)
    except Exception as exc:
        result=dict(verdict='RED',error=repr(exc));print(repr(exc),file=sys.stderr)
    result.update(registration_sha256=gates.sha(gates.REG),manifest_sha256=gates.sha(ART/'SOURCE_MANIFEST.json'),
                  elapsed_seconds=time.monotonic()-start)
    gates.create_json(args.out/'summary.json',result);print(json.dumps(result,indent=2))
    return 0 if result['verdict']=='GREEN' else 1


if __name__=='__main__':sys.exit(main())
