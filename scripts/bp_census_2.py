#!/usr/bin/env python3
"""BP-KERNEL-2 real step: instrumented a then f; no OFF repeat.

Prior art: BP-CENSUS-1 (2026) driver/profiler and CPU adapter (taken); gprof
(Graham/Kessler/McKusick 1982), CUDA events (NVIDIA 2007+), AdamW (Loshchilov
and Hutter 2019), checkpointing/PyTorch (Paszke et al. 2019), taken through
unchanged GRAPA. Ours: explicit native variant arms and gate provenance.
Unverified — lead to check these authors/titles; no network in this seat.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
import argparse
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import time
import uuid
import numpy as np
import bp_kernel_2 as kernel

ROOT=kernel.ROOT
ART=kernel.ART/'census'
REG=ART/'registration.json'
ENGINE=ROOT/'tensor_cuda'
GRAPA=Path('/mnt/ForgeRealm/wt/grapa-bp1')
PARENT_ART=GRAPA/'artifacts/bp_census_1'
sys.path.insert(0,str(GRAPA/'scripts'))
spec=importlib.util.spec_from_file_location('bp_census_1',GRAPA/'scripts/bp_census_1.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
CONFIG=dict(parent.CONFIG,arm_order=['a','f'])
sha,create_json=kernel.sha,kernel.create_json


def register():
    kr=kernel.verify_registration()
    pr=json.loads((PARENT_ART/'registration.json').read_text())
    pins={p:h for p,h in pr['pins'].items() if Path(p).is_relative_to(GRAPA)}
    pins[str(PARENT_ART/'registration.json')]=sha(PARENT_ART/'registration.json')
    pins[str(PARENT_ART/'receipt.json')]=sha(PARENT_ART/'receipt.json')
    for p,h in pins.items():
        if sha(p)!=h: raise ValueError('parent source drift: '+p)
    if sha(parent.CHECKPOINT)!=pr['checkpoint']['sha256']: raise ValueError('checkpoint drift')
    create_json(REG,dict(schema_version=1,experiment='BP-CENSUS-2',config=CONFIG,budget=kernel.BUDGET,
        prediction=kernel.STEP_PREDICTION,falsifier=kernel.STEP_FALSIFIER,
        kernel_registration_sha256=sha(kernel.REG),pins=pins,checkpoint=pr['checkpoint'],tokenizer=pr['tokenizer'],
        tokens_path=str(PARENT_ART/'tokens.npy'),tokens_sha256=sha(PARENT_ART/'tokens.npy'),
        rowdot_rule='Use cell 1 recorded f route; if missing due to incomplete cell use f_pass_a, TIMING-ONLY / NOT A VALID STEP.',
        gate_rule='f kernel gate RED or incomplete => run f arm TIMING-ONLY / NOT A VALID STEP; no valid-step prediction claim.',
        off_control=dict(repeated=False,parent_overhead_percent=1.3,evidence=str(PARENT_ART/'receipt.json')),
        engine_binary=kr['engine_binary'],scope='same checkpoint/tokens/config as parent; 2 warmups + 5 measured per arm; a then f'))
    REG.with_suffix('.sha256').write_text(sha(REG)+'\n')
    print(f'{REG} sha256={sha(REG)}')


def verify():
    kernel.verify_registration()
    if sha(REG)!=REG.with_suffix('.sha256').read_text().strip(): raise ValueError('census registration drift')
    r=json.loads(REG.read_text())
    if r['config']!=CONFIG or r['budget']!=kernel.BUDGET or r['kernel_registration_sha256']!=sha(kernel.REG) or r['prediction']!=kernel.STEP_PREDICTION or r['falsifier']!=kernel.STEP_FALSIFIER:
        raise ValueError('census protocol drift')
    for p,h in r['pins'].items():
        if sha(p)!=h: raise ValueError('census pin drift: '+p)
    if sha(parent.CHECKPOINT)!=r['checkpoint']['sha256'] or parent.checkpoint_metadata()!=r['checkpoint']['metadata']:
        raise ValueError('checkpoint drift')
    return r


def gate_source(dry):
    if dry: return dict(eligible=False,route='f_pass_a',label='CPU plumbing only',receipt_sha256=None)
    path=kernel.ART/'receipt.json'
    r=kernel.validate_receipt(json.loads(path.read_text()))
    if r['registration_sha256']!=sha(kernel.REG) or r['reference_sha256']!=sha(kernel.REFERENCE) or r['mode']!='run':
        raise ValueError('kernel gate provenance drift')
    route=r['rowdot_route'] or 'f_pass_a'
    green=(r['correctness'] or {}).get('candidates',{}).get('f',{}).get('verdict')=='GREEN'
    eligible=green and r['verdict'] in ('CONFIRMED','SCALAR_DOTS_NEXT')
    return dict(eligible=eligible,route=route,receipt_sha256=sha(path),kernel_verdict=r['verdict'],
        label='kernel-gated step timing; no model-quality claim' if eligible else 'TIMING-ONLY / NOT A VALID STEP')


def run_steps(dry,reg,gate,emit):
    # Prior art: BP-CENSUS-1 run_steps (2026), copied thin driver; same model,
    # checkpoint loader, optimizer, scopes, tokens, cleanup and 2+5 samples.
    # Ours: both arms instrumented and explicit engine setter per arm.
    prof=parent.Profiler(dry)
    sys.path.insert(0,str(GRAPA))
    if dry:
        from bp_census_1_cpu import install
        import torch
        torch.set_num_threads(2)
        tc=install(ENGINE/'tensor_cuda',prof)
    else:
        os.environ['TC_OP_TIMING']='1'
        sys.path.insert(0,str(ENGINE))
        import tensor_cuda as tc
        if not Path(tc.__file__).resolve().is_relative_to(ENGINE): raise ValueError('wrong engine import')
        if sha(tc._C.__file__)!=reg['engine_binary']['sha256']: raise ValueError('engine binary drift')
        import re
        fingerprint=re.search(r'source_pin = "([a-f0-9]+)"',(ENGINE/'include/tc/bp_op_timing.h').read_text())[1]
        if tc._C.bp_source_pin()!=fingerprint: raise ValueError('timing hook drift')
        emit(dict(kind='engine_binary',path=tc._C.__file__,sha256=sha(tc._C.__file__),fingerprint=fingerprint))
    prof.tc=tc
    train=importlib.import_module('grapa.train')
    if train.tc is not tc: raise ValueError('GRAPA engine import drift')
    tc.set_alloc_pooling(True)
    cfg=train.MLAConfig200();cfg.window=2048;cfg.compute_dtype='bfloat16'
    if dry:
        for k,v in dict(d_model=16,n_layers=2,n_heads=2,q_lora_rank=8,kv_lora_rank=4,
                        qk_nope_dim=4,qk_rope_dim=2,v_head_dim=4,d_ff=24,vocab=32,window=8,head_dim=2).items(): setattr(cfg,k,v)
    tokens=np.load(reg['tokens_path'],allow_pickle=False)
    if dry: tokens=tokens[:,:cfg.window+1]%cfg.vocab
    x,y=tokens[:,:-1],tokens[:,1:]
    for arm in CONFIG['arm_order']:
        selected='a' if arm=='a' else gate['route']
        if not dry:
            tc._C.bp_kernel_2_set_variant(selected)
            if tc._C.bp_kernel_2_get_variant()!=selected: raise ValueError('variant setter failed')
        np.random.seed(CONFIG['seed']);model=train.GRAPA_MLA(cfg)
        if dry: opt=train.AdamW(model.parameters(),lr=CONFIG['lr'])
        else:
            opt,meta=train.checkpoint.load(str(parent.CHECKPOINT),model,train.AdamW,{'lr':CONFIG['lr']})
            if meta!=reg['checkpoint']['metadata']: raise ValueError('checkpoint metadata mismatch')
        if any(p.dtype!='float32' for p in model.parameters()): raise ValueError('non-FP32 masters')
        emit(dict(kind='arm_start',arm=arm,variant=selected,optimizer_t=opt.t,lr=opt.lr,
            label='control' if arm=='a' else gate['label'],
            config={k:getattr(cfg,k) for k in ('window','n_layers','d_model','vocab','compute_dtype','checkpoint_blocks')}))
        for index in range(7):
            prof.configure(True);tc.synchronize()
            if not dry: tc._C.bp_wall_begin()
            start=time.perf_counter()
            with prof.region('step'):
                with prof.region('input_and_zero_grad'):
                    opt.zero_grad();target=tc.tensor(y,dtype='int64')
                with prof.region('initial_forward'):
                    logits=model(x,refine_percentile=.15);loss=train.nll_loss(logits,target)
                with prof.region('backward'): loss.backward()
                with prof.region('adam_update'): opt.step()
                with prof.region('cleanup'):
                    opt.zero_grad();del logits,loss,target;tc.empty_cache()
            tc.synchronize();wall_ms=(time.perf_counter()-start)*1000
            device_ms=None if dry else float(tc._C.bp_wall_end())
            rows=prof.get_rows();table,backward,detail=parent.component_table(rows,device_ms or wall_ms)
            emit(dict(kind='step',arm=arm,index=index,warmup=index<2,wall_ms=wall_ms,device_ms=device_ms,
                components=table,non_replay_backward_ms=backward,named_scopes_ms=detail,
                allocation=parent.allocation(rows,dry),raw_scopes=rows))
        del opt,model;tc.empty_cache()
    if not dry: tc._C.bp_kernel_2_set_variant('a')


def summarize(events,reg,gate,dry,elapsed,error=None):
    steps={v:[e for e in events if e.get('kind')=='step' and e['arm']==v and not e['warmup']] for v in ('a','f')}
    warmups={v:[e for e in events if e.get('kind')=='step' and e['arm']==v and e['warmup']] for v in ('a','f')}
    complete=all(len(steps[v])==5 and len(warmups[v])==2 for v in steps)
    means={v:statistics.mean(s['wall_ms'] for s in data) if data else None for v,data in steps.items()}
    observed=complete and all(math.isfinite(s['wall_ms']) and s['wall_ms']>0 and s['components']['native_apa_backward']['ms']>0 for data in steps.values() for s in data)
    reproduces=complete and 11800*.9<=means['a']<=11800*1.1
    verdict='INCONCLUSIVE'
    if dry and complete and not error: verdict='DRY_RUN'
    elif complete and observed and not error:
        if not gate['eligible']: verdict='TIMING_ONLY_NOT_A_VALID_STEP'
        elif not reproduces: verdict='A_REPRODUCTION_FAILED'
        else: verdict='CONFIRMED' if means['f']<=5000 else 'STEP_GT_5S'
    components={v:{k:statistics.mean(s['components'][k]['ms'] for s in data) for k in parent.COMPONENTS} for v,data in steps.items() if data}
    return validate_receipt(dict(schema_version=1,experiment='BP-CENSUS-2',mode='dry-run' if dry else 'run',
        evidence_class='CPU plumbing only' if dry else 'bounded native training-step census',registration_sha256=sha(REG),
        gate_source=gate,config=CONFIG,budget=kernel.BUDGET,checkpoint=reg['checkpoint'],tokens_sha256=reg['tokens_sha256'],
        off_control=reg['off_control'],steps=steps,warmups=warmups,mean_wall_ms=means,a_reproduces_within_10_percent=reproduces,
        component_mean_ms=components,largest_f_component=max(components['f'],key=components['f'].get) if 'f' in components else None,
        arm_starts=[e for e in events if e.get('kind')=='arm_start'],
        engine_binary=next((e for e in events if e.get('kind')=='engine_binary'),None),
        verdict=verdict,error=error,elapsed_seconds=elapsed,
        residuals=['OFF control not repeated; BP-CENSUS-1 overhead was 1.3%.',
                   'Kernel gate uses seeded inputs; does not validate checkpoint gradients or training quality.',
                   'Boundary allocation samples are not exact high-water measurements.']))


def validate_receipt(r):
    if r['schema_version']!=1 or r['experiment']!='BP-CENSUS-2': raise ValueError('census schema')
    if r['verdict'] not in ('INCONCLUSIVE','DRY_RUN','TIMING_ONLY_NOT_A_VALID_STEP','A_REPRODUCTION_FAILED','CONFIRMED','STEP_GT_5S'): raise ValueError('census verdict')
    if r['mode']=='dry-run' and r['verdict'] not in ('DRY_RUN','INCONCLUSIVE'): raise ValueError('CPU native verdict')
    if r['verdict']!='INCONCLUSIVE':
        for v in ('a','f'):
            if len(r['steps'][v])!=5 or len(r['warmups'][v])!=2: raise ValueError('incomplete census')
            for s in r['steps'][v]:
                if set(s['components'])!=set(parent.COMPONENTS): raise ValueError('component schema')
    json.dumps(r,allow_nan=False);return r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('register','dry-run','run','worker'): p.add_argument('--'+flag,action='store_true')
    args=p.parse_args()
    if args.register: register();return
    reg=verify()
    if args.dry_run:
        gate=gate_source(True);events=[];start=time.monotonic();run_steps(True,reg,gate,events.append)
        r=summarize(events,reg,gate,True,time.monotonic()-start)
        path=ART/('dry_'+uuid.uuid4().hex)/'receipt.json';path.parent.mkdir()
        create_json(path,r);print(f'{r["verdict"]} {path}');return
    if args.worker:
        if os.environ.get('BP_KERNEL_2_WORKER')!='1': raise ValueError('supervisor required')
        gate=json.loads((ART/'gate_input_registration.json').read_text())
        if gate!=gate_source(False): raise ValueError('cell 1 gate drift')
        def emit(e): print(json.dumps(e,allow_nan=False),flush=True)
        emit(dict(kind='gpu',description=kernel.gpu_preflight()));run_steps(False,reg,gate,emit);return
    if not args.run: raise ValueError('choose --dry-run or --run')
    gate=gate_source(False)
    create_json(ART/'gate_input_registration.json',gate)
    kernel.supervise(Path(__file__).resolve(),ART,sha(REG),lambda events,error,elapsed:summarize(events,reg,gate,False,elapsed,error))

if __name__=='__main__': main()
