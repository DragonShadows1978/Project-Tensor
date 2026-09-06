#!/usr/bin/env python3
"""One registered job per foreground lease. No installed-runtime fallback.

Prior art: standard controlled experiments, content-addressed receipts and
leased process execution. Algorithms/measurements are credited at their code
sites in apa_sp3_model.py and apa_sp3_metrics.py. SP3 orchestration is new.
"""
from __future__ import annotations
import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from apa_sp3_common import (ART, ROOT, REG_SHA, Red, fingerprint, load_runtime, protocol,
                           publish, read, registration, require_pass, sha, upward_float32,
                           verify_sources)
from apa_sp3_a4_registry import KINDS, overlay


def base_cells():
    # Prior art: ordinary interleaved repeats / independent-process controls.
    # Lead amendment 2 (2026) replaces irrecoverable historical-value gates.
    out = [{'id': 'kernel96', 'kind': 'kernel', 'estimate_s': [2, 20], 'depends': []}]
    repeats = []
    for rep in (1,2):
        for arm in ('A','B'):
            name = f'g0_{arm}_{rep}'
            out.append(dict(id=name, kind='baseline', arm=arm, repeat=rep, bits=4, S=1024,
                            estimate_s=[60,480], depends=repeats[-1:]))
            repeats.append(name)
    out += [dict(id='g0', kind='parity', bits=4, S=1024, estimate_s=[1,10], depends=repeats),
            dict(id='ppl_b4_D_1024',kind='ppl',arm='D',bits=4,S=1024,
                 estimate_s=[140,480],depends=['g0','kernel96'])]
    for bits in (4, 8):
        primary = bits == 4
        for S in (1024, 8192):
            for arm in ('A', 'B'):
                if S == 1024 and primary:
                    continue  # G0 contains A/B; these are real cells in that receipt
                out.append(dict(id=f'ppl_b{bits}_{arm}_{S}', kind='ppl', arm=arm, bits=bits, S=S,
                                estimate_s=[40, 200] if S == 1024 else [120, 510], depends=['g0']))
        out.append(dict(id=f'capture_b{bits}_B_1024', kind='capture', bits=bits, arm='B', S=1024,
                        estimate_s=[50, 250], depends=['g0']))
        for layer in range(62):
            out.append(dict(id=f'margin_b{bits}_B_1024_l{layer:02d}', kind='margin', bits=bits,
                            arm='B', S=1024, layer=layer, estimate_s=[3, 60],
                            depends=[f'capture_b{bits}_B_1024']))
        out.append(dict(id=f'calibration_b{bits}', kind='calibration', bits=bits,
                        estimate_s=[1, 5], depends=[f'margin_b{bits}_B_1024_l{i:02d}' for i in range(62)]))
        for i in range(8):
            out.append(dict(id=f'match_b{bits}_{i}', kind='match', bits=bits, trial=i,
                            estimate_s=[45, 250], depends=['g0', 'ppl_b4_D_1024', f'calibration_b{bits}'] +
                            ([f'match_b{bits}_{i-1}'] if i else [])))
        out.append(dict(id=f'freeze_b{bits}', kind='freeze', bits=bits, estimate_s=[1, 5],
                        depends=[f'match_b{bits}_{i}' for i in range(8)]))
        for S in (1024, 8192):
            for arm in ('C', 'D'):
                if primary and S == 1024 and arm == 'D':
                    continue
                out.append(dict(id=f'ppl_b{bits}_{arm}_{S}', kind='ppl', arm=arm, bits=bits, S=S,
                                estimate_s=[45, 180] if S == 1024 else [120, 510],
                                depends=['g0', 'ppl_b4_D_1024', f'freeze_b{bits}'] +
                                ([f'ppl_b{bits}_A_{S}'] if arm=='D' and S==8192 else [])))
            for arm in ('B', 'C'):
                if S == 1024 and arm == 'B':
                    continue
                out.append(dict(id=f'capture_b{bits}_{arm}_{S}', kind='capture', bits=bits, arm=arm, S=S,
                                estimate_s=[50, 250] if S == 1024 else [180, 510],
                                depends=['g0', 'ppl_b4_D_1024', f'freeze_b{bits}']))
                for layer in range(62):
                    out.append(dict(id=f'margin_b{bits}_{arm}_{S}_l{layer:02d}', kind='margin', bits=bits,
                                    arm=arm, S=S, layer=layer, estimate_s=[3,60] if S==1024 else [30,510],
                                    depends=[f'capture_b{bits}_{arm}_{S}']))
        for S in (2048, 8192, 32768):
            for arm in ('A', 'B', 'C'):
                out.append(dict(id=f'decode_b{bits}_{arm}_{S}', kind='decode', bits=bits, arm=arm, S=S,
                                estimate_s=[60,510], depends=['g0', 'ppl_b4_D_1024', f'freeze_b{bits}']))
    # E is bulk4 primary only, conditioned on ALL B/C real-activation margins.
    out.append(dict(id='eq_b4', kind='eq', bits=4, estimate_s=[1,10],
                    depends=[f'margin_b4_{arm}_{S}_l{i:02d}' for arm in ('B','C')
                             for S in (1024,8192) for i in range(62)]))
    for S in (1024,8192):
        out.append(dict(id=f'ppl_b4_E_{S}', kind='ppl', arm='E', bits=4, S=S,
                        estimate_s=[45,180] if S==1024 else [120,510],
                        depends=['g0', 'ppl_b4_D_1024', 'eq_b4']))
        out.append(dict(id=f'capture_b4_E_{S}',kind='capture',arm='E',bits=4,S=S,
                        estimate_s=[50,250] if S==1024 else [180,510],
                        depends=[f'ppl_b4_E_{S}','eq_b4']))
        for layer in range(62):
            out.append(dict(id=f'margin_b4_E_{S}_l{layer:02d}',kind='margin',bits=4,
                            arm='E',S=S,layer=layer,estimate_s=[3,60] if S==1024 else [30,510],
                            depends=[f'capture_b4_E_{S}']))
    out.append(dict(id='eq_check_E',kind='eq_check',bits=4,estimate_s=[1,10],
                    depends=['eq_b4']+[f'margin_b4_E_{S}_l{i:02d}' for S in (1024,8192) for i in range(62)]))
    for x in out:
        if x['kind'] in ('ppl','match','capture','decode'):
            # Includes the retained in-process guard: twelve 1024 prefills
            # plus the requested arm, before the unchanged 480s deadline.
            x['estimate_s']=[max(x['estimate_s'][0],140),480]
        x['estimate_s'][1]=min(x['estimate_s'][1],480)
        x['optional_secondary'] = x.get('bits') == 8
        x['worker_timeout_s'] = 480
        x['job_ceiling_s'] = 590
        x['estimate_scope'] = 'planning estimate, unmeasured; 8192/32768 may OOM or exceed ceiling'
    return [c for c in out if not c['optional_secondary']] + [c for c in out if c['optional_secondary']]


def cells():
    return overlay(base_cells())


def g0_guard(model, ids):
    """Retain r1's in-process control, now against fresh PROTOCOL-2 baselines."""
    from apa_sp3_model import score_windows
    baselines = require_pass('g0')['result']
    numbers = {}
    for arm in ('A', 'B'):
        model.set(arm, bits=4)
        numbers[arm] = score_windows(model, ids)
    targets = {a:baselines[a]['ppl'] for a in ('A','B')}
    # Preserve the numbers BEFORE throwing: a miss must not disappear into an exception.
    publish(ART / 'progress' / f'g0_inprocess.{os.getpid()}.{time.time_ns()}.json',
            {'evidence_class': 'model perplexity', 'numbers': numbers, 'targets': targets,
             'protocol_sha256': sha(ART/'protocol_amendment.json')})
    if any(not math.isfinite(numbers[a]['ppl']) or abs(numbers[a]['ppl']-targets[a]) > .001
           or numbers[a]['target_sha256'] != baselines[a]['target_sha256'] for a in ('A','B')):
        raise Red('G0_DETERMINISM_MISS: ' + json.dumps({a:numbers[a]['ppl'] for a in numbers}) +
                  '; fresh baseline targets '+json.dumps(targets)+' tolerance=0.001; stop all PPL arms', numbers)
    return numbers


def g0_repeats():
    # Prior art: independent-process determinism check, standard experimental
    # control. Do not average repeated PPL to conceal a determinism miss.
    proto, ids = protocol()
    names = proto['g0']['order'][:-1]
    rr = {n:require_pass(n) for n in names}
    identities = [r['process_identity'] for r in rr.values()]
    fresh = len({json.dumps(p,sort_keys=True) for p in identities}) == 4
    numbers = {}
    for a in ('A','B'):
        first, second = (rr[f'g0_{a}_{i}']['result'] for i in (1,2))
        for i,r in enumerate((first,second),1):
            if r.get('arm') != a or r.get('repeat') != i or r.get('targets') != 3072:
                raise Red('G0_DETERMINISM_MISS: incomplete or wrong-arm repeat',rr)
        numbers[a] = first
        numbers[a+'_repeat_2'] = second
    import hashlib
    target_sha = hashlib.sha256(np.concatenate([ids[w*1024+512:(w+1)*1024]
                                for w in range(6)]).astype('<i8').tobytes()).hexdigest()
    diffs = {a:numbers[a+'_repeat_2']['ppl']-numbers[a]['ppl'] for a in ('A','B')}
    result = dict(evidence_class='model perplexity',**numbers,repeat_differences=diffs,
                  fresh_processes=fresh,process_identities=identities,
                  protocol_source=proto['scoring_source_path'],baseline='repeat 1',
                  tolerance=.001,B_minus_A=numbers['B']['ppl']-numbers['A']['ppl'])
    result['B_minus_A_prediction_within_0_3'] = abs(result['B_minus_A']) <= .3
    result['B_minus_A_is_gate'] = False
    if (not fresh or any(not math.isfinite(v) or abs(v)>.001 for v in diffs.values())
            or any(r['result'].get('target_sha256') != target_sha for r in rr.values())):
        raise Red('G0_DETERMINISM_MISS: fresh-process/target/repeat check failed',result)
    return result


def kernel96():
    """GPU execution pin; CPU tests use the independent materialized reference."""
    tc = load_runtime()
    import _apa_sp3_diag as diag
    os.environ['TC_APA_SP'] = '1'
    os.environ['TC_APA_SELECTIVE_PATH'] = '0'
    from apa_sp3_model import last512  # ensure plumbing imports without loading weights
    rng = np.random.default_rng(20260906)
    rows = []
    for L,S in ((7,11),(1,4097)):
        for dtype in ('float32','float16','bfloat16'):
            arrays = [rng.normal(size=shape).astype(np.float32)*.2 for shape in
                      ((1,2,L,96),(1,2,S,96),(1,2,S,96),(1,2,S,96))]
            arrays[3][...,64:] = 0
            q,k,kq,v = [tc.tensor(x).astype(dtype) for x in arrays]
            out,mask = tc._C.apa_selective_attention_sp(q,k,kq,v,96**-.5,
                          float(np.finfo(np.float32).max), True, None, True)
            qt,kt,vt = [x.float().numpy() for x in (q,k,v)]
            scores = (qt.astype(np.float64) @ kt.astype(np.float64).swapaxes(-1,-2))*96**-.5
            valid = np.arange(S)[None,:] < (S-L+np.arange(L)+1)[:,None]
            scores = np.where(valid, scores, -np.inf)
            weights = np.exp(scores-scores.max(-1,keepdims=True))
            expected = (weights/weights.sum(-1,keepdims=True)) @ vt.astype(np.float64)
            tol = .001 if dtype=='float32' else .02
            np.testing.assert_allclose(out.float().numpy(), expected, atol=tol, rtol=tol)
            np.testing.assert_array_equal(mask.numpy(), np.broadcast_to(valid,(1,2,L,S)))
            # Native score probe must reconstruct native SP decisions, including
            # split-K local prefix resets. Known floating-point boundary risk
            # is a hard failure, never a tolerance on the boolean mask.
            trace=diag.bulk_scores(q,kq,96**-.5).numpy()
            _,chosen=tc._C.apa_selective_attention_sp(q,k,kq,v,96**-.5,.125,True,None,True)
            expected_mask=np.zeros(trace.shape,bool)
            for start in range(0,S,2048 if L==1 else S):
                b=trace[...,start:start+(2048 if L==1 else S)]
                expected_mask[...,start:start+b.shape[-1]]=b >= np.maximum.accumulate(b,axis=-1)-np.float32(.125)
            expected_mask &= valid
            np.testing.assert_array_equal(chosen.numpy(),expected_mask)
            if L>1:
                native=tc.apa_selective_attention(q,k,kq,v,96**-.5,1.2815515655446004,True)
                copied,bmask=diag.selective(q,k,kq,v,96**-.5,1.2815515655446004,True)
                np.testing.assert_array_equal(native.numpy(),copied.numpy())
                bulk=tc.matmul(q,kq.transpose(-2,-1))*96**-.5
                rank=tc.matmul(q,k.transpose(-2,-1))*96**-.5
                native=tc.apa_blend_softmax(bulk,rank,1.2815515655446004,L,0,0)
                copied,bmask=diag.blend(bulk,rank,1.2815515655446004,L,0)
                np.testing.assert_array_equal(native.numpy(),copied.numpy())
            rows.append({'L':L,'S':S,'D':96,'VD':96,'dtype':dtype,'status':'PASS'})
    return {'evidence_class':'kernel sweep','pins':rows,
            'scope':'D=96 prefill/split-K numeric and diagnostic pin; no model quality'}


def execute(cell):
    kind, bits = cell['kind'], cell.get('bits',4)
    if kind in KINDS:
        from apa_sp3_a4_jobs import execute as execute_a4
        return execute_a4(cell)
    for d in cell['depends']:
        require_pass(d)
    if kind == 'kernel':
        return kernel96()
    if kind == 'parity':
        return g0_repeats()  # aggregation only; never creates a model
    if kind == 'margin':
        from apa_sp3_metrics import analyze_capture
        cap = ART/'captures'/f"b{bits}_{cell['arm']}_{cell['S']}"/f"layer{cell['layer']:02d}"
        return analyze_capture(cap, ART/'scratch'/f"{cell['id']}.{os.getpid()}.f32")
    if kind == 'calibration':
        from apa_sp3_metrics import initial_delta
        rr = [require_pass(d)['result'] for d in cell['depends']]
        target = sum(r['selected'] for r in rr)/sum(r['pairs'] for r in rr)
        delta, estimated = initial_delta(rr,target)
        return {'target_fraction':target,'initial_delta':delta,'initial_predicted_fraction':estimated,
                'scope':'initializer from B activations; actual C match required'}
    if kind == 'freeze':
        matches = [require_pass(d)['result'] for d in cell['depends']]
        accepted = [m for m in matches if m['matched']]
        if not accepted:
            raise Red('C_FRACTION_MISS: 8 trials exhausted; no threshold relaxation')
        return accepted[0]
    if kind == 'eq':
        # Prior art: SP2 (2026) widens the BLASST (Yuan 2025/2026)
        # log-ratio threshold by 2*eq. SP3 measures a finite model envelope.
        rows = [require_pass(d)['result'] for d in cell['depends']]
        eq = upward_float32(max(r['sp_error']['max'] for r in rows))
        return {'eq':eq,'delta':upward_float32(math.log(1000)+2*eq),'epsilon':.001,
                'layers_and_lengths':len(rows),'scope':'finite B/C activation envelope; E transfer unverified'}
    if kind == 'eq_check':
        envelope=require_pass('eq_b4')['result']['eq']
        actual=max(require_pass(d)['result']['sp_error']['max'] for d in cell['depends'] if d!='eq_b4')
        result={'calibration_eq':envelope,'E_eq':actual,'held':actual<=envelope,
                'scope':'finite E activations only; not a universal certificate or CUDA rounding proof'}
        if not result['held']:
            raise Red('E_EMPIRICAL_ENVELOPE_VIOLATED',result)
        return result
    # Every model job fails closed on protocol pins before CUDA load.
    proto, ids = protocol()
    from apa_sp3_model import Model, capture_space, score_windows
    if kind == 'match' and cell['trial']:
        prev = require_pass(f"match_b{bits}_{cell['trial']-1}")['result']
        if prev['matched']:
            return dict(prev, carried_without_gpu=True)
    model = Model()
    if kind == 'baseline':
        model.set(cell['arm'],bits=4)
        return dict(score_windows(model,ids),arm=cell['arm'],repeat=cell['repeat'],
                    evidence_class='model perplexity',protocol_source=proto['scoring_source_path'])
    parity = g0_guard(model, ids)  # A/B in this very process, before any SP arm.
    if kind == 'match':
        from apa_sp3_metrics import next_delta
        cal = require_pass(f'calibration_b{bits}')['result']
        previous = [require_pass(f'match_b{bits}_{i}')['result'] for i in range(cell['trial'])]
        delta = next_delta(previous,cal['target_fraction']) if previous else cal['initial_delta']
        model.set('C',bits,delta,observe=True)
        m = model.forward(ids[:1024],score=False)['refinement']
        return {'delta':delta,'fraction':m['fraction'],'target_fraction':cal['target_fraction'],
                'matched':abs(m['fraction']-cal['target_fraction'])<=.01,'trial':cell['trial'],
                'per_layer':m,'selection_used_ppl':False,'inprocess_g0':parity}
    arm,S=cell['arm'],cell['S']
    delta=None
    if arm=='C': delta=require_pass(f'freeze_b{bits}')['result']['delta']
    if arm=='D': delta=float(np.finfo(np.float32).max)
    if arm=='E': delta=require_pass('eq_b4')['result']['delta']
    if kind=='capture':
        space=capture_space(S)
        dest=ART/'captures'/f'b{bits}_{arm}_{S}'
        model.set(arm,bits,delta,observe=True,capture=dest)
        measured=model.forward(ids[:S],score=False)
        return dict(measured,arm=arm,S=S,bits=bits,delta=delta,disk_estimate=space,
                    capture=str(dest),evidence_class='kernel sweep',inprocess_g0=parity)
    model.set(arm,bits,delta)
    if kind=='decode':
        return dict(model.decode(ids,S,delta),arm=arm,S=S,bits=bits,delta=delta,
                    evidence_class='kernel sweep',inprocess_g0=parity)
    result=score_windows(model,ids,S)
    if arm in ('C','D','E'):
        # Counts measured outside timing on another identical-token forward.
        model.set(arm,bits,delta,observe=True)
        observed=model.forward(ids[:S],score=False)
        result['refinement']=observed['refinement']
        result['refinement_scope']='registered single prefix at token 0; separate from six-window PPL'
    if arm=='D':
        if result['refinement']['fraction']!=1.:
            raise Red('D_NOT_REFINE_ALL')
        if S==1024:
            result['D_minus_A']=result['ppl']-require_pass('g0')['result']['A']['ppl']
            if abs(result['D_minus_A'])>.005:
                publish(ART/'progress'/f'D_miss.{os.getpid()}.{time.time_ns()}.json',result)
                raise Red('D_NE_A: '+json.dumps(result)+'; stop SP model arms',result)
        else:
            a=require_pass(f'ppl_b{bits}_A_{S}')['result']
            if abs(result['ppl']-a['ppl'])>.005:
                raise Red('D_NE_A_LONG: '+str(result['ppl']-a['ppl']))
    if arm=='C' and S==1024:
        target=require_pass(f'freeze_b{bits}')['result']['target_fraction']
        if abs(result['refinement']['fraction']-target)>.01:
            raise Red('C_FRACTION_MISS_ON_SCORED_FORWARD')
    return dict(result,arm=arm,S=S,bits=bits,delta=delta,evidence_class='model perplexity',
                inprocess_g0=parity,eq_scope='finite B/C envelope, unverified on E' if arm=='E' else None)


def work(job):
    all_cells={c['id']:c for c in cells()}
    if job not in all_cells:
        raise Red('unknown cell')
    if os.environ.get('APA_SP3_LEASE')!='1':
        raise Red('invoke through leased shell runner')
    from apa_sp3_common import job_path, cell_fingerprint
    from apa_sp3_a4_provenance import bridge
    dest=job_path(job)
    if dest.exists():
        raise Red('job receipt already exists; use summary/resume; no automatic RED retry')
    cell=all_cells[job]
    start=time.perf_counter()
    receipt={'job':job,'cell':cell,'registration_sha256':REG_SHA,'fingerprint':cell_fingerprint(cell),
             'fingerprint_schema':'apa_sp3_per_kind_v1',
             'fingerprint_amendment_sha256':bridge()['effective_sha256'],
             'protocol_sha256':sha(ART/'protocol_amendment.json') if (ART/'protocol_amendment.json').exists() else None,
             'status':'RED','pid':os.getpid(),
             'process_identity':dict(pid=os.getpid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                                     start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
             'evidence_class':'model perplexity' if cell['kind'] in ('ppl','parity','baseline') else 'kernel sweep'}
    try:
        verify_sources()
        receipt['dependencies']={d:sha(job_path(d)) for d in cell['depends'] if job_path(d).exists()}
        from apa_sp3_common import receipt_validation
        with receipt_validation():
            receipt['result']=execute(cell)
        receipt['status']='PASS'
    except Exception as e:
        receipt['error']=f'{type(e).__name__}: {e}'
        receipt['traceback']=traceback.format_exc()
        if isinstance(e,Red) and e.details is not None:
            receipt['result']=e.details
        if isinstance(e,Red) and str(e).startswith(('G0_DETERMINISM_MISS','G0_PARITY_MISS','D_NE_A','D_NOT_REFINE_ALL')):
            stop=ART/'STOP_MODEL_ARMS.json'
            if not stop.exists():
                publish(stop,{'job':job,'error':str(e),'registration_sha256':REG_SHA,
                              'rule':'G0 or D hard rail; no later model jobs under this registration'})
    receipt['wall_s']=time.perf_counter()-start
    publish(dest,receipt)
    print(json.dumps({'job':job,'status':receipt['status'],'error':receipt.get('error')}),flush=True)
    return 0 if receipt['status']=='PASS' else 1


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--worker')
    p.add_argument('--next',action='store_true')
    p.add_argument('--bits',type=int,choices=(4,8),default=4)
    a=p.parse_args()
    if a.dry_run:
        proto, _ = protocol()
        print(json.dumps({'status':'PLANNED_NOT_RUN','registration_sha256':REG_SHA,
                          'protocol':proto,'cells':cells()},indent=2))
        return 0
    if a.next:
        for c in cells():
            if c.get('bits',4)!=a.bits:
                continue
            from apa_sp3_common import job_path
            path=job_path(c['id'])
            if path.exists():
                require_pass(c['id'])  # RED/stale is a stop, not a resume skip
                continue
            print(c['id'])
            return 0
        return 0
    if a.worker:
        return work(a.worker)
    p.error('use --dry-run, --worker or --next')


if __name__=='__main__':
    raise SystemExit(main())
