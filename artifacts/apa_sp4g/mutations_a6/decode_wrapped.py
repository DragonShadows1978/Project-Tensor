"""A6 foreground cells. Prior art: SP4G A4/A5 (2026) leased DAG,
residual sum-of-squares, native population replay and clean decode;
classical bisection. New ruling dependencies only, no kernel changes.
"""
import math, os, shutil, subprocess, sys, time, traceback
from contextlib import nullcontext
import numpy as np
from apa_sp4g_a6_common import *
from apa_sp4g_a6_registry import cells, CPU_KINDS
from apa_sp4g_a5_gpu import deadline_guard
from apa_sp4g_gpu import choose_trial
from apa_sp4g_metrics import upward


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise Red('A6_COOPERATIVE_WORKER_RAIL')


def compare_residuals(reference, treatment, S=2048, deadline=float('inf')):
    # SP4G A4 (2026) aggregation: global sum of squared errors, reference A
    # denominator, ALL executed rows; not an average of block ratios.
    from apa_sp4g_a4_model import LAYERS, validate_coverage
    ma, mt = [read(R/r['manifest']) for r in (reference, treatment)]
    if (ma['arm'], mt['arm'], ma['S'], mt['S']) != ('A', 'A32', S, S):
        raise Red('A6_RESIDUAL_MANIFEST_ARM_OR_LENGTH')
    rows = []
    for layer in [str(l) for l in LAYERS]+['final_norm']:
        ar, tr = [m['residuals'][layer] for m in (ma, mt)]
        validate_coverage(ar, S); validate_coverage(tr, S)
        if [(r['lo'], r['n']) for r in ar] != [(r['lo'], r['n']) for r in tr]:
            raise Red('A6_PROPAGATION_SCHEDULE_MISMATCH')
        ns = ds = maximum = 0.; blocks = []
        for ra, rt in zip(ar, tr):
            check_deadline(deadline)
            x, y = [np.load(R/r['file']['path'], allow_pickle=False).astype(np.float64) for r in (ra, rt)]
            if (x.shape != y.shape or x.shape != (1, ra['n'], 3840)
                    or not np.isfinite(x).all() or not np.isfinite(y).all()):
                raise Red('A6_PROPAGATION_INVALID_ARRAY')
            err = y-x; n = float(np.sum(err*err)); d = float(np.sum(x*x)); m = float(np.max(np.abs(err)))
            ns += n; ds += d; maximum = max(maximum, m)
            blocks.append(dict(lo=ra['lo'], n=ra['n'], relative_frobenius=math.sqrt(n)/max(math.sqrt(d), 1e-30), max_abs=m))
        rows.append(dict(layer=layer, relative_frobenius=math.sqrt(ns)/max(math.sqrt(ds), 1e-30),
                         max_abs=maximum, rows=S-1, blocks=blocks))
    return rows


def freeze_match(trials, target):
    for name, r in trials:
        if math.isfinite(r['fraction']) and abs(r['fraction']-target) <= .01:
            return dict(delta=r['delta'], fraction=r['fraction'], target=target,
                        match_abs=abs(r['fraction']-target), trial=r.get('carried_from', name),
                        population='actual PPL window0 global-layer query pairs, queries0..2046')
    raise Red('A6_C_MATCH_FAILED_12_REGISTERED_TRIALS')


def measurement_context(owner, kind, deadline):
    # SP4G A5 (2026) cooperative guard. SP3 A6 / June Gemma (2026) clean
    # timing forbids any attention wrapper or trace for decode.
    return deadline_guard(owner, deadline)


def resident_no_signals():
    # Inherited NVIDIA own-PID snapshot, remove subprocess timeout because
    # timeout can signal a process. No signal-based bounding is authorized.
    run = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    rows = [line.split(',') for line in run.stdout.splitlines() if line.strip()]
    mine = [float(x[1]) for x in rows if int(x[0]) == os.getpid()]
    if len(mine) != 1:
        raise Red('A6_OWN_PID_RESIDENCY_UNAVAILABLE')
    return mine[0]


def execute(c):
    if c != by_id().get(c.get('id')) or c['S'] >= 16384:
        raise Red('A6_UNREGISTERED_OR_LONG_RAIL')
    kind = c['kind']; deadline = time.monotonic()+c['worker_s']
    if kind == 'aggregate':
        rows = [require_a6(d)['result'] for d in c['depends']]
        n = sum(r['targets'] for r in rows); loss = sum(r['total_nll'] for r in rows)
        if n != 4096:
            raise Red('A6_SHORT_TARGET_TOTAL')
        pairs = sum(r['global_fraction']['pairs'] for r in rows)
        selected = sum(r['global_fraction']['selected'] for r in rows)
        if pairs <= 0:
            raise Red('A6_EMPTY_SELECTION_POPULATION')
        return dict(ppl=math.exp(loss/n), targets=n, total_nll=loss,
                    global_fraction=dict(pairs=pairs, selected=selected, fraction=selected/pairs))
    if kind == 'freeze':
        target = require_a6('ppl_capture_B_2048_w0')['result']['global_fraction']['fraction']
        return freeze_match([(d, require_a6(d)['result']) for d in c['depends'] if d.startswith('trial_a6_')], target)
    if kind == 'margin':
        from apa_sp4g_a6_metrics import whole_layer
        r = whole_layer(c, deadline)
        check_deadline(deadline)
        return r
    if kind == 'eq':
        rows = [require_a6(d)['result'] for d in c['depends'] if d.startswith('margin_')]
        if len(rows) != 32 or any(not math.isfinite(r['eq_sp']) or r['eq_sp'] < 0 for r in rows):
            raise Red('A6_EQ_SOURCE_COUNT_OR_NONFINITE')
        eq = upward(max(r['eq_sp'] for r in rows))
        return dict(eq=eq, epsilon=.01, delta=upward(math.log(100)+2*eq), source_count=32,
                    calibration='finite empirical maximum over B/C PPL global-layer keys at2048/8192; conditional bound only, not universal')
    delta = None
    if c['arm'] == 'C':
        if kind == 'trial':
            prev = [(d, require_a6(d)['result']) for d in c['depends'] if d.startswith('trial_a6_')]
            target = require_a6('ppl_capture_B_2048_w0')['result']['global_fraction']['fraction']
            choice = choose_trial(prev, target)
            if choice['carry']:
                r = require_a6(choice['carry'])['result']
                return dict(r, carried_from=r.get('carried_from', choice['carry']), no_model_execution=True)
            delta = choice['delta']
        else:
            delta = require_a6('freeze_a6')['result']['delta']
    elif c['arm'] == 'E':
        delta = require_a6('eq_a6')['result']['delta']
    from apa_sp4g_a6_model import A32ResidualModel, CaptureModel
    import apa_sp4g_model as base
    original_resident = base.resident; base.resident = resident_no_signals
    owner = None
    try:
        if kind == 'propagation_check':
            owner = A32ResidualModel(c)
        elif kind == 'ppl_capture':
            owner = CaptureModel(c, delta)
        else:
            owner = base.Model(c, delta, observe=kind in ('trial', 'ppl'))
        check_deadline(deadline)
        with measurement_context(owner, kind, deadline):
            ids = tokens(); start = c['window']*2048 if c['S'] == 2048 else 0
            if kind == 'decode':
                r = owner.decode(ids)
                check_deadline(deadline)
                return r
            r = owner.perplexity(ids[start:start+c['S']], 1024 if c['S'] == 2048 else 512)
            if kind == 'propagation_check':
                r = owner.finish_residual(owner.finish_precision(r))
                ref32 = require_a6('diag_a2_fp32_A_2048_w0')['result']
                reference = require_a6('diag_a4_propagation_A_2048_w0')['result']
                want = [(p['layer'], p['L'], p['S']) for p in ref32['probes']]
                got = [(p['layer'], p['n'], p['S_all']) for p in r['dtype_pins']]
                if got != want or r['ppl'] != ref32['ppl']:
                    raise Red('A6_A32_SCHEDULE_OR_PPL_REPRODUCTION_FAILED')
                rows = compare_residuals(reference, r, c['S'], deadline)
                r.update(per_layer=rows, decision=propagation_verdict(rows), same_schedule=True,
                         reference_ppl_bitwise=True, A_bf16_ppl=reference['ppl'], A32_ppl=r['ppl'],
                         floor_comparison=floor_comparison(r['ppl'], reference['ppl']),
                         norm_reference='A bf16; sqrt(sum squared error over ALL rows)/sqrt(sum squared A)',
                         evidence_class='standard-fp32 versus standard-bf16 residual diagnostic; observed amplification is not a causal proof or chaos certification')
            elif kind == 'ppl_capture':
                r = owner.finish_capture(r)
            elif kind == 'trial':
                r.update(fraction=r['global_fraction']['fraction'])
            check_deadline(deadline)
            return dict(r, delta=delta, arm=c['arm'], S=c['S'])
    finally:
        if owner is not None:
            owner.close()
        base.resident = original_resident


def preflight(c):
    if c != by_id().get(c.get('id')) or c['S'] >= 16384:
        raise Red('A6_UNREGISTERED_OR_LONG_RAIL')
    preserved(); tokens(); verify_weight()
    gate = read(A/'CPU_GATES_A6.json'); seal = read(SEAL)
    if (gate['status'] != 'PASS_CPU_ONLY' or gate['fingerprint_amendment_sha256'] != sha(SEAL)
            or fingerprint(c) != seal['fingerprint']):
        raise Red('A6_CPU_GATE_OR_FINGERPRINT_REQUIRED')
    for p, h in gate['execution_sha256'].items():
        if sha(R/p) != h:
            raise Red('A6_CPU_GATE_STALE: '+p)
    if path_a6(c['id']).exists():
        require_a6(c['id'])
        return 'DONE'
    for d in c['depends']:
        require_a6(d)
    if shutil.disk_usage(A).free < 12*(1 << 30):
        raise Red('A6_DISK_RAIL_12GIB')
    return 'CPU' if c['kind'] in CPU_KINDS else 'GPU'


def receipt(c, status, result, error=None):
    return dict(cell=c, status=status, registration_sha256=REG_SHA, a6_registration_sha256=REGISTRATION_SHA,
                fingerprint_schema='apa_sp4g_a6_isolated_v1', fingerprint=fingerprint(c),
                dependencies={d: sha(path_a6(d)) for d in c['depends']}, result=result, error=error,
                evidence_class='model perplexity' if c['kind'] in ('ppl', 'ppl_capture', 'aggregate')
                else 'diagnostic / finite calibration / G2-G3; not standalone model quality')


def worker(c):
    decision = preflight(c)
    if decision == 'DONE':
        raise Red('A6_CREATE_ONLY_RECEIPT_EXISTS')
    if decision == 'GPU':
        if os.environ.get('APA_SP4G_LEASE') != '1':
            raise Red('GPU_LEASE_REQUIRED')
        import fcntl
        held, expected = os.fstat(9), os.stat('/tmp/forge-gpu.lock')
        if (held.st_dev, held.st_ino) != (expected.st_dev, expected.st_ino):
            raise Red('GPU_LEASE_WRONG_FILE')
        fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()
    try:
        result = execute(c)
        passed = c['id'] != DIAG or result['decision']['C_E_unblocked']
        j = receipt(c, 'PASS' if passed else 'RED', result, None if passed else result['decision']['stop_reason'])
    except BaseException as error:
        j = receipt(c, 'RED', {}, f'{type(error).__name__}: {error}')
        j['traceback'] = traceback.format_exc(); j['worker_wall_s'] = time.monotonic()-started
        publish(path_a6(c['id']), j)
        raise
    j['worker_wall_s'] = time.monotonic()-started
    publish(path_a6(c['id']), j)
    print(json.dumps(dict(status=j['status'], cell=c['id'], receipt=str(path_a6(c['id']).relative_to(R)),
                          decision=result.get('decision'), ppl=result.get('ppl')), indent=2))
    if not passed:
        raise Red(j['error'])


def main():
    action = sys.argv[1]
    if action == 'list':
        print(json.dumps(cells(), indent=2)); return
    if action == 'next':
        for c in cells():
            if path_a6(c['id']).exists():
                require_a6(c['id']); continue
            preflight(c)
            print(c['id']); return
        return
    if action == 'idle':
        r = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, check=True)
        if r.stdout.strip():
            raise Red('A6_GPU_BUSY: '+r.stdout.strip())
        return
    if action == 'finish':
        c = by_id()[sys.argv[2]]; rc, log = int(sys.argv[3]), sys.argv[4]
        if not path_a6(c['id']).exists():
            publish(path_a6(c['id']), receipt(c, 'RED', dict(returncode=rc, log=log, log_sha256=sha(R/log)),
                                             'worker exited without completed receipt'))
        require_a6(c['id'])
        return
    if len(sys.argv) != 3 or sys.argv[2] not in by_id():
        raise Red('A6_UNKNOWN_CELL')
    c = by_id()[sys.argv[2]]
    if action == 'preflight':
        print(preflight(c))
    elif action == 'worker':
        worker(c)
    else:
        raise Red('A6_UNKNOWN_ACTION')


if __name__ == '__main__':
    token = VALIDATION.set({})
    try:
        main()
    finally:
        VALIDATION.reset(token)
