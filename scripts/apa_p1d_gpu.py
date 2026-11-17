#!/usr/bin/env python3
"""APA-P1-D GPU lines L1-L6 (run only inside forge-gpu-run) and their CPU steps.

Modes (each GPU mode is one registered line; see artifacts/apa_p1d/EXPECTED.json):
  l3                 R-E reconstruction check (Build B)
  l5 [--max-seconds] packed margins, SP2's G2 procedure on packed-r2 bulk (Build B)
  l5-freeze          CPU: freeze the packed table the way SP2's freeze() does
  l1                 default/explicit/opt-out equivalence (children: arm A, arm B)
  l2                 refusals, fallbacks, delta helper vs native, prefill (Build B)
  l4                 bytes and own-process device memory (children per case, Build B)
  l6 --files ...     P150 regression files on arms A and B, per-ID outcomes
  list               CPU: SP2 targets for the 4-bit packed table (L01 instrument check)

Prior art: P1/AB-1 paired unsigned-view bit equality and pytest per-ID receipts
(Project-Tensor, 2026); SP2's G2 measurement procedure (data, stats, freeze rule;
imported from scripts/apa_sp2_gpu.py, not copied); NVIDIA cudaMemGetInfo for
device memory. No new algorithm here.
"""
import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/apa_p1d'
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tensor_cuda/tests'))

DENSE_TABLE = Path('/mnt/ForgeRealm/Project-Tensor/artifacts/apa_sp2/e_q_table.json')
DENSE_TABLE_SHA = '5b6f8bfb405b40c03da21ad8d50eb40c11b85ed9559ee717d2a105c90c0e618d'
SP2_REGISTRATION = Path('/mnt/ForgeRealm/Project-Tensor/artifacts/apa_sp2/registration.json')
SP2_REGISTRATION_SHA = '3b8df8bfdd6e3d351395f96ca654bfe9070a26bfaed390d73cc620546a7ff027'
PACKED_TABLE = ART / 'packed_e_q_table.json'
# Arm B's staged package directory (arms/B, or a restaged arms/B_v<n> after a
# Python-only change; every result records the module and file pins it used).
ARM_B = os.environ.get('APA_P1D_ARM_B', 'B')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False, default=str)
        f.write('\n')


def step(log, label, **fields):
    """Post-device step receipt with a terminal label (L48)."""
    log.append(dict(step=label, t=time.time(), **fields))
    print('STEP', label, json.dumps(fields, default=str)[:300], flush=True)


def load_arm(name):
    name = ARM_B if name == 'B' else name
    arm = json.loads((ART / 'arms' / name / 'ARM.json').read_text())
    package = ROOT / arm['package']
    for path, digest in arm['python_files'].items():
        rel = Path(path).relative_to('tensor_cuda/tensor_cuda')
        assert sha(package / rel) == digest, ('arm python drift', path)
    assert sha(ROOT / arm['module']) == arm['module_sha256']
    sys.path.insert(0, str(package.parent))
    import tensor_cuda as tc
    assert Path(tc.__file__).resolve().parent == package.resolve(), tc.__file__
    assert sha(tc._C.__file__) == arm['module_sha256']
    return tc, arm


def uview(a):
    a = np.ascontiguousarray(a)
    return a.view({1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[a.dtype.itemsize])


def informative(a):
    u = uview(np.asarray(a)).ravel()
    return bool(u.size) and bool(np.any(u != 0)) and bool(np.any(u != u[0]))


def bitwise(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype != b.dtype or a.shape != b.shape:
        return dict(equal=False, reason=f'dtype/shape {a.dtype}{a.shape} vs {b.dtype}{b.shape}')
    return dict(equal=bool(np.array_equal(uview(a), uview(b))), informative=[informative(a), informative(b)],
                dtype=str(a.dtype), shape=list(a.shape))


def dense_margins(tc):
    return tc.RegisteredMargins.load(DENSE_TABLE, expected_sha256=DENSE_TABLE_SHA)


def packed_margins(tc):
    return tc.PackedRegisteredMargins.load(PACKED_TABLE, expected_sha256=tc.apa_packed.PACKED_TABLE_SHA256)


def sp2_registration():
    assert sha(SP2_REGISTRATION) == SP2_REGISTRATION_SHA
    assert (ROOT / 'artifacts/apa_sp2/registration.sha256').read_text().split()[0] == SP2_REGISTRATION_SHA
    return json.loads(SP2_REGISTRATION.read_text())


# ------------------------------------------------------------------ L3
def l3(args):
    tc, arm = load_arm('B')
    log = []
    rows = []
    with tc.no_grad():
        for D in (64, 128):
            for S in (255, 2048):
                for B in (1, 2):
                    rng = np.random.default_rng(np.random.SeedSequence([20261115, 3, D, S, B]))
                    k = rng.standard_normal((B, 2, S, D), dtype=np.float32)
                    kt = tc.tensor(k)
                    dense = tc.quantize_sp2_keys(kt, 4, storage='dense').numpy()
                    p = tc.quantize_sp2_keys(kt, 4, storage='packed')
                    assert isinstance(p, tc.PackedSpKeys)
                    recon = p._dense_reconstruction().numpy()
                    cmp = bitwise(dense, recon)
                    kd = kt.float()
                    norms = (kd * kd).sum([-1], True).pow(0.5).numpy()[..., 0]
                    pnorm = p.norms.numpy()[:, :, :S]
                    row = dict(D=D, S=S, B=B, KV=2, comparison=cmp,
                               mismatched_elements=int(np.count_nonzero(uview(dense) != uview(recon))),
                               elements=int(dense.size),
                               max_abs_diff=float(np.max(np.abs(dense.astype(np.float64) - recon))),
                               norms_bitwise_equal_fraction=float(np.mean(uview(norms) == uview(pnorm))),
                               rows_with_any_mismatch=int(np.count_nonzero(np.any(uview(dense) != uview(recon), -1))),
                               rows=int(B * 2 * S))
                    rows.append(row)
                    step(log, f'l3_D{D}_S{S}_B{B}', equal=cmp['equal'], mismatched=row['mismatched_elements'])
                    del kt, p
    all_equal = all(r['comparison']['equal'] and all(r['comparison']['informative']) for r in rows)
    result = dict(line='L3', arm=arm['arm'], module_sha256=arm['module_sha256'], cases=rows,
                  all_bitwise_equal=all_equal,
                  r_e_branch='transient_reconstruction' if all_equal else 'refuse_prefill_on_packed',
                  steps=log, terminal='L3_DONE')
    save(Path(args.out) / 'RESULT.json', result)
    print('L3 all_bitwise_equal', all_equal)
    return 0


# ------------------------------------------------------------------ L5
def l5_targets(reg):
    from apa_sp2_common import targets, decode_target
    out = []
    for target in targets(reg)[0]:
        kind, bits, shape, index, _ = decode_target(target, reg)
        if kind == 'eq' and bits == 4:
            out.append((target, shape, index))
    return out


def l5_measure(tc, reg, target, shape, index):
    """SP2 G2 procedure (measure()) with the bulk scores from packed-r2 arithmetic.

    Imported pieces: data() and stats() from scripts/apa_sp2_gpu.py. The prefill
    classes' sampled queries are each scored as an L=1 decode call on the packed
    path (the per-pair bulk error depends only on the query and the key).
    """
    from apa_sp2_gpu import data, stats
    spo2 = tc._spo2
    chunks = []
    pins = 0
    pin_max = 0.
    scale = np.float32(1 / math.sqrt(shape['D']))
    S, D = shape['S'], shape['D']
    for draw in range(reg['eq_rule']['draws_per_shape']):
        q, k, v, indices = data(shape, index, reg['eq_rule']['measurement_seed'], draw, True)
        kt, vt = tc.tensor(k), tc.tensor(v)
        p = tc.quantize_sp2_keys(kt, 4, storage='packed')
        assert isinstance(p, tc.PackedSpKeys)
        codes = p.codes.numpy()[:, :, :S]
        norms = p.norms.numpy()[:, :, :S].astype(np.float64)
        cb = p.codebook.numpy().astype(np.float64)
        rot = p.rotation.numpy().astype(np.float64)
        ix = np.empty((*codes.shape[:3], D), np.int64)
        ix[..., 0::2] = codes & 15
        ix[..., 1::2] = codes >> 4
        bulk = np.empty((shape['B'], shape['H'], len(indices), S), np.float32)
        for row in range(len(indices)):
            qt = tc.tensor(np.ascontiguousarray(q[:, :, row:row + 1, :]))
            out = spo2.packed_run(qt, kt, vt, p._packed_tuple(), float(scale), 0.0, False, None, False,
                                  fmt='tq4', carry='partition', dump=True, gqa_path='r2')
            bulk[:, :, row, :] = out[2].numpy()[:, :, 0, :]
            del qt, out
        key_indices = np.unique(np.r_[np.arange(17), np.arange(S - 17, S)])
        for b in range(shape['B']):
            for h in range(shape['H']):
                kh = h // (shape['H'] // shape['KVH'])
                # Instrument pin (not the statistic): the dumped bulk is the packed
                # score of (rotated q, codebook[codes], norm), recomputed in fp64.
                qr = q[b, h, :2].astype(np.float64) @ rot[kh].T
                want = (qr @ cb[ix[b, kh, key_indices]].T) * norms[b, kh, key_indices] * float(scale)
                got = bulk[b, h, :2][:, key_indices].astype(np.float64)
                err = np.abs(got - want)
                pin_max = max(pin_max, float(err.max()))
                assert np.all(err <= 1e-4 + 1e-5 * np.abs(want)), ('packed bulk pin', target, float(err.max()))
                pins += want.size
                truth = q[b, h].astype(np.float64) @ k[b, kh].astype(np.float64).T * float(scale)
                lengths = S - shape['L'] + indices + 1 if shape['causal'] else np.full(len(indices), S)
                valid = np.arange(S)[None, :] < lengths[:, None]
                chunks.append(np.abs(bulk[b, h].astype(np.float64) - truth)[valid])
        del kt, vt, p
    errors = np.concatenate(chunks)
    return errors, stats(errors), dict(checked_scores=pins, max_abs_deviation=pin_max)


def l5(args):
    tc, arm = load_arm('B')
    os.environ['TC_APA_SP'] = '1'  # L5 repair 1: the native packed entry requires it
    reg = sp2_registration()
    mreg = json.loads((ART / 'MARGIN_REGISTRATION.json').read_text())
    assert sha(ART / 'MARGIN_REGISTRATION.json') == tc.apa_packed.PACKED_REGISTRATION_SHA256
    out = ART / 'margins'
    out.mkdir(exist_ok=True)
    start = time.monotonic()
    done = 0
    with tc.no_grad():
        for target, shape, index in l5_targets(reg):
            if (out / f'{target}.json').exists():
                continue
            if time.monotonic() - start > args.max_seconds:
                print('L5 time budget reached; resume with the same line', flush=True)
                break
            t0 = time.monotonic()
            errors, st, pin = l5_measure(tc, reg, target, shape, index)
            path = out / f'{target}.npy'
            with path.open('xb') as f:
                np.save(f, errors, allow_pickle=False)
            save(out / f'{target}.json', dict(status='PASS', target=target, bits=4, shape=shape,
                 statistics=st, error_values=str(path.relative_to(ROOT)), error_values_sha256=sha(path),
                 consistency_pin=pin, seconds=time.monotonic() - t0, module_sha256=arm['module_sha256'],
                 margin_registration_sha256=sha(ART / 'MARGIN_REGISTRATION.json'),
                 evidence_class='GPU packed-r2 bulk vs FP64 exact logits; kernel sweep, not model quality'))
            done += 1
            print('L5', target, st, flush=True)
    remaining = [t for t, _, _ in l5_targets(reg) if not (out / f'{t}.json').exists()]
    print('L5 measured', done, 'remaining', len(remaining), flush=True)
    save(Path(args.out) / 'RESULT.json', dict(line='L5', measured_this_run=done, remaining=remaining,
         module_sha256=arm['module_sha256'], terminal='L5_DONE' if not remaining else 'L5_PARTIAL'))
    return 0


def l5_freeze(args):
    """SP2 freeze(): pool per (bits, D), stats, e_q = nextafter(float32(max), +inf)."""
    from apa_sp2_gpu import stats
    reg = sp2_registration()
    mreg = json.loads((ART / 'MARGIN_REGISTRATION.json').read_text())
    rows = []
    for target, shape, index in l5_targets(reg):
        row = json.loads((ART / 'margins' / f'{target}.json').read_text())
        assert row['status'] == 'PASS'
        row['_receipt'] = f'artifacts/apa_p1d/margins/{target}.json'
        rows.append(row)
    assert len(rows) == 32
    entries = {}
    for D in (64, 128):
        group = [r for r in rows if r['shape']['D'] == D]
        arrays = []
        for r in group:
            assert sha(ROOT / r['error_values']) == r['error_values_sha256']
            arrays.append(np.load(ROOT / r['error_values'], allow_pickle=False))
        values = np.concatenate(arrays)
        info = stats(values)
        info['e_q'] = float(np.nextafter(np.float32(info['max']), np.float32(np.inf)))
        info['receipts'] = {r['_receipt']: sha(ROOT / r['_receipt']) for r in group}
        entries[f'4:{D}'] = info
    table = dict(status='FROZEN', registration_sha256=sha(ART / 'MARGIN_REGISTRATION.json'),
                 quantizer=mreg['quantizer_identity'], statistic=reg['eq_rule']['statistic'],
                 measurement_classes=len(rows), entries=entries,
                 procedure='SP2 G2 (scripts/apa_sp2_gpu.py data/stats/freeze rule), 4-bit targets, packed-r2 bulk',
                 scope_note='kernel sweep; this establishes nothing about model quality',
                 validity=reg['eq_rule']['validity'])
    if PACKED_TABLE.exists():
        if json.loads(PACKED_TABLE.read_text()) != table:
            raise RuntimeError('immutable packed table exists with different evidence')
    else:
        save(PACKED_TABLE, table)
    pin = ART / 'packed_e_q_table.sha256'
    if not pin.exists():
        with pin.open('x') as f:
            f.write(sha(PACKED_TABLE) + '  packed_e_q_table.json\n')
    assert pin.read_text().split()[0] == sha(PACKED_TABLE)
    print('FROZEN', sha(PACKED_TABLE))
    return 0


# ------------------------------------------------------------------ L1
GROUPS = (1, 2, 3, 4, 5, 6, 7, 8)


def l1_cases():
    out = []
    for D in (64, 128):
        for B in (1, 2):
            for G in GROUPS:
                for S in (255, 256, 257, 2048):
                    for causal in (False, True):
                        for sinks in (False, True):
                            out.append(dict(D=D, B=B, KV=2, H=2 * G, S=S, causal=causal, sinks=sinks))
    return out


def case_id(c):
    return f"D{c['D']}_B{c['B']}_H{c['H']}_KV{c['KV']}_S{c['S']}_c{int(c['causal'])}_s{int(c['sinks'])}"


def case_inputs(c, i):
    rng = np.random.default_rng(np.random.SeedSequence([20261115, 1, i]))
    q = (3 * rng.standard_normal((c['B'], c['H'], 1, c['D']))).astype(np.float32)
    k = rng.standard_normal((c['B'], c['KV'], c['S'], c['D']), dtype=np.float32)
    v = rng.standard_normal((c['B'], c['KV'], c['S'], c['D']), dtype=np.float32)
    s = rng.standard_normal((c['H'],), dtype=np.float32) if c['sinks'] else None
    return q, k, v, s


L1_EPSILON = 0.01


def l1_arm(args):
    tc, arm = load_arm(args.arm)
    os.environ.pop('TC_APA_SP_STORAGE', None)
    os.environ['TC_APA_SP'] = '1'
    dm = dense_margins(tc)
    pm = packed_margins(tc) if args.arm == 'B' else None
    arrays = {}
    routes = {}
    with tc.no_grad():
        for i, c in enumerate(l1_cases()):
            q, k, v, s = case_inputs(c, i)
            qt, kt, vt = tc.tensor(q), tc.tensor(k), tc.tensor(v)
            st = tc.tensor(s) if s is not None else None
            scale = float(np.float32(1 / math.sqrt(c['D'])))
            cid = case_id(c)

            def run(kq, margins, tag):
                out, mask = tc.apa_selective_attention_sp(qt, kt, kq, vt, scale, L1_EPSILON, bulk_bits=4,
                                                          margins=margins, is_causal=c['causal'], sinks=st,
                                                          diagnostics=True)
                plain = tc.apa_selective_attention_sp(qt, kt, kq, vt, scale, L1_EPSILON, bulk_bits=4,
                                                      margins=margins, is_causal=c['causal'], sinks=st)
                arrays[f'{cid}/{tag}/out'] = out.numpy()
                arrays[f'{cid}/{tag}/mask'] = mask.numpy()
                arrays[f'{cid}/{tag}/plain'] = plain.numpy()
            if args.arm == 'A':
                kq = tc.quantize_sp2_keys(kt, 4)
                assert type(kq).__name__ == 'Tensor'
                run(kq, dm, 'dense_public')
            else:
                kq = tc.quantize_sp2_keys(kt, 4)
                r_q = tc.apa_packed.last_route('quantize')
                assert isinstance(kq, tc.PackedSpKeys) and r_q['source'] == 'default'
                run(kq, pm, 'default')
                r_a = tc.apa_packed.last_route('attention')
                kq2 = tc.quantize_sp2_keys(kt, 4, storage='packed')
                assert isinstance(kq2, tc.PackedSpKeys)
                run(kq2, pm, 'explicit_packed')
                kd = tc.quantize_sp2_keys(kt, 4, storage='dense')
                assert type(kd).__name__ == 'Tensor'
                run(kd, dm, 'dense_argument')
                r_d = tc.apa_packed.last_route('attention')
                os.environ['TC_APA_SP_STORAGE'] = 'dense'
                ke = tc.quantize_sp2_keys(kt, 4)
                r_e = tc.apa_packed.last_route('quantize')
                del os.environ['TC_APA_SP_STORAGE']
                assert type(ke).__name__ == 'Tensor' and r_e['source'] == 'environment'
                run(ke, dm, 'dense_environment')
                routes[cid] = dict(default_quantize=r_q, default_attention=r_a, dense_attention=r_d,
                                   env_quantize=r_e)
            del qt, kt, vt, st
    path = Path(args.out) / f'arm_{args.arm}.npz'
    with path.open('xb') as f:
        np.savez(f, **arrays)
    save(Path(args.out) / f'arm_{args.arm}.json', dict(arm=arm, cases=len(l1_cases()), arrays=len(arrays),
         npz_sha256=sha(path), routes=routes))
    print('L1 arm', args.arm, 'arrays', len(arrays), flush=True)
    return 0


def child(argv, log, limit):
    t0 = time.monotonic()
    with Path(log).open('x') as f:
        try:
            code = subprocess.run(argv, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=limit).returncode
        except subprocess.TimeoutExpired:
            code = 124
    return dict(argv=[str(a) for a in argv], exit=code, seconds=time.monotonic() - t0, log=str(log))


def l1(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = []
    runs = []
    for arm in ('A', 'B'):
        r = child([sys.executable, '-B', __file__, 'l1-arm', '--arm', arm, '--out', str(out)],
                  out / f'arm_{arm}.log', 700)
        runs.append(r)
        step(log, f'l1_arm_{arm}_exit', exit=r['exit'])
        if r['exit']:
            save(out / 'RESULT.json', dict(line='L1', status='FAIL_CHILD', runs=runs, steps=log, terminal='L1_CHILD_FAILED'))
            return 1
    A = np.load(out / 'arm_A.npz')
    Bz = np.load(out / 'arm_B.npz')
    rows = []
    for c in l1_cases():
        cid = case_id(c)
        row = dict(case=cid)
        for name in ('out', 'mask', 'plain'):
            row[f'default_vs_explicit_{name}'] = bitwise(Bz[f'{cid}/default/{name}'], Bz[f'{cid}/explicit_packed/{name}'])
            row[f'dense_argument_vs_A_{name}'] = bitwise(Bz[f'{cid}/dense_argument/{name}'], A[f'{cid}/dense_public/{name}'])
            row[f'dense_environment_vs_A_{name}'] = bitwise(Bz[f'{cid}/dense_environment/{name}'], A[f'{cid}/dense_public/{name}'])
        # Not a gate: the packed route differs from dense by design (own arithmetic).
        row['packed_vs_dense_mask_differs'] = not np.array_equal(Bz[f'{cid}/default/mask'], A[f'{cid}/dense_public/mask'])
        rows.append(row)
    step(log, 'l1_compared', cases=len(rows))
    comps = [v for r in rows for k, v in r.items() if isinstance(v, dict)]
    equal = sum(v['equal'] for v in comps)
    informative_ok = sum(v.get('informative') == [True, True] for v in comps)
    out_informative = sum(v.get('informative') == [True, True] for r in rows for k, v in r.items()
                          if isinstance(v, dict) and not k.endswith('mask'))
    status = 'PASS' if equal == len(comps) else 'FAIL'
    save(out / 'RESULT.json', dict(line='L1', status=status, cases=len(rows), comparisons=len(comps),
         bitwise_equal=equal, both_sides_informative=informative_ok,
         output_comparisons_informative=out_informative, runs=runs, rows=rows, steps=log,
         terminal='L1_DONE'))
    print('L1', status, equal, '/', len(comps), 'informative', informative_ok, flush=True)
    return 0 if status == 'PASS' else 1


# ------------------------------------------------------------------ L2
class Spy:
    def __init__(self, target, names):
        self._target, self._names, self.calls = target, set(names), []

    def __getattr__(self, name):
        attr = getattr(self._target, name)
        if name in self._names:
            def wrapped(*a, **k):
                self.calls.append(name)
                return attr(*a, **k)
            return wrapped
        return attr


def l2(args):
    tc, arm = load_arm('B')
    from test_apa_p1d_policy import delta_fixtures
    os.environ.pop('TC_APA_SP_STORAGE', None)
    os.environ['TC_APA_SP'] = '1'
    dm, pm = dense_margins(tc), packed_margins(tc)
    checks = []
    log = []
    real_spo2 = tc._spo2
    spo2 = Spy(real_spo2, ['packed_run', 'packed_prepare', 'packed_append'])
    tc._spo2 = spo2
    real_quant = tc.quant._quantize_keys
    qcalls = []
    tc.quant._quantize_keys = lambda *a: qcalls.append(1) or real_quant(*a)
    real_eps = tc._C.apa_selective_attention_sp_epsilon
    ecalls = []
    tc._C.apa_selective_attention_sp_epsilon = lambda *a: ecalls.append(1) or real_eps(*a)

    def device_calls():
        return len(spo2.calls) + len(qcalls) + len(ecalls)

    def expect(name, fn, exc, match=None, before_launch=True):
        n0 = device_calls()
        try:
            fn()
            got, msg = None, ''
        except Exception as e:  # noqa: BLE001
            got, msg = type(e).__name__, str(e)
        ok = got == exc.__name__ and (match is None or match in msg)
        launched = device_calls() - n0
        if before_launch:
            ok = ok and launched == 0
        checks.append(dict(check=name, expected=exc.__name__, match=match, raised=got, message=msg,
                           device_calls_during=launched, ok=ok))
        step(log, name, ok=ok)

    def record(name, ok, **info):
        checks.append(dict(check=name, ok=bool(ok), **info))
        step(log, name, ok=bool(ok))

    rng = np.random.default_rng(np.random.SeedSequence([20261115, 2]))
    D, S, KV, H = 128, 257, 2, 8
    k = rng.standard_normal((1, KV, S, D), dtype=np.float32)
    q = (3 * rng.standard_normal((1, H, 1, D))).astype(np.float32)
    v = rng.standard_normal((1, KV, S, D), dtype=np.float32)
    scale = float(np.float32(1 / math.sqrt(D)))
    with tc.no_grad():
        kt, qt, vt = tc.tensor(k), tc.tensor(q), tc.tensor(v)
        # R-A
        for bad in ('Packed', 'int4', '', 1):
            expect(f'RA_storage_argument_{bad!r}', lambda: tc.quantize_sp2_keys(kt, 4, storage=bad), ValueError, 'storage must be')
        for bad in ('Packed', '', 'none'):
            os.environ['TC_APA_SP_STORAGE'] = bad
            expect(f'RA_storage_environment_{bad!r}', lambda: tc.quantize_sp2_keys(kt, 4), ValueError, 'TC_APA_SP_STORAGE')
            del os.environ['TC_APA_SP_STORAGE']
        # R-B
        k2 = tc.quantize_sp2_keys(kt, 2)
        route = tc.apa_packed.last_route('quantize')
        k2d = tc.quantize_sp2_keys(kt, 2, storage='dense')
        record('RB_two_bit_default_takes_dense_with_reason', type(k2).__name__ == 'Tensor' and
               route['storage'] == 'dense' and 'bulk_bits=2' in (route['fallback_reason'] or '') and
               bitwise(k2.numpy(), k2d.numpy())['equal'], route=route)
        expect('RB_two_bit_explicit_packed_raises', lambda: tc.quantize_sp2_keys(kt, 2, storage='packed'), ValueError, 'cannot be honoured')
        # R-D
        p = tc.quantize_sp2_keys(kt, 4)
        kd = tc.quantize_sp2_keys(kt, 4, storage='dense')
        call = tc.apa_selective_attention_sp
        expect('RD_dense_table_on_packed', lambda: call(qt, kt, p, vt, scale, .01, bulk_bits=4, margins=dm), ValueError, 'dense margin table')
        expect('RD_packed_table_on_dense', lambda: call(qt, kt, kd, vt, scale, .01, bulk_bits=4, margins=pm), ValueError, 'packed margin table')
        expect('RD_bulk_bits_mismatch', lambda: call(qt, kt, p, vt, scale, .01, bulk_bits=2, margins=pm), ValueError, 'bulk_bits')
        kcopy = tc.tensor(k)
        expect('RD_k_not_exact_source', lambda: call(qt, kcopy, p, vt, scale, .01, bulk_bits=4, margins=pm), ValueError, 'exact-K source')
        expect('RD_no_margins', lambda: call(qt, kt, p, vt, scale, .01, bulk_bits=4, margins=None), ValueError, 'packed margin')
        expect('RD_wrong_scale', lambda: call(qt, kt, p, vt, .1, .01, bulk_bits=4, margins=pm), ValueError, 'scale')
        expect('RD_epsilon_zero', lambda: call(qt, kt, p, vt, scale, 0., bulk_bits=4, margins=pm), ValueError, 'epsilon')
        qcpu = tc.tensor(q, device='cpu')
        expect('RD_query_on_cpu', lambda: call(qcpu, kt, p, vt, scale, .01, bulk_bits=4, margins=pm), ValueError, 'device')
        del os.environ['TC_APA_SP']
        expect('RD_tc_apa_sp_unset', lambda: call(qt, kt, p, vt, scale, .01, bulk_bits=4, margins=pm), RuntimeError, 'TC_APA_SP=1')
        os.environ['TC_APA_SP'] = '1'
        kst = tc.tensor(k)
        ps = tc.quantize_sp2_keys(kst, 4)
        ps.invalidate()
        expect('RD_stale_generation', lambda: call(qt, kst, ps, vt, scale, .01, bulk_bits=4, margins=pm), ValueError, 'stale')
        ked = tc.tensor(k)
        pe = tc.quantize_sp2_keys(ked, 4)
        ked.fill_(0.5)
        expect('RD_exact_k_edited_in_place', lambda: call(qt, ked, pe, vt, scale, .01, bulk_bits=4, margins=pm), ValueError, 'edited in place')
        # Append: visible rows follow k; appended object equals a fresh one bitwise.
        kshort = tc.tensor(np.ascontiguousarray(k[:, :, :200]))
        pa = tc.PackedSpKeys(kshort, capacity=S)
        expect('RD_visible_rows_mismatch', lambda: call(qt, kt, pa, vt, scale, .01, bulk_bits=4, margins=pm), ValueError, 'exact-K source')
        pa.append(kt)
        o1, m1 = call(qt, kt, pa, vt, scale, .01, bulk_bits=4, margins=pm, diagnostics=True)
        o2, m2 = call(qt, kt, p, vt, scale, .01, bulk_bits=4, margins=pm, diagnostics=True)
        record('RC_append_equals_fresh_bitwise', bitwise(o1.numpy(), o2.numpy())['equal'] and
               bitwise(m1.numpy(), m2.numpy())['equal'] and informative(o1.numpy()) and
               bitwise(pa.codes.numpy(), p.codes.numpy())['equal'] and bitwise(pa.norms.numpy(), p.norms.numpy())['equal'],
               generation=pa.generation)
        route = tc.apa_packed.last_route('attention')
        record('RH_attention_route_packed', route['storage'] == 'packed' and route['backend'] == tc.apa_packed.PACKED_BACKEND, route=route)
        # Raw dense _C entries refuse a packed object (pybind type check).
        tc._C.apa_selective_attention_sp_epsilon = real_eps
        for name, fn in [('apa_selective_attention_sp_epsilon', lambda: tc._C.apa_selective_attention_sp_epsilon(qt, kt, p, vt, scale, .01, .5, False, None, False)),
                         ('apa_selective_attention_sp', lambda: tc._C.apa_selective_attention_sp(qt, kt, p, vt, scale, 8., False, None, False)),
                         ('apa_sp2_scores', lambda: tc._C.apa_sp2_scores(qt, kt, p, scale, 0, 1, True))]:
            expect(f'RI_raw_dense_entry_refuses_packed_{name}', fn, TypeError, None, before_launch=False)
        tc._C.apa_selective_attention_sp_epsilon = lambda *a: ecalls.append(1) or real_eps(*a)
        # Delta helper versus the native rule on the registered boundary fixtures.
        mism = []
        for eps, e in delta_fixtures():
            try:
                native = tc._C.apa_sp2_delta(eps, e)
            except RuntimeError:
                native = 'raise'
            try:
                helper = tc.apa_packed.derive_delta(eps, e)
            except ValueError:
                helper = 'raise'
            if native != helper:
                mism.append([eps, e, native, helper])
        for eps, e in [(0, 0), (-.1, 0), (1.1, 0), (np.nan, 0), (.1, -1), (.1, np.inf), (.1, 1e100), (1., 2e38)]:
            ok_native = ok_helper = False
            try:
                tc._C.apa_sp2_delta(eps, e)
            except RuntimeError:
                ok_native = True
            try:
                tc.apa_packed.derive_delta(eps, e)
            except ValueError:
                ok_helper = True
            if not (ok_native and ok_helper):
                mism.append([eps, e, 'domain', ok_native, ok_helper])
        record('G3_delta_helper_equals_native', not mism, fixtures=len(delta_fixtures()), mismatches=mism)
        # Delta forwarded on the packed route equals the native derivation for the table entry.
        route = tc.apa_packed.last_route('attention')
        record('G3_route_delta_equals_native', route['delta'] == tc._C.apa_sp2_delta(.01, route['e_q']),
               delta=route['delta'], e_q=route['e_q'])
        # R-E prefill per the branch L3 selected.
        qpre = tc.tensor((3 * rng.standard_normal((1, H, 4, D))).astype(np.float32))
        branch = json.loads((ART / 'gpu/L3/RESULT.json').read_text())['r_e_branch']
        if branch == 'refuse_prefill_on_packed':
            expect('RE_prefill_on_packed_refused', lambda: call(qpre, kt, p, vt, scale, .01, bulk_bits=4, margins=pm,
                   is_causal=True), ValueError, "storage='dense'")
            outd = call(qpre, kt, kd, vt, scale, .01, bulk_bits=4, margins=dm, is_causal=True)
            record('RE_prefill_on_dense_runs', informative(outd.numpy()))
        else:
            o = call(qpre, kt, p, vt, scale, .01, bulk_bits=4, margins=pm, is_causal=True)
            od = call(qpre, kt, kd, vt, scale, .01, bulk_bits=4, margins=dm, is_causal=True)
            record('RE_prefill_transient_equals_dense', bitwise(o.numpy(), od.numpy())['equal'])
    tc._spo2 = real_spo2
    ok = all(c['ok'] for c in checks)
    save(Path(args.out) / 'RESULT.json', dict(line='L2', status='PASS' if ok else 'FAIL', checks=checks,
         module_sha256=arm['module_sha256'], steps=log, terminal='L2_DONE'))
    print('L2', 'PASS' if ok else 'FAIL', sum(c['ok'] for c in checks), '/', len(checks), flush=True)
    return 0 if ok else 1


# ------------------------------------------------------------------ L4
def cuda_used():
    lib = ctypes.CDLL('libcudart.so.12')
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    assert lib.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)) == 0
    return total.value - free.value


def smi_own_mib():
    out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, timeout=30).stdout
    for line in out.splitlines():
        pid, mib = [x.strip() for x in line.split(',')]
        if int(pid) == os.getpid():
            return int(mib)
    return None


L4_MODELS = {'qwen3_1.7b': 16, 'qwen3_4b': 32}


def l4_case(args):
    tc, arm = load_arm('B')
    os.environ.pop('TC_APA_SP_STORAGE', None)
    os.environ['TC_APA_SP'] = '1'
    S, H, KV, D = args.S, L4_MODELS[args.model], 8, 128
    rng = np.random.default_rng(np.random.SeedSequence([20261115, 4, S]))
    with tc.no_grad():
        warm = tc.tensor(rng.standard_normal((1, KV, 8, D), dtype=np.float32))
        tc.quantize_sp2_keys(warm, 4, storage='dense')
        tc.quantize_sp2_keys(warm, 4)
        tc.synchronize()
        del warm
        gc.collect()
        tc.synchronize()
        # Allocator granularity probe: used-memory step of small allocations.
        g0 = cuda_used()
        probes = []
        for n in (1, 1 << 18, (1 << 19) + 1):
            t = tc.tensor(np.zeros(n, np.float32))
            tc.synchronize()
            probes.append(dict(bytes=4 * n, used_delta=cuda_used() - g0))
            del t
            gc.collect()
            tc.synchronize()
        granularity = max(2 << 20, max(p['used_delta'] for p in probes[:1]))
        k = tc.tensor(rng.standard_normal((1, KV, S, D), dtype=np.float32))
        tc.synchronize()
        m0, s0 = cuda_used(), smi_own_mib()
        kq = tc.quantize_sp2_keys(k, 4, storage='dense')
        tc.synchronize()
        m_dense, s_dense = cuda_used(), smi_own_mib()
        dense_bytes = int(np.prod(kq.shape)) * 4
        del kq
        gc.collect()
        tc.synchronize()
        m1, s1 = cuda_used(), smi_own_mib()
        p = tc.quantize_sp2_keys(k, 4)
        tc.synchronize()
        m_packed, s_packed = cuda_used(), smi_own_mib()
        assert isinstance(p, tc.PackedSpKeys)
        owned = {name: int(np.prod(t.shape)) * {'uint8': 1, 'float32': 4}[t.dtype]
                 for name, t in p.owned_buffers().items()}
        tables = {name: int(np.prod(t.shape)) * 4 for name, t in p.fixed_tables().items()}
        reachable = []
        for name in type(p).__slots__:
            if name != '__weakref__' and hasattr(p, name):
                reachable.append((name, getattr(p, name)))
        reachable += list(vars(p._ws).items())
        k_shaped = [name for name, x in reachable if type(x).__name__ == 'Tensor' and tuple(x.shape) == tuple(k.shape)]
        tensors = {name: list(x.shape) for name, x in reachable if type(x).__name__ == 'Tensor'}
        q = tc.tensor((3 * rng.standard_normal((1, H, 1, D))).astype(np.float32))
        v = tc.tensor(rng.standard_normal((1, KV, S, D), dtype=np.float32))
        out = tc.apa_selective_attention_sp(q, k, p, v, float(np.float32(1 / math.sqrt(D))), .01, bulk_bits=4,
                                            margins=packed_margins(tc))
        decode_ok = informative(out.numpy())
    keys = KV * S
    predicted = keys * (512 - 68)
    dense_delta, packed_delta = m_dense - m0, m_packed - m1
    measured = dense_delta - packed_delta
    G = granularity
    result = dict(model=args.model, H=H, KV=KV, D=D, S=S, owned=owned, owned_total=sum(owned.values()),
                  owned_per_key_per_kv_head=sum(owned.values()) / keys, fixed_tables=tables,
                  dense_kq_bytes=dense_bytes, tensors_reachable=tensors, k_shaped_owned=k_shaped,
                  cuda_used=dict(before=m0, dense=m_dense, after_dense_free=m1, packed=m_packed),
                  smi_own_mib=dict(before=s0, dense=s_dense, after_dense_free=s1, packed=s_packed),
                  dense_delta=dense_delta, packed_delta=packed_delta, measured_difference=measured,
                  predicted_difference=predicted, granularity=G, granularity_probes=probes,
                  checks=dict(bytes_68=sum(owned.values()) == 68 * keys,
                              no_k_shaped=not k_shaped,
                              memory_at_or_below=packed_delta <= dense_delta - predicted + 3 * G,
                              memory_within_granularity=abs(measured - predicted) <= 3 * G,
                              decode_ok=decode_ok))
    save(Path(args.out) / f'{args.model}_S{S}.json', result)
    print('L4', args.model, S, result['checks'], flush=True)
    return 0 if all(result['checks'].values()) else 1


def l4(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs, log = [], []
    for model in L4_MODELS:
        for S in (2048, 8192, 32768):
            r = child([sys.executable, '-B', __file__, 'l4-case', '--model', model, '--S', str(S), '--out', str(out)],
                      out / f'{model}_S{S}.log', 400)
            runs.append(r)
            step(log, f'l4_{model}_S{S}', exit=r['exit'])
    cases = [json.loads((out / f'{m}_S{S}.json').read_text()) for m in L4_MODELS for S in (2048, 8192, 32768)
             if (out / f'{m}_S{S}.json').exists()]
    ok = len(cases) == 6 and all(all(c['checks'].values()) for c in cases)
    save(out / 'RESULT.json', dict(line='L4', status='PASS' if ok else 'FAIL', runs=runs,
         summary=[dict(model=c['model'], S=c['S'], per_key=c['owned_per_key_per_kv_head'],
                       measured=c['measured_difference'], predicted=c['predicted_difference'],
                       granularity=c['granularity'], checks=c['checks']) for c in cases],
         steps=log, terminal='L4_DONE'))
    print('L4', 'PASS' if ok else 'FAIL', flush=True)
    return 0 if ok else 1


# ------------------------------------------------------------------ L6
def outcome_rows(a, b):
    rows = []
    for name in sorted(set(a['reports']) | set(b['reports'])):
        x, y = a['reports'].get(name, []), b['reports'].get(name, [])
        xv = [(r['when'], r['outcome']) for r in x]
        yv = [(r['when'], r['outcome']) for r in y]
        row = dict(id=name, A=xv, B=yv, equal=xv == yv)
        if xv != yv:
            row.update(A_messages=[r['message'] for r in x if r['message']],
                       B_messages=[r['message'] for r in y if r['message']])
        rows.append(row)
    return rows


def l6(args):
    # L6 repair 1: arm A runs with cwd in the e2fb85a1 tree, so receipt paths must be absolute.
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    trees = {'A': Path('/tmp/apap1d_treeA'), 'B': ROOT}
    files = args.files
    runs, log, results = [], [], []
    start = time.monotonic()
    # L6 repair 2 (load): resumable like L5. A file whose two receipts exist from
    # an earlier attempt is compared, not rerun; no new file starts after
    # --max-seconds, so the line ends before its outer timeout. Attempt n > 1
    # writes <tag>_<arm>.r<n>.{json,log}; earlier attempts stay on disk.
    for f in files:
        tag = Path(f).stem
        done = sorted(n for n in range(1, 20)
                      if all((out / (f'{tag}_{a}.json' if n == 1 else f'{tag}_{a}.r{n}.json')).exists() for a in 'AB'))
        if done:
            n = done[-1]
            recs = {a: json.loads((out / (f'{tag}_{a}.json' if n == 1 else f'{tag}_{a}.r{n}.json')).read_text()) for a in 'AB'}
            step(log, f'l6_{tag}_reused_attempt_{n}')
        elif time.monotonic() - start > args.max_seconds:
            results.append(dict(file=f, status='NOT_RUN_TIME_BUDGET'))
            continue
        else:
            recs = None
        n = 1 + max([0] + [int(p.name.split('.r')[1].split('.')[0]) if '.r' in p.name else 1
                           for p in out.glob(f'{tag}_A*.log')])
        for armname in (() if recs is not None else ('A', 'B')):
            if recs is None:
                recs = {}
            arm = json.loads((ART / 'arms' / (ARM_B if armname == 'B' else armname) / 'ARM.json').read_text())
            tree = trees[armname]
            env = os.environ.copy()
            env.pop('TC_APA_SP_STORAGE', None)
            env.update(PYTHONPATH=str((ROOT / arm['package']).parent) + ':' + str(tree / 'tensor_cuda/tests') + ':' +
                       str(ROOT / 'artifacts/apa_p1'), PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTHONDONTWRITEBYTECODE='1',
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       TMPDIR='/tmp/apap1d_l6')
            Path(env['TMPDIR']).mkdir(exist_ok=True)
            suffix = '' if n == 1 else f'.r{n}'
            receipt = out / f'{tag}_{armname}{suffix}.json'
            cmd = [sys.executable, str(ROOT / 'artifacts/apa_p1/pytest_receipt.py'), str(receipt), '-q', '--tb=short',
                   '-ra', '-p', 'no:cacheprovider', '--basetemp', f'/tmp/apap1d_l6/bt_{tag}_{armname}{suffix}', f]
            t0 = time.monotonic()
            with (out / f'{tag}_{armname}{suffix}.log').open('x') as fh:
                try:
                    code = subprocess.run(cmd, cwd=tree, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=600).returncode
                except subprocess.TimeoutExpired:
                    code = 124
            runs.append(dict(file=f, arm=armname, exit=code, seconds=time.monotonic() - t0, cwd=str(tree)))
            step(log, f'l6_{tag}_{armname}', exit=code)
            recs[armname] = json.loads(receipt.read_text()) if receipt.exists() else None
        a, b = recs['A'], recs['B']
        if a is None or b is None:
            results.append(dict(file=f, status='FAIL_NO_RECEIPT'))
            continue
        rows = outcome_rows(a, b)
        assert a['module']['sha256'] == json.loads((ART / 'arms/A/ARM.json').read_text())['module_sha256']
        assert b['module']['sha256'] == json.loads((ART / 'arms' / ARM_B / 'ARM.json').read_text())['module_sha256']
        results.append(dict(file=f, ids=len(rows), collected_equal=a['collection'] == b['collection'],
                            collection_errors=[len(a['collection_errors']), len(b['collection_errors'])],
                            differences=sum(not r['equal'] for r in rows),
                            outcomes_A=_count(a), outcomes_B=_count(b), rows=rows))
    complete = all('ids' in r for r in results)
    ok = complete and all(r.get('differences') == 0 and r.get('collected_equal') for r in results)
    status = 'PASS' if ok else ('FAIL' if complete or any(r.get('differences') for r in results) else 'PARTIAL')
    save(out / f'RESULT.{time.time_ns()}.json', dict(line='L6', status=status, complete=complete, files=files,
         results=results, runs=runs, steps=log, terminal='L6_DONE' if complete else 'L6_PARTIAL'))
    print('L6', status, [(r['file'], r.get('ids'), r.get('differences'), r.get('status')) for r in results], flush=True)
    return 0 if ok or status == 'PARTIAL' else 1


def _count(rec):
    import collections
    c = collections.Counter()
    for name, reps in rec['reports'].items():
        outs = [(r['when'], r['outcome']) for r in reps]
        if any(o == 'failed' and w != 'call' for w, o in outs):
            c['ERROR'] += 1
        elif any(o == 'failed' for w, o in outs):
            c['FAILED'] += 1
        elif any(o == 'skipped' for w, o in outs):
            c['SKIPPED'] += 1
        else:
            c['PASSED'] += 1
    return dict(c)


def list_mode(args):
    reg = sp2_registration()
    for target, shape, index in l5_targets(reg):
        print(target)
    print('count', len(l5_targets(reg)))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['l1', 'l1-arm', 'l2', 'l3', 'l4', 'l4-case', 'l5', 'l5-freeze', 'l6', 'list'])
    ap.add_argument('--out')
    ap.add_argument('--arm')
    ap.add_argument('--model')
    ap.add_argument('--S', type=int)
    ap.add_argument('--files', nargs='*')
    ap.add_argument('--max-seconds', type=float, default=1500)
    args = ap.parse_args()
    if args.mode not in ('l5-freeze', 'list'):
        assert os.environ.get('CUDA_VISIBLE_DEVICES') != '', 'GPU modes run inside forge-gpu-run'
    if args.mode in ('l1', 'l2', 'l3', 'l4', 'l5', 'l6'):
        Path(args.out).mkdir(parents=True, exist_ok=True)
    fn = {'l1': l1, 'l1-arm': l1_arm, 'l2': l2, 'l3': l3, 'l4': l4, 'l4-case': l4_case, 'l5': l5,
          'l5-freeze': l5_freeze, 'l6': l6, 'list': list_mode}[args.mode]
    try:
        code = fn(args)
    except BaseException:
        if args.out:
            Path(args.out).mkdir(parents=True, exist_ok=True)
            (Path(args.out) / f'exception_{args.mode}_{time.time_ns()}.txt').write_text(traceback.format_exc())
        raise
    raise SystemExit(code)


if __name__ == '__main__':
    main()
