"""P1-F2 and P1-F3 LF2 packed-decode cases (Project-Tensor, 2026).

Prior art: IEEE 754-2008 classification and P1's row-set oracles.
Retain r2, public packed and dense comparators; omit unrelated engine routes.
"""
from pathlib import Path
import math
import os

import numpy as np

from apa_pf2f2_build import ROOT, sha
import apa_pf2f2_p1_inputs as R

PACKED_TABLE = ROOT / 'artifacts/apa_p1d/packed_e_q_table.json'
DENSE_TABLE = Path('/mnt/ForgeRealm/Project-Tensor/artifacts/apa_sp2/e_q_table.json')
DENSE_TABLE_SHA = '5b6f8bfb405b40c03da21ad8d50eb40c11b85ed9559ee717d2a105c90c0e618d'
FLT_MAX = float(np.finfo(np.float32).max)
OVF = float(np.float32(1e20))


def run_raw(tc, c, name, *, delta, carry, sinks, diag, build):
    assert name in ('r2_tq4', 'r2_int4')
    fmt = name.removeprefix('r2_')
    result = tc._spo2.packed_run(c.q, c.k, c.v, c.ws[fmt].packed, c.scale, delta,
                                False, sinks, diag, fmt=fmt, carry=carry, dump=diag, gqa_path='r2')
    return result if diag else (result,)


QKV_POS = {'q': [(0, 0, 0, 5), (0, 5, 0, 77), (0, 7, 0, 127)], 'k': [(0, 0, 100, 5), (0, 1, 700, 64), (0, 0, 1023, 127)],
           'v': [(0, 0, 100, 5), (0, 1, 700, 64), (0, 1, 0, 0)]}
VALUES = {'nan': np.nan, 'posinf': np.inf, 'neginf': -np.inf}


class Case:
    """REV1-style inputs (REV1 make_inputs, imported) and packed state for one build."""

    def __init__(self, tc, R, seed, B, H, KVH, S, D, dtype, plant, dup=False, mutate=None, sinks=False):
        self.tc, self.H, self.KVH, self.G, self.S, self.D, self.dtype = tc, H, KVH, H // KVH, S, D, dtype
        q, k, v = R.make_inputs(seed, B, H, KVH, S, D, plant=plant, dup=dup)
        if mutate:
            mutate(q, k, v)
        self.qn, self.kn, self.vn = q, k, v
        T = lambda x: tc.tensor(np.ascontiguousarray(x), dtype=dtype)
        self.T = T
        self.q, self.k, self.v = T(q), T(k), T(v)
        self.scale = float(np.float32(1 / np.sqrt(D)))
        self.ws = {f: tc.ApaSpq1Workspace(self.k, S, fmt=f) for f in ('tq4', 'int4')}
        self.sinks = T(R.rnd_bf16(np.linspace(1.0, 3.0, H).astype(np.float32))) if sinks else None


def npf(t):
    return np.asarray(t.numpy())


def row_sets(o, H):
    o = o.astype(np.float32).reshape(H, -1)
    nonfinite = sorted(h for h in range(H) if not np.all(np.isfinite(o[h])))
    zero = sorted(h for h in range(H) if np.all(o[h] == 0))
    return nonfinite, zero


def lf2_cases():
    cases = []
    for t, positions in QKV_POS.items():
        for i, pos in enumerate(positions):
            for vname, val in VALUES.items():
                declared = 'all_finite' if (t, pos, vname) == ('k', (0, 0, 100, 5), 'posinf') else None
                cases.append(dict(id=f'{t}_{vname}_p{i}', group='qkv', tensor=t, pos=pos, value=val, nan=vname == 'nan',
                                  declared=declared))
    for s in ('nan', 'posinf'):
        cases.append(dict(id=f'sink_{s}', group='sinks', value=VALUES[s], nan=False, declared='all_finite'))
    cases.append(dict(id='flt_max_g4', group='flt_max_g4', nan=False, declared=None))
    for s in ('posinf', 'neginf'):
        cases.append(dict(id=f'all_negative_logits_q_{s}', group='allneg', value=VALUES[s], nan=False, declared='all_finite'))
    return cases


def make_case(tc, R, c, dtype):
    if c['group'] == 'qkv':
        def mutate(q, k, v):
            dict(q=q, k=k, v=v)[c['tensor']][c['pos']] = c['value']
    elif c['group'] == 'flt_max_g4':
        def mutate(q, k, v):
            q[:] = 0.0
            q[:, 0:4] = 1.0
            k[:] = 1.0
    elif c['group'] == 'allneg':
        def mutate(q, k, v):
            q[:] = 0.0
            q[:, 0:4, 0, 0] = c['value']
            k[:] = -1.0 if c['value'] > 0 else 1.0
    else:
        mutate = None
    cc = Case(tc, R, 11, 1, 8, 2, 1024, 128, dtype, plant=False, mutate=mutate, sinks=c['group'] == 'sinks')
    if c['group'] == 'sinks':
        s = npf(cc.sinks).astype(np.float32)
        s[2] = c['value']
        cc.sinks = tc.tensor(np.ascontiguousarray(s), dtype=dtype)
    if c['group'] == 'flt_max_g4':
        cc.scale = FLT_MAX
    return cc


def classify(o, H):
    nonfinite, zero = row_sets(np.asarray(o), H)
    return dict(nonfinite=nonfinite, zero=zero)


def gate_checks(checks, case_id, entry, dtype, got, dense, nan_case, comparator):
    tag = f'{entry} {dtype} {case_id} vs {comparator}'
    if 'ERROR' in got:
        checks.append(dict(id=f'G2 entry ran: {tag}', ok=False, error=got['ERROR']))
        return
    en, dn = set(got['nonfinite']), set(dense['nonfinite'])
    if nan_case:
        checks.append(dict(id=f'G2a NaN row-set equality: {tag}', ok=en == dn, entry=sorted(en), dense=sorted(dn)))
    checks.append(dict(id=f'G2b containment (never hides): {tag}', ok=dn <= en, entry=sorted(en), dense=sorted(dn),
                       hidden=sorted(dn - en)))
    zero_bad = sorted(set(got['zero']) & dn)
    checks.append(dict(id=f'G2c no all-zero row where dense is non-finite: {tag}', ok=zero_bad == [], rows=zero_bad))
    checks.append(dict(id=f'G2d listed (allowed): {tag}', ok=True, report=True, extra_nonfinite=sorted(en - dn)))


def l19(checks, case_id, dtype, dense, H, declared):
    n = len(dense['nonfinite'])
    if declared == 'all_finite':
        checks.append(dict(id=f'L19 dense all-finite as declared: {dtype} {case_id}', ok=n == 0, dense=dense['nonfinite']))
    elif declared == 'single_row':
        checks.append(dict(id=f'L19 single-row case (declared): {dtype} {case_id}', ok=True, report=True,
                           dense=dense['nonfinite']))
    else:
        checks.append(dict(id=f'L19 dense informative: {dtype} {case_id}', ok=0 < n < H, dense=dense['nonfinite']))


def public_pair(tc, pm, dm, c, sinks):
    kq = tc.quantize_sp2_keys(c.k, 4)
    assert isinstance(kq, tc.PackedSpKeys), 'default route did not choose PackedSpKeys'
    out = npf(tc.apa_selective_attention_sp(c.q, c.k, kq, c.v, c.scale, 0.01, bulk_bits=4, margins=pm, sinks=sinks))
    kd = tc.quantize_sp2_keys(c.k, 4, storage='dense')
    opt = npf(tc.apa_selective_attention_sp(c.q, c.k, kd, c.v, c.scale, 0.01, bulk_bits=4, margins=dm, sinks=sinks))
    return out, opt


def lf2_run(tc, R, build, only=None):
    """only: optional set of (case_id) to run (MUT m02)."""
    os.environ['TC_APA_SP'] = '1'
    os.environ.pop('TC_APA_SP_STORAGE', None)
    pm = tc.PackedRegisteredMargins.load(PACKED_TABLE)
    assert sha(DENSE_TABLE) == DENSE_TABLE_SHA
    dm = tc.RegisteredMargins.load(DENSE_TABLE, expected_sha256=DENSE_TABLE_SHA)
    H = 8
    res, checks = {}, []
    with tc.no_grad():
        for dtype in ('bfloat16', 'float32'):
            for c in lf2_cases():
                if only is not None and c['id'] not in only:
                    continue
                cc = make_case(tc, R, c, dtype)
                sinks = cc.sinks if c['group'] == 'sinks' else None
                dense = classify(npf(tc._C.apa_selective_attention_sp(cc.q, cc.k, cc.k, cc.v, cc.scale, 2.0, False, sinks,
                                                                      False)), H)
                rec = res.setdefault(dtype, {}).setdefault(c['id'], {})
                rec['dense'] = dense
                l19(checks, c['id'], dtype, dense, H, c['declared'])
                names = ('r2_tq4', 'r2_int4')
                for name in names:
                    try:
                        o = run_raw(tc, cc, name, delta=2.0, carry='partition', sinks=sinks, diag=False, build=build)[0]
                        rec[name] = classify(npf(o), H)
                    except Exception as e:   # a failed check, never a pass
                        rec[name] = dict(ERROR=repr(e)[:300])
                    gate_checks(checks, c['id'], name, dtype, rec[name], dense, c['nan'], 'dense_entry')
                if dtype == 'float32':
                    if c['group'] == 'flt_max_g4':
                        try:
                            public_pair(tc, pm, dm, cc, sinks)
                            rec['public_default'] = dict(ERROR='FLT_MAX scale was not refused by the packed table')
                        except ValueError as e:
                            rec['public_default'] = dict(refused_before_device_work=str(e)[:200])
                        checks.append(dict(id=f'FLT_MAX scale refused on the public route before device work: {c["id"]}',
                                           ok='refused_before_device_work' in rec['public_default']))
                        continue
                    try:
                        out, opt = public_pair(tc, pm, dm, cc, sinks)
                        rec['public_default'], rec['dense_opt_out'] = classify(out, H), classify(opt, H)
                    except Exception as e:
                        rec['public_default'] = dict(ERROR=repr(e)[:300])
                        rec['dense_opt_out'] = dict(nonfinite=[], zero=[])
                    gate_checks(checks, c['id'], 'public_default', dtype, rec['public_default'], dense, c['nan'],
                                'dense_entry')
                    gate_checks(checks, c['id'], 'public_default', dtype, rec['public_default'], rec['dense_opt_out'],
                                c['nan'], 'dense_opt_out')
        # REV2 r02 exactly (fp32 B1 H1 KV1 S3 D128, q=k=ones, v=arange/512, scale=FLT_MAX)
        if only is None or 'flt_max_r02' in only:
            def mutate(q, k, v):
                q[:] = 1.0
                k[:] = 1.0
                v[:] = (np.arange(3 * 128, dtype=np.float32).reshape(1, 1, 3, 128) / 512)
            cc = Case(tc, R, 11, 1, 1, 1, 3, 128, 'float32', plant=False, mutate=mutate)
            cc.scale = FLT_MAX
            dense = classify(npf(tc._C.apa_selective_attention_sp(cc.q, cc.k, cc.k, cc.v, cc.scale, 2.0, False, None,
                                                                  False)), 1)
            rec = res.setdefault('float32', {}).setdefault('flt_max_r02', {})
            rec['dense'] = dense
            l19(checks, 'flt_max_r02', 'float32', dense, 1, 'single_row')
            for name in ('r2_tq4', 'r2_int4'):
                try:
                    out = run_raw(tc, cc, name, delta=2.0, carry='partition', sinks=None, diag=False, build=build)[0]
                    rec[name] = classify(npf(out), 1)
                except Exception as e:
                    rec[name] = dict(ERROR=repr(e)[:300])
                gate_checks(checks, 'flt_max_r02', name, 'float32', rec[name], dense, False, 'dense_entry')
        # REV2 r01 default-route cases, unchanged (D64, B1 H1 KV1 S3)
        r01 = [('k_posinf', 'k', np.inf), ('k_neginf', 'k', -np.inf), ('k_nan', 'k', np.nan), ('q_posinf', 'q', np.inf),
               ('q_neginf', 'q', -np.inf), ('q_nan', 'q', np.nan), ('v_nan', 'v', np.nan), ('v_posinf', 'v', np.inf),
               ('v_neginf', 'v', -np.inf), ('zero_keys', None, 0.), ('q_posinf_all_negative_logits', 'q', np.inf),
               ('q_neginf_all_negative_logits', 'q', -np.inf)]
        for name, operand, bad in r01:
            cid = 'r01_' + name
            if only is not None and cid not in only:
                continue
            q = np.ones((1, 1, 1, 64), np.float32)
            k = np.zeros((1, 1, 3, 64), np.float32)
            v = np.arange(3 * 64, dtype=np.float32).reshape(1, 1, 3, 64) / 256
            if 'all_negative_logits' in name:
                q.fill(0)
                k.fill(-1. if bad > 0 else 1.)
            if operand == 'q':
                q[0, 0, 0, 0] = bad
            elif operand == 'k':
                k[0, 0, 1, 0] = bad
            elif operand == 'v':
                v[0, 0, 1, 0] = bad
            Q, K, V = [tc.tensor(np.ascontiguousarray(x)) for x in (q, k, v)]
            pk = tc.quantize_sp2_keys(K, 4)
            assert isinstance(pk, tc.PackedSpKeys)
            dk = tc.quantize_sp2_keys(K, 4, storage='dense')
            rec = res.setdefault('float32', {}).setdefault(cid, {})
            try:
                rec['public_default'] = classify(npf(tc.apa_selective_attention_sp(Q, K, pk, V, .125, .5, bulk_bits=4,
                                                                                   margins=pm)), 1)
            except Exception as e:
                rec['public_default'] = dict(ERROR=repr(e)[:300])
            rec['dense_opt_out'] = classify(npf(tc.apa_selective_attention_sp(Q, K, dk, V, .125, .5, bulk_bits=4,
                                                                              margins=dm)), 1)
            rec['dense'] = classify(npf(tc._C.apa_selective_attention_sp(Q, K, K, V, .125, 2.0, False, None, False)), 1)
            l19(checks, cid, 'float32', rec['dense'], 1, 'single_row')
            for comp in ('dense', 'dense_opt_out'):
                gate_checks(checks, cid, 'public_default', 'float32', rec['public_default'], rec[comp],
                            name.endswith('_nan'), 'dense_entry' if comp == 'dense' else 'dense_opt_out')
    return res, checks


def norm_repr(x):
    x = float(x)
    return 'nan' if math.isnan(x) else ('inf' if math.isinf(x) else repr(x))


def overflow_run(tc, R, build, only=None):
    """REV3 r01's overflow, generalized per the order: one finite key element 1e20 (squared norm overflows fp32)."""
    import tensor_cuda.quant as quant
    os.environ['TC_APA_SP'] = '1'
    os.environ.pop('TC_APA_SP_STORAGE', None)
    pm = tc.PackedRegisteredMargins.load(PACKED_TABLE)
    assert sha(DENSE_TABLE) == DENSE_TABLE_SHA
    dm = tc.RegisteredMargins.load(DENSE_TABLE, expected_sha256=DENSE_TABLE_SHA)
    res, checks = {}, []
    H, KV = 8, 2
    with tc.no_grad():
        for D in (128, 64):
            for s in ('qpos', 'qneg'):
                cid = f'ovf_D{D}_{s}'
                if only is not None and cid not in only:
                    continue

                def mutate(q, k, v, s=s):
                    if s == 'qneg':
                        q[...] = -q
                    k[0, 0, 100, 5] = OVF
                for dtype in ('bfloat16', 'float32'):
                    cc = Case(tc, R, 11, 1, H, KV, 1024, D, dtype, plant=False, mutate=mutate)
                    assert all(np.isfinite(x).all() for x in (cc.qn, cc.kn, cc.vn)), 'every input finite'
                    rec = res.setdefault(dtype, {}).setdefault(cid, {})
                    rec['stored_norm_ws_tq4'] = norm_repr(npf(cc.ws['tq4'].scales)[0, 0, 100])
                    dense = classify(npf(tc._C.apa_selective_attention_sp(cc.q, cc.k, cc.k, cc.v, cc.scale, 2.0,
                                                                             False, None, False)), H)
                    kqd = quant._quantize_keys(cc.k, *quant._tables(D, 4, KV, True, 'cuda'))
                    dq = classify(npf(tc._C.apa_selective_attention_sp(cc.q, cc.k, kqd, cc.v, cc.scale, 2.0,
                                                                          False, None, False)), H)
                    rec['dense_entry'], rec['dense_sp_quantized'] = dense, dq
                    l19(checks, cid + ' dense_entry', dtype, dense, H, 'all_finite')
                    l19(checks, cid + ' dense_sp_quantized', dtype, dq, H, None)
                    comps = [('dense_entry', dense), ('dense_sp_quantized', dq)]
                    for name in ('r2_tq4',):
                        try:
                            o = run_raw(tc, cc, name, delta=2.0, carry='partition', sinks=None, diag=False,
                                           build=build)[0]
                            rec[name] = classify(npf(o), H)
                        except Exception as e:   # a failed check, never a pass
                            rec[name] = dict(ERROR=repr(e)[:300])
                        for cname, comp in comps:
                            gate_checks(checks, cid, name, dtype, rec[name], comp, False, cname)
                    if dtype == 'float32':
                        kq = tc.quantize_sp2_keys(cc.k, 4)
                        assert isinstance(kq, tc.PackedSpKeys)
                        rec['stored_norm_public'] = norm_repr(npf(kq.norms)[0, 0, 100])
                        try:
                            out, opt = public_pair(tc, pm, dm, cc, None)
                            rec['public_default'], rec['dense_opt_out'] = classify(out, H), classify(opt, H)
                        except Exception as e:
                            rec['public_default'] = dict(ERROR=repr(e)[:300])
                            rec['dense_opt_out'] = dict(nonfinite=[], zero=[])
                        l19(checks, cid + ' dense_opt_out', dtype, rec['dense_opt_out'], H, None)
                        for cname, comp in comps + [('dense_opt_out', rec['dense_opt_out'])]:
                            gate_checks(checks, cid, 'public_default', dtype, rec['public_default'], comp, False,
                                           cname)
        for s, sign in (('qneg', -1.0), ('qpos', 1.0)):
            cid = f'r01_ovf_{s}'
            if only is not None and cid not in only:
                continue
            q = np.full((1, 1, 1, 64), sign, np.float32)
            k = np.zeros((1, 1, 3, 64), np.float32)
            k[0, 0, 1, 0] = OVF
            v = np.arange(192, dtype=np.float32).reshape(1, 1, 3, 64) / np.float32(256)
            Q, K, V = [tc.tensor(np.ascontiguousarray(x)) for x in (q, k, v)]
            pk = tc.quantize_sp2_keys(K, 4)
            assert isinstance(pk, tc.PackedSpKeys)
            dk = tc.quantize_sp2_keys(K, 4, storage='dense')
            rec = res.setdefault('float32', {}).setdefault(cid, {})
            rec['stored_norm_public'] = norm_repr(npf(pk.norms)[0, 0, 1])
            try:
                rec['public_default'] = classify(npf(tc.apa_selective_attention_sp(Q, K, pk, V, .125, .5, bulk_bits=4,
                                                                                      margins=pm)), 1)
            except Exception as e:
                rec['public_default'] = dict(ERROR=repr(e)[:300])
            rec['dense_opt_out'] = classify(npf(tc.apa_selective_attention_sp(Q, K, dk, V, .125, .5, bulk_bits=4,
                                                                                 margins=dm)), 1)
            rec['dense_entry'] = classify(npf(tc._C.apa_selective_attention_sp(Q, K, K, V, .125, 2.0, False, None,
                                                                                  False)), 1)
            l19(checks, cid + ' dense_entry', 'float32', rec['dense_entry'], 1, 'all_finite')
            l19(checks, cid + ' dense_opt_out', 'float32', rec['dense_opt_out'], 1, 'single_row')
            for comp in ('dense_entry', 'dense_opt_out'):
                gate_checks(checks, cid, 'public_default', 'float32', rec['public_default'], rec[comp], False, comp)
    return res, checks
