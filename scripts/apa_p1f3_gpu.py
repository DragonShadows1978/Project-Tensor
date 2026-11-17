#!/usr/bin/env python3
"""APA-P1-F3 GPU lines LF1, LF2, LF3, LF6 and MUT (run inside forge-gpu-run), plus their CPU steps.

Modes:
  stage <F3|mutant:<name>>   CPU: stage an importable arm (copies, never symlinks)
  shim                       CPU: compile F1's LD_PRELOAD cudaLaunchKernel counter (source imported) into scratch
  precheck <arm>             CPU, before the lease: Build F2 module == 6f03b2cd...; Build F3 module == its receipt
                             and the working tree's engine source map == the receipt's; mutants == their receipts
  cpu-stage <line>           CPU: run a line's code up to its first device call (staged-script lesson)
  lf1 | lf2 | lf3 | lf6 | mut     GPU lines (artifacts/apa_p1f3/EXPECTED.json gpu_lines)
  (lf1-arm, lf3-bnd-row, mut-child: child processes)

F2's line machinery is imported from scripts/apa_p1f2_gpu.py, not copied: lf2_run (F2's LF2 cases), gate_checks,
l19, classify, public_pair, expect_refusal, load_arm, child, child_env and cuda_used; with F1's (scripts/
apa_p1f1_gpu.py): Case, run_raw, lf1_arm, reimport, launches, SHIM_SOURCE, bitwise/informative. F2's module
globals that name its arms, receipts and scratch (arm_dir, arm_root, receipt, precheck, SHIM_DIR, SHIM, SCRATCH)
are pointed at this round's arms (F2 as the before-build, F3 as the new one) and scratch; nothing is edited.
Prior art: P1-D/F1/F2 paired unsigned-view bit equality, LD_PRELOAD launch counting and per-ID pytest receipts
(Project-Tensor 2026), imported; REV3's r01 overflow repro (restated: same shapes and values); IEEE 754 overflow
and non-finite classification; DLPack re-import with a hand-set shape (F1/F2). No new algorithm here.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/apa_p1f3'
F2_ART = ROOT / 'artifacts/apa_p1f2'
F2_SHA = '6f03b2cdd5d7f9bee4456ed34759c6c710cccfab9a50f9ac0fc6a678c076f579'
F2_MODULE = 'artifacts/apa_p1f2/arms/F2/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so'
TREE0 = Path('/tmp/apap1f3_tree0')        # git archive 25f59b9c (this branch as committed)
SHIM_DIR = Path('/tmp/apap1f3_shim')
SHIM = SHIM_DIR / 'liblaunchcount.so'
SCRATCH = '/tmp/apap1f3_gpu'
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tensor_cuda/tests'))
import apa_p1f1_gpu as f1  # noqa: E402
import apa_p1f2_gpu as f2  # noqa: E402
import apa_p1f3_build as f3b  # noqa: E402

sha, save, step, bitwise, informative, npf = f1.sha, f1.save, f1.step, f1.bitwise, f1.informative, f1.npf
OVF = float(np.float32(1e20))
INT_MAX = 2**31 - 1


# ------------------------------------------------------------------ arms and receipts
def arm_dir(name):
    if name == 'F2':
        return F2_ART / 'arms/F2'
    if name == 'F3':
        return ART / 'arms/F3'
    return ART / 'mutants' / name.split(':', 1)[1] / 'arm'


def arm_root(name):
    return ROOT


def mutant_spec(m):
    return {x['name']: x for x in json.loads((ART / 'mutants/MUTANTS.json').read_text())['mutants']}[m]


def mutant_receipt(m):
    recs = [json.loads(p.read_text()) for p in sorted((ART / 'mutants' / m).glob('build*/BUILD.json'))]
    recs = [r for r in recs if r.get('status') == 'PASS']
    assert recs, f'no passing build receipt for mutant {m}'
    return max(recs, key=lambda r: r.get('attempt', 1))


def receipt(name):
    if name == 'F2':
        return dict(status='PASS', label='F2', module=F2_MODULE, module_sha256=F2_SHA)
    if name == 'F3':
        return f3b.f3_receipt()
    m = name.split(':', 1)[1]
    if mutant_spec(m)['kind'] == 'python':
        return f3b.f3_receipt()
    return mutant_receipt(m)


def precheck(name):
    """Refuse (raise) a module whose sha differs from its receipt; for F3 also a drifted working tree (L103)."""
    rec = receipt(name)
    assert rec.get('status', 'PASS') == 'PASS'
    arm = json.loads((arm_dir(name) / 'ARM.json').read_text())
    got = sha(ROOT / arm['module'])
    assert got == rec['module_sha256'] == arm['module_sha256'], f'{name}: module sha {got} != receipt'
    if name == 'F2':
        assert got == F2_SHA, 'Build F2 module sha256 differs from 6f03b2cd...'
    if name == 'F3':
        now = f3b.f2b.f1b.engine_sources(ROOT)
        want = rec.get('source_sha256_after', rec['source_sha256'])
        drift = sorted(k for k in set(now) | set(want) if now.get(k) != want.get(k))
        assert not drift, f'F3: working tree differs from the build receipt: {drift}'
    return dict(arm=name, module_sha256=got)


# Point F2's machinery at this round's arms, receipts and scratch (module globals; nothing edited).
f2.arm_dir, f2.arm_root, f2.receipt, f2.precheck = arm_dir, arm_root, receipt, precheck
f2.SHIM_DIR, f2.SHIM, f2.SCRATCH = SHIM_DIR, SHIM, SCRATCH


def stage(name):
    if name == 'F3':
        rec = f3b.f3_receipt()
        module, src = ROOT / rec['module'], ROOT / 'tensor_cuda/tensor_cuda'
    else:
        m = name.split(':', 1)[1]
        spec = mutant_spec(m)
        rec = f3b.f3_receipt() if spec['kind'] == 'python' else mutant_receipt(m)
        module = ROOT / rec['module']
        src = (ART / 'mutants' / m / 'tree/tensor_cuda/tensor_cuda') if spec['kind'] == 'python' else \
            ROOT / 'tensor_cuda/tensor_cuda'
    assert rec['status'] == 'PASS' and sha(module) == rec['module_sha256']
    dest = arm_dir(name) / 'tensor_cuda'
    dest.mkdir(parents=True)
    files = {}
    for path in sorted(src.rglob('*.py')):
        if '__pycache__' in path.parts:
            continue
        out = dest / path.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, out)
        files['tensor_cuda/tensor_cuda/' + str(path.relative_to(src))] = sha(out)
    shutil.copyfile(module, dest / module.name)
    record = dict(arm=name, package=str(dest.relative_to(ROOT)), python_files=files, python_source=str(src),
                  module=str((dest / module.name).relative_to(ROOT)), module_sha256=sha(dest / module.name),
                  build_record=str(rec.get('label')))
    assert record['module_sha256'] == rec['module_sha256']
    (dest.parent / 'ARM.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))
    return 0


def build_shim():
    SHIM_DIR.mkdir(exist_ok=True)
    src = SHIM_DIR / 'launchcount.c'
    src.write_text(f1.SHIM_SOURCE)
    subprocess.run(['timeout', '60s', 'gcc', '-O2', '-shared', '-fPIC', str(src), '-o', str(SHIM), '-ldl'], check=True)
    print('shim', SHIM, sha(SHIM), 'source sha', hashlib.sha256(f1.SHIM_SOURCE.encode()).hexdigest())
    return 0


def ensure_counter():
    """Re-exec this script under the LD_PRELOAD launch counter, once (F1/F2 method)."""
    if f1.launches() is None:
        assert os.environ.get('LD_PRELOAD') != str(SHIM), 'counter preloaded but not visible'
        os.execve(sys.executable, [sys.executable, '-B', str(Path(__file__).resolve()), *sys.argv[1:]],
                  f2.child_env())


# ------------------------------------------------------------------ LF1
def lf1_arm(args):
    import apa_p1d_gpu as g
    name = args.arm
    f1.ART = F2_ART if name == 'F2' else ART    # f1.lf1_arm points P1-D's loader at <ART>/arms/<name>
    g.ROOT = ROOT
    f1.build_receipt = lambda n: dict(module_sha256=receipt(name)['module_sha256'])
    return f1.lf1_arm(args)


def lf1(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log, runs = [], []
    for name in ('F2', 'F3'):
        precheck(name)
        r = f2.child([sys.executable, '-B', __file__, 'lf1-arm', '--arm', name, '--out', str(out)],
                     out / f'arm_{name}.log', 1200, f2.child_env(preload=False))
        runs.append(r)
        step(log, f'lf1_arm_{name}_exit', exit=r['exit'])
        if r['exit']:
            save(out / 'RESULT.json', dict(line='LF1', status='FAIL_CHILD', runs=runs, steps=log, terminal='LF1_CHILD_FAILED'))
            return 1
    rows = []
    A, B = np.load(out / 'l1_F2/arm_B.npz'), np.load(out / 'l1_F3/arm_B.npz')
    missing = sorted(set(A.files) ^ set(B.files))
    for key in sorted(set(A.files) & set(B.files)):
        rows.append(dict(part='L1_grid', key=key, **bitwise(A[key], B[key])))
    RA, RB = np.load(out / 'raw_F2.npz'), np.load(out / 'raw_F3.npz')
    missing += sorted(set(RA.files) ^ set(RB.files))
    for key in sorted(set(RA.files) & set(RB.files)):
        rows.append(dict(part='raw', key=key, **bitwise(RA[key], RB[key])))
    step(log, 'lf1_compared', rows=len(rows))
    eq = sum(r['equal'] for r in rows)
    inf_both = sum(r.get('informative') == [True, True] for r in rows)
    uninformative = [r['key'] for r in rows if r.get('informative') != [True, True]]
    status = 'PASS' if rows and eq == len(rows) and not missing and not uninformative else 'FAIL'
    save(out / 'RESULT.json', dict(line='LF1', status=status, comparisons=len(rows), bitwise_equal=eq,
                                   both_sides_informative=inf_both, key_set_mismatch=missing,
                                   uninformative_arrays=uninformative,
                                   not_equal=[r for r in rows if not r['equal']][:50], runs=runs, steps=log,
                                   module_sha256=dict(F2=F2_SHA, F3=receipt('F3')['module_sha256']),
                                   terminal='LF1_DONE'))
    print('LF1', status, eq, '/', len(rows), 'informative both', inf_both, 'missing', len(missing), flush=True)
    return 0 if status == 'PASS' else 1


# ------------------------------------------------------------------ LF2 overflow cases
TQ4_ENTRIES = {('bfloat16', 128): ['r2_tq4', 'tc_tq4', 'eb2b', 'eb3', 'eb4'], ('bfloat16', 64): ['r2_tq4', 'tc_tq4'],
               ('float32', 128): ['r2_tq4'], ('float32', 64): ['r2_tq4']}
OVF_CASES = [f'ovf_D{D}_{s}' for D in (128, 64) for s in ('qpos', 'qneg')] + ['r01_ovf_qneg', 'r01_ovf_qpos']


def norm_repr(x):
    x = float(x)
    return 'nan' if math.isnan(x) else ('inf' if math.isinf(x) else repr(x))


def overflow_run(tc, R, build, only=None):
    """REV3 r01's overflow, generalized per the order: one finite key element 1e20 (squared norm overflows fp32)."""
    import tensor_cuda.quant as quant
    os.environ['TC_APA_SP'] = '1'
    os.environ.pop('TC_APA_SP_STORAGE', None)
    pm = tc.PackedRegisteredMargins.load(f2.PACKED_TABLE)
    assert sha(f2.DENSE_TABLE) == f2.DENSE_TABLE_SHA
    dm = tc.RegisteredMargins.load(f2.DENSE_TABLE, expected_sha256=f2.DENSE_TABLE_SHA)
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
                    cc = f1.Case(tc, R, 11, 1, H, KV, 1024, D, dtype, plant=False, mutate=mutate)
                    assert all(np.isfinite(x).all() for x in (cc.qn, cc.kn, cc.vn)), 'every input finite'
                    rec = res.setdefault(dtype, {}).setdefault(cid, {})
                    rec['stored_norm_ws_tq4'] = norm_repr(npf(cc.ws['tq4'].scales)[0, 0, 100])
                    dense = f2.classify(npf(tc._C.apa_selective_attention_sp(cc.q, cc.k, cc.k, cc.v, cc.scale, 2.0,
                                                                             False, None, False)), H)
                    kqd = quant._quantize_keys(cc.k, *quant._tables(D, 4, KV, True, 'cuda'))
                    dq = f2.classify(npf(tc._C.apa_selective_attention_sp(cc.q, cc.k, kqd, cc.v, cc.scale, 2.0,
                                                                          False, None, False)), H)
                    rec['dense_entry'], rec['dense_sp_quantized'] = dense, dq
                    f2.l19(checks, cid + ' dense_entry', dtype, dense, H, 'all_finite')
                    f2.l19(checks, cid + ' dense_sp_quantized', dtype, dq, H, None)
                    comps = [('dense_entry', dense), ('dense_sp_quantized', dq)]
                    for name in TQ4_ENTRIES[(dtype, D)]:
                        try:
                            o = f1.run_raw(tc, cc, name, delta=2.0, carry='partition', sinks=None, diag=False,
                                           build=build)[0]
                            rec[name] = f2.classify(o, H)
                        except Exception as e:   # a failed check, never a pass
                            rec[name] = dict(ERROR=repr(e)[:300])
                        for cname, comp in comps:
                            f2.gate_checks(checks, cid, name, dtype, rec[name], comp, False, cname)
                    if dtype == 'float32':
                        kq = tc.quantize_sp2_keys(cc.k, 4)
                        assert isinstance(kq, tc.PackedSpKeys)
                        rec['stored_norm_public'] = norm_repr(npf(kq.norms)[0, 0, 100])
                        try:
                            out, opt = f2.public_pair(tc, pm, dm, cc, None)
                            rec['public_default'], rec['dense_opt_out'] = f2.classify(out, H), f2.classify(opt, H)
                        except Exception as e:
                            rec['public_default'] = dict(ERROR=repr(e)[:300])
                            rec['dense_opt_out'] = dict(nonfinite=[], zero=[])
                        f2.l19(checks, cid + ' dense_opt_out', dtype, rec['dense_opt_out'], H, None)
                        for cname, comp in comps + [('dense_opt_out', rec['dense_opt_out'])]:
                            f2.gate_checks(checks, cid, 'public_default', dtype, rec['public_default'], comp, False,
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
                rec['public_default'] = f2.classify(npf(tc.apa_selective_attention_sp(Q, K, pk, V, .125, .5, bulk_bits=4,
                                                                                      margins=pm)), 1)
            except Exception as e:
                rec['public_default'] = dict(ERROR=repr(e)[:300])
            rec['dense_opt_out'] = f2.classify(npf(tc.apa_selective_attention_sp(Q, K, dk, V, .125, .5, bulk_bits=4,
                                                                                 margins=dm)), 1)
            rec['dense_entry'] = f2.classify(npf(tc._C.apa_selective_attention_sp(Q, K, K, V, .125, 2.0, False, None,
                                                                                  False)), 1)
            f2.l19(checks, cid + ' dense_entry', 'float32', rec['dense_entry'], 1, 'all_finite')
            f2.l19(checks, cid + ' dense_opt_out', 'float32', rec['dense_opt_out'], 1, 'single_row')
            for comp in ('dense_entry', 'dense_opt_out'):
                f2.gate_checks(checks, cid, 'public_default', 'float32', rec['public_default'], rec[comp], False, comp)
    return res, checks


def lf2(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = []
    precheck('F3')
    tc, arm = f2.load_arm('F3')
    R = f1.rev1()
    step(log, 'lf2_loaded', module_sha256=arm['module_sha256'])
    res, checks = f2.lf2_run(tc, R, 'F3')
    step(log, 'lf2_inherited_measured', checks=len(checks))
    ores, ochecks = overflow_run(tc, R, 'F3')
    step(log, 'lf2_overflow_measured', checks=len(ochecks))
    allc = checks + ochecks
    gated = [c for c in allc if not c.get('report')]
    status = 'PASS' if gated and all(c['ok'] for c in gated) else 'FAIL'
    norms = {f'{dt} {cid}': {k: v for k, v in r.items() if k.startswith('stored_norm')}
             for dt, cases in ores.items() for cid, r in cases.items()}
    save(out / 'RESULT.json', dict(line='LF2', arm='F3', module_sha256=arm['module_sha256'], status=status,
                                   gated=len(gated), passed=sum(c['ok'] for c in gated),
                                   gated_inherited=sum(not c.get('report') for c in checks),
                                   gated_overflow=sum(not c.get('report') for c in ochecks),
                                   failed=[c for c in gated if not c['ok']],
                                   loud_rows=[c for c in allc if c['id'].startswith('G2d') and c['extra_nonfinite']],
                                   overflow_stored_norms=norms, checks=allc, results=res, overflow_results=ores,
                                   steps=log, terminal='LF2_DONE'))
    print('LF2', status, sum(c['ok'] for c in gated), '/', len(gated), [c['id'] for c in gated if not c['ok']][:20],
          'norms', norms, flush=True)
    return 0 if status == 'PASS' else 1


# ------------------------------------------------------------------ LF3 (R3', R1')
class StatefulEntries(dict):
    """Hand-built margin mapping: pinned values to the first `pinned_reads` item reads, 0.0 afterwards."""

    def __init__(self, pins, pinned_reads):
        super().__init__(pins)
        self.pinned_reads, self.reads = pinned_reads, 0

    def __getitem__(self, key):
        self.reads += 1
        return dict.__getitem__(self, key) if self.reads <= self.pinned_reads else 0.0


def fake(tc, shape, dtype):
    """A tensor of the given shape over a small real buffer (F1 reimport(), imported; F2 LF3's method)."""
    return f1.reimport(tc, tc.empty((64,), dtype=dtype), tuple(shape), 0)


def eb2b_operands(tc, B, H, KV, S, cap, real=False):
    T = (lambda s, d: tc.empty(s, dtype=d)) if real else (lambda s, d: fake(tc, s, d))
    q = T((B, H, 1, 128), 'bfloat16')
    k, v = T((B, KV, S, 128), 'bfloat16'), T((B, KV, S, 128), 'bfloat16')
    vi, vs = T((B, KV, S, 128), 'uint8'), T((B, KV, S), 'float32')
    packed = (T((B, KV, cap, 64), 'uint8'), T((B, KV, cap), 'float32'), T((KV, 128, 128), 'float32'),
              tc.empty((16,), dtype='float32'))
    return q, k, v, vi, vs, packed


def r3p_rows(tc):
    """name -> (callable, expected exception type, substrings). Every operand geometry-consistent."""
    sp = tc._spo2
    scale = float(np.float32(1 / math.sqrt(128)))
    rows = {}

    def eb2b(ops):
        q, k, v, vi, vs, packed = ops
        return sp.eb2b_run(q, k, v, vi, vs, packed, scale, 1 / 32, delta=2.0, carry='partition')
    rows['eb2b_BH_at_bound'] = (lambda: eb2b(eb2b_operands(tc, 1, 536870911, 1, 1, 1)), RuntimeError,
                                ['apa_eb2b: geometry/GQA/range mismatch'])
    rows['eb2b_BH_past_bound'] = (lambda: eb2b(eb2b_operands(tc, 1, 536870912, 134217728, 1, 1)), ValueError,
                                  ['apa_eb2b', 'B*H', '536870912'])
    rows['eb2b_BH_2p32'] = (lambda: eb2b(eb2b_operands(tc, 65536, 65536, 8192, 1, 1)), ValueError,
                            ['apa_eb2b', 'B*H', '4294967296'])
    S = 2**31
    for dt, fmt, path, entry in (('float32', 'tq4', 'r2', 'packed_run_r2_fp32'), ('bfloat16', 'tq4', 'tc', 'packed_run_tc_tq4'),
                                 ('bfloat16', 'int4', 'tc', 'packed_run_tc_int4')):
        def fn(dt=dt, fmt=fmt, path=path):
            q = tc.empty((1, 8, 1, 128), dtype=dt)
            k, v = fake(tc, (1, 2, S, 128), dt), fake(tc, (1, 2, S, 128), dt)
            packed = (fake(tc, (1, 2, S, 64), 'uint8'), fake(tc, (1, 2, S), 'float32'),
                      tc.empty((2, 128, 128), dtype='float32'), tc.empty((16,), dtype='float32'))
            return sp.packed_run(q, k, v, packed, scale, 2.0, False, None, False, fmt=fmt, carry='partition', gqa_path=path)
        rows[f'extent_k_S_2p31_{entry}'] = (fn, ValueError, ['apa_spq4_run (packed_run)', "operand 'k' axis 2", '2147483648'])

    def q_big():
        q = fake(tc, (S, 8, 1, 128), 'float32')
        k, v = tc.empty((1, 2, 16, 128), dtype='float32'), tc.empty((1, 2, 16, 128), dtype='float32')
        packed = (tc.empty((1, 2, 16, 64), dtype='uint8'), tc.empty((1, 2, 16), dtype='float32'),
                  tc.empty((2, 128, 128), dtype='float32'), tc.empty((16,), dtype='float32'))
        return sp.packed_run(q, k, v, packed, scale, 2.0, False, None, False, fmt='tq4', carry='partition', gqa_path='r2')
    rows['extent_q_B_2p31_packed_run_r2_fp32'] = (q_big, ValueError, ['apa_spq4_run (packed_run)', "operand 'q' axis 0",
                                                                      '2147483648'])

    def big_eb(kind):
        q = tc.empty((1, 8, 1, 128), dtype='bfloat16')
        _, k, v, vi, vs, packed = eb2b_operands(tc, 1, 8, 2, S, S)
        packed = (packed[0], packed[1], tc.empty((2, 128, 128), dtype='float32'), packed[3])
        if kind == 'eb2b':
            return eb2b((q, k, v, vi, vs, packed))
        if kind == 'eb3':
            return sp.eb3_run(q, k, v, packed, scale, 2.0, None, 'partition', False, False, False, 0)
        return sp.eb4_run(q, k, v, vi, vs, packed, scale, 2.0, 1 / 32, None, 'partition', False, None, False, False)
    for kind in ('eb2b', 'eb3', 'eb4'):
        rows[f'extent_k_S_2p31_{kind}'] = (lambda kind=kind: big_eb(kind), ValueError,
                                           [f'apa_{kind}: ', "operand 'k' axis 2", '2147483648'])

    def zero(kind):
        dt = 'float32' if kind == 'packed_run_r2' else 'bfloat16'
        q = tc.empty((0, 8, 1, 128), dtype=dt)
        _, k, v, vi, vs, packed = eb2b_operands(tc, 1, 8, 2, 16, 16, real=True)
        if kind == 'packed_run_r2':
            k, v = tc.empty((1, 2, 16, 128), dtype='float32'), tc.empty((1, 2, 16, 128), dtype='float32')
            packed = (packed[0], packed[1], tc.empty((2, 128, 128), dtype='float32'), packed[3])
            return sp.packed_run(q, k, v, packed, scale, 2.0, False, None, False, fmt='tq4', carry='partition',
                                 gqa_path='r2')
        if kind == 'eb2b':
            return eb2b((q, k, v, vi, vs, packed))
        if kind == 'eb3':
            return sp.eb3_run(q, k, v, packed, scale, 2.0, None, 'partition', False, False, False, 0)
        return sp.eb4_run(q, k, v, vi, vs, packed, scale, 2.0, 1 / 32, None, 'partition', False, None, False, False)
    for kind in ('packed_run_r2', 'eb2b', 'eb3', 'eb4'):
        rows[f'zero_q_B0_{kind}'] = (lambda kind=kind: zero(kind), RuntimeError, ['dimensions must be positive'])
    return rows


def r1p_row(tc, c, kq, pm):
    """R1': the public route with a stateful hand-built mapping (pinned to 4 reads) vs the loaded pinned table."""
    P = tc.apa_packed
    ref = npf(tc.apa_selective_attention_sp(c.q, c.k, kq, c.v, c.scale, 0.01, bulk_bits=4, margins=pm))
    m = P.PackedRegisteredMargins(P.PACKED_TABLE_SHA256, StatefulEntries(dict(P.PACKED_PINNED_ENTRIES), 4))
    b0 = f1.launches()
    rec = dict(id="LF3 R1' stateful mapping (pinned to 4 reads, 0.0 after): runs with the pinned e_q, output bitwise "
                  "equal to the loaded pinned table's")
    try:
        got = npf(tc.apa_selective_attention_sp(c.q, c.k, kq, c.v, c.scale, 0.01, bulk_bits=4, margins=m))
        route = P.last_route('attention')
        b = bitwise(ref, got)
        rec.update(b, e_q=route['e_q'], mapping_reads=m.entries.reads, launches=f1.launches() - b0,
                   ok=b['equal'] and b['informative'] == [True, True] and
                   np.float64(route['e_q']).view(np.uint64) == np.float64(P.PACKED_PINNED_ENTRIES['4:128']).view(np.uint64)
                   and f1.launches() - b0 > 0)
    except BaseException as e:
        rec.update(ok=False, raised=type(e).__name__ + ': ' + str(e)[:300], mapping_reads=m.entries.reads)
    return rec


def lf3_checks(tc, R, names=None, with_r1=True):
    os.environ['TC_APA_SP'] = '1'
    os.environ.pop('TC_APA_SP_STORAGE', None)
    pm = tc.PackedRegisteredMargins.load(f2.PACKED_TABLE)
    checks = []
    with tc.no_grad():
        c = f1.Case(tc, R, 3, 1, 8, 2, 1024, 128, 'float32', plant=False)
        kq = tc.quantize_sp2_keys(c.k, 4)

        def controls():
            b0 = f1.launches()
            a = npf(tc._spo2.packed_rotate(c.q, c.ws['tq4'].rotation, 2, True))
            b = npf(tc.apa_selective_attention_sp(c.q, c.k, kq, c.v, c.scale, 0.01, bulk_bits=4, margins=pm))
            return a, b, f1.launches() - b0
        ca, cb, cl = controls()
        for name, (fn, exc, match) in r3p_rows(tc).items():
            if names is not None and name not in names:
                continue
            checks.append(f2.expect_refusal(dict(id=f"LF3 R3' row {name}", expected=f'{exc.__name__} {match}'), fn,
                                            exc, match))
        if with_r1:
            checks.append(r1p_row(tc, c, kq, pm))
        tc._C.synchronize()
        ca2, cb2, cl2 = controls()
        for label, x, y in (('packed_rotate', ca, ca2), ('public default route', cb, cb2)):
            b = bitwise(x, y)
            checks.append(dict(id=f'LF3 control after the refusals is bitwise equal: {label}',
                               ok=b['equal'] and b['informative'] == [True, True], **b))
        checks.append(dict(id='LF3 controls launched both times', ok=cl > 0 and cl2 > 0, before=cl, after=cl2))
    return checks


def lf3_bnd_row(args):
    """Build F2's behaviour on one LF3 input (child process; L60 record, no gate)."""
    rec = dict(row=args.row, arm='F2')
    try:
        tc, arm = f2.load_arm('F2')
        rec['module_sha256'] = arm['module_sha256']
        os.environ['TC_APA_SP'] = '1'
        R = f1.rev1()
        with tc.no_grad():
            if args.row == 'r1p_stateful_mapping':
                c = f1.Case(tc, R, 3, 1, 8, 2, 1024, 128, 'float32', plant=False)
                kq = tc.quantize_sp2_keys(c.k, 4)
                pm = tc.PackedRegisteredMargins.load(f2.PACKED_TABLE)
                r = r1p_row(tc, c, kq, pm)
                rec.update(result='returned' if 'raised' not in r else r['raised'],
                           e_q_at_use=r.get('e_q'), output_bitwise_equal_to_pinned=r.get('equal'),
                           mapping_reads=r.get('mapping_reads'), launches=r.get('launches'))
            else:
                fn = r3p_rows(tc)[args.row][0]
                b0, m0 = f1.launches(), f2.cuda_used()
                try:
                    fn()
                    rec['result'] = 'returned (silent acceptance)'
                except BaseException as e:
                    rec['result'] = type(e).__name__ + ': ' + str(e)[:300]
                rec['launches'] = None if b0 is None else f1.launches() - b0
                rec['device_bytes_in_use_delta'] = f2.cuda_used() - m0
        try:
            tc._C.synchronize()
            tc.tensor(np.arange(4, dtype=np.float32)).numpy()
            rec['context_after'] = 'usable'
        except BaseException as e2:
            rec['context_after'] = 'BROKEN: ' + str(e2)[:200]
    except BaseException as e:
        rec['harness_exception'] = type(e).__name__ + ': ' + str(e)[:300]
    print('BND_JSON ' + json.dumps(rec, default=str), flush=True)
    return 0


def lf3(args):
    ensure_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    precheck('F3')
    precheck('F2')
    log = []
    tc, arm = f2.load_arm('F3')
    R = f1.rev1()
    checks = lf3_checks(tc, R)
    step(log, 'lf3_f3_measured', checks=len(checks))
    status = 'PASS' if checks and all(c['ok'] for c in checks) else 'FAIL'
    save(out / 'RESULT.json', dict(line='LF3', arm='F3', module_sha256=arm['module_sha256'], status=status,
                                   method_nothing_launched='LD_PRELOAD counter of cudaLaunchKernel (F1 SHIM_SOURCE)',
                                   inputs='large extents by DLPack re-import over small real buffers (F1 reimport)',
                                   checks=checks, failed=[c['id'] for c in checks if not c['ok']], steps=log,
                                   terminal='LF3_F3_DONE'))
    print('LF3 F3', status, [c['id'] for c in checks if not c['ok']], flush=True)
    bnd, runs = [], []
    for row in list(r3p_rows(tc)) + ['r1p_stateful_mapping']:
        r = f2.child([sys.executable, '-B', __file__, 'lf3-bnd-row', '--row', row], out / f'bnd_F2_{row}.log', 240,
                     f2.child_env())
        runs.append(r)
        text = Path(r['log']).read_text()
        line = [l for l in text.splitlines() if l.startswith('BND_JSON ')]
        bnd.append(json.loads(line[-1][9:]) if line else dict(row=row, child_exit=r['exit'], note='no child result'))
        bnd[-1]['child_exit'] = r['exit']
    save(out / 'BND_F2.json', dict(arm='F2', module_sha256=F2_SHA, rows=bnd, runs=runs, terminal='LF3_BND_DONE'))
    print('LF3 BND rows', len(bnd), flush=True)
    return 0 if status == 'PASS' else 1


# ------------------------------------------------------------------ LF6
def lf6(args):
    import apa_p1d_gpu as g
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in ('F2', 'F3'):
        precheck(name)
    trees = {'F2': TREE0, 'F3': ROOT}
    assert (TREE0 / 'tensor_cuda/tests').is_dir(), 'stage /tmp/apap1f3_tree0 (git archive 25f59b9c) first'
    runs, log, results = [], [], []
    start = time.monotonic()
    for f in args.files:
        tag = Path(f).stem
        if time.monotonic() - start > args.max_seconds:
            results.append(dict(file=f, status='NOT_RUN_TIME_BUDGET'))
            continue
        recs = {}
        for armname in ('F2', 'F3'):
            arm = json.loads((arm_dir(armname) / 'ARM.json').read_text())
            env = f2.child_env(preload=False)
            env.pop('LD_PRELOAD', None)
            env.update(PYTHONPATH=str((ROOT / arm['package']).parent) + ':' + str(trees[armname] / 'tensor_cuda/tests') +
                       ':' + str(ROOT / 'artifacts/apa_p1'), PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', OMP_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', TMPDIR='/tmp/apap1f3_l6')
            Path(env['TMPDIR']).mkdir(exist_ok=True)
            rec_path = out / f'{tag}_{armname}.json'
            cmd = [sys.executable, str(ROOT / 'artifacts/apa_p1/pytest_receipt.py'), str(rec_path), '-q', '--tb=short',
                   '-ra', '-p', 'no:cacheprovider', '--basetemp', f'/tmp/apap1f3_l6/bt_{tag}_{armname}', f]
            t0 = time.monotonic()
            with (out / f'{tag}_{armname}.log').open('x') as fh:
                try:
                    code = subprocess.run(cmd, cwd=trees[armname], env=env, stdout=fh, stderr=subprocess.STDOUT,
                                          timeout=700).returncode
                except subprocess.TimeoutExpired:
                    code = 124
            runs.append(dict(file=f, arm=armname, exit=code, seconds=time.monotonic() - t0, cwd=str(trees[armname])))
            step(log, f'lf6_{tag}_{armname}', exit=code)
            recs[armname] = json.loads(rec_path.read_text()) if rec_path.exists() else None
        a, b = recs['F2'], recs['F3']
        if a is None or b is None:
            results.append(dict(file=f, status='FAIL_NO_RECEIPT'))
            continue
        rows = g.outcome_rows(a, b)
        assert a['module']['sha256'] == F2_SHA and b['module']['sha256'] == receipt('F3')['module_sha256']
        results.append(dict(file=f, ids=len(rows), collected_equal=a['collection'] == b['collection'],
                            collection_errors=[len(a['collection_errors']), len(b['collection_errors'])],
                            differences=sum(not r['equal'] for r in rows), outcomes_F2=g._count(a),
                            outcomes_F3=g._count(b), rows=rows))
    complete = all('ids' in r for r in results)
    ok = complete and all(r.get('differences') == 0 and r.get('collected_equal') for r in results)
    status = 'PASS' if ok else ('FAIL' if complete or any(r.get('differences') for r in results) else 'PARTIAL')
    save(out / 'RESULT.json', dict(line='LF6', status=status, complete=complete, files=args.files, results=results,
                                   runs=runs, steps=log, terminal='LF6_DONE' if complete else 'LF6_PARTIAL'))
    print('LF6', status, [(r['file'], r.get('ids'), r.get('differences'), r.get('status')) for r in results], flush=True)
    return 0 if ok else 1


# ------------------------------------------------------------------ MUT
def mut_child(args):
    if args.line == 'LF3':
        ensure_counter()
    name = args.arm
    tc, arm = f2.load_arm(name)
    R = f1.rev1()
    spec = mutant_spec(name.split(':', 1)[1])['gpu']
    os.environ['TC_APA_SP'] = '1'
    norms = None
    if spec['line'] == 'LF2':
        res, checks = overflow_run(tc, R, name, only=set(spec['cases']))
        norms = {f'{dt} {cid}': {k: v for k, v in r.items() if k.startswith('stored_norm')}
                 for dt, cases in res.items() for cid, r in cases.items()}
    else:
        checks = lf3_checks(tc, R, names=set(spec['rows']), with_r1=False)
    print('MUT_JSON ' + json.dumps(dict(arm=name, module_sha256=arm['module_sha256'], stored_norms=norms,
                                        checks=[dict(id=c['id'], ok=c['ok'], raised=c.get('raised'),
                                                     launches=c.get('launches')) for c in checks]), default=str),
          flush=True)
    return 0


def mut(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    m = json.loads((ART / 'mutants/MUTANTS.json').read_text())
    log, results = [], []
    for full in [x for x in m['mutants'] if 'gpu' in x]:
        spec = dict(full['gpu'], name=full['name'])
        arm = 'mutant:' + spec['name']
        precheck(arm)
        logf = out / f'mut_{spec["name"]}.log'
        r = f2.child([sys.executable, '-B', __file__, 'mut-child', '--arm', arm, '--line', spec['line'], '--out', str(out)],
                     logf, 900, f2.child_env())
        text = logf.read_text()
        line = [l for l in text.splitlines() if l.startswith('MUT_JSON ')]
        detail = json.loads(line[-1][9:]) if line else dict(exit=r['exit'])
        detail['child_exit'] = r['exit']
        failed_ids = [c['id'] for c in detail.get('checks', []) if not c['ok']]
        killed = spec['expected_failure'] in failed_ids
        results.append(dict(name=spec['name'], line=spec['line'], expected_failure=spec['expected_failure'],
                            killed=killed, failed_assertions=failed_ids, detail=detail))
        step(log, f'mut_{spec["name"]}', killed=killed)
        print('MUT', spec['name'], 'KILLED' if killed else 'SURVIVED', flush=True)
    status = 'PASS' if results and all(r['killed'] for r in results) else 'FAIL'
    save(out / 'RESULT.json', dict(line='MUT', status=status, results=results, steps=log, terminal='MUT_DONE'))
    print('MUT', status, sum(r['killed'] for r in results), '/', len(results), flush=True)
    return 0 if status == 'PASS' else 1


# ------------------------------------------------------------------ CPU staging check
def cpu_stage(args):
    """Run the lines' code up to the first device call on CPU (imports, receipts, table loads, case lists)."""
    rec = dict(line=args.target, steps=[])
    for name in ('F2', 'F3'):
        rec['steps'].append(f'precheck {name}: {precheck(name)}')
    tc, arm = f2.load_arm('F3')
    rec['steps'].append(f'load_arm F3 ok {arm["module_sha256"]}')
    pm = tc.PackedRegisteredMargins.load(f2.PACKED_TABLE)
    rec['steps'].append('pinned table loads: ' + pm.sha256)
    assert sha(f2.DENSE_TABLE) == f2.DENSE_TABLE_SHA
    tc.RegisteredMargins.load(f2.DENSE_TABLE, expected_sha256=f2.DENSE_TABLE_SHA)
    rec['steps'].append('dense table loads')
    import tensor_cuda.quant as quant
    rec['quant_has'] = [hasattr(quant, n) for n in ('_tables', '_quantize_keys')]
    R = f1.rev1()
    rec['steps'].append('rev1 repro imported')
    rec['lf2_inherited_cases'] = [c['id'] for c in f2.lf2_cases()]
    rec['lf2_overflow_cases'] = OVF_CASES
    rec['f2_imported'] = all(hasattr(f2, n) for n in ('lf2_run', 'gate_checks', 'l19', 'classify', 'public_pair',
                                                     'expect_refusal', 'load_arm', 'child', 'child_env'))
    rec['f1_imported'] = all(hasattr(f1, n) for n in ('Case', 'run_raw', 'reimport', 'lf1_arm', 'launches', 'SHIM_SOURCE'))
    rec['tree0'] = (TREE0 / 'tensor_cuda/tests').is_dir()
    m = (ART / 'mutants/MUTANTS.json')
    rec['mutants'] = [x['name'] for x in json.loads(m.read_text())['mutants']] if m.exists() else None
    try:
        rows = r3p_rows(tc)
        rec['lf3_rows'] = list(rows)
        rows['eb2b_BH_2p32'][0]()
        rec['first_device_call'] = 'unexpectedly succeeded on CPU'
    except BaseException as e:
        rec['first_device_call'] = 'stopped at the first device call as expected: ' + type(e).__name__ + ': ' + str(e)[:200]
    print('CPU_STAGE_JSON ' + json.dumps(rec, default=str))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode')
    ap.add_argument('target', nargs='?')
    ap.add_argument('--out')
    ap.add_argument('--arm')
    ap.add_argument('--row')
    ap.add_argument('--line')
    ap.add_argument('--files', nargs='*')
    ap.add_argument('--max-seconds', type=float, default=1500)
    args = ap.parse_args()
    if args.mode == 'stage':
        return stage(args.target)
    if args.mode == 'shim':
        return build_shim()
    if args.mode == 'precheck':
        print(json.dumps(precheck(args.target)))
        return 0
    if args.mode == 'cpu-stage':
        return cpu_stage(args)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') != '', 'GPU modes run inside forge-gpu-run'
    fn = {'lf1': lf1, 'lf1-arm': lf1_arm, 'lf2': lf2, 'lf3': lf3, 'lf3-bnd-row': lf3_bnd_row, 'lf6': lf6,
          'mut': mut, 'mut-child': mut_child}[args.mode]
    try:
        code = fn(args)
    except BaseException:
        if args.out:
            Path(args.out).mkdir(parents=True, exist_ok=True)
            (Path(args.out) / f'exception_{args.mode}_{time.time_ns()}.txt').write_text(traceback.format_exc())
        raise
    raise SystemExit(code)


if __name__ == '__main__':
    raise SystemExit(main())
