#!/usr/bin/env python3
"""APA-P1-F2 GPU lines LF1-LF4, LF6 and MUT (run inside forge-gpu-run), plus their CPU steps.

Modes:
  stage <F2|mutant:<name>>   CPU: stage an importable arm (copies, never symlinks)
  shim                       CPU: compile F1's LD_PRELOAD cudaLaunchKernel counter (source imported) into scratch
  precheck <arm>             CPU, before the lease: module sha vs receipt; Build F vs 5c171477...; for F2 the
                             working tree's engine source map vs the build receipt (L103)
  cpu-stage <line>           CPU: run a line's code up to its first device call (staged-script lesson)
  lf1 | lf2 | lf3 | lf4 | lf6 | mut     GPU lines (artifacts/apa_p1f2/EXPECTED.json gpu_lines)
  (lf1-arm, lf3-bnd-row, mut-child: child processes)

F1's line machinery is imported from scripts/apa_p1f1_gpu.py, not copied: Case (REV1 inputs + packed
state), run_raw (every raw packed entry), row_sets, bitwise/uview/informative, the launch-counter shim
source and launches(), reimport (DLPack re-import), and lf1_arm (F1's LF1 cases, with its ART/arm/receipt
globals pointed at each build). P1-D's l1_arm, cuda_used and outcome_rows are imported likewise.
Prior art: P1-D/F1 paired unsigned-view bit equality and per-ID pytest receipts (Project-Tensor 2026);
REV1/REV2 repro inputs (imported or restated unchanged); LD_PRELOAD interposition; IEEE 754 non-finite
classification; NVIDIA cudaMemGetInfo. No new algorithm here.
"""
import argparse
import ctypes
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
ART = ROOT / 'artifacts/apa_p1f2'
P1D_WT = Path('/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1d')
F_SHA = '5c17147715ffc422b0aa7ba8fae826c554a4caff63b1231c22ce580867efc243'
TREE_F0 = Path('/tmp/apap1f2_treeF0')        # git archive d01118b4 (this branch as committed)
SHIM_DIR = Path('/tmp/apap1f2_shim')
SHIM = SHIM_DIR / 'liblaunchcount.so'
SCRATCH = '/tmp/apap1f2_gpu'
DENSE_TABLE = Path('/mnt/ForgeRealm/Project-Tensor/artifacts/apa_sp2/e_q_table.json')
DENSE_TABLE_SHA = '5b6f8bfb405b40c03da21ad8d50eb40c11b85ed9559ee717d2a105c90c0e618d'
PACKED_TABLE = ROOT / 'artifacts/apa_p1d/packed_e_q_table.json'
FLT_MAX = float(np.finfo(np.float32).max)
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tensor_cuda/tests'))
import apa_p1f1_gpu as f1  # noqa: E402
import apa_p1f1_build as f1b  # noqa: E402

sha, save, step, bitwise, uview, informative, npf = f1.sha, f1.save, f1.step, f1.bitwise, f1.uview, f1.informative, f1.npf


# ------------------------------------------------------------------ arms and receipts
def f2_receipt():
    recs = [json.loads(p.read_text()) for p in sorted(ART.glob('build_F2*/BUILD.json'))]
    recs = [r for r in recs if r.get('status') == 'PASS' and r['label'] == 'F2']
    assert recs, 'no passing Build F2 receipt'
    return max(recs, key=lambda r: r.get('attempt', 1))


def arm_dir(name):
    if name == 'F':
        return P1D_WT / 'artifacts/apa_p1f1/arms/F'
    if name == 'F2':
        return ART / 'arms/F2'
    return ART / 'mutants' / name.split(':', 1)[1] / 'arm'


def arm_root(name):
    return P1D_WT if name == 'F' else ROOT


def receipt(name):
    if name == 'F':
        return dict(module='artifacts/apa_p1f1/arms/F/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so',
                    module_sha256=F_SHA)
    if name == 'F2':
        return f2_receipt()
    m = name.split(':', 1)[1]
    if json.loads((ART / 'mutants' / m / 'MUTANT.json').read_text())['kind'] == 'python':
        return f2_receipt()   # a Python-only mutant runs on Build F2's module (repair: precheck receipt)
    return mutant_receipt(m)


def mutant_receipt(m):
    recs = [json.loads(p.read_text()) for p in sorted((ART / 'mutants' / m).glob('build*/BUILD.json'))]
    recs = [r for r in recs if r.get('status') == 'PASS']
    assert recs, f'no passing build receipt for mutant {m}'
    return max(recs, key=lambda r: r.get('attempt', 1))


def precheck(name):
    """Refuse (raise) a module whose sha differs from its receipt, and for F2 a drifted working tree."""
    rec = receipt(name)
    assert rec.get('status', 'PASS') == 'PASS'
    arm = json.loads((arm_dir(name) / 'ARM.json').read_text())
    got = sha(arm_root(name) / arm['module'])
    assert got == rec['module_sha256'] == arm['module_sha256'], f'{name}: module sha {got} != receipt'
    if name == 'F':
        assert got == F_SHA
    if name == 'F2':
        now = f1b.engine_sources(ROOT)
        want = rec.get('source_sha256_after', rec['source_sha256'])
        drift = sorted(k for k in set(now) | set(want) if now.get(k) != want.get(k))
        assert not drift, f'F2: working tree differs from the build receipt: {drift}'
    return dict(arm=name, module_sha256=got)


def stage(name):
    if name == 'F2':
        rec = f2_receipt()
        module, src = ROOT / rec['module'], ROOT / 'tensor_cuda/tensor_cuda'
    else:
        mdir = ART / 'mutants' / name.split(':', 1)[1]
        spec = json.loads((mdir / 'MUTANT.json').read_text())
        rec = mutant_receipt(mdir.name) if spec['kind'] != 'python' else f2_receipt()
        module = ROOT / rec['module']
        src = (mdir / 'tree/tensor_cuda/tensor_cuda') if spec['kind'] == 'python' else ROOT / 'tensor_cuda/tensor_cuda'
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


def load_arm(name):
    arm = json.loads((arm_dir(name) / 'ARM.json').read_text())
    root = arm_root(name)
    package = root / arm['package']
    for path, digest in arm['python_files'].items():
        assert sha(package / Path(path).relative_to('tensor_cuda/tensor_cuda')) == digest, ('arm python drift', path)
    assert sha(root / arm['module']) == arm['module_sha256']
    sys.path.insert(0, str(package.parent))
    import tensor_cuda as tc
    assert Path(tc.__file__).resolve().parent == package.resolve(), tc.__file__
    assert sha(tc._C.__file__) == arm['module_sha256']
    return tc, arm


# ------------------------------------------------------------------ processes
def build_shim():
    SHIM_DIR.mkdir(exist_ok=True)
    src = SHIM_DIR / 'launchcount.c'
    src.write_text(f1.SHIM_SOURCE)
    subprocess.run(['timeout', '60s', 'gcc', '-O2', '-shared', '-fPIC', str(src), '-o', str(SHIM), '-ldl'], check=True)
    print('shim', SHIM, sha(SHIM), 'source sha', hashlib.sha256(f1.SHIM_SOURCE.encode()).hexdigest())


def child_env(preload=True):
    env = os.environ.copy()
    env.pop('TC_APA_SP_STORAGE', None)
    env['LD_PRELOAD'] = str(SHIM) if preload else ''
    env.update(PYTHONDONTWRITEBYTECODE='1', TMPDIR=SCRATCH)
    Path(SCRATCH).mkdir(exist_ok=True)
    return env


def child(argv, log, limit, env):
    t0 = time.monotonic()
    with Path(log).open('x') as f:
        try:
            code = subprocess.run(argv, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=limit, env=env).returncode
        except subprocess.TimeoutExpired:
            code = 124
    return dict(argv=[str(a) for a in argv], exit=code, seconds=time.monotonic() - t0, log=str(log))


def ensure_counter():
    """Re-exec under the LD_PRELOAD launch counter (F1's LF4 repair), once."""
    if f1.launches() is None:
        assert os.environ.get('LD_PRELOAD') != str(SHIM), 'counter preloaded but not visible'
        os.execve(sys.executable, [sys.executable, '-B', str(Path(__file__).resolve()), *sys.argv[1:]], child_env())


def cuda_used():
    import apa_p1d_gpu as g
    return g.cuda_used()


# ------------------------------------------------------------------ LF1
def lf1_arm(args):
    import apa_p1d_gpu as g
    name = args.arm
    if name == 'F':
        f1.ART, g.ROOT = P1D_WT / 'artifacts/apa_p1f1', P1D_WT
    else:
        f1.ART, g.ROOT = ART, ROOT
    f1.build_receipt = lambda n: dict(module_sha256=receipt(name)['module_sha256'])
    return f1.lf1_arm(args)


def lf1(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log, runs = [], []
    for name in ('F', 'F2'):
        precheck(name)
        r = child([sys.executable, '-B', __file__, 'lf1-arm', '--arm', name, '--out', str(out)], out / f'arm_{name}.log',
                  1200, child_env(preload=False))
        runs.append(r)
        step(log, f'lf1_arm_{name}_exit', exit=r['exit'])
        if r['exit']:
            save(out / 'RESULT.json', dict(line='LF1', status='FAIL_CHILD', runs=runs, steps=log, terminal='LF1_CHILD_FAILED'))
            return 1
    rows = []
    A, B = np.load(out / 'l1_F/arm_B.npz'), np.load(out / 'l1_F2/arm_B.npz')
    l1_missing = sorted(set(A.files) ^ set(B.files))
    for key in sorted(set(A.files) & set(B.files)):
        rows.append(dict(part='L1_grid', key=key, **bitwise(A[key], B[key])))
    RA, RB = np.load(out / 'raw_F.npz'), np.load(out / 'raw_F2.npz')
    missing = sorted(set(RA.files) ^ set(RB.files)) + l1_missing
    for key in sorted(set(RA.files) & set(RB.files)):
        rows.append(dict(part='raw', key=key, **bitwise(RA[key], RB[key])))
    step(log, 'lf1_compared', rows=len(rows))
    eq = sum(r['equal'] for r in rows)
    inf_both = sum(r.get('informative') == [True, True] for r in rows)
    uninformative = [r['key'] for r in rows if r.get('informative') != [True, True]]
    out_uninformative = [k for k in uninformative if k.endswith('/out') or k.endswith('/0') or k.endswith('/plain')]
    status = 'PASS' if eq == len(rows) and not missing and not out_uninformative else 'FAIL'
    save(out / 'RESULT.json', dict(line='LF1', status=status, comparisons=len(rows), bitwise_equal=eq,
                                   both_sides_informative=inf_both, key_set_mismatch=missing,
                                   uninformative_arrays=uninformative, uninformative_outputs=out_uninformative,
                                   not_equal=[r for r in rows if not r['equal']][:50], runs=runs, steps=log,
                                   module_sha256=dict(F=F_SHA, F2=receipt('F2')['module_sha256']), terminal='LF1_DONE'))
    print('LF1', status, eq, '/', len(rows), 'informative both', inf_both, 'uninformative outputs', len(out_uninformative),
          flush=True)
    return 0 if status == 'PASS' else 1


# ------------------------------------------------------------------ LF2
QKV_POS = {'q': [(0, 0, 0, 5), (0, 5, 0, 77), (0, 7, 0, 127)], 'k': [(0, 0, 100, 5), (0, 1, 700, 64), (0, 0, 1023, 127)],
           'v': [(0, 0, 100, 5), (0, 1, 700, 64), (0, 1, 0, 0)]}
VALUES = {'nan': np.nan, 'posinf': np.inf, 'neginf': -np.inf}
SINK_ENTRIES = {'bfloat16': ['d1_ordered', 'd1_tensor', 'r2_tq4', 'r2_int4', 'tc_tq4', 'tc_int4', 'eb3', 'eb4'],
                'float32': ['d1_ordered', 'r2_tq4', 'r2_int4']}


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
    cc = f1.Case(tc, R, 11, 1, 8, 2, 1024, 128, dtype, plant=False, mutate=mutate, sinks=c['group'] == 'sinks')
    if c['group'] == 'sinks':
        s = f1.npf(cc.sinks).astype(np.float32)
        s[2] = c['value']
        cc.sinks = tc.tensor(np.ascontiguousarray(s), dtype=dtype)
    if c['group'] == 'flt_max_g4':
        cc.scale = FLT_MAX
    return cc


def classify(o, H):
    nonfinite, zero = f1.row_sets(np.asarray(o), H)
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
                names = SINK_ENTRIES[dtype] if c['group'] == 'sinks' else f1.NAN_ENTRIES[dtype]
                for name in names:
                    try:
                        o = f1.run_raw(tc, cc, name, delta=2.0, carry='partition', sinks=sinks, diag=False, build=build)[0]
                        rec[name] = classify(o, H)
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
            cc = f1.Case(tc, R, 11, 1, 1, 1, 3, 128, 'float32', plant=False, mutate=mutate)
            cc.scale = FLT_MAX
            dense = classify(npf(tc._C.apa_selective_attention_sp(cc.q, cc.k, cc.k, cc.v, cc.scale, 2.0, False, None,
                                                                  False)), 1)
            rec = res.setdefault('float32', {}).setdefault('flt_max_r02', {})
            rec['dense'] = dense
            l19(checks, 'flt_max_r02', 'float32', dense, 1, 'single_row')
            for name in ('d1_ordered', 'r2_tq4', 'r2_int4'):
                try:
                    rec[name] = classify(f1.run_raw(tc, cc, name, delta=2.0, carry='partition', sinks=None, diag=False,
                                                    build=build)[0], 1)
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


def lf2(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = []
    precheck('F2')
    tc, arm = load_arm('F2')
    R = f1.rev1()
    step(log, 'lf2_loaded', module_sha256=arm['module_sha256'])
    res, checks = lf2_run(tc, R, 'F2')
    step(log, 'lf2_measured', checks=len(checks))
    gated = [c for c in checks if not c.get('report')]
    status = 'PASS' if gated and all(c['ok'] for c in gated) else 'FAIL'
    save(out / 'RESULT.json', dict(line='LF2', arm='F2', module_sha256=arm['module_sha256'], status=status,
                                   gated=len(gated), passed=sum(c['ok'] for c in gated),
                                   failed=[c for c in gated if not c['ok']],
                                   loud_rows=[c for c in checks if c['id'].startswith('G2d') and c['extra_nonfinite']],
                                   checks=checks, results=res, steps=log, terminal='LF2_DONE'))
    print('LF2', status, sum(c['ok'] for c in gated), '/', len(gated), [c['id'] for c in gated if not c['ok']][:20],
          flush=True)
    return 0 if status == 'PASS' else 1


# ------------------------------------------------------------------ LF3
def fake_capacity(tc, kv, D, cap):
    """codes [1,kv,cap,D/2] uint8 and scales [1,kv,cap] fp32 over small real buffers (F1 reimport(), imported)."""
    codes = f1.reimport(tc, tc.empty((64,), dtype='uint8'), (1, kv, cap, D // 2), 0)
    scales = f1.reimport(tc, tc.empty((16,), dtype='float32'), (1, kv, cap), 0)
    return codes, scales


def lf3_inputs(tc, R):
    E = lambda shape, dt='float32': tc.empty(shape, dtype=dt)
    c = f1.Case(tc, R, 3, 1, 8, 2, 1024, 128, 'float32', plant=False)
    cb = f1.Case(tc, R, 3, 1, 8, 2, 1024, 128, 'bfloat16', plant=False)
    rot = c.ws['tq4'].rotation
    bnd = c.ws['tq4'].boundaries
    rot0 = lambda kv: E((kv, 128, 128))
    rows = {}
    rows['rotate_q_B0'] = lambda: tc._spo2.packed_rotate(E((0, 8, 1, 128)), rot, 2, True)
    rows['rotate_q_H0'] = lambda: tc._spo2.packed_rotate(E((1, 0, 1, 128)), rot, 2, True)
    rows['prepare_k_B0'] = lambda: tc._spo2.packed_prepare(E((0, 2, 64, 128)), rot, 0)
    rows['prepare_k_KV0'] = lambda: tc._spo2.packed_prepare(E((1, 0, 64, 128)), rot0(0), 0)
    rows['prepare_k_S0'] = lambda: tc._spo2.packed_prepare(E((1, 2, 0, 128)), rot, 0)
    for fmt in ('tq4', 'int4'):
        for label, (B, KV) in (('B0', (0, 2)), ('KV0', (1, 0))):
            def fn(B=B, KV=KV, fmt=fmt):
                return tc._spo2.packed_append(E((B, KV, 64, 128)), E((B, KV, 64, 64), 'uint8'), E((B, KV, 64)),
                                              E((B, KV, 64, 128)), E((B * KV * 64,)), bnd, 0, fmt == 'tq4')
            rows[f'append_{fmt}_k_{label}'] = fn
    rows['int4_stats_q_B0'] = lambda: tc._spo2.int4_stats(E((0, 8, 1, 128)), E((0, 2, 64, 64), 'uint8'), E((0, 2, 64)),
                                                         64, c.scale, 1.0)
    rows['int4_stats_q_H0'] = lambda: tc._spo2.int4_stats(E((1, 0, 1, 128)), E((1, 2, 64, 64), 'uint8'), E((1, 2, 64)),
                                                         64, c.scale, 1.0)
    rows['int4_stats_codes_KV0'] = lambda: tc._spo2.int4_stats(c.q, E((1, 0, 64, 64), 'uint8'), E((1, 0, 64)), 64,
                                                              c.scale, 1.0)
    for label, shape in (('B0', (0, 2, 64, 128)), ('KV0', (1, 0, 64, 128)), ('S0', (1, 2, 0, 128))):
        rows[f'eb2b_quantize_v_{label}'] = lambda shape=shape: tc._spo2.eb2b_quantize(E(shape, 'bfloat16'))
        flags = (shape[0], shape[1], (shape[2] + 63) // 64)
        rows[f'eb2b_values_v_{label}'] = lambda shape=shape, flags=flags: tc._spo2.eb2b_values(
            E(shape, 'bfloat16'), E(shape, 'uint8'), E(shape[:3]), E(flags, 'uint8'))
    return c, cb, rows


def eb3_case(tc, R, G):
    return f1.Case(tc, R, 5, 1, G, 1, 256, 128, 'bfloat16', plant=False)


def margin_rows(tc, c, pm):
    P = tc.apa_packed
    from types import MappingProxyType
    pins = dict(P.PACKED_PINNED_ENTRIES)

    class Experimental(tc.PackedRegisteredMargins):
        pass
    replaced = tc.PackedRegisteredMargins.load(PACKED_TABLE)
    object.__setattr__(replaced, 'entries', MappingProxyType({'4:64': 0.0, '4:128': 0.0}))
    tables = {
        'margins_wrong_sha': tc.PackedRegisteredMargins('0' * 64, MappingProxyType(pins)),
        'margins_pinned_sha_zero_entries': tc.PackedRegisteredMargins(P.PACKED_TABLE_SHA256, MappingProxyType(
            {'4:64': 0.0, '4:128': 0.0})),
        'margins_pinned_sha_extra_key': tc.PackedRegisteredMargins(P.PACKED_TABLE_SHA256, MappingProxyType(
            dict(pins, **{'2:64': 0.0}))),
        'margins_subclass': Experimental(P.PACKED_TABLE_SHA256, MappingProxyType(pins)),
        'margins_replaced_after_load': replaced,
    }
    kq = tc.quantize_sp2_keys(c.k, 4)
    rows = {name: (lambda m=m: tc.apa_selective_attention_sp(c.q, c.k, kq, c.v, c.scale, 0.01, bulk_bits=4, margins=m))
            for name, m in tables.items()}
    rows['margins_load_override'] = lambda: tc.PackedRegisteredMargins.load(PACKED_TABLE, expected_sha256='0' * 64)
    return rows, kq


def expect_refusal(rec, fn, exc, match):
    b0 = f1.launches()
    m0 = cuda_used()
    try:
        fn()
        rec.update(raised=None, ok=False, note='DID NOT RAISE')
    except BaseException as e:
        rec['raised'] = type(e).__name__ + ': ' + str(e)[:300]
        rec['ok'] = isinstance(e, exc) and all(m in str(e) for m in match) and 'CUDA error' not in str(e)
    rec['launches'] = None if b0 is None else f1.launches() - b0
    rec['device_bytes_in_use_delta'] = cuda_used() - m0
    rec['ok'] = bool(rec['ok'] and rec['launches'] == 0)
    return rec


def lf3_f2(log):
    tc, arm = load_arm('F2')
    os.environ['TC_APA_SP'] = '1'
    os.environ.pop('TC_APA_SP_STORAGE', None)
    R = f1.rev1()
    pm = tc.PackedRegisteredMargins.load(PACKED_TABLE)
    checks = []
    with tc.no_grad():
        c, cb, helper_rows = lf3_inputs(tc, R)
        kq = tc.quantize_sp2_keys(c.k, 4)

        def controls():
            b0 = f1.launches()
            a = npf(tc._spo2.packed_rotate(c.q, c.ws['tq4'].rotation, 2, True))
            b = npf(tc.apa_selective_attention_sp(c.q, c.k, kq, c.v, c.scale, 0.01, bulk_bits=4, margins=pm))
            return a, b, f1.launches() - b0
        ca, cbv, cl = controls()
        step(log, 'lf3_controls_before', launches=cl)
        for name, fn in helper_rows.items():
            checks.append(expect_refusal(dict(id=f'LF3 raw helper zero extent raises ValueError, no launch: {name}'), fn,
                                         ValueError, ['positive']))
        # EB3 timeline
        for G in (3, 5, 6, 7, 8):
            e = eb3_case(tc, R, G)
            checks.append(expect_refusal(dict(id=f'LF3 EB3 timeline=True G={G} raises before any launch'),
                                         lambda e=e: tc.apa_eb3(e.q, e.k, e.v, e.ws['tq4'].packed, e.scale, 2.0,
                                                                timeline=True), ValueError, ['timeline']))
        for G in (2, 4):
            e = eb3_case(tc, R, G)
            b0 = f1.launches()
            try:
                r = tc.apa_eb3(e.q, e.k, e.v, e.ws['tq4'].packed, e.scale, 2.0, timeline=True)
                o = npf(r[0])
                checks.append(dict(id=f'LF3 EB3 timeline=True G={G} still runs', ok=informative(o) and f1.launches() - b0 > 0,
                                   launches=f1.launches() - b0))
            except BaseException as ex:
                checks.append(dict(id=f'LF3 EB3 timeline=True G={G} still runs', ok=False, raised=repr(ex)[:300]))
        # margins
        mrows, _ = margin_rows(tc, c, pm)
        for name, fn in mrows.items():
            checks.append(expect_refusal(dict(id=f'LF3 hand-built margin table refused before any launch: {name}'), fn,
                                         ValueError, ['pin']))
        b0 = f1.launches()
        ok = informative(npf(tc.apa_selective_attention_sp(c.q, c.k, kq, c.v, c.scale, 0.01, bulk_bits=4, margins=pm)))
        checks.append(dict(id='LF3 positive control: the loaded pinned table launches', ok=ok and f1.launches() - b0 > 0,
                           launches=f1.launches() - b0))
        # capacity
        cap = 2**31
        for name, fn in (('quantize_sp2_keys_capacity', lambda: tc.quantize_sp2_keys(c.k, 4, capacity=cap)),
                         ('PackedSpKeys_capacity', lambda: tc.PackedSpKeys(c.k, capacity=cap)),
                         ('ApaSpq1Workspace_capacity', lambda: tc.ApaSpq1Workspace(c.k, cap, fmt='tq4'))):
            rec = expect_refusal(dict(id=f'LF3 capacity 2**31 refused before allocation: {name}'), fn, ValueError,
                                 ['capacity'])
            rec['ok'] = bool(rec['ok'] and rec['device_bytes_in_use_delta'] == 0)
            checks.append(rec)
        codes_f, sc_f = fake_capacity(tc, 2, 128, cap)
        rot, cbk, bnd = c.ws['tq4'].rotation, c.ws['tq4'].codebook, c.ws['tq4'].boundaries
        rotated, norms = tc._spo2.packed_prepare(c.k, rot, 0)
        checks.append(expect_refusal(dict(id='LF3 native packed_append with capacity 2**31 refused (names codes/capacity)'),
                                     lambda: tc._spo2.packed_append(c.k, codes_f, sc_f, rotated, norms, bnd, 0, True),
                                     ValueError, ['codes', 'does not fit']))
        checks.append(expect_refusal(dict(id='LF3 native packed_run (r2) with capacity 2**31 refused (names capacity)'),
                                     lambda: tc._C.apa_spq1_dump(c.q, c.k, c.v, (codes_f, sc_f, rot, cbk), c.scale, 2.0,
                                                                 fmt='tq4', gqa_path='r2'),
                                     ValueError, ['capacity']))
        tc._C.synchronize()
        ca2, cbv2, cl2 = controls()
        step(log, 'lf3_controls_after', launches=cl2)
        for label, x, y in (('packed_rotate', ca, ca2), ('public default route', cbv, cbv2)):
            b = bitwise(x, y)
            checks.append(dict(id=f'LF3 control after the refusals is bitwise equal: {label}', ok=b['equal'] and
                               b['informative'] == [True, True], **b))
        checks.append(dict(id='LF3 controls launched both times', ok=cl > 0 and cl2 > 0, before=cl, after=cl2))
    return checks, arm


def lf3_bnd_row(args):
    """Build F's behaviour on one refusal input (child process; L60 record, no gate)."""
    tc, arm = load_arm('F')
    os.environ['TC_APA_SP'] = '1'
    R = f1.rev1()
    pm = tc.PackedRegisteredMargins.load(PACKED_TABLE)
    row, rec = args.row, dict(row=args.row, arm='F', module_sha256=arm['module_sha256'])
    try:
        with tc.no_grad():
            c, cb, rows = lf3_inputs(tc, R)
            if row.startswith('eb3_timeline_G'):
                e = eb3_case(tc, R, int(row.rsplit('G', 1)[1]))
                fn = lambda: tc.apa_eb3(e.q, e.k, e.v, e.ws['tq4'].packed, e.scale, 2.0, timeline=True)
            elif row.startswith('margins_'):
                try:
                    mrows, _ = margin_rows(tc, c, pm)
                    fn = mrows[row]
                except AttributeError as e:   # Build F has no PACKED_PINNED_ENTRIES: rebuild the same objects by hand
                    from types import MappingProxyType
                    pins = {'4:64': 1.010646104812622, '4:128': 0.8716110587120056}
                    P = tc.apa_packed
                    kq = tc.quantize_sp2_keys(c.k, 4)

                    class Experimental(tc.PackedRegisteredMargins):
                        pass
                    replaced = tc.PackedRegisteredMargins.load(PACKED_TABLE)
                    object.__setattr__(replaced, 'entries', MappingProxyType({'4:64': 0.0, '4:128': 0.0}))
                    m = dict(margins_wrong_sha=tc.PackedRegisteredMargins('0' * 64, MappingProxyType(pins)),
                             margins_pinned_sha_zero_entries=tc.PackedRegisteredMargins(P.PACKED_TABLE_SHA256,
                                 MappingProxyType({'4:64': 0.0, '4:128': 0.0})),
                             margins_pinned_sha_extra_key=tc.PackedRegisteredMargins(P.PACKED_TABLE_SHA256,
                                 MappingProxyType(dict(pins, **{'2:64': 0.0}))),
                             margins_subclass=Experimental(P.PACKED_TABLE_SHA256, MappingProxyType(pins)),
                             margins_replaced_after_load=replaced).get(row)
                    if row == 'margins_load_override':
                        fn = lambda: tc.PackedRegisteredMargins.load(PACKED_TABLE, expected_sha256='0' * 64)
                    else:
                        fn = lambda: tc.apa_selective_attention_sp(c.q, c.k, kq, c.v, c.scale, 0.01, bulk_bits=4, margins=m)
            elif row == 'quantize_sp2_keys_capacity':
                fn = lambda: tc.quantize_sp2_keys(c.k, 4, capacity=2**31)
            elif row == 'PackedSpKeys_capacity':
                fn = lambda: tc.PackedSpKeys(c.k, capacity=2**31)
            elif row == 'ApaSpq1Workspace_capacity':
                fn = lambda: tc.ApaSpq1Workspace(c.k, 2**31, fmt='tq4')
            else:
                fn = rows[row]
            b0, m0 = f1.launches(), cuda_used()
            try:
                r = fn()
                rec['result'] = 'returned (silent acceptance)'
                outs = [x for x in (r if isinstance(r, (tuple, list)) else [r]) if hasattr(x, 'numpy')]
                if outs:
                    o = npf(outs[0]).astype(np.float32)
                    rec['output'] = dict(finite=bool(np.isfinite(o).all()), all_zero=bool(np.all(o == 0)))
            except BaseException as e:
                rec['result'] = type(e).__name__ + ': ' + str(e)[:300]
            rec['launches'] = None if b0 is None else f1.launches() - b0
            rec['device_bytes_in_use_delta'] = cuda_used() - m0
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
    precheck('F2')
    precheck('F')
    log = []
    checks, arm = lf3_f2(log)
    step(log, 'lf3_f2_measured', checks=len(checks))
    status = 'PASS' if checks and all(c['ok'] for c in checks) else 'FAIL'
    save(out / 'RESULT.json', dict(line='LF3', arm='F2', module_sha256=arm['module_sha256'], status=status,
                                   method_nothing_launched='LD_PRELOAD counter of cudaLaunchKernel (F1 SHIM_SOURCE)',
                                   checks=checks, failed=[c['id'] for c in checks if not c['ok']], steps=log,
                                   terminal='LF3_F2_DONE'))
    print('LF3 F2', status, [c['id'] for c in checks if not c['ok']], flush=True)
    # Build F boundary rows (L60), each in its own child.
    bnd, runs = [], []
    rows = (['rotate_q_B0', 'rotate_q_H0', 'prepare_k_B0', 'prepare_k_KV0', 'prepare_k_S0'] +
            [f'append_{f}_k_{l}' for f in ('tq4', 'int4') for l in ('B0', 'KV0')] +
            ['int4_stats_q_B0', 'int4_stats_q_H0', 'int4_stats_codes_KV0'] +
            [f'eb2b_{h}_v_{l}' for h in ('quantize', 'values') for l in ('B0', 'KV0', 'S0')] +
            [f'eb3_timeline_G{g}' for g in (3, 5, 6, 7, 8)] +
            ['margins_wrong_sha', 'margins_pinned_sha_zero_entries', 'margins_pinned_sha_extra_key', 'margins_subclass',
             'margins_replaced_after_load', 'margins_load_override',
             'quantize_sp2_keys_capacity', 'PackedSpKeys_capacity', 'ApaSpq1Workspace_capacity'])
    for row in rows:
        r = child([sys.executable, '-B', __file__, 'lf3-bnd-row', '--row', row], out / f'bnd_F_{row}.log', 240, child_env())
        runs.append(r)
        text = Path(r['log']).read_text()
        line = [l for l in text.splitlines() if l.startswith('BND_JSON ')]
        bnd.append(json.loads(line[-1][9:]) if line else dict(row=row, child_exit=r['exit'], note='no child result'))
        bnd[-1]['child_exit'] = r['exit']
    save(out / 'BND_F.json', dict(arm='F', module_sha256=F_SHA, rows=bnd, runs=runs, terminal='LF3_BND_DONE'))
    print('LF3 BND rows', len(bnd), flush=True)
    return 0 if status == 'PASS' else 1


# ------------------------------------------------------------------ LF4
def lf4_inputs(D):
    rng = np.random.default_rng(np.random.SeedSequence([20261116, 4, D]))
    scale_c = np.exp(0.5 * rng.standard_normal(D)).astype(np.float32)
    for c in (0, D // 4, D // 2, 3 * D // 4):
        scale_c[c] *= 6.0
    bias = (0.1 * rng.standard_normal(D)).astype(np.float32)
    k = (rng.standard_normal((1, 2, 257, D)).astype(np.float32) * scale_c + bias).astype(np.float32)
    v = rng.standard_normal((1, 2, 257, D)).astype(np.float32)
    qs = rng.standard_normal((65, 1, 8, 1, D)).astype(np.float32)
    return k, v, qs


def lf4_run(tc, log, D_list=(64, 128)):
    os.environ['TC_APA_SP'] = '1'
    os.environ.pop('TC_APA_SP_STORAGE', None)
    pm = tc.PackedRegisteredMargins.load(PACKED_TABLE)
    checks = []
    S0, N = 192, 256
    with tc.no_grad():
        for D in D_list:
            k, v, qs = lf4_inputs(D)
            scale = float(np.float32(1 / math.sqrt(D)))
            T = lambda x: tc.tensor(np.ascontiguousarray(x))
            k0 = T(k[:, :, :S0])
            obj = tc.quantize_sp2_keys(k0, 4, capacity=N)
            ok_obj = isinstance(obj, tc.PackedSpKeys) and obj.capacity == N
            checks.append(dict(id=f'LF4 D{D}: factory capacity forwarded', ok=ok_obj,
                               capacity=getattr(obj, 'capacity', None)))
            kt = k0
            for t in range(0, N - S0 + 1):
                S = S0 + t
                if t:
                    kt = T(k[:, :, :S])
                    try:
                        obj.append(kt)
                    except BaseException as e:
                        checks.append(dict(id=f'LF4 D{D} step {t}: append', ok=False, raised=repr(e)[:300]))
                        break
                vt, qt = T(v[:, :, :S]), T(qs[t])
                ref = tc.quantize_sp2_keys(kt, 4)
                rec = dict(id=f'LF4 D{D} step {t} (S={S}): codes, norms, output bitwise vs per-step rebuild')
                a_c, b_c = npf(obj.codes)[:, :, :S], npf(ref.codes)[:, :, :S]
                a_n, b_n = npf(obj.norms)[:, :, :S], npf(ref.norms)[:, :, :S]
                o1 = npf(tc.apa_selective_attention_sp(qt, kt, obj, vt, scale, 0.01, bulk_bits=4, margins=pm))
                o2 = npf(tc.apa_selective_attention_sp(qt, kt, ref, vt, scale, 0.01, bulk_bits=4, margins=pm))
                rec.update(codes=bitwise(a_c, b_c), norms=bitwise(a_n, b_n), output=bitwise(o1, o2))
                rec['ok'] = all(rec[x]['equal'] and rec[x]['informative'] == [True, True] for x in ('codes', 'norms', 'output'))
                checks.append(rec)
            step(log, f'lf4_D{D}_steps', steps=t)
            # full: one more append raises before any launch, object unchanged
            full = dict(id=f'LF4 D{D}: append past capacity N={N} raises before any launch, object unchanged')
            before = (npf(obj.codes).copy(), npf(obj.norms).copy(), obj.visible_rows, obj.generation)
            k_over = T(k[:, :, :N + 1])
            b0 = f1.launches()
            try:
                obj.append(k_over)
                full.update(raised=None, ok=False)
            except ValueError as e:
                full['raised'] = str(e)[:200]
                full['ok'] = 'capacity' in str(e)
            except BaseException as e:
                full.update(raised=repr(e)[:200], ok=False)
            full['launches'] = None if b0 is None else f1.launches() - b0
            after = (npf(obj.codes), npf(obj.norms), obj.visible_rows, obj.generation)
            full['codes_unchanged'] = bitwise(before[0], after[0])['equal']
            full['norms_unchanged'] = bitwise(before[1], after[1])['equal']
            full['visible_rows'] = [before[2], after[2]]
            full['generation'] = [before[3], after[3]]
            full['ok'] = bool(full['ok'] and full['launches'] == 0 and full['codes_unchanged'] and full['norms_unchanged']
                              and before[2] == after[2] == N)
            checks.append(full)
    return checks


def lf4(args):
    ensure_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    precheck('F2')
    log = []
    tc, arm = load_arm('F2')
    step(log, 'lf4_loaded', module_sha256=arm['module_sha256'], launch_counter=f1.launches() is not None)
    checks = lf4_run(tc, log)
    step(log, 'lf4_measured', checks=len(checks))
    status = 'PASS' if len(checks) == 2 * (1 + 65 + 1) and all(c['ok'] for c in checks) else 'FAIL'
    save(out / 'RESULT.json', dict(line='LF4', arm='F2', module_sha256=arm['module_sha256'], status=status,
                                   checks=checks, failed=[c['id'] for c in checks if not c['ok']], steps=log,
                                   terminal='LF4_DONE'))
    print('LF4', status, sum(c['ok'] for c in checks), '/', len(checks), [c['id'] for c in checks if not c['ok']][:10],
          flush=True)
    return 0 if status == 'PASS' else 1


# ------------------------------------------------------------------ LF6
def lf6(args):
    import apa_p1d_gpu as g
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in ('F', 'F2'):
        precheck(name)
    trees = {'F': TREE_F0, 'F2': ROOT}
    assert (TREE_F0 / 'tensor_cuda/tests').is_dir(), 'stage /tmp/apap1f2_treeF0 (git archive d01118b4) first'
    runs, log, results = [], [], []
    start = time.monotonic()
    for f in args.files:
        tag = Path(f).stem
        if time.monotonic() - start > args.max_seconds:
            results.append(dict(file=f, status='NOT_RUN_TIME_BUDGET'))
            continue
        recs = {}
        for armname in ('F', 'F2'):
            arm = json.loads((arm_dir(armname) / 'ARM.json').read_text())
            env = child_env(preload=False)
            env.pop('LD_PRELOAD', None)
            env.update(PYTHONPATH=str((arm_root(armname) / arm['package']).parent) + ':' +
                       str(trees[armname] / 'tensor_cuda/tests') + ':' + str(ROOT / 'artifacts/apa_p1'),
                       PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', TMPDIR='/tmp/apap1f2_l6')
            Path(env['TMPDIR']).mkdir(exist_ok=True)
            rec_path = out / f'{tag}_{armname}.json'
            cmd = [sys.executable, str(ROOT / 'artifacts/apa_p1/pytest_receipt.py'), str(rec_path), '-q', '--tb=short',
                   '-ra', '-p', 'no:cacheprovider', '--basetemp', f'/tmp/apap1f2_l6/bt_{tag}_{armname}', f]
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
        a, b = recs['F'], recs['F2']
        if a is None or b is None:
            results.append(dict(file=f, status='FAIL_NO_RECEIPT'))
            continue
        rows = g.outcome_rows(a, b)
        assert a['module']['sha256'] == F_SHA and b['module']['sha256'] == receipt('F2')['module_sha256']
        results.append(dict(file=f, ids=len(rows), collected_equal=a['collection'] == b['collection'],
                            collection_errors=[len(a['collection_errors']), len(b['collection_errors'])],
                            differences=sum(not r['equal'] for r in rows), outcomes_F=g._count(a),
                            outcomes_F2=g._count(b), rows=rows))
    complete = all('ids' in r for r in results)
    ok = complete and all(r.get('differences') == 0 and r.get('collected_equal') for r in results)
    status = 'PASS' if ok else ('FAIL' if complete or any(r.get('differences') for r in results) else 'PARTIAL')
    save(out / 'RESULT.json', dict(line='LF6', status=status, complete=complete, files=args.files, results=results,
                                   runs=runs, steps=log, terminal='LF6_DONE' if complete else 'LF6_PARTIAL'))
    print('LF6', status, [(r['file'], r.get('ids'), r.get('differences'), r.get('status')) for r in results], flush=True)
    return 0 if ok else 1


# ------------------------------------------------------------------ MUT
def mut_child(args):
    if f1.launches() is None and args.line in ('LF3', 'LF4'):
        ensure_counter()
    name = args.arm
    tc, arm = load_arm(name)
    R = f1.rev1()
    spec = {x['name']: x for x in json.loads((ART / 'mutants/MUTANTS.json').read_text())['mutants']}[name.split(':', 1)[1]]
    spec = spec['gpu']
    os.environ['TC_APA_SP'] = '1'
    log = []
    if spec['line'] == 'LF2':
        _, checks = lf2_run(tc, R, name, only=set(spec['cases']))
    elif spec['line'] == 'LF3':
        checks, _ = lf3_f2_subset(tc, R, spec)
    elif spec['line'] == 'LF4':
        checks = lf4_run(tc, log, D_list=(64,))
    else:
        checks = []
    print('MUT_JSON ' + json.dumps(dict(arm=name, module_sha256=arm['module_sha256'],
                                        checks=[dict(id=c['id'], ok=c['ok'], raised=c.get('raised'),
                                                     launches=c.get('launches')) for c in checks]), default=str),
          flush=True)
    return 0


def lf3_f2_subset(tc, R, spec):
    checks = []
    with tc.no_grad():
        c, cb, helper_rows = lf3_inputs(tc, R)
        for row in spec['rows']:
            if row.startswith('eb3_timeline_G'):
                G = int(row.rsplit('G', 1)[1])
                e = eb3_case(tc, R, G)
                checks.append(expect_refusal(dict(id=f'LF3 EB3 timeline=True G={G} raises before any launch'),
                                             lambda e=e: tc.apa_eb3(e.q, e.k, e.v, e.ws['tq4'].packed, e.scale, 2.0,
                                                                    timeline=True), ValueError, ['timeline']))
            else:
                checks.append(expect_refusal(dict(id=f'LF3 raw helper zero extent raises ValueError, no launch: {row}'),
                                             helper_rows[row], ValueError, ['positive']))
    return checks, None


def mut(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    m = json.loads((ART / 'mutants/MUTANTS.json').read_text())
    log, results = [], []
    names = args.rows or [x['name'] for x in m['mutants'] if x['runs_on'] == 'gpu']
    for full in [x for x in m['mutants'] if x['name'] in names]:
        spec = dict(full['gpu'], name=full['name'])
        arm = 'mutant:' + spec['name']
        precheck(arm)
        logf = out / f'mut_{spec["name"]}.log'
        r = child([sys.executable, '-B', __file__, 'mut-child', '--arm', arm, '--line', spec['line'], '--out', str(out)],
                  logf, 900, child_env())
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
    """Run each line's code up to its first device call on CPU (imports, table loads, argument parsing)."""
    rec = dict(line=args.target, steps=[])
    tc, arm = load_arm('F2')
    rec['steps'].append(f'load_arm F2 ok {arm["module_sha256"]}')
    pm = tc.PackedRegisteredMargins.load(PACKED_TABLE)
    rec['steps'].append('pinned table loads: ' + pm.sha256)
    assert sha(DENSE_TABLE) == DENSE_TABLE_SHA
    dm = tc.RegisteredMargins.load(DENSE_TABLE, expected_sha256=DENSE_TABLE_SHA)
    rec['steps'].append('dense table loads')
    R = f1.rev1()
    rec['steps'].append('rev1 repro imported')
    rec['lf2_cases'] = [c['id'] for c in lf2_cases()]
    k, v, qs = lf4_inputs(64)
    rec['lf4_inputs'] = [list(k.shape), list(v.shape), list(qs.shape), bool(np.isfinite(k).all())]
    import apa_p1d_gpu as g
    rec['p1d_gpu_imported'] = hasattr(g, 'l1_arm') and hasattr(g, 'outcome_rows') and hasattr(g, 'cuda_used')
    rec['f1_imported'] = all(hasattr(f1, n) for n in ('Case', 'run_raw', 'row_sets', 'reimport', 'lf1_arm', 'NAN_ENTRIES'))
    m = json.loads((ART / 'mutants/MUTANTS.json').read_text()) if (ART / 'mutants/MUTANTS.json').exists() else None
    rec['mutants'] = None if m is None else [x['name'] for x in m['mutants']]
    try:
        tc.tensor(np.zeros(1, np.float32))
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
    ap.add_argument('--rows', nargs='*')
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
    fn = {'lf1': lf1, 'lf1-arm': lf1_arm, 'lf2': lf2, 'lf3': lf3, 'lf3-bnd-row': lf3_bnd_row, 'lf4': lf4, 'lf6': lf6,
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
