#!/usr/bin/env python3
"""APA-P1-F1 GPU lines LF1-LF6 and MUT (run inside forge-gpu-run), plus their CPU steps.

Modes:
  stage <F|FH|mutant:<name>>   CPU: stage an importable arm (copies, never symlinks)
  shim                         CPU: compile the LD_PRELOAD cudaLaunchKernel counter (scratch)
  precheck <line>              CPU, before the lease: module sha vs receipt; for F/FH the
                               working tree's engine source map vs the build receipt
  lf1 | lf2 | lf3 | lf4 | lf5 | lf6 | mut     GPU lines (see artifacts/apa_p1f1/EXPECTED.json)
  (lf1-arm, lf3-row, lf4-child, lf5-fresh, mut-child: child processes)

Prior art: P1-D/AB-1 paired unsigned-view bit equality and pytest per-ID receipts
(Project-Tensor 2026; scripts/apa_p1d_gpu.py imported, not copied); REV1's repro
inputs and DLPack misalignment method (artifacts/apa_p1_rev1/repro.py, imported);
LD_PRELOAD symbol interposition (dynamic linker, glibc dlsym RTLD_NEXT) to count
kernel launches; IEEE 754 non-finite classification. No new algorithm here.
"""
import argparse
import ctypes
import hashlib
import importlib.util
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
ART = ROOT / 'artifacts/apa_p1f1'
P1D = ROOT / 'artifacts/apa_p1d'
BUILD_B_SHA = '8728e67e26027dbafe8a61592decb38e68171fa7b5634239ce0af7897c4e8c4e'
REV1_REPRO = Path('/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod/'
                  'artifacts/apa_p1_rev1/repro.py')
SHIM_DIR = Path('/tmp/apap1f1_shim')
SHIM = SHIM_DIR / 'liblaunchcount.so'
TREE_B = Path('/tmp/apap1f1_treeB')
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tensor_cuda/tests'))


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


def rev1():
    spec = importlib.util.spec_from_file_location('apa_p1_rev1_repro', REV1_REPRO)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


# ------------------------------------------------------------------ arms and receipts
def build_receipt(name):
    if name == 'B':
        return dict(module=str((P1D / 'build_B/lib/_tensor_cuda.cpython-312-x86_64-linux-gnu.so').relative_to(ROOT)),
                    module_sha256=BUILD_B_SHA)
    if name in ('F', 'FH'):
        recs = sorted((ART).glob(f'build_{name}*/BUILD.json'))
        recs = [json.loads(p.read_text()) for p in recs]
        recs = [r for r in recs if r.get('status') == 'PASS' and r['label'] == name]
        assert recs, f'no passing build receipt for {name}'
        return recs[-1]
    if name.startswith('mutant:'):
        return json.loads((ART / 'mutants' / name.split(':', 1)[1] / 'build/BUILD.json').read_text())
    raise ValueError(name)


def arm_dir(name):
    if name == 'B':
        return P1D / 'arms/B_v2'
    if name in ('F', 'FH'):
        return ART / 'arms' / name
    return ART / 'mutants' / name.split(':', 1)[1] / 'arm'


def engine_sources():
    from apa_p1f1_build import engine_sources as es
    return es(ROOT)


def precheck(name):
    """Refuse (raise) a module whose sha differs from its receipt, and for F/FH a drifted tree."""
    rec = build_receipt(name)
    assert rec.get('status', 'PASS') == 'PASS'
    module = ROOT / rec['module']
    got = sha(module)
    assert got == rec['module_sha256'], f'{name}: module sha {got} != receipt {rec["module_sha256"]}'
    if name == 'B':
        assert got == BUILD_B_SHA
    arm = json.loads((arm_dir(name) / 'ARM.json').read_text())
    assert arm['module_sha256'] == rec['module_sha256'], f'{name}: staged arm module differs from receipt'
    assert sha(ROOT / arm['module']) == arm['module_sha256']
    if name in ('F', 'FH'):
        now = engine_sources()
        want = rec.get('source_sha256_after', rec['source_sha256'])
        drift = sorted(k for k in set(now) | set(want) if now.get(k) != want.get(k))
        assert not drift, f'{name}: working tree differs from the build receipt: {drift}'
    return dict(arm=name, module=rec['module'], module_sha256=got)


def stage(name):
    rec = build_receipt(name)
    assert rec['status'] == 'PASS'
    dest = arm_dir(name) / 'tensor_cuda'
    dest.mkdir(parents=True)
    src = ROOT / 'tensor_cuda/tensor_cuda'
    files = {}
    for path in sorted(src.rglob('*.py')):
        if '__pycache__' in path.parts:
            continue
        out = dest / path.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, out)
        files[str(path.relative_to(ROOT))] = sha(out)
    module = ROOT / rec['module']
    assert sha(module) == rec['module_sha256']
    shutil.copyfile(module, dest / module.name)
    record = dict(arm=name, package=str(dest.relative_to(ROOT)), python_files=files,
                  module=str((dest / module.name).relative_to(ROOT)), module_sha256=sha(dest / module.name),
                  build_record=rec.get('label'))
    assert record['module_sha256'] == rec['module_sha256']
    (dest.parent / 'ARM.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


def load_arm(name):
    arm = json.loads((arm_dir(name) / 'ARM.json').read_text())
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


# ------------------------------------------------------------------ launch counter (LD_PRELOAD)
SHIM_SOURCE = r'''
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
typedef struct { unsigned x, y, z; } apa_dim3;
typedef int (*launch_fn)(const void*, apa_dim3, apa_dim3, void**, size_t, void*);
static unsigned long long launches = 0;
unsigned long long apa_p1f1_launches(void) { return __atomic_load_n(&launches, __ATOMIC_SEQ_CST); }
int cudaLaunchKernel(const void* f, apa_dim3 g, apa_dim3 b, void** a, size_t s, void* st) {
  static launch_fn real = 0;
  /* libcudart is a dependency of the RTLD_LOCAL extension module, so it is not
     in the global scope RTLD_NEXT searches: take the loaded library's handle. */
  if (!real) {
    void* h = dlopen("libcudart.so.12", RTLD_NOLOAD | RTLD_LAZY);
    if (h) real = (launch_fn)dlsym(h, "cudaLaunchKernel");
  }
  if (!real) real = (launch_fn)dlsym(RTLD_NEXT, "cudaLaunchKernel");
  __atomic_add_fetch(&launches, 1, __ATOMIC_SEQ_CST);
  return real(f, g, b, a, s, st);
}
'''


def build_shim():
    SHIM_DIR.mkdir(exist_ok=True)
    src = SHIM_DIR / 'launchcount.c'
    src.write_text(SHIM_SOURCE)
    subprocess.run(['timeout', '60s', 'gcc', '-O2', '-shared', '-fPIC', str(src), '-o', str(SHIM), '-ldl'], check=True)
    print('shim', SHIM, sha(SHIM))


def launches():
    fn = getattr(ctypes.CDLL(None), 'apa_p1f1_launches', None)
    if fn is None:
        return None
    fn.restype = ctypes.c_ulonglong
    return int(fn())


def child_env(extra=None):
    env = os.environ.copy()
    env['LD_PRELOAD'] = str(SHIM)
    env.update(PYTHONDONTWRITEBYTECODE='1', TMPDIR='/tmp/apap1f1_gpu')
    Path(env['TMPDIR']).mkdir(exist_ok=True)
    env.update(extra or {})
    return env


def child(argv, log, limit, env=None):
    t0 = time.monotonic()
    with Path(log).open('x') as f:
        try:
            code = subprocess.run(argv, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=limit,
                                  env=env or child_env()).returncode
        except subprocess.TimeoutExpired:
            code = 124
    return dict(argv=[str(a) for a in argv], exit=code, seconds=time.monotonic() - t0, log=str(log))


def npf(t):
    return np.asarray(t.numpy())


# ------------------------------------------------------------------ shared inputs
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
        self.vi = self.vs = None
        if dtype == 'bfloat16' and D == 128:
            self.vi, self.vs = tc.apa_eb2b_quantize(self.v)
        self.sinks = T(R.rnd_bf16(np.linspace(1.0, 3.0, H).astype(np.float32))) if sinks else None


def as_list(r):
    if isinstance(r, (tuple, list)):
        return [x for x in r]
    return [r]


def arrays_of(r):
    out = []
    for x in as_list(r):
        out.append(None if x is None else npf(x))
    return out


def run_raw(tc, c, name, *, delta, carry, sinks, diag, build, variant=0):
    """One raw packed-entry call; returns the list of returned arrays (None kept)."""
    if name in ('d1_ordered', 'd1_tensor', 'd1_baseline'):
        dots = 'tensor' if name == 'd1_tensor' else 'ordered'
        base = name == 'd1_baseline'
        return arrays_of(tc._C.apa_spo2_dump(c.q, c.k, c.k, c.v, c.scale, delta, False, sinks, diag, base, diag,
                                             carry=carry, dots=dots))
    if name[:3] in ('r2_', 'tc_'):
        path, fmt = name.split('_')
        return arrays_of(tc._C.apa_spq1_dump(c.q, c.k, c.v, c.ws[fmt].packed, c.scale, delta, False, sinks, diag,
                                             fmt=fmt, carry=carry, dump=diag, gqa_path=path))
    if name == 'eb2b':
        if build == 'B':
            assert delta == 8.1 and carry == 'row'
            r = tc.apa_eb2b(c.q, c.k, c.v, c.vi, c.vs, c.ws['tq4'].packed, c.scale, 1 / 32, diagnostics=diag)
        else:
            r = tc.apa_eb2b(c.q, c.k, c.v, c.vi, c.vs, c.ws['tq4'].packed, c.scale, 1 / 32, delta=delta,
                            carry=carry, diagnostics=diag)
        return arrays_of(r)
    if name == 'eb3':
        return arrays_of(tc.apa_eb3(c.q, c.k, c.v, c.ws['tq4'].packed, c.scale, delta, sinks=sinks, carry=carry,
                                    diagnostics=diag, variant=variant))
    if name == 'eb4':
        return arrays_of(tc.apa_eb4(c.q, c.k, c.v, c.vi, c.vs, c.ws['tq4'].packed, c.scale, delta, 1 / 32,
                                    sinks=sinks, carry=carry, diagnostics=diag))
    raise ValueError(name)


# ------------------------------------------------------------------ LF1
RULE_CONFIGS = [
    ('bf16_G4', 'bfloat16', 1, 8, 2, ['d1_ordered', 'd1_tensor', 'r2_tq4', 'r2_int4', 'tc_tq4', 'tc_int4', 'eb2b', 'eb3', 'eb4']),
    ('bf16_G1', 'bfloat16', 1, 2, 2, ['d1_ordered', 'r2_tq4', 'r2_int4']),
    ('f32_G4', 'float32', 1, 8, 2, ['d1_ordered', 'r2_tq4', 'r2_int4']),
    ('f32_G1', 'float32', 1, 2, 2, ['d1_ordered', 'r2_tq4']),
]


def lf1_arm(args):
    name = args.arm
    R = rev1()
    import apa_p1d_gpu as g
    # P1-D's L1 grid, its own code: point its arm loader at this build's staged arm.
    if name == 'B':
        g.ART, g.ARM_B = P1D, 'B_v2'
    else:
        g.ART, g.ARM_B = ART, name
    out = Path(args.out)
    l1dir = out / f'l1_{name}'
    l1dir.mkdir(parents=True)
    ns = argparse.Namespace(arm='B', out=str(l1dir))
    assert g.l1_arm(ns) == 0
    import tensor_cuda as tc
    assert sha(tc._C.__file__) == build_receipt(name)['module_sha256']
    os.environ['TC_APA_SP'] = '1'
    arrays = {}
    with tc.no_grad():
        for tag, dtype, B, H, KVH, entries in RULE_CONFIGS:
            c = Case(tc, R, 20261115, B, H, KVH, 4096, 128, dtype, plant=True, dup=True, sinks=True)
            arrays[f'{tag}/quant/tq4_codes'] = npf(c.ws['tq4'].codes)
            arrays[f'{tag}/quant/tq4_norms'] = npf(c.ws['tq4'].scales)
            arrays[f'{tag}/quant/int4_codes'] = npf(c.ws['int4'].codes)
            arrays[f'{tag}/quant/int4_scales'] = npf(c.ws['int4'].scales)
            if c.vi is not None:
                arrays[f'{tag}/quant/eb2b_vi'] = npf(c.vi)
                arrays[f'{tag}/quant/eb2b_vs'] = npf(c.vs)
            for entry in entries:
                if entry == 'eb2b':
                    for diag in (False, True):
                        for i, a in enumerate(run_raw(tc, c, 'eb2b', delta=8.1, carry='row', sinks=None, diag=diag, build=name)):
                            if a is not None:
                                arrays[f'{tag}/eb2b/d8.1/row/s0/diag{int(diag)}/{i}'] = a
                    continue
                for carry in ('partition', 'row'):
                    for delta in (2.0, 0.0):
                        for s in (False, True):
                            for diag in (False, True):
                                key = f'{tag}/{entry}/d{delta}/{carry}/s{int(s)}/diag{int(diag)}'
                                r = run_raw(tc, c, entry, delta=delta, carry=carry, sinks=c.sinks if s else None,
                                            diag=diag, build=name)
                                for i, a in enumerate(r):
                                    if a is not None:
                                        arrays[f'{key}/{i}'] = a
                if entry == 'eb3':
                    for variant in range(1, 9):
                        r = run_raw(tc, c, 'eb3', delta=2.0, carry='partition', sinks=None, diag=False, build=name,
                                    variant=variant)
                        arrays[f'{tag}/eb3_variant{variant}/out'] = r[0]
            if tag in ('bf16_G4', 'f32_G4'):
                for i, a in enumerate(run_raw(tc, c, 'd1_baseline', delta=2.0, carry='partition', sinks=None, diag=True,
                                              build=name)):
                    if a is not None:
                        arrays[f'{tag}/d1_baseline/{i}'] = a
        # REV1 eb3_S70000 (NC > 256 general merge path), variants 0 and 8.
        c = Case(tc, R, 7, 1, 2, 1, 70000, 128, 'bfloat16', plant=True)
        for variant in (0, 8):
            r = run_raw(tc, c, 'eb3', delta=2.0, carry='partition', sinks=None, diag=False, build=name, variant=variant)
            arrays[f'eb3_S70000/variant{variant}/out'] = r[0]
    path = out / f'raw_{name}.npz'
    with path.open('xb') as f:
        np.savez(f, **arrays)
    save(out / f'raw_{name}.json', dict(arm=name, arrays=len(arrays), npz_sha256=sha(path)))
    print('LF1 arm', name, 'raw arrays', len(arrays), flush=True)
    return 0


def lf1(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log, runs = [], []
    for name in ('B', 'F'):
        precheck(name)
        r = child([sys.executable, '-B', __file__, 'lf1-arm', '--arm', name, '--out', str(out)],
                  out / f'arm_{name}.log', 900, env=child_env({'LD_PRELOAD': ''}))
        runs.append(r)
        step(log, f'lf1_arm_{name}_exit', exit=r['exit'])
        if r['exit']:
            save(out / 'RESULT.json', dict(line='LF1', status='FAIL_CHILD', runs=runs, steps=log, terminal='LF1_CHILD_FAILED'))
            return 1
    rows = []
    A, Bz = np.load(out / 'l1_B/arm_B.npz'), np.load(out / 'l1_F/arm_B.npz')
    assert set(A.files) == set(Bz.files), 'L1 grid array sets differ'
    for key in sorted(A.files):
        rows.append(dict(part='L1_grid', key=key, **bitwise(A[key], Bz[key])))
    RA, RB = np.load(out / 'raw_B.npz'), np.load(out / 'raw_F.npz')
    missing = sorted(set(RA.files) ^ set(RB.files))
    for key in sorted(set(RA.files) & set(RB.files)):
        rows.append(dict(part='raw', key=key, **bitwise(RA[key], RB[key])))
    step(log, 'lf1_compared', rows=len(rows))
    eq = sum(r['equal'] for r in rows)
    inf_both = sum(r.get('informative') == [True, True] for r in rows)
    # Arrays that cannot be informative by construction are listed (e.g. an all-false
    # mask at delta 0 or a footprint row); the gate needs every comparison equal and
    # every output array informative on both sides.
    uninformative = [r['key'] for r in rows if r.get('informative') != [True, True]]
    out_uninformative = [k for k in uninformative if k.endswith('/out') or k.endswith('/0') or k.endswith('/plain')]
    status = 'PASS' if eq == len(rows) and not missing and not out_uninformative else 'FAIL'
    save(out / 'RESULT.json', dict(line='LF1', status=status, comparisons=len(rows), bitwise_equal=eq,
                                   both_sides_informative=inf_both, key_set_mismatch=missing,
                                   uninformative_arrays=uninformative, uninformative_outputs=out_uninformative,
                                   not_equal=[r for r in rows if not r['equal']][:50], runs=runs, steps=log,
                                   terminal='LF1_DONE'))
    print('LF1', status, eq, '/', len(rows), 'informative both', inf_both, 'uninformative outputs', len(out_uninformative),
          flush=True)
    return 0 if status == 'PASS' else 1


# ------------------------------------------------------------------ LF2
NAN_ENTRIES = {'bfloat16': ['d1_ordered', 'd1_tensor', 'r2_tq4', 'r2_int4', 'tc_tq4', 'tc_int4', 'eb2b', 'eb3', 'eb4'],
               'float32': ['d1_ordered', 'r2_tq4', 'r2_int4']}
NAN_CASES = [
    ('q_nan_rev1', 'q', (0, 0, 0, 5), np.nan), ('q_nan_more', 'q', (0, 5, 0, 77), np.nan),
    ('k_nan_rev1', 'k', (0, 0, 100, 5), np.nan), ('k_nan_more', 'k', (0, 1, 700, 64), np.nan),
    ('v_nan_a', 'v', (0, 0, 100, 5), np.nan), ('v_nan_more', 'v', (0, 1, 700, 64), np.nan),
    ('q_inf_rev1', 'q', (0, 0, 0, 5), np.inf), ('q_inf_more', 'q', (0, 5, 0, 77), np.inf),
    ('k_inf_rev1', 'k', (0, 0, 100, 5), np.inf), ('k_inf_more', 'k', (0, 1, 700, 64), np.inf),
]


def row_sets(o, H):
    o = o.astype(np.float32).reshape(H, -1)
    nonfinite = sorted(h for h in range(H) if not np.all(np.isfinite(o[h])))
    zero = sorted(h for h in range(H) if np.all(o[h] == 0))
    return nonfinite, zero


def supp_cases(R):
    """Supplementary +Inf-in-k case (added after LF2 r2: the registered k_inf_rev1 case left the dense side
    finite everywhere, L19). Rule, from the inputs only: key 100 of KV head 0 (REV1's key), first dimension d
    where heads 0..3 of q have both signs, so the dense side has finite and non-finite rows."""
    q, _, _ = R.make_inputs(11, 1, 8, 2, 1024, 128, plant=False)
    for d in range(128):
        s = np.sign(q[0, 0:4, 0, d])
        if (s > 0).any() and (s < 0).any():
            return [(f'k_inf_key100_dim{d}', 'k', (0, 0, 100, d), np.inf)]
    raise AssertionError('no dimension with both signs')


def lf2_run(tc, R, build, cases=None):
    H, KVH, S, D = 8, 2, 1024, 128
    res, checks = {}, []
    with tc.no_grad():
        for dtype, names in NAN_ENTRIES.items():
            for case, tensor, pos, value in (cases or NAN_CASES):
                def mutate(q, k, v, tensor=tensor, pos=pos, value=value):
                    dict(q=q, k=k, v=v)[tensor][pos] = value
                c = Case(tc, R, 11, 1, H, KVH, S, D, dtype, plant=False, mutate=mutate)
                dense = npf(tc._C.apa_selective_attention_sp(c.q, c.k, c.k, c.v, c.scale, 2.0, False, None, False))
                dn, dz = row_sets(dense, H)
                rec = res.setdefault(dtype, {}).setdefault(case, {})
                rec['dense'] = dict(nonfinite=dn, zero=dz)
                l19 = bool(dn) and len(dn) < H
                checks.append(dict(id=f'L19 dense informative {dtype} {case}', ok=l19))
                for name in names:
                    try:
                        o = run_raw(tc, c, name, delta=2.0, carry='partition', sinks=None, diag=False, build=build)[0]
                        en, ez = row_sets(o, H)
                        rec[name] = dict(nonfinite=en, zero=ez)
                    except Exception as e:  # recorded as a failed check, never a pass
                        rec[name] = dict(ERROR=repr(e))
                        en, ez = None, None
                    nan_case = '_nan' in case
                    if nan_case:
                        checks.append(dict(id=f'G2 NaN row-set equality: {name} {dtype} {case}', ok=en == dn,
                                           entry=en, dense=dn))
                    else:
                        checks.append(dict(id=f'G2b Inf row-set equality: {name} {dtype} {case}', ok=en == dn,
                                           entry=en, dense=dn, prediction=True))
                    zero_bad = sorted(set(ez or []) & set(dn)) if ez is not None else None
                    checks.append(dict(id=f'G2 no all-zero row where dense is non-finite: {name} {dtype} {case}',
                                       ok=zero_bad == [], rows=zero_bad))
    return res, checks


def lf2(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = []
    name = args.arm or 'F'
    precheck(name)
    tc, arm = load_arm(name)
    os.environ['TC_APA_SP'] = '1'
    R = rev1()
    step(log, 'lf2_loaded', arm=name, module_sha256=arm['module_sha256'])
    supp = getattr(args, 'supp', False)
    res, checks = lf2_run(tc, R, name, supp_cases(R) if supp else None)
    if supp:
        name = name + '_supp'
    step(log, 'lf2_measured', checks=len(checks))
    g2 = [c for c in checks if not c.get('prediction')]
    g2b = [c for c in checks if c.get('prediction')]
    status = 'PASS' if all(c['ok'] for c in g2) else 'FAIL'
    held = all(c['ok'] for c in g2b)
    save(out / f'RESULT{"" if name == "F" else "_" + name.replace(":", "_")}.json',
         dict(line='LF2', arm=name, module_sha256=arm['module_sha256'], status=status,
              G2b='HELD' if held else 'NOT HELD', failed=[c for c in g2 if not c['ok']],
              g2b_misses=[c for c in g2b if not c['ok']], checks=checks, results=res, steps=log, terminal='LF2_DONE'))
    print('LF2', name, status, 'G2b', 'HELD' if held else 'NOT HELD',
          'failed', [c['id'] for c in g2 if not c['ok']][:20], flush=True)
    return 0 if status == 'PASS' else 1


# ------------------------------------------------------------------ LF3
def misaligned(tc, R, arr, dtype):
    """REV1's DLPack re-import: data pointer = base + one element (2 B bf16/uint8 pairs, 4 B fp32)."""
    if dtype == 'uint8':
        # Two-byte offset for byte operands: REV1's method with a uint8 view of two elements.
        flat = np.zeros(arr.size + 32, np.uint8)
        flat[2:2 + arr.size] = arr.ravel()
        t = tc.tensor(flat, dtype='uint8')
        return reimport(tc, t, arr.shape, 2)
    return R.misaligned(tc, arr, dtype)


def reimport(tc, t, shape, byte_offset):
    cap = t.__dlpack__()
    get = ctypes.pythonapi.PyCapsule_GetPointer
    get.restype = ctypes.c_void_p
    get.argtypes = [ctypes.py_object, ctypes.c_char_p]
    p = get(cap, b'dltensor')
    nd = len(shape)
    shp = (ctypes.c_int64 * nd)(*shape)
    st = [1] * nd
    for i in range(nd - 2, -1, -1):
        st[i] = st[i + 1] * shape[i + 1]
    strides = (ctypes.c_int64 * nd)(*st)
    ctypes.c_int32.from_address(p + 16).value = nd
    ctypes.c_void_p.from_address(p + 24).value = ctypes.addressof(shp)
    ctypes.c_void_p.from_address(p + 32).value = ctypes.addressof(strides)
    ctypes.c_uint64.from_address(p + 40).value = byte_offset
    out = tc.from_dlpack(cap)
    reimport._keep = getattr(reimport, '_keep', []) + [t, shp, strides, cap]
    return out


def lf3_call(tc, c, row, ops):
    """Call the census row's entry with operands from ops (dict)."""
    q, k, v = ops.get('q', c.q), ops.get('k', c.k), ops.get('v', c.v)
    pk = c.ws['tq4'].packed if not row.startswith('spq4') else c.ws['int4'].packed
    codes = ops.get('codes', pk[0])
    rot = ops.get('rotation', pk[2])
    packed = (codes, pk[1], rot, pk[3])
    vi = ops.get('vi', c.vi)
    if row.startswith('d1'):
        return tc._C.apa_spo2_dump(q, k, ops.get('kq', c.k), v, c.scale, 2.0, carry='partition', dots='ordered')
    if row.startswith('r2_G1'):
        return tc._C.apa_spq1_dump(q, k, v, packed, c.scale, 2.0, fmt='tq4', carry='partition', gqa_path='r2')
    if row.startswith('spq4'):
        return tc._C.apa_spq1_dump(q, k, v, packed, c.scale, 2.0, fmt='int4', carry='partition', gqa_path='tc')
    if row.startswith('spq6'):
        return tc._C.apa_spq1_dump(q, k, v, packed, c.scale, 2.0, fmt='tq4', carry='partition', gqa_path='tc')
    if row.startswith('eb2b_values'):
        flags = tc.tensor(np.ones((1, c.KVH, (c.S + 63) // 64), np.uint8), dtype='uint8')
        return tc._spo2.eb2b_values(v, vi, c.vs, flags if row.endswith('_v') else
                                    tc.tensor(np.zeros((1, c.KVH, (c.S + 63) // 64), np.uint8), dtype='uint8'))
    if row.startswith('eb2b'):
        if LF3_ARM[0] == 'B':   # Build B (boundary record only): no delta/carry arguments
            return tc.apa_eb2b(q, k, v, vi, c.vs, packed, c.scale, 1 / 32, force_bf16=row.endswith('_v'))
        return tc.apa_eb2b(q, k, v, vi, c.vs, packed, c.scale, 1 / 32, delta=2.0, carry='partition',
                           force_bf16=row.endswith('_v'))
    if row.startswith('eb3'):
        return tc.apa_eb3(q, k, v, packed, c.scale, 2.0, carry='partition')
    if row.startswith('eb4'):
        return tc.apa_eb4(q, k, v, vi, c.vs, packed, c.scale, 2.0, 1 / 32, carry='partition',
                          force_bf16=row.endswith('_v'))
    if row.startswith('dense'):
        return tc._C.apa_selective_attention_sp(q, k, k, v, c.scale, 2.0)
    raise ValueError(row)


ACCEPT_ROWS = ['dense_q', 'dense_k', 'dense_v', 'dense_k_G1', 'r2public_q', 'r2public_v']
LF3_ARM = ['F']


def lf3_row(args):
    LF3_ARM[0] = args.arm
    tc, arm = load_arm(args.arm)
    os.environ['TC_APA_SP'] = '1'
    R = rev1()
    row = args.row
    r = dict(row=row, arm=args.arm, module_sha256=arm['module_sha256'], launch_counter=launches() is not None)
    try:
        census = {x['id']: x for x in json.loads((ART / 'ALIGNMENT_CENSUS.json').read_text())['rows']}
        G1 = row.startswith('d1') or row.startswith('r2_G1') or row == 'dense_k_G1'
        dtype = 'float32' if row in ('d1_k_fp32',) or row.startswith('r2public') else 'bfloat16'
        H, KVH = (2, 2) if G1 else (8, 2)
        c = Case(tc, R, 3, 1, H, KVH, 1024, 128, dtype, plant=False)
        if row.startswith('r2public'):
            m = tc.PackedRegisteredMargins.load(P1D / 'packed_e_q_table.json',
                                                expected_sha256=tc.apa_packed.PACKED_TABLE_SHA256)
            kq = tc.quantize_sp2_keys(c.k, 4)
            assert isinstance(kq, tc.PackedSpKeys)
            opname = row.split('_', 1)[1]
            fn = lambda ops: tc.apa_selective_attention_sp(ops.get('q', c.q), c.k, kq, ops.get('v', c.v), c.scale,
                                                           0.01, bulk_bits=4, margins=m)
            arr = dict(q=c.qn, v=c.vn)[opname]
        else:
            opname = row.split('_')[-1] if not row.startswith('dense') else row.split('_')[1]
            if row == 'd1_k_fp32':
                opname = 'k'
            fn = lambda ops: lf3_call(tc, c, row, ops)
            pk = c.ws['int4' if row.startswith('spq4') else 'tq4']
            arr = {'q': c.qn, 'k': c.kn, 'kq': c.kn, 'v': c.vn,
                   'codes': None if opname != 'codes' else npf(pk.codes),
                   'rotation': None if opname != 'rotation' else npf(pk.rotation),
                   'vi': None if opname != 'vi' else npf(c.vi)}[opname]
        odt = {'codes': 'uint8', 'vi': 'uint8', 'rotation': 'float32'}.get(opname, dtype)
        r['operand'], r['operand_dtype'] = opname, odt
        a0 = launches()
        out_a = arrays_of(fn({}))
        r['aligned_launches'] = None if a0 is None else launches() - a0
        r['aligned_informative'] = informative(out_a[0])
        mis = misaligned(tc, R, np.ascontiguousarray(arr), odt)
        r['ptr_mod16'] = int(mis.data_ptr() % 16)
        r['byte_offset'] = 2 if odt in ('uint8', 'bfloat16') else 4
        if row in ACCEPT_ROWS or row.startswith('r2public'):
            out_m = arrays_of(fn({opname: mis} if opname != 'k' or not row.startswith('dense') else {'k': mis}))
            r['misaligned_bitwise_equal'] = bool(np.array_equal(uview(out_a[0]), uview(out_m[0])))
            r['ok'] = r['misaligned_bitwise_equal'] and r['aligned_informative']
        else:
            spec = census[row]
            b0 = launches()
            try:
                fn({opname: mis})
                r['refusal'] = 'DID NOT RAISE'
                r['refused_valueerror'] = False
            except ValueError as e:
                msg = str(e)
                r['refusal'] = 'ValueError: ' + msg[:300]
                r['refused_valueerror'] = True
                r['message_names_operand'] = f"'{opname}'" in msg
                r['message_names_alignment'] = f"{spec['width']}-byte aligned" in msg
                r['message_names_entry'] = msg.split(':')[0].startswith('apa_')
            except BaseException as e:
                r['refusal'] = type(e).__name__ + ': ' + str(e)[:300]
                r['refused_valueerror'] = False
            r['launches_during_refusal'] = None if b0 is None else launches() - b0
            out_b = arrays_of(fn({}))
            r['rerun_bitwise_equal'] = bool(np.array_equal(uview(out_a[0]), uview(out_b[0])))
            tc._C.synchronize()
            r['ok'] = bool(r['refused_valueerror'] and r.get('message_names_operand') and
                           r.get('message_names_alignment') and r.get('message_names_entry') and
                           r['launches_during_refusal'] == 0 and r['rerun_bitwise_equal'] and
                           r['aligned_informative'] and (r['aligned_launches'] or 0) > 0)
            r['assertion'] = 'LF3: misaligned call raises ValueError before any launch; aligned rerun bitwise equal'
    except BaseException as e:
        r['ok'] = False
        r['exception'] = type(e).__name__ + ': ' + str(e)[:400]
        r['trace'] = traceback.format_exc()[-1500:]
        try:
            tc._C.synchronize()
            tc.tensor(np.arange(4, dtype=np.float32)).numpy()
            r['context_after_exception'] = 'usable'
        except BaseException as e2:
            r['context_after_exception'] = 'BROKEN: ' + str(e2)[:200]
    print('CHILD_JSON ' + json.dumps(r, default=str), flush=True)
    return 0


def run_rows(out, arm, rows, tag):
    results, runs = [], []
    for row in rows:
        log = out / f'{tag}_{row}.log'
        r = child([sys.executable, '-B', __file__, 'lf3-row', '--arm', arm, '--row', row], log, 240)
        runs.append(r)
        text = log.read_text()
        line = [l for l in text.splitlines() if l.startswith('CHILD_JSON ')]
        res = json.loads(line[-1][11:]) if line else dict(row=row, ok=False, exception='no child result',
                                                          child_exit=r['exit'])
        res['child_exit'] = r['exit']
        results.append(res)
        print('LF3', row, 'ok' if res.get('ok') else 'FAIL', res.get('refusal', res.get('misaligned_bitwise_equal')),
              flush=True)
    return results, runs


def lf3(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arm = args.arm or 'F'
    precheck(arm)
    log = []
    census = [x['id'] for x in json.loads((ART / 'ALIGNMENT_CENSUS.json').read_text())['rows']]
    rows = args.rows or (census + ACCEPT_ROWS)
    results, runs = run_rows(out, arm, rows, 'row')
    step(log, 'lf3_rows_done', rows=len(results))
    status = 'PASS' if results and all(r.get('ok') for r in results) else 'FAIL'
    save(out / f'RESULT{"" if arm == "F" else "_" + arm.replace(":", "_")}.json',
         dict(line='LF3', arm=arm, status=status, rows=results, failed=[r['row'] for r in results if not r.get('ok')],
              runs=runs, steps=log, terminal='LF3_DONE'))
    print('LF3', arm, status, [r['row'] for r in results if not r.get('ok')], flush=True)
    return 0 if status == 'PASS' else 1


# ------------------------------------------------------------------ LF4
def expect(checks, cid, fn, exc, match=None, launches_zero=True):
    b0 = launches()
    rec = dict(id=cid)
    try:
        fn()
        rec.update(raised=None, ok=False, note='DID NOT RAISE')
    except BaseException as e:
        rec['raised'] = type(e).__name__ + ': ' + str(e)[:300]
        ok = isinstance(e, exc) and (match is None or match in str(e))
        rec['ok'] = ok
    rec['launches'] = None if b0 is None else launches() - b0
    if launches_zero:
        rec['ok'] = bool(rec['ok'] and rec['launches'] == 0)
    checks.append(rec)
    return rec


def lf4_child(args):
    """REV1's int4_stats KVH=0 case, alone in this process."""
    tc, arm = load_arm(args.arm)
    os.environ['TC_APA_SP'] = '1'
    R = rev1()
    c = Case(tc, R, 3, 1, 8, 2, 1024, 128, 'bfloat16', plant=False)
    codes0 = tc.empty((1, 0, c.S, 64), dtype='uint8')
    sc0 = tc.empty((1, 0, c.S), dtype='float32')
    try:
        tc._spo2.int4_stats(c.q, codes0, sc0, c.S, c.scale, 1.0)
        print('CHILD_RESULT returned without error', flush=True)
        return 3
    except ValueError as e:
        print('CHILD_RESULT ValueError: ' + str(e), flush=True)
        return 0


def _no_core():
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def lf4_run(tc, R, arm, out, checks):
    os.environ['TC_APA_SP'] = '1'
    # 1. REV1 int4_stats case in a child (a SIGFPE would kill only the child).
    log = out / f'int4_stats_child_{arm.replace(":", "_")}.log'
    t0 = time.monotonic()
    with log.open('x') as f:
        try:
            code = subprocess.run([sys.executable, '-B', __file__, 'lf4-child', '--arm', arm], cwd=ROOT, stdout=f,
                                  stderr=subprocess.STDOUT, timeout=120, env=child_env(), preexec_fn=_no_core).returncode
        except subprocess.TimeoutExpired:
            code = 124
    text = log.read_text()
    checks.append(dict(id='LF4 int4_stats KVH=0 raises ValueError and the child exits 0', exit=code,
                       ok=code == 0 and 'CHILD_RESULT ValueError' in text, seconds=time.monotonic() - t0,
                       result=[l for l in text.splitlines() if l.startswith('CHILD_RESULT')]))
    c = Case(tc, R, 20261115, 1, 8, 2, 4096, 128, 'bfloat16', plant=True, dup=True)
    c1 = Case(tc, R, 4, 1, 2, 2, 1024, 128, 'bfloat16', plant=False)
    entries = {
        'd1': lambda cc, **kw: tc._C.apa_spo2_dump(cc.q, cc.k, cc.k, cc.v, cc.scale, 2.0, carry=kw.get('carry', 'partition'), dots=kw.get('dots', 'ordered')),
        'r2': lambda cc, **kw: tc._C.apa_spq1_dump(cc.q, cc.k, cc.v, cc.ws['tq4'].packed, cc.scale, 2.0, fmt='tq4', carry=kw.get('carry', 'partition'), gqa_path='r2'),
        'tc_int4': lambda cc, **kw: tc._C.apa_spq1_dump(cc.q, cc.k, cc.v, cc.ws['int4'].packed, cc.scale, 2.0, fmt='int4', carry=kw.get('carry', 'partition'), gqa_path='tc'),
        'tc_tq4': lambda cc, **kw: tc._C.apa_spq1_dump(cc.q, cc.k, cc.v, cc.ws['tq4'].packed, cc.scale, 2.0, fmt='tq4', carry=kw.get('carry', 'partition'), gqa_path='tc'),
        'eb2b': lambda cc, **kw: tc.apa_eb2b(cc.q, cc.k, cc.v, cc.vi, cc.vs, cc.ws['tq4'].packed, cc.scale, 1 / 32, delta=8.1, carry=kw.get('carry', 'row')),
        'eb3': lambda cc, **kw: tc.apa_eb3(cc.q, cc.k, cc.v, cc.ws['tq4'].packed, cc.scale, 2.0, carry=kw.get('carry', 'partition')),
        'eb4': lambda cc, **kw: tc.apa_eb4(cc.q, cc.k, cc.v, cc.vi, cc.vs, cc.ws['tq4'].packed, cc.scale, 2.0, 1 / 32, carry=kw.get('carry', 'partition')),
    }
    # positive control: an accepted call counts launches
    b0 = launches()
    entries['eb3'](c)
    checks.append(dict(id='LF4 launch counter positive control (accepted EB3 call counts > 0)',
                       launches=None if b0 is None else launches() - b0,
                       ok=b0 is not None and launches() - b0 > 0))
    # 2. empty extents
    def empty_case(B, H, KVH, S):
        E = lambda shape, dt='bfloat16': tc.empty(shape, dtype=dt)
        cc = type('E', (), {})()
        cc.q, cc.k, cc.v = E((B, H, 1, 128)), E((B, KVH, S, 128)), E((B, KVH, S, 128))
        cc.scale = c.scale
        cc.vi, cc.vs = E((B, KVH, S, 128), 'uint8'), E((B, KVH, S), 'float32')
        pk = c.ws['tq4'].packed
        codes = E((B, KVH, max(S, 1), 64), 'uint8')
        scl = E((B, KVH, max(S, 1)), 'float32')
        cc.ws = {'tq4': type('W', (), {'packed': (codes, scl, pk[2], pk[3])})(),
                 'int4': type('W', (), {'packed': (codes, scl, pk[2], pk[3])})()}
        return cc
    for label, dims in (('B=0', (0, 8, 2, 256)), ('H=0', (1, 0, 2, 256)), ('S=0', (1, 8, 2, 0)), ('KV=0', (1, 8, 0, 256))):
        try:
            cc = empty_case(*dims)
        except Exception as e:
            checks.append(dict(id=f'LF4 empty extents {label}: inputs', ok=False, note=repr(e)))
            continue
        for name, fn in entries.items():
            expect(checks, f'LF4 empty extents {label}: {name} raises', lambda fn=fn, cc=cc: fn(cc),
                   (RuntimeError, ValueError), None)
    # 3. TC_APA_SP unset
    del os.environ['TC_APA_SP']
    try:
        for name, fn in entries.items():
            expect(checks, f'LF4 TC_APA_SP unset: {name} raises before any launch', lambda fn=fn: fn(c), RuntimeError,
                   'TC_APA_SP')
    finally:
        os.environ['TC_APA_SP'] = '1'
    # 4. EB2b delta/carry change the selection (bf16_G4 REV1 rule inputs, S=4096)
    def eb2b_mask(delta, carry):
        r = tc.apa_eb2b(c.q, c.k, c.v, c.vi, c.vs, c.ws['tq4'].packed, c.scale, 1 / 32, delta=delta, carry=carry,
                        diagnostics=True)
        return npf(r[1]).astype(bool)
    try:
        m81, m20, mpart = eb2b_mask(8.1, 'row'), eb2b_mask(2.0, 'row'), eb2b_mask(8.1, 'partition')
        checks.append(dict(id='LF4 EB2b delta changes the selected count (delta 2.0 vs 8.1, carry row, bf16_G4 S=4096)',
                           selected_8_1=int(m81.sum()), selected_2_0=int(m20.sum()), ok=int(m81.sum()) != int(m20.sum())))
        checks.append(dict(id='LF4 EB2b carry changes the selection (partition vs row, delta 8.1)',
                           selected_row=int(m81.sum()), selected_partition=int(mpart.sum()),
                           ok=not np.array_equal(m81, mpart)))
    except BaseException as e:   # recorded as failed checks (Build B has no delta/carry), never a pass
        checks.append(dict(id='LF4 EB2b delta changes the selected count (delta 2.0 vs 8.1, carry row, bf16_G4 S=4096)',
                           ok=False, raised=repr(e)[:300]))
    expect(checks, 'LF4 EB2b without delta raises TypeError',
           lambda: tc.apa_eb2b(c.q, c.k, c.v, c.vi, c.vs, c.ws['tq4'].packed, c.scale, 1 / 32, carry='row'), TypeError)
    expect(checks, 'LF4 EB2b without carry raises TypeError',
           lambda: tc.apa_eb2b(c.q, c.k, c.v, c.vi, c.vs, c.ws['tq4'].packed, c.scale, 1 / 32, delta=8.1), TypeError)
    expect(checks, 'LF4 EB2b negative delta raises', lambda: tc.apa_eb2b(c.q, c.k, c.v, c.vi, c.vs, c.ws['tq4'].packed,
           c.scale, 1 / 32, delta=-1.0, carry='row'), RuntimeError, 'delta')
    # 5. R7 knobs
    base = {}
    for n in ('tc_tq4', 'r2', 'eb3', 'eb4'):
        try:
            base[n] = npf(as_list(entries[n](c))[0])
        except BaseException as e:
            checks.append(dict(id=f'LF4 knob baseline call {n}', ok=False, raised=repr(e)[:300]))
            base[n] = np.zeros(1, np.float32)
    knob_cases = [('APA_FXTC1_VARIANT', v, 'tc_tq4') for v in ('0', '1', '3', 'x')] + \
                 [('TC_APA_SPQ1_REFERENCE', '1', 'r2'), ('APA_P1_EARLY_COUNTER', '1', 'eb3'),
                  ('APA_P1_EARLY_COUNTER', '1', 'eb4')]
    for var, value, name in knob_cases:
        os.environ[var] = value
        try:
            expect(checks, f'LF4 knob {var}={value} on {name} raises naming it', lambda name=name: entries[name](c),
                   ValueError, var)
        finally:
            del os.environ[var]
    for var, value, name in (('APA_FXTC1_VARIANT', '2', 'tc_tq4'), ('TC_APA_SPQ1_REFERENCE', '0', 'r2'),
                             ('APA_P1_EARLY_COUNTER', '0', 'eb3'), ('APA_P1_EARLY_COUNTER', '0', 'eb4')):
        os.environ[var] = value
        try:
            got = npf(as_list(entries[name](c))[0])
            checks.append(dict(id=f'LF4 knob {var}={value} (default) accepted on {name}, bitwise equal to unset',
                               ok=bool(np.array_equal(uview(got), uview(base[name]))) and informative(got)))
        except BaseException as e:
            checks.append(dict(id=f'LF4 knob {var}={value} (default) accepted on {name}', ok=False, raised=repr(e)))
        finally:
            del os.environ[var]
    # 6. dots and carry
    cf = Case(tc, R, 4, 1, 8, 2, 1024, 128, 'float32', plant=False)
    expect(checks, 'LF4 dots=tensor on float32 raises', lambda: entries['d1'](cf, dots='tensor'), RuntimeError, 'dots')
    expect(checks, 'LF4 dots=bogus raises', lambda: entries['d1'](c, dots='bogus'), RuntimeError, 'dots')
    for name in entries:
        expect(checks, f'LF4 carry=bogus on {name} raises', lambda name=name: entries[name](c, carry='bogus'),
               RuntimeError, 'carry')
    # also on the G=1 path of d1/r2
    expect(checks, 'LF4 carry=bogus on d1 G1 raises', lambda: entries['d1'](c1, carry='bogus'), RuntimeError, 'carry')
    return checks


def lf4(args):
    if launches() is None:
        # LF4 repair (r1 ran without the launch counter): re-exec this same process
        # image with the LD_PRELOAD counter, so every 'nothing launched' check counts.
        assert os.environ.get('LD_PRELOAD') != str(SHIM), 'counter preloaded but not visible'
        os.execve(sys.executable, [sys.executable, '-B', str(Path(__file__).resolve()), *sys.argv[1:]], child_env())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arm = args.arm or 'F'
    precheck(arm)
    log = []
    tc, a = load_arm(arm)
    R = rev1()
    step(log, 'lf4_loaded', arm=arm, module_sha256=a['module_sha256'], launch_counter=launches() is not None)
    checks = lf4_run(tc, R, arm, out, [])
    step(log, 'lf4_measured', checks=len(checks))
    status = 'PASS' if checks and all(c['ok'] for c in checks) else 'FAIL'
    save(out / f'RESULT{"" if arm == "F" else "_" + arm.replace(":", "_")}.json',
         dict(line='LF4', arm=arm, status=status, method_nothing_launched='LD_PRELOAD counter of cudaLaunchKernel '
              '(scripts/apa_p1f1_gpu.py SHIM_SOURCE); positive control included', checks=checks,
              failed=[c['id'] for c in checks if not c['ok']], steps=log, terminal='LF4_DONE'))
    print('LF4', arm, status, [c['id'] for c in checks if not c['ok']], flush=True)
    return 0 if status == 'PASS' else 1


# ------------------------------------------------------------------ LF5 (hooks build)
def lf5_inputs(tc):
    rng = np.random.default_rng(np.random.SeedSequence([20261115, 5]))
    R = rev1()
    q = R.rnd_bf16(rng.standard_normal((1, 2, 1, 128)))
    k = R.rnd_bf16(0.5 * rng.standard_normal((1, 1, 257, 128)))
    v = R.rnd_bf16(rng.standard_normal((1, 1, 257, 128)))
    T = lambda x: tc.tensor(np.ascontiguousarray(x), dtype='bfloat16')
    qt, kt, vt = T(q), T(k), T(v)
    w = tc.ApaSpq1Workspace(kt, 257, fmt='tq4')
    vi, vs = tc.apa_eb2b_quantize(vt)
    return qt, kt, vt, w, vi, vs, float(np.float32(1 / np.sqrt(128)))


def lf5_call(tc, family, inputs):
    qt, kt, vt, w, vi, vs, scale = inputs
    if family == 'eb3':
        return npf(tc.apa_eb3(qt, kt, vt, w.packed, scale, 8.1, carry='row', poison=True))
    return npf(tc.apa_eb4(qt, kt, vt, vi, vs, w.packed, scale, 8.1, 1 / 32, carry='row', poison=True))


def lf5_fresh(args):
    tc, arm = load_arm(args.arm)
    os.environ['TC_APA_SP'] = '1'
    os.environ.pop('APA_P1_EARLY_COUNTER', None)
    out = lf5_call(tc, args.family, lf5_inputs(tc))
    np.save(Path(args.out) / f'fresh_{args.family}_{args.arm.replace(":", "_")}.npy', out)
    print('FRESH', args.family, informative(out), flush=True)
    return 0


def lf5_run(tc, arm, out, checks):
    os.environ['TC_APA_SP'] = '1'
    native = tc._spo2
    inputs = lf5_inputs(tc)
    qt, kt, vt, w, vi, vs, scale = inputs
    for family in ('eb3', 'eb4'):
        os.environ['APA_P1_EARLY_COUNTER'] = 'timeout'
        try:
            native._p1_test_run(family == 'eb4', qt, kt, vt, vi, vs, w.packed, scale, 8.1, 1 / 32, True, 0)
            checks.append(dict(id=f'LF5 {family}: forced test-hook timeout raised', ok=False, note='DID NOT RAISE'))
        except RuntimeError as e:
            checks.append(dict(id=f'LF5 {family}: forced test-hook timeout raised', ok='timed out' in str(e),
                               raised=str(e)[:200]))
        finally:
            os.environ.pop('APA_P1_EARLY_COUNTER', None)
        slots = list(native._p1_test_merge_tickets(1))
        checks.append(dict(id=f'LF5 {family}: ticket slot reads zero after the forced timeout', slots=slots,
                           ok=slots == [0]))
        got = lf5_call(tc, family, inputs)
        r = child([sys.executable, '-B', __file__, 'lf5-fresh', '--arm', arm, '--family', family, '--out', str(out)],
                  out / f'fresh_{family}_{arm.replace(":", "_")}.log', 240)
        fresh = np.load(out / f'fresh_{family}_{arm.replace(":", "_")}.npy') if r['exit'] == 0 else None
        checks.append(dict(id=f'LF5 {family}: next fused merge on the slot equals a fresh process, bitwise',
                           child_exit=r['exit'], ok=fresh is not None and bitwise(got, fresh)['equal'] and
                           informative(got) and informative(fresh), **({} if fresh is None else bitwise(got, fresh))))
    # knobs as in S3 (hooks build)
    R = rev1()
    c = Case(tc, R, 20261115, 1, 8, 2, 4096, 128, 'bfloat16', plant=True)
    tq = lambda: npf(tc._C.apa_spq1_dump(c.q, c.k, c.v, c.ws['tq4'].packed, c.scale, 2.0, fmt='tq4', gqa_path='tc'))
    v2 = tq()
    os.environ['APA_FXTC1_VARIANT'] = '0'
    try:
        v0 = tq()
        checks.append(dict(id='LF5 S3 knob: APA_FXTC1_VARIANT=0 on tc tq4 runs and differs from variant 2',
                           ok=not np.array_equal(uview(v0), uview(v2)) and informative(v0)))
        try:
            tc.apa_eb3(c.q, c.k, c.v, c.ws['tq4'].packed, c.scale, 2.0)
            checks.append(dict(id='LF5 S3 knob: apa_eb3 with APA_FXTC1_VARIANT=0 raises as today', ok=False))
        except RuntimeError as e:
            checks.append(dict(id='LF5 S3 knob: apa_eb3 with APA_FXTC1_VARIANT=0 raises as today',
                               ok='variant B only' in str(e), raised=str(e)[:200]))
    finally:
        del os.environ['APA_FXTC1_VARIANT']
    os.environ['TC_APA_SPQ1_REFERENCE'] = '1'
    try:
        ref = npf(tc._C.apa_spq1_dump(c.q, c.k, c.v, c.ws['tq4'].packed, c.scale, 2.0, fmt='tq4', gqa_path='r2'))
        checks.append(dict(id='LF5 S3 knob: TC_APA_SPQ1_REFERENCE=1 on packed r2 G4 runs', ok=informative(ref)))
    except BaseException as e:
        checks.append(dict(id='LF5 S3 knob: TC_APA_SPQ1_REFERENCE=1 on packed r2 G4 runs', ok=False, raised=repr(e)))
    finally:
        del os.environ['TC_APA_SPQ1_REFERENCE']
    for family in ('eb3', 'eb4'):
        sep = native._p1_test_run(False, qt, kt, vt, vi, vs, w.packed, scale, 8.1, 1 / 32, True, 8)[0]
        os.environ['APA_P1_EARLY_COUNTER'] = '1'
        try:
            got = native._p1_test_run(family == 'eb4', qt, kt, vt, vi, vs, w.packed, scale, 8.1, 1 / 32, True, 0)[0]
        finally:
            os.environ.pop('APA_P1_EARLY_COUNTER', None)
        a, b = npf(sep).astype(np.float32), npf(got).astype(np.float32)
        checks.append(dict(id=f'LF5 S3 knob: APA_P1_EARLY_COUNTER=1 handshake ({family}) equals separate merge (variant 8)',
                           ok=bool(np.array_equal(uview(a), uview(b))) and informative(a)))
    return checks


def lf5(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arm = args.arm or 'FH'
    precheck(arm)
    log = []
    tc, a = load_arm(arm)
    step(log, 'lf5_loaded', arm=arm, module_sha256=a['module_sha256'], hooks=hasattr(tc._spo2, '_p1_test_run'))
    checks = lf5_run(tc, arm, out, [])
    step(log, 'lf5_measured', checks=len(checks))
    status = 'PASS' if checks and all(c['ok'] for c in checks) else 'FAIL'
    save(out / f'RESULT{"" if arm == "FH" else "_" + arm.replace(":", "_")}.json',
         dict(line='LF5', arm=arm, status=status, checks=checks, failed=[c['id'] for c in checks if not c['ok']],
              steps=log, terminal='LF5_DONE'))
    print('LF5', arm, status, [c['id'] for c in checks if not c['ok']], flush=True)
    return 0 if status == 'PASS' else 1


# ------------------------------------------------------------------ LF6
def lf6(args):
    import apa_p1d_gpu as g
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in ('B', 'F'):
        precheck(name)
    trees = {'B': TREE_B, 'F': ROOT}
    assert (TREE_B / 'tensor_cuda/tests').is_dir(), 'stage /tmp/apap1f1_treeB (git archive f01201cf) first'
    runs, log, results = [], [], []
    start = time.monotonic()
    for f in args.files:
        tag = Path(f).stem
        if time.monotonic() - start > args.max_seconds:
            results.append(dict(file=f, status='NOT_RUN_TIME_BUDGET'))
            continue
        recs = {}
        for armname in ('B', 'F'):
            arm = json.loads((arm_dir(armname) / 'ARM.json').read_text())
            env = os.environ.copy()
            env.pop('TC_APA_SP_STORAGE', None)
            env.pop('LD_PRELOAD', None)
            env.update(PYTHONPATH=str((ROOT / arm['package']).parent) + ':' + str(trees[armname] / 'tensor_cuda/tests') +
                       ':' + str(ROOT / 'artifacts/apa_p1'), PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
                       PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       TMPDIR='/tmp/apap1f1_l6')
            Path(env['TMPDIR']).mkdir(exist_ok=True)
            receipt = out / f'{tag}_{armname}.json'
            cmd = [sys.executable, str(ROOT / 'artifacts/apa_p1/pytest_receipt.py'), str(receipt), '-q', '--tb=short',
                   '-ra', '-p', 'no:cacheprovider', '--basetemp', f'/tmp/apap1f1_l6/bt_{tag}_{armname}', f]
            t0 = time.monotonic()
            with (out / f'{tag}_{armname}.log').open('x') as fh:
                try:
                    code = subprocess.run(cmd, cwd=trees[armname], env=env, stdout=fh, stderr=subprocess.STDOUT,
                                          timeout=700).returncode
                except subprocess.TimeoutExpired:
                    code = 124
            runs.append(dict(file=f, arm=armname, exit=code, seconds=time.monotonic() - t0, cwd=str(trees[armname])))
            step(log, f'lf6_{tag}_{armname}', exit=code)
            recs[armname] = json.loads(receipt.read_text()) if receipt.exists() else None
        a, b = recs['B'], recs['F']
        if a is None or b is None:
            results.append(dict(file=f, status='FAIL_NO_RECEIPT'))
            continue
        rows = g.outcome_rows(a, b)
        assert a['module']['sha256'] == BUILD_B_SHA
        assert b['module']['sha256'] == build_receipt('F')['module_sha256']
        results.append(dict(file=f, ids=len(rows), collected_equal=a['collection'] == b['collection'],
                            collection_errors=[len(a['collection_errors']), len(b['collection_errors'])],
                            differences=sum(not r['equal'] for r in rows), outcomes_B=g._count(a),
                            outcomes_F=g._count(b), rows=rows))
    complete = all('ids' in r for r in results)
    ok = complete and all(r.get('differences') == 0 and r.get('collected_equal') for r in results)
    status = 'PASS' if ok else ('FAIL' if complete or any(r.get('differences') for r in results) else 'PARTIAL')
    save(out / f'RESULT.{time.time_ns()}.json', dict(line='LF6', status=status, complete=complete, files=args.files,
         results=results, runs=runs, steps=log, terminal='LF6_DONE' if complete else 'LF6_PARTIAL'))
    print('LF6', status, [(r['file'], r.get('ids'), r.get('differences'), r.get('status')) for r in results], flush=True)
    return 0 if ok else 1


# ------------------------------------------------------------------ MUT
def mut_child(args):
    """One mutant arm against its named line subset; prints the named assertions' outcomes."""
    name = args.arm
    tc, arm = load_arm(name)
    R = rev1()
    out = Path(args.out)
    m = json.loads((ART / 'mutants/MUTANTS.json').read_text())
    spec = {x['name']: x for x in m['mutants']}[name.split(':', 1)[1]]
    os.environ['TC_APA_SP'] = '1'
    checks = []
    if spec['line'] == 'LF2':
        _, checks = lf2_run(tc, R, name)
    elif spec['line'] == 'LF4':
        checks = lf4_run(tc, R, name, out, [])
    elif spec['line'] == 'LF5':
        checks = lf5_run(tc, name, out, [])
    print('MUT_JSON ' + json.dumps(dict(arm=name, module_sha256=arm['module_sha256'],
                                        checks=[dict(id=c['id'], ok=c['ok']) for c in checks]), default=str), flush=True)
    return 0


def mut(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    m = json.loads((ART / 'mutants/MUTANTS.json').read_text())
    log, results = [], []
    names = args.rows or [x['name'] for x in m['mutants'] if x['runs_on'] == 'gpu']
    for spec in [x for x in m['mutants'] if x['name'] in names]:
        arm = 'mutant:' + spec['name']
        precheck(arm)
        if spec['line'] == 'LF3':
            res, runs = run_rows(out, arm, [spec['row']], f'mut_{spec["name"]}')
            failed_ids = [spec['expected_failure']] if not res[0].get('ok') else []
            detail = res[0]
        else:
            logf = out / f'mut_{spec["name"]}.log'
            r = child([sys.executable, '-B', __file__, 'mut-child', '--arm', arm, '--out', str(out)], logf, 900)
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
    save(out / f'RESULT.{time.time_ns()}.json', dict(line='MUT', status=status, results=results, steps=log,
                                                    terminal='MUT_DONE'))
    print('MUT', status, sum(r['killed'] for r in results), '/', len(results), flush=True)
    return 0 if status == 'PASS' else 1


def bnd(args):
    """L60 boundary record: Build B on LF2's, LF3's and LF4's inputs (no gate; children per part)."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    precheck('B')
    runs = []
    for mode, sub in (('lf2', 'LF2'), ('lf3', 'LF3'), ('lf4', 'LF4')):
        runs.append(child([sys.executable, '-B', __file__, mode, '--arm', 'B', '--out', str(out / sub)],
                          out / f'{sub}_B.log', 600))
        print('BND', mode, runs[-1]['exit'], flush=True)
    save(out / 'RESULT.json', dict(line='BND', arm='B', runs=runs, terminal='BND_DONE'))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode')
    ap.add_argument('target', nargs='?')
    ap.add_argument('--out')
    ap.add_argument('--arm')
    ap.add_argument('--row')
    ap.add_argument('--rows', nargs='*')
    ap.add_argument('--family')
    ap.add_argument('--files', nargs='*')
    ap.add_argument('--max-seconds', type=float, default=1500)
    ap.add_argument('--supp', action='store_true')
    args = ap.parse_args()
    if args.mode == 'stage':
        return stage(args.target)
    if args.mode == 'shim':
        return build_shim()
    if args.mode == 'precheck':
        print(json.dumps(precheck(args.target)))
        return 0
    assert os.environ.get('CUDA_VISIBLE_DEVICES') != '', 'GPU modes run inside forge-gpu-run'
    fn = {'lf1': lf1, 'lf1-arm': lf1_arm, 'lf2': lf2, 'lf3': lf3, 'lf3-row': lf3_row, 'lf4': lf4,
          'lf4-child': lf4_child, 'lf5': lf5, 'lf5-fresh': lf5_fresh, 'lf6': lf6, 'mut': mut,
          'mut-child': mut_child, 'bnd': bnd}[args.mode]
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
