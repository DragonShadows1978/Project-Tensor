#!/usr/bin/env python3
"""APA-P1-PF2 G4 (refusal rows, acceptance rows, BOUNDARY.json) and G5 (allocation high-water census), GPU lines.

  g4 --out D            : children per arm (PF, F3) run the same rows; writes G4 RESULT.json and BOUNDARY.json
  g4-arm --arm A        : (child) the rows on one arm, under the allocation/launch shim
  g5 --out D            : (child per geometry) the census on Build PF
  shim                  : CPU: compile the LD_PRELOAD shim into /tmp/apap1pf2_shim

Evidence that a refusal comes before any device work (P1-D L2's form, extended as F1/F3 did for native refusals):
the Python device entries are spied (spo2.packed_prefill/packed_run/packed_prepare/packed_append, the dense quantizer,
the dense epsilon entry); a Python refusal must show 0 spied calls; a refusal raised inside the native entry shows
1 spied call and must show 0 kernel launches (cudaLaunchKernel counter) and 0 bytes allocated (cudaMalloc/
cudaMallocAsync counter) during the call. Both counts are recorded for every row.

G5: the shim tracks live bytes of cudaMalloc/cudaMallocAsync minus cudaFree/cudaFreeAsync and their high-water;
the call's high-water above the bytes live before it must be <= out + mask (if diagnostics) + one granule
(2 MiB, P1-D L4's minimum). Allocator pooling is off (raw allocations) and on (stream-ordered) in separate cases.
Prior art: LD_PRELOAD interposition (POSIX dlsym RTLD_NEXT); F1's launch-count shim (Project-Tensor 2026);
P1-D L2's spy and L4's device-memory method (Project-Tensor 2026). No new technique.
"""
import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import apa_p1pf2_gpu as G  # noqa: E402

SHIM_DIR = Path('/tmp/apap1pf2_shim')
SHIM = SHIM_DIR / 'libpf2count.so'
SHIM_SRC = r'''
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <pthread.h>
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static unsigned long long launches_n, allocs_n, cur, peak;
#define NS (1u << 18)
static void* K[NS]; static size_t V[NS];
static void put(void* p, size_t s) {
  if (!p) return; pthread_mutex_lock(&mu);
  size_t h = ((uintptr_t)p >> 8) & (NS - 1);
  while (K[h] && K[h] != (void*)1) h = (h + 1) & (NS - 1);
  K[h] = p; V[h] = s; cur += s; allocs_n++; if (cur > peak) peak = cur; pthread_mutex_unlock(&mu); }
/* libcudart is a dependency of the RTLD_LOCAL extension module, not in the global scope RTLD_NEXT searches:
   take the loaded library's handle first (F1's shim, Project-Tensor 2026). */
static void* real(const char* n) {
  void* h = dlopen("libcudart.so.12", RTLD_NOLOAD | RTLD_LAZY); void* f = h ? dlsym(h, n) : 0;
  return f ? f : dlsym(RTLD_NEXT, n); }
static void del(void* p) {
  if (!p) return; pthread_mutex_lock(&mu);
  size_t h = ((uintptr_t)p >> 8) & (NS - 1);
  for (size_t n = 0; n < NS && K[h]; ++n, h = (h + 1) & (NS - 1))
    if (K[h] == p) { cur -= V[h]; K[h] = (void*)1; break; }
  pthread_mutex_unlock(&mu); }
int cudaMalloc(void** p, size_t s) { static int (*f)(void**, size_t); if (!f) f = real("cudaMalloc");
  int r = f(p, s); if (!r) put(*p, s); return r; }
int cudaFree(void* p) { static int (*f)(void*); if (!f) f = real("cudaFree"); del(p); return f(p); }
int cudaMallocAsync(void** p, size_t s, void* st) { static int (*f)(void**, size_t, void*);
  if (!f) f = real("cudaMallocAsync"); int r = f(p, s, st); if (!r) put(*p, s); return r; }
int cudaFreeAsync(void* p, void* st) { static int (*f)(void*, void*); if (!f) f = real("cudaFreeAsync");
  del(p); return f(p, st); }
typedef struct { unsigned x, y, z; } d3;
int cudaLaunchKernel(const void* fn, d3 g, d3 b, void** a, size_t sm, void* st) {
  static int (*f)(const void*, d3, d3, void**, size_t, void*); if (!f) f = real("cudaLaunchKernel");
  __atomic_add_fetch(&launches_n, 1, __ATOMIC_SEQ_CST); return f(fn, g, b, a, sm, st); }
unsigned long long pf2_launches(void) { return launches_n; }
unsigned long long pf2_allocs(void) { return allocs_n; }
unsigned long long pf2_cur(void) { return cur; }
unsigned long long pf2_peak(void) { return peak; }
void pf2_reset_peak(void) { pthread_mutex_lock(&mu); peak = cur; pthread_mutex_unlock(&mu); }
'''


def build_shim():
    SHIM_DIR.mkdir(exist_ok=True)
    src = SHIM_DIR / 'pf2count.c'
    src.write_text(SHIM_SRC)
    subprocess.run(['timeout', '60s', 'gcc', '-O2', '-shared', '-fPIC', str(src), '-o', str(SHIM), '-ldl', '-lpthread'],
                   check=True)
    print('shim', SHIM, G.sha(SHIM))
    return 0


class Counter:
    def __init__(self):
        lib = ctypes.CDLL(None)
        self.f = {n: getattr(lib, 'pf2_' + n) for n in ('launches', 'allocs', 'cur', 'peak')}
        for f in self.f.values():
            f.restype = ctypes.c_ulonglong
        self.reset = lib.pf2_reset_peak

    def __call__(self, n):
        return int(self.f[n]())


def ensure_shim():
    if getattr(ctypes.CDLL(None), 'pf2_launches', None) is None:
        assert os.environ.get('LD_PRELOAD') != str(SHIM), 'shim preloaded but not visible'
        env = dict(os.environ, LD_PRELOAD=str(SHIM))
        os.execve(sys.executable, [sys.executable, '-X', 'faulthandler', '-u', '-B', '-P', str(Path(__file__).resolve()), *sys.argv[1:]], env)


def np_informative(a):
    u = np.ascontiguousarray(a).reshape(-1)
    u = u.view({1: np.uint8, 2: np.uint16, 4: np.uint32}[u.dtype.itemsize])
    return bool(u.size) and bool((u != 0).any()) and bool((u != u[0]).any())


def fake(tc, shape, dtype):
    """A tensor of the given shape over a small real buffer (F1 reimport(), imported; F3 LF3's fake())."""
    import apa_p1f1_gpu as f1
    return f1.reimport(tc, tc.empty((64,), dtype=dtype), tuple(shape), 0)


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


# ------------------------------------------------------------------ G4
def g4_arm(args):
    ensure_shim()
    cnt = Counter()
    tc, arm = G.load_arm(args.arm)
    os.environ.pop('TC_APA_SP_STORAGE', None)
    os.environ['TC_APA_SP'] = '1'
    dm = tc.RegisteredMargins.load(G.DENSE_TABLE, expected_sha256=G.DENSE_TABLE_SHA)
    pm = tc.PackedRegisteredMargins.load(G.PACKED_TABLE)
    real_spo2 = tc._spo2
    spo2 = Spy(real_spo2, ['packed_prefill', 'packed_run', 'packed_prepare', 'packed_append'])
    tc._spo2 = spo2
    real_quant = tc.quant._quantize_keys
    qcalls = []
    tc.quant._quantize_keys = lambda *a: qcalls.append(1) or real_quant(*a)
    real_eps = tc._C.apa_selective_attention_sp_epsilon
    ecalls = []
    tc._C.apa_selective_attention_sp_epsilon = lambda *a: ecalls.append(1) or real_eps(*a)
    rows = []

    def expect(name, fn, exc, match):
        n0, l0, a0 = len(spo2.calls) + len(qcalls) + len(ecalls), cnt('launches'), cnt('allocs')
        tc.synchronize()
        try:
            fn()
            got, msg = None, ''
        except Exception as e:  # noqa: BLE001
            got, msg = type(e).__name__, str(e)
        tc.synchronize()
        calls = len(spo2.calls) + len(qcalls) + len(ecalls) - n0
        launches, allocs = cnt('launches') - l0, cnt('allocs') - a0
        raised_ok = got == exc and (match is None or match in msg)
        before_device = calls == 0 or (launches == 0 and allocs == 0)
        rows.append(dict(id=name, kind='refusal', expected=exc, match=match, raised=got, message=msg[:300],
                         spied_calls=calls, launches=launches, allocations=allocs,
                         evidence='python (0 spied calls)' if calls == 0 else 'native (0 launches, 0 allocations)',
                         ok=raised_ok and before_device))

    rng = np.random.default_rng(np.random.SeedSequence([20261116, 4]))
    D, S, KV, H, L = 128, 257, 2, 8, 33
    k = rng.standard_normal((1, KV, S, D), dtype=np.float32)
    v = rng.standard_normal((1, KV, S, D), dtype=np.float32)
    q = (3 * rng.standard_normal((1, H, L, D))).astype(np.float32)
    s = rng.standard_normal((H,), dtype=np.float32)
    scale = float(np.float32(1 / math.sqrt(D)))
    call = tc.apa_selective_attention_sp
    with tc.no_grad():
        kt, vt, qt, st = tc.tensor(k), tc.tensor(v), tc.tensor(q), tc.tensor(s)
        p = tc.quantize_sp2_keys(kt, 4)
        assert isinstance(p, tc.PackedSpKeys)
        qcalls.clear()
        spo2.calls.clear()
        del os.environ['TC_APA_SP']
        expect('PF_tc_apa_sp_unset', lambda: call(qt, kt, p, vt, scale, .01, bulk_bits=4, margins=pm, is_causal=True),
               'RuntimeError', 'TC_APA_SP=1')
        os.environ['TC_APA_SP'] = '1'
        expect('PF_dense_table_on_packed', lambda: call(qt, kt, p, vt, scale, .01, bulk_bits=4, margins=dm, is_causal=True),
               'ValueError', 'dense margin table')
        fake_table = tc.PackedRegisteredMargins('0' * 64, dict(pm.entries))
        expect('PF_unpinned_packed_table', lambda: call(qt, kt, p, vt, scale, .01, bulk_bits=4, margins=fake_table, is_causal=True),
               'ValueError', 'pinned')
        q3 = tc.tensor(q[0])
        expect('PF_query_rank_three', lambda: call(q3, kt, p, vt, scale, .01, bulk_bits=4, margins=pm, is_causal=True),
               'ValueError', 'rank four')
        expect('PF_bulk_bits_mismatch', lambda: call(qt, kt, p, vt, scale, .01, bulk_bits=2, margins=pm, is_causal=True),
               'ValueError', 'bulk_bits')
        kcopy = tc.tensor(k)
        expect('PF_check_source_mismatch', lambda: call(qt, kcopy, p, vt, scale, .01, bulk_bits=4, margins=pm, is_causal=True),
               'ValueError', 'exact-K source')
        qc, vc, sc = tc.tensor(q, device='cpu'), tc.tensor(v, device='cpu'), tc.tensor(s, device='cpu')
        s4 = tc.tensor(np.ascontiguousarray(s[:4]))
        expect('PF_q_other_device', lambda: call(qc, kt, p, vt, scale, .01, bulk_bits=4, margins=pm, is_causal=True),
               'ValueError', 'device')
        expect('PF_v_other_device', lambda: call(qt, kt, p, vc, scale, .01, bulk_bits=4, margins=pm, is_causal=True),
               'ValueError', 'device')
        expect('PF_sinks_other_device', lambda: call(qt, kt, p, vt, scale, .01, bulk_bits=4, margins=pm, is_causal=True,
                                                     sinks=sc), 'ValueError', 'device')
        qlong = tc.tensor((3 * rng.standard_normal((1, H, S + 1, D))).astype(np.float32))
        expect('PF_causal_S_less_than_L', lambda: call(qlong, kt, p, vt, scale, .01, bulk_bits=4, margins=pm,
                                                      is_causal=True), 'RuntimeError', 'S >= L')
        vbig = tc.tensor(rng.standard_normal((1, KV, S, 640), dtype=np.float32))
        expect('PF_VD_above_cap', lambda: call(qt, kt, p, vbig, scale, .01, bulk_bits=4, margins=pm, is_causal=True),
               'RuntimeError', 'dimension cap')
        q96 = tc.tensor((3 * rng.standard_normal((1, H, L, 96))).astype(np.float32))
        expect('PF_D_not_packed', lambda: call(q96, kt, p, vt, scale, .01, bulk_bits=4, margins=pm, is_causal=True),
               'ValueError', None)
        vbf = tc.tensor(v, dtype='bfloat16')
        expect('PF_mixed_dtypes', lambda: call(qt, kt, p, vbf, scale, .01, bulk_bits=4, margins=pm, is_causal=True),
               'RuntimeError', 'dtype')
        qbf = tc.tensor(q, dtype='bfloat16')
        expect('PF_query_bf16_on_packed_fp32', lambda: call(qbf, kt, p, vt, scale, .01, bulk_bits=4, margins=pm,
                                                           is_causal=True), 'ValueError', 'fp32')
        expect('PF_nonfinite_scale', lambda: call(qt, kt, p, vt, float('nan'), .01, bulk_bits=4, margins=pm,
                                                  is_causal=True), 'ValueError', 'scale')
        expect('PF_nonpositive_scale', lambda: call(qt, kt, p, vt, -scale, .01, bulk_bits=4, margins=pm, is_causal=True),
               'ValueError', 'scale')
        expect('PF_epsilon_out_of_domain', lambda: call(qt, kt, p, vt, scale, 0., bulk_bits=4, margins=pm, is_causal=True),
               'ValueError', 'epsilon')
        expect('PF_epsilon_nonfinite', lambda: call(qt, kt, p, vt, scale, float('nan'), bulk_bits=4, margins=pm,
                                                    is_causal=True), 'ValueError', 'epsilon')
        expect('PF_sinks_wrong_shape', lambda: call(qt, kt, p, vt, scale, .01, bulk_bits=4, margins=pm, is_causal=True,
                                                    sinks=s4), 'RuntimeError', 'sinks')
        kshort = tc.tensor(np.ascontiguousarray(k[:, :, :200]))
        pa = tc.PackedSpKeys(kshort, capacity=S)
        expect('PF_visible_rows_not_k_rows', lambda: call(qt, kt, pa, vt, scale, .01, bulk_bits=4, margins=pm,
                                                          is_causal=True), 'ValueError', 'exact-K source')
        raw = getattr(spo2, 'packed_prefill', None) if hasattr(real_spo2, 'packed_prefill') else None
        # Fake-extent operands over small real buffers, built before the row so the row's own allocation count is
        # the entry's (F3 LF3's fake(), F1 reimport()).
        FK = [fake(tc, (65536, 8, 8192, D), 'float32'), fake(tc, (65536, KV, 8192, D), 'float32'),
              fake(tc, (65536, KV, 8192, D), 'float32'), fake(tc, (65536, KV, 8192, D // 2), 'uint8'),
              fake(tc, (65536, KV, 8192), 'float32')]
        q16, k16, v16 = tc.tensor(q, dtype='float16'), tc.tensor(k, dtype='float16'), tc.tensor(v, dtype='float16')
        for name, fn, exc, match in [
            ('PF_raw_fmt_int4', lambda: raw(qt, kt, vt, p._packed_tuple(), scale, 6.3, True, None, False, fmt='int4'),
             'ValueError', 'tq4'),
            ('PF_raw_negative_delta', lambda: raw(qt, kt, vt, p._packed_tuple(), scale, -1.0, True, None, False,
                                                  fmt='tq4'), 'RuntimeError', 'delta'),
            ('PF_raw_nonfinite_delta', lambda: raw(qt, kt, vt, p._packed_tuple(), scale, float('inf'), True, None,
                                                   False, fmt='tq4'), 'RuntimeError', 'delta'),
            ('PF_raw_dump_in_default_build', lambda: raw(qt, kt, vt, p._packed_tuple(), scale, 6.3, True, None, False,
                                                         fmt='tq4', dump=True), 'ValueError', 'TEST_HOOKS'),
            ('PF_raw_launch_range', lambda: raw(FK[0], FK[1], FK[2], (FK[3], FK[4], p.rotation, p.codebook),
                                                scale, 6.3, False, None, False, fmt='tq4'),
             'RuntimeError', 'launch range'),
            ('PF_raw_fp16', lambda: raw(q16, k16, v16, p._packed_tuple(), scale, 6.3, True, None, False, fmt='tq4'),
             'RuntimeError', 'packed K')]:
            expect(name, fn, exc, match)
        if args.arm == 'F3':
            try:
                call(qt, kt, p, vt, scale, .01, bulk_bits=4, margins=pm, is_causal=True)
                f3 = 'ran'
            except Exception as e:  # noqa: BLE001
                f3 = f'{type(e).__name__}: {e}'
            rows.append(dict(id='F3_prefill_on_packed', kind='before-build behaviour', result=f3, ok=True))
        else:
            # Acceptance rows (the one intended change of P1-D L2: RE_prefill_on_packed_refused -> runs).
            spo2.calls.clear()
            out, mask = call(qt, kt, p, vt, scale, .01, bulk_bits=4, margins=pm, is_causal=True, sinks=st, diagnostics=True)
            route = tc.apa_packed.last_route('attention')
            e_pin = tc.apa_packed.PACKED_PINNED_ENTRIES['4:128']
            d_pin = tc.apa_packed.derive_delta(.01, e_pin)
            r_out, r_mask = real_spo2.packed_prefill(qt, kt, vt, p._packed_tuple(), scale, d_pin, True, st, True, fmt='tq4')
            o, m, ro, rm = out.numpy(), mask.numpy(), r_out.numpy(), r_mask.numpy()
            rows.append(dict(id='g4_prefill_on_packed_runs', kind='acceptance (replaces P1-D RE_prefill_on_packed_refused)',
                             ok=np_informative(o) and np_informative(m) and spo2.calls == ['packed_prefill'],
                             spied_calls=list(spo2.calls)))
            rows.append(dict(id='g4_route_record_packed_delta', kind='acceptance', route=route,
                             ok=(route is not None and route.get('backend') == 'spq_packed_prefill_tq4_row' and
                                 route.get('storage') == 'packed' and route.get('l') == L and route.get('e_q') == e_pin and
                                 route.get('delta') == d_pin and route.get('fallback_reason') is None)))
            rows.append(dict(id='g4_public_equals_raw_entry_bitwise', kind='acceptance',
                             ok=bool(np.array_equal(o.view(np.uint32), ro.view(np.uint32)) and np.array_equal(m, rm))))
        # Caller census (L90) is CPU (artifacts/apa_p1pf2/CALLER_CENSUS.json).
    tc._spo2 = real_spo2
    print('G4_JSON ' + json.dumps(dict(arm=arm, rows=rows), default=str), flush=True)
    return 0


def g4(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for armname in ('PF', 'F3'):
        G.precheck(armname)
        log = out / f'arm_{armname}.log'
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        with log.open('x') as f:
            code = subprocess.run([sys.executable, '-B', '-P', __file__, 'g4-arm', '--arm', armname], stdout=f,
                                  stderr=subprocess.STDOUT, env=env, timeout=600, cwd=ROOT).returncode
        line = [x for x in log.read_text().splitlines() if x.startswith('G4_JSON ')]
        res[armname] = dict(exit=code, **(json.loads(line[-1][8:]) if line else {}))
    pf = res['PF'].get('rows', [])
    ref = [r for r in pf if r['kind'] == 'refusal']
    acc = [r for r in pf if r['kind'] != 'refusal']
    f3 = {r['id']: r for r in res['F3'].get('rows', [])}
    boundary = [dict(id=r['id'], input=r['id'], PF=dict(raised=r['raised'], message=r['message'],
                                                        spied_calls=r['spied_calls'], launches=r['launches']),
                     F3=dict(raised=f3.get(r['id'], {}).get('raised'), message=f3.get(r['id'], {}).get('message')))
                for r in ref]
    status = 'PASS' if ref and all(r['ok'] for r in ref) and acc and all(r['ok'] for r in acc) else 'FAIL'
    G.save(out / 'RESULT.json', dict(line='G4', status=status, refusals=len(ref), refusals_ok=sum(r['ok'] for r in ref),
                                     acceptance=acc, rows=pf, arms={k: dict(exit=v['exit'], arm=v.get('arm'))
                                                                    for k, v in res.items()},
                                     F3_rows=res['F3'].get('rows'), terminal='G4_DONE'))
    G.save(G.ART / 'BOUNDARY.json', dict(note='each refusal input on Build PF and on Build F3 (L60)', rows=boundary))
    print('G4', status, sum(r['ok'] for r in ref), '/', len(ref), [r['id'] for r in pf if not r['ok']], flush=True)
    return 0


# ------------------------------------------------------------------ G5
G5_GEOM = {'qwen3_1.7b': 16, 'qwen3_4b': 32}
G5_SL = [(2048, 2048), (8192, 8192), (32768, 2048)]
GRANULE = 2 << 20


def g5_case(args):
    ensure_shim()
    cnt = Counter()
    tc, arm = G.load_arm(args.arm)
    os.environ.pop('TC_APA_SP_STORAGE', None)
    os.environ['TC_APA_SP'] = '1'
    pm = tc.PackedRegisteredMargins.load(G.PACKED_TABLE)
    H, KV, D = G5_GEOM[args.model], 8, 128
    rows = []
    rng = np.random.default_rng(np.random.SeedSequence([20261116, 5, H]))
    scale = float(np.float32(1 / math.sqrt(D)))
    with tc.no_grad():
        for S, L in G5_SL:
            k = tc.tensor(rng.standard_normal((1, KV, S, D), dtype=np.float32))
            v = tc.tensor(rng.standard_normal((1, KV, S, D), dtype=np.float32))
            q = tc.tensor((3 * rng.standard_normal((1, H, L, D))).astype(np.float32))
            p = tc.quantize_sp2_keys(k, 4)
            assert isinstance(p, tc.PackedSpKeys)
            # warm-up (module/kernel load, any one-time context allocation), then release it
            w = tc.apa_selective_attention_sp(q, k, p, v, scale, .01, bulk_bits=4, margins=pm, is_causal=True)
            del w
            tc.synchronize()
            for pooling in (False, True):
                for diag in (False, True):
                    tc.empty_cache()
                    tc.set_alloc_pooling(pooling)
                    tc.synchronize()
                    before = cnt('cur')
                    cnt.reset()
                    a0 = cnt('allocs')
                    r = tc.apa_selective_attention_sp(q, k, p, v, scale, .01, bulk_bits=4, margins=pm, is_causal=True,
                                                      diagnostics=diag)
                    tc.synchronize()
                    hw = cnt('peak') - before
                    route = tc.apa_packed.last_route('attention')
                    out_b = H * L * D * 4
                    mask_b = H * L * S if diag else 0
                    key_shaped = KV * S * D * 4
                    rows.append(dict(model=args.model, H=H, S=S, L=L, pooling=pooling, diagnostics=diag,
                                     high_water_above_resident=hw, out_bytes=out_b, mask_bytes=mask_b,
                                     granule=GRANULE, bound=out_b + mask_b + GRANULE, allocations=cnt('allocs') - a0,
                                     key_shaped_bytes=key_shaped, route_backend=route['backend'],
                                     within_bound=hw <= out_b + mask_b + GRANULE,
                                     no_key_shaped_transient=hw - out_b - mask_b < key_shaped))
                    del r
                    tc.set_alloc_pooling(False)
                    tc.synchronize()
            del k, v, q, p
            tc.empty_cache()
    print('G5_JSON ' + json.dumps(dict(arm=arm, rows=rows)), flush=True)
    return 0


def g5(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arm = args.arm or 'PF'
    rows, runs = [], []
    for model in G5_GEOM:
        log = out / f'{model}.log'
        with log.open('x') as f:
            code = subprocess.run([sys.executable, '-B', '-P', __file__, 'g5-case', '--arm', arm, '--model', model],
                                  stdout=f, stderr=subprocess.STDOUT, timeout=900, cwd=ROOT,
                                  env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1')).returncode
        runs.append(dict(model=model, exit=code))
        line = [x for x in log.read_text().splitlines() if x.startswith('G5_JSON ')]
        if line:
            rows += json.loads(line[-1][8:])['rows']
    ok = len(rows) == 24 and all(r['within_bound'] and r['no_key_shaped_transient'] and
                                 r['route_backend'] == 'spq_packed_prefill_tq4_row' for r in rows)
    G.save(out / 'RESULT.json', dict(line='G5', arm=arm, status='PASS' if ok else 'FAIL', runs=runs, rows=rows,
                                     g5_no_key_shaped_transient=all(r['no_key_shaped_transient'] for r in rows) and
                                     len(rows) == 24, terminal='G5_DONE'))
    print('G5', 'PASS' if ok else 'FAIL', flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode')
    ap.add_argument('--out')
    ap.add_argument('--arm')
    ap.add_argument('--model')
    args = ap.parse_args()
    if args.mode == 'shim':
        return build_shim()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') != '', 'GPU modes run inside forge-gpu-run'
    return {'g4': g4, 'g4-arm': g4_arm, 'g5': g5, 'g5-case': g5_case}[args.mode](args)


if __name__ == '__main__':
    raise SystemExit(main())
