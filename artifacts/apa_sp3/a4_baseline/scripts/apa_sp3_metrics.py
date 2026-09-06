#!/usr/bin/env python3
"""All-pair model-activation audit, one captured layer per bounded CPU job.

Prior art: ThriftAttention (Sharratt 2026) weight-sensitive quantization error;
BLASST (Yuan et al. 2025/2026) max-relative softmax ratio; SP2 (2026) 2*eq
margin. Standard empirical order statistics and memory-mapped arrays are used.
New here: the MiniCPM3 captured-activation audit; no new bound is claimed.
"""
from __future__ import annotations
import math
import hashlib
from pathlib import Path
import numpy as np
from apa_sp3_common import ART, Red, read, sha, upward_float32, load_runtime


def nearest_rank(values, p):
    if not len(values):
        raise Red('empty percentile population')
    k = max(0, math.ceil(p * len(values)) - 1)
    return float(np.partition(values, k)[k])


def tail(exact, selected):
    """Dense exact probability of UNREFINED (still present) keys."""
    exact = np.asarray(exact, np.float64)
    if not exact.size or not np.isfinite(exact).all():
        raise Red('invalid eligible exact scores')
    wrel = np.exp(exact - exact.max())
    skipped = ~np.asarray(selected, bool)
    return (float(wrel[skipped].sum() / wrel.sum()),
            float(wrel[skipped].max()) if skipped.any() else 0.)


def analyze_capture(directory, scratch, bulk_provider=None):
    directory, scratch = Path(directory), Path(scratch)
    meta = read(directory / 'capture.json')
    for p, digest in meta['files'].items():
        if sha(directory / p) != digest:
            raise Red(f'activation capture changed: {p}')
    q, k, kq = [np.load(directory / (n + '.npy'), mmap_mode='r') for n in ('q', 'k', 'kq')]
    packed = np.load(directory / 'selected.pack.npy', mmap_mode='r')
    B, H, L, D = q.shape
    S = k.shape[2]
    lengths = S-L+np.arange(L)+1 if meta['causal'] else np.full(L, S)
    count = int(B * H * lengths.sum())
    if count != meta['pairs'] or D != 96 or B != 1:
        raise Red('capture pair/shape mismatch')
    if scratch.exists():
        raise Red('scratch already exists; do not overwrite interrupted evidence')
    actual_scale=float(np.float32(meta['scale']))
    # Prior art: existing SP warp/FMA dot, separate observational CUDA probe.
    # CPU tests inject a provider explicitly. Production never falls back to
    # a mathematical NumPy bulk dot for the measured e_q statistic.
    tc=diag=None
    if bulk_provider is None:
        tc=load_runtime()
        import _apa_sp3_diag as diag
    native_bulk=None
    if tc is not None and meta.get('native_bulk_chunks'):
        # Reproduce the ACTUAL B cuBLAS batch/chunk geometry and both BF16
        # roundings. Hashes taken during B capture must match byte-for-byte.
        # This is replay of existing Perry APA code, not a different quantizer.
        kqt_full=tc.tensor(np.asarray(kq,np.float32)).astype(meta['dtype'])
        native_bulk=np.empty((B,H,L,S),np.float32)
        for chunk in meta['native_bulk_chunks']:
            lo,n=chunk['row0'],chunk['length']
            qt=tc.tensor(np.asarray(q[:,:,lo:lo+n],np.float32)).astype(meta['dtype'])
            b=(tc.matmul(qt,kqt_full.transpose(-2,-1))*meta['scale']).float().numpy()
            if hashlib.sha256(b.tobytes()).hexdigest()!=chunk['sha256']:
                raise Red('B native bulk replay SHA mismatch')
            native_bulk[:,:,lo:lo+n]=b
        del kqt_full,qt,b
    errors = np.memmap(scratch, dtype='<f4', mode='w+', shape=(count,))
    sp_scratch=scratch.with_suffix('.sp.f32')
    if sp_scratch.exists():
        raise Red('SP scratch already exists')
    sp_errors=np.memmap(sp_scratch,dtype='<f4',mode='w+',shape=(count,)) if native_bulk is not None else None
    masses, relatives, selected_count = [], [], 0
    errsum, errmax, cursor, sp_sum, sp_max = 0., 0., 0, 0., 0.
    # Grid occupancy uses ceil(gap/step): candidate d selects gap<=d.
    # BLASST-style calibration on B activations is only an initializer; C's
    # actual native fraction is subsequently measured and required to match.
    hist = np.zeros(4098, np.int64)
    try:
        for h in range(H):
            kh, kqh = np.asarray(k[0, h], np.float64), np.asarray(kq[0, h], np.float64)
            if tc is not None:
                kqt=tc.tensor(np.asarray(kq[:,h:h+1],np.float32)).astype(meta['dtype'])
            for lo in range(0, L, 16):
                hi = min(L, lo+16)
                qs = np.asarray(q[0, h, lo:hi], np.float64)
                exact = (qs @ kh.T) * actual_scale
                if bulk_provider is None:
                    qt=tc.tensor(np.asarray(q[:,h:h+1,lo:hi],np.float32)).astype(meta['dtype'])
                    bulk=diag.bulk_scores(qt,kqt,actual_scale).numpy()[0,0]
                    del qt
                else:
                    bulk=np.asarray(bulk_provider(h,lo,hi,actual_scale),np.float32)
                masks = np.unpackbits(packed[0, h, lo:hi], axis=-1, count=S, bitorder='little')
                for r, n in enumerate(lengths[lo:hi]):
                    ex, bu, mask = exact[r, :n], bulk[r, :n], masks[r, :n].astype(bool)
                    if not np.isfinite(bu).all():
                        raise Red('nonfinite model bulk scores')
                    er = np.abs(bu.astype(np.float64)-ex)
                    sp_sum+=float(er.sum(dtype=np.float64));sp_max=max(sp_max,float(er.max()))
                    if sp_errors is not None:
                        sp_errors[cursor:cursor+n]=er
                        er=np.abs(native_bulk[0,h,lo+r,:n].astype(np.float64)-ex)
                    errors[cursor:cursor+n] = er
                    cursor += int(n)
                    errsum += float(er.sum(dtype=np.float64))
                    errmax = max(errmax, float(er.max()))
                    mass, relative = tail(ex, mask)
                    masses.append(mass)
                    relatives.append(relative)
                    selected_count += int(mask.sum())
                    gaps = np.maximum.accumulate(bu)-bu
                    indices = np.minimum(np.ceil(gaps*128), 4097).astype(np.int64)
                    hist += np.bincount(indices, minlength=4098)
        if cursor != count or selected_count != meta['selected']:
            raise Red('all-pair/native-mask count mismatch')
        # In-place selection avoids a second 5.4 GB copy at 8192. Scratch is
        # owned by this invocation. Only float32 storage affects percentiles;
        # sum and maximum use the original float64 dot errors.
        ix = [math.ceil(p*count)-1 for p in (.99, .999)]
        errors.partition(ix)
        p99, p999 = [float(errors[i]) for i in ix]
        if sp_errors is not None:
            sp_errors.partition(ix)
            sp_p99,sp_p999=[float(sp_errors[i]) for i in ix]
        else:sp_p99,sp_p999=p99,p999
        return {'layer': meta['layer'], 'arm': meta['arm'], 'bits': meta['bits'], 'S': S,
                'pairs': count, 'queries': len(masses), 'selected': selected_count,
                'fraction': selected_count/count, 'delta': meta['delta'],
                'error': {'mean': errsum/count, 'p99': p99, 'p99_9': p999, 'max': errmax,
                          'eq_upward_float32': upward_float32(errmax)},
                'sp_error': {'mean':sp_sum/count,'p99':sp_p99,'p99_9':sp_p999,'max':sp_max,
                             'eq_upward_float32':upward_float32(sp_max)},
                'unrefined_mass': {'mean': float(np.mean(masses)),
                                  'p99': nearest_rank(np.asarray(masses), .99), 'max': max(masses)},
                'max_skipped_relative_weight': max(relatives),
                'grid_gap_histogram': hist.tolist(), 'percentile_convention': 'nearest-rank, float32 stored errors',
                'score_convention': 'error uses actual arm bulk: B blend replay verified by captured SHA, otherwise native SP-order FP32 dot; sp_error always uses SP dot for E margin; exact float64 Q.K at actual float32 scale',
                'capture_sha256': sha(directory / 'capture.json'),
                'evidence_class': 'kernel sweep',
                'scope': 'real model activations; establishes nothing about model quality by itself'}
    finally:
        del errors
        scratch.unlink(missing_ok=True)
        if sp_errors is not None:
            del sp_errors
            sp_scratch.unlink(missing_ok=True)


def initial_delta(rows, target):
    h = np.sum([r['grid_gap_histogram'] for r in rows], axis=0, dtype=np.int64)
    if h.sum() <= 0:
        raise Red('empty calibration histogram')
    fractions = np.cumsum(h[:4097], dtype=np.int64)/h.sum()
    i = int(np.argmin(np.abs(fractions-target)))
    return i/128., float(fractions[i])


def next_delta(previous, target):
    if not previous:
        raise Red('missing C initial trial')
    lo, hi = 0., 32.
    for trial in previous:
        if trial['fraction'] < target:
            lo = max(lo, trial['delta'])
        else:
            hi = min(hi, trial['delta'])
    if hi <= lo:
        raise Red('C realised model fraction broke calibration bracket')
    return float(np.float32((lo+hi)/2))
