#!/usr/bin/env python3
"""All-pair model-activation audit, one captured layer per bounded CPU job.

Prior art: ThriftAttention (Sharratt 2026) weight-sensitive quantization error;
BLASST (Yuan et al. 2025/2026) max-relative softmax ratio; SP2 (2026) 2*eq
margin. Standard empirical order statistics and memory-mapped arrays are used.
New here: the MiniCPM3 captured-activation audit; no new bound is claimed.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
from apa_sp3_common import ART, Red, read, sha, upward_float32


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
    skipped = np.asarray(selected, bool)
    return (float(wrel[skipped].sum() / wrel.sum()),
            float(wrel[skipped].max()) if skipped.any() else 0.)


def analyze_capture(directory, scratch):
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
    errors = np.memmap(scratch, dtype='<f4', mode='w+', shape=(count,))
    masses, relatives, selected_count = [], [], 0
    errsum, errmax, cursor = 0., 0., 0
    # Grid occupancy uses ceil(gap/step): candidate d selects gap<=d.
    # BLASST-style calibration on B activations is only an initializer; C's
    # actual native fraction is subsequently measured and required to match.
    hist = np.zeros(4098, np.int64)
    try:
        for h in range(H):
            kh, kqh = np.asarray(k[0, h], np.float64), np.asarray(kq[0, h], np.float64)
            for lo in range(0, L, 16):
                hi = min(L, lo+16)
                qs = np.asarray(q[0, h, lo:hi], np.float64)
                exact = (qs @ kh.T) * meta['scale']
                bulk = (qs @ kqh.T) * meta['scale']
                masks = np.unpackbits(packed[0, h, lo:hi], axis=-1, count=S, bitorder='little')
                for r, n in enumerate(lengths[lo:hi]):
                    ex, bu, mask = exact[r, :n], bulk[r, :n], masks[r, :n].astype(bool)
                    if not np.isfinite(bu).all():
                        raise Red('nonfinite model bulk scores')
                    er = np.abs(bu-ex)
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
        return {'layer': meta['layer'], 'arm': meta['arm'], 'bits': meta['bits'], 'S': S,
                'pairs': count, 'queries': len(masses), 'selected': selected_count,
                'fraction': selected_count/count, 'delta': meta['delta'],
                'error': {'mean': errsum/count, 'p99': p99, 'p99_9': p999, 'max': errmax,
                          'eq_upward_float32': upward_float32(errmax)},
                'unrefined_mass': {'mean': float(np.mean(masses)),
                                  'p99': nearest_rank(np.asarray(masses), .99), 'max': max(masses)},
                'max_skipped_relative_weight': max(relatives),
                'grid_gap_histogram': hist.tolist(), 'percentile_convention': 'nearest-rank, float32 stored errors',
                'score_convention': 'float64 dot of captured BF16 Q,K,Kq; exact eligible domain',
                'capture_sha256': sha(directory / 'capture.json'),
                'evidence_class': 'kernel sweep',
                'scope': 'real model activations; establishes nothing about model quality by itself'}
    finally:
        del errors
        scratch.unlink(missing_ok=True)


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
