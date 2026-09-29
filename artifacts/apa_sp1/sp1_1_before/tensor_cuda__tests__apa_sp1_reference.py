"""APA-SP1 independent NumPy specifications; no CUDA imports.

Finite, representable fp32 scores are the numerical domain. Mathematical
single-pass equivalence does not promise bit equivalence to CUDA FMA/expf.
The original selector uses population E[a*a]-E[a]**2, not a sample variance.
"""
from __future__ import annotations

import ast
from pathlib import Path
import numpy as np

F = np.float32
ROOT = Path(__file__).resolve().parents[2]


def zmask(bulk, z, lengths=None):
    bulk = np.asarray(bulk, dtype=F)
    valid = np.ones(bulk.shape, bool) if lengths is None else (
        np.arange(bulk.shape[-1]) < np.asarray(lengths)[..., None])
    a = np.where(valid, np.abs(bulk), F(0))
    n = valid.sum(-1).astype(F)
    mean = a.sum(-1, dtype=F) / n
    var = (a * a).sum(-1, dtype=F) / n - mean * mean
    thr = mean + F(z) * np.sqrt(np.maximum(var, F(0)))
    return valid & (np.abs(bulk) >= thr[..., None]), thr


def prefix_mask(bulk, delta, lengths=None):
    bulk = np.asarray(bulk, dtype=F)
    mask = bulk >= np.maximum.accumulate(bulk, axis=-1) - F(delta)
    if lengths is not None:
        mask &= np.arange(bulk.shape[-1]) < np.asarray(lengths)[..., None]
    return mask


def dense_scores(scores, values, lengths=None, sinks=None):
    """Independent materialized softmax, batched or single row, fp32."""
    scores, values = np.asarray(scores, dtype=F), np.asarray(values, dtype=F)
    if lengths is not None:
        scores = np.where(np.arange(scores.shape[-1]) < np.asarray(lengths)[..., None],
                          scores, F(-np.inf))
    maximum = scores.max(-1)
    if sinks is not None:
        maximum = np.maximum(maximum, np.asarray(sinks, dtype=F))
    w = np.exp(scores - maximum[..., None])
    den = w.sum(-1, dtype=F)
    if sinks is not None:
        den += np.exp(np.asarray(sinks, dtype=F) - maximum)
    return np.einsum('...s,...sv->...v', w, values, optimize=False) / den[..., None]


def online_batch(bulk, exact, values, delta, lengths=None, sinks=None):
    """Explicit ascending-key selection + online-softmax, independent of scan.

    Dots are passed in to allow vectorized q/K generation in the random sweep;
    row_single_visit below also checks computing them at the visit itself.
    """
    N, S = bulk.shape
    lengths = np.full(N, S) if lengths is None else np.asarray(lengths)
    mx = np.full(N, -np.inf, dtype=F)
    m = np.full(N, -np.inf, dtype=F)
    den = np.zeros(N, dtype=F)
    acc = np.zeros((N, values.shape[-1]), dtype=F)
    selected = np.zeros((N, S), dtype=bool)
    for j in range(S):
        valid = j < lengths
        mx = np.maximum(mx, bulk[:, j])
        take = (bulk[:, j] >= mx - F(delta)) & valid
        selected[:, j] = take
        score = np.where(take, exact[:, j], bulk[:, j])
        nxt = np.maximum(m, score)
        corr = np.exp(m - nxt)
        w = np.exp(score - nxt)
        den = np.where(valid, den * corr + w, den)
        acc = np.where(valid[:, None], acc * corr[:, None] + w[:, None] * values[:, j], acc)
        m = np.where(valid, nxt, m)
    if sinks is not None:
        sinks = np.asarray(sinks, dtype=F)
        nxt = np.maximum(m, sinks)
        corr = np.exp(m - nxt)
        den = den * corr + np.exp(sinks - nxt)
        acc *= corr[:, None]
    return acc / den[:, None], selected


def row_single_visit(q, k, kq, v, scale, delta, sink=None):
    """Literal one visit to each Kq row, selected K row and V row; O(D+VD).

    The returned mask is optional *reference instrumentation*, not kernel
    workspace. This path never forms an exact dot for an unselected key.
    """
    m, mx, den = F(-np.inf), F(-np.inf), F(0)
    acc = np.zeros(v.shape[-1], dtype=F)
    mask = []
    for kj, kqj, vj in zip(k, kq, v):
        bulk = F(np.dot(q, kqj) * F(scale))
        mx = np.maximum(mx, bulk)
        take = bool(bulk >= mx - F(delta))
        score = F(np.dot(q, kj) * F(scale)) if take else bulk
        nxt = np.maximum(m, score)
        corr, w = np.exp(m - nxt), np.exp(score - nxt)
        den = F(den * corr + w)
        acc = acc * corr + w * vj
        m = nxt
        mask.append(take)
    if sink is not None:
        nxt = np.maximum(m, F(sink))
        corr = np.exp(m - nxt)
        den = den * corr + np.exp(F(sink) - nxt)
        acc *= corr
    return acc / den, np.asarray(mask)


def row_two_pass(q, k, kq, v, scale, z, sink=None):
    bulk = (kq @ q) * F(scale)
    mask, thr = zmask(bulk, z)
    # Recompute BULK in pass two, exact dot only on selected rows.
    score = (kq @ q) * F(scale)
    score[mask] = (k[mask] @ q) * F(scale)
    return dense_scores(score, v, sinks=sink), mask, thr


def row_buffered_one_read(q, k, kq, v, scale, z, sink=None):
    """Q1 loophole, NOT the proposed kernel: all-exact, O(S*(VD+2)) buffer."""
    records = [(F(np.dot(q, b) * F(scale)), F(np.dot(q, e) * F(scale)), x.copy())
               for b, e, x in zip(kq, k, v)]
    bulk = np.asarray([x[0] for x in records], dtype=F)
    exact = np.asarray([x[1] for x in records], dtype=F)
    values = np.asarray([x[2] for x in records], dtype=F)
    mask, _ = zmask(bulk, z)
    return dense_scores(np.where(mask, exact, bulk), values, sinks=sink), mask


def tensor_reference(q, k, kq, v, scale, threshold, causal=False, sinks=None,
                     rule='zscore', chunk=32):
    """Full tensor fp32 reference, bounded host scratch; no sampled rows.

    Q2 is the independent offline prefix-scan specification. The explicit
    online emulator is cross-checked separately, including every random draw.
    """
    B, H, L, D = q.shape
    KVH, S, VD = k.shape[1], k.shape[2], v.shape[3]
    out = np.empty((B, H, L, VD), dtype=F)
    for b in range(B):
        for h in range(H):
            kh = h // (H // KVH)
            for start in range(0, L, chunk):
                end = min(L, start + chunk)
                qr = q[b, h, start:end]
                lengths = S - L + np.arange(start, end) + 1 if causal else np.full(end-start, S)
                exact = (qr @ k[b, kh].T) * F(scale)
                bulk = (qr @ kq[b, kh].T) * F(scale)
                if rule == 'zscore':
                    mask, _ = zmask(bulk, threshold, lengths)
                elif rule == 'prefix':
                    mask = prefix_mask(bulk, threshold, lengths)
                elif rule == 'dense':
                    mask = np.ones_like(bulk, dtype=bool)
                else:
                    raise ValueError(rule)
                vv = np.broadcast_to(v[b, kh], (end-start, S, VD))
                sink = None if sinks is None else np.full(end-start, sinks[h], dtype=F)
                out[b, h, start:end] = dense_scores(np.where(mask, exact, bulk), vv, lengths, sink)
    return out


def original_numpy_reference(name='ref_selective'):
    """Load only the immutable pure function, without importing CUDA tests."""
    path = Path(__file__).with_name('test_apa_selective.py')
    tree = ast.parse(path.read_text())
    fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name)
    scope = {'np': np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), scope)
    return scope[name]


def counterexample(z):
    q = np.array([1], dtype=F)
    k = np.array([[2], [0], [0], [-10]], dtype=F)
    v = np.array([[1], [0], [0], [0]], dtype=F)
    records = []
    for last in [0, 10]:
        kq = np.array([[1], [0], [0], [last]], dtype=F)
        out, mask, thr = row_two_pass(q, k, kq, v, 1, z)
        buffered, bm = row_buffered_one_read(q, k, kq, v, 1, z)
        np.testing.assert_array_equal(mask, bm)
        np.testing.assert_allclose(out, buffered, atol=1e-3, rtol=1e-3)
        records.append(dict(last=last, threshold=float(thr), mask=mask.tolist(), output=out.tolist()))
    assert records[0]['mask'][0] and not records[1]['mask'][0]
    return records
