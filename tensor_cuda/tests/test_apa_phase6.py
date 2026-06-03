"""Phase-6 tests: APA-Quant attention on the standalone engine.

Forward is checked against a NumPy reference built from the *same* codebook /
rotations (z-score refinement path). Backward is exact autograd of the forward
(not the reference's approximation), so we only assert it runs and is finite.
"""

import numpy as np
import pytest

import tensor_cuda as tc
from tensor_cuda import quant


def _np(t):
    return t.numpy().astype(np.float64)


def _ref(q, k, v, bits, pct, causal):
    B, H, S, D = k.shape
    scale = 1.0 / np.sqrt(D)
    cb = quant.build_codebook(D, bits)
    bnd = (0.5 * (cb[:-1] + cb[1:])).astype(np.float32)
    rot = np.stack([quant._rotation(D, h * 1337 + 42) for h in range(H)])

    norms = np.sqrt((k * k).sum(-1, keepdims=True))
    unit = k / (norms + 1e-12)
    rotated = np.einsum("bhsd,hed->bhse", unit, rot)
    idx = np.searchsorted(bnd, rotated.ravel(), side="left").reshape(rotated.shape)
    centroids = cb[idx]
    recon = np.einsum("bhse,hed->bhsd", centroids, rot) * norms

    bulk = np.einsum("bhld,bhsd->bhls", q, recon) * scale
    ranking = np.einsum("bhld,bhsd->bhls", q, k) * scale
    if causal:
        cm = np.triu(np.full((q.shape[2], S), -1e9, np.float32), 1)
        bulk = bulk + cm; ranking = ranking + cm
    if pct >= 1.0:
        scores = ranking
    else:
        absr = np.abs(ranking)
        if causal:
            absr = np.where(np.triu(np.ones((q.shape[2], S), bool), 1), 0.0, absr)
        from tensor_cuda.quant import _norm_ppf
        z = _norm_ppf(1.0 - pct)
        thr = absr.mean(-1, keepdims=True) + z * absr.std(-1, keepdims=True)
        mask = absr >= thr
        scores = np.where(mask, ranking, bulk)
    scores = scores - scores.max(-1, keepdims=True)
    w = np.exp(scores); w /= w.sum(-1, keepdims=True)
    return np.einsum("bhls,bhsd->bhld", w, v)


def _case(B, H, L, S, D, bits=2, pct=0.15, causal=False, tol=2e-2, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((B, H, L, D)).astype(np.float32)
    k = rng.standard_normal((B, H, S, D)).astype(np.float32)
    v = rng.standard_normal((B, H, S, D)).astype(np.float32)
    ref = _ref(q, k, v, bits, pct, causal)
    out = tc.apa_quant_attention(tc.tensor(q), tc.tensor(k), tc.tensor(v),
                                 bulk_bits=bits, refine_percentile=pct, is_causal=causal)
    err = np.abs(_np(out) - ref).max()
    assert err < tol, f"APA forward err {err:.2e} (B{B}H{H}L{L}S{S}D{D} bits{bits} causal{causal})"


def test_apa_forward():
    _case(2, 4, 64, 64, 32)


def test_apa_causal():
    _case(1, 2, 48, 48, 32, causal=True)


def test_apa_bits():
    for bits in (2, 4):
        _case(1, 2, 64, 64, 32, bits=bits)


def test_apa_full_precision():
    _case(2, 2, 40, 40, 32, pct=1.0, tol=1e-4)


def test_apa_backward_finite():
    rng = np.random.default_rng(3)
    q = tc.tensor(rng.standard_normal((2, 2, 32, 16)).astype(np.float32), requires_grad=True)
    k = tc.tensor(rng.standard_normal((2, 2, 32, 16)).astype(np.float32), requires_grad=True)
    v = tc.tensor(rng.standard_normal((2, 2, 32, 16)).astype(np.float32), requires_grad=True)
    out = tc.apa_quant_attention(q, k, v, bulk_bits=2, refine_percentile=0.2)
    out.sum().backward()
    for g in (q.grad, k.grad, v.grad):
        assert g is not None and np.all(np.isfinite(g.numpy()))


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
