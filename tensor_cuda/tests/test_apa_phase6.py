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
        # APA-MF2.1 (review F9): BOTTOM-RIGHT like the engine and the statistics below
        # (row i of L sees j <= S-L+i); was triu(k=1), top-left on S > L.
        # APA-MF2.2 Amendment 3 (review V5): a -inf PREDICATE like the engine (was a finite -1e9
        # added bias: a visible score below -1e9 lost to the masked keys -- Q = 8e0, K0 = -2e9e0 gave
        # [0, 0, 0] instead of [1, 0, 0]).
        vis_c = np.arange(S)[None, :] <= (S - q.shape[2] + np.arange(q.shape[2]))[:, None]
        bulk = np.where(vis_c, bulk, -np.inf); ranking = np.where(vis_c, ranking, -np.inf)
    if pct >= 1.0:
        scores = ranking
    else:
        # Selection is on |bulk| (the cheap quantized scores), matching the fused
        # apa_selective_kernel: APA picks which keys to refine using only the
        # signal it already has, never the expensive |ranking| it is deciding
        # whether to compute. (kernels.cu apa_selective_kernel.)
        # The selection DEFINITION (APA-MF2.1; Amendment 3 review V5): population moments over
        # the VISIBLE keys only (row i of L sees j <= S-L+i, count S-L+i+1; all S when
        # non-causal), CENTERED two-pass in fp64 (was one-pass E[a^2] - E[a]^2, which cancels on
        # near-ties), th = mean + z*sqrt(var), sel = visible && |bulk| >= th. Prior art: centered
        # two-pass variance, Chan, Golub & LeVeque (1983); FlashAttention-2 bottom-right (Dao 2023).
        Lq = q.shape[2]
        vis = (np.arange(S)[None, :] <= (S - Lq + np.arange(Lq))[:, None]) if causal \
            else np.ones((Lq, S), bool)
        absr = np.where(vis, np.abs(bulk).astype(np.float64), 0.0)
        from tensor_cuda.quant import _norm_ppf
        z = _norm_ppf(1.0 - pct)
        cnt = vis.sum(-1, keepdims=True)
        mean = absr.sum(-1, keepdims=True) / cnt
        dev = np.where(vis, absr - mean, 0.0)
        var = (dev * dev).sum(-1, keepdims=True) / cnt
        thr = mean + z * np.sqrt(var)
        mask = vis & (absr >= thr)
        scores = np.where(mask, ranking, bulk)
    # A row whose every score is -inf gets weight 0 (the lead's policy, Amendment 2).
    blocked = np.all(np.isneginf(scores), -1, keepdims=True)
    scores = np.where(blocked, 0.0, scores)
    scores = scores - scores.max(-1, keepdims=True)
    w = np.exp(scores); w /= w.sum(-1, keepdims=True); w = np.where(blocked, 0.0, w)
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
