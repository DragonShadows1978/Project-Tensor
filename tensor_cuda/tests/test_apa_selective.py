"""Validate the fused sparse selective APA kernel against a NumPy reference of
the SAME algorithm (the user's design / the Rust true-selective reference):

  bulk_j = q . k_quant_j                          (quantized, all keys)
  thr    = mean(|bulk|) + z * std(|bulk|)         (threshold from BULK scores)
  score_j = (|bulk_j| >= thr) ? q . k_exact_j     (full precision, selected only)
                              : bulk_j
  out = softmax(score) @ v

The kernel must match this reference. We pass k_quant explicitly (a perturbed
copy of k) so the test is independent of the codebook quantizer.
"""
import numpy as np
import tensor_cuda as tc


def ref_selective(q, k, kq, v, scale, z, causal):
    B, H, L, D = q.shape
    KVH, S = k.shape[1], k.shape[2]
    group = H // KVH                       # GQA: query head h -> KV head h//group
    out = np.zeros((B, H, L, D), np.float32)
    for b in range(B):
        for h in range(H):
            kh = h // group
            for i in range(L):
                # BOTTOM-RIGHT causal: query i is at absolute key index
                # (S-L)+i, sees keys 0..(S-L)+i. Reduces to i+1 at S==L.
                s_max = ((S - L) + i + 1) if causal else S
                qi = q[b, h, i]
                bulk = (kq[b, kh, :s_max] @ qi) * scale         # (s_max,)
                a = np.abs(bulk)
                mean = a.mean()
                std = np.sqrt(max(a.var(), 0.0))
                thr = mean + z * std
                score = np.empty(s_max, np.float32)
                for j in range(s_max):
                    if abs(bulk[j]) >= thr:
                        score[j] = (k[b, kh, j] @ qi) * scale    # exact, selected
                    else:
                        score[j] = bulk[j]
                m = score.max()
                w = np.exp(score - m)
                w /= w.sum()
                out[b, h, i] = w @ v[b, kh, :s_max]
    return out


def ref_blend_softmax_sink(bulk, rank, sinks, zthr):
    B, H, L, S = bulk.shape
    out = np.zeros_like(bulk, dtype=np.float32)
    for b in range(B):
        for h in range(H):
            for i in range(L):
                bk = bulk[b, h, i].astype(np.float32)
                rk = rank[b, h, i].astype(np.float32)
                valid = bk > -5e3
                a = np.abs(bk[valid])
                if a.size:
                    thr = a.mean() + zthr * np.sqrt(max(a.var(), 0.0))
                else:
                    thr = 0.0
                score = np.where(valid & (np.abs(bk) >= thr), rk, bk)
                combined = np.concatenate([score, [float(sinks[h])]]).astype(np.float32)
                w = np.exp(combined - combined.max())
                w /= w.sum()
                out[b, h, i] = w[:-1]
    return out


def _run(causal, L=64, S=None, D=64, H=4, KVH=None):
    rng = np.random.default_rng(0)
    B = 1
    S = L if S is None else S
    KVH = H if KVH is None else KVH
    q = (rng.standard_normal((B, H, L, D)) * 0.1).astype(np.float32)
    k = (rng.standard_normal((B, KVH, S, D)) * 0.1).astype(np.float32)
    v = (rng.standard_normal((B, KVH, S, D)) * 0.1).astype(np.float32)
    kq = (k + rng.standard_normal((B, KVH, S, D)) * 0.01).astype(np.float32)
    scale = 1.0 / np.sqrt(D)
    refine_pct = 0.15
    # z for top-percentile selection (same as engine _norm_ppf(1-pct)).
    from tensor_cuda.quant import _norm_ppf
    z = _norm_ppf(1.0 - refine_pct)

    expected = ref_selective(q, k, kq, v, scale, z, causal)
    got = tc.apa_selective_attention(
        tc.tensor(q), tc.tensor(k), tc.tensor(kq), tc.tensor(v),
        float(scale), float(z), causal).numpy()
    diff = np.abs(got - expected)
    assert got.shape == expected.shape, (got.shape, expected.shape)
    assert diff.max() < 1e-3, f"causal={causal} max diff {diff.max()}"


def test_selective_noncausal():
    _run(False)


def test_selective_causal():
    _run(True)


def test_selective_noncausal_d512():
    _run(False, D=512)          # Gemma 4 global head_dim


def test_selective_causal_d512():
    _run(True, D=512)


def test_selective_rectangular_cache():
    """S>L: queries continue onto a cached prefix (the bottom-right
    causal regime). The old top-left s_max=i+1 SILENTLY blinded queries
    to the most recent S-L keys here — this case is what catches it."""
    _run(True, L=32, S=256, D=64)
    _run(True, L=16, S=512, D=512)    # Gemma-shaped chunked-prefill


def test_selective_mqa_d512():
    """Gemma 4 global: MQA (1 KV head, 16 q heads), head_dim 512,
    rectangular cache."""
    _run(True, L=8, S=300, D=512, H=16, KVH=1)


def test_selective_actually_sparse():
    """Sanity: with a low percentile, most keys keep the BULK (quantized) score,
    i.e. the result differs from full-exact attention but is close to it."""
    rng = np.random.default_rng(1)
    B, H, L, D = 1, 2, 48, 64
    q = (rng.standard_normal((B, H, L, D)) * 0.1).astype(np.float32)
    k = (rng.standard_normal((B, H, L, D)) * 0.1).astype(np.float32)
    v = (rng.standard_normal((B, H, L, D)) * 0.1).astype(np.float32)
    kq = (k + rng.standard_normal((B, H, L, D)) * 0.02).astype(np.float32)
    scale = 1.0 / np.sqrt(D)
    from tensor_cuda.quant import _norm_ppf
    z = _norm_ppf(1.0 - 0.15)
    sel = tc.apa_selective_attention(
        tc.tensor(q), tc.tensor(k), tc.tensor(kq), tc.tensor(v),
        float(scale), float(z), False).numpy()
    # full exact attention for reference
    qd, kd, vd = q[0], k[0], v[0]
    sc = np.einsum('hid,hjd->hij', qd, kd) * scale
    w = np.exp(sc - sc.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True)
    full = np.einsum('hij,hjd->hid', w, vd)[None]
    # selective should be close to full (it refines the most important keys) but
    # not identical (bulk approximates the rest).
    assert np.abs(sel - full).max() < 0.05


def test_apa_blend_softmax_sink_matches_reference_and_omits_sink_column():
    rng = np.random.default_rng(20260706)
    bulk = (rng.standard_normal((2, 3, 4, 11)) * 0.2).astype(np.float32)
    rank = (bulk + rng.standard_normal(bulk.shape) * 0.05).astype(np.float32)
    sinks = rng.standard_normal((3,)).astype(np.float32) * 0.3
    bulk[:, :, :, -2:] = -1.0e4
    rank[:, :, :, -2:] = -1.0e4
    zthr = 0.4

    got = tc.apa_blend_softmax_sink(
        tc.tensor(bulk), tc.tensor(rank), tc.tensor(sinks), zthr
    ).numpy()
    expected = ref_blend_softmax_sink(bulk, rank, sinks, zthr)

    assert got.shape == bulk.shape
    np.testing.assert_allclose(got, expected, rtol=2e-5, atol=2e-5)
    assert np.all(got[:, :, :, -2:] < 1e-6)


def test_apa_blend_softmax_sink_sink_reduces_key_mass():
    bulk = np.zeros((1, 2, 1, 5), dtype=np.float32)
    rank = bulk.copy()
    sinks = np.asarray([-10.0, 4.0], dtype=np.float32)

    got = tc.apa_blend_softmax_sink(
        tc.tensor(bulk), tc.tensor(rank), tc.tensor(sinks), 0.0
    ).numpy()

    assert got[0, 0, 0].sum() > 0.99
    assert got[0, 1, 0].sum() < 0.1


if __name__ == "__main__":
    test_selective_noncausal()
    test_selective_causal()
    test_selective_actually_sparse()
    print("selective APA kernel matches reference: OK")
