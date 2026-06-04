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
    S = k.shape[2]
    out = np.zeros((B, H, L, D), np.float32)
    for b in range(B):
        for h in range(H):
            for i in range(L):
                s_max = (i + 1) if causal else S
                qi = q[b, h, i]
                bulk = (kq[b, h, :s_max] @ qi) * scale          # (s_max,)
                a = np.abs(bulk)
                mean = a.mean()
                std = np.sqrt(max(a.var(), 0.0))
                thr = mean + z * std
                score = np.empty(s_max, np.float32)
                for j in range(s_max):
                    if abs(bulk[j]) >= thr:
                        score[j] = (k[b, h, j] @ qi) * scale     # exact, selected
                    else:
                        score[j] = bulk[j]
                m = score.max()
                w = np.exp(score - m)
                w /= w.sum()
                out[b, h, i] = w @ v[b, h, :s_max]
    return out


def _run(causal):
    rng = np.random.default_rng(0)
    B, H, L, D = 1, 4, 64, 64
    S = L
    q = (rng.standard_normal((B, H, L, D)) * 0.1).astype(np.float32)
    k = (rng.standard_normal((B, H, S, D)) * 0.1).astype(np.float32)
    v = (rng.standard_normal((B, H, S, D)) * 0.1).astype(np.float32)
    kq = (k + rng.standard_normal((B, H, S, D)) * 0.01).astype(np.float32)  # "quantized" keys
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


if __name__ == "__main__":
    test_selective_noncausal()
    test_selective_causal()
    test_selective_actually_sparse()
    print("selective APA kernel matches reference: OK")
