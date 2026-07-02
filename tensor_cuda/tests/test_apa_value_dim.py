import numpy as np
import pytest

import tensor_cuda as tc


def _ref_selective(q, k, kq, v, scale, zthr, causal):
    B, H, L, D = q.shape
    KVH, S, VD = k.shape[1], k.shape[2], v.shape[3]
    group = H // KVH
    out = np.zeros((B, H, L, VD), np.float32)
    for b in range(B):
        for h in range(H):
            kh = h // group
            for i in range(L):
                s_max = ((S - L) + i + 1) if causal else S
                qi = q[b, h, i]
                bulk = (kq[b, kh, :s_max] @ qi) * scale
                a = np.abs(bulk)
                thr = a.mean() + zthr * np.sqrt(max(a.var(), 0.0))
                score = np.empty(s_max, np.float32)
                for j in range(s_max):
                    score[j] = ((k[b, kh, j] @ qi) * scale
                                if abs(bulk[j]) >= thr else bulk[j])
                w = np.exp(score - score.max())
                w /= w.sum()
                out[b, h, i] = w @ v[b, kh, :s_max]
    return out


def _case():
    rng = np.random.default_rng(7)
    B, H, KVH, L, S, D, VD = 1, 2, 2, 5, 5, 6, 3
    q = (rng.standard_normal((B, H, L, D)) * 0.1).astype(np.float32)
    k = (rng.standard_normal((B, KVH, S, D)) * 0.1).astype(np.float32)
    kq = (k + rng.standard_normal((B, KVH, S, D)) * 0.01).astype(np.float32)
    v = (rng.standard_normal((B, KVH, S, VD)) * 0.1).astype(np.float32)
    scale = 1.0 / np.sqrt(D)
    zthr = 0.0
    return q, k, kq, v, scale, zthr


def test_selective_inference_allows_value_dim_different_from_score_dim():
    q, k, kq, v, scale, zthr = _case()
    got = tc.apa_selective_attention(
        tc.tensor(q), tc.tensor(k), tc.tensor(kq), tc.tensor(v),
        float(scale), float(zthr), True).numpy()
    expected = _ref_selective(q, k, kq, v, scale, zthr, True)
    assert got.shape == expected.shape
    assert np.abs(got - expected).max() < 1e-6


def test_selective_train_allows_value_dim_different_from_score_dim():
    q, k, kq, v, scale, zthr = _case()
    qt = tc.tensor(q, requires_grad=True)
    kt = tc.tensor(k, requires_grad=True)
    vt = tc.tensor(v, requires_grad=True)
    out = tc.apa_selective_train(qt, kt, tc.tensor(kq), vt,
                                 float(scale), float(zthr), True)
    expected = _ref_selective(q, k, kq, v, scale, zthr, True)
    assert out.shape == expected.shape
    assert np.abs(out.numpy() - expected).max() < 1e-6
    out.sum().backward()
    assert qt.grad.shape == q.shape
    assert kt.grad.shape == k.shape
    assert vt.grad.shape == v.shape
    assert np.all(np.isfinite(qt.grad.numpy()))
    assert np.all(np.isfinite(kt.grad.numpy()))
    assert np.all(np.isfinite(vt.grad.numpy()))


def test_reshape_rejects_element_count_mismatch():
    x = tc.zeros(2, 3)
    with pytest.raises(RuntimeError, match="element count mismatch"):
        x.reshape([5])
