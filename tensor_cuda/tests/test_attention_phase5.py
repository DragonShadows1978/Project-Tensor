"""Phase-5 tests: attention + transformer (forward shapes, parity, training)."""

import numpy as np
import pytest

import tensor_cuda as tc
from tensor_cuda import nn, functional as F


def _np(t):
    return t.numpy().astype(np.float64)


def _ref_sdpa(q, k, v, causal=False):
    scale = 1.0 / np.sqrt(q.shape[-1])
    s = (q @ np.swapaxes(k, -1, -2)) * scale
    if causal:
        L, S = q.shape[-2], k.shape[-2]
        s = s + np.triu(np.full((L, S), -1e9, np.float32), 1)
    s = s - s.max(-1, keepdims=True)
    w = np.exp(s); w /= w.sum(-1, keepdims=True)
    return w @ v


def test_sdpa_matches_numpy():
    rng = np.random.default_rng(0)
    q = rng.standard_normal((2, 3, 8, 16)).astype(np.float32)
    k = rng.standard_normal((2, 3, 8, 16)).astype(np.float32)
    v = rng.standard_normal((2, 3, 8, 16)).astype(np.float32)
    out = F.scaled_dot_product_attention(tc.tensor(q), tc.tensor(k), tc.tensor(v))
    assert np.allclose(_np(out), _ref_sdpa(q, k, v), atol=1e-3)


def test_sdpa_causal():
    rng = np.random.default_rng(1)
    q = rng.standard_normal((1, 2, 6, 16)).astype(np.float32)
    k = rng.standard_normal((1, 2, 6, 16)).astype(np.float32)
    v = rng.standard_normal((1, 2, 6, 16)).astype(np.float32)
    out = F.scaled_dot_product_attention(tc.tensor(q), tc.tensor(k), tc.tensor(v), is_causal=True)
    assert np.allclose(_np(out), _ref_sdpa(q, k, v, causal=True), atol=1e-3)


def test_mha_and_encoder_shapes():
    x = tc.tensor(np.random.randn(2, 10, 32).astype(np.float32))
    mha = nn.MultiheadAttention(32, 4)
    assert mha(x).shape == (2, 10, 32)
    layer = nn.TransformerEncoderLayer(32, 4, 64)
    assert layer(x).shape == (2, 10, 32)


def test_rmsnorm():
    x = tc.tensor((np.random.randn(4, 16) * 3).astype(np.float32))
    out = _np(nn.RMSNorm(16)(x))
    assert np.all(np.isfinite(out))


def test_transformer_trains():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((8, 10, 32)).astype(np.float32)
    target = rng.standard_normal((8, 10, 32)).astype(np.float32)
    layer = nn.TransformerEncoderLayer(32, 4, 64)
    from tensor_cuda import optim
    opt = optim.Adam(layer.parameters(), lr=1e-3)
    first = last = None
    for _ in range(30):
        out = layer(tc.tensor(X))
        loss = tc.mse_loss(out, tc.tensor(target))
        opt.zero_grad(); loss.backward(); opt.step()
        v = float(loss.numpy()); first = v if first is None else first; last = v
    assert last < first, f"transformer did not improve: {first:.3f} -> {last:.3f}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
