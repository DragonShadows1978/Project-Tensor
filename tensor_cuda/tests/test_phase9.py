"""Phase-9 tests: einsum, topk, depthwise/separable conv, extra losses."""

import numpy as np
import pytest

import tensor_cuda as tc
from tensor_cuda import nn


def _np(t):
    return t.numpy().astype(np.float64)


def test_einsum_matches_numpy():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((2, 3, 4)).astype(np.float32)
    b = rng.standard_normal((2, 4, 5)).astype(np.float32)
    out = tc.einsum("bij,bjk->bik", tc.tensor(a), tc.tensor(b))
    assert np.allclose(_np(out), np.einsum("bij,bjk->bik", a, b), atol=1e-3)

    q = rng.standard_normal((2, 3, 6, 8)).astype(np.float32)
    k = rng.standard_normal((2, 3, 6, 8)).astype(np.float32)
    out2 = tc.einsum("bhld,bhsd->bhls", tc.tensor(q), tc.tensor(k))
    assert np.allclose(_np(out2), np.einsum("bhld,bhsd->bhls", q, k), atol=1e-3)


def test_einsum_grad():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((3, 4)).astype(np.float32)
    a = tc.tensor(A, requires_grad=True)
    b = tc.tensor(rng.standard_normal((4, 5)).astype(np.float32))
    out = tc.einsum("ij,jk->ik", a, b)
    out.sum().backward()
    assert np.all(np.isfinite(a.grad.numpy()))


def test_topk():
    x = tc.tensor(np.array([[1.0, 5.0, 3.0, 2.0, 4.0]], np.float32))
    vals, idx = x.topk(3, True)
    assert np.allclose(_np(vals), [[5, 4, 3]])
    assert np.allclose(idx.numpy().ravel(), [1, 4, 2])


def test_depthwise_separable_conv():
    x = tc.tensor(np.random.randn(2, 4, 8, 8).astype(np.float32))
    assert nn.DepthwiseConv2D(4, 3, padding=1)(x).shape == (2, 4, 8, 8)
    assert nn.SeparableConv2D(4, 6, 3, padding=1)(x).shape == (2, 6, 8, 8)


def test_cosine_triplet_losses():
    rng = np.random.default_rng(2)
    a = tc.tensor(rng.standard_normal((5, 8)).astype(np.float32))
    b = tc.tensor(rng.standard_normal((5, 8)).astype(np.float32))
    y = tc.tensor(np.array([1, -1, 1, -1, 1], np.float32))
    assert np.isfinite(float(nn.CosineEmbeddingLoss()(a, b, y).numpy()))
    n = tc.tensor(rng.standard_normal((5, 8)).astype(np.float32))
    assert np.isfinite(float(nn.TripletMarginLoss()(a, b, n).numpy()))


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
