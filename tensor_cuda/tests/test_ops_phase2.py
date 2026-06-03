"""Phase-2 tests: broadened op surface (forward + autograd via finite diff)."""

import numpy as np
import pytest

import tensor_cuda as tc


def _np(t):
    return t.numpy().astype(np.float64)


def test_pow_compare_where():
    a = tc.tensor([[1.0, -2.0], [3.0, -4.0]])
    assert np.allclose(_np(a.pow(2.0)), [[1, 4], [9, 16]])
    mask = a.__gt__(0.0)
    assert np.allclose(_np(mask), [[1, 0], [1, 0]])
    out = tc.where(mask, a, tc.tensor([[9.0, 9.0], [9.0, 9.0]]))
    assert np.allclose(_np(out), [[1, 9], [3, 9]])


def test_reductions():
    x = np.random.randn(4, 5).astype(np.float32)
    t = tc.tensor(x)
    assert np.allclose(_np(t.max([1])), x.max(1), atol=1e-5)
    assert np.allclose(_np(t.min([0])), x.min(0), atol=1e-5)
    assert np.allclose(_np(t.var([1])), x.var(1), atol=1e-4)
    assert np.allclose(_np(t.std([1])), x.std(1), atol=1e-4)


def test_shape_ops():
    x = np.random.randn(2, 3, 4).astype(np.float32)
    t = tc.tensor(x)
    assert _np(t.permute([2, 0, 1])).shape == (4, 2, 3)
    assert np.allclose(_np(t.transpose(0, 1)), x.transpose(1, 0, 2), atol=1e-6)
    assert _np(t.unsqueeze(0)).shape == (1, 2, 3, 4)
    assert _np(t.flatten(1, 2)).shape == (2, 12)


def test_cat_stack():
    a = tc.tensor(np.ones((2, 3), np.float32))
    b = tc.tensor(np.zeros((2, 3), np.float32))
    assert _np(tc.cat([a, b], 0)).shape == (4, 3)
    assert _np(tc.stack([a, b], 0)).shape == (2, 2, 3)
    assert np.allclose(_np(tc.cat([a, b], 1))[:, :3], 1.0)


def test_log_softmax():
    x = np.random.randn(3, 6).astype(np.float32)
    out = _np(tc.tensor(x).log_softmax(-1))
    ref = x - x.max(-1, keepdims=True)
    ref = ref - np.log(np.exp(ref).sum(-1, keepdims=True))
    assert np.allclose(out, ref, atol=1e-4)


def test_cross_entropy_grad():
    rng = np.random.default_rng(0)
    logits0 = rng.standard_normal((5, 4)).astype(np.float32)
    labels = np.array([0, 1, 2, 3, 0])

    W = tc.tensor(logits0, requires_grad=True)
    loss = tc.cross_entropy(W, labels)
    loss.backward()
    g = W.grad.numpy().astype(np.float64)

    def f(L):
        z = L - L.max(-1, keepdims=True)
        lse = z - np.log(np.exp(z).sum(-1, keepdims=True))
        oh = np.zeros_like(L); oh[np.arange(5), labels] = 1
        return float(-(oh * lse).sum(-1).mean())

    eps = 1e-3
    gfd = np.zeros_like(logits0, dtype=np.float64)
    L = logits0.astype(np.float64)
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            o = L[i, j]
            L[i, j] = o + eps; fp = f(L)
            L[i, j] = o - eps; fm = f(L)
            L[i, j] = o
            gfd[i, j] = (fp - fm) / (2 * eps)
    assert np.allclose(g, gfd, atol=1e-2), f"max err {np.abs(g - gfd).max():.2e}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
