"""Phase-1 engine tests: numerical correctness of forward and autograd.

Gradients are checked against central finite differences (the engine's grads are
exact, so gradcheck applies here, unlike the APA approximation). Run after build:

    PYTHONPATH=. python -m pytest tests/test_engine.py
"""

import numpy as np
import pytest

import tensor_cuda as tc


def _np(t):
    return t.numpy().astype(np.float64)


def test_forward_arithmetic():
    a = tc.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = tc.tensor([[10.0, 20.0], [30.0, 40.0]])
    assert np.allclose(_np(a + b), [[11, 22], [33, 44]])
    assert np.allclose(_np(a * b), [[10, 40], [90, 160]])
    assert np.allclose(_np(a * 2.0), [[2, 4], [6, 8]])


def test_broadcasting():
    a = tc.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])   # (2,3)
    b = tc.tensor([10.0, 20.0, 30.0])                    # (3,)
    assert np.allclose(_np(a + b), [[11, 22, 33], [14, 25, 36]])


def test_matmul_forward():
    A = np.random.randn(4, 5).astype(np.float32)
    B = np.random.randn(5, 3).astype(np.float32)
    out = tc.matmul(tc.tensor(A), tc.tensor(B))
    assert np.allclose(_np(out), A @ B, atol=1e-4)


def test_batched_matmul():
    A = np.random.randn(2, 4, 5).astype(np.float32)
    B = np.random.randn(2, 5, 3).astype(np.float32)
    out = tc.matmul(tc.tensor(A), tc.tensor(B))
    assert np.allclose(_np(out), A @ B, atol=1e-4)


def test_softmax():
    x = np.random.randn(3, 7).astype(np.float32)
    out = _np(tc.tensor(x).softmax(-1))
    ex = np.exp(x - x.max(-1, keepdims=True))
    assert np.allclose(out, ex / ex.sum(-1, keepdims=True), atol=1e-5)


def _grad_fd(f, x, eps=1e-3):
    """Central finite-difference gradient of scalar f wrt numpy array x."""
    g = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        orig = x[i]
        x[i] = orig + eps; fp = f(x)
        x[i] = orig - eps; fm = f(x)
        x[i] = orig
        g[i] = (fp - fm) / (2 * eps)
        it.iternext()
    return g


def test_autograd_matmul_mse():
    rng = np.random.default_rng(0)
    W0 = rng.standard_normal((4, 3)).astype(np.float32)
    X = rng.standard_normal((6, 4)).astype(np.float32)
    Y = rng.standard_normal((6, 3)).astype(np.float32)

    def loss_of(W):
        w = tc.tensor(W, requires_grad=True)
        x = tc.tensor(X)
        y = tc.tensor(Y)
        pred = tc.matmul(x, w).gelu()
        return tc.mse_loss(pred, y)

    loss = loss_of(W0)
    loss.backward()
    g_analytic = loss.grad  # not a leaf path; check W grad instead

    w = tc.tensor(W0, requires_grad=True)
    pred = tc.matmul(tc.tensor(X), w).gelu()
    L = tc.mse_loss(pred, tc.tensor(Y))
    L.backward()
    g = w.grad.numpy().astype(np.float64)

    def scalar_loss(Wnp):
        wv = tc.tensor(Wnp.astype(np.float32))
        p = tc.matmul(tc.tensor(X), wv).gelu()
        return float(tc.mse_loss(p, tc.tensor(Y)).numpy())

    g_fd = _grad_fd(scalar_loss, W0.astype(np.float64))
    assert np.allclose(g, g_fd, atol=1e-2), f"max err {np.abs(g - g_fd).max():.2e}"


def test_train_linear_regression():
    """A real training loop should drive the loss down."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((64, 8)).astype(np.float32)
    true_W = rng.standard_normal((8, 1)).astype(np.float32)
    Y = X @ true_W

    W = tc.tensor(rng.standard_normal((8, 1)) * 0.01, requires_grad=True)
    lr = 0.1
    first = last = None
    for step in range(50):
        pred = tc.matmul(tc.tensor(X), W)
        loss = tc.mse_loss(pred, tc.tensor(Y))
        W.zero_grad()
        loss.backward()
        gW = W.grad.numpy()
        W = tc.tensor(W.numpy() - lr * gW, requires_grad=True)  # SGD step
        val = float(loss.numpy())
        if step == 0:
            first = val
        last = val
    assert last < first * 0.1, f"loss did not converge: {first:.3f} -> {last:.3f}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
