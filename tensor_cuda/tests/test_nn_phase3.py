"""Phase-3 tests: nn.Module layers + optimizers train end-to-end."""

import numpy as np
import pytest

import tensor_cuda as tc
from tensor_cuda import nn, optim


def test_linear_shapes_and_params():
    lin = nn.Linear(8, 4)
    assert len(lin.parameters()) == 2
    x = tc.tensor(np.random.randn(5, 8).astype(np.float32))
    assert lin(x).shape == (5, 4)


def test_layernorm_normalizes():
    ln = nn.LayerNorm(16)
    x = tc.tensor((np.random.randn(3, 16) * 5 + 2).astype(np.float32))
    out = ln(x).numpy().astype(np.float64)
    assert np.allclose(out.mean(-1), 0, atol=1e-3)
    assert np.allclose(out.std(-1), 1, atol=1e-2)


def test_embedding_lookup_and_grad():
    emb = nn.Embedding(10, 4)
    idx = tc.tensor(np.array([1, 3, 3], dtype=np.int64), dtype="int64")
    out = emb(idx)
    assert out.shape == (3, 4)
    out.sum().backward()
    g = emb.weight.grad.numpy()
    assert np.allclose(g[3], 2.0, atol=1e-4)   # index 3 used twice
    assert np.allclose(g[1], 1.0, atol=1e-4)
    assert np.allclose(g[0], 0.0)


def test_train_mlp_classification():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((128, 16)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)

    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 2))
    opt = optim.Adam(model.parameters(), lr=1e-2)

    first = last = None
    for step in range(60):
        logits = model(tc.tensor(X))
        loss = tc.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        v = float(loss.numpy())
        first = v if first is None else first
        last = v
    assert last < first * 0.6, f"MLP did not learn: {first:.3f} -> {last:.3f}"


def test_sgd_step_decreases_loss():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((64, 4)).astype(np.float32)
    W_true = rng.standard_normal((4, 1)).astype(np.float32)
    Y = X @ W_true

    lin = nn.Linear(4, 1, bias=False)
    opt = optim.SGD(lin.parameters(), lr=0.1)
    first = last = None
    for _ in range(80):
        loss = tc.mse_loss(lin(tc.tensor(X)), tc.tensor(Y))
        opt.zero_grad(); loss.backward(); opt.step()
        v = float(loss.numpy()); first = v if first is None else first; last = v
    assert last < first * 0.1


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
