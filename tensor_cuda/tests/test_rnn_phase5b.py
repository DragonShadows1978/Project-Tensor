"""Phase-5b tests: RNN/LSTM/GRU, RoPE, extra losses."""

import numpy as np
import pytest

import tensor_cuda as tc
from tensor_cuda import nn, functional as F


def test_rnn_lstm_gru_shapes():
    x = tc.tensor(np.random.randn(3, 7, 8).astype(np.float32))
    for cls in (nn.RNN, nn.LSTM, nn.GRU):
        out = cls(8, 16)(x)
        assert out.shape == (3, 7, 16), (cls.__name__, out.shape)


def test_lstm_trains():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((6, 5, 4)).astype(np.float32)
    Y = rng.standard_normal((6, 5, 8)).astype(np.float32)
    model = nn.LSTM(4, 8)
    from tensor_cuda import optim
    opt = optim.Adam(model.parameters(), lr=1e-2)
    first = last = None
    for _ in range(40):
        loss = tc.mse_loss(model(tc.tensor(X)), tc.tensor(Y))
        opt.zero_grad(); loss.backward(); opt.step()
        v = float(loss.numpy()); first = v if first is None else first; last = v
    assert last < first, f"{first:.3f} -> {last:.3f}"


def test_losses():
    pred = tc.tensor(np.array([[0.5, -1.0], [2.0, 0.0]], np.float32), requires_grad=True)
    tgt = tc.tensor(np.array([[1.0, 0.0], [1.0, 1.0]], np.float32))
    for loss_fn in (nn.L1Loss(), nn.SmoothL1Loss(), nn.MSELoss()):
        l = loss_fn(pred, tgt)
        assert np.isfinite(float(l.numpy()))
    bce = nn.BCEWithLogitsLoss()(pred, tgt)
    assert np.isfinite(float(bce.numpy()))


def test_rope_shape_and_norm():
    x = tc.tensor(np.random.randn(2, 4, 6, 16).astype(np.float32))
    cos, sin = F.rope_tables(6, 16, "cuda")
    out = F.apply_rotary(x, cos, sin)
    assert out.shape == (2, 4, 6, 16)
    # RoPE is a rotation -> preserves per-vector norm
    a = x.numpy().astype(np.float64); b = out.numpy().astype(np.float64)
    assert np.allclose((a * a).sum(-1), (b * b).sum(-1), atol=1e-2)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
