"""Phase-3b tests: Conv2D / pooling / BatchNorm2D (forward parity + training)."""

import numpy as np
import pytest

import tensor_cuda as tc
from tensor_cuda import nn


def _np(t):
    return t.numpy().astype(np.float64)


def _ref_conv(x, w, b, stride, pad):
    N, C, H, W = x.shape
    OC, _, kh, kw = w.shape
    OH = (H + 2 * pad - kh) // stride + 1
    OW = (W + 2 * pad - kw) // stride + 1
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    out = np.zeros((N, OC, OH, OW), np.float64)
    for oh in range(OH):
        for ow in range(OW):
            patch = xp[:, :, oh*stride:oh*stride+kh, ow*stride:ow*stride+kw]
            out[:, :, oh, ow] = np.tensordot(patch, w, axes=([1, 2, 3], [1, 2, 3]))
    return out + b.reshape(1, -1, 1, 1)


def test_conv2d_forward():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3, 8, 8)).astype(np.float32)
    conv = nn.Conv2D(3, 4, 3, stride=1, padding=1)
    w = conv.weight.numpy(); b = conv.bias.numpy()
    out = _np(conv(tc.tensor(x)))
    assert out.shape == (2, 4, 8, 8)
    assert np.allclose(out, _ref_conv(x, w.astype(np.float64), b.astype(np.float64), 1, 1), atol=1e-2)


def test_maxpool_avgpool():
    x = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    mp = _np(nn.MaxPool2D(2)(tc.tensor(x)))
    assert np.allclose(mp, [[[[5, 7], [13, 15]]]])
    ap = _np(nn.AvgPool2D(2)(tc.tensor(x)))
    assert np.allclose(ap, [[[[2.5, 4.5], [10.5, 12.5]]]])


def test_conv_pool_backward_finite():
    x = tc.tensor(np.random.randn(2, 3, 8, 8).astype(np.float32), requires_grad=True)
    conv = nn.Conv2D(3, 4, 3, padding=1)
    out = nn.MaxPool2D(2)(conv(x))
    out.sum().backward()
    assert np.all(np.isfinite(conv.weight.grad.numpy()))
    assert np.all(np.isfinite(x.grad.numpy()))


def test_tiny_cnn_trains():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((8, 1, 8, 8)).astype(np.float32)
    Y = rng.standard_normal((8, 2, 8, 8)).astype(np.float32)
    conv = nn.Conv2D(1, 2, 3, padding=1)
    from tensor_cuda import optim
    opt = optim.Adam(conv.parameters(), lr=1e-2)
    first = last = None
    for _ in range(40):
        loss = tc.mse_loss(conv(tc.tensor(X)), tc.tensor(Y))
        opt.zero_grad(); loss.backward(); opt.step()
        v = float(loss.numpy()); first = v if first is None else first; last = v
    assert last < first


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
