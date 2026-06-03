"""Phase-7/8 tests: extended ops, norms, conv1d, decoder, tying, scheduler family."""

import numpy as np
import pytest

import tensor_cuda as tc
from tensor_cuda import nn, optim, functional as F


def _np(t):
    return t.numpy().astype(np.float64)


def test_math_ops():
    x = tc.tensor(np.array([0.5, 1.0, 2.0], np.float32))
    assert np.allclose(_np(x.log2()), np.log2([0.5, 1, 2]), atol=1e-4)
    assert np.allclose(_np(x.tan()), np.tan([0.5, 1, 2]), atol=1e-3)
    assert np.allclose(_np(x.floor()), [0, 1, 2])


def test_prod_cumsum_flip_gather():
    x = tc.tensor(np.array([[1.0, 2.0, 3.0]], np.float32))
    assert np.allclose(_np(x.prod([1])), [6.0])
    assert np.allclose(_np(x.cumsum(1)), [[1, 3, 6]])
    assert np.allclose(_np(x.flip([1])), [[3, 2, 1]])
    idx = tc.tensor(np.array([[2, 0]], np.int64), dtype="int64")
    assert np.allclose(_np(x.gather(1, idx)), [[3, 1]])


def test_argmax():
    x = tc.tensor(np.array([[1.0, 9.0, 3.0]], np.float32))
    assert _np(x.argmax(1))[0] == 1


def test_norms_and_conv1d():
    x = tc.tensor(np.random.randn(4, 8).astype(np.float32))
    assert nn.BatchNorm1D(8)(x).shape == (4, 8)
    y = tc.tensor(np.random.randn(2, 6, 4, 4).astype(np.float32))
    assert nn.GroupNorm(3, 6)(y).shape == (2, 6, 4, 4)
    assert nn.InstanceNorm2D(6)(y).shape == (2, 6, 4, 4)
    seq = tc.tensor(np.random.randn(2, 3, 10).astype(np.float32))
    assert nn.Conv1D(3, 5, 3, padding=1)(seq).shape == (2, 5, 10)


def test_decoder_and_alibi():
    x = tc.tensor(np.random.randn(2, 5, 32).astype(np.float32))
    mem = tc.tensor(np.random.randn(2, 7, 32).astype(np.float32))
    dec = nn.TransformerDecoderLayer(32, 4, 64)
    assert dec(x, mem).shape == (2, 5, 32)
    bias = F.build_alibi_bias(4, 6)
    assert bias.shape == (4, 6, 6)


def test_weight_tie_single_update():
    emb = nn.Embedding(10, 4)
    head = nn.Linear(4, 10)
    tc.weight_tie(emb, "weight", head, "weight")
    assert head.weight is emb.weight
    params = emb.parameters() + head.parameters()
    opt = optim.SGD(params, lr=0.1)
    assert len(opt.params) == len({id(p) for p in params})  # deduped


def test_scheduler_family():
    lin = nn.Linear(2, 2)
    opt = optim.SGD(lin.parameters(), lr=1.0)
    sch = optim.MultiStepLR(opt, [2, 4], gamma=0.5)
    seen = []
    for _ in range(5):
        sch.step(); seen.append(round(opt.lr, 4))
    # Milestones [2, 4] with 1-indexed stepping drop at the 0->1 and 2->3
    # boundaries (matches PyTorch: [1.0, 0.5, 0.5, 0.25, 0.25]).
    assert seen[0] != seen[1]  # dropped at milestone 2
    assert seen[2] != seen[3]  # dropped at milestone 4
    assert seen == [1.0, 0.5, 0.5, 0.25, 0.25]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
