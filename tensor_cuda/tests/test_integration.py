"""End-to-end integration: a tiny causal transformer LM on the full stack.

Exercises Embedding -> TransformerEncoderLayer(causal) -> Linear head ->
cross_entropy -> AdamW, plus indexing, checkpoint round-trip, and APA attention
as a drop-in for standard attention.
"""

import numpy as np
import pytest

import tensor_cuda as tc
from tensor_cuda import nn, optim


class TinyLM(nn.Module):
    def __init__(self, vocab, d, heads, layers):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(
            [nn.TransformerEncoderLayer(d, heads, 2 * d) for _ in range(layers)])
        self.head = nn.Linear(d, vocab)

    def forward(self, idx):
        x = self.emb(idx)
        for blk in self.blocks:
            x = blk(x, is_causal=True)
        return self.head(x)


def test_tiny_lm_trains_and_checkpoints(tmp_path):
    rng = np.random.default_rng(0)
    V, D, B, T = 16, 32, 4, 8
    idx_np = rng.integers(0, V, size=(B, T)).astype(np.int64)
    labels = idx_np.reshape(-1)  # dummy next-token-ish targets

    model = TinyLM(V, D, heads=4, layers=2)
    opt = optim.AdamW(model.parameters(), lr=3e-3)

    idx = tc.tensor(idx_np, dtype="int64")
    first = last = None
    for _ in range(40):
        logits = model(idx).reshape([B * T, V])
        loss = tc.cross_entropy(logits, labels)
        opt.zero_grad(); loss.backward(); opt.step()
        v = float(loss.numpy()); first = v if first is None else first; last = v
    assert last < first * 0.8, f"LM did not learn: {first:.3f} -> {last:.3f}"

    # checkpoint round-trip
    path = str(tmp_path / "lm.npz")
    tc.save_checkpoint(path, model, step=40)
    out_before = model(idx).numpy()
    model2 = TinyLM(V, D, heads=4, layers=2)
    extra = tc.load_checkpoint(path, model2)
    assert int(extra["step"]) == 40
    assert np.allclose(model2(idx).numpy(), out_before, atol=1e-4)


def test_indexing():
    x = tc.tensor(np.arange(24).reshape(2, 3, 4).astype(np.float32))
    assert np.allclose(x[0].numpy(), np.arange(12).reshape(3, 4))
    assert np.allclose(x[1, 2].numpy(), np.arange(20, 24))
    assert x[:, 1:3].numpy().shape == (2, 2, 4)


def test_apa_drop_in_for_attention():
    rng = np.random.default_rng(1)
    q = tc.tensor(rng.standard_normal((2, 4, 32, 16)).astype(np.float32), requires_grad=True)
    k = tc.tensor(rng.standard_normal((2, 4, 32, 16)).astype(np.float32), requires_grad=True)
    v = tc.tensor(rng.standard_normal((2, 4, 32, 16)).astype(np.float32), requires_grad=True)
    out = tc.apa_quant_attention(q, k, v, bulk_bits=2, refine_percentile=0.2)
    assert out.shape == (2, 4, 32, 16)
    out.sum().backward()
    assert np.all(np.isfinite(q.grad.numpy()))


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
