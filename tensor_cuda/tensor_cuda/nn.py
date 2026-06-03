"""Neural-network layers — a thin Python wrapper over the C++/CUDA engine.

Modules are parameter containers; all compute (matmul, activations, embedding,
norm math) runs in C++ kernels via `tensor_cuda` ops. This mirrors the reference
`tensor_gpu_v2._nn` API.
"""

from __future__ import annotations

import math

import numpy as np

import tensor_cuda as tc


def parameter(array, *, device="cuda", dtype="float32"):
    return tc.tensor(array, device=device, dtype=dtype, requires_grad=True)


class Module:
    def __init__(self):
        object.__setattr__(self, "_params", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "_buffers", {})
        object.__setattr__(self, "training", True)

    def __setattr__(self, name, value):
        if isinstance(value, tc.Tensor):
            if value.requires_grad:
                self._params[name] = value
            else:
                self._buffers[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def parameters(self):
        out = list(self._params.values())
        for m in self._modules.values():
            out.extend(m.parameters())
        return out

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def train(self, mode=True):
        object.__setattr__(self, "training", mode)
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        bound = 1.0 / math.sqrt(in_features)
        self.weight = parameter(
            np.random.uniform(-bound, bound, (out_features, in_features)))
        self.bias = parameter(np.zeros(out_features)) if bias else None

    def forward(self, x):
        out = tc.matmul(x, self.weight.transpose(0, 1))
        if self.bias is not None:
            out = out + self.bias
        return out


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight = parameter(
            np.random.randn(num_embeddings, embedding_dim) * 0.02)

    def forward(self, idx):
        return tc.embedding(self.weight, idx)


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        dim = normalized_shape if isinstance(normalized_shape, int) else int(np.prod(normalized_shape))
        self.weight = parameter(np.ones(dim))
        self.bias = parameter(np.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean([-1], True)
        xc = x - mean
        var = (xc * xc).mean([-1], True)
        inv = (var + self.eps).pow(-0.5)
        return xc * inv * self.weight + self.bias


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        mask = (np.random.rand(*x.shape) > self.p).astype(np.float32) / (1 - self.p)
        return x * tc.tensor(mask, device=x.device.split(":")[0])


class ReLU(Module):
    def forward(self, x): return x.relu()


class GELU(Module):
    def forward(self, x): return x.gelu()


class SiLU(Module):
    def forward(self, x): return x.silu()


class Sigmoid(Module):
    def forward(self, x): return x.sigmoid()


class Tanh(Module):
    def forward(self, x): return x.tanh()


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)
        for i, l in enumerate(layers):
            self._modules[str(i)] = l

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class RMSNorm(Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = parameter(np.ones(dim))
        self.eps = eps

    def forward(self, x):
        ms = (x * x).mean([-1], True)
        return x * (ms + self.eps).pow(-0.5) * self.weight


class MultiheadAttention(Module):
    def __init__(self, embed_dim, num_heads, bias=True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = Linear(embed_dim, embed_dim, bias)
        self.k_proj = Linear(embed_dim, embed_dim, bias)
        self.v_proj = Linear(embed_dim, embed_dim, bias)
        self.out_proj = Linear(embed_dim, embed_dim, bias)

    def _split(self, x, B, L):
        return x.reshape([B, L, self.num_heads, self.head_dim]).transpose(1, 2)

    def forward(self, x, attn_mask=None, is_causal=False):
        from . import functional as F
        B, L, _ = x.shape
        q = self._split(self.q_proj(x), B, L)
        k = self._split(self.k_proj(x), B, L)
        v = self._split(self.v_proj(x), B, L)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask, is_causal)
        out = out.transpose(1, 2).reshape([B, L, self.embed_dim])
        return self.out_proj(out)


class TransformerEncoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, activation="gelu"):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.act = activation

    def _ff(self, x):
        h = self.linear1(x)
        h = h.gelu() if self.act == "gelu" else h.relu()
        return self.linear2(h)

    def forward(self, x, attn_mask=None, is_causal=False):
        x = self.norm1(x + self.self_attn(x, attn_mask, is_causal))
        x = self.norm2(x + self._ff(x))
        return x


class MSELoss(Module):
    def forward(self, pred, target): return tc.mse_loss(pred, target)


class CrossEntropyLoss(Module):
    def forward(self, logits, target): return tc.cross_entropy(logits, target)
