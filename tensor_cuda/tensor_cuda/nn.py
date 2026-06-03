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


class MSELoss(Module):
    def forward(self, pred, target): return tc.mse_loss(pred, target)


class CrossEntropyLoss(Module):
    def forward(self, logits, target): return tc.cross_entropy(logits, target)
