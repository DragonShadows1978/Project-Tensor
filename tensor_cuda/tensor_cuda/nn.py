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


# ------------------------------------------------------------- convolution
def _pair(x):
    return x if isinstance(x, (tuple, list)) else (x, x)


class Conv2D(Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.kh, self.kw = _pair(kernel_size)
        self.sh, self.sw = _pair(stride)
        self.ph, self.pw = _pair(padding)
        fan_in = in_ch * self.kh * self.kw
        bound = 1.0 / math.sqrt(fan_in)
        self.weight = parameter(np.random.uniform(
            -bound, bound, (out_ch, in_ch, self.kh, self.kw)))
        self.bias = parameter(np.zeros(out_ch)) if bias else None

    def forward(self, x):
        N = x.shape[0]
        K = self.in_ch * self.kh * self.kw
        cols = tc._C.im2col(x, self.kh, self.kw, self.sh, self.sw, self.ph, self.pw)
        L = cols.shape[-1]
        W2 = self.weight.reshape([self.out_ch, K])
        out = tc.matmul(cols.transpose(1, 2), W2.transpose(0, 1))  # (N, L, out)
        OH = (x.shape[2] + 2 * self.ph - self.kh) // self.sh + 1
        OW = (x.shape[3] + 2 * self.pw - self.kw) // self.sw + 1
        out = out.transpose(1, 2).reshape([N, self.out_ch, OH, OW])
        if self.bias is not None:
            out = out + self.bias.reshape([1, self.out_ch, 1, 1])
        return out


class MaxPool2D(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kh, self.kw = _pair(kernel_size)
        s = stride if stride is not None else kernel_size
        self.sh, self.sw = _pair(s)
        self.ph, self.pw = _pair(padding)

    def forward(self, x):
        return tc._C.max_pool2d(x, self.kh, self.kw, self.sh, self.sw, self.ph, self.pw)


class AvgPool2D(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kh, self.kw = _pair(kernel_size)
        s = stride if stride is not None else kernel_size
        self.sh, self.sw = _pair(s)
        self.ph, self.pw = _pair(padding)

    def forward(self, x):
        return tc._C.avg_pool2d(x, self.kh, self.kw, self.sh, self.sw, self.ph, self.pw)


class BatchNorm2D(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps, self.momentum, self.C = eps, momentum, num_features
        self.weight = parameter(np.ones(num_features))
        self.bias = parameter(np.zeros(num_features))
        self.running_mean = tc.zeros(num_features)      # buffers (no grad)
        self.running_var = tc.ones(num_features)

    def forward(self, x):
        w = self.weight.reshape([1, self.C, 1, 1])
        b = self.bias.reshape([1, self.C, 1, 1])
        if self.training:
            mean = x.mean([0, 2, 3], True)
            var = x.var([0, 2, 3], True)
            mn = mean.numpy().ravel(); vr = var.numpy().ravel()
            rm = self.running_mean.numpy(); rv = self.running_var.numpy()
            self.running_mean = tc.tensor((1 - self.momentum) * rm + self.momentum * mn)
            self.running_var = tc.tensor((1 - self.momentum) * rv + self.momentum * vr)
        else:
            mean = self.running_mean.reshape([1, self.C, 1, 1]).detach()
            var = self.running_var.reshape([1, self.C, 1, 1]).detach()
        return (x - mean) * (var + self.eps).pow(-0.5) * w + b


# ------------------------------------------------------------- recurrent
class RNNCell(Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.ih = Linear(input_size, hidden_size)
        self.hh = Linear(hidden_size, hidden_size)

    def forward(self, x, h):
        return (self.ih(x) + self.hh(h)).tanh()


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.ih = Linear(input_size, 4 * hidden_size)
        self.hh = Linear(hidden_size, 4 * hidden_size)

    def forward(self, x, state):
        h, c = state
        g = self.ih(x) + self.hh(h)
        H = self.hidden_size
        i = g.slice(-1, 0, H).sigmoid()
        f = g.slice(-1, H, H).sigmoid()
        gg = g.slice(-1, 2 * H, H).tanh()
        o = g.slice(-1, 3 * H, H).sigmoid()
        c2 = f * c + i * gg
        h2 = o * c2.tanh()
        return h2, c2


class GRUCell(Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.ih = Linear(input_size, 3 * hidden_size)
        self.hh = Linear(hidden_size, 3 * hidden_size)

    def forward(self, x, h):
        H = self.hidden_size
        xi, hi = self.ih(x), self.hh(h)
        r = (xi.slice(-1, 0, H) + hi.slice(-1, 0, H)).sigmoid()
        z = (xi.slice(-1, H, H) + hi.slice(-1, H, H)).sigmoid()
        n = (xi.slice(-1, 2 * H, H) + r * hi.slice(-1, 2 * H, H)).tanh()
        return (z * -1.0 + 1.0) * n + z * h


class _RNNBase(Module):
    """Iterates a cell over the time dimension (input (B, T, input_size))."""
    cell_cls = None
    has_cell_state = False

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = self.cell_cls(input_size, hidden_size)

    def forward(self, x):
        B, T, _ = x.shape
        dev = x.device.split(":")[0]
        h = tc.zeros(B, self.hidden_size, device=dev)
        c = tc.zeros(B, self.hidden_size, device=dev) if self.has_cell_state else None
        outs = []
        for t in range(T):
            xt = x.slice(1, t, 1).reshape([B, x.shape[2]])
            if self.has_cell_state:
                h, c = self.cell(xt, (h, c))
            else:
                h = self.cell(xt, h)
            outs.append(h)
        return tc.stack(outs, dim=1)  # (B, T, hidden)


class RNN(_RNNBase):
    cell_cls = RNNCell


class LSTM(_RNNBase):
    cell_cls = LSTMCell
    has_cell_state = True


class GRU(_RNNBase):
    cell_cls = GRUCell


# ------------------------------------------------------------- losses
class MSELoss(Module):
    def forward(self, pred, target): return tc.mse_loss(pred, target)


class L1Loss(Module):
    def forward(self, pred, target):
        return (pred - target).abs().mean()


class BCEWithLogitsLoss(Module):
    def forward(self, x, y):
        # max(x,0) - x*y + log(1 + exp(-|x|))
        term = x.relu() - x * y + (x.abs() * -1.0).exp().__add__(1.0).log()
        return term.mean()


class SmoothL1Loss(Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred, target):
        d = (pred - target).abs()
        quad = d.pow(2.0) * (0.5 / self.beta)
        lin = d - 0.5 * self.beta
        small = d.__lt__(self.beta)  # detached 0/1
        return tc.where(small, quad, lin).mean()


class CrossEntropyLoss(Module):
    def forward(self, logits, target): return tc.cross_entropy(logits, target)
