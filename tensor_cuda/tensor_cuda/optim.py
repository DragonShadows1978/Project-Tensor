"""Optimizers — thin wrappers calling the in-place C++/CUDA step kernels."""

from __future__ import annotations

import tensor_cuda as tc

_C = tc._C


def _zeros_like_param(p):
    return tc.zeros(tuple(p.shape), dtype=p.dtype, device=p.device.split(":")[0])


class Optimizer:
    def __init__(self, params):
        self.params = list(params)

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()


class SGD(Optimizer):
    def __init__(self, params, lr=1e-2, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr, self.momentum, self.weight_decay = lr, momentum, weight_decay
        self._buf = {id(p): _zeros_like_param(p) for p in self.params}

    def step(self):
        for p in self.params:
            g = p.grad
            if g is None:
                continue
            _C.sgd_step(p, g, self._buf[id(p)], self.lr, self.momentum, self.weight_decay)


class Adam(Optimizer):
    _decoupled = False

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params)
        self.lr, self.b1, self.b2, self.eps = lr, betas[0], betas[1], eps
        self.weight_decay = weight_decay
        self.t = 0
        self._m = {id(p): _zeros_like_param(p) for p in self.params}
        self._v = {id(p): _zeros_like_param(p) for p in self.params}

    def step(self):
        self.t += 1
        for p in self.params:
            g = p.grad
            if g is None:
                continue
            _C.adam_step(p, g, self._m[id(p)], self._v[id(p)], self.lr, self.b1,
                         self.b2, self.eps, self.t, self.weight_decay, self._decoupled)


class AdamW(Adam):
    """Adam with decoupled weight decay."""
    _decoupled = True
