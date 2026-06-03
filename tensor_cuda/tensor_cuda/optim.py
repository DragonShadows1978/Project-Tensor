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


class RMSprop(Optimizer):
    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0.0):
        super().__init__(params)
        self.lr, self.alpha, self.eps, self.weight_decay = lr, alpha, eps, weight_decay
        self._sq = {id(p): _zeros_like_param(p) for p in self.params}

    def step(self):
        for p in self.params:
            g = p.grad
            if g is None:
                continue
            _C.rmsprop_step(p, g, self._sq[id(p)], self.lr, self.alpha, self.eps, self.weight_decay)


class Adagrad(Optimizer):
    def __init__(self, params, lr=1e-2, eps=1e-10, weight_decay=0.0):
        super().__init__(params)
        self.lr, self.eps, self.weight_decay = lr, eps, weight_decay
        self._acc = {id(p): _zeros_like_param(p) for p in self.params}

    def step(self):
        for p in self.params:
            g = p.grad
            if g is None:
                continue
            _C.adagrad_step(p, g, self._acc[id(p)], self.lr, self.eps, self.weight_decay)


def clip_grad_norm_(params, max_norm, norm_type=2.0):
    """Clip gradients by global norm (in place). Returns the pre-clip norm."""
    params = [p for p in params if p.grad is not None]
    total = 0.0
    for p in params:
        g = p.grad
        total += float((g * g).sum().numpy())
    total = total ** 0.5
    coef = max_norm / (total + 1e-6)
    if coef < 1.0:
        for p in params:
            _C.scale_(p.grad, coef)
    return total


# -------------------------------------------------------------- LR schedulers
class _Scheduler:
    def __init__(self, optimizer):
        self.opt = optimizer
        self.base_lr = optimizer.lr
        self.t = 0

    def step(self):
        self.t += 1
        self.opt.lr = self.get_lr()

    def get_lr(self):
        raise NotImplementedError


class StepLR(_Scheduler):
    def __init__(self, optimizer, step_size, gamma=0.1):
        super().__init__(optimizer)
        self.step_size, self.gamma = step_size, gamma

    def get_lr(self):
        return self.base_lr * (self.gamma ** (self.t // self.step_size))


class CosineAnnealingLR(_Scheduler):
    def __init__(self, optimizer, T_max, eta_min=0.0):
        super().__init__(optimizer)
        self.T_max, self.eta_min = T_max, eta_min

    def get_lr(self):
        import math
        return self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (
            1 + math.cos(math.pi * min(self.t, self.T_max) / self.T_max))


class LinearWarmupCosineDecay(_Scheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, eta_min=0.0):
        super().__init__(optimizer)
        self.warmup, self.total, self.eta_min = warmup_steps, total_steps, eta_min

    def get_lr(self):
        import math
        if self.t < self.warmup:
            return self.base_lr * self.t / max(1, self.warmup)
        prog = (self.t - self.warmup) / max(1, self.total - self.warmup)
        return self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1 + math.cos(math.pi * min(prog, 1.0)))
