"""Optimizers — thin wrappers calling the in-place C++/CUDA step kernels."""

from __future__ import annotations

import numpy as np

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


class Lion(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params)
        self.lr, self.b1, self.b2, self.weight_decay = lr, betas[0], betas[1], weight_decay
        self._m = {id(p): _zeros_like_param(p) for p in self.params}

    def step(self):
        for p in self.params:
            g = p.grad
            if g is None:
                continue
            _C.lion_step(p, g, self._m[id(p)], self.lr, self.b1, self.b2, self.weight_decay)


class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params)
        self.lr, self.b1, self.b2, self.eps = lr, betas[0], betas[1], eps
        self.weight_decay = weight_decay
        self.t = 0
        self._m = {id(p): _zeros_like_param(p) for p in self.params}
        self._v = {id(p): _zeros_like_param(p) for p in self.params}

    def step(self):
        self.t += 1
        b1, b2, t = self.b1, self.b2, self.t
        bc1 = 1 - b1 ** t
        bc2 = 1 - b2 ** t
        rho_inf = 2.0 / (1 - b2) - 1
        rho_t = rho_inf - 2 * t * (b2 ** t) / bc2
        rectified = rho_t > 4
        rect = 0.0
        if rectified:
            rect = ((rho_t - 4) * (rho_t - 2) * rho_inf /
                    ((rho_inf - 4) * (rho_inf - 2) * rho_t)) ** 0.5
        for p in self.params:
            g = p.grad
            if g is None:
                continue
            _C.radam_step(p, g, self._m[id(p)], self._v[id(p)], self.lr, b1, b2,
                          self.eps, bc1, bc2, rect, rectified, self.weight_decay, False)


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


class GradScaler:
    """Dynamic loss scaling for fp16 training.

    Usage::
        scaler = GradScaler()
        loss = scaler.scale_loss(criterion(model(x), y))
        loss.backward()
        scaler.step(opt)   # unscales, skips on overflow, adjusts scale
    """
    def __init__(self, init_scale=65536.0, growth_factor=2.0,
                 backoff_factor=0.5, growth_interval=2000):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self._good_steps = 0

    def scale_loss(self, loss):
        return loss * self.scale

    def step(self, optimizer):
        finite = True
        for p in optimizer.params:
            g = p.grad
            if g is None:
                continue
            if not np.isfinite(float((g * g).sum().numpy())):
                finite = False
                break
        if finite:
            inv = 1.0 / self.scale
            for p in optimizer.params:
                if p.grad is not None:
                    _C.scale_(p.grad, inv)
            optimizer.step()
            self._good_steps += 1
            if self._good_steps % self.growth_interval == 0:
                self.scale *= self.growth_factor
        else:
            self.scale *= self.backoff_factor
            self._good_steps = 0
        return finite

    def update(self):  # API-compat no-op (scale already updated in step)
        pass


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


class MultiStepLR(_Scheduler):
    def __init__(self, optimizer, milestones, gamma=0.1):
        super().__init__(optimizer)
        self.milestones = sorted(milestones)
        self.gamma = gamma

    def get_lr(self):
        k = sum(1 for m in self.milestones if self.t >= m)
        return self.base_lr * (self.gamma ** k)


class ExponentialLR(_Scheduler):
    def __init__(self, optimizer, gamma):
        super().__init__(optimizer)
        self.gamma = gamma

    def get_lr(self):
        return self.base_lr * (self.gamma ** self.t)


class ConstantLR(_Scheduler):
    def __init__(self, optimizer, factor=1.0 / 3, total_iters=5):
        super().__init__(optimizer)
        self.factor, self.total_iters = factor, total_iters

    def get_lr(self):
        return self.base_lr * (self.factor if self.t < self.total_iters else 1.0)


class LinearLR(_Scheduler):
    def __init__(self, optimizer, start_factor=1.0 / 3, end_factor=1.0, total_iters=5):
        super().__init__(optimizer)
        self.start, self.end, self.total_iters = start_factor, end_factor, total_iters

    def get_lr(self):
        frac = min(self.t, self.total_iters) / max(1, self.total_iters)
        return self.base_lr * (self.start + (self.end - self.start) * frac)


class OneCycleLR(_Scheduler):
    def __init__(self, optimizer, max_lr, total_steps, pct_start=0.3, div_factor=25.0):
        super().__init__(optimizer)
        self.max_lr, self.total, self.pct_start = max_lr, total_steps, pct_start
        self.div = div_factor

    def get_lr(self):
        import math
        warm = self.pct_start * self.total
        if self.t <= warm:
            frac = self.t / max(1, warm)
            return self.max_lr / self.div + (self.max_lr - self.max_lr / self.div) * frac
        frac = (self.t - warm) / max(1, self.total - warm)
        return self.max_lr * 0.5 * (1 + math.cos(math.pi * min(frac, 1.0)))


class CyclicLR(_Scheduler):
    def __init__(self, optimizer, base_lr, max_lr, step_size_up=2000):
        super().__init__(optimizer)
        self.lo, self.hi, self.step_up = base_lr, max_lr, step_size_up

    def get_lr(self):
        cycle = self.t // (2 * self.step_up)
        x = abs(self.t / self.step_up - 2 * cycle - 1)
        return self.lo + (self.hi - self.lo) * max(0.0, 1 - x)


class ReduceLROnPlateau:
    def __init__(self, optimizer, mode="min", factor=0.1, patience=10, min_lr=0.0):
        self.opt = optimizer
        self.mode, self.factor, self.patience, self.min_lr = mode, factor, patience, min_lr
        self.best = None
        self.bad = 0

    def step(self, metric):
        improved = (self.best is None or
                    (metric < self.best if self.mode == "min" else metric > self.best))
        if improved:
            self.best = metric
            self.bad = 0
        else:
            self.bad += 1
            if self.bad > self.patience:
                self.opt.lr = max(self.min_lr, self.opt.lr * self.factor)
                self.bad = 0
