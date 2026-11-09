"""CPU-only engine stand-in for Python GroupNorm reach and refusal checks.
Prior art: REV1 shim_trace.py and gn_emulate.py (float32/per-dtype rounding);
reverse-mode expression graphs, Karpathy micrograd (2020, unverified — lead to
check), chain rule. A2 adds minimal backward wiring to execute the real new
mixed-dtype tests on CPU. It is not evidence of CUDA or engine correctness.
"""
import importlib.util
import sys
import types
from pathlib import Path
import numpy as np

TOOL = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("gn1_rev1_emulate", TOOL.parent / "rev1/gn_emulate.py")
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)


def rnd(a, dtype):
    if dtype in ("int64", "uint8", "bool"):
        return np.asarray(a).astype(dtype)
    return E.rnd(a, dtype)


class Tensor:
    def __init__(self, data, dtype="float32", requires_grad=False, parents=(), vjp=None):
        self._d = rnd(data, dtype)
        self.dtype = dtype
        self.requires_grad = requires_grad or any(t.requires_grad for t in parents)
        self.parents = parents
        self.vjp = vjp
        self.grad = None

    shape = property(lambda self: self._d.shape)
    ndim = property(lambda self: self._d.ndim)
    def numpy(self):
        return self._d.copy()
    def detach(self):
        return Tensor(self._d, self.dtype)
    def _new(self, data, vjp):
        return Tensor(data, self.dtype, parents=(self,), vjp=lambda g:(vjp(g),))
    def astype(self, dtype):
        if dtype == self.dtype:
            return self
        return Tensor(self._d, dtype, parents=(self,), vjp=lambda g:(rnd(g, self.dtype),))
    def reshape(self, shape):
        if any(v < 0 for v in shape) or int(np.prod(shape)) != self._d.size:
            raise RuntimeError("reshape: element count mismatch")
        return self._new(self._d.reshape(shape), lambda g:g.reshape(self.shape))
    def permute(self, dims):
        return self._new(self._d.transpose(dims), lambda g:g.transpose(np.argsort(dims)))
    def sum(self, axes=None, keep=False):
        axes = tuple(range(self.ndim)) if axes is None else tuple(a % self.ndim for a in axes)
        data = self._d
        for axis in sorted(axes, reverse=True):
            data = E.rsum(data, axis, True, self.dtype)
        if not keep:
            data = data.squeeze(axes)
        def back(g):
            if not keep:
                for axis in sorted(axes):
                    g = np.expand_dims(g, axis)
            return np.broadcast_to(g, self.shape)
        return self._new(data, back)
    def mean(self, axes, keep):
        return self.sum(axes, keep) * np.float32(1 / np.prod([self.shape[a] for a in axes]))
    def var(self, axes, keep):
        d = self - self.mean(axes, True)
        return (d*d).mean(axes, keep)
    def pow(self, power):
        with np.errstate(all="ignore"):
            data = np.power(self._d, power)
        return self._new(data, lambda g:rnd(g * rnd(rnd(np.power(self._d, power-1), self.dtype) * np.float32(power), self.dtype), self.dtype))
    def _binary(self, other, op, derivatives):
        if not isinstance(other, Tensor):
            other = Tensor(np.float32(other), self.dtype)
        if other.dtype != self.dtype:
            raise RuntimeError("ew_binary dtype mismatch")
        def reduce(g, shape):
            while g.ndim > len(shape):
                g = E.rsum(g, 0, False, self.dtype)
            for axis, size in enumerate(shape):
                if size == 1 and g.shape[axis] > 1:
                    g = E.rsum(g, axis, True, self.dtype)
            return g.reshape(shape)
        def vjp(g):
            a, b = derivatives(g, self._d, other._d)
            return reduce(rnd(a, self.dtype), self.shape), reduce(rnd(b, self.dtype), other.shape)
        return Tensor(op(self._d, other._d), self.dtype, parents=(self, other), vjp=vjp)
    def __add__(self, other):
        return self._binary(other, np.add, lambda g,a,b:(g,g))
    __radd__ = __add__
    def __sub__(self, other):
        return self._binary(other, np.subtract, lambda g,a,b:(g,-g))
    def __mul__(self, other):
        return self._binary(other, np.multiply, lambda g,a,b:(g*b,g*a))
    __rmul__ = __mul__
    def backward(self):
        order, seen = [], set()
        def visit(t):
            if id(t) not in seen:
                seen.add(id(t))
                for p in t.parents:
                    visit(p)
                order.append(t)
        visit(self)
        grads = {id(self): np.ones(self.shape)}
        for t in reversed(order):
            if id(t) not in grads:
                continue
            g = rnd(grads[id(t)], t.dtype)
            if t.requires_grad:
                t.grad = Tensor(g, t.dtype)
            if t.vjp:
                for p, gp in zip(t.parents, t.vjp(g)):
                    if p.requires_grad:
                        old = grads.get(id(p))
                        grads[id(p)] = rnd(gp if old is None else old+gp, p.dtype)


def install(path):
    tc = types.ModuleType("tensor_cuda")
    tc.__file__ = str(path.parent / "__init__.py")
    tc.__path__ = [str(path.parent)]
    tc.__spec__ = importlib.util.spec_from_file_location("tensor_cuda", tc.__file__, submodule_search_locations=tc.__path__)
    tc.Tensor = Tensor
    tc.tensor = lambda data, device="cuda", dtype="float32", requires_grad=False:Tensor(data, dtype, requires_grad)
    sys.modules["tensor_cuda"] = tc
    spec = importlib.util.spec_from_file_location("tensor_cuda.nn", path)
    nn = importlib.util.module_from_spec(spec)
    sys.modules["tensor_cuda.nn"] = nn
    spec.loader.exec_module(nn)
    tc.nn = nn
    return tc
