"""Vendored fp64 VJPs for the TensorCUDA backward tests.

Source: Project-Voice/research/backward64/ops.py, Voice main 24c1eea.
Source SHA-256: fb3fca078cc98198dc497665257a140d19030183c70dd4328154832555642f01.

Prior art: Project-Voice backward64 (2026) supplies these definitions
verbatim. TensorCUDA vendors this subset and gates exact parity.
"""

import numpy as np

FP64 = np.float64


# Prior art: Maas et al. (2013), leaky rectification. Taken: its piecewise
# slope. Voice uses the positive branch at zero.
# Unverified — lead to check: Maas Rectifier Nonlinearities 2013.
def leakyrelu_vjp(x, g, *, slope):
    return (np.asarray(g, FP64) * np.where(np.asarray(x, FP64) < 0, slope, 1.0),)


def _unbroadcast(g, shape):
    # Prior art: NumPy broadcasting (NumPy 2006 onward). Ours: summing
    # replicated cotangents over the forward broadcast axes.
    padded = (1,) * (g.ndim - len(shape)) + shape
    axes = tuple(i for i, (a, b) in enumerate(zip(g.shape, padded)) if a != b)
    return g.sum(axis=axes, keepdims=True).reshape(shape)


# Prior art: Ziyin et al. (2020) Snake, Kokoro (hexgrad, 2024–25) graph
# factorization. Voice uses a hand rule for both appearances of the same alpha.
# Unverified — lead to check: Ziyin periodic activation Snake 2020.
def snake_vjp(x, alpha, g):
    x, alpha, g = (np.asarray(v, FP64) for v in (x, alpha, g))
    phase = alpha * x
    sine = np.sin(phase)
    d_square = 2 * sine * np.cos(phase)
    dx = g * (1.0 + d_square)
    dalpha = g * (d_square * x / alpha - np.square(sine) / np.square(alpha))
    return _unbroadcast(dx, x.shape), _unbroadcast(dalpha, alpha.shape)


def add_vjp(x, y, g):
    g = np.asarray(g, FP64)
    return _unbroadcast(g, np.shape(x)), _unbroadcast(g, np.shape(y))


def div_constant_vjp(x, g, *, divisor):
    return (np.asarray(g, FP64) / divisor,)
