"""Verbatim Voice 24c1eea research/backward64/ops.py excerpts.

Source SHA-256 fb3fca078cc98198dc497665257a140d19030183c70dd4328154832555642f01.
"""

import numpy as np

FP64 = np.float64


# Prior art: Ulyanov et al. (2016) Instance Normalization, Huang and
# Belongie (2017) AdaIN, Li et al. (2023) StyleTTS 2. Taken: the normalization
# and affine forms. Ours: explicit quotient/centering VJPs and graph binding.
# Unverified — lead to check: Instance Normalization 2016, AdaIN 2017.
def _instance_stats(x, eps):
    x = np.asarray(x, FP64)
    centered = x - x.mean(axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True) + eps)
    return centered * inv_std, inv_std


def instance_norm1d(x, scale, bias, *, eps):
    normalized, _ = _instance_stats(x, eps)
    return normalized * np.asarray(scale, FP64)[None, :, None] + np.asarray(bias, FP64)[None, :, None]


def instance_norm1d_vjp(x, scale, bias, g, *, eps):
    normalized, inv_std = _instance_stats(x, eps)
    g = np.asarray(g, FP64)
    dn = g * np.asarray(scale, FP64)[None, :, None]
    dx = inv_std * (dn - dn.mean(axis=-1, keepdims=True)
                    - normalized * (dn * normalized).mean(axis=-1, keepdims=True))
    return dx, (g * normalized).sum(axis=(0, 2)), g.sum(axis=(0, 2))


def _style(s, W, b):
    h = np.asarray(s, FP64) @ np.asarray(W, FP64).T + np.asarray(b, FP64)
    gamma, beta = np.split(h[:, :, None], 2, axis=1)
    return 1.0 + gamma, beta


def adain(x, s, W, b, scale, bias, *, eps):
    gamma1, beta = _style(s, W, b)
    return gamma1 * instance_norm1d(x, scale, bias, eps=eps) + beta


def adain_vjp(x, s, W, b, scale, bias, g, *, eps):
    g = np.asarray(g, FP64)
    gamma1, _ = _style(s, W, b)
    norm = instance_norm1d(x, scale, bias, eps=eps)
    dh = np.concatenate(((g * norm).sum(axis=-1), g.sum(axis=-1)), axis=1)
    dx, dscale, dbias = instance_norm1d_vjp(x, scale, bias, g * gamma1, eps=eps)
    return dx, dh @ np.asarray(W, FP64), dh.T @ np.asarray(s, FP64), dh.sum(axis=0), dscale, dbias


# Prior art: ONNX Resize (2017, operator specification) and Pad reflect,
# NumPy indexed gather/scatter. Taken: forward index conventions. Ours:
# explicit accumulation across repeated indices for their fp64 adjoints.
def upsample_nearest1d(x, *, factor=2):
    return np.repeat(np.asarray(x, FP64), factor, axis=-1)


def upsample_nearest1d_vjp(x, g, *, factor=2):
    return (np.asarray(g, FP64).reshape(*np.shape(x), factor).sum(axis=-1),)


def _reflection_indices(length, pads):
    return np.pad(np.arange(length), pads, mode="reflect")


def reflection_pad1d(x, *, pads):
    x = np.asarray(x, FP64)
    return x[..., _reflection_indices(x.shape[-1], pads)]


def reflection_pad1d_vjp(x, g, *, pads):
    dx = np.zeros(np.shape(x), dtype=FP64)
    np.add.at(dx, (..., _reflection_indices(dx.shape[-1], pads)), np.asarray(g, FP64))
    return (dx,)


