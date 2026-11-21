# Copied convolution subset of Voice research/backward64/ops.py (24c1eea).
# Source SHA-256 fb3fca078cc98198dc497665257a140d19030183c70dd4328154832555642f01.
"""Kokoro decoder primitives. VJPs take the forward inputs and an output cotangent."""
import numpy as np

FP64 = np.float64


def _windows(x, kernel, stride, dilation, pads):
    padded = np.pad(x, ((0, 0), (0, 0), pads))
    extent = dilation * (kernel - 1) + 1
    windows = np.lib.stride_tricks.sliding_window_view(padded, extent, axis=-1)
    return windows[:, :, ::stride, ::dilation], padded.shape


# Prior art: ONNX Conv (2017) cross-correlation and Caffe (Jia et al., 2014)
# im2col/GEMM. Taken: patch packing. Ours: the fp64 gather/scatter and both
# kernel and activation adjoints. Unverified — lead to check: Caffe im2col.
def conv1d(x, w, b=None, *, stride=1, dilation=1, groups=1, pads=(0, 0)):
    x, w = np.asarray(x, FP64), np.asarray(w, FP64)
    n, _, _ = x.shape
    co, ci, kernel = w.shape
    patches, _ = _windows(x, kernel, stride, dilation, pads)
    length = patches.shape[2]
    result = np.empty((n, co, length), dtype=FP64)
    for part in range(groups):
        inputs = slice(part * ci, (part + 1) * ci)
        outputs = slice(part * (co // groups), (part + 1) * (co // groups))
        col = np.ascontiguousarray(patches[:, inputs].transpose(0, 2, 1, 3))
        product = col.reshape(n * length, ci * kernel) @ w[outputs].reshape(-1, ci * kernel).T
        result[:, outputs] = product.reshape(n, length, -1).transpose(0, 2, 1)
    if b is not None:
        result += np.asarray(b, FP64)[None, :, None]
    return result


def conv1d_vjp(x, w, b, g, *, stride=1, dilation=1, groups=1, pads=(0, 0)):
    x, w, g = (np.asarray(v, FP64) for v in (x, w, g))
    n, co, length = g.shape
    _, ci, kernel = w.shape
    patches, padded_shape = _windows(x, kernel, stride, dilation, pads)
    dxpad, dw = np.zeros(padded_shape, dtype=FP64), np.empty_like(w)
    for part in range(groups):
        inputs = slice(part * ci, (part + 1) * ci)
        outputs = slice(part * (co // groups), (part + 1) * (co // groups))
        col = np.ascontiguousarray(patches[:, inputs].transpose(0, 2, 1, 3))
        grad = np.ascontiguousarray(g[:, outputs].transpose(0, 2, 1)).reshape(n * length, -1)
        dw[outputs] = (grad.T @ col.reshape(n * length, -1)).reshape(-1, ci, kernel)
        dcol = (grad @ w[outputs].reshape(-1, ci * kernel)).reshape(n, length, ci, kernel)
        for tap in range(kernel):
            start = tap * dilation
            dxpad[:, inputs, start:start + length * stride:stride] += dcol[..., tap].transpose(0, 2, 1)
    return dxpad[..., pads[0]:pads[0] + x.shape[-1]], dw, None if b is None else g.sum(axis=(0, 2))


def _transpose_slices(tap, dilation, stride, left, in_length, out_length):
    offset = tap * dilation - left
    first = min(in_length, max(0, -(offset // stride)))
    end = min(in_length, max(first, (out_length - 1 - offset) // stride + 1))
    return slice(first, end), slice(offset + first * stride, offset + end * stride, stride)


# Prior art: ONNX ConvTranspose (2017), Dumoulin and Visin (2016)
# "A guide to convolution arithmetic". Taken: strided scatter and extent formula.
# Ours: per-tap in-bounds slices and grouped fp64 VJPs.
# Unverified — lead to check: Dumoulin Visin transposed convolution arithmetic.
def conv_transpose1d(x, w, b=None, *, stride=1, dilation=1, groups=1, pads=(0, 0), output_padding=0):
    x, w = np.asarray(x, FP64), np.asarray(w, FP64)
    n, ci, length = x.shape
    _, co_group, kernel = w.shape
    out_length = stride * (length - 1) + dilation * (kernel - 1) + 1 + output_padding - sum(pads)
    xg = x.reshape(n, groups, ci // groups, length)
    wg = w.reshape(groups, ci // groups, co_group, kernel)
    result = np.zeros((n, groups, co_group, out_length), dtype=FP64)
    for tap in range(kernel):
        source, target = _transpose_slices(tap, dilation, stride, pads[0], length, out_length)
        if source.start < source.stop:
            result[..., target] += wg[None, ..., tap].swapaxes(-2, -1) @ xg[..., source]
    result = result.reshape(n, groups * co_group, out_length)
    if b is not None:
        result += np.asarray(b, FP64)[None, :, None]
    return result


def conv_transpose1d_vjp(x, w, b, g, *, stride=1, dilation=1, groups=1, pads=(0, 0), output_padding=0):
    x, w, g = (np.asarray(v, FP64) for v in (x, w, g))
    n, ci, length = x.shape
    _, co_group, kernel = w.shape
    xg = x.reshape(n, groups, ci // groups, length)
    wg = w.reshape(groups, ci // groups, co_group, kernel)
    gg = g.reshape(n, groups, co_group, g.shape[-1])
    dxg, dwg = np.zeros_like(xg), np.zeros_like(wg)
    for tap in range(kernel):
        source, target = _transpose_slices(tap, dilation, stride, pads[0], length, g.shape[-1])
        if source.start < source.stop:
            go = gg[..., target]
            dxg[..., source] += wg[None, ..., tap] @ go
            dwg[..., tap] = (xg[..., source] @ go.swapaxes(-2, -1)).sum(axis=0)
    return dxg.reshape(x.shape), dwg.reshape(w.shape), None if b is None else g.sum(axis=(0, 2))

