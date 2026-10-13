"""float64 CPU oracles and data builders for tc_signal's P1T-M4 tests (numpy, same process).

Norm oracles: ONNX InstanceNormalization / LayerNormalization / GroupNormalization-21, computed as a plain two-pass in
float64 (numpy's pairwise mean): mean, centred values, biased variance, xc / sqrt(var + eps) * scale + bias. They share
no code or formula detail with the CUDA kernel (no shift, no fp32, no blocked summation). test_norms.py cross-checks them
against torch's float64 F.instance_norm / F.layer_norm / F.group_norm.

Exact-op oracles: integer index maps on the uint32 view (so every bit, NaN payloads included, is compared):
upsample y[..., i] = x[..., i // s]; reflect pad via numpy's np.pad(mode='reflect') (cross-checked against the ONNX
index formula and torch's F.pad(mode='reflect')); rint = numpy's np.rint (IEEE roundTiesToEven), cross-checked against
torch.round in float64 (Voice's oracle).

Normwise N = ||y - o||_inf / ||o||_inf (Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed. 2002; Voice's
common.normwise; o == 0 everywhere passes only if y is exactly 0). Prior art for the oracles: the ONNX operator specs;
Chan, Golub & LeVeque (1983) for two-pass variance. Ours: the transcription and the data builders.
"""
import zlib

import numpy as np

import m4_census as C

F64 = np.float64


def rng(name):
    return np.random.default_rng(C.SEED + zlib.crc32(name.encode()) % C.SEED_MOD)


# ------------------------------------------------------------------------------------------------ norms
def norm_geometry(op, shape, arg):
    """(rows, L, param_shape) of a norm problem on a C-contiguous x."""
    if op == 'instance':
        N, Ch, T = shape
        return N * Ch, T, (Ch,)
    if op == 'group':
        N, Ch, T = shape
        return N * arg, (Ch // arg) * T, (Ch,)
    r = len(shape)
    a = arg + r if arg < 0 else arg
    return int(np.prod(shape[:a], dtype=np.int64)), int(np.prod(shape[a:], dtype=np.int64)), tuple(shape[a:])


def norm_data(name, op, shape, arg, data, affine):
    """x (fp32, C-contiguous), scale, bias (fp32 or None) for one cell."""
    g = rng(name)
    rows, L, pshape = norm_geometry(op, shape, arg)
    if data == 'act':
        # activation-like: per last-axis row (channel / token) offset and spread
        lead = shape[:-1] if len(shape) > 1 else ()
        off = g.standard_normal(lead + (1,)) * C.OFFSET_SD
        spread = np.exp(g.standard_normal(lead + (1,)) * C.LOG_SPREAD_SD)
        x = off + spread * g.standard_normal(shape)
    else:
        # per normalization row
        z = g.standard_normal((rows, L))
        if data == 'const':
            x = np.broadcast_to(g.standard_normal((rows, 1)) * C.OFFSET_SD, (rows, L))
        elif data == 'offset':
            x = C.BIG_OFFSET + C.BIG_OFFSET_STD * z
        elif data == 'neg_offset':
            x = -C.BIG_OFFSET + C.BIG_OFFSET_STD * z
        elif data == 'small_var':
            x = C.SMALL_VAR_MEAN + C.SMALL_VAR_STD * z
        else:
            raise ValueError(data)
        x = x.reshape(shape)
    x = np.ascontiguousarray(x, dtype=np.float32)
    scale = g.standard_normal(pshape).astype(np.float32) if affine in ('both', 'scale') else None
    bias = g.standard_normal(pshape).astype(np.float32) if affine in ('both', 'bias') else None
    return x, scale, bias


def _affine_view(op, shape, p):
    if p is None:
        return None
    p = p.astype(F64)
    if op in ('instance', 'group'):
        return p.reshape(1, -1, 1)
    return p  # layer: x.shape[axis:], broadcasts against the trailing axes


def norm_oracle(op, x, scale, bias, arg, eps):
    rows, L, _ = norm_geometry(op, x.shape, arg)
    xr = x.astype(F64).reshape(rows, L)
    m = xr.mean(axis=1, keepdims=True)
    xc = xr - m
    var = (xc * xc).mean(axis=1, keepdims=True)
    y = (xc / np.sqrt(var + float(eps))).reshape(x.shape)
    s, b = _affine_view(op, x.shape, scale), _affine_view(op, x.shape, bias)
    if s is not None:
        y = y * s
    if b is not None:
        y = y + b
    return y


def bias_expected(op, shape, bias):
    """What a zero-variance norm must return: the bias broadcast over x (or zeros without a bias)."""
    if bias is None:
        return np.zeros(shape, np.float32)
    if op in ('instance', 'group'):
        return np.broadcast_to(bias.reshape(1, -1, 1), shape).astype(np.float32)
    return np.broadcast_to(bias, shape).astype(np.float32)


def normwise(y, o):
    y = np.asarray(y, dtype=F64)
    den = float(np.max(np.abs(o)))
    num = float(np.max(np.abs(y - o)))
    if den == 0.0:
        return 0.0 if num == 0.0 else float('inf')
    return num / den


# ------------------------------------------------------------------------------------------------ exact ops
def special_data(name, shape):
    """fp32 data with NaN payloads (signalling and negative NaNs too), +-0, +-Inf and subnormals mixed in."""
    g = rng(name)
    x = (g.standard_normal(shape) * np.exp2(g.integers(-8, 8, shape))).astype(np.float32)
    w = x.view(np.uint32).copy()
    mask = g.random(shape) < C.SPECIAL_FRACTION
    w[mask] = g.choice(np.array(C.SPECIAL_WORDS, dtype=np.uint32), size=int(mask.sum()))
    return w.view(np.float32)


def act_data(name, shape):
    g = rng(name)
    return (g.standard_normal(shape) * C.OFFSET_SD).astype(np.float32)


def upsample_oracle(x, s):
    T = x.shape[-1]
    idx = np.arange(T * s) // s
    return np.ascontiguousarray(x.view(np.uint32)[..., idx]).view(np.float32)


def pad_oracle(x, pb, pe):
    w = x.view(np.uint32)
    widths = [(0, 0)] * (x.ndim - 1) + [(pb, pe)]
    return np.ascontiguousarray(np.pad(w, widths, mode='reflect')).view(np.float32)


def pad_index_formula(T, pb, pe):
    """The ONNX/Voice index map (oracles.pad): j in [-pb, T+pe) -> |j| below 0, 2(T-1)-j at or above T."""
    j = np.arange(-pb, T + pe)
    return np.where(j < 0, -j, np.where(j >= T, 2 * (T - 1) - j, j))


def rint_oracle(x):
    """IEEE roundTiesToEven in float64 (exact for every fp32 input), cast back to fp32 (exact: the result is an
    integer-valued fp32 or a pass-through)."""
    with np.errstate(invalid='ignore'):        # signalling NaNs quieten on the casts; NaN == NaN in the compare
        return np.rint(x.astype(F64)).astype(np.float32)


def rint_data(name, shape, data):
    g = rng(name)
    if data == 'durations':
        x = np.abs(g.normal(C.DURATION_MEAN, C.DURATION_SD, shape))
        ties = g.random(shape) < C.DURATION_TIE_FRACTION
        x[ties] = np.floor(x[ties]) + 0.5
        return x.astype(np.float32)
    if data == 'order_vector':
        v = np.array(C.RINT_ORDER_VECTOR, dtype=np.float32)
        return np.ascontiguousarray(np.stack([v, -v]))
    if data == 'halves':
        v = np.arange(-C.HALVES_LIMIT, C.HALVES_LIMIT, dtype=np.float64) + 0.5
        return np.ascontiguousarray(np.stack([v, v + 0.25]).astype(np.float32))
    if data == 'wide':
        lo, hi = C.WIDE_LOG2_RANGE
        return (g.standard_normal(shape) * np.exp2(g.integers(lo, hi, shape))).astype(np.float32)
    if data == 'special':
        return special_data(name, shape)
    raise ValueError(data)


def bits_equal_nan_aware(a, b):
    """fp32 bit patterns equal, except that any NaN equals any NaN (LD9)."""
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    if a.shape != b.shape:
        return False
    na, nb = np.isnan(a), np.isnan(b)
    return bool(np.array_equal(na, nb) and np.array_equal(a.view(np.uint32)[~na], b.view(np.uint32)[~nb]))


def bit_mismatches_nan_aware(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    both_nan = np.isnan(a) & np.isnan(b)
    return int(np.sum((a.view(np.uint32) != b.view(np.uint32)) & ~both_nan))


def bits_equal_strict(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return a.shape == b.shape and bool(np.array_equal(a.view(np.uint32), b.view(np.uint32)))
