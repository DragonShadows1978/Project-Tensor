"""Independent float64 NumPy oracle for tc_signal.lstm (P1T-M3), plus its torch cross-check.

`lstm_onnx_f64` transcribes the ONNX LSTM operator (layout 0, default activations Sigmoid/Tanh/Tanh, no clip, no
peepholes, input_forget 0) directly in NumPy float64:
    z = x_t . W[d]^T + h . R[d]^T + Wb[d] + Rb[d]          gate blocks along 4H: i, o, f, c (ONNX)
    i, o, f = sigmoid(z_i), sigmoid(z_o), sigmoid(z_f);  c~ = tanh(z_c)
    c = f*c + i*c~;  h = o*tanh(c)
direction "reverse" (or slot 1 of "bidirectional") walks t = T-1 .. 0 and writes Y[t, d]. It shares no code with the
CUDA implementation (no cuBLAS, no fused cell, different summation structure) nor with Voice's torch oracle.

`lstm_torch_f64` is the cross-check: torch.nn.LSTM in float64 (CPU) after converting the ONNX layout (gate order
i, o, f, c -> torch's i, f, g, o; bias halves Wb -> bias_ih, Rb -> bias_hh; slot 1 -> the `_reverse` parameters;
torch's output [T, N, D*H] -> [T, D, N, H]). Two independent formulations agreeing to ~1e-15 is the oracle's own check.

Metric: normwise N = ||y - o||_inf / ||o||_inf (Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed. 2002;
the same metric as Voice's runner, common.normwise).

Prior art: ONNX LSTM operator spec (onnx/docs/Operators.md); Hochreiter & Schmidhuber 1997; Gers, Schmidhuber &
Cummins 2000; the sign-split stable sigmoid (standard practice). The ONNX -> torch gate conversion is the same idea as
Voice's `oracles.onnx_to_torch_gates` (written independently here). Ours: only the transcription.
"""
import contextlib

import numpy as np

from tcs_lstm_census import ORACLE_BLAS_THREADS

F64 = np.float64


def _blas_threads():
    """Caps BLAS threads while the oracle runs (speed only), when threadpoolctl is installed; otherwise no cap."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return contextlib.nullcontext()
    return threadpool_limits(limits=ORACLE_BLAS_THREADS, user_api='blas')
GATES_ONNX = ('i', 'o', 'f', 'c')
_DIRS = {'forward': ('forward',), 'reverse': ('reverse',), 'bidirectional': ('forward', 'reverse')}


def sigmoid_f64(z):
    """Stable logistic in float64: exp only of a non-positive argument."""
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def _f64(a):
    return None if a is None else np.asarray(a, dtype=F64)


def lstm_onnx_f64(x, W, R, B=None, *, direction, hidden_size, initial_h=None, initial_c=None, return_zmax=False):
    """Returns (Y [T, D, N, H], Y_h [D, N, H], Y_c [D, N, H]) in float64 (and max |z| over every gate pre-activation
    when return_zmax)."""
    with _blas_threads():
        return _lstm_onnx_f64(x, W, R, B, direction=direction, hidden_size=hidden_size, initial_h=initial_h,
                              initial_c=initial_c, return_zmax=return_zmax)


def _lstm_onnx_f64(x, W, R, B, *, direction, hidden_size, initial_h, initial_c, return_zmax):
    x, W, R, B = _f64(x), _f64(W), _f64(R), _f64(B)
    T, N, I = x.shape
    H = int(hidden_size)
    dirs = _DIRS[direction]
    D = len(dirs)
    assert W.shape == (D, 4 * H, I) and R.shape == (D, 4 * H, H)
    if B is None:
        B = np.zeros((D, 8 * H), F64)
    h0 = np.zeros((D, N, H), F64) if initial_h is None else _f64(initial_h)
    c0 = np.zeros((D, N, H), F64) if initial_c is None else _f64(initial_c)
    blk = {g: slice(k * H, (k + 1) * H) for k, g in enumerate(GATES_ONNX)}
    Y = np.zeros((T, D, N, H), F64)
    Yh = np.zeros((D, N, H), F64)
    Yc = np.zeros((D, N, H), F64)
    zmax = 0.0
    for d, dname in enumerate(dirs):
        Wb, Rb = B[d, :4 * H], B[d, 4 * H:]
        h, c = h0[d].copy(), c0[d].copy()                      # [N, H]
        steps = range(T) if dname == 'forward' else range(T - 1, -1, -1)
        for t in steps:
            z = x[t] @ W[d].T + h @ R[d].T + Wb + Rb             # [N, 4H]
            zmax = max(zmax, float(np.abs(z).max()))
            i = sigmoid_f64(z[:, blk['i']])
            o = sigmoid_f64(z[:, blk['o']])
            f = sigmoid_f64(z[:, blk['f']])
            g = np.tanh(z[:, blk['c']])
            c = f * c + i * g
            h = o * np.tanh(c)
            Y[t, d] = h
        Yh[d], Yc[d] = h, c
    return (Y, Yh, Yc, zmax) if return_zmax else (Y, Yh, Yc)


def onnx_to_torch_gate_rows(a):
    """ONNX gate blocks (i, o, f, c) along axis 0 -> torch.nn.LSTM's (i, f, g, o)."""
    H = a.shape[0] // 4
    b = {g: a[k * H:(k + 1) * H] for k, g in enumerate(GATES_ONNX)}
    return np.concatenate([b['i'], b['f'], b['c'], b['o']], axis=0)


def lstm_torch_f64(x, W, R, B=None, *, direction, hidden_size, initial_h=None, initial_c=None):
    """The same LSTM through torch.nn.LSTM (float64, CPU) after the ONNX -> torch layout conversion."""
    import torch
    x, W, R, B = _f64(x), _f64(W), _f64(R), _f64(B)
    T, N, I = x.shape
    H = int(hidden_size)
    dirs = _DIRS[direction]
    D = len(dirs)
    if B is None:
        B = np.zeros((D, 8 * H), F64)
    h0 = np.zeros((D, N, H), F64) if initial_h is None else _f64(initial_h)
    c0 = np.zeros((D, N, H), F64) if initial_c is None else _f64(initial_c)
    if direction == 'reverse':
        # torch has no reverse-only LSTM: a forward LSTM on the time-flipped input, flipped back, is the reverse pass.
        xs = x[::-1].copy()
    else:
        xs = x
    m = torch.nn.LSTM(I, H, num_layers=1, bias=True, batch_first=False,
                      bidirectional=(direction == 'bidirectional')).to(torch.float64)
    with torch.no_grad():
        for d in range(D):
            suf = '_reverse' if d == 1 else ''
            getattr(m, 'weight_ih_l0' + suf).copy_(torch.from_numpy(onnx_to_torch_gate_rows(W[d])))
            getattr(m, 'weight_hh_l0' + suf).copy_(torch.from_numpy(onnx_to_torch_gate_rows(R[d])))
            getattr(m, 'bias_ih_l0' + suf).copy_(torch.from_numpy(onnx_to_torch_gate_rows(B[d, :4 * H])))
            getattr(m, 'bias_hh_l0' + suf).copy_(torch.from_numpy(onnx_to_torch_gate_rows(B[d, 4 * H:])))
        out, (hn, cn) = m(torch.from_numpy(np.ascontiguousarray(xs)), (torch.from_numpy(h0), torch.from_numpy(c0)))
    Y = out.numpy().reshape(T, N, D, H).transpose(0, 2, 1, 3)
    if direction == 'reverse':
        Y = Y[::-1]
    return np.ascontiguousarray(Y), hn.numpy(), cn.numpy()


def normwise(y, o):
    """N over float64 values; o == 0 everywhere passes only if y is exactly 0 (Voice common.normwise rule)."""
    y = np.asarray(y, dtype=F64)
    o = np.asarray(o, dtype=F64)
    den = float(np.abs(o).max())
    num = float(np.abs(y - o).max())
    if den == 0.0:
        return 0.0 if num == 0.0 else float('inf')
    return num / den


def ulp_error(y32, o64):
    """Elementwise |y - o| in units of the fp32 ulp of o's binade: ulp(o) = 2^(floor(log2|o|) - 23), floored at the
    fp32 subnormal spacing 2^-149 (no float32 rounding of o first, so no factor-2 slack at powers of two)."""
    y = np.asarray(y32, dtype=F64)
    o = np.asarray(o64, dtype=F64)
    with np.errstate(divide='ignore'):
        e = np.floor(np.log2(np.abs(o)))
    e = np.maximum(e, -126.0)
    spacing = np.ldexp(1.0, (e - 23.0).astype(np.int64))
    return np.abs(y - o) / spacing
