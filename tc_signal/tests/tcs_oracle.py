"""float64 CPU oracles for tc_signal's own tests (torch, same process), plus the normwise metric.

conv1d: ONNX Conv 1-D = zero-pad the time axis by (pad_begin, pad_end), then torch's conv1d with padding 0.
conv_transpose1d: ONNX ConvTranspose 1-D = torch's conv_transpose1d with padding 0 and output_padding 0 (the "full"
output, length stride*(T-1) + dilation*(K-1) + 1), zero-extended at the end when output_padding reaches past it, then
the window [pad_begin, pad_begin + L) with L = stride*(T-1) + output_padding + dilation*(K-1) + 1 - pb - pe, then bias.
This construction is independent of the CUDA kernels' formulas (im2col / gather); test_oracle_matches_torch_native
checks it against torch's own symmetric-padding path where both apply.

Normwise N = ||y - o||_inf / ||o||_inf (Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed. 2002; the same
metric as Voice's runner, common.normwise). Prior art for the oracles: ONNX Conv/ConvTranspose specs; torch.nn.functional.
"""
import torch
import torch.nn.functional as F

F64 = torch.float64


def _cpu64(t):
    return None if t is None else t.detach().to(device='cpu', dtype=F64)


def conv1d_f64(x, w, b, *, stride, pads, dilation, groups):
    xc, wc, bc = _cpu64(x), _cpu64(w), _cpu64(b)
    xp = F.pad(xc, (int(pads[0]), int(pads[1])))
    return F.conv1d(xp, wc, bc, stride=stride, padding=0, dilation=dilation, groups=groups)


def conv_transpose1d_f64(x, w, b, *, stride, pads, output_padding, dilation, groups):
    xc, wc, bc = _cpu64(x), _cpu64(w), _cpu64(b)
    T, K = xc.shape[-1], wc.shape[-1]
    pb, pe = int(pads[0]), int(pads[1])
    full = F.conv_transpose1d(xc, wc, None, stride=stride, padding=0, output_padding=0, dilation=dilation,
                              groups=groups)
    L = stride * (T - 1) + output_padding + dilation * (K - 1) + 1 - pb - pe
    need = pb + L
    if need > full.shape[-1]:
        full = F.pad(full, (0, need - full.shape[-1]))
    y = full[:, :, pb:pb + L].clone()
    if bc is not None:
        y += bc.view(1, -1, 1)
    return y


def normwise(y, o):
    """N over the float64 values; o == 0 everywhere passes only if y is exactly 0 (Voice common.normwise rule)."""
    y = y.detach().to(device='cpu', dtype=F64)
    den = o.abs().max().item()
    num = (y - o).abs().max().item()
    if den == 0.0:
        return 0.0 if num == 0.0 else float('inf')
    return num / den


def bits_equal(a, b):
    """Bitwise equality of two fp32 tensors (on any device)."""
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))


def bit_mismatches(a, b):
    return int((a.contiguous().view(torch.int32) != b.contiguous().view(torch.int32)).sum().item())
