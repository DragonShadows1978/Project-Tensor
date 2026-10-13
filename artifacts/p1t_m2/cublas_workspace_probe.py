"""P1T-M2: measure the device memory cuBLAS itself allocates for tc_signal's handle (the one exception to "C++ allocates
no device memory"), via cudaMemGetInfo deltas (torch.cuda.mem_get_info) around (1) handle creation and (2) the first
SGEMM. All tc_signal operands, outputs and scratch are pre-allocated by torch before the measurement.

Run: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<worktree>/tc_signal python3 artifacts/p1t_m2/cublas_workspace_probe.py
Prior art: none needed (a measurement); no prior art known to me.
"""
import json

import torch

import tc_signal as ts  # noqa: F401  (loads the module; no CUDA work at import)
from tc_signal import _tc_signal as _C


def free():
    torch.cuda.synchronize()
    return torch.cuda.mem_get_info()[0]


def main():
    x = torch.randn(1, 512, 1193, device='cuda')
    w = torch.randn(512, 512, 3, device='cuda')
    out = torch.empty(1, 512, 1193, device='cuda')
    scratch = torch.empty(512 * 3 * 1193, device='cuda')
    f0 = free()
    cfg = _C.cublas_config(torch.cuda.current_device())     # creates the handle
    f1 = free()
    xc, wc, bc, oshape, sn = _C.conv1d_plan(x, w, None, 1, 1, 1, 1, 1)
    _C.conv1d_run(xc, wc, bc, out, scratch, 1, 1, 1, 1, 1)  # first SGEMM on the handle
    f2 = free()
    xc, wc, bc, oshape, sn = _C.conv1d_plan(x, w, None, 1, 1, 1, 1, 1)
    _C.conv1d_run(xc, wc, bc, out, scratch, 1, 1, 1, 1, 1)  # second call: steady state
    f3 = free()
    mib = 1024 * 1024
    res = {'handle_config': cfg, 'build_info': _C.build_info(),
           'handle_creation_MiB': (f0 - f1) / mib, 'first_sgemm_MiB': (f1 - f2) / mib,
           'second_call_MiB': (f2 - f3) / mib, 'total_cublas_MiB': (f0 - f3) / mib,
           'note': 'cudaMemGetInfo deltas (device-wide, driver page granularity); torch held every tc_signal operand'}
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
