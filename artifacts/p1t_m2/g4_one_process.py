"""P1T-M2 G4 (decoupling), the one-process part: production `tensor_cuda` (read-only, on PYTHONPATH after tc_signal)
and `tc_signal` import into ONE process, both work, and a tc_signal conv on torch tensors still passes the 1e-5 gate.
Also lists every CUDA runtime library mapped into the process (/proc/self/maps), to show that tc_signal and core share
one libcudart.so.12 instance (torch brings its own CUDA 13 runtime, a separate soname).

Run: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<worktree>/tc_signal:/mnt/ForgeRealm/Project-Tensor/tensor_cuda \
         python3 artifacts/p1t_m2/g4_one_process.py
Prior art: none needed (an import/link probe); no prior art known to me.
"""
import json
import os
import re
import sys

GATE_N = 1e-5


def mapped_cuda_libs():
    libs = set()
    with open('/proc/self/maps') as f:
        for line in f:
            m = re.search(r'(/\S*lib(cudart|cublas|cublasLt)\S*\.so\S*)', line)
            if m:
                libs.add(os.path.realpath(m.group(1)) + '  (mapped as ' + m.group(1) + ')')
    return sorted(libs)


def main():
    out = {'PYTHONPATH': os.environ.get('PYTHONPATH'), 'sys_path_head': sys.path[:4]}
    import tensor_cuda                                   # production core, read-only
    out['tensor_cuda_file'] = tensor_cuda.__file__
    out['tensor_cuda_C'] = getattr(getattr(tensor_cuda, '_C', None), '__file__', None)
    import tc_signal as ts
    out['tc_signal_file'] = ts.__file__
    out['tc_signal_C'] = ts._tc_signal.__file__
    import numpy as np
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(ts.__file__), '..', 'tests'))
    from tcs_oracle import conv1d_f64, conv_transpose1d_f64, normwise

    # core still works in this process
    a = tensor_cuda.tensor(np.arange(6, dtype=np.float32))
    out['core_op'] = (a * 2.0 + 1.0).numpy().tolist()

    # the default allocator: production tensor_cuda has no empty() before M1 -> RuntimeError naming set_allocator
    try:
        ts.conv1d(torch.randn(1, 4, 16, device='cuda'), torch.randn(3, 4, 3, device='cuda'))
        out['default_allocator'] = 'NO ERROR (unexpected before M1)'
    except RuntimeError as e:
        out['default_allocator'] = f'RuntimeError: {e}'

    ts.set_allocator(lambda s: torch.empty(s, device='cuda', dtype=torch.float32))
    g = torch.Generator().manual_seed(20261012)
    x = torch.randn(1, 512, 1193, generator=g).cuda()
    w = (torch.randn(512, 512, 3, generator=g) / (512 * 3) ** 0.5).cuda()
    b = torch.randn(512, generator=g).cuda()
    y = ts.conv1d(x, w, b, padding=(1, 1))
    wt = (torch.randn(512, 256, 20, generator=g) / (512 * 20) ** 0.5).cuda()
    xt = torch.randn(1, 512, 64, generator=g).cuda()
    yt = ts.conv_transpose1d(xt, wt, b[:256].contiguous(), stride=10, padding=(5, 5))
    torch.cuda.synchronize()
    n1 = normwise(y, conv1d_f64(x, w, b, stride=1, pads=(1, 1), dilation=1, groups=1))
    n2 = normwise(yt, conv_transpose1d_f64(xt, wt, b[:256], stride=10, pads=(5, 5), output_padding=0, dilation=1,
                                           groups=1))
    out['conv1d_[1,512,1193]_K3_N'] = n1
    out['conv_transpose1d_[1,512,64]_K20_s10_N'] = n2
    out['mapped_cuda_libs'] = mapped_cuda_libs()
    cudart12 = sorted({l.split('  ')[0] for l in out['mapped_cuda_libs'] if 'libcudart.so.12' in l})
    out['distinct_libcudart_so_12_files'] = cudart12
    ok = (n1 <= GATE_N and n2 <= GATE_N and len(cudart12) == 1 and 'set_allocator' in out['default_allocator']
          and out['core_op'] == [1.0, 3.0, 5.0, 7.0, 9.0, 11.0])
    out['G4_one_process'] = 'PASS' if ok else 'FAIL'
    print(json.dumps(out, indent=1))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
