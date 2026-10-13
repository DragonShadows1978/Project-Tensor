"""P1T-M2 G5 (informational): per-call overhead of tc_signal.conv1d, median of 200 calls after 20 warm-ups.

Shapes (order G5): x [1, 1, 8] K 3 and x [1, 512, 1193] K 3, C_out = C_in, padding (1, 1) (Kokoro's K3 attribute
set; the order leaves C_out and pads open). Operands are torch CUDA fp32 tensors; the allocator is torch.empty.

Per call:
 * wall_us     : host time of ts.conv1d(...) from Python, no sync (wrapper + capsules + plan + allocation + launches);
 * gpu_us      : CUDA-event time between events recorded right before and right after the call on the legacy default
                 stream (kernels plus any GPU idle gaps while the host is still launching);
 * wall_sync_us: host time of the call plus torch.cuda.synchronize() (end-to-end latency);
 * breakdown (host, same loop, mirroring ts.conv1d): plan (3 x __dlpack__(stream=1) + peek + sizing), alloc (out +
   scratch via torch.empty), run (consume 5 capsules incl. out/scratch __dlpack__, validate, launch, deleters);
 * one_dlpack_us: one torch `x.__dlpack__(stream=1, max_version=(1, 3))` call (the per-operand capsule cost; the
   capsule is dropped unconsumed, so torch's own capsule destructor runs).
Reference (informational only): torch.nn.functional.conv1d on the same tensors (cuDNN; torch's defaults, which allow
TF32 in cuDNN convolutions, so it is a speed reference, not a numerics peer).

Prior art: CUDA-event timing and median-of-N after warm-up are standard GPU benchmarking practice (no single source).
"""
import json
import statistics
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

import tc_signal as ts
from tc_signal import _tc_signal as _C

WARMUP = 20
ITERS = 200
SHAPES = [((1, 1, 8), 1, 3), ((1, 512, 1193), 512, 3)]    # (x shape, C_out, K)
PADS = (1, 1)


def med(v):
    return statistics.median(v)


def bench(fn):
    """Returns per-call lists: wall (no sync), gpu (events), wall+sync."""
    wall, gpu, wall_sync = [], [], []
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for i in range(WARMUP + ITERS):
        torch.cuda.synchronize()
        e0.record()
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        e1.record()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        if i >= WARMUP:
            wall.append((t1 - t0) * 1e6)
            gpu.append(e0.elapsed_time(e1) * 1e3)
            wall_sync.append((t2 - t0) * 1e6)
    return wall, gpu, wall_sync


def breakdown(x, w, b):
    plan, alloc, run = [], [], []
    for i in range(WARMUP + ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        xc, wc, bc, oshape, sn = _C.conv1d_plan(x, w, b, 1, PADS[0], PADS[1], 1, 1)
        t1 = time.perf_counter()
        out = torch.empty(tuple(oshape), device='cuda')
        scratch = torch.empty((sn,), device='cuda') if sn else None
        t2 = time.perf_counter()
        _C.conv1d_run(xc, wc, bc, out, scratch, 1, PADS[0], PADS[1], 1, 1)
        t3 = time.perf_counter()
        torch.cuda.synchronize()
        if i >= WARMUP:
            plan.append((t1 - t0) * 1e6)
            alloc.append((t2 - t1) * 1e6)
            run.append((t3 - t2) * 1e6)
    return plan, alloc, run


def one_dlpack(x):
    v = []
    for i in range(WARMUP + ITERS):
        t0 = time.perf_counter()
        c = x.__dlpack__(stream=1, max_version=(1, 3))
        t1 = time.perf_counter()
        del c
        if i >= WARMUP:
            v.append((t1 - t0) * 1e6)
    return v


def main():
    ts.set_allocator(lambda s: torch.empty(s, device='cuda', dtype=torch.float32))
    smi = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,clocks.sm,utilization.gpu,memory.used',
                          '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
    rows = []
    for xs, cout, k in SHAPES:
        x = torch.randn(*xs, device='cuda')
        w = torch.randn(cout, xs[1], k, device='cuda') / (xs[1] * k) ** 0.5
        b = torch.randn(cout, device='cuda')
        tw, tg, ts_ = bench(lambda: ts.conv1d(x, w, b, padding=PADS))
        rw, rg, rs = bench(lambda: F.conv1d(x, w, b, padding=PADS[0]))
        pl, al, ru = breakdown(x, w, b)
        dl = one_dlpack(x)
        rows.append({
            'x': list(xs), 'w': [cout, xs[1], k], 'pads': list(PADS),
            'tc_signal': {'wall_us': med(tw), 'gpu_us': med(tg), 'wall_sync_us': med(ts_),
                          'plan_us': med(pl), 'alloc_us': med(al), 'run_us': med(ru), 'one_dlpack_us': med(dl)},
            'torch_F_conv1d_reference': {'wall_us': med(rw), 'gpu_us': med(rg), 'wall_sync_us': med(rs)},
        })
    res = {'warmup': WARMUP, 'iters': ITERS, 'statistic': 'median', 'nvidia_smi_before': smi,
           'torch': torch.__version__, 'rows': rows}
    print(json.dumps(res, indent=1))
    print()
    print('| x | w | tc_signal wall us | tc_signal GPU us | tc_signal wall+sync us | plan / alloc / run us '
          '| one __dlpack__ us | torch F.conv1d wall / GPU / wall+sync us |')
    print('|---|---|---|---|---|---|---|---|')
    for r in rows:
        t, q = r['tc_signal'], r['torch_F_conv1d_reference']
        print(f"| {r['x']} | {r['w']} | {t['wall_us']:.1f} | {t['gpu_us']:.1f} | {t['wall_sync_us']:.1f} | "
              f"{t['plan_us']:.1f} / {t['alloc_us']:.1f} / {t['run_us']:.1f} | {t['one_dlpack_us']:.2f} | "
              f"{q['wall_us']:.1f} / {q['gpu_us']:.1f} / {q['wall_sync_us']:.1f} |")
    return 0


if __name__ == '__main__':
    sys.exit(main())
