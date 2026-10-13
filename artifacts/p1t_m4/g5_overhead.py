"""P1T-M4 G5 (informational): microseconds per call of the six tc_signal M4 ops, TensorCUDA tensors as the producer.

No torch in this process. Inputs are uploaded once with tensor_cuda.tensor. Outputs come from tc_signal's default
allocator (tensor_cuda.empty), except in the "wall, out=" column. CUDA events come from libcudart.so.12 through ctypes:
the same runtime that TensorCUDA and tc_signal link, and the same legacy default stream (0).

Per op and shape, after WARMUP calls, the median of CALLS calls in four separate loops:
  wall        host perf_counter around the call, no sync inside the timed region (synced between calls, untimed);
  wall, out=  the same, with a preallocated output (no allocator call);
  GPU event   cudaEventRecord(start, 0); call; cudaEventRecord(end, 0); cudaEventSynchronize(end); elapsed;
  wall+sync   perf_counter around the call plus cudaDeviceSynchronize;
  GPU event, out=  the GPU-event loop with a preallocated output (launch + kernel, no allocation);
  alloc       tensor_cuda.empty(out_shape) alone (the default allocator's share of `wall`).
Evidence class: micro-benchmark (launch overhead and kernel time at these shapes; not model speed).

Prior art: CUDA event timing (CUDA C Programming Guide, "Events"); median-of-N micro-benchmarking is standard practice.
Ours: the harness. No prior art claimed.
"""
import ctypes
import json
import statistics
import sys
import time

import numpy as np

import tensor_cuda as tc
import tc_signal as ts

WARMUP = 20          # order G5
CALLS = 200          # order G5
SEED = 20261012

cudart = ctypes.CDLL('libcudart.so.12')
for fn in ('cudaEventCreate', 'cudaEventRecord', 'cudaEventSynchronize', 'cudaEventElapsedTime',
           'cudaDeviceSynchronize', 'cudaRuntimeGetVersion'):
    getattr(cudart, fn).restype = ctypes.c_int
cudart.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
cudart.cudaEventSynchronize.argtypes = [ctypes.c_void_p]
cudart.cudaEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]


def ck(rc, what):
    if rc != 0:
        raise RuntimeError(f'{what} failed: cudaError {rc}')


def event():
    e = ctypes.c_void_p()
    ck(cudart.cudaEventCreate(ctypes.byref(e)), 'cudaEventCreate')
    return e


def sync():
    ck(cudart.cudaDeviceSynchronize(), 'cudaDeviceSynchronize')


rng = np.random.default_rng(SEED)


def up(*shape):
    return tc.tensor(rng.standard_normal(shape).astype(np.float32) * 3.0, dtype='float32')


def cases():
    """(op, shape label, call(out=None) -> output, out shape)."""
    out = []
    for size, xs in (('tiny', (1, 8, 16)), ('kokoro', (1, 512, 7160))):
        x, s, b = up(*xs), up(xs[1]), up(xs[1])
        out.append(('instance_norm1d', size, xs, lambda o=None, x=x, s=s, b=b: ts.instance_norm1d(x, s, b, out=o), xs))
    for size, xs in (('tiny', (1, 8, 16)), ('kokoro', (1, 82, 768))):
        x, s, b = up(*xs), up(xs[-1]), up(xs[-1])
        out.append(('layer_norm', size, xs,
                    lambda o=None, x=x, s=s, b=b: ts.layer_norm(x, s, b, axis=-1, eps=1e-12, out=o), xs))
    for size, xs, g in (('tiny', (1, 8, 16), 2), ('kokoro', (1, 256, 3000), 8)):
        x, s, b = up(*xs), up(xs[1]), up(xs[1])
        out.append((f'group_norm (G={g})', size, xs,
                    lambda o=None, x=x, s=s, b=b, g=g: ts.group_norm(x, g, s, b, out=o), xs))
    for size, xs, k in (('tiny', (1, 8, 16), 2), ('kokoro', (1, 1, 1193), 300)):
        x = up(*xs)
        out.append((f'upsample_nearest1d (x{k})', size, xs, lambda o=None, x=x, k=k: ts.upsample_nearest1d(x, k, out=o),
                    xs[:-1] + (xs[-1] * k,)))
    for size, xs in (('tiny', (1, 8, 16)), ('kokoro', (1, 22, 7160))):
        x = up(*xs)
        out.append(('reflection_pad1d (1, 0)', size, xs, lambda o=None, x=x: ts.reflection_pad1d(x, (1, 0), out=o),
                    xs[:-1] + (xs[-1] + 1,)))
    for size, xs in (('tiny', (1, 8, 16)), ('kokoro', (1, 82))):
        x = up(*xs)
        out.append(('rint', size, xs, lambda o=None, x=x: ts.rint(x, out=o), xs))
    return out


def med_us(v):
    return statistics.median(v) * 1e6


def main():
    ts.set_allocator(None)                     # the default allocator: tensor_cuda.empty
    v = ctypes.c_int()
    cudart.cudaRuntimeGetVersion(ctypes.byref(v))
    e0, e1 = event(), event()
    rows = []
    for op, size, xs, call, oshape in cases():
        preout = tc.empty(*oshape, dtype='float32')
        for _ in range(WARMUP):
            call()
            call(preout)
        sync()
        wall, wall_out, gpu, wall_sync, gpu_out, alloc = [], [], [], [], [], []
        for _ in range(CALLS):
            t0 = time.perf_counter()
            y = call()
            wall.append(time.perf_counter() - t0)
            sync()
            del y
        for _ in range(CALLS):
            t0 = time.perf_counter()
            call(preout)
            wall_out.append(time.perf_counter() - t0)
            sync()
        for _ in range(CALLS):
            ck(cudart.cudaEventRecord(e0, None), 'cudaEventRecord')
            y = call()
            ck(cudart.cudaEventRecord(e1, None), 'cudaEventRecord')
            ck(cudart.cudaEventSynchronize(e1), 'cudaEventSynchronize')
            ms = ctypes.c_float()
            ck(cudart.cudaEventElapsedTime(ctypes.byref(ms), e0, e1), 'cudaEventElapsedTime')
            gpu.append(ms.value * 1e-3)
            del y
        for _ in range(CALLS):
            t0 = time.perf_counter()
            y = call()
            sync()
            wall_sync.append(time.perf_counter() - t0)
            del y
        for _ in range(CALLS):
            ck(cudart.cudaEventRecord(e0, None), 'cudaEventRecord')
            call(preout)
            ck(cudart.cudaEventRecord(e1, None), 'cudaEventRecord')
            ck(cudart.cudaEventSynchronize(e1), 'cudaEventSynchronize')
            ms = ctypes.c_float()
            ck(cudart.cudaEventElapsedTime(ctypes.byref(ms), e0, e1), 'cudaEventElapsedTime')
            gpu_out.append(ms.value * 1e-3)
        for _ in range(CALLS):
            t0 = time.perf_counter()
            t = tc.empty(*oshape, dtype='float32')
            alloc.append(time.perf_counter() - t0)
            del t
            sync()
        rows.append({'op': op, 'size': size, 'x_shape': list(xs), 'out_shape': list(oshape),
                     'wall_us': med_us(wall), 'wall_out_us': med_us(wall_out), 'gpu_event_us': med_us(gpu),
                     'wall_sync_us': med_us(wall_sync), 'gpu_event_out_us': med_us(gpu_out),
                     'alloc_us': med_us(alloc)})
    print(f'cudart {v.value}; tc_signal {ts.__file__}; tensor_cuda {tc.__file__}; warmup {WARMUP}, calls {CALLS}, median')
    print('| op | size | x | wall us | GPU event us | wall+sync us | wall, out= us | GPU event, out= us | '
          'tensor_cuda.empty alone us |')
    print('|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        print(f"| {r['op']} | {r['size']} | {r['x_shape']} | {r['wall_us']:.1f} | {r['gpu_event_us']:.1f} | "
              f"{r['wall_sync_us']:.1f} | {r['wall_out_us']:.1f} | {r['gpu_event_out_us']:.1f} | {r['alloc_us']:.1f} |")
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'w') as f:
            json.dump(rows, f, indent=1)
            f.write('\n')


if __name__ == '__main__':
    main()
