"""P1T-FIX1 G4 supplement (informational): like g4_bench.py, but every output is preallocated (out=...), so the timing
excludes the allocator (tensor_cuda.empty -> cudaMalloc/cudaFree, which syncs) and isolates tc_signal's own per-call
path, where the F01 bookkeeping (drain query, event record, pending push) lands. conv1d still allocates its im2col
scratch through the default allocator; conv1d_k1 (w [8, 8, 1], no pads) is the direct path with no scratch.
Median of 200 calls after 20 warm-ups; host = the call only. Prior art: none needed."""
import json
import statistics
import sys
import time

import numpy as np

import tc_signal as ts
import tensor_cuda as tc

WARM, N = 20, 200
rng = np.random.default_rng(20261012)
x = tc.tensor(rng.normal(0, 1, (1, 8, 16)).astype(np.float32))
w3 = tc.tensor(rng.normal(0, 0.1, (8, 8, 3)).astype(np.float32))
w1 = tc.tensor(rng.normal(0, 0.1, (8, 8, 1)).astype(np.float32))
r = tc.tensor(rng.normal(0, 3, (1, 82)).astype(np.float32))
y16 = tc.empty(1, 8, 16)
y82 = tc.empty(1, 82)
ops = {'conv1d_1x8x16_out': lambda: ts.conv1d(x, w3, padding=1, out=y16),
       'conv1d_k1_1x8x16_out': lambda: ts.conv1d(x, w1, out=y16),
       'rint_1x82_out': lambda: ts.rint(r, out=y82)}


def bench(fn):
    for _ in range(WARM):
        fn()
    tc.synchronize()
    t = []
    for _ in range(N):
        t0 = time.perf_counter_ns()
        fn()
        t.append(time.perf_counter_ns() - t0)
    tc.synchronize()
    return statistics.median(t) / 1000.0


res = {'label': sys.argv[1], 'tc_signal_so': ts._C.__file__}
for name, fn in ops.items():
    res[name + '_host_us'] = round(bench(fn), 2)
print(json.dumps(res))
