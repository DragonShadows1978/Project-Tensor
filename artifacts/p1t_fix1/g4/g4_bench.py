"""P1T-FIX1 G4 (informational): tc_signal per-call wall time, median of 200 calls after 20 warm-ups, for conv1d on
x [1, 8, 16] (w [8, 8, 3], padding 1: the im2col path, output + scratch from the default allocator tensor_cuda.empty)
and rint on x [1, 82], TensorCUDA tensors as operands (as Voice's adapter). Two timings per op:
  host  = perf_counter_ns around the call only (asynchronous launch; this is where the F01 bookkeeping lands);
  sync  = the call followed by tensor_cuda.synchronize() (a measurement-only device sync, not on the product path).
Prints one JSON line. Prior art: none needed (a stopwatch). The build under test is whatever PYTHONPATH selects."""
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
w = tc.tensor(rng.normal(0, 0.1, (8, 8, 3)).astype(np.float32))
r = tc.tensor(rng.normal(0, 3, (1, 82)).astype(np.float32))
ops = {'conv1d_1x8x16': lambda: ts.conv1d(x, w, padding=1), 'rint_1x82': lambda: ts.rint(r)}


def bench(fn, sync):
    for _ in range(WARM):
        fn()
    tc.synchronize()
    t = []
    for _ in range(N):
        t0 = time.perf_counter_ns()
        fn()
        if sync:
            tc.synchronize()
        t.append(time.perf_counter_ns() - t0)
    tc.synchronize()
    return statistics.median(t) / 1000.0


res = {'label': sys.argv[1], 'tc_signal_so': ts._C.__file__, 'core_so': tc._C.__file__}
for name, fn in ops.items():
    res[name + '_host_us'] = round(bench(fn, False), 2)
    res[name + '_sync_us'] = round(bench(fn, True), 2)
st = getattr(ts._C, 'pending_stats', None)
res['pending_stats'] = dict(st()) if st else None
print(json.dumps(res))
