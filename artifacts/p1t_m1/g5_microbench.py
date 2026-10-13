"""P1T-M1 G5 (informational) microbench + Amendment 1 timing. Mine.

Driver mode (no args): holds nothing itself; run it under the GPU lock. For each of RUNS runs it starts one FRESH
python process per engine, alternating which engine goes first, and prints per-run seconds and the medians.

Engines: production's installed engine (/mnt/ForgeRealm/Project-Tensor/tensor_cuda, read-only; PYTHONDONTWRITEBYTECODE
keeps it unwritten) and this worktree's build.

Workloads (identical code in both engines; production has neither fill_ nor empty, so the cross-engine pair uses
calls both engines have):
  * alloc_free: N cycles of `y = x * 1.0; del y` on a 256-element fp32 tensor (requires_grad False): one Storage
    allocation, one one-block ew_scalar launch, one Storage free, in the default pooling state (raw
    cudaMalloc/cudaFree). This is the Storage constructor/destructor path P1T-M1 touched. (r1 used `x.float()`, which
    allocates nothing: ops.cpp cast() returns its input when the dtype already matches. r1 is void for this row.)
  * alloc_free_pooled: the same cycles after `set_alloc_pooling(True)` (cudaMallocAsync/cudaFreeAsync on the legacy
    stream), the mode inference transients run in.
  * fill_sized: N calls of `_C.scale_(t, 1.0)` on a 256-element fp32 tensor: an in-place one-block launch of the same
    shape fill_ uses (nblk(256) x 256 threads), then one synchronize.
  * fill_ (this build only, no production counterpart): N calls of `t.fill_(1.0)`, then one synchronize.
Amendment 1 (this build only, separate fresh process): 20 timed runs (after 2 warm-ups) of `tc.empty(1, 512, 7160)`
and of `tc.zeros(1, 512, 7160)`, each followed by `tc.synchronize()` inside the timer and freed outside it; medians.

Prior art: interleaved A/B in fresh processes with medians is ordinary benchmarking practice (House Rules section 4
measurement laws); no algorithmic prior art claimed.
"""
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time

N = 100_000
RUNS = 5
SMALL = 256
EMPTY_SHAPE = (1, 512, 7160)
EMPTY_RUNS = 20
EMPTY_WARMUP = 2
PROD = '/mnt/ForgeRealm/Project-Tensor/tensor_cuda'
MINE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'tensor_cuda'))


def child(kind):
    import tensor_cuda as tc
    so = tc._C.__file__
    out = {'engine_so': so, 'so_sha256': hashlib.sha256(open(so, 'rb').read()).hexdigest()}
    if kind == 'bench':
        x = tc.zeros(SMALL)
        for _ in range(100):                                   # warm-up
            y = x * 1.0; del y
            tc._C.scale_(x, 1.0)
        tc.synchronize()
        t0 = time.perf_counter()
        for _ in range(N):
            y = x * 1.0
            del y
        tc.synchronize()
        out['alloc_free_s'] = time.perf_counter() - t0
        tc.set_alloc_pooling(True)
        for _ in range(100):
            y = x * 1.0; del y
        tc.synchronize()
        t0 = time.perf_counter()
        for _ in range(N):
            y = x * 1.0
            del y
        tc.synchronize()
        out['alloc_free_pooled_s'] = time.perf_counter() - t0
        tc.set_alloc_pooling(False)
        t0 = time.perf_counter()
        for _ in range(N):
            tc._C.scale_(x, 1.0)
        tc.synchronize()
        out['fill_sized_scale_s'] = time.perf_counter() - t0
        if hasattr(x, 'fill_'):
            for _ in range(100):
                x.fill_(1.0)
            tc.synchronize()
            t0 = time.perf_counter()
            for _ in range(N):
                x.fill_(1.0)
            tc.synchronize()
            out['fill__s'] = time.perf_counter() - t0
    else:                                                      # amendment 1 timing
        res = {}
        for name, fn in (('empty', tc.empty), ('zeros', tc.zeros)):
            ts = []
            for i in range(EMPTY_WARMUP + EMPTY_RUNS):
                t0 = time.perf_counter()
                a = fn(*EMPTY_SHAPE)
                tc.synchronize()
                dt = time.perf_counter() - t0
                assert tuple(a.shape) == EMPTY_SHAPE and a.dtype == 'float32'
                del a
                tc.synchronize()
                if i >= EMPTY_WARMUP:
                    ts.append(dt)
            res[name] = ts
        out['amendment1'] = res
    print('RESULT ' + json.dumps(out))


def spawn(engine_root, kind):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=f'{engine_root}/tensor_cuda:{engine_root}')
    p = subprocess.run([sys.executable, os.path.abspath(__file__), '--child', kind], env=env, capture_output=True,
                       text=True, cwd='/tmp', timeout=300)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-2000:])
    line = [l for l in p.stdout.splitlines() if l.startswith('RESULT ')][-1]
    return json.loads(line[len('RESULT '):])


def main():
    rows = {'prod': [], 'mine': []}
    for r in range(RUNS):
        order = ('prod', 'mine') if r % 2 == 0 else ('mine', 'prod')
        for eng in order:
            res = spawn(PROD if eng == 'prod' else MINE, 'bench')
            rows[eng].append(res)
            print(f'run {r + 1} {eng}: ' + ', '.join(f'{k}={v:.4f}' for k, v in res.items() if k.endswith('_s')) +
                  f'  [{res["so_sha256"][:8]}]', flush=True)
    print('--- medians of 5 (seconds per 100,000 calls)')
    summary = {}
    for key in ('alloc_free_s', 'alloc_free_pooled_s', 'fill_sized_scale_s', 'fill__s'):
        mp = [x[key] for x in rows['prod'] if key in x]
        mm = [x[key] for x in rows['mine'] if key in x]
        if not mm:
            continue
        line = f'{key}: mine median {statistics.median(mm):.4f} s ({1e6 * statistics.median(mm) / N:.3f} us/call)'
        if mp:
            slow = statistics.median(mm) / statistics.median(mp) - 1.0
            line += (f'; prod median {statistics.median(mp):.4f} s ({1e6 * statistics.median(mp) / N:.3f} us/call); '
                     f'mine/prod - 1 = {100 * slow:+.2f}%' + ('  FLAG (> 2%)' if slow > 0.02 else ''))
            summary[key] = {'prod': mp, 'mine': mm, 'slowdown_pct': 100 * slow}
        else:
            summary[key] = {'mine': mm}
        print(line)
    am = spawn(MINE, 'amendment1')['amendment1']
    me, mz = statistics.median(am['empty']), statistics.median(am['zeros'])
    print(f'Amendment 1: [1, 512, 7160] fp32, median of {EMPTY_RUNS} (after {EMPTY_WARMUP} warm-ups): '
          f'empty {1e3 * me:.4f} ms, zeros {1e3 * mz:.4f} ms, zeros/empty = {mz / me:.1f}x')
    summary['amendment1'] = am
    summary['engines'] = {'prod': rows['prod'][0]['so_sha256'], 'mine': rows['mine'][0]['so_sha256']}
    print('JSON ' + json.dumps(summary))


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == '--child':
        child(sys.argv[2])
    else:
        main()
