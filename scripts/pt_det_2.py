#!/usr/bin/env python3
"""PT-DET-2 gather gates. Import is CPU-only; GPU entry requires the lead guard.

Prior art: PT-DET-1/CC46 (2026) repeated calls, FP64 NumPy scatter reference,
interleaved timing and content receipts, reused. Ours: true loss vs colliding
gather cases, shared-family coverage. Kernel prior art: deterministic_gather.h.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np
import pt_det_1 as d

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/pt_det_2'
REG = ART / 'REGISTRATION.json'
REG_SHA = 'f39756c6bf261d4be5595d255797ad81cb7f2b21c329afe3600a6f68e1d49675'


def registration():
    if d.sha(REG) != REG_SHA:
        raise ValueError('PT_DET_2 registration drift')
    r = json.loads(REG.read_text())
    for item in (r['order'], r['inherited_registration']):
        if d.sha(item['path']) != item['sha256']:
            raise ValueError('PT_DET_2 order/inherited registration drift')
    finding = r['source_finding']
    if d.sha(finding['path']) != finding['sha256']:
        raise ValueError('PT_DET_2 loss source drift')
    return r


def reference(shape, dim, index, grad):
    # Prior art: NumPy add.at scatter semantics. Independent coordinate-grid
    # reference, not the C++ flattened mapping or sorted reduction algorithm.
    axes = list(np.indices(index.shape, dtype=np.int64))
    axes[dim % len(shape)] = index
    out = np.zeros(shape, np.float64)
    np.add.at(out, tuple(axes), np.asarray(grad, np.float64))
    return out


def cases():
    r = registration()['gather_gate']
    batches, source = d.batches()
    if source['kind'] != 'real_CC46':
        raise d.Blocked('PT_DET_2 registered real loss-target archive absent')
    with np.load(source['path'], allow_pickle=False) as archive:
        targets = [(name, np.ascontiguousarray(archive[name][1:], dtype=np.int64)) for name, _ in batches]
    for i, (name, target) in enumerate(targets):
        if target.shape != (4096,) or np.any(target < 0) or np.any(target >= 8192):
            raise ValueError('invalid loss targets')
        for k in (1, 4):
            index = np.repeat(target.reshape(1, 4096, 1), k, axis=2)
            grad = np.random.default_rng(r['seed'] + i * 10 + k).standard_normal(index.shape, dtype=np.float32)
            if k == 4:
                grad[0, ::8, :] = [1e8, 1, -1e8, 2]
            yield dict(name=f'{name}_k{k}', batch=name, shape=r['shape'], dim=r['dim'],
                       kind='true_loss' if k == 1 else 'same_row_collisions',
                       repeated_class_fraction=1 - len(np.unique(target))/len(target),
                       duplicate_destination_fraction=1 - 1/k, source=source,
                       index=index, grad=grad)


def compile_cpu(out):
    scratch = out / 'compiler_tmp'; scratch.mkdir()
    library = out / 'gather_cpu_bridge.so'
    command = ['g++', '-std=c++17', '-O2', '-shared', '-fPIC', '-I' + str(ROOT/'tensor_cuda/include'),
               str(ROOT/'tests/pt_det_2_cpu_bridge.cpp'), '-o', str(library)]
    subprocess.run(command, check=True, timeout=60,
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES='', TMPDIR=str(scratch)))
    fn = ctypes.CDLL(str(library)).pt_det_gather_cpu
    fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 2
    fn.restype = ctypes.c_int
    def call(shape, dim, index, grad):
        shape = np.asarray(shape, np.int64)
        index = np.ascontiguousarray(index, dtype=np.int64)
        grad = np.ascontiguousarray(grad, dtype=np.float32)
        if grad.shape != index.shape:
            raise ValueError('mismatched source shape')
        index_shape = np.asarray(index.shape, np.int64)
        if len(shape) != len(index_shape):
            raise ValueError('mismatched rank')
        result = np.full(tuple(shape), np.nan, np.float32)
        if fn(grad.ctypes.data, index.ctypes.data, result.ctypes.data,
              shape.ctypes.data, index_shape.ctypes.data, len(shape), dim):
            raise ValueError('invalid CPU gather input')
        return result
    return call


def assess(rows, *, gpu, functional=None):
    expected = {f'{name}_k{k}' for name in ('x_32055', 'x_32083', 'x_32110') for k in (1, 4)}
    if len(rows) != 6 or {r['name'] for r in rows} != expected:
        return 'RED'
    for r in rows:
        if (len(r['hashes']) != 5 or len(set(r['hashes'])) != 1 or not r['bitwise'] or
                not np.isfinite(r['relative_L2']) or not 0 <= r['relative_L2'] <= 1e-6):
            return 'RED'
        if gpu:
            times = r['timing_samples_ms']
            if any(len(times[mode]) != 10 or not all(np.isfinite(x) and x > 0 for x in times[mode])
                   for mode in ('off', 'on')):
                return 'RED'
            atomic = statistics.median(times['off']); det = statistics.median(times['on'])
            if r['atomic_ms'] != atomic or r['deterministic_ms'] != det or r['ratio'] != det/atomic or det/atomic > 2:
                return 'RED'
    if gpu and (not functional or set(functional) != {'gather_autograd', 'topk_autograd', 'dtype_conversions'}
                or not all(functional.values())):
        return 'RED'
    return 'GREEN'


def gpu_functional(tc):
    # Prior art: reverse-mode vector-Jacobian checking using a weighted scalar
    # loss; PT-DET-1 shared switch tests (2026). Independent NumPy reference.
    tc.set_deterministic(True)
    shape = (2, 7); index = np.array([[2, 2, 2, 2], [2, 1, 2, 1]], np.int64)
    grad = np.array([[1e8, 1, -1e8, 2], [1, 2, 3, 4]], np.float32)
    a = tc.tensor(np.arange(14, dtype=np.float32).reshape(shape), requires_grad=True)
    (a.gather(-1, tc.tensor(index, dtype='int64')) * tc.tensor(grad)).sum([0, 1], False).backward()
    gather_ok = a.grad.numpy().tobytes() == reference(shape, -1, index, grad).astype(np.float32).tobytes()
    b = tc.tensor(np.arange(14, dtype=np.float32).reshape(shape), requires_grad=True)
    values, indices = b.topk(3)
    upstream = np.array([[1, 2, 3], [4, 5, 6]], np.float32)
    (values * tc.tensor(upstream)).sum([0, 1], False).backward()
    topk_ok = (np.array_equal(indices.numpy(), np.array([[6, 5, 4], [6, 5, 4]], np.int64)) and
               b.grad.numpy().tobytes() == reference(shape, -1, indices.numpy(), upstream).astype(np.float32).tobytes())
    # Exactly representable values isolate dtype dispatch/cast without imposing
    # the FP32 <=1e-6 gate on low-precision rounding.
    small = np.array([[.5, .25, -.5, 2], [1, 2, 3, 4]], np.float32)
    ref = reference(shape, -1, index, small).astype(np.float32)
    dtype_ok = True
    for src_dtype in ('float16', 'bfloat16'):
        for out_dtype in ('float32', src_dtype):
            got = tc._C._gather_backward(tc.tensor(small, dtype=src_dtype), tc.tensor(index, dtype='int64'),
                                         list(shape), -1, out_dtype).float().numpy()
            dtype_ok = dtype_ok and got.tobytes() == ref.tobytes()
        got = tc._C._gather_backward(tc.tensor(small), tc.tensor(index, dtype='int64'),
                                     list(shape), -1, src_dtype).float().numpy()
        dtype_ok = dtype_ok and got.tobytes() == ref.tobytes()
    return dict(gather_autograd=bool(gather_ok), topk_autograd=bool(topk_ok), dtype_conversions=bool(dtype_ok))


def gather(out, *, gpu=False):
    registration()
    if gpu:
        tc, manifest = d.load_fork()
        tc.set_alloc_pooling(True)
        tc.set_deterministic(True)
    else:
        cpu = compile_cpu(out)
    rows = []
    for case in cases():
        shape, dim, index, grad = (case[k] for k in ('shape', 'dim', 'index', 'grad'))
        ref = reference(shape, dim, index, grad)
        if gpu:
            ix = tc.tensor(index, dtype='int64'); g = tc.tensor(grad)
            tc.set_deterministic(True)
        hashes = []; errors = []
        for _ in range(5):
            got = (tc._C._gather_backward(g, ix, shape, dim).numpy() if gpu else cpu(shape, dim, index, grad))
            hashes.append(d.array_digest(got)['sha256']); errors.append(d.rel_l2(got, ref))
            del got
        row = {k: v for k, v in case.items() if k not in ('index', 'grad')}
        row.update(hashes=hashes, bitwise=len(set(hashes)) == 1, relative_L2=max(errors),
                   index_sha256=d.array_digest(index)['sha256'], grad_sha256=d.array_digest(grad)['sha256'])
        del ref
        if gpu:
            def call(mode):
                tc.set_deterministic(mode)
                result = tc._C._gather_backward(g, ix, shape, dim)
                tc.synchronize()
                del result
            for _ in range(3):
                call(False); call(True)
            timing = {False: [], True: []}
            for repeat in range(10):
                for mode in ((False, True) if repeat % 2 == 0 else (True, False)):
                    tc.synchronize(); start = time.perf_counter(); call(mode)
                    timing[mode].append((time.perf_counter() - start) * 1000)
            atomic = statistics.median(timing[False]); det = statistics.median(timing[True])
            row.update(atomic_ms=atomic, deterministic_ms=det, ratio=det/atomic,
                       timing_samples_ms={'off': timing[False], 'on': timing[True]})
        rows.append(row)
        d.create_json(out / (case['name'] + '.json'), row)
        print('PT_DET_2 CASE', case['name'], 'bitwise', row['bitwise'], 'rel_L2', row['relative_L2'], flush=True)
    functional = gpu_functional(tc) if gpu else None
    if gpu: tc.set_deterministic(False)
    verdict = assess(rows, gpu=gpu, functional=functional)
    result = dict(verdict=verdict, registration_sha256=REG_SHA, rows=rows,
                  evidence_class='GPU gather backward microbenchmark' if gpu else 'CPU shared address/reduction only; no GPU timing',
                  gpu_executed=gpu, functional=functional)
    if gpu:
        d.verify_manifest()
        result.update(binary=manifest['binary'], manifest_sha256=d.sha(d.ART/d.MANIFEST_NAME))
    else:
        result['source_pins'] = {p: d.sha(ROOT/p) for p in ('tensor_cuda/include/tc/deterministic_embed.h',
                                 'tensor_cuda/include/tc/deterministic_gather.h', 'tests/pt_det_2_cpu_bridge.cpp')}
    d.create_json(out/'summary.json', result)
    print('PT_DET_2', 'GATHER' if gpu else 'CPU_GATHER', verdict, 'cases', len(rows),
          'max_rel_L2', max(row['relative_L2'] for row in rows))
    return 0 if verdict == 'GREEN' else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('cpu', 'gather'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--lead-gpu', action='store_true')
    p.add_argument('--lock-fd', type=int)
    args = p.parse_args(); registration(); out = args.out.resolve()
    if not any(out.is_relative_to(base) and out != base for base in (ART, d.ART)):
        p.error('out must be a fresh PT-DET-1 or PT-DET-2 artifact subdirectory')
    out.mkdir(parents=True, exist_ok=False)
    try:
        if args.command == 'gather':
            d.create_json(out/'lock_receipt.json', d.require_lead(args.lead_gpu, args.lock_fd))
        return gather(out, gpu=args.command == 'gather')
    except Exception as exc:
        blocked = isinstance(exc, (d.Blocked, d.storage.StorageBlocked))
        result = dict(verdict='BLOCKED' if blocked else 'RED', reason=str(exc),
                      registration_sha256=REG_SHA, gpu_executed=False if isinstance(exc, d.Blocked) else 'unknown')
        d.create_json(out/'summary.json', result)
        print('PT_DET_2', result['verdict'], str(exc), file=sys.stderr)
        return 2 if blocked else 1


if __name__ == '__main__':
    sys.exit(main())
