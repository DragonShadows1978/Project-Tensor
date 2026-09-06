#!/usr/bin/env python3
"""Identical-input speed chain. GPU use only through apa_spd1_lead_gpu.sh.

Prior art: dense scaled attention, Vaswani et al. (2017); PyTorch SDPA
contributors (2023-2026), forced math/efficient/FlashAttention-2 (Dao, 2023);
TurboQuant (Zandieh et al., 2025), repository MSE reconstruction only;
APA (David/Project-Tensor, 2026), existing two-pass z-score and SP1 kernels;
BLASST (Yuan et al., 2025/2026), running-max criterion; online normalizer
(Milakov & Gimelshein, 2018); split-K/Flash-Decoding (Dao et al., 2023);
ThriftAttention (Sharratt, 2026), related selective mixed precision, not called.
Reused implementations are unchanged. New work is benchmark integration.
Sources and exact reuse boundaries: artifacts/apa_spd1/PRIOR_ART.md.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
import warnings

import numpy as np

from apa_spd1_common import (ART, ROOT, SCOPE, fingerprint, frozen_sp2, grouped_dense,
    inputs, inventory, load_runtime, matching_fraction, metrics, registration,
    registry, rounded_host, sdpa_call, sha, summarize_samples, write_json)


class CudaTelemetry:
    """NVIDIA CUDA runtime API (2026 docs), E1 pool + SP1 event approach.

    Uses the SAME CUDA 12.6 runtime as the pinned engine and stream zero, which
    is also explicitly selected for PyTorch. UsedMemHigh counts live allocation,
    not reserved pool backing. No polling thread or NVML peak approximation.
    """
    def __init__(self):
        self.lib = ctypes.CDLL('/usr/local/cuda-12.6/lib64/libcudart.so')
        p, pp, i = ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int
        signatures = {
            'cudaDeviceGetDefaultMemPool': [pp, i],
            'cudaMemPoolGetAttribute': [p, i, p], 'cudaMemPoolSetAttribute': [p, i, p],
            'cudaEventCreate': [pp], 'cudaEventRecord': [p, p],
            'cudaEventSynchronize': [p], 'cudaEventDestroy': [p],
            'cudaEventElapsedTime': [ctypes.POINTER(ctypes.c_float), p, p],
        }
        for name, args in signatures.items():
            fn = getattr(self.lib, name); fn.argtypes = args; fn.restype = i
        self.pool = p(); self.start = p(); self.end = p()
        self.check(self.lib.cudaDeviceGetDefaultMemPool(ctypes.byref(self.pool), 0))
        self.check(self.lib.cudaEventCreate(ctypes.byref(self.start)))
        self.check(self.lib.cudaEventCreate(ctypes.byref(self.end)))

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f'CUDA telemetry error {code}')

    def pool_value(self, attr):
        val = ctypes.c_uint64()
        self.check(self.lib.cudaMemPoolGetAttribute(self.pool, attr, ctypes.byref(val)))
        return val.value

    def reset_peak(self):
        zero = ctypes.c_uint64(0)
        self.check(self.lib.cudaMemPoolSetAttribute(self.pool, 8, ctypes.byref(zero)))

    def timed(self, call):
        wall = time.perf_counter()
        self.check(self.lib.cudaEventRecord(self.start, None))
        out = call()
        self.check(self.lib.cudaEventRecord(self.end, None))
        self.check(self.lib.cudaEventSynchronize(self.end))
        ms = ctypes.c_float()
        self.check(self.lib.cudaEventElapsedTime(ctypes.byref(ms), self.start, self.end))
        wall_ms = (time.perf_counter() - wall) * 1000
        del out
        return ms.value, wall_ms

    def close(self):
        self.check(self.lib.cudaEventDestroy(self.start))
        self.check(self.lib.cudaEventDestroy(self.end))


def bulk_keys(tc, k, shape):
    if shape['bulk_method'] == 'turboquant_mse':
        # Zandieh et al. (2025) rotation + scalar quantization; reuse local
        # Lloyd-Max tables and reconstruction (Max 1960, Lloyd 1982). This
        # MSE-only implementation has no QJL residual or packed KV residency.
        from tensor_cuda.quant import _tables, _quantize_keys
        tables = _tables(shape['D'], shape['bulk_bits'], shape['KVH'], True, 'cuda')
        return _quantize_keys(k, *tables)
    # E1 (2026) symmetric per-key signed INT4 (-7..7). Generic scalar uniform
    # quantization, not TurboQuant. Same precision/rounding on reconstructed K
    # for both APA contenders; preparation is excluded from timing and peak.
    host = k.numpy().astype(np.float32)
    scales = np.max(np.abs(host), axis=-1, keepdims=True) / np.float32(7)
    safe = np.where(scales > 0, scales, np.float32(1))
    recon = np.clip(np.rint(host / safe), -7, 7) * scales
    return tc.tensor(rounded_host(recon, k.dtype), dtype=k.dtype)


def sp2_call(tc, shape, tensors):
    files = frozen_sp2()
    if not files:
        return None, dict(status='UNAVAILABLE', reason='SP2 frozen e_q table absent at runtime', searched='/mnt/ForgeRealm/Project-Tensor/artifacts/apa_sp2')
    manifest_path = ART / 'sp2_adapter.json'
    if not manifest_path.exists():
        return None, dict(status='BLOCKED_SP2_INTERFACE', reason='frozen candidate exists, but SP2 launcher interface is not yet supplied; no substitute delta', frozen_files=files)
    # Contract is an explicit, separately amended integration point, not a
    # guessed launcher API. No table fitting is allowed in this hook.
    m = json.loads(manifest_path.read_text())
    table, launcher = Path(m['frozen_table']).resolve(), Path(m['launcher']).resolve()
    allowed = Path('/mnt/ForgeRealm/Project-Tensor')
    if str(table) not in files or not launcher.is_relative_to(allowed):
        raise RuntimeError('SP2 adapter provenance outside authorized main paths')
    for p, expected in [(table, m['table_sha256']), (launcher, m['launcher_sha256'])]:
        if sha(p) != expected:
            raise RuntimeError(f'SP2 pin mismatch: {p}')
    spec = importlib.util.spec_from_file_location('apa_spd1_external_sp2', launcher)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    result = getattr(module, m['entry_point'])(tc=tc, shape=shape, tensors=tensors,
                                            epsilon=1e-3, frozen_path=str(table))
    if not isinstance(result, dict) or not callable(result.get('call')):
        raise RuntimeError('SP2 adapter must return {call: callable, metadata: dict}')
    return result['call'], dict(status='READY', epsilon=1e-3, adapter=m,
                                metadata=result.get('metadata', {}))


def fraction_diagnostics(tc, shape, tensors):
    q, k, kq, v = tensors
    # SP1's GPU diagnostic path is used only outside all timed/peak calls.
    out, mask = tc._C.apa_selective_attention_sp(q, k, kq, v,
        1 / math.sqrt(shape['D']), shape['delta'], shape['causal'], diagnostics=True)
    tc.synchronize()
    gm = mask.numpy().astype(bool)
    del out, mask
    L, S = shape['L'], shape['S']
    selected = np.unique(np.linspace(0, L - 1, min(L, 32), dtype=int))
    # SP1's z-score rule; sample count is bounded, full output accuracy is not
    # sampled. CPU dot/reduction order differs from the CUDA baseline; honest
    # estimate only (SP1.1 already demonstrated threshold-boundary sensitivity).
    qh, kh = q.numpy().astype(np.float32), kq.numpy().astype(np.float32)
    z = statistics.NormalDist().inv_cdf(1 - shape['refine_percentile'])
    total = old_count = sp_count = 0
    for b in range(shape['B']):
        for h in range(shape['H']):
            bulk = qh[b, h, selected] @ kh[b, h // (shape['H'] // shape['KVH'])].T
            bulk *= np.float32(1 / math.sqrt(shape['D']))
            for row, query in enumerate(selected):
                length = S - L + query + 1 if shape['causal'] else S
                scores = np.abs(bulk[row, :length])
                threshold = scores.mean() + np.float32(z) * scores.std()
                old_count += int((scores >= threshold).sum())
                sp_count += int(gm[b, h, query, :length].sum())
                total += int(length)
    all_valid = shape['B'] * shape['H'] * (L*S - L*(L-1)//2 if shape['causal'] else L*S)
    result = dict(sampled_query_indices=selected.tolist(), sampled_valid_pairs=total,
                  gpu_sp_full_fraction=float(gm.sum() / all_valid),
                  gpu_sp_sample_fraction=sp_count / total, cpu_two_pass_sample_fraction=old_count / total)
    result.update(matching_fraction(shape, sp_count / total, old_count / total))
    return result


def unsupported(exc):
    text = str(exc).lower()
    return isinstance(exc, RuntimeError) and any(s in text for s in
        ['no available kernel', 'no viable backend', 'not supported', 'only supports',
         'requires sm', 'does not support', 'must have the same num_heads'])


def measure_cell(shape, index, reg):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('G2 BLOCKED: no CUDA-capable device is detected; lead must run the leased card')
    tc = load_runtime()
    if list(torch.cuda.get_device_capability(0)) != reg['gpu']['required_arch']:
        raise RuntimeError('registered sm_89 device required; amend separately for another architecture')
    if torch.cuda.memory.get_allocator_backend() != 'native':
        raise RuntimeError('torch native allocator required for allocated-memory telemetry')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if os.environ.get('NVIDIA_TF32_OVERRIDE') != '0':
        raise RuntimeError('NVIDIA_TF32_OVERRIDE=0 is required for engine fp32 reference')
    tc.set_alloc_pooling(False)
    raw = inputs(shape, index, reg['inputs']['seed'])
    hashes = [hashlib.sha256(x.tobytes()).hexdigest() for x in raw]
    engine, pyt = {}, {}
    for dtype in ('float32', 'bfloat16'):
        host = [rounded_host(x, dtype) for x in raw]
        tq, tk, tv = [tc.tensor(x, dtype=dtype) for x in host]
        engine[dtype] = (tq, tk, bulk_keys(tc, tk, shape), tv)
        pyt[dtype] = tuple(torch.from_numpy(x).to(device='cuda', dtype=getattr(torch, dtype)) for x in host)
        for target, other, want in zip((tq, tk, tv), pyt[dtype], host):
            if not np.array_equal(target.numpy().astype(np.float32), want):
                raise RuntimeError(f'engine input conversion differs: {dtype}')
            if not np.array_equal(other.float().cpu().numpy(), want):
                raise RuntimeError(f'torch input conversion differs: {dtype}')
    def sync():
        tc.synchronize(); torch.cuda.synchronize()
    sync()
    tc.set_alloc_pooling(True)
    telemetry = CudaTelemetry()
    specs = registry()
    rows = {s['id']: dict(s, status='PENDING') for s in specs}
    funcs = {}
    scale = 1 / math.sqrt(shape['D'])
    z = statistics.NormalDist().inv_cdf(1 - shape['refine_percentile'])
    for s in specs:
        name, dtype = s['id'], s['dtype']
        q, k, kq, v = engine[dtype]
        if name.startswith('engine_dense'):
            funcs[name] = lambda q=q, k=k, v=v: grouped_dense(tc, q, k, v, shape)
        elif name.startswith('apa_two_pass'):
            funcs[name] = lambda t=engine[dtype]: tc.apa_selective_attention(*t, scale, z, shape['causal'])
        elif name.startswith('apa_sp_'):
            funcs[name] = lambda t=engine[dtype]: tc._C.apa_selective_attention_sp(*t, scale, shape['delta'], shape['causal'])
        elif name.startswith('torch_'):
            funcs[name] = lambda s=s, t=pyt[dtype]: sdpa_call(torch, s, *t, shape)
            rows[name]['gqa_path'] = ('measured repeat_interleave K/V' if s.get('backend') == 'EFFICIENT_ATTENTION' else 'native enable_gqa')
        elif name.startswith('flash_attn'):
            try:
                import flash_attn
                rows[name]['version'] = flash_attn.__version__
                if tuple(int(x) for x in flash_attn.__version__.split('.')[:2]) < (2, 1):
                    raise ImportError('flash_attn >=2.1 required for bottom-right causal convention')
                # FlashAttention-2 (Dao, 2023), native GQA, layout views.
                funcs[name] = lambda t=pyt[dtype]: flash_attn.flash_attn_func(
                    *(x.transpose(1, 2) for x in t), dropout_p=0, softmax_scale=scale,
                    causal=shape['causal']).transpose(1, 2)
            except ImportError as exc:
                rows[name].update(status='UNAVAILABLE', reason=str(exc))
        elif name == 'apa_sp2_fp32':
            call, info = sp2_call(tc, shape, engine[dtype]); rows[name].update(info)
            if call is not None:
                funcs[name] = call
    results = dict(shape=shape, input_sha256=hashes, environment=inventory(),
                   input_conversion_checked='all q/k/v elements equal across runtimes for float32 and bfloat16',
                   device_name=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
                   stream='legacy default stream 0; PyTorch default_stream context',
                   scope_note=SCOPE, rows=list(rows.values()))
    try:
        with tc.no_grad(), torch.inference_mode(), torch.cuda.stream(torch.cuda.default_stream()):
            if torch.cuda.current_stream().cuda_stream != 0:
                raise RuntimeError('single stream contract violated')
            reference = funcs['engine_dense_fp32']().numpy().astype(np.float32)
            sync()
            if not np.isfinite(reference).all():
                raise RuntimeError('nonfinite engine fp32 reference')
            for s in specs:
                name = s['id']
                if name not in funcs:
                    continue
                try:
                    with warnings.catch_warnings(record=True) as emitted:
                        warnings.simplefilter('always')
                        out = funcs[name](); sync()
                        got = (out.float().cpu().numpy() if s['owner'] == 'torch' else out.numpy()).astype(np.float32)
                        del out
                    rows[name]['warnings'] = [str(w.message) for w in emitted]
                    rows[name]['accuracy'] = metrics(got, reference)
                    if not np.isfinite(got).all():
                        rows[name].update(status='RED_NONFINITE'); del funcs[name]; continue
                    if name == 'torch_math_fp32':
                        rows[name]['fp32_parity'] = bool(np.allclose(got, reference,
                            atol=reg['measurement']['accuracy']['fp32_atol'],
                            rtol=reg['measurement']['accuracy']['fp32_rtol']))
                    for _ in range(reg['measurement']['warmups']):
                        warm = funcs[name](); del warm
                    sync()
                except Exception as exc:
                    if not s['required'] and unsupported(exc):
                        rows[name].update(status='UNAVAILABLE', reason=f'{type(exc).__name__}: {exc}')
                        del funcs[name]
                    else:
                        rows[name].update(status='ERROR', reason=f'{type(exc).__name__}: {exc}')
                        raise
            for dtype in engine:
                rows['apa_sp_' + ('fp32' if dtype == 'float32' else 'bf16')]['fraction'] = fraction_diagnostics(tc, shape, engine[dtype])
            gc.collect(); sync(); tc.empty_cache(); torch.cuda.empty_cache(); sync()
            active = list(funcs)
            samples = {n: [] for n in active}; walls = {n: [] for n in active}
            for name in active:
                rows[name]['raw_cuda_samples_ms'] = samples[name]
                rows[name]['raw_wall_samples_ms'] = walls[name]
            # SP1 interleaved A/B generalized to a rotating Latin-style order.
            # Counterbalancing is classical experimental design (Fisher, 1935);
            # this deterministic rotation is harness work, no new algorithm.
            for rep in range(reg['measurement']['repetitions']):
                order = active[rep % len(active):] + active[:rep % len(active)]
                if rep % 2:
                    order = order[::-1]
                for name in order:
                    sync()
                    ms, wall = telemetry.timed(funcs[name])
                    sync(); samples[name].append(ms); walls[name].append(wall)
            # Memory telemetry is separate from CUDA-event timing. All contender
            # outputs are released before resetting their own allocator's peak.
            peaks = {n: [] for n in active}
            for rep in range(3):
                for name in active[rep:] + active[:rep]:
                    gc.collect(); sync()
                    owner = rows[name]['owner']
                    if owner == 'torch':
                        base = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
                    else:
                        base = telemetry.pool_value(7); telemetry.reset_peak()
                    out = funcs[name](); sync()
                    peak = torch.cuda.max_memory_allocated() if owner == 'torch' else telemetry.pool_value(8)
                    if peak < base:
                        raise RuntimeError('allocator peak below baseline')
                    output_bytes = (shape['B'] * shape['H'] * shape['L'] * shape['D']
                                    * (4 if rows[name]['dtype'] == 'float32' else 2))
                    if peak - base < output_bytes:
                        raise RuntimeError('allocator peak does not cover the live output; memory measurement invalid')
                    peaks[name].append(int(peak - base)); del out; sync()
            for name in active:
                rows[name].update(status='OK', timing=summarize_samples(samples[name]),
                    wall=summarize_samples(walls[name]), peak_allocated_delta_bytes=max(peaks[name]),
                    peak_samples_bytes=peaks[name], memory_source=('torch native allocator allocated high-water' if rows[name]['owner'] == 'torch' else 'CUDA default pool UsedMemHigh'),
                    memory_scope='call-local, includes output; excludes resident q/k/v/kq and bulk preparation')
            results['status'] = 'COMPLETE'
            results['numeric_status'] = ('PASS' if rows['torch_math_fp32'].get('fp32_parity') and
                all(r['status'] != 'RED_NONFINITE' for r in rows.values()) else 'RED')
            results['coverage_status'] = ('PASS' if all(r['status'] in ('OK', 'UNAVAILABLE')
                                                      for r in rows.values()) else 'BLOCKED')
    except Exception:
        results.update(status='ERROR', error=traceback.format_exc())
        for row in rows.values():
            if row['status'] in ('PENDING', 'READY'):
                row.update(status='BLOCKED_AFTER_ERROR', reason=results['error'].splitlines()[-1])
    finally:
        telemetry.close()
    return results


def worker(cell_id, stamp):
    # Lease is a live inherited flock descriptor, not merely an env flag.
    import fcntl
    if os.environ.get('APA_SPD1_LEASED') != '1':
        raise RuntimeError('use the leased runner')
    fd = 9
    lock = Path('/tmp/forge-gpu.lock')
    if os.fstat(fd).st_ino != lock.stat().st_ino or os.fstat(fd).st_dev != lock.stat().st_dev:
        raise RuntimeError('wrong inherited lease descriptor')
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    reg = registration()
    index, shape = next((i, s) for i, s in enumerate(reg['shapes']) if s['id'] == cell_id)
    pins = fingerprint()
    path = ART / 'gpu' / f'{cell_id}.{stamp}.receipt.json'
    try:
        result = measure_cell(shape, index, reg)
    except Exception:
        result = dict(status='BLOCKED', shape=shape, error=traceback.format_exc(),
            rows=[dict(s, status='BLOCKED', reason='cell setup failed; see error') for s in registry()])
    result.update(fingerprint=pins, evidence_class='kernel sweep attempt', scope_note=SCOPE)
    if fingerprint() != pins:
        result.update(status='RED_SOURCE_DRIFT', error='harness/SP2 files changed during cell')
    write_json(path, result, exclusive=True)
    print(json.dumps(dict(status=result['status'], receipt=str(path))), flush=True)
    return 0 if (result['status'] == 'COMPLETE' and result['numeric_status'] == 'PASS'
                 and result.get('coverage_status') == 'PASS') else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', nargs='?', default='list', choices=['list', 'inventory', 'summary', 'next', '_worker', '_finish'])
    p.add_argument('target', nargs='?'); p.add_argument('stamp', nargs='?'); p.add_argument('rc', nargs='?')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    reg = registration()
    if args.dry_run or args.mode == 'list':
        for i, s in enumerate(reg['shapes']):
            print(json.dumps(dict(index=i, **s, contenders=[x['id'] for x in registry()])))
        return 0
    if args.mode == 'inventory':
        result = inventory(); write_json(ART / 'CPU_INVENTORY.json', result)
        print(json.dumps(result, indent=2)); return 0
    if args.mode == 'summary':
        from apa_spd1_report import summary
        summary(); return 0
    if args.mode == 'next':
        pins = fingerprint()
        for s in reg['shapes']:
            attempted = False
            for path in (ART / 'gpu').glob(s['id'] + '.*.exit.json'):
                if json.loads(path.read_text()).get('fingerprint') == pins:
                    attempted = True
            if not attempted:
                print(s['id']); return 0
        print('DONE'); return 0
    if args.mode == '_worker':
        return worker(args.target, args.stamp)
    if args.mode == '_finish':
        if args.target not in {s['id'] for s in reg['shapes']} or not args.stamp.isdigit():
            raise ValueError('invalid attempt identity')
        write_json(ART / 'gpu' / f'{args.target}.{args.stamp}.exit.json',
            dict(cell=args.target, stamp=args.stamp, exit_code=int(args.rc), fingerprint=fingerprint(),
                 evidence_class='foreground worker process exit; timeout remains RED'), exclusive=True)
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
