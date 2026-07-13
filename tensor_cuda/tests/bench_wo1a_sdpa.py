"""CUDA-event benchmark receipt for WO-1A's DiT joint-attention geometry.

The script deliberately retains composed intermediates while sampling device
memory, which makes the score/weight allocation visible instead of relying on
allocator-pool accounting after function locals have been released.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import ctypes.util
import json
import math
import os
import time

import numpy as np

import tensor_cuda as tc
from tensor_cuda import functional as F


class _CudaRuntime:
    def __init__(self):
        errors = []
        candidates = [ctypes.util.find_library("cudart"), "libcudart.so"]
        for candidate in candidates:
            if not candidate:
                continue
            try:
                self.lib = ctypes.CDLL(candidate)
                break
            except OSError as exc:
                errors.append(str(exc))
        else:
            raise RuntimeError("could not load CUDA runtime: " + "; ".join(errors))
        self.lib.cudaEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cudaEventCreate.restype = ctypes.c_int
        self.lib.cudaEventDestroy.argtypes = [ctypes.c_void_p]
        self.lib.cudaEventDestroy.restype = ctypes.c_int
        self.lib.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.cudaEventRecord.restype = ctypes.c_int
        self.lib.cudaEventSynchronize.argtypes = [ctypes.c_void_p]
        self.lib.cudaEventSynchronize.restype = ctypes.c_int
        self.lib.cudaEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]
        self.lib.cudaEventElapsedTime.restype = ctypes.c_int
        self.lib.cudaMemGetInfo.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        self.lib.cudaMemGetInfo.restype = ctypes.c_int

    def _check(self, code, action):
        if code:
            raise RuntimeError(f"CUDA runtime {action} failed with status {code}")

    def elapsed_ms(self, fn):
        start = ctypes.c_void_p()
        stop = ctypes.c_void_p()
        self._check(self.lib.cudaEventCreate(ctypes.byref(start)), "event create(start)")
        self._check(self.lib.cudaEventCreate(ctypes.byref(stop)), "event create(stop)")
        try:
            wall0 = time.perf_counter()
            self._check(self.lib.cudaEventRecord(start, None), "event record(start)")
            result = fn()
            self._check(self.lib.cudaEventRecord(stop, None), "event record(stop)")
            self._check(self.lib.cudaEventSynchronize(stop), "event synchronize(stop)")
            wall_ms = (time.perf_counter() - wall0) * 1000.0
            elapsed = ctypes.c_float()
            self._check(self.lib.cudaEventElapsedTime(ctypes.byref(elapsed), start, stop), "event elapsed")
            return float(elapsed.value), wall_ms, result
        finally:
            self.lib.cudaEventDestroy(start)
            self.lib.cudaEventDestroy(stop)

    def used_mib(self):
        free = ctypes.c_size_t()
        total = ctypes.c_size_t()
        self._check(self.lib.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)), "mem get info")
        return (total.value - free.value) / (1024.0 * 1024.0)


@contextlib.contextmanager
def _env(name, value):
    old = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def _composed_retained(q, k, v, scale, memory_samples, runtime):
    scores = tc.matmul(q, k, alpha=scale, trans_b=True)
    tc.synchronize()
    memory_samples.append(runtime.used_mib())
    if hasattr(tc, "causal_softmax"):
        B, H, L, S = scores.shape
        weights = tc.causal_softmax(scores.reshape([B, H * L, 1, S])).reshape([B, H, L, S])
    else:  # pragma: no cover - all supported WO-1A builds have this primitive
        weights = scores.softmax(-1)
    tc.synchronize()
    memory_samples.append(runtime.used_mib())
    out = tc.matmul(weights, v)
    tc.synchronize()
    memory_samples.append(runtime.used_mib())
    return out, scores, weights


def _run_once(label, fn, runtime, warmup):
    for _ in range(warmup):
        result = fn()
        tc.synchronize()
        del result
    tc.empty_cache()
    tc.synchronize()
    baseline_mib = runtime.used_mib()
    event_ms, wall_ms, result = runtime.elapsed_ms(fn)
    tc.synchronize()
    return {
        "label": label,
        "cuda_event_ms": event_ms,
        "host_wall_ms": wall_ms,
        "baseline_used_mib": baseline_mib,
        "result": result,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    runtime = _CudaRuntime()
    rng = np.random.default_rng(20260712)
    np_dtype = np.float16 if args.dtype == "float16" else np.float32
    B, H, L, D = 1, 16, 4442, 64
    arrays = [
        (rng.standard_normal((B, H, L, D)).astype(np.float32) * 0.125).astype(np_dtype)
        for _ in range(3)
    ]
    q, k, v = [tc.tensor(a, dtype=args.dtype) for a in arrays]
    scale = 1.0 / math.sqrt(D)
    tc.synchronize()
    input_used_mib = runtime.used_mib()

    with tc.no_grad(), _env("TC_FUSED_SDPA_NONCAUSAL", "1"):
        fused = _run_once(
            "fused",
            lambda: F.scaled_dot_product_attention(q, k, v), runtime, args.warmup)
    fused_out = fused.pop("result")
    fused["peak_used_mib"] = runtime.used_mib()
    fused["peak_delta_from_inputs_mib"] = fused["peak_used_mib"] - input_used_mib
    del fused_out
    tc.empty_cache()
    tc.synchronize()

    retained_samples = []
    with tc.no_grad(), _env("TC_FUSED_SDPA_NONCAUSAL", "0"):
        composed = _run_once(
            "composed",
            lambda: F.scaled_dot_product_attention(q, k, v), runtime, args.warmup)
        composed_timed_out = composed.pop("result")
        del composed_timed_out
        tc.empty_cache()
        tc.synchronize()
        composed_out, scores, weights = _composed_retained(q, k, v, scale, retained_samples, runtime)
    del composed_out, scores, weights
    composed["peak_used_mib"] = max(retained_samples)
    composed["peak_delta_from_inputs_mib"] = composed["peak_used_mib"] - input_used_mib
    composed["retained_stage_used_mib"] = retained_samples

    receipt = {
        "work_order": "WO-1A",
        "build_expectation": "Release",
        "shape": [B, H, L, D],
        "dtype": args.dtype,
        "input_used_mib": input_used_mib,
        "fused": fused,
        "composed": composed,
        "notes": [
            "Timing is CUDA-event device elapsed time plus host wall time.",
            "Composed peak is sampled while scores and weights are deliberately retained.",
            "Values are CUDA-runtime device usage; unrelated-process baseline remains visible.",
        ],
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
