"""Per-slot repeat bytes, and isolated backward live bytes and runtime."""
import ctypes as C
import hashlib
import json
import os
from statistics import median
import time

import numpy as np
import pytest
import tc_signal as ts
import tensor_cuda as tc

from bwd_r3_support.cases import SLOTS, exact, fixture, gradients, require_gpu, tracked

REPEAT = (("bp2", "conv_transpose1d_02"), ("bp2", "conv_transpose1d_01"),
          ("bp1", "tail_conv_transpose_real_kokoro"))
MEASURE = (("bp1", "conv1d_kokoro"), ("bp2", "conv1d_06"), ("bp2", "conv1d_12"),
           ("bp2", "conv1d_23"), ("bp1", "conv_transpose1d_kokoro"),
           ("bp2", "conv_transpose1d_03"), *REPEAT[1:])


@pytest.mark.parametrize("bp,label", REPEAT)
def test_gpu_repeat(bp, label):
    case = fixture(bp, label)
    require_gpu()
    count = int(os.environ.get("BWD_R3_RERUNS", "10"))
    assert count in (1, 10)
    original = gradients(case)
    for a in original:
        assert a.size and np.isfinite(a).all() and (np.ptp(a) > 0 if a.size > 1 else a.item() != 0)
    for _ in range(count - 1):
        for left, right in zip(original, gradients(case)):
            exact(left, right)
    ts.synchronize()
    cfg = ts._C._conv_backward_config(0)
    row = dict(case=label, count=count, cublas=cfg["cublas_version"], sm=cfg["sm_count"],
               hashes=[hashlib.sha256(a.view(np.uint32)).hexdigest() for a in original])
    print("R3_REPEAT " + json.dumps(row), flush=True)


@pytest.mark.parametrize("bp,label", MEASURE)
@pytest.mark.parametrize("slots", ("xwb", "x", "w"))
def test_gpu_memory_and_runtime(bp, label, slots, monkeypatch):
    case = fixture(bp, label)
    require_gpu()
    tc.set_alloc_pooling(True)
    try:
        _measure(case, slots, monkeypatch)
    finally:
        tc.set_alloc_pooling(False)


def _measure(case, slots, monkeypatch):
    col, part = ts._C._conv_backward_scratch(case.op != "conv1d", case.x.shape, case.w.shape,
                                             case.seed.shape, case.native_attrs) if "w" in slots else (0, 0)
    active = tuple(slot in slots for slot in SLOTS)
    diagnostic = os.environ.get("BWD_R3_R2_DIAGNOSTIC")
    if diagnostic is not None:
        assert case.label == "conv_transpose1d_kokoro" and slots == "xwb" and col > 0
        assert diagnostic in ("no-prior-gemm", "prior-forward-gemm", "warm-initialize")

    # NVIDIA CUDA 12.6 (2024), default-pool used-byte high watermark. R3
    # samples only backward allocations, after forward and graph creation.
    cuda = C.CDLL("libcudart.so.12")
    pool = C.c_void_p()
    assert cuda.cudaDeviceGetDefaultMemPool(C.byref(pool), 0) == 0

    def used(attr):
        amount = C.c_uint64()
        assert cuda.cudaMemPoolGetAttribute(pool, attr, C.byref(amount)) == 0
        return amount.value

    timings = []
    run = ts._C._conv_backward_run

    def timed(*args):
        ts.synchronize()
        started = time.perf_counter()
        run(*args)
        ts.synchronize()
        timings.append((time.perf_counter() - started) * 1000)

    monkeypatch.setattr(ts._C, "_conv_backward_run", timed)
    steps = {}

    def once(*, first=False):
        leaves, value, seed = tracked(case, active)
        loss = (value * seed).sum()
        ts.synchronize()
        if first and diagnostic:
            if diagnostic == "prior-forward-gemm":
                # Conv1d uses a separate cuBLAS handle. The matched ConvTranspose
                # forward above uses a CUDA gather and issues no GEMM.
                x = tc.tensor(np.ones((1, 16, 16), dtype=np.float32))
                w = tc.tensor(np.ones((16, 16, 1), dtype=np.float32))
                ts.synchronize()
                started = time.perf_counter()
                ts.conv1d(x, w)
                ts.synchronize()
                steps["prior_forward_gemm_ms"] = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            ts._C._conv_backward_config(0)
            ts.synchronize()
            steps["strict_handle_initialize_ms"] = (time.perf_counter() - started) * 1000
        assert cuda.cudaMemPoolSetAttribute(pool, 8, C.byref(C.c_uint64(0))) == 0
        baseline = used(7)
        started = time.perf_counter()
        loss.backward()
        ts.synchronize()
        elapsed = (time.perf_counter() - started) * 1000
        high = used(8)
        assert len(timings) == 1 and high > baseline
        assert tuple(leaf.grad is not None for leaf in leaves) == active
        return dict(native_ms=timings.pop(), backward_ms=elapsed, pool_peak_increment_bytes=high - baseline)

    # Python timeit (Python docs, 2024): repeated elapsed-time samples. r2
    # uses one independent graph per sample and reports both median and range.
    first = once(first=True)
    once()
    samples = [once() for _ in range(5)]
    cfg = ts._C._conv_backward_config(0)
    assert max(col, part) * 4 <= cfg["scratch_cap_bytes_per_buffer"]
    g = case.native_attrs[-1]
    right, tile_time = (case.seed.shape[1], case.x.shape[-1]) if case.op != "conv1d" else (
        case.x.shape[1], case.seed.shape[-1])
    if col:
        assert col == right // g * case.w.shape[2] * min(cfg["tile_time"], tile_time)
        assert part == int(np.prod(case.w.shape)) // g
    stats = {}
    for name in ("native_ms", "backward_ms"):
        values = [sample[name] for sample in samples]
        stats[name] = dict(median=median(values), min=min(values), max=max(values), samples=values)
    warm_high = max(sample["pool_peak_increment_bytes"] for sample in samples)
    grads = sum(a.nbytes for slot, a in zip(SLOTS, case.inputs) if slot in slots)
    active_workspace = cfg["workspace_bytes"] if col else 0
    row = dict(case=case.label, slots=slots, pid=os.getpid(), diagnostic=diagnostic,
               x_shape=case.x.shape, w_shape=case.w.shape, g_shape=case.seed.shape,
               gradient_bytes=grads, pack_bytes=col * 4, partial_bytes=part * 4,
               cublas_version=cfg["cublas_version"], sm_count=cfg["sm_count"],
               resident_private_workspace_bytes=cfg["workspace_bytes"],
               active_gemm_workspace_bytes=active_workspace,
               first_call=first, warm_up_calls=1, warm=stats, **steps,
               warm_device_pool_peak_increment_bytes=warm_high,
               warm_active_backward_peak_bytes=warm_high + active_workspace,
               first_active_backward_peak_bytes=first["pool_peak_increment_bytes"] + active_workspace)
    print("R3_MEASURE " + json.dumps(row), flush=True)
