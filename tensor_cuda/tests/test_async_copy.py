"""PT-RING-1 GPU author baseline. Lead slot only; never a CPU skip-pass.

Prior art: NVIDIA CUDA (12.6, 2024) event dependency/byte-copy tests, PyTorch
record_stream lifetime idiom (2016 onward); CheckFreq (Mohan et al., 2021)
snapshot-then-drain. Ours: this engine's legacy-stream hazard and dtype matrix.
"""
import gc
import os
from pathlib import Path
import sys
import threading

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def tc():
    if os.environ.get("PT_RING_1_LEAD_GPU") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES", "") in ("", "-1"):
        pytest.fail("BLOCKED: PT-RING-1 GPU tests require the lead's explicit GPU slot")
    sys.path.insert(0, str(ROOT / "tensor_cuda"))
    import tensor_cuda
    assert Path(tensor_cuda._C.__file__).resolve().is_relative_to(ROOT)
    return tensor_cuda


DTYPES = {"float32": (np.uint32, [0x80000000, 0x7fc01234, 0x7fa00001, 0xffc01234, 0x3f800000]),
          "float16": (np.uint16, [0x8000, 0x7e15, 0x7c01, 0xfe11, 0x3c00]),
          "bfloat16": (np.uint16, [0x8000, 0x7fc1, 0x7f81, 0xffc2, 0x3f80]),
          "int64": (np.uint64, [0, 2**63, 2**64-1, 0x0102030405060708]),
          "uint8": (np.uint8, [0, 255, 127, 1, 42]),
          "bool": (np.uint8, [0, 1, 1, 0])}
SHAPES = [(), (19,), (3, 7), (2, 3, 5), (2, 3, 4, 7),
          (0,), (2, 0), (2, 0, 3), (2, 3, 0, 4)]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("method", ["memcpy", "kernel"])
def test_all_bits_ranks_empty_and_batched(tc, dtype, shape, method):
    rawdtype, patterns = DTYPES[dtype]
    raw = np.resize(np.array(patterns, rawdtype), int(np.prod(shape))).reshape(shape).tobytes()
    src = tc._C._ring_from_bytes(raw, shape, dtype)
    dst = tc.empty_like(src)
    one = tc.empty_like(src)
    host = tc.pinned_empty(shape, dtype)
    with tc.no_grad():
        assert one.copy_(src) is one
        staged = tc.copy_many_([dst], [one], method=method)
    ready = tc.copy_to_host_async([dst], [host], tc.Stream(), after=staged)
    ready.synchronize()
    assert ready.query() and host.numpy().tobytes() == raw
    assert np.asarray(host).tobytes() == raw
    assert host.shape == shape and host.dtype == dtype and host.nbytes == len(raw)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_scalar_nan_payload(tc, dtype):
    rawtype, patterns = DTYPES[dtype]
    for bitpattern in patterns:
        raw = np.array(bitpattern, rawtype).tobytes()
        src = tc._C._ring_from_bytes(raw, (), dtype)
        host = tc.pinned_empty((), dtype)
        tc.copy_to_host_async([src], [host], tc.Stream()).synchronize()
        assert host.numpy().tobytes() == raw


@pytest.mark.parametrize("method", ["memcpy", "kernel"])
def test_mixed_batch_chunks_and_late_validation(tc, method):
    sizes = [0, 1, 15, 16, 17, 65535, 65536, 65537, 200001]
    arrays = [np.arange(n, dtype=np.uint8) for n in sizes]
    src = [tc.tensor(a, dtype="uint8") for a in arrays]
    dst = [tc._C._ring_empty(a.shape, "uint8") for a in arrays]
    host = [tc.pinned_empty(a.shape, "uint8") for a in arrays]
    with tc.no_grad(): tc.copy_many_(dst, src, method=method)
    tc.copy_to_host_async(dst, host, tc.Stream()).synchronize()
    assert [h.numpy().tobytes() for h in host] == [a.tobytes() for a in arrays]
    untouched = tc.tensor([9, 9], dtype="uint8")
    with tc.no_grad(), pytest.raises(ValueError, match="shape/dtype"):
        tc.copy_many_([untouched, dst[-1]], [tc.tensor([1, 2], dtype="uint8"), src[1]], method=method)
    assert untouched.numpy().tobytes() == bytes([9, 9])


def test_pageable_wrong_type_size_dtype_and_duplicate_rejected(tc):
    source = tc.tensor([1., 2.])
    stream = tc.Stream()
    with pytest.raises(TypeError): tc.copy_to_host_async([source], [np.empty(2, np.float32)], stream)
    for host in [tc.pinned_empty((3,), "float32"), tc.pinned_empty((2,), "float16")]:
        with pytest.raises(ValueError, match="shape/dtype"): tc.copy_to_host_async([source], [host], stream)
    host = tc.pinned_empty((2,), "float32")
    with pytest.raises(ValueError, match="overlapping"): tc.copy_to_host_async([source, source], [host, host], stream)
    with pytest.raises(ValueError, match="non-blocking"): tc.copy_to_host_async([source], [host], tc.Stream(False))
    with pytest.raises(ValueError, match="lengths"): tc.copy_to_host_async([source], [], stream)


def test_autograd_alias_and_event_guards(tc):
    source = tc.tensor([1., 2.], requires_grad=True)
    destination = tc.tensor([0., 0.])
    with pytest.raises(RuntimeError, match="no_grad"): destination.copy_(source)
    with pytest.raises(RuntimeError, match="no_grad"): tc.copy_many_([destination], [source])
    with tc.no_grad():
        with pytest.raises(ValueError, match="overlapping"): tc.copy_many_([destination, destination], [source, source])
        with pytest.raises(ValueError, match="overlapping"): tc.copy_many_([destination, source], [source, destination])
        tc.copy_many_([source], [source]).synchronize()  # exact self-copy is safe
    event = tc.Event()
    with pytest.raises(ValueError, match="not been recorded"): tc.legacy_stream().wait(event)
    with pytest.raises(ValueError, match="not been recorded"): event.query()
    event.record()
    with pytest.raises(ValueError, match="one-shot"): event.record()
    event.synchronize()
    with pytest.raises(ValueError, match="timing"): event.elapsed_time(event)
    tc.copy_to_host_async([], [], tc.Stream()).synchronize()
    free, total = tc.mem_get_info()
    assert 0 < free <= total


def test_noncontiguous_numpy_inputs_materialized_by_existing_factory(tc):
    array = np.arange(8*9, dtype=np.float32).reshape(8, 9)[::-2, ::2].T
    assert not array.flags.c_contiguous
    source = tc.from_numpy(array)
    host = tc.pinned_empty(array.shape, "float32")
    tc.copy_to_host_async([source], [host], tc.Stream()).synchronize()
    assert host.numpy().tobytes() == array.tobytes()
    transposed = source.transpose_last()
    h2 = tc.pinned_empty(transposed.shape, "float32")
    tc.copy_to_host_async([transposed], [h2], tc.Stream()).synchronize()
    assert h2.numpy().tobytes() == array.T.tobytes()


def test_pending_lifetime_numpy_owner_and_pinned_budget(tc):
    tc.collect_async_copies(wait=True); gc.collect()
    baseline = tc.pinned_bytes()
    host = tc.pinned_empty((1024,), "uint8")
    view = host.numpy()
    view[:] = 23
    assert np.shares_memory(view, np.asarray(host))
    source = tc.tensor(np.arange(1024, dtype=np.uint8), dtype="uint8")
    stream = tc.Stream()
    tc._C._ring_delay(100, stream)
    event = tc.copy_to_host_async([source], [host], stream)
    assert not event.query()
    del source, host, event, stream
    gc.collect()
    assert tc.pinned_bytes() >= baseline + 1024
    tc.collect_async_copies(wait=True)
    assert view.tobytes() == np.arange(1024, dtype=np.uint8).tobytes()
    del view; gc.collect()
    assert tc.pinned_bytes() == baseline
    old = tc.pinned_memory_limit()
    try:
        tc.set_pinned_memory_limit(baseline + 31)
        with pytest.raises(RuntimeError, match="limit exceeded"): tc.pinned_empty((32,), "uint8")
        assert tc.pinned_bytes() == baseline
    finally: tc.set_pinned_memory_limit(old)


@pytest.mark.parametrize("method", ["memcpy", "kernel"])
def test_staging_hazard_recipe(tc, method):
    live = tc.tensor(np.arange(4096, dtype=np.float32))
    update = tc.tensor(np.full(4096, 71, np.float32))
    stage = tc._C._ring_empty(live.shape, live.dtype)
    stream = tc.Stream(); legacy = tc.legacy_stream()
    hosts = [tc.pinned_empty(live.shape, live.dtype) for _ in range(3)]
    before = live.numpy().tobytes()
    with tc.no_grad():
        tc.copy_many_([stage], [live], method=method)
        staged = tc.Event().record(legacy)
        tc._C._ring_delay(50, stream)
        host_done = tc.copy_to_host_async([stage], [hosts[0]], stream, after=staged)
        live.copy_(update)  # live may change immediately; stage must not
        legacy.wait(host_done)  # critical edge: never overwrite in-flight stage
        tc.copy_many_([stage], [live], method=method)
        second = tc.copy_to_host_async([stage], [hosts[1]], stream, after=tc.Event().record())
        legacy.wait(second)
        tc.copy_many_([stage], [live], method=method)
        third = tc.copy_to_host_async([stage], [hosts[2]], stream)
    third.synchronize()
    assert hosts[0].numpy().tobytes() == before
    assert hosts[1].numpy().tobytes() == hosts[2].numpy().tobytes() == update.numpy().tobytes()


def test_pooled_storage_lifetime_on_side_stream(tc):
    # Restore engine pooling policy before leaving this independent test.
    tc.collect_async_copies(wait=True)
    tc.set_alloc_pooling(True)
    try:
        source = tc.tensor(np.arange(8192, dtype=np.float32))
        host = tc.pinned_empty(source.shape, source.dtype)
        stream = tc.Stream(); tc._C._ring_delay(50, stream)
        done = tc.copy_to_host_async([source], [host], stream)
        del source; gc.collect()
        scratch = [tc.tensor(np.full(8192, -1, np.float32)) for _ in range(8)]
        done.synchronize()
        assert host.numpy().tobytes() == np.arange(8192, dtype=np.float32).tobytes()
        del scratch
    finally:
        tc.collect_async_copies(wait=True); tc.set_alloc_pooling(False)


def test_event_wait_releases_gil(tc):
    # Real blocking C++ event wait, with an independent Python worker. This
    # verifies the Python thread can progress while CUDA is outstanding.
    stream = tc.Stream(); started = threading.Event(); finish = threading.Event()
    counts = [0]
    def count():
        started.set()
        while not finish.is_set(): counts[0] += 1
    worker = threading.Thread(target=count)
    worker.start(); started.wait()
    try:
        tc._C._ring_delay(200, stream)
        done = tc.Event().record(stream)
        assert not done.query()
        before = counts[0]
        done.synchronize()
        assert counts[0] > before
    finally:
        finish.set(); worker.join(timeout=2)
        assert not worker.is_alive()
