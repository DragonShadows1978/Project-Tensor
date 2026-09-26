# PT-RING-1: GRAPA pinned host ring API

**CPU build and author checks only; GPU acceptance is BLOCKED until the lead's
slot.** No throughput improvement, overlap, full engine regression, or numerical
golden result is claimed. This API is additive in the registered fork; the live
engine and GRAPA training code were not changed.

## Exact capture recipe

`live` is the fixed ordered list of model parameters **and buffers**, followed
by Adam `m` and `v`. The run-7 checkpoint schema is pinned in
`artifacts/pt_ring_1/GEOMETRY.json`: 434 model entries + 338 `m` + 338 `v` =
1,110 tensors, 2,939,091,552 bytes. This is a CPU inspection of the checkpoint,
not a measurement of training memory or time. Optimizer step counters, RNG,
loader position and other small metadata still need the GRAPA tripwire's normal
CPU snapshot at the same step; this API copies tensor storage only.

```python
import tensor_cuda as tc

# Once, outside the training critical path. Existing CC47-B staging can also
# be supplied if every tensor matches live's shape and dtype and owns storage.
legacy = tc.legacy_stream()
copy_stream = tc.Stream(non_blocking=True)
device_staging = [tc.empty_like(t) for t in live]
host_slots = [[tc.pinned_empty(t.shape, t.dtype) for t in live]
              for _ in range(3)]
host_ready = [None, None, None]
previous_host = None

def capture_after_step(step):
    global previous_host
    i = step % 3
    # Finish any CPU consumer of slot i before calling this function. A
    # completed CUDA event alone does not finish a CPU reader of the slot.
    with tc.no_grad():
        if previous_host is not None:
            legacy.wait(previous_host)  # GPU-side wait, does not block Python
        tc.copy_many_(device_staging, live)  # default method="memcpy", legacy
        e_staged = tc.Event().record(legacy)
    e_host = tc.copy_to_host_async(device_staging, host_slots[i], copy_stream,
                                  after=e_staged)
    host_ready[i] = previous_host = e_host
    return i, e_host

# Training order:
# loss.backward()
# opt.step()                      # in-place live updates on legacy stream
# i, ready = capture_after_step(step)
# ... next training step may now use/update live ...

# To inspect a completed checkpoint:
# ready.synchronize()             # or poll ready.query()
# arrays = [b.numpy() for b in host_slots[i]]
# Consume/copy arrays before the ring reuses slot i. The arrays are views.

# Drain at shutdown / before discarding or repurposing staging:
# tc.collect_async_copies(wait=True)
```

The dependency argument is reasoning: legacy order makes stage copy follow
`opt.step()`; the side stream waits until stage is ready; next stage overwrite
waits for previous D2H completion. Thus live may change after staging while the
captured stage remains unchanged until its host copy completes. A direct D2H
from live lacks this property. Storage retention prevents deallocation, **not
mutation**. CUDA completion gates host readability; separate CPU consumer
ownership gates slot reuse. The lead's positive and missing-wait controls test
this reasoning on one device; they are not a universal concurrency proof.

## Calls and contracts

| Call | Behavior |
| --- | --- |
| `pinned_empty(shape, dtype="float32")` | Uninitialized `PinnedBuffer`, `cudaHostAllocDefault`; zero-byte shapes allocate nothing. |
| `buffer.numpy()` / `np.asarray(buffer)` | Zero-copy writable view; its owner retains pinned storage. Read/write only after the producing event and before slot reuse. |
| `empty_like(tensor)` | Independent uninitialized CUDA storage, same shape/dtype, `requires_grad=False`; allocation outside the step. |
| `Tensor.copy_(src)` | Same shape/dtype, bitwise D2D on legacy/current engine stream, returns self; requires `no_grad()`. |
| `copy_many_(dsts, srcs, method="memcpy")` | One Python call; one C++ `cudaMemcpyAsync` per nonempty non-self pair; returns completion Event. |
| `copy_many_(..., method="kernel")` | Explicit comparison path: pinned descriptor H2D + one chunked byte-copy kernel + stream-ordered descriptor free. All overhead is measured; no automatic method selection. |
| `Stream(non_blocking=True)` | Owned side stream. `False` is allowed for general events/waits but rejected for async D2H. |
| `legacy_stream()` | Borrowed explicit legacy stream used by existing kernels; no stream context manager or kernel dispatch switch. |
| `Event(enable_timing=False)` | Uses `cudaEventDisableTiming` by default. Events are one-shot, including copy completion events. |
| `event.record(stream=None)` | Record once; `None` means legacy; returns event. Create a fresh event for another record. |
| `stream.wait(event)` | GPU dependency via `cudaStreamWaitEvent`; event must already be recorded. |
| `event.query()` / `.synchronize()` | Test/wait completion; unrecorded events raise. Completed requests are also reclaimed. |
| `stream.synchronize()` | Host wait, GIL released; reclaims completed requests. |
| `start.elapsed_time(end)` | Milliseconds between two recorded timing-enabled events. Synchronize end before reading; no implicit host wait. |
| `copy_to_host_async(tensors, buffers, stream, after=None)` | Whole-list C++ enqueue, GIL released; returns completion Event. If omitted, `after` is recorded on legacy automatically. |
| `pinned_bytes()` | Bytes in all engine-owned pinned allocations in this process, including views/in-flight copies and kernel descriptors. External libraries' pinned memory is not counted. |
| `pinned_memory_limit()` / `set_pinned_memory_limit(nbytes)` | Default 12 GiB budget; cannot lower below currently held bytes. Exceeding the budget raises before allocation. |
| `collect_async_copies(wait=False)` | Reclaim completed operands; `wait=True` drains recorded pending copies. Returns number still retained. |
| `mem_get_info()` | `(free_bytes, total_bytes)` from `cudaMemGetInfo` on logical CUDA device 0. |

All six engine dtypes are supported. `bfloat16` views expose **raw native
`uint16` bits**, not converted FP32 values; `float16` exposes NumPy float16,
bool exposes NumPy bool, and float32/uint8/int64 preserve their native dtype.
Copies never perform floating arithmetic, so negative zero and NaN payloads
must survive bitwise. The GPU matrix covers all six dtypes, ranks 0–4, empties,
scalar payloads and 64 KiB chunk boundaries. Existing `Tensor.numpy()` semantics
are unchanged, including its bf16 upcast.

NDArrays in this engine are contiguous. Existing tensor factories materialize
noncontiguous NumPy inputs, and engine transpose/permute materialize outputs.
Pinned destinations must be actual `PinnedBuffer` objects; arbitrary ndarray
destinations, including slices of pinned views, are rejected rather than
silently using pageable staging. Shapes/dtypes/list lengths, byte overflow and
all batch aliases are checked before enqueue. Duplicate/overlapping writes and
cross-pair source/destination overlap are refused. Exact self-copy is allowed.
CUDA runtime failures may occur after partial submission: the error propagates,
affected caches are invalidated and the stream is drained before resources are
released; there is no transactional rollback of tensor values.

This surface supports the engine's **single logical device 0/current context**.
It refuses another current device. `cudaHostAllocPortable` is not needed for
this scope; multi-context use would require revisiting allocation flags and
device metadata. No cross-device/context behavior is claimed. `no_grad` remains
the engine's existing process-global policy; this change does not make concurrent
training/mutation from multiple Python threads safe.

## Lifetime and limits

Pending requests hold C++ shared owners of actual device `Storage`, pinned
buffers, stream and completion event. Dropping the returned event, tensor
names, or buffer names does not release storage under an outstanding copy.
No background worker or CUDA host callback is used. Subsequent copy/pinned
allocation calls, completion waits/queries, or explicit collection reclaim
completed requests. An otherwise idle process can retain completed operands
until collection; call `collect_async_copies(wait=True)` for deterministic
release. NumPy views retain their buffer independently. Normal process shutdown
drains outstanding streams before releasing owners. A CUDA error during drain
is reported and resources are conservatively retained rather than assumed safe.

The observed host memlock limit is 8,417,419,264 bytes, below the exact
8,817,274,656-byte ring. NVIDIA's driver enforcement is **unmeasured** in this
seat. Driver allocation errors propagate immediately when returned, with no
retry or pageable fallback. The explicit 12 GiB budget avoids asking the driver
above that engine-owned cap. It cannot guarantee a responsive driver call or
identify the machine's physical pinning limit. The lead's allocation lane has a
90-second process timeout and reports timeout as BLOCKED, never GREEN. It tests
the full ring and a deterministic explicit-budget failure, not host exhaustion.

## Lead-only gate commands

The lead supplies an authorized exclusive slot and manages its GPU lock outside
this harness. The seat/harness never opens or inspects the production lock.
Use a fresh output directory for every invocation; receipts are create-only.

```bash
cd /mnt/ForgeRealm/wt/pt-tf32
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
  python -B scripts/pt_ring_1.py --lead-gpu --gate all \
  --out artifacts/pt_ring_1/lead_01
```

Individual `--gate` choices: `correctness`, `hazard`, `overlap`, `ring`,
`limits`, `goldens`, `engine_suite`. Every invocation checks the additive source
seal and both fork binaries. The full sequence stops on the first non-GREEN
lane. CPU-hidden/unauthorized entry exits 2 before engine import/output creation.
The overall cap is 900 s (10 s receipt reserve); non-suite lanes get 90 s,
including both golden subprocesses together, and the engine suite gets 300 s.
Timeout signals only the worker started by this harness. Workers run sequentially.

Performance protocol: two warmups, seven interleaved A/B samples with alternating
order. Overlap uses 2,940,000,000 bytes and a calibrated >=400 ms synthetic
compute loop. Every sample must show intersecting event intervals, and median
concurrent/solo compute wall ratio must be <=1.03. Ring timing uses the exact
checkpoint geometry and three pinned slots; both methods' host enqueue time
and legacy event time (including previous-copy wait) are reported. Default
`memcpy` must have both medians <=50.9 ms (1% of the registered 5.09 s step).
All three host slots must contain the expected bits. These are microbenchmarks,
not run-7 training throughput measurements. Selecting a different default after
measurement would require a new recorded decision.

The suite worker preloads the fork and uses Python audit hooks to reject live
engine imports/opens. Historical campaign-specific pins may still fail; they
are not rewritten or skipped by this order. The finite fp32/bf16 golden checks
compare the preserved pre-change binary against the rebuilt binary in separate
processes; they do not certify every kernel or full-model replay.

## Prior art

- **NVIDIA CUDA 12.6 (2024):** pinned allocation, non-blocking streams, events,
  byte copies and timing are taken from the runtime mechanism, not invented.
  Primary [memory API](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-runtime-api/group__CUDART__MEMORY.html)
  and [stream synchronization behavior](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-runtime-api/stream-sync-behavior.html)
  verified through the web tool in this seat.
- **PyTorch (2016 onward; tutorials 2024):** API shape, empty-like staging and
  completion-bound lifetime ideas are taken. Primary
  [pinned/non-blocking transfer tutorial](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html),
  [record_stream](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html),
  and [DCP async checkpoint recipe](https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html)
  verified. DCP's persistence pipeline is related context, not a claim that its
  implementation is the same GPU staging pipeline.
- **NVIDIA Apex (project 2018 onward):** chunked multi-tensor pointer metadata
  batching is taken as precedent. Primary
  [multi_tensor_apply source](https://github.com/NVIDIA/apex/blob/master/csrc/multi_tensor_apply.cuh)
  verified; the date identifies the project, not the precise kernel's debut.
  This implementation uses a device descriptor array and binary search over
  chunk prefixes, plus aligned uint4 or byte copying; no new copy algorithm is
  claimed. Sorted interval lookup uses standard C++ library binary search.
- **CheckFreq, Mohan et al., FAST 2021:** two-phase snapshot/persistence context,
  verified in the [primary publication](https://www.microsoft.com/en-us/research/publication/checkfreq-frequent-fine-grained-dnn-checkpointing/).
  **Check-N-Run, Eisenman et al., NSDI 2022**, and **Gemini, Wang et al., SOSP
  2023**, are related checkpoint-system context, verified via
  [USENIX](https://www.usenix.org/conference/nsdi22/presentation/eisenman) and
  [the authors' paper](https://zhuangwang93.github.io/docs/Gemini_SOSP23.pdf).
  Their compression, selection, replication and distributed recovery algorithms
  are not implemented here.
- **pybind11, Jakob (2015), Python PEP 578 (2019), ctypes (2006), NIST SHA256
  (2001), POSIX and local PT-DET/PT-TF32 receipts (2026):** standard ownership,
  GIL, audit, CPU bridge and receipt mechanisms reused. These dates are known
  background rather than newly verified publication claims.

The local work is the legacy-stream integration, explicit ownership/alias/budget
contracts, and run-7 gate construction. Prior art comments also appear at the
code sites and in `docs/PT_RING_1_LEDGER.md`.
