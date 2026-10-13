# tc_signal: TensorCUDA add-on for signal ops

`tc_signal` is a separate package next to `tensor_cuda/`. It holds the ops Project-Voice needs to run Kokoro-82M on
TensorCUDA, and it keeps them out of TensorCUDA's core. David, 2026-10-12: "I don't want to build it directly INTO the
core of TensorCUDA - It needs to be separate". Plan: `Fork/tensor_fork_voice/docs/P1T_TENSOR_ALT_PLAN.md` §4.

Status (P1T-M2): `conv1d` (T2) and `conv_transpose1d` (T3). T4-T6 come in later milestones.

- **No core dependency.** It does not link `_tensor_cuda`, include `tensor_cuda/include`, or compile any core source.
- **Same CUDA runtime as core.** It links the shared CUDA 12.6 runtime (`libcudart.so.12`) and `libcublas.so.12`, with
  RUNPATH `/usr/local/cuda-12.6/lib64`, the same path core's module resolves. In one process the runtime is loaded
  once, so tc_signal and core share one runtime instance, one context and one default memory pool.
- **Any DLPack CUDA producer works.** Operands can be TensorCUDA tensors (after P1T-M1), torch tensors, or bare DLPack
  capsules.

## Build

```bash
# offline pybind11 v2.12.0 (the same as core); the source tree is read-only
PYBIND11_SOURCE_DIR=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src ./build.sh 89
# -> tc_signal/_tc_signal.cpython-312-x86_64-linux-gnu.so
```

`build.sh [arch]` mirrors `tensor_cuda/build.sh`:
- It uses the newest `/usr/local/cuda-*` (CUDA 12.6 here). Override it with `CUDA_HOME`.
- The architecture defaults to 89.
- It builds Release, with IPO/LTO off, as core does.
- `PYBIND11_SOURCE_DIR` is passed as `-DFETCHCONTENT_SOURCE_DIR_PYBIND11`. Without it, CMake fetches v2.12.0 from
  GitHub.

`CMakeLists.txt` refuses two kinds of flag at configure time:
- fast-math flags (`--use_fast_math`);
- `--default-stream` flags. `conv.cu` also refuses `CUDA_API_PER_THREAD_DEFAULT_STREAM`.

`build/` is git-ignored.

## Use

```python
# PYTHONPATH=<Project-Tensor>/tc_signal
import torch, tc_signal as ts
ts.set_allocator(lambda s: torch.empty(s, device="cuda"))     # until P1T-M1 gives tensor_cuda.empty
y = ts.conv1d(x, w, b, stride=1, padding=(3, 3), dilation=3)
z = ts.conv_transpose1d(y, wt, bt, stride=10, padding=(5, 5))
```

## API

```python
tc_signal.conv1d(x, w, b=None, *, stride=1, padding=0, dilation=1, groups=1, out=None)
tc_signal.conv_transpose1d(x, w, b=None, *, stride=1, padding=0, output_padding=0, dilation=1, groups=1, out=None)
tc_signal.set_allocator(fn)   # fn(shape: tuple) -> fp32 CUDA DLPack tensor; returns the previous allocator
tc_signal.synchronize()       # P1T-FIX1: wait for the legacy stream, release every operand tc_signal still holds
```

The semantics are ONNX `Conv` and `ConvTranspose`, 1-D.

**Shapes**
- `x` is `[N, C_in, T]`.
- `b` is `[C_out]` or `None`.
- `padding` is an int or `(pad_begin, pad_end)`.
- `conv1d`: `w` is `[C_out, C_in/groups, K]`. The output length is
  `(T + pb + pe - dilation*(K-1) - 1)//stride + 1`.
- `conv_transpose1d`: `w` is `[C_in, C_out/groups, K]` (the ONNX layout). The output length is
  `stride*(T-1) + output_padding + dilation*(K-1) + 1 - pb - pe`. Output positions past the full transposed length
  hold only the bias.

**Attribute limits.** `stride`, `dilation` and `groups` must be >= 1; pads >= 0; `output_padding` >= 0 and
< `max(stride, dilation)`. An output length <= 0 is refused.

**Allocation**
- With `out=None`, the output comes from the module allocator. The function always returns `out`.
- The default allocator is `tensor_cuda.empty(*shape, dtype="float32")`, which P1T-M1 adds. Until then it raises a
  `RuntimeError` that names `set_allocator`.
- `set_allocator(None)` restores the default.
- The im2col scratch for `conv1d` also comes from the allocator. Its shape is `(C_in*K*T_out,)`, sized by the C++
  helper `conv1d_plan`. The scratch is skipped for the direct path (K 1, stride 1, no pads).

## Operand contract (DLPack consumer)

Every operand, the output and the scratch included, is an object with `__dlpack__` or a bare DLPack capsule.

**Acquisition**
1. If the object has `__dlpack_device__()`, it is asked first. A non-CUDA producer is refused there, before
   `__dlpack__` is called, because CPU producers such as NumPy reject a CUDA stream argument.
2. Then `obj.__dlpack__(stream=1, max_version=(1, 3))` is called. Stream 1 is the legacy default stream, the
   consumer's stream. Written operands also pass `copy=False`.
3. If the producer raises `TypeError` for the newer keywords, it is called again as `obj.__dlpack__(stream=1)`. This
   is the Python array API fallback.

**Both capsule kinds are read**
- `"dltensor"` (`DLManagedTensor`);
- `"dltensor_versioned"` (`DLManagedTensorVersioned`). For this kind, DLPack major version 1 is required.

**Consumption**
- A capsule is consumed once: it is renamed to `"used_*"`, and tc_signal owns the managed tensor from then on.
- Each producer deleter is called exactly once (P1T-FIX1, F01: see "Streams and lifetimes" for when):
  - for a call that reached its launch (success, or an error after a partial launch): only once the legacy stream
    has passed that call's launches, from the deferred-release list;
  - immediately, during unwinding, for a call refused after consumption but before any launch (nothing was queued);
  - by the producer's own capsule destructor if tc_signal never consumed the capsule (a refusal in the planning phase).

**Required and optional operands** (P1T-FIX1, F02). `None` is accepted only for an optional operand: `b` (conv),
`B` / `initial_h` / `initial_c` (lstm), `scale` / `bias` (norms), and `out` in the Python API (then the allocator makes
it). `None` for any required operand (`x` of all nine ops, `w` of both convolutions, `W` and `R` of the LSTM, and every
operand of the low-level `_tc_signal.*_run` calls) raises `TypeError` before any planning, allocation or consumption.

**Validation**, each failure a `TypeError` or `ValueError` (see Errors):
- device `kDLCUDA`, with one ordinal for every operand;
- dtype float32 (`kDLFloat`, 32 bits, 1 lane);
- the exact number of dimensions;
- no empty dimension;
- C-contiguous: NULL strides, or the compact stride on every dimension of extent > 1 (size-1 dimensions may carry
  any stride);
- `data + byte_offset` used as the base pointer, 4-byte aligned;
- exact shapes;
- written operands must not be read-only or a producer copy (versioned flags);
- the output's byte range must not overlap any input's or the scratch's, and the scratch must not overlap the inputs;
- checked 64-bit arithmetic (P1T-FIX1, F03): a negative rank, a NULL shape array, an element count, an fp32 byte total
  above `INT64_MAX`, or `data + byte_offset + bytes` wrapping the address space is refused (`ValueError`), and so is
  every planned size that overflows: output lengths (`conv1d`, `conv_transpose1d`, `T * scale`, `T + pads`), output
  and scratch element counts and bytes (im2col, LSTM `G_x`/`G_h`), LSTM `T*N` and `D*N*H`, norm rows and row lengths.
  A wrapped value is never used, never becomes an "empty" or negative extent, and no narrowing cast happens first.
- the zero-extent rule (P1T-FIX2, N2): every extent counts, also next to a zero one. The product of all extents with
  a zero counted as 1, and its fp32 byte size, must fit in int64, or the operand (or planned shape) is refused with
  `ValueError` before the empty-operand check; `[0, 2**62, 4]` is an overflow, not an empty operand. The same rule
  as NumPy's array constructor; core `tensor_cuda.empty` / `from_dlpack` / `__dlpack__` apply it too.
- the aggregate byte count (P1T-FIX3, V3-02): the fp32 byte counts of all operands of one call (inputs, outputs and
  scratch) are summed with checked addition before the first launch; a sum above `INT64_MAX` is refused with
  `ValueError` ("the aggregate fp32 byte count of this call's operands (their sum) overflows 64-bit arithmetic"), and
  the operands are released at once. Example: two disjoint fp32 descriptors of shape `(2**61 - 1,)` are each valid
  (`2**63 - 4` bytes) but their sum is not. The check runs in every entry point (it is the first step of the
  deferred-release retainer each one constructs right before its launch).

## Streams and lifetimes

- Everything is queued on the legacy default stream: the im2col, GEMM and bias work, and the ConvTranspose gather.
  This is TensorCUDA's only stream, and torch's default stream; both share the device's primary context.
  `tests/test_contract.py` proves the ordering both ways against torch with no explicit sync.
- `cudaGetLastError` is checked after every launch, and any error raises a `RuntimeError`.
- **Devices and CUDA error state (P1T-FIX4).** Every launch path makes the operands' device current through one
  checked guard (`src/cuda_guard.h`) and restores the caller's device; a failed restore raises on a success path, and
  where nothing can raise (a destructor) it is counted (`pending_stats()['device_restore_failures']`) and raised once
  by the next `synchronize()`. An event is destroyed on its own device. The calling thread's CUDA last-error slot
  (shared with TensorCUDA core: one `libcudart.so.12` instance) follows two rules:
  - whoever reports a CUDA error consumes it, so a failed call is reported exactly once, never again by a later,
    unrelated launch check;
  - right before a call's first launch, an error already pending in the slot (left by earlier work outside the call)
    is taken and raised as pre-existing (`RuntimeError`, nothing launched, operands released at once); so a launch
    check reports only that call's own error.
- **Operand lifetime contract (P1T-FIX1, F01: event-backed deferred release).** A consumed DLPack manager can be the
  LAST owner of its memory (a bare capsule, or a temporary copy a producer made for `__dlpack__`), and its producer
  may allocate on another stream. So tc_signal keeps every consumed manager of a call (inputs, outputs and scratch;
  the LSTM's whole set of up to 10) alive until the legacy stream has passed that call's last launch:
  - after the launches (and also when a launch throws part-way), one CUDA event is recorded on the legacy stream of
    the operands' device, and the call's managers go onto a module-level FIFO pending list with that event;
  - completed entries (`cudaEventQuery` == success) are released in FIFO order, on the calling host thread with the
    GIL held, at the start of the next tc_signal call; `tc_signal.synchronize()` synchronizes the legacy stream and
    releases everything; an `atexit` hook does the same at interpreter exit;
  - the list is bounded (`src/tuning.h`: 1024 entries, 256 MiB of operand bytes); a call that pushes it past a bound
    synchronizes the legacy stream of every pending device and releases the whole list. The byte total is saturating
    (P1T-FIX3, V3-02): a push whose sum would exceed `INT64_MAX` pins the total at `INT64_MAX` (which is over the
    bound, so that push drains, as exact arithmetic would) until the list is empty again, and nothing on that path
    throws (`pending_stats()['bytes_saturated']` / `['bytes_saturations']`). Below `INT64_MAX` the total is exact, so
    a total of exactly 256 MiB stays within the bound and one byte more drains;
  - deleters are never called from a CUDA host callback;
  - if the completion event cannot be recorded, the legacy stream is synchronized instead; if that fails too (a broken
    CUDA context), the managers are deliberately leaked: tc_signal never releases memory the GPU may still use.
  So a producer may free or reuse its memory on any stream as soon as its deleter runs; the old cross-stream caveat
  (keep non-legacy-stream tensors alive yourself) no longer applies. The cost: memory of the last calls is handed back
  at the next tc_signal call (or `synchronize()`), not at the end of the call. Choice: the deferred release (the
  order's preferred design) rather than a legacy-stream sync before release, which would block every call
  (receipts and per-call cost: `docs/P1T_FIX1_REPORT.md`). Counters: `_tc_signal.pending_stats()`.
- **Errors on the release paths (P1T-FIX2, N1).** A CUDA failure is never reported as success:
  - `tc_signal.synchronize()` raises `RuntimeError` naming the CUDA error when the current-device query
    (`cudaGetDevice`) fails, even with nothing pending (for example on a machine with no usable device), or when a
    device's legacy stream cannot be synchronized (that device's pending operands are deliberately leaked);
  - P1T-FIX4 (V4-04): `synchronize()` first takes the calling thread's pre-existing CUDA error (for example a refused
    launch, which queues no work, so every stream sync succeeds), still drains (releases what completed, leaks what it
    cannot prove done), and then raises `RuntimeError` naming that error. It takes the error (`cudaGetLastError`)
    rather than peeking: it reports it, so leaving it would make the next unrelated launch check report it again; a
    sticky error stays in the CUDA context anyway, and the no-device state is never reset by the runtime;
  - a release inside a tc_signal call cannot raise (it runs in a destructor). If its fallback sync or a bound drain
    fails there, the failure is recorded (`pending_stats()['unreported_release_errors']`) and the NEXT
    `synchronize()` raises it, once;
  - the `atexit` hook calls `synchronize()` and, on an error, writes one line to stderr
    (`tc_signal: synchronize() at exit failed (...)`) instead of raising. On a machine without a usable CUDA device,
    a process that imported tc_signal therefore logs that line at exit.
- **Allocation failure on the release paths (P1T-FIX2, N3)** never terminates the process. Releasing (the next-call
  drain, the fallback sync, the bound drain, `synchronize()`, the exit hook) allocates nothing, except the pending-list
  push, whose failure falls back to a synchronous release (`pending_stats()['push_alloc_failures']`). Everything else a
  release needs (the owner vector, the completion event and its pool slot) is allocated before the launch, where a
  failure refuses the call with `MemoryError` and releases its operands at once. Building `synchronize()`'s error
  message may allocate; a failure there is a `MemoryError`. P1T-FIX6 (round-6 review V6-03): every CUDA failure that
  message would have named (also those of `synchronize()`'s own drain: the current-device query, a stream sync, a
  completion-event query) is kept, with no allocation, until the `RuntimeError` naming it exists; so after such a
  `MemoryError` the next `synchronize()` raises it, exactly once (`pending_stats()['unreported_drain_failures']`).
- **A throwing producer deleter (P1T-FIX2, N4)** is contained: tc_signal catches whatever it throws, counts it
  (`pending_stats()['deleter_exceptions']`), never calls it again and goes on with the remaining managers. The DLPack
  ABI declares the deleter as a C function, so such a producer is already out of contract; tc_signal neither touches
  its memory again nor retries.
- **Interpreter exit (P1T-FIX5, round-5 review V5-01).** Every add-on callback that may take the GIL is an *admitted
  section* of one release lifecycle (`src/dlpack_operand.cpp`), independent of the pending list: each producer deleter
  call (the immediate release of a refused call, the next-call drain, `synchronize()`, the bound drain, the fallback
  sync), each drop of the last reference to a capsule tc_signal holds (its destructor is producer code: for an
  unconsumed capsule it calls the producer's deleter; P1T-FIX6 also admits consumed capsules, whose destructor may
  still run Python), and the producer's `__dlpack_device__` / `__dlpack__` calls. The `atexit` hook first runs
  `synchronize()`, then closes admission for good and waits, GIL released and with **no time bound**, until no admitted
  section of another thread is running; sections on its own thread's stack are excluded (the hook may run inside a
  callback). CPython runs atexit hooks before it marks the runtime finalizing, so finalization never starts under a
  running callback (before FIX5 a finite deleter still running on a daemon thread made CPython 3.12 exit that thread
  when it re-took the GIL, and the process aborted: `FATAL: exception not rethrown`, rc -6). After the close:
  - a deleter is NOT called: the manager is deliberately leaked (`pending_stats()['deleters_leaked_after_close']`);
  - a capsule reference is kept, so its destructor never runs (`['capsules_kept_after_close']`);
  - a producer `__dlpack__` is not called: the call raises `RuntimeError` (`['callbacks_refused_after_close']`).
  No add-on code enters Python for them. The wait writes a stderr line after `tc_signal._RELEASE_EXIT_DIAG_S` (5 s),
  then another each time the time waited doubles. **Consequence:** a producer deleter, capsule destructor or
  `__dlpack__` that never returns makes interpreter exit wait for it (the producer's contract violation; returning
  early would abort the process). Ctrl-C does not interrupt that wait; SIGTERM / SIGKILL still end the process.
  The GIL is released with `PyEval_SaveThread` and re-taken with an explicit `PyEval_RestoreThread` in an ordinary
  (not noexcept) frame, after the wait's mutex is dropped, so an admitted deleter that needs the GIL can finish, and a
  daemon thread that CPython exits while it re-takes the GIL unwinds cleanly (P1T-FIX6: the exit wait is a raw
  CPython function, so no pybind11 dispatcher frame owning a Python object is on that stack).
  **P1T-FIX6 (round-6 review V6-01): only the exit hook waits for another thread.** FIX5 also made `synchronize()`
  wait for releases running on other threads; a caller holding a lock that such a release needed then deadlocked
  (thread A holds a producer's allocator lock and calls `synchronize()`; thread B's refused operand's deleter needs
  that lock). That wait is removed: a release another thread's call already started finishes on that thread, and
  `synchronize()` releases what is still pending. **The exit hook's contract:** the exiting thread must not hold a lock
  that an admitted callback needs (CPython's own rule for the non-daemon threads it joins at exit; atexit callbacks
  normally run with no application lock held).
  Counters: `release_admission_closed`, `releases_in_flight`, `callbacks_in_flight`, `admitted_releases`,
  `admitted_callbacks`, `exit_hook_calls`, `exit_in_flight_at_close`, `exit_wait_us`, `exit_wait_diagnostics`,
  `exit_wait_overruns`, `deleters_leaked_finalizing`, `capsules_kept_finalizing`, `lifecycle_lock_failures` (FIX5's
  `sync_waits_for_other_threads`, `sync_wait_us`, `sync_wait_diagnostics` are gone with that wait).
- **Arguments are never converted by calling their Python methods (P1T-FIX6, sweep (g)).** Every `_tc_signal`
  binding takes `(*args, **kwargs)` and reads its arguments with C-level checks (`src/py_args.h`): ints are `int` or a
  subclass (an overridden `__index__` / `__int__` is not called), reals are `float` / `int` or subclasses, flags are
  `bool` / `int`, string lists are a `list` / `tuple` of `str`; anything else raises `TypeError` naming the parameter
  and the type, and a wrong arity or unknown keyword raises `TypeError` without calling `repr()`. Reason: such a method
  runs on a native stack, and on a daemon thread at interpreter exit it could be exited by CPython mid-conversion
  (core's `__dlpack__` crashed this way: round-6 review V6-02). The public wrappers (`tc_signal.conv1d`, ...) convert
  their arguments in Python first (`operator.index`, `float`), so they behave as before.
- **Recorded CUDA failures (P1T-FIX5, V5-02 and the class sweep).** A failed completion-event query in the next-call
  drain (anything but NotReady) and a failed completion-event record (the call then synchronizes instead) are recorded
  BEFORE their error-slot entry is consumed (`pending_stats()['event_query_failures']` / `['event_record_failures']`)
  and raised exactly once by the next `synchronize()`, which still releases what completed and leaks what it cannot
  prove done. Before FIX5 a one-shot query failure was consumed and never reported.
- Producers that are asked for `stream=1` order their pending work before the op (DLPack's forward direction).
- Outputs are written on the legacy stream. A consumer on another stream must order itself after it.

## cuBLAS

- There is one handle per device, created on first use, kept for the life of the process, and never destroyed at
  exit, which could race CUDA's own teardown.
- The handle settings are set explicitly:
  - `cublasSetStream(h, 0)`, the legacy stream;
  - `cublasSetMathMode(h, CUBLAS_DEFAULT_MATH)`: never TF32;
  - `cublasSetAtomicsMode(h, CUBLAS_ATOMICS_NOT_ALLOWED)`;
  - `cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST)`.
- `tc_signal._tc_signal.cublas_config(device)` reports these settings.
- **A new handle is owned until it is cached (P1T-FIX5, V5-03).** From `cublasCreate` until the cache insertion
  commits, a scope guard owns the handle: if a setting fails, the handle is destroyed and the destroy's status named in
  the `RuntimeError` (FIX4); if the insertion itself fails (host allocation, `MemoryError`), the guard destroys it while
  the exception propagates, checks the status and counts a failure without throwing
  (`pending_stats()['cublas_handle_destroy_failures']`). Nothing is cached then, and the next call creates a new handle.
- **cuBLAS's own internal workspace** is the one piece of device memory tc_signal causes to be allocated outside the
  caller's allocator. cuBLAS allocates it itself (cudaMalloc) for the handle.
  - Measured with `artifacts/p1t_m2/cublas_workspace_probe.py` as a cudaMemGetInfo delta on the RTX 4070 SUPER with
    cuBLAS 12.6.4.
  - The handle's first use costs **10.0 MiB**, and the runtime initialisation is counted in that figure.
  - The first and later SGEMMs add **0 MiB**.

## Numerics

- **fp32 math throughout.** There are no fast-math intrinsics and no TF32.
  - nvcc 12.6 defines no preprocessor macro for `--use_fast_math`, so fast-math cannot be refused at compile time.
  - Instead, `fp32_divide_probe` is compiled with the package's flags, and a test compares it bitwise against host IEEE
    division and checks that subnormal results are not flushed.
- **Deterministic.** Each output element is produced by one thread, or by one cuBLAS dot product, in a fixed order.
  There is no `atomicAdd` anywhere in the package.
- **conv1d**
  - Per batch element n: the im2col kernel computes `col[(c*K + k), t] = x[n, c, t*stride - pb + k*dilation]`, with 0
    outside `[0, T)`.
  - Then one `cublasSgemmStridedBatched` call over the groups computes `Y_g = W_g · Col_g`, in row-major terms. Through
    column-major cuBLAS it runs as `Y_g^T = Col_g^T · W_g^T`, with `m = T_out`, `n = C_out/groups`,
    `k = C_in/groups*K`.
  - A bias kernel follows.
  - The direct path for K 1, stride 1 and no pads runs the GEMM on `x` itself.
  - The accumulation order is cuBLAS's own, in fp32. The bias is added after the GEMM, with one rounding.
- **conv_transpose1d** uses the gather form, with one thread per output element (n, o, t):
  - `y = b[o] + Σ_k Σ_{c in group(o)} w[c, o - g·O_g, k] · x[n, c, (t + pb - k·dilation)/stride]`;
  - the sum runs over the taps where `t + pb - k·dilation` is >= 0, divisible by `stride`, and has its quotient < T;
  - the fixed order is k ascending (outer), then c ascending (inner), with `fmaf` accumulation from 0 and the bias
    added last.
- **Accuracy** on Voice's real fixtures (`run_p1t.py`, `tolerances_r2.json`): the largest normwise error is 1.4e-6
  against the 1e-5 gate. The worst ConvTranspose case, Kokoro's ups.0 with 1,024 terms per output, is at 1.23e-6. The
  order's blocked/pairwise option was therefore not needed.

## Errors

| Error | When |
|---|---|
| `TypeError` | a required operand is `None` (F02); an operand is not a DLPack producer or capsule, or is a capsule of another kind; the dtype is not float32 (fp16 operands are refused: expand them first); an attribute is not an int (bool included); `padding` has the wrong type; `set_allocator` gets a non-callable; `__dlpack__` returns something other than a capsule |
| `ValueError` | the device is not CUDA; the device ordinals differ; an operand is non-contiguous, misaligned, empty, of the wrong rank or of the wrong shape; the groups do not divide the channels; an attribute is out of range; the output length is <= 0; `output_padding >= max(stride, dilation)`; the output overlaps another operand; a written operand is read-only or a producer copy; a capsule was already consumed; the DLPack major version is not 1; a size, byte total, address range or planned geometry overflows 64-bit arithmetic (F03); the operands of one call total more than `INT64_MAX` bytes (V3-02) |
| `RuntimeError` | the default allocator is used without `tensor_cuda.empty` (the message names `set_allocator`); a CUDA launch or cuBLAS error; `synchronize()` cannot synchronize a device (its pending operands are then leaked, never released early); a CUDA error was already pending in the calling thread before a call's first launch, or when `synchronize()` was called (P1T-FIX4); the caller's CUDA device could not be restored (P1T-FIX4); an earlier failed event query or record, raised once by `synchronize()` (P1T-FIX5); a call that would invoke a producer's `__dlpack__` after the exit hook closed admission (P1T-FIX5) |
| `MemoryError` | an allocation failed before the launch (the call is refused, its operands released at once), while building `synchronize()`'s error message, or while caching a new cuBLAS handle (the handle is destroyed; P1T-FIX5) |

## Tests

```bash
cd <Project-Tensor> && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<Project-Tensor>/tc_signal \
  python3 -m pytest tc_signal/tests -q -p no:cacheprovider
```

- `test_conv.py` covers every census cell (`tcs_census.py`): Kokoro's attribute sets and the edge cells. For each one
  it checks the exact shape, that every element is written (the allocator pre-fills NaN), normwise N <= 1e-5 against a
  torch float64 CPU oracle (`tcs_oracle.py`), and that two calls are bitwise equal. It also checks the oracle against
  torch's native padding path.
- `test_contract.py` covers:
  - every documented refusal;
  - the DLPack consumer rules, using torch and a hand-built ctypes producer (`tcs_producer.py`): stream=1,
    max_version, copy=False, the TypeError fallback, `byte_offset`, size-1 strides, versioned and legacy capsules, bare
    capsules, and exactly one deleter call on every path;
  - the allocator;
  - stream ordering against torch;
  - the cuBLAS settings;
  - the fast-math canary.
- `test_fix1_findings.py` (P1T-FIX1) covers the review findings: the side-stream last-owner race (bare capsule and
  producer temporary), a partial-launch exception, the LSTM capsule set, release at the next call / `synchronize()` /
  `atexit`, the bytes bound, `None` for every required operand of the nine ops (each in its own process), and the
  overflow refusals in validation and in every plan function.
- `test_fix2_findings.py` (P1T-FIX2) covers the round-2 findings N1-N4 (release-path CUDA errors, the zero-extent
  rule, allocation failure and throwing deleters on every release path; native UBSan and fail-the-Nth-allocation
  harness `fix2_native_harness.cpp`).
- `test_fix3_findings.py` (P1T-FIX3) covers V3-02: the aggregate pre-launch refusal through real entry points (with
  and without a visible GPU), and the native UBSan harness `fix3_native_harness.cpp` for the saturating pending-list
  total, the bound comparison at the boundary and the real retainer destructor near `INT64_MAX`.
- `test_fix4_findings.py` (P1T-FIX4) covers V4-04 (`synchronize()` after a staged synchronous launch error, with
  nothing pending and with entries pending), the pre-launch error check of the ops, and the device/error sweep through
  an `LD_PRELOAD` shim over `libcudart.so.12` / `libcublas.so.12` that simulates a second current device on this
  one-GPU host (every launch, event create/record/destroy and sync on the operands' device, the caller's device
  restored; failed switch, failed restore, failed event destroy, failed cuBLAS setting).
- `test_fix5_findings.py` (P1T-FIX5) covers V5-01 (a deleter, capsule destructor or producer `__dlpack__` in flight
  on a daemon thread at ordinary interpreter exit: 0.1 / 5.5 / 12 s on the immediate-refusal and deferred paths, no
  visible GPU, the doubling diagnostic schedule; releases and callbacks after the close; the hook inside its own
  callback; a throwing C++ deleter; `synchronize()` NOT waiting for another thread's release since P1T-FIX6), V5-02 (one-shot query
  failures, reported once) and the failed event record, and V5-03 (a real `operator new` failure at each cache's
  insertion through an `LD_PRELOAD` C++ shim; the reviewer's CPU probe regenerated from these sources, `fix5_regen.py`).
- `test_fix6_findings.py` (P1T-FIX6) covers V6-01 (the reviewer's lock inversion with and without a visible GPU; a
  release started after `synchronize()` began; the exit hook still waiting, also for a consumed capsule's destructor),
  V6-03 (a drain failure of each kind followed by a message-allocation failure: raised once by the next
  `synchronize()`, on the real module and on a host trace module of this source) and sweep (g) (every binding's typed
  parameters refuse a recording object without calling it; arity / keyword errors never call `repr()`; keyword
  conversions on a daemon thread at exit are refused, not run).
- The module skips only when torch is not importable or there is no CUDA device.
- `TC_SIGNAL_TEST_RECEIPT=<path>` writes each cell's N and determinism to JSON.

## Vendored `include/dlpack/dlpack.h`

The header is DLPack v1.3 (dmlc/dlpack, Apache-2.0; header "Copyright (c) 2017 - by Contributors"). It was copied
byte-for-byte from torch 2.11's `torch/include/ATen/dlpack.h` (sha256
`62d0acfd441700f9b970ef2d694bde8628ccd4a36b0b5d311592768350f68d73`). That copy includes torch's
`TORCH_STABLE_ONLY` / `TORCH_TARGET_VERSION` include guard, which is inert here.

## Prior art

- **im2col + GEMM convolution:** Chellapilla, Puri & Simard, 2006. tc_signal adds the 1-D form with dilation and
  asymmetric pads, and batches the groups in a strided-batched SGEMM.
- **Gather-form transposed convolution:** the adjoint of strided convolution (Dumoulin & Visin, 2016). tc_signal's own
  parts are the loop order and the thread mapping.
- **Semantics:** the ONNX `Conv` and `ConvTranspose` operator specs.
- **DLPack:** dmlc/dlpack v1.3, and the Python array API data-interchange protocol (`__dlpack__(stream=, max_version=,
  copy=)`, `__dlpack_device__`). NumPy's and PyTorch's `from_dlpack` use the same TypeError fallback.
- **Pluggable allocator:** after CuPy's `cupy.cuda.set_allocator`. tc_signal's allocator returns DLPack tensors where
  CuPy's returns pointers.
- **Plan/run split** (size the scratch, let the caller allocate, then execute): the workspace-query pattern of
  cuDNN/cuBLASLt.
- **Fast-math canary:** no specific prior art known to me.
- **Deferred operand release (P1T-FIX1, F01):** PyTorch's `Tensor.record_stream` and its CUDA caching allocator, which
  defers a block's reuse until events recorded on the streams that used it complete, polled from the host
  (`process_events`), with an event pool; CuPy's stream-ordered memory pool; the DLPack consumer contract. Ours: the
  consumer-side FIFO of DLPack managers drained at tc_signal entry points with the GIL held, the count/bytes bound, and
  the leak-on-unfenceable rule. Details: `src/dlpack_operand.h`.
- **Checked arithmetic (P1T-FIX1, F03):** `__builtin_mul_overflow` / `__builtin_add_overflow` (GCC 5, 2015); C23
  `ckd_mul` / `ckd_add`; SEI CERT C INT30-C / INT32-C. Ours: where it is applied. Details: `src/checked_math.h`.
- **Aggregate check and saturating pending total (P1T-FIX3, V3-02):** the same checked builtins for the per-call sum;
  saturating arithmetic (DSP/SIMD saturated add, e.g. MMX/SSE `PADDS`, ARM `QADD`) for the pending-list total. Ours:
  checking the aggregate before the first launch, and the sticky saturated state that keeps the bound comparison
  conservative until the list empties. Details: `src/dlpack_operand.cpp` (`note_push`).
- **Release lifecycle at interpreter exit (P1T-FIX5, V5-01):** core's own `ImportReleaseLifecycle`
  (`tensor_cuda/src/bindings.cpp`, P1T-FIX2 N5 / P1T-FIX3 V3-01), reused: closing flag + in-flight count + condition
  variable, the unbounded wait with the GIL released and the doubling stderr schedule, the thread-local depth for
  self-exclusion, leak-without-Python after closing; CPython's atexit-before-finalizing order and `threading._shutdown`;
  RCU grace periods (McKenney & Slingwine, 1998) and the Linux `percpu_ref` kill-then-wait; the store-buffering
  (Dekker / Dijkstra, 1965) flag-plus-counter handshake; glibc/libstdc++ forced unwinding (`abi::__forced_unwind`).
  Ours: admitting the add-on's deleter, capsule-drop and producer-callback sites, and the explicit GIL re-take. Details:
  `src/dlpack_operand.cpp` (release lifecycle block).
- **No wait in ordinary calls (P1T-FIX6, V6-01):** Coffman, Elphick & Shoshani (1971), circular wait as a necessary
  deadlock condition; lock-ordering discipline. Taken: an ordinary call never blocks on another thread's callback, so
  no cycle through a caller-held lock can form. RCU (synchronize_rcu waits only for pre-existing readers) is why a
  separate "wait for releases" API would need a generation boundary and a no-locks-held contract; none is offered.
- **C-level argument reading (P1T-FIX6, sweep (g)):** CPython's C argument parser (keyword matching without `__eq__`),
  PyTorch 2.11's `Tensor.__dlpack__` (`type(stream) is not int` -> `TypeError`), pybind11 `py::args` / `py::kwargs`.
  Ours: applying it to every binding. Details: `src/py_args.h`.
- **Persist before reporting (P1T-FIX6, V6-03):** SQLite's OOM testing discipline (an error must survive the failure of
  its own report). Ours: clearing the records only after the Python `RuntimeError` exists.
- **Handle ownership until commit (P1T-FIX5, V5-03):** RAII and the scope guard with dismissal (Alexandrescu &
  Marginean, C/C++ Users Journal, 2000). Ours: checking and counting the cleanup status during unwinding.
- **Record before consume (P1T-FIX5, V5-02):** the CUDA 12.6 `cudaEventQuery` / per-thread error-slot contracts; no
  specific prior art known to me for the deferred one-shot report.
