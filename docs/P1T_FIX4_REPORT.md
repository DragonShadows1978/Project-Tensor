# P1T-FIX4 report: round-4 findings V4-01..V4-04 closed; device and CUDA-error sweep of the P1-T code

**Verdict: G1 PASS, G2 PASS, G3 PASS, G4 PASS, G5 PASS.**
- **V4-01 (MAJOR), export fence on the wrong device.** The fence now runs on the tensor's device under a checked guard:
  read the caller's device, switch, create the event, record it on the legacy stream, make the consumer wait, destroy
  the event, restore the caller's device. Every return is checked. On a failure the caller's device is restored first,
  then a RuntimeError names the failed step.
  - Before the fence, the tensor's device label is checked against its allocation (`cudaPointerGetAttributes`).
  - On one GPU the reviewer's `core_export_device.py` is INCONCLUSIVE (exit 3) on both builds.
  - An LD_PRELOAD shim simulates "device 1 current". On 62c8d9f it logs `create@1 record@1 wait@1 destroy@1`. On this
    build it logs `set0 create@0 record@0 wait@0 destroy@0 set1`, and the current device stays 1.
  - Mutant (a), the guard removed, fails 8 of these tests.
- **V4-03 (MINOR), import ignored a failed `cudaGetDevice`.** The result is now checked before the Storage is built and
  before the capsule is renamed. A failure raises RuntimeError naming the call. `core_import_query_error.py`: rc 1 on
  62c8d9f, rc 0 (PASS) here. Mutant (b) fails both the new test and the probe.
- **V4-04 (MINOR), `synchronize()` ignored an error already pending.** It now takes the pending error first
  (`cudaGetLastError`), drains as before, and then raises naming it. `sync_existing_error.py`: rc 1 on 62c8d9f, rc 0
  (PASS) here. Mutant (c) fails 3 tests and the probe.
- **V4-02 (NIT).** The `from_dlpack` docstring now describes the wait with no time bound.
- **G2 sweep.** All 148 product call sites in `git diff 9df5bef -- tensor_cuda tc_signal` are listed:
  - 114 CUDA runtime / cuBLAS calls and kernel launches, plus 34 error-string lookups;
  - 0 sites unannotated;
  - verdicts: **OK 90, FIXED 58**. 54 FIXED sites have a test. 4 have a stated reason why a test is not feasible.
  - The 45 test-code sites are listed too. One of them was also FIXED.
- **G3, no regression.**
  - Every round-1 to round-4 probe passes, or is INCONCLUSIVE only for lack of 2 GPUs.
  - `addon_lifetime` passes 3/3 runs, 0/12 incorrect each.
  - `core_exit_overrun` during and bounded give rc 0.
  - Voice T1-T6: PASS 236/236.
  - Combined session: the failing ids are exactly the 52 baseline ids (`G1 PASS`).
- **G4.** Mutants (a), (b) and (c) are all killed.
- **G5.** Scope is clean, the probe files are byte-identical, and no new core `__global__` was added.

Seat: Opus 5.5 (high). Order: `orders/P1T_FIX4_DEVICE_AND_ERRORS.md` (3c7dc22). Ledger: `docs/P1T_FIX4_LEDGER.md`.
Receipts: `artifacts/p1t_fix4/` and `logs/p1t_fix4_*`.
- "Before" binaries: wt_fix3 (62c8d9f), read-only. Core `35b82616…7a23`, tc_signal `22286121…535a`.
- After binaries (`artifacts/p1t_fix4/build_sha256.txt`):
  - core `_tensor_cuda.cpython-312-x86_64-linux-gnu.so`: `1da017ca2ec917019acc1ae9220fca87a74594257c053050056ed55eb2cd9b48`
  - `_tc_signal.cpython-312-x86_64-linux-gnu.so`: `909af0c5e0c76cfd16c4016abb72b35bdde5a01cc96505334c8cc36876426352`

## Per-finding table

| Finding | Fix (file:line) | Proving tests (new) | Probe before (62c8d9f) | Probe after (this build) |
|---|---|---|---|---|
| **V4-01** (MAJOR): the export fence was created and recorded on the CURRENT device | `tensor_cuda/src/bindings.cpp`:<br>• `:382` design and prior-art block<br>• `:414` `dlpack_fence_consumer(stream, device)`: checked `cudaGetDevice`, then switch, create, record (legacy stream of the tensor's device), wait, destroy on its own device, restore. Restore before the raise; each failed step named; slot entries consumed.<br>• `:526` label proof (BufferError on a mismatch)<br>• `:529` the call with `a.device.index`<br>`tensor_cuda/src/kernels.cu`:<br>• `:322` `cuda_storage_device_mismatch` (`cudaPointerGetAttributes`)<br>Docstring: `__dlpack__` (stream/device paragraph) | `tensor_cuda/tests/test_dlpack.py`, LD_PRELOAD shim `_FIX4_SHIM_C`:<br>• `:1496` runs_on_the_tensor_device_and_restores[handle, 2]: log `set0 create@0 record@0 wait@0 destroy@0 set1`, current 1<br>• `:1506` same device: no switch<br>• `:1516`/`:1530` failure of record / destroy / switch / create / wait: RuntimeError naming the step, device restored, slot clean<br>• `:1542` failed restore: raised and named<br>• `:1536`/`:1550` label proof: query failure and mismatch | `core_export_device.py`: **exit 3** (INCONCLUSIVE, one GPU)<br>shim tests on 62c8d9f: log `create@1 record@1 wait@1 destroy@1` (wrong device)<br>`core_interop_cpu` SPY trace: `tensor_device=0 recorded_device=1 caller_device=1` (it embeds 62c8d9f code, so it gives this on both arms) | `core_export_device.py`: **exit 3** (INCONCLUSIVE, one GPU)<br>shim tests: all pass. Mutant (a): 8 fail |
| **V4-03** (MINOR): the import ignored a failed `cudaGetDevice` and consumed the capsule | `tensor_cuda/src/bindings.cpp:900-923`: checked before Storage construction and the rename; RuntimeError naming `cudaGetDevice` and the CUDA error; slot entry consumed; `cur` starts at -1 | `test_dlpack.py`:<br>• `:1656` no device: versioned and legacy capsules give RuntimeError naming cudaGetDevice/cudaErrorNoDevice, the name is unchanged, and the deleter is not called; `empty()` also refuses with its own cudaGetDevice error<br>• `:1712` sticky context error: the census case (see design) | `core_import_query_error.py` (nogpu): **rc 1** `AssertionError: core import succeeded after failed device query; name=b'used_dltensor'`<br>`core_interop_cpu` import: rc 0 `IMPORT SUCCEEDED name=used_dltensor` | `core_import_query_error.py`: **rc 0** `PASS: failed device query refused before consumption`<br>`core_interop_cpu` import regenerated from this tree: rc 1 `IMPORT REFUSED from_dlpack: the current-device query cudaGetDevice failed (cudaErrorNoDevice …)` |
| **V4-04** (MINOR): `synchronize()` did not look at an error already pending in the slot | `tc_signal/src/dlpack_operand.cpp`:<br>• `:903` design and justification block<br>• `:926` `prior = cudaGetLastError()` first; held in `PendingState` until its message exists<br>• the message clause<br>`dlpack_operand.h`, `README.md` | `tc_signal/tests/test_fix4_findings.py`:<br>• `:158` nothing pending: NULL kernel (98) and a REAL launch-configuration refusal (9; 4096-thread block, built with nvcc by the test)<br>• `:167` entries pending: drained and released (none leaked), then raised<br>• both: slot clean after, and the next `synchronize()` succeeds (reported once) | `sync_existing_error.py`: **rc 1** `AssertionError: synchronize returned success with a preexisting launch error; slot=98`<br>`sync_slot_cpu`: rc 0 `synchronize returned success; previous_error_slot=9` | `sync_existing_error.py`: **rc 0** `PASS: staged error reported 98 98 …`<br>`sync_slot_cpu`: rc 1 `ERROR … already pending … (cudaErrorInvalidConfiguration …)` |
| **V4-02** (NIT): the docstring promised a bounded exit wait | `tensor_cuda/tensor_cuda/__init__.py:107-116`: no time bound; the knob is only the first-diagnostic delay; Ctrl-C does not interrupt the wait | source comparison | `:107-110` "at most `_DLPACK_SHUTDOWN_WAIT_S` seconds" | text matches the hook's documentation |

### V4-01 design

- **The device used.**
  - The device is the tensor's label, and it is proven against the allocation of element 0 (`cudaPointerGetAttributes`).
  - Why the proof is needed: the engine labels its own allocations `cuda:0` (`Device::from_string`, which predates
    P1-T) but allocates on whatever device is current.
  - So with device 1 current, a guard that trusted the label would fence device 0, while the engine's writes were on
    device 1. That is the V4-01 race again.
  - A mismatch, or memory the driver does not know, raises BufferError before any fence or capsule.
  - On one GPU the label always matches, so this changes nothing on this host. The shim tests prove the refusal.
- **Where the consumer waits.** The wait is issued inside the guard. So `stream=2` means the per-thread default stream of
  the TENSOR's device. That matches PyTorch 2.11's consumer, which passes `torch.cuda.current_stream(f'cuda:{device}')`
  (read in the installed `torch/utils/dlpack.py`).
- **A stream handle may belong to any device.** Two primary sources say so:
  - CUDA 12.6 `cuda_runtime_api.h`, `cudaStreamWaitEvent`: "event may be from a different device than stream";
  - Programming Guide 3.2.9.3: "cudaStreamWaitEvent() will succeed even if the input stream and input event are
    associated to different devices", while "cudaEventRecord() will fail if the input event and input stream are
    associated to different devices".
  - Receipt: `artifacts/p1t_fix4/cuda_doc_quotes.txt`, fetched through the proxy.
- **Restore before raise.** On a failure the caller's device is restored first, then RuntimeError names each failed
  step. If the restore itself fails, the message says the thread is still on the tensor's device.

### V4-03: the sticky-error census

- Measured: in a context broken by a device-side assert, `cudaGetDevice`, `cudaSetDevice(0)` and
  `cudaPointerGetAttributes` still succeed. Stream and event calls return 710 (`artifacts/p1t_fix4/cuda_semantics.txt`).
- So the import's only CUDA query passes, and it takes ownership exactly once (`used_dltensor_versioned`).
- The Storage's release then cannot drain the broken context. It leaks deliberately (`leaked_drain_failure 1`, the
  deleter is not called): F05's rule.
- I did not add a context-health query to the import. The import does not touch the GPU, and the release path already
  refuses to hand back memory it cannot prove idle. Test `:1712`.

### V4-04: clear or peek? Clear (take)

The measured receipts (`cuda_semantics.txt`) and the Programming Guide 3.2.12 say:

| Error kind | What happens in the slot | Consequence for synchronize() |
|---|---|---|
| Non-sticky (a refused launch, a failed API call) | Stays in the slot. A successful call does not reset it. The next failing call overwrites it (measured 98 → 101). | synchronize() reports it, so it consumes it. A peek would leave it for the next unrelated launch check of this shared `libcudart.so.12` instance (any tc_signal op, or core's `cuda_check_last`), which would report it a second time as its own. A peek could not preserve it anyway: a failing sync overwrites it. |
| Sticky (a device fault) | Lives in the CONTEXT. Taking the slot entry resets only the slot (710 then 0). Every later call that touches the context returns it again. | Clearing hides nothing. The drain's syncs fail, and the pending operands are leaked and reported, as before. |
| No usable device | `cudaErrorNoDevice` is never reset by `cudaGetLastError` (measured). | It is named on every synchronize(). |

- If building the message fails (MemoryError), the taken error is kept in `PendingState` and the next synchronize()
  raises it. It is never lost.
- The same rule runs through the sweep: whoever reports an error consumes it.

## G2: the sweep table

**How it was made.**
- `artifacts/p1t_fix4/sweep/extract_sites.py` lists every CUDA runtime / cuBLAS call token and every `<<<` launch on
  the added lines of `git diff 9df5bef -- tensor_cuda tc_signal` (working tree), plus the new files. It skips comments
  and string literals.
- `sweep/funcs.py` finds the function that encloses each site, by brace matching.
- `sweep/annotate.py` joins each site to exactly one annotation rule. An unmatched site fails the script.
- Outputs: `sweep/sweep.json` (one record per site) and `sweep/sweep_table.md` (below; sites with the same annotation
  in one function share a row).
- Classes: (a) implicit current device, (b) unchecked or ignored return or error state.

**Calls into pre-existing core code.** The P1-T code also calls core code that predates P1-T and is outside the order's
scope. Their own CUDA calls are not in the P1-T diff, so they are not rows. They are:
- `tensor_empty` → `NDArray::empty` → the `Storage` constructor (`cudaMalloc` / `cudaMallocAsync`, checked, raises
  without consuming; `empty` now consumes, row 95);
- `fill_` → `cuda_check_last` (`cudaGetLastError`, raises).

Product call sites: 148 (114 CUDA runtime/cuBLAS calls and kernel launches + 34 error-string lookups) in 111 table rows; verdicts: OK 90, FIXED 58, other 0. Test-code sites: 45 in 6 files (OK, except the FIX2 harness pre-drain sync: FIXED). Unannotated: 0.

| # | file:lines | function | call | device | device established by | return / error handling | verdict | proving test |
|---|---|---|---|---|---|---|---|---|
| 1 | `tcs/conv.cu:55` | `(message building, several)` | `cudaGetErrorName / cudaGetErrorString` | host only (error-string lookup) | host | returns a static string; nothing to check | **OK** | - |
| 2 | `tcs/conv.cu:56` | `check_cuda` | `cudaGetLastError` | error slot (calling thread; per runtime instance, not per device) | - | NEW: the raise consumes the failed call (same rule as lstm.cu) | **FIXED (b)** | `ccnone` |
| 3 | `tcs/conv.cu:62` | `check_launch` | `cudaGetLastError` | error slot (calling thread; per runtime instance, not per device) | - | post-launch check: reads and consumes; raises; the slot was clean right before the first launch (pre-launch take) | **OK** | `pre` |
| 4 | `tcs/conv.cu:89` | `handle_for` | `cublasCreate` | the operands' device (handle bound to the current device at creation) | guard (caller's DeviceGuard) | checked: RuntimeError | **OK** | `dev` |
| 5 | `tcs/conv.cu:95` | `handle_for` | `cublasDestroy` | the new handle's device | guard (caller's DeviceGuard) | NEW (failure path of a setting): status checked and named in the raise | **FIXED (b)** | `cublas` |
| 6 | `tcs/conv.cu:101` | `handle_for` | `cublasSetStream` | handle (its device current) | guard | checked; on failure the new handle is destroyed, RuntimeError (62c8d9f leaked it) | **FIXED (b)** | `cublas` |
| 7 | `tcs/conv.cu:102` | `handle_for` | `cublasSetMathMode` | handle (its device current) | guard | checked; on failure the new handle is destroyed, RuntimeError (62c8d9f leaked it) | **FIXED (b)** | `cublas` |
| 8 | `tcs/conv.cu:103` | `handle_for` | `cublasSetAtomicsMode` | handle (its device current) | guard | checked; on failure the new handle is destroyed, RuntimeError (62c8d9f leaked it) | **FIXED (b)** | `cublas` |
| 9 | `tcs/conv.cu:104` | `handle_for` | `cublasSetPointerMode` | handle (its device current) | guard | checked; on failure the new handle is destroyed, RuntimeError (62c8d9f leaked it) | **FIXED (b)** | `cublas` |
| 10 | `tcs/conv.cu:189` | `fp32_divide_probe` | `<<< kernel launch` | legacy stream of the operands' device | guard (DeviceGuard on common_device) | post-launch check (check_launch / check_cuda(cudaGetLastError)); slot clean before the first launch (pre-launch take) | **OK** | `dev` |
| 11 | `tcs/conv.cu:215,235` | `conv1d_forward` | `<<< kernel launch` | legacy stream of the operands' device | guard (DeviceGuard on common_device) | post-launch check (check_launch / check_cuda(cudaGetLastError)); slot clean before the first launch (pre-launch take) | **OK** | `dev` |
| 12 | `tcs/conv.cu:225` | `conv1d_forward` | `cublasSgemmStridedBatched` | legacy stream of the operands' device (handle's stream) | guard | status checked: RuntimeError; then check_launch | **OK** | `dev` |
| 13 | `tcs/conv.cu:247` | `conv_transpose1d_forward` | `<<< kernel launch` | legacy stream of the operands' device | guard (DeviceGuard on common_device) | post-launch check (check_launch / check_cuda(cudaGetLastError)); slot clean before the first launch (pre-launch take) | **OK** | `dev` |
| 14 | `tcs/conv.cu:262` | `cublas_config` | `cublasGetMathMode` | handle (its device current) | guard (cublas_config's DeviceGuard) | checked: RuntimeError | **OK** | `dev` |
| 15 | `tcs/conv.cu:263` | `cublas_config` | `cublasGetAtomicsMode` | handle (its device current) | guard | checked: RuntimeError | **OK** | `dev` |
| 16 | `tcs/conv.cu:264` | `cublas_config` | `cublasGetPointerMode` | handle (its device current) | guard | checked: RuntimeError | **OK** | `dev` |
| 17 | `tcs/conv.cu:265` | `cublas_config` | `cublasGetStream` | handle (its device current) | guard | checked: RuntimeError | **OK** | `dev` |
| 18 | `tcs/conv.cu:279` | `build_info` | `cudaRuntimeGetVersion` | host (runtime version) | host | checked: RuntimeError (check_cuda consumes) | **OK** | - |
| 19 | `tcs/conv.cu:280,281,282` | `build_info` | `cublasGetProperty` | host (library version) | host | checked: RuntimeError | **OK** | - |
| 20 | `tcs/conv.cu:283` | `build_info` | `cudaGetDevice` | host query (reported as current_device) | proven current | checked: RuntimeError (now consumed) | **FIXED (b) via check_cuda** | `ccnone` |
| 21 | `tcs/cuda_guard.h:44` | `raise_cuda_failure` | `cudaGetLastError` | error slot (calling thread; per runtime instance, not per device) | - | rule 1: every raise after a failed call consumes it (62c8d9f: the four guard copies raised without consuming) | **FIXED (b)** | `guard` |
| 22 | `tcs/cuda_guard.h:46` | `(message building, several)` | `cudaGetErrorName / cudaGetErrorString` | host only (error-string lookup) | host | returns a static string; nothing to check | **OK** | - |
| 23 | `tcs/cuda_guard.h:58` | `DeviceGuard` | `cudaGetDevice` | host query (caller's device) | proven current | checked: RuntimeError, consumed | **FIXED (b) (shared guard; was 4 copies)** | `guard` |
| 24 | `tcs/cuda_guard.h:61` | `DeviceGuard` | `cudaSetDevice` | the operands' device (DLPack device_id of every operand, checked equal: common_device) | guard (switch) | checked: RuntimeError naming the device, consumed, nothing switched | **FIXED (b)** | `guard` |
| 25 | `tcs/cuda_guard.h:69` | `restore` | `cudaSetDevice` | the caller's device (success-path restore) | proven current | NEW: checked, RuntimeError (the call's operands stay retained until the queued work completes); 62c8d9f only restored in the destructor and ignored the result | **FIXED (b)** | `guard` |
| 26 | `tcs/cuda_guard.h:79` | `~DeviceGuard` | `cudaSetDevice` | the caller's device (restore while unwinding) | proven current | was IGNORED; now checked: consumed, counted, raised once by the next synchronize() | **FIXED (b)** | `guard` |
| 27 | `tcs/cuda_guard.h:81` | `~DeviceGuard` | `cudaGetLastError` | error slot (calling thread; per runtime instance, not per device) | - | consumes the failed restore | **FIXED (b)** | `guard` |
| 28 | `tcs/cuda_guard.h:99` | `QuietDeviceGuard` | `cudaGetDevice` | host query (caller's device) | proven current | checked: reported via ok()/error(), consumed (unchanged from 62c8d9f) | **OK** | `fix1` |
| 29 | `tcs/cuda_guard.h:101,108` | `QuietDeviceGuard` | `cudaGetLastError` | error slot (calling thread; per runtime instance, not per device) | - | consumes the guard's own failure (unchanged) | **OK** | `fix1` |
| 30 | `tcs/cuda_guard.h:106` | `QuietDeviceGuard` | `cudaSetDevice` | the entry's / operands' device | guard (switch) | checked: reported via ok()/error(), consumed (unchanged) | **OK** | `fix1` |
| 31 | `tcs/cuda_guard.h:117` | `~QuietDeviceGuard` | `cudaSetDevice` | the caller's device (restore) | proven current | was IGNORED; now checked: consumed, counted, raised once by the next synchronize() | **FIXED (b)** | `quiet` |
| 32 | `tcs/cuda_guard.h:119` | `~QuietDeviceGuard` | `cudaGetLastError` | error slot (calling thread; per runtime instance, not per device) | - | consumes the failed restore | **FIXED (b)** | `quiet` |
| 33 | `tcs/dlpack_operand.cpp:495` | `sync_device_legacy` | `cudaStreamSynchronize` | legacy stream of the entry's device | guard (QuietDeviceGuard) | checked: failure -> that device's entries leaked and reported (N1) | **OK** | `fix1` |
| 34 | `tcs/dlpack_operand.cpp:496` | `sync_device_legacy` | `cudaGetLastError` | error slot (calling thread; per runtime instance, not per device) | - | consumes the failed sync (reported by N1 paths) | **OK** | `fix1` |
| 35 | `tcs/dlpack_operand.cpp:585` | `destroy_event_on_device` | `cudaEventDestroy` | the event's own device (62c8d9f: whatever was current) | guard (QuietDeviceGuard) | was IGNORED; now checked: consumed, counted, raised once by the next synchronize() | **FIXED (a)+(b)** | `destroy` |
| 36 | `tcs/dlpack_operand.cpp:587` | `destroy_event_on_device` | `cudaGetLastError` | error slot (calling thread; per runtime instance, not per device) | - | consumes the failed destroy | **FIXED (b)** | `destroy` |
| 37 | `tcs/dlpack_operand.cpp:700` | `sync_and_release_all_impl` | `cudaGetDevice` | host query (caller's current device, also synced) | proven current | checked: failure is an error of synchronize() (N1) | **OK** | `fix1` |
| 38 | `tcs/dlpack_operand.cpp:702,719,722` | `sync_and_release_all_impl` | `cudaGetLastError` | error slot (calling thread; per runtime instance, not per device) | - | consumes its own failures | **OK** | `fix1` |
| 39 | `tcs/dlpack_operand.cpp:714` | `sync_and_release_all_impl` | `cudaEventQuery` | the entry's event (any device: cross-device query legal per CUDA PG 3.2.9.3) | operand (entry) device, no switch needed | checked: not ready -> stop/leak by rule; error -> leak + reported (N1) | **OK** | `fix1` |
| 40 | `tcs/dlpack_operand.cpp:719` | `sync_and_release_all_impl` | `cudaPeekAtLastError` | error slot (calling thread; per runtime instance, not per device) | - | defensive NotReady clear (PG 3.2.12: NotReady is never put in the slot) | **OK** | `fix1` |
| 41 | `tcs/dlpack_operand.cpp:751` | `acquire_event` | `cudaEventCreateWithFlags` | the operands' device | guard (QuietDeviceGuard) | checked: RuntimeError before any launch, consumed | **OK** | `dev` |
| 42 | `tcs/dlpack_operand.cpp:753` | `acquire_event` | `cudaGetLastError` | error slot (calling thread; per runtime instance, not per device) | - | consumes the failed create | **OK** | `dev` |
| 43 | `tcs/dlpack_operand.cpp:755,756,817,942,948,949,954,963,964,971,972` | `(message building, several)` | `cudaGetErrorName / cudaGetErrorString` | host only (error-string lookup) | host | returns a static string; nothing to check | **OK** | - |
| 44 | `tcs/dlpack_operand.cpp:811` | `RetainUntilComplete::RetainUntilComplete` | `cudaGetLastError` | error slot (calling thread; per runtime instance, not per device) | - | NEW pre-launch take at the one constructor every launch path shares: a pending error raises as pre-existing, nothing launched, operands released at once (62c8d9f: launched, then reported it as its own launch failure) | **FIXED (b)** | `pre` |
| 45 | `tcs/dlpack_operand.cpp:838` | `RetainUntilComplete::~RetainUntilComplete` | `cudaEventRecord` | legacy stream of the operands' device | guard (QuietDeviceGuard) | checked: failure -> synchronous fallback, then release or leak (F01/N1) | **OK** | `dev` |
| 46 | `tcs/dlpack_operand.cpp:854` | `RetainUntilComplete::~RetainUntilComplete` | `cudaGetLastError` | error slot (calling thread; per runtime instance, not per device) | - | consumes the failed record | **OK** | `fix1` |
| 47 | `tcs/dlpack_operand.cpp:881` | `drain_completed` | `cudaEventQuery` | the entry's event (any device; cross-device query legal) | operand (entry) device, no switch needed | checked: not ready -> stop; error -> entry left for synchronize(), which re-queries and reports/leaks it | **OK** | `query` |
| 48 | `tcs/dlpack_operand.cpp:890` | `drain_completed` | `cudaPeekAtLastError` | error slot (calling thread; per runtime instance, not per device) | - | defensive NotReady clear | **OK** | `query` |
| 49 | `tcs/dlpack_operand.cpp:890` | `drain_completed` | `cudaGetLastError` | error slot (calling thread; per runtime instance, not per device) | - | defensive NotReady clear | **OK** | `query` |
| 50 | `tcs/dlpack_operand.cpp:892` | `drain_completed` | `cudaGetLastError` | error slot (calling thread; per runtime instance, not per device) | - | NEW: consumes a non-NotReady query failure (62c8d9f left it, so the same call's launch check reported it as its own) | **FIXED (b)** | `query` |
| 51 | `tcs/dlpack_operand.cpp:926` | `synchronize_and_release_all` | `cudaGetLastError` | error slot (calling thread; per runtime instance, not per device) | - | NEW V4-04 capture: takes the pre-existing error first, drains as before, then raises naming it | **FIXED (b) V4-04** | `v404` |
| 52 | `tcs/lstm.cu:76` | `(message building, several)` | `cudaGetErrorName / cudaGetErrorString` | host only (error-string lookup) | host | returns a static string; nothing to check | **OK** | - |
| 53 | `tcs/lstm.cu:77` | `check_cuda` | `cudaGetLastError` | error slot (calling thread; per runtime instance, not per device) | - | NEW: the raise consumes the failed call (62c8d9f left it for the next launch check) | **FIXED (b)** | `checkcuda` |
| 54 | `tcs/lstm.cu:83` | `check_launch` | `cudaGetLastError` | error slot (calling thread; per runtime instance, not per device) | - | post-launch check: reads and consumes; raises; the slot was clean right before the first launch (pre-launch take) | **OK** | `pre` |
| 55 | `tcs/lstm.cu:112` | `lstm_handle_for` | `cublasCreate` | the operands' device (handle bound to the current device at creation) | guard (caller's DeviceGuard) | checked: RuntimeError | **OK** | `dev` |
| 56 | `tcs/lstm.cu:118` | `lstm_handle_for` | `cublasDestroy` | the new handle's device | guard (caller's DeviceGuard) | NEW (failure path of a setting): status checked and named in the raise | **FIXED (b)** | `cublas` |
| 57 | `tcs/lstm.cu:124` | `lstm_handle_for` | `cublasSetStream` | handle (its device current) | guard | checked; on failure the new handle is destroyed, RuntimeError (62c8d9f leaked it) | **FIXED (b)** | `cublas` |
| 58 | `tcs/lstm.cu:125` | `lstm_handle_for` | `cublasSetMathMode` | handle (its device current) | guard | checked; on failure the new handle is destroyed, RuntimeError (62c8d9f leaked it) | **FIXED (b)** | `cublas` |
| 59 | `tcs/lstm.cu:126` | `lstm_handle_for` | `cublasSetAtomicsMode` | handle (its device current) | guard | checked; on failure the new handle is destroyed, RuntimeError (62c8d9f leaked it) | **FIXED (b)** | `cublas` |
| 60 | `tcs/lstm.cu:127` | `lstm_handle_for` | `cublasSetPointerMode` | handle (its device current) | guard | checked; on failure the new handle is destroyed, RuntimeError (62c8d9f leaked it) | **FIXED (b)** | `cublas` |
| 61 | `tcs/lstm.cu:216,220` | `lstm_forward` | `cudaMemcpyAsync` | legacy stream of the operands' device | guard | checked: RuntimeError (now consumed by check_cuda) | **FIXED (b) via check_cuda** | `checkcuda` |
| 62 | `tcs/lstm.cu:218,222` | `lstm_forward` | `cudaMemsetAsync` | legacy stream of the operands' device | guard | checked: RuntimeError (now consumed by check_cuda) | **FIXED (b) via check_cuda** | `checkcuda` |
| 63 | `tcs/lstm.cu:230,256` | `lstm_forward` | `<<< kernel launch` | legacy stream of the operands' device | guard (DeviceGuard on common_device) | post-launch check (check_launch / check_cuda(cudaGetLastError)); slot clean before the first launch (pre-launch take) | **OK** | `dev` |
| 64 | `tcs/lstm.cu:236` | `lstm_forward` | `cublasSgemm` | legacy stream of the operands' device (handle's stream) | guard | status checked: RuntimeError; then check_launch | **OK** | `dev` |
| 65 | `tcs/lstm.cu:249` | `lstm_forward` | `cublasSgemmStridedBatched` | legacy stream of the operands' device (handle's stream) | guard | status checked: RuntimeError; then check_launch | **OK** | `dev` |
| 66 | `tcs/lstm.cu:271` | `lstm_cublas_config` | `cublasGetMathMode` | handle (its device current) | guard (cublas_config's DeviceGuard) | checked: RuntimeError | **OK** | `dev` |
| 67 | `tcs/lstm.cu:272` | `lstm_cublas_config` | `cublasGetAtomicsMode` | handle (its device current) | guard | checked: RuntimeError | **OK** | `dev` |
| 68 | `tcs/lstm.cu:273` | `lstm_cublas_config` | `cublasGetPointerMode` | handle (its device current) | guard | checked: RuntimeError | **OK** | `dev` |
| 69 | `tcs/lstm.cu:274` | `lstm_cublas_config` | `cublasGetStream` | handle (its device current) | guard | checked: RuntimeError | **OK** | `dev` |
| 70 | `tcs/norms.cu:72` | `(message building, several)` | `cudaGetErrorName / cudaGetErrorString` | host only (error-string lookup) | host | returns a static string; nothing to check | **OK** | - |
| 71 | `tcs/norms.cu:73` | `check_cuda` | `cudaGetLastError` | error slot (calling thread; per runtime instance, not per device) | - | NEW: the raise consumes the failed call (same rule as lstm.cu) | **FIXED (b)** | `ccnone` |
| 72 | `tcs/norms.cu:158` | `row_norm_forward` | `<<< kernel launch` | legacy stream of the operands' device | guard (DeviceGuard on common_device) | post-launch check (check_launch / check_cuda(cudaGetLastError)); slot clean before the first launch (pre-launch take) | **OK** | `dev` |
| 73 | `tcs/norms.cu:160` | `row_norm_forward` | `cudaGetLastError` | error slot (calling thread; per runtime instance, not per device) | - | post-launch check, raise | **OK** | `dev` |
| 74 | `tcs/small_ops.cu:44` | `(message building, several)` | `cudaGetErrorName / cudaGetErrorString` | host only (error-string lookup) | host | returns a static string; nothing to check | **OK** | - |
| 75 | `tcs/small_ops.cu:45` | `check_cuda` | `cudaGetLastError` | error slot (calling thread; per runtime instance, not per device) | - | NEW: the raise consumes the failed call (same rule as lstm.cu) | **FIXED (b)** | `ccnone` |
| 76 | `tcs/small_ops.cu:99` | `upsample_nearest1d_forward` | `<<< kernel launch` | legacy stream of the operands' device | guard (DeviceGuard on common_device) | post-launch check (check_launch / check_cuda(cudaGetLastError)); slot clean before the first launch (pre-launch take) | **OK** | `dev` |
| 77 | `tcs/small_ops.cu:101` | `upsample_nearest1d_forward` | `cudaGetLastError` | error slot (calling thread; per runtime instance, not per device) | - | post-launch check, raise | **OK** | `dev` |
| 78 | `tcs/small_ops.cu:111` | `reflection_pad1d_forward` | `<<< kernel launch` | legacy stream of the operands' device | guard (DeviceGuard on common_device) | post-launch check (check_launch / check_cuda(cudaGetLastError)); slot clean before the first launch (pre-launch take) | **OK** | `dev` |
| 79 | `tcs/small_ops.cu:113` | `reflection_pad1d_forward` | `cudaGetLastError` | error slot (calling thread; per runtime instance, not per device) | - | post-launch check, raise | **OK** | `dev` |
| 80 | `tcs/small_ops.cu:120` | `rint_forward` | `<<< kernel launch` | legacy stream of the operands' device | guard (DeviceGuard on common_device) | post-launch check (check_launch / check_cuda(cudaGetLastError)); slot clean before the first launch (pre-launch take) | **OK** | `dev` |
| 81 | `tcs/small_ops.cu:121` | `rint_forward` | `cudaGetLastError` | error slot (calling thread; per runtime instance, not per device) | - | post-launch check, raise | **OK** | `dev` |
| 82 | `core/bindings.cpp:431,921,1042` | `(message building, several)` | `cudaGetErrorName / cudaGetErrorString` | host only (error-string lookup) | host | returns a static string; nothing to check | **OK** | - |
| 83 | `core/bindings.cpp:434` | `dlpack_fence_consumer` | `cudaGetDevice` | host query (caller's device) | proven current | checked: RuntimeError, consumed, nothing switched (62c8d9f: never asked) | **FIXED (a) V4-01** | `v401f` |
| 84 | `core/bindings.cpp:436,444,467` | `dlpack_fence_consumer` | `cudaGetLastError` | error slot (calling thread; per runtime instance, not per device) | - | consumes the fence's own failed calls (reported by the raise) | **FIXED (b)** | `v401f` |
| 85 | `core/bindings.cpp:442` | `dlpack_fence_consumer` | `cudaSetDevice` | the tensor's device (label proven against the allocation) | guard (switch) | checked: RuntimeError, caller unchanged, consumed | **FIXED (a) V4-01** | `v401f` |
| 86 | `core/bindings.cpp:453` | `dlpack_fence_consumer` | `cudaEventCreateWithFlags` | the tensor's device (62c8d9f: the CURRENT device) | guard | checked: device restored, consumed, RuntimeError naming the step | **FIXED (a) V4-01** | `v401` |
| 87 | `core/bindings.cpp:456` | `dlpack_fence_consumer` | `cudaEventRecord` | legacy stream of the tensor's device (62c8d9f: of the CURRENT device) | guard | checked: device restored, consumed, RuntimeError | **FIXED (a) V4-01** | `v401` |
| 88 | `core/bindings.cpp:460` | `dlpack_fence_consumer` | `cudaStreamWaitEvent` | the consumer's stream (handle: any device, cross-device wait legal per CUDA PG 3.2.9.3; 2: per-thread stream of the tensor's device) | guard (issued with the tensor's device current) | checked: device restored, consumed, RuntimeError | **FIXED (a) V4-01** | `v401` |
| 89 | `core/bindings.cpp:464` | `dlpack_fence_consumer` | `cudaEventDestroy` | the event's own device | guard | was IGNORED in 62c8d9f; now checked: RuntimeError after the restore | **FIXED (b)** | `v401f` |
| 90 | `core/bindings.cpp:465` | `dlpack_fence_consumer` | `cudaSetDevice` | the caller's device (restore) | proven current (read above) | checked: RuntimeError says the thread is still on the tensor's device; consumed | **FIXED (a) V4-01** | `v401f` |
| 91 | `core/bindings.cpp:917` | `tensor_from_dlpack_capsule` | `cudaGetDevice` | host query (the import's current device) | proven current | was IGNORED in 62c8d9f (cur stayed 0); now checked BEFORE Storage construction and the capsule rename: RuntimeError naming cudaGetDevice, consumed | **FIXED (b) V4-03** | `v403` |
| 92 | `core/bindings.cpp:919` | `tensor_from_dlpack_capsule` | `cudaGetLastError` | error slot (calling thread; per runtime instance, not per device) | - | consumes the failed query (reported by the raise) | **FIXED (b) V4-03** | `v403` |
| 93 | `core/bindings.cpp:1038` | `tensor_empty` | `cudaGetDevice` | host query (the allocation's device = current) | proven current | checked: RuntimeError, nothing allocated; the label is set to it (62c8d9f: label cuda:0 whatever was current) | **FIXED (a)** | `empty` |
| 94 | `core/bindings.cpp:1040` | `tensor_empty` | `cudaGetLastError` | error slot (calling thread; per runtime instance, not per device) | - | consumes the failed device query (reported by the raise) | **FIXED (b)** | `v403` |
| 95 | `core/bindings.cpp:1061` | `tensor_empty` | `cudaGetLastError` | error slot (calling thread; per runtime instance, not per device) | - | consumes a failed allocation's entry (the engine's Storage constructor raises without consuming) | **FIXED (b)** | `empty` |
| 96 | `core/kernels.cu:234` | `drain_device_legacy_stream` | `cudaGetDevice` | host query (caller's current device) | proven current | checked: failure -> no release, deliberate leak (F05); slot entry consumed | **OK** | `fix1` |
| 97 | `core/kernels.cu:237` | `drain_device_legacy_stream` | `cudaSetDevice` | the Storage's device (label = producer device, checked = current at import) | guard (switch) | checked: failure -> leak; consumed | **OK** | `fix1` |
| 98 | `core/kernels.cu:240` | `drain_device_legacy_stream` | `cudaStreamSynchronize` | legacy stream of the Storage's device | guard | checked: failure -> leak (no completion proof); consumed | **OK** | `fix1` |
| 99 | `core/kernels.cu:241` | `drain_device_legacy_stream` | `cudaGetLastError` | error slot (calling thread; per runtime instance, not per device) | - | consumes the drain's own failure (reported by the leak counter) | **OK** | `fix1` |
| 100 | `core/kernels.cu:242` | `drain_device_legacy_stream` | `cudaSetDevice` | the caller's device (restore) | proven current (read above) | was IGNORED in 62c8d9f; now checked: counted (device_restore_failures), consumed, never thrown (destructor path) | **FIXED (b)** | `drain` |
| 101 | `core/kernels.cu:243` | `drain_device_legacy_stream` | `cudaGetLastError` | error slot (calling thread; per runtime instance, not per device) | - | consumes the failed restore's entry | **FIXED (b)** | `drain` |
| 102 | `core/kernels.cu:254` | `raise_cuda_failure` | `cudaGetLastError` | error slot (calling thread; per runtime instance, not per device) | - | report-and-consume rule for every raise after a failed call (guard, label proof) | **FIXED (b)** | `fill` |
| 103 | `core/kernels.cu:255,466` | `(message building, several)` | `cudaGetErrorName / cudaGetErrorString` | host only (error-string lookup) | host | returns a static string; nothing to check | **OK** | - |
| 104 | `core/kernels.cu:268` | `CheckedDeviceGuard` | `cudaGetDevice` | host query (caller's device) | proven current | checked: RuntimeError, consumed, nothing switched | **FIXED (a) new guard** | `fill` |
| 105 | `core/kernels.cu:271` | `CheckedDeviceGuard` | `cudaSetDevice` | the tensor's device (label proven against the allocation) | guard (switch) | checked: RuntimeError naming the device, consumed | **FIXED (a) new guard** | `fill` |
| 106 | `core/kernels.cu:282` | `restore` | `cudaSetDevice` | the caller's device (success-path restore) | proven current (read by the guard) | checked: RuntimeError (the write is queued and counted in the revision first) | **FIXED (a)** | `fill` |
| 107 | `core/kernels.cu:290` | `~CheckedDeviceGuard` | `cudaSetDevice` | the caller's device (restore while unwinding) | proven current | checked: counted, consumed, never thrown | **FIXED (a)** | `fill` |
| 108 | `core/kernels.cu:291` | `~CheckedDeviceGuard` | `cudaGetLastError` | error slot (calling thread; per runtime instance, not per device) | - | consumes the failed restore's entry | **FIXED (b)** | `fill` |
| 109 | `core/kernels.cu:325` | `cuda_storage_device_mismatch` | `cudaPointerGetAttributes` | host-side pointer query (no device needs to be current; legal for any device) | host | checked: RuntimeError, consumed; result compared with the label (mismatch / unregistered -> refusal) | **FIXED (a) new label proof** | `label` |
| 110 | `core/kernels.cu:463` | `fill_` | `cudaGetLastError` | error slot (calling thread; per runtime instance, not per device) | - | pre-launch take: a pending error raises as pre-existing, nothing launched (CUDA PG 3.2.12 rule) | **FIXED (b)** | `fillpend` |
| 111 | `core/kernels.cu:469` | `fill_` | `<<< kernel launch` | legacy stream of the tensor's device (62c8d9f: the CURRENT device) | guard + label proof | post-launch cuda_check_last (pre-existing helper: cudaGetLastError, raise); a pending error was taken first | **FIXED (a)** | `fill` |

Test keys:
- `v401`: core test_fix4_v401_export_fence_* (7 cases) + mutant (a)
- `v401f`: core test_fix4_v401_export_fence_failure_* / create_or_wait_* / failed_restore_*
- `label`: core test_fix4_export_refuses_a_label_* / fill_refuses_a_label_* / export_label_proof_query_failure_*
- `fill`: core test_fix4_fill_launches_on_the_tensor_device_* / fill_guard_failures_* (3)
- `fillpend`: core test_fix4_fill_reports_a_pending_error_*
- `drain`: core test_fix4_storage_drain_failed_restore_is_counted_not_ignored
- `v403`: core test_fix4_v403_import_refuses_* (+ empty in no-device) + mutant (b); probe core_import_query_error.py
- `empty`: core test_fix4_empty_failed_allocation_is_reported_once / empty_labels_the_current_device_*
- `v404`: tcs test_fix4_v404_* (3) + mutant (c); probe sync_existing_error.py
- `pre`: tcs test_fix4_op_reports_a_pending_error_as_preexisting_*
- `guard`: tcs test_fix4_failed_switch_* / failed_restore_on_the_success_path_* / failed_restore_while_unwinding_*
- `quiet`: tcs test_fix4_failed_restore_where_nothing_can_raise_* (QuietDeviceGuard in ~RetainUntilComplete)
- `dev`: tcs test_fix4_device_rint_* / device_conv1d_* / device_lstm_* (pin: every logged call on the operand device, caller restored)
- `checkcuda`: tcs test_fix4_check_cuda_raise_consumes_the_failed_call
- `destroy`: tcs test_fix4_event_is_destroyed_on_its_own_device[ok, fail]
- `query`: tcs test_fix4_failed_event_query_in_the_drain_is_consumed_there
- `cublas`: tcs test_fix4_failed_cublas_setting_* (conv.cu handle) / failed_lstm_cublas_setting_* (lstm.cu handle)
- `ccnone`: no test feasible: this check_cuda copy only ever receives cudaGetLastError()'s own result (already consumed; the extra take is a no-op) or, in build_info, calls that fail only with no usable device, where the runtime keeps cudaErrorNoDevice in the slot permanently (measured), so consumption is unobservable; the identical lstm.cu copy is tested (checkcuda)
- `fix1`: existing FIX1/FIX2 tests (test_fix1_f05_*, tcs test_fix2_findings N1/N3)

Test code:

| file | role | sites (line call) | handling | verdict |
|---|---|---|---|---|
| `tc_signal/tests/fix2_native_harness.cpp` | FIX2 native harness | 163 cudaStreamSynchronize; 164 cudaGetLastError; 166 cudaGetErrorName; 233 cudaMalloc | cudaMalloc checked (pre-existing); the pre-drain cudaStreamSynchronize result was IGNORED and is now checked (the run fails, slot consumed): FIXED (b), test code, exercised by tcs test_fix2_findings oom_sweep | FIXED (b) (test code) |
| `tc_signal/tests/fix3_native_harness.cpp` | FIX3 native harness (unchanged by FIX4) | 260 cudaGetDevice | checked: the harness exits 2 (pre-existing) | OK (test code) |
| `tc_signal/tests/fix4_cudart_shim.c` | interposer DEFINITION (LD_PRELOAD shim); forwards to the real call through real_<name>() | 51 cudaGetDevice; 56 cudaSetDevice; 68 cudaEventCreateWithFlags; 74 cudaEventRecord; 80 cudaEventQuery; 85 cudaEventDestroy; 91 cudaStreamSynchronize; 96 cudaLaunchKernel; 102 cudaMemsetAsync; 107 cublasSetMathMode; 112 cublasDestroy_v2 | returns the real result, or a GENUINE injected failure (real cudaSetDevice(-1)) | OK (test code) |
| `tc_signal/tests/fix4_stage_kernel.cu` | test helper: stages a real launch-configuration refusal | 12 cudaGetLastError; 13 <<< kernel launch; 14 cudaPeekAtLastError | returned to the test, which asserts it | OK (test code) |
| `tc_signal/tests/test_fix4_findings.py` | test script call (ctypes): stages / reads the error slot | 100 cudaGetLastError; 101 cudaLaunchKernel; 144 cudaPeekAtLastError | asserted by the test | OK (test code) |
| `tensor_cuda/tests/test_dlpack.py` | test script call (ctypes on the shared libcudart.so.12) or embedded shim definition | 789 cudaGetDevice; 795 cudaSetDevice; 806 cudaStreamSynchronize; 1331 cudaGetDevice; 1332 cudaSetDevice; 1342 cudaEventCreateWithFlags; 1347 cudaEventRecord; 1352 cudaStreamWaitEvent; 1357 cudaEventDestroy; 1362 cudaStreamSynchronize; 1363 cudaLaunchKernel; 1368 cudaPointerGetAttributes; 1398 cudaStreamCreateWithFlags; 1627 cudaGetDevice; 1692 cudaStreamQuery; 1693 cudaGetLastError; 1737 cudaGetLastError; 1738 cudaLaunchKernel; 1739 cudaPeekAtLastError; 1739 cudaPeekAtLastError; 1742 cudaPeekAtLastError; 1758 cudaGetLastError; 1776 cudaGetLastError | asserted by the test | OK (test code) |

**What the sweep found and fixed, by class.**
- **(a) Implicit current device** (13 sites FIXED, plus 1 site in both classes):
  - the V4-01 fence;
  - `fill_` launched on the current device. It now uses a guard plus the label proof;
  - `empty` labeled `cuda:0` whatever device was current. The label is now the checked current device, and an explicit
    `cuda:N` must name it;
  - tc_signal destroyed recycled-failure events on whatever device was current. Now an event is destroyed on its own
    device;
  - the label proof at export and `fill_`.
- **(b) Unchecked or ignored returns and error state** (44 sites FIXED):
  - every device restore was ignored: the four tc_signal `DeviceGuard` copies, `QuietDeviceGuard`, and core's external
    drain;
  - the fence's `cudaEventDestroy` result was ignored;
  - raises did not consume the failed call's slot entry: the guards, `check_cuda`, the import, and `empty`'s failed
    allocation;
  - `drain_completed` left a query failure in the slot;
  - ops reported an error that was already pending as their own launch failure, after launching. They now take it right
    before the first launch;
  - V4-03 and V4-04;
  - a new cuBLAS handle leaked when one of its settings failed;
  - test code: the FIX2 harness's pre-drain sync result.
- **Where nothing can raise** (destructors), a failure is counted and consumed:
  - core reports it in `_dlpack_release_stats()['device_restore_failures']`;
  - tc_signal counts it in `pending_stats()['device_restore_failures' / 'event_destroy_failures']`, and the next
    `synchronize()` raises it once.

## Probe before/after table

Every probe in `artifacts/p1t_rev4/probes/README.md`, on both builds. Logs: `artifacts/p1t_fix4/probe_runs/rev4_*`,
summaries `rev4_summary_{before,after}_{cpu,gpu,rf}.txt`. Every run was a fresh process under
`flock -w 900 /tmp/forge-gpu.lock timeout <README timeout>`. Conditions: `nogpu` means `CUDA_VISIBLE_DEVICES=` (the
reviewer's sandbox); `gpu` means the device is visible.

| Probe | How it ran | Before (62c8d9f) | After (this build) | Result |
|---|---|---|---|---|
| `core_export_device.py` | unmodified, GPU | exit 3 `INCONCLUSIVE: two GPUs required` | exit 3 | INCONCLUSIVE (one GPU); V4-01 is decided by the shim tests and mutant (a) |
| `core_import_query_error.py` | unmodified, nogpu | **rc 1** `core import succeeded after failed device query; name=b'used_dltensor'` | **rc 0** `PASS` | **FIXED** |
| `sync_existing_error.py` | unmodified, GPU | **rc 1** `synchronize returned success with a preexisting launch error; slot=98` | **rc 0** `PASS: staged error reported 98 98` | **FIXED** |
| `addon_lifetime.py` | unmodified, GPU | rc 0, 12/12 correct, 0 incorrect | rc 0, 12/12 correct, 0 incorrect | PASS |
| `core_edges.py` | unmodified, GPU | rc 0 | rc 0 | PASS |
| `core_device_drain.py` | unmodified, GPU | exit 3 | exit 3 | INCONCLUSIVE (one GPU) |
| `core_exit_overrun.py during` | unmodified, GPU | rc 0 (3.2 s) | rc 0 (3.4 s) | PASS |
| `core_exit_overrun.py bounded` | unmodified, GPU | rc 0 (7.6 s; 5.5 s deleter finished, then finalization) | rc 0 (7.8 s) | PASS |
| `numerics_and_streams.py` | unmodified, GPU | rc 0 PASS | rc 0 PASS | PASS |
| `release_failures.py` | before: path copy (root = wt_fix3: its own tests on its own build); after: original | rc 0, 66 passed, no skip | rc 0, 66 passed, no skip | PASS |
| `cuda_errors_multidevice.py` | unmodified, GPU | exit 3 | exit 3 | INCONCLUSIVE (one GPU) |
| `host_checks.py` (15 cases + core_cpu 3 + cpu_native 2) | path copy, nogpu / gpu | nogpu 20/20 rc 0; gpu 19 rc 0 + `oom_sync` -6 | same rc in all 40 case-condition pairs; texts identical, except the new pre-existing clause in the nogpu atexit line (`rev4_host_checks_compare.txt`) | PASS |
| `operand_census.py` | path copy, nogpu / gpu | PASS, 17 cases, 7 once-only | same | PASS |
| `real_metadata_plans.py` | path copy, nogpu / gpu | PASS, 129 cases | same | PASS |
| `lifecycle.cpp` + `lifecycle_timeout.py` control / short / default | probe source unmodified (its slice is verbatim in both arms), nogpu / gpu | rc 0 x6 | rc 0 x6 | PASS |
| `core_interop_cpu.cpp` SPY | built and run as written | rc 0 trace `tensor_device=0 recorded_device=1 caller_device=1` | same (it embeds 62c8d9f code: arm-independent) | not a decider (see D9) |
| `core_interop_cpu.cpp` import, as written | nogpu | rc 0 `IMPORT SUCCEEDED name=used_dltensor` | same (62c8d9f slice) | not a decider |
| `core_interop_cpu.cpp` import, regenerated from this tree (D9) | nogpu | n/a | **rc 1** `IMPORT REFUSED … cudaGetDevice failed (cudaErrorNoDevice …)` | **FIXED** |
| `sync_slot_cpu.cpp` | built against each arm's `dlpack_operand.cpp` (D9), nogpu / gpu | rc 0 `synchronize returned success; previous_error_slot=9` | **rc 1** `ERROR … already pending … (cudaErrorInvalidConfiguration …)` | **FIXED** (the trace no longer reproduces the defect) |
| `fix2_harness ck_census` | each arm's harness, UBSan, nogpu / gpu | 18 cases, 0 failures, no UBSan | same | PASS |
| `fix3_harness accounting` | each arm's harness, UBSan, nogpu / gpu | 17 cases, 0 failures, no UBSan | same | PASS |

**Rounds 1 to 3, after build** (`probe_runs/round123_summary_after.txt`):
- round 1:
  - `addon_lifetime` runs 1-3: exit 0, 0/12 incorrect, 12/12 correct each;
  - `core_edges` 0;
  - `core_shutdown --normal` 0 (`PASS: ordinary foreign-thread release`) and finalizing 0 (`finalizing: True … PASS`);
  - `numerics_and_streams` 0;
  - `host_checks`: every action's rc and final line identical to FIX3's receipt (`round1_host_checks_vs_fix3.txt`);
  - `core_device_drain` 3 (INCONCLUSIVE);
- round 2: `core_cpu` regenerated (self-check identical=True); `host_checks` nogpu 20/20 rc 0; gpu 19 rc 0 plus
  `oom_sync` -6 (FIX2 D4: the probe's own pybind11 teardown, the same on both arms);
- round 3: `core_exit_overrun` during 0 (3.2 s) and bounded 0 (7.9 s).

## Gate table

| Gate | Criterion (registered in the ledger at 02:19, before any change) | Result | Receipt |
|---|---|---|---|
| G1 | `core_import_query_error.py` and `sync_existing_error.py` fail on 62c8d9f and pass here | **PASS**: rc 1 → 0 and rc 1 → 0 | `probe_runs/rev4_core_import_query_error_*`, `rev4_sync_existing_error_*` |
| G1 | `core_export_device.py` INCONCLUSIVE; the shim test proves the guard switches and restores, and fails on its mutant | **PASS**: exit 3 on both. Shim: `set0 create@0 record@0 wait@0 destroy@0 set1`, current 1 (62c8d9f: `create@1 …`). Mutant (a): 8 failed | `test_dlpack.py:1496-1549`, `g4/mut_a_*` |
| G1 | V4-02 docstring updated | **PASS** | `__init__.py:107-116` |
| G2 | sweep table complete; every site OK or FIXED; every FIXED site has a test or a stated reason | **PASS**: 148/148 product sites annotated (0 unannotated); OK 90, FIXED 58: 54 with a test, 4 with a stated reason (`ccnone`); 45 test-code sites listed | `sweep/sweep.json`, `sweep/sweep_table.md` |
| G3 | every round-1..4 probe passes or is INCONCLUSIVE only for lack of 2 GPUs | **PASS** (the three two-GPU probes give exit 3) | `probe_runs/` |
| G3 | `addon_lifetime` 3/3 runs, 0 incorrect | **PASS** | `probe_runs/round1_addon_lifetime_run{1,2,3}.log` |
| G3 | `core_exit_overrun` during and bounded rc 0 | **PASS** (round 3 and round 4 copies) | `probe_runs/round3_*`, `rev4_core_exit_overrun_*_after.log` |
| G3 | Voice T1-T6 PASS 236/236 | **PASS** | `logs/p1t_fix4_g3_voice_T1toT6.log`, `artifacts/p1t_fix4/p1t_T1toT6.json` |
| G3 | combined session = the 52 baseline ids | **PASS**: 52 = 52, 0 new, 0 no longer failing | `logs/p1t_fix4_g3_combined_session.log`, `logs/p1t_fix4_g3_compare.txt` |
| G4 | mutant (a), the V4-01 guard removed, fails its new test | **PASS** (killed, 8 tests) | `g4/mut_a_*` |
| G4 | mutant (b), the V4-03 check removed, fails its new test | **PASS** (killed; the probe also fails) | `g4/mut_b_*` |
| G4 | mutant (c), the V4-04 capture removed, fails its new test | **PASS** (killed, 3 tests; the probe also fails) | `g4/mut_c_*` |
| G5 | diff only in the Exercises paths plus `docs/P1T_FIX4_*.md`, `artifacts/p1t_fix4/**`, `logs/**`; probe files byte-identical; no new core `__global__` | **PASS** | `artifacts/p1t_fix4/g5_scope.txt` |

**Voice's verdict line** (`artifacts/p1t_fix4/p1t_T1toT6.json` summary):
`verdict PASS exit_code 0 counts {"PASS": 236, "FAIL": 0, "SKIP": 0} (T1 11, T2 144, T3 28, T4 12, T5 28, T6 13) f1 all_f1_ok true max_N 2.9320924600942896e-06 min_margin_vs_gate 6.821067299957125 (T4/shared__LSTM/max_1) seconds 76.9`.
These figures are identical to FIX3's, except the wall time (D14).

**Compare output** (`logs/p1t_fix4_g3_compare.txt`):
```
G1 compare (APA-MF2.2): baseline = /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2/logs/apa_mf1_g1_baseline_unchanged.log; now = logs/p1t_fix4_g3_combined_session.log
summary: 37 failed, 2729 passed, 47 skipped, 2 warnings, 15 errors in 445.45s (0:07:25)
baseline ids 52 now ids 52
baseline ids that no longer fail: 0 []
new failing ids: 0 []
new MF2.2 test files with any FAILED/ERROR: 0
G1 PASS
```
Passed is 2729, against FIX3's 2689: core added 22 tests and tc_signal added 18. The 37 failures and 15 errors are the
baseline APA files.

## G4 mutant table

The trees are `/tmp/p1t_fix4_mut_<m>/`, each a copy of the final worktree with one asserted edit
(`artifacts/p1t_fix4/g4/make_mutants.py`; diffs in `g4/mut_<m>.diff`). The runner is `g4/run_mutant.sh`. Each tree's
test file and shim are cmp-identical to the worktree's.

| Mutant | Edit | .so sha256 | New tests on the mutant | Probe on the mutant | Verdict |
|---|---|---|---|---|---|
| (a) V4-01 guard removed | `dlpack_fence_consumer`: `const bool switched = false;`, so the fence runs on the CURRENT device, as in 62c8d9f; the label proof stays | core `bfc62122aea1fc0e2a3726e3affef0ec0d8e833c5fd077d84195102f028f6c3a` | `-k fix4`: **8 failed**, 14 passed: all 8 fence tests (switch, per-thread stream, 5 failure modes, failed restore). Log `create@1 record@1 wait@1 destroy@1` | `core_export_device.py` exit 3 (one GPU) | KILLED |
| (b) V4-03 check removed | import: `int cur = 0; cudaGetDevice(&cur);` (result ignored, as in 62c8d9f) | core `42c363aa9395bcea8cd4423823093b13c855556952128c1df9e1d90dbe59c78f` | **1 failed** (`test_fix4_v403_import_refuses_*`: `used_dltensor`, raised None), 21 passed | `core_import_query_error.py` **rc 1** `core import succeeded after failed device query; name=b'used_dltensor'` | KILLED |
| (c) V4-04 capture removed | `synchronize_and_release_all`: `const cudaError_t prior = cudaSuccess;` | tc_signal `fcabca867c6bd0cf575749309a338c997e7b333225e2423e84a4068ca4527653` | **3 failed** (the V4-04 tests: NULL kernel, launch configuration, entries pending), 15 passed | `sync_existing_error.py` **rc 1** `synchronize returned success with a preexisting launch error; slot=98` | KILLED |

**Before-witness** (the new tests on the 62c8d9f binaries; `artifacts/p1t_fix4/before_witness/`):
- Core: 20 failed, 2 passed. The 2 passes are the same-device export and the sticky-import census, which pin behavior
  that was already right.
- tc_signal: 14 failed, 4 passed. The 4 passes are the device-placement pins (rint cold and warm, conv1d, LSTM):
  tc_signal's launch paths already switched to the operand device.
- Notable witnesses:
  - fence `create@1 …`;
  - `fill_` `launch@1`;
  - import `used_dltensor` with no device;
  - `synchronize()` returned success with 98 or 9 pending;
  - the slot left at 101 after a failed switch or a failed memset;
  - a failed restore or destroy left in the slot, then reported later by core as `CUDA error at to_host: invalid
    device ordinal`;
  - `empty`'s failed allocation left 2 in the slot.

## Operand census → tests and probes

| Census item (order) | Covered by |
|---|---|
| V4-01: export with current == tensor device | `test_fix4_v401_export_fence_same_device_does_not_switch` (no switch; create/record/wait/destroy at 0); the existing same-device stream tests (`test_export_stream_ordering_torch_side_stream`) |
| V4-01: export with current != tensor device (simulated, one GPU) | the shim tests above, for a stream handle and for 2; failure of each step; failed restore; mutant (a). The real two-GPU probe is INCONCLUSIVE here |
| V4-03: `CUDA_VISIBLE_DEVICES=""` | `test_fix4_v403_import_refuses_*` (both capsule kinds, and `empty`); `core_import_query_error.py`; the regenerated `core_interop_cpu` import |
| V4-03: sticky prior error | `test_fix4_v403_import_with_a_sticky_context_error_*` (a torch device-side assert in a subprocess; `cudaStreamQuery` == 710 proves the context is broken for core's runtime) |
| V4-04: a synchronous launch-configuration error, nothing pending | `test_fix4_v404_…[sync_launch_config]` (a real `cudaErrorInvalidConfiguration`); `[sync_null_kernel]` and `sync_existing_error.py` (98) |
| V4-04: the same with entries pending | `test_fix4_v404_synchronize_still_drains_pending_entries_then_raises` |
| Sweep: every call site | the sweep table; the 40 new tests and their test keys |

## Prior art

- **CUDA 12.6 runtime.** Sources: the local header docs (`/usr/local/cuda-12.6/include/cuda_runtime_api.h`) and the
  Programming Guide (receipt `artifacts/p1t_fix4/cuda_doc_quotes.txt`, fetched through the AtlasForge proxy). These were
  verified in this seat:
  - `cudaSetDevice` / `cudaGetDevice` and the current-device model: "Any streams or events created from this host
    thread will be associated with device";
  - `cudaEventRecord`: "event and stream must be on the same CUDA context". PG 3.2.9.3: "cudaEventRecord() will fail if
    the input event and input stream are associated to different devices";
  - `cudaStreamWaitEvent`: "event may be from a different device than stream". PG 3.2.9.3: "cudaStreamWaitEvent() will
    succeed even if the input stream and input event are associated to different devices";
  - PG 3.2.9.3: `cudaEventQuery` / `cudaEventSynchronize` succeed across devices. This is why the drain's cross-device
    queries need no switch;
  - PG 3.2.12: one error variable per host thread, "overwritten by the error code every time an error occurs";
    `cudaGetLastError` resets it and `cudaPeekAtLastError` does not; clear it "just before the kernel launch"; NotReady
    is never put in it.
  - Taken: these contracts. Ours: the empirical sticky / no-device / overwrite receipts (`cuda_semantics.txt`).
- **PyTorch 2.11, read in the installed package.**
  - `c10::cuda::CUDAGuard` / `InlineDeviceGuard`: the destructor restores through `uncheckedSetDevice` →
    `C10_CUDA_CHECK_WARN`, which consumes the slot with `cudaGetLastError` and warns, and never throws.
  - `c10/cuda/CUDAEvent.h`: an event is destroyed on its creating device ("to avoid creating a CUDA context on other
    devices").
  - `torch/_tensor.py` `Tensor.__dlpack__`: refuses a current-device mismatch with BufferError. It does not guard.
  - `torch/utils/dlpack.py`: the consumer passes the current stream of the tensor's device.
  - Taken: the scoped guard, consume-and-continue in destructors, and destroy-on-own-device.
  - Not taken: torch's refusal. The order chose the guard.
- **The order's own prior-art list**, as checked:
  - "PyTorch CUDAGuard / OptionalCUDAGuard around event record in its DLPack path" is not what torch 2.11's Python
    export does: it refuses (D1). The guard prior art is c10's guard in general.
  - CuPy's `cupy.cuda.Device` context manager: not re-read here. Unverified, lead to check: "cupy.cuda.Device __exit__
    setDevice restore".
- **The DLPack and array API stream handoff** (dmlc 2017+, Consortium 2021+): the producer orders the consumer's stream.
- **Testing.**
  - LD_PRELOAD symbol interposition (glibc ld.so), as in FIX1's F05 shim. Ours: the per-call device log, the
    genuine-slot failure injection, the cuBLAS hooks.
  - DeMillo, Lipton & Sayward (1978) mutation testing.
- **Ours (no prior art known to me):**
  - proving the engine label against the allocation (`cudaPointerGetAttributes`) at the DLPack and `fill_` boundary;
  - placing the whole fence, consumer wait included, inside the guard, with restore before raise;
  - the pre-launch take at the one constructor that every tc_signal launch path shares;
  - routing failures that cannot raise to the next `synchronize()`;
  - holding V4-04's taken error until its message exists.
- These annotations are also at the code sites: `bindings.cpp:382` block; `kernels.cu` drain / guard / label proof /
  `fill_` blocks; `cuda_guard.h` header; `dlpack_operand.cpp:903` block and the RetainUntilComplete constructor; the
  test headers. They are also in the ledger.

## Deviations (with reasons)

- **D1. Order premise.** The order lists "PyTorch CUDAGuard / OptionalCUDAGuard around event record in its DLPack path"
  as prior art. Torch 2.11's `Tensor.__dlpack__` refuses the mismatch instead (read in `torch/_tensor.py`). I
  implemented the order's guard, as written, and recorded the discrepancy.
- **D2. Beyond the literal V4-01 fix: the label proof.** Export and `fill_` check the tensor's label against its
  allocation (`cudaPointerGetAttributes`, nonempty tensors). They refuse a mismatch with BufferError (export) or
  RuntimeError (`fill_`).
  - Reason: without it the guard trusts the engine's `cuda:0` label, and it would fence the wrong device for an engine
    tensor allocated while another device was current.
  - Behavior change: only on multi-GPU hosts, or for memory the driver does not know.
- **D3. `empty` semantics.**
  - A CUDA tensor is now labeled with the current device (a checked `cudaGetDevice`), and an explicit `cuda:N` must
    name it (ValueError otherwise).
  - With no device, `empty(device="cuda")` now raises "empty: the current-device query cudaGetDevice failed", where it
    used to say "cudaMalloc failed". A zero-size CUDA `empty` with no device now also raises.
  - On this one-GPU host, labels are unchanged.
- **D4. Pre-launch error take** in tc_signal ops (the RetainUntilComplete constructor, after `acquire_event`) and in
  `fill_`.
  - A call with an error already pending now refuses before launching, naming the error as pre-existing. Before, it
    launched and then raised the error as its own.
  - Placing the take after `acquire_event` keeps the no-device refusals unchanged.
- **D5. V4-04 message text.**
  - `synchronize()` errors now include the pre-existing clause.
  - With no device, the runtime keeps `cudaErrorNoDevice` permanently, so every no-device synchronize also names it as
    pre-existing. This is the only text change in the reviewer's nogpu `host_checks` (rc identical, 40/40).
- **D6. Restore failures now raise on success paths.** This covers tc_signal `DeviceGuard::restore()`, core
  `CheckedDeviceGuard::restore()` and the fence. Where nothing can raise, they are counted (new stats keys:
  `device_restore_failures`, `event_destroy_failures`, `unreported_unraisable_errors`).
- **D7. cuBLAS.** A new handle whose setting fails is destroyed (a new `cublasDestroy` call), and the destroy's status
  is named.
- **D8. Test-code edit outside the new files.** `tc_signal/tests/fix2_native_harness.cpp` (FIX2's harness, in
  Exercises) now checks its pre-drain sync. `test_fix2_findings.py` passes 19/19.
- **D9. Probe handling** (originals byte-identical; every copy or regeneration asserted and diffed):
  - path-only copies of `host_checks.py`, `operand_census.py` and `real_metadata_plans.py`: they hard-code the
    reviewer's absent worktree `wt_p1trev4` and write under it;
  - a path-only copy of `release_failures.py` on the before arm (FIX3 D2);
  - `sync_slot_cpu.cpp`: the reviewer recorded no compile command (`logs/p1t_rev4_sync_slot_build.log` is empty). I built
    it with `build_cpu.sh`'s `cpu_native` flags against each arm's source;
  - `core_interop_cpu.cpp` embeds 62c8d9f code, so as written it is arm-independent and reproduces the 62c8d9f traces on
    both arms. Its import half was regenerated for the after arm (`regen_core_interop_import.py`; self-check: True);
  - the export SPY half was not regenerated: the after export path calls a `kernels.cu` helper (the label proof) that
    the probe's kernels slice lacks. V4-01 is decided on the real binary by the shim tests;
  - `lifecycle.cpp` needed no regeneration: its slice is verbatim in both arms.
- **D10. Conditions.** CPU probes ran in two conditions (nogpu and gpu), and every probe run, CPU included, went under the
  GPU lock. The battery is split into cpu, gpu and rf parts, so each part completes inside one foreground call.
- **D11. Network.**
  - The core build (`./build.sh 89`, as ordered) fetched pybind11 v2.12.0 from GitHub through FetchContent (as FIX3
    D17).
  - Mutant core builds used production's `tensor_cuda/build/_deps/pybind11-src` read-only.
  - The CUDA Programming Guide was fetched through the AtlasForge proxy (read-only).
- **D12. Own errors, fixed before acceptance.**
  - The shim's function-pointer macro did not compile.
  - Allocations had to be made before the simulated switch.
  - A parametrize decorator was misplaced.
  - The conv prealloc broke after the allocator became pop-based.
  - One own code issue: `fill_` bumped the revision after a raising restore. Fixed by bumping first; this was core r2.
  - All are in the ledger.
- **D13. Re-runs.** After core r2 and the test additions, the after-arm battery, rounds 1-3, Voice and the combined
  session were re-run on the final binaries. The mutants were re-made from the final sources. The combined session ran
  three times; the final run is 03:58-04:05. The receipts above are the final runs.
- **D14. Timing.** Other users' processes kept the host's load average at 26-33 throughout. Voice wall time was 76.9 s
  (FIX3: 21.3 s) and the combined session took 445 s (FIX3: 261 s). These are not performance measurements, and no
  performance claim is made.
- **D15. `__global__` in test code.** `tc_signal/tests/fix4_stage_kernel.cu` has one test-only `__global__`, built by
  the test with nvcc and never executed. It is not core and not product. The core count is 0.
- **D16. Background task.** My first sweep-attribution helper was quadratic and was auto-backgrounded after 120 s. I
  stopped my own task and rewrote it. No gate was involved.
- **D17. Before-witness quality.** Some 62c8d9f failures are message-level (for example `cudaSetDevice` without the
  device number). The defect witnesses are the ones listed under the mutant table.

## Honest residuals (not claimed fixed, or not tested)

- **Real two-GPU execution was not run** (one GPU). V4-01 is shown by call placement on the real runtime (shim, device 0
  real) and by the guard's logic. The reviewer's three two-GPU probes remain INCONCLUSIVE.
- **`__dlpack_device__` reports the label without proof.** A mislabeled engine tensor reports `(2, 0)`, and `__dlpack__`
  then refuses it.
- **Pre-existing core engine (out of scope).**
  - `Device::from_string` maps any `cuda:N` to index 0.
  - Engine ops other than `fill_` launch on the current device and label their outputs as before.
  - The `Storage` constructor raises on a failed allocation without consuming the slot entry (`empty` consumes it now;
    other engine callers do not).
  - Minimal fix, stated but not implemented: label allocations with the current device in the Storage / NDArray
    constructor, and consume in its raise.
- **Import into a sticky (broken) context proceeds.** It takes ownership once, and the release leaks (by design; see the
  V4-03 sticky-error census).
- **tc_signal trusts each operand's DLPack `device_id`** (the DLPack contract; all operands must agree). It does not
  prove pointers, because the reviewer's CPU probes use fake addresses.
- **A failed restore is reported, not repaired.** The calling thread stays on the tensor's device.
- **The 4 FIXED sites marked `ccnone` have no test.** The reason is in the legend.
- **Performance was not measured.** Free-threaded CPython is unsupported (unchanged).

## Diff stat

Code changed by this order (`git diff --stat 62c8d9f -- tensor_cuda tc_signal`, working tree):
```
 tc_signal/README.md                     |  22 +-
 tc_signal/src/conv.cu                   |  47 ++-
 tc_signal/src/dlpack_operand.cpp        | 176 ++++++++---
 tc_signal/src/dlpack_operand.h          |   8 +-
 tc_signal/src/lstm.cu                   |  46 ++-
 tc_signal/src/norms.cu                  |  25 +-
 tc_signal/src/small_ops.cu              |  27 +-
 tc_signal/tests/fix2_native_harness.cpp |   8 +-
 tensor_cuda/src/bindings.cpp            | 193 ++++++++++--
 tensor_cuda/src/kernels.cu              | 131 ++++++++-
 tensor_cuda/tensor_cuda/__init__.py     |  23 +-
 tensor_cuda/tests/test_dlpack.py        | 504 ++++++++++++++++++++++++++++++++
 12 files changed, 1050 insertions(+), 160 deletions(-)
```
- New untracked files:
  - `tc_signal/src/cuda_guard.h` (134 lines);
  - `tc_signal/tests/test_fix4_findings.py` (478);
  - `tc_signal/tests/fix4_cudart_shim.c` (124);
  - `tc_signal/tests/fix4_stage_kernel.cu` (15);
  - `docs/P1T_FIX4_LEDGER.md`, this report, `artifacts/p1t_fix4/**` and `logs/p1t_fix4_*`.
- The literal `git diff --stat 62c8d9f` also lists the lead's round-4 receipt and order commits already on voice-fix4: 59
  files under `artifacts/p1t_rev4/`, `logs/p1t_rev4*`, `docs/P1T_REV4_*` and `orders/`.
- `.gitignore` ignores `*.log` and `*.json`, so those receipts need `git add -f`.

## Changed files

- Core:
  - `tensor_cuda/src/bindings.cpp`: the fence, the label proof call, import, empty, stats, docstrings;
  - `tensor_cuda/src/kernels.cu`: drain restore, guard, label proof, `fill_`;
  - `tensor_cuda/tensor_cuda/__init__.py`: docstrings;
  - `tensor_cuda/tests/test_dlpack.py`: the FIX4 block.
- tc_signal:
  - `src/cuda_guard.h` (new), `src/conv.cu`, `src/lstm.cu`, `src/norms.cu`, `src/small_ops.cu`,
    `src/dlpack_operand.cpp`, `src/dlpack_operand.h`;
  - `README.md`;
  - tests: `test_fix4_findings.py`, `fix4_cudart_shim.c`, `fix4_stage_kernel.cu` (new) and `fix2_native_harness.cpp`.
- Docs: `docs/P1T_FIX4_REPORT.md`, `docs/P1T_FIX4_LEDGER.md`.
- Receipts: `artifacts/p1t_fix4/**`, `logs/p1t_fix4_*`.
- Built and gitignored: both `.so` files and their `build/` directories.

**Scratch left (nothing deleted):**
- `/tmp/p1t_fix4_probes/{before,after,rev2_after}`: probe builds and path copies;
- `/tmp/p1t_fix4_mut_{a,b,c}`: mutant trees and builds;
- `/tmp/p1t_fix4_scratch`: semantics scripts, shim builds, report parts;
- pytest temporary directories under `/tmp/pytest-of-vader/`.
