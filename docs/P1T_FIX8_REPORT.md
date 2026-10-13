# P1T-FIX8 report: round-8 findings V8-01 and V8-02 closed; sweeps (k) consumed CUDA errors and (l) the last reader at exit

**Verdict:**

- **V8-01 and V8-02 are fixed, each with tests.** Sweeps (k) and (l) are complete: every FIXED site is fixed, with a
  test or a stated reason.
- **G2, G3, G4 and G5 PASS.** G3 covers the probes of rounds 1-8, `gpu_regressions` fix7 / fix56 / semantics with zero
  skips, Voice PASS 249/249, exactly the 52 baseline ids in the combined session, and real-torch interop. G4: the
  mutants (e), (f) and (g) are all KILLED.
- **G1 passes for three of its four probes. It is RED as written for `diagnostic_events_gpu.py`**: rc 1 on 6991939
  and rc 1 here (D2, D12).
  - On this build every row of that probe passes: rc 1 -> rc 0, and each marker failure is reported, naming its call.
  - Its parent check, "a failed create / record still gets a wait diagnostic", needs an exit wait. None exists on this
    host: under torch 2.11.0, `torch.cuda.stream(torch.cuda.ExternalStream(0))` makes a non-blocking stream current,
    so the legacy-stream sync at exit has nothing to wait for.
  - A copy that changes only that line, putting the work on torch's default (legacy) stream, gives rc 1 on 6991939 and
    **rc 0** here.
  - The lead decides whether that copy closes the row. I did not change the product to print a wait that did not
    happen.
- **Amended by "Lead send-back L2"** (the last section):
  - The lead ACCEPTED D2 / D12: the legacy-stream copy is the G1 evidence for `diagnostic_events_gpu.py`.
  - L2 fixed `DeviceWaitWatchdog::stop()` and re-ran the gates on the new tc_signal build.
  - Binaries, line numbers and gate lines above this section are those of the pre-L2 build.

Seat: Opus 5.5 (high), the resumed FIX7 seat. Order: `orders/P1T_FIX8_ERROR_VISIBILITY_AT_EXIT.md` (f9e6bdd). Ledger:
`docs/P1T_FIX8_LEDGER.md`. Receipts: `artifacts/p1t_fix8/` and `logs/p1t_fix8_*`. "Before" = the 6991939 build in
`/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_p1trev8`, used read-only as the
before arm itself (its root holds the round-8 probes; binaries core `750112413b78...`, tc_signal `ca5770c152f6...`);
probes that write receipts run there as path-only copies writing under `/tmp/p1t_fix8_*` and
`artifacts/p1t_fix8/probe_runs/` (D1).

## Per-finding table

| Finding | Fix (file:line) | Proving tests (new) | Probe before (6991939) | Probe after (this build) |
|---|---|---|---|---|
| **V8-01** (MINOR): the L1 exit device-wait marker erased its own CUDA failures (create / record / query / destroy consumed by `cudaGetLastError`, no record); a failed create / record fell through to an undiagnosed stream wait; "finished" could be printed with no proof | `tc_signal/src/dlpack_operand.cpp`: `:1167` `note_exit_marker_failure` (counted in `exit_marker_failures`, recorded through `note_unraisable_failure` as `{call, Purpose::kExitWaitMarker, device, err}` BEFORE the slot entry is consumed; raised once by the running `synchronize()`), `:1173` `destroy_marker`, `:1185` `wait_device_work_diagnosed` (returns whether completion was PROVED; `:1213` NotReady is not a failure), `:1229` `DeviceWaitWatchdog` (a failed create / record / query hands the stream sync to a watchdog thread on the same schedule, `exit_device_wait_watchdogs`), `:1297` `sync_device_legacy` ("finished" only after a successful query or a successful stream sync); `FailedCall::kEventCreate`, `Purpose::kExitWaitMarker`, `describe()` and `purpose_name()` texts; test hooks `cuda:exit_marker_{create,record,query,destroy}` | `tc_signal/tests/test_fix8_findings.py`: trace module (each call x {plain, message OOM}: reported once, naming call, purpose, device; watchdog lines and "finished (cudaStreamSynchronize returned)" for create / record / query, "finished (its marker event completed)" for destroy; query + sync failure: no "finished"; healthy control); real module on the GPU (each call x {plain, OOM}, healthy control) | `diagnostic_errors_cpu.py` **rc 1** (4 / 4 erased: `error: null`); `diagnostic_events_gpu.py` **rc 1** (4 / 4 rows rc 1, "one-shot marker failure erased") | `diagnostic_errors_cpu.py` **rc 0** (each of the 4 marker failures raised once by `synchronize()`, naming its call; slot 0); `diagnostic_events_gpu.py` as written **rc 1** (RED, D2: all 4 rows rc 0 and each names its call; the parent's "waiting" check fails because the work is not on the legacy stream under torch 2.11); legacy-stream copy **rc 0** (before: rc 1) |
| **V8-02** (MINOR): both native exit loggers swallowed `bad_alloc` while building the final report and wrote nothing, keeping records no later reader exists to read | Add-on: `dlpack_operand.cpp:1998` `summarize_unreported` / `:2026` `write_summary_line` (allocation-free: count and first record, static strings, `cudaGetErrorName`, C stdio), `:2073` `exit_log_unreported` (full report, else the summary, then the records consumed), `:2052` `note_raised_at_exit` / `:2098` `exit_report_written` / `:2104` `exit_log_unwritten_report` (a report `synchronize()` RAISED inside the runner is kept as a plain-data summary until the hook acknowledges writing it); `tc_signal/src/bindings.cpp:458` runner: the last reader after the hook in EVERY case (`:453` `exit_runner_last_reader`), no `e.what()` (it allocates inside a noexcept function: `:428` `ascii_message_of`), new raw `:508` `_exit_report_written`; `tc_signal/__init__.py` hook acknowledges. Core: `tensor_cuda/src/kernels.cu:426` `dlpack_unreported_unraisable_raw` (allocation-free snapshot, the site text is a static string); `tensor_cuda/src/bindings.cpp:1674` `write_core_summary`, `:1692` `core_exit_last_reader` (full message, else the summary; then acknowledged; plus an R7 report the hook raised but did not write), `:1732` runner (last reader in every case), `:1625` `dlpack_raise_unraisable` keeps the exit report, raw `_dlpack_exit_report_written` and test hook `_dlpack_test_fail_exit_report`; `tensor_cuda/__init__.py` hook acknowledges | `test_fix8_findings.py`: exit logger control (full report), OOM (summary naming count and first record), stderr closed (no crash, consumed), a raised report the hook could not write (runner writes it), one it wrote (not repeated); `tensor_cuda/tests/test_dlpack.py` `test_fix8_v802_core_*`: the same four for the core runner (F05 shim records the R7 failure) | `verify_exit_log_cpu.py` **rc 1** (OOM arm: `record_still_pending= True`, no CUDA stderr; control PASS); `verify_core_exit_log_cpu.py` **rc 1** (OOM arm: no CUDA stderr; control PASS) | `verify_exit_log_cpu.py` **rc 0** (OOM arm: the allocation-free summary line, naming the count and the first record; `record_still_pending= False`; control unchanged); `verify_core_exit_log_cpu.py` **rc 0** (OOM arm: "tensor_cuda: at interpreter exit, 1 CUDA failure(s) ... (allocation-free summary), the first: host probe error [CUDA device 0: cudaErrorInvalidValue]"; control unchanged) |

## G2: sweep (k), consumed CUDA errors

Method: every site in the P1-T code (`git diff 9df5bef..HEAD -- tensor_cuda tc_signal`: all of `tc_signal/src/*`, and
the P1-T lines of `tensor_cuda/src/kernels.cu` and `bindings.cpp`) that calls `cudaGetLastError`, branches on
`cudaPeekAtLastError`, ignores or overwrites a CUDA return code, or raises after consuming one. Found by grep of the
current sources (comment lines excluded) plus reading each enclosing function; no CUDA call's return value is ignored
(the only statement-form CUDA calls are `cudaGetLastError` consumptions, every one listed). Rule (order): the failure is
recorded (tagged: call, purpose, device, error) BEFORE it is consumed and reaches `synchronize()` or the exit logger,
or it is reported by its own raise whose report exists before the consumption, or it is not a failure (cited).
Line numbers are for this build.

| # | Site (file:line) | What is consumed / dropped | How it is reported now | Verdict | Test / reason |
|---|---|---|---|---|---|
| k1 | `dlpack_operand.cpp:1167-1218` exit marker: create (`:1187`), record (`:1198`), query non-NotReady (`:1208`), destroy (`:1173`) | each call's own failure (V8-01) | `note_exit_marker_failure` records `{call, kExitWaitMarker, device, err}` first, then `cudaGetLastError`; raised once by the running `synchronize()` | **FIXED** (V8-01) | `test_fix8_v801_*` (trace + real), probes `diagnostic_errors_cpu.py`, `diagnostic_events_gpu.py` |
| k2 | `dlpack_operand.cpp:1213` marker query `cudaPeekAtLastError() == cudaErrorNotReady` | `cudaErrorNotReady` | not a failure: the documented "still running" answer; consumed only when it is what the slot holds | NOT A FAILURE | CUDA 12.6 Runtime API, `cudaEventQuery`: "Returns cudaSuccess if all captured work has been completed, or cudaErrorNotReady if any captured work is incomplete" (fetched: `artifacts/p1t_fix8/citations/cuda126_runtime_api_event.txt`); CUDA 12.6 Programming Guide 3.2.12: "cudaErrorNotReady that may be returned by cudaStreamQuery() and cudaEventQuery() is not considered an error and is therefore not reported by cudaPeekAtLastError() or cudaGetLastError()" (`artifacts/p1t_fix4/cuda_doc_quotes.txt:33`) |
| k3 | `dlpack_operand.cpp:1531` sync drain phase 2, NotReady branch | `cudaErrorNotReady` | as k2 | NOT A FAILURE | as k2 |
| k4 | `dlpack_operand.cpp:1858` next-call drain, NotReady branch | `cudaErrorNotReady` | as k2 | NOT A FAILURE | as k2 |
| k5 | `dlpack_operand.cpp:1537` sync drain phase 2, a failed query | `cudaEventQuery`'s error | `rep.add({kEventQuery, kQueryEvent, dev, r})` first, then consumed; `drain_unreported` persists it before any allocation (V6-03) | OK (FIX7) | FIX7 V7-02 tests |
| k6 | `dlpack_operand.cpp:1861` next-call drain, a failed query | `cudaEventQuery`'s error | `note_unraisable_failure({kEventQuery, kDrainQueryEvent, ...})` first (FIX5 V5-02), then consumed | OK (FIX5) | FIX5 V5-02 tests |
| k7 | `dlpack_operand.cpp:1509` sync drain, the current-device query | `cudaGetDevice`'s error | `rep.query_failed` / `query_error` assigned first, then consumed (FIX8: the order made literal; before, consumed then assigned in the same noexcept block, nothing could intervene) | OK | FIX6 V6-03 tests (`cuda:current_device`) |
| k8 | `dlpack_operand.cpp:2123` `synchronize()` entry, the pre-existing slot error | the thread's pending error | read with `cudaPeekAtLastError`, recorded (`prior_error` / `prior_unreported`), then consumed (FIX8; FIX4-FIX7 read it by consuming it and recorded it on the next line) | OK | FIX4 V4-04 / FIX7 tests (prior error raised once) |
| k9 | `dlpack_operand.cpp:1318` `sync_device_legacy`, a failed stream sync | `cudaStreamSynchronize`'s error | consumed, and the value RETURNED; its only caller (`DeviceSyncSet::sync`) stores it unconditionally (`res[]`, or the spill record), and `sync_and_release_all_impl` adds it to the report; nothing can drop it in between (noexcept, a value) | OK | FIX7 V7-02 tests (`cuda:sync`) |
| k10 | `cuda_guard.h:176` `QuietDeviceGuard` constructor, a failed `cudaGetDevice` / `cudaSetDevice` | the guard's own call | consumed; the value kept in `err_` (`failed_at_switch()` says which call); every user records or raises it: `sync_device_legacy` (`:1298`, returned as k9), `destroy_event_on_device` (`:1384`, recorded), `~RetainUntilComplete` (`:1778`, recorded `kRecordEvent`), `acquire_event` (`:1564`, raised, or recorded under OOM: k15) | OK | FIX7 sweep (j) guard tests |
| k11 | `cuda_guard.h:194` `~QuietDeviceGuard`, a failed restore | `cudaSetDevice(prev)` | `note_unraisable_device_restore_failure` first, then consumed (FIX8 order; FIX4-FIX7 consumed first, same noexcept block) | OK | FIX4 restore tests |
| k12 | `cuda_guard.h:152` `~DeviceGuard`, a failed restore | `cudaSetDevice(prev)` | as k11 | OK | FIX4 restore tests |
| k13 | `cuda_guard.h:65` `raise_cuda_failure` (DeviceGuard constructor / `restore()`, `conv.cu` / `lstm.cu` callers) | the failed call's slot entry | the report (message and `std::runtime_error`) is built first; consumed only then (`throw_after_consuming`, `:57`); a `bad_alloc` while building leaves the entry in the slot for the next check | **FIXED** (FIX7 consumed, then built) | `test_fix8_sweep_k_raise_helpers_consume_only_after_their_report_exists[cuda]` |
| k14 | `cuda_guard.h:89` `raise_call_failure_on` / `check_call_on` (`lstm.cu:225-233`, ...: `cudaMemcpyAsync`, `cudaMemsetAsync`) | the failed call's slot entry | as k13 | **FIXED** | `..._exist[call]` |
| k15 | `dlpack_operand.cpp:1566` `acquire_event`, a failed `cudaEventCreateWithFlags` (or its guard) | the failed call's slot entry (the guard's was consumed by the guard) | report built first, then consumed; if it cannot be built, the failure is recorded `{kEventCreate or the guard's call, kCreateEvent, device, err}` before the `bad_alloc` propagates, and the next `synchronize()` raises it | **FIXED** | `test_fix8_sweep_k_acquire_event_failure_survives_its_report_allocation_failure` |
| k16 | `cuda_guard.h:99` `check_launch_on` (launch checks: `conv.cu:96`, `:220`, `:280`; `lstm.cu:103`, `:243`, `:260`, `:269`; `norms.cu:156`; `small_ops.cu:97`, `:109`, `:117`) | the launch check's slot entry | read with `cudaPeekAtLastError`; consumed only once the report exists | **FIXED** | `..._exist[launch]` |
| k17 | `cuda_guard.h:109` `check_after_cublas_on` (`conv.cu:261`, `lstm.cu:251`, `:266`) | as k16 | as k16 | **FIXED** | `..._exist[cublas]` |
| k18 | `conv.cu:52` `check_cuda` (`cudaRuntimeGetVersion` / `cudaGetDevice` in `build_info`) | the failed call's slot entry | message built first (FIX4), and now the `std::runtime_error` too, then consumed (`throw_after_consuming`) | **FIXED** (FIX7 constructed the exception after consuming) | same helper as k13 (`..._exist[cuda]` exercises `throw_after_consuming`) |
| k19 | `cuda_guard.h:119` `raise_cublas_failure_on` / `check_cublas` (cuBLAS statuses) | none: a `cublasStatus_t`, not the CUDA runtime's error state (cuBLAS 12.6 imports no libcudart symbol: FIX7 sweep (j) receipt) | reported by its raise | OK (not an error-state site) | reasoning |
| k20 | `dlpack_operand.cpp:1751` `RetainUntilComplete` constructor, the pre-existing slot error | the thread's pending error | read with `cudaPeekAtLastError`, report built, then consumed (`throw_after_consuming`) | **FIXED** (FIX7 consumed, then built) | reasoning: same helper as k13 / k16 (tested there); the trace has no `RetainUntilComplete` constructor path under OOM |
| k21 | `dlpack_operand.cpp:1806` `~RetainUntilComplete`, a failed `cudaEventRecord` (or its guard) | the record's slot entry | `note_unraisable_failure({record_call, kRecordEvent, ...})` first (FIX5), then consumed | OK (FIX5) | FIX5 sweep (d) tests |
| k22 | `dlpack_operand.cpp:1383` `destroy_event_on_device` | `cudaEventDestroy`'s error | `note_unraisable_failure` first, then consumed (FIX4 / FIX7) | OK | FIX4 destroy tests |
| k23 | `kernels.cu:289` `drain_device_legacy_stream` (three steps) and its restore | each step's error | `note_core_unraisable` first, then consumed (FIX5) | OK | FIX5 / FIX7 drain tests |
| k24 | `kernels.cu:371` `~CheckedDeviceGuard` restore | `cudaSetDevice(prev)` | `note_core_unraisable` first, then consumed (FIX5) | OK | FIX4 fill guard tests |
| k25 | `kernels.cu:336` `raise_cuda_failure` (CheckedDeviceGuard constructor / `restore()`, the label proof `:476`) | the failed call's slot entry | report built first, then consumed (`throw_after_consuming`, `:331`) | **FIXED** | `test_fix8_sweep_k_core_raise_consumes_only_after_its_report_exists[*]` (same helper through `:421`) |
| k26 | `kernels.cu:652` `fill_`, the pre-existing slot error | the thread's pending error | peek, report, consume | **FIXED** | reasoning: same helper as k25 (tested); fill_'s pending path is reached only after its own allocations |
| k27 | `kernels.cu:669` `fill_` launch check | the launch check's slot entry | peek, report, consume | **FIXED** | as k26 |
| k28 | `bindings.cpp:453`, `:459` export fence: `cudaGetDevice` / `cudaSetDevice` | the failed call's slot entry | report built first, then consumed (`throw_cuda_report_after_consuming`, `kernels.cu:421`) | **FIXED** | as k25 |
| k29 | `bindings.cpp:489` export fence: create / record / wait / destroy / restore | every failed call's slot entry | as k28 (FIX7 consumed before building the message) | **FIXED** | as k25 |
| k30 | `bindings.cpp:1145` `from_dlpack` current-device query | `cudaGetDevice`'s error | as k28 | **FIXED** | `..._core_raise_consumes_only_after_its_report_exists[from_dlpack]` |
| k31 | `bindings.cpp:1264` `empty` current-device query | `cudaGetDevice`'s error | as k28 | **FIXED** | `..._core_raise_..._exists[empty]` |
| k32 | `bindings.cpp:1283` `empty` allocation catch | the engine allocation's slot entry | consumed, then the engine's already-built `std::runtime_error` is rethrown (nothing allocated after the consumption) | OK | FIX4 empty tests |
| k33 | `kernels.cu:255` `note_core_unraisable` (lock / bad_alloc catch) | none (bookkeeping) | the count is atomic and recorded even when the mutex fails (counted, unnamed) | OK | FIX5 |
| k34 | `cublasDestroy` of a handle that could not be cached (`note_cublas_handle_destroy_failure`) | a cuBLAS status, while another exception propagates | counted with its first status (`pending_stats()`), the propagating exception is the report (FIX5 V5-03, closed by rounds 5-6) | OK (not an error-state site) | FIX5 V5-03 tests |

Summary (k): 34 rows. **FIXED 16** (k1, k13-k18, k20, k25-k31), **NOT A FAILURE 3** (k2-k4, cited), **OK 15** (k5-k12,
k19, k21-k24, k32-k34). Every FIXED row has a test or a stated reason (k20, k26, k27: the same reordered helper,
tested at k13 / k16 / k25). One part of a raise is not ours: after the C++ report exists and the slot entry is
consumed, pybind11 turns the `std::runtime_error` into a Python `RuntimeError`; if CPython cannot allocate that
string, it raises `MemoryError` instead (N1 below).

## G2: sweep (l), the last reader at exit

Method: every path at interpreter exit that is the final reader of recorded failures, and every catch on a reporting
path. A last reader passes if, when its rich report cannot be built (`bad_alloc` / `MemoryError`, a failing `str()`,
a failing or closed stderr), an allocation-free fallback still writes, with C stdio, the record count and the first
record (call, purpose, device, `cudaGetErrorName`).

| # | Last reader (file:line) | Rich report | Fallback when it cannot be built | Verdict | Test |
|---|---|---|---|---|---|
| l1 | add-on `exit_log_unreported` (`dlpack_operand.cpp:2073`) | `report_message` (allocates) | `write_summary_line` (`:2026`): count, first record (call, purpose, device, error name), C stdio; records then consumed. FIX7: nothing written, records kept with no later reader | **FIXED** (V8-02) | `test_fix8_v802_exit_logger_*` (control, OOM, stderr closed); probe `verify_exit_log_cpu.py` |
| l2 | add-on native runner (`bindings.cpp:458`) | its synchronize() failure: the RuntimeError's own text (`ascii_message_of`, no allocation); then l1 and l3 | the raised report's summary (l3); FIX7 called `e.what()` (allocates inside a noexcept function: `std::terminate` under OOM) and ran l1 only when the hook raised | **FIXED** | `test_fix8_v802_*` (the runner is the last reader when the hook returned too: `..._hook_could_not_write_...`) |
| l3 | add-on raised-but-unwritten report (`dlpack_operand.cpp:2104`, kept by `note_raised_at_exit` `:2052`) | the hook's own line | `write_summary_line` of the kept plain-data summary | **FIXED** (new) | `test_fix8_v802_report_the_hook_could_not_write_is_written_by_the_runner`, `..._hook_wrote_is_not_repeated` |
| l4 | add-on hook (`tc_signal/__init__.py`) step-1 line and the post-close `synchronize()` retry line | f-strings of the RuntimeError / MemoryError, `sys.stderr.write` | if building or writing any line fails (MemoryError, a failing `str()`, stderr None / raising / closed), the hook does not acknowledge, and the runner writes l3's summary; a retry that raised MemoryError leaves its records, which l1 writes | **FIXED** | as l3; `test_fix7_v701_exit_hook_closes_when_stderr_is_gone_or_raises` (still passes) |
| l5 | core native runner (`tensor_cuda/src/bindings.cpp:1732`) | `unraisable_message` (allocates) | `core_exit_last_reader` (`:1692`): `dlpack_unreported_unraisable_raw` (no allocation) then `write_core_summary` (`:1674`): count, the static site text, device, error name; then acknowledged. FIX7: only when the hook raised, and a `bad_alloc` wrote nothing | **FIXED** (V8-02) | `test_fix8_v802_core_runner_*`; probe `verify_core_exit_log_cpu.py` |
| l6 | core hook's R7 logging (`tensor_cuda/__init__.py`, `_dlpack_raise_unraisable`) | f-string of the RuntimeError, `sys.stderr.write` | as l4: no acknowledgement -> the runner writes the kept summary (`dlpack_raise_unraisable` `:1625` keeps it while the runner runs); a MemoryError leaves the record, which l5 writes | **FIXED** | `test_fix8_v802_core_report_the_hook_could_not_write_...`, `..._hook_wrote_is_not_repeated` |
| l7 | both hooks interrupted at their first bytecode | none | the runners (l2, l5) after quiescence; the hook's exception restored unchanged | OK (FIX7) + l1 / l5 fallbacks | FIX7 interrupted-hook tests; `test_fix8_v802_*_oom` (KeyboardInterrupt still printed) |

Catches on reporting paths:

| # | Catch (file:line) | What it swallows | Verdict |
|---|---|---|---|
| c1 | `exit_log_unreported` `catch (const std::exception&)` | `bad_alloc` building the report | **FIXED**: falls through to the summary (l1) |
| c2 | add-on runner `catch (py::error_already_set&)` (its synchronize()) | the RuntimeError (records already consumed) | **FIXED**: its own text written allocation-free, else l3 |
| c3 | add-on runner `catch (const std::exception& e)` | `bad_alloc` from synchronize() (records kept) | OK: `e.what()` of `bad_alloc` is a static string; the records are written by l1 |
| c4 | add-on runner `catch (const std::exception&)` around the close | nothing (not reached: the close allocates and throws nothing) | OK |
| c5 | add-on hook `except Exception` around each stderr write | a failing stderr | **FIXED**: no acknowledgement -> l3 |
| c6 | core `core_exit_last_reader` `catch (const std::exception&)` | `bad_alloc` building the message | **FIXED**: the summary (l5) |
| c7 | core runner `catch (py::error_already_set&)` -> `discard_as_unraisable` | a Python error building the close's result dict (after the close) | OK: not a CUDA record; CPython's unraisable hook reports it |
| c8 | core runner `catch (const std::exception&)` building the close's result | `bad_alloc` after the close | OK: not a CUDA record (the close and wait happened) |
| c9 | core hook `except Exception` around `_dlpack_raise_unraisable` | MemoryError (record kept) | **FIXED**: the record is now read by l5 also when the hook returned |
| c10 | core hook `except Exception` around each stderr write | a failing stderr | **FIXED**: no acknowledgement -> the runner (l6) |
| c11 | `note_core_unraisable` / `dlpack_ack_unraisable` / `dlpack_unreported_unraisable_raw` lock catches | a mutex failure (never observed) | OK: the atomic count still records / consumes (unnamed) |
| c12 | `synchronize_and_release_all` / `dlpack_raise_unraisable` `catch (const std::bad_alloc&)` | `bad_alloc` capturing the RuntimeError | OK: rethrown as MemoryError, records kept (V6-03 / R7) |
| c13 | `call_deleter_contained` `catch (...)` | a producer deleter's C++ exception | OK: not a CUDA failure; counted (`deleter_exceptions`, FIX3) |

Summary (l): 7 last readers, **FIXED 6**, **OK 1** (l7, now backed by the l1 / l5 fallbacks); 13 catches, **FIXED 6**,
**OK 7**. Every last reader writes the count and the first record allocation-free when its rich report cannot be built.

## FIX7's rows corrected (i24, i25, j24, a3, a9)

| FIX7 row | FIX7 said | Corrected (FIX8) |
|---|---|---|
| i24 add-on: a CUDA failure is reported, never lost, at exit | FIXED (runner's synchronize() + `exit_log_unreported`; the post-close retry) | **PARTLY** in FIX7, through V8-02 (`exit_log_unreported` wrote nothing on `bad_alloc`; the runner's `e.what()` could `std::terminate` under OOM; a raised report the hook could not write was lost) and V8-01 (the marker's own failures were never recorded). **FIXED in FIX8**: (l) l1-l4, (k) k1. |
| i25 core: the R7 record is logged at exit | FIXED (logging attempted whatever the close did; the runner logs it when the hook was interrupted) | **PARTLY** in FIX7, through V8-02 (the runner wrote nothing on `bad_alloc`; it read the record only when the hook raised; an R7 report the hook could not write was lost). **FIXED in FIX8**: (l) l5, l6. |
| j24 add-on runner lines | OK (new) | **PARTLY** in FIX7, through V8-02 (as i24: no fallback; `e.what()` inside a noexcept function). **FIXED in FIX8**: l1-l3. |
| a3 (L1 addendum) add-on device syncs of exit | FIXED (a polled marker event, diagnosed) | **PARTLY** in FIX7 L1, through V8-01: the marker's own create / record / query / destroy failures were consumed and never reported; a failed create or record fell through to an undiagnosed stream wait; "finished" could be printed after a failed query. **FIXED in FIX8**: k1 (recorded, raised once), the watchdog fallback (the wait is diagnosed whatever the marker does), "finished" only after proof. |
| a9 (L1 addendum) core `_dlpack_raise_unraisable` and the R7 record mutex at exit | OK: "its holders only copy a record (no call, no Python, no wait while holding it)" | The conclusion stands, the rationale is corrected (reviewer): `dlpack_unreported_unraisable` / `dlpack_first_unraisable` copy the site text into a `std::string` while holding the mutex, so a holder can allocate (and throw `bad_alloc`, which unwinds the `lock_guard` and releases the mutex). The hold is still bounded (one string copy, no CUDA call, no Python, no wait), so the lock wait at exit cannot be unbounded. The exit path no longer allocates under it at all: the runner reads `dlpack_unreported_unraisable_raw` (pointer, device, error: no allocation). |

## Not a defect (with justification)

| # | Item | Why it is not a defect in our code | Citation / receipt |
|---|---|---|---|
| N1 | After a raise's C++ report exists and its error-slot entry is consumed, CPython may be unable to create the Python `RuntimeError`'s message string; the call then raises `MemoryError` and the CUDA failure's text is not shown | The failure is still raised (never a success) and our side has done all it can: the report (message and exception object) was built BEFORE the slot was consumed (k13-k31). Turning a C++ exception into a Python one is pybind11's translator (`PyErr_SetString(PyExc_RuntimeError, e.what())`), and when CPython cannot allocate that string its own error machinery leaves the `MemoryError` the allocator set: interpreter behaviour, not P1-T's. `synchronize()` and `_dlpack_raise_unraisable`, whose records are ours to keep, already build the Python exception themselves and keep their records when it fails (V6-03, R7) | CPython v3.12.3 `Python/errors.c` `_PyErr_SetString` (`artifacts/p1t_fix8/citations/cpython_3.12.3_errors_c_PyErr_SetString.txt`: the error is set only "if (value != NULL)"); pybind11 v2.12.0 `translate_exception` (`artifacts/p1t_fix8/citations/pybind11_2.12_translate_exception.txt`) |
| N2 | A user who calls a hook directly (not through the native runner), or unregisters the runner, has no last reader at exit | Deliberate removal of the documented exit protection (FIX6 N4, FIX7 N2, both accepted); the documented registration is `atexit.register(_C._exit_hook_runner, ...)` | README "Interpreter exit"; Python 3.12 `atexit` docs (`artifacts/p1t_fix7/citations/python312_atexit.txt`) |
| N3 | With stderr's file descriptor closed, nothing of a fallback line reaches anyone | There is no channel left: C stdio is the last one, and writing to a closed descriptor fails without side effects (the line is attempted, nothing crashes, the records are acknowledged so nothing is reported twice). The order's census requires exactly "no crash, and the record is still acknowledged" | POSIX `write(2)` (EBADF); `test_fix8_v802_*_stderr_closed_*` |

FIX7's N1-N4 stand as round 8 accepted them.

## User-visible changes (each justified)

- **U1. A failed call of the exit device wait's own marker event is raised by the running `synchronize()`**
  (`RuntimeError`: "the marker event of interpreter exit's diagnosed wait for the legacy stream of device D could not
  be created (cudaEventCreateWithFlags: ...)" / recorded / queried / destroyed), once; `pending_stats()`
  `exit_marker_failures`. Correct: V8-01; FIX2 N1 / FIX5 V5-02 (a CUDA failure on a P1-T path is reported, diagnostic
  code included).
- **U2 (updated by lead send-back L2). When the marker cannot be created, recorded or queried, the device wait at exit
  is diagnosed by a watchdog thread** (the same schedule; lines end "; diagnosed by a watchdog thread, because the
  wait's marker event failed"; `exit_device_wait_watchdogs`).
  - Correct: L1's rule (no silent exit wait), for the case FIX7 left silent. The thread touches no CUDA state and no
    Python object.
  - Stopping it waits until the thread has finished. `run()` sets a flag as its very last action. `stop()` repeats its
    lock attempt and its wakeup every `kExitDevicePollMs` (1 ms) until that flag is set, and only then joins.
  - A failed lock (its wakeup possibly missed) costs about one interval per failure, never a silent wait until the
    watchdog's next diagnostic. Measured: 2.2 ms for 1 failure and 4.3 ms for 3, against 4.6 s with FIX8's stop().
  - A failed join detaches only a thread that no longer touches anything.
  - Counters: `exit_device_wait_watchdog_lock_failures` and `exit_device_wait_watchdog_detaches`.
  - Stated limit: if writing a line blocks (stderr is a full pipe nobody reads), `stop()` waits for that write, as
    the join did before L2. The waiting thread's own next line would block on the same stderr.
- **U3. "finished after ..." is printed only after completion was proved**, and says how ("(its marker event
  completed)" or "(cudaStreamSynchronize returned)"). Correct: V8-01 (FIX7 could print it after a failed query).
- **U4. The native exit runners are the last reader in every case, not only when the hook raised**: after the hook
  and after the close and the wait, they write the recorded failures nobody raised (add-on and core). Correct: V8-02 /
  (l): a hook that returned after its report raised `MemoryError` left records nothing would read.
- **U5. When the full exit report cannot be built, an allocation-free summary is written instead** (count and first
  record: call, purpose, device, error name; `exit_log_fallbacks` / `shutdown_log_fallbacks`), and the records are
  consumed. Correct: V8-02 (FIX7 wrote nothing and kept records no later reader exists to read).
- **U6. A report `synchronize()` (or the core's R7 logging) raised at exit that the hook could not write is written by
  the runner** ("... raised a CUDA failure report that the exit hook could not write: ..."). The hooks acknowledge their
  lines with the new raw functions `_C._exit_report_written()` / `_C._dlpack_exit_report_written()`. Correct: (l) l3 /
  l6; a hook that wrote everything leaves nothing to repeat.
- **U7. A raise builds its report before it consumes the CUDA error slot**: if that report cannot be built (C++
  `bad_alloc` -> `MemoryError`), the failure stays in the slot and the next `synchronize()` (or launch check) raises it
  as a pre-existing error; `acquire_event` records it instead (its guard already consumed it). Exception types and
  messages are unchanged otherwise. Correct: (k), FIX5's record-before-consume rule applied to the raising paths.
- **U8. New counters, attributes and test hooks**: `pending_stats()` `exit_marker_failures`,
  `exit_device_wait_watchdogs`, `exit_log_fallbacks`, `exit_report_unwritten` (L2:
  `exit_device_wait_watchdog_lock_failures`, `exit_device_wait_watchdog_detaches`); `_dlpack_release_stats()`
  `shutdown_log_fallbacks`, `unraisable_unreported`, `exit_report_unwritten`; raw functions `_exit_report_written`,
  `_dlpack_exit_report_written`, `_dlpack_test_fail_exit_report` (test hook); `_test_inject` sites
  `alloc:exit_log`, `cuda:exit_marker_{create,record,query,destroy}`. Additive.

## Probe before/after table

Every run a fresh process under `flock -w 900 /tmp/forge-gpu.lock timeout <bound>`. Round 8:
`artifacts/p1t_fix8/run_rev8_probes.sh` (summaries `artifacts/p1t_fix8/probe_runs/rev8_{before,after}.summary`, logs in
`probe_runs/rev8_{before,after}/`). Before = `wt_p1trev8` itself (read-only; binaries core `750112413b78...`, tc_signal
`ca5770c152f6...`, recorded in `rev8_before/binaries_before.txt`); after = this worktree (core `d6b1b7ad1d87...`,
tc_signal `29409a0026e0...`). The probes run unmodified from each arm's tree, except the path-only copies of
`make_rev8_copies.py` (build dir, arm root and receipt paths only; diffs in `probe_runs/rev8_*/`, D1). The V8 trace
modules (`diagnostic_errors_cpu.cpp`, `exit_log_oom_cpu.cpp`), the core OOM host module (`build_core_host_oom.py`) and
the GPU event shim are built from each arm's own sources with the reviewer's command lines.

| Probe (round 8) | Before (6991939) | After (this build) | Result |
|---|---|---|---|
| `diagnostic_errors_cpu.py` (V8-01, CPU trace) | **rc 1**: 4 / 4 one-shot marker failures erased (`error: null`, `slot: 0`) | **rc 0**: each raises naming its call ("the marker event of interpreter exit's diagnosed wait for the legacy stream of device 0 could not be created (cudaEventCreateWithFlags: ...)", recorded / queried / destroyed), slot 0 | **FIXED** |
| `verify_exit_log_cpu.py` (V8-02 add-on; children `exit_log_oom_cpu.py --control / --oom`) | **rc 1**: OOM arm writes nothing (`record_still_pending= True`); control names cudaEventQuery | **rc 0**: OOM arm "1 CUDA failure(s) (the full report could not be built: it failed to allocate; allocation-free summary), the first: cudaEventQuery failed, querying a pending entry's completion event, on CUDA device 0: cudaErrorInvalidValue", `record_still_pending= False`; control unchanged | **FIXED** |
| `verify_core_exit_log_cpu.py` (V8-02 core; children `core_exit_log_oom_cpu.py --control / --oom`) | **rc 1**: OOM arm writes nothing; control logs the record | **rc 0**: OOM arm "tensor_cuda: at interpreter exit, 1 CUDA failure(s) ... (allocation-free summary), the first: host probe error [CUDA device 0: cudaErrorInvalidValue]"; control unchanged | **FIXED** |
| `diagnostic_events_gpu.py` (V8-01, GPU, LD_PRELOAD event shim) **as written** | **rc 1**: 4 / 4 rows rc 1, "one-shot marker failure erased" | **rc 1** (RED, as written): all 4 rows rc 0 and each message names its call; the parent's last check ("interpreter exit is waiting" for create / record) fails because the probe's GPU work is not on the legacy stream: under torch 2.11, `torch.cuda.stream(torch.cuda.ExternalStream(0))` makes a NON-BLOCKING stream current (`cuda_stream=0x13d89d80`, `cudaStreamGetFlags` = 1), so `cudaStreamSynchronize(cudaStreamLegacy)` returns at once and there is no exit wait to diagnose (receipt `artifacts/p1t_fix8/extstream_receipt.txt`) | probe precondition not met (D2) |
| `diagnostic_events_gpu.py`, scenario copy with the work on torch's default (legacy) stream (`probe_copies/diagnostic_events_gpu_legacy.py`, one-line diff) | **rc 1**: create / record / query rows rc 1 with **0** "waiting" lines (the silent wait) and no failure named; destroy row rc 1 (not named) | **rc 0**: every row rc 0, 6 "waiting" lines each, every failure named | **FIXED** |
| builds: `build_cpu.sh`, `build_core_host.py`, `build_core_host_oom.py`, diag / oom modules, event shim | rc 0 (all six) | rc 0 (all six) | PASS |
| `run_cpu_checks.py` | 14 / 14 rc 0 | 14 / 14 rc 0 | PASS |
| `argument_census_cpu.py` / `operand_census_cpu.py` / `attribution_cpu.py` | PASS 1067 / 17 / 22 | PASS 1067 / 17 / 22 | PASS |
| `fix7_cpu_census.py` / `lock_checks_cpu.py` | PASS 36 children / 3 stagings | PASS 36 / 3 | PASS |
| `knob_exit_cpu.py`, `core_knob_exit_cpu.py` (each with `--control`), `query_diagnostic_cpu.py` | rc 0 (all five) | rc 0 (all five) | PASS |
| `gpu_regressions.py fix7` / `fix56` / `semantics` / `multidevice` | PASS 115 / 157 / 504, zero skips; exit 3 | PASS 115 / 157 / 504, zero skips; exit 3 | PASS (multidevice INCONCLUSIVE: one GPU) |

Earlier rounds, after arm (FIX7's runners copied to `artifacts/p1t_fix8/runners/` and retargeted by `sed` to this
worktree and `/tmp/p1t_fix8_*`; logic unchanged; absolute output directories; summaries
`artifacts/p1t_fix8/probe_runs/{rev7,rev6,rev5,rev4,rounds123}_after*.summary`). Outcomes equal FIX7's after-arm runs
(`artifacts/p1t_fix7/probe_runs/*`, compared line by line: only the new stats keys and wall times differ):

| Round | Result |
|---|---|
| 7 (`run_rev7_probes.sh`) | every G1 probe rc 0 (`knob_exit_cpu` no GPU / GPU, `core_knob_exit_cpu`, `core_knob_exit_gpu`, `query_diagnostic_cpu` / `gpu`); every control rc 0; `run_cpu_checks` 14 / 14; `argument_census` 1067; `gpu_regressions fix56` 157 and `semantics` 504, zero skips; `multidevice` exit 3 |
| 6 (`run_rev6_probes.sh`) | 30 / 30 runs rc 0 |
| 5 (`run_rev5_probes.sh` cpu / buildcpu / gpu / rf) | as FIX6 / FIX7: `exit_gate trace` as written rc 1 and its `--control` rc 1 (its child models the hook as `synchronize()` alone: FIX6 D7), its exit-entry copy rc 0; `handle_insert_oom_cpu` / `fill_grid_cpu` as written rc 1 (they embed 1475fae code), regenerated from this tree rc 0 / rc 0; every other run rc 0 (`release_failures` 66 passed, `addon_lifetime` 12 / 12, `core_exit_overrun` during 3.1 s / bounded 7.6 s); `core_export_device`, `core_device_drain`, `cuda_errors_multidevice` exit 3 (one GPU) |
| 4 (`run_rev4_cpu_probes.sh`) | as FIX7: 29 runs rc 0; `sync_slot_cpu` rc 1 (arm-independent, as FIX5-FIX7); `core_interop` import regenerated from this tree refuses (rc 1, the correct outcome). The first run's regeneration did not compile (the regenerated `from_dlpack` calls kernels.cu's new `throw_cuda_report_after_consuming`, which the probe's embedded 62c8d9f slices lack); the regen script copy now inserts that helper's text from the tree (D3); first run kept as `rev4_after_first_run*` |
| 1-3 (`run_rounds123_probes.sh`) | `addon_lifetime` 3 runs 12 / 12, 0 incorrect each; `core_edges`, `core_shutdown` (normal, finalizing), `numerics_and_streams`, round-1 / round-2 `host_checks` rc 0; `core_exit_overrun` during 3.1 s / bounded 7.5 s rc 0; `core_device_drain` exit 3 (one GPU) |

## Gate table

| Gate | Result | Receipt |
|---|---|---|
| G1 `diagnostic_errors_cpu.py` | **PASS**: 6991939 rc 1 -> this build rc 0 | `artifacts/p1t_fix8/probe_runs/rev8_{before,after}/diagnostic_errors_cpu.log` |
| G1 `verify_exit_log_cpu.py` (OOM arm of `exit_log_oom_cpu.py`) | **PASS**: rc 1 (`record_still_pending= True`, nothing written) -> rc 0 (summary line written, `record_still_pending= False`) | `.../verify_exit_log_cpu.log`, `exit_log_oom_cpu--oom.log` |
| G1 `core_exit_log_oom_cpu.py` (OOM arm, judged by `verify_core_exit_log_cpu.py`) | **PASS**: rc 1 (nothing written) -> rc 0 (summary line written) | `.../verify_core_exit_log_cpu.log`, `core_exit_log_oom_cpu--oom.log` |
| G1 `diagnostic_events_gpu.py` | **RED as written**: rc 1 -> rc 1. All 4 rows now rc 0 and every failure is named; the parent's "interpreter exit is waiting" check fails because the probe's work is not on the legacy stream on this host (D2). Legacy-stream scenario copy: rc 1 -> **rc 0** | `.../diagnostic_events_gpu.log`, `.../diagnostic_events_gpu_legacy.log`, `artifacts/p1t_fix8/extstream_receipt.txt` |
| G1 controls | **PASS** on both arms: `exit_log_oom_cpu --control`, `core_exit_log_oom_cpu --control` rc 0, and the verifiers' control arms pass | `probe_runs/rev8_{before,after}.summary` |
| G1 new regressions (every finding, every census case) | **PASS**: `test_fix8_findings.py` 29 / 29 and the 6 FIX8 tests of `test_dlpack.py`, inside the combined session (3036 passed = FIX7's 3001 + 35) | `logs/p1t_fix8_g3_combined_session.log` |
| G2 sweeps | **PASS**. (k): 34 rows, FIXED 16 / NOT A FAILURE 3 (cited) / OK 15. (l): 7 last readers (FIXED 6, OK 1) and 13 catches (FIXED 6, OK 7). Every FIXED row has a test or a stated reason. FIX7's i24, i25, j24, a3 and a9 corrected | this report, sections G2 |
| G3 probes, rounds 1-8 | **PASS**, with these items handled as FIX6 and FIX7 handled them: `diagnostic_events_gpu.py` as written (the G1 row above); round-5 `exit_gate trace` as written rc 1 (FIX6 D7), its exit-entry copy rc 0; round-5 `handle_insert_oom_cpu` / `fill_grid_cpu` as written rc 1 (they embed 1475fae code), regenerated from this tree rc 0; round-4 `sync_slot_cpu` rc 1 (independent of the arm, as in FIX5-FIX7); round-4 `core_interop` regenerated import refuses, rc 1 (the correct outcome); `multidevice`, `core_export_device`, `core_device_drain`, `cuda_errors_multidevice` exit 3: INCONCLUSIVE, one GPU | `probe_runs/*.summary` |
| G3 `gpu_regressions.py` | **PASS**: fix7 115, fix56 157, semantics 504, all with zero skips (both arms) | `probe_runs/rev8_after/gpu_regressions_{fix7,fix56,semantics}.log` |
| G3 Voice runner | **PASS 249/249** (T1 11, T2 144, T3 28, T4 12, T5 28, T6 13, T1R 1, T4S 12; FAIL / SKIP / ERROR 0). Voice `git status --short`: 0 lines before and after. Adapter sha256 `600f38b466464ee8bbdd95d009cdf363739bdff79a7e246dcbb4a270dbe43377` | `logs/p1t_fix8_g3_voice.log`, `artifacts/p1t_fix8/p1t_all.json`, `logs/p1t_fix8_voice_git_status_{before,after}.txt` |
| G3 combined session | **PASS**: 37 failed, 3036 passed, 47 skipped, 15 errors (12:16). The compare finds exactly the 52 baseline ids: 0 fixed, 0 new | `logs/p1t_fix8_g3_combined_session.log`, `logs/p1t_fix8_g3_compare.txt` |
| G3 real-torch DLPack interop | **PASS**: 22 passed, 235 deselected (torch 2.11.0+cu130) | `logs/p1t_fix8_g3_torch_interop.log` |
| G4 mutants | **PASS**: (e), (f) and (g) each KILLED by their new tests and their round-8 probe | `artifacts/p1t_fix8/g4/summary_{e,f,g}.txt` |
| G5 scope | **PASS**. The diff against 1cd8824 touches only the Exercises paths, `docs/P1T_FIX8_*.md`, `artifacts/p1t_fix8/**` and `logs/**`; HEAD vs 1cd8824 is `orders/` only. 114 probe files byte-identical; `artifacts/p1t_rev*/` unchanged vs HEAD. Counts of 0 for: new core `__global__`, device atomics, fast-math, TF32, `cudaDeviceSynchronize`, host callbacks, `os._exit` / `abort` / `std::terminate`. No escaping throw in 50 noexcept / destructor bodies. tc_signal does not link `_tensor_cuda` | `artifacts/p1t_fix8/g5_scope.txt` |

Binaries and test files were the same at the probes, at the combined session and at close: `artifacts/p1t_fix8/build_sha256_at_probes.txt`, `build_sha256_at_combined.txt`, `test_files_sha256_at_combined.txt` (`sha256sum -c`: OK).

### Voice's verdict line

```
 "verdict": "PASS",
```
(`logs/p1t_fix8_g3_voice.log:10`; counts `"PASS": 249, "FAIL": 0, "SKIP": 0, "ERROR": 0`; runner exit 0)

### Compare output

```
G1 compare (APA-MF2.2): baseline = /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2/logs/apa_mf1_g1_baseline_unchanged.log; now = logs/p1t_fix8_g3_combined_session.log
summary: 37 failed, 3036 passed, 47 skipped, 2 warnings, 15 errors in 736.17s (0:12:16)
baseline ids 52 now ids 52
baseline ids that no longer fail: 0 []
new failing ids: 0 []
new MF2.2 test files with any FAILED/ERROR: 0
G1 PASS
compare_exit=0
```

### Diff stat (against 6991939)

```
$ git diff --stat 6991939 -- tensor_cuda tc_signal   (6991939 == HEAD f9e6bdd for these paths: 'git diff --stat 6991939 HEAD -- tensor_cuda tc_signal' is empty)
 tc_signal/README.md                 |  22 +-
 tc_signal/src/bindings.cpp          |  91 +++++++-
 tc_signal/src/conv.cu               |   5 +-
 tc_signal/src/cuda_guard.h          |  36 ++-
 tc_signal/src/dlpack_operand.cpp    | 444 ++++++++++++++++++++++++++++++++----
 tc_signal/src/dlpack_operand.h      |  13 ++
 tc_signal/tc_signal/__init__.py     |  16 ++
 tensor_cuda/src/bindings.cpp        | 201 ++++++++++++----
 tensor_cuda/src/kernels.cu          |  42 +++-
 tensor_cuda/tensor_cuda/__init__.py |  17 +-
 tensor_cuda/tests/test_dlpack.py    | 165 ++++++++++++++
 11 files changed, 937 insertions(+), 115 deletions(-)

untracked (new file, the lead adds it):
 tc_signal/tests/test_fix8_findings.py | 456 +++ (new)

totals with the new test file: 12 files changed, 1393 insertions(+), 115 deletions(-)
```

### Binaries (sha256)

```
d6b1b7ad1d87e0d7d5eed70cff94b33e469e6f18b857d78679110e6b2dce3c24  tensor_cuda/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
29409a0026e05df16dfdca81f70430cbcf2cb3d3f56ecd1556eb2ab55b2183b1  tc_signal/tc_signal/_tc_signal.cpython-312-x86_64-linux-gnu.so
```

## G4: mutant table

Each mutant is a copy of this worktree's product tree under `/tmp/p1t_fix8_mut_<m>/` (binaries, tests and every
`artifacts/p1t_rev*/probes` directory included) with ONE asserted edit (`artifacts/p1t_fix8/g4/make_mutants.py`, diffs
`g4/mut_{e,f,g}.diff`), run by `g4/run_mutant.sh` once, at the end, on the final build (every run under the GPU lock).
Before each run the mutant's test files were checked byte-identical to this worktree's and its product sources
identical except for the mutation (`summary_*.txt`, first lines). (e) and (f) rebuild tc_signal; (g) rebuilds the core
offline (D7). The round-8 probe of each runs from the mutant tree, its host module built from the mutant's sources.

| Mutant | The one edit | New tests (this round's, unmodified) | Round-8 probe | Verdict |
|---|---|---|---|---|
| (e) the record of a failed marker call removed | `dlpack_operand.cpp` `note_exit_marker_failure`: the `note_unraisable_failure(...)` call replaced by `(void)` casts (the failure is counted, then consumed silently, as in 6991939) | `test_fix8_findings.py -k v801`: **17 failed**, 2 passed (every create / record / query / destroy case, trace and real module, plain and OOM; the proof test; the 2 healthy controls pass) | `diagnostic_errors_cpu.py` **rc 1** (4 / 4 `error: null`); `diagnostic_events_gpu.py` **rc 1** (4 / 4 rows "erased") | **KILLED** |
| (f) the add-on exit logger's allocation-free fallback removed | `dlpack_operand.cpp` `exit_log_unreported`: `return;` right after the failed full report (nothing written, records kept: 6991939) | `-k v802`: **2 failed** (`..._writes_an_allocation_free_summary_when_the_report_cannot_allocate`, `..._with_stderr_closed_does_not_crash_and_consumes`), 3 passed (control, hook-could-not-write, hook-wrote: they do not pass through the fallback) | `verify_exit_log_cpu.py` **rc 1** (OOM arm: nothing on stderr) | **KILLED** |
| (g) the core runner's allocation-free fallback removed | `tensor_cuda/src/bindings.cpp` `core_exit_last_reader`: `if (!written)` -> `if (false && !written)` | `test_dlpack.py -k fix8_v802`: **2 failed** (`..._core_runner_writes_an_allocation_free_summary_...`, `..._core_runner_with_stderr_closed_...`), 2 passed | `verify_core_exit_log_cpu.py` **rc 1** (OOM arm: nothing on stderr) | **KILLED** |

Receipts: `artifacts/p1t_fix8/g4/summary_{e,f,g}.txt`, `run_{e,f,g}/` (tests, probe logs, binaries' sha256).

A note on (e)'s `diagnostic_events_gpu.py`: as written, that probe fails on the clean build too (D2), so its rc 1 alone
does not show the kill. Its rows do: on the clean build every row is rc 0 and names its call; under (e) every row is
rc 1, "AssertionError: one-shot marker failure erased" (`g4/run_e/diagnostic_events_gpu.log`). (e) is also killed by
`diagnostic_errors_cpu.py` (rc 1, 4 / 4 `error: null`) and by 17 of its new tests.

## Deviations (each with its reason)

- **D1. Path-only copies of ten round-8 scripts** (`artifacts/p1t_fix8/make_rev8_copies.py`: `build_cpu.sh`,
  `build_core_host.py`, `build_core_host_oom.py`, `run_cpu_checks.py`, `argument_census_cpu.py`, `attribution_cpu.py`,
  `fix7_cpu_census.py`, `gpu_regressions.py`, `lock_checks_cpu.py`, `operand_census_cpu.py`). Reason: they write into
  `artifacts/p1t_rev8/` (which must stay byte-identical) and `/tmp/p1t_rev8_build`, and find the arm through
  `parents[3]` / `Path.cwd()`. Only those literals are substituted (each asserted present); diffs in
  `probe_runs/rev8_*/`. The two V8 trace modules and the shim are built with the reviewer's command lines
  (`logs/p1t_rev8_r2.log`) from each arm's sources. The G1 probes themselves ran unmodified.
- **D2. `diagnostic_events_gpu.py` as written FAILS on this build (RED, reported as such).** All four of its rows pass
  (each marker failure is reported, naming its call); the parent's check that a failed create / record "still gets a
  wait diagnostic" cannot be met because the probe's GPU work is not on the legacy stream:
  - under torch 2.11.0+cu130, `torch.cuda.stream(torch.cuda.ExternalStream(0))` makes a non-blocking stream current
    (`cuda_stream=0x13d89d80`, `cudaStreamGetFlags` = 1);
  - so `cudaStreamSynchronize(cudaStreamLegacy)` returns at once and there is no exit wait to diagnose. Receipt:
    `artifacts/p1t_fix8/extstream_receipt.txt`.
  - Printing a "waiting" line with nothing to wait for would be false, so the product is not changed to satisfy it.
  - A scenario copy that queues the same work on torch's default stream (`cuda_stream=0x0`, the legacy stream), with a
    one-line diff (`artifacts/p1t_fix8/probe_copies/diagnostic_events_gpu_legacy.{py,diff}`), FAILS on 6991939 (silent
    waits, failures not named) and PASSES here.
  - My own real-module tests stage the work the same way (`test_fix8_v801_real_module_*`).
- **D3. The round-4 core_interop regeneration copy was extended** (`artifacts/p1t_fix8/runners/regen_core_interop_import.py`).
  - The regenerated `from_dlpack` calls kernels.cu's new `throw_cuda_report_after_consuming`, which the probe's
    embedded 62c8d9f slices lack, so the first run's regeneration did not compile.
  - The copy now inserts that helper's text from the tree, renamed (a tooling change; the reviewer probe is unchanged).
  - First run kept as `probe_runs/rev4_after_first_run*`.
- **D4. The before arm is `wt_p1trev8` itself**, read-only (allowed by the order); all writes went to
  `/tmp/p1t_fix8_*` and `artifacts/p1t_fix8/`.
- **D5. The fallback for a failed marker is a watchdog thread, not the order's suggested polled `cudaStreamQuery`.**
  - The order calls `cudaStreamQuery` "one candidate".
  - It is not among the eleven CUDA runtime functions the reviewers' host traces stub (`release_trace.cpp`,
    `diagnostic_errors_cpu.cpp`, `exit_log_oom_cpu.cpp`). Calling it from `dlpack_operand.cpp` would leave those modules
    with an undefined symbol at import, failing the round-6 and round-8 CPU probes.
  - The watchdog uses no CUDA call. It diagnoses the stream sync on the same schedule and is joined before the sync's
    result is used.
- **D6. Sweep (k) covers the raising paths too.** Their report (message and exception object) is now built before the
  slot entry is consumed (U7). The remaining step, pybind11's translation into a Python `RuntimeError` under CPython
  memory exhaustion, is N1, with citations.
- **D7. Mutant (g)'s core was built offline** in `/tmp/p1t_fix8_mut_g`, with build.sh's cmake commands plus
  `FETCHCONTENT_SOURCE_DIR_PYBIND11` (production's pybind11 source, read-only) and `FETCHCONTENT_FULLY_DISCONNECTED`.
  The worktree's own core build used `./build.sh 89` as ordered. Its first configure fetched pybind11 v2.12.0 through
  FetchContent, as in FIX3-FIX7.
- **D8. New raw functions:** `_exit_report_written`, `_dlpack_exit_report_written` and the test hook
  `_dlpack_test_fail_exit_report` (U8). The round-8 argument census counts the typed (CallArgs) bindings only (still
  28 + the raw exit entry); it passed 1067 checks on this build.
- **D9. My own mistakes, fixed before acceptance:**
  - the core `.so` was not rebuilt after a stats-key edit;
  - the OOM shim used `dlsym(RTLD_NEXT)` (a segfault with an RTLD_LOCAL-loaded libcudart);
  - the first core-side helper sat in an anonymous namespace (link);
  - a stale FIX7 signature in the G5 checker.
- **D10. The end-of-round battery ran once, detached, with a completion marker** (`run_final_battery.sh`,
  `logs/p1t_fix8_final_battery.done`), as the order's budget rule requires. It ran mutants, Voice, the combined session
  (about 13 min, beyond my tool's 10-minute foreground limit) and torch interop. Each was read only after its output
  was complete.
- **D11. Network:** citations were fetched through the AtlasForge web proxy (CUDA 12.6 Runtime API event and error
  pages, CPython 3.12.3 `Python/errors.c`; `artifacts/p1t_fix8/citations/`). The pybind11 v2.12.0 excerpt is from the
  production FetchContent source (read-only). The rest is marked "unverified, lead to check", with search terms.
- **D12. A G1 probe is RED as written, and I did not change the product to make it pass.** This is D2 seen from the
  gate. G1 requires `diagnostic_events_gpu.py` to pass on this build; as written it does not.
  - What it reports for V8-01 is fixed: rows rc 1 -> rc 0, and every failure is named.
  - Its remaining assertion needs an exit wait that does not exist on this host's torch.
  - The legacy-stream copy is offered as the evidence. Whether it closes the row is the lead's decision.

## Prior art

- **V8-01 / sweep (k)**: FIX5 V5-02's record-before-consume rule (taken as is, applied to the marker calls and to
  the raising paths); the CUDA 12.6 Runtime API error-slot contract: `cudaGetLastError` "returns the last error ...
  and resets it to cudaSuccess", `cudaPeekAtLastError` "does not reset the error"
  (`artifacts/p1t_fix8/citations/cuda126_runtime_api_error.txt`); `cudaEventQuery` returns `cudaErrorNotReady` "if
  any captured work is incomplete" (`.../cuda126_runtime_api_event.txt`) and the Programming Guide's "not considered an
  error" (`artifacts/p1t_fix4/cuda_doc_quotes.txt:33`) - taken: NotReady is not a failure, everything else is;
  watchdog threads for blocking calls (the kernel's hung-task detector; unverified here, lead to check
  "hung_task_timeout_secs") - taken: a second thread that reports while the first is blocked; FIX3 V3-01's diagnostic
  schedule. Ours: the marker's failure census, the watchdog fallback tied to one wait's schedule, the proof rule for
  "finished", and building a raise's report before its consumption.
- **V8-02 / sweep (l)**: allocation-free last-chance reporting, as in async-signal-safe logging (POSIX `write(2)` is
  async-signal-safe; only static strings and the caller's own integers are formatted) and SQLite's OOM-testing
  discipline (an error must survive the failure of its own reporting; https://www.sqlite.org/testing.html, as the
  round-8 review cited) - taken: the principle and the fault injection; C++17 `std::bad_alloc` (what the fallback
  catches; its `what()` is a static string); `cudaGetErrorName` returning a runtime-owned `const char*` the caller
  never frees (`.../cuda126_runtime_api_error.txt`); CPython 3.12 atexit (an exception escaping a callback is printed,
  the rest still run). Ours: the plain-data summary of the tagged records, the acknowledged exit report (a report a
  hook raised is consumed only once the hook says it wrote it), and the runner as the last reader in every case.
- **Testing**: the round-8 reviewer's probes (diagnostic_errors_cpu, exit_log_oom_cpu, core_exit_log_oom_cpu,
  diagnostic_events_gpu: their stub layout, OOM injection and staging reused); LD_PRELOAD interposition (FIX1 / FIX4 /
  FIX7 shims), here of `operator new` and `cudaGetDevice`; DeMillo, Lipton & Sayward (1978) mutation testing. Ours: the
  proof assertions, the acknowledged-report stagings, the raising-path OOM census.
- These annotations are also at the code sites (`dlpack_operand.cpp` marker block, watchdog, summary / exit report
  block; `cuda_guard.h` `throw_after_consuming`; `bindings.cpp` runner; core `kernels.cu` `throw_after_consuming` and
  `dlpack_unreported_unraisable_raw`; core `bindings.cpp` exit report block) and in the ledger.

## Lead send-back L2: stopping the exit device wait's watchdog

**Ruling recorded: D2 / D12 ACCEPTED by the lead.**

- The lead reproduced the receipt independently. Under torch 2.11.0+cu130, `ExternalStream(0)` is a non-blocking
  stream (`cudaStreamGetFlags` = 1), and a legacy-stream sync returns in 0.0000 s while the device sync takes 0.08 s.
- The one-line legacy-stream copy is the faithful scenario and counts as the G1 evidence for `diagnostic_events_gpu.py`.
- The as-written run stays recorded as RED, with that reason.

**Finding (lead): `DeviceWaitWatchdog::stop()`** (`tc_signal/src/dlpack_operand.cpp` 1248-1265 before L2).

- (a) If `th_.join()` throws (a `pthread_join` failure), `stop()` detached a thread that could still be inside `run()`.
  That thread uses `mu_`, `cv_` and the caller's `DeviceWaitSchedule& s_`, all destroyed when `sync_device_legacy`
  returns: a use-after-free at exit.
- (b) If the `lock_guard` in `stop()` throws, the `notify_all` can be missed (the watchdog between its predicate check
  and its wait). The watchdog then sleeps until `s_.next` and writes nothing: a silent extra wait at exit.

### Fix (file:line, this build)

| Item | Where | What |
|---|---|---|
| `run()` sets `exited_` as its very last action | `dlpack_operand.cpp:1293-1296` (`run()`), `:1298` (`diagnose()`, the former body of `run()`), `:1322` (`std::atomic<bool> exited_`) | `run()` = `diagnose(); exited_.store(true, release);`. Every member use, including the `unique_lock`'s unlock of `mu_`, ends inside `diagnose()`. After the store the thread touches no member and not the caller's schedule. |
| `stop()` loops until `exited_`, then joins | `:1267-1290` | Sets `done_`. If the thread started (`th_.joinable()`), it loops until `exited_` (acquire): attempts the lock (a failure is caught and counted), calls `notify_all`, checks `exited_`, then sleeps `tuning::kExitDevicePollMs` (1 ms). It joins only then. If the join fails, it counts the failure and detaches, which is safe because `run()` has finished. No early return. |
| Counters | `:1118-1119`, `pending_stats()` `:2254-2255` | `exit_device_wait_watchdog_lock_failures`, `exit_device_wait_watchdog_detaches` |
| Code-site annotation | `:1231-1246` | The stop protocol, the stated limit and prior art |
| README | "Interpreter exit" FIX8 bullet, counters, test description | Stopping waits until the thread has finished; the new counters; the L2 tests |

No other product file changed; the core binary is unchanged. `dlpack_operand.cpp` line references in the tables above
this section are for the pre-L2 source: lines 1118-1228 shift by +2, the watchdog class by +19, and every line from
1297 on by +31.

### Tests (new: `tc_signal/tests/test_fix8_findings.py`, 4 cases, through an LD_PRELOAD interposer)

**Staging.** The interposer wraps `pthread_create`, the mutex lock and unlock, `pthread_cond_clockwait`,
`pthread_cond_broadcast`, `pthread_join` and `pthread_detach`.
- It recognizes the watchdog as the first thread the main thread creates once armed, right before the exit runner.
  This is cross-checked by `exit_device_wait_watchdogs == 1`.
- It filters by thread and by caller (return address in `_tc_signal*.so`).
- The marker cannot be created (test hook), about 0.4 s of GPU work is on the legacy stream, and the native runner
  calls a hook that calls `synchronize()`.
- The diagnostic delay is the default 5 s, so a missed wakeup would leave the watchdog asleep for about 4.6 s.

| Test | Staging | Asserts |
|---|---|---|
| `test_fix8_l2_watchdog_stop_control` | nothing fails | joined normally, no detach, the wait from `stop()`'s first lock attempt to the join's return < 0.5 s |
| `test_fix8_l2_failed_stop_lock_costs_no_silent_wait[1]`, `[3]` | the next n main-thread stop-lock attempts fail, and the notify after each failed lock is withheld (the worst case of (b)) | rc 0, no abort marker; all n failures injected and counted by the product; n notifies withheld; the wait from the first stop-lock attempt to the join's return < 0.5 s; no failed join, no detach |
| `test_fix8_l2_failed_join_detaches_only_after_the_watchdog_finished` | `pthread_join` fails once for the watchdog thread; the watchdog is held 50 ms after the wakeup `stop()` sends (a preempted watchdog) | rc 0, no abort marker; one failed join, one detach (product counter 1); **no** watchdog mutex or wait call after the detach, and its last one came before the detach (`run()` had finished) |

Measured on this build (`artifacts/p1t_fix8/l2/measurements_l2_build.txt`) and on mutant (h)
(`artifacts/p1t_fix8/g4/run_h/witness_summary.txt`), with the same shim and script:

| Staging | This build | Mutant (h) = FIX8's `stop()` |
|---|---|---|
| control | stop to join 0.0011 s | (the control test passes) |
| stop-lock fails 1x | stop to join **0.0022 s**; 1 failure, 1 notify withheld; rc 0 | stop to join **4.6173 s**, nothing written (silent); rc 0 |
| stop-lock fails 3x | stop to join **0.0043 s**; 3 / 3 failures, 3 notifies withheld; rc 0 | stop to join **4.6194 s** (FIX8 makes one attempt: 1 failure) |
| join fails 1x | detached after the watchdog's last call (-0.0001 s), **0** calls after the detach; rc 0 | detached while the watchdog was still inside `run()`: **2** calls after the detach, the last 0.050 s after it (the use-after-free); rc 0 (no crash this time, nothing guaranteed) |

### Sweep (l) addendum: every native thread P1-T starts

Method: grep of `std::thread`, `pthread_create`, `std::async`, `std::jthread` and `PyThread_start` over `tc_signal/src`
and `tensor_cuda/src`, plus the two packages' Python hooks. One site; the core starts no native thread.

| # | Thread (file:line) | Started | Stopped | Joined | Detached | Verdict | Test |
|---|---|---|---|---|---|---|---|
| t1 | `DeviceWaitWatchdog` (`dlpack_operand.cpp:1252`) | by `sync_device_legacy`, inside the exit runner's window only, when the marker failed. If it cannot start, a line says so (the wait still happens). | `stop()` sets `done_`, then repeats lock attempt, `notify_all` and a 1 ms sleep until `run()`'s last action `exited_`. A failed lock costs one interval; a missed wakeup is repeated. | after `exited_`, by `stop()` (called explicitly before the sync's result is used, and again by the destructor, where it does nothing) | only if `pthread_join` fails, after `exited_`. No member and not the caller's schedule is touched after it. | **FIXED** (L2; before: (a) and (b)) | `test_fix8_l2_*` (4) |

Summary (l) addendum: 1 native thread, **FIXED 1**.
- Stated limit: a watchdog blocked writing to a blocked stderr keeps `stop()` waiting, as the join did. The main
  thread's own next line would block on the same stderr.

### U2 (updated)

U2 in "User-visible changes" above now states the stop protocol, its measured cost and the two counters.

### Re-run on the L2 build (tc_signal changed; the core did not)

| Gate | Result | Receipt |
|---|---|---|
| G1 round-8 probes, after arm | `diagnostic_errors_cpu` **rc 0**; `verify_exit_log_cpu` **rc 0** (`exit_log_oom_cpu --control` / `--oom` rc 0, OOM arm: the allocation-free summary); `verify_core_exit_log_cpu` **rc 0** (`core_exit_log_oom_cpu --control` / `--oom` rc 0); `diagnostic_events_gpu.py` as written rc 1 (accepted D2: 4 / 4 rows rc 0, "failed marker setup fell back to a silent wait" only); legacy-stream copy **rc 0** (4 / 4 rows rc 0) | `artifacts/p1t_fix8/probe_runs/rev8_after_l2.summary`, `rev8_after_l2/` (runner `artifacts/p1t_fix8/run_rev8_g1_l2.sh`) |
| `test_fix8_findings.py` | **33 passed** (FIX8's 29 + L2's 4) | `logs/p1t_fix8_l2_test_fix8_findings.log` |
| G4 mutants | (e) refreshed **KILLED**, (f) refreshed **KILLED**, (h) **KILLED** (table below) | `artifacts/p1t_fix8/g4/summary_{e_l2,f_l2,h}.txt` |
| Voice runner | **PASS 249/249** (T1 11, T2 144, T3 28, T4 12, T5 28, T6 13, T1R 1, T4S 12); Voice `git status --short` 0 lines before and after; adapter sha256 `600f38b466464ee8bbdd95d009cdf363739bdff79a7e246dcbb4a270dbe43377` | `logs/p1t_fix8_l2_voice.log`, `artifacts/p1t_fix8/p1t_all_l2.json` |
| Combined session | **37 failed, 3040 passed** (3036 + 4), 47 skipped, 15 errors (12:23); compare: exactly the 52 baseline ids, 0 fixed, 0 new | `logs/p1t_fix8_l2_combined_session.log`, `logs/p1t_fix8_l2_compare.txt` |
| Real-torch interop subset | 22 passed (torch 2.11.0+cu130) | `logs/p1t_fix8_l2_torch_interop.log` |
| G5 | **PASS**: paths within scope; HEAD vs 1cd8824 `orders/` only; 114 probe files and the 57 under `artifacts/p1t_rev8` byte-identical; 0 new core `__global__`, device atomics, fast-math, TF32, `cudaDeviceSynchronize`, host callbacks, `os._exit` / `abort` / `std::terminate`; no escaping throw in 51 noexcept / destructor bodies (`diagnose()` added) | `artifacts/p1t_fix8/g5_scope_l2.txt` |

Binaries were identical at the start of the battery, at the combined session and at close
(`build_sha256_at_l2_battery_start.txt`, `build_sha256_at_l2_combined.txt`; `sha256sum -c` OK). The test files were
unchanged since the combined session (`test_files_sha256_at_l2_combined.txt`).

Voice's verdict line:

```
 "verdict": "PASS",
```
(`logs/p1t_fix8_l2_voice.log:10`; counts `"PASS": 249, "FAIL": 0, "SKIP": 0, "ERROR": 0`; runner exit 0)

Compare output:

```
G1 compare (APA-MF2.2): baseline = /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2/logs/apa_mf1_g1_baseline_unchanged.log; now = logs/p1t_fix8_l2_combined_session.log
summary: 37 failed, 3040 passed, 47 skipped, 2 warnings, 15 errors in 743.00s (0:12:23)
baseline ids 52 now ids 52
baseline ids that no longer fail: 0 []
new failing ids: 0 []
new MF2.2 test files with any FAILED/ERROR: 0
G1 PASS
compare_exit=0
```

### Mutant table (L2)

Each mutant tree is new, under `/tmp` (the earlier trees stay; no deletes). Each carries one asserted edit
(`artifacts/p1t_fix8/g4/make_mutants.py`; diffs `g4/mut_e_l2.diff`, `mut_f_l2.diff`, `mut_h.diff`) and was built from
the L2 sources. The test files were checked byte-identical to this worktree's, and the product sources identical
except for the mutation.

| Mutant | The one edit | New tests (unmodified) | Round-8 probe / witness | Verdict |
|---|---|---|---|---|
| (e) refreshed: the record of a failed marker call removed | as before (edit lines identical to `mut_e.diff`) | `-k v801`: **17 failed**, 2 passed | `diagnostic_errors_cpu.py` **rc 1** (4 / 4 `error: null`); `diagnostic_events_gpu.py` rc 1 with every row rc 1, "erased" | **KILLED** |
| (f) refreshed: the add-on exit logger's allocation-free fallback removed | as before (identical edit lines) | `-k v802`: **2 failed**, 3 passed | `verify_exit_log_cpu.py` **rc 1** | **KILLED** |
| (h) `stop()` reverted to FIX8's code (one lock attempt, one notify, join, detach at once on failure) | `mut_h.diff` (36 lines) | `-k l2`: **3 failed** (`lock[1]`, `lock[3]`, `join`), 1 passed (control) | no round-8 probe exists for a lead finding. Behavioural witness, same shim: stop-lock 1x, 4.62 s silent wait; join 1x, 2 watchdog calls after the detach | **KILLED** |

Note on (h): in each failing test, the first assertion that fails is a product counter (0 where L2 counts: FIX8's
`stop()` never counts). The witness runs (`g4/run_h/witness_*.log`) show the behaviour behind it, with the same
staging: the 4.6 s silent wait and the calls after the detach.

### Diff stat (against 6991939, after L2)

```
$ git diff --stat 6991939 -- tensor_cuda tc_signal   (after lead send-back L2; 6991939 == HEAD f9e6bdd for these paths)
 tc_signal/README.md                 |  27 +-
 tc_signal/src/bindings.cpp          |  91 ++++++-
 tc_signal/src/conv.cu               |   5 +-
 tc_signal/src/cuda_guard.h          |  36 ++-
 tc_signal/src/dlpack_operand.cpp    | 477 ++++++++++++++++++++++++++++++++----
 tc_signal/src/dlpack_operand.h      |  13 +
 tc_signal/tc_signal/__init__.py     |  16 ++
 tensor_cuda/src/bindings.cpp        | 201 ++++++++++++---
 tensor_cuda/src/kernels.cu          |  42 +++-
 tensor_cuda/tensor_cuda/__init__.py |  17 +-
 tensor_cuda/tests/test_dlpack.py    | 165 +++++++++++++
 11 files changed, 975 insertions(+), 115 deletions(-)

untracked (new file, the lead adds it):
 tc_signal/tests/test_fix8_findings.py | 688 +++ (new)
```

### Binaries (sha256, after L2)

```
d6b1b7ad1d87e0d7d5eed70cff94b33e469e6f18b857d78679110e6b2dce3c24  tensor_cuda/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
30aee4e67fbe80320863cf9d0d4b7748dca46c0db7a3689f70bcc9d9e8dc782b  tc_signal/tc_signal/_tc_signal.cpython-312-x86_64-linux-gnu.so
```
(core unchanged; tc_signal before L2: `29409a0026e05df16dfdca81f70430cbcf2cb3d3f56ecd1556eb2ab55b2183b1`)

### L2 deviations (each with its reason)

- **L2-D1. The lock-failure staging withholds every notify that follows a failed stop-lock.** This is a stronger
  adversary than the real race, in which only a notify that arrives between the watchdog's predicate check and its
  wait is lost. Reasons:
  - it is deterministic;
  - with a delivered notify, the loop ends at the first iteration, so the n failures would not all reach `stop()`.
  
  The new code is bounded even under this adversary.
- **L2-D2. The mutant tooling gained an optional tag argument and case (h).** `make_mutants.py <m> [tag]` and
  `run_mutant.sh <m> <out> [tag]` were extended.
  - With no tag, they behave as before. Pre-L2 copies are in `artifacts/p1t_fix8/l2/*.pre_l2.*`.
  - (e) and (f) were refreshed in new trees (`/tmp/p1t_fix8_mut_{e,f}_l2`), because `make_mutants.py` refuses an
    existing target and nothing is deleted.
- **L2-D3. Mutant (h)'s witness runs are receipts beyond the order.** They were run with the tests' shim and script
  (copies in `g4/run_h/`), to show behaviour where the tests' first failing assertion is a counter.
- **L2-D4. The re-run was limited to the lead's list.** The list: the round-8 G1 probes and their controls, the legacy
  copy, `test_fix8_findings.py`, mutants (e), (f) and (h), Voice, the combined session and G5.
  - The G1 probes ran through a new G1-only runner, `artifacts/p1t_fix8/run_rev8_g1_l2.sh`. It uses the same commands
    as `run_rev8_probes.sh`, restricted to G1.
  - Not re-run: the inherited round-8 probes, `gpu_regressions.py` and rounds 1-7. Reason: the change is
    tc_signal-only, and the combined session covers both packages' test suites.
  - The real-torch interop subset (3 s) was re-run as well.
- **L2-D5. The G5 checker gained `diagnose()` in its list of noexcept bodies.** Pre-L2 copy:
  `artifacts/p1t_fix8/l2/g5_scope.pre_l2.py`.

### Prior art (L2)

- **The exit flag as the thread's last write, read with acquire before its state is reclaimed.** This is the C++11
  release / acquire publication idiom: a release store observed by an acquire load orders every earlier write and use
  before what follows. See Boehm & Adve (2008), "Foundations of the C++ concurrency memory model".
  - Taken: the ordering guarantee.
  - Mine: placing it as `run()`'s last action and the reclaim rule of `stop()` (join, or detach only after the flag).
  - **Unverified, lead to check**: "thread done flag release acquire before detach reclaim".
- **The lost wakeup, and repeated notification as its cure.** The lost-wakeup problem of condition variables; Lampson
  & Redell (1980), "Experience with Processes and Monitors in Mesa": waiters re-check their condition, and timeouts
  bound a missed notify.
  - Taken: re-notifying on a bounded period instead of trusting one notify.
  - Mine: tying the period to `kExitDevicePollMs` and to `exited_`.
  - **Unverified, lead to check**: "lost wakeup condition variable notify without mutex", "Lampson Redell Mesa monitors
    1980".
- **Testing.**
  - LD_PRELOAD interposition with `dlsym(RTLD_NEXT)` and caller filtering by return address plus `dladdr`, as in
    FIX7's lock-failure shims. Here it covers `pthread_create` (a trampoline that marks the thread's return),
    `pthread_cond_clockwait` / `pthread_cond_broadcast`, `pthread_join` and `pthread_detach`.
  - Fault injection of a preempted thread (a hold after the wakeup).
  - DeMillo, Lipton & Sayward (1978), mutation testing.
  - Mine: the after-detach witness and the stop-wait measurement.
- These annotations are also at the code site (`dlpack_operand.cpp:1231-1246`), in the test module's L2 header and in
  the ledger.
