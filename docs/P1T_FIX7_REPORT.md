# P1T-FIX7 report: round-7 findings V7-01 and V7-02 closed; sweeps (i) exit-guarantee reachability and (j) error attribution

**Verdict: every gate PASSES (first report; the lead's send-back L1 is answered in the section "Lead send-back L1" at the end, where every listed gate was re-run on the L1 build and passes).** V7-01 (MAJOR) and V7-02 (MINOR) are closed. The five round-7 G1 probes fail on
429446d (rc 134 / 134 / 134 / 1 / 1 / 1, counting `knob_exit_cpu` twice: no GPU and GPU) and pass on this build (rc 0);
every `--control` passes on both. Sweep (i) has 26 rows: FIXED 18, OK 8, and every FIXED row has a test. Beyond V7-01's
knob, it found a third way 429446d could skip the exit close: a failed lifecycle-mutex lock in the core (row i19). Under
one injected lock failure 429446d aborts at exit (rc -6); that is fixed and tested. Sweep (j) has 35 rows: FIXED 21,
OK 14, and every FIXED row has a test. G3: every round-1..7 probe passes (multidevice INCONCLUSIVE on one GPU; round-5
`exit_gate trace` handled as in FIX6 D7); Voice PASS 249/249; the combined session gives `G1 PASS` with exactly the 52
baseline ids, 0 new and 2996 passed; real-torch interop passes 22 / 22. G4: mutants (a), (b) and (c) are KILLED. G5
PASSES. No defect residual remains; the "Not a defect" table holds 4 cited items.

Seat: Opus 5.5 (high). Order: `orders/P1T_FIX7_EXIT_HOOK_REACHABILITY_AND_ATTRIBUTION.md` (420e821). Ledger:
`docs/P1T_FIX7_LEDGER.md`. Receipts: `artifacts/p1t_fix7/` and `logs/p1t_fix7_*`. "Before" = the 429446d build in
`wt_fix6` (read-only), run through the probe tree `/tmp/p1t_fix7_before_tree` (byte-identical copies of wt_fix6's
sources, tests and binaries: core `e7d4a3a313562d80af0401f1deba819260fa95a6ea86abb3a53443c3e17b5a10`, tc_signal
`ef44b9c455f26a7b40d3429d70b7e08ebd4b35444689811de90428e063f2f197`, both equal to 429446d's; the reviewer's core slices
regenerated from it equal the reviewer's pinned hashes, 7/7: `probe_runs/rev7_before/slices_selfcheck_before.txt`).

## Per-finding table

| Finding | Fix (file:line) | Proving tests (new) | Probe before (429446d) | Probe after (this build) |
|---|---|---|---|---|
| **V7-01** (MAJOR): the exit hooks skipped their close-and-wait when the diagnostic knob could not be converted (`float(1 << 4096)`: OverflowError, not in `(TypeError, ValueError)`; the outer `except Exception` logged and returned; a finite deleter still running on a daemon thread was overtaken by finalization: abort) | Add-on: `tc_signal/tc_signal/__init__.py:163` hook rewritten: the knob is never converted (passed as is, a deleted knob passes None), step 1 `synchronize()` in `try` and the close in its `finally` (`:222-231`), every stderr line after the close (`:232-260`); `:160` knob initial value = native default; `:266` registered through the native runner. Native: `src/py_args.h:239` `raw_exit_diag_arg` (shape errors TypeError, any VALUE readable at C level or the default `tuning.h` `kExitDiagDefaultS`, never an error before the close; the value rule is `:222` `raw_exit_diag_value`); `src/bindings.cpp:356` `raw_exit_close_admission`; `:417` `raw_exit_hook_runner` (atexit calls it with the hook: when the hook RAISES before closing it runs synchronize(), closes, waits, logs the unreported failures, re-raises; when the hook returns it does nothing more; L1: while it runs, the device syncs of exit are diagnosed, `:399` the knob read at C level from the hook's globals); `src/dlpack_operand.cpp:1422-1480` `exit_close_and_wait` / `exit_close_admission` / `exit_runner_close_and_wait`, `:1759` `exit_log_unreported`; L1: `:276` `wait_without_gil` (its polling fallback now diagnosed), `:1049` `wait_device_work_diagnosed` and `:1463` its window (the device waits of exit diagnosed). Core: `tensor_cuda/tensor_cuda/__init__.py:207` hook (no conversion, the close is its first call, R7 logging attempted whatever the close did), `:204` knob, `:293` runner registration; `tensor_cuda/src/bindings.cpp:1495` `raw_exit_diag_arg`, `:1546` `raw_dlpack_begin_shutdown`, `:984` `dlpack_begin_shutdown_impl` (`:939` `lock_retrying`: a lifecycle-mutex lock failure no longer rethrows before `closing` is set; L1: the retry wait is diagnosed like the exit wait, `shutdown_lock_diagnostics`), `:1640` `raw_dlpack_exit_hook_runner`, `:828` exit tunables. Docs: both hook docstrings, `from_dlpack` docstring, `tc_signal/README.md` "Interpreter exit" | `tc_signal/tests/test_fix7_findings.py`: `:267` 22 knob classes at real exit with a deleter in flight (+`:284` 4 without a GPU); `:296` synchronize() raising RuntimeError / MemoryError / KeyboardInterrupt / SystemExit / BaseException subclass; `:312` KeyboardInterrupt at the hook's first bytecode; `:325` `_C` gone; `:334` stderr None / raising / raising KeyboardInterrupt; `:512` interrupted hook still releases and reports; `:540` V6-03 at exit; `:594` native entry value census; `:642` runner; L1: `:420` the polling fallback diagnosed, `:467` the device wait of exit diagnosed. `tensor_cuda/tests/test_dlpack.py`: `:2612` 22 knob classes (core), `:2626` first-bytecode interrupt, `:2635` `_C` gone, `:2643` stderr, `:2797` native entry + runner, `:2899` interrupted hook still reports the R7 record; L1: `:2724` the lock-retry wait diagnosed | `knob_exit_cpu.py` no GPU **rc 134**, GPU **rc 134**; `core_knob_exit_cpu.py` **rc 134**; `core_knob_exit_gpu.py` **rc 1** (child `FATAL: exception not rethrown`, core dumped); every `--control` rc 0 | `knob_exit_cpu.py` no GPU **rc 0**, GPU **rc 0**; `core_knob_exit_cpu.py` **rc 0**; `core_knob_exit_gpu.py` **rc 0** (`DELETER finished` before `FINALIZING True`); every `--control` rc 0 |
| **V7-02** (MINOR): a failed completion-event query after a successful stream sync was reported as "the legacy stream of device 0 could not be synchronized" (untyped `dev` / `err` arrays) | `tc_signal/src/dlpack_operand.cpp:762-939`: `FailedCall` / `Purpose` tag (`:780`), `FailureRecord` (call, purpose, device, error; `:799`), `describe()` (`:821`) builds every message from the tag; `SyncReport` (`:907`) stores tagged records, `merge` keeps them, `first()` (`:936`) returns one consistent record; phase 1 `:1307-1308` (the call that failed, the spill device), phase 2 `:1324` (`kEventQuery`, recorded before its slot entry is consumed); `report_message` (`:1703`) shared by synchronize() and the exit runner; recording allocation-free and saturating (`sat_inc` / `sat_add`, `:759`) | `test_fix7_findings.py` (host trace module of this source with per-call failure plans): `:870` query after a successful sync, `:880` sync, `:888` both, `:896` spill, `:904` current-device query, each also with a message-allocation failure; `:1029` real module (`cuda:event_query` / `cuda:sync` / `cuda:current_device`, each with and without message OOM). Each message names the call and its device exactly once | `query_diagnostic_cpu.py` **rc 1** ("the legacy stream of device 0 could not be synchronized"); `query_diagnostic_gpu.py` **rc 1** | `query_diagnostic_cpu.py` **rc 0**; `query_diagnostic_gpu.py` **rc 0** ("the completion event of a pending entry on device 0 could not be queried (cudaEventQuery: ...)") |

## G2: sweep (i), paths that reach an exit guarantee

Method: for each exit hook, every statement before its close-and-wait entry and every exception it can raise (CPython
3.12 semantics: an asynchronous exception, such as KeyboardInterrupt from Ctrl-C, is delivered only where the eval loop
checks for pending signals: a frame's first instruction (`RESUME`), after a call returns, and at a backward jump; the
bytecode of both hooks was disassembled, ledger "Sweep (i) bytecode"), then the native entries' internals. A row is
FIXED when FIX6 (429446d) could skip the close on it.

| # | Statement / step (file:line) | What can be raised there | How the close-and-wait is reached now | Verdict | Test / reason |
|---|---|---|---|---|---|
| i1 | add-on: atexit calls `_C._exit_hook_runner(hook)` -> `PyObject_CallNoArgs` -> the hook's first instruction (`tc_signal/src/bindings.cpp:417`) | KeyboardInterrupt (a pending SIGINT is delivered at `RESUME`, before any statement) | the hook raises before closing; the runner sees no hook close counted, runs synchronize(), closes, waits (default delay), logs unreported failures, re-raises | **FIXED** (FIX6: no close at all) | `test_fix7_v701_keyboardinterrupt_at_the_hooks_first_bytecode_is_covered_by_the_runner`, `..._interrupted_hook_still_releases_and_reports_recorded_failures` |
| i2 | add-on: `C = _C` (`tc_signal/__init__.py:214`) | NameError if the module lost `_C` | as i1 (the hook raises; the runner closes) | **FIXED** (FIX6: `_C.synchronize` NameError logged, then `_C._exit_close_admission` NameError: no close) | `test_fix7_v701_hook_whose_module_lost_C_is_covered_by_the_runner` |
| i3 | add-on: local initialisations (`:215-217`) | nothing (constant stores) | - | OK | bytecode (no call, no jump) |
| i4 | add-on: `C.synchronize()` (`:219`) raising an Exception (RuntimeError: a CUDA failure; MemoryError: the message) | as stated | caught (`sync_error`), the `finally` closes, the line is written after the close | **FIXED** (FIX6 wrote the line BEFORE closing: a raising stderr could only be survived by its inner try; now nothing touches stderr before the close) | `test_fix7_v701_exit_hook_closes_when_synchronize_raises_anything[RuntimeError, MemoryError]` |
| i5 | add-on: `C.synchronize()` raising a BaseException, or an asynchronous one after it returns | KeyboardInterrupt, SystemExit, any BaseException subclass | the `finally` closes and waits, then the exception propagates (atexit prints it); the runner's wait is then a no-op | **FIXED** (FIX6: `except Exception` let it skip the close) | `..._when_synchronize_raises_anything[KeyboardInterrupt, SystemExit, Marker]` |
| i6 | add-on finally: `knob = _RELEASE_EXIT_DIAG_S` (`:225`) | NameError (the knob deleted); an asynchronous exception at the except handler's backward jump | NameError -> `missing = True`, knob None -> the native default; an asynchronous exception there -> the hook raises before closing -> the runner closes (i1) | **FIXED** (FIX6: `float(<deleted>)` NameError -> outer `except Exception` -> no close) | `..._whatever_the_knob_holds[deleted]` (+ no-GPU variant); i1 tests for the runner path |
| i7 | add-on finally: `C._exit_close_admission(knob)` (`:229`): the knob's VALUE | FIX6 converted it with `float()`: OverflowError (huge int: V7-01), TypeError / ValueError, ANY exception or BaseException a `__float__` raises, a non-float return, user code that blocks | no conversion in Python; the native entry reads float / int (subclasses) at C level, anything else (and an int beyond the double range) selects the default; it ALWAYS closes and waits | **FIXED** (V7-01) | `..._whatever_the_knob_holds` (22 classes: huge int +/-, +/-inf, NaN, negative, zero, float / int subclasses whose methods raise, bool, str, None, `__float__` raising Exception / KeyboardInterrupt / SystemExit / a BaseException subclass, returning a non-float, an `__index__`-only object, numpy float32 / float64, deleted; zero methods called) + no-GPU variants |
| i8 | add-on native entry: argument SHAPE (`py_args.h:239`) | TypeError for a malformed call (arity, unknown / duplicate keyword, missing) | not reachable from the hook (it always passes one positional argument); a malformed direct call does nothing, as any Python call would | OK | `test_fix7_exit_entry_reads_any_value...` (5 malformed calls: TypeError, admission still open) |
| i9 | add-on native entry: VALUE reading (`py_args.h:222-237` (`raw_exit_diag_value`)) | FIX6: TypeError for another type, OverflowError for an int beyond the double range, both BEFORE the close | `PyFloat_AS_DOUBLE` / `PyLong_AsDouble` (overflow cleared) / default; never raises | **FIXED** | `test_fix7_exit_entry_reads_any_value...` (14 values x 2 call styles, 0 method calls); the reviewer's `argument_census_cpu.py` (1067 checks, 0 calls) |
| i10 | add-on native close (`dlpack_operand.cpp:1422` `exit_close_and_wait`): `closing.store`, `pending().q.size()`, `wait_for_other_threads` | nothing: the store is an atomic, the wait releases the GIL, its mutex / condition-variable failures fall back to polling (`wait_without_gil`, FIX5; L1: the polling is diagnosed too, addendum a2) and it never returns early | - | OK | FIX5/FIX6 exit tests; reasoning |
| i11 | add-on: building the result dict after the wait (`:1446`) | MemoryError | after the close and the wait; caught in the finally (`close_error`), logged | OK (FIX6 likewise closed first) | `test_fix7_sweep_i_exit_close_step_raising_after_the_close_is_logged_and_skips_nothing[gpu, nogpu]` (the hook's `_C` a proxy whose exit entry closes for real, then raises MemoryError) |
| i12 | add-on: every stderr write (`:255-260`) | any Exception (stderr None, closed, raising) caught per line; a BaseException propagates | after the close | **FIXED** (moved after the close) | `test_fix7_v701_exit_hook_closes_when_stderr_is_gone_or_raises[none, raises, raises_keyboardinterrupt]` |
| i13 | add-on runner (`bindings.cpp:417`) itself | nothing before its close: C code, no eval-loop boundary; `synchronize_and_release_all` failures are caught and written | - | OK (new) | `test_fix7_exit_runner_closes_only_when_the_hook_raised` |
| i14 | core: atexit calls `_C._dlpack_exit_hook_runner(hook)` -> the hook's first instruction (`tensor_cuda/src/bindings.cpp:1640`) | KeyboardInterrupt at `RESUME` | the runner closes, waits, logs the unreported R7 record, re-raises | **FIXED** | `test_fix7_v701_core_keyboardinterrupt_at_the_hooks_first_bytecode...`, `..._interrupted_core_hook_still_reports_recorded_failures` |
| i15 | core: `C = _C` (`tensor_cuda/__init__.py:246`) | NameError | as i14 | **FIXED** | `test_fix7_v701_core_hook_whose_module_lost_C...` |
| i16 | core: `knob = _DLPACK_SHUTDOWN_WAIT_S` (`:248`) | NameError (deleted); an asynchronous exception at the handler's jump | None -> default; asynchronous -> the runner | **FIXED** (FIX6: `float(<deleted>)` NameError -> no close) | `..._core_exit_hook_closes_and_waits_whatever_the_knob_holds[deleted]` |
| i17 | core: `C._dlpack_begin_shutdown(knob)` (`:254`): the knob's value | as i7 | no conversion; C-level read or default; always closes | **FIXED** (V7-01 core half) | the core 22-class census; probes `core_knob_exit_cpu.py`, `core_knob_exit_gpu.py` |
| i18 | core native entry (`bindings.cpp:1495`, `:1546`) | as i8 / i9 | as i8 / i9 | i8: OK; i9: **FIXED** | `test_fix7_core_exit_entry_reads_any_value...` (4 malformed calls, 9 values x 2 styles) |
| i19 | core native close (`bindings.cpp:984`): locking the lifecycle mutex before `closing = true` | `std::system_error` (C++17 [thread.mutex.requirements.mutex]); FIX3-FIX6 re-took the GIL and RETHREW: the close was skipped | `lock_retrying` (`:939`) retries until the lock is taken (GIL released, counted in `shutdown_lock_retries`); the same for the re-lock after each diagnostic; lead send-back L1: that retry wait is diagnosed like the exit wait (addendum a5) | **FIXED** | `test_fix7_sweep_i_core_exit_close_retries_a_failed_lifecycle_lock_instead_of_skipping_the_close[1, 3]` (an LD_PRELOAD `pthread_mutex_lock` interposer fails the next N locks the main thread makes from `_tensor_cuda*.so`, armed right before the runner at a real exit with a deleter in flight: `shutdown_lock_retries` = N, the hook closed and waited, the deleter finished before finalization). Before / after receipt: `artifacts/p1t_fix7/lock_failure_receipt_{before,after}_{A,B}.txt`: on 429446d one injected failure makes `_dlpack_begin_shutdown` raise `RuntimeError: Invalid argument` with admission left open (A), and at a real exit the hook logs "DLPack exit hook failed (RuntimeError: Invalid argument)", finalization overtakes the deleter and the process aborts (`FATAL: exception not rethrown`, rc -6) (B); this build: rc 0, the deleter finished before finalization |
| i20 | core native wait: `cv.wait_until` on steady_clock | nothing in libstdc++ (pthread_cond_clockwait errors other than a timeout are ignored); FIX5's catch-all that restores the GIL is kept for anything unforeseen | - | OK | reasoning |
| i21 | core: result dict after the wait | MemoryError | after the close; caught (`hook_error`), logged, and the R7 logging still runs | **FIXED** (FIX6: the same outer try skipped the R7 logging) | `test_fix7_sweep_i_core_close_step_raising_after_the_close_does_not_skip_the_recorded_failure_log` (the hook's `_C` a proxy whose exit entry closes for real, then raises MemoryError; the R7 record is still logged) |
| i22 | core: `C._dlpack_raise_unraisable()` and stderr writes | RuntimeError (logged), MemoryError (logged, record kept), stderr failures caught per line, BaseException propagates | after the close; a BaseException then reaches the runner, which logs any record left | OK | `..._core_exit_hook_closes_when_stderr_is_gone_or_raises` |

Other guarantees a hook gives on the way out, checked for early exits:

| # | Guarantee | Early exit that skipped it in 429446d | Now | Verdict | Test |
|---|---|---|---|---|---|
| i23 | add-on: pending operands released at exit (hook step 1) | the hook interrupted at its first bytecode, or `_C` lost | the runner runs `synchronize()` before its close when no hook close happened | **FIXED** | `..._interrupted_hook_still_releases_and_reports_recorded_failures` (entries 0, released += pending) |
| i24 | add-on: a CUDA failure is reported, never lost (N1 / V6-03), at exit | (a) interrupted hook: never reported; (b) the message allocation fails at exit: only "MemoryError" was logged, the kept failure had no later reader | (a) the runner's synchronize() raises it and `exit_log_unreported` writes what remains; (b) the hook calls synchronize() once more after the close, which raises the kept failure | **FIXED** | (a) as i23; (b) `test_fix7_v701_exit_reports_a_failure_whose_message_could_not_be_allocated` |
| i25 | core: FIX6's raise-once logging of the R7 record at exit | (a) the close step raising (same `try`); (b) an interrupted hook | (a) attempted whatever the close did; (b) the runner logs it (C stdio) and acknowledges it | **FIXED** | (b) `test_fix7_v701_interrupted_core_hook_still_reports_recorded_failures`; (a) `test_fix7_sweep_i_core_close_step_raising_after_the_close_does_not_skip_the_recorded_failure_log` (i21) |
| i26 | both: the diagnostic lines (overrun, "finished after") | an interrupted hook | the runner writes its own line when it waited for someone | OK | i1 / i14 tests (runner line asserted) |

Summary (i): 26 rows: **FIXED 18** (i1, i2, i4-i7, i9, i12, i14-i19, i21, i23-i25), **OK 8** (i3, i8, i10, i11, i13,
i20, i22, i26). Every FIXED row has a test (i19 through an LD_PRELOAD lock-failure interposer, i21 / i25(a) through a proxy `_C`; the OK row i11 has one too). The Python `try` / `finally` guarantee (language reference, "The try statement")
covers every row inside the hooks' frames; the native runners cover the frame boundary (i1, i14) and every other
exception that escapes a hook before its close (i2, i6, i15, i16 asynchronous cases). A hook that RETURNS without
closing is not covered by design: that would be a hook defect, and an always-closing backstop would hide it (the
registered G4 mutants (a) / (b) are exactly that; both are killed by the round-7 probes).

Addendum (lead send-back L1): every exit wait or retry loop of both packages, 12 more rows: **FIXED 4** (a2, a3, a5, a6),
**OK 8**; the table is in the section "Lead send-back L1".

## G2: sweep (j), error attribution

Every failure P1-T records or reports, after this round. Rule (order): the message names the call that actually failed
and the right device (or says the device is unknown / not applicable), and never names an operation that succeeded;
recording stays allocation-free and saturating.

| # | Site (file:line) | FIX6 (429446d) text / record | Now | Verdict | Test |
|---|---|---|---|---|---|
| j1 | add-on synchronize() phase 1 current-device query (`dlpack_operand.cpp:1294`) | "the current-device query cudaGetDevice failed (E)" | unchanged text (now from `describe`); no device (the query reads which device is current) | OK | `..._current_device_query_failure_names_cudaGetDevice_and_no_device` (trace, +OOM); real `cuda:current_device` |
| j2 | phase 1 per-device stream sync `rep.add` (`:1307`) | any failure (the guard's cudaGetDevice / cudaSetDevice included) printed as "the legacy stream of device D could not be synchronized (E)" | tagged from `SyncResult`: cudaStreamSynchronize -> "could not be synchronized (cudaStreamSynchronize: E)"; guard -> "was not synchronized: switching to device D failed (cudaSetDevice: E)" / "...the current-device query before switching to device D failed (cudaGetDevice: E)" | **FIXED** | `..._sync_failure_names_cudaStreamSynchronize`, `..._guard_switch_failure_names_cudaSetDevice_not_the_sync`, `..._guard_read_failure_names_cudaGetDevice_not_the_sync` (each +OOM); real `cuda:sync` |
| j3 | spill path `rep.add(-1, spill_error)` (`:1308`; `DeviceSyncSet` `:1114`) | "the legacy stream of device -1 could not be synchronized" | the first failing untracked device is named, with its call, and the conservative rule is stated | **FIXED** | `..._spill_failure_names_its_device` (+OOM) |
| j4 | phase 2 completion-event query `rep.add` (`:1324`) | V7-02 | "the completion event of a pending entry on device D could not be queried (cudaEventQuery: E)" | **FIXED** (V7-02) | `..._query_failure_after_a_successful_sync_names_cudaEventQuery`, `..._sync_and_query_failures_are_each_named_once` (+OOM); real `cuda:event_query`; probes `query_diagnostic_cpu/gpu.py` |
| j5 | `SyncReport::merge` into `drain_unreported` (V6-03 persistence) | kept (dev, err) only | keeps the tags; saturating | **FIXED** | every `[message_oom]` variant above |
| j6 | `SyncReport::unnamed` (beyond `kSyncReportMax`) | "N more device failure(s)" (they may be query failures) | "N more failure(s) of this drain, not named (the report names the first 8)" | **FIXED** | `..._failures_beyond_the_report_capacity_are_counted_not_mislabelled` |
| j7 | bound drain `note_release_failure` (`:1602`) | `first_device()` paired with `first_error()`: with the current-device query AND a device failing, "the first on device 1 (<the query's error>)" | `SyncReport::first()` (`:936`): one consistent record (the query, which ran first); the path ("in a bound drain") named | **FIXED** | `..._bound_drain_names_one_consistent_first_failure` |
| j8 | fallback sync `note_release_failure` (`:1594`) | "the first on device D (E)": no call named, a guard failure indistinguishable | tagged (the guard's call or cudaStreamSynchronize), the path ("the synchronous fallback of a call that could not queue its completion event") named | **FIXED** | `..._fallback_sync_failure_and_record_failure_are_named` |
| j9 | completion-event record (`~RetainUntilComplete`, `:1579`) | "recording ... (cudaEventRecord ...)" also when the guard's switch / read failed (no record attempted) | cudaEventRecord text unchanged for a record failure; a guard failure: "a call's completion event on device D was not recorded (...): switching to device D failed (cudaSetDevice: E)" | **FIXED** | `..._record_guard_failure_names_cudaSetDevice_not_cudaEventRecord`; `..._fallback_sync_failure_and_record_failure_are_named` |
| j10 | event destroy (`destroy_event_on_device`, `:1170`) | "cudaEventDestroy on device D" also when the guard failed (cudaEventDestroy never called) | guard failure: "a completion event of device D was not destroyed: switching to device D failed (cudaSetDevice: E)"; destroy failure unchanged; recorded before its slot entry is consumed | **FIXED** | `..._destroy_guard_failure_names_cudaSetDevice_not_cudaEventDestroy`, `..._event_destroy_failure_is_named`; FIX4 shim test |
| j11 | next-call drain query (`drain_completed`, `:1634`) | "cudaEventQuery of a pending entry's completion event in the next-call drain ... on device D (E)" | unchanged text (tagged record) | OK | FIX5 tests (`cudaEventQuery of a pending entry`); `..._interrupted_hook_still_releases...` |
| j12 | caller-device restore (`note_unraisable_device_restore_failure`, `:1481`; guards `cuda_guard.h:143`, `:185`) | "restoring the caller's CUDA device (cudaSetDevice) on device P (E)" | unchanged (P is the device being restored: the call's target) | OK | FIX4 shim tests |
| j13 | synchronize() prior-error slot (`report_message`, `:1709`) | named the error, not its call / device (unknown) | adds "the call that set it and its device are unknown: the slot records neither" | **FIXED** (wording) | `..._a_pending_error_says_its_call_and_device_are_unknown` |
| j14 | `RetainUntilComplete` pending error before the first launch (`:1530`) | as j13 | as j13 | **FIXED** (wording) | same test (`prior_launch`) |
| j15 | `acquire_event` (`:1359`) | "cannot create the completion event on CUDA device D: E" (no call; a guard failure looked like a creation failure) | names cudaEventCreateWithFlags, or the guard's cudaSetDevice / cudaGetDevice and "no event was created" | **FIXED** | `..._event_creation_guard_failure_names_cudaSetDevice` |
| j16 | launch checks after kernels (`cuda_guard.h:85` `check_launch_on`; conv.cu, lstm.cu, norms.cu, small_ops.cu, 9 sites) | "<kernel> failed: E" (cudaGetLastError may report an asynchronous error of earlier work: the kernel may not have failed); no device | "CUDA error at the launch check of <kernel> on CUDA device D: E (... the launch's own error, or an asynchronous error of earlier work on that device ...)" | **FIXED** | corrected FIX4 `test_fix4_failed_restore_while_unwinding...` (refused launch, shim) |
| j17 | checks after cuBLAS calls (`cuda_guard.h:95` `check_after_cublas_on`; conv.cu `:260`, lstm.cu `:251`, `:266`) | "cublasSgemmStridedBatched failed: E" after it RETURNED SUCCESS (cuBLAS 12.6 imports no libcudart symbol, so it never sets this runtime's error state) | "CUDA error found by the check after <fn> on CUDA device D, which returned CUBLAS_STATUS_SUCCESS: E (... an asynchronous error of earlier work ...)" | **FIXED** | `test_fix7_sweep_j_check_after_a_successful_cublas_call_does_not_blame_it` (LD_PRELOAD, genuine slot error) |
| j18 | runtime calls on the launch path (lstm.cu `:225-233`: cudaMemcpyAsync / cudaMemsetAsync) | "initial_h zero failed: E": the API and device not named | "... (cudaMemsetAsync on CUDA device D returned it)" | **FIXED** | strengthened FIX4 `test_fix4_check_cuda_raise_consumes_the_failed_call` |
| j19 | cuBLAS statuses (`raise_cublas_failure_on`, `cuda_guard.h:105`; handle settings in conv.cu / lstm.cu) | API named, device not | "(on CUDA device D)" / "(the new handle of CUDA device D)" | **FIXED** | strengthened FIX4 cuBLAS-setting tests |
| j20 | `build_info` (`conv.cu:308-312`) | cudaRuntimeGetVersion / cudaGetDevice / cublasGetProperty failures | no device involved | OK | reasoning |
| j21 | `DeviceGuard` raises (`cuda_guard.h:120`, `:123`, `:132`) | "cudaGetDevice failed", "cudaSetDevice(D) failed", restore of P after D | unchanged: right call, right devices | OK | FIX4 shim tests |
| j22 | add-on exit lines (`__init__.py:232-260`) | "synchronize() at exit failed", "exit hook failed (OverflowError...)" | sync failure (unchanged), the post-close retry, a post-close result failure, the unusable-knob line, the overrun line | OK / new | i-rows tests |
| j23 | add-on exit wait diagnostic (`dlpack_operand.cpp:292`) | count, seconds | unchanged | OK | FIX5 tests |
| j24 | add-on runner lines (`bindings.cpp:428`, `:446`; `exit_log_unreported` `dlpack_operand.cpp:1759`) | - | new: synchronize() run by the runner, the runner's wait, unreported failures | OK (new) | i1 / i23 tests |
| k1 | core imported-Storage drain record (`kernels.cu:289`) | "(cudaGetDevice / cudaSetDevice / cudaStreamSynchronize)" for any of the three | one text per call that failed; the Storage's device | **FIXED** | `test_fix7_sweep_j_core_drain_record_names_the_one_call_that_failed[fail_get, fail_set, fail_sync]` |
| k2 | core drain restore record (`:315`) | "restoring the caller's CUDA device ... (cudaSetDevice)" + the restored device | unchanged | OK | `test_fix5_sweep_d_core_drain_restore_failure_is_recorded` |
| k3 | core `CheckedDeviceGuard` (`:343-368`) | cudaGetDevice / cudaSetDevice(D) / restore texts | unchanged | OK | FIX4 fill guard tests |
| k4 | core fill_ pending error (`:624`) | named the error only | adds "the call that set it and its device are unknown" | **FIXED** (wording) | strengthened `test_fix4_fill_reports_a_pending_error_as_such...` |
| k5 | core fill_ launch check (`:641`) | baseline `cuda_check_last("fill_")`: "CUDA error at fill_: <string>" (no error name, kernel, device) | "CUDA error at fill_: Name: String (cudaGetLastError right after launching fill_kernel on CUDA device D: the launch's own error, or an asynchronous error of earlier work on that device)" | **FIXED** | `..._fill_launch_check_names_kernel_and_device_without_blaming_the_launch` |
| k6 | core label proof query (`:449`) | "cudaPointerGetAttributes failed" | adds element 0, the label and "the device of its allocation is unknown" | **FIXED** (wording) | `..._label_proof_query_failure_says_the_device_is_unknown` |
| k7 | core export fence (`bindings.cpp:448-486`) | names each failed step's call, the tensor's device, the restore | unchanged | OK | FIX4 fence tests |
| k8 | core `from_dlpack` / `empty` current-device query (`:1144`, `:1265`) | names cudaGetDevice | unchanged | OK | FIX4 tests |
| k9 | core R7 message (`unraisable_message`, `bindings.cpp:1596`) | "tensor_cuda.synchronize: N CUDA failure(s) ..., the first: <what> on device D (E)"; at exit logged as "tensor_cuda: at exit: tensor_cuda.synchronize: ..." (a call that did not happen) | "tensor_cuda: N CUDA failure(s) ... (reported once, by tensor_cuda.synchronize() or at interpreter exit), the first: <what> [CUDA device D: E]"; the hook adds "(logged by tensor_cuda's interpreter-exit hook)", the runner "(logged by the exit runner ...)" | **FIXED** | corrected `test_fix6_r7_...[fail]`; `..._interrupted_core_hook_still_reports...` |
| k10 | core exit lines and wait diagnostic (`__init__.py:257-287`; `bindings.cpp:1023`) | timed_out line, "exit hook failed" | unusable-knob line, post-close result failure, R7 line (k9) | OK / new | core i-row tests |
| k11 | core runner line (`bindings.cpp:1644`) | - | new | OK (new) | i14 test |

Summary (j): 35 rows (24 add-on, 11 core): **FIXED 21** (j2-j10, j13-j19, k1, k4, k5, k6, k9), **OK 14** (j1, j11, j12,
j20-j24, k2, k3, k7, k8, k10, k11). Every FIXED row has a test. Recording: every add-on record is a POD copy
(`FailureRecord`, 12 bytes) into fixed storage, counts saturate (`sat_inc` / `sat_add`); core records keep a static
string pointer, device and error under the existing mutex (no allocation); messages are built only when reported
(V6-03 order kept: persist, build, raise, then clear).

## FIX6's rows corrected (h8, g17, and the (g) / (h) rows that sweep (i) changes)

| FIX6 row | FIX6 said | Corrected (FIX7) |
|---|---|---|
| h8 core `import_release_impl` `PyGILState_Ensure` | "only on a thread that does not hold the GIL: core releases the GIL nowhere in engine code, so every P1-T Storage drop holds it and the call is reentrant" | Reason wrong, conclusion right (round-7 review). A native DLPack consumer (C / C++ code, possibly on a thread that never held the GIL) can drop an exported context, and with it the last reference of an imported Storage, without holding the GIL; `PyGILState_Ensure` there is not always reentrant. It is still safe: it is called after the admission mutex is released (no P1-T mutex is held while it waits for the GIL), and the DLPack Python specification puts acquiring the GIL on whoever runs Python in a deleter. Application finalizer locks are FIX6's corrected N6. |
| g17 add-on plan / run bindings | "20" | **18** plan / run bindings (9 ops x 2); with the two probe bindings of g18 the count is 20. All are still accounted for. |
| g13 core `tensor_cuda.empty`, `synchronize()`, `_dlpack_shutdown_hook()` | OK: "`operator.index(s)` in Python; none" (Python-only conversion) | Incomplete for the hook: its `float(knob)` was Python-only, but its exception path skipped the close (V7-01). **FIXED in FIX7**: the hook converts nothing; the native entry reads the knob at C level and always closes; the native runner covers exceptions that escape the hook (sweep (i) i14-i21). |
| g26 add-on Python wrappers incl. `_release_pending_at_exit` | OK: "they pass exact ints / floats to `_C`" | Same correction for the hook (`float(_RELEASE_EXIT_DIAG_S)` and its exception path). **FIXED in FIX7** (sweep (i) i1-i13). The other wrappers are unchanged (Python frames, no native frame below them). |
| g10 core `_C._dlpack_begin_shutdown(first_diag_s)` (raw) | FIXED: "raw CPython function, caller-owned args, C-level real" (TypeError for any non-real) | Still raw and C-level; FIX7 changes the VALUE rule (V7-01): an unreadable value selects the default and still closes and waits; only a malformed call is TypeError. |
| g21 add-on `_exit_close_admission(first_diag_s)` (raw) | FIXED: as g10 | As g10. |
| h2 add-on `close_admission_and_wait` exit wait | OK (exit only, documented contract) | The wait's mechanism was right; its REACHABILITY was not checked (round-7: PARTLY, V7-01). **FIXED in FIX7**: reached on every path that runs a statement of the hook, and through the runner otherwise. |
| h9 core `dlpack_begin_shutdown` exit wait | OK | As h2; plus the lifecycle-mutex lock failure no longer rethrows before the close (sweep (i) i19). |
| g14 core import release admission | OK (admitted, waited for at exit) | Unchanged mechanism; the exit wait it relies on is now reached on every path (h9). |

## Not a defect (with justification)

| # | Item | Why it is not a defect in our code | Citation / receipt |
|---|---|---|---|
| N1 | A hook that RETURNS without having closed is not rescued by the native runner | By design, not an omission: the hooks' code reaches the close on every path that runs one of their statements (sweep (i)), so a returning hook that did not close can only be a defect introduced in the hook itself; a backstop that always closed would hide exactly that defect (the registered G4 mutants (a) and (b) would then pass the round-7 probes). Exceptions escaping a hook before its close, the one class a Python function cannot cover itself, are covered | G4 table (mutants (a) / (b) killed by the probes); `bindings.cpp` runner comments |
| N2 | An application that removes the exit protection (`atexit.unregister(tc_signal._C._exit_hook_runner)`, `atexit._clear()`, `os._exit`) | Deliberate removal of a documented shutdown hook, as FIX6's N4 (the reviewer accepted it); only the name to unregister changed (U3 below) | Python 3.12 `atexit` docs (`atexit.unregister`; fetched: `artifacts/p1t_fix7/citations/python312_atexit.txt`); README "Interpreter exit" |
| N3 | A knob that can only be read by running its own Python code (`numpy.float32`, `Decimal`, `Fraction`, a str, objects with `__float__`) no longer sets the delay | Not a loss of function: the knob is a diagnostic delay only; reading such a value means running user code before the close, which is the V7-01 class itself (it can raise anything, return anything, or block). The default is used and stderr says so; float / int and their subclasses (numpy.float64, bool, IntEnum) are still read | order Required fix 1 ("Handle any conversion failure with a safe default"); CPython `PyFloat_AS_DOUBLE` / `PyLong_AsDouble` read subclasses without calling their methods |
| N4 | After a Ctrl-C, CPython prints "Exception ignored in atexit callback: ... KeyboardInterrupt" for the interrupted hook | That is CPython's documented handling of an exception escaping an atexit callback; the runner deliberately re-raises the hook's own exception instead of hiding it | Python 3.12 `atexit` docs ("If an exception is raised during execution of the exit handlers, a traceback is printed ... and the exception information is saved"; fetched: `artifacts/p1t_fix7/citations/python312_atexit.txt`) |

FIX6's N1-N9 stand as the round-7 review accepted them (N4 with the registration name of U3).

## User-visible changes (each justified)

- **U1. The native exit entries close and wait for ANY value of their diagnostic argument**
  (`tc_signal._C._exit_close_admission`, `tensor_cuda._C._dlpack_begin_shutdown`): a value that is not a float / int
  (subclasses included), or an int beyond the double range, now selects the default delay (5 s) instead of raising
  TypeError / OverflowError; the result reports `first_diag_s` (the delay used, NaN / negative -> 0, capped at a day)
  and `first_diag_default_used` (core also `others_in_flight_at_close`). A malformed call is still TypeError and
  closes nothing. Correct: the order requires it ("a diagnostic argument they cannot read ... must not prevent the
  close. It falls back to the default diagnostic delay, closes and waits"); the argument is diagnostics only, the
  close is a memory-safety guarantee.
- **U2. The hooks no longer convert the knob in Python** (`float(knob)` is gone): values that `float()` accepted only by
  running their own code (a str such as "7.5", numpy.float32, Decimal, Fraction, objects with `__float__`) select the
  default delay (stderr says so after the wait). Correct: see N3.
- **U3. atexit now calls a native runner with the hook** (`atexit.register(_C._exit_hook_runner,
  _release_pending_at_exit)`, core `_C._dlpack_exit_hook_runner`). `atexit.unregister(tc_signal._release_pending_at_exit)`
  therefore no longer finds it; the documented way is `atexit.unregister(tc_signal._C._exit_hook_runner)`. Correct: only
  a native callable can cover the hook's first bytecode (a Python `try` cannot), and the runner acts only when the hook
  raises (N1). The hook functions keep their names and can still be called directly.
- **U4. A BaseException that arrives during a hook (KeyboardInterrupt, SystemExit) no longer skips the close**: the hook
  closes and waits, then the exception propagates (atexit prints it). Correct: Python's try / finally semantics; FIX6
  skipped the close, which is V7-01's failure mode.
- **U5. When a hook raises before closing, the runner does the hook's work natively**: synchronize() (add-on; its
  failure written with C stdio), the close and wait, and the stderr line for any CUDA failure nobody reported (both
  packages). Correct: the guarantees "operands released at exit" and "a CUDA failure is never lost" (N1 / V6-03 / R7)
  would otherwise be skipped by an early exit (sweep (i) i23-i25).
- **U6. After a MemoryError from synchronize() at exit, the hook calls synchronize() once more, after the close**, so
  the CUDA failure V6-03 kept is logged (after the close nothing is released; late releases leak, as documented).
  Correct: V6-03's guarantee had no reader at exit.
- **U7. Messages name the failed call and its device** (sweep (j) rows j2-j19, k1-k9): synchronize()'s drain texts
  (event query, guard failures, spill device, unnamed count, bound drain, fallback, record / destroy guard failures,
  pending-error note), launch checks ("CUDA error at the launch check of <kernel> on CUDA device D: ..." replaces
  "<kernel> failed: ..."), the check after a successful cuBLAS call, runtime-call and cuBLAS-status suffixes, event
  creation, core drain records, fill_'s launch check and pending-error note, the label-proof query, and the core R7
  message ("tensor_cuda: N CUDA failure(s) ..." replaces "tensor_cuda.synchronize: ..."; the exit hook appends "(logged
  by tensor_cuda's interpreter-exit hook)"). Exception TYPES are unchanged. Correct: the order's (j) rule; each old text
  named an operation that succeeded or did not name the one that failed.
- **U8. New counters and attributes**: `pending_stats()` `exit_diag_default_used`, `exit_runner_closes`,
  `exit_runner_in_flight_at_close`; `_dlpack_release_stats()` `shutdown_diag_default_used`, `shutdown_runner_closes`,
  `shutdown_runner_in_flight_at_close`, `shutdown_lock_retries`, and (L1) `shutdown_lock_diagnostics`; (L1)
  `pending_stats()` `exit_wait_poll_diagnostics`, `exit_device_wait_diagnostics`; `_tc_signal._RELEASE_EXIT_DIAG_DEFAULT_S`,
  `_tensor_cuda._DLPACK_SHUTDOWN_DIAG_DEFAULT_S` (the knobs' initial values, so the Python knob and the native fallback
  are one constant), `_exit_hook_runner`, `_dlpack_exit_hook_runner`. Additive.
- **U9. Core's exit close retries a failed lifecycle-mutex lock instead of rethrowing it before closing, and that retry
  wait is diagnosed** (never observed with glibc's default mutex; shown with injected failures: on 429446d one failed
  lock made `_dlpack_begin_shutdown` raise `RuntimeError: Invalid argument` with admission left open, and at a real exit
  the process aborted, rc -6). Now it retries `kDlpackShutdownLockRetryMs` (10 ms) apart, counted in
  `shutdown_lock_retries`; lead send-back L1: once the retries of one lock attempt have lasted the first-diagnostic
  delay (the knob, clamped), a stderr line names what is waited for ("cannot take the DLPack release lifecycle lock to
  close DLPack release admission (<error>): N failed attempt(s) over S s; it retries every 10 ms and waits (it never
  returns early)"), then another each time that time doubles, counted in `shutdown_lock_diagnostics` (and in
  `shutdown_diagnostics` for the hook's close), and a last line says when the lock was taken; the same for the re-lock
  after each diagnostic of the timed wait (sweep (i) i19, addendum a5 / a6). Correct: returning early lets
  finalization overtake a running release (V3-01); waiting is the documented outcome for anything that blocks exit,
  and, like every exit wait (FIX3 V3-01), it is diagnosed, never silent.
- **U10. The add-on's exit waits are never silent either (lead send-back L1, addendum a2 / a3).** The exit wait's
  fallback when the lifecycle mutex or condition variable fails (FIX5: poll every 10 ms) now writes the same schedule of
  lines, saying that it polls and why (`pending_stats()` `exit_wait_poll_diagnostics`). While the native exit runner
  runs (around the hook and its own steps), every device sync of exit (the hook's `synchronize()`, its retry after a
  MemoryError, the runner's own `synchronize()`) is preceded by a polled wait on an event recorded on that device's
  legacy stream: a line after the knob's delay (read at C level from the hook's module; default if unreadable), then
  each time the time waited doubles, and a last "finished after" line (`exit_device_wait_diagnostics`). Each device
  sync at exit gains at most about one poll interval (1 ms, `tuning::kExitDevicePollMs`); the sync that follows keeps
  its own error and its V7-02 attribution; an ordinary `synchronize()` during the run is unchanged (one relaxed atomic
  load). Correct: the lead's L1 rule, every exit wait that can block is diagnosed (FIX3 V3-01's contract); before, a
  kernel that never finished kept exit waiting with nothing on stderr.

## Probe before/after table

Every run a fresh process under `flock -w 900 /tmp/forge-gpu.lock timeout <bound>`. nogpu = `CUDA_VISIBLE_DEVICES=`.
Round 7: `artifacts/p1t_fix7/run_rev7_probes.sh` (summaries `probe_runs/rev7_{before,after}/summary.txt`). The
probes that only read run unmodified from each arm's tree (before: `/tmp/p1t_fix7_before_tree`, byte-identical probe
copies; after: this worktree); the five that write into the reviewer's scratch / receipts run as path-only copies
(`make_rev7_copies.py`; diffs `probe_runs/rev7_*/path_copies/`). Host modules (`_p1t_rev6_trace`, `_p1t_rev7_core`) are
regenerated from each arm's sources; on the before arm the regenerated core slices equal the reviewer's pins 7/7.

| Probe (round 7) | How it ran | Before (429446d) | After (this build) | Result |
|---|---|---|---|---|
| `knob_exit_cpu.py` nogpu | unmodified, the arm's package | **rc 134** (`FATAL: exception not rethrown`) | **rc 0** (`DELETER finished` before `FINALIZING True`) | **FIXED** |
| `knob_exit_cpu.py` GPU visible | unmodified | **rc 134** (`exit hook failed (OverflowError: int too large to convert to float)`, core dumped) | **rc 0** (stderr: the knob "holds an int beyond the range of a C double; ... default ... 5 s") | **FIXED** |
| `knob_exit_cpu.py --control` nogpu / GPU | unmodified | rc 0 / rc 0 | rc 0 / rc 0 | PASS |
| `core_knob_exit_cpu.py` | unmodified; host module regenerated from the arm (path-only `build_core_host.py`) | **rc 134** | **rc 0** | **FIXED** |
| `core_knob_exit_cpu.py --control` | same | rc 0 | rc 0 | PASS |
| `core_knob_exit_gpu.py` | unmodified, real core, one GPU | **rc 1** (child aborted: `FATAL: exception not rethrown`) | **rc 0** (`PASS core ordinary exit quiescence`) | **FIXED** |
| `core_knob_exit_gpu.py --control` | unmodified | rc 0 | rc 0 | PASS |
| `query_diagnostic_cpu.py` | unmodified; trace module from the arm (path-only `build_cpu.sh`) | **rc 1** ("the legacy stream of device 0 could not be synchronized") | **rc 0** ("the completion event of a pending entry on device 0 could not be queried (cudaEventQuery: ...)") | **FIXED** |
| `query_diagnostic_gpu.py` | unmodified, real module | **rc 1** | **rc 0** | **FIXED** |
| `argument_census_cpu.py` | path-only copy (receipt path) | PASS 1067 checks, 0 calls | PASS 1067 checks, 0 calls | PASS |
| `run_cpu_checks.py` | path-only copy | 14/14 rc 0 | 14/14 rc 0 | PASS |
| `build_cpu.sh`, `build_core_host.py` | path-only copies | rc 0 / rc 0 (slices = pins) | rc 0 / rc 0 | PASS |
| `gpu_regressions.py fix56` | path-only copy (receipt path) | rc 1: 2 failed, 155 passed (both causes outside 429446d's product: the before tree lacked artifacts/p1t_rev5 (D2), and the FIX6 test's dangling capsule names (D6); rerun on the completed tree: 1 failed, 156 passed, the failure being D6's test defect) | **PASS 157 tests, zero skips** | PASS |
| `gpu_regressions.py semantics` | path-only copy | PASS 504, zero skips | **PASS 504, zero skips** | PASS |
| `gpu_regressions.py multidevice` | path-only copy | exit 3 | exit 3 | INCONCLUSIVE (one GPU) |
| FIX7 receipt, not a reviewer probe: `lock_failure_receipt.py` (sweep (i) i19; one injected `pthread_mutex_lock` failure in the core's exit close) | this worktree's shim and exit script, each arm's package | A: `RuntimeError: Invalid argument`, admission open; B (real exit, deleter in flight): **rc -6** (`DLPack exit hook failed (RuntimeError: Invalid argument)`, `FATAL: exception not rethrown`) | A: returns after one 10 ms retry, admission closed; B: **rc 0**, deleter finished before finalization | **FIXED** |

Earlier rounds, after arm (FIX6's runners retargeted by `sed` to this worktree and /tmp/p1t_fix7_*,
`artifacts/p1t_fix7/runners/`; logic unchanged; summaries `probe_runs/{rev6,rev5,rev4,rounds123}_after/summary.txt`):

| Round | Result |
|---|---|
| 6 (`run_rev6_probes.sh`) | 30 / 30 runs rc 0: run_cpu_checks 16 cases (incl. `diagnostic_oom_{query,sync,device}`, `sync_lock`, `operand_census`, regenerated `core_host_cpu` arithmetic / `casts_exit bool|index` and controls), `sync_lock_inversion` nogpu / GPU (`rescue=False`), `sync_late_release` (`waited_for_later=False`), `core_cast_exit_gpu bool|index` + controls |
| 5 (`run_rev5_probes.sh` cpu / buildcpu / gpu / rf) | `exit_gate full` nogpu / GPU + controls rc 0; `exit_gate gpu` + control rc 0 (the daemon's later producer call is refused after the close, V5-01's rule; PASS printed); `exit_gate trace` as written **rc 1** and `--control` **rc 1** (its child models the hook as `synchronize()` alone: FIX6 D7, unchanged); its exit-entry copy (hook = `synchronize(); close(5.0)`) **rc 0**; `query_error_cpu` / `_single_gpu` rc 0; `handle_insert_oom_cpu` / `fill_grid_cpu` as written rc 1 (they embed 1475fae code: arm-independent, as in FIX5 / FIX6), regenerated from this tree rc 0 / rc 0 (`creates=1 destroys=1`; the huge extent refused); `fill_grid_single_gpu` rc 0; `device_error_sweep_single_gpu` rc 0 (40 passed); `sync_existing_error`, `core_edges`, `numerics_and_streams`, `core_import_query_error` rc 0; `addon_lifetime` rc 0, 12/12, 0 incorrect; `core_exit_overrun` during rc 0 (3.3 s), bounded rc 0 (7.8 s); `release_failures` rc 0 (66 passed); `core_export_device`, `core_device_drain`, `cuda_errors_multidevice` exit 3 (one GPU) |
| 4 (`run_rev4_cpu_probes.sh`) | identical to FIX6's after arm: host_checks 20 / 20 rc 0 in both conditions (`rev4_host_checks_vs_fix6.txt`: every difference is the FIX7 pending-error clause, U7), fix2 ck_census 18 / 0 failures, fix3 accounting 17 / 0, no UBSan line, lifecycle_timeout control / short / default rc 0 (both conditions), operand_census PASS 17, real_metadata_plans PASS 129, `sync_slot_cpu` rc 1 (as FIX5 / FIX6: arm-independent), core_interop as written reproduces its embedded traces, its import regenerated from this tree refuses (rc 1), `core_import_query_error` rc 0 |
| 1-3 (`run_rounds123_probes.sh`) | `addon_lifetime` runs 1-3 rc 0, 12 / 12 correct, 0 incorrect each; `core_edges` 0; `core_shutdown --normal` 0, finalizing 0; `numerics_and_streams` 0; `host_checks` (round 1, round 2 nogpu / GPU) identical to FIX6's receipts after path normalization and the U7 clause (`round12_host_checks_vs_fix6.txt`); `core_device_drain` 3 (one GPU); round-3 `core_exit_overrun` during 0 (3.3 s), bounded 0 (7.7 s) |

## Gate table (first report, before L1; re-run on the L1 build in "Lead send-back L1")

| Gate | Requirement | Result | Receipt |
|---|---|---|---|
| G1 | `knob_exit_cpu.py` (no GPU, GPU), `core_knob_exit_cpu.py`, `core_knob_exit_gpu.py`, `query_diagnostic_cpu.py`, `query_diagnostic_gpu.py` fail on 429446d and pass on this build | **PASS**: before rc 134 / 134 / 134 / 1 / 1 / 1; after rc 0 on all six | `probe_runs/rev7_{before,after}/summary.txt` |
| G1 | the `--control` variants pass on both builds | **PASS**: rc 0 on both arms (knob_exit_cpu no GPU / GPU, core_knob_exit_cpu, core_knob_exit_gpu) | same |
| G1 | the new regression for each finding and each (i) value class passes | **PASS**: 22 knob classes x 2 packages (+4 no-GPU), every synchronize() exception kind, first-bytecode interrupt, `_C` lost, stderr gone / raising, V6-03 at exit, entry census, runner; V7-02 trace plans and real-module sites | final combined session (none of them in the failing set; `logs/p1t_fix7_g3_combined_session.log`), `logs/p1t_fix7_new_sweep_i_tests.log` |
| G2 | sweep tables (i) and (j) complete; every site OK or FIXED; every FIXED site has a test or a stated reason; FIX6's h8, g17 and every (g) / (h) row (i) changes corrected | **PASS**: (i) 26 rows, FIXED 18 / OK 8, every FIXED row tested; (j) 35 rows, FIXED 21 / OK 14, every FIXED row tested; corrected rows h8, g17, g13, g26, g10, g21, h2, h9, g14 | this report |
| G3 | every round-1..7 probe passes or is INCONCLUSIVE only for lack of 2 GPUs; round-5 `exit_gate trace` as in FIX6 | **PASS** (multidevice probes exit 3: one GPU; `exit_gate trace` as written rc 1 as FIX6 D7, its exit-entry copy rc 0; round-4 `sync_slot_cpu` / round-5 `handle_insert_oom_cpu`, `fill_grid_cpu` as written arm-independent as in FIX5 / FIX6, regenerated rc 0) | probe tables above |
| G3 | `argument_census_cpu.py` all checks; `run_cpu_checks.py` 14/14 | **PASS**: 1067 checks, 0 conversion calls; 14/14 rc 0 | `probe_runs/rev7_after/` |
| G3 | `gpu_regressions.py fix56` and `semantics` PASS with zero skips | **PASS**: 157 / 504, zero skips | `probe_runs/rev7_after/` |
| G3 | `addon_lifetime` 3/3 runs, 0 incorrect | **PASS**: 12 / 12 correct each run | `probe_runs/rounds123_after/` |
| G3 | `core_exit_overrun` during and bounded rc 0 | **PASS**: rc 0 (3.3 s) / rc 0 (7.7 s); round-5 copy rc 0 / rc 0 | `probe_runs/{rounds123,rev5}_after/` |
| G3 | Voice runner PASS 249/249 | **PASS**: `"verdict": "PASS"`, 249 PASS, 0 FAIL / SKIP / ERROR | `logs/p1t_fix7_g3_voice_preL1.log`, `artifacts/p1t_fix7/p1t_all_preL1.json` |
| G3 | combined session = exactly the 52 baseline ids | **PASS**: compare `G1 PASS`; baseline ids 52, now 52; 0 baseline ids no longer failing, 0 new failing ids; 37 failed, 2996 passed, 47 skipped (FIX6: 47), 15 errors, all of them the baseline ids | `logs/preL1_p1t_fix7_g3_combined_session.log`, `logs/preL1_p1t_fix7_g3_compare.txt` |
| G3 | real-torch DLPack interop passes | **PASS**: 22 real-torch tests (torch 2.11.0+cu130), 0 skipped, incl. `test_fix6_v602_real_torch_consumers_still_import_zero_copy`; Voice T1 PASS 11/11 | `logs/p1t_fix7_g3_torch_interop_preL1.log`, combined session, `logs/p1t_fix7_g3_voice_preL1.log` |
| G4 | mutants (a), (b), (c) each fail their new tests and their round-7 probe | **PASS**: all three KILLED (table below) | `artifacts/p1t_fix7/g4/` |
| G5 | the diff against 8035b3b touches only the Exercises paths + `docs/P1T_FIX7_*.md`, `artifacts/p1t_fix7/**`, `logs/**`; reviewer probe files byte-identical; no new core `__global__` | **PASS**: every changed path in scope (HEAD vs 8035b3b adds only the order); 0 new core `__global__`, device atomics, fast-math, TF32, product `cudaDeviceSynchronize`, host callbacks, `os._exit` / `abort` / `std::terminate`; no escaping throw in 32 listed noexcept / destructor bodies; tc_signal not linked to `_tensor_cuda`; 92 reviewer probe files byte-identical; `artifacts/p1t_rev*/` unchanged vs HEAD; `artifacts/p1t_rev7/` 41 / 41 files equal their pre-change hashes | `artifacts/p1t_fix7/g5_scope.txt` |

Voice's verdict line (first report: `logs/p1t_fix7_g3_voice_preL1.log`): `"verdict": "PASS"` (T1 11, T2 144, T3 28, T4 12, T5 28, T6 13, T1R 1, T4S 12: 249 PASS, 0 FAIL / SKIP / ERROR; `"failed_cases": []`; 43.1 s).

Compare output (first report: `logs/preL1_p1t_fix7_g3_compare.txt`):

```
G1 compare (APA-MF2.2): baseline = /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2/logs/apa_mf1_g1_baseline_unchanged.log; now = logs/p1t_fix7_g3_combined_session.log
summary: 37 failed, 2996 passed, 47 skipped, 2 warnings, 15 errors in 819.55s (0:13:39)
baseline ids 52 now ids 52
baseline ids that no longer fail: 0 []
new failing ids: 0 []
new MF2.2 test files with any FAILED/ERROR: 0
G1 PASS
```

## G4: mutant table

Each mutant is a copy of this worktree's product tree under `/tmp/p1t_fix7_mut_<m>/` with ONE asserted edit
(`artifacts/p1t_fix7/g4/make_mutants.py`, diffs `g4/mut_{a,b,c}.diff`), run by `g4/run_mutant.sh` (every run under the
GPU lock). Before each run the mutant's test files were checked byte-identical to this worktree's final ones and its
product sources identical except for the mutation (`summary3_*.txt`, first lines). (a) and (b) change Python only (the
copied .so files equal this build's, sha256 in `run3_*/binaries_*.txt`); (c) rebuilds tc_signal from the mutated source.

| Mutant | The one edit | New tests (this round's, unmodified) | Round-7 probe | Verdict |
|---|---|---|---|---|
| (a) add-on knob fallback narrowed back to `(TypeError, ValueError)`, close skipped on any other exception | `tc_signal/__init__.py` `_release_pending_at_exit` body replaced by 429446d's, verbatim (`float(knob)` with `except (TypeError, ValueError)`, the close inside the same outer `try`, `except Exception` logs and returns); the native runner registration kept | `test_fix7_findings.py -k v701`: **24 failed**, 14 passed (huge / negative-huge int, float / int subclasses, str, None, `__float__` raising Exception / KeyboardInterrupt / SystemExit / BaseException, non-float return, `__index__`-only, numpy float32, deleted; no-GPU variants; synchronize() raising MemoryError / KeyboardInterrupt / SystemExit / BaseException; `_C` lost; stderr raising KeyboardInterrupt; V6-03 at exit) | `knob_exit_cpu.py` no GPU **rc 134**, GPU **rc 134** (`exit hook failed (OverflowError: int too large to convert to float)`, `FATAL: exception not rethrown`) | **KILLED** |
| (b) the same in the core hook | `tensor_cuda/__init__.py` `_dlpack_shutdown_hook` body replaced by 429446d's, verbatim | `test_dlpack.py -k "fix7_v701 or fix7_core_knob"`: **16 failed**, 13 passed (the same knob classes, `_C` lost, stderr raising KeyboardInterrupt) | `core_knob_exit_cpu.py` **rc 134**; `core_knob_exit_gpu.py` **rc 1** (child `FATAL: exception not rethrown`, core dumped) | **KILLED** |
| (c) failure-kind tag dropped | `dlpack_operand.cpp` `SyncReport::add` stores every record as `{kStreamSync, kSyncStream, device, err}` | `test_fix7_findings.py -k "v702 or sweep_j"`: **13 failed**, 16 passed (query after a successful sync, sync + query, spill, guard switch / read, capacity, real-module `cuda:event_query`; each with and without message OOM) | `query_diagnostic_cpu.py` **rc 1**, `query_diagnostic_gpu.py` **rc 1** ("the legacy stream of device 0 could not be synchronized (cudaStreamSynchronize: ...)") | **KILLED** |

Receipts: `artifacts/p1t_fix7/g4/summary3_{a,b,c}.txt`, `run3_{a,b,c}/` (the final run, test files as of the end of
this round). Earlier runs `summary_*` (test files before the 11:56 strengthenings) and `summary2_*` (before the sweep (i)
follow-up tests of 12:20) gave the same verdicts (D11).

## Deviations (each with its reason)

- **D1. Path-only copies of five round-7 scripts** (`make_rev7_copies.py`: `build_cpu.sh`, `build_core_host.py`,
  `run_cpu_checks.py`, `argument_census_cpu.py`, `gpu_regressions.py`). Reason: they hard-code the reviewer's build dir /
  scratch and write receipts into `artifacts/p1t_rev7/`, which must stay byte-identical (G5). Every substitution is a
  literal asserted to occur once; logic, modes, bounds and PASS rules untouched; diffs in
  `probe_runs/rev7_{before,after}/path_copies/`. The G1 probes ran unmodified.
- **D2. The before arm ran from `/tmp/p1t_fix7_before_tree`**, byte-identical copies of wt_fix6's sources, tests and
  429446d binaries plus byte-identical reviewer probe copies (`make_before_tree.sh`, hashes in
  `probe_runs/rev7_before/`). Reason: wt_fix6 is read-only and has no `artifacts/p1t_rev7/`, and the probes find their
  package through `parents[3]` of their own path. The first tree lacked `artifacts/p1t_rev5/probes` (one before-arm
  fix56 failure, `test_fix5_v503_reviewer_cpu_probe_regenerated...`); every probe directory was then copied (92 files,
  byte-identical) and fix56 rerun on the completed tree: 1 failed, 156 passed, the remaining failure being D6's test
  defect (`probe_runs/rev7_before/gpu_regressions_fix56_rerun_completed_tree.log`).
- **D3. FIX6's earlier-round runners retargeted by `sed`** to this worktree and `/tmp/p1t_fix7_*`
  (`artifacts/p1t_fix7/runners/`; logic unchanged), with absolute output directories as ordered.
- **D4. Round-5 `exit_gate.py trace` as written fails on this build (rc 1, and its `--control` rc 1)**, as in FIX6 (its
  D7): its child models the exit hook as `synchronize()` alone; the exit-entry copy (hook = `synchronize(); close(5.0)`)
  passes. Handled exactly as the order says ("as in FIX6").
- **D5. Tests corrected in place** because they pinned behaviour this order changes (`artifacts/p1t_fix7/test_corrections.diff`):
  FIX6's `_exit_close_admission` typed census case moved to a subprocess (in-process it would close the pytest
  process's admission); FIX6's exit-keyword cases expect CLOSED with the default instead of REFUSED (U1, both packages);
  two in-process `_dlpack_begin_shutdown(<unreadable>)` census cases moved to a subprocess test; FIX6's R7 exit-log
  text (U7 / k9); FIX4's launch-check text (j16) and strengthened API / device suffix assertions; `fix5_regen.py` gains a
  `check_cublas_on` stub so the regenerated round-5 handle probe still compiles. No assertion was weakened: every
  change asserts the new, stricter text or the order's new rule.
- **D6. A defect in FIX6's own test fixed** (`test_fix6_exit_hook_still_waits_for_an_admitted_callback_on_another_thread`,
  `_EXIT_SCRIPT`): its capsule names were bytes literals of the `-c` module code, and `PyCapsule_New` keeps the name
  POINTER, which must outlive the capsule (CPython C-API "Capsules"); the capsules are read at exit after that code
  object was freed ("a capsule named '@JHu'"). Names are now module globals. Receipt `artifacts/p1t_fix7/capsule_name_receipt.txt`
  (429446d binaries: original script x3 not ordered, fixed script x3 ordered). The reviewer probes keep their literals
  alive (not affected).
- **D7. My own mistakes, fixed before acceptance:** the guard_get failure plan counted device 0's guard read
  (nth 2 -> 3); the stderr-'raises' staging left a broken `sys.stderr` for CPython's final flush (rc 120 from
  `Py_FinalizeEx`, not the hook: `_report` restores it); the core stderr staging had nothing to write (an unreadable
  knob now gives it a line); the core exit script registered `_report` after the import (it ran before the hook: moved
  before); a broken assert in the label-proof test; my first label-proof text dropped FIX4's "cudaPointerGetAttributes
  failed" prefix (restored); I briefly removed FIX5's catch-all in the core wait (it re-takes the GIL: restored); a
  `pytest.skip` that would have broken the zero-skip gate (replaced by a subprocess check). And one reporting mistake
  caught before the report: my sweep (i) draft said rows i19 and i21 had "no injection point"; both were injectable (an
  LD_PRELOAD `pthread_mutex_lock` interposer; a proxy `_C`), so tests were written (and i11's), and i19 turned out to
  abort 429446d at exit (rc -6) under one injected lock failure.
- **D8. The combined session ran detached** (`run_in_background`, the order's exact command, a completion marker
  `logs/p1t_fix7_g3_combined_session.done`), read only after the marker existed: it takes about 14 min at this load,
  beyond my tool's 10-minute foreground ceiling. It ran twice: the preliminary run (`*_prelim.*`, G1 PASS, 2991 passed)
  predates the three sweep (i) follow-up test functions (5 cases: 2991 + 5 = 2996); the final run, on the final test files (hashes
  `artifacts/p1t_fix7/test_files_sha256_at_combined_preL1.txt`) and the final binaries, was the first report's receipt
  (renamed `logs/preL1_p1t_fix7_g3_combined_session.*` / `logs/preL1_p1t_fix7_g3_compare.txt` when the L1 build's
  session ran; see "Lead send-back L1").
- **D9. Network:** the core build (`./build.sh 89`, as ordered) fetched pybind11 v2.12.0 through FetchContent, as
  FIX3-FIX6 did; tc_signal used production's pybind11 source (read-only), as ordered.
- **D10. Citations fetched through the AtlasForge web proxy** (Python 3.12 atexit docs, CPython 3.12.3 bytecodes.c
  excerpts: `artifacts/p1t_fix7/citations/`); the rest is marked "unverified, lead to check" with search terms.
- **D11. The mutants ran three times** (test files changed twice after the first run: the 11:56 strengthenings and the
  12:20 sweep (i) follow-up tests); verdicts identical each time; the third run is the receipt.
- **D12. Load:** the host ran at a load average of about 25 (other users) throughout. The combined session took 13-14
  min (FIX6's: 10.5 min); that reflects the load and this round's added tests (about 110 more, most of them
  subprocess exits that wait for a 0.6 s deleter), not a slower build.
- **D13. Voice's runner ran at 11:54**, after the final product build (core 11:49, tc_signal 11:45; no product source
  changed since: the binaries' sha256 equal the combined session's) and before the test-only edits, which Voice does not
  load.
- **D14. Product code unchanged after the first report's receipts** (until the lead's send-back L1, which changed it
  and re-ran the listed gates, below): the sweep (i) follow-up of 12:20 added tests only (and this
  report's corrections); the round-1..7 probe receipts, Voice and the mutants' product trees are therefore on the final
  product (the .so sha256 below equal those in `build_sha256_at_combined.txt` and in the mutant (a) / (b) trees).

## Diff stat (first report, before L1; the final one is in "Lead send-back L1")

```
 tc_signal/README.md                   |  49 +++-
 tc_signal/src/bindings.cpp            |  97 ++++++-
 tc_signal/src/conv.cu                 |  41 +--
 tc_signal/src/cuda_guard.h            |  61 ++++
 tc_signal/src/dlpack_operand.cpp      | 505 ++++++++++++++++++++++++++--------
 tc_signal/src/dlpack_operand.h        |  14 +
 tc_signal/src/lstm.cu                 |  66 ++---
 tc_signal/src/norms.cu                |  14 +-
 tc_signal/src/py_args.h               |  44 ++-
 tc_signal/src/small_ops.cu            |  18 +-
 tc_signal/src/tuning.h                |   6 +
 tc_signal/tc_signal/__init__.py       | 116 +++++---
 tc_signal/tests/fix5_regen.py         |  10 +-
 tc_signal/tests/test_fix4_findings.py |   8 +-
 tc_signal/tests/test_fix6_findings.py |  57 +++-
 tensor_cuda/src/bindings.cpp          | 244 +++++++++++++---
 tensor_cuda/src/kernels.cu            |  44 ++-
 tensor_cuda/tensor_cuda/__init__.py   |  92 +++++--
 tensor_cuda/tests/test_dlpack.py      | 485 +++++++++++++++++++++++++++++++-
 19 files changed, 1644 insertions(+), 327 deletions(-)
# new (untracked) files under tensor_cuda/ tc_signal/:
tc_signal/tests/test_fix7_findings.py | 972 lines, new
```

## Binaries (sha256; first report, before L1: `artifacts/p1t_fix7/build_sha256_at_combined_preL1.txt`; the final build is in "Lead send-back L1")

```
17bf6fd772f7515dfc13b5739eb3e2c475c5cc620dc906556c7e9ea0f2b18857  tensor_cuda/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
b46253594458d3dfc249addf0dcaa85902108d025194a0b7ec9dcfb277e995c8  tc_signal/tc_signal/_tc_signal.cpython-312-x86_64-linux-gnu.so
```

## Prior art

- **V7-01 / sweep (i)**: Python's `try` / `finally` guarantee, which holds for KeyboardInterrupt and every other
  exception raised inside the `try` (language reference, "The try statement"; taken: the close in a `finally`);
  CPython 3.12's eval-breaker placement (`RESUME` with oparg < 2, `JUMP_BACKWARD` and the `CALL` family check for pending
  signals: `Python/bytecodes.c` at v3.12.3, citation `artifacts/p1t_fix7/citations/cpython_3.12.3_bytecodes_eval_breaker.txt`),
  which is why a Python function cannot guard its own first instruction (taken: the reason for a native runner);
  CPython's `atexit` (an exception escaping a handler is printed and the remaining handlers still run;
  `artifacts/p1t_fix7/citations/python312_atexit.txt`; taken: a handler's failure must not cancel the next step) and
  `threading._shutdown` (an unconditional exit-time join; taken: quiescence that no configuration can switch off);
  CPython's C argument parser (`Python/getargs.c`: call-shape errors are TypeError before the body runs; taken: malformed
  calls still refused); `PyFloat_AS_DOUBLE` / `PyLong_AsDouble` (subclasses read without calling their methods).
  Ours: the native runner that acts only when the hook raised (so a hook that returns without closing stays visible to
  tests), and the rule that a diagnostic value can never gate the exit close (no prior art known to me for the exact
  rule).
- **V7-02 / sweep (j)**: `std::error_code` (C++11 `<system_error>`), which pairs a value with its category so the same
  number is never read in the wrong domain (taken: value + discriminating tag); tagged / discriminated records in
  general; the distinct return contracts of `cudaEventQuery` and `cudaStreamSynchronize` and the note that
  `cudaGetLastError` "may also return error codes from previous, asynchronous launches" (CUDA 12.6 Runtime API; taken: the
  attribution rules); PyTorch's `C10_CUDA_KERNEL_LAUNCH_CHECK` message ("CUDA kernel errors might be asynchronously
  reported at some other API call, so the stacktrace below might be incorrect", c10/cuda/CUDAException.h; taken: say the
  asynchronous possibility instead of blaming the last launch). Ours: the (call, purpose) tag on the add-on's
  fixed-capacity records and the per-call texts. Unverified here (lead to check): "CUDA 12.6 cudaGetLastError may also
  return error codes from previous asynchronous launches", "c10 CUDAException.h CUDA kernel errors might be
  asynchronously reported".
- **Allocation-free recording**: SQLite's OOM-testing discipline (an error must survive the failure of its own
  reporting; unverified here, lead to check "SQLite testing OOM anomaly"); saturating counters (as FIX3's V3-02 total).
- **Testing**: the round-7 reviewer's probes (scenarios reused and extended to the order's value census);
  `_thread.interrupt_main` (CPython; simulates SIGINT to deliver KeyboardInterrupt at a chosen boundary);
  LD_PRELOAD interposition with `dlsym(RTLD_NEXT)` (glibc ld.so; FIX1 / FIX4 shims), here also over
  `pthread_mutex_lock`, filtered by the caller's return address resolved with `dladdr` (the classic allocation
  interposers do the same; unverified here, lead to check "LD_PRELOAD interposer __builtin_return_address dladdr caller
  filter"); DeMillo, Lipton & Sayward (1978) mutation testing. Ours: the per-call failure plans of the trace module,
  the interrupted-hook staging, the targeted lock-failure plan and the proxy `_C` that raises after a real close.
- **Lead send-back L1 (every exit wait diagnosed)**: FIX3 V3-01's own diagnostic schedule (a line after the
  first-diagnostic delay, then each time the time waited doubles; taken as is) applied to the core's lock-retry loop,
  the add-on's polling fallback and the device waits of exit; that schedule follows the kernel's repeated stall warnings
  (RCU CPU stall warnings, the hung-task detector; unverified here, lead to check "RCU CPU stall warning repeated",
  "hung_task_timeout_secs"); event polling as a host-side progress check (cudaEventQuery's documented NotReady
  contract; unverified here, lead to check "cudaEventQuery cudaErrorNotReady"). Ours: confining the device-wait
  diagnostics to the exit runner's window, and reading the knob there at C level from the hook's module globals.
- **Test defect fixed (capsule names)**: the CPython C-API "Capsules" contract (the name string "must outlive the
  capsule"). Nothing new.
- These annotations are also at the code sites (`tc_signal/src/dlpack_operand.cpp` failure-record and exit blocks,
  `cuda_guard.h` attribution block, `py_args.h` `raw_exit_diag_arg`, `bindings.cpp` runner; core `bindings.cpp`
  `dlpack_begin_shutdown_impl`, `raw_exit_diag_arg`, runner, `unraisable_message`; `kernels.cu` drain and fill_ blocks;
  both hooks' docstrings) and in the ledger.

## Lead send-back L1

**L1 (lead send-back): my i19 fix added an exit retry loop that never wrote to stderr. FIXED.**

`lock_retrying` is in core `tensor_cuda/src/bindings.cpp`. In the first report it retried forever without writing anything, both before the
close and at the re-lock after each timed-wait diagnostic. With a lifecycle lock that keeps failing, exit hung
silently, which breaks FIX3 V3-01's contract that an exit wait is always diagnosed. The lead's finding is correct.

What changed (L1 product diff `artifacts/p1t_fix7/l1/l1_product.diff`, pre-L1 sources
`artifacts/p1t_fix7/preL1_sources/`: 8 files, 242 insertions, 31 deletions):

- **Core:** `tensor_cuda/src/bindings.cpp:939` is now `lock_retrying(lk, first, purpose, diagnostics)`.
  - The first line goes out once the retries of one lock attempt have lasted the hook's first-diagnostic delay (the
    caller's clamped `first_diag_s`). Another follows each time that time doubles.
  - Each line names the lock and its purpose, the error, the number of failed attempts and the seconds spent, and says
    the loop retries every `kDlpackShutdownLockRetryMs` (10 ms) and never returns early. A last line says when the lock
    was taken.
  - Lines are written with C stdio, with the GIL released and no allocation.
  - Each line is counted in `shutdown_lock_diagnostics` (new stats key) and in the call's `diagnostics`, which for the
    hook's close means `shutdown_diagnostics` too.
  - Both call sites use it: before `closing = true` (`:1011`) and at the re-lock (`:1033`). The close still happens
    the moment the lock is taken.
- **Add-on, from the sweep (i) addendum below:**
  - The polling fallback of the exit wait (`tc_signal/src/dlpack_operand.cpp:276` `wait_without_gil`) now uses the
    same schedule and says that it polls and why (`exit_wait_poll_diagnostics`).
  - The device syncs of exit are diagnosed. While the native runner runs (`tc_signal/src/bindings.cpp:418`, RAII
    window), `sync_device_legacy` (`dlpack_operand.cpp:1098`) first polls an event recorded on the legacy stream
    (`:1049` `wait_device_work_diagnosed`), on the same schedule and with a "finished after" line
    (`exit_device_wait_diagnostics`).
  - The delay for that device wait is the hook's knob, read at C level from the hook's module globals
    (`bindings.cpp:399`; `py_args.h:222` `raw_exit_diag_value`, which is the value rule factored out of
    `raw_exit_diag_arg`, unchanged).
  - New tunable: `tuning::kExitDevicePollMs = 1`.
  - The device wait uses only CUDA calls this file already makes, so the reviewers' host traces still link.
- **Docs:** both hook docstrings, the `_dlpack_begin_shutdown`, `_dlpack_release_stats` and `_exit_hook_runner`
  docstrings, and the README "Interpreter exit" section (every exit wait is diagnosed).
- **Tests** (all new; every one passes on the L1 build and the earlier cases are unchanged):
  - core `test_fix7_l1_core_lock_retry_wait_is_diagnosed_like_the_exit_wait[close, relock_after_diagnostic]`:
    - 15 injected `pthread_mutex_lock` failures outlast a 0.05 s delay, then the lock succeeds;
    - it asserts the stderr lines, `shutdown_lock_diagnostics` equal to the line count, the "took the lock" line, the
      hook's close and wait, the deleter finishing before finalization, and rc 0;
    - the shim gained `shim_lock_arm_after(k, n)`, so the second case lets the close's lock through and fails the
      re-lock after the first timed-wait line.
  - add-on `test_fix7_l1_exit_wait_polling_fallback_is_diagnosed`: the exit wait's lock fails once, it polls, and the
    lines are counted.
  - add-on `test_fix7_l1_device_wait_of_exit_is_diagnosed[diag, default]`: about 0.6 s of `torch.cuda._sleep` is
    queued at exit. With a 0.05 s delay the lines appear and are counted; with the default 5 s there are none; an
    in-run `synchronize()` never writes one.

### Sweep (i) addendum: every exit wait or retry loop that can block (lead send-back L1)

Method: every place where the exit path of either package waits or retries — the hooks, the native runners, the
native exit entries and everything they call — was checked. For each one the table says what it can block on and how
it is diagnosed (the FIX3 V3-01 contract: a stderr line after the first-diagnostic delay, then each time the time
waited doubles; C stdio; never silent; never an early return). Line numbers are for the L1 sources.

| # | Exit wait / retry loop (file:line) | Can block on | How it is diagnosed now | Verdict | Test / reason |
|---|---|---|---|---|---|
| a1 | add-on exit wait, condition variable (`tc_signal/src/dlpack_operand.cpp:285`, `wait_without_gil`) | producer callbacks admitted on other threads (a deleter that never returns) | a line after the knob's delay (from the start of the wait), then each time the time waited doubles; the mutex is released while writing; `exit_wait_diagnostics`, `exit_wait_overruns` | OK (since FIX5 V5-01) | FIX5 / FIX6 exit tests; FIX7's knob classes zero / negative / nan (diagnose at once) write its lines while the 0.6 s deleter runs |
| a2 | add-on exit wait, polling fallback after a lifecycle mutex / condition-variable failure (`:307`) | the same | was **silent** (FIX5 to the first FIX7 report). Now it continues the same schedule, and each line says it is polling every 10 ms because the lifecycle mutex or condition variable failed. Counted in `exit_wait_poll_diagnostics` and in `exit_wait_diagnostics` | **FIXED** | `test_fix7_l1_exit_wait_polling_fallback_is_diagnosed` (an LD_PRELOAD `pthread_mutex_lock` interposer fails the wait's lock once; deleter in flight; 0.05 s delay). Receipt `l1/receipt_{preL1,after}_addon_poll.txt`: 0 lines before, 3 after (4 in the test runs) |
| a3 | add-on device syncs of exit: the hook's step-1 `synchronize()` (`tc_signal/__init__.py:219`), its post-close retry (`:235`), the runner's own `synchronize()` (`tc_signal/src/bindings.cpp:426`); all of them reach `sync_device_legacy` → `cudaStreamSynchronize` (`dlpack_operand.cpp:1098`) | the work queued on each device's legacy stream (a kernel that never finishes) | was **silent**. Now, inside the runner's window (`bindings.cpp:418`; delay = the hook's knob read at C level, `:399`), each sync is preceded by a polled event wait (`dlpack_operand.cpp:1049`): a line after the delay, then each doubling, then "finished after". Counted in `exit_device_wait_diagnostics`. An ordinary `synchronize()` is unchanged | **FIXED** | `test_fix7_l1_device_wait_of_exit_is_diagnosed[diag, default]` (about 0.6 s of `torch.cuda._sleep` on the legacy stream, queued just before the runner; 0.05 s gives lines; the default 5 s gives none; an in-run `synchronize()` gives none). Receipt `l1/receipt_{preL1,after}_addon_dev.txt`: 0 lines before, 4 after |
| a4 | add-on: the pending releases that the exit `synchronize()` runs on the exiting thread itself (producer deleters) | the producer's own code | not a P1-T wait: the exiting thread is executing the producer's deleter, and no code of ours runs while it does. A deleter that never returns breaks the producer's DLPack contract (FIX6 N1; the DLPack spec says a deleter must return) | OK | reasoning |
| a5 | core: the lock-retry loop before `closing = true` (`tensor_cuda/src/bindings.cpp:939` `lock_retrying`, called at `:1011`) | a lifecycle mutex whose lock keeps failing | was **silent** (this round's first i19 fix). Now a line once the retries have lasted the delay (`first_diag_s` after the clamp), then each doubling, then one line when the lock is finally taken. Each line names the lock, the error, the number of failed attempts and the seconds spent, and says the loop retries every 10 ms and never returns early. Counted in `shutdown_lock_diagnostics` and, for the hook's close, in `shutdown_diagnostics`. GIL released, C stdio, no allocation (`what()` is the exception's own string) | **FIXED** | `test_fix7_l1_core_lock_retry_wait_is_diagnosed_like_the_exit_wait[close]` (15 failures, 0.05 s); mutant (d) is killed by it. Receipt `l1/receipt_{preL1,after}_core_lock.txt`: 0 lines before, 2 after, plus the "took the lock" line |
| a6 | core: the same loop at the re-lock after each diagnostic of the timed wait (`:1033`) | the same | the same function, purpose "again after a diagnostic (admission is already closed)" | **FIXED** | `test_fix7_l1_core_lock_retry_wait_is_diagnosed_like_the_exit_wait[relock_after_diagnostic]` (the close's lock is let through; the 15 failures hit the re-lock after the first timed-wait line) |
| a7 | core timed wait (`:1017`, `cv.wait_until`) | releases admitted on other threads | a line after the delay, then each doubling, then "finished after" (FIX3 V3-01); the mutex is released while writing; `shutdown_diagnostics`, `shutdown_timeouts` | OK (since FIX3) | FIX3+ exit tests; its lines appear in the a5 / a6 stagings |
| a8 | core: the runner's close (`:1640` → `dlpack_begin_shutdown_impl(..., runner=true)`) | a5 to a7 | the same code paths and lines; the lock lines are counted in `shutdown_lock_diagnostics` | OK (covered by a5 to a7) | i14 tests |
| a9 | core: `_dlpack_raise_unraisable` and the R7 record registry mutex (`kernels.cu:393`, `:405`, `:420`) at exit | the registry mutex | its holders only copy a record (no call, no Python, no wait while holding it), so the wait cannot be unbounded | OK | reasoning |
| a10 | core: imported-Storage drains at finalization (`kernels.cu:289` `drain_device_legacy_stream`, from `~Storage`) | device work on that Storage's device | not an exit wait: this is the release path of every imported Storage drop, the same during the run. The exit hook does not run it; CPython's finalization drops whatever objects remain | OK | reasoning (unchanged behaviour of every drop) |
| a11 | add-on `leave_count`'s lock (`dlpack_operand.cpp:206`, on the releasing thread) | the lifecycle mutex, briefly | not an exit wait: it runs on the releasing thread. A failure is counted (`lifecycle_lock_failures`), and the exit waiter's deadlines and polling still see the release leave | OK | FIX5 reasoning; a2 test |
| a12 | every stderr write of exit (all of the lines above) | a full stderr pipe | this is the diagnostic channel itself. Every wait releases its mutex before writing (FIX3 / FIX5), so a blocked stderr never stops a release or callback from leaving | OK | reasoning |

Summary (addendum): 12 rows. **FIXED 4** (a2, a3, a5, a6), each with a test; mutant (d) checks a5. **OK 8** (a1, a4, a7,
a8, a9, a10, a11, a12). Row a3 goes beyond the four waits the lead named: the sweep found it and it is fixed with a
test (L1-D1).

### L1 before / after

The pre-L1 arm is the first report's build: core `17bf6fd7...` from `/tmp/p1t_fix7_mut_a` and tc_signal `b4625359...`
from `/tmp/p1t_fix7_mut_b`. Each is the unmutated half of an earlier G4 tree, and each staging imports only that
package. The L1 arm is this worktree. Each staging is the test's own script, run by
`artifacts/p1t_fix7/l1_receipt.py`; receipts are in `artifacts/p1t_fix7/l1/receipt_{preL1,after}_*.txt`.

| Staging | Pre-L1 | L1 build |
|---|---|---|
| core: 15 failed lifecycle-mutex locks at the exit close, 0.05 s delay, deleter in flight | rc 0, deleter finished before finalization, **0** lock-retry lines (silent) | rc 0, deleter finished before finalization, **2** lock-retry lines ("6 failed attempt(s) over 0.1 s", "11 ... over 0.1 s") + "took the DLPack release lifecycle lock ... after 15 failed attempt(s) over 0.2 s; exit continues" |
| add-on: the exit wait's lifecycle mutex fails once (polling fallback), 0.05 s delay, deleter in flight | rc 0, deleter finished before finalization, **0** polling lines (silent) | rc 0, deleter finished before finalization, **3** polling lines (4 in the test runs), counted |
| add-on: about 0.6 s of GPU work queued at exit, 0.05 s delay | rc 0, **0** device-wait lines (silent) | rc 0, **4** device-wait lines + "finished after 0.5 s", counted (4); in-run `synchronize()`: 0 |
| i19 receipt, after arm (`lock_failure_receipt_afterL1_{A,B}.txt`) | (first report: 429446d rc -6) | A: returns after one 10 ms retry, admission closed, `shutdown_lock_retries` 1; B: **rc 0**, deleter finished before finalization |

### New binaries (sha256; `artifacts/p1t_fix7/build_sha256_at_combined.txt`)

```
750112413b78450937ba126e56ab700e3f9e16d1092f88af239b13fee48f9245  tensor_cuda/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
ca5770c152f6946921ce8c06681a2ecad6beaa81232b948cceac2a7a99605c37  tc_signal/tc_signal/_tc_signal.cpython-312-x86_64-linux-gnu.so
```

### Gate lines (L1 build; after arm only, as the lead specified)

| Gate | Result | Receipt |
|---|---|---|
| G1 round-7 probes | **PASS**: `knob_exit_cpu.py` no GPU rc 0, GPU rc 0; `core_knob_exit_cpu.py` rc 0; `core_knob_exit_gpu.py` rc 0 (`PASS core ordinary exit quiescence`); `query_diagnostic_cpu.py` rc 0; `query_diagnostic_gpu.py` rc 0 | `artifacts/p1t_fix7/probe_runs/rev7_afterL1/summary.txt` |
| G1 controls | **PASS**: every `--control` rc 0 (knob_exit_cpu no GPU / GPU, core_knob_exit_cpu, core_knob_exit_gpu) | same |
| G3 `run_cpu_checks.py` / `argument_census_cpu.py` | **PASS**: 14 / 14 rc 0; PASS 1067 argument checks, 29 add-on bindings, zero conversion calls | same |
| G3 `gpu_regressions.py fix56` / `semantics` | **PASS**: 157 passed, zero skips / 504 passed, zero skips (`multidevice` exit 3: one GPU) | same |
| G3 Voice runner | **PASS 249/249**: `"verdict": "PASS"`, T1 11, T2 144, T3 28, T4 12, T5 28, T6 13, T1R 1, T4S 12, 0 FAIL / SKIP / ERROR, `"failed_cases": []`, 26.4 s; adapter sha256 600f38b4... unchanged; Project-Voice `git status` clean before and after | `logs/p1t_fix7_g3_voice.log`, `artifacts/p1t_fix7/p1t_all.json` |
| G3 combined session | **PASS**: compare `G1 PASS`; baseline ids 52, now 52; 0 baseline ids no longer failing, 0 new failing ids; 37 failed, 3001 passed (2996 + the 5 L1 cases), 47 skipped (as before), 15 errors, all of them the baseline ids | `logs/p1t_fix7_g3_combined_session.log`, `logs/p1t_fix7_g3_compare.txt` |
| G3 real-torch interop | **PASS**: 22 passed, 0 skipped (torch 2.11.0+cu130), incl. `test_fix6_v602_real_torch_consumers_still_import_zero_copy` | `logs/p1t_fix7_g3_torch_interop.log` |
| G4 mutants (a)-(c) refreshed with the L1 build, plus (d) | **PASS**: all four KILLED (table below) | `artifacts/p1t_fix7/g4/summaryL1f_{a,b,c,d}.txt`, `runL1f_*/` |
| G5 scope | **PASS**: every changed path in scope; 0 new core `__global__`, device atomics, fast-math, TF32, product `cudaDeviceSynchronize`, host callbacks, `os._exit` / `abort` / `std::terminate`; no escaping throw in 35 listed noexcept / destructor bodies (the 3 L1 noexcept bodies added); tc_signal not linked to `_tensor_cuda`; 92 reviewer probe files byte-identical; `artifacts/p1t_rev*/` unchanged vs HEAD; `artifacts/p1t_rev7/` 41 / 41 files unchanged | `artifacts/p1t_fix7/g5_scope.txt` |
| L1 tests | **PASS**: 10 passed, 0 skipped (the 5 L1 cases and the 5 sweep (i) follow-up cases: i19 `[1]` / `[3]`, i21, i11 `[gpu]` / `[nogpu]`) | `logs/p1t_fix7_l1_tests.log` |

Voice's verdict line (`logs/p1t_fix7_g3_voice.log`): `"verdict": "PASS"` (249 PASS, 0 FAIL / SKIP / ERROR; `"failed_cases": []`).

Compare output (`logs/p1t_fix7_g3_compare.txt`):

```
G1 compare (APA-MF2.2): baseline = /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2/logs/apa_mf1_g1_baseline_unchanged.log; now = logs/p1t_fix7_g3_combined_session.log
summary: 37 failed, 3001 passed, 47 skipped, 2 warnings, 15 errors in 732.50s (0:12:12)
baseline ids 52 now ids 52
baseline ids that no longer fail: 0 []
new failing ids: 0 []
new MF2.2 test files with any FAILED/ERROR: 0
G1 PASS
```

### Mutant table (L1)

These are fresh trees `/tmp/p1t_fix7_mutL1_<m>`, made by `make_mutants.py <m> /tmp/p1t_fix7_mutL1_<m> _L1`, with diffs
in `g4/mut_<m>_L1.diff`. Each is a copy of the L1 tree, binaries included, with one asserted edit. Mutant (c) rebuilds
tc_signal, and mutant (d) rebuilds the core offline (L1-D3). Before each run, the tree's test files were checked
byte-identical to this worktree's final tests, and its product sources identical except for the mutation.

| Mutant | The one edit | New tests (unmodified) | Probe / receipt | Verdict |
|---|---|---|---|---|
| (a) add-on knob fallback narrowed back to `(TypeError, ValueError)`, close skipped on any other exception | `_release_pending_at_exit` body = 429446d's, verbatim | `-k v701`: **24 failed**, 14 passed | `knob_exit_cpu.py` no GPU **rc 134**, GPU **rc 134** | **KILLED** |
| (b) the same in the core hook | `_dlpack_shutdown_hook` body = 429446d's, verbatim | `-k "fix7_v701 or fix7_core_knob"`: **16 failed**, 13 passed | `core_knob_exit_cpu.py` **rc 134**; `core_knob_exit_gpu.py` **rc 1** | **KILLED** |
| (c) failure-kind tag dropped | `SyncReport::add` stores every record as a stream-sync failure | `-k "v702 or sweep_j"`: **13 failed**, 16 passed | `query_diagnostic_cpu.py` **rc 1**, `query_diagnostic_gpu.py` **rc 1** | **KILLED** |
| (d) (L1) the diagnostic removed from the core's lock-retry loop | `lock_retrying`: `if (now >= next) {` -> `if (false && now >= next) {` (the line, its counters and the doubling disabled) | `-k "fix7_l1_core or sweep_i_core_exit_close_retries"`: **2 failed** (`..._lock_retry_wait_is_diagnosed_like_the_exit_wait[close]` and `[relock_after_diagnostic]`: `assert lines`, only the timed wait's lines on stderr), 2 passed (the i19 retry tests `[1]`, `[3]`: the retry itself is intact) | no round-7 probe covers L1 (that probe set predates it); the L1 test is its teeth | **KILLED** |

The first L1 pass (`summaryL1_{a,b,c,d}.txt`, run before the re-lock case was added) gave the same verdicts; for (d)
it was 1 failed (the then single L1 test) and 2 passed.

### Deviations of the amendment (each with its reason)

- **L1-D1. Wider than the lead's list.** The addendum fixes two silent waits beyond `lock_retrying`.
  - The add-on's polling fallback (a2) was one the lead named.
  - The device syncs of exit (a3) were not. The sweep found them: a kernel that never finishes made the hook's
    `synchronize()` wait silently.
  - Fixing a3 adds product code: the runner's window and the diagnosed event wait. The cost is at most one 1 ms poll per
    device sync at exit; ordinary calls are unchanged.
  - Row a10 (core drains at finalization) is deliberately left OK. It is the release path of every imported-Storage
    drop, not a wait the exit path adds. That is stated so the lead can rule on it.
- **L1-D2. The pre-L1 arm for the L1 receipts** is assembled from the unmutated halves of the earlier G4 trees, whose
  binaries carry the first report's sha256s (core 17bf6fd7, tc_signal b4625359). Reason: the worktree was rebuilt and
  the old binaries exist only there. The pre-L1 sources are saved in `artifacts/p1t_fix7/preL1_sources/` (SHA256SUMS).
- **L1-D3. Mutant (d)'s core was built offline**, with build.sh's two cmake commands plus
  `FETCHCONTENT_SOURCE_DIR_PYBIND11` (production's pybind11 source, read-only) and `FETCHCONTENT_FULLY_DISCONNECTED`,
  in `/tmp/p1t_fix7_mutL1_d`. Reason: a fresh tree outside the worktree, with no network fetch. The worktree's own core
  build used `./build.sh 89` as ordered (incremental).
- **L1-D4. The mutants ran twice on the L1 trees.** The final pass (`summaryL1f_*`) is the receipt; the first
  (`summaryL1_*`) predates the re-lock test case. The verdicts are identical.
- **L1-D5. The report's tables were re-pointed.** The file:line references in the per-finding table and in sweep
  tables (i) and (j) now point at the L1 sources, mapped line by line from `preL1_sources` with difflib. FIX6's
  corrected rows and the first report's receipts keep their first-report paths. Receipts the L1 run replaced were
  renamed `*_preL1*` / `preL1_*`, not deleted.
- **L1-D6. Only the after arm was re-run** for the round-7 probes, as the lead specified. The first report's before
  arm stands.
- **L1-D7. The combined session ran detached again** (about 14 min, beyond my tool's 10-minute foreground limit). It
  used the order's exact command and a completion marker, and was read only after the marker existed.
- **L1-D8. Receipt plumbing.** `l1_receipt.py` and `lock_failure_receipt.py` run the tests' own scripts. Where the
  pre-L1 build lacks the L1 stats keys, they read the stats with `.get`; they change nothing else.

### Diff stat after L1 (`git diff --stat 429446d -- tensor_cuda tc_signal`, read-only git; `artifacts/p1t_fix7/diff_stat_L1.txt`)

```
 tc_signal/README.md                   |  60 +++-
 tc_signal/src/bindings.cpp            | 125 ++++++-
 tc_signal/src/conv.cu                 |  41 ++-
 tc_signal/src/cuda_guard.h            |  61 ++++
 tc_signal/src/dlpack_operand.cpp      | 622 +++++++++++++++++++++++++++-------
 tc_signal/src/dlpack_operand.h        |  18 +
 tc_signal/src/lstm.cu                 |  66 ++--
 tc_signal/src/norms.cu                |  14 +-
 tc_signal/src/py_args.h               |  56 ++-
 tc_signal/src/small_ops.cu            |  18 +-
 tc_signal/src/tuning.h                |  10 +
 tc_signal/tc_signal/__init__.py       | 120 +++++--
 tc_signal/tests/fix5_regen.py         |  10 +-
 tc_signal/tests/test_fix4_findings.py |   8 +-
 tc_signal/tests/test_fix6_findings.py |  57 +++-
 tensor_cuda/src/bindings.cpp          | 294 +++++++++++++---
 tensor_cuda/src/kernels.cu            |  44 ++-
 tensor_cuda/tensor_cuda/__init__.py   |  98 ++++--
 tensor_cuda/tests/test_dlpack.py      | 529 ++++++++++++++++++++++++++++-
 19 files changed, 1917 insertions(+), 334 deletions(-)
# new (untracked) files under tensor_cuda/ tc_signal/:
tc_signal/tests/test_fix7_findings.py | 1114 lines, new
```
