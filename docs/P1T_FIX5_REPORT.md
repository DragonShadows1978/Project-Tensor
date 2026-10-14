# P1T-FIX5 report: round-5 findings V5-01..V5-04 closed; sweeps of classes (c) exit callbacks, (d) consumed-but-unrecorded errors, (e) raw-resource exception safety, (f) launch-geometry narrowing

**Verdict: G1 PASS (with three probe-property items stated, D3/D4), G2 PASS, G3 PASS, G4 PASS, G5 PASS.**
- **V5-01 (MAJOR), interpreter exit overtook a producer deleter on another thread.** tc_signal now has a release
  lifecycle (`src/dlpack_operand.cpp`): admission, a closing flag, in-flight counts and a per-thread self-exclusion
  depth, independent of the pending list. Every producer deleter call goes through it (immediate refusal, next-call
  drain, synchronize, bound drain, fallback), and so do the two other add-on callbacks the class sweep found: the drop
  of the last reference to an unconsumed capsule (its destructor calls the producer's deleter) and the producer's
  `__dlpack__` / `__dlpack_device__` calls. The atexit hook runs `synchronize()`, then closes admission and waits, GIL
  released and with no time bound (stderr line after 5 s, then each time the wait doubles), until no admitted section
  of another thread is running. After the close, deleters are not called (leaked, counted), capsule references are
  kept (counted) and producer callbacks are refused (RuntimeError, counted): nothing enters Python.
  `synchronize()` also waits (without closing) for releases already running on other threads.
  - `exit_gate.py full` (no GPU and GPU visible), `trace` and `gpu`: rc 1 on 1475fae (child abort), rc 0 here.
  - Census: 0.1 / 5.5 / 12 s deleters on the immediate and deferred paths, ordinary exit, diagnostics at 5 s and 10 s:
    all pass. Mutant (a) fails 10 new tests and all four exit probes (rc -6).
- **V5-02 (MINOR).** The drain records a non-NotReady query failure before consuming the slot; the next `synchronize()`
  raises it once. `query_error_cpu.py` and `query_error_single_gpu.py`: rc 1 → 0. Mutant (b) fails 3 tests and both
  probes. FIX4 rows 47/50 corrected.
- **V5-03 (MINOR).** A scope guard owns a new cuBLAS handle until the cache insertion commits, in both caches. A real
  `operator new` failure at the insertion: creates 1 = destroys 1, next call works. Regenerated reviewer probe rc 1 → 0.
  Mutant (c) fails 3 tests.
- **V5-04 (MINOR).** `fill_` computes its block count in int64 and refuses (ValueError) beyond the CUDA grid x limit
  2^31-1 before any launch and before the revision bump. `fill_grid_single_gpu.py` rc 1 → 0. Mutant (d) fails 2 tests
  and the probe.
- **Sweeps (G2).** (c) 13 paths, FIXED 6 (including one in core: core's exit wait re-took the GIL in a destructor and
  aborted a non-main waiter, 2 of 5 runs on 1475fae, `terminate called without an active exception`), OK 7, 10
  catch-alls listed; (d) 38 rows, FIXED 5 (V5-02, a failed event record, three core destructor-path sites), OK 33;
  (e) 15 rows, FIXED 2 (V5-03), OK 13; (f) 16 rows, FIXED 1 (V5-04), OK 15. Every FIXED row has a test.
- **G3.** Every round-1..5 probe passes or is INCONCLUSIVE for lack of 2 GPUs; `addon_lifetime` 3/3 runs, 0 incorrect;
  `core_exit_overrun` during / bounded rc 0; Voice T1-T6 + T1R + T4S PASS 249/249; combined session = the 52 baseline
  ids (`G1 PASS`).
- **G5.** Diff only in the Exercises paths + FIX5 docs / artifacts / logs; probe files byte-identical (registered
  aggregate hash); no new core `__global__`; tc_signal not linked to core.

Seat: Opus 5.5 (high). Order: `orders/P1T_FIX5_ADDON_EXIT_AND_CLASSES.md` (3491eeb). Ledger: `docs/P1T_FIX5_LEDGER.md`.
Receipts: `artifacts/p1t_fix5/` and `logs/p1t_fix5_*`. "Before" = wt_fix4's prebuilt 1475fae binaries (read-only):
core `1da017ca2ec917019acc1ae9220fca87a74594257c053050056ed55eb2cd9b48`, tc_signal
`909af0c5e0c76cfd16c4016abb72b35bdde5a01cc96505334c8cc36876426352`.

## Per-finding table

| Finding | Fix (file:line) | Proving tests (new) | Probe before (1475fae) | Probe after (this build) |
|---|---|---|---|---|
| **V5-01** (MAJOR): interpreter exit overtook an add-on producer deleter running on another thread | `tc_signal/src/dlpack_operand.cpp`:<br>• `:78` design + prior-art block (release lifecycle)<br>• `:205` `admit` (seq_cst "increment, then read closing"; leak/refuse after close or while finalizing), `:218` `leave`<br>• `:256` `wait_without_gil`, `:291` `wait_for_other_threads` (PyEval_SaveThread, unbounded cv wait, doubling stderr diagnostics, explicit PyEval_RestoreThread in a non-noexcept frame; self-exclusion by thread-local depth)<br>• `:515`/`:518` `call_deleter_contained`: every deleter call admitted (immediate refusal, next-call drain, synchronize, bound drain, fallback)<br>• `:311` `CapsuleRef::drop`: last-reference drop of an unconsumed capsule admitted (capsule-destructor path)<br>• `:339`/`:343` `to_capsule`: producer `__dlpack__` callbacks admitted (`CallbackSection`)<br>• `:1075` `close_admission_and_wait` (exit), `:1285` synchronize()'s non-closing wait<br>`tc_signal/src/dlpack_operand.h` (CapsuleRef, docs); `tc_signal/src/{bindings,lstm_bindings,m4_bindings}.cpp` (CapsuleRef call sites; `bindings.cpp:411` `_exit_close_admission`); `tc_signal/tc_signal/__init__.py:151` `_RELEASE_EXIT_DIAG_S`, `:154` hook (synchronize, then close + wait); `tuning.h` (3 knobs) | `tc_signal/tests/test_fix5_findings.py`:<br>• `:255` immediate-refusal deleter in flight at exit, 0.1 / 5.5 / 12 s<br>• `:264` deferred release in flight at exit, 0.1 / 5.5 / 12 s<br>• `:272` immediate path with no visible GPU<br>• `:277` capsule destructor in flight at exit<br>• `:285` producer `__dlpack__` in flight at exit; late capsule drop kept<br>• `:344` releases / callbacks after close: leaked or refused, counted, no Python<br>• `:420`/`:432` hook inside its own deleter / its own producer callback: no self-wait<br>• `:552`/`:561` throwing C++ deleter contained once (main thread; in flight at exit)<br>• `:616` synchronize() waits for another thread's release<br>• `:250` no `os._exit` in the census script | `exit_gate.py full` nogpu **rc 1** (child abort), gpu **rc 1** `FATAL: exception not rethrown`; `trace` **rc 1** (child -6); `gpu` **rc 1** (`FATAL`); `--control` of full / gpu rc 0 | `exit_gate.py full` nogpu **rc 0**, gpu **rc 0**; `trace` **rc 0**; `gpu` **rc 0** (each `DELETER finished` before `FINALIZING True`); `--control` rc 0 |
| **V5-02** (MINOR): a one-shot event-query failure in the drain was consumed and never reported | `tc_signal/src/dlpack_operand.cpp:1242`: `drain_completed` records a non-NotReady failure (`event_query_failures` + `note_unraisable_failure`) BEFORE consuming the slot; the next `synchronize()` raises it once (FIX4's deferred routing) | `test_fix5_findings.py:730` one pending / one pending then drained / three completed pending (raised once, all released, none leaked, next sync succeeds); `test_fix4_findings.py` query test corrected (D2) | `query_error_cpu.py` **rc 1** (`synchronize returned success`); `query_error_single_gpu.py` **rc 1** (`one-shot query error disappeared`) | `query_error_cpu.py` **rc 0** (`PASS reported … cudaEventQuery … cudaErrorInvalidValue`); `query_error_single_gpu.py` **rc 0** (`PASS: query error retained and reported once`) |
| **V5-03** (MINOR): a new cuBLAS handle leaked when the cache insertion threw | `tc_signal/src/conv.cu:101` and `tc_signal/src/lstm.cu:119` `NewHandleOwner` scope guard from `cublasCreate` until `handles.emplace` (`:125` / `:143`) commits; settle clears `h` after its own destroy; the guard checks the destroy status and counts a failure (`note_cublas_handle_destroy_failure`, `cuda_guard.h`, `dlpack_operand.cpp:1069`) without throwing | `test_fix5_findings.py:877` real `operator new` failure at each cache's insertion (LD_PRELOAD C++ shim): MemoryError, creates 1 = destroys 1, next call works; `:889` reviewer CPU probe regenerated from these sources | `handle_insert_oom_cpu.cpp` as written **rc 1** (creates 1, destroys 0, both); regenerated from 1475fae (byte-identical) **rc 1** | as written **rc 1** (it embeds 1475fae's two functions: arm-independent, D3); regenerated from this tree **rc 0** (`conv creates=1 destroys=1`, `lstm creates=1 destroys=1`) |
| **V5-04** (MINOR): `fill_` narrowed its block count | `tensor_cuda/src/kernels.cu:533` `kFillMaxGridX = 2^31-1` (CUDA 12.6 grid x limit), `:535` `fill_grid_blocks` (int64 ceil; `std::invalid_argument` → ValueError beyond the limit), `:555` called first in `fill_` (before the label proof, guard, launch, revision bump), `:566` launch with the checked count; `bindings.cpp` `fill_` docstring | `tensor_cuda/tests/test_dlpack.py:1911` largest accepted extent launches grid 2^31-1 x 256 (exact cover, trapped); `:1922` one more and 2^40+1 refused before any launch, revision unchanged; `:1931` zero size; `:1936` small extent unchanged + device `cudaDevAttrMaxGridDimX` = 2147483647 | `fill_grid_single_gpu.py` **rc 1** (`oversized fill was not refused`); `fill_grid_cpu.cpp` as written and regenerated (identical) **rc 1** (`grid=1 covered=256`) | `fill_grid_single_gpu.py` **rc 0** (`REFUSAL … 4294967297 blocks … launches 0`); `fill_grid_cpu.cpp` as written **rc 1** (baseline `nblk`, unchanged by order); regenerated from this tree **rc 0** (2^40+1 and 2^40+256 refused) |

## G2: the four sweep tables

Scope: `git diff 9df5bef -- tensor_cuda tc_signal` on the final tree (every P1-T line, FIX1-FIX5), product code; test
code listed at the end of each table. Line numbers are the final tree's. Verdicts: **OK** (already correct; reason
given) or **FIXED** (changed by FIX5; test, or a stated reason why none is feasible). Evidence class per row: test
(named), probe (named) or reasoning (source read).

### (c) Callbacks during interpreter exit

Every path in the add-on and in core's P1-T surface that can invoke a producer deleter or any other callback that may
take the GIL, and every GIL re-acquisition on those paths.

| # | Site (file:line) | Callback / GIL take | Reached from | Admitted (waited for at exit) or leaks without Python? | Verdict | Test / reason |
|---|---|---|---|---|---|---|
| c1 | `tcs/dlpack_operand.cpp:515` `call_deleter_contained` (via `ManagedOwner::reset`, `:532`) | producer deleter | (a) immediate release of a refused call while unwinding (`~ManagedOwner` in `Operand`'s constructor, `~Operand`); (b) next-call drain `drain_completed` → `release_entry`; (c) `synchronize()` / the exit hook's synchronize → `sync_and_release_all_impl` → `release_entry`; (d) bound drain in `~RetainUntilComplete`; (e) fallback sync in `~RetainUntilComplete`; (f) `ManagedOwner` move-assignment (`reset()` of the target; empty at every call site) | admitted (kRelease); after close / while finalizing: deleter NOT called, manager leaked, counted (`deleters_leaked_after_close` / `_finalizing`) | **FIXED** (V5-01) | `test_fix5_v501_immediate_refusal_*` (0.1/5.5/12 s), `_deferred_release_*` (0.1/5.5/12 s), `_releases_and_callbacks_after_close_*`, `_throwing_cpp_deleter_*`; probes `exit_gate.py full/trace/gpu` |
| c2 | `tcs/dlpack_operand.cpp:311` `CapsuleRef::drop` | the producer's PyCapsule destructor (calls its deleter for an unconsumed capsule) | the last reference to a capsule tc_signal acquired: `Peeked`/`Peek4` members of all 10 plans (`bindings.cpp`, `lstm_bindings.cpp`, `m4_bindings.cpp`) when a plan refuses after acquiring; the `to_capsule(...)` temporaries of every `*_run` and `divide_probe` | admitted (kRelease) when it is the last reference of an unconsumed `dltensor*` capsule; after close / finalizing: the reference is kept, counted (`capsules_kept_after_close` / `_finalizing`); any other drop is a plain decref (no destructor that releases) | **FIXED** (sweep c; 1475fae dropped these in `~py::object`, unadmitted) | `test_fix5_v501_capsule_destructor_in_flight_at_exit`, `_producer_dlpack_callback_in_flight_at_exit_and_late_capsule_drop_is_kept`, `_exit_hook_inside_its_own_producer_callback_*` (2 kept) |
| c3 | `tcs/dlpack_operand.cpp:339`-`:381` `to_capsule` (`CallbackSection`, `:343`) | the producer's attribute lookups, `__dlpack_device__`, `__dlpack__` (Python code; may drop and re-take the GIL) | every plan and run that receives a producer object | admitted (kCallback); after close: refused before any callback (RuntimeError), counted (`callbacks_refused_after_close`) | **FIXED** (sweep c) | `_producer_dlpack_callback_in_flight_at_exit_*`, `_releases_and_callbacks_after_close_*` (`__dlpack__` never called), `_exit_hook_inside_its_own_producer_callback_*` |
| c4 | `tcs/dlpack_operand.cpp:291` `wait_for_other_threads` (`synchronize()` `:1285`; exit `:1075`) | GIL re-acquisition (`PyEval_RestoreThread`) after the wait | `tc_signal.synchronize()` on any thread; the exit hook | not a callback: explicit restore in non-noexcept frames up to pybind11's dispatcher, which rethrows `abi::__forced_unwind` (pybind11.h:1037), so a daemon thread exited there unwinds cleanly; the mutex is dropped first | **FIXED** (new code; designed so) | `test_fix5_v501_synchronize_waits_for_a_release_running_on_another_thread`; the census tests |
| c5 | `tcs/tc_signal/__init__.py:154` `_release_pending_at_exit` | the exit hook itself (Python) | atexit (main thread), or a user call | runs synchronize, then `_exit_close_admission`; self-exclusion for its own callbacks | **FIXED** (V5-01) | `_exit_hook_inside_its_own_deleter_*`, `_inside_its_own_producer_callback_*` |
| c6 | other `py::object` decrefs in add-on C++ frames (pybind11 argument holders, tuples returned by plans) | none that runs producer code: arguments are referenced by the caller for the whole call; tuples are returned; every producer-created object is held by a `CapsuleRef` or inside `to_capsule`'s admitted section | all entry points | n/a | **OK** | reasoning (source read of the 10 plans, 10 runs, `divide_probe`) |
| c7 | `core/bindings.cpp:739` `import_release_impl` (`PyGILState_Ensure` + producer deleter) | producer deleter, GIL take | imported Storage destructor (`kernels.cu` `Storage::~Storage`, any thread) | admitted by core's `ImportReleaseLifecycle`; after close / finalizing: leaked without the GIL (FIX2 N5 / FIX3 V3-01) | **OK** (unchanged) | `test_fix2_n5_*`, `test_fix3_v301_*` (pass); probe `core_exit_overrun.py` during / bounded rc 0 |
| c8 | `core/bindings.cpp:358`/`:366` `capsule_destructor_*` → `export_deleter_*` | DLPack deleter of a core export: `delete ctx` → `Storage` release | GC of an unconsumed core-exported capsule (GIL held by CPython's dealloc) | the only Python-reaching continuation is an imported Storage's release = c7 (admitted); engine memory is freed with CUDA calls only | **OK** | reasoning; `test_fix1_f06_*` (ownership) |
| c9 | `core/bindings.cpp:349`/`:352` `export_deleter_*` called by a consumer (tc_signal's admitted release, torch) | as c8 | any thread | as c8 | **OK** | reasoning |
| c10 | `core/bindings.cpp:823`-`:854` `dlpack_begin_shutdown` | GIL re-acquisition after core's exit wait | core's atexit hook; any thread calling `tensor_cuda._dlpack_shutdown_hook()` | 1475fae re-took the GIL in `~gil_scoped_release` (a destructor): a non-main waiter that CPython exits there terminates (2 of 5 runs on 1475fae: `terminate called without an active exception`, rc -6). Now explicit restore in this non-noexcept frame | **FIXED** (sweep c) | `test_fix5_sweep_c_core_exit_wait_on_a_non_main_thread_never_aborts[0-4]` (5/5 rc 0; 1475fae 2/5 rc -6) |
| c11 | `core/bindings.cpp:479` `tensor_dlpack` argument casts (`stream`, `max_version`, `dl_device`, `copy`) | a user object's `__index__` / `__bool__` / sequence methods (Python) | `Tensor.__dlpack__` | not a release; ordinary frames with no catch-all and no noexcept: a forced unwind propagates to pybind11's dispatcher (rethrow). Not admitted (residual R3) | **OK** | reasoning |
| c12 | `core/tensor_cuda/__init__.py` `from_dlpack` | `obj.__dlpack__` | Python frames only (no C++ frame on the stack) | n/a | **OK** | reasoning |
| c13 | `core/kernels.cu` `Storage::~Storage` (external) | → c7 after the drain | any thread | as c7 | **OK** | FIX1/FIX2 tests |

`catch (...)` on those paths, and what each does with a forced unwind (`abi::__forced_unwind`):

| # | Site | What it does with a forced unwind | Can an interpreter-made forced unwind reach it after FIX5? |
|---|---|---|---|
| k1 | `tcs/dlpack_operand.cpp:521` `call_deleter_contained` `catch (...)` around the deleter | swallows it, so glibc aborts ("FATAL: exception not rethrown"); rethrowing would meet `reset()`'s noexcept (std::terminate) | No: an admitted deleter is waited for before finalization, and after close none is called. Remaining origin: the producer's own `pthread_exit` / cancellation inside its deleter (DLPack contract violation; residual R2) |
| k2 | `tcs/dlpack_operand.cpp` `recycle_event` `catch (...)` | would swallow | No: no Python and no GIL take inside its try block |
| k3 | `tcs/dlpack_operand.cpp` `~RetainUntilComplete` pending push `catch (...)` | would swallow | No: no Python inside its try block |
| k4 | `tcs/dlpack_operand.cpp` `leave_count` / `wait_without_gil` `catch (const std::system_error&)` | typed: does not catch it | No Python inside |
| k5 | `tcs/dlpack_operand.cpp` `to_capsule` `catch (py::cast_error&)`, `catch (py::error_already_set&)` | typed: it propagates | n/a (it propagates) |
| k6 | pybind11 v2.12 dispatcher (`pybind11.h:1037`) | `catch (abi::__forced_unwind&) { throw; }` before its catch-all | rethrows |
| k7 | `core/bindings.cpp` `import_release_impl` three `catch (...)` (lock, deleter, leave) | lock / leave: no Python inside; deleter: as k1 | No (core admission, as k1) |
| k8 | `core/bindings.cpp:849` `dlpack_begin_shutdown` `catch (...) { PyEval_RestoreThread; throw; }` (FIX5) | rethrows | No Python in its try block |
| k9 | `core/kernels.cu:249` `note_core_unraisable` `catch (const std::exception&)` | typed | No Python |
| k10 | `core/bindings.cpp` `tensor_empty` `catch (const std::runtime_error&)` + rethrow | typed | n/a |

Summary (c): 13 paths: **FIXED 6** (c1, c2, c3, c4, c5, c10), **OK 7**; 10 handlers listed (k1-k10), none can receive
an interpreter-made forced unwind after FIX5. Test code: the FIX5 test shims/producers (ctypes deleters, the native
throwers) are the callbacks under test.

### (d) Consumed but unrecorded errors

Every site that consumes the calling thread's CUDA error slot (`cudaGetLastError`, or `cudaPeekAtLastError` followed by
it). "Raise" = the consuming call reports the error itself.

| # | Site | Function | Disposition | Verdict | Test / reason |
|---|---|---|---|---|---|
| d1 | `tcs/cuda_guard.h:50` | `raise_cuda_failure` | consume + raise | OK | FIX4 `guard` tests |
| d2 | `tcs/cuda_guard.h:87` | `~DeviceGuard` restore | consume + recorded (`note_unraisable_device_restore_failure`: first named, counted, raised once by next synchronize) | OK (FIX4) | FIX4 `launch_and_restore_fail` |
| d3 | `tcs/cuda_guard.h:107` | `QuietDeviceGuard` cudaGetDevice | consume; the error is returned by `error()` to every caller, each of which records or raises it (d6, d7, d11, d13) | OK | FIX4 `fix1` |
| d4 | `tcs/cuda_guard.h:114` | `QuietDeviceGuard` cudaSetDevice | as d3 | OK | FIX4 `fix1` |
| d5 | `tcs/cuda_guard.h:125` | `~QuietDeviceGuard` restore | consume + recorded (as d2) | OK (FIX4) | FIX4 `quiet` |
| d6 | `tcs/dlpack_operand.cpp:787` | `sync_device_legacy` | consume; `r` returned → recorded by every caller (`DeviceSyncSet` → `SyncReport` → synchronize raise / bound-drain `note_release_failure`; fallback `note_release_failure`) | OK | FIX2 N1 tests |
| d7 | `tcs/dlpack_operand.cpp:878` | `destroy_event_on_device` | consume + recorded (`event_destroy_failures`, `note_unraisable_failure`) | OK (FIX4) | FIX4 `destroy` |
| d8 | `tcs/dlpack_operand.cpp:993` | `sync_and_release_all_impl` cudaGetDevice | consume + `rep.query_failed` → raised by synchronize / recorded by the bound drain | OK | FIX2 N1 `empty_sync` |
| d9 | `tcs/dlpack_operand.cpp:1010` | `sync_and_release_all_impl` NotReady | benign: CUDA 12.6 PG 3.2.12, NotReady "is not considered an error" and the runtime does not put it in the slot; cleared defensively only if present | OK | reasoning + citation |
| d10 | `tcs/dlpack_operand.cpp:1013` | `sync_and_release_all_impl` other query error | consume + `rep.add` → raised / recorded; entry leaked (no completion proof) | OK | FIX2 N1 |
| d11 | `tcs/dlpack_operand.cpp:1044` | `acquire_event` create failure | consume + raise | OK | FIX4 `dev` |
| d12 | `tcs/dlpack_operand.cpp:1140` | `RetainUntilComplete` pre-launch take | taken + raised as pre-existing | OK (FIX4) | FIX4 `pre` |
| d13 | `tcs/dlpack_operand.cpp:1192` | `~RetainUntilComplete` failed event record (or the guard's failed switch) | 1475fae: consumed, never recorded (when the fallback sync succeeded the CUDA failure was lost). Now recorded first (`event_record_failures`, `note_unraisable_failure`), raised once by the next synchronize | **FIXED** | `test_fix5_sweep_d_failed_event_record_is_recorded_and_reported_once` |
| d14 | `tcs/dlpack_operand.cpp:1240` | `drain_completed` NotReady | benign (as d9) | OK | reasoning + citation |
| d15 | `tcs/dlpack_operand.cpp:1246` | `drain_completed` other query error (FIX4 rows 47/50) | 1475fae: consumed, never recorded (V5-02). Now recorded first (`event_query_failures`, `note_unraisable_failure`), raised once | **FIXED** (V5-02) | `test_fix5_v502_*` (3 modes), corrected `test_fix4_failed_event_query_in_the_drain_is_consumed_there`; probes `query_error_cpu.py`, `query_error_single_gpu.py` |
| d16 | `tcs/dlpack_operand.cpp:1293` | `synchronize_and_release_all` prior take | recorded (`prior_*`) + raised (V4-04) | OK | FIX4 `v404` |
| d17-d26 | `tcs/conv.cu:56,62`; `lstm.cu:77,83`; `norms.cu:73,160`; `small_ops.cu:45,101,113,121` | `check_cuda` / `check_launch` | consume + raise (a success leaves nothing consumed) | OK | FIX4 `checkcuda`, `dev`; `ccnone` reason |
| d27 | `core/bindings.cpp:440` | fence cudaGetDevice | consume + raise | OK | FIX4 `v401f` |
| d28 | `core/bindings.cpp:448` | fence cudaSetDevice | consume + raise | OK | FIX4 `v401f` |
| d29 | `core/bindings.cpp:471` | fence step failures | consume + raise | OK | FIX4 `v401f` |
| d30 | `core/bindings.cpp:942` | import cudaGetDevice | consume + raise | OK | FIX4 `v403` |
| d31 | `core/bindings.cpp:1063` | `empty` cudaGetDevice | consume + raise | OK | FIX4 `v403` |
| d32 | `core/bindings.cpp:1084` | `empty` failed allocation | consume + rethrow | OK | FIX4 `empty` |
| d33 | `core/kernels.cu:284` | external Storage drain failure | 1475fae: consumed, only the leak counted. Now recorded first (`note_core_unraisable`: count + first site/device/error) | **FIXED** | `test_fix5_sweep_d_core_drain_failure_is_recorded` |
| d34 | `core/kernels.cu:290` | drain's restore failure | as d33 | **FIXED** | `test_fix5_sweep_d_core_drain_restore_failure_is_recorded` |
| d35 | `core/kernels.cu:301` | core `raise_cuda_failure` | consume + raise | OK | FIX4 `fill` |
| d36 | `core/kernels.cu:343` | `~CheckedDeviceGuard` restore while unwinding | as d33 | **FIXED** | `test_fix5_sweep_d_core_fill_unwinding_restore_failure_is_recorded` |
| d37 | `core/kernels.cu:559` | `fill_` pre-launch take | raise as pre-existing | OK | FIX4 `fillpend` |
| d38 | `core/kernels.cu:568` | `cuda_check_last("fill_")` (baseline helper) | consume + raise | OK | `test_fix5_v504_fill_largest_*` (the trapped launch is raised, slot clean) |

Core's record (d33/d34/d36) is reported through `tensor_cuda._C._dlpack_release_stats()` (`unraisable_errors`,
`first_unraisable_what` / `_device` / `_error`): core's `synchronize()` is the baseline engine `cuda_sync`
(`cudaDeviceSynchronize` + `cuda_check_last`), outside the P1-T surface, and is not changed to raise them (D6).
Test code that touches the slot (not product): `tests/test_fix4_findings.py` and `tensor_cuda/tests/test_dlpack.py`
ctypes staging (`cudaGetLastError` / `cudaPeekAtLastError` asserted by the tests), `fix4_stage_kernel.cu:12,14`, the
shims' `shim_slot_peek` / `fill_shim_slot` (peek only).

**FIX4 rows 47/50 corrected.** Row 47 (`drain_completed` `cudaEventQuery`) and row 50 (`drain_completed`
`cudaGetLastError` of a non-NotReady failure) said "error → entry left for synchronize(), which re-queries and
reports/leaks it" (OK / FIXED (b)). That was false for a one-shot failure: the re-query succeeded and nothing was
reported (V5-02). Corrected verdict: **FIXED by FIX5 (V5-02)**, d15 above: recorded before consumption and raised once
by the next synchronize(); proving tests `test_fix5_v502_*` and the corrected FIX4 test.

Summary (d): 38 rows (36 product sites, d17-d26 grouped): **FIXED 5** (d13, d15, d33, d34, d36), **OK 33**, benign
NotReady 2 of them with citation; every FIXED row has a test.

### (e) Raw-resource exception safety

Every CUDA / cuBLAS / host resource in the P1-T code created before an operation that can throw.

| # | Resource (site) | Created before | Owned until commit by | Verdict | Test / reason |
|---|---|---|---|---|---|
| e1 | cuBLAS handle, `tcs/conv.cu` `handle_for` (`cublasCreate`) | 4 settings (may fail) and `handles.emplace` (may throw `std::bad_alloc`) | FIX4: explicit destroy on a failed setting; 1475fae: nothing through the insertion (V5-03). Now `NewHandleOwner` (`:101`) until `emplace` commits (`:125`) | **FIXED** (V5-03) | `test_fix5_v503_cache_insertion_failure_destroys_the_new_handle[conv]`, `_reviewer_cpu_probe_regenerated_*` |
| e2 | cuBLAS handle, `tcs/lstm.cu` `lstm_handle_for` | same | `NewHandleOwner` (`:119`) until `:143` | **FIXED** (V5-03) | `[lstm]`, regenerated probe |
| e3 | completion event, `tcs/dlpack_operand.cpp` `acquire_event` (`cudaEventCreateWithFlags`) | nothing after creation throws (`event_pool[device]` and the recycle-slot `reserve` precede it) | returned to `RetainUntilComplete` | OK | FIX2 N3 `alloc:event_pool_reserve` |
| e4 | the same event in `RetainUntilComplete`'s constructor | the pre-launch error check (throws) | recycled before that throw builds its message | OK | FIX4 `pre` |
| e5 | the event and the managers in `~RetainUntilComplete` | the pending-list push (may throw) | push failure caught → synchronous fallback, event recycled | OK | FIX2 N3 `alloc:pending_push` |
| e6 | pooled events, `recycle_event` | `push_back` within reserved capacity | on failure destroyed on its own device | OK | FIX2 N3 `alloc:event_recycle` |
| e7 | event of a leaked entry (`leak_entry`) | n/a | deliberately not recycled (may belong to a broken context): a documented leak, not an exception path | OK | reasoning (FIX1) |
| e8 | consumed DLPack managers (`ManagedOwner`, `Operand`) | validation, shape checks, launches | RAII: released once, or leaked by rule | OK | FIX1/FIX2 tests |
| e9 | capsule references (`CapsuleRef`, FIX5) | plan validation | RAII (admitted drop) | OK (new) | c2 tests |
| e10 | pending state / lifecycle singletons | n/a | created at import, never destroyed (by design) | OK | reasoning |
| e11 | `core/bindings.cpp:414` fence event | record / wait (C calls, no throw) | destroyed before any throwing step (message building) | OK | FIX4 `v401f` |
| e12 | `core/bindings.cpp` `DLPackExportCtx` | vector assigns, `PyCapsule_New` | `unique_ptr` until the capsule takes it (F06) | OK | `test_fix1_f06_*` |
| e13 | `core/bindings.cpp` imported `Storage` (`make_shared`) | rename, `NDArray` / `Tensor::make` | rename failure → no-op release; later throws release through the Storage | OK | FIX1/FIX4 import tests |
| e14 | `core/bindings.cpp` `tensor_empty` allocation | `Tensor::make` | `NDArray` / `Storage` RAII | OK | FIX4 `empty` |
| e15 | device switches (`DeviceGuard`, `QuietDeviceGuard`, `CheckedDeviceGuard`, the fence's switch) | launches / raises | restored on every path (FIX4) | OK | FIX4 `guard`, `quiet`, `fill`, `v401f` |

Test code: the FIX4/FIX5 shims and native harnesses create no resource across a throwing step that the tests rely on.
Summary (e): 15 rows: **FIXED 2** (e1, e2; both tested), **OK 13**.

### (f) Launch-geometry narrowing

Every kernel launch (and every cuBLAS call with integer dimensions) in the P1-T code.

| # | Launch (site) | Grid / block computation | Coverage of every accepted extent | Verdict | Test / reason |
|---|---|---|---|---|---|
| f1 | `tcs/conv.cu:72`, `lstm.cu:93`, `small_ops.cu:50` `grid_for(total, block)` | int64 `ceil(total / block)`, clamped to [1, kGridCap = 2^20] BEFORE the unsigned conversion (exact); block a constexpr 256 (`tuning.h`, `lstm_tuning.h`, `m4_ops.h`) | every kernel it feeds is an int64 grid-stride loop (`i += gridDim.x * blockDim.x`): any total is covered | OK | reasoning; census tests (`test_conv`, `test_lstm`, `test_norms`, `test_small_ops`) |
| f2 | `tcs/conv.cu:211` `fp32_divide_probe_kernel` | f1 | grid-stride over n (validated >= 1) | OK | `test_contract` canary |
| f3 | `tcs/conv.cu:237` `im2col1d_kernel` | f1 over C_in*K*T_out (= the checked scratch numel, `ck_mul`, `ck_fp32_bytes`) | grid-stride | OK | `test_conv` |
| f4 | `tcs/conv.cu:247` `cublasSgemmStridedBatched` | m = T_out, n = C_out/groups, k = C_in/groups*K, lda/ldb/ldc: `check_int_range` <= INT_MAX before the int casts; batch = groups (`check_attr` <= INT_MAX); strides `long long` | cuBLAS sizes its own grid | OK | FIX1 F03 `conv_lda_guard` (host_checks) |
| f5 | `tcs/conv.cu:257` `bias_add_kernel` | f1 over N*C_out*T_out (`ck_shape_numel`) | grid-stride | OK | `test_conv` |
| f6 | `tcs/conv.cu:269` `conv_transpose1d_gather_kernel` | f1 over the checked output size | grid-stride | OK | `test_conv` |
| f7 | `tcs/lstm.cu:247` `lstm_bias_fill_kernel` | f1 over D*T*N*4H (<= the checked scratch numel) | grid-stride | OK | `test_lstm` |
| f8 | `tcs/lstm.cu:253` `cublasSgemm` (input projection) | 4H, T*N, I checked <= INT_MAX | cuBLAS | OK | FIX1 `lstm_n_guard` |
| f9 | `tcs/lstm.cu:264`/`:273` `lstm_cell_kernel` | f1 over D*N*H (checked <= INT_MAX) | grid-stride | OK | `test_lstm` |
| f10 | `tcs/lstm.cu:266` `cublasSgemmStridedBatched` (recurrent) | 4H, N, H checked; batch D in {1, 2} | cuBLAS | OK | `test_lstm` |
| f11 | `tcs/norms.cu:157`/`:158` `row_norm_kernel` | grid = clamp(rows, 1, 2^20) (exact); block B = 256 constexpr | rows loop `row += gridDim.x` (int64); inner loops over L step B (int64) | OK | `test_norms` |
| f12 | `tcs/small_ops.cu:99` `upsample_nearest1d_kernel` | f1 over rows*T*scale (`ck_shape_numel`) | grid-stride | OK | `test_small_ops` |
| f13 | `tcs/small_ops.cu:111` `reflection_pad1d_kernel` | f1 over rows*(T+pb+pe) | grid-stride | OK | `test_small_ops` |
| f14 | `tcs/small_ops.cu:120` `rint_kernel` | f1 over numel | grid-stride | OK | `test_small_ops` |
| f15 | `core/kernels.cu:555`/`:566` `fill_` → `fill_kernel` (NOT grid-stride; one thread per element) | 1475fae: `nblk(n)` narrowed ceil(n/256) to int (2^40+1 → grid 1). Now `fill_grid_blocks`: int64 ceil; > 2^31-1 refused (ValueError) before the label proof, guard, launch and revision bump; launch with the checked count | blocks * 256 >= n for every accepted n (largest accepted: (2^31-1)*256, covered exactly) | **FIXED** (V5-04) | `test_fix5_v504_*` (4); probes `fill_grid_single_gpu.py`, regenerated `fill_grid_cpu.cpp` |
| f16 | test code `tc_signal/tests/fix4_stage_kernel.cu:13` | `<<<1, 4096>>>`: a deliberately invalid block (stages a real launch-configuration error; never runs) | n/a | OK (test code) | FIX4 `v404` |

Headroom of every grid-stride increment: each total is <= INT64_MAX / 4 (fp32 byte checks) and the stride is <= 2^20 *
256 = 2^28, so `i + stride` never overflows int64 (round-5 review R4 agreed). Summary (f): 16 rows (14 product launches
or cuBLAS calls, the shared helper, one test launch): **FIXED 1** (f15, tested), **OK 15**.

## Probe before/after table

Every probe in the round-5 report's probe table, on both builds; then the round-4-only probes on both builds and rounds
1-3 on this build. Every run a fresh process under `flock -w 900 /tmp/forge-gpu.lock timeout <probe timeout>`.
Conditions: nogpu = `CUDA_VISIBLE_DEVICES=`, gpu = device visible. Logs `artifacts/p1t_fix5/probe_runs/`, summaries
`rev5_summary_{before,after}_{cpu,gpu,rf,buildcpu}.txt`, `rev4_summary_{before,after}_cpu.txt`,
`round123_summary_after.txt`. Path-only copies and regenerations: D3, D5, D8.

| Probe (round 5) | How it ran | Before (1475fae) | After (this build) | Result |
|---|---|---|---|---|
| `exit_gate.py full` nogpu | unmodified | **rc 1** child rc -6, `CALLBACK IN PROGRESS entries=0`, no `DELETER finished` | **rc 0** `DELETER finished` then `FINALIZING True`, PASS | **FIXED** |
| `exit_gate.py full --control` nogpu | unmodified | rc 0 | rc 0 | PASS |
| `exit_gate.py full` gpu | unmodified | **rc 1** `FATAL: exception not rethrown` | **rc 0** PASS | **FIXED** |
| `exit_gate.py full --control` gpu | unmodified | rc 0 | rc 0 | PASS |
| `exit_gate.py trace` | trace module built from each arm's `dlpack_operand.cpp` (D5) | **rc 1** `HOOK returned entries=0`, `FATAL: exception not rethrown` | **rc 0** `DELETER finished` before `HOOK returned`, PASS | **FIXED** |
| `exit_gate.py trace --control` | same | **rc 1** (the trace child ignores `--control`: same staging, D4) | rc 0 | see D4 |
| `exit_gate.py gpu` | unmodified | **rc 1** `FATAL: exception not rethrown` | **rc 0** PASS | **FIXED** |
| `exit_gate.py gpu --control` | unmodified | rc 0 | rc 0 | PASS |
| `query_error_cpu.py` | trace module (D5) | **rc 1** `FAIL synchronize returned success` | **rc 0** `PASS reported … cudaEventQuery … (cudaErrorInvalidValue …)` | **FIXED** |
| `query_error_single_gpu.py` | `LD_PRELOAD=/tmp/p1t_rev5_query_once.so` (existing, byte-identical to a fresh build) | **rc 1** `one-shot query error disappeared before synchronize` | **rc 0** `PASS: query error retained and reported once; safe completion release` | **FIXED** |
| `handle_insert_oom_cpu.cpp` as written | `g++ -std=c++17 -O1` | rc 1 `conv creates=1 destroys=0`, `lstm creates=1 destroys=0` | rc 1 (same: it embeds 1475fae's functions) | arm-independent (D3) |
| `handle_insert_oom_cpu.cpp` regenerated from each arm (self-check: from 1475fae = the probe, byte-identical) | `fix5_regen.py` | **rc 1** (identical to the probe) | **rc 0** `conv creates=1 destroys=1`, `lstm creates=1 destroys=1` | **FIXED** |
| `fill_grid_cpu.cpp` as written | `g++` | rc 1 `n=1099511627777 grid=1 covered=256` | rc 1 (same: baseline `nblk`, unchanged by the order) | arm-independent (D3) |
| `fill_grid_cpu.cpp` regenerated (self-check: from 1475fae = the probe) | `fix5_regen.py` (main's geometry line too, D3) | **rc 1** (identical) | **rc 0** `n=1 grid=1 covered=256`; 2^40+1 and 2^40+256 `refused: fill_: … 4294967297 blocks …` | **FIXED** |
| `fill_grid_single_gpu.py` | `LD_PRELOAD=/tmp/p1t_rev5_fill_grid.so` (existing, byte-identical) | **rc 1** `oversized fill was not refused` | **rc 0** `REFUSAL fill_: 1099511627777 elements need 4294967297 blocks … launches 0`, PASS | **FIXED** |
| `device_error_sweep_single_gpu.py` | path copy (root + basetemp, D8) | rc 0, 40 passed (its own tree) | rc 0, 40 passed | PASS |
| `core_export_device.py` | unmodified | exit 3 (two GPUs) | exit 3 | INCONCLUSIVE (one GPU) |
| `core_import_query_error.py` | unmodified, nogpu | rc 0 PASS | rc 0 PASS | PASS |
| `sync_existing_error.py` | unmodified | rc 0 `PASS: staged error reported 98 98` | rc 0 | PASS |
| `addon_lifetime.py` | unmodified | rc 0, 12/12 correct, 0 incorrect | rc 0, 12/12, 0 incorrect | PASS |
| `core_edges.py` | unmodified | rc 0 | rc 0 | PASS |
| `core_device_drain.py` | unmodified | exit 3 | exit 3 | INCONCLUSIVE (one GPU) |
| `core_exit_overrun.py during` | unmodified | rc 0 (3.3 s) | rc 0 (3.3 s) | PASS |
| `core_exit_overrun.py bounded` | unmodified | rc 0 (7.8 s) | rc 0 (7.7 s) | PASS |
| `numerics_and_streams.py` | unmodified | rc 0 PASS | rc 0 PASS | PASS |
| `release_failures.py` | before: path copy (root = wt_fix4); after: original | rc 0, 66 passed, no skip | rc 0, 66 passed, no skip | PASS |
| `cuda_errors_multidevice.py` | unmodified | exit 3 | exit 3 | INCONCLUSIVE (one GPU) |
| `build_cpu.sh` | path copy (build dir, D8), cwd = the arm's tree | rc 0 | rc 0 | PASS |

**Round 4 (the CPU probes that round 5 does not repeat; both arms; `run_rev4_cpu_probes.sh`):** host_checks 20/20
cases with identical rc AND text on both arms, nogpu and gpu (`rev4_host_checks_compare.txt`); `fix2_harness ck_census`
18 cases / 0 failures and `fix3_harness accounting` 17 / 0, no UBSan line, both arms and conditions; `lifecycle_timeout`
control / short / default rc 0 in all 12 arm-condition runs (after arm: the slice regenerated from this tree, FIX3's
regen_lifecycle_slice.py, selfcheck identical=True, D8); operand_census PASS 17 cases, real_metadata_plans PASS 129
cases, both arms and conditions; `sync_slot_cpu` rc 1 on both arms (the FIX4-fixed defect trace is not reproduced);
`core_interop_cpu` as written reproduces its embedded 62c8d9f traces on both arms (arm-independent, FIX4 D9), and its
import regenerated from this tree refuses (rc 1, `IMPORT REFUSED … cudaGetDevice failed`).

**Rounds 1-3 (this build; `run_rounds123_probes.sh`):** `addon_lifetime` runs 1-3 rc 0, 12/12 correct, 0 incorrect
each; `core_edges` 0; `core_shutdown --normal` 0 (`PASS: ordinary foreign-thread release`) and finalizing 0 (`PASS:
shutdown foreign-thread release completed`); `numerics_and_streams` 0; round-1 `host_checks` identical to FIX4's receipt
once the worktree name is normalized; `core_device_drain` 3; round-3 `core_exit_overrun` during 0 (3.3 s) and bounded
0 (7.8 s); round-2 `host_checks` identical to FIX4's receipt in rc and text, nogpu 20/20 rc 0, gpu 19 rc 0 + `oom_sync`
-6 (FIX2 D4: the probe's own pybind11 teardown, unchanged) (`round12_host_checks_vs_fix4.txt`).

## Gate table

| Gate | Criterion (registered in the ledger at 05:19, before any change) | Result | Receipt |
|---|---|---|---|
| G1 | `exit_gate.py full` (both conditions), `trace`, `gpu` fail on 1475fae and pass here | **PASS**: rc 1 → 0 for all four | `probe_runs/rev5_exit_gate_*` |
| G1 | `query_error_single_gpu.py`, `query_error_cpu.py` fail before, pass here | **PASS**: rc 1 → 0, rc 1 → 0 | `rev5_query_error_*` |
| G1 | `handle_insert_oom_cpu.cpp` fails before, passes here | **PASS as regenerated** (rc 1 → 0); as written rc 1 on both arms (it embeds 1475fae code; D3) | `rev5_handle_oom_*`, `regen_*.txt` |
| G1 | `fill_grid_cpu.cpp` and `fill_grid_single_gpu.py` fail before, pass here | `fill_grid_single_gpu.py` **PASS** (rc 1 → 0); `fill_grid_cpu.cpp` **PASS as regenerated** (rc 1 → 0), as written rc 1 on both arms (it embeds the unchanged baseline `nblk`; D3) | `rev5_fill_grid_*` |
| G1 | the `--control` variants pass on both builds | full and gpu: **PASS** (rc 0 on both); trace --control: rc 1 on 1475fae (the trace child has no control logic; D4) | `rev5_exit_gate_*_control_*` |
| G1 | the new regression for each finding passes | **PASS**: tc_signal 23 new tests + the corrected FIX4 test; core 13 new tests | combined session |
| G2 | sweep tables (c)-(f) complete; every site OK or FIXED; every FIXED site has a test or a stated reason; FIX4 rows 47/50 corrected | **PASS**: (c) 13 paths FIXED 6 / OK 7, 10 handlers; (d) 38 rows FIXED 5 / OK 33; (e) 15 rows FIXED 2 / OK 13; (f) 16 rows FIXED 1 / OK 15; every FIXED row tested; rows 47/50 corrected | G2 section |
| G3 | every round-1..5 probe passes or is INCONCLUSIVE only for lack of 2 GPUs | **PASS** (three two-GPU probes exit 3) | `probe_runs/` |
| G3 | `addon_lifetime` 3/3 runs, 0 incorrect | **PASS** | `round1_addon_lifetime_run{1,2,3}.log` |
| G3 | `core_exit_overrun` during and bounded rc 0 | **PASS** (round 3 and round 5 copies) | `round3_*`, `rev5_core_exit_overrun_*_after.log` |
| G3 | Voice T1-T6 + T1R + T4S PASS 249/249 | **PASS** | `logs/p1t_fix5_g3_voice.log`, `artifacts/p1t_fix5/p1t_all.json` |
| G3 | combined session = the 52 baseline ids | **PASS**: 52 = 52, 0 new, 0 no longer failing (final run 06:34-06:43, 2765 passed; pass 1 also `G1 PASS`, D9) | `logs/p1t_fix5_g3_combined_session.log`, `logs/p1t_fix5_g3_compare.txt` |
| G4 | mutants (a)-(d) each fail their new test | **PASS**: all four killed | `artifacts/p1t_fix5/g4/` |
| G5 | diff vs b2d7918 only in Exercises + FIX5 docs / artifacts / logs; probe files byte-identical; no new core `__global__` | **PASS** (`g5_scope.py`: every changed path allowed; HEAD vs b2d7918 = the order file only; 0 new core `__global__`, 0 atomics / fast-math / TF32 / device-wide syncs / host callbacks / exit calls in product code; no escaping throw in the 18 destructors and noexcept bodies checked; CMakeLists unchanged and no `_tensor_cuda` in DT_NEEDED; probe files byte-identical, aggregate = the registered hash) | `artifacts/p1t_fix5/g5_scope.txt` |

**Voice's verdict line** (`artifacts/p1t_fix5/p1t_all.json` summary):
`verdict PASS exit_code 0 counts {"PASS": 249, "FAIL": 0, "SKIP": 0, "ERROR": 0} (T1 11, T2 144, T3 28, T4 12, T5 28, T6 13, T1R 1, T4S 12) seconds 55.9`.

**Compare output** (`logs/p1t_fix5_g3_compare.txt`, final run):
```
G1 compare (APA-MF2.2): baseline = /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2/logs/apa_mf1_g1_baseline_unchanged.log; now = logs/p1t_fix5_g3_combined_session.log
summary: 37 failed, 2765 passed, 47 skipped, 2 warnings, 15 errors in 549.48s (0:09:09)
baseline ids 52 now ids 52
baseline ids that no longer fail: 0 []
new failing ids: 0 []
new MF2.2 test files with any FAILED/ERROR: 0
G1 PASS
```
Passed is 2765 against FIX4's 2729: core added 13 tests, tc_signal 23 (the FIX4 query test was corrected in place). The 37 failures and 15 errors are the baseline APA files.

## G4 mutant table

Trees `/tmp/p1t_fix5_mut_<m>/` from `artifacts/p1t_fix5/g4/make_mutants.py` (one asserted edit each; diffs
`g4/mut_<m>.diff`); runner `g4/run_mutant.sh` (the mutant tree's test files cmp-identical to the worktree's).

| Mutant | Edit | .so sha256 | New tests on the mutant | Probes on the mutant | Verdict |
|---|---|---|---|---|---|
| (a) V5-01 admission wait removed | `wait_for_other_threads`: `if (true) return w;` (no exit wait, no synchronize() wait) | tc_signal `099a3543fb164f4a2f585f5d54f0e74cf3d6aa133f76cd7c5191376eb6d3f6fa` | `-k v501`: **10 failed**, 6 passed. Failed: immediate 5.5 / 12 s, deferred 0.1 / 5.5 / 12 s, no-GPU, capsule destructor, producer callback, throwing deleter at exit (rc -6 aborts), synchronize wait. Passed: immediate 0.1 s (it finishes before finalization by timing), the after-close and self-exclusion cases (they do not rely on the wait), the script check | `exit_gate.py full` nogpu / gpu, `trace`, `gpu`: all **rc 1** (child rc -6) | KILLED |
| (b) V5-02 record removed | `drain_completed`: the record lines deleted (consume only, as 1475fae) | tc_signal `7ecbea3c100c2115d18cb837bd5b4cdecb29e51bc7232d5b6a8c879171b5044a` | `-k v502`: **3 failed** (all three modes: synchronize returned success) | `query_error_single_gpu.py` **rc 1**, `query_error_cpu.py` **rc 1** | KILLED |
| (c) V5-03 ownership removed | both `~NewHandleOwner`: `return;` (never destroys) | tc_signal `51804f1b0bec6ef6e336c55fce0d74142d43f913e2671eaa98fb4b8cdaca2c89` | `-k v503`: **3 failed** (conv and lstm: creates 1, destroys 0; the probe regenerated from the mutant's sources rc 1) | (the regenerated probe is one of the 3) | KILLED |
| (d) V5-04 refusal removed | `fill_grid_blocks`: `if (false)` before the throw | core `df448146a2b6c01bbf285790c5d038ab4bc23b1204f869453390c8ed45b562ce` | `-k fix5_v504`: **2 failed** (one_more: launched grid 2^31, `RuntimeError` not `ValueError`; huge: launched, narrowed), 3 passed (largest, zero, small: unaffected by the refusal) | `fill_grid_single_gpu.py` **rc 1** `oversized fill was not refused` | KILLED |

## Operand census → tests and probes

| Census item (order) | Covered by |
|---|---|
| V5-01: foreign-thread deleters of 0.1 s, 5.5 s, 12 s, immediate-refusal path, ordinary exit | `test_fix5_v501_immediate_refusal_deleter_in_flight_at_exit[0.1, 5.5, 12.0]` (diagnostics: none / 5.0 s / 5.0 and 10.0 s); `exit_gate.py full` both conditions |
| V5-01: the same on the deferred path | `test_fix5_v501_deferred_release_in_flight_at_exit[0.1, 5.5, 12.0]`; `exit_gate.py gpu`; `exit_gate.py trace` (the reviewer's trace) |
| V5-01: a release that starts after the hook closed admission (leaks without entering Python, counted) | `test_fix5_v501_releases_and_callbacks_after_close_leak_or_refuse_without_entering_python` (4 deleters leaked, none called, `deleters_leaked_after_close` +4, `admitted_releases` +0; a producer `__dlpack__` refused and never called); `_producer_dlpack_callback_in_flight_at_exit_and_late_capsule_drop_is_kept` (`capsules_kept_after_close` 1, destructor not run) |
| V5-01: the hook invoked from inside a callback on the same thread (no deadlock) | `test_fix5_v501_exit_hook_inside_its_own_deleter_does_not_wait_for_itself` (hook < 1 s, own release excluded); `_inside_its_own_producer_callback_*` (hook inside `__dlpack__`, then 2 capsule drops kept) |
| V5-01: a deleter that raises an ordinary C++ exception (contained once) | `test_fix5_v501_throwing_cpp_deleter_contained_once_on_the_main_thread` (2 calls, `deleter_exceptions` +2, in flight back to 0); `_throwing_cpp_deleter_in_flight_at_exit_is_contained_once` (a native deleter that releases the GIL, sleeps 0.6 s, re-takes it and throws, on a daemon thread at exit: rc 0, `deleter_exceptions` 2); FIX2 N4 tests (pass) |
| V5-02: one-shot query failures with nothing else pending, and with completed entries pending; reported exactly once; next synchronize() succeeds | `test_fix5_v502_*[one_pending, one_pending_then_drained, completed_pending]` (genuine slot failure through `fix4_cudart_shim.c`); `query_error_single_gpu.py`, `query_error_cpu.py` |
| V5-03: an injected insertion failure for each cache; handle destroyed; next call creates a working handle | `test_fix5_v503_cache_insertion_failure_destroys_the_new_handle[conv, lstm]` (a REAL operator new failure at the insertion; creates 1 = destroys 1; the next call: creates 2, destroys 1, correct output, settings intact); regenerated `handle_insert_oom_cpu.cpp` |
| V5-04: the largest accepted extent, one more, a zero-size fill_, synthetic metadata, no terabyte allocation | `test_fix5_v504_*` (largest (2^31-1)*256: grid 2^31-1 x 256, launch trapped by the shim; +1 and 2^40+1 refused, no launch, revision unchanged; zero: no launch); `fill_grid_single_gpu.py` |

## Prior art

- **Core's own `ImportReleaseLifecycle`** (`tensor_cuda/src/bindings.cpp`, P1T-FIX2 N5, P1T-FIX3 V3-01). Reused for
  V5-01: closing flag + in-flight count + condition variable; the unbounded wait with the GIL released and the doubling
  stderr diagnostic schedule; the thread-local depth for self-exclusion; "after closing, leak without entering
  Python"; the atexit-before-finalizing argument (Py_FinalizeEx ordering, verified by FIX2 in CPython v3.12.3
  pylifecycle.c). Different: the add-on's releasers already hold the GIL (no PyGILState_Ensure), so admission is two
  seq_cst atomics, not a mutex section; two section kinds (releases, producer callbacks); the GIL re-taken explicitly,
  not by a destructor (core's own wait changed to match, sweep c); synchronize() waits too.
- **CPython 3.12**: atexit before finalizing (`Py_FinalizeEx`); a non-main thread that takes the GIL during finalization
  exits (`PyEval_RestoreThread` → `take_gil` → `_PyThreadState_MustExit` → `PyThread_exit_thread` → `pthread_exit`,
  `Python/ceval_gil.c`); `threading._shutdown` (finalization waits, unbounded, for non-daemon threads). Taken: these
  contracts. Unverified in this seat (lead to check): "CPython 3.12.3 ceval_gil.c take_gil _PyThreadState_MustExit
  PyThread_exit_thread".
- **glibc / libstdc++ forced unwinding** (`abi::__forced_unwind` must be rethrown by a catch-all, otherwise glibc aborts
  "FATAL: exception not rethrown"; a noexcept frame terminates); pybind11 v2.12's dispatcher rethrows it
  (`pybind11.h:1037`, read here). Taken: keep forced unwinds out of noexcept frames and catch-alls.
- **RCU grace periods** (McKenney & Slingwine, "Read-Copy Update: Using Execution History to Solve Concurrency
  Problems", 1998) and the Linux `percpu_ref` kill-then-wait-for-zero. Taken: stop admitting, wait unboundedly for what
  already started. Not taken: an epoch-based grace period for synchronize()'s non-closing wait (R4).
- **Store-buffering / Dekker handshake** (Dijkstra, "Cooperating Sequential Processes", 1965, for Dekker's algorithm;
  the SB litmus test of x86-TSO, Sewell et al. 2010): with a sequentially consistent flag and counter, at least one
  side sees the other. Taken: admission without a lock. Unverified (lead to check): "Sewell x86-TSO 2010 store
  buffering".
- **Scope guards / RAII** (Alexandrescu & Marginean, "Generic: Change the Way You Write Exception-Safe Code - Forever",
  C/C++ Users Journal, December 2000; Stroustrup's RAII; `std::unique_ptr` with a custom deleter). Taken for V5-03: own
  until commit, release on unwinding. Ours: checking and counting the cleanup status during unwinding.
- **CUDA 12.6 grid-dimension limits**: CUDA C++ Programming Guide 12.6, "Technical Specifications per Compute
  Capability", maximum x-dimension of a grid = 2^31-1 (every compute capability CUDA 12.6 supports); measured
  `cudaDevAttrMaxGridDimX` = 2147483647 on this host (test). Table number unverified (lead to check: "CUDA 12.6
  Programming Guide maximum x-dimension of a grid of thread blocks"). ISO C++17 [conv.integral] (an out-of-range signed
  conversion is implementation-defined) and C++ Core Guidelines ES.46 / `gsl::narrow`. Taken: compute wide, check, then
  narrow. Ours: refusing at fill_'s boundary instead of capping.
- **Sticky versus non-sticky CUDA errors** (CUDA 12.6 PG 3.2.12; FIX4's measured receipts
  `artifacts/p1t_fix4/cuda_semantics.txt`): a non-sticky query failure is one-shot (a later query can succeed), which is
  why V5-02 must record before consuming; a sticky one stays in the context and later calls report it anyway. NotReady
  "is not considered an error" (PG 3.2.12): not recorded.
- **Testing**: LD_PRELOAD interposition (glibc ld.so; the reviewer's shims, FIX1/FIX4 shims), fail-the-next-allocation
  fault injection (SQLite's OOM harness, Linux failslab), DeMillo, Lipton & Sayward (1978) mutation testing.
- **Ours (no prior art known to me):** admitting the last-reference drop of an unconsumed capsule; admitting the
  producer `__dlpack__` callbacks of a DLPack consumer; the explicit-restore placement so that a CPython-made forced
  unwind never starts in a destructor; recording core's destructor-path CUDA errors next to their counters.
- These annotations are also at the code sites (`dlpack_operand.cpp` lifecycle block, `drain_completed`,
  `~RetainUntilComplete`; `conv.cu` / `lstm.cu` handle caches; `kernels.cu` V5-04 and sweep (d) blocks; `bindings.cpp`
  `dlpack_begin_shutdown`; `tc_signal/__init__.py` hook; README prior-art list) and in the ledger.

## Deviations (with reasons)

- **D1. Scope of V5-01 beyond the deleter call: capsule drops and producer callbacks are admitted too, and
  synchronize() waits.** Sweep (c) asked that every callback that may take the GIL be admitted or leak without Python.
  The class contains two more add-on members than the deleter call: dropping the last reference to an unconsumed
  capsule (its destructor calls the producer's deleter; reached when a plan refuses after acquiring) and the producer's
  `__dlpack__` / `__dlpack_device__` calls. Both are now admitted. Behavior changes: (1) after the exit hook closed
  admission, a call that needs a producer's `__dlpack__` raises RuntimeError (a bare capsule still works; its releases
  leak); (2) `synchronize()` waits for releases already running on other threads (needed for the reviewer's trace
  probe, whose exit hook is `synchronize_and_release_all` itself), skipped inside the caller's own callback.
- **D2. A FIX4 test was corrected.** `test_fix4_failed_event_query_in_the_drain_is_consumed_there` asserted that the
  next `synchronize()` SUCCEEDS after a drain query failure, which pinned V5-02 (FIX4 rows 47/50). It now asserts the
  stronger behavior: raised once, naming `cudaEventQuery`, the following synchronize() succeeds, everything released.
- **D3. Two probes embed product code.** `handle_insert_oom_cpu.cpp` embeds 1475fae's two cache functions and
  `fill_grid_cpu.cpp` the baseline `nblk` helper (which the order keeps unchanged), so as written they are rc 1 on any
  build. They ran as written on both arms AND regenerated from each arm's sources (`tc_signal/tests/fix5_regen.py`;
  self-check: regenerating from 1475fae reproduces both probes byte for byte). For `handle_insert_oom_cpu.cpp` the
  regeneration substitutes the two functions and adds a counting stub of `note_cublas_handle_destroy_failure` (diff
  `probe_runs/handle_insert_oom_regen_after.diff`). For `fill_grid_cpu.cpp` it also changes main's geometry line to the
  fill_ path's (fill_grid_blocks, its refusal counted as the probe header's "refused" outcome; diff
  `fill_grid_regen_after.diff`): a change beyond a slice substitution. The decisive V5-04 probe is
  `fill_grid_single_gpu.py` on the real binary, unmodified (rc 1 → 0).
- **D4. `exit_gate.py trace --control`** passes `--control` to `addon_exit_cpu.py`, which never reads it (no join), so
  on 1475fae the control variant fails like `trace` (rc 1). The order expected every control variant to pass on both
  builds; this one cannot, by the probe's construction. Reported as observed.
- **D5. Trace module build command reconstructed.** The reviewer's `logs/p1t_rev5_trace_build.log` is empty. Built as
  `g++ -std=c++17 -O1 -shared -fPIC addon_trace.cpp -I<arm>/tc_signal/src -I<arm>/tc_signal/include -I<pybind11>
  -I<cuda> $(python3-config --includes)` per arm, matching the reviewer's binary (no libcudart dependency; `nm` lists no
  undefined cuda* symbol on either arm, `trace_build_*.log`).
- **D6. Core's recorded destructor-path errors are reported through `_dlpack_release_stats()`**, not by core's
  `synchronize()`: that is the baseline engine `cuda_sync`, outside the P1-T surface ("Does not cover: pre-existing core
  code"). The tc_signal side raises its records at the next `synchronize()` as ordered.
- **D7. A core fix outside the four findings, found by sweep (c):** `dlpack_begin_shutdown` (FIX3 code, reviewed CLOSED
  for V3-01) re-took the GIL in `py::gil_scoped_release`'s destructor; a non-main waiter could be terminated (1475fae: 2
  of 5 runs rc -6). Now an explicit restore; the lifecycle probe's embedded slice therefore changed, and the after arm
  compiles it regenerated (D8).
- **D8. Probe plumbing, all path-only or regeneration, originals byte-identical:** `make_probe_copies.py`
  (`device_error_sweep_single_gpu.py`: root and `--basetemp` → `/tmp/p1t_fix5_probes/<arm>/`; `release_failures.py`
  root = wt_fix4 on the before arm; `build_cpu.sh` build dir); FIX4's round-4 helpers and runner and its rounds-1..3
  runner retargeted to this worktree; the after arm's `lifecycle.cpp` regenerated with FIX3's `regen_lifecycle_slice.py`
  (selfcheck identical=True). The reviewer's shims at `/tmp/p1t_rev5_query_once.so` and `/tmp/p1t_rev5_fill_grid.so`
  were verified byte-identical to fresh builds of the probe sources and used read-only, not rebuilt over.
- **D9. The combined session ran as a background job inside this turn.** Its wall time (566 s first pass) is at my
  tool's 10-minute foreground ceiling, so it ran detached under the order's exact command and I read its result only
  after its completion marker existed. Two passes: pass 1 (06:21-06:31, `logs/p1t_fix5_g3_combined_session_pass1.log`,
  compare `G1 PASS`, 2764 passed) predates one added test (`test_fix5_sweep_d_core_drain_failure_is_recorded`); the
  final pass on the final files is the receipt.
- **D10. Network.** The core build (`./build.sh 89`, as ordered) re-cloned pybind11 v2.12.0 from GitHub through
  FetchContent (as FIX3 D17 / FIX4 D11). Mutant builds used production's pybind11 source read-only (offline).
- **D11. My own errors, found and fixed before acceptance (ledger):** a V5-02 test staging that could not hold three
  pending entries (drains release completed entries; the add-on's first launch synchronizes; a discarded engine tensor
  is freed with cudaFree); an after-close script that kept two ctypes-destructor capsules alive into module teardown
  (segfault reproduced on 1475fae with no tc_signal call: a test-helper lifetime issue); a self-exclusion staging that
  refused its first operand too early; a thrower staging that waited for the wrong deleter; an awk line counter that
  skipped empty lines; a G5 checker that matched a comment containing `__global__`. None changed a gate threshold.
- **D12. Mutant (d)'s copied test file** predated the last added test; it was refreshed (tests only) and the mutant
  re-run; the V5-04 results are the same (2 failed, the probe rc 1).
- **D13. Registration count.** The round-1..5 probe set is 73 files; the registration's "82" counted `ls` directory
  headers. The registered aggregate hash was computed over the files and matches.
- **D14. Timing.** Load average 26-30 throughout (other users). Wall times are not performance measurements.
- **D15. A report-file write to scratch was refused by the harness** (a draft of these sections); the sections were
  written into this report instead.

## Honest residuals (not claimed fixed, or not tested)

- **R1. No real two-GPU execution** (one GPU): the three two-GPU probes remain INCONCLUSIVE.
- **R2. A producer deleter that calls `pthread_exit` or is cancelled** still aborts in the add-on's and core's catch-all
  (k1, k7): a DLPack contract violation. FIX5 removes only the interpreter-made forced unwind.
- **R3. Core's `__dlpack__` argument casts** may run a user object's Python methods from core C++ frames; they are not
  admitted. A forced unwind there propagates (no catch-all, no noexcept) to pybind11's rethrow; py::object destructors
  on that path decref without the GIL, a generic pybind11 / CPython daemon-thread property.
- **R4. synchronize()'s non-closing wait is count-based**, not an epoch grace period: it ends at the first instant no
  release of another thread is in flight, so continuously overlapping releases on other threads can extend it, and a
  release that never returns makes it wait (with diagnostics). Safety (it never returns while a release that started
  before it still runs) holds; liveness under that contention is not bounded.
- **R5. A producer callback that never returns makes interpreter exit wait** (documented consequence, as core's V3-01).
- **R6. Code that unregisters tc_signal's atexit hook** (`atexit.unregister`) removes the exit protection; the
  finalizing check still keeps releases during finalization out of Python.
- **R7. Core's recorded unraisable errors are not raised by core's synchronize()** (D6).
- **R8. The new test suites were not run as a suite against the 1475fae binaries** (they read FIX5-only stats keys);
  the before evidence is the reviewer's probes on both arms, the core sweep-(c) test on 1475fae (2/5 abort) and the four
  killed mutants.
- **R9. Performance was not measured** (the per-call admission cost: two atomic increments and decrements, and one
  uncontended mutex lock/unlock per deleter call or producer callback).
- **R10. Free-threaded CPython and subinterpreters remain unsupported** (unchanged).

## Diff stat

Code changed by this order (`git diff --stat 1475fae -- tensor_cuda tc_signal`, working tree):
```
 tc_signal/README.md                   |  57 ++++-
 tc_signal/src/bindings.cpp            |  34 ++-
 tc_signal/src/conv.cu                 |  26 ++-
 tc_signal/src/cuda_guard.h            |   6 +
 tc_signal/src/dlpack_operand.cpp      | 408 +++++++++++++++++++++++++++++++++-
 tc_signal/src/dlpack_operand.h        |  56 ++++-
 tc_signal/src/lstm.cu                 |  21 +-
 tc_signal/src/lstm_bindings.cpp       |  19 +-
 tc_signal/src/m4_bindings.cpp         |  22 +-
 tc_signal/src/tuning.h                |  11 +
 tc_signal/tc_signal/__init__.py       |  60 ++++-
 tc_signal/tests/test_fix4_findings.py |   7 +-
 tensor_cuda/src/bindings.cpp          |  52 ++++-
 tensor_cuda/src/kernels.cu            | 105 ++++++++-
 tensor_cuda/tests/test_dlpack.py      | 247 ++++++++++++++++++++
 15 files changed, 1071 insertions(+), 60 deletions(-)
```
- New untracked files: `tc_signal/tests/test_fix5_findings.py` (902 lines), `tc_signal/tests/fix5_regen.py` (65); `docs/P1T_FIX5_LEDGER.md`, this report, `artifacts/p1t_fix5/**`, `logs/p1t_fix5_*`.
- `git diff --stat b2d7918` (tracked): 16 files changed, 1315 insertions(+), 60 deletions(-) (the 15 above plus the lead's order file on HEAD).

## .so sha256 (final build)

- core `tensor_cuda/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so`:
  `9af40433f69e4d5582a4a83bb65165ee4e9f72cdcf35c28a7ad4f4cceca15510`
- `tc_signal/tc_signal/_tc_signal.cpython-312-x86_64-linux-gnu.so`:
  `3324bb4a44dde6a45e62fcd91114d60ed1ef3268a2b35f7f1aebbe08c67a2552`

## Changed files

- Core: `tensor_cuda/src/kernels.cu` (V5-04, sweep (d) record), `tensor_cuda/src/bindings.cpp` (sweep (c) exit-wait GIL
  restore, sweep (d) stats keys, fill_ doc), `tensor_cuda/tests/test_dlpack.py` (FIX5 block, 13 tests).
- tc_signal: `src/dlpack_operand.{h,cpp}`, `src/conv.cu`, `src/lstm.cu`, `src/cuda_guard.h`, `src/tuning.h`,
  `src/bindings.cpp`, `src/lstm_bindings.cpp`, `src/m4_bindings.cpp`, `tc_signal/__init__.py`, `README.md`; tests
  `test_fix5_findings.py` (new, 23 tests), `fix5_regen.py` (new), `test_fix4_findings.py` (D2).
- Docs: `docs/P1T_FIX5_REPORT.md`, `docs/P1T_FIX5_LEDGER.md`. Receipts: `artifacts/p1t_fix5/**`, `logs/p1t_fix5_*`.
- Built and gitignored: both `.so` files and their `build/` directories. `.gitignore` ignores `*.log` and `*.json`: the
  receipts need `git add -f`.

**Scratch left (nothing deleted):** `/tmp/p1t_fix5_probes/{before,after}` (probe builds, path copies, regenerated
probes, rev4 builds), `/tmp/p1t_fix5_probes/rev2_after`, `/tmp/p1t_fix5_mut_{a,b,c,d}` (mutant trees and builds),
`/tmp/p1t_fix5_scratch` (bisection scripts, pytest basetemps, the sweep-table draft), pytest temporary directories under
`/tmp/pytest-of-vader/`.
