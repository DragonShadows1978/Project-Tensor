# P1T-FIX6 report: round-6 findings V6-01..V6-03 closed; sweeps (g) user Python inside native frames and (h) waits vs caller-held locks

**Verdict: G1 PASS, G2 PASS, G3 PASS (one item handled as the order's Required fix 1 prescribes: the round-5
`exit_gate.py trace` child models the exit hook as `synchronize()` alone and now fails as written, rc 1; the path-only
copy that calls the trace module's real exit entry passes, rc 0), G4 PASS, G5 PASS.**
- **V6-01 (MAJOR) closed:** `tc_signal.synchronize()` no longer waits for any other thread (FIX5's wait removed, not
  bounded); only the two exit hooks wait, under a documented no-lock contract. `sync_lock_inversion_cpu.py` rc 1 -> 0 in
  both conditions; `sync_late_release_cpu.py` rc 1 -> 0. Mutant (a) killed (6 tests, 3 probes).
- **V6-02 (MAJOR) closed, option (ii) refusal:** `__dlpack__` reads its arguments at C level and refuses every type it
  cannot read without calling a Python method; torch / tc_signal / tensor_cuda consumers unchanged (test).
  `core_cast_exit_gpu.py bool|index` rc 139 -> 0; regenerated `core_host_cpu.py casts_exit bool|index` rc -6 -> 0.
  Mutant (b) killed (30 tests, 2 probes).
- **V6-03 (MINOR) closed:** synchronize()'s own drain failures are persisted with no allocation before any message is
  built, and every record is cleared only once the Python RuntimeError exists. `release_trace_cpu.py
  diagnostic_oom_{query,sync,device}` rc 1 -> 0. Mutant (c) killed (7 tests, 3 probes).
- **Sweeps:** (g) 27 rows, FIXED 17 (every P1-T binding of both modules now takes `(*args, **kwargs)` with C-level
  conversion; the consumed-capsule drop is admitted; the two GIL-releasing exit waits are raw CPython functions), OK 10;
  (h) 14 rows, FIXED 1 (V6-01), OK 13. FIX5 sweep-(c) rows c2, c4, c6, c10, c11 and k1 corrected.
- **Lead directive (no residuals):** FIX5's R7 (a recorded core CUDA failure nobody raised) is FIXED (tested); R4 is
  removed with the wait; R2, R5, R6 and the environment items are in the "Not a defect" table with citations and, for R2,
  measurements on torch and NumPy. No residuals list.
- **A regression of my own, found by the suite and fixed:** my first rebinding of core's exit wait as
  `(*args, **kwargs)` gave pybind11 an object only the dispatcher owned across the GIL-released wait
  (`test_fix5_sweep_c_core_exit_wait_on_a_non_main_thread_never_aborts` rc -11, 3 of 5); the two exit waits are now raw
  CPython functions (5/5 pass).
- **G3:** every round-1..5 probe passes or is INCONCLUSIVE for lack of 2 GPUs (exit_gate trace as above);
  addon_lifetime 3/3, 0 incorrect; core_exit_overrun during / bounded rc 0; Voice PASS 249/249; combined session = the
  52 baseline ids (`G1 PASS`, 2886 passed); real-torch interop test passes.

Seat: Opus 5.5 (high). Order: `orders/P1T_FIX6_SYNC_WAIT_AND_USER_CALLBACKS.md` (6ab46a0). Ledger:
`docs/P1T_FIX6_LEDGER.md`. Receipts: `artifacts/p1t_fix6/` and `logs/p1t_fix6_*`. "Before" = wt_fix5's prebuilt acd42fa
binaries (read-only): core `9af40433f69e4d5582a4a83bb65165ee4e9f72cdcf35c28a7ad4f4cceca15510`, tc_signal
`3324bb4a44dde6a45e62fcd91114d60ed1ef3268a2b35f7f1aebbe08c67a2552`. Mid-task lead directive (ledger "~07:40 Lead directive"): no known
defect may be listed as a residual; it is fixed with a test, or shown with a citation to be not a defect in our code.

## Per-finding table

| Finding | Fix (file:line) | Proving tests (new) | Probe before (acd42fa) | Probe after (this build) |
|---|---|---|---|---|
| **V6-01** (MAJOR): `tc_signal.synchronize()` waited for releases running on other threads; a finite deleter needing a lock the caller held deadlocked it (FIX5's own D1) | `tc_signal/src/dlpack_operand.cpp`: `:122` design block (V6-01, prior art); `:1354` `synchronize_and_release_all` no longer calls the wait (FIX5's `:1284-1291` block removed); `:304` `wait_for_other_threads` is exit-only (`:1119` `close_admission_and_wait` its only caller); the three `sync_wait*` counters and `tuning::kSyncReleaseWaitFirstDiagS` removed. The exit wait itself is a raw CPython function (`tc_signal/src/bindings.cpp:352` `raw_exit_close_admission`, `py_args.h:203`), see sweep row g21 and the corrected c4 row below. Docs: `tc_signal/__init__.py` `synchronize`, `_release_pending_at_exit` (the exit hook's no-lock contract), README | `tc_signal/tests/test_fix6_findings.py`: `:154` lock inversion, gpu and nogpu (returns with the lock still held, no rescue); `:362` a release started after synchronize() began is not waited for (host trace module of this source); `:255` the exit hook still waits (refused-operand deleter; consumed-capsule destructor); corrected `test_fix5_findings.py:630` (synchronize does NOT wait) | `sync_lock_inversion_cpu.py` nogpu **rc 1** (`BLOCKED_BEFORE_CUDA=1`, rescue), gpu **rc 1**; `sync_late_release_cpu.py` **rc 1** (`waited_for_later=True`, 0.703 s) | `sync_lock_inversion_cpu.py` nogpu **rc 0**, gpu **rc 0** (`rescue=False`); `sync_late_release_cpu.py` **rc 0** (`waited_for_later=False`, 0.000 s, final build); `exit_gate.py full` / `gpu` rc 0 as written |
| **V6-02** (MAJOR): core `Tensor.__dlpack__` ran user Python in argument conversion (`__bool__`, `__index__`, sequence `__len__` / `__getitem__` / `__iter__`); at exit the process crashed (FIX5's R3) | `tensor_cuda/src/bindings.cpp`: `:488` design block (option (ii), justification, accepted types, prior art); `:542` `tensor_dlpack` reads every argument at C level first (`PyLong_AsLongLongAndOverflow`, `PyTuple_GET_ITEM` / `PyList_GET_ITEM`, bool identity) and refuses anything else with TypeError before anything runs; `:426` `dlpack_fence_consumer` takes the already-read stream; `:1515` the binding is `(*args, **kwargs)` + `CoreCallArgs` (`:1256`) | `tensor_cuda/tests/test_dlpack.py`: `:2184` 35 argument x value cases (recording objects, non-tuple sequences, iterables, tuple / list / int subclasses whose overrides must not run): refused or accepted, zero Python calls; `:2296` the reviewer's exit staging for stream / copy / max_version / dl_device x `__bool__` / `__index__` / `__len__` / `__getitem__` / `__iter__` (11 cases), rc 0, method never run; `:2213` real torch consumers (`torch.from_dlpack` with and without `device` / `copy`, `torch.utils.dlpack.from_dlpack`, legacy and versioned capsules) zero-copy; `:2200` IntEnum device / numpy.bool_ | `core_cast_exit_gpu.py bool` **rc 139**, `index` **rc 139**, controls rc 0 / 0; `core_host_cpu.py casts_exit bool` **rc -6**, `index` **rc -6** (regenerated from acd42fa: slices = the reviewer's pins), controls rc 0 | `core_cast_exit_gpu.py bool` **rc 0**, `index` **rc 0** (`EXPORT ended`, refused, method never called), controls rc 0; `core_host_cpu.py casts_exit bool` **rc 0**, `index` **rc 0** (regenerated from this tree), controls rc 0 |
| **V6-03** (MINOR): a transient failure of synchronize()'s own drain disappeared when its message allocation failed | `tc_signal/src/dlpack_operand.cpp`: `:729` `SyncReport` (moved above `PendingState`; `merge` `:754`, saturating, allocation-free); `:830` `PendingState::drain_unreported` + `drain_unreported_calls`; `:1364` persisted BEFORE the first allocation point; `:1421` the Python RuntimeError is created and captured (`py::error_already_set`) before `clear_reported_records` (`:1338`) runs; a bad_alloc anywhere earlier keeps every record. New stats key `unreported_drain_failures`; test-hook sites `cuda:current_device` (`:1031`), `cuda:event_query` (`:1048`, `:1263`) | `test_fix6_findings.py:457` real module, sync / current-device / event-query failure + message OOM: MemoryError, then the original error raised once, then success; `:404` the same three on the host trace module; corrected FIX2 `test_n3_...[alloc:error_message]` (`test_fix2_findings.py:393`) | `release_trace_cpu.py diagnostic_oom_query` **rc 1**, `_sync` **rc 1**, `_device` **rc 1** (`second synchronize returned success`) | all three **rc 0** (`PASS retained error`) |

## G2: sweep (g), user Python run from inside a native frame

Scope: every pybind11 binding of the P1-T surface (`git diff 9df5bef..HEAD -- tensor_cuda tc_signal`) and every Python
call made from a native frame of that surface. Mechanism (ledger "07:32-07:40 Diagnosis receipts"; receipts `artifacts/p1t_fix6/` and
`/tmp/p1t_fix6_scratch/exp/`): user Python that releases the GIL on a daemon thread across the start of finalization is
exited by CPython 3.12 at the GIL re-take (pthread_exit, a forced unwind); the unwind runs the C++ cleanup of every
frame above it; a Python object whose last reference is owned by such a frame is deallocated with the thread's state
freed: SIGSEGV. Measured on acd42fa: `_C.empty(shape=SeqLen())` rc 139 2/2 (the dispatcher's private kwargs copy),
positional rc 0 2/2, no-match `repr()` rc 0 2/2; gdb on `core_cast_exit_gpu.py bool`: the faulting frame is the
dispatcher's unwind cleanup. The verdict rule is the order's: admitted (and waited for at exit), refused before it is
invoked, or harmless (no live py::object or C++ cleanup on the native stack across a GIL re-take).

How each binding is fixed (one design, applied everywhere): every P1-T binding is declared `(*args, **kwargs)`, for
which pybind11 v2.12 copies, converts and looks up nothing and cannot fail to match (scratch module
`/tmp/p1t_fix6_scratch/pbtest`: a method's self arrives as `args[0]`; zero arguments still dispatch), and `CallArgs`
(`tc_signal/src/py_args.h:65`) / `CoreCallArgs` (`tensor_cuda/src/bindings.cpp:1256`) match positions and keyword
names with `PyUnicode_CompareWithASCIIString` and accept only types whose value is read without calling a Python method
(int and subclasses; float and subclasses; bool; tuple / list read with the C macros; str). A refusal names the
parameter and `tp_name`. The two bodies that wait with the GIL released (the exit waits) are raw CPython functions,
because pybind11 always leaves a dispatcher-owned object on their stack (row g10).

| # | Binding / site (file:line) | Argument conversions or calls that can run user Python | Verdict | Test / reason |
|---|---|---|---|---|
| g1 | core `Tensor.__dlpack__` (`bindings.cpp:1515`, body `:542`) | body: `copy.cast<bool>()` -> `__bool__`; `max_version` / `dl_device` `.cast<py::sequence>()`, `py::len`, `[i].cast<int>()` -> `__len__`, `__getitem__`, `__iter__`, `__index__`; `stream.cast<long long>()` -> `__int__` on overflow. Dispatcher: named-keyword lookups (str-subclass `__eq__`), its kwargs copy (the crashing object), no-match `repr()` | **FIXED** (V6-02) | `test_fix6_v602_*` (4 tests, 48 cases); probes `core_cast_exit_gpu.py`, `core_host_cpu.py casts_exit` |
| g2 | core `Tensor.__dlpack_device__` (`:1526`) | self caster: a non-Tensor self reaches pybind11's foreign-module lookup `hasattr(type(self), ...)` (a Python metaclass `__getattr__`); no-match `repr()` | **FIXED** (sweep g) | `test_fix6_sweep_g_core_bindings_refuse_foreign_arguments_without_calling_them` (`:2306`), `..._arity_and_keyword_errors_never_call_repr` (`:2337`) |
| g3 | core `Tensor.data_ptr` (`:1532`) | as g2 | **FIXED** | as g2 |
| g4 | core `Tensor.fill_(value)` (`:1540`) | `double` caster -> `__float__` / `__index__`; self as g2; with `value=` a dispatcher kwargs copy | **FIXED** | as g2; `test_fix6_sweep_g_core_keyword_conversion_at_exit_is_refused_not_run[fill]` (`:2404`) |
| g5 | core `Tensor.mark_modified` (`:1562`) | as g2 | **FIXED** | as g2 |
| g6 | core `Tensor.revision` getter (`:1570`) | as g2 (`Tensor.revision.fget(obj)`) | **FIXED** | as g2 |
| g7 | core `_C._from_dlpack_capsule` (`:2328`; body `:974`) | body: capsule C API only (no user Python); dispatcher: no-match `repr()` | **FIXED** (dispatcher) | `..._arity_and_keyword_errors_never_call_repr`; body: reasoning (source read) |
| g8 | core `_C.empty(shape, dtype, device)` (`:2334`) | `std::vector<int64_t>` caster -> `__len__`, `__getitem__` / `__iter__`, items' `__index__`; with keywords the crashing kwargs copy (`shape=SeqLen()` rc 139 on acd42fa); str casters C-level | **FIXED** | `..._refuse_foreign_arguments...`; `..._keyword_conversion_at_exit...[empty]` |
| g9 | core `_C._dlpack_release_stats()` (`:2345`) | no arguments; no-match `repr()` of stray arguments | **FIXED** | `..._arity_and_keyword_errors_never_call_repr` |
| g10 | core `_C._dlpack_begin_shutdown(first_diag_s)` (`:2404`, raw `:1430`) | `double` caster -> `__float__`; its BODY waits with the GIL released, so CPython can exit a non-main caller at the re-take: FIX5's typed binding owned a `PyDict_Copy` when called with a keyword; a `(*args, **kwargs)` binding owns a new empty kwargs dict when called positionally (my own interim rebinding: `test_fix5_sweep_c_core_exit_wait_on_a_non_main_thread_never_aborts` rc -11 in 3 of 5 runs, ledger 08:04-08:28) | **FIXED**: raw CPython function, caller-owned args, C-level real | `test_fix5_sweep_c_core_exit_wait_on_a_non_main_thread_never_aborts[0-4]` 5/5; `..._keyword_conversion_at_exit...[shutdown]`; `..._refuse_foreign_arguments...` |
| g11 | core `_C._dlpack_raise_unraisable()` (new, `:2391`) | none | **OK** (by construction: `CoreCallArgs` with no parameters) | `..._arity_and_keyword_errors_never_call_repr` |
| g12 | core `tensor_cuda.from_dlpack(obj)` (Python) | `obj.__dlpack_device__()`, `int(dev_type)`, `obj.__dlpack__(...)`: Python frames only, no native frame below them | **OK** | reasoning (a forced unwind through the eval loop's C frames runs no C++ cleanup); `_C._from_dlpack_capsule` is g7 |
| g13 | core `tensor_cuda.empty(*shape)`, `synchronize()`, `_dlpack_shutdown_hook()` (Python) | `operator.index(s)` in Python; none | **OK** | reasoning; their `_C` calls are g8, g9-g11 |
| g14 | core import release `import_release_impl` (`:840`) | the producer deleter (any Python), PyGILState_Ensure | **OK** (admitted, waited for at exit: FIX2 N5 / FIX3 V3-01) | FIX2/FIX3 tests; probes `core_exit_overrun.py`, `core_shutdown.py` |
| g15 | core exported-capsule destructors / export deleters (`:368`) | C++ only; an imported Storage released through them is g14 | **OK** | FIX1 F06 tests |
| g16 | decrefs in core P1-T frames (`CoreCallArgs::object`, returned self, capsule results) | references the caller also owns; no last-reference drop | **OK** | reasoning |
| g17 | add-on plan / run bindings (20: conv1d, conv_transpose1d, lstm, instance / layer / group norm, upsample_nearest1d, reflection_pad1d, rint; `bindings.cpp:392-461`, `lstm_bindings.cpp:251-263`, `m4_bindings.cpp:360-461`) | `int64_t` casters -> `__index__`; `double` (eps) -> `__float__` / `__index__`; with keywords the dispatcher kwargs copy; no-match `repr()`. Operands (py::object) only reach Python inside to_capsule (g22) | **FIXED** | `test_fix6_sweep_g_typed_parameters_refuse_foreign_objects_without_calling_them` (21 bindings, `:551`), `..._arity_and_keyword_errors_never_call_repr` (29 bindings, `:592`), `..._keyword_conversion_at_exit...` (`:659`) |
| g18 | add-on `fp32_divide_probe`, `_test_rint_then_raise` (operands only) | no-match `repr()` | **FIXED** | `..._arity_and_keyword_errors_never_call_repr` |
| g19 | add-on `cublas_config(device)`, `lstm_cublas_config(device)` | `int` caster -> `__index__` | **FIXED** | `..._typed_parameters...[cublas_config, lstm_cublas_config]` |
| g20 | add-on `pending_stats(reset_high_water)`, `_test_inject(names)`, `build_info()`, `m4_tuning()`, `_test_injection_armed()`, `synchronize()` | `bool` caster -> `__bool__`; `std::vector<std::string>` caster -> `__len__`, `__getitem__`; no-match `repr()` | **FIXED** | `..._typed_parameters...[pending_stats, _test_inject]`, `test_fix6_sweep_g_test_inject_reads_a_list_without_calling_methods` (`:580`), `..._keyword_conversion_at_exit...[pending_stats]` |
| g21 | add-on `_exit_close_admission(first_diag_s)` (raw, `bindings.cpp:352`) | `double` caster; a GIL-releasing body (as g10) | **FIXED**: raw CPython function | `..._typed_parameters...[_exit_close_admission]`, `..._keyword_conversion_at_exit...[close]`; FIX5 exit census tests |
| g22 | add-on `to_capsule` producer calls (`dlpack_operand.cpp`, `CallbackSection`) | attribute lookups, `__dlpack_device__`, its tuple's item casts, `__dlpack__`, a non-capsule result's `__del__` | **OK** (admitted kCallback, waited for at exit; FIX5 c3) | FIX5 census tests |
| g23 | add-on deleter calls (`call_deleter_contained`) | producer deleter (any Python) | **OK** (admitted kRelease; FIX5 c1) | FIX5 census tests; `test_fix6_exit_hook_still_waits...[deleter]` |
| g24 | add-on `CapsuleRef::drop` (`dlpack_operand.cpp:340`) | last reference of an unconsumed capsule: admitted (FIX5); last reference of a CONSUMED capsule: the producer's capsule destructor ran UNADMITTED (it is producer code, a ctypes / Cython destructor runs Python) | **FIXED** (sweep g): every last-reference capsule drop admitted, kept after close | `test_fix6_sweep_g_consumed_capsule_drop_is_an_admitted_release` (`:667`; acd42fa: destructor runs with 0 releases in flight), `test_fix6_exit_hook_still_waits...[consumed_capsule_drop]` |
| g25 | add-on decrefs of non-last references (returned tuples, `fn` / `cap` inside to_capsule's section, `CallArgs::object`) | none reaches zero outside an admitted section | **OK** | reasoning (source read of the 29 bindings) |
| g26 | add-on Python wrappers (`conv1d`, `conv_transpose1d`, `lstm`, norms, small ops: `_int`, `_eps`, `_pads`, the allocator call), `set_allocator`, `_release_pending_at_exit` | user `__index__` / `__float__` / `__len__` / allocator: Python frames only | **OK** | reasoning; they pass exact ints / floats to `_C` (operator.index, float) |
| g27 | pybind11 dispatcher internals for every P1-T binding (no-match `repr()` / `str.format`; named-keyword dict lookups; self foreign-module lookup; kwargs copy / empty-dict creation) | as listed | **FIXED** by the `(*args, **kwargs)` + raw-function design (no named parameter, no typed caster, no possible mismatch) | g1-g21 tests |

Summary (g): 27 rows (g1-g16 core: 11 bindings and 5 other call sites; g17-g26 add-on: 29 bindings in g17-g21 plus
their callbacks; g27 the dispatcher internals of all of them): **FIXED 17** (g1-g10, g17-g21, g24, g27), **OK 10**
(g11-g16, g22, g23, g25, g26). Every FIXED row has a test.

## G2: sweep (h), waits that can deadlock against caller-held locks

Every wait in the P1-T code that blocks on another thread or on the device (inventory: grep of `wait_until`, `lock_guard`,
`unique_lock`, `sleep_for`, `PyEval_RestoreThread`, `PyGILState_Ensure`, `cudaStreamSynchronize`,
`cudaDeviceSynchronize` over `tc_signal/src` and over the P1-T lines of `tensor_cuda/src` (`git diff 9df5bef`)).

| # | Wait (file:line) | Blocks on | Caller-held locks a callback on the other thread could need | Ordinary call or exit? | Verdict | Test / reason |
|---|---|---|---|---|---|---|
| h1 | tc_signal `synchronize()` -> FIX5's `wait_for_other_threads(false, ...)` (FIX5 `dlpack_operand.cpp:1284-1291`) | other threads' admitted releases (their producer deleters) | any: the reviewer's producer allocator lock | ordinary | **FIXED** (V6-01: removed) | `test_fix6_v601_*`, mutant (a) |
| h2 | tc_signal `close_admission_and_wait` -> `wait_without_gil` cv wait (`:274-292`) | admitted sections of other threads | the exiting thread's own locks | exit only | **OK** (the order permits only the exit hooks to wait; contract documented: the exiting thread must not hold a lock an admitted callback needs, as for CPython's join of non-daemon threads) | FIX5 census + `test_fix6_exit_hook_still_waits...` |
| h3 | tc_signal `wait_without_gil` polling fallback (`:297`) | same as h2 (only if the mutex fails) | same | exit only | **OK** | reasoning |
| h4 | tc_signal `wait_for_other_threads` `PyEval_RestoreThread` (`:317`) | the GIL | the waiter holds no lock (mutex released first) | exit only | **OK** | FIX5 c4 |
| h5 | tc_signal `leave_count` `lock_guard(lc.mu)` (`:210`) | the exit waiter's mutex | none: the critical section is empty; the waiter holds the mutex only to evaluate atomics (cv wait and diagnostics drop it) | ordinary | **OK** | reasoning (no callback, no GIL wait under lc.mu) |
| h6 | tc_signal `sync_device_legacy` `cudaStreamSynchronize(legacy)` (`:850`; synchronize, bound drain, fallback) | the device's legacy stream | none of P1-T's: P1-T enqueues no host function (grep: no `cudaLaunchHostFunc` / `cudaStreamAddCallback`) | ordinary | **OK** (a wait on the GPU, not on a host thread; see "Not a defect" N5 for host functions the application itself enqueues) | reasoning + grep |
| h7 | core `import_release_impl` `lock_guard(lc.mu)` x2 (`:840` body) | the exit waiter's mutex | none: no Python, no GIL wait under it | ordinary (Storage release) | **OK** | FIX2 N5 design |
| h8 | core `import_release_impl` `PyGILState_Ensure` | the GIL | only on a thread that does not hold the GIL: core releases the GIL nowhere in engine code (grep: only the exit wait), so every P1-T Storage drop holds it and the call is reentrant | ordinary | **OK** | reasoning + grep; the DLPack Python spec puts the GIL on the deleter (citation file) |
| h9 | core `dlpack_begin_shutdown` cv wait (`:899` body) + `PyEval_RestoreThread` | admitted import releases of other threads; the GIL | the exiting thread's own locks | exit only | **OK** (as h2, documented in `tensor_cuda._dlpack_shutdown_hook`) | FIX3 / FIX5 tests |
| h10 | core `_dlpack_release_stats` `lock_guard(lc.mu)` with the GIL held | the mutex | none: no thread waits for the GIL while holding lc.mu | ordinary (stats) | **OK** | FIX2 note at the site |
| h11 | core record mutex `r.mu` in `note_core_unraisable` / `dlpack_first_unraisable` / `dlpack_unreported_unraisable` / `dlpack_ack_unraisable` (`kernels.cu:255`, `:391`, `:400`) | the record's mutex | none: brief, no callback, no GIL wait under it | ordinary | **OK** (two new users in FIX6) | reasoning |
| h12 | core external Storage drain `cudaStreamSynchronize(0)` (`kernels.cu` `drain_device_legacy_stream`) | the Storage device's stream | as h6 | ordinary (last reference of an imported tensor) | **OK** | as h6 |
| h13 | core `__dlpack__` fence `cudaStreamWaitEvent` | nothing on the host (a device-side wait) | none | ordinary | **OK** | reasoning |
| h14 | add-on / core producer deleters run on the calling thread (next-call drain, synchronize, refusal; Storage release) | not a wait: a synchronous call | a non-reentrant lock the caller holds and the deleter needs would self-deadlock | ordinary | **OK** (not a wait on another thread; see "Not a defect" N6) | DLPack spec citation |

Summary (h): 14 rows: **FIXED 1** (h1, tested, mutant (a) killed), **OK 13**. After FIX6 no ordinary call waits for
another thread's user callback; the two exit hooks are the only cross-thread waits, documented.

## FIX5's sweep (c) rows that (g) changes, corrected

| FIX5 row | FIX5 said | Corrected (FIX6) |
|---|---|---|
| c2 `CapsuleRef::drop` | admitted for the last reference of an UNCONSUMED capsule; "any other drop is a plain decref (no destructor that releases)" | Incomplete: the last reference of a CONSUMED capsule still runs the producer's destructor (producer code, possibly Python) unadmitted. **FIXED by FIX6** (g24): every last-reference capsule drop is admitted or kept |
| c4 `wait_for_other_threads` (synchronize `:1285`; exit `:1075`) | GIL re-take explicit; "designed so" | synchronize()'s use was the V6-01 deadlock: **removed**. The exit use remains, now behind a raw CPython function (g21), because a pybind11 dispatcher frame below the GIL re-take owned a Python object |
| c6 other py::object decrefs in add-on frames | OK | Incomplete: it did not consider the dispatcher's typed CONVERSIONS, which run user Python before the body. **FIXED by FIX6** (g17-g21) |
| c10 core `dlpack_begin_shutdown` | FIXED (explicit restore) | Still true for the restore; but its pybind11 binding owned a `PyDict_Copy` when called with a keyword (crash on a non-main thread at exit). **FIXED by FIX6** (g10): raw CPython function |
| c11 core `tensor_dlpack` argument casts | **OK** ("a forced unwind propagates to pybind11's rethrow"; residual R3) | **Wrong**: the unwind deallocates the dispatcher's kwargs copy (SIGSEGV; V6-02). **FIXED by FIX6** (g1) |
| k1 catch-all in `call_deleter_contained`, "remaining origin: the producer's own pthread_exit (residual R2)" | a residual | Not a residual: **Not a defect in our code**, N1 below (cited, measured on torch and NumPy) |

## Not a defect (with justification)

Each row was a FIX5 residual or a sweep finding that stays as is because it is not a defect in our code. Citations are
saved with the receipts.

| # | Item | Why it is not a defect in our code | Citation / receipt |
|---|---|---|---|
| N1 | FIX5 R2: a producer deleter that calls `pthread_exit` (or is cancelled) aborts the process in tc_signal's / core's catch-all | The DLPack deleter is a destructor that the consumer calls from its own destructor and that must return; terminating the calling thread inside it cannot be survived by ANY consumer. Measured with the same C deleter on a worker thread: torch 2.11 consumer rc 134 (`terminate called without an active exception`), NumPy 2.4.2 consumer rc 124 (the thread exits holding the GIL; the interpreter hangs until the 20 s timeout), tc_signal rc 134 (`FATAL: exception not rethrown`). Rethrowing instead would kill the thread while it holds the GIL (NumPy's outcome) | DLPack Python spec ("the deleter of the DLManagedTensor will be called by the destructor of the consumer library object", `artifacts/p1t_fix6/citations/dlpack_python_spec_2026-10-13.txt`); vendored `dlpack.h` v1.3 `:302-307` ("Destructor ... The destructor deletes the argument self as well") and `:397` ("As a C function, must not thrown C++ exceptions"); libstdc++ 13 `bits/cxxabi_forced.h` (`__forced_unwind`, "Thrown as part of forced unwinding"); receipts `artifacts/p1t_fix6/r2_receipt/` |
| N2 | FIX5 R5: a producer callback that never returns makes interpreter exit wait | Same contract (a destructor must return; `del t` on a torch tensor with such a deleter blocks forever too). The exit wait is the order's sanctioned design ("only the exit hooks may [wait], and their wait is documented"); the alternative, returning early, aborts the process (round-3 V3-01, round-5 V5-01 receipts). CPython's own exit joins non-daemon threads with no bound | DLPack Python spec (as N1); Python `threading` docs ("The entire Python program exits when no alive non-daemon threads are left"); this order's (h) text |
| N3 | FIX5 R4: synchronize()'s count-based wait could be extended by overlapping releases | Removed with the wait itself (V6-01). The exit wait closes admission before it counts, so only sections admitted before the close can extend it (RCU-style grace period) | `test_fix6_v601_release_started_after_synchronize_began_is_not_waited_for`; FIX5 after-close tests |
| N4 | FIX5 R6: `atexit.unregister(tc_signal._release_pending_at_exit)` removes the exit protection | The application deliberately removes a documented shutdown hook (Python docs: `atexit.unregister` "Remove func from the list of functions to be run at interpreter shutdown"); the finalizing check still keeps every release that starts during finalization out of Python | Python `atexit` docs; README "Interpreter exit" |
| N5 | (sweep h) a host function the APPLICATION enqueued on the legacy stream that needs a lock the caller holds while it calls a synchronizing tc_signal / core API | P1-T enqueues no host function; a stream sync waits for whatever the application put on that stream, as `cudaStreamSynchronize` / `torch.cuda.synchronize` do; the lock order between the application's own host function and its own lock is the application's | CUDA 12.6 Runtime API `cudaLaunchHostFunc` (host functions are stream work, ordered with it; lead to check the exact sentence: "cudaLaunchHostFunc enqueued work ordering"); grep receipt (no host-callback launch in P1-T code, G5 check 2) |
| N6 | (sweep h) a producer deleter that needs a non-reentrant lock held by the thread that triggers the release | Deleters run when the consumer releases (DLPack spec: from the consumer's destructor), on that thread; torch's `del t` under the same lock deadlocks the same way. Not a wait on another thread | DLPack Python spec (as N1) |
| N7 | FIX5 R1 (two-GPU probes INCONCLUSIVE), R8 (new suites not run against the before binaries), R9 (performance not measured) | Test-environment limits and evidence notes, not code. R8 is covered by the before-arm probes and the three killed mutants | probe tables below |
| N8 | FIX5 R10: free-threaded CPython and subinterpreters | Not reachable configurations: an extension that does not declare `Py_mod_gil` re-enables the GIL in free-threaded CPython (PEP 703), and single-phase-init modules (pybind11 v2.12) are refused in isolated subinterpreters (PEP 489 / 684) | PEP 703, PEP 684 (lead to check exact sections: "Py_mod_gil re-enable GIL", "single-phase init subinterpreter ImportError") |
| N9 | FIX5 R7: core's recorded unraisable CUDA errors were raised by nothing | Listed only so every FIX5 residual has a disposition: it WAS a defect and is **FIXED** (tensor_cuda.synchronize() raises it once; the exit hook logs it; `test_fix6_r7_*`, corrected FIX4 / FIX5 fill tests) | `tensor_cuda/__init__.py:1371`, `bindings.cpp:1466`, `kernels.cu:391-420` |

## Probe before/after table

Every run a fresh process under `flock -w 900 /tmp/forge-gpu.lock timeout <probe bound>`. nogpu = `CUDA_VISIBLE_DEVICES=`.
Round 6 (`artifacts/p1t_fix6/run_rev6_probes.sh`; logs `probe_runs/rev6_before/`, `probe_runs/rev6_after_final/`;
summaries `probe_runs_rev6_{before,after_final}_summary.txt`; the after column is the FINAL build, deviation D10):

| Probe (round 6) | How it ran | Before (acd42fa) | After (this build) | Result |
|---|---|---|---|---|
| `sync_lock_inversion_cpu.py` nogpu | unmodified; the arm's real package | **rc 1** `BLOCKED_BEFORE_CUDA=1`, rescue | **rc 0** (RuntimeError cudaErrorNoDevice raised, no wait) | **FIXED** |
| `sync_lock_inversion_cpu.py` gpu | unmodified | **rc 1** `elapsed=1.251 rescue=True` | **rc 0** `elapsed=0.222 rescue=False` | **FIXED** |
| `sync_lock` (run_cpu_checks job) | path-only copy of the driver (D8) | **rc 1** | **rc 0** | **FIXED** |
| `sync_late_release_cpu.py` | trace module built from each arm's `dlpack_operand.cpp` | **rc 1** `waited_for_later=True` (0.703 s) | **rc 0** `waited_for_later=False` (0.000 s) | PASS |
| `core_cast_exit_gpu.py bool` | unmodified, real core, one GPU | **rc 139** (core dumped) | **rc 0** `EXPORT ended` (refused, never called) | **FIXED** |
| `core_cast_exit_gpu.py index` | unmodified | **rc 139** | **rc 0** | **FIXED** |
| `core_cast_exit_gpu.py bool / index --control` | unmodified | rc 0 / rc 0 | rc 0 / rc 0 | PASS |
| `core_host_cpu.py casts_exit bool` | regenerated from each arm (self-check: the 4 slices from acd42fa equal the reviewer's pinned hashes, `slices_selfcheck_before.txt`) | **rc -6** (`handle::dec_ref() PyGILState_Check() failure` in the exit unwind; the reviewer saw -11) | **rc 0** | **FIXED** |
| `core_host_cpu.py casts_exit index` | regenerated | **rc -6** | **rc 0** | **FIXED** |
| `core_host_cpu.py casts_exit bool / index --control` | regenerated | rc 0 / rc 0 | rc 0 / rc 0 | PASS |
| `core_host_cpu.py arithmetic` | regenerated | rc 0 | rc 0 | PASS |
| `release_trace_cpu.py diagnostic_oom_query` | trace module from each arm | **rc 1** `second synchronize returned success` | **rc 0** `PASS retained error` | **FIXED** |
| `release_trace_cpu.py diagnostic_oom_sync` | same | **rc 1** | **rc 0** | **FIXED** |
| `release_trace_cpu.py diagnostic_oom_device` | same | **rc 1** | **rc 0** | **FIXED** |
| `release_trace_cpu.py` query_once / multidevice_restore / prior_oom / inside_callback / deferred_exit | same | rc 0 x5 | rc 0 x5 | PASS |
| fix2_harness ck_census, fix3_harness accounting, cpu_native oom_sync / throw_deleter, operand_census | path-only copies of the build and driver | rc 0 x5 | rc 0 x5 | PASS |

Round 5 (`run_rev5_probes.sh`, FIX5's runner retargeted; `rev5_summary_{before,after}.txt`):

| Probe (round 5) | How it ran | Before (acd42fa) | After (this build) | Result |
|---|---|---|---|---|
| `exit_gate.py full` nogpu / gpu, each with `--control` | unmodified | rc 0 x4 | rc 0 x4 | PASS |
| `exit_gate.py trace` | trace module from each arm (FIX5 D5 build) | rc 0 | **rc 1** (child -6, `FATAL: exception not rethrown`): the child's hook is `synchronize()` alone, which no longer waits | **Required fix 1** (anticipated by the order; D7) |
| `exit_gate.py trace --control` | same (the child ignores `--control`, FIX5 D4) | rc 0 | **rc 1** | as above |
| `exit_gate.py trace`, path-only copy whose hook calls `synchronize(); close(5.0)` (the real exit entry) | one added binding in a copy of addon_trace.cpp; diff `trace_exit_entry_copy_*.txt` | rc 0 | **rc 0** `DELETER finished` before `FINALIZING True` | PASS (deviation D9) |
| `exit_gate.py gpu` and `--control` | unmodified | rc 0 / rc 0 | rc 0 / rc 0 (the daemon's later producer call is refused after the close, by V5-01's rule; the probe PASSes) | PASS |
| `query_error_cpu.py`, `query_error_single_gpu.py` (shim byte-identical to a fresh build) | as FIX5 | rc 0 / rc 0 | rc 0 / rc 0 | PASS |
| `handle_insert_oom_cpu.cpp` as written / regenerated | as FIX5 (its D3) | rc 1 / rc 0 | rc 1 / rc 0 | as FIX5 (arm-independent as written) |
| `fill_grid_cpu.cpp` as written / regenerated; `fill_grid_single_gpu.py` | as FIX5 | rc 1 / rc 0; rc 0 | rc 1 / rc 0; rc 0 | as FIX5 |
| `device_error_sweep_single_gpu.py` | path copy (root, basetemp) | rc 0, 40 passed | rc 0, 40 passed | PASS |
| `core_export_device.py`, `core_device_drain.py`, `cuda_errors_multidevice.py` | unmodified | exit 3 x3 | exit 3 x3 | INCONCLUSIVE (one GPU) |
| `core_import_query_error.py`, `sync_existing_error.py`, `core_edges.py`, `numerics_and_streams.py` | unmodified | rc 0 x4 | rc 0 x4 | PASS |
| `addon_lifetime.py` | unmodified | rc 0, 12/12 | rc 0, 12/12, 0 incorrect | PASS |
| `core_exit_overrun.py during / bounded` | unmodified | rc 0 (3.4 s) / rc 0 (7.8 s) | rc 0 (3.2 s) / rc 0 (7.9 s) | PASS |
| `release_failures.py` | before: path copy (root wt_fix5); after: original | rc 0, 66 passed | rc 0, 66 passed | PASS |
| `build_cpu.sh` | path copy (build dir) | rc 0 | rc 0 | PASS |

Round 4 (`run_rev4_cpu_probes.sh` retargeted; both arms; `rev4_summary_{before,after}.txt`): identical on both arms.
host_checks 20/20 cases identical after path normalization, nogpu and gpu (`rev4_host_checks_compare.txt`);
fix2_harness ck_census 18 cases / 0 failures and fix3_harness accounting 17 / 0, no UBSan line; lifecycle_timeout
control / short / default rc 0 in all 12 arm-condition runs; operand_census PASS 17 cases; real_metadata_plans PASS 129;
sync_slot_cpu rc 1 on both arms (as FIX5; the after arm's message now starts `RuntimeError: `, D11); core_interop as
written reproduces its embedded 62c8d9f traces on both arms (arm-independent); its import regenerated from this tree
refuses (rc 1); core_import_query_error rc 0.

Rounds 1-3 (`run_rounds123_probes.sh` retargeted; this build; `round123_summary_after.txt`): `addon_lifetime` runs 1-3
rc 0, 12/12 correct, 0 incorrect each; `core_edges` 0; `core_shutdown --normal` 0, finalizing 0; `numerics_and_streams`
0; `core_device_drain` 3 (one GPU); round-3 `core_exit_overrun` during 0 (3.4 s), bounded 0 (7.8 s); round-1 and
round-2 `host_checks` (nogpu and gpu) identical to FIX5's receipts after normalization (`round12_host_checks_vs_fix5.txt`).

## Gate table

| Gate | Criterion (order; registered in the ledger at 07:21 before any change) | Result | Receipt |
|---|---|---|---|
| G1 | `sync_lock_inversion_cpu.py` both conditions fail on acd42fa, pass here | **PASS** rc 1 -> 0, rc 1 -> 0 | `probe_runs/rev6_*` |
| G1 | `core_cast_exit_gpu.py bool` and `index` | **PASS** rc 139 -> 0, rc 139 -> 0 | same |
| G1 | `core_host_cpu.py casts_exit bool` and `index`, regenerated | **PASS** rc -6 -> 0, rc -6 -> 0 (self-check: acd42fa slices = the reviewer's pins) | same |
| G1 | `release_trace_cpu.py diagnostic_oom_query`, `_sync`, `_device`, regenerated | **PASS** rc 1 -> 0 x3 | same |
| G1 | `sync_late_release_cpu.py` passes here | **PASS** rc 0 (acd42fa rc 1) | same |
| G1 | `--control` variants pass on both builds | **PASS** (4 controls rc 0 on both) | same |
| G1 | the new regression for each finding passes | **PASS**: tc_signal `test_fix6_findings.py` 67, core `test_fix6_*` 54, all in the combined session | `logs/p1t_fix6_g3_combined_session.log` |
| G2 | tables (g) and (h) complete, every site OK or FIXED, FIXED rows tested; FIX5 (c) rows corrected | **PASS**: (g) 27 rows FIXED 17 / OK 10; (h) 14 rows FIXED 1 / OK 13; c2, c4, c6, c10, c11, k1 corrected; "Not a defect" table cited | sections above |
| G3 | every round-1..5 probe passes or is INCONCLUSIVE only for lack of 2 GPUs; `exit_gate trace` handled as Required fix 1 | **PASS** (three two-GPU probes exit 3; `trace` as written rc 1, its exit-entry copy rc 0) | `rev5_summary_*`, `rev4_summary_*`, `round123_summary_after.txt` |
| G3 | `addon_lifetime` 3/3 runs, 0 incorrect | **PASS** | `probe_runs/rounds123_after/round1_addon_lifetime_run{1,2,3}.log` |
| G3 | `core_exit_overrun` during and bounded rc 0 | **PASS** (round 3 and round 5 copies) | same dirs |
| G3 | Voice T1-T6 + T1R + T4S PASS 249/249 | **PASS** | `logs/p1t_fix6_g3_voice.log`, `artifacts/p1t_fix6/p1t_all.json` |
| G3 | combined session = exactly the 52 baseline ids | **PASS**: 52 = 52, 0 new, 0 no longer failing | `logs/p1t_fix6_g3_compare.txt` |
| G3 | real-torch DLPack interop passes | **PASS** (`test_fix6_v602_real_torch_consumers_still_import_zero_copy`; Voice T1) | combined session |
| G4 | mutants (a), (b), (c) each fail their new test | **PASS**: all three killed | `artifacts/p1t_fix6/g4/` |
| G5 | diff vs 123610f only in Exercises + FIX6 docs / artifacts / logs; probe files byte-identical; no new core `__global__` | **PASS** | `artifacts/p1t_fix6/g5_scope.txt` |

**Voice's verdict line** (`artifacts/p1t_fix6/p1t_all.json` summary):
`verdict PASS exit_code 0 counts {"PASS": 249, "FAIL": 0, "SKIP": 0, "ERROR": 0} (T1 11, T2 144, T3 28, T4 12, T5 28, T6 13, T1R 1, T4S 12) seconds 68.1`.

**Compare output** (`logs/p1t_fix6_g3_compare.txt`):
```
G1 compare (APA-MF2.2): baseline = /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2/logs/apa_mf1_g1_baseline_unchanged.log; now = logs/p1t_fix6_g3_combined_session.log
summary: 37 failed, 2886 passed, 47 skipped, 2 warnings, 15 errors in 625.08s (0:10:25)
baseline ids 52 now ids 52
baseline ids that no longer fail: 0 []
new failing ids: 0 []
new MF2.2 test files with any FAILED/ERROR: 0
G1 PASS
```
Passed is 2886 against FIX5's 2765: tc_signal added 67 tests and core 54 (the corrected FIX2 / FIX4 / FIX5 tests
changed in place). The 37 failures and 15 errors are the baseline APA files.

## G4 mutant table

Trees `/tmp/p1t_fix6_mut_<m>/` from `artifacts/p1t_fix6/g4/make_mutants.py` (one asserted edit each; diffs
`g4/mut_<m>.diff`); runner `g4/run_mutant.sh` (each mutant's test files cmp-identical to the worktree's; trace modules and
core host slices built from the mutant's sources).

| Mutant | Edit | .so sha256 (the rebuilt module) | New tests on the mutant | Probes on the mutant | Verdict |
|---|---|---|---|---|---|
| (a) V6-01 wait restored | `synchronize_and_release_all`: `if (t_depth[...] == 0) (void)wait_for_other_threads(5.0);` before the drain (FIX5's placement) | tc_signal `eacd0df5b9cef7dc95a993c6ae6f76e84276c19763742d13eb122074417b813a` | `-k "v601 or exit_hook_still_waits or synchronize_does_not_wait"`: **6 failed** of 6 (lock inversion gpu / nogpu; late release; both exit-hook cases, where the close finds nothing in flight because synchronize already waited; the corrected FIX5 no-wait test) | `sync_lock_inversion` nogpu **rc 1**, gpu **rc 1** (rescue); `sync_late_release` **rc 1** (`waited_for_later=True`) | KILLED |
| (b) V6-02 refusal removed | `tensor_dlpack`'s C-level reading replaced by acd42fa's casts (`copy.cast<bool>()`, `cast<py::sequence>()` + `py::len` + item `cast<int>()`, `stream.cast<long long>()`) | core `436788fffb4b5a21f1050b62ef6057d5359bef78ea6259880e10ac1b02bfd4fc` | `-k fix6_v602`: **30 failed**, 17 passed (the passing ones are accepted-value cases that call no method) | regenerated `core_host_cpu casts_exit bool` / `index` **rc 134**; `core_cast_exit_gpu bool` / `index` rc 0 although the method ran (`CAST started`): the (*args, **kwargs) binding alone removes the dispatcher-owned kwargs copy whose deallocation crashed, an independent second layer (the tests detect the method running) | KILLED |
| (c) V6-03 persistence removed | the drain report merged into `drain_unreported` only AFTER the message test hook (the first allocation point), as acd42fa | tc_signal `a896fe33955a3651fc54c1ba91532745398289557e95cb33e77b501874554679` | `-k "v603 or (n3_allocation and error_message)"`: **7 failed** of 7 (3 host-trace modes, 3 real-module sites, the corrected FIX2 N3 case) | `release_trace_cpu diagnostic_oom_query` / `_sync` / `_device` **rc 1** x3 | KILLED |

## Prior art

- **V6-01**: Coffman, Elphick & Shoshani (1971) "System Deadlocks" (circular wait as a necessary condition) and
  lock-ordering discipline: an ordinary call never blocks on another thread's callback, so no cycle through a
  caller-held lock can form. RCU grace periods (McKenney & Slingwine 1998; synchronize_rcu waits only for readers that
  began before it): why a separate wait-for-releases API would need a pre-call generation boundary and a no-locks-held
  contract (none offered, no caller needs it). CPython's `threading._shutdown` (exit joins non-daemon threads): the model
  for the exit hook's documented contract. Taken: the wait-for graph argument and the exit-only placement. Ours: none
  beyond applying them.
- **V6-02 / sweep (g)**: the Python array API `__dlpack__` signature (Consortium for Python Data API Standards,
  2021-2023); PyTorch 2.11 `Tensor.__dlpack__` (`type(stream) is not int` -> TypeError, read in the installed
  `torch/_tensor.py`); CPython 3.12 `PyLong_AsLongLongAndOverflow` / `PyFloat_AsDouble` (no `__index__` / `__float__` for
  int / float subclasses), `PyTuple_GET_ITEM` / `PyList_GET_ITEM`, `PyUnicode_CompareWithASCIIString`, the `%U` format;
  CPython's C argument parser (keyword matching without `__eq__`; unverified, lead to check "CPython getargs.c
  find_keyword _PyUnicode_Equal"); CPython's METH_VARARGS | METH_KEYWORDS convention (caller-owned arguments) for the
  raw exit-wait functions; pybind11 v2.12 `py::args` / `py::kwargs` and its dispatcher (`pybind11.h:717-1160`, read
  here); CPython 3.12 finalization and GIL rules (`Py_FinalizeEx`, `PyEval_RestoreThread` -> `take_gil` ->
  `PyThread_exit_thread`; unverified, lead to check "CPython 3.12 ceval_gil.c take_gil _PyThreadState_MustExit");
  glibc / libstdc++ forced unwinding (`bits/cxxabi_forced.h`, read here). Taken: refusal instead of conversion,
  C-level reading, caller-owned arguments. Ours: the mechanism finding (a dispatcher-owned last reference is what
  crashes), refusing for every binding, and the raw functions for the GIL-releasing waits.
- **V6-03**: SQLite's OOM testing discipline (an error must survive the failure of its own report; unverified, lead to
  check "SQLite malloc failure testing anomaly"); ordinary commit-after-side-effect ordering. Ours: capturing the Python
  RuntimeError (`PyErr_SetString`, `py::error_already_set`) before clearing the records.
- **FIX5 R7 (core record raised once)**: tc_signal's own N1 / V6-03 raise-once rule; PyTorch's `C10_CUDA_CHECK_WARN`
  (an error found in a destructor is reported, never dropped). Ours: deferring the raise to `tensor_cuda.synchronize()`.
- **Testing**: the round-6 reviewer's probes (scenarios reused); LD_PRELOAD interposition and fail-the-next-allocation
  injection (glibc ld.so, SQLite, Linux failslab); DeMillo, Lipton & Sayward (1978) mutation testing. Ours: the
  all-methods recording object and the per-binding refusal census. No prior art known to me for the recording object.
- These annotations are also at the code sites (`dlpack_operand.cpp` V6-01 / V6-03 blocks, `py_args.h`,
  `bindings.cpp` V6-02 / sweep (g) / raw-function / R7 blocks, `kernels.cu` R7 block) and in the ledger.

## Deviations and user-visible behavior changes (each with its reason)

User-visible changes, each justified as correct (lead directive):
- **U1. `tc_signal.synchronize()` no longer waits for other threads** (reverses FIX5's D1). A release that another
  thread's call already popped may still be running when synchronize() returns. Correct: that release belongs to the
  other thread's call (DLPack: the deleter runs in the releasing consumer's context); an ordinary call that waits for
  another thread's callback is a deadlock wherever the caller holds a lock the callback needs (V6-01, Coffman 1971), and
  the order forbids it; the exit hook still waits, so exit safety (V5-01) is unchanged. Stats keys
  `sync_waits_for_other_threads`, `sync_wait_us`, `sync_wait_diagnostics` removed with the wait;
  `unreported_drain_failures` added.
- **U2. `Tensor.__dlpack__` refuses argument types it cannot read without calling Python** (TypeError): numpy.bool_ for
  `copy`, numpy integer scalars (not int subclasses), non-tuple / non-list sequences, floats; `stream` beyond int64 is
  ValueError (was a pybind11 cast error). Correct: the array API types `stream` int | None, `max_version` tuple[int, int]
  | None, `dl_device` tuple | None, `copy` bool | None; PyTorch 2.11's own `__dlpack__` refuses a non-int stream the same
  way; every real consumer passes plain values (torch, NumPy, tc_signal, tensor_cuda: tests); reading any other type
  means running its Python methods on a native stack (V6-02's crash). Lists, int subclasses (bool, IntEnum) and tuple
  subclasses are still accepted (read at C level).
- **U3. Every P1-T binding (core `__dlpack__`, `__dlpack_device__`, `data_ptr`, `fill_`, `mark_modified`, `revision`,
  `_C.empty`, `_C._from_dlpack_capsule`, `_C._dlpack_*`; every `tc_signal._C` function) reads its arguments at C level**:
  ints must be int (or a subclass), reals float / int, flags bool / int, name lists list / tuple of str; others TypeError
  naming the parameter; out-of-range ints OverflowError (pybind11 said TypeError); arity / keyword errors are our own
  TypeError (no `repr()` of the arguments); `help()` shows `(*args, **kwargs)` with the real signature as the
  docstring's first line. `Tensor.fill_(np.float32(1))` is now refused (np.float64, a float subclass, is accepted).
  Correct: these conversions ran user Python on native stacks (sweep g; `_C.empty(shape=SeqLen())` rc 139 at exit on
  acd42fa); the public wrappers (`tc_signal.conv1d` ..., `tensor_cuda.empty`, `from_dlpack`) convert in Python first and
  behave exactly as before (Voice 249/249, combined session).
- **U4. After a MemoryError from `tc_signal.synchronize()`, the next call raises the CUDA failure that message would have
  named** (V6-03). Correct: a CUDA failure is never reported as success (FIX2 N1); it is reported exactly once.
- **U5. `tensor_cuda.synchronize()` raises, once, a CUDA failure recorded earlier on a P1-T core path that could not
  raise; the exit hook logs it** (FIX5 R7). Correct: the same N1 rule; before, such a failure (an imported tensor's
  device drain failed, its producer memory deliberately leaked) was visible only in a private stats dict.
- **U6. The exit hook closes admission without first waiting**, so a release that starts after the close (for example
  the second operand's deleter of a refused call whose first deleter was in flight) is leaked by V5-01's after-close rule
  more often than under FIX5. Correct: that is the documented rule; the alternative is the V6-01 wait.
- **U7. The last reference of a CONSUMED capsule is an admitted release** (kept, not dropped, after the close). Correct:
  its destructor is producer code (sweep g, g24); a consumed capsule kept at exit holds no producer memory.
- **U8. `_C._dlpack_begin_shutdown` and `tc_signal._C._exit_close_admission` are builtin (raw CPython) functions**, with
  the same names, arguments and return values. Correct: required so that no pybind11-owned object is on their stack
  across the GIL-released wait (g10 / g21).
- **U9. Two release-path test hooks added** (`cuda:current_device`, `cuda:event_query`; one relaxed atomic load per site
  in production, as FIX2's hooks). Needed for the V6-03 regression on the real module.

Process deviations:
- **D1. Lead directive applied mid-task** (ledger "~07:40"): residuals replaced by fixes (R7) or the cited "Not a
  defect" table (R2, R5, R6, environment items); R4 removed with the wait.
- **D2. Tests corrected in place** (they pinned the defects being fixed): FIX5 `test_fix5_findings.py` (5 census
  assertions moved from synchronize()'s removed wait to the exit hook's `exit_in_flight_at_close`, certain for the 5.5 s /
  12 s cases and timing-tolerant at 0.1 s where the deleter may finish during the hook's first CUDA call; the thrower
  census now asserts x's release leaked after the close; `test_fix5_v501_synchronize_waits...` inverted and renamed
  `..._does_not_wait...`; `_run` finds the RESULT marker anywhere in a line, because two threads' print() calls can
  interleave); FIX2 `test_n3_...[alloc:error_message]` (it asserted the drain failure was lost: V6-03); FIX4
  `test_fix4_fill_guard...` and FIX5 `test_fix5_sweep_d_core_fill_unwinding...` (their script called
  tensor_cuda.synchronize() expecting success after an unraisable failure: R7). Diffs
  `artifacts/p1t_fix6/test_fix5_corrections.diff`, `test_fix2_corrections.diff`; the test_dlpack changes are in the diff
  stat.
- **D3. My own regression, found and fixed before acceptance:** the interim `(*args, **kwargs)` rebinding of
  `_dlpack_begin_shutdown` (rc -11 in 3 of 5 runs of FIX5's non-main-thread exit test), fixed by the raw functions (U8).
  Also my own test mistakes, fixed before acceptance: a flag-overflow probe that expected an error a flag cannot raise;
  two accept-probes that would have closed admission or touched a missing GPU in-process (excluded); a G5 regex that
  matched a pre-existing docstring (narrowed to compiler flags, reason in the checker).
- **D4. V6-02 option (ii)** chosen and justified at the code site (`bindings.cpp:488`) and in U2.
- **D5. The sweep (g) fix is broader than V6-02's site** (every binding of both modules): the class sweep found the same
  defect in every typed parameter (dispatcher conversions), in pybind11's mismatch / keyword paths, and in FIX5's capsule
  drop (g24).
- **D6. R2 (pthread_exit deleter) not changed:** "Not a defect" N1, with citations and measurements (torch aborts, NumPy
  hangs, tc_signal aborts).
- **D7. `exit_gate.py trace` as written fails on this build** (order Required fix 1; the round-6 reviewer prescribes
  this design); the path-only copy calling the real exit entry passes (D9).
- **D8. Probe plumbing, all path-only or regeneration; the originals stay byte-identical (G5):** `make_rev6_copies.py`
  (build_host_probes.sh: bin dir and the trace source path; build_core_host.py: root, out dir, slice receipt;
  run_cpu_checks.py: root, probe dir, bin, logs, results, and the operand-census copy's result path, which would
  otherwise write `artifacts/p1t_rev6/operand_census.json`); the core host module's compile command reconstructed from
  the reviewer's binary (libcudart + libubsan: `-fsanitize=undefined ... -lcudart`); `<bin>/tc_signal` holds a
  byte-identical copy of the arm's built package (sha256 recorded) instead of the reviewer's CMake rebuild; FIX5's
  round-4 / round-5 / rounds-1..3 runners and helpers retargeted to this worktree and `/tmp/p1t_fix6_probes` (logic
  unchanged).
- **D9. The trace exit-entry copy:** `addon_trace.cpp` + one binding `m.def("close",&tcs::close_admission_and_wait)`;
  `addon_exit_cpu.py`'s hook `n.synchronize()` -> `n.synchronize(); n.close(5.0)`; `exit_gate.py` pointed at that child
  (diffs in `probe_runs/rev5_*/trace_exit_entry_copy_*.txt`).
- **D10. Round-6 after-arm receipts re-run on the final build:** the 08:03 run used my first build (core 55276e88...,
  tc_signal 672f22a3...); the 09:10 re-run on the final build (`probe_runs/rev6_after_final/`) is the receipt; the
  outcomes are identical.
- **D11. Message prefix:** `synchronize()` now raises a captured `py::error_already_set`, so a C++ caller's `what()` reads
  `RuntimeError: tc_signal.synchronize: ...` (round-4 `sync_slot_cpu` text; rc unchanged; Python sees the same
  RuntimeError message).
- **D12. The combined session ran detached** (its 625 s exceed my tool's 10-minute foreground ceiling), with the order's
  exact command and a completion marker, and was read only after the marker existed (FIX5 D9 precedent).
- **D13. Network:** the core build (`./build.sh 89`, as ordered) fetched pybind11 v2.12.0 through FetchContent (FIX3-5
  precedent). Mutant (b)'s core was built offline against production's pybind11 source (read-only). The reviewer's rev5
  shims were verified byte-identical to fresh builds and used read-only.
- **D14. Citations fetched through the AtlasForge web proxy** (the DLPack Python spec, saved under
  `artifacts/p1t_fix6/citations/`); the others are marked "lead to check" where I could not verify them.
- **D15. Ledger times** were first written as estimates and corrected from file mtimes (ledger note).
- **D16. Timing:** load average about 25 throughout (other users); wall times are not performance measurements.

**Scratch left (nothing deleted):** `/tmp/p1t_fix6_scratch` (experiments, gdb logs, test-file snapshots, r2 builds),
`/tmp/p1t_fix6_rev6/{before,after}` (round-6 builds), `/tmp/p1t_fix6_probes/{before,after,rev2_after}` (earlier-round
builds and copies), `/tmp/p1t_fix6_shims`, `/tmp/p1t_fix6_mut_{a,b,c}` (mutant trees and builds), and pytest temporary
directories under `/tmp/pytest-of-vader/`.

## Diff stat

Code and tests changed by this order (`git diff --stat acd42fa -- tensor_cuda tc_signal`, working tree), plus the
untracked new files:
```
 tc_signal/README.md                   |  55 +++-
 tc_signal/src/bindings.cpp            | 212 ++++++++++----
 tc_signal/src/dlpack_operand.cpp      | 277 +++++++++++-------
 tc_signal/src/dlpack_operand.h        |  21 +-
 tc_signal/src/lstm_bindings.cpp       |  54 ++--
 tc_signal/src/m4_bindings.cpp         | 149 +++++++---
 tc_signal/src/tuning.h                |   8 +-
 tc_signal/tc_signal/__init__.py       |  21 +-
 tc_signal/tests/test_fix2_findings.py |   9 +
 tc_signal/tests/test_fix5_findings.py |  43 ++-
 tensor_cuda/src/bindings.cpp          | 525 ++++++++++++++++++++++++++++++----
 tensor_cuda/src/kernels.cu            |  47 +++
 tensor_cuda/tensor_cuda/__init__.py   |  18 ++
 tensor_cuda/tests/test_dlpack.py      | 405 ++++++++++++++++++++++++++
 14 files changed, 1541 insertions(+), 303 deletions(-)
untracked: docs/P1T_FIX6_LEDGER.md (139 lines)
untracked: docs/P1T_FIX6_REPORT.md (170 lines)
untracked: tc_signal/src/py_args.h (250 lines)
untracked: tc_signal/tests/test_fix6_findings.py (703 lines)
```
`git diff --stat 123610f` (tracked, all paths): 15 files changed, 1788 insertions(+), 303 deletions(-) (the 14 above plus
the lead's order file on HEAD). Untracked new files also include `artifacts/p1t_fix6/**` and `logs/p1t_fix6_*`
(`.gitignore` ignores `*.log` / `*.json`: those receipts need `git add -f`).

## .so sha256 (final build)

- core `tensor_cuda/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so`:
  `e7d4a3a313562d80af0401f1deba819260fa95a6ea86abb3a53443c3e17b5a10`
- `tc_signal/tc_signal/_tc_signal.cpython-312-x86_64-linux-gnu.so`:
  `ef44b9c455f26a7b40d3429d70b7e08ebd4b35444689811de90428e063f2f197`

## Changed files

- Core: `tensor_cuda/src/bindings.cpp` (V6-02, CoreCallArgs and the rebinding, raw exit wait, R7 raise),
  `tensor_cuda/src/kernels.cu` (R7 not-yet-raised cursor), `tensor_cuda/tensor_cuda/__init__.py` (synchronize raises R7;
  exit hook logs), `tensor_cuda/tests/test_dlpack.py` (FIX6 block, 54 tests; FIX4 / FIX5 fill tests corrected).
- tc_signal: `src/dlpack_operand.{h,cpp}` (V6-01, V6-03, g24, hooks), `src/py_args.h` (new), `src/bindings.cpp`,
  `src/lstm_bindings.cpp`, `src/m4_bindings.cpp` (rebinding; raw exit wait), `src/tuning.h`, `tc_signal/__init__.py`,
  `README.md`; tests `test_fix6_findings.py` (new, 67), `test_fix5_findings.py`, `test_fix2_findings.py` (corrected).
- Docs: `docs/P1T_FIX6_REPORT.md`, `docs/P1T_FIX6_LEDGER.md`. Receipts: `artifacts/p1t_fix6/**`, `logs/p1t_fix6_*`.
