# P1T-FIX2 report: round-2 review findings N1-N5 closed in core DLPack and tc_signal

**Verdict: G1 PASS, G2 PASS, G3 PASS, G4 PASS.**
- N1-N4 each reproduce on d116ac0 with the reviewer's probes and pass on this build in the reviewer's condition (no
  usable CUDA device). Each has new regression tests that pass here and fail on their mutant.
- N5 has four foreign-thread exit-phase tests. All pass here. On d116ac0 the in-flight scenario aborts (rc -6), which
  reproduces FIX1's residual.
- The N2 UBSan cases are clean: the reviewer's core_cpu prefix/suffix, regenerated from the new sources, and my
  tc_signal census.
- G2 holds:
  - every round-1 probe passes, including `addon_lifetime` 3/3 runs with 0 incorrect;
  - Voice's runner gives PASS 236/236;
  - the combined session fails exactly the 52 baseline ids.
- All 6 mutants are killed: the order's (a)-(e), with (d) split into a tc_signal and a core variant.

**One probe result needs reading (deviation D4).** The reviewer's `cpu_native oom_sync`, run unmodified on a GPU host,
still exits -6 on this build.
- It does so AFTER printing `PASS: nonallocating completion path`.
- gdb places the terminate in the probe's own teardown (`pybind11::finalize_interpreter`), not in tc_signal.
- In the reviewer's GPU-less condition it passes (rc 0).
- A one-line disarm variant passes in both conditions; the same variant on d116ac0 still terminates in tc_signal.

Seat: Opus 5.5 (high). Order: `orders/P1T_FIX2_ROBUSTNESS.md` (4f62b97). Ledger: `docs/P1T_FIX2_LEDGER.md`.
Receipts: `artifacts/p1t_fix2/`, `logs/p1t_fix2_*`.
- "Before" binaries are wt_fix1 (d116ac0): core `f29ad500…`, tc_signal `3d80352e…`.
- After binaries (`artifacts/p1t_fix2/build_sha256.txt`):
  - core `_tensor_cuda.cpython-312-x86_64-linux-gnu.so`: `7729cb7aff475cbef0d477a63de2e946305d8672bb445252c9c909f0d100b3f5`
  - `_tc_signal.cpython-312-x86_64-linux-gnu.so`: `afd1f01daeea33a4788b17464ba40a5aa02884f6ebd31d36034b8ce2c890b9f6`

## Per-item table

| Item | Fix (file:line) | Proving tests (new) | Probe before (d116ac0) | Probe after (this build) |
|---|---|---|---|---|
| **N1** `synchronize()` reported success when `cudaGetDevice` failed | `tc_signal/src/dlpack_operand.cpp`:<br>• `:644` `sync_and_release_all_impl`; `:652` a failed device query is recorded as an error, where it used to be skipped<br>• `:798` `synchronize_and_release_all` raises RuntimeError naming every failure, and also release-path failures recorded since the last call (`:769` fallback sync, `:776` bound drain, `note_release_failure`)<br>• `:831` a message that cannot be built is a MemoryError<br>`tc_signal/tc_signal/__init__.py:140`: the atexit hook logs one stderr line and never raises.<br>README "Errors on the release paths" (`README.md:154`). | `tc_signal/tests/test_fix2_findings.py`:<br>• `:144` device query fails (CUDA_VISIBLE_DEVICES='') -> RuntimeError naming cudaGetDevice / cudaErrorNoDevice<br>• `:152` the atexit hook logs and does not raise<br>• `:207` an injected sync failure raises, names cudaErrorLaunchFailure and leaks (deleters 0)<br>• `:219` [deferred, bound_drain] a destructor-side failure is raised once by the next synchronize() | `empty_sync` (no GPU): **rc 9** `cudaGetDevice= 100 device= -1 / synchronize returned success` | `empty_sync` (no GPU): **rc 0** `SAFE ERROR: tc_signal.synchronize: CUDA error, so the release is not reported as a success: the current-device query cudaGetDevice failed (cudaErrorNoDevice: …)` |
| **N2** zero-extent shapes reached signed overflow | Core `tensor_cuda/src/bindings.cpp`:<br>• `:306` `checked_extent_bound`: every extent multiplied, a 0 counted as 1<br>• `:316` `checked_numel`; `:323` `checked_nbytes` (the bound times the element size)<br>• applied in `empty`, `from_dlpack`, and `__dlpack__` (`:441`, before the fence and `contiguous_strides`)<br>`tensor_cuda/src/kernels.cu`:<br>• `:80` `numel_of` returns 0 at a zero extent without multiplying, otherwise checks<br>• `:89` `contiguous_strides` is checked, and the unused product past axis 0 is gone<br>• both are the second line, throwing std::overflow_error<br>tc_signal:<br>• `checked_math.h:63` `ck_extent_bound`; `:71` `ck_prod` and `:85` `ck_shape_numel` no longer exempt zeros (every plan uses them)<br>• `dlpack_operand.cpp:162` `validate_tensor` checks the bound before the empty-operand refusal | `tensor_cuda/tests/test_dlpack.py`:<br>• `:918` empty refuses the census (11 shapes x cpu/cuda)<br>• `:925` within-rule shapes stay empty<br>• `:931` import refuses (unconsumed, deleter 0)<br>• `:940` import within the rule<br>• `:951` export refuses (reshape of an empty tensor)<br>• `:961` export within the rule<br>• `:641` FIX1 test re-pointed (D5)<br>`test_fix2_findings.py`:<br>• `:285` all 9 plans x census (104 cases) refuse as an *overflow*<br>• `:316` native UBSan census of `ck_*` and `validate_tensor` | `prefix` **rc 1** `core_cpu.cpp:65:29: runtime error: signed integer overflow: 4 * 4611686018427387904 …`<br>`suffix` **rc 1** `core_cpu.cpp:73:9: runtime error: … 4611686018427387904 * 4 …` | `prefix` / `suffix` **rc 0** `SAFE REFUSAL: probe: the element count of shape [4611686018427387904, 4, 0] overflows int64 (every nonzero extent counts, …)`, no UBSan line (regenerated probe, D2) |
| **N3** `bad_alloc` in the release path called std::terminate | `dlpack_operand.cpp`:<br>• `:644` the in-place two-phase sync makes no allocation (no take-all deque, no std::set or std::vector; fixed-capacity `DeviceSyncSet` `:520` / `SyncReport`)<br>• `:686` `acquire_event` reserves the event's pool slot when the event is created (constructor path)<br>• `:584` `recycle_event` never allocates and is caught<br>• `:747` the pending push is caught and falls back to a synchronous release (counted)<br>• `:720` owner vector reserved in the constructor<br>• `:714` `init_pending_state()` at import (`bindings.cpp:410`)<br>• `:831` message building -> MemoryError | `test_fix2_findings.py`:<br>• `:386` named sites, each in its own process: owner_reserve, event_pool_reserve, pending_push, event_recycle, error_message<br>• `:322` native real-allocation sweep: fail the k-th `operator new` on the destructor, drain and synchronize paths, prefill 0..23 and 1024 | `oom_sync` **rc -6** `terminate called after throwing an instance of 'std::bad_alloc'` (both conditions; gdb: `_Deque_base::_M_allocate_map` inside `sync_and_release_all_impl`) | no GPU: **rc 0** `PASS: allocation failure propagated safely`.<br>GPU: **rc -6** after `PASS: nonallocating completion path`, in pybind11 teardown (D4).<br>Disarm variant: **rc 0** in both conditions. |
| **N4** a throwing producer deleter called std::terminate | tc_signal: `dlpack_operand.cpp:238` `call_deleter_contained` (catch-all, counted), used by `ManagedOwner::reset` `:248`. It covers the immediate, deferred, bound-drain, fallback-sync, synchronize() and atexit paths. The pointer is cleared first, so a deleter is never called twice, and the loops go on.<br>Core: `bindings.cpp:607` `import_release_impl` catch-all (`:632`, counted) | `test_fix2_findings.py:480`: 6 paths, each in its own process, with 2 native throwing deleters per call (each runs once and the second still runs; `deleter_exceptions` 2).<br>`test_dlpack.py:1052`: main thread and foreign pthread | `throw_deleter` **rc -6** `terminate called after throwing an instance of 'std::runtime_error'` | `throw_deleter` **rc 0** `PASS: producer exception contained` |
| **N5** check-then-act window before PyGILState_Ensure at shutdown | Core `bindings.cpp`:<br>• `:523` design comment<br>• `:582` `ImportReleaseLifecycle` (mutex, closing, in_flight, condition variable)<br>• `:607` admission under the mutex, then Ensure / deleter / Release, then leave and notify<br>• `:653` `dlpack_begin_shutdown`: closing set under the mutex; bounded wait with the GIL released (`:667`)<br>`tensor_cuda/__init__.py:178` knob `_DLPACK_SHUTDOWN_WAIT_S = 5.0`, `:181` atexit hook (logs, never raises). | `test_dlpack.py`:<br>• `:1159` in flight across the hook: completes, then finalization<br>• `:1172` a release starting during the wait leaks<br>• `:1183` a release after the hook leaks<br>• `:1192` the wait is bounded and reported | `core_shutdown.py` (round 1) passes on both builds (window not hit).<br>My N5 "during" scenario on d116ac0: **rc -6** `terminate called without an active exception` | N5 tests 4/4 pass; `core_shutdown.py` normal / finalizing PASS |

## Probe before/after lines (verbatim; `artifacts/p1t_fix2/probe_runs/rev2_host_checks_{before,after}_{nogpu,gpu}.{log,json}`)

```
BEFORE (d116ac0), CUDA_VISIBLE_DEVICES= (the reviewer's condition)
empty_sync 9 cudaGetDevice= 100 device= -1
synchronize returned success
prefix 1 checked_numel=0
artifacts/p1t_rev2/probes/core_cpu.cpp:65:29: runtime error: signed integer overflow: 4 * 4611686018427387904 cannot be represented in type 'long int'
suffix 1 checked_numel=0
empty succeeded nbytes=0
artifacts/p1t_rev2/probes/core_cpu.cpp:73:9: runtime error: signed integer overflow: 4611686018427387904 * 4 cannot be represented in type 'long int'
oom_sync -6 ARMED: next host allocation fails; no CUDA call reached yet
terminate called after throwing an instance of 'std::bad_alloc'
throw_deleter -6 ARMED: real ManagedOwner reset calls throwing native producer
terminate called after throwing an instance of 'std::runtime_error'

AFTER (this build), CUDA_VISIBLE_DEVICES=
empty_sync 0 cudaGetDevice= 100 device= -1
SAFE ERROR: tc_signal.synchronize: CUDA error, so the release is not reported as a success: the current-device query cudaGetDevice failed (cudaErrorNoDevice: no CUDA-capable device is detected)
prefix 0 checked_numel=SAFE REFUSAL: probe: the element count of shape [4611686018427387904, 4, 0] overflows int64 (every nonzero extent counts, also next to a zero extent: strides and partial counts are formed from them); refused
suffix 0 checked_numel=SAFE REFUSAL: probe: the element count of shape [0, 4611686018427387904, 4] overflows int64 (every nonzero extent counts, also next to a zero extent: strides and partial counts are formed from them); refused
oom_sync 0 PASS: allocation failure propagated safely
throw_deleter 0 PASS: producer exception contained

BEFORE (d116ac0), GPU visible
empty_sync 0 cudaGetDevice= 0 device= 0 / synchronize returned success / COMPLETED      (not a witness: the device query succeeds)
prefix 1 ... runtime error ... / suffix 1 ... runtime error ...                         (as above)
oom_sync -6 ... terminate called after throwing an instance of 'std::bad_alloc'
throw_deleter -6 ... terminate called after throwing an instance of 'std::runtime_error'

AFTER (this build), GPU visible
empty_sync 0 cudaGetDevice= 0 device= 0 / synchronize returned success / COMPLETED
prefix 0 SAFE REFUSAL ... / suffix 0 SAFE REFUSAL ...
oom_sync -6 PASS: nonallocating completion path / ARMED: ... / terminate called after throwing an instance of 'std::bad_alloc'   <- in pybind11::finalize_interpreter (D4)
throw_deleter 0 PASS: producer exception contained

Supplementary disarm variant (cpu_native.cpp + one line `fail_next = false;` after the call):
before nogpu 134 terminate (bad_alloc) | before gpu 134 terminate (bad_alloc)
after  nogpu 0 PASS: allocation failure propagated safely | after gpu 0 PASS: nonallocating completion path

Native harness (tc_signal/tests/fix2_native_harness.cpp):
ck_census  on d116ac0 sources: {"cases": 18, "failures": 14}  | on this build: {"cases": 18, "failures": 0}, no UBSan line
oom_sweep  on d116ac0 sources: terminate at the 1024-prefill bound drain (rc 134) | on this build: {"runs": 225, "failures": 0, "injected_in_dtor": 4, "allocs_in_drain_and_sync": 0, "pass": true}

N5 "during" scenario (run_n5_scenarios.py, the test module's own script):
d116ac0: rc -6  "terminate called without an active exception" (deleter_done [false])
this build: rc 0, FINAL finalizing=True, in_flight 1 at close -> 0, released 1
```

Other round-1 lines (this build, `probe_runs/round1_*`):
- `addon_lifetime` runs 1-3: each exit 0, 0/12 incorrect, 0/12 reused, 12/12 correct.
- `core_edges` exit 0.
- `core_shutdown --normal` exit 0, "PASS: ordinary foreign-thread release".
- `core_shutdown` (finalizing) exit 0: "finalizing: True", then "PASS: shutdown foreign-thread release completed".
- `numerics_and_streams` exit 0, PASS.
- Round-1 `host_checks.py`: identical to FIX1's accepted result (None cases give TypeError, overflow cases ValueError,
  rint_scalar "()").
- `core_device_drain` exit 3, INCONCLUSIVE (one GPU).

## Gate table

| Gate | Criterion (order; registered in the ledger before any run) | Result | Receipt |
|---|---|---|---|
| G1 | N1-N4 reproduce on d116ac0 (reviewer probe or an exact re-implementation) and pass here | **PASS**: N1-N4 reproduce in the reviewer's no-GPU condition and pass here. For N3 on a GPU host the unmodified probe still exits -6, in its own teardown (D4: receipts and variant). | `probe_runs/rev2_host_checks_*`, `rev2_oom_sync_*` |
| G1 | a new test per item passes here | **PASS**: 19/19 tc_signal (`test_fix2_findings.py`) and 60/60 core `test_fix2_*`, inside the combined session | `logs/p1t_fix2_g2_combined_session.log` |
| G1 | N5's test passes | **PASS**, 4/4. On d116ac0, scenario "during" gives rc -6. | `before_witness/n5_scenarios_on_d116ac0.log` |
| G1 | no UBSan runtime error in the N2 cases | **PASS**: core_cpu prefix/suffix clean (regenerated probe); native ck_census 18/18, no diagnostic | `probe_runs/rev2_host_checks_after_*`, test `:316` |
| G2 | every round-1 probe passes; addon_lifetime 3/3 runs, 0 incorrect | **PASS**: 3/3 runs, 0/12 incorrect each; all others PASS; device_drain INCONCLUSIVE as before | `probe_runs/round1_summary.txt` |
| G2 | Voice runner T1-T6 exits 0 with PASS 236/236 | **PASS**: exit 0, PASS 236/236 (T1 11, T2 144, T3 28, T4 12, T5 28, T6 13); max_N 2.9320924600942896e-06, min margin 6.821x, identical to FIX1 | `logs/p1t_fix2_g2_voice_T1toT6.log`, `artifacts/p1t_fix2/p1t_T1toT6.json` |
| G2 | the combined session shows exactly the 52 baseline ids | **PASS**: 37 failed, 2678 passed, 47 skipped, 15 errors; 52 = 52, 0 new | `logs/p1t_fix2_g2_combined_session.log`, `logs/p1t_fix2_g2_compare.txt` |
| G3 | mutants (a)-(e) each fail their new test; (c) must terminate in the test's subprocess | **PASS**: all 6 killed; (c) rc -6 `terminate called after throwing an instance of 'std::bad_alloc'` | `artifacts/p1t_fix2/g3/` |
| G4 | diff only in the Exercises paths and allowed receipts; no new core `__global__`; probe files byte-identical | **PASS** | `artifacts/p1t_fix2/g4_scope.txt` |

**G2 verdict line:** `G2 PASS: round-1 probes all PASS (addon_lifetime 3/3 runs, 0/12 incorrect each; core_edges, core_shutdown normal+finalizing, numerics_and_streams, host_checks PASS; core_device_drain INCONCLUSIVE one GPU as before); Voice runner T1-T6 exit 0 PASS 236/236; combined core+tc_signal session FAILED/ERROR = 52 baseline ids (0 new, 0 no longer failing).`

**Compare output** (`logs/p1t_fix2_g2_compare.txt`):
```
G1 compare (APA-MF2.2): baseline = /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2/logs/apa_mf1_g1_baseline_unchanged.log; now = logs/p1t_fix2_g2_combined_session.log
summary: 37 failed, 2678 passed, 47 skipped, 2 warnings, 15 errors in 239.27s (0:03:59)
baseline ids 52 now ids 52
baseline ids that no longer fail: 0 []
new failing ids: 0 []
new MF2.2 test files with any FAILED/ERROR: 0
G1 PASS
```

## G3 mutant table

Trees: `/tmp/p1t_fix2_mut_<m>/`, one asserted edit each (`g3/make_mutants.py`, diffs `g3/mut_<m>.diff`).
- Runner: `g3/run_mutant.sh`.
- Each test file is `cmp`-identical to the worktree's. The core-mutant runs used the final test file (re-run after the
  N5 harness correction).

| Mutant | Edit | .so sha256 | New tests on the mutant | What the tests observe | Verdict |
|---|---|---|---|---|---|
| (a) N1 error swallowed | `dlpack_operand.cpp`: `rep.query_failed = false` | tc_signal `180b46b1…980d8f` | **2 failed**, 17 passed | device query: `{'raised': False}` ("synchronize() reported success with no usable CUDA device"); atexit: no log line | KILLED |
| (b) N2 zero-extent check removed | `bindings.cpp` `checked_extent_bound`: `if (dims[i] == 0) return 0` before multiplying | core `8d2077aa…89d12c7` | **44 failed** (all N2 refusals: empty 22, import 11, export 11), 16 passed | 42 `DID NOT RAISE <class 'ValueError'>`; 2 export cases raise `OverflowError: contiguous_strides: a stride of a shape overflows int64` (the second line) | KILLED |
| (c) N3 catch removed | `~RetainUntilComplete`: the try/catch around the pending push removed | tc_signal `ceb186cd…a281d1` | **2 failed** | `[alloc:pending_push]`: **rc -6** `terminate called after throwing an instance of 'std::bad_alloc'` in the test's subprocess; native sweep: terminate | KILLED |
| (d1) N4 catch removed, tc_signal | `call_deleter_contained`: direct call | tc_signal `2344d991…07ff6e` | **6 failed** (all 6 paths) | rc -6 `terminate called after throwing an instance of 'std::runtime_error'` | KILLED |
| (d2) N4 catch removed, core | `import_release_impl`: direct call | core `b4586553…bec9256` | **2 failed** (main, foreign) | rc -6 `terminate … 'std::runtime_error' what(): fix2 test producer deleter threw` | KILLED |
| (e) N5 in-flight wait removed | `dlpack_begin_shutdown`: the `cv.wait_for` removed (admission still closes) | core `3b0762f4…6dfb6` | **3 failed** (during, nested, bounded); "after" passes, as it should (the flag is kept) | during: after the hook `in_flight=1, released=0, deleter_done=[False]`, at_close 1, wait 0 us, then **rc -6** "FATAL: exception not rethrown" when the sleeping deleter re-takes the GIL during finalization. nested: same, rc -6. bounded: `shutdown_timeouts=0` (no wait to bound). (`g3/mut_e_n5_scenarios.log`) | KILLED |

## N5 design

**Problem.** FIX1's guard was a plain check before `PyGILState_Ensure`.
- A foreign thread could pass the check just before the main thread entered `Py_FinalizeEx` and set the finalizing flag.
- In CPython 3.12 that thread's `PyGILState_Ensure` then runs `PyThread_exit_thread`. In v3.12.3 the path is
  `take_gil` -> `_PyThreadState_MustExit`, at ceval_gil.c:337/346 and pystate.c:3032.
- The forced unwind through the noexcept Storage destructor ends in `std::terminate`.

**Ordering premise (verified in source).** In CPython v3.12.3 `Python/pylifecycle.c`, `Py_FinalizeEx` runs:
1. `wait_for_thread_shutdown` (1800);
2. `_PyAtExit_Call` (1815): every Python atexit callback, while `Py_IsInitialized()` is still 1;
3. only then `_PyInterpreterState_SetFinalizing` / `_PyRuntimeState_SetFinalizing` (1832-1833);
4. then `runtime->initialized = 0` (1834).

The sources are kept at `/tmp/p1t_fix2_scratch/cpython/` (sha256 in the ledger).

**Protocol** (`tensor_cuda/src/bindings.cpp:523-680`):
1. **Admission.** A release takes `ImportReleaseLifecycle::mu` and checks `interpreter_accepts_gil_state()` (FIX1's
   second line) and `closing`. If either refuses, it leaks (counted as `leaked_interpreter_shutdown` /
   `leaked_closing`) without touching the GIL. Otherwise `++in_flight`.
2. **Work.** It releases the mutex, then:
   - `PyGILState_Ensure`;
   - the producer deleter, inside a catch-all (N4);
   - `PyGILState_Release`.
3. **Leave.** Only after `PyGILState_Release`, it takes the mutex, runs `--in_flight` and calls `notify_all` at zero.
   So the hook cannot let finalization start while the thread still touches the interpreter.
4. **Close.** tensor_cuda's atexit callback `_dlpack_shutdown_hook` calls `_C._dlpack_begin_shutdown(timeout)`:
   - **GIL released** (`py::gil_scoped_release`), because in-flight releases need the GIL to finish;
   - it sets `closing = true` under the mutex;
   - it waits on the condition variable until `in_flight == 0`, bounded by `_DLPACK_SHUTDOWN_WAIT_S` (5 s);
   - it drops the mutex before re-taking the GIL.
5. **Lock order.** No thread ever waits for the GIL while holding `mu`. The hook drops it first, and releasers never call
   Python under it. So there is no GIL/mutex deadlock.
6. **Result.**
   - Once the hook returns with `in_flight == 0`, every later release leaks without the GIL.
   - No admitted release is still between Ensure and Release when CPython sets the finalizing flag.
   - The F04 check stays as the second line.

**Bounded wait.** If a deleter outlives the bound, the hook returns, writes one stderr line ("N DLPack release(s) still
running after X s at exit…") and records `shutdown_timeouts` / `still_in_flight`. That thread can still be terminated
by CPython at finalization. This residual is stated, not closed: a deleter that blocks forever cannot be made safe
from outside.

**Test staging.** The tests release a real imported Storage on a foreign pthread, as core_shutdown.py does.
- **"In flight" means admitted and inside a slow deleter.** A thread merely blocked in `PyGILState_Ensure` cannot be
  held there until the hook's C call: the eval loop hands it the GIL at the next bytecode boundary (ledger 23:4x).
- The finalizer that drops the GIL during teardown is held by its own module, not by `__main__`. An unfinished
  deleter's frame keeps `__main__`'s globals alive, which would hide the d116ac0 abort.

## Prior art

- **N1:**
  - PyTorch's CUDA caching allocator `process_events`: an in-place FIFO query of recorded events, stopping at the first
    one not ready. Taken: query-and-pop in place.
  - CPython's atexit, which prints a callback's exception and never raises it. Taken: log, don't raise, at exit.
  - Ours: the device-query-is-an-error rule, the deferred report of destructor-side failures, the per-device failure
    set and leak-instead-of-release.
- **N2:**
  - NumPy's array constructor rule: itemsize times every nonzero extent must fit. Measured here: numpy 2.4.2 refuses
    `np.empty((0, 2**62, 4))` and `np.empty((0, 2**61), float32)`.
  - PyTorch 2.11 refuses the element-level part only: "Stride calculation overflowed" for (0, 2**62, 4); "Storage size
    calculation overflowed" for (2**62, 4, 0).
  - GCC 5 (2015) `__builtin_mul_overflow`, C23 `ckd_mul`, SEI CERT INT32-C; UBSan.
  - Taken: NumPy's rule and the builtins. Ours: applying them to TensorCUDA empty / from_dlpack / `__dlpack__`, the
    helpers, and every tc_signal plan.
- **N3:**
  - Fault injection: SQLite's OOM test harness (fail the Nth malloc, swept) and the Linux kernel's failslab /
    fail_page_alloc; the reviewer's cpu_native.cpp `operator new` interposer.
  - "No allocation on the failure/cleanup path" (C++ Core Guidelines E.16).
  - Taken: fail-the-Nth and named-site injection. Ours: the site list, the release-or-leak-exactly-once invariant, the
    pre-reserved event pool slot and the allocation-free in-place drain.
- **N4:**
  - C++11 `noexcept` semantics: an exception escaping a noexcept function calls `std::terminate` ([except.terminate]).
  - C++ Core Guidelines C.36 / E.16 ("destructors must not fail"); Sutter & Alexandrescu, *C++ Coding Standards*
    (2004), item 51.
  - Taken: catch at the boundary. Ours: counting and continuing per manager.
- **N5:**
  - CPython 3.12.3 `Py_FinalizeEx` ordering (atexit before finalizing; verified in source above), the
    `PyGILState_Ensure` rules (C-API docs: a thread is terminated if it takes the GIL while finalizing) and CPython's
    own `threading._shutdown` (wait_for_thread_shutdown), which is the same "stop admitting, then wait" idea.
  - pybind11's `gil_scoped_acquire` / `gil_scoped_release` notes: do not touch the GIL during interpreter shutdown.
  - The quiescence pattern of an admission counter plus a drain wait, e.g. the Linux kernel's `percpu_ref` (kill, then
    wait for zero).
  - PyTorch and CuPy shutdown handling of CUDA resources: both deliberately leak or skip frees at exit (PyTorch never
    destroys its caching allocator at exit; CuPy's pools are freed best-effort). Taken: leak-at-exit is acceptable.
  - Ours: the placement of the protocol around `PyGILState_Ensure` in the DLPack import release, the bound, and its
    reporting.
- **Tests:**
  - The reviewer's probes (scenarios reused) and DeMillo, Lipton & Sayward (1978) mutation testing.
  - LD_PRELOAD-free delay staging through a GIL-holding `ctypes.PyDLL` call and a slow ctypes deleter (ours).
- **Unverified in this seat (lead to check):**
  - "numpy ctors.c PyArray_NewFromDescr_int is_zero overflow";
  - "c10 computeStorageNbytesContiguous safe_multiplies_u64";
  - "PyTorch CUDACachingAllocator process_events";
  - "percpu_ref_kill_and_confirm";
  - "C++ Core Guidelines E.16 C.36";
  - "Sutter Alexandrescu C++ Coding Standards item 51";
  - "SQLite malloc failure testing";
  - "Linux failslab fault injection";
  - "PyTorch CUDACachingAllocator intentionally leaked at exit";
  - "CuPy memory pool free at exit";
  - "pybind11 gil_scoped_acquire finalization".

## Deviations (with reasons)

- **D1. host_checks.py ran through a path-only copy.**
  - The round-2 original hard-codes `/tmp/p1t_rev2_build_review/tc_signal`, `/tmp/p1t_rev2_core_cpu` and
    `/tmp/p1t_rev2_cpu_native`, and writes its JSON into the committed `artifacts/p1t_rev2/`. Both are outside my
    writable area.
  - `make_host_checks_copy.py` makes exactly 5 asserted literal substitutions; the diff is in
    `probe_runs/host_checks_paths_*.diff`.
  - The original stays byte-identical.
- **D2. The "after" core_cpu probe is regenerated.**
  - The committed probe embeds verbatim d116ac0 code, so compiling it unmodified can only test d116ac0.
  - `regen_core_cpu_probe.py` cuts the reviewer's four slices, with the reviewer's markers, from the new sources into
    the reviewer's fixed text.
  - Self-check: regenerating from d116ac0 reproduces the committed probe byte for byte (`cmp` identical). The
    regenerated diff is `probe_runs/core_cpu_regen_fix2.diff`.
  - It is compiled with the reviewer's exact UBSan command. cpu_native.cpp is compiled unmodified.
- **D3. Two conditions.** N1's `empty_sync` reproduces only when `cudaGetDevice` fails. The reviewer's sandbox had no
  usable device, so the probes ran with `CUDA_VISIBLE_DEVICES=` (the witness) and with the GPU visible. Both are
  reported.
- **D4. `oom_sync` on a GPU host.** After the fix the probe's call allocates nothing on success, so its armed
  "next allocation fails" fires later, in the probe's own `pybind11::scoped_interpreter` destructor (gdb:
  `finalize_interpreter -> get_local_internals`, cpu_native.cpp:42), and terminates there.
  - I report it as is: rc -6 after `PASS: nonallocating completion path`.
  - I add receipts: the before/after backtraces, and a one-line disarm variant (`make_disarm_variant.py`). The variant
    gives before -6 / -6 and after 0 / 0.
  - I also add the native sweep, which shows 0 allocations in drain/synchronize.
- **D5. FIX1 test changed.** `test_fix1_f03_empty_zero_extent_is_empty_not_overflow` pinned `(0, 2^62, 2^62)` as an
  empty tensor, which is exactly the N2 defect. It now uses `(0, 2^30, 2^30)`, which is within the rule. The old shape
  is asserted refused by the new N2 tests. Nothing was loosened.
- **D6. Mutant (d) split.** It is split into (d1) tc_signal and (d2) core, one catch each; both are killed.
- **D7. Test hooks in production code** (as FIX1's `_test_rint_then_raise`):
  - tc_signal `_C._test_inject(names)` and `_C._test_injection_armed()`: one-shot named-site failures, costing one
    relaxed atomic load per site;
  - extended `pending_stats()` and `_dlpack_release_stats()` counters.
  - Core has no injection hook: N5 is staged naturally.
- **D8. Leak-check order.** Inside the admission critical section the finalizing check comes before the closing check.
  Both leak; the order only keeps FIX1's `leaked_interpreter_shutdown` attribution, so FIX1's finalizing test passes
  unchanged.
- **D9. Beyond the literal list** (all in scope):
  - core `numel_of` / `contiguous_strides` hardened as a second line (std::overflow_error, values unchanged otherwise);
  - tc_signal's `synchronize()` message, which iostream would have truncated under OOM, is now a MemoryError;
  - `init_pending_state()` at import.
- **D10. Exit-log noise.** On a machine with no usable CUDA device, every process that imported tc_signal logs one
  stderr line at exit. This follows from N1 combined with the atexit-logs rule, and the README documents it.
- **D11. Native builds in tests.** The native-harness tests need `g++`. They use the pybind11 headers at the production
  `tensor_cuda/build/_deps/pybind11-src/include`, read-only (override: `TC_SIGNAL_PYBIND11_INCLUDE`). A missing compiler
  or missing headers FAILS the test; it does not skip.
- **D12. Reused helper.** Round-1 `release_worker.so` is the existing `/tmp/p1t_rev_native/` build, the path
  core_shutdown.py hard-codes. It was used read-only and not rebuilt (sha256 `45875b4f…`).
- **D13. Before-witness quality.** Run against d116ac0, my new tests fail mostly because a hook is missing
  (AttributeError `_test_inject`, KeyError on the new stats keys), not because of the defect. The defect witnesses are:
  - the reviewer's probes;
  - the native harness on d116ac0 sources (ck_census 14/18 failures; oom_sweep terminate);
  - N4 rc -6 (5 tc_signal paths, core main and foreign);
  - N2 "empty operand" instead of "overflows" (tc_signal), and DID NOT RAISE (core, 44);
  - the N5 "during" scenario, rc -6.
- **D14. Build warnings.** The worktree's incremental core rebuilds print 6 nvlink "Skipping incompatible lib*.a" lines.
  The fresh mutant builds of the same sources print none, so I read them as a CMake reconfigure artifact. The
  kernels.cu `lane` warning is pre-existing.
- **D15. Git and network use.**
  - Read-only git only (status / diff / log / ls-files / check-ignore), which this order allows.
  - The CPython v3.12.3 sources were fetched over the network with `curl` into `/tmp/p1t_fix2_scratch`.
- **D16. Runs outside the GPU lock.** A few short CPU-only steps ran without the lock: the exploratory native-harness
  `ck_census` runs (no CUDA call), the numpy/torch CPU-tensor prior-art check, and the g++ compiles. Every GPU-touching
  run used `flock -w 900 /tmp/forge-gpu.lock`.

## Honest residuals (not claimed fixed, or not tested)

- **N5 bound:** a producer deleter that outlives `_DLPACK_SHUTDOWN_WAIT_S` can still be terminated by CPython at
  finalization. It is reported on stderr and counted.
- **Forced unwind:** a deleter that calls `pthread_exit` or is cancelled (`abi::__forced_unwind`) cannot be contained by
  any C++ handler. Mutant (e) shows glibc's "FATAL: exception not rethrown" abort when finalization kills a thread
  inside a deleter.
- **CPython's own allocation:** `PyGILState_Ensure` allocation failure is CPython's `Py_FatalError`, outside our
  control.
- **Coverage of the native sweep:** it hit the pending-push allocation 4 times. Where a std::deque node boundary falls
  varies between runs, so the deterministic proof for that site is the named-site test.
- **Platform:** tc_signal's release paths rely on the GIL; free-threaded CPython is unsupported (unchanged from FIX1).
- **Not measured:** per-call cost. My change adds one relaxed atomic load per injection site, and the event-pool reserve
  on the first event of a device; the per-call path gets no extra CUDA call. This is reasoning, not a measurement:
  performance is out of scope.
- **Multi-GPU:** not exercised (one GPU). `DeviceSyncSet` was not tested beyond a single device.

## Diff stat

Code changed by this order (`git diff --stat d116ac0 -- tensor_cuda tc_signal`, working tree):
```
 tc_signal/README.md                 |  24 +++
 tc_signal/src/bindings.cpp          |  40 +++-
 tc_signal/src/checked_math.h        |  37 +++-
 tc_signal/src/dlpack_operand.cpp    | 412 +++++++++++++++++++++++++++---------
 tc_signal/src/dlpack_operand.h      |  38 +++-
 tc_signal/src/tuning.h              |   5 +
 tc_signal/tc_signal/__init__.py     |  30 ++-
 tensor_cuda/include/tc/core.h       |   3 +
 tensor_cuda/src/bindings.cpp        | 240 ++++++++++++++++++---
 tensor_cuda/src/kernels.cu          |  21 +-
 tensor_cuda/tensor_cuda/__init__.py |  51 ++++-
 tensor_cuda/tests/test_dlpack.py    | 328 +++++++++++++++++++++++++++-
 12 files changed, 1085 insertions(+), 144 deletions(-)
```
- New untracked files: `tc_signal/tests/test_fix2_findings.py` (502 lines), `tc_signal/tests/fix2_native_harness.cpp`
  (355), `docs/P1T_FIX2_LEDGER.md`, this report, `artifacts/p1t_fix2/**`, and `logs/p1t_fix2_*`.
- The literal `git diff --stat d116ac0` also lists the lead's round-2 commits already on voice-fix2: `artifacts/p1t_rev2/*`,
  `docs/P1T_REV2_LEDGER.md`, `logs/p1t_rev2_r1.log`, `orders/P1T_FIX2_ROBUSTNESS.md` and `orders/P1T_REV2_BLIND_REVIEW.md`
  (25 files, 26163+/144-).
- `.gitignore` ignores `*.log` and `*.json`, so the receipts with those extensions need `git add -f`.

## Changed files

- Core:
  - `tensor_cuda/src/bindings.cpp`, `tensor_cuda/src/kernels.cu`;
  - `tensor_cuda/include/tc/core.h` (comment);
  - `tensor_cuda/tensor_cuda/__init__.py`, `tensor_cuda/tests/test_dlpack.py`.
- tc_signal:
  - `src/{bindings.cpp, checked_math.h, dlpack_operand.cpp, dlpack_operand.h, tuning.h}`;
  - `tc_signal/__init__.py`, `README.md`;
  - new `tests/test_fix2_findings.py` and `tests/fix2_native_harness.cpp`.
- Docs: `docs/P1T_FIX2_REPORT.md`, `docs/P1T_FIX2_LEDGER.md`.
- Receipts: `artifacts/p1t_fix2/**`, `logs/p1t_fix2_*`.
- Built (gitignored): core and tc_signal `.so` and `build/`.

**Scratch left (nothing deleted):**
- `/tmp/p1t_fix2_probes/{before,after}` (probe builds);
- `/tmp/p1t_fix2_mut_{a,b,c,d1,d2,e}` (mutant trees and builds);
- `/tmp/p1t_fix2_scratch` (CPython sources, harness builds, the N5 script copy);
- pytest tmp dirs under `/tmp/pytest-of-vader/` (native helper builds).
