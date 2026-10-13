# P1T-FIX7 ledger (implementation seat, Opus 5.5 high)

Order: `orders/P1T_FIX7_EXIT_HOOK_REACHABILITY_AND_ATTRIBUTION.md` (420e821; immutable plan). Worktree
`/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix7`, branch `voice-fix7`.
Scratch: `/tmp/p1t_fix7_*`. No git writes, no subagents, no deletes. Every GPU command under
`flock -w 900 /tmp/forge-gpu.lock timeout ...`.

## 2026-10-13 10:41 Registration (written before any source change, gate run or probe run)

Environment: RTX 4070 SUPER (one GPU), native driver, Python 3.12.3, 12 cores, load average about 26 (other users).

"Before" arm: the prebuilt 429446d binaries in `.../WorkTree/tensor_fork_voice_wt_fix6` (read-only):
core `e7d4a3a313562d80af0401f1deba819260fa95a6ea86abb3a53443c3e17b5a10`,
tc_signal `ef44b9c455f26a7b40d3429d70b7e08ebd4b35444689811de90428e063f2f197` (both equal FIX6's report and the
lead's LEAD_VERIFICATION.md). wt_fix6 `git status` clean, HEAD 429446d.

Probe files (rounds 1-7, `artifacts/p1t_rev*/probes/**`): 92 files, per-file hashes in
`artifacts/p1t_fix7/probe_files_before.sha256` (sha256 of that list
`8260efcdd365114b56c6934648110f1f8484f46176cacd7c2e31c81eb2b10ce3`); every file under `artifacts/p1t_rev7/` (41) in
`artifacts/p1t_fix7/rev7_all_files_before.sha256`. G5 requires the same lists at the end.
Voice adapter `artifacts/p1t_final/tensorcuda_adapter.py` sha256
`600f38b466464ee8bbdd95d009cdf363739bdff79a7e246dcbb4a270dbe43377` (must stay unchanged).

Gates, copied from the order without change (thresholds registered here, before any run):

- G1 (closed): each fails on 429446d and passes on this build: `knob_exit_cpu.py`; `core_knob_exit_cpu.py`;
  `core_knob_exit_gpu.py`; `query_diagnostic_cpu.py`; `query_diagnostic_gpu.py`. The `--control` variants pass on
  both builds. The new regression for each finding, and for each (i) value class, passes.
- G2 (sweeps): tables (i) and (j) complete; every site OK or FIXED; every FIXED site has a test or a stated reason.
  FIX6's rows h8 and g17, and any (g)/(h) row that (i) changes, are corrected.
- G3 (no regression): every round-1..7 probe passes, or is INCONCLUSIVE only for lack of 2 GPUs (round-5
  `exit_gate trace` handled as in FIX6: as written plus its exit-entry copy); `argument_census_cpu.py` passes all its
  checks; `run_cpu_checks.py` 14/14; `gpu_regressions.py fix56` and `semantics` PASS with zero skips;
  `addon_lifetime` 3/3 runs, 0 incorrect; `core_exit_overrun` during and bounded rc 0; Voice PASS 249/249; combined
  session = exactly the 52 baseline ids; real-torch DLPack interop passes.
- G4 (teeth): mutants built outside the worktree, each must fail its new tests and its round-7 probe: (a) the add-on
  hook's knob fallback narrowed back to `(TypeError, ValueError)` with the close skipped on any other exception;
  (b) the same in the core hook; (c) the failure-kind tag dropped (query failures recorded as stream failures).
- G5 (scope): the diff against 8035b3b touches only the Exercises paths plus `docs/P1T_FIX7_*.md`,
  `artifacts/p1t_fix7/**`, `logs/**`; the probe files under `artifacts/p1t_rev*` byte-identical; no new core
  `__global__`.

Forbidden configuration (order): any ordinary (non-exit) API call that waits for another thread's user callback;
returning to ordinary finalization while an admitted release or callback runs; editing a reviewer probe, a Voice test
or Voice's adapter; `os._exit` / `abort` / `std::terminate` as an error path; throwing from destructors; device-wide
syncs on the per-call path; fast-math, TF32, atomics, new core `__global__`; linking tc_signal to `_tensor_cuda`;
writing outside the target or `/tmp`.

Defect-residual rule (order, David): no known real defect is listed as a residual; it is fixed with a test, or shown
with a citation to be not a defect in our code ("Not a defect" table). Every user-visible change is justified.

Constraints found while reading (before any change), binding on the design:
- The reviewer's host builds compile slices of this tree: `artifacts/p1t_rev7/probes/build_core_host.py` takes
  `kernels.cu` 'namespace tc {'..'namespace {\ntemplate <typename T> __device__', the fill span, `bindings.cpp`
  'constexpr const char* kCapLegacy'..'const char* const kDlpackDoc', 'struct CoreParam {'..'}  // namespace\n\nPYBIND11_MODULE',
  the `__dlpack__`..`data_ptr` method span and the `_from_dlpack_capsule`..`empty_cache` def span, and calls
  `note_core_unraisable(const char*, int, cudaError_t)`, `import_release_impl(DLManagedTensor*)`,
  `release_lifecycle().in_flight`, `tc::fill_grid_blocks`. Every core change must stay inside those spans and keep
  those signatures.
- `core_knob_exit_cpu.py` execs the `_dlpack_shutdown_hook` FunctionDef with globals `{_C, sys,
  _DLPACK_SHUTDOWN_WAIT_S}` only: the hook may reference nothing else of the module.
- `artifacts/p1t_rev6/probes/release_trace.cpp` (and FIX6's test trace) includes `dlpack_operand.cpp` with stubs for
  exactly cudaGetDevice, cudaSetDevice, cudaStreamSynchronize, cudaEventCreateWithFlags, cudaEventRecord,
  cudaEventDestroy, cudaEventQuery, cudaGetLastError, cudaPeekAtLastError, cudaGetErrorName, cudaGetErrorString, and
  binds `close_admission_and_wait(double)`: `dlpack_operand.cpp` / `cuda_guard.h` may call no other cudart function
  and must keep that signature.
- `argument_census_cpu.py` asserts exactly 28 `const CallArgs c("...")` bindings in the add-on (the raw exit entry is
  the 29th): no new CallArgs binding may be added.
- The reviewer's probes write receipts into `artifacts/p1t_rev7/` and build into `/tmp/p1t_rev7_build` (the
  reviewer's retained scratch); both arms run path-only copies redirected to `/tmp/p1t_fix7_*` and
  `artifacts/p1t_fix7/`, so no reviewer receipt or scratch is overwritten.

## 10:42-11:05 Design and first implementation (receipts below)

Design decisions (details and prior art at the code sites and in the report):
- V7-01, Python side (both hooks): no conversion of the knob in Python. The hook passes the knob object as is to the
  native exit entry (a deleted knob passes None); the close-and-wait call sits in a `finally` (add-on: after
  `synchronize()`), and every stderr line is written after it. Python's try/finally cannot cover a frame's FIRST
  bytecode (CPython 3.12 checks pending signals at RESUME), so atexit registers a native runner
  (`_C._exit_hook_runner` / `_C._dlpack_exit_hook_runner`) with the hook as its argument: when the hook RAISES, the
  runner closes and waits itself (default delay) and logs any recorded failure nobody reported; when the hook RETURNS
  it does nothing more. That split is deliberate: an always-closing backstop would also mask a hook that returns
  without closing (the registered G4 mutants (a)/(b) would then pass the round-7 probes); with the runner acting only
  on an exception, the hook's own reachability stays observable.
- V7-01, native side: the raw entries read the value at C level (float / int and subclasses); any unreadable value
  (another type, an int beyond the double range) selects the default (tuning constant, exported so the Python knob's
  initial value is the same constant) and STILL closes and waits. Malformed calls (arity / keyword) stay TypeError and
  close nothing. Core's close also no longer rethrows a lifecycle-mutex lock failure before closing (retried).
- V7-02: tc_signal failure records carry a (call, purpose) tag + device + error (FailureRecord), kept through
  SyncReport::merge; every message is built from the tag. The sweep (j) changes ride on the same records.

Constraint found and honoured: the reviewer's round-6 core host build binds `dlpack_begin_shutdown` as `close` with
one argument, so the one-argument function is kept and the new behaviour lives in `dlpack_begin_shutdown_impl`.

## 11:05-11:15 Builds and first receipts

- core build (unchanged source, configure + FetchContent pybind11 v2.12.0 over the network as in FIX3-6): 1m39s;
  FIX7 build 1m37s; tc_signal build 12 s. No warnings.
- Smoke (after, no GPU): `knob_exit_cpu.py` rc 0 and `--control` rc 0; scratch exit scenarios
  (`/tmp/p1t_fix7_scratch/exp/tcs_exit_scenarios.py`): interrupt at the hook's first bytecode -> the runner closed and
  waited 0.6 s for 1 callback, rc 0; synchronize() raising KeyboardInterrupt -> the hook's finally closed
  (exit_in_flight_at_close 1), KI re-raised after it, rc 0.
- `core_knob_exit_gpu.py` (after) rc 0, `--control` rc 0; `query_diagnostic_gpu.py` (after) rc 0.

## 11:09-11:28 Round-7 probes, before arm (429446d)

Before tree `/tmp/p1t_fix7_before_tree` (`artifacts/p1t_fix7/make_before_tree.sh`): byte-identical copies of wt_fix6's
sources, tests and binaries (core e7d4a3a3..., tc_signal ef44b9c4..., both equal 429446d) and of the probe files.
Runner `artifacts/p1t_fix7/run_rev7_probes.sh` (path-only copies by `make_rev7_copies.py`; builds in
/tmp/p1t_fix7_rev7/before). Summary `artifacts/p1t_fix7/probe_runs/rev7_before/summary.txt`:
knob_exit_cpu nogpu rc 134 / gpu rc 134 (controls 0 / 0); core_knob_exit_cpu rc 134 (control 0); core_knob_exit_gpu
rc 1, child aborted (control 0); query_diagnostic_cpu rc 1; query_diagnostic_gpu rc 1; run_cpu_checks 14/14 rc 0;
argument_census PASS 1067; gpu_regressions semantics PASS 504, multidevice exit 3; fix56 rc 1 (2 failed, see next).

## 11:28-11:38 A test defect found by the before arm (fixed) and a before-tree omission (fixed)

- `test_fix6_findings.py::test_fix6_exit_hook_still_waits_for_an_admitted_callback_on_another_thread[deleter]` failed
  on the before tree (3/3 reruns), passed in wt_fix6. Cause (receipt `artifacts/p1t_fix7/capsule_name_receipt.txt`):
  "TypeError: tc_signal: operand 'out': a capsule named '@JHu' is not a DLPack capsule". PyCapsule_New keeps the NAME
  POINTER, which must outlive the capsule (CPython C-API "Capsules"); the script passed bytes literals of its
  module-level code, whose code object CPython frees after the -c body ran; the capsules are read at exit. Fixed in
  the test (names held as module globals); same pattern avoided in the new FIX7 exit script. Original x3 not ordered,
  fixed x3 ordered, same 429446d binaries. (The reviewer probes keep their literals in function code objects that stay
  alive: not affected.)
- `test_fix5_v503_reviewer_cpu_probe_regenerated...` failed on the before tree only because the tree lacked
  artifacts/p1t_rev5/probes: every reviewer probe directory is now copied (92 files, byte-identical to this worktree's).

## 11:15-11:43 Tests (written while the before arm ran)

- New `tc_signal/tests/test_fix7_findings.py` (67 tests) and the core FIX7 block of `tensor_cuda/tests/test_dlpack.py`.
- Corrected tests (they pinned behaviour the order changes), diffs `artifacts/p1t_fix7/test_corrections.diff`:
  FIX6 `_exit_close_admission` typed census case now runs in a subprocess (in-process it would close the pytest
  process's admission); FIX6 exit-keyword case expects CLOSED with the default instead of REFUSED (both packages);
  FIX6 core census's two `_dlpack_begin_shutdown(<unreadable>)` cases moved to a subprocess test; FIX6 R7 exit-log text;
  FIX4 launch-check text. `fix5_regen.py` gains a stub for the device-naming `check_cublas_on` (as it stubs
  note_cublas_handle_destroy_failure) so the regenerated round-5 handle probe still compiles.
- My own test mistakes, fixed before acceptance: guard_get plan counted device 0's guard read (nth 2 -> 3); the stderr
  'raises' staging left a broken sys.stderr for CPython's final flush (rc 120 from Py_FinalizeEx, not the hook):
  `_report` restores it; the core stderr staging had nothing to write (now an unreadable knob gives it a line).
- tc_signal/tests: 751 passed (5m30s). Core `test_dlpack.py -k fix1..fix7`: 208 passed + the corrected stderr case.

## 11:43-12:03 Sweep follow-ups, final product changes, receipts

- Sweep (i) follow-up: the add-on runner now also runs `synchronize()` when the hook raised before ANY hook close was
  counted (the guarantee "pending operands released at exit" was skipped by an interrupted hook); the hook retries
  `synchronize()` once after the close when its first call raised MemoryError (V6-03's kept failure had no reader at
  exit); core's `dlpack_begin_shutdown_impl` keeps FIX5's catch-all that re-takes the GIL (only for unforeseen
  exceptions; the lock failures are retried). Sweep (j) follow-ups: fill_'s pending-error note, the label-proof query
  text keeps FIX4's "cudaPointerGetAttributes failed" prefix, the exit entry reports the delay it actually used
  (NaN / negative -> 0, capped at a day: both packages alike).
- `fix5_regen.py`: stub for `check_cublas_on` (the regenerated round-5 handle probe compiles; test passes).
- Bytecode receipt for sweep (i): `dis` of the add-on hook: the eval-breaker points before the close are RESUME (first
  instruction), the CALL of `C.synchronize()` (inside the try), and the JUMP_BACKWARD that leaves the `except
  NameError` handler inside the finally; CPython 3.12.3 bytecodes.c (citation
  `artifacts/p1t_fix7/citations/cpython_3.12.3_bytecodes_eval_breaker.txt`) checks pending signals at RESUME (oparg < 2),
  JUMP_BACKWARD and the CALL family. The first and the last are why the native runner exists.
- Citations fetched through the AtlasForge web proxy: Python 3.12 atexit docs (`citations/python312_atexit.txt`),
  CPython 3.12.3 bytecodes.c excerpts.
- Strengthened FIX4 assertions (API + device suffixes; fill_ pending note) and new trace modes (unnamed count, prior
  error, prior error before a launch).
- Round-7 probes, after arm (`probe_runs/rev7_after/summary.txt`): every G1 probe rc 0 (knob_exit_cpu no GPU / GPU,
  core_knob_exit_cpu, core_knob_exit_gpu, query_diagnostic_cpu / gpu), every control rc 0, run_cpu_checks 14/14,
  argument_census PASS 1067, gpu_regressions fix56 PASS 157 zero skips, semantics PASS 504 zero skips, multidevice
  exit 3 (one GPU).
- Voice runner (`logs/p1t_fix7_g3_voice.log`, `artifacts/p1t_fix7/p1t_all.json`): PASS 249/249, 43.1 s, adapter sha
  600f38b4... unchanged, Project-Voice `git status` clean before and after.
- G4 mutants built (`artifacts/p1t_fix7/g4/make_mutants.py`, diffs `mut_{a,b,c}.diff`); first run with the test files
  as of ~11:55 (summary_{a,b,c}.txt), second run after the test strengthenings were copied into the mutant trees
  (summary2_*.txt; superseded by the third run below). G5 checker `artifacts/p1t_fix7/g5_scope.py` (FIX6's, retargeted): PASS (preliminary).

## 12:16-12:30 Preliminary combined session; sweep (i) rows i11, i19, i21 given tests (a reporting mistake caught)

- Preliminary combined session (detached, marker; test files before the follow-up below): `G1 PASS`, 52 / 52 baseline
  ids, 0 new, 2991 passed, 47 skipped (FIX6: 47) (`logs/p1t_fix7_g3_combined_session_prelim.log`,
  `logs/p1t_fix7_g3_compare_prelim.txt`).
- While drafting the report I re-checked my own "no injection point" claims in sweep (i). They were wrong: i19 (the core
  close's lifecycle-mutex lock) is injectable with an LD_PRELOAD interposer of `pthread_mutex_lock` (the core imports it
  from glibc: `nm -D`), and i21 / i25(a) (the core exit entry raising after its close) and i11 (the same, add-on) with a
  proxy `_C` whose exit entry closes for real, then raises MemoryError. Tests added (tests only; no product change):
  `test_fix7_sweep_i_core_exit_close_retries_a_failed_lifecycle_lock_instead_of_skipping_the_close[1, 3]`,
  `test_fix7_sweep_i_core_close_step_raising_after_the_close_does_not_skip_the_recorded_failure_log`,
  `test_fix7_sweep_i_exit_close_step_raising_after_the_close_is_logged_and_skips_nothing[gpu, nogpu]`: 32 passed (these
  and the core FIX7 exit group, `logs/p1t_fix7_new_sweep_i_tests.log`). The "Not a defect" N4 ("not testable") is
  withdrawn; N5 renumbered N4.
- i19 receipt (`artifacts/p1t_fix7/lock_failure_receipt.py`, `lock_failure_receipt_{before,after}_{A,B}.txt`): on 429446d
  ONE injected lock failure makes `_dlpack_begin_shutdown` raise `RuntimeError: Invalid argument` with admission left
  open (A), and at a real exit with a deleter in flight the process aborts, rc -6, `FATAL: exception not rethrown` (B).
  This build: one 10 ms retry, closed (A); rc 0, the deleter finished before finalization (B). So i19 was a real
  defect, not a theoretical one; FIX7's `lock_retrying` fixes it.
- Prior art for the new tests: LD_PRELOAD interposition with dlsym(RTLD_NEXT) (glibc ld.so; FIX1's F05 shim), caller
  filtering by return address + dladdr (the classic allocation interposers; unverified here, lead to check
  "LD_PRELOAD interposer __builtin_return_address dladdr caller filter"). Ours: the targeted lock-failure plan, the
  proxy `_C` that raises after a real close.
- Before-arm fix56 rerun on the completed before tree: 1 failed, 156 passed; the failure is FIX6's capsule-name test
  defect (the rev5 cause gone) (`probe_runs/rev7_before/gpu_regressions_fix56_rerun_completed_tree.log`).

## 12:30-12:55 Final receipts

- G4, third run (test files re-synced, cmp-checked identical; product sources identical but for the mutation):
  (a) `-k v701` 24 failed / 14 passed, `knob_exit_cpu` no GPU rc 134, GPU rc 134; (b) `-k "fix7_v701 or
  fix7_core_knob"` 16 failed / 13 passed, `core_knob_exit_cpu` rc 134, `core_knob_exit_gpu` rc 1; (c) `-k "v702 or
  sweep_j"` 13 failed / 16 passed, `query_diagnostic_cpu` rc 1, `query_diagnostic_gpu` rc 1. All KILLED
  (`artifacts/p1t_fix7/g4/summary3_*.txt`, `run3_*/`).
- G5 (`artifacts/p1t_fix7/g5_scope.txt`): PASS (every changed path in scope; no new core `__global__`, device atomics,
  fast-math, TF32, product `cudaDeviceSynchronize`, host callbacks, `os._exit` / `abort` / `std::terminate`; no escaping
  throw in the 32 listed noexcept / destructor bodies; tc_signal not linked to `_tensor_cuda`; 92 reviewer probe files
  byte-identical; `artifacts/p1t_rev*/` unchanged vs HEAD). `artifacts/p1t_rev7/` all 41 files equal
  `rev7_all_files_before.sha256`.
- Diff stat (`artifacts/p1t_fix7/diff_stat.txt`): 19 files changed, 1644 insertions(+), 327 deletions(-) vs 429446d
  under tensor_cuda/ and tc_signal/, plus the new `tc_signal/tests/test_fix7_findings.py` (972 lines).
- Binaries (`artifacts/p1t_fix7/build_sha256_at_combined.txt`): core
  `17bf6fd772f7515dfc13b5739eb3e2c475c5cc620dc906556c7e9ea0f2b18857`, tc_signal
  `b46253594458d3dfc249addf0dcaa85902108d025194a0b7ec9dcfb277e995c8`.
- Final combined session (12:38-12:51, detached, marker `logs/p1t_fix7_g3_combined_session.done`; final test files,
  hashes `artifacts/p1t_fix7/test_files_sha256_at_combined.txt`, unchanged at its end; binaries unchanged): 37 failed,
  2996 passed, 47 skipped, 15 errors; compare `G1 PASS`, baseline ids 52 / now 52, 0 no longer failing, 0 new
  (`logs/p1t_fix7_g3_combined_session.log`, `logs/p1t_fix7_g3_compare.txt`).
- Real-torch interop (`logs/p1t_fix7_g3_torch_interop.log`): 22 passed, 0 skipped (torch 2.11.0+cu130), incl.
  `test_fix6_v602_real_torch_consumers_still_import_zero_copy`.
- Report written: `docs/P1T_FIX7_REPORT.md`. Every gate PASS.

## 13:00-13:55 Lead send-back L1 (the lock-retry loop retried silently)

- The lead's gate (13:0x): `lock_retrying` (core, added for i19) retried forever without writing anything, which
  breaks FIX3 V3-01's "an exit wait is diagnosed, never silent". Required: diagnose it like the exit wait; add a
  sweep (i) addendum covering every exit wait or retry loop of both packages; add a test (failures outlasting a 0.05 s
  delay) and mutant (d); then re-run the listed gates. The lead's finding is correct: my i19 fix added a silent loop.
- Fix (core, `tensor_cuda/src/bindings.cpp:939`): `lock_retrying(lk, first, purpose, diagnostics)`. Once the retries of
  one lock attempt have lasted `first` (the caller's clamped first_diag_s), a C-stdio line goes out (GIL released, no
  allocation). It names the lock and its purpose, the exception's what(), the number of failed attempts and the
  seconds, and says it retries every kDlpackShutdownLockRetryMs and never returns early. Another line follows each
  time that time doubles, and a last line when the lock is taken. Every line is counted in
  g_shutdown_lock_diagnostics (stats `shutdown_lock_diagnostics`) and in the call's `diagnostics` (so the hook's close
  also counts it in `shutdown_diagnostics`). Both call sites are covered: before the close, and the re-lock after each
  timed-wait diagnostic. The close still happens the moment the lock is taken.
- Sweep (i) addendum, i.e. every exit wait or retry loop of both packages; table in the report:
  - the add-on's polling fallback (FIX5, `wait_without_gil`) was silent too. Fixed: it continues the same schedule,
    and its lines say it is polling and why (`exit_wait_poll_diagnostics`);
  - the device syncs of the add-on's exit (the hook's step-1 `synchronize()`, its post-close retry, the runner's
    `synchronize()` → `cudaStreamSynchronize`) waited silently for the GPU work queued on the legacy stream. Fixed: the
    native runner opens a window (RAII, closed on return). Inside it, `sync_device_legacy` first polls an event
    recorded on the legacy stream, with the same schedule and a "finished after" line (`exit_device_wait_diagnostics`).
    The delay is the hook's knob, read at C level from the hook's globals (`PyDict_GetItemString`, then
    `raw_exit_diag_value`, the value rule factored out of `raw_exit_diag_arg` and otherwise unchanged).
  - Only CUDA calls dlpack_operand.cpp already makes are used, so the reviewers' host traces still link (release_trace.cpp
    stubs exactly those).
  - The new tunable is `tuning::kExitDevicePollMs = 1`.
  - Ordinary `synchronize()` / drains are unchanged: one relaxed atomic load.
- Tests:
  - core `test_fix7_l1_core_lock_retry_wait_is_diagnosed_like_the_exit_wait[close, relock_after_diagnostic]`; the shim
    gained `shim_lock_arm_after(k, n)` so the second case lets the close's lock through and fails the re-lock;
  - add-on `test_fix7_l1_exit_wait_polling_fallback_is_diagnosed`, with its own `_tc_signal` lock shim;
  - add-on `test_fix7_l1_device_wait_of_exit_is_diagnosed[diag, default]`, with ~0.6 s of `torch.cuda._sleep` queued
    just before the runner. It also checks that an in-run `synchronize()` writes nothing.
  - All pass on the L1 build.
- Before / after (`artifacts/p1t_fix7/l1/receipt_{preL1,after}_*.txt`, `l1_receipt.py`):
  - The pre-L1 arm uses the earlier G4 trees' unmutated halves, which carry the FIX7 report's binaries: core
    17bf6fd7 from /tmp/p1t_fix7_mut_a, tc_signal b4625359 from /tmp/p1t_fix7_mut_b.
  - Their sources are saved in `artifacts/p1t_fix7/preL1_sources/` (SHA256SUMS); the L1 product diff is
    `artifacts/p1t_fix7/l1/l1_product.diff`.
  - core lock-retry lines: 0 before, 2 after (plus the "took" line);
  - add-on polling lines: 0 before, 4 after;
  - add-on device-wait lines: 0 before, 4 after.
  - Every staging gave rc 0, and the deleter finished before finalization in both arms. The pre-L1 runs were simply
    silent.
- Builds: tc_signal `./build.sh 89` (production's pybind11 source); core `./build.sh 89` (incremental).
- Mutants on refreshed trees `/tmp/p1t_fix7_mutL1_{a,b,c,d}` (make_mutants.py: dst and tag arguments; mutant (d)
  disables the `if (now >= next)` diagnostic block of lock_retrying). Mutant (d)'s core is built offline: build.sh's
  cmake commands plus FETCHCONTENT_SOURCE_DIR_PYBIND11 = production's pybind11 source (read-only) and
  FETCHCONTENT_FULLY_DISCONNECTED.
  - First L1 pass (`summaryL1_*`): all four killed. For (d): 1 failed (the L1 test, no lock line), 2 passed (the i19
    retry tests, unchanged).
  - The re-lock case was then added, the tests re-synced, and the final pass (`summaryL1f_*`) is the receipt.
- Prior art (also at the code sites `lock_retrying`, `wait_without_gil`, `wait_device_work_diagnosed`, the runner):
  - FIX3 V3-01's schedule, taken as is;
  - the kernel's repeated stall warnings (RCU CPU stall warnings, hung-task detector). Unverified — lead to check,
    search "RCU CPU stall warning repeated", "hung_task_timeout_secs";
  - cudaEventQuery's NotReady contract, used for host-side progress polling. Unverified — lead to check, search
    "cudaEventQuery cudaErrorNotReady";
  - ours: the exit-runner window and the C-level knob read from the hook's globals.
- Final receipts on the L1 build (binaries core `750112413b78450937ba126e56ab700e3f9e16d1092f88af239b13fee48f9245`,
  tc_signal `ca5770c152f6946921ce8c06681a2ecad6beaa81232b948cceac2a7a99605c37`):
  - round-7 probes, after arm (`probe_runs/rev7_afterL1/summary.txt`):
    - every G1 probe rc 0 and every control rc 0;
    - run_cpu_checks 14/14, argument_census PASS 1067;
    - fix56 PASS 157 with zero skips, semantics PASS 504 with zero skips, multidevice exit 3 (one GPU);
  - Voice PASS 249/249, adapter unchanged, Project-Voice clean before and after;
  - the i19 receipt after arm: A returns after one 10 ms retry with admission closed; B rc 0;
  - the L1 before/after receipts: 0 lines before in all three stagings; after, 2 lock lines, 3 polling lines and 4
    device-wait lines;
  - mutants (a)-(d) on the refreshed trees (`summaryL1f_*`): all KILLED;
  - G5 PASS (35 noexcept / destructor bodies checked); `artifacts/p1t_rev7` 41/41 files unchanged;
  - real-torch interop 22 passed, 0 skipped;
  - L1 tests 10 / 10 (`logs/p1t_fix7_l1_tests.log`);
  - combined session (detached with a marker; 13:34-13:47; tests and binaries unchanged during it): 37 failed, 3001
    passed, 47 skipped, 15 errors; compare `G1 PASS`, 52 / 52 baseline ids, 0 new
    (`logs/p1t_fix7_g3_combined_session.log`, `logs/p1t_fix7_g3_compare.txt`).
- The first report's receipts that these replaced were renamed, not deleted: `logs/preL1_*`, `*_preL1*`.
- Report amended: per-finding table and sweep (i) / (j) line references re-pointed to the L1 sources; U8, U9 and U10;
  the addendum table; the section "Lead send-back L1".

## Prior art (Prior Art Directive: the same text as the report's "Prior art" section)

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
- **Test defect fixed (capsule names)**: the CPython C-API "Capsules" contract (the name string "must outlive the
  capsule"). Nothing new.
- These annotations are also at the code sites (`tc_signal/src/dlpack_operand.cpp` failure-record and exit blocks,
  `cuda_guard.h` attribution block, `py_args.h` `raw_exit_diag_arg`, `bindings.cpp` runner; core `bindings.cpp`
  `dlpack_begin_shutdown_impl`, `raw_exit_diag_arg`, runner, `unraisable_message`; `kernels.cu` drain and fill_ blocks;
  both hooks' docstrings) and in the ledger.
