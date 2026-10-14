# P1T-FIX6 ledger (implementation seat, Opus 5.5 high)

Order: `orders/P1T_FIX6_SYNC_WAIT_AND_USER_CALLBACKS.md` (6ab46a0; immutable plan). Worktree
`/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix6`, branch `voice-fix6`.
Scratch: `/tmp/p1t_fix6_*`. No git writes, no subagents, no deletes. Every GPU command under
`flock -w 900 /tmp/forge-gpu.lock timeout ...`.

## 2026-10-13 07:21 Registration (written before any change, build or gate run)

Environment: RTX 4070 SUPER, driver 595.99.02 (native), Python 3.12.3, load average about 26 (other users).

"Before" arm: the prebuilt acd42fa binaries in `.../WorkTree/tensor_fork_voice_wt_fix5` (read-only):
core `9af40433f69e4d5582a4a83bb65165ee4e9f72cdcf35c28a7ad4f4cceca15510`,
tc_signal `3324bb4a44dde6a45e62fcd91114d60ed1ef3268a2b35f7f1aebbe08c67a2552` (both equal the FIX5 report's final build).

Probe files (rounds 1-6, `artifacts/p1t_rev*/probes/**`): 82 files, per-file hashes in
`artifacts/p1t_fix6/probe_files_before.sha256` (sha256 of that list
`f78bc96eb03bacb9f0f06a134db65e9101578865bcfc83c6d77c4be084268160`). G5 requires the same list at the end.
Voice adapter `artifacts/p1t_final/tensorcuda_adapter.py` sha256
`600f38b466464ee8bbdd95d009cdf363739bdff79a7e246dcbb4a270dbe43377` (must stay unchanged).

Gates, copied from the order without change (thresholds registered here, before any run):

- G1 (closed): each fails on acd42fa and passes on this build: `sync_lock_inversion_cpu.py` in both conditions
  (no visible GPU, GPU visible); `core_cast_exit_gpu.py bool` and `index`; `core_host_cpu.py casts_exit bool` and
  `index`, regenerated; `release_trace_cpu.py diagnostic_oom_query`, `_sync`, `_device`, regenerated.
  `sync_late_release_cpu.py` passes on this build. The `--control` variants pass on both builds. The new regression
  for each finding passes.
- G2 (sweeps): tables (g) and (h) complete; every site OK or FIXED; every FIXED site has a test or a stated reason;
  FIX5's affected sweep-(c) rows corrected.
- G3 (no regression): every round-1..5 probe passes, or is INCONCLUSIVE only for lack of 2 GPUs (round-5
  `exit_gate trace` handled as Required fix 1: reported as written, plus a path-only copy calling the trace module's
  real exit entry point, recorded as a deviation); `addon_lifetime` 3/3 runs, 0 incorrect; `core_exit_overrun`
  during and bounded rc 0; Voice T1-T6 + T1R + T4S PASS 249/249; combined session = exactly the 52 baseline ids;
  real-torch DLPack interop passes.
- G4 (teeth): mutants built outside the worktree, each fails its new test: (a) the wait for other threads restored in
  ordinary `synchronize()`; (b) the V6-02 admission or refusal removed; (c) the V6-03 persistence removed.
- G5 (scope): diff against 123610f only in the Exercises paths plus `docs/P1T_FIX6_*.md`, `artifacts/p1t_fix6/**`,
  `logs/**`; probe files byte-identical; no new core `__global__`.

Forbidden configuration (order): any ordinary (non-exit) API call that waits for another thread's user callback;
returning to ordinary finalization while an admitted release or admitted user-Python section runs; editing a reviewer
probe, a Voice test or Voice's adapter; `os._exit` / `abort` / `std::terminate` as an error path; throwing from
destructors; device-wide syncs on the per-call path; fast-math, TF32, atomics, new core `__global__`; linking
tc_signal to `_tensor_cuda`; writing outside the target or `/tmp`.

## ~07:40 Lead directive received mid-task (binding; changes no gate, order file unchanged)

Verbatim summary of the coordinator's message: a known real defect may not be listed as a "residual"; fix it in this
round with a test. Source: David 2026-10-13, "Bugs must be squashed" and "bugs don't get to be post-hoc accepted".
Includes FIX5's R2 (a producer deleter that calls pthread_exit) and R4 (the add-on wait's liveness under continuously
overlapping releases) unless the V6-01 fix removes them. Something may stay unfixed only with a citation showing it is
NOT a defect in our code (a "Not a defect (with justification)" table). Any deviation that changes user-visible
behavior must be justified as correct, not just disclosed.

## 07:32-07:40 Diagnosis receipts (before any change; acd42fa = wt_fix5 prebuilt, read-only)

- V6-02 crash mechanism, gdb on `core_cast_exit_gpu.py bool` (acd42fa, GPU lock): SIGSEGV in thread "python3
  (Exiting)", frame #1 inside `_tensor_cuda.so` (pybind11 dispatcher cleanup), frame #0 in CPython; no P1-T frame
  left on the stack (`/tmp/p1t_fix6_scratch/exp/gdb_cast_bool_before{,2}.txt`).
- Scratch experiment `/tmp/p1t_fix6_scratch/exp/dispatch_exit.py` (acd42fa, no GPU visible; daemon thread at
  ordinary exit, the user method sleeps 0.6 s across the start of finalization), rc per mode:
  `pure` (pure Python control) 0; `empty_len` (`_C.empty(SeqLen(), ...)`, list caster `__len__`, positional) 0, 0;
  `empty_len_kw` (`_C.empty(shape=SeqLen(), ...)`, same conversion, keyword) **139, 139**; `empty_repr`
  (no-match message `repr()`, positional) 0; `empty_repr_kw` (no-match `repr()`, with keywords) 0, 0;
  `relu_repr` (pre-existing binding, no-match `repr()`) 0.
- Conclusion (reasoning from these receipts and pybind11 v2.12 `pybind11.h:717-1160`): CPython 3.12 exits the
  thread (pthread_exit, forced unwind) when the user method re-takes the GIL after finalization began; the unwind
  runs C++ cleanup in the frames above it; the crash comes from a Python object whose LAST reference dies in that
  cleanup: the dispatcher's private copy of the kwargs dict (made when a keyword argument fills a named parameter)
  is deallocated with the exiting thread's state already freed. Positional calls and the no-match path hold only
  references the caller also owns (decremented, never deallocated), which is why they did not crash here. They still
  have live `py::object`s and C++ cleanup on the stack across the GIL re-take, so by the order's criterion they are
  not "harmless": they are handled below, not argued away.

## 07:40-08:04 Implementation (first pass; builds 07:52 / 07:58) and round-6 probes on both arms (08:02 / 08:03)

Changes (details in the report): tc_signal `dlpack_operand.{h,cpp}` (V6-01 wait removed; V6-03 persisted drain record
and raise-then-clear; CapsuleRef admits every last-reference capsule drop; two test-hook sites), `tuning.h` (retired
knob), new `py_args.h` + all three binding files ((*args, **kwargs) + C-level conversion); core `bindings.cpp`
(V6-02 strict reading in `tensor_dlpack`; CoreCallArgs; every P1-T binding rebound; `_dlpack_raise_unraisable`),
`kernels.cu` (not-yet-raised cursor of the unraisable record), `tensor_cuda/__init__.py` (synchronize raises the record
once; exit hook logs it). Builds: `logs/p1t_fix6_build_tc_signal_1.log` rc 0, `logs/p1t_fix6_build_core_1.log` rc 0
(FetchContent fetched pybind11 v2.12.0 from GitHub, as FIX3 D17 / FIX4 D11 / FIX5 D10).

Round-6 probes, `artifacts/p1t_fix6/run_rev6_probes.sh` (path-only copies by `make_rev6_copies.py`; summaries
`artifacts/p1t_fix6/probe_runs_rev6_{before,after}_summary.txt`, logs `probe_runs/rev6_{before,after}/`):
- before (acd42fa): sync_lock nogpu rc 1, gpu rc 1; sync_late_release rc 1 (`waited_for_later=True`);
  core_cast_exit_gpu bool / index rc 139, --control rc 0 / 0; run_cpu_checks: diagnostic_oom_query/sync/device rc 1,
  casts_exit bool/index rc -6 (`pybind11::handle::dec_ref() PyGILState_Check() failure`, the same unwind; the
  reviewer saw -11), controls rc 0, sync_lock rc 1, others rc 0. Regeneration self-check: the four slices taken from
  acd42fa equal the reviewer's pinned hashes (`slices_selfcheck_before.txt`, all_identical=True).
- after (this build): every one of those rc 0 (sync_lock nogpu / gpu, sync_late_release `waited_for_later=False`,
  core_cast_exit_gpu bool / index and controls, diagnostic_oom_query/sync/device, casts_exit bool/index and controls,
  the rest unchanged rc 0). The after-arm slices differ from the pin, as expected (this tree's code).

## 08:04-08:28 (first labelled "08:40-09:05"; code comments cite that label) Test runs, three test corrections, one regression of my own found and fixed

- tc_signal suite (first full run): 3 failed / 681 passed. (1) FIX2 `test_n3_...[alloc:error_message]` asserted that
  the synchronize() AFTER a message-allocation MemoryError succeeds, i.e. it pinned V6-03 (the drain failure was
  lost); corrected: the next synchronize() raises it once, then recovery succeeds (`artifacts/p1t_fix6/
  test_fix2_corrections.diff`). (2) FIX5 thrower census asserted 2 deleter calls: it relied on synchronize()'s removed
  wait, which let x's release start before the hook closed admission; with the close immediate, x's release starts
  after the close and is leaked by V5-01's rule (deleters_leaked_after_close 1); corrected. (3) FIX5 capsule-destructor
  census failed 2 of 6 runs with the right result printed: a harness race (print() writes text and newline separately;
  the worker's line was cut by the main thread's RESULT line); `_run` now finds the marker anywhere in a line (both
  test files). Also FIX5 immediate / deferred census at 0.1 s: "the hook waited" is timing-dependent (the deleter may
  finish during the hook's first CUDA call); kept certain for 5.5 / 12 s, order + rc asserted for all
  (`test_fix5_corrections.diff`).
- core test_dlpack.py (first run): 5 failed. Three are `test_fix5_sweep_c_core_exit_wait_on_a_non_main_thread_never_
  aborts[0,2,4]` rc -11: MY REGRESSION. `_dlpack_begin_shutdown` rebound as (*args, **kwargs): for a positional call
  pybind11 creates a new empty kwargs dict (pybind11.h step 4b, `kwargs = dict()`) owned only by the dispatcher; the
  exit wait releases the GIL; on a non-main thread CPython exits the thread at the re-take and the unwind deallocates
  that dict: SIGSEGV (the V6-02 mechanism). FIX5's typed binding had the mirror defect for a KEYWORD call
  (PyDict_Copy). Fix: the two bindings whose body waits with the GIL released (core `_dlpack_begin_shutdown`, add-on
  `_exit_close_admission`) become raw CPython functions (caller-owned args/kwargs; no Python object held by our frames
  across the wait). The two fill_ failures are investigated next.

## 08:28-08:35 Fix of my regression (final builds 08:28), core tests, FIX5 residual dispositions

- Raw CPython functions for the two GIL-releasing exit waits (core `_dlpack_begin_shutdown`, add-on
  `_exit_close_admission`); rebuilt (`logs/p1t_fix6_build_*_2.log`, rc 0). `test_fix5_sweep_c_core_exit_wait_on_a_
  non_main_thread_never_aborts[0-4]` 5/5 pass again; tc_signal FIX6 tests 67 pass.
- FIX4 `test_fix4_fill_guard_failures_raise_count_and_consume[fill_fail_launch_and_restore]` and FIX5
  `test_fix5_sweep_d_core_fill_unwinding_restore_failure_is_recorded` failed: their script called tc.synchronize()
  after an unraisable fill_ restore failure and expected success. With R7 fixed that synchronize() reports it once;
  corrected to assert that report (`sync_reported`) and a successful second synchronize().
- Core FIX6 tests added (54), all pass.
- FIX5 R2 (producer deleter calls pthread_exit) receipt, `artifacts/p1t_fix6/r2_receipt/` (no GPU, worker thread, a
  C deleter that calls pthread_exit): torch 2.11 consumer rc 134 `terminate called without an active exception`;
  NumPy 2.4.2 consumer rc 124 (timeout 20: the thread exited holding the GIL, the interpreter hangs); tc_signal rc 134
  `FATAL: exception not rethrown`. Citations: DLPack Python spec (fetched 2026-10-13, saved
  `artifacts/p1t_fix6/citations/dlpack_python_spec_2026-10-13.txt`): "the deleter of the DLManagedTensor will be called
  by the destructor of the consumer library object"; vendored dlpack.h v1.3 lines 302-307 / 355-360 (the deleter is the
  "Destructor") and 397 ("As a C function, must not thrown C++ exceptions"); libstdc++ 13 bits/cxxabi_forced.h
  (`__forced_unwind`: "Thrown as part of forced unwinding"). Disposition: NOT a defect in our code (producer contract
  violation that no consumer can survive); goes to the report's "Not a defect" table, not a residuals list.
- FIX5 R4 (synchronize()'s count-based wait liveness): REMOVED with the wait itself (V6-01). The exit wait closes
  admission first, so only sections admitted before the close count and none can extend it.

## 08:35-08:58 Earlier-round probes, Voice, mutants built, G5 checker

- Round 5 (`artifacts/p1t_fix6/run_rev5_probes.sh` = FIX5's runner retargeted, plus the Required-fix-1 trace copy;
  summaries `rev5_summary_{before,after}.txt`, logs `probe_runs/rev5_{before,after}/`): shims `/tmp/p1t_rev5_query_once.so`
  and `/tmp/p1t_rev5_fill_grid.so` byte-identical to fresh gcc builds of the probe sources (sha 83f9aa98... / f6007cac...),
  used read-only. Before (acd42fa): every probe as FIX5 reported. After (this build): exit_gate full nogpu / gpu and
  controls rc 0; exit_gate gpu and control rc 0 (in the gpu mode the daemon thread's LATER producer call is refused
  after the close, a RuntimeError printed by threading's excepthook, the probe PASSes: V5-01's after-close rule);
  **exit_gate trace as written rc 1 (child rc -6, FATAL: exception not rethrown), --control rc 1**: the trace child
  models the exit hook as synchronize() alone, which no longer waits (the order's Required fix 1 anticipates this);
  the path-only copy whose hook calls synchronize() + close_admission_and_wait (the real exit entry; one added binding in
  a copy of addon_trace.cpp, diff in `trace_exit_entry_copy_*.txt`) rc 0 on both arms. query_error_cpu / single_gpu,
  regenerated handle_insert_oom / fill_grid, fill_grid_single_gpu, device_error_sweep (40 passed), sync_existing_error,
  addon_lifetime (12/12), core_edges, core_exit_overrun during / bounded, numerics_and_streams, core_import_query_error,
  release_failures (66 passed), build_cpu.sh: rc 0; the three two-GPU probes exit 3; as-written handle / fill rc 1 on
  both arms (they embed 1475fae / baseline code, FIX5 D3).
- Round 4 (`run_rev4_cpu_probes.sh` retargeted; both arms): identical results on both arms (host_checks 20/20 cases
  identical after path normalization, both conditions: `rev4_host_checks_compare.txt`); sync_slot_cpu rc 1 on both (its
  message now carries a `RuntimeError: ` prefix: synchronize() raises a captured py::error_already_set, whose what()
  includes the type); core_interop as written reproduces its embedded 62c8d9f traces on both; regenerated import refuses.
- Rounds 1-3 (`run_rounds123_probes.sh` retargeted): addon_lifetime 3/3 runs, 12/12 correct, 0 incorrect;
  core_edges 0; core_shutdown --normal 0 and finalizing 0; numerics_and_streams 0; core_device_drain 3; round-3
  core_exit_overrun during 0 (3.4 s) / bounded 0 (7.8 s); round-1 and round-2 host_checks identical to FIX5's receipts
  (normalized; `round12_host_checks_vs_fix5.txt`).
- Voice runner (order's exact command): rc 0, verdict PASS 249/249 (T1 11, T2 144, T3 28, T4 12, T5 28, T6 13, T1R 1,
  T4S 12), 68.1 s; Project-Voice `git status` clean afterwards.
- Combined session launched about 08:57 detached (order's exact command; completion marker /tmp/p1t_fix6_scratch/combined.done;
  read only after the marker exists, as FIX5 D9).
- Mutants (a) (b) (c) built under /tmp/p1t_fix6_mut_{a,b,c} (`g4/make_mutants.py`, diffs `g4/mut_*.diff`, sha256
  `g4/mut_*_sha256.txt`); (b)'s core built offline against production's pybind11 source (read-only).
- G5 checker `artifacts/p1t_fix6/g5_scope.py`: first dry run flagged one "fast-math" line, the pre-existing
  fp32_divide_probe docstring "Fast-math canary: ..." (acd42fa bindings.cpp:371) moved by my registration rewrite; the
  pattern was narrowed to compiler flags (use_fast_math / ffast-math / ftz / prec-div), reason recorded in the checker.

Note on times: section headers were first written with estimated times; corrected at 09:13 from the files' mtimes
(builds `logs/p1t_fix6_build_*`, probe summaries, `logs/p1t_fix6_g3_*`). The order of events is unchanged.

## 08:57-09:12 Combined session, mutants, final-build re-run of the round-6 probes, G5

- Combined session (order's exact command, detached with a completion marker, read after it existed): pytest rc 1
  (the baseline APA files), `37 failed, 2886 passed, 47 skipped, 2 warnings, 15 errors in 625.08s`; compare
  (`logs/p1t_fix6_g3_compare.txt`): baseline ids 52, now 52, 0 no longer failing, 0 new, `G1 PASS`. 2886 passed =
  FIX5's 2765 + 121 new FIX6 tests (tc_signal 67, core 54).
- G4 (`g4/run_mutant.sh`; mutant test files cmp-identical to the worktree's):
  (a) wait restored: `-k "v601 or exit_hook_still_waits or synchronize_does_not_wait"` 6 failed / 6; probes
  sync_lock_inversion nogpu rc 1, gpu rc 1 (rescue), sync_late_release rc 1 (waited_for_later=True): KILLED.
  (b) refusal removed: `-k fix6_v602` 30 failed, 17 passed (the accepted-value cases); probes: core_host_cpu casts_exit
  bool / index regenerated from the mutant rc 134 (`pybind11::handle::dec_ref()` during the exit unwind);
  core_cast_exit_gpu bool / index rc 0 on the mutant although the methods ran ("CAST started"): the (*args, **kwargs)
  binding alone removes the dispatcher-owned kwargs copy whose deallocation crashed (an independent second layer):
  KILLED by the tests and the host probe.
  (c) persistence removed: `-k "v603 or (n3_allocation and error_message)"` 7 failed / 7; probes diagnostic_oom_query /
  sync / device rc 1: KILLED.
- The round-6 after-arm probes of 08:03 ran on the FIRST build (core 55276e88..., tc_signal 672f22a3...), before the
  raw-function fix. Re-run at 09:10 on the final build (core e7d4a3a3..., tc_signal ef44b9c4...;
  `probe_runs/rev6_after_final/`, summary `probe_runs_rev6_after_final_summary.txt`): identical outcomes, every G1 probe
  rc 0. Rounds 1-5, Voice and the combined session ran on the final build (their binary records).
- G5 (`artifacts/p1t_fix6/g5_scope.txt`): PASS (every changed path allowed; HEAD vs 123610f = the order file only;
  0 new core __global__ / atomics / fast-math flags / TF32 / cudaDeviceSynchronize / host callbacks / exit calls in the
  2130 added code lines; no escaping throw in the 23 destructors and noexcept bodies checked; CMakeLists unchanged and no
  _tensor_cuda in DT_NEEDED; the 82 probe files equal the registered per-file hashes; artifacts/p1t_rev*/ unchanged).
- Final .so sha256: core `e7d4a3a313562d80af0401f1deba819260fa95a6ea86abb3a53443c3e17b5a10`, tc_signal
  `ef44b9c455f26a7b40d3429d70b7e08ebd4b35444689811de90428e063f2f197` (`artifacts/p1t_fix6/build_sha256.txt`).
- Read-only trees: `git status` of wt_fix5 and Project-Voice clean after all runs.

## Prior art (ledger copy; details in the report and at the code sites)

V6-01: Coffman, Elphick & Shoshani 1971 (circular wait), lock ordering, RCU (McKenney & Slingwine 1998), CPython
threading._shutdown. V6-02 / sweep (g): Python array API __dlpack__ signature (2021-2023), PyTorch 2.11
Tensor.__dlpack__ `type(stream) is not int` refusal, CPython 3.12 PyLong / PyFloat / PyTuple C API and the
METH_VARARGS | METH_KEYWORDS convention, CPython getargs.c keyword matching (unverified, lead to check), pybind11 v2.12
py::args / py::kwargs and dispatcher, glibc / libstdc++ forced unwinding. V6-03: SQLite OOM testing (unverified, lead to
check). FIX5 R7: tc_signal's raise-once rule, PyTorch C10_CUDA_CHECK_WARN. Testing: round-6 reviewer probes, LD_PRELOAD /
fault injection, DeMillo, Lipton & Sayward 1978. No prior art known to me for the all-methods recording object, or for
"the dispatcher-owned last reference is what crashes" as a finding.
