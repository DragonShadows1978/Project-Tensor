# P1T-FIX2 ledger (Opus 5.5 high)

Order: `orders/P1T_FIX2_ROBUSTNESS.md` (fork commit 4f62b97, immutable). Worktree: `tensor_fork_voice_wt_fix2`
(branch voice-fix2 = d116ac0 + round-2 receipts). Report: `docs/P1T_FIX2_REPORT.md`. Receipts: `artifacts/p1t_fix2/`,
`logs/p1t_fix2_*`. Scratch: `/tmp/p1t_fix2_*`.

## 2026-10-12 23:09 EDT: read-in

- Read the order, `/mnt/Shared/HOUSE_RULES.md`, `AGENTS.md`, `docs/P1T_REV2_LEDGER.md`, `artifacts/p1t_rev2/REGISTRATION.md`,
  the three round-2 probes, `logs/p1t_rev2_r1.log` ~7900-7960 / 9040-9690 (probe sources, compile commands, results),
  `docs/P1T_FIX1_REPORT.md`, the round-1 probe README.
- Mechanism check (HOUSE_RULES 3): the order measures the round-2 findings N1-N5 on the code paths the reviewer's probes
  exercise (tc_signal `synchronize_and_release_all`, `ManagedOwner::reset`, core `checked_numel` / `numel_of` /
  `contiguous_strides`, core `import_release_*`). No scope mismatch.
- Read-only git used (allowed by this order): `git log`, `git status`, `git diff --stat d116ac0 HEAD -- tensor_cuda tc_signal`
  (empty: the worktree's sources equal d116ac0).
- "Before" binaries: `wt_fix1` prebuilt, sha256 core `f29ad500...`, tc_signal `3d80352e...` (= FIX1 report); `git status`
  of wt_fix1 is clean.

## Registration (written before any gate run; thresholds copied from the order, nothing loosened)

- G1: each of N1-N4 reproduces on d116ac0 (reviewer probe, or an exact re-implementation if the probe cannot run
  unmodified) and passes on this build; each has a new test passing here; N5's test passes; no UBSan runtime error in
  the N2 cases (reviewer's core_cpu prefix/suffix, regenerated from the new sources; plus my census).
- G2: every round-1 probe passes (addon_lifetime 3 runs, 0 incorrect); Voice runner T1-T6 exit 0, PASS 236/236; one
  combined pytest session (tensor_cuda/tests + tc_signal/tests, selector_accuracy ignored) whose FAILED/ERROR ids are
  exactly the 52 baseline ids (apa_mf2_2_g1_compare.py prints G1 PASS).
- G3: mutants (a) N1 error swallowed, (b) N2 zero-extent check removed, (c) N3 catch removed (must terminate in the
  test's subprocess), (d) N4 catch removed, (e) N5 in-flight wait removed: each fails its new test. Built outside the
  worktree (`/tmp/p1t_fix2_mut_*`).
- G4: diff against d116ac0 only in the Exercises paths + `docs/P1T_FIX2_*.md`, `artifacts/p1t_fix2/**`, `logs/**`; no new
  core `__global__`; reviewer probe files byte-identical.
- Stop rule: a failed gate is reported as is; no test, probe or tolerance is weakened.

## 23:20 CPython 3.12 ordering check (N5 premise)

- Fetched CPython v3.12.3 sources (the interpreter here is 3.12.3) into `/tmp/p1t_fix2_scratch/cpython/`
  (sha256 pylifecycle.c 11c54b0c..., pystate.c 15965ddb..., ceval_gil.c a9621cb0...).
- `Py_FinalizeEx` (pylifecycle.c:1782): `wait_for_thread_shutdown` (1800) -> `_PyAtExit_Call` (1815) ->
  `_PyInterpreterState_SetFinalizing` / `_PyRuntimeState_SetFinalizing` (1832-1833) -> `runtime->initialized = 0`
  (1834). So every Python `atexit` callback runs while `Py_IsInitialized()` is 1 and before the runtime is marked
  finalizing. Premise of the order's N5 design confirmed for 3.12.3.
- `take_gil` (ceval_gil.c:337) calls `PyThread_exit_thread()` when `_PyThreadState_MustExit` (pystate.c:3032: the
  runtime/interpreter finalizing thread is set and is not this thread); `PyGILState_Ensure` (pystate.c:2199) reaches it
  through `PyEval_RestoreThread`. That is the F04/N5 abort: pthread_exit's forced unwind through the noexcept Storage
  destructor -> std::terminate.

## 23:22 Before-reproduction of the round-2 probes (d116ac0 build)

- `artifacts/p1t_fix2/regen_core_cpu_probe.py`: regenerates core_cpu.cpp from any tree with the reviewer's slice
  markers; self-check on the d116ac0 tree reproduces the committed probe byte for byte (`cmp`: identical).
- `artifacts/p1t_fix2/build_rev2_probes.sh before <wt_fix1> <committed core_cpu.cpp>`: both build exit 0 (reviewer's
  compile commands; cpu_native.cpp compiled unmodified with -I<tree>/tc_signal/src).
- `artifacts/p1t_fix2/make_host_checks_copy.py`: path-only copy of host_checks.py (5 literal substitutions, diff
  printed); the original stays byte-identical. Deviation: the original hard-codes `/tmp/p1t_rev2_*` paths and writes
  into `artifacts/p1t_rev2/`.
- Runs (GPU lock): `probe_runs/rev2_host_checks_before_nogpu.{log,json}` (CUDA_VISIBLE_DEVICES= : the reviewer's GPU-less
  condition) and `..._before_gpu.{log,json}`:
  - N1 `empty_sync`: no GPU rc 9 "cudaGetDevice= 100 device= -1 / synchronize returned success" (REPRODUCED); with the GPU
    visible rc 0 (cudaGetDevice succeeds; not a witness).
  - N2 `prefix` rc 1 UBSan `core_cpu.cpp:65:29 ... 4 * 4611686018427387904`; `suffix` rc 1 `:73:9 ... 4611686018427387904 * 4`
    (REPRODUCED, both conditions).
  - N3 `oom_sync` rc -6 `terminate called after throwing an instance of 'std::bad_alloc'` (REPRODUCED, both).
  - N4 `throw_deleter` rc -6 `terminate ... 'std::runtime_error' what(): producer deleter threw` (REPRODUCED, both).

## 23:24-23:36 Fixes (design and prior art at each code site; summary in the report)

- N2 prior-art measurement (no GPU; numpy 2.4.2, torch 2.11 CPU): np.empty refuses (0, 2**62, 4), (2**62, 4, 0),
  (0, 2**63-1) and (0, 2**61) float32 ("array is too big"), accepts (0, 2**60); torch.empty refuses (0, 2**62, 4) ("Stride
  calculation overflowed") and (2**62, 4, 0) ("Storage size calculation overflowed") but accepts (0, 2**63-1) and
  (0, 2**61) float32. Rule adopted = NumPy's: itemsize x every nonzero extent must fit int64.
- tc_signal: `checked_math.h` (ck_extent_bound; ck_prod / ck_shape_numel no longer exempt zeros); `dlpack_operand.cpp`
  validate_tensor (extent bound before the empty-operand refusal); N4 `call_deleter_contained` in ManagedOwner::reset;
  release section rewritten for N1/N3 (allocation-free in-place two-phase sync_and_release_all_impl with a fixed
  DeviceSyncSet / SyncReport, event-pool slot reserved at event creation, pending push caught -> fallback, release-path
  failures recorded and raised once by the next synchronize, `if (!os) throw bad_alloc`); `init_pending_state()` at module
  import; TEST HOOK `_test_inject` (named one-shot sites); `tuning.h` kSyncMaxDevices / kSyncReportMax; `__init__.py`
  synchronize docs + atexit logs to stderr; README sections.
- core: `bindings.cpp` checked_extent_bound / checked_numel / checked_nbytes (zero-extent rule), `__dlpack__` refuses before
  the fence, N4 catch in import_release_impl, N5 ImportReleaseLifecycle (mutex, closing, in_flight, condition variable)
  + `_dlpack_begin_shutdown`, extended `_dlpack_release_stats`; `kernels.cu` numel_of (zero first, checked) and
  contiguous_strides (checked, no unused product); `core.h` comment; `__init__.py` `_DLPACK_SHUTDOWN_WAIT_S = 5.0` knob
  and the atexit hook (logs, never raises).
- Builds: `logs/p1t_fix2_core_build_r1.log` (exit 0), `logs/p1t_fix2_tc_signal_build_r1.log` (exit 0, 0 warnings).

## 23:37 First after-run of the round-2 probes (r1 build) and a finding about the reviewer's oom_sync probe

- nogpu: empty_sync rc 0 "SAFE ERROR ... cudaErrorNoDevice"; prefix/suffix rc 0 "SAFE REFUSAL", no UBSan; oom_sync rc 0
  (but its what() was the TRUNCATED "tc_signal.synch": std::ostream swallowed the injected bad_alloc and set badbit) ->
  fixed: `if (!os) throw std::bad_alloc()` (MemoryError instead of a truncated message).
- gpu: oom_sync printed "PASS: nonallocating completion path", then rc -6. gdb
  (`probe_runs/rev2_oom_sync_gpu_backtrace_after.txt`): the terminate is in `pybind11::finalize_interpreter ->
  get_local_internals` from the probe's own `scoped_interpreter` destructor (cpu_native.cpp:42): the probe armed "the next
  allocation fails", our path made none, so the armed failure fired in pybind11's teardown. On d116ac0 the terminate is
  in our code (`sync_and_release_all_impl` -> `std::_Deque_base::_M_allocate_map`, `..._backtrace_before.txt`).
  Supplementary disarm variant (`make_disarm_variant.py`, one inserted line) added; the original probe is reported as is.

## 23:39-23:50 Native harness (tc_signal/tests/fix2_native_harness.cpp) and tests

- ck_census (UBSan): after 18/18 cases OK; on d116ac0 sources 14/18 fail ("empty operand", not overflow; ck_prod
  returns 0 for zero-extent overflow shapes). oom_sweep: first version flagged 3 "k-th allocation not reached" runs:
  the deque's node offset differs between the count run and the re-run, so the criterion was changed to invariant
  failures only, with k swept to max(count, 2) and the push injection required to be hit at least once; prefill extended
  to 0..23. After: 225 runs, 0 failures, 4 injected push failures (all fell back), 0 allocations in drain/sync. On
  d116ac0 sources: terminate at the 1024-prefill bound drain (rc 134).
- tc_signal/tests/test_fix2_findings.py: 19 tests, all pass (first run).
- tensor_cuda/tests/test_dlpack.py: N2 census (empty/import/export), N4 (main, foreign), N5 (4 scenarios); FIX1's
  `test_fix1_f03_empty_zero_extent_is_empty_not_overflow` shape changed (0, 2^62, 2^62) -> (0, 2^30, 2^30) (deviation:
  it pinned the N2 defect).
- First N5 run: 3 failures, all mine. (1) the in-flight staging (a thread blocked in PyGILState_Ensure behind a GIL-holding
  usleep) cannot survive to the hook: the eval loop hands the GIL over at the next bytecode boundary, so the release
  finished before the hook; restaged as "admitted and inside a slow deleter", with `_start_in_flight` waiting until the
  deleter has started. (2) FIX1's finalizing test expected `leaked_interpreter_shutdown`; the closing flag was checked
  first, so it was counted as `leaked_closing`: the order of the two leak checks was swapped (both leak; attribution only).
  Second run: 80 passed (fix1 + fix2 subsets).
- Core rebuilt (`logs/p1t_fix2_core_build_r2.log`).

## 23:52 Source frozen; final builds

- `logs/p1t_fix2_core_build_final.log` exit 0 (pre-existing kernels.cu:3161 `lane` warning; 6 nvlink "Skipping
  incompatible lib*.a" lines appear in this worktree's incremental rebuilds only, not in the fresh mutant builds),
  `logs/p1t_fix2_tc_signal_build_final.log` exit 0, 0 warnings.
- `.so` sha256 (`artifacts/p1t_fix2/build_sha256.txt`): core `7729cb7aff475cbef0d477a63de2e946305d8672bb445252c9c909f0d100b3f5`,
  tc_signal `afd1f01daeea33a4788b17464ba40a5aa02884f6ebd31d36034b8ce2c890b9f6`. Source hashes:
  `artifacts/p1t_fix2/source_sha256_frozen.txt` (re-verified OK before G1/G2).

## 23:53-00:05 G3 mutants (outside the worktree, /tmp/p1t_fix2_mut_*)

- `g3/make_mutants.py` (one asserted edit each, diffs `g3/mut_*.diff`), `g3/run_mutant.sh` (build + the tree's own
  cmp-identical test file). (d) split into d1 (tc_signal) and d2 (core).
- a: 2 failed (N1 device query: `{'raised': False}`; atexit: no log line). c: 2 failed (pending_push rc -6 "terminate
  called after throwing an instance of 'std::bad_alloc'"; native sweep terminate). d1: 6/6 N4 failed (rc -6 "terminate
  called after throwing an instance of 'std::runtime_error'"). b: 44 failed (33+9 DID NOT RAISE, 2 OverflowError from the
  hardened contiguous_strides instead of ValueError). d2: 2/2 N4 failed (rc -6). e: 3 failed (during, nested, bounded;
  details `g3/mut_e_n5_scenarios.log`).
- After the N5 test change below, b/d2/e were re-run with the refreshed test file: same kills (b 44, d2 2, e 3).

## 00:00 N5 witness on d116ac0 and a test-harness correction

- `run_n5_scenarios.py` on the d116ac0 core: "during" rc 0 but no FINAL line and the in-flight deleter cut off at exit.
  Cause: the unfinished deleter's frame keeps `__main__`'s globals alive, so a `__main__` Finalizer is never collected.
  Fix (test harness only): the Finalizer is held by its own module. Re-run on d116ac0: "during" rc -6 "terminate called
  without an active exception" (FIX1's residual reproduced, `before_witness/n5_scenarios_on_d116ac0.log`); on this build
  4/4 N5 tests pass.

## 00:01-00:12 G1/G2 runs

- `run_g1_rev2_probes.sh`: before/after x nogpu/gpu (receipts `probe_runs/rev2_*`); disarm variant before -6/-6, after 0/0.
- Before-witness of the new tests on d116ac0 binaries: tc_signal 17 failed / 2 passed (the 2 are the native-harness tests,
  which compile the CURRENT sources: not a witness; the harness on d116ac0 sources is in `before_witness/native_*`);
  core 50 failed / 10 passed. Many before failures are a missing hook (AttributeError `_test_inject`, KeyError on new
  stats keys), not the defect; the defect witnesses are listed in the report.
- `run_g2_round1_probes.sh`: addon_lifetime 3/3 exit 0, 0/12 incorrect, 0/12 reused each; core_edges 0; core_shutdown
  normal 0 / finalizing 0; numerics 0; round-1 host_checks identical to FIX1's; core_device_drain 3 (INCONCLUSIVE, one GPU).
- Voice runner T1-T6: exit 0, PASS 236/236, max_N 2.9320924600942896e-06, min margin 6.821x (identical to FIX1).
- Combined session: 37 failed, 2678 passed, 47 skipped, 15 errors in 239 s; compare: 52 = 52 baseline ids, 0 new, G1 PASS.
- `g4_scope.py`: G4 PASS (all changed paths in scope; 0 new __global__, 0 device atomics / fast-math / TF32 /
  cudaDeviceSynchronize / host callbacks / terminate-abort calls added; probe files byte-identical).

## Prior art (as annotated at the code sites)

- N1: PyTorch CUDACachingAllocator process_events (in-place FIFO event query); CPython atexit prints, never raises.
- N2: NumPy array constructor rule (measured above); GCC 5 checked builtins, C23 ckd_*, CERT INT32-C; UBSan.
- N3: SQLite OOM harness / Linux failslab (fail-the-Nth / named-site fault injection); C++ Core Guidelines E.16.
- N4: C++11 noexcept -> std::terminate; C++ Core Guidelines C.36/E.16; Sutter & Alexandrescu (2004) item 51.
- N5: CPython 3.12.3 Py_FinalizeEx ordering (verified in source); threading._shutdown; percpu_ref-style quiescence.
- Unverified in this seat (lead to check): "numpy ctors.c PyArray_NewFromDescr_int is_zero overflow", "c10
  computeStorageNbytesContiguous safe_multiplies_u64", "percpu_ref_kill_and_confirm", "C++ Core Guidelines E.16",
  "Sutter Alexandrescu C++ Coding Standards item 51", "SQLite malloc failure testing".

## 00:20 Report

- `docs/P1T_FIX2_REPORT.md` written (per-item table, gate table, probe lines, G2 verdict, compare output, mutant table,
  N5 design, prior art, deviations D1-D16, residuals, diff stat, .so sha256s). G1 PASS, G2 PASS, G3 PASS, G4 PASS.
