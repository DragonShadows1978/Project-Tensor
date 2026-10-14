# P1T-FIX3 ledger (order `orders/P1T_FIX3_EXIT_QUIESCENCE.md`, fork commit 021abbb)

Seat: Opus 5.5 (high), Claude Agent-tool seat. Writable target: this worktree (`voice-fix3`) and `/tmp/p1t_fix3_*`.
Append-only. Times are local (EDT).

## 00:54 Registration (before any source change or gate run; immutable from here on)

Identity at start:
- HEAD 021abbbced0ae665b845721119808548758c0b80 (= voice-fix2 5f9f955 + round-3 receipts + this order).
- Order sha256 `20e032bf9f1664cc9a6c3ec361bbd9e741e03238aff546e15a5a434248eb8921`.
- Frozen sources (5f9f955 content) sha256:
  - `tensor_cuda/src/bindings.cpp` `ffebd71f39eab5eee73ae00be9bb73f29dcf7501b4523a2ab4b881a947c2f183`
  - `tensor_cuda/tensor_cuda/__init__.py` `ff5dc737e1d92f256db1864a6e9cff7585b1f78c58b6b4a3fad2dc61066505d9`
  - `tensor_cuda/tests/test_dlpack.py` `08788cd8e8e62e65adb758949831a90edd8d1d798ee91e9c8986f911dd1fd7c2`
  - `tc_signal/src/dlpack_operand.h` `caf4223112b70eaa5d53f75c65800f90b889452e6defea6b6078b90481d4def1`
  - `tc_signal/src/dlpack_operand.cpp` `600bde3398e931984461b537df976630b0d0d4c58ea3e4d881b93c307730cce9`
  - `tc_signal/README.md` `5eae63822389960d10f75a51af5166f619e5e607970fc955639006c823a92c54`
- Reviewer probe files (rounds 1-3), aggregate sha256 of `sha256sum` lines:
  `36502118ca786939c77714f46c4041760d4c8d4b5f73f69e6056347aec97a82b`.
- "Before" binaries (wt_fix2, 5f9f955, read-only): core `7729cb7aff475cbef0d477a63de2e946305d8672bb445252c9c909f0d100b3f5`,
  tc_signal `afd1f01daeea33a4788b17464ba40a5aa02884f6ebd31d36034b8ce2c890b9f6` (match the FIX2 report).
- Host: RTX 4070 SUPER (one GPU), native driver 595.99.02, CUDA 12.6, CPython 3.12.3, g++ 13.3.0.

Mechanism check (HOUSE_RULES §3): the mechanism doc `docs/P1T_REV3_REPORT.md` defines V3-01 on
`tensor_cuda/src/bindings.cpp` `dlpack_begin_shutdown` + `__init__.py` `_dlpack_shutdown_hook` (exercised by
`lifecycle_timeout.py` through a verbatim slice and by `core_exit_overrun.py` through the real built module), and V3-02
on `tc_signal/src/dlpack_operand.cpp` `~RetainUntilComplete` / `note_push` / the bound comparison (exercised by
`pending_bytes.cpp`, which `#include`s the real source). The order's Exercises match. No scope mismatch.

Gates (copied from the order; thresholds fixed now):
- G1 (closed):
  - `lifecycle_timeout.py` (modes control/short/default) exits 0 in every mode on the after build, and `default`
    (5.5 s deleter vs 5 s wait) exits non-zero (abort) on 5f9f955; `core_exit_overrun.py` `during` and `bounded` exit
    0 on the after build within the probe's timeout 20, and `bounded` aborts or fails on 5f9f955.
  - `pending_bytes` (queue and entry) prints no UBSan `runtime error` and exits 0 on the after build; reports (UBSan,
    rc != 0) on 5f9f955.
  - the new regression for each finding passes on the after build.
- G2 (no regression): every round-3 probe passes (two-GPU probes may be INCONCLUSIVE exit 3 on this one-GPU host);
  round-1 `addon_lifetime` 3/3 runs exit 0 with 0 incorrect trials, round-1 core_edges, core_shutdown (normal and
  finalizing) and numerics_and_streams exit 0; Voice runner T1-T6 PASS 236/236; combined core + tc_signal session
  FAILED/ERROR ids == the 52 baseline ids (compare script prints `G1 PASS`).
- G3 (teeth): mutant (a) = the old bounded return restored in `dlpack_begin_shutdown` -> the new V3-01 overrun test
  FAILS; mutant (b) = the aggregate pre-launch check removed -> the new V3-02 pre-launch test FAILS. Mutant trees under
  `/tmp/p1t_fix3_mut_{a,b}`, built outside the worktree.
- G4 (scope): diff vs 5f9f955 touches only the Exercises paths + `docs/P1T_FIX3_*.md`, `artifacts/p1t_fix3/**`,
  `logs/**` (plus the lead's receipt/order commits already on HEAD, listed separately); probe files byte-identical
  (aggregate hash above); no new core `__global__`.

Design decisions registered before coding:
- V3-01: `dlpack_begin_shutdown` keeps its single-argument signature (the reviewer's `lifecycle_slice.cpp` driver binds
  it as `close(bound)`), closes admission, and waits for `in_flight == 0` with NO return before quiescence. The
  argument becomes the delay of the first stderr diagnostic. Later diagnostics follow a geometric (doubling) schedule
  measured from the start of the wait (bounded rate without a new literal). Logging is done from C++ with the GIL and
  the mutex released. A thread that calls the hook from inside its own admitted release is not waited for by itself
  (thread-local depth), so that case cannot self-deadlock.
- V3-02: the aggregate operand byte count of a call is computed with checked addition in the `RetainUntilComplete`
  constructor (constructed after validation, right before the first launch, in every entry point), refusing with
  ValueError on overflow; the destructor reuses that value (no addition there). Queue accounting
  (`note_push` / `note_pop`): SATURATE (chosen over drain-before-add; justification in the report), with a sticky
  saturated state that keeps the bound comparison conservative until the list is empty.
- Regenerated/adapted probe inputs (if needed) are made by scripts with asserted substitutions, kept under
  `artifacts/p1t_fix3/`, and the originals stay byte-identical.

## 00:56-01:20 Implementation

V3-01 (core):
- `tensor_cuda/src/bindings.cpp`:
  - design comment block "P1T-FIX3, round-3 review finding V3-01" (consequence, GIL handling, self-exclusion, prior
    art); FIX2's "Bounded" bullet replaced by a pointer to it;
  - `thread_local long long t_admitted_depth` (admitted releases on the current thread's stack);
  - `import_release_impl`: `++t_admitted_depth` after admission, `--` after leaving; `notify_all` on every leave
    (the hook may wait for a nonzero count, its own thread's releases);
  - `dlpack_begin_shutdown(double first_diag_s)`: same single-argument signature (the reviewer's slice driver binds
    `close(bound)` to it); closes admission; `while (!cv.wait_until(lk, next, in_flight <= own))` logs one stderr line
    (count running, seconds waited) with the mutex dropped and keeps waiting; `next = now + max(waited, 1 tick)`
    (doubling); NO return before quiescence; returns `{in_flight_at_close, still_in_flight, timed_out (= overran),
    waited_us, diagnostics, own_thread_in_flight}`; new counter `shutdown_diagnostics`;
  - `#include <cstdio>` (diagnostics without the GIL).
- `tensor_cuda/tensor_cuda/__init__.py`: knob comment (now the first-diagnostic delay, wait unbounded); docstring of
  `_dlpack_shutdown_hook` states the consequence (a never-returning deleter makes exit wait; producer contract
  violation; honest vs abort; Ctrl-C does not interrupt, SIGTERM/SIGKILL do); an unusable knob value becomes NaN
  (diagnose at once, still wait) instead of skipping the wait; a final "finished after X s; interpreter exit
  continues" line after an overrun.
- `tensor_cuda/tests/test_dlpack.py`: `_FIX2_N5_SCRIPT` loses the "bounded" mode and its `os._exit(0)` (it pinned
  V3-01); new modes "overrun <s>", "idle", "self". New tests: `test_fix3_v301_script_has_no_immediate_exit_escape`,
  `test_fix3_v301_admitted_release_finishes_before_finalization_ordinary_exit[0.1|5.5|12.0]` (default knob 5.0),
  `test_fix3_v301_idle_exit_does_not_wait`, `test_fix3_v301_hook_inside_own_release_does_not_wait_for_itself`.
  FIX2's `test_fix2_n5_the_wait_is_bounded_and_reported` removed (deviation D1 in the report).
- First run (01:03): 8 passed, 1 failed: my own assertion in the self test expected `shutdown_still_in_flight == 1`;
  by construction it counts OTHER threads' releases at return (always 0) and the own release is
  `own_thread_in_flight`. My C++ comment had said the opposite: comment and assertion corrected (01:04), re-run PASS.

V3-02 (tc_signal):
- `dlpack_operand.h`: `RetainUntilComplete::bytes_`; design note; `aggregate_operand_bytes` declared.
- `dlpack_operand.cpp`:
  - `aggregate_operand_bytes` (ck_add over present operands; ValueError "the aggregate fp32 byte count of this call's
    operands (their sum) overflows 64-bit arithmetic; refused ...");
  - constructor: `bytes_ = aggregate_operand_bytes(ops_)` FIRST (before owner reserve, event, launch); every entry point
    constructs the retainer right before its first launch: 9 sites (grep `RetainUntilComplete keep(`: conv1d_run,
    conv_transpose1d_run, fp32_divide_probe, norm_run (the three norms), upsample_run, pad_run, rint_run,
    _test_rint_then_raise, lstm_run), each preceded only by host validation;
  - destructor: `e.bytes = bytes_` (no addition);
  - `note_push` / `note_pop` noexcept, saturating with a sticky `bytes_saturated` state cleared when the list empties;
    `pending_over_bound()` used by the destructor; stats `bytes_saturated`, `bytes_saturations`.
  - Choice registered at 00:54 (saturate); justification in the code comment and the report.
- No change to any `*_bindings.cpp`: the pre-launch check lives in the retainer constructor that every entry point
  already calls right before its launch (the Exercises allowed binding edits for this check; none were needed).
- `README.md`: validation bullet, bound bullet, Errors table, Tests list, Prior art.
- New `tests/test_fix3_findings.py` (6 tests) and `tests/fix3_native_harness.cpp` (accounting 17 cases, dtor 11).
  Direct harness runs (01:10): accounting 17/17, dtor 11/11, no UBSan line. pytest (01:13): 6 passed.

Builds: `logs/p1t_fix3_core_build_r1.log` (exit 0, only the pre-existing kernels.cu `lane` warning),
`logs/p1t_fix3_tc_signal_build_r1.log` (exit 0, 0 warnings), `logs/p1t_fix3_core_build_r2.log` (incremental after a
comment-only edit; the 6 nvlink "Skipping incompatible" lines are FIX2's D14 noise; the .so is byte-identical).
After binaries (`artifacts/p1t_fix3/build_sha256.txt`): core `35b8261680760851e5d150f851d58ad8f0fbec617deb225dd6d7565010487a23`,
tc_signal `222861219e29e8486d7a6f92ee874f5b4342c6f26d11a3440bf9bb99787e535a`. Final sources:
`artifacts/p1t_fix3/source_sha256_final.txt`.

Prior art recorded at the code sites: CPython threading._shutdown / wait_for_thread_shutdown (unbounded join),
RCU grace periods (McKenney & Slingwine 1998) and percpu_ref, kernel hung-task / RCU stall warnings (repeated
reports); saturating arithmetic (MMX/SSE PADDS, ARM QADD), GCC 5 checked builtins / C23 ckd_add, CERT INT32-C, UBSan.
Unverified (lead to check): "synchronize_rcu grace period readers McKenney 1998", "rcu_cpu_stall_timeout repeat
interval", "hung_task_warnings", "PADDSW saturating add MMX 1997".

## 01:01-01:30 Gate runs (all GPU-touching commands under `flock -w 900 /tmp/forge-gpu.lock timeout ...`)

- 01:01 smoke: reviewer `core_exit_overrun.py bounded` on the after build: rc 0, one diagnostic at 5.0 s, "finished
  after 5.5 s", `FINAL finalizing=True`.
- 01:11 probe tooling (`artifacts/p1t_fix3/`): `regen_lifecycle_slice.py` (selfcheck identical=True: regenerating from
  5f9f955 reproduces the committed `lifecycle_slice.cpp` byte for byte; base slice sha256 0a56c6f3... = review pin),
  `make_probe_copies.py` (host_checks 5, operand_census 2, real_metadata_plans 2, release_failures(before) 1 asserted
  substitutions; diffs next to each copy under /tmp/p1t_fix3_probes/<arm>/), `make_pending_bytes_adapt.py` (the
  reviewer-authorized entry-branch adaptation), `build_rev3_probes.sh` (all 12 builds exit 0),
  `run_rev3_probes.sh`, `run_rounds12_probes.sh`. Core_cpu's embedded slices are verbatim in the after sources
  (checked against source_slices.json), so core_cpu.cpp is compiled unmodified on both arms.
- 01:12-01:14 round-3 BEFORE (5f9f955; `probe_runs/rev3_summary_before.txt`): lifecycle short/default abort (134 =
  SIGABRT under timeout, "FATAL: exception not rethrown"); core_exit_overrun bounded abort (134); pending_bytes
  entry/queue UBSan (pending_bytes.cpp:28, dlpack_operand.cpp:602); adapted entry (gpu) UBSan at
  dlpack_operand.cpp:731; everything else as in the review (addon 12/12 correct, release_failures 66 passed, two-GPU
  probes exit 3).
- 01:14-01:16 round-3 AFTER (`probe_runs/rev3_summary_after.txt`): lifecycle control/short/default rc 0 in both
  conditions (short: diagnostics at 0.1/0.2/0.4 s, then finished, FINALIZING True); core_exit_overrun during and
  bounded rc 0 (bounded wall 7.5 s); pending_bytes queue rc 0 (prints 9223372036854775807, no UBSan); adapted entry
  rc 0 "SAFE REFUSAL: ... aggregate ... overflows ..."; unmodified entry still rc 1 at pending_bytes.cpp:28 (the
  probe's own copy of the old statement, by construction); host_checks identical to before per condition (gpu
  oom_sync -6 after "PASS: nonallocating completion path" on BOTH arms = FIX2 D4); ck_census 18/0; operand_census 17;
  real_metadata 129; addon 12/12; core_edges, numerics PASS; release_failures 66 passed; two-GPU probes exit 3.
- 01:16-01:17 round 1 and round 2 on the after build (`probe_runs/round12_summary_after.txt`): addon_lifetime 3/3 runs
  exit 0, 0/12 incorrect each; core_edges 0; core_shutdown --normal 0 and finalizing 0; numerics 0; round-1
  host_checks identical to FIX2's accepted receipt (`round1_host_checks_vs_fix2.txt`: True); core_device_drain 3;
  round-2 host_checks (FIX2's committed copier + regenerated core_cpu, selfcheck identical=True) all 20 cases as in
  FIX2 (nogpu all rc 0; gpu oom_sync -6 = D4).
- 01:17 Voice runner T1-T6: exit 0, verdict PASS, 236/236 (T1 11, T2 144, T3 28, T4 12, T5 28, T6 13), max_N
  2.9320924600942896e-06, min margin 6.821067299957125 (T4/shared__LSTM/max_1): identical to FIX2.
- 01:18-01:22 combined session: 37 failed, 2689 passed, 47 skipped, 15 errors; compare: 52 = 52 baseline ids, 0 new,
  0 no longer failing, `G1 PASS`. Failures only in the baseline APA files. Passed +11 vs FIX2 (2678): core +6 new
  -1 removed, tc_signal +6.
- 01:23-01:25 G3: mutant (a) core sha 8b5e79d2...: 2 failed (overrun 5.5 and 12.0: deleter_done [False]), 7 passed;
  reviewer probe bounded on mutant (a): 134 "FATAL: exception not rethrown". Mutant (b) tc_signal sha d636ff01...: 6/6
  failed (gpu: the unchecked call launched on the fake addresses, cudaErrorIllegalAddress inside the test subprocess;
  nogpu: RuntimeError instead of ValueError; native: aggregate cases and D2 fail). GPU sanity after: rint correct,
  nvidia-smi P8 218 MiB.
- 01:27 G4 r1: FAIL printed by two CHECKER defects (diagnosed, not scope violations): (1) the probe aggregate hash
  was recomputed through /bin/sh (dash), whose glob order differs from bash's (same lines; bash reproduces the
  registered 36502118... exactly); (2) the destructor scan flagged FIX2's pre-existing test-injection
  `throw std::bad_alloc()` that its own `catch (...)` catches inside the same destructor (present in 5f9f955).
  Checker fixed (bash; exempt throws inside try/catch(...)), r1 output kept as `g4_scope_r1_checker_defects.txt`.
  G4 r2: PASS.
- 01:29 before-witness of the new tests on 5f9f955 (`before_witness/`): core fix3 5 failed / 1 passed (5.5 and 12:
  deleter_done False = the defect; self: hook_s 5.0001 = FIX2 waited its whole bound for its own release; 0.1 and
  idle: KeyError 'shutdown_diagnostics', a new stats key, NOT a defect witness); tc_signal prelaunch nogpu 2 failed
  (RuntimeError instead of ValueError). The gpu prelaunch variant was deliberately not run on 5f9f955 (it would launch
  on fake addresses; mutant (b) already shows that).
- Disclosure: the fresh core configure (`./build.sh 89`, as ordered) populated pybind11 v2.12.0 (3e9dfa28) by
  FetchContent GIT_REPOSITORY from github.com (network access by CMake). CPU-only compiles/scripts ran outside the GPU
  lock; every GPU-touching command used it.

## 01:35 Report and final checks

- `docs/P1T_FIX3_REPORT.md` written: per-finding table, full round-3 probe before/after table, gate table, Voice
  verdict line, compare output, mutant table, census map, prior art, deviations D1-D17, residuals, diff stat, both .so
  sha256s.
- Final G4 re-run after the report existed (`artifacts/p1t_fix3/g4_scope.txt`).
- No stray writes: no new `__pycache__` in the worktree, no file in /mnt/ForgeRealm/Project-Voice newer than the order,
  wt_fix2 `git status` clean. Production Project-Tensor was only read (pybind11 headers under tensor_cuda/build/_deps).
- Verdict: G1 PASS, G2 PASS, G3 PASS, G4 PASS. Scratch left (nothing deleted): /tmp/p1t_fix3_probes/{before,after,
  rev2_after}, /tmp/p1t_fix3_mut_{a,b}, /tmp/p1t_fix3_scratch, /tmp/p1t_fix3_slice_check.txt, pytest tmp dirs.
