# P1T-FIX8 ledger

Seat: Opus 5.5 (high), the resumed FIX7 seat. Order: `orders/P1T_FIX8_ERROR_VISIBILITY_AT_EXIT.md` (f9e6bdd),
immutable. Writable target: `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix8`
(branch voice-fix8, 1cd8824 + f9e6bdd), plus scratch `/tmp/p1t_fix8_*`. Environment: RTX 4070 SUPER (one GPU),
native driver, Python 3.12.3, load average about 9.

## 2026-10-13 15:08 Registration (written before any source change, build, gate run or probe run)

- Before arm: `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_p1trev8` (1cd8824,
  read-only), binaries core `750112413b78450937ba126e56ab700e3f9e16d1092f88af239b13fee48f9245`, tc_signal
  `ca5770c152f6946921ce8c06681a2ecad6beaa81232b948cceac2a7a99605c37` (= FIX7 + L1, 6991939). The round-8 probes run
  from that tree (they find their package through parents[3]); probes that write into `artifacts/p1t_rev8/` run as
  path-only copies that write under `/tmp/p1t_fix8_*` / `artifacts/p1t_fix8/`.
- Reviewer files registered before any change: `artifacts/p1t_fix8/probe_files_before.sha256` (114 probe files of
  rounds 1-8), `artifacts/p1t_fix8/rev8_all_files_before.sha256` (57 files under artifacts/p1t_rev8). They must stay
  byte-identical (G5).
- Gates as the order states them (G1-G5). Budget rule (order "How to run"): the combined session, Voice's runner and
  the mutants run ONCE, at the end, on the final build; re-run only if product code changes after.
- Plan:
  - V8-01: record every failed marker call (create, record, query, destroy) as a tagged FailureRecord before its slot
    entry is consumed, through the existing raise-once path; when the marker cannot be created / recorded or its
    query fails, the stream sync that follows is diagnosed by a watchdog thread (no CUDA call beyond the 11 the
    reviewers' host traces stub; cudaStreamQuery is not among them, so it is not used); "finished" only after a
    successful query or a successful stream sync.
  - V8-02: allocation-free fallbacks (C stdio, static strings, cudaGetErrorName) in both native exit loggers; the
    runners become the final reader after the hook in every case; records consumed by a raise the hook then could not
    write are kept as a POD exit report until the hook acknowledges writing it.
  - Sweeps (k) and (l) over `git diff 9df5bef..HEAD -- tensor_cuda tc_signal`; fix what they find, with tests.
- Rules: no git writes, no subagents, no deletes, never kill a process I did not start, GPU under the flock, absolute
  output directories, no defect residuals, prior art at code sites + here + the report.

## 15:10-15:43 Implementation (sources last changed 15:42:53; core built 15:30, tc_signal 15:43)

- V8-01 (`tc_signal/src/dlpack_operand.cpp`):
  - `note_exit_marker_failure` records each failed marker call (create, record, query non-NotReady, destroy) as
    `{call, Purpose::kExitWaitMarker, device, err}` through `note_unraisable_failure` BEFORE `cudaGetLastError`;
    counted in `exit_marker_failures`. New tags: `FailedCall::kEventCreate`, `Purpose::kExitWaitMarker`,
    `Purpose::kCreateEvent`.
  - `wait_device_work_diagnosed` returns whether completion was PROVED (its query returned cudaSuccess).
  - Otherwise `sync_device_legacy` runs the stream sync under a `DeviceWaitWatchdog` thread on the same schedule
    (`DeviceWaitSchedule` shared); "finished" is printed only after a successful query or stream sync.
  - Only the 11 CUDA calls the reviewers' host traces stub are used (no cudaStreamQuery: the order's candidate would
    not link in `release_trace.cpp` / `diagnostic_errors_cpu.cpp`).
  - Test hooks `cuda:exit_marker_*`, `alloc:exit_log`.
- V8-02, add-on:
  - `summarize_unreported` / `write_summary_line` give the allocation-free count and first record; `exit_log_unreported`
    falls back to them and then consumes.
  - `note_raised_at_exit` keeps a plain-data summary of what synchronize() raised inside the runner until the hook
    acknowledges (`_exit_report_written`, new raw METH_NOARGS); `exit_log_unwritten_report` writes it otherwise.
  - The runner is the last reader after the hook in every case (`exit_runner_last_reader`); it no longer calls
    `e.what()` (it allocates inside a noexcept function) and uses `ascii_message_of`.
  - The hook acknowledges after writing every line.
- V8-02, core:
  - `dlpack_unreported_unraisable_raw` (kernels.cu) is an allocation-free snapshot.
  - `core_exit_last_reader` writes the rich message, else `write_core_summary`, then acknowledges; it also writes an R7
    report the hook raised (`dlpack_raise_unraisable` keeps it while the runner runs) but did not acknowledge
    (`_dlpack_exit_report_written`).
  - The runner is the last reader in every case. Test hook `_dlpack_test_fail_exit_report`.
  - Stats keys `shutdown_log_fallbacks`, `unraisable_unreported`, `exit_report_unwritten`.
- Sweep (k), raising paths, built before consumed:
  - `cuda_guard.h`: `throw_after_consuming` (report first, then `cudaGetLastError`), and launch checks peek.
  - `conv.cu` `check_cuda`; `acquire_event` (records `kCreateEvent` on `bad_alloc`); `RetainUntilComplete`'s pending
    error.
  - core `kernels.cu` `throw_after_consuming` / `throw_cuda_report_after_consuming`, used by `raise_cuda_failure`, fill_
    (pending error and launch check), the export fence, from_dlpack and empty.
  - Literal record-then-consume order in the current-device query, the prior-slot read and both guard destructors.
- Docs: README "Interpreter exit" (FIX8 bullet, counters, test description); both hook docstrings; runner docstrings.
- Prior art at the code sites: FIX5 V5-02; the CUDA 12.6 error-slot / event contracts; POSIX write(2)
  async-signal-safe logging; SQLite OOM testing; C++17 bad_alloc; the hung-task watchdog (unverified, lead to check
  "hung_task_timeout_secs").

## 15:20-15:44 Tests and first runs (test files 15:27 / 15:31)

- New `tc_signal/tests/test_fix8_findings.py` (29 cases) and the core FIX8 block of `tensor_cuda/tests/test_dlpack.py`
  (6 cases). All pass on this build. Full `tc_signal/tests`: 788 passed. Full `test_dlpack.py`: 257 passed.
- My mistakes, fixed before acceptance:
  - the core `.so` was not rebuilt after the stats-key edit (a KeyError in the first core test run);
  - the OOM shim used `dlsym(RTLD_NEXT)`, which fails for an RTLD_LOCAL-loaded libcudart (a segfault): it now opens
    the runtime as FIX1's F05 shim does;
  - the first core-side `throw_after_consuming` sat in kernels.cu's anonymous namespace: renamed into an external
    `throw_cuda_report_after_consuming` for bindings.cpp;
  - a stale FIX7 signature in the G5 checker.
- CUDA 12.6 Runtime API (event, error pages) and CPython 3.12.3 errors.c citations fetched through the AtlasForge web
  proxy (`artifacts/p1t_fix8/citations/`).

## 15:44-16:17 Probes (all rounds), sweep tables, citations (citations 15:48-15:49)

- Round 8, both arms (`artifacts/p1t_fix8/run_rev8_probes.sh`, `make_rev8_copies.py`; the before arm is wt_p1trev8 itself,
  read-only):
  - G1 probes: `diagnostic_errors_cpu` 1 -> 0, `verify_exit_log_cpu` 1 -> 0, `verify_core_exit_log_cpu` 1 -> 0;
  - `diagnostic_events_gpu` as written: 1 -> **1**. Every row now passes (each message names its call); the parent
    fails only its "interpreter exit is waiting" check for create / record. Cause: under torch 2.11,
    `torch.cuda.stream(torch.cuda.ExternalStream(0))` makes a NON-BLOCKING stream current (receipt
    `artifacts/p1t_fix8/extstream_receipt.txt`), so the legacy-stream sync has nothing to wait for and no exit wait
    exists to diagnose. The product must not print a wait that did not happen.
  - A scenario copy with the work on torch's default (legacy) stream (`probe_copies/diagnostic_events_gpu_legacy.py`,
    one line): 1 -> 0. The before arm shows the silent waits and the unnamed failures; the after arm shows 6 lines per
    row and every failure named.
  - Controls and inherited probes: unchanged on both arms (14 / 1067 / 17 / 22 / 36 / 3; fix7 115, fix56 157,
    semantics 504, all with zero skips; multidevice exit 3).
- Rounds 7, 6, 5, 4 and 1-3, after arm (FIX7's runners, copied and retargeted): outcomes equal FIX7's after-arm runs.
  - Round 4's regenerated core_interop import first failed to COMPILE: the regenerated `from_dlpack` calls the new
    `throw_cuda_report_after_consuming`, which the probe's embedded 62c8d9f slices lack. The regen script copy now
    inserts the helper's text from the tree; rerun: it refuses (rc 1), as FIX7.
- Sweep tables written (report): (k) 34 rows, FIXED 16 / NOT A FAILURE 3 / OK 15; (l) 7 last readers (FIXED 6, OK 1)
  and 13 catches (FIXED 6, OK 7); FIX7's i24 / i25 / j24 / a3 / a9 corrected; N1-N3; U1-U8.
- Mutants made (`artifacts/p1t_fix8/g4/make_mutants.py`: e, f, g; diffs `mut_{e,f,g}.diff`).
- The end-of-round battery runs once, on the final build: mutants, Voice, combined session, torch interop
  (`artifacts/p1t_fix8/run_final_battery.sh`, detached, marker `logs/p1t_fix8_final_battery.done`).

Times corrected at close: three of the headings above first carried estimated times. They now give the file times
(`ls --time-style`, recorded at 16:40).

## 16:17-16:33 End-of-round battery (once, on the final build; `run_final_battery.sh`, detached, marker 16:33:32)

- Binaries at the probes, at the combined session and at close are identical: core
  `d6b1b7ad1d87e0d7d5eed70cff94b33e469e6f18b857d78679110e6b2dce3c24`, tc_signal
  `29409a0026e05df16dfdca81f70430cbcf2cb3d3f56ecd1556eb2ab55b2183b1`. Test files are unchanged since the combined
  session.
- G4, 16:17-16:20: every mutant fails its new tests and its round-8 probe; all KILLED.
  - (e): `-k v801` 17 failed / 2 passed; `diagnostic_errors_cpu` rc 1; `diagnostic_events_gpu` rows rc 1 "erased".
  - (f): `-k v802` 2 failed / 3 passed; `verify_exit_log_cpu` rc 1.
  - (g): `-k fix8_v802` 2 failed / 2 passed; `verify_core_exit_log_cpu` rc 1.
- Voice, 16:20-16:21: `"verdict": "PASS"`, 249 / 249, runner exit 0. Voice `git status --short` shows 0 lines before
  and after.
- Combined session, 16:21-16:33: 37 failed, 3036 passed, 47 skipped, 15 errors. Compare: exactly the 52 baseline ids,
  0 new, "G1 PASS" (`logs/p1t_fix8_g3_compare.txt`).
- Real-torch interop: 22 passed (torch 2.11.0+cu130).

## 16:35-16:45 Close

- G5: PASS (`artifacts/p1t_fix8/g5_scope.txt`).
  - Reviewer files byte-identical: 114 probe files and the 57 files under `artifacts/p1t_rev8`.
  - Diff stat recorded (`artifacts/p1t_fix8/diff_stat.txt`): 11 tracked files, +937 / -115, plus the new
    `test_fix8_findings.py` (456 lines).
- Report filled. Gate verdict:
  - G2-G5 PASS.
  - G1 PASS for three probes. It is RED as written for `diagnostic_events_gpu.py`: the probe's work is not on the legacy
    stream under torch 2.11. Its legacy-stream copy gives rc 1 -> rc 0. Reported as D2 / D12 for the lead's decision.
- Prior art:
  - at the code sites, in the report and in the 15:10 entry above;
  - the hung-task watchdog analogy is unverified: lead to check ("hung_task_timeout_secs", "softlockup detector");
  - SQLite's OOM testing: https://www.sqlite.org/testing.html, as the reviewer cited it;
  - DeMillo, Lipton & Sayward (1978) for mutation testing.


## 16:41-17:12 Lead send-back L2 (stopping the exit device wait's watchdog)

- Lead ruling: D2 / D12 ACCEPTED. The lead reproduced `ExternalStream(0)` as non-blocking. The legacy-stream copy is
  the G1 evidence; the as-written run stays recorded as RED with that reason.
- Finding: `DeviceWaitWatchdog::stop()`.
  - (a) A failed join detached a thread that could still be inside `run()`: a use-after-free of `mu_` / `cv_` / `s_`.
  - (b) A failed stop-lock could miss the notify: a silent wait until `s_.next`.
- Fix, `tc_signal/src/dlpack_operand.cpp`, tc_signal only:
  - `run()` is `diagnose(); exited_.store(true, release)`. Its last action is the store; no member is touched after it.
  - `stop()` sets `done_`, then loops until `exited_`: lock attempt (a failure is caught and counted), `notify_all`,
    a check, a sleep of `kExitDevicePollMs`. Then it joins; a failed join is counted and detached. No early return.
  - New counters: `exit_device_wait_watchdog_lock_failures`, `exit_device_wait_watchdog_detaches`.
  - Code-site annotation, with the stated limit (a blocked stderr write) and prior art.
  - README updated.
  - Pre-L2 copies kept in `/tmp/p1t_fix8_l2/before/` (tc_signal sha256 `29409a00...`).
- Build: `./build.sh 89` as ordered (`logs/p1t_fix8_l2_tc_signal_build.log`, rc 0). New tc_signal sha256:
  `30aee4e67fbe80320863cf9d0d4b7748dca46c0db7a3689f70bcc9d9e8dc782b`. The core is unchanged (`d6b1b7ad...`).
- Tests, 4 new in `test_fix8_findings.py`, through an LD_PRELOAD interposer of the thread, mutex, condition-variable,
  join and detach calls:
  - control;
  - stop-lock fails n in {1, 3}, with the notify after each failed lock withheld;
  - join fails once, with the watchdog held 50 ms after its wakeup.
  - Measured on this build: stop to join 0.0011 / 0.0022 / 0.0043 s; on the join failure, 0 watchdog calls after the
    detach.
- Re-run (economical, as the lead listed):
  - round-8 G1 probes after arm: diag 0, verify_exit_log 0, verify_core_exit_log 0, events as written 1 (accepted),
    legacy copy 0;
  - `test_fix8_findings.py`: 33 passed;
  - mutants, all KILLED: e_l2 (17 failed / 2 passed; diag rc 1), f_l2 (2 / 3; verify rc 1), h (3 / 1);
  - Voice: PASS 249/249;
  - combined session: 37 failed / 3040 passed / 47 skipped / 15 errors; compare 52 ids, 0 new, G1 PASS;
  - torch interop: 22 passed;
  - G5: PASS.
- Mutant (h): the first failing assertion in each test is a counter. Witness runs with the same shim show the
  behaviour: stop-lock 4.62 s silent; join, 2 watchdog calls after the detach.
- Tooling: `make_mutants.py` / `run_mutant.sh` take an optional tag, and case (h) was added; the G5 checker lists
  `diagnose()`; new runners `run_rev8_g1_l2.sh` and `run_final_battery_l2.sh`. Pre-L2 copies are in
  `artifacts/p1t_fix8/l2/`.
- Prior art:
  - C++11 release / acquire publication (Boehm & Adve 2008) for the exit flag;
  - the lost-wakeup problem and Lampson & Redell (1980), Mesa monitors, for repeated notification;
  - both unverified here, lead to check (search terms in the report);
  - LD_PRELOAD interposition (FIX7 shims) and mutation testing (DeMillo, Lipton & Sayward 1978).
