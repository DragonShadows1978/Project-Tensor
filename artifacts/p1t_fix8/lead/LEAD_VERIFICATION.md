# P1T-FIX8 lead verification (Project-Tensor lead, 2026-10-13, about 17:50 EDT)

All runs were fresh processes, with every GPU run under `flock -w 900 /tmp/forge-gpu.lock`, on native driver
595.99.02.

**Arms:**
- **Before:** the 6991939 build in `wt_p1trev8`, the build the round-8 reviewer reviewed: core `75011241…`,
  `_tc_signal` `ca5770c1…`. Its `git status` shows only committed receipts.
- **After:** this worktree's final build, after the L2 send-back. Both hashes match the seat's L2 amendment, both
  binaries are newer than every source file, and they were identical at the start and end of the chain:
  - core `d6b1b7ad1d87e0d7d5eed70cff94b33e469e6f18b857d78679110e6b2dce3c24`;
  - `_tc_signal` `30aee4e67fbe80320863cf9d0d4b7748dca46c0db7a3689f70bcc9d9e8dc782b`.

**Harnesses:**
- I used the seat's runners with my own arm names and absolute output directories: `lead_run_rev8_probes.sh` (a copy of
  `run_rev8_probes.sh`), `runners/run_rev{7,6,5}_probes.sh`, `run_rounds123_probes.sh` and `g4/run_mutant.sh`.
- Mutant (g) is a fresh tree, `/tmp/p1t_fix8_mut_g_lead`, made from the final sources. The seat's (g) tree predates L2.
- The chain is `lead_chain.sh`, with its log in `lead_chain.log`.

## Checks

| Check | Result | Receipt |
|---|---|---|
| Scope | Every path changed against 1cd8824 is in the order's Exercises, plus the FIX8 order (f9e6bdd), `docs/P1T_FIX8_*`, `artifacts/p1t_fix8/**` and `logs/**` | `git diff 1cd8824` |
| Probe files | `artifacts/p1t_rev*` are byte-identical to HEAD, with no untracked additions | — |
| New core `__global__` | 0 | — |
| Round-8 G1, before (6991939) | `diagnostic_errors_cpu` **rc 1** (4/4 erased); `verify_exit_log_cpu` **rc 1** (OOM arm: `record_still_pending= True`); `verify_core_exit_log_cpu` **rc 1**; `diagnostic_events_gpu` as written **rc 1**; the legacy-stream copy **rc 1** (4/4 rows rc 1). Controls rc 0 | `rev8_before_summary.txt`, `diagnostic_events_gpu_legacy_before.log` |
| Round-8 G1, after | `diagnostic_errors_cpu` **rc 0**; `verify_exit_log_cpu` **rc 0** (OOM arm writes the allocation-free summary); `verify_core_exit_log_cpu` **rc 0**; the legacy-stream copy **rc 0** (4/4 rows rc 0, 24 "waiting" lines). `diagnostic_events_gpu` as written rc 1: its 4/4 rows pass, and the parent's "waiting" check cannot hold on this host (ruling D2 below) | `rev8_after_summary.txt`, `diagnostic_events_gpu_legacy_after.log` |
| Round-8 others, after | `run_cpu_checks` 14/14; `argument_census` 1067; `operand_census` 17; `attribution` 22; `lock_checks` 3; `knob_exit_cpu` and `core_knob_exit_cpu` with controls rc 0; `query_diagnostic_cpu` rc 0; `gpu_regressions` fix7 **115**, fix56 **157**, semantics **504**, all zero skips; multidevice exit 3 | `rev8_after_summary.txt` |
| Round-8 `fix7_cpu_census`, after | **2 of 3 runs rc 1, the third PASS 36/36.** Both failures are a race in the probe harness, not in the product: in the failing child, the test script's daemon thread prints `REFUSED` while the main thread prints `RESULT {…}`. Python's `print` writes the text and its newline as two writes, so the lines interleave as `REFUSEDRESULT {…}`. The probe accepts only lines that start with `RESULT `, so it saw no result (`TypeError: 'NoneType'`). The interleaved JSON meets every check: the deleter finished before finalization, `calls == []`, admission closed, and nothing in flight. FIX7's own test parser already looks for `RESULT {` anywhere in a line (`test_fix7_findings.py:74`) | `rev8_after/fix7_cpu_census_run0.json` and `_run1.json` (both contain `REFUSEDRESULT`), `fix7_cpu_census_rerun{1,2}.log`, `fix7_cpu_census.json` (36/36) |
| Round 7, after | every G1 probe and every control rc 0; `run_cpu_checks` 14/14; `argument_census` 1067; `gpu_regressions` fix56 157 / semantics 504, zero skips; multidevice 3 | `rev7_after_summary.txt` |
| Round 6, after | 30/30 runs rc 0 | `rev6_after_summary.txt` |
| Round 5, after | Everything else rc 0, with these exceptions, unchanged from FIX6 and FIX7: `exit_gate trace` as written and its control give rc 1 (FIX6 D7); `handle_insert_oom_cpu` and `fill_grid_cpu` as written give rc 1 (they embed 1475fae code), while their regenerated forms give rc 0; the two-GPU probes exit 3 | `rev5_after_summary.txt` |
| Rounds 1–3, after | `addon_lifetime` 3 runs, 12/12 each with 0 incorrect; `core_edges`, `core_shutdown` (normal and finalizing), `numerics_and_streams` and `host_checks` all rc 0; `core_exit_overrun` during (3.1 s) and bounded (7.6 s) rc 0; `core_device_drain` exit 3 | `rounds123_after_summary.txt` |
| Mutant (e), marker record removed | tests `-k v801`: 17 failed, 2 passed; `diagnostic_errors_cpu` rc 1 (`error: null`); `diagnostic_events_gpu` rows rc 1. **KILLED** | `mut_e_summary.txt` |
| Mutant (f), add-on allocation-free fallback removed | tests `-k v802`: 2 failed, 3 passed; `verify_exit_log_cpu` rc 1. **KILLED** | `mut_f_summary.txt` |
| Mutant (g), core allocation-free fallback removed (fresh tree, final sources) | tests `-k fix8_v802`: 2 failed, 2 passed; `verify_core_exit_log_cpu` rc 1. **KILLED** | `mut_g_summary.txt` |
| Mutant (h), watchdog `stop()` reverted to the pre-L2 code | tests `-k l2`: 3 failed, 1 passed (the control). **KILLED**. The seat's witness runs show the behaviour: a 4.6 s silent wait, and watchdog calls after the detach | `mut_h_summary.txt`; `g4/run_h/witness_summary.txt` |
| Each mutant tree differs from this worktree in exactly one product file, its tests are byte-identical, and (e)/(f)/(h) (tc_signal) and (g) (core) were rebuilt after the edit | yes | `diff -rq`; the summaries |
| Voice runner, T1–T6 plus T1R and T4S | **PASS 249/249**, adapter sha 600f38b4, Project-Voice `git status` empty before and after | `voice_runner.log`, `p1t_all_lead.json`, `voice_git_status_*.txt` |
| Combined core plus tc_signal session | 37 failed, **3040 passed**, 47 skipped, 15 errors (721 s). The compare finds 52 = 52 baseline ids, 0 new and 0 no longer failing: **G1 PASS** | `combined_session.log`, `combined_compare.txt` |

## Gate of the seat report (David, 2026-10-13: "bugs don't get to be post-hoc accepted")

**First report: one item sent back (L2).** `DeviceWaitWatchdog::stop()` had two problems:
- it detached a thread that could still be inside `run()` when `join()` failed, which is a use-after-free of `mu_`,
  `cv_` and the caller's schedule;
- a failed stop-lock could miss the notify, giving a silent extra wait until the next diagnostic time.

The fix: `run()` publishes `exited_` with a release store as its last action; `stop()` re-notifies every 1 ms until it
reads `exited_` with an acquire load, then joins, and detaches only after that.
- I read the code (`dlpack_operand.cpp:1248-1325`): every member use, including the `unique_lock`'s unlock, ends inside
  `diagnose()`, before the store.
- Four new tests use an interposer. Mutant (h) is killed. The seat's witness shows the pre-L2 behaviour: a 4.6 s silent
  wait, and 2 calls after the detach.

**Ruling D2 / D12 (`diagnostic_events_gpu.py` as written): ACCEPTED.**
- I reproduced it independently. Under torch 2.11.0+cu130, `torch.cuda.ExternalStream(0).cuda_stream` is a
  non-blocking stream (`cudaStreamGetFlags` = 1) and not the legacy stream. `cudaStreamSynchronize(cudaStreamLegacy)`
  returned in 0.0000 s while the device sync took 0.079 s.
- So the probe's precondition, an exit wait on the legacy stream, does not hold on this host.
- The one-line copy that queues the work on `torch.cuda.default_stream()` is the faithful scenario: rc 1 on 6991939, rc 0
  here.
- The product was correctly not changed to print a wait that does not happen.

**Residuals.** None.

**"Not a defect" items.**
- **N1** (pybind11 / CPython cannot allocate the RuntimeError's message string): accepted. The failure is still raised,
  and the report was built before the slot was consumed. Citations are in `citations/` (CPython `_PyErr_SetString`,
  pybind11 `translate_exception`).
- **N2** (removing the documented exit protection): accepted, as in FIX6 N4 and FIX7 N2.
- **N3** (stderr's file descriptor closed): accepted. No channel is left; POSIX `write(2)` returns EBADF.

**User-visible changes U1–U8.** I accept each one:
- **U1** marker failures raised once (V8-01);
- **U2** the watchdog-diagnosed wait, with L2's stop protocol;
- **U3** "finished" only after proof;
- **U4** the runners act as last reader in every case;
- **U5** the allocation-free summary;
- **U6** a report the hook could not write is written by the runner, with acknowledgements so nothing is repeated;
- **U7** a raise builds its report before it consumes the slot. This is FIX5's record-before-consume rule;
- **U8** additive counters and raw functions.

**Deviations.**
- D5 (a watchdog thread, not `cudaStreamQuery` polling) is justified: the reviewers' host traces stub only eleven CUDA
  functions, and the watchdog uses none.
- D3 (the round-4 regeneration copy gained the new helper's text) is tooling only.
- L2-D1 (an adversary stronger than the real race) is welcome.
- L2-D4 (the inherited round-8 probes not re-run after L2) is why I ran them all here. That run found the
  `fix7_cpu_census` harness race above, which is not a product finding.

## Verdict

FIX8, with the L2 send-back, is verified. Review round 9 may start on this commit.
