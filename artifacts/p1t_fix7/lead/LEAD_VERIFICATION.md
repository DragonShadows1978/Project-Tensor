# P1T-FIX7 lead verification (Project-Tensor lead, 2026-10-13, about 14:25 EDT)

All runs were fresh processes. Every GPU run was under `flock -w 900 /tmp/forge-gpu.lock`, on native driver 595.99.02.

**Arms:**
- **Before:** the 429446d build in `wt_p1trev7`: core `e7d4a3a3…`, `_tc_signal` `ef44b9c4…`, which are the FIX6 hashes.
  Its `git status` was clean. This is the tree in which the lead confirmed round 7's findings.
- **After:** this worktree's final build, after the L1 send-back. Both hashes match the seat's amendment, and both
  binaries are newer than every source file:
  - core `750112413b78450937ba126e56ab700e3f9e16d1092f88af239b13fee48f9245`;
  - `_tc_signal` `ca5770c152f6946921ce8c06681a2ecad6beaa81232b948cceac2a7a99605c37`.

**Harnesses:**
- I re-ran the seat's runners with my own arm names and absolute output directories: `run_rev7_probes.sh`, and
  `runners/run_rev{6,5}_probes.sh`, `run_rounds123_probes.sh` and `g4/run_mutant.sh`.
- The before arm used a lead copy of the round-7 runner, trimmed to the G1 probes: `lead_rev7_g1_probes.sh`.
- The whole after-arm chain is `lead_chain.sh`, with its log in `lead_chain.log`.

## Checks

| Check | Result | Receipt |
|---|---|---|
| Scope | Every path changed against 8035b3b is in the order's Exercises, plus the FIX7 order itself (commit 420e821), `docs/P1T_FIX7_*`, `artifacts/p1t_fix7/**` and `logs/**`. In `tensor_cuda/src/kernels.cu` the hunks touch only P1-T code: `note_core_unraisable` and `drain_device_legacy_stream`, `cuda_storage_device_mismatch` and `fill_`, none of which exists in 9df5bef. The `bindings.cpp` hunks touch only the P1-T exit entry and runner, `CoreCallArgs`, `raw_one_real_arg`, `dlpack_raise_unraisable` and P1-T stats keys and registrations | `git diff 8035b3b` |
| Probe files | `artifacts/p1t_rev*` are byte-identical to HEAD, with no untracked additions | — |
| New core `__global__` | 0 | — |
| Round-7 G1, before (429446d) | `knob_exit_cpu` no GPU **rc 134**, GPU **rc 134**; `core_knob_exit_cpu` **rc 134**; `core_knob_exit_gpu` **rc 1** (`FATAL: exception not rethrown`); `query_diagnostic_cpu` **rc 1**, `query_diagnostic_gpu` **rc 1** ("the legacy stream of device 0 could not be synchronized"). Every `--control` rc 0. The core slices equal the reviewer's pins (`all_identical= True`) | `rev7_before_summary.txt`, `rev7_before/` |
| Round-7 G1, after | all six **rc 0**. The knob lines say the default delay was used; V7-02 now reads "the completion event of a pending entry on device 0 could not be queried (cudaEventQuery: …)". Every `--control` rc 0 | `rev7_after_summary.txt`, `rev7_after/` |
| Round-7 others, after | `run_cpu_checks` 14/14 rc 0; `argument_census_cpu` PASS 1067 checks, 29 bindings, 0 conversion calls; `gpu_regressions` fix56 **157 passed, zero skips**, semantics **504 passed, zero skips**, multidevice exit 3 (one GPU) | same |
| Round 6, after | every check rc 0, including `sync_lock_inversion` no GPU / GPU (`rescue=False`), `sync_late_release`, `core_cast_exit_gpu` bool / index and controls, and the regenerated `core_host casts_exit` | `rev6_after_summary.txt` |
| Round 5, after | rc 0 for: `exit_gate full` (no GPU / GPU, with controls); `exit_gate gpu` and its control; `query_error_cpu` / `_single_gpu`; `fill_grid_single_gpu`; `device_error_sweep_single_gpu`; `sync_existing_error`; `addon_lifetime` 12/12; `core_edges`; `core_exit_overrun` during (3.1 s) and bounded (7.6 s); `numerics_and_streams`; `release_failures` (66 passed); `core_import_query_error`. `exit_gate trace` as written gives rc 1, and so does its control, exactly as in FIX6 (D7): its child models the hook as `synchronize()` alone. Its exit-entry copy gives rc 0. The as-written `handle_insert_oom_cpu` and `fill_grid_cpu` embed 1475fae code (rc 1 on every arm since FIX5); their regenerated forms give rc 0. The two-GPU probes exit 3 | `rev5_after_summary.txt` |
| Rounds 1–3, after | `addon_lifetime` runs 1–3 rc 0, each 12/12 correct, 0 incorrect; `core_edges`, `core_shutdown` normal and finalizing, `numerics_and_streams` and `host_checks` all rc 0; `core_exit_overrun` during and bounded rc 0; `core_device_drain` exit 3 | `rounds123_after_summary.txt` |
| Mutant (a), add-on hook reverted to 429446d's | tests `-k v701`: 24 failed, 14 passed; `knob_exit_cpu` no GPU / GPU rc 134 / 134. **KILLED** | `mut_a_summary.txt` |
| Mutant (b), core hook reverted to 429446d's | tests: 16 failed, 13 passed; `core_knob_exit_cpu` rc 134, `core_knob_exit_gpu` rc 1. **KILLED** | `mut_b_summary.txt` |
| Mutant (c), failure-kind tag dropped | tests: 13 failed, 16 passed; `query_diagnostic_cpu` / `_gpu` rc 1 / 1. **KILLED** | `mut_c_summary.txt` |
| Mutant (d), L1 diagnostic removed from `lock_retrying` | the 2 L1 core tests (`[close]` and `[relock_after_diagnostic]`) FAIL, while the 2 i19 retry tests pass (the retry itself is intact). **KILLED** | `mut_d_summary.txt` |
| Each mutant tree (`/tmp/p1t_fix7_mutL1_*`) differs from this worktree in exactly one product file, its test files are byte-identical, and (c) and (d) were rebuilt after the edit | yes | `diff -rq`; the summaries' first lines |
| Voice runner, T1–T6 plus T1R and T4S | **PASS 249/249** (T1 11, T2 144, T3 28, T4 12, T5 28, T6 13, T1R 1, T4S 12), 28.2 s; adapter sha 600f38b4, unchanged; Project-Voice `git status` clean before and after | `voice_runner.log`, `p1t_all_lead.json`, `voice_git_status_{before,after}.txt` |
| Combined core plus tc_signal session | 37 failed, 3001 passed, 47 skipped, 15 errors (720.8 s). The compare finds 52 = 52 baseline ids, 0 new and 0 no longer failing: **G1 PASS** | `combined_session.log`, `combined_compare.txt` |

## Gate of the seat report (David, 2026-10-13: "bugs don't get to be post-hoc accepted")

**First report: one item sent back (L1).**
- Its i19 fix (`lock_retrying`) retried a failing lifecycle lock forever without writing anything to stderr. That
  breaks the FIX3 V3-01 contract that every exit wait is diagnosed.
- The seat fixed it and swept every exit wait or retry loop in both packages (addendum rows a1–a12). The sweep found
  and fixed two more silent waits in the add-on: the polling fallback (a2) and the device syncs of exit (a3). Each has a
  test, and mutant (d) checks a5.

**Residuals.** None. The report lists no residual defect.

**"Not a defect" items.**
- **N1** (a hook that returns without closing is not rescued): accepted. Every path that runs a hook statement reaches
  the close (sweep (i)). A backstop that always closed would hide a hook defect; mutants (a) and (b) are exactly that
  defect, and the round-7 probes kill both.
- **N2** (the application removes the atexit registration): accepted, as FIX6 N4 was by round 7. The Python 3.12
  `atexit` documentation is in `citations/python312_atexit.txt`.
- **N3** (knob values that only Python code can read now select the default): accepted. The knob is a diagnostic
  delay, and reading such a value means running user code before the close, which is the V7-01 class itself.
- **N4** (CPython prints "Exception ignored in atexit callback" after a Ctrl-C): accepted. That is CPython's
  documented handling, and the runner deliberately re-raises the hook's own exception.

**Addendum row a10** (core drains of imported Storages at finalization are not diagnosed). The seat left this for the
lead to rule on. **Ruling: not a defect.**
- `drain_device_legacy_stream` is the release path of every imported-Storage drop, exactly the same during the run. The
  exit path neither adds it nor calls it; CPython's finalization drops whatever objects remain.
- A kernel that never finishes blocks every `cudaStreamSynchronize` (CUDA Runtime API: "Blocks until stream has
  completed all operations"), in any library. So this is not a wait that P1-T's exit adds.
- The add-on's a3 differs: the exit hook itself calls `synchronize()`.

**Checked in the L1 code, and not a defect:**
- `exit_diag_of_hook` reads `_RELEASE_EXIT_DIAG_S` with `PyDict_GetItemString` on the hook's globals. A dict lookup can
  call a colliding non-str key's `__eq__` (errors suppressed). That needs a crafted object planted in the module's
  `__dict__`, and the hook's own `LOAD_GLOBAL` of the same name has the same property. So the native read adds nothing
  new.
- `wait_device_work_diagnosed` creates and records its own event under the existing device guard and clears only errors
  of its own calls. The stream sync that follows still reports the device's state. The window is RAII-scoped to the
  runner, so no ordinary `synchronize()` changes.

**User-visible changes U1–U10.** I accept each one:
- **U1–U3** (the exit entries close for any value; no Python conversion; atexit registers the native runner) follow from
  the order's Required fix 1 and the language rule that a Python function cannot guard its own first instruction.
- **U4–U6** keep the exit guarantees on every path (Python's try / finally; V6-03 has a reader at exit).
- **U7** (messages name the failed call and its device) is the order's (j) rule. Exception types are unchanged.
- **U8** (new stats keys) is additive.
- **U9 / U10** (the lock retry, and every exit wait diagnosed) are the i19 fix plus L1.

**Deviations D1–D14 and L1-D1–L1-D8.** I checked each one and none is a scope or rule violation:
- D1 consists of path-only copies, with the diffs kept.
- D2 is the before tree. I used my own before arm (`wt_p1trev7`) instead, and it gives the same verdicts.
- D5 consists of test corrections, each asserting the new stricter rule.
- D6 fixes a defect in a FIX6 test (capsule name lifetime; CPython C-API "Capsules").
- L1-D1 (a3 fixed beyond the list I sent) is welcome.

## Verdict

FIX7, with the L1 send-back, is verified. Review round 8 may start on this commit.
