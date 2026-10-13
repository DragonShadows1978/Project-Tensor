# P1T-FIX3 report: round-3 findings V3-01 (exit quiescence) and V3-02 (aggregate byte accounting) closed

**Verdict: G1 PASS, G2 PASS, G3 PASS, G4 PASS.**
- **V3-01.** The exit hook now waits, with no bound, until every admitted release has finished. It logs at a bounded
  rate while it waits.
  - The reviewer's `lifecycle_timeout.py` (short, default) and `core_exit_overrun.py` (bounded) abort on 5f9f955 (rc
    134 = SIGABRT under `timeout`, "FATAL: exception not rethrown"). They exit 0 on this build.
  - New normal-exit regressions (no `os._exit`) pass: deleters of 0.1 s, 5.5 s and 12 s with the default 5 s knob,
    idle, and the hook called inside the thread's own release.
- **V3-02.** Each call's aggregate operand bytes are checked before the first launch (ValueError). The pending-list
  total saturates without throwing.
  - The reviewer's `pending_bytes` queue mode is clean under UBSan here (5f9f955: overflow at
    `dlpack_operand.cpp:602`).
  - The adapted entry driver refuses cleanly here (5f9f955: overflow at `dlpack_operand.cpp:731`).
  - New regressions pass.
- **G2 holds:**
  - every round-3 probe passes, except the two two-GPU probes, which are INCONCLUSIVE (exit 3, one GPU);
  - round-1 `addon_lifetime` passes 3/3 runs with 0 incorrect, and the other round-1 and round-2 probes pass;
  - Voice T1-T6 gives PASS 236/236;
  - the combined session fails exactly the 52 baseline ids.
- **G3:** both mutants are killed. **G4:** scope clean, probe files byte-identical, no new `__global__`.
- **One honest note.** The reviewer's unmodified `pending_bytes` *entry* mode still reports UBSan on this build. It
  overflows inside the probe's own copy of 5f9f955's statement (`pending_bytes.cpp:28`), not in product code, so it
  cannot pass on any build. As the reviewer instructed, I adapted that driver to call the real retainer (D4).

Seat: Opus 5.5 (high). Order: `orders/P1T_FIX3_EXIT_QUIESCENCE.md` (021abbb). Ledger: `docs/P1T_FIX3_LEDGER.md`.
Receipts: `artifacts/p1t_fix3/`, `logs/p1t_fix3_*`.
- "Before" binaries are wt_fix2 (5f9f955, read-only): core `7729cb7a…b3f5`, tc_signal `afd1f01d…b9f6`.
- After binaries (`artifacts/p1t_fix3/build_sha256.txt`):
  - core `_tensor_cuda.cpython-312-x86_64-linux-gnu.so`: `35b8261680760851e5d150f851d58ad8f0fbec617deb225dd6d7565010487a23`
  - `_tc_signal.cpython-312-x86_64-linux-gnu.so`: `222861219e29e8486d7a6f92ee874f5b4342c6f26d11a3440bf9bb99787e535a`

## Per-finding table

| Finding | Fix (file:line) | Proving tests (new) | Probe before (5f9f955) | Probe after (this build) |
|---|---|---|---|---|
| **V3-01** (MAJOR): the exit wait returned to finalization while an admitted release was running | `tensor_cuda/src/bindings.cpp`:<br>• `:557` design comment<br>• `:625` `thread_local t_admitted_depth`; `:672` / `:691` its increment and decrement<br>• `:711` `dlpack_begin_shutdown`: `:726` admission closed for good; `:729` `while (!cv.wait_until(lk, next, in_flight <= own))`, with no return before quiescence; `:735` stderr diagnostic (count running, seconds waited), written with the mutex dropped; `:744` doubling schedule<br>• `:1786` binding docstring (consequence)<br>`tensor_cuda/tensor_cuda/__init__.py`:<br>• `:185` knob, now the first-diagnostic delay<br>• `:188` `_dlpack_shutdown_hook` docstring (consequence; Ctrl-C)<br>• `:206-209`: an unusable knob gives NaN, which diagnoses at once and still waits | `tensor_cuda/tests/test_dlpack.py`:<br>• `:1237` `test_fix3_v301_admitted_release_finishes_before_finalization_ordinary_exit[0.1, 5.5, 12.0]`: default knob 5.0, ordinary exit, rc 0, deleter completed, `FINAL finalizing=True`, diagnostics at [] / [5.0] / [5.0, 10.0]<br>• `:1260` idle<br>• `:1271` hook inside its own release (no self-wait)<br>• `:1232` the script has no `os._exit`<br>• N5 "nested" (never admitted) and "after" (after closing) kept | `lifecycle_timeout.py`:<br>• `short` **rc 134** `HOOK {... 'still_in_flight': 1, 'timed_out': True, 'waited_us': 100064} / FINALIZING True / FATAL: exception not rethrown`<br>• `default` **rc 134** (`waited_us 5000065`)<br>`core_exit_overrun.py bounded`: **rc 134** `tensor_cuda: 1 DLPack release(s) still running after 5.0 s at exit; ... / FATAL: exception not rethrown` | `lifecycle_timeout.py`:<br>• `short` **rc 0**: diagnostics at 0.1 / 0.2 / 0.4 s, `DELETER finished`, `HOOK {... 'still_in_flight': 0, 'timed_out': True, 'waited_us': 595122, 'diagnostics': 3}`, `FINALIZING True`<br>• `default` **rc 0** (`waited_us 5495163`, 1 diagnostic)<br>`core_exit_overrun.py bounded`: **rc 0**, wall 7.5 s, `still running after 5.0 s` / `finished after 5.5 s; interpreter exit continues` / `FINAL finalizing=True` |
| **V3-02** (MINOR): unchecked aggregate / queue byte additions | `tc_signal/src/dlpack_operand.cpp`:<br>• `:760` `aggregate_operand_bytes` (`ck_add`, ValueError)<br>• `:777` called FIRST in the `RetainUntilComplete` constructor, before any allocation, event or launch, at all 9 retainer sites (every entry point)<br>• `:789` the destructor uses the checked sum (no addition)<br>• `:623` `note_push` saturating, noexcept<br>• `:638` `note_pop` (sticky saturated state, cleared when the list is empty)<br>• `:654` `pending_over_bound`, used at `:833`<br>• `:444` state fields<br>`dlpack_operand.h:140,159,167`; `README.md` | `tc_signal/tests/test_fix3_findings.py`:<br>• `:119` `test_fix3_v302_aggregate_overflow_refused_before_any_launch[rint, divide] x [gpu, nogpu]`: real entry points, ValueError naming the aggregate overflow, every deleter once at once, nothing queued<br>• `:162` `test_fix3_v302_native_byte_accounting_overflow_safe[accounting, dtor]`: `fix3_native_harness.cpp` under UBSan; 17 accounting cases (saturation, pop/empty, exact INT64_MAX, boundary of both bounds, aggregate) and 11 real-retainer cases with CUDA events and no launch (saturated push drains, constructor refusal, exactly-at-bound stays) | `pending_bytes queue` **rc 1** `dlpack_operand.cpp:602:12: runtime error: signed integer overflow: 4 + 9223372036854775804`<br>adapted `entry` (GPU) **rc 1** `dlpack_operand.cpp:731:13: runtime error: signed integer overflow: 9223372036854775804 + 9223372036854775804` | `pending_bytes queue` **rc 0** `9223372036854775807`, no UBSan line<br>adapted `entry` **rc 0** `SAFE REFUSAL: tc_signal: the aggregate fp32 byte count of this call's operands (their sum) overflows 64-bit arithmetic; refused (no wrapped size is ever used)` (both conditions) |

### V3-01 design, GIL and consequence

- **Admission** is unchanged from FIX2 (mutex, `closing`, `in_flight`, F04 as the second line), and it stays closed
  once the hook has run.
- **No early return.** The hook returns only when `in_flight` minus the caller's own admitted releases is 0.
  `still_in_flight` at return is therefore always 0.
- **Diagnostics.**
  - The first line comes after `_DLPACK_SHUTDOWN_WAIT_S` (5.0 s, unchanged). The next comes whenever the time
    waited has doubled (5, 10, 20, 40 s …).
  - So a deleter that hangs for an hour logs about 10 lines. The rate is bounded with no new literal; the minimum step
    is one clock tick, so a zero knob cannot spin.
  - After an overrun the Python hook writes a final "finished after X s; interpreter exit continues" line.
- **GIL: yes, the wait drops and re-takes it correctly.**
  - The whole wait, diagnostics included, runs inside `py::gil_scoped_release`, and the condition-variable wait drops
    the mutex. So an admitted deleter can take the GIL (`PyGILState_Ensure`), run Python, re-take the GIL after
    `time.sleep`, release it, and then take the mutex to leave.
  - Each diagnostic is written (C `stderr`, no Python) with the mutex dropped.
  - The mutex is released before `gil_scoped_release` re-takes the GIL. FIX2's lock order holds: no thread waits
    for the GIL while holding the mutex.
  - Evidence: in the 5.5 s and 12 s tests the deleter is a ctypes Python callback. It sleeps with the GIL released and
    must re-take the GIL to return and set `done = True`. It completed while the hook waited (`deleter_done [True]`,
    `released 1`), and then CPython finalized (`FINAL finalizing=True`, rc 0).
- **Self-exclusion** (beyond the literal order; D9).
  - A thread that runs the hook from inside its own admitted release does not wait for that release (thread-local
    depth). That release cannot finish while its thread waits, and it necessarily finishes before that thread can go
    on to finalize.
  - Without this, the unbounded wait would self-deadlock. 5f9f955 waited its whole 5 s bound there: the test
    measured `hook_s` 5.0001 on 5f9f955 and under 1 s here.
- **Consequence (documented in the docstrings).**
  - A producer deleter that never returns makes interpreter exit wait forever. That is the producer's DLPack contract
    violation (a deleter must return), and waiting is the honest outcome. FIX2's bound turned the same situation into
    an abort (rc -6).
  - Ctrl-C does not interrupt the wait, because no bytecode runs on the main thread. SIGTERM and SIGKILL still end the
    process.

### V3-02: why saturate rather than drain-before-add

- `note_push` and `note_pop` are pure accounting, called inside the noexcept destructor and the drain loops, so they
  may neither throw nor drain.
- Draining runs producer deleters (Python, possibly re-entering tc_signal), and it cannot run in the middle of a push.
- A drain cannot guarantee headroom anyway: entries pushed by a re-entrant call during the drain stay queued. So a
  non-throwing, non-overflowing rule is needed in any case.
- Saturation is that rule:
  - checked add; on overflow the total is pinned at INT64_MAX and marked saturated (counted);
  - pops keep it pinned until the list is empty, where the true total is exactly 0;
  - a pinned total over-reports and never under-reports.
- **Bound comparison at the boundary:**
  - exact below INT64_MAX: exactly `kPendingMaxBytes` stays queued, and 4 more bytes (fp32 granularity) drain, in
    both the accounting unit and the real destructor;
  - conservative above it: any overflowing push drains, as exact arithmetic would, because any total above INT64_MAX
    is far above 256 MiB;
  - with the pre-launch check, each entry's bytes is an exact int64, so only the queue total can saturate. That takes
    more than 8 EiB of claimed operands, which is metadata-only.

## Probe before/after table (every probe in `artifacts/p1t_rev3/probes/README.md`)

Logs are in `artifacts/p1t_fix3/probe_runs/rev3_*`, with summaries in `rev3_summary_{before,after}.txt`. Conditions:
- `nogpu` = `CUDA_VISIBLE_DEVICES=` (the reviewer's sandbox);
- `gpu` = the device visible.

All runs were fresh processes under the GPU lock.

| Probe | How it ran | Before (5f9f955) | After (this build) | Result |
|---|---|---|---|---|
| `host_checks.py` (15 cases + 5 native) | path-only copy (D2), nogpu / gpu | nogpu: 20/20 rc 0. gpu: 19 rc 0, `oom_sync` -6 after `PASS: nonallocating completion path` | identical per condition (case, rc, text) | PASS, unchanged. The gpu `oom_sync` is FIX2's D4, on both arms (the probe's own pybind11 teardown) |
| `core_cpu.cpp` (normal / prefix / suffix) | compiled unmodified with UBSan (its slices are verbatim in the after sources), via host_checks | 3/3 rc 0, `SAFE REFUSAL` prefix/suffix, no UBSan | same | PASS |
| `cpu_native.cpp` (oom_sync, throw_deleter) | compiled unmodified against the arm's `dlpack_operand.cpp`, via host_checks | nogpu 0 / 0; gpu -6 (D4) / 0 | same | PASS (D4 unchanged) |
| `lifecycle_timeout.py control` | unmodified; module from the committed slice (before) or the regenerated slice (after, D3) | rc 0 | rc 0 | PASS |
| `lifecycle_timeout.py short` (0.6 s deleter, 0.1 s) | same | **rc 134**, FATAL: exception not rethrown | **rc 0** | **FIXED** |
| `lifecycle_timeout.py default` (5.5 s deleter, 5 s) | same | **rc 134**, FATAL: exception not rethrown | **rc 0** | **FIXED** |
| `pending_bytes.cpp queue` | compiled unmodified with UBSan | **rc 1** `dlpack_operand.cpp:602:12` overflow | **rc 0** (`9223372036854775807`) | **FIXED** |
| `pending_bytes.cpp` entry, unmodified | same | rc 1 `pending_bytes.cpp:28:14` | rc 1 `pending_bytes.cpp:28:14` | **N/A by construction** (D4): the overflowing statement is the probe's own copy |
| `pending_bytes` entry, adapted driver (D4) | real retainer on the same operands | gpu **rc 1** `dlpack_operand.cpp:731:13` overflow; nogpu rc 0 "cannot create the completion event" (not a witness) | **rc 0** `SAFE REFUSAL ... aggregate ... overflows` (both conditions) | **FIXED** |
| `operand_census.py` | path-only copy (D2), nogpu / gpu | PASS, 17 cases, 7 once-only | same | PASS |
| `real_metadata_plans.py` | path-only copy (D2), nogpu / gpu | PASS, 129 cases | same | PASS |
| `core_exit_overrun.py during` | unmodified, GPU, timeout 20 | rc 0 | rc 0 | PASS |
| `core_exit_overrun.py bounded` | unmodified, GPU, timeout 20 | **rc 134**, FATAL: exception not rethrown | **rc 0** (wall 7.5 s) | **FIXED** |
| `addon_lifetime.py` | unmodified, GPU | rc 0, 12/12 correct, 0 incorrect | rc 0, 12/12 correct, 0 incorrect | PASS |
| `core_device_drain.py` | unmodified, GPU | exit 3 | exit 3 | INCONCLUSIVE (needs 2 GPUs) |
| `core_edges.py` | unmodified, GPU | rc 0 | rc 0 | PASS |
| `numerics_and_streams.py` | unmodified, GPU | rc 0, PASS | rc 0, PASS | PASS |
| `cuda_errors_multidevice.py` | unmodified, GPU | exit 3 | exit 3 | INCONCLUSIVE (needs 2 GPUs) |
| `release_failures.py` | after: original; before: root copy (D2, wt_fix2's own tests on its build) | rc 0, 66 passed, no skip | rc 0, 66 passed, no skip | PASS |
| `native_harness ck_census` (README CPU command) | `fix2_native_harness.cpp` against the arm's source, UBSan, nogpu / gpu | 18 cases, 0 failures, no UBSan | same | PASS |

Support files: `build_cpu.sh` is mirrored by `build_rev3_probes.sh` (same flags, into `/tmp/p1t_fix3_probes/<arm>`).
`lifecycle_slice.cpp` is regenerated for the after build (D3). `release_worker.cpp` is not needed by round 3; round
1 uses the existing `/tmp/p1t_rev_native` build read-only (sha `45875b4f…`).

**Round 1, after build** (`probe_runs/round12_summary_after.txt`):
- `addon_lifetime` runs 1-3 exit 0, each 0/12 incorrect, 12/12 correct, 0/12 reused;
- `core_edges` 0;
- `core_shutdown --normal` 0 ("PASS: ordinary foreign-thread release");
- `core_shutdown` finalizing 0 ("finalizing: True … PASS: shutdown foreign-thread release completed");
- `numerics_and_streams` 0;
- `host_checks` identical to FIX2's accepted receipt: action, rc and final line all True
  (`round1_host_checks_vs_fix2.txt`);
- `core_device_drain` 3 (INCONCLUSIVE).

**Round 2, after build:** `host_checks` through FIX2's committed copier, with `core_cpu` regenerated (selfcheck
identical=True). nogpu: 20/20 rc 0. gpu: as FIX2 (`oom_sync` -6, D4).

## Gate table

| Gate | Criterion (registered in the ledger at 00:54, before any change) | Result | Receipt |
|---|---|---|---|
| G1 | `lifecycle_timeout.py` and `core_exit_overrun.py` pass here and fail on 5f9f955 | **PASS**: short/default 134 → 0 (both conditions); bounded 134 → 0; control and during 0 → 0 | `probe_runs/rev3_lifecycle_*`, `rev3_core_exit_overrun_*` |
| G1 | `pending_bytes.cpp` clean under UBSan here, reports on 5f9f955 | **PASS** for queue (602 overflow → clean) and the adapted entry (731 overflow → clean refusal). The unmodified entry mode reports on both by construction (D4) | `probe_runs/rev3_pending_bytes_*` |
| G1 | the new regression of each finding passes | **PASS**: V3-01 6/6 core; V3-02 6/6 tc_signal (and inside the combined session) | `logs/p1t_fix3_g2_combined_session.log` |
| G2 | every round-3 probe passes, or is INCONCLUSIVE only where it needs 2 GPUs | **PASS**: core_device_drain and cuda_errors_multidevice exit 3; everything else passes | `probe_runs/rev3_summary_after.txt` |
| G2 | round-1 addon_lifetime 3/3 runs with 0 incorrect; the other round-1 probes pass | **PASS** | `probe_runs/round12_summary_after.txt` |
| G2 | Voice runner T1-T6 PASS 236/236 | **PASS**: exit 0, 236/236 | `logs/p1t_fix3_g2_voice_T1toT6.log`, `artifacts/p1t_fix3/p1t_T1toT6.json` |
| G2 | combined session = the 52 baseline ids | **PASS**: 52 = 52, 0 new, 0 no longer failing | `logs/p1t_fix3_g2_combined_session.log`, `logs/p1t_fix3_g2_compare.txt` |
| G3 | mutant (a), the bounded return restored, fails its new test | **PASS** (killed) | `artifacts/p1t_fix3/g3/mut_a_*` |
| G3 | mutant (b), the aggregate pre-launch check removed, fails its new test | **PASS** (killed) | `artifacts/p1t_fix3/g3/mut_b_*` |
| G4 | diff only in the Exercises paths and allowed receipts; probe files byte-identical; no new core `__global__` | **PASS** (r2; r1 printed FAIL from two checker defects, D13) | `artifacts/p1t_fix3/g4_scope.txt`, `g4_scope_r1_checker_defects.txt` |

**G2 verdict line:** `G2 PASS: round-3 probes all PASS (core_device_drain, cuda_errors_multidevice INCONCLUSIVE: one GPU); round-1 addon_lifetime 3/3 runs 0/12 incorrect each, core_edges, core_shutdown normal+finalizing, numerics_and_streams, host_checks PASS; round-2 host_checks as FIX2; Voice runner T1-T6 exit 0 PASS 236/236; combined core+tc_signal session FAILED/ERROR = 52 baseline ids (0 new, 0 no longer failing).`

**Voice's verdict line** (`artifacts/p1t_fix3/p1t_T1toT6.json` summary):
`verdict PASS, exit_code 0, counts {"PASS": 236, "FAIL": 0, "SKIP": 0} (T1 11, T2 144, T3 28, T4 12, T5 28, T6 13), f1 all_f1_ok true, max_N 2.9320924600942896e-06, min_margin_vs_gate 6.821067299957125 (T4/shared__LSTM/max_1)`.
These figures are identical to FIX2's.

**Compare output** (`logs/p1t_fix3_g2_compare.txt`):
```
G1 compare (APA-MF2.2): baseline = /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2/logs/apa_mf1_g1_baseline_unchanged.log; now = logs/p1t_fix3_g2_combined_session.log
summary: 37 failed, 2689 passed, 47 skipped, 2 warnings, 15 errors in 261.39s (0:04:21)
baseline ids 52 now ids 52
baseline ids that no longer fail: 0 []
new failing ids: 0 []
new MF2.2 test files with any FAILED/ERROR: 0
G1 PASS
```
The 37 failures and 15 errors are all in the baseline APA test files. Passed is 2689 against FIX2's 2678: core added
6 tests and lost 1 removed test (D1), and tc_signal added 6.

## G3 mutant table

Trees are `/tmp/p1t_fix3_mut_<m>/`, each with one asserted edit (`artifacts/p1t_fix3/g3/make_mutants.py`, diffs in
`g3/mut_<m>.diff`). The runner is `g3/run_mutant.sh`. Each test file is `cmp`-identical to the worktree's.

| Mutant | Edit | .so sha256 | New tests on the mutant | What the tests observe | Verdict |
|---|---|---|---|---|---|
| (a) V3-01: the bounded return restored | `bindings.cpp` wait loop: `break;` after the first diagnostic, so the wait returns at `_DLPACK_SHUTDOWN_WAIT_S` (5 s) with the release still running (FIX2's semantics) | core `8b5e79d2c4f21e44a846d18ce13e599ba978f7427ea133ef882e0f7b4f5eb592` | `-k "fix3 or n5"`: **2 failed**, 7 passed | overrun 5.5 and 12.0: after the hook `deleter_done [False, False]` (it returned before the deleter finished). The reviewer's `core_exit_overrun.py bounded` on the same mutant gives **rc 134** `FATAL: exception not rethrown` (`g3/mut_a_core_exit_overrun_bounded.log`). The 0.1 s, idle, self and FIX2 N5 tests pass, as they should | KILLED |
| (b) V3-02: the aggregate pre-launch check removed | `aggregate_operand_bytes`: `ck_add` replaced by a silent saturating add (no refusal; the call goes on to its launch) | tc_signal `d636ff01b913a5eec460f9bc1f3ee851e957dde6ec8203d042ce1a53e340a50f` | **6 failed** (all) | gpu: the call launched onto the fake addresses (`cudaErrorIllegalAddress` inside the test's subprocess, rc 1); nogpu: `RuntimeError` (event creation) instead of `ValueError`; native accounting: the aggregate cases not refused; native dtor: D2 not refused | KILLED |

GPU sanity after mutant (b)'s deliberate fault: `rint` correct, `nvidia-smi` P8, 218 MiB.

**Before-witness of the new tests on 5f9f955** (`artifacts/p1t_fix3/before_witness/`):
- core: 5 failed, 1 passed.
  - 5.5 and 12.0 fail on `deleter_done [False]`: the defect.
  - self fails on `hook_s 5.0001`: FIX2 self-waited its whole bound.
  - 0.1 and idle fail only on `KeyError: 'shutdown_diagnostics'` (a new stats key). They are not defect witnesses
    (D15).
- tc_signal prelaunch, nogpu: 2 failed, `RuntimeError` instead of `ValueError`.

## Operand census → tests and probes

| Census item | Covered by |
|---|---|
| foreign-thread deleters of 0.1 s, 5.5 s and 12 s across the exit wait, ordinary exit, no `os._exit` | `test_fix3_v301_..._ordinary_exit[0.1, 5.5, 12.0]` (default knob 5.0); reviewer `core_exit_overrun.py bounded` (5.5 s) and `lifecycle_timeout.py` (0.6 s and 5.5 s) |
| "after closing" | FIX2 `test_fix2_n5_release_after_the_exit_hook_leaks_without_gil` (`leaked_closing 1`, deleter never called) |
| "never admitted" | FIX2 `test_fix2_n5_release_starting_during_the_wait_leaks_without_gil` (refused while the hook waits for A, `leaked_closing 1`); FIX1 F04 finalizing test and round-1 `core_shutdown` finalizing (`leaked_interpreter_shutdown`); plus new idle (nothing admitted: no wait) |
| two disjoint `2**61 − 1` fp32 operands (metadata only) | `test_fix3_v302_aggregate_..._before_any_launch[rint-*]` (real entry point); native accounting and dtor D2; reviewer `pending_bytes` (adapted entry) |
| queue totals near INT64_MAX | native accounting (reviewer queue case 4 + 9223372036854775804; an exact INT64_MAX sum; INT64_MAX + 1 byte); native dtor D1 (real destructor: 8 + (2^63 − 8)); reviewer `pending_bytes queue` |
| the bound comparison at the boundary | native accounting (exactly `kPendingMaxBytes` within, +4 over, back within after pop; 1024 entries within, 1025 over); native dtor D3 (real destructor: exactly at the bound stays queued, +4 drains) |

## Prior art

- **V3-01:**
  - CPython 3.12.3 `Py_FinalizeEx` ordering: atexit callbacks run before the runtime is marked finalizing (verified in
    FIX2).
  - `threading._shutdown` / `wait_for_thread_shutdown`, which joins non-daemon threads with no bound before
    finalizing. This is the same "finalization waits for work already started" rule, one level up.
  - RCU-style grace periods: `synchronize_rcu` waits, with no bound, for every reader that began before the grace
    period, while new readers do not delay it (McKenney & Slingwine, 1998). Also the Linux kernel's `percpu_ref`
    (kill, then wait for zero).
  - Repeated stall reports while a wait goes on: the kernel's hung-task and RCU CPU-stall warnings.
  - Taken: the unbounded quiescence wait with admission closed, and periodic stall reports.
  - Ours: the placement at the CPython atexit boundary of the DLPack import release, the doubling report schedule and
    the same-thread exclusion.
- **V3-02:**
  - Checked addition: GCC 5 (2015) `__builtin_add_overflow`, C23 `ckd_add`, SEI CERT INT32-C.
  - Saturating arithmetic: DSP/SIMD saturated adds, e.g. MMX/SSE `PADDS`, ARM `QADD`.
  - UBSan (GCC/Clang, 2013-).
  - Taken: checked add, refused before the launch; saturation on the release path.
  - Ours: the per-call aggregate check placed in the retainer constructor, and the sticky saturated state that keeps
    the bound comparison conservative until the list empties.
- **Tests and probe tooling:**
  - The reviewer's probes: their scenarios are reused, and the reviewer's own slice generator and FIX2's
    `regen_core_cpu_probe.py` are the method behind `regen_lifecycle_slice.py`.
  - DeMillo, Lipton & Sayward (1978) mutation testing.
- **Unverified in this seat** (lead to check): "synchronize_rcu grace period readers McKenney 1998",
  "rcu_cpu_stall_timeout repeat interval", "hung_task_warnings", "PADDSW saturating add MMX 1997",
  "percpu_ref_kill_and_confirm", "CPython wait_for_thread_shutdown 3.12".
- These annotations are also at the code sites (`bindings.cpp:557` block, `__init__.py:174-185`,
  `dlpack_operand.cpp:603` block, `:760`, the harness and test headers) and in the ledger.

## Deviations (with reasons)

- **D1. FIX2 test removed.** `test_fix2_n5_the_wait_is_bounded_and_reported` is gone, along with its script's
  "bounded" mode and `os._exit(0)`.
  - It asserted the defect: the hook returned at 0.3 s with `still_in_flight 1`, and the test then ended the process
    with `os._exit` so that finalization never ran (the review cites `test_dlpack.py:1089-1090,1192-1200`).
  - It is replaced by stricter ordinary-exit tests. Nothing was loosened. N5's "during", "nested" and "after" tests
    are unchanged and pass.
- **D2. Path-only probe copies** (`make_probe_copies.py`: every substitution is asserted once and has a diff; the
  originals are byte-identical):
  - `host_checks.py` hard-codes the reviewer's worktree as ROOT, writes its JSON there, and runs the reviewer's
    `/tmp/p1t_rev3_build` binaries;
  - `operand_census.py` and `real_metadata_plans.py` write into the committed `artifacts/p1t_rev3/`;
  - `release_failures.py` always prepends its own tree to PYTHONPATH, so on the before arm the copy's root is wt_fix2.
- **D3. `lifecycle_slice.cpp` is regenerated for the after build.** It embeds 5f9f955's lifecycle code verbatim.
  - `regen_lifecycle_slice.py` splits it around that slice and re-cuts the same region from the new source with the
    same markers.
  - Self-check: regenerating from 5f9f955 reproduces the committed file byte for byte. The diff is
    `probe_runs/lifecycle_slice_regen_after.diff`.
  - `dlpack_begin_shutdown` kept its single-double signature, so the reviewer's driver tail applies unchanged.
    `lifecycle_timeout.py` itself is unmodified.
- **D4. `pending_bytes` entry mode.**
  - The unmodified mode overflows on every build: the statement under test is a copy inside the probe. It reports on
    both arms.
  - Per the reviewer's instruction to adapt the driver, `make_pending_bytes_adapt.py` replaces exactly that loop with
    the real `RetainUntilComplete` on the same operands, with no launch.
  - Its 5f9f955 witness needs a visible GPU: without one, event creation fails first.
- **D5. Two conditions for the CPU probes**: nogpu (the reviewer's) and gpu (ours). Both are reported.
- **D6. host_checks `oom_sync` on a GPU host** gives rc -6 on both arms, after `PASS: nonallocating completion
  path`. This is FIX2's D4: the probe's own `pybind11::finalize_interpreter` teardown. It is unchanged, and rc 0 in
  the reviewer's condition.
- **D7. Semantics of the private `_dlpack_begin_shutdown`.**
  - The argument is renamed `timeout_s` → `first_diag_s` (positional-compatible).
  - The return dict gains `diagnostics` and `own_thread_in_flight`.
  - `timed_out` now means "the wait outlasted the first-diagnostic delay", and `still_in_flight` is always 0 at
    return.
  - In the stats, `shutdown_timeouts` counts such overruns, and `shutdown_diagnostics` is new.
- **D8. Notify on every leave**, not only at zero: the wait can target a nonzero count (the caller's own releases).
  This adds one `notify_all` per admitted release, which is cheap with no waiter. Performance was not measured (out of
  scope).
- **D9. Self-exclusion (thread-local depth)** goes beyond the literal order. Without it, the order's unbounded wait
  would self-deadlock when the hook runs inside its own thread's admitted release. It is tested
  (`test_fix3_v301_hook_inside_own_release_does_not_wait_for_itself`).
- **D10. Unusable knob.** If `float(_DLPACK_SHUTDOWN_WAIT_S)` fails, the hook passes NaN, which diagnoses at once and
  still waits. Before, the hook would have failed before closing admission.
- **D11. No `*_bindings.cpp` edit.** The pre-launch check is the first step of the `RetainUntilComplete`
  constructor. Every entry point constructs one right before its first launch, after host validation only: 9 sites,
  namely conv1d_run, conv_transpose1d_run, fp32_divide_probe, norm_run (the three norms), upsample_run, pad_run,
  rint_run, _test_rint_then_raise and lstm_run.
- **D12. Mutant (b) GPU variant.** The mutant's unchecked call launched a kernel onto fake addresses. The resulting
  `cudaErrorIllegalAddress` stayed inside the test's subprocess, and the GPU sanity check passed afterwards. I did not
  run the gpu prelaunch variant on 5f9f955, for the same reason.
- **D13. G4 r1 printed FAIL** because of two checker defects:
  - it recomputed the probe hash through `/bin/sh` (dash), whose glob order differs from bash's (same lines; bash
    reproduces the registered `36502118…`);
  - it flagged FIX2's pre-existing test-injection `throw std::bad_alloc()` in the destructor, which the destructor's
    own `catch (...)` catches.

  Both were fixed in the checker and r2 passes. The r1 output is kept.
- **D14. Own test error, first run.** The self test expected `shutdown_still_in_flight == 1`. By construction it
  counts other threads' releases (0); my C++ comment had stated the opposite. I corrected the comment and the
  assertion and re-ran (ledger 01:03-01:04). The comment fix left the binary byte-identical.
- **D15. Before-witness quality.** On 5f9f955, the new 0.1 s and idle tests fail only on a missing stats key, not on
  the defect. The defect witnesses are the reviewer's probes, the 5.5 s and 12 s tests (`deleter_done False`), the
  self test (5.0 s self-wait), and mutants (a) and (b).
- **D16. Round 1 and round 2 ran on the after build only**, as a no-regression check. The before arm is the
  round-3 table.
- **D17. Network and lock.**
  - The fresh core configure, run with `./build.sh 89` as ordered, fetched pybind11 v2.12.0 (3e9dfa28) from
    github.com through the CMakeLists' FetchContent GIT_REPOSITORY.
  - CPU-only compiles, generators and the harness `accounting` mode ran outside the GPU lock. Every GPU-touching
    command used `flock -w 900 /tmp/forge-gpu.lock timeout …`.

## Honest residuals (not claimed fixed, or not tested)

- **A deleter that never returns** makes interpreter exit wait forever, by design and documented. Ctrl-C does not
  interrupt the wait.
- **Lock failure on leave.** If `std::mutex::lock` throws when a release leaves (never observed; counted in
  `lock_failures`), `in_flight` stays raised and exit waits forever. FIX2 timed out after 5 s there and could then
  abort.
- **Hook failure before closing.** If the hook itself fails before setting `closing` (a theoretical
  lock/system_error), it logs and returns without coordination. This is unchanged from FIX2.
- **Forced unwind** (a deleter that calls `pthread_exit` or is cancelled) is still not containable (FIX2 residual).
- **Saturated accounting** over-reports until the list empties, which only causes extra bound drains. It is
  reachable only with more than 8 EiB of claimed operands.
- **Multi-GPU** is not exercised: two probes are INCONCLUSIVE on this one-GPU host.
- **The unmodified `pending_bytes` entry mode** cannot pass on any build (D4).
- **Free-threaded CPython** is unsupported (unchanged).
- **Performance** was not measured (out of scope).

## Diff stat

Code changed by this order (`git diff --stat 5f9f955 -- tensor_cuda tc_signal`, working tree):
```
 tc_signal/README.md                 |  24 ++++++-
 tc_signal/src/dlpack_operand.cpp    |  78 +++++++++++++++++---
 tc_signal/src/dlpack_operand.h      |  20 +++++-
 tensor_cuda/src/bindings.cpp        | 140 +++++++++++++++++++++++++++++-------
 tensor_cuda/tensor_cuda/__init__.py |  44 +++++++++---
 tensor_cuda/tests/test_dlpack.py    | 122 ++++++++++++++++++++++++++-----
 6 files changed, 363 insertions(+), 65 deletions(-)
```
- New untracked files: `tc_signal/tests/test_fix3_findings.py` (167 lines), `tc_signal/tests/fix3_native_harness.cpp`
  (319), `docs/P1T_FIX3_LEDGER.md`, this report, `artifacts/p1t_fix3/**` and `logs/p1t_fix3_*`.
- The literal `git diff --stat 5f9f955` also lists the lead's round-3 receipt and order commits already on voice-fix3
  (62 files under `artifacts/p1t_rev3/`, `logs/p1t_rev3*`, `docs/P1T_REV3_*` and `orders/`).
- `.gitignore` ignores `*.log` and `*.json`, so those receipts need `git add -f`.

## Changed files

- Core: `tensor_cuda/src/bindings.cpp` (import-release lifecycle only, plus `<cstdio>`),
  `tensor_cuda/tensor_cuda/__init__.py` (the atexit hook and knob), `tensor_cuda/tests/test_dlpack.py`.
- tc_signal: `src/dlpack_operand.h`, `src/dlpack_operand.cpp`, `README.md`; new `tests/test_fix3_findings.py` and
  `tests/fix3_native_harness.cpp`.
- Docs: `docs/P1T_FIX3_REPORT.md`, `docs/P1T_FIX3_LEDGER.md`.
- Receipts: `artifacts/p1t_fix3/**` and `logs/p1t_fix3_*`.
- Built and gitignored: the core and tc_signal `.so` files and their `build/` directories.

**Scratch left (nothing deleted):**
- `/tmp/p1t_fix3_probes/{before,after,rev2_after}`: probe builds and path copies;
- `/tmp/p1t_fix3_mut_{a,b}`: mutant trees and builds;
- `/tmp/p1t_fix3_scratch`: a direct harness build;
- `/tmp/p1t_fix3_slice_check.txt`;
- pytest temporary directories under `/tmp/pytest-of-vader/`.
