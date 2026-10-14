# P1T-FIX8: close round-8 findings V8-01 and V8-02, and sweep their two defect classes across the P1-T code

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix8`
(branch `voice-fix8`: FIX7 6991939, plus the round-8 order and receipts, 1cd8824). Edits, core and `tc_signal`
builds, and GPU runs inside it are AUTHORIZED. You may also write scratch and mutant builds under `/tmp/p1t_fix8_*`.

## Why

David, verbatim: "If the Blind Review passes, I approve the merger." Also, verbatim: "Bugs must be squashed." and
"bugs don't get to be post-hoc accepted."

Round 8 of the independent review (`docs/P1T_REV8_REPORT.md`, `artifacts/p1t_rev8/findings.json`) gave REJECT, with
**0 BLOCKING, 0 MAJOR and 2 MINOR**: R1 SAFE, R2 UNSOUND, R3 UNSOUND, R4 CORRECT, R5 PASS.
- V7-01 and V7-02 are CLOSED. So are N5, V3-01 and V5-01, the three items round 7 left PARTLY closed.
- It accepted U1–U4 and U6–U9, and all four FIX7 "Not a defect" items.
- Old item N1 ("a CUDA failure on a P1-T path is recorded and reported, never lost") is PARTLY closed again, through
  the two findings below.
- The lead reproduced V8-01 on the GPU (`artifacts/p1t_rev8/lead/diagnostic_events_gpu.log`: the create, record and
  query failures are all erased).

The findings:
- **V8-01 (MINOR). The L1 exit device-wait marker erases its own CUDA failures.**
  - Site: `tc_signal/src/dlpack_operand.cpp` `wait_device_work_diagnosed`, around lines 1049–1094 (FIX7 L1, row a3,
    user-visible change U10).
  - **Scenario:** inside the exit runner's window, `cudaEventCreateWithFlags`, `cudaEventRecord`, `cudaEventQuery` or
    `cudaEventDestroy` fails once with a non-sticky error, and the later `cudaStreamSynchronize` succeeds. The helper
    clears the error with `cudaGetLastError` and stores no `FailureRecord`, so the failure is never reported.
  - A failed create or record also falls straight through to an undiagnosed stream wait, which breaks a3's own
    guarantee.
  - After a query error that follows a diagnostic line, the helper can print "finished after", although nothing
    proved completion.
- **V8-02 (MINOR). The native exit loggers silently drop recorded CUDA failures when building the report fails to
  allocate.**
  - Sites: `tc_signal/src/dlpack_operand.cpp` `exit_log_unreported` (around 1759–1770) and the core runner's
    reporting in `tensor_cuda/src/bindings.cpp` (around 1660–1672).
  - **Scenario:** the hook is interrupted, so the native runner is the last reader. A CUDA failure is already
    recorded, and building its rich message (`report_message` / `unraisable_message`, or the core record snapshot)
    throws `bad_alloc`. The empty catch writes nothing. Keeping the record does not help after interpreter exit,
    because no later `synchronize()` will run.
- **Corrections to FIX7's report** (for your corrected-rows table):
  - rows i24, i25, j24 and a3 become PARTLY, through these two findings;
  - row a9's rationale must account for `std::string` allocation (the reviewer's correction).

**Each round has found a new member of a defect class.** So fix these sites, and also sweep both classes across ALL
of the P1-T code (`git diff 9df5bef..HEAD -- tensor_cuda tc_signal`):

- **(k) Consumed CUDA errors.** List every site in P1-T code that consumes the runtime's error state or drops a CUDA
  return code:
  - every `cudaGetLastError` call;
  - every `cudaPeekAtLastError` branch;
  - every CUDA call whose return value is ignored or overwritten;
  - every `if (r != cudaSuccess) cudaGetLastError();`.

  For each, show one of these:
  - the failure is recorded, tagged with call, purpose, device and error, BEFORE it is consumed, and reaches the
    raise-once path (`synchronize()`) or the exit logger; or
  - it is not a failure. `cudaErrorNotReady` from `cudaEventQuery` or `cudaStreamQuery` is the documented "still
    running" answer. Cite the CUDA 12.6 Runtime API.

  Diagnostic-only code is covered too: a failure in code that exists only to diagnose is still a CUDA failure on a
  P1-T path.
- **(l) The last reader at exit.** List every path at interpreter exit that is the final reader of recorded failures:
  - both native runners;
  - both hooks;
  - the post-close `synchronize()` retry;
  - `exit_log_unreported`;
  - the core R7 logging.

  For each, show that when building the rich report fails (`bad_alloc` / `MemoryError`, or a failing `str()` /
  stderr), an allocation-free fallback still writes, with C stdio, the record count and the first record (call,
  purpose, device, and the error's name from `cudaGetErrorName`, which returns a static string). Also list every empty
  or swallowing `catch` on a reporting path, with its verdict.

## Defect residuals: none (David's rule)

- A known real defect may not be listed as a "residual". Fix it in this round, with a test.
- An item may stay unfixed only if you show, with a citation, that it is NOT a defect in our code. Put it in a
  "Not a defect (with justification)" table.
- Any change to user-visible behaviour must be justified as correct, not just disclosed.
- The lead gates your report against this rule before the next review. A defect parked as a residual comes back to
  you.

## Mechanism & scope

- **Measures:**
  - whether V8-01 and V8-02 are closed: the round-8 probes fail on 6991939 and pass on your build;
  - whether sweeps (k) and (l) leave no site unhandled;
  - that nothing regresses: all round-1..8 probes, Voice's runner (T1–T6 plus T1R and T4S), and the combined
    session (52 baseline ids).
- **Mechanism doc:** `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix8/docs/P1T_REV8_REPORT.md`.
  Also:
  - `artifacts/p1t_rev8/findings.json`;
  - the probe headers in `artifacts/p1t_rev8/probes/`;
  - `artifacts/p1t_rev8/lead/` (the lead's GPU runs);
  - `docs/P1T_FIX7_REPORT.md` (its sweeps, the L1 addendum and its "Not a defect" table);
  - `artifacts/p1t_fix7/lead/LEAD_VERIFICATION.md`.
- **Exercises:**
  - add-on: `tc_signal/src/*`, `tc_signal/tc_signal/*`, `tc_signal/tests/*`, `tc_signal/README.md`;
  - core: `tensor_cuda/src/bindings.cpp` (the P1-T exit runner and error reporting), `tensor_cuda/src/kernels.cu`
    (P1-T sites the sweeps find), `tensor_cuda/tensor_cuda/__init__.py`, `tensor_cuda/tests/test_dlpack.py`;
  - every reviewer probe, run read-only.
- **Does not cover:**
  - pre-existing core code outside the P1-T surface;
  - new features;
  - performance;
  - Voice's tests (READ-ONLY);
  - real two-GPU execution (this host has one GPU).
- **Required evidence:**
  - the per-finding table (fix at file:line, proving test, probe before and after);
  - sweep tables (k) and (l): every site with a verdict (OK / FIXED / NOT A FAILURE, the last with its citation) and,
    for FIXED, a test or a stated reason why none is feasible;
  - FIX7's corrected rows (i24, i25, j24, a3, a9);
  - the "Not a defect (with justification)" table, if any;
  - the user-visible changes, each justified;
  - the regression logs;
  - the mutant table;
  - the diff stat against 6991939;
  - both `.so` sha256s.
- **Operand census:**
  - **V8-01:** one-shot failure of each marker call (create, record, query, destroy). Each must be reported exactly
    once, naming its call, purpose and device. A failed create or record must still give a diagnosed, completed wait.
    "Finished" is printed only when completion was proved. Run each with and without message-allocation failure, plus
    a healthy control that writes no failure.
  - **V8-02:** both native exit loggers, add-on and core:
    - a recorded failure plus rich-message allocation failure: the fallback line names the count and the first record;
    - the control without allocation failure: the rich message, unchanged;
    - stderr closed: no crash, and the record is still acknowledged.

    The hook's own exception is preserved, and quiescence (close and wait) still happens first.
- **Baseline isolation:**
  - "Before" means the 6991939 build, which the reviewer reviewed. Its binaries are in
    `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_p1trev8`, together with the
    round-8 probes, and its root is where the probes find their package (`parents[3]`). Use that tree read-only as
    the before arm, or a probe tree holding byte-identical copies, and report which.
  - "After" means your build, in fresh processes, under the GPU lock.
  - The combined session is compared with the 52-id baseline log named below.
  - Mutants are built outside the worktree.
- **Forbidden configuration:**
  - any ordinary (non-exit) API call that waits for another thread's user callback (V6-01 stays closed);
  - returning to ordinary finalization while an admitted release or callback is running;
  - any path that skips the close-and-wait (V7-01 stays closed);
  - an exit wait without its diagnostic (L1 stays closed);
  - editing any reviewer probe or Voice test, or changing Voice's adapter;
  - `os._exit`, `abort` or `std::terminate` as an error path;
  - throwing from destructors;
  - device-wide syncs on the per-call path;
  - fast-math, TF32, atomics or new core `__global__` kernels;
  - linking `tc_signal` to `_tensor_cuda`;
  - writing outside the target or `/tmp`.

## Required fixes

1. **V8-01.**
   - Each failed marker call is recorded as a tagged `FailureRecord` before its error is consumed, and is reported
     through the existing raise-once path.
   - If the marker cannot be created or recorded, the wait must still be diagnosed. A polled `cudaStreamQuery` on the
     legacy stream is one candidate; it returns `cudaErrorNotReady` while work is pending (CUDA 12.6 Runtime API).
   - Print "finished" only after completion is proved.
2. **V8-02.** Both native exit loggers get an allocation-free fallback. With C stdio and no allocation, it writes the
   record count and the first record when the rich report cannot be built. Quiescence still comes first, and the
   hook's exception is preserved.
3. **Sweeps (k) and (l).** Fix every site they find, with a test where one is feasible.
4. **Documentation.** The README and the docstrings state it: a CUDA failure on a P1-T path, diagnostic code
   included, is reported, at exit too, even when the report cannot allocate.

## How to run

- **GPU:** every GPU command runs as `flock -w 900 /tmp/forge-gpu.lock timeout …`. Use the native driver only. You
  are the only seat running in this lane.
- **Budget:** the Claude 7-day budget is at caution. Be economical:
  - run the combined session, Voice's runner and the mutants ONCE, at the end, on the final build;
  - re-run one only if product code changes after it;
  - reuse FIX7's runners (`artifacts/p1t_fix7/run_rev7_probes.sh`, `artifacts/p1t_fix7/runners/`,
    `artifacts/p1t_fix7/g4/`), retargeted to this worktree.
- **Builds:**
  - core: `cd <target>/tensor_cuda && ./build.sh 89`;
  - `tc_signal`: `cd <target>/tc_signal && PYBIND11_SOURCE_DIR=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src ./build.sh 89`.
- **Package path:** `PYTHONPATH=<target>/tensor_cuda/tensor_cuda:<target>/tensor_cuda:<target>/tc_signal`, plus
  `PYTHONDONTWRITEBYTECODE=1`.
- **Round-8 probes.** Run them on both arms, with the commands in their headers and in `docs/P1T_REV8_REPORT.md`
  "Probes". The CPU builds are `probes/build_cpu.sh`, `probes/build_core_host.py` and `probes/build_core_host_oom.py`;
  rebuild them from each arm's own tree, with a self-check. The probes:
  - `diagnostic_errors_cpu.py` (with `diagnostic_errors_cpu.cpp`): V8-01, CPU;
  - `verify_exit_log_cpu.py` (child `exit_log_oom_cpu.py`): V8-02 add-on, the OOM arm and the control;
  - `core_exit_log_oom_cpu.py`: V8-02 core, the OOM arm and the control;
  - `diagnostic_events_gpu.py`, with `diagnostic_event_shim.cpp` built as its header says: V8-01 on the GPU;
  - `lock_checks_cpu.py`, `fix7_cpu_census.py` (36 children), `attribution_cpu.py` (22 assertions),
    `argument_census_cpu.py` (1067 checks), `operand_census_cpu.py` (17), `run_cpu_checks.py` (14),
    `knob_exit_cpu.py`, `core_knob_exit_cpu.py` (each with `--control`) and `query_diagnostic_cpu.py`;
  - `gpu_regressions.py fix7`, `fix56`, `semantics` and `multidevice` (multidevice exits 3 on one GPU).

  The round-8 probes write their receipts into `artifacts/p1t_rev8/`, which must stay byte-identical. Use path-only
  copies that write elsewhere, and report them, as FIX7's D1 did.
- **Earlier probes.** Run the round-7 G1 probes and their controls (FIX7's `run_rev7_probes.sh`), the round-6 and
  round-5 runners, and rounds 1–3 (`run_rounds123_probes.sh`), all retargeted, after arm only, with ABSOLUTE output
  directories.
- **Voice's runner, T1–T6 plus T1R and T4S.** Run it from `/mnt/ForgeRealm/Project-Voice`, which is READ-ONLY to you:
  ```
  cd /mnt/ForgeRealm/Project-Voice && flock -w 900 /tmp/forge-gpu.lock timeout 1200 env PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=<pp> python3 engine/tensor_requests/p1t_tests/run_p1t.py --adapter <target>/artifacts/p1t_final/tensorcuda_adapter.py \
    --items T1,T2,T3,T4,T5,T6,T1R,T4S --fixtures /mnt/ForgeRealm/Project-Voice/artifacts/p1t/fixtures \
    --tolerances engine/tensor_requests/p1t_tests/tolerances_r2.json --out <target>/artifacts/p1t_fix8/p1t_all.json
  ```
- **The combined session.** It takes about 12–14 minutes; run it detached, with a completion marker, and read it only
  after the marker exists.
  `cd <target> && flock -w 900 /tmp/forge-gpu.lock timeout 3000 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<pp> python3 -m pytest tensor_cuda/tests tc_signal/tests -q -p no:cacheprovider --ignore=tensor_cuda/tests/test_selector_accuracy.py`.
  Compare it with
  `python3 /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2/scripts/apa_mf2_2_g1_compare.py /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2/logs/apa_mf1_g1_baseline_unchanged.log <log>`.
  The new tests add passes; the failing set must stay exactly the 52 baseline ids.

## Gates

- **G1 (closed):** each of these fails on 6991939 and passes on your build:
  - `diagnostic_errors_cpu.py`;
  - `verify_exit_log_cpu.py` (the OOM arm);
  - `core_exit_log_oom_cpu.py` (the OOM arm);
  - `diagnostic_events_gpu.py`.

  The controls pass on both builds. The new regressions for each finding and each census case pass.
- **G2 (sweeps):** tables (k) and (l) are complete. Every site is OK, FIXED or NOT A FAILURE (cited), and every FIXED
  site has a test or a stated reason. FIX7's rows i24, i25, j24, a3 and a9 are corrected.
- **G3 (no regression):**
  - every round-1..8 probe passes, or is INCONCLUSIVE only for lack of 2 GPUs. The round-5 `exit_gate trace` is
    handled as in FIX6 and FIX7;
  - `gpu_regressions.py fix7`, `fix56` and `semantics` PASS with zero skips;
  - Voice's runner gives PASS 249/249;
  - the combined session shows exactly the 52 baseline ids;
  - real-torch DLPack interop passes.
- **G4 (teeth):** build these mutants outside the worktree. Each must fail its new tests and its round-8 probe:
  - (e) the record of a failed marker call removed, so it is consumed silently again;
  - (f) the add-on exit logger's allocation-free fallback removed;
  - (g) the core runner's allocation-free fallback removed.
- **G5 (scope):**
  - the diff against 1cd8824 touches only the Exercises paths, plus `docs/P1T_FIX8_*.md`,
    `artifacts/p1t_fix8/**` and `logs/**`;
  - the probe files under `artifacts/p1t_rev*` are byte-identical;
  - no new core `__global__`.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. Your order file is
  immutable.
- NO git writes (read-only git is fine). NO subagents. Run every gate to completion before you read its result.
- No deletes, anywhere. Never kill a process you did not start.
- These are READ-ONLY: production `/mnt/ForgeRealm/Project-Tensor` (outside `Fork/`), `/mnt/ForgeRealm/Project-Voice`,
  `wt_p1trev8` (your before arm), and every other fork or worktree.
- RED honesty: a failed gate is a result. Report it with receipts; never weaken a gate, probe, test or tolerance.
- Prior Art Directive (AGENTS.md): annotate at the code site, in the ledger and in the report. Known prior art:
  - CUDA 12.6 Runtime API: the error-slot contract of `cudaGetLastError` / `cudaPeekAtLastError`, `cudaEventQuery` /
    `cudaStreamQuery` returning `cudaErrorNotReady`, and `cudaGetErrorName` returning static strings;
  - FIX5 V5-02's record-before-consume rule;
  - allocation-free last-chance reporting, as in async-signal-safe logging (POSIX `write(2)`) and SQLite's
    OOM-testing discipline;
  - C++17 `std::bad_alloc`.
- If the AFTERIMAGE memory hook denies a Write, read what it shows you and retry. If a report Write is refused
  outright, put the full report in your final message.

## Done

- Write `docs/P1T_FIX8_REPORT.md` and `docs/P1T_FIX8_LEDGER.md`. The report holds:
  - the per-finding table;
  - sweep tables (k) and (l);
  - FIX7's corrected rows;
  - the "Not a defect" table;
  - the user-visible changes;
  - the probe before/after table;
  - the gate table;
  - the mutant table;
  - prior art.
- Your final message must contain, verbatim:
  - the per-finding table;
  - each sweep table, with a summary count;
  - the "Not a defect" table and the user-visible changes;
  - the probe before/after table;
  - the gate table;
  - Voice's verdict line;
  - the compare output;
  - the mutant table;
  - the diff stat;
  - both `.so` sha256s;
  - every deviation, with its reason.
