# P1T-FIX7: close round-7 findings V7-01 and V7-02, and sweep their two defect classes across the P1-T code

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix7`
(branch `voice-fix7`: FIX6 429446d, plus the round-7 order and receipts, 8035b3b). Edits, core and `tc_signal` builds,
and GPU runs inside it are AUTHORIZED. You may also write scratch and mutant builds under `/tmp/p1t_fix7_*`.

## Why

David, verbatim: "If the Blind Review passes, I approve the merger." Also, verbatim: "Bugs must be squashed." and
"bugs don't get to be post-hoc accepted."

Round 7 of the independent review (`docs/P1T_REV7_REPORT.md`, `artifacts/p1t_rev7/findings.json`) gave REJECT:
R1 SAFE, R2 UNSOUND, R3 UNSOUND, R4 CORRECT, R5 PASS. It accepted all of FIX6's user-visible changes (U1–U9) and all
of its "Not a defect" items (N1–N9). Of the 25 earlier items, 22 are CLOSED; N5, V3-01 and V5-01 are only PARTLY
closed, all through V7-01. The lead reproduced both findings on the GPU (`artifacts/p1t_rev7/lead/`).

- **V7-01 (MAJOR). The exit hooks skip their close-and-wait when the diagnostic-delay knob cannot be converted.**
  - Sites: `tc_signal/tc_signal/__init__.py:194-209` (`_release_pending_at_exit`, knob `_RELEASE_EXIT_DIAG_S`) and
    `tensor_cuda/tensor_cuda/__init__.py:217-240` (`_dlpack_shutdown_hook`, knob `_DLPACK_SHUTDOWN_WAIT_S`).
  - **Scenario:** the knob is set to `1 << 4096`. At exit, `float(knob)` raises OverflowError, which the fallback
    `except (TypeError, ValueError)` does not catch. So the outer `except Exception` logs and returns without calling
    the close-and-wait entry. A finite producer deleter is still running on a daemon thread. Finalization overtakes
    it, and the process aborts.
  - FIX6 exposed this in the add-on: before FIX6, `synchronize()` waited first. The core copy predates FIX6.
  - **Lead receipts on 429446d:**
    - `core_knob_exit_gpu.py`: child rc -6;
    - `knob_exit_cpu.py` with the GPU visible: rc 134;
    - both `--control` runs: rc 0.
  - **The reviewer's fix:** the close-and-wait must not depend on the diagnostic configuration. Handle any conversion
    failure with a safe default, still call the close, and add finite-callback exit regressions for both hooks.
- **V7-02 (MINOR). A failed completion-event query is reported as a failed stream synchronization.**
  - Sites: `tc_signal/src/dlpack_operand.cpp:1048-1058` (phase 2 `cudaEventQuery`), `:1389-1392` (the message), and
    `SyncReport` `:734-759`.
  - **Scenario:** phase 1's legacy-stream sync succeeds, and phase 2's `cudaEventQuery` fails once. The failure goes
    into the same untyped `dev` / `err` arrays as stream failures. The message then says "the legacy stream of device
    0 could not be synchronized", which is false, and it never names `cudaEventQuery`.
  - **Lead receipt:** `query_diagnostic_gpu.py` FAILs on 429446d ("query failure attributed to sync").
  - **The reviewer's fix:** carry an allocation-free failure-kind tag in `SyncReport`, keep it through `merge`, and
    build the message from it.
- **Two reasoning corrections in FIX6's report** (docs only, for your report's corrected-rows table):
  - h8: a native DLPack consumer may drop an exported context without holding the GIL, so `PyGILState_Ensure` there
    is not always reentrant. The conclusion stands, because no P1-T mutex is held at that point.
  - g17: there are 18 plan/run bindings (9 ops × 2), not 20.

**Each round has found a new member of a defect class.** So fix these sites, and also sweep both classes across ALL
of the P1-T code (`git diff 9df5bef..HEAD -- tensor_cuda tc_signal`):

- **(i) Paths that reach an exit guarantee.** For each exit hook (core `_dlpack_shutdown_hook` and the add-on
  `_release_pending_at_exit`), list every statement before the close-and-wait entry, and every exception it can
  raise. Then show that the close-and-wait entry is reached on EVERY path:
  - the knob holds any object: a huge int, ±inf, NaN, a negative value, a float or int subclass, a string, None, an
    object whose `__float__` raises any exception (including a BaseException subclass) or returns a non-float, or the
    attribute has been deleted;
  - `_C.synchronize()` raises anything at exit;
  - stderr is closed or raises;
  - an exception arrives asynchronously at any bytecode boundary before the close (KeyboardInterrupt from Ctrl-C
    during exit).

  Apply the same rule to the native raw entries (`_C._exit_close_admission`, `_C._dlpack_begin_shutdown`): a
  diagnostic argument they cannot read (wrong type, an int beyond the double range) must not prevent the close. It
  falls back to the default diagnostic delay, closes and waits. Also check every other guarantee that a hook gives on
  the way out (FIX6's raise-once logging, for example) and is skipped by an early exit.
- **(j) Error attribution.** List every failure that P1-T records or reports: every `SyncReport.add` / `rep.add`, the
  spill path's `rep.add(-1, ...)`, the prior-error slot, core's recorded unraisable errors, the exit-hook messages,
  and every message that names a CUDA API or a device. For each, check that the message names the API that actually
  failed (`cudaStreamSynchronize`, `cudaEventQuery`, `cudaGetDevice`, `cudaEventRecord`, …) and the right device, or
  says honestly that the device is unknown, and never names an operation that succeeded. Recording stays
  allocation-free and saturating (V6-03).

## Defect residuals: none (David's rule)

- A known real defect may not be listed as a "residual". Fix it in this round, with a test.
- An item may stay unfixed only if you show, with a citation, that it is NOT a defect in our code. Put it in a
  "Not a defect (with justification)" table.
- Any change to user-visible behaviour must be justified as correct, not just disclosed.
- The lead gates your report against this rule before the next review. A defect parked as a residual comes back to
  you.

## Mechanism & scope

- **Measures:**
  - whether V7-01 and V7-02 are closed: the round-7 probes fail on 429446d and pass on your build;
  - whether sweeps (i) and (j) leave no site unhandled;
  - that nothing regresses: all round-1..7 probes, Voice's runner (T1–T6 plus T1R and T4S), and the combined
    session (52 baseline ids).
- **Mechanism doc:** `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix7/docs/P1T_REV7_REPORT.md`.
  Also:
  - `artifacts/p1t_rev7/findings.json`;
  - the probe headers in `artifacts/p1t_rev7/probes/`;
  - `artifacts/p1t_rev7/lead/` (the lead's GPU runs);
  - `docs/P1T_FIX6_REPORT.md` (its sweeps and its "Not a defect" table);
  - `artifacts/p1t_fix6/lead/LEAD_VERIFICATION.md`.
- **Exercises:**
  - add-on: `tc_signal/src/*`, `tc_signal/tc_signal/*`, `tc_signal/tests/*`, `tc_signal/README.md`;
  - core: `tensor_cuda/src/bindings.cpp` (the P1-T exit entry and error reporting),
    `tensor_cuda/src/kernels.cu` (P1-T sites the sweeps find), `tensor_cuda/tensor_cuda/__init__.py`,
    `tensor_cuda/tests/test_dlpack.py`;
  - every reviewer probe, run read-only.
- **Does not cover:**
  - pre-existing core code outside the P1-T surface;
  - new features;
  - performance;
  - Voice's tests (READ-ONLY);
  - real two-GPU execution (this host has one GPU).
- **Required evidence:**
  - the per-finding table (fix at file:line, proving test, probe before and after);
  - sweep tables (i) and (j): every site with a verdict (OK / FIXED) and, for FIXED, a test or a stated reason why
    none is feasible;
  - FIX6's corrected rows (h8, g17, and any (g)/(h) row that (i) changes);
  - the "Not a defect (with justification)" table, if any;
  - the user-visible changes, each justified;
  - the regression logs;
  - the mutant table;
  - the diff stat against 429446d;
  - both `.so` sha256s.
- **Operand census:**
  - **V7-01, for both hooks:**
    - every knob value class listed in (i);
    - the knob deleted;
    - `synchronize()` raising at exit (add-on);
    - KeyboardInterrupt raised inside the hook before the close (a test can raise it from the knob's `__float__`, or
      from a patched `_C.synchronize`);
    - each case with a finite producer deleter running on another thread at exit: it must finish before
      finalization, and the process exits 0;
    - the default knob as a control.
  - **V7-02:**
    - a phase-2 event-query failure after a successful phase-1 sync;
    - a phase-1 sync failure;
    - both at once;
    - the spill path;
    - a current-device query failure;
    - each with message-allocation failure (V6-03 must still hold);
    - every message names the right API and device, exactly once.
- **Baseline isolation:**
  - "Before" means the 429446d build, prebuilt in
    `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix6`; use it read-only.
  - The round-7 probes find their package through their own path (`parents[3]`). For the before arm, use a path-only
    copy, or a probe tree holding byte-identical copies of the 429446d binaries, and report it.
  - "After" means your build, in fresh processes, under the GPU lock.
  - The combined session is compared with the 52-id baseline log named below.
  - Mutants are built outside the worktree.
- **Forbidden configuration:**
  - any ordinary (non-exit) API call that waits for another thread's user callback (V6-01 stays closed);
  - returning to ordinary finalization while an admitted release or callback is running;
  - editing any reviewer probe or Voice test, or changing Voice's adapter;
  - `os._exit`, `abort` or `std::terminate` as an error path;
  - throwing from destructors;
  - device-wide syncs on the per-call path;
  - fast-math, TF32, atomics or new core `__global__` kernels;
  - linking `tc_signal` to `_tensor_cuda`;
  - writing outside the target or `/tmp`.

## Required fixes

1. **V7-01.** In both hooks, the close-and-wait is reached on every path, whatever the knob holds and whatever is
   raised before it. A `try` / `finally` structure is the obvious shape. The native raw entries close and wait even
   when their diagnostic argument is unreadable.
2. **V7-02.** Failure records carry their kind (and device). Messages are built from the kind. Recording stays
   allocation-free.
3. **Sweeps (i) and (j).** Fix every site that they find, with a test where one is feasible.
4. **Documentation.** The README and the hook docstrings state the guarantee: diagnostic configuration never gates
   exit quiescence.

## How to run

- **GPU:** every GPU command runs as `flock -w 900 /tmp/forge-gpu.lock timeout …`. Use the native driver only. You
  are the only seat running in this lane.
- **Builds:**
  - core: `cd <target>/tensor_cuda && ./build.sh 89`;
  - `tc_signal`: `cd <target>/tc_signal && PYBIND11_SOURCE_DIR=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src ./build.sh 89`.
- **Package path:** `PYTHONPATH=<target>/tensor_cuda/tensor_cuda:<target>/tensor_cuda:<target>/tc_signal`, plus
  `PYTHONDONTWRITEBYTECODE=1`.
- **Round-7 probes.** Run every probe in `artifacts/p1t_rev7/probes/` on both builds, with the commands in their
  headers and in `docs/P1T_REV7_REPORT.md` "CPU receipts and probes":
  - `knob_exit_cpu.py` and `core_knob_exit_cpu.py`, with and without `--control`;
  - `core_knob_exit_gpu.py`, with and without `--control`;
  - `query_diagnostic_cpu.py` and `query_diagnostic_gpu.py`;
  - `argument_census_cpu.py` (1067 checks);
  - `run_cpu_checks.py` (14 checks), built by `build_cpu.sh` / `build_core_host.py`;
  - `gpu_regressions.py fix56`, `semantics` and `multidevice` (multidevice exits 3 on one GPU).

  Probes that embed or regenerate product source slices are rebuilt from each arm's own tree, with a self-check, as
  FIX4–FIX6 did. Report path-only copies.
- **Earlier probes.** Then run the round-6 probes (`artifacts/p1t_rev6/probes/`, as FIX6 did:
  `artifacts/p1t_fix6/run_rev6_probes.sh`) and the round-5 probes (`artifacts/p1t_fix6/run_rev5_probes.sh`), retargeted
  to this worktree. The round-5 `exit_gate.py trace` as written still fails, as in FIX6 (its D7). Its exit-entry
  copy must pass. Then run round 1's `addon_lifetime` (3 runs) and round 3's `core_exit_overrun` (during and bounded).
  - Give the runners ABSOLUTE output directories. A relative one breaks the before arm's regeneration step, because
    the runner `cd`s into the arm's root.
- **Voice's runner, T1–T6 plus T1R and T4S.** Run it from `/mnt/ForgeRealm/Project-Voice`, which is READ-ONLY to you:
  ```
  cd /mnt/ForgeRealm/Project-Voice && flock -w 900 /tmp/forge-gpu.lock timeout 1200 env PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=<pp> python3 engine/tensor_requests/p1t_tests/run_p1t.py --adapter <target>/artifacts/p1t_final/tensorcuda_adapter.py \
    --items T1,T2,T3,T4,T5,T6,T1R,T4S --fixtures /mnt/ForgeRealm/Project-Voice/artifacts/p1t/fixtures \
    --tolerances engine/tensor_requests/p1t_tests/tolerances_r2.json --out <target>/artifacts/p1t_fix7/p1t_all.json
  ```
- **The combined session.** It takes about 10 minutes; run it to completion before you read its result.
  `cd <target> && flock -w 900 /tmp/forge-gpu.lock timeout 1500 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<pp> python3 -m pytest tensor_cuda/tests tc_signal/tests -q -p no:cacheprovider --ignore=tensor_cuda/tests/test_selector_accuracy.py`.
  Compare it with
  `python3 /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2/scripts/apa_mf2_2_g1_compare.py /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2/logs/apa_mf1_g1_baseline_unchanged.log <log>`.
  The new tests add passes; the failing set must stay exactly the 52 baseline ids.

## Gates

- **G1 (closed):** each of these fails on 429446d and passes on your build:
  - `knob_exit_cpu.py`;
  - `core_knob_exit_cpu.py`;
  - `core_knob_exit_gpu.py`;
  - `query_diagnostic_cpu.py`;
  - `query_diagnostic_gpu.py`.

  The `--control` variants pass on both builds. The new regression for each finding, and for each (i) value class,
  passes.
- **G2 (sweeps):** tables (i) and (j) are complete. Every site is OK or FIXED, and every FIXED site has a test or a
  stated reason. FIX6's rows h8 and g17, and any (g)/(h) row that (i) changes, are corrected.
- **G3 (no regression):**
  - every round-1..7 probe passes, or is INCONCLUSIVE only for lack of 2 GPUs. The round-5 `exit_gate trace` is
    handled as in FIX6;
  - `argument_census_cpu.py` passes all its checks, and `run_cpu_checks.py` passes 14/14;
  - `gpu_regressions.py fix56` and `semantics` PASS with zero skips;
  - `addon_lifetime` passes 3/3 runs with 0 incorrect;
  - `core_exit_overrun` during and bounded give rc 0;
  - Voice's runner gives PASS 249/249;
  - the combined session shows exactly the 52 baseline ids;
  - real-torch DLPack interop passes.
- **G4 (teeth):** build these mutants outside the worktree. Each must fail its new tests and its round-7 probe:
  - (a) the add-on hook's knob fallback narrowed back to `(TypeError, ValueError)`, with the close skipped on any
    other exception;
  - (b) the same in the core hook;
  - (c) the failure-kind tag dropped, so query failures are recorded as stream failures.
- **G5 (scope):**
  - the diff against 8035b3b touches only the Exercises paths, plus `docs/P1T_FIX7_*.md`, `artifacts/p1t_fix7/**`
    and `logs/**`;
  - the probe files under `artifacts/p1t_rev*` are byte-identical;
  - no new core `__global__`.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. Your order file is
  immutable.
- NO git writes (read-only git is fine). NO subagents. Run every gate to completion before you read its result.
- No deletes, anywhere. Never kill a process you did not start.
- These are READ-ONLY: production `/mnt/ForgeRealm/Project-Tensor` (outside `Fork/`), `/mnt/ForgeRealm/Project-Voice`,
  and every other fork or worktree.
- RED honesty: a failed gate is a result. Report it with receipts; never weaken a gate, probe, test or tolerance.
- Prior Art Directive (AGENTS.md): annotate at the code site, in the ledger and in the report. Known prior art:
  - CPython's own `threading._shutdown` and `atexit` handling, and its rule that an exit handler's exception is
    printed and the remaining handlers still run;
  - Python's `try` / `finally` guarantee, including for KeyboardInterrupt (the language reference, "The try
    statement");
  - CUDA 12.6's distinct return contracts for `cudaEventQuery` and `cudaStreamSynchronize`;
  - tagged ("discriminated") error records, as in `std::error_code` (C++11), which pairs a value with its category;
  - allocation-free error bookkeeping (SQLite's OOM-testing discipline).
- If the AFTERIMAGE memory hook denies a Write, read what it shows you and retry. If a report Write is refused
  outright, put the full report in your final message.

## Done

- Write `docs/P1T_FIX7_REPORT.md` and `docs/P1T_FIX7_LEDGER.md`. The report holds the per-finding table, sweep tables
  (i) and (j), FIX6's corrected rows, the "Not a defect" table, the user-visible changes, the probe before/after
  table, the gate table, the mutant table and prior art.
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
