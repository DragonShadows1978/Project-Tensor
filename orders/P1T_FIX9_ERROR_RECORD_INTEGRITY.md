# P1T-FIX9: close round-9 findings V9-01 and V9-02, and sweep their two defect classes across the P1-T code

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix9`
(branch `voice-fix9`: FIX8 e4cf1d6, plus the round-9 order and receipts, fd5fcb6). Edits, core and `tc_signal`
builds, and GPU runs inside it are AUTHORIZED. You may also write scratch and mutant builds under `/tmp/p1t_fix9_*`.

## Why

David, verbatim: "If the Blind Review passes, I approve the merger." Also, verbatim: "Bugs must be squashed." and
"bugs don't get to be post-hoc accepted."

David, 2026-10-13 evening, verbatim: "On THIS project which has bare access to metal, bugs can not be acceptable." A
review passes only with no open defect finding of any severity. MINOR findings are bugs too.

Round 9 of the independent review (`docs/P1T_REV9_REPORT.md`, `artifacts/p1t_rev9/findings.json`) gave REJECT, with
**0 BLOCKING, 0 MAJOR and 2 MINOR**: R1 SAFE, R2 UNSOUND, R3 SOUND, R4 CORRECT, R5 PASS.
- The add-on is now SOUND.
- V8-01 and V8-02 are fixed at their sites.
- Old item N1 ("a CUDA failure on a P1-T path is recorded and reported, never lost") stays PARTLY closed, through two
  core sites.
- Both findings have actual-source CPU reproductions. The lead re-ran them on e4cf1d6: `artifacts/p1t_rev9/lead/`.

The findings:
- **V9-01 (MINOR). The core export fence loses an earlier CUDA failure when a cleanup call also fails and the report
  cannot be allocated.**
  - Sites: `tensor_cuda/src/bindings.cpp` around lines 465–489, and `tensor_cuda/src/kernels.cu` around 331–334.
  - **Scenario:**
    - `cudaEventRecord` returns `cudaErrorInvalidValue`;
    - then `cudaEventDestroy` returns `cudaErrorInvalidResourceHandle` and overwrites the thread's error slot;
    - building the combined message or exception then throws `bad_alloc`.
  - **Result:** the record error lived only in a local variable. After unwinding, only the destroy error is in the
    slot, and nothing is recorded, so no later check can report the original error. The single-failure OOM case keeps
    it, and the compound case without OOM reports both; only compound plus OOM loses it.
  - Probe: `core_fence_cpu.py` (built by `build_core_fence_cpu.py`).
- **V9-02 (MINOR). The core exit fallback keeps a different snapshot from the report it acknowledges.**
  - Sites: `tensor_cuda/src/bindings.cpp` around lines 1632–1653 and 1715–1721, and `tensor_cuda/src/kernels.cu`
    around 411–438.
  - **Scenario:**
    - `dlpack_raise_unraisable` reads the allocation-free raw snapshot;
    - a native thread then records the first CUDA failure;
    - then the rich snapshot is read. The rich RuntimeError names that failure, and n = 1 is acknowledged, but the
      retained exit summary is the earlier, empty raw snapshot;
    - the hook cannot write (`sys.stderr` is None), so the runner prints an unnamed failure with a false "concurrent"
      explanation and consumes the summary.
  - **Result:** the real call, device and error never reach anyone. No lock or allocation failure is needed.
  - Probe: `core_snapshot_cpu.py` (built by `build_core_race_cpu.py`, with `core_snapshot_shim.c`).

**Each round has found a new member of a defect class.** This time both are in the error-reporting machinery itself.
So fix these sites, and also sweep both classes across ALL of the P1-T code, BOTH packages
(`git diff 9df5bef..HEAD -- tensor_cuda tc_signal`):

- **(m) A failure held only in a local.** List every site where a CUDA failure, or a failure record, lives only in a
  local variable or the runtime's error slot while something can still run before it is persisted or reported:
  - another CUDA call, which can overwrite the slot;
  - an allocation, which can throw;
  - a lock, which can fail;
  - user code.

  For each, show that it is persisted first, as an allocation-free tagged record, or that nothing can intervene.
- **(n) Report and acknowledgement from one snapshot.** List every place where recorded failures are read to build a
  report and are later acknowledged, consumed, retained or summarized. Cover:
  - `synchronize()`'s drain report and its clear;
  - the add-on exit logger and its acknowledgements;
  - the core R7 raise and its acknowledgement;
  - the core exit summary and the runner;
  - `drain_unreported`.

  For each, show that the report, the acknowledgement and anything retained all refer to the SAME snapshot (taken once
  under the lock, or checked by a generation counter), so that a record arriving in between is neither lost nor
  mislabelled.

**Structural direction (recommended, not mandated).** These findings come from error reporting that reads shared
state more than once. Prefer one design over another patch: a single allocation-free record store per package, with
one snapshot-and-acknowledge API keyed by a generation (or sequence) number, used by every reporter. If you choose it,
show that behaviour is unchanged everywhere else. If you choose site-by-site fixes instead, say why.

## Defect residuals: none (David's rule)

- A known real defect may not be listed as a "residual". Fix it in this round, with a test.
- An item may stay unfixed only if you show, with a citation, that it is NOT a defect in our code. Put it in a
  "Not a defect (with justification)" table.
- Any change to user-visible behaviour must be justified as correct, not just disclosed.
- The lead gates your report against this rule before the next review. A defect parked as a residual comes back to
  you.

## Mechanism & scope

- **Measures:**
  - whether V9-01 and V9-02 are closed: the round-9 probes fail on e4cf1d6 and pass on your build;
  - whether sweeps (m) and (n) leave no site unhandled;
  - that nothing regresses: all round-1..9 probes, Voice's runner (T1–T6 plus T1R and T4S), and the combined
    session (52 baseline ids).
- **Mechanism doc:** `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix9/docs/P1T_REV9_REPORT.md`.
  Also:
  - `artifacts/p1t_rev9/findings.json`;
  - the probe headers in `artifacts/p1t_rev9/probes/`;
  - `artifacts/p1t_rev9/lead/` (the lead's runs);
  - `docs/P1T_FIX8_REPORT.md` (its sweeps (k) and (l), the L2 addendum and its "Not a defect" table);
  - `artifacts/p1t_fix8/lead/LEAD_VERIFICATION.md`.
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
  - sweep tables (m) and (n): every site with a verdict (OK / FIXED) and, for FIXED, a test or a stated reason why
    none is feasible;
  - FIX8's corrected rows (any (k) or (l) row that V9-01 or V9-02 changes);
  - the "Not a defect (with justification)" table, if any;
  - the user-visible changes, each justified;
  - the regression logs;
  - the mutant table;
  - the diff stat against e4cf1d6;
  - both `.so` sha256s.
- **Operand census:**
  - **V9-01:** the export fence, for each fence call that can fail (current-device query, switch, create, record,
    wait, destroy, restore). For each:
    - a single failure;
    - a failure followed by a failing cleanup call (the compound case);
    - each of those with report-allocation failure.

    Every failed call must be reported, or persisted and then reported by the next `synchronize()` or at exit, naming
    its call and device, exactly once. Include a healthy control.
  - **V9-02:** the core report snapshot and its acknowledgement:
    - a first record arriving on another native thread between the snapshot and the acknowledgement, at every
      boundary of the current code;
    - a second record arriving the same way;
    - each with the hook able to write and unable to write (`sys.stderr` None).

    What is reported, retained and acknowledged must be the same records, and none may be lost or mislabelled.
  - **Sweeps (m) and (n):** each FIXED site gets the matching case.
- **Baseline isolation:**
  - "Before" means the e4cf1d6 build, which the reviewer reviewed. Its binaries are in
    `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_p1trev9`, together with the
    round-9 probes, and its root is where the probes find their package (`parents[3]` or `Path.cwd()`). Use that tree read-only as
    the before arm, or a probe tree holding byte-identical copies, and report which.
  - "After" means your build, in fresh processes, under the GPU lock.
  - The combined session is compared with the 52-id baseline log named below.
  - Mutants are built outside the worktree.
- **Forbidden configuration:**
  - any ordinary (non-exit) API call that waits for another thread's user callback (V6-01 stays closed);
  - returning to ordinary finalization while an admitted release or callback is running;
  - any path that skips the close-and-wait (V7-01 stays closed);
  - an exit wait without its diagnostic (L1 stays closed);
  - a native thread reclaimed while it can still run (L2 stays closed);
  - a CUDA failure consumed without being recorded first (V8-01 / (k) stay closed);
  - editing any reviewer probe or Voice test, or changing Voice's adapter;
  - `os._exit`, `abort` or `std::terminate` as an error path;
  - throwing from destructors;
  - device-wide syncs on the per-call path;
  - fast-math, TF32, atomics or new core `__global__` kernels;
  - linking `tc_signal` to `_tensor_cuda`;
  - writing outside the target or `/tmp`.

## Required fixes

1. **V9-01.** Every failed fence or cleanup call is persisted as an allocation-free tagged record (call, device,
   error) before the next CUDA call or allocation can run. Records are acknowledged only once the combined report
   exists. If the report cannot be built, the records stay for the next `synchronize()` or the exit reporter.
2. **V9-02.** The core takes ONE snapshot under the record mutex: count, static site pointer, device, error, and a
   generation or sequence number. It builds the rich message from that snapshot, retains that exact snapshot, and
   acknowledges exactly the records it covers. A record that arrives later is reported later and is never lost or
   mislabelled.
3. **Sweeps (m) and (n),** across both packages. Fix every site they find, with a test where one is feasible.
4. **Documentation.** The README and the docstrings state the rule: every recorded failure is reported exactly once,
   under its own name, even when reports race or cannot allocate.

## How to run

- **GPU:** every GPU command runs as `flock -w 900 /tmp/forge-gpu.lock timeout …`. Use the native driver only. You
  are the only seat running in this lane.
- **Budget:** the Claude 7-day budget is at caution. Be economical:
  - run the combined session, Voice's runner and the mutants ONCE, at the end, on the final build;
  - re-run one only if product code changes after it;
  - reuse FIX8's runners (`artifacts/p1t_fix8/run_rev8_probes.sh`, `run_rev8_g1_l2.sh`, `artifacts/p1t_fix8/runners/`,
    `artifacts/p1t_fix8/g4/` in `wt_fix8`), copied and retargeted to this worktree.
- **Builds:**
  - core: `cd <target>/tensor_cuda && ./build.sh 89`;
  - `tc_signal`: `cd <target>/tc_signal && PYBIND11_SOURCE_DIR=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src ./build.sh 89`.
- **Package path:** `PYTHONPATH=<target>/tensor_cuda/tensor_cuda:<target>/tensor_cuda:<target>/tc_signal`, plus
  `PYTHONDONTWRITEBYTECODE=1`.
- **Round-9 probes.** Run them on both arms, with the commands in their headers and in `docs/P1T_REV9_REPORT.md`
  "Probe list".
  - **CPU builds:** `probes/build_cpu.sh`, `build_core_host.py`, `build_core_host_oom.py`, `build_core_fence_cpu.py`
    and `build_core_race_cpu.py`. Rebuild them from each arm's own tree, with a self-check. Several use `Path.cwd()` as
    the arm root and `/tmp/p1t_rev9_build` as scratch: use path-only copies that build in `/tmp/p1t_fix9_rev9/<arm>`.
  - **The probes:**
    - `core_fence_cpu.py`: V9-01;
    - `core_snapshot_cpu.py` (with `core_snapshot_shim.c`): V9-02;
    - every other probe in the list: `acknowledgement_cpu.py`, `watchdog_cpu.py` (with `watchdog_shim.c`),
      `diagnostic_errors_cpu.py`, `verify_exit_log_cpu.py`, `verify_core_exit_log_cpu.py`, `lock_checks_cpu.py`,
      `fix7_cpu_census.py`, `attribution_cpu.py`, `argument_census_cpu.py`, `operand_census_cpu.py`,
      `run_cpu_checks.py`, `knob_exit_cpu.py`, `core_knob_exit_cpu.py` (with `--control`) and `query_diagnostic_cpu.py`;
    - `diagnostic_events_gpu_legacy.py` with its shim (GPU);
    - `gpu_regressions.py fix8`, `fix7`, `fix56`, `semantics` and `multidevice` (multidevice exits 3 on one GPU).
  - The round-9 probes write their receipts into `artifacts/p1t_rev9/`, which must stay byte-identical. Use path-only
    copies that write elsewhere, and report them.
- **Earlier probes.** Run the round-8 G1 probes (FIX8's `run_rev8_g1_l2.sh`, with the legacy-stream copy), the
  round-7 G1 probes and their controls, the round-6 and round-5 runners, and rounds 1–3, all retargeted, after arm
  only, with ABSOLUTE output directories. Copy FIX8's runners from `wt_fix8`; do not edit them in place.
- **Voice's runner, T1–T6 plus T1R and T4S.** Run it from `/mnt/ForgeRealm/Project-Voice`, which is READ-ONLY to you:
  ```
  cd /mnt/ForgeRealm/Project-Voice && flock -w 900 /tmp/forge-gpu.lock timeout 1200 env PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=<pp> python3 engine/tensor_requests/p1t_tests/run_p1t.py --adapter <target>/artifacts/p1t_final/tensorcuda_adapter.py \
    --items T1,T2,T3,T4,T5,T6,T1R,T4S --fixtures /mnt/ForgeRealm/Project-Voice/artifacts/p1t/fixtures \
    --tolerances engine/tensor_requests/p1t_tests/tolerances_r2.json --out <target>/artifacts/p1t_fix9/p1t_all.json
  ```
- **The combined session.** It takes about 12–14 minutes; run it detached, with a completion marker, and read it only
  after the marker exists.
  `cd <target> && flock -w 900 /tmp/forge-gpu.lock timeout 3000 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<pp> python3 -m pytest tensor_cuda/tests tc_signal/tests -q -p no:cacheprovider --ignore=tensor_cuda/tests/test_selector_accuracy.py`.
  Compare it with
  `python3 /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2/scripts/apa_mf2_2_g1_compare.py /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2/logs/apa_mf1_g1_baseline_unchanged.log <log>`.
  The new tests add passes; the failing set must stay exactly the 52 baseline ids.

## Gates

- **G1 (closed):** each of these fails on e4cf1d6 and passes on your build:
  - `core_fence_cpu.py`;
  - `core_snapshot_cpu.py`.

  Their controls pass on both builds. The new regressions for each finding and each census case pass.
- **G2 (sweeps):** tables (m) and (n) cover both packages and are complete. Every site is OK or FIXED, and every FIXED
  site has a test or a stated reason. FIX8's affected rows are corrected.
- **G3 (no regression):**
  - every round-1..9 probe passes, or is INCONCLUSIVE only for lack of 2 GPUs. The round-5 `exit_gate trace`, the
    as-written round-5 probes that embed 1475fae code, and the as-written `diagnostic_events_gpu.py` (ruling D2) are
    handled as in FIX6–FIX8;
  - `gpu_regressions.py fix8`, `fix7`, `fix56` and `semantics` PASS with zero skips;
  - the legacy-stream `diagnostic_events_gpu` copy passes;
  - Voice's runner gives PASS 249/249;
  - the combined session shows exactly the 52 baseline ids;
  - real-torch DLPack interop passes.
- **G4 (teeth):** build these mutants outside the worktree. Each must fail its new tests and its round-9 probe:
  - (i) the fence's persistence of a failed call removed, so a cleanup failure plus report OOM loses it again;
  - (j) the snapshot split again: the retained or acknowledged snapshot read separately from the one the report is
    built from.
  - If you choose the structural store, add (i2), a mutant that breaks its generation check.
- **G5 (scope):**
  - the diff against fd5fcb6 touches only the Exercises paths, plus `docs/P1T_FIX9_*.md`,
    `artifacts/p1t_fix9/**` and `logs/**`;
  - the probe files under `artifacts/p1t_rev*` are byte-identical;
  - no new core `__global__`.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. Your order file is
  immutable.
- NO git writes (read-only git is fine). NO subagents. Run every gate to completion before you read its result.
- No deletes, anywhere. Never kill a process you did not start.
- These are READ-ONLY: production `/mnt/ForgeRealm/Project-Tensor` (outside `Fork/`), `/mnt/ForgeRealm/Project-Voice`,
  `wt_p1trev9` (your before arm), `wt_fix8`, and every other fork or worktree.
- RED honesty: a failed gate is a result. Report it with receipts; never weaken a gate, probe, test or tolerance.
- Prior Art Directive (AGENTS.md): annotate at the code site, in the ledger and in the report. Known prior art:
  - CUDA 12.6 Programming Guide 3.2.12: one per-host-thread runtime error variable, overwritten by the next failing
    call;
  - FIX5 V5-02's record-before-consume rule, and FIX8's (k) / (l) sweeps;
  - consistent snapshots under one lock (C++17 mutex synchronization; POSIX threads), and sequence or generation
    counters (seqlocks; the Linux kernel's `seqcount_t`);
  - SQLite's OOM-testing discipline.
- If the AFTERIMAGE memory hook denies a Write, read what it shows you and retry. If a report Write is refused
  outright, put the full report in your final message.

## Done

- Write `docs/P1T_FIX9_REPORT.md` and `docs/P1T_FIX9_LEDGER.md`. The report holds:
  - the per-finding table;
  - sweep tables (m) and (n);
  - FIX8's corrected rows;
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
