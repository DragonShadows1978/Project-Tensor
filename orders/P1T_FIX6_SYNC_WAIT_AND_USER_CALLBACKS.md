# P1T-FIX6: close round-6 findings V6-01..V6-03, and sweep two more defect classes across the new code

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix6`
(branch `voice-fix6`: FIX5 acd42fa, plus the round-6 order and partial receipts, 123610f). Edits, core and `tc_signal`
builds, and GPU runs inside it are AUTHORIZED. You may also write scratch and mutant builds under `/tmp/p1t_fix6_*`.

## Why

David, verbatim: "If the Blind Review passes, I approve the merger."

The provider's content filter stopped round 6 of the independent review early, before its closure table. Its
early-written findings (`docs/P1T_REV6_REPORT.md`, `artifacts/p1t_rev6/findings.json`) already record REJECT:
R2 UNSOUND and R3 UNSOUND, while R1 is SAFE, R4 CORRECT and R5 PASS. The lead reproduced both MAJOR findings on the GPU
(`artifacts/p1t_rev6/lead/`).

- **V6-01 (MAJOR). FIX5's own D1 change deadlocks.** `tc_signal.synchronize()` now waits for add-on releases running
  on other threads (`tc_signal/src/dlpack_operand.cpp:1284-1289; 291-305; 249-264`).
  - **Scenario:** thread A holds an application lock (say, a producer allocator lock) and calls `synchronize()`.
    Thread B is releasing a refused operand, and its finite producer deleter needs the same lock. A waits for B, and B
    waits for A. Releasing the GIL does not release the application lock.
  - **Lead receipt:** `sync_lock_inversion_cpu.py` on acd42fa gives rc 1 (a rescue thread was needed).
  - **The reviewer's fix:** keep the close-and-wait in the real exit hook, and remove the wait for other threads'
    callbacks from ordinary `synchronize()`. If callback completion is wanted as a separate API, it needs an explicit
    no-locks-held contract and a pre-call generation boundary (see `sync_late_release_cpu.py`). Keep admission and the
    safe leak during exit.
- **V6-02 (MAJOR). The core exit hook does not wait for user Python run by `__dlpack__` argument conversion**
  (`tensor_cuda/src/bindings.cpp:483-515; 799-870`). This was FIX5's residual R3.
  - **Scenario:** a daemon thread calls `t.__dlpack__(copy=Flag())` or `t.__dlpack__(max_version=(Index(), 3))`. The
    user `__bool__` / `__index__` sleeps with the GIL released. The core hook sees nothing in flight and returns, and
    finalization begins. When the callback re-takes the GIL, pybind11 objects are still live on the native stack, and
    the process segfaults.
  - **Lead receipts:** `core_cast_exit_gpu.py bool|index` gives rc 139 on acd42fa and also on 1475fae, so it predates
    FIX5. The `--control` runs give rc 0.
  - **The reviewer's fix (either):**
    - (i) admit the whole export argument-conversion section into the core exit lifecycle, and close and wait for it;
    - (ii) refuse, before invoking them, any arguments that would need arbitrary Python conversion.

    Cover sequence length and item methods as well as `__index__` / `__bool__`, and keep the explicit GIL restore
    and self-exclusion.
  - **Constraint:** interop must keep working with real consumers. That means `torch.from_dlpack` on a TensorCUDA
    tensor, at least: torch passes plain Python ints for `stream` and tuples of ints for `max_version`. Voice's adapter
    (`artifacts/p1t_final/tensorcuda_adapter.py`) must stay unchanged.
  - Pick (i) or (ii), justify the choice, and say what each accepted argument type is.
- **V6-03 (MINOR). Transient `synchronize()` failures disappear when building their diagnostic hits `bad_alloc`**
  (`dlpack_operand.cpp:986-1025; 1298-1304; 1314-1355`). The fix: before any message allocation, persist every
  current-drain failure with allocation-free bookkeeping, and clear it only after a diagnostic has been raised.
  - Reviewer receipts: `release_trace_cpu.py diagnostic_oom_{query,sync,device}` give rc 1.

**Every round has found new members of a defect class, sometimes one created by the previous fix** (V6-01 came from
FIX5's D1). So fixing these sites is not enough. Sweep these two classes across ALL of the P1-T code
(`git diff 9df5bef..HEAD -- tensor_cuda tc_signal`), and re-check FIX5's sweep (c) in their light:

- **(g) User Python run from inside a native frame.** List every pybind11 binding in the P1-T surface:
  - core: `__dlpack__`, `__dlpack_device__`, the import entry points, `data_ptr`, `fill_`, `mark_modified`,
    `revision`, `empty`, and the `_dlpack_*` helpers;
  - add-on: every op, plan and run binding, `set_allocator`, `synchronize`, and the stats.

  For each, list every argument conversion or call that can run user Python: `__index__`, `__bool__`, `__float__`,
  `__len__`, `__getitem__`, `__iter__`, `__dlpack__`, `__dlpack_device__`, allocator callables, and decrefs that can
  run `__del__`. Each must be admitted (and waited for at exit), refused before it is invoked, or shown harmless
  (no live `py::object` or C++ cleanup on the native stack across a GIL re-take). Remember that pybind11's typed
  argument casters run conversions in the dispatcher, before the function body.
- **(h) Waits that can deadlock against caller-held locks.** List every wait in the P1-T code that blocks on another
  thread (condition-variable waits, `synchronize()`, the exit hooks). Say which caller-held locks a user callback on
  the other thread could need. Ordinary API calls must not wait for another thread's user callback; only the exit
  hooks may, and their wait is documented.

## Mechanism & scope

- **Measures:**
  - whether V6-01..V6-03 are closed: the round-6 probes fail on acd42fa and pass on your build;
  - whether sweeps (g) and (h) leave no site unhandled;
  - that nothing regresses: all round-1..5 probes (see the exit_gate note below), Voice's runner (T1–T6 plus T1R and
    T4S), and the combined session (52 baseline ids).
- **Mechanism doc:** `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix6/docs/P1T_REV6_REPORT.md`.
  Also:
  - `artifacts/p1t_rev6/findings.json`;
  - the probe headers in `artifacts/p1t_rev6/probes/`; `run_cpu_checks.py` and `build_host_probes.sh` have the
    CPU commands and build;
  - `artifacts/p1t_rev6/lead/` (the lead's GPU runs);
  - `docs/P1T_FIX5_REPORT.md` (the add-on lifecycle, D1 and sweep (c)) and `docs/P1T_FIX3_REPORT.md` (core's
    lifecycle).
- **Exercises:**
  - add-on: `tc_signal/src/*`, `tc_signal/tc_signal/*`, `tc_signal/tests/*`, `tc_signal/README.md`;
  - core: `tensor_cuda/src/bindings.cpp` (`__dlpack__` and the other P1-T bindings, and the import/export lifecycle),
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
  - sweep tables (g) and (h), every site with a verdict (OK / FIXED) and, for FIXED, a test or a stated reason why
    none is feasible;
  - FIX5's sweep (c) rows that (g) changes, corrected;
  - the regression logs;
  - the mutant table;
  - the diff stat against acd42fa;
  - both `.so` sha256s.
- **Operand census:**
  - **V6-01:**
    - the reviewer's lock inversion;
    - `synchronize()` called while a release is running on another thread, with and without a lock held;
    - the exit hook still waiting for an admitted release on another thread (rerun the round-5 `exit_gate full` and
      `gpu` probes);
    - a release that starts after `synchronize()` has begun.
  - **V6-02:**
    - `__bool__`, `__index__`, sequence `__len__` / `__getitem__` and `__iter__` in each `__dlpack__` argument
      (`stream`, `max_version`, `dl_device`, `copy`), on a daemon thread at ordinary exit;
    - plain Python ints, bools, None and tuples, which are accepted;
    - torch's real `torch.from_dlpack` consumer path.
  - **V6-03:** message-allocation failure after a transient query, sync or device failure. The error survives for the
    next `synchronize()` and is reported exactly once.
- **Baseline isolation:**
  - "Before" means the acd42fa build, prebuilt in
    `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix5`; use it read-only.
  - "After" means your build, in fresh processes, under the GPU lock.
  - The combined session is compared with the 52-id baseline log named below.
  - Mutants are built outside the worktree.
- **Forbidden configuration:**
  - any ordinary (non-exit) API call that waits for another thread's user callback;
  - returning to ordinary finalization while an admitted release or admitted user-Python section is running;
  - editing any reviewer probe or Voice test, or changing Voice's adapter;
  - `os._exit`, `abort` or `std::terminate` as an error path;
  - throwing from destructors;
  - device-wide syncs on the per-call path;
  - fast-math, TF32, atomics or new core `__global__` kernels;
  - linking `tc_signal` to `_tensor_cuda`;
  - writing outside the target or `/tmp`.

## Required fixes

1. **V6-01.** Ordinary `synchronize()` must not wait for other threads' callbacks; only the exit path may.
   - The round-5 `exit_gate.py trace` probe models the exit hook as `synchronize()`. If your fix makes it fail as
     written, report that.
   - Also run a path-only copy that calls the trace module's real exit entry point, and record it as a deviation. The
     round-6 reviewer prescribes exactly this design.
   - The `full` and `gpu` exit_gate modes use the real package hook, and must still pass as written.
2. **V6-02.** Option (i) or (ii), as above, with real-torch interop proven by a test.
3. **V6-03.** Persist current-drain failures before building any message.
4. **Sweeps (g) and (h).** Fix every site that they find, with a test where one is feasible.

## How to run

- **GPU:** every GPU command runs as `flock -w 900 /tmp/forge-gpu.lock timeout …`. Use the native driver only. You
  are the only seat running in this lane.
- **Builds:**
  - core: `cd <target>/tensor_cuda && ./build.sh 89`;
  - `tc_signal`: `cd <target>/tc_signal && PYBIND11_SOURCE_DIR=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src ./build.sh 89`.
- **Package path:** `PYTHONPATH=<target>/tensor_cuda/tensor_cuda:<target>/tensor_cuda:<target>/tc_signal`, plus
  `PYTHONDONTWRITEBYTECODE=1`.
- **Round-6 probes.** Run every probe in `artifacts/p1t_rev6/probes/` on both builds:
  - `sync_lock_inversion_cpu.py`, with and without a visible GPU;
  - `core_cast_exit_gpu.py bool|index`, with and without `--control`;
  - the CPU checks from `run_cpu_checks.py`, built by `build_host_probes.sh`. Use path-only copies where a probe
    hard-codes the reviewer's paths, and report them;
  - `sync_late_release_cpu.py`.

  `core_host_cpu.py` and `release_trace.cpp` embed slices of product code. Regenerate them from each arm's own tree
  with a self-check, as FIX4 and FIX5 did, and report it.
- **Earlier probes.** Then run:
  - the round-5 probes (exit_gate `full` both conditions, `trace`, `gpu`, with `--control`; `query_error_single_gpu`;
    `fill_grid_single_gpu`; shims built with `gcc -shared -fPIC -O1 <probe>.c -o /tmp/p1t_rev5_<name>.so -ldl`);
  - the round-4 probes;
  - round 1's `addon_lifetime` (3 runs), `core_edges`, `core_shutdown` and `numerics_and_streams`;
  - round 3's `core_exit_overrun` (during and bounded).
- **Voice's runner, T1–T6 plus T1R and T4S.** Run it from `/mnt/ForgeRealm/Project-Voice`, which is READ-ONLY to you:
  ```
  cd /mnt/ForgeRealm/Project-Voice && flock -w 900 /tmp/forge-gpu.lock timeout 1200 env PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=<pp> python3 engine/tensor_requests/p1t_tests/run_p1t.py --adapter <target>/artifacts/p1t_final/tensorcuda_adapter.py \
    --items T1,T2,T3,T4,T5,T6,T1R,T4S --fixtures /mnt/ForgeRealm/Project-Voice/artifacts/p1t/fixtures \
    --tolerances engine/tensor_requests/p1t_tests/tolerances_r2.json --out <target>/artifacts/p1t_fix6/p1t_all.json
  ```
- **The combined session.** It takes about 10 minutes; run it to completion before you read its result.
  `cd <target> && flock -w 900 /tmp/forge-gpu.lock timeout 1500 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<pp> python3 -m pytest tensor_cuda/tests tc_signal/tests -q -p no:cacheprovider --ignore=tensor_cuda/tests/test_selector_accuracy.py`.
  Compare it with
  `python3 /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2/scripts/apa_mf2_2_g1_compare.py /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2/logs/apa_mf1_g1_baseline_unchanged.log <log>`.

## Gates

- **G1 (closed):** each of these fails on acd42fa and passes on your build:
  - `sync_lock_inversion_cpu.py`, in both conditions;
  - `core_cast_exit_gpu.py bool` and `index`;
  - `core_host_cpu.py casts_exit bool` and `index`, regenerated;
  - `release_trace_cpu.py diagnostic_oom_query`, `_sync` and `_device`, regenerated.

  `sync_late_release_cpu.py` passes on your build. The `--control` variants pass on both builds. The new regression
  for each finding passes.
- **G2 (sweeps):** tables (g) and (h) are complete. Every site is OK or FIXED, and every FIXED site has a test or a
  stated reason. FIX5's affected sweep-(c) rows are corrected.
- **G3 (no regression):**
  - every round-1..5 probe passes, or is INCONCLUSIVE only for lack of 2 GPUs. The round-5 `exit_gate trace` is
    handled as in Required fix 1;
  - `addon_lifetime` passes 3/3 runs with 0 incorrect;
  - `core_exit_overrun` during and bounded give rc 0;
  - Voice's runner (T1–T6 plus T1R and T4S) gives PASS 249/249;
  - the combined session shows exactly the 52 baseline ids;
  - real-torch DLPack interop passes.
- **G4 (teeth):** build these mutants outside the worktree. Each must fail its new test:
  - (a) the wait for other threads restored in ordinary `synchronize()`;
  - (b) the V6-02 admission or refusal removed;
  - (c) the V6-03 persistence removed.
- **G5 (scope):**
  - the diff against 123610f touches only the Exercises paths, plus `docs/P1T_FIX6_*.md`, `artifacts/p1t_fix6/**`
    and `logs/**`;
  - the probe files are byte-identical;
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
  - Coffman et al. (1971) deadlock conditions, and lock-ordering discipline;
  - the CPython 3.12 finalization and GIL rules (`Py_FinalizeEx`, `PyEval_RestoreThread`);
  - pybind11's argument casters and dispatcher;
  - the DLPack Python spec and the array API's `__dlpack__` argument types;
  - PyTorch's `Tensor.__dlpack__` argument handling (read the installed torch);
  - RCU grace periods (`synchronize_rcu` waits only for readers that existed before it);
  - allocation-free error bookkeeping (as in the SQLite OOM-testing discipline).
- If the AFTERIMAGE memory hook denies a Write, read what it shows you and retry. If a report Write is refused
  outright, put the full report in your final message.

## Done

- Write `docs/P1T_FIX6_REPORT.md` (the per-finding table, sweep tables (g) and (h), the probe before/after table, gate
  table, mutant table and prior art) and `docs/P1T_FIX6_LEDGER.md`.
- Your final message must contain, verbatim:
  - the per-finding table;
  - each sweep table, with a summary count;
  - the probe before/after table;
  - the gate table;
  - Voice's verdict line;
  - the compare output;
  - the mutant table;
  - the diff stat;
  - both `.so` sha256s;
  - every deviation, with its reason.
