# P1T-FIX5: close round-5 findings V5-01..V5-04, and sweep their four defect classes across the new code

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix5`
(branch `voice-fix5`: FIX4 1475fae, plus the round-5 order and receipts, b2d7918). Edits, core and `tc_signal`
builds, and GPU runs inside it are AUTHORIZED. You may also write scratch and mutant builds under `/tmp/p1t_fix5_*`.

## Why

David, verbatim: "If the Blind Review passes, I approve the merger."

Round 5 of the independent review (`docs/P1T_REV5_REPORT.md`, `artifacts/p1t_rev5/findings.json`) CLOSED all 18
earlier items (F01–F07, N1–N5, V3-01, V3-02, V4-01–V4-04). It rated R1 SAFE, R2 SOUND and R5 PASS. It said REJECT
because of one MAJOR finding, with R3 UNSOUND and R4 DEFECTS. The lead re-ran the GPU probes on 1475fae, and each
finding reproduces (`artifacts/p1t_rev5/lead/`).

- **V5-01 (MAJOR).** Interpreter exit overtakes a `tc_signal` producer deleter that is already running on another
  thread (`tc_signal/src/dlpack_operand.cpp:238-260,668-673,878-900`; `tc_signal/tc_signal/__init__.py:139-155`).
  - **Scenario:** a daemon thread's call refuses a versioned read-only output, and the Operand constructor runs the
    producer deleter while unwinding. The deleter releases the GIL. The atexit hook sees an empty queue and returns,
    and finalization begins. When the deleter re-takes the GIL, CPython 3.12 exits the thread with a forced unwind.
    That unwind hits a catch-all/noexcept boundary, and the process aborts (rc −6, `FATAL: exception not rethrown`).
  - **Deferred release has the same gap:** the entry is popped from the queue before its deleter runs, so the hook
    cannot see it.
  - **Lead receipts** (all on 1475fae):
    - `exit_gate.py full` (with and without a visible GPU) and `exit_gate.py gpu` (deferred) FAIL with child rc −6;
    - the `--control` variants (the worker joined before exit) PASS.
  - **The reviewer's fix:** an admission / closing / in-flight protocol for add-on releases, independent of the
    pending queue. At atexit the hook closes admission and waits, with the GIL released, until every admitted release
    on another thread has left. Use self-exclusion for a hook that runs inside its own callback. Late releases leak
    without entering Python. Cover the immediate-refusal, deferred, fallback and bound-drain release paths.
    - Core already solved the same problem for its own imports: the `ImportReleaseLifecycle` work in FIX2 and FIX3
      (V3-01) in `tensor_cuda/src/bindings.cpp`, and `docs/P1T_FIX3_REPORT.md`. Reuse its design, and say what you
      reuse. Do not link the add-on to core.
- **V5-02 (MINOR).** A one-shot `cudaEventQuery` failure in `drain_completed` is consumed and never recorded
  (`dlpack_operand.cpp:878-894`), so a later `synchronize()` silently succeeds. The same applies to sweep rows 47/50 in
  FIX4's table. The fix: record any query failure other than NotReady before consuming it, and report it once at the
  next `synchronize()`, while completion and release stay safe.
  - Lead receipt: `query_error_single_gpu.py` FAILS on 1475fae.
- **V5-03 (MINOR).** Both cuBLAS handle caches (`conv.cu:89-105`, `lstm.cu:112-128`) leak a newly created handle
  when the map insertion throws. The fix: keep the new handle owned until the insertion commits, and destroy it on
  every exception, checking the cleanup status without throwing during unwinding.
- **V5-04 (MINOR).** `fill_` narrows its block count (`tensor_cuda/src/kernels.cu:469`, which calls the baseline
  `nblk` helper and `fill_kernel` at 398-403). For 2^40+1 fp32 elements it launches one block, writes only 256
  elements and reports success. The fix: compute the block count in int64 in the `fill_` path and refuse an extent
  beyond the supported grid before any launch and before the revision is bumped. Refusal keeps the no-new-core-kernel
  plan; a grid-stride fill would be a new core kernel, which is forbidden.
  - Lead receipt: `fill_grid_single_gpu.py` FAILS on 1475fae.

**Every round has found new members of a defect CLASS** after the previous round fixed the old ones. So fixing these
four sites is not enough: sweep each finding's class across ALL of the P1-T code (`git diff 9df5bef..HEAD --
tensor_cuda tc_signal`), so that round 6 finds nothing more of these kinds. The four classes are (c)–(f), continuing
FIX4's (a) and (b):

- **(c) Callbacks during interpreter exit.** List every path, in the add-on AND in core's P1-T surface, that can invoke
  a producer deleter or any other callback that may take the GIL. Show that each one is admitted, and therefore waited
  for at exit, or else leaks without entering Python. List every `catch (...)` on such a path, and say what it does
  with a forced unwind (`abi::__forced_unwind`).
- **(d) Consumed but unrecorded errors.** List every site that consumes the CUDA error slot (`cudaGetLastError`)
  without raising. Each one must record the error for the next `synchronize()`, or be justified as benign (for
  example, NotReady), with the citation.
- **(e) Raw-resource exception safety.** List every CUDA or cuBLAS resource (handle, event, stream, allocation) in the
  P1-T code that is created before an operation that can throw. Each must be owned (RAII or try/catch) until its
  ownership is committed.
- **(f) Launch-geometry narrowing.** List every kernel launch in the P1-T code (`tc_signal` and core `fill_`). For
  each, show that its grid and block counts are computed without narrowing, and that every accepted extent is
  either covered by the launch or refused before it.

## Mechanism & scope

- **Measures:**
  - whether V5-01..V5-04 are closed: the round-5 probes fail on 1475fae and pass on your build;
  - whether sweeps (c)–(f) leave no site in any of those classes unhandled;
  - that nothing regresses: all round-1..5 probes, Voice's runner (T1–T6 plus T1R and T4S), and the combined session
    (52 baseline ids).
- **Mechanism doc:** `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix5/docs/P1T_REV5_REPORT.md`.
  Also `artifacts/p1t_rev5/findings.json`, the probe headers in `artifacts/p1t_rev5/probes/` (each states its run rule;
  the report's probe table has the commands), `artifacts/p1t_rev5/lead/` (the lead's GPU runs), and
  `docs/P1T_FIX3_REPORT.md` and `docs/P1T_FIX4_REPORT.md` (the core lifecycle design and FIX4's sweep).
- **Exercises:**
  - add-on: `tc_signal/src/*`, `tc_signal/tc_signal/*`, `tc_signal/tests/*`, `tc_signal/README.md`;
  - core: `tensor_cuda/src/kernels.cu` (the `fill_` path, and any P1-T site that sweeps (c)–(f) find),
    `tensor_cuda/src/bindings.cpp` (P1-T sites that the sweeps find), `tensor_cuda/tensor_cuda/__init__.py`,
    `tensor_cuda/tests/test_dlpack.py`;
  - every reviewer probe, run read-only.
- **Does not cover:**
  - pre-existing core code outside the P1-T surface (the shared `nblk` helper and the other engine kernels stay
    unchanged);
  - new features;
  - performance;
  - Voice's tests (READ-ONLY);
  - real two-GPU execution (this host has one GPU).
- **Required evidence:**
  - the per-finding table (fix at file:line, proving test, probe before and after);
  - the four sweep tables (c)–(f), every site with a verdict (OK / FIXED) and, for FIXED, a test or a stated reason
    why none is feasible;
  - FIX4's rows 47/50 corrected;
  - the regression logs;
  - the mutant table;
  - the diff stat against 1475fae;
  - both `.so` sha256s.
- **Operand census:**
  - **V5-01:** foreign-thread deleters of 0.1 s, 5.5 s and 12 s, through both the immediate-refusal path and the
    deferred path, at ordinary interpreter exit (no `os._exit`). Also:
    - a release that starts after the hook has closed admission (it leaks without entering Python, and is counted);
    - the hook invoked from inside a callback on the same thread (self-exclusion: no deadlock);
    - a deleter that raises an ordinary C++ exception (still contained once).
  - **V5-02:** one-shot query failures with nothing else pending, and with completed entries pending. The error is
    reported exactly once, and the next `synchronize()` succeeds.
  - **V5-03:** an injected insertion failure for each cache. The handle is destroyed, and the next call creates a
    working handle.
  - **V5-04:** `fill_` at the grid boundary (the largest accepted extent, and one more), plus a zero-size `fill_`.
    Use synthetic metadata through a shim, as the reviewer's probe does; never allocate terabytes.
- **Baseline isolation:**
  - "Before" means the 1475fae build, prebuilt in
    `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix4`; use it read-only.
  - "After" means your build, in fresh processes, under the GPU lock.
  - The combined session is compared with the 52-id baseline log named below.
  - Mutants are built outside the worktree.
- **Forbidden configuration:**
  - returning to ordinary finalization while an admitted add-on release is running;
  - entering Python (taking the GIL) from a release after admission has closed;
  - editing any reviewer probe or Voice test;
  - `os._exit`, `abort` or `std::terminate` as an error path;
  - throwing from destructors;
  - device-wide syncs on the per-call path;
  - fast-math, TF32, atomics or new core `__global__` kernels;
  - linking `tc_signal` to `_tensor_cuda`;
  - writing outside the target or `/tmp`.

## Required fixes

1. **V5-01.** Build the add-on release lifecycle: admission, a closing flag, an in-flight count and a self-exclusion
   depth.
   - The add-on atexit hook closes admission. It then waits, with the GIL released, until no admitted release is in
     flight on another thread.
   - The wait has no time bound. After a delay it logs a diagnostic line to stderr, then keeps logging at a bounded
     rate, as core does after V3-01.
   - Every release path goes through it: the immediate refusal while unwinding, the deferred drain, the fallback,
     the bound drain, and any capsule destructor path.
   - A release that arrives after admission has closed leaks without entering Python, and a stats key counts it.
   - Document the consequence: a producer deleter that never returns makes interpreter exit wait.
   - Say how the wait releases and re-takes the GIL, so that an admitted deleter that needs the GIL can finish.
2. **V5-02.** Record any query failure other than NotReady before consuming it. Report it once at the next
   `synchronize()`; FIX4's deferred-error routing can carry it.
3. **V5-03.** Own the handle through insertion in both caches.
4. **V5-04.** In the `fill_` path, compute the block count in int64 and refuse before any launch and before the
   revision bump. Cite the grid-dimension limit you use.
5. **The sweeps (c)–(f).** Fix every site that they find, with a test where a test is feasible.

## How to run

- **GPU:** every GPU command runs as `flock -w 900 /tmp/forge-gpu.lock timeout …`. Use the native driver only. You
  are the only seat running in this lane.
- **Builds:**
  - core: `cd <target>/tensor_cuda && ./build.sh 89`;
  - `tc_signal`: `cd <target>/tc_signal && PYBIND11_SOURCE_DIR=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src ./build.sh 89`.
- **Package path:** `PYTHONPATH=<target>/tensor_cuda/tensor_cuda:<target>/tensor_cuda:<target>/tc_signal`, plus
  `PYTHONDONTWRITEBYTECODE=1`.
- **Round-5 probes.** Run every probe in the round-5 report's probe table on both builds, as written. Use the GPU
  ones too: `exit_gate.py full|trace|gpu` with and without `--control`, and `full` both with `CUDA_VISIBLE_DEVICES=`
  and with the GPU visible.
  - Build the shims as the lead did: `gcc -shared -fPIC -O1 <probe>.c -o /tmp/p1t_rev5_<name>.so -ldl`, at the paths
    the probes hard-code. `query_error_cpu.py` and `addon_exit_cpu.py` need the trace module that
    `artifacts/p1t_rev5/probes/build_cpu.sh` builds.
  - Make path-only copies where a probe hard-codes the reviewer's paths, and report them.
  - Then run the round-4 probes, round 1's `addon_lifetime` (3 runs), `core_edges`, `core_shutdown` and
    `numerics_and_streams`, and round 3's `core_exit_overrun` (during and bounded).
- **Voice's runner, T1–T6 plus T1R and T4S.** Run it from `/mnt/ForgeRealm/Project-Voice`, which is READ-ONLY to you:
  ```
  cd /mnt/ForgeRealm/Project-Voice && flock -w 900 /tmp/forge-gpu.lock timeout 1200 env PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=<pp> python3 engine/tensor_requests/p1t_tests/run_p1t.py --adapter <target>/artifacts/p1t_final/tensorcuda_adapter.py \
    --items T1,T2,T3,T4,T5,T6,T1R,T4S --fixtures /mnt/ForgeRealm/Project-Voice/artifacts/p1t/fixtures \
    --tolerances engine/tensor_requests/p1t_tests/tolerances_r2.json --out <target>/artifacts/p1t_fix5/p1t_all.json
  ```
- **The combined session:**
  `cd <target> && flock -w 900 /tmp/forge-gpu.lock timeout 1500 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<pp> python3 -m pytest tensor_cuda/tests tc_signal/tests -q -p no:cacheprovider --ignore=tensor_cuda/tests/test_selector_accuracy.py`.
  Compare it with
  `python3 /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2/scripts/apa_mf2_2_g1_compare.py /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2/logs/apa_mf1_g1_baseline_unchanged.log <log>`.

## Gates

- **G1 (closed):** each of these fails on 1475fae and passes on your build:
  - `exit_gate.py full` (both conditions), `exit_gate.py trace` and `exit_gate.py gpu`;
  - `query_error_single_gpu.py` and `query_error_cpu.py`;
  - `handle_insert_oom_cpu.cpp`;
  - `fill_grid_cpu.cpp` and `fill_grid_single_gpu.py`.

  The `--control` variants pass on both builds. The new regression for each finding passes.
- **G2 (sweeps):** the tables (c)–(f) are complete. Every site is OK or FIXED, and every FIXED site has a test or a
  stated reason. FIX4's rows 47/50 are corrected.
- **G3 (no regression):**
  - every round-1..5 probe passes, or is INCONCLUSIVE only for lack of 2 GPUs;
  - `addon_lifetime` passes 3/3 runs with 0 incorrect;
  - `core_exit_overrun` during and bounded give rc 0;
  - Voice's runner (T1–T6 plus T1R and T4S) gives PASS 249/249;
  - the combined session shows exactly the 52 baseline ids.
- **G4 (teeth):** build these mutants outside the worktree. Each must fail its new test:
  - (a) the V5-01 admission wait removed;
  - (b) the V5-02 record removed;
  - (c) the V5-03 ownership removed;
  - (d) the V5-04 refusal removed.
- **G5 (scope):**
  - the diff against b2d7918 touches only the Exercises paths, plus `docs/P1T_FIX5_*.md`, `artifacts/p1t_fix5/**`
    and `logs/**`;
  - the probe files are byte-identical;
  - no new core `__global__`.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. Your order file is
  immutable.
- NO git writes (read-only git is fine). NO subagents. No background waits: run every gate synchronously to
  completion.
- No deletes, anywhere. Never kill a process you did not start.
- These are READ-ONLY: production `/mnt/ForgeRealm/Project-Tensor` (outside `Fork/`), `/mnt/ForgeRealm/Project-Voice`,
  and every other fork or worktree.
- RED honesty: a failed gate is a result. Report it with receipts; never weaken a gate, probe, test or tolerance.
- Prior Art Directive (AGENTS.md): annotate at the code site, in the ledger and in the report. Known prior art:
  - the CPython 3.12 `Py_FinalizeEx` ordering, and thread exit when the GIL is taken during finalization
    (`PyEval_RestoreThread`, `ceval_gil.c`);
  - glibc/libstdc++ forced unwinding (`abi::__forced_unwind`);
  - core's own `ImportReleaseLifecycle` (FIX2/FIX3);
  - RCU-style grace periods;
  - scope guards / RAII (Alexandrescu and Marginean, 2000);
  - CUDA 12.6 grid-dimension limits (Programming Guide, compute-capability table);
  - sticky versus non-sticky CUDA errors.
- If the AFTERIMAGE memory hook denies a Write, read what it shows you and retry. If a report Write is refused
  outright, put the full report in your final message.

## Done

- Write `docs/P1T_FIX5_REPORT.md` (the per-finding table, the four sweep tables, the probe before/after table, gate
  table, mutant table and prior art) and `docs/P1T_FIX5_LEDGER.md`.
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
