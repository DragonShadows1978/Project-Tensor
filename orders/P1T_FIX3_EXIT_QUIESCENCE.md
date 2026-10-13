# P1T-FIX3: close round-3 review findings V3-01 (exit quiescence) and V3-02 (aggregate byte accounting)

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix3`
(branch `voice-fix3`: `voice-fix2` 5f9f955 plus the round-3 receipts). Edits, core and `tc_signal` builds, and GPU
runs inside it are AUTHORIZED. You may also write scratch and mutant builds under `/tmp/p1t_fix3_*`.

## Why

David, verbatim: "If the Blind Review passes, I approve the merger."

Round 3 of the independent review (`docs/P1T_REV3_REPORT.md`, `artifacts/p1t_rev3/findings.json`) said REJECT on one
MAJOR and one MINOR finding. Everything else is closed. F03, F04 and N5 are "PARTLY closed" only through these two
findings.

- **V3-01 (MAJOR).** The exit hook's bounded wait (`_DLPACK_SHUTDOWN_WAIT_S = 5.0`; core `bindings.cpp:653-682`,
  `tensor_cuda/__init__.py:181-199`) returns to ordinary finalization while an admitted, Python-dependent release is
  still running. A 5.5 s foreign-thread deleter then re-acquires the GIL during finalization and the process aborts
  (rc −6).
  - The reviewer's fix: keep finalization behind quiescence for every admitted release. A timeout may log diagnostics
    but must not return to ordinary finalization while admitted releases are running. Add a normal-exit overrun
    regression that does not use `os._exit`.
  - CPU reproduction: `artifacts/p1t_rev3/probes/lifecycle_timeout.py` with `lifecycle_slice.cpp`.
  - GPU probe: `artifacts/p1t_rev3/probes/core_exit_overrun.py`.
- **V3-02 (MINOR).** The aggregate deferred-release byte totals use unchecked signed additions
  (`tc_signal/src/dlpack_operand.cpp:602,731,774`). The reviewer's fix:
  - check the aggregate operand bytes before the first launch (ValueError);
  - make the queue accounting and the bound comparison overflow-safe without throwing in the destructor (headroom
    check or drain-before-add);
  - no throwing `ck_add` in the destructor.
  - Probe: `artifacts/p1t_rev3/probes/pending_bytes.cpp`.

## Mechanism & scope

- **Measures:** whether V3-01 and V3-02 are closed:
  - the reviewer's probes fail on 5f9f955 and pass on your build;
  - each finding gets a new regression test that fails on its mutant;
  - nothing regresses: all round-1, round-2 and round-3 probes, Voice's runner T1–T6, and the combined core plus
    `tc_signal` session (52 baseline ids).
- **Mechanism doc:** `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix3/docs/P1T_REV3_REPORT.md`.
  Also `artifacts/p1t_rev3/findings.json`, `artifacts/p1t_rev3/probes/README.md` (how to run every probe; several are
  marked "LEAD GPU ONLY", and you have the GPU) and `docs/P1T_FIX2_REPORT.md` (the N5 design you are amending).
- **Exercises:**
  - core: `tensor_cuda/src/bindings.cpp` (the import-release lifecycle only), `tensor_cuda/tensor_cuda/__init__.py`
    (the atexit hook and knob), `tensor_cuda/tests/test_dlpack.py`;
  - add-on: `tc_signal/src/dlpack_operand.{h,cpp}`, `tc_signal/src/*_bindings.cpp` / `bindings.cpp` (the pre-launch
    aggregate check only), `tc_signal/tests/*`, `tc_signal/README.md`;
  - every reviewer probe, run read-only.
- **Does not cover:**
  - anything else in core or `tc_signal`;
  - new features;
  - performance;
  - Voice's tests (READ-ONLY).
- **Required evidence:**
  - the per-finding table (fix at file:line, proving test, probe before and after);
  - the complete before/after table of every probe in `artifacts/p1t_rev3/probes/README.md`, GPU ones included;
  - the regression logs;
  - the mutant table;
  - the diff stat against 5f9f955;
  - both `.so` sha256s.
- **Operand census:**
  - **V3-01:** foreign-thread deleters of 0.1 s, 5.5 s and 12 s across the exit wait, in ordinary interpreter exit (no
    `os._exit`), plus the "after closing" and "never admitted" cases.
  - **V3-02:** two disjoint operands of `2**61 − 1` fp32 elements (metadata only); queue totals near INT64_MAX; and
    the bound comparison at the boundary.
- **Baseline isolation:**
  - "Before" means the 5f9f955 build, prebuilt in
    `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix2`; use it read-only.
  - "After" means your build, in fresh processes, under the GPU lock.
  - Core's suite is compared with the 52-id baseline log named below.
  - Mutants are built outside the worktree.
- **Forbidden configuration:**
  - returning to ordinary finalization while admitted Python-dependent releases are running;
  - `os._exit` / `_exit` / `abort` in product code;
  - a throwing check in a destructor;
  - editing any reviewer probe or Voice test;
  - fast-math, TF32, atomics or new core kernels;
  - writing outside the target or `/tmp`.

## Required fixes

1. **V3-01.** The exit hook waits until `in_flight == 0`, with no bound on ordinary finalization.
   - After `_DLPACK_SHUTDOWN_WAIT_S` it logs one diagnostic line to stderr (count in flight, seconds waited), then
     keeps logging at a bounded rate while it continues to wait. It never returns early.
   - Admission stays closed, so no new release is admitted.
   - Document the consequence in a docstring and in your report: a producer deleter that never returns makes interpreter
     exit wait. That is the producer's contract violation, and the honest outcome, compared with an abort.
   - Say whether the wait drops and re-takes the GIL correctly, so that an admitted deleter which needs the GIL can
     finish.
   - Add a regression: a 5.5 s foreign-thread deleter across the wait, ordinary exit, which must return rc 0 with the
     deleter completed. Use a lower knob value in the test if you like, as long as the deleter outlives the knob.
2. **V3-02.**
   - Before the first launch of each `tc_signal` call, check the aggregate operand bytes with overflow-safe addition
     (ValueError on overflow).
   - In the retainer destructor and `note_push`/`note_pop`, keep the queue byte accounting overflow-safe without
     throwing: saturate, or drain before an addition would overflow. Choose one and justify it.
   - The bound comparison must stay correct at the boundary.

## How to run

- **GPU:** every GPU command runs as `flock -w 900 /tmp/forge-gpu.lock timeout …`. Use the native driver only. You
  are the only seat running.
- **Builds:**
  - core: `cd <target>/tensor_cuda && ./build.sh 89`;
  - `tc_signal`: `cd <target>/tc_signal && PYBIND11_SOURCE_DIR=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src ./build.sh 89`.
- **Package path:** `PYTHONPATH=<target>/tensor_cuda/tensor_cuda:<target>/tensor_cuda:<target>/tc_signal`, plus
  `PYTHONDONTWRITEBYTECODE=1`. Probes that need the native helper use `/tmp/p1t_rev_native` (round 1) as their
  READMEs say.
- **Probes:** run every probe in `artifacts/p1t_rev3/probes/README.md` on both builds, as written. Rebuild CPU probes
  from their sources into `/tmp/p1t_fix3_probes/`. Then re-run round 1's `addon_lifetime` (3 runs), `core_edges`,
  `core_shutdown` and `numerics_and_streams`.
- **Voice's runner, T1–T6.** Run it from `/mnt/ForgeRealm/Project-Voice`, which is READ-ONLY to you:
  ```
  cd /mnt/ForgeRealm/Project-Voice && flock -w 900 /tmp/forge-gpu.lock timeout 1200 env PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=<pp> python3 engine/tensor_requests/p1t_tests/run_p1t.py --adapter <target>/artifacts/p1t_final/tensorcuda_adapter.py \
    --items T1,T2,T3,T4,T5,T6 --fixtures /mnt/ForgeRealm/Project-Voice/artifacts/p1t/fixtures \
    --tolerances engine/tensor_requests/p1t_tests/tolerances_r2.json --out <target>/artifacts/p1t_fix3/p1t_T1toT6.json
  ```
- **The combined session:**
  `cd <target> && flock -w 900 /tmp/forge-gpu.lock timeout 1500 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<pp> python3 -m pytest tensor_cuda/tests tc_signal/tests -q -p no:cacheprovider --ignore=tensor_cuda/tests/test_selector_accuracy.py`.
  Compare it with
  `python3 /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2/scripts/apa_mf2_2_g1_compare.py /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2/logs/apa_mf1_g1_baseline_unchanged.log <log>`.

## Gates

- **G1 (closed):**
  - `lifecycle_timeout.py` and `core_exit_overrun.py` pass on your build and fail on 5f9f955;
  - `pending_bytes.cpp` is clean under UBSan on your build and reports on 5f9f955;
  - the new regression for each finding passes.
- **G2 (no regression):**
  - every round-3 probe passes, or is reported INCONCLUSIVE only where it needs 2 GPUs;
  - round 1's `addon_lifetime` passes 3/3 runs with 0 incorrect, and the other round-1 probes pass;
  - Voice's runner on T1–T6 gives PASS 236/236;
  - the combined session shows exactly the 52 baseline ids.
- **G3 (teeth):** build these mutants outside the worktree. Each must fail its new test:
  - (a) the bounded return restored (the old 5 s return);
  - (b) the aggregate pre-launch check removed.
- **G4 (scope):**
  - the diff against 5f9f955 touches only the Exercises paths, plus `docs/P1T_FIX3_*.md`, `artifacts/p1t_fix3/**`
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
  - the CPython 3.12 `Py_FinalizeEx` ordering, and `threading._shutdown`, which joins non-daemon threads without a
    bound;
  - quiescence/RCU-style grace periods;
  - checked and saturating arithmetic (GCC builtins, C23 `ckd_*`);
  - UBSan.
- If the AFTERIMAGE memory hook denies a Write, read what it shows you and retry. If a report Write is refused
  outright, put the full report in your final message.

## Done

- Write `docs/P1T_FIX3_REPORT.md` (the per-finding table, the full probe before/after table, gate table, mutant table
  and prior art) and `docs/P1T_FIX3_LEDGER.md`.
- Your final message must contain, verbatim:
  - the per-finding table;
  - the probe before/after table;
  - the gate table;
  - Voice's verdict line;
  - the compare output;
  - the mutant table;
  - the diff stat;
  - both `.so` sha256s;
  - every deviation, with its reason.
