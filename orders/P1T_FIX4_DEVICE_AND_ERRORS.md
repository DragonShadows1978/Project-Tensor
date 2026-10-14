# P1T-FIX4: close round-4 findings V4-01..V4-04, and sweep their defect class across the new code

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix4`
(branch `voice-fix4`: `voice-fix3` 62c8d9f plus the round-4 receipts). Edits, core and `tc_signal` builds, and GPU
runs inside it are AUTHORIZED. You may also write scratch and mutant builds under `/tmp/p1t_fix4_*`.

## Why

David, verbatim: "If the Blind Review passes, I approve the merger."

Round 4 of the independent review (`docs/P1T_REV4_REPORT.md`, `artifacts/p1t_rev4/findings.json`) CLOSED all 14
earlier items, rated R4 (kernels) CORRECT and R5 (separation and build) PASS. It still said REJECT, because of one MAJOR
finding:

- **V4-01 (MAJOR).** The core export fence is on the wrong device (`tensor_cuda/src/bindings.cpp:376-400,446`).
  `__dlpack__(stream=…)` creates and records its event on the CURRENT device, not the tensor's device. With device 1
  current, a device-0 consumer waits on an unrelated event. The fix: guard event creation and recording with the
  tensor's device and restore the caller's device, or refuse a current-device mismatch.
- **V4-03 (MINOR).** Core import ignores a `cudaGetDevice` failure (`bindings.cpp:817-823,888-901`). `cur` stays 0, so
  a device-0 capsule is accepted and renamed `used_dltensor`. The fix: check the return value and raise a named
  RuntimeError before the Storage is constructed and before the capsule is renamed.
- **V4-04 (MINOR).** `tc_signal.synchronize()` ignores an error already present in the calling thread's CUDA error
  slot (`tc_signal/src/dlpack_operand.cpp:510-516,688-704,858-861`). The fix: capture the existing error on entry,
  drain safely, and raise a named error even if every stream sync succeeds.
- **V4-02 (NIT).** The `from_dlpack` docstring (`tensor_cuda/tensor_cuda/__init__.py:107-110`) still describes the
  removed bounded exit wait.

Each round has found new members of two defect CLASSES: implicit current-device assumptions (F05, V4-01) and unchecked
or ignored CUDA return and error state (N1, V4-03, V4-04). So fixing the four sites is not enough. Sweep both classes
across ALL new code, so the next review finds nothing more of either kind.

## Mechanism & scope

- **Measures:**
  - whether V4-01..V4-04 are closed: the reviewer's probes fail on 62c8d9f and pass on your build, or are
    INCONCLUSIVE only for lack of a second GPU (with a single-device test that proves the guard);
  - whether a sweep of every CUDA runtime/cuBLAS call in the new code leaves none with an unguarded device assumption
    or an unchecked or ignored return;
  - that nothing regresses: all round-1..4 probes, Voice's runner T1–T6, and the combined session (52 baseline ids).
- **Mechanism doc:** `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix4/docs/P1T_REV4_REPORT.md`.
  Also `artifacts/p1t_rev4/findings.json`, `artifacts/p1t_rev4/probes/README.md` (how to run each probe) and
  `docs/P1T_FIX3_REPORT.md`.
- **Exercises:**
  - core: `tensor_cuda/src/bindings.cpp` (the DLPack/empty/fill_/mark_modified surface and the import-release
    lifecycle), `tensor_cuda/src/kernels.cu` (the external Storage path and `fill_`), `tensor_cuda/tensor_cuda/__init__.py`,
    `tensor_cuda/tests/test_dlpack.py`;
  - add-on: `tc_signal/src/*`, `tc_signal/tc_signal/*`, `tc_signal/tests/*`, `tc_signal/README.md`;
  - every reviewer probe, run read-only.
- **Does not cover:**
  - pre-existing core code outside the P1-T surface;
  - new features;
  - performance;
  - Voice's tests (READ-ONLY);
  - real two-GPU execution (this host has one GPU).
- **Required evidence:**
  - the per-finding table (fix at file:line, proving test, probe before and after);
  - the SWEEP TABLE: every CUDA runtime/cuBLAS call site in the P1-T diff (`git diff 9df5bef..HEAD -- tensor_cuda
    tc_signal`), with the device it runs on, how that device is established (guard, operand device, or proven
    current), how its return is handled, and your verdict (OK / FIXED);
  - the regression logs;
  - the mutant table;
  - the diff stat against 62c8d9f;
  - both `.so` sha256s.
- **Operand census:**
  - **V4-01:** export with the current device equal to and different from the tensor's device. With one GPU, simulate
    the mismatch with an LD_PRELOAD or ctypes shim, as FIX1 did for F05, and assert the guard switches and restores.
  - **V4-03:** import with `CUDA_VISIBLE_DEVICES=""` and with a sticky prior error.
  - **V4-04:** `synchronize()` after a synchronous launch-configuration error, with nothing pending and with entries
    pending.
  - **Sweep:** every call site found.
- **Baseline isolation:**
  - "Before" means the 62c8d9f build, prebuilt in
    `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix3`; use it read-only.
  - "After" means your build, in fresh processes, under the GPU lock.
  - The combined session is compared with the 52-id baseline log named below.
  - Mutants are built outside the worktree.
- **Forbidden configuration:**
  - editing any reviewer probe or Voice test;
  - `os._exit`, `abort` or `std::terminate` as an error path;
  - throwing from destructors;
  - device-wide syncs on the per-call path;
  - fast-math, TF32, atomics or new core `__global__` kernels;
  - writing outside the target or `/tmp`.

## Required fixes

1. **V4-01.** Export guards event creation and recording with the tensor's device (switch, record, restore; errors
   raised with the device restored). Also consider whether `cudaStreamWaitEvent` on the consumer stream needs that
   stream's device. Per CUDA semantics a cross-device event wait is legal; cite the docs.
2. **V4-03.** Import checks `cudaGetDevice`, and every other CUDA query, before any ownership transfer: no Storage
   is constructed, the capsule is not renamed, and a named RuntimeError is raised.
3. **V4-04.** `synchronize()` captures the calling thread's existing CUDA error on entry. It still releases what can
   safely be released, or leaks what cannot, and then raises a named error that includes the pre-existing one. Decide
   whether to clear or peek the error slot, and justify the choice against CUDA's sticky versus non-sticky error
   semantics.
4. **V4-02.** Update the docstring.
5. **The sweep.** Fix every site in the two classes that the sweep finds, with a test where a test is feasible.

## How to run

- **GPU:** every GPU command runs as `flock -w 900 /tmp/forge-gpu.lock timeout …`. Use the native driver only. You
  are the only seat running.
- **Builds:**
  - core: `cd <target>/tensor_cuda && ./build.sh 89`;
  - `tc_signal`: `cd <target>/tc_signal && PYBIND11_SOURCE_DIR=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src ./build.sh 89`.
- **Package path:** `PYTHONPATH=<target>/tensor_cuda/tensor_cuda:<target>/tensor_cuda:<target>/tc_signal`, plus
  `PYTHONDONTWRITEBYTECODE=1`. Probes that need the native helper use `/tmp/p1t_rev_native`, as their READMEs say.
- **Probes:** run every probe in `artifacts/p1t_rev4/probes/README.md` on both builds, as written. Make path-only
  copies where a probe hard-codes the reviewer's paths, and report them. Then run round 1's `addon_lifetime` (3 runs),
  `core_edges`, `core_shutdown` and `numerics_and_streams`, and round 3's `core_exit_overrun` (during and bounded).
- **Voice's runner, T1–T6.** Run it from `/mnt/ForgeRealm/Project-Voice`, which is READ-ONLY to you:
  ```
  cd /mnt/ForgeRealm/Project-Voice && flock -w 900 /tmp/forge-gpu.lock timeout 1200 env PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=<pp> python3 engine/tensor_requests/p1t_tests/run_p1t.py --adapter <target>/artifacts/p1t_final/tensorcuda_adapter.py \
    --items T1,T2,T3,T4,T5,T6 --fixtures /mnt/ForgeRealm/Project-Voice/artifacts/p1t/fixtures \
    --tolerances engine/tensor_requests/p1t_tests/tolerances_r2.json --out <target>/artifacts/p1t_fix4/p1t_T1toT6.json
  ```
- **The combined session:**
  `cd <target> && flock -w 900 /tmp/forge-gpu.lock timeout 1500 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<pp> python3 -m pytest tensor_cuda/tests tc_signal/tests -q -p no:cacheprovider --ignore=tensor_cuda/tests/test_selector_accuracy.py`.
  Compare it with
  `python3 /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2/scripts/apa_mf2_2_g1_compare.py /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2/logs/apa_mf1_g1_baseline_unchanged.log <log>`.

## Gates

- **G1 (closed):**
  - `core_import_query_error.py` and `sync_existing_error.py` fail on 62c8d9f and pass on your build;
  - `core_export_device.py` is INCONCLUSIVE (one GPU), and your shim test proves that V4-01's guard switches and
    restores, and fails on its mutant;
  - the V4-02 docstring is updated.
- **G2 (sweep):** the sweep table is complete. Every site is OK or FIXED, and every FIXED site has a test, or a stated
  reason why none is feasible.
- **G3 (no regression):**
  - every round-1..4 probe passes, or is INCONCLUSIVE only for lack of 2 GPUs;
  - `addon_lifetime` passes 3/3 runs with 0 incorrect;
  - `core_exit_overrun` during and bounded give rc 0;
  - Voice's runner on T1–T6 gives PASS 236/236;
  - the combined session shows exactly the 52 baseline ids.
- **G4 (teeth):** build these mutants outside the worktree. Each must fail its new test:
  - (a) the V4-01 guard removed;
  - (b) the V4-03 check removed;
  - (c) the V4-04 capture removed.
- **G5 (scope):**
  - the diff against 62c8d9f touches only the Exercises paths, plus `docs/P1T_FIX4_*.md`, `artifacts/p1t_fix4/**`
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
  - CUDA 12.6 runtime current-device semantics;
  - `cudaEventRecord` and `cudaStreamWaitEvent` device rules;
  - sticky versus non-sticky errors (`cudaGetLastError` / `cudaPeekAtLastError`);
  - PyTorch `CUDAGuard` / `OptionalCUDAGuard` around event record in its DLPack path;
  - CuPy's `Device` context.
- If the AFTERIMAGE memory hook denies a Write, read what it shows you and retry. If a report Write is refused
  outright, put the full report in your final message.

## Done

- Write `docs/P1T_FIX4_REPORT.md` (the per-finding table, sweep table, probe before/after table, gate table, mutant
  table and prior art) and `docs/P1T_FIX4_LEDGER.md`.
- Your final message must contain, verbatim:
  - the per-finding table;
  - the sweep table, with a summary count;
  - the probe before/after table;
  - the gate table;
  - Voice's verdict line;
  - the compare output;
  - the mutant table;
  - the diff stat;
  - both `.so` sha256s;
  - every deviation, with its reason.
