# P1T-FIX1: close the P1T-REV round-1 findings (F01–F07) in core DLPack and `tc_signal`

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix1`
(branch `voice-fix1`, which is `voice-int` 00590b7 plus the review receipts). Edits, builds and GPU runs inside it are
AUTHORIZED. You may also write scratch and mutant builds under `/tmp/p1t_fix1_*`.

## Why

**The work so far.** David, 2026-10-12: "I don't want to build it directly INTO the core of TensorCUDA - It needs to be
separate I guess." Following that, M1–M4 built core DLPack interop plus the `tc_signal` add-on.
- Project-Voice accepted all six items: PASS 236/236 on their runner (their LD62).
- House rule §10 requires a cross-model review before promotion to production.

**The review rejected it.** The blind Codex review (gpt-6.1-sol high) said **REJECT**:
- Report: `docs/P1T_REV_REPORT.md`; findings: `artifacts/p1t_rev/findings.json`.
- Findings: 1 BLOCKING, 4 MAJOR, 2 MINOR.

**What the lead's GPU probes showed** (`artifacts/p1t_rev/lead_probe_runs/`):
- **F01 is REPRODUCED.** `addon_lifetime.py`: the add-on read reused storage (trial 11, reused and incorrect), so this
  is a real use-after-free.
- **F03 is REPRODUCED.** `core_edges.py`: rank 9 and element/byte count wraparound are accepted.
- **F04 did NOT reproduce on CPython 3.12.** `core_shutdown.py` passed both `--normal` and finalizing. Fix it anyway;
  the CPython docs support the concern.
- **F05 cannot be probed here,** because there is one GPU. Fix it by construction.
- `numerics_and_streams.py`: PASS.

Your job is to fix F01–F06 and document F07, prove each fix, and keep every acceptance result green. After you, a
round-2 blind review follows.

## Mechanism & scope

- **Measures:**
  - whether each finding F01–F07 is closed, shown by the reviewer's own probe (or host check) flipping from FAIL to
    PASS, plus a new regression test per finding;
  - that nothing regresses: Voice's runner on T1–T6, `tc_signal`'s tests, `test_dlpack.py`, and core's full suite
    (52 baseline ids);
  - the per-call cost of the F01 fix.
- **Mechanism doc:** `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix1/docs/P1T_REV_REPORT.md`
  (the findings, their scenarios and suggested fixes). Also `artifacts/p1t_rev/findings.json` and
  `artifacts/p1t_rev/probes/README.md`, which says how to run each probe.
- **Exercises:**
  - core: `tensor_cuda/src/bindings.cpp`, `tensor_cuda/src/kernels.cu` (only the Storage external path),
    `tensor_cuda/include/tc/core.h`, `tensor_cuda/tensor_cuda/__init__.py`, `tensor_cuda/tests/test_dlpack.py`;
  - add-on: `tc_signal/src/*`, `tc_signal/tc_signal/*`, `tc_signal/tests/*`, `tc_signal/README.md`;
  - the reviewer's probes under `artifacts/p1t_rev/probes/`, run read-only.
- **Does not cover:**
  - new ops or features;
  - speed work beyond keeping per-call cost reasonable;
  - any core code outside the DLPack/empty/fill_/mark_modified surface and the Storage external path;
  - Voice's tests (READ-ONLY);
  - multi-GPU execution (F05 is verified by construction plus a single-device test that the device is restored).
- **Required evidence:**
  - a per-finding table (finding, fix with file:line, proving test, probe before and after);
  - the full logs of the regression gates;
  - the F01 cost table;
  - the mutant table;
  - the diff stat against 00590b7;
  - the built `.so` sha256s for core and `tc_signal`.
- **Operand census:**
  - F01: torch producers on a side `torch.cuda.Stream()`, bare capsules, producer temporaries, partial-launch
    exceptions, and the LSTM's per-call capsule set;
  - F02: None for every required operand of all nine `tc_signal` ops, plus core `from_dlpack(None)`;
  - F03: rank 9, `(1<<62,)` fp32, `(1<<32, 1<<32)`, offset + bytes overflow on import, and reflect/upsample/conv/LSTM
    geometry overflow;
  - F04: foreign-thread release while finalizing;
  - F05: drop an imported tensor while a different device is current (simulate with `cudaSetDevice` on the same
    device index, and assert the device is restored);
  - F06: the bad_alloc path, by reasoning plus RAII.
- **Baseline isolation:**
  - Every "before" result is the lead's run on 00590b7 (`artifacts/p1t_rev/lead_probe_runs/`).
  - Run every "after" result on your rebuilt core and `tc_signal`, in fresh processes, under the GPU lock.
  - The core suite is compared with the 52-id baseline log named below.
  - Mutants are built outside the worktree.
- **Forbidden configuration:**
  - weakening, skipping or editing the reviewer's probes or Voice's tests;
  - calling deleters from CUDA host callbacks;
  - device-wide syncs (`cudaDeviceSynchronize`) on the per-call path;
  - fast-math, TF32 or atomics;
  - any new `__global__` kernel in core;
  - writing outside the target or `/tmp`.

## Required fixes

1. **F01 (BLOCKING): consumed operand ownership through GPU completion.**
   - In `tc_signal`, every consumed DLPack manager (inputs, outputs, scratch) must stay alive until the legacy stream
     has passed the last kernel that uses it. That includes exception paths after a partial launch.
   - **Preferred design:** event-backed deferred release.
     - After the launch, record a `cudaEvent` on the legacy stream and push (event, managers) onto a module-level
       pending list.
     - Release completed entries (`cudaEventQuery == cudaSuccess`), in FIFO order, from the host thread holding the
       GIL. Do this at the next `tc_signal` call, in an explicit `tc_signal.synchronize()` that syncs the legacy
       stream and releases everything, and from an `atexit` hook.
     - Bound the pending list (count and bytes). When it is exceeded, synchronize the legacy stream and drain.
     - Never release from a CUDA host callback.
   - **Acceptable alternative,** if you justify it: a checked legacy-stream sync before release, with its measured
     cost.
   - Pin the choice in the README's contract.
2. **F02: required operands.**
   - None for any required operand raises `TypeError` before planning, in all nine ops. Distinguish required from
     optional operands.
   - Make access to an absent Operand's metadata fail safely as a second line of defence.
   - Core `from_dlpack(None)` raises `TypeError` (check it).
3. **F03: checked arithmetic.**
   - Core `empty` and import refuse rank > `TC_MAX_DIMS` (8).
   - Use checked non-negative element multiplication, element-size multiplication, and `byte_offset + nbytes` and
     address addition. Overflow raises `ValueError` and never wraps to empty.
   - In `tc_signal`, every plan function checks output-length, scratch and byte arithmetic, including reflect,
     upsample, conv, conv_transpose, LSTM and the norms, before any narrowing cast or launch.
4. **F04: finalization-safe release.**
   - When the interpreter is finalizing or not initialized (`Py_IsFinalizing` / `_Py_IsFinalizing` on 3.12, and
     `Py_IsInitialized`), do NOT call `PyGILState_Ensure`. Deliberately leak the manager and say so in a comment and in
     the report.
   - Ordinary releases are unchanged.
5. **F05: device-correct drain.** The external Storage destructor:
   1. switches to `Storage.device.index` if that is not current;
   2. syncs that device's legacy stream;
   3. on error, does NOT release, and leaks rather than risk a use-after-free;
   4. restores the caller's device.
6. **F06:** own the export context with a `unique_ptr` until the capsule takes ownership.
7. **F07:** document the `Storage` layout change (40 → 56 bytes) as a rollout prerequisite: any native client of core
   headers needs a full rebuild, and none is known. Put it in the M1 report addendum and the `tc_signal` README is not
   needed. The promotion note goes in your report.

## How to run

- **GPU:** every GPU command runs as `flock -w 900 /tmp/forge-gpu.lock timeout 600 …`. Use the native driver only.
  You are the only seat running.
- **Builds:**
  - core: `cd <target>/tensor_cuda && ./build.sh 89`;
  - `tc_signal`: `cd <target>/tc_signal && PYBIND11_SOURCE_DIR=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src ./build.sh 89`.
- **Package path:** `PYTHONPATH=<target>/tensor_cuda/tensor_cuda:<target>/tensor_cuda:<target>/tc_signal`, plus
  `PYTHONDONTWRITEBYTECODE=1`.
- **The reviewer's probes:** run each as `artifacts/p1t_rev/probes/README.md` says. Compile the
  `core_shutdown.py` helper exactly as given. Do not edit them.
- **The reviewer's host checks:** `artifacts/p1t_rev/host_checks.py` (None inputs). After the fix, they must raise
  cleanly, with no SIGSEGV.
- **Voice's runner, T1–T6.** Run it from `/mnt/ForgeRealm/Project-Voice`, which is READ-ONLY to you:
  ```
  cd /mnt/ForgeRealm/Project-Voice && flock -w 900 /tmp/forge-gpu.lock timeout 1200 env PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=<pp> python3 engine/tensor_requests/p1t_tests/run_p1t.py --adapter <target>/artifacts/p1t_final/tensorcuda_adapter.py \
    --items T1,T2,T3,T4,T5,T6 --fixtures /mnt/ForgeRealm/Project-Voice/artifacts/p1t/fixtures \
    --tolerances engine/tensor_requests/p1t_tests/tolerances_r2.json --out <target>/artifacts/p1t_fix1/p1t_T1toT6.json
  ```
- **The core suite:**
  `cd <target> && flock -w 900 /tmp/forge-gpu.lock timeout 900 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<target>/tensor_cuda/tensor_cuda:<target>/tensor_cuda python3 -m pytest tensor_cuda/tests -q -p no:cacheprovider --ignore=tensor_cuda/tests/test_selector_accuracy.py`.
  Compare it with
  `python3 /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2/scripts/apa_mf2_2_g1_compare.py /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2/logs/apa_mf1_g1_baseline_unchanged.log <log>`.

## Gates

- **G1 (findings closed):**
  - `addon_lifetime.py` PASSES in 3 separate runs, with 0 incorrect trials;
  - `core_edges.py` PASSES;
  - `core_shutdown.py` passes both `--normal` and finalizing;
  - `numerics_and_streams.py` PASSES;
  - `host_checks.py` shows clean exceptions and no crash on the None cases;
  - each finding F01–F06 has a new regression test that fails on its mutant (G3) and passes on your build.
- **G2 (no regression):**
  - Voice's runner on T1–T6 exits 0 with PASS 236/236;
  - `tc_signal` tests all pass;
  - `test_dlpack.py` passes;
  - the core suite's FAILED/ERROR set equals the 52 baseline ids.
- **G3 (teeth):** build these mutants outside the worktree. Each must fail its new test, and (a) must also fail
  `addon_lifetime.py`:
  - (a) immediate release restored, i.e. the F01 fix removed;
  - (b) the F02 None guard removed;
  - (c) the F03 rank check removed;
  - (d) the F05 device switch removed (show that the test detects the device NOT being switched or restored).
- **G4 (cost, informational):**
  - `tc_signal` per-call wall time, median of 200 after 20 warm-ups, before (the 00590b7 build at
    `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_int`) and after, for
    `conv1d` [1, 8, 16] and `rint` [1, 82];
  - the pending-list high-water mark during Voice's runner.
- **G5 (scope):**
  - the diff against 00590b7 touches only the Exercises paths, plus `docs/P1T_FIX1_*.md`, `artifacts/p1t_fix1/**`
    and `logs/**`;
  - there are no new `__global__` kernels in core;
  - the reviewer's probe files are byte-identical.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. Your order file is
  immutable.
- NO git (the lead commits). NO subagents. No background waits: run every gate synchronously to completion.
- No deletes, anywhere. Never kill a process you did not start.
- These are READ-ONLY: production `/mnt/ForgeRealm/Project-Tensor` (outside `Fork/`), `/mnt/ForgeRealm/Project-Voice`,
  and every other fork or worktree.
- RED honesty: a failed gate is a result. Report it with receipts; never weaken a gate, probe, test or tolerance to
  pass.
- Prior Art Directive (AGENTS.md): annotate at the code site, in the ledger and in the report. Known prior art:
  - PyTorch's `Tensor.record_stream` and caching-allocator event-deferred frees;
  - CuPy's memory-pool stream ordering;
  - the DLPack consumer contract;
  - CPython 3.12 finalization and GIL rules;
  - checked arithmetic (`__builtin_mul_overflow`, C23 `ckd_mul`).
- If the AFTERIMAGE memory hook denies a Write, read what it shows you and retry. Do not route around it through the
  shell. If a report Write is refused outright, put the full report in your final message.

## Done

- Write `docs/P1T_FIX1_REPORT.md`: the per-finding table, the gate table, F01's design and cost, the mutant table, the
  F07 rollout note and prior art. Also write `docs/P1T_FIX1_LEDGER.md`.
- Your final message must contain, verbatim:
  - the per-finding table;
  - the gate table;
  - the probe before/after lines;
  - the G2 verdict line;
  - the core compare output;
  - the G3 mutant table;
  - the G4 table;
  - the diff stat;
  - both `.so` sha256s;
  - every deviation, with its reason.
