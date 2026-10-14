# P1T-REV5: independent code review (round 5) of TensorCUDA's DLPack interop and the `tc_signal` add-on package

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_p1trev5`.
Edits there are AUTHORIZED, but only under `artifacts/p1t_rev5/`, `logs/` and `docs/P1T_REV5_*.md`. You may also
write scratch under `/tmp/p1t_rev5_*`. This is a CODE REVIEW: do not modify any source, test or build file anywhere.

## Context

TensorCUDA is a CUDA tensor engine. To let a text-to-speech port (Project-Voice) run on it, two things were added:
- **Core:** a small interop surface in core: DLPack export/import, `data_ptr`, `fill_`, `mark_modified`, and a
  device-side `empty`.
- **Add-on:** a separate package, `tc_signal/`. It provides 1-D convolutions, an ONNX-semantics LSTM, three
  normalizations, nearest upsample, reflection pad and round-half-to-even. It exchanges tensors with core only through
  DLPack.

This is the fifth round of independent review before the work enters the project's main branch:
- **Round 1** (`docs/P1T_REV_REPORT.md`) found F01–F07, which were fixed in FIX1 (`docs/P1T_FIX1_REPORT.md`).
- **Round 2** (`docs/P1T_REV2_LEDGER.md`, `artifacts/p1t_rev2/`) stopped early, without a verdict. Its CPU checks had
  found N1–N5, which were fixed in FIX2 (`docs/P1T_FIX2_REPORT.md`).
- **Round 3** (`docs/P1T_REV3_REPORT.md`, `artifacts/p1t_rev3/`) found V3-01 (MAJOR) and V3-02 (MINOR), which were
  fixed in FIX3 (`docs/P1T_FIX3_REPORT.md`).
- **Round 4** (`docs/P1T_REV4_REPORT.md`, `artifacts/p1t_rev4/`) closed all fourteen earlier items. It found:
  - V4-01 (MAJOR): the export fence event was created and recorded on the current device, not the tensor's device;
  - V4-03 (MINOR): import ignored a failed `cudaGetDevice` and still took ownership of the capsule;
  - V4-04 (MINOR): `synchronize()` did not report a CUDA error that was already pending;
  - V4-02 (NIT): a docstring still described the old bounded exit wait.

  FIX4 (`docs/P1T_FIX4_REPORT.md`, `docs/P1T_FIX4_LEDGER.md`) fixed them. Because rounds 3 and 4 both found instances
  of the same two defect classes, FIX4 also swept every CUDA runtime/cuBLAS call site in the P1-T diff for:
  - (a) an implicit current-device assumption;
  - (b) an unchecked or ignored return or error state.

  Its sweep table is `artifacts/p1t_fix4/sweep/sweep_table.md`. FIX4 also went beyond the literal fixes; its report
  lists these as deviations D2–D7:
  - a device-label check at export and `fill_`;
  - `empty` labelling the current device;
  - taking an already-pending error before a launch;
  - failed device restores raising on success paths;
  - cuBLAS handle cleanup.

The project owner, David, approved the merge on this condition, verbatim: "If the Blind Review passes, I approve the
merger." The review passes on APPROVE or APPROVE-WITH-NITS, with no BLOCKING or MAJOR finding open. Review it to the
same strict standard as rounds 1 to 4.

Write your findings as ordinary engineering defects: what goes wrong, a concrete reproduction or scenario, its
severity, and a fix.

## Mechanism & scope

- **Measures:** whether `9df5bef..1475fae`, restricted to `tensor_cuda/` and `tc_signal/`, is correct, robust in its
  error handling, free of regressions to existing core behaviour, and ready for the main branch. Specifically:
  - each earlier item (F01–F07, N1–N5, V3-01, V3-02, V4-01–V4-04) is CLOSED, PARTLY closed or OPEN, with evidence;
  - the FIX4 changes, including its deviations D2–D7, introduce no new defects;
  - the FIX4 sweep table is complete and its verdicts are right. Treat it as a claim to check: look for call sites
    it missed, and spot-check its OK and FIXED rows against the code.
- **Mechanism doc:** `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_p1trev5/docs/P1T_REV4_REPORT.md`
  (the round-4 findings). Also:
  - `docs/P1T_REV_REPORT.md`, `docs/P1T_REV2_LEDGER.md` and `docs/P1T_REV3_REPORT.md` (rounds 1–3);
  - `artifacts/p1t_rev{,2,3,4}/` (earlier findings and probes);
  - `docs/P1T_FIX1_REPORT.md` to `docs/P1T_FIX4_REPORT.md` (the claimed fixes);
  - `tc_signal/README.md` (the add-on contract);
  - the approved design `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/docs/P1T_TENSOR_ALT_PLAN.md`.
- **Exercises:** read the full code under review, `git diff 9df5bef..1475fae -- tensor_cuda tc_signal`, and the FIX4
  delta, `git diff 62c8d9f..1475fae -- tensor_cuda tc_signal`.
  - Main areas:
    - core `tensor_cuda/src/bindings.cpp`:
      - the export fence (`dlpack_fence_consumer`) and its device switch and restore;
      - the import's device query;
      - `empty`;
      - the import-release lifecycle;
    - core `tensor_cuda/src/kernels.cu`:
      - `cuda_storage_device_mismatch`, `CheckedDeviceGuard` and `fill_`;
      - `drain_device_legacy_stream` and the external Storage destructor;
    - `tc_signal/src/cuda_guard.h` (new);
    - `tc_signal/src/dlpack_operand.{h,cpp}`:
      - the pre-launch error take in `RetainUntilComplete`;
      - `synchronize` and its handling of a pre-existing error;
      - deferred release and the pending queue;
    - the `check_cuda` and `check_launch` helpers and cuBLAS handle creation in `conv.cu`, `lstm.cu`, `norms.cu` and
      `small_ops.cu`;
    - the `__init__.py` files of both packages.
  - CPU-only checks are allowed: compile, `nm`/`ldd`/`objdump`, host-side scripts, and the earlier rounds' CPU probes
    (read-only; copy them to `/tmp` if you need to rebuild).
- **Does not cover:**
  - Voice's acceptance tests and tolerances, which were reviewed separately;
  - performance;
  - GPU execution. This sandbox has no GPU. When a claim needs one, write a short probe under
    `artifacts/p1t_rev5/probes/` with a one-line pass/fail rule; the lead runs it before acting on the verdict. The
    lead's host has ONE GPU: a probe that needs two GPUs must exit 3 (INCONCLUSIVE) there, so where you can, also
    give a single-GPU form (an `LD_PRELOAD` or ctypes interposer, like `tc_signal/tests/fix4_cudart_shim.c`);
  - pre-existing core code that P1-T does not change (for example `Device::from_string`, or engine ops other than
    `fill_`). These are baseline, not findings, unless the P1-T surface makes them reachable in a new way. If so,
    say how.
- **Required evidence:**
  - a closure table for F01–F07, N1–N5, V3-01, V3-02 and V4-01–V4-04 (status and the lines you read);
  - a sweep-table check: the call sites you found missing, if any, and the rows you spot-checked with your verdict on
    each;
  - every new finding with file:line, severity (BLOCKING / MAJOR / MINOR / NIT), scenario and fix;
  - R1–R6 verdicts;
  - the probe list.
- **Operand census:**
  - zero-size and 0-d tensors;
  - extents near int64 limits, including next to a zero extent;
  - non-zero `byte_offset`;
  - size-1 dimensions with unusual strides;
  - read-only and versioned capsules;
  - capsules consumed twice;
  - producers on non-default streams;
  - one operand passed twice in a call;
  - producer deleters that raise;
  - host allocation failure on the release path;
  - interpreter exit with releases pending or in progress on other threads;
  - CUDA errors already present when `synchronize()`, an op or `fill_` is called (sticky and non-sticky);
  - multi-device pending entries;
  - the current device different from the tensor's device at export, import, `fill_` and each `tc_signal` op;
  - a tensor whose device label differs from where its memory is;
  - `empty` with `cuda`, `cuda:N`, `cpu` and zero-size shapes;
  - a failed device restore on a success path and while unwinding.
- **Baseline isolation:**
  - The baseline is the main branch at `9df5bef`, plus the earlier rounds' findings.
  - The lead's receipts are claims to check, not facts. They are in `artifacts/p1t_fix4/lead/`:
    - Voice's runner gives 249/249, including Voice's new items T1R and T4S;
    - core plus `tc_signal` in one session shows only the 52 known baseline failures;
    - the round-4 probes fail on 62c8d9f and pass on FIX4, except `core_export_device.py`, which is INCONCLUSIVE for
      lack of a second GPU;
    - the FIX4 tests pass, and the V4-01 mutant fails 8 of them.
- **Forbidden configuration:**
  - editing anything outside `artifacts/p1t_rev5/`, `logs/` and `docs/P1T_REV5_*.md`;
  - running GPU code;
  - git writes. Read-only git is ALLOWED.

## Questions

- **R1, effect on existing core behaviour:** SAFE / RISK.
- **R2, core DLPack correctness (lifetimes, stream ordering, device handling, error handling):** SOUND / UNSOUND.
- **R3, the `tc_signal` operand contract (deferred release, validation, device handling, error reporting):** SOUND /
  UNSOUND.
- **R4, kernel and arithmetic correctness, including the checked-arithmetic rules:** CORRECT / DEFECTS.
- **R5, package separation and build:** PASS / FAIL.
- **R6, overall:** APPROVE / APPROVE-WITH-NITS / REJECT, for merging into the main branch.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is immutable.
- NO subagents. No background waits. Run everything synchronously to completion.
- Never delete anything: no `rm`, no `git clean`, and nothing like them. You are not covered by Claude's hooks, so
  this is on you.
- Never kill a process you did not start.
- Everything outside your writable paths is READ-ONLY. That includes the main checkout
  `/mnt/ForgeRealm/Project-Tensor` (outside `Fork/`), `/mnt/ForgeRealm/Project-Voice`, and every other fork or worktree.
- Form your own view from the code before you read the fix reports.
- If you cannot settle something, say so and write the probe. Never guess a verdict.
- Prior Art Directive (AGENTS.md): name the standard or source behind a finding: the DLPack spec, the Python array
  API, CUDA stream and event semantics, the CPython 3.12 lifecycle docs, the C++ standard, or the ONNX operator specs.
- **Write early.** Create `docs/P1T_REV5_REPORT.md` and `artifacts/p1t_rev5/findings.json` early, and update them as
  you go, so that a partial run still leaves your findings on disk.

## Done

- Write `docs/P1T_REV5_REPORT.md` (the R1–R6 table, the closure table, the sweep-table check, the new findings table
  and the probe list) and `artifacts/p1t_rev5/findings.json` (with `verdicts`, `closure`, `sweep_check` and
  `findings`).
- Your final message must contain, verbatim:
  - the R1–R6 table;
  - the closure table;
  - every BLOCKING and MAJOR finding in full;
  - the MINOR and NIT counts;
  - the sweep-table check summary;
  - the probe paths, with what each one decides.
