# P1T-REV4: independent code review (round 4) of TensorCUDA's DLPack interop and the `tc_signal` add-on package

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_p1trev4`.
Edits there are AUTHORIZED, but only under `artifacts/p1t_rev4/`, `logs/` and `docs/P1T_REV4_*.md`. You may also
write scratch under `/tmp/p1t_rev4_*`. This is a CODE REVIEW: do not modify any source, test or build file anywhere.

## Context

TensorCUDA is a CUDA tensor engine. To let a text-to-speech port (Project-Voice) run on it, two things were added:
- **Core:** a small interop surface in core: DLPack export/import, `data_ptr`, `fill_`, `mark_modified`, and a
  device-side `empty`.
- **Add-on:** a separate package, `tc_signal/`. It provides 1-D convolutions, an ONNX-semantics LSTM, three
  normalizations, nearest upsample, reflection pad and round-half-to-even. It exchanges tensors with core only through
  DLPack.

This is the fourth round of independent review before the work enters the project's main branch:
- **Round 1** (`docs/P1T_REV_REPORT.md`) found F01–F07, which were fixed in FIX1 (`docs/P1T_FIX1_REPORT.md`).
- **Round 2** (`docs/P1T_REV2_LEDGER.md`, `artifacts/p1t_rev2/`) stopped early, without a verdict. Its CPU checks had
  found N1–N5, which were fixed in FIX2 (`docs/P1T_FIX2_REPORT.md`).
- **Round 3** (`docs/P1T_REV3_REPORT.md`, `artifacts/p1t_rev3/`) closed everything except V3-01 (MAJOR: the exit wait
  returned to finalization while admitted releases were running) and V3-02 (MINOR: unchecked aggregate byte sums).
  Both were fixed in FIX3 (`docs/P1T_FIX3_REPORT.md`).

The project owner, David, approved the merge on this condition, verbatim: "If the Blind Review passes, I approve the
merger." The review passes on APPROVE or APPROVE-WITH-NITS, with no BLOCKING or MAJOR finding open. Review it to the
same strict standard as rounds 1 to 3.

Write your findings as ordinary engineering defects: what goes wrong, a concrete reproduction or scenario, its
severity, and a fix.

## Mechanism & scope

- **Measures:** whether `9df5bef..62c8d9f`, restricted to `tensor_cuda/` and `tc_signal/`, is correct, robust in its
  error handling, free of regressions to existing core behaviour, and ready for the main branch. Specifically:
  - each earlier item (F01–F07, N1–N5, V3-01, V3-02) is CLOSED, PARTLY closed or OPEN, with evidence;
  - the FIX3 changes (and earlier fixes) introduce no new defects.
- **Mechanism doc:** `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_p1trev4/docs/P1T_REV_REPORT.md`
  (round-1 findings). Also:
  - `docs/P1T_REV2_LEDGER.md` and `artifacts/p1t_rev2/` (round 2);
  - `docs/P1T_REV3_REPORT.md` and `artifacts/p1t_rev3/` (round 3);
  - `docs/P1T_FIX1_REPORT.md`, `docs/P1T_FIX2_REPORT.md` and `docs/P1T_FIX3_REPORT.md` (the claimed fixes);
  - `tc_signal/README.md` (the add-on contract);
  - the approved design `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/docs/P1T_TENSOR_ALT_PLAN.md`.
- **Exercises:** read the full code under review, `git diff 9df5bef..62c8d9f -- tensor_cuda tc_signal`, and the FIX3
  delta, `git diff 5f9f955..62c8d9f -- tensor_cuda tc_signal`.
  - Main areas:
    - `tc_signal/src/dlpack_operand.{h,cpp}`: deferred operand release, the pending queue, `synchronize`, error
      reporting, and the deleter-call wrapper;
    - `tc_signal/src/checked_math.h` and every `*_plan` function;
    - core `tensor_cuda/src/bindings.cpp`: checked shape helpers, import/export, `empty`, the import-release
      lifecycle (mutex, closing flag, in-flight counter, atexit wait);
    - core `tensor_cuda/src/kernels.cu`: the external Storage destructor, `numel_of`, `contiguous_strides`;
    - the `__init__.py` files of both packages: atexit hooks.
  - CPU-only checks are allowed: compile, `nm`/`ldd`/`objdump`, host-side scripts, and the earlier rounds' CPU probes
    (read-only; copy them to `/tmp` if you need to rebuild).
- **Does not cover:**
  - Voice's acceptance tests and tolerances, which were reviewed separately;
  - performance;
  - GPU execution. This sandbox has no GPU. When a claim needs one, write a short probe under
    `artifacts/p1t_rev4/probes/` with a one-line pass/fail rule; the lead runs it before acting on the verdict.
- **Required evidence:**
  - a closure table for F01–F07, N1–N5, V3-01 and V3-02 (status and the lines you read);
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
  - CUDA errors already present when `synchronize()` is called;
  - multi-device pending entries.
- **Baseline isolation:**
  - The baseline is the main branch at `9df5bef`, plus the earlier rounds' findings.
  - The lead's receipts are claims to check, not facts: `artifacts/p1t_fix3/lead/` (Voice's runner 236/236; core plus
    `tc_signal` in one session showing only the 52 known baseline failures; the round-1 probes passing).
- **Forbidden configuration:**
  - editing anything outside `artifacts/p1t_rev4/`, `logs/` and `docs/P1T_REV4_*.md`;
  - running GPU code;
  - git writes. Read-only git is ALLOWED.

## Questions

- **R1, effect on existing core behaviour:** SAFE / RISK.
- **R2, core DLPack correctness (lifetimes, stream ordering, error handling):** SOUND / UNSOUND.
- **R3, the `tc_signal` operand contract (deferred release, validation, error reporting):** SOUND / UNSOUND.
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
- **Write early.** Create `docs/P1T_REV4_REPORT.md` and `artifacts/p1t_rev4/findings.json` early, and update them as
  you go, so that a partial run still leaves your findings on disk.

## Done

- Write `docs/P1T_REV4_REPORT.md` (the R1–R6 table, the closure table, the new findings table and the probe list) and
  `artifacts/p1t_rev4/findings.json` (with `verdicts`, `closure` and `findings`).
- Your final message must contain, verbatim:
  - the R1–R6 table;
  - the closure table;
  - every BLOCKING and MAJOR finding in full;
  - the MINOR and NIT counts;
  - the probe paths, with what each one decides.
