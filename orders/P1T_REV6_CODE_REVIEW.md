# P1T-REV6: independent code review (round 6) of TensorCUDA's DLPack interop and the `tc_signal` add-on package

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_p1trev6`.
Edits there are AUTHORIZED, but only under `artifacts/p1t_rev6/`, `logs/` and `docs/P1T_REV6_*.md`. You may also
write scratch under `/tmp/p1t_rev6_*`. This is a CODE REVIEW: do not modify any source, test or build file anywhere.

## Context

TensorCUDA is a CUDA tensor engine. To let a text-to-speech port (Project-Voice) run on it, two things were added:
- **Core:** a small interop surface in core: DLPack export/import, `data_ptr`, `fill_`, `mark_modified`, and a
  device-side `empty`.
- **Add-on:** a separate package, `tc_signal/`. It provides 1-D convolutions, an ONNX-semantics LSTM, three
  normalizations, nearest upsample, reflection pad and round-half-to-even. It exchanges tensors with core only through
  DLPack.

This is the sixth round of independent review before the work enters the project's main branch:
- **Round 1** (`docs/P1T_REV_REPORT.md`) found F01–F07, which were fixed in FIX1.
- **Round 2** (`docs/P1T_REV2_LEDGER.md`) stopped early; its CPU checks found N1–N5, which were fixed in FIX2.
- **Round 3** (`docs/P1T_REV3_REPORT.md`) found V3-01 and V3-02, which were fixed in FIX3.
- **Round 4** (`docs/P1T_REV4_REPORT.md`) found V4-01..V4-04, which were fixed in FIX4. FIX4 also swept two defect
  classes, (a) implicit current device and (b) unchecked CUDA returns, across every CUDA call site.
- **Round 5** (`docs/P1T_REV5_REPORT.md`, `artifacts/p1t_rev5/`) closed all 18 earlier items. It found:
  - V5-01 (MAJOR): interpreter exit overtook an add-on producer deleter running on another thread;
  - V5-02 (MINOR): a one-shot event-query error was lost;
  - V5-03 (MINOR): a cuBLAS handle leaked when the cache insertion threw;
  - V5-04 (MINOR): `fill_` narrowed its block count.

  FIX5 (`docs/P1T_FIX5_REPORT.md`, `docs/P1T_FIX5_LEDGER.md`) fixed them, and swept four more classes across all the
  P1-T code:
  - (c) callbacks during interpreter exit;
  - (d) errors consumed but not recorded;
  - (e) raw-resource exception safety;
  - (f) launch-geometry narrowing.

  The sweep tables are in that report. Sweep (c) found one more core defect, which FIX5 also fixed: the core exit
  wait re-took the GIL inside a destructor (deviation D7).

The project owner, David, approved the merge on this condition, verbatim: "If the Blind Review passes, I approve the
merger." The review passes on APPROVE or APPROVE-WITH-NITS, with no BLOCKING or MAJOR finding open. Review it to the
same strict standard as rounds 1 to 5.

Write your findings as ordinary engineering defects: what goes wrong, a concrete reproduction or scenario, its
severity, and a fix.

## Mechanism & scope

- **Measures:** whether `9df5bef..acd42fa`, restricted to `tensor_cuda/` and `tc_signal/`, is correct, robust in its
  error handling, free of regressions to existing core behaviour, and ready for the main branch. Specifically:
  - each earlier item (F01–F07, N1–N5, V3-01, V3-02, V4-01–V4-04, V5-01–V5-04) is CLOSED, PARTLY closed or OPEN, with
    evidence;
  - the FIX5 changes, including its deviations, introduce no new defects;
  - the FIX5 sweep tables (c)–(f) are complete and their verdicts are right. Treat them as claims to check: look for
    sites they missed, and spot-check their rows against the code.
- **Judge these FIX5 behaviour changes and residuals explicitly.** For each, say whether it is acceptable or a defect,
  and give the severity:
  - (D1) `tc_signal.synchronize()` now waits, with the GIL released and no time bound, for add-on releases already
    running on other threads. Before FIX5 it returned without waiting for them. Consider lock-order inversions with a
    producer deleter, and a deleter that never returns.
  - (D1) After the add-on exit hook, a call that needs a producer `__dlpack__` raises RuntimeError.
  - (R2) A producer deleter that itself calls `pthread_exit` still aborts.
  - (R3) Core's `__dlpack__` argument casts (user `__index__` / `__bool__`) are not admitted.
  - (R4) The add-on wait is count-based; its liveness when releases overlap continuously is not bounded.
- **Mechanism doc:** `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_p1trev6/docs/P1T_REV5_REPORT.md`
  (the round-5 findings). Also:
  - `docs/P1T_REV_REPORT.md`, `docs/P1T_REV2_LEDGER.md`, `docs/P1T_REV3_REPORT.md` and `docs/P1T_REV4_REPORT.md`
    (rounds 1–4);
  - `artifacts/p1t_rev{,2,3,4,5}/` (earlier findings and probes);
  - `docs/P1T_FIX1_REPORT.md` to `docs/P1T_FIX5_REPORT.md` (the claimed fixes);
  - `tc_signal/README.md` (the add-on contract);
  - the approved design `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/docs/P1T_TENSOR_ALT_PLAN.md`.
- **Exercises:** read the full code under review, `git diff 9df5bef..acd42fa -- tensor_cuda tc_signal`, and the FIX5
  delta, `git diff 1475fae..acd42fa -- tensor_cuda tc_signal`.
  - Main areas:
    - `tc_signal/src/dlpack_operand.{h,cpp}`:
      - the new release lifecycle (`admit`, `leave`, `wait_for_other_threads`, `close_admission_and_wait`, the
        self-exclusion depth);
      - `call_deleter_contained`, `CapsuleRef::drop` and `to_capsule`;
      - `drain_completed` and its error recording;
      - `synchronize`;
    - `tc_signal/tc_signal/__init__.py` (the exit hook) and `tc_signal/src/bindings.cpp`, `lstm_bindings.cpp`,
      `m4_bindings.cpp` (the call sites);
    - `NewHandleOwner` in `tc_signal/src/conv.cu` and `lstm.cu`;
    - core `tensor_cuda/src/kernels.cu`: `fill_grid_blocks` and `fill_`, and the recording of unraisable errors;
    - core `tensor_cuda/src/bindings.cpp`: `dlpack_begin_shutdown` (the explicit GIL restore) and the release stats.
  - CPU-only checks are allowed: compile, `nm`/`ldd`/`objdump`, host-side scripts, and the earlier rounds' CPU probes
    (read-only; copy them to `/tmp` if you need to rebuild).
- **Does not cover:**
  - Voice's acceptance tests and tolerances, which were reviewed separately;
  - performance;
  - GPU execution. This sandbox has no GPU. When a claim needs one, write a short probe under
    `artifacts/p1t_rev6/probes/` with a one-line pass/fail rule; the lead runs it before acting on the verdict. The
    lead's host has ONE GPU: a probe that needs two GPUs must exit 3 (INCONCLUSIVE) there, so where you can, also
    give a single-GPU form (an `LD_PRELOAD` or ctypes interposer). A probe must not embed a copy of product code that
    stops it from passing on a fixed build; build it against the sources under review instead;
  - pre-existing core code that P1-T does not change (for example the shared `nblk` helper, `Device::from_string`,
    or engine ops other than `fill_`). These are baseline, not findings, unless the P1-T surface makes them reachable
    in a new way. If so, say how.
- **Required evidence:**
  - a closure table for the 22 earlier items (status and the lines you read);
  - a sweep-table check for (c)–(f): the sites you found missing, if any, and the rows you spot-checked, with your
    verdict on each;
  - a verdict on each D1 / R2 / R3 / R4 item above;
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
  - interpreter exit with releases pending or in progress on other threads, through the immediate-refusal and the
    deferred paths, and the hook called from inside a callback;
  - `synchronize()` called while another thread's release is running;
  - CUDA errors already present, or arising once, when `synchronize()`, an op or `fill_` is called (sticky and
    non-sticky);
  - multi-device pending entries;
  - the current device different from the tensor's device at export, import, `fill_` and each `tc_signal` op;
  - `fill_` at the grid boundary;
  - a failed device restore on a success path and while unwinding.
- **Baseline isolation:**
  - The baseline is the main branch at `9df5bef`, plus the earlier rounds' findings.
  - The lead's receipts are claims to check, not facts. They are in `artifacts/p1t_fix5/lead/`:
    - Voice's runner gives 249/249, including T1R and T4S;
    - core plus `tc_signal` in one session shows only the 52 known baseline failures;
    - the round-5 GPU probes fail on 1475fae and pass on FIX5;
    - the FIX4 and FIX5 tests pass, and the V5-01 mutant fails 11 of them.
- **Forbidden configuration:**
  - editing anything outside `artifacts/p1t_rev6/`, `logs/` and `docs/P1T_REV6_*.md`;
  - running GPU code;
  - git writes. Read-only git is ALLOWED.

## Questions

- **R1, effect on existing core behaviour:** SAFE / RISK.
- **R2, core DLPack correctness (lifetimes, stream ordering, device handling, error handling, interpreter exit):**
  SOUND / UNSOUND.
- **R3, the `tc_signal` operand contract (deferred release, validation, device handling, error reporting,
  interpreter exit):** SOUND / UNSOUND.
- **R4, kernel and arithmetic correctness, including the checked-arithmetic rules and launch geometry:** CORRECT /
  DEFECTS.
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
- **Write early.** Create `docs/P1T_REV6_REPORT.md` and `artifacts/p1t_rev6/findings.json` early, and update them as
  you go, so that a partial run still leaves your findings on disk.

## Done

- Write `docs/P1T_REV6_REPORT.md` (the R1–R6 table, the closure table, the sweep-table check, the D1/R2/R3/R4
  verdicts, the new findings table and the probe list) and `artifacts/p1t_rev6/findings.json` (with `verdicts`,
  `closure`, `sweep_check`, `behaviour_changes` and `findings`).
- Your final message must contain, verbatim:
  - the R1–R6 table;
  - the closure table;
  - the D1/R2/R3/R4 verdicts;
  - every BLOCKING and MAJOR finding in full;
  - the MINOR and NIT counts;
  - the sweep-table check summary;
  - the probe paths, with what each one decides.
