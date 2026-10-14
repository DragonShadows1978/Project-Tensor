# P1T-REV8: independent code review (round 8) of TensorCUDA's DLPack interop and the `tc_signal` add-on package

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_p1trev8`.
Edits there are AUTHORIZED, but only under `artifacts/p1t_rev8/`, `logs/` and `docs/P1T_REV8_*.md`. You may also
write scratch under `/tmp/p1t_rev8_*`. This is a CODE REVIEW: do not modify any source, test or build file anywhere.

## Context

TensorCUDA is a CUDA tensor engine. To let a text-to-speech port (Project-Voice) run on it, two things were added:
- **Core:** a small interop surface: DLPack export/import, `data_ptr`, `fill_`, `mark_modified`, and a device-side
  `empty`.
- **Add-on:** a separate package, `tc_signal/`, with 1-D convolutions, an ONNX-semantics LSTM, three normalizations,
  nearest upsample, reflection pad and round-half-to-even. It exchanges tensors with core only through DLPack.

This is the eighth round of independent review before the work enters the project's main branch.

- **The last complete closure table** is round 7's, in `docs/P1T_REV7_REPORT.md` and `artifacts/p1t_rev7/findings.json`.
  - It recorded 22 of the 25 earlier items CLOSED and 3 PARTLY closed.
  - It accepted every FIX6 user-visible change and every "Not a defect" item.
  - It found two new defects:
    - V7-01 (MAJOR): both interpreter-exit hooks converted a diagnostic setting with `float()` before their
      close-and-wait. An out-of-range value made them skip the close, and the process ended abnormally when exit
      overtook a release still running on another thread;
    - V7-02 (MINOR): a failed completion-event query was reported as a failed stream synchronization.
- **FIX7** (`docs/P1T_FIX7_REPORT.md`, `docs/P1T_FIX7_LEDGER.md`) fixed both. It also swept two defect classes across
  all P1-T code:
  - (i) paths that can skip an exit guarantee;
  - (j) error messages and records that name the wrong call or device.
- **The lead sent one item back before this review: L1.** The new lock-retry loop in the core exit close could wait
  without a stderr diagnostic. It is fixed, and the "Lead send-back L1" section of the report covers it.
- **The FIX7 report also holds:**
  - a "Not a defect (with justification)" table (N1–N4);
  - a list of user-visible changes (U1–U9);
  - the corrected FIX6 sweep rows.
- **The lead's verification** is `artifacts/p1t_fix7/lead/LEAD_VERIFICATION.md`.

The project owner, David, approved the merge on this condition, verbatim: "If the Blind Review passes, I approve the
merger." The review passes on APPROVE or APPROVE-WITH-NITS, with no BLOCKING or MAJOR finding open. He has also ruled
that no real defect is waived. So report every real defect you find, at its true severity, and do not accept a known
defect as a "residual".

Write your findings as ordinary engineering defects: what goes wrong, a concrete reproduction or scenario, its
severity, and a fix.

## Mechanism & scope

- **Measures:** whether `9df5bef..6991939`, restricted to `tensor_cuda/` and `tc_signal/`, is correct, robust in its
  error handling, free of regressions to existing core behaviour, and ready for the main branch. Specifically:
  - Each of the 27 earlier items (F01–F07, N1–N5, V3-01, V3-02, V4-01–V4-04, V5-01–V5-04, V6-01–V6-03, V7-01, V7-02)
    is CLOSED, PARTLY closed or OPEN, with evidence.
    - For the 22 that round 7 recorded CLOSED, it is enough to confirm that the FIX7 changes at their sites did not
      reopen them.
    - Verify the 3 PARTLY items, V7-01 and V7-02 from the code and the tests.
  - The FIX7 changes introduce no new defects. That includes the native exit runners, the C-level reading of the
    exit entries' argument, the lock retry and its diagnostic, and the failure-record tags.
  - The FIX7 sweep tables (i) and (j), and the L1 addendum, are complete and their verdicts are right. Treat them as
    claims to check: look for sites they missed, and spot-check rows against the code.
- **Judge explicitly:**
  - each FIX7 user-visible change (U1–U9): acceptable, or a defect with severity;
  - each FIX7 "Not a defect" item (N1–N4): correctly not a defect in this code, or a defect with severity.
- **Mechanism doc:** `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_p1trev8/docs/P1T_FIX7_REPORT.md`
  (the claimed fixes and sweeps). Also:
  - `docs/P1T_REV7_REPORT.md` (the last complete closure table);
  - `artifacts/p1t_rev7/findings.json`;
  - `artifacts/p1t_fix7/lead/LEAD_VERIFICATION.md`;
  - `tc_signal/README.md` (the add-on contract).

  Reports from rounds 1–6 and FIX1–FIX6 exist under `docs/`. Open them only when you need one specific detail, not
  as background reading.
- **Exercises:** read the code under review, `git diff 9df5bef..6991939 -- tensor_cuda tc_signal`, and the FIX7 delta,
  `git diff 429446d..6991939 -- tensor_cuda tc_signal`.
  - Main areas:
    - core `tensor_cuda/src/bindings.cpp`: the exit entry `_dlpack_begin_shutdown`, its argument reading, the exit
      runner `_dlpack_exit_hook_runner`, `dlpack_begin_shutdown_impl` and the lock retry, and the recording and
      reporting of errors that could not be raised;
    - core `tensor_cuda/src/kernels.cu` (drain records, `fill_`) and `tensor_cuda/tensor_cuda/__init__.py` (the exit
      hook and its registration);
    - `tc_signal/src/py_args.h` (`raw_exit_diag_arg`) and `tc_signal/src/bindings.cpp` (the exit entry and runner);
    - `tc_signal/src/dlpack_operand.cpp`:
      - `FailureRecord`, `SyncReport` and `describe`;
      - `synchronize_and_release_all`;
      - `exit_close_and_wait`;
      - `exit_log_unreported`;
      - the release lifecycle;
    - `tc_signal/tc_signal/__init__.py` (the exit hook and its registration);
    - the launch and cuBLAS check helpers in `tc_signal/src/cuda_guard.h` and their call sites.
  - CPU-only checks are allowed: compile, `nm`/`ldd`/`objdump`, host-side scripts, and the earlier rounds' CPU
    checks. They are read-only: copy them to `/tmp` if you need to rebuild.
- **Does not cover:**
  - Voice's acceptance tests and tolerances, which were reviewed separately;
  - performance;
  - GPU execution: this environment has no GPU. When a claim needs one, write a short probe under
    `artifacts/p1t_rev8/probes/` with a one-line pass/fail rule, and the lead runs it before acting on the verdict.
    - The lead's host has ONE GPU, so a probe that needs two GPUs must exit 3 (INCONCLUSIVE) there. Where you can,
      also give a single-GPU form.
    - A probe must not embed a copy of product code that would stop it passing on a fixed build. Build it against the
      sources under review instead;
  - pre-existing core code that P1-T does not change, for example the engine's other bindings. These are baseline,
    not findings, unless the P1-T surface makes them reachable in a new way. If it does, say how.
- **Required evidence:**
  - the closure table for the 27 items, with status and the lines you read;
  - the sweep-table check for (i), (j) and the L1 addendum: the sites you found missing, if any, and the rows you
    spot-checked, with your verdict on each;
  - a verdict on each of U1–U9 and N1–N4;
  - every new finding with file:line, severity (BLOCKING / MAJOR / MINOR / NIT), scenario and fix;
  - the R1–R6 verdicts;
  - the probe list.
- **Operand census:**
  - zero-size and 0-d tensors;
  - extents near int64 limits;
  - non-zero `byte_offset`;
  - read-only and versioned capsules;
  - capsules consumed twice;
  - producers on non-default streams;
  - one operand passed twice in a call;
  - producer deleters that raise;
  - host allocation failure on the release and error-reporting paths;
  - argument objects of unusual types (subclasses of int, float, tuple, list and str; numpy scalars; objects with
    Python-level conversion methods), passed positionally and by keyword, to every P1-T binding and to both exit
    entries;
  - interpreter exit with releases or argument conversions in progress on other threads, with the diagnostic
    settings holding unusual values or deleted, and with an exception arriving during an exit hook;
  - `synchronize()` called while another thread's release is running;
  - CUDA errors already present, or arising once, at `synchronize()`, an op, `fill_` and export;
  - multi-device pending entries;
  - the current device different from the tensor's device.
- **Baseline isolation:**
  - The baseline is the main branch at `9df5bef`, plus the earlier rounds' findings.
  - The lead's receipts are claims to check, not facts. They are in `artifacts/p1t_fix7/lead/`:
    - Voice's runner gives 249/249;
    - core plus `tc_signal` in one session shows only the 52 known baseline failures;
    - the round-7 probes fail on 429446d and pass on FIX7;
    - the FIX7 mutants are killed by the new tests.
- **Forbidden configuration:**
  - editing anything outside `artifacts/p1t_rev8/`, `logs/` and `docs/P1T_REV8_*.md`;
  - running GPU code;
  - git writes. Read-only git is ALLOWED.

## Questions

- **R1, effect on existing core behaviour:** SAFE / RISK.
- **R2, core DLPack correctness (lifetimes, stream ordering, device handling, argument handling, error handling,
  interpreter exit):** SOUND / UNSOUND.
- **R3, the `tc_signal` operand contract (deferred release, validation, argument handling, device handling, error
  reporting, interpreter exit):** SOUND / UNSOUND.
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
- Everything outside your writable paths is READ-ONLY. That includes the main checkout `/mnt/ForgeRealm/Project-Tensor`
  (outside `Fork/`), `/mnt/ForgeRealm/Project-Voice`, and every other fork or worktree.
- Form your own view from the code before you read the fix report's verdicts.
- If you cannot settle something, say so and write the probe. Never guess a verdict.
- Prior Art Directive (AGENTS.md): name the standard or source behind a finding: the DLPack spec, the Python array
  API, CUDA stream and event semantics, the CPython 3.12 C-API and lifecycle docs, pybind11, the C++ standard, or the
  ONNX operator specs.
- **Write early.** Create `docs/P1T_REV8_REPORT.md` and `artifacts/p1t_rev8/findings.json` within your first few
  steps, and update them after each area you finish. A partial run must still leave every finding and verdict so far
  on disk.

## Done

- Write `docs/P1T_REV8_REPORT.md` and `artifacts/p1t_rev8/findings.json`. The report holds the R1–R6 table, the
  closure table, the sweep-table check, the U/N verdicts, the new findings table and the probe list. The JSON holds
  `verdicts`, `closure`, `sweep_check`, `behaviour_changes`, `not_a_defect` and `findings`.
- Your final message must contain, verbatim:
  - the R1–R6 table;
  - the closure table;
  - the U1–U9 and N1–N4 verdicts;
  - every BLOCKING and MAJOR finding in full;
  - the MINOR and NIT counts;
  - the sweep-table check summary;
  - the probe paths, with what each one decides.
