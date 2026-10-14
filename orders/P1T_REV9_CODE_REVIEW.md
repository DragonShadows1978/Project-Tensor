# P1T-REV9: independent code review (round 9) of TensorCUDA's DLPack interop and the `tc_signal` add-on package

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_p1trev9`.
Edits there are AUTHORIZED, but only under `artifacts/p1t_rev9/`, `logs/` and `docs/P1T_REV9_*.md`. You may also
write scratch under `/tmp/p1t_rev9_*`. This is a CODE REVIEW: do not modify any source, test or build file anywhere.

## Context

TensorCUDA is a CUDA tensor engine. To let a text-to-speech port (Project-Voice) run on it, two things were added:
- **Core:** a small interop surface: DLPack export/import, `data_ptr`, `fill_`, `mark_modified`, and a device-side
  `empty`.
- **Add-on:** a separate package, `tc_signal/`, with 1-D convolutions, an ONNX-semantics LSTM, three normalizations,
  nearest upsample, reflection pad and round-half-to-even. It exchanges tensors with core only through DLPack.

This is the ninth round of independent review before the work enters the project's main branch.

- **The last complete closure table** is round 8's, in `docs/P1T_REV8_REPORT.md` and `artifacts/p1t_rev8/findings.json`.
  - It recorded V7-01, V7-02, N5, V3-01 and V5-01 CLOSED, and N1 PARTLY closed again.
  - It accepted FIX7's U1–U4 and U6–U9, and all four of its "Not a defect" items.
  - It found two new defects, both MINOR and about error visibility at interpreter exit:
    - V8-01: the exit device-wait marker erased its own CUDA failures. A failed create or record fell back to an
      undiagnosed wait, and "finished" could be printed without proof;
    - V8-02: the native exit loggers wrote nothing when building the final report failed to allocate.
- **FIX8** (`docs/P1T_FIX8_REPORT.md`, `docs/P1T_FIX8_LEDGER.md`) fixed both. It also swept two defect classes across
  all P1-T code:
  - (k) CUDA errors consumed without being recorded first;
  - (l) the last reader of recorded failures at exit, which needs an allocation-free fallback.
- **The lead sent one item back before this review: L2.** The exit wait's new watchdog thread could be detached while
  still running, and a failed stop lock could cost a silent extra wait. It is fixed; see the report's "Lead send-back
  L2" section.
- **Ruling on one probe.** The round-8 GPU probe `diagnostic_events_gpu.py`, run as written, fails on FIX8 for a
  reason outside the product. Under torch 2.11, `torch.cuda.ExternalStream(0)` is a non-blocking stream, so there is
  no legacy-stream exit wait to diagnose. The lead reproduced this independently. A one-line copy that queues the work
  on the legacy stream fails on 6991939 and passes on FIX8, and the lead accepted it as the evidence (FIX8 D2 / D12).
- **A probe-harness race, not a product finding.** The round-8 probe `fix7_cpu_census.py` failed 2 of 3 runs on
  FIX8 and passed 36/36 on the third. The test script's daemon thread prints `REFUSED` while its main thread
  prints `RESULT {…}`. Python's `print` writes the text and its newline separately, so the lines can interleave as
  `REFUSEDRESULT {…}`, and the probe accepts only lines that start with `RESULT `. The interleaved JSON meets every
  check. The receipts are in `artifacts/p1t_fix8/lead/LEAD_VERIFICATION.md`.
- **The FIX8 report also holds:**
  - a "Not a defect (with justification)" table (N1–N3);
  - a list of user-visible changes (U1–U8);
  - FIX7's corrected rows (i24, i25, j24, a3, a9).
- **The lead's verification** is `artifacts/p1t_fix8/lead/LEAD_VERIFICATION.md`.

The project owner, David, approved the merge on this condition, verbatim: "If the Blind Review passes, I approve the
merger." The review passes on APPROVE or APPROVE-WITH-NITS, with no BLOCKING or MAJOR finding open. He has also ruled
that no real defect is waived. So report every real defect you find, at its true severity, and do not accept a known
defect as a "residual".

Write your findings as ordinary engineering defects: what goes wrong, a concrete reproduction or scenario, its
severity, and a fix.

## Mechanism & scope

- **Measures:** whether `9df5bef..e4cf1d6`, restricted to `tensor_cuda/` and `tc_signal/`, is correct, robust in its
  error handling, free of regressions to existing core behaviour, and ready for the main branch. Specifically:
  - Each of the 29 earlier items (F01–F07, N1–N5, V3-01, V3-02, V4-01–V4-04, V5-01–V5-04, V6-01–V6-03, V7-01, V7-02,
    V8-01, V8-02) is CLOSED, PARTLY closed or OPEN, with evidence.
    - For the 26 that round 8 recorded CLOSED, it is enough to confirm that the FIX8 changes at their sites did not
      reopen them.
    - Verify N1, V8-01 and V8-02 from the code and the tests.
  - The FIX8 changes introduce no new defects. That includes:
    - the recording of marker failures;
    - the watchdog thread and its stop protocol;
    - the allocation-free summaries;
    - the acknowledged exit reports;
    - the runners acting as last reader in every case;
    - every raise building its report before it consumes the CUDA error slot.
  - The FIX8 sweep tables (k) and (l), and the L2 addendum, are complete and their verdicts are right. Treat them as
    claims to check: look for sites they missed, and spot-check rows against the code.
- **Judge explicitly:**
  - each FIX8 user-visible change (U1–U8): acceptable, or a defect with severity;
  - each FIX8 "Not a defect" item (N1–N3): correctly not a defect in this code, or a defect with severity;
  - the lead's ruling on `diagnostic_events_gpu.py` (FIX8 D2 / D12): sound or not.
- **Mechanism doc:** `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_p1trev9/docs/P1T_FIX8_REPORT.md`
  (the claimed fixes and sweeps). Also:
  - `docs/P1T_REV8_REPORT.md` (the last complete closure table);
  - `artifacts/p1t_rev8/findings.json`;
  - `artifacts/p1t_fix8/lead/LEAD_VERIFICATION.md`;
  - `tc_signal/README.md` (the add-on contract).

  Reports from rounds 1–7 and FIX1–FIX7 exist under `docs/`. Open them only when you need one specific detail, not
  as background reading.
- **Exercises:** read the code under review, `git diff 9df5bef..e4cf1d6 -- tensor_cuda tc_signal`, and the FIX8 delta,
  `git diff 6991939..e4cf1d6 -- tensor_cuda tc_signal`.
  - Main areas:
    - `tc_signal/src/dlpack_operand.cpp`:
      - `wait_device_work_diagnosed`, `note_exit_marker_failure` and `DeviceWaitWatchdog` (with `stop()` and `run()`);
      - `sync_device_legacy`;
      - `summarize_unreported`, `write_summary_line` and `exit_log_unreported`;
      - `note_raised_at_exit` and the exit-report acknowledgement;
      - the release lifecycle;
    - `tc_signal/src/bindings.cpp` (the exit runner as last reader, `ascii_message_of`) and `tc_signal/src/cuda_guard.h`
      (the raise helpers that consume the error slot only after their report exists);
    - core `tensor_cuda/src/bindings.cpp`: `write_core_summary`, `core_exit_last_reader`, the exit runner and
      `dlpack_raise_unraisable`;
    - core `tensor_cuda/src/kernels.cu`: `dlpack_unreported_unraisable_raw`, `throw_cuda_report_after_consuming` and
      its callers;
    - both packages' `__init__.py` exit hooks (the report acknowledgements).
  - CPU-only checks are allowed: compile, `nm`/`ldd`/`objdump`, host-side scripts, and the earlier rounds' CPU
    checks. They are read-only: copy them to `/tmp` if you need to rebuild.
- **Does not cover:**
  - Voice's acceptance tests and tolerances, which were reviewed separately;
  - performance;
  - GPU execution: this environment has no GPU. When a claim needs one, write a short probe under
    `artifacts/p1t_rev9/probes/` with a one-line pass/fail rule, and the lead runs it before acting on the verdict.
    - The lead's host has ONE GPU, so a probe that needs two GPUs must exit 3 (INCONCLUSIVE) there. Where you can,
      also give a single-GPU form.
    - A probe must not embed a copy of product code that would stop it passing on a fixed build. Build it against the
      sources under review instead;
  - pre-existing core code that P1-T does not change, for example the engine's other bindings. These are baseline,
    not findings, unless the P1-T surface makes them reachable in a new way. If it does, say how.
- **Required evidence:**
  - the closure table for the 29 items, with status and the lines you read;
  - the sweep-table check for (k), (l) and the L2 addendum: the sites you found missing, if any, and the rows you
    spot-checked, with your verdict on each;
  - a verdict on each of U1–U8 and N1–N3, and on the D2 / D12 probe ruling;
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
  - CUDA call failures inside diagnostic-only code at exit, and host allocation failure while the final report at
    exit is built;
  - failures of thread creation, mutex locking and thread joining on the exit paths;
  - `synchronize()` called while another thread's release is running;
  - CUDA errors already present, or arising once, at `synchronize()`, an op, `fill_` and export;
  - multi-device pending entries;
  - the current device different from the tensor's device.
- **Baseline isolation:**
  - The baseline is the main branch at `9df5bef`, plus the earlier rounds' findings.
  - The lead's receipts are claims to check, not facts. They are in `artifacts/p1t_fix8/lead/`:
    - Voice's runner gives 249/249;
    - core plus `tc_signal` in one session shows only the 52 known baseline failures;
    - the round-8 probes fail on 6991939 and pass on FIX8 (for `diagnostic_events_gpu.py`, the legacy-stream copy);
    - the FIX8 mutants are killed by the new tests.
- **Forbidden configuration:**
  - editing anything outside `artifacts/p1t_rev9/`, `logs/` and `docs/P1T_REV9_*.md`;
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
- **Write early.** Create `docs/P1T_REV9_REPORT.md` and `artifacts/p1t_rev9/findings.json` within your first few
  steps, and update them after each area you finish. A partial run must still leave every finding and verdict so far
  on disk.

## Done

- Write `docs/P1T_REV9_REPORT.md` and `artifacts/p1t_rev9/findings.json`. The report holds the R1–R6 table, the
  closure table, the sweep-table check, the U/N verdicts, the new findings table and the probe list. The JSON holds
  `verdicts`, `closure`, `sweep_check`, `behaviour_changes`, `not_a_defect` and `findings`.
- Your final message must contain, verbatim:
  - the R1–R6 table;
  - the closure table;
  - the U1–U8 and N1–N3 verdicts, and the D2 / D12 ruling;
  - every BLOCKING and MAJOR finding in full;
  - the MINOR and NIT counts;
  - the sweep-table check summary;
  - the probe paths, with what each one decides.
