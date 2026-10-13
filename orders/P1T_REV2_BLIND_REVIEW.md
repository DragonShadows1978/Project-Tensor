# P1T-REV2: round-2 blind cross-model review of core DLPack interop plus the `tc_signal` add-on, after FIX1

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_p1trev2`.
Edits there are AUTHORIZED, but only under `artifacts/p1t_rev2/`, `logs/` and `docs/P1T_REV2_*.md`. You may also
write scratch under `/tmp/p1t_rev2_*`. This is a REVIEW: do not modify any source, test or build file anywhere.

## Why

Round 1 (`docs/P1T_REV_REPORT.md`, `artifacts/p1t_rev/findings.json`) REJECTED promotion of `9df5bef..00590b7`. It
found 1 BLOCKING (F01), 4 MAJOR (F02–F05) and 2 MINOR (F06–F07) issues.

A fix seat then closed them on branch `voice-fix1`, and the lead verified the fixes at commit `d116ac0`. Per the fix
report (`docs/P1T_FIX1_REPORT.md`) and the lead's re-runs (`artifacts/p1t_fix1/lead/`):
- your round-1 probes now pass;
- Voice's runner gives PASS 236/236;
- core's suite shows only its 52 baseline failures.

The lead also fixed one test-harness bug: conflicting `argtypes` on the shared `ctypes.pythonapi` function objects.

Production promotion of this work is pre-approved by David, CONDITIONAL on this review passing. His words, verbatim:
"If the Blind Review passes, I approve the merger." So your verdict decides whether this merges into Project-Tensor
production main tonight. Be as adversarial as in round 1. Passing means APPROVE or APPROVE-WITH-NITS, with no BLOCKING
or MAJOR finding open.

## Mechanism & scope

- **Measures:** whether `9df5bef..d116ac0`, restricted to `tensor_cuda/` and `tc_signal/`, is correct, memory-safe,
  free of regressions and safe to promote. Specifically:
  - each round-1 finding F01–F07 is CLOSED, PARTLY closed, or OPEN, with evidence;
  - the FIX1 changes introduce no new defects. The deferred-release machinery is the most complex new code, so give
    it the most attention.
- **Mechanism doc:** `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_p1trev2/docs/P1T_REV_REPORT.md`
  (your round-1 findings). Also:
  - `docs/P1T_FIX1_REPORT.md` and `docs/P1T_FIX1_M1_ADDENDUM.md` (the claimed fixes);
  - `tc_signal/README.md` (the contract);
  - the approved design `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/docs/P1T_TENSOR_ALT_PLAN.md`.
- **Exercises:** read the full code under review, `git diff 9df5bef..d116ac0 -- tensor_cuda tc_signal`, and in
  particular the round-2 delta, `git diff 00590b7..d116ac0 -- tensor_cuda tc_signal`.
  - Key areas: `tc_signal/src/dlpack_operand.{h,cpp}` (RetainUntilComplete, the pending FIFO, drain,
    `synchronize_and_release_all`, bounds, fallback, leak paths, the atexit hook) and `tc_signal/src/checked_math.h`.
  - Also the plan functions in `bindings.cpp`, `lstm_bindings.cpp` and `m4_bindings.cpp`.
  - In core: `tensor_cuda/src/bindings.cpp` (checked helpers, import/export, `empty`, the finalization guard, RAII ctx)
    and `tensor_cuda/src/kernels.cu` (`drain_device_legacy_stream`, the external Storage destructor).
  - CPU-only checks are allowed: compile, `nm`/`ldd`/`objdump`, host-side scripts.
- **Does not cover:**
  - re-deriving Voice's tests or tolerances;
  - performance tuning;
  - GPU execution. Your sandbox has no GPU. When a claim needs one, write a probe under
    `artifacts/p1t_rev2/probes/`; the lead runs it before acting on your verdict.
- **Required evidence:**
  - for each of F01–F07: CLOSED, PARTLY or OPEN, with the lines you read;
  - every NEW finding with file:line, severity (BLOCKING / MAJOR / MINOR / NIT), a concrete failure scenario and a fix;
  - R1–R6 verdicts as in round 1;
  - GPU probe scripts for any claim you cannot settle on the CPU, each with a one-line pass/fail rule.
- **Operand census:** round 1's census, plus the cases the new code makes possible:
  - re-entrancy, such as a deleter that calls back into `tc_signal` while the pending list is being drained;
  - exceptions thrown inside a producer deleter during drain;
  - interpreter shutdown with entries still pending (the order of atexit against module teardown);
  - multi-device pending entries;
  - event-pool exhaustion or recycling;
  - bound-drain hysteresis;
  - an operand passed twice in one call (aliasing);
  - `synchronize()` called with an error already sticky on the context.
- **Baseline isolation:** the baseline is production main `9df5bef`, plus your own round-1 findings. The lead's
  receipts are claims to probe, not facts.
- **Forbidden configuration:**
  - editing anything outside `artifacts/p1t_rev2/`, `logs/` and `docs/P1T_REV2_*.md`;
  - running GPU code;
  - git writes. Read-only git (`diff`, `log`, `show`) is ALLOWED.

## Questions

- **R1, core regression risk:** SAFE / RISK.
- **R2, DLPack correctness and memory safety in core:** SOUND / UNSOUND.
- **R3, the `tc_signal` operand contract, including the deferred-release design:** SOUND / UNSOUND.
- **R4, kernel and numerics correctness, including the new checked arithmetic:** CORRECT / DEFECTS.
- **R5, decoupling and build:** PASS / FAIL.
- **R6, overall:** APPROVE / APPROVE-WITH-NITS / REJECT, for promotion into production main.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is immutable.
- NO subagents. No background waits. Run everything synchronously to completion.
- **No deletes, anywhere.** Agents never delete files; you are not covered by Claude's hooks, so this is on you. Never
  run `rm`, `git clean`, or anything like them.
- Never kill a process you did not start.
- Everything outside your writable paths is READ-ONLY. That includes production `/mnt/ForgeRealm/Project-Tensor`
  (outside `Fork/`), `/mnt/ForgeRealm/Project-Voice`, and every other fork or worktree.
- Form your own view from the code before you read FIX1's report.
- RED honesty: if you cannot settle something, say so and write the probe. Never guess a verdict.
- Prior Art Directive (AGENTS.md): when a finding rests on a known pitfall or standard, name the source.

## Done

- Write `docs/P1T_REV2_REPORT.md` with: the R1–R6 table; the F01–F07 closure table; the new findings table (id,
  severity, file:line, scenario, fix); and the probe list.
- Write `artifacts/p1t_rev2/findings.json`, a machine-readable copy with `verdicts`, `closure` and `findings`.
- Your final message must contain, verbatim:
  - the R1–R6 table;
  - the F01–F07 closure table;
  - every BLOCKING and MAJOR finding in full;
  - the MINOR and NIT counts;
  - the probe paths, with what each one decides.
