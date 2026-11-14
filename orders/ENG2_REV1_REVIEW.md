# ENG2-REV1: cross-model review of ENG2 (dtype refusal and float32 half norms), before the lead's GPU run

YOUR WRITABLE TARGET is
`/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng2/artifacts/eng2/rev1/` (branch `eng2`,
at the commit carrying this order). Writing there, CPU runs, and scratch under `/tmp/eng2_rev1_*` are AUTHORIZED.
Point `TMPDIR` at the scratch directory for every command.

**Read-only:**
- everything else in this worktree;
- every other worktree;
- the production repository.

Read-only git is allowed.

**Searching:** never walk `/`, `/mnt` or `/mnt/Shared` recursively, and never enter `/mnt/Shared/ForgeRealm_Link`. Search
inside this worktree, by path.

## What you review

A Claude seat (Opus 5.5) wrote ENG2 under `orders/ENG2_DTYPE_REFUSAL_AND_HALF_NORMS.md`. Its receipts are commit
`2237299b`, under `artifacts/eng2/`. Its report, the seat's final message, sits in the lead's commit message and in
`artifacts/eng2/LEDGER.md`.
- **The source change:** `git diff bc02d3a1 2237299b -- tensor_cuda/tensor_cuda` touches `__init__.py` and `nn.py`.
- **New tests:** `tensor_cuda/tests/test_eng2.py`, 262 IDs.
- **Tooling:** ENG1's tool, copied to `artifacts/eng2/tool/`, with changes in `artifacts/eng2/tool_changes.diff` and
  `COPY_DEPS.json`.

## Questions (answer each, with receipts)

1. **F1, the refusal.**
   - Is every unsupported dtype string, and every non-string, refused with TypeError before any allocation or random
     draw, in `tc.tensor` and every `_factory` maker?
   - Are all 6 supported dtypes unchanged, and is the default path with float64 data unchanged?
   - Check the seat's census (`artifacts/eng2/L60_dtype_census_*.json`) against the code, for 5 entries of your
     choice.
   - Is any other Python entry in `tensor_cuda/tensor_cuda/` still a silent fallback?
2. **N1, the four forwards.**
   - For float16 and bfloat16 inputs, are the statistics computed in float32, with one cast back?
   - Is the float32 path's op sequence exactly the baseline's? Show it by diffing the op sequence, not by trusting the
     comments.
   - Is BatchNorm1D's running-statistics update still the same rule in float32?
   - Does autograd through the new `astype` calls give the right gradient dtype?
3. **Tests.**
   - Are the E3 helpers imported unchanged from `test_eng1.py`, with tolerances not re-derived?
   - Are the two extra test groups (`test_n1_half_rule_bits`, `test_n1_batchnorm_running_stats`) sound, and is the
     derivation of the running-statistics tolerance correct?
   - Do the baseline's predicted failures (106 IDs: 60 F1 on CPU and 46 N1 from emulation) follow from the code?
4. **Mutants.** Are all 15 single-edit mutants valid? Is each predicted killer plausible? Spot-check 4 mutants by
   reasoning or a CPU emulation.
5. **Tool changes.** Does the copied tool keep:
   - the parent-after-child receipt property;
   - device-mode seal verification (the ENG1-A3 lesson);
   - dependency checks?

   Is the new L38 pin branch in `evidence.regression_reasons` narrow? It may exempt only the 10 pinned
   `test_eng1.py` IDs, and only for their registered failure, the TypeError text. Construct a receipt where a pinned
   ID fails for another reason: the evaluator must refuse it.
6. **The G3 build explanation.** Re-run the seat's SASS comparison (`cuobjdump -sass`, module IDs normalized) on the
   base and cand modules, and confirm that it is identical.

## Method

- **CPU only:** `CUDA_VISIBLE_DEVICES` empty, `nice -n 19`, one thread, every command under 10 minutes.
- **A finding** has an ID (`E2R-Fnn`), a severity (BLOCKER, MAJOR, MINOR or NIT), the file and line, a reproduction or a
  precise argument, and a suggested fix. A question checked and found sound is reported as checked, with what you ran.
- **NITs follow the 11-03 rule:** they are fixed once and closed, with no further review round.
- Write `artifacts/eng2/rev1/REPORT.md` and `artifacts/eng2/rev1/findings.json` early.

## Mechanism & scope

- **Measures:** whether ENG2's source change, tests, mutants and copied tool are correct and complete before its GPU
  run.
- **Mechanism doc:** `orders/ENG2_DTYPE_REFUSAL_AND_HALF_NORMS.md`.
- **Exercises:** reading and CPU checks of the ENG2 diff, tests and tool.
- **Does not cover:**
  - GPU execution;
  - the C++ defects ENG2 recorded (numpy_dtype, bool readback, device strings);
  - promotion.
- **Operand census:**
  - the two source files;
  - `test_eng2.py`;
  - the 15 mutants;
  - the tool diff.
- **Baseline isolation:** `bc02d3a1` is the baseline.
- **Required evidence:** `REPORT.md` and `findings.json`, with each check's commands and outputs.
- **Forbidden configuration:**
  - edits outside the writable target;
  - GPU use;
  - git writes;
  - deleting anything.

## Seat

Codex GPT-6 Astra, reasoning xhigh. ENG2 was written by Claude, so this is the cross-model review.

**Rules:**
- No git writes (read-only git is allowed). No subagents. No background waits. Run every check in the foreground.
- **No deletes.** You are a Codex seat, and the Claude delete guard does not cover you: never run `rm`, `rmdir`,
  `unlink`, `git clean`, `find -delete` or an inline delete call.
- Never kill a process you did not start. Other seats and the lead's KA1 and MEM1 work may be running.

## Registration checklist (order-lint)

- **L01:** the instruments are python3 with NumPy and pytest, `cuobjdump` from CUDA 12.6, and read-only git. Run each
  with `--version` first.
- **L09:** question 6 compares SASS text after replacing module-ID strings with one token. Every other line is
  compared exactly.
- **L16:** the cheapest passing review is "checked, no findings" without running anything. Excluded: each answer names
  what was run, or the `file:line` read, and question 5 requires a constructed refusal.
- **L19:** any array compared is checked nonzero and non-constant first.
- **L25:** any bitwise or emulation check runs both sides in one process, with one thread.
- **L04:** n/a. "Fallback" means the silent dtype fallback under review; there is no builder with fallback rungs.
- **L41:** the review runs no registered gate. For the mutants it spot-checks (question 4), it shows, by a CPU
  emulation or a traced test run, that the predicted killer executes the mutated line. The N1 kills on the device are
  the lead's G5.
- **L56:**
  - (a) A slip in running your own tools goes under Deviations: re-run, and continue.
  - (b) An exception in your own scripts is repaired at most 3 times, each ledgered.
  - A review has no registered data gate.
- **L60:** question 1's census covers every dtype-string entry in `tensor_cuda/tensor_cuda/`, with its baseline
  behaviour for "float64", "double", "int32", an unknown string and a non-string.
- **L65:** no deletes, as stated.
- **L82:** n/a. The writable target is one new folder, and no product code is edited.

## Done

Your final message gives, verbatim:
- every finding;
- every question's answer, with what you ran;
- `findings.json`'s sha256;
- a "Prior art" heading for any technique you cite or propose.
