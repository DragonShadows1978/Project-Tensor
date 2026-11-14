# ENG2-REV2: narrow cross-model re-review of ENG2-A1's five fixes, before the lead's GPU run

YOUR WRITABLE TARGET is
`/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng2/artifacts/eng2/rev2/` (branch `eng2`,
at the commit carrying this order). Writing there, CPU runs, and scratch under `/tmp/eng2_rev2_*` are AUTHORIZED.
Point `TMPDIR` at the scratch directory for every command.

**Read-only:**
- everything else in this worktree, including `artifacts/eng2/rev1/` and `artifacts/eng2/a1/`;
- every other worktree;
- the production repository.

Read-only git is allowed.

**Searching:** never walk `/`, `/mnt` or `/mnt/Shared` recursively. Search inside this worktree, by path.

## What you review

A Claude seat wrote ENG2-A1 (`orders/ENG2_A1_REV1_FIXES.md`), answering your own REV1 findings E2R-F01 to F05
(`artifacts/eng2/rev1/findings.json`). The receipts are commit `1a4beed1`, under `artifacts/eng2/a1/`, and the diff is
`artifacts/eng2/a1/A1_SOURCE.diff`.

**The review is narrow.** Check only that A1 fixed the five findings correctly and introduced nothing new. Do not
re-review the N1 norm code, which REV1 found sound.

## Questions (answer each, with receipts)

1. **E2R-F01.** Is every non-string refused before any membership test, conversion, allocation, random draw or cache
   lookup, in `tensor`, every `make` and `_causal_mask`? Are all six supported strings unchanged? Is the `name=`
   parameter of `_np_dtype` safe? A1 notes a residual: `str` subclasses such as `numpy.str_` pass `isinstance`. Is
   that a defect, or acceptable? Argue it.
2. **E2R-F02.** Re-run your `probe_edges.py` and `edge_probes.py` scenarios against the fixed `evidence.py`, from
   copies under `rev2/`. Is every negative refused? Does the real candidate pin receipt (A1's P0) pass? Can any
   failure that is not TypeError, or not the exact registered message, still pass?
3. **E2R-F03.** Does every registered killer match itself, and nothing outside its killer set, over the 298-ID census?
   Re-run your matcher probe, and check A1's `MUTANT_EVAL_CPU.json` claims (MF1 12/12, MF2 48/48, survivors refused).
4. **E2R-F04.** Re-run your `cross_arm_review.py` from a copy. Do the 35 and 20 rows match with identical hashes? Are a
   changed hash, a duplicate, and a missing row each caught?
5. **E2R-F05.** Is the census complete for `tensor_cuda/tensor_cuda/` now? Is `_causal_mask`'s guard correct for all
   six supported names?
6. **Nothing new broken.** Is the scope sound: A1's `AG3_SCOPE.json` against `git diff 9452ef61 1a4beed1`? Re-run the
   candidate's CPU subset with CUDA hidden: expect 157 passed. Was the in-place re-staging of the arms faithful
   (A1 deviation 2)? Check 3 arms against `make_arm`'s recipe.

## Method

- **CPU only:** `CUDA_VISIBLE_DEVICES` empty, `nice -n 19`, one thread, every command under 10 minutes.
- **A finding** has an ID (`E2R2-Fnn`), a severity (BLOCKER, MAJOR, MINOR or NIT), the file and line, a reproduction
  or a precise argument, and a suggested fix. A question checked and found sound is reported as checked, with what
  you ran.
- **NITs follow the 11-03 rule:** they are fixed once and closed. Write `artifacts/eng2/rev2/REPORT.md` and
  `findings.json` early.

## Mechanism & scope

- **Measures:** whether ENG2-A1 fixed E2R-F01 to F05 without regressions.
- **Mechanism doc:** `orders/ENG2_A1_REV1_FIXES.md`.
- **Exercises:** the A1 diff, the fixed tool, REV1's probes (re-run from copies), and the CPU subset.
- **Does not cover:**
  - the N1 norm code;
  - the C++ dtype defects;
  - GPU execution;
  - promotion.
- **Operand census:**
  - the five findings;
  - the 15 killer sets over 298 IDs;
  - the 35 and 20 cross-arm rows;
  - the census entries.
- **Baseline isolation:** `bc02d3a1` is the engine baseline. REV1's receipts are read-only.
- **Required evidence:** `REPORT.md` and `findings.json`, with commands and outputs.
- **Forbidden configuration:**
  - edits outside the writable target;
  - GPU use;
  - git writes;
  - deleting anything.

## Seat

Codex GPT-6 Astra, reasoning xhigh. A1 was written by Claude, so this is the cross-model re-review.

**Rules:**
- No git writes (read-only git is allowed). No subagents. No background waits.
- **No deletes.** You are a Codex seat, and the Claude delete guard does not cover you: never run `rm`, `rmdir`,
  `unlink`, `git clean`, `find -delete` or an inline delete call.
- Never kill a process you did not start.

## Registration checklist (order-lint)

- **L01:** the instruments are python3 with NumPy and pytest, and read-only git. Run each with `--version` first.
- **L04:** n/a. There is no builder with fallback rungs.
- **L09:** the cross-arm comparison is by normalized ID and hash, exactly. Logs and timing are excluded.
- **L16:** the cheapest passing re-review is "fixed" without re-running. Excluded: questions 2-4 require re-runs.
- **L19:** arrays compared are checked nonzero and non-constant.
- **L25:** any emulation runs both sides in one process, with one thread.
- **L41:** the spot-checked mutants' killers are shown executing their hunks, by trace.
- **L30:** the hashes compared in question 4 are re-derived by re-running `cross_arm_review.py`. No stored hash is
  trusted without that reproduction.
- **L47:** the expected values have receipts:
  - 157 CPU IDs: `artifacts/eng2/a1/AG2_CPU.json`;
  - 298 census IDs: `artifacts/eng2/tool/ids_eng2.txt`;
  - 35 and 20 rows: `score_gpu.py`'s `CROSS_ROWS` and A1's `TOOL_CHECKS.json`;
  - MF1 12 and MF2 48: `artifacts/eng2/a1/MUTANT_EVAL_CPU.json`.
- **L56:** (a) A slip in running your own tools goes under Deviations: re-run, and continue. (b) An exception in your
  own scripts is repaired at most 3 times, each ledgered.
- **L60:** question 5 covers every dtype-string entry, and each refusal's baseline behaviour.
- **L65:** no deletes, as stated.
- **L82:** n/a. The writable target is one new folder.

## Done

Your final message gives, verbatim:
- every finding;
- every question's answer, with what you ran;
- `findings.json`'s sha256;
- a "Prior art" heading.
