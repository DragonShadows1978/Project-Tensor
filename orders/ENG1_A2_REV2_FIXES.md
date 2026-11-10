# ENG1-A2: fix ENG1-REV2's two findings (complete the source-reader exclusion list; record non-finite census patterns)

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng1`
(branch `eng1`, at the commit carrying this order). Edits and CPU runs are AUTHORIZED, limited to:
- `artifacts/eng1/tool/source_reading_tests.json`, the staged census and selection files that `stage.py` writes, and
  a new seal file;
- `artifacts/eng1/a1/SOURCE_READING_TESTS.md`. Supersede it with an `a2/` copy, and leave the original as it is;
- `tensor_cuda/tests/test_eng1.py`, the E3 census row functions only (`e3_distance` and the row writer);
- new files under `artifacts/eng1/a2/`;
- `logs/`, and scratch under `/tmp/eng1_a2_*`.

**Read-only:** the engine sources (the candidate pin `3e6d5805…` stays), every other test, and every earlier
receipt, including `rev1/`, `a1/` and `rev2/`. Every other worktree and production are read-only too. Read-only git is
allowed.

**Searching:** never walk `/`, `/mnt` or `/mnt/Shared` recursively, and never enter `/mnt/Shared/ForgeRealm_Link`.

## The fixes (binding)

1. **R2-F01, the incomplete exclusion list.** Add the 6 tests (7 IDs) that ENG1-REV2 found, with the path each reads,
   to the source-reader list and to the `a2/` table. The IDs are in `artifacts/eng1/rev2/REPORT.md` and
   `findings.json`.
   - Rerun the `stage.py` census. Following the review, the comparison set becomes 2110 IDs. Report the exact count.
   - Seal the staging into a new seal file. The old seals stay.
   - Run one line in audit mode on CPU with the stand-in.
   - **Completeness check:** rerun ENG1-REV2's runtime read trace (`artifacts/eng1/rev2/q2_source_readers.py` and
     `q2_classify_reads.py`, as they stand, with output into `a2/`) against the new list. Show that 0 observed source
     readers are missing from it. List any reader observed only by name, with its reason.
2. **R2-F02, empty distance fields for non-finite rows.** In the E3 census rows, also record:
   - `nonfinite_pattern_equal`, computed as
     `np.array_equal(np.where(np.isfinite(got),0,got), np.where(np.isfinite(emu),0,emu), equal_nan=True)`;
   - the maximum distance over the elements that are finite in both vectors (null when there are none).

   E3 stays measurement only. This is a NIT, fixed once and closed under the 11-03 rule.

**After the fixes:**
- rerun the CPU subset on the candidate with CUDA hidden. Expect 47 passed;
- show `git diff 7cf46bd5 -- . ':!artifacts/eng1' ':!orders' ':!logs'` touches only `test_eng1.py`;
- show `tensor_cuda/src/` is byte-identical.

**Acceptance:** ACCEPT when all of these hold:
- the completeness check shows 0 missing;
- the census count and the new seal are recorded;
- the audit line passes;
- the CPU subset passes 47 of 47;
- the scope holds.

The lead closes R2-F01 by rerunning that same completeness check. Both findings are mechanical, so there is no
further review round.

## Mechanism & scope

- **Measures:** whether the source-reader exclusion list is complete under a runtime read trace, and whether the E3
  rows record the non-finite pattern and the finite distance.
- **Mechanism doc:**
  - `artifacts/eng1/rev2/REPORT.md`;
  - `artifacts/eng1/rev2/findings.json`;
  - `orders/ENG1_A1_REV1_FIXES.md`;
  - `artifacts/eng1/a1/REPORT.md`.
- **Exercises:**
  - the list;
  - the stage census;
  - the seal;
  - one audit line;
  - REV2's read-trace scripts;
  - the E3 row functions on CPU;
  - the CPU subset.
- **Does not cover:**
  - engine sources;
  - GPU execution (G5);
  - promotion.
- **Required evidence:**
  - the diff;
  - the `a2/` table;
  - the census count;
  - the new seal's sha256;
  - the audit line;
  - the completeness-check output;
  - the CPU subset result;
  - `a2/REPORT.md` and `a2/LEDGER.md`.
- **Operand census:**
  - the 22 listed source readers (16 + 6);
  - 2110 comparison IDs;
  - 4 non-finite E3 rows;
  - 47 CPU IDs.
- **Baseline isolation:** both arms keep their pins, and only the list, the staging files and the E3 row writer change.
- **Forbidden configuration:**
  - engine source edits;
  - editing other tests;
  - editing earlier receipts;
  - GPU use;
  - git writes;
  - deleting anything.

## Registration checklist (order-lint)

- **L01:** the instruments are the worktree at the dispatch commit, `stage.py`, REV2's trace scripts, NumPy and
  pytest. All exist at `2439d339`.
- **L05:** the owners are the tooling list (F01) and the test file (F02).
- **L07:** n/a. Nothing visual.
- **L09:** test IDs are exact strings. The seal and pins are sha256 strings. The pattern-equality field is the
  stated NumPy comparison.
- **L16:** the cheapest pass adds the 6 names without rerunning the trace. The completeness check requires the trace
  output.
- **L21:** the remote is public. No personal names, and home paths are written as `$HOME/...`.
- **L25:** threads at 1.
- **L26:** the lead touches nothing in this worktree while the seat runs.
- **L34:** a Codex seat (gpt-6.1-sol, xhigh), on CPU. Closure is by the lead's rerun of the trace, not another
  review round, because both findings are mechanical.
- **L40:** outputs go under `artifacts/eng1/a2/`, a new seal file, or `/tmp/eng1_a2_*`. Earlier receipts stay as
  they are.
- **L42:** budget 90 minutes.
- **L45:** the audit line counts only from the parent's record of the child's exit, written after the child exits.
- **L46:** record a sha256 map of `tensor_cuda/` before the first write. Show it unchanged except `test_eng1.py`.
- **L47:** the expected values have receipts: the 6 tests and 2110 IDs are in `rev2/REPORT.md`, and the 47 CPU IDs
  in `a1/gates.json`.
- **L58:** `CUDA_VISIBLE_DEVICES` is empty for every run.
- **L65:** no deletes. Anything that must go stays where it is and is listed in the report, and scratch goes under
  `/tmp`.
- **L82:** the write list is the files the two fixes touch, and it was named from REV2's findings, which cite each
  file and line.
- **L80:** the scope diff is against `7cf46bd5`, A1's receipts commit. The dispatch commit and REV2's receipts
  (`2439d339`) add only orders and `artifacts/eng1`, which the diff excludes. Name the dispatch hash in the report.
- **L87, L92, L93, L94, L95:** n/a. No seeds, no gradient check, no graph-position sample, no null arm, no fitted
  bins.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is immutable.
- NO git writes. NO subagents. GPU never. Never kill a process you did not start.
- **Prior Art Directive:** REV2's annotations stand. The read trace uses PEP 578 audit hooks; credit REV2's scripts.

## Done

Your final message contains:
- the ACCEPT or REJECT result with the receipts;
- the diff;
- the census count;
- the new seal's sha256;
- the completeness-check output (0 missing, or the list);
- the CPU subset result;
- every file written.
