# GN1-REV6: narrow cross-model review of GN1 A8 under contract C, before the GPU run

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_gn1/artifacts/gn1/rev6/`
(branch `gn1`, at the commit carrying this order). Writing there, CPU runs, and scratch under `/tmp/gn1_rev6_*` are
AUTHORIZED. Point `TMPDIR` at the scratch directory.

**Read-only:** everything else in this worktree (the product, the tests, every receipt under `artifacts/gn1/`), every
other worktree, and production. Read-only git is allowed.

**The installed runner:** you may run the installed `forge-gpu-run` only on CPU children, with its summary, window
and lock redirected to scratch, as REV5 did. Never touch a GPU or the real window and lock files, and leave
`/mnt/ForgeRealm/grapa_run7` strictly alone.

**Searching:** never walk `/`, `/mnt` or `/mnt/Shared` recursively, and never enter `/mnt/Shared/ForgeRealm_Link`.
Search inside this worktree, by path.

## What you review

GN1-REV5 (`artifacts/gn1/rev5/REPORT.md`) rejected A5 on R5-F01 to R5-F05:
- F01: runner identity;
- F02: the PATH and `FORGE_GPU_*` environment;
- F03: deeply nested JSON;
- F04: `raw_exit`;
- F05: the post-exit seal hash.

The rounds since:
- **A6** fixed them, but its own control showed that mode fields edited after a run could promote a CPU root.
- **The lead's ruling** `artifacts/gn1/LEAD_RULING_A7.md` changed the scorer's contract (C, below).
- **A7** built per-line device evidence, then was stopped by the provider.
- **A8** (`orders/GN1_A8_CONTRACT.md`, receipts `artifacts/gn1/a8/`, commit `76f52a05`, ruling
  `artifacts/gn1/LEAD_RULING_A8.md`) finished it:
  - D2, positive device evidence on every line;
  - D3, demote-only modes;
  - D4, the controls classified under C;
  - G1 to G6 PASS.

The product and the test file are unchanged since `cf922891`.

This review is narrow: the `a6` to `a8` tooling delta, REV5's findings as they stand in A8, and whatever A8
introduced.

## Contract C (what this review judges against)

- **The bar:** the scorer must never accept a root unless every staged line ran the sealed programs on the device and
  met its registered expectation (`gpu_lines.json`).
- **In scope:** every root an honest run of the staged script can produce, including:
  - wrong programs, a wrong PATH or `FORGE_GPU_*` environment, or a wrapped runner;
  - a hidden or missing GPU;
  - interruptions and timeouts;
  - missing, empty, truncated or unparsable receipts.
- **Out of scope (observations, not findings):**
  - deliberate edits to well-formed receipts after the run. The lead's GPU wrapper writes and checks a sha256 manifest
    of the run root at run end, and commits it with the verdict;
  - A6's other out-of-scope items.

## Questions

1. **Each REV5 finding.** For R5-F01 to R5-F05: is the fix in A8's tooling correct and complete?
   - Re-run REV5's own reproductions (`rev5/hostile_rev5.py` and its probes) against A8's real parent, child and
     scorer.
   - Add at least one new case per MINOR finding.

   FIXED / NOT FIXED / OUT OF SCOPE (with the reason) for each.
2. **Device evidence (D2).** Can a root be accepted when any line did not run on the device? Consider at least:
   - the GPU hidden, and an empty or unset `CUDA_VISIBLE_DEVICES`;
   - a stand-in child, and a child at the right path with the wrong bytes;
   - a failed `cudaGetDevice` or device-properties query, and an empty device name.

   Also check that the child writes the evidence itself, before its tests run, and that the parent never rewrites
   it. SOUND / UNSOUND.
3. **Demote-only modes (D3).** Can any mode field in a parent receipt or in the root turn acceptance on? Is a parent
   receipt whose declared mode disagrees with its child's evidence refused? SOUND / UNSOUND.
4. **The classification (D4).**
   - Are the 32 class (iii) targets in `a8/D4_classification.md` really edits of well-formed receipts after a run?
   - Is every run-condition or abnormal-ending target in class (i) or (ii), and refused?

   CONFIRMED / NOT CONFIRMED.
5. **Regression.** Under A8's tooling:
   - the regeneration matches the seal;
   - the honest 58-line CPU run passes every line check and is refused only for lack of device evidence;
   - every mutant's witness reaches its edit site.

   SOUND / UNSOUND.
6. **Overall:** ACCEPT (no findings) or REJECT (with the findings).

## Method

- CPU only, with `CUDA_VISIBLE_DEVICES` empty except where a case sets it as an input. GPU never.
- **A finding** has an ID (`R6-Fnn`), a severity (BLOCKER, MAJOR, MINOR or NIT), the file and line, a reproduction or
  a precise argument, and a suggested fix.
- Report every real in-scope defect, NITs included. Report out-of-scope items as observations.
- **Write early:** create `artifacts/gn1/rev6/findings.json` in your first steps and keep it current.
- **The report:** write `artifacts/gn1/rev6/REPORT.md`. If your tool refuses to write a report file, do not work
  around it. Put the full report in your final message instead, and the lead will file it.

## Mechanism & scope

- **Measures:** whether A8's scorer accepts only roots whose every line ran the sealed programs on the device and met
  its registered expectation, under contract C, with REV5's findings closed.
- **Mechanism doc:**
  - `artifacts/gn1/LEAD_RULING_A7.md`;
  - `artifacts/gn1/LEAD_RULING_A8.md`;
  - `orders/GN1_A8_CONTRACT.md`;
  - `artifacts/gn1/rev5/REPORT.md`.
- **Exercises:**
  - A8's parent, child and scorer under REV5's reproductions and your own cases, with the installed runner
    redirected to scratch;
  - the device-evidence query with the GPU hidden;
  - the regeneration;
  - the mutant traces.
- **Does not cover:**
  - GPU execution (the lead's run);
  - the product and the oracle REV1 found sound;
  - C's out-of-scope items;
  - performance.
- **Required evidence:**
  - `findings.json` and the report;
  - the scripts and outputs of every case;
  - the classification check;
  - the regeneration comparison.
- **Operand census:**
  - R5-F01 to R5-F05;
  - every device-evidence field;
  - every mode field;
  - the 32 class (iii) targets;
  - every mutant.
- **Baseline isolation:** A6's and A8's receipts are read in place. Your runs write only under `rev6/` and
  `/tmp/gn1_rev6_*`.
- **Forbidden configuration:**
  - any write outside `rev6/` and `/tmp/gn1_rev6_*`;
  - editing the product, the tests or any receipt;
  - GPU use;
  - git writes;
  - deleting anything.

## Registration checklist (order-lint)

- **L01:** the instruments all exist:
  - REV5's scripts;
  - A8's tooling and receipts;
  - the installed runner;
  - NumPy 2.4.2 and pytest 9.0.2.
- **L05:** findings are attributed to the product (`nn.py`), the tests, or the GPU tooling under `artifacts/gn1/a8/`.
- **L07:** n/a. Nothing visual.
- **L09:**
  - device fields and modes are compared as exact strings and integers;
  - programs by sha256;
  - exits as integers.
- **L16:** the cheapest passing review re-reads A8's tables and agrees. Questions 1 to 5 require your own cases,
  REV5's reproductions re-run, and an independent regeneration.
- **L17:** n/a. Nothing visual.
- **L21:** the remote is public. No personal names, no raw quotes, home paths as `$HOME/...`, and no local tool paths.
- **L25:** threads at 1.
- **L26:** the lead touches nothing in this worktree while you run.
- **L33:** this review runs no GPU command. The installed runner is used only on CPU children, with its window and
  lock redirected to scratch. A8's staged GPU lines are read, not run on a GPU.
- **L34:** a Claude seat (Opus 5.5) reviews Codex's work, which makes the review cross-model.
- **L35:** the reviewer registers nothing and narrows nothing. Contract C is the lead's (`LEAD_RULING_A7.md`).
  Question 4 checks the seat's classification.
- **L40:** outputs go under `artifacts/gn1/rev6/` or `/tmp/gn1_rev6_*`.
- **L41:** question 5 checks that every mutant's witness reaches its edit site.
- **L42:** budget 60 minutes. If it runs out, report what was done and what was not.
- **L45:** your runs record each child's exit after it exits.
- **L48:** where you exercise A8's staged lines on CPU, run them exactly as written, with only the runner redirected
  as above.
- **L58:** `CUDA_VISIBLE_DEVICES` is empty for every run, except as a case input in question 2.
- **L65:** n/a. You are a Claude Agent-tool seat, and the delete guard covers you. Delete nothing anyway. Scratch goes
  under `/tmp`.
- **L67:** the receipt budget is measured from earlier rounds. REV5 kept 396 KB in the worktree and wrote its runs to
  `/tmp`. A8's whole round wrote 843 MB. The worst case here is about 2 GB, mostly in `/tmp` on `/` (58 GB free when
  registered). Stop and report if `/` falls below 20 GB free or `/mnt/ForgeRealm` below 25 GB.
- **L70:** n/a. No bound in this review separates rounding from a defect between two device computations.
- **L73:** contract C excludes deliberate edits to well-formed receipts after the run. The lead's sha256 manifest
  covers them: it is written and checked at run end, and committed with the verdict.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is immutable.
- NO git writes. NO subagents. No edits outside `artifacts/gn1/rev6/`. Never kill a process you did not start.
- Prior Art Directive: name your sources for any method you use.

## Done

Your final message contains:
- the verdict;
- every finding in full;
- what you checked and ran for each of questions 1 to 6;
- the files you wrote.
