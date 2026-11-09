# GN1-REV2: cross-model re-review of GN1 A2 (the REV1 fixes) before the GPU run

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_gn1/artifacts/gn1/rev2/`
(branch `gn1`, at the commit carrying this order). Writing there, CPU runs, and scratch under `/tmp/gn1_rev2_*` are
AUTHORIZED. Point `TMPDIR` at the scratch directory.

**Read-only:** everything else in this worktree (the product, the tests, every receipt under `artifacts/gn1/`), every
other worktree, and production. Read-only git is allowed.

**Searching:** never walk `/`, `/mnt` or `/mnt/Shared` recursively, and never enter `/mnt/Shared/ForgeRealm_Link`.
Search inside this worktree, by path.

## What you review

GN1-REV1 (`artifacts/gn1/rev1/REPORT.md`, `findings.json`) rejected the GroupNorm fix with findings R1-F01 to
R1-F11. A Codex seat answered them in amendment A2:
- the order: `orders/GN1_A2_REV1_FIXES.md`;
- the commit: `bd30498a`;
- the receipts: `artifacts/gn1/a2/`, with the fix table in `FINDINGS_FIXED.md` and the report in `REPORT.md`.

Some receipts are on disk but not committed: `G4_hostile.json` and the arm, hostile-run and audit trees. They are
listed in `a2/LEAD_HELD_BACK.txt`.

This review is narrower than REV1. It covers:
- the A2 delta (`a2/changes_vs_a1.patch`, `a2/nn_vs_0c6a4244.diff`, and A2's tooling compared with A1's);
- whether each REV1 finding is truly closed;
- whatever A2 introduced.

Do not re-review what REV1 found sound unless A2 changed it. That covers the float64 oracle, the forward and backward
math, and the honest-error bounds.

## Questions

1. **Each REV1 finding.** For R1-F01 to R1-F11: is the fix correct and complete? Re-run REV1's own hostile cases (H1
   to H6, `rev1/hostile_scorer.py`) against A2's real parent and scorer. Add at least one new case per MAJOR finding
   that A2's controls did not try. FIXED / NOT FIXED for each.
2. **D5 validation.** Does GroupNorm now refuse exactly the inputs outside its domain? Check:
   - non-positive and non-integer group and channel counts, and `bool`;
   - rank < 2, a channel mismatch, and non-float inputs and parameters.

   Does it still accept everything it should? Check every dtype combination A1 accepted, and parameters that need no
   gradient. Compare with A2's PyTorch probe (`a2/pytorch_behavior.json`). SOUND / UNSOUND.
3. **The kill rule and the device veto.** Can a mutant be counted as killed without a real GroupNorm assertion or
   refusal? Can an honest kill be refused? For example: a device-pattern word inside an honest assertion message, or
   a legitimate failed and error dual JUnit entry. SOUND / UNSOUND.
4. **The scorer.** Can a forged or stale receipt still produce `G5 passed`? Can an honest run be refused? Check:
   - a retried label;
   - an exit 124 from each cause;
   - the run-root guard.

   SOUND / UNSOUND.
5. **Mutants.** Is each mutant single-site with a traced witness? Is V3's equivalence proof (`a2/EQUIVALENCES.md`)
   correct? Name any decision in the A2 code that no mutant covers. SOUND / UNSOUND.
6. **Arms and regeneration.** Does `regenerate_arms.py` rebuild the arms byte-identically from the commit and the pin?
   Run it into scratch and compare the sha256 maps. SOUND / UNSOUND.
7. **Overall:** ACCEPT (no findings) or REJECT (with the findings).

## Method

- CPU only, with `CUDA_VISIBLE_DEVICES` empty. GPU never.
- **A finding** has an ID (`R2-Fnn`), a severity (BLOCKER, MAJOR, MINOR or NIT), the file and line, a reproduction
  or a precise argument, and a suggested fix.
- Report every real defect, NITs included. The house rule is no waivers.
- **Write early:** create `artifacts/gn1/rev2/findings.json` in your first steps and keep it current.
- **The report:** write `artifacts/gn1/rev2/REPORT.md`. If your tool refuses to write a report file, do not work
  around it. Put the full report in your final message instead, and the lead will file it.

## Mechanism & scope

- **Measures:** whether A2 closes every REV1 finding without new defects, so that the GPU run can only report real
  kills and real regressions, and GroupNorm refuses exactly the inputs outside its domain.
- **Mechanism doc:**
  - `artifacts/gn1/rev1/REPORT.md`;
  - `orders/GN1_A2_REV1_FIXES.md`;
  - `artifacts/gn1/a2/REPORT.md`;
  - `artifacts/gn1/a2/FINDINGS_FIXED.md`.
- **Exercises:**
  - A2's class and validation on CPU;
  - A2's parent, child and scorer under REV1's and your own hostile inputs;
  - the mutant traces;
  - the arm regeneration.
- **Does not cover:**
  - GPU execution;
  - the oracle and the math REV1 found sound;
  - other layers;
  - performance.
- **Required evidence:**
  - `findings.json` and the report;
  - the scripts and outputs of every hostile run and probe;
  - the regeneration comparison.
- **Operand census:**
  - R1-F01 to R1-F11;
  - H1 to H6, plus your new cases;
  - every D5 refusal and acceptance class;
  - every mutant, V1 to V3 included.
- **Baseline isolation:** REV1's receipts and A2's are read in place. Your runs write only under `rev2/` and
  `/tmp/gn1_rev2_*`.
- **Forbidden configuration:**
  - any write outside `rev2/` and `/tmp/gn1_rev2_*`;
  - editing the product, the tests or any receipt;
  - GPU use;
  - git writes;
  - deleting anything.

## Registration checklist (order-lint)

- **L01:** the instruments all exist: REV1's scripts, A2's tooling and receipts, NumPy, and pytest 9.0.2.
- **L05:** findings are attributed to the product (`nn.py`), the tests, or the GPU tooling under `artifacts/gn1/a2/`.
- **L07:** n/a. Nothing visual.
- **L09:**
  - JUnit outcomes are compared ID by ID as exact strings;
  - arms are compared by sha256 maps;
  - exits are compared as integers.
- **L16:** the cheapest passing review re-reads A2's control table and agrees. Questions 1, 3, 4 and 6 require your own
  hostile runs, your new cases, and an independent regeneration.
- **L17:** n/a. Nothing visual.
- **L21:** the remote is public. No personal names, no raw quotes, home paths as `$HOME/...`, and no local tool paths.
- **L25:** threads at 1.
- **L26:** the lead touches nothing in this worktree while you run.
- **L34:** a Claude seat (Opus 5.5) reviews Codex's work, which makes the review cross-model.
- **L40:** outputs go under `artifacts/gn1/rev2/` or `/tmp/gn1_rev2_*`.
- **L41:** question 5 checks that every mutant's witness reaches its edit site.
- **L42:** budget 60 minutes. If it runs out, report what was done and what was not.
- **L45:** your hostile runs record each child's exit after it exits.
- **L58:** `CUDA_VISIBLE_DEVICES` is empty for every run.
- **L65:** n/a. You are a Claude Agent-tool seat, and the delete guard covers you. Delete nothing anyway. Scratch goes
  under `/tmp`.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is immutable.
- NO git writes. NO subagents. No edits outside `artifacts/gn1/rev2/`. Never kill a process you did not start.
- Prior Art Directive: name your sources for any method you use.

## Done

Your final message contains:
- the verdict;
- every finding in full;
- what you checked and ran for each of questions 1 to 6;
- the files you wrote.
