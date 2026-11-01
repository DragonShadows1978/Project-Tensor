# GN1-REV3: cross-model re-review of GN1 A3 (the REV2 fixes) before the GPU run

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_gn1/artifacts/gn1/rev3/`
(branch `gn1`, at the commit carrying this order). Writing there, CPU runs, and scratch under `/tmp/gn1_rev3_*` are
AUTHORIZED. Point `TMPDIR` at the scratch directory.

**Read-only:** everything else in this worktree (the product, the tests, every receipt under `artifacts/gn1/`), every
other worktree, and production. Read-only git is allowed.

**The installed runner:** you may run the installed `forge-gpu-run` only as REV2 did. Redirect its summary, window
and lock to scratch (`rev2/rev2_forge.py`), and give it a CPU child. Never touch a GPU or the real window and lock
files.

**Searching:** never walk `/`, `/mnt` or `/mnt/Shared` recursively, and never enter `/mnt/Shared/ForgeRealm_Link`.
Search inside this worktree, by path.

## What you review

GN1-REV2 (`artifacts/gn1/rev2/REPORT.md`, `findings.json`) rejected A2 with findings R2-F01 to R2-F11, 4 MINOR and
7 NIT. REV1's MAJORs were already closed. A Codex seat answered REV2 in amendment A3:
- the order: `orders/GN1_A3_REV2_FIXES.md`, which carries the lead's ruling on R2-F02 (option (a), its D2);
- the commit: `cf922891`;
- the receipts: `artifacts/gn1/a3/`, with the report in `REPORT.md` and the execution amendments in `AMENDMENTS.md`.

Some receipts are on disk but not committed: the arms, the run trees and the large JSON files. They are listed in
`a3/LEAD_HELD_BACK.txt`.

This review is narrow. It covers:
- the A3 delta (A3's tooling against A2's, and `nn.py` against `bd30498a`);
- whether each REV2 finding is truly closed;
- the two execution amendments A3 disclosed;
- whatever A3 introduced.

Do not re-review what REV1 and REV2 found sound unless A3 changed it.

## Questions

1. **Each REV2 finding.** For R2-F01 to R2-F11: is the fix correct and complete?
   - Re-run REV2's own reproductions against A3's real parent, child and scorer: B1 to B5 and C7 through the installed
     runner redirected to scratch (`rev2/rev2_forge.py`), P1 and P2 (`rev2/q3_probe.py`), `d5_probe.py`, and the
     uncovered-decision probe.
   - Add at least one new case per MINOR finding.
   - FIXED / NOT FIXED for each.
2. **The kill rule under the lead's ruling (A3's D2).**
   - Does a mutant's engine refusal count as a kill only when the candidate passed the same input in the same root,
     the text carries no device pattern, and the refusal is in a GroupNorm call phase?
   - Are the real-test controls real: the real test file through the real plugin?
   - Can an honest kill be refused, or a false one accepted?

   SOUND / UNSOUND.
3. **Runner outcomes.** Does A3 recognise every outcome the installed runner can log? Check against the runner's
   source.
   - Are "not admitted" attempts retryable?
   - Is the timeout cause right for each exit 124 and 137?
   - Is an interrupted attempt evaluated rather than retried?

   SOUND / UNSOUND.
4. **D5 validation.**
   - Are NumPy integer counts accepted and `numpy.bool_` refused?
   - Are wrong parameter shapes refused before any tensor op?
   - Does everything A2 accepted still pass, with bit-identical values?

   SOUND / UNSOUND.
5. **The execution amendments (`a3/AMENDMENTS.md`).**
   - Is deleting the explicit `numpy.bool_` clause really equivalent under the installed NumPy (2.4.2)? Check
     `operator.index` on `numpy.bool_` yourself. Is the negation mutant a valid replacement?
   - Is each replacement witness a fair isolation, not a weakened test?
   - Does the 20 s to 50 s parent bound change touch any acceptance predicate?

   SOUND / UNSOUND.
6. **Mutants and seal.**
   - Is every mutant single-site, with a call-phase trace (M14 by default binding)?
   - Are V4 and the D5 mutants killed by their witnesses on the CPU stand-in?
   - Does the import-traced seal pin every module the sealed suites import?
   - Does the TMPDIR guard resolve paths?

   SOUND / UNSOUND.
7. **Overall:** ACCEPT (no findings) or REJECT (with the findings).

## Method

- CPU only, with `CUDA_VISIBLE_DEVICES` empty. GPU never.
- **A finding** has an ID (`R2-Fnn`), a severity (BLOCKER, MAJOR, MINOR or NIT), the file and line, a reproduction
  or a precise argument, and a suggested fix.
- Report every real defect, NITs included. The house rule is no waivers.
- **Write early:** create `artifacts/gn1/rev3/findings.json` in your first steps and keep it current.
- **The report:** write `artifacts/gn1/rev3/REPORT.md`. If your tool refuses to write a report file, do not work
  around it. Put the full report in your final message instead, and the lead will file it.

## Mechanism & scope

- **Measures:** whether A3 closes every REV2 finding without new defects, so that the GPU run can only report real
  kills and real regressions, and GroupNorm refuses exactly the inputs outside its domain.
- **Mechanism doc:**
  - `artifacts/gn1/rev2/REPORT.md`;
  - `orders/GN1_A3_REV2_FIXES.md`;
  - `artifacts/gn1/a2/REPORT.md`;
  - `artifacts/gn1/a2/FINDINGS_FIXED.md`.
- **Exercises:**
  - A3's class and validation on CPU;
  - A3's parent, child and scorer under REV2's reproductions and your own hostile inputs, with the installed runner
    redirected to scratch;
  - the mutant traces;
  - the arm regeneration.
- **Does not cover:**
  - GPU execution;
  - the oracle and the math REV1 found sound;
  - findings REV2 confirmed fixed, unless A3 changed them;
  - other layers;
  - performance.
- **Required evidence:**
  - `findings.json` and the report;
  - the scripts and outputs of every hostile run and probe;
  - the regeneration comparison.
- **Operand census:**
  - R2-F01 to R2-F11;
  - the two A3 amendments;
  - H1 to H6, plus your new cases;
  - every D5 refusal and acceptance class;
  - every mutant, V1 to V3 included.
- **Baseline isolation:** REV2's receipts and A3's are read in place. Your runs write only under `rev3/` and
  `/tmp/gn1_rev3_*`.
- **Forbidden configuration:**
  - any write outside `rev3/` and `/tmp/gn1_rev3_*`;
  - editing the product, the tests or any receipt;
  - GPU use;
  - git writes;
  - deleting anything.

## Registration checklist (order-lint)

- **L01:** the instruments all exist: REV2's scripts, A3's tooling and receipts, the installed runner, NumPy 2.4.2,
  and pytest 9.0.2.
- **L05:** findings are attributed to the product (`nn.py`), the tests, or the GPU tooling under `artifacts/gn1/a2/`.
- **L07:** n/a. Nothing visual.
- **L09:**
  - JUnit outcomes are compared ID by ID as exact strings;
  - arms are compared by sha256 maps;
  - exits are compared as integers.
- **L16:** the cheapest passing review re-reads A3's control table and agrees. Questions 1, 3, 4 and 6 require your own
  hostile runs, your new cases, and an independent regeneration.
- **L17:** n/a. Nothing visual.
- **L21:** the remote is public. No personal names, no raw quotes, home paths as `$HOME/...`, and no local tool paths.
- **L25:** threads at 1.
- **L26:** the lead touches nothing in this worktree while you run.
- **L34:** a Claude seat (Opus 5.5) reviews Codex's work, which makes the review cross-model.
- **L40:** outputs go under `artifacts/gn1/rev3/` or `/tmp/gn1_rev3_*`.
- **L41:** question 5 checks that every mutant's witness reaches its edit site.
- **L42:** budget 60 minutes. If it runs out, report what was done and what was not.
- **L45:** your hostile runs record each child's exit after it exits.
- **L58:** `CUDA_VISIBLE_DEVICES` is empty for every run.
- **L65:** n/a. You are a Claude Agent-tool seat, and the delete guard covers you. Delete nothing anyway. Scratch goes
  under `/tmp`.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is immutable.
- NO git writes. NO subagents. No edits outside `artifacts/gn1/rev3/`. Never kill a process you did not start.
- Prior Art Directive: name your sources for any method you use.

## Done

Your final message contains:
- the verdict;
- every finding in full;
- what you checked and ran for each of questions 1 to 6;
- the files you wrote.
