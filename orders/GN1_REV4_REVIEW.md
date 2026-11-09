# GN1-REV4: cross-model re-review of GN1 A4 (the REV3 fixes) before the GPU run

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_gn1/artifacts/gn1/rev4/`
(branch `gn1`, at the commit carrying this order). Writing there, CPU runs, and scratch under `/tmp/gn1_rev4_*` are
AUTHORIZED. Point `TMPDIR` at the scratch directory.

**Read-only:** everything else in this worktree (the product, the tests, every receipt under `artifacts/gn1/`), every
other worktree, and production. Read-only git is allowed.

**The installed runner:** you may run the installed `forge-gpu-run` only as REV2 did. Redirect its summary, window
and lock to scratch (`rev2/rev2_forge.py`), and give it a CPU child. Never touch a GPU or the real window and lock
files.

**Searching:** never walk `/`, `/mnt` or `/mnt/Shared` recursively, and never enter `/mnt/Shared/ForgeRealm_Link`.
Search inside this worktree, by path.

## What you review

GN1-REV3 (`artifacts/gn1/rev3/REPORT.md`, `findings.json`) rejected A3 with findings R3-F01 to R3-F06, 3 MINOR and
3 NIT. It also noted that the scorer did not bind each line's bounds to `gpu_lines.json`; the lead ruled that this be
fixed (A4's D7). A Codex seat answered REV3 in amendment A4:
- the order: `orders/GN1_A4_REV3_FIXES.md`;
- the commit: `a14c0aad`;
- the receipts: `artifacts/gn1/a4/`, with `REPORT.md` and the execution amendments in `AMENDMENTS.md`.

The trees and large files are on disk but not committed; `a4/LEAD_HELD_BACK.txt` lists them. The product and the
test file are unchanged since `cf922891`.

This review is narrow. It covers:
- the A4 delta (A4's tooling against A3's);
- whether each REV3 finding and D7 is truly closed;
- A4's new execution correction (the runtime-census instrument, in `AMENDMENTS.md`);
- whatever A4 introduced.

Do not re-review what REV1, REV2 and REV3 found sound unless A4 changed it.

## Questions

1. **Each REV3 finding, and D7.** For R3-F01 to R3-F06 and D7: is the fix correct and complete?
   - Re-run REV3's own reproductions against A4's real parent, child and scorer: B6, B7, B7b and B8 through the
     installed runner redirected to scratch; C7b; E5; `seal_runtime_imports.py`; `npbool_deletion_probe.py`.
   - Add at least one new case per MINOR finding.
   - FIXED / NOT FIXED for each.
2. **Admission.** Is an attempt "not admitted" only when it has no start evidence at all? Can a started attempt ever
   be retried to a pass? Can an honest lock-wait refusal be wrongly counted as admitted? SOUND / UNSOUND.
3. **Timeout causes and robustness.** Is the cause right for every exit 124 and 137, including SIGTERM inside the
   runner's kill grace? Does any receipt problem crash a parent or the scorer instead of being recorded? SOUND /
   UNSOUND.
4. **The runtime seal.** Does it cover what a run actually imports? Do tampered sealed modules fail? Are unsealed
   modules listed without failing the line? Was the census instrument's change (setup report instead of call report)
   neutral to the seal's scope? SOUND / UNSOUND.
5. **Equivalences and bound binding.** Are the two `numpy.bool_` deletion records correct, and does their live
   `operator.index` check fail the line when its premise is false? Does the scorer refuse every mismatch with
   `gpu_lines.json` (outer bound, `--max`, `--min`, inner timeout, expected exits)? SOUND / UNSOUND.
6. **Arms and regeneration.** Do the arms regenerate byte-identically, and does an honest run of every label pass the
   scorer? SOUND / UNSOUND.
7. **Overall:** ACCEPT (no findings) or REJECT (with the findings).

## Method

- CPU only, with `CUDA_VISIBLE_DEVICES` empty. GPU never.
- **A finding** has an ID (`R4-Fnn`), a severity (BLOCKER, MAJOR, MINOR or NIT), the file and line, a reproduction
  or a precise argument, and a suggested fix.
- Report every real defect, NITs included. The house rule is no waivers.
- **Write early:** create `artifacts/gn1/rev4/findings.json` in your first steps and keep it current.
- **The report:** write `artifacts/gn1/rev4/REPORT.md`. If your tool refuses to write a report file, do not work
  around it. Put the full report in your final message instead, and the lead will file it.

## Mechanism & scope

- **Measures:** whether A4 closes every REV3 finding without new defects, so that the GPU run can only report real
  kills and real regressions, and GroupNorm refuses exactly the inputs outside its domain.
- **Mechanism doc:**
  - `artifacts/gn1/rev3/REPORT.md`;
  - `orders/GN1_A4_REV3_FIXES.md`;
  - `artifacts/gn1/a2/REPORT.md`;
  - `artifacts/gn1/a2/FINDINGS_FIXED.md`.
- **Exercises:**
  - A4's parent, child and scorer under REV3's reproductions and your own hostile inputs, with the installed runner
    redirected to scratch;
  - the mutant traces;
  - the arm regeneration.
- **Does not cover:**
  - GPU execution;
  - the oracle and the math REV1 found sound;
  - findings REV2 and REV3 confirmed fixed, unless A4 changed them;
  - other layers;
  - performance.
- **Required evidence:**
  - `findings.json` and the report;
  - the scripts and outputs of every hostile run and probe;
  - the regeneration comparison.
- **Operand census:**
  - R3-F01 to R3-F06 and D7;
  - A4's census-instrument correction;
  - H1 to H6, plus your new cases;
  - every D5 refusal and acceptance class;
  - every mutant, V1 to V3 included.
- **Baseline isolation:** REV3's receipts and A4's are read in place. Your runs write only under `rev4/` and
  `/tmp/gn1_rev4_*`.
- **Forbidden configuration:**
  - any write outside `rev4/` and `/tmp/gn1_rev4_*`;
  - editing the product, the tests or any receipt;
  - GPU use;
  - git writes;
  - deleting anything.

## Registration checklist (order-lint)

- **L01:** the instruments all exist: REV3's scripts, A4's tooling and receipts, the installed runner, NumPy 2.4.2,
  and pytest 9.0.2.
- **L05:** findings are attributed to the product (`nn.py`), the tests, or the GPU tooling under `artifacts/gn1/a2/`.
- **L07:** n/a. Nothing visual.
- **L09:**
  - JUnit outcomes are compared ID by ID as exact strings;
  - arms are compared by sha256 maps;
  - exits are compared as integers.
- **L16:** the cheapest passing review re-reads A4's control table and agrees. Questions 1, 3, 4 and 6 require your own
  hostile runs, your new cases, and an independent regeneration.
- **L17:** n/a. Nothing visual.
- **L21:** the remote is public. No personal names, no raw quotes, home paths as `$HOME/...`, and no local tool paths.
- **L25:** threads at 1.
- **L26:** the lead touches nothing in this worktree while you run.
- **L33:** this review runs no GPU command. The installed runner is used only on CPU children, with its window and
  lock redirected to scratch, as REV2 did. A4's staged GPU lines are read, not run.
- **L48:** where you exercise A4's staged lines on CPU, run them exactly as written (same script, same argv), with
  only the runner redirected as above.
- **L34:** a Claude seat (Opus 5.5) reviews Codex's work, which makes the review cross-model.
- **L40:** outputs go under `artifacts/gn1/rev4/` or `/tmp/gn1_rev4_*`.
- **L41:** question 5 checks that every mutant's witness reaches its edit site.
- **L42:** budget 60 minutes. If it runs out, report what was done and what was not.
- **L45:** your hostile runs record each child's exit after it exits.
- **L58:** `CUDA_VISIBLE_DEVICES` is empty for every run.
- **L70:** n/a. No bound in this review separates rounding from a defect between two device computations.
- **L65:** n/a. You are a Claude Agent-tool seat, and the delete guard covers you. Delete nothing anyway. Scratch goes
  under `/tmp`.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is immutable.
- NO git writes. NO subagents. No edits outside `artifacts/gn1/rev4/`. Never kill a process you did not start.
- Prior Art Directive: name your sources for any method you use.

## Done

Your final message contains:
- the verdict;
- every finding in full;
- what you checked and ran for each of questions 1 to 6;
- the files you wrote.
