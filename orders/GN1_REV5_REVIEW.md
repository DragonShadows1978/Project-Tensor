# GN1-REV5: cross-model re-review of GN1 A5 (the REV4 fixes) before the GPU run

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_gn1/artifacts/gn1/rev5/`
(branch `gn1`, at the commit carrying this order). Writing there, CPU runs, and scratch under `/tmp/gn1_rev5_*` are
AUTHORIZED. Point `TMPDIR` at the scratch directory.

**Read-only:** everything else in this worktree (the product, the tests, every receipt under `artifacts/gn1/`), every
other worktree, and production. Read-only git is allowed.

**The installed runner:** you may run the installed `forge-gpu-run` only as REV2 did. Redirect its summary, window
and lock to scratch (`rev2/rev2_forge.py`), and give it a CPU child. Never touch a GPU or the real window and lock
files.

**Searching:** never walk `/`, `/mnt` or `/mnt/Shared` recursively, and never enter `/mnt/Shared/ForgeRealm_Link`.
Search inside this worktree, by path.

## What you review

GN1-REV4 (`artifacts/gn1/rev4/REPORT.md`, `findings.json`) rejected A4 with 2 MINOR and 2 NIT findings, R4-F01 to
R4-F04:
- D7's bound check was a token search;
- the staged default kill grace equalled the outer kill delay;
- `timeout_cause` could raise;
- the parent receipt was not atomic.

A Codex seat answered REV4 in amendment A5:
- the order: `orders/GN1_A5_REV4_FIXES.md`;
- the commit: `24ddbff3`;
- the receipts: `artifacts/gn1/a5/`, with `REPORT.md` and `AMENDMENTS.md`.

The trees are on disk but not committed; `a5/LEAD_HELD_BACK.txt` lists them. The product and the test file are
unchanged since `cf922891`.

This review is narrow. It covers:
- the A5 delta (A5's tooling against A4's);
- whether each REV4 finding is truly closed;
- whatever A5 introduced.

Do not re-review what earlier reviews found sound unless A5 changed it.

## Questions

1. **Each REV4 finding.** For R4-F01 to R4-F04: is the fix correct and complete?
   - Re-run REV4's own reproductions against A5's real parent, child and scorer: the five D7 spellings, N2b and N2c
     with the grace unset and under A5's staging, N4 and N5.
   - Add at least one new case per MINOR finding.
   - FIXED / NOT FIXED for each.
2. **Exact-form binding.** Is every accepted line's command exactly the registered argv? Can any spelling, wrapper,
   environment variable (for example `FORGE_GPU_KILL_GRACE` changed outside the staged script) or argument order make
   the runner or `timeout` apply a different bound while the scorer accepts it? SOUND / UNSOUND.
3. **Kill-grace ordering.** Check A5's arithmetic against the installed runner's source. Under the staged values, does
   every interruption path end with the runner's own row written and no surviving child? SOUND / UNSOUND.
4. **Robustness.** Can any malformed or truncated receipt, or any runner row, still crash a parent or the scorer
   instead of becoming a recorded reason? SOUND / UNSOUND.
5. **Regression.** Do the earlier sound areas still hold under A5's tooling: admission, the runtime seal, the
   equivalences, regeneration, and an honest run of every label? SOUND / UNSOUND.
6. **Overall:** ACCEPT (no findings) or REJECT (with the findings).

## Method

- CPU only, with `CUDA_VISIBLE_DEVICES` empty. GPU never.
- **A finding** has an ID (`R5-Fnn`), a severity (BLOCKER, MAJOR, MINOR or NIT), the file and line, a reproduction
  or a precise argument, and a suggested fix.
- Report every real defect, NITs included. The house rule is no waivers.
- **Write early:** create `artifacts/gn1/rev5/findings.json` in your first steps and keep it current.
- **The report:** write `artifacts/gn1/rev5/REPORT.md`. If your tool refuses to write a report file, do not work
  around it. Put the full report in your final message instead, and the lead will file it.

## Mechanism & scope

- **Measures:** whether A5 closes every REV4 finding without new defects, so that the GPU run can only report real
  kills and real regressions, and GroupNorm refuses exactly the inputs outside its domain.
- **Mechanism doc:**
  - `artifacts/gn1/rev4/REPORT.md`;
  - `orders/GN1_A5_REV4_FIXES.md`;
  - `artifacts/gn1/a2/REPORT.md`;
  - `artifacts/gn1/a2/FINDINGS_FIXED.md`.
- **Exercises:**
  - A5's parent, child and scorer under REV4's reproductions and your own hostile inputs, with the installed runner
    redirected to scratch;
  - the mutant traces;
  - the arm regeneration.
- **Does not cover:**
  - GPU execution;
  - the oracle and the math REV1 found sound;
  - findings earlier reviews confirmed fixed, unless A5 changed them;
  - other layers;
  - performance.
- **Required evidence:**
  - `findings.json` and the report;
  - the scripts and outputs of every hostile run and probe;
  - the regeneration comparison.
- **Operand census:**
  - R4-F01 to R4-F04;
  - H1 to H6, plus your new cases;
  - every D5 refusal and acceptance class;
  - every mutant, V1 to V3 included.
- **Baseline isolation:** REV4's receipts and A5's are read in place. Your runs write only under `rev5/` and
  `/tmp/gn1_rev5_*`.
- **Forbidden configuration:**
  - any write outside `rev5/` and `/tmp/gn1_rev5_*`;
  - editing the product, the tests or any receipt;
  - GPU use;
  - git writes;
  - deleting anything.

## Registration checklist (order-lint)

- **L01:** the instruments all exist: REV4's scripts, A5's tooling and receipts, the installed runner, NumPy 2.4.2,
  and pytest 9.0.2.
- **L05:** findings are attributed to the product (`nn.py`), the tests, or the GPU tooling under `artifacts/gn1/a2/`.
- **L07:** n/a. Nothing visual.
- **L09:**
  - JUnit outcomes are compared ID by ID as exact strings;
  - arms are compared by sha256 maps;
  - exits are compared as integers.
- **L16:** the cheapest passing review re-reads A5's control table and agrees. Questions 1, 3, 4 and 6 require your own
  hostile runs, your new cases, and an independent regeneration.
- **L17:** n/a. Nothing visual.
- **L21:** the remote is public. No personal names, no raw quotes, home paths as `$HOME/...`, and no local tool paths.
- **L25:** threads at 1.
- **L26:** the lead touches nothing in this worktree while you run.
- **L33:** this review runs no GPU command. The installed runner is used only on CPU children, with its window and
  lock redirected to scratch, as REV2 did. A5's staged GPU lines are read, not run.
- **L48:** where you exercise A5's staged lines on CPU, run them exactly as written (same script, same argv), with
  only the runner redirected as above.
- **L34:** a Claude seat (Opus 5.5) reviews Codex's work, which makes the review cross-model.
- **L40:** outputs go under `artifacts/gn1/rev5/` or `/tmp/gn1_rev5_*`.
- **L41:** question 5 checks that every mutant's witness reaches its edit site.
- **L42:** budget 60 minutes. If it runs out, report what was done and what was not.
- **L45:** your hostile runs record each child's exit after it exits.
- **L58:** `CUDA_VISIBLE_DEVICES` is empty for every run.
- **L70:** n/a. No bound in this review separates rounding from a defect between two device computations.
- **L65:** n/a. You are a Claude Agent-tool seat, and the delete guard covers you. Delete nothing anyway. Scratch goes
  under `/tmp`.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is immutable.
- NO git writes. NO subagents. No edits outside `artifacts/gn1/rev5/`. Never kill a process you did not start.
- Prior Art Directive: name your sources for any method you use.

## Done

Your final message contains:
- the verdict;
- every finding in full;
- what you checked and ran for each of questions 1 to 6;
- the files you wrote.
