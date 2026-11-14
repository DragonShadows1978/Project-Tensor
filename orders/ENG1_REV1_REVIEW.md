# ENG1-REV1: cross-model review of the six tensor_cuda fixes, before their GPU run

YOUR WRITABLE TARGET is
`/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng1/artifacts/eng1/rev1/` (branch `eng1`, at
the commit carrying this order). Writing there, CPU runs, and scratch under `/tmp/eng1_rev1_*` are AUTHORIZED. Point
`TMPDIR` at the scratch directory for every command.

**Read-only:**
- everything else in this worktree: the product, the tests, the round's receipts under `artifacts/eng1/` and the
  builds;
- every other worktree, and production.

Read-only git is allowed.

**Searching:** never walk `/`, `/mnt` or `/mnt/Shared` recursively, and never enter `/mnt/Shared/ForgeRealm_Link`.
Search inside this worktree, by path.

## What you review

A Claude seat (Opus 5.5) fixed six engine defects in one round: order `orders/ENG1_ENGINE_DEFECTS.md`, receipts
commit `ec96f49f`, report `artifacts/eng1/REPORT.md`, ledger `artifacts/eng1/LEDGER.md`. The diff against `1680764d`
is `artifacts/eng1/source.diff`.
- **E1:** `NDArray::reshape` now infers a single `-1`.
- **E2:** `pow_scalar`'s backward computes in float32 for float16 and bfloat16 inputs.
- **E4:** `rms_norm`, `causal_softmax`, `apa_blend_softmax` and `topk_nd` handle empty extents, and `topk_nd`
  refuses k outside [0, S].
- **E5:** `make_dims` uses floor division and refuses non-positive output extents.
- **E6:** the terrain tests write their `.ppm` files under `tmp_path`.
- **E3:** a half-precision census of the norm layers, as measurement only.

The lead's GPU run (G5) follows your review. It executes the 22 staged lines in
`artifacts/eng1/tool/STAGED_GPU_COMMANDS.sh`.

## Questions (answer each, with receipts)

1. **Correctness of each fix.** Read `source.diff`.
   - Does E1 match PyTorch's `reshape`/`view` rules for every shape it accepts or refuses? Is any valid shape now
     refused, or any invalid one accepted? Do the 30 engine-side reshape callers the seat listed really keep their
     results?
   - Is E2's float32 path numerically right, and is the float32-input path bit-identical to before?
   - Are E4's empty results the right shape and dtype? Is the topk refusal placed before any allocation or launch?
     Does any other caller of these four functions rely on the old behaviour?
   - Does E5's floor division change any accepted geometry? Check every caller of `make_dims`.
   - Does E6 change only where files are written, and nothing that the terrain tests assert?
2. **The tests' independence.** Are the float64 NumPy references in `tensor_cuda/tests/test_eng1.py` independent of
   the engine's outputs? Re-derive E2's gradient and one E3 backward yourself, and check one case of each with your
   own NumPy.
3. **Tolerances.** Re-check the derivations in the report.
   - Is any tolerance loose enough that a wrong implementation would pass? For each mutant M01–M14, bound the error
     it causes on its registered witness, and compare that with the witness's tolerance.
   - Is any tolerance too tight for honest float16, bfloat16 or float32 arithmetic on these operands? Bound the
     honest error on the worst-conditioned case.
4. **Mutants.** Is each mutant one edit that changes behaviour on a registered input, and does its registered
   witness execute the edited line? Name any decision in the fixes that no mutant covers and no test pins.
5. **The staged lines.** Read `STAGED_GPU_COMMANDS.sh`, the adapted runner and the scorer under
   `artifacts/eng1/tool/`, and `TOOLING_CHANGES.diff` against GN1's accepted A8 originals in
   `tool/_gn1_a8_originals/`.
   - Does a line count as passing only from the child's own exit code and its JUnit file, recorded after the child
     exits?
   - Does each line's expected outcome match what the tests should do on that arm?
   - Do the 27 hunks keep that property?

   Read the code paths and run the scorer on the seat's audit output (`tool/audit_final/`), as it stands, on CPU.
6. **Arms.** Do the baseline and candidate arms each load their own pinned module and nothing else? Check the
   preload method the seat describes (decision D4) with one import probe per arm on CPU, with CUDA hidden.
7. **Scope.** Is anything outside the named functions changed? Is anything the order required missing? The seat
   corrected several premises (F1–F6, D1–D4 in the report). Are those corrections right?
8. **The new finding F2:** the engine has no float64, and `tc.tensor(dtype="float64")` silently builds float32. Find
   where the silent conversion happens, and say what the smallest honest fix would be: a refusal or a warning. This
   is advice only; the fix belongs to a later round.

## Method

- **CPU only,** with `CUDA_VISIBLE_DEVICES` empty, `nice -n 19` and one thread. The machine is heavily loaded. Keep
  every command under 10 minutes.
- **A finding** has an ID (`R1-Fnn`), a severity (BLOCKER, MAJOR, MINOR or NIT), the file and line, a reproduction or
  a precise argument, and the fix you suggest. A question you checked and found sound is reported as checked, with
  what you ran.
- **Write early.** Create `artifacts/eng1/rev1/REPORT.md` and `artifacts/eng1/rev1/findings.json` in your first few
  steps, and keep them up to date as you go.

## Mechanism & scope

- **Measures:**
  - whether the six fixes are correct;
  - whether their tests, tolerances and mutants can detect a wrong implementation;
  - whether the staged GPU lines report only outcomes the child processes produced.
- **Mechanism doc:**
  - `orders/ENG1_ENGINE_DEFECTS.md`;
  - `artifacts/eng1/REPORT.md`;
  - `artifacts/eng1/LEDGER.md`.
- **Exercises:** the diff, the tests' references on CPU, the tolerance derivations, the mutants, the staged tooling
  read and run on the existing audit output, and one import probe per arm.
- **Does not cover:**
  - GPU execution (G5);
  - the engine's other operations;
  - performance;
  - promotion.
- **Required evidence:**
  - `findings.json` and `REPORT.md`;
  - the scripts behind every computation, with their outputs;
  - each mutant-against-tolerance bound;
  - the honest-error bound for the worst-conditioned case.
- **Operand census:**
  - the diff;
  - every test ID class (E1, E2, E3, E4, E5, E6);
  - mutants M01–M14 and the C_COPY and C_UBSAN controls;
  - the staged tooling;
  - both arm classes.
- **Baseline isolation:** the review reads frozen receipts. Its own runs write only under `rev1/` and
  `/tmp/eng1_rev1_*`.
- **Forbidden configuration:**
  - any write outside `rev1/` and scratch;
  - editing the product, the tests or any receipt;
  - GPU use;
  - git writes;
  - deleting anything.

## Registration checklist (order-lint)

- **L01:** the instruments are the receipts named above, NumPy, pytest and the pinned modules. All exist (commit
  `ec96f49f`).
- **L05:** findings are attributed to the engine source, the test file, or the staged tooling.
- **L07:** n/a. Nothing visual. The terrain images' content is not under review, only where they are written.
- **L09:** identity claims are checked by their own rules: the float32 path as uint32 views, the diff scope by sha256
  maps (`tracked_before.json`, `tracked_after.json`), and test IDs as exact strings.
- **L16:** the cheapest passing review reads the report and agrees. Questions 2, 3, 4 and 6 require your own
  computation, with receipts.
- **L21:** the remote is public. No personal names, no raw quotes, home paths as `$HOME/...`.
- **L25:** threads at 1.
- **L26:** the lead touches nothing in this worktree while you run.
- **L34:** a Codex seat reviews a Claude seat's work, which makes the review cross-model.
- **L40:** outputs go under `artifacts/eng1/rev1/` or `/tmp/eng1_rev1_*`.
- **L41:** question 4 asks for each mutant's witness to execute its edited line, shown by a trace or by an argument
  from the call path.
- **L42:** budget 3 hours. If it runs out, report what was done and what was not.
- **L45:** your runs record each child's exit after it exits.
- **L58:** `CUDA_VISIBLE_DEVICES` is empty for every run.
- **L65:** the delete rule is no deletes. Anything that must go stays where it is and is listed in your report, and
  scratch goes under `/tmp`.
- **L92:** E2's check against central differences is on `x^p` for x > 0, which is smooth. State whether any checked
  path crosses a kink.
- **L93:** n/a. No unit sample is drawn by graph position.
- **L46:** the review's own runs are confined to `rev1/` and scratch. Before the first run, record a sha256 map of the
  worktree without `artifacts/eng1/rev1/`. Show it unchanged at the end.
- **L81:** expected behaviours under review come from PyTorch's documented semantics and the order's design, not
  from a probe. Where the seat took one from its own probe (F1, the topk garbage; F3, truncation), re-derive it from
  the source at `1680764d`.
- **L90:** question 1 names the treatment for each fixed function. The census of callers (30 reshape callers, the
  `make_dims` callers, the callers of the four E4 functions) is checked item by item, not sampled.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is immutable.
- NO git writes. NO subagents. No edits outside `artifacts/eng1/rev1/`. Never kill a process you did not start.
  GPU never.
- **Prior Art Directive:** name your sources for any method you use (error bounds, central differences, mutation
  testing).

## Done

Your final message contains:
- the verdict: ACCEPT (no findings) or REJECT (with the findings);
- every finding, with its ID, severity, file:line, reproduction and suggested fix;
- for each of questions 1 to 8, what you checked and ran;
- the files you wrote.
