# ENG1-A1: fix the five ENG1-REV1 findings (tests, census readout and GPU-line tooling; no engine source change)

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng1`
(branch `eng1`, at the commit carrying this order). Edits, CPU runs and mutant builds are AUTHORIZED, limited to:
- `tensor_cuda/tests/test_eng1.py`;
- `artifacts/eng1/tool/`: the runner, the scorer, the staged script, `gpu_lines.json` and the seal, every change
  ledgered;
- new mutant builds under `artifacts/eng1/mutants/M15/` and `M16/`, built from scratch copies as round 1's were;
- new files under `artifacts/eng1/a1/`;
- `logs/`, and scratch under `/tmp/eng1_a1_*`. Point `TMPDIR` at it for every command.

**Read-only:**
- the engine sources (`tensor_cuda/src/`), so the candidate module stays the pinned `3e6d5805…`;
- `tensor_cuda/tests/test_terrain_render.py` and every other test;
- round 1's receipts and the review's receipts (`artifacts/eng1/rev1/`, commit `5031c765`);
- every other worktree, and production.

Read-only git is allowed.

**Searching:** never walk `/`, `/mnt` or `/mnt/Shared` recursively, and never enter `/mnt/Shared/ForgeRealm_Link`.
Search inside this worktree, by path.

## Why

The cross-model review ENG1-REV1 (Codex) rejected round 1 with five findings. It did not fault the six C++ fixes; the
findings concern the tests, the census readout and the GPU-line tooling. The review is in `artifacts/eng1/rev1/`:
`REPORT.md`, `findings.json`, and the probe scripts it cites. Read them first.

## The fixes (binding; the lead chose these)

1. **R1-F01, the E3 census readout.** The census stays measurement only, and no E3 ID becomes a gate.
   - Relabel round 1's n·u tolerances as "accuracy targets (not error bounds)" in the test file and the report.
   - Add an honest-arithmetic reference for each E3 case: a NumPy emulation of the same operation sequence that
     `nn.py` runs, in the same half dtype, rounding to nearest-even after every op. Use float32 where the class
     upcasts, as GroupNorm does.
   - The census row records:
     - the engine's error against the float64 reference;
     - the emulation's error against the float64 reference;
     - the engine's distance from the emulation.
   - Reproduce the review's two cases (RMSNorm float16 near_eps, LayerNorm bfloat16 near_eps) with the emulation, and
     record them.
2. **R1-F02, binding subprocesses to the arm.** Every staged line's child prepends its arm's pinned package parent to
   the `PYTHONPATH` it passes on, so any Python subprocess a test starts imports the same pinned module.
   - **The preflight:** each line first runs a subprocess origin check under the same environment and cwd. It
     records `tensor_cuda.__file__` and the module's sha256, and refuses the line if the pin differs.
   - The review's `artifacts/eng1/rev1/d4_child_probe.py` scenario must now import the arm's pin in the subprocess, on
     both arms, on CPU.
   - **Tests that read candidate source files by path** are listed in their own class, by test ID and the path each
     reads, and are excluded from the baseline/candidate regression comparison. They still run on the candidate.
3. **R1-F03, the bfloat16 final cast.**
   - Add E2 witnesses whose upstream gradients are not powers of two, in float16 and in bfloat16. Include the
     review's witness: x = 0.0002994537353515625, g = 0.30078125, p = 3, in bfloat16.
   - Each new witness's tolerance is derived the same way as round 1's E2 tolerances.
   - Add mutant **M15**: a bfloat16-only early cast. The new bfloat16 witness must kill it.
4. **R1-F04, the OW = 0 refusal.**
   - Add refusal cases with OH > 0 and OW = 0 for all four public callers (im2col, col2im, avg_pool2d, max_pool2d),
     for example input (0,1,6,2), kernel (3,3), stride (1,1), padding (0,0), which gives OH = 4 and OW = 0.
   - Add mutant **M16**: `OW <= 0` weakened to `OW < 0`. The new cases must kill it on CPU.
5. **R1-F05, the reshape caller count.** Correct the count to 37 engine calls on 32 lines, plus 9 NumPy calls. Keep
   an itemized census in `artifacts/eng1/a1/CALLER_CENSUS.md`, which may reuse the review's `CALLER_CENSUS.md` with
   credit, and correct the report's statement.

**After the fixes:**
- re-stage the GPU lines, adding M15 and M16;
- seal the staging again as a new seal file; the old seal stays;
- run one line in audit mode on CPU with the stand-in;
- rerun the CPU subset on the candidate, which must all pass;
- run the new mutants' CPU kills where they are CPU-reachable (M16 is; M15 needs the GPU).

## Gates

- **G1, scope.** `git diff 5031c765 -- . ':!artifacts/eng1' ':!orders' ':!logs'` touches only `test_eng1.py`.
  `tensor_cuda/src/` is byte-identical, and the candidate module's sha256 is still `3e6d5805…`.
- **G2, CPU.**
  - The new IDs collect.
  - The CPU subset passes on the candidate, new F04 cases included.
  - M16 is killed by them.
  - The emulation reproduces the review's two numbers within stated tolerance.
- **G3, isolation.**
  - The subprocess origin check passes on both arms, and each records its own pin.
  - The review's probe scenario imports the arm's pin, on both arms.
  - The excluded source-reading tests are listed.
- **G4, staging.** The new staged script and seal exist, the audit-mode line passes, and every line names its
  expected exit.

**Registered acceptance:** ACCEPT when G1–G4 pass. The lead then sends this amendment for a narrow cross-model review
of the five fixes only, before G5. The 11-03 rule applies: NITs are fixed once and closed.

## Mechanism & scope

- **Measures:**
  - whether the E3 census compares the engine against honest half arithmetic;
  - whether the GPU lines bind every Python subprocess to the arm's pinned module;
  - whether the new witnesses and mutants pin the bfloat16 final cast and the OW = 0 refusal;
  - whether the caller count is correct.
- **Mechanism doc:**
  - `orders/ENG1_ENGINE_DEFECTS.md`;
  - `artifacts/eng1/REPORT.md`;
  - `artifacts/eng1/rev1/REPORT.md`;
  - `artifacts/eng1/rev1/findings.json`.
- **Exercises:** the test file on CPU, the NumPy emulation, the staged tooling in audit mode, the review's probe
  scenario, and the M15 and M16 builds.
- **Does not cover:**
  - any engine source change;
  - GPU execution (G5, the lead's);
  - changing a norm class;
  - promotion.
- **Operand census:**
  - E3: 40 cases, each with an emulation row;
  - E2: the new non-power-of-two witnesses in float16 and bfloat16;
  - E5: 4 callers × the new OH > 0, OW = 0 case;
  - 2 new mutants;
  - 22 + 2 staged lines.
- **Required evidence:**
  - the diff;
  - the census emulation receipts;
  - the origin-check receipts for both arms;
  - the excluded-test list;
  - the M15 and M16 build and kill receipts;
  - the new staged script and seal;
  - the audit line;
  - `artifacts/eng1/a1/REPORT.md` and `LEDGER.md`.
- **Baseline isolation:** both arms keep their pinned modules, and only the test file and the tooling change.
- **Forbidden configuration:**
  - any engine source edit;
  - editing another test;
  - a tolerance chosen after seeing GPU results;
  - GPU use or a CUDA context;
  - git writes;
  - deleting anything.

## Registration checklist (order-lint)

- **L01:** the instruments are the worktree at the dispatch commit, round 1's CMake build setup with CUDA 12.6 (for
  mutant builds only), NumPy and pytest. All exist (round 1 used them).
- **L05:** the owners are the test file (F01, F03, F04), the tooling (F02) and the report (F05).
- **L07:** n/a. Nothing visual.
- **L09:** test IDs are exact strings, module pins are sha256 strings, and the census errors are numbers recorded
  with the rounding used.
- **L16:** the cheapest pass relabels the tolerances and stops there. Fix 1's emulation rows, fix 2's origin check,
  and the M15 and M16 kills exclude it.
- **L21:** the remote is public. No personal names, no raw quotes, home paths as `$HOME/...`.
- **L25:** threads at 1, except the `-j2` mutant builds.
- **L26:** the lead touches nothing in this worktree while the seat runs.
- **L34:** a Codex seat (gpt-6.1-sol, xhigh), on CPU. The narrow re-review goes to a Claude seat (cross-model).
- **L40:** outputs go under `artifacts/eng1/a1/`, the two new mutant directories, the tooling directory (ledgered),
  or `/tmp/eng1_a1_*`.
- **L41:** M15 and M16 are each shown executing in the test that kills them.
- **L42:** budget 3 hours. Two mutant builds at `-j2` take about 10 minutes each on this loaded machine. If time runs
  out, report what was done.
- **L45:** each line's outcome counts only from the parent's record of the child's exit, written after the child
  exits.
- **L46:** before the first write, record a sha256 map of `tensor_cuda/` without `build-*/`. Show it unchanged except
  `test_eng1.py`.
- **L47:** the review's numbers and paths are the expected values: `numerics.py`, `coverage_gaps.py`,
  `d4_child_probe.py` and `census_scope.py` under `artifacts/eng1/rev1/` (commit `5031c765`).
- **L55:** the module pins and test IDs are exact.
- **L58:** `CUDA_VISIBLE_DEVICES` is empty for every run.
- **L65:** the delete rule is no deletes. Anything that must go stays where it is and is listed in the report, and
  scratch goes under `/tmp`.
- **L87:** n/a. No seeded sampling.
- **L92:** n/a for kinks. The E2 witnesses are at x > 0, where `x^p` is smooth.
- **L93:** n/a.
- **L80:** G1 diffs against `5031c765`, the review receipts commit. The dispatch commit adds only this order, which
  `':!orders'` excludes. Name the dispatch commit's hash in the report.
- **L81:** the expected behaviours come from the review's own probes, with paths above. Re-derive each one from the
  code before relying on it:
  - the M15 witness from the bfloat16 rounding arithmetic;
  - the OH = 4, OW = 0 geometry from `make_dims`'s floor formula;
  - the subprocess import failure by rerunning the probe.
- **L82:** the write list is the files that the five fixes must touch: `test_eng1.py` for F01, F03 and F04; the
  tooling under `artifacts/eng1/tool/` for F02; `a1/` for F05 and the receipts; plus the two new mutant directories.
  `grep -rn "PYTHONPATH\|sys.path\|preload" artifacts/eng1/tool/*.py` lists the tooling sites F02 touches. List each
  hit with its disposition.
- **L90:** each operand set gets one treatment:
  - all 40 E3 cases get an emulation row;
  - all four `make_dims` callers get the OW = 0 case;
  - every staged line gets the origin check.

  No set is sampled.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is immutable.
- NO git writes. NO subagents. GPU never. Never kill a process you did not start.
- **CPU courtesy:** nice 19, one heavy process at a time, every command under 20 minutes.
- **Prior Art Directive:** round 1's annotations stand. For the emulation, name the IEEE 754 round-to-nearest-even
  emulation, and NumPy's float16 and an explicit bfloat16 rounding.
- **Write early:** create `artifacts/eng1/a1/REPORT.md` and `LEDGER.md` in your first steps.

## Done

Your final message contains:
- G1–G4 with receipts;
- the test-file diff;
- the census emulation table for the review's two cases;
- the origin-check results for both arms;
- the excluded-test list;
- M15 and M16 with their kill receipts;
- the new staged lines and the new seal's sha256;
- every file written, and anything left to bin.
