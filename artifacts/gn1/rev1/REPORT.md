# GN1-REV1 review report

- **Reviewer:** Claude Opus 5.5 (Agent-tool seat). This is a cross-model review of work by Codex (GPT-6.1 Sol).
- **Order:** `orders/GN1_REV1_REVIEW.md` (1498cf9b).
- **What was reviewed:** round 1 (50b3418f) and A1 (0c6a4244).
- **Timing and mode:** started 2026-11-01 09:53 EST; about 22 minutes; CPU only.

**Note from the lead.** The seat's tool refused to let it write this file at the end. The lead copied the seat's final
message here, unedited apart from this note. `findings.json` is the seat's own file and is the machine-readable record.

**Verdict: REJECT.** 11 findings: 0 BLOCKER, 2 MAJOR, 5 MINOR, 4 NIT.

The GroupNorm math is correct on every configuration the tests cover:
- the float32 path equals round 1's;
- the float64 reference is independent of the engine;
- the tolerances are neither too loose nor too tight.

The defects are in the GPU supervision and in the test coverage and validation:
- a device fault can count as a mutant kill;
- the parent trusts the child's summary without checking it against the line it ran;
- six mutants edit more than one site;
- some decisions are untested;
- the class accepts some inputs it should refuse.

## Findings

**R1-F01, MAJOR.** `artifacts/gn1/a1/run_gpu_line.py:49-52`
- **Defect:** the device-failure filter does not match the engine's own error text. The engine reports kernel faults as
  "CUDA error at <op>: <message>" (`kernels.cu:100-105`), and none of the filter's six words appear in that text. One
  fault fails every later test, so the mutant counts as killed. A1's MUTANTS.md says device failures do not count.
- **Reproduction (hostile case H2):** the real parent (sha256 equal to the seal) ran M07. A stand-in child reported 351
  failures reading "CUDA error at reduce_lastdim: an illegal memory access was encountered". The receipt shows
  passed=true with 351 killers. The control H2c, worded "cudaErrorIllegalAddress", was rejected.
- **Fix:** add "CUDA error at", "cudaMallocAsync failed", "out of memory", "illegal memory access", "unspecified launch
  failure" and "launch timed out". Or invert the rule: a kill needs no device-pattern failure and at least one
  GroupNorm test failing with an AssertionError or a registered refusal.

**R1-F02, MAJOR.** `run_gpu_line.py:22-68, 99-101, 115`
- **Defect:** the parent trusts the child's summary. It never checks the summary's arm, suite, nn.py sha256 or engine
  sha256 against its own command and the seal, and it never reads `junit.xml`. Any failing test counts as a kill,
  including the pure-NumPy `test_reference_*` tests that no mutant can affect.
- **Reproduction:**
  - H3: an M09 line whose child summary says arm=baseline, with reshape failures, passes with 290 killers.
  - H4: a candidate_focus line passes on an all-pass summary while `junit.xml` reports 3 failures.
- **Fix:**
  - require the summary's arm and suite to equal the parent's own arguments;
  - require the nn and engine sha256 to equal the seal and the pin;
  - require the JUnit outcomes to equal the reports;
  - for mutation lines, require a killer outside `test_reference_*`.

**R1-F03, MINOR.** `score_gpu.py:101-110`
- **Defect:** the G5 scorer reads only the parent receipts. It does not check START.json or its seal, the child
  results, or the expected exits in `gpu_lines.json`. It also accepts a run root outside `a1/`, which the parent and
  child both refuse.
- **Reproduction (H1, H1b):** 22 forged parent receipts, with no START.json and no child results, give G5 passed and
  scorer exit 0, inside the mirror and under `/tmp`.
- **Fix:** re-verify START.json, each child result, each exit against the expected one, and the F02 bindings, and
  apply the same run-root guard.

**R1-F04, MINOR.** Mutants M06, M07, M08, M09, M15 and M17
- **Defect:** these six edit more than one site, against round 1's "one edit" rule. For M07, M08 and M09 the
  registered witness takes the float32 fast path and runs only line 544. The second edit, at `nn.py:547` (the
  mixed-precision affine), is never reached.
- **Reproduction:** `shim_trace_results.json` (edited lines [544, 547], executed [544]).
- **Fix:** split each into single-site mutants, each with a witness that reaches it. For example,
  `test_forward_values[float16-r3-n2-g2-default]` reaches line 547. For M06, M15 and M17, either split them or record
  why the multi-line edit is the minimal runnable form.

**R1-F05, MINOR.** `nn.py:542-543`, `nn.py:530`: decisions that no mutant covers and no test pins.
- **Defect:** no test gives weight and bias different dtypes, and no test uses a non-float input. Three one-edit
  variants give the same forward as the candidate on all 320 value tests:
  - **V1** drops the bias dtype check. For float32 x, float32 weight and float16 bias it raises "ew_binary dtype
    mismatch"; the candidate returns the correct float32 output.
  - **V2** drops the weight dtype check, with the mirror failure (float32 x, float16 weight, float32 bias).
  - **V3** upcasts whenever `dtype != "float32"`, which turns an int64 input into a truncated int64 output where the
    candidate refuses it.
- **Reproduction:** `decisions_probe_results.json`.
- **Fix:** add mixed weight/bias dtype tests and an int64-input refusal test, and register V1 to V3 as mutants.

**R1-F06, MINOR.** `nn.py:515`, `535`, `544-547`: accepted inputs that should be refused.
- **Defect:**
  - `GroupNorm(0, 6)` raises ZeroDivisionError, not the class's ValueError.
  - `GroupNorm(-2, 6)` is accepted, because 6 % -2 == 0. Its forward builds the shape `[N, -2, -3*...]`, which
    `NDArray::reshape` accepts: the element count multiplies the signs (`kernels.cu:80-88`, `757-760`).
  - A channel mismatch is refused only by an incidental weight-reshape error.
  - Rank 0 or 1 input raises IndexError.
- **Reproduction:** the `decisions_probe` input-validation block. The engine step is argued from source, because the
  engine cannot allocate without a GPU. PyTorch's behaviour is unverified; lead to check.
- **Fix:** raise ValueError for non-positive or non-integer group and channel counts, and in forward for
  `x.ndim < 2` or `x.shape[1] != C`. Add refusal tests and mutants.

**R1-F07, MINOR.** `STAGED_GPU_COMMANDS.sh:38-39`; `run_gpu_line.py:24-25, 63-67, 102-114`
- **Defect:** the candidate regression line requires every test in the 10 files to pass. That is stricter than the
  order's rule, which counts only tests that pass on baseline and fail on the candidate. A failure already present on
  both arms turns G5 RED with no regression, and no receipt shows these files all pass at `0e1dea3` on this GPU. The
  risk is a false RED, not a false PASS.
- **Fix:** accept child exit 0 or 1, and pass when `regressions.json` is empty and every candidate failure also failed
  on baseline. Or cite an all-pass receipt for `0e1dea3`.

**R1-F08, NIT.** `run_gpu_line.py:77-79, 94-96`
- **Defect:** a lease refusal (exit 75 or 76) uses up the label in its run root. A retry crashes with FileExistsError
  and writes no receipt (H5). Because later lines depend on earlier ones in the same root, all 22 lines must then rerun
  in a new root.
- **Fix:** record a "not admitted" attempt in a numbered subdirectory and allow a retry.

**R1-F09, NIT.** `gn1/TOLERANCES.md:7-8` ("~7x", "~70x gamma_71"); `a1/TOLERANCES.md:13` ("20*u32*.75*317 < 2.84e-4")
- **Defect:** the budgets are stated as unexplained multipliers, not derived. They do hold: the reviewer's worst-case
  first-order bound peaks at 0.907 of the float32 budget (`test_low_variance[backward-var1e-4-half-*]`), and the
  emulated error at 0.014.
- **Fix:** replace the multipliers with a real bound, state the powf accuracy assumption, and record the margin.

**R1-F10, NIT.** `a1/arms/*`
- **Defect:** the 20 arms G5 runs are not in commit `0c6a4244`, so G5 can run only in this worktree. Each arm's engine
  is an absolute symlink into the worktree package, as round 1's arms also were. Integrity holds: the seal pins every
  file's content by hash, and the reviewer's maps match it.
- **Fix:** load the pinned module by explicit path, or copy it while keeping the sha256 check. Commit the arm sources
  or a script that regenerates them.

**R1-F11, NIT.** `run_gpu_line.py:118`
- **Defect:** the receipt's `timeout` flag is also true for forge-gpu-run's window-closed exit, since both return 124
  (forge-gpu-run lines 184-185). Pass or fail is unaffected, because 124 never equals an expected exit, but the
  recorded cause can be wrong.
- **Fix:** read `forge.log` and record forge-gpu-run's own outcome.

## What was checked and run, per question

1. **Correctness:**
   - Read the diff and the engine ops it relies on: the biased `var` (`ops.cpp:915-919`), `cast` and its backward
     (`ops.cpp:1058-1065`), the mixed-dtype refusal (`kernels.cu:897`), and the float32/float16/bfloat16 dispatch.
     The engine has no float64.
   - The AST comparison shows the float32 path matches round 1's nine statements (`float32_identity_recheck.txt`).
   - Every tested dtype and parameter combination computes in float32, casts back once, and returns each gradient in
     its leaf's dtype.
   - Rank 2 and `G = C` are correct: singleton groups return the bias exactly, with dx exactly 0.
   - Float16 input with bfloat16 parameters, and parameters that need no gradient, are correct.
   - Non-float input is refused inside the engine, but nothing pins that (F05).
   - Defects: F05 and F06.
2. **Oracle independence:**
   - The backward re-derived: dx = (s/m)(m·q − Σq − x̂·Σ(q·x̂)) with q = p·w; dw = Σ p·x̂; db = Σ p. It matches the
     test's formula.
   - The reviewer's own loop-based float64 reference agrees with the test's on all 320 value tests to 3.6e-15.
   - Its own central differences agree with its own backward to 9.4e-10.
   - Hand case: x = [1, 3], G = 1, eps = 0, p = [1, 0] gives y = [−1, 1], dx = [0, 0], dw = [−1, 0], db = [1, 0],
     matching exactly.
   - The oracle reads only the represented inputs.
3. **Tolerances:**
   - Each value mutant on its witness exceeds the tolerance by 4,907x to 53,185x (M01-M04, M12, M13).
   - Honest error on those witnesses is 0.001 to 0.005 of the tolerance.
   - M05 is killed by a reshape refusal and M15 by a float16 overflow to infinity. The rest are killed by assertions
     or refusals.
   - Honest float32 error peaks at 0.014 of the budget (engine-graph emulation, worst of three summation orders).
   - The worst-case bound peaks at 0.907, assuming powf is accurate to 4 ulp (unverified; lead to check).
   - The A1 output-rounding formula is correct; its worst emulated ratio is 0.96, from one bfloat16 rounding.
4. **Mutants:**
   - Each arm's real `nn.py` was run on its witness under a NumPy stand-in engine, with line tracing.
   - Every witness reaches an edited line. M14 is reached by call-path argument.
   - All 18 mutants load and change behaviour. Six are multi-site (F04).
   - Two further variants are killed by the tests: centering omitted (304 tests) and bfloat16 not upcast (101 tests).
   - Uncovered decisions are in F05.
5. **Supervision and scoring:**
   - Read all five scripts and the real forge-gpu-run. The 18 labels match their arms, and the nested time bounds
     are consistent.
   - Hostile runs went through the real parent and scorer, copied byte-identically (checked against the seal) into a
     `/tmp` mirror, with a stand-in forge-gpu-run.
   - Defects exposed: H2, H3, H4, H1 and H1b, plus H5 (F08).
   - Correctly rejected: H2c, H3b and H6 (a child that hangs after writing a pass).
   - The parent recorded each child's exit after the child exited.
6. **Arms:**
   - sha256 maps of all 20 arms, with symlinks resolved, show each differs from the candidate only in `nn.py`.
   - The baseline equals `0e1dea3`. The seal matches the disk, and the single engine file matches the pin.
   - One import probe per arm class (baseline, candidate, M01, audit with CUDA hidden), using the child's exact
     `sys.path` recipe, resolved every import to the right arm and the pinned module.
   - There is no `conftest.py`, no pytest config section, and no reload or subprocess in the regression files.
7. **Scope:**
   - Only `nn.py` (all hunks within GroupNorm, lines 510-548) and the new test file changed outside
     `artifacts/gn1`, `orders` and `logs`.
   - 7,595 tracked sha256 values recomputed from disk: 0 mismatches.
   - The 10 regression files equal a grep of every test importing `tensor_cuda.nn`.
   - Everything the orders required is present. Mutant granularity (F04) and the tolerance derivations (F09) are
     weaker than the orders ask.

## Prior art

- **Wu and He (2018):** the forward formula. The backward is the reviewer's own chain-rule derivation.
- **Wilkinson (1963) and Higham (2002, chapters 3-4):** the rounding model, the gamma_n bound and running error
  analysis. Their transcription onto this engine's op graph is the reviewer's own.
- **IEEE 754:** round-to-nearest-even, used for the float16 and bfloat16 emulation.
- **CUDA powf at 4 ulp:** unverified; lead to check.
- **DeMillo, Lipton and Sayward (1978):** mutation adequacy.
- **The reachability-infection-propagation model (Offutt; Voas, 1990s):** unverified; lead to check.
- **Meszaros (2007):** the test-stub pattern.
- **Central differences:** the classical method.
- **SHA-256 (FIPS 180):** content pinning.

## Files

All under `artifacts/gn1/rev1/`:
- `findings.json`;
- `gn_emulate.py`, `run_numerics.py` and `.log`, `numerics_results.json`;
- `shim_trace.py` and `.log`, `shim_trace_results.json`;
- `hostile_scorer.py` and `.log`, `hostile_results.json`;
- `arms_probe.py` and `.log`, `arms_probe_results.json`;
- `decisions_probe.py` and `.log`, `decisions_probe_results.json`;
- `float32_identity_recheck.txt`, `scope_recheck.txt`.

The scratch directories `/tmp/gn1_rev1_scratch/` and `/tmp/gn1_rev1_mirror/` are left in place.
