# ENG2: refuse unsupported dtype strings, and compute four half-precision norms in float32 (CPU round; GPU lines staged)

**Dispatched 2026-11-14** on branch `eng2`, from `eng1` at `bc02d3a1`: production main plus GN1, plus ENG1 (whose G5
passed 24/24 on 11-14, with promotion pending the owner's approval). ENG2 promotes after ENG1.

## Why

Two open defects in Tensor's register, both found by ENG1:
- **F1, FLOAT64-SILENT.** `tc.tensor(..., dtype=s)` and every `_factory` maker look up `_NP_DTYPE.get(dtype,
  np.float32)` (`tensor_cuda/tensor_cuda/__init__.py:48` and `:60`).
  - Any dtype string the engine does not support (`"float64"`, `"double"`, `"int32"`, a typo) silently builds a
    float32 tensor.
  - The engine has no float64 dtype (`tensor_cuda/include/tc/core.h:30`).
  - ENG1-REV1 asked for the smallest honest fix: a refusal.
- **N1, NORM-HALF.** ENG1's E3 census (G5, 11-14, `artifacts/eng1/tool/gpu_runs/g5_1114/candidate_eng1/`) measured, in
  float16 forward on the `near_eps` case:

  | Class | Error | Tolerance | Error / tolerance |
  |---|---:|---:|---:|
  | LayerNorm | 0.01773 | 0.01429 | 1.24× |
  | RMSNorm | 0.01566 | 0.00843 | 1.86× |
  | BatchNorm1D | 0.01319 | 0.00811 | 1.63× |
  | InstanceNorm2D | 0.00768 | 0.00669 | 1.15× |

  - GroupNorm stays within tolerance on both cases. GN1 made it compute in float32 for half inputs
    (`tensor_cuda/tensor_cuda/nn.py:569-591`).
  - The engine error equals ENG1's CPU emulation of the same half arithmetic, so the error is the half-precision
    statistics themselves, not a kernel fault.
  - LayerNorm, for example, computes mean, variance and `pow(-0.5)` in the input dtype (`nn.py:132-137`).

## The changes

1. **F1.** In `tc.tensor` and `_factory` (`__init__.py:42-62`), a dtype string that is neither in `_NP_DTYPE` nor in
   `_DEVICE_CAST_DTYPE` raises `TypeError` naming the supported dtypes, before any allocation.
   - Every supported dtype, and the default `dtype="float32"` path with float64 input data, behave exactly as today.
   - Also check the C++ `dtype_from_string` and every other Python entry that maps a dtype string (`astype`, `to`,
     and any others your census finds). Report each one: does it already refuse, or does it fall back silently? Fix
     the Python fallbacks the same way. List any C++ fallback for the lead; do not change C++ in this round.
2. **N1.** In `LayerNorm`, `RMSNorm`, `BatchNorm1D` and `InstanceNorm2D` (`nn.py:124`, `:280`, `:485`, `:592`), follow
   GroupNorm's pattern:
   - for float16 and bfloat16 inputs, compute in float32, then cast the output back to the input dtype;
   - convert half weights and biases to float32 for the affine step.
   - **The float32 path must stay unchanged in op sequence**, so it stays bit-identical to the baseline.
   - Running statistics (BatchNorm1D) keep their existing dtype and update rule. Report what they are.
   - No other class or op changes.
3. **Tests:** a new `tensor_cuda/tests/test_eng2.py`.
   - **F1:**
     - the refusal for `"float64"`, `"double"`, `"int32"` and an unknown string, in `tc.tensor` and in every
       factory;
     - each supported dtype still builds its own dtype, with the same bytes as the baseline arm;
     - the default path still converts float64 data to float32.
   - **N1:**
     - forward and backward, in float16 and bfloat16, on E3's `regular` and `near_eps` cases;
     - for the four classes, plus GroupNorm as a control;
     - against E3's float64 NumPy reference, at E3's derived tolerances. Reuse `e3_inputs`, `e3_reference`,
       `e3_tolerance` and `e3_distance` from `test_eng1.py` by import or verbatim copy, and ledger which;
     - float32 forward and backward bit-identical to the baseline arm, on the same inputs.
   - Tolerances come from E3's derivation, unchanged. Never tune them to results.
4. **Mutants,** listed before they run, one edit each:
   - **N1:**
     - (MN1) the cast back omitted, so the output is float32;
     - (MN2) the upcast only for float16, not bfloat16;
     - (MN3) the statistics still in the input dtype, for one class at a time (four mutants);
     - (MN4) the float32 path given an extra cast.
   - **F1:**
     - (MF1) the silent fallback restored at `:48`;
     - (MF2) the silent fallback restored at `:60`.

   The lead kills each on the GPU. A survivor counts only with an equivalence proof.
5. **GPU lines (the lead runs them).** Copy ENG1's accepted tool (`artifacts/eng1/tool/`, as A3 left it) into
   `artifacts/eng2/tool/`, and ledger every change.
   - **Device-mode routing:** the staged command's device-mode path must verify the seal of the files on disk (the
     ENG1-A3 lesson). Show it with a dry run (G4).
   - **The lines:**
     - (a) the baseline arm on `test_eng2.py`, where the N1 and F1 tests fail as predicted, per ID;
     - (b) the candidate, where everything passes;
     - (c) the candidate against the baseline on `test_eng1.py`, `test_gn1*.py`, and every existing test file that
       builds one of the five norm classes or calls `tc.tensor`/a factory with a non-default dtype. No test that
       passes on the baseline fails on the candidate;
     - (d) one line per mutant.
   - Name each line's expected exit. The parent records the child's exit after the child exits.

## Gates

- **G1, scope.** Against `bc02d3a1`, only the lines named in changes 1-2 and the new test file change.
  - **The seat's form (no git):** whole-file sha256 maps of the worktree, taken before any edit and after. Every
    other tracked file must be byte-identical.
  - **The lead's form:** `git diff bc02d3a1 -- . ':!artifacts/eng2' ':!orders' ':!logs'`.
- **G2, CPU checks.**
  - The references match hand-computed values on constructed inputs.
  - F1's refusals run on CPU: the refusal comes before any device call. Show it with `CUDA_VISIBLE_DEVICES` empty.
  - The test files import and collect, and list their IDs.
  - Both arms import with `CUDA_VISIBLE_DEVICES` empty.
- **G3, build.** Both arms build (exit 0). Module sha256s are recorded. The engine's C++ is unchanged, so the
  candidate's `.so` sha256 equals the baseline's, or the difference is explained.
- **G4, staging.**
  - Every line has its expected exit and bounds.
  - A device-mode dry run (`FORGE_GPU_DRY_RUN=1`) of every root line (one with no dependency on an earlier line)
    verifies the seal and every arm, and reaches the governor (exit 78).
  - Every dependent line verifies the seal and every arm, then refuses at its dependency check. Its runner argv,
    given to the governor in dry-run mode, exits 78.
  - The audit route passes on CPU.
- **G5 (the lead's GPU run, after this round):** (a) fails as predicted; (b) passes; (c) shows no regression;
  (d) kills every mutant.

**Registered acceptance:** ACCEPT when G1 to G4 pass. G5 is the lead's, after a cross-model review. NIT findings in that
review are fixed once and closed, with no further review round (owner, 11-03).

## Lead predictions (registered here, before any data)

- **P301:** with float32 statistics, all four classes fall within E3's tolerance on `near_eps`, in float16 and
  bfloat16, forward and backward.
- **P302:** float32 forward and backward are bit-identical to the baseline, for all five classes.
- **P303:** no existing caller in this worktree, Project-Voice's `engine/` or GRAPA's `src/` passes an unsupported
  dtype string to `tc.tensor` or a factory. The lead's grep found none.
- **P304:** at least one C++ dtype-string path also falls back silently, or else `dtype_from_string` refuses. Either
  is recorded.

## Mechanism & scope

- **Measures:** whether unsupported dtype strings are refused rather than silently downcast, and whether the four
  norm classes meet E3's float16/bfloat16 tolerance when computed in float32, with float32 behaviour unchanged.
- **Mechanism doc:**
  - `orders/ENG1_ENGINE_DEFECTS.md` (E3);
  - `artifacts/eng1/a3/LEAD_RULING_A3.md`;
  - `tensor_cuda/tensor_cuda/nn.py` (GroupNorm, `:510-591`).
- **Exercises:**
  - `tc.tensor` and the factories;
  - the five norm classes, forward and backward, in float32, float16 and bfloat16;
  - ENG1's staged-line tooling.
- **Does not cover:**
  - adding a float64 dtype to the engine;
  - C++ dtype parsing changes;
  - other layers or ops;
  - performance;
  - promotion;
  - TX3's candidate changes.
- **Operand census:**
  - the two Python fallback sites and any others found;
  - the four classes plus GroupNorm;
  - the E3 cases `regular` and `near_eps`;
  - the dtypes float32, float16 and bfloat16.
- **Baseline isolation:** the baseline arm is `bc02d3a1` built unchanged, beside the candidate, as ENG1 did.
- **Required evidence:**
  - the diff and sha256 maps;
  - the reference checks;
  - the build receipts;
  - the mutant list;
  - the staged script with expected exits;
  - the dry-run and audit receipts;
  - the census of dtype-string entries.
- **Forbidden configuration:**
  - GPU execution;
  - changing tolerances after seeing results;
  - any change outside the writable list;
  - git writes;
  - deleting anything yourself.

## Where to work

**Seat:** Claude Opus 5.5, high effort (Agent tool).

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng2`
(branch `eng2`, at the commit carrying this order). Edits, CPU runs and engine builds are AUTHORIZED, limited to:
- `tensor_cuda/tensor_cuda/__init__.py`: the dtype-string fallbacks (`:48`, `:60`), plus any other Python
  dtype-string fallback your census finds in `tensor_cuda/tensor_cuda/` (ledger each one);
- `tensor_cuda/tensor_cuda/nn.py`: the `forward` methods of `LayerNorm`, `RMSNorm`, `BatchNorm1D` and
  `InstanceNorm2D`;
- the new `tensor_cuda/tests/test_eng2.py`;
- everything under `artifacts/eng2/`;
- the build directory `tensor_cuda/build-eng2/` and the built module it installs into this worktree's package;
- `logs/eng2_*`;
- scratch under `/tmp/eng2_*`. Point `TMPDIR` at it for every command.

**Read-only:**
- every other file here, including `artifacts/eng1/` (copy its tool, never edit it);
- every other worktree, the production repository, Project-Voice and GRAPA.

The lead may run GPU lines elsewhere, and APA-MEM1 (another seat) and KA1's CPU run may be active. Never run their
commands or signal their processes.

**Rules:**
- No git, not even read-only. No subagents. No background waits: run every gate in the foreground. Every command
  stays under 10 minutes, with `timeout`.
- **No GPU.** Staged lines are dry-run only, with `FORGE_GPU_DRY_RUN=1`.
- **No deletes.** Use `delete-me` if anything must go. Never run `rm`, `rmdir`, `unlink`, `git clean`,
  `find -delete` or an inline delete call.
- Never kill a process you did not start. Never walk `/`, `/mnt` or `/mnt/Shared` recursively. Search inside named
  roots.
- **Your final message is the report.** If the harness refuses a write of `REPORT.md`, put everything in the final
  message: the lead copies it verbatim.

## Registration checklist (order-lint)

- **L01:** the instruments are ENG1's tool, NumPy and the module build. Run each with `--version` or a no-op first.
- **L02:** G2's import and collection are the smoke. G5 is the first device run.
- **L09:** the bitwise comparisons (P302, the F1 supported-dtype bytes) compare the output arrays only, viewed as
  same-width unsigned integers, with dtype and shape checked first. Logs and receipts are excluded.
- **L16:** two cheapest passing results, both excluded:
  - tolerances loosened until half passes: E3's derivation is reused unchanged;
  - an upcast that also changes float32 numerics: P302 and MN4.
  A refusal that also breaks the default path is caught by the F1 default-path test.
- **L19:** references are checked nonzero and non-constant before any bitwise or tolerance comparison.
- **L25:** both arms run in the same process kind, with one thread for BLAS and ORT.
- **L41:** each mutant is shown executing inside the test it must fail.
- **L04:** n/a. "Fallback" here is the silent dtype fallback being removed; there is no builder with fallback rungs.
- **L07:** n/a. No visual form is involved. Line 9's "look up" is a dictionary lookup.
- **L17:** n/a. There is no visual domain.
- **L30:** G3's `.so` equality is bound by reproduction, not by a stored hash. Both arms are built in this round by
  the same build script, and their sha256s are compared.
- **L38:** before freezing line (c)'s expected outcomes, trace every read of a norm class's output and every
  dtype-string result in the inherited test files: `test_eng1.py`, `test_gn1*.py`, and the files line (c) selects.
  Trace through helpers and the variables they flow into, not by grepping assert lines.
  - List each pin in `artifacts/eng2/L38_PINS.json`. A pin this order changes is listed as expected to change, with
    the new outcome; it is not a regression. One example is an E3 census row that records a half-precision error.
  - No inherited test is edited.
- **L40:** the copied tool's default writes are its RUNS root and TMPDIR. The build writes `tensor_cuda/build-eng2/`.
  Grep the copy for hard-coded `eng1` paths and roots, rebind each to `artifacts/eng2/`, and list each one with its
  sha256 before and after in `artifacts/eng2/COPY_DEPS.json`. No earlier run executed the copy.
- **L80:** the scope base is the fixed hash `bc02d3a1` (ENG1's head), never a moving ref. The lead's dispatch commit
  adds only `orders/ENG2_DTYPE_REFUSAL_AND_HALF_NORMS.md`, which G1 excludes (`':!orders'`).
- **L47:** the expected values have receipts:
  - the E3 rows above: `artifacts/eng1/tool/gpu_runs/g5_1114/candidate_eng1/attempt-0001/census_*.json`;
  - the fallback lines: `__init__.py:48,60`;
  - GroupNorm's pattern: `nn.py:569-591`.
- **L48:** the staged lines are run as written by G4's dry run. The only addition is `FORGE_GPU_DRY_RUN=1`.
  Dependents are gated on their dependency refusal, with their argv reaching exit 78. This is the ENG1-A3 lesson:
  `run_gpu_line.py:39-42`. Post-device steps are driven by the audit route.
- **L56 (b):** a failed G2, G3 or G4 is a harness defect: at most 3 repairs per gate, ledgered verbatim.
- **L60:** the census covers every dtype-string entry in `tensor_cuda/tensor_cuda/` and the C++ `dtype_from_string`.
  For each one, record the baseline behaviour for `"float64"`, `"double"`, `"int32"` and an unknown string: refused,
  mis-served (silent float32), or handled.
  - Refusing a mis-served string is the fix.
  - No supported dtype is refused.
  - Empty shapes keep today's behaviour for every supported dtype.
- **L65:** no deletes, as stated.
- **L67:** this round writes builds (about 1 GB each) and receipts, under 5 GB. The lead's `df -B1` at dispatch is in
  the dispatch commit message. Stop any write over 1 MB below 60 GB free.
- **L79:** dry runs stop before the governor's lease (`FORGE_GPU_DRY_RUN=1`, exit 78).
- **L82:** `grep -n '_NP_DTYPE' tensor_cuda/tensor_cuda/__init__.py` gives 4 hits.
  - **Writable:** the fallbacks at `:48` and `:60`.
  - **Read-only:** `:27` (the map definition, unchanged) and `:37` (a comment).
  - The `nn.py` targets are the four `forward` methods named above. GroupNorm (`:510-591`) is read-only, as the
    pattern.
- **L90:** the roles in the census:
  - product: the 2 fallback sites, plus any found;
  - harness: ENG1's copied tool;
  - test: `test_eng2.py`.
  The treatment: the product sites are fixed, the harness is copied with every change ledgered, and the test is new.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is immutable.
- A failed gate is a result, reported with receipts.
- **Prior Art Directive:** annotate the float32-statistics pattern and the refusal with known prior art, or write "no
  prior art known to me". The lead's starting list is unverified — lead to check:
  - PyTorch's half normalization accumulates in float32;
  - Micikevicius et al. (2018), "Mixed Precision Training".

## Done

Your final message gives, verbatim:
- the diff;
- G1 to G4 with their receipts;
- the dtype-string census table;
- the mutant list;
- the staged lines with their expected exits;
- P301–P304 as far as CPU can tell (P301 and P302 are GPU-scored by the lead; give the CPU emulation's
  prediction);
- every deviation;
- a "Prior art" heading.
