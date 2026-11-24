# ENG4-F1: fix the ENG4-REV1 blind findings (dtype check placement, zero-size GEMMs, byte order, a self-contained test)

Follows: ENG4_MATMUL_CONTRACT.md

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng4`
(branch `eng4`, at the commit carrying this order). The writable set is ENG4's (`orders/ENG4_MATMUL_CONTRACT.md`), with
these differences:
- the new build arm is `tensor_cuda/build-eng4/F1/` (and mutant arms `tensor_cuda/build-eng4/m*_f1/`). Leave
  `build-eng4/B/`, `E/` and the existing mutant arms unchanged;
- receipts go under `artifacts/eng4/f1/`;
- new GPU lines go in `artifacts/eng4/GPU_LINES_F1.json`, with new line names. `GPU_LINES.json`, `artifacts/eng4/gpu/`,
  and the scripts that `g2_B`, `g3_B` and `boundary_B` run are read-only: the lead is running those three lines now.
  If an F1 line needs different logic, add a new script or a new mode;
- the installed worktree module becomes F1 at the end of the round.

**Read-only:** as in ENG4, plus the review's worktree `tensor_fork_apa_wt_eng4rev1` (its `artifacts/eng4_rev1/`).
**GPU:** none. You stage and dry-run; the lead runs.

## Why

The Codex review ENG4-REV1 (run 20261123T133050Z-3954548) sealed its blind first pass
(`tensor_fork_apa_wt_eng4rev1/artifacts/eng4_rev1/findings.v1.json`, sha256 `b4a60ee1`). Then a provider content
filter stopped it. Its findings, none blocking:
- **F01 (MAJOR):** an int64, uint8 or bool pair clears the operand predicate. `NDArray out` is then allocated before the
  dtype refusal (`matmul.cu`, the `else throw` after the GEMMs), so a CUDA allocation runs before that refusal.
- **F02 (MAJOR, pre-existing):** zero extents reach the GEMM.
  - With K = 0 and positive M and N, the call passes lda = 0 (invalid for cuBLAS). The discarded status leaves `out`
    unwritten, where NumPy gives zeros. The fp32 backward already returns zeros for K = 0 (`ops.cpp`, `if (!k)`).
  - With N = 0 it passes ldc = 0.
- **F03 (MINOR, pre-existing):** `numpy_dtype` matches kind and size only. A non-native byte order of a listed dtype (for
  example `>f4`) is accepted and its bytes are copied unswapped.
- **Check 8:** `tests/test_eng4_matmul.py`'s N1 test imports a script under `scripts/` and reads a recorded JSON from the
  repo. On a clean production checkout (no `scripts/eng4_*`, no ignored receipts) it fails. This is the TV1 /
  TEST-ARTIFACT-DEPS class.

## The fixes (binding)

- **X1 (F01).** The supported-dtype check (float32, float16, bfloat16) moves into `matmul_operand_plan`, before any
  allocation. The message keeps the prefix "matmul supports float32/float16/bfloat16 only". The fp32 backward's own
  dtype refusal is unchanged.
- **X2 (F02).** In the forward, after the predicate accepts:
  - an output with zero elements is returned without a GEMM;
  - K = 0 with a nonempty output returns zeros of the output shape and dtype, without a GEMM (as the fp32 backward
    does).

  The forward checks the cuBLAS status of the GEMM it does call, and raises on a non-success status, naming it. This
  changes the result only where Build B returned unwritten memory (K = 0); list those cells as intended changes.
- **X3 (F03).** `numpy_dtype` refuses a non-native byte order of a listed kind and size, naming the dtype. Public
  `tc.tensor` casts first, so its results do not change; `tc.from_numpy` and `_C.tensor` refuse.
- **X4 (check 8).** `test_eng4_matmul.py` imports nothing from `scripts/` and reads no file outside the test. For the
  listed-dtype check, compute the expected bytes in the test (the array's own bytes), so it passes on a clean checkout.
  Show it: run the file from a copy of the tracked tree with `git archive` (read-only git) plus the F1 module, with no
  `scripts/eng4_*` and no `artifacts/eng4/`.
- **X5.** Every other ENG4 ruling stands (M1 to M8, N1). No knob.

## Gates

- **H1, CPU** (`CUDA_VISIBLE_DEVICES` empty), on F1: all ENG4 G1 tests, plus these, through the public API on host
  operands:
  - int64, uint8 and bool pairs refuse before any CUDA call;
  - `>f4`, `>f2` and `>i8` refuse on `tc.from_numpy` and `_C.tensor`;
  - `tc.tensor` of a `>f4` array keeps its values.
- **H2, the clean-checkout run** of X4 passes.
- **H3, CPU suite per ID,** Build B against F1: identical outcomes except the listed intended changes.
- **H4, mutants,** written in `artifacts/eng4/f1/MUTANTS_F1.json` before any runs:
  - ENG4's five, rebuilt on F1's source as `m01_f1` to `m05_f1`, each killed as before (`m02_f1` CPU part, plus a staged
    GPU line);
  - `m06_dtype_after_alloc`: the dtype check moved back after the allocation (H1, host int64 pair);
  - `m07_byteorder`: the byte-order check dropped (H1);
  - `m08_k0_gemm`: the K = 0 early return dropped (a staged GPU line; it must fail on the zero check).
- **H5, staged GPU lines** for F1, in `GPU_LINES_F1.json`, dry-run 78 each, in ENG4's form (≤ 500 s, `--min` ≤ 480,
  write-once paths listed):
  - `g2_F1`: the ENG-MM1 refusal under memcheck, 0 errors;
  - `g3_F1`: the G3 grid. Nonzero cells must be bitwise equal to `g3_B`'s outputs (which the lead is producing now at
    `artifacts/eng4/gpu/g3_B/`). Zero-size cells must equal NumPy's result (an empty array, or zeros for K = 0). Record
    B's zero-size values beside them;
  - `boundary_F1`, `own_F1`, `mut_m02_f1`, `mut_m08`;
  - `g4_B_01` … and `g4_F1_01` …: the suite on GPU, Build B and F1, on this round's test tree. Re-use ENG4's g4 logic
    under the new names.
- **H6, scope:** writes only in the writable set; no GPU use; no git writes; nothing deleted.

**Registered acceptance:** the seat's part passes when H1 to H4 (CPU parts) and H6 pass and H5 is staged and dry-run
clean. ENG4 with F1 is ACCEPTED when the lead's GPU lines pass. A Codex review then continues, covering ENG4 and F1.

## Mechanism & scope

- **Measures:** whether F01, F02, F03 and check 8 are fixed without changing any other accepted result.
- **Mechanism doc:** `orders/ENG4_MATMUL_CONTRACT.md` (M1 to M8, N1) and this order (X1 to X5).
- **Exercises:** `matmul.cu` (the predicate and the forward's tail), `bindings.cpp` (`numpy_dtype`),
  `test_eng4_matmul.py`; Builds B and F1, plus mutant arms; CPU tests; a clean-checkout copy; staged GPU lines.
- **Does not cover:** new broadcasting; host-tensor compute (HOST-COMPUTE-HMM); a current-device check; performance;
  promotion.
- **Required evidence:** the F1 diff against `5174da719`; F1 and mutant build receipts with module sha256s; H1 to H4
  results; `INTENDED_CHANGES_F1.json` (written before the after-run); `GPU_LINES_F1.json` with dry-run exits;
  `artifacts/eng4/f1/REPORT_F1.md`.
- **Operand census:** the F01 dtypes (int64, uint8, bool); the zero-size cells (M, N, K, batch = 0); the byte orders;
  every test file read and import in `test_eng4_matmul.py`.
- **Baseline isolation:** Build B (production `f7282b1d`) for results; Build E for the predicate's refusal classes.
- **Forbidden configuration:** any change to a nonzero-size accepted result; a refusal after a CUDA call; editing B-side
  lines or their scripts; edits outside the writable set; git writes; deleting anything yourself.

## Machine rules

As in ENG4 (one CPU-busy process at a time at `nice -n 19`; builds and jobs over 1 GB in their own scope with
`MemoryMax=12G` and `OOMPolicy=continue`, started only with `MemAvailable` ≥ 12 GB; stop under 40 GB free on
`/mnt/ForgeRealm`). The ceiling is 4 hours of wall time.

## Registration checklist (order-lint)

- **L01:** the instruments are ENG4's. Re-run their `--version` checks once.
- **L06:** n/a. No count threshold; `MemAvailable` ≥ 12 GB is a memory floor.
- **L18:** n/a. No owner requirement is implemented behind an option; no knob (X5).
- **L25:** both sides of `g3_F1` run the same cuBLAS calls (the GEMM block is unchanged) on the same device, each in
  its own process, with no host thread pool on the path. The grid covers classes (a), (b) and (c), `trans_b` both
  ways, `alpha` 1 and 0.5, three dtypes and the zero-size cells. The production call shapes are ENG4's census rows
  (classes a and b), and ENG4's L55 probe shows their GEMM arguments are mapped by unchanged code.
- **L37:** n/a. No lever, knob or spec key.
- **L67:** measured at `5174da719`: one module build arm is 46–65 MB (`build-eng4/m01_count_not_shape` 46,410,692 B;
  `E` 65,021,414 B), and ENG4's receipts are 22,831,424 B. This round adds about 9 arms (F1 and 8 mutants) plus
  receipts, under 1 GB. The disk had 118,559,555,584 B free at sealing (`df -B1`, Mon Nov 23 08:38 EST). The stop is
  40 GB free, checked by you before each build; on a hit, stop cleanly and write the report.
- **L02:** GPU lines carry an explicit `--min`.
- **L04:** no fallback. X1 and X3 refuse. X2 returns NumPy's result for zero sizes, as the fp32 backward already does.
- **L09:** H5's `g3_F1` compares per element on same-width unsigned views, after dtype and shape checks. Zero-size
  cells compare against NumPy's array: an empty array is compared by shape and dtype; zeros by value.
- **L10:** nonzero-size accepted results are held to identity with Build B.
- **L12:** inherited gates: ENG4's G1 and the suite per ID. Intended changes are listed before the after-run.
- **L16:** the cheapest passing work would fix X1 and X3 and leave K = 0 unwritten. Excluded by `g3_F1`'s NumPy check
  on K = 0 and by `m08_k0_gemm`.
- **L19:** every bitwise comparison first checks that each side is nonzero and non-constant, at full width (zero-size
  cells use L09's rule instead).
- **L21:** public remote: no personal names, no raw quotes, home paths as `$HOME/...`.
- **L35:** registrations may fix cases and seeds; they may not narrow anything a gate covers. If a clause cannot hold,
  stop and report RED.
- **L38:** existing tests are unchanged except `test_eng4_matmul.py` (X4), listed.
- **L40:** pytest with `-p no:cacheprovider`, `--basetemp` and `TMPDIR` under `/tmp/eng4_*`; TERRAIN-TEST-WRITES as in
  ENG4.
- **L41:** each mutant names its expected failing assertion before it runs. Show that the killing test executes the
  mutated line.
- **L45:** staged lines record `forge-gpu-run`'s exit, the inner `timeout` exit and the log's sha256.
- **L47:** the mutants' expected failures come from this order.
- **L55:** X2 holds only if the early returns see the same M, N, K the predicate computed. Show one witness each for
  M = 0, N = 0, K = 0 and batch = 0.
- **L56:** (a) a slip in running your own tools goes under Deviations; (b) an exception in your own scripts is
  repaired at most 3 times, each recorded.
- **L58:** H1 runs with `CUDA_VISIBLE_DEVICES` empty.
- **L76:** the lead writes nothing in the writable set while the seat runs, except the running B-side lines' outputs
  under `artifacts/eng4/gpu/` and `logs/eng4_lead_*`.
- **L77:** every mutant can fail, and each is run (GPU ones as staged lines).
- **L79:** `FORGE_GPU_DRY_RUN=1` returns 78. Use it once per staged line.
- **L99:** the lead's readings at `5174da719`, run before sealing:
  - `matmul.cu`: `NDArray out(...)` comes before the dtype dispatch's `else throw`;
  - the GEMM calls pass `(int)K` as A's leading dimension and `(int)N` as C's, and discard the returned status;
  - `ops.cpp`: the fp32 backward returns `NDArray::zeros` when `k == 0`;
  - `bindings.cpp`: `numpy_dtype` tests `kind()` and `itemsize()` only.
- **L103:** each staged line refuses, before the lease, a module whose sha256 differs from its build receipt.
- **L104:** no ratio bar; every gate is exact.
- **L109:** helpers run as `python3 -B -P`; none shares a standard-library module name.
- **L114:** big jobs in the seat's own scope (Claude seat). Staged GPU lines set no RLIMIT_AS; the lead runs them in a
  `MemoryMax=12G` scope, and `GPU_LINES_F1.json` lists each line's expected peak host bytes.
- **L118:** the consumers of the changed checks are ENG4's census rows. Re-check only the F01 dtypes and the zero-size
  callers (none expected).
- **L119:** n/a. The diff is product code and its test. Findings follow ENG4-REV1's rule: blocking = a named code path
  giving a wrong result, an out-of-storage access, or a refusal of a combination that was correct before.

## Rules

- This order is immutable. NO git writes (read-only git, including `git archive` for H2, is allowed). NO subagents.
  No background waits; every command under an explicit timeout.
- No deletes, in the leave-and-list form. Never kill a process you did not start, and never touch the lead's running
  lines.
- **Report:** write `artifacts/eng4/f1/REPORT_F1.md` and keep it current. If the write is refused, your final message is
  the report.
- **Prior Art Directive:** annotate each code site and the report. Known: the cuBLAS leading-dimension requirements
  (cuBLAS documentation: lda ≥ max(1, rows)); BLAS's convention that C need not be set when beta = 0; NumPy's
  `matmul` with zero-size operands and `dtype.isnative`; ENG3-F5's self-contained test pattern (this repo).

## Done

Your final message gives, verbatim: H1 to H6 (PASS, FAIL or STAGED); the diff summary; every module sha256; the
intended changes; the mutant table; each staged line with its dry-run exit; deviations; every file written.
