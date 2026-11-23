<!-- Filed by the lead from the seat's final message (the harness refused the seat's own write twice). Content unchanged
except where marked LEAD; rows and bullets merged for length. Receipts: LEDGER_F1.md. -->

# ENG4-F1 report — the ENG4-REV1 blind findings fixed

Seat: Claude Opus 5.5 (Agent tool). Order `orders/ENG4_F1_REV1_FIXES.md` at 80b8c4cb3 (sha256 2d990927…).

> **LEAD note on the incident's cause (revised 2026-11-23 10:15 EST; it replaces an earlier lead note that was
> wrong).** The seat's diagnosis is right. Its smoke harness replaced `pyenv()` with one that set
> `CUDA_VISIBLE_DEVICES=""` (ENG4 `LEDGER.md` line 131, written at the time), and compute-sanitizer removes that
> variable from the program it launches. The lead measured it inside a forge lease (label `tensor-cvd-probe`; receipt in
> the lead ledger): with `CUDA_VISIBLE_DEVICES=""`, plain python3 saw no device (rc 100), while under
> `compute-sanitizer --tool memcheck` the target had no `CUDA_VISIBLE_DEVICES` at all and saw 1 device. The staged
> step helper `pyenv()` (`scripts/eng4_gpu_line.py:206-208`) also pops the variable, by design for use inside the
> lease. That is a second route, not the one that fired here. GRAPA run 9's log shows the 08:20:16-22 window inside
> leg 44 (08:03:20-08:21:19, rc=0, peak 10,728 MiB, in line with its neighbours; GRAPA confirmed no harm). The
> 09:18:19-21 window fell in a cooldown.

## Incident: GPU use outside the lease (makes H6 FAIL and corrects ENG4's G6 "no GPU use" to false)

The seat's CPU "plumbing smokes" ran the g2 step functions (which call compute-sanitizer) and so used the GPU outside
`forge-gpu-run` while GRAPA run 9 held it:
- ENG4 smoke 08:20:16–08:20:22 EST: Build B noop round trip, ENG-MM1 witness (29 errors, 48 "Invalid __global__ read",
  raised "CUDA error at to_host: unspecified launch failure"), pooled (same); Build E noop, witness and pooled
  (refused, 0 errors).
- F1 smoke 09:18:19–09:18:21: F1 noop round trip, witness (refused, 0 errors).

About 8 seconds of short CUDA contexts in total. Evidence copied (not moved) to `f1/incident_gpu_use/` with
`SHA256SUMS`; these are not gate receipts. Every other smoke child stayed off the GPU. No sanitizer smoke ran after
09:18:21; every staged dry run stops inside forge at exit 78. Unknown: whether ENG4's L01 no-op (`compute-sanitizer`
on `/bin/true`) opened the device.

## H1 to H6

- **H1 PASS:** CPU, CUDA hidden, on F1: 121 passed, 58 GPU tests skipped (`f1/h1/H1_F1_final.xml`). The same file fails
  78 tests on E and 104 on B; `test_h1_tensor_casts_byteorder` passes on all three.
- **H2 PASS:** clean checkout (`git archive HEAD` at 80b8c4cb3 minus `scripts/eng4_*` and `artifacts/eng4`, plus F1's
  four changed files and the F1 module): 121 passed, 58 skipped. An audit hook shows the run opens only
  `pyproject.toml`, the package and the test file (`f1/H2_*`).
- **H3 PASS:** CPU suite per ID, B vs F1: 4038 IDs each, 0 differences (`f1/H3_CPU_COMPARE.json`).
- **H4 PASS (CPU parts):** all 8 mutants killed on their named assertions with the marker seen; m02_f1 and m08 GPU
  parts STAGED.
- **H5 STAGED:** 26 lines, each dry run 78.
- **H6 FAIL:** (1) the GPU-use incident; (2) three mutant build directories sit outside the pattern
  `build-eng4/m*_f1/` (`m06_dtype_after_alloc/`, `m07_byteorder/`, `m08_k0_gemm/`), left in place because moving them
  would invalidate their receipts and staged lines. No git writes, nothing deleted. `artifacts/eng4/gpu/`,
  `GPU_LINES.json` (d7645b97…, unchanged) and the B-side scripts are untouched.

## Diff summary (`f1/F1_source.diff` against 5174da719, sha256 9fba0128…; 5 files, +219/−36)

- **X1 (F01), `matmul.cu`:** the predicate refuses dtypes other than float32/float16/bfloat16 before any allocation
  ("matmul supports float32/float16/bfloat16 only: operand dtype <d> (A …, B …)"); the dispatch's final `else` stays as
  a guard. `core.h` adds the enum value `UnsupportedDtype`.
- **X2 (F02), `matmul.cu`:** after the predicate, an empty output is returned as is (no allocation, no GEMM); K = 0
  returns `NDArray::zeros(out_shape)`; the three GEMM calls keep their status and raise on failure, naming it
  (`cublasGetStatusName`), after the existing CUDA error check. GEMM arguments unchanged.
- **X3 (F03), `bindings.cpp`:** `numpy_dtype` refuses `not dtype.isnative` with TypeError "tensor: non-native byte order
  NumPy dtype <d> …".
- **X4 (check 8), `test_eng4_matmul.py`:** no import from `scripts/`, no file reads; the listed-dtype check compares each
  array's own bytes through both entries; ENG4's zero-size INT_MAX controls moved to the zero-size test; new H1 tests
  and a GPU zero-size test (m08's line).
- `ops.cpp` unchanged. `scripts/eng4_build.py` gained a `--receipts` directory option.

L55 (`f1/l55/`): a probe on F1's real `matmul.cu` shows the zero-size witnesses take the early returns with the
predicate's own m, n, k and out_shape — M = 0, N = 0 and batch = 0 return empty; K = 0 (classes b and c too) returns
zeros; no GEMM runs; the nonzero control's GEMM arguments are ENG4's.

## Module sha256s

| Arm | sha256 |
|---|---|
| F1 | 7b80c6dca52e380a5aa2f082c00f2951f23dfe4b7b798fab6c4cde1c683817b0 (installed worktree module) |
| m01_f1 | 25dca0e6026288407ca473cc01461581851c488e834956a41668565e5a55c062 |
| m02_f1 | e0e4381ed515d5647285d1f93a6a5f3abdd3b408212ec195ceb41cc03474041c |
| m03_f1 | c820e0bf3c156d9b9803eb10f9dd90f33db6061c8632be4c3510a20673804678 |
| m04_f1 | abf6c1599d21772a3265758714522246a1979428baebf2dca70e990355980e26 |
| m05_f1 | 290302ca0ccc1d4e5e354f7b494d9a72f7c8276cdc167c62b0483f775df4a4e1 |
| m06_dtype_after_alloc | d1b99ed046bc5c47bc42322b2530193e9011c0a0f6cf5ed82cd80b804f3948bd |
| m07_byteorder | 04382dafec9ed0d094bd18d01422a2de67cb0d37ca61dfd0c222b604cee04a09 |
| m08_k0_gemm | 3b78368f74ea358b4737b1a6d9751832fd8fd282dc1a658d7d5b3e050b02e3d2 |

B (20881ace…) and E (a875936c…) unchanged.

## Intended changes (`f1/INTENDED_CHANGES_F1.json`, 08:47:53, before both H3 runs; sha256 bb5f2ade…)

- Suite IDs: only ENG4's three M7 IDs, carried over; no new IDs (H3 observed none).
- G3 cells: the 12 K = 0 cells (a_K0, b_K0, b_rank2_K0, c_K0 × 3 dtypes), held to NumPy's zeros, with B's unwritten
  values recorded beside them.
- `test_eng4_matmul.py` changes, listed per L38.

## Mutants (`f1/MUTANTS_F1.json` 08:47:33, before the first build; `f1/MUTANT_RESULTS_F1.json`)

| Mutant | CPU killing tests | Result |
|---|---|---|
| m01_f1 | 1 | KILLED 1/1 |
| m02_f1 | 2 (+2 on GPU) | KILLED 2/2; GPU line mut_m02_f1 STAGED |
| m03_f1 | 7 | KILLED 7/7 |
| m04_f1 | 2 | KILLED 2/2 |
| m05_f1 | 6 | KILLED 6/6 |
| m06_dtype_after_alloc | 12 (host K=0, empty-output and batched K=0 pairs, plus the zero-element CUDA pair; × int64/uint8/bool) | KILLED 12/12 |
| m07_byteorder | 6 | KILLED 6/6 |
| m08_k0_gemm | 15 (H1 K=0 cells) | KILLED 15/15 on "zero check"; GPU line mut_m08 (15 tests) STAGED |

m06 excludes the host nonempty K > 0 pair (the late refusal there follows a host malloc, not a CUDA call, with the same
message, so no test can see the mutant through it). m01_f1–m05_f1 are ENG4's hunks unchanged.

## Staged lines (`artifacts/eng4/GPU_LINES_F1.json`, sha256 11053ea4…)

Run from the worktree root as `python3 -B -P scripts/eng4_f1_gpu_line.py run <line>` (forge command
`forge-gpu-run --min <s> --max 540 --label eng4f1-<line> -- timeout <s>s python3 -B -P scripts/eng4_f1_gpu_line.py step <line>`);
receipts in `artifacts/eng4/f1/gpu/<line>/`.

| Line | --min | inner timeout | peak host bytes | dry run |
|---|---|---|---|---|
| g2_F1 | 240 | 480 s | 3 GiB | 78 |
| g3_F1 | 120 | 480 s | 3 GiB | 78 |
| boundary_F1 | 180 | 480 s | 3 GiB | 78 |
| own_F1 | 120 | 480 s | 3 GiB | 78 |
| mut_m02_f1 | 120 | 480 s | 3 GiB | 78 |
| mut_m08 | 180 | 480 s | 3 GiB | 78 |
| g4_B_01 … g4_B_10 | 480 | 500 s | 6 GiB | 78 each |
| g4_F1_01 … g4_F1_10 | 480 | 500 s | 6 GiB | 78 each |

The g4 lines continue the series f1gpu_B and f1gpu_F1 under a 330 s test budget per part, and restore F1 afterwards.
CPU commands after the GPU lines:
- `python3 -B -P scripts/eng4_f1_g3_compare.py artifacts/eng4/gpu/g3_B/g3 artifacts/eng4/f1/gpu/g3_F1/g3 artifacts/eng4/f1/G3_F1_COMPARE.json`
- `python3 -B -P scripts/eng4_f1_g4_compare.py f1gpu_B f1gpu_F1 artifacts/eng4/f1/H3_GPU_COMPARE.json`
- `python3 -B -P scripts/eng4_f1_mutant_results.py gpu`

## Deviations

1. The GPU-use incident (above; cause corrected by the lead).
2. Mutant arm directories m06/m07/m08 outside the `m*_f1` pattern.
3. Report file refused by the harness twice.
4. L56(a) slips: a stray no-op `cp build-eng4-placeholder` in the H2 command; the F1 suite launch lost its MemAvailable
   print (checked right after: 36.1 GB).
5. Memory-hook holds on Write: four scripts needed retries; content unchanged.
6. The g4 lines reuse ENG4's names (`g4_B_01` …) as the order lists them, in separate directories (`f1/gpu/`) under the
   label `eng4f1-`.
7. ENG4 scripts reused unchanged by the F1 lines: `eng4_g2.py`, `eng4_g3.py`, `eng4_boundary_gpu.py`;
   `eng4_make_arms.py` imported read-only.

## Prior art

cuBLAS leading-dimension requirement (lda/ldb/ldc ≥ max(1, rows)), `cublasStatus_t` and `cublasGetStatusName` (cuBLAS
documentation); BLAS reference xGEMM quick return (M or N = 0) and C set to zero when K = 0 with beta = 0; NumPy
`matmul` on zero-size operands and `dtype.isnative`; this repo's ENG3-F5 self-contained test pattern and ENG2 F1's dtype
refusal; mutation testing (DeMillo, Lipton and Sayward, 1978); link-time stubs for the L55 probe (unverified, lead to
check). Ours: the placement of each check and early return, and the probes.

## Files written

Engine: `tensor_cuda/src/matmul.cu`, `tensor_cuda/src/bindings.cpp`, `tensor_cuda/include/tc/core.h`,
`tensor_cuda/tests/test_eng4_matmul.py`, the installed module (F1; swapped to B and back during f1cpu_B). Scripts:
modified `scripts/eng4_build.py`; new `scripts/eng4_f1_make_arms.py`, `eng4_f1_suite.py`, `eng4_f1_g4_compare.py`,
`eng4_f1_g3_compare.py`, `eng4_f1_gpu_line.py`, `eng4_f1_mutant_results.py` (all `python3 -B -P`; no stdlib names).
`artifacts/eng4/GPU_LINES_F1.json`; `artifacts/eng4/f1/` (`LEDGER_F1.md`, `preflight_L01_F1.txt`, `before_F1/`,
`F1_src/`, `F1_source.diff`, `build/` (9 arms), `h1/`, `H2_*`, `l55/`, `tools/l55_f1_probe.cpp`, `MUTANTS_F1.json`,
`killmap_used_for_registration.json`, `mutants/`, `MUTANT_RESULTS_F1.json`, `INTENDED_CHANGES_F1.json`,
`g4/f1cpu_B`, `g4/f1cpu_F1`, `H3_CPU_COMPARE.json`, `gpu_dry/` (26), `incident_gpu_use/`). Builds:
`tensor_cuda/build-eng4/F1/`, `m01_f1/` … `m05_f1/`, `m06_dtype_after_alloc/`, `m07_byteorder/`, `m08_k0_gemm/`;
`pkg/` for F1, m02_f1, m08. Logs: `logs/eng4_f1_build_F1.log`, `eng4_f1_build_mutants.log`, `eng4_f1_h3_cpu_B.log`,
`eng4_f1_h3_cpu_F1.log`. Scratch left and listed: `/tmp/eng4_h2/`, `/tmp/eng4_l55f1/`, `/tmp/eng4_src_m0*_f1/`,
`/tmp/eng4_src_m06_dtype_after_alloc/`, `/tmp/eng4_src_m07_byteorder/`, `/tmp/eng4_src_m08_k0_gemm/`, `/tmp/eng4_tmp/`
(including `smoke/` and `smoke_f1/`, behind the incident).
