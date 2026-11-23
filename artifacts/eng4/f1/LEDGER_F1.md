# ENG4-F1 ledger (seat: Claude Opus 5.5, Agent tool; order orders/ENG4_F1_REV1_FIXES.md at 80b8c4cb3, sha256 2d990927...)

Times EST, 2026-11-23. No git writes (read-only git: show, archive, status, diff). TMPDIR=/tmp/eng4_tmp.
The lead's B-side lines (g2_B, g3_B, boundary_B): artifacts/eng4/gpu/, GPU_LINES.json, eng4_gpu_line.py, eng4_g2.py,
eng4_g3.py, eng4_boundary_gpu.py were not edited. Every F1 receipt is under artifacts/eng4/f1/.

## Setup
- 08:39 order read in full; findings read (tensor_fork_apa_wt_eng4rev1/artifacts/eng4_rev1/findings.v1.json,
  sha256 b4a60ee1..., read-only). L01 re-run (f1/preflight_L01_F1.txt): cmake 3.28.3, nvcc 12.6.85, pytest 9.0.2,
  Python 3.12.3 + numpy 2.4.2, compute-sanitizer 2024.3.0 rc 0, forge-gpu-run --status rc 75 (busy). Disk 118.5 GB
  free, MemAvailable 36.4 GB.
- Before copies of the four F1 files from 5174da719: f1/before_F1/ (identical to the worktree at start).

## Product and test edits (diff: f1/F1_source.diff against 5174da719)
- core.h: MatmulOperandIssue gains UnsupportedDtype (declaration only).
- matmul.cu X1: the predicate refuses a pair outside float32/float16/bfloat16 right after the dtype-mismatch check
  ("matmul supports float32/float16/bfloat16 only: operand dtype <d> (A ..., B ...)"); the dispatch's final else
  stays as an unreachable guard.
- matmul.cu X2: after the predicate: numel(out_shape) == 0 -> return NDArray(out_shape) (0 bytes, no allocation, no
  GEMM); plan.k == 0 -> NDArray::zeros(out_shape). The GEMM block's arguments are unchanged; the three GEMM calls'
  cublasStatus_t is kept and raised on non-success via cublasGetStatusName, AFTER the existing cuda_check_last (so a
  missing device keeps B's message).
- bindings.cpp X3: numpy_dtype refuses `not dtype.isnative` first: TypeError "tensor: non-native byte order NumPy dtype
  <d> (byteorder '<b>'); convert it first ...".
- test_eng4_matmul.py X4 + H1: no import from scripts/, no file read; listed-dtype test compares the array's own bytes
  (both entries); ENG4's four INT_MAX zero-size controls move to test_h1_zero_size_returns_numpy_result; new
  test_h1_unsupported_dtype_refuses, test_h1_byteorder_refuses, test_h1_tensor_casts_byteorder,
  test_h1_zero_size_returns_numpy_result, test_gpu_zero_size_returns_numpy_result.
- ops.cpp unchanged (its switch's default branch covers the new enum value; its fp32 dtype refusal is unchanged).
- scripts/eng4_build.py: optional --receipts dir (default unchanged), so F1 receipts land in f1/build.

## Build F1 (f1/build/): configure + build exit 0, module 7b80c6dca52e380a5aa2f082c00f2951f23dfe4b7b798fab6c4cde1c683817b0,
  source map 82961c6e... (47 files). Built in place, so the worktree module is F1.

## H1 (CUDA hidden, F1): f1/h1/H1_F1_try1.xml: 121 passed, 58 GPU tests skipped.
- Discrimination: same file on E (pkg build-eng4/E/pkg): 78 failed (6 byteorder, 15 unsupported dtype, 57 zero size);
  on B: 104 failed (those plus ENG4's 26). test_h1_tensor_casts_byteorder passes on B, E and F1 (unchanged result).

## H2 clean checkout (f1/H2_clean_checkout.txt, f1/H2_junit.xml, f1/H2_open_audit.txt)
- /tmp/eng4_h2/tree = `git archive HEAD -- . ':(exclude)scripts/eng4_*' ':(exclude)artifacts/eng4'` (HEAD 80b8c4cb3)
  + F1's four changed files + the F1 module: 0 scripts/eng4_* files, no artifacts/eng4. Result 121 passed, 58 skipped.
- Audit hook (sys.addaudithook 'open') during that run: inside the copy only pyproject.toml, the package and the test
  file itself are opened.
- Slip (L56(a)): the H2 command line carried a stray `cp build-eng4-placeholder 2>/dev/null` (no such file, no
  effect).

## L55 (f1/tools/l55_f1_probe.cpp, f1/l55/L55_F1_zero.jsonl)
- F1's real matmul.cu with stub cuBLAS/kernels.cu symbols, no CUDA call. M0 [3,0,5]x[5,4]: plan m0 n4 k5 batch3
  out [3,0,4] -> empty_return, no GEMM. N0 (and trans_b): out [3,2,0] -> empty_return. K0 (trans_b, class b
  [2,3,2,0]x[2,3,0,4], class c [3,2,0]x[1,1,0,4]): plan k 0 -> zeros_return with NDArray::zeros(plan out_shape),
  no GEMM. batch0 (classes a, b): empty_return. Control [3,2,5]x[5,4]: one GEMM, m(N)4 n(M)2 k5 ldB 4 ldA 5 ldC 4
  batch 3 (ENG4's mapping).

## Registrations
- MUTANTS_F1.json 08:47:3x (sha256 cbeeb706...): m01_f1..m05_f1 (ENG4's hunks via import, unchanged), m06, m07, m08;
  killing tests from a collection of the test file (f1/killmap_used_for_registration.json). m06 excludes the host
  nonempty K > 0 pair (there the late refusal follows a host malloc, not a CUDA call, with the same message).
- INTENDED_CHANGES_F1.json 08:47:53 (sha256 bb5f2ade...), before both H3 runs (08:48:15, after-run 08:52:40): suite IDs = ENG4's three M7
  IDs (carried), none new; the 12 G3 K0 cells; the test_eng4_matmul.py changes.
- Census greps (L118): rg for '>f', '>i', newbyteorder, byteswap in the test trees: one hit,
  test_apa_sp3.py:226 (`ids.astype('>i8')` saved to .npy for a protocol check that raises its own Red); H3 shows it
  unchanged. rg for int64/uint8/bool or zero extents at matmul call sites in tests: none. Product callers (ENG4
  census rows) pass float weights/activations; none fixes an int dtype or a zero extent.

## H3 CPU (CUDA hidden; scripts/eng4_f1_suite.py = eng4_suite.py with arms B/F1, restore F1, f1/g4)
- f1cpu_B 08:48:15-08:52:34 (peak RSS 2,519,972 kB), f1cpu_F1 08:52:40-08:56:32 (2,650,404 kB): both 4038 IDs, 1388 passed,
  2174 failed, 461 skipped, 15 error, 1 collection error. f1/H3_CPU_COMPARE.json: 0 differences -> PASS.
- Launch slip (L56(a)): the F1 suite's launch line chained a MemAvailable print before `&`, so its output was lost;
  checked right after: 36.1 GB.

## Mutants (H4): trees by scripts/eng4_f1_make_arms.py (diffs in f1/mutants/), builds 08:57:06-09:16:51 one at a time in own
  scopes (f1/build/BUILD_<id>.json). CPU runs (f1/MUTANT_RESULTS_F1.json): m01_f1 1/1, m02_f1 2/2, m03_f1 7/7,
  m04_f1 2/2, m05_f1 6/6, m06 12/12, m07 6/6, m08 15/15 killed on the named assertion with the executed marker in each
  killing test's capture. (X2 changes how m02_f1/m03_f1 die on their zero-element witnesses: 'DID NOT RAISE' where
  the call now returns an empty result instead of reaching the GEMM; still the named pytest.raises.)
- Frozen packages: build-eng4/{F1,m02_f1,m08_k0_gemm}/pkg (module from each receipted copy).

## INCIDENT: GPU use outside the lease (H6 "no GPU use" and ENG4's G6 "no GPU use" are both FALSE)
- compute-sanitizer does not honour CUDA_VISIBLE_DEVICES="": under it the Python child created CUDA tensors and ran
  kernels on the real GPU. My CPU plumbing smokes ran the g2 step functions (which call compute-sanitizer) with
  CUDA_VISIBLE_DEVICES="" believing the GPU hidden:
  - ENG4 smoke 08:20:16-08:20:22 EST: Build B noop (round trip ok), ENG-MM1 witness (29 errors, 48 "Invalid __global__
    read", call raised 'CUDA error at to_host: unspecified launch failure'), pooled (same); Build E noop, witness
    (refused, 0 errors), pooled (refused, 0 errors).
  - F1 smoke 09:18:19-09:18:21 EST: Build F1 noop (round trip ok), witness (refused, 0 errors).
  About 8 seconds of short CUDA contexts in total, outside forge-gpu-run, while GRAPA run 9 held the GPU.
  Evidence copied (not moved) to f1/incident_gpu_use/ with SHA256SUMS. Every other smoke child honoured the hidden
  GPU (plain python: CUDA tensor creation failed with 'cudaMalloc failed: no CUDA-capable device'; pytest children
  skipped GPU tests). No further sanitizer smoke was run after 09:18:21. ENG4's L01 no-op ran compute-sanitizer on
  /bin/true (no CUDA API call by the target; whether the sanitizer itself opened the device is not known).
- These smoke outputs are NOT gate receipts. They agree with the staged expectations (B: OOB reads; E and F1: refusal
  with 0 errors) but were not run under the lease.

## H5 staged (GPU_LINES_F1.json): 26 lines, each dry-run once with FORGE_GPU_DRY_RUN=1: all exit 78, module checks ok
  (f1/gpu_dry/). No artifacts/eng4/f1/gpu/ directory created.
## H1 final: f1/h1/H1_F1_final.xml 121 passed, 58 skipped (F1 module 7b80c6dc..., the installed worktree module).
## H6 scope: writes in the writable set except (1) the incident above and (2) three mutant build directories named
  by the order's mutant ids but outside its pattern build-eng4/m*_f1/: build-eng4/m06_dtype_after_alloc/,
  m07_byteorder/, m08_k0_gemm/ (left in place: moving them would invalidate their receipts and the staged lines).
  scripts/eng4_build.py (an ENG4 script of mine, not run by the B-side lines) gained the --receipts option.
  No git writes, nothing deleted, artifacts/eng4/gpu/ and GPU_LINES.json untouched (sha256 d7645b97... unchanged).
