# ENG4-F3 ledger (seat: Claude Opus 5.5, Agent tool; order orders/ENG4_F3_REV3_FILL_STATUS.md at 251588034, sha256 f5404764...)

Times EST, 2026-11-23. Worktree tensor_fork_apa_wt_eng4f3 only; tensor_fork_apa_wt_eng4 was only read (the F2 module
copy by path, sha256-checked). No GPU use of any kind: no staged line, no step function, no compute-sanitizer, nothing
that removes CUDA_VISIBLE_DEVICES; each staged line's only smoke is its FORGE_GPU_DRY_RUN=1 dry run. No git writes.

## Setup (10:30-10:32)
- Order read in full. L01 (f3/preflight_L01_F3.txt): cmake 3.28.3, nvcc 12.6, pytest 9.0.2, Python 3.12.3 + numpy 2.4.2,
  forge status busy. Disk 93.5 GB free, MemAvailable 32.1 GB.
- F2 sources: git show 9b7eb1796:<file> into f3/F2_src (match BUILD_F2.json's source map); the eng4 worktree's
  build-eng4/F2 module sha256 c913e8d9... = BUILD_F2.json; copied (scripts/eng4_f3_make_arms.py copy_f2) into this
  worktree's build-eng4/F2/module, receipt f3/build/BUILD_F2_copy.json.

## K4 reading (before any edit)
- tensor_cuda.empty consumes a failed allocation's slot entry (bindings.cpp tensor_empty, P1T-FIX4 sweep): no pending
  error through it.
- The forward's own output allocation does leave one: kernels.cu Storage::Storage (CUDA, pooling off) retries a failed
  cudaMalloc once (clearing the slot before the retry) and throws "cudaMalloc failed: ..." without consuming the retry's
  entry. A K = 0 matmul of zero-element CUDA operands (2^19, 0) x (0, 2^19) needs a 1 TiB output: building the operands
  makes no CUDA call (0 bytes; from_host returns before cuda_check_last), the allocation fails at once, and
  cudaErrorMemoryAllocation (non-sticky) stays pending. NDArray::full returns early for zero elements, so an empty
  backward seed makes no CUDA call either.
- So a pending error can be made through the public API -> K4 witness line pending_F2_F3 (forward variant gated; a
  gradient variant recorded). F2 should raise it under "matmul zero fill"; F3 should return zeros, the stale error
  surfacing at the next general check (tc.synchronize, which drains it).

## Edits (f3/F3_source.diff against 9b7eb1796; new scripts as new-file hunks)
- core.h: declaration of zeros_checked.
- matmul.cu: zeros_checked beside the forward: NDArray(shape, dtype, device) as NDArray::zeros allocates; 0 bytes: no
  fill, no CUDA call; host: std::memset; CUDA: cudaMemset with its returned status checked (raise "<where>: cudaMemset
  failed: <name>: <string>"); never reads or clears the slot. K = 0 path: return zeros_checked(plan.out_shape, a.dtype,
  a.device, "matmul zero fill"). The F2 cuda_check_last there is gone.
- ops.cpp: the empty-batch branch: zero_b = zeros_checked(b_data.shape, b_data.dtype, b_data.device, "matmul backward
  zero gradient"); its cuda_check_last is gone (an empty B now makes no CUDA call at all).
- test_eng4_matmul.py: test_k1_empty_b_no_fill (host, float16/bfloat16, trans_b both) and
  test_gpu_k4_pending_error_not_raised_by_fill (GPU, own_F3).
- L55 (f3/l55/L55_call_sites.txt): both sites pass the same shape, dtype and device as in F2 (plan.out_shape, a.dtype,
  a.device; b_data.shape, b_data.dtype, b_data.device); the helper allocates with the same NDArray constructor call as
  NDArray::zeros; zeros_checked has exactly these two callers (L118).

## Builds (f3/build): F3 10:32:51-10:35:38 exit 0, module f005d765e3aed86b537c9660bea4a2bfffc86a121d724c6d7086ec399baeae96;
  m11_f3 10:45:05-10:47:59 exit 0, module 6624bc64cae5d350d4a61eb6edd44917fa558ca0b191283ad8ec65de91e8e604.

## K1 (CUDA hidden): f3/k1/K1_F3.xml 157 passed, 83 skipped; the same file on F2 (build-eng4/F2/pkg): 157 passed,
  83 skipped; every test passing on F2 passes on F3.
## K2: f3/K2_clean_checkout.txt: git archive of 251588034 minus scripts/eng4_* and artifacts/eng4, plus F3's four
  files and module: 157 passed, 83 skipped; opened outside the package: pyproject.toml, the test file.
## K3: f3cpu_F2 10:37:48-10:40:35, f3cpu_F3 10:40:53-10:43:54 (scripts/eng4_f3_suite.py): 4038 IDs each,
  1241 passed / 1991 failed / 431 skipped / 375 error / 1 collection error in this worktree (it lacks the eng4
  worktree's gitignored fixtures, hence more errors than ENG4's runs; identical per ID); f3/K3_CPU_COMPARE.json PASS.
## K5: MUTANTS_F3.json 10:33 (before the m11 tree and build): m11_f3_status_ignored registered UNKILLABLE BY TEST with
  its reason. Built and run on the CPU file: 157 passed (survives, as registered); its line is in the CUDA branch, so
  the marker never runs on the CPU lane (f3/MUTANT_RESULTS_F3.json).
## L48: f3/POST_DEVICE_STEPS.json: 23 steps, all at their expected labels (g3z host readback of every zero-size cell
  + judge/verdict negatives; pending judge positives and negatives + verdicts; own junit counting; lease_guard False).
## K6: f3/GPU_LINES_F3.json: own_F3, g3z_F3, pending_F2_F3; one dry run each, all exit 78, module checks ok; no
  f3/gpu directory created.
## K7: writes only in the writable set (git status listing in the report). Deviation: scripts/eng4_build.py's scope
  sets TMPDIR=/tmp/eng4_tmp for the compiler (an existing script, run unchanged), so build scratch went there rather
  than /tmp/eng4f3_*.
