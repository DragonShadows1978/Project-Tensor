# ENG4-F2 ledger (seat: Claude Opus 5.5, Agent tool; order orders/ENG4_F2_REV2_HARDENING.md at 3f8bffe80, sha256 c6ca7096...)

Times EST, 2026-11-23. No git writes (read-only git: show, archive, status, diff). No GPU use of any kind: no line, no
step function and nothing that removes or overrides CUDA_VISIBLE_DEVICES was run; the only "smoke" of each staged
line is its FORGE_GPU_DRY_RUN=1 dry run. TMPDIR=/tmp/eng4_tmp.

## Setup
- 09:47 order read in full; REV2 findings.json (sha256 cb615fb3...) and CPU_WITNESSES.log (73d69284...) read
  (read-only). L01 (f2/preflight_L01_F2.txt): cmake 3.28.3, nvcc 12.6, pytest 9.0.2, Python 3.12.3 + numpy 2.4.2,
  compute-sanitizer 2024.3.0, forge status cooldown. Disk 106.6 GB free, MemAvailable 34.0 GB.
- L99 readings confirmed at HEAD: matmul.cu:98 `p.batch *= a.shape[d]`, :103 `p.batch > INT_MAX`, :153 the K = 0
  `return NDArray::zeros(...)`. Before copies: f2/before_F2/. (artifacts/eng4/f1/MUTANT_RESULTS_F1.json was already
  modified in the worktree at start: the lead's GPU merge; untouched by me.)
- Record of the F1 incident cause (for L58): in both earlier smokes my wrapper had replaced pyenv() with a version that
  set CUDA_VISIBLE_DEVICES=""; the plain-python children honoured it (CUDA tensor creation failed), the
  compute-sanitizer children reached the GPU. Either way, direct step calls are the route; none was made this round.

## Edits (diff f2/F2_source.diff against 3f8bffe80, new scripts as new-file hunks)
- matmul.cu Y1: out_shape built first; a zero leading extent -> batch 0; otherwise extent by extent, refusing
  (BatchCount, "GEMM batch count above INT_MAX: the product of A's leading extents exceeds 2147483647 at leading
  dimension <d> (<product> x <extent>)") when `extent > INT_MAX / batch`; no product above INT_MAX is ever formed.
- matmul.cu Y3: K = 0 path keeps NDArray::zeros, then `cuda_check_last("matmul zero fill")` for CUDA outputs.
- ops.cpp Y2: in Tensor matmul's gradient, after A's (empty) gradient: unequal ranks and a zero leading extent of A
  -> B gets NDArray::zeros(B's shape, dtype, device) (fill checked with cuda_check_last for CUDA) and returns before
  the dB GEMM and reduce_to. reduce_to, NDArray::zeros and every core member unchanged.
- test_eng4_matmul.py: new J1 tests (Y1 zero-leading, refuses-before-overflow, BatchCount boundary; Y2 host
  float16/bfloat16 zero gradient) and test_gpu_bwd0_zero_gradient; still self-contained.

## Build F2 (f2/build): 09:49:24-09:51:49 exit 0, module c913e8d9e1eff88e6cbc0f68f6f2d03076baa022cc6addd92843215dea7fbd32.
## J1 (CUDA hidden, F2): f2/j1/J1_F2_try1.xml and J1_F2_final.xml: 153 passed, 82 GPU tests skipped.
- Same file on F1 (build-eng4/F1/pkg, read-only import): 20 failed = 4 test_j1_y1_refuses_before_overflow (F1 accepted
  the wrapped products and stopped at the output byte bound: OverflowError) + 16 Y2 cases ('reshape: element count
  mismatch'); test_j1_y1_zero_leading_is_batch0 passes on F1 as well (a zero-holding naive product is 0 too).
## J2 (f2/J2_clean_checkout.txt, J2_junit.xml): git archive HEAD (3f8bffe80) minus scripts/eng4_* and artifacts/eng4,
  plus F2's three changed files and the F2 module: 153 passed, 82 skipped; files opened outside the package:
  pyproject.toml, the test file (and its pyc lookup).
## L55 (f2/tools/l55_f2_probe.cpp, f2/l55/): F2's real matmul.cu, stubbed cuBLAS, no CUDA call. Accepted: zero leading
  (2^32,2^32,0) and (0,2^32) batch 0 = out batch product 0 (empty return); (2147483647) and (1,2147483647) batch
  2147483647 = out product; (32768,65535) 2147450880; control 3. Refused at the first offending dimension:
  (65536,32768), (2^31), and the naive-over-2^63 (2^32,2^32) and (3,2^62). Slip: the probe's stub numel_of (copied
  from F1's) multiplied before checking for 0; fixed to check zero first before compiling.
## Registrations
- MUTANTS_F2.json 09:54:07 (sha256 db6b5ed9...), before any mutant tree or build: m09_f2_naive_product (4 killing
  tests; the order's 2^32 x 2^32 x 0 witness recorded as unkillable by test, with the reading), m10_f2_empty_batch_raise
  (16 CPU + 24 GPU tests).
- INTENDED_CHANGES_F2.json 09:54:2x (sha256 fa558007...), before both J3 runs: suite IDs = ENG4's M7 IDs carried,
  none new; Y2 raise -> zero gradient on the listed pairs; Y1 over-2^63 products now refused.
## J3 CPU (scripts/eng4_f2_suite.py): f2cpu_B 09:54:38-09:59:22, f2cpu_F2 09:59:28-10:04:04: 4038 IDs each, 1388/2174/
  461/15 + 1 collection error; f2/J3_CPU_COMPARE.json 0 differences -> PASS.
## J4 mutants: arms build-eng4/m09_f2 (a2e81be1...) and m10_f2 (5e20548f...), 10:04:32-10:09:53.
- CPU run 1: m10 16/16 killed; m09's 4 tests all failed with the mutant's OverflowError escaping
  pytest.raises(RuntimeError), which MUTANTS_F2.json registered as m09's named failure, but the classifier's text list
  was narrower. L56(b) repair 1 of 3: the classifier gained the registered OverflowError text; run 1 kept as
  f2/mutants/superseded_try1/. Run 2: m09 4/4, m10 16/16 killed, marker seen in each killing test.
## L48: f2/POST_DEVICE_STEPS.json: 34 steps, all at their expected terminal labels (readback per dtype incl. empty and
  largest; g3 compare identical/bitflip/K0 nonzero/empty wrong shape/L19 constant/dtype; bwd0 host record+judge,
  verdict all-zero/one bit/raised/shape/dtype/empty/died/B mode; mut_m10 classify failed-named/other/passed/skipped/
  died; lease_guard False outside forge).
## J5: GPU_LINES_F2.json (sha256 7dc296c6...): 25 lines, each dry-run once: all exit 78, module checks ok; no
  artifacts/eng4/f2/gpu/ directory created.
## J6: writes only in the writable set (see the report's file list); B/E/F1 arms, GPU_LINES.json (d7645b97...),
  GPU_LINES_F1.json (11053ea4...), artifacts/eng4/gpu/ and f1/gpu/ untouched; no GPU use; no git writes; nothing
  deleted (two files moved, not removed: MUTANT_RESULTS_F2 run 1 into f2/mutants/superseded_try1/, and a first
  F2_source.diff that included the lead's file, moved to /tmp/eng4_tmp/).
- sed slip (L56(a)): the first derivation of eng4_f2_mutant_results.py failed on a '#' in the replacement and left
  an empty file; regenerated at once. A pgrep wait slip ('\|' in ERE) ended a wait early; re-waited.
