# ENG4 ledger (seat: Claude Opus 5.5, Agent tool; order orders/ENG4_MATMUL_CONTRACT.md at 345a77242, sha256 d4f42fd3...)

Times EST (UTC-5), 2026-11-23, unless marked UTC. No git writes; read-only git only. TMPDIR=/tmp/eng4_tmp.

## Setup
- 07:35 order read in full (sha256 d4f42fd3cec1570238c21aade83e742827a0b42d0985e5bd2c1e8bd4e41fd62a), HOUSE_RULES.md,
  AGENTS.md. Mechanism docs read: matmul.cu:25-83, ops.cpp:783-851, core.h:9-11, bindings.cpp:77-85. ENG3's LEDGER.md
  and scripts (eng3 worktree, read-only) for the build commands.
- 07:36 L01 (`preflight_L01.txt`): cmake 3.28.3; nvcc 12.6.85; pytest 9.0.2; Python 3.12.3 + numpy 2.4.2;
  compute-sanitizer 2024.3.0 (the $HOME/.local/bin wrapper to the CUDA 12.6 build) `--version` rc 0; no-op
  `compute-sanitizer --tool memcheck /bin/true` with CUDA hidden: rc 255, "Target application terminated before first
  instrumented API call" (the instrument loads and runs; /bin/true makes no CUDA call, so nothing is instrumented; a
  CUDA no-op needs the GPU and is staged inside the G2 lines). forge-gpu-run `--status --min 30 --max 540` rc 75,
  state busy (GRAPA run 9 holds the GPU, as the order says). Disk 125.2 GB free; MemAvailable 34.9 GB.
- 07:40 before copies of the writable engine files and the M7 test file: artifacts/eng4/before/ (from `git show HEAD:`).
  Build B source tree: `git archive HEAD` of tensor_cuda/{CMakeLists.txt,src,include,tensor_cuda} into
  /tmp/eng4_src_B (diff -r against the worktree src: identical).

## Build B
- configure_B 12:39:32-12:39:37 UTC exit 0; build_B 12:39:37-12:42:03 UTC exit 0 (`build/build_receipts.json`).
  Command (scripts/eng4_build.py): `systemd-run --user --scope -q -p MemoryMax=12G -p OOMPolicy=continue
  -p CPUWeight=50 --setenv=TMPDIR=/tmp/eng4_tmp --setenv=CUDACXX=/usr/local/cuda-12.6/bin/nvcc nice -n 19 cmake
  -S /tmp/eng4_src_B/tensor_cuda -B tensor_cuda/build-eng4/B -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
  -DFETCHCONTENT_SOURCE_DIR_PYBIND11=<production>/tensor_cuda/build/_deps/pybind11-src
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON`, then `cmake --build ... -j6` (ENG3's flags). The build ran detached and was
  waited on with `timeout 590 tail --pid`.
- Module sha256 20881ace0ef7f3fd1feb324f1f75a63589d523d0f943857de7564c5f0e297c40 (tensor_cuda/build-eng4/B/module/);
  source map 47 files, sha256 69e283c0... (`build/BUILD_B.json`).

## L99 seat check (Build B, CUDA hidden, host operands; `boundary/B_cpu.json`, `boundary/B_cpu.log`)
- ENG-MM1 shapes A [4,2,3] x B [2,3,4] float32 host: RuntimeError "CUDA error at matmul: no CUDA-capable device is
  detected". That message comes from `cuda_check_last("matmul")` (matmul.cu:81), which runs after the cuBLAS call
  (reasoning from source): no refusal; the call reaches cuBLAS (whose handle creation fails with no device).
- Every M3 and M4 witness on Build B gives the same CUDA error, except host A + zero-element CUDA B -> the same, and
  zero-element CUDA A + host B -> "cudaMalloc failed: no CUDA-capable device is detected" (the output is allocated on
  A's device before the GEMM). Build B refuses none of them.

## Census (M1/N1) reading, before any product edit (07:41-07:45)
- L82 grep + rg over tensor_cuda/src and the package: C++ callers ops.cpp:785/826/864/903, kernels.cu:4336/4818,
  gemm_apa.cu:779, bindings 2050 (`__matmul__`) and 2075; Python nn.py:107/386/448, functional.py:171/177/200,
  quant.py:125/127/248/291/292/331, apa_packed.py:389, __init__.py:343. Each that fixes ranks passes class a or b
  (gemm_apa: validate() fixes v's [B,KVH]); none fixes a refused pair -> no M1 RED. The census FILE was written
  later (08:00, scripts/eng4_census.py), after the edits; its rows are checked against Build B's lines.
- N1 reading: numpy_dtype's only caller is tensor_from_numpy (bindings.cpp:88), whose only binding is `_C.tensor`
  (bindings.cpp:2071). The Python wrappers tc.tensor, the factories, embedding, cross_entropy and the bfloat16 path
  cast to a listed dtype first; tc.from_numpy does not. No consumer-root site calls tc.from_numpy or _C.tensor.
- Build B on unlisted dtypes (`n1/B_cpu_unlisted.txt`): float64 -> float32 garbage values; int32 -> float32 bit
  patterns of the integers; uint16 -> six float32 read from a 12-byte buffer, i.e. a 24-byte read: a HOST
  out-of-bounds read of 12 bytes beyond the NumPy buffer (reasoning: numel 6 x 4 bytes vs 6 x 2 bytes held).

## Product edits (before copies in artifacts/eng4/before/)
- core.h: declarations only: enum MatmulOperandIssue, struct MatmulOperandPlan, matmul_operand_plan().
- matmul.cu: matmul_operand_plan (rank, dtype, inner dim, device, INT_MAX for M/N/K, batch count, batch classes
  a/b/c); tc::matmul calls it first and throws before any CUDA call; the GEMM code below is unchanged and reads
  M, K, Kb, N, batch, out_shape and b_batched from the plan. The stride rule stays (ndb == nda) -> K*N else 0, so
  rank 2 x rank 2 keeps stride K*N with batch 1. Old messages kept as prefixes ("matmul needs >=2D inputs",
  "matmul dtype mismatch", "matmul inner dim mismatch"); new refusals start "matmul refused: " and name both shapes
  and devices.
- ops.cpp: matmul_backward keeps its CUDA fp32 same-device check, then uses the plan (batch SHAPE check replaces the
  count check; messages keep the BWD text "invalid GEMM dimensions", "invalid GEMM batch count", "batch size
  mismatch"); the Tensor matmul refusal stays, with a comment that it is unreachable through the forward.
- bindings.cpp: numpy_dtype throws py::type_error (TypeError) naming the dtype, kind and itemsize.
- Prior art at each site (cuBLAS strided-batched contract, NumPy matmul, batched BLAS unverified; NumPy TypeError and
  ENG2 F1 for N1).
- M7: test_bwd_r1_matmul.py::test_gpu_refuses_forward_first_batch_only now asserts the forward refusal naming both
  shapes and no gradient (same parametrization, still @gpu).

## Build E
- configure_E 12:48:16-12:48:20 UTC, build_E 12:48:20-12:50:54 UTC, exit 0; module sha256
  a875936c39b201e46e571040b13cbdd3779f99e9974fcfe341aff362cf910c82 (built into the worktree package, copied to
  tensor_cuda/build-eng4/E/module/); source map b32574fb... (`build/BUILD_E.json`).

## G1 (CPU, CUDA hidden) on Build E: `g1/G1_E_try1.xml`: 37 passed, 13 skipped (GPU tests)
- Discrimination: the same test file against Build B (`g1/G1_tests_on_B.xml`): 26 failed (every M3, M4 and N1
  refusal test), 11 passed (accepted controls and the listed-dtype comparison).
- N1 Build B reference `n1/B_cpu.json` (scripts/eng4_n1_record.py, seed 4400, Build B package /tmp/eng4_src_B, own
  process).
- E boundary run `boundary/E_cpu.json`: every refusal class refuses with the shapes named; the 8 accepted controls
  give Build B's CUDA error (they pass the predicate and reach the GEMM).

## L55 (`l55/`; artifacts/eng4/tools/l55_probe.cu)
- The probe compiles each build's REAL src/matmul.cu (B: /tmp/eng4_src_B, E: worktree) and links stub cuBLAS entry
  points that record the arguments; NDArrays are built field by field; no CUDA call. Repair 1 of 3 (L56(b)): nvcc
  rejected the stubs (cuBLAS declarations seen as host+device); the probe is compiled by g++ as host C++ and only
  matmul.cu by nvcc.
- E: accept_a plan_batched_b false, stride_B 0, batch 4; accept_b true, stride_B 12 (= K*N), batch 4; rank 2 x 2
  true, stride 12, batch 1; accept_c false, stride 0, batch 2 (and trans_b). B and E pass identical cuBLAS arguments
  and output shapes on all five accepted witnesses (`l55/L55_accepted_compare.json`).
- B on the M3 witnesses: ENG-MM1 A [4,2,3] x B [2,3,4]: stride_B 12, batch 4, B read end 48 of 24 elements (out of
  bounds); all-ones B [1,3,4] under A [2,2,3]: read end 24 of 12 (out of bounds); larger B batch and equal count with
  a different shape: in bounds but wrong pairing; unequal ranks: stride 0, first B matrix only. E refuses all six.

## Census file (M1, L90): `CENSUS.json` (scripts/eng4_census.py, sha256 31835d29...)
- Rows checked against Build B's lines (/tmp/eng4_src_B) with each row's Build E line(s) recorded. Role counts:
  product 23, test direct 31, consumer sites 23 (Voice) + 138 (GRAPA, outside artifacts/), of which TensorCUDA files
  7 + 50; GRAPA artifacts/ copies 472 sites (counted by file); harness n/a. product_refused: 0.
- Consumer notes: GRAPA grapa/codebooks.py:79/85 pass A (B,1,S,D) rank 4 x R (1,D,D) rank 3 = class c (accepted);
  grapa/attention.py repeats KV heads (_repeat_kv) before its matmuls = class b; Voice kokoro_tc/model.py 810/812
  class b, 725/728 class a. GRAPA research scripts (cc10/cc46/cc58/cc69/cc72, orders/seeds) are "dynamic".

## Registrations (before what they govern)
- INTENDED_CHANGES.json 07:55:02 (sha256 01ac22dd...): the three M7 IDs; written before G4's after-run (08:14).
- MUTANTS.json 07:55:48 (sha256 69143137...): m01-m05, hunks against Build E, killing tests and named assertions
  from the order's G5 list; written before any mutant tree was built (first configure 07:55:59).

## Mutants (G5): builds 07:55:59-08:08:50 (one at a time, own scopes), MUTANT_RESULTS.json 08:09
- Modules: m01 6ee509b1..., m02 0e2bc406..., m03 e47d2d49..., m04 c37db25c..., m05 1aa8b722... (BUILD_<id>.json).
- CPU runs (CUDA hidden, one pytest process per killing test, the mutant's own package): every killing test failed
  on its named pytest.raises ("Regex pattern did not match" for m01-m04, "DID NOT RAISE" for m05), and each test's
  captured stderr holds "ENG4-MUTANT <id> executed" (L41). m01 1/1, m02 2/2 (CPU witness), m03 7/7, m04 2/2,
  m05 6/6. m02's GPU line mut_m02 is staged.

## G4 CPU half (CUDA hidden)
- cpu_B 08:08:5x-08:13:58 (Build B files + module installed, then E restored; peak RSS 2,523,020 kB); cpu_E
  08:14:3x-08:18:59 (peak RSS 2,631,228 kB). Both: 4038 IDs, 1388 passed, 2174 failed, 461 skipped, 15 error,
  1 collection error (test_selector_accuracy.py). Per-ID compare (G4_CPU_COMPARE.json): 0 differences -> PASS.
  The M7 IDs are skipped on the CPU lane in both builds (as INTENDED_CHANGES.json predicted). The 2174 CPU failures
  are identical per ID in both builds (GPU tests without a skip marker fail with no device).
- test_terrain_render.py excluded and listed (L40); no worktree writes by tests in either run.

## Frozen packages for the GPU lines (eng4_make_arms.py pkg): build-eng4/{B,E,m02_no_device_check}/pkg, module
  copied from each arm's receipted module (not from the installed one), sha256 matches BUILD_<arm>.json.

## GPU lines staged (GPU_LINES.json, sha256 d7645b97...) and dry-run once each (gpu_dry/<line>.json): 28 lines
  (g2_B, g2_E, g3_B, g3_E, boundary_B, boundary_E, own_E, mut_m02, g4_B_01..10, g4_E_01..10): every dry run exit 78,
  "would_start": false, state busy; module checks ok; no gpu/<line>/ created.
- After the dry runs, two step-logic changes (the forge argv, which is what the dry run covers, is unchanged):
  (1) g2_B's pass condition no longer requires the call to return (memcheck may abort the faulting kernel); it
  rests on memcheck's out-of-bounds report. (2) mut_m02's kill rule reads the junit case: killed = failed on the
  named pytest.raises, or the process died in the test; a skipped or passed test survives.
- CPU plumbing smoke of the step functions (children forced to CUDA_VISIBLE_DEVICES="", outputs in
  /tmp/eng4_tmp/smoke/): g2/g3/boundary/mut_m02 run end to end; g3 fails at tensor creation with no device, as it
  must on the CPU. Budget continuation smoke (series artifacts/eng4/g4/smoke_budget_E, not a gate): part 1 179 IDs,
  part 2 375 IDs, 178 deselected, overlap only the module-skip pseudo-ID.

## G1 final: `g1/G1_E_final.xml` 37 passed, 13 skipped (GPU), Build E module a875936c.
## G6: git status --porcelain --ignored: writes only in the writable set (five edited files, the new test,
  scripts/eng4_*.py, artifacts/eng4/, tensor_cuda/build-eng4/, logs/eng4_*, the package module); no deletions; no
  git writes; no GPU use (every GPU-touching command was a FORGE_GPU_DRY_RUN=1 dry run).
