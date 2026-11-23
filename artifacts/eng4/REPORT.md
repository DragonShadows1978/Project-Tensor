<!-- Filed by the lead from the seat's final message (the harness refused the seat's own REPORT.md write). Content
unchanged; the lead merged a few table rows and bullets for length. The full message is in the lead session transcript.
The seat's early stub is kept as REPORT_STUB_0751.md. Lead verification and GPU results: see LEAD_VERIFY.md when written. -->

# ENG4 report — the matmul forward refuses every operand shape it cannot compute

Seat: Claude Opus 5.5 (Agent tool). Order `orders/ENG4_MATMUL_CONTRACT.md` at 345a77242 (sha256 d4f42fd3…).
Worktree `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng4`. Receipts: `artifacts/eng4/LEDGER.md`.

## Verdict

**The seat's part passes:** G1 PASS, G5 PASS (CPU-killable mutants), G6 PASS; G2, G3, G4's GPU half and m02 are STAGED with every staged line's dry run exiting 78. The round is ACCEPTED only after the lead's GPU lines pass.

- **G1** (CPU refusals and N1): **PASS** — Build E, CUDA hidden: 37 passed, 13 GPU tests skipped (`g1/G1_E_final.xml`; also `G1_E_try1.xml`). The same file against Build B: the 26 refusal tests fail, the 11 controls pass (`g1/G1_tests_on_B.xml`).
- **G2** (ENG-MM1 under memcheck): **STAGED** (g2_B, g2_E; dry run 78).
- **G3** (accepted results bitwise): **STAGED** (g3_B, g3_E; 123 seeded cells; the compare is a CPU step after).
- **G4** (no regression): CPU half **PASS** (4038 IDs per build, 0 differences, `G4_CPU_COMPARE.json`); GPU half **STAGED** (g4_B_01..10, g4_E_01..10).
- **G5** (mutants): **PASS** on CPU — all 5 killed on their named assertions, executed-marker in every killing test; m02 GPU line **STAGED**.
- **G6** (scope): **PASS** — writes only in the writable set; nothing deleted; no git writes; no GPU use (every GPU command was a `FORGE_GPU_DRY_RUN=1` dry run).

## Census summary (`CENSUS.json`; M1, N1, L90)

Role counts: product 23 sites; test 31 direct sites; consumer Project-Voice/engine 23 (7 in tensor_cuda-importing files); consumer GRAPA-Native-LLM 138 (50 TensorCUDA) plus 472 in GRAPA `artifacts/` copies (per file); harness n/a.

Product paths (Build B line numbers):
- class a: nn.Linear (`nn.py:107`), Conv2D (`:386`), ConvTranspose2D (`:448`), int4/intn linear (`kernels.cu:4336`, `:4818`) and their backwards (`ops.cpp:864`, `:903`);
- class b: `quant.py:125/127` (Rb expanded to [B,H,D,D]), `apa_packed.py:389`, `gemm_apa.cu:779` (`validate()` fixes v's [B,KVH]);
- dynamic, documented (B,H,L,D), class b when callers follow the docs: SDPA (`functional.py:171/177/200`), apa_quant_attention and `_definition_selection` (`quant.py:248/291/292/331`);
- dynamic entries: `tc.matmul` (`__init__.py:343`), the binding (`bindings.cpp:2075`), Tensor `@` (`:2050`), `ops::matmul` (`ops.cpp:826`), the backward's lower-precision path (`ops.cpp:785`, class a or b by construction).

**No product path fixes a refused pair → no M1 RED.**

Consumers (report only): GRAPA `grapa/codebooks.py:79/85` A(B,1,S,D) × R(1,D,D) = class c (accepted); GRAPA `attention.py` repeats KV heads before matmul (class b); Voice `kokoro_tc/model.py:725/728` class a, `:810/812` class b.

Tests with batch dims: `test_bwd_r1_matmul.py:164` (M7, now refused), `:179` (class c); `test_bwd_r1_core.py:24` oracle cases (batch_shared = a; batched and the strided canary = b).

N1: numpy_dtype's only caller is tensor_from_numpy (`bindings.cpp:88`), whose only binding is `_C.tensor` (`:2071`). tc.tensor, the factories, embedding, cross_entropy and the bf16 path cast to a listed dtype first; **tc.from_numpy does not cast**. No consumer-root site calls `tc.from_numpy` or `_C.tensor`.

## Accepted and refused classes, each with its witness

**Accepted** (forward's cuBLAS arguments identical to Build B's — L55 probe, real `matmul.cu` with stub cuBLAS, `l55/L55_accepted_compare.json`):

| Class | Witness (A × B) | plan batched_b | B stride | batch |
|---|---|---|---|---|
| a | [4,2,3] × [3,4] | false | 0 | 4 |
| b | [4,2,3] × [4,3,4] | true | 12 (K·N) | 4 |
| b (rank-2 × rank-2) | [2,3] × [3,4] | true | 12 (as before) | 1 |
| c | [2,2,3] × [1,1,3,4]; trans_b with [1,1,4,3] | false | 0 | 2 |

INT_MAX boundary (M/N/K/batch = 2^31−1, zero-element operands) is not refused. Accepted pairs on the CPU reach the GEMM and fail with the no-device CUDA error, exactly as Build B.

**Refused before any CUDA call** (G1, `BOUNDARY.json`; message `matmul refused: <issue> (A [..] on <dev>, B [..] on <dev>, trans_b=..)`):
- M3: equal ranks smaller B batch [4,2,3]×[2,3,4] (ENG-MM1) + trans_b; equal ranks larger B batch [2,2,3]×[4,3,4] and [1,2,3]×[2,3,4]; equal count, different shape [2,3,2,3]×[3,2,3,4]; equal ranks all-ones B under larger A [2,2,3]×[1,3,4] and [2,3,2,3]×[1,1,3,4]; unequal ranks non-all-ones B batch [2,3]×[2,3,4], [2,1,2,3]×[2,3,4], [2,2,3]×[1,2,3,4] + trans_b.
- M4 device: host [2,0] × CUDA zero-element [0,3] and the reverse (the GPU line uses nonzero [3,5]×[5,4] both orders).
- M4 size (zero-element operands): M = 2^31; N = 2^31 (and trans_b); K = 2^31 (and trans_b); batch 2^31 with rank-2 B; batch 2^16·2^15 at equal ranks.

**Build B on the same classes:** with CUDA hidden, B refuses none — every case reaches the cuBLAS call ("CUDA error at matmul: no CUDA-capable device is detected"); a CUDA A with a host B fails earlier at cudaMalloc. The L55 probe shows on CPU what B computes: ENG-MM1 uses B stride 12 over batch 4 → reads B up to element 48 of 24 (OOB); an all-ones B under a larger A reads up to 24 of 12 (OOB).

**N1:** `tc._C.tensor` and `tc.from_numpy` raise TypeError for float64, int32, uint16 ("tensor: unsupported NumPy dtype …; supported: float32, float16, int64, uint8, bool"); every listed dtype keeps Build B's bytes (`n1/B_cpu.json`, seed 4400). Build B on uint16 read 24 bytes from a 12-byte buffer (a host OOB read), which N1 now refuses.

## Diff summary (`artifacts/eng4/source.diff`, sha256 64b62812…; `new_test_file.diff`)

- `core.h` (+17): declarations only — `MatmulOperandIssue`, `MatmulOperandPlan`, `matmul_operand_plan()`.
- `matmul.cu` (+130/−14): the predicate (rank, dtype, inner dim, device, M/N/K > INT_MAX, batch count > INT_MAX, batch classes); `tc::matmul` calls it first (old messages kept as prefixes); GEMM block and stride rule unchanged ((ndb == nda) → K·N, else 0).
- `ops.cpp` (+22/−15): `matmul_backward` uses the shared predicate (batch SHAPE check replaces the count check; BWD message texts kept); the `Tensor matmul` refusal stays with a comment that it is now unreachable through the forward.
- `bindings.cpp` (+7/−1): numpy_dtype throws `py::type_error` naming the dtype.
- `test_bwd_r1_matmul.py`: the M7 change only. New `tensor_cuda/tests/test_eng4_matmul.py`.
- No knob; no new broadcasting.

## Every module sha256

| Build | Module sha256 |
|---|---|
| B | 20881ace0ef7f3fd1feb324f1f75a63589d523d0f943857de7564c5f0e297c40 (source map 69e283c0…, 47 files) |
| E | a875936c39b201e46e571040b13cbdd3779f99e9974fcfe341aff362cf910c82 (source map b32574fb…) |
| m01_count_not_shape | 6ee509b1fc8fe309d5289f51ce9991932dbcd7f8572488bf8fd901c26f62d14b |
| m02_no_device_check | 0e2bc4064eb410b72ff31f472974b2f641b8cde95e93ce2e4b89ea96f9feb184 |
| m03_no_intmax | e47d2d49cad743055b28352d56944fb271b256a3cc36a08d8b727f2af558ce87 |
| m04_ones_any_rank | c37db25c68d29d5e1a8899dc3905e1749da8bfcf714e8c0604852f07b0ada940 |
| m05_numpy_dtype_fallback | 1aa8b722340c74324460b9c305f542058162e80bac98c5e2e4a410404cc855a8 |

Frozen packages `build-eng4/{B,E,m02_no_device_check}/pkg` carry the same sha256s.

## G4's CPU per-ID result and intended changes

cpu_B vs cpu_E: 4038 IDs each; identical outcomes (1388 passed, 2174 failed, 461 skipped, 15 error; 1 collection error `test_selector_accuracy.py`) → **0 differences, PASS**. The 2174 CPU failures are the same IDs in both (GPU tests without a skip marker). `test_bwd_r1_native.py::test_host_native_backward_checks[batch-mismatch]` passes in both (in E via the shape check). `test_terrain_render.py` excluded and listed (L40). No test wrote into the worktree.

Intended changes (`INTENDED_CHANGES.json`, written 07:55, before the 08:14 after-run; ruling M7): `test_bwd_r1_matmul.py::test_gpu_refuses_forward_first_batch_only[a_batch0-b_batch0 | a_batch1-b_batch1 | a_batch2-b_batch2]` — old assertion: forward returns, backward raises "forward uses stride 0 and only the first B matrix", no gradients; new: `tc.matmul` raises RuntimeError matching both shapes, a.grad and b.grad None. Expected outcome: skipped on the CPU lane in both builds (observed), passed on GPU in both. No other existing test changed.

## Mutant table (`MUTANTS.json` written 07:55:48; first mutant build started 07:55:59)

| Mutant | Killing CPU tests | Result |
|---|---|---|
| m01_count_not_shape | `test_g1_m3_refuses[eq_count_diff_shape]` | KILLED 1/1 (regex did not match; marker seen) |
| m02_no_device_check | `test_g1_m4_device_refuses[host_a_cuda_b, cuda_a_host_b]` | KILLED 2/2; GPU line mut_m02 STAGED |
| m03_no_intmax | `test_g1_m4_size_refuses[7 cases]` | KILLED 7/7 |
| m04_ones_any_rank | `test_g1_m3_refuses[eq_rank_ones_b_under_larger_a, …_rank4]` | KILLED 2/2 |
| m05_numpy_dtype_fallback | `test_g1_n1_refuses_unlisted[6 cases]` | KILLED 6/6 (DID NOT RAISE) |

## Each staged GPU line with its dry-run exit

Lead runs each from the worktree root: `python3 -B -P scripts/eng4_gpu_line.py run <line>` (inside: `forge-gpu-run --min <s> --max 540 --label eng4-<line> -- timeout <s>s python3 -B -P scripts/eng4_gpu_line.py step <line>`).

| Line | --min | inner timeout | expected peak host bytes | dry-run exit |
|---|---|---|---|---|
| g2_B / g2_E | 240 | 480s | 3 GiB | 78 / 78 |
| g3_B / g3_E | 120 | 480s | 3 GiB | 78 / 78 |
| boundary_B / boundary_E | 180 | 480s | 3 GiB | 78 / 78 |
| own_E | 120 | 480s | 3 GiB | 78 |
| mut_m02 | 120 | 480s | 3 GiB | 78 |
| g4_B_01 … g4_B_10 | 480 | 500s | 6 GiB (estimate: CPU suite peaked 2.5–2.6 GB + CUDA context) | 78 each |
| g4_E_01 … g4_E_10 | 480 | 500s | 6 GiB (same) | 78 each |

Each line refuses before the lease if `gpu/<line>/` exists or a module differs from its receipt (L103); retries forge 75/76 (nothing ran) after 300 s, at most 6; records forge exit, inner exit, timed_out (124) and the log sha256 (L45). The g4 lines continue one series per build under a 330 s test budget and exit 0 at once when the series is complete. Write-once paths are listed per line (L119).

CPU follow-ups after the GPU lines:
- `python3 -B -P scripts/eng4_g3.py compare artifacts/eng4/gpu/g3_B/g3 artifacts/eng4/gpu/g3_E/g3 artifacts/eng4/G3_COMPARE.json`
- `python3 -B -P scripts/eng4_g4_compare.py gpu_B gpu_E artifacts/eng4/G4_GPU_COMPARE.json`
- `python3 -B -P scripts/eng4_mutant_results.py gpu`

## Deviations

1. Census timing: every product call site was read and classified before any edit (07:41–07:45); `CENSUS.json` was written after (08:00), its rows checked against Build B's line numbers.
2. G1 N1 entry name: the order names `tc._C.tensor_from_numpy`, which does not exist; the only binding of the C++ function is `_C.tensor` (`bindings.cpp:2071`), so the tests use `tc._C.tensor` and `tc.from_numpy`. Treated as a naming slip in the order; lead to confirm.
3. Read-only reads beyond L64: ENG3's helper scripts and `build_receipts.json` (eng3 worktree), a WorkTree listing, the `$HOME/.local/bin/compute-sanitizer` wrapper, CUDA 12.6 `cublas_api.h` (stub signatures), `forge-gpu-run --help`.
4. L56(b) repair 1 of 3: nvcc rejected the L55 probe's stubs; the probe is compiled with g++ and only `matmul.cu` with nvcc.
5. L56(a) slip: a bad array view in the inline G3-compare self-test; fixed and re-run.
6. Step logic changed after the dry runs (forge argv unchanged): g2_B passes on memcheck's OOB report alone (no need for the call to return); mut_m02 counts a kill only on a named-assertion failure or process death, read from junit.
7. N1 refuses with TypeError (the order names no exception type).
8. Additions beyond the minimum: boundary_B/E and own_E lines; a pooled-allocator memcheck sub-run in g2 (recorded, not gated); accepted-control tests.
9. CPU no-op through compute-sanitizer used /bin/true (rc 255, nothing instrumented); the CUDA no-op runs inside each g2 line.
10. matmul_backward on unequal ranks with a non-singleton B batch now refuses through the shared predicate (M5), where it accepted with stride 0 before; this path is unreachable through the forward.

## Residuals and gaps (not claimed fixed)

- No current-device check in the forward (gap, as the order says).
- Two host operands still reach cuBLAS (HOST-COMPUTE-HMM, out of scope).
- The forward's unsupported-dtype refusal still comes after the output allocation, which is a CUDA call (pre-existing; outside M3/M4; unchanged).
- numpy_dtype still accepts a non-native byte order of a listed kind and size (e.g. '>f4') as a listed dtype (pre-existing).
- K = 0: whether cuBLAS writes zeros into the output is measured by G3's zero-size cells, not assumed.
- SDPA / apa_quant_attention callers that pass k/v with fewer heads than q now get a refusal instead of a silently wrong result; no in-repo or consumer caller does this.

## Prior art

- cuBLAS strided-batched GEMM contract (cuBLAS docs, `cublas<t>gemmStridedBatched` / `cublasGemmStridedBatchedEx`) — why ENG-MM1 reads out of bounds; stride rule kept unchanged.
- Batched BLAS (Dongarra et al., 2017) — unverified, lead to check ("batched BLAS standard strided").
- NumPy matmul broadcasting (numpy.matmul docs) — which pairs are meaningful; reference shapes for zero-size class a/b; its broadcasting not adopted.
- Mutation testing (DeMillo, Lipton and Sayward, 1978).
- compute-sanitizer memcheck (NVIDIA CUDA toolkit docs).
- Link-time stubs for the L55 probe (Meszaros, xUnit Test Patterns, 2007) — unverified, lead to check.
- N1: NumPy's TypeError for an unknown dtype; ENG2 F1's `_np_dtype` (this repo).
- Harness: pytest hooks and JUnit XML; ENG3's helper scripts (this repo).
- Ours: the a/b/c class split matching the backward, the refusal set, the shared predicate's placement, the budgeted resumable suite chunking, the L55 probe. No other prior art known to me.

## Every file written

Under the worktree: engine and tests — `tensor_cuda/include/tc/core.h`, `tensor_cuda/src/matmul.cu`, `tensor_cuda/src/ops.cpp`, `tensor_cuda/src/bindings.cpp`, `tensor_cuda/tests/test_bwd_r1_matmul.py`, `tensor_cuda/tests/test_eng4_matmul.py` (new), `tensor_cuda/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so` (Build E module; swapped to B and back during the cpu_B run and the budget smoke, verified E after). Scripts (all `python3 -B -P`; none shares a stdlib name): `scripts/eng4_build.py`, `eng4_boundary.py`, `eng4_boundary_gpu.py`, `eng4_census.py`, `eng4_g2.py`, `eng4_g3.py`, `eng4_g4_compare.py`, `eng4_gpu_line.py`, `eng4_make_arms.py`, `eng4_mutant_results.py`, `eng4_n1_record.py`, `eng4_suite.py`. `artifacts/eng4/`: `LEDGER.md`, `REPORT.md` (early stub), `preflight_L01.txt`, `CENSUS.json`, `INTENDED_CHANGES.json`, `MUTANTS.json`, `MUTANT_RESULTS.json`, `BOUNDARY.json`, `GPU_LINES.json`, `G4_CPU_COMPARE.json`, `source.diff`, `new_test_file.diff`, `before/`, `E_src/`, `build/` (receipts + `BUILD_<arm>.json`, 7 arms), `boundary/`, `n1/`, `g1/`, `l55/`, `tools/l55_probe.cu`, `mutants/` (5 diffs + `runs/`), `g4/cpu_B`, `g4/cpu_E`, `g4/smoke_budget_E` (a smoke, not a gate), `gpu_dry/` (28 dry-run receipts). Builds: `tensor_cuda/build-eng4/{B,E,m01..m05}/` (each `module/`; `pkg/` + `pkg.json` for B, E, m02). Logs: `logs/eng4_build_B.log`, `eng4_build_E.log`, `eng4_build_mutants.log`, `eng4_g4_cpu_B.log`, `eng4_g4_cpu_E.log`. Scratch left in place (not deleted): `/tmp/eng4_tmp` (pytest basetemps, smoke outputs), `/tmp/eng4_src_B`, `/tmp/eng4_src_m01_count_not_shape` … `/tmp/eng4_src_m05_numpy_dtype_fallback`, `/tmp/eng4_l55`.
