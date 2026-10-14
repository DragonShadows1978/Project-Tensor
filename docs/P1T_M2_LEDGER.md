# P1T-M2 implementation ledger: `tc_signal` (Conv1d T2, ConvTranspose1d T3)

Order: `orders/P1T_M2_TC_SIGNAL_CONV.md` (fork commit d50b513, immutable). Seat: Claude Opus 5.5 (implementation seat,
Agent-tool subagent). Target: `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_m2`.
Times are local (EDT), from `date`.

## 2026-10-12 19:26 — read-in (evidence class: reading)

- Read the order in full, `/mnt/Shared/HOUSE_RULES.md`, `AGENTS.md`.
- Mechanism docs read: `Fork/tensor_fork_voice/docs/P1T_TENSOR_ALT_PLAN.md` §4 (add-on contract: DLPack capsules for
  every operand incl. outputs, legacy default stream, shared cudart/cuBLAS 12.6, default math mode, no atomics, no
  fast-math); Voice `research/kernels/TENSOR_IMPL_PLAN_P1T.md` §4 (numerics) and §5 T2/T3 (formulas);
  `engine/tensor_requests/p1t_tests/adapter_api.py` (ONNX Conv/ConvTranspose 1-D, host numpy in/out, fp16 storage =
  round weights to fp16, fp32 math); `run_p1t.py` (two calls per case, bitwise determinism, N vs fp64 oracle, gate
  1e-5 for T2/T3 fp32 and fp16-storage; `--require-f1` NOT passed, so the F1 line 5e-6 is reported, not gating);
  `oracles.py` conv1d/conv_transpose1d.
- Mechanism check (HOUSE_RULES §3): what the order exercises (fp32 im2col + cuBLAS SGEMM; gather ConvTranspose; DLPack
  consumption on the legacy stream; Voice's runner on the real fixtures) IS the mechanism the plan §4 defines. No
  scope mismatch.
- Fixture census (manifest `/mnt/ForgeRealm/Project-Voice/artifacts/p1t/fixtures/manifest.json`): T2 72 cases
  (36 nodes x {short_07, max_1}), T3 14 cases. Attribute sets as the order lists, plus one T3 set the order's census
  line omits: `decoder/generator/ConvTranspose(_1)`, K 20 stride 5 pads (0, 0), C_in 11 -> C_out 1, x up to
  [1, 11, 143161]. It is covered by G2 (real data) and added to my own census.
- Environment: driver 595.99.02 (native), RTX 4070 SUPER; `/usr/local/cuda-12` -> `/usr/local/cuda-12.6` (nvcc
  12.6.85); cmake 3.28.3; python 3.12.3; torch 2.11.0+cu130 (torch carries its own CUDA 13 runtime; tc_signal links the
  12.6 runtime like core, both attach to the device's primary context, whose legacy default stream they share).
- Core reference (`/mnt/ForgeRealm/Project-Tensor/tensor_cuda/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so`):
  `ldd` -> `libcudart.so.12 => /usr/local/cuda-12.6/lib64/libcudart.so.12`, RUNPATH `/usr/local/cuda-12.6/lib64`,
  cudart symbols undefined (`U cudaLaunchKernel@libcudart.so.12`): shared cudart.
- Production `tensor_cuda` has no `empty` (only `empty_cache`, `__init__.py:1202`) and no DLPack: the default
  allocator must raise the RuntimeError naming `set_allocator`.

## 2026-10-12 19:40 — pre-registered decisions (before any build or GPU run)

- **Gates:** exactly the order's G1-G5. Own-test gate N <= 1e-5 vs torch fp64 CPU oracle; determinism = two calls
  bitwise equal. Not changed after results.
- **Accumulation-order rail (ConvTranspose gather):** the kernel accumulates in fp32 in a fixed order, taps k
  ascending (outer), input channels c ascending within the group (inner), bias added last. If any Kokoro case (G2,
  real data) shows N > 5e-6 (the runner's F1 line, half the gate) with this order, I switch to a blocked order and
  report both. Otherwise the sequential order stays.
- **Conv1d accumulation:** cuBLAS SGEMM's own order (default math, atomics not allowed), bias added by a separate kernel
  after the GEMM, as the order specifies.
- **No git** (order Rules): G4's `git diff 9df5bef --stat` is not run by me; I give a non-git equivalent (every file
  under the worktree newer than the checkout, with line counts) and flag it as a deviation for the lead to re-run.

## 2026-10-12 19:33-19:41 — package written, first builds (evidence class: build)

- Wrote `tc_signal/` (CMakeLists.txt, build.sh, .gitignore, include/dlpack/dlpack.h vendored byte-identical from
  torch's ATen/dlpack.h v1.3, sha256 62d0acfd...8d73, src/{tuning.h, dlpack_operand.{h,cpp}, conv_ops.h, conv.cu,
  bindings.cpp}, tc_signal/__init__.py).
- Build 1 (19:40, offline pybind11 from core's `_deps/pybind11-src`, v2.12.0): OK, but `ldd` resolved
  `libcudart.so.12 => /usr/local/cuda-12.6/targets/x86_64-linux/lib/libcudart.so.12`, while production core resolves
  `/usr/local/cuda-12.6/lib64/libcudart.so.12`. Cause: production `_tensor_cuda*.so` (Oct 8) was linked by
  `scripts/apa_*_build.sh` with `-Wl,-rpath,/usr/local/cuda-12.6/lib64`; CMake's automatic build RPATH records the
  symlink-resolved directory instead (same file: `lib64 -> targets/x86_64-linux/lib`). Fix: CMakeLists sets
  `SKIP_BUILD_RPATH ON` and `LINKER:-rpath,<realpath(toolkit)>/lib64`. Build 2 (19:41): RUNPATH
  `/usr/local/cuda-12.6/lib64`, ldd now resolves exactly core's path. Stale build dir moved to
  `/tmp/p1t_m2_stale_build_1` (no deletes).
- Smoke (GPU, 19:42): conv1d N 1.0e-7, conv_transpose1d N 8.0e-8 vs torch fp64; cuBLAS handle default math,
  atomics not allowed, stream 0.

## 2026-10-12 19:43 — finding: nvcc 12.6 defines no macro for --use_fast_math (evidence class: build probe)

- A `#if defined(__USE_FAST_MATH__) #error` guard did NOT fire under `nvcc -use_fast_math` (scratch
  `/tmp/p1t_m2_scratch/fm.cu`, exit 0). `nvcc --dryrun -use_fast_math` shows only cicc flags `-ftz=1 -prec_div=0
  -prec_sqrt=0 -fmad=1 -fast-math`, no -D macro. The inert guard was removed (it would have been false assurance).
- Replacement: (1) CMakeLists refuses fast-math/default-stream flags at configure time; (2) a runtime canary kernel
  `fp32_divide_probe` compiled in conv.cu with the package flags; `test_fast_math_canary_ieee_division_and_subnormals`
  checks 4096 fp32 quotients bitwise against host IEEE division and that subnormal quotients are not flushed.
  Prior art: none specific known to me (denormal flushing as a fast-math symptom: CUDA C Programming Guide).

## 2026-10-12 19:45-19:48 — own tests, development run (evidence class: unit test, GPU)

- Added `__dlpack_device__` pre-check in `to_capsule` (array API order: a CPU producer such as NumPy rejects
  stream=1 before tc_signal could raise its documented ValueError).
- Test bug fixed (not a gate change): refusal fragments are now `re.escape`d (`w.shape[1]` was read as a regex class).
- Dev run: `157 passed in 11.10s`; 61 census cells, max N 3.845e-07 (convT k20_s10_p55_512to256), all deterministic.

## 2026-10-12 19:49-19:50 — G2 development run (evidence class: end-to-end gate on real fixtures)

- Adapter `artifacts/p1t_m2/tc_signal_torch_adapter.py` copied from Voice `tensorcuda_stub.py` (sha256 d0c5a596...fc0b)
  and changed per the order (T2/T3 through tc_signal on torch CUDA tensors; fp16 storage = `.to(float16).float()`;
  everything else NotImplementedError; no `import tensor_cuda`).
- Runner `--items T2,T3` with `tolerances_r2.json`: exit 0, verdict PASS, 172/172 (T2 144 = 72 x {wfp32, wfp16};
  T3 28), max N 1.394e-06 (T2 resblocks.2 convs1.2 short_07), T3 max N 1.228e-06 (ups.0 max_1 wfp32), all F1 lines
  (5e-6) held, all deterministic. 45.9 s.
- Accumulation-order rail (registered 19:40): T3 max N 1.23e-6 <= 5e-6, so the rail did not fire; the sequential order
  (k ascending outer, c ascending inner, bias last) stays.

## 2026-10-12 19:52-19:58 — G3 mutants (evidence class: mutation test; G1 + G2 on each mutant)

- Builder `/tmp/p1t_m2_mutants/make_mutants.py` (copy: `artifacts/p1t_m2/mutants/make_mutants.py`) copies the
  sources to `/tmp/p1t_m2_mut_<id>/tc_signal` and applies one registered defect, each replacement asserted to match
  exactly once; diffs in `artifacts/p1t_m2/mutants/<id>/mutant_<id>.diff`. Built and run outside the worktree
  (adapter copied into each mutant root). After the final build, `diff -r` confirmed every mutant tree = final source
  + its mutation only (README.md absent from the copies, written later).
- (a) im2col ignores dilation (`conv.cu:128`): G1 FAIL 10/157 (the 10 dilated census cells, N 0.65-1.24);
  G2 FAIL exit 1, 48/172 (the 12 dilated generator nodes x 2 cases x 2 storages), N 0.289-1.26. KILLED.
- (b) pad_begin/pad_end swapped (both plans, `bindings.cpp`): G1 FAIL 13/157 = exactly the 13 asymmetric-pad cells
  (7 conv1d + 6 convT), N 0.47-1.40, no symmetric cell failed; G2 PASS exit 0, 172/172 (every Kokoro pad is
  symmetric). KILLED by the asymmetric own cells only, as the order requires.
- (c) TF32 on the cuBLAS handle (`conv.cu:107`): G1 FAIL 25/157 (20 conv1d census cells N 1.8e-4-3.2e-4,
  4 byte_offset tests, the handle-config probe); G2 FAIL exit 1, 126/172 (T2 126/144 fail, min failing N 2.27e-5,
  max N 5.65e-4). Survivors: 18 T2 cases with C_out <= 22 or C_in = 1 (F0_proj, F0_conv, generator stft conv,
  noise_convs.1, conv_post; N <= 1.2e-6; cuBLAS does not pick tensor-op kernels for those shapes) and all T3 (no GEMM).
  KILLED.
- (d) conv_transpose1d as an atomicAdd scatter: N gates all pass (G1 convT max N 2.3e-6; G2 max N 1.5e-6), but the
  determinism checks fail: G1 FAIL 14/157 (14/21 convT cells, 5-55,337 mismatching elements; the 7 survivors are the
  cells with <= 2 contributions per output: depthwise x2, T<K, stride>K x2, dilation span, output length 1); G2 FAIL
  exit 1, 28/28 T3 cases non-deterministic, 3 to 1,696,791 mismatching elements per case (9,117,600 in total),
  T2 144/144 PASS. KILLED by determinism.
- (e) EXTRA, not in the order's list: `--use_fast_math` forced on the CUDA sources past the configure guard
  (`target_compile_options`). G1 FAIL 1/157: only `test_fast_math_canary_ieee_division_and_subnormals` (subnormal
  quotients flushed to 0); G2 PASS exit 0 with identical N. Shows the canary has teeth and that fast-math is otherwise
  invisible to the conv gates (consistent with Voice's tolerances_r2 blind-spot note on fast-exp).

## 2026-10-12 19:55-20:06 — final build and final gate runs (evidence class: build / unit test / end-to-end gate)

- cuBLAS memory probe (`artifacts/p1t_m2/cublas_workspace_probe.py`): handle creation 10.0 MiB (cudaMemGetInfo
  delta, includes the CUDA-12 runtime's first use), first and second SGEMM 0 MiB.
- Final clean build 20:00 (dev build dir moved to `/tmp/p1t_m2_stale_build_2`): `logs/p1t_m2_build.log`, exit 0,
  0 warnings; `.so` sha256 3db14dc4af85e5376b850d21ec568f842189c1a4d76f47f87fcf80c6d23e4a25.
- G4 linkage (`artifacts/p1t_m2/g4_ldd.txt`): `libcudart.so.12 => /usr/local/cuda-12.6/lib64/libcudart.so.12` for
  both tc_signal and production core (SAME PATH: yes); no `_tensor_cuda`, no cudart_static on the link line, 0 cuda*
  symbols defined, 13 imported from libcudart.so.12. (A plain `nm` count of 0 in the first receipt block is vacuous
  because the module is stripped; the receipt says so and adds the real checks.)
- G1 final (20:01): `157 passed in 5.23s` (9.7 s wall incl. lock and imports); census receipt
  `artifacts/p1t_m2/g1_census_receipt.json`: 61 cells, all pass, all deterministic, max N 3.845e-07.
- G2 final (20:01): exit 0, verdict PASS, 172/172, max N 1.394e-06, identical to the dev run.
- G4 one process (20:02-20:06, most of it waiting on the shared GPU lock): production tensor_cuda + tc_signal + torch in
  one process; one libcudart.so.12 file mapped (`libcudart.so.12.6.77`; torch maps its own libcudart.so.13); core op
  OK; default allocator raises the RuntimeError naming set_allocator; conv1d N 2.74e-7, convT N 4.48e-7. PASS.
  No file under production `tensor_cuda/tensor_cuda` changed (find -newer: none).
- G5 (20:06): table in the report; host overhead ~150-200 us per call, dominated by torch's Python-level
  `__dlpack__` (18-22 us per operand, 5 operands per conv1d call).

## 2026-10-12 20:08-20:13 — extra mutant (e), report, non-git diff stat (evidence class: mutation test / reasoning)

- Mutant (e) (extra, not in the order): `--use_fast_math` via `target_compile_options` in a /tmp copy. G1: 1 failed
  (only the canary: subnormal quotients flushed to 0); G2 PASS with identical N. Receipts `artifacts/p1t_m2/mutants/e/`.
- Report `docs/P1T_M2_REPORT.md` written (gate table, API, build, kernel design with file:line citations, accumulation
  order, mutant table, overhead table, G4 details, deviations, prior art).
- Non-git diff stat `artifacts/p1t_m2/diffstat_nongit.txt`: 78 new files, 0 outside the allowed paths; root .gitignore
  hides *.log/*.json (receipts need `git add -f`) and *.so (build output, not for commit).
- No write outside the target and /tmp: `find` of production Project-Tensor (Fork/ pruned), Project-Voice
  engine/ and artifacts/p1t newer than 19:26 printed nothing.
- Scratch left in /tmp (no deletes): /tmp/p1t_m2_scratch, /tmp/p1t_m2_mutants, /tmp/p1t_m2_mut_{a,b,c,d,e},
  /tmp/p1t_m2_stale_build_{1,2}.

## Prior art (Prior Art Directive; also at the code sites and in the report)

- im2col + GEMM: Chellapilla, Puri & Simard (2006). Taken: unfold + one SGEMM. Ours: 1-D unfold with dilation and
  asymmetric pads, per-batch scratch reuse, groups as strided-batched SGEMM. (`src/conv.cu` header and :115-118.)
- Gather-form transposed convolution: adjoint of strided conv (Dumoulin & Visin 2016). Taken: tap condition. Ours:
  fixed loop order, thread mapping, fmaf. (`src/conv.cu` header and :142-146.)
- ONNX Conv/ConvTranspose specs: semantics.
- DLPack v1.3 (dmlc/dlpack) and the Python array API data-interchange protocol: capsule ABI, stream=1, max_version
  fallback, `__dlpack_device__` first. (`src/dlpack_operand.h` header.)
- CuPy `cupy.cuda.set_allocator`: pluggable allocator (ours returns DLPack tensors). (`tc_signal/__init__.py` header.)
- cuDNN/cuBLASLt workspace-query pattern: plan/run split. (`src/bindings.cpp` header.)
- Mutation testing: DeMillo, Lipton & Sayward (1978). Normwise error: Higham (2002).
- Fast-math canary: no specific prior art known to me.
- Unverified in the sandbox, lead to check: search "Chellapilla Puri Simard 2006", "Dumoulin Visin convolution
  arithmetic 2016", "DLPack 1.3", "array API __dlpack__ max_version copy", "cupy set_allocator".
