# P1T-M2 report: the `tc_signal` add-on, with Conv1d (T2) and ConvTranspose1d (T3)

**Verdict: G1 PASS, G2 PASS, G3 PASS (every registered mutant is killed), G4 PASS, G5 reported.**
- **G4 deviation:** I did not run `git diff 9df5bef --stat`, because the order's Rules say NO git. A non-git
  equivalent is given instead, and the lead should re-run the git form.
- **Order and receipts:** order `orders/P1T_M2_TC_SIGNAL_CONV.md` (fork d50b513, immutable); ledger
  `docs/P1T_M2_LEDGER.md`.
- **Seat:** Claude Opus 5.5 implementation seat, 2026-10-12, in worktree
  `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_m2` (branch voice-m2, base
  9df5bef).
- **Built module:** `tc_signal/tc_signal/_tc_signal.cpython-312-x86_64-linux-gnu.so`, sha256
  `3db14dc4af85e5376b850d21ec568f842189c1a4d76f47f87fcf80c6d23e4a25`.

## Gate table

| Gate | Result | Receipt |
|---|---|---|
| G1: own tests (normwise N <= 1e-5 vs fp64 oracle on every census cell, two calls bitwise equal, every refusal) | **PASS**: `157 passed in 5.23s` (9.7 s wall). 61 census cells (26 Kokoro attribute sets, 35 edge cells), max N 3.845e-07, all deterministic, all elements written (NaN-prefilled allocator). 41 parametrized refusals, 13 further refusal tests, 20 consumer-rule / allocator / stream-order / config tests, and 22 oracle-construction checks (61 + 41 + 13 + 20 + 22 = 157) | `logs/p1t_m2_g1_own_tests.log`, `artifacts/p1t_m2/g1_census_receipt.json` |
| G2: Voice runner, `--items T2,T3`, `tolerances_r2.json` | **PASS**: exit 0, verdict PASS, 172/172 (T2 144, T3 28; fp32 and fp16 weight storage), max N 1.394e-06, min margin 7.17x, all deterministic, all F1 lines held | `artifacts/p1t_m2/p1t_T2T3_torchadapter.json`, `logs/p1t_m2_g2_runner.log` |
| G3: mutants (a)-(d) on G1 and G2 | **PASS**: (a), (b) and (c) fail at least one gate; (b) is caught only by the asymmetric own cells (13/13); (d) fails the determinism checks in G1 (14 cells) and G2 (28/28 T3 cases) | `artifacts/p1t_m2/mutants/` |
| G4: decoupling | **PASS**: ldd gives the same `libcudart.so.12` path as core, no `_tensor_cuda`, no static cudart; the grep is empty; one process with production `tensor_cuda` and `tc_signal` works and its convs pass. Deviation: non-git diff stat | `artifacts/p1t_m2/g4_ldd.txt`, `logs/p1t_m2_g4_one_process.log`, `artifacts/p1t_m2/diffstat_nongit.txt` |
| G5: overhead (informational) | reported: host wall per call 150 us ([1,1,8]) and 203 us ([1,512,1193]); GPU 163 us and 317 us | `logs/p1t_m2_g5_overhead.log` |

**G2 verdict line:**
- From `logs/p1t_m2_g2_runner.log`: `"verdict": "PASS",` `"exit_code": 0,`, then `runner exit=0`.
- Report: `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_m2/artifacts/p1t_m2/p1t_T2T3_torchadapter.json`
  (sha256 `9c92249472b2d764b456cd53ce00863378948179d8f1d733498e478c8e10467d`).
- Adapter sha256: `d9d58f8cfbed1e3df4aec0f47e57078f6e413395225f54ae2cf58b7bf4f7bd6c`.
- Fixture manifest sha256: `89499fa537d109a1b3a616f88dfec1423fcb586ac26f0f94687fb4b8cea8639f`.

| G2 slice | Cases | PASS | Max N (case) |
|---|---|---|---|
| T2 wfp32 | 72 | 72 | 1.394e-06 (generator resblocks.2 convs1.2, short_07) |
| T2 wfp16 | 72 | 72 | 1.340e-06 (same node) |
| T3 wfp32 | 14 | 14 | 1.228e-06 (generator ups.0, max_1) |
| T3 wfp16 | 14 | 14 | 1.265e-06 (generator ups.1, max_1) |

## API (exact, as ordered)

```python
tc_signal.conv1d(x, w, b=None, *, stride=1, padding=0, dilation=1, groups=1, out=None)
tc_signal.conv_transpose1d(x, w, b=None, *, stride=1, padding=0, output_padding=0, dilation=1, groups=1, out=None)
tc_signal.set_allocator(fn)   # fn(shape: tuple) -> fp32 CUDA DLPack tensor; returns the previous allocator
```

The semantics are ONNX Conv and ConvTranspose, 1-D.

- **Shapes**
  - `x` is `[N, C_in, T]`; `b` is `[C_out]` or None; `padding` is an int or `(pad_begin, pad_end)`.
  - conv1d: `w` is `[C_out, C_in/groups, K]`, and the output length is `(T+pb+pe-dilation*(K-1)-1)//stride + 1`.
  - conv_transpose1d: `w` is `[C_in, C_out/groups, K]`, and the output length is
    `stride*(T-1) + output_padding + dilation*(K-1) + 1 - pb - pe`.
- **Refusals**
  - An output length <= 0, or `output_padding >= max(stride, dilation)`, raises ValueError.
  - All the error classes are listed in `tc_signal/README.md` ("Errors").
- **Allocation:** `out=None` allocates through the module allocator, and the function returns `out`.
  - The default allocator is `tensor_cuda.empty(*shape, dtype="float32")`.
  - That function does not exist on this branch, nor in production main. There the default allocator raises a
    RuntimeError that names `set_allocator` (receipt: `logs/p1t_m2_g4_one_process.log`, "default_allocator").
  - `set_allocator(None)` restores the default.
- **Code:** the Python layer is `tc_signal/tc_signal/__init__.py:86` (conv1d), `:102` (conv_transpose1d) and `:57`
  (set_allocator).

## Build steps

```bash
cd tc_signal
PYBIND11_SOURCE_DIR=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src ./build.sh 89
# -> tc_signal/tc_signal/_tc_signal.cpython-312-x86_64-linux-gnu.so ; use with PYTHONPATH=<repo>/tc_signal
```

- `build.sh` mirrors `tensor_cuda/build.sh`:
  - newest `/usr/local/cuda-*`: `/usr/local/cuda-12` resolves to `/usr/local/cuda-12.6` (nvcc 12.6.85);
  - arch 89 by default;
  - Release build.
- `PYBIND11_SOURCE_DIR` is passed as `-DFETCHCONTENT_SOURCE_DIR_PYBIND11`, giving pybind11 v2.12.0 offline. Core's
  tree is only read.
- `CMakeLists.txt` settings:
  - `CMAKE_CUDA_RUNTIME_LIBRARY Shared` (`:23`);
  - IPO/LTO off, as core does (`:18`);
  - links `CUDA::cudart CUDA::cublas` only (`:56`);
  - refuses fast-math and default-stream flags at configure time (`:31`).
- RUNPATH: the CMakeLists pins it to `<realpath(toolkit)>/lib64` = `/usr/local/cuda-12.6/lib64` (`:68`, `:71`).
  - Production core's module was linked by `scripts/apa_*_build.sh` with that rpath.
  - CMake's automatic RPATH would instead record the symlink-resolved `targets/x86_64-linux/lib`: the same file, but a
    different path string, which would fail G4's "same path" check (ledger 19:40).
- Clean build log: `logs/p1t_m2_build.log` (exit 0, 0 warnings).

## Kernel design (with code citations)

All work goes on the legacy default stream (`src/conv.cu:49`, `kLegacyStream = 0`; per-thread default stream refused
at `conv.cu:36`), and `cudaGetLastError` is checked after every launch (`conv.cu:221, 235, 241, 253`). No atomics are
used anywhere, and there is no fast-math.

- **Operands (DLPack consumer)**, in `src/dlpack_operand.cpp`:
  - `to_capsule` (`:62`) asks `__dlpack_device__()` first and refuses non-CUDA there (`:69`).
  - It then calls `__dlpack__(stream=1, max_version=(1, 3)[, copy=False for written operands])`, and falls back to
    `__dlpack__(stream=1)` on TypeError (`:97`).
  - A capsule is consumed by renaming it `used_*` (`:197`, `:207`). `ManagedOwner::reset` (`:174`) calls each
    producer deleter exactly once, after the launches or during unwinding.
  - `validate_tensor` (`:116`) checks the device, the fp32 dtype, the rank, non-empty dimensions, C-contiguity (size-1
    dimensions free) and alignment.
  - `byte_offset` is honoured (`:212`).
  - `common_device` (`:263`) and `expect_disjoint` (`:281`) check the device ordinals and that the written operands do
    not overlap any input.
- **Plan/run split** (`src/bindings.cpp`):
  - `conv1d_plan` (`:218`) peeks the input capsules without consuming them and sizes the output and the im2col scratch
    (`plan_conv1d`, `:58`; the sign-safe output length at `:96`; scratch `C_in*K*T_out` at `:110`).
  - The Python layer allocates both. `conv1d_run` (`:231`) consumes everything, validates it and launches.
  - ConvTranspose follows the same pattern (`:261`, `:274`; the `output_padding` rule at `:130`).
  - C++ allocates no device memory. cuBLAS's own allocation, measured at 10.0 MiB at handle creation and 0 MiB per
    GEMM, is documented in the README (receipt `artifacts/p1t_m2/cublas_workspace_probe.txt`).
- **cuBLAS handle** (`conv.cu:100`): created lazily, one per device. The settings are explicit:
  - `cublasSetStream(h, 0)` (`:106`);
  - `CUBLAS_DEFAULT_MATH` (`:107`);
  - `CUBLAS_ATOMICS_NOT_ALLOWED` (`:108`);
  - pointer mode host.
- **Conv1d** (`conv.cu:199`), for each batch element n:
  - `im2col1d_kernel` (`:119`) computes `col[(c*K+k), t] = x[n, c, t*stride - pb + k*dilation]`, zero outside
    (`:128`).
  - One `cublasSgemmStridedBatched` call (`:229`) runs over the groups. The row-major `Y_g = W_g·Col_g` is computed as
    the column-major `Y_g^T = Col_g^T·W_g^T`, with m = T_out, n = C_out/groups, k = C_in/groups·K,
    lda = T_out, ldb = C_in/groups·K and ldc = T_out.
  - `bias_add_kernel` (`:134`) follows.
  - Direct path: when K = 1, stride is 1 and there are no pads, the GEMM runs on `x` itself with no scratch
    (`conv.cu:212-215`).
  - The scratch is reused across n in stream order.
- **ConvTranspose1d**: `conv_transpose1d_gather_kernel` (`conv.cu:147`) runs one thread per output element
  (n, o, t) and computes `y = b[o] + Σ_k Σ_{c∈group(o)} w[c, o−g·O_g, k]·x[n, c, (t+pb−k·d)/s]`. Only taps with
  `t+pb−k·d >= 0`, divisible by s (`:167`), with a quotient < T contribute. The fixed order is k ascending (outer),
  then c ascending (inner), with `fmaf` (`:172`) and the bias added last (`:174`). Depthwise (groups = C_in) is the
  same kernel with an inner loop of length 1.
- **Fast-math canary.** nvcc 12.6 defines no macro for `--use_fast_math` (ledger 19:43), so the compile-time guard was
  dropped. `fp32_divide_probe_kernel` (`conv.cu:183`) is compiled with the package flags and tested bitwise against
  host IEEE division, plus a check that subnormals are not flushed.

## Accumulation-order choice

- **Pre-registered rail** (ledger 19:40, before any build): ConvTranspose stays sequential fp32 (k outer ascending,
  c inner ascending, fmaf, bias last) unless a Kokoro case on real data shows N > 5e-6. 5e-6 is the runner's F1
  line, half the gate.
- **Measured:** the T3 max N on the real fixtures is 1.228e-06 (wfp32) and 1.265e-06 (wfp16). The worst case is
  ups.0/ups.1, with up to 512 channels x 2 taps = 1,024 terms per output. **The rail did not fire, and the sequential
  order is kept.** No blocked or pairwise order was implemented.
- **Conv1d** uses cuBLAS SGEMM's own fp32 order (default math, atomics not allowed), with the bias added after the GEMM
  by one fp32 add. The T2 max N on real data is 1.394e-06.

## G3 mutant table

| Mutant | Defect (diff) | G1 own tests | G2 Voice runner | Killed by |
|---|---|---|---|---|
| (a) | im2col ignores dilation: `k * dilation` -> `k` (`mutants/a/mutant_a.diff`) | **FAIL**: 10 failed / 157, the 10 dilated census cells, N 0.65-1.24 | **FAIL** exit 1: 48/172 failed, the 12 dilated generator nodes x 2 cases x {fp32, fp16}; N 0.289-1.26 | G1 and G2 (N gate) |
| (b) | pad_begin and pad_end swapped, both ops (`mutants/b/mutant_b.diff`) | **FAIL**: 13 failed / 157, exactly the 13 asymmetric-pad cells (7 conv1d + 6 convT), N 0.47-1.40; no symmetric cell failed | PASS exit 0, 172/172 (every Kokoro pad is symmetric) | **G1 asymmetric cells only** (as required) |
| (c) | TF32 math on the cuBLAS handle: `CUBLAS_TF32_TENSOR_OP_MATH` (`mutants/c/mutant_c.diff`) | **FAIL**: 25 failed / 157 (20 conv1d census cells N 1.8e-4-3.2e-4, 4 byte_offset tests, the handle-config probe) | **FAIL** exit 1: 126/172 failed (T2 126/144), min failing N 2.27e-5, max N 5.65e-4; the 18 surviving T2 cases have C_out <= 22 or C_in = 1 (N <= 1.2e-6); T3 has no GEMM | G1 and G2 (N gate) |
| (d) | ConvTranspose as an atomicAdd scatter (init + `atomicAdd`) (`mutants/d/mutant_d.diff`) | **FAIL**: 14 failed / 157; 14/21 convT cells non-deterministic with **5 to 55,337** mismatching elements; N gates all pass (max N 2.3e-6). The 7 deterministic survivors have <= 2 contributions per output | **FAIL** exit 1: **28/28 T3 cases non-deterministic**, **3 to 1,696,791** mismatching elements per case (9,117,600 in total; ups.0 max_1 1,696,791/2,096,640 = 80.9%); 0 N-gate failures; T2 144/144 PASS | determinism check (G1 and G2) |
| (e) extra, not ordered | `--use_fast_math` forced on the CUDA sources (`mutants/e/mutant_e.diff`) | **FAIL**: 1 failed / 157, only the fast-math canary (subnormal quotients flushed to 0) | PASS exit 0, N identical to the clean build | canary only |

Survivors (an honest account; none is a gate failure):
- (b) survives Voice's runner, as expected, because Kokoro has no asymmetric pads.
- (c) survives on small-GEMM cases where cuBLAS does not choose TF32 tensor-op kernels.
- (d) passes every N gate; only the determinism check catches it.
- (e) is invisible to every conv numeric gate.

## G5 overhead table (median of 200 calls after 20 warm-ups)

Settings:
- conv1d with C_out = C_in and pads (1, 1). The order leaves C_out and the pads open, so Kokoro's K3 set is used.
- torch CUDA operands and the torch allocator.
- RTX 4070 SUPER, driver 595.99.02, under the GPU lock.

| x | w | tc_signal wall us | tc_signal GPU us | tc_signal wall+sync us | plan / alloc / run us | one torch `__dlpack__` us | torch F.conv1d wall / GPU / wall+sync us |
|---|---|---|---|---|---|---|---|
| [1, 1, 8] | [1, 1, 3] | 149.6 | 162.8 | 172.5 | 83.3 / 19.9 / 82.9 | 18.02 | 36.3 / 50.2 / 60.3 |
| [1, 512, 1193] | [512, 512, 3] | 202.9 | 317.2 | 325.2 | 81.1 / 20.5 / 86.7 | 22.45 | 74.6 / 151.6 / 157.2 |

Column meanings:
- **wall** is host time from Python, with no sync.
- **GPU** is the CUDA-event time on the legacy stream, around the call. For the tiny shape this is host launch
  spacing: the GPU idles while the host prepares the call.
- **plan / alloc / run** were measured in a separate loop, so they do not sum exactly to wall:
  - plan: 3 x `__dlpack__`, peek and sizing;
  - alloc: output and scratch through `torch.empty`;
  - run: consumption of 5 capsules, including the out and scratch `__dlpack__` calls, validation, 3 launches and the
    deleters.

The per-call cost is dominated by torch's Python-level `__dlpack__`, at 18-22 us per operand. A conv1d with bias
calls it 5 times, about 100 us of the ~150 us. With TensorCUDA producers after M1, that term becomes TensorCUDA's own
C++ `__dlpack__`, which is unmeasured here.

**Evidence class:** micro-benchmark. It covers launch overhead only, not model speed (P2).

## G4 details

**`ldd`**, from `artifacts/p1t_m2/g4_ldd.txt`:
```
tc_signal: libcudart.so.12 => /usr/local/cuda-12.6/lib64/libcudart.so.12
core:      libcudart.so.12 => /usr/local/cuda-12.6/lib64/libcudart.so.12
SAME PATH: yes
```
- `ldd tc_signal` has no `_tensor_cuda` line and no static cudart.
- The link line has no `cudart_static`; it links `-lcudart` and `.../libcudart.so`.
- 0 cuda* symbols are defined in the module, and 13 are imported `@libcudart.so.12`.
- RUNPATH is `/usr/local/cuda-12.6/lib64`.

**Grep:**
`grep -rn "tensor_cuda/include\|_tensor_cuda" tc_signal/src tc_signal/CMakeLists.txt` prints nothing (exit 1).

**One process** (`logs/p1t_m2_g4_one_process.log`):
- Setup: `PYTHONPATH=<wt>/tc_signal:/mnt/ForgeRealm/Project-Tensor/tensor_cuda`, importing production `tensor_cuda`
  (`.../tensor_cuda/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so`), `tc_signal` and torch.
- A core op works: `[1, 3, 5, 7, 9, 11]`.
- The default allocator raises the RuntimeError naming `set_allocator`.
- conv1d [1,512,1193] K3 has N 2.74e-7; conv_transpose1d K20 s10 has N 4.48e-7.
- Exactly one `libcudart.so.12` file is mapped (`libcudart.so.12.6.77`); torch maps its own `libcudart.so.13`.
- Result: `"G4_one_process": "PASS"`.

**Diff stat:** see "Deviations" 1. The non-git equivalent is in `artifacts/p1t_m2/diffstat_nongit.txt` and in the final
message.

## Deviations from the order, with reasons

1. **G4 `git diff 9df5bef --stat` was not run.**
   - Reason: the order's Rules say "NO git (the lead commits)", and the dispatch's hard rules repeat it.
   - Substitute: `artifacts/p1t_m2/diffstat_nongit.txt` lists every file under the worktree newer than the dispatch
     (19:26), with line counts, and checks that each lies in `tc_signal/**`, `docs/P1T_M2_*.md`,
     `artifacts/p1t_m2/**` or `logs/**`.
   - Lead action: run the git form.
   - Note: the root `.gitignore` ignores `*.log`, `*.json` and `*.so`, so the log and JSON receipts need `git add -f`.
2. **RUNPATH set explicitly.** `SKIP_BUILD_RPATH` plus `LINKER:-rpath,/usr/local/cuda-12.6/lib64`, instead of CMake's
   automatic RPATH.
   - Reason: without it, `ldd` resolves the same file through a different path string
     (`targets/x86_64-linux/lib`), and G4 requires the same path as core's production module.
3. **Additions to the operand contract:**
   - `__dlpack_device__()` is queried before `__dlpack__`.
   - `max_version=(1, 3)` is passed, with the TypeError fallback.
   - `copy=False` is passed for written operands.
   - Extra refusals: empty dimensions, misalignment, read-only or copied outputs, and scratch overlap.
   - Reason: the array API consumer rules, plus safety. Without the `__dlpack_device__` check, a CPU producer would
     reject `stream=1` with its own error, before the documented ValueError.
4. **Additions to the private module `_tc_signal`:** `cublas_config`, `build_info` and `fp32_divide_probe`, the last
   being a small fast-math canary kernel.
   - Reason: nvcc 12.6 has no fast-math macro (ledger 19:43), so the "no fast-math" rule needs a runtime witness.
     Mutant (e) proves it has teeth.
   - The public API is exactly the order's three functions.
5. **`set_allocator(None)`** restores the default allocator. This is an extension; the order specifies only `fn`.
6. **One cuBLAS handle per device**, not a single global handle. On this single-GPU machine that is one handle; it
   keeps a handle bound to its device if a second GPU ever appears.
7. **The census adds the ConvTranspose set K 20, stride 5, pads (0, 0), 11 -> 1.** It is in Voice's fixtures
   (`decoder/generator/ConvTranspose`, `_1`), but the order's census line omits it.
8. **G5 shape details** were chosen (C_out = C_in, pads (1, 1)), because the order leaves them open.
9. **Own-test command:** the receipt run added `-rA`, to list every test id, and `TC_SIGNAL_TEST_RECEIPT`, to write the
   per-cell N table, to the order's `pytest tc_signal/tests -q -p no:cacheprovider`. The gate is unchanged.
10. **The vendored `dlpack.h` is byte-identical to torch's copy** (sha256 62d0acfd...8d73). That copy carries torch's
    inert `TORCH_STABLE_ONLY` / `TORCH_TARGET_VERSION` guard; the license header is intact.
11. **An extra mutant (e)** goes beyond the order's list. Reason: to test the fast-math canary.

## Findings and residuals

**Findings**
- **Production core's `.so` is not a CMake build.** It was linked by `scripts/apa_*_build.sh` (Oct 8), with RUNPATH
  `/usr/local/cuda-12.6/lib64`. A fresh `tensor_cuda/build.sh` build would record `targets/x86_64-linux/lib`. That is
  the same file, but anyone comparing ldd paths should know.
- **nvcc 12.6 defines no preprocessor macro for `--use_fast_math`**, so source-level `#error` guards are inert. The
  guard is the configure-time check plus the runtime canary.

**Residuals**
- **Overhead:** about 150 us per call on the host with torch producers (G5). It is not optimized: speed is P2, out of
  scope.
- **The cross-stream lifetime caveat is documented, not enforced** (README "Streams and lifetimes"). Deleters run after
  the launch and before GPU completion. This is safe for producers on the legacy stream; others must order themselves.
- **Multi-GPU is untested on real hardware:** this machine has one GPU. Ordinal mismatch is tested only with a fake
  producer (device_id 1).
- **The acceptance run with a TensorCUDA producer and the `tensor_cuda.empty` allocator is not done.** It needs M1, and
  it is the lead's run.
- **No blind or cross-model red team was run on this code.** Under HOUSE_RULES §8 that is the lead's step.

## Prior art

| What | Source (year) | Taken | Ours |
|---|---|---|---|
| im2col + GEMM convolution | Chellapilla, Puri & Simard, "High Performance Convolutional Neural Networks for Document Processing" (2006) | unfolding receptive fields into a matrix for one SGEMM | the 1-D unfold with dilation and asymmetric pads; per-batch scratch reuse; groups as a strided-batched SGEMM |
| Row-major GEMM via column-major BLAS (`C^T = B^T A^T`) | standard BLAS practice (no single source) | the mapping | none |
| Gather-form transposed convolution | Dumoulin & Visin, "A guide to convolution arithmetic for deep learning" (2016), transposed conv as the adjoint of strided conv | the tap condition `(t + pb − k·d) % s == 0` | the fixed loop order, the thread mapping and fmaf accumulation |
| Semantics | ONNX Conv and ConvTranspose operator specs | pads (begin, end), the ConvTranspose weight layout, the output_padding rule | none |
| DLPack ABI and capsule rules | dmlc/dlpack (2017-), header v1.3 | DLManagedTensor(Versioned), capsule names, consumer calls the deleter, `used_*` rename | none |
| `__dlpack__(stream=, max_version=, copy=)`, `__dlpack_device__` | Python array API standard, data-interchange (2021-2023); NumPy and PyTorch `from_dlpack` | stream=1 for the legacy stream, the TypeError fallback, the device query first | applying it to output and scratch operands (copy=False) |
| Pluggable allocator | CuPy `cupy.cuda.set_allocator` (2017-) | caller-controlled allocation | the allocator returns DLPack tensors, not pointers |
| Plan/run (size the workspace, the caller allocates, execute) | cuDNN/cuBLASLt workspace-query APIs | the pattern | applying it to DLPack operands |
| Mutation testing | DeMillo, Lipton & Sayward (1978) | the method | the four registered defects are the order's; (e) is mine |
| Normwise error metric | Higham, *Accuracy and Stability of Numerical Algorithms* (2nd ed., 2002) | the metric (as in Voice's runner) | none |
| Fast-math canary (IEEE division and subnormal check) | no specific prior art known to me (the CUDA C Programming Guide documents FTZ as a fast-math effect) | none | the canary |

Unverified in this sandbox (lead to check): exact titles and years above. Search terms: "Chellapilla Puri Simard 2006
convolutional document processing", "Dumoulin Visin convolution arithmetic 2016", "DLPack v1.3 dlpack.h",
"array API __dlpack__ max_version copy 2023.12", "cupy.cuda.set_allocator".

## Files (all new; nothing outside the allowed paths changed)

- **Package:** `tc_signal/` contains `CMakeLists.txt`, `build.sh`, `.gitignore`, `README.md`,
  `include/dlpack/dlpack.h`, `src/{tuning.h, dlpack_operand.h, dlpack_operand.cpp, conv_ops.h, conv.cu, bindings.cpp}`,
  `tc_signal/__init__.py` (plus the built `.so`, gitignored), and
  `tests/{conftest.py, tcs_census.py, tcs_oracle.py, tcs_producer.py, test_conv.py, test_contract.py}`.
- **Artifacts:** `artifacts/p1t_m2/` contains `tc_signal_torch_adapter.py`, `p1t_T2T3_torchadapter.json`,
  `g1_census_receipt.json`, `g4_ldd.txt`, `g4_one_process.py`, `g5_overhead.py`, `cublas_workspace_probe.py`,
  `cublas_workspace_probe.txt`, `diffstat_nongit.txt`, and
  `mutants/{make_mutants.py, summarize.py, mutant_summary.json, a/, b/, c/, d/, e/}`.
- **Logs:** `logs/p1t_m2_build.log`, `logs/p1t_m2_g1_own_tests.log`, `logs/p1t_m2_g2_runner.log`,
  `logs/p1t_m2_g4_one_process.log` and `logs/p1t_m2_g5_overhead.log`.
- **Docs:** `docs/P1T_M2_REPORT.md` and `docs/P1T_M2_LEDGER.md`.
- **Scratch outside the worktree, never deleted:**
  - `/tmp/p1t_m2_scratch`;
  - `/tmp/p1t_m2_mutants`;
  - `/tmp/p1t_m2_mut_{a,b,c,d,e}`;
  - `/tmp/p1t_m2_stale_build_1` and `/tmp/p1t_m2_stale_build_2`, the moved development build directories.
