# P1T-M2: the `tc_signal` add-on package, with Conv1d (T2) and ConvTranspose1d (T3)

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_m2`
(branch `voice-m2`, based on production main `9df5bef`). Edits, builds and GPU runs inside it are AUTHORIZED. You
may also write scratch and mutant builds under `/tmp/p1t_m2_*`.

## Authority

David, 2026-10-12, in the Project-Tensor session, verbatim:
- "I don't want to build it directly INTO the core of TensorCUDA - It needs to be separate I guess. I don't want to
  bloat the kernels if we don't have too."
- "I'd follow your reccomended option": a separate add-on package inside the Project-Tensor repo.

The plan is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/docs/P1T_TENSOR_ALT_PLAN.md` (fork 326fdfe).
Project-Voice approved it on 2026-10-12 (their LD51), including "M2 alongside M1" and "report µs per call".

## Background

Project-Voice is porting Kokoro-82M to run natively on TensorCUDA.
- Core TensorCUDA gets only DLPack interop. That is order M1, built in parallel in another worktree; do not touch it.
- Every other op Voice needs goes into a NEW package, `tc_signal/`, at the repo root next to `tensor_cuda/`.
- This order builds the package skeleton plus its first two ops.
- The package must work with ANY DLPack CUDA producer. Until M1 lands you develop and test with torch CUDA tensors.
  The lead merges M1 later and runs the TensorCUDA acceptance.

## Mechanism & scope

- **Measures:**
  - whether `tc_signal.conv1d` and `tc_signal.conv_transpose1d` match ONNX Conv/ConvTranspose semantics (1-D) within
    normwise 1e-5 of an fp64 oracle, deterministically, on Kokoro's real operands and on edge shapes;
  - whether the package is fully decoupled from core (no link, include or source dependency);
  - the per-call overhead.
- **Mechanism doc:** `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/docs/P1T_TENSOR_ALT_PLAN.md` §4 (the add-on
  contract and API). Also `/mnt/ForgeRealm/Project-Voice/research/kernels/TENSOR_IMPL_PLAN_P1T.md` §4 (numerics) and
  §5 T2/T3 (formulas), and `/mnt/ForgeRealm/Project-Voice/engine/tensor_requests/p1t_tests/adapter_api.py` (the
  ONNX semantics of `conv1d`/`conv_transpose1d`).
- **Exercises:**
  - new `tc_signal/`: CMake, `build.sh`, `include/dlpack/dlpack.h` vendored, `src/*.cu|*.cpp`,
    `tc_signal/__init__.py`, `tests/`, `README.md`, `.gitignore`;
  - a torch-backed adapter `artifacts/p1t_m2/tc_signal_torch_adapter.py`;
  - Project-Voice's runner `run_p1t.py --items T2,T3`.
- **Does not cover:**
  - T4–T6;
  - anything under `tensor_cuda/` (core is OFF LIMITS: no edits, no includes, no linking);
  - DLPack export from `tc_signal` (it only consumes);
  - fp16 compute (fp16 operands are refused; the adapter expands weights to fp32);
  - autograd;
  - speed beyond correctness-first (P2);
  - the TensorCUDA-backed acceptance run, which needs M1 and is the lead's.
- **Required evidence:**
  - the build log, plus `ldd` of `_tc_signal*.so` (it must show the SAME `libcudart.so.12` path as core's, and no
    `_tensor_cuda`);
  - the own-test log;
  - the runner report JSON and verdict line for `--items T2,T3`;
  - the mutant table;
  - the overhead table;
  - the diff stat.
- **Operand census:**
  - **Kokoro attribute sets** (from the fixtures manifest; read it):
    - conv1d: K ∈ {1, 3, 5, 7, 11, 12, 20}; dilation ∈ {1, 3, 5}; stride ∈ {1, 2, 5, 6}; symmetric pads; group 1;
    - conv_transpose1d: K 20 stride 10 pads (5, 5); K 12 stride 6 pads (3, 3); depthwise group ∈ {512, 1090}, K 3,
      stride 2, pads (1, 1), output_padding 1.
  - **Your own edge cells:** asymmetric pads (pb ≠ pe); groups ∈ {2, 4, C}; T < K; stride > K; dilation·(K−1) ≥ T;
    N = 2; output length 1; C_out not a multiple of 32.
- **Baseline isolation:**
  - The oracle is torch fp64 on CPU, computed in the same process.
  - The runner's oracles are Voice's, and are not yours to change.
  - Mutants are built and run outside the worktree.
  - Core's suite is not affected, because core is not touched. G4 checks that.
- **Forbidden configuration:**
  - `--use_fast_math` or `-use_fast_math`, and `__expf`/`__fdividef`-style intrinsics;
  - TF32/`CUBLAS_TF32_TENSOR_OP_MATH`, or setting `NVIDIA_TF32_OVERRIDE`;
  - `atomicAdd` or any order-dependent reduction;
  - static cudart (`cudart_static`);
  - linking or including anything from `tensor_cuda/`;
  - the retired 595.84 driver override;
  - writing outside the target or `/tmp`.

## Spec

### Package layout and build

- `tc_signal/CMakeLists.txt` and `tc_signal/build.sh [arch]`. Mirror `tensor_cuda/build.sh`: the newest
  `/usr/local/cuda-*` (12.6), arch default 89. The build drops `_tc_signal.cpython-312-x86_64-linux-gnu.so` into
  `tc_signal/tc_signal/`.
- **pybind11 v2.12.0, the same as core, and offline.** `build.sh` must honour `PYBIND11_SOURCE_DIR` and pass it as
  `-DFETCHCONTENT_SOURCE_DIR_PYBIND11`. Build with
  `PYBIND11_SOURCE_DIR=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src`, which is read-only.
- **Link the CUDA runtime SHARED** (`CMAKE_CUDA_RUNTIME_LIBRARY Shared`), so one cudart instance, one context and one
  default memory pool are shared with core. Link `CUDA::cublas`. Disable IPO/LTO as core does.
- Vendor `tc_signal/include/dlpack/dlpack.h` from
  `/home/vader/.local/lib/python3.12/site-packages/torch/include/ATen/dlpack.h` (v1.3), with the license header intact.
- Add `tc_signal/.gitignore` for `build/`.
- Usage: `PYTHONPATH=<repo>/tc_signal` makes `import tc_signal` work.

### Operand contract (C++)

- Every operand, outputs and scratch included, arrives as an object with `__dlpack__` or as a bare capsule. Consume
  it with `__dlpack__(stream=1)`: you are a consumer on the legacy default stream.
- Validate every operand, with a clear `ValueError`/`TypeError` on failure:
  - device kDLCUDA, and the same ordinal for every operand;
  - dtype float32 (kDLFloat, 32 bits, 1 lane);
  - C-contiguous (NULL strides, or compact strides; size-1 dimensions may have any stride);
  - `byte_offset` honoured;
  - exact expected shapes;
  - the output does not overlap any input's byte range.
- Launch everything on the legacy default stream. Check `cudaGetLastError` after every launch and raise on error.
- C++ allocates NO device memory. Scratch (the im2col buffer) is sized by a C++ helper, allocated by the Python layer,
  and passed in.
- The one cuBLAS handle is created lazily, with `cublasSetStream(h, 0)` and `cublasSetMathMode(h, CUBLAS_DEFAULT_MATH)`
  set explicitly. Document cuBLAS's own internal workspace.
- Call each capsule's deleter after the launch, as the DLPack consumer rules require. The cross-stream lifetime
  caveat goes in the README: a producer on a non-legacy stream must keep its tensor alive until the legacy stream has
  passed the op.

### Python API (exact)

```python
tc_signal.conv1d(x, w, b=None, *, stride=1, padding=0, dilation=1, groups=1, out=None)
tc_signal.conv_transpose1d(x, w, b=None, *, stride=1, padding=0, output_padding=0, dilation=1, groups=1, out=None)
tc_signal.set_allocator(fn)   # fn(shape: tuple) -> fp32 CUDA DLPack tensor; returns the previous allocator
```

- **Shapes:**
  - x is [N, C_in, T].
  - For conv1d, w is [C_out, C_in/groups, K].
  - For conv_transpose1d, w is [C_in, C_out/groups, K] (the ONNX layout).
  - b is [C_out] or None.
  - `padding` is an int or `(pad_begin, pad_end)`.
- **Output lengths:**
  - conv1d: `(T + pb + pe - dilation*(K-1) - 1)//stride + 1`.
  - conv_transpose1d: `stride*(T-1) + output_padding + dilation*(K-1) + 1 - pb - pe`.
  - A length ≤ 0, or `output_padding >= max(stride, dilation)`, raises `ValueError`.
- **Allocation:** `out=None` allocates through the module allocator, and the function returns `out`.
  - The default allocator is `tensor_cuda.empty(*shape, dtype="float32")`, which M1 adds.
  - When `tensor_cuda` lacks `empty` (as on this branch), the default allocator raises a clear `RuntimeError` naming
    `set_allocator`.
  - Your tests use `set_allocator(lambda s: torch.empty(s, device="cuda"))`.

### Kernels (fp32, deterministic, no atomics, no fast-math)

- **conv1d:**
  - Per (n, group): a 1-D im2col kernel builds `col[(c, k), t] = x[n, c, t*stride - pb + k*dilation]`, zero outside.
  - `cublasSgemm` then computes `y[o, t] = Σ w[o, (c, k)] · col[(c, k), t]`; mind the row/column-major mapping.
  - A bias kernel follows.
  - A fast path for K=1, stride 1, no pads, groups 1 (a plain GEMM on x) is optional.
- **conv_transpose1d:** the gather form, one thread per output element (n, o, t), in a fixed loop order:
  - `y[n, o, t] = b[o] + Σ_{c ∈ group(o)} Σ_k w[c, o - g·O_g, k] · x[n, c, (t + pb - k·dilation)/stride]`;
  - only taps where `(t + pb - k·dilation) % stride == 0` and the quotient is in [0, T) contribute.
  - The depthwise case is groups == C_in, with O_g = 1 here.
  - A GEMM plus a gather-col2im is also acceptable. Scatter is not.
- Accumulate in fp32. If a Kokoro case comes close to its gate, you may use a blocked or pairwise order; report what
  you chose.

### Torch-backed adapter (for the G2 pre-check)

Copy `/mnt/ForgeRealm/Project-Voice/engine/tensor_requests/p1t_tests/adapters/tensorcuda_stub.py` to
`artifacts/p1t_m2/tc_signal_torch_adapter.py`. Implement `conv1d` and `conv_transpose1d` per `adapter_api.py` (host
numpy in, host numpy out):
- upload with torch to CUDA;
- for `weight_storage == 'fp16'`, round w to fp16 and expand it exactly back to fp32 (`torch.float16` then `.float()`);
- call `tc_signal` with the torch allocator;
- `torch.cuda.synchronize()`;
- download.

Leave every other function raising `NotImplementedError`. This adapter is a pre-check only. The TensorCUDA-backed
adapter comes after M1 merges.

## How to run

- **GPU:** every GPU command runs as `flock -w 900 /tmp/forge-gpu.lock timeout 600 …`. If a lock wait times out,
  retry. M1's seat shares the lock, so keep each GPU process short. Use the native driver only (595.99.02).
- **Own tests:**
  `cd <target> && flock -w 900 /tmp/forge-gpu.lock timeout 600 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<target>/tc_signal python3 -m pytest tc_signal/tests -q -p no:cacheprovider`
  Default wall time ≤ 30 s. The skip guard is narrow: skip only when torch is not importable or there is no CUDA
  device; any other exception fails.
- **Voice's runner (G2).** Run it from `/mnt/ForgeRealm/Project-Voice`, which is READ-ONLY to you; outputs go to your
  target:
  ```
  cd /mnt/ForgeRealm/Project-Voice && flock -w 900 /tmp/forge-gpu.lock timeout 600 env PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=<target>/tc_signal python3 engine/tensor_requests/p1t_tests/run_p1t.py \
    --adapter <target>/artifacts/p1t_m2/tc_signal_torch_adapter.py --items T2,T3 \
    --fixtures /mnt/ForgeRealm/Project-Voice/artifacts/p1t/fixtures \
    --tolerances engine/tensor_requests/p1t_tests/tolerances_r2.json --out <target>/artifacts/p1t_m2/p1t_T2T3_torchadapter.json
  ```
  Do not edit any Project-Voice file. If you believe one of their tests is wrong, report the case and evidence.

## Gates

- **G1 (own tests):** every `tc_signal/tests` case passes. That is normwise ≤ 1e-5 against the fp64 oracle on every
  census cell, two calls bitwise equal, and every refusal raising the documented error.
- **G2 (Voice pre-check):** the runner with `--items T2,T3` and `tolerances_r2.json` exits 0 with verdict PASS. Every
  T2 and T3 case passes, including the fp16-weight-storage variants.
- **G3 (teeth):** build these mutants outside the worktree and run G1 plus G2 on each:
  - (a) im2col ignores dilation;
  - (b) pad_begin and pad_end are swapped;
  - (c) TF32 math on the cuBLAS handle;
  - (d) conv_transpose1d as an `atomicAdd` scatter.

  (a), (b) and (c) must FAIL at least one gate, and (b) must be caught by your asymmetric cells. For (d), report the
  determinism-check outcome with counts. A survivor is an honest RED with your explanation.
- **G4 (decoupling):**
  - `ldd` shows the same `libcudart.so.12` path as core's `_tensor_cuda*.so`, and no `_tensor_cuda` or static
    cudart.
  - `git diff 9df5bef --stat` shows only `tc_signal/**`, `docs/P1T_M2_*.md`, `artifacts/p1t_m2/**` and `logs/**`.
  - `grep -rn "tensor_cuda/include\|_tensor_cuda" tc_signal/src tc_signal/CMakeLists.txt` is empty.
  - In ONE process, `import tensor_cuda` (production's package, read-only, on PYTHONPATH after yours) and
    `import tc_signal` both work, and a conv on torch tensors still passes.
- **G5 (overhead, informational):** report µs per call, as the median of 200 calls after 20 warm-ups, for conv1d on
  [1, 1, 8] K 3 and on [1, 512, 1193] K 3. Give both the wall time from Python and the GPU time from CUDA events, so
  the per-call wrapper and capsule cost is visible.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. Your order file is
  immutable.
- NO git (the lead commits). NO subagents. No background waits: run every gate synchronously to completion.
- **No deletes, anywhere.** Agents never delete files; you are not covered by Claude's hooks, so this is on you.
  Anything that should go stays where it is, or sits in `/tmp`, which is exempt.
- Never kill a process you did not start.
- Production `/mnt/ForgeRealm/Project-Tensor` (outside `Fork/`), `/mnt/ForgeRealm/Project-Voice`, the M1 worktree and
  every other fork or worktree are READ-ONLY.
- RED honesty: a failed gate is a result. Report it with receipts; never weaken a gate, test or tolerance to pass.
- Prior Art Directive (AGENTS.md): annotate at the code site, in the ledger and in the report. Known prior art:
  - im2col: Chellapilla, Puri and Simard, 2006;
  - gather-form transposed convolution as the adjoint of strided convolution: Dumoulin and Visin, 2016;
  - ONNX Conv/ConvTranspose specs;
  - DLPack: dmlc/dlpack v1.3, and the array API stream protocol.

## Done

- Write `docs/P1T_M2_REPORT.md`: the gate table, the API, the build steps, kernel design with code citations, the
  accumulation-order choice, the mutant table, the overhead table and prior art. Also write `docs/P1T_M2_LEDGER.md`
  with timestamped steps.
- Your final message must contain, verbatim:
  - the gate table;
  - the G2 verdict line and report path;
  - the G3 mutant table;
  - the G4 `ldd` lines, diff stat and grep output;
  - the G5 table;
  - the built `.so` sha256;
  - every deviation from this order, with its reason.
