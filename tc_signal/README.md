# tc_signal: TensorCUDA add-on for signal ops

`tc_signal` is a separate package next to `tensor_cuda/`. It holds the ops Project-Voice needs to run Kokoro-82M on
TensorCUDA, and it keeps them out of TensorCUDA's core. David, 2026-10-12: "I don't want to build it directly INTO the
core of TensorCUDA - It needs to be separate". Plan: `Fork/tensor_fork_voice/docs/P1T_TENSOR_ALT_PLAN.md` §4.

Status (P1T-M2): `conv1d` (T2) and `conv_transpose1d` (T3). T4-T6 come in later milestones.

- **No core dependency.** It does not link `_tensor_cuda`, include `tensor_cuda/include`, or compile any core source.
- **Same CUDA runtime as core.** It links the shared CUDA 12.6 runtime (`libcudart.so.12`) and `libcublas.so.12`, with
  RUNPATH `/usr/local/cuda-12.6/lib64`, the same path core's module resolves. In one process the runtime is loaded
  once, so tc_signal and core share one runtime instance, one context and one default memory pool.
- **Any DLPack CUDA producer works.** Operands can be TensorCUDA tensors (after P1T-M1), torch tensors, or bare DLPack
  capsules.

## Build

```bash
# offline pybind11 v2.12.0 (the same as core); the source tree is read-only
PYBIND11_SOURCE_DIR=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src ./build.sh 89
# -> tc_signal/_tc_signal.cpython-312-x86_64-linux-gnu.so
```

`build.sh [arch]` mirrors `tensor_cuda/build.sh`:
- It uses the newest `/usr/local/cuda-*` (CUDA 12.6 here). Override it with `CUDA_HOME`.
- The architecture defaults to 89.
- It builds Release, with IPO/LTO off, as core does.
- `PYBIND11_SOURCE_DIR` is passed as `-DFETCHCONTENT_SOURCE_DIR_PYBIND11`. Without it, CMake fetches v2.12.0 from
  GitHub.

`CMakeLists.txt` refuses two kinds of flag at configure time:
- fast-math flags (`--use_fast_math`);
- `--default-stream` flags. `conv.cu` also refuses `CUDA_API_PER_THREAD_DEFAULT_STREAM`.

`build/` is git-ignored.

## Use

```python
# PYTHONPATH=<Project-Tensor>/tc_signal
import torch, tc_signal as ts
ts.set_allocator(lambda s: torch.empty(s, device="cuda"))     # until P1T-M1 gives tensor_cuda.empty
y = ts.conv1d(x, w, b, stride=1, padding=(3, 3), dilation=3)
z = ts.conv_transpose1d(y, wt, bt, stride=10, padding=(5, 5))
```

## API

```python
tc_signal.conv1d(x, w, b=None, *, stride=1, padding=0, dilation=1, groups=1, out=None)
tc_signal.conv_transpose1d(x, w, b=None, *, stride=1, padding=0, output_padding=0, dilation=1, groups=1, out=None)
tc_signal.set_allocator(fn)   # fn(shape: tuple) -> fp32 CUDA DLPack tensor; returns the previous allocator
```

The semantics are ONNX `Conv` and `ConvTranspose`, 1-D.

**Shapes**
- `x` is `[N, C_in, T]`.
- `b` is `[C_out]` or `None`.
- `padding` is an int or `(pad_begin, pad_end)`.
- `conv1d`: `w` is `[C_out, C_in/groups, K]`. The output length is
  `(T + pb + pe - dilation*(K-1) - 1)//stride + 1`.
- `conv_transpose1d`: `w` is `[C_in, C_out/groups, K]` (the ONNX layout). The output length is
  `stride*(T-1) + output_padding + dilation*(K-1) + 1 - pb - pe`. Output positions past the full transposed length
  hold only the bias.

**Attribute limits.** `stride`, `dilation` and `groups` must be >= 1; pads >= 0; `output_padding` >= 0 and
< `max(stride, dilation)`. An output length <= 0 is refused.

**Allocation**
- With `out=None`, the output comes from the module allocator. The function always returns `out`.
- The default allocator is `tensor_cuda.empty(*shape, dtype="float32")`, which P1T-M1 adds. Until then it raises a
  `RuntimeError` that names `set_allocator`.
- `set_allocator(None)` restores the default.
- The im2col scratch for `conv1d` also comes from the allocator. Its shape is `(C_in*K*T_out,)`, sized by the C++
  helper `conv1d_plan`. The scratch is skipped for the direct path (K 1, stride 1, no pads).

## Operand contract (DLPack consumer)

Every operand, the output and the scratch included, is an object with `__dlpack__` or a bare DLPack capsule.

**Acquisition**
1. If the object has `__dlpack_device__()`, it is asked first. A non-CUDA producer is refused there, before
   `__dlpack__` is called, because CPU producers such as NumPy reject a CUDA stream argument.
2. Then `obj.__dlpack__(stream=1, max_version=(1, 3))` is called. Stream 1 is the legacy default stream, the
   consumer's stream. Written operands also pass `copy=False`.
3. If the producer raises `TypeError` for the newer keywords, it is called again as `obj.__dlpack__(stream=1)`. This
   is the Python array API fallback.

**Both capsule kinds are read**
- `"dltensor"` (`DLManagedTensor`);
- `"dltensor_versioned"` (`DLManagedTensorVersioned`). For this kind, DLPack major version 1 is required.

**Consumption**
- A capsule is consumed once: it is renamed to `"used_*"`, and tc_signal owns the managed tensor from then on.
- Each producer deleter is called exactly once:
  - after the launch on success;
  - during unwinding on any error after consumption;
  - by the producer's own capsule destructor if tc_signal never consumed the capsule (a refusal in the planning phase).

**Validation**, each failure a `TypeError` or `ValueError` (see Errors):
- device `kDLCUDA`, with one ordinal for every operand;
- dtype float32 (`kDLFloat`, 32 bits, 1 lane);
- the exact number of dimensions;
- no empty dimension;
- C-contiguous: NULL strides, or the compact stride on every dimension of extent > 1 (size-1 dimensions may carry
  any stride);
- `data + byte_offset` used as the base pointer, 4-byte aligned;
- exact shapes;
- written operands must not be read-only or a producer copy (versioned flags);
- the output's byte range must not overlap any input's or the scratch's, and the scratch must not overlap the inputs.

## Streams and lifetimes

- Everything is queued on the legacy default stream: the im2col, GEMM and bias work, and the ConvTranspose gather.
  This is TensorCUDA's only stream, and torch's default stream; both share the device's primary context.
  `tests/test_contract.py` proves the ordering both ways against torch with no explicit sync.
- `cudaGetLastError` is checked after every launch, and any error raises a `RuntimeError`.
- Each capsule's deleter is called right after the launches, before the GPU has run them. This is safe when the
  producer frees on the legacy stream, or later in stream order: TensorCUDA does, and so does torch's caching
  allocator on its default stream.
- **Cross-stream lifetime caveat:** a producer whose tensor lives on a non-legacy stream must keep that tensor alive,
  and must not reuse its memory, until the legacy stream has passed the tc_signal op. For example, record an event on
  the legacy stream after the call and make its own stream wait on it, or synchronize. Producers that are asked for
  `stream=1` order their pending work before the op; DLPack has no mechanism for the reverse direction.
- Outputs are written on the legacy stream. A consumer on another stream must order itself after it.

## cuBLAS

- There is one handle per device, created on first use, kept for the life of the process, and never destroyed at
  exit, which could race CUDA's own teardown.
- The handle settings are set explicitly:
  - `cublasSetStream(h, 0)`, the legacy stream;
  - `cublasSetMathMode(h, CUBLAS_DEFAULT_MATH)`: never TF32;
  - `cublasSetAtomicsMode(h, CUBLAS_ATOMICS_NOT_ALLOWED)`;
  - `cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST)`.
- `tc_signal._tc_signal.cublas_config(device)` reports these settings.
- **cuBLAS's own internal workspace** is the one piece of device memory tc_signal causes to be allocated outside the
  caller's allocator. cuBLAS allocates it itself (cudaMalloc) for the handle.
  - Measured with `artifacts/p1t_m2/cublas_workspace_probe.py` as a cudaMemGetInfo delta on the RTX 4070 SUPER with
    cuBLAS 12.6.4.
  - The handle's first use costs **10.0 MiB**, and the runtime initialisation is counted in that figure.
  - The first and later SGEMMs add **0 MiB**.

## Numerics

- **fp32 math throughout.** There are no fast-math intrinsics and no TF32.
  - nvcc 12.6 defines no preprocessor macro for `--use_fast_math`, so fast-math cannot be refused at compile time.
  - Instead, `fp32_divide_probe` is compiled with the package's flags, and a test compares it bitwise against host IEEE
    division and checks that subnormal results are not flushed.
- **Deterministic.** Each output element is produced by one thread, or by one cuBLAS dot product, in a fixed order.
  There is no `atomicAdd` anywhere in the package.
- **conv1d**
  - Per batch element n: the im2col kernel computes `col[(c*K + k), t] = x[n, c, t*stride - pb + k*dilation]`, with 0
    outside `[0, T)`.
  - Then one `cublasSgemmStridedBatched` call over the groups computes `Y_g = W_g · Col_g`, in row-major terms. Through
    column-major cuBLAS it runs as `Y_g^T = Col_g^T · W_g^T`, with `m = T_out`, `n = C_out/groups`,
    `k = C_in/groups*K`.
  - A bias kernel follows.
  - The direct path for K 1, stride 1 and no pads runs the GEMM on `x` itself.
  - The accumulation order is cuBLAS's own, in fp32. The bias is added after the GEMM, with one rounding.
- **conv_transpose1d** uses the gather form, with one thread per output element (n, o, t):
  - `y = b[o] + Σ_k Σ_{c in group(o)} w[c, o - g·O_g, k] · x[n, c, (t + pb - k·dilation)/stride]`;
  - the sum runs over the taps where `t + pb - k·dilation` is >= 0, divisible by `stride`, and has its quotient < T;
  - the fixed order is k ascending (outer), then c ascending (inner), with `fmaf` accumulation from 0 and the bias
    added last.
- **Accuracy** on Voice's real fixtures (`run_p1t.py`, `tolerances_r2.json`): the largest normwise error is 1.4e-6
  against the 1e-5 gate. The worst ConvTranspose case, Kokoro's ups.0 with 1,024 terms per output, is at 1.23e-6. The
  order's blocked/pairwise option was therefore not needed.

## Errors

| Error | When |
|---|---|
| `TypeError` | an operand is not a DLPack producer or capsule, or is a capsule of another kind; the dtype is not float32 (fp16 operands are refused: expand them first); an attribute is not an int (bool included); `padding` has the wrong type; `set_allocator` gets a non-callable; `__dlpack__` returns something other than a capsule |
| `ValueError` | the device is not CUDA; the device ordinals differ; an operand is non-contiguous, misaligned, empty, of the wrong rank or of the wrong shape; the groups do not divide the channels; an attribute is out of range; the output length is <= 0; `output_padding >= max(stride, dilation)`; the output overlaps another operand; a written operand is read-only or a producer copy; a capsule was already consumed; the DLPack major version is not 1 |
| `RuntimeError` | the default allocator is used without `tensor_cuda.empty` (the message names `set_allocator`); a CUDA launch or cuBLAS error |

## Tests

```bash
cd <Project-Tensor> && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<Project-Tensor>/tc_signal \
  python3 -m pytest tc_signal/tests -q -p no:cacheprovider
```

- `test_conv.py` covers every census cell (`tcs_census.py`): Kokoro's attribute sets and the edge cells. For each one
  it checks the exact shape, that every element is written (the allocator pre-fills NaN), normwise N <= 1e-5 against a
  torch float64 CPU oracle (`tcs_oracle.py`), and that two calls are bitwise equal. It also checks the oracle against
  torch's native padding path.
- `test_contract.py` covers:
  - every documented refusal;
  - the DLPack consumer rules, using torch and a hand-built ctypes producer (`tcs_producer.py`): stream=1,
    max_version, copy=False, the TypeError fallback, `byte_offset`, size-1 strides, versioned and legacy capsules, bare
    capsules, and exactly one deleter call on every path;
  - the allocator;
  - stream ordering against torch;
  - the cuBLAS settings;
  - the fast-math canary.
- The module skips only when torch is not importable or there is no CUDA device.
- `TC_SIGNAL_TEST_RECEIPT=<path>` writes each cell's N and determinism to JSON.

## Vendored `include/dlpack/dlpack.h`

The header is DLPack v1.3 (dmlc/dlpack, Apache-2.0; header "Copyright (c) 2017 - by Contributors"). It was copied
byte-for-byte from torch 2.11's `torch/include/ATen/dlpack.h` (sha256
`62d0acfd441700f9b970ef2d694bde8628ccd4a36b0b5d311592768350f68d73`). That copy includes torch's
`TORCH_STABLE_ONLY` / `TORCH_TARGET_VERSION` include guard, which is inert here.

## Prior art

- **im2col + GEMM convolution:** Chellapilla, Puri & Simard, 2006. tc_signal adds the 1-D form with dilation and
  asymmetric pads, and batches the groups in a strided-batched SGEMM.
- **Gather-form transposed convolution:** the adjoint of strided convolution (Dumoulin & Visin, 2016). tc_signal's own
  parts are the loop order and the thread mapping.
- **Semantics:** the ONNX `Conv` and `ConvTranspose` operator specs.
- **DLPack:** dmlc/dlpack v1.3, and the Python array API data-interchange protocol (`__dlpack__(stream=, max_version=,
  copy=)`, `__dlpack_device__`). NumPy's and PyTorch's `from_dlpack` use the same TypeError fallback.
- **Pluggable allocator:** after CuPy's `cupy.cuda.set_allocator`. tc_signal's allocator returns DLPack tensors where
  CuPy's returns pointers.
- **Plan/run split** (size the scratch, let the caller allocate, then execute): the workspace-query pattern of
  cuDNN/cuBLASLt.
- **Fast-math canary:** no specific prior art known to me.
