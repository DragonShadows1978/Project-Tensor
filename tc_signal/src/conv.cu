// tc_signal 1-D convolutions (P1T-M2: T2 Conv1d, T3 ConvTranspose1d). fp32, deterministic, no atomics, no fast-math.
//
// Numerics contract (Project-Voice TENSOR_IMPL_PLAN_P1T.md §4; tc_signal plan §4):
//  * fp32 math throughout; cuBLAS in CUBLAS_DEFAULT_MATH (never TF32) with atomics not allowed;
//  * every output element is produced by exactly one thread (or one cuBLAS dot product) in a fixed order, so two calls
//    on the same inputs are bitwise equal; no atomicAdd anywhere in this package;
//  * no intrinsics such as __expf/__fdividef and no --use_fast_math (CMakeLists.txt refuses it). FMA contraction is
//    nvcc's default (--fmad=true, not a fast-math flag); the gather kernel spells its FMA out with fmaf().
//
// Prior art:
//  * im2col + GEMM convolution: Chellapilla, Puri & Simard, "High Performance Convolutional Neural Networks for
//    Document Processing" (2006). Taken: unfold the receptive fields into a matrix and run one SGEMM. Ours: the 1-D
//    unfold with dilation and separate (pad_begin, pad_end), the per-batch-element scratch reuse, and the group batch
//    as a strided-batched SGEMM.
//  * Row-major GEMM through column-major BLAS by computing C^T = B^T A^T: standard BLAS practice (no single source).
//  * Gather-form transposed convolution: transposed convolution is the adjoint of the strided convolution
//    (Dumoulin & Visin, "A guide to convolution arithmetic for deep learning", 2016); evaluating each output as a sum
//    over the input taps that land on it (instead of scattering inputs) is the usual deterministic formulation.
//    Taken: the tap condition (t + pb - k*dilation) % stride == 0. Ours: the fixed loop order (k outer ascending,
//    c inner ascending, bias last) and the thread mapping.
//  * Semantics: ONNX Conv / ConvTranspose operator specs (1-D case; pads = (begin, end), ConvTranspose weight layout
//    [C_in, C_out/groups, K], output_padding appended at the end).
#include "conv_ops.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "cuda_guard.h"  // P1T-FIX4: shared checked DeviceGuard
#include "tuning.h"

#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
#error "tc_signal queues on the legacy default stream (stream 0); do not build it with --default-stream per-thread"
#endif
// nvcc 12.6 defines no preprocessor macro for --use_fast_math (it only passes -ftz=1 -prec_div=0 -prec_sqrt=0 to
// cicc), so fast-math cannot be refused here at compile time. CMakeLists.txt refuses the flag at configure time, and
// fp32_divide_probe_kernel below is a runtime canary compiled with this TU's flags (tests/test_conv.py checks it).

namespace tcs {

namespace {

// Stream 0 is the legacy default stream in this translation unit (per-thread default stream refused above). It is the
// same stream TensorCUDA queues on, and the stream DLPack producers were asked to order against (stream=1).
constexpr cudaStream_t kLegacyStream = 0;

void check_cuda(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    std::ostringstream os;
    os << "tc_signal: " << what << " failed: " << cudaGetErrorName(e) << ": " << cudaGetErrorString(e);
    // P1T-FIX4 (cuda_guard.h rule 1): this raise reports the failed call, so it consumes its entry; P1T-FIX8 (sweep
    // (k)): only once the report exists (cuda_guard.h throw_after_consuming).
    throw_after_consuming(os);
  }
}

// Checked after every kernel launch and after every cuBLAS call (launch-configuration and sticky errors surface here as
// RuntimeError). P1T-FIX7 (sweep (j)): through cuda_guard.h check_launch_on / check_after_cublas_on, which name the
// device and never claim that a kernel or a cuBLAS call failed when cudaGetLastError may be reporting an asynchronous
// error of earlier work (FIX6: "<kernel> failed", "cublasSgemmStridedBatched failed" after it returned success).
constexpr const char* kPrefix = "tc_signal: ";

void check_cublas(cublasStatus_t s, const char* what) {  // no device involved (cublasGetProperty)
  if (s != CUBLAS_STATUS_SUCCESS) {
    std::ostringstream os;
    os << "tc_signal: " << what << " failed: cuBLAS status " << static_cast<int>(s);
    throw std::runtime_error(os.str());
  }
}
void check_cublas_on(cublasStatus_t s, const char* what, int device) {
  if (s != CUBLAS_STATUS_SUCCESS) raise_cublas_failure_on(kPrefix, what, device, static_cast<int>(s));
}

unsigned int grid_for(int64_t total, int block) {
  const int64_t blocks = (total + block - 1) / block;
  return static_cast<unsigned int>(std::max<int64_t>(1, std::min<int64_t>(blocks, tuning::kGridCap)));
}

// DeviceGuard (makes `device` current for the call, restores the caller's device) is tc_signal's shared, checked guard
// in cuda_guard.h since P1T-FIX4 (this copy ignored the restoring cudaSetDevice's result).

// One cuBLAS handle per device, created lazily on first use and kept for the life of the process (destroying it at
// interpreter exit can race CUDA's own teardown). All entry points run under the GIL, so no extra lock is needed.
// cuBLAS allocates its own internal workspace for this handle on first use (cudaMalloc inside cuBLAS); that is the
// only device memory tc_signal causes to be allocated outside the caller's allocator (README "cuBLAS").
cublasHandle_t handle_for(int device) {
  static std::unordered_map<int, cublasHandle_t> handles;
  auto it = handles.find(device);
  if (it != handles.end()) return it->second;
  cublasHandle_t h = nullptr;
  check_cublas_on(cublasCreate(&h), "cublasCreate", device);
  // P1T-FIX5, round-5 review finding V5-03 (MINOR): FIX4 destroyed the new handle when a SETTING failed, but the handle
  // had no owner through the cache insertion below: when handles.emplace threw (std::bad_alloc for the map node) the
  // handle leaked, with its device resources (review probe handle_insert_oom_cpu.cpp: creates 1, destroys 0). Now `owned`
  // owns it from creation until the insertion commits: on ANY exception before the commit its destructor destroys the
  // handle, checks the destroy's status and, never throwing while the exception propagates, counts a failure
  // (note_cublas_handle_destroy_failure). A failed setting still destroys it explicitly first and names that destroy's
  // status in the raise (FIX4), clearing `h` so the owner does not destroy it twice. Nothing is cached on failure, so
  // the next call starts again from cublasCreate.
  // Prior art: RAII (Stroustrup) and the scope guard with dismissal on commit (Alexandrescu & Marginean, "Generic: Change
  // the Way You Write Exception-Safe Code - Forever", C/C++ Users Journal, 2000; std::unique_ptr with a custom deleter
  // is the library form). Taken: own until commit, release on unwinding. Ours: checking and counting the cleanup status.
  struct NewHandleOwner {
    cublasHandle_t& h;
    ~NewHandleOwner() {
      if (h == nullptr) return;  // committed to the cache, or already destroyed by settle
      const cublasStatus_t ds = cublasDestroy(h);
      if (ds != CUBLAS_STATUS_SUCCESS) note_cublas_handle_destroy_failure(static_cast<int>(ds));
    }
  } owned{h};
  // P1T-FIX4 (sweep): a setting that fails no longer leaks the new handle: it is destroyed (on this device, which the
  // caller's DeviceGuard keeps current), the destroy's own status is checked and named, and the call raises. Nothing is
  // cached, so the next call retries from cublasCreate.
  auto settle = [&h, device](cublasStatus_t st, const char* what) {
    if (st == CUBLAS_STATUS_SUCCESS) return;
    const cublasStatus_t ds = cublasDestroy(h);
    h = nullptr;  // P1T-FIX5: destroyed here; the owner must not destroy it again
    std::ostringstream os;
    os << "tc_signal: " << what << " failed: cuBLAS status " << static_cast<int>(st) << " (the new handle of CUDA device "
       << device << ")";  // P1T-FIX7 (sweep (j)): the device named
    if (ds != CUBLAS_STATUS_SUCCESS) os << "; destroying the new handle also failed: cuBLAS status " << static_cast<int>(ds);
    throw std::runtime_error(os.str());
  };
  settle(cublasSetStream(h, kLegacyStream), "cublasSetStream(0)");
  settle(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH), "cublasSetMathMode(CUBLAS_DEFAULT_MATH)");
  settle(cublasSetAtomicsMode(h, CUBLAS_ATOMICS_NOT_ALLOWED), "cublasSetAtomicsMode(NOT_ALLOWED)");
  settle(cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode(HOST)");
  handles.emplace(device, h);  // may throw (host allocation): `owned` destroys h, nothing is cached
  const cublasHandle_t cached = h;
  h = nullptr;  // committed: the cache owns the handle now
  return cached;
}

// ------------------------------------------------------------------------------------------------ kernels
// 1-D im2col for one batch element (Chellapilla et al. 2006, 1-D with dilation and asymmetric zero padding):
//   col[(c*K + k), t] = x[c, t*stride - pad_begin + k*dilation]   (0 outside [0, T))
// Rows (c, k) are c-major, so group g's rows [g*Cg*K, (g+1)*Cg*K) form one contiguous [Cg*K, T_out] block that
// matches w[g*Og:(g+1)*Og] viewed as [Og, Cg*K].
__global__ void im2col1d_kernel(const float* __restrict__ x, float* __restrict__ col, int64_t C, int64_t T, int64_t K,
                                int64_t T_out, int64_t stride, int64_t pad_begin, int64_t dilation) {
  const int64_t total = C * K * T_out;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t t = i % T_out;
    const int64_t r = i / T_out;
    const int64_t k = r % K;
    const int64_t c = r / K;
    const int64_t src = t * stride - pad_begin + k * dilation;
    col[i] = (src >= 0 && src < T) ? x[c * T + src] : 0.0f;
  }
}

// y[n, o, t] += b[o], one thread per output element (fp32 add, one rounding).
__global__ void bias_add_kernel(float* __restrict__ y, const float* __restrict__ b, int64_t C_out, int64_t T_out,
                                int64_t total) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t o = (i / T_out) % C_out;
    y[i] = y[i] + b[o];
  }
}

// Gather-form ConvTranspose1d (adjoint of strided conv, Dumoulin & Visin 2016), one thread per output element:
//   y[n, o, t] = b[o] + sum_{k=0..K-1} sum_{c in group(o)} w[c, o - g*Og, k] * x[n, c, (t + pb - k*dilation)/stride]
// over the taps with (t + pb - k*dilation) >= 0, divisible by stride, and quotient < T. Fixed order: k ascending
// (outer), c ascending within the group (inner), fp32 fmaf accumulation from 0, bias added last. No atomics.
__global__ void conv_transpose1d_gather_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                               const float* __restrict__ b, float* __restrict__ y, int64_t N,
                                               int64_t C_in, int64_t T, int64_t C_out, int64_t K, int64_t T_out,
                                               int64_t stride, int64_t pad_begin, int64_t dilation, int64_t groups) {
  const int64_t Cg = C_in / groups;
  const int64_t Og = C_out / groups;
  const int64_t total = N * C_out * T_out;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t t = i % T_out;
    const int64_t r = i / T_out;
    const int64_t o = r % C_out;
    const int64_t n = r / C_out;
    const int64_t g = o / Og;
    const int64_t ol = o - g * Og;
    const float* xg = x + (n * C_in + g * Cg) * T;   // x[n, g*Cg + cl, :] = xg[cl*T + :]
    const float* wg = w + (g * Cg * Og + ol) * K;    // w[g*Cg + cl, ol, k] = wg[cl*Og*K + k]
    float acc = 0.0f;
    for (int64_t k = 0; k < K; ++k) {
      const int64_t s = t + pad_begin - k * dilation;
      if (s < 0 || s % stride != 0) continue;
      const int64_t q = s / stride;
      if (q >= T) continue;
      const float* xp = xg + q;
      const float* wp = wg + k;
      for (int64_t cl = 0; cl < Cg; ++cl) acc = fmaf(wp[cl * Og * K], xp[cl * T], acc);
    }
    if (b != nullptr) acc = acc + b[o];
    y[i] = acc;
  }
}

// Fast-math canary: out[i] = x[i] / y[i] in fp32. Built with this TU's flags, so --use_fast_math (-ftz=1,
// -prec_div=0) shows up as flushed subnormals or non-IEEE quotients, which the tests compare bitwise against the host's
// IEEE fp32 division. Prior art: no specific source known to me (it is the usual "test the build flags by their
// observable effect" idea; denormal flushing as a fast-math symptom is documented in the CUDA C Programming Guide).
__global__ void fp32_divide_probe_kernel(const float* __restrict__ x, const float* __restrict__ y,
                                         float* __restrict__ out, int64_t n) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x)
    out[i] = x[i] / y[i];
}

}  // namespace

void fp32_divide_probe(int device, const float* x, const float* y, float* out, int64_t n) {
  DeviceGuard guard(device);
  fp32_divide_probe_kernel<<<grid_for(n, tuning::kProbeBlock), tuning::kProbeBlock, 0, kLegacyStream>>>(x, y, out, n);
  check_launch_on(kPrefix, "fp32_divide_probe_kernel", device);
  guard.restore();  // P1T-FIX4: success-path restore is checked (raises on failure)
}

// ------------------------------------------------------------------------------------------------ Conv1d
void conv1d_forward(int device, const Conv1dGeom& g, const float* x, const float* w, const float* b, float* y,
                    float* col) {
  DeviceGuard guard(device);
  cublasHandle_t h = handle_for(device);
  const int64_t Cg = g.C_in / g.groups;
  const int64_t Og = g.C_out / g.groups;
  const int64_t CgK = Cg * g.K;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  for (int64_t n = 0; n < g.N; ++n) {
    const float* xn = x + n * g.C_in * g.T;
    const float* A = nullptr;   // the [Cg*K, T_out] row-major operand of group 0 for this batch element
    long long strideA = 0;
    if (g.direct) {
      // K == 1, stride 1, no pads: x[n] itself is the col matrix (T_out == T), no scratch.
      A = xn;
      strideA = static_cast<long long>(Cg * g.T);
    } else {
      // The scratch is reused for every n: the next im2col is queued after this n's GEMM on the same stream.
      const int64_t total = g.C_in * g.K * g.T_out;
      im2col1d_kernel<<<grid_for(total, tuning::kIm2colBlock), tuning::kIm2colBlock, 0, kLegacyStream>>>(
          xn, col, g.C_in, g.T, g.K, g.T_out, g.stride, g.pad_begin, g.dilation);
      check_launch_on(kPrefix, "im2col1d_kernel", device);
      A = col;
      strideA = static_cast<long long>(CgK * g.T_out);
    }
    float* yn = y + n * g.C_out * g.T_out;
    // Row-major Y_g[Og, T_out] = W_g[Og, CgK] . Col_g[CgK, T_out], computed as the column-major product
    // Y_g^T[T_out, Og] = Col_g^T[T_out, CgK] . W_g^T[CgK, Og]: m = T_out, n = Og, k = CgK, both ops N,
    // lda = T_out (col), ldb = CgK (w), ldc = T_out (y); one batch entry per group.
    check_cublas_on(cublasSgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(g.T_out),
                                           static_cast<int>(Og), static_cast<int>(CgK), &alpha, A,
                                           static_cast<int>(g.T_out), strideA, w, static_cast<int>(CgK),
                                           static_cast<long long>(Og * CgK), &beta, yn, static_cast<int>(g.T_out),
                                           static_cast<long long>(Og * g.T_out), static_cast<int>(g.groups)),
                 "cublasSgemmStridedBatched", device);
    check_after_cublas_on(kPrefix, "cublasSgemmStridedBatched", device);
  }
  if (b != nullptr) {
    const int64_t total = g.N * g.C_out * g.T_out;
    bias_add_kernel<<<grid_for(total, tuning::kBiasBlock), tuning::kBiasBlock, 0, kLegacyStream>>>(y, b, g.C_out,
                                                                                                   g.T_out, total);
    check_launch_on(kPrefix, "bias_add_kernel", device);
  }
  guard.restore();  // P1T-FIX4
}

// ------------------------------------------------------------------------------------------------ ConvTranspose1d
void conv_transpose1d_forward(int device, const ConvTranspose1dGeom& g, const float* x, const float* w,
                              const float* b, float* y) {
  DeviceGuard guard(device);
  const int64_t total = g.N * g.C_out * g.T_out;
  conv_transpose1d_gather_kernel<<<grid_for(total, tuning::kConvTransposeBlock), tuning::kConvTransposeBlock, 0,
                                   kLegacyStream>>>(x, w, b, y, g.N, g.C_in, g.T, g.C_out, g.K, g.T_out, g.stride,
                                                    g.pad_begin, g.dilation, g.groups);
  check_launch_on(kPrefix, "conv_transpose1d_gather_kernel", device);
  guard.restore();  // P1T-FIX4
}

// ------------------------------------------------------------------------------------------------ introspection
CublasConfig cublas_config(int device) {
  DeviceGuard guard(device);
  cublasHandle_t h = handle_for(device);
  cublasMath_t math;
  cublasAtomicsMode_t atomics;
  cublasPointerMode_t pmode;
  cudaStream_t s;
  check_cublas_on(cublasGetMathMode(h, &math), "cublasGetMathMode", device);
  check_cublas_on(cublasGetAtomicsMode(h, &atomics), "cublasGetAtomicsMode", device);
  check_cublas_on(cublasGetPointerMode(h, &pmode), "cublasGetPointerMode", device);
  check_cublas_on(cublasGetStream(h, &s), "cublasGetStream", device);
  CublasConfig c;
  c.math_mode = static_cast<int>(math);
  c.atomics_mode = static_cast<int>(atomics);
  c.pointer_mode = static_cast<int>(pmode);
  c.legacy_stream = (s == kLegacyStream);
  c.default_math = (math == CUBLAS_DEFAULT_MATH);
  c.atomics_not_allowed = (atomics == CUBLAS_ATOMICS_NOT_ALLOWED);
  guard.restore();  // P1T-FIX4
  return c;
}

BuildInfo build_info() {
  BuildInfo bi{};
  check_cuda(cudaRuntimeGetVersion(&bi.cudart_version), "cudaRuntimeGetVersion");
  check_cublas(cublasGetProperty(MAJOR_VERSION, &bi.cublas_major), "cublasGetProperty(MAJOR)");
  check_cublas(cublasGetProperty(MINOR_VERSION, &bi.cublas_minor), "cublasGetProperty(MINOR)");
  check_cublas(cublasGetProperty(PATCH_LEVEL, &bi.cublas_patch), "cublasGetProperty(PATCH)");
  check_cuda(cudaGetDevice(&bi.current_device), "cudaGetDevice");
  return bi;
}

}  // namespace tcs
