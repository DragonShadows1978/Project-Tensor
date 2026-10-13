// tc_signal LSTM (P1T-M3, T4): ONNX LSTM semantics, fp32, deterministic, no atomics, no fast-math.
//
// Semantics (ONNX LSTM operator, layout 0, default activations Sigmoid/Tanh/Tanh, no clip, no peepholes,
// input_forget 0): x [T, N, I]; W [D, 4H, I]; R [D, 4H, H]; B [D, 8H] = [Wb; Rb]; gate blocks along 4H in the ONNX order
// i, o, f, c (NOT torch's i, f, g, o). Per direction slot d and step:
//     z  = (x_t . W[d]^T + Wb[d]) + h_{t-1} . R[d]^T + Rb[d]
//     i  = sigmoid(z_i), o = sigmoid(z_o), f = sigmoid(z_f), c~ = tanh(z_c)
//     c  = f * c_{t-1} + i * c~,  h = o * tanh(c)
// D = 2 (bidirectional): slot 0 runs forward, slot 1 runs in reverse; a reverse slot at loop step s reads time
// T-1-s and writes Y[T-1-s, d]. Y_h / Y_c are the state after each slot's last step.
//
// Structure (Project-Voice TENSOR_IMPL_PLAN_P1T.md §5 T4; LD51 amendment A2; order P1T-M3 "Structure"):
//  1. initial state -> Y_h / Y_c (D2D copy or zero), on the legacy stream;
//  2. one cublasSgemm per direction for the input projection of ALL steps: G_x[d] = X[T*N, I] . W[d]^T, accumulated
//     onto G_x pre-filled with Wb[d] (beta = 1; beta = 0 without B);
//  3. a host C++ loop over the T steps (inside the one _tc_signal call; no Python per step, no per-step DLPack):
//     one cublasSgemmStridedBatched over the D slots, G_h[d] = h_{t-1}[d] . R[d]^T (h = Y_h), then one fused cell
//     kernel (adds Rb, activations, c/h update, writes Y_c, Y_h and Y).
//
// Numerics contract (TENSOR_IMPL_PLAN_P1T.md §4):
//  * fp32 everywhere; cuBLAS in CUBLAS_DEFAULT_MATH (never TF32), atomics not allowed, pointer mode host;
//  * transcendentals are the CUDA math library's accurate expf() and tanhf() (documented max error 2 ulp each); none of
//    the double-underscore fast intrinsics (exp, tanh, divide) and no fast-math compiler flag (CMakeLists.txt refuses it
//    for the whole package);
//  * every element of every kernel is written by exactly one thread with a fixed operation order, and cuBLAS runs with
//    atomics not allowed on one stream, so two calls on the same inputs are bitwise equal. No atomic adds anywhere.
//
// Prior art:
//  * LSTM: Hochreiter & Schmidhuber, "Long Short-Term Memory" (1997); the forget gate: Gers, Schmidhuber & Cummins,
//    "Learning to Forget: Continual Prediction with LSTM" (2000). Taken: the cell equations.
//  * ONNX LSTM operator specification (onnx/docs/Operators.md, LSTM): gate order i,o,f,c, B = [Wb; Rb], the Y / Y_h /
//    Y_c layouts, the reverse direction. Taken: the semantics, verbatim.
//  * Input projection hoisted out of the recurrence as one large GEMM over all time steps, leaving only the h . R^T
//    product inside the step loop: Appleyard, Kocisky & Blunsom, "Optimizing Performance of Recurrent Neural Networks
//    on GPUs" (2016), the design cuDNN's RNN API uses. Taken: that split. Ours: the plain correctness-first form (one
//    strided-batched GEMM + one fused cell kernel per step, host loop), with no persistent kernel and no stream overlap.
//  * Numerically stable logistic sigmoid by splitting on the sign of z (evaluate exp of a non-positive argument only):
//    standard practice (e.g. the log-sum-exp / softplus folklore; no single source known to me). Taken: the split.
//  * Row-major GEMM through column-major BLAS by computing C^T = B^T A^T: standard BLAS practice (no single source).
// The recurrent state kept in the Y_h / Y_c outputs (no separate state scratch) is ours; no prior art known to me.
#include "lstm_ops.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "lstm_tuning.h"

#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
#error "tc_signal queues on the legacy default stream (stream 0); do not build it with --default-stream per-thread"
#endif

namespace tcs {

namespace {

// Stream 0 is the legacy default stream in this translation unit (per-thread default stream refused above): the stream
// TensorCUDA queues on and the stream DLPack producers were asked to order against (stream=1).
constexpr cudaStream_t kLegacyStream = 0;

// ONNX LSTM gate block indices along the 4H axis (Operators.md, LSTM: "iofc").
constexpr int64_t kGateI = 0;
constexpr int64_t kGateO = 1;
constexpr int64_t kGateF = 2;
constexpr int64_t kGateC = 3;

void check_cuda(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    std::ostringstream os;
    os << "tc_signal.lstm: " << what << " failed: " << cudaGetErrorName(e) << ": " << cudaGetErrorString(e);
    throw std::runtime_error(os.str());
  }
}

// Checked after every kernel launch and cuBLAS call (launch-configuration and sticky errors surface as RuntimeError).
void check_launch(const char* what) { check_cuda(cudaGetLastError(), what); }

void check_cublas(cublasStatus_t s, const char* what) {
  if (s != CUBLAS_STATUS_SUCCESS) {
    std::ostringstream os;
    os << "tc_signal.lstm: " << what << " failed: cuBLAS status " << static_cast<int>(s);
    throw std::runtime_error(os.str());
  }
}

unsigned int grid_for(int64_t total, int block) {
  const int64_t blocks = (total + block - 1) / block;
  return static_cast<unsigned int>(std::max<int64_t>(1, std::min<int64_t>(blocks, lstm_tuning::kGridCap)));
}

// Makes `device` current for the call and restores the caller's device afterwards. (M2's DeviceGuard in conv.cu has
// internal linkage and the order forbids editing conv.cu, so this is a copy.)
class DeviceGuard {
 public:
  explicit DeviceGuard(int device) {
    check_cuda(cudaGetDevice(&prev_), "cudaGetDevice");
    if (prev_ != device) {
      check_cuda(cudaSetDevice(device), "cudaSetDevice");
      changed_ = true;
    }
  }
  ~DeviceGuard() {
    if (changed_) cudaSetDevice(prev_);
  }
  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;

 private:
  int prev_ = 0;
  bool changed_ = false;
};

// cuBLAS handle: one per device, created lazily on first use and kept for the life of the process (destroying it at
// interpreter exit can race CUDA's own teardown), with M2's exact settings: legacy stream, CUBLAS_DEFAULT_MATH (never
// TF32), atomics not allowed, pointer mode host. M2's handle_for() in conv.cu has internal linkage and the order
// forbids editing conv.cu, so the LSTM keeps its own handle (deviation, P1T_M3_REPORT.md). All entry points run under
// the GIL, so no extra lock is needed. cuBLAS allocates its own internal workspace for this handle (cudaMalloc inside
// cuBLAS); that is the only device memory the LSTM causes to be allocated outside the caller's allocator.
cublasHandle_t lstm_handle_for(int device) {
  static std::unordered_map<int, cublasHandle_t> handles;
  auto it = handles.find(device);
  if (it != handles.end()) return it->second;
  cublasHandle_t h = nullptr;
  check_cublas(cublasCreate(&h), "cublasCreate");
  check_cublas(cublasSetStream(h, kLegacyStream), "cublasSetStream(0)");
  check_cublas(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH), "cublasSetMathMode(CUBLAS_DEFAULT_MATH)");
  check_cublas(cublasSetAtomicsMode(h, CUBLAS_ATOMICS_NOT_ALLOWED), "cublasSetAtomicsMode(NOT_ALLOWED)");
  check_cublas(cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode(HOST)");
  handles.emplace(device, h);
  return h;
}

// ------------------------------------------------------------------------------------------------ device functions
// Logistic sigmoid, stable for large |z|: exp is only ever evaluated at a non-positive argument, so it never overflows
// (z >= 0: 1 / (1 + e^{-z}); z < 0: e^{z} / (1 + e^{z})). expf is the CUDA math library's accurate expf (2 ulp max),
// never the fast intrinsic. Prior art: the standard sign split (see the file header).
__device__ __forceinline__ float sigmoid_stable(float z) {
  if (z >= 0.0f) return 1.0f / (1.0f + expf(-z));
  const float e = expf(z);
  return e / (1.0f + e);
}

// ------------------------------------------------------------------------------------------------ kernels
// G_x[d, r, q] = Wb[d, q] for every projected row r in [0, T*N) and q in [0, 4H): the bias the input-projection GEMM
// then accumulates onto (beta = 1). One thread per element, a plain copy.
__global__ void lstm_bias_fill_kernel(float* __restrict__ gx, const float* __restrict__ B, int64_t D, int64_t rows,
                                      int64_t H4, int64_t bias_row) {
  const int64_t total = D * rows * H4;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t q = i % H4;
    const int64_t d = i / (rows * H4);
    gx[i] = B[d * bias_row + q];
  }
}

// One LSTM step for every direction slot at once; one thread per state element idx = (d*N + n)*H + j.
//   gx: G_x [D, T, N, 4H] (x . W^T + Wb); gh: G_h [D, N, 4H] (h_{t-1} . R^T for this step); B: [D, 8H] or NULL;
//   h / c: the state [D, N, H] (= Y_h / Y_c), read and written in place by the owning thread only; Y [T, D, N, H].
// Fixed per-element operation order: z = (gx + gh) + Rb; c = fmaf(f, c_prev, i * c~); h = o * tanhf(c).
__global__ void lstm_cell_kernel(const float* __restrict__ gx, const float* __restrict__ gh,
                                 const float* __restrict__ B, float* __restrict__ h, float* __restrict__ c,
                                 float* __restrict__ Y, int64_t T, int64_t s, int64_t D, int64_t N, int64_t H,
                                 int64_t first_reverse) {
  const int64_t H4 = kLstmGates * H;
  const int64_t total = D * N * H;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t j = idx % H;
    const int64_t r = idx / H;
    const int64_t n = r % N;
    const int64_t d = r / N;
    const int64_t tt = (d >= first_reverse) ? (T - 1 - s) : s;   // the reverse slot walks time backwards
    const float* gxp = gx + ((d * T + tt) * N + n) * H4;
    const float* ghp = gh + (d * N + n) * H4;
    float zi = gxp[kGateI * H + j] + ghp[kGateI * H + j];
    float zo = gxp[kGateO * H + j] + ghp[kGateO * H + j];
    float zf = gxp[kGateF * H + j] + ghp[kGateF * H + j];
    float zc = gxp[kGateC * H + j] + ghp[kGateC * H + j];
    if (B != nullptr) {
      const float* rb = B + d * (kLstmBiasHalves * H4) + H4;     // Rb[d] = B[d, 4H:8H]
      zi = zi + rb[kGateI * H + j];
      zo = zo + rb[kGateO * H + j];
      zf = zf + rb[kGateF * H + j];
      zc = zc + rb[kGateC * H + j];
    }
    const float ig = sigmoid_stable(zi);
    const float og = sigmoid_stable(zo);
    const float fg = sigmoid_stable(zf);
    const float cg = tanhf(zc);
    const float cn = fmaf(fg, c[idx], ig * cg);
    const float hn = og * tanhf(cn);
    c[idx] = cn;
    h[idx] = hn;
    Y[((tt * D + d) * N + n) * H + j] = hn;
  }
}

}  // namespace

// ------------------------------------------------------------------------------------------------ LSTM forward
void lstm_forward(int device, const LstmGeom& g, const float* x, const float* W, const float* R, const float* B,
                  const float* h0, const float* c0, float* Y, float* Y_h, float* Y_c, float* scratch) {
  DeviceGuard guard(device);
  cublasHandle_t hdl = lstm_handle_for(device);
  const int64_t H4 = kLstmGates * g.H;
  const int64_t rows = g.T * g.N;                 // projected rows, r = t*N + n
  const int64_t state = g.D * g.N * g.H;          // elements of Y_h / Y_c
  const size_t state_bytes = static_cast<size_t>(state) * sizeof(float);
  float* gx = scratch;                            // [D, T, N, 4H]
  float* gh = scratch + lstm_gx_numel(g);         // [D, N, 4H]
  const float one = 1.0f;
  const float zero = 0.0f;

  // 1. Initial state into the outputs that carry the recurrence.
  if (h0 != nullptr)
    check_cuda(cudaMemcpyAsync(Y_h, h0, state_bytes, cudaMemcpyDeviceToDevice, kLegacyStream), "initial_h copy");
  else
    check_cuda(cudaMemsetAsync(Y_h, 0, state_bytes, kLegacyStream), "initial_h zero");
  if (c0 != nullptr)
    check_cuda(cudaMemcpyAsync(Y_c, c0, state_bytes, cudaMemcpyDeviceToDevice, kLegacyStream), "initial_c copy");
  else
    check_cuda(cudaMemsetAsync(Y_c, 0, state_bytes, kLegacyStream), "initial_c zero");

  // 2. Input projection of all steps, one SGEMM per direction (+ Wb through beta = 1 on the pre-filled buffer).
  //    Row-major G_x[d][rows, 4H] = X[rows, I] . W[d][4H, I]^T, as the column-major product
  //    G_x^T[4H, rows] = op_T(W_d viewed [I, 4H], ld I) . X^T[I, rows] (ld I): m = 4H, n = rows, k = I, ldc = 4H.
  float beta_x = zero;
  if (B != nullptr) {
    const int64_t total = g.D * rows * H4;
    lstm_bias_fill_kernel<<<grid_for(total, lstm_tuning::kBiasFillBlock), lstm_tuning::kBiasFillBlock, 0,
                            kLegacyStream>>>(gx, B, g.D, rows, H4, kLstmBiasHalves * H4);
    check_launch("lstm_bias_fill_kernel");
    beta_x = one;
  }
  for (int64_t d = 0; d < g.D; ++d) {
    check_cublas(cublasSgemm(hdl, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(H4), static_cast<int>(rows),
                             static_cast<int>(g.I), &one, W + d * H4 * g.I, static_cast<int>(g.I), x,
                             static_cast<int>(g.I), &beta_x, gx + d * rows * H4, static_cast<int>(H4)),
                 "cublasSgemm (input projection)");
    check_launch("cublasSgemm (input projection)");
  }

  // 3. The recurrence: a host loop over steps inside this one call. Per step, one strided-batched SGEMM over the D
  //    slots, G_h[d][N, 4H] = h[d][N, H] . R[d][4H, H]^T (column-major: m = 4H, n = N, k = H, op_T on R viewed
  //    [H, 4H] ld H, h viewed [H, N] ld H, ldc = 4H), then the fused cell kernel. Stream order makes step s+1's GEMM
  //    read the h that step s's cell kernel wrote.
  const unsigned int cell_grid = grid_for(state, lstm_tuning::kCellBlock);
  for (int64_t s = 0; s < g.T; ++s) {
    check_cublas(cublasSgemmStridedBatched(hdl, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(H4), static_cast<int>(g.N),
                                           static_cast<int>(g.H), &one, R, static_cast<int>(g.H),
                                           static_cast<long long>(H4 * g.H), Y_h, static_cast<int>(g.H),
                                           static_cast<long long>(g.N * g.H), &zero, gh, static_cast<int>(H4),
                                           static_cast<long long>(g.N * H4), static_cast<int>(g.D)),
                 "cublasSgemmStridedBatched (recurrent)");
    check_launch("cublasSgemmStridedBatched (recurrent)");
    lstm_cell_kernel<<<cell_grid, lstm_tuning::kCellBlock, 0, kLegacyStream>>>(gx, gh, B, Y_h, Y_c, Y, g.T, s, g.D,
                                                                                 g.N, g.H, g.first_reverse);
    check_launch("lstm_cell_kernel");
  }
}

// ------------------------------------------------------------------------------------------------ introspection
LstmCublasConfig lstm_cublas_config(int device) {
  DeviceGuard guard(device);
  cublasHandle_t h = lstm_handle_for(device);
  cublasMath_t math;
  cublasAtomicsMode_t atomics;
  cublasPointerMode_t pmode;
  cudaStream_t s;
  check_cublas(cublasGetMathMode(h, &math), "cublasGetMathMode");
  check_cublas(cublasGetAtomicsMode(h, &atomics), "cublasGetAtomicsMode");
  check_cublas(cublasGetPointerMode(h, &pmode), "cublasGetPointerMode");
  check_cublas(cublasGetStream(h, &s), "cublasGetStream");
  LstmCublasConfig c;
  c.math_mode = static_cast<int>(math);
  c.atomics_mode = static_cast<int>(atomics);
  c.pointer_mode = static_cast<int>(pmode);
  c.legacy_stream = (s == kLegacyStream);
  c.default_math = (math == CUBLAS_DEFAULT_MATH);
  c.atomics_not_allowed = (atomics == CUBLAS_ATOMICS_NOT_ALLOWED);
  return c;
}

}  // namespace tcs
