// Strict first-order fp32 convolution adjoints, on the DLPack legacy stream.
#include "conv_backward.h"
#include "cuda_guard.h"
#include "tuning.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <memory>
#include <stdexcept>
#include <unordered_map>

namespace tcs {
namespace {

constexpr char prefix[] = "tc_signal: convolution backward: ";
constexpr int block = tuning::kConvBackwardBlock;
constexpr int tile = tuning::kConvBackwardTile;
constexpr int64_t scratch_cap = tuning::kConvBackwardScratchBytes;
constexpr int64_t workspace_size = tuning::kConvBackwardWorkspaceBytes;

int grid(int64_t count, int width = block) {
  return static_cast<int>(std::min<int64_t>(tuning::kGridCap, 1 + (count - 1) / width));
}

void check_blas(cublasStatus_t status, const char* what, int device) {
  if (status != CUBLAS_STATUS_SUCCESS)
    raise_cublas_failure_on(prefix, what, device, static_cast<int>(status));
}

// cuBLAS 12.6 (NVIDIA, 2024) reproducibility, explicit workspace and strict
// compute. BWD-R1 (2026) precedent: the call and check read the same members.
// R3 owns a separate handle and workspace per device on the legacy stream.
class StrictGemm {
 public:
  explicit StrictGemm(int device) : device_(device) {}

  void initialize() {
    check_blas(cublasCreate(&handle_), "cublasCreate", device_);
    check_blas(cublasSetStream(handle_, cudaStreamLegacy), "cublasSetStream", device_);
    check_blas(cublasSetMathMode(handle_, CUBLAS_PEDANTIC_MATH), "cublasSetMathMode", device_);
    check_blas(cublasSetAtomicsMode(handle_, CUBLAS_ATOMICS_NOT_ALLOWED), "cublasSetAtomicsMode", device_);
    check_blas(cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode", device_);
    check_call_on(prefix, "allocate private workspace", "cudaMalloc", device_, cudaMalloc(&workspace_, workspace_size));
    check_blas(cublasSetWorkspace(handle_, workspace_, workspace_size), "cublasSetWorkspace", device_);
    installed_bytes_ = workspace_size;
  }

  // Used on failed initialization or cache insertion, with device still current.
  ~StrictGemm() {
    if (handle_) {
      const auto status = cublasDestroy(handle_);
      if (status != CUBLAS_STATUS_SUCCESS) note_cublas_handle_destroy_failure(static_cast<int>(status));
    }
    if (workspace_) cudaFree(workspace_);
  }

  ConvBackwardConfig config(bool diagnostics = false) const {
    cudaStream_t stream;
    cublasMath_t math;
    cublasAtomicsMode_t atomics;
    cublasPointerMode_t pointer;
    check_blas(cublasGetStream(handle_, &stream), "cublasGetStream", device_);
    check_blas(cublasGetMathMode(handle_, &math), "cublasGetMathMode", device_);
    check_blas(cublasGetAtomicsMode(handle_, &atomics), "cublasGetAtomicsMode", device_);
    check_blas(cublasGetPointerMode(handle_, &pointer), "cublasGetPointerMode", device_);
    ConvBackwardConfig result{stream == cudaStreamLegacy, math == CUBLAS_PEDANTIC_MATH,
                              compute_ == CUBLAS_COMPUTE_32F_PEDANTIC, atomics == CUBLAS_ATOMICS_NOT_ALLOWED,
                              pointer == CUBLAS_POINTER_MODE_HOST, algo_ == CUBLAS_GEMM_DEFAULT,
                              0, 0, installed_bytes_, tile, scratch_cap};
    if (diagnostics) {
      check_blas(cublasGetVersion(handle_, &result.cublas_version), "cublasGetVersion", device_);
      check_call_on(prefix, "query SM count", "cudaDeviceGetAttribute", device_,
                    cudaDeviceGetAttribute(&result.sm_count, cudaDevAttrMultiProcessorCount, device_));
    }
    return result;
  }

  void verify() const {
    // cuBLAS resets the installed workspace on cublasSetStream and offers
    // no workspace getter. Only initialize() sets this handle's stream,
    // before installing the private workspace recorded by installed_bytes_.
    const auto cfg = config();
    if (!(cfg.legacy && cfg.pedantic_math && cfg.pedantic_compute && cfg.no_atomics &&
          cfg.host_pointers && cfg.default_algo && workspace_ && cfg.workspace_bytes == workspace_size))
      throw std::runtime_error("tc_signal: convolution backward: strict fp32 GEMM configuration mismatch");
  }

  void run(int columns, int rows, int count, int left_stride,
           const float* packed, const float* left, float* partial) const {
    const float alpha = 1.0f, beta = 0.0f;
    // Row-major left @ packed.T becomes column-major packed.T @ left.
    check_blas(cublasGemmEx(handle_, CUBLAS_OP_T, CUBLAS_OP_N, columns, rows, count,
                           &alpha, packed, CUDA_R_32F, count, left, CUDA_R_32F, left_stride,
                           &beta, partial, CUDA_R_32F, columns, compute_, algo_), "cublasGemmEx", device_);
    check_after_cublas_on(prefix, "cublasGemmEx", device_);
  }

 private:
  int device_;
  const cublasComputeType_t compute_ = CUBLAS_COMPUTE_32F_PEDANTIC;
  const cublasGemmAlgo_t algo_ = CUBLAS_GEMM_DEFAULT;
  cublasHandle_t handle_ = nullptr;
  void* workspace_ = nullptr;
  int64_t installed_bytes_ = 0;
};

StrictGemm& handle_for(int device) {
  // Bindings hold the GIL. As in tc_signal forward, retain initialized CUDA
  // state until process teardown to avoid ordering races with CUDA shutdown.
  static std::unordered_map<int, StrictGemm*> handles;
  const auto found = handles.find(device);
  if (found != handles.end()) return *found->second;
  auto handle = std::make_unique<StrictGemm>(device);
  handle->initialize();
  auto* result = handle.get();
  handles.emplace(device, result);
  handle.release();
  return *result;
}

__device__ int64_t connected_time(const ConvBackwardGeom& g, int64_t time, int64_t tap) {
  return time * g.stride - g.pad_begin + tap * g.dilation;
}

__device__ bool in_range(int64_t t, int64_t length) { return t >= 0 && t < length; }

// Dumoulin and Visin (2016): transpose the strided correlation index.
// R3 gathers in tap/channel order with ONNX grouped weight indexing.
__global__ void input_grad(ConvBackwardGeom g, const float* w, const float* dy, float* dx) {
  const int64_t ci = g.cin / g.groups, co = g.cout / g.groups;
  const int64_t size = g.batch * g.cin * g.input_time;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    const int64_t q = i % g.input_time;
    const int64_t c = i / g.input_time % g.cin;
    const int64_t n = i / g.input_time / g.cin;
    const int64_t group = c / ci;
    float sum = 0.0f;
    for (int64_t k = 0; k < g.taps; ++k) {
      int64_t t;
      if (g.transposed) {
        t = connected_time(g, q, k);
      } else {
        const int64_t pos = q + g.pad_begin - k * g.dilation;
        if (pos < 0 || pos % g.stride != 0) continue;
        t = pos / g.stride;
      }
      if (!in_range(t, g.output_time)) continue;
      for (int64_t o = 0; o < co; ++o) {
        const int64_t wi = g.transposed ? (c * co + o) * g.taps + k
                                       : ((group * co + o) * ci + c % ci) * g.taps + k;
        const int64_t gi = (n * g.cout + group * co + o) * g.output_time + t;
        sum = fmaf(w[wi], dy[gi], sum);
      }
    }
    dx[i] = sum;
  }
}

// Chellapilla, Puri and Simard (2006): im2col paired with GEMM. R3 packs
// only one group of x (Conv) or dy (ConvTranspose) for each time tile.
__global__ void pack(ConvBackwardGeom g, const float* source, float* col,
                     int64_t rows, int64_t begin, int count, int64_t limit) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < rows * count; i += gridDim.x * blockDim.x) {
    const int64_t q = connected_time(g, begin + i % count, i / count % g.taps);
    const int64_t c = i / count / g.taps;
    col[i] = in_range(q, limit) ? source[c * limit + q] : 0.0f;
  }
}

__global__ void merge_partial(const float* part, float* dw, int64_t size, bool first) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x)
    dw[i] = first ? part[i] : dw[i] + part[i];
}

// Mark Harris (2007), sequential-addressed CUDA reduction. R3 fixes the
// lane partition and tree width and reuses the result per coefficient/tile.
__device__ float reduce_block(float value, float* shared) {
  shared[threadIdx.x] = value;
  __syncthreads();
  for (int offset = block / 2; offset; offset /= 2) {
    if (threadIdx.x < offset) shared[threadIdx.x] += shared[threadIdx.x + offset];
    __syncthreads();
  }
  const float sum = shared[0];
  __syncthreads();
  return sum;
}

__global__ void weight_tree(ConvBackwardGeom g, const float* x, const float* dy, float* dw,
                            int64_t n, int64_t begin, int count, bool first) {
  __shared__ float scratch[block];
  const int64_t ci = g.cin / g.groups, co = g.cout / g.groups;
  const int64_t size = g.groups * ci * co * g.taps;
  for (int64_t i = blockIdx.x; i < size; i += gridDim.x) {
    const int64_t k = i % g.taps;
    const int64_t remaining = i / g.taps;
    const int64_t c = g.transposed ? remaining / co : remaining / ci / co * ci + remaining % ci;
    const int64_t o = g.transposed ? c / ci * co + remaining % co : remaining / ci;
    const float* xr = x + (n * g.cin + c) * g.input_time;
    const float* gr = dy + (n * g.cout + o) * g.output_time;
    float sum = 0.0f;
    for (int j = threadIdx.x; j < count; j += block) {
      const int64_t time = begin + j;
      const int64_t mapped = connected_time(g, time, k);
      if (g.transposed) {
        if (in_range(mapped, g.output_time)) sum = fmaf(xr[time], gr[mapped], sum);
      } else if (in_range(mapped, g.input_time)) {
        sum = fmaf(xr[mapped], gr[time], sum);
      }
    }
    sum = reduce_block(sum, scratch);
    if (threadIdx.x == 0) dw[i] = first ? sum : dw[i] + sum;
  }
}

__global__ void bias_grad(ConvBackwardGeom g, const float* dy, float* db) {
  __shared__ float scratch[block];
  const int64_t size = g.batch * g.output_time;
  for (int64_t o = blockIdx.x; o < g.cout; o += gridDim.x) {
    float sum = 0.0f;
    for (int64_t i = threadIdx.x; i < size; i += block)
      sum += dy[(i / g.output_time * g.cout + o) * g.output_time + i % g.output_time];
    sum = reduce_block(sum, scratch);
    if (threadIdx.x == 0) db[o] = sum;
  }
}

}  // namespace

std::pair<int64_t, int64_t> conv_backward_scratch(const ConvBackwardGeom& g) {
  const int64_t ci = g.cin / g.groups, co = g.cout / g.groups;
  const int64_t size = ci * co * g.taps;
  const int64_t length = g.transposed ? g.input_time : g.output_time;
  const int64_t rows = (g.transposed ? co : ci) * g.taps;
  const int64_t count = std::min<int64_t>(length, tile);
  constexpr int64_t cap = scratch_cap / sizeof(float);
  if (size <= tuning::kConvBackwardTreeCoefficients || size > cap ||
      rows > cap / count || length > INT_MAX) return {0, 0};
  return {rows * count, size};
}

void conv_backward(int device, const ConvBackwardGeom& g, const float* x, const float* w, const float* dy,
                   float* dx, float* dw, float* db, float* col, float* part) {
  DeviceGuard guard(device);
  if (dx) {
    input_grad<<<grid(g.batch * g.cin * g.input_time), block, 0, cudaStreamLegacy>>>(g, w, dy, dx);
    check_launch_on(prefix, "input_grad", device);
  }
  if (db) {
    bias_grad<<<grid(g.cout, 1), block, 0, cudaStreamLegacy>>>(g, dy, db);
    check_launch_on(prefix, "bias_grad", device);
  }
  if (dw) {
    auto* gemm = col ? &handle_for(device) : nullptr;
    if (gemm) gemm->verify();
    const int64_t left_rows = (g.transposed ? g.cin : g.cout) / g.groups;
    const int64_t columns = (g.transposed ? g.cout : g.cin) / g.groups * g.taps;
    const int64_t length = g.transposed ? g.input_time : g.output_time;
    const int64_t limit = g.transposed ? g.output_time : g.input_time;
    const int64_t size = left_rows * columns;
    const int64_t tiles = 1 + (length - 1) / tile;
    // DESIGN §5: one owned rectangle at a time. Merge in (n,time) order
    // before overwriting it, and use the same order in the fixed tree lane.
    for (int64_t n = 0; n < g.batch; ++n) {
      for (int64_t step = 0; step < tiles; ++step) {
        const int64_t begin = step * tile;
        const int count = static_cast<int>(std::min<int64_t>(tile, length - begin));
        const bool first = n == 0 && step == 0;
        if (!gemm) {
          weight_tree<<<grid(size * g.groups, 1), block, 0, cudaStreamLegacy>>>(g, x, dy, dw, n, begin, count, first);
          check_launch_on(prefix, "weight_tree", device);
          continue;
        }
        for (int64_t group = 0; group < g.groups; ++group) {
          const float* source = (g.transposed ? dy : x) + (n * g.groups + group) * (columns / g.taps) * limit;
          const float* left = (g.transposed ? x : dy) + (n * g.groups + group) * left_rows * length + begin;
          pack<<<grid(columns * count), block, 0, cudaStreamLegacy>>>(g, source, col, columns, begin, count, limit);
          check_launch_on(prefix, "pack", device);
          gemm->run(columns, left_rows, count, length, col, left, part);
          merge_partial<<<grid(size), block, 0, cudaStreamLegacy>>>(part, dw + group * size, size, first);
          check_launch_on(prefix, "merge_partial", device);
        }
      }
    }
  }
  guard.restore();
}

ConvBackwardConfig conv_backward_config(int device) {
  DeviceGuard guard(device);
  const auto cfg = handle_for(device).config(true);
  guard.restore();
  return cfg;
}

}  // namespace tcs
