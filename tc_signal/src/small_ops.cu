// tc_signal T6 small ops (P1T-M4): nearest upsample x k (1-D), reflection pad (1-D), rint. Bit-exact by construction:
// the two gathers move 32-bit words (one load and one store per output element, through unsigned-integer pointers, so
// NaN payloads, signed zeros and subnormals are copied bit for bit), and rint is the math library's rintf (IEEE 754
// roundTiesToEven in the default rounding mode, sign of zero kept, NaN and +-Inf pass through). No fast-math flag, no
// atomic operations: each output element is written by exactly one thread. TensorCUDA core's `round` (half away from
// zero, used by its quantizers) is untouched; rint is a new op here.
//
// Semantics (the shipped Kokoro nodes):
//  * upsample_nearest1d: ONNX Resize mode=nearest, coordinate_transformation_mode=asymmetric, nearest_mode=floor,
//    scales [1, ..., 1, s] with s an integer >= 1: y[..., i] = x[..., floor(i / s)], i in [0, T*s). Integer division is
//    the floor for these non-negative operands, so no float coordinate is ever formed.
//  * reflection_pad1d: ONNX Pad mode=reflect on the last axis (= torch reflection_pad1d): with s = i - pad_begin,
//    y[..., i] = x[..., |s|] for s < 0, x[..., 2(T-1) - s] for s >= T, else x[..., s]. pad_begin, pad_end < T.
//  * rint: ONNX Round (half to even).
//
// Prior art: ONNX operator specs (Resize, Pad, Round); IEEE 754-2008 roundTiesToEven / C99 rint; torch's
// reflection_pad1d and nearest upsample for the same index maps. Ours: nothing beyond the one-thread-per-output CUDA
// mapping (no prior art known to me for anything else here).
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>

#include "m4_ops.h"
#include "tuning.h"

#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
#error "tc_signal queues on the legacy default stream (stream 0); do not build it with --default-stream per-thread"
#endif

namespace tcs {

namespace {

constexpr cudaStream_t kLegacyStream = 0;  // the legacy default stream (TensorCUDA's only stream)

void check_cuda(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    std::ostringstream os;
    os << "tc_signal: " << what << " failed: " << cudaGetErrorName(e) << ": " << cudaGetErrorString(e);
    throw std::runtime_error(os.str());
  }
}

unsigned int grid_for(int64_t total, int block) {
  const int64_t blocks = (total + block - 1) / block;
  return static_cast<unsigned int>(std::max<int64_t>(1, std::min<int64_t>(blocks, tuning::kGridCap)));
}

// Makes `device` current for the call and restores the caller's device afterwards.
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

// y[r, i] = x[r, i / scale]: one thread per output element, a 32-bit word copy.
__global__ void upsample_nearest1d_kernel(const uint32_t* __restrict__ x, uint32_t* __restrict__ y, int64_t total,
                                          int64_t T, int64_t T_out, int64_t scale) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t t = i % T_out;
    const int64_t r = i / T_out;
    y[i] = x[r * T + t / scale];
  }
}

// Reflect pad of the last axis: one thread per output element, a 32-bit word copy.
__global__ void reflection_pad1d_kernel(const uint32_t* __restrict__ x, uint32_t* __restrict__ y, int64_t total,
                                        int64_t T, int64_t T_out, int64_t pad_begin) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t t = i % T_out;
    const int64_t r = i / T_out;
    int64_t s = t - pad_begin;
    if (s < 0)
      s = -s;
    else if (s >= T)
      s = 2 * (T - 1) - s;
    y[i] = x[r * T + s];
  }
}

// y = rintf(x): round half to even (the default rounding mode), sign of zero kept, NaN/Inf pass through.
__global__ void rint_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x)
    y[i] = rintf(x[i]);
}

}  // namespace

void upsample_nearest1d_forward(int device, int64_t rows, int64_t T, int64_t scale, const float* x, float* y) {
  DeviceGuard guard(device);
  const int64_t T_out = T * scale;
  const int64_t total = rows * T_out;
  constexpr int B = m4tuning::kCopyBlock;
  upsample_nearest1d_kernel<<<grid_for(total, B), B, 0, kLegacyStream>>>(
      reinterpret_cast<const uint32_t*>(x), reinterpret_cast<uint32_t*>(y), total, T, T_out, scale);
  check_cuda(cudaGetLastError(), "upsample_nearest1d_kernel");
}

void reflection_pad1d_forward(int device, int64_t rows, int64_t T, int64_t pad_begin, int64_t pad_end, const float* x,
                              float* y) {
  DeviceGuard guard(device);
  const int64_t T_out = T + pad_begin + pad_end;
  const int64_t total = rows * T_out;
  constexpr int B = m4tuning::kCopyBlock;
  reflection_pad1d_kernel<<<grid_for(total, B), B, 0, kLegacyStream>>>(
      reinterpret_cast<const uint32_t*>(x), reinterpret_cast<uint32_t*>(y), total, T, T_out, pad_begin);
  check_cuda(cudaGetLastError(), "reflection_pad1d_kernel");
}

void rint_forward(int device, int64_t n, const float* x, float* y) {
  DeviceGuard guard(device);
  constexpr int B = m4tuning::kCopyBlock;
  rint_kernel<<<grid_for(n, B), B, 0, kLegacyStream>>>(x, y, n);
  check_cuda(cudaGetLastError(), "rint_kernel");
}

}  // namespace tcs
