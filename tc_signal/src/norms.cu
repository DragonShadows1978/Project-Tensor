// tc_signal T5 norms (P1T-M4): InstanceNormalization, LayerNormalization and GroupNormalization-21 as one row-norm
// kernel. fp32 statistics, biased variance (ONNX), deterministic, no atomic operations, no fast-math flag.
//
// Every norm here is "normalize each contiguous row of a C-contiguous tensor over itself, then apply an affine":
//   instance_norm1d  x [N, C, T]: rows (n, c), L = T,              scale/bias entry c            (NormAffine::kPerRow)
//   layer_norm       x [d0..dr): rows prod(d0..d_{axis-1}), L = prod(d_axis..), entry j (scale/bias are x.shape[axis:])
//   group_norm       x [N, C, T]: rows (n, g), L = (C/G) * T,       entry g*(C/G) + j/T (per channel, GroupNorm-21)
//
// Statistics, per row, fp32 (shifted two-pass):
//   K    = x[row, 0]                                   (the shift: any sample of the row)
//   d    = (sum_j (x_j - K)) / L                       (pass 1: the mean of the shifted row)
//   var  = (sum_j ((x_j - K) - d)^2) / L               (pass 2: the centred sum of squares, biased)
//   rstd = 1.0f / sqrtf(var + eps)                     (math-library sqrtf, IEEE-rounded; IEEE division)
//   y_j  = fmaf(((x_j - K) - d) * rstd, scale, bias)   (pass 3)
// Why the shift: the row mean itself is never formed in fp32. On a row with mean 1e4 and std 1e-2 (the order's
// large-offset cell) the fp32 mean is off by up to half an ulp of 1e4 = 4.9e-4 = 0.05 std, which shifts every output
// by 0.05 and fails the 1e-5 gate whatever the summation order (receipt: artifacts/p1t_m4/sim_large_offset.txt).
// With the shift, x_j - K is exact on such rows (Sterbenz lemma: x_j and K within a factor 2), d is a small number
// computed to full relative precision, and the output error scales with the row's spread, not with its offset.
//
// Reduction order (fixed, so two calls are bitwise equal): one block per row (grid-stride over rows), kNormBlock
// threads. Thread t sums its terms j = t, t + B, t + 2B, ... in ascending order, kNormChunk terms at a time into a partial
// sum that is then added to the thread total (blocked summation), and the B thread totals are combined by a fixed
// shared-memory tree (pairs at stride B/2, B/4, ..., 1). The order depends only on L, kNormBlock and kNormChunk.
//
// rstd choice: 1.0f / sqrtf(v) is two correctly rounded operations (<= ~1 ulp, ~1.2e-7 relative), 80x under the
// order's 1e-5 normwise gate. rsqrtf (<= 2 ulp per the CUDA math documentation) would also pass; the IEEE pair is used
// because the package's fast-math canary (tests/test_contract.py) already witnesses IEEE division and no flushing.
//
// Prior art:
//  * Instance normalization: Ulyanov, Vedaldi & Lempitsky, "Instance Normalization: The Missing Ingredient for Fast
//    Stylization" (2016). Layer normalization: Ba, Kiros & Hinton, "Layer Normalization" (2016). Group normalization:
//    Wu & He, "Group Normalization" (2018). Semantics: ONNX InstanceNormalization, LayerNormalization (axis, epsilon,
//    biased variance), GroupNormalization-21 (per-channel scale/bias, as torch.nn.GroupNorm). Taken: the definitions.
//  * Variance algorithms: Welford (1962) for one-pass updating; Chan, Golub & LeVeque, "Algorithms for computing the
//    sample variance: analysis and recommendations" (1983) for the two-pass algorithm, its error analysis and the
//    "shifted data" variant (shift by an estimate of the mean, e.g. the first sample); Knuth, TAOCP vol. 2 §4.2.2.
//    Taken: two-pass with a shift by the first sample. Ours: the CUDA mapping (one block per row, three passes over
//    global memory, the shift broadcast from x[row, 0]).
//  * Blocked (chunked) summation and its error bound: Higham, "Accuracy and Stability of Numerical Algorithms" (2nd ed.,
//    2002), §4 (summation); Castaldo, Whaley & Chronopoulos, "Reducing floating point error in dot product using the
//    superblock family of algorithms" (2008). Taken: the bound. Ours: the chunk size and its place inside a strided
//    thread loop ahead of a fixed tree.
//  * Deterministic shared-memory tree reduction: the standard CUDA reduction pattern (Harris, "Optimizing Parallel
//    Reduction in CUDA", NVIDIA 2007). Taken: the tree. Ours: nothing beyond fixing B.
//  Unverified in this sandbox (lead to check): exact titles/years above; search terms "Chan Golub LeVeque 1983 sample
//  variance shifted", "Welford 1962 corrected sums of squares", "Castaldo Whaley superblock 2008".
#include <cuda_runtime.h>

#include <algorithm>
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

// Sum of every thread's `v` over the block, by a fixed tree (stride B/2, B/4, ..., 1); every thread gets the total.
template <int B>
__device__ __forceinline__ float block_tree_sum(float v, float* red) {
  red[threadIdx.x] = v;
  __syncthreads();
#pragma unroll
  for (int s = B / 2; s > 0; s >>= 1) {
    if (static_cast<int>(threadIdx.x) < s) red[threadIdx.x] = red[threadIdx.x] + red[threadIdx.x + s];
    __syncthreads();
  }
  const float total = red[0];
  __syncthreads();  // every thread has read red[0] before the next call overwrites red
  return total;
}

template <int B>
__global__ void __launch_bounds__(B) row_norm_kernel(const float* __restrict__ x, const float* __restrict__ scale,
                                                     const float* __restrict__ bias, float* __restrict__ y,
                                                     int64_t rows, int64_t L, float eps, int affine, int64_t channels,
                                                     int64_t groups, int64_t ch_per_group, int64_t T) {
  __shared__ float red[B];
  const int64_t tid = threadIdx.x;
  const int64_t chunk_span = static_cast<int64_t>(B) * m4tuning::kNormChunk;
  const float Lf = static_cast<float>(L);  // exact for L < 2^24
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const float* __restrict__ xr = x + row * L;
    float* __restrict__ yr = y + row * L;
    const float K = xr[0];

    // Pass 1: the mean of the shifted row.
    float acc = 0.0f;
    for (int64_t base = tid; base < L; base += chunk_span) {
      const int64_t end = min(L, base + chunk_span);
      float part = 0.0f;
      for (int64_t j = base; j < end; j += B) part = part + (xr[j] - K);
      acc = acc + part;
    }
    const float d = block_tree_sum<B>(acc, red) / Lf;

    // Pass 2: the centred sum of squares (biased variance).
    acc = 0.0f;
    for (int64_t base = tid; base < L; base += chunk_span) {
      const int64_t end = min(L, base + chunk_span);
      float part = 0.0f;
      for (int64_t j = base; j < end; j += B) {
        const float c = (xr[j] - K) - d;
        part = fmaf(c, c, part);
      }
      acc = acc + part;
    }
    const float var = block_tree_sum<B>(acc, red) / Lf;
    const float rstd = 1.0f / sqrtf(var + eps);

    // Pass 3: normalize and apply the affine.
    for (int64_t j = tid; j < L; j += B) {
      const float c = (xr[j] - K) - d;
      int64_t p;
      if (affine == static_cast<int>(NormAffine::kPerRow))
        p = row % channels;
      else if (affine == static_cast<int>(NormAffine::kPerElem))
        p = j;
      else
        p = (row % groups) * ch_per_group + j / T;
      const float s = scale != nullptr ? scale[p] : 1.0f;
      const float b = bias != nullptr ? bias[p] : 0.0f;
      yr[j] = fmaf(c * rstd, s, b);
    }
  }
}

}  // namespace

void row_norm_forward(int device, const RowNormGeom& g, const float* x, const float* scale, const float* bias,
                      float* y) {
  DeviceGuard guard(device);
  constexpr int B = m4tuning::kNormBlock;
  const unsigned int grid = static_cast<unsigned int>(std::max<int64_t>(1, std::min<int64_t>(g.rows, tuning::kGridCap)));
  row_norm_kernel<B><<<grid, B, 0, kLegacyStream>>>(x, scale, bias, y, g.rows, g.L, g.eps, static_cast<int>(g.affine),
                                                    g.channels, g.groups, g.ch_per_group, g.T);
  check_cuda(cudaGetLastError(), "row_norm_kernel");
}

}  // namespace tcs
