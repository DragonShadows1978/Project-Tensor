// APA-Quant attention: custom CUDA kernels.
//
// These kernels cover the APA-specific, fused work that was the dominant
// Python/CuPy overhead in the reference implementation:
//
//   1. quantize_gather   - per-element searchsorted against the Lloyd-Max
//                           boundary table followed by a codebook gather.
//                           Replaces xp.searchsorted + fancy-index gather.
//   2. mix_scores        - fused refinement decision: out = |ranking| >= thr
//                           ? ranking : bulk, while emitting the boolean
//                           refine mask consumed by the backward pass.
//                           Replaces xp.abs + xp.where + mask materialisation.
//
// The heavy matmuls (rotation, Q.Kq, Q.K, P.V and their gradients) are left to
// cuBLAS via ATen matmul -- the same path PyTorch's "math" SDPA backend uses --
// so that an APA-vs-PyTorch benchmark compares algorithms, not BLAS quality.
//
// All kernels are templated on the tensor scalar type (fp32 / fp16) and
// accumulate in float for numerical stability.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>

namespace apa {

namespace {

constexpr int kThreads = 256;

inline int blocks_for(int64_t n) {
  return static_cast<int>((n + kThreads - 1) / kThreads);
}

// std::lower_bound equivalent => numpy searchsorted(side='left').
// Returns the count of boundary entries strictly less than `x`, i.e. the first
// index `i` with boundaries[i] >= x. Result lies in [0, n_boundaries] which is
// exactly a valid index into the codebook of length n_boundaries + 1.
template <typename scalar_t>
__device__ __forceinline__ int searchsorted_left(const scalar_t* __restrict__ boundaries,
                                                  int n_boundaries,
                                                  float x) {
  int lo = 0;
  int hi = n_boundaries;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (static_cast<float>(boundaries[mid]) < x) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

template <typename scalar_t>
__global__ void quantize_gather_kernel(const scalar_t* __restrict__ rotated,
                                       const scalar_t* __restrict__ boundaries,
                                       const scalar_t* __restrict__ codebook,
                                       scalar_t* __restrict__ out,
                                       int n_boundaries,
                                       int64_t numel) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= numel) return;
  float x = static_cast<float>(rotated[idx]);
  int code = searchsorted_left<scalar_t>(boundaries, n_boundaries, x);
  out[idx] = codebook[code];
}

// Fused refinement: for each (b,h,l,s) compare |ranking| against a per-row
// threshold thr[b,h,l]. Refined positions keep the full-precision ranking
// score; the rest fall back to the quantized bulk score. The boolean mask is
// emitted (as uint8) for the backward pass.
template <typename scalar_t>
__global__ void mix_scores_kernel(const scalar_t* __restrict__ ranking,
                                  const scalar_t* __restrict__ bulk,
                                  const float* __restrict__ thr,  // (rows,)
                                  scalar_t* __restrict__ out,
                                  uint8_t* __restrict__ mask,
                                  int64_t S,
                                  int64_t numel) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= numel) return;
  int64_t row = idx / S;
  float r = static_cast<float>(ranking[idx]);
  float t = thr[row];
  bool refine = fabsf(r) >= t;
  out[idx] = refine ? ranking[idx] : bulk[idx];
  mask[idx] = refine ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0);
}

}  // namespace

// rotated: any shape, contiguous, trailing dim already rotated.
// boundaries: (n_boundaries,)  codebook: (n_boundaries + 1,)
at::Tensor quantize_gather(const at::Tensor& rotated,
                           const at::Tensor& boundaries,
                           const at::Tensor& codebook) {
  TORCH_CHECK(rotated.is_cuda(), "rotated must be CUDA");
  TORCH_CHECK(boundaries.is_cuda() && codebook.is_cuda(), "tables must be CUDA");
  TORCH_CHECK(codebook.numel() == boundaries.numel() + 1,
              "codebook length must be n_boundaries + 1");
  auto rotated_c = rotated.contiguous();
  auto boundaries_c = boundaries.contiguous();
  auto codebook_c = codebook.contiguous();
  auto out = at::empty_like(rotated_c);
  const int64_t numel = rotated_c.numel();
  if (numel == 0) return out;

  const at::cuda::OptionalCUDAGuard guard(at::device_of(rotated_c));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int n_boundaries = static_cast<int>(boundaries_c.numel());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      rotated_c.scalar_type(), "quantize_gather", [&] {
        quantize_gather_kernel<scalar_t><<<blocks_for(numel), kThreads, 0, stream>>>(
            rotated_c.data_ptr<scalar_t>(),
            boundaries_c.data_ptr<scalar_t>(),
            codebook_c.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            n_boundaries,
            numel);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// ranking, bulk: (..., S) contiguous and same shape.
// thr: (rows,) float32 where rows = numel / S.
// Returns {mixed_scores (same dtype as ranking), refine_mask (uint8)}.
std::tuple<at::Tensor, at::Tensor> mix_scores(const at::Tensor& ranking,
                                              const at::Tensor& bulk,
                                              const at::Tensor& thr) {
  TORCH_CHECK(ranking.is_cuda() && bulk.is_cuda() && thr.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(ranking.sizes() == bulk.sizes(), "ranking/bulk shape mismatch");
  auto ranking_c = ranking.contiguous();
  auto bulk_c = bulk.contiguous();
  auto thr_c = thr.contiguous().to(at::kFloat);
  const int64_t S = ranking_c.size(-1);
  const int64_t numel = ranking_c.numel();
  const int64_t rows = (S > 0) ? numel / S : 0;
  TORCH_CHECK(thr_c.numel() == rows, "thr must have numel == rows");

  auto out = at::empty_like(ranking_c);
  auto mask = at::empty(ranking_c.sizes(),
                        ranking_c.options().dtype(at::kByte));
  if (numel == 0) return {out, mask};

  const at::cuda::OptionalCUDAGuard guard(at::device_of(ranking_c));
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      ranking_c.scalar_type(), "mix_scores", [&] {
        mix_scores_kernel<scalar_t><<<blocks_for(numel), kThreads, 0, stream>>>(
            ranking_c.data_ptr<scalar_t>(),
            bulk_c.data_ptr<scalar_t>(),
            thr_c.data_ptr<float>(),
            out.data_ptr<scalar_t>(),
            mask.data_ptr<uint8_t>(),
            S,
            numel);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, mask};
}

}  // namespace apa
