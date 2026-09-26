#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

// Prior art: NVIDIA CUB stable radix sort / segmented reduction (CUDA 12.6,
// 2024), and PyTorch deterministic index_add (1.9, 2021): group equal keys
// and give each output a fixed reduction order. Demmel & Nguyen, "Fast
// Reproducible Floating-Point Summation" (2013), motivates reproducibility;
// this is NOT their order-independent accumulator. Ours: stable position
// order, FP64 local sum, one final FP32 rounding, shared CPU/CUDA arithmetic.
// https://github.com/NVIDIA/cccl/blob/v2.5.0/cub/cub/device/device_radix_sort.cuh
// https://pytorch.org/blog/pytorch-1-9-released/
// https://www.acsel-lab.com/arithmetic/arith21/papers/p54.pdf
#ifdef __CUDACC__
#define TC_DET_HD __host__ __device__
#else
#define TC_DET_HD
#endif

namespace tc::det_embed {
template <class T>
TC_DET_HD inline float segment_sum(const T* grad, const int64_t* positions,
                                 int64_t first, int64_t last,
                                 int64_t width, int64_t feature) {
  double sum = 0.;
  for (int64_t p = first; p < last; ++p)
    sum += static_cast<double>(static_cast<float>(grad[positions[p] * width + feature]));
  return static_cast<float>(sum);
}

inline std::vector<int64_t> sorted_positions(const int64_t* ids, int64_t n, int64_t vocab) {
  if (n < 0 || vocab <= 0) throw std::runtime_error("det embedding: invalid dimensions");
  for (int64_t i = 0; i < n; ++i)
    if (ids[i] < 0 || ids[i] >= vocab) throw std::runtime_error("det embedding: token out of range");
  std::vector<int64_t> positions(n);
  std::iota(positions.begin(), positions.end(), 0);
  std::stable_sort(positions.begin(), positions.end(),
                   [ids](int64_t a, int64_t b) { return ids[a] < ids[b]; });
  return positions;
}

inline void backward_cpu(const float* grad, const int64_t* ids, float* out,
                         int64_t n, int64_t vocab, int64_t width) {
  if (width <= 0) throw std::runtime_error("det embedding: invalid width");
  auto positions = sorted_positions(ids, n, vocab);
  std::fill(out, out + vocab * width, 0.f);
  for (int64_t first = 0; first < n;) {
    int64_t last = first + 1, token = ids[positions[first]];
    while (last < n && ids[positions[last]] == token) ++last;
    for (int64_t feature = 0; feature < width; ++feature)
      out[token * width + feature] = segment_sum(grad, positions.data(), first, last, width, feature);
    first = last;
  }
}
}  // namespace tc::det_embed
#undef TC_DET_HD
