#pragma once

#include "tc/core.h"
#include "tc/deterministic_embed.h"
#include <climits>
#include <limits>

// Prior art: CUB/Merrill/NVIDIA stable radix sort and sorted segmented
// scatter-add (2011 onward; CUDA 12.6, 2024); PyTorch deterministic indexing
// (1.9, 2021). Taken: group equal destinations, one fixed-order owner per
// segment. Ours: contiguous n-D gather destination mapping and integration
// with the PT-DET-1 (2026) shared FP64 sum. Demmel/Nguyen (ARITH 2013) is
// reproducibility context only; this is not an order-independent accumulator.
// https://pytorch.org/blog/pytorch-1-9-released/
// https://www.acsel-lab.com/arithmetic/arith21/papers/p54.pdf
// CUB stability: installed cub/device/device_radix_sort.cuh, "Stability".
#ifdef __CUDACC__
#define TC_GATHER_HD __host__ __device__
#else
#define TC_GATHER_HD
#endif

namespace tc::det_gather {
struct Spec {
  int ndim, dim;
  int64_t index_strides[TC_MAX_DIMS], output_strides[TC_MAX_DIMS];
  int64_t axis_size, count, output_count;
};

inline Spec make_spec(const Shape& shape, const Shape& index_shape, int dim) {
  const int nd = static_cast<int>(shape.size());
  if (nd < 1 || nd > TC_MAX_DIMS || index_shape.size() != shape.size())
    throw std::runtime_error("det gather: invalid rank");
  if (dim < 0) dim += nd;
  if (dim < 0 || dim >= nd) throw std::runtime_error("det gather: invalid dimension");
  Spec s{};
  s.ndim = nd; s.dim = dim; s.axis_size = shape[dim];
  s.count = 1; s.output_count = 1;
  for (int d = nd - 1; d >= 0; --d) {
    if (shape[d] < 0 || index_shape[d] < 0 || (d != dim && index_shape[d] > shape[d]))
      throw std::runtime_error("det gather: invalid shape");
    s.index_strides[d] = s.count;
    s.output_strides[d] = s.output_count;
    if ((index_shape[d] && s.count > INT_MAX / index_shape[d]) ||
        (shape[d] && s.output_count > std::numeric_limits<int64_t>::max() / shape[d]))
      throw std::runtime_error("det gather: unsupported dimensions");
    s.count *= index_shape[d]; s.output_count *= shape[d];
  }
  // Output/temporary byte counts must fit signed engine allocation arithmetic.
  if (s.output_count > std::numeric_limits<int64_t>::max() / int64_t(sizeof(float)))
    throw std::runtime_error("det gather: output too large");
  return s;
}

TC_GATHER_HD inline int64_t destination(int64_t position, int64_t index, const Spec& s) {
  if (index < 0 || index >= s.axis_size) return -1;
  int64_t rem = position, offset = 0;
  for (int d = 0; d < s.ndim; ++d) {
    const int64_t coordinate = rem / s.index_strides[d];
    rem -= coordinate * s.index_strides[d];
    offset += (d == s.dim ? index : coordinate) * s.output_strides[d];
  }
  return offset;
}

inline void backward_cpu(const float* grad, const int64_t* index, float* out, const Spec& s) {
  if (s.output_count) std::fill(out, out + s.output_count, 0.f);
  if (!s.count) return;
  std::vector<int64_t> keys(s.count);
  for (int64_t p = 0; p < s.count; ++p) {
    keys[p] = destination(p, index[p], s);
    if (keys[p] < 0) throw std::runtime_error("det gather: index out of range");
  }
  auto positions = det_embed::sorted_positions(keys.data(), s.count, s.output_count);
  for (int64_t first = 0; first < s.count;) {
    const int64_t key = keys[positions[first]];
    int64_t last = first + 1;
    while (last < s.count && keys[positions[last]] == key) ++last;
    out[key] = det_embed::segment_sum(grad, positions.data(), first, last, 1, 0);
    first = last;
  }
}
}  // namespace tc::det_gather
#undef TC_GATHER_HD
