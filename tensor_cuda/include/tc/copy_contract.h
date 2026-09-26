// PT-RING-1. Prior art: CUDA 12.6 (NVIDIA, 2024) byte-copy contracts;
// std::lower_bound (C++98) interval lookup. Taken: checked byte ranges and
// sorted interval validation. Ours: this engine's batch alias policy.
#pragma once
#include "tc/core.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace tc::ring {
inline size_t checked_bytes(const Shape& shape, DType dtype) {
  if (shape.size() > TC_MAX_DIMS) throw std::invalid_argument("copy: rank exceeds TC_MAX_DIMS");
  size_t item = 0;
  switch (dtype) {
    case DType::Float32: item = 4; break;
    case DType::Float16: case DType::BFloat16: item = 2; break;
    case DType::Int64: item = 8; break;
    case DType::Bool: case DType::Uint8: item = 1; break;
    default: throw std::invalid_argument("copy: invalid dtype");
  }
  bool empty = false;
  for (auto dim : shape) {
    if (dim < 0) throw std::invalid_argument("copy: negative dimension");
    empty |= dim == 0;
  }
  size_t count = 1;
  // Check even empty shapes: engine contiguous_strides uses int64 products.
  for (auto dim : shape) {
    const auto factor = static_cast<size_t>(dim ? dim : 1);
    if (count > static_cast<size_t>(INT64_MAX) / item / factor)
      throw std::overflow_error("copy: shape byte count overflow");
    count *= factor;
  }
  return empty ? 0 : count * item;
}

struct Range { uintptr_t begin, end; size_t pair; };
inline Range range(const void* ptr, size_t bytes, size_t pair) {
  const auto begin = reinterpret_cast<uintptr_t>(ptr);
  if (bytes && (!ptr || begin > UINTPTR_MAX - bytes))
    throw std::invalid_argument("copy: invalid address range");
  return {begin, begin + bytes, pair};
}

inline void validate_aliases(std::vector<Range> writes, const std::vector<Range>& reads) {
  writes.erase(std::remove_if(writes.begin(), writes.end(),
                             [](const Range& r) { return r.begin == r.end; }), writes.end());
  std::sort(writes.begin(), writes.end(), [](const Range& a, const Range& b) {
    return a.begin < b.begin;
  });
  for (size_t i = 1; i < writes.size(); ++i)
    if (writes[i].begin < writes[i-1].end)
      throw std::invalid_argument("copy: overlapping destinations");
  for (const auto& read : reads) {
    if (read.begin == read.end) continue;
    auto it = std::lower_bound(writes.begin(), writes.end(), read.begin,
                              [](const Range& w, uintptr_t b) { return w.end <= b; });
    for (; it != writes.end() && it->begin < read.end; ++it)
      if (!(it->pair == read.pair && it->begin == read.begin && it->end == read.end))
        throw std::invalid_argument("copy: overlapping source/destination batch");
  }
}
}  // namespace tc::ring
