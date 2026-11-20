#pragma once

#include <cstdint>
#include <utility>

namespace tcs {

struct ConvBackwardGeom {
  int64_t batch, cin, cout, input_time, output_time, taps;
  int64_t stride, pad_begin, output_padding, dilation, groups;
  bool transposed;
};

std::pair<int64_t, int64_t> conv_backward_scratch(const ConvBackwardGeom& g);
void conv_backward(int device, const ConvBackwardGeom& g, const float* x, const float* w, const float* dy,
                   float* dx, float* dw, float* db, float* col, float* part);

struct ConvBackwardConfig {
  bool legacy, pedantic_math, pedantic_compute, no_atomics, host_pointers, default_algo;
  int cublas_version, sm_count;
  int64_t workspace_bytes, tile, scratch_cap_bytes;
};
ConvBackwardConfig conv_backward_config(int device);

}  // namespace tcs
