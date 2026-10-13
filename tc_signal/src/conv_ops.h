// Host launchers for tc_signal's 1-D convolutions. Raw device pointers in, everything queued on the legacy default
// stream; no device memory is allocated here (the im2col scratch is passed in; cuBLAS's own internal workspace is the
// one exception, see README "cuBLAS").
#pragma once

#include <cstdint>

namespace tcs {

struct Conv1dGeom {
  int64_t N, C_in, T;      // x [N, C_in, T]
  int64_t C_out, K;        // w [C_out, C_in/groups, K]
  int64_t stride, pad_begin, pad_end, dilation, groups;
  int64_t T_out;           // (T + pb + pe - dilation*(K-1) - 1)//stride + 1
  bool direct;             // K == 1, stride 1, no pads: GEMM on x itself, no im2col, no scratch
};

struct ConvTranspose1dGeom {
  int64_t N, C_in, T;      // x [N, C_in, T]
  int64_t C_out, K;        // w [C_in, C_out/groups, K] (ONNX layout)
  int64_t stride, pad_begin, pad_end, output_padding, dilation, groups;
  int64_t T_out;           // stride*(T-1) + output_padding + dilation*(K-1) + 1 - pb - pe
};

// y = conv1d(x, w) + b. `col` holds C_in*K*T_out floats (one batch element at a time), or is NULL when g.direct.
void conv1d_forward(int device, const Conv1dGeom& g, const float* x, const float* w, const float* b, float* y,
                    float* col);

// y = conv_transpose1d(x, w) + b (gather form, one thread per output element).
void conv_transpose1d_forward(int device, const ConvTranspose1dGeom& g, const float* x, const float* w,
                              const float* b, float* y);

// Fast-math canary (see conv.cu): out[i] = x[i] / y[i], fp32, on the legacy stream.
void fp32_divide_probe(int device, const float* x, const float* y, float* out, int64_t n);

struct CublasConfig {
  int math_mode;           // cublasMath_t
  int atomics_mode;        // cublasAtomicsMode_t
  int pointer_mode;        // cublasPointerMode_t
  bool legacy_stream;      // the handle's stream is 0 (the legacy default stream)
  bool default_math;       // math_mode == CUBLAS_DEFAULT_MATH
  bool atomics_not_allowed;
};
// The (lazily created) handle's settings on `device`.
CublasConfig cublas_config(int device);

struct BuildInfo {
  int cudart_version;      // cudaRuntimeGetVersion
  int cublas_major, cublas_minor, cublas_patch;
  int current_device;
};
BuildInfo build_info();

}  // namespace tcs
