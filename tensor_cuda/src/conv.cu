// conv.cu: im2col/col2im and 2D pooling kernels (NCHW).
//
// Conv2D is built in Python as im2col -> matmul -> reshape; autograd flows
// through these as a differentiable im2col op (col2im is its backward). Pooling
// has dedicated forward/backward kernels. Backward scatter-adds accumulate in
// fp32 then cast to the array dtype (matches the embedding-grad approach).

#include "tc/core.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>

namespace tc {
namespace {

template <typename T> __device__ __forceinline__ float ldf(const T* p, int64_t i);
template <> __device__ __forceinline__ float ldf<float>(const float* p, int64_t i) { return p[i]; }
template <> __device__ __forceinline__ float ldf<__half>(const __half* p, int64_t i) { return __half2float(p[i]); }
template <typename T> __device__ __forceinline__ void stf(T* p, int64_t i, float v);
template <> __device__ __forceinline__ void stf<float>(float* p, int64_t i, float v) { p[i] = v; }
template <> __device__ __forceinline__ void stf<__half>(__half* p, int64_t i, float v) { p[i] = __float2half(v); }

constexpr int kT = 256;
inline int nblk(int64_t n) { return (int)((n + kT - 1) / kT); }

struct ConvDims { int N, C, H, W, kh, kw, sh, sw, ph, pw, OH, OW; };

template <typename T>
__global__ void im2col_kernel(const T* x, T* cols, ConvDims d, int64_t n) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  int64_t L = (int64_t)d.OH * d.OW, K = (int64_t)d.C * d.kh * d.kw;
  int64_t l = idx % L, k = (idx / L) % K, bn = idx / (L * K);
  int oh = (int)(l / d.OW), ow = (int)(l % d.OW);
  int c = (int)(k / (d.kh * d.kw)), rem = (int)(k % (d.kh * d.kw));
  int ki = rem / d.kw, kj = rem % d.kw;
  int ih = oh * d.sh - d.ph + ki, iw = ow * d.sw - d.pw + kj;
  float v = 0.f;
  if (ih >= 0 && ih < d.H && iw >= 0 && iw < d.W)
    v = ldf<T>(x, ((bn * d.C + c) * d.H + ih) * d.W + iw);
  stf<T>(cols, idx, v);
}

template <typename T>
__global__ void col2im_kernel(const T* cols, float* xg, ConvDims d, int64_t n) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  int64_t L = (int64_t)d.OH * d.OW, K = (int64_t)d.C * d.kh * d.kw;
  int64_t l = idx % L, k = (idx / L) % K, bn = idx / (L * K);
  int oh = (int)(l / d.OW), ow = (int)(l % d.OW);
  int c = (int)(k / (d.kh * d.kw)), rem = (int)(k % (d.kh * d.kw));
  int ki = rem / d.kw, kj = rem % d.kw;
  int ih = oh * d.sh - d.ph + ki, iw = ow * d.sw - d.pw + kj;
  if (ih >= 0 && ih < d.H && iw >= 0 && iw < d.W)
    atomicAdd(&xg[((bn * d.C + c) * d.H + ih) * d.W + iw], ldf<T>(cols, idx));
}

template <typename T>
__global__ void avgpool_kernel(const T* x, T* out, ConvDims d, int64_t n) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  int64_t L = (int64_t)d.OH * d.OW;
  int ow = (int)(idx % d.OW), oh = (int)((idx / d.OW) % d.OH);
  int64_t c = (idx / L) % d.C, bn = idx / (L * d.C);
  float acc = 0.f;
  for (int ki = 0; ki < d.kh; ++ki)
    for (int kj = 0; kj < d.kw; ++kj) {
      int ih = oh * d.sh - d.ph + ki, iw = ow * d.sw - d.pw + kj;
      if (ih >= 0 && ih < d.H && iw >= 0 && iw < d.W)
        acc += ldf<T>(x, ((bn * d.C + c) * d.H + ih) * d.W + iw);
    }
  stf<T>(out, idx, acc / (d.kh * d.kw));
}

template <typename T>
__global__ void avgpool_bwd_kernel(const T* g, float* xg, ConvDims d, int64_t n) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  int64_t L = (int64_t)d.OH * d.OW;
  int ow = (int)(idx % d.OW), oh = (int)((idx / d.OW) % d.OH);
  int64_t c = (idx / L) % d.C, bn = idx / (L * d.C);
  float gv = ldf<T>(g, idx) / (d.kh * d.kw);
  for (int ki = 0; ki < d.kh; ++ki)
    for (int kj = 0; kj < d.kw; ++kj) {
      int ih = oh * d.sh - d.ph + ki, iw = ow * d.sw - d.pw + kj;
      if (ih >= 0 && ih < d.H && iw >= 0 && iw < d.W)
        atomicAdd(&xg[((bn * d.C + c) * d.H + ih) * d.W + iw], gv);
    }
}

template <typename T>
__global__ void maxpool_kernel(const T* x, T* out, int64_t* argmax, ConvDims d, int64_t n) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  int64_t L = (int64_t)d.OH * d.OW;
  int ow = (int)(idx % d.OW), oh = (int)((idx / d.OW) % d.OH);
  int64_t c = (idx / L) % d.C, bn = idx / (L * d.C);
  float best = -3.4e38f; int64_t bi = -1;
  for (int ki = 0; ki < d.kh; ++ki)
    for (int kj = 0; kj < d.kw; ++kj) {
      int ih = oh * d.sh - d.ph + ki, iw = ow * d.sw - d.pw + kj;
      if (ih >= 0 && ih < d.H && iw >= 0 && iw < d.W) {
        int64_t off = ((bn * d.C + c) * d.H + ih) * d.W + iw;
        float v = ldf<T>(x, off);
        if (v > best) { best = v; bi = off; }
      }
    }
  stf<T>(out, idx, best);
  argmax[idx] = bi;
}

template <typename T>
__global__ void maxpool_bwd_kernel(const T* g, const int64_t* argmax, float* xg, int64_t n) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  int64_t bi = argmax[idx];
  if (bi >= 0) atomicAdd(&xg[bi], ldf<T>(g, idx));
}

#define DISP(DT, T, ...) do { \
  if ((DT) == DType::Float32) { using T = float; __VA_ARGS__; } \
  else if ((DT) == DType::Float16) { using T = __half; __VA_ARGS__; } \
  else throw std::runtime_error("conv/pool: float32/float16 only"); } while (0)

ConvDims make_dims(const Shape& xs, int kh, int kw, int sh, int sw, int ph, int pw) {
  ConvDims d;
  d.N = (int)xs[0]; d.C = (int)xs[1]; d.H = (int)xs[2]; d.W = (int)xs[3];
  d.kh = kh; d.kw = kw; d.sh = sh; d.sw = sw; d.ph = ph; d.pw = pw;
  d.OH = (d.H + 2 * ph - kh) / sh + 1;
  d.OW = (d.W + 2 * pw - kw) / sw + 1;
  return d;
}
}  // namespace

NDArray im2col(const NDArray& x, int kh, int kw, int sh, int sw, int ph, int pw) {
  ConvDims d = make_dims(x.shape, kh, kw, sh, sw, ph, pw);
  Shape os = {(int64_t)d.N, (int64_t)d.C * kh * kw, (int64_t)d.OH * d.OW};
  NDArray out(os, x.dtype, x.device);
  int64_t n = out.numel();
  if (n) { DISP(x.dtype, T, { im2col_kernel<T><<<nblk(n), kT>>>(static_cast<T*>(x.data_ptr()), static_cast<T*>(out.data_ptr()), d, n); }); cuda_check_last("im2col"); }
  return out;
}
NDArray col2im(const NDArray& cols, const Shape& x_shape, int kh, int kw, int sh, int sw, int ph, int pw) {
  ConvDims d = make_dims(x_shape, kh, kw, sh, sw, ph, pw);
  NDArray xgf = NDArray::zeros(x_shape, DType::Float32, cols.device);
  int64_t n = cols.numel();
  if (n) { DISP(cols.dtype, T, { col2im_kernel<T><<<nblk(n), kT>>>(static_cast<T*>(cols.data_ptr()), static_cast<float*>(xgf.data_ptr()), d, n); }); cuda_check_last("col2im"); }
  return xgf.astype(cols.dtype);
}
NDArray avgpool2d(const NDArray& x, int kh, int kw, int sh, int sw, int ph, int pw) {
  ConvDims d = make_dims(x.shape, kh, kw, sh, sw, ph, pw);
  NDArray out({(int64_t)d.N, (int64_t)d.C, (int64_t)d.OH, (int64_t)d.OW}, x.dtype, x.device);
  int64_t n = out.numel();
  if (n) { DISP(x.dtype, T, { avgpool_kernel<T><<<nblk(n), kT>>>(static_cast<T*>(x.data_ptr()), static_cast<T*>(out.data_ptr()), d, n); }); cuda_check_last("avgpool"); }
  return out;
}
NDArray avgpool2d_bwd(const NDArray& g, const Shape& x_shape, int kh, int kw, int sh, int sw, int ph, int pw) {
  ConvDims d = make_dims(x_shape, kh, kw, sh, sw, ph, pw);
  NDArray xgf = NDArray::zeros(x_shape, DType::Float32, g.device);
  int64_t n = g.numel();
  if (n) { DISP(g.dtype, T, { avgpool_bwd_kernel<T><<<nblk(n), kT>>>(static_cast<T*>(g.data_ptr()), static_cast<float*>(xgf.data_ptr()), d, n); }); cuda_check_last("avgpool_bwd"); }
  return xgf.astype(g.dtype);
}
NDArray maxpool2d(const NDArray& x, int kh, int kw, int sh, int sw, int ph, int pw, NDArray& argmax_out) {
  ConvDims d = make_dims(x.shape, kh, kw, sh, sw, ph, pw);
  Shape os = {(int64_t)d.N, (int64_t)d.C, (int64_t)d.OH, (int64_t)d.OW};
  NDArray out(os, x.dtype, x.device);
  argmax_out = NDArray(os, DType::Int64, x.device);
  int64_t n = out.numel();
  if (n) { DISP(x.dtype, T, { maxpool_kernel<T><<<nblk(n), kT>>>(static_cast<T*>(x.data_ptr()), static_cast<T*>(out.data_ptr()), static_cast<int64_t*>(argmax_out.data_ptr()), d, n); }); cuda_check_last("maxpool"); }
  return out;
}
NDArray maxpool2d_bwd(const NDArray& g, const NDArray& argmax, const Shape& x_shape) {
  NDArray xgf = NDArray::zeros(x_shape, DType::Float32, g.device);
  int64_t n = g.numel();
  if (n) { DISP(g.dtype, T, { maxpool_bwd_kernel<T><<<nblk(n), kT>>>(static_cast<T*>(g.data_ptr()), static_cast<int64_t*>(argmax.data_ptr()), static_cast<float*>(xgf.data_ptr()), n); }); cuda_check_last("maxpool_bwd"); }
  return xgf.astype(g.dtype);
}

}  // namespace tc
