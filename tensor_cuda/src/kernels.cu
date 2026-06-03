// kernels.cu: device memory, NDArray, and hand-written CUDA kernels.
//
// Kernels are dtype-generic: values are loaded and computed in fp32 and stored
// back in the array's dtype, so fp32 and fp16 share one implementation.

#include "tc/core.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace tc {

// ------------------------------------------------------------ dtype / device
size_t dtype_size(DType dt) {
  switch (dt) {
    case DType::Float32: return 4;
    case DType::Float16: return 2;
    case DType::Int64: return 8;
    case DType::Bool: return 1;
  }
  return 4;
}
const char* dtype_name(DType dt) {
  switch (dt) {
    case DType::Float32: return "float32";
    case DType::Float16: return "float16";
    case DType::Int64: return "int64";
    case DType::Bool: return "bool";
  }
  return "float32";
}
DType dtype_from_string(const std::string& s) {
  if (s == "float32" || s == "float" || s == "f32") return DType::Float32;
  if (s == "float16" || s == "half" || s == "f16") return DType::Float16;
  if (s == "int64" || s == "long") return DType::Int64;
  if (s == "bool") return DType::Bool;
  throw std::runtime_error("unknown dtype: " + s);
}

std::string Device::str() const {
  return is_cuda() ? ("cuda:" + std::to_string(index)) : "cpu";
}
Device Device::from_string(const std::string& s) {
  Device d;
  if (s.rfind("cpu", 0) == 0) { d.type = DeviceType::CPU; d.index = 0; }
  else { d.type = DeviceType::CUDA; d.index = 0; }
  return d;
}

int64_t numel_of(const Shape& shape) {
  int64_t n = 1;
  for (int64_t s : shape) n *= s;
  return shape.empty() ? 1 : n;
}
Shape contiguous_strides(const Shape& shape) {
  Shape st(shape.size());
  int64_t acc = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    st[i] = acc;
    acc *= shape[i];
  }
  return st;
}

void cuda_check_last(const char* where) {
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error at ") + where + ": " +
                             cudaGetErrorString(e));
  }
}
void cuda_sync() {
  cudaDeviceSynchronize();
  cuda_check_last("sync");
}

// --------------------------------------------------------------------- Storage
Storage::Storage(size_t nbytes_, Device device_) : nbytes(nbytes_), device(device_) {
  if (nbytes == 0) { ptr = nullptr; return; }
  if (device.is_cuda()) {
    cudaError_t e = cudaMalloc(&ptr, nbytes);
    if (e != cudaSuccess)
      throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(e));
  } else {
    ptr = std::malloc(nbytes);
  }
}
Storage::~Storage() {
  if (!ptr) return;
  if (device.is_cuda()) cudaFree(ptr);
  else std::free(ptr);
}

// --------------------------------------------------------------------- NDArray
NDArray::NDArray(const Shape& shape_, DType dtype_, Device device_)
    : shape(shape_), dtype(dtype_), device(device_) {
  storage = std::make_shared<Storage>(numel_of(shape_) * dtype_size(dtype_), device_);
}
NDArray NDArray::empty(const Shape& shape, DType dtype, Device device) {
  return NDArray(shape, dtype, device);
}
void* NDArray::data_ptr() const {
  return static_cast<char*>(storage->ptr) + offset * dtype_size(dtype);
}

namespace {
template <typename T> __device__ __forceinline__ float ld(const T* p, int64_t i);
template <> __device__ __forceinline__ float ld<float>(const float* p, int64_t i) { return p[i]; }
template <> __device__ __forceinline__ float ld<__half>(const __half* p, int64_t i) { return __half2float(p[i]); }
template <typename T> __device__ __forceinline__ void st(T* p, int64_t i, float v);
template <> __device__ __forceinline__ void st<float>(float* p, int64_t i, float v) { p[i] = v; }
template <> __device__ __forceinline__ void st<__half>(__half* p, int64_t i, float v) { p[i] = __float2half(v); }

constexpr int kT = 256;
inline int nblk(int64_t n) { return static_cast<int>((n + kT - 1) / kT); }

template <typename T>
__global__ void fill_kernel(T* p, int64_t n, float v) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) st<T>(p, i, v);
}
template <typename SrcT, typename DstT>
__global__ void cast_kernel(const SrcT* s, DstT* d, int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) st<DstT>(d, i, ld<SrcT>(s, i));
}
}  // namespace

#define DISPATCH_FLOAT(DT, T, ...)                                       \
  do {                                                                   \
    if ((DT) == DType::Float32) { using T = float; __VA_ARGS__; }        \
    else if ((DT) == DType::Float16) { using T = __half; __VA_ARGS__; }  \
    else throw std::runtime_error("op supports float32/float16 only");   \
  } while (0)

NDArray NDArray::zeros(const Shape& shape, DType dtype, Device device) {
  NDArray a(shape, dtype, device);
  if (a.numel() > 0) cudaMemset(a.data_ptr(), 0, a.numel() * dtype_size(dtype));
  return a;
}
NDArray NDArray::full(const Shape& shape, double value, DType dtype, Device device) {
  NDArray a(shape, dtype, device);
  int64_t n = a.numel();
  if (n == 0) return a;
  DISPATCH_FLOAT(dtype, T, {
    fill_kernel<T><<<nblk(n), kT>>>(static_cast<T*>(a.data_ptr()), n, (float)value);
  });
  cuda_check_last("full");
  return a;
}
NDArray NDArray::from_host(const void* src, const Shape& shape, DType dtype, Device device) {
  NDArray a(shape, dtype, device);
  size_t bytes = a.numel() * dtype_size(dtype);
  if (bytes == 0) return a;
  if (device.is_cuda()) cudaMemcpy(a.data_ptr(), src, bytes, cudaMemcpyHostToDevice);
  else std::memcpy(a.data_ptr(), src, bytes);
  cuda_check_last("from_host");
  return a;
}
void NDArray::to_host(void* dst) const {
  size_t bytes = numel() * dtype_size(dtype);
  if (bytes == 0) return;
  if (device.is_cuda()) cudaMemcpy(dst, data_ptr(), bytes, cudaMemcpyDeviceToHost);
  else std::memcpy(dst, data_ptr(), bytes);
  cuda_check_last("to_host");
}
NDArray NDArray::clone() const {
  NDArray out(shape, dtype, device);
  size_t bytes = numel() * dtype_size(dtype);
  if (bytes) cudaMemcpy(out.data_ptr(), data_ptr(), bytes, cudaMemcpyDeviceToDevice);
  return out;
}
NDArray NDArray::to(Device dev) const {
  if (dev.is_cuda() == device.is_cuda()) return clone();
  NDArray out(shape, dtype, dev);
  size_t bytes = numel() * dtype_size(dtype);
  cudaMemcpyKind k = device.is_cuda() ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
  if (bytes) cudaMemcpy(out.data_ptr(), data_ptr(), bytes, k);
  return out;
}
NDArray NDArray::astype(DType dt) const {
  if (dt == dtype) return clone();
  NDArray out(shape, dt, device);
  int64_t n = numel();
  if (n == 0) return out;
  DISPATCH_FLOAT(dtype, ST, {
    DISPATCH_FLOAT(dt, DT, {
      cast_kernel<ST, DT><<<nblk(n), kT>>>(static_cast<ST*>(data_ptr()),
                                           static_cast<DT*>(out.data_ptr()), n);
    });
  });
  cuda_check_last("astype");
  return out;
}
NDArray NDArray::reshape(const Shape& new_shape) const {
  NDArray out = *this;  // shares storage
  out.shape = new_shape;
  return out;
}

// ------------------------------------------------------------- broadcasting
namespace {
struct DimSpec {
  int ndim;
  int64_t out_shape[TC_MAX_DIMS];
  int64_t out_str[TC_MAX_DIMS];   // contiguous strides of output
  int64_t a_str[TC_MAX_DIMS];     // broadcast strides into a (0 = broadcast)
  int64_t b_str[TC_MAX_DIMS];
};

Shape broadcast_shape(const Shape& a, const Shape& b) {
  int na = (int)a.size(), nb = (int)b.size(), n = na > nb ? na : nb;
  Shape out(n);
  for (int i = 0; i < n; ++i) {
    int64_t ad = (i < n - na) ? 1 : a[i - (n - na)];
    int64_t bd = (i < n - nb) ? 1 : b[i - (n - nb)];
    if (ad != bd && ad != 1 && bd != 1)
      throw std::runtime_error("shapes not broadcastable");
    out[i] = ad > bd ? ad : bd;
  }
  return out;
}
// broadcast strides of input `in` aligned to output of rank n.
void bcast_strides(const Shape& in, const Shape& out, int64_t* dst) {
  int n = (int)out.size(), nin = (int)in.size();
  Shape ist = contiguous_strides(in);
  for (int d = 0; d < n; ++d) {
    int id = d - (n - nin);
    if (id < 0) dst[d] = 0;
    else if (in[id] == 1 && out[d] != 1) dst[d] = 0;
    else dst[d] = ist[id];
  }
}

template <typename T>
__global__ void binary_kernel(const T* a, const T* b, T* out, DimSpec s,
                              int64_t n, int op) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  int64_t rem = idx, ao = 0, bo = 0;
  for (int d = 0; d < s.ndim; ++d) {
    int64_t c = rem / s.out_str[d];
    rem -= c * s.out_str[d];
    ao += c * s.a_str[d];
    bo += c * s.b_str[d];
  }
  float x = ld<T>(a, ao), y = ld<T>(b, bo), r;
  switch (op) { case 0: r = x + y; break; case 1: r = x - y; break;
                case 2: r = x * y; break; default: r = x / y; }
  st<T>(out, idx, r);
}

template <typename T>
__global__ void scalar_kernel(const T* a, T* out, int64_t n, float s, int op, int lhs) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = ld<T>(a, i), r;
  float l = lhs ? s : x, rr = lhs ? x : s;
  switch (op) { case 0: r = l + rr; break; case 1: r = l - rr; break;
                case 2: r = l * rr; break; default: r = l / rr; }
  st<T>(out, i, r);
}

template <typename T>
__global__ void unary_kernel(const T* a, T* out, int64_t n, int op) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = ld<T>(a, i), r;
  switch (op) {
    case U_NEG: r = -x; break;
    case U_EXP: r = expf(x); break;
    case U_LOG: r = logf(x); break;
    case U_SQRT: r = sqrtf(x); break;
    case U_RELU: r = x > 0 ? x : 0; break;
    case U_SIGMOID: r = 1.f / (1.f + expf(-x)); break;
    case U_TANH: r = tanhf(x); break;
    case U_GELU: r = 0.5f * x * (1.f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x))); break;
    case U_SILU: r = x / (1.f + expf(-x)); break;
    case U_RECIP: r = 1.f / x; break;
    case U_ABS: r = fabsf(x); break;
    case U_SIGN: r = (x > 0) - (x < 0); break;
    default: r = x;
  }
  st<T>(out, i, r);
}

template <typename T>
__global__ void ge_kernel(const T* a, T* out, int64_t n, float s) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) st<T>(out, i, ld<T>(a, i) >= s ? 1.f : 0.f);
}
}  // namespace

NDArray ew_binary(const NDArray& a, const NDArray& b, int op) {
  if (a.dtype != b.dtype) throw std::runtime_error("ew_binary dtype mismatch");
  Shape out_shape = broadcast_shape(a.shape, b.shape);
  NDArray out(out_shape, a.dtype, a.device);
  int64_t n = out.numel();
  if (n == 0) return out;
  DimSpec s{};
  s.ndim = (int)out_shape.size();
  Shape ostr = contiguous_strides(out_shape);
  for (int d = 0; d < s.ndim; ++d) { s.out_shape[d] = out_shape[d]; s.out_str[d] = ostr[d]; }
  bcast_strides(a.shape, out_shape, s.a_str);
  bcast_strides(b.shape, out_shape, s.b_str);
  DISPATCH_FLOAT(a.dtype, T, {
    binary_kernel<T><<<nblk(n), kT>>>(static_cast<T*>(a.data_ptr()),
        static_cast<T*>(b.data_ptr()), static_cast<T*>(out.data_ptr()), s, n, op);
  });
  cuda_check_last("ew_binary");
  return out;
}
NDArray ew_scalar(const NDArray& a, double scalar, int op, bool scalar_lhs) {
  NDArray out(a.shape, a.dtype, a.device);
  int64_t n = a.numel();
  if (n == 0) return out;
  DISPATCH_FLOAT(a.dtype, T, {
    scalar_kernel<T><<<nblk(n), kT>>>(static_cast<T*>(a.data_ptr()),
        static_cast<T*>(out.data_ptr()), n, (float)scalar, op, scalar_lhs ? 1 : 0);
  });
  cuda_check_last("ew_scalar");
  return out;
}
NDArray ew_unary(const NDArray& a, int op) {
  NDArray out(a.shape, a.dtype, a.device);
  int64_t n = a.numel();
  if (n == 0) return out;
  DISPATCH_FLOAT(a.dtype, T, {
    unary_kernel<T><<<nblk(n), kT>>>(static_cast<T*>(a.data_ptr()),
        static_cast<T*>(out.data_ptr()), n, op);
  });
  cuda_check_last("ew_unary");
  return out;
}
NDArray ge_scalar(const NDArray& a, double sval) {
  NDArray out(a.shape, a.dtype, a.device);
  int64_t n = a.numel();
  if (n == 0) return out;
  DISPATCH_FLOAT(a.dtype, T, {
    ge_kernel<T><<<nblk(n), kT>>>(static_cast<T*>(a.data_ptr()),
        static_cast<T*>(out.data_ptr()), n, (float)sval);
  });
  cuda_check_last("ge_scalar");
  return out;
}

// ------------------------------------------------------------- reductions
namespace {
struct ReduceSpec {
  int ndim;
  int64_t in_shape[TC_MAX_DIMS];
  int64_t in_str[TC_MAX_DIMS];    // contiguous strides of input
  int64_t out_str[TC_MAX_DIMS];   // contiguous strides of keepdim output
  int reduced[TC_MAX_DIMS];       // 1 if dim is reduced
  int64_t red_size;               // product of reduced dim sizes
};

template <typename T, int MODE>  // 0=sum 1=max 2=min
__global__ void reduce_kernel(const T* in, T* out, ReduceSpec s, int64_t out_n) {
  int64_t oidx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (oidx >= out_n) return;
  // decode out coords (reduced dims are size 1 here) -> base input offset
  int64_t rem = oidx, base = 0;
  int64_t coord[TC_MAX_DIMS];
  for (int d = 0; d < s.ndim; ++d) {
    int64_t c = rem / s.out_str[d];
    rem -= c * s.out_str[d];
    coord[d] = s.reduced[d] ? 0 : c;
    base += coord[d] * s.in_str[d];
  }
  float acc = MODE == 0 ? 0.f : (MODE == 1 ? -3.4e38f : 3.4e38f);
  for (int64_t r = 0; r < s.red_size; ++r) {
    int64_t rr = r, off = 0;
    for (int d = s.ndim - 1; d >= 0; --d) {
      if (!s.reduced[d]) continue;
      int64_t sz = s.in_shape[d];
      int64_t c = rr % sz; rr /= sz;
      off += c * s.in_str[d];
    }
    float v = ld<T>(in, base + off);
    if (MODE == 0) acc += v;
    else if (MODE == 1) acc = v > acc ? v : acc;
    else acc = v < acc ? v : acc;
  }
  st<T>(out, oidx, acc);
}

NDArray reduce_impl(const NDArray& a, std::vector<int> axes, bool keepdim, int mode) {
  int nd = a.ndim();
  if (axes.empty()) for (int d = 0; d < nd; ++d) axes.push_back(d);
  ReduceSpec s{};
  s.ndim = nd;
  Shape istr = contiguous_strides(a.shape);
  Shape keep_shape(nd);
  s.red_size = 1;
  for (int d = 0; d < nd; ++d) { s.in_shape[d] = a.shape[d]; s.in_str[d] = istr[d]; s.reduced[d] = 0; keep_shape[d] = a.shape[d]; }
  for (int ax : axes) { int d = ax < 0 ? ax + nd : ax; s.reduced[d] = 1; s.red_size *= a.shape[d]; keep_shape[d] = 1; }
  Shape ostr = contiguous_strides(keep_shape);
  for (int d = 0; d < nd; ++d) s.out_str[d] = ostr[d];
  NDArray out(keep_shape, a.dtype, a.device);
  int64_t out_n = out.numel();
  if (out_n > 0) {
    DISPATCH_FLOAT(a.dtype, T, {
      if (mode == 1) reduce_kernel<T, 1><<<nblk(out_n), kT>>>(static_cast<T*>(a.data_ptr()), static_cast<T*>(out.data_ptr()), s, out_n);
      else if (mode == 2) reduce_kernel<T, 2><<<nblk(out_n), kT>>>(static_cast<T*>(a.data_ptr()), static_cast<T*>(out.data_ptr()), s, out_n);
      else reduce_kernel<T, 0><<<nblk(out_n), kT>>>(static_cast<T*>(a.data_ptr()), static_cast<T*>(out.data_ptr()), s, out_n);
    });
    cuda_check_last("reduce");
  }
  if (keepdim) return out;
  Shape squeezed;
  for (int d = 0; d < nd; ++d) if (!s.reduced[d]) squeezed.push_back(a.shape[d]);
  if (squeezed.empty()) squeezed.push_back(1);
  return out.reshape(squeezed);
}
}  // namespace

NDArray reduce_sum(const NDArray& a, const std::vector<int>& axes, bool keepdim) {
  return reduce_impl(a, axes, keepdim, 0);
}
NDArray reduce_max(const NDArray& a, const std::vector<int>& axes, bool keepdim) {
  return reduce_impl(a, axes, keepdim, 1);
}
NDArray reduce_min(const NDArray& a, const std::vector<int>& axes, bool keepdim) {
  return reduce_impl(a, axes, keepdim, 2);
}
NDArray reduce_to(const NDArray& a, const Shape& target) {
  if (a.shape == target) return a.clone();
  // align target to a's rank by left-padding with 1s
  int nd = a.ndim(), nt = (int)target.size();
  std::vector<int> axes;
  for (int d = 0; d < nd; ++d) {
    int td = d - (nd - nt);
    int64_t tsz = (td < 0) ? 1 : target[td];
    if (tsz == 1 && a.shape[d] > 1) axes.push_back(d);
  }
  NDArray summed = axes.empty() ? a.clone() : reduce_sum(a, axes, /*keepdim=*/true);
  return summed.reshape(target);
}

// ------------------------------------------------------------- shape ops
namespace {
struct PermSpec { int ndim; int64_t out_shape[TC_MAX_DIMS]; int64_t out_str[TC_MAX_DIMS]; int64_t in_str_perm[TC_MAX_DIMS]; };
template <typename T>
__global__ void permute_kernel(const T* in, T* out, PermSpec s, int64_t n) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  int64_t rem = idx, ioff = 0;
  for (int d = 0; d < s.ndim; ++d) { int64_t c = rem / s.out_str[d]; rem -= c * s.out_str[d]; ioff += c * s.in_str_perm[d]; }
  st<T>(out, idx, ld<T>(in, ioff));
}
}  // namespace

NDArray permute(const NDArray& a, const std::vector<int>& dims) {
  int nd = a.ndim();
  Shape in_str = contiguous_strides(a.shape);
  Shape out_shape(nd);
  PermSpec s{}; s.ndim = nd;
  for (int d = 0; d < nd; ++d) { int sd = dims[d] < 0 ? dims[d] + nd : dims[d]; out_shape[d] = a.shape[sd]; s.in_str_perm[d] = in_str[sd]; }
  Shape ostr = contiguous_strides(out_shape);
  for (int d = 0; d < nd; ++d) { s.out_shape[d] = out_shape[d]; s.out_str[d] = ostr[d]; }
  NDArray out(out_shape, a.dtype, a.device);
  int64_t n = out.numel();
  if (n > 0) {
    DISPATCH_FLOAT(a.dtype, T, {
      permute_kernel<T><<<nblk(n), kT>>>(static_cast<T*>(a.data_ptr()), static_cast<T*>(out.data_ptr()), s, n);
    });
    cuda_check_last("permute");
  }
  return out;
}
NDArray transpose2d_last(const NDArray& a) {
  int nd = a.ndim();
  std::vector<int> dims(nd);
  for (int d = 0; d < nd; ++d) dims[d] = d;
  std::swap(dims[nd - 1], dims[nd - 2]);
  return permute(a, dims);
}

// ------------------------------------------------------------- pow / compare / where
namespace {
template <typename T>
__global__ void pow_kernel(const T* a, T* out, int64_t n, float p) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) st<T>(out, i, powf(ld<T>(a, i), p));
}
__device__ __forceinline__ float cmp(float x, float y, int op) {
  switch (op) { case 0: return x > y; case 1: return x >= y; case 2: return x < y;
                case 3: return x <= y; case 4: return x == y; default: return x != y; }
}
template <typename T>
__global__ void compare_kernel(const T* a, const T* b, T* out, DimSpec s, int64_t n, int op) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  int64_t rem = idx, ao = 0, bo = 0;
  for (int d = 0; d < s.ndim; ++d) { int64_t c = rem / s.out_str[d]; rem -= c * s.out_str[d]; ao += c * s.a_str[d]; bo += c * s.b_str[d]; }
  st<T>(out, idx, cmp(ld<T>(a, ao), ld<T>(b, bo), op));
}
template <typename T>
__global__ void compare_scalar_kernel(const T* a, T* out, int64_t n, float s, int op) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) st<T>(out, i, cmp(ld<T>(a, i), s, op));
}
struct Dim3Spec { int ndim; int64_t out_str[TC_MAX_DIMS]; int64_t c_str[TC_MAX_DIMS]; int64_t x_str[TC_MAX_DIMS]; int64_t y_str[TC_MAX_DIMS]; };
template <typename T>
__global__ void where_kernel(const T* c, const T* x, const T* y, T* out, Dim3Spec s, int64_t n) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  int64_t rem = idx, co = 0, xo = 0, yo = 0;
  for (int d = 0; d < s.ndim; ++d) { int64_t cc = rem / s.out_str[d]; rem -= cc * s.out_str[d]; co += cc * s.c_str[d]; xo += cc * s.x_str[d]; yo += cc * s.y_str[d]; }
  st<T>(out, idx, ld<T>(c, co) != 0.f ? ld<T>(x, xo) : ld<T>(y, yo));
}
}  // namespace

NDArray ew_pow(const NDArray& a, double p) {
  NDArray out(a.shape, a.dtype, a.device);
  int64_t n = a.numel();
  if (n) { DISPATCH_FLOAT(a.dtype, T, { pow_kernel<T><<<nblk(n), kT>>>(static_cast<T*>(a.data_ptr()), static_cast<T*>(out.data_ptr()), n, (float)p); }); cuda_check_last("pow"); }
  return out;
}
NDArray compare(const NDArray& a, const NDArray& b, int op) {
  if (a.dtype != b.dtype) throw std::runtime_error("compare dtype mismatch");
  Shape os = broadcast_shape(a.shape, b.shape);
  NDArray out(os, a.dtype, a.device);
  int64_t n = out.numel();
  if (!n) return out;
  DimSpec s{}; s.ndim = (int)os.size();
  Shape ostr = contiguous_strides(os);
  for (int d = 0; d < s.ndim; ++d) { s.out_shape[d] = os[d]; s.out_str[d] = ostr[d]; }
  bcast_strides(a.shape, os, s.a_str); bcast_strides(b.shape, os, s.b_str);
  DISPATCH_FLOAT(a.dtype, T, { compare_kernel<T><<<nblk(n), kT>>>(static_cast<T*>(a.data_ptr()), static_cast<T*>(b.data_ptr()), static_cast<T*>(out.data_ptr()), s, n, op); });
  cuda_check_last("compare");
  return out;
}
NDArray compare_scalar(const NDArray& a, double sv, int op) {
  NDArray out(a.shape, a.dtype, a.device);
  int64_t n = a.numel();
  if (n) { DISPATCH_FLOAT(a.dtype, T, { compare_scalar_kernel<T><<<nblk(n), kT>>>(static_cast<T*>(a.data_ptr()), static_cast<T*>(out.data_ptr()), n, (float)sv, op); }); cuda_check_last("compare_scalar"); }
  return out;
}
NDArray where_nd(const NDArray& c, const NDArray& x, const NDArray& y) {
  Shape os = broadcast_shape(broadcast_shape(c.shape, x.shape), y.shape);
  NDArray out(os, x.dtype, x.device);
  int64_t n = out.numel();
  if (!n) return out;
  Dim3Spec s{}; s.ndim = (int)os.size();
  Shape ostr = contiguous_strides(os);
  for (int d = 0; d < s.ndim; ++d) s.out_str[d] = ostr[d];
  bcast_strides(c.shape, os, s.c_str); bcast_strides(x.shape, os, s.x_str); bcast_strides(y.shape, os, s.y_str);
  DISPATCH_FLOAT(x.dtype, T, { where_kernel<T><<<nblk(n), kT>>>(static_cast<T*>(c.data_ptr()), static_cast<T*>(x.data_ptr()), static_cast<T*>(y.data_ptr()), static_cast<T*>(out.data_ptr()), s, n); });
  cuda_check_last("where");
  return out;
}
NDArray broadcast_to(const NDArray& a, const Shape& shape) {
  return ew_binary(NDArray::zeros(shape, a.dtype, a.device), a, 0);
}

// ------------------------------------------------------------- cat / slice
namespace {
struct DimCopySpec { int ndim; int64_t iter_shape[TC_MAX_DIMS]; int64_t iter_str[TC_MAX_DIMS]; int64_t big_str[TC_MAX_DIMS]; int dim; int64_t off; };
template <typename T, int FORWARD>  // FORWARD: big[bigoff]=small[i]; else small[i]=big[bigoff]
__global__ void dimcopy_kernel(const T* small_in, T* small_out, const T* big_in, T* big_out, DimCopySpec s, int64_t n) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  int64_t rem = idx, bigoff = 0;
  for (int d = 0; d < s.ndim; ++d) {
    int64_t c = rem / s.iter_str[d]; rem -= c * s.iter_str[d];
    if (d == s.dim) c += s.off;
    bigoff += c * s.big_str[d];
  }
  if (FORWARD) st<T>(big_out, bigoff, ld<T>(small_in, idx));
  else st<T>(small_out, idx, ld<T>(big_in, bigoff));
}
}  // namespace

NDArray cat_nd(const std::vector<NDArray>& arrs, int dim) {
  if (arrs.empty()) throw std::runtime_error("cat: empty input");
  int nd = arrs[0].ndim();
  if (dim < 0) dim += nd;
  Shape os = arrs[0].shape;
  int64_t total = 0;
  for (auto& a : arrs) total += a.shape[dim];
  os[dim] = total;
  NDArray out(os, arrs[0].dtype, arrs[0].device);
  Shape big_str = contiguous_strides(os);
  int64_t running = 0;
  for (auto& a : arrs) {
    DimCopySpec s{}; s.ndim = nd; s.dim = dim; s.off = running;
    Shape istr = contiguous_strides(a.shape);
    for (int d = 0; d < nd; ++d) { s.iter_shape[d] = a.shape[d]; s.iter_str[d] = istr[d]; s.big_str[d] = big_str[d]; }
    int64_t n = a.numel();
    if (n) { DISPATCH_FLOAT(out.dtype, T, { dimcopy_kernel<T, 1><<<nblk(n), kT>>>(static_cast<T*>(a.data_ptr()), nullptr, nullptr, static_cast<T*>(out.data_ptr()), s, n); }); }
    running += a.shape[dim];
  }
  cuda_check_last("cat");
  return out;
}
NDArray slice_nd(const NDArray& a, int dim, int64_t start, int64_t len) {
  int nd = a.ndim();
  if (dim < 0) dim += nd;
  Shape os = a.shape; os[dim] = len;
  NDArray out(os, a.dtype, a.device);
  Shape big_str = contiguous_strides(a.shape);
  DimCopySpec s{}; s.ndim = nd; s.dim = dim; s.off = start;
  Shape ostr = contiguous_strides(os);
  for (int d = 0; d < nd; ++d) { s.iter_shape[d] = os[d]; s.iter_str[d] = ostr[d]; s.big_str[d] = big_str[d]; }
  int64_t n = out.numel();
  if (n) { DISPATCH_FLOAT(a.dtype, T, { dimcopy_kernel<T, 0><<<nblk(n), kT>>>(nullptr, static_cast<T*>(out.data_ptr()), static_cast<T*>(a.data_ptr()), nullptr, s, n); }); }
  cuda_check_last("slice");
  return out;
}

}  // namespace tc
