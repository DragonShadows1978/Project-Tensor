// kernels.cu: device memory, NDArray, and hand-written CUDA kernels.
//
// Kernels are dtype-generic: values are loaded and computed in fp32 and stored
// back in the array's dtype, so fp32 and fp16 share one implementation.

#include "tc/core.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace tc {

// ------------------------------------------------------------ dtype / device
size_t dtype_size(DType dt) {
  switch (dt) {
    case DType::Float32: return 4;
    case DType::Float16: return 2;
    case DType::Int64: return 8;
    case DType::Bool: return 1;
    case DType::Uint8: return 1;
    case DType::BFloat16: return 2;
  }
  return 4;
}
const char* dtype_name(DType dt) {
  switch (dt) {
    case DType::Float32: return "float32";
    case DType::Float16: return "float16";
    case DType::Int64: return "int64";
    case DType::Bool: return "bool";
    case DType::Uint8: return "uint8";
    case DType::BFloat16: return "bfloat16";
  }
  return "float32";
}
DType dtype_from_string(const std::string& s) {
  if (s == "float32" || s == "float" || s == "f32") return DType::Float32;
  if (s == "float16" || s == "half" || s == "f16") return DType::Float16;
  if (s == "int64" || s == "long") return DType::Int64;
  if (s == "bool") return DType::Bool;
  if (s == "uint8" || s == "u8" || s == "byte") return DType::Uint8;
  if (s == "bfloat16" || s == "bf16" || s == "bfloat") return DType::BFloat16;
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
// Stream-ordered allocation: cudaMallocAsync/cudaFreeAsync on the legacy
// default stream (ALL engine kernels + cuBLAS run on it, so stream-ordering ==
// program-ordering here).
//
// WHY: nsys on a MiniCPM3 prefill measured cudaMalloc+cudaFree as the #1 cost
// of the whole engine — ~6,700 alloc/free pairs per forward, cudaFree's
// implicit device-sync serializing CPU and GPU (8.6s of blocking API time
// across 3 forwards vs 0.14s of kernel launches). The async pool removes the
// sync and recycles within the stream.
//
// NOTE: a HOMEMADE caching free-list (exact-size-keyed) was tried earlier and
// REVERTED — exact-size bucketing pinned ~2GB of differently-sized transients
// and cut the Mistral refine=0.05 ceiling 16384->8192 (measured). The driver's
// suballocating pool is NOT that design, and the release threshold below is
// deliberately small so idle memory returns to the driver at sync points. The
// context-ceiling spot-check is a MANDATORY gate on any change here.
namespace {
constexpr uint64_t kPoolReleaseThreshold = 256ull * 1024 * 1024;  // 256MB
void init_default_pool() {
  static bool done = false;
  if (done) return;
  done = true;
  int dev = 0;
  cudaGetDevice(&dev);
  cudaMemPool_t pool;
  if (cudaDeviceGetDefaultMemPool(&pool, dev) == cudaSuccess) {
    uint64_t thr = kPoolReleaseThreshold;
    cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &thr);
  }
}
}  // namespace

void empty_cache() {
  // Return all pooled-but-unused memory to the driver.
  int dev = 0;
  cudaGetDevice(&dev);
  cudaMemPool_t pool;
  if (cudaDeviceGetDefaultMemPool(&pool, dev) == cudaSuccess)
    cudaMemPoolTrimTo(pool, 0);
}

// LOAD/RUNTIME SPLIT: pooling is gated on a global flag callers enable AFTER
// weight loading (tc.set_alloc_pooling(True)). Persistents (weights, scales,
// norm params — allocated while the flag is off) live in raw cudaMalloc
// memory; ONLY forward-pass transients enter the pool. The wall bisect forced
// this design: pooled LIVE blocks pin reserved chunks that TrimTo cannot
// release, and size-capped hybrids measured either wall loss (<=16MB cap:
// Mistral r0.05@16384 OOM) or no speedup (<=1MB cap: 2539ms vs 2611 raw — at
// S=1024 nearly every transient exceeds 1MB). With a transients-only pool the
// OOM retry's sync+trim empties the pool COMPLETELY, so the raw retry sees
// exactly the pure-raw engine's free memory at the wall.
namespace {
bool g_pool_transients = false;
}

void set_alloc_pooling(bool enabled) {
  init_default_pool();
  g_pool_transients = enabled;
  if (!enabled) empty_cache();
}

Storage::Storage(size_t nbytes_, Device device_) : nbytes(nbytes_), device(device_) {
  if (nbytes == 0) { ptr = nullptr; return; }
  if (device.is_cuda()) {
    init_default_pool();
    pooled = g_pool_transients;
    cudaError_t e = pooled ? cudaMallocAsync(&ptr, nbytes, 0)
                           : cudaMalloc(&ptr, nbytes);
    if (e == cudaErrorMemoryAllocation) {
      // Near the wall: sync materializes every pending cudaFreeAsync, trim
      // returns all idle pool reservations to the driver, then retry once.
      cudaGetLastError();  // clear sticky error
      cudaStreamSynchronize(0);
      empty_cache();
      e = pooled ? cudaMallocAsync(&ptr, nbytes, 0) : cudaMalloc(&ptr, nbytes);
    }
    if (e != cudaSuccess)
      throw std::runtime_error(std::string(pooled ? "cudaMallocAsync failed: "
                                                  : "cudaMalloc failed: ") +
                               cudaGetErrorString(e));
  } else {
    ptr = std::malloc(nbytes);
  }
}
Storage::~Storage() {
  if (!ptr) return;
  if (device.is_cuda()) {
    if (pooled) cudaFreeAsync(ptr, 0);
    else cudaFree(ptr);
  } else {
    std::free(ptr);
  }
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
template <> __device__ __forceinline__ float ld<__nv_bfloat16>(const __nv_bfloat16* p, int64_t i) { return __bfloat162float(p[i]); }
template <typename T> __device__ __forceinline__ void st(T* p, int64_t i, float v);
template <> __device__ __forceinline__ void st<float>(float* p, int64_t i, float v) { p[i] = v; }
template <> __device__ __forceinline__ void st<__half>(__half* p, int64_t i, float v) { p[i] = __float2half(v); }
template <> __device__ __forceinline__ void st<__nv_bfloat16>(__nv_bfloat16* p, int64_t i, float v) { p[i] = __float2bfloat16(v); }

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
    else if ((DT) == DType::BFloat16) { using T = __nv_bfloat16; __VA_ARGS__; } \
    else throw std::runtime_error("op supports float32/float16/bfloat16 only"); \
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
namespace {
// float/half -> uint8 (round, clamp [0,255]) and uint8 -> float/half.
template <typename T>
__global__ void cast_to_u8_kernel(const T* s, uint8_t* d, int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = ld<T>(s, i);
    v = fminf(255.f, fmaxf(0.f, rintf(v)));
    d[i] = (uint8_t)v;
  }
}
template <typename T>
__global__ void cast_from_u8_kernel(const uint8_t* s, T* d, int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) st<T>(d, i, (float)s[i]);
}
}  // namespace

NDArray NDArray::astype(DType dt) const {
  if (dt == dtype) return clone();
  NDArray out(shape, dt, device);
  int64_t n = numel();
  if (n == 0) return out;
  if (dt == DType::Uint8 && dtype != DType::Uint8) {
    DISPATCH_FLOAT(dtype, ST, {
      cast_to_u8_kernel<ST><<<nblk(n), kT>>>(static_cast<ST*>(data_ptr()),
                                             static_cast<uint8_t*>(out.data_ptr()), n);
    });
    cuda_check_last("astype_u8");
    return out;
  }
  if (dtype == DType::Uint8 && dt != DType::Uint8) {
    DISPATCH_FLOAT(dt, DT, {
      cast_from_u8_kernel<DT><<<nblk(n), kT>>>(static_cast<uint8_t*>(data_ptr()),
                                               static_cast<DT*>(out.data_ptr()), n);
    });
    cuda_check_last("astype_from_u8");
    return out;
  }
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
                case 2: r = x * y; break; case 3: r = x / y; break;
                case 4: r = fmaxf(x, y); break; default: r = fminf(x, y); }
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
    case U_SIN: r = sinf(x); break;
    case U_COS: r = cosf(x); break;
    case U_TAN: r = tanf(x); break;
    case U_ASIN: r = asinf(x); break;
    case U_ACOS: r = acosf(x); break;
    case U_ATAN: r = atanf(x); break;
    case U_SINH: r = sinhf(x); break;
    case U_COSH: r = coshf(x); break;
    case U_LOG2: r = log2f(x); break;
    case U_LOG10: r = log10f(x); break;
    case U_FLOOR: r = floorf(x); break;
    case U_CEIL: r = ceilf(x); break;
    case U_ROUND: r = roundf(x); break;
    case U_ISNAN: r = isnan(x) ? 1.f : 0.f; break;
    case U_ISINF: r = isinf(x) ? 1.f : 0.f; break;
    case U_ISFINITE: r = isfinite(x) ? 1.f : 0.f; break;
    default: r = x;
  }
  st<T>(out, i, r);
}

template <typename T>
__global__ void nan_to_num_kernel(const T* a, T* out, int64_t n, float nan, float pinf, float ninf) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = ld<T>(a, i);
  if (isnan(x)) x = nan;
  else if (isinf(x)) x = x > 0 ? pinf : ninf;
  st<T>(out, i, x);
}

template <typename T>
__global__ void clamp_kernel(const T* a, T* out, int64_t n, float lo, float hi) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = ld<T>(a, i);
  st<T>(out, i, x < lo ? lo : (x > hi ? hi : x));
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
NDArray ew_clamp(const NDArray& a, double lo, double hi) {
  NDArray out(a.shape, a.dtype, a.device);
  int64_t n = a.numel();
  if (n) { DISPATCH_FLOAT(a.dtype, T, { clamp_kernel<T><<<nblk(n), kT>>>(static_cast<T*>(a.data_ptr()), static_cast<T*>(out.data_ptr()), n, (float)lo, (float)hi); }); cuda_check_last("clamp"); }
  return out;
}
NDArray ew_nan_to_num(const NDArray& a, double nan, double pinf, double ninf) {
  NDArray out(a.shape, a.dtype, a.device);
  int64_t n = a.numel();
  if (n) { DISPATCH_FLOAT(a.dtype, T, { nan_to_num_kernel<T><<<nblk(n), kT>>>(static_cast<T*>(a.data_ptr()), static_cast<T*>(out.data_ptr()), n, (float)nan, (float)pinf, (float)ninf); }); cuda_check_last("nan_to_num"); }
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

template <typename T, int MODE>  // 0=sum 1=max 2=min 3=prod
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
  float acc = MODE == 0 ? 0.f : (MODE == 1 ? -3.4e38f : (MODE == 2 ? 3.4e38f : 1.f));
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
    else if (MODE == 2) acc = v < acc ? v : acc;
    else acc *= v;
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
      else if (mode == 3) reduce_kernel<T, 3><<<nblk(out_n), kT>>>(static_cast<T*>(a.data_ptr()), static_cast<T*>(out.data_ptr()), s, out_n);
      else reduce_kernel<T, 0><<<nblk(out_n), kT>>>(static_cast<T*>(a.data_ptr()), static_cast<T*>(out.data_ptr()), s, out_n);
    });
    cuda_check_last("reduce");
  }
  if (keepdim) return out;
  Shape squeezed;
  for (int d = 0; d < nd; ++d) if (!s.reduced[d]) squeezed.push_back(a.shape[d]);
  // All axes reduced -> rank-0 scalar (empty shape), not shape (1,).
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
NDArray reduce_prod(const NDArray& a, const std::vector<int>& axes, bool keepdim) {
  return reduce_impl(a, axes, keepdim, 3);
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
// ------------------------------------------------------------- embedding + optim
namespace {
template <typename T>
__global__ void embed_fwd_kernel(const T* w, const int64_t* idx, T* out, int64_t row, int64_t n) {
  int64_t k = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n) return;
  int64_t i = k / row, j = k % row;
  st<T>(out, k, ld<T>(w, idx[i] * row + j));
}
template <typename T>
__global__ void embed_bwd_kernel(const T* grad, const int64_t* idx, float* gWf, int64_t row, int64_t n) {
  int64_t k = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n) return;
  int64_t i = k / row, j = k % row;
  atomicAdd(&gWf[idx[i] * row + j], ld<T>(grad, k));
}
template <typename T>
__global__ void sgd_kernel(T* p, const T* g, T* buf, int64_t n, float lr, float mom, float wd) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float gi = ld<T>(g, i) + wd * ld<T>(p, i);
  float b = mom * ld<T>(buf, i) + gi;
  st<T>(buf, i, b);
  st<T>(p, i, ld<T>(p, i) - lr * b);
}
template <typename T>
__global__ void adam_kernel(T* p, const T* g, T* m, T* v, int64_t n, float lr,
                            float b1, float b2, float eps, float bc1, float bc2,
                            float wd, int decoupled) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float pi = ld<T>(p, i);
  float gi = ld<T>(g, i);
  if (!decoupled) gi += wd * pi;
  float mi = b1 * ld<T>(m, i) + (1.f - b1) * gi;
  float vi = b2 * ld<T>(v, i) + (1.f - b2) * gi * gi;
  st<T>(m, i, mi); st<T>(v, i, vi);
  float mhat = mi / bc1, vhat = vi / bc2;
  if (decoupled) pi -= lr * wd * pi;  // AdamW decoupled weight decay
  st<T>(p, i, pi - lr * mhat / (sqrtf(vhat) + eps));
}
template <typename T>
__global__ void lion_kernel(T* p, const T* g, T* m, int64_t n, float lr,
                            float b1, float b2, float wd) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float gi = ld<T>(g, i), mi = ld<T>(m, i), pi = ld<T>(p, i);
  float upd = b1 * mi + (1.f - b1) * gi;
  float s = upd > 0 ? 1.f : (upd < 0 ? -1.f : 0.f);
  st<T>(p, i, pi - lr * (s + wd * pi));        // decoupled weight decay
  st<T>(m, i, b2 * mi + (1.f - b2) * gi);
}
template <typename T>
__global__ void radam_kernel(T* p, const T* g, T* m, T* v, int64_t n, float lr,
                             float b1, float b2, float eps, float bc1, float bc2,
                             float rect, int rectified, float wd, int decoupled) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float pi = ld<T>(p, i), gi = ld<T>(g, i);
  if (!decoupled) gi += wd * pi;
  float mi = b1 * ld<T>(m, i) + (1.f - b1) * gi;
  float vi = b2 * ld<T>(v, i) + (1.f - b2) * gi * gi;
  st<T>(m, i, mi); st<T>(v, i, vi);
  float mhat = mi / bc1;
  if (decoupled) pi -= lr * wd * pi;
  if (rectified) { float vhat = sqrtf(vi / bc2) + eps; pi -= lr * rect * mhat / vhat; }
  else pi -= lr * mhat;
  st<T>(p, i, pi);
}
template <typename T>
__global__ void rmsprop_kernel(T* p, const T* g, T* sq, int64_t n, float lr,
                               float alpha, float eps, float wd) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float gi = ld<T>(g, i) + wd * ld<T>(p, i);
  float s = alpha * ld<T>(sq, i) + (1.f - alpha) * gi * gi;
  st<T>(sq, i, s);
  st<T>(p, i, ld<T>(p, i) - lr * gi / (sqrtf(s) + eps));
}
template <typename T>
__global__ void adagrad_kernel(T* p, const T* g, T* acc, int64_t n, float lr, float eps, float wd) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float gi = ld<T>(g, i) + wd * ld<T>(p, i);
  float a = ld<T>(acc, i) + gi * gi;
  st<T>(acc, i, a);
  st<T>(p, i, ld<T>(p, i) - lr * gi / (sqrtf(a) + eps));
}
template <typename T>
__global__ void scale_kernel(T* p, int64_t n, float s) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) st<T>(p, i, ld<T>(p, i) * s);
}
}  // namespace

void lion_step(NDArray& param, const NDArray& grad, NDArray& m, double lr,
               double b1, double b2, double wd) {
  int64_t n = param.numel();
  if (!n) return;
  DISPATCH_FLOAT(param.dtype, T, {
    lion_kernel<T><<<nblk(n), kT>>>(static_cast<T*>(param.data_ptr()),
        static_cast<T*>(grad.data_ptr()), static_cast<T*>(m.data_ptr()), n,
        (float)lr, (float)b1, (float)b2, (float)wd);
  });
  cuda_check_last("lion_step");
}
void radam_step(NDArray& param, const NDArray& grad, NDArray& m, NDArray& v,
                double lr, double b1, double b2, double eps, double bc1, double bc2,
                double rect, bool rectified, double wd, bool decoupled) {
  int64_t n = param.numel();
  if (!n) return;
  DISPATCH_FLOAT(param.dtype, T, {
    radam_kernel<T><<<nblk(n), kT>>>(static_cast<T*>(param.data_ptr()),
        static_cast<T*>(grad.data_ptr()), static_cast<T*>(m.data_ptr()),
        static_cast<T*>(v.data_ptr()), n, (float)lr, (float)b1, (float)b2, (float)eps,
        (float)bc1, (float)bc2, (float)rect, rectified ? 1 : 0, (float)wd, decoupled ? 1 : 0);
  });
  cuda_check_last("radam_step");
}
void rmsprop_step(NDArray& param, const NDArray& grad, NDArray& sq, double lr,
                  double alpha, double eps, double wd) {
  int64_t n = param.numel();
  if (!n) return;
  DISPATCH_FLOAT(param.dtype, T, {
    rmsprop_kernel<T><<<nblk(n), kT>>>(static_cast<T*>(param.data_ptr()),
        static_cast<T*>(grad.data_ptr()), static_cast<T*>(sq.data_ptr()), n,
        (float)lr, (float)alpha, (float)eps, (float)wd);
  });
  cuda_check_last("rmsprop_step");
}
void adagrad_step(NDArray& param, const NDArray& grad, NDArray& acc, double lr,
                  double eps, double wd) {
  int64_t n = param.numel();
  if (!n) return;
  DISPATCH_FLOAT(param.dtype, T, {
    adagrad_kernel<T><<<nblk(n), kT>>>(static_cast<T*>(param.data_ptr()),
        static_cast<T*>(grad.data_ptr()), static_cast<T*>(acc.data_ptr()), n,
        (float)lr, (float)eps, (float)wd);
  });
  cuda_check_last("adagrad_step");
}
void scale_(NDArray& param, double s) {
  int64_t n = param.numel();
  if (!n) return;
  DISPATCH_FLOAT(param.dtype, T, {
    scale_kernel<T><<<nblk(n), kT>>>(static_cast<T*>(param.data_ptr()), n, (float)s);
  });
  cuda_check_last("scale_");
}

namespace {
template <typename T>
__global__ void apa_qg_kernel(const T* rotated, const T* boundaries, const T* codebook,
                              T* out, int nb, int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = ld<T>(rotated, i);
  int lo = 0, hi = nb;
  while (lo < hi) { int mid = (lo + hi) >> 1; if (ld<T>(boundaries, mid) < x) lo = mid + 1; else hi = mid; }
  st<T>(out, i, ld<T>(codebook, lo));
}
}  // namespace

NDArray apa_quantize_gather(const NDArray& rotated, const NDArray& boundaries,
                            const NDArray& codebook) {
  NDArray out(rotated.shape, rotated.dtype, rotated.device);
  int64_t n = rotated.numel();
  int nb = (int)boundaries.numel();
  if (n) {
    DISPATCH_FLOAT(rotated.dtype, T, {
      apa_qg_kernel<T><<<nblk(n), kT>>>(static_cast<T*>(rotated.data_ptr()),
          static_cast<T*>(boundaries.data_ptr()), static_cast<T*>(codebook.data_ptr()),
          static_cast<T*>(out.data_ptr()), nb, n);
    });
    cuda_check_last("apa_quantize_gather");
  }
  return out;
}

// ------------------------------------------------------- selective APA attention
// Fused sparse APA-Quant attention. One block per (b,h,query-row); blockDim
// threads cooperate over the S keys. Mirrors the Rust selective reference:
//   1) bulk_j = q . k_quant_j   over ALL keys (cheap, quantized keys)
//   2) build a refine threshold from the BULK scores (z-score: mean+z*std of
//      |bulk|), so the top ~refine_percentile keys are "selected"
//   3) score_j = selected ? (q . k_exact_j)   [full precision, ONLY here]
//                         : bulk_j
//   4) online softmax over score_j, accumulate sum_j softmax_j * v_j
// The full-precision dot is computed ONLY for selected keys — that is the whole
// point: ~85% of keys never get an exact matmul. Never materializes L x S.
//
// q,k,k_quant,v: (B,H,L,D) / (B,H,S,D) row-major. out: (B,H,L,D). z is the
// precomputed z-score for refine_percentile (host-side _norm_ppf). is_causal
// masks keys j>i. D is capped at TC_APA_MAXD for the per-thread reduction buffers.
constexpr int TC_APA_MAXD = 256;
// DMAX is the compile-time size of the per-thread acc[] register array. Sizing it
// to the actual head_dim (64/128) instead of the 256 worst case keeps acc[] in
// registers/L1 instead of spilling to local (global-backed) memory — a large win
// on the selective kernel's hot inner loop. The launcher dispatches the smallest
// DMAX >= D. qsh/osh stay TC_APA_MAXD: they are __shared__, not per-thread, so
// they cost shared memory (cheap, plentiful) not registers.
template <typename T, int DMAX>
__global__ void apa_selective_kernel(
    const T* q, const T* k, const T* kq, const T* v, T* out,
    int B, int H, int L, int S, int D, float scale, float zthr,
    int is_causal, int KVH, int group) {
  // GQA-aware: q has H query heads; k/kq/v have KVH key/value heads (KVH <= H,
  // group = H/KVH). Query head h reads KV head h/group, so the KV tensors are
  // NEVER expanded to H heads — saves materializing the 4x-repeated k_rep/v_rep.
  int row = blockIdx.x;            // flat (b,h,i) over the Q heads
  int i = row % L;
  int bh = row / L;
  int b = bh / H;
  int h = bh % H;
  int kv_h = h / group;            // which of the KVH heads this query head uses
  int tid = threadIdx.x;
  int nt = blockDim.x;

  const T* qrow = q + (int64_t)row * D;
  int64_t kvbh = (int64_t)b * KVH + kv_h;
  const T* kbase = k + kvbh * S * D;
  const T* kqbase = kq + kvbh * S * D;
  const T* vbase = v + kvbh * S * D;

  // Load this query vector into shared memory (D <= DMAX <= TC_APA_MAXD).
  __shared__ float qsh[DMAX];
  for (int d = tid; d < D; d += nt) qsh[d] = ld<T>(qrow, d);
  __syncthreads();

  int s_max = is_causal ? (i + 1) : S;   // causal: keys 0..i only
  __shared__ float red[256];

  // Pass 1: bulk scores -> sum/sumsq of |bulk| for the z-score threshold. Each
  // thread also CACHES its bulk dots so pass 2 never recomputes them. We cache
  // up to a fixed window in registers via re-derivation only when needed; since
  // s_max can be large, we instead keep the bulk dot per strided key in a
  // per-thread running form and recompute exact dots only for selected keys.
  float sum = 0.f, sumsq = 0.f;
  for (int j = tid; j < s_max; j += nt) {
    const T* kqj = kqbase + (int64_t)j * D;
    float dot = 0.f;
    for (int d = 0; d < D; ++d) dot += qsh[d] * ld<T>(kqj, d);
    float a = fabsf(dot * scale);
    sum += a; sumsq += a * a;
  }
  red[tid] = sum; __syncthreads();
  for (int off = nt / 2; off > 0; off >>= 1) { if (tid < off) red[tid] += red[tid+off]; __syncthreads(); }
  float total = red[0]; __syncthreads();
  red[tid] = sumsq; __syncthreads();
  for (int off = nt / 2; off > 0; off >>= 1) { if (tid < off) red[tid] += red[tid+off]; __syncthreads(); }
  float total_sq = red[0]; __syncthreads();

  float cnt = (float)s_max;
  float mean = total / cnt;
  float var = total_sq / cnt - mean * mean;
  float thr = mean + zthr * sqrtf(fmaxf(var, 0.f));

  // Pass 2: SINGLE pass with a per-thread online softmax (FlashAttention-style).
  // Each thread maintains its own running max m, denom l, and weighted-value
  // accumulator acc[D] over its strided keys — computing each key's final score
  // exactly ONCE. The exact dot is taken only for selected keys. No atomics; the
  // per-thread partial softmaxes are merged by a block reduction at the end.
  float m = -1e30f, l = 0.f;
  float acc[DMAX];               // DMAX == smallest power-of-two head_dim >= D
  for (int d = 0; d < D; ++d) acc[d] = 0.f;

  for (int j = tid; j < s_max; j += nt) {
    const T* kqj = kqbase + (int64_t)j * D;
    float bulk = 0.f;
    for (int d = 0; d < D; ++d) bulk += qsh[d] * ld<T>(kqj, d);
    bulk *= scale;
    float score;
    if (fabsf(bulk) >= thr) {
      const T* kj = kbase + (int64_t)j * D;
      float ex = 0.f;
      for (int d = 0; d < D; ++d) ex += qsh[d] * ld<T>(kj, d);
      score = ex * scale;
    } else {
      score = bulk;
    }
    // online softmax update
    float m_new = fmaxf(m, score);
    float corr = __expf(m - m_new);
    float w = __expf(score - m_new);
    l = l * corr + w;
    const T* vj = vbase + (int64_t)j * D;
    for (int d = 0; d < D; ++d) acc[d] = acc[d] * corr + w * ld<T>(vj, d);
    m = m_new;
  }

  // Merge the per-thread online-softmax states into a single result. Reduce the
  // global max first, then rescale each thread's (l, acc) to that max and sum.
  red[tid] = m; __syncthreads();
  for (int off = nt / 2; off > 0; off >>= 1) { if (tid < off) red[tid] = fmaxf(red[tid], red[tid+off]); __syncthreads(); }
  float gmax = red[0]; __syncthreads();

  float rescale = __expf(m - gmax);
  // denom
  red[tid] = l * rescale; __syncthreads();
  for (int off = nt / 2; off > 0; off >>= 1) { if (tid < off) red[tid] += red[tid+off]; __syncthreads(); }
  float denom = red[0]; __syncthreads();
  float inv = denom > 0.f ? 1.f / denom : 0.f;

  // Sum rescaled acc[d] across threads. The previous merge had all nt threads
  // atomicAdd into the same osh[d] addresses — nt-way same-address contention,
  // fully serialized per dim. Instead: each warp reduces its lanes' acc[d] with
  // warp shuffles (no atomics, no bank conflicts) and writes one per-warp partial
  // per dim into shared; then a SINGLE __syncthreads precedes combining the (few)
  // per-warp partials. nt is 128 (== 4 warps) by construction, so this is 2
  // barriers total instead of ~D serialized atomic rounds.
  const unsigned FULL = 0xffffffffu;
  int lane = tid & 31;
  int warp = tid >> 5;
  int nwarp = (nt + 31) >> 5;     // == 4 for nt=128
  __shared__ float osh[DMAX];
  __shared__ float wpart[4][DMAX];   // [warp][dim] partials; nt=128 -> 4 warps
  for (int d = 0; d < D; ++d) {
    float val = acc[d] * rescale;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      val += __shfl_down_sync(FULL, val, off);
    if (lane == 0) wpart[warp][d] = val;
  }
  __syncthreads();
  // Combine the nwarp per-warp partials per dim, distributed across threads.
  for (int d = tid; d < D; d += nt) {
    float s = 0.f;
    for (int w = 0; w < nwarp; ++w) s += wpart[w][d];
    osh[d] = s;
  }
  __syncthreads();

  T* orow = out + (int64_t)row * D;
  for (int d = tid; d < D; d += nt) st<T>(orow, d, osh[d] * inv);
}

NDArray apa_selective_attention(const NDArray& q, const NDArray& k,
                                const NDArray& kq, const NDArray& v,
                                float scale, float zthr, bool is_causal) {
  int B = q.shape[0], H = q.shape[1], L = q.shape[2], D = q.shape[3];
  int S = k.shape[2];
  // GQA: k/kq/v carry KVH heads (<= H). group = H/KVH query heads per KV head.
  int KVH = (int)k.shape[1];
  int group = (KVH > 0) ? (H / KVH) : 1;
  NDArray out({(int64_t)B, (int64_t)H, (int64_t)L, (int64_t)D}, q.dtype, q.device);
  int rows = B * H * L;
  int threads = 128;
  if (rows > 0) {
    // Dispatch the smallest compile-time DMAX >= D so acc[] stays register/L1
    // resident. Real head_dims are 64 (TinyLlama) and 128 (Mistral/Qwen/OLMoE);
    // 256 is the safety fallback (== the old behaviour) for anything larger.
    if (D > TC_APA_MAXD) {
      throw std::runtime_error("apa_selective: head_dim exceeds TC_APA_MAXD");
    }
    DISPATCH_FLOAT(q.dtype, T, {
      auto launch = [&](auto dmax_tag) {
        constexpr int DMAX = decltype(dmax_tag)::value;
        apa_selective_kernel<T, DMAX><<<rows, threads>>>(
            static_cast<T*>(q.data_ptr()), static_cast<T*>(k.data_ptr()),
            static_cast<T*>(kq.data_ptr()), static_cast<T*>(v.data_ptr()),
            static_cast<T*>(out.data_ptr()), B, H, L, S, D, scale, zthr,
            is_causal ? 1 : 0, KVH, group);
      };
      if (D <= 64)        launch(std::integral_constant<int, 64>{});
      else if (D <= 128)  launch(std::integral_constant<int, 128>{});
      else                launch(std::integral_constant<int, TC_APA_MAXD>{});
    });
    cuda_check_last("apa_selective");
  }
  return out;
}

// ----------------------------------------------- APA blend+softmax (post-matmul)
// Given precomputed bulk and ranking score matrices (each (rows, S), produced by
// cuBLAS), produce softmax weights for selective APA in ONE fused kernel:
//   thr_row = mean(|bulk|) + zthr*std(|bulk|)   (over valid keys)
//   score_j = |bulk_j| >= thr ? ranking_j : bulk_j
//   weights = softmax(score)
// Selection is on |bulk| (the cheap quantized scores), matching the fused
// apa_selective_kernel and the Python reference: APA picks which keys to refine
// using only the signal it already has, never the expensive |ranking| it is
// deciding whether to recompute.
// One block per row; threads cooperate over S. This replaces the
// abs/mean/std/ge/where/softmax op chain with a single launch. Output overwrites
// into `out` (rows, S). Causal masking is expected to be ALREADY baked into
// bulk/rank (masked keys set to a large negative value before this kernel), so
// masked positions exp() to ~0 naturally; they are also excluded from the stat.
template <typename T>
__global__ void apa_blend_softmax_kernel2(const T* bulk, const T* rank, T* out,
                                          int S, float zthr) {
  int r = blockIdx.x;
  int tid = threadIdx.x, nt = blockDim.x;
  const T* brow = bulk + (int64_t)r * S;
  const T* rrow = rank + (int64_t)r * S;
  T* orow = out + (int64_t)r * S;

  // Masked keys are marked with a large-negative score by the caller (in BOTH
  // bulk and rank); exclude them from the |bulk| mean/std (otherwise they poison
  // the threshold) — they also exp() to ~0 in the softmax so they drop out there.
  // The causal mask bias is -1e4 (functional.py / _cublas_blend_attention); set
  // the cutoff safely between real scaled logits (O(+-50)) and -1e4 so a
  // legitimate very-negative logit is never mistaken for a masked key.
  const float MASK_LIM = -5e3f;
  __shared__ float red[256];
  float sum = 0.f, sumsq = 0.f, vcount = 0.f;
  for (int j = tid; j < S; j += nt) {
    float bk = ld<T>(brow, j);
    if (bk <= MASK_LIM) continue;
    float a = fabsf(bk); sum += a; sumsq += a*a; vcount += 1.f;
  }
  red[tid] = sum; __syncthreads();
  for (int o = nt/2; o > 0; o >>= 1) { if (tid < o) red[tid]+=red[tid+o]; __syncthreads(); }
  float total = red[0]; __syncthreads();
  red[tid] = sumsq; __syncthreads();
  for (int o = nt/2; o > 0; o >>= 1) { if (tid < o) red[tid]+=red[tid+o]; __syncthreads(); }
  float total_sq = red[0]; __syncthreads();
  red[tid] = vcount; __syncthreads();
  for (int o = nt/2; o > 0; o >>= 1) { if (tid < o) red[tid]+=red[tid+o]; __syncthreads(); }
  float cnt = fmaxf(red[0], 1.f); __syncthreads();
  float mean = total/cnt;
  float thr = mean + zthr * sqrtf(fmaxf(total_sq/cnt - mean*mean, 0.f));

  // Single fused pass: compute the blended score per key once, track running
  // max + online-softmax denom (FlashAttention merge), write the UNnormalized
  // exp weight. Final normalization (1/denom) is folded into the caller's
  // weights@V matmul instead of a 4th re-read/re-write pass — so this kernel now
  // makes 2 passes (stat + this) instead of 4. denom is written per-row to
  // `out`'s companion? No: we renormalize here in one extra cheap reduce, but
  // never re-read the full row — we keep each thread's written weights and the
  // block denom, then a final scale uses the already-resident values via shared.
  float m = -1e30f, l = 0.f;
  // First, the per-thread online softmax over its strided keys (one read each).
  // Store nothing yet; we need the global max before exp. To avoid a separate
  // max pass we use the online-softmax rescale trick across the strided scan.
  // Selection on |bulk|: a masked key (bulk <= MASK_LIM) keeps its large-negative
  // bulk value (exp()s to ~0); otherwise refine to rank when |bulk| >= thr.
  for (int j = tid; j < S; j += nt) {
    float bk = ld<T>(brow, j);
    float sc = (bk <= MASK_LIM) ? bk : ((fabsf(bk) >= thr) ? ld<T>(rrow, j) : bk);
    float m_new = fmaxf(m, sc);
    l = l * __expf(m - m_new) + __expf(sc - m_new);
    m = m_new;
  }
  // merge per-thread (m,l) -> global
  red[tid] = m; __syncthreads();
  for (int o = nt/2; o > 0; o >>= 1) { if (tid < o) red[tid]=fmaxf(red[tid],red[tid+o]); __syncthreads(); }
  float gmax = red[0]; __syncthreads();
  red[tid] = l * __expf(m - gmax); __syncthreads();
  for (int o = nt/2; o > 0; o >>= 1) { if (tid < o) red[tid]+=red[tid+o]; __syncthreads(); }
  float denom = red[0]; __syncthreads();
  float inv = denom > 0.f ? 1.f/denom : 0.f;
  // second pass: write normalized weights (recompute blended score — cheaper
  // than a 3rd read of a stored weight, and exact dots aren't involved here).
  // Clamp sc-gmax at the low end so masked/very-negative scores hit a safe
  // __expf input (expf flushes to 0 below ~-88 for fp32; clamp avoids any
  // denormal/edge behaviour) — weight is ~0 there anyway.
  for (int j = tid; j < S; j += nt) {
    float bk = ld<T>(brow, j);
    float sc = (bk <= MASK_LIM) ? bk : ((fabsf(bk) >= thr) ? ld<T>(rrow, j) : bk);
    st<T>(orow, j, __expf(fmaxf(sc - gmax, -88.f)) * inv);
  }
}

NDArray apa_blend_softmax(const NDArray& bulk, const NDArray& rank,
                          float zthr, const NDArray* /*unused*/) {
  int nd = bulk.ndim();
  int64_t S = bulk.shape[nd-1];
  int64_t rows = bulk.numel() / S;
  NDArray out(bulk.shape, bulk.dtype, bulk.device);
  if (rows > 0) {
    DISPATCH_FLOAT(bulk.dtype, T, {
      apa_blend_softmax_kernel2<T><<<(int)rows, 256>>>(
          static_cast<T*>(bulk.data_ptr()), static_cast<T*>(rank.data_ptr()),
          static_cast<T*>(out.data_ptr()), (int)S, zthr);
    });
    cuda_check_last("apa_blend_softmax");
  }
  return out;
}

// ------------------------------------------------------------- int4 quant linear
// Dequantize a per-group 4-bit weight into a (K, N) fp16/fp32 matrix laid out
// TRANSPOSED relative to the (N, K) logical weight, so the result feeds straight
// into matmul(x, W_kn) to compute y = x @ dequant(W)^T. One thread per output
// element of the (K, N) transposed matrix.
//   packed : (N, K/2) uint8, even col = low nibble, odd col = high nibble
//   scales : (N, G) fp16,  zeros : (N, G) fp16,  G = K/group_size
template <typename T>
__global__ void int4_dequant_t_kernel(const uint8_t* packed, const __half* scales,
                                      const __half* zeros, T* out_kn,
                                      int64_t N, int64_t K, int group_size,
                                      int64_t G, int64_t n) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;            // idx walks the (K, N) transposed output
  int64_t k = idx / N;             // input-feature index  [0, K)
  int64_t col = idx - k * N;       // output-feature index [0, N)  (== row of W)
  int64_t byte_idx = col * (K / 2) + (k >> 1);
  uint8_t byte = packed[byte_idx];
  int q = (k & 1) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
  int64_t g = k / group_size;
  float scale = __half2float(scales[col * G + g]);
  // zeros == nullptr selects the SYMMETRIC-8 convention (z = -8*s): the
  // exact q4_0 grid (w = s*(q-8)) without materializing a zeros tensor
  // that is pure redundancy (QAT import path).
  float zero = zeros ? __half2float(zeros[col * G + g]) : -8.f * scale;
  float w = (float)q * scale + zero;
  st<T>(out_kn, idx, w);
}

NDArray int4_dequant(const NDArray& packed, const NDArray& scales,
                     const NDArray& zeros, int group_size, DType out_dtype) {
  // The kernel reinterprets scales/zeros as __half* unconditionally; fp32 inputs
  // would be read as 2-byte halves (garbage weights, silently). Fail loud.
  // An EMPTY zeros tensor selects the symmetric-8 convention (z = -8*s).
  if (scales.dtype != DType::Float16 ||
      (zeros.numel() > 0 && zeros.dtype != DType::Float16)) {
    throw std::runtime_error(
        "int4_dequant: scales and zeros must be float16 (got scales=" +
        std::string(dtype_name(scales.dtype)) + ", zeros=" +
        std::string(dtype_name(zeros.dtype)) + ")");
  }
  if (packed.dtype != DType::Uint8) {
    throw std::runtime_error("int4_dequant: packed weights must be uint8 (got " +
                             std::string(dtype_name(packed.dtype)) + ")");
  }
  int64_t N = packed.shape[0];
  int64_t K = packed.shape[1] * 2;
  int64_t G = K / group_size;
  // Output is the TRANSPOSED dequantized weight: (K, N).
  NDArray out({K, N}, out_dtype, packed.device);
  int64_t n = K * N;
  if (n) {
    DISPATCH_FLOAT(out_dtype, T, {
      int4_dequant_t_kernel<T><<<nblk(n), kT>>>(
          static_cast<uint8_t*>(packed.data_ptr()),
          static_cast<__half*>(scales.data_ptr()),
          zeros.numel() ? static_cast<__half*>(zeros.data_ptr()) : nullptr,
          static_cast<T*>(out.data_ptr()), N, K, group_size, G, n);
    });
    cuda_check_last("int4_dequant");
  }
  return out;
}

NDArray int4_linear(const NDArray& x, const NDArray& packed,
                    const NDArray& scales, const NDArray& zeros, int group_size) {
  // Dequant W into (K, N) and matmul: (..., K) @ (K, N) -> (..., N).
  NDArray w_kn = int4_dequant(packed, scales, zeros, group_size, x.dtype);
  return matmul(x, w_kn);
}

// ----------------------------------------------- fused int4 dequant-GEMM (#12)
// y[M,N] = x[M,K] @ dequant(W)[N,K]^T, with the int4 weight dequantized into
// shared-memory tiles INSIDE the GEMM — never materializing the full (K,N) fp16
// weight buffer that int4_linear's two-stage path allocates (~112 MB for a
// 14336x4096 down_proj; ~416 MB/layer of transient). This removes that peak-VRAM
// transient (the real prize) and the extra round-trip of writing+rereading the
// dequantized weight. A classic 16x16 shared-memory tiled GEMM; the B (weight)
// tile is unpacked + scaled on load. Correctness-identical to int4_linear; this
// is an OPT-IN path (the cuBLAS two-stage stays the default until benchmarked to
// win — a hand GEMM can lose to cuBLAS at large N).
//   packed (N,K/2) uint8, scales/zeros (N,G) fp16, G=K/group_size.
#define TC_I4_TILE 16
template <typename T>
__global__ void int4_gemm_fused_kernel(
    const T* x, const uint8_t* packed, const __half* scales, const __half* zeros,
    T* y, int M, int N, int K, int group_size, int G) {
  __shared__ float xs[TC_I4_TILE][TC_I4_TILE];   // x tile  [row][k]
  __shared__ float ws[TC_I4_TILE][TC_I4_TILE];   // Wdq tile [k][col]
  int row = blockIdx.y * TC_I4_TILE + threadIdx.y;   // m in [0,M)
  int col = blockIdx.x * TC_I4_TILE + threadIdx.x;   // n in [0,N)
  float acc = 0.f;
  for (int k0 = 0; k0 < K; k0 += TC_I4_TILE) {
    // load x[row, k0+tx]
    int kx = k0 + threadIdx.x;
    xs[threadIdx.y][threadIdx.x] =
        (row < M && kx < K) ? ld<T>(x, (int64_t)row * K + kx) : 0.f;
    // load + dequant W[col, k0+ty]  (weight row = output col)
    int kw = k0 + threadIdx.y;
    float wv = 0.f;
    if (col < N && kw < K) {
      int64_t byte_idx = (int64_t)col * (K / 2) + (kw >> 1);
      uint8_t byte = packed[byte_idx];
      int q = (kw & 1) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
      int g = kw / group_size;
      float sc = __half2float(scales[(int64_t)col * G + g]);
      float ze = zeros ? __half2float(zeros[(int64_t)col * G + g])
                       : -8.f * sc;          // symmetric-8 (q4_0 grid)
      wv = (float)q * sc + ze;
    }
    ws[threadIdx.y][threadIdx.x] = wv;
    __syncthreads();
    #pragma unroll
    for (int t = 0; t < TC_I4_TILE; ++t) acc += xs[threadIdx.y][t] * ws[t][threadIdx.x];
    __syncthreads();
  }
  if (row < M && col < N) st<T>(y, (int64_t)row * N + col, acc);
}

// ---------------------------------------------- int4 GEMV (M==1 decode path)
// y[N] = x[K] . dequant(W)[N,K]. The 16x16 tiled GEMM above wastes 15/16 of
// each tile at M=1 and refetches scales per element; decode is GEMV-shaped,
// so stream each weight row ONCE at packed-int4 width instead. One warp per
// output n: lanes read the row as uchar4 (8 weights, one quant group when
// group_size % 8 == 0 — guaranteed by the host guard), x staged in dynamic
// shared memory as fp32, fp32 accumulate (same as the GEMM), warp-shuffle
// reduce. Memory-bound at packed size: ~30x less traffic than the tile path.
template <typename T>
__global__ void int4_gemv_kernel(
    const T* __restrict__ x, const uint8_t* __restrict__ packed,
    const __half* __restrict__ scales, const __half* __restrict__ zeros,
    T* __restrict__ y, int N, int K, int group_size, int G) {
  extern __shared__ float xs[];                       // K floats
  for (int k = threadIdx.x; k < K; k += blockDim.x) xs[k] = ld<T>(x, k);
  __syncthreads();
  const int warps = blockDim.x >> 5;
  const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n = blockIdx.x * warps + wid;
  if (n >= N) return;
  const int Kb = K >> 1;                              // packed bytes per row
  const uchar4* row4 = reinterpret_cast<const uchar4*>(packed + (int64_t)n * Kb);
  const __half* srow = scales + (int64_t)n * G;
  const __half* zrow = zeros ? zeros + (int64_t)n * G : nullptr;
  float acc = 0.f;
  const int nb4 = Kb >> 2;
  for (int b4 = lane; b4 < nb4; b4 += 32) {
    uchar4 v = row4[b4];
    int k0 = b4 << 3;                                 // first of 8 weights
    int g = k0 / group_size;
    float sc = __half2float(srow[g]);
    float ze = zrow ? __half2float(zrow[g]) : -8.f * sc;  // symmetric-8
    const float* xk = xs + k0;
    acc += ((float)(v.x & 0x0F) * sc + ze) * xk[0]
         + ((float)((v.x >> 4) & 0x0F) * sc + ze) * xk[1]
         + ((float)(v.y & 0x0F) * sc + ze) * xk[2]
         + ((float)((v.y >> 4) & 0x0F) * sc + ze) * xk[3]
         + ((float)(v.z & 0x0F) * sc + ze) * xk[4]
         + ((float)((v.z >> 4) & 0x0F) * sc + ze) * xk[5]
         + ((float)(v.w & 0x0F) * sc + ze) * xk[6]
         + ((float)((v.w >> 4) & 0x0F) * sc + ze) * xk[7];
  }
  for (int off = 16; off; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
  if (lane == 0) st<T>(y, n, acc);
}

NDArray int4_linear_fused(const NDArray& x, const NDArray& packed,
                          const NDArray& scales, const NDArray& zeros,
                          int group_size) {
  if (scales.dtype != DType::Float16 ||
      (zeros.numel() > 0 && zeros.dtype != DType::Float16))
    throw std::runtime_error("int4_linear_fused: scales/zeros must be float16");
  if (packed.dtype != DType::Uint8)
    throw std::runtime_error("int4_linear_fused: packed must be uint8");
  int nd = x.ndim();
  int64_t K = x.shape[nd - 1];
  int64_t N = packed.shape[0];
  if (packed.shape[1] * 2 != K)
    throw std::runtime_error("int4_linear_fused: K mismatch");
  int64_t M = x.numel() / K;
  int G = (int)(K / group_size);
  Shape os;
  for (int d = 0; d < nd - 1; ++d) os.push_back(x.shape[d]);
  os.push_back(N);
  NDArray out(os, x.dtype, x.device);
  if (M == 0 || N == 0) return out;
  // GEMV fast path: decode-shaped calls. Guards: 8-weight vector loads must
  // stay inside one quant group and one row, and x must fit in shared memory.
  // Above the 48KB default the kernel opts in to large dynamic shmem
  // (sm_86 allows ~99KB) — without this, K=15360 rows (Gemma 4 ffn_down,
  // ~21% of all decode weight reads) fall to the tile path and decode
  // crawls (measured 4.6 tok/s at the ready gate).
  if (M == 1 && (group_size % 8) == 0 && (K % 8) == 0 &&
      (size_t)K * sizeof(float) <= 96 * 1024) {
    const int threads = 256, warps = threads / 32;
    dim3 ggrid((int)((N + warps - 1) / warps));
    size_t shmem = (size_t)K * sizeof(float);
    DISPATCH_FLOAT(x.dtype, T, {
      if (shmem > 48 * 1024) {
        cudaFuncSetAttribute(int4_gemv_kernel<T>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             96 * 1024);
      }
      int4_gemv_kernel<T><<<ggrid, threads, shmem>>>(
          static_cast<T*>(x.data_ptr()), static_cast<uint8_t*>(packed.data_ptr()),
          static_cast<__half*>(scales.data_ptr()),
          zeros.numel() ? static_cast<__half*>(zeros.data_ptr()) : nullptr,
          static_cast<T*>(out.data_ptr()), (int)N, (int)K, group_size, (int)G);
    });
    cuda_check_last("int4_gemv");
    return out;
  }
  dim3 block(TC_I4_TILE, TC_I4_TILE);
  dim3 grid((int)((N + TC_I4_TILE - 1) / TC_I4_TILE),
            (int)((M + TC_I4_TILE - 1) / TC_I4_TILE));
  DISPATCH_FLOAT(x.dtype, T, {
    int4_gemm_fused_kernel<T><<<grid, block>>>(
        static_cast<T*>(x.data_ptr()), static_cast<uint8_t*>(packed.data_ptr()),
        static_cast<__half*>(scales.data_ptr()),
        zeros.numel() ? static_cast<__half*>(zeros.data_ptr()) : nullptr,
        static_cast<T*>(out.data_ptr()), (int)M, (int)N, (int)K, group_size, G);
  });
  cuda_check_last("int4_linear_fused");
  return out;
}

NDArray embedding_forward(const NDArray& weight, const NDArray& idx) {
  int64_t V = weight.shape[0];
  int64_t row = weight.numel() / V;
  int64_t nidx = idx.numel();
  Shape os(idx.shape);
  for (size_t d = 1; d < weight.shape.size(); ++d) os.push_back(weight.shape[d]);
  NDArray out(os, weight.dtype, weight.device);
  int64_t n = nidx * row;
  if (n) {
    DISPATCH_FLOAT(weight.dtype, T, {
      embed_fwd_kernel<T><<<nblk(n), kT>>>(static_cast<T*>(weight.data_ptr()),
          static_cast<int64_t*>(idx.data_ptr()), static_cast<T*>(out.data_ptr()), row, n);
    });
    cuda_check_last("embedding_fwd");
  }
  return out;
}
NDArray embedding_backward(const NDArray& grad, const NDArray& idx,
                           const Shape& weight_shape, DType weight_dtype) {
  int64_t V = weight_shape[0];
  int64_t row = numel_of(weight_shape) / V;
  int64_t nidx = idx.numel();
  NDArray gWf = NDArray::zeros(weight_shape, DType::Float32, grad.device);
  int64_t n = nidx * row;
  if (n) {
    DISPATCH_FLOAT(grad.dtype, T, {
      embed_bwd_kernel<T><<<nblk(n), kT>>>(static_cast<T*>(grad.data_ptr()),
          static_cast<int64_t*>(idx.data_ptr()), static_cast<float*>(gWf.data_ptr()), row, n);
    });
    cuda_check_last("embedding_bwd");
  }
  return gWf.astype(weight_dtype);
}
void axpy_(NDArray& param, const NDArray& other, double alpha) {
  // param += alpha*other  (shapes equal)
  NDArray upd = ew_binary(param, ew_scalar(other, alpha, 2, false), 0);
  size_t bytes = param.numel() * dtype_size(param.dtype);
  cudaMemcpy(param.data_ptr(), upd.data_ptr(), bytes, cudaMemcpyDeviceToDevice);
}
void sgd_step(NDArray& param, const NDArray& grad, NDArray& buf, double lr,
              double momentum, double wd) {
  int64_t n = param.numel();
  if (!n) return;
  DISPATCH_FLOAT(param.dtype, T, {
    sgd_kernel<T><<<nblk(n), kT>>>(static_cast<T*>(param.data_ptr()),
        static_cast<T*>(grad.data_ptr()), static_cast<T*>(buf.data_ptr()), n,
        (float)lr, (float)momentum, (float)wd);
  });
  cuda_check_last("sgd_step");
}
void adam_step(NDArray& param, const NDArray& grad, NDArray& m, NDArray& v,
               double lr, double b1, double b2, double eps, int64_t t,
               double wd, bool decoupled) {
  int64_t n = param.numel();
  if (!n) return;
  float bc1 = 1.f - powf((float)b1, (float)t);
  float bc2 = 1.f - powf((float)b2, (float)t);
  DISPATCH_FLOAT(param.dtype, T, {
    adam_kernel<T><<<nblk(n), kT>>>(static_cast<T*>(param.data_ptr()),
        static_cast<T*>(grad.data_ptr()), static_cast<T*>(m.data_ptr()),
        static_cast<T*>(v.data_ptr()), n, (float)lr, (float)b1, (float)b2,
        (float)eps, bc1, bc2, (float)wd, decoupled ? 1 : 0);
  });
  cuda_check_last("adam_step");
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
// ------------------------------------------------------------- arg/cumsum/gather/flip
namespace {
template <typename T>
__global__ void arg_kernel(const T* a, int64_t* out, int64_t outer, int64_t A, int64_t inner, int is_max) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t n = outer * inner;
  if (idx >= n) return;
  int64_t o = idx / inner, i = idx % inner;
  float best = is_max ? -3.4e38f : 3.4e38f; int64_t bj = 0;
  for (int64_t j = 0; j < A; ++j) {
    float v = ld<T>(a, o * A * inner + j * inner + i);
    if ((is_max && v > best) || (!is_max && v < best)) { best = v; bj = j; }
  }
  out[idx] = bj;
}
template <typename T>
__global__ void cumsum_kernel(const T* a, T* out, int64_t outer, int64_t A, int64_t inner) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t n = outer * inner;
  if (idx >= n) return;
  int64_t o = idx / inner, i = idx % inner;
  float acc = 0.f;
  for (int64_t j = 0; j < A; ++j) {
    int64_t off = o * A * inner + j * inner + i;
    acc += ld<T>(a, off);
    st<T>(out, off, acc);
  }
}
struct GatherSpec { int ndim; int64_t idx_str[TC_MAX_DIMS]; int64_t a_str[TC_MAX_DIMS]; int64_t idx_shape[TC_MAX_DIMS]; int dim; };
template <typename T>
__global__ void gather_kernel(const T* a, const int64_t* index, T* out, GatherSpec s, int64_t n) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  int64_t rem = idx, aoff = 0;
  for (int d = 0; d < s.ndim; ++d) {
    int64_t c = rem / s.idx_str[d]; rem -= c * s.idx_str[d];
    if (d == s.dim) c = index[idx];
    aoff += c * s.a_str[d];
  }
  st<T>(out, idx, ld<T>(a, aoff));
}
template <typename T>
__global__ void scatter_add_kernel(const T* src, const int64_t* index, float* out, GatherSpec s, int64_t n) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  int64_t rem = idx, aoff = 0;
  for (int d = 0; d < s.ndim; ++d) {
    int64_t c = rem / s.idx_str[d]; rem -= c * s.idx_str[d];
    if (d == s.dim) c = index[idx];
    aoff += c * s.a_str[d];
  }
  atomicAdd(&out[aoff], ld<T>(src, idx));
}
struct FlipSpec { int ndim; int64_t shape[TC_MAX_DIMS]; int64_t str[TC_MAX_DIMS]; int flip[TC_MAX_DIMS]; };
template <typename T>
__global__ void flip_kernel(const T* a, T* out, FlipSpec s, int64_t n) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  int64_t rem = idx, aoff = 0;
  for (int d = 0; d < s.ndim; ++d) {
    int64_t c = rem / s.str[d]; rem -= c * s.str[d];
    if (s.flip[d]) c = s.shape[d] - 1 - c;
    aoff += c * s.str[d];
  }
  st<T>(out, idx, ld<T>(a, aoff));
}
void axis_split(const Shape& shape, int axis, int64_t& outer, int64_t& A, int64_t& inner) {
  int nd = (int)shape.size();
  if (axis < 0) axis += nd;
  outer = 1; inner = 1; A = shape[axis];
  for (int d = 0; d < axis; ++d) outer *= shape[d];
  for (int d = axis + 1; d < nd; ++d) inner *= shape[d];
}
}  // namespace

NDArray reduce_arg(const NDArray& a, int axis, bool is_max) {
  int nd = a.ndim();
  if (axis < 0) axis += nd;
  int64_t outer, A, inner; axis_split(a.shape, axis, outer, A, inner);
  Shape os; for (int d = 0; d < nd; ++d) if (d != axis) os.push_back(a.shape[d]);
  if (os.empty()) os.push_back(1);
  NDArray out(os, DType::Int64, a.device);
  int64_t n = outer * inner;
  if (n) { DISPATCH_FLOAT(a.dtype, T, { arg_kernel<T><<<nblk(n), kT>>>(static_cast<T*>(a.data_ptr()), static_cast<int64_t*>(out.data_ptr()), outer, A, inner, is_max ? 1 : 0); }); cuda_check_last("arg"); }
  return out;
}
NDArray cumsum_nd(const NDArray& a, int axis) {
  int64_t outer, A, inner; axis_split(a.shape, axis, outer, A, inner);
  NDArray out(a.shape, a.dtype, a.device);
  int64_t n = outer * inner;
  if (n) { DISPATCH_FLOAT(a.dtype, T, { cumsum_kernel<T><<<nblk(n), kT>>>(static_cast<T*>(a.data_ptr()), static_cast<T*>(out.data_ptr()), outer, A, inner); }); cuda_check_last("cumsum"); }
  return out;
}
NDArray gather_nd(const NDArray& a, int dim, const NDArray& index) {
  int nd = a.ndim();
  if (dim < 0) dim += nd;
  NDArray out(index.shape, a.dtype, a.device);
  GatherSpec s{}; s.ndim = nd; s.dim = dim;
  Shape istr = contiguous_strides(index.shape), astr = contiguous_strides(a.shape);
  for (int d = 0; d < nd; ++d) { s.idx_str[d] = istr[d]; s.a_str[d] = astr[d]; s.idx_shape[d] = index.shape[d]; }
  int64_t n = out.numel();
  if (n) { DISPATCH_FLOAT(a.dtype, T, { gather_kernel<T><<<nblk(n), kT>>>(static_cast<T*>(a.data_ptr()), static_cast<int64_t*>(index.data_ptr()), static_cast<T*>(out.data_ptr()), s, n); }); cuda_check_last("gather"); }
  return out;
}
NDArray scatter_add_nd(const Shape& shape, DType dtype, int dim, const NDArray& index, const NDArray& src) {
  int nd = (int)shape.size();
  if (dim < 0) dim += nd;
  NDArray outf = NDArray::zeros(shape, DType::Float32, src.device);
  GatherSpec s{}; s.ndim = nd; s.dim = dim;
  Shape istr = contiguous_strides(index.shape), astr = contiguous_strides(shape);
  for (int d = 0; d < nd; ++d) { s.idx_str[d] = istr[d]; s.a_str[d] = astr[d]; s.idx_shape[d] = index.shape[d]; }
  int64_t n = src.numel();
  if (n) { DISPATCH_FLOAT(src.dtype, T, { scatter_add_kernel<T><<<nblk(n), kT>>>(static_cast<T*>(src.data_ptr()), static_cast<int64_t*>(index.data_ptr()), static_cast<float*>(outf.data_ptr()), s, n); }); cuda_check_last("scatter_add"); }
  return outf.astype(dtype);
}
namespace {
template <typename T>
__global__ void topk_kernel(const T* a, T* vals, int64_t* idxs,
                            int64_t outer, int64_t S, int k, int largest) {
  int64_t row = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= outer) return;
  const T* arow = a + row * S;
  float prev_val = largest ? 3.4e38f : -3.4e38f;
  int64_t prev_idx = -1;
  for (int p = 0; p < k; ++p) {
    float best_val = largest ? -3.4e38f : 3.4e38f;
    int64_t best_idx = -1;
    for (int64_t j = 0; j < S; ++j) {
      float v = ld<T>(arow, j);
      bool after = largest ? (v < prev_val || (v == prev_val && j > prev_idx))
                           : (v > prev_val || (v == prev_val && j > prev_idx));
      if (!after) continue;
      bool better;
      if (best_idx < 0) better = true;
      else better = largest ? (v > best_val || (v == best_val && j < best_idx))
                            : (v < best_val || (v == best_val && j < best_idx));
      if (better) { best_val = v; best_idx = j; }
    }
    st<T>(vals, row * k + p, best_val);
    idxs[row * k + p] = best_idx;
    prev_val = best_val; prev_idx = best_idx;
  }
}
}  // namespace

std::tuple<NDArray, NDArray> topk_nd(const NDArray& a, int k, bool largest) {
  int nd = a.ndim();
  int64_t S = a.shape[nd - 1];
  int64_t outer = a.numel() / S;
  Shape os = a.shape; os[nd - 1] = k;
  NDArray vals(os, a.dtype, a.device);
  NDArray idxs(os, DType::Int64, a.device);
  if (outer > 0) {
    DISPATCH_FLOAT(a.dtype, T, {
      topk_kernel<T><<<nblk(outer), kT>>>(static_cast<T*>(a.data_ptr()),
          static_cast<T*>(vals.data_ptr()), static_cast<int64_t*>(idxs.data_ptr()),
          outer, S, k, largest ? 1 : 0);
    });
    cuda_check_last("topk");
  }
  return {vals, idxs};
}

NDArray flip_nd(const NDArray& a, const std::vector<int>& dims) {
  int nd = a.ndim();
  FlipSpec s{}; s.ndim = nd;
  Shape str = contiguous_strides(a.shape);
  for (int d = 0; d < nd; ++d) { s.shape[d] = a.shape[d]; s.str[d] = str[d]; s.flip[d] = 0; }
  for (int d : dims) { int dd = d < 0 ? d + nd : d; s.flip[dd] = 1; }
  NDArray out(a.shape, a.dtype, a.device);
  int64_t n = a.numel();
  if (n) { DISPATCH_FLOAT(a.dtype, T, { flip_kernel<T><<<nblk(n), kT>>>(static_cast<T*>(a.data_ptr()), static_cast<T*>(out.data_ptr()), s, n); }); cuda_check_last("flip"); }
  return out;
}

NDArray pad_into(const NDArray& small, const Shape& big_shape, int dim, int64_t start) {
  int nd = (int)big_shape.size();
  if (dim < 0) dim += nd;
  NDArray big = NDArray::zeros(big_shape, small.dtype, small.device);
  Shape big_str = contiguous_strides(big_shape);
  DimCopySpec s{}; s.ndim = nd; s.dim = dim; s.off = start;
  Shape istr = contiguous_strides(small.shape);
  for (int d = 0; d < nd; ++d) { s.iter_shape[d] = small.shape[d]; s.iter_str[d] = istr[d]; s.big_str[d] = big_str[d]; }
  int64_t n = small.numel();
  if (n) { DISPATCH_FLOAT(small.dtype, T, { dimcopy_kernel<T, 1><<<nblk(n), kT>>>(static_cast<T*>(small.data_ptr()), nullptr, nullptr, static_cast<T*>(big.data_ptr()), s, n); }); }
  cuda_check_last("pad_into");
  return big;
}

// ------------------------------------------------------------ fused RMSNorm
// One block per row; replaces the 9-launch / 9-alloc unfused chain
// (cast-up, mul, reduce, mul_scalar, add_scalar, pow, mul, mul-weight,
// cast-down) with a single kernel and a single output allocation. Numerics
// mirror the chain: fp32 accumulate, ms = ssum * (1/D), inv = powf(ms+eps,
// -0.5f) (matching ew_pow), out = (x_f32 * inv) * w, one rounding at the
// store. Only the summation ORDER differs (tree vs serial reduce).
namespace {
template <typename XT, typename OT>
__global__ void rms_norm_kernel(const XT* x, const float* w, OT* o,
                                int64_t D, float inv_d, float eps) {
  // fp64 accumulation: costs nothing (bandwidth-bound) and makes the mean
  // MORE accurate than the fp32 serial chain — measured to keep ppl inside
  // the +-0.05 gate where fp32 tree-reduce drifted 0.004 past it.
  __shared__ double sh[kT];
  const XT* xr = x + (int64_t)blockIdx.x * D;
  OT* orow = o + (int64_t)blockIdx.x * D;
  double ss = 0.0;
  for (int64_t c = threadIdx.x; c < D; c += blockDim.x) {
    float v = ld<XT>(xr, c);
    ss += (double)v * (double)v;
  }
  sh[threadIdx.x] = ss;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if ((int)threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
    __syncthreads();
  }
  float inv = powf((float)(sh[0] * (double)inv_d) + eps, -0.5f);
  for (int64_t c = threadIdx.x; c < D; c += blockDim.x)
    st<OT>(orow, c, ld<XT>(xr, c) * inv * w[c]);
}
}  // namespace

// ----------------------------------------------------- fused causal softmax
// Row-wise softmax with the BOTTOM-RIGHT-aligned causal bound computed from
// indices: row i of L queries over S keys sees columns 0..(S-L)+i (the
// rectangular KV-cache/prefix case; square reduces to standard causal). No
// mask tensor exists — masked columns are never read or written non-zero.
// Numerics: replaces eager's [+(-1e4) bias, max, sub, exp(->16-bit tensor),
// sum, div] chain. With the -1e4 bias eager's masked terms underflow to
// exactly 0.0, so skipping them is equivalent; max is order-independent; the
// sum here accumulates UNROUNDED fp32 expf values in fp64 (eager serially
// fp32-sums 16-bit-rounded exps) — strictly more accurate, single rounding
// at the store. One kernel + one alloc, ~3 row passes vs eager's ~6.
namespace {
template <typename T>
__global__ void causal_softmax_kernel(const T* x, T* o, int64_t L, int64_t S) {
  __shared__ float shm[kT];
  __shared__ double shd[kT];
  int64_t row = blockIdx.x;
  int64_t i = row % L;                  // query index within the L rows
  int64_t visible = S - L + i + 1;      // bottom-right causal bound (S >= L)
  const T* xr = x + row * S;
  T* orow = o + row * S;

  float mx = -3.0e38f;
  for (int64_t c = threadIdx.x; c < visible; c += blockDim.x)
    mx = fmaxf(mx, ld<T>(xr, c));
  shm[threadIdx.x] = mx;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if ((int)threadIdx.x < s)
      shm[threadIdx.x] = fmaxf(shm[threadIdx.x], shm[threadIdx.x + s]);
    __syncthreads();
  }
  float m = shm[0];

  double ss = 0.0;
  for (int64_t c = threadIdx.x; c < visible; c += blockDim.x)
    ss += (double)expf(ld<T>(xr, c) - m);
  shd[threadIdx.x] = ss;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if ((int)threadIdx.x < s) shd[threadIdx.x] += shd[threadIdx.x + s];
    __syncthreads();
  }
  float inv = (float)(1.0 / shd[0]);

  for (int64_t c = threadIdx.x; c < S; c += blockDim.x)
    st<T>(orow, c, c < visible ? expf(ld<T>(xr, c) - m) * inv : 0.0f);
}
}  // namespace

NDArray causal_softmax(const NDArray& scores) {
  if (scores.ndim() < 2) throw std::runtime_error("causal_softmax needs >=2D scores");
  int64_t L = scores.shape[scores.ndim() - 2];
  int64_t S = scores.shape[scores.ndim() - 1];
  if (S < L) throw std::runtime_error("causal_softmax: S < L (queries outnumber keys)");
  int64_t rows = scores.numel() / S;
  NDArray out(scores.shape, scores.dtype, scores.device);
  if (rows == 0 || S == 0) return out;
  DISPATCH_FLOAT(scores.dtype, T, {
    causal_softmax_kernel<T><<<(int)rows, kT>>>(
        static_cast<T*>(scores.data_ptr()), static_cast<T*>(out.data_ptr()), L, S);
  });
  cuda_check_last("causal_softmax");
  return out;
}

NDArray rms_norm(const NDArray& x, const NDArray& w, double eps, DType out_dtype) {
  if (x.ndim() < 1) throw std::runtime_error("rms_norm needs >=1D input");
  if (w.dtype != DType::Float32)
    throw std::runtime_error("rms_norm expects fp32 weight (chain stores fp32)");
  int64_t D = x.shape[x.ndim() - 1];
  if (w.numel() != D) throw std::runtime_error("rms_norm weight/dim mismatch");
  int64_t rows = x.numel() / D;
  NDArray out(x.shape, out_dtype, x.device);
  if (rows == 0 || D == 0) return out;
  DISPATCH_FLOAT(x.dtype, XT, {
    DISPATCH_FLOAT(out_dtype, OT, {
      rms_norm_kernel<XT, OT><<<(int)rows, kT>>>(
          static_cast<XT*>(x.data_ptr()), static_cast<float*>(w.data_ptr()),
          static_cast<OT*>(out.data_ptr()), D, 1.0f / (float)D, (float)eps);
    });
  });
  cuda_check_last("rms_norm");
  return out;
}

// -------------------------------------------- fused Gated DeltaNet step
// One decode token through one GDN layer in ONE launch (inference-only).
// Folds the l2norm(q,k), gate math (sigmoid / softplus / exp) and the
// delta-rule state update + readout that the composed path spends ~40
// Python-dispatched launches on. Reference math = HF Qwen3.5
// torch_recurrent_gated_delta_rule (decay applied BEFORE the delta
// correction; l2norm uses SUM + eps; q scaled 1/sqrt(Dk) after norm).
//
//   q,k:  (B, Hk, Dk) fp32  RAW post-conv heads (kernel maps v-head h
//                            to kq-head h / (H/Hk) and l2norms inside)
//   v:    (B, H,  Dv) fp32
//   a,b:  (B, H)      fp32  raw in_proj_a / in_proj_b outputs
//   A_neg,dt_bias: H fp32 elements (-exp(A_log), dt_bias)
//   state:(B, H, Dk, Dv) fp32 — FUNCTIONAL: a NEW state is returned
//   out:  (B, H, Dv) fp32
//
// Launch: grid (Dv/4, H, B), block (32, 4) — each warp owns one state
// COLUMN j; lane l holds rows l, l+32, ... in registers (Dk<=256).
__global__ void gated_delta_step_kernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ a,
    const float* __restrict__ b, const float* __restrict__ A_neg,
    const float* __restrict__ dtb, const float* __restrict__ state_in,
    float* __restrict__ state_out, float* __restrict__ out,
    int H, int Hk, int Dk, int Dv) {
  const int j = blockIdx.x * blockDim.y + threadIdx.y;   // state column
  const int h = blockIdx.y, bb = blockIdx.z, lane = threadIdx.x;
  if (j >= Dv) return;
  const int hk = h / (H / Hk);
  const float* kr = k + ((int64_t)bb * Hk + hk) * Dk;
  const float* qr = q + ((int64_t)bb * Hk + hk) * Dk;
  const int R = Dk >> 5;                                 // rows per lane
  float kl[8], ql[8], sl[8];
  float sk = 0.f, sq = 0.f;
  for (int r = 0; r < R; ++r) {
    const int i = lane + (r << 5);
    kl[r] = kr[i];
    ql[r] = qr[i];
    sk += kl[r] * kl[r];
    sq += ql[r] * ql[r];
  }
  for (int off = 16; off; off >>= 1) {
    sk += __shfl_down_sync(0xffffffffu, sk, off);
    sq += __shfl_down_sync(0xffffffffu, sq, off);
  }
  sk = __shfl_sync(0xffffffffu, sk, 0);
  sq = __shfl_sync(0xffffffffu, sq, 0);
  const float rk = rsqrtf(sk + 1e-6f);
  const float rq = rsqrtf(sq + 1e-6f) * rsqrtf((float)Dk);
  // gates (redundant per thread — scalar work)
  const float av = a[(int64_t)bb * H + h] + dtb[h];
  const float softplus = fmaxf(av, 0.f) + log1pf(expf(-fabsf(av)));
  const float alpha = expf(A_neg[h] * softplus);
  const float beta = 1.f / (1.f + expf(-b[(int64_t)bb * H + h]));
  // state column j: decay-first delta rule + readout
  const float* sin_ = state_in + (((int64_t)bb * H + h) * Dk) * Dv + j;
  float* sout_ = state_out + (((int64_t)bb * H + h) * Dk) * Dv + j;
  float kv = 0.f;
  for (int r = 0; r < R; ++r) {
    sl[r] = sin_[(int64_t)(lane + (r << 5)) * Dv];
    kv += sl[r] * (kl[r] * rk);
  }
  for (int off = 16; off; off >>= 1)
    kv += __shfl_down_sync(0xffffffffu, kv, off);
  kv = __shfl_sync(0xffffffffu, kv, 0);
  const float delta =
      (v[((int64_t)bb * H + h) * Dv + j] - alpha * kv) * beta;
  float o = 0.f;
  for (int r = 0; r < R; ++r) {
    sl[r] = alpha * sl[r] + (kl[r] * rk) * delta;
    sout_[(int64_t)(lane + (r << 5)) * Dv] = sl[r];
    o += sl[r] * (ql[r] * rq);
  }
  for (int off = 16; off; off >>= 1)
    o += __shfl_down_sync(0xffffffffu, o, off);
  if (lane == 0) out[((int64_t)bb * H + h) * Dv + j] = o;
}

std::pair<NDArray, NDArray> gated_delta_step(
    const NDArray& q, const NDArray& k, const NDArray& v,
    const NDArray& a, const NDArray& b, const NDArray& A_neg,
    const NDArray& dt_bias, const NDArray& state) {
  const NDArray* ins[8] = {&q, &k, &v, &a, &b, &A_neg, &dt_bias, &state};
  for (int t = 0; t < 8; ++t)
    if (ins[t]->dtype != DType::Float32)
      throw std::runtime_error("gated_delta_step: all inputs must be fp32");
  if (state.ndim() != 4 || v.ndim() != 3 || q.ndim() != 3)
    throw std::runtime_error("gated_delta_step: bad ranks");
  const int B = (int)state.shape[0], H = (int)state.shape[1];
  const int Dk = (int)state.shape[2], Dv = (int)state.shape[3];
  const int Hk = (int)q.shape[1];
  if (Dk % 32 || Dk > 256 || Dv % 4)
    throw std::runtime_error("gated_delta_step: Dk%32==0<=256, Dv%4==0");
  if ((int)v.shape[1] != H || H % Hk)
    throw std::runtime_error("gated_delta_step: head mismatch");
  NDArray out({(int64_t)B, (int64_t)H, (int64_t)Dv}, DType::Float32,
              state.device);
  NDArray new_state(state.shape, DType::Float32, state.device);
  dim3 grid(Dv / 4, H, B), block(32, 4);
  gated_delta_step_kernel<<<grid, block>>>(
      static_cast<float*>(q.data_ptr()), static_cast<float*>(k.data_ptr()),
      static_cast<float*>(v.data_ptr()), static_cast<float*>(a.data_ptr()),
      static_cast<float*>(b.data_ptr()),
      static_cast<float*>(A_neg.data_ptr()),
      static_cast<float*>(dt_bias.data_ptr()),
      static_cast<float*>(state.data_ptr()),
      static_cast<float*>(new_state.data_ptr()),
      static_cast<float*>(out.data_ptr()), H, Hk, Dk, Dv);
  cuda_check_last("gated_delta_step");
  return {out, new_state};
}

}  // namespace tc
