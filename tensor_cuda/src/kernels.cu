// kernels.cu: device memory, NDArray, and hand-written CUDA kernels.
//
// Kernels are dtype-generic: values are loaded and computed in fp32 and stored
// back in the array's dtype, so fp32 and fp16 share one implementation.

#include "tc/core.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>  // EXP-APA-3: WMMA HMMA for the Q-tiled attention skeleton

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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
  if (numel_of(new_shape) != numel()) {
    throw std::runtime_error("reshape: element count mismatch");
  }
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
    case U_GELU_EXACT: r = 0.5f * x * (1.f + erff(0.7071067811865475f * x)); break;
    case U_ERF: r = erff(x); break;
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

// NOTE: a fused SwiGLU kernel (silu(gate)·up, one launch — kernel-opt plan
// Phase 5) was tried and REVERTED: kernel-level +38-40% but e2e +0.7% < the
// registered 2% loop gate; KERNEL_OPT_IMPLEMENTATION_LEDGER 2026-07-07.
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

// Trailing-axis fast path: one BLOCK per row (strided loads + shared-
// memory tree). The generic kernel below parallelizes over OUTPUTS
// only — for few-outputs/large-reduction shapes (softmax max/sum over
// the key axis: 16 outputs x 700 elements) that is 16 serial threads
// on the whole GPU, measured 533us for an 11K-element max.
template <typename T, int MODE>  // 0=sum 1=max 2=min 3=prod
__global__ void reduce_lastdim_kernel(const T* __restrict__ in,
                                      T* __restrict__ out,
                                      int64_t R, int64_t n_rows) {
  int64_t row = blockIdx.x;
  if (row >= n_rows) return;
  const T* p = in + row * R;
  float acc = MODE == 0 ? 0.f
            : (MODE == 1 ? -3.4e38f : (MODE == 2 ? 3.4e38f : 1.f));
  for (int64_t i = threadIdx.x; i < R; i += blockDim.x) {
    float v = ld<T>(p, i);
    if (MODE == 0) acc += v;
    else if (MODE == 1) acc = v > acc ? v : acc;
    else if (MODE == 2) acc = v < acc ? v : acc;
    else acc *= v;
  }
  __shared__ float sh[256];
  sh[threadIdx.x] = acc;
  __syncthreads();
  for (int off = blockDim.x >> 1; off; off >>= 1) {
    if (threadIdx.x < off) {
      float o = sh[threadIdx.x + off];
      if (MODE == 0) sh[threadIdx.x] += o;
      else if (MODE == 1) sh[threadIdx.x] =
          o > sh[threadIdx.x] ? o : sh[threadIdx.x];
      else if (MODE == 2) sh[threadIdx.x] =
          o < sh[threadIdx.x] ? o : sh[threadIdx.x];
      else sh[threadIdx.x] *= o;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) st<T>(out, row, sh[0]);
}

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
  // trailing-block fast path: reduced axes form the contiguous tail of
  // the shape -> rows are contiguous, one block per row
  bool trailing = true;
  {
    int first_red = nd;
    for (int d = 0; d < nd; ++d) if (s.reduced[d]) { first_red = d; break; }
    for (int d = first_red; d < nd; ++d) if (!s.reduced[d]) trailing = false;
    if (first_red == nd) trailing = false;       // nothing reduced
  }
  if (out_n > 0 && trailing && s.red_size > 1) {
    DISPATCH_FLOAT(a.dtype, T, {
      if (mode == 1) reduce_lastdim_kernel<T, 1><<<(unsigned)out_n, 256>>>(static_cast<T*>(a.data_ptr()), static_cast<T*>(out.data_ptr()), s.red_size, out_n);
      else if (mode == 2) reduce_lastdim_kernel<T, 2><<<(unsigned)out_n, 256>>>(static_cast<T*>(a.data_ptr()), static_cast<T*>(out.data_ptr()), s.red_size, out_n);
      else if (mode == 3) reduce_lastdim_kernel<T, 3><<<(unsigned)out_n, 256>>>(static_cast<T*>(a.data_ptr()), static_cast<T*>(out.data_ptr()), s.red_size, out_n);
      else reduce_lastdim_kernel<T, 0><<<(unsigned)out_n, 256>>>(static_cast<T*>(a.data_ptr()), static_cast<T*>(out.data_ptr()), s.red_size, out_n);
    });
    cuda_check_last("reduce_lastdim");
  } else if (out_n > 0) {
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
  param.mark_modified();
  m.mark_modified();
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
  param.mark_modified();
  m.mark_modified();
  v.mark_modified();
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
  param.mark_modified();
  sq.mark_modified();
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
  param.mark_modified();
  acc.mark_modified();
}
void scale_(NDArray& param, double s) {
  int64_t n = param.numel();
  if (!n) return;
  DISPATCH_FLOAT(param.dtype, T, {
    scale_kernel<T><<<nblk(n), kT>>>(static_cast<T*>(param.data_ptr()), n, (float)s);
  });
  cuda_check_last("scale_");
  param.mark_modified();
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
constexpr int TC_APA_MAXD = 512;   // bumped 256->512 for Gemma 4 global
                                   // (MQA head_dim 512); nvcc-verified to
                                   // compile clean at D=512 (42 regs, 0
                                   // spills — acc[] is already local, the
                                   // register-cliff fear was imaginary)
// DMAX is the compile-time size of the per-thread acc[] register array. Sizing it
// to the actual head_dim (64/128) instead of the 256 worst case keeps acc[] in
// registers/L1 instead of spilling to local (global-backed) memory — a large win
// on the selective kernel's hot inner loop. The launcher dispatches the smallest
// DMAX >= D. qsh/osh stay TC_APA_MAXD: they are __shared__, not per-thread, so
// they cost shared memory (cheap, plentiful) not registers.
// ----------------------------------------------------------------------------
// Warp-cooperative key processing (kernel-opt Addendum 2, workstream A5).
//
// Problem (ncu receipt, docs/KERNEL_OPT_PLAN_ADDENDUM_2.md): the prior loop
// (`for j = tid; j < s_max; j += nt`) gives each THREAD its own key row. At
// any fixed dimension offset d, the 32 lanes of a warp are then reading 32
// DIFFERENT key rows, D floats apart — one cache-line sector per lane instead
// of one sector serving the whole warp (25.3 sectors/request measured,
// ~4 optimal; 88.1% of warp stall cycles on L1TEX scoreboard waits).
//
// Fix: assign KEYS to WARPS, not threads. `for j = warp; j < s_max; j += nwarp`
// — each of the block's `nwarp` warps (nt=128 -> nwarp=4) owns a disjoint
// strided subset of keys, exactly mirroring the old per-thread strided
// assignment one level up. Within one warp's key j, the 32 lanes split the D
// dimension: lane `lane` reads elements `lane, lane+32, lane+64, ...` of the
// SAME key row. At a fixed unrolled step every lane's address differs by one
// element (4 bytes) — a single coalesced 128B-or-less transaction serves the
// whole warp instead of 32. A `__shfl_down_sync` butterfly reduces the 32
// partial products to the per-warp dot, then `__shfl_sync(mask, v, 0)`
// broadcasts it back to all 32 lanes (every lane needs the scalar score to
// keep its own online-softmax state in lockstep — cheap registers-only
// broadcast, no shared memory, no extra syncthreads).
//
// Per-DMAX lane-to-element mapping (all D in {64,128,256,512} divide evenly
// by warpSize=32, so no ragged last-lane case exists for the score dim):
//   D=64  -> 2 elements/lane   D=128 -> 4/lane   D=256 -> 8/lane   D=512 -> 16/lane
// Plain scalar `ld<T>` reads are kept (not float2/half2 vector loads): the
// coalescing win already comes from 32 lanes hitting one contiguous 128B (f32)
// or 64B (f16/bf16) span per step — vectorizing further would halve the
// instruction count but the access is already fully coalesced, and a vector
// path would need three more ld<T> specializations (half2/bfloat162 packed
// loads through a currently scalar-only ld<T> abstraction) for a gain ncu
// would show as "instruction count", not "sectors/request" (already ~1
// optimal post-fix). Left as a documented follow-up, not required here.
//
// Value accumulation shares the exact same per-thread-per-key stall pattern
// (`for d = 0..VD: acc[d] = ...; ld<T>(vj, d)` — same uncoalesced row-per-
// thread read). Folded into the SAME warp-cooperative loop: since every lane
// in the warp now holds an IDENTICAL (m, l, score) after the broadcast, lane
// ownership of `acc` is split across the warp instead of replicated — lane
// `lane` owns acc slots `d = lane, lane+32, ...` (register footprint per
// thread drops from VD to ceil(VD/32), e.g. 512 -> 16 — a register WIN, not
// just parity) and reads `v[d]` at exactly the offsets it owns (same
// coalesced-across-lanes pattern as the score dot). No cross-lane reduction
// needed for acc at all (each lane's slice is disjoint, not a partial sum),
// which is a genuine simplification over the old per-thread-full-acc + final
// warp-shfl-reduce path.
//
// APA invariant preserved exactly: same bulk dot per key (now warp-summed
// instead of thread-summed — different FLOATING-POINT REDUCTION ORDER,
// same mathematical sum), same sum/sumsq -> mean/var -> z-score thr, same
// |bulk*scale| >= thr refine decision, same online-softmax recurrence
// (m, l updates use the identical formulas, just computed redundantly on
// all 32 lanes of the owning warp instead of once per thread). Only WHICH
// lanes compute what changed; the algorithm is bit-for-bit the same modulo
// float reassociation, exactly the class of change the sharpened invariant
// (ledger 2026-07-07) explicitly allows.
//
// DISPATCH (A5 lead iteration, 2026-07-07): the warp-cooperative path is NOT
// unconditional. Indicative measurement showed it wins big everywhere the
// D-loop is long or the grid is full (all prefill; decode at D>=128:
// +58%..+478%) but LOSES at D=64 decode (GPT-OSS geometry, -14%..-29%): with
// only 2 elements/lane per key, the shuffle/broadcast overhead is not
// amortized and warp-granularity key streaming (4 independent streams/block)
// surrenders the ILP that per-thread streaming (128 streams/block) had —
// and that shape was never memory-pattern-bound in the first place (Phase
// 0.2 ncu: compute/launch-shape-bound at 3% SM throughput; A1 split-K was
// its fix). So BOTH paths are compiled, selected by the WCOOP template
// parameter, and the launchers choose per launch:
//   - DMAX == 64 && decode-shaped (apa_selective_decode_shaped: L==1 and
//     rows < 2*TC_APA_SM_COUNT, mirroring the split-K heuristic's notion of
//     decode)                         -> WCOOP=false (per-thread, pre-A5 code)
//   - DMAX == 64 otherwise (prefill)  -> WCOOP=true
//   - DMAX >= 128 (all shapes)        -> WCOOP=true
//   - split-K stats/split kernels at DMAX==64 -> WCOOP=false ALWAYS (that
//     path is decode-only by construction, dispatch-gated on L==1)
// The WCOOP=false branches below are the exact pre-A5 per-thread code,
// restored verbatim so the regressed shapes return to their pre-A5 timings
// and bit-identical outputs.
// ----------------------------------------------------------------------------

template <typename T, int DMAX, bool WCOOP>
__global__ void apa_selective_kernel(
    const T* q, const T* k, const T* kq, const T* v, T* out,
    int B, int H, int L, int S, int D, int VD, float scale, float zthr,
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
  const unsigned FULL = 0xffffffffu;
  int lane = tid & 31;
  int warp = tid >> 5;
  int nwarp = nt >> 5;              // == 4 for nt=128

  const T* qrow = q + (int64_t)row * D;
  int64_t kvbh = (int64_t)b * KVH + kv_h;
  const T* kbase = k + kvbh * S * D;
  const T* kqbase = kq + kvbh * S * D;
  const T* vbase = v + kvbh * S * VD;

  // Load this query vector into shared memory (D <= DMAX <= TC_APA_MAXD).
  __shared__ float qsh[DMAX];
  for (int d = tid; d < D; d += nt) qsh[d] = ld<T>(qrow, d);
  __syncthreads();

  // BOTTOM-RIGHT causal: query row i sits at ABSOLUTE key index
  // (S-L)+i (the prefill-continuation / cache regime where S>L), so it
  // sees keys 0..(S-L)+i. The old s_max=i+1 was top-left — correct only
  // at S==L (square, no cache), and SILENTLY blinds queries to the most
  // recent S-L keys otherwise (the 121->11M ppl bug class). Reduces to
  // i+1 exactly when S==L.
  int s_max = is_causal ? ((S - L) + i + 1) : S;
  __shared__ float red[256];

  // Pass 1: bulk scores -> sum/sumsq of |bulk| for the z-score threshold.
  // WCOOP=true: each warp owns a strided subset of keys; within a key, lanes
  // split D contiguously and shfl-reduce the dot (see block comment).
  // WCOOP=false: pre-A5 per-thread strided keys (each thread's sum/sumsq is a
  // genuine disjoint partial).
  float sum = 0.f, sumsq = 0.f;
  if constexpr (WCOOP) {
    for (int j = warp; j < s_max; j += nwarp) {
      const T* kqj = kqbase + (int64_t)j * D;
      float dot = 0.f;
      for (int d = lane; d < D; d += 32) dot += qsh[d] * ld<T>(kqj, d);
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1) dot += __shfl_down_sync(FULL, dot, off);
      dot = __shfl_sync(FULL, dot, 0);   // broadcast the warp's dot to all lanes
      float a = fabsf(dot * scale);
      sum += a; sumsq += a * a;
    }
  } else {
    for (int j = tid; j < s_max; j += nt) {
      const T* kqj = kqbase + (int64_t)j * D;
      float dot = 0.f;
      for (int d = 0; d < D; ++d) dot += qsh[d] * ld<T>(kqj, d);
      float a = fabsf(dot * scale);
      sum += a; sumsq += a * a;
    }
  }
  // WCOOP: sum/sumsq are per-WARP totals, identical across all 32 lanes of
  // the owning warp (redundant, not partial) — only lane 0 contributes, else
  // each warp's contribution is overcounted 32x. Per-thread: all contribute.
  red[tid] = (WCOOP && lane != 0) ? 0.f : sum; __syncthreads();
  for (int off = nt / 2; off > 0; off >>= 1) { if (tid < off) red[tid] += red[tid+off]; __syncthreads(); }
  float total = red[0]; __syncthreads();
  red[tid] = (WCOOP && lane != 0) ? 0.f : sumsq; __syncthreads();
  for (int off = nt / 2; off > 0; off >>= 1) { if (tid < off) red[tid] += red[tid+off]; __syncthreads(); }
  float total_sq = red[0]; __syncthreads();

  float cnt = (float)s_max;
  float mean = total / cnt;
  float var = total_sq / cnt - mean * mean;
  float thr = mean + zthr * sqrtf(fmaxf(var, 0.f));

  // Pass 2: SINGLE pass with an online softmax (FlashAttention-style).
  // WCOOP=true: per-WARP state, all 32 lanes carrying identical (m, l)
  // (redundant scalar work, cheap) while VALUE accumulation is split across
  // lanes: lane `lane` owns acc slots d = lane, lane+32, ... (ceil(VD/32)
  // registers). WCOOP=false: pre-A5 per-thread state, full acc[VD] each.
  float m = -1e30f, l = 0.f;
  constexpr int ACCN = WCOOP ? ((DMAX + 31) / 32) : DMAX;
  float acc[ACCN];
  if constexpr (WCOOP) {
    #pragma unroll
    for (int dl = 0; dl < ACCN; ++dl) acc[dl] = 0.f;
  } else {
    for (int d = 0; d < VD; ++d) acc[d] = 0.f;
  }

  if constexpr (WCOOP) {
    for (int j = warp; j < s_max; j += nwarp) {
      const T* kqj = kqbase + (int64_t)j * D;
      float bulk = 0.f;
      for (int d = lane; d < D; d += 32) bulk += qsh[d] * ld<T>(kqj, d);
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1) bulk += __shfl_down_sync(FULL, bulk, off);
      bulk = __shfl_sync(FULL, bulk, 0);
      bulk *= scale;
      float score;
      if (fabsf(bulk) >= thr) {
        const T* kj = kbase + (int64_t)j * D;
        float ex = 0.f;
        for (int d = lane; d < D; d += 32) ex += qsh[d] * ld<T>(kj, d);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) ex += __shfl_down_sync(FULL, ex, off);
        ex = __shfl_sync(FULL, ex, 0);
        score = ex * scale;
      } else {
        score = bulk;
      }
      // online softmax update — identical formula on every lane (redundant,
      // not partitioned: m/l are scalars, not worth splitting across lanes).
      float m_new = fmaxf(m, score);
      float corr = __expf(m - m_new);
      float w = __expf(score - m_new);
      l = l * corr + w;
      const T* vj = vbase + (int64_t)j * VD;
      #pragma unroll
      for (int dl = 0; dl < ACCN; ++dl) {
        int d = lane + dl * 32;
        if (d < VD) acc[dl] = acc[dl] * corr + w * ld<T>(vj, d);
      }
      m = m_new;
    }
  } else {
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
      const T* vj = vbase + (int64_t)j * VD;
      for (int d = 0; d < VD; ++d) acc[d] = acc[d] * corr + w * ld<T>(vj, d);
      m = m_new;
    }
  }

  // Merge the per-thread (WCOOP=false) or per-warp-replicated (WCOOP=true)
  // online-softmax states. m's max-reduction is duplicate-safe either way;
  // l is a SUM, so under WCOOP only lane 0 of each warp contributes (every
  // lane holds the identical warp-total — 32x overcount otherwise).
  red[tid] = m; __syncthreads();
  for (int off = nt / 2; off > 0; off >>= 1) { if (tid < off) red[tid] = fmaxf(red[tid], red[tid+off]); __syncthreads(); }
  float gmax = red[0]; __syncthreads();

  float rescale = __expf(m - gmax);
  // denom
  red[tid] = (WCOOP && lane != 0) ? 0.f : (l * rescale); __syncthreads();
  for (int off = nt / 2; off > 0; off >>= 1) { if (tid < off) red[tid] += red[tid+off]; __syncthreads(); }
  float denom = red[0]; __syncthreads();
  float inv = denom > 0.f ? 1.f / denom : 0.f;

  // Combine per-warp partials per dim. WCOOP=true: acc is lane-DISJOINT
  // (each lane owns distinct d's), so each lane just writes its owned slots
  // into wpart[warp][d] — no shfl-reduce needed. WCOOP=false (pre-A5): each
  // thread holds a full partial acc[d]; warp shfl-reduce then lane 0 writes.
  // The cross-warp combine below is common to both.
  __shared__ float osh[DMAX];
  __shared__ float wpart[4][DMAX];   // [warp][dim] partials; nt=128 -> 4 warps
  if constexpr (WCOOP) {
    #pragma unroll
    for (int dl = 0; dl < ACCN; ++dl) {
      int d = lane + dl * 32;
      if (d < VD) wpart[warp][d] = acc[dl] * rescale;
    }
  } else {
    for (int d = 0; d < VD; ++d) {
      float val = acc[d] * rescale;
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(FULL, val, off);
      if (lane == 0) wpart[warp][d] = val;
    }
  }
  __syncthreads();
  for (int d = tid; d < VD; d += nt) {
    float s = 0.f;
    for (int w = 0; w < nwarp; ++w) s += wpart[w][d];
    osh[d] = s;
  }
  __syncthreads();

  T* orow = out + (int64_t)row * VD;
  for (int d = tid; d < VD; d += nt) st<T>(orow, d, osh[d] * inv);
}

// ============================================================================
// APA SELECTIVE — SPLIT-K DECODE PATH (kernel-opt Addendum 1, workstream A1).
//
// Problem (ncu receipt, Phase 0.2): apa_selective_kernel launches one block
// per (b,h,query-row) = B*H*L blocks. At decode L=1, that is only B*H blocks
// (16 on a real shape) against 56 SMs -> 8.33% achieved occupancy, "0.0 full
// waves" — most of the chip sits idle while a handful of blocks stride over
// a long S alone. Prefill (L large) already fills the grid; this path is
// decode-only by construction (see the dispatch heuristic below).
//
// Fix: flash-decoding-style split-K. The APA invariant (docs/
// KERNEL_OPT_PLAN_ADDENDUM_1.md) requires the SAME threshold as the existing
// kernel — derived from full-key-range bulk-score statistics — and the SAME
// per-key bulk-vs-refine decision and softmax math. Splitting the key range
// across blocks would make each block see only a slice of the keys, which
// cannot reproduce a global mean/var without a cross-block reduction. So the
// threshold is computed in an separate, dedicated stage over the FULL key
// range (mirroring apa_selective_kernel's pass 1 exactly, same recompute-not-
// cache structure as the fused kernel), and only pass 2 (already a pure
// per-key function of the precomputed threshold) is split across blocks:
//
//   [[stats kernel]]  1 block/row, all keys -> thr[row]   (== fused pass 1)
//   [[split kernel]]  P blocks/row, keys strided within a partition -> one
//                     online-softmax partial (m, l, acc[VD]) per (row, part)
//                     (== fused pass 2, restricted to the partition's keys)
//   [[merge kernel]]  1 block/row, reduce the P partials with the standard
//                     online-softmax rescale (exp(m_i - m_max)) -> same
//                     result as the fused kernel's own end-of-pass-2 merge,
//                     just one extra reduction level (partitions instead of
//                     threads-within-a-block).
//
// This is semantically identical to today's kernel: same bulk dots, same
// z-score threshold, same |bulk|>=thr selection, same online-softmax math,
// just computed by three smaller launches whose combined grid (rows *
// (1 + P + 1) blocks, dominated by rows*P) fills the SM count instead of
// leaving it at rows blocks total.
//
// Sink handling: apa_selective_sink_kernel folds the learned sink logit into
// (m, l) via tid==0 AFTER pass 2's per-thread loop but BEFORE the block's own
// merge reduction (kernels.cu, "GPT-OSS learned sink" comment above). Folding
// it once per partition in the split kernel would double-count it P times;
// per the addendum's constraint #4, it must fold exactly once, at the merge
// stage — so the split-K sink variant folds the sink in apa_selective_
// merge_sink_kernel, once, after all partitions are combined into (gmax, l).
//
// Only the decode grid-underfill case is addressed. Prefill and the training
// fwd/bwd kernels above/below are untouched (out of scope, addendum #5/#6).
// ============================================================================

// Number of SMs on the target part (RTX 4070 SUPER / Ada, SM 8.9) this
// codebase already documents as its baseline (see the TC_APA_MAXD comment
// above and the ledger's 56-SM ncu receipts). Not read from device props:
// the dispatch heuristic only needs a conservative order-of-magnitude
// estimate of "is the grid small relative to the chip", and a compile-time
// constant keeps the launcher branch-free and avoids a cudaGetDeviceProperties
// round trip on every call. If this code is ported to a part with a very
// different SM count, this constant (and the heuristic below) should move to
// a runtime cudaGetDeviceProperties query — left as a follow-up, not required
// for the Ada target this plan is scoped to.
constexpr int TC_APA_SM_COUNT = 56;

// Partition width: each split-K block strides over this many keys at most
// per partition before the partition boundary — chosen as a multiple of the
// warp-strided loop width (nt=128, so 128 keys/thread-iteration) so the
// per-partition key count divides evenly into full strided passes with no
// short final iteration doing wasted work across most threads.
constexpr int TC_APA_SPLITK_PART_KEYS = 128 * 16;  // 2048 keys/partition

// apa_selective_stats_kernel: identical arithmetic to apa_selective_kernel's
// pass 1 (bulk dot over ALL keys in [0, s_max) -> mean/var of |bulk| -> z-
// score threshold). One block per (b,h,query-row); writes thr[row] to a
// small workspace instead of continuing on to pass 2 in the same block. This
// is the "global stats over the full key range" stage the addendum requires.
template <typename T, int DMAX, bool WCOOP>
__global__ void apa_selective_stats_kernel(
    const T* q, const T* kq, float* thr_out,
    int B, int H, int L, int S, int D, float scale, float zthr,
    int is_causal, int KVH, int group) {
  int row = blockIdx.x;
  int i = row % L;
  int bh = row / L;
  int b = bh / H;
  int h = bh % H;
  int kv_h = h / group;
  int tid = threadIdx.x;
  int nt = blockDim.x;
  const unsigned FULL = 0xffffffffu;
  int lane = tid & 31;
  int warp = tid >> 5;
  int nwarp = nt >> 5;

  const T* qrow = q + (int64_t)row * D;
  int64_t kvbh = (int64_t)b * KVH + kv_h;
  const T* kqbase = kq + kvbh * S * D;

  __shared__ float qsh[DMAX];
  for (int d = tid; d < D; d += nt) qsh[d] = ld<T>(qrow, d);
  __syncthreads();

  int s_max = is_causal ? ((S - L) + i + 1) : S;
  __shared__ float red[256];

  // A5 dual-path (see apa_selective_kernel's DISPATCH comment): WCOOP=true
  // is warp-cooperative key loads; WCOOP=false is the pre-A5 per-thread
  // loop. Identical arithmetic either way, only the lane<->element
  // assignment and float reduction order differ.
  float sum = 0.f, sumsq = 0.f;
  if constexpr (WCOOP) {
    for (int j = warp; j < s_max; j += nwarp) {
      const T* kqj = kqbase + (int64_t)j * D;
      float dot = 0.f;
      for (int d = lane; d < D; d += 32) dot += qsh[d] * ld<T>(kqj, d);
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1) dot += __shfl_down_sync(FULL, dot, off);
      dot = __shfl_sync(FULL, dot, 0);
      float a = fabsf(dot * scale);
      sum += a; sumsq += a * a;
    }
  } else {
    for (int j = tid; j < s_max; j += nt) {
      const T* kqj = kqbase + (int64_t)j * D;
      float dot = 0.f;
      for (int d = 0; d < D; ++d) dot += qsh[d] * ld<T>(kqj, d);
      float a = fabsf(dot * scale);
      sum += a; sumsq += a * a;
    }
  }
  // WCOOP: per-warp totals duplicated across the warp's 32 lanes — only
  // lane 0 contributes (else 32x overcount). Per-thread: all contribute.
  red[tid] = (WCOOP && lane != 0) ? 0.f : sum; __syncthreads();
  for (int off = nt / 2; off > 0; off >>= 1) { if (tid < off) red[tid] += red[tid+off]; __syncthreads(); }
  float total = red[0]; __syncthreads();
  red[tid] = (WCOOP && lane != 0) ? 0.f : sumsq; __syncthreads();
  for (int off = nt / 2; off > 0; off >>= 1) { if (tid < off) red[tid] += red[tid+off]; __syncthreads(); }
  float total_sq = red[0]; __syncthreads();

  if (tid == 0) {
    float cnt = (float)s_max;
    float mean = total / cnt;
    float var = total_sq / cnt - mean * mean;
    thr_out[row] = mean + zthr * sqrtf(fmaxf(var, 0.f));
  }
}

// apa_selective_split_kernel: pass 2 restricted to one partition of keys.
// Identical per-key logic to apa_selective_kernel's pass 2 (bulk dot; if
// |bulk*scale| >= thr, replace with the exact dot; online-softmax update),
// using the PRECOMPUTED thr[row] from the stats kernel above (same value the
// fused kernel would have derived itself). Grid is (rows, num_parts): block
// (row, p) covers keys [p*part_keys, min(s_max, (p+1)*part_keys)) with the
// same nt-strided inner loop as the fused kernel, just bounded to the
// partition's slice. Writes one online-softmax partial (m, l, acc[VD]) to
// workspace row (row*num_parts + p) — merged by apa_selective_merge_kernel.
template <typename T, int DMAX, bool WCOOP>
__global__ void apa_selective_split_kernel(
    const T* q, const T* k, const T* kq, const T* v, const float* thr_in,
    float* part_m, float* part_l, float* part_acc,
    int B, int H, int L, int S, int D, int VD, float scale,
    int is_causal, int KVH, int group, int num_parts, int part_keys) {
  int row = blockIdx.x;
  int part = blockIdx.y;
  int i = row % L;
  int bh = row / L;
  int b = bh / H;
  int h = bh % H;
  int kv_h = h / group;
  int tid = threadIdx.x;
  int nt = blockDim.x;
  const unsigned FULL = 0xffffffffu;
  int lane = tid & 31;
  int warp = tid >> 5;
  int nwarp = nt >> 5;

  int s_max = is_causal ? ((S - L) + i + 1) : S;
  int j0 = part * part_keys;
  int j1 = j0 + part_keys;
  if (j1 > s_max) j1 = s_max;

  int64_t part_row = (int64_t)row * num_parts + part;
  float* pacc = part_acc + part_row * VD;
  if (j0 >= j1) {
    // Empty partition (s_max shorter than the full split-K grid at small
    // causal prefixes): contribute the online-softmax identity element so
    // the merge kernel's reduction is a no-op for this partition, exactly
    // as if this partition's stride of the fused kernel's loop had done
    // zero iterations.
    if (tid == 0) { part_m[part_row] = -1e30f; part_l[part_row] = 0.f; }
    for (int d = tid; d < VD; d += nt) pacc[d] = 0.f;
    return;
  }

  const T* qrow = q + (int64_t)row * D;
  int64_t kvbh = (int64_t)b * KVH + kv_h;
  const T* kbase = k + kvbh * S * D;
  const T* kqbase = kq + kvbh * S * D;
  const T* vbase = v + kvbh * S * VD;
  float thr = thr_in[row];

  __shared__ float qsh[DMAX];
  for (int d = tid; d < D; d += nt) qsh[d] = ld<T>(qrow, d);
  __syncthreads();

  // A5 dual-path (see apa_selective_kernel's DISPATCH comment). WCOOP=true:
  // keys assigned to warps within [j0, j1), value accumulation lane-split.
  // WCOOP=false: pre-A5 per-thread strided keys, full acc[VD] per thread.
  float m = -1e30f, l = 0.f;
  constexpr int ACCN = WCOOP ? ((DMAX + 31) / 32) : DMAX;
  float acc[ACCN];
  if constexpr (WCOOP) {
    #pragma unroll
    for (int dl = 0; dl < ACCN; ++dl) acc[dl] = 0.f;
  } else {
    for (int d = 0; d < VD; ++d) acc[d] = 0.f;
  }

  if constexpr (WCOOP) {
    for (int j = j0 + warp; j < j1; j += nwarp) {
      const T* kqj = kqbase + (int64_t)j * D;
      float bulk = 0.f;
      for (int d = lane; d < D; d += 32) bulk += qsh[d] * ld<T>(kqj, d);
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1) bulk += __shfl_down_sync(FULL, bulk, off);
      bulk = __shfl_sync(FULL, bulk, 0);
      bulk *= scale;
      float score;
      if (fabsf(bulk) >= thr) {
        const T* kj = kbase + (int64_t)j * D;
        float ex = 0.f;
        for (int d = lane; d < D; d += 32) ex += qsh[d] * ld<T>(kj, d);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) ex += __shfl_down_sync(FULL, ex, off);
        ex = __shfl_sync(FULL, ex, 0);
        score = ex * scale;
      } else {
        score = bulk;
      }
      float m_new = fmaxf(m, score);
      float corr = __expf(m - m_new);
      float w = __expf(score - m_new);
      l = l * corr + w;
      const T* vj = vbase + (int64_t)j * VD;
      #pragma unroll
      for (int dl = 0; dl < ACCN; ++dl) {
        int d = lane + dl * 32;
        if (d < VD) acc[dl] = acc[dl] * corr + w * ld<T>(vj, d);
      }
      m = m_new;
    }
  } else {
    for (int j = j0 + tid; j < j1; j += nt) {
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
      float m_new = fmaxf(m, score);
      float corr = __expf(m - m_new);
      float w = __expf(score - m_new);
      l = l * corr + w;
      const T* vj = vbase + (int64_t)j * VD;
      for (int d = 0; d < VD; ++d) acc[d] = acc[d] * corr + w * ld<T>(vj, d);
      m = m_new;
    }
  }

  // Merge this block's threads/warps into one (m, l, acc) partial for the
  // partition — same reduction tree as apa_selective_kernel's pass 2 merge,
  // just scoped to this partition's keys instead of the whole row. Under
  // WCOOP, l is a SUM duplicated across a warp's lanes, so only lane 0
  // contributes (else 32x overcount); m's max is duplicate-safe either way.
  __shared__ float red[256];
  red[tid] = m; __syncthreads();
  for (int off = nt / 2; off > 0; off >>= 1) { if (tid < off) red[tid] = fmaxf(red[tid], red[tid+off]); __syncthreads(); }
  float pmax = red[0]; __syncthreads();

  float rescale = __expf(m - pmax);
  red[tid] = (WCOOP && lane != 0) ? 0.f : (l * rescale); __syncthreads();
  for (int off = nt / 2; off > 0; off >>= 1) { if (tid < off) red[tid] += red[tid+off]; __syncthreads(); }
  float pl = red[0]; __syncthreads();

  __shared__ float wpart[4][DMAX];
  if constexpr (WCOOP) {
    #pragma unroll
    for (int dl = 0; dl < ACCN; ++dl) {
      int d = lane + dl * 32;
      if (d < VD) wpart[warp][d] = acc[dl] * rescale;
    }
  } else {
    for (int d = 0; d < VD; ++d) {
      float val = acc[d] * rescale;
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(FULL, val, off);
      if (lane == 0) wpart[warp][d] = val;
    }
  }
  __syncthreads();
  for (int d = tid; d < VD; d += nt) {
    float s = 0.f;
    for (int w = 0; w < nwarp; ++w) s += wpart[w][d];
    pacc[d] = s;
  }
  if (tid == 0) { part_m[part_row] = pmax; part_l[part_row] = pl; }
}

// apa_selective_merge_kernel: one block per row, reduces num_parts partials
// with the standard online-softmax merge (rescale by exp(m_i - m_max), sum).
// Optional sink fold-in happens HERE, exactly once, after the partitions are
// combined — per the addendum's constraint that in split-K the sink logit
// must fold exactly once at the merge stage (mirrors apa_selective_sink_
// kernel's tid==0 sink fold, which happens once per row there too).
template <typename T, int VDMAX>
__global__ void apa_selective_merge_kernel(
    const float* part_m, const float* part_l, const float* part_acc,
    const T* sinks, T* out, int H, int VD, int num_parts, int use_sink) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int nt = blockDim.x;
  int h = row % H;   // row is flat (b,h,i); h = (row / L) % H in general, but
                      // since L==1 on the split-K decode path (dispatch-gated
                      // below), row % H == h exactly (b*H + h, i always 0).

  const float* pm = part_m + (int64_t)row * num_parts;
  const float* pl = part_l + (int64_t)row * num_parts;
  const float* pacc = part_acc + (int64_t)row * num_parts * VD;

  __shared__ float red[256];
  float m = -1e30f;
  for (int p = tid; p < num_parts; p += nt) m = fmaxf(m, pm[p]);
  red[tid] = m; __syncthreads();
  for (int off = nt / 2; off > 0; off >>= 1) { if (tid < off) red[tid] = fmaxf(red[tid], red[tid+off]); __syncthreads(); }
  float gmax = red[0]; __syncthreads();

  float lsum = 0.f;
  for (int p = tid; p < num_parts; p += nt) lsum += pl[p] * __expf(pm[p] - gmax);
  red[tid] = lsum; __syncthreads();
  for (int off = nt / 2; off > 0; off >>= 1) { if (tid < off) red[tid] += red[tid+off]; __syncthreads(); }
  float l = red[0]; __syncthreads();

  __shared__ float osh[VDMAX];
  for (int d = tid; d < VD; d += nt) {
    float s = 0.f;
    for (int p = 0; p < num_parts; ++p) s += pacc[(int64_t)p * VD + d] * __expf(pm[p] - gmax);
    osh[d] = s;
  }
  __syncthreads();

  if (use_sink) {
    // Exactly the fused apa_selective_sink_kernel's fold: sink logit joins
    // (m, l) via the same online-softmax update, contributing zero value
    // mass. Done once here (post-merge), not per-partition, so a sink never
    // gets counted num_parts times.
    if (tid == 0) {
      float sink = ld<T>(sinks, h);
      float m_new = fmaxf(gmax, sink);
      float corr = __expf(gmax - m_new);
      l = l * corr + __expf(sink - m_new);
      red[0] = corr;   // stash the rescale for the value-accumulator below
      red[1] = l;
      red[2] = m_new;
    }
    __syncthreads();
    float corr = red[0];
    l = red[1];
    for (int d = tid; d < VD; d += nt) osh[d] *= corr;
    __syncthreads();
  }

  float inv = l > 0.f ? 1.f / l : 0.f;
  T* orow = out + (int64_t)row * VD;
  for (int d = tid; d < VD; d += nt) st<T>(orow, d, osh[d] * inv);
}

// Dispatch heuristic (addendum #5): split-K only helps when the fused
// kernel's one-block-per-row grid underfills the SM count AND there is
// enough key-range work per row to make three launches + a cross-block
// reduction worth it. Conditions measured directly, not guessed:
//   - L == 1: decode-only, by construction (addendum #5: "prefill and
//     training paths untouched"). This also keeps the merge kernel's sink
//     fold-in simple — it recovers the query head from `row % H`, which is
//     only valid when L==1 (row == b*H + h exactly, no i term to fold in);
//     generalizing to L>1 would need row's i/h decomposition threaded
//     through the workspace layout too, which prefill does not need.
//   - rows < 2*TC_APA_SM_COUNT: the underfill regime the ncu receipt found
//     (16 blocks on 56 SMs, "0.0 full waves" — even 2x undersubscribed is
//     comfortably inside the reported problem). At L=1, rows==B*H; real B*H
//     (16..1024+) rarely exceeds 112 in single-request decode, so this is
//     effectively redundant with L==1 but kept explicit for batched decode.
//   - S >= TC_APA_SPLITK_PART_KEYS * 2: below this, a single partition would
//     already cover most/all of s_max, so splitting into >=2 real partitions
//     buys nothing over the fused kernel's own strided loop (128 threads
//     already stride the same keys inside one block) while paying 2 extra
//     kernel launches and a workspace round-trip. Only bother once there are
//     at least 2 full partitions of headroom.
// Prefill (L>1, rows large) and training fwd/bwd (separate kernels
// entirely, untouched) fall outside this heuristic by construction.
inline bool apa_selective_use_splitk(int rows, int S, int L) {
  return L == 1 && rows < 2 * TC_APA_SM_COUNT && S >= 2 * TC_APA_SPLITK_PART_KEYS;
}

// ============================================================================
// A5 WCOOP DISPATCH RULE (lead iteration on Addendum 2, 2026-07-07).
//
// The warp-cooperative key-load path (WCOOP=true) wins wherever the per-key
// D-loop is long or the grid is full — all prefill shapes and all D>=128
// decode (indicative: +58%..+478%) — but LOSES at D=64 decode (GPT-OSS
// geometry, indicative −14%..−29%): at 2 elements/lane the shuffle overhead
// is unamortized and warp-granularity key streaming surrenders the ILP of
// per-thread streaming, on a shape that was launch-shape-bound, never
// memory-pattern-bound (Phase 0.2 ncu). Both paths stay compiled; launchers
// select per launch:
//   - DMAX == 64 && decode-shaped        -> WCOOP=false (pre-A5 per-thread)
//   - DMAX == 64 && not decode-shaped    -> WCOOP=true  (prefill: coalescing wins)
//   - DMAX >= 128 (any shape)            -> WCOOP=true
//   - split-K stats/split at DMAX == 64  -> WCOOP=false ALWAYS (that path is
//     decode-only by construction, dispatch-gated on L==1)
// "Decode-shaped" mirrors the split-K heuristic's notion of decode: a single
// query row per (b,h) with a grid too small to fill the SMs.
// ============================================================================
inline bool apa_selective_decode_shaped(int rows, int L) {
  return L == 1 && rows < 2 * TC_APA_SM_COUNT;
}

// Env override for testing both paths regardless of shape (validation only —
// production dispatch uses apa_selective_use_splitk above). 0=auto (default),
// 1=force fused, 2=force split-K.
inline int apa_selective_path_override() {
  const char* v = std::getenv("TC_APA_SELECTIVE_PATH");
  if (!v) return 0;
  if (v[0] == '1') return 1;
  if (v[0] == '2') return 2;
  return 0;
}

// Shared split-K launcher used by both the plain and sink entry points.
// use_sink selects whether the merge kernel folds a learned sink logit.
template <typename T>
static NDArray apa_selective_splitk_dispatch(
    const NDArray& q, const NDArray& k, const NDArray& kq, const NDArray& v,
    const NDArray* sinks, float scale, float zthr, bool is_causal,
    int B, int H, int L, int S, int D, int VD, int KVH, int group, int cap) {
  int rows = B * H * L;
  NDArray out({(int64_t)B, (int64_t)H, (int64_t)L, (int64_t)VD}, q.dtype, q.device);
  if (rows == 0) return out;

  int part_keys = TC_APA_SPLITK_PART_KEYS;
  int num_parts = (S + part_keys - 1) / part_keys;
  if (num_parts < 1) num_parts = 1;

  // Workspace: per-(row,part) online-softmax partials. m/l are one float
  // each; acc is VD floats. Always fp32 regardless of q/k/v dtype — the
  // fused kernel's own accumulators are fp32 too (ld<T> upconverts on read).
  NDArray thr_ws({(int64_t)rows}, DType::Float32, q.device);
  NDArray part_m({(int64_t)rows * num_parts}, DType::Float32, q.device);
  NDArray part_l({(int64_t)rows * num_parts}, DType::Float32, q.device);
  NDArray part_acc({(int64_t)rows * num_parts * VD}, DType::Float32, q.device);

  int threads = 128;
  auto launch = [&](auto dmax_tag, auto wcoop_tag) {
    constexpr int DMAX = decltype(dmax_tag)::value;
    constexpr bool WCOOP = decltype(wcoop_tag)::value;
    apa_selective_stats_kernel<T, DMAX, WCOOP><<<rows, threads>>>(
        static_cast<T*>(q.data_ptr()), static_cast<T*>(kq.data_ptr()),
        static_cast<float*>(thr_ws.data_ptr()), B, H, L, S, D, scale, zthr,
        is_causal ? 1 : 0, KVH, group);
    cuda_check_last("apa_selective_splitk_stats");

    dim3 grid2((unsigned)rows, (unsigned)num_parts);
    apa_selective_split_kernel<T, DMAX, WCOOP><<<grid2, threads>>>(
        static_cast<T*>(q.data_ptr()), static_cast<T*>(k.data_ptr()),
        static_cast<T*>(kq.data_ptr()), static_cast<T*>(v.data_ptr()),
        static_cast<float*>(thr_ws.data_ptr()),
        static_cast<float*>(part_m.data_ptr()), static_cast<float*>(part_l.data_ptr()),
        static_cast<float*>(part_acc.data_ptr()), B, H, L, S, D, VD, scale,
        is_causal ? 1 : 0, KVH, group, num_parts, part_keys);
    cuda_check_last("apa_selective_splitk_split");

    apa_selective_merge_kernel<T, DMAX><<<rows, threads>>>(
        static_cast<float*>(part_m.data_ptr()), static_cast<float*>(part_l.data_ptr()),
        static_cast<float*>(part_acc.data_ptr()),
        sinks ? static_cast<T*>(sinks->data_ptr()) : nullptr,
        static_cast<T*>(out.data_ptr()), H, VD, num_parts, sinks ? 1 : 0);
    cuda_check_last("apa_selective_splitk_merge");
  };
  // A5 WCOOP dispatch (see apa_selective_decode_shaped's comment block):
  // split-K is decode-only by construction (dispatch-gated on L==1), so at
  // DMAX==64 the per-thread path applies unconditionally here.
  if (cap <= 64)        launch(std::integral_constant<int, 64>{},  std::false_type{});
  else if (cap <= 128)  launch(std::integral_constant<int, 128>{}, std::true_type{});
  else if (cap <= 256)  launch(std::integral_constant<int, 256>{}, std::true_type{});
  else                  launch(std::integral_constant<int, 512>{}, std::true_type{});
  return out;
}

NDArray apa_selective_attention(const NDArray& q, const NDArray& k,
                                const NDArray& kq, const NDArray& v,
                                float scale, float zthr, bool is_causal) {
  int B = q.shape[0], H = q.shape[1], L = q.shape[2], D = q.shape[3];
  int S = k.shape[2];
  int VD = v.shape[3];
  // GQA: k/kq/v carry KVH heads (<= H). group = H/KVH query heads per KV head.
  int KVH = (int)k.shape[1];
  int group = (KVH > 0) ? (H / KVH) : 1;
  if (k.shape[3] != D || kq.shape[3] != D) throw std::runtime_error("apa_selective: q/k/kq dim mismatch");
  if (v.shape[0] != B || v.shape[1] != KVH || v.shape[2] != S) throw std::runtime_error("apa_selective: v shape mismatch");
  int rows = B * H * L;
  int threads = 128;
  int cap = D > VD ? D : VD;
  if (rows > 0 && cap > TC_APA_MAXD) {
    throw std::runtime_error("apa_selective: head_dim exceeds TC_APA_MAXD "
                             "(512); raise the cap + add a dispatch arm");
  }
  // Dispatch: the split-K path (A1, addendum 1) only helps the decode grid-
  // underfill case (small rows, long S) that the fused one-block-per-row
  // kernel leaves idle on most SMs; the fused kernel stays default for
  // well-filled grids (prefill, training — unaffected) per addendum #5.
  // TC_APA_SELECTIVE_PATH env override (0=auto,1=fused,2=split-K) exists so
  // tests can force either path independent of shape.
  int override_path = apa_selective_path_override();
  if (override_path == 2 && L != 1) {
    throw std::runtime_error("apa_selective: TC_APA_SELECTIVE_PATH=2 (force "
                             "split-K) requires L==1 (decode-shape only)");
  }
  bool use_splitk = (override_path == 2 && L == 1) ||
      (override_path == 0 && rows > 0 && apa_selective_use_splitk(rows, S, L));
  if (use_splitk && rows > 0) {
    NDArray out;
    DISPATCH_FLOAT(q.dtype, T, {
      out = apa_selective_splitk_dispatch<T>(
          q, k, kq, v, nullptr, scale, zthr, is_causal,
          B, H, L, S, D, VD, KVH, group, cap);
    });
    cuda_check_last("apa_selective_splitk");
    return out;
  }
  NDArray out({(int64_t)B, (int64_t)H, (int64_t)L, (int64_t)VD}, q.dtype, q.device);
  if (rows > 0) {
    // Dispatch the smallest compile-time DMAX >= D so acc[] stays register/L1
    // resident. Real head_dims are 64 (TinyLlama) and 128 (Mistral/Qwen/OLMoE);
    // 256 is the safety fallback (== the old behaviour) for anything larger.
    // WCOOP per the A5 dispatch rule (apa_selective_decode_shaped comment
    // block): per-thread at DMAX==64 decode shapes, warp-cooperative else.
    DISPATCH_FLOAT(q.dtype, T, {
      auto launch = [&](auto dmax_tag, auto wcoop_tag) {
        constexpr int DMAX = decltype(dmax_tag)::value;
        constexpr bool WCOOP = decltype(wcoop_tag)::value;
        apa_selective_kernel<T, DMAX, WCOOP><<<rows, threads>>>(
            static_cast<T*>(q.data_ptr()), static_cast<T*>(k.data_ptr()),
            static_cast<T*>(kq.data_ptr()), static_cast<T*>(v.data_ptr()),
            static_cast<T*>(out.data_ptr()), B, H, L, S, D, VD, scale, zthr,
            is_causal ? 1 : 0, KVH, group);
      };
      if (cap <= 64) {
        if (apa_selective_decode_shaped(rows, L))
          launch(std::integral_constant<int, 64>{}, std::false_type{});
        else
          launch(std::integral_constant<int, 64>{}, std::true_type{});
      }
      else if (cap <= 128)  launch(std::integral_constant<int, 128>{}, std::true_type{});
      else if (cap <= 256)  launch(std::integral_constant<int, 256>{}, std::true_type{});
      else                  launch(std::integral_constant<int, 512>{}, std::true_type{});
    });
    cuda_check_last("apa_selective");
  }
  return out;
}

template <typename T, int DMAX, bool WCOOP>
__global__ void apa_selective_sink_kernel(
    const T* q, const T* k, const T* kq, const T* v, const T* sinks, T* out,
    int B, int H, int L, int S, int D, int VD, float scale, float zthr,
    int is_causal, int KVH, int group) {
  int row = blockIdx.x;
  int i = row % L;
  int bh = row / L;
  int b = bh / H;
  int h = bh % H;
  int kv_h = h / group;
  int tid = threadIdx.x;
  int nt = blockDim.x;
  const unsigned FULL = 0xffffffffu;
  int lane = tid & 31;
  int warp = tid >> 5;
  int nwarp = nt >> 5;

  const T* qrow = q + (int64_t)row * D;
  int64_t kvbh = (int64_t)b * KVH + kv_h;
  const T* kbase = k + kvbh * S * D;
  const T* kqbase = kq + kvbh * S * D;
  const T* vbase = v + kvbh * S * VD;

  __shared__ float qsh[DMAX];
  for (int d = tid; d < D; d += nt) qsh[d] = ld<T>(qrow, d);
  __syncthreads();

  int s_max = is_causal ? ((S - L) + i + 1) : S;
  __shared__ float red[256];

  // A5 dual-path (see apa_selective_kernel's DISPATCH comment): WCOOP=true
  // is warp-cooperative key loads, WCOOP=false the pre-A5 per-thread loop;
  // identical here modulo the sink fold below.
  float sum = 0.f, sumsq = 0.f;
  if constexpr (WCOOP) {
    for (int j = warp; j < s_max; j += nwarp) {
      const T* kqj = kqbase + (int64_t)j * D;
      float dot = 0.f;
      for (int d = lane; d < D; d += 32) dot += qsh[d] * ld<T>(kqj, d);
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1) dot += __shfl_down_sync(FULL, dot, off);
      dot = __shfl_sync(FULL, dot, 0);
      float a = fabsf(dot * scale);
      sum += a;
      sumsq += a * a;
    }
  } else {
    for (int j = tid; j < s_max; j += nt) {
      const T* kqj = kqbase + (int64_t)j * D;
      float dot = 0.f;
      for (int d = 0; d < D; ++d) dot += qsh[d] * ld<T>(kqj, d);
      float a = fabsf(dot * scale);
      sum += a;
      sumsq += a * a;
    }
  }
  // WCOOP: per-warp totals duplicated across the warp's 32 lanes — only
  // lane 0 contributes (else 32x overcount). Per-thread: all contribute.
  red[tid] = (WCOOP && lane != 0) ? 0.f : sum; __syncthreads();
  for (int off = nt / 2; off > 0; off >>= 1) {
    if (tid < off) red[tid] += red[tid + off];
    __syncthreads();
  }
  float total = red[0]; __syncthreads();
  red[tid] = (WCOOP && lane != 0) ? 0.f : sumsq; __syncthreads();
  for (int off = nt / 2; off > 0; off >>= 1) {
    if (tid < off) red[tid] += red[tid + off];
    __syncthreads();
  }
  float total_sq = red[0]; __syncthreads();

  float cnt = (float)s_max;
  float mean = total / cnt;
  float var = total_sq / cnt - mean * mean;
  float thr = mean + zthr * sqrtf(fmaxf(var, 0.f));

  float m = -1e30f, l = 0.f;
  constexpr int ACCN = WCOOP ? ((DMAX + 31) / 32) : DMAX;
  float acc[ACCN];
  if constexpr (WCOOP) {
    #pragma unroll
    for (int dl = 0; dl < ACCN; ++dl) acc[dl] = 0.f;
  } else {
    for (int d = 0; d < VD; ++d) acc[d] = 0.f;
  }

  if constexpr (WCOOP) {
    for (int j = warp; j < s_max; j += nwarp) {
      const T* kqj = kqbase + (int64_t)j * D;
      float bulk = 0.f;
      for (int d = lane; d < D; d += 32) bulk += qsh[d] * ld<T>(kqj, d);
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1) bulk += __shfl_down_sync(FULL, bulk, off);
      bulk = __shfl_sync(FULL, bulk, 0);
      bulk *= scale;
      float score;
      if (fabsf(bulk) >= thr) {
        const T* kj = kbase + (int64_t)j * D;
        float ex = 0.f;
        for (int d = lane; d < D; d += 32) ex += qsh[d] * ld<T>(kj, d);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) ex += __shfl_down_sync(FULL, ex, off);
        ex = __shfl_sync(FULL, ex, 0);
        score = ex * scale;
      } else {
        score = bulk;
      }
      float m_new = fmaxf(m, score);
      float corr = __expf(m - m_new);
      float w = __expf(score - m_new);
      l = l * corr + w;
      const T* vj = vbase + (int64_t)j * VD;
      #pragma unroll
      for (int dl = 0; dl < ACCN; ++dl) {
        int d = lane + dl * 32;
        if (d < VD) acc[dl] = acc[dl] * corr + w * ld<T>(vj, d);
      }
      m = m_new;
    }
  } else {
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
      float m_new = fmaxf(m, score);
      float corr = __expf(m - m_new);
      float w = __expf(score - m_new);
      l = l * corr + w;
      const T* vj = vbase + (int64_t)j * VD;
      for (int d = 0; d < VD; ++d) acc[d] = acc[d] * corr + w * ld<T>(vj, d);
      m = m_new;
    }
  }

  // GPT-OSS learned sink: denominator only, zero value contribution. Folded
  // EXACTLY ONCE PER BLOCK (the sink logit must enter the row's softmax
  // total a single time).
  // WCOOP=true: folded into WARP 0's (m, l, acc) — warp 0's lane 0 computes
  // the fold (sink depends only on h, same for every lane); the corrected
  // (m, l, corr) are then broadcast to warp 0's other 31 lanes so the WHOLE
  // warp's acc slice (not just lane 0's own d's) gets rescaled consistently
  // — warp 0 collectively owns all VD dims across its lanes, same as tid==0
  // owning the full acc[VD] in the pre-A5 kernel. Warps 1..nwarp-1 are
  // untouched (their m/l/acc stay pre-fold).
  // WCOOP=false: the exact pre-A5 fold — thread 0 owns a full acc[VD]
  // partial, so it alone folds and rescales its own accumulator.
  if constexpr (WCOOP) {
    if (warp == 0) {
      float m_for_fold = m, l_for_fold = l, corr = 1.f;
      if (lane == 0) {
        float sink = ld<T>(sinks, h);
        float m_new = fmaxf(m, sink);
        corr = __expf(m - m_new);
        l_for_fold = l * corr + __expf(sink - m_new);
        m_for_fold = m_new;
      }
      m = __shfl_sync(FULL, m_for_fold, 0);
      l = __shfl_sync(FULL, l_for_fold, 0);
      corr = __shfl_sync(FULL, corr, 0);
      #pragma unroll
      for (int dl = 0; dl < ACCN; ++dl) acc[dl] *= corr;
    }
  } else {
    if (tid == 0) {
      float sink = ld<T>(sinks, h);
      float m_new = fmaxf(m, sink);
      float corr = __expf(m - m_new);
      l = l * corr + __expf(sink - m_new);
      for (int d = 0; d < VD; ++d) acc[d] *= corr;
      m = m_new;
    }
  }

  red[tid] = m; __syncthreads();
  for (int off = nt / 2; off > 0; off >>= 1) {
    if (tid < off) red[tid] = fmaxf(red[tid], red[tid + off]);
    __syncthreads();
  }
  float gmax = red[0]; __syncthreads();

  // WCOOP: l is a SUM, identical across a warp's 32 lanes (either pre-fold
  // warp total, or warp 0's post-fold total) — only lane 0 of each warp
  // contributes, else 32x overcount. Per-thread: all contribute.
  float rescale = __expf(m - gmax);
  red[tid] = (WCOOP && lane != 0) ? 0.f : (l * rescale); __syncthreads();
  for (int off = nt / 2; off > 0; off >>= 1) {
    if (tid < off) red[tid] += red[tid + off];
    __syncthreads();
  }
  float denom = red[0]; __syncthreads();
  float inv = denom > 0.f ? 1.f / denom : 0.f;

  __shared__ float osh[DMAX];
  __shared__ float wpart[4][DMAX];
  if constexpr (WCOOP) {
    #pragma unroll
    for (int dl = 0; dl < ACCN; ++dl) {
      int d = lane + dl * 32;
      if (d < VD) wpart[warp][d] = acc[dl] * rescale;
    }
  } else {
    for (int d = 0; d < VD; ++d) {
      float val = acc[d] * rescale;
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(FULL, val, off);
      if (lane == 0) wpart[warp][d] = val;
    }
  }
  __syncthreads();
  for (int d = tid; d < VD; d += nt) {
    float s = 0.f;
    for (int w = 0; w < nwarp; ++w) s += wpart[w][d];
    osh[d] = s;
  }
  __syncthreads();

  T* orow = out + (int64_t)row * VD;
  for (int d = tid; d < VD; d += nt) st<T>(orow, d, osh[d] * inv);
}

NDArray apa_selective_attention_sink(const NDArray& q, const NDArray& k,
                                     const NDArray& kq, const NDArray& v,
                                     const NDArray& sinks, float scale,
                                     float zthr, bool is_causal) {
  int B = q.shape[0], H = q.shape[1], L = q.shape[2], D = q.shape[3];
  int S = k.shape[2];
  int VD = v.shape[3];
  int KVH = (int)k.shape[1];
  int group = (KVH > 0) ? (H / KVH) : 1;
  if (k.shape[3] != D || kq.shape[3] != D)
    throw std::runtime_error("apa_selective_sink: q/k/kq dim mismatch");
  if (v.shape[0] != B || v.shape[1] != KVH || v.shape[2] != S)
    throw std::runtime_error("apa_selective_sink: v shape mismatch");
  if (sinks.ndim() != 1 || sinks.shape[0] != H)
    throw std::runtime_error("apa_selective_sink: sinks shape mismatch");
  if (sinks.dtype != q.dtype)
    throw std::runtime_error("apa_selective_sink: sinks dtype mismatch");
  int rows = B * H * L;
  int threads = 128;
  int cap = D > VD ? D : VD;
  if (rows > 0 && cap > TC_APA_MAXD) {
    throw std::runtime_error("apa_selective_sink: head_dim exceeds TC_APA_MAXD");
  }
  // Dispatch: same split-K heuristic as apa_selective_attention (see its
  // comment) — decode-only, small rows, long S. The merge kernel folds the
  // sink logit exactly once, post-merge (addendum #4), via use_sink=1.
  int override_path = apa_selective_path_override();
  if (override_path == 2 && L != 1) {
    throw std::runtime_error("apa_selective_sink: TC_APA_SELECTIVE_PATH=2 "
                             "(force split-K) requires L==1 (decode-shape only)");
  }
  bool use_splitk = (override_path == 2 && L == 1) ||
      (override_path == 0 && rows > 0 && apa_selective_use_splitk(rows, S, L));
  if (use_splitk && rows > 0) {
    NDArray out;
    DISPATCH_FLOAT(q.dtype, T, {
      out = apa_selective_splitk_dispatch<T>(
          q, k, kq, v, &sinks, scale, zthr, is_causal,
          B, H, L, S, D, VD, KVH, group, cap);
    });
    cuda_check_last("apa_selective_sink_splitk");
    return out;
  }
  NDArray out({(int64_t)B, (int64_t)H, (int64_t)L, (int64_t)VD},
              q.dtype, q.device);
  if (rows > 0) {
    // WCOOP per the A5 dispatch rule (apa_selective_decode_shaped comment
    // block): per-thread at DMAX==64 decode shapes, warp-cooperative else.
    DISPATCH_FLOAT(q.dtype, T, {
      auto launch = [&](auto dmax_tag, auto wcoop_tag) {
        constexpr int DMAX = decltype(dmax_tag)::value;
        constexpr bool WCOOP = decltype(wcoop_tag)::value;
        apa_selective_sink_kernel<T, DMAX, WCOOP><<<rows, threads>>>(
            static_cast<T*>(q.data_ptr()), static_cast<T*>(k.data_ptr()),
            static_cast<T*>(kq.data_ptr()), static_cast<T*>(v.data_ptr()),
            static_cast<T*>(sinks.data_ptr()), static_cast<T*>(out.data_ptr()),
            B, H, L, S, D, VD, scale, zthr, is_causal ? 1 : 0, KVH, group);
      };
      if (cap <= 64) {
        if (apa_selective_decode_shaped(rows, L))
          launch(std::integral_constant<int, 64>{}, std::false_type{});
        else
          launch(std::integral_constant<int, 64>{}, std::true_type{});
      }
      else if (cap <= 128)  launch(std::integral_constant<int, 128>{}, std::true_type{});
      else if (cap <= 256)  launch(std::integral_constant<int, 256>{}, std::true_type{});
      else                  launch(std::integral_constant<int, 512>{}, std::true_type{});
    });
    cuda_check_last("apa_selective_sink");
  }
  return out;
}


// ============================================================================
// APA SELECTIVE — TRAINING forward + backward (O(L) memory, the graft-native
// training path). The inference kernel above never materializes the L x L
// score matrix; these mirror it so training does not either. The selection is
// a STOP-GRADIENT (POC 2.1): the threshold and the bulk/refined CHOICE are
// constants; gradients flow only through the score dots and the softmax. That
// makes this backward far simpler than a general FlashAttention backward — no
// gradient through the threshold, and unselected keys feed gradient through q
// only (kq is detached).
//
// Forward (training) additionally saves, per query row, the logsumexp
// L_se = gmax + log(denom) and the threshold thr, so the backward recomputes
// every key's softmax weight exactly without storing the L x L matrix.
// ============================================================================

template <typename T, int DMAX, bool WCOOP>
__global__ void apa_selective_fwd_train_kernel(
    const T* q, const T* k, const T* kq, const T* v, T* out,
    float* lse_out, float* thr_out,
    int B, int H, int L, int S, int D, int VD, float scale, float zthr,
    int is_causal, int KVH, int group) {
  int row = blockIdx.x;
  int i = row % L;
  int bh = row / L;
  int b = bh / H;
  int h = bh % H;
  int kv_h = h / group;
  int tid = threadIdx.x;
  int nt = blockDim.x;
  const unsigned FULL = 0xffffffffu;
  int lane = tid & 31, warp = tid >> 5, nwarp = nt >> 5;

  const T* qrow = q + (int64_t)row * D;
  int64_t kvbh = (int64_t)b * KVH + kv_h;
  const T* kbase = k + kvbh * S * D;
  const T* kqbase = kq + kvbh * S * D;
  const T* vbase = v + kvbh * S * VD;

  __shared__ float qsh[DMAX];
  for (int d = tid; d < D; d += nt) qsh[d] = ld<T>(qrow, d);
  __syncthreads();

  int s_max = is_causal ? ((S - L) + i + 1) : S;
  __shared__ float red[256];

  // Pass 1: |bulk| stats -> z-score threshold (identical to inference).
  // A5 dual-path (surgical port from apa_selective_kernel — same design,
  // see that kernel's DISPATCH comment; backward is untouched, out of scope
  // per the addendum).
  float sum = 0.f, sumsq = 0.f;
  if constexpr (WCOOP) {
    for (int j = warp; j < s_max; j += nwarp) {
      const T* kqj = kqbase + (int64_t)j * D;
      float dot = 0.f;
      for (int d = lane; d < D; d += 32) dot += qsh[d] * ld<T>(kqj, d);
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1) dot += __shfl_down_sync(FULL, dot, off);
      dot = __shfl_sync(FULL, dot, 0);
      float a = fabsf(dot * scale);
      sum += a; sumsq += a * a;
    }
  } else {
    for (int j = tid; j < s_max; j += nt) {
      const T* kqj = kqbase + (int64_t)j * D;
      float dot = 0.f;
      for (int d = 0; d < D; ++d) dot += qsh[d] * ld<T>(kqj, d);
      float a = fabsf(dot * scale);
      sum += a; sumsq += a * a;
    }
  }
  // WCOOP: per-warp totals duplicated across the warp's 32 lanes — only
  // lane 0 contributes (else 32x overcount). Per-thread: all contribute.
  red[tid] = (WCOOP && lane != 0) ? 0.f : sum; __syncthreads();
  for (int off = nt/2; off>0; off>>=1){ if(tid<off) red[tid]+=red[tid+off]; __syncthreads(); }
  float total = red[0]; __syncthreads();
  red[tid] = (WCOOP && lane != 0) ? 0.f : sumsq; __syncthreads();
  for (int off = nt/2; off>0; off>>=1){ if(tid<off) red[tid]+=red[tid+off]; __syncthreads(); }
  float total_sq = red[0]; __syncthreads();
  float cnt = (float)s_max;
  float mean = total / cnt;
  float var = total_sq / cnt - mean * mean;
  float thr = mean + zthr * sqrtf(fmaxf(var, 0.f));

  // Pass 2: online softmax -> output, accumulating m and l (same as inference).
  float m = -1e30f, l = 0.f;
  constexpr int ACCN = WCOOP ? ((DMAX + 31) / 32) : DMAX;
  float acc[ACCN];
  if constexpr (WCOOP) {
    #pragma unroll
    for (int dl = 0; dl < ACCN; ++dl) acc[dl] = 0.f;
  } else {
    for (int d = 0; d < VD; ++d) acc[d] = 0.f;
  }
  if constexpr (WCOOP) {
    for (int j = warp; j < s_max; j += nwarp) {
      const T* kqj = kqbase + (int64_t)j * D;
      float bulk = 0.f;
      for (int d = lane; d < D; d += 32) bulk += qsh[d] * ld<T>(kqj, d);
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1) bulk += __shfl_down_sync(FULL, bulk, off);
      bulk = __shfl_sync(FULL, bulk, 0);
      bulk *= scale;
      float score;
      if (fabsf(bulk) >= thr) {
        const T* kj = kbase + (int64_t)j * D;
        float ex = 0.f;
        for (int d = lane; d < D; d += 32) ex += qsh[d] * ld<T>(kj, d);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) ex += __shfl_down_sync(FULL, ex, off);
        ex = __shfl_sync(FULL, ex, 0);
        score = ex * scale;
      } else score = bulk;
      float m_new = fmaxf(m, score);
      float corr = __expf(m - m_new);
      float w = __expf(score - m_new);
      l = l * corr + w;
      const T* vj = vbase + (int64_t)j * VD;
      #pragma unroll
      for (int dl = 0; dl < ACCN; ++dl) {
        int d = lane + dl * 32;
        if (d < VD) acc[dl] = acc[dl] * corr + w * ld<T>(vj, d);
      }
      m = m_new;
    }
  } else {
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
      } else score = bulk;
      float m_new = fmaxf(m, score);
      float corr = __expf(m - m_new);
      float w = __expf(score - m_new);
      l = l * corr + w;
      const T* vj = vbase + (int64_t)j * VD;
      for (int d = 0; d < VD; ++d) acc[d] = acc[d] * corr + w * ld<T>(vj, d);
      m = m_new;
    }
  }
  // merge (m,l,acc); under WCOOP l is a SUM duplicated identically across a
  // warp's 32 lanes, so only lane 0 contributes (else 32x overcount) — m's
  // max is duplicate-safe either way.
  red[tid] = m; __syncthreads();
  for (int off = nt/2; off>0; off>>=1){ if(tid<off) red[tid]=fmaxf(red[tid],red[tid+off]); __syncthreads(); }
  float gmax = red[0]; __syncthreads();
  float rescale = __expf(m - gmax);
  red[tid] = (WCOOP && lane != 0) ? 0.f : (l * rescale); __syncthreads();
  for (int off = nt/2; off>0; off>>=1){ if(tid<off) red[tid]+=red[tid+off]; __syncthreads(); }
  float denom = red[0]; __syncthreads();
  float inv = denom > 0.f ? 1.f / denom : 0.f;

  __shared__ float osh[DMAX];
  __shared__ float wpart[4][DMAX];
  if constexpr (WCOOP) {
    #pragma unroll
    for (int dl = 0; dl < ACCN; ++dl) {
      int d = lane + dl * 32;
      if (d < VD) wpart[warp][d] = acc[dl] * rescale;
    }
  } else {
    for (int d = 0; d < VD; ++d) {
      float val = acc[d] * rescale;
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(FULL, val, off);
      if (lane == 0) wpart[warp][d] = val;
    }
  }
  __syncthreads();
  for (int d = tid; d < VD; d += nt) {
    float s = 0.f; for (int w = 0; w < nwarp; ++w) s += wpart[w][d];
    osh[d] = s;
  }
  __syncthreads();
  T* orow = out + (int64_t)row * VD;
  for (int d = tid; d < VD; d += nt) st<T>(orow, d, osh[d] * inv);
  // SAVE backward state: per-row logsumexp and threshold.
  if (tid == 0) {
    lse_out[row] = gmax + logf(fmaxf(denom, 1e-30f));
    thr_out[row] = thr;
  }
}

// Backward. One block per query row. Recomputes each key's score and softmax
// weight from saved (lse, thr), then accumulates dV, dK (selected only), and a
// thread-local dQ that is block-reduced and written once. dK/dV use atomicAdd
// (multiple query rows touch the same KV row). Two passes over keys: pass A
// computes the softmax-jacobian row term rowdot = sum_j p_j (dO . v_j); pass B
// uses it to form dScore_j and scatters the gradients.
template <typename T, int DMAX>
__global__ void apa_selective_bwd_kernel(
    const T* q, const T* k, const T* kq, const T* v, const T* dO,
    const float* lse, const float* thr_in,
    T* dq, T* dk, T* dv,
    int B, int H, int L, int S, int D, int VD, float scale,
    int is_causal, int KVH, int group) {
  int row = blockIdx.x;
  int i = row % L;
  int bh = row / L;
  int b = bh / H;
  int h = bh % H;
  int kv_h = h / group;
  int tid = threadIdx.x, nt = blockDim.x;

  const T* qrow = q + (int64_t)row * D;
  const T* dOrow = dO + (int64_t)row * VD;
  int64_t kvbh = (int64_t)b * KVH + kv_h;
  const T* kbase = k + kvbh * S * D;
  const T* kqbase = kq + kvbh * S * D;
  const T* vbase = v + kvbh * S * VD;
  T* dkbase = dk + kvbh * S * D;
  T* dvbase = dv + kvbh * S * VD;

  __shared__ float qsh[DMAX];
  __shared__ float dOsh[DMAX];
  for (int d = tid; d < D; d += nt) qsh[d] = ld<T>(qrow, d);
  for (int d = tid; d < VD; d += nt) dOsh[d] = ld<T>(dOrow, d);
  __syncthreads();

  int s_max = is_causal ? ((S - L) + i + 1) : S;
  float L_se = lse[row];
  float thr = thr_in[row];
  __shared__ float red[256];

  // Pass A: rowdot = sum_j p_j * (dO . v_j)
  float partial = 0.f;
  for (int j = tid; j < s_max; j += nt) {
    const T* kqj = kqbase + (int64_t)j * D;
    float bulk = 0.f;
    for (int d = 0; d < D; ++d) bulk += qsh[d] * ld<T>(kqj, d);
    bulk *= scale;
    float score;
    if (fabsf(bulk) >= thr) {
      const T* kj = kbase + (int64_t)j * D;
      float ex = 0.f; for (int d = 0; d < D; ++d) ex += qsh[d] * ld<T>(kj, d);
      score = ex * scale;
    } else score = bulk;
    float p = __expf(score - L_se);
    const T* vj = vbase + (int64_t)j * VD;
    float dov = 0.f; for (int d = 0; d < VD; ++d) dov += dOsh[d] * ld<T>(vj, d);
    partial += p * dov;
  }
  red[tid] = partial; __syncthreads();
  for (int off = nt/2; off>0; off>>=1){ if(tid<off) red[tid]+=red[tid+off]; __syncthreads(); }
  float rowdot = red[0]; __syncthreads();

  // Pass B: per-key gradients. dScore_j = p_j * (dO.v_j - rowdot).
  // dV_j += p_j * dO ; dQ += dScore_j*scale * (k_j or kq_j) ; dK_j += dScore_j*scale*q (selected).
  float dq_acc[DMAX];
  for (int d = 0; d < D; ++d) dq_acc[d] = 0.f;
  for (int j = tid; j < s_max; j += nt) {
    const T* kqj = kqbase + (int64_t)j * D;
    float bulk = 0.f;
    for (int d = 0; d < D; ++d) bulk += qsh[d] * ld<T>(kqj, d);
    bulk *= scale;
    bool sel = fabsf(bulk) >= thr;
    float score;
    const T* kj = kbase + (int64_t)j * D;
    if (sel) { float ex=0.f; for(int d=0;d<D;++d) ex+=qsh[d]*ld<T>(kj,d); score=ex*scale; }
    else score = bulk;
    float p = __expf(score - L_se);
    const T* vj = vbase + (int64_t)j * VD;
    float dov = 0.f; for (int d = 0; d < VD; ++d) dov += dOsh[d] * ld<T>(vj, d);
    float dscore = p * (dov - rowdot);
    T* dvj = dvbase + (int64_t)j * VD;
    T* dkj = dkbase + (int64_t)j * D;
    for (int d = 0; d < VD; ++d) {
      // dV_j += p * dO
      atomicAdd(&dvj[d], (T)(p * dOsh[d]));
    }
    for (int d = 0; d < D; ++d) {
      // dQ accumulates dscore*scale * key (selected: k_j; else: kq_j detached -> still feeds q)
      float kv = sel ? ld<T>(kj, d) : ld<T>(kqj, d);
      dq_acc[d] += dscore * scale * kv;
      // dK only for selected keys (kq is detached, so unselected dk = 0)
      if (sel) atomicAdd(&dkj[d], (T)(dscore * scale * qsh[d]));
    }
  }
  // reduce dq_acc across threads -> write dq[row]
  const unsigned FULL = 0xffffffffu;
  int lane = tid & 31, warp = tid >> 5, nwarp = (nt + 31) >> 5;
  __shared__ float wpart[4][DMAX];
  for (int d = 0; d < D; ++d) {
    float val = dq_acc[d];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(FULL, val, off);
    if (lane == 0) wpart[warp][d] = val;
  }
  __syncthreads();
  T* dqrow = dq + (int64_t)row * D;
  for (int d = tid; d < D; d += nt) {
    float s = 0.f; for (int w = 0; w < nwarp; ++w) s += wpart[w][d];
    st<T>(dqrow, d, s);
  }
}

// Training forward: returns out and fills the saved (lse, thr) NDArrays.
std::tuple<NDArray, NDArray, NDArray> apa_selective_fwd_train(
    const NDArray& q, const NDArray& k, const NDArray& kq, const NDArray& v,
    float scale, float zthr, bool is_causal) {
  int B=q.shape[0], H=q.shape[1], L=q.shape[2], D=q.shape[3];
  int S=k.shape[2];
  int VD=v.shape[3];
  int KVH=(int)k.shape[1]; int group=(KVH>0)?(H/KVH):1;
  if (k.shape[3] != D || kq.shape[3] != D) throw std::runtime_error("apa_selective_train: q/k/kq dim mismatch");
  if (v.shape[0] != B || v.shape[1] != KVH || v.shape[2] != S) throw std::runtime_error("apa_selective_train: v shape mismatch");
  NDArray out({(int64_t)B,(int64_t)H,(int64_t)L,(int64_t)VD}, q.dtype, q.device);
  NDArray lse({(int64_t)B,(int64_t)H,(int64_t)L}, DType::Float32, q.device);
  NDArray thr({(int64_t)B,(int64_t)H,(int64_t)L}, DType::Float32, q.device);
  int rows=B*H*L, threads=128;
  if (rows>0) {
    int cap = D > VD ? D : VD;
    if (cap>TC_APA_MAXD) throw std::runtime_error("apa_selective_train: head_dim exceeds cap");
    // WCOOP per the A5 dispatch rule (apa_selective_decode_shaped comment
    // block). Training is virtually always L>1 (whole sequences), so this
    // resolves to warp-cooperative in practice; the decode-shaped guard is
    // kept for rule consistency with the inference launchers.
    DISPATCH_FLOAT(q.dtype, T, {
      auto launch=[&](auto tag, auto wtag){ constexpr int DMAX=decltype(tag)::value;
        constexpr bool WCOOP=decltype(wtag)::value;
        apa_selective_fwd_train_kernel<T,DMAX,WCOOP><<<rows,threads>>>(
          static_cast<T*>(q.data_ptr()),static_cast<T*>(k.data_ptr()),
          static_cast<T*>(kq.data_ptr()),static_cast<T*>(v.data_ptr()),
          static_cast<T*>(out.data_ptr()),
          static_cast<float*>(lse.data_ptr()),static_cast<float*>(thr.data_ptr()),
          B,H,L,S,D,VD,scale,zthr,is_causal?1:0,KVH,group); };
      if (cap<=64) {
        if (apa_selective_decode_shaped(rows, L))
          launch(std::integral_constant<int,64>{}, std::false_type{});
        else
          launch(std::integral_constant<int,64>{}, std::true_type{});
      }
      else if (cap<=128) launch(std::integral_constant<int,128>{}, std::true_type{});
      else if (cap<=256) launch(std::integral_constant<int,256>{}, std::true_type{});
      else launch(std::integral_constant<int,512>{}, std::true_type{});
    });
    cuda_check_last("apa_selective_fwd_train");
  }
  return {out, lse, thr};
}

// Training backward: returns dq, dk, dv.
std::tuple<NDArray, NDArray, NDArray> apa_selective_bwd(
    const NDArray& q, const NDArray& k, const NDArray& kq, const NDArray& v,
    const NDArray& dO, const NDArray& lse, const NDArray& thr,
    float scale, bool is_causal) {
  int B=q.shape[0], H=q.shape[1], L=q.shape[2], D=q.shape[3];
  int S=k.shape[2];
  int VD=v.shape[3];
  int KVH=(int)k.shape[1]; int group=(KVH>0)?(H/KVH):1;
  if (k.shape[3] != D || kq.shape[3] != D) throw std::runtime_error("apa_selective_bwd: q/k/kq dim mismatch");
  if (v.shape[0] != B || v.shape[1] != KVH || v.shape[2] != S) throw std::runtime_error("apa_selective_bwd: v shape mismatch");
  if (dO.shape[0] != B || dO.shape[1] != H || dO.shape[2] != L || dO.shape[3] != VD) throw std::runtime_error("apa_selective_bwd: dO shape mismatch");
  NDArray dq({(int64_t)B,(int64_t)H,(int64_t)L,(int64_t)D}, q.dtype, q.device);
  NDArray dk = NDArray::zeros({(int64_t)B,(int64_t)KVH,(int64_t)S,(int64_t)D}, q.dtype, q.device);
  NDArray dv = NDArray::zeros({(int64_t)B,(int64_t)KVH,(int64_t)S,(int64_t)VD}, q.dtype, q.device);
  int rows=B*H*L, threads=128;
  if (rows>0) {
    int cap = D > VD ? D : VD;
    if (cap>TC_APA_MAXD) throw std::runtime_error("apa_selective_bwd: head_dim exceeds cap");
    DISPATCH_FLOAT(q.dtype, T, {
      auto launch=[&](auto tag){ constexpr int DMAX=decltype(tag)::value;
        apa_selective_bwd_kernel<T,DMAX><<<rows,threads>>>(
          static_cast<T*>(q.data_ptr()),static_cast<T*>(k.data_ptr()),
          static_cast<T*>(kq.data_ptr()),static_cast<T*>(v.data_ptr()),
          static_cast<T*>(dO.data_ptr()),
          static_cast<float*>(lse.data_ptr()),static_cast<float*>(thr.data_ptr()),
          static_cast<T*>(dq.data_ptr()),static_cast<T*>(dk.data_ptr()),
          static_cast<T*>(dv.data_ptr()),
          B,H,L,S,D,VD,scale,is_causal?1:0,KVH,group); };
      if (cap<=64) launch(std::integral_constant<int,64>{});
      else if (cap<=128) launch(std::integral_constant<int,128>{});
      else if (cap<=256) launch(std::integral_constant<int,256>{});
      else launch(std::integral_constant<int,512>{});
    });
    cuda_check_last("apa_selective_bwd");
  }
  return {dq, dk, dv};
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
// into `out` (rows, S).
//
// Causal/window bounds (Phase 3.1, board item 4a): two calling conventions.
//   1) Sentinel (legacy): Lq <= 0. Caller has ALREADY baked causal/window
//      masking into bulk/rank as a large-negative additive bias (functional.py
//      _causal_mask, -1e4). Masked keys are detected via bk <= MASK_LIM and
//      excluded from the stat; they still get read once (still O(S) memory
//      traffic on both bulk and rank) but need no separate mask tensor.
//   2) Index-arithmetic (new): Lq > 0. No mask tensor is read or required —
//      each row loops only over its valid key range, computed from the exact
//      bottom-right causal convention in functional.py's _causal_mask
//      (`np.triu(full(L,S,-1e4), k=1+(S-L))`, i.e. row i sees keys
//      0..(S-L)+i inclusive) and, for sliding-window layers, GPT-OSS's
//      _gpt_oss_attention_mask (`k_abs <= q_abs & k_abs > q_abs - window`,
//      i.e. keys (q_abs-window, q_abs] inclusive). row0 is the ABSOLUTE
//      query-chunk start (the tiled drivers slice queries into blocks; a row's
//      absolute query index is row0 + (r % Lq)); Lq is the FULL query length L
//      (not the chunk length) so S-Lq is the correct cache-prefix offset.
//      window <= 0 means no sliding window (full causal from key 0).
//
// BOUNDED is a compile-time template parameter (same dispatch pattern as A5's
// WCOOP): the launcher picks the instantiation on Lq>0. BOUNDED=false must
// compile to codegen identical to the pre-3.1 kernel — a first draft carried
// the bounds logic as a RUNTIME branch and the lead's regression scan caught
// the legacy path ~9% slower at prefill_L512-class shapes (0.49→0.536ms)
// from the added branch + register growth; gate receipt = ptxas registers
// back to the pre-3.1 values for the BOUNDED=false instantiation.
template <typename T, bool BOUNDED>
__global__ void apa_blend_softmax_kernel2(const T* bulk, const T* rank, T* out,
                                          int S, float zthr,
                                          int Lchunk, int Lq, int64_t row0,
                                          int window) {
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
  int lo = 0, hi = S;  // valid key range [lo, hi) when BOUNDED
  if constexpr (BOUNDED) {
    // r flattens as (b*H + h)*Lchunk + i_local over THIS launch's rows only
    // (Lchunk = this chunk's query count, e.g. `blk` in the tiled callers) —
    // Lq (the FULL query length L) is a different quantity, used only to
    // compute the absolute cache-prefix offset S-Lq below. Using Lq here
    // instead of Lchunk was Phase 3.1's first-draft bug: caught by a
    // multi-chunk parity re-run (attn_block < L), max|Δout| ~21 vs ~0.
    int i_local = r % Lchunk;
    int64_t abs_i = row0 + i_local;
    int64_t valid_hi = (int64_t)S - Lq + abs_i + 1;  // bottom-right causal bound
    hi = (int)(valid_hi < 0 ? 0 : (valid_hi > S ? S : valid_hi));
    lo = (window > 0) ? (int)((hi - window) < 0 ? 0 : (hi - window)) : 0;
  }
  __shared__ float red[256];
  float sum = 0.f, sumsq = 0.f, vcount = 0.f;
  if constexpr (BOUNDED) {
    for (int j = lo + tid; j < hi; j += nt) {
      float a = fabsf(ld<T>(brow, j)); sum += a; sumsq += a*a; vcount += 1.f;
    }
  } else {
    for (int j = tid; j < S; j += nt) {
      float bk = ld<T>(brow, j);
      if (bk <= MASK_LIM) continue;
      float a = fabsf(bk); sum += a; sumsq += a*a; vcount += 1.f;
    }
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
  // Selection on |bulk|: a masked key (bulk <= MASK_LIM, sentinel path) keeps
  // its large-negative bulk value (exp()s to ~0); otherwise refine to rank
  // when |bulk| >= thr. Bounds path: every key in [lo,hi) is valid by
  // construction, no sentinel check needed.
  if constexpr (BOUNDED) {
    for (int j = lo + tid; j < hi; j += nt) {
      float bk = ld<T>(brow, j);
      float sc = (fabsf(bk) >= thr) ? ld<T>(rrow, j) : bk;
      float m_new = fmaxf(m, sc);
      l = l * __expf(m - m_new) + __expf(sc - m_new);
      m = m_new;
    }
  } else {
    for (int j = tid; j < S; j += nt) {
      float bk = ld<T>(brow, j);
      float sc = (bk <= MASK_LIM) ? bk : ((fabsf(bk) >= thr) ? ld<T>(rrow, j) : bk);
      float m_new = fmaxf(m, sc);
      l = l * __expf(m - m_new) + __expf(sc - m_new);
      m = m_new;
    }
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
  // denormal/edge behaviour) — weight is ~0 there anyway. Bounds path: every
  // out-of-range column (both below lo and at/above hi) must be written 0 —
  // the caller's cat/matmul reads the full (rows,S) row, so zero it rather
  // than leaving it uninitialized.
  if constexpr (BOUNDED) {
    for (int j = tid; j < S; j += nt) {
      if (j < lo || j >= hi) { st<T>(orow, j, 0.0f); continue; }
      float bk = ld<T>(brow, j);
      float sc = (fabsf(bk) >= thr) ? ld<T>(rrow, j) : bk;
      st<T>(orow, j, __expf(fmaxf(sc - gmax, -88.f)) * inv);
    }
  } else {
    for (int j = tid; j < S; j += nt) {
      float bk = ld<T>(brow, j);
      float sc = (bk <= MASK_LIM) ? bk : ((fabsf(bk) >= thr) ? ld<T>(rrow, j) : bk);
      st<T>(orow, j, __expf(fmaxf(sc - gmax, -88.f)) * inv);
    }
  }
}

NDArray apa_blend_softmax(const NDArray& bulk, const NDArray& rank,
                          float zthr, const NDArray* /*unused*/,
                          int Lq, int64_t row0, int window) {
  int nd = bulk.ndim();
  int64_t S = bulk.shape[nd-1];
  // Lchunk = this call's query-row count (bulk's second-to-last dim, e.g.
  // the tiled callers' `blk`) — NOT Lq (the model's full query length),
  // which only enters the S-Lq cache-prefix offset. Falls back to `rows`
  // itself for the (rare, non-4D) case nd<2 so BOUNDED still degrades
  // safely rather than dividing by zero.
  int64_t Lchunk = (nd >= 2) ? bulk.shape[nd-2] : (bulk.numel() / (S > 0 ? S : 1));
  int64_t rows = bulk.numel() / S;
  NDArray out(bulk.shape, bulk.dtype, bulk.device);
  if (rows > 0) {
    // Compile-time dispatch (A5 WCOOP pattern): the legacy Lq<=0 sentinel
    // path runs the BOUNDED=false instantiation, codegen-identical to the
    // pre-3.1 kernel (no bounds branch, no register growth).
    DISPATCH_FLOAT(bulk.dtype, T, {
      if (Lq > 0) {
        apa_blend_softmax_kernel2<T, true><<<(int)rows, 256>>>(
            static_cast<T*>(bulk.data_ptr()), static_cast<T*>(rank.data_ptr()),
            static_cast<T*>(out.data_ptr()), (int)S, zthr, (int)Lchunk, Lq,
            row0, window);
      } else {
        apa_blend_softmax_kernel2<T, false><<<(int)rows, 256>>>(
            static_cast<T*>(bulk.data_ptr()), static_cast<T*>(rank.data_ptr()),
            static_cast<T*>(out.data_ptr()), (int)S, zthr, (int)Lchunk, Lq,
            row0, window);
      }
    });
    cuda_check_last("apa_blend_softmax");
  }
  return out;
}

// Bounds convention identical to apa_blend_softmax_kernel2 above. Lq <= 0 is
// the legacy sentinel path (mask baked in as -1e4 bias, MASK_LIM-detected);
// Lq > 0 is index-arithmetic: row0 is the absolute query-chunk start, Lq is
// the FULL query length L, window <= 0 means full causal (no sliding). L here
// (the kernel's existing 3rd param) is the CHUNK length used only to recover
// which head `h` a flat row belongs to (r / L) % H — unrelated to Lq.
// BOUNDED dispatch as in apa_blend_softmax_kernel2: BOUNDED=false must be
// codegen-identical to the pre-3.1 kernel (lead's regression scan receipt).
template <typename T, bool BOUNDED>
__global__ void apa_blend_softmax_sink_kernel(const T* bulk, const T* rank,
                                              const T* sinks, T* out,
                                              int H, int L, int S,
                                              float zthr,
                                              int Lq, int64_t row0, int window) {
  int r = blockIdx.x;
  int tid = threadIdx.x, nt = blockDim.x;
  int h = (r / L) % H;
  const T* brow = bulk + (int64_t)r * S;
  const T* rrow = rank + (int64_t)r * S;
  T* orow = out + (int64_t)r * S;

  const float MASK_LIM = -5e3f;
  int lo = 0, hi = S;
  if constexpr (BOUNDED) {
    // r flattens as (b*H + h)*L + i_local (L = this launch's chunk length,
    // the SAME L used for `h` above) — Lq (full query length) only enters
    // the S-Lq cache-prefix offset. Using Lq here instead of L was Phase
    // 3.1's first-draft bug (matches the non-sink kernel's fix above).
    int i_local = r % L;
    int64_t abs_i = row0 + i_local;
    int64_t valid_hi = (int64_t)S - Lq + abs_i + 1;
    hi = (int)(valid_hi < 0 ? 0 : (valid_hi > S ? S : valid_hi));
    lo = (window > 0) ? (int)((hi - window) < 0 ? 0 : (hi - window)) : 0;
  }
  __shared__ float red[256];
  float sum = 0.f, sumsq = 0.f, vcount = 0.f;
  if constexpr (BOUNDED) {
    for (int j = lo + tid; j < hi; j += nt) {
      float a = fabsf(ld<T>(brow, j));
      sum += a; sumsq += a * a; vcount += 1.f;
    }
  } else {
    for (int j = tid; j < S; j += nt) {
      float bk = ld<T>(brow, j);
      if (bk <= MASK_LIM) continue;
      float a = fabsf(bk);
      sum += a;
      sumsq += a * a;
      vcount += 1.f;
    }
  }
  red[tid] = sum; __syncthreads();
  for (int o = nt / 2; o > 0; o >>= 1) {
    if (tid < o) red[tid] += red[tid + o];
    __syncthreads();
  }
  float total = red[0]; __syncthreads();
  red[tid] = sumsq; __syncthreads();
  for (int o = nt / 2; o > 0; o >>= 1) {
    if (tid < o) red[tid] += red[tid + o];
    __syncthreads();
  }
  float total_sq = red[0]; __syncthreads();
  red[tid] = vcount; __syncthreads();
  for (int o = nt / 2; o > 0; o >>= 1) {
    if (tid < o) red[tid] += red[tid + o];
    __syncthreads();
  }
  float cnt = fmaxf(red[0], 1.f); __syncthreads();
  float mean = total / cnt;
  float thr = mean + zthr * sqrtf(fmaxf(total_sq / cnt - mean * mean, 0.f));

  float sink = ld<T>(sinks, h);
  float m = -1e30f, l = 0.f;
  if constexpr (BOUNDED) {
    for (int j = lo + tid; j < hi; j += nt) {
      float bk = ld<T>(brow, j);
      float sc = (fabsf(bk) >= thr) ? ld<T>(rrow, j) : bk;
      float m_new = fmaxf(m, sc);
      l = l * __expf(m - m_new) + __expf(sc - m_new);
      m = m_new;
    }
  } else {
    for (int j = tid; j < S; j += nt) {
      float bk = ld<T>(brow, j);
      float sc = (bk <= MASK_LIM) ? bk : ((fabsf(bk) >= thr) ? ld<T>(rrow, j) : bk);
      float m_new = fmaxf(m, sc);
      l = l * __expf(m - m_new) + __expf(sc - m_new);
      m = m_new;
    }
  }
  if (tid == 0) {
    float m_new = fmaxf(m, sink);
    l = l * __expf(m - m_new) + __expf(sink - m_new);
    m = m_new;
  }

  red[tid] = m; __syncthreads();
  for (int o = nt / 2; o > 0; o >>= 1) {
    if (tid < o) red[tid] = fmaxf(red[tid], red[tid + o]);
    __syncthreads();
  }
  float gmax = red[0]; __syncthreads();
  red[tid] = l * __expf(m - gmax); __syncthreads();
  for (int o = nt / 2; o > 0; o >>= 1) {
    if (tid < o) red[tid] += red[tid + o];
    __syncthreads();
  }
  float denom = red[0]; __syncthreads();
  float inv = denom > 0.f ? 1.f / denom : 0.f;

  if constexpr (BOUNDED) {
    for (int j = tid; j < S; j += nt) {
      if (j < lo || j >= hi) { st<T>(orow, j, 0.0f); continue; }
      float bk = ld<T>(brow, j);
      float sc = (fabsf(bk) >= thr) ? ld<T>(rrow, j) : bk;
      st<T>(orow, j, __expf(fmaxf(sc - gmax, -88.f)) * inv);
    }
  } else {
    for (int j = tid; j < S; j += nt) {
      float bk = ld<T>(brow, j);
      float sc = (bk <= MASK_LIM) ? bk : ((fabsf(bk) >= thr) ? ld<T>(rrow, j) : bk);
      st<T>(orow, j, __expf(fmaxf(sc - gmax, -88.f)) * inv);
    }
  }
}

NDArray apa_blend_softmax_sink(const NDArray& bulk, const NDArray& rank,
                               const NDArray& sinks, float zthr,
                               int Lq, int64_t row0, int window) {
  if (bulk.ndim() != 4 || rank.ndim() != 4)
    throw std::runtime_error("apa_blend_softmax_sink: bulk/rank must be rank-4");
  if (bulk.dtype != rank.dtype || bulk.dtype != sinks.dtype)
    throw std::runtime_error("apa_blend_softmax_sink: dtype mismatch");
  if (rank.shape != bulk.shape)
    throw std::runtime_error("apa_blend_softmax_sink: rank shape mismatch");
  int64_t B = bulk.shape[0], H = bulk.shape[1], L = bulk.shape[2], S = bulk.shape[3];
  if (sinks.ndim() != 1 || sinks.shape[0] != H)
    throw std::runtime_error("apa_blend_softmax_sink: sinks shape mismatch");
  NDArray out(bulk.shape, bulk.dtype, bulk.device);
  int64_t rows = B * H * L;
  if (rows > 0) {
    // Compile-time dispatch (A5 WCOOP pattern): legacy Lq<=0 runs the
    // BOUNDED=false instantiation, codegen-identical to the pre-3.1 kernel.
    DISPATCH_FLOAT(bulk.dtype, T, {
      if (Lq > 0) {
        apa_blend_softmax_sink_kernel<T, true><<<(int)rows, 256>>>(
            static_cast<T*>(bulk.data_ptr()), static_cast<T*>(rank.data_ptr()),
            static_cast<T*>(sinks.data_ptr()), static_cast<T*>(out.data_ptr()),
            (int)H, (int)L, (int)S, zthr, Lq, row0, window);
      } else {
        apa_blend_softmax_sink_kernel<T, false><<<(int)rows, 256>>>(
            static_cast<T*>(bulk.data_ptr()), static_cast<T*>(rank.data_ptr()),
            static_cast<T*>(sinks.data_ptr()), static_cast<T*>(out.data_ptr()),
            (int)H, (int)L, (int)S, zthr, Lq, row0, window);
      }
    });
    cuda_check_last("apa_blend_softmax_sink");
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
// NOTE: an 8-way->4-way shared-mem bank-conflict pad for the x staging was
// tried and REVERTED (A3, KERNEL_OPT_IMPLEMENTATION_LEDGER 2026-07-07): the
// pad's extra index arithmetic cost more than the conflicts, which are
// latency-hidden behind the DRAM-bound packed-weight reads (-3.4% median on
// clean A/B). Don't re-add padding here without new evidence.
template <typename T>
__global__ void int4_gemv_kernel(
    const T* __restrict__ x, const uint8_t* __restrict__ packed,
    const __half* __restrict__ scales, const __half* __restrict__ zeros,
    T* __restrict__ y, int N, int K, int group_size, int G) {
  // x staged in NATIVE dtype: fp32 staging of a bf16/fp16 input added
  // no information and doubled shared memory (K=15360 was 60KB —
  // 1 block/SM; native bf16 is 30KB). Accumulation stays fp32.
  extern __shared__ unsigned char xs_raw[];
  T* xs = reinterpret_cast<T*>(xs_raw);
  for (int k = threadIdx.x; k < K; k += blockDim.x)
    xs[k] = x[k];
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
    const T* xk = xs + k0;
    acc += ((float)(v.x & 0x0F) * sc + ze) * ld<T>(xk, 0)
         + ((float)((v.x >> 4) & 0x0F) * sc + ze) * ld<T>(xk, 1)
         + ((float)(v.y & 0x0F) * sc + ze) * ld<T>(xk, 2)
         + ((float)((v.y >> 4) & 0x0F) * sc + ze) * ld<T>(xk, 3)
         + ((float)(v.z & 0x0F) * sc + ze) * ld<T>(xk, 4)
         + ((float)((v.z >> 4) & 0x0F) * sc + ze) * ld<T>(xk, 5)
         + ((float)(v.w & 0x0F) * sc + ze) * ld<T>(xk, 6)
         + ((float)((v.w >> 4) & 0x0F) * sc + ze) * ld<T>(xk, 7);
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
  if (M == 1 && (group_size % 8) == 0 && (K % 8) == 0) {
    const int threads = 256, warps = threads / 32;
    dim3 ggrid((int)((N + warps - 1) / warps));
    bool launched = false;
    DISPATCH_FLOAT(x.dtype, T, {
      // native-dtype staging: bf16/fp16 x needs K*2 bytes (fp32 K*4).
      size_t shmem = (size_t)K * sizeof(T);
      if (shmem <= 96 * 1024) {
        if (shmem > 48 * 1024) {
          cudaFuncSetAttribute(int4_gemv_kernel<T>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               96 * 1024);
        }
        int4_gemv_kernel<T><<<ggrid, threads, shmem>>>(
            static_cast<T*>(x.data_ptr()),
            static_cast<uint8_t*>(packed.data_ptr()),
            static_cast<__half*>(scales.data_ptr()),
            zeros.numel() ? static_cast<__half*>(zeros.data_ptr()) : nullptr,
            static_cast<T*>(out.data_ptr()), (int)N, (int)K, group_size,
            (int)G);
        launched = true;
      }
    });
    if (launched) {
      cuda_check_last("int4_gemv");
      return out;
    }
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

// -------------------------------------- fused group-32 W8A16 tile GEMM (Q2-K0b)
// y[M,N] = x[M,K] @ dequant(codes)[N,K]^T. Trinity stores one uint8 code per
// weight with signed q represented by code=q+128, plus one fp16 scale per
// [output, 32-K] group. Weight values are dequantized and ROUNDED to fp16 in
// shared memory before HMMA consumes them, matching the explicit full-dequant-
// to-fp16 reference. Accumulators remain fp32 until the final fp16 output store.
// No global (N,K) dequant buffer exists.
//
// K0b replaces K0's four-warp 64x16 / 16x64, K=32 skeleton with the same
// reuse discipline that closed EXP-APA-3/K1:
//   * one eight-warp CTA owns a 64x64 output tile;
//   * a 64-wide K slice is staged once and reused by every resident output
//     fragment (64-way reuse of both activation and weight values);
//   * activations load as aligned 16-byte vectors; codes load four-wide, with
//     one scale load warp-broadcast to the eight code vectors in each group;
//   * weights stay in their native [N,K] orientation in shared memory and feed
//     WMMA B as col-major, avoiding K0's scalar shared-memory transpose;
//   * +8 half bank skew mirrors K1's K/V staging layout;
//   * the operand stage and fp32 output spill alias in a union because their
//     lifetimes do not overlap (18,432 bytes total static shared memory).
//
// The two K0 launch-config names remain API/DET compatibility selectors and
// intentionally dispatch this one production tile. K is still only required
// to be divisible by 32: the final half-full K=64 stage is zero padded.
constexpr int kW8Group = 32;
constexpr int kW8TileM = 64;
constexpr int kW8TileN = 64;
constexpr int kW8TileK = 64;
constexpr int kW8LdK = kW8TileK + 8;
constexpr int kW8Warps = 8;
constexpr int kW8WarpN = 32;
constexpr int kW8CLd = kW8WarpN + 4;

struct __align__(16) W8A16Operands {
  __half x[kW8TileM][kW8LdK];
  // Native code orientation [output_n][k] == col-major GEMM B[K,N].
  __half w[kW8TileN][kW8LdK];
};

union __align__(16) W8A16Shared {
  W8A16Operands op;
  float c[kW8Warps][16][kW8CLd];
};

__global__ __launch_bounds__(256) void w8a16_gemm_kernel(
    const __half* __restrict__ x, const uint8_t* __restrict__ codes,
    const __half* __restrict__ scales, __half* __restrict__ y,
    int M, int N, int K, int G) {
  __shared__ W8A16Shared sh;

  using namespace nvcuda;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int warp_m = warp >> 1;  // four 16-row warp bands
  const int warp_n = warp & 1;   // each warp owns one 32-column band
  const int block_m0 = (int)blockIdx.y * kW8TileM;
  const int block_n0 = (int)blockIdx.x * kW8TileN;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc0;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc1;
  wmma::fill_fragment(acc0, 0.f);
  wmma::fill_fragment(acc1, 0.f);

  for (int k0 = 0; k0 < K; k0 += kW8TileK) {
    // Two aligned 16-byte activation vectors per thread cover 64x64 halves.
    #pragma unroll
    for (int it = 0; it < 2; ++it) {
      const int vi = tid + it * 256;
      const int lm = vi >> 3;
      const int lk = (vi & 7) << 3;
      const int m = block_m0 + lm;
      uint4 value = make_uint4(0u, 0u, 0u, 0u);
      if (m < M && k0 + lk < K) {
        value = *reinterpret_cast<const uint4*>(x + (int64_t)m * K + k0 + lk);
      }
      *reinterpret_cast<uint4*>(&sh.op.x[lm][lk]) = value;
    }

    // Four code vectors per thread cover 64x64 bytes. A half-warp owns one
    // output row, so every uchar4 load is coalesced along the native K axis.
    // Lanes 0/8 in each half-warp load the two group scales and broadcast.
    #pragma unroll
    for (int it = 0; it < 4; ++it) {
      const int vi = tid + it * 256;
      const int ln = vi >> 4;
      const int kvec = vi & 15;
      const int lk = kvec << 2;
      const int n = block_n0 + ln;
      const bool valid = n < N && k0 + lk < K;
      float scale = 0.f;
      if ((lane & 7) == 0 && valid) {
        scale = __half2float(scales[(int64_t)n * G + ((k0 + lk) >> 5)]);
      }
      const int scale_lane = (lane & 16) + ((kvec >> 3) << 3);
      scale = __shfl_sync(0xffffffffu, scale, scale_lane);

      uchar4 cv = make_uchar4(128, 128, 128, 128);
      if (valid) {
        cv = *reinterpret_cast<const uchar4*>(codes + (int64_t)n * K + k0 + lk);
      }
      __half* dst = &sh.op.w[ln][lk];
      dst[0] = __float2half_rn(((int)cv.x - 128) * scale);
      dst[1] = __float2half_rn(((int)cv.y - 128) * scale);
      dst[2] = __float2half_rn(((int)cv.z - 128) * scale);
      dst[3] = __float2half_rn(((int)cv.w - 128) * scale);
    }
    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < kW8TileK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b0;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b1;
      wmma::load_matrix_sync(a, &sh.op.x[warp_m * 16][kk], kW8LdK);
      wmma::load_matrix_sync(b0, &sh.op.w[warp_n * kW8WarpN][kk], kW8LdK);
      wmma::load_matrix_sync(b1, &sh.op.w[warp_n * kW8WarpN + 16][kk], kW8LdK);
      wmma::mma_sync(acc0, a, b0, acc0);
      wmma::mma_sync(acc1, a, b1, acc1);
    }
    // All warps must finish consuming the operand half of the union before
    // any thread overwrites it with the next K stage (or final fp32 spill).
    __syncthreads();
  }

  wmma::store_matrix_sync(&sh.c[warp][0][0], acc0, kW8CLd,
                          wmma::mem_row_major);
  wmma::store_matrix_sync(&sh.c[warp][0][16], acc1, kW8CLd,
                          wmma::mem_row_major);
  __syncwarp();
  for (int i = lane; i < 16 * kW8WarpN; i += 32) {
    const int lm = i >> 5;
    const int ln = i & 31;
    const int m = block_m0 + warp_m * 16 + lm;
    const int n = block_n0 + warp_n * kW8WarpN + ln;
    if (m < M && n < N)
      y[(int64_t)m * N + n] = __float2half_rn(sh.c[warp][lm][ln]);
  }
}

NDArray w8a16_matmul(const NDArray& x, const NDArray& codes,
                     const NDArray& scales, int launch_config) {
  const char* where = "w8a16_matmul";
  if (x.dtype != DType::Float16)
    throw std::runtime_error(std::string(where) + ": activations must be float16");
  if (codes.dtype != DType::Uint8)
    throw std::runtime_error(std::string(where) + ": codes must be uint8");
  if (scales.dtype != DType::Float16)
    throw std::runtime_error(std::string(where) + ": scales must be float16");
  if (!x.device.is_cuda() || !codes.device.is_cuda() || !scales.device.is_cuda() ||
      x.device.index != codes.device.index || x.device.index != scales.device.index)
    throw std::runtime_error(std::string(where) + ": all inputs must share one CUDA device");
  if (x.ndim() < 2)
    throw std::runtime_error(std::string(where) + ": activations must have rank >= 2");
  if (codes.ndim() != 2 || scales.ndim() != 2)
    throw std::runtime_error(std::string(where) + ": codes and scales must be rank-2");
  const int64_t K = x.shape.back();
  if (K <= 0 || (K % kW8Group) != 0)
    throw std::runtime_error(std::string(where) +
                             ": activation K must be positive and divisible by 32");
  const int64_t N = codes.shape[0];
  if (codes.shape[1] != K)
    throw std::runtime_error(std::string(where) + ": codes K mismatch");
  if (scales.shape[0] != N || scales.shape[1] != K / kW8Group)
    throw std::runtime_error(std::string(where) +
                             ": scales shape must be (out_features, K/32)");
  if (launch_config != 0 && launch_config != 1)
    throw std::runtime_error(std::string(where) + ": launch_config must be 0 or 1");
  const int64_t M = x.numel() / K;
  if (M > std::numeric_limits<int>::max() || N > std::numeric_limits<int>::max() ||
      K > std::numeric_limits<int>::max())
    throw std::runtime_error(std::string(where) + ": dimensions exceed CUDA int limits");

  Shape out_shape = x.shape;
  out_shape.back() = N;
  NDArray out(out_shape, DType::Float16, x.device);
  if (M == 0 || N == 0) return out;

  constexpr int threads = kW8Warps * 32;
  dim3 grid((unsigned)((N + kW8TileN - 1) / kW8TileN),
            (unsigned)((M + kW8TileM - 1) / kW8TileM));
  w8a16_gemm_kernel<<<grid, threads>>>(
      static_cast<__half*>(x.data_ptr()),
      static_cast<uint8_t*>(codes.data_ptr()),
      static_cast<__half*>(scales.data_ptr()),
      static_cast<__half*>(out.data_ptr()), (int)M, (int)N, (int)K,
      (int)(K / kW8Group));
  cuda_check_last(where);
  return out;
}

// ------------------------------------------------ packed INT2/INT3 quant linear
// General low-bit weight path. Rows are bit-packed little-endian:
// q[k]'s bit b is stored at bit offset k*bits+b within packed[row].
// K is explicit because 3-bit rows carry byte padding.
static inline void check_intn_bits(int bits, const char* where) {
  if (bits != 2 && bits != 3) {
    throw std::runtime_error(std::string(where) +
                             ": bits must be 2 or 3 for this native path");
  }
}

static inline int64_t intn_packed_bytes(int64_t K, int bits) {
  return (K * (int64_t)bits + 7) / 8;
}

static inline void validate_intn_weight(const NDArray& packed,
                                        const NDArray& scales,
                                        const NDArray& zeros, int bits,
                                        int64_t K, int group_size,
                                        const char* where) {
  check_intn_bits(bits, where);
  if (group_size <= 0)
    throw std::runtime_error(std::string(where) + ": group_size must be > 0");
  if (K <= 0)
    throw std::runtime_error(std::string(where) + ": in_features must be > 0");
  if ((K % group_size) != 0)
    throw std::runtime_error(std::string(where) +
                             ": in_features must be divisible by group_size");
  if (packed.dtype != DType::Uint8)
    throw std::runtime_error(std::string(where) + ": packed must be uint8");
  if (scales.dtype != DType::Float16 ||
      (zeros.numel() > 0 && zeros.dtype != DType::Float16))
    throw std::runtime_error(std::string(where) +
                             ": scales/zeros must be float16");
  if (packed.ndim() != 2 || scales.ndim() != 2)
    throw std::runtime_error(std::string(where) +
                             ": packed and scales must be rank-2");
  int64_t N = packed.shape[0];
  int64_t G = K / group_size;
  int64_t bytes = intn_packed_bytes(K, bits);
  if (packed.shape[1] != bytes)
    throw std::runtime_error(std::string(where) + ": packed byte width mismatch");
  if (scales.shape[0] != N || scales.shape[1] != G)
    throw std::runtime_error(std::string(where) + ": scales shape mismatch");
  if (zeros.numel() > 0 &&
      (zeros.ndim() != 2 || zeros.shape[0] != N || zeros.shape[1] != G))
    throw std::runtime_error(std::string(where) + ": zeros shape mismatch");
}

__device__ __forceinline__ int intn_read_q(const uint8_t* row, int64_t k,
                                           int bits) {
  int64_t bit0 = k * (int64_t)bits;
  int q = 0;
  #pragma unroll
  for (int b = 0; b < 3; ++b) {
    if (b < bits) {
      int64_t bit = bit0 + b;
      q |= (int)((row[bit >> 3] >> (bit & 7)) & 1u) << b;
    }
  }
  return q;
}

template <typename T>
__global__ void intn_dequant_t_kernel(const uint8_t* packed,
                                      const __half* scales,
                                      const __half* zeros, T* out_kn,
                                      int64_t N, int64_t K,
                                      int64_t bytes_per_row, int bits,
                                      int group_size, int64_t G, int64_t n) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  int64_t k = idx / N;
  int64_t col = idx - k * N;
  const uint8_t* row = packed + col * bytes_per_row;
  int q = intn_read_q(row, k, bits);
  int64_t g = k / group_size;
  float scale = __half2float(scales[col * G + g]);
  float zero = zeros ? __half2float(zeros[col * G + g])
                     : -(float)(1 << (bits - 1)) * scale;
  st<T>(out_kn, idx, (float)q * scale + zero);
}

NDArray intn_dequant(const NDArray& packed, const NDArray& scales,
                     const NDArray& zeros, int bits, int64_t in_features,
                     int group_size, DType out_dtype) {
  int64_t K = in_features;
  validate_intn_weight(packed, scales, zeros, bits, K, group_size,
                       "intn_dequant");
  int64_t N = packed.shape[0];
  int64_t G = K / group_size;
  int64_t bytes = intn_packed_bytes(K, bits);
  NDArray out({K, N}, out_dtype, packed.device);
  int64_t n = K * N;
  if (n) {
    DISPATCH_FLOAT(out_dtype, T, {
      intn_dequant_t_kernel<T><<<nblk(n), kT>>>(
          static_cast<uint8_t*>(packed.data_ptr()),
          static_cast<__half*>(scales.data_ptr()),
          zeros.numel() ? static_cast<__half*>(zeros.data_ptr()) : nullptr,
          static_cast<T*>(out.data_ptr()), N, K, bytes, bits, group_size, G, n);
    });
    cuda_check_last("intn_dequant");
  }
  return out;
}

NDArray intn_linear(const NDArray& x, const NDArray& packed,
                    const NDArray& scales, const NDArray& zeros, int bits,
                    int64_t in_features, int group_size) {
  if (x.ndim() < 1)
    throw std::runtime_error("intn_linear: x must have at least one dimension");
  if (x.shape[x.ndim() - 1] != in_features)
    throw std::runtime_error("intn_linear: x last dimension mismatches K");
  NDArray w_kn = intn_dequant(packed, scales, zeros, bits, in_features,
                              group_size, x.dtype);
  return matmul(x, w_kn);
}

template <typename T>
__global__ void intn_gemm_fused_kernel(
    const T* x, const uint8_t* packed, const __half* scales, const __half* zeros,
    T* y, int M, int N, int K, int bytes_per_row, int bits, int group_size,
    int G) {
  __shared__ float xs[TC_I4_TILE][TC_I4_TILE];
  __shared__ float ws[TC_I4_TILE][TC_I4_TILE];
  int row = blockIdx.y * TC_I4_TILE + threadIdx.y;
  int col = blockIdx.x * TC_I4_TILE + threadIdx.x;
  float acc = 0.f;
  for (int k0 = 0; k0 < K; k0 += TC_I4_TILE) {
    int kx = k0 + threadIdx.x;
    xs[threadIdx.y][threadIdx.x] =
        (row < M && kx < K) ? ld<T>(x, (int64_t)row * K + kx) : 0.f;
    int kw = k0 + threadIdx.y;
    float wv = 0.f;
    if (col < N && kw < K) {
      const uint8_t* prow = packed + (int64_t)col * bytes_per_row;
      int q = intn_read_q(prow, kw, bits);
      int g = kw / group_size;
      float sc = __half2float(scales[(int64_t)col * G + g]);
      float ze = zeros ? __half2float(zeros[(int64_t)col * G + g])
                       : -(float)(1 << (bits - 1)) * sc;
      wv = (float)q * sc + ze;
    }
    ws[threadIdx.y][threadIdx.x] = wv;
    __syncthreads();
    #pragma unroll
    for (int t = 0; t < TC_I4_TILE; ++t) {
      acc += xs[threadIdx.y][t] * ws[t][threadIdx.x];
    }
    __syncthreads();
  }
  if (row < M && col < N) st<T>(y, (int64_t)row * N + col, acc);
}

template <typename T>
__global__ void intn_gemv_kernel(
    const T* __restrict__ x, const uint8_t* __restrict__ packed,
    const __half* __restrict__ scales, const __half* __restrict__ zeros,
    T* __restrict__ y, int N, int K, int bytes_per_row, int bits,
    int group_size, int G) {
  extern __shared__ unsigned char xs_raw[];
  T* xs = reinterpret_cast<T*>(xs_raw);
  for (int k = threadIdx.x; k < K; k += blockDim.x) xs[k] = x[k];
  __syncthreads();
  const int warps = blockDim.x >> 5;
  const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n = blockIdx.x * warps + wid;
  if (n >= N) return;
  const uint8_t* row = packed + (int64_t)n * bytes_per_row;
  const __half* srow = scales + (int64_t)n * G;
  const __half* zrow = zeros ? zeros + (int64_t)n * G : nullptr;
  float acc = 0.f;
  for (int k = lane; k < K; k += 32) {
    int q = intn_read_q(row, k, bits);
    int g = k / group_size;
    float sc = __half2float(srow[g]);
    float ze = zrow ? __half2float(zrow[g])
                    : -(float)(1 << (bits - 1)) * sc;
    acc += ((float)q * sc + ze) * ld<T>(xs, k);
  }
  for (int off = 16; off; off >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  }
  if (lane == 0) st<T>(y, n, acc);
}

NDArray intn_linear_fused(const NDArray& x, const NDArray& packed,
                          const NDArray& scales, const NDArray& zeros,
                          int bits, int64_t in_features, int group_size) {
  if (x.ndim() < 1)
    throw std::runtime_error(
        "intn_linear_fused: x must have at least one dimension");
  int64_t K = in_features;
  if (x.shape[x.ndim() - 1] != K)
    throw std::runtime_error("intn_linear_fused: x last dimension mismatches K");
  validate_intn_weight(packed, scales, zeros, bits, K, group_size,
                       "intn_linear_fused");
  int64_t M = x.numel() / K;
  int64_t N = packed.shape[0];
  int64_t G = K / group_size;
  int64_t bytes = intn_packed_bytes(K, bits);
  Shape os;
  for (int d = 0; d < x.ndim() - 1; ++d) os.push_back(x.shape[d]);
  os.push_back(N);
  NDArray out(os, x.dtype, x.device);
  if (M == 0 || N == 0) return out;
  if (M == 1) {
    const int threads = 256, warps = threads / 32;
    dim3 ggrid((int)((N + warps - 1) / warps));
    bool launched = false;
    DISPATCH_FLOAT(x.dtype, T, {
      size_t shmem = (size_t)K * sizeof(T);
      if (shmem <= 96 * 1024) {
        if (shmem > 48 * 1024) {
          cudaFuncSetAttribute(intn_gemv_kernel<T>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               96 * 1024);
        }
        intn_gemv_kernel<T><<<ggrid, threads, shmem>>>(
            static_cast<T*>(x.data_ptr()),
            static_cast<uint8_t*>(packed.data_ptr()),
            static_cast<__half*>(scales.data_ptr()),
            zeros.numel() ? static_cast<__half*>(zeros.data_ptr()) : nullptr,
            static_cast<T*>(out.data_ptr()), (int)N, (int)K, (int)bytes, bits,
            group_size, (int)G);
        launched = true;
      }
    });
    if (launched) {
      cuda_check_last("intn_gemv");
      return out;
    }
  }
  dim3 block(TC_I4_TILE, TC_I4_TILE);
  dim3 grid((int)((N + TC_I4_TILE - 1) / TC_I4_TILE),
            (int)((M + TC_I4_TILE - 1) / TC_I4_TILE));
  DISPATCH_FLOAT(x.dtype, T, {
    intn_gemm_fused_kernel<T><<<grid, block>>>(
        static_cast<T*>(x.data_ptr()), static_cast<uint8_t*>(packed.data_ptr()),
        static_cast<__half*>(scales.data_ptr()),
        zeros.numel() ? static_cast<__half*>(zeros.data_ptr()) : nullptr,
        static_cast<T*>(out.data_ptr()), (int)M, (int)N, (int)K, (int)bytes,
        bits, group_size, (int)G);
  });
  cuda_check_last("intn_linear_fused");
  return out;
}

// ------------------------------------------------ GPT-OSS MXFP4 expert linear
// GPT-OSS stores each expert projection as:
//   blocks [N, G, 16] uint8  -> 16 bytes = 32 FP4 values per K-group
//   scales [N, G] uint8      -> E8M0 exponent, scale = 2^(scale - 127)
// Dequant layout is conceptually W_kn [K, N], but this op never materializes it.
// Branchless E2M1 decode. nibble = sign(1) | exp(2) | mant(1). ncu measured
// 60.94% branch efficiency (~16k divergent branches/launch) on the 16-way
// switch this replaces (Project-Tensor kernel-opt A2, addendum 1). Per-lane
// nibbles differ across a warp, so a switch/table-index compiles to a
// divergent branch chain; this builds the IEEE-754 fp32 bit pattern directly
// with integer arithmetic + predicated (branchless) selects instead:
//   - e2 = (q>>1)&3 is the E2M1 exponent field, m = q&1 the mantissa bit.
//   - For e2!=0 (normal): value = (1+0.5*m) * 2^(e2-1)
//     -> fp32 biased exponent = 126+e2, mantissa top bit = m.
//   - For e2==0, m==1 (subnormal 0.5): equals normal encoding with e2=0
//     -> fp32 exponent 126+e2 == 126, but mantissa bit must be forced 0
//     (0.5 = 1.0 x 2^-1, not 1.5 x 2^-1), so mantissa is gated on e2!=0.
//   - For q&0x7==0 (true zero): exponent field must be 0, gated by nz.
// `nz`/`enz` are 0/1 ints from boolean predicates (SASS `setp` + select, not
// a branch) — no per-lane divergent control flow. Verified bit-identical to
// the old switch for all 16 nibble values (see test_mxfp4_branchless_decode
// unit test / TC_MXFP4_REFERENCE_DECODE below).
__device__ __forceinline__ float mxfp4_value(int q) {
  q &= 0x0F;
  int sign_bit = (q >> 3) & 1;
  int e2 = (q >> 1) & 0x3;
  int m = q & 1;
  int nz = (q & 0x7) != 0;             // whole magnitude nibble nonzero
  int enz = e2 != 0;                   // exponent field nonzero
  unsigned exp_field = (unsigned)((126 + e2) * nz);
  unsigned mant_field = (unsigned)((m << 22) * enz);
  unsigned bits = ((unsigned)sign_bit << 31) | (exp_field << 23) | mant_field;
  float out;
  memcpy(&out, &bits, sizeof(out));
  return out;
}

#ifdef TC_MXFP4_REFERENCE_DECODE
// Reference implementation kept for the exhaustive parity unit test only
// (the original 16-way switch, pre-A2). Not compiled into the shipped
// kernels; enabled only when building the standalone parity-test TU.
__device__ __forceinline__ float mxfp4_value_reference(int q) {
  switch (q & 0x0F) {
    case 0x0: return 0.0f;
    case 0x1: return 0.5f;
    case 0x2: return 1.0f;
    case 0x3: return 1.5f;
    case 0x4: return 2.0f;
    case 0x5: return 3.0f;
    case 0x6: return 4.0f;
    case 0x7: return 6.0f;
    case 0x8: return -0.0f;
    case 0x9: return -0.5f;
    case 0xA: return -1.0f;
    case 0xB: return -1.5f;
    case 0xC: return -2.0f;
    case 0xD: return -3.0f;
    case 0xE: return -4.0f;
    default: return -6.0f;
  }
}
#endif  // TC_MXFP4_REFERENCE_DECODE

__device__ __forceinline__ float mxfp4_read_weight(
    const uint8_t* __restrict__ blocks, const uint8_t* __restrict__ scales,
    int n, int k, int G) {
  int g = k >> 5;                       // 32 dequantized weights per group
  int j = k & 31;
  uint8_t byte = blocks[((int64_t)n * G + g) * 16 + (j >> 1)];
  int q = (j & 1) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
  int exp = (int)scales[(int64_t)n * G + g] - 127;
  return ldexpf(mxfp4_value(q), exp);
}

// Unscaled nibble decode for mxfp4_gemv_kernel: n and g are warp-uniform
// there (32 lanes stride k by 32, so g=k>>5 is identical across the warp for
// each loop iteration) — the plain mxfp4_read_weight above would issue the
// same scales[n*G+g] global read redundantly from all 32 lanes every
// iteration. Splitting the block-byte fetch (per-lane, genuinely divergent
// addresses) from the scale fetch (warp-uniform) lets the gemv loop hoist
// one scale read per group instead of one per element (A2, addendum 1).
__device__ __forceinline__ float mxfp4_value_at(
    const uint8_t* __restrict__ blocks, int n, int k, int G) {
  int g = k >> 5;
  int j = k & 31;
  uint8_t byte = blocks[((int64_t)n * G + g) * 16 + (j >> 1)];
  int q = (j & 1) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
  return mxfp4_value(q);
}

template <typename T>
__global__ void mxfp4_gemm_kernel(
    const T* __restrict__ x, const uint8_t* __restrict__ blocks,
    const uint8_t* __restrict__ scales, T* __restrict__ y,
    int M, int N, int K, int G) {
  __shared__ float xs[TC_I4_TILE][TC_I4_TILE];
  __shared__ float ws[TC_I4_TILE][TC_I4_TILE];
  int row = blockIdx.y * TC_I4_TILE + threadIdx.y;
  int col = blockIdx.x * TC_I4_TILE + threadIdx.x;
  float acc = 0.f;
  for (int k0 = 0; k0 < K; k0 += TC_I4_TILE) {
    int kx = k0 + threadIdx.x;
    xs[threadIdx.y][threadIdx.x] =
        (row < M && kx < K) ? ld<T>(x, (int64_t)row * K + kx) : 0.f;
    int kw = k0 + threadIdx.y;
    ws[threadIdx.y][threadIdx.x] =
        (col < N && kw < K) ? mxfp4_read_weight(blocks, scales, col, kw, G)
                            : 0.f;
    __syncthreads();
    #pragma unroll
    for (int t = 0; t < TC_I4_TILE; ++t) {
      acc += xs[threadIdx.y][t] * ws[t][threadIdx.x];
    }
    __syncthreads();
  }
  if (row < M && col < N) st<T>(y, (int64_t)row * N + col, acc);
}

template <typename T>
__global__ void mxfp4_gemv_kernel(
    const T* __restrict__ x, const uint8_t* __restrict__ blocks,
    const uint8_t* __restrict__ scales, T* __restrict__ y,
    int N, int K, int G) {
  extern __shared__ unsigned char xs_raw[];
  T* xs = reinterpret_cast<T*>(xs_raw);
  for (int k = threadIdx.x; k < K; k += blockDim.x) xs[k] = x[k];
  __syncthreads();
  const int warps = blockDim.x >> 5;
  const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n = blockIdx.x * warps + wid;
  if (n >= N) return;
  float acc = 0.f;
  for (int k = lane; k < K; k += 32) {
    // g = k>>5 is identical across all 32 lanes this iteration (k spans
    // exactly one group of 32 per iteration) -> one scale read per group,
    // not one per lane/element.
    int g = k >> 5;
    int exp = (int)scales[(int64_t)n * G + g] - 127;
    acc += ldexpf(mxfp4_value_at(blocks, n, k, G), exp) * ld<T>(xs, k);
  }
  for (int off = 16; off; off >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  }
  if (lane == 0) st<T>(y, n, acc);
}

static inline void validate_mxfp4_weight(const NDArray& blocks,
                                         const NDArray& scales,
                                         const char* where) {
  if (blocks.dtype != DType::Uint8 || scales.dtype != DType::Uint8)
    throw std::runtime_error(std::string(where) +
                             ": blocks and scales must be uint8");
  if (blocks.ndim() != 3 || scales.ndim() != 2)
    throw std::runtime_error(std::string(where) +
                             ": blocks must be rank-3 and scales rank-2");
  if (blocks.shape[2] != 16)
    throw std::runtime_error(std::string(where) +
                             ": blocks last dimension must be 16 bytes");
  if (blocks.shape[0] != scales.shape[0] ||
      blocks.shape[1] != scales.shape[1])
    throw std::runtime_error(std::string(where) + ": scales shape mismatch");
}

static inline void validate_mxfp4_expert_weight(const NDArray& blocks,
                                                const NDArray& scales,
                                                int64_t expert_idx,
                                                const char* where) {
  if (blocks.dtype != DType::Uint8 || scales.dtype != DType::Uint8)
    throw std::runtime_error(std::string(where) +
                             ": blocks and scales must be uint8");
  if (blocks.ndim() != 4 || scales.ndim() != 3)
    throw std::runtime_error(std::string(where) +
                             ": blocks must be rank-4 and scales rank-3");
  if (blocks.shape[3] != 16)
    throw std::runtime_error(std::string(where) +
                             ": blocks last dimension must be 16 bytes");
  if (blocks.shape[0] != scales.shape[0] ||
      blocks.shape[1] != scales.shape[1] ||
      blocks.shape[2] != scales.shape[2])
    throw std::runtime_error(std::string(where) + ": scales shape mismatch");
  if (expert_idx < 0 || expert_idx >= blocks.shape[0])
    throw std::runtime_error(std::string(where) + ": expert_idx out of range");
}

static NDArray mxfp4_linear_launch(const NDArray& x,
                                   const uint8_t* blocks_ptr,
                                   const uint8_t* scales_ptr,
                                   int64_t N, int64_t G,
                                   const char* where) {
  if (x.ndim() < 1)
    throw std::runtime_error(std::string(where) +
                             ": x must have at least one dimension");
  int64_t K = G * 32;
  if (x.shape[x.ndim() - 1] != K)
    throw std::runtime_error(std::string(where) +
                             ": x last dimension mismatches K");
  int64_t M = x.numel() / K;
  Shape os;
  for (int d = 0; d < x.ndim() - 1; ++d) os.push_back(x.shape[d]);
  os.push_back(N);
  NDArray out(os, x.dtype, x.device);
  if (M == 0 || N == 0) return out;

  if (M == 1) {
    const int threads = 256, warps = threads / 32;
    dim3 ggrid((int)((N + warps - 1) / warps));
    bool launched = false;
    DISPATCH_FLOAT(x.dtype, T, {
      size_t shmem = (size_t)K * sizeof(T);
      if (shmem <= 96 * 1024) {
        if (shmem > 48 * 1024) {
          cudaFuncSetAttribute(mxfp4_gemv_kernel<T>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               96 * 1024);
        }
        mxfp4_gemv_kernel<T><<<ggrid, threads, shmem>>>(
            static_cast<T*>(x.data_ptr()),
            blocks_ptr,
            scales_ptr,
            static_cast<T*>(out.data_ptr()), (int)N, (int)K, (int)G);
        launched = true;
      }
    });
    if (launched) {
      cuda_check_last(where);
      return out;
    }
  }

  dim3 block(TC_I4_TILE, TC_I4_TILE);
  dim3 grid((int)((N + TC_I4_TILE - 1) / TC_I4_TILE),
            (int)((M + TC_I4_TILE - 1) / TC_I4_TILE));
  DISPATCH_FLOAT(x.dtype, T, {
    mxfp4_gemm_kernel<T><<<grid, block>>>(
        static_cast<T*>(x.data_ptr()),
        blocks_ptr,
        scales_ptr,
        static_cast<T*>(out.data_ptr()), (int)M, (int)N, (int)K, (int)G);
  });
  cuda_check_last(where);
  return out;
}

NDArray mxfp4_linear(const NDArray& x, const NDArray& blocks,
                     const NDArray& scales) {
  validate_mxfp4_weight(blocks, scales, "mxfp4_linear");
  return mxfp4_linear_launch(
      x,
      static_cast<uint8_t*>(blocks.data_ptr()),
      static_cast<uint8_t*>(scales.data_ptr()),
      blocks.shape[0],
      blocks.shape[1],
      "mxfp4_linear");
}

NDArray mxfp4_linear_expert(const NDArray& x, const NDArray& blocks,
                            const NDArray& scales, int64_t expert_idx) {
  validate_mxfp4_expert_weight(blocks, scales, expert_idx,
                               "mxfp4_linear_expert");
  int64_t N = blocks.shape[1];
  int64_t G = blocks.shape[2];
  int64_t block_offset = expert_idx * N * G * 16;
  int64_t scale_offset = expert_idx * N * G;
  return mxfp4_linear_launch(
      x,
      static_cast<uint8_t*>(blocks.data_ptr()) + block_offset,
      static_cast<uint8_t*>(scales.data_ptr()) + scale_offset,
      N,
      G,
      "mxfp4_linear_expert");
}

// =====================================================================
// KV-cache INT4 storage (D-grouped, symmetric-8) — the tail-law primitive.
//
// int4_dequant above is WEIGHT-shaped: it packs along K (in-features) and
// emits a transposed (K,N) matrix for a GEMM. The KV cache is a different
// animal: (B, KV, S, D), packed along the contiguous innermost dim D, and
// reads come out as a (B, KV, n, D) SLICE [lo:lo+n) — never the whole cap.
// So it needs its own pair of kernels. Convention matches rt_int4 / the
// QAT q4_0 grid already used for weights: group-32, symmetric-8
//   scale = max|x_group| / 8   (8 == half the 4-bit code span)
//   q     = round(x/scale) + 8   clamped to [0,15]
//   x_hat = (q - 8) * scale
// Packed layout: (B, KV, S, D/2) uint8 — element 2t -> low nibble of byte t,
// element 2t+1 -> high nibble. Scales: (B, KV, S, D/32) in COMPUTE dtype
// (bf16 here; a fp16-hardcode would dtype-mismatch the attention matmul,
// the same trap _kv_quant documents). The segfault that killed the naive
// uint8 strided-copy was st<uint8> instantiated through DISPATCH_FLOAT —
// these kernels touch uint8 only as a raw uint8_t* (pack writes bytes by
// hand; unpack READS bytes and st<T>'s a FLOAT out), so that template
// arm is never instantiated.
//
// GROUP must divide D. NIB packing assumes D even (always true: D=512/256).

// pack: float (B,KV,S,D) + precomputed scales (B,KV,S,G) -> uint8 (B,KV,S,D/2)
// one thread per OUTPUT BYTE (== two source elements, same group since
// GROUP is even and >= 2).
template <typename T>
__global__ void kv_int4_pack_kernel(const T* __restrict__ x,
                                    const T* __restrict__ scales,
                                    uint8_t* __restrict__ packed,
                                    int64_t D, int group, int64_t G,
                                    int64_t nbytes) {
  int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;  // output byte
  if (b >= nbytes) return;
  int64_t row = b / (D / 2);            // flattened (B*KV*S) row index
  int64_t bin = b - row * (D / 2);      // byte within the row [0, D/2)
  int64_t d0 = bin * 2;                 // first source element in this byte
  int64_t g = d0 / group;               // group index (both elems share it)
  float s = ld<T>(scales, row * G + g);
  float invs = s > 0.f ? s : 1.f;       // guard div-by-zero (all-zero group)
  int64_t base = row * D + d0;
  float x0 = ld<T>(x, base);
  float x1 = ld<T>(x, base + 1);
  // DIVIDE (not reciprocal-multiply) so the quant grid is bit-identical to
  // the rt_int4 reference that measured +5.55% ppl — recip-mul flips ~2 in
  // 35k codes at half-step boundaries, harmless but avoidable for free.
  int q0 = (int)lrintf(x0 / invs) + 8; q0 = q0 < 0 ? 0 : (q0 > 15 ? 15 : q0);
  int q1 = (int)lrintf(x1 / invs) + 8; q1 = q1 < 0 ? 0 : (q1 > 15 ? 15 : q1);
  packed[b] = (uint8_t)((q1 << 4) | q0);   // even -> low nibble, odd -> high
}

// unpack a slice: uint8 (B,KV,S,D/2) + scales (B,KV,S,G) -> float (B,KV,n,D),
// reading source rows [lo, lo+n) on the S axis. One thread per OUTPUT elem.
// The packed/scales rows are addressed at (s_out + lo); the output is dense.
template <typename T>
__global__ void kv_int4_unpack_kernel(const uint8_t* __restrict__ packed,
                                      const T* __restrict__ scales,
                                      T* __restrict__ out,
                                      int64_t BKV, int64_t S, int64_t D,
                                      int group, int64_t G, int64_t lo,
                                      int64_t n_out, int64_t total) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;  // output elem
  if (idx >= total) return;
  int64_t d = idx % D;
  int64_t so = (idx / D) % n_out;        // output row on S [0, n_out)
  int64_t bkv = idx / (D * n_out);       // flattened (B*KV) index
  int64_t src_row = bkv * S + (so + lo); // source row in packed/scales
  int64_t byte_idx = src_row * (D / 2) + (d >> 1);
  uint8_t byte = packed[byte_idx];
  int q = (d & 1) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
  int64_t g = d / group;
  float s = ld<T>(scales, src_row * G + g);
  st<T>(out, idx, (float)(q - 8) * s);
}

// Quantize+pack a KV tensor. x: (B,KV,S,D) compute dtype. Returns the packed
// uint8 buffer; scales are computed here and written into `scales_out`
// (caller-allocated (B,KV,S,G) compute dtype) so the ring can store them
// alongside. Symmetric-8, group-32 along D.
NDArray kv_int4_pack(const NDArray& x, NDArray& scales_out, int group) {
  if (x.ndim() != 4)
    throw std::runtime_error("kv_int4_pack: expects (B,KV,S,D)");
  int64_t B = x.shape[0], KV = x.shape[1], S = x.shape[2], D = x.shape[3];
  if (D % group != 0 || D % 2 != 0)
    throw std::runtime_error("kv_int4_pack: D must be even and divisible by group");
  int64_t G = D / group;
  int64_t rows = B * KV * S;
  // scales = max|x| over each group of `group` elems on D. Reshape to
  // (rows, G, group), abs, max over last -> (rows, G), * (1/8). Done with
  // engine ops so no bespoke reduction kernel (and it stays autograd-free
  // via the no_grad caller).
  NDArray xg = x.reshape({rows, G, (int64_t)group});
  NDArray amax = reduce_max(ew_unary(xg, U_ABS), {2}, /*keepdim=*/false); // (rows,G)
  NDArray scales = ew_scalar(amax, 1.0 / 8.0, /*op=*/2, false);  // * 1/8
  scales = ew_scalar(scales, 1e-8, /*op=*/0, false);            // + eps
  scales = scales.astype(x.dtype);
  // hand the scales back in the (B,KV,S,G) view the ring stores
  scales_out = scales.reshape({B, KV, S, G});
  NDArray packed({B, KV, S, D / 2}, DType::Uint8, x.device);
  int64_t nbytes = rows * (D / 2);
  if (nbytes) {
    DISPATCH_FLOAT(x.dtype, T, {
      kv_int4_pack_kernel<T><<<nblk(nbytes), kT>>>(
          static_cast<T*>(x.data_ptr()),
          static_cast<T*>(scales_out.data_ptr()),
          static_cast<uint8_t*>(packed.data_ptr()),
          D, group, G, nbytes);
    });
    cuda_check_last("kv_int4_pack");
  }
  return packed;
}

// Dequantize a slice [lo:lo+n) of a packed KV buffer -> (B,KV,n,D) compute
// dtype. packed: (B,KV,S,D/2) uint8; scales: (B,KV,S,G) compute dtype.
NDArray kv_int4_unpack(const NDArray& packed, const NDArray& scales,
                       int group, int64_t lo, int64_t n, DType out_dtype) {
  if (packed.dtype != DType::Uint8)
    throw std::runtime_error("kv_int4_unpack: packed must be uint8");
  int64_t B = packed.shape[0], KV = packed.shape[1], S = packed.shape[2];
  int64_t D = packed.shape[3] * 2;
  if (lo < 0 || n < 0 || lo + n > S)
    throw std::runtime_error("kv_int4_unpack: slice out of range");
  int64_t G = D / group;
  NDArray out({B, KV, n, D}, out_dtype, packed.device);
  int64_t total = B * KV * n * D;
  if (total) {
    DISPATCH_FLOAT(out_dtype, T, {
      kv_int4_unpack_kernel<T><<<nblk(total), kT>>>(
          static_cast<uint8_t*>(packed.data_ptr()),
          static_cast<T*>(scales.data_ptr()),
          static_cast<T*>(out.data_ptr()),
          B * KV, S, D, group, G, lo, n, total);
    });
    cuda_check_last("kv_int4_unpack");
  }
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
  if (bytes) param.mark_modified();
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
  param.mark_modified();
  buf.mark_modified();
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
  param.mark_modified();
  m.mark_modified();
  v.mark_modified();
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

// --------------------------------------------------------- last-axis argmax
// Phase 1.1 (KERNEL_OPT_IMPLEMENTATION_PLAN.md): decode-loop hygiene. The
// generic `arg_kernel` above is one-thread-per-output-row with a serial scan
// over the reduced axis — correct for the general axis-reduce case, but for
// the decode hot shape (outer=1 row, N~152k vocab) that puts the ENTIRE
// reduction on a single thread. This kernel is block-per-row instead: each
// row is grid-stride-loaded across blockDim.x threads, reduced with
// warp-shuffle (intra-warp) then a small shared-mem stage (inter-warp), so a
// single-row call still lights up all of kT's warps. Two-stage reduction:
//   1. each thread scans its strided slice of the row into a running
//      (best_val, best_idx) with tie-break "lowest index wins" (matches
//      numpy.argmax, which returns the FIRST occurrence of the max);
//   2. warp-shuffle butterfly reduces the 32 per-thread pairs to one per
//      warp, then warp 0 reduces the (kT/32) per-warp pairs from shared mem.
// Ties are broken consistently at every stage by preferring the lower index
// on equality, so the reduction tree order never changes the result.
namespace {
__device__ __forceinline__ void argmax_pair_reduce(float& val, int64_t& idx,
                                                    float ov, int64_t oi) {
  // keep `val,idx` if it already wins; otherwise take `ov,oi`. Lower index
  // wins ties so the combine is associative/commutative under that rule.
  if (ov > val || (ov == val && oi < idx)) { val = ov; idx = oi; }
}
template <typename T>
__global__ void argmax_last_axis_kernel(const T* a, int64_t* out, int64_t N) {
  int64_t row = blockIdx.x;
  const T* arow = a + row * N;

  float best_val = -3.4e38f;
  int64_t best_idx = 0;
  for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
    float v = ld<T>(arow, j);
    argmax_pair_reduce(best_val, best_idx, v, j);
  }

  // intra-warp butterfly (32 lanes -> 1)
  const unsigned FULL = 0xffffffffu;
  for (int off = 16; off > 0; off >>= 1) {
    float ov = __shfl_down_sync(FULL, best_val, off);
    int64_t oi = __shfl_down_sync(FULL, best_idx, off);
    argmax_pair_reduce(best_val, best_idx, ov, oi);
  }

  // inter-warp: lane 0 of each warp stages to shared mem, warp 0 finishes.
  __shared__ float sval[32];
  __shared__ int64_t sidx[32];
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int nwarp = (blockDim.x + 31) >> 5;
  if (lane == 0) { sval[warp] = best_val; sidx[warp] = best_idx; }
  __syncthreads();
  if (warp == 0) {
    best_val = (lane < nwarp) ? sval[lane] : -3.4e38f;
    best_idx = (lane < nwarp) ? sidx[lane] : 0;
    for (int off = 16; off > 0; off >>= 1) {
      float ov = __shfl_down_sync(FULL, best_val, off);
      int64_t oi = __shfl_down_sync(FULL, best_idx, off);
      if (lane + off < nwarp) argmax_pair_reduce(best_val, best_idx, ov, oi);
    }
    if (lane == 0) out[row] = best_idx;
  }
}
}  // namespace

NDArray argmax_last_axis(const NDArray& a) {
  int nd = a.ndim();
  if (nd < 1) throw std::runtime_error("argmax_last_axis needs >=1D input");
  int64_t N = a.shape[nd - 1];
  if (N <= 0) throw std::runtime_error("argmax_last_axis: empty last axis");
  int64_t outer = a.numel() / N;
  Shape os; for (int d = 0; d < nd - 1; ++d) os.push_back(a.shape[d]);
  if (os.empty()) os.push_back(1);
  NDArray out(os, DType::Int64, a.device);
  if (outer > 0) {
    // block-per-row; kT (256) threads/row grid-stride the vocab-sized axis.
    DISPATCH_FLOAT(a.dtype, T, {
      argmax_last_axis_kernel<T><<<(unsigned)outer, kT>>>(
          static_cast<T*>(a.data_ptr()), static_cast<int64_t*>(out.data_ptr()), N);
    });
    cuda_check_last("argmax_last_axis");
  }
  return out;
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

// ------------------------------------------------ fused non-causal SDPA
// One CUDA block owns one (batch, head, query) row. Four warps stream
// disjoint key subsequences and keep independent online-softmax states; the
// block then merges those states with the usual exp(m_i - m_global) rule.
// Lanes cooperate on the D-wide dot product and each lane owns a disjoint
// slice of the value accumulator, so the only shared state is q plus four
// partial output vectors. No Lq x Lk score or weight tensor is allocated.
namespace {
template <typename T, int DMAX>
__global__ void fused_sdpa_noncausal_kernel(
    const T* q, const T* k, const T* v, T* out,
    int Lq, int Lk, int D, float scale) {
  const int row = blockIdx.x;  // flattened (B,H,Lq) query row
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int nwarp = blockDim.x >> 5;  // fixed at four for the launcher
  constexpr unsigned FULL = 0xffffffffu;

  const T* qrow = q + (int64_t)row * D;
  const int64_t bh = row / Lq;
  const T* kbase = k + bh * (int64_t)Lk * D;
  const T* vbase = v + bh * (int64_t)Lk * D;

  __shared__ float qsh[DMAX];
  for (int d = tid; d < D; d += blockDim.x) qsh[d] = ld<T>(qrow, d);
  __syncthreads();

  // Each warp walks every nwarp-th key and maintains its own online-softmax
  // numerator/denominator. m and l are intentionally replicated across a
  // warp; each lane owns only values d=lane, lane+32, ... .
  float m = -3.0e38f;
  float l = 0.f;
  constexpr int ACCN = (DMAX + 31) / 32;
  float acc[ACCN];
  #pragma unroll
  for (int dl = 0; dl < ACCN; ++dl) acc[dl] = 0.f;

  for (int j = warp; j < Lk; j += nwarp) {
    const T* kj = kbase + (int64_t)j * D;
    float dot = 0.f;
    for (int d = lane; d < D; d += 32) dot += qsh[d] * ld<T>(kj, d);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      dot += __shfl_down_sync(FULL, dot, off);
    const float score = __shfl_sync(FULL, dot, 0) * scale;

    const float m_new = fmaxf(m, score);
    const float corr = expf(m - m_new);
    const float w = expf(score - m_new);
    l = l * corr + w;
    const T* vj = vbase + (int64_t)j * D;
    #pragma unroll
    for (int dl = 0; dl < ACCN; ++dl) {
      const int d = lane + dl * 32;
      if (d < D) acc[dl] = acc[dl] * corr + w * ld<T>(vj, d);
    }
    m = m_new;
  }

  // Merge the four warp-local online-softmax partials. Only lane zero owns
  // each warp's m/l state; output accumulators are already lane-disjoint.
  __shared__ float red[128];
  red[tid] = lane == 0 ? m : -3.0e38f;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (tid < off) red[tid] = fmaxf(red[tid], red[tid + off]);
    __syncthreads();
  }
  const float gmax = red[0];
  // red[] is immediately reused for the denominator reduction.  Every warp
  // must finish loading the max before thread 0 can overwrite red[0]; without
  // this barrier, later warps can use warp 0's scaled denominator as gmax.
  __syncthreads();
  const float rescale = expf(m - gmax);

  red[tid] = lane == 0 ? l * rescale : 0.f;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (tid < off) red[tid] += red[tid + off];
    __syncthreads();
  }
  const float inv_denom = red[0] > 0.f ? 1.f / red[0] : 0.f;

  __shared__ float partial[4][DMAX];
  #pragma unroll
  for (int dl = 0; dl < ACCN; ++dl) {
    const int d = lane + dl * 32;
    if (d < D) partial[warp][d] = acc[dl] * rescale;
  }
  __syncthreads();

  T* outrow = out + (int64_t)row * D;
  for (int d = tid; d < D; d += blockDim.x) {
    float total = 0.f;
    #pragma unroll
    for (int w = 0; w < 4; ++w) total += partial[w][d];
    st<T>(outrow, d, total * inv_denom);
  }
}

// EXP-APA-3 Q-tile dispatch hook; defined after the qtile kernels below
// (same anonymous namespace). Returns true when the Q-tiled path handled
// the call (fp16, D <= 64, TC_ATTN_QTILE != 0).
bool qtile_launch_sdpa(const NDArray& q, const NDArray& k, const NDArray& v,
                       NDArray& out, int Lq, int Lk, int D, float scale);
}  // namespace

NDArray fused_sdpa_noncausal(const NDArray& q, const NDArray& k,
                             const NDArray& v, float scale) {
  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4)
    throw std::runtime_error("fused_sdpa_noncausal expects q/k/v rank 4");
  if (q.dtype != k.dtype || q.dtype != v.dtype)
    throw std::runtime_error("fused_sdpa_noncausal dtype mismatch");
  if (q.device.type != k.device.type || q.device.index != k.device.index ||
      q.device.type != v.device.type || q.device.index != v.device.index)
    throw std::runtime_error("fused_sdpa_noncausal device mismatch");
  if (q.dtype != DType::Float16 && q.dtype != DType::Float32)
    throw std::runtime_error("fused_sdpa_noncausal supports float16/float32 only");

  const int64_t B = q.shape[0], H = q.shape[1];
  const int64_t Lq64 = q.shape[2], D64 = q.shape[3], Lk64 = k.shape[2];
  if (k.shape[0] != B || v.shape[0] != B || k.shape[1] != H || v.shape[1] != H ||
      k.shape[3] != D64 || v.shape[2] != Lk64 || v.shape[3] != D64)
    throw std::runtime_error("fused_sdpa_noncausal shape mismatch");
  if (D64 <= 0 || D64 > 128)
    throw std::runtime_error("fused_sdpa_noncausal requires 1 <= head_dim <= 128");
  if (Lq64 < 0 || Lk64 <= 0)
    throw std::runtime_error("fused_sdpa_noncausal requires non-empty key sequence");
  const int64_t rows64 = B * H * Lq64;
  if (rows64 > std::numeric_limits<int>::max() || Lq64 > std::numeric_limits<int>::max() ||
      Lk64 > std::numeric_limits<int>::max())
    throw std::runtime_error("fused_sdpa_noncausal shape exceeds CUDA launch range");

  NDArray out(q.shape, q.dtype, q.device);
  if (rows64 == 0) return out;
  const int rows = (int)rows64;
  const int Lq = (int)Lq64;
  const int Lk = (int)Lk64;
  const int D = (int)D64;
  // EXP-APA-3: Q-tiled K/V-reuse skeleton for the fp16 production shapes;
  // fp32 keeps the streaming skeleton (registered 1.3e-8 fp32 class).
  if (qtile_launch_sdpa(q, k, v, out, Lq, Lk, D, scale)) return out;
  constexpr int threads = 128;
  DISPATCH_FLOAT(q.dtype, T, {
    if (D <= 64) {
      fused_sdpa_noncausal_kernel<T, 64><<<rows, threads>>>(
          static_cast<T*>(q.data_ptr()), static_cast<T*>(k.data_ptr()),
          static_cast<T*>(v.data_ptr()), static_cast<T*>(out.data_ptr()),
          Lq, Lk, D, scale);
    } else {
      fused_sdpa_noncausal_kernel<T, 128><<<rows, threads>>>(
          static_cast<T*>(q.data_ptr()), static_cast<T*>(k.data_ptr()),
          static_cast<T*>(v.data_ptr()), static_cast<T*>(out.data_ptr()),
          Lq, Lk, D, scale);
    }
  });
  cuda_check_last("fused_sdpa_noncausal");
  return out;
}

// ------------------------------------------- fused non-causal INT4-bulk APA
// EXP-APA-2 (HY3D DiT). Fuses the app-level "composed" APA chain
// (ColdCast hy3d_tc/apa.py, EXP-APA-1) into two kernels:
//
//   1. apa_int4_pack_kernel — per-key symmetric INT4 quantization of K,
//      ONE group per D<=128 key vector (EXP-APA-1's convention: D=64 < the
//      engine affine group_size 128 -> a single group). The grid mirrors the
//      composed instrument BIT FOR BIT: amax over |k| in fp32, scale =
//      amax * (float)(1.0/7.0), zero-guarded to 1.0, codes =
//      clamp(roundf(x * (1.f/scale)), -7, 7)  — roundf (U_ROUND) and
//      reciprocal-MULTIPLY, exactly like the app's
//      (kf * safe_scale.reciprocal()).round().clamp(-qmax, qmax).
//      NOTE: this is intentionally NOT kv_int4_pack's symmetric-8 grid
//      (scale=amax/8, lrintf, divide, +8 offset) — EXP-APA-1 registered the
//      symmetric-7 grid and its semantics are settled; do not re-litigate.
//      Codes are stored biased (+7 -> [0,14]) two per byte (even element ->
//      low nibble), per-key scale in fp32.
//
//   2. apa_int4_sdpa_noncausal_kernel — one block per (b,h,query) row, four
//      warps, warp-cooperative key streaming (the apa_selective_kernel A5
//      pattern) with the fused_sdpa_noncausal online-softmax merge:
//        pass 1: bulk scores from the PACKED INT4 K (in-register dequant,
//                dq rounded to T so values equal the composed fp16 kq
//                tensor), accumulating sum/sumsq of |bulk| for the engine's
//                registered Gaussian-quantile threshold
//                thr = mean + z * std (APA paper 2.1 step 2, population
//                variance / ddof=0 like ops::var).
//        pass 2: re-walk keys (recompute-not-store, the apa_selective
//                choice: ~2.3 KB shared/block instead of a 17.8 KB bulk-
//                score cache, keeping occupancy), refine keys with
//                |bulk| >= thr by an exact full-precision K dot
//                (mix_scores_kernel semantics: out = refine ? exact : bulk,
//                nothing dropped), stream the blended score through the
//                online softmax and accumulate V (full precision) —
//                denominator exact over ALL keys.
//      No Lq x Lk score/weight/mask tensor ever exists in global memory.
//
// fp16 parity with the composed path (the K-EQ gate class): when T is
// __half and r < 1, bulk and refined scores are rounded to fp16 before the
// threshold test / blend / softmax, mirroring the fp16 GEMM-output tensors
// the composed chain feeds to where()/softmax(). With refine_all (r >= 1)
// there is no bulk pass and scores stay fp32 — the kernel then IS the
// fused_sdpa_noncausal math (the K-SDPA gate class, 3643efe: fp32 <=1.3e-8,
// fp16 <=3.8e-6 vs blocked fp32 reference at DiT shapes). expf (not __expf)
// keeps softmax parity with U_EXP.
//
// THE THRESHOLD IS AN fp16 LADDER, OVERFLOW INCLUDED (EXP-APA-2 finding).
// The composed chain computes thr = mean + z*std through ENGINE OPS on fp16
// tensors: reduce_sum stores the RAW ROW SUM as fp16 BEFORE mul_scalar's
// 1/cnt — at the DiT single-stream sites sum(|bulk|) exceeds 65504 for many
// rows (measured: 207/142144 rows at the first single block, 108129/142144
// at the last), the fp16 sum saturates to +inf, thr becomes +inf and those
// rows refine NOTHING (pure bulk INT4 attention). A "better" fp32 threshold
// is NOT parity: a 1-ulp thr16 shift alone moves the output rel-fro by up
// to 1.6e-1 at the last single block (scores ~361 +/- 0.6 live on a 0.25-
// step fp16 lattice, so the mask is lattice-tie dominated), and the naive
// fp32 sumsq - mean^2 form is catastrophically cancelled there (4862 rows
// with negative variance; realized refine 0.22 vs composed 0.16). So for
// __half inputs pass 1 replicates the composed value ladder exactly:
//   sum16   = f2h(fp32-tree-sum of |bulk16|)         (reduce_sum output)
//   mean16  = f2h(h2f(sum16) * (float)(1.0/cnt))     (mul_scalar)
//   d16_j   = f2h(|bulk16_j| - h2f(mean16))          (sub)
//   dd16_j  = f2h(h2f(d16_j)^2)                      (mul)
//   var16   = f2h(h2f(f2h(fp32-tree-sum dd)) * inv)  (reduce_sum+mul_scalar)
//   std16   = f2h(sqrtf(h2f(var16)))                 (sqrt)
//   sz16    = f2h(h2f(std16) * zthr)                 (mul_scalar z)
//   thr     = h2f(f2h(h2f(mean16) + h2f(sz16)))      (add)
// The only remaining nondeterminism vs the engine is fp32 SUM ORDER (tree
// here vs sequential-per-row there), which flips an f2h rounding with
// probability ~3e-4/row and costs ~1e-4 rel-fro (measured estimate) — the
// gate's reduction-order slack. float32 inputs keep the pure fp32 two-sweep
// (composed fp32 has no rounding ladder and fp32 sums cannot overflow here).
// Bulk scores are cached in dynamic shared memory ((Lk)*sizeof(T), capped in
// the launcher), so keys are packed-INT4-read ONCE and pass 2 re-reads only
// the cache plus full-precision K rows for refined keys.
namespace {

constexpr float kApaInt4InvQmax = (float)(1.0 / 7.0);  // matches ew_scalar's (float) cast

template <typename T>
__global__ void apa_int4_pack_kernel(const T* __restrict__ k,
                                     uint8_t* __restrict__ codes,
                                     float* __restrict__ kscale,
                                     int64_t nkeys, int D) {
  // One warp per key vector; lanes stride the packed bytes (element pairs).
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t key = (int64_t)blockIdx.x * (blockDim.x >> 5) + warp;
  constexpr unsigned FULL = 0xffffffffu;
  if (key >= nkeys) return;
  const T* krow = k + key * D;

  float amax = 0.f;
  for (int d = lane; d < D; d += 32) amax = fmaxf(amax, fabsf(ld<T>(krow, d)));
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    amax = fmaxf(amax, __shfl_down_sync(FULL, amax, off));
  amax = __shfl_sync(FULL, amax, 0);

  const float scale = amax * kApaInt4InvQmax;
  const float safe = scale > 0.f ? scale : 1.f;   // composed: where(scale>0, scale, 1)
  const float recip = 1.f / safe;                 // composed: safe_scale.reciprocal()
  const int half_d = (D + 1) >> 1;
  for (int b = lane; b < half_d; b += 32) {
    const float x0 = ld<T>(krow, 2 * b);
    const float x1 = (2 * b + 1 < D) ? ld<T>(krow, 2 * b + 1) : 0.f;
    float c0 = roundf(x0 * recip);                // U_ROUND == roundf
    float c1 = roundf(x1 * recip);
    c0 = c0 < -7.f ? -7.f : (c0 > 7.f ? 7.f : c0);
    c1 = c1 < -7.f ? -7.f : (c1 > 7.f ? 7.f : c1);
    const int q0 = (int)c0 + 7;                   // biased nibble [0,14]
    const int q1 = (int)c1 + 7;
    codes[key * half_d + b] = (uint8_t)((q1 << 4) | q0);
  }
  if (lane == 0) kscale[key] = safe;
}

// In-register dequant of one packed byte -> two bulk-K values. For __half
// inputs the composed instrument materializes dq K as an fp16 tensor
// (dq.astype(input_dtype)); rounding through __float2half reproduces those
// exact values so the bulk dot walks the same numbers.
template <typename T>
__device__ __forceinline__ void apa_int4_dequant2(uint8_t byte, float safe,
                                                  float& dq0, float& dq1) {
  const float c0 = (float)((int)(byte & 0xF) - 7);
  const float c1 = (float)((int)(byte >> 4) - 7);
  dq0 = c0 * safe;
  dq1 = c1 * safe;
  if constexpr (std::is_same<T, __half>::value) {
    dq0 = __half2float(__float2half(dq0));
    dq1 = __half2float(__float2half(dq1));
  }
}

// F-A1 causal/selective INT4 helpers.  Unlike the EXP-APA-2 fp16 composed
// path, selective attention keeps its established fp32 score/statistics
// semantics: packed codes are dequantized in registers and accumulated in
// fp32 for every input dtype.  This deliberately does not use EXP-APA-2's
// fp16 materialization-rounding special case in apa_int4_dequant2.
template <typename T>
__device__ __forceinline__ float apa_selective_int4_dot_warp(
    const float* __restrict__ qsh, const uint8_t* __restrict__ cj,
    float safe, int D) {
  constexpr unsigned FULL = 0xffffffffu;
  const int lane = threadIdx.x & 31;
  const int packed_d = (D + 1) >> 1;
  float dot = 0.f;
  for (int b = lane; b < packed_d; b += 32) {
    const uint8_t byte = cj[b];
    const float dq0 = (float)((int)(byte & 0xF) - 7) * safe;
    const float dq1 = (float)((int)(byte >> 4) - 7) * safe;
    dot += qsh[2 * b] * dq0;
    if (2 * b + 1 < D) dot += qsh[2 * b + 1] * dq1;
  }
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    dot += __shfl_down_sync(FULL, dot, off);
  return __shfl_sync(FULL, dot, 0);
}

template <typename T>
__device__ __forceinline__ float apa_selective_int4_dot_serial(
    const float* __restrict__ qsh, const uint8_t* __restrict__ cj,
    float safe, int D) {
  const int packed_d = (D + 1) >> 1;
  float dot = 0.f;
  for (int b = 0; b < packed_d; ++b) {
    const uint8_t byte = cj[b];
    const float dq0 = (float)((int)(byte & 0xF) - 7) * safe;
    const float dq1 = (float)((int)(byte >> 4) - 7) * safe;
    dot += qsh[2 * b] * dq0;
    if (2 * b + 1 < D) dot += qsh[2 * b + 1] * dq1;
  }
  return dot;
}

template <typename T>
__device__ __forceinline__ float apa_selective_exact_dot_warp(
    const float* __restrict__ qsh, const T* __restrict__ kj, int D) {
  constexpr unsigned FULL = 0xffffffffu;
  const int lane = threadIdx.x & 31;
  float dot = 0.f;
  for (int d = lane; d < D; d += 32) dot += qsh[d] * ld<T>(kj, d);
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    dot += __shfl_down_sync(FULL, dot, off);
  return __shfl_sync(FULL, dot, 0);
}

template <typename T>
__device__ __forceinline__ float apa_selective_exact_dot_serial(
    const float* __restrict__ qsh, const T* __restrict__ kj, int D) {
  float dot = 0.f;
  for (int d = 0; d < D; ++d) dot += qsh[d] * ld<T>(kj, d);
  return dot;
}

// One-block-per-query F-A1 path.  This is the selective family's existing
// WCOOP/per-thread algorithm with kq reads replaced by transient packed-code
// reads.  Pass 1 computes population mean/variance of |bulk*scale|; pass 2
// keeps every key, replacing only selected bulk scores with exact-K scores.
template <typename T, int DMAX, bool WCOOP>
__global__ void apa_selective_int4_kernel(
    const T* q, const T* k, const uint8_t* codes, const float* kscale,
    const T* v, T* out, int B, int H, int L, int S, int D, int VD,
    float scale, float zthr, int is_causal, int KVH, int group,
    int packed_d) {
  const int row = blockIdx.x;
  const int i = row % L;
  const int bh = row / L;
  const int b = bh / H;
  const int h = bh % H;
  const int kv_h = h / group;
  const int tid = threadIdx.x;
  const int nt = blockDim.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int nwarp = nt >> 5;
  constexpr unsigned FULL = 0xffffffffu;

  const T* qrow = q + (int64_t)row * D;
  const int64_t kvbh = (int64_t)b * KVH + kv_h;
  const T* kbase = k + kvbh * S * D;
  const uint8_t* cbase = codes + kvbh * S * packed_d;
  const float* sbase = kscale + kvbh * S;
  const T* vbase = v + kvbh * S * VD;
  const int s_max = is_causal ? ((S - L) + i + 1) : S;

  __shared__ float qsh[DMAX];
  __shared__ float red[256];
  for (int d = tid; d < D; d += nt) qsh[d] = ld<T>(qrow, d);
  __syncthreads();

  float sum = 0.f, sumsq = 0.f;
  if constexpr (WCOOP) {
    for (int j = warp; j < s_max; j += nwarp) {
      const float dot = apa_selective_int4_dot_warp<T>(
          qsh, cbase + (int64_t)j * packed_d, sbase[j], D);
      const float a = fabsf(dot * scale);
      sum += a;
      sumsq += a * a;
    }
  } else {
    for (int j = tid; j < s_max; j += nt) {
      const float dot = apa_selective_int4_dot_serial<T>(
          qsh, cbase + (int64_t)j * packed_d, sbase[j], D);
      const float a = fabsf(dot * scale);
      sum += a;
      sumsq += a * a;
    }
  }
  red[tid] = (WCOOP && lane != 0) ? 0.f : sum;
  __syncthreads();
  for (int off = nt / 2; off > 0; off >>= 1) {
    if (tid < off) red[tid] += red[tid + off];
    __syncthreads();
  }
  const float total = red[0];
  __syncthreads();
  red[tid] = (WCOOP && lane != 0) ? 0.f : sumsq;
  __syncthreads();
  for (int off = nt / 2; off > 0; off >>= 1) {
    if (tid < off) red[tid] += red[tid + off];
    __syncthreads();
  }
  const float total_sq = red[0];
  __syncthreads();
  const float mean = total / (float)s_max;
  const float var = total_sq / (float)s_max - mean * mean;
  const float thr = mean + zthr * sqrtf(fmaxf(var, 0.f));

  float m = -1e30f, l = 0.f;
  constexpr int ACCN = WCOOP ? ((DMAX + 31) / 32) : DMAX;
  float acc[ACCN];
  if constexpr (WCOOP) {
    #pragma unroll
    for (int dl = 0; dl < ACCN; ++dl) acc[dl] = 0.f;
  } else {
    for (int d = 0; d < VD; ++d) acc[d] = 0.f;
  }

  if constexpr (WCOOP) {
    for (int j = warp; j < s_max; j += nwarp) {
      float bulk = apa_selective_int4_dot_warp<T>(
          qsh, cbase + (int64_t)j * packed_d, sbase[j], D) * scale;
      float score = bulk;
      if (fabsf(bulk) >= thr)
        score = apa_selective_exact_dot_warp<T>(
            qsh, kbase + (int64_t)j * D, D) * scale;
      const float m_new = fmaxf(m, score);
      const float corr = __expf(m - m_new);
      const float w = __expf(score - m_new);
      l = l * corr + w;
      const T* vj = vbase + (int64_t)j * VD;
      #pragma unroll
      for (int dl = 0; dl < ACCN; ++dl) {
        const int d = lane + dl * 32;
        if (d < VD) acc[dl] = acc[dl] * corr + w * ld<T>(vj, d);
      }
      m = m_new;
    }
  } else {
    for (int j = tid; j < s_max; j += nt) {
      float bulk = apa_selective_int4_dot_serial<T>(
          qsh, cbase + (int64_t)j * packed_d, sbase[j], D) * scale;
      float score = bulk;
      if (fabsf(bulk) >= thr)
        score = apa_selective_exact_dot_serial<T>(
            qsh, kbase + (int64_t)j * D, D) * scale;
      const float m_new = fmaxf(m, score);
      const float corr = __expf(m - m_new);
      const float w = __expf(score - m_new);
      l = l * corr + w;
      const T* vj = vbase + (int64_t)j * VD;
      for (int d = 0; d < VD; ++d)
        acc[d] = acc[d] * corr + w * ld<T>(vj, d);
      m = m_new;
    }
  }

  red[tid] = m;
  __syncthreads();
  for (int off = nt / 2; off > 0; off >>= 1) {
    if (tid < off) red[tid] = fmaxf(red[tid], red[tid + off]);
    __syncthreads();
  }
  const float gmax = red[0];
  __syncthreads();
  const float rescale = __expf(m - gmax);
  red[tid] = (WCOOP && lane != 0) ? 0.f : l * rescale;
  __syncthreads();
  for (int off = nt / 2; off > 0; off >>= 1) {
    if (tid < off) red[tid] += red[tid + off];
    __syncthreads();
  }
  const float denom = red[0];
  __syncthreads();

  __shared__ float wpart[4][DMAX];
  if constexpr (WCOOP) {
    #pragma unroll
    for (int dl = 0; dl < ACCN; ++dl) {
      const int d = lane + dl * 32;
      if (d < VD) wpart[warp][d] = acc[dl] * rescale;
    }
  } else {
    for (int d = 0; d < VD; ++d) {
      float val = acc[d] * rescale;
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(FULL, val, off);
      if (lane == 0) wpart[warp][d] = val;
    }
  }
  __syncthreads();
  T* orow = out + (int64_t)row * VD;
  const float inv = denom > 0.f ? 1.f / denom : 0.f;
  for (int d = tid; d < VD; d += nt) {
    float value = 0.f;
    for (int w = 0; w < nwarp; ++w) value += wpart[w][d];
    st<T>(orow, d, value * inv);
  }
}

// Split-K stage 1: same full-range population statistics as the monolithic
// kernel.  The packed workspace is shared with all split partitions.
template <typename T, int DMAX, bool WCOOP>
__global__ void apa_selective_int4_stats_kernel(
    const T* q, const uint8_t* codes, const float* kscale, float* thr_out,
    int B, int H, int L, int S, int D, float scale, float zthr,
    int is_causal, int KVH, int group, int packed_d) {
  const int row = blockIdx.x;
  const int i = row % L;
  const int bh = row / L;
  const int b = bh / H;
  const int h = bh % H;
  const int kv_h = h / group;
  const int tid = threadIdx.x;
  const int nt = blockDim.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int nwarp = nt >> 5;
  const T* qrow = q + (int64_t)row * D;
  const int64_t kvbh = (int64_t)b * KVH + kv_h;
  const uint8_t* cbase = codes + kvbh * S * packed_d;
  const float* sbase = kscale + kvbh * S;
  const int s_max = is_causal ? ((S - L) + i + 1) : S;
  __shared__ float qsh[DMAX];
  __shared__ float red[256];
  for (int d = tid; d < D; d += nt) qsh[d] = ld<T>(qrow, d);
  __syncthreads();

  float sum = 0.f, sumsq = 0.f;
  if constexpr (WCOOP) {
    for (int j = warp; j < s_max; j += nwarp) {
      const float dot = apa_selective_int4_dot_warp<T>(
          qsh, cbase + (int64_t)j * packed_d, sbase[j], D);
      const float a = fabsf(dot * scale);
      sum += a;
      sumsq += a * a;
    }
  } else {
    for (int j = tid; j < s_max; j += nt) {
      const float dot = apa_selective_int4_dot_serial<T>(
          qsh, cbase + (int64_t)j * packed_d, sbase[j], D);
      const float a = fabsf(dot * scale);
      sum += a;
      sumsq += a * a;
    }
  }
  red[tid] = (WCOOP && lane != 0) ? 0.f : sum;
  __syncthreads();
  for (int off = nt / 2; off > 0; off >>= 1) {
    if (tid < off) red[tid] += red[tid + off];
    __syncthreads();
  }
  const float total = red[0];
  __syncthreads();
  red[tid] = (WCOOP && lane != 0) ? 0.f : sumsq;
  __syncthreads();
  for (int off = nt / 2; off > 0; off >>= 1) {
    if (tid < off) red[tid] += red[tid + off];
    __syncthreads();
  }
  if (tid == 0) {
    const float mean = total / (float)s_max;
    const float var = red[0] / (float)s_max - mean * mean;
    thr_out[row] = mean + zthr * sqrtf(fmaxf(var, 0.f));
  }
}

// Split-K stage 2: one online-softmax partial per key partition.  The common
// apa_selective_merge_kernel performs stage 3, so its storage and merge order
// remain identical to the established bf16-kq decode family.
template <typename T, int DMAX, bool WCOOP>
__global__ void apa_selective_int4_split_kernel(
    const T* q, const T* k, const uint8_t* codes, const float* kscale,
    const T* v, const float* thr_in, float* part_m, float* part_l,
    float* part_acc, int B, int H, int L, int S, int D, int VD, float scale,
    int is_causal, int KVH, int group, int packed_d, int num_parts,
    int part_keys) {
  const int row = blockIdx.x;
  const int part = blockIdx.y;
  const int i = row % L;
  const int bh = row / L;
  const int b = bh / H;
  const int h = bh % H;
  const int kv_h = h / group;
  const int tid = threadIdx.x;
  const int nt = blockDim.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int nwarp = nt >> 5;
  constexpr unsigned FULL = 0xffffffffu;
  const int s_max = is_causal ? ((S - L) + i + 1) : S;
  const int j0 = part * part_keys;
  const int j1 = min(s_max, j0 + part_keys);
  const int64_t part_row = (int64_t)row * num_parts + part;
  float* pacc = part_acc + part_row * VD;
  if (j0 >= j1) {
    if (tid == 0) {
      part_m[part_row] = -1e30f;
      part_l[part_row] = 0.f;
    }
    for (int d = tid; d < VD; d += nt) pacc[d] = 0.f;
    return;
  }

  const T* qrow = q + (int64_t)row * D;
  const int64_t kvbh = (int64_t)b * KVH + kv_h;
  const T* kbase = k + kvbh * S * D;
  const uint8_t* cbase = codes + kvbh * S * packed_d;
  const float* sbase = kscale + kvbh * S;
  const T* vbase = v + kvbh * S * VD;
  const float thr = thr_in[row];
  __shared__ float qsh[DMAX];
  for (int d = tid; d < D; d += nt) qsh[d] = ld<T>(qrow, d);
  __syncthreads();

  float m = -1e30f, l = 0.f;
  constexpr int ACCN = WCOOP ? ((DMAX + 31) / 32) : DMAX;
  float acc[ACCN];
  if constexpr (WCOOP) {
    #pragma unroll
    for (int dl = 0; dl < ACCN; ++dl) acc[dl] = 0.f;
  } else {
    for (int d = 0; d < VD; ++d) acc[d] = 0.f;
  }

  if constexpr (WCOOP) {
    for (int j = j0 + warp; j < j1; j += nwarp) {
      float bulk = apa_selective_int4_dot_warp<T>(
          qsh, cbase + (int64_t)j * packed_d, sbase[j], D) * scale;
      float score = bulk;
      if (fabsf(bulk) >= thr)
        score = apa_selective_exact_dot_warp<T>(
            qsh, kbase + (int64_t)j * D, D) * scale;
      const float m_new = fmaxf(m, score);
      const float corr = __expf(m - m_new);
      const float w = __expf(score - m_new);
      l = l * corr + w;
      const T* vj = vbase + (int64_t)j * VD;
      #pragma unroll
      for (int dl = 0; dl < ACCN; ++dl) {
        const int d = lane + dl * 32;
        if (d < VD) acc[dl] = acc[dl] * corr + w * ld<T>(vj, d);
      }
      m = m_new;
    }
  } else {
    for (int j = j0 + tid; j < j1; j += nt) {
      float bulk = apa_selective_int4_dot_serial<T>(
          qsh, cbase + (int64_t)j * packed_d, sbase[j], D) * scale;
      float score = bulk;
      if (fabsf(bulk) >= thr)
        score = apa_selective_exact_dot_serial<T>(
            qsh, kbase + (int64_t)j * D, D) * scale;
      const float m_new = fmaxf(m, score);
      const float corr = __expf(m - m_new);
      const float w = __expf(score - m_new);
      l = l * corr + w;
      const T* vj = vbase + (int64_t)j * VD;
      for (int d = 0; d < VD; ++d)
        acc[d] = acc[d] * corr + w * ld<T>(vj, d);
      m = m_new;
    }
  }

  __shared__ float red[256];
  red[tid] = m;
  __syncthreads();
  for (int off = nt / 2; off > 0; off >>= 1) {
    if (tid < off) red[tid] = fmaxf(red[tid], red[tid + off]);
    __syncthreads();
  }
  const float pmax = red[0];
  __syncthreads();
  const float rescale = __expf(m - pmax);
  red[tid] = (WCOOP && lane != 0) ? 0.f : l * rescale;
  __syncthreads();
  for (int off = nt / 2; off > 0; off >>= 1) {
    if (tid < off) red[tid] += red[tid + off];
    __syncthreads();
  }
  const float pl = red[0];
  __syncthreads();
  __shared__ float wpart[4][DMAX];
  if constexpr (WCOOP) {
    #pragma unroll
    for (int dl = 0; dl < ACCN; ++dl) {
      const int d = lane + dl * 32;
      if (d < VD) wpart[warp][d] = acc[dl] * rescale;
    }
  } else {
    for (int d = 0; d < VD; ++d) {
      float val = acc[d] * rescale;
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(FULL, val, off);
      if (lane == 0) wpart[warp][d] = val;
    }
  }
  __syncthreads();
  for (int d = tid; d < VD; d += nt) {
    float value = 0.f;
    for (int w = 0; w < nwarp; ++w) value += wpart[w][d];
    pacc[d] = value;
  }
  if (tid == 0) {
    part_m[part_row] = pmax;
    part_l[part_row] = pl;
  }
}

template <typename T>
static NDArray apa_selective_int4_splitk_dispatch(
    const NDArray& q, const NDArray& k, const NDArray& codes,
    const NDArray& kscales, const NDArray& v, float scale, float zthr,
    bool is_causal, int B, int H, int L, int S, int D, int VD, int KVH,
    int group, int cap, int packed_d) {
  const int rows = B * H * L;
  const int part_keys = TC_APA_SPLITK_PART_KEYS;
  int num_parts = (S + part_keys - 1) / part_keys;
  if (num_parts < 1) num_parts = 1;
  NDArray out({(int64_t)B, (int64_t)H, (int64_t)L, (int64_t)VD},
              q.dtype, q.device);
  NDArray thr_ws({(int64_t)rows}, DType::Float32, q.device);
  NDArray part_m({(int64_t)rows * num_parts}, DType::Float32, q.device);
  NDArray part_l({(int64_t)rows * num_parts}, DType::Float32, q.device);
  NDArray part_acc({(int64_t)rows * num_parts * VD}, DType::Float32, q.device);
  constexpr int threads = 128;
  auto launch = [&](auto dmax_tag, auto wcoop_tag) {
    constexpr int DMAX = decltype(dmax_tag)::value;
    constexpr bool WCOOP = decltype(wcoop_tag)::value;
    apa_selective_int4_stats_kernel<T, DMAX, WCOOP><<<rows, threads>>>(
        static_cast<T*>(q.data_ptr()),
        static_cast<uint8_t*>(codes.data_ptr()),
        static_cast<float*>(kscales.data_ptr()),
        static_cast<float*>(thr_ws.data_ptr()), B, H, L, S, D, scale, zthr,
        is_causal ? 1 : 0, KVH, group, packed_d);
    cuda_check_last("apa_selective_int4_splitk_stats");
    dim3 grid((unsigned)rows, (unsigned)num_parts);
    apa_selective_int4_split_kernel<T, DMAX, WCOOP><<<grid, threads>>>(
        static_cast<T*>(q.data_ptr()), static_cast<T*>(k.data_ptr()),
        static_cast<uint8_t*>(codes.data_ptr()),
        static_cast<float*>(kscales.data_ptr()), static_cast<T*>(v.data_ptr()),
        static_cast<float*>(thr_ws.data_ptr()),
        static_cast<float*>(part_m.data_ptr()),
        static_cast<float*>(part_l.data_ptr()),
        static_cast<float*>(part_acc.data_ptr()), B, H, L, S, D, VD, scale,
        is_causal ? 1 : 0, KVH, group, packed_d, num_parts, part_keys);
    cuda_check_last("apa_selective_int4_splitk_split");
    apa_selective_merge_kernel<T, DMAX><<<rows, threads>>>(
        static_cast<float*>(part_m.data_ptr()),
        static_cast<float*>(part_l.data_ptr()),
        static_cast<float*>(part_acc.data_ptr()), nullptr,
        static_cast<T*>(out.data_ptr()), H, VD, num_parts, 0);
    cuda_check_last("apa_selective_int4_splitk_merge");
  };
  if (cap <= 64)
    launch(std::integral_constant<int, 64>{}, std::false_type{});
  else if (cap <= 128)
    launch(std::integral_constant<int, 128>{}, std::true_type{});
  else if (cap <= 256)
    launch(std::integral_constant<int, 256>{}, std::true_type{});
  else
    launch(std::integral_constant<int, 512>{}, std::true_type{});
  return out;
}

template <typename T, int DMAX>
__global__ void apa_int4_sdpa_noncausal_kernel(
    const T* __restrict__ q, const T* __restrict__ k,
    const uint8_t* __restrict__ codes, const float* __restrict__ kscale,
    const T* __restrict__ v, T* __restrict__ out,
    int Lq, int Lk, int D, float scale, float zthr, float inv_cnt,
    int refine_all) {
  const int row = blockIdx.x;  // flattened (B,H,Lq) query row
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int nwarp = blockDim.x >> 5;  // fixed at four for the launcher
  constexpr unsigned FULL = 0xffffffffu;
  // Round scores to fp16 on the fp16 path (composed-path value parity); the
  // all-refine path stays fp32 like fused_sdpa_noncausal.
  constexpr bool kIsHalf = std::is_same<T, __half>::value;

  const T* qrow = q + (int64_t)row * D;
  const int64_t bh = row / Lq;
  const T* kbase = k + bh * (int64_t)Lk * D;
  const int half_d = D >> 1;
  const uint8_t* cbase = codes + bh * (int64_t)Lk * half_d;
  const float* sbase = kscale + bh * (int64_t)Lk;
  const T* vbase = v + bh * (int64_t)Lk * D;

  __shared__ float qsh[DMAX];
  for (int d = tid; d < D; d += blockDim.x) qsh[d] = ld<T>(qrow, d);
  __syncthreads();

  __shared__ float red[128];
  // Per-row bulk-score cache (dynamic; launcher passes Lk * sizeof(T)).
  extern __shared__ unsigned char apa_smem_raw[];
  T* sbulk = reinterpret_cast<T*>(apa_smem_raw);

  // Pass 1: bulk INT4 scores -> shared cache + the composed-parity quantile
  // threshold ladder. Skipped entirely under refine_all (r >= 1).
  float thr = -3.0e38f;
  if (!refine_all) {
    for (int j = warp; j < Lk; j += nwarp) {
      const uint8_t* cj = cbase + (int64_t)j * half_d;
      const float safe = sbase[j];
      float dot = 0.f;
      for (int b = lane; b < half_d; b += 32) {
        float dq0, dq1;
        apa_int4_dequant2<T>(cj[b], safe, dq0, dq1);
        dot += qsh[2 * b] * dq0 + qsh[2 * b + 1] * dq1;
      }
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        dot += __shfl_down_sync(FULL, dot, off);
      if (lane == 0) st<T>(sbulk, j, dot * scale);  // fp16 rounding via st<half>
    }
    __syncthreads();

    // Sweep A: fp32 tree-sum of |bulk| -> engine reduce_sum + mul_scalar
    // ladder (fp16 sum saturation INCLUDED for __half).
    float part = 0.f;
    for (int j = tid; j < Lk; j += blockDim.x) part += fabsf(ld<T>(sbulk, j));
    red[tid] = part;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
      if (tid < off) red[tid] += red[tid + off];
      __syncthreads();
    }
    const float total = red[0];
    __syncthreads();
    float meanv;
    if constexpr (kIsHalf) {
      const __half sum16 = __float2half(total);            // reduce_sum output (may be +inf)
      const __half mean16 = __float2half(__half2float(sum16) * inv_cnt);  // mul_scalar
      meanv = __half2float(mean16);
    } else {
      meanv = total * inv_cnt;
    }

    // Sweep B: fp32 tree-sum of the composed (a - mean)^2 chain.
    float part2 = 0.f;
    for (int j = tid; j < Lk; j += blockDim.x) {
      const float a = fabsf(ld<T>(sbulk, j));
      if constexpr (kIsHalf) {
        const __half d16 = __float2half(a - meanv);        // sub
        const float df = __half2float(d16);
        part2 += __half2float(__float2half(df * df));      // mul
      } else {
        const float d = a - meanv;
        part2 += d * d;
      }
    }
    red[tid] = part2;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
      if (tid < off) red[tid] += red[tid + off];
      __syncthreads();
    }
    const float total_dd = red[0];
    __syncthreads();
    if constexpr (kIsHalf) {
      const __half ddsum16 = __float2half(total_dd);       // reduce_sum output
      const __half var16 = __float2half(__half2float(ddsum16) * inv_cnt);  // mul_scalar
      const __half std16 = __float2half(sqrtf(__half2float(var16)));       // sqrt
      const __half sz16 = __float2half(__half2float(std16) * zthr);        // mul_scalar z
      thr = __half2float(__float2half(meanv + __half2float(sz16)));        // add
    } else {
      const float var = total_dd * inv_cnt;                // population (ddof=0), ops::var
      thr = meanv + sqrtf(var) * zthr;
    }
  }

  // Pass 2: blend + online softmax + V accumulation (fused_sdpa_noncausal
  // streaming state); bulk scores come from the shared cache.
  float m = -3.0e38f;
  float l = 0.f;
  constexpr int ACCN = (DMAX + 31) / 32;
  float acc[ACCN];
  #pragma unroll
  for (int dl = 0; dl < ACCN; ++dl) acc[dl] = 0.f;

  for (int j = warp; j < Lk; j += nwarp) {
    float score = 0.f;
    bool refine = true;
    if (!refine_all) {
      const float bulk = ld<T>(sbulk, j);
      score = bulk;
      refine = fabsf(bulk) >= thr;  // mix_scores: refine = |ranking| >= thr
    }
    if (refine) {
      const T* kj = kbase + (int64_t)j * D;
      float ex = 0.f;
      for (int d = lane; d < D; d += 32) ex += qsh[d] * ld<T>(kj, d);
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        ex += __shfl_down_sync(FULL, ex, off);
      float exact = __shfl_sync(FULL, ex, 0) * scale;
      if (kIsHalf && !refine_all) exact = __half2float(__float2half(exact));
      score = exact;
    }

    const float m_new = fmaxf(m, score);
    const float corr = expf(m - m_new);
    const float w = expf(score - m_new);
    l = l * corr + w;
    const T* vj = vbase + (int64_t)j * D;
    #pragma unroll
    for (int dl = 0; dl < ACCN; ++dl) {
      const int d = lane + dl * 32;
      if (d < D) acc[dl] = acc[dl] * corr + w * ld<T>(vj, d);
    }
    m = m_new;
  }

  // Merge the four warp-local online-softmax partials (fused_sdpa_noncausal
  // merge, including the r2 barrier between the max and denom reductions).
  red[tid] = lane == 0 ? m : -3.0e38f;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (tid < off) red[tid] = fmaxf(red[tid], red[tid + off]);
    __syncthreads();
  }
  const float gmax = red[0];
  __syncthreads();
  const float rescale = expf(m - gmax);

  red[tid] = lane == 0 ? l * rescale : 0.f;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (tid < off) red[tid] += red[tid + off];
    __syncthreads();
  }
  const float inv_denom = red[0] > 0.f ? 1.f / red[0] : 0.f;

  __shared__ float partial[4][DMAX];
  #pragma unroll
  for (int dl = 0; dl < ACCN; ++dl) {
    const int d = lane + dl * 32;
    if (d < D) partial[warp][d] = acc[dl] * rescale;
  }
  __syncthreads();

  T* outrow = out + (int64_t)row * D;
  for (int d = tid; d < D; d += blockDim.x) {
    float total = 0.f;
    #pragma unroll
    for (int w = 0; w < 4; ++w) total += partial[w][d];
    st<T>(outrow, d, total * inv_denom);
  }
}

// --------------------------------------------- EXP-APA-3: Q-tiled skeleton
// The EXP-APA-2 kernels above are ONE-QUERY-ROW-PER-BLOCK: every query row
// re-streams all of K and V from global memory, so at the DiT joint shape
// (2,16,4442,64) the pure streaming skeleton costs 152 ms/call against the
// cuBLAS-composed 37 ms (EXP-APA-2 K-PERF receipt) — a data-reuse deficit,
// not an APA cost (the APA machinery adds only ~45 ms on top).
//
// This Q-tiled rewrite is flash-attention-shaped:
//   * a TILE of kQtTQ=64 query rows is resident per CTA (Q staged once into
//     shared memory, held for the whole call),
//   * K/V (and, for APA, the packed-INT4 bulk K) are staged through shared
//     memory in kQtTK=32-key tiles and REUSED by all 64 resident queries, so
//     K/V global traffic drops by ~64x vs the streaming skeleton,
//   * QK^T (exact and INT4-dequant bulk) runs on tensor cores: WMMA
//     m16n16k16 HMMA, fp16 operands, fp32 accumulators — products of fp16
//     values are exact in fp32, so this is the same value class as the
//     streaming kernel's fp32 shfl-tree dot, differing only in summation
//     ORDER (the gates' allowed reduction-order slack),
//   * softmax weights and the PV accumulation stay fp32 on CUDA cores
//     (fp16 P-fragments through HMMA would put ~2^-11 relative rounding on
//     every softmax weight and break the registered K-SDPA gate class of
//     fp32-weight streaming; PV is ~80 GFLOP at the DiT shape — cheap).
//
// dp4a/IMMA for the INT4 bulk pass was evaluated and REJECTED on semantics,
// not speed: the registered bulk score is full-precision-fp16 Q dotted with
// PER-ELEMENT fp16-ROUNDED dequantized K (f2h(code*scale)); an integer dot
// would require quantizing Q and would abandon the per-element dq rounding —
// both change bulk values on the lattice-tie-dominated threshold mask
// (EXP-APA-2 measured up to 1.6e-1 rel-fro per fp16 ulp of threshold
// motion). The bit-faithful bulk pass is dequant-HMMA by construction.
//
// APA two-pass structure on this skeleton: the fp16 threshold ladder needs
// mean16 BEFORE the per-element (a-mean16)^2 terms, and the blend needs thr
// before any weight, so the key axis is walked three times (sum, dd-sum,
// blend). A 64-query bulk-score cache would be Tq*Lk*2B = 568 KB — far past
// shared memory — so instead of caching, walks 2 and 3 RECOMPUTE the bulk
// scores from the packed codes (32 B/key vs 128 B/key fp16 K) through the
// exact same dequant+MMA+f2h(x*scale) path; the recomputation is
// deterministic, so all three walks see bit-identical bulk16 values, which
// is what the EXP-APA-2 shared-cache achieved. No Lq x Lk tensor and no
// global workspace of any kind is materialized.
//
// fp32 inputs KEEP the streaming skeleton: TF32/HMMA tensor-core paths
// cannot hold the registered fp32 K-SDPA class (1.3e-8), and fp32 CUDA-core
// tiles would be a separate kernel for a dtype the DiT never runs at
// attention (EXP-APA-2 dtype table: attention compute is fp16). Dispatch is
// internal only — same ops, same signatures; TC_ATTN_QTILE=0 forces the
// legacy streaming path (A/B receipts + the EXP-APA-2 reference outputs).
//
// Shared-memory budget per CTA (all static, no dynamic smem):
//   Qs 64x72 fp16 = 9216 B, Ks 32x72 fp16 = 4608 B, Vs 32x72 fp16 = 4608 B,
//   Ss 64x36 fp32 = 9216 B (scores, then reused as the fp32 weight tile),
//   APA only: DQs 32x72 fp16 = 4608 B, Sb 64x40 fp16 = 5120 B.
//   APA total ~36.5 KB (2 CTAs/SM on sm_89), SDPA ~27.6 KB (3 CTAs/SM).
namespace qtile {

constexpr int kTQ = 64;        // query rows resident per CTA
constexpr int kTK = 32;        // K/V tile depth staged through shared memory
constexpr int kD = 64;         // padded head_dim (actual D <= 64 zero-filled)
constexpr int kLdH = kD + 8;   // fp16 tile leading dim (bank skew, mult of 8)
constexpr int kLdS = kTK + 4;  // fp32 score tile leading dim (mult of 4)
constexpr int kLdB = kTK + 8;  // fp16 bulk tile leading dim (mult of 8)
constexpr int kThreads = 128;  // 4 warps x 16 query rows each

// Stage a (rows_valid x D) fp16 global tile into zero-padded smem
// [tile_rows][kLdH]. 16-byte vector path when rows are 16B-aligned (D%8==0).
__device__ __forceinline__ void stage_half(const __half* __restrict__ g,
                                           int rows_valid, int D,
                                           int tile_rows,
                                           __half* __restrict__ sh) {
  const int tid = threadIdx.x;
  if ((D & 7) == 0) {
    constexpr int vpr = kD / 8;  // uint4 slots per padded row
    const int dv = D / 8;
    for (int i = tid; i < tile_rows * vpr; i += blockDim.x) {
      const int r = i / vpr, c8 = i - r * vpr;
      uint4 val = make_uint4(0u, 0u, 0u, 0u);
      if (r < rows_valid && c8 < dv)
        val = *reinterpret_cast<const uint4*>(g + (int64_t)r * D + c8 * 8);
      *reinterpret_cast<uint4*>(sh + r * kLdH + c8 * 8) = val;
    }
  } else {
    for (int i = tid; i < tile_rows * kD; i += blockDim.x) {
      const int r = i / kD, c = i - r * kD;
      __half val = __float2half(0.f);
      if (r < rows_valid && c < D) val = g[(int64_t)r * D + c];
      sh[r * kLdH + c] = val;
    }
  }
}

// Dequantize a packed-INT4 key tile into zero-padded fp16 smem. The fp16
// values are the SAME __float2half(code * safe) the composed instrument
// materializes and apa_int4_dequant2 reproduces — the dq rounding is
// load-bearing for the threshold mask.
__device__ __forceinline__ void stage_bulk_dq(const uint8_t* __restrict__ cbase,
                                              const float* __restrict__ sbase,
                                              int64_t j0, int rows_valid, int D,
                                              __half* __restrict__ sh) {
  const int tid = threadIdx.x;
  const int half_d = D >> 1;
  constexpr int pairs = kD / 2;
  for (int i = tid; i < kTK * pairs; i += blockDim.x) {
    const int r = i / pairs, b = i - r * pairs;
    float d0 = 0.f, d1 = 0.f;
    if (r < rows_valid && b < half_d) {
      const uint8_t byte = cbase[(j0 + r) * half_d + b];
      const float safe = sbase[j0 + r];
      d0 = (float)((int)(byte & 0xF) - 7) * safe;
      d1 = (float)((int)(byte >> 4) - 7) * safe;
    }
    sh[r * kLdH + 2 * b] = __float2half(d0);
    sh[r * kLdH + 2 * b + 1] = __float2half(d1);
  }
}

// One warp computes its 16-row band of the (kTQ x kTK) score tile:
// C = A_band(16 x kD) * B^T(kD x kTK) via m16n16k16 HMMA, fp32 accumulate.
// Bt is the K-layout (kTK x kLdH) smem tile; loading it col_major with
// ldm=kLdH reads it as K^T. Ends with __syncwarp so the band is readable
// across the lanes of this warp (no CTA-wide sync required: bands are
// warp-disjoint).
__device__ __forceinline__ void mma_scores(const __half* __restrict__ Qs,
                                           const __half* __restrict__ Bt,
                                           float* __restrict__ Ss, int warp) {
  using namespace nvcuda;
  #pragma unroll
  for (int n = 0; n < kTK / 16; ++n) {
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fill_fragment(c, 0.f);
    #pragma unroll
    for (int kk = 0; kk < kD / 16; ++kk) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
      wmma::load_matrix_sync(a, Qs + (16 * warp) * kLdH + kk * 16, kLdH);
      wmma::load_matrix_sync(b, Bt + (16 * n) * kLdH + kk * 16, kLdH);
      wmma::mma_sync(c, a, b, c);
    }
    wmma::store_matrix_sync(Ss + (16 * warp) * kLdS + 16 * n, c, kLdS,
                            wmma::mem_row_major);
  }
  __syncwarp();
}

// kAPA=true : APA at r<1 — INT4 bulk + per-row selection statistics +
//             mix_scores blend + fp16 score rounding. kLadder picks the
//             statistics (EXP-APA-4/K2):
//   kLadder=false (v2 DEFAULT): fp32 Welford running mean/M2 over |bulk16|
//             [Welford 1962, "Note on a method for calculating corrected
//             sums of squares and products"; lane-pair combination via the
//             parallel merge of Chan/Golub/LeVeque 1983]. ONE stats walk
//             (Welford needs no pre-computed mean, so the ladder's second
//             walk is reclaimed). The bulk16 VALUES keep EXP-APA-2's
//             registered semantics (f2h(fp32-MMA x scale)); only the
//             selection statistics leave the fp16 lattice. Chosen over the
//             naive fp32 sumsq - mean^2 shortcut because EXP-APA-2 measured
//             that form catastrophically cancelled at DiT magnitudes
//             (|bulk| ~ 361 +/- 0.6 -> 4,862 negative variances); Welford's
//             update keeps the running M2 a sum of non-negative terms, so
//             variance stays finite and non-negative and r means r at every
//             block (the fp16 ladder saturated sum16 to +inf on up to 76%
//             of rows at the last single-stream block and refined NOTHING).
//   kLadder=true (TC_APA_THR=ladder16): the EXP-APA-2/3 fp16 overflow
//             ladder, byte-for-byte (walks 1-2 below untouched) — the
//             preserved forensic instrument.
// kAPA=false: exact streaming SDPA math (fused_sdpa_noncausal semantics and
//             the APA refine_all case) — scores stay fp32, no bulk walks,
//             kLadder unused.
// refine_count (nullable): TC_APA_FRAC=1 instrumentation — refined (row,key)
//             pairs accumulated with one warp-reduced atomicAdd per warp.
// Row ownership: warp w owns rows 16w..16w+15; a LANE PAIR owns one row
// (sub = lane&1 -> score columns [16*sub,16*sub+16) of each K tile and
// output columns [32*sub, 32*sub+32)); per-row softmax state (m, l) and the
// threshold statistics live in registers, pair-combined with shfl_xor.
template <bool kAPA, bool kLadder>
__global__ void __launch_bounds__(kThreads)
attn_noncausal_f16_kernel(const __half* __restrict__ q,
                          const __half* __restrict__ k,
                          const uint8_t* __restrict__ codes,
                          const float* __restrict__ kscale,
                          const __half* __restrict__ v,
                          __half* __restrict__ out, int Lq, int Lk, int D,
                          int num_qtiles, float scale, float zthr,
                          float inv_cnt,
                          unsigned long long* __restrict__ refine_count) {
  const int tile = blockIdx.x % num_qtiles;
  const int64_t bh = blockIdx.x / num_qtiles;
  const int q0 = tile * kTQ;
  const int qrows = min(kTQ, Lq - q0);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int half_d = D >> 1;
  constexpr unsigned FULL = 0xffffffffu;

  const __half* qbase = q + (bh * Lq + q0) * (int64_t)D;
  const __half* kbase = k + bh * (int64_t)Lk * D;
  const __half* vbase = v + bh * (int64_t)Lk * D;
  const uint8_t* cbase = kAPA ? codes + bh * (int64_t)Lk * half_d : nullptr;
  const float* sbase = kAPA ? kscale + bh * (int64_t)Lk : nullptr;

  __shared__ __half Qs[kTQ * kLdH];
  __shared__ __half Ks[kTK * kLdH];
  __shared__ __half Vs[kTK * kLdH];
  __shared__ float Ss[kTQ * kLdS];
  __shared__ __half DQs[kAPA ? kTK * kLdH : 2];
  __shared__ __half Sb[kAPA ? kTQ * kLdB : 2];

  stage_half(qbase, qrows, D, kTQ, Qs);
  __syncthreads();

  const int rloc = 16 * warp + (lane >> 1);  // CTA-local query row
  const int sub = lane & 1;
  const int ntiles = (Lk + kTK - 1) / kTK;

  float thr = -3.0e38f;  // refine-everything unless the APA stats say else
  if constexpr (kAPA && !kLadder) {
    // Stats walk (v2 DEFAULT, EXP-APA-4): SHIFTED fp32 Welford running
    // mean/M2 over the |bulk16| row, one walk. Formulation (cited):
    // Welford 1962 one-pass updates applied to x_j = |bulk16_j| - shift
    // with shift = the row's FIRST |bulk16| value (the shifted one-pass
    // form Chan/Golub/LeVeque 1983 recommend); each lane of the row's lane
    // pair runs its own sequential Welford over its key columns
    // (16*sub..16*sub+15 of each tile, ascending tiles), the two lane
    // states merge with the Chan parallel formula (both lanes share the
    // row shift), variance is shift-invariant, and mean = shift + mean_x.
    // WHY shifted (K2-STATS receipt): unshifted one-pass error grows with
    // the row's conditioning kappa = mean/std — at the DiT single-stream
    // near-constant rows (|bulk| ~ 361 +/- 0.6, kappa ~ 600) measured std
    // rel err vs fp64 was 5.4e-3-class; shifting re-centers kappa to O(1)
    // and lands mean/std/thr in the registered 1e-6 class. The naive fp32
    // sumsq - mean^2 shortcut stays banned (EXP-APA-2: 4,862 negative
    // variances on those same rows). Bulk16 values ride the identical
    // deterministic dequant+MMA+f2h(x*scale) path as the blend walk.
    float wn = 0.f, wmean = 0.f, wm2 = 0.f, wshift = 0.f;
    for (int t = 0; t < ntiles; ++t) {
      const int64_t j0 = (int64_t)t * kTK;
      const int jrows = min(kTK, Lk - (int)j0);
      stage_bulk_dq(cbase, sbase, j0, jrows, D, DQs);
      __syncthreads();
      mma_scores(Qs, DQs, Ss, warp);
      if (t == 0)  // row's first |bulk16|; column 0 valid (Lk >= 1), both
                   // pair lanes read the same smem slot -> identical shift
        wshift = fabsf(__half2float(__float2half(Ss[rloc * kLdS] * scale)));
      #pragma unroll
      for (int c = 0; c < 16; ++c) {
        const int j = 16 * sub + c;
        if (j < jrows) {
          const float a =
              fabsf(__half2float(__float2half(Ss[rloc * kLdS + j] * scale)));
          const float x = a - wshift;
          wn += 1.f;                       // exact: wn <= Lk < 2^24
          const float d = x - wmean;
          wmean += d / wn;
          wm2 += d * (x - wmean);          // non-negative increment class
        }
      }
      __syncthreads();
    }
    // Pair merge (Chan et al. 1983); empty-side states carry zero weight.
    const float on = __shfl_xor_sync(FULL, wn, 1);
    const float om = __shfl_xor_sync(FULL, wmean, 1);
    const float o2 = __shfl_xor_sync(FULL, wm2, 1);
    const float nab = wn + on;             // = Lk >= 1 (launcher-enforced)
    const float dm = om - wmean;
    const float meanx = wmean + dm * (on / nab);
    const float m2 = wm2 + o2 + dm * dm * (wn * (on / nab));
    const float var = fmaxf(m2 / nab, 0.f);  // population (ddof=0), ops::var
    thr = (wshift + meanx) + sqrtf(var) * zthr;
  }
  if constexpr (kAPA && kLadder) {
    // Walk 1: bulk INT4 scores -> fp32 sum of |bulk16| per row, then the
    // engine reduce_sum/mul_scalar fp16 ladder (saturation to +inf INCLUDED
    // — the EXP-APA-2 finding; those rows refine nothing).
    float psum = 0.f;
    for (int t = 0; t < ntiles; ++t) {
      const int64_t j0 = (int64_t)t * kTK;
      const int jrows = min(kTK, Lk - (int)j0);
      stage_bulk_dq(cbase, sbase, j0, jrows, D, DQs);
      __syncthreads();
      mma_scores(Qs, DQs, Ss, warp);
      float part = 0.f;
      #pragma unroll
      for (int c = 0; c < 16; ++c) {
        const int j = 16 * sub + c;
        if (j < jrows)
          part += fabsf(__half2float(__float2half(Ss[rloc * kLdS + j] * scale)));
      }
      psum += part;
      __syncthreads();
    }
    psum += __shfl_xor_sync(FULL, psum, 1);
    const __half sum16 = __float2half(psum);              // reduce_sum output
    const __half mean16 =
        __float2half(__half2float(sum16) * inv_cnt);      // mul_scalar
    const float meanv = __half2float(mean16);

    // Walk 2: composed (a - mean)^2 fp16 chain -> variance/threshold ladder.
    // Bulk scores are recomputed through the identical deterministic path,
    // so the a values are bit-identical to walk 1's.
    float psum2 = 0.f;
    for (int t = 0; t < ntiles; ++t) {
      const int64_t j0 = (int64_t)t * kTK;
      const int jrows = min(kTK, Lk - (int)j0);
      stage_bulk_dq(cbase, sbase, j0, jrows, D, DQs);
      __syncthreads();
      mma_scores(Qs, DQs, Ss, warp);
      float part = 0.f;
      #pragma unroll
      for (int c = 0; c < 16; ++c) {
        const int j = 16 * sub + c;
        if (j < jrows) {
          const float a =
              fabsf(__half2float(__float2half(Ss[rloc * kLdS + j] * scale)));
          const __half d16 = __float2half(a - meanv);     // sub
          const float df = __half2float(d16);
          part += __half2float(__float2half(df * df));    // mul
        }
      }
      psum2 += part;
      __syncthreads();
    }
    psum2 += __shfl_xor_sync(FULL, psum2, 1);
    const __half ddsum16 = __float2half(psum2);           // reduce_sum output
    const __half var16 =
        __float2half(__half2float(ddsum16) * inv_cnt);    // mul_scalar
    const __half std16 = __float2half(sqrtf(__half2float(var16)));  // sqrt
    const __half sz16 = __float2half(__half2float(std16) * zthr);   // mul z
    thr = __half2float(__float2half(meanv + __half2float(sz16)));   // add
  }

  // Walk 3 (the only walk for SDPA/refine_all): blend + online softmax + PV.
  float m = -3.0e38f;
  float l = 0.f;
  unsigned int nref = 0;  // TC_APA_FRAC: refined pairs this lane (kAPA only)
  float o[32];
  #pragma unroll
  for (int i = 0; i < 32; ++i) o[i] = 0.f;

  for (int t = 0; t < ntiles; ++t) {
    const int64_t j0 = (int64_t)t * kTK;
    const int jrows = min(kTK, Lk - (int)j0);
    stage_half(kbase + j0 * D, jrows, D, kTK, Ks);
    stage_half(vbase + j0 * D, jrows, D, kTK, Vs);
    if constexpr (kAPA) stage_bulk_dq(cbase, sbase, j0, jrows, D, DQs);
    __syncthreads();

    if constexpr (kAPA) {
      // Bulk tile first (same f2h(x*scale) as walks 1-2), frozen into Sb,
      // then the exact tile overwrites the same warp-owned Ss band.
      mma_scores(Qs, DQs, Ss, warp);
      #pragma unroll
      for (int c = 0; c < 16; ++c) {
        const int j = 16 * sub + c;
        Sb[rloc * kLdB + j] = __float2half(Ss[rloc * kLdS + j] * scale);
      }
      __syncwarp();
    }
    mma_scores(Qs, Ks, Ss, warp);

    // Score finalize + tile max (2 lanes per row; pair-combined).
    float sv[16];
    float tmax = -3.0e38f;
    #pragma unroll
    for (int c = 0; c < 16; ++c) {
      const int j = 16 * sub + c;
      float s = -3.0e38f;
      if (j < jrows) {
        if constexpr (kAPA) {
          // mix_scores: refine = |bulk| >= thr -> exact (fp16-rounded), else
          // the bulk score is KEPT (nothing dropped).
          const float e16 = __half2float(
              __float2half(Ss[rloc * kLdS + j] * scale));
          const float bk = __half2float(Sb[rloc * kLdB + j]);
          const bool refine = fabsf(bk) >= thr;
          if (refine_count != nullptr && rloc < qrows)
            nref += refine ? 1u : 0u;      // padding rows excluded
          s = refine ? e16 : bk;
        } else {
          s = Ss[rloc * kLdS + j] * scale;  // fp32, unrounded (SDPA class)
        }
        tmax = fmaxf(tmax, s);
      }
      sv[c] = s;
    }
    tmax = fmaxf(tmax, __shfl_xor_sync(FULL, tmax, 1));
    const float m_new = fmaxf(m, tmax);
    const float corr = expf(m - m_new);

    float wsum = 0.f;
    #pragma unroll
    for (int c = 0; c < 16; ++c) {
      const int j = 16 * sub + c;
      float w = 0.f;
      if (j < jrows) {
        w = expf(sv[c] - m_new);  // expf: U_EXP parity, as the legacy kernel
        wsum += w;
      }
      Ss[rloc * kLdS + j] = w;  // Ss band becomes the fp32 weight tile
    }
    l = l * corr + (wsum + __shfl_xor_sync(FULL, wsum, 1));
    m = m_new;
    __syncwarp();  // pair lane's weight columns must be visible below

    // PV on CUDA cores: fp32 weights x fp16 V, fp32 accumulate (the
    // streaming kernel's precision class). Lane owns 32 output columns.
    #pragma unroll
    for (int i = 0; i < 32; ++i) o[i] *= corr;
    for (int j = 0; j < jrows; ++j) {
      const float w = Ss[rloc * kLdS + j];
      const __half2* vrow =
          reinterpret_cast<const __half2*>(Vs + j * kLdH + 32 * sub);
      #pragma unroll
      for (int i2 = 0; i2 < 16; ++i2) {
        const float2 vv = __half22float2(vrow[i2]);
        o[2 * i2] += w * vv.x;
        o[2 * i2 + 1] += w * vv.y;
      }
    }
    __syncthreads();  // Ks/Vs/DQs/Ss are re-staged next tile
  }

  const float inv = l > 0.f ? 1.f / l : 0.f;
  if (rloc < qrows) {
    __half* orow = out + (bh * Lq + q0 + rloc) * (int64_t)D;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
      const int c = 32 * sub + i;
      if (c < D) orow[c] = __float2half(o[i] * inv);
    }
  }

  if constexpr (kAPA) {
    // TC_APA_FRAC flush: warp-reduce the lane counts, one atomic per warp.
    if (refine_count != nullptr) {
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        nref += __shfl_down_sync(FULL, nref, off);
      if (lane == 0 && nref > 0)
        atomicAdd(refine_count, (unsigned long long)nref);
    }
  }
}

// TC_ATTN_QTILE=0 forces the legacy streaming skeleton (bit-identical
// EXP-APA-2 reference outputs + A/B perf receipts). Read per call so one
// process can compare both paths.
inline bool enabled() {
  const char* e = std::getenv("TC_ATTN_QTILE");
  return !(e && std::strcmp(e, "0") == 0);
}

// EXP-APA-4 (K2): threshold-statistics mode for the Q-tile APA kernel.
// Default (unset / "welford") = fp32 Welford selection statistics (v2);
// TC_APA_THR=ladder16 = the EXP-APA-2/3 fp16 overflow ladder, preserved
// bit-faithfully (forensics + A/B lever). Read per call. The legacy
// streaming kernels (TC_ATTN_QTILE=0) are the frozen EXP-APA-2 instrument
// and always run the ladder regardless of TC_APA_THR.
inline bool thr_ladder16() {
  const char* e = std::getenv("TC_APA_THR");
  if (!e || !*e || std::strcmp(e, "welford") == 0) return false;
  if (std::strcmp(e, "ladder16") == 0) return true;
  throw std::runtime_error("TC_APA_THR must be 'welford' or 'ladder16'");
}

// EXP-APA-4 realized-refine-fraction instrumentation (TC_APA_FRAC=1): a
// process-global device counter of refined (row,key) pairs plus a host-side
// denominator, accumulated across calls until tc::apa_refine_stats(reset)
// reads them. Null counter pointer = instrumentation off (no atomics).
inline bool frac_enabled() {
  const char* e = std::getenv("TC_APA_FRAC");
  return e && std::strcmp(e, "1") == 0;
}
unsigned long long* g_refine_dev = nullptr;
unsigned long long g_refine_total = 0;

}  // namespace qtile

bool qtile_launch_sdpa(const NDArray& q, const NDArray& k, const NDArray& v,
                       NDArray& out, int Lq, int Lk, int D, float scale) {
  if (q.dtype != DType::Float16 || D > 64 || !qtile::enabled()) return false;
  const int num_qtiles = (Lq + qtile::kTQ - 1) / qtile::kTQ;
  const int64_t blocks =
      (int64_t)q.shape[0] * q.shape[1] * num_qtiles;  // <= rows64 <= INT_MAX
  qtile::attn_noncausal_f16_kernel<false, false>
      <<<(unsigned)blocks, qtile::kThreads>>>(
          static_cast<__half*>(q.data_ptr()), static_cast<__half*>(k.data_ptr()),
          nullptr, nullptr, static_cast<__half*>(v.data_ptr()),
          static_cast<__half*>(out.data_ptr()), Lq, Lk, D, num_qtiles, scale,
          0.f, 0.f, nullptr);
  cuda_check_last("fused_sdpa_noncausal[qtile]");
  return true;
}
}  // namespace

NDArray apa_selective_attention_int4(const NDArray& q, const NDArray& k,
                                     const NDArray& v, float scale,
                                     float zthr, bool is_causal) {
  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4)
    throw std::runtime_error("apa_selective_int4 expects q/k/v rank 4");
  if (q.dtype != k.dtype || q.dtype != v.dtype)
    throw std::runtime_error("apa_selective_int4 dtype mismatch");
  if (q.dtype != DType::Float32 && q.dtype != DType::Float16 &&
      q.dtype != DType::BFloat16)
    throw std::runtime_error(
        "apa_selective_int4 supports float32/float16/bfloat16 only");
  if (q.device.type != k.device.type || q.device.index != k.device.index ||
      q.device.type != v.device.type || q.device.index != v.device.index)
    throw std::runtime_error("apa_selective_int4 device mismatch");

  const int64_t B64 = q.shape[0], H64 = q.shape[1], L64 = q.shape[2];
  const int64_t D64 = q.shape[3], KVH64 = k.shape[1], S64 = k.shape[2];
  const int64_t VD64 = v.shape[3];
  if (B64 < 0 || H64 <= 0 || L64 < 0 || D64 <= 0 || KVH64 <= 0 ||
      S64 <= 0 || VD64 <= 0)
    throw std::runtime_error("apa_selective_int4 requires positive geometry");
  if (k.shape[0] != B64 || k.shape[3] != D64 ||
      v.shape[0] != B64 || v.shape[1] != KVH64 || v.shape[2] != S64)
    throw std::runtime_error("apa_selective_int4 shape mismatch");
  if (H64 % KVH64 != 0)
    throw std::runtime_error(
        "apa_selective_int4 requires query_heads divisible by kv_heads");
  if (is_causal && S64 < L64)
    throw std::runtime_error(
        "apa_selective_int4 bottom-right causal requires S >= L");
  const int64_t cap64 = D64 > VD64 ? D64 : VD64;
  if (cap64 > TC_APA_MAXD)
    throw std::runtime_error(
        "apa_selective_int4 head/value dim exceeds TC_APA_MAXD (512)");
  const int64_t rows64 = B64 * H64 * L64;
  const int64_t nkeys64 = B64 * KVH64 * S64;
  if (B64 > std::numeric_limits<int>::max() ||
      H64 > std::numeric_limits<int>::max() ||
      L64 > std::numeric_limits<int>::max() ||
      S64 > std::numeric_limits<int>::max() ||
      D64 > std::numeric_limits<int>::max() ||
      VD64 > std::numeric_limits<int>::max() ||
      KVH64 > std::numeric_limits<int>::max() ||
      rows64 > std::numeric_limits<int>::max())
    throw std::runtime_error("apa_selective_int4 shape exceeds CUDA launch range");

  NDArray out({B64, H64, L64, VD64}, q.dtype, q.device);
  if (rows64 == 0) return out;
  const int B = (int)B64, H = (int)H64, L = (int)L64;
  const int S = (int)S64, D = (int)D64, VD = (int)VD64;
  const int KVH = (int)KVH64, group = H / KVH;
  const int cap = (int)cap64;
  const int packed_d = (D + 1) >> 1;
  constexpr int threads = 128;

  // Transient call-owned workspace: exactly ceil(D/2) bytes plus one fp32
  // scale per (batch, KV head, key).  It is stream-ordered and freed when
  // this call's local NDArrays leave scope; no persistent kq ring exists.
  NDArray codes({B64, KVH64, S64, (int64_t)packed_d}, DType::Uint8, q.device);
  NDArray kscales({B64, KVH64, S64}, DType::Float32, q.device);
  const int64_t pack_blocks64 = (nkeys64 + 3) / 4;  // four warps/block
  if (pack_blocks64 > std::numeric_limits<unsigned>::max())
    throw std::runtime_error("apa_selective_int4 pack grid exceeds CUDA range");

  DISPATCH_FLOAT(q.dtype, T, {
    apa_int4_pack_kernel<T><<<(unsigned)pack_blocks64, threads>>>(
        static_cast<T*>(k.data_ptr()), static_cast<uint8_t*>(codes.data_ptr()),
        static_cast<float*>(kscales.data_ptr()), nkeys64, D);
    cuda_check_last("apa_selective_int4_pack");

    const int override_path = apa_selective_path_override();
    if (override_path == 2 && L != 1)
      throw std::runtime_error(
          "apa_selective_int4: TC_APA_SELECTIVE_PATH=2 requires L==1");
    const bool use_splitk = (override_path == 2 && L == 1) ||
        (override_path == 0 && apa_selective_use_splitk((int)rows64, S, L));
    if (use_splitk) {
      out = apa_selective_int4_splitk_dispatch<T>(
          q, k, codes, kscales, v, scale, zthr, is_causal, B, H, L, S, D,
          VD, KVH, group, cap, packed_d);
    } else {
      auto launch = [&](auto dmax_tag, auto wcoop_tag) {
        constexpr int DMAX = decltype(dmax_tag)::value;
        constexpr bool WCOOP = decltype(wcoop_tag)::value;
        apa_selective_int4_kernel<T, DMAX, WCOOP><<<(unsigned)rows64, threads>>>(
            static_cast<T*>(q.data_ptr()), static_cast<T*>(k.data_ptr()),
            static_cast<uint8_t*>(codes.data_ptr()),
            static_cast<float*>(kscales.data_ptr()),
            static_cast<T*>(v.data_ptr()), static_cast<T*>(out.data_ptr()),
            B, H, L, S, D, VD, scale, zthr, is_causal ? 1 : 0, KVH, group,
            packed_d);
      };
      if (cap <= 64) {
        if (apa_selective_decode_shaped((int)rows64, L))
          launch(std::integral_constant<int, 64>{}, std::false_type{});
        else
          launch(std::integral_constant<int, 64>{}, std::true_type{});
      } else if (cap <= 128) {
        launch(std::integral_constant<int, 128>{}, std::true_type{});
      } else if (cap <= 256) {
        launch(std::integral_constant<int, 256>{}, std::true_type{});
      } else {
        launch(std::integral_constant<int, 512>{}, std::true_type{});
      }
      cuda_check_last("apa_selective_int4");
    }
  });
  return out;
}

// EXP-APA-4 (K2): read (and optionally reset) the TC_APA_FRAC counters.
std::pair<unsigned long long, unsigned long long> apa_refine_stats(bool reset) {
  unsigned long long refined = 0;
  if (qtile::g_refine_dev != nullptr) {
    if (cudaMemcpy(&refined, qtile::g_refine_dev, sizeof(refined),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
      throw std::runtime_error("apa_refine_stats: counter copy failed");
    if (reset &&
        cudaMemset(qtile::g_refine_dev, 0, sizeof(refined)) != cudaSuccess)
      throw std::runtime_error("apa_refine_stats: counter reset failed");
  }
  const unsigned long long total = qtile::g_refine_total;
  if (reset) qtile::g_refine_total = 0;
  return {refined, total};
}

NDArray apa_int4_sdpa_noncausal(const NDArray& q, const NDArray& k,
                                const NDArray& v, float scale, float zthr,
                                bool refine_all) {
  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4)
    throw std::runtime_error("apa_int4_sdpa_noncausal expects q/k/v rank 4");
  if (q.dtype != k.dtype || q.dtype != v.dtype)
    throw std::runtime_error("apa_int4_sdpa_noncausal dtype mismatch");
  if (q.device.type != k.device.type || q.device.index != k.device.index ||
      q.device.type != v.device.type || q.device.index != v.device.index)
    throw std::runtime_error("apa_int4_sdpa_noncausal device mismatch");
  if (q.dtype != DType::Float16 && q.dtype != DType::Float32)
    throw std::runtime_error("apa_int4_sdpa_noncausal supports float16/float32 only");

  const int64_t B = q.shape[0], H = q.shape[1];
  const int64_t Lq64 = q.shape[2], D64 = q.shape[3], Lk64 = k.shape[2];
  if (k.shape[0] != B || v.shape[0] != B || k.shape[1] != H || v.shape[1] != H ||
      k.shape[3] != D64 || v.shape[2] != Lk64 || v.shape[3] != D64)
    throw std::runtime_error("apa_int4_sdpa_noncausal shape mismatch");
  if (D64 <= 0 || D64 > 128 || (D64 & 1))
    throw std::runtime_error("apa_int4_sdpa_noncausal requires even head_dim in [2,128]");
  if (Lq64 < 0 || Lk64 <= 0)
    throw std::runtime_error("apa_int4_sdpa_noncausal requires non-empty key sequence");
  const int64_t rows64 = B * H * Lq64;
  const int64_t nkeys64 = B * H * Lk64;
  if (rows64 > std::numeric_limits<int>::max() || Lq64 > std::numeric_limits<int>::max() ||
      Lk64 > std::numeric_limits<int>::max())
    throw std::runtime_error("apa_int4_sdpa_noncausal shape exceeds CUDA launch range");

  NDArray out(q.shape, q.dtype, q.device);
  if (rows64 == 0) return out;
  const int Lq = (int)Lq64;
  const int Lk = (int)Lk64;
  const int D = (int)D64;
  constexpr int threads = 128;
  // engine mean = mul_scalar(sum, 1.0/cnt): double reciprocal cast to float.
  const float inv_cnt = (float)(1.0 / (double)Lk64);
  // EXP-APA-3: Q-tiled skeleton for fp16 D<=64 (no Lk-dependent shared
  // memory, so no key-length cap); fp32 and D>64 keep the streaming
  // skeleton. TC_ATTN_QTILE=0 forces legacy (the EXP-APA-2 reference).
  const bool use_qtile =
      (q.dtype == DType::Float16 && D <= 64 && qtile::enabled());
  // Per-row bulk cache in dynamic shared memory (legacy path only); stay
  // under the 48 KB default static+dynamic budget (DiT joint is 4442 keys =
  // 8.9 KB fp16).
  const size_t smem = refine_all ? 0 : (size_t)Lk * dtype_size(q.dtype);
  if (!use_qtile && smem > 44 * 1024)
    throw std::runtime_error(
        "apa_int4_sdpa_noncausal: key sequence too long for the shared bulk "
        "cache (S*sizeof(dtype) must be <= 44KB); use the composed path");

  // Bulk-K INT4 codes (2/byte) + per-key fp32 scale. Query-independent,
  // packed once per call; skipped entirely under refine_all.
  NDArray codes({B, H, Lk64, D64 / 2}, DType::Uint8, q.device);
  NDArray kscales({B, H, Lk64}, DType::Float32, q.device);
  if (use_qtile) {
    if (!refine_all) {
      const int warps_per_block = threads / 32;
      const int64_t pack_blocks =
          (nkeys64 + warps_per_block - 1) / warps_per_block;
      apa_int4_pack_kernel<__half><<<(unsigned)pack_blocks, threads>>>(
          static_cast<__half*>(k.data_ptr()),
          static_cast<uint8_t*>(codes.data_ptr()),
          static_cast<float*>(kscales.data_ptr()), nkeys64, D);
      cuda_check_last("apa_int4_pack");
    }
    const int num_qtiles = (Lq + qtile::kTQ - 1) / qtile::kTQ;
    const int64_t blocks = (int64_t)B * H * num_qtiles;  // <= rows64
    if (refine_all) {
      // r >= 1: no bulk pass; the kernel is exact streaming SDPA (fp32
      // scores, no fp16 rounding) — same as the legacy refine_all contract.
      // TC_APA_THR is deliberately not consulted (threshold-free path).
      qtile::attn_noncausal_f16_kernel<false, false>
          <<<(unsigned)blocks, qtile::kThreads>>>(
              static_cast<__half*>(q.data_ptr()),
              static_cast<__half*>(k.data_ptr()), nullptr, nullptr,
              static_cast<__half*>(v.data_ptr()),
              static_cast<__half*>(out.data_ptr()), Lq, Lk, D, num_qtiles,
              scale, 0.f, 0.f, nullptr);
    } else {
      // TC_APA_FRAC=1: lazily allocate the process-global refined-pair
      // counter and accumulate the denominator (padding rows never count).
      unsigned long long* frac_ptr = nullptr;
      if (qtile::frac_enabled()) {
        if (qtile::g_refine_dev == nullptr) {
          if (cudaMalloc(&qtile::g_refine_dev, sizeof(unsigned long long)) !=
                  cudaSuccess ||
              cudaMemset(qtile::g_refine_dev, 0, sizeof(unsigned long long)) !=
                  cudaSuccess)
            throw std::runtime_error(
                "apa_int4_sdpa_noncausal: TC_APA_FRAC counter alloc failed");
        }
        frac_ptr = qtile::g_refine_dev;
        qtile::g_refine_total +=
            (unsigned long long)rows64 * (unsigned long long)Lk;
      }
      if (qtile::thr_ladder16()) {
        qtile::attn_noncausal_f16_kernel<true, true>
            <<<(unsigned)blocks, qtile::kThreads>>>(
                static_cast<__half*>(q.data_ptr()),
                static_cast<__half*>(k.data_ptr()),
                static_cast<uint8_t*>(codes.data_ptr()),
                static_cast<float*>(kscales.data_ptr()),
                static_cast<__half*>(v.data_ptr()),
                static_cast<__half*>(out.data_ptr()), Lq, Lk, D, num_qtiles,
                scale, zthr, inv_cnt, frac_ptr);
      } else {
        qtile::attn_noncausal_f16_kernel<true, false>
            <<<(unsigned)blocks, qtile::kThreads>>>(
                static_cast<__half*>(q.data_ptr()),
                static_cast<__half*>(k.data_ptr()),
                static_cast<uint8_t*>(codes.data_ptr()),
                static_cast<float*>(kscales.data_ptr()),
                static_cast<__half*>(v.data_ptr()),
                static_cast<__half*>(out.data_ptr()), Lq, Lk, D, num_qtiles,
                scale, zthr, inv_cnt, frac_ptr);
      }
    }
    cuda_check_last("apa_int4_sdpa_noncausal[qtile]");
    return out;
  }
  DISPATCH_FLOAT(q.dtype, T, {
    if (!refine_all) {
      const int warps_per_block = threads / 32;
      const int64_t pack_blocks = (nkeys64 + warps_per_block - 1) / warps_per_block;
      apa_int4_pack_kernel<T><<<(unsigned)pack_blocks, threads>>>(
          static_cast<T*>(k.data_ptr()),
          static_cast<uint8_t*>(codes.data_ptr()),
          static_cast<float*>(kscales.data_ptr()), nkeys64, D);
      cuda_check_last("apa_int4_pack");
    }
    if (D <= 64) {
      apa_int4_sdpa_noncausal_kernel<T, 64><<<(unsigned)rows64, threads, smem>>>(
          static_cast<T*>(q.data_ptr()), static_cast<T*>(k.data_ptr()),
          static_cast<uint8_t*>(codes.data_ptr()),
          static_cast<float*>(kscales.data_ptr()),
          static_cast<T*>(v.data_ptr()), static_cast<T*>(out.data_ptr()),
          Lq, Lk, D, scale, zthr, inv_cnt, refine_all ? 1 : 0);
    } else {
      apa_int4_sdpa_noncausal_kernel<T, 128><<<(unsigned)rows64, threads, smem>>>(
          static_cast<T*>(q.data_ptr()), static_cast<T*>(k.data_ptr()),
          static_cast<uint8_t*>(codes.data_ptr()),
          static_cast<float*>(kscales.data_ptr()),
          static_cast<T*>(v.data_ptr()), static_cast<T*>(out.data_ptr()),
          Lq, Lk, D, scale, zthr, inv_cnt, refine_all ? 1 : 0);
    }
  });
  cuda_check_last("apa_int4_sdpa_noncausal");
  return out;
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
// NOTE: single-pass online softmax (Phase 3.2, incl. shape-dispatched) tried and
// REVERTED — best +8-11% < 15% gate; KERNEL_OPT_IMPLEMENTATION_LEDGER 2026-07-07.
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

// ------------------------------------------------------- ring-buffer write
// In-place block write of L rows into a (.., CAP, D) buffer at ring
// positions (start + l) % CAP. THE decode-cache primitive: replaces the
// per-token cat+trim pair that allocated/copied the whole cache every
// step (~670MB/token across Gemma 4's 40 sliding layers — measured
// 0.8s/tok decode past 1K context from allocator churn alone).
// MUTATES buf: inference-only (throws under grad), and callers own the
// sharing contract — a written buffer must never be aliased by a held
// cache (copy on save/mount).
template <typename T>
__global__ void write_rows_kernel(T* __restrict__ buf,
                                  const T* __restrict__ src,
                                  int64_t CAP, int64_t L, int64_t D,
                                  int64_t start, int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int64_t d = i % D;
  int64_t l = (i / D) % L;
  int64_t b = i / (D * L);
  int64_t row = (start + l) % CAP;
  buf[(b * CAP + row) * D + d] = src[i];
}

void write_rows(NDArray& buf, const NDArray& src, int64_t start) {
  if (buf.dtype != src.dtype)
    throw std::runtime_error("write_rows: dtype mismatch");
  if (buf.ndim() < 2 || src.ndim() < 2)
    throw std::runtime_error("write_rows: need >=2D buf and src");
  int64_t D = buf.shape[buf.ndim() - 1];
  int64_t CAP = buf.shape[buf.ndim() - 2];
  int64_t Ls = src.shape[src.ndim() - 2];
  if (src.shape[src.ndim() - 1] != D)
    throw std::runtime_error("write_rows: last-dim mismatch");
  int64_t lead_b = buf.numel() / (CAP * D);
  int64_t lead_s = src.numel() / (Ls * D);
  if (lead_b != lead_s)
    throw std::runtime_error("write_rows: leading-dim mismatch");
  if (Ls > CAP)
    throw std::runtime_error("write_rows: src rows exceed capacity");
  int64_t n = src.numel();
  if (n) {
    // write_rows is a raw row copy — works for any POD type. uint8 is
    // dispatched explicitly (the float-only DISPATCH_FLOAT macro throws
    // on it) so quantized KV-storage rings (uint8 kb/vb + scale buffers)
    // can ring-write in place exactly like the bf16 rings.
    if (buf.dtype == DType::Uint8) {
      write_rows_kernel<uint8_t><<<nblk(n), kT>>>(
          static_cast<uint8_t*>(buf.data_ptr()),
          static_cast<uint8_t*>(src.data_ptr()), CAP, Ls, D, start, n);
      cuda_check_last("write_rows");
    } else {
      DISPATCH_FLOAT(buf.dtype, T, {
        write_rows_kernel<T><<<nblk(n), kT>>>(
            static_cast<T*>(buf.data_ptr()),
            static_cast<T*>(src.data_ptr()), CAP, Ls, D, start, n);
      });
      cuda_check_last("write_rows");
    }
    buf.mark_modified();
  }
}

// -------------------------------------------------------- cache row splice
// Functional arena surgery: output = old[:head] + insert + old[tail_start:].
// Copies raw bytes so bf16/fp16/fp32/uint8 cache payloads all share the same
// kernel and exact-storage semantics.
__global__ void splice_rows_bytes_kernel(
    uint8_t* __restrict__ out, const uint8_t* __restrict__ old_cache,
    const uint8_t* __restrict__ insert, int64_t old_seq, int64_t insert_seq,
    int64_t out_seq, int64_t row_bytes, int64_t head_tokens,
    int64_t tail_start, int64_t nbytes) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nbytes) return;
  int64_t out_stride = out_seq * row_bytes;
  int64_t old_stride = old_seq * row_bytes;
  int64_t insert_stride = insert_seq * row_bytes;
  int64_t outer = i / out_stride;
  int64_t rem = i - outer * out_stride;
  int64_t row = rem / row_bytes;
  int64_t byte = rem - row * row_bytes;
  const uint8_t* src = nullptr;
  if (row < head_tokens) {
    src = old_cache + outer * old_stride + row * row_bytes + byte;
  } else if (row < head_tokens + insert_seq) {
    int64_t ins_row = row - head_tokens;
    src = insert + outer * insert_stride + ins_row * row_bytes + byte;
  } else {
    int64_t tail_row = tail_start + (row - head_tokens - insert_seq);
    src = old_cache + outer * old_stride + tail_row * row_bytes + byte;
  }
  out[i] = *src;
}

__global__ void export_rows_bytes_kernel(
    uint8_t* __restrict__ out, const uint8_t* __restrict__ cache,
    int64_t old_seq, int64_t out_seq, int64_t row_bytes,
    int64_t start, int64_t nbytes) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nbytes) return;
  int64_t out_stride = out_seq * row_bytes;
  int64_t old_stride = old_seq * row_bytes;
  int64_t outer = i / out_stride;
  int64_t rem = i - outer * out_stride;
  int64_t row = rem / row_bytes;
  int64_t byte = rem - row * row_bytes;
  out[i] = cache[outer * old_stride + (start + row) * row_bytes + byte];
}

static int64_t product_dims(const Shape& shape, int first, int last) {
  int64_t out = 1;
  for (int i = first; i < last; ++i) out *= shape[i];
  return out;
}

static int normalize_dim(int dim, int nd, const char* where) {
  int d = dim < 0 ? dim + nd : dim;
  if (d < 0 || d >= nd) throw std::runtime_error(std::string(where) + ": dim out of range");
  return d;
}

NDArray export_rows(const NDArray& cache, int dim, int64_t start,
                    int64_t len) {
  if (!cache.device.is_cuda())
    throw std::runtime_error("export_rows: CUDA tensor required");
  if (cache.ndim() == 0)
    throw std::runtime_error("export_rows: rank must be nonzero");
  int d = normalize_dim(dim, cache.ndim(), "export_rows");
  int64_t old_seq = cache.shape[d];
  if (start < 0 || len < 0 || start + len > old_seq)
    throw std::runtime_error("export_rows: invalid span");
  Shape out_shape = cache.shape;
  out_shape[d] = len;
  NDArray out(out_shape, cache.dtype, cache.device);
  int64_t inner_elems = product_dims(cache.shape, d + 1, cache.ndim());
  int64_t row_bytes = inner_elems * (int64_t)dtype_size(cache.dtype);
  int64_t nbytes = out.numel() * (int64_t)dtype_size(cache.dtype);
  if (nbytes > 0) {
    export_rows_bytes_kernel<<<nblk(nbytes), kT>>>(
        static_cast<uint8_t*>(out.data_ptr()),
        static_cast<const uint8_t*>(cache.data_ptr()),
        old_seq, len, row_bytes, start, nbytes);
    cuda_check_last("export_rows");
  }
  return out;
}

NDArray splice_rows(const NDArray& old_cache, const NDArray& insert,
                    int dim, int64_t head_tokens, int64_t tail_start) {
  if (old_cache.dtype != insert.dtype)
    throw std::runtime_error("splice_rows: dtype mismatch");
  if (old_cache.device.type != insert.device.type ||
      old_cache.device.index != insert.device.index)
    throw std::runtime_error("splice_rows: device mismatch");
  if (!old_cache.device.is_cuda())
    throw std::runtime_error("splice_rows: CUDA tensors required");
  if (old_cache.ndim() == 0 || insert.ndim() != old_cache.ndim())
    throw std::runtime_error("splice_rows: rank mismatch");
  int d = normalize_dim(dim, old_cache.ndim(), "splice_rows");
  for (int i = 0; i < old_cache.ndim(); ++i) {
    if (i != d && old_cache.shape[i] != insert.shape[i])
      throw std::runtime_error("splice_rows: shape mismatch outside splice dim");
  }
  int64_t old_seq = old_cache.shape[d];
  int64_t insert_seq = insert.shape[d];
  if (head_tokens < 0 || tail_start < head_tokens || tail_start > old_seq)
    throw std::runtime_error("splice_rows: invalid head/tail");
  Shape out_shape = old_cache.shape;
  out_shape[d] = head_tokens + insert_seq + (old_seq - tail_start);
  NDArray out(out_shape, old_cache.dtype, old_cache.device);
  int64_t inner_elems = product_dims(old_cache.shape, d + 1, old_cache.ndim());
  int64_t row_bytes = inner_elems * (int64_t)dtype_size(old_cache.dtype);
  int64_t nbytes = out.numel() * (int64_t)dtype_size(old_cache.dtype);
  if (nbytes > 0) {
    splice_rows_bytes_kernel<<<nblk(nbytes), kT>>>(
        static_cast<uint8_t*>(out.data_ptr()),
        static_cast<const uint8_t*>(old_cache.data_ptr()),
        static_cast<const uint8_t*>(insert.data_ptr()),
        old_seq, insert_seq, out_shape[d], row_bytes, head_tokens,
        tail_start, nbytes);
    cuda_check_last("splice_rows");
  }
  return out;
}

NDArray evict_rows(const NDArray& old_cache, int dim, int64_t head_tokens,
                   int64_t drop_tokens) {
  if (!old_cache.device.is_cuda())
    throw std::runtime_error("evict_rows: CUDA tensor required");
  if (old_cache.ndim() == 0)
    throw std::runtime_error("evict_rows: rank must be nonzero");
  int d = normalize_dim(dim, old_cache.ndim(), "evict_rows");
  int64_t old_seq = old_cache.shape[d];
  if (head_tokens < 0 || drop_tokens < 0 || head_tokens > old_seq ||
      drop_tokens > old_seq - head_tokens)
    throw std::runtime_error("evict_rows: drop exceeds live tail");
  Shape insert_shape = old_cache.shape;
  insert_shape[d] = 0;
  NDArray empty(insert_shape, old_cache.dtype, old_cache.device);
  return splice_rows(old_cache, empty, d, head_tokens, head_tokens + drop_tokens);
}

// ------------------------------------------------------ fused RoPE apply
// out = x * cos[pos0+l] + rotate_half(x) * sin[pos0+l], one launch.
// The composed chain (2 table slices + 2 x-slices + neg + cat + 2 muls
// + add) is ~8 Python-dispatched launches PER TENSOR — at 2 tensors x
// 40 sliding layers that was ~14 ms of pure launch overhead per decoded
// token on the Gemma 4 port. Tables are (T, D) in x's dtype; rotation
// math in fp32. Inference-only.
template <typename T>
__global__ void rope_kernel(const T* __restrict__ x, const T* __restrict__ cs,
                            const T* __restrict__ sn, T* __restrict__ y,
                            int64_t pos0, int64_t L, int64_t D, int64_t n,
                            float sin_sign, int pair_swap) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int64_t d = i % D;
  int64_t l = (i / D) % L;
  int64_t half = D >> 1;
  int64_t base = i - d;
  auto src_d = [half, pair_swap](int64_t dd) {
    return pair_swap ? ((dd < half) ? 2 * dd : 2 * (dd - half) + 1) : dd;
  };
  float xv = ld<T>(x, base + src_d(d));
  float xr = (d < half)
      ? -ld<T>(x, base + src_d(d + half))
      : ld<T>(x, base + src_d(d - half));
  int64_t t = (pos0 + l) * D + d;
  st<T>(y, i, xv * ld<T>(cs, t) + xr * ld<T>(sn, t) * sin_sign);
}

NDArray rope_apply(const NDArray& x, const NDArray& cs, const NDArray& sn,
                   int64_t pos0, bool inverse, bool pair_swap) {
  if (x.ndim() < 2) throw std::runtime_error("rope_apply needs >=2D input");
  int64_t D = x.shape[x.ndim() - 1];
  int64_t L = x.shape[x.ndim() - 2];
  if (D % 2) throw std::runtime_error("rope_apply: odd head_dim");
  if (cs.dtype != x.dtype || sn.dtype != x.dtype)
    throw std::runtime_error("rope_apply: table/x dtype mismatch");
  if (cs.ndim() != 2 || cs.shape[1] != D || sn.shape[1] != D)
    throw std::runtime_error("rope_apply: tables must be (T, D)");
  if (pos0 + L > cs.shape[0])
    throw std::runtime_error("rope_apply: position past table end");
  NDArray out(x.shape, x.dtype, x.device);
  int64_t n = x.numel();
  if (n) {
    DISPATCH_FLOAT(x.dtype, T, {
      rope_kernel<T><<<nblk(n), kT>>>(
          static_cast<T*>(x.data_ptr()), static_cast<T*>(cs.data_ptr()),
          static_cast<T*>(sn.data_ptr()), static_cast<T*>(out.data_ptr()),
          pos0, L, D, n, inverse ? -1.0F : 1.0F, pair_swap ? 1 : 0);
    });
    cuda_check_last("rope_apply");
  }
  return out;
}

template <typename T>
__global__ void export_rope_rows_kernel(
    const T* __restrict__ cache, const T* __restrict__ cs,
    const T* __restrict__ sn, T* __restrict__ out, int64_t old_seq,
    int64_t out_seq, int64_t D, int64_t start, int64_t pos0,
    int64_t n, float sin_sign, int pair_swap) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int64_t d = i % D;
  int64_t l = (i / D) % out_seq;
  int64_t outer = i / (out_seq * D);
  int64_t half = D >> 1;
  int64_t src_base = (outer * old_seq + start + l) * D;
  auto src_d = [half, pair_swap](int64_t dd) {
    return pair_swap ? ((dd < half) ? 2 * dd : 2 * (dd - half) + 1) : dd;
  };
  float xv = ld<T>(cache, src_base + src_d(d));
  float xr = (d < half)
      ? -ld<T>(cache, src_base + src_d(d + half))
      : ld<T>(cache, src_base + src_d(d - half));
  int64_t t = (pos0 + l) * D + d;
  st<T>(out, i, xv * ld<T>(cs, t) + xr * ld<T>(sn, t) * sin_sign);
}

NDArray export_rope_rows(const NDArray& cache, const NDArray& cs,
                         const NDArray& sn, int dim, int64_t start,
                         int64_t len, int64_t pos0, bool inverse,
                         bool pair_swap) {
  if (!cache.device.is_cuda())
    throw std::runtime_error("export_rope_rows: CUDA tensor required");
  if (cache.ndim() < 2)
    throw std::runtime_error("export_rope_rows: needs >=2D input");
  int d = normalize_dim(dim, cache.ndim(), "export_rope_rows");
  if (d != cache.ndim() - 2)
    throw std::runtime_error("export_rope_rows: seq dim must be -2");
  int64_t D = cache.shape[cache.ndim() - 1];
  int64_t old_seq = cache.shape[d];
  if (D % 2) throw std::runtime_error("export_rope_rows: odd head_dim");
  if (start < 0 || len < 0 || start + len > old_seq)
    throw std::runtime_error("export_rope_rows: invalid span");
  if (cache.dtype != cs.dtype || cache.dtype != sn.dtype)
    throw std::runtime_error("export_rope_rows: table/cache dtype mismatch");
  if (cs.ndim() != 2 || cs.shape[1] != D || sn.shape[1] != D)
    throw std::runtime_error("export_rope_rows: tables must be (T, D)");
  if (pos0 < 0 || pos0 + len > cs.shape[0])
    throw std::runtime_error("export_rope_rows: position past table end");
  Shape out_shape = cache.shape;
  out_shape[d] = len;
  NDArray out(out_shape, cache.dtype, cache.device);
  int64_t n = out.numel();
  if (n) {
    DISPATCH_FLOAT(cache.dtype, T, {
      export_rope_rows_kernel<T><<<nblk(n), kT>>>(
          static_cast<T*>(cache.data_ptr()), static_cast<T*>(cs.data_ptr()),
          static_cast<T*>(sn.data_ptr()), static_cast<T*>(out.data_ptr()),
          old_seq, len, D, start, pos0, n, inverse ? -1.0F : 1.0F,
          pair_swap ? 1 : 0);
    });
    cuda_check_last("export_rope_rows");
  }
  return out;
}

std::tuple<NDArray, NDArray> export_row_pair(
    const NDArray& raw_cache, const NDArray& rope_cache, const NDArray& cs,
    const NDArray& sn, int raw_dim, int rope_dim, int64_t raw_start,
    int64_t rope_start, int64_t len, int64_t pos0, bool inverse,
    bool pair_swap) {
  NDArray raw = export_rows(raw_cache, raw_dim, raw_start, len);
  NDArray rope = export_rope_rows(
      rope_cache, cs, sn, rope_dim, rope_start, len, pos0, inverse,
      pair_swap);
  return std::make_tuple(raw, rope);
}

std::tuple<std::vector<NDArray>, std::vector<NDArray>> export_row_pairs(
    const std::vector<NDArray>& raw_caches,
    const std::vector<NDArray>& rope_caches, const NDArray& cs,
    const NDArray& sn, int raw_dim, int rope_dim,
    const std::vector<int64_t>& raw_starts,
    const std::vector<int64_t>& rope_starts, int64_t len, int64_t pos0,
    bool inverse, bool pair_swap) {
  size_t n = raw_caches.size();
  if (rope_caches.size() != n || raw_starts.size() != n ||
      rope_starts.size() != n) {
    throw std::runtime_error("export_row_pairs: input list size mismatch");
  }
  std::vector<NDArray> raw_out;
  std::vector<NDArray> rope_out;
  raw_out.reserve(n);
  rope_out.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    auto out = export_row_pair(raw_caches[i], rope_caches[i], cs, sn, raw_dim,
                               rope_dim, raw_starts[i], rope_starts[i], len,
                               pos0, inverse, pair_swap);
    raw_out.push_back(std::get<0>(out));
    rope_out.push_back(std::get<1>(out));
  }
  return std::make_tuple(raw_out, rope_out);
}

std::tuple<std::vector<NDArray>, std::vector<NDArray>> swap_row_pairs_with_rope(
    const std::vector<NDArray>& raw_caches,
    const std::vector<NDArray>& rope_caches,
    const std::vector<NDArray>& raw_inserts,
    const std::vector<NDArray>& rope_inserts, const NDArray& cs,
    const NDArray& sn, int raw_dim, int rope_dim, int64_t head_tokens,
    int64_t tail_start, int64_t pos0, bool pair_swap) {
  size_t n = raw_caches.size();
  if (rope_caches.size() != n || raw_inserts.size() != n ||
      rope_inserts.size() != n) {
    throw std::runtime_error("swap_row_pairs_with_rope: input list size mismatch");
  }
  std::vector<NDArray> raw_out;
  std::vector<NDArray> rope_out;
  raw_out.reserve(n);
  rope_out.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    raw_out.push_back(splice_rows(raw_caches[i], raw_inserts[i], raw_dim,
                                  head_tokens, tail_start));
    NDArray rope_insert = rope_apply(rope_inserts[i], cs, sn, pos0, false,
                                     pair_swap);
    rope_out.push_back(splice_rows(rope_caches[i], rope_insert, rope_dim,
                                   head_tokens, tail_start));
  }
  return std::make_tuple(raw_out, rope_out);
}

std::tuple<std::vector<NDArray>, std::vector<NDArray>> evict_row_pairs(
    const std::vector<NDArray>& raw_caches,
    const std::vector<NDArray>& rope_caches, int raw_dim, int rope_dim,
    int64_t head_tokens, int64_t drop_tokens) {
  size_t n = raw_caches.size();
  if (rope_caches.size() != n) {
    throw std::runtime_error("evict_row_pairs: input list size mismatch");
  }
  std::vector<NDArray> raw_out;
  std::vector<NDArray> rope_out;
  raw_out.reserve(n);
  rope_out.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    raw_out.push_back(evict_rows(raw_caches[i], raw_dim, head_tokens,
                                 drop_tokens));
    rope_out.push_back(evict_rows(rope_caches[i], rope_dim, head_tokens,
                                  drop_tokens));
  }
  return std::make_tuple(raw_out, rope_out);
}

std::tuple<std::vector<NDArray>, std::vector<NDArray>, int64_t>
arena_row_pair_transaction(
    const std::vector<NDArray>& raw_caches,
    const std::vector<NDArray>& rope_caches,
    const std::vector<NDArray>& raw_inserts,
    const std::vector<NDArray>& rope_inserts, const NDArray& cs,
    const NDArray& sn, int raw_dim, int rope_dim, int64_t sink_tokens,
    int64_t current_mount_tokens, int64_t arena_width, bool pair_swap) {
  size_t layers = raw_caches.size();
  if (rope_caches.size() != layers) {
    throw std::runtime_error("arena_row_pair_transaction: cache list size mismatch");
  }
  if (sink_tokens < 0 || current_mount_tokens < 0 || arena_width < 0) {
    throw std::runtime_error("arena_row_pair_transaction: negative arena state");
  }
  if (raw_inserts.empty() && rope_inserts.empty()) {
    auto out = evict_row_pairs(raw_caches, rope_caches, raw_dim, rope_dim,
                               sink_tokens, current_mount_tokens);
    return std::make_tuple(std::get<0>(out), std::get<1>(out), (int64_t)0);
  }
  if (raw_inserts.size() != layers || rope_inserts.size() != layers) {
    throw std::runtime_error("arena_row_pair_transaction: insert list size mismatch");
  }
  int rd = normalize_dim(raw_dim, raw_inserts[0].ndim(),
                         "arena_row_pair_transaction raw");
  int pd = normalize_dim(rope_dim, rope_inserts[0].ndim(),
                         "arena_row_pair_transaction rope");
  int64_t mount_tokens = raw_inserts[0].shape[rd];
  if (mount_tokens < 0 || mount_tokens > arena_width) {
    throw std::runtime_error("arena_row_pair_transaction: mount exceeds arena width");
  }
  if (rope_inserts[0].shape[pd] != mount_tokens) {
    throw std::runtime_error("arena_row_pair_transaction: insert token mismatch");
  }
  for (size_t i = 1; i < layers; ++i) {
    int rdi = normalize_dim(raw_dim, raw_inserts[i].ndim(),
                            "arena_row_pair_transaction raw");
    int pdi = normalize_dim(rope_dim, rope_inserts[i].ndim(),
                            "arena_row_pair_transaction rope");
    if (raw_inserts[i].shape[rdi] != mount_tokens ||
        rope_inserts[i].shape[pdi] != mount_tokens) {
      throw std::runtime_error("arena_row_pair_transaction: layer token mismatch");
    }
  }
  auto out = swap_row_pairs_with_rope(
      raw_caches, rope_caches, raw_inserts, rope_inserts, cs, sn, raw_dim,
      rope_dim, sink_tokens, sink_tokens + current_mount_tokens, sink_tokens,
      pair_swap);
  return std::make_tuple(std::get<0>(out), std::get<1>(out), mount_tokens);
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
