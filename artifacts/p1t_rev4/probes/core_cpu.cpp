// CPU probe compiled from verbatim reviewed source slices (source_slices.json).
// Prior art: GCC/Clang UBSan (LLVM project, 2012+) and C++ signed-overflow rules;
// taken: instrumentation; ours: zero-extent prefix/suffix adversarial cases. No CUDA stubs.
#include <pybind11/embed.h>
#include <cuda_runtime.h>
#include "tc/core.h"
#include "tc/autograd.h"
#include <atomic>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
namespace py=pybind11;
namespace tc {
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

// P1T-FIX2, review finding N2: both helpers used plain signed products, which overflowed (undefined behaviour) for
// zero-extent shapes such as [2^62, 4, 0] (numel_of: the prefix product 2^64 is formed before the 0 is reached) and
// [0, 2^62, 4] (contiguous_strides: the stride of axis 0 is 2^64) even though the element count is 0 (round-2 probe
// core_cpu prefix/suffix, UBSan). The DLPack/empty entry points now refuse such shapes first (bindings.cpp
// checked_numel/checked_nbytes, the zero-extent rule); these helpers are the second line:
//  * numel_of returns 0 for any shape with a zero extent WITHOUT multiplying (that is the exact count), and otherwise
//    multiplies with overflow checks;
//  * contiguous_strides multiplies with overflow checks and no longer forms the unused product past axis 0.
// An overflow throws std::overflow_error (Python OverflowError) instead of wrapping. Values are unchanged whenever no
// overflow occurs. Prior art: GCC/Clang __builtin_mul_overflow (GCC 5, 2015), C23 ckd_mul, CERT INT32-C. Ours: where.
int64_t numel_of(const Shape& shape) {
  for (int64_t s : shape)
    if (s == 0) return 0;
  int64_t n = 1;
  for (int64_t s : shape)
    if (__builtin_mul_overflow(n, s, &n))
      throw std::overflow_error("numel_of: the element count of a shape overflows int64");
  return n;  // 1 for a 0-d shape
}
Shape contiguous_strides(const Shape& shape) {
  Shape st(shape.size());
  int64_t acc = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    st[i] = acc;
    if (i > 0 && __builtin_mul_overflow(acc, shape[i], &acc))
      throw std::overflow_error("contiguous_strides: a stride of a shape overflows int64");
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
// P1T-M1 (DLPack import): wrap a producer's memory. No allocation, no pool.
Storage::Storage(void* external_ptr, size_t nbytes_, Device device_,
                 ExternalRelease release, void* release_ctx)
    : ptr(external_ptr), nbytes(nbytes_), device(device_),
      external_release(release), external_ctx(release_ctx) {}

// P1T-FIX1, F05: device-correct drain for an external (imported) Storage.
//  1. If `dev` is not the current device, switch to it (the caller may have
//     switched devices since the import; draining the CURRENT device's legacy
//     stream would leave this allocation's last use pending).
//  2. Synchronize that device's legacy stream (stream 0 in this TU).
//  3. Report failure (any step) so the caller does NOT release, and clear the
//     runtime's last-error slot so an unrelated later launch check does not
//     report this destructor's error.
//  4. Restore the caller's device.
// No device-wide sync (cudaDeviceSynchronize): one stream sync, as before.
// Prior art: the CUDA runtime current-device model (cudaGetDevice/cudaSetDevice;
// the legacy default stream is per device) and the scoped device guard of
// PyTorch (c10::cuda::CUDAGuard) / CuPy (cupy.cuda.Device as a context manager).
// Taken: switch-then-restore. Ours: its placement before the DLPack deleter and
// the leak-on-failure rule.
namespace {
std::atomic<unsigned long long> g_external_drain_leaks{0};

bool drain_device_legacy_stream(int dev) {
  int prev = -1;
  cudaError_t e = cudaGetDevice(&prev);
  bool switched = false;
  if (e == cudaSuccess && prev != dev) {
    e = cudaSetDevice(dev);
    switched = (e == cudaSuccess);
  }
  if (e == cudaSuccess) e = cudaStreamSynchronize(0);
  if (switched) cudaSetDevice(prev);
  if (e != cudaSuccess) cudaGetLastError();
  return e == cudaSuccess;
}
}  // namespace

unsigned long long storage_external_drain_leaks() { return g_external_drain_leaks.load(); }

Storage::~Storage() {
  if (external_release) {
    // Imported (DLPack) memory belongs to its producer: never cudaFree it and
    // never hand it to the pool. ORDER MATTERS:
    //  1. Drain the engine stream. Every engine kernel, cuBLAS call and
    //     cudaMemcpy runs on stream 0 (the legacy default stream; this TU is
    //     built without --default-stream per-thread), so once it is idle no
    //     queued engine work can still read or write this memory. The engine
    //     keeps no per-Storage "last use" event, so "everything queued so far"
    //     is the only sound fence; an event recorded here and waited on would
    //     cover exactly the same work at three API calls instead of one. A
    //     deferred release (host callback) is not allowed to run the
    //     producer's deleter, which may call CUDA (torch's caching allocator).
    //  2. Then hand the memory back (the DLPack deleter, GIL held, in
    //     bindings.cpp). A producer may reuse it at once on its own stream.
    // Prior art: the DLPack deleter contract (dmlc/dlpack, 2017+) and the
    // array API's stream rules (2021+); the drain-then-delete order is the
    // standard CUDA "free only after last use" rule. Ours: its placement in
    // the Storage destructor.
    // P1T-FIX1 F05: drain THIS Storage's device (drain_device_legacy_stream
    // above); if the drain fails, deliberately leak instead of releasing.
    if (device.is_cuda() && !drain_device_legacy_stream(device.index)) {
      g_external_drain_leaks.fetch_add(1);
      return;
    }
    external_release(external_ctx);
    return;
  }
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



Tensor Tensor::make(NDArray data, bool requires_grad) {
  auto v = std::make_shared<Variable>();
  v->data = std::move(data);
  v->requires_grad = requires_grad;
  v->op = "leaf";
  return Tensor(v);
}

}
using namespace tc;
std::string shape_str(const int64_t* v, int n) {
  std::string s = "[";
  for (int i = 0; i < n; ++i) s += (i ? ", " : "") + std::to_string(v[i]);
  return s + "]";
}

// P1T-FIX1, F03: checked element count and byte size for the two entry points that turn arbitrary caller shapes into
// a Storage (empty, from_dlpack). Every extent is >= 0 here (callers refuse negatives first); a zero extent makes the
// count 0 whatever the other extents are. Overflow raises ValueError and never wraps to an empty or short Storage.
// The byte ceiling is INT64_MAX (every byte total, offset and pointer difference stays a valid int64/ptrdiff_t).
// Prior art: GCC/Clang __builtin_mul_overflow / __builtin_add_overflow (GCC 5, 2015); C23 ckd_mul / ckd_add
// (<stdckdint.h>, ISO/IEC 9899:2024); SEI CERT C INT30-C / INT32-C. Taken: the checked-operation primitive. Ours:
// where it is applied. Unverified here (lead to check): "GCC 5 __builtin_mul_overflow", "C23 stdckdint".
//
// P1T-FIX2, review finding N2 (the zero-extent rule). FIX1 returned a count of 0 for any shape with a zero extent
// WITHOUT looking at the other extents, so [0, 2^62, 4] and [2^62, 4, 0] were accepted as empty tensors; the engine
// then formed their strides and partial counts (contiguous_strides, numel_of, byte strides) and overflowed (round-2
// probe core_cpu prefix/suffix: UBSan signed overflow). The rule now: the EXTENT BOUND, the product of every extent
// with a zero counted as 1, must fit in int64 (checked_numel), and so must the bound times the element size
// (checked_nbytes), whatever the other extents are. Every quantity the engine later forms from such a shape (count,
// each contiguous stride, each byte stride, the byte size) is at most that bound, so none can overflow. Zero-extent
// shapes whose bound fits (e.g. [0, 2^60] float32) stay real empty tensors.
// Prior art: NumPy's array constructor applies the same rule (zero dimensions skipped, the rest multiplied into the
// byte size with overflow checks): numpy 2.4.2 refuses np.empty((0, 2**62, 4), float32) and (0, 2**61) float32 ("array
// is too big"), measured in the P1T-FIX2 ledger. PyTorch 2.11 refuses the element-level part only. Taken: NumPy's
// rule. Ours: applying it to TensorCUDA's empty / from_dlpack / __dlpack__. Source-level citation unverified here (lead
// to check): "numpy ctors.c PyArray_NewFromDescr_int is_zero overflow".
int64_t checked_extent_bound(const int64_t* dims, int n, const char* who) {
  int64_t p = 1;
  for (int i = 0; i < n; ++i)
    if (__builtin_mul_overflow(p, dims[i] > 1 ? dims[i] : int64_t{1}, &p))
      throw py::value_error(std::string(who) + ": the element count of shape " + shape_str(dims, n) +
                            " overflows int64 (every nonzero extent counts, also next to a zero extent: strides and "
                            "partial counts are formed from them); refused");
  return p;
}

int64_t checked_numel(const int64_t* dims, int n, const char* who) {
  const int64_t bound = checked_extent_bound(dims, n, who);
  for (int i = 0; i < n; ++i)
    if (dims[i] == 0) return 0;
  return bound;
}

int64_t checked_nbytes(int64_t numel, size_t itemsize, const int64_t* dims, int n, const char* who) {
  const int64_t bound = checked_extent_bound(dims, n, who);
  int64_t b = 0;
  if (__builtin_mul_overflow(bound, static_cast<int64_t>(itemsize), &b))
    throw py::value_error(std::string(who) + ": the byte size of shape " + shape_str(dims, n) + " (" +
                          std::to_string(itemsize) + "-byte elements) overflows int64 (every nonzero extent counts, "
                          "also next to a zero extent); refused");
  return numel == 0 ? 0 : b;  // numel is 0 or the bound itself (checked_numel)
}

Tensor tensor_empty(std::vector<int64_t> shape, const std::string& dtype,
                    const std::string& device) {
  if (shape.size() > static_cast<size_t>(TC_MAX_DIMS))
    throw py::value_error("empty: " + std::to_string(shape.size()) +
                          " dimensions exceed TensorCUDA's maximum rank " +
                          std::to_string(TC_MAX_DIMS) + " (TC_MAX_DIMS)");
  for (int64_t s : shape)
    if (s < 0) throw py::value_error("empty: negative dimensions are not allowed");
  const DType dt = dtype_from_string(dtype);
  const int nd = static_cast<int>(shape.size());
  const int64_t numel = checked_numel(shape.data(), nd, "empty");
  const int64_t nbytes = checked_nbytes(numel, dtype_size(dt), shape.data(), nd, "empty");
  NDArray a = NDArray::empty(Shape(shape.begin(), shape.end()), dt, Device::from_string(device));
  if (nbytes > 0 && a.storage->ptr == nullptr) throw std::bad_alloc();  // -> MemoryError
  return Tensor::make(a, false);
}



int main(int argc,char**argv) {
 py::scoped_interpreter python{};
 if(argc!=2) return 2;
 try {
   std::string mode=argv[1];
   Shape s = mode == "prefix" ? Shape{1LL<<62,4,0} : Shape{0,1LL<<62,4};
   if (mode=="normal") s={0,2,4};
   std::cout<<"checked_numel="<<checked_numel(s.data(),s.size(),"probe")<<std::endl;
   Tensor t=tensor_empty(s,"float32","cpu");
   std::cout<<"empty succeeded nbytes="<<t.data().storage->nbytes<<std::endl;
   auto strides=contiguous_strides(t.data().shape);
   std::cout<<"strides built"<<std::endl;
 } catch(const std::exception& e) { std::cout<<"SAFE REFUSAL: "<<e.what()<<std::endl; return 0; }
 return 0;
}
