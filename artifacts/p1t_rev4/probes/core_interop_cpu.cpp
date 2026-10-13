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
// CPU-only review probe. Prior art: DLPack 2017+ / CUDA 12.6 device-scoped event semantics.
// Taken: unchanged core source slices and ABI; ours: deterministic call tracing, never GPU numerics.
#include <pybind11/stl.h>
#include <dlpack/dlpack.h>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <sstream>
#ifdef SPY
static int current_device=1, recorded_device=-1;
extern "C" cudaError_t cudaGetDevice(int* p) { *p=current_device; return cudaSuccess; }
extern "C" cudaError_t cudaSetDevice(int d) { current_device=d; return cudaSuccess; }
extern "C" cudaError_t cudaEventCreateWithFlags(cudaEvent_t* p,unsigned) { *p=reinterpret_cast<cudaEvent_t>(100);return cudaSuccess; }
extern "C" cudaError_t cudaEventRecord(cudaEvent_t,cudaStream_t) { recorded_device=current_device;return cudaSuccess; }
extern "C" cudaError_t cudaStreamWaitEvent(cudaStream_t,cudaEvent_t,unsigned) { return cudaSuccess; }
extern "C" cudaError_t cudaEventDestroy(cudaEvent_t) { return cudaSuccess; }
extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
#endif
using namespace tc;
constexpr const char* kCapLegacy = "dltensor";
constexpr const char* kCapVersioned = "dltensor_versioned";
constexpr const char* kCapUsedLegacy = "used_dltensor";
constexpr const char* kCapUsedVersioned = "used_dltensor_versioned";

DLDataType to_dl_dtype(DType dt) {
  auto mk = [](int code, int bits) {
    DLDataType d;
    d.code = static_cast<uint8_t>(code);
    d.bits = static_cast<uint8_t>(bits);
    d.lanes = 1;
    return d;
  };
  switch (dt) {
    case DType::Float32: return mk(kDLFloat, 32);
    case DType::Float16: return mk(kDLFloat, 16);
    case DType::BFloat16: return mk(kDLBfloat, 16);
    case DType::Int64: return mk(kDLInt, 64);
    case DType::Uint8: return mk(kDLUInt, 8);
    case DType::Bool: return mk(kDLBool, 8);
  }
  throw py::buffer_error("__dlpack__: dtype has no DLPack mapping");
}

// Exact matches only: TensorCUDA never converts on import.
bool from_dl_dtype(const DLDataType& d, DType* out) {
  if (d.lanes != 1) return false;
  if (d.code == kDLFloat && d.bits == 32) { *out = DType::Float32; return true; }
  if (d.code == kDLFloat && d.bits == 16) { *out = DType::Float16; return true; }
  if (d.code == kDLBfloat && d.bits == 16) { *out = DType::BFloat16; return true; }
  if (d.code == kDLInt && d.bits == 64) { *out = DType::Int64; return true; }
  if (d.code == kDLUInt && d.bits == 8) { *out = DType::Uint8; return true; }
  if (d.code == kDLBool && d.bits == 8) { *out = DType::Bool; return true; }
  return false;
}

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

// ---- export: the manager context OWNS a reference to the Storage, so the
// memory outlives the producer Tensor's last Python reference.
struct DLPackExportCtx {
  std::shared_ptr<Storage> owner;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  DLManagedTensor legacy{};
  DLManagedTensorVersioned versioned{};
};

void export_deleter_legacy(DLManagedTensor* self) {
  delete static_cast<DLPackExportCtx*>(self->manager_ctx);
}
void export_deleter_versioned(DLManagedTensorVersioned* self) {
  delete static_cast<DLPackExportCtx*>(self->manager_ctx);
}

// PyCapsule destructors: an UNCONSUMED capsule (still named dltensor*) owns
// its manager and must call the deleter; a consumed one ("used_*") does not.
void capsule_destructor_legacy(PyObject* cap) {
  if (!PyCapsule_IsValid(cap, kCapLegacy)) return;
  PyObject *et, *ev, *etb;
  PyErr_Fetch(&et, &ev, &etb);
  auto* m = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(cap, kCapLegacy));
  if (m && m->deleter) m->deleter(m);
  PyErr_Restore(et, ev, etb);
}
void capsule_destructor_versioned(PyObject* cap) {
  if (!PyCapsule_IsValid(cap, kCapVersioned)) return;
  PyObject *et, *ev, *etb;
  PyErr_Fetch(&et, &ev, &etb);
  auto* m = static_cast<DLManagedTensorVersioned*>(
      PyCapsule_GetPointer(cap, kCapVersioned));
  if (m && m->deleter) m->deleter(m);
  PyErr_Restore(et, ev, etb);
}

// Stream contract (array API): `stream` is the CONSUMER's stream. TensorCUDA
// queues all work on the legacy default stream. None/1 = the consumer is on
// the legacy stream too (nothing to do); -1 = no synchronization; 2 = the
// per-thread default stream; > 2 = a cudaStream_t. For 2 and handles, record
// an event on the legacy stream and make the consumer wait on it.
void dlpack_fence_consumer(const py::object& stream) {
  if (stream.is_none()) return;
  if (!py::isinstance<py::int_>(stream))
    throw py::type_error("__dlpack__: stream must be an int or None");
  const long long s = stream.cast<long long>();
  if (s == -1 || s == 1) return;
  if (s == 0)
    throw py::value_error(
        "__dlpack__: stream=0 is ambiguous for CUDA under the DLPack protocol; "
        "use None or 1 (legacy default stream), 2 (per-thread default stream) "
        "or a cudaStream_t handle");
  if (s < -1) throw py::value_error("__dlpack__: invalid stream " + std::to_string(s));
  cudaStream_t consumer =
      (s == 2) ? cudaStreamPerThread
               : reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(s));
  cudaEvent_t ev = nullptr;
  cudaError_t e = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
  if (e == cudaSuccess) e = cudaEventRecord(ev, cudaStreamLegacy);  // engine stream
  if (e == cudaSuccess) e = cudaStreamWaitEvent(consumer, ev, 0);
  // Safe right after the wait is enqueued: the dependency is captured at the
  // cudaStreamWaitEvent call and the runtime frees the event once it fires.
  if (ev) cudaEventDestroy(ev);
  if (e != cudaSuccess) {
    cudaGetLastError();
    throw std::runtime_error(std::string("__dlpack__: stream fence failed: ") +
                             cudaGetErrorString(e));
  }
}

py::object tensor_dlpack(Tensor& t, py::object stream, py::object max_version,
                         py::object dl_device, py::object copy) {
  if (!t.defined() || !t.data().defined())
    throw py::buffer_error("__dlpack__: undefined tensor");
  const NDArray& a = t.data();
  if (!a.device.is_cuda())
    throw py::buffer_error(
        "__dlpack__: TensorCUDA exports CUDA tensors only; this tensor is on "
        "cpu (CPU DLPack export is not supported)");
  if (!copy.is_none() && copy.cast<bool>())
    throw py::buffer_error(
        "__dlpack__: copy=True is not supported; TensorCUDA exports zero-copy "
        "only (pass copy=None or copy=False)");
  if (!dl_device.is_none()) {
    py::sequence d = dl_device.cast<py::sequence>();
    if (py::len(d) != 2)
      throw py::value_error("__dlpack__: dl_device must be a (device_type, device_id) pair");
    const int dt = d[0].cast<int>();
    const int id = d[1].cast<int>();
    if (dt != kDLCUDA || id != a.device.index)
      throw py::buffer_error(
          "__dlpack__: dl_device=(" + std::to_string(dt) + ", " +
          std::to_string(id) + ") differs from this tensor's device (2, " +
          std::to_string(a.device.index) +
          "); a cross-device export needs a copy, which TensorCUDA does not make");
  }
  bool versioned = false;
  if (!max_version.is_none()) {
    py::sequence mv = max_version.cast<py::sequence>();
    if (py::len(mv) < 1)
      throw py::value_error("__dlpack__: max_version must be a (major, minor) pair");
    versioned = mv[0].cast<int>() >= 1;
  }
  // P1T-FIX2, N2: a tensor whose shape breaks the zero-extent rule (reachable through reshape of an empty tensor, e.g.
  // empty((0,)).reshape([0, 2**62, 4])) cannot be described with int64 strides: refused (ValueError) before the
  // consumer fence and before contiguous_strides forms them.
  {
    const int64_t n_ex = checked_numel(a.shape.data(), a.ndim(), "__dlpack__");
    checked_nbytes(n_ex, dtype_size(a.dtype), a.shape.data(), a.ndim(), "__dlpack__");
  }

  dlpack_fence_consumer(stream);

  // P1T-FIX1, F06: the context (and its shared Storage reference) is owned by a unique_ptr until the capsule takes it,
  // so a throw anywhere below (bad_alloc in the vector assignments, a failed PyCapsule_New) frees it exactly once.
  // Prior art: C++ RAII (Stroustrup) and std::unique_ptr's release() hand-off (C++11); pybind11/CPython capsule
  // ownership. Nothing new.
  auto ctx = std::make_unique<DLPackExportCtx>();
  ctx->owner = a.storage;
  ctx->shape.assign(a.shape.begin(), a.shape.end());
  Shape st = contiguous_strides(a.shape);
  ctx->strides.assign(st.begin(), st.end());
  const int nd = a.ndim();
  DLTensor dl{};
  dl.data = a.storage->ptr;  // allocation base; element 0 is data + byte_offset
  dl.byte_offset = static_cast<uint64_t>(a.offset) * dtype_size(a.dtype);
  dl.device.device_type = kDLCUDA;
  dl.device.device_id = a.device.index;
  dl.ndim = nd;
  dl.dtype = to_dl_dtype(a.dtype);
  dl.shape = nd ? ctx->shape.data() : nullptr;
  dl.strides = nd ? ctx->strides.data() : nullptr;

  PyObject* cap = nullptr;
  if (versioned) {
    DLManagedTensorVersioned& mt = ctx->versioned;
    mt.version.major = DLPACK_MAJOR_VERSION;
    mt.version.minor = DLPACK_MINOR_VERSION;
    mt.manager_ctx = ctx.get();
    mt.deleter = export_deleter_versioned;
    mt.flags = 0;
    mt.dl_tensor = dl;
    cap = PyCapsule_New(&mt, kCapVersioned, capsule_destructor_versioned);
  } else {
    DLManagedTensor& mt = ctx->legacy;
    mt.dl_tensor = dl;
    mt.manager_ctx = ctx.get();
    mt.deleter = export_deleter_legacy;
    cap = PyCapsule_New(&mt, kCapLegacy, capsule_destructor_legacy);
  }
  if (!cap) throw py::error_already_set();  // ctx is freed by its unique_ptr
  ctx.release();  // owned by the capsule from here: its destructor, or the consumer's deleter call, frees it
  return py::reinterpret_steal<py::object>(cap);
}

py::tuple tensor_dlpack_device(Tensor& t) {
  if (!t.defined() || !t.device().is_cuda())
    throw py::buffer_error(
        "__dlpack_device__: TensorCUDA exports CUDA tensors only; this tensor "
        "is on cpu");
  return py::make_tuple(static_cast<int>(kDLCUDA), t.device().index);
}

// ---- import: the Storage keeps the producer's manager alive; its destructor
// drains the Storage device's engine stream (kernels.cu) and then calls this,
// which calls the producer's deleter with the GIL held.
//
// P1T-FIX1, F04 (finalization-safe release). The last reference can die on any
// thread, including a non-Python thread while CPython is finalizing. Once
// finalization has started (or the interpreter is gone), PyGILState_Ensure is
// unsafe: CPython 3.12 terminates a non-main thread that tries to take the GIL
// during finalization (in the middle of this deleter), and newer runtimes may
// hang it. So in that state this function does NOT call PyGILState_Ensure and
// DELIBERATELY LEAKS the producer's manager (the memory is not handed back; the
// process is exiting). Ordinary releases are unchanged.
// Detection: Py_IsInitialized() plus Py_IsFinalizing() (3.13+) / the exported
// _Py_IsFinalizing() (3.12). Measured on CPython 3.12.3 (P1T_FIX1 ledger): during
// module teardown Py_IsInitialized() is already 0 and _Py_IsFinalizing() is 1,
// because 3.12 clears `initialized` when it sets the finalizing thread; the
// explicit finalizing test keeps this correct if a runtime orders those two
// stores differently. FIX1's residual (a thread that passed this test an instant
// before finalization started could still reach PyGILState_Ensure) is closed by
// P1T-FIX2 N5 below.
// Prior art: the CPython 3.12 C-API docs (Initialization, Finalization, and
// Threads: PyGILState_Ensure / Py_IsFinalizing, "calling this function from a
// thread when the runtime is finalizing will terminate the thread"); pybind11's
// guidance not to touch the GIL during interpreter shutdown. Taken: the rule.
// Ours: the deliberate-leak policy at this call site and its counters.
//
// P1T-FIX2, review finding N5 (lifecycle coordination). The check above and
// PyGILState_Ensure were not atomic with respect to the start of finalization: a
// foreign thread could pass the check, the main thread could then reach
// Py_FinalizeEx and mark the runtime finalizing, and the foreign thread's
// PyGILState_Ensure would then run PyThread_exit_thread (CPython 3.12.3
// ceval_gil.c take_gil -> _PyThreadState_MustExit), whose forced unwind through
// the noexcept Storage destructor is std::terminate (FIX1 mutant (e): rc -6).
// Closed with an admission protocol under one mutex (ImportReleaseLifecycle):
//  * a releasing thread takes the mutex, checks `closing` (and, second line of
//    defence, interpreter_accepts_gil_state()), and only then increments
//    `in_flight`; it calls PyGILState_Ensure WITHOUT the mutex, runs the deleter,
//    calls PyGILState_Release, and only after that decrements `in_flight` under
//    the mutex and notifies. A release that finds `closing` set leaks (counted).
//  * tensor_cuda registers a Python atexit callback (`_dlpack_shutdown_hook` in
//    tensor_cuda/__init__.py) that calls dlpack_begin_shutdown: with the GIL
//    RELEASED (in-flight releases need it to finish) it sets `closing` under the
//    mutex and waits on a condition variable until `in_flight` is 0 (FIX2 bounded
//    this wait by a timeout; P1T-FIX3 V3-01 below removed the bound).
//  * Ordering (verified in CPython v3.12.3 Python/pylifecycle.c, Py_FinalizeEx):
//    _PyAtExit_Call (line 1815) runs every atexit callback BEFORE
//    _PyInterpreterState_SetFinalizing / _PyRuntimeState_SetFinalizing (1832-1833)
//    and before runtime->initialized = 0 (1834). So once the hook has returned
//    with in_flight == 0, no thread is between admission and PyGILState_Release,
//    and every later release leaks without touching the GIL; nothing can call
//    PyGILState_Ensure after the finalizing flag is set.
//  * Lock order: no thread ever waits for the GIL while holding the mutex (the
//    hook drops the mutex before re-taking the GIL; releasers never call Python
//    under it), so the protocol cannot deadlock with the GIL.
//  * (FIX2 had a bound here: after tensor_cuda._DLPACK_SHUTDOWN_WAIT_S the hook
//    returned with the release still running. Round-3 review V3-01 showed that a
//    finite 5.5 s deleter then wakes during finalization and aborts the process.
//    Replaced by P1T-FIX3 below: the wait has no bound.)
//
// P1T-FIX3, round-3 review finding V3-01 (MAJOR, exit quiescence). The hook never
// returns to ordinary finalization while an admitted release is running:
//  * dlpack_begin_shutdown waits until in_flight == 0, with NO bound. Its argument
//    (tensor_cuda._DLPACK_SHUTDOWN_WAIT_S) is now only the delay of the first
//    stderr diagnostic (count still running, seconds waited); later diagnostics
//    follow a geometric schedule (each one when the time waited has doubled), so
//    a deleter that hangs for hours logs a few dozen lines at most, and the wait
//    goes on. Admission stays closed throughout (no new release is admitted).
//  * GIL: the whole wait, diagnostics included, runs inside gil_scoped_release, so
//    an admitted deleter that needs the GIL (PyGILState_Ensure, a ctypes or
//    Python deleter, time.sleep re-taking it) can always finish; the condition
//    variable wait drops the mutex; each diagnostic is written with the mutex
//    dropped (a blocked stderr cannot stop a release from leaving); the GIL is
//    re-taken only after the mutex is released. The lock order of FIX2 holds: no
//    thread waits for the GIL while holding the mutex.
//  * Self-exclusion: a thread that runs the hook from inside its own admitted
//    release (a deleter that calls the exit hook, e.g. atexit._run_exitfuncs())
//    does not wait for itself: releases on its own stack (thread-local depth)
//    cannot finish while it waits, and they necessarily finish before that thread
//    can go on to finalize. Without this, the unbounded wait would self-deadlock.
//  * Consequence (documented in tensor_cuda._dlpack_shutdown_hook too): a producer
//    deleter that never returns makes interpreter exit wait forever. That is the
//    producer's contract violation (a DLPack deleter must return); waiting is the
//    honest outcome, where FIX2's bound turned it into an abort (rc -6). Ctrl-C
//    does not interrupt the wait (no bytecode runs on the main thread during it);
//    SIGTERM / SIGKILL still end the process.
// Prior art: CPython's threading._shutdown / Py_FinalizeEx wait_for_thread_shutdown,
// which joins every non-daemon thread with no bound before finalizing (the same
// "finalization waits for work already started" rule, one level up); RCU-style
// grace periods (synchronize_rcu waits, unbounded, for every reader that entered
// before the grace period began, while new readers do not delay it; McKenney &
// Slingwine 1998) and the Linux kernel's percpu_ref (kill, then wait for zero);
// rate-limited / backed-off diagnostics such as the kernel's hung-task and RCU
// stall warnings (repeated at growing intervals while the wait goes on). Taken:
// the unbounded quiescence wait with admission closed, and periodic stall
// reports. Ours: its placement at the CPython atexit boundary of the DLPack
// import release, the doubling report schedule and the same-thread exclusion.
// Unverified here (lead to check): "synchronize_rcu grace period readers
// McKenney 1998", "rcu_cpu_stall_timeout repeat interval", "hung_task_warnings".
// Prior art: CPython's own atexit-before-finalizing ordering (above) and its
// threading._shutdown join of non-daemon threads (pylifecycle.c
// wait_for_thread_shutdown), which is the same "stop admitting, wait for the
// in-flight ones" idea one level up; the admission-counter + condition-variable
// drain is the classic reader-count / quiescence pattern (e.g. the Linux
// kernel's percpu_ref: kill, then wait for the count to drop to zero). Taken:
// the pattern. Ours: its
// placement around PyGILState_Ensure in the DLPack import release. Unverified here
// (lead to check): "percpu_ref_kill_and_confirm quiesce", "pybind11
// gil_scoped_acquire finalization".
//
// P1T-FIX2, review finding N4: the producer's deleter is called inside a
// catch-all (counted in deleter_exceptions); this function is reached from the
// noexcept Storage destructor, so an escaping C++ exception would be
// std::terminate. Called at most once per Storage (the Storage owns the manager).
std::atomic<unsigned long long> g_import_released{0};
std::atomic<unsigned long long> g_import_leaked_shutdown{0};
std::atomic<unsigned long long> g_import_leaked_closing{0};      // N5: release after the exit hook closed
std::atomic<unsigned long long> g_import_deleter_exceptions{0};  // N4
std::atomic<unsigned long long> g_import_lock_failures{0};       // N5: mutex failure (never observed) -> leak
std::atomic<unsigned long long> g_shutdown_calls{0};
std::atomic<unsigned long long> g_shutdown_in_flight_at_close{0};
std::atomic<unsigned long long> g_shutdown_timeouts{0};
std::atomic<unsigned long long> g_shutdown_still_in_flight{0};
std::atomic<unsigned long long> g_shutdown_wait_us{0};
std::atomic<unsigned long long> g_shutdown_diagnostics{0};       // V3-01: stderr lines written while waiting

// V3-01: admitted releases on the CURRENT thread's stack (incremented on admission, decremented after leaving). The
// exit hook does not wait for these: they cannot finish while the hook's thread waits (self-exclusion, see above).
thread_local long long t_admitted_depth = 0;

struct ImportReleaseLifecycle {
  std::mutex mu;
  std::condition_variable cv;
  bool closing = false;
  long long in_flight = 0;
};

// Never destroyed (a foreign thread may release after static destruction began). Constructed at module import
// (PYBIND11_MODULE below), so no release path is the first caller (its allocation could fail there).
ImportReleaseLifecycle& release_lifecycle() {
  static ImportReleaseLifecycle* s = new ImportReleaseLifecycle();
  return *s;
}

bool interpreter_accepts_gil_state() {
  if (!Py_IsInitialized()) return false;
#if PY_VERSION_HEX >= 0x030D0000
  if (Py_IsFinalizing()) return false;
#else
  if (_Py_IsFinalizing()) return false;
#endif
  return true;
}

template <typename M>
void import_release_impl(M* m) noexcept {
  if (!m || !m->deleter) return;
  ImportReleaseLifecycle& lc = release_lifecycle();
  bool admitted = false;
  try {
    std::lock_guard<std::mutex> lk(lc.mu);
    // Both checks leak without touching the GIL; the order only decides which counter records it (a release during
    // finalization keeps FIX1's leaked_interpreter_shutdown attribution).
    if (!interpreter_accepts_gil_state()) {
      g_import_leaked_shutdown.fetch_add(1);         // F04: second line of defence
    } else if (lc.closing) {
      g_import_leaked_closing.fetch_add(1);          // N5: after the exit hook closed admission: deliberate leak
    } else {
      ++lc.in_flight;
      admitted = true;
    }
  } catch (...) {
    g_import_lock_failures.fetch_add(1);
  }
  if (!admitted) return;
  ++t_admitted_depth;                                // V3-01: this thread now has an admitted release on its stack
  PyGILState_STATE g = PyGILState_Ensure();
  try {
    m->deleter(m);
    g_import_released.fetch_add(1);
  } catch (...) {
    g_import_deleter_exceptions.fetch_add(1);        // N4: contained (GIL still held while the exception dies)
  }
  PyGILState_Release(g);
  // Leave only after PyGILState_Release: the exit hook must not let finalization start while this thread still
  // touches the interpreter. notify_all on every leave (not only at zero): the hook may be waiting for a nonzero
  // count, its own thread's admitted releases (V3-01 self-exclusion).
  try {
    std::lock_guard<std::mutex> lk(lc.mu);
    --lc.in_flight;
    lc.cv.notify_all();
  } catch (...) {
    g_import_lock_failures.fetch_add(1);
  }
  --t_admitted_depth;
}

void import_release_legacy(void* ctx) noexcept {
  import_release_impl(static_cast<DLManagedTensor*>(ctx));
}
void import_release_versioned(void* ctx) noexcept {
  import_release_impl(static_cast<DLManagedTensorVersioned*>(ctx));
}

// N5 + V3-01: the atexit side (tensor_cuda/__init__.py _dlpack_shutdown_hook). Called with the GIL held by the main
// thread. Closes admission, then waits until every admitted release of OTHER threads has left (in_flight minus this
// thread's own depth == 0). There is NO bound: it never returns while one is still running (V3-01). `first_diag_s`
// (the knob tensor_cuda._DLPACK_SHUTDOWN_WAIT_S) is the time waited before the first stderr diagnostic; each later one
// is written when the time waited has doubled since the previous one (geometric schedule: k lines take 2^(k-1) times
// the first delay, so the line rate is bounded and falls as the wait goes on). NaN or negative: diagnose at once.
// Returns {in_flight_at_close, still_in_flight, timed_out, waited_us, diagnostics, own_thread_in_flight}; `timed_out`
// now means "the wait outlasted the first-diagnostic delay" (it no longer means the hook gave up), `still_in_flight`
// (other threads' admitted releases still running at return) is always 0 now, and `own_thread_in_flight` is the
// calling thread's own admitted releases that it did not wait for (0 at a real interpreter exit).
py::dict dlpack_begin_shutdown(double first_diag_s) {
  ImportReleaseLifecycle& lc = release_lifecycle();
  if (!(first_diag_s >= 0.0)) first_diag_s = 0.0;           // NaN or negative: diagnose at once (the wait is the same)
  if (first_diag_s > 86400.0) first_diag_s = 86400.0;       // keep the chrono conversion finite
  const auto first = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double>(first_diag_s));
  const long long own = t_admitted_depth;                   // this thread's own admitted releases: never waited for
  long long at_close = 0;
  long long remaining = 0;
  bool overran = false;
  unsigned long long diagnostics = 0;
  const auto t0 = std::chrono::steady_clock::now();
  {
    py::gil_scoped_release nogil;                           // admitted releases need the GIL to finish
    std::unique_lock<std::mutex> lk(lc.mu);
    lc.closing = true;                                      // admission closed for good: nothing new can start
    at_close = lc.in_flight;
    auto next = t0 + first;
    while (!lc.cv.wait_until(lk, next, [&lc, own] { return lc.in_flight <= own; })) {
      // Still running at the diagnostic time: report, then keep waiting (never return early: V3-01).
      overran = true;
      const long long running = lc.in_flight - own;
      lk.unlock();                                          // a blocked stderr must not stop a release from leaving
      const auto now = std::chrono::steady_clock::now();
      std::fprintf(stderr,
                   "tensor_cuda: %lld DLPack release(s) admitted before exit still running after %.1f s; interpreter "
                   "exit waits for them (no new release is admitted)\n",
                   running, std::chrono::duration<double>(now - t0).count());
      std::fflush(stderr);
      ++diagnostics;
      // Next line when the time waited has doubled (at least one clock tick later: no busy loop at a zero delay).
      const auto waited = now - t0;
      const auto tick = std::chrono::steady_clock::duration(1);
      next = now + (waited > tick ? waited : tick);
      lk.lock();
    }
    remaining = lc.in_flight - own;
  }  // lk is released before nogil re-takes the GIL
  const long long waited_us =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
  g_shutdown_calls.fetch_add(1);
  g_shutdown_in_flight_at_close.fetch_add(static_cast<unsigned long long>(at_close));
  if (overran) g_shutdown_timeouts.fetch_add(1);
  g_shutdown_still_in_flight.store(static_cast<unsigned long long>(remaining));
  g_shutdown_wait_us.fetch_add(static_cast<unsigned long long>(waited_us));
  g_shutdown_diagnostics.fetch_add(diagnostics);
  py::dict d;
  d["in_flight_at_close"] = at_close;
  d["still_in_flight"] = remaining;
  d["timed_out"] = overran;
  d["waited_us"] = waited_us;
  d["diagnostics"] = diagnostics;
  d["own_thread_in_flight"] = own;
  return d;
}
void import_release_noop(void*) {}

Tensor tensor_from_dlpack_capsule(py::object obj) {
  PyObject* cap = obj.ptr();
  bool versioned = false;
  if (PyCapsule_IsValid(cap, kCapVersioned)) {
    versioned = true;
  } else if (!PyCapsule_IsValid(cap, kCapLegacy)) {
    if (PyCapsule_CheckExact(cap)) {
      const char* name = PyCapsule_GetName(cap);
      PyErr_Clear();
      throw py::value_error(
          std::string("from_dlpack: capsule named '") + (name ? name : "NULL") +
          "' is not an unconsumed DLPack capsule (a capsule can be consumed "
          "only once)");
    }
    throw py::type_error(
        "from_dlpack: expected an object with __dlpack__ or a DLPack capsule");
  }
  DLTensor* dl = nullptr;
  void* manager = nullptr;
  uint64_t flags = 0;
  if (versioned) {
    auto* m = static_cast<DLManagedTensorVersioned*>(
        PyCapsule_GetPointer(cap, kCapVersioned));
    if (!m) throw py::error_already_set();
    if (m->version.major != DLPACK_MAJOR_VERSION)
      throw py::buffer_error(
          "from_dlpack: DLPack major version " + std::to_string(m->version.major) +
          " is not supported (this build speaks DLPack " +
          std::to_string(DLPACK_MAJOR_VERSION) + ".x)");
    dl = &m->dl_tensor;
    flags = m->flags;
    manager = m;
  } else {
    auto* m = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(cap, kCapLegacy));
    if (!m) throw py::error_already_set();
    dl = &m->dl_tensor;
    manager = m;
  }
  // Every refusal below leaves the capsule unconsumed, so its destructor
  // hands the memory back to the producer.
  if (flags & DLPACK_FLAG_BITMASK_READ_ONLY)
    throw py::buffer_error(
        "from_dlpack: the producer marked this tensor read-only; TensorCUDA "
        "tensors are writable, so read-only imports are refused");
  if (dl->device.device_type != kDLCUDA)
    throw py::buffer_error(
        "from_dlpack: only CUDA device memory (kDLCUDA = 2) can be imported; "
        "got device type " + std::to_string(static_cast<int>(dl->device.device_type)) +
        (dl->device.device_type == kDLCPU ? " (CPU)" : ""));
  int cur = 0;
  cudaGetDevice(&cur);
  if (dl->device.device_id != cur)
    throw py::buffer_error(
        "from_dlpack: tensor is on CUDA device " + std::to_string(dl->device.device_id) +
        " but TensorCUDA runs on device " + std::to_string(cur) +
        " (multi-GPU import is not supported)");
  DType dt;
  if (!from_dl_dtype(dl->dtype, &dt))
    throw py::buffer_error(
        "from_dlpack: DLPack dtype (code=" + std::to_string(dl->dtype.code) +
        ", bits=" + std::to_string(dl->dtype.bits) + ", lanes=" +
        std::to_string(dl->dtype.lanes) +
        ") has no exact TensorCUDA dtype; TensorCUDA never converts on import "
        "(supported: float32, float16, bfloat16, int64, uint8, bool)");
  if (dl->ndim < 0 || dl->ndim > TC_MAX_DIMS)
    throw py::buffer_error("from_dlpack: ndim " + std::to_string(dl->ndim) +
                           " is outside TensorCUDA's 0.." +
                           std::to_string(TC_MAX_DIMS));
  const int nd = dl->ndim;
  if (nd > 0 && dl->shape == nullptr)
    throw py::buffer_error("from_dlpack: NULL shape array for a tensor with dimensions");
  Shape shape(nd);
  for (int i = 0; i < nd; ++i) {
    if (dl->shape[i] < 0)
      throw py::buffer_error("from_dlpack: negative extent in shape " +
                             shape_str(dl->shape, nd));
    shape[i] = dl->shape[i];
  }
  // F03 (P1T-FIX1): checked count (ValueError on overflow, never a wrapped size).
  const int64_t numel = checked_numel(dl->shape, nd, "from_dlpack");
  // C-contiguous only; a size-1 dimension may carry any stride (it is never
  // stepped). Strides == NULL means compact row-major (DLPack < 1.2).
  if (nd > 0 && dl->strides && numel > 0) {
    int64_t expected = 1;
    for (int i = nd - 1; i >= 0; --i) {
      if (shape[i] != 1 && dl->strides[i] != expected)
        throw py::buffer_error(
            "from_dlpack: non-contiguous strides " + shape_str(dl->strides, nd) +
            " for shape " + shape_str(dl->shape, nd) +
            "; TensorCUDA imports C-contiguous (row-major) tensors only "
            "(size-1 dimensions may have any stride). Make the tensor "
            "contiguous on the producer side first.");
      expected *= shape[i];
    }
  }
  const size_t isz = dtype_size(dt);
  if (dl->byte_offset % isz != 0)
    throw py::buffer_error("from_dlpack: byte_offset " +
                           std::to_string(dl->byte_offset) +
                           " is not a multiple of the element size " +
                           std::to_string(isz));
  if (numel > 0 && dl->data == nullptr)
    throw py::buffer_error("from_dlpack: null data pointer for a non-empty tensor");
  // F03 (P1T-FIX1): checked byte size, byte_offset + bytes (the Storage's
  // nbytes) and data + byte_offset + bytes (one past the last byte); each
  // overflow raises ValueError and leaves the capsule unconsumed.
  const int64_t nbytes = checked_nbytes(numel, isz, dl->shape, nd, "from_dlpack");
  int64_t storage_bytes = 0;
  if (dl->byte_offset > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      __builtin_add_overflow(static_cast<int64_t>(dl->byte_offset), nbytes, &storage_bytes))
    throw py::value_error("from_dlpack: byte_offset " + std::to_string(dl->byte_offset) +
                          " + byte size " + std::to_string(nbytes) + " overflows int64; refused");
  uintptr_t end_address = 0;
  if (__builtin_add_overflow(reinterpret_cast<uintptr_t>(dl->data),
                             static_cast<uintptr_t>(storage_bytes), &end_address))
    throw py::value_error("from_dlpack: data + byte_offset + byte size overflows the address space; refused");

  Device dev;
  dev.type = DeviceType::CUDA;
  dev.index = cur;
  auto storage = std::make_shared<Storage>(
      dl->data, static_cast<size_t>(storage_bytes),
      dev, versioned ? import_release_versioned : import_release_legacy, manager);
  // Consume the capsule only once the Storage owns the manager.
  if (PyCapsule_SetName(cap, versioned ? kCapUsedVersioned : kCapUsedLegacy) != 0) {
    storage->external_release = import_release_noop;  // the capsule still owns it
    throw py::error_already_set();
  }
  NDArray a;
  a.storage = std::move(storage);
  a.shape = std::move(shape);
  a.dtype = dt;
  a.device = dev;
  a.offset = static_cast<int64_t>(dl->byte_offset / isz);
  return Tensor::make(a, false);
}

// Amendment 1: device-side empty (no host round-trip, no fill).
// P1T-FIX1, F03: refuses rank > TC_MAX_DIMS (the engine's op metadata arrays
// hold 8 axes) and any element-count or byte-size overflow, with ValueError,
// before allocating; a CPU allocation failure raises MemoryError instead of
// returning a nonempty tensor over a NULL buffer.
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


int main() {
 py::scoped_interpreter python{};
#ifdef SPY
 float dummy=0; Device dev;dev.type=DeviceType::CUDA;dev.index=0;
 NDArray a;a.storage=std::make_shared<Storage>(&dummy,4,dev,import_release_noop,nullptr);a.shape={1};a.dtype=DType::Float32;a.device=dev;
 Tensor t=Tensor::make(a,false);
 auto cap=tensor_dlpack(t,py::int_(0x1000),py::none(),py::none(),py::none());
 auto* m=static_cast<DLManagedTensor*>(PyCapsule_GetPointer(cap.ptr(),kCapLegacy));
 std::cout<<"tensor_device="<<m->dl_tensor.device.device_id<<" recorded_device="<<recorded_device<<" caller_device="<<current_device<<std::endl;
 return recorded_device==1 && m->dl_tensor.device.device_id==0 ? 0:1;
#else
 int queried=-1;const auto err=cudaGetDevice(&queried);cudaGetLastError();
 std::cout<<"cudaGetDevice_error="<<err<<" device="<<queried<<std::endl;
 DLManagedTensor m{}; int64_t dim=0;m.dl_tensor.device={kDLCUDA,0};m.dl_tensor.dtype={kDLFloat,32,1};m.dl_tensor.ndim=1;m.dl_tensor.shape=&dim;
 auto cap=py::reinterpret_steal<py::object>(PyCapsule_New(&m,kCapLegacy,nullptr));
 try { auto t=tensor_from_dlpack_capsule(cap);std::cout<<"IMPORT SUCCEEDED name="<<PyCapsule_GetName(cap.ptr())<<std::endl;return err!=cudaSuccess?0:3; }
 catch(const std::exception& e) {std::cout<<"IMPORT REFUSED "<<e.what()<<std::endl;return 1;}
#endif
}
