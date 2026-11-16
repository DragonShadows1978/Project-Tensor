// bindings.cpp: pybind11 module exposing the C++/CUDA engine to Python.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>  // P1T-FIX3 V3-01: exit-wait diagnostics written without the GIL
#include <initializer_list>  // P1T-FIX6 sweep (g): CoreCallArgs
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>  // P1T-FIX6 sweep (g): std::overflow_error -> OverflowError
#include <string>
#include <system_error>  // P1T-FIX7: the lifecycle mutex's lock failure (retried in the exit close)
#include <thread>  // P1T-FIX7: the retry pause
#include <utility>
#include <vector>

#include "dlpack/dlpack.h"  // vendored dmlc/dlpack v1.3 (git blob 9252a448)
#include "tc/autograd.h"
#include "tc/bp_op_timing.h"
#include "tc/ops.h"

namespace tc {
// P1T-FIX4 (defined in kernels.cu; declared here as the file does for its other cross-TU helpers, because core.h is
// outside the FIX4 order's scope): the allocation-vs-label device proof of the P1-T surface, and the counter of
// caller-device restores that failed where nothing could raise.
std::string cuda_storage_device_mismatch(const NDArray& a, const char* who);
unsigned long long dlpack_device_restore_failures();
// P1T-FIX5 (sweep d, defined in kernels.cu): CUDA errors of the P1-T core paths that cannot raise (external Storage
// drain and its restore, fill_'s guard unwinding), recorded before their slot entry is consumed: count, and the first.
unsigned long long dlpack_unraisable_errors();
void dlpack_first_unraisable(std::string* what, int* device, int* error);
// P1T-FIX6 (FIX5 residual R7, a recorded error that nothing ever raised; defined in kernels.cu): the part of that
// record not raised yet (count, and the first of them), and its consumption once a RuntimeError naming it exists.
void dlpack_unreported_unraisable(unsigned long long* count, std::string* what, int* device, int* error);
void dlpack_ack_unraisable(unsigned long long count) noexcept;
// P1T-FIX8 (sweep (k), defined in kernels.cu): throw a RuntimeError whose report exists before the slot is consumed.
[[noreturn]] void throw_cuda_report_after_consuming(const std::string& msg);
// P1T-FIX8 (V8-02, defined in kernels.cu): the same snapshot without allocating (the site text is a static string).
void dlpack_unreported_unraisable_raw(unsigned long long* count, const char** what, int* device, int* error) noexcept;
// Prior art: pybind11 (Jakob 2015) and BP-KERNEL-2 binding pattern, taken.
void bp_kernel_4_set_variant(const std::string&);
std::string bp_kernel_4_get_variant();
std::tuple<NDArray,NDArray,NDArray> apa_selective_fwd_train_variant(
    const NDArray&,const NDArray&,const NDArray&,const NDArray&,float,float,bool,const std::string&);
std::tuple<NDArray,NDArray,NDArray,NDArray,NDArray> apa_selective_fwd_bk4(
    const NDArray&,const NDArray&,const NDArray&,const NDArray&,float,float,bool,const std::string&,bool);

void bp_kernel_3_profile(bool);
std::vector<float> bp_kernel_3_halves();
void bp_kernel_2_set_variant(const std::string&);
std::string bp_kernel_2_get_variant();
std::tuple<NDArray, NDArray, NDArray> apa_selective_bwd_bk1_cuda(
    const NDArray&, const NDArray&, const NDArray&, const NDArray&,
    const NDArray&, const NDArray&, const NDArray&, const NDArray&,
    float, bool, const std::string&);
std::tuple<NDArray, NDArray, NDArray> apa_selective_bwd_variant(
    const NDArray&, const NDArray&, const NDArray&, const NDArray&,
    const NDArray&, const NDArray&, const NDArray&, const NDArray&,
    float, bool, const std::string&);
}

namespace py = pybind11;
using namespace tc;

namespace {

DType numpy_dtype(const py::array& a) {
  auto dt = a.dtype();
  if (dt.kind() == 'f' && dt.itemsize() == 4) return DType::Float32;
  if (dt.kind() == 'f' && dt.itemsize() == 2) return DType::Float16;
  if (dt.kind() == 'i' && dt.itemsize() == 8) return DType::Int64;
  if (dt.kind() == 'u' && dt.itemsize() == 1) return DType::Uint8;  // packed int4
  if (dt.kind() == 'b') return DType::Bool;
  return DType::Float32;  // others are force-cast to float32 by the wrapper
}

Tensor tensor_from_numpy(py::array arr, const std::string& device, bool requires_grad) {
  DType dt = numpy_dtype(arr);
  // Force a C-contiguous buffer of the matching element size.
  py::array contig = py::array::ensure(arr, py::array::c_style);
  auto info = contig.request();
  Shape shape(info.shape.begin(), info.shape.end());
  Device dev = Device::from_string(device);
  NDArray nd = NDArray::from_host(info.ptr, shape, dt, dev);
  return Tensor::make(nd, requires_grad);
}

py::array tensor_to_numpy(Tensor& t) {
  NDArray a = t.data();
  // numpy has no bf16 — upcast to fp32 on device so the host buffer is fp32.
  if (a.dtype == DType::BFloat16) a = a.astype(DType::Float32);
  std::vector<py::ssize_t> shape(a.shape.begin(), a.shape.end());
  py::array out;
  if (a.dtype == DType::Float16) out = py::array(py::dtype("float16"), shape);
  else if (a.dtype == DType::Int64) out = py::array(py::dtype("int64"), shape);
  else if (a.dtype == DType::Uint8) out = py::array(py::dtype("uint8"), shape);
  else out = py::array(py::dtype("float32"), shape);
  NDArray host = a.device.is_cuda() ? a : a;  // to_host handles D2H
  host.to_host(out.request().ptr);
  return out;
}

py::object grad_of(Tensor& t) {
  if (!t.v->grad.defined()) return py::none();
  return py::cast(Tensor::make(t.v->grad, false));
}

py::tuple tensor_args(const std::vector<Tensor>& inputs) {
  py::tuple args(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) args[i] = py::cast(inputs[i]);
  return args;
}

Tensor checkpoint_py(py::function fn, std::vector<Tensor> inputs) {
  bool any_req = false;
  for (auto& t : inputs) any_req = any_req || t.requires_grad();
  bool req = grad_enabled() && any_req;
  if (!req) {
    py::object obj = fn(*tensor_args(inputs));
    return obj.cast<Tensor>();
  }

  py::object obj;
  {
    NoGradGuard guard;
    obj = fn(*tensor_args(inputs));
  }
  Tensor out0 = obj.cast<Tensor>();
  NDArray out_data = out0.data();

  return Tensor::from_op(out_data, inputs, "checkpoint",
      [fn, inputs](const NDArray& g) {
        py::gil_scoped_acquire gil;

        std::vector<Tensor> replay_inputs;
        replay_inputs.reserve(inputs.size());
        for (auto& t : inputs) {
          replay_inputs.push_back(Tensor::make(t.data(), t.requires_grad()));
        }

        bool prev = grad_enabled();
        set_grad_enabled(true);
        py::object replay_obj;
        try {
          // Prior art: CUDA event scopes (NVIDIA; see bp_op_timing.h).
          bp::Scope replay_scope("checkpoint_replay");
          replay_obj = fn(*tensor_args(replay_inputs));
        } catch (...) {
          set_grad_enabled(prev);
          throw;
        }
        Tensor replay_out = replay_obj.cast<Tensor>();
        try {
          replay_out.backward(g);
        } catch (...) {
          set_grad_enabled(prev);
          throw;
        }
        set_grad_enabled(prev);

        for (size_t i = 0; i < inputs.size(); ++i) {
          if (inputs[i].requires_grad() && replay_inputs[i].grad().defined()) {
            inputs[i].v->accumulate_grad(replay_inputs[i].grad());
          }
        }
      });
}

// __getitem__ for ints and unit-step slices (single key or per-dim tuple),
// composed from the differentiable slice/squeeze ops.
Tensor getitem(Tensor& t, py::object key) {
  std::vector<py::object> keys;
  if (py::isinstance<py::tuple>(key))
    for (auto k : key.cast<py::tuple>()) keys.push_back(py::reinterpret_borrow<py::object>(k));
  else
    keys.push_back(key);

  Tensor cur = t;
  int dim = 0;
  for (auto& k : keys) {
    if (py::isinstance<py::int_>(k)) {
      int64_t i = k.cast<int64_t>();
      int64_t sz = cur.shape()[dim];
      if (i < 0) i += sz;
      cur = ops::squeeze(ops::slice(cur, dim, i, 1), dim);  // dim removed; keep `dim`
    } else if (py::isinstance<py::slice>(k)) {
      size_t start, stop, step, len;
      k.cast<py::slice>().compute(cur.shape()[dim], &start, &stop, &step, &len);
      if (step != 1) throw std::runtime_error("getitem: only step==1 slices supported");
      cur = ops::slice(cur, dim, (int64_t)start, (int64_t)len);
      ++dim;
    } else {
      throw std::runtime_error("getitem: unsupported index type");
    }
  }
  return cur;
}

std::vector<TerrainRenderObject> terrain_objects_from_python(
    const py::object& objects) {
  std::vector<TerrainRenderObject> parsed;
  if (objects.is_none()) return parsed;
  if (!py::isinstance<py::list>(objects)) {
    throw std::runtime_error("terrain_render: objects must be a list or None");
  }
  const py::list entries = py::reinterpret_borrow<py::list>(objects);
  if (entries.size() > 16) {
    throw std::runtime_error("terrain_render: objects supports at most 16 entries");
  }
  parsed.reserve(entries.size());
  for (py::ssize_t index = 0; index < entries.size(); ++index) {
    const py::handle entry = entries[index];
    if ((!py::isinstance<py::tuple>(entry) &&
         !py::isinstance<py::list>(entry)) ||
        py::len(entry) != 3) {
      throw std::runtime_error(
          "terrain_render: each object must be (grid, origin, palette)");
    }
    const py::sequence terms = py::reinterpret_borrow<py::sequence>(entry);
    Tensor grid = py::cast<Tensor>(terms[0]);
    const std::vector<float> origin =
        py::cast<std::vector<float>>(terms[1]);
    Tensor palette = py::cast<Tensor>(terms[2]);
    if (origin.size() != 3 || !std::isfinite(origin[0]) ||
        !std::isfinite(origin[1]) || !std::isfinite(origin[2])) {
      throw std::runtime_error(
          "terrain_render: object origin must be a finite three-vector");
    }
    TerrainRenderObject object{};
    object.grid = grid.data();
    object.origin[0] = origin[0];
    object.origin[1] = origin[1];
    object.origin[2] = origin[2];
    object.palette = palette.data();
    parsed.push_back(std::move(object));
  }
  return parsed;
}

// ===================================================================== DLPack
// P1T-M1 (Project-Voice T1): zero-copy DLPack export/import.
//
// Prior art: DLPack (dmlc/dlpack, 2017-; v1.3 header vendored verbatim at
// include/dlpack/dlpack.h); the Python array API standard's data-interchange
// protocol (__dlpack__(stream=, max_version=, dl_device=, copy=) and
// __dlpack_device__; Consortium for Python Data API Standards, 2021-2023);
// PyTorch's ATen DLConvertor/torch.utils.dlpack and CuPy's from_dlpack
// (owner keepalive: the manager context owns a reference to the producer's
// storage, and the consumer keeps the manager until its tensor dies; capsule
// renamed "used_*" on consumption). Taken: the protocol, capsule naming and
// the keepalive pattern; the event-record + cudaStreamWaitEvent fence is the
// standard CUDA cross-stream idiom. Ours: wiring it to tc::Storage (a
// shared_ptr<Storage> owner on export; an external-release Storage that
// drains the engine stream before the deleter on import, kernels.cu).
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
//
// P1T-FIX4, round-4 review finding V4-01 (MAJOR). 62c8d9f created and recorded
// the event on the CURRENT device: with device 1 current, a device-0 tensor's
// consumer waited on device 1's legacy stream, and the device-0 writes were not
// ordered. Now the fence runs on THE TENSOR'S DEVICE (`device`, a label that
// tensor_dlpack has proven against the allocation, cuda_storage_device_mismatch):
//  * the caller's device is read (checked) and, if it differs, `device` is made
//    current (checked); then the event is created, recorded on the legacy
//    stream (of `device` now), the consumer's stream waits on it, the event is
//    destroyed on its own device, and the caller's device is restored;
//  * every return is checked; on any failure the caller's device is restored
//    FIRST, the failed calls' slot entry is consumed, and one RuntimeError names
//    each failed step (and says whether the restore itself failed);
//  * the wait is issued with `device` current, so stream=2 means the per-thread
//    default stream of the tensor's device. That matches PyTorch 2.11's DLPack
//    consumer, which passes the current stream OF THE TENSOR'S DEVICE
//    (torch.utils.dlpack.from_dlpack: torch.cuda.current_stream(f'cuda:{ext_device[1]}'),
//    read in the installed package). A handle (> 2) may belong to any device:
//    CUDA 12.6 cudaStreamWaitEvent: "event may be from a different device than
//    stream" (cuda_runtime_api.h); Programming Guide 3.2.9.3: "cudaStreamWaitEvent()
//    will succeed even if the input stream and input event are associated to
//    different devices", while "cudaEventRecord() will fail if the input event and
//    input stream are associated to different devices", which is why creation and
//    record need the tensor's device current (receipt artifacts/p1t_fix4/cuda_doc_quotes.txt).
// No device-wide sync is added (one event, as before).
// Prior art: PyTorch's scoped c10::cuda::CUDAGuard and CuPy's cupy.cuda.Device
// context (switch, act, restore); PyTorch's CUDAEvent destroys an event on its
// creating device ("to avoid creating a CUDA context on other devices",
// c10/cuda/CUDAEvent.h, torch 2.11 headers). PyTorch 2.11's own Tensor.__dlpack__
// does NOT switch: it refuses a current-device mismatch with BufferError
// (torch/_tensor.py). The order chose the guard (switch, record, restore). Taken:
// the scoped guard and destroy-on-own-device. Ours: placing the whole fence,
// consumer wait included, inside the guard, and restore-before-raise.
// P1T-FIX6 (V6-02): `stream` arrives already read by DlpackArgs (None -> has_stream false; an int's value): this
// function calls no Python at all.
void dlpack_fence_consumer(bool has_stream, long long s, int device) {
  if (!has_stream) return;
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
  const std::string dev = std::to_string(device);
  auto name = [](cudaError_t err) {
    return std::string(cudaGetErrorName(err)) + ": " + cudaGetErrorString(err);
  };
  // P1T-FIX8 (sweep (k)): every raise below builds its report before consuming the slot entry (throw_after_consuming,
  // kernels.cu): a bad_alloc while building it leaves the failure in the slot for the next check.
  int prev = -1;
  cudaError_t e = cudaGetDevice(&prev);
  if (e != cudaSuccess)
    throw_cuda_report_after_consuming("__dlpack__: stream fence: cudaGetDevice failed (" + name(e) +
                          "); nothing was switched or recorded");
  const bool switched = (prev != device);
  if (switched) {
    e = cudaSetDevice(device);
    if (e != cudaSuccess)
      throw_cuda_report_after_consuming("__dlpack__: stream fence: cannot make the tensor's CUDA device " + dev +
                            " current (cudaSetDevice: " + name(e) + "); the caller's device " +
                            std::to_string(prev) + " is unchanged and nothing was recorded");
  }
  // `device` is current from here until the restore below.
  cudaEvent_t ev = nullptr;
  const char* step = "cudaEventCreateWithFlags";
  e = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
  if (e == cudaSuccess) {
    step = "cudaEventRecord on the legacy stream";
    e = cudaEventRecord(ev, cudaStreamLegacy);  // engine stream of `device`
  }
  if (e == cudaSuccess) {
    step = "cudaStreamWaitEvent on the consumer's stream";
    e = cudaStreamWaitEvent(consumer, ev, 0);
  }
  // Safe right after the wait is enqueued: the dependency is captured at the
  // cudaStreamWaitEvent call and the runtime frees the event once it fires.
  const cudaError_t de = ev ? cudaEventDestroy(ev) : cudaSuccess;  // on its own device
  const cudaError_t re = switched ? cudaSetDevice(prev) : cudaSuccess;
  if (e == cudaSuccess && de == cudaSuccess && re == cudaSuccess) return;
  // the failed calls are reported by the raise below; their slot entry is consumed once that report exists
  std::string msg = "__dlpack__: stream fence on the tensor's CUDA device " + dev + " failed:";
  if (e != cudaSuccess) msg += std::string(" ") + step + " (" + name(e) + ");";
  if (de != cudaSuccess) msg += " cudaEventDestroy (" + name(de) + ");";
  if (re != cudaSuccess)
    msg += " restoring the caller's CUDA device " + std::to_string(prev) + " also failed (cudaSetDevice: " + name(re) +
           "), so the calling thread's current device is still " + dev;
  else if (switched)
    msg += " the caller's CUDA device " + std::to_string(prev) + " was restored";
  throw_cuda_report_after_consuming(msg);
}

// P1T-FIX6, round-6 review finding V6-02 (MAJOR; FIX5's residual R3). tensor_dlpack converted its arguments with
// pybind11 casts that run the argument's own Python methods: copy.cast<bool>() calls __bool__, dl_device /
// max_version .cast<py::sequence>() + py::len() + item .cast<int>() call __len__, __getitem__ / __iter__ and __index__,
// and stream.cast<long long>() may fall back to __int__. That user Python ran on this native stack. On a daemon thread
// whose method released the GIL across the start of interpreter finalization, CPython 3.12 exits the thread when it
// re-takes the GIL (pthread_exit, a forced unwind); the unwind deallocated the pybind11 dispatcher's private kwargs
// copy with the thread's state already freed: SIGSEGV (core_cast_exit_gpu.py bool|index rc 139 on acd42fa and
// 1475fae; gdb receipt artifacts/p1t_fix6). Core's exit hook could not see it (it waits only for import releases).
// Choice: option (ii) of the review, REFUSE before invoking anything, rather than (i) admit and wait at exit:
//  * (ii) adds no wait and no lock (sweep (h): no new place where an exit can hang), and leaves no user Python in a
//    native frame at all, so there is nothing for the exit hook to coordinate;
//  * every real consumer passes plain values (torch 2.11 from_dlpack: stream an int, max_version (1, 0), dl_device a
//    tuple of ints from torch._C._torchDeviceToDLDevice, copy a bool or absent -- read in the installed torch; tc_signal:
//    stream=1, max_version=(1, 3), copy=False; tensor_cuda.from_dlpack: stream=1, max_version=(1, 3)), all accepted;
//  * (i) would make interpreter exit wait for arbitrary user __bool__/__index__ code (sweep (h): a new exit wait that
//    any user method can extend) for no consumer that needs it.
// Accepted argument types, each read with C-level calls that never call a Python method of the argument:
//  * stream: None, or an int (int, bool, int subclasses such as IntEnum: PyLong_AsLongLongAndOverflow reads a
//    PyLong_Check object's value directly, CPython 3.12 Objects/longobject.c); out of the int64 range -> ValueError;
//  * max_version: None, or a tuple or list (PyTuple_GET_SIZE / PyList_GET_ITEM, which ignore a subclass's __len__ /
//    __getitem__) of length >= 1 whose first item (the major version, the only item read, as before) is an int;
//  * dl_device: None, or a tuple or list of exactly 2 ints (IntEnum device types accepted: torch annotates dl_device
//    as tuple[IntEnum, int]);
//  * copy: None, a bool, or an int (nonzero = true, read as above). Anything else, numpy.bool_ included, is refused:
//    reading it needs its type's nb_bool, and recognising numpy's bool by type name is spoofable by any class named
//    "numpy.bool" (pybind11's is_numpy_bool); the array API types copy as bool | None.
// Every other type raises TypeError naming the argument and the type (tp_name), before anything else happens: no repr,
// no __str__. The pybind11 binding of __dlpack__ itself takes (*args, **kwargs) (PYBIND11_MODULE below), so neither
// the dispatcher nor a mismatch message runs Python either.
// Prior art: PyTorch 2.11 Tensor.__dlpack__ (`if stream is not None and type(stream) is not int: raise
// TypeError("stream must be ``int`` or ``none``")`, read in the installed torch/_tensor.py) refuses instead of converting;
// the Python array API __dlpack__ signature (Consortium for Python Data API Standards, 2021-2023); CPython 3.12's
// PyLong_AsLongLongAndOverflow / PyTuple_GET_ITEM contracts. Taken: refusal of non-int / non-tuple arguments. Ours:
// the int-subclass and list acceptance read at C level, the numpy.bool_ refusal argument, and refusing for every
// argument of every P1-T binding (sweep (g)).
namespace {
[[noreturn]] void dlpack_refuse_arg(const char* arg, const char* expected, PyObject* o) {
  throw py::type_error(std::string("__dlpack__: ") + arg + " must be " + expected + ", not " + Py_TYPE(o)->tp_name +
                       " (P1T-FIX6: refused without calling any of its Python methods)");
}
// An int argument's value, read without calling Python; `overflow` is -1 / +1 beyond the int64 range.
long long dlpack_int(PyObject* o, int* overflow) {
  *overflow = 0;
  const long long v = PyLong_AsLongLongAndOverflow(o, overflow);
  if (v == -1 && *overflow == 0 && PyErr_Occurred()) throw py::error_already_set();  // not reached for a PyLong
  return v;
}
// A tuple's or list's size and items with the C macros (a subclass's __len__ / __getitem__ is never called).
Py_ssize_t dlpack_seq_size(PyObject* o) { return PyTuple_Check(o) ? PyTuple_GET_SIZE(o) : PyList_GET_SIZE(o); }
PyObject* dlpack_seq_item(PyObject* o, Py_ssize_t i) {
  return PyTuple_Check(o) ? PyTuple_GET_ITEM(o, i) : PyList_GET_ITEM(o, i);
}
}  // namespace

py::object tensor_dlpack(Tensor& t, py::object stream, py::object max_version,
                         py::object dl_device, py::object copy) {
  // P1T-FIX6 (V6-02): read every argument first, with C-level calls only (see above).
  bool has_stream = false;
  long long stream_value = 0;
  if (!stream.is_none()) {
    if (!PyLong_Check(stream.ptr())) dlpack_refuse_arg("stream", "an int or None", stream.ptr());
    int of = 0;
    stream_value = dlpack_int(stream.ptr(), &of);
    if (of != 0) throw py::value_error("__dlpack__: invalid stream (beyond the 64-bit range)");
    has_stream = true;
  }
  bool versioned = false;
  if (!max_version.is_none()) {
    PyObject* mv = max_version.ptr();
    if (!PyTuple_Check(mv) && !PyList_Check(mv))
      dlpack_refuse_arg("max_version", "None or a (major, minor) tuple of ints", mv);
    if (dlpack_seq_size(mv) < 1)
      throw py::value_error("__dlpack__: max_version must be a (major, minor) pair");
    PyObject* major = dlpack_seq_item(mv, 0);
    if (!PyLong_Check(major)) dlpack_refuse_arg("max_version[0] (the major version)", "an int", major);
    int of = 0;
    const long long mj = dlpack_int(major, &of);
    versioned = of > 0 || (of == 0 && mj >= 1);
  }
  bool has_dl_device = false;
  long long dl_type = 0, dl_id = 0;
  if (!dl_device.is_none()) {
    PyObject* d = dl_device.ptr();
    if (!PyTuple_Check(d) && !PyList_Check(d))
      dlpack_refuse_arg("dl_device", "None or a (device_type, device_id) tuple of ints", d);
    if (dlpack_seq_size(d) != 2)
      throw py::value_error("__dlpack__: dl_device must be a (device_type, device_id) pair");
    PyObject* d0 = dlpack_seq_item(d, 0);
    PyObject* d1 = dlpack_seq_item(d, 1);
    if (!PyLong_Check(d0)) dlpack_refuse_arg("dl_device[0] (the device type)", "an int", d0);
    if (!PyLong_Check(d1)) dlpack_refuse_arg("dl_device[1] (the device id)", "an int", d1);
    int of0 = 0, of1 = 0;
    dl_type = dlpack_int(d0, &of0);
    dl_id = dlpack_int(d1, &of1);
    if (of0 != 0 || of1 != 0)
      throw py::value_error("__dlpack__: dl_device entries must fit in 64-bit integers");
    has_dl_device = true;
  }
  bool want_copy = false;
  if (!copy.is_none()) {
    PyObject* c = copy.ptr();
    if (c == Py_True || c == Py_False) {
      want_copy = (c == Py_True);
    } else if (PyLong_Check(c)) {
      int of = 0;
      const long long v = dlpack_int(c, &of);
      want_copy = of != 0 || v != 0;
    } else {
      dlpack_refuse_arg("copy", "None or a bool", c);
    }
  }
  if (!t.defined() || !t.data().defined())
    throw py::buffer_error("__dlpack__: undefined tensor");
  const NDArray& a = t.data();
  if (!a.device.is_cuda())
    throw py::buffer_error(
        "__dlpack__: TensorCUDA exports CUDA tensors only; this tensor is on "
        "cpu (CPU DLPack export is not supported)");
  if (want_copy)
    throw py::buffer_error(
        "__dlpack__: copy=True is not supported; TensorCUDA exports zero-copy "
        "only (pass copy=None or copy=False)");
  if (has_dl_device && (dl_type != kDLCUDA || dl_id != a.device.index))
    throw py::buffer_error(
        "__dlpack__: dl_device=(" + std::to_string(dl_type) + ", " +
        std::to_string(dl_id) + ") differs from this tensor's device (2, " +
        std::to_string(a.device.index) +
        "); a cross-device export needs a copy, which TensorCUDA does not make");
  // P1T-FIX2, N2: a tensor whose shape breaks the zero-extent rule (reachable through reshape of an empty tensor, e.g.
  // empty((0,)).reshape([0, 2**62, 4])) cannot be described with int64 strides: refused (ValueError) before the
  // consumer fence and before contiguous_strides forms them.
  {
    const int64_t n_ex = checked_numel(a.shape.data(), a.ndim(), "__dlpack__");
    checked_nbytes(n_ex, dtype_size(a.dtype), a.shape.data(), a.ndim(), "__dlpack__");
  }

  // P1T-FIX4 (V4-01 and the device sweep): the device this export publishes (dl.device_id below) and fences on is the
  // tensor's label, proven first against the allocation of element 0 (kernels.cu cuda_storage_device_mismatch:
  // cudaPointerGetAttributes). A mismatch, which the engine's cuda:0 labeling produces for a tensor it allocated while
  // another device was current, is refused (BufferError) before any fence or capsule; it would otherwise describe the
  // memory with the wrong device and fence the wrong device's stream (the V4-01 race again).
  {
    const std::string mismatch = cuda_storage_device_mismatch(a, "__dlpack__");
    if (!mismatch.empty()) throw py::buffer_error(mismatch);
  }
  dlpack_fence_consumer(has_stream, stream_value, a.device.index);

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
//  * GIL: the whole wait, diagnostics included, runs with the GIL released (P1T-FIX5: an
//    explicit PyEval_SaveThread / PyEval_RestoreThread pair, see dlpack_begin_shutdown), so
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
// P1T-FIX7 (V7-01): closes whose diagnostic argument could not be read (the default delay was used), the exit runner's
// own closes after the Python hook raised before closing (and the releases of other threads it then found running),
// and mutex failures retried by the close (never observed).
std::atomic<unsigned long long> g_shutdown_diag_default_used{0};
std::atomic<unsigned long long> g_shutdown_runner_closes{0};
std::atomic<unsigned long long> g_shutdown_runner_in_flight_at_close{0};
std::atomic<unsigned long long> g_shutdown_lock_retries{0};
// P1T-FIX7, lead send-back L1: stderr lines written while the close retried a failed lifecycle-mutex lock (also
// counted in g_shutdown_diagnostics when the hook's own close wrote them).
std::atomic<unsigned long long> g_shutdown_lock_diagnostics{0};

// P1T-FIX7 (V7-01) exit tunables of the import-release lifecycle (P1-T core; none bounds the wait, none changes
// numerics): the default first-diagnostic delay (the initial value of the Python knob
// tensor_cuda._DLPACK_SHUTDOWN_WAIT_S, exported as _C._DLPACK_SHUTDOWN_DIAG_DEFAULT_S, and the delay used when the knob
// cannot be read at C level); the cap that keeps the chrono conversion finite (FIX3's literal 86400, now named); the
// pause between retries of the lifecycle mutex when locking it fails in the close (never observed).
constexpr double kDlpackShutdownDiagDefaultS = 5.0;
constexpr double kDlpackShutdownMaxDiagDelayS = 86400.0;
constexpr int kDlpackShutdownLockRetryMs = 10;

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
// P1T-FIX7, round-7 review finding V7-01 (MAJOR): reached by the Python hook with ANY knob value (the raw entry reads
// it at C level and falls back to kDlpackShutdownDiagDefaultS: `diag_default_used`), and by the atexit runner after the
// hook raised before closing (`runner`: counted apart). Sweep (i) also found one path INSIDE this function that could
// skip the close: locking the lifecycle mutex can throw std::system_error (C++17 [thread.mutex.requirements.mutex]),
// and FIX3-FIX6 re-took the GIL and rethrew before `closing` was set (with one injected lock failure 429446d aborts at
// exit, rc -6: artifacts/p1t_fix7/lock_failure_receipt_before_B.txt). The lock is now retried (GIL released,
// kDlpackShutdownLockRetryMs apart, counted) until it succeeds, so the close is never skipped and the function never
// returns before the wait has finished (a permanently broken mutex makes exit wait, as a deleter that never returns
// does, instead of letting finalization overtake a release).
// P1T-FIX7, lead send-back L1: that retry loop is an exit wait too, so it is diagnosed exactly like the timed wait
// below (FIX3 V3-01: never silent). Once the retries of one lock attempt have lasted the first-diagnostic delay (the
// caller's clamped first_diag_s), a line goes to stderr (C stdio, GIL released, no allocation: what() is the
// exception's own string) naming what is waited for, how many attempts failed and for how long; then another each
// time the time spent retrying has doubled. Each line is counted (g_shutdown_lock_diagnostics, and in the caller's
// `diagnostics`). When the lock is finally taken after a diagnostic, one more line says so. Prior art: FIX3's V3-01
// diagnostic schedule (first delay, then doubling), itself after the kernel's repeated stall warnings (RCU CPU stall
// warnings, hung-task detector; unverified here, lead to check "RCU CPU stall warning repeated"); ours: applying it to
// a lock-retry loop.
namespace {
void lock_retrying(std::unique_lock<std::mutex>& lk, std::chrono::steady_clock::duration first, const char* purpose,
                   unsigned long long& diagnostics) {
  using clock = std::chrono::steady_clock;
  unsigned long long failed = 0;
  unsigned long long written = 0;
  clock::time_point r0{};
  clock::time_point next{};
  for (;;) {
    try {
      lk.lock();
      if (written > 0) {
        std::fprintf(stderr,
                     "tensor_cuda: interpreter exit took the DLPack release lifecycle lock %s after %llu failed "
                     "attempt(s) over %.1f s; exit continues\n",
                     purpose, failed, std::chrono::duration<double>(clock::now() - r0).count());
        std::fflush(stderr);
      }
      return;
    } catch (const std::system_error& e) {
      g_shutdown_lock_retries.fetch_add(1);
      const auto now = clock::now();
      if (failed++ == 0) {
        r0 = now;
        next = r0 + first;
      }
      if (now >= next) {
        std::fprintf(stderr,
                     "tensor_cuda: interpreter exit cannot take the DLPack release lifecycle lock %s (%s): %llu failed "
                     "attempt(s) over %.1f s; it retries every %d ms and waits (it never returns early)\n",
                     purpose, e.what(), failed, std::chrono::duration<double>(now - r0).count(),
                     kDlpackShutdownLockRetryMs);
        std::fflush(stderr);
        ++written;
        ++diagnostics;
        g_shutdown_lock_diagnostics.fetch_add(1);
        const auto waited = now - r0;            // next line when the time spent retrying has doubled
        const auto tick = clock::duration(1);    // (at least one clock tick later: no line per attempt at a zero delay)
        next = now + (waited > tick ? waited : tick);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(kDlpackShutdownLockRetryMs));
    }
  }
}
}  // namespace

py::dict dlpack_begin_shutdown_impl(double first_diag_s, bool diag_default_used, bool runner) {
  ImportReleaseLifecycle& lc = release_lifecycle();
  if (!(first_diag_s >= 0.0)) first_diag_s = 0.0;           // NaN or negative: diagnose at once (the wait is the same)
  if (first_diag_s > kDlpackShutdownMaxDiagDelayS) first_diag_s = kDlpackShutdownMaxDiagDelayS;  // finite chrono
  const auto first = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double>(first_diag_s));
  const long long own = t_admitted_depth;                   // this thread's own admitted releases: never waited for
  long long at_close = 0;
  long long remaining = 0;
  bool overran = false;
  unsigned long long diagnostics = 0;
  const auto t0 = std::chrono::steady_clock::now();
  // P1T-FIX5 (sweep (c), callbacks during interpreter exit): the GIL is released with PyEval_SaveThread and re-taken
  // with an explicit PyEval_RestoreThread in THIS frame (not noexcept), no longer by py::gil_scoped_release's
  // destructor. CPython 3.12 exits a non-main thread that re-takes the GIL during finalization (take_gil ->
  // PyThread_exit_thread -> pthread_exit, a forced unwind); started inside a destructor, that unwind meets the
  // destructor's implicit noexcept (std::terminate, abort). A non-main thread can be here: the hook is a plain Python
  // function (tensor_cuda._dlpack_shutdown_hook), and two waiters (it and the main thread's real atexit call) wake
  // together when the last admitted release leaves. Now such a thread's forced unwind travels through ordinary frames
  // to the raw function's caller. The mutex is still released before the GIL is re-taken. Same design as tc_signal's
  // release lifecycle (dlpack_operand.cpp). Prior art: CPython's PyEval_SaveThread / PyEval_RestoreThread contract;
  // glibc/libstdc++ forced unwinding (abi::__forced_unwind must propagate; a noexcept frame terminates); pybind11's
  // gil_scoped_release::disarm note ("PyEval_RestoreThread() should not be called if runtime is finalizing"). Ours:
  // the explicit restore placement.
  PyThreadState* gil_state = PyEval_SaveThread();           // admitted releases need the GIL to finish
  try {
    std::unique_lock<std::mutex> lk(lc.mu, std::defer_lock);
    lock_retrying(lk, first, "to close DLPack release admission", diagnostics);  // P1T-FIX7: never skips the close
    lc.closing = true;                                      // admission closed for good: nothing new can start
    at_close = lc.in_flight;
    auto next = t0 + first;
    // libstdc++'s wait_until on steady_clock does not throw (pthread_cond_clockwait errors other than a timeout are
    // ignored) and the predicate reads two integers; the re-lock after a diagnostic is retried as above.
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
      lock_retrying(lk, first, "again after a diagnostic (admission is already closed)", diagnostics);
    }
    remaining = lc.in_flight - own;
  } catch (...) {
    // Not reachable (see above; FIX5's safety net, kept): never leave this frame without the GIL. No Python ran in the
    // try block, so this is never a forced unwind of the interpreter's making; it is rethrown unchanged.
    PyEval_RestoreThread(gil_state);
    throw;
  }  // lk is released before the GIL is re-taken
  PyEval_RestoreThread(gil_state);                          // explicit, in this non-noexcept frame (see above)
  const long long waited_us =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
  const long long others_at_close = at_close - own > 0 ? at_close - own : 0;
  if (runner) {
    g_shutdown_runner_closes.fetch_add(1);
    g_shutdown_runner_in_flight_at_close.fetch_add(static_cast<unsigned long long>(others_at_close));
  } else {
    g_shutdown_calls.fetch_add(1);
    g_shutdown_in_flight_at_close.fetch_add(static_cast<unsigned long long>(at_close));
    if (overran) g_shutdown_timeouts.fetch_add(1);
    g_shutdown_still_in_flight.store(static_cast<unsigned long long>(remaining));
    g_shutdown_wait_us.fetch_add(static_cast<unsigned long long>(waited_us));
    g_shutdown_diagnostics.fetch_add(diagnostics);
    if (diag_default_used) g_shutdown_diag_default_used.fetch_add(1);
  }
  py::dict d;  // built after the close and the wait: a MemoryError here cannot skip them
  d["in_flight_at_close"] = at_close;
  d["still_in_flight"] = remaining;
  d["timed_out"] = overran;
  d["waited_us"] = waited_us;
  d["diagnostics"] = diagnostics;
  d["own_thread_in_flight"] = own;
  d["others_in_flight_at_close"] = others_at_close;      // P1T-FIX7
  d["first_diag_s"] = first_diag_s;                      // P1T-FIX7: the delay actually used (after NaN / cap handling)
  d["first_diag_default_used"] = diag_default_used;      // P1T-FIX7: the knob could not be read; the default was used
  return d;
}

// The one-argument form (FIX2-FIX6 signature, kept: the reviewers' host slices bind it as `close`).
py::dict dlpack_begin_shutdown(double first_diag_s) { return dlpack_begin_shutdown_impl(first_diag_s, false, false); }
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
  // P1T-FIX4, round-4 review finding V4-03 (MINOR). 62c8d9f ignored this query's
  // result: on failure (no usable device, or an initialization error) `cur`
  // stayed 0, a device-0 capsule was accepted, a Storage was built on an
  // unvalidated device and the capsule was renamed "used_*". Now the result is
  // checked here, before the Storage exists and before the rename: a failure
  // raises RuntimeError naming cudaGetDevice and the CUDA error, the capsule
  // stays unconsumed (its own destructor returns the memory to the producer),
  // and the failed call's slot entry is consumed (this raise reports it). This
  // is the only CUDA call of the import (the sweep: no other query). A sticky
  // context error does not make cudaGetDevice fail (measured, receipt
  // artifacts/p1t_fix4/cuda_semantics.txt); such an import proceeds, and the
  // Storage's release later finds the drain failing and leaks instead of
  // releasing (P1T-FIX1 F05).
  // Prior art: the CUDA 12.6 contract that cudaGetDevice returns an error code
  // (cudaErrorNoDevice, cudaErrorInsufficientDriver, ...) and the DLPack rule that
  // a refused capsule keeps its ownership. Nothing new.
  int cur = -1;
  const cudaError_t qe = cudaGetDevice(&cur);
  if (qe != cudaSuccess)  // P1T-FIX8 (sweep (k)): the report exists before the slot entry is consumed
    throw_cuda_report_after_consuming(std::string("from_dlpack: the current-device query cudaGetDevice failed (") +
                          cudaGetErrorName(qe) + ": " + cudaGetErrorString(qe) +
                          "); refused before taking ownership (the capsule is not consumed)");
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
  // P1T-FIX4 (sweep, implicit current device). The engine's Storage constructor allocates on the CURRENT device, but
  // Device::from_string (pre-existing core code) labels every CUDA string index 0, so with another device current
  // 62c8d9f returned memory of that device labeled cuda:0 (and cuda:N was silently ignored). Now a CUDA tensor is
  // labeled with the current device, read with a checked cudaGetDevice (RuntimeError naming it, slot entry consumed),
  // and an explicit "cuda:N" must name the current device (ValueError otherwise; TensorCUDA does not switch devices to
  // allocate). The label then equals the allocation's device, which the P1-T export and fill_ prove before use.
  // Prior art: torch.empty(device=...) / cupy.empty under cupy.cuda.Device (an allocation belongs to an explicit,
  // recorded device). Taken: label = allocation device. Ours: refusing an index other than the current device.
  Device dev = Device::from_string(device);
  if (dev.is_cuda()) {
    int cur = -1;
    const cudaError_t qe = cudaGetDevice(&cur);
    if (qe != cudaSuccess)  // P1T-FIX8 (sweep (k)): the report exists before the slot entry is consumed
      throw_cuda_report_after_consuming(std::string("empty: the current-device query cudaGetDevice failed (") +
                            cudaGetErrorName(qe) + ": " + cudaGetErrorString(qe) + "); nothing was allocated");
    const std::string prefix = "cuda:";
    if (device.compare(0, prefix.size(), prefix) == 0) {
      const std::string idx = device.substr(prefix.size());
      if (idx.empty() || idx.size() > 9 || idx.find_first_not_of("0123456789") != std::string::npos)
        throw py::value_error("empty: device '" + device + "' is not 'cuda', 'cuda:<index>' or 'cpu'");
      if (std::stoi(idx) != cur)
        throw py::value_error("empty: device '" + device + "' is not the current CUDA device (cuda:" +
                              std::to_string(cur) + "); TensorCUDA allocates on the current device only");
    }
    dev.index = cur;
  }
  NDArray a;
  try {
    a = NDArray::empty(Shape(shape.begin(), shape.end()), dt, dev);
  } catch (const std::runtime_error&) {
    // A failed CUDA allocation (the engine's Storage constructor raises without consuming the failed call's slot
    // entry): this raise reports it, so it is consumed here (P1T-FIX4 sweep).
    if (dev.is_cuda()) cudaGetLastError();
    throw;
  }
  if (nbytes > 0 && a.storage->ptr == nullptr) throw std::bad_alloc();  // -> MemoryError
  return Tensor::make(a, false);
}

const char* const kDlpackDoc = R"doc(Export this tensor zero-copy as a DLPack capsule (Python array API protocol).

stream: the CONSUMER's CUDA stream, as a DLPack int. TensorCUDA queues all of its
  work on the legacy default stream.
  * None or 1: the consumer is on the legacy default stream; nothing to do.
  * -1: no synchronization.
  * 2 (per-thread default stream) or any other handle (a cudaStream_t): TensorCUDA
    records a CUDA event on the legacy default stream of THIS TENSOR'S device and
    makes the consumer's stream wait on it before returning, so the consumer sees
    every TensorCUDA write queued so far. If another device is current, TensorCUDA
    switches to the tensor's device for the fence and restores the caller's device
    (also when the fence fails, before raising RuntimeError). With 2, the wait is
    on the per-thread default stream of the tensor's device; a handle may belong
    to any device (a cross-device event wait is legal in CUDA).
  * 0 is ambiguous under the protocol and raises ValueError.
Device proof (P1T-FIX4): before anything else, the tensor's device label is
checked against its allocation (cudaPointerGetAttributes); a tensor whose memory
is on another device than its label (the engine labels its own allocations
cuda:0, so one allocated while another device was current) raises BufferError.
max_version: None (or major < 1) returns a legacy "dltensor" capsule; (1, x) or
  later returns "dltensor_versioned" (DLPack 1.3 header, flags 0).
dl_device: None or (2, this tensor's device); anything else raises BufferError.
copy: None or False. copy=True raises BufferError (TensorCUDA never copies on
  export).

The capsule's manager holds a reference to the tensor's Storage, so the memory
outlives this Tensor's last Python reference. data + byte_offset addresses
element 0; strides are C-contiguous. An unconsumed capsule releases its
reference when it is garbage collected.

Raises BufferError for CPU tensors.

External in-place writes (mark_modified rule): TensorCUDA's derived-tensor
caches key on the Storage revision, which only TensorCUDA's own in-place ops
bump. After anything outside TensorCUDA (a DLPack consumer, or a kernel given
data_ptr()) writes into this tensor's memory, call t.mark_modified() before
passing t (or any tensor sharing its Storage) to a TensorCUDA op. Today the
only revision-keyed cache is terrain_render's density_filter cache on the
materials tensor; ops that read device memory directly (arithmetic, matmul,
norms, attention) need nothing. Calling mark_modified() when it is not needed
is harmless.)doc";

// ------------------------------------------------------------------------------------------------ P1T-FIX6 sweep (g)
// User Python run from inside a native frame. pybind11's typed argument casters convert in the dispatcher, before the
// function body (an int parameter calls __index__, a double __float__ / __index__, a std::vector __len__ /
// __getitem__ and each item's conversion), its no-match TypeError calls repr() on every argument, and its keyword
// matching uses dict lookups that may call a str subclass's __eq__. Any of that user Python can release the GIL on a
// daemon thread across the start of finalization; CPython 3.12 then exits the thread when it re-takes the GIL and the
// forced unwind runs the dispatcher's C++ cleanup (its private kwargs copy is deallocated with the thread's state
// freed: `_C.empty(shape=SeqLen())` rc 139 2/2 on acd42fa, ledger 07:32-07:40). Every P1-T binding of this module is
// therefore declared as (*args, **kwargs) -- for that signature the dispatcher copies, converts and looks up nothing
// and cannot mismatch (a method's self arrives as args[0]) -- and CoreCallArgs below matches positions and keyword
// names with C-level comparisons (PyUnicode_CompareWithASCIIString reads the string data, also for a str subclass);
// the converters accept exactly the types whose value is read without calling a Python method (int and subclasses via
// PyLong_AsLongLongAndOverflow; float and subclasses via PyFloat_AsDouble; tuple / list via the C macros; str via
// PyUnicode_AsUTF8AndSize; bytes via PyBytes_AS_STRING) and refuse everything else with TypeError naming the type.
// Same design as the add-on's tc_signal/src/py_args.h (a copy: core includes nothing from the add-on, and this order
// adds no core file). Prior art: CPython's C argument parser (Python/getargs.c: keyword matching without __eq__;
// unverified, lead to check "getargs.c find_keyword"); PyTorch 2.11 Tensor.__dlpack__'s `type(stream) is not int`
// refusal (installed torch/_tensor.py); pybind11's py::args / py::kwargs. Ours: applying it to every P1-T binding.
struct CoreParam {
  const char* name;
  bool required = true;
};

class CoreCallArgs {
 public:
  static constexpr size_t kMax = 8;
  CoreCallArgs(const char* fn, const py::args& args, const py::kwargs& kwargs, std::initializer_list<CoreParam> params,
               size_t positional_max = kMax)
      : fn_(fn) {
    for (const CoreParam& p : params) params_[n_++] = p;
    PyObject* a = args.ptr();
    const size_t n_pos = static_cast<size_t>(PyTuple_GET_SIZE(a));
    const size_t pos_ok = positional_max < n_ ? positional_max : n_;
    if (n_pos > pos_ok)
      throw py::type_error(prefix() + "takes at most " + std::to_string(pos_ok) + " positional argument(s) (" +
                           std::to_string(n_pos) + " given)");
    for (size_t i = 0; i < n_pos; ++i) v_[i] = PyTuple_GET_ITEM(a, static_cast<Py_ssize_t>(i));
    PyObject* k = kwargs.ptr();
    if (k != nullptr && PyDict_Check(k)) {
      Py_ssize_t pos = 0;
      PyObject* key = nullptr;
      PyObject* value = nullptr;
      while (PyDict_Next(k, &pos, &key, &value)) {
        if (!PyUnicode_Check(key)) throw py::type_error(prefix() + "keywords must be strings");
        size_t j = 0;
        while (j < n_ && PyUnicode_CompareWithASCIIString(key, params_[j].name) != 0) ++j;
        if (j == n_) throw py::type_error(prefix() + "got an unexpected keyword argument '" + utf8(key) + "'");
        if (v_[j] != nullptr) throw py::type_error(prefix() + "got multiple values for argument '" + params_[j].name + "'");
        v_[j] = value;
      }
    }
    for (size_t j = 0; j < n_; ++j)
      if (params_[j].required && v_[j] == nullptr)
        throw py::type_error(prefix() + "missing required argument '" + params_[j].name + "'");
  }
  bool given(size_t i) const { return v_[i] != nullptr; }
  PyObject* raw(size_t i) const { return v_[i]; }
  py::object object(size_t i) const { return v_[i] ? py::reinterpret_borrow<py::object>(v_[i]) : py::none(); }
  std::string prefix() const { return std::string(fn_) + "(): "; }
  [[noreturn]] void refuse(size_t i, const char* expected) const {
    throw py::type_error(prefix() + "argument '" + params_[i].name + "' must be " + expected + ", not " +
                         Py_TYPE(v_[i])->tp_name + " (P1T-FIX6: refused without calling any of its Python methods)");
  }
  // `self` of a Tensor method (args[0]): a Tensor (or subclass) instance, checked with PyObject_TypeCheck (C level),
  // so the cast below takes pybind11's exact / subtype path only (no implicit or foreign-module conversion).
  Tensor& tensor(size_t i) const {
    static PyTypeObject* tensor_type = reinterpret_cast<PyTypeObject*>(py::type::of<Tensor>().ptr());
    if (!PyObject_TypeCheck(v_[i], tensor_type)) refuse(i, "a tensor_cuda Tensor");
    return py::handle(v_[i]).cast<Tensor&>();
  }
  int64_t i64(size_t i) const {
    PyObject* o = v_[i];
    if (!PyLong_Check(o)) refuse(i, "an int");
    int overflow = 0;
    const long long v = PyLong_AsLongLongAndOverflow(o, &overflow);
    if (overflow != 0)
      throw std::overflow_error(prefix() + "argument '" + params_[i].name + "' does not fit in a signed 64-bit integer");
    return static_cast<int64_t>(v);
  }
  double real(size_t i) const {
    PyObject* o = v_[i];
    if (PyFloat_Check(o)) return PyFloat_AsDouble(o);
    if (PyLong_Check(o)) {
      const double v = PyLong_AsDouble(o);
      if (v == -1.0 && PyErr_Occurred()) throw py::error_already_set();  // OverflowError for a huge int
      return v;
    }
    refuse(i, "a real number (float or int)");
  }
  std::string str(size_t i) const {
    PyObject* o = v_[i];
    if (PyUnicode_Check(o)) {
      Py_ssize_t n = 0;
      const char* p = PyUnicode_AsUTF8AndSize(o, &n);
      if (p == nullptr) throw py::error_already_set();
      return std::string(p, static_cast<size_t>(n));
    }
    if (PyBytes_Check(o)) return std::string(PyBytes_AS_STRING(o), static_cast<size_t>(PyBytes_GET_SIZE(o)));
    refuse(i, "a str");
  }
  // A shape: a tuple or list of ints (items read with the C macros), at most kMaxRank checked by the caller.
  std::vector<int64_t> shape(size_t i) const {
    PyObject* o = v_[i];
    const bool is_list = PyList_Check(o);
    if (!is_list && !PyTuple_Check(o)) refuse(i, "a tuple or list of ints");
    const Py_ssize_t n = is_list ? PyList_GET_SIZE(o) : PyTuple_GET_SIZE(o);
    std::vector<int64_t> out;
    out.reserve(static_cast<size_t>(n));
    for (Py_ssize_t j = 0; j < n; ++j) {
      PyObject* e = is_list ? PyList_GET_ITEM(o, j) : PyTuple_GET_ITEM(o, j);
      if (!PyLong_Check(e))
        throw py::type_error(prefix() + "argument '" + params_[i].name + "' must contain ints, not " +
                             Py_TYPE(e)->tp_name + " (P1T-FIX6: refused without calling any of its Python methods)");
      int overflow = 0;
      const long long v = PyLong_AsLongLongAndOverflow(e, &overflow);
      if (overflow != 0)
        throw std::overflow_error(prefix() + "an extent of '" + params_[i].name +
                                  "' does not fit in a signed 64-bit integer");
      out.push_back(static_cast<int64_t>(v));
    }
    return out;
  }

 private:
  static std::string utf8(PyObject* s) {
    Py_ssize_t n = 0;
    const char* p = PyUnicode_AsUTF8AndSize(s, &n);
    if (p == nullptr) {
      PyErr_Clear();
      return "?";
    }
    return std::string(p, static_cast<size_t>(n));
  }
  const char* fn_;
  CoreParam params_[kMax] = {};
  PyObject* v_[kMax] = {};
  size_t n_ = 0;
};

// P1T-FIX6 (found by this order's own test run; sweep (g) x sweep (c) of FIX5): the exit wait is the one P1-T body that
// releases the GIL and may therefore be exited by CPython (a non-main thread that re-takes the GIL after finalization
// began: pthread_exit, a forced unwind). Any Python object that only a C++ frame on that stack owns is deallocated by
// the unwind with the thread's state freed (SIGSEGV). Through pybind11 there always is one: a (*args, **kwargs)
// binding called positionally gets a NEW empty kwargs dict (pybind11.h step 4b), and FIX5's typed binding called with a
// keyword got a PyDict_Copy (step 2). test_fix5_sweep_c_core_exit_wait_on_a_non_main_thread_never_aborts: rc -11 in 3
// of 5 runs with the (*args, **kwargs) binding. So `_dlpack_begin_shutdown` is a raw CPython function: CPython passes
// `args` / `kwargs` owned by the caller's frame, the argument is read at C level (as CoreCallArgs does), and this frame
// owns no Python object across the wait (the result dict exists only after the GIL is back). No catch-all: a forced
// unwind (not a std::exception) passes through the typed handlers. Prior art: CPython's METH_VARARGS | METH_KEYWORDS
// calling convention (caller-owned argument tuple and dict). Ours: choosing it for the GIL-releasing exit wait.
// P1T-FIX7, round-7 review finding V7-01 (MAJOR): the exit entry's argument is diagnostics only. FIX6 read it with a
// refusing reader (TypeError / OverflowError before the close), and the Python hook converted the knob with float()
// first: a knob holding 1 << 4096 raised OverflowError there and the hook skipped the close (reviewer probes
// core_knob_exit_cpu.py / core_knob_exit_gpu.py: rc 134 / -6 on 429446d). Now (same rule as the add-on's py_args.h
// raw_exit_diag_arg): the call's SHAPE is checked first (at most one positional argument, only the keyword `name`, no
// duplicate, present; a malformed call raises TypeError and does nothing, as any Python call would; the hook's call
// shape is fixed); the VALUE never prevents the close: a float or an int (subclasses included) is read at C level, no
// Python method called; anything else (None, str, objects with __float__, numpy scalars that are not float subclasses,
// an int beyond the double range) selects `fallback` and sets *default_used. Returns false only for a malformed call.
// Prior art: CPython's C argument parser (Python/getargs.c: shape errors are TypeError before the body runs). Ours:
// the rule that a diagnostic value never gates the exit close (no prior art known to me).
bool raw_exit_diag_arg(const char* fn, const char* name, PyObject* args, PyObject* kwargs, double fallback, double* out,
                       bool* default_used) {
  PyObject* v = nullptr;
  const Py_ssize_t n = args ? PyTuple_GET_SIZE(args) : 0;
  if (n > 1) {
    PyErr_Format(PyExc_TypeError, "%s(): takes at most 1 positional argument (%zd given)", fn, n);
    return false;
  }
  if (n == 1) v = PyTuple_GET_ITEM(args, 0);
  if (kwargs != nullptr && PyDict_Check(kwargs)) {
    Py_ssize_t pos = 0;
    PyObject* key = nullptr;
    PyObject* value = nullptr;
    while (PyDict_Next(kwargs, &pos, &key, &value)) {
      if (!PyUnicode_Check(key)) {
        PyErr_Format(PyExc_TypeError, "%s(): keywords must be strings", fn);
        return false;
      }
      if (PyUnicode_CompareWithASCIIString(key, name) != 0) {  // %U copies the str's data: no __str__ is called
        PyErr_Format(PyExc_TypeError, "%s(): got an unexpected keyword argument '%U'", fn, key);
        return false;
      }
      if (v != nullptr) {
        PyErr_Format(PyExc_TypeError, "%s(): got multiple values for argument '%s'", fn, name);
        return false;
      }
      v = value;
    }
  }
  if (v == nullptr) {
    PyErr_Format(PyExc_TypeError, "%s(): missing required argument '%s'", fn, name);
    return false;
  }
  *default_used = false;
  if (PyFloat_Check(v)) {  // float and subclasses: the C value, no __float__
    *out = PyFloat_AS_DOUBLE(v);
    return true;
  }
  if (PyLong_Check(v)) {  // int and subclasses (bool included): the C value, no __index__ / __float__
    const double d = PyLong_AsDouble(v);
    if (!(d == -1.0 && PyErr_Occurred())) {
      *out = d;
      return true;
    }
    PyErr_Clear();  // beyond the double range: unreadable, the default applies
  }
  *out = fallback;
  *default_used = true;
  return true;
}

PyObject* raw_dlpack_begin_shutdown(PyObject*, PyObject* args, PyObject* kwargs) {
  double first_diag_s = kDlpackShutdownDiagDefaultS;
  bool default_used = false;
  if (!raw_exit_diag_arg("_dlpack_begin_shutdown", "first_diag_s", args, kwargs, kDlpackShutdownDiagDefaultS,
                         &first_diag_s, &default_used))
    return nullptr;
  try {
    return dlpack_begin_shutdown_impl(first_diag_s, default_used, false).release().ptr();
  } catch (py::error_already_set& e) {
    e.restore();
  } catch (const std::bad_alloc&) {
    PyErr_NoMemory();
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
  }
  return nullptr;
}

const char kBeginShutdownDoc[] =
    "_dlpack_begin_shutdown(first_diag_s)\n\n"
    "P1T-FIX2 (N5) / P1T-FIX3 (V3-01), called by tensor_cuda's atexit hook: stop admitting GIL-acquiring DLPack "
    "import releases (later ones leak, counted) and wait, GIL released, with NO bound until every admitted one has "
    "finished (never returns while one is running). After first_diag_s seconds it writes a stderr line (count "
    "running, seconds waited), then another each time the time waited doubles. A producer deleter that never "
    "returns therefore makes interpreter exit wait (the producer's contract violation; the honest outcome, where a "
    "bounded return let finalization abort the process). Returns {in_flight_at_close, still_in_flight, timed_out, "
    "waited_us, diagnostics, own_thread_in_flight, others_in_flight_at_close, first_diag_s, first_diag_default_used}. "
    "P1T-FIX6: a raw CPython function (no pybind11 dispatcher frame on the stack while it waits). P1T-FIX7 (V7-01): "
    "first_diag_s is diagnostics only and never prevents the close: a float or an int (subclasses included) is read "
    "without calling its methods; any other value, or an int beyond the double range, uses the default "
    "(_DLPACK_SHUTDOWN_DIAG_DEFAULT_S) and the call still closes and waits (first_diag_default_used=True). Only a "
    "malformed call (wrong arity or keyword) raises TypeError, doing nothing. P1T-FIX7 (lead send-back L1): if "
    "locking the release lifecycle mutex fails (never observed), the close retries it and that retry wait is "
    "diagnosed like the release wait (first line after first_diag_s, then each time the retrying time doubles; "
    "_dlpack_release_stats() shutdown_lock_retries / shutdown_lock_diagnostics); diagnostics counts those lines too.";
PyMethodDef kBeginShutdownDef = {"_dlpack_begin_shutdown",
                                 reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)(void)>(&raw_dlpack_begin_shutdown)),
                                 METH_VARARGS | METH_KEYWORDS, kBeginShutdownDoc};

// P1T-FIX6 (FIX5 residual R7): raises the recorded CUDA errors of the P1-T core paths that could not raise where they
// happened (an imported Storage's drain or its device restore, fill_'s guard unwinding; FIX5 sweep (d) recorded them,
// but only _dlpack_release_stats() showed them: a CUDA failure was reported nowhere). tensor_cuda.synchronize() calls
// this after the engine sync, and tensor_cuda's exit hook logs it. Same discipline as tc_signal.synchronize() (V6-03):
// the record is consumed only once the RuntimeError naming it exists and is captured, so a MemoryError while reporting
// loses nothing. Prior art: tc_signal.synchronize()'s N1/V6-03 raise-once rule (this repository); PyTorch's
// C10_CUDA_CHECK_WARN (an error in a destructor is reported, never dropped). Ours: deferring it to the next synchronize.
// P1T-FIX7 (sweep (j)): the text no longer starts "tensor_cuda.synchronize:" (the exit hook logged it as
// "tensor_cuda: at exit: tensor_cuda.synchronize: ...", naming a call that did not happen at exit); it names the
// reporting paths instead, and each record's site string names the CUDA call that failed (kernels.cu), followed by
// the device it concerned.
std::string unraisable_message(unsigned long long n, const std::string& what, int dev, int err) {
  std::string msg = "tensor_cuda: " + std::to_string(n) +
                    " CUDA failure(s) on DLPack release / fill_ paths that could not raise where they happened (P1-T "
                    "core; reported once, by tensor_cuda.synchronize() or at interpreter exit), the first: ";
  if (what.empty())
    msg += "(not named: recorded while an earlier report was being raised)";
  else
    msg += what + " [CUDA device " + std::to_string(dev) + ": " + cudaGetErrorName(static_cast<cudaError_t>(err)) +
           ": " + cudaGetErrorString(static_cast<cudaError_t>(err)) + "]";
  return msg;
}

// P1T-FIX8, round-8 review finding V8-02 (MINOR), sweep (l) "the last reader at exit": the core exit runner is the last
// reader of the R7 record in every case. What _dlpack_raise_unraisable RAISES inside the runner (the hook's R7 logging)
// is acknowledged only when the hook says it wrote the line (_dlpack_exit_report_written); until then a plain-data copy
// (count, the static site text, device, error) is kept here, and the runner writes it with C stdio if the hook could
// not (MemoryError building the line, stderr None or raising, the hook interrupted). GIL held for every access (the
// runner, the hook and the raise all hold it). Prior art: allocation-free last-chance reporting (async-signal-safe
// logging with POSIX write(2); SQLite's OOM-testing discipline); ours: the acknowledged exit report.
struct CoreExitReport {
  bool in_runner = false;  // the exit runner is running (its window)
  bool unwritten = false;
  unsigned long long count = 0;
  const char* what = nullptr;  // a static site string, or nullptr (not named)
  int device = -1;
  int error = 0;
};
CoreExitReport g_core_exit_report;
std::atomic<unsigned long long> g_shutdown_log_fallbacks{0};   // exit log lines written by the allocation-free path
std::atomic<bool> g_test_fail_exit_report{false};              // TEST HOOK (P1T-FIX8 V8-02 tests): one-shot

void dlpack_raise_unraisable() {
  unsigned long long n = 0;
  std::string what;
  int dev = -1, err = 0;
  const char* raw_what = nullptr;  // P1T-FIX8: the static site text, for the exit report
  unsigned long long raw_n = 0;
  int raw_dev = -1, raw_err = 0;
  dlpack_unreported_unraisable_raw(&raw_n, &raw_what, &raw_dev, &raw_err);
  dlpack_unreported_unraisable(&n, &what, &dev, &err);
  if (n == 0) return;
  const std::string msg = unraisable_message(n, what, dev, err);
  PyErr_SetString(PyExc_RuntimeError, msg.c_str());
  if (!PyErr_ExceptionMatches(PyExc_RuntimeError)) {
    PyErr_Clear();
    throw std::bad_alloc();  // -> MemoryError; the record stays for the next call
  }
  try {
    py::error_already_set raised;  // captures the RuntimeError (a bad_alloc here consumes nothing)
    if (g_core_exit_report.in_runner) {  // P1T-FIX8 (V8-02): kept until the hook says it wrote it
      if (!g_core_exit_report.unwritten) {
        g_core_exit_report.what = raw_what;
        g_core_exit_report.device = raw_dev;
        g_core_exit_report.error = raw_err;
      }
      const unsigned long long c = g_core_exit_report.count;
      g_core_exit_report.count = (n > ~0ULL - c) ? ~0ULL : c + n;
      g_core_exit_report.unwritten = true;
    }
    dlpack_ack_unraisable(n);      // noexcept: consumed exactly once, now that the diagnostic exists
    throw raised;
  } catch (const std::bad_alloc&) {
    PyErr_Clear();
    throw;
  }
}

// P1T-FIX7 (V7-01 sweep (i), the guarantees a hook gives on the way out): what atexit calls, as
// atexit.register(_dlpack_exit_hook_runner, _dlpack_shutdown_hook). The hook reaches its close on every path that
// runs any of its statements. An exception that escapes it before its close (a KeyboardInterrupt delivered at its first
// bytecode: CPython 3.12 checks for pending signals in the RESUME that starts every Python frame) is handled here:
// close and wait with the default delay (a no-op wait when the hook had already closed), write the recorded P1-T
// failures nobody reported (C stdio, no Python: the hook's own logging did not run), then hand the hook's exception to
// atexit unchanged. When the hook RETURNS, nothing more happens: a hook that returns without closing is a hook defect
// its tests and the round-7 probes must see, not one this runner hides. Raw METH_O: no pybind11 frame, no catch-all.
// Prior art: CPython's atexit (an exception in one callback is printed, the rest still run); Python's try / finally,
// which holds inside a frame but not at its first instruction. Ours: the native wrapper for that one boundary.
// P1T-FIX8, round-8 review finding V8-02 (MINOR), sweep (l): the allocation-free line for a record count and its first
// record (static site text, device, cudaGetErrorName's static name: CUDA 12.6 Runtime API; unverified here, lead to
// check "cudaGetErrorName returns static string"). C stdio only; a closed stderr makes it fail quietly, never crash.
void write_core_summary(const char* head, const char* why, unsigned long long n, const char* what, int dev,
                        int err) noexcept {
  if (what != nullptr)
    std::fprintf(stderr, "%s %llu CUDA failure(s) on DLPack release / fill_ paths (P1-T core; %s; allocation-free "
                 "summary), the first: %s [CUDA device %d: %s]\n",
                 head, n, why, what, dev, cudaGetErrorName(static_cast<cudaError_t>(err)));
  else
    std::fprintf(stderr, "%s %llu CUDA failure(s) on DLPack release / fill_ paths (P1-T core; %s; allocation-free "
                 "summary), the first not named (recorded while an earlier report was being raised)\n",
                 head, n, why);
  std::fflush(stderr);
  g_shutdown_log_fallbacks.fetch_add(1);
}

// P1T-FIX8 (V8-02): the runner's last reading, after the hook (returned or raised) and after quiescence. (1) The R7
// records nobody raised: the full message (unraisable_message), or, when it cannot allocate (FIX7 caught bad_alloc and
// wrote nothing), the allocation-free summary; consumed either way (written once: there is no later reader). (2) A
// report the hook's _dlpack_raise_unraisable raised here but the hook did not acknowledge writing: its summary.
void core_exit_last_reader(bool hook_raised) noexcept {
  const char* head_tail = hook_raised ? "(logged by the exit runner: the exit hook did not finish)"
                                      : "(logged by the exit runner: the exit hook did not report it)";
  unsigned long long n = 0;
  const char* what = nullptr;
  int dev = -1, err = 0;
  dlpack_unreported_unraisable_raw(&n, &what, &dev, &err);  // no allocation
  if (n > 0) {
    bool written = false;
    try {
      if (g_test_fail_exit_report.exchange(false)) throw std::bad_alloc();  // TEST HOOK (P1T-FIX8 V8-02 tests)
      const std::string msg = unraisable_message(n, what != nullptr ? std::string(what) : std::string(), dev, err);
      std::fprintf(stderr, "%s %s\n", msg.c_str(), head_tail);
      std::fflush(stderr);
      written = true;
    } catch (const std::exception&) {
      // bad_alloc building the message: the allocation-free summary below (V8-02).
    }
    if (!written)
      write_core_summary("tensor_cuda: at interpreter exit,", "the full report could not be built: it failed to allocate",
                         n, what, dev, err);
    dlpack_ack_unraisable(n);
  }
  if (g_core_exit_report.unwritten) {
    write_core_summary("tensor_cuda: at interpreter exit, a report of", "raised at exit but not written by the exit "
                       "hook; written here by the exit runner, its last reader",
                       g_core_exit_report.count, g_core_exit_report.what, g_core_exit_report.device,
                       g_core_exit_report.error);
    g_core_exit_report.unwritten = false;
    g_core_exit_report.count = 0;
  }
}

struct CoreExitRunnerWindow {
  CoreExitRunnerWindow() noexcept { g_core_exit_report.in_runner = true; }
  ~CoreExitRunnerWindow() { g_core_exit_report.in_runner = false; }
  CoreExitRunnerWindow(const CoreExitRunnerWindow&) = delete;
  CoreExitRunnerWindow& operator=(const CoreExitRunnerWindow&) = delete;
};

PyObject* raw_dlpack_exit_hook_runner(PyObject*, PyObject* hook) {
  const CoreExitRunnerWindow window;  // P1T-FIX8 (V8-02): what is raised in here is kept until written
  PyObject* r = PyObject_CallNoArgs(hook);
  if (r != nullptr) {
    core_exit_last_reader(false);  // P1T-FIX8 (V8-02): the last reader also when the hook returned
    return r;
  }
  PyObject *type = nullptr, *value = nullptr, *tb = nullptr;
  PyErr_Fetch(&type, &value, &tb);  // keep the hook's exception
  try {
    const py::dict d = dlpack_begin_shutdown_impl(kDlpackShutdownDiagDefaultS, false, true);
    const long long others = d["others_in_flight_at_close"].cast<long long>();
    if (others > 0) {
      std::fprintf(stderr,
                   "tensor_cuda: the DLPack exit hook raised before it closed release admission; the native exit "
                   "runner closed it and waited %.1f s for %lld release(s) admitted on other threads\n",
                   d["waited_us"].cast<long long>() / 1e6, others);
      std::fflush(stderr);
    }
  } catch (py::error_already_set& e) {
    e.discard_as_unraisable("tensor_cuda exit runner");  // the close and the wait had already happened
  } catch (const std::exception&) {
    // bad_alloc building the result: the close and the wait had already happened.
  }
  core_exit_last_reader(true);  // P1T-FIX8 (V8-02): after quiescence; the hook's exception is restored below
  PyErr_Restore(type, value, tb);
  return nullptr;
}

// P1T-FIX8 (V8-02): the hook's acknowledgement that it wrote what _dlpack_raise_unraisable raised at exit. Raw
// METH_NOARGS. Never raises.
PyObject* raw_dlpack_exit_report_written(PyObject*, PyObject*) {
  g_core_exit_report.unwritten = false;
  g_core_exit_report.count = 0;
  Py_RETURN_NONE;
}

// P1T-FIX8 (V8-02 regression tests only; not an API): the exit runner's next full report fails as if it could not
// allocate, so its allocation-free summary is written instead. Raw METH_NOARGS.
PyObject* raw_dlpack_test_fail_exit_report(PyObject*, PyObject*) {
  g_test_fail_exit_report.store(true);
  Py_RETURN_NONE;
}

const char kExitRunnerDoc[] =
    "_dlpack_exit_hook_runner(hook)\n\n"
    "P1T-FIX7 (V7-01): what tensor_cuda registers with atexit, as atexit.register(_dlpack_exit_hook_runner, "
    "_dlpack_shutdown_hook). Calls hook(); if the hook raises (for example a KeyboardInterrupt delivered at its "
    "first bytecode), closes DLPack release admission and waits itself (default diagnostic delay), writes any "
    "recorded P1-T CUDA failure nobody reported to stderr, then re-raises the hook's exception. Does nothing more "
    "about closing when the hook returns. P1T-FIX8 (V8-02): in every case it is the last reader of the recorded "
    "P1-T failures: after the hook and after the close and the wait, it writes those nobody raised, and a report the "
    "hook's R7 logging raised but did not acknowledge writing (_dlpack_exit_report_written), with an "
    "allocation-free summary (count, first site, device, error name) when the full message cannot be built. Remove "
    "it with atexit.unregister(tensor_cuda._C._dlpack_exit_hook_runner).";
PyMethodDef kExitReportWrittenDef = {"_dlpack_exit_report_written", &raw_dlpack_exit_report_written, METH_NOARGS,
                                     "_dlpack_exit_report_written()\n\nP1T-FIX8 (V8-02): called by tensor_cuda's "
                                     "exit hook once its lines are written; not an API for other callers."};
PyMethodDef kTestFailExitReportDef = {"_dlpack_test_fail_exit_report", &raw_dlpack_test_fail_exit_report, METH_NOARGS,
                                      "_dlpack_test_fail_exit_report()\n\nTEST HOOK (P1T-FIX8 V8-02 tests only): "
                                      "the exit runner's next full report fails as if it could not allocate."};
PyMethodDef kExitRunnerDef = {"_dlpack_exit_hook_runner", reinterpret_cast<PyCFunction>(&raw_dlpack_exit_hook_runner),
                              METH_O, kExitRunnerDoc};

}  // namespace

PYBIND11_MODULE(_tensor_cuda, m) {
  m.doc() = "Project Tensor — standalone CUDA tensor engine (no PyTorch).";

  py::class_<Tensor>(m, "Tensor")
      .def_property_readonly("shape", [](Tensor& t) {
        return py::tuple(py::cast(t.shape()));
      })
      .def_property_readonly("ndim", [](Tensor& t) { return t.ndim(); })
      .def_property_readonly("dtype", [](Tensor& t) { return std::string(dtype_name(t.dtype())); })
      .def_property_readonly("device", [](Tensor& t) { return t.device().str(); })
      .def_property("requires_grad",
                    [](Tensor& t) { return t.requires_grad(); },
                    [](Tensor& t, bool r) { t.set_requires_grad(r); })
      .def_property_readonly("grad", &grad_of)
      .def("numpy", &tensor_to_numpy)
      // P1T-M1: DLPack interop (see the DLPack section above for prior art).
      // P1T-FIX6 (V6-02, sweep (g)): every P1-T method takes (*args, **kwargs) and converts with CoreCallArgs (C-level
      // only; design above), so no Python method of an argument runs in the dispatcher or here.
      .def("__dlpack__",
           [](py::args a, py::kwargs k) {
             const CoreCallArgs c("__dlpack__", a, k,
                                  {{"self"}, {"stream", false}, {"max_version", false}, {"dl_device", false},
                                   {"copy", false}},
                                  /*positional_max=*/1);
             return tensor_dlpack(c.tensor(0), c.object(1), c.object(2), c.object(3), c.object(4));
           },
           (std::string("__dlpack__(self, *, stream=None, max_version=None, dl_device=None, copy=None)\n\n") +
            kDlpackDoc)
               .c_str())
      .def("__dlpack_device__",
           [](py::args a, py::kwargs k) {
             const CoreCallArgs c("__dlpack_device__", a, k, {{"self"}});
             return tensor_dlpack_device(c.tensor(0));
           },
           "__dlpack_device__(self)\n\n(2, device_ordinal) for a CUDA tensor (kDLCUDA = 2); BufferError on CPU.")
      .def("data_ptr",
           [](py::args a, py::kwargs k) -> uintptr_t {
             const CoreCallArgs c("data_ptr", a, k, {{"self"}});
             Tensor& t = c.tensor(0);
             if (!t.defined() || !t.data().defined()) return 0;
             return reinterpret_cast<uintptr_t>(t.data().data_ptr());
           },
           "data_ptr(self)\n\nDevice address of element 0, as an int (storage base + offset).")
      .def("fill_",
           [](py::args a, py::kwargs k) {
             const CoreCallArgs c("fill_", a, k, {{"self"}, {"value"}});
             Tensor& t = c.tensor(0);
             const double value = c.real(1);
             tc::fill_(t.data(), value);
             return c.object(0);
           },
           "fill_(self, value)\n\n"
           "Fill in place with `value`, queued on the legacy default stream "
           "(asynchronous; existing fill_kernel). float32/float16/bfloat16 CUDA "
           "tensors only. Bumps the storage revision. Not recorded by autograd. "
           "Returns self. P1T-FIX4: launched on the tensor's own device (proven "
           "against its allocation; the caller's device is switched to it and "
           "restored), and a CUDA error already pending in the calling thread is "
           "raised as such before the launch (RuntimeError, nothing launched). "
           "P1T-FIX5: an extent whose one-thread-per-element launch would need "
           "more than 2^31-1 blocks of 256 threads (the CUDA grid x-dimension "
           "limit), i.e. more than 549755813632 elements, raises ValueError before "
           "any launch; nothing is written and the revision is unchanged. "
           "P1T-FIX6: `value` must be a float or an int (subclasses included); any "
           "other type raises TypeError without any of its methods being called.")
      .def("mark_modified",
           [](py::args a, py::kwargs k) {
             const CoreCallArgs c("mark_modified", a, k, {{"self"}});
             c.tensor(0).data().mark_modified();
           },
           "mark_modified(self)\n\n"
           "Bump the Storage revision. Call after an external (non-TensorCUDA) "
           "in-place write, e.g. through DLPack or data_ptr(); see __dlpack__.")
      .def_property_readonly("revision",
           py::cpp_function([](py::args a, py::kwargs k) {
             const CoreCallArgs c("revision", a, k, {{"self"}});
             return c.tensor(0).data().revision();
           }),
           "Storage revision (read-only): bumped by TensorCUDA in-place ops and mark_modified().")
      .def("backward", [](Tensor& t) { t.backward(); })
      .def("zero_grad", &Tensor::zero_grad)
      .def("reshape", [](Tensor& t, std::vector<int64_t> s) { return ops::reshape(t, s); })
      .def("transpose_last", [](Tensor& t) { return ops::transpose_last(t); })
      .def("sum", [](Tensor& t, std::vector<int> axes, bool keepdim) { return ops::sum(t, axes, keepdim); },
           py::arg("axes") = std::vector<int>{}, py::arg("keepdim") = false)
      .def("mean", [](Tensor& t, std::vector<int> axes, bool keepdim) { return ops::mean(t, axes, keepdim); },
           py::arg("axes") = std::vector<int>{}, py::arg("keepdim") = false)
      .def("relu", [](Tensor& t) { return ops::relu(t); })
      .def("gelu", [](Tensor& t) { return ops::gelu(t); })
      .def("gelu_exact", [](Tensor& t) { return ops::gelu_exact(t); })
      .def("silu", [](Tensor& t) { return ops::silu(t); })
      .def("abs", [](Tensor& t) { return ops::abs(t); })
      .def("detach", [](Tensor& t) { return ops::detach(t); })
      .def("half", [](Tensor& t) { return ops::cast(t, DType::Float16); })
      .def("float", [](Tensor& t) { return ops::cast(t, DType::Float32); })
      .def("astype", [](Tensor& t, const std::string& dt) {
        return ops::cast(t, dtype_from_string(dt));
      })
      .def("sin", [](Tensor& t) { return ops::sin(t); })
      .def("cos", [](Tensor& t) { return ops::cos(t); })
      .def("tan", [](Tensor& t) { return ops::tan(t); })
      .def("asin", [](Tensor& t) { return ops::asin(t); })
      .def("acos", [](Tensor& t) { return ops::acos(t); })
      .def("atan", [](Tensor& t) { return ops::atan(t); })
      .def("sinh", [](Tensor& t) { return ops::sinh(t); })
      .def("cosh", [](Tensor& t) { return ops::cosh(t); })
      .def("log2", [](Tensor& t) { return ops::log2(t); })
      .def("log10", [](Tensor& t) { return ops::log10(t); })
      .def("sign", [](Tensor& t) { return ops::sign(t); })
      .def("floor", [](Tensor& t) { return ops::floor(t); })
      .def("ceil", [](Tensor& t) { return ops::ceil(t); })
      .def("round", [](Tensor& t) { return ops::round(t); })
      .def("isnan", [](Tensor& t) { return ops::isnan(t); })
      .def("isinf", [](Tensor& t) { return ops::isinf(t); })
      .def("isfinite", [](Tensor& t) { return ops::isfinite(t); })
      .def("nan_to_num", [](Tensor& t, double n, double p, double m) { return ops::nan_to_num(t, n, p, m); },
           py::arg("nan") = 0.0, py::arg("posinf") = 1e30, py::arg("neginf") = -1e30)
      .def("reciprocal", [](Tensor& t) { return ops::reciprocal(t); })
      .def("clamp", [](Tensor& t, double lo, double hi) { return ops::clamp(t, lo, hi); })
      .def("maximum", [](Tensor& a, Tensor& b) { return ops::maximum(a, b); })
      .def("minimum", [](Tensor& a, Tensor& b) { return ops::minimum(a, b); })
      .def("slice", [](Tensor& t, int dim, int64_t start, int64_t len) { return ops::slice(t, dim, start, len); })
      .def("__getitem__", &getitem)
      .def("sigmoid", [](Tensor& t) { return ops::sigmoid(t); })
      .def("tanh", [](Tensor& t) { return ops::tanh(t); })
      .def("exp", [](Tensor& t) { return ops::exp(t); })
      .def("log", [](Tensor& t) { return ops::log(t); })
      .def("sqrt", [](Tensor& t) { return ops::sqrt(t); })
      .def("softmax", [](Tensor& t, int axis) { return ops::softmax(t, axis); }, py::arg("axis") = -1)
      .def("log_softmax", [](Tensor& t, int axis) { return ops::log_softmax(t, axis); }, py::arg("axis") = -1)
      .def("pow", [](Tensor& t, double p) { return ops::pow_scalar(t, p); })
      .def("max", [](Tensor& t, std::vector<int> ax, bool kd) { return ops::max(t, ax, kd); }, py::arg("axes") = std::vector<int>{}, py::arg("keepdim") = false)
      .def("min", [](Tensor& t, std::vector<int> ax, bool kd) { return ops::min(t, ax, kd); }, py::arg("axes") = std::vector<int>{}, py::arg("keepdim") = false)
      .def("var", [](Tensor& t, std::vector<int> ax, bool kd) { return ops::var(t, ax, kd); }, py::arg("axes") = std::vector<int>{}, py::arg("keepdim") = false)
      .def("std", [](Tensor& t, std::vector<int> ax, bool kd) { return ops::std(t, ax, kd); }, py::arg("axes") = std::vector<int>{}, py::arg("keepdim") = false)
      .def("prod", [](Tensor& t, std::vector<int> ax, bool kd) { return ops::prod(t, ax, kd); }, py::arg("axes") = std::vector<int>{}, py::arg("keepdim") = false)
      .def("argmax", [](Tensor& t, int ax) { return ops::argmax(t, ax); }, py::arg("axis") = -1)
      .def("argmin", [](Tensor& t, int ax) { return ops::argmin(t, ax); }, py::arg("axis") = -1)
      .def("argmax_last_axis", [](Tensor& t) { return ops::argmax_last_axis(t); })
      .def("cumsum", [](Tensor& t, int ax) { return ops::cumsum(t, ax); }, py::arg("axis") = -1)
      .def("gather", [](Tensor& t, int dim, Tensor& idx) { return ops::gather(t, dim, idx); })
      .def("flip", [](Tensor& t, std::vector<int> dims) { return ops::flip(t, dims); })
      .def("topk", [](Tensor& t, int k, bool largest) {
        auto pr = ops::topk(t, k, largest);
        return py::make_tuple(std::get<0>(pr), std::get<1>(pr));
      }, py::arg("k"), py::arg("largest") = true)
      .def("sort", [](Tensor& t, bool descending) {
        auto pr = ops::topk(t, t.shape().back(), descending);
        return py::make_tuple(std::get<0>(pr), std::get<1>(pr));
      }, py::arg("descending") = false)
      .def("permute", [](Tensor& t, std::vector<int> d) { return ops::permute(t, d); })
      .def("transpose", [](Tensor& t, int a, int b) { return ops::transpose(t, a, b); })
      .def("squeeze", [](Tensor& t, int d) { return ops::squeeze(t, d); }, py::arg("dim") = 0)
      .def("unsqueeze", [](Tensor& t, int d) { return ops::unsqueeze(t, d); })
      .def("expand", [](Tensor& t, std::vector<int64_t> s) { return ops::expand(t, s); })
      .def("flatten", [](Tensor& t, int s, int e) { return ops::flatten(t, s, e); }, py::arg("start_dim") = 0, py::arg("end_dim") = -1)
      .def("masked_fill", [](Tensor& t, Tensor& m, double v) { return ops::masked_fill(t, m, v); })
      .def("gt", [](Tensor& a, Tensor& b) { return ops::compare(a, b, 0); })
      .def("ge", [](Tensor& a, Tensor& b) { return ops::compare(a, b, 1); })
      .def("lt", [](Tensor& a, Tensor& b) { return ops::compare(a, b, 2); })
      .def("le", [](Tensor& a, Tensor& b) { return ops::compare(a, b, 3); })
      .def("eq", [](Tensor& a, Tensor& b) { return ops::compare(a, b, 4); })
      .def("__gt__", [](Tensor& a, double s) { return ops::compare_scalar(a, s, 0); })
      .def("__ge__", [](Tensor& a, double s) { return ops::compare_scalar(a, s, 1); })
      .def("__lt__", [](Tensor& a, double s) { return ops::compare_scalar(a, s, 2); })
      .def("__le__", [](Tensor& a, double s) { return ops::compare_scalar(a, s, 3); })
      .def("__pow__", [](Tensor& a, double p) { return ops::pow_scalar(a, p); })
      // operator overloads (Tensor op Tensor and Tensor op scalar)
      .def("__add__", [](Tensor& a, Tensor& b) { return ops::add(a, b); })
      .def("__add__", [](Tensor& a, double s) { return ops::add_scalar(a, s); })
      .def("__radd__", [](Tensor& a, double s) { return ops::add_scalar(a, s); })
      .def("__sub__", [](Tensor& a, Tensor& b) { return ops::sub(a, b); })
      .def("__sub__", [](Tensor& a, double s) { return ops::add_scalar(a, -s); })
      .def("__mul__", [](Tensor& a, Tensor& b) { return ops::mul(a, b); })
      .def("__mul__", [](Tensor& a, double s) { return ops::mul_scalar(a, s); })
      .def("__rmul__", [](Tensor& a, double s) { return ops::mul_scalar(a, s); })
      .def("__truediv__", [](Tensor& a, Tensor& b) { return ops::div(a, b); })
      .def("__truediv__", [](Tensor& a, double s) { return ops::mul_scalar(a, 1.0 / s); })
      .def("__matmul__", [](Tensor& a, Tensor& b) { return ops::matmul(a, b); })
      .def("__neg__", [](Tensor& a) { return ops::neg(a); });

  py::class_<ApaInt4Workspace, std::shared_ptr<ApaInt4Workspace>>(
      m, "ApaInt4Workspace")
      .def_property_readonly("capacity",
          [](const ApaInt4Workspace& w) { return w.capacity; })
      .def_property_readonly("valid_rows",
          [](const ApaInt4Workspace& w) { return w.valid_rows; })
      .def_property_readonly("nbytes", [](const ApaInt4Workspace& w) {
        return (int64_t)(w.codes.storage->nbytes + w.scales.storage->nbytes);
      })
      .def("reset", &apa_int4_workspace_reset,
           "Invalidate all derived rows after a non-append K mutation.")
      .def("_rewind", [](ApaInt4Workspace& w, int64_t valid_rows) {
        if (valid_rows < 0 || valid_rows > w.valid_rows)
          throw std::runtime_error("workspace rewind must decrease valid_rows");
        w.valid_rows = valid_rows;
      }, py::arg("valid_rows"));

  // factory
  m.def("tensor", &tensor_from_numpy, py::arg("array"), py::arg("device") = "cuda",
        py::arg("requires_grad") = false);

  // free-function ops
  m.def("matmul", &ops::matmul, py::arg("a"), py::arg("b"),
        py::arg("alpha") = 1.f, py::arg("trans_b") = false);
  m.def("rms_norm", &ops::rms_norm, py::arg("x"), py::arg("w"), py::arg("eps"));
  m.def("rope_apply", &ops::rope_apply, py::arg("x"), py::arg("cos"),
        py::arg("sin"), py::arg("pos0"), py::arg("inverse") = false,
        py::arg("pair_swap") = false);
  m.def("write_rows", &ops::write_rows, py::arg("buf"), py::arg("src"),
        py::arg("start"));
  m.def("export_rows", &ops::export_rows, py::arg("cache"),
        py::arg("dim"), py::arg("start"), py::arg("len"));
  m.def("export_rope_rows", &ops::export_rope_rows, py::arg("cache"),
        py::arg("cos"), py::arg("sin"), py::arg("dim"),
        py::arg("start"), py::arg("len"), py::arg("pos0"),
        py::arg("inverse") = false, py::arg("pair_swap") = false);
  m.def("export_row_pair", &ops::export_row_pair, py::arg("raw_cache"),
        py::arg("rope_cache"), py::arg("cos"), py::arg("sin"),
        py::arg("raw_dim"), py::arg("rope_dim"), py::arg("raw_start"),
        py::arg("rope_start"), py::arg("len"), py::arg("pos0"),
        py::arg("inverse") = false, py::arg("pair_swap") = false);
  m.def("export_row_pairs", &ops::export_row_pairs, py::arg("raw_caches"),
        py::arg("rope_caches"), py::arg("cos"), py::arg("sin"),
        py::arg("raw_dim"), py::arg("rope_dim"), py::arg("raw_starts"),
        py::arg("rope_starts"), py::arg("len"), py::arg("pos0"),
        py::arg("inverse") = false, py::arg("pair_swap") = false);
  m.def("swap_row_pairs_with_rope", &ops::swap_row_pairs_with_rope,
        py::arg("raw_caches"), py::arg("rope_caches"),
        py::arg("raw_inserts"), py::arg("rope_inserts"), py::arg("cos"),
        py::arg("sin"), py::arg("raw_dim"), py::arg("rope_dim"),
        py::arg("head_tokens"), py::arg("tail_start"), py::arg("pos0"),
        py::arg("pair_swap") = false);
  m.def("evict_row_pairs", &ops::evict_row_pairs, py::arg("raw_caches"),
        py::arg("rope_caches"), py::arg("raw_dim"), py::arg("rope_dim"),
        py::arg("head_tokens"), py::arg("drop_tokens"));
  m.def("arena_row_pair_transaction", &ops::arena_row_pair_transaction,
        py::arg("raw_caches"), py::arg("rope_caches"),
        py::arg("raw_inserts"), py::arg("rope_inserts"), py::arg("cos"),
        py::arg("sin"), py::arg("raw_dim"), py::arg("rope_dim"),
        py::arg("sink_tokens"), py::arg("current_mount_tokens"),
        py::arg("arena_width"), py::arg("pair_swap") = false);
  m.def("splice_rows", &ops::splice_rows, py::arg("old_cache"),
        py::arg("insert"), py::arg("dim"), py::arg("head_tokens"),
        py::arg("tail_start"));
  m.def("evict_rows", &ops::evict_rows, py::arg("old_cache"),
        py::arg("dim"), py::arg("head_tokens"), py::arg("drop_tokens"));
  m.def("dda_raycast",
        [](Tensor& grid, Tensor& origins, Tensor& directions, int max_steps) {
          auto out = ops::dda_raycast(grid, origins, directions, max_steps);
          return py::make_tuple(
              std::get<0>(out), std::get<1>(out), std::get<2>(out),
              std::get<3>(out), std::get<4>(out), std::get<5>(out));
        },
        py::arg("grid_u8"), py::arg("origins_f32"),
        py::arg("directions_f32"), py::arg("max_steps"));
  m.def("raster_winner_scatter_min",
        [](Tensor& pixel_indices, Tensor& depth_keys, Tensor& face_ids,
           int64_t pixel_count, int threads) {
          auto out = ops::raster_winner_scatter_min(
              pixel_indices, depth_keys, face_ids, pixel_count, threads);
          return py::make_tuple(std::get<0>(out), std::get<1>(out));
        },
        py::arg("pixel_indices_i64"), py::arg("depth_keys_i64"),
        py::arg("face_ids_i64"), py::arg("pixel_count"),
        py::arg("threads") = 256);
  m.def("raster_triangle_winners",
        [](Tensor& clip_positions, Tensor& faces, int height, int width,
           int face_threads) {
          auto out = ops::raster_triangle_winners(
              clip_positions, faces, height, width, face_threads);
          return py::make_tuple(std::get<0>(out), std::get<1>(out));
        },
        py::arg("clip_positions_f32"), py::arg("faces_i64"),
        py::arg("height"), py::arg("width"),
        py::arg("face_threads") = 256);
  m.def("raster_winner_resolve",
        [](Tensor& clip_positions, Tensor& faces, Tensor& winner_face_ids,
           int pixel_threads) {
          auto out = ops::raster_winner_resolve(
              clip_positions, faces, winner_face_ids, pixel_threads);
          return py::make_tuple(std::get<0>(out), std::get<1>(out));
        },
        py::arg("clip_positions_f32"), py::arg("faces_i64"),
        py::arg("winner_face_i64"), py::arg("pixel_threads") = 256);
  m.def("rasterize_clip",
        [](Tensor& clip_positions, Tensor& faces, int height, int width,
           int face_threads, int pixel_threads) {
          auto out = ops::rasterize_clip(
              clip_positions, faces, height, width, face_threads,
              pixel_threads);
          return py::make_tuple(std::get<0>(out), std::get<1>(out),
                                std::get<2>(out));
        },
        py::arg("clip_positions_f32"), py::arg("faces_i64"),
        py::arg("height"), py::arg("width"),
        py::arg("face_threads") = 256, py::arg("pixel_threads") = 256);
  m.def("bake_back_project",
        [](Tensor& atlas_positions_h, Tensor& view, Tensor& view_depth,
           Tensor& view_reliable, Tensor& view_cosine,
           Tensor& world_to_camera, Tensor& image_projection,
           float depth_threshold, int threads) {
          auto out = ops::bake_back_project(
              atlas_positions_h, view, view_depth, view_reliable,
              view_cosine, world_to_camera, image_projection,
              depth_threshold, threads);
          return py::make_tuple(std::get<0>(out), std::get<1>(out),
                                std::get<2>(out), std::get<3>(out));
        },
        py::arg("atlas_positions_h_f32"), py::arg("view_f32"),
        py::arg("view_depth_f32"), py::arg("view_reliable_u8"),
        py::arg("view_cosine_f32"), py::arg("world_to_camera_f32"),
        py::arg("image_projection_f32"), py::arg("depth_threshold"),
        py::arg("threads") = 256);
  m.def("bake_cosine_blend",
        [](Tensor& view_colors, Tensor& view_cosine, Tensor& view_valid,
           Tensor& view_weights, Tensor& view_enabled, int threads) {
          auto out = ops::bake_cosine_blend(
              view_colors, view_cosine, view_valid, view_weights,
              view_enabled, threads);
          return py::make_tuple(std::get<0>(out), std::get<1>(out),
                                std::get<2>(out));
        },
        py::arg("view_colors_f32"), py::arg("view_cosine_f32"),
        py::arg("view_valid_u8"), py::arg("view_weights_f32"),
        py::arg("view_enabled_u8"), py::arg("threads") = 256);
  m.def("inpaint_island_passes",
        [](Tensor& positions, Tensor& vertex_colors, Tensor& vertex_mask,
           Tensor& neighbor_offsets, Tensor& neighbors,
           Tensor& island_offsets, Tensor& island_occurrences,
           int pass_count_cap, int threads) {
          auto out = ops::inpaint_island_passes(
              positions, vertex_colors, vertex_mask, neighbor_offsets,
              neighbors, island_offsets, island_occurrences,
              pass_count_cap, threads);
          return py::make_tuple(std::get<0>(out), std::get<1>(out),
                                std::get<2>(out));
        },
        py::arg("positions_f32"), py::arg("vertex_colors_f32"),
        py::arg("vertex_mask_f32"), py::arg("neighbor_offsets_i64"),
        py::arg("neighbors_i64"), py::arg("island_offsets_i64"),
        py::arg("island_occurrences_i64"), py::arg("pass_count_cap"),
        py::arg("threads") = 128);
  m.def("terrain_render",
        [](Tensor& materials, Tensor& palette,
           const std::vector<float>& position,
           const std::vector<float>& forward,
           const std::vector<float>& right, const std::vector<float>& up,
           float half_width, float half_height, int width, int height,
           const std::vector<float>& light_direction, int max_steps,
           const std::string& surface_mode, int density_filter, int detail,
           const py::object& objects, int grounding, float z_horizon,
           float fog_start, float fog_full) {
          if (position.size() != 3 || forward.size() != 3 ||
              right.size() != 3 || up.size() != 3 ||
              light_direction.size() != 3) {
            throw std::runtime_error(
                "terrain_render: camera vectors and light_direction must have length 3");
          }
          TerrainRenderCamera camera{};
          TerrainRenderLight light{};
          for (int axis = 0; axis < 3; ++axis) {
            camera.position[axis] = position[axis];
            camera.forward[axis] = forward[axis];
            camera.right[axis] = right[axis];
            camera.up[axis] = up[axis];
            light.direction[axis] = light_direction[axis];
          }
          camera.half_width = half_width;
          camera.half_height = half_height;
          camera.width = width;
          camera.height = height;
          if (max_steps <= 0) {
            throw std::runtime_error("terrain_render: max_steps must be positive");
          }
          if (surface_mode != "blocky" && surface_mode != "smooth") {
            throw std::runtime_error(
                "terrain_render: surface_mode must be 'blocky' or 'smooth'");
          }
          if (density_filter < 0 || density_filter > 2) {
            throw std::runtime_error(
                "terrain_render: density_filter must be 0, 1, or 2");
          }
          if (surface_mode == "blocky" && density_filter != 0) {
            throw std::runtime_error(
                "terrain_render: density_filter is only supported in smooth mode");
          }
          if (detail != 0 && detail != 1) {
            throw std::runtime_error(
                "terrain_render: detail must be 0 or 1");
          }
          if (grounding != 0 && grounding != 1) {
            throw std::runtime_error(
                "terrain_render: grounding must be 0 or 1");
          }
          if (!std::isfinite(z_horizon) || !std::isfinite(fog_start) ||
              !std::isfinite(fog_full)) {
            throw std::runtime_error(
                "terrain_render: grounding parameters must be finite");
          }
          if (fog_start < 0.0f || fog_full <= fog_start) {
            throw std::runtime_error(
                "terrain_render: fog range must satisfy 0 <= fog_start < fog_full");
          }
          // TerrainRenderConstants predates WO-8A and is used by the C++ op
          // ABI.  Reserve its otherwise-invalid negative range for the
          // pybind-only smooth selector; terrain.cu decodes it before launch.
          const int encoded_max_steps =
              surface_mode == "smooth" ? -max_steps : max_steps;
          TerrainRenderConstants constants{
              encoded_max_steps, density_filter, detail, grounding,
              z_horizon, fog_start, fog_full};
          const auto render_objects = terrain_objects_from_python(objects);
          auto out = render_objects.empty()
                         ? ops::terrain_render(materials, camera, light,
                                               palette, constants)
                         : ops::terrain_render(materials, camera, light,
                                               palette, constants,
                                               render_objects);
          return py::make_tuple(std::get<0>(out), std::get<1>(out));
        },
        py::arg("materials_u8"), py::arg("palette_u8"),
        py::arg("position"), py::arg("forward"), py::arg("right"),
        py::arg("up"), py::arg("half_width"), py::arg("half_height"),
        py::arg("width"), py::arg("height"), py::arg("light_direction"),
        py::arg("max_steps"), py::arg("surface_mode") = "blocky",
        py::arg("density_filter") = 0, py::arg("detail") = 0,
        py::arg("objects") = py::none(), py::arg("grounding") = 0,
        py::arg("z_horizon") = 0.0f, py::arg("fog_start") = 600.0f,
        py::arg("fog_full") = 2400.0f);
  m.def("causal_softmax", &ops::causal_softmax, py::arg("scores"));
  m.def("fused_sdpa_noncausal", &ops::fused_sdpa_noncausal,
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"));
  m.def("apa_int4_sdpa_noncausal", &ops::apa_int4_sdpa_noncausal,
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"),
        py::arg("zthr"), py::arg("refine_all") = false);
  m.def("apa_refine_stats",
        [](bool reset) {
          const auto stats = tc::apa_refine_stats(reset);
          return py::make_tuple(stats.first, stats.second);
        },
        py::arg("reset") = false);
  m.def("argmax_last_axis", &ops::argmax_last_axis, py::arg("a"));
  m.def("mse_loss", &ops::mse_loss);
  m.def("cross_entropy", &ops::cross_entropy);
  m.def("add", &ops::add);
  m.def("mul", &ops::mul);
  m.def("where", &ops::where);
  m.def("cat", [](std::vector<Tensor> ts, int dim) { return ops::cat(ts, dim); }, py::arg("tensors"), py::arg("dim") = 0);
  m.def("stack", [](std::vector<Tensor> ts, int dim) { return ops::stack(ts, dim); }, py::arg("tensors"), py::arg("dim") = 0);
  m.def("embedding", &ops::embedding);
  m.def("apa_quantize_gather", [](Tensor& r, Tensor& b, Tensor& c) {
    return Tensor::make(tc::apa_quantize_gather(r.data(), b.data(), c.data()), false);
  });
  // INT4 group-quantized linear. Weights are frozen (uint8/fp16 buffers,
  // never differentiable); the VJP covers x only — dx = g @ W^T with the
  // weight re-dequantized transiently at backward time (ops.cpp).
  m.def("int4_linear", &ops::int4_linear,
        py::arg("x"), py::arg("packed"), py::arg("scales"), py::arg("zeros"),
        py::arg("group_size") = 128);
  m.def("int4_linear_fused", &ops::int4_linear_fused,
        py::arg("x"), py::arg("packed"), py::arg("scales"), py::arg("zeros"),
        py::arg("group_size") = 128);
  m.def("w8a16_matmul", &ops::w8a16_matmul,
        py::arg("x"), py::arg("codes"), py::arg("scales"),
        py::arg("launch_config") = 0);
  m.def("intn_linear", &ops::intn_linear,
        py::arg("x"), py::arg("packed"), py::arg("scales"), py::arg("zeros"),
        py::arg("bits"), py::arg("in_features"),
        py::arg("group_size") = 128);
  m.def("intn_linear_fused", &ops::intn_linear_fused,
        py::arg("x"), py::arg("packed"), py::arg("scales"), py::arg("zeros"),
        py::arg("bits"), py::arg("in_features"),
        py::arg("group_size") = 128);
  m.def("mxfp4_linear", &ops::mxfp4_linear,
        py::arg("x"), py::arg("blocks"), py::arg("scales"));
  m.def("mxfp4_linear_expert", &ops::mxfp4_linear_expert,
        py::arg("x"), py::arg("blocks"), py::arg("scales"),
        py::arg("expert_idx"));
  // Differentiable O(L) selective attention (graft-native training path).
  m.def("apa_selective_train", [](Tensor& q, Tensor& k, Tensor& kq, Tensor& v,
                                  double scale, double zthr, bool is_causal) {
    return ops::apa_selective_train(q, k, kq, v, (float)scale, (float)zthr,
                                    is_causal);
  }, py::arg("q"), py::arg("k"), py::arg("kq"), py::arg("v"),
     py::arg("scale"), py::arg("zthr"), py::arg("is_causal") = false);
  // Fused GDN decode step (inference-only, functional: (out, new_state)).
  m.def("gated_delta_step", [](Tensor& q, Tensor& k, Tensor& v, Tensor& a,
                               Tensor& b, Tensor& A_neg, Tensor& dt_bias,
                               Tensor& state) {
    auto r = tc::gated_delta_step(q.data(), k.data(), v.data(), a.data(),
                                  b.data(), A_neg.data(), dt_bias.data(),
                                  state.data());
    return py::make_tuple(Tensor::make(r.first, false),
                          Tensor::make(r.second, false));
  }, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("a"), py::arg("b"),
     py::arg("A_neg"), py::arg("dt_bias"), py::arg("state"));
  m.def("int4_dequant", [](Tensor& packed, Tensor& scales, Tensor& zeros,
                           int group_size, const std::string& out_dtype) {
    return Tensor::make(
        tc::int4_dequant(packed.data(), scales.data(), zeros.data(), group_size,
                         dtype_from_string(out_dtype)),
        false);
  }, py::arg("packed"), py::arg("scales"), py::arg("zeros"),
     py::arg("group_size") = 128, py::arg("out_dtype") = "float16");
  m.def("intn_dequant", [](Tensor& packed, Tensor& scales, Tensor& zeros,
                           int bits, int64_t in_features, int group_size,
                           const std::string& out_dtype) {
    return Tensor::make(
        tc::intn_dequant(packed.data(), scales.data(), zeros.data(), bits,
                         in_features, group_size,
                         dtype_from_string(out_dtype)),
        false);
  }, py::arg("packed"), py::arg("scales"), py::arg("zeros"),
     py::arg("bits"), py::arg("in_features"),
     py::arg("group_size") = 128, py::arg("out_dtype") = "float16");
  // KV-cache INT4 storage (D-grouped symmetric-8). Distinct from int4_dequant
  // (weight path, K-grouped). pack -> (packed_u8, scales); unpack reads a
  // [lo:lo+n) slice on S. Inference-only (no autograd: the cache is frozen).
  m.def("kv_int4_pack", [](Tensor& x, int group) {
    NDArray scales;
    NDArray packed = tc::kv_int4_pack(x.data(), scales, group);
    return py::make_tuple(Tensor::make(packed, false),
                          Tensor::make(scales, false));
  }, py::arg("x"), py::arg("group") = 32);
  m.def("kv_int4_unpack", [](Tensor& packed, Tensor& scales, int group,
                             int64_t lo, int64_t n, const std::string& out_dtype) {
    return Tensor::make(
        tc::kv_int4_unpack(packed.data(), scales.data(), group, lo, n,
                           dtype_from_string(out_dtype)),
        false);
  }, py::arg("packed"), py::arg("scales"), py::arg("group") = 32,
     py::arg("lo") = 0, py::arg("n") = 0, py::arg("out_dtype") = "bfloat16");
  // Fused sparse selective APA attention (inference only, no autograd).
// APA_SP1_ADDITION_BEGIN binding
  // Separate opt-in API; no existing Python wrapper/launcher is redirected.
  m.def("apa_selective_attention_sp", [](Tensor& q, Tensor& k, Tensor& kq,
      Tensor& v, double scale, double delta, bool is_causal,
      py::object sinks_obj, bool diagnostics) -> py::object {
    const NDArray* sinks = nullptr;
    if (!sinks_obj.is_none()) sinks = &sinks_obj.cast<Tensor&>().data();
    NDArray selected;
    // Validate geometry in the launcher before any diagnostic allocation.
    // The launcher creates selected only after validation when requested.
    NDArray result = tc::apa_selective_attention_sp(q.data(), k.data(),
        kq.data(), v.data(), (float)scale, (float)delta, is_causal,
        sinks, diagnostics ? &selected : nullptr);
    Tensor output = Tensor::make(result, false);
    if (diagnostics) return py::make_tuple(output, Tensor::make(selected, false));
    return py::cast(output);
  }, py::arg("q"), py::arg("k"), py::arg("kq"), py::arg("v"),
     py::arg("scale"), py::arg("delta"), py::arg("is_causal") = false,
     py::arg("sinks") = py::none(), py::arg("diagnostics") = false);
  m.def("apa_sp1_1_baseline_diagnostics", [](Tensor& q, Tensor& kq) {
    auto result = tc::apa_sp1_1_baseline_diagnostics(q.data(),kq.data());
    return py::make_tuple(Tensor::make(result.first,false),Tensor::make(result.second,false));
  }, py::arg("q"), py::arg("kq"));
  // APA-SP2 derived-margin entry; original direct-delta binding stays intact.
  m.def("apa_sp2_delta", &tc::apa_sp2_delta, py::arg("epsilon"), py::arg("e_q"));
  m.def("apa_selective_attention_sp_epsilon", [](Tensor& q, Tensor& k, Tensor& kq,
      Tensor& v, double scale, double epsilon, double e_q, bool is_causal,
      py::object sinks_obj, bool diagnostics) -> py::object {
    const NDArray* sinks = nullptr;
    if (!sinks_obj.is_none()) sinks = &sinks_obj.cast<Tensor&>().data();
    NDArray selected;
    NDArray result = tc::apa_selective_attention_sp_epsilon(q.data(), k.data(),
        kq.data(), v.data(), (float)scale, epsilon, e_q, is_causal,
        sinks, diagnostics ? &selected : nullptr);
    Tensor output = Tensor::make(result, false);
    if (diagnostics) return py::make_tuple(output, Tensor::make(selected, false));
    return py::cast(output);
  }, py::arg("q"), py::arg("k"), py::arg("kq"), py::arg("v"), py::arg("scale"),
     py::arg("epsilon"), py::arg("e_q"), py::arg("is_causal") = false,
     py::arg("sinks") = py::none(), py::arg("diagnostics") = false);
  m.def("apa_sp2_scores", [](Tensor& q, Tensor& k, Tensor& kq, double scale,
                           int first, int count, bool wcoop) {
    auto r = tc::apa_sp2_scores(q.data(),k.data(),kq.data(),(float)scale,first,count,wcoop);
    return py::make_tuple(Tensor::make(r.first,false),Tensor::make(r.second,false));
  }, py::arg("q"), py::arg("k"), py::arg("kq"), py::arg("scale"),
     py::arg("first"), py::arg("count"), py::arg("wcoop"));
  m.def("apa_sp2_baseline_thresholds", [](Tensor& q, Tensor& kq, double scale,
                                         double zthr, bool causal) {
    return Tensor::make(tc::apa_sp2_baseline_thresholds(q.data(),kq.data(),
        (float)scale,(float)zthr,causal),false);
  }, py::arg("q"), py::arg("kq"), py::arg("scale"), py::arg("zthr"), py::arg("causal"));
// APA_SP1_ADDITION_END binding
  m.def("apa_selective_attention", [](Tensor& q, Tensor& k, Tensor& kq, Tensor& v,
                                      double scale, double zthr, bool is_causal) {
    return Tensor::make(
        tc::apa_selective_attention(q.data(), k.data(), kq.data(), v.data(),
                                    (float)scale, (float)zthr, is_causal),
        false);
  }, py::arg("q"), py::arg("k"), py::arg("kq"), py::arg("v"),
     py::arg("scale"), py::arg("zthr"), py::arg("is_causal") = false);
  m.def("apa_selective_attention_int4",
      [](Tensor& q, Tensor& k, Tensor& v, double scale, double zthr,
         bool is_causal) {
    return Tensor::make(
        tc::apa_selective_attention_int4(q.data(), k.data(), v.data(),
                                         (float)scale, (float)zthr,
                                         is_causal),
        false);
  }, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"),
     py::arg("zthr"), py::arg("is_causal") = false);
  m.def("apa_int4_workspace", [](Tensor& k, int64_t capacity) {
    return tc::apa_int4_workspace_create(k.data(), capacity);
  }, py::arg("k"), py::arg("capacity") = 0);
  m.def("apa_selective_attention_int4_workspace",
      [](Tensor& q, Tensor& k, Tensor& v,
         std::shared_ptr<ApaInt4Workspace> workspace, double scale,
         double zthr, bool is_causal) {
    if (!workspace) throw std::runtime_error("workspace must not be None");
    return Tensor::make(tc::apa_selective_attention_int4_workspace(
        q.data(), k.data(), v.data(), *workspace, (float)scale, (float)zthr,
        is_causal), false);
  }, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("workspace"),
     py::arg("scale"), py::arg("zthr"), py::arg("is_causal") = false);
  m.def("_apa_int4_profile_stages",
      [](Tensor& q, Tensor& k, Tensor& v,
         double scale, double zthr, bool is_causal, int warmup, int repeats,
         int profile_append_rows,
         std::shared_ptr<ApaInt4Workspace> workspace) {
    return tc::apa_selective_attention_int4_profile(
        q.data(), k.data(), v.data(), workspace.get(), (float)scale,
        (float)zthr, is_causal, warmup, repeats, profile_append_rows);
  }, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"), py::arg("zthr"),
     py::arg("is_causal") = false, py::arg("warmup") = 2,
     py::arg("repeats") = 10, py::arg("profile_append_rows") = 0,
     py::arg("workspace") = nullptr);
  m.def("_apa_int4_decode_plan", &tc::apa_int4_decode_plan,
        py::arg("rows"), py::arg("sequence"), py::arg("grid_fill") = false,
        py::arg("cache_bulk") = false, py::arg("split_stats") = false);
  m.def("apa_gemm_selective_attention",
      [](Tensor& q, Tensor& k, Tensor& v, double scale, double zthr,
         bool is_causal, int64_t Lq, int64_t row0, int64_t window) {
    return Tensor::make(
        tc::apa_gemm_selective_attention(
            q.data(), k.data(), v.data(), (float)scale, (float)zthr,
            is_causal, (int)Lq, row0, (int)window),
        false);
  }, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"),
     py::arg("zthr"), py::arg("is_causal") = false,
     py::arg("Lq") = 0, py::arg("row0") = 0, py::arg("window") = 0);
  m.def("apa_gemm_selective_attention_cached",
      [](Tensor& q, Tensor& k, Tensor& v, Tensor& k_codes,
         Tensor& k_scales, double scale, double zthr, bool is_causal,
         int64_t Lq, int64_t row0, int64_t window) {
    return Tensor::make(
        tc::apa_gemm_selective_attention(
            q.data(), k.data(), v.data(), (float)scale, (float)zthr,
            is_causal, (int)Lq, row0, (int)window,
            &k_codes.data(), &k_scales.data()),
        false);
  }, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("k_codes"),
     py::arg("k_scales"), py::arg("scale"), py::arg("zthr"),
     py::arg("is_causal") = false, py::arg("Lq") = 0,
     py::arg("row0") = 0, py::arg("window") = 0);
  m.def("apa_gemm_selective_quantize_k", [](Tensor& k) {
    auto r = tc::apa_gemm_selective_quantize_k(k.data());
    return py::make_tuple(Tensor::make(r.first, false),
                          Tensor::make(r.second, false));
  }, py::arg("k"));
  m.def("apa_gemm_selective_stats", [](bool reset) {
    auto r = tc::apa_gemm_selective_stats(reset);
    return py::make_tuple(std::get<0>(r), std::get<1>(r),
                          std::get<2>(r), std::get<3>(r));
  }, py::arg("reset") = false);
  m.def("apa_gemm_selective_debug",
      [](Tensor& q, Tensor& k, double scale, double zthr,
         bool is_causal, int64_t Lq, int64_t row0, int64_t window) {
    auto r = tc::apa_gemm_selective_debug(
        q.data(), k.data(), (float)scale, (float)zthr, is_causal,
        (int)Lq, row0, (int)window);
    return py::make_tuple(
        Tensor::make(std::get<0>(r), false),
        Tensor::make(std::get<1>(r), false),
        Tensor::make(std::get<2>(r), false),
        Tensor::make(std::get<3>(r), false),
        Tensor::make(std::get<4>(r), false),
        Tensor::make(std::get<5>(r), false),
        Tensor::make(std::get<6>(r), false));
  }, py::arg("q"), py::arg("k"), py::arg("scale"), py::arg("zthr"),
     py::arg("is_causal") = false, py::arg("Lq") = 0,
     py::arg("row0") = 0, py::arg("window") = 0);
  m.def("apa_selective_attention_sink",
      [](Tensor& q, Tensor& k, Tensor& kq, Tensor& v, Tensor& sinks,
         double scale, double zthr, bool is_causal) {
    return Tensor::make(
        tc::apa_selective_attention_sink(q.data(), k.data(), kq.data(), v.data(),
                                         sinks.data(), (float)scale,
                                         (float)zthr, is_causal),
        false);
  }, py::arg("q"), py::arg("k"), py::arg("kq"), py::arg("v"),
     py::arg("sinks"), py::arg("scale"), py::arg("zthr"),
     py::arg("is_causal") = false);
  // APA selective TRAINING forward: O(L) memory, saves (lse, thr) for backward.
  // Returns (out, lse, thr) as plain (non-grad) tensors; Python wires autograd.
  m.def("apa_selective_fwd_train", [](Tensor& q, Tensor& k, Tensor& kq, Tensor& v,
                                      double scale, double zthr, bool is_causal) {
    auto r = tc::apa_selective_fwd_train(q.data(), k.data(), kq.data(), v.data(),
                                         (float)scale, (float)zthr, is_causal);
    return py::make_tuple(Tensor::make(std::get<0>(r), false),
                          Tensor::make(std::get<1>(r), false),
                          Tensor::make(std::get<2>(r), false));
  }, py::arg("q"), py::arg("k"), py::arg("kq"), py::arg("v"),
     py::arg("scale"), py::arg("zthr"), py::arg("is_causal") = false);
  // APA selective backward: returns (dq, dk, dv).
  m.def("apa_selective_bwd", [](Tensor& q, Tensor& k, Tensor& kq, Tensor& v,
                                Tensor& dO, Tensor& lse, Tensor& thr,
                                double scale, bool is_causal) {
    auto r = tc::apa_selective_bwd(q.data(), k.data(), kq.data(), v.data(),
                                   dO.data(), lse.data(), thr.data(),
                                   (float)scale, is_causal);
    return py::make_tuple(Tensor::make(std::get<0>(r), false),
                          Tensor::make(std::get<1>(r), false),
                          Tensor::make(std::get<2>(r), false));
  }, py::arg("q"), py::arg("k"), py::arg("kq"), py::arg("v"), py::arg("dO"),
     py::arg("lse"), py::arg("thr"), py::arg("scale"), py::arg("is_causal") = false);
  // BP-KERNEL-1 opt-in bare op. Prior art: Project-Tensor bindings (2026);
  // ours: explicit census dispatch, no default binding/autograd change.
  m.def("apa_selective_bwd_variant", [](Tensor& q, Tensor& k, Tensor& kq, Tensor& v,
          Tensor& dO, Tensor& lse, Tensor& thr, Tensor& out,
          double scale, bool is_causal, const std::string& variant) {
    auto r = tc::apa_selective_bwd_variant(q.data(),k.data(),kq.data(),v.data(),
        dO.data(),lse.data(),thr.data(),out.data(),(float)scale,is_causal,variant);
    return py::make_tuple(Tensor::make(std::get<0>(r),false),
                         Tensor::make(std::get<1>(r),false),
                         Tensor::make(std::get<2>(r),false));
  }, py::arg("q"),py::arg("k"),py::arg("kq"),py::arg("v"),py::arg("dO"),
     py::arg("lse"),py::arg("thr"),py::arg("out"),py::arg("scale"),
     py::arg("is_causal"),py::arg("variant"));
  // Fused APA blend+softmax over precomputed bulk/rank score matrices.
  // Phase 3.1 (board item 4a): Lq<=0 (default) is the legacy sentinel path —
  // causal/window masking must already be baked into bulk/rank as
  // large-negative bias. Lq>0 is the index-arithmetic path — no mask tensor
  // needed; row0 = absolute query-chunk start, Lq = full query length,
  // window<=0 = full causal else sliding width. See tc/core.h for the exact
  // convention.
  m.def("apa_blend_softmax", [](Tensor& bulk, Tensor& rank, double zthr,
                                int64_t Lq, int64_t row0, int64_t window) {
    return Tensor::make(tc::apa_blend_softmax(bulk.data(), rank.data(),
                                              (float)zthr, nullptr,
                                              (int)Lq, row0, (int)window),
                        false);
  }, py::arg("bulk"), py::arg("rank"), py::arg("zthr"),
     py::arg("Lq") = 0, py::arg("row0") = 0, py::arg("window") = 0);
  m.def("apa_blend_softmax_sink", [](Tensor& bulk, Tensor& rank, Tensor& sinks,
                                     double zthr, int64_t Lq, int64_t row0,
                                     int64_t window) {
    return Tensor::make(
        tc::apa_blend_softmax_sink(bulk.data(), rank.data(), sinks.data(),
                                   (float)zthr, (int)Lq, row0, (int)window),
        false);
  }, py::arg("bulk"), py::arg("rank"), py::arg("sinks"), py::arg("zthr"),
     py::arg("Lq") = 0, py::arg("row0") = 0, py::arg("window") = 0);
  m.def("im2col", &ops::im2col);
  m.def("col2im", &ops::col2im);
  m.def("avg_pool2d", &ops::avg_pool2d);
  m.def("max_pool2d", &ops::max_pool2d);

  // in-place optimizer steps (param/state mutated on device)
  m.def("sgd_step", [](Tensor& p, Tensor& g, Tensor& buf, double lr, double mom, double wd) {
    tc::sgd_step(p.data(), g.data(), buf.data(), lr, mom, wd);
  });
  m.def("adam_step", [](Tensor& p, Tensor& g, Tensor& m_, Tensor& v_, double lr,
                        double b1, double b2, double eps, int64_t t, double wd, bool dec) {
    tc::adam_step(p.data(), g.data(), m_.data(), v_.data(), lr, b1, b2, eps, t, wd, dec);
  });
  m.def("rmsprop_step", [](Tensor& p, Tensor& g, Tensor& sq, double lr, double alpha, double eps, double wd) {
    tc::rmsprop_step(p.data(), g.data(), sq.data(), lr, alpha, eps, wd);
  });
  m.def("lion_step", [](Tensor& p, Tensor& g, Tensor& m_, double lr, double b1, double b2, double wd) {
    tc::lion_step(p.data(), g.data(), m_.data(), lr, b1, b2, wd);
  });
  m.def("radam_step", [](Tensor& p, Tensor& g, Tensor& m_, Tensor& v_, double lr, double b1,
                         double b2, double eps, double bc1, double bc2, double rect, bool rectified,
                         double wd, bool dec) {
    tc::radam_step(p.data(), g.data(), m_.data(), v_.data(), lr, b1, b2, eps, bc1, bc2, rect, rectified, wd, dec);
  });
  m.def("adagrad_step", [](Tensor& p, Tensor& g, Tensor& acc, double lr, double eps, double wd) {
    tc::adagrad_step(p.data(), g.data(), acc.data(), lr, eps, wd);
  });
  m.def("scale_", [](Tensor& t, double s) { tc::scale_(t.data(), s); });
  m.def("axpy_", [](Tensor& p, Tensor& o, double a) { tc::axpy_(p.data(), o.data(), a); });

  // Prior art: Project-Tensor explicit dispatch (2026); ours: opt-in census selection.
  // Prior art: CUDA events (NVIDIA 2007+), taken; opt-in half reporting ours.
  m.def("bp_kernel_4_set_variant", &tc::bp_kernel_4_set_variant);
  m.def("bp_kernel_4_get_variant", &tc::bp_kernel_4_get_variant);
  m.def("apa_selective_fwd_train_variant", [](Tensor& q,Tensor& k,Tensor& kq,Tensor& v,double scale,double zthr,bool causal,const std::string& variant) {
    auto r=tc::apa_selective_fwd_train_variant(q.data(),k.data(),kq.data(),v.data(),float(scale),float(zthr),causal,variant);
    return py::make_tuple(Tensor::make(std::get<0>(r),false),Tensor::make(std::get<1>(r),false),Tensor::make(std::get<2>(r),false));
  });
  m.def("bp_kernel_4_diagnostic", [](Tensor& q,Tensor& k,Tensor& kq,Tensor& v,double scale,double zthr,bool causal,const std::string& variant) {
    auto r=tc::apa_selective_fwd_bk4(q.data(),k.data(),kq.data(),v.data(),float(scale),float(zthr),causal,variant,true);
    return py::make_tuple(Tensor::make(std::get<0>(r),false),Tensor::make(std::get<1>(r),false),Tensor::make(std::get<2>(r),false),Tensor::make(std::get<3>(r),false),Tensor::make(std::get<4>(r),false));
  });
  m.def("bp_kernel_3_profile", &tc::bp_kernel_3_profile);
  m.def("bp_kernel_3_halves", &tc::bp_kernel_3_halves);
  m.def("bp_kernel_2_set_variant", &tc::bp_kernel_2_set_variant);
  m.def("bp_kernel_2_get_variant", &tc::bp_kernel_2_get_variant);
  // BP-CENSUS-1: no timer is enabled merely by importing the extension.
  m.def("bp_source_pin", []() { return std::string(bp::source_pin); });
  m.def("bp_wall_begin", &bp::wall_begin);
  m.def("bp_wall_end", &bp::wall_end);
  m.def("bp_configure", &bp::configure);
  m.def("bp_begin", &bp::begin);
  m.def("bp_end", &bp::end);
  m.def("bp_rows", []() {
    py::list out;
    auto sample_dict = [](const bp::Sample& s) {
      py::dict d; d["device_used_bytes"] = s.used; d["device_total_bytes"] = s.total;
      d["pool_used_bytes"] = s.pool_used; d["pool_reserved_bytes"] = s.pool_reserved;
      return d;
    };
    for (const auto& row : bp::rows) {
      py::dict d; d["name"] = row.name;
      d["inclusive_ms"] = row.inclusive_ms; d["exclusive_ms"] = row.exclusive_ms;
      d["before"] = sample_dict(row.before); d["after"] = sample_dict(row.after);
      out.append(d);
    }
    return out;
  });
  // grad mode
  m.def("is_grad_enabled", &grad_enabled);
  m.def("set_grad_enabled", &set_grad_enabled);
  m.def("checkpoint", &checkpoint_py, py::arg("fn"), py::arg("inputs"));
  m.def("synchronize", &cuda_sync);
  // P1T-M1: DLPack import of a capsule (tensor_cuda.from_dlpack wraps it), and
  // Amendment 1's device-side empty (tensor_cuda.empty wraps it).
  // P1T-FIX6 (sweep (g)): (*args, **kwargs) + CoreCallArgs, as the P1-T methods above.
  m.def("_from_dlpack_capsule",
        [](py::args a, py::kwargs k) {
          const CoreCallArgs c("_from_dlpack_capsule", a, k, {{"capsule"}});
          return tensor_from_dlpack_capsule(c.object(0));
        },
        "_from_dlpack_capsule(capsule)\n\nImport a DLPack capsule (tensor_cuda.from_dlpack wraps it).");
  m.def("empty",
        [](py::args a, py::kwargs k) {
          const CoreCallArgs c("empty", a, k, {{"shape"}, {"dtype", false}, {"device", false}});
          std::vector<int64_t> shape = c.shape(0);
          const std::string dtype = c.given(1) ? c.str(1) : std::string("float32");
          const std::string device = c.given(2) ? c.str(2) : std::string("cuda");
          return tensor_empty(std::move(shape), dtype, device);
        },
        "empty(shape, dtype='float32', device='cuda')\n\n"
        "Uninitialized tensor (tensor_cuda.empty wraps it). P1T-FIX6: shape must be a tuple or list of ints.");
  // P1T-FIX1 (F04/F05): counters of the DLPack import release policy (tests).
  m.def("_dlpack_release_stats", [](py::args a, py::kwargs k) {
    const CoreCallArgs c("_dlpack_release_stats", a, k, {});
    (void)c;
    py::dict d;
    d["released"] = g_import_released.load();
    d["leaked_interpreter_shutdown"] = g_import_leaked_shutdown.load();
    d["leaked_drain_failure"] = storage_external_drain_leaks();
    // P1T-FIX2 (N4/N5).
    d["deleter_exceptions"] = g_import_deleter_exceptions.load();
    d["leaked_closing"] = g_import_leaked_closing.load();
    d["lock_failures"] = g_import_lock_failures.load();
    {
      ImportReleaseLifecycle& lc = release_lifecycle();
      std::lock_guard<std::mutex> lk(lc.mu);  // GIL held, but no thread waits for the GIL while holding lc.mu
      d["in_flight"] = lc.in_flight;
      d["closing"] = lc.closing;
    }
    d["shutdown_calls"] = g_shutdown_calls.load();
    d["shutdown_in_flight_at_close"] = g_shutdown_in_flight_at_close.load();
    d["shutdown_timeouts"] = g_shutdown_timeouts.load();
    d["shutdown_still_in_flight"] = g_shutdown_still_in_flight.load();
    d["shutdown_wait_us"] = g_shutdown_wait_us.load();
    d["shutdown_diagnostics"] = g_shutdown_diagnostics.load();  // P1T-FIX3 (V3-01)
    // P1T-FIX7 (V7-01): hook closes whose knob could not be read (default delay used), the exit runner's own closes
    // after the hook raised before closing (and the other threads' releases it then waited for), lock retries.
    d["shutdown_diag_default_used"] = g_shutdown_diag_default_used.load();
    d["shutdown_runner_closes"] = g_shutdown_runner_closes.load();
    d["shutdown_runner_in_flight_at_close"] = g_shutdown_runner_in_flight_at_close.load();
    d["shutdown_lock_retries"] = g_shutdown_lock_retries.load();
    d["shutdown_lock_diagnostics"] = g_shutdown_lock_diagnostics.load();  // P1T-FIX7 L1
    d["shutdown_log_fallbacks"] = g_shutdown_log_fallbacks.load();  // P1T-FIX8 V8-02: allocation-free exit lines
    {  // P1T-FIX8 (V8-02): the recorded failures not raised yet (the last reader at exit consumes them)
      unsigned long long un = 0;
      const char* uw = nullptr;
      int ud = -1, ue = 0;
      dlpack_unreported_unraisable_raw(&un, &uw, &ud, &ue);
      d["unraisable_unreported"] = un;
      d["exit_report_unwritten"] = g_core_exit_report.unwritten;
    }
    d["device_restore_failures"] = dlpack_device_restore_failures();  // P1T-FIX4 (sweep): counted, never thrown
    // P1T-FIX5 (sweep d): every unraisable CUDA error of the P1-T core paths, recorded before its slot entry was
    // consumed (count), and the first one: site, device, CUDA error name ("" / -1 / "cudaSuccess" when none).
    d["unraisable_errors"] = dlpack_unraisable_errors();
    {
      std::string what;
      int dev = -1, err = 0;
      dlpack_first_unraisable(&what, &dev, &err);
      d["first_unraisable_what"] = what;
      d["first_unraisable_device"] = dev;
      d["first_unraisable_error"] = std::string(cudaGetErrorName(static_cast<cudaError_t>(err)));
    }
    return d;
  }, "P1T-FIX1: imported-Storage releases, deliberate leaks during interpreter shutdown (F04) and deliberate leaks "
     "after a failed device drain (F05); P1T-FIX2: deleter exceptions contained (N4), leaks after the exit hook "
     "closed, the in-flight count and the exit hook's wait (N5); P1T-FIX3: stderr diagnostics written while the exit "
     "hook waited (V3-01; shutdown_timeouts now counts waits that outlasted the first-diagnostic delay); P1T-FIX4: "
     "device_restore_failures, restores of the caller's CUDA device that failed where nothing could raise (the "
     "external Storage drain, or a fill_ guard unwinding after another error); P1T-FIX5: unraisable_errors and the "
     "first one (first_unraisable_what / _device / _error), every CUDA error of those paths recorded before its "
     "error-slot entry was consumed; P1T-FIX7: shutdown_diag_default_used (hook closes whose knob could not be read), "
     "shutdown_runner_closes / shutdown_runner_in_flight_at_close (the exit runner's own closes after the hook raised "
     "before closing), shutdown_lock_retries (failed lifecycle-mutex locks the exit close retried) and "
     "shutdown_lock_diagnostics (the stderr lines that retry loop wrote: it is diagnosed like the exit wait).");
  // P1T-FIX2 (N5): construct the release lifecycle state now, so no release path is the first to allocate it.
  (void)release_lifecycle();
  m.def("_dlpack_raise_unraisable",
        [](py::args a, py::kwargs k) {
          const CoreCallArgs c("_dlpack_raise_unraisable", a, k, {});
          (void)c;
          dlpack_raise_unraisable();
        },
        "_dlpack_raise_unraisable()\n\n"
        "P1T-FIX6 (FIX5 residual R7): raise RuntimeError once for the CUDA failures recorded on P1-T core paths that "
        "could not raise (imported Storage drain / restore, fill_ unwinding) and were not raised yet; called by "
        "tensor_cuda.synchronize() and logged by the exit hook. The record is consumed only once the RuntimeError "
        "exists (MemoryError leaves it in place).");
  // P1T-FIX6: a raw CPython function (raw_dlpack_begin_shutdown above): nothing on this stack owns a Python object
  // across the GIL-released wait, whatever the calling style.
  m.add_object("_dlpack_begin_shutdown", py::reinterpret_steal<py::object>(PyCFunction_NewEx(&kBeginShutdownDef,
                                                                                              nullptr, nullptr)));
  // P1T-FIX7 (V7-01): the atexit runner (raw_dlpack_exit_hook_runner) and the default diagnostic delay, exported so the
  // Python knob's initial value and the native fallback are one constant.
  m.add_object("_dlpack_exit_report_written",  // P1T-FIX8 (V8-02)
               py::reinterpret_steal<py::object>(PyCFunction_NewEx(&kExitReportWrittenDef, nullptr, nullptr)));
  m.add_object("_dlpack_test_fail_exit_report",  // P1T-FIX8 (V8-02 tests)
               py::reinterpret_steal<py::object>(PyCFunction_NewEx(&kTestFailExitReportDef, nullptr, nullptr)));
  m.add_object("_dlpack_exit_hook_runner", py::reinterpret_steal<py::object>(PyCFunction_NewEx(&kExitRunnerDef,
                                                                                                nullptr, nullptr)));
  m.attr("_DLPACK_SHUTDOWN_DIAG_DEFAULT_S") = kDlpackShutdownDiagDefaultS;
  m.def("empty_cache", &empty_cache);
  m.def("set_alloc_pooling", &set_alloc_pooling);
}
