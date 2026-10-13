// DLPack consumer implementation. Prior art and design notes: see dlpack_operand.h.
#include "dlpack_operand.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <deque>
#include <new>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "checked_math.h"
#include "cuda_guard.h"  // P1T-FIX4: DeviceGuard / QuietDeviceGuard and the error-slot rules
#include "tuning.h"

namespace tcs {

namespace {

constexpr const char* kCapsule = "dltensor";
constexpr const char* kCapsuleVersioned = "dltensor_versioned";
constexpr const char* kUsedCapsule = "used_dltensor";
constexpr const char* kUsedCapsuleVersioned = "used_dltensor_versioned";

std::string type_name(py::handle obj) { return Py_TYPE(obj.ptr())->tp_name; }

std::string capsule_name(py::handle cap) {
  const char* n = PyCapsule_GetName(cap.ptr());
  if (n == nullptr && PyErr_Occurred()) throw py::error_already_set();
  return n ? std::string(n) : std::string();
}

[[noreturn]] void refuse_value(const char* role, const std::string& msg) {
  throw py::value_error(std::string("tc_signal: operand '") + role + "': " + msg);
}

[[noreturn]] void refuse_type(const char* role, const std::string& msg) {
  throw py::type_error(std::string("tc_signal: operand '") + role + "': " + msg);
}

// Checks a capsule's name and returns its kind (true = versioned). Consumed and foreign capsules are refused.
bool capsule_kind(py::handle cap, const char* role) {
  const std::string n = capsule_name(cap);
  if (n == kCapsuleVersioned) return true;
  if (n == kCapsule) return false;
  if (n == kUsedCapsule || n == kUsedCapsuleVersioned)
    refuse_value(role, "the DLPack capsule was already consumed (name '" + n + "'); a capsule can be consumed once");
  refuse_type(role, "a capsule named '" + n + "' is not a DLPack capsule ('dltensor' or 'dltensor_versioned')");
}

void* capsule_pointer(py::handle cap, const char* name) {
  void* p = PyCapsule_GetPointer(cap.ptr(), name);
  if (p == nullptr) throw py::error_already_set();
  return p;
}

void check_version(const DLManagedTensorVersioned* m, const char* role) {
  if (m->version.major != DLPACK_MAJOR_VERSION) {
    std::ostringstream os;
    os << "DLPack major version " << m->version.major << " is not supported (this consumer reads major "
       << DLPACK_MAJOR_VERSION << ")";
    refuse_value(role, os.str());
  }
}

}  // namespace

// ------------------------------------------------------------------------------------------------ capsule acquisition
py::object to_capsule(py::handle obj, const char* role, bool writable) {
  if (PyCapsule_CheckExact(obj.ptr())) return py::reinterpret_borrow<py::object>(obj);
  if (!py::hasattr(obj, "__dlpack__"))
    refuse_type(role, "expected a DLPack producer (an object with __dlpack__) or a DLPack capsule, got " +
                          type_name(obj));
  // Array API order: ask the producer's device first. A non-CUDA producer is refused here, before __dlpack__ is
  // called with a CUDA stream argument (CPU producers such as NumPy reject any stream other than None).
  if (py::hasattr(obj, "__dlpack_device__")) {
    int device_type = -1;
    try {
      py::tuple dev = obj.attr("__dlpack_device__")().cast<py::tuple>();
      device_type = dev[0].cast<int>();
    } catch (py::cast_error&) {
      refuse_type(role, "__dlpack_device__() did not return a (device_type, device_id) tuple of ints");
    }
    if (device_type != kDLCUDA) {
      std::ostringstream os;
      os << "device type " << device_type << " is not kDLCUDA (2); tc_signal takes CUDA device memory only";
      refuse_value(role, os.str());
    }
  }
  py::object fn = obj.attr("__dlpack__");
  py::object cap;
  const py::tuple max_version = py::make_tuple(tuning::kDlpackMaxMajor, tuning::kDlpackMaxMinor);
  try {
    // Consumer stream = the legacy default stream (DLPack int 1): the producer orders its pending work before it.
    if (writable)
      cap = fn(py::arg("stream") = tuning::kDlpackLegacyStream, py::arg("max_version") = max_version,
               py::arg("copy") = false);
    else
      cap = fn(py::arg("stream") = tuning::kDlpackLegacyStream, py::arg("max_version") = max_version);
  } catch (py::error_already_set& e) {
    // A producer that predates the 2023 keywords rejects them with TypeError: retry with the stream argument only
    // (the array API standard's documented fallback; NumPy and PyTorch consumers do the same).
    if (!e.matches(PyExc_TypeError)) throw;
    cap = fn(py::arg("stream") = tuning::kDlpackLegacyStream);
  }
  if (!PyCapsule_CheckExact(cap.ptr()))
    refuse_type(role, "__dlpack__ returned " + type_name(cap) + ", not a PyCapsule");
  return cap;
}

const DLTensor* peek(py::handle capsule, const char* role) {
  if (!PyCapsule_CheckExact(capsule.ptr())) refuse_type(role, "expected a DLPack capsule, got " + type_name(capsule));
  if (capsule_kind(capsule, role)) {
    auto* m = static_cast<DLManagedTensorVersioned*>(capsule_pointer(capsule, kCapsuleVersioned));
    check_version(m, role);
    return &m->dl_tensor;
  }
  auto* m = static_cast<DLManagedTensor*>(capsule_pointer(capsule, kCapsule));
  return &m->dl_tensor;
}

// ------------------------------------------------------------------------------------------------ validation
int64_t validate_tensor(const DLTensor* t, const char* role, int ndim) {
  if (t->device.device_type != kDLCUDA) {
    std::ostringstream os;
    os << "device type " << static_cast<int>(t->device.device_type)
       << " is not kDLCUDA (2); tc_signal takes CUDA device memory only";
    refuse_value(role, os.str());
  }
  if (t->dtype.code != kDLFloat || t->dtype.bits != 32 || t->dtype.lanes != 1) {
    std::ostringstream os;
    os << "dtype (code " << static_cast<int>(t->dtype.code) << ", bits " << static_cast<int>(t->dtype.bits)
       << ", lanes " << t->dtype.lanes
       << ") is not float32; tc_signal computes in fp32 only (expand fp16 weights to fp32 first)";
    refuse_type(role, os.str());
  }
  // F03 (P1T-FIX1): a negative rank or a NULL shape array (with rank > 0) is refused before the shape is read.
  if (t->ndim < 0) {
    std::ostringstream os;
    os << "negative number of dimensions " << t->ndim;
    refuse_value(role, os.str());
  }
  if (t->ndim > 0 && t->shape == nullptr) refuse_value(role, "NULL shape array for a tensor with dimensions");
  if (t->ndim != ndim) {
    std::ostringstream os;
    os << "expected " << ndim << " dimensions, got " << t->ndim;
    refuse_value(role, os.str());
  }
  // F03: checked element count and fp32 byte total, before the stride walk below multiplies extents and before any
  // later byte range is formed.
  // P1T-FIX2, review finding N2 (zero-extent rule, checked_math.h ck_extent_bound): EVERY extent is multiplied (a zero
  // or negative one counted as 1) BEFORE the empty-operand refusal, so a shape such as [0, 2^62, 4] is refused as an
  // overflow, never waved through because its count is 0. Prior art: NumPy's array constructor rule (see
  // checked_math.h); the checked builtins (GCC 5, 2015).
  int64_t numel = 1;
  for (int i = 0; i < t->ndim; ++i) {
    if (__builtin_mul_overflow(numel, t->shape[i] > 1 ? t->shape[i] : int64_t{1}, &numel))
      refuse_value(role, "element count of shape " + shape_str(shape_of(t)) +
                             " overflows int64 (every nonzero extent counts, also next to a zero extent)");
  }
  int64_t nbytes = 0;
  if (__builtin_mul_overflow(numel, static_cast<int64_t>(sizeof(float)), &nbytes))
    refuse_value(role, "byte size of shape " + shape_str(shape_of(t)) + " (fp32) overflows int64");
  for (int i = 0; i < t->ndim; ++i)
    if (t->shape[i] <= 0) refuse_value(role, "empty operand (shape " + shape_str(shape_of(t)) + ")");
  // Every extent is >= 1 from here: numel is the element count and nbytes its fp32 byte size.
  if (t->strides != nullptr) {
    // C-contiguous: every dimension of extent > 1 has the compact stride; size-1 dimensions may carry any stride
    // (they are never stepped), as the order and the DLPack/array API conventions allow.
    int64_t want = 1;
    for (int i = t->ndim - 1; i >= 0; --i) {
      if (t->shape[i] != 1 && t->strides[i] != want) {
        std::ostringstream os;
        os << "not C-contiguous (shape " << shape_str(shape_of(t)) << ", strides [";
        for (int j = 0; j < t->ndim; ++j) os << (j ? ", " : "") << t->strides[j];
        os << "]); pass a contiguous tensor";
        refuse_value(role, os.str());
      }
      want *= t->shape[i];
    }
  }
  if (t->data == nullptr) refuse_value(role, "NULL data pointer");
  // F03: data + byte_offset (element 0) and element 0 + nbytes (one past the end) must not wrap the address space.
  uintptr_t p = 0;
  uintptr_t end = 0;
  if (__builtin_add_overflow(reinterpret_cast<uintptr_t>(t->data), static_cast<uintptr_t>(t->byte_offset), &p) ||
      __builtin_add_overflow(p, static_cast<uintptr_t>(nbytes), &end))
    refuse_value(role, "data + byte_offset + byte size overflows the address space");
  if (p % alignof(float) != 0) refuse_value(role, "data + byte_offset is not 4-byte aligned");
  return numel;
}

void require_operand(py::handle obj, const char* role) {
  // F02 (P1T-FIX1): None is never a tensor. Refused with TypeError before any planning, allocation or consumption.
  if (obj.is_none())
    refuse_type(role, "is required: expected a DLPack producer (an object with __dlpack__) or a DLPack capsule, got "
                      "None (only optional operands such as a bias may be None)");
}

// ------------------------------------------------------------------------------------------------ ManagedOwner
ManagedOwner::ManagedOwner(ManagedOwner&& o) noexcept : legacy(o.legacy), versioned(o.versioned) {
  o.legacy = nullptr;
  o.versioned = nullptr;
}

ManagedOwner& ManagedOwner::operator=(ManagedOwner&& o) noexcept {
  if (this != &o) {
    reset();
    legacy = o.legacy;
    versioned = o.versioned;
    o.legacy = nullptr;
    o.versioned = nullptr;
  }
  return *this;
}

// P1T-FIX2, review finding N4: a producer deleter that throws a C++ exception must not terminate the process. Every
// deleter call in tc_signal goes through reset() (the immediate release of a refused call, the deferred drain, the
// bound drain, the fallback sync, synchronize() and the atexit hook all end here), and reset() is noexcept, so an
// escaping exception used to reach std::terminate (round-2 probe cpu_native throw_deleter, rc -6). Now each call is
// wrapped: the exception is caught (catch-all), counted (pending_stats()['deleter_exceptions']) and dropped; the owner
// pointer is cleared BEFORE the call, so a deleter is never called twice, and the caller (a loop over an entry's
// managers) keeps going with the remaining managers. The DLPack ABI declares the deleter as a C function, so a
// throwing deleter already breaks the producer's contract; containing it is the consumer's choice, not a recovery
// (the producer's memory may or may not have been freed: tc_signal does not touch it again either way).
// Prior art: C++11 noexcept semantics ([except.terminate]: an exception escaping a noexcept function calls
// std::terminate) and the "destructors/cleanup must not throw" guideline (C++ Core Guidelines C.36/E.16, Sutter &
// Alexandrescu 2004, item 51). Taken: catch at the noexcept boundary. Ours: counting and continuing per manager.
// Known gap (not closed): a deleter that calls pthread_exit or is cancelled unwinds with abi::__forced_unwind, which
// may not be swallowed (glibc aborts); no C++ handler can contain that.
namespace {
std::atomic<unsigned long long> g_deleter_exceptions{0};  // static storage: counting never allocates

template <typename M>
void call_deleter_contained(M* m) noexcept {
  if (m == nullptr || m->deleter == nullptr) return;
  try {
    m->deleter(m);
  } catch (...) {
    g_deleter_exceptions.fetch_add(1, std::memory_order_relaxed);
  }
}
}  // namespace

void ManagedOwner::reset() noexcept {
  // DLPack consumer rule: when done, call the producer's deleter (it may be NULL). Called with the GIL held (every
  // tc_signal entry point runs under the GIL), so a deleter that touches Python objects is safe.
  if (legacy != nullptr) {
    DLManagedTensor* m = legacy;
    legacy = nullptr;  // cleared first: never called twice, even if the deleter throws or re-enters
    call_deleter_contained(m);
  }
  if (versioned != nullptr) {
    DLManagedTensorVersioned* m = versioned;
    versioned = nullptr;
    call_deleter_contained(m);
  }
}

unsigned long long deleter_exceptions() noexcept { return g_deleter_exceptions.load(std::memory_order_relaxed); }

// ------------------------------------------------------------------------------------------------ Operand
Operand::Operand(py::handle capsule, const char* role, int ndim, bool writable) : role_(role) {
  if (!PyCapsule_CheckExact(capsule.ptr())) refuse_type(role, "expected a DLPack capsule, got " + type_name(capsule));
  const bool versioned = capsule_kind(capsule, role);
  // Consume: take the pointer, rename the capsule to "used_*" (so its destructor no longer calls the deleter), and
  // own the managed tensor from here on (owner_ calls the deleter on every path, including a throw below).
  if (versioned) {
    auto* m = static_cast<DLManagedTensorVersioned*>(capsule_pointer(capsule, kCapsuleVersioned));
    if (PyCapsule_SetName(capsule.ptr(), kUsedCapsuleVersioned) != 0) throw py::error_already_set();
    owner_.versioned = m;
    check_version(m, role);
    if (writable && (m->flags & DLPACK_FLAG_BITMASK_READ_ONLY))
      refuse_value(role, "the producer marked this tensor read-only; tc_signal writes this operand");
    if (writable && (m->flags & DLPACK_FLAG_BITMASK_IS_COPIED))
      refuse_value(role, "the producer exported a copy; writes to it would not reach the caller's tensor");
    t_ = &m->dl_tensor;
  } else {
    auto* m = static_cast<DLManagedTensor*>(capsule_pointer(capsule, kCapsule));
    if (PyCapsule_SetName(capsule.ptr(), kUsedCapsule) != 0) throw py::error_already_set();
    owner_.legacy = m;
    t_ = &m->dl_tensor;
  }
  numel_ = validate_tensor(t_, role, ndim);
  data_ = reinterpret_cast<float*>(static_cast<char*>(t_->data) + t_->byte_offset);  // byte_offset honoured
}

Operand::~Operand() = default;

Operand::Operand(Operand&& o) noexcept
    : owner_(std::move(o.owner_)), t_(o.t_), data_(o.data_), numel_(o.numel_), role_(o.role_) {
  o.t_ = nullptr;
  o.data_ = nullptr;
  o.numel_ = 0;
}

Operand& Operand::operator=(Operand&& o) noexcept {
  if (this != &o) {
    owner_ = std::move(o.owner_);
    t_ = o.t_;
    data_ = o.data_;
    numel_ = o.numel_;
    role_ = o.role_;
    o.t_ = nullptr;
    o.data_ = nullptr;
    o.numel_ = 0;
  }
  return *this;
}

const DLTensor* Operand::meta() const {
  // F02 second line of defence: an absent (None) operand has no metadata; never dereference NULL.
  if (t_ == nullptr)
    refuse_type(role_[0] ? role_ : "?", "is absent (None); its shape/device cannot be read");
  return t_;
}

int64_t Operand::numel() const {
  meta();
  return numel_;  // checked at construction (validate_tensor)
}

void Operand::release() {
  owner_.reset();
  t_ = nullptr;
  data_ = nullptr;
  numel_ = 0;
}

ManagedOwner Operand::take_owner() {
  ManagedOwner o(std::move(owner_));
  t_ = nullptr;
  data_ = nullptr;
  numel_ = 0;
  return o;
}

// ------------------------------------------------------------------------------------------------ helpers
std::vector<int64_t> shape_of(const DLTensor* t) { return std::vector<int64_t>(t->shape, t->shape + t->ndim); }

std::string shape_str(const std::vector<int64_t>& s) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < s.size(); ++i) os << (i ? ", " : "") << s[i];
  os << "]";
  return os.str();
}

void expect_shape(const Operand& op, const std::vector<int64_t>& want) {
  std::vector<int64_t> got(op.ndim());
  for (int i = 0; i < op.ndim(); ++i) got[i] = op.dim(i);
  if (got != want) refuse_value(op.role(), "shape " + shape_str(got) + " != expected " + shape_str(want));
}

int common_device(const std::vector<const Operand*>& ops) {
  int dev = -1;
  const char* first = "";
  for (const Operand* op : ops) {
    if (op == nullptr || !op->present()) continue;
    if (dev < 0) {
      dev = op->device();
      first = op->role();
    } else if (op->device() != dev) {
      std::ostringstream os;
      os << "on CUDA device " << op->device() << " but operand '" << first << "' is on CUDA device " << dev
         << "; every operand must be on the same device";
      refuse_value(op->role(), os.str());
    }
  }
  return dev;
}

void expect_disjoint(const Operand& written, const Operand& other) {
  if (!written.present() || !other.present()) return;
  if (written.byte_begin() < other.byte_end() && other.byte_begin() < written.byte_end())
    refuse_value(written.role(), std::string("its memory overlaps operand '") + other.role() +
                                     "'; the written operand must not alias any other operand");
}

// ------------------------------------------------------------------------------------------------ deferred release (F01)
// Design and prior art: dlpack_operand.h, RetainUntilComplete. Every function here runs on the calling host thread with
// the GIL held (every _tc_signal entry point is a pybind11 function that keeps the GIL), so the pending list needs no
// lock. Deleters may run Python (a ctypes producer) and may even re-enter tc_signal; therefore an entry is always taken
// OFF the list before its deleters run, and nothing holds a reference into the list across a deleter call.
//
// P1T-FIX2 (round-2 review N1, N3) changed the release paths as follows:
//  * N3, no allocation failure terminates. The noexcept functions here (drain_completed, the RetainUntilComplete
//    destructor, sync_and_release_all_impl, recycle_event) no longer allocate, except the pending-list push in the
//    destructor, which is caught and falls back to a synchronous release (counted). sync_and_release_all_impl works on
//    the live list in place (phase 1 synchronizes the devices into a fixed-capacity set, phase 2 pops completed
//    entries), instead of a take-all std::deque swap and a std::set / std::vector of devices, which allocated inside a
//    noexcept function (libstdc++'s deque allocates even when default-constructed: the round-2 probe oom_sync, rc -6).
//    The recycle slot of every completion event is reserved when the event is created (constructor path, before any
//    launch), so recycling never allocates. The owner vector is reserved in the constructor, as before.
//  * N1, no CUDA error is reported as success: a failed current-device query (cudaGetDevice) is now an error of
//    synchronize() (it used to be skipped, so an empty list "synchronized" successfully on a machine with no usable
//    device: probe empty_sync); a stream sync that fails is an error; and a release-path failure that happens where
//    nothing can raise (the fallback sync and the bound drain inside the destructor) is recorded and raised by the next
//    synchronize() (once), and logged by the atexit hook.
// Prior art for the in-place two-phase drain: PyTorch's CUDA caching allocator processes its event list in place from
// the host (process_events: query each recorded event, stop at the first one not ready). Taken: query-and-pop in FIFO
// order. Ours: the sync-then-pop split with the per-device failure set, and leak-instead-of-release for entries whose
// device could not be synchronized. Fixed-capacity, allocation-free bookkeeping on failure paths is ordinary
// "no allocation on the error path" practice (e.g. the C++ Core Guidelines E.16 discussion); no specific prior art
// claimed for the set itself.
namespace test_hooks {
namespace {
std::atomic<unsigned> g_armed{0};
}  // namespace
void arm(unsigned bits) noexcept { g_armed.fetch_or(bits, std::memory_order_relaxed); }
void clear() noexcept { g_armed.store(0, std::memory_order_relaxed); }
unsigned armed() noexcept { return g_armed.load(std::memory_order_relaxed); }
bool take(unsigned bit) noexcept {
  if ((g_armed.load(std::memory_order_relaxed) & bit) == 0) return false;  // the production path: one relaxed load
  return (g_armed.fetch_and(~bit, std::memory_order_relaxed) & bit) != 0;
}
}  // namespace test_hooks

namespace {

// TEST HOOK: the CUDA error an armed kCuda* site reports instead of calling CUDA (tests only; never in production).
constexpr cudaError_t kTestInjectedCudaError = cudaErrorLaunchFailure;

struct PendingEntry {
  cudaEvent_t event = nullptr;   // recorded on `device`'s legacy stream right after the call's last launch
  int device = -1;
  int64_t bytes = 0;             // sum of the fp32 operand sizes this entry keeps alive
  std::vector<ManagedOwner> owners;
};

// Completion events of one device: `free` holds completed events for reuse; its capacity is kept >= `created` (the
// number of live events of this device), so returning an event to it never allocates (N3).
struct DeviceEvents {
  std::vector<cudaEvent_t> free;
  size_t created = 0;
};

struct PendingState {
  std::deque<PendingEntry> q;
  int64_t bytes = 0;            // V3-02: saturating sum of the queued entries' bytes (exact while !bytes_saturated)
  bool bytes_saturated = false;            // V3-02: a push overflowed; `bytes` is pinned at INT64_MAX until empty
  unsigned long long bytes_saturations = 0;  // V3-02: how many times the total saturated
  long long managers = 0;
  long long high_water_entries = 0;
  long long high_water_managers = 0;
  int64_t high_water_bytes = 0;
  unsigned long long retained_entries = 0;   // entries ever pushed
  unsigned long long released_entries = 0;
  unsigned long long released_managers = 0;
  unsigned long long bound_drains = 0;       // a push exceeded a bound -> synchronize + release everything
  unsigned long long fallback_syncs = 0;     // the event could not be recorded/queued -> the legacy stream was synced
  unsigned long long leaked_managers = 0;    // no completion proof (broken context): deliberately not released
  unsigned long long push_alloc_failures = 0;     // N3: the pending-list push failed to allocate (-> fallback sync)
  unsigned long long event_recycle_failures = 0;  // N3: an event could not be pooled (destroyed instead)
  unsigned long long release_errors = 0;          // N1: CUDA failures on the release paths (fallback sync, bound drain)
  // N1: release-path failures not yet raised by synchronize() (the first one is named; all are counted).
  unsigned long long unreported_count = 0;
  unsigned long long unreported_leaked = 0;
  int unreported_device = -1;
  cudaError_t unreported_error = cudaSuccess;
  // P1T-FIX4 (sweep): CUDA failures on paths that cannot raise and leak no operand (a restore of the caller's device,
  // an event destroy). Counted; the first is named and all are raised once by the next synchronize().
  unsigned long long device_restore_failures = 0;
  unsigned long long event_destroy_failures = 0;
  unsigned long long unraisable_unreported = 0;
  const char* unraisable_what = "";
  int unraisable_device = -1;
  cudaError_t unraisable_error = cudaSuccess;
  // P1T-FIX4 V4-04: the calling thread's pre-existing error taken by synchronize() on entry, held until its message
  // exists (so a MemoryError while building the message does not lose it: the next synchronize() raises it).
  unsigned long long prior_unreported = 0;
  cudaError_t prior_error = cudaSuccess;
  std::unordered_map<int, DeviceEvents> event_pool;   // per device
};

// Never destroyed: no static destructor may call a producer deleter at process exit (Python may be gone by then).
// Constructed at module import (init_pending_state), so the first call from a noexcept release path never allocates.
PendingState& pending() {
  static PendingState* state = new PendingState();
  return *state;
}

// QuietDeviceGuard (makes `device` current for its scope, never throws) moved to cuda_guard.h in P1T-FIX4, where its
// restore is now checked (62c8d9f ignored the restoring cudaSetDevice's result).

// Synchronizes the legacy stream of `device` (switching to it and back). Never throws, never allocates.
cudaError_t sync_device_legacy(int device) noexcept {
  QuietDeviceGuard g(device);
  if (!g.ok()) return g.error();
  const cudaError_t r =
      test_hooks::take(test_hooks::kCudaSync) ? kTestInjectedCudaError : cudaStreamSynchronize(cudaStreamLegacy);
  if (r != cudaSuccess) cudaGetLastError();
  return r;
}

// N1/N3: the devices one sync_and_release_all_impl call synchronized, with each result. Fixed capacity, no allocation.
// A device beyond the capacity is still synchronized; its failure is remembered only as `spill_failed`, which makes
// every unrecorded device count as failed (conservative: its not-yet-complete entries are leaked, never released).
struct DeviceSyncSet {
  int dev[tuning::kSyncMaxDevices];
  cudaError_t err[tuning::kSyncMaxDevices];
  int n = 0;
  bool spill_failed = false;
  cudaError_t spill_error = cudaSuccess;

  void sync(int d) noexcept {
    for (int i = 0; i < n; ++i)
      if (dev[i] == d) return;
    const cudaError_t r = sync_device_legacy(d);
    if (n < tuning::kSyncMaxDevices) {
      dev[n] = d;
      err[n] = r;
      ++n;
    } else if (r != cudaSuccess) {
      spill_failed = true;
      spill_error = r;
    }
  }
  bool failed(int d) const noexcept {
    for (int i = 0; i < n; ++i)
      if (dev[i] == d) return err[i] != cudaSuccess;
    return spill_failed;
  }
};

// What one sync_and_release_all_impl call found (fixed capacity: the first kSyncReportMax failures are named, the
// rest only counted).
struct SyncReport {
  bool query_failed = false;               // cudaGetDevice failed (N1: never a success)
  cudaError_t query_error = cudaSuccess;
  int n = 0;
  int dev[tuning::kSyncReportMax];
  cudaError_t err[tuning::kSyncReportMax];
  int unnamed = 0;                         // failures beyond the named ones
  unsigned long long leaked_managers = 0;  // managers leaked by this call (no completion proof)

  void add(int d, cudaError_t e) noexcept {
    if (n < tuning::kSyncReportMax) {
      dev[n] = d;
      err[n] = e;
      ++n;
    } else {
      ++unnamed;
    }
  }
  bool failed() const noexcept { return query_failed || n > 0 || unnamed > 0 || leaked_managers > 0; }
  int first_device() const noexcept { return n > 0 ? dev[0] : -1; }
  cudaError_t first_error() const noexcept { return query_failed ? query_error : (n > 0 ? err[0] : cudaErrorUnknown); }
};

// N1: a CUDA failure on a release path that cannot raise (inside the noexcept destructor). Raised by the next
// synchronize(), logged at exit. Never allocates.
void note_release_failure(PendingState& st, int device, cudaError_t err, unsigned long long leaked) noexcept {
  ++st.release_errors;
  if (st.unreported_count == 0) {
    st.unreported_device = device;
    st.unreported_error = err;
  }
  ++st.unreported_count;
  st.unreported_leaked += leaked;
}

// P1T-FIX4 (sweep): a failure that cannot raise where it happens and leaks no operand (see PendingState). Never
// allocates.
void note_unraisable_failure(PendingState& st, const char* what, int device, cudaError_t err) noexcept {
  if (st.unraisable_unreported == 0) {
    st.unraisable_what = what;
    st.unraisable_device = device;
    st.unraisable_error = err;
  }
  ++st.unraisable_unreported;
}

// P1T-FIX4 (sweep): an event is destroyed on its own device (62c8d9f destroyed it on whatever device was current and
// ignored the result). Prior art: PyTorch's CUDAEvent destructor, "event destruction done on creating device to avoid
// creating a CUDA context on other devices" (c10/cuda/CUDAEvent.h, torch 2.11 headers: CUDAGuard, then
// C10_CUDA_CHECK_WARN(cudaEventDestroy)). Taken: destroy under the event's device, checked, never thrown. Ours: the
// count and the deferred raise by the next synchronize().
void destroy_event_on_device(PendingState& st, int device, cudaEvent_t ev) noexcept {
  QuietDeviceGuard g(device);
  const cudaError_t r = g.ok() ? cudaEventDestroy(ev) : g.error();
  if (r != cudaSuccess) {
    cudaGetLastError();  // the guard consumed its own failure already; this consumes cudaEventDestroy's
    ++st.event_destroy_failures;
    note_unraisable_failure(st, "cudaEventDestroy", device, r);
  }
}

void recycle_event(PendingState& st, int device, cudaEvent_t ev) noexcept {
  if (ev == nullptr) return;
  DeviceEvents* de = nullptr;
  try {
    auto it = st.event_pool.find(device);  // find, never operator[]: no insertion on the release path
    if (it != st.event_pool.end()) de = &it->second;
    // N3: the slot was reserved when the event was created; refuse (never allocate) if it somehow is not there.
    if (test_hooks::take(test_hooks::kAllocEventRecycle) || de == nullptr || de->free.size() >= de->free.capacity())
      throw std::bad_alloc();
    de->free.push_back(ev);  // within capacity: no allocation
  } catch (...) {
    ++st.event_recycle_failures;
    if (de != nullptr && de->created > 0) --de->created;
    destroy_event_on_device(st, device, ev);  // P1T-FIX4: on its own device, checked
  }
}

// P1T-FIX3, round-3 review finding V3-02: the pending-list byte total is SATURATING. note_push / note_pop run inside the
// noexcept RetainUntilComplete destructor and the drain loops, so they may neither throw nor drain; an unchecked
// signed addition here was undefined behaviour for near-limit totals (review probe pending_bytes queue: 4 +
// 9223372036854775804), and a wrapped negative total could have suppressed the bound drain.
//  * Push: checked addition; on overflow the total is pinned at INT64_MAX and marked saturated (counted). Every entry's
//    own bytes is a checked int64 sum >= 0 (aggregate_operand_bytes, before the launch), so only the queue total can
//    overflow.
//  * Pop: exact subtraction while not saturated (the total is then the exact sum of the queued entries, so it never
//    goes below 0); while saturated the total stays pinned at INT64_MAX until the list is empty, where the true total is
//    0 again. A pinned total over-reports, never under-reports, so the bound comparison (pending_over_bound) is exact
//    below INT64_MAX and conservative above it: a push that would overflow always drains, as the exact arithmetic would
//    have (any total above INT64_MAX is far above kPendingMaxBytes).
// Chosen over drain-before-add because these functions are pure accounting: draining runs producer deleters (Python,
// possibly re-entering tc_signal) and cannot run in the middle of a push; and a drain cannot guarantee headroom anyway
// (entries pushed by a re-entrant call during the drain stay queued), so a non-throwing, non-overflowing rule is needed
// in any case. Saturation is that rule; the drain itself is left to the bound comparison right after the push.
// Prior art: saturating arithmetic (DSP/SIMD saturated add, e.g. Intel MMX/SSE PADDS, 1997-; ARM QADD) and the checked
// builtins it is built from (GCC 5 __builtin_add_overflow, 2015; C23 ckd_add); SEI CERT INT32-C. Taken: checked
// add with a saturated result. Ours: the sticky saturated state that is cleared only when the list empties, so pops
// never make a saturated total under-report. Unverified here (lead to check): "PADDSW saturating add MMX 1997".
void note_push(PendingState& st, const PendingEntry& e) noexcept {
  int64_t sum = 0;
  if (!st.bytes_saturated && __builtin_add_overflow(st.bytes, e.bytes, &sum)) {
    st.bytes_saturated = true;
    ++st.bytes_saturations;
  }
  st.bytes = st.bytes_saturated ? std::numeric_limits<int64_t>::max() : sum;
  st.managers += static_cast<long long>(e.owners.size());
  ++st.retained_entries;
  st.high_water_entries = std::max<long long>(st.high_water_entries, static_cast<long long>(st.q.size()));
  st.high_water_managers = std::max(st.high_water_managers, st.managers);
  st.high_water_bytes = std::max(st.high_water_bytes, st.bytes);
}

// Called right after the entry was taken off the list.
void note_pop(PendingState& st, const PendingEntry& e) noexcept {
  if (st.bytes_saturated) {
    if (st.q.empty()) {  // the true total is 0 again
      st.bytes = 0;
      st.bytes_saturated = false;
    }
  } else {
    int64_t rest = 0;
    // Exact while not saturated (e.bytes <= st.bytes); the guard only keeps a corrupted total from wrapping.
    st.bytes = (__builtin_sub_overflow(st.bytes, e.bytes, &rest) || rest < 0) ? 0 : rest;
  }
  st.managers -= static_cast<long long>(e.owners.size());
}

// V3-02: the bound comparison after a push. Exact below INT64_MAX (a total of exactly kPendingMaxBytes is within the
// bound, one byte more is over it, as before); a saturated total is over it.
bool pending_over_bound(const PendingState& st) noexcept {
  return static_cast<long long>(st.q.size()) > tuning::kPendingMaxEntries || st.bytes_saturated ||
         st.bytes > tuning::kPendingMaxBytes;
}

// Runs the entry's deleters (they may run Python and re-enter tc_signal; the entry is already off the list). A throwing
// deleter is contained in ManagedOwner::reset (N4) and the loop goes on with the remaining managers.
void release_entry(PendingState& st, PendingEntry& e) noexcept {
  recycle_event(st, e.device, e.event);
  e.event = nullptr;
  ++st.released_entries;
  st.released_managers += e.owners.size();
  for (ManagedOwner& o : e.owners) o.reset();
}

// Returns the number of managers leaked.
unsigned long long leak_entry(PendingState& st, PendingEntry& e) noexcept {
  unsigned long long n = 0;
  for (ManagedOwner& o : e.owners) {
    if (!o.empty()) ++n;
    o.leak();
  }
  st.leaked_managers += n;
  // The event is not recycled either: it may belong to a broken context.
  e.event = nullptr;
  return n;
}

// Phase 1: synchronize the legacy stream of the current device and of every device with pending entries (no deleter
// runs, so iterating the live list is safe). Phase 2: pop entries in FIFO order and release each one whose event has
// completed; leak (counted) an entry with no completion proof whose device could not be synchronized or whose event
// query failed; stop at the first entry that is merely not ready on a device that DID synchronize: it was pushed after
// phase 1 by a re-entrant call (a deleter that calls tc_signal) and is left pending. Every entry is popped before its
// deleters run, so a re-entrant drain or bound drain pops each entry at most once. No allocation, never throws.
SyncReport sync_and_release_all_impl() noexcept {
  PendingState& st = pending();
  SyncReport rep;
  DeviceSyncSet set;
  int cur = -1;
  const cudaError_t qe = cudaGetDevice(&cur);
  if (qe != cudaSuccess) {
    cudaGetLastError();
    rep.query_failed = true;  // N1: was silently skipped (synchronize() then "succeeded" with no device)
    rep.query_error = qe;
  } else {
    set.sync(cur);
  }
  for (const PendingEntry& e : st.q) set.sync(e.device);
  for (int i = 0; i < set.n; ++i)
    if (set.err[i] != cudaSuccess) rep.add(set.dev[i], set.err[i]);
  if (set.spill_failed) rep.add(-1, set.spill_error);

  while (!st.q.empty()) {
    const cudaError_t r = cudaEventQuery(st.q.front().event);
    const int dev = st.q.front().device;
    const bool complete = (r == cudaSuccess);
    if (!complete) {
      if (r == cudaErrorNotReady) {
        if (cudaPeekAtLastError() == cudaErrorNotReady) cudaGetLastError();
        if (!set.failed(dev)) break;  // pushed after phase 1 (re-entrant call): not ours to force
      } else {
        cudaGetLastError();           // the event itself cannot be queried: no completion proof (N1: an error)
        rep.add(dev, r);
      }
    }
    PendingEntry e = std::move(st.q.front());
    st.q.pop_front();
    note_pop(st, e);
    if (complete)
      release_entry(st, e);
    else
      rep.leaked_managers += leak_entry(st, e);
  }
  return rep;
}

cudaEvent_t acquire_event(int device) {
  PendingState& st = pending();
  DeviceEvents& de = st.event_pool[device];  // may allocate: constructor path, nothing has been launched yet
  if (!de.free.empty()) {
    cudaEvent_t ev = de.free.back();
    de.free.pop_back();
    return ev;
  }
  // N3: reserve the recycle slot of the event about to be created HERE, where an allocation failure only refuses the
  // call (MemoryError, operands released at once), so returning the event to the pool later never allocates.
  if (test_hooks::take(test_hooks::kAllocEventPoolReserve)) throw std::bad_alloc();
  if (de.free.capacity() < de.created + 1) de.free.reserve(std::max<size_t>(2 * de.free.capacity(), de.created + 1));
  QuietDeviceGuard g(device);
  cudaEvent_t ev = nullptr;
  cudaError_t r = g.ok() ? cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) : g.error();
  if (r != cudaSuccess) {
    cudaGetLastError();
    std::ostringstream os;
    os << "tc_signal: cannot create the completion event on CUDA device " << device << ": " << cudaGetErrorName(r)
       << ": " << cudaGetErrorString(r);
    throw std::runtime_error(os.str());
  }
  ++de.created;
  return ev;
}

}  // namespace

void init_pending_state() { (void)pending(); }

// P1T-FIX4 (cuda_guard.h rule 2): a restore of the caller's device that failed where nothing can raise.
void note_unraisable_device_restore_failure(int device, cudaError_t err) noexcept {
  PendingState& st = pending();
  ++st.device_restore_failures;
  note_unraisable_failure(st, "restoring the caller's CUDA device (cudaSetDevice)", device, err);
}

int64_t aggregate_operand_bytes(const std::vector<Operand*>& ops) {
  // V3-02 (see dlpack_operand.h): checked sum of the present operands' fp32 bytes; ValueError on overflow (ck_add).
  // Prior art: the checked builtins (GCC 5 __builtin_add_overflow, 2015; C23 ckd_add), as in checked_math.h. Ours:
  // checking the per-call aggregate before the first launch, where a refusal can still raise.
  int64_t total = 0;
  for (const Operand* op : ops) {
    if (op == nullptr || !op->present()) continue;
    const int64_t b = op->numel() * static_cast<int64_t>(sizeof(float));  // checked at construction (F03)
    total = ck_add(total, b, "the aggregate fp32 byte count of this call's operands (their sum)");
  }
  return total;
}

RetainUntilComplete::RetainUntilComplete(int device, std::initializer_list<Operand*> ops)
    : device_(device), ops_(ops) {
  // V3-02: refuse a call whose operands' aggregate byte count overflows int64, FIRST: before any allocation, event or
  // launch (ValueError; the operands are then released at once by their own destructors, nothing was queued).
  bytes_ = aggregate_operand_bytes(ops_);
  // N3: every allocation the release needs happens here (may throw: then nothing has been launched and the operands
  // are released at once by their own destructors); the destructor below allocates only for the pending-list push.
  if (test_hooks::take(test_hooks::kAllocOwnerReserve)) throw std::bad_alloc();
  owners_.reserve(ops_.size());
  event_ = acquire_event(device_);  // if it throws, nothing else is held and nothing has been launched
  // P1T-FIX4 (sweep, CUDA error state): every launch of every entry point follows this constructor, and each launch is
  // checked by reading the calling thread's last-error slot. 62c8d9f never looked at the slot BEFORE the first launch,
  // so an error already pending there (a refused launch elsewhere in this thread, a failure some other code did not
  // consume) was reported after this call's launches as this call's own launch failure, with the work queued anyway.
  // Now it is taken here, right before the first launch, and raised as what it is: RuntimeError naming the pending
  // error, nothing launched, and the operands released at once by their own destructors (nothing was queued on them;
  // the completion event goes back to its pool). From here on a launch check can only report this call's own error.
  // Placed after acquire_event so a machine with no usable device keeps its earlier refusal ("cannot create the
  // completion event", cudaErrorNoDevice, which the runtime keeps in the slot permanently: receipt
  // artifacts/p1t_fix4/cuda_semantics.txt).
  // Prior art: CUDA 12.6 Programming Guide 3.2.12: "one has to make sure that the runtime error variable is set to
  // cudaSuccess just before the kernel launch, for example, by calling cudaGetLastError() just before the kernel
  // launch" (artifacts/p1t_fix4/cuda_doc_quotes.txt). Taken: that rule. Ours: raising the pending error as pre-existing
  // instead of discarding it, at the one constructor every launch path shares.
  const cudaError_t pending_err = cudaGetLastError();
  if (pending_err != cudaSuccess) {
    recycle_event(pending(), device_, event_);  // never allocates (its recycle slot was reserved by acquire_event)
    event_ = nullptr;
    std::ostringstream os;
    os << "tc_signal: a CUDA error was already pending in this thread before this call's first launch ("
       << cudaGetErrorName(pending_err) << ": " << cudaGetErrorString(pending_err)
       << "), from earlier work outside this call; taken from the error slot and reported here, nothing was launched";
    throw std::runtime_error(os.str());
  }
}

RetainUntilComplete::~RetainUntilComplete() {
  PendingState& st = pending();
  PendingEntry e;
  e.device = device_;
  e.bytes = bytes_;  // V3-02: summed with checked addition in the constructor; no addition here
  for (Operand* op : ops_) {
    if (op == nullptr || !op->present()) continue;
    owners_.push_back(op->take_owner());  // reserved: no allocation
  }
  e.owners = std::move(owners_);
  cudaError_t r = cudaSuccess;
  {
    QuietDeviceGuard g(device_);
    r = !g.ok() ? g.error()
                : (test_hooks::take(test_hooks::kCudaEventRecord) ? kTestInjectedCudaError
                                                                  : cudaEventRecord(event_, cudaStreamLegacy));
  }
  if (r == cudaSuccess) {
    e.event = event_;
    // N3: the push may allocate (a new deque node); this destructor is noexcept, so a failure must be caught here
    // and turned into the synchronous fallback below (round-2 review: bad_alloc on the release path terminated).
    try {
      if (test_hooks::take(test_hooks::kAllocPendingPush)) throw std::bad_alloc();
      st.q.push_back(std::move(e));
      note_push(st, st.q.back());
    } catch (...) {
      ++st.push_alloc_failures;
      r = cudaErrorMemoryAllocation;  // e was not moved (push_back's strong guarantee): use the fallback below
      e.event = nullptr;
    }
  } else {
    cudaGetLastError();
  }
  if (r != cudaSuccess) {
    // Fallback: no queued event, so synchronize this device's legacy stream; release on success, leak on failure.
    ++st.fallback_syncs;
    const cudaError_t sr = sync_device_legacy(device_);
    if (sr == cudaSuccess) {
      recycle_event(st, device_, event_);
      ++st.released_entries;
      st.released_managers += e.owners.size();
      for (ManagedOwner& o : e.owners) o.reset();  // N4: a throwing deleter is contained per manager
    } else {
      const unsigned long long leaked = leak_entry(st, e);
      note_release_failure(st, device_, sr, leaked);  // N1: raised by the next synchronize()
    }
    return;
  }
  if (pending_over_bound(st)) {  // V3-02: overflow-safe (saturating total), exact at the boundary
    ++st.bound_drains;
    const SyncReport rep = sync_and_release_all_impl();
    if (rep.failed()) note_release_failure(st, rep.first_device(), rep.first_error(), rep.leaked_managers);  // N1
  }
}

void drain_completed() noexcept {
  PendingState& st = pending();
  while (!st.q.empty()) {
    const cudaError_t r = cudaEventQuery(st.q.front().event);
    if (r != cudaSuccess) {
      // cudaErrorNotReady: still pending (FIFO: later entries wait too). CUDA 12.6 does not leave NotReady in the
      // last-error slot (ledger receipt; Programming Guide 3.2.12: NotReady "is not considered an error"); clear it
      // defensively if a runtime ever does. Any other error (a broken context) leaves the entry for
      // synchronize_and_release_all, which re-queries it and reports and leaks it if the device cannot synchronize.
      // P1T-FIX4 (sweep): that other error's slot entry is consumed here (62c8d9f left it, so the next, unrelated
      // launch check reported it as its own); the entry itself still carries the failure to synchronize().
      if (r == cudaErrorNotReady) {
        if (cudaPeekAtLastError() == cudaErrorNotReady) cudaGetLastError();
      } else {
        cudaGetLastError();
      }
      return;
    }
    PendingEntry e = std::move(st.q.front());
    st.q.pop_front();
    note_pop(st, e);
    release_entry(st, e);
  }
}

// P1T-FIX4, round-4 review finding V4-04 (MINOR). 62c8d9f never looked at the calling thread's CUDA last-error slot:
// after a synchronous launch refusal (cudaErrorInvalidConfiguration, a NULL kernel, ...) that queued no work, every
// stream sync succeeded and synchronize() returned success with the error still pending. Now:
//  * FIRST, before any query, sync or release can replace it, the existing error is TAKEN (cudaGetLastError) and held;
//  * the drain runs exactly as before (sync every relevant device's legacy stream, release what completed, leak what
//    has no completion proof); a pending launch refusal does not stop the drain;
//  * then RuntimeError is raised naming the pre-existing error (and any drain failure), even if every sync succeeded.
// Clear (take), not peek (decision justified against the CUDA semantics, receipts artifacts/p1t_fix4/):
//  * non-sticky errors (a refused launch, a failed API call) stay in the slot until someone takes them, survive
//    successful calls, and are OVERWRITTEN by the next failing call (measured 98 -> 101). This function reports the
//    error, so it consumes it: peeking would leave it for the next unrelated launch check of this shared
//    libcudart.so.12 instance (any tc_signal op, or core's cuda_check_last) to report a second time as its own; and a
//    failing sync here would overwrite it anyway, so a peek could not preserve it reliably.
//  * sticky errors (a device-side fault) live in the CONTEXT: taking the slot entry resets only the slot (measured:
//    710, then 0), every later call that touches the context still returns the error, so nothing is hidden; the drain's
//    syncs fail and the pending operands are leaked and reported, as before.
//  * the runtime's own no-device state (cudaErrorNoDevice) is not reset by cudaGetLastError at all (measured).
// Prior art: NVIDIA CUDA 12.6 cudaGetLastError / cudaPeekAtLastError ("returns this variable and resets it" versus
// "does not reset"; Programming Guide 3.2.12) and cudaStreamSynchronize; PyTorch's convention of consuming the slot when
// it reports (C10_CUDA_KERNEL_LAUNCH_CHECK = C10_CUDA_CHECK(cudaGetLastError()); C10_CUDA_CHECK_WARN, torch 2.11
// headers). Taken: report-and-consume. Ours: capturing at synchronize() entry and still draining before raising.
void synchronize_and_release_all() {
  PendingState& st = pending();
  const cudaError_t prior = cudaGetLastError();
  if (prior != cudaSuccess) {
    if (st.prior_unreported == 0) st.prior_error = prior;
    ++st.prior_unreported;
  }
  const SyncReport rep = sync_and_release_all_impl();
  if (!rep.failed() && st.unreported_count == 0 && st.unraisable_unreported == 0 && st.prior_unreported == 0) return;
  // N1: any CUDA failure is an error of synchronize(), never a success. N3: building the message may allocate; a
  // bad_alloc here propagates as MemoryError (an error, not a success, and not a terminate). The earlier release-path
  // failures are cleared only once their message exists, so they are raised exactly once.
  if (test_hooks::take(test_hooks::kAllocErrorMessage)) throw std::bad_alloc();
  std::ostringstream os;
  os << "tc_signal.synchronize: CUDA error, so the release is not reported as a success:";
  const char* sep = " ";
  if (st.prior_unreported > 0) {
    os << sep << "a CUDA error was already pending in this thread when synchronize() was called ("
       << cudaGetErrorName(st.prior_error) << ": " << cudaGetErrorString(st.prior_error)
       << "), from earlier work (for example a refused kernel launch), not from the release; taken from the error slot "
          "and reported here";
    sep = "; ";
  }
  if (rep.query_failed) {
    os << sep << "the current-device query cudaGetDevice failed (" << cudaGetErrorName(rep.query_error) << ": "
       << cudaGetErrorString(rep.query_error) << ")";
    sep = "; ";
  }
  for (int i = 0; i < rep.n; ++i) {
    os << sep << "the legacy stream of device " << rep.dev[i] << " could not be synchronized ("
       << cudaGetErrorName(rep.err[i]) << ": " << cudaGetErrorString(rep.err[i]) << ")";
    sep = "; ";
  }
  if (rep.unnamed > 0) os << sep << rep.unnamed << " more device failure(s)";
  if (rep.leaked_managers > 0)
    os << "; " << rep.leaked_managers
       << " consumed operand manager(s) still pending there were deliberately leaked, not released";
  if (st.unreported_count > 0) {
    os << sep << st.unreported_count << " earlier release-path failure(s) since the last synchronize, the first on device "
       << st.unreported_device << " (" << cudaGetErrorName(st.unreported_error) << ": "
       << cudaGetErrorString(st.unreported_error) << "), deliberately leaked " << st.unreported_leaked
       << " operand manager(s)";
    sep = "; ";
  }
  if (st.unraisable_unreported > 0) {
    os << sep << st.unraisable_unreported
       << " CUDA failure(s) since the last synchronize on paths that cannot raise and leak no operand, the first: "
       << st.unraisable_what << " on device " << st.unraisable_device << " (" << cudaGetErrorName(st.unraisable_error)
       << ": " << cudaGetErrorString(st.unraisable_error) << ")";
  }
  // std::ostream swallows a bad_alloc from its buffer and only sets badbit (a truncated message would follow): report
  // the allocation failure itself instead.
  if (!os) throw std::bad_alloc();
  std::string msg = os.str();
  st.unreported_count = 0;
  st.unreported_leaked = 0;
  st.unreported_device = -1;
  st.unreported_error = cudaSuccess;
  st.unraisable_unreported = 0;
  st.unraisable_what = "";
  st.unraisable_device = -1;
  st.unraisable_error = cudaSuccess;
  st.prior_unreported = 0;
  st.prior_error = cudaSuccess;
  throw std::runtime_error(msg);
}

py::dict pending_stats(bool reset_high_water) {
  PendingState& st = pending();
  py::dict d;
  d["entries"] = static_cast<long long>(st.q.size());
  d["managers"] = st.managers;
  d["bytes"] = st.bytes;
  d["high_water_entries"] = st.high_water_entries;
  d["high_water_managers"] = st.high_water_managers;
  d["high_water_bytes"] = st.high_water_bytes;
  d["retained_entries"] = st.retained_entries;
  d["released_entries"] = st.released_entries;
  d["released_managers"] = st.released_managers;
  d["bound_drains"] = st.bound_drains;
  d["fallback_syncs"] = st.fallback_syncs;
  d["leaked_managers"] = st.leaked_managers;
  d["max_entries"] = tuning::kPendingMaxEntries;
  d["max_bytes"] = tuning::kPendingMaxBytes;
  // P1T-FIX2 counters.
  d["deleter_exceptions"] = deleter_exceptions();           // N4
  d["push_alloc_failures"] = st.push_alloc_failures;        // N3
  d["event_recycle_failures"] = st.event_recycle_failures;  // N3
  d["bytes_saturated"] = st.bytes_saturated;                // P1T-FIX3 V3-02
  d["bytes_saturations"] = st.bytes_saturations;            // P1T-FIX3 V3-02
  d["release_errors"] = st.release_errors;                  // N1
  d["unreported_release_errors"] = st.unreported_count;     // N1
  d["device_restore_failures"] = st.device_restore_failures;  // P1T-FIX4 (sweep): caller-device restores that failed
  d["event_destroy_failures"] = st.event_destroy_failures;    // P1T-FIX4 (sweep)
  d["unreported_unraisable_errors"] = st.unraisable_unreported;  // P1T-FIX4: raised by the next synchronize()
  if (reset_high_water) {
    st.high_water_entries = static_cast<long long>(st.q.size());
    st.high_water_managers = st.managers;
    st.high_water_bytes = st.bytes;
  }
  return d;
}

}  // namespace tcs
