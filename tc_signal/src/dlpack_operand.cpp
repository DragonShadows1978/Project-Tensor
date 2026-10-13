// DLPack consumer implementation. Prior art and design notes: see dlpack_operand.h.
#include "dlpack_operand.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <deque>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "checked_math.h"
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
  for (int i = 0; i < t->ndim; ++i)
    if (t->shape[i] <= 0) refuse_value(role, "empty operand (shape " + shape_str(shape_of(t)) + ")");
  // F03: checked element count and fp32 byte total (every extent is >= 1 here), before the stride walk below
  // multiplies extents and before any later byte range is formed.
  int64_t numel = 1;
  for (int i = 0; i < t->ndim; ++i) {
    if (__builtin_mul_overflow(numel, t->shape[i], &numel))
      refuse_value(role, "element count of shape " + shape_str(shape_of(t)) + " overflows int64");
  }
  int64_t nbytes = 0;
  if (__builtin_mul_overflow(numel, static_cast<int64_t>(sizeof(float)), &nbytes))
    refuse_value(role, "byte size of shape " + shape_str(shape_of(t)) + " (fp32) overflows int64");
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

void ManagedOwner::reset() noexcept {
  // DLPack consumer rule: when done, call the producer's deleter (it may be NULL). Called with the GIL held (every
  // tc_signal entry point runs under the GIL), so a deleter that touches Python objects is safe.
  if (legacy != nullptr) {
    DLManagedTensor* m = legacy;
    legacy = nullptr;
    if (m->deleter) m->deleter(m);
  }
  if (versioned != nullptr) {
    DLManagedTensorVersioned* m = versioned;
    versioned = nullptr;
    if (m->deleter) m->deleter(m);
  }
}

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
// OFF the list before its deleters run, and nothing iterates the list across a deleter call.
namespace {

struct PendingEntry {
  cudaEvent_t event = nullptr;   // recorded on `device`'s legacy stream right after the call's last launch
  int device = -1;
  int64_t bytes = 0;             // sum of the fp32 operand sizes this entry keeps alive
  std::vector<ManagedOwner> owners;
};

struct PendingState {
  std::deque<PendingEntry> q;
  int64_t bytes = 0;
  long long managers = 0;
  long long high_water_entries = 0;
  long long high_water_managers = 0;
  int64_t high_water_bytes = 0;
  unsigned long long retained_entries = 0;   // entries ever pushed
  unsigned long long released_entries = 0;
  unsigned long long released_managers = 0;
  unsigned long long bound_drains = 0;       // a push exceeded a bound -> synchronize + release everything
  unsigned long long fallback_syncs = 0;     // the event could not be recorded -> the legacy stream was synchronized
  unsigned long long leaked_managers = 0;    // no completion proof (broken context): deliberately not released
  std::unordered_map<int, std::vector<cudaEvent_t>> event_pool;   // completed events, reused per device
};

// Never destroyed: no static destructor may call a producer deleter at process exit (Python may be gone by then).
PendingState& pending() {
  static PendingState* state = new PendingState();
  return *state;
}

// Makes `device` current for its scope and restores the caller's device; never throws (ok() reports failure, and a
// failed switch clears the runtime's last-error slot so a later, unrelated launch check does not report it).
class QuietDeviceGuard {
 public:
  explicit QuietDeviceGuard(int device) noexcept {
    if (cudaGetDevice(&prev_) != cudaSuccess) {
      cudaGetLastError();
      ok_ = false;
      return;
    }
    if (prev_ != device) {
      if (cudaSetDevice(device) != cudaSuccess) {
        cudaGetLastError();
        ok_ = false;
        return;
      }
      switched_ = true;
    }
  }
  ~QuietDeviceGuard() {
    if (switched_) cudaSetDevice(prev_);
  }
  QuietDeviceGuard(const QuietDeviceGuard&) = delete;
  QuietDeviceGuard& operator=(const QuietDeviceGuard&) = delete;
  bool ok() const { return ok_; }

 private:
  int prev_ = 0;
  bool ok_ = true;
  bool switched_ = false;
};

void recycle_event(PendingState& st, int device, cudaEvent_t ev) noexcept {
  if (ev == nullptr) return;
  try {
    st.event_pool[device].push_back(ev);
  } catch (...) {
    cudaEventDestroy(ev);
  }
}

void note_push(PendingState& st, const PendingEntry& e) {
  st.bytes += e.bytes;
  st.managers += static_cast<long long>(e.owners.size());
  ++st.retained_entries;
  st.high_water_entries = std::max<long long>(st.high_water_entries, static_cast<long long>(st.q.size()));
  st.high_water_managers = std::max(st.high_water_managers, st.managers);
  st.high_water_bytes = std::max(st.high_water_bytes, st.bytes);
}

void note_pop(PendingState& st, const PendingEntry& e) {
  st.bytes -= e.bytes;
  st.managers -= static_cast<long long>(e.owners.size());
}

// Runs the entry's deleters (they may run Python and re-enter tc_signal; the entry is already off the list).
void release_entry(PendingState& st, PendingEntry& e) noexcept {
  recycle_event(st, e.device, e.event);
  e.event = nullptr;
  ++st.released_entries;
  st.released_managers += e.owners.size();
  for (ManagedOwner& o : e.owners) o.reset();
}

void leak_entry(PendingState& st, PendingEntry& e) noexcept {
  for (ManagedOwner& o : e.owners) {
    if (!o.empty()) ++st.leaked_managers;
    o.leak();
  }
  // The event is not recycled either: it may belong to a broken context.
  e.event = nullptr;
}

// Takes the whole list off, synchronizes the legacy stream of the current device and of every device in it, then
// releases each entry whose device synchronized and leaks the others. Entries pushed by deleters while this runs (a
// re-entrant tc_signal call) land on the fresh list and are not touched here. Returns the failed devices' errors.
std::vector<std::pair<int, cudaError_t>> sync_and_release_all_impl() noexcept {
  PendingState& st = pending();
  std::deque<PendingEntry> taken;
  taken.swap(st.q);
  st.bytes = 0;
  st.managers = 0;
  std::vector<std::pair<int, cudaError_t>> failed;
  try {
    std::set<int> devices;
    int cur = -1;
    if (cudaGetDevice(&cur) == cudaSuccess)
      devices.insert(cur);
    else
      cudaGetLastError();
    for (const PendingEntry& e : taken) devices.insert(e.device);
    for (int d : devices) {
      QuietDeviceGuard g(d);
      cudaError_t r = g.ok() ? cudaStreamSynchronize(cudaStreamLegacy) : cudaErrorInvalidDevice;
      if (r != cudaSuccess) {
        cudaGetLastError();
        failed.emplace_back(d, r);
      }
    }
  } catch (...) {
    // Only bad_alloc can land here: without a completion proof, leak everything taken.
    for (PendingEntry& e : taken) leak_entry(st, e);
    failed.emplace_back(-1, cudaErrorMemoryAllocation);
    return failed;
  }
  while (!taken.empty()) {
    PendingEntry e = std::move(taken.front());
    taken.pop_front();
    bool ok = true;
    for (const auto& f : failed)
      if (f.first == e.device) ok = false;
    if (ok)
      release_entry(st, e);
    else
      leak_entry(st, e);
  }
  return failed;
}

cudaEvent_t acquire_event(int device) {
  PendingState& st = pending();
  auto it = st.event_pool.find(device);
  if (it != st.event_pool.end() && !it->second.empty()) {
    cudaEvent_t ev = it->second.back();
    it->second.pop_back();
    return ev;
  }
  QuietDeviceGuard g(device);
  cudaEvent_t ev = nullptr;
  cudaError_t r = g.ok() ? cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) : cudaErrorInvalidDevice;
  if (r != cudaSuccess) {
    cudaGetLastError();
    std::ostringstream os;
    os << "tc_signal: cannot create the completion event on CUDA device " << device << ": " << cudaGetErrorName(r)
       << ": " << cudaGetErrorString(r);
    throw std::runtime_error(os.str());
  }
  return ev;
}

}  // namespace

RetainUntilComplete::RetainUntilComplete(int device, std::initializer_list<Operand*> ops)
    : device_(device), ops_(ops) {
  owners_.reserve(ops_.size());
  event_ = acquire_event(device_);  // last: if it throws, nothing else is held and nothing has been launched
}

RetainUntilComplete::~RetainUntilComplete() {
  PendingState& st = pending();
  PendingEntry e;
  e.device = device_;
  for (Operand* op : ops_) {
    if (op == nullptr || !op->present()) continue;
    e.bytes += op->numel() * static_cast<int64_t>(sizeof(float));  // checked at construction (F03)
    owners_.push_back(op->take_owner());                           // reserved: no allocation
  }
  e.owners = std::move(owners_);
  cudaError_t r = cudaSuccess;
  {
    QuietDeviceGuard g(device_);
    r = g.ok() ? cudaEventRecord(event_, cudaStreamLegacy) : cudaErrorInvalidDevice;
  }
  if (r == cudaSuccess) {
    e.event = event_;
    try {
      st.q.push_back(std::move(e));
      note_push(st, st.q.back());
    } catch (...) {
      r = cudaErrorMemoryAllocation;  // e was not moved (push_back's strong guarantee): use the fallback below
      e.event = nullptr;
    }
  } else {
    cudaGetLastError();
  }
  if (r != cudaSuccess) {
    // Fallback: no event, so synchronize this device's legacy stream; release on success, leak on failure.
    ++st.fallback_syncs;
    bool synced = false;
    {
      QuietDeviceGuard g(device_);
      synced = g.ok() && cudaStreamSynchronize(cudaStreamLegacy) == cudaSuccess;
      if (!synced) cudaGetLastError();
    }
    if (synced) {
      recycle_event(st, device_, event_);
      ++st.released_entries;
      st.released_managers += e.owners.size();
      for (ManagedOwner& o : e.owners) o.reset();
    } else {
      leak_entry(st, e);
    }
    return;
  }
  if (static_cast<long long>(st.q.size()) > tuning::kPendingMaxEntries || st.bytes > tuning::kPendingMaxBytes) {
    ++st.bound_drains;
    sync_and_release_all_impl();
  }
}

void drain_completed() noexcept {
  PendingState& st = pending();
  while (!st.q.empty()) {
    const cudaError_t r = cudaEventQuery(st.q.front().event);
    if (r != cudaSuccess) {
      // cudaErrorNotReady: still pending (FIFO: later entries wait too). CUDA 12.6 does not leave NotReady in the
      // last-error slot (ledger receipt); clear it defensively if a runtime ever does. Any other error (a broken
      // context) leaves the entry for synchronize_and_release_all, which leaks it if the device cannot synchronize.
      if (r == cudaErrorNotReady && cudaPeekAtLastError() == cudaErrorNotReady) cudaGetLastError();
      return;
    }
    PendingEntry e = std::move(st.q.front());
    st.q.pop_front();
    note_pop(st, e);
    release_entry(st, e);
  }
}

void synchronize_and_release_all() {
  const auto failed = sync_and_release_all_impl();
  if (!failed.empty()) {
    std::ostringstream os;
    os << "tc_signal.synchronize: the legacy stream could not be synchronized on";
    for (const auto& f : failed)
      os << " device " << f.first << " (" << cudaGetErrorName(f.second) << ": " << cudaGetErrorString(f.second) << ")";
    os << "; the operands still pending there were deliberately leaked, not released";
    throw std::runtime_error(os.str());
  }
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
  if (reset_high_water) {
    st.high_water_entries = static_cast<long long>(st.q.size());
    st.high_water_managers = st.managers;
    st.high_water_bytes = st.bytes;
  }
  return d;
}

}  // namespace tcs
