// DLPack consumer implementation. Prior art and design notes: see dlpack_operand.h.
#include "dlpack_operand.h"

#include <cstring>
#include <sstream>

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
void validate_tensor(const DLTensor* t, const char* role, int ndim) {
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
  if (t->ndim != ndim) {
    std::ostringstream os;
    os << "expected " << ndim << " dimensions, got " << t->ndim;
    refuse_value(role, os.str());
  }
  for (int i = 0; i < t->ndim; ++i)
    if (t->shape[i] <= 0) refuse_value(role, "empty operand (shape " + shape_str(shape_of(t)) + ")");
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
  const uintptr_t p = reinterpret_cast<uintptr_t>(t->data) + static_cast<uintptr_t>(t->byte_offset);
  if (p % alignof(float) != 0) refuse_value(role, "data + byte_offset is not 4-byte aligned");
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
  validate_tensor(t_, role, ndim);
  data_ = reinterpret_cast<float*>(static_cast<char*>(t_->data) + t_->byte_offset);  // byte_offset honoured
}

Operand::~Operand() = default;

Operand::Operand(Operand&& o) noexcept : owner_(std::move(o.owner_)), t_(o.t_), data_(o.data_), role_(o.role_) {
  o.t_ = nullptr;
  o.data_ = nullptr;
}

Operand& Operand::operator=(Operand&& o) noexcept {
  if (this != &o) {
    owner_ = std::move(o.owner_);
    t_ = o.t_;
    data_ = o.data_;
    role_ = o.role_;
    o.t_ = nullptr;
    o.data_ = nullptr;
  }
  return *this;
}

int64_t Operand::numel() const {
  int64_t n = 1;
  for (int i = 0; i < t_->ndim; ++i) n *= t_->shape[i];
  return n;
}

void Operand::release() {
  owner_.reset();
  t_ = nullptr;
  data_ = nullptr;
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

}  // namespace tcs
