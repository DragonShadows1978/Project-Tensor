// DLPack consumer side of tc_signal: turn any DLPack CUDA producer (or a bare capsule) into a validated fp32 operand.
//
// Prior art: DLPack (dmlc/dlpack, 2017-; header v1.3 vendored in include/dlpack/dlpack.h) for the DLManagedTensor /
// DLManagedTensorVersioned ABI and the capsule rules ("dltensor" / "dltensor_versioned", renamed to "used_*" on
// consumption, consumer calls the deleter); the Python array API standard's data-interchange protocol
// (`__dlpack__(stream=, max_version=, copy=)`, Consortium for Python Data API Standards, 2021-2023) for the stream
// argument and the max_version/TypeError fallback, which NumPy's and PyTorch's `from_dlpack` also implement. Ours: the
// two-phase use (peek shapes for sizing without consuming, then consume and validate everything right before the launch),
// and the validation set (device, dtype, compact strides with size-1 dimensions free, byte_offset, alignment, output
// overlap, read-only/copied refusal for written operands).
#pragma once

#include <dlpack/dlpack.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <string>
#include <vector>

namespace tcs {
namespace py = pybind11;

// A DLPack capsule for `obj`: `obj` itself when it is a capsule, else `obj.__dlpack__(stream=1, max_version=(1, 3)
// [, copy=False])`, falling back to `obj.__dlpack__(stream=1)` when the producer rejects the newer keywords with a
// TypeError (pre-2023 producers). `writable` adds copy=False: a written operand must alias the producer's memory.
py::object to_capsule(py::handle obj, const char* role, bool writable);

// The DLTensor inside an unconsumed capsule (the capsule keeps ownership). Raises ValueError for a consumed capsule,
// TypeError for a capsule of another kind.
const DLTensor* peek(py::handle capsule, const char* role);

// Device / dtype / layout checks shared by peek-time and consume-time validation (raises TypeError/ValueError).
void validate_tensor(const DLTensor* t, const char* role, int ndim);

// Owner of one consumed managed tensor (legacy or versioned). Calls the producer's deleter exactly once, in reset() or
// in the destructor. It is a separate member so that a throw from Operand's constructor (after the capsule has been
// consumed) still runs it: C++ destroys fully constructed members when a constructor throws.
struct ManagedOwner {
  DLManagedTensor* legacy = nullptr;
  DLManagedTensorVersioned* versioned = nullptr;
  ManagedOwner() = default;
  ~ManagedOwner() { reset(); }
  ManagedOwner(ManagedOwner&& o) noexcept;
  ManagedOwner& operator=(ManagedOwner&& o) noexcept;
  ManagedOwner(const ManagedOwner&) = delete;
  ManagedOwner& operator=(const ManagedOwner&) = delete;
  void reset() noexcept;
};

// One consumed DLPack operand. Owns the managed tensor (through ManagedOwner) and calls the producer's deleter exactly
// once: in release() or in the destructor, so every exception path after consumption still returns the tensor to its
// producer.
class Operand {
 public:
  Operand() = default;
  Operand(py::handle capsule, const char* role, int ndim, bool writable);
  ~Operand();
  Operand(Operand&& o) noexcept;
  Operand& operator=(Operand&& o) noexcept;
  Operand(const Operand&) = delete;
  Operand& operator=(const Operand&) = delete;

  bool present() const { return t_ != nullptr; }
  float* data() const { return data_; }
  int64_t dim(int i) const { return t_->shape[i]; }
  int ndim() const { return t_->ndim; }
  int device() const { return t_->device.device_id; }
  int64_t numel() const;
  uintptr_t byte_begin() const { return reinterpret_cast<uintptr_t>(data_); }
  uintptr_t byte_end() const { return byte_begin() + static_cast<uintptr_t>(numel()) * sizeof(float); }
  const char* role() const { return role_; }
  void release();

 private:
  ManagedOwner owner_;
  const DLTensor* t_ = nullptr;
  float* data_ = nullptr;
  const char* role_ = "";
};

std::vector<int64_t> shape_of(const DLTensor* t);
std::string shape_str(const std::vector<int64_t>& s);

// Exact-shape check (ValueError).
void expect_shape(const Operand& op, const std::vector<int64_t>& want);
// Same CUDA ordinal for every present operand (ValueError); returns the ordinal.
int common_device(const std::vector<const Operand*>& ops);
// Byte ranges of `written` and `other` must not overlap (ValueError).
void expect_disjoint(const Operand& written, const Operand& other);

}  // namespace tcs
