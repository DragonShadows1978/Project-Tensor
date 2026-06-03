// Project Tensor — standalone CUDA tensor library (no PyTorch, no CuPy).
//
// core.h: device memory (Storage), the raw n-dimensional array (NDArray) that
// every CUDA kernel operates on, dtype/device descriptors, and the host-side
// op declarations. NDArray carries NO autograd state — it is the workhorse.
// Autograd lives one layer up (autograd.h).
//
// Phase 1 design choices (see ROADMAP.md):
//   * NDArrays are always contiguous, row-major. transpose/permute physically
//     materialize; reshape is metadata-only. This keeps kernels simple and
//     correct; strided views are a later optimization.
//   * Kernels are dtype-generic by computing in fp32 (load->float, store->T),
//     so fp32 and fp16 share one code path.
//   * Max rank is 8 (TC_MAX_DIMS), enough for NCHW + attention (B,H,L,D).

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tc {

constexpr int TC_MAX_DIMS = 8;

// ----------------------------------------------------------------------- dtype
enum class DType : int8_t { Float32 = 0, Float16 = 1, Int64 = 2, Bool = 3 };

size_t dtype_size(DType dt);
const char* dtype_name(DType dt);
DType dtype_from_string(const std::string& s);

// ---------------------------------------------------------------------- device
enum class DeviceType : int8_t { CPU = 0, CUDA = 1 };

struct Device {
  DeviceType type = DeviceType::CUDA;
  int index = 0;
  bool is_cuda() const { return type == DeviceType::CUDA; }
  std::string str() const;
  static Device from_string(const std::string& s);
};

using Shape = std::vector<int64_t>;

int64_t numel_of(const Shape& shape);
Shape contiguous_strides(const Shape& shape);

// --------------------------------------------------------------------- Storage
// Reference-counted device (or host) memory buffer.
struct Storage {
  void* ptr = nullptr;
  size_t nbytes = 0;
  Device device;

  Storage(size_t nbytes, Device device);
  ~Storage();
  Storage(const Storage&) = delete;
  Storage& operator=(const Storage&) = delete;
};

// --------------------------------------------------------------------- NDArray
// A view-able handle over a Storage: shape + dtype + element offset. In Phase 1
// the data is contiguous; shape/strides are kept in sync by contiguous_strides.
class NDArray {
 public:
  std::shared_ptr<Storage> storage;
  Shape shape;
  DType dtype = DType::Float32;
  Device device;
  int64_t offset = 0;  // in elements

  NDArray() = default;
  NDArray(const Shape& shape, DType dtype, Device device);

  static NDArray empty(const Shape& shape, DType dtype, Device device);
  static NDArray zeros(const Shape& shape, DType dtype, Device device);
  static NDArray full(const Shape& shape, double value, DType dtype, Device device);

  int64_t numel() const { return numel_of(shape); }
  int ndim() const { return static_cast<int>(shape.size()); }
  bool defined() const { return storage != nullptr; }
  void* data_ptr() const;

  // Host<->device transfer (host buffer is contiguous, matching `dtype`).
  static NDArray from_host(const void* src, const Shape& shape, DType dtype, Device device);
  void to_host(void* dst) const;

  NDArray to(Device device) const;
  NDArray astype(DType dtype) const;
  NDArray clone() const;
  NDArray reshape(const Shape& new_shape) const;  // metadata-only (contiguous)
  NDArray contiguous() const { return *this; }     // Phase 1: always contiguous
};

// ------------------------------------------------------------- raw CUDA ops
// All return freshly-allocated contiguous NDArrays. Declared here, launched in
// src/kernels.cu and src/matmul.cu.

// Elementwise binary with NumPy broadcasting. op: 0=add 1=sub 2=mul 3=div.
NDArray ew_binary(const NDArray& a, const NDArray& b, int op);
NDArray ew_scalar(const NDArray& a, double scalar, int op, bool scalar_lhs);

// Elementwise unary. op enum mirrors UnaryOp below.
enum UnaryOp {
  U_NEG, U_EXP, U_LOG, U_SQRT, U_RELU, U_SIGMOID, U_TANH, U_GELU, U_SILU,
  U_RECIP, U_ABS, U_SIGN,
};
NDArray ew_unary(const NDArray& a, int op);

// Reductions. axes empty => reduce all. Returns reduced array (keepdim aware).
NDArray reduce_sum(const NDArray& a, const std::vector<int>& axes, bool keepdim);
NDArray reduce_max(const NDArray& a, const std::vector<int>& axes, bool keepdim);
// Sum `a` down to `target` shape (NumPy-broadcast reduction); used by autograd.
NDArray reduce_to(const NDArray& a, const Shape& target);

// Shape ops.
NDArray transpose2d_last(const NDArray& a);            // swap last two dims
NDArray permute(const NDArray& a, const std::vector<int>& dims);

// Batched matmul over leading dims; last two dims are (M,K)x(K,N). cuBLAS.
NDArray matmul(const NDArray& a, const NDArray& b);

// Fill / compare helpers.
NDArray ge_scalar(const NDArray& a, double s);  // (a >= s) as same dtype 0/1

// CUDA bookkeeping.
void cuda_sync();
void cuda_check_last(const char* where);

}  // namespace tc
