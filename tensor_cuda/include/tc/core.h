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
NDArray ew_pow(const NDArray& a, double exponent);

// Comparisons. op: 0=gt 1=ge 2=lt 3=le 4=eq 5=ne. Result is same dtype, 0/1.
NDArray compare(const NDArray& a, const NDArray& b, int op);     // broadcasting
NDArray compare_scalar(const NDArray& a, double s, int op);
// Select: cond != 0 ? x : y (all NumPy-broadcast together).
NDArray where_nd(const NDArray& cond, const NDArray& x, const NDArray& y);

// Reductions. axes empty => reduce all. Returns reduced array (keepdim aware).
NDArray reduce_sum(const NDArray& a, const std::vector<int>& axes, bool keepdim);
NDArray reduce_max(const NDArray& a, const std::vector<int>& axes, bool keepdim);
NDArray reduce_min(const NDArray& a, const std::vector<int>& axes, bool keepdim);
// Sum `a` down to `target` shape (NumPy-broadcast reduction); used by autograd.
NDArray reduce_to(const NDArray& a, const Shape& target);

// Shape ops.
NDArray transpose2d_last(const NDArray& a);            // swap last two dims
NDArray permute(const NDArray& a, const std::vector<int>& dims);
NDArray broadcast_to(const NDArray& a, const Shape& shape);

// Concatenate along `dim`; slice [start, start+len) along `dim` (cat backward).
NDArray cat_nd(const std::vector<NDArray>& arrs, int dim);
NDArray slice_nd(const NDArray& a, int dim, int64_t start, int64_t len);

// Batched matmul over leading dims; last two dims are (M,K)x(K,N). cuBLAS.
NDArray matmul(const NDArray& a, const NDArray& b);

// Fill / compare helpers.
NDArray ge_scalar(const NDArray& a, double s);  // (a >= s) as same dtype 0/1

// Embedding: gather rows of `weight` (V, ...) by int64 `idx` (any shape).
// Output shape = idx.shape ++ weight.shape[1:].
NDArray embedding_forward(const NDArray& weight, const NDArray& idx);
// Scatter-add `grad` back into a zeroed weight-shaped tensor (accumulated fp32).
NDArray embedding_backward(const NDArray& grad, const NDArray& idx,
                           const Shape& weight_shape, DType weight_dtype);

// In-place optimizer steps (compute in fp32, store in param dtype).
void sgd_step(NDArray& param, const NDArray& grad, NDArray& momentum_buf,
              double lr, double momentum, double weight_decay);
void adam_step(NDArray& param, const NDArray& grad, NDArray& m, NDArray& v,
               double lr, double b1, double b2, double eps, int64_t t,
               double weight_decay, bool decoupled);
// param.data += alpha * other  (in place); used by misc utilities.
void axpy_(NDArray& param, const NDArray& other, double alpha);

// CUDA bookkeeping.
void cuda_sync();
void cuda_check_last(const char* where);

}  // namespace tc
