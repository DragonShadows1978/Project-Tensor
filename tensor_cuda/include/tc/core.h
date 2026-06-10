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
#include <tuple>
#include <vector>

namespace tc {

constexpr int TC_MAX_DIMS = 8;

// ----------------------------------------------------------------------- dtype
enum class DType : int8_t { Float32 = 0, Float16 = 1, Int64 = 2, Bool = 3, Uint8 = 4, BFloat16 = 5 };

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
  bool pooled = false;  // allocated via cudaMallocAsync (small) vs raw cudaMalloc (large)

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
  U_RECIP, U_ABS, U_SIGN, U_SIN, U_COS,
  U_TAN, U_ASIN, U_ACOS, U_ATAN, U_SINH, U_COSH,
  U_LOG2, U_LOG10, U_FLOOR, U_CEIL, U_ROUND, U_ISNAN, U_ISINF, U_ISFINITE,
};
NDArray ew_unary(const NDArray& a, int op);
NDArray ew_pow(const NDArray& a, double exponent);
NDArray ew_clamp(const NDArray& a, double lo, double hi);  // clamp to [lo,hi]
NDArray ew_nan_to_num(const NDArray& a, double nan, double posinf, double neginf);

// Comparisons. op: 0=gt 1=ge 2=lt 3=le 4=eq 5=ne. Result is same dtype, 0/1.
NDArray compare(const NDArray& a, const NDArray& b, int op);     // broadcasting
NDArray compare_scalar(const NDArray& a, double s, int op);
// Select: cond != 0 ? x : y (all NumPy-broadcast together).
NDArray where_nd(const NDArray& cond, const NDArray& x, const NDArray& y);

// Reductions. axes empty => reduce all. Returns reduced array (keepdim aware).
NDArray reduce_sum(const NDArray& a, const std::vector<int>& axes, bool keepdim);
NDArray reduce_max(const NDArray& a, const std::vector<int>& axes, bool keepdim);
NDArray reduce_min(const NDArray& a, const std::vector<int>& axes, bool keepdim);
NDArray reduce_prod(const NDArray& a, const std::vector<int>& axes, bool keepdim);
// Sum `a` down to `target` shape (NumPy-broadcast reduction); used by autograd.
NDArray reduce_to(const NDArray& a, const Shape& target);

// argmax/argmin along a single axis (returns int64, axis removed).
NDArray reduce_arg(const NDArray& a, int axis, bool is_max);
// Cumulative sum along `axis` (same shape).
NDArray cumsum_nd(const NDArray& a, int axis);
// gather along `dim` using int64 `index` (index.shape == output shape).
NDArray gather_nd(const NDArray& a, int dim, const NDArray& index);
// scatter-add `src` into a zeroed `shape` tensor along `dim` by `index` (gather bwd).
NDArray scatter_add_nd(const Shape& shape, DType dtype, int dim,
                       const NDArray& index, const NDArray& src);
// reverse `a` along the given dims.
NDArray flip_nd(const NDArray& a, const std::vector<int>& dims);
// top-k along the last axis -> (values same dtype, indices int64), last dim = k.
std::tuple<NDArray, NDArray> topk_nd(const NDArray& a, int k, bool largest);

// Shape ops.
NDArray transpose2d_last(const NDArray& a);            // swap last two dims
NDArray permute(const NDArray& a, const std::vector<int>& dims);
NDArray broadcast_to(const NDArray& a, const Shape& shape);

// Concatenate along `dim`; slice [start, start+len) along `dim` (cat backward).
NDArray cat_nd(const std::vector<NDArray>& arrs, int dim);
NDArray slice_nd(const NDArray& a, int dim, int64_t start, int64_t len);
// Scatter `small` into a zeroed `big_shape` tensor at [start..) along dim (slice backward).
NDArray pad_into(const NDArray& small, const Shape& big_shape, int dim, int64_t start);

// Batched matmul over leading dims; last two dims are (M,K)x(K,N). cuBLAS.
// trans_b reads b as (N,K) row-major via OP_T (no transpose copy); alpha is
// folded into the GEMM (applied in the fp32 accumulator, before the 16-bit store).
NDArray matmul(const NDArray& a, const NDArray& b, float alpha = 1.f, bool trans_b = false);

// Enable/disable the transients pool (call AFTER weight loading; see Storage
// in kernels.cu — persistents must stay raw or they pin pool chunks at walls).
void set_alloc_pooling(bool enabled);

// Fused bottom-right-aligned causal softmax over the last dim of (...,L,S)
// scores (S >= L). Masked columns are never read; exact-zero like the eager
// -1e4-bias path. Inference-only at the Tensor level.
NDArray causal_softmax(const NDArray& scores);

// Fused RMSNorm over the last dim: out = x * rsqrt(mean(x^2) + eps) * w, fp32
// accumulate, single kernel + single output alloc (vs the 9-op chain).
// w must be fp32; out_dtype is typically x's dtype.
NDArray rms_norm(const NDArray& x, const NDArray& w, double eps, DType out_dtype);

// INT4 group-quantized linear: y = x @ dequant(W)^T.
//   x       : (..., K) fp16/fp32 activations (K == in_features)
//   packed  : (N, K/2) uint8 — two 4-bit weights per byte, even=low nibble,
//             odd=high nibble; N == out_features
//   scales  : (N, K/group_size) fp16 per-group scale
//   zeros   : (N, K/group_size) fp16 per-group zero point (min)
// Dequant rule mirrors the reference QuantizedLinear: w = q*scale + zero, with
// q the 4-bit code in [0,15]. Output dtype = x.dtype, shape (..., N).
NDArray int4_dequant(const NDArray& packed, const NDArray& scales,
                     const NDArray& zeros, int group_size, DType out_dtype);
NDArray int4_linear(const NDArray& x, const NDArray& packed,
                    const NDArray& scales, const NDArray& zeros, int group_size);
// Fused dequant-GEMM variant: same result as int4_linear but dequantizes the
// int4 weight into shared-memory tiles inside the GEMM, avoiding the full (K,N)
// fp16 weight transient. Opt-in (a custom GEMM may lose to cuBLAS at large N).
NDArray int4_linear_fused(const NDArray& x, const NDArray& packed,
                          const NDArray& scales, const NDArray& zeros, int group_size);

// Fused sparse APA-Quant attention. q,k,kq,v: (B,H,L,D)/(B,H,S,D). For each
// query row, the refine threshold is built from the quantized (bulk) scores;
// the full-precision dot is computed ONLY for keys whose |bulk| >= threshold
// (mean+zthr*std), the rest keep their quantized score. Online softmax over the
// resulting scores. Never materializes the L x S score matrix. Inference only.
NDArray apa_selective_attention(const NDArray& q, const NDArray& k,
                                const NDArray& kq, const NDArray& v,
                                float scale, float zthr, bool is_causal);

// Fused APA blend+softmax over precomputed score matrices (each (..., S)) from
// cuBLAS: per row, thr = mean(|rank|)+zthr*std(|rank|); score = |rank|>=thr ?
// rank : bulk; returns softmax(score) weights. row_smax (int32, one per row) or
// null gives the causal valid-key count per row. Replaces the abs/mean/std/where/
// softmax op chain with a single launch.
NDArray apa_blend_softmax(const NDArray& bulk, const NDArray& rank,
                          float zthr, const NDArray* row_smax);

// Fill / compare helpers.
NDArray ge_scalar(const NDArray& a, double s);  // (a >= s) as same dtype 0/1

// APA quantization: per-element searchsorted against `boundaries` (sorted)
// followed by a gather from `codebook` (len = boundaries+1). Same dtype out.
NDArray apa_quantize_gather(const NDArray& rotated, const NDArray& boundaries,
                            const NDArray& codebook);

// Conv helpers (NCHW). im2col -> (N, C*kh*kw, OH*OW); col2im scatter-adds back.
NDArray im2col(const NDArray& x, int kh, int kw, int sh, int sw, int ph, int pw);
NDArray col2im(const NDArray& cols, const Shape& x_shape, int kh, int kw,
               int sh, int sw, int ph, int pw);
// Pooling (NCHW). maxpool writes the flat argmax (int64) for backward.
NDArray avgpool2d(const NDArray& x, int kh, int kw, int sh, int sw, int ph, int pw);
NDArray avgpool2d_bwd(const NDArray& g, const Shape& x_shape, int kh, int kw,
                      int sh, int sw, int ph, int pw);
NDArray maxpool2d(const NDArray& x, int kh, int kw, int sh, int sw, int ph, int pw,
                  NDArray& argmax_out);
NDArray maxpool2d_bwd(const NDArray& g, const NDArray& argmax, const Shape& x_shape);

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
void lion_step(NDArray& param, const NDArray& grad, NDArray& m,
               double lr, double b1, double b2, double weight_decay);
void radam_step(NDArray& param, const NDArray& grad, NDArray& m, NDArray& v,
                double lr, double b1, double b2, double eps, double bc1, double bc2,
                double rect, bool rectified, double weight_decay, bool decoupled);
void rmsprop_step(NDArray& param, const NDArray& grad, NDArray& sq_avg,
                  double lr, double alpha, double eps, double weight_decay);
void adagrad_step(NDArray& param, const NDArray& grad, NDArray& acc,
                  double lr, double eps, double weight_decay);
// param.data += alpha * other  (in place); param.data *= s (in place).
void axpy_(NDArray& param, const NDArray& other, double alpha);
void scale_(NDArray& param, double s);

// CUDA bookkeeping.
void cuda_sync();
void cuda_check_last(const char* where);

// Release all device blocks held idle by the caching allocator back to the
// driver. Live tensors are unaffected. Call under memory pressure or to measure
// true steady-state usage.
void empty_cache();

}  // namespace tc
