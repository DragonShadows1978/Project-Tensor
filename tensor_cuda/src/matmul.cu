// matmul.cu: batched matrix multiply via cuBLAS.
//
// Row-major C(M,N) = A(M,K) @ B(K,N). cuBLAS is column-major, so we compute
// C^T = B^T @ A^T by swapping operands: a column-major gemm with (N,M,K) and
// ldc=N produces exactly the row-major (M,N) result. Batched over leading dims
// with a simple per-batch loop (strided-batched is a later optimization).

#include "tc/core.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <climits>
#include <cstring>
#include <stdexcept>
#include <string>

namespace tc {

namespace {
cublasHandle_t g_handle = nullptr;
cublasHandle_t handle() {
  if (!g_handle) cublasCreate(&g_handle);
  return g_handle;
}
}  // namespace

// ENG4 (ENG-MM1, M2-M5): the operand predicate shared by this forward and ops::matmul_backward (ops.cpp).
// It reads shapes, dtypes and devices only; it makes no CUDA call, so every refusal below happens before the
// output allocation, the cuBLAS handle and the GEMM.
//
// Prior art: the strided-batched GEMM contract (cuBLAS documentation, cublas<t>gemmStridedBatched and
// cublasGemmStridedBatchedEx: matrix i of B starts at B + i * strideB, so strideB = 0 reuses one matrix, and the
// caller must make every batchCount-strided matrix lie inside its buffer; the batched BLAS proposal, Dongarra et
// al., 2017 -- year and title unverified, lead to check: "batched BLAS standard strided").
// NumPy's matmul broadcasting rule (numpy.matmul docs: leading dimensions broadcast, the result takes the broadcast
// batch shape) is the reference for which pairs are meaningful; this predicate accepts only the subset the GEMM
// below computes as written (equal batch shapes, or a single B matrix) and adds no broadcasting. Ours: the class
// split (a)/(b)/(c) that matches the backward the engine already differentiates, the refusal set, and the shared
// placement. The INT_MAX checks mirror the fp32 backward's (ops.cpp, BWD) for the int arguments of the cuBLAS API.
namespace {
std::string shape_text(const Shape& s) {
  std::string t = "[";
  for (size_t i = 0; i < s.size(); ++i) {
    if (i) t += ", ";
    t += std::to_string(s[i]);
  }
  return t + "]";
}

std::string operand_text(const NDArray& a, const NDArray& b, bool trans_b) {
  return "A " + shape_text(a.shape) + " on " + a.device.str() + ", B " + shape_text(b.shape) + " on " +
         b.device.str() + ", trans_b=" + (trans_b ? "true" : "false");
}

MatmulOperandPlan refuse(MatmulOperandPlan p, MatmulOperandIssue issue, const std::string& what, const NDArray& a,
                         const NDArray& b, bool trans_b) {
  p.issue = issue;
  p.detail = what + " (" + operand_text(a, b, trans_b) + ")";
  return p;
}
}  // namespace

// ENG4-F3 Z1 (ENG4-REV3 F01): NDArray::zeros (kernels.cu, unchanged) discards cudaMemset's returned status, and F2
// read the thread's last-error slot afterwards instead, so a failed fill was seen only through the slot and an
// unrelated earlier error left pending there could be raised under the fill's label. This helper allocates exactly as
// NDArray::zeros does (the NDArray constructor), fills with cudaMemset (CUDA) or std::memset (host), and checks the
// fill's own return status; zero bytes make no fill and no CUDA call. It never reads or clears the slot.
// Prior art: the CUDA Runtime API's per-call return status, separate from its last-error slot (cudaGetLastError /
// cudaPeekAtLastError; CUDA 12.6 Runtime API); TensorCUDA's per-call status checks (ops.cpp check_backward_cuda).
// Ours: the helper and its two call sites.
NDArray zeros_checked(const Shape& shape, DType dtype, Device device, const char* where) {
  NDArray out(shape, dtype, device);
  const int64_t n = out.numel();
  if (n == 0) return out;
  const size_t bytes = static_cast<size_t>(n) * dtype_size(dtype);
  if (device.is_cuda()) {
    const cudaError_t status = cudaMemset(out.data_ptr(), 0, bytes);
    if (status != cudaSuccess)
      throw std::runtime_error(std::string(where) + ": cudaMemset failed: " + cudaGetErrorName(status) + ": " +
                               cudaGetErrorString(status));
  } else {
    std::memset(out.data_ptr(), 0, bytes);
  }
  return out;
}

MatmulOperandPlan matmul_operand_plan(const NDArray& a, const NDArray& b, bool trans_b) {
  MatmulOperandPlan p;
  const int nda = a.ndim(), ndb = b.ndim();
  if (nda < 2 || ndb < 2) return refuse(p, MatmulOperandIssue::Rank, "both operands need rank >= 2", a, b, trans_b);
  if (a.dtype != b.dtype)
    return refuse(p, MatmulOperandIssue::Dtype,
                  std::string("dtypes differ: A ") + dtype_name(a.dtype) + ", B " + dtype_name(b.dtype), a, b,
                  trans_b);
  // ENG4-F1 X1 (ENG4R1-F01): the GEMM below supports these three dtypes only. The check sits here, before the
  // forward allocates its output (a CUDA allocation for CUDA operands), instead of in the dispatch's final else.
  if (a.dtype != DType::Float32 && a.dtype != DType::Float16 && a.dtype != DType::BFloat16)
    return refuse(p, MatmulOperandIssue::UnsupportedDtype, std::string("operand dtype ") + dtype_name(a.dtype), a,
                  b, trans_b);
  // trans_b: C = alpha * A @ B^T with B stored row-major (N, K).
  p.m = a.shape[nda - 2];
  p.k = a.shape[nda - 1];
  p.kb = trans_b ? b.shape[ndb - 1] : b.shape[ndb - 2];
  p.n = trans_b ? b.shape[ndb - 2] : b.shape[ndb - 1];
  if (p.k != p.kb)
    return refuse(p, MatmulOperandIssue::InnerDim,
                  "inner dimensions differ: A's K " + std::to_string(p.k) + ", B's K " + std::to_string(p.kb), a, b,
                  trans_b);
  // M4: one device for both operands (a host and a CUDA operand, or two CUDA indices, are refused).
  if (a.device.type != b.device.type || a.device.index != b.device.index)
    return refuse(p, MatmulOperandIssue::Device, "operands are on different devices", a, b, trans_b);
  // M4: the cuBLAS API takes M, N, K, the leading dimensions (K or N) and the batch count as int.
  if (p.m > INT_MAX || p.n > INT_MAX || p.k > INT_MAX)
    return refuse(p, MatmulOperandIssue::Dim,
                  "GEMM dimension above INT_MAX: M " + std::to_string(p.m) + ", N " + std::to_string(p.n) + ", K " +
                      std::to_string(p.k),
                  a, b, trans_b);
  // Batch = product of A's leading dims. ENG4-F2 Y1 (ENG4R2-F01): a metadata reshape of a zero-element tensor can
  // carry leading extents whose plain product overflows int64 (for example 2^32 x 2^32 x 0), so the product is no
  // longer formed blindly. If any leading extent is 0 the batch is 0 (and passes the INT_MAX check); otherwise the
  // extents are multiplied one at a time and the pair is refused (BatchCount) as soon as the next product would
  // exceed INT_MAX, which keeps every intermediate product <= INT_MAX: no signed overflow is reachable. Values are
  // unchanged wherever the old product did not overflow (the same extents, multiplied in the same order).
  // Prior art: checked multiplication before overflow (CERT C INT32-C; GCC/Clang __builtin_mul_overflow, GCC 5,
  // 2015); the division-before-multiplication bound is the INT32-C form. Ours: the zero-first rule and the placement.
  for (int d = 0; d < nda - 2; ++d) p.out_shape.push_back(a.shape[d]);
  p.out_shape.push_back(p.m);
  p.out_shape.push_back(p.n);
  bool zero_leading = false;
  for (int d = 0; d < nda - 2; ++d) zero_leading = zero_leading || a.shape[d] == 0;
  p.batch = zero_leading ? 0 : 1;
  for (int d = 0; d < nda - 2 && !zero_leading; ++d) {
    const int64_t extent = a.shape[d];  // >= 1 here (no zero; NDArray extents are never negative)
    if (extent > INT_MAX / p.batch)
      return refuse(p, MatmulOperandIssue::BatchCount,
                    "GEMM batch count above INT_MAX: the product of A's leading extents exceeds " +
                        std::to_string(INT_MAX) + " at leading dimension " + std::to_string(d) + " (" +
                        std::to_string(p.batch) + " x " + std::to_string(extent) + ")",
                    a, b, trans_b);
    p.batch *= extent;
  }
  // M2/M3: the batch classes. The stride rule is the forward's original one, (ndb == nda) -> B stride K*N, else 0,
  // so every accepted pair keeps its pre-ENG4 strides (rank 2 x rank 2 is class (b) with an empty batch shape).
  if (nda == ndb) {
    // (b) equal ranks: the batch shapes must be equal dimension by dimension (a count match is not enough: equal
    // counts with different shapes pair matrices that NumPy would not pair). An all-ones B batch under a larger A
    // batch is refused too: no new broadcasting this round.
    for (int d = 0; d < nda - 2; ++d)
      if (a.shape[d] != b.shape[d])
        return refuse(p, MatmulOperandIssue::BatchShape,
                      "equal ranks need equal batch shapes, dimension by dimension", a, b, trans_b);
    p.batched_b = true;
  } else if (ndb > 2) {
    // (c) unequal ranks with a batched B: B is one matrix only if every leading extent is 1.
    for (int d = 0; d < ndb - 2; ++d)
      if (b.shape[d] != 1)
        return refuse(p, MatmulOperandIssue::BatchShape,
                      "unequal ranks need B rank 2 or a B batch of ones (every leading B extent 1)", a, b, trans_b);
    p.batched_b = false;
  } else {
    p.batched_b = false;  // (a) B rank 2 under a batched A
  }
  return p;
}

NDArray matmul(const NDArray& a, const NDArray& b, float alpha, bool trans_b) {
  // ENG4: every operand check runs here, before any CUDA call (allocation, handle, GEMM). The first three messages
  // keep their pre-ENG4 text as a prefix.
  const MatmulOperandPlan plan = matmul_operand_plan(a, b, trans_b);
  switch (plan.issue) {
    case MatmulOperandIssue::None: break;
    case MatmulOperandIssue::Rank: throw std::runtime_error("matmul needs >=2D inputs: " + plan.detail);
    case MatmulOperandIssue::Dtype: throw std::runtime_error("matmul dtype mismatch: " + plan.detail);
    case MatmulOperandIssue::InnerDim: throw std::runtime_error("matmul inner dim mismatch: " + plan.detail);
    case MatmulOperandIssue::UnsupportedDtype:
      throw std::runtime_error("matmul supports float32/float16/bfloat16 only: " + plan.detail);
    default: throw std::runtime_error("matmul refused: " + plan.detail);
  }

  // ENG4-F1 X2 (ENG4R1-F02): zero-size GEMMs never reach cuBLAS. Before, they did: K = 0 passed lda = 0 and N = 0
  // passed ldc = 0, both invalid (cuBLAS requires every leading dimension >= max(1, rows)); the status was discarded
  // and a nonempty K = 0 output was left unwritten. Now an empty output (M, N or the batch count 0) is returned as
  // is, and K = 0 with a nonempty output returns zeros, which is NumPy's matmul result for these shapes and what the
  // fp32 backward already returns (ops.cpp, `if (!k)`). Both use the predicate's own out_shape and K.
  // Prior art: cuBLAS documentation (lda/ldb/ldc >= max(1, rows)); NumPy's matmul on zero-size operands (an empty
  // result, or zeros when the contracted dimension is 0, the empty sum); the BLAS reference xGEMM quick-return
  // (M = 0 or N = 0 returns; K = 0 with beta = 0 sets C = 0). Ours: placement after the predicate.
  if (numel_of(plan.out_shape) == 0) return NDArray(plan.out_shape, a.dtype, a.device);  // 0 bytes: no allocation
  // ENG4-F2 Y3 / ENG4-F3 Z2: the K = 0 zeros come from zeros_checked, which checks cudaMemset's own return status
  // (a failed fill raises "matmul zero fill: cudaMemset failed: ...") and never reads the last-error slot.
  if (plan.k == 0) return zeros_checked(plan.out_shape, a.dtype, a.device, "matmul zero fill");

  // trans_b: C = alpha * A @ B^T with B stored row-major (N, K) — consumes the
  // operand without a materialized transpose copy (cuBLAS reads it via OP_T).
  int64_t M = plan.m, K = plan.k;
  int64_t Kb = plan.kb;
  int64_t N  = plan.n;

  // Batch = product of a's leading dims (b matches it or is one matrix: the predicate above).
  int64_t batch = plan.batch;
  Shape out_shape = plan.out_shape;
  bool b_batched = plan.batched_b;  // == (ndb == nda) for every accepted pair

  NDArray out(out_shape, a.dtype, a.device);
  int64_t strideA = M * K, strideB = Kb * N, strideC = M * N;
  // b may be a 2D operand broadcast across the batch: stride 0 reuses it (cuBLAS
  // strided-batched supports a zero stride). Collapse the former per-batch loop
  // into a single strided-batched launch — one kernel-launch + cuBLAS heuristic
  // pick for the whole batch instead of `batch` of them.
  int64_t sB = b_batched ? strideB : 0;
  float beta = 0.f;
  void* ap = a.data_ptr();
  void* bp = b.data_ptr();
  void* cp = out.data_ptr();
  // Column-major view: first operand slot holds B. Row-major B(N,K) read via
  // OP_T is the col-major (K,N) buffer with ld=K — same bytes, no copy.
  cublasOperation_t opB = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
  int ldb = trans_b ? (int)K : (int)N;

  cublasStatus_t status = CUBLAS_STATUS_NOT_SUPPORTED;
  if (a.dtype == DType::Float32) {
    status = cublasSgemmStridedBatched(
        handle(), opB, CUBLAS_OP_N, (int)N, (int)M, (int)K,
        &alpha, reinterpret_cast<const float*>(bp), ldb, sB,
        reinterpret_cast<const float*>(ap), (int)K, strideA,
        &beta, reinterpret_cast<float*>(cp), (int)N, strideC, (int)batch);
  } else if (a.dtype == DType::Float16) {
    status = cublasGemmStridedBatchedEx(
        handle(), opB, CUBLAS_OP_N, (int)N, (int)M, (int)K,
        &alpha, bp, CUDA_R_16F, ldb, sB, ap, CUDA_R_16F, (int)K, strideA,
        &beta, cp, CUDA_R_16F, (int)N, strideC, (int)batch,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  } else if (a.dtype == DType::BFloat16) {
    status = cublasGemmStridedBatchedEx(
        handle(), opB, CUBLAS_OP_N, (int)N, (int)M, (int)K,
        &alpha, bp, CUDA_R_16BF, ldb, sB, ap, CUDA_R_16BF, (int)K, strideA,
        &beta, cp, CUDA_R_16BF, (int)N, strideC, (int)batch,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  } else {
    // unreachable since ENG4-F1 X1 (the predicate refuses other dtypes before the allocation); kept as a guard
    throw std::runtime_error("matmul supports float32/float16/bfloat16 only");
  }
  cuda_check_last("matmul");
  // ENG4-F1 X2: the GEMM's own status is no longer discarded. (The CUDA error check above runs first, so a missing
  // device keeps its pre-F1 message.) Prior art: cuBLAS documentation, cublasStatus_t and cublasGetStatusName.
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string("matmul: cuBLAS GEMM returned ") + cublasGetStatusName(status) + " (" +
                             std::to_string(static_cast<int>(status)) + ")");
  return out;
}

}  // namespace tc
