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
#include <stdexcept>

namespace tc {

namespace {
cublasHandle_t g_handle = nullptr;
cublasHandle_t handle() {
  if (!g_handle) cublasCreate(&g_handle);
  return g_handle;
}
}  // namespace

NDArray matmul(const NDArray& a, const NDArray& b) {
  int nda = a.ndim(), ndb = b.ndim();
  if (nda < 2 || ndb < 2) throw std::runtime_error("matmul needs >=2D inputs");
  if (a.dtype != b.dtype) throw std::runtime_error("matmul dtype mismatch");

  int64_t M = a.shape[nda - 2], K = a.shape[nda - 1];
  int64_t Kb = b.shape[ndb - 2], N = b.shape[ndb - 1];
  if (K != Kb) throw std::runtime_error("matmul inner dim mismatch");

  // Batch = product of a's leading dims (b must match or be 2D broadcast).
  int64_t batch = 1;
  Shape out_shape;
  for (int d = 0; d < nda - 2; ++d) { batch *= a.shape[d]; out_shape.push_back(a.shape[d]); }
  out_shape.push_back(M); out_shape.push_back(N);
  bool b_batched = (ndb == nda);

  NDArray out(out_shape, a.dtype, a.device);
  int64_t strideA = M * K, strideB = Kb * N, strideC = M * N;
  float alpha = 1.f, beta = 0.f;

  for (int64_t bi = 0; bi < batch; ++bi) {
    const char* ap = static_cast<char*>(a.data_ptr()) + bi * strideA * dtype_size(a.dtype);
    const char* bp = static_cast<char*>(b.data_ptr()) + (b_batched ? bi * strideB : 0) * dtype_size(b.dtype);
    char* cp = static_cast<char*>(out.data_ptr()) + bi * strideC * dtype_size(a.dtype);
    if (a.dtype == DType::Float32) {
      cublasSgemm(handle(), CUBLAS_OP_N, CUBLAS_OP_N, (int)N, (int)M, (int)K,
                  &alpha, reinterpret_cast<const float*>(bp), (int)N,
                  reinterpret_cast<const float*>(ap), (int)K,
                  &beta, reinterpret_cast<float*>(cp), (int)N);
    } else if (a.dtype == DType::Float16) {
      cublasGemmEx(handle(), CUBLAS_OP_N, CUBLAS_OP_N, (int)N, (int)M, (int)K,
                   &alpha, bp, CUDA_R_16F, (int)N, ap, CUDA_R_16F, (int)K,
                   &beta, cp, CUDA_R_16F, (int)N, CUBLAS_COMPUTE_32F,
                   CUBLAS_GEMM_DEFAULT);
    } else if (a.dtype == DType::BFloat16) {
      cublasGemmEx(handle(), CUBLAS_OP_N, CUBLAS_OP_N, (int)N, (int)M, (int)K,
                   &alpha, bp, CUDA_R_16BF, (int)N, ap, CUDA_R_16BF, (int)K,
                   &beta, cp, CUDA_R_16BF, (int)N, CUBLAS_COMPUTE_32F,
                   CUBLAS_GEMM_DEFAULT);
    } else {
      throw std::runtime_error("matmul supports float32/float16 only");
    }
  }
  cuda_check_last("matmul");
  return out;
}

}  // namespace tc
