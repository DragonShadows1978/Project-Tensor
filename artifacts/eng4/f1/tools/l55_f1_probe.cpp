// ENG4-F1 L55 probe: X2's early returns see the predicate's own M, N, K and out_shape. Compiled with F1's REAL
// src/matmul.cu (nvcc) and this file (g++, host C++), with stub definitions of the few kernels.cu symbols matmul.cu
// links against (NDArray::zeros records its call) and of the cuBLAS entry points (they record a GEMM call). No GPU and
// no CUDA runtime call. For each witness (M = 0, N = 0, K = 0, batch = 0, plus a nonzero control) it prints the
// plan's m, n, k, batch and out_shape and which path the forward took: "empty_return" (no zeros, no GEMM),
// "zeros_return" (NDArray::zeros with the plan's out_shape, no GEMM) or "gemm" (with the GEMM's m, n, k, lds, batch).
// Build: g++ -std=c++17 -I<tree>/include -I<cuda>/include -c l55_f1_probe.cpp; nvcc -c <tree>/src/matmul.cu; link
// with -lcudart_static. Prior art: link-time test doubles (stubs); ENG4's l55_probe.cu (this repo). Ours: this probe.
#include "tc/core.h"

#include <cublas_v2.h>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace {
struct Rec {
  int gemm_calls = 0, zeros_calls = 0;
  int m = 0, n = 0, k = 0, lda = 0, ldb = 0, ldc = 0, batch = 0;
  tc::Shape zeros_shape;
} g;
}  // namespace

namespace tc {
const char* dtype_name(DType dt) { return dt == DType::Float32 ? "float32" : "other"; }
std::string Device::str() const { return is_cuda() ? ("cuda:" + std::to_string(index)) : "cpu"; }
NDArray::NDArray(const Shape& s, DType d, Device v) : shape(s), dtype(d), device(v) {}
void* NDArray::data_ptr() const { return const_cast<NDArray*>(this); }
void cuda_check_last(const char*) {}
int64_t numel_of(const Shape& s) {
  int64_t n = 1;
  for (auto d : s) {
    if (d == 0) return 0;
    n *= d;
  }
  return n;
}
NDArray NDArray::zeros(const Shape& s, DType d, Device v) {
  ++g.zeros_calls;
  g.zeros_shape = s;
  return NDArray(s, d, v);
}
}  // namespace tc

extern "C" {
cublasStatus_t cublasCreate_v2(cublasHandle_t* h) {
  *h = reinterpret_cast<cublasHandle_t>(0x1);
  return CUBLAS_STATUS_SUCCESS;
}
const char* cublasGetStatusName(cublasStatus_t) { return "STUB"; }
cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t, cublasOperation_t, cublasOperation_t, int m, int n, int k,
                                         const float*, const float*, int lda, long long, const float*, int ldb,
                                         long long, const float*, float*, int ldc, long long, int batchCount) {
  ++g.gemm_calls;
  g.m = m; g.n = n; g.k = k; g.lda = lda; g.ldb = ldb; g.ldc = ldc; g.batch = batchCount;
  return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t, cublasOperation_t, cublasOperation_t, int m, int n, int k,
                                          const void*, const void*, cudaDataType, int lda, long long, const void*,
                                          cudaDataType, int ldb, long long, const void*, void*, cudaDataType, int ldc,
                                          long long, int batchCount, cublasComputeType_t, cublasGemmAlgo_t) {
  ++g.gemm_calls;
  g.m = m; g.n = n; g.k = k; g.lda = lda; g.ldb = ldb; g.ldc = ldc; g.batch = batchCount;
  return CUBLAS_STATUS_SUCCESS;
}
}

static std::string txt(const tc::Shape& s) {
  std::string t = "[";
  for (size_t i = 0; i < s.size(); ++i) t += (i ? ", " : "") + std::to_string(s[i]);
  return t + "]";
}

static void probe(const char* name, const tc::Shape& sa, const tc::Shape& sb, bool trans_b) {
  tc::NDArray a, b;
  a.shape = sa; b.shape = sb;
  a.device = b.device = tc::Device{tc::DeviceType::CUDA, 0};
  auto p = tc::matmul_operand_plan(a, b, trans_b);
  g = Rec{};
  std::string path;
  tc::Shape out_shape;
  try {
    tc::NDArray out = tc::matmul(a, b, 1.f, trans_b);
    out_shape = out.shape;
    path = g.gemm_calls ? "gemm" : (g.zeros_calls ? "zeros_return" : "empty_return");
  } catch (const std::exception& e) {
    path = std::string("raised: ") + e.what();
  }
  std::printf("{\"witness\": \"%s\", \"a\": \"%s\", \"b\": \"%s\", \"trans_b\": %s, \"plan\": {\"issue\": %d, "
              "\"m\": %lld, \"n\": %lld, \"k\": %lld, \"batch\": %lld, \"out_shape\": \"%s\"}, \"path\": \"%s\", "
              "\"returned_shape\": \"%s\", \"zeros_shape\": \"%s\", \"gemm_calls\": %d, \"zeros_calls\": %d, "
              "\"gemm\": {\"m(N)\": %d, \"n(M)\": %d, \"k\": %d, \"ld_B\": %d, \"ld_A\": %d, \"ld_C\": %d, "
              "\"batch\": %d}}\n",
              name, txt(sa).c_str(), txt(sb).c_str(), trans_b ? "true" : "false", (int)p.issue, (long long)p.m,
              (long long)p.n, (long long)p.k, (long long)p.batch, txt(p.out_shape).c_str(), path.c_str(),
              txt(out_shape).c_str(), txt(g.zeros_shape).c_str(), g.gemm_calls, g.zeros_calls, g.m, g.n, g.k,
              g.lda, g.ldb, g.ldc, g.batch);
}

int main() {
  probe("M0", {3, 0, 5}, {5, 4}, false);
  probe("N0", {3, 2, 5}, {5, 0}, false);
  probe("N0_trans", {3, 2, 5}, {0, 5}, true);
  probe("K0", {3, 2, 0}, {0, 4}, false);
  probe("K0_trans", {3, 2, 0}, {4, 0}, true);
  probe("K0_class_b", {2, 3, 2, 0}, {2, 3, 0, 4}, false);
  probe("K0_class_c", {3, 2, 0}, {1, 1, 0, 4}, false);
  probe("batch0", {0, 2, 5}, {5, 4}, false);
  probe("batch0_class_b", {0, 3, 2, 5}, {0, 3, 5, 4}, false);
  probe("control_nonzero", {3, 2, 5}, {5, 4}, false);
  return 0;
}
