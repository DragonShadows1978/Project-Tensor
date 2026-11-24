// ENG4 L55 probe: compile a build's REAL src/matmul.cu on the host and replace only the cuBLAS entry points (and the
// few kernels.cu core definitions matmul.cu links against) with stubs that print the arguments the forward passes.
// No GPU and no CUDA runtime call: NDArray is built field by field (default constructor), data_ptr() returns the
// object's address, and the stubs return CUBLAS_STATUS_SUCCESS.
//
// For each witness it prints: the shapes; on Build E the shared predicate's verdict (tc::matmul_operand_plan); then
// either the refusal text or the strided-batched arguments the forward hands cuBLAS (column-major slots: the first
// operand is B, so "strideA" below is B's stride), plus the last element of B and A the GEMM would read against each
// operand's element count. On Build B this shows the ENG-MM1 arithmetic (B read past its end) on the CPU.
//
// Build: nvcc -std=c++17 -I<tree>/include [-DENG4_HAS_PLAN] -o <out> l55_probe.cu <tree>/src/matmul.cu
// Prior art: link-time test doubles (stub/fake functions; Meszaros, "xUnit Test Patterns", 2007 -- unverified, lead
// to check); the cuBLAS strided-batched argument contract (cuBLAS docs). Ours: this probe.
#include "tc/core.h"

#include <cublas_v2.h>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace tc {
const char* dtype_name(DType dt) {
  switch (dt) {
    case DType::Float32: return "float32";
    case DType::Float16: return "float16";
    case DType::BFloat16: return "bfloat16";
    default: return "other";
  }
}
std::string Device::str() const { return is_cuda() ? ("cuda:" + std::to_string(index)) : "cpu"; }
NDArray::NDArray(const Shape& s, DType d, Device v) : shape(s), dtype(d), device(v) {}
void* NDArray::data_ptr() const { return const_cast<NDArray*>(this); }
void cuda_check_last(const char*) {}
}  // namespace tc

namespace {
struct Call {
  bool seen = false;
  int m = 0, n = 0, k = 0, lda = 0, ldb = 0, ldc = 0, batch = 0;
  long long sA = 0, sB = 0, sC = 0;
  int opA = 0;
} g_call;
}  // namespace

extern "C" {
cublasStatus_t cublasCreate_v2(cublasHandle_t* h) {
  *h = reinterpret_cast<cublasHandle_t>(0x1);
  return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t, cublasOperation_t transa, cublasOperation_t, int m, int n,
                                         int k, const float*, const float*, int lda, long long strideA, const float*,
                                         int ldb, long long strideB, const float*, float*, int ldc,
                                         long long strideC, int batchCount) {
  g_call = {true, m, n, k, lda, ldb, ldc, batchCount, strideA, strideB, strideC, (int)transa};
  return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t, cublasOperation_t transa, cublasOperation_t, int m, int n,
                                          int k, const void*, const void*, cudaDataType, int lda, long long strideA,
                                          const void*, cudaDataType, int ldb, long long strideB, const void*, void*,
                                          cudaDataType, int ldc, long long strideC, int batchCount,
                                          cublasComputeType_t, cublasGemmAlgo_t) {
  g_call = {true, m, n, k, lda, ldb, ldc, batchCount, strideA, strideB, strideC, (int)transa};
  return CUBLAS_STATUS_SUCCESS;
}
}

using tc::NDArray;
using tc::Shape;

static NDArray arr(const Shape& s, tc::DType dt = tc::DType::Float32) {
  NDArray a;
  a.shape = s;
  a.dtype = dt;
  a.device = tc::Device{tc::DeviceType::CUDA, 0};
  return a;
}
static long long numel(const Shape& s) {
  long long n = 1;
  for (auto d : s) n *= d;
  return n;
}
static std::string txt(const Shape& s) {
  std::string t = "[";
  for (size_t i = 0; i < s.size(); ++i) t += (i ? ", " : "") + std::to_string(s[i]);
  return t + "]";
}

static void probe(const char* cls, const Shape& sa, const Shape& sb, bool trans_b) {
  NDArray a = arr(sa), b = arr(sb);
  std::printf("{\"class\": \"%s\", \"a\": \"%s\", \"b\": \"%s\", \"trans_b\": %s", cls, txt(sa).c_str(),
              txt(sb).c_str(), trans_b ? "true" : "false");
#ifdef ENG4_HAS_PLAN
  auto p = tc::matmul_operand_plan(a, b, trans_b);
  std::printf(", \"plan_issue\": %d, \"plan_batched_b\": %s, \"plan_batch\": %lld", (int)p.issue,
              p.batched_b ? "true" : "false", (long long)p.batch);
#endif
  g_call = Call{};
  try {
    NDArray out = tc::matmul(a, b, 1.f, trans_b);
    // Column-major slots: first operand = our B (stride sA here), second = our A (stride sB here).
    const long long b_matrix = (long long)g_call.k * g_call.m;  // K*N elements per B matrix
    const long long a_matrix = (long long)g_call.k * g_call.n;  // M*K elements per A matrix
    const long long b_last = (long long)(g_call.batch - 1) * g_call.sA + b_matrix;  // one past the last B element
    const long long a_last = (long long)(g_call.batch - 1) * g_call.sB + a_matrix;
    std::printf(", \"cublas\": {\"m(N)\": %d, \"n(M)\": %d, \"k\": %d, \"ld_B\": %d, \"stride_B\": %lld, "
                "\"ld_A\": %d, \"stride_A\": %lld, \"ld_C\": %d, \"stride_C\": %lld, \"batch\": %d}, "
                "\"B_elements_read_end\": %lld, \"B_numel\": %lld, \"B_out_of_bounds\": %s, "
                "\"A_elements_read_end\": %lld, \"A_numel\": %lld, \"out\": \"%s\"}\n",
                g_call.m, g_call.n, g_call.k, g_call.lda, g_call.sA, g_call.ldb, g_call.sB, g_call.ldc, g_call.sC,
                g_call.batch, b_last, numel(sb), b_last > numel(sb) ? "true" : "false", a_last, numel(sa),
                txt(out.shape).c_str());
  } catch (const std::exception& e) {
    std::string m = e.what();
    for (auto& c : m) if (c == '"') c = '\'';
    std::printf(", \"refused\": \"%s\"}\n", m.c_str());
  }
}

int main() {
  // L55: one witness per accepted class.
  probe("accept_a", {4, 2, 3}, {3, 4}, false);
  probe("accept_b", {4, 2, 3}, {4, 3, 4}, false);
  probe("accept_b_rank2", {2, 3}, {3, 4}, false);
  probe("accept_c", {2, 2, 3}, {1, 1, 3, 4}, false);
  probe("accept_c_trans", {2, 2, 3}, {1, 1, 4, 3}, true);
  // M3 classes (refused on E; on B they show what the forward computed).
  probe("m3_ENG-MM1_smaller_b_batch", {4, 2, 3}, {2, 3, 4}, false);
  probe("m3_larger_b_batch", {2, 2, 3}, {4, 3, 4}, false);
  probe("m3_eq_count_diff_shape", {2, 3, 2, 3}, {3, 2, 3, 4}, false);
  probe("m3_ones_b_under_larger_a", {2, 2, 3}, {1, 3, 4}, false);
  probe("m3_uneq_rank_batched_b", {2, 3}, {2, 3, 4}, false);
  probe("m3_uneq_rank_partial_ones", {2, 2, 3}, {1, 2, 3, 4}, false);
  return 0;
}
