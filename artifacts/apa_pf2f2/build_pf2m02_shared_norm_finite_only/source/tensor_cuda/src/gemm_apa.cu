// APAMQ-SB2: cuBLASLt INT8 bulk + bounded compact refinement.
//
// The public tensor dtype surface intentionally stays unchanged. Signed INT8
// codes use the bit pattern of uint8 NDArrays; INT32 cuBLASLt accumulators use
// a 4-byte internal allocation and are never exposed except widened to int64
// by the gate-only debug entry. No score is ever stored in BF16.

#include "tc/core.h"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>

namespace tc {
namespace {

constexpr int kThreads = 256;
constexpr size_t kLtWorkspaceBytes = 32ull * 1024 * 1024;
constexpr int kRefinePairChunk = 262144;
// Chunk-owned score/index/probability storage. Call-global K codes/scales,
// output, and the cuBLASLt workspace leave headroom below the 384 MiB rail.
constexpr size_t kChunkTransientBudget = 288ull * 1024 * 1024;
// Above this gather footprint, a chunk uses the dense exact BF16->FP32 GEMM
// fallback. It has identical selected-vs-bulk score semantics and avoids an
// unbounded duplicated-Q/K gather workspace.
constexpr size_t kSparseGatherBudget = 128ull * 1024 * 1024;

#if defined(TC_APA_GEMM_DEBUG_ASSERTS)
#define SB2_DASSERT(expr) assert(expr)
#else
#define SB2_DASSERT(expr) ((void)0)
#endif

struct SbShape {
  int B, H, KVH, group, L, S, D, VD, rows, M, Spad, batches;
  int full_lq;
};

void check_cuda(cudaError_t status, const char* where) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(where) + ": " +
                             cudaGetErrorString(status));
}

void check_blas(cublasStatus_t status, const char* where) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string(where) + ": cuBLAS status " +
                             std::to_string((int)status));
}

cublasLtHandle_t lt_handle() {
  static cublasLtHandle_t h = nullptr;
  static std::once_flag once;
  std::call_once(once, [&] { check_blas(cublasLtCreate(&h), "cublasLtCreate"); });
  return h;
}

cublasHandle_t blas_handle() {
  static cublasHandle_t h = nullptr;
  static std::once_flag once;
  std::call_once(once, [&] { check_blas(cublasCreate(&h), "cublasCreate"); });
  return h;
}

SbShape validate(const NDArray& q, const NDArray& k, const NDArray* v,
                 bool is_causal, int Lq, int64_t row0, int window,
                 const char* name) {
  if (q.ndim() != 4 || k.ndim() != 4 || (v && v->ndim() != 4))
    throw std::runtime_error(std::string(name) + " expects rank-4 q/k/v");
  if (q.dtype != DType::BFloat16 || k.dtype != DType::BFloat16 ||
      (v && v->dtype != DType::BFloat16))
    throw std::runtime_error(std::string(name) + " requires bfloat16 q/k/v");
  if (!q.device.is_cuda() || !k.device.is_cuda() ||
      q.device.index != k.device.index ||
      (v && (!v->device.is_cuda() || v->device.index != q.device.index)))
    throw std::runtime_error(std::string(name) + " requires one CUDA device");

  const int64_t B = q.shape[0], H = q.shape[1], L = q.shape[2], D = q.shape[3];
  const int64_t KVH = k.shape[1], S = k.shape[2];
  const int64_t VD = v ? v->shape[3] : D;
  if (B <= 0 || H <= 0 || L <= 0 || D <= 0 || KVH <= 0 || S <= 0 ||
      VD <= 0 || k.shape[0] != B || k.shape[3] != D || H % KVH != 0)
    throw std::runtime_error(std::string(name) + " invalid q/k geometry");
  if (v && (v->shape[0] != B || v->shape[1] != KVH ||
            v->shape[2] != S))
    throw std::runtime_error(std::string(name) + " invalid v geometry");
  // CUDA's signed INT8 tensor-core layouts require a four-byte reduction
  // granularity. Registered SB1 dimensions (128/512) are also multiples of 16.
  if ((D & 3) != 0)
    throw std::runtime_error(std::string(name) + " requires D divisible by 4 for INT8 cuBLASLt");
  int64_t full_lq = Lq > 0 ? Lq : L;
  if (is_causal) {
    if (S < full_lq || row0 < 0 || row0 + L > full_lq)
      throw std::runtime_error(std::string(name) +
                               " invalid bottom-right causal Lq/row0 bounds");
  } else if (window > 0) {
    throw std::runtime_error(std::string(name) +
                             " window requires is_causal=True");
  }
  const int64_t rows = B * H * L;
  const int64_t M = (H / KVH) * L;
  const int64_t batches = B * KVH;
  const int64_t Spad = (S + 3) & ~int64_t(3);
  for (int64_t x : {B, H, L, D, KVH, S, VD, rows, M, batches, Spad, full_lq})
    if (x > std::numeric_limits<int>::max())
      throw std::runtime_error(std::string(name) + " shape exceeds INT32/cuBLASLt range");
  if (rows > 0 && S > std::numeric_limits<int>::max() / rows)
    throw std::runtime_error(std::string(name) + " score matrix exceeds compact-index range");
  return {(int)B, (int)H, (int)KVH, (int)(H / KVH), (int)L, (int)S,
          (int)D, (int)VD, (int)rows, (int)M, (int)Spad, (int)batches,
          (int)full_lq};
}

// Both Q and K use this exact quantizer. roundf is round-to-nearest with
// halfway cases away from zero (unlike NumPy rint's ties-to-even).
__global__ void quantize_bf16_rows(const __nv_bfloat16* src, uint8_t* codes,
                                   float* scales, int rows, int D) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  if (row >= rows) return;
  const __nv_bfloat16* in = src + (int64_t)row * D;
  float local = 0.f;
  for (int d = tid; d < D; d += blockDim.x)
    local = fmaxf(local, fabsf(__bfloat162float(in[d])));
  __shared__ float red[kThreads];
  red[tid] = local;
  __syncthreads();
  for (int off = blockDim.x / 2; off; off >>= 1) {
    if (tid < off) red[tid] = fmaxf(red[tid], red[tid + off]);
    __syncthreads();
  }
  const float scale = red[0] > 0.f ? red[0] * (1.f / 127.f) : 1.f;
  const float recip = 1.f / scale;
  for (int d = tid; d < D; d += blockDim.x) {
    float c = roundf(__bfloat162float(in[d]) * recip);
    c = fminf(127.f, fmaxf(-127.f, c));
    codes[(int64_t)row * D + d] = (uint8_t)(int8_t)(int)c;
  }
  if (tid == 0) scales[row] = scale;
}

void quantize(const NDArray& x, NDArray& codes, NDArray& scales) {
  const int rows = (int)(x.numel() / x.shape.back());
  const int D = (int)x.shape.back();
  quantize_bf16_rows<<<rows, kThreads>>>(
      static_cast<const __nv_bfloat16*>(x.data_ptr()),
      static_cast<uint8_t*>(codes.data_ptr()),
      static_cast<float*>(scales.data_ptr()), rows, D);
  cuda_check_last("apa_gemm_selective quantize");
}

void validate_cached_k(const NDArray& k, const NDArray& codes,
                       const NDArray& scales) {
  Shape scale_shape{k.shape[0], k.shape[1], k.shape[2]};
  if (codes.dtype != DType::Uint8 || codes.shape != k.shape ||
      codes.device.type != k.device.type || codes.device.index != k.device.index)
    throw std::runtime_error("apa_gemm_selective cached K codes must be uint8 with K shape/device");
  if (scales.dtype != DType::Float32 || scales.shape != scale_shape ||
      scales.device.type != k.device.type || scales.device.index != k.device.index)
    throw std::runtime_error("apa_gemm_selective cached K scales must be float32 [B,KVH,S]");
}

void lt_int8_bulk(const NDArray& qcodes, const NDArray& kcodes,
                  NDArray& accum, const SbShape& s) {
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t ad = nullptr, bd = nullptr, cd = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;
  check_blas(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32I, CUDA_R_32I),
             "SB1 cublasLtMatmulDescCreate");
  cublasOperation_t trans_b = CUBLAS_OP_T;
  check_blas(cublasLtMatmulDescSetAttribute(
                 op, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)),
             "SB1 set TRANSB");
  check_blas(cublasLtMatrixLayoutCreate(&ad, CUDA_R_8I, s.M, s.D, s.D),
             "SB1 A layout");
  check_blas(cublasLtMatrixLayoutCreate(&bd, CUDA_R_8I, s.S, s.D, s.D),
             "SB1 B layout");
  check_blas(cublasLtMatrixLayoutCreate(&cd, CUDA_R_32I, s.M, s.S, s.Spad),
             "SB1 C layout");
  cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  for (auto layout : {ad, bd, cd})
    check_blas(cublasLtMatrixLayoutSetAttribute(
                   layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
               "SB1 set row order");
  const int batch_count = s.batches;
  const int64_t stride_a = (int64_t)s.M * s.D;
  const int64_t stride_b = (int64_t)s.S * s.D;
  const int64_t stride_c = (int64_t)s.M * s.Spad;
  for (auto layout : {ad, bd, cd})
    check_blas(cublasLtMatrixLayoutSetAttribute(
                   layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                   &batch_count, sizeof(batch_count)),
               "SB1 set batch count");
  check_blas(cublasLtMatrixLayoutSetAttribute(
                 ad, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                 &stride_a, sizeof(stride_a)), "SB1 set A stride");
  check_blas(cublasLtMatrixLayoutSetAttribute(
                 bd, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                 &stride_b, sizeof(stride_b)), "SB1 set B stride");
  check_blas(cublasLtMatrixLayoutSetAttribute(
                 cd, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                 &stride_c, sizeof(stride_c)), "SB1 set C stride");

  check_blas(cublasLtMatmulPreferenceCreate(&pref), "SB1 preference create");
  size_t workspace_bytes = kLtWorkspaceBytes;
  check_blas(cublasLtMatmulPreferenceSetAttribute(
                 pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                 &workspace_bytes, sizeof(workspace_bytes)),
             "SB1 preference workspace");
  cublasLtMatmulHeuristicResult_t heur[8];
  int returned = 0;
  check_blas(cublasLtMatmulAlgoGetHeuristic(
                 lt_handle(), op, ad, bd, cd, cd, pref, 8, heur, &returned),
             "SB1 cublasLt heuristic");
  int pick = -1;
  for (int i = 0; i < returned; ++i)
    if (heur[i].state == CUBLAS_STATUS_SUCCESS &&
        heur[i].workspaceSize <= workspace_bytes) { pick = i; break; }
  if (pick < 0)
    throw std::runtime_error("SB1 cuBLASLt found no INT8->INT32 row-major algorithm");
  NDArray workspace({(int64_t)workspace_bytes}, DType::Uint8, qcodes.device);
  const int32_t alpha = 1, beta = 0;
  check_blas(cublasLtMatmul(
                 lt_handle(), op, &alpha, qcodes.data_ptr(), ad,
                 kcodes.data_ptr(), bd, &beta, accum.data_ptr(), cd,
                 accum.data_ptr(), cd, &heur[pick].algo,
                 workspace.data_ptr(), workspace_bytes, 0),
             "SB1 cublasLtMatmul INT8->INT32");
  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(cd);
  cublasLtMatrixLayoutDestroy(bd);
  cublasLtMatrixLayoutDestroy(ad);
  cublasLtMatmulDescDestroy(op);
}

__device__ __forceinline__ void valid_range(int row, const SbShape s,
                                             int64_t row0, int is_causal,
                                             int window, int& lo, int& hi) {
  lo = 0;
  hi = s.S;
  if (!is_causal) return;
  int i = row % s.L;
  int64_t valid_hi = (int64_t)s.S - s.full_lq + row0 + i + 1;
  hi = (int)(valid_hi < 0 ? 0 : (valid_hi > s.S ? s.S : valid_hi));
  lo = window > 0 ? max(0, hi - window) : 0;
  SB2_DASSERT(lo >= 0 && lo <= hi && hi <= s.S);
}

__device__ __forceinline__ bool select_score(float score, float threshold) {
  // Deliberately positive-form. The SB1 appender used !(abs < threshold),
  // which differs from abs >= threshold for NaNs and could append a pair that
  // the count pass did not reserve.
  return isfinite(score) && isfinite(threshold) && fabsf(score) >= threshold;
}

__global__ void scale_stats_count(const int32_t* accum,
                                  const float* qscales,
                                  const float* kscales, float* scores,
                                  float* thresholds, int* counts, SbShape s,
                                  float attn_scale, float zthr,
                                  int is_causal, int64_t row0, int window) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  if (row >= s.rows) return;
  int i = row % s.L;
  int bh = row / s.L;
  int b = bh / s.H;
  int h = bh % s.H;
  int kh = h / s.group;
  int mrow = (h % s.group) * s.L + i;
  int batch = b * s.KVH + kh;
  const int32_t* arow = accum + ((int64_t)batch * s.M + mrow) * s.Spad;
  const float* ks = kscales + ((int64_t)b * s.KVH + kh) * s.S;
  float* out = scores + (int64_t)row * s.S;
  int lo, hi;
  valid_range(row, s, row0, is_causal, window, lo, hi);
  float sum = 0.f, sumsq = 0.f;
  const float qs = qscales[row] * attn_scale;
  for (int j = tid; j < s.S; j += blockDim.x) {
    float v = 0.f;
    if (j >= lo && j < hi) {
      v = (float)arow[j] * (qs * ks[j]);
      float a = fabsf(v);
      sum += a;
      sumsq += a * a;
    }
    out[j] = v;
  }
  __shared__ float red[kThreads];
  red[tid] = sum;
  __syncthreads();
  for (int off = blockDim.x / 2; off; off >>= 1) {
    if (tid < off) red[tid] += red[tid + off];
    __syncthreads();
  }
  float total = red[0];
  red[tid] = sumsq;
  __syncthreads();
  for (int off = blockDim.x / 2; off; off >>= 1) {
    if (tid < off) red[tid] += red[tid + off];
    __syncthreads();
  }
  float denom = (float)max(hi - lo, 1);
  float mean = total / denom;
  float thr = mean + zthr * sqrtf(fmaxf(red[0] / denom - mean * mean, 0.f));
  if (tid == 0) thresholds[row] = thr;
  __syncthreads();
  int local_count = 0;
  for (int j = lo + tid; j < hi; j += blockDim.x)
    local_count += select_score(out[j], thr);
  __shared__ int ired[kThreads];
  ired[tid] = local_count;
  __syncthreads();
  for (int off = blockDim.x / 2; off; off >>= 1) {
    if (tid < off) ired[tid] += ired[tid + off];
    __syncthreads();
  }
  if (tid == 0) counts[row] = (hi > lo) ? ired[0] : 0;
}

unsigned long long valid_pairs(const SbShape& s, bool causal, int64_t row0,
                               int window) {
  if (!causal) return (unsigned long long)s.rows * s.S;
  unsigned long long total = 0;
  for (int b = 0; b < s.B; ++b)
    for (int h = 0; h < s.H; ++h)
      for (int i = 0; i < s.L; ++i) {
        int64_t hi64 = (int64_t)s.S - s.full_lq + row0 + i + 1;
        int hi = (int)std::max<int64_t>(0, std::min<int64_t>(s.S, hi64));
        int lo = window > 0 ? std::max(0, hi - window) : 0;
        total += (unsigned)(hi - lo);
      }
  return total;
}

// counters = {chunk_attempted, call_selected, call_dropped, call_overflow}.
// The index buffers are capacity-sized before launch. Selection beyond cap is
// deliberately clamped, counted, and surfaced through the public stats API.
__global__ void compact_pairs_bounded(
    const float* scores, const float* thresholds,
    int* pair_rows, int* pair_keys, unsigned long long* counters, int capacity,
    int clamp_overflow, SbShape s, int is_causal, int64_t row0, int window) {
  int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = (int64_t)s.rows * s.S;
  if (linear >= total) return;
  int row = (int)(linear / s.S);
  int key = (int)(linear - (int64_t)row * s.S);
  int lo, hi;
  valid_range(row, s, row0, is_causal, window, lo, hi);
  if (key < lo || key >= hi ||
      !select_score(scores[linear], thresholds[row])) return;
  unsigned long long dst64 = atomicAdd(counters, 1ull);
  atomicAdd(counters + 1, 1ull);
  if (dst64 >= (unsigned long long)capacity) {
    if (clamp_overflow) atomicAdd(counters + 2, 1ull);
    atomicExch(counters + 3, 1ull);
    return;
  }
  int dst = (int)dst64;
  SB2_DASSERT(dst >= 0 && dst < capacity);
  pair_rows[dst] = row;
  pair_keys[dst] = key;
}

__global__ void gather_selected_bf16(
    const __nv_bfloat16* q, const __nv_bfloat16* k,
    const int* pair_rows, const int* pair_keys, int pair0, int pairs,
    const unsigned long long* chunk_attempted,
    __nv_bfloat16* qg, __nv_bfloat16* kg, SbShape s) {
  int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = (int64_t)pairs * s.D;
  if (linear >= total) return;
  int p = (int)(linear / s.D);
  int d = (int)(linear - (int64_t)p * s.D);
  unsigned long long attempted = *chunk_attempted;
  int live = attempted > (unsigned long long)pair0
                 ? (int)min(attempted - (unsigned long long)pair0,
                            (unsigned long long)pairs)
                 : 0;
  if (p >= live) {
    qg[linear] = __float2bfloat16(0.f);
    kg[linear] = __float2bfloat16(0.f);
    return;
  }
  int row = pair_rows[pair0 + p];
  int key = pair_keys[pair0 + p];
  SB2_DASSERT(row >= 0 && row < s.rows);
  SB2_DASSERT(key >= 0 && key < s.S);
  int bh = row / s.L;
  int b = bh / s.H;
  int h = bh % s.H;
  int kh = h / s.group;
  qg[linear] = q[(int64_t)row * s.D + d];
  kg[linear] = k[(((int64_t)b * s.KVH + kh) * s.S + key) * s.D + d];
}

__global__ void scatter_refined(float* scores, const float* exact,
                                const int* pair_rows, const int* pair_keys,
                                int pair0, int pairs,
                                const unsigned long long* chunk_attempted,
                                int rows, int S) {
  int p = (int)((int64_t)blockIdx.x * blockDim.x + threadIdx.x);
  if (p >= pairs) return;
  unsigned long long attempted = *chunk_attempted;
  int live = attempted > (unsigned long long)pair0
                 ? (int)min(attempted - (unsigned long long)pair0,
                            (unsigned long long)pairs)
                 : 0;
  if (p >= live) return;
  int src = pair0 + p;
  SB2_DASSERT(pair_rows[src] >= 0 && pair_rows[src] < rows);
  SB2_DASSERT(pair_keys[src] >= 0 && pair_keys[src] < S);
  scores[(int64_t)pair_rows[src] * S + pair_keys[src]] = exact[p];
}

void refine_pairs(const NDArray& q, const NDArray& k, NDArray& scores,
                  const NDArray& pair_rows, const NDArray& pair_keys,
                  const NDArray& counters, int capacity, float scale,
                  const SbShape& s) {
  if (capacity == 0) return;
  const int cap = std::min(capacity, kRefinePairChunk);
  NDArray qg({cap, (int64_t)s.D}, DType::BFloat16, q.device);
  NDArray kg({cap, (int64_t)s.D}, DType::BFloat16, q.device);
  NDArray exact({cap}, DType::Float32, q.device);
  for (int base = 0; base < capacity; base += cap) {
    int n = std::min(cap, capacity - base);
    int64_t elems = (int64_t)n * s.D;
    gather_selected_bf16<<<(unsigned)((elems + kThreads - 1) / kThreads),
                            kThreads>>>(
        static_cast<const __nv_bfloat16*>(q.data_ptr()),
        static_cast<const __nv_bfloat16*>(k.data_ptr()),
        static_cast<const int*>(pair_rows.data_ptr()),
        static_cast<const int*>(pair_keys.data_ptr()), base, n,
        static_cast<const unsigned long long*>(counters.data_ptr()),
        static_cast<__nv_bfloat16*>(qg.data_ptr()),
        static_cast<__nv_bfloat16*>(kg.data_ptr()), s);
    cuda_check_last("SB1 gather selected Q/K");
    const float beta = 0.f;
    check_blas(cublasGemmStridedBatchedEx(
                   blas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
                   1, 1, s.D, &scale,
                   qg.data_ptr(), CUDA_R_16BF, 1, s.D,
                   kg.data_ptr(), CUDA_R_16BF, s.D, s.D,
                   &beta, exact.data_ptr(), CUDA_R_32F, 1, 1,
                   n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
               "SB1 skinny BF16 selected GEMM");
    scatter_refined<<<(n + kThreads - 1) / kThreads, kThreads>>>(
        static_cast<float*>(scores.data_ptr()),
        static_cast<const float*>(exact.data_ptr()),
        static_cast<const int*>(pair_rows.data_ptr()),
        static_cast<const int*>(pair_keys.data_ptr()), base, n,
        static_cast<const unsigned long long*>(counters.data_ptr()),
        s.rows, s.S);
    cuda_check_last("SB1 scatter refined scores");
  }
}

__global__ void copy_query_chunk(const __nv_bfloat16* q,
                                 __nv_bfloat16* qchunk, int B, int H,
                                 int full_L, int chunk_L, int D, int q0) {
  int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = (int64_t)B * H * chunk_L * D;
  if (linear >= total) return;
  int d = (int)(linear % D);
  int64_t t = linear / D;
  int i = (int)(t % chunk_L);
  int h = (int)((t / chunk_L) % H);
  int b = (int)(t / ((int64_t)chunk_L * H));
  SB2_DASSERT(q0 + i >= 0 && q0 + i < full_L);
  qchunk[linear] = q[(((int64_t)b * H + h) * full_L + q0 + i) * D + d];
}

__global__ void copy_output_chunk(const __nv_bfloat16* chunk,
                                  __nv_bfloat16* out, int B, int H,
                                  int full_L, int chunk_L, int VD, int q0) {
  int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = (int64_t)B * H * chunk_L * VD;
  if (linear >= total) return;
  int d = (int)(linear % VD);
  int64_t t = linear / VD;
  int i = (int)(t % chunk_L);
  int h = (int)((t / chunk_L) % H);
  int b = (int)(t / ((int64_t)chunk_L * H));
  SB2_DASSERT(q0 + i >= 0 && q0 + i < full_L);
  out[(((int64_t)b * H + h) * full_L + q0 + i) * VD + d] = chunk[linear];
}

void bf16_exact_scores(const NDArray& q, const NDArray& k, NDArray& exact,
                       float scale, const SbShape& s) {
  const float beta = 0.f;
  const int64_t stride_q = (int64_t)s.M * s.D;
  const int64_t stride_k = (int64_t)s.S * s.D;
  const int64_t stride_c = (int64_t)s.M * s.S;
  // Row-major [M,D]@[S,D]^T is column-major [S,M] = K * Q^T.
  check_blas(cublasGemmStridedBatchedEx(
                 blas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                 s.S, s.M, s.D, &scale,
                 k.data_ptr(), CUDA_R_16BF, s.D, stride_k,
                 q.data_ptr(), CUDA_R_16BF, s.D, stride_q,
                 &beta, exact.data_ptr(), CUDA_R_32F, s.S, stride_c,
                 s.batches, CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP),
             "SB2 dense exact BF16->FP32 GEMM");
}

__global__ void blend_dense_exact(float* scores, const float* exact,
                                  const float* thresholds, SbShape s,
                                  int is_causal, int64_t row0, int window) {
  int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = (int64_t)s.rows * s.S;
  if (linear >= total) return;
  int row = (int)(linear / s.S);
  int key = (int)(linear - (int64_t)row * s.S);
  int lo, hi;
  valid_range(row, s, row0, is_causal, window, lo, hi);
  if (key >= lo && key < hi &&
      select_score(scores[linear], thresholds[row]))
    scores[linear] = exact[linear];
}

__global__ void fp32_score_softmax_bf16(const float* scores,
                                        __nv_bfloat16* probs, SbShape s,
                                        int is_causal, int64_t row0,
                                        int window) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int lo, hi;
  valid_range(row, s, row0, is_causal, window, lo, hi);
  const float* in = scores + (int64_t)row * s.S;
  __nv_bfloat16* out = probs + (int64_t)row * s.S;
  float local_max = -3.402823466e38f;
  for (int j = lo + tid; j < hi; j += blockDim.x)
    local_max = fmaxf(local_max, in[j]);
  __shared__ float red[kThreads];
  red[tid] = local_max;
  __syncthreads();
  for (int off = blockDim.x / 2; off; off >>= 1) {
    if (tid < off) red[tid] = fmaxf(red[tid], red[tid + off]);
    __syncthreads();
  }
  float mx = red[0];
  float sum = 0.f;
  for (int j = lo + tid; j < hi; j += blockDim.x)
    sum += __expf(fmaxf(in[j] - mx, -88.f));
  red[tid] = sum;
  __syncthreads();
  for (int off = blockDim.x / 2; off; off >>= 1) {
    if (tid < off) red[tid] += red[tid + off];
    __syncthreads();
  }
  float inv = red[0] > 0.f ? 1.f / red[0] : 0.f;
  for (int j = tid; j < s.S; j += blockDim.x) {
    float p = (j >= lo && j < hi)
                  ? __expf(fmaxf(in[j] - mx, -88.f)) * inv
                  : 0.f;
    out[j] = __float2bfloat16(p);
  }
}

__global__ void widen_accum_and_mask(const int32_t* accum,
                                     const float* scores,
                                     const float* thresholds,
                                     int64_t* wide, uint8_t* selected,
                                     SbShape s, int is_causal,
                                     int64_t row0, int window) {
  int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = (int64_t)s.rows * s.S;
  if (linear >= total) return;
  int row = (int)(linear / s.S);
  int j = (int)(linear - (int64_t)row * s.S);
  int i = row % s.L;
  int bh = row / s.L;
  int b = bh / s.H;
  int h = bh % s.H;
  int kh = h / s.group;
  int mrow = (h % s.group) * s.L + i;
  int batch = b * s.KVH + kh;
  wide[linear] = (int64_t)accum[((int64_t)batch * s.M + mrow) * s.Spad + j];
  int lo, hi;
  valid_range(row, s, row0, is_causal, window, lo, hi);
  selected[linear] = (j >= lo && j < hi &&
                      select_score(scores[linear], thresholds[row])) ? 1 : 0;
}

struct BulkState {
  NDArray qcodes, qscales, kcodes, kscales, accum, scores, thresholds, counts;
};

BulkState make_bulk(const NDArray& q, const NDArray& k, float scale,
                    float zthr, bool is_causal, int64_t row0, int window,
                    const SbShape& s, const NDArray* cached_codes,
                    const NDArray* cached_scales) {
  BulkState x{
      NDArray(q.shape, DType::Uint8, q.device),
      NDArray({q.shape[0], q.shape[1], q.shape[2]}, DType::Float32, q.device),
      NDArray(), NDArray(),
      NDArray({s.batches, s.M, s.Spad}, DType::Float32, q.device),
      NDArray({q.shape[0], q.shape[1], q.shape[2], k.shape[2]},
              DType::Float32, q.device),
      NDArray({s.rows}, DType::Float32, q.device),
      NDArray({s.rows}, DType::Float32, q.device)};
  quantize(q, x.qcodes, x.qscales);
  if (cached_codes) {
    validate_cached_k(k, *cached_codes, *cached_scales);
    x.kcodes = *cached_codes;
    x.kscales = *cached_scales;
  } else {
    x.kcodes = NDArray(k.shape, DType::Uint8, k.device);
    x.kscales = NDArray({k.shape[0], k.shape[1], k.shape[2]},
                        DType::Float32, k.device);
    quantize(k, x.kcodes, x.kscales);
  }
  lt_int8_bulk(x.qcodes, x.kcodes, x.accum, s);
  scale_stats_count<<<s.rows, kThreads>>>(
      static_cast<const int32_t*>(x.accum.data_ptr()),
      static_cast<const float*>(x.qscales.data_ptr()),
      static_cast<const float*>(x.kscales.data_ptr()),
      static_cast<float*>(x.scores.data_ptr()),
      static_cast<float*>(x.thresholds.data_ptr()),
      static_cast<int*>(x.counts.data_ptr()), s, scale, zthr,
      is_causal ? 1 : 0, row0, window);
  cuda_check_last("SB1 scale/stats/select-count");
  return x;
}

unsigned long long g_selected = 0;
unsigned long long g_valid = 0;
unsigned long long g_overflow_calls = 0;
unsigned long long g_dropped = 0;
std::mutex g_stats_mutex;

double nominal_refine_fraction(float zthr) {
  if (!std::isfinite(zthr)) return zthr < 0.f ? 1.0 : 0.0;
  return std::max(0.0, std::min(1.0,
      0.5 * std::erfc((double)zthr * 0.70710678118654752440)));
}

int choose_query_chunk(const SbShape& full, float zthr) {
  const double cap_fraction = std::min(1.0, 2.0 * nominal_refine_fraction(zthr));
  // accum + bulk scores + optional dense exact + BF16 probabilities + compact
  // row/key indices. This deliberately assumes the dense fallback so changing
  // route cannot violate the budget.
  const double bytes_per_score = 4.0 + 4.0 + 4.0 + 2.0 + 8.0 * cap_fraction;
  const double bytes_per_query_position =
      (double)full.B * full.H * full.S * bytes_per_score;
  int chunk = bytes_per_query_position > 0.0
                  ? (int)(kChunkTransientBudget / bytes_per_query_position)
                  : full.L;
  return std::max(1, std::min(full.L, chunk));
}

int compact_capacity(unsigned long long valid, float zthr) {
  const double fraction = std::min(1.0, 2.0 * nominal_refine_fraction(zthr));
  const double raw = std::ceil(fraction * (double)valid);
  if (raw >= (double)std::numeric_limits<int>::max())
    return std::numeric_limits<int>::max();
  return (int)raw;
}

}  // namespace

std::pair<NDArray, NDArray> apa_gemm_selective_quantize_k(const NDArray& k) {
  if (k.ndim() != 4 || k.dtype != DType::BFloat16 || !k.device.is_cuda() ||
      k.shape[0] <= 0 || k.shape[1] <= 0 || k.shape[2] <= 0 ||
      k.shape[3] <= 0 || (k.shape[3] & 3) != 0)
    throw std::runtime_error(
        "apa_gemm_selective_quantize_k requires CUDA BF16 [B,KVH,S,D], D%4=0");
  NDArray codes(k.shape, DType::Uint8, k.device);
  NDArray scales({k.shape[0], k.shape[1], k.shape[2]},
                 DType::Float32, k.device);
  quantize(k, codes, scales);
  return {codes, scales};
}

NDArray apa_gemm_selective_attention(
    const NDArray& q, const NDArray& k, const NDArray& v, float scale,
    float zthr, bool is_causal, int Lq, int64_t row0, int window,
    const NDArray* cached_codes, const NDArray* cached_scales) {
  if ((cached_codes == nullptr) != (cached_scales == nullptr))
    throw std::runtime_error("apa_gemm_selective requires both cached K codes and scales");
  SbShape s = validate(q, k, &v, is_causal, Lq, row0, window,
                       "apa_gemm_selective_attention");
  NDArray owned_kcodes, owned_kscales;
  const NDArray* use_kcodes = cached_codes;
  const NDArray* use_kscales = cached_scales;
  if (cached_codes) {
    validate_cached_k(k, *cached_codes, *cached_scales);
  } else {
    owned_kcodes = NDArray(k.shape, DType::Uint8, k.device);
    owned_kscales = NDArray({k.shape[0], k.shape[1], k.shape[2]},
                            DType::Float32, k.device);
    quantize(k, owned_kcodes, owned_kscales);
    use_kcodes = &owned_kcodes;
    use_kscales = &owned_kscales;
  }

  NDArray out({q.shape[0], q.shape[1], q.shape[2], v.shape[3]},
              DType::BFloat16, q.device);
  NDArray counters({4}, DType::Int64, q.device);
  check_cuda(cudaMemsetAsync(counters.data_ptr(), 0, 4 * sizeof(uint64_t), 0),
             "SB2 initialize call counters");

  const int query_chunk = choose_query_chunk(s, zthr);
  unsigned long long valid_total = 0;
  for (int q0 = 0; q0 < s.L; q0 += query_chunk) {
    const int lc = std::min(query_chunk, s.L - q0);
    NDArray qchunk({s.B, s.H, lc, s.D}, DType::BFloat16, q.device);
    int64_t qelems = qchunk.numel();
    copy_query_chunk<<<(unsigned)((qelems + kThreads - 1) / kThreads),
                       kThreads>>>(
        static_cast<const __nv_bfloat16*>(q.data_ptr()),
        static_cast<__nv_bfloat16*>(qchunk.data_ptr()), s.B, s.H, s.L,
        lc, s.D, q0);
    cuda_check_last("SB2 copy query sub-chunk");

    SbShape cs = s;
    cs.L = lc;
    cs.rows = s.B * s.H * lc;
    cs.M = s.group * lc;
    const int64_t chunk_row0 = row0 + q0;
    BulkState x = make_bulk(qchunk, k, scale, zthr, is_causal, chunk_row0,
                            window, cs, use_kcodes, use_kscales);
    const unsigned long long chunk_valid =
        valid_pairs(cs, is_causal, chunk_row0, window);
    valid_total += chunk_valid;
    const int capacity = compact_capacity(chunk_valid, zthr);
    NDArray pair_rows({capacity}, DType::Float32, q.device);
    NDArray pair_keys({capacity}, DType::Float32, q.device);
    const size_t gather_bytes = (size_t)capacity *
        ((size_t)2 * cs.D * sizeof(__nv_bfloat16) + sizeof(float));
    const bool sparse_refine = gather_bytes <= kSparseGatherBudget;
    check_cuda(cudaMemsetAsync(counters.data_ptr(), 0, sizeof(uint64_t), 0),
               "SB2 reset chunk append cursor");
    int64_t matrix_elems = (int64_t)cs.rows * cs.S;
    compact_pairs_bounded<<<
        (unsigned)((matrix_elems + kThreads - 1) / kThreads), kThreads>>>(
        static_cast<const float*>(x.scores.data_ptr()),
        static_cast<const float*>(x.thresholds.data_ptr()),
        static_cast<int*>(pair_rows.data_ptr()),
        static_cast<int*>(pair_keys.data_ptr()),
        static_cast<unsigned long long*>(counters.data_ptr()), capacity,
        sparse_refine ? 1 : 0, cs, is_causal ? 1 : 0, chunk_row0, window);
    cuda_check_last("SB2 bounded compact selected pairs");

    if (sparse_refine) {
      x.accum = NDArray();
      refine_pairs(qchunk, k, x.scores, pair_rows, pair_keys, counters,
                   capacity, scale, cs);
    } else {
      NDArray exact({cs.batches, cs.M, cs.S}, DType::Float32, q.device);
      bf16_exact_scores(qchunk, k, exact, scale, cs);
      blend_dense_exact<<<
          (unsigned)((matrix_elems + kThreads - 1) / kThreads), kThreads>>>(
          static_cast<float*>(x.scores.data_ptr()),
          static_cast<const float*>(exact.data_ptr()),
          static_cast<const float*>(x.thresholds.data_ptr()), cs,
          is_causal ? 1 : 0, chunk_row0, window);
      cuda_check_last("SB2 blend dense exact selected scores");
      x.accum = NDArray();
    }

    NDArray probs({cs.B, cs.H, cs.L, cs.S}, DType::BFloat16, q.device);
    fp32_score_softmax_bf16<<<cs.rows, kThreads>>>(
        static_cast<const float*>(x.scores.data_ptr()),
        static_cast<__nv_bfloat16*>(probs.data_ptr()), cs,
        is_causal ? 1 : 0, chunk_row0, window);
    cuda_check_last("SB2 fp32-score softmax");
    NDArray pg = probs.reshape({cs.B, cs.KVH, cs.M, cs.S});
    NDArray outg = matmul(pg, v, 1.f, false);
    NDArray outchunk = outg.reshape({cs.B, cs.H, cs.L, cs.VD});
    int64_t oelems = outchunk.numel();
    copy_output_chunk<<<(unsigned)((oelems + kThreads - 1) / kThreads),
                        kThreads>>>(
        static_cast<const __nv_bfloat16*>(outchunk.data_ptr()),
        static_cast<__nv_bfloat16*>(out.data_ptr()), s.B, s.H, s.L,
        lc, s.VD, q0);
    cuda_check_last("SB2 copy output sub-chunk");
  }

  // The only device-to-host synchronization in an attention call. No compact
  // count is read before or between refine launches.
  unsigned long long host_counters[4] = {};
  check_cuda(cudaMemcpy(host_counters, counters.data_ptr(),
                        sizeof(host_counters), cudaMemcpyDeviceToHost),
             "SB2 call stats readback");
  {
    std::lock_guard<std::mutex> lock(g_stats_mutex);
    g_selected += host_counters[1];
    g_valid += valid_total;
    g_dropped += host_counters[2];
    g_overflow_calls += host_counters[3] ? 1ull : 0ull;
  }
  return out;
}

std::tuple<unsigned long long, unsigned long long,
           unsigned long long, unsigned long long>
apa_gemm_selective_stats(bool reset) {
  std::lock_guard<std::mutex> lock(g_stats_mutex);
  auto out = std::make_tuple(g_selected, g_valid, g_overflow_calls, g_dropped);
  if (reset) g_selected = g_valid = g_overflow_calls = g_dropped = 0;
  return out;
}

std::tuple<NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray>
apa_gemm_selective_debug(const NDArray& q, const NDArray& k, float scale,
                         float zthr, bool is_causal, int Lq,
                         int64_t row0, int window) {
  SbShape s = validate(q, k, nullptr, is_causal, Lq, row0, window,
                       "apa_gemm_selective_debug");
  BulkState x = make_bulk(q, k, scale, zthr, is_causal, row0, window, s,
                          nullptr, nullptr);
  NDArray wide({q.shape[0], q.shape[1], q.shape[2], k.shape[2]},
               DType::Int64, q.device);
  NDArray selected(wide.shape, DType::Uint8, q.device);
  int64_t n = (int64_t)s.rows * s.S;
  widen_accum_and_mask<<<(unsigned)((n + kThreads - 1) / kThreads),
                          kThreads>>>(
      static_cast<const int32_t*>(x.accum.data_ptr()),
      static_cast<const float*>(x.scores.data_ptr()),
      static_cast<const float*>(x.thresholds.data_ptr()),
      static_cast<int64_t*>(wide.data_ptr()),
      static_cast<uint8_t*>(selected.data_ptr()), s,
      is_causal ? 1 : 0, row0, window);
  cuda_check_last("SB1 debug widen/mask");
  return {x.qcodes, x.qscales, x.kcodes, x.kscales,
          wide, x.scores, selected};
}

}  // namespace tc
