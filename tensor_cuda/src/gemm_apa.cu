// APAMQ-SB1: cuBLASLt INT8 bulk + compact gather/BF16 refine.
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

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <algorithm>
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
    local_count += fabsf(out[j]) >= thr;
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

__global__ void compact_pairs(const float* scores, const float* thresholds,
                              int* cursors, int* pair_rows, int* pair_keys,
                              SbShape s, int is_causal, int64_t row0,
                              int window) {
  int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = (int64_t)s.rows * s.S;
  if (linear >= total) return;
  int row = (int)(linear / s.S);
  int key = (int)(linear - (int64_t)row * s.S);
  int lo, hi;
  valid_range(row, s, row0, is_causal, window, lo, hi);
  if (key < lo || key >= hi || fabsf(scores[linear]) < thresholds[row]) return;
  int dst = atomicAdd(cursors + row, 1);
  pair_rows[dst] = row;
  pair_keys[dst] = key;
}

__global__ void gather_selected_bf16(
    const __nv_bfloat16* q, const __nv_bfloat16* k,
    const int* pair_rows, const int* pair_keys, int pair0, int pairs,
    __nv_bfloat16* qg, __nv_bfloat16* kg, SbShape s) {
  int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = (int64_t)pairs * s.D;
  if (linear >= total) return;
  int p = (int)(linear / s.D);
  int d = (int)(linear - (int64_t)p * s.D);
  int row = pair_rows[pair0 + p];
  int key = pair_keys[pair0 + p];
  int bh = row / s.L;
  int b = bh / s.H;
  int h = bh % s.H;
  int kh = h / s.group;
  qg[linear] = q[(int64_t)row * s.D + d];
  kg[linear] = k[(((int64_t)b * s.KVH + kh) * s.S + key) * s.D + d];
}

__global__ void scatter_refined(float* scores, const float* exact,
                                const int* pair_rows, const int* pair_keys,
                                int pair0, int pairs, int S) {
  int p = (int)((int64_t)blockIdx.x * blockDim.x + threadIdx.x);
  if (p >= pairs) return;
  int src = pair0 + p;
  scores[(int64_t)pair_rows[src] * S + pair_keys[src]] = exact[p];
}

void refine_pairs(const NDArray& q, const NDArray& k, NDArray& scores,
                  const NDArray& pair_rows, const NDArray& pair_keys,
                  int total_pairs, float scale, const SbShape& s) {
  if (total_pairs == 0) return;
  const int cap = std::min(total_pairs, kRefinePairChunk);
  NDArray qg({cap, (int64_t)s.D}, DType::BFloat16, q.device);
  NDArray kg({cap, (int64_t)s.D}, DType::BFloat16, q.device);
  NDArray exact({cap}, DType::Float32, q.device);
  for (int base = 0; base < total_pairs; base += cap) {
    int n = std::min(cap, total_pairs - base);
    int64_t elems = (int64_t)n * s.D;
    gather_selected_bf16<<<(unsigned)((elems + kThreads - 1) / kThreads),
                            kThreads>>>(
        static_cast<const __nv_bfloat16*>(q.data_ptr()),
        static_cast<const __nv_bfloat16*>(k.data_ptr()),
        static_cast<const int*>(pair_rows.data_ptr()),
        static_cast<const int*>(pair_keys.data_ptr()), base, n,
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
        static_cast<const int*>(pair_keys.data_ptr()), base, n, s.S);
    cuda_check_last("SB1 scatter refined scores");
  }
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
                      fabsf(scores[linear]) >= thresholds[row]) ? 1 : 0;
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
std::mutex g_stats_mutex;

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
  BulkState x = make_bulk(q, k, scale, zthr, is_causal, row0, window, s,
                          cached_codes, cached_scales);

  NDArray offsets({(int64_t)s.rows + 1}, DType::Float32, q.device);
  thrust::device_ptr<int> countp(static_cast<int*>(x.counts.data_ptr()));
  thrust::device_ptr<int> offsetp(static_cast<int*>(offsets.data_ptr()));
  thrust::exclusive_scan(countp, countp + s.rows, offsetp);
  int last_offset = 0, last_count = 0;
  check_cuda(cudaMemcpy(&last_offset, offsetp.get() + s.rows - 1, sizeof(int),
                        cudaMemcpyDeviceToHost), "SB1 selected offset readback");
  check_cuda(cudaMemcpy(&last_count, countp.get() + s.rows - 1, sizeof(int),
                        cudaMemcpyDeviceToHost), "SB1 selected count readback");
  if (last_count > std::numeric_limits<int>::max() - last_offset)
    throw std::runtime_error("SB1 compact selected count overflow");
  int total = last_offset + last_count;
  check_cuda(cudaMemcpy(offsetp.get() + s.rows, &total, sizeof(int),
                        cudaMemcpyHostToDevice), "SB1 write final offset");
  {
    std::lock_guard<std::mutex> lock(g_stats_mutex);
    g_selected += (unsigned)total;
    g_valid += valid_pairs(s, is_causal, row0, window);
  }

  NDArray pair_rows({total}, DType::Float32, q.device);
  NDArray pair_keys({total}, DType::Float32, q.device);
  NDArray cursors({s.rows}, DType::Float32, q.device);
  check_cuda(cudaMemcpyAsync(cursors.data_ptr(), offsets.data_ptr(),
                             (size_t)s.rows * sizeof(int),
                             cudaMemcpyDeviceToDevice, 0),
             "SB1 initialize compact cursors");
  int64_t matrix_elems = (int64_t)s.rows * s.S;
  compact_pairs<<<(unsigned)((matrix_elems + kThreads - 1) / kThreads),
                   kThreads>>>(
      static_cast<const float*>(x.scores.data_ptr()),
      static_cast<const float*>(x.thresholds.data_ptr()),
      static_cast<int*>(cursors.data_ptr()),
      static_cast<int*>(pair_rows.data_ptr()),
      static_cast<int*>(pair_keys.data_ptr()), s, is_causal ? 1 : 0,
      row0, window);
  cuda_check_last("SB1 compact selected pairs");
  // The INT32 matrix is dead before the potentially large gather/refine leg.
  x.accum = NDArray();
  refine_pairs(q, k, x.scores, pair_rows, pair_keys, total, scale, s);

  NDArray probs({q.shape[0], q.shape[1], q.shape[2], k.shape[2]},
                DType::BFloat16, q.device);
  fp32_score_softmax_bf16<<<s.rows, kThreads>>>(
      static_cast<const float*>(x.scores.data_ptr()),
      static_cast<__nv_bfloat16*>(probs.data_ptr()), s,
      is_causal ? 1 : 0, row0, window);
  cuda_check_last("SB1 fp32-score softmax");
  NDArray pg = probs.reshape({s.B, s.KVH, s.M, s.S});
  NDArray outg = matmul(pg, v, 1.f, false);
  return outg.reshape({s.B, s.H, s.L, s.VD});
}

std::pair<unsigned long long, unsigned long long>
apa_gemm_selective_stats(bool reset) {
  std::lock_guard<std::mutex> lock(g_stats_mutex);
  auto out = std::make_pair(g_selected, g_valid);
  if (reset) g_selected = g_valid = 0;
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
