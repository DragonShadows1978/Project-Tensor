#include <tuple>
// GENERATED: literal pinned production bodies plus diagnostic stores.
// Prior art: Perry APA 2026, Milakov/Gimelshein 2018, Dao FA2 2023.
#include "tc/core.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
namespace apa_sp4g {
using namespace tc;
template<typename T> __device__ float ld(const T* p,int64_t i) { return (float)p[i]; }
template<typename T> __device__ void st(T* p,int64_t i,float v) { p[i]=(T)v; }
template<typename T,bool BOUNDED>
__global__ void apa_blend_softmax_kernel2(const T* bulk, const T* rank, T* out,
                                          int S, float zthr,
                                          int Lchunk, int Lq, int64_t row0,
                                          int window, uint8_t* selected) {
  int r = blockIdx.x;
  int tid = threadIdx.x, nt = blockDim.x;
  const T* brow = bulk + (int64_t)r * S;
  const T* rrow = rank + (int64_t)r * S;
  T* orow = out + (int64_t)r * S;

  // Masked keys are marked with a large-negative score by the caller (in BOTH
  // bulk and rank); exclude them from the |bulk| mean/std (otherwise they poison
  // the threshold) — they also exp() to ~0 in the softmax so they drop out there.
  // The causal mask bias is -1e4 (functional.py / _cublas_blend_attention); set
  // the cutoff safely between real scaled logits (O(+-50)) and -1e4 so a
  // legitimate very-negative logit is never mistaken for a masked key.
  const float MASK_LIM = -5e3f;
  int lo = 0, hi = S;  // valid key range [lo, hi) when BOUNDED
  if constexpr (BOUNDED) {
    // r flattens as (b*H + h)*Lchunk + i_local over THIS launch's rows only
    // (Lchunk = this chunk's query count, e.g. `blk` in the tiled callers) —
    // Lq (the FULL query length L) is a different quantity, used only to
    // compute the absolute cache-prefix offset S-Lq below. Using Lq here
    // instead of Lchunk was Phase 3.1's first-draft bug: caught by a
    // multi-chunk parity re-run (attn_block < L), max|Δout| ~21 vs ~0.
    int i_local = r % Lchunk;
    int64_t abs_i = row0 + i_local;
    int64_t valid_hi = (int64_t)S - Lq + abs_i + 1;  // bottom-right causal bound
    hi = (int)(valid_hi < 0 ? 0 : (valid_hi > S ? S : valid_hi));
    lo = (window > 0) ? (int)((hi - window) < 0 ? 0 : (hi - window)) : 0;
  }
  __shared__ float red[256];
  float sum = 0.f, sumsq = 0.f, vcount = 0.f;
  if constexpr (BOUNDED) {
    for (int j = lo + tid; j < hi; j += nt) {
      float a = fabsf(ld<T>(brow, j)); sum += a; sumsq += a*a; vcount += 1.f;
    }
  } else {
    for (int j = tid; j < S; j += nt) {
      float bk = ld<T>(brow, j);
      if (bk <= MASK_LIM) continue;
      float a = fabsf(bk); sum += a; sumsq += a*a; vcount += 1.f;
    }
  }
  red[tid] = sum; __syncthreads();
  for (int o = nt/2; o > 0; o >>= 1) { if (tid < o) red[tid]+=red[tid+o]; __syncthreads(); }
  float total = red[0]; __syncthreads();
  red[tid] = sumsq; __syncthreads();
  for (int o = nt/2; o > 0; o >>= 1) { if (tid < o) red[tid]+=red[tid+o]; __syncthreads(); }
  float total_sq = red[0]; __syncthreads();
  red[tid] = vcount; __syncthreads();
  for (int o = nt/2; o > 0; o >>= 1) { if (tid < o) red[tid]+=red[tid+o]; __syncthreads(); }
  float cnt = fmaxf(red[0], 1.f); __syncthreads();
  float mean = total/cnt;
  float thr = mean + zthr * sqrtf(fmaxf(total_sq/cnt - mean*mean, 0.f));


  // SP3 observation only: literal decision, after unchanged statistics.
  for (int j=tid; j<S; j+=nt) {
    float bk=ld<T>(brow,j);
    bool valid=BOUNDED ? (j>=lo && j<hi) : (bk>MASK_LIM);
    selected[(int64_t)r*S+j]=valid && fabsf(bk)>=thr;
  }
  // Single fused pass: compute the blended score per key once, track running
  // max + online-softmax denom (FlashAttention merge), write the UNnormalized
  // exp weight. Final normalization (1/denom) is folded into the caller's
  // weights@V matmul instead of a 4th re-read/re-write pass — so this kernel now
  // makes 2 passes (stat + this) instead of 4. denom is written per-row to
  // `out`'s companion? No: we renormalize here in one extra cheap reduce, but
  // never re-read the full row — we keep each thread's written weights and the
  // block denom, then a final scale uses the already-resident values via shared.
  float m = -1e30f, l = 0.f;
  // First, the per-thread online softmax over its strided keys (one read each).
  // Store nothing yet; we need the global max before exp. To avoid a separate
  // max pass we use the online-softmax rescale trick across the strided scan.
  // Selection on |bulk|: a masked key (bulk <= MASK_LIM, sentinel path) keeps
  // its large-negative bulk value (exp()s to ~0); otherwise refine to rank
  // when |bulk| >= thr. Bounds path: every key in [lo,hi) is valid by
  // construction, no sentinel check needed.
  if constexpr (BOUNDED) {
    for (int j = lo + tid; j < hi; j += nt) {
      float bk = ld<T>(brow, j);
      float sc = (fabsf(bk) >= thr) ? ld<T>(rrow, j) : bk;
      float m_new = fmaxf(m, sc);
      l = l * __expf(m - m_new) + __expf(sc - m_new);
      m = m_new;
    }
  } else {
    for (int j = tid; j < S; j += nt) {
      float bk = ld<T>(brow, j);
      float sc = (bk <= MASK_LIM) ? bk : ((fabsf(bk) >= thr) ? ld<T>(rrow, j) : bk);
      float m_new = fmaxf(m, sc);
      l = l * __expf(m - m_new) + __expf(sc - m_new);
      m = m_new;
    }
  }
  // merge per-thread (m,l) -> global
  red[tid] = m; __syncthreads();
  for (int o = nt/2; o > 0; o >>= 1) { if (tid < o) red[tid]=fmaxf(red[tid],red[tid+o]); __syncthreads(); }
  float gmax = red[0]; __syncthreads();
  red[tid] = l * __expf(m - gmax); __syncthreads();
  for (int o = nt/2; o > 0; o >>= 1) { if (tid < o) red[tid]+=red[tid+o]; __syncthreads(); }
  float denom = red[0]; __syncthreads();
  float inv = denom > 0.f ? 1.f/denom : 0.f;
  // second pass: write normalized weights (recompute blended score — cheaper
  // than a 3rd read of a stored weight, and exact dots aren't involved here).
  // Clamp sc-gmax at the low end so masked/very-negative scores hit a safe
  // __expf input (expf flushes to 0 below ~-88 for fp32; clamp avoids any
  // denormal/edge behaviour) — weight is ~0 there anyway. Bounds path: every
  // out-of-range column (both below lo and at/above hi) must be written 0 —
  // the caller's cat/matmul reads the full (rows,S) row, so zero it rather
  // than leaving it uninitialized.
  if constexpr (BOUNDED) {
    for (int j = tid; j < S; j += nt) {
      if (j < lo || j >= hi) { st<T>(orow, j, 0.0f); continue; }
      float bk = ld<T>(brow, j);
      float sc = (fabsf(bk) >= thr) ? ld<T>(rrow, j) : bk;
      st<T>(orow, j, __expf(fmaxf(sc - gmax, -88.f)) * inv);
    }
  } else {
    for (int j = tid; j < S; j += nt) {
      float bk = ld<T>(brow, j);
      float sc = (bk <= MASK_LIM) ? bk : ((fabsf(bk) >= thr) ? ld<T>(rrow, j) : bk);
      st<T>(orow, j, __expf(fmaxf(sc - gmax, -88.f)) * inv);
    }
  }
}
template<typename T,int DMAX,bool WCOOP>
__global__ void apa_selective_kernel(
    const T* q, const T* k, const T* kq, const T* v, T* out,
    int B, int H, int L, int S, int D, int VD, float scale, float zthr,
    int is_causal, int KVH, int group, uint8_t* selected, float* bscores) {
  // GQA-aware: q has H query heads; k/kq/v have KVH key/value heads (KVH <= H,
  // group = H/KVH). Query head h reads KV head h/group, so the KV tensors are
  // NEVER expanded to H heads — saves materializing the 4x-repeated k_rep/v_rep.
  int row = blockIdx.x;            // flat (b,h,i) over the Q heads
  int i = row % L;
  int bh = row / L;
  int b = bh / H;
  int h = bh % H;
  int kv_h = h / group;            // which of the KVH heads this query head uses
  int tid = threadIdx.x;
  int nt = blockDim.x;
  const unsigned FULL = 0xffffffffu;
  int lane = tid & 31;
  int warp = tid >> 5;
  int nwarp = nt >> 5;              // == 4 for nt=128

  const T* qrow = q + (int64_t)row * D;
  int64_t kvbh = (int64_t)b * KVH + kv_h;
  const T* kbase = k + kvbh * S * D;
  const T* kqbase = kq + kvbh * S * D;
  const T* vbase = v + kvbh * S * VD;

  // Load this query vector into shared memory (D <= DMAX <= TC_APA_MAXD).
  __shared__ float qsh[DMAX];
  for (int d = tid; d < D; d += nt) qsh[d] = ld<T>(qrow, d);
  __syncthreads();

  // BOTTOM-RIGHT causal: query row i sits at ABSOLUTE key index
  // (S-L)+i (the prefill-continuation / cache regime where S>L), so it
  // sees keys 0..(S-L)+i. The old s_max=i+1 was top-left — correct only
  // at S==L (square, no cache), and SILENTLY blinds queries to the most
  // recent S-L keys otherwise (the 121->11M ppl bug class). Reduces to
  // i+1 exactly when S==L.
  int s_max = is_causal ? ((S - L) + i + 1) : S;
  __shared__ float red[256];

  for (int j=s_max+tid; j<S; j+=nt) selected[(int64_t)row*S+j]=0;
  // Pass 1: bulk scores -> sum/sumsq of |bulk| for the z-score threshold.
  // WCOOP=true: each warp owns a strided subset of keys; within a key, lanes
  // split D contiguously and shfl-reduce the dot (see block comment).
  // WCOOP=false: pre-A5 per-thread strided keys (each thread's sum/sumsq is a
  // genuine disjoint partial).
  float sum = 0.f, sumsq = 0.f;
  if constexpr (WCOOP) {
    for (int j = warp; j < s_max; j += nwarp) {
      const T* kqj = kqbase + (int64_t)j * D;
      float dot = 0.f;
      for (int d = lane; d < D; d += 32) dot += qsh[d] * ld<T>(kqj, d);
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1) dot += __shfl_down_sync(FULL, dot, off);
      dot = __shfl_sync(FULL, dot, 0);   // broadcast the warp's dot to all lanes
      float a = fabsf(dot * scale);
      sum += a; sumsq += a * a;
    }
  } else {
    for (int j = tid; j < s_max; j += nt) {
      const T* kqj = kqbase + (int64_t)j * D;
      float dot = 0.f;
      for (int d = 0; d < D; ++d) dot += qsh[d] * ld<T>(kqj, d);
      float a = fabsf(dot * scale);
      sum += a; sumsq += a * a;
    }
  }
  // WCOOP: sum/sumsq are per-WARP totals, identical across all 32 lanes of
  // the owning warp (redundant, not partial) — only lane 0 contributes, else
  // each warp's contribution is overcounted 32x. Per-thread: all contribute.
  red[tid] = (WCOOP && lane != 0) ? 0.f : sum; __syncthreads();
  for (int off = nt / 2; off > 0; off >>= 1) { if (tid < off) red[tid] += red[tid+off]; __syncthreads(); }
  float total = red[0]; __syncthreads();
  red[tid] = (WCOOP && lane != 0) ? 0.f : sumsq; __syncthreads();
  for (int off = nt / 2; off > 0; off >>= 1) { if (tid < off) red[tid] += red[tid+off]; __syncthreads(); }
  float total_sq = red[0]; __syncthreads();

  float cnt = (float)s_max;
  float mean = total / cnt;
  float var = total_sq / cnt - mean * mean;
  float thr = mean + zthr * sqrtf(fmaxf(var, 0.f));

  // Pass 2: SINGLE pass with an online softmax (FlashAttention-style).
  // WCOOP=true: per-WARP state, all 32 lanes carrying identical (m, l)
  // (redundant scalar work, cheap) while VALUE accumulation is split across
  // lanes: lane `lane` owns acc slots d = lane, lane+32, ... (ceil(VD/32)
  // registers). WCOOP=false: pre-A5 per-thread state, full acc[VD] each.
  float m = -1e30f, l = 0.f;
  constexpr int ACCN = WCOOP ? ((DMAX + 31) / 32) : DMAX;
  float acc[ACCN];
  if constexpr (WCOOP) {
    #pragma unroll
    for (int dl = 0; dl < ACCN; ++dl) acc[dl] = 0.f;
  } else {
    for (int d = 0; d < VD; ++d) acc[d] = 0.f;
  }

  if constexpr (WCOOP) {
    for (int j = warp; j < s_max; j += nwarp) {
      const T* kqj = kqbase + (int64_t)j * D;
      float bulk = 0.f;
      for (int d = lane; d < D; d += 32) bulk += qsh[d] * ld<T>(kqj, d);
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1) bulk += __shfl_down_sync(FULL, bulk, off);
      bulk = __shfl_sync(FULL, bulk, 0);
      bulk *= scale;
      float score;
      if (lane == 0) { selected[(int64_t)row*S+j] = fabsf(bulk)>=thr; bscores[(int64_t)row*S+j]=bulk; }
      if (fabsf(bulk) >= thr) {
        const T* kj = kbase + (int64_t)j * D;
        float ex = 0.f;
        for (int d = lane; d < D; d += 32) ex += qsh[d] * ld<T>(kj, d);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) ex += __shfl_down_sync(FULL, ex, off);
        ex = __shfl_sync(FULL, ex, 0);
        score = ex * scale;
      } else {
        score = bulk;
      }
      // online softmax update — identical formula on every lane (redundant,
      // not partitioned: m/l are scalars, not worth splitting across lanes).
      float m_new = fmaxf(m, score);
      float corr = __expf(m - m_new);
      float w = __expf(score - m_new);
      l = l * corr + w;
      const T* vj = vbase + (int64_t)j * VD;
      #pragma unroll
      for (int dl = 0; dl < ACCN; ++dl) {
        int d = lane + dl * 32;
        if (d < VD) acc[dl] = acc[dl] * corr + w * ld<T>(vj, d);
      }
      m = m_new;
    }
  } else {
    for (int j = tid; j < s_max; j += nt) {
      const T* kqj = kqbase + (int64_t)j * D;
      float bulk = 0.f;
      for (int d = 0; d < D; ++d) bulk += qsh[d] * ld<T>(kqj, d);
      bulk *= scale;
      float score;
      if (fabsf(bulk) >= thr) {
        const T* kj = kbase + (int64_t)j * D;
        float ex = 0.f;
        for (int d = 0; d < D; ++d) ex += qsh[d] * ld<T>(kj, d);
        score = ex * scale;
      } else {
        score = bulk;
      }
      // online softmax update
      float m_new = fmaxf(m, score);
      float corr = __expf(m - m_new);
      float w = __expf(score - m_new);
      l = l * corr + w;
      const T* vj = vbase + (int64_t)j * VD;
      for (int d = 0; d < VD; ++d) acc[d] = acc[d] * corr + w * ld<T>(vj, d);
      m = m_new;
    }
  }

  // Merge the per-thread (WCOOP=false) or per-warp-replicated (WCOOP=true)
  // online-softmax states. m's max-reduction is duplicate-safe either way;
  // l is a SUM, so under WCOOP only lane 0 of each warp contributes (every
  // lane holds the identical warp-total — 32x overcount otherwise).
  red[tid] = m; __syncthreads();
  for (int off = nt / 2; off > 0; off >>= 1) { if (tid < off) red[tid] = fmaxf(red[tid], red[tid+off]); __syncthreads(); }
  float gmax = red[0]; __syncthreads();

  float rescale = __expf(m - gmax);
  // denom
  red[tid] = (WCOOP && lane != 0) ? 0.f : (l * rescale); __syncthreads();
  for (int off = nt / 2; off > 0; off >>= 1) { if (tid < off) red[tid] += red[tid+off]; __syncthreads(); }
  float denom = red[0]; __syncthreads();
  float inv = denom > 0.f ? 1.f / denom : 0.f;

  // Combine per-warp partials per dim. WCOOP=true: acc is lane-DISJOINT
  // (each lane owns distinct d's), so each lane just writes its owned slots
  // into wpart[warp][d] — no shfl-reduce needed. WCOOP=false (pre-A5): each
  // thread holds a full partial acc[d]; warp shfl-reduce then lane 0 writes.
  // The cross-warp combine below is common to both.
  __shared__ float osh[DMAX];
  __shared__ float wpart[4][DMAX];   // [warp][dim] partials; nt=128 -> 4 warps
  if constexpr (WCOOP) {
    #pragma unroll
    for (int dl = 0; dl < ACCN; ++dl) {
      int d = lane + dl * 32;
      if (d < VD) wpart[warp][d] = acc[dl] * rescale;
    }
  } else {
    for (int d = 0; d < VD; ++d) {
      float val = acc[d] * rescale;
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(FULL, val, off);
      if (lane == 0) wpart[warp][d] = val;
    }
  }
  __syncthreads();
  for (int d = tid; d < VD; d += nt) {
    float s = 0.f;
    for (int w = 0; w < nwarp; ++w) s += wpart[w][d];
    osh[d] = s;
  }
  __syncthreads();

  T* orow = out + (int64_t)row * VD;
  for (int d = tid; d < VD; d += nt) st<T>(orow, d, osh[d] * inv);
}

// Prior art: literal warp/FMA dot order from the existing SP1 prefill kernel.
// Observational score probe only; no attention path or selector changes.
template<typename T> __global__ void bulk_scores_kernel(const T* q,const T* kq,float* o,int H,int L,int S,int D,float scale) {
 int row=blockIdx.x,lane=threadIdx.x,h=(row/L)%H,b=row/(L*H);
 float qr[16];
 #pragma unroll
 for(int t=0;t<16;++t) {int d=lane+32*t;qr[t]=d<D ? ld<T>(q,(int64_t)row*D+d):0.f;}
 for(int j=0;j<S;++j) {
   int64_t koff=((int64_t)b*S+j)*D;float dot=0.f;
   #pragma unroll
   for(int t=0;t<16;++t) {int d=lane+32*t;if(d<D) dot+=qr[t]*ld<T>(kq,koff+d);}
   #pragma unroll
   for(int off=16;off>0;off>>=1) dot+=__shfl_down_sync(0xffffffffu,dot,off);
   float bulk=__shfl_sync(0xffffffffu,dot,0)*scale;
   if(lane==0)o[(int64_t)row*S+j]=bulk;
 }
}
NDArray bulk_scores(const NDArray& q,const NDArray& kq,float scale) {
 int B=q.shape[0],H=q.shape[1],L=q.shape[2],S=kq.shape[2],D=q.shape[3];
 NDArray o({B,H,L,S},DType::Float32,q.device);
 #define GO(T) bulk_scores_kernel<T><<<B*H*L,32>>>((T*)q.data_ptr(),(T*)kq.data_ptr(),(float*)o.data_ptr(),H,L,S,D,scale);
 if(q.dtype==DType::BFloat16) { GO(__nv_bfloat16) } else if(q.dtype==DType::Float32) { GO(float) } else { GO(__half) }
 #undef GO
 cuda_check_last("apa_sp4g_bulk_scores");return o;
}
std::pair<NDArray,NDArray> blend(const NDArray& b,const NDArray& r,float z,int Lq,int row0) {
 NDArray o(b.shape,b.dtype,b.device),m(b.shape,DType::Uint8,b.device);
 int S=b.shape[3], L=b.shape[2], rows=b.numel()/S;
 #define GO(T) if(Lq>0) apa_blend_softmax_kernel2<T,true><<<rows,256>>>((T*)b.data_ptr(),(T*)r.data_ptr(),(T*)o.data_ptr(),S,z,L,Lq,row0,0,(uint8_t*)m.data_ptr()); else apa_blend_softmax_kernel2<T,false><<<rows,256>>>((T*)b.data_ptr(),(T*)r.data_ptr(),(T*)o.data_ptr(),S,z,L,0,row0,0,(uint8_t*)m.data_ptr());
 if(b.dtype==DType::BFloat16) { GO(__nv_bfloat16) } else if(b.dtype==DType::Float32) { GO(float) } else { GO(__half) }
 #undef GO
 cuda_check_last("apa_sp4g_blend_diagnostic"); return {o,m};
}
std::tuple<NDArray,NDArray,NDArray> selective(const NDArray& q,const NDArray& k,const NDArray& kq,const NDArray& v,float scale,float z,bool causal) {
 int B=q.shape[0],H=q.shape[1],L=q.shape[2],S=k.shape[2],D=q.shape[3],VD=v.shape[3];
 NDArray o({B,H,L,VD},q.dtype,q.device),m({B,H,L,S},DType::Uint8,q.device),bs({B,H,L,S},DType::Float32,q.device);
 #define GO(T) apa_selective_kernel<T,512,true><<<B*H*L,128>>>((T*)q.data_ptr(),(T*)k.data_ptr(),(T*)kq.data_ptr(),(T*)v.data_ptr(),(T*)o.data_ptr(),B,H,L,S,D,VD,scale,z,causal,1,H,(uint8_t*)m.data_ptr(),(float*)bs.data_ptr());
 if(q.dtype==DType::BFloat16) { GO(__nv_bfloat16) } else if(q.dtype==DType::Float32) { GO(float) } else { GO(__half) }
 #undef GO
 cuda_check_last("apa_sp4g_selective_diagnostic"); return {o,m,bs};
}
} // namespace
