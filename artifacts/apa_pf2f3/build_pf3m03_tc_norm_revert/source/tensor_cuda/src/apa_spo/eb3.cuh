// APA_EB3_BEGIN latency
// APA-EB3: variant-B TQ4 decode at the SAME arithmetic, bitwise, with fewer
// microseconds. Included from spq4.cuh only under TC_APA_EB3 (opt-in build);
// the default paths (spq1::rotate, spq6::bulk_mma_tq, spq4::select_pv,
// spo2::merge_kernel) are not touched and remain the baseline.
//
// What may change (scheduling, memory-access pattern, launch structure and
// resource tuning only); every fp32 operation on every output element keeps
// its operands and its order:
//  1. rotate_q: coalesced query rotation. R rows staged through shared memory
//     once per (batch, KV head, ROT_DB output dims) for all G query heads.
//     Each output keeps spq1::rotate's sequential __fmaf_rn chain i=0..D-1.
//     Optional: L2 prefetch of the packed codes + norms the bulk kernel reads.
//  2. bulk: spq6::bulk_mma_tq<D,2> (FXTC1 variant B). bulk_copy is the
//     kernel instruction-for-instruction (launch-bounds knob only);
//     bulk_swap computes the same tile transposed (swap-AB): A = centroids
//     (16 keys in M), B = query hi/lo (G heads in N). Each output element
//     (head h, key j) is the same mma.sync.m16n8k16 dot product: the same 16
//     products in the same k-slots (dim tig*DPL+4s+e -> slot 2tig+e and
//     dim +2+e -> slot 2tig+8+e), the same three MMAs per step (hi*hi,
//     hi*lo_q, lo_c*hi) into one fp32 accumulator, s ascending, then
//     (acc*norm)*scale. Only the element's (row, column) position in the
//     MMA tile differs (D^T = B^T A^T); half the MMAs because M is no
//     longer 16-row padding for G<=8 heads. G1 is the bitwise arbiter.
//     Optional: L2 prefetch of this chunk's V rows (for select_pv).
//  3. select_pv: spq4::select_pv (default instantiation) S1-S4 verbatim with
//     shared arrays sized by the compile-time group G instead of MAXG=8,
//     min-blocks launch bounds, an L2 prefetch of the chunk's V rows at CTA
//     start, and the Flash-Decoding merge fused into the last CTA of each
//     (batch, KV head) group (threadfence + self-resetting atomicInc ticket).
//
// Prior art (Prior Art Directive; code-site annotations):
//  * TurboQuant (Zandieh et al. 2025, arXiv:2504.19874): rotate-the-query
//    identity <R q, c> scores keys stored as rotated Lloyd-Max codes. TAKEN,
//    unchanged from SPQ1/SPQ6. Ours: only the rotation kernel's schedule.
//  * Shared-memory staging for coalesced matrix access (NVIDIA CUDA C++ Best
//    Practices Guide, coalescing; M. Harris 2013, "An Efficient Matrix
//    Transpose in CUDA C/C++", padded tiles). TAKEN: stage + pad (D+1).
//  * Swap-AB / operand transposition for small-M GEMM tiles (CUTLASS-style
//    "swap A and B" for skinny problems; FlashInfer 2024/2025 GQA M-packing
//    is the opposite choice that SPQ4 inherited). Unverified as a single
//    citation, lead to check: "CUTLASS swap AB small M", "swapAB GEMM
//    tensor core decode". Ours: applying it to the TQ4 LUT bulk tile with the
//    k-slot permutation and the three-term split held fixed.
//  * Flash-Decoding (Dao, Haziza, Massa, Sizov 2023): split-K partials and a
//    rescaled merge. TAKEN, unchanged arithmetic (spo2::merge_kernel).
//  * Threadfence reduction / last-block pattern (NVIDIA CUDA Samples,
//    "threadFenceReduction"; CUDA C++ Programming Guide, Memory Fence
//    Functions example): __threadfence then atomicInc ticket; the last block
//    reduces the partials. TAKEN. Ours: per (batch, KV head) tickets, the
//    atomicInc wrap value NC-1 resets the ticket without a launch, and the
//    last CTA merges G heads with SPQ's ordered chain.
//  * Programmatic dependent launch (CUDA C++ Programming Guide, PDL): NOT
//    used. griddepcontrol requires sm_90; on sm_89 ptxas rejects it (EB3
//    ledger receipt), so no device-side wait exists for an early launch.
//  * Software prefetch into L2 ahead of use (PTX ISA prefetch.global.L2;
//    classic software prefetching, Mowry, Lam, Gupta 1992). TAKEN. Ours: the
//    cross-kernel schedule (rotation prefetches the bulk's codes, bulk
//    prefetches select_pv's V) under an L2-capacity budget.
//  * __launch_bounds__ / occupancy (CUDA C++ Programming Guide). TAKEN.
//  * Mutation testing (DeMillo, Lipton, Sayward 1978): APA_EB3_MUTANT hooks,
//    0 in every non-mutant build.
// No prior art known to me for this exact combination; no novelty claimed.
// APA-P1-F1 R12 (P1R-F12): the persistent merge/choose tickets below assume the
// single-stream ticket contract (ONE stream, the CUDA legacy default stream).
// nvcc --default-stream per-thread defines CUDA_API_PER_THREAD_DEFAULT_STREAM;
// such a build would share ticket slots across concurrent streams, so it is
// refused at compile time. No prior art known to me beyond #error itself.
#if defined(CUDA_API_PER_THREAD_DEFAULT_STREAM)
#error "apa_spo/eb3.cuh: single-stream ticket contract violated: the fused-merge tickets require the CUDA legacy default stream; a per-thread default stream build (CUDA_API_PER_THREAD_DEFAULT_STREAM) is not supported"
#endif
#include "eb3_tuning.cuh"
#ifndef APA_EB3_MUTANT
#define APA_EB3_MUTANT 0
#endif
#ifndef TC_APA_P1_TEST_HOOKS
#define TC_APA_P1_TEST_HOOKS 0
#endif
namespace eb3 {
#if TC_APA_P1_TEST_HOOKS
// Prior art: DeMillo, Lipton and Sayward (1978) mutation testing; CUDA
// fence/atomic happens-before; survey APA-2P S3 (2026), deterministic
// early-counter design. Taken: negative control and explicit ordering;
// ours: integration of its poisoned-partial completion handshake. No prior
// art known to me for this exact fixture. Literature unverified -- lead to
// check: mutation testing 1978; CUDA memory fence atomic release acquire.
// The default build contains NONE of this hook or its kernel parameters.
// APA_P1_EARLY_DEVICE_BEGIN (CPU compiles this actual structural code too)
// signal: the state word chunk 0's completion writes (0 = the word chunk 1
// waits on; 2 = a word nobody waits on, the APA-P1-F1 forced-timeout mode).
struct EarlyCounterHook {unsigned* state=nullptr;unsigned long long ticks=0;unsigned signal=0;};
__device__ __forceinline__ bool early_counter_wait(EarlyCounterHook hook,int c){
  if(!hook.state || c!=1)return true;
  unsigned timed_out=0;
  if(threadIdx.x==0){
    const auto start=clock64();
    while(atomicAdd(hook.state,0u)==0u){
      if(clock64()-start>=hook.ticks){timed_out=1;atomicExch(hook.state+1,1u);break;}
    }
  }
  // CUDA bar.red.or broadcasts the leader timeout without introducing a
  // static shared object into this translation unit (NVIDIA CUDA API).
  return __syncthreads_or(timed_out)!=1;
}
__device__ __forceinline__ void early_counter_done(EarlyCounterHook hook,int c){
  if(!hook.state || c!=0)return;
  // Every writer has finished ALL possible merge/output stores before the
  // leader releases chunk 1. Also called on the non-merging early exit.
  __threadfence();__syncthreads();
  if(threadIdx.x==0)atomicExch(hook.state+hook.signal,1u);
}
// APA_P1_EARLY_DEVICE_END
// APA_P1_EARLY_COUNTER: "1" = the S3 handshake; "timeout" (APA-P1-F1, hooks
// build only) = the same hook with chunk 0's completion written to a word
// chunk 1 never reads, so chunk 1's poll times out deterministically (LF5).
inline int early_counter_mode(){
  const char* flag=std::getenv("APA_P1_EARLY_COUNTER");
  if(flag && std::strcmp(flag,"1")==0)return 1;
  if(flag && std::strcmp(flag,"timeout")==0)return 2;
  return 0;
}
inline EarlyCounterHook early_counter_setup(NDArray& state,const Device& device,
    int bh,int nc,bool eligible,float* pm,float* pl,float* pa,int rows,int vd){
  const int mode=early_counter_mode();
  if(!mode)return {};
  if(bh!=1 || nc!=2 || !eligible)throw std::runtime_error("APA P1 early-counter requires one bh, NC=2, registered fused path");
  // Three state words: [0] completion signal, [1] timeout flag, [2] the
  // forced-timeout mode's unread completion word.
  state=NDArray({12},DType::Uint8,device);int khz=0;
  if(cudaDeviceGetAttribute(&khz,cudaDevAttrClockRate,device.index)!=cudaSuccess || khz<=0)throw std::runtime_error("APA P1 clock rate");
  // Poison all partials: chunk 0 overwrites its own; chunk 1 stays poisoned
  // until the completion signal. 0xffffffff is an explicit fp32 NaN sentinel.
  if(cudaMemsetAsync(state.data_ptr(),0,12,0)!=cudaSuccess ||
     cudaMemsetAsync(pm,0xff,(size_t)rows*nc*4,0)!=cudaSuccess ||
     cudaMemsetAsync(pl,0xff,(size_t)rows*nc*4,0)!=cudaSuccess ||
     cudaMemsetAsync(pa,0xff,(size_t)rows*nc*vd*4,0)!=cudaSuccess)throw std::runtime_error("APA P1 early-counter poison");
  constexpr unsigned long long poll_milliseconds=2000;
  return {(unsigned*)state.data_ptr(),(unsigned long long)khz*poll_milliseconds,mode==2?2u:0u};
}
inline void early_counter_check(EarlyCounterHook hook,const void* kernel){
  if(!hook.state)return;
  int blocks=0,sms=0;
  if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,128,0)!=cudaSuccess)
    throw std::runtime_error("APA P1 early-counter occupancy");
  int device=0;
  if(cudaGetDevice(&device)!=cudaSuccess || cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device)!=cudaSuccess || blocks*sms<2)
    throw std::runtime_error("APA P1 early-counter needs two concurrently resident CTAs");
}
// APA-P1-F1 R5 (P1R-F05): a timed-out chunk returned before its atomicInc,
// so its merge-ticket slot(s) [0, slots) are left non-zero and the next fused
// merge on that slot would fire one CTA early. The slots are reset to zero
// (after the device sync, before the throw), so the next call behaves as in a
// fresh process. Prior art: CUDA Samples threadFenceReduction tickets (reset
// to a known count between uses); ours: the reset on the hook's failure path.
inline void early_counter_finish(EarlyCounterHook hook,unsigned* ctr,int slots){
  if(!hook.state)return;
  unsigned state[2]={0,0};
  if(cudaDeviceSynchronize()!=cudaSuccess || cudaMemcpy(state,hook.state,sizeof(state),cudaMemcpyDeviceToHost)!=cudaSuccess)throw std::runtime_error("APA P1 early-counter completion");
  if(state[1]){
    if(ctr && slots>0 && (cudaMemset(ctr,0,sizeof(unsigned)*(size_t)slots)!=cudaSuccess || cudaDeviceSynchronize()!=cudaSuccess))
      throw std::runtime_error("APA P1 early-counter poll timed out; merge ticket reset FAILED");
    throw std::runtime_error("APA P1 early-counter poll timed out (FAIL, never a mutant kill)");
  }
  if(state[0]!=1)throw std::runtime_error("APA P1 early-counter missing completion signal");
}
#endif
using bf16=__nv_bfloat16;
constexpr int C=spq4::C,NT=spq4::NT,MAXG=spq4::MAXG,PSTRIDE=spq4::PSTRIDE;

// ---- diagnostics: per-CTA timeline (SM id, clock64, globaltimer) ----------
// Used only by the occupancy/profile receipts (TL=true instantiations).
// Prior art: NVIDIA PTX special registers %smid/%globaltimer and clock64;
// no prior art known to me for the CTA-residency integral built on them.
__device__ __forceinline__ unsigned smid_(){unsigned r;asm volatile("mov.u32 %0, %%smid;":"=r"(r));return r;}
__device__ __forceinline__ unsigned long long gtimer_(){unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;}
struct Stamp{unsigned long long c0=0,g0=0;};
template<bool TL> __device__ __forceinline__ void tl_begin(Stamp& s){
  if constexpr(TL){if(threadIdx.x==0){s.c0=clock64();s.g0=gtimer_();}}
}
template<bool TL> __device__ __forceinline__ void tl_end(unsigned long long* tl,int64_t cta,const Stamp& s){
  if constexpr(TL){if(threadIdx.x==0){unsigned long long* p=tl+cta*5;p[0]=smid_();p[1]=s.c0;p[2]=clock64();p[3]=s.g0;p[4]=gtimer_();}}
}
// L2 software prefetch of [base, base+bytes), 128 B lines, strided over
// `workers` threads (PTX prefetch.global.L2; no register, no arithmetic).
__device__ __forceinline__ void l2_prefetch(const void* base,int64_t bytes,int64_t worker,int64_t workers){
  const char* p=reinterpret_cast<const char*>(base);
  for(int64_t off=worker*128;off<bytes;off+=workers*128)asm volatile("prefetch.global.L2 [%0];"::"l"(p+off));
}
struct Prefetch{const void* a=nullptr;int64_t abytes=0;const void* b=nullptr;int64_t bbytes=0;const void* c=nullptr;int64_t cbytes=0;};

// ---- 1. query rotation ------------------------------------------------------
// out[row,d] = fma_{i=0..D-1}(q[row,i], R[kv,d,i], x), x0=0, exactly as
// spq1::rotate (quant.py rotates row vectors by R^T). Grid (B*KVH, D/DB).
// Prior art: TurboQuant 2025 identity (taken); shared-memory staging with a
// D+1 pad (CUDA Best Practices; Harris 2013) (taken). Ours: the CTA shape.
template<int D,int DB> __global__ void __launch_bounds__(128) rotate_q(const bf16* q,const float* r,float* out,int H,int KVH,Prefetch pf){
  static_assert(DB*MAXG<=128 && D%DB==0 && D%4==0,"rotation tile");
  __shared__ float rs[DB][D+1];
  __shared__ float qs[MAXG][D+1];
  int bk=blockIdx.x,b=bk/KVH,kh=bk%KVH,G=H/KVH,d0=blockIdx.y*DB,tid=threadIdx.x;
  const float4* src=reinterpret_cast<const float4*>(r+((int64_t)kh*D+d0)*D);
  for(int i=tid;i<DB*D/4;i+=128){
    int row=(4*i)/D,col=(4*i)%D;
#if APA_EB3_MUTANT==1
    // Mutant 1: R transposed (stages R[kv][i][d] instead of R[kv][d][i]).
    for(int e=0;e<4;++e)rs[row][col+e]=r[((int64_t)kh*D+col+e)*D+d0+row];
#else
    float4 x=src[i];rs[row][col]=x.x;rs[row][col+1]=x.y;rs[row][col+2]=x.z;rs[row][col+3]=x.w;
#endif
  }
  const bf16* qg=q+(int64_t)(b*H+kh*G)*D;
  for(int i=tid;i<G*D;i+=128)qs[i/D][i%D]=__bfloat162float(qg[i]);   // = ld<bf16>, exact
  if(pf.a){
    // Bulk operands (packed codes, then norms) streamed toward L2 while the
    // rotation runs. Budgeted by the host against the L2 size.
    int64_t worker=((int64_t)blockIdx.x*gridDim.y+blockIdx.y)*128+tid,workers=(int64_t)gridDim.x*gridDim.y*128;
    l2_prefetch(pf.a,pf.abytes,worker,workers);
    if(pf.b)l2_prefetch(pf.b,pf.bbytes,worker,workers);
    if(pf.c)l2_prefetch(pf.c,pf.cbytes,worker,workers);
  }
  __syncthreads();
  if(tid<G*DB){
    int g=tid/DB,dd=tid%DB;float x=0;
#if APA_EB3_MUTANT==2
    for(int i=D-1;i>=0;--i)x=__fmaf_rn(qs[g][i],rs[dd][i],x);   // Mutant 2: fma order reversed
#else
    #pragma unroll 16
    for(int i=0;i<D;++i)x=__fmaf_rn(qs[g][i],rs[dd][i],x);
#endif
    out[(int64_t)(b*H+kh*G+g)*D+d0+dd]=x;
  }
}

// ---- 2a. bulk, instruction-for-instruction copy (launch-bounds knob) --------
// spq6::bulk_mma_tq<D,2> verbatim (FXTC1 variant B, mutant hooks removed).
// Used for the 'before' timeline (MINB=1 = baseline footprint).
template<int D,int MINB,bool TL> __global__ void __launch_bounds__(128,MINB) bulk_copy(const float* qr,const uint8_t* packed,
 const float* scales,const float* cb,float* values,float* cm,int H,int KVH,int S,int NC,int capacity,float scale,unsigned long long* tl){
  constexpr int DPL=D/4,NS=DPL/4,NW=DPL/8;
  __shared__ float wmax[4][MAXG];
  __shared__ unsigned ctab[256];
  __shared__ unsigned ctlo[256];
  int bh=blockIdx.x,c=blockIdx.y+gridDim.y*blockIdx.z;if(c>=NC)return;
  Stamp st;tl_begin<TL>(st);
  int G=H/KVH,b=bh/KVH,kvh=bh%KVH,h0=kvh*G,tid=threadIdx.x,lane=tid&31,warp=tid>>5,r=lane>>2,tig=lane&3;
  int start=c*C,count=min(C,S-start);
  for(int i=tid;i<256;i+=128){
    ctab[i]=spq6::pack_bf2(cb[i&15],cb[i>>4]);
    int ia=i&15,ib=i>>4;float c0=cb[ia],c1=cb[ib];
    ctlo[i]=spq6::pack_bf2(c0-spq6::bf16_val(c0),c1-spq6::bf16_val(c1));
  }
  uint2 qh[NS];
  uint2 ql[NS];
  #pragma unroll
  for(int s=0;s<NS;++s){qh[s]=make_uint2(0u,0u);ql[s]=make_uint2(0u,0u);}
  if(r<G){
    int qrow=b*H+h0+r;
    const float4* qp=reinterpret_cast<const float4*>(qr+(int64_t)qrow*D+tig*DPL);
    #pragma unroll
    for(int s=0;s<NS;++s){
      float4 f=qp[s];
      unsigned x=spq6::pack_bf2(f.x,f.y),y=spq6::pack_bf2(f.z,f.w);qh[s]=make_uint2(x,y);
      ql[s]=make_uint2(spq6::pack_bf2(f.x-spq6::bf2_lo(x),f.y-spq6::bf2_hi(x)),spq6::pack_bf2(f.z-spq6::bf2_lo(y),f.w-spq6::bf2_hi(y)));
    }
  }
  float mx=-INFINITY;
  const uint8_t* pk=packed+((int64_t)bh*capacity+start)*(D/2)+tig*(DPL/2);
  unsigned w[8][NW];
  #pragma unroll
  for(int t=0;t<8;++t){
    int j=(warp+4*t)*8+r;
    #pragma unroll
    for(int i=0;i<NW;++i)w[t][i]=0u;
    if(j<count){
      if constexpr(NW==4){uint4 u=*reinterpret_cast<const uint4*>(pk+(int64_t)j*(D/2));w[t][0]=u.x;w[t][1]=u.y;w[t][2]=u.z;w[t][3]=u.w;}
      else{uint2 u=*reinterpret_cast<const uint2*>(pk+(int64_t)j*(D/2));w[t][0]=u.x;w[t][1]=u.y;}
    }
  }
  float ks[8][2];
  #pragma unroll
  for(int t=0;t<8;++t){
    #pragma unroll
    for(int i=0;i<2;++i){
      int j=(warp+4*t)*8+2*tig+i;ks[t][i]=0.f;
      if(j<count)ks[t][i]=scales[(int64_t)bh*capacity+start+j];
    }
  }
  __syncthreads();   // ctab ready
  #pragma unroll
  for(int t=0;t<8;++t){
    float acc[4]={0.f,0.f,0.f,0.f};
    #pragma unroll
    for(int s=0;s<NS;++s){
      unsigned x=(w[t][s>>1]>>(16*(s&1)))&0xffffu;
      unsigned b0=ctab[x&0xffu],b1=ctab[x>>8];
      const uint2 qa=qh[s],qb=ql[s];
      spq4::mma_bf16(acc,qa.x,qa.y,b0,b1);                 // q_hi * C_hi
      spq4::mma_bf16(acc,qb.x,qb.y,b0,b1);                 // q_lo * C_hi
      unsigned l0=ctlo[x&0xffu],l1=ctlo[x>>8];
      spq4::mma_bf16(acc,qa.x,qa.y,l0,l1);                 // q_hi * C_lo
    }
    #pragma unroll
    for(int i=0;i<2;++i){
      int j=(warp+4*t)*8+2*tig+i;
      if(r<G && j<count){
        float x=__fmul_rn(__fmul_rn(acc[i],ks[t][i]),scale);
        values[(int64_t)(b*H+h0+r)*S+start+j]=x;mx=fmaxf(mx,x);
      }
    }
  }
  mx=fmaxf(mx,__shfl_xor_sync(0xffffffffu,mx,1));mx=fmaxf(mx,__shfl_xor_sync(0xffffffffu,mx,2));
  if(tig==0 && r<MAXG)wmax[warp][r]=mx;
  __syncthreads();
  if(tid<G)cm[(int64_t)(b*H+h0+tid)*NC+c]=fmaxf(fmaxf(wmax[0][tid],wmax[1][tid]),fmaxf(wmax[2][tid],wmax[3][tid]));
  tl_end<TL>(tl,(int64_t)bh*NC+c,st);
}

// ---- 2b. bulk, swap-AB tile (same products, same k-slots, half the MMAs) ----
// A fragment (16x16, row = key): a0 = (key mt*16+r, slots 2tig..+1),
// a1 = (key mt*16+8+r, same slots), a2/a3 = slots 2tig+8..+9. B fragment
// (16x8, col = head): b0/b1 = the query hi (or lo) words SPQ6 used as a0/a2.
// D fragment: c0,c1 = (key r, heads 2tig, 2tig+1); c2,c3 = (key r+8, same).
// PTX ISA m16n8k16 bf16 fragment layouts (taken verbatim).
__device__ __forceinline__ void mma_full(float (&c)[4],unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1){
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
template<int D,int MINB,bool XPF,bool TL> __global__ void __launch_bounds__(128,MINB) bulk_swap(const float* qr,const uint8_t* packed,
 const float* scales,const float* cb,float* values,float* cm,int H,int KVH,int S,int NC,int capacity,float scale,
 const bf16* vpf,int VD,unsigned long long* tl){
  constexpr int DPL=D/4,NS=DPL/4,NW=DPL/8,MT=C/16/4;   // m-tiles of 16 keys per warp
  __shared__ float wmax[4][MAXG];
  __shared__ unsigned ctab[256];
  __shared__ unsigned ctlo[256];
  int bh=blockIdx.x,c=blockIdx.y+gridDim.y*blockIdx.z;if(c>=NC)return;
  Stamp st;tl_begin<TL>(st);
  int G=H/KVH,b=bh/KVH,kvh=bh%KVH,h0=kvh*G,tid=threadIdx.x,lane=tid&31,warp=tid>>5,r=lane>>2,tig=lane&3;
  int start=c*C,count=min(C,S-start);
  if constexpr(XPF){
    // select_pv's V rows for this chunk, toward L2 (host-budgeted).
    if(vpf)l2_prefetch(vpf+((int64_t)bh*S+start)*VD,(int64_t)count*VD*(int64_t)sizeof(bf16),tid,NT);
  }
  for(int i=tid;i<256;i+=128){
    ctab[i]=spq6::pack_bf2(cb[i&15],cb[i>>4]);
    int ia=i&15,ib=i>>4;float c0=cb[ia],c1=cb[ib];
    ctlo[i]=spq6::pack_bf2(c0-spq6::bf16_val(c0),c1-spq6::bf16_val(c1));
  }
  uint2 qh[NS];
  uint2 ql[NS];
  #pragma unroll
  for(int s=0;s<NS;++s){qh[s]=make_uint2(0u,0u);ql[s]=make_uint2(0u,0u);}
  if(r<G){
    int qrow=b*H+h0+r;
    const float4* qp=reinterpret_cast<const float4*>(qr+(int64_t)qrow*D+tig*DPL);
    #pragma unroll
    for(int s=0;s<NS;++s){
      float4 f=qp[s];
      unsigned x=spq6::pack_bf2(f.x,f.y),y=spq6::pack_bf2(f.z,f.w);qh[s]=make_uint2(x,y);
      ql[s]=make_uint2(spq6::pack_bf2(f.x-spq6::bf2_lo(x),f.y-spq6::bf2_hi(x)),spq6::pack_bf2(f.z-spq6::bf2_lo(y),f.w-spq6::bf2_hi(y)));
    }
  }
  const uint8_t* pk=packed+((int64_t)bh*capacity+start)*(D/2)+tig*(DPL/2);
  unsigned w0[MT][NW],w1[MT][NW];
  float k0[MT],k1[MT];
  #pragma unroll
  for(int t=0;t<MT;++t){
    int j0=(warp+4*t)*16+r,j1=j0+8;
    #pragma unroll
    for(int i=0;i<NW;++i){w0[t][i]=0u;w1[t][i]=0u;}
    if(j0<count){uint4 u=*reinterpret_cast<const uint4*>(pk+(int64_t)j0*(D/2));w0[t][0]=u.x;w0[t][1]=u.y;w0[t][2]=u.z;w0[t][3]=u.w;}
    if(j1<count){uint4 u=*reinterpret_cast<const uint4*>(pk+(int64_t)j1*(D/2));w1[t][0]=u.x;w1[t][1]=u.y;w1[t][2]=u.z;w1[t][3]=u.w;}
    k0[t]=j0<count?scales[(int64_t)bh*capacity+start+j0]:0.f;
    k1[t]=j1<count?scales[(int64_t)bh*capacity+start+j1]:0.f;
  }
  __syncthreads();   // ctab ready
  float mx0=-INFINITY,mx1=-INFINITY;   // heads 2tig, 2tig+1
  #pragma unroll
  for(int t=0;t<MT;++t){
    float acc[4]={0.f,0.f,0.f,0.f};
    #pragma unroll
    for(int s=0;s<NS;++s){
      unsigned x0=(w0[t][s>>1]>>(16*(s&1)))&0xffffu,x1=(w1[t][s>>1]>>(16*(s&1)))&0xffffu;
      unsigned a0=ctab[x0&0xffu],a2=ctab[x0>>8],a1=ctab[x1&0xffu],a3=ctab[x1>>8];
      mma_full(acc,a0,a1,a2,a3,qh[s].x,qh[s].y);            // C_hi * q_hi
      mma_full(acc,a0,a1,a2,a3,ql[s].x,ql[s].y);            // C_hi * q_lo
      unsigned l0=ctlo[x0&0xffu],l2=ctlo[x0>>8],l1=ctlo[x1&0xffu],l3=ctlo[x1>>8];
      mma_full(acc,l0,l1,l2,l3,qh[s].x,qh[s].y);            // C_lo * q_hi
    }
    int j0=(warp+4*t)*16+r,j1=j0+8;
    #pragma unroll
    for(int e=0;e<2;++e){
      int head=2*tig+e;
      if(head<G){
        int64_t rowS=(int64_t)(b*H+h0+head)*S+start;
        if(j0<count){float x=__fmul_rn(__fmul_rn(acc[e],k0[t]),scale);values[rowS+j0]=x;if(e)mx1=fmaxf(mx1,x);else mx0=fmaxf(mx0,x);}
        if(j1<count){float x=__fmul_rn(__fmul_rn(acc[2+e],k1[t]),scale);values[rowS+j1]=x;if(e)mx1=fmaxf(mx1,x);else mx0=fmaxf(mx0,x);}
      }
    }
  }
  #pragma unroll
  for(int o=4;o<32;o<<=1){mx0=fmaxf(mx0,__shfl_xor_sync(0xffffffffu,mx0,o));mx1=fmaxf(mx1,__shfl_xor_sync(0xffffffffu,mx1,o));}
  if(r==0){wmax[warp][2*tig]=mx0;wmax[warp][2*tig+1]=mx1;}
  __syncthreads();
  if(tid<G)cm[(int64_t)(b*H+h0+tid)*NC+c]=fmaxf(fmaxf(wmax[0][tid],wmax[1][tid]),fmaxf(wmax[2][tid],wmax[3][tid]));
  tl_end<TL>(tl,(int64_t)bh*NC+c,st);
}

// ---- 3. select + exact K + softmax + PV (+ fused merge) --------------------
// spq4::select_pv default instantiation verbatim in S1-S4 (SP row-carry rule,
// refined-union gather, bf16 MMA exact dots, bf16-rounded P with fp32 l, bf16
// MMA PV), with GT = H/KVH fixed at compile time. SG = shared-array rows:
// GT (resource-tuned) or MAXG (the baseline footprint, BIG=true).
// VPF: prefetch.global.L2 of this chunk's V rows at CTA start (PTX ISA),
// so V streams from DRAM while S1-S3 run; the PV loads/arithmetic are as
// before. FUSE: Flash-Decoding merge in the last CTA (see header).
template<int D,int VD,int GT,bool DIAG,int MINB,bool BIG,bool VPF,bool FUSE,bool TL,bool KPF=false>
__global__ void __launch_bounds__(128,MINB) select_pv(const bf16* q,const bf16* k,const bf16* v,
 const float* bulk,const float* cm,uint8_t* mask,float* prefix,float* pm,float* pl,float* pa,
 int H,int KVH,int S,int NC,int PK,float scale,float delta,
 unsigned* ctr,const bf16* sinks,bf16* out,unsigned long long* tl
#if TC_APA_P1_TEST_HOOKS
 ,EarlyCounterHook test_hook
#endif
 ){
  constexpr int G=GT,SG=BIG?MAXG:GT;
  constexpr int DPL=D/4,NS=DPL/4,NU4=DPL/8,NTW=VD/32,U=VD/8;
  __shared__ float scs[SG][C];
  __shared__ uint8_t use[SG][C],any_[C];
  __shared__ __align__(16) bf16 Ps[SG][PSTRIDE];
  __shared__ __align__(16) bf16 Vs[32][VD+8];
  __shared__ float lsum[SG];
  int bh=blockIdx.x,c=blockIdx.y+gridDim.y*blockIdx.z;if(c>=NC)return;
#if TC_APA_P1_TEST_HOOKS
  // Hold chunk 1 BEFORE any pm/pl/pa partial store (survey S3).
  if(!early_counter_wait(test_hook,c))return;
#endif
  Stamp st;tl_begin<TL>(st);
  int b=bh/KVH,h0=(bh%KVH)*G,tid=threadIdx.x,lane=tid&31,warp=tid>>5,r=lane>>2,tig=lane&3;
  int start=c*C,count=min(C,S-start);
  const uint4* vbase=reinterpret_cast<const uint4*>(v+((int64_t)bh*S+start)*VD);
  if constexpr(VPF){
    // L2 prefetch of the chunk's V rows; PV still loads every V byte.
    l2_prefetch(v+((int64_t)bh*S+start)*VD,(int64_t)count*VD*(int64_t)sizeof(bf16),tid,NT);
  }
  // ---- S1: SP selection, warp per head (max-prefix scan, no CTA barriers) ----
  for(int g=warp;g<G;g+=4){
    int64_t row=(int64_t)b*H+h0+g,base=row*S+start;
    float cv=-INFINITY;int first=(start/PK)*PK/C;
    #pragma unroll 4
    for(int p=first+lane;p<c;p+=32)cv=fmaxf(cv,cm[row*NC+p]);
    float carry=spq4::warp_max(cv);
    #pragma unroll
    for(int t=0;t<8;++t){
      int j=t*32+lane;float x=j<count?bulk[base+j]:-INFINITY,p=x;
      #pragma unroll
      for(int o=1;o<32;o<<=1){float y=__shfl_up_sync(0xffffffffu,p,o);if(lane>=o)p=fmaxf(p,y);}
      p=fmaxf(p,carry);
      bool refine=j<count && x>=p-delta;
      carry=__shfl_sync(0xffffffffu,p,31);
      scs[g][j]=x;use[g][j]=refine;
      if(j<count){mask[base+j]=refine;if constexpr(DIAG)prefix[base+j]=p;}
    }
  }
  __syncthreads();
  for(int j=tid;j<C;j+=NT){
    unsigned a=0;
    for(int g=0;g<G;++g)a|=use[g][j];
    any_[j]=a;
    // KPF: L2 prefetch of the refined-union K row (D bf16) as soon as the
    // union is known; S2 then loads it (later t-groups hit L2). No arithmetic.
    if constexpr(KPF){if(a && j<count)l2_prefetch(k+((int64_t)bh*S+start+j)*D,(int64_t)D*(int64_t)sizeof(bf16),0,1);}
  }
  __syncthreads();
  // ---- S2: exact dots, each refined K row of the KV head read once for all G heads ----
  {
    uint2 qa[NS];
    #pragma unroll
    for(int s=0;s<NS;++s)qa[s]=make_uint2(0u,0u);
    if(r<G){
      const uint2* qp=reinterpret_cast<const uint2*>(q+((int64_t)b*H+h0+r)*D+tig*DPL);
      #pragma unroll
      for(int s=0;s<NS;++s)qa[s]=qp[s];
    }
    #pragma unroll 2
    for(int t=0;t<8;++t){
      int gi=warp+4*t,key=gi*8+r;bool used=key<count && any_[key];
      if(!__any_sync(0xffffffffu,used))continue;
      uint4 kw[NU4];
      #pragma unroll
      for(int i=0;i<NU4;++i)kw[i]=make_uint4(0u,0u,0u,0u);
      if(used){
        const uint4* kp=reinterpret_cast<const uint4*>(k+((int64_t)bh*S+start+key)*D+tig*DPL);
        #pragma unroll
        for(int i=0;i<NU4;++i)kw[i]=kp[i];
      }
      float acc[4]={0.f,0.f,0.f,0.f};
      #pragma unroll
      for(int s=0;s<NS;++s){
        uint4 u=kw[s>>1];unsigned b0=(s&1)?u.z:u.x,b1=(s&1)?u.w:u.y;
        spq4::mma_bf16(acc,qa[s].x,qa[s].y,b0,b1);
      }
      #pragma unroll
      for(int i=0;i<2;++i){
        int kk=gi*8+2*tig+i;
        if(r<G && kk<count && use[r][kk])scs[r][kk]=__fmul_rn(acc[i],scale);
      }
    }
  }
  __syncthreads();
  // ---- S3: per-head chunk softmax; P rounded to bf16, l = fp32 sum of the rounded P ----
  for(int g=warp;g<G;g+=4){
    int64_t row=(int64_t)b*H+h0+g;
    float m=-INFINITY;
    for(int j=lane;j<count;j+=32)m=fmaxf(m,scs[g][j]);
    m=spq4::warp_max(m);float sum=0.f;
    for(int j=lane;j<C;j+=32){
      float p=j<count?__expf(scs[g][j]-m):0.f;bf16 pb=__float2bfloat16_rn(p);
      Ps[g][j]=pb;sum+=__bfloat162float(pb);
    }
    #pragma unroll
    for(int o=16;o;o>>=1)sum+=__shfl_xor_sync(0xffffffffu,sum,o);
    if(lane==0){lsum[g]=sum;pm[row*NC+c]=m;}
  }
  __syncthreads();
  if(tid<G)pl[((int64_t)b*H+h0+tid)*NC+c]=lsum[tid];
  // ---- S4: PV on tensor cores, M = G heads (padded to 16), N = VD, K = keys ----
  float acc[NTW][4];
  #pragma unroll
  for(int i=0;i<NTW;++i){acc[i][0]=acc[i][1]=acc[i][2]=acc[i][3]=0.f;}
  uint4 nxt[U/4];
  auto fetch=[&](int t0){
    #pragma unroll
    for(int i=0;i<U/4;++i){
      int idx=tid+NT*i,key=t0+idx/U;
      nxt[i]=key<count?vbase[(int64_t)key*U+idx%U]:make_uint4(0u,0u,0u,0u);
    }
  };
  fetch(0);
  for(int t0=0;t0<count;t0+=32){
    #pragma unroll
    for(int i=0;i<U/4;++i){int idx=tid+NT*i;*reinterpret_cast<uint4*>(&Vs[idx/U][(idx%U)*8])=nxt[i];}
    __syncthreads();
    if(t0+32<count)fetch(t0+32);
    #pragma unroll
    for(int ks=0;ks<2;++ks){
      int key0=t0+ks*16;if(key0>=count)break;
      unsigned a0=0u,a2=0u;
      if(r<G){a0=*reinterpret_cast<const unsigned*>(&Ps[r][key0+2*tig]);a2=*reinterpret_cast<const unsigned*>(&Ps[r][key0+8+2*tig]);}
      int rb=ks*16+2*tig;
      #pragma unroll
      for(int nt=0;nt<NTW;++nt){
        int n=warp*(VD/4)+nt*8+r;
        unsigned b0=(unsigned)__bfloat16_as_ushort(Vs[rb][n])|((unsigned)__bfloat16_as_ushort(Vs[rb+1][n])<<16);
        unsigned b1=(unsigned)__bfloat16_as_ushort(Vs[rb+8][n])|((unsigned)__bfloat16_as_ushort(Vs[rb+9][n])<<16);
        spq4::mma_bf16(acc[nt],a0,a2,b0,b1);
      }
    }
    __syncthreads();
  }
  if(r<G){int head=r;
    int64_t pr=((int64_t)b*H+h0+head)*NC+c;
    #pragma unroll
    for(int nt=0;nt<NTW;++nt){
      int d=warp*(VD/4)+nt*8+2*tig;
      *reinterpret_cast<float2*>(&pa[pr*VD+d])=make_float2(acc[nt][0],acc[nt][1]);
    }
  }
  if constexpr(!FUSE){tl_end<TL>(tl,(int64_t)bh*NC+c,st);return;}
  else{
  // ---- fused Flash-Decoding merge: last CTA of the (batch, KV head) group ----
  // Threadfence-reduction last-block pattern (NVIDIA CUDA Samples). Writers:
  // every thread fences its pm/pl/pa stores, then one ticket per CTA. The
  // ticket wraps to 0 on the last arrival (atomicInc limit NC-1), so the
  // counters need no reset launch. Reader: L2-only loads (__ldcg) after a
  // fence. Arithmetic = spo2::merge_kernel<bf16> (Acc=float) verbatim:
  // mx = fmaxf chain over c ascending (then sink); sum = fma(l_c, e_c, sum)
  // over c ascending, then + e_sink; acc_d = fma(pa_cd, e_c, acc_d) over c
  // ascending; out = den!=0 ? acc/den (IEEE div) : 0 (APA-P1-F1 R1), RN to bf16. e_c is
  // __expf(m_c - mx), computed once per (head, chunk) and reused (the same
  // instruction on the same operands as the per-thread recomputation).
  // Latency layout (ours): m tile loaded in parallel into shared memory,
  // the ordered chains then read shared memory; pa read as float4 rows in
  // batches of MERGE_BATCH independent loads ahead of each ordered chain.
  __shared__ unsigned last_s;
  __shared__ float mxs[GT],dens[GT];
  __threadfence();
  __syncthreads();
  if(tid==0){
#if APA_EB3_MUTANT==6
    unsigned prev=atomicAdd(ctr+bh,1u);                    // Mutant 6: stale counter, never reset
#else
    unsigned prev=atomicInc(ctr+bh,(unsigned)(NC-1));      // wraps to 0 on the last arrival
#endif
#if APA_EB3_MUTANT==3
    last_s=prev==(unsigned)(NC-2);                         // Mutant 3: fires one CTA early
#else
    last_s=prev==(unsigned)(NC-1);
#endif
  }
  __syncthreads();
  if(!last_s){
#if TC_APA_P1_TEST_HOOKS
    early_counter_done(test_hook,c);
#endif
    tl_end<TL>(tl,(int64_t)bh*NC+c,st);return;
  }
  __threadfence();
  const int64_t row0=(int64_t)b*H+h0;
  constexpr int Q4=VD/4,QPT=(G*Q4+NT-1)/NT,MB=tune::MERGE_BATCH/QPT;
  float sum=0.f;
  float4 macc[QPT];
  #pragma unroll
  for(int qi=0;qi<QPT;++qi)macc[qi]=make_float4(0.f,0.f,0.f,0.f);
  if(NC<=C){
    // Single tile (S <= C*C): every operand fetched in one L2 round trip.
    // pa batch 0 is in flight while m/l are staged and the max chain runs.
    const int n=NC;
    float* lsh=reinterpret_cast<float*>(&Vs[0][0]);    // V tiles are dead here
    static_assert(sizeof(Vs)>=sizeof(float)*(size_t)G*C,"l staging fits in Vs");
    float4 buf[QPT][MB];
    #pragma unroll
    for(int qi=0;qi<QPT;++qi){
      int qd=tid+qi*NT,g=qd/Q4,d0=(qd%Q4)*4;
      const float4* a=reinterpret_cast<const float4*>(pa+(row0+(qd<G*Q4?g:0))*NC*VD+d0);
      #pragma unroll
      for(int u=0;u<MB;++u){
#if APA_EB3_MUTANT==4
        int cc=n-1-u;
#else
        int cc=u;
#endif
        buf[qi][u]=(qd<G*Q4 && u<n)?__ldcg(a+(int64_t)cc*Q4):make_float4(0.f,0.f,0.f,0.f);
      }
    }
    for(int i=tid;i<G*n;i+=NT){int g=i/n,cc=i%n;scs[g][cc]=__ldcg(pm+(row0+g)*NC+cc);lsh[g*C+cc]=__ldcg(pl+(row0+g)*NC+cc);}
    __syncthreads();
    if(tid<G){
      float x=-INFINITY;
      for(int cc=0;cc<n;++cc)x=fmaxf(x,scs[tid][cc]);
#if APA_EB3_MUTANT!=5
      if(sinks)x=fmaxf(x,__bfloat162float(sinks[h0+tid]));
#endif
      mxs[tid]=x;
    }
    __syncthreads();
    for(int i=tid;i<G*n;i+=NT){int g=i/n,cc=i%n;scs[g][cc]=__expf(scs[g][cc]-mxs[g]);}
    __syncthreads();
    if(tid<G){
      for(int i=0;i<n;++i){
#if APA_EB3_MUTANT==4
        int cc=n-1-i;                                      // Mutant 4: chunk order reversed
#else
        int cc=i;
#endif
        sum=__fmaf_rn(lsh[tid*C+cc],scs[tid][cc],sum);
      }
#if APA_EB3_MUTANT!=5
      if(sinks)sum=__fadd_rn(sum,__expf(__bfloat162float(sinks[h0+tid])-mxs[tid]));
#endif
      dens[tid]=sum;
    }
    #pragma unroll
    for(int qi=0;qi<QPT;++qi){
      int qd=tid+qi*NT;
      if(qd<G*Q4){
        int g=qd/Q4,d0=(qd%Q4)*4;
        const float4* a=reinterpret_cast<const float4*>(pa+(row0+g)*NC*VD+d0);
        #pragma unroll
        for(int u=0;u<MB;++u){
#if APA_EB3_MUTANT==4
          int cc=n-1-u;
#else
          int cc=u;
#endif
          if(u<n){float e=scs[g][cc];
            macc[qi].x=__fmaf_rn(buf[qi][u].x,e,macc[qi].x);macc[qi].y=__fmaf_rn(buf[qi][u].y,e,macc[qi].y);
            macc[qi].z=__fmaf_rn(buf[qi][u].z,e,macc[qi].z);macc[qi].w=__fmaf_rn(buf[qi][u].w,e,macc[qi].w);}
        }
        for(int i0=MB;i0<n;i0+=MB){
          float4 bb[MB];
          #pragma unroll
          for(int u=0;u<MB;++u){
#if APA_EB3_MUTANT==4
            int cc=n-1-(i0+u);
#else
            int cc=i0+u;
#endif
            bb[u]=(i0+u<n)?__ldcg(a+(int64_t)cc*Q4):make_float4(0.f,0.f,0.f,0.f);
          }
          #pragma unroll
          for(int u=0;u<MB;++u){
#if APA_EB3_MUTANT==4
            int cc=n-1-(i0+u);
#else
            int cc=i0+u;
#endif
            if(i0+u<n){float e=scs[g][cc];
              macc[qi].x=__fmaf_rn(bb[u].x,e,macc[qi].x);macc[qi].y=__fmaf_rn(bb[u].y,e,macc[qi].y);
              macc[qi].z=__fmaf_rn(bb[u].z,e,macc[qi].z);macc[qi].w=__fmaf_rn(bb[u].w,e,macc[qi].w);}
          }
        }
      }
    }
    __syncthreads();
  }else{
  // General path (NC > C): tiles of C chunks, two passes over m.
  float mxr=-INFINITY;
  for(int c0=0;c0<NC;c0+=C){
    int n=min(C,NC-c0);
    for(int i=tid;i<G*n;i+=NT){int g=i/n,cc=i%n;scs[g][cc]=__ldcg(pm+(row0+g)*NC+c0+cc);}
    __syncthreads();
    if(tid<G){for(int cc=0;cc<n;++cc)mxr=fmaxf(mxr,scs[tid][cc]);}
    __syncthreads();
  }
  if(tid<G){
#if APA_EB3_MUTANT!=5
    if(sinks)mxr=fmaxf(mxr,__bfloat162float(sinks[h0+tid]));
#endif
    mxs[tid]=mxr;
  }
  __syncthreads();
  for(int c0=0;c0<NC;c0+=C){
    int n=min(C,NC-c0);
    for(int i=tid;i<G*n;i+=NT){int g=i/n,cc=i%n;scs[g][cc]=__expf(__ldcg(pm+(row0+g)*NC+c0+cc)-mxs[g]);}
    __syncthreads();
    if(tid<G){
      const float* l=pl+(row0+tid)*NC+c0;
      #pragma unroll 16
      for(int i=0;i<n;++i){
#if APA_EB3_MUTANT==4
        int cc=n-1-i;
#else
        int cc=i;
#endif
        sum=__fmaf_rn(__ldcg(l+cc),scs[tid][cc],sum);
      }
    }
    #pragma unroll
    for(int qi=0;qi<QPT;++qi){
      int qd=tid+qi*NT;
      if(qd<G*Q4){
        int g=qd/Q4,d0=(qd%Q4)*4;
        const float4* a=reinterpret_cast<const float4*>(pa+((row0+g)*NC+c0)*VD+d0);
        for(int i0=0;i0<n;i0+=MB){
          float4 bb[MB];
          #pragma unroll
          for(int u=0;u<MB;++u){
#if APA_EB3_MUTANT==4
            int cc=n-1-(i0+u);
#else
            int cc=i0+u;
#endif
            bb[u]=(i0+u<n)?__ldcg(a+(int64_t)cc*Q4):make_float4(0.f,0.f,0.f,0.f);
          }
          #pragma unroll
          for(int u=0;u<MB;++u){
#if APA_EB3_MUTANT==4
            int cc=n-1-(i0+u);
#else
            int cc=i0+u;
#endif
            if(i0+u<n){float e=scs[g][cc];
              macc[qi].x=__fmaf_rn(bb[u].x,e,macc[qi].x);macc[qi].y=__fmaf_rn(bb[u].y,e,macc[qi].y);
              macc[qi].z=__fmaf_rn(bb[u].z,e,macc[qi].z);macc[qi].w=__fmaf_rn(bb[u].w,e,macc[qi].w);}
          }
        }
      }
    }
    __syncthreads();
  }
  if(tid<G){
#if APA_EB3_MUTANT!=5
    if(sinks)sum=__fadd_rn(sum,__expf(__bfloat162float(sinks[h0+tid])-mxs[tid]));
#endif
    dens[tid]=sum;
  }
  __syncthreads();
  }
  #pragma unroll
  for(int qi=0;qi<QPT;++qi){
    int qd=tid+qi*NT;
    if(qd<G*Q4){
      int g=qd/Q4,d0=(qd%Q4)*4;float den=dens[g];bf16* o=out+(row0+g)*VD+d0;
      // APA-P1-F1 R1 (P1R-F01): guard only an exactly-zero denominator; a NaN
      // den now propagates (IEEE 754) instead of writing 0. APA-P1-F2 R5
      // (P1R2-F06), the claim stated narrowly: den is an fma chain of l_c*e_c
      // (+ e_sink). When every intermediate score and den are finite, den >= 0
      // and den>0 and den!=0 select the same branch, so the guard change
      // preserves every such result. It does not preserve results from finite
      // operands that overflow (e.g. a FLT_MAX scale: score +Inf, exp(Inf-Inf)
      // NaN), from a NaN or +Inf sink, or from a +-Inf query over all-negative
      // logits: those follow IEEE propagation to NaN where the old guard wrote
      // 0 (the loud direction; LF2 of APA-P1-F2 pins them). Milakov and
      // Gimelshein 2018 / FlashAttention (Dao et al. 2022) merge, unchanged.
      o[0]=__float2bfloat16_rn(den!=0.f?__fdiv_rn(macc[qi].x,den):0.f);
      o[1]=__float2bfloat16_rn(den!=0.f?__fdiv_rn(macc[qi].y,den):0.f);
      o[2]=__float2bfloat16_rn(den!=0.f?__fdiv_rn(macc[qi].z,den):0.f);
      o[3]=__float2bfloat16_rn(den!=0.f?__fdiv_rn(macc[qi].w,den):0.f);
    }
  }
#if TC_APA_P1_TEST_HOOKS
  early_counter_done(test_hook,c);
#endif
  tl_end<TL>(tl,(int64_t)bh*NC+c,st);
  }
}

// ---- persistent per-device tickets (APA-P1-S3 r2b) -------------------------
// Contract: ONE stream, the CUDA legacy default stream (0), on each device.
// A launch on any other stream is out of contract. atomicInc/fence arithmetic
// is inherited unchanged from CUDA Samples threadFenceReduction (NVIDIA).
// Prior art: CUDA stream-ordered lifetime APIs (cudaMallocAsync, NVIDIA 2021)
// and std::mutex/map ownership (ISO C++11, 2011). Taken: stream ordering and
// last-writer tickets; ours: device-keyed merge/choose slots and a launch lock.
// Literature unverified -- lead to check: CUDA Samples threadFenceReduction;
// CUDA 11.2 stream ordered memory allocator 2021. No new reduction.
// APA_P1_HOST_TICKETS_BEGIN (also compiled by the CPU ownership test)
struct TicketSlot { unsigned* buffer=nullptr; int capacity=0; };
struct DeviceTickets { TicketSlot merge,choose; int64_t l2=-1; };
inline std::recursive_mutex& ticket_mutex(){static std::recursive_mutex m;return m;}
inline std::unordered_map<int,DeviceTickets>& device_tickets(){
  // Process lifetime, as before. Do not call CUDA from static destructors.
  static std::unordered_map<int,DeviceTickets> states;return states;
}
// Reuse the engine guard: checked success-path restore, counted unwind errors.
using TicketDeviceGuard=CheckedDeviceGuard;
inline unsigned* ticket_buffer(bool choose,int n,int minimum,int64_t* bytes){
  std::lock_guard<std::recursive_mutex> lock(ticket_mutex());
  int device=0;
  if(cudaGetDevice(&device)!=cudaSuccess)throw std::runtime_error("apa tickets: get device");
  auto& state=device_tickets()[device];auto& slot=choose?state.choose:state.merge;
  if(n>slot.capacity){
    const int want=std::max(n,minimum);unsigned* next=nullptr;
    if(cudaMalloc(&next,sizeof(unsigned)*(size_t)want)!=cudaSuccess)throw std::runtime_error("apa tickets: allocation");
    if(cudaMemset(next,0,sizeof(unsigned)*(size_t)want)!=cudaSuccess){
      cudaFree(next);throw std::runtime_error("apa tickets: zero");
    }
    if(slot.buffer){
      // Only the CURRENT (guarded operand) device is synchronized. The launch
      // lock remains held from lookup through enqueue, so no caller can hold
      // an old pointer that has not yet been submitted when growth frees it.
      if(cudaDeviceSynchronize()!=cudaSuccess){cudaFree(next);throw std::runtime_error("apa tickets: growth sync");}
      if(cudaFree(slot.buffer)!=cudaSuccess){cudaFree(next);throw std::runtime_error("apa tickets: growth free");}
    }
    slot.buffer=next;slot.capacity=want;
  }
  if(bytes)*bytes=(int64_t)slot.capacity*(int64_t)sizeof(unsigned);
  return slot.buffer;
}
inline unsigned* counters(int n,int64_t* bytes=nullptr){
  return ticket_buffer(false,n,tune::COUNTER_MIN_ENTRIES,bytes);
}
inline int64_t l2_bytes(){
  std::lock_guard<std::recursive_mutex> lock(ticket_mutex());
  int device=0;
  if(cudaGetDevice(&device)!=cudaSuccess)throw std::runtime_error("apa tickets: L2 device");
  auto& cached=device_tickets()[device].l2;
  if(cached<0){
    int value=0;
    if(cudaDeviceGetAttribute(&value,cudaDevAttrL2CacheSize,device)!=cudaSuccess)throw std::runtime_error("apa tickets: L2 size");
    cached=value;
  }
  return cached;
}
// APA_P1_HOST_TICKETS_END
#if TC_APA_P1_TEST_HOOKS
// APA-P1-F1 hooks-build test accessor (LF5): copies the first n merge-ticket
// slots of the current device. No prior art claimed; a device-to-host copy.
inline std::vector<unsigned> early_counter_merge_slots(int n){
  std::lock_guard<std::recursive_mutex> lock(ticket_mutex());
  std::vector<unsigned> out((size_t)(n>0?n:0),0u);
  if(n<=0)return out;
  unsigned* p=ticket_buffer(false,n,tune::COUNTER_MIN_ENTRIES,nullptr);
  if(cudaDeviceSynchronize()!=cudaSuccess || cudaMemcpy(out.data(),p,sizeof(unsigned)*(size_t)n,cudaMemcpyDeviceToHost)!=cudaSuccess)
    throw std::runtime_error("APA P1 test: merge ticket read");
  return out;
}
#endif

// ---- variant table (runtime `variant`; 0 = registered path) -----------------
// rot: 0 spq1::rotate (baseline kernel), 1 eb3::rotate_q<128,ROT_DB>;
//      rpf: rotation prefetches toward L2: 1 the bulk's codes+norms,
//      2 codes+norms then V (budget L2_FRACTION of L2).
// bulk: 0 spq6::bulk_mma_tq<128,2> (baseline kernel), 1 bulk_copy MINB1
//       (baseline footprint), 2 bulk_swap BULK_MINB, 3 bulk_swap
//       BULK_MINB_ALT; xpf: bulk prefetches the chunk's V (budget).
// sel: 0 spq4::select_pv (baseline kernel), 1 eb3 tuned (G smem, SEL_MINB),
//      2 eb3 baseline footprint (MAXG smem, MINB1).
// vpf/fuse/kpf: select options (sel 1 only); fuse=false uses
//      spo2::merge_kernel; kpf = L2 prefetch of refined-union K rows.
struct Variant{int rot;int rpf;int bulk;bool xpf;int sel;bool vpf,fuse,kpf;const char* name;};
constexpr Variant VARIANTS[]={
  // 0: the registered (gated, timed) EB3 path; every other row is a descriptive ablation.
  {tune::REG_ROT,tune::REG_RPF,tune::REG_BULK,tune::REG_XPF,1,tune::REG_VPF,tune::REG_FUSE,tune::REG_KPF,"registered"},
  {0,0,0,false,1,false,false,false,"A1 occupancy-only select (baseline rotate/bulk/merge)"},
  {1,0,0,false,0,false,false,false,"A2 rotation-only (baseline bulk/select/merge)"},
  {1,0,0,false,1,true ,false,false,"A3 rotate_q + baseline bulk + tuned select (V prefetch), separate merge"},
  {1,0,2,false,1,true ,true ,false,"A4 rotate_q + swap-AB bulk + tuned select (V prefetch, fused merge), no cross-kernel prefetch"},
  {1,1,2,true ,1,true ,true ,true ,"A5 registered + refined-K L2 prefetch"},
  {0,0,1,false,2,false,false,false,"A6 baseline-footprint clones (timeline 'before')"},
  {1,2,2,false,1,true ,true ,false,"A7 rotation prefetches codes+norms+V, swap-AB bulk without V prefetch"},
  {1,1,2,true ,1,true ,false,false,"A8 registered without fused merge (separate merge kernel)"},
};
constexpr int NVARIANTS=(int)(sizeof(VARIANTS)/sizeof(VARIANTS[0]));
} // namespace eb3

// ---- launcher ----------------------------------------------------------------
// Validation and arena as apa_spq6_tq4_tc (SPQ6/FXTC1 2026), variant B only.
// Returns {out, rotated q, mask, bulk, prefix, bulk timeline, select timeline,
// footprint}; entries past `out` are defined only with diag/timeline.
std::vector<NDArray> apa_eb3_run(const NDArray& q,const NDArray& k,const NDArray& v,const NDArray& codes,const NDArray& sc,
  const NDArray& rot,const NDArray& cb,float scale,float delta,const NDArray* sinks,const std::string& carry_mode,
  bool diag,bool poison,bool timeline,int variant){
  // APA-P1-F3 R3' (P1R3-F03): q/k/v extents above INT_MAX raise ValueError naming this entry and the operand, first.
  apa_spo_require_attention_extents("apa_eb3",{{"q",&q},{"k",&k},{"v",&v}});
  if(carry_mode!="partition" && carry_mode!="row")throw std::runtime_error("carry must be partition or row");
  const char* flag=std::getenv("TC_APA_SP");
  if(!flag || std::strcmp(flag,"1")!=0)throw std::runtime_error("apa_sp: experimental entry requires TC_APA_SP=1 (default OFF)");
  if(const char* ve=std::getenv("APA_FXTC1_VARIANT"))
    if(std::strcmp(ve,"2")!=0)throw std::runtime_error("apa_eb3: variant B only (APA_FXTC1_VARIANT must be unset or 2)");
  if(!std::isfinite(scale) || scale<=0.f || !std::isfinite(delta) || delta<0.f)throw std::runtime_error("apa_sp: scale must be finite positive; delta finite nonnegative");
  if(q.ndim()!=4 || k.ndim()!=4 || v.ndim()!=4)throw std::runtime_error("apa_eb3: q/k/v must be rank four");
  if(q.dtype!=DType::BFloat16 || k.dtype!=DType::BFloat16 || v.dtype!=DType::BFloat16)throw std::runtime_error("apa_eb3: bfloat16 q/k/v required");
  for(const NDArray* a:{&q,&k,&v,&codes,&sc,&rot,&cb})if(!a->device.is_cuda() || a->device.index!=q.device.index)throw std::runtime_error("apa_eb3: inputs must share one CUDA device");
  apa_spo_require_extents("apa_eb3",{&q,&k,&v});   // R11: empty extents raise as the dense entry
#if !TC_APA_P1_TEST_HOOKS
  (void)apa_spo_knob("APA_P1_EARLY_COUNTER","0");   // R7: the hook is compiled out of the default build
#endif
  const int64_t B=q.shape[0],H=q.shape[1],L=q.shape[2],D=q.shape[3],KVH=k.shape[1],S=k.shape[2],VD=v.shape[3];
  if(L!=1)throw std::runtime_error("apa_eb3: requires L=1");
  if(KVH<=0 || H%KVH!=0 || k.shape[0]!=B || k.shape[3]!=D || v.shape[0]!=B || v.shape[1]!=KVH || v.shape[2]!=S)throw std::runtime_error("apa_eb3: shape/GQA mismatch");
  if(D!=128 || VD!=128)throw std::runtime_error("apa_eb3: D=VD=128 only");
  int G=(int)(H/KVH);
  if(G<2 || G>spq4::MAXG)throw std::runtime_error("apa_eb3: needs 2 <= H/KVH <= 8");
  if(S<1 || B*H>std::numeric_limits<int>::max()/4)throw std::runtime_error("apa_eb3: dimension range");
  apa_spo_require_key_range("apa_eb3",S);   // APA-P1-F2 R3 (P1R2-F04)
  if(rot.shape!=Shape({KVH,D,D}) || rot.dtype!=DType::Float32)throw std::runtime_error("apa_eb3: rotation shape/dtype");
  if(cb.shape!=Shape({16}) || cb.dtype!=DType::Float32)throw std::runtime_error("apa_eb3: codebook shape/dtype");
  if(sinks && (sinks->ndim()!=1 || sinks->shape[0]!=H || sinks->dtype!=q.dtype || !sinks->device.is_cuda() || sinks->device.index!=q.device.index))throw std::runtime_error("apa_sp: sinks must be [H] with matching dtype/device");
  apa_spq1_buffer_check(k,codes,sc);
  if(variant<0 || variant>=eb3::NVARIANTS)throw std::runtime_error("apa_eb3: unknown variant");
  const eb3::Variant VA=eb3::VARIANTS[variant];
  if(variant!=eb3::tune::REGISTERED_VARIANT && G!=2 && G!=4)throw std::runtime_error("apa_eb3: ablation variants are built for G=2,4 only");
  if(diag && variant!=eb3::tune::REGISTERED_VARIANT)throw std::runtime_error("apa_eb3: diagnostics only on the registered variant");
  if(timeline && variant!=eb3::tune::REGISTERED_VARIANT && variant!=6 && variant!=1)throw std::runtime_error("apa_eb3: timeline only on variants 0, 1, 6");
  // APA-P1-F2 R4 (P1R2-F05): a timeline select (ON=true) is never REG in byG
  // below, so its templates exist for G=2 and G=4 only, for every variant that
  // allows a timeline (0: registered select with ON; 1: occupancy-only; 6:
  // baseline footprint). Other G used to pass every check here, launch the
  // rotation and the bulk, and only then refuse inside byG. Refused now
  // before the lock, the arena and any launch. Plain validation; no prior art.
  if(timeline && G!=2 && G!=4)
    throw std::invalid_argument("apa_eb3: timeline=True has select templates for G=H/KVH in {2,4} only (got G="+
      std::to_string(G)+"); refused before any allocation or launch");
  // R2 (P1R-F02) census for this launcher (every variant): q uint2 (eb3.cuh
  // select_pv S2, spq4.cuh:220 for sel 0), k uint4 (S2), v uint4 (vbase), codes
  // uint4 (every bulk kernel, D=128); rot float4 only in rotate_q (VA.rot==1).
  apa_spo_require_aligned("apa_eb3","q",q,8);
  apa_spo_require_aligned("apa_eb3","k",k,16);
  apa_spo_require_aligned("apa_eb3","v",v,16);
  apa_spo_require_aligned("apa_eb3","codes",codes,16);
  if(VA.rot==1)apa_spo_require_aligned("apa_eb3","rotation",rot,16);
  // Prior art: CUDA legacy-default-stream ordering and RAII (C++11, 2011).
  // Serialize host submission through growth; no steady-state device sync.
  std::lock_guard<std::recursive_mutex> ticket_launch_lock(eb3::ticket_mutex());
  eb3::TicketDeviceGuard operand_device(q.device.index,"apa_eb3");
  int np,pk;apa_int4_fixed_partition_layout((int)S,np,pk);
  if(pk%spq4::C)throw std::runtime_error("apa_eb3: partition must align to chunk");
  int nc=(int)((S+spq4::C-1)/spq4::C),rows=(int)(B*H);
  if(carry_mode=="row")pk=nc*spq4::C;
  dim3 grid((unsigned)(B*KVH),std::min(nc,65535),(nc+65534)/65535);
  int64_t RS=(int64_t)rows*S,RC=(int64_t)rows*nc,TLN=timeline?(int64_t)B*KVH*nc*5*8:0;
  // APA-DS1 amendment A2 (2026-11-03): align every arena view for 16B consumers.
  // Prior art: aligned bump allocation (ISO C++17, 2017) and NVIDIA CUDA
  // Programming Guide natural vector alignment (2024 version unverified;
  // lead to check: CUDA float4 natural alignment). Taken: cursor round-up;
  // ours: apply it to this arena. Device code and arithmetic are unchanged.
  constexpr int64_t kArenaAlignment = 16;
  std::vector<int64_t> sizes={4*(int64_t)rows*D,RS,4*RS,diag?4*RS:0,4*RC,4*RC,4*RC,4*RC*VD,TLN,TLN};
  int64_t total=0;for(auto n:sizes){if(n==0)continue;total=(total+kArenaAlignment-1)&~(kArenaAlignment-1);total+=n;} // A3: zero-byte slots; aligned bump allocation (ISO C++17, 2017).
  NDArray arena({total},DType::Uint8,q.device),out({B,H,1,VD},q.dtype,q.device);
  if(poison){
    if(cudaMemsetAsync(arena.data_ptr(),0xff,(size_t)total,0)!=cudaSuccess || cudaMemsetAsync(out.data_ptr(),0xff,(size_t)(B*H*VD*2),0)!=cudaSuccess)
      throw std::runtime_error("apa_eb3: poison memset");
  }
  int64_t cursor=0;
  auto view=[&](Shape shape,DType dtype)->NDArray{
    cursor=(cursor+kArenaAlignment-1)&~(kArenaAlignment-1);NDArray x=arena;x.shape=shape;x.dtype=dtype;
    x.offset=cursor/dtype_size(dtype);cursor+=x.numel()*dtype_size(dtype);return x;};
  NDArray pq=view({B,H,1,D},DType::Float32);
  NDArray mask=view({B,H,1,S},DType::Uint8),bulk=view({B,H,1,S},DType::Float32),prefix;
  if(diag)prefix=view({B,H,1,S},DType::Float32);
  NDArray cm=view({RC},DType::Float32),pm=view({RC},DType::Float32),pl=view({RC},DType::Float32),pa=view({RC*VD},DType::Float32);
  NDArray tlb,tls;
  if(timeline){tlb=view({B*KVH,nc,5},DType::Int64);tls=view({B*KVH,nc,5},DType::Int64);}
  if(cursor!=total)throw std::runtime_error("apa_eb3: workspace layout mismatch");
  auto Q=(spq4::bf16*)q.data_ptr();auto K=(spq4::bf16*)k.data_ptr();auto V=(spq4::bf16*)v.data_ptr();
  auto PQ=(float*)pq.data_ptr();auto BS=(float*)bulk.data_ptr();auto CM=(float*)cm.data_ptr();auto M=(uint8_t*)mask.data_ptr();
  auto PX=prefix.defined()?(float*)prefix.data_ptr():nullptr;
  auto PM=(float*)pm.data_ptr();auto PL=(float*)pl.data_ptr();auto PA=(float*)pa.data_ptr();
  auto TLB=timeline?(unsigned long long*)tlb.data_ptr():nullptr;auto TLS=timeline?(unsigned long long*)tls.data_ptr():nullptr;
  auto SNK=sinks?(spq4::bf16*)sinks->data_ptr():nullptr;auto OUT=(spq4::bf16*)out.data_ptr();
  int capacity=(int)codes.shape[2];
  const uint8_t* CODES=(const uint8_t*)codes.data_ptr();const float* SC=(const float*)sc.data_ptr();const float* CB=(const float*)cb.data_ptr();
#if TC_APA_P1_TEST_HOOKS
  NDArray test_state;
  auto test_hook=eb3::early_counter_setup(test_state,q.device,(int)(B*KVH),nc,
    variant==0 && VA.sel==1 && VA.fuse,PM,PL,PA,rows,(int)VD);
#endif
  // L2-capacity budget for cross-kernel prefetch (tuning: L2_FRACTION).
  const int64_t budget=(int64_t)(eb3::l2_bytes()*eb3::tune::L2_FRACTION);
  const int64_t code_bytes=(int64_t)codes.numel(),norm_bytes=(int64_t)sc.numel()*4,v_bytes=(int64_t)B*KVH*S*VD*2;
  const bool do_rpf=VA.rpf>=1 && code_bytes+norm_bytes<=budget;
  const bool do_rpfv=VA.rpf>=2 && code_bytes+norm_bytes+v_bytes<=budget;
  const bool do_xpf=VA.xpf && code_bytes+norm_bytes+v_bytes<=budget;
  // 1. rotation
  eb3::Prefetch pf;
  if(do_rpf){pf.a=CODES;pf.abytes=code_bytes;pf.b=SC;pf.bbytes=norm_bytes;}
  if(do_rpfv){pf.c=V;pf.cbytes=v_bytes;}
  if(VA.rot==1){
    eb3::rotate_q<128,eb3::tune::ROT_DB><<<dim3((unsigned)(B*KVH),128/eb3::tune::ROT_DB),128>>>(Q,(const float*)rot.data_ptr(),PQ,(int)H,(int)KVH,pf);
  }else{
    spq1::rotate<spq4::bf16><<<rows,128>>>(Q,(float*)rot.data_ptr(),PQ,(int)H,(int)KVH,128,true);
  }
  cuda_check_last("apa_eb3_rotate");
  // 2. bulk
  {
    const int Hi=(int)H,KVi=(int)KVH,Si=(int)S;
    const spq4::bf16* VP=do_xpf?V:nullptr;
    switch(VA.bulk){
      case 0: spq6::bulk_mma_tq<128,2><<<grid,128>>>(PQ,CODES,SC,CB,BS,CM,Hi,KVi,Si,nc,capacity,scale);break;
      case 1:
        if(timeline)eb3::bulk_copy<128,1,true><<<grid,128>>>(PQ,CODES,SC,CB,BS,CM,Hi,KVi,Si,nc,capacity,scale,TLB);
        else eb3::bulk_copy<128,1,false><<<grid,128>>>(PQ,CODES,SC,CB,BS,CM,Hi,KVi,Si,nc,capacity,scale,nullptr);
        break;
      case 2:
        if(timeline){
          if(VA.xpf)eb3::bulk_swap<128,eb3::tune::BULK_MINB,true,true><<<grid,128>>>(PQ,CODES,SC,CB,BS,CM,Hi,KVi,Si,nc,capacity,scale,VP,(int)VD,TLB);
          else eb3::bulk_swap<128,eb3::tune::BULK_MINB,false,true><<<grid,128>>>(PQ,CODES,SC,CB,BS,CM,Hi,KVi,Si,nc,capacity,scale,nullptr,(int)VD,TLB);
        }else{
          if(VA.xpf)eb3::bulk_swap<128,eb3::tune::BULK_MINB,true,false><<<grid,128>>>(PQ,CODES,SC,CB,BS,CM,Hi,KVi,Si,nc,capacity,scale,VP,(int)VD,nullptr);
          else eb3::bulk_swap<128,eb3::tune::BULK_MINB,false,false><<<grid,128>>>(PQ,CODES,SC,CB,BS,CM,Hi,KVi,Si,nc,capacity,scale,nullptr,(int)VD,nullptr);
        }
        break;
      case 3:
        if(VA.xpf)eb3::bulk_swap<128,eb3::tune::BULK_MINB_ALT,true,false><<<grid,128>>>(PQ,CODES,SC,CB,BS,CM,Hi,KVi,Si,nc,capacity,scale,VP,(int)VD,nullptr);
        else eb3::bulk_swap<128,eb3::tune::BULK_MINB_ALT,false,false><<<grid,128>>>(PQ,CODES,SC,CB,BS,CM,Hi,KVi,Si,nc,capacity,scale,nullptr,(int)VD,nullptr);
        break;
      default: throw std::runtime_error("apa_eb3: bulk id");
    }
    cuda_check_last("apa_eb3_bulk");
  }
  // 3. select (+ merge)
  unsigned* CTR=nullptr;
  if(VA.sel==1 && VA.fuse)CTR=eb3::counters((int)(B*KVH));
  {
    const int Hi=(int)H,KVi=(int)KVH,Si=(int)S;
    auto launch=[&](auto gt,auto dg,auto big,auto vpf,auto fuse,auto on,auto kpf){
      constexpr int GT=decltype(gt)::value;constexpr bool DG=decltype(dg)::value,BG=decltype(big)::value,VP=decltype(vpf)::value,FU=decltype(fuse)::value,ON=decltype(on)::value,KP=decltype(kpf)::value;
      constexpr int MB=BG?1:eb3::tune::SEL_MINB;
#if TC_APA_P1_TEST_HOOKS
      eb3::early_counter_check(test_hook,(const void*)eb3::select_pv<128,128,GT,DG,MB,BG,VP,FU,ON,KP>);
#endif
      eb3::select_pv<128,128,GT,DG,MB,BG,VP,FU,ON,KP><<<grid,128>>>(Q,K,V,BS,CM,M,PX,PM,PL,PA,Hi,KVi,Si,nc,pk,scale,delta,CTR,SNK,OUT,ON?TLS:nullptr
#if TC_APA_P1_TEST_HOOKS
        ,test_hook
#endif
      );};
    using T=std::true_type;using F=std::false_type;
    // All G in [2,8] are instantiated only for the registered select options;
    // ablations are G=2,4.
    auto byG=[&](auto dg,auto big,auto vpf,auto fuse,auto on,auto kpf){
      constexpr bool REG=!decltype(big)::value && decltype(vpf)::value==eb3::tune::REG_VPF && decltype(fuse)::value==eb3::tune::REG_FUSE && decltype(kpf)::value==eb3::tune::REG_KPF && !decltype(on)::value;
      switch(G){
        case 2: launch(std::integral_constant<int,2>{},dg,big,vpf,fuse,on,kpf);break;
        case 4: launch(std::integral_constant<int,4>{},dg,big,vpf,fuse,on,kpf);break;
        default:
          if constexpr(REG){
            switch(G){
              case 3: launch(std::integral_constant<int,3>{},dg,big,vpf,fuse,on,kpf);break;
              case 5: launch(std::integral_constant<int,5>{},dg,big,vpf,fuse,on,kpf);break;
              case 6: launch(std::integral_constant<int,6>{},dg,big,vpf,fuse,on,kpf);break;
              case 7: launch(std::integral_constant<int,7>{},dg,big,vpf,fuse,on,kpf);break;
              case 8: launch(std::integral_constant<int,8>{},dg,big,vpf,fuse,on,kpf);break;
              default: throw std::runtime_error("apa_eb3: G range");
            }
          }else throw std::runtime_error("apa_eb3: G not instantiated for this variant");
      }
    };
    if(VA.sel==0){
      spq4::select_pv<128,128,false><<<grid,128>>>(Q,K,V,BS,CM,M,PX,PM,PL,PA,Hi,KVi,Si,nc,pk,scale,delta);
    }else if(VA.sel==2){
      if(timeline)byG(F{},T{},F{},F{},T{},F{});else byG(F{},T{},F{},F{},F{},F{});
    }else{
      // sel==1: the (vpf, fuse, kpf) combinations used by the variant table;
      // diag only for the registered combination, timeline for registered
      // and occupancy-only.
      auto opts=[&](auto vpf,auto fuse,auto kpf){
        constexpr bool REGC=decltype(vpf)::value==eb3::tune::REG_VPF && decltype(fuse)::value==eb3::tune::REG_FUSE && decltype(kpf)::value==eb3::tune::REG_KPF;
        constexpr bool OCC=!decltype(vpf)::value && !decltype(fuse)::value && !decltype(kpf)::value;
        if(diag){if constexpr(REGC)byG(T{},F{},vpf,fuse,F{},kpf);else throw std::runtime_error("apa_eb3: diag combination");}
        else if(timeline){if constexpr(REGC || OCC)byG(F{},F{},vpf,fuse,T{},kpf);else throw std::runtime_error("apa_eb3: timeline combination");}
        else byG(F{},F{},vpf,fuse,F{},kpf);
      };
      if(!VA.vpf && !VA.fuse && !VA.kpf)opts(F{},F{},F{});
      else if(VA.vpf && !VA.fuse && !VA.kpf)opts(T{},F{},F{});
      else if(VA.vpf && VA.fuse && !VA.kpf)opts(T{},T{},F{});
      else if(VA.vpf && VA.fuse && VA.kpf)opts(T{},T{},T{});
      else throw std::runtime_error("apa_eb3: select option combination not instantiated");
    }
    cuda_check_last("apa_eb3_select_pv");
  }
  if(!(VA.sel==1 && VA.fuse)){
    spo2::merge_kernel<spq4::bf16><<<rows,128>>>(PM,PL,PA,SNK,OUT,(int)H,(int)VD,nc);
    cuda_check_last("apa_eb3_merge");
  }
  int64_t ctr_bytes=0;if(CTR)eb3::counters(0,&ctr_bytes);
  NDArray foot({6},DType::Int64,Device::from_string("cpu"));
  {auto* f=(int64_t*)foot.data_ptr();f[0]=total;f[1]=B*H*VD*2;f[2]=ctr_bytes;f[3]=(VA.sel==1 && VA.fuse)?3:4;f[4]=do_rpf;f[5]=do_xpf;}
#if TC_APA_P1_TEST_HOOKS
  eb3::early_counter_finish(test_hook,CTR,(int)(B*KVH));   // R5: resets the merge slot on a timeout
#endif
  operand_device.restore();
  return {out,pq,mask,bulk,prefix,tlb,tls,foot};
}

// Static resource census (cudaFuncGetAttributes + occupancy API) for the
// baseline kernels and the EB3 kernels, G=2 and G=4. Prior art: CUDA
// occupancy calculator API (NVIDIA). Diagnostic only.
std::vector<std::tuple<std::string,int,int,int,int,int>> apa_eb3_resources(){
  std::vector<std::tuple<std::string,int,int,int,int,int>> out;
  auto add=[&](const char* name,const void* fn){
    cudaFuncAttributes a{};int blocks=0;
    if(cudaFuncGetAttributes(&a,fn)!=cudaSuccess)throw std::runtime_error("apa_eb3_resources: attributes");
    if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,fn,128,0)!=cudaSuccess)throw std::runtime_error("apa_eb3_resources: occupancy");
    out.emplace_back(name,a.numRegs,(int)a.sharedSizeBytes,(int)a.localSizeBytes,a.maxThreadsPerBlock,blocks);
  };
  add("baseline spq1::rotate<bf16>",(const void*)spq1::rotate<spq4::bf16>);
  add("baseline spq6::bulk_mma_tq<128,2>",(const void*)spq6::bulk_mma_tq<128,2>);
  add("baseline spq4::select_pv<128,128,false>",(const void*)spq4::select_pv<128,128,false>);
  add("baseline spo2::merge_kernel<bf16>",(const void*)spo2::merge_kernel<spq4::bf16>);
  add("eb3 rotate_q<128,ROT_DB>",(const void*)eb3::rotate_q<128,eb3::tune::ROT_DB>);
  add("eb3 bulk_copy<128,1> (baseline footprint)",(const void*)eb3::bulk_copy<128,1,false>);
  add("eb3 bulk_swap<128,BULK_MINB>",(const void*)eb3::bulk_swap<128,eb3::tune::BULK_MINB,false,false>);
  add("eb3 bulk_swap<128,BULK_MINB,V prefetch>",(const void*)eb3::bulk_swap<128,eb3::tune::BULK_MINB,true,false>);
  add("eb3 bulk_swap<128,BULK_MINB_ALT,V prefetch>",(const void*)eb3::bulk_swap<128,eb3::tune::BULK_MINB_ALT,true,false>);
  add("eb3 select_pv G2 tuned (VPF,FUSE,KPF)",(const void*)eb3::select_pv<128,128,2,false,eb3::tune::SEL_MINB,false,true,true,false,true>);
  add("eb3 select_pv G4 tuned (VPF,FUSE,KPF)",(const void*)eb3::select_pv<128,128,4,false,eb3::tune::SEL_MINB,false,true,true,false,true>);
  add("eb3 select_pv G2 tuned (VPF,FUSE)",(const void*)eb3::select_pv<128,128,2,false,eb3::tune::SEL_MINB,false,true,true,false>);
  add("eb3 select_pv G4 tuned (VPF,FUSE)",(const void*)eb3::select_pv<128,128,4,false,eb3::tune::SEL_MINB,false,true,true,false>);
  add("eb3 select_pv G2 tuned (VPF)",(const void*)eb3::select_pv<128,128,2,false,eb3::tune::SEL_MINB,false,true,false,false>);
  add("eb3 select_pv G4 tuned (VPF)",(const void*)eb3::select_pv<128,128,4,false,eb3::tune::SEL_MINB,false,true,false,false>);
  add("eb3 select_pv G2 occupancy-only",(const void*)eb3::select_pv<128,128,2,false,eb3::tune::SEL_MINB,false,false,false,false>);
  add("eb3 select_pv G4 occupancy-only",(const void*)eb3::select_pv<128,128,4,false,eb3::tune::SEL_MINB,false,false,false,false>);
  add("eb3 select_pv G2 baseline footprint",(const void*)eb3::select_pv<128,128,2,false,1,true,false,false,false>);
  add("eb3 select_pv G4 baseline footprint",(const void*)eb3::select_pv<128,128,4,false,1,true,false,false,false>);
  return out;
}
#if TC_APA_P1_TEST_HOOKS
// APA-P1-F1 hooks-build bridge for the LF5 ticket read (block-scope
// declaration in the binding; tc/ops.h was outside F1's writable set). No
// prior art claimed; plumbing.
}  // namespace tc (closed for the bridge; reopened below)
std::vector<unsigned> apa_p1f1_merge_tickets(int n){return tc::eb3::early_counter_merge_slots(n);}
namespace tc {
#endif
// APA_EB3_END latency
