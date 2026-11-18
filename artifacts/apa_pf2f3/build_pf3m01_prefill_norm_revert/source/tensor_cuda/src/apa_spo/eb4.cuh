// APA_EB4_BEGIN tk8
// APA-EB4: the tk8 value-precision decision (arm T-fresh, the control) on EB3's registered variant-B path.
// Included from spq4.cuh only under TC_APA_EB4 (the build also defines TC_APA_EB3 and TC_APA_EB2B, whose device
// functions are reused). New kernels only: no EB3, EB2b or default kernel is edited, so their SASS is unchanged.
//
// Path (3 launches, as EB3). Registered configuration = eb4_tuning.cuh defaults (CHOOSE_SITE 1):
//  1. eb3::rotate_q<128,ROT_DB> with EB3's registered codes+norms L2 prefetch (the same kernel EB3 launches).
//  2. eb4::bulk_stats = eb3::bulk_swap<128,BULK_MINB,XPF,false> instruction-for-instruction for scores and
//     chunk maxima, plus by-products in its epilogue:
//       * EB2b's per-(row, chunk) statistics, same fp32 arithmetic and order as eb2b::statistics: per 64-key block
//         z = (0 + e(lane)) + e(lane+32), e = __expf(bulk - cm_c), xor-butterfly 16..1; blog = z>0 ? logf(z)+cm_c
//         : -inf; cz = (zs0+zs1)+(zs2+zs3). The bulk values are staged in shared memory (bitwise the values
//         written to global), so nothing is re-read from DRAM;
//       * an L2 prefetch of the chunk's int8 V + scales for select (tuning BULK_XPF, L2-budgeted);
//       * (development alternative, CHOOSE_SITE 0 only) the tk8 choose in the last CTA of each (batch, KV head)
//         via the threadfence/atomicInc last-block ticket, Batcher bitonic sort as EB2b. Not registered: it put
//         7.7 us (8K) to 20 us (32K) of serial sort on the bulk tail (ledger, Development).
//  3. eb4::select_mixed = eb3::select_pv registered instantiation (VPF, fused merge) S1-S3 and the merge verbatim,
//     plus, before S1, the tk8 choose for the CTA's own 4 blocks (CHOOSE_SITE 1): per head lz with EB2b's
//     arithmetic (exact max; ordered fma chain over chunks, operands by warp shuffle), every block's score
//     max_g __expf(blog - lz) as EB2b, and each own block's stable descending rank by comparison counting, i.e. its
//     position under EB2b's (score desc, id asc) order, so the flags equal EB2b's bitonic flags bitwise. Every
//     CTA of a (batch, KV head) computes identical scores. S4 reads each 32-key V tile through EB2b's loader rule
//     (bf16 row, or int8 row + fp32 per-token scale dequantized by eb2b::value8 to bf16 in registers) into the
//     unchanged bf16 shared tile and bf16 PV mma. The CTA-start L2 prefetch follows the flags.
// Every key stays in the normalizer; only the V operand of int8 blocks changes. With every flag forced to bf16 the
// output is EB3's bitwise (G1.1); with T-fresh flags it is EB2b's tk8 output bitwise (G1.2).
//
// Prior art (Prior Art Directive; code-site annotations):
//  * tk8 rule: ThriftAttention (Sharratt 2026, arXiv 2605.23081) important blocks at high precision, the rest low,
//    one softmax; Calver (arXiv 2605.20868) per-block INT8->FP16 promotion from mass. TAKEN via E2a/EB2b. Ours:
//    nothing new in the rule.
//  * int8 per-token V: KIVI (Liu et al. 2024, arXiv 2402.02750) per-token value quantization; here symmetric
//    absmax/127 with fp32 scales (EB2b format). TAKEN.
//  * Flash-Decoding (Dao, Haziza, Massa, Sizov 2023): split-K partials + merge. TAKEN unchanged (EB3 fused merge).
//  * BLASST (Yuan et al. 2025, arXiv 2512.12087): running-max refine criterion, inherited unchanged in S1.
//  * Rank by comparison counting (enumeration sort: Knuth, TAOCP Vol. 3 §5.2, "comparison counting"; parallel
//    rank/enumeration sort, Muller & Preparata 1975). Unverified as citations, lead to check: "comparison counting
//    sort Knuth", "Muller Preparata 1975 enumeration sort". TAKEN: rank = count of predecessors under a strict
//    total order. Ours: each CTA ranking only its own 4 blocks, redundantly per CTA, instead of a global sort.
//  * Last-CTA pattern (development alternative): NVIDIA CUDA Samples threadFenceReduction. TAKEN (as EB3).
//  * Batcher (1968) bitonic sorting network with an index tie-break (as EB2b), development alternative. TAKEN.
//  * Online normalizer statistics (Milakov & Gimelshein 2018) as split partial sums. TAKEN (EB2b arithmetic).
//  * Software L2 prefetch (Mowry, Lam, Gupta 1992; PTX prefetch.global.L2). TAKEN (EB3 helper).
//  * Mutation testing (DeMillo, Lipton, Sayward 1978): APA_EB4_MUTANT hooks, 0 in every non-mutant build.
// No prior art known to me for this exact assembly (statistics in the swap-AB TQ4 bulk epilogue, per-CTA redundant
// exact rank in select); no novelty claimed.
#include "eb4_tuning.cuh"
#ifndef APA_EB4_MUTANT
#define APA_EB4_MUTANT 0
#endif
namespace eb4 {
using bf16=__nv_bfloat16;
constexpr int C=spq4::C,NT=spq4::NT,MAXG=spq4::MAXG,PSTRIDE=spq4::PSTRIDE,BLK=64,BPC=C/BLK;
static_assert(BPC==4,"four 64-key blocks per 256-key chunk (EB2b statistics layout)");
static_assert(MAXG*C>=2*tune::NSORT_MAX,"sort scratch reuses the staged bulk values");

// ---- prefetch of a chunk's V operand toward L2, per 64-key block ------------------------------------------
// mode 1: bf16 rows of bf16 blocks, int8 rows + scales of int8 blocks; mode 2: bf16 rows of bf16 blocks only.
// PTX prefetch.global.L2 (no register, no arithmetic). Mowry/Lam/Gupta 1992 software prefetching.
__device__ __forceinline__ void prefetch_v(const bf16* v,const int8_t* vi,const float* vs,const uint8_t* hfl,
  int64_t bh,int S,int start,int count,int VD,int mode,int tid){
  #pragma unroll
  for(int bl=0;bl<BPC;++bl){
    int k0=bl*BLK;if(k0>=count)break;int nk=min(BLK,count-k0);
    int64_t row=bh*S+start+k0;
    if(hfl[bl])eb3::l2_prefetch(v+row*VD,(int64_t)nk*VD*(int64_t)sizeof(bf16),tid,NT);
    else if(mode==1){eb3::l2_prefetch(vi+row*VD,(int64_t)nk*VD,tid,NT);eb3::l2_prefetch(vs+row,(int64_t)nk*4,tid,NT);}
  }
}

// ---- lz of one head by one warp: eb2b::choose arithmetic -------------------------------------------------
// m = max over the head's chunk maxima (fmaxf is exact and order-free); z = fma chain over chunks ascending with
// e_c = __expf(cm_c - m), exactly EB2b's `z += cz*__expf(cm-m)` (its SASS is FFMA, z from 0); lz = logf(z) + m.
// The chain operands travel by warp shuffle so the ordered chain never waits on L2 (EB2b's sequential loop did).
__device__ __forceinline__ float lz_warp(const float* cm,const float* cz,int64_t row,int NC,int lane){
  float m=-INFINITY;
  for(int cc=lane;cc<NC;cc+=32)m=fmaxf(m,__ldcg(cm+row*NC+cc));
  m=spq4::warp_max(m);
  float z=0.f;
  for(int c0=0;c0<NC;c0+=32){
    int cc=c0+lane;float a=0.f,e=0.f;
    if(cc<NC){a=__ldcg(cz+row*NC+cc);e=__expf(__ldcg(cm+row*NC+cc)-m);}
    int nn=min(32,NC-c0);
#if APA_EB4_MUTANT==8
    for(int i=nn-1;i>=0;--i)z=__fmaf_rn(__shfl_sync(0xffffffffu,a,i),__shfl_sync(0xffffffffu,e,i),z);   // Mutant 8: chain order reversed within each 32-chunk group
#else
    if(nn==32){
      // Full group: unrolled so the shuffles issue ahead of the ordered fma chain (same order, same operands).
      #pragma unroll
      for(int i=0;i<32;++i)z=__fmaf_rn(__shfl_sync(0xffffffffu,a,i),__shfl_sync(0xffffffffu,e,i),z);
    }else{
      for(int i=0;i<nn;++i)z=__fmaf_rn(__shfl_sync(0xffffffffu,a,i),__shfl_sync(0xffffffffu,e,i),z);
    }
#endif
  }
  return logf(z)+m;
}

// ---- 2. bulk + EB2b statistics + last-CTA tk8 choose ---------------------------------------------------------
// Scores/maxima: eb3::bulk_swap<D,BULK_MINB,XPF,false> verbatim (same products, k-slots, MMA order, (acc*norm)*scale).
template<int D,int XPF,bool BCH> __global__ void __launch_bounds__(128,eb3::tune::BULK_MINB) bulk_stats(const float* qr,const uint8_t* packed,
 const float* scales,const float* cb,float* values,float* cm,int H,int KVH,int S,int NC,int capacity,float scale,
 const bf16* vpf,const int8_t* vipf,const float* vspf,int VD,
 float* cz,float* blog,int NB,unsigned* ctr,uint8_t* flags,float* scores,float* logz,float fraction,bool force){
  constexpr int DPL=D/4,NS=DPL/4,NW=DPL/8,MT=C/16/4;
  __shared__ float wmax[4][MAXG];
  __shared__ unsigned ctab[256];
  __shared__ unsigned ctlo[256];
  __shared__ __align__(16) float sv[MAXG][C];      // staged bulk values; reused as the sort scratch in the last CTA
  __shared__ float msh[MAXG],zs[MAXG][BPC],lz[MAXG];
  __shared__ unsigned last_s;
  int bh=blockIdx.x,c=blockIdx.y+gridDim.y*blockIdx.z;if(c>=NC)return;
  int G=H/KVH,b=bh/KVH,kvh=bh%KVH,h0=kvh*G,tid=threadIdx.x,lane=tid&31,warp=tid>>5,r=lane>>2,tig=lane&3;
  int start=c*C,count=min(C,S-start);
  if constexpr(XPF==1){
    if(vpf)eb3::l2_prefetch(vpf+((int64_t)bh*S+start)*VD,(int64_t)count*VD*(int64_t)sizeof(bf16),tid,NT);
  }else if constexpr(XPF==2){
    // int8 V rows + fp32 scales of this chunk toward L2 (host-budgeted); select reads them for int8 blocks.
    if(vipf){eb3::l2_prefetch(vipf+((int64_t)bh*S+start)*VD,(int64_t)count*VD,tid,NT);eb3::l2_prefetch(vspf+(int64_t)bh*S+start,(int64_t)count*4,tid,NT);}
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
      eb3::mma_full(acc,a0,a1,a2,a3,qh[s].x,qh[s].y);            // C_hi * q_hi
      eb3::mma_full(acc,a0,a1,a2,a3,ql[s].x,ql[s].y);            // C_hi * q_lo
      unsigned l0=ctlo[x0&0xffu],l2=ctlo[x0>>8],l1=ctlo[x1&0xffu],l3=ctlo[x1>>8];
      eb3::mma_full(acc,l0,l1,l2,l3,qh[s].x,qh[s].y);            // C_lo * q_hi
    }
    int j0=(warp+4*t)*16+r,j1=j0+8;
    #pragma unroll
    for(int e=0;e<2;++e){
      int head=2*tig+e;
      if(head<G){
        int64_t rowS=(int64_t)(b*H+h0+head)*S+start;
        if(j0<count){float x=__fmul_rn(__fmul_rn(acc[e],k0[t]),scale);values[rowS+j0]=x;sv[head][j0]=x;if(e)mx1=fmaxf(mx1,x);else mx0=fmaxf(mx0,x);}
        if(j1<count){float x=__fmul_rn(__fmul_rn(acc[2+e],k1[t]),scale);values[rowS+j1]=x;sv[head][j1]=x;if(e)mx1=fmaxf(mx1,x);else mx0=fmaxf(mx0,x);}
      }
    }
  }
  #pragma unroll
  for(int o=4;o<32;o<<=1){mx0=fmaxf(mx0,__shfl_xor_sync(0xffffffffu,mx0,o));mx1=fmaxf(mx1,__shfl_xor_sync(0xffffffffu,mx1,o));}
  if(r==0){wmax[warp][2*tig]=mx0;wmax[warp][2*tig+1]=mx1;}
  __syncthreads();
  if(tid<G){
    float m=fmaxf(fmaxf(wmax[0][tid],wmax[1][tid]),fmaxf(wmax[2][tid],wmax[3][tid]));
    cm[(int64_t)(b*H+h0+tid)*NC+c]=m;msh[tid]=m;
  }
  __syncthreads();
  // ---- EB2b statistics (eb2b::statistics arithmetic): warp w = block c*4+w, every head of the group ----
  // Milakov/Gimelshein 2018 online-normalizer partials; Flash-Decoding 2023 split statistics. Every key counted.
  for(int g=0;g<G;++g){
    float m=msh[g],z=0.f;
    #pragma unroll
#if APA_EB4_MUTANT==10
    for(int t=0;t<1;++t){int j=warp*BLK+lane+32*t;if(j<count)z+=__expf(sv[g][j]-m);}   // Mutant 10: half of each block summed
#else
    for(int t=0;t<2;++t){int j=warp*BLK+lane+32*t;if(j<count)z+=__expf(sv[g][j]-m);}
#endif
    #pragma unroll
    for(int o=16;o;o>>=1)z+=__shfl_xor_sync(0xffffffffu,z,o);
    if(lane==0){
      zs[g][warp]=z;int blk=c*BPC+warp;
      if(blk<NB)blog[(int64_t)(b*H+h0+g)*NB+blk]=z>0.f?logf(z)+m:-INFINITY;
    }
  }
  __syncthreads();
  if(tid<G)cz[(int64_t)(b*H+h0+tid)*NC+c]=(zs[tid][0]+zs[tid][1])+(zs[tid][2]+zs[tid][3]);
  if constexpr(!BCH)return;   // CHOOSE_SITE 1: the choose runs in select
  // ---- last CTA of this (batch, KV head): the tk8 choose (eb2b::choose arithmetic) ----
  // threadFenceReduction last-block pattern (NVIDIA CUDA Samples): every writer fences, one ticket per CTA, the
  // ticket wraps to 0 on the last arrival (atomicInc limit NC-1), the reader loads L2 (__ldcg) after a fence.
  __threadfence();
  __syncthreads();
  if(tid==0){
    unsigned prev=atomicInc(ctr+bh,(unsigned)(NC-1));
    last_s=prev==(unsigned)(NC-1);
  }
  __syncthreads();
  if(!last_s)return;
  __threadfence();
  const int row0=b*H+h0;
  for(int g=warp;g<G;g+=4){
    float l=lz_warp(cm,cz,row0+g,NC,lane);
    if(lane==0){lz[g]=l;if(logz)logz[row0+g]=l;}
  }
  __syncthreads();
  float* ss=&sv[0][0];int* ids=reinterpret_cast<int*>(ss+tune::NSORT_MAX);
  int nsort=1;while(nsort<NB)nsort<<=1;
  for(int blk=tid;blk<nsort;blk+=NT){
    float score=-INFINITY;
    if(blk<NB){
#if APA_EB4_MUTANT==6
      int ng=1;                                                 // Mutant 6: GQA score from head 0 only
#else
      int ng=G;
#endif
      for(int g=0;g<ng;++g)score=fmaxf(score,__expf(__ldcg(blog+(int64_t)(row0+g)*NB+blk)-lz[g]));
      if(scores)scores[(int64_t)bh*NB+blk]=score;
    }
    ss[blk]=score;ids[blk]=blk;
  }
  __syncthreads();
  // Batcher 1968 bitonic network, comparator (score descending, id ascending) = stable descending order.
  for(int kk=2;kk<=nsort;kk<<=1)for(int jj=kk>>1;jj;jj>>=1){
    for(int i=tid;i<nsort;i+=NT){int other=i^jj;if(other>i){
      float a=ss[i],bb=ss[other];int ia=ids[i],ib=ids[other];
#if APA_EB4_MUTANT==7
      bool better=a>bb || (a==bb && ia>ib);                     // Mutant 7: ties go to the higher block index
#else
      bool better=a>bb || (a==bb && ia<ib);
#endif
      bool swap=(i&kk)?better:!better;
      if(swap){ss[i]=bb;ss[other]=a;ids[i]=ib;ids[other]=ia;}
    }}
    __syncthreads();
  }
  int n=int(ceilf(fraction*float(NB)));
#if APA_EB4_MUTANT==5
  n=max(0,n-1);                                                 // Mutant 5: cutoff off by one
#endif
  for(int rank=tid;rank<nsort;rank+=NT)if(ids[rank]<NB)
    flags[(int64_t)bh*NB+ids[rank]]=force || rank<n;
}

// ---- 3. select + exact K + softmax + mixed-V PV + fused merge ------------------------------------------------
// eb3::select_pv<D,VD,GT,DIAG,SEL_MINB,false,true,true,false,false> verbatim except: the chunk's 4 block flags are
// read at CTA start (hf), the CTA-start V prefetch follows them (tune::SEL_VPF), and S4's fetch takes each key's
// V row as bf16 (flag set) or int8 + scale -> bf16 via eb2b::value8 (EB2b loader rule; KIVI 2024 per-token V;
// ThriftAttention 2026 block precision). The shared tile, the PV mma, l and the merge are unchanged.
template<int D,int VD,int GT,bool DIAG,bool SCH>
__global__ void __launch_bounds__(128,eb3::tune::SEL_MINB) select_mixed(const bf16* q,const bf16* k,const bf16* v,
 const int8_t* vi,const float* vs,const uint8_t* flags,uint8_t* used,
 const float* cz,const float* blog,uint8_t* fl_out,float* sc_out,float* lz_out,const uint8_t* inject,float fraction,bool force,int vpf_mode,
 const float* bulk,const float* cm,uint8_t* mask,float* prefix,float* pm,float* pl,float* pa,
 int H,int KVH,int S,int NC,int PK,float scale,float delta,unsigned* ctr,const bf16* sinks,bf16* out
#if TC_APA_P1_TEST_HOOKS
 ,eb3::EarlyCounterHook test_hook
#endif
 ){
  constexpr int G=GT,SG=GT;
  constexpr int DPL=D/4,NS=DPL/4,NU4=DPL/8,NTW=VD/32,U=VD/8;
  __shared__ float scs[SG][C];
  __shared__ uint8_t use[SG][C],any_[C];
  __shared__ __align__(16) bf16 Ps[SG][PSTRIDE];
  __shared__ __align__(16) bf16 Vs[32][VD+8];
  __shared__ float lsum[SG];
  __shared__ uint8_t hf[BPC];
  int bh=blockIdx.x,c=blockIdx.y+gridDim.y*blockIdx.z;if(c>=NC)return;
#if TC_APA_P1_TEST_HOOKS
  // Survey S3: hold chunk 1 before ALL partial stores, not after pm/pl.
  if(!eb3::early_counter_wait(test_hook,c))return;
#endif
  int b=bh/KVH,h0=(bh%KVH)*G,tid=threadIdx.x,lane=tid&31,warp=tid>>5,r=lane>>2,tig=lane&3;
  int start=c*C,count=min(C,S-start);
  const int NB=(S+BLK-1)/BLK;
  uint8_t own[BPC]={0,0,0,0};
  if constexpr(SCH){
    // ---- tk8 choose for this CTA's 4 blocks (CHOOSE_SITE 1), eb2b::choose arithmetic for lz and every block
    // score; the rank of block i is #{j : s_j > s_i or (s_j == s_i and j < i)}, i.e. its position under EB2b's
    // (score desc, id asc) total order, so the flags equal the bitonic sort's. Every CTA of the (batch, KV head)
    // computes the same scores (same operands, same instructions). Scratch: the V tile (dead until S4).
    __shared__ float lzs[GT];
    __shared__ int cnt_s[4][BPC];
    static_assert(sizeof(Vs)>=sizeof(float)*(size_t)tune::NSORT_MAX,"score scratch fits in Vs");
    float* ssc=reinterpret_cast<float*>(&Vs[0][0]);
    const int64_t row0=(int64_t)b*H+h0;
    for(int g=warp;g<G;g+=4){
      float l=lz_warp(cm,cz,row0+g,NC,lane);
      if(lane==0){lzs[g]=l;if(lz_out && c==0)lz_out[row0+g]=l;}
    }
    __syncthreads();
    for(int j=tid;j<NB;j+=NT){
      float sc=-INFINITY;
#if APA_EB4_MUTANT==6
      const int ng=1;                                               // Mutant 6: GQA score from head 0 only
#else
      const int ng=G;
#endif
      for(int g=0;g<ng;++g)sc=fmaxf(sc,__expf(__ldcg(blog+(row0+g)*NB+j)-lzs[g]));
      ssc[j]=sc;
    }
    __syncthreads();
    float si[BPC];int cnt[BPC];
    #pragma unroll
    for(int bl=0;bl<BPC;++bl){int i=c*BPC+bl;si[bl]=i<NB?ssc[i]:-INFINITY;cnt[bl]=0;}
    for(int j=tid;j<NB;j+=NT){
      float sj=ssc[j];
      #pragma unroll
      for(int bl=0;bl<BPC;++bl){
        int i=c*BPC+bl;
#if APA_EB4_MUTANT==7
        cnt[bl]+=(sj>si[bl]) || (sj==si[bl] && j>i);                // Mutant 7: ties go to the higher block index
#else
        cnt[bl]+=(sj>si[bl]) || (sj==si[bl] && j<i);
#endif
      }
    }
    #pragma unroll
    for(int bl=0;bl<BPC;++bl){
      int x=cnt[bl];
      #pragma unroll
      for(int o=16;o;o>>=1)x+=__shfl_xor_sync(0xffffffffu,x,o);
      if(lane==0)cnt_s[warp][bl]=x;
    }
    __syncthreads();
    int n=int(ceilf(fraction*float(NB)));
#if APA_EB4_MUTANT==5
    n=max(0,n-1);                                                   // Mutant 5: cutoff off by one
#endif
    #pragma unroll
    for(int bl=0;bl<BPC;++bl){
      int rank=(cnt_s[0][bl]+cnt_s[1][bl])+(cnt_s[2][bl]+cnt_s[3][bl]);
      own[bl]=force || rank<n;
    }
    if(tid<BPC){int blk=c*BPC+tid;if(blk<NB){if(fl_out)fl_out[(int64_t)bh*NB+blk]=own[tid];if(sc_out)sc_out[(int64_t)bh*NB+blk]=si[tid];}}
  }
  {
    // Block flags of this chunk: computed above (CHOOSE_SITE 1), written by bulk_stats' last CTA (CHOOSE_SITE 0),
    // or injected by the caller (G1.3).
    uint8_t f4[BPC];
    #pragma unroll
    for(int bl=0;bl<BPC;++bl){
      int blk=c*BPC+bl;
      const uint8_t* src=inject?inject:flags;
      uint8_t f=(SCH && !inject)?own[bl]:(blk<NB?src[(int64_t)bh*NB+blk]:(uint8_t)0);
#if APA_EB4_MUTANT==1
      f=1;                                                      // Mutant 1: flags ignored, all bf16
#elif APA_EB4_MUTANT==2
      f=0;                                                      // Mutant 2: flags ignored, all int8
#endif
      f4[bl]=f!=0;
    }
    if(tid<BPC){hf[tid]=f4[tid];int blk=c*BPC+tid;if constexpr(DIAG){if(used && blk<NB)used[(int64_t)bh*NB+blk]=f4[tid];}}
    if(vpf_mode)prefetch_v(v,vi,vs,f4,bh,S,start,count,VD,vpf_mode,tid);
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
      int gi=warp+4*t,key=gi*8+r;bool usedk=key<count && any_[key];
      if(!__any_sync(0xffffffffu,usedk))continue;
      uint4 kw[NU4];
      #pragma unroll
      for(int i=0;i<NU4;++i)kw[i]=make_uint4(0u,0u,0u,0u);
      if(usedk){
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
  // ---- S4: PV on tensor cores; V tile per key from bf16 or int8 (EB2b loader rule) ----
  float acc[NTW][4];
  #pragma unroll
  for(int i=0;i<NTW;++i){acc[i][0]=acc[i][1]=acc[i][2]=acc[i][3]=0.f;}
  uint4 nxt[U/4];
  auto fetch=[&](int t0){
    #pragma unroll
    for(int i=0;i<U/4;++i){
      int idx=tid+NT*i,key=t0+idx/U;
      if(key<count){
        int64_t tok=(int64_t)bh*S+start+key,off=tok*VD+(idx%U)*8;
#if APA_EB4_MUTANT==9
        bool high=hf[min((key+1)>>6,BPC-1)]!=0;                 // Mutant 9: block boundary off by one
#else
        bool high=hf[key>>6]!=0;
#endif
#if APA_EB4_MUTANT==3
        nxt[i]=high?*reinterpret_cast<const uint4*>(v+off):eb2b::value8(vi+off,1.f);           // Mutant 3: scale dropped
#elif APA_EB4_MUTANT==4
        nxt[i]=high?*reinterpret_cast<const uint4*>(v+off):eb2b::value8(vi+off,vs[min(tok+1,(int64_t)bh*S+S-1)]);   // Mutant 4: neighbour's scale
#else
        nxt[i]=high?*reinterpret_cast<const uint4*>(v+off):eb2b::value8(vi+off,vs[tok]);
#endif
      }else nxt[i]=make_uint4(0u,0u,0u,0u);
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
  // ---- fused Flash-Decoding merge: eb3::select_pv's last-CTA merge verbatim (spo2::merge_kernel arithmetic) ----
  __shared__ unsigned last_s;
  __shared__ float mxs[GT],dens[GT];
  __threadfence();
  __syncthreads();
  if(tid==0){
    unsigned prev=atomicInc(ctr+bh,(unsigned)(NC-1));      // wraps to 0 on the last arrival
    last_s=prev==(unsigned)(NC-1);
  }
  __syncthreads();
  if(!last_s){
#if TC_APA_P1_TEST_HOOKS
    eb3::early_counter_done(test_hook,c);
#endif
    return;
  }
  __threadfence();
  const int64_t row0=(int64_t)b*H+h0;
  constexpr int Q4=VD/4,QPT=(G*Q4+NT-1)/NT,MB=eb3::tune::MERGE_BATCH/QPT;
  float sum=0.f;
  float4 macc[QPT];
  #pragma unroll
  for(int qi=0;qi<QPT;++qi)macc[qi]=make_float4(0.f,0.f,0.f,0.f);
  if(NC<=C){
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
        int cc=u;
        buf[qi][u]=(qd<G*Q4 && u<n)?__ldcg(a+(int64_t)cc*Q4):make_float4(0.f,0.f,0.f,0.f);
      }
    }
    for(int i=tid;i<G*n;i+=NT){int g=i/n,cc=i%n;scs[g][cc]=__ldcg(pm+(row0+g)*NC+cc);lsh[g*C+cc]=__ldcg(pl+(row0+g)*NC+cc);}
    __syncthreads();
    if(tid<G){
      float x=-INFINITY;
      for(int cc=0;cc<n;++cc)x=fmaxf(x,scs[tid][cc]);
      if(sinks)x=fmaxf(x,__bfloat162float(sinks[h0+tid]));
      mxs[tid]=x;
    }
    __syncthreads();
    for(int i=tid;i<G*n;i+=NT){int g=i/n,cc=i%n;scs[g][cc]=__expf(scs[g][cc]-mxs[g]);}
    __syncthreads();
    if(tid<G){
      for(int i=0;i<n;++i){int cc=i;sum=__fmaf_rn(lsh[tid*C+cc],scs[tid][cc],sum);}
      if(sinks)sum=__fadd_rn(sum,__expf(__bfloat162float(sinks[h0+tid])-mxs[tid]));
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
          int cc=u;
          if(u<n){float e=scs[g][cc];
            macc[qi].x=__fmaf_rn(buf[qi][u].x,e,macc[qi].x);macc[qi].y=__fmaf_rn(buf[qi][u].y,e,macc[qi].y);
            macc[qi].z=__fmaf_rn(buf[qi][u].z,e,macc[qi].z);macc[qi].w=__fmaf_rn(buf[qi][u].w,e,macc[qi].w);}
        }
        for(int i0=MB;i0<n;i0+=MB){
          float4 bb[MB];
          #pragma unroll
          for(int u=0;u<MB;++u){int cc=i0+u;bb[u]=(i0+u<n)?__ldcg(a+(int64_t)cc*Q4):make_float4(0.f,0.f,0.f,0.f);}
          #pragma unroll
          for(int u=0;u<MB;++u){
            int cc=i0+u;
            if(i0+u<n){float e=scs[g][cc];
              macc[qi].x=__fmaf_rn(bb[u].x,e,macc[qi].x);macc[qi].y=__fmaf_rn(bb[u].y,e,macc[qi].y);
              macc[qi].z=__fmaf_rn(bb[u].z,e,macc[qi].z);macc[qi].w=__fmaf_rn(bb[u].w,e,macc[qi].w);}
          }
        }
      }
    }
    __syncthreads();
  }else{
  float mxr=-INFINITY;
  for(int c0=0;c0<NC;c0+=C){
    int n=min(C,NC-c0);
    for(int i=tid;i<G*n;i+=NT){int g=i/n,cc=i%n;scs[g][cc]=__ldcg(pm+(row0+g)*NC+c0+cc);}
    __syncthreads();
    if(tid<G){for(int cc=0;cc<n;++cc)mxr=fmaxf(mxr,scs[tid][cc]);}
    __syncthreads();
  }
  if(tid<G){
    if(sinks)mxr=fmaxf(mxr,__bfloat162float(sinks[h0+tid]));
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
      for(int i=0;i<n;++i){int cc=i;sum=__fmaf_rn(__ldcg(l+cc),scs[tid][cc],sum);}
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
          for(int u=0;u<MB;++u){int cc=i0+u;bb[u]=(i0+u<n)?__ldcg(a+(int64_t)cc*Q4):make_float4(0.f,0.f,0.f,0.f);}
          #pragma unroll
          for(int u=0;u<MB;++u){
            int cc=i0+u;
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
    if(sinks)sum=__fadd_rn(sum,__expf(__bfloat162float(sinks[h0+tid])-mxs[tid]));
    dens[tid]=sum;
  }
  __syncthreads();
  }
  #pragma unroll
  for(int qi=0;qi<QPT;++qi){
    int qd=tid+qi*NT;
    if(qd<G*Q4){
      int g=qd/Q4,d0=(qd%Q4)*4;float den=dens[g];bf16* o=out+(row0+g)*VD+d0;
      // APA-P1-F1 R1 (P1R-F01): only an exactly-zero den writes 0; NaN now
      // propagates (IEEE 754). APA-P1-F2 R5 (P1R2-F06), stated narrowly: the
      // guard preserves every result whose intermediate scores and den are
      // finite (then den >= 0 and >0 equals !=0). Finite operands that
      // overflow, NaN/+Inf sinks and +-Inf queries over all-negative logits
      // follow IEEE propagation to NaN (the loud direction). Merge as eb3.cuh.
      o[0]=__float2bfloat16_rn(den!=0.f?__fdiv_rn(macc[qi].x,den):0.f);
      o[1]=__float2bfloat16_rn(den!=0.f?__fdiv_rn(macc[qi].y,den):0.f);
      o[2]=__float2bfloat16_rn(den!=0.f?__fdiv_rn(macc[qi].z,den):0.f);
      o[3]=__float2bfloat16_rn(den!=0.f?__fdiv_rn(macc[qi].w,den):0.f);
    }
  }
#if TC_APA_P1_TEST_HOOKS
  eb3::early_counter_done(test_hook,c);
#endif
}

// ---- persistent choose tickets, separate from merge on EACH CUDA device ---
// Contract: one stream, CUDA legacy default stream; every other stream is out
// of contract. Prior art: CUDA Samples threadFenceReduction, CUDA stream-ordered
// lifetime APIs (NVIDIA 2021), ISO C++11 mutex/map (2011). Taken: existing
// tickets/order; ours: per-device choose ownership via EB3's locked registry.
// Unverified -- lead to check: CUDA 11.2 stream ordered memory allocator 2021.
inline unsigned* counters(int n,int64_t* bytes=nullptr){
  return eb3::ticket_buffer(true,n,tune::COUNTER_MIN_ENTRIES,bytes);
}
} // namespace eb4

// ---- launcher ----------------------------------------------------------------------------------------------
// Validation and arena as apa_eb3_run (EB3 2026), plus EB2b's int8 V operands and the tk8 scratch. Returns
// {out, flags (bulk choose), used (select's flags), scores, logz, cm, cz, blog, mask, bulk, prefix, rotated q,
//  footprint}; entries past `out` are defined only with diag. `inject` (uint8 [B,KVH,NB]) replaces the choose's
// flags as select's input (G1.3); the choose still runs and still writes `flags`.
std::vector<NDArray> apa_eb4_run(const NDArray& q,const NDArray& k,const NDArray& v,const NDArray& vi,const NDArray& vs,
  const NDArray& codes,const NDArray& sc,const NDArray& rot,const NDArray& cb,float scale,float delta,float fraction,
  const NDArray* sinks,const std::string& carry_mode,bool force_bf16,const NDArray* inject,bool diag,bool poison){
  // APA-P1-F3 R3' (P1R3-F03): q/k/v extents above INT_MAX raise ValueError naming this entry and the operand, first.
  apa_spo_require_attention_extents("apa_eb4",{{"q",&q},{"k",&k},{"v",&v}});
  if(carry_mode!="partition" && carry_mode!="row")throw std::runtime_error("carry must be partition or row");
  const char* flag=std::getenv("TC_APA_SP");
  if(!flag || std::strcmp(flag,"1")!=0)throw std::runtime_error("apa_sp: experimental entry requires TC_APA_SP=1 (default OFF)");
  if(const char* ve=std::getenv("APA_FXTC1_VARIANT"))
    if(std::strcmp(ve,"2")!=0)throw std::runtime_error("apa_eb4: variant B only (APA_FXTC1_VARIANT must be unset or 2)");
  if(!std::isfinite(scale) || scale<=0.f || !std::isfinite(delta) || delta<0.f)throw std::runtime_error("apa_sp: scale must be finite positive; delta finite nonnegative");
  if(!std::isfinite(fraction) || fraction<=0.f || fraction>1.f)throw std::runtime_error("apa_eb4: fraction must be in (0,1]");
  if(q.ndim()!=4 || k.ndim()!=4 || v.ndim()!=4)throw std::runtime_error("apa_eb4: q/k/v must be rank four");
  if(q.dtype!=DType::BFloat16 || k.dtype!=DType::BFloat16 || v.dtype!=DType::BFloat16)throw std::runtime_error("apa_eb4: bfloat16 q/k/v required");
  for(const NDArray* a:{&q,&k,&v,&vi,&vs,&codes,&sc,&rot,&cb})if(!a->device.is_cuda() || a->device.index!=q.device.index)throw std::runtime_error("apa_eb4: inputs must share one CUDA device");
  apa_spo_require_extents("apa_eb4",{&q,&k,&v});   // R11: empty extents raise as the dense entry
#if !TC_APA_P1_TEST_HOOKS
  (void)apa_spo_knob("APA_P1_EARLY_COUNTER","0");   // R7: the hook is compiled out of the default build
#endif
  const int64_t B=q.shape[0],H=q.shape[1],L=q.shape[2],D=q.shape[3],KVH=k.shape[1],S=k.shape[2],VD=v.shape[3];
  if(L!=1)throw std::runtime_error("apa_eb4: requires L=1");
  if(KVH<=0 || H%KVH!=0 || k.shape[0]!=B || k.shape[3]!=D || v.shape[0]!=B || v.shape[1]!=KVH || v.shape[2]!=S)throw std::runtime_error("apa_eb4: shape/GQA mismatch");
  if(D!=128 || VD!=128)throw std::runtime_error("apa_eb4: D=VD=128 only");
  if(vi.dtype!=DType::Uint8 || vi.shape!=v.shape || vs.dtype!=DType::Float32 || vs.shape!=Shape({B,KVH,S}))throw std::runtime_error("apa_eb4: int8 V (uint8 bits, v's shape) and fp32 per-token scales [B,KVH,S] required");
  int G=(int)(H/KVH);
  if(G!=2 && G!=4 && G!=8)throw std::runtime_error("apa_eb4: H/KVH must be 2, 4 or 8 (instantiated groups)");
  if(S<1 || S>(int64_t)eb4::BLK*eb4::tune::NSORT_MAX || B*H>std::numeric_limits<int>::max()/4)throw std::runtime_error("apa_eb4: dimension range (S <= 65536)");
  if(rot.shape!=Shape({KVH,D,D}) || rot.dtype!=DType::Float32)throw std::runtime_error("apa_eb4: rotation shape/dtype");
  if(cb.shape!=Shape({16}) || cb.dtype!=DType::Float32)throw std::runtime_error("apa_eb4: codebook shape/dtype");
  if(sinks && (sinks->ndim()!=1 || sinks->shape[0]!=H || sinks->dtype!=q.dtype || !sinks->device.is_cuda() || sinks->device.index!=q.device.index))throw std::runtime_error("apa_sp: sinks must be [H] with matching dtype/device");
  const int64_t NB=(S+eb4::BLK-1)/eb4::BLK;
  if(inject && (inject->dtype!=DType::Uint8 || inject->shape!=Shape({B,KVH,NB}) || !inject->device.is_cuda() || inject->device.index!=q.device.index))throw std::runtime_error("apa_eb4: inject must be CUDA uint8 [B,KVH,ceil(S/64)]");
  apa_spq1_buffer_check(k,codes,sc);
  // R2 (P1R-F02) census for this launcher: rot float4 (eb3::rotate_q), codes
  // uint4 (bulk_stats), q uint2 / k uint4 (select_mixed S2), v uint4 (bf16
  // rows of the V fetch), vi uint2 (eb2b::value8 on int8 rows).
  apa_spo_require_aligned("apa_eb4","rotation",rot,16);
  apa_spo_require_aligned("apa_eb4","codes",codes,16);
  apa_spo_require_aligned("apa_eb4","q",q,8);
  apa_spo_require_aligned("apa_eb4","k",k,16);
  apa_spo_require_aligned("apa_eb4","v",v,16);
  apa_spo_require_aligned("apa_eb4","vi",vi,8);
  // Same locked enqueue interval as EB3, including choose and merge slots.
  // Prior art: legacy-default-stream ordering and C++11 RAII (2011).
  std::lock_guard<std::recursive_mutex> ticket_launch_lock(eb3::ticket_mutex());
  eb3::TicketDeviceGuard operand_device(q.device.index,"apa_eb4");
  int np,pk;apa_int4_fixed_partition_layout((int)S,np,pk);
  if(pk%spq4::C)throw std::runtime_error("apa_eb4: partition must align to chunk");
  int nc=(int)((S+spq4::C-1)/spq4::C),rows=(int)(B*H);
  if(carry_mode=="row")pk=nc*spq4::C;
  dim3 grid((unsigned)(B*KVH),std::min(nc,65535),(nc+65534)/65535);
  int64_t RS=(int64_t)rows*S,RC=(int64_t)rows*nc,RB=(int64_t)rows*NB,BF=(int64_t)B*KVH*NB;
  // APA-DS1 amendment A2 (2026-11-03): align every arena view for 16B consumers.
  // Prior art: aligned bump allocation (ISO C++17, 2017) and NVIDIA CUDA
  // Programming Guide natural vector alignment (2024 version unverified;
  // lead to check: CUDA float4 natural alignment). Taken: cursor round-up;
  // ours: apply it to this arena. Device code and arithmetic are unchanged.
  constexpr int64_t kArenaAlignment = 16;
  std::vector<int64_t> sizes={4*(int64_t)rows*D,RS,4*RS,diag?4*RS:0,4*RC,4*RC,4*RC,4*RC*VD,4*RC,4*RB,BF,diag?BF:0,diag?4*BF:0,diag?4*(int64_t)rows:0};
  int64_t total=0;for(auto n:sizes){if(n==0)continue;total=(total+kArenaAlignment-1)&~(kArenaAlignment-1);total+=n;} // A3: zero-byte slots; aligned bump allocation (ISO C++17, 2017).
  NDArray arena({total},DType::Uint8,q.device),out({B,H,1,VD},q.dtype,q.device);
  if(poison){
    if(cudaMemsetAsync(arena.data_ptr(),0xff,(size_t)total,0)!=cudaSuccess || cudaMemsetAsync(out.data_ptr(),0xff,(size_t)(B*H*VD*2),0)!=cudaSuccess)
      throw std::runtime_error("apa_eb4: poison memset");
  }
  int64_t cursor=0;
  auto view=[&](Shape shape,DType dtype)->NDArray{
    cursor=(cursor+kArenaAlignment-1)&~(kArenaAlignment-1);NDArray x=arena;x.shape=shape;x.dtype=dtype;
    x.offset=cursor/dtype_size(dtype);cursor+=x.numel()*dtype_size(dtype);return x;};
  NDArray pq=view({B,H,1,D},DType::Float32);
  NDArray mask=view({B,H,1,S},DType::Uint8),bulk=view({B,H,1,S},DType::Float32),prefix;
  if(diag)prefix=view({B,H,1,S},DType::Float32);
  NDArray cm=view({B,H,nc},DType::Float32),pm=view({RC},DType::Float32),pl=view({RC},DType::Float32),pa=view({RC*VD},DType::Float32);
  NDArray cz=view({B,H,nc},DType::Float32),blog=view({B,H,NB},DType::Float32),flags=view({B,KVH,NB},DType::Uint8),used,scores,logz;
  if(diag){used=view({B,KVH,NB},DType::Uint8);scores=view({B,KVH,NB},DType::Float32);logz=view({B,H},DType::Float32);}
  if(cursor!=total)throw std::runtime_error("apa_eb4: workspace layout mismatch");
  auto Q=(spq4::bf16*)q.data_ptr();auto K=(spq4::bf16*)k.data_ptr();auto V=(spq4::bf16*)v.data_ptr();
  auto VI=(const int8_t*)vi.data_ptr();auto VS=(const float*)vs.data_ptr();
  auto PQ=(float*)pq.data_ptr();auto BS=(float*)bulk.data_ptr();auto CM=(float*)cm.data_ptr();auto M=(uint8_t*)mask.data_ptr();
  auto PX=prefix.defined()?(float*)prefix.data_ptr():nullptr;
  auto PM=(float*)pm.data_ptr();auto PL=(float*)pl.data_ptr();auto PA=(float*)pa.data_ptr();
  auto CZ=(float*)cz.data_ptr();auto BL=(float*)blog.data_ptr();auto FL=(uint8_t*)flags.data_ptr();
  auto US=diag?(uint8_t*)used.data_ptr():nullptr;auto SCO=diag?(float*)scores.data_ptr():nullptr;auto LZ=diag?(float*)logz.data_ptr():nullptr;
  auto SNK=sinks?(spq4::bf16*)sinks->data_ptr():nullptr;auto OUT=(spq4::bf16*)out.data_ptr();
  int capacity=(int)codes.shape[2];
  const uint8_t* CODES=(const uint8_t*)codes.data_ptr();const float* SC=(const float*)sc.data_ptr();const float* CB=(const float*)cb.data_ptr();
#if TC_APA_P1_TEST_HOOKS
  NDArray test_state;
  auto test_hook=eb3::early_counter_setup(test_state,q.device,(int)(B*KVH),nc,true,PM,PL,PA,rows,(int)VD);
#endif
  // L2-capacity budgets: EB3's rotation prefetch (codes+norms, eb3::tune::L2_FRACTION) unchanged; the bulk's
  // V-operand prefetch budgeted with eb4::tune::L2_FRACTION against the bytes it would prefetch.
  const int64_t code_bytes=(int64_t)codes.numel(),norm_bytes=(int64_t)sc.numel()*4;
  const int64_t vbf_bytes=(int64_t)B*KVH*S*VD*2,vi8_bytes=(int64_t)B*KVH*S*(VD+4);
  const bool do_rpf=eb3::tune::REG_RPF>=1 && code_bytes+norm_bytes<=(int64_t)(eb3::l2_bytes()*eb3::tune::L2_FRACTION);
  const int64_t budget=(int64_t)(eb3::l2_bytes()*eb4::tune::L2_FRACTION);
  const int xpf=eb4::tune::BULK_XPF==1?(code_bytes+norm_bytes+vbf_bytes<=budget?1:0):eb4::tune::BULK_XPF==2?(code_bytes+norm_bytes+vi8_bytes<=budget?2:0):0;
  // 1. rotation (EB3's registered kernel and prefetch)
  eb3::Prefetch pf;
  if(do_rpf){pf.a=CODES;pf.abytes=code_bytes;pf.b=SC;pf.bbytes=norm_bytes;}
  eb3::rotate_q<128,eb3::tune::ROT_DB><<<dim3((unsigned)(B*KVH),128/eb3::tune::ROT_DB),128>>>(Q,(const float*)rot.data_ptr(),PQ,(int)H,(int)KVH,pf);
  cuda_check_last("apa_eb4_rotate");
  // 2. bulk + statistics + tk8 choose
  unsigned* CTRC=eb4::tune::CHOOSE_SITE==0?eb4::counters((int)(B*KVH)):nullptr;
  {
    const int Hi=(int)H,KVi=(int)KVH,Si=(int)S;
    constexpr bool BCH=eb4::tune::CHOOSE_SITE==0;
    switch(xpf){
      case 1: eb4::bulk_stats<128,1,BCH><<<grid,128>>>(PQ,CODES,SC,CB,BS,CM,Hi,KVi,Si,nc,capacity,scale,V,nullptr,nullptr,(int)VD,CZ,BL,(int)NB,CTRC,FL,SCO,LZ,fraction,force_bf16);break;
      case 2: eb4::bulk_stats<128,2,BCH><<<grid,128>>>(PQ,CODES,SC,CB,BS,CM,Hi,KVi,Si,nc,capacity,scale,nullptr,VI,VS,(int)VD,CZ,BL,(int)NB,CTRC,FL,SCO,LZ,fraction,force_bf16);break;
      default: eb4::bulk_stats<128,0,BCH><<<grid,128>>>(PQ,CODES,SC,CB,BS,CM,Hi,KVi,Si,nc,capacity,scale,nullptr,nullptr,nullptr,(int)VD,CZ,BL,(int)NB,CTRC,FL,SCO,LZ,fraction,force_bf16);break;
    }
    cuda_check_last("apa_eb4_bulk_stats");
  }
  // 3. select + mixed-V PV + fused merge
  unsigned* CTR=eb3::counters((int)(B*KVH));
  const uint8_t* INJ=inject?(const uint8_t*)inject->data_ptr():nullptr;
  // select's CTA-start prefetch: SEL_VPF, except that int8 rows already prefetched by the bulk (xpf 2) are skipped
  // when SEL_VPF_SKIP_XPF is set (tuning).
  const int svpf=(eb4::tune::SEL_VPF==1 && xpf==2 && eb4::tune::SEL_VPF_SKIP_XPF)?2:eb4::tune::SEL_VPF;
  {
    const int Hi=(int)H,KVi=(int)KVH,Si=(int)S;
    constexpr bool SCH=eb4::tune::CHOOSE_SITE==1;
    auto launch=[&](auto gt,auto dg){
      constexpr int GT=decltype(gt)::value;constexpr bool DG=decltype(dg)::value;
#if TC_APA_P1_TEST_HOOKS
      eb3::early_counter_check(test_hook,(const void*)eb4::select_mixed<128,128,GT,DG,SCH>);
#endif
      eb4::select_mixed<128,128,GT,DG,SCH><<<grid,128>>>(Q,K,V,VI,VS,FL,US,CZ,BL,SCH?FL:nullptr,SCH?SCO:nullptr,SCH?LZ:nullptr,INJ,fraction,force_bf16,svpf,
        BS,CM,M,PX,PM,PL,PA,Hi,KVi,Si,nc,pk,scale,delta,CTR,SNK,OUT
#if TC_APA_P1_TEST_HOOKS
        ,test_hook
#endif
      );};
    auto byG=[&](auto dg){
      switch(G){
        case 2: launch(std::integral_constant<int,2>{},dg);break;
        case 4: launch(std::integral_constant<int,4>{},dg);break;
        case 8: launch(std::integral_constant<int,8>{},dg);break;
        default: throw std::runtime_error("apa_eb4: G range");
      }
    };
    if(diag)byG(std::true_type{});else byG(std::false_type{});
    cuda_check_last("apa_eb4_select_mixed");
  }
  int64_t ctr_bytes=0,ctrc_bytes=0;eb3::counters(0,&ctr_bytes);if(CTRC)eb4::counters(0,&ctrc_bytes);
  NDArray foot({7},DType::Int64,Device::from_string("cpu"));
  {auto* f=(int64_t*)foot.data_ptr();f[0]=total;f[1]=B*H*VD*2;f[2]=ctr_bytes;f[3]=ctrc_bytes;f[4]=3;f[5]=do_rpf;f[6]=xpf*10+svpf;}
#if TC_APA_P1_TEST_HOOKS
  eb3::early_counter_finish(test_hook,CTR,(int)(B*KVH));   // R5: resets the merge slot on a timeout
#endif
  operand_device.restore();
  return {out,flags,used,scores,logz,cm,cz,blog,mask,bulk,prefix,pq,foot};
}

// Static resource census of the EB4 kernels (cudaFuncGetAttributes + occupancy API, NVIDIA). Diagnostic only.
std::vector<std::tuple<std::string,int,int,int,int,int>> apa_eb4_resources(){
  std::vector<std::tuple<std::string,int,int,int,int,int>> out;
  auto add=[&](const char* name,const void* fn){
    cudaFuncAttributes a{};int blocks=0;
    if(cudaFuncGetAttributes(&a,fn)!=cudaSuccess)throw std::runtime_error("apa_eb4_resources: attributes");
    if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,fn,128,0)!=cudaSuccess)throw std::runtime_error("apa_eb4_resources: occupancy");
    out.emplace_back(name,a.numRegs,(int)a.sharedSizeBytes,(int)a.localSizeBytes,a.maxThreadsPerBlock,blocks);
  };
  add("eb3 bulk_swap<128,BULK_MINB,V prefetch> (EB3 registered)",(const void*)eb3::bulk_swap<128,eb3::tune::BULK_MINB,true,false>);
  constexpr bool BCH=eb4::tune::CHOOSE_SITE==0;constexpr bool SCH=!BCH;
  add("eb4 bulk_stats<128,0>",(const void*)eb4::bulk_stats<128,0,BCH>);
  add("eb4 bulk_stats<128,1>",(const void*)eb4::bulk_stats<128,1,BCH>);
  add("eb4 bulk_stats<128,2>",(const void*)eb4::bulk_stats<128,2,BCH>);
  add("eb3 select_pv G2 (EB3 registered)",(const void*)eb3::select_pv<128,128,2,false,eb3::tune::SEL_MINB,false,true,true,false>);
  add("eb3 select_pv G4 (EB3 registered)",(const void*)eb3::select_pv<128,128,4,false,eb3::tune::SEL_MINB,false,true,true,false>);
  add("eb4 select_mixed G2",(const void*)eb4::select_mixed<128,128,2,false,SCH>);
  add("eb4 select_mixed G4",(const void*)eb4::select_mixed<128,128,4,false,SCH>);
  return out;
}
// APA_EB4_END tk8
