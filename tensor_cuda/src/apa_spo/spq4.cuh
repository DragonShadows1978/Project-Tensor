// APA_SPQ4_BEGIN gqa_tc
// APA-SPQ4: GQA packed decode on tensor cores. One CTA per (KV head, 256-key
// chunk) serves ALL G query heads of the group: packed codes, refined exact K
// rows and V rows are each read once per KV head.
//
// Prior art (Prior Art Directive):
//  * FlashInfer (Ye et al. 2025, arXiv:2501.01005; blog 2024-02-02): GQA decode
//    packs the G query heads of a group into the M dimension of an MMA so the
//    KV operand is read once. TAKEN: the M = heads packing. Ours: the packed
//    INT4 bulk operand, the APA running-max selection fused around it.
//  * Marlin (Frantar et al. 2024, arXiv:2408.11743): INT4 weights unpacked to
//    16-bit in registers with magic-number integer tricks, fed to mma.sync.
//    TAKEN: the idea of unpacking nibbles straight into the MMA B fragment.
//    The bf16 "0x4300|c" exponent trick (128+c, then subtract 135) is the
//    standard bf16 analogue of Marlin's fp16 "0x6400|c" (1024+c) trick; the
//    values -7..7 stay exact. Ours: sym-7 codes, key-major B, per-key scale
//    applied after the dot.
//  * Flash-Decoding (Dao/Haziza/Massa/Sizov 2023): split-K + merge, inherited
//    (merge kernel unchanged). BLASST (Yuan et al. 2025, arXiv:2512.12087)
//    running-max criterion and APA/SP (Project-Tensor 2026): selection rule
//    unchanged (refine iff bulk_j >= prefix_max - delta, partition/row carry).
//  * NVIDIA PTX ISA mma.sync.m16n8k16 bf16 fragment layout (taken verbatim);
//    ldmatrix is NOT used (padded-stride shared reads are bank-conflict free).
//  * K-slot permutation: the dot over D is invariant to a bijection of the
//    reduction index applied identically to A and B (re-derivation of the
//    standard "permute K to get contiguous B loads" trick, e.g. Marlin's weight
//    pre-shuffling, CUTLASS interleaved layouts). Ours: lane tig owns D/4
//    CONTIGUOUS dims so a quad reads one contiguous 64 B (D128) packed row.
//  * Numerics: Fasi/Higham/Mikaitis/Pranesh 2021 (tensor-core accumulation
//    order differs from ordered fp32), taken as the reason selection flips are
//    gated (G2), not bitwise.
// Precision: bulk tier STORED packed 4-bit + fp32 scale (4.25 bit/elem @D128);
// bulk COMPUTED as bf16 mma (codes-7 exact in bf16, q bf16 exact) with fp32
// accumulate, then fp32 acc*scale_key*softmax_scale (both fp32 multiplies).
// Exact K dots: bf16 mma (bf16 products are exact in fp32; only accumulation
// order differs from r2's ordered FMA). PV: P rounded to bf16, V bf16, fp32
// accumulate; the normalizer l is the fp32 sum of the SAME bf16-rounded P.
#ifndef APA_SPQ4_MUTANT
#define APA_SPQ4_MUTANT 0   // DeMillo et al. 1978 compile-time mutants; 0 in production
#endif
#ifdef TC_APA_EB2B
#include "eb2b_values.cuh"
#endif
namespace spq4 {
constexpr int C=256,NT=128,MAXG=8,PSTRIDE=264;
using bf16=__nv_bfloat16;

// A rows 8..15 (padding of M=16) are literal zero: a1=a3=0.
__device__ __forceinline__ void mma_bf16(float (&c)[4],unsigned a0,unsigned a2,unsigned b0,unsigned b1){
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a0),"r"(0u),"r"(a2),"r"(0u),"r"(b0),"r"(b1));
}
// byte = two nibbles (even dim in the low nibble) -> bf16x2 (lo-7, hi-7), exact.
__device__ __forceinline__ unsigned nib2bf(unsigned byte){
#if APA_SPQ4_MUTANT==7
  unsigned w=0x43004300u|(byte>>4)|((byte&15u)<<16);
#else
  unsigned w=0x43004300u|(byte&15u)|((byte>>4)<<16);
#endif
  const unsigned m=0x43074307u;  // 135 = 128+7 in bf16
  __nv_bfloat162 a=*reinterpret_cast<const __nv_bfloat162*>(&w),b=*reinterpret_cast<const __nv_bfloat162*>(&m);
  __nv_bfloat162 r=__hsub2(a,b);return *reinterpret_cast<unsigned*>(&r);
}
__device__ __forceinline__ float warp_max(float x){
  #pragma unroll
  for(int o=16;o;o>>=1)x=fmaxf(x,__shfl_xor_sync(0xffffffffu,x,o));return x;
}
// Bulk: values[row,j] = ((sum_d q*(code-7)) * scale_j) * softmax_scale; cm = chunk max per head.
template<int D> __global__ void __launch_bounds__(128) bulk_mma(const bf16* q,const uint8_t* packed,
 const float* scales,float* values,float* cm,int H,int KVH,int S,int NC,int capacity,float scale){
  constexpr int DPL=D/4,NS=DPL/4,NW=DPL/8;
  __shared__ float wmax[4][MAXG];
  int bh=blockIdx.x,c=blockIdx.y+gridDim.y*blockIdx.z;if(c>=NC)return;
  int G=H/KVH,b=bh/KVH,kvh=bh%KVH,h0=kvh*G,tid=threadIdx.x,lane=tid&31,warp=tid>>5,r=lane>>2,tig=lane&3;
  int start=c*C,count=min(C,S-start);
  // q A-fragments, dims permuted: lane tig owns dims [tig*DPL, tig*DPL+DPL); step s covers 4 dims.
  uint2 qa[NS];
  #pragma unroll
  for(int s=0;s<NS;++s)qa[s]=make_uint2(0u,0u);
  if(r<G){
#if APA_SPQ4_MUTANT==2
    int qrow=b*H+r*KVH+kvh;   // wrong KV head for the query head (interleaved map)
#else
    int qrow=b*H+h0+r;
#endif
    const uint2* qp=reinterpret_cast<const uint2*>(q+(int64_t)qrow*D+tig*DPL);
    #pragma unroll
    for(int s=0;s<NS;++s)qa[s]=qp[s];
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
#if APA_SPQ4_MUTANT==3
      if(j<count)ks[t][i]=scales[(int64_t)bh*capacity+start+j+1];   // scale from the neighbour key
#else
      if(j<count)ks[t][i]=scales[(int64_t)bh*capacity+start+j];
#endif
    }
  }
  #pragma unroll
  for(int t=0;t<8;++t){
    float acc[4]={0.f,0.f,0.f,0.f};
    #pragma unroll
    for(int s=0;s<NS;++s){
      unsigned x=(w[t][s>>1]>>(16*(s&1)))&0xffffu;
      mma_bf16(acc,qa[s].x,qa[s].y,nib2bf(x&0xffu),nib2bf(x>>8));
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
}
// Select (SP running-max rule) + refined-union exact dots (bf16 mma) + softmax + PV (bf16 mma).
#ifdef TC_APA_EB2B
// Prior art: BLASST/Yuan et al. 2025 running maximum; Flash-Decoding/Dao
// et al. 2023 split-K; KIVI/Liu et al. 2024 per-token V; ThriftAttention/
// Sharratt 2026 block precision. Taken: inherited S1-S3, MMA and merge;
// ours: opt-in value fetch and diagnostic clock64 counters. Default false
// erases the additions from every pre-existing instantiation.
template<int D,int VD,bool DIAG,bool MIXED=false,bool PROFILE=false>
#else
template<int D,int VD,bool DIAG> __global__ void __launch_bounds__(128) select_pv(const bf16* q,const bf16* k,const bf16* v,
#endif
#ifdef TC_APA_EB2B
__global__ void __launch_bounds__(128) select_pv(const bf16* q,const bf16* k,const bf16* v,
#endif
 const float* bulk,const float* cm,uint8_t* mask,float* prefix,float* pm,float* pl,float* pa,
 int H,int KVH,int S,int NC,int PK,float scale,float delta
#ifdef TC_APA_EB2B
 ,const int8_t* vi=nullptr,const float* vs=nullptr,const uint8_t* flags=nullptr,
 unsigned long long* ticks=nullptr
#endif
 ){
  constexpr int DPL=D/4,NS=DPL/4,NU4=DPL/8,NTW=VD/32,U=VD/8;
  __shared__ float scs[MAXG][C];
  __shared__ uint8_t use[MAXG][C],any_[C];
  __shared__ __align__(16) bf16 Ps[MAXG][PSTRIDE];
  __shared__ __align__(16) bf16 Vs[32][VD+8];   // padded stride: bank-conflict-free pair reads; 32-key V tile (23 KB CTA => 4 CTAs/SM), register prefetch of the next tile
  __shared__ float lsum[MAXG];
  int bh=blockIdx.x,c=blockIdx.y+gridDim.y*blockIdx.z;if(c>=NC)return;
  int G=H/KVH,b=bh/KVH,h0=(bh%KVH)*G,tid=threadIdx.x,lane=tid&31,warp=tid>>5,r=lane>>2,tig=lane&3;
  int start=c*C,count=min(C,S-start);
  const uint4* vbase=reinterpret_cast<const uint4*>(v+((int64_t)bh*S+start)*VD);
#ifdef TC_APA_EB2B
  unsigned long long stamp=0,sel_ticks=0,k_ticks=0,v_ticks=0,pv_ticks=0,soft_ticks=0;
  if constexpr(PROFILE) { __syncthreads(); if(tid==0)stamp=clock64(); }
#endif
  // ---- S1: SP selection, warp per head (max-prefix scan, no CTA barriers) ----
  for(int g=warp;g<G;g+=4){
    int64_t row=(int64_t)b*H+h0+g,base=row*S+start;
    float cv=-INFINITY;int first=(start/PK)*PK/C;
#if APA_SPQ4_MUTANT==8
    for(int p=first+lane;p<c-1;p+=32)cv=fmaxf(cv,cm[row*NC+p]);   // omits the preceding chunk
#else
    #pragma unroll 4
    for(int p=first+lane;p<c;p+=32)cv=fmaxf(cv,cm[row*NC+p]);
#endif
    float carry=warp_max(cv);
    #pragma unroll
    for(int t=0;t<8;++t){
      int j=t*32+lane;float x=j<count?bulk[base+j]:-INFINITY,p=x;
      #pragma unroll
      for(int o=1;o<32;o<<=1){float y=__shfl_up_sync(0xffffffffu,p,o);if(lane>=o)p=fmaxf(p,y);}
      p=fmaxf(p,carry);
#if APA_SPQ4_MUTANT==6
      bool refine=j<count && x>p-delta;
#else
      bool refine=j<count && x>=p-delta;
#endif
      carry=__shfl_sync(0xffffffffu,p,31);
      scs[g][j]=x;use[g][j]=refine;
      if(j<count){mask[base+j]=refine;if constexpr(DIAG)prefix[base+j]=p;}
    }
  }
  __syncthreads();
  for(int j=tid;j<C;j+=NT){
    unsigned a=0;
#if APA_SPQ4_MUTANT==5
    for(int g=0;g<G-1;++g)a|=use[g][j];   // refined-union gather misses the last head
#else
    for(int g=0;g<G;++g)a|=use[g][j];
#endif
    any_[j]=a;
  }
  __syncthreads();
  // ---- S2: exact dots, each refined K row of the KV head read once for all G heads ----
#ifdef TC_APA_EB2B
  if constexpr(PROFILE) { if(tid==0){sel_ticks=clock64()-stamp;stamp=clock64();} }
#endif
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
        mma_bf16(acc,qa[s].x,qa[s].y,b0,b1);
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
#ifdef TC_APA_EB2B
  if constexpr(PROFILE) { if(tid==0){k_ticks=clock64()-stamp;stamp=clock64();} }
#endif
  for(int g=warp;g<G;g+=4){
    int64_t row=(int64_t)b*H+h0+g;
    float m=-INFINITY;
    for(int j=lane;j<count;j+=32)m=fmaxf(m,scs[g][j]);
    m=warp_max(m);float sum=0.f;
    for(int j=lane;j<C;j+=32){
      float p=j<count?__expf(scs[g][j]-m):0.f;bf16 pb=__float2bfloat16_rn(p);
      Ps[g][j]=pb;sum+=__bfloat162float(pb);
    }
    #pragma unroll
    for(int o=16;o;o>>=1)sum+=__shfl_xor_sync(0xffffffffu,sum,o);
    if(lane==0){lsum[g]=sum;pm[row*NC+c]=m;}
  }
  __syncthreads();
  if(tid<G){
#if APA_SPQ4_MUTANT==4
    pl[((int64_t)b*H+h0+tid)*NC+c]=lsum[0];   // normalizer from head 0 for every head
#else
    pl[((int64_t)b*H+h0+tid)*NC+c]=lsum[tid];
#endif
  }
  // ---- S4: PV on tensor cores, M = G heads (padded to 16), N = VD, K = keys ----
#ifdef TC_APA_EB2B
  if constexpr(PROFILE) { __syncthreads(); if(tid==0){soft_ticks=clock64()-stamp;stamp=clock64();} }
#endif
  float acc[NTW][4];
  #pragma unroll
  for(int i=0;i<NTW;++i){acc[i][0]=acc[i][1]=acc[i][2]=acc[i][3]=0.f;}
  uint4 nxt[U/4];
  auto fetch=[&](int t0){
    #pragma unroll
    for(int i=0;i<U/4;++i){
      int idx=tid+NT*i,key=t0+idx/U;
#ifdef TC_APA_EB2B
      if constexpr(MIXED){
        // KIVI 2024 per-token V and ThriftAttention 2026 block precision;
        // shared GQA flag, no key omitted. The bf16 branch is the original
        // vector read; int8 is converted before the unchanged shared/MMA.
        if(key>=count){nxt[i]=make_uint4(0u,0u,0u,0u);continue;}
        nxt[i]=eb2b::fetch_value8(v,vi,vs,flags,bh,start+key,(idx%U)*8,S,VD);
      }else
#endif
      nxt[i]=key<count?vbase[(int64_t)key*U+idx%U]:make_uint4(0u,0u,0u,0u);
    }
  };
  fetch(0);
  for(int t0=0;t0<count;t0+=32){
    #pragma unroll
    for(int i=0;i<U/4;++i){int idx=tid+NT*i;*reinterpret_cast<uint4*>(&Vs[idx/U][(idx%U)*8])=nxt[i];}
    __syncthreads();
#ifdef TC_APA_EB2B
    if constexpr(PROFILE) { if(tid==0){v_ticks+=clock64()-stamp;stamp=clock64();} }
#endif
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
        mma_bf16(acc[nt],a0,a2,b0,b1);
      }
    }
    __syncthreads();
#ifdef TC_APA_EB2B
    if constexpr(PROFILE) { if(tid==0){pv_ticks+=clock64()-stamp;stamp=clock64();} }
#endif
  }
#ifdef TC_APA_EB2B
  if constexpr(PROFILE)if(tid==0){
    // NVIDIA clock64 (diagnostic only): CTA phase totals include barriers;
    // prefetch overlaps MMA and is charged to PV after the first fetch.
    auto p=ticks+((int64_t)bh*NC+c)*6;
    p[0]=sel_ticks;p[1]=k_ticks;p[2]=soft_ticks;p[3]=v_ticks;p[4]=pv_ticks;
  }
  if constexpr(MIXED && DIAG)if(tid==0){
    // Unique value operand byte census (not measured DRAM transactions).
    // E2a accounting: 256 B bf16 or 128 B int8 + 4 B scale per token.
    unsigned long long bytes=0;
    for(int j=0;j<count;++j){bool high=eb2b::high_flag(flags,bh,start+j,S);
      bytes+=high?VD*2:VD+4;
    }
    auto p=ticks+((int64_t)bh*NC+c)*6;
    if constexpr(!PROFILE)for(int i=0;i<5;++i)p[i]=0;
    p[5]=bytes;
  }
#endif
#if APA_SPQ4_MUTANT==1
  if(r<MAXG){int head=r%G;   // padded rows (zero) written back over real heads
#else
  if(r<G){int head=r;
#endif
    int64_t pr=((int64_t)b*H+h0+head)*NC+c;
    #pragma unroll
    for(int nt=0;nt<NTW;++nt){
      int d=warp*(VD/4)+nt*8+2*tig;
      *reinterpret_cast<float2*>(&pa[pr*VD+d])=make_float2(acc[nt][0],acc[nt][1]);
    }
  }
}
} // namespace spq4

// Launcher. Reuses SPQ1 buffer validation and the SPO2 merge kernel (unchanged).
std::tuple<NDArray,NDArray,NDArray,NDArray> apa_spq4_tc(const NDArray& q,const NDArray& k,const NDArray& v,
  const NDArray& codes,const NDArray& sc,float scale,float delta,bool causal,const NDArray* sinks,bool diag,const std::string& carry_mode){
  // APA-P1-F3 R3' (P1R3-F03): q/k/v extents above INT_MAX raise ValueError naming this entry and the operand, first.
  apa_spo_require_attention_extents("apa_spq4 tc int4",{{"q",&q},{"k",&k},{"v",&v}});
  if(carry_mode!="partition" && carry_mode!="row")throw std::runtime_error("carry must be partition or row");
  const char* flag=std::getenv("TC_APA_SP");
  if(!flag || std::strcmp(flag,"1")!=0)throw std::runtime_error("apa_sp: experimental entry requires TC_APA_SP=1 (default OFF)");
  if(!std::isfinite(scale) || scale<=0.f || !std::isfinite(delta) || delta<0.f)throw std::runtime_error("apa_sp: scale must be finite positive; delta finite nonnegative");
  if(q.ndim()!=4 || k.ndim()!=4 || v.ndim()!=4)throw std::runtime_error("apa_spq4: q/k/v must be rank four");
  if(q.dtype!=DType::BFloat16 || k.dtype!=DType::BFloat16 || v.dtype!=DType::BFloat16)throw std::runtime_error("apa_spq4: gqa_path='tc' requires bfloat16 q/k/v (BF16 mma)");
  for(const NDArray* a:{&q,&k,&v})if(!a->device.is_cuda() || a->device.index!=q.device.index)throw std::runtime_error("apa_spq4: inputs must share one CUDA device");
  apa_spo_require_extents("apa_spq4",{&q,&k,&v});   // R11: empty extents raise as the dense entry
  const int64_t B=q.shape[0],H=q.shape[1],L=q.shape[2],D=q.shape[3],KVH=k.shape[1],S=k.shape[2],VD=v.shape[3];
  if(L!=1)throw std::runtime_error("apa_spq4: requires L=1");
  if(KVH<=0 || H%KVH!=0 || k.shape[0]!=B || k.shape[3]!=D || v.shape[0]!=B || v.shape[1]!=KVH || v.shape[2]!=S)throw std::runtime_error("apa_spq4: shape/GQA mismatch");
  if(D!=64 && D!=128)throw std::runtime_error("apa_spq4: D must be 64 or 128");
  if(VD!=64 && VD!=128)throw std::runtime_error("apa_spq4: VD must be 64 or 128");
  int G=(int)(H/KVH);
  if(G<2 || G>spq4::MAXG)throw std::runtime_error("apa_spq4: tc path needs 2 <= H/KVH <= 8");
  if(S<1 || B*H>std::numeric_limits<int>::max()/4)throw std::runtime_error("apa_spq4: dimension range");
  apa_spo_require_key_range("apa_spq4 tc int4",S);   // APA-P1-F2 R3 (P1R2-F04)
  if(sinks && (sinks->ndim()!=1 || sinks->shape[0]!=H || sinks->dtype!=q.dtype || !sinks->device.is_cuda() || sinks->device.index!=q.device.index))throw std::runtime_error("apa_sp: sinks must be [H] with matching dtype/device");
  apa_spq1_buffer_check(k,codes,sc);
  // R2 (P1R-F02) census: q uint2 (spq4.cuh:86 bulk_mma, :220 select_pv), codes
  // uint4 at D128 / uint2 at D64 (:99/:100), k uint4 (:232), v uint4 (:168).
  apa_spo_require_aligned("apa_spq4 tc int4","q",q,8);
  apa_spo_require_aligned("apa_spq4 tc int4","codes",codes,D==128?16:8);
  apa_spo_require_aligned("apa_spq4 tc int4","k",k,16);
  apa_spo_require_aligned("apa_spq4 tc int4","v",v,16);
  int np,pk;apa_int4_fixed_partition_layout((int)S,np,pk);
  if(pk%spq4::C)throw std::runtime_error("apa_spq4: partition must align to chunk");
  int nc=(int)((S+spq4::C-1)/spq4::C),rows=(int)(B*H);
  if(carry_mode=="row")pk=nc*spq4::C;
  dim3 grid((unsigned)(B*KVH),std::min(nc,65535),(nc+65534)/65535);
  int64_t RS=(int64_t)rows*S,RC=(int64_t)rows*nc;
  // APA-DS1 amendment A2 (2026-11-03): align every arena view for 16B consumers.
  // Prior art: aligned bump allocation (ISO C++17, 2017) and NVIDIA CUDA
  // Programming Guide natural vector alignment (2024 version unverified;
  // lead to check: CUDA float4 natural alignment). Taken: cursor round-up;
  // ours: apply it to this arena. Device code and arithmetic are unchanged.
  constexpr int64_t kArenaAlignment = 16;
  std::vector<int64_t> sizes={RS,4*RS,diag?4*RS:0,4*RC,4*RC,4*RC,4*RC*VD};
  int64_t total=0;for(auto n:sizes){if(n==0)continue;total=(total+kArenaAlignment-1)&~(kArenaAlignment-1);total+=n;} // A3: zero-byte slots; aligned bump allocation (ISO C++17, 2017).
  NDArray arena({total},DType::Uint8,q.device),out({B,H,1,VD},q.dtype,q.device);
  int64_t cursor=0;
  auto view=[&](Shape shape,DType dtype)->NDArray{
    cursor=(cursor+kArenaAlignment-1)&~(kArenaAlignment-1);NDArray x=arena;x.shape=shape;x.dtype=dtype;
    x.offset=cursor/dtype_size(dtype);cursor+=x.numel()*dtype_size(dtype);return x;};
  NDArray mask=view({B,H,1,S},DType::Uint8),bulk=view({B,H,1,S},DType::Float32),prefix;
  if(diag)prefix=view({B,H,1,S},DType::Float32);
  NDArray cm=view({RC},DType::Float32),pm=view({RC},DType::Float32),pl=view({RC},DType::Float32),pa=view({RC*VD},DType::Float32);
  if(cursor!=total)throw std::runtime_error("apa_spq4: workspace layout mismatch");
  auto Q=(spq4::bf16*)q.data_ptr();auto K=(spq4::bf16*)k.data_ptr();auto V=(spq4::bf16*)v.data_ptr();
  auto BS=(float*)bulk.data_ptr();auto CM=(float*)cm.data_ptr();auto M=(uint8_t*)mask.data_ptr();
  auto PX=prefix.defined()?(float*)prefix.data_ptr():nullptr;
  auto PM=(float*)pm.data_ptr();auto PL=(float*)pl.data_ptr();auto PA=(float*)pa.data_ptr();
  int capacity=(int)codes.shape[2];
  if(D==128)spq4::bulk_mma<128><<<grid,128>>>(Q,(uint8_t*)codes.data_ptr(),(float*)sc.data_ptr(),BS,CM,(int)H,(int)KVH,(int)S,nc,capacity,scale);
  else spq4::bulk_mma<64><<<grid,128>>>(Q,(uint8_t*)codes.data_ptr(),(float*)sc.data_ptr(),BS,CM,(int)H,(int)KVH,(int)S,nc,capacity,scale);
  cuda_check_last("apa_spq4_bulk");
  auto sel=[&](auto dd,auto vv,auto dg){
    constexpr int DD=decltype(dd)::value,VV=decltype(vv)::value;constexpr bool DG=decltype(dg)::value;
    spq4::select_pv<DD,VV,DG><<<grid,128>>>(Q,K,V,BS,CM,M,PX,PM,PL,PA,(int)H,(int)KVH,(int)S,nc,pk,scale,delta);};
  auto pick=[&](auto dg){
    if(D==128){if(VD==128)sel(std::integral_constant<int,128>{},std::integral_constant<int,128>{},dg);else sel(std::integral_constant<int,128>{},std::integral_constant<int,64>{},dg);}
    else{if(VD==128)sel(std::integral_constant<int,64>{},std::integral_constant<int,128>{},dg);else sel(std::integral_constant<int,64>{},std::integral_constant<int,64>{},dg);}};
  if(diag)pick(std::true_type{});else pick(std::false_type{});
  cuda_check_last("apa_spq4_select_pv");
  spo2::merge_kernel<spq4::bf16><<<rows,128>>>(PM,PL,PA,sinks?(spq4::bf16*)sinks->data_ptr():nullptr,(spq4::bf16*)out.data_ptr(),(int)H,(int)VD,nc);
  cuda_check_last("apa_spq4_merge");
  (void)causal;
  return {out,mask,bulk,prefix};
}
// APA_SPQ6_BEGIN include
#include "spq6.cuh"
// APA_SPQ6_END include
#ifdef TC_APA_EB2B
#include "eb2b.cuh"
#endif
// APA_EB3_BEGIN include
// APA-EB3 opt-in latency path (separate entry apa_eb3_run). Prior art and
// scope in eb3.cuh. Without TC_APA_EB3 this block is empty.
#ifdef TC_APA_EB3
#include "eb3.cuh"
#endif
// APA_EB3_END include
// APA_EB4_BEGIN include
// APA-EB4 opt-in tk8 path on EB3 (separate entry apa_eb4_run). Prior art and scope in eb4.cuh.
// Without TC_APA_EB4 this block is empty.
#ifdef TC_APA_EB4
#include "eb4.cuh"
#endif
// APA_EB4_END include
// Dispatcher behind the gqa_path keyword. "r2" (default) is the unchanged SPQ1.1 r2 call.
// APA-SPQ6: fmt="tq4" is now accepted under "tc" (spq6.cuh); fmt="int4" path unchanged.
std::tuple<NDArray,NDArray,NDArray,NDArray> apa_spq4_run(const NDArray& q,const NDArray& k,const NDArray& v,const NDArray& codes,const NDArray& sc,const NDArray& r,const NDArray& cb,float scale,float delta,bool causal,const NDArray* sinks,bool diag,const std::string& fmt,const std::string& carry,const std::string& gqa_path){
  // APA-P1-F3 R3' (P1R3-F03): q/k/v extents above INT_MAX raise ValueError naming this entry and the operand, first.
  apa_spo_require_attention_extents("apa_spq4_run (packed_run)",{{"q",&q},{"k",&k},{"v",&v}});
  if(gqa_path=="r2")return apa_spq1_run(q,k,v,codes,sc,r,cb,scale,delta,causal,sinks,diag,fmt,carry);
  if(gqa_path!="tc")throw std::runtime_error("gqa_path must be r2 or tc");
  if(fmt!="int4" && fmt!="tq4")throw std::runtime_error("gqa_path='tc' supports fmt='int4' or fmt='tq4'");
  if(k.ndim()==4 && q.ndim()==4 && k.shape[1]>0 && q.shape[1]==k.shape[1])  // MHA: keep r2's path
    return apa_spq1_run(q,k,v,codes,sc,r,cb,scale,delta,causal,sinks,diag,fmt,carry);
  apa_spq1_buffer_check(k,codes,sc);
  // R11 (P1R-F11): apa_spq6_tq4_tc validates everything (G range included)
  // and only then rotates the query; it takes the rotation, not a rotated q.
  if(fmt=="tq4")return apa_spq6_tq4_tc(q,k,v,codes,sc,r,cb,scale,delta,causal,sinks,diag,carry);
  return apa_spq4_tc(q,k,v,codes,sc,scale,delta,causal,sinks,diag,carry);
}
// APA_SPQ4_END gqa_tc
