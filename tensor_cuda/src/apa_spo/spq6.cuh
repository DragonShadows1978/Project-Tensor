// APA_SPQ6_BEGIN tq4_tc
// APA-SPQ6: TurboQuant4 packed bulk tier on the SPQ4 tensor-core GQA path (gqa_path="tc", fmt="tq4").
// Included from spq4.cuh (needs spq4:: mma/select/constants); selection, exact-K, softmax, PV and merge
// are the SPQ4 kernels UNCHANGED. Only the bulk stage differs.
//
// Prior art (Prior Art Directive):
//  * TurboQuant (Zandieh et al. 2025, arXiv:2504.19874): rotate keys by a per-head random orthogonal
//    matrix R, quantize each coordinate with a Lloyd-Max scalar codebook (16 centroids), keep the key norm.
//    score = norm * <R q, centroid(code)>. TAKEN: the format and the rotate-the-query identity; the
//    codes/centroids/norms/R come unchanged from SPQ1's workspace.
//  * FlashInfer (Ye et al. 2025, arXiv:2501.01005): G query heads as the M dimension of an MMA (as SPQ4).
//  * Marlin (Frantar et al. 2024, arXiv:2408.11743): dequantize 4-bit codes in registers straight into the
//    mma B fragment. TAKEN: the idea. Marlin's INT4 uses an integer magic-number trick; a Lloyd-Max
//    codebook is a non-affine map, so here the unpack is a 256-entry byte -> bf16x2 shared-memory LUT
//    (QuaRot/QServe/AWQ-style LUT dequant is the generic analogue; no single source verified: unverified,
//    lead to check: "LUT dequantization codebook 4-bit tensor core", "NF4 dequant LUT bitsandbytes").
//  * Split-bf16 operand (q = q_hi + q_lo, two MMAs): the standard 2xBF16 / "bf16x3" emulation trick
//    (Henry et al. 2019 "Leveraging the bfloat16 AI datatype for higher-precision computations",
//    arXiv:1904.06376; Ootomo & Yokota 2022 error-corrected tensor-core GEMM). TAKEN: the hi/lo split.
//    SPQ6 applied it to the query operand only (QSPLIT=2 ablation); APA-SPQ-FXTC1 (this edit) splits BOTH operands:
//    q = q_hi + q_lo, C = C_hi + C_lo (each bf16, 16 mantissa bits total) and keeps q_hi*C_hi + q_hi*C_lo + q_lo*C_hi
//    (the 3-term "bf16x3" form; the q_lo*C_lo term is ~2^-18 relative and only in variant C). Fixes the review
//    finding (apa-tcrev, gpt-6.1-sol: single-bf16 rounding of q and of the centroids each flips near-prefix-max rankings).
//    TAKEN: Henry et al. 2019 split + Ootomo & Yokota 2022 (error-corrected tensor-core GEMM, drop the lo*lo term;
//    they also keep correction terms in a SEPARATE accumulator and add late because tensor-core accumulation rounds
//    toward zero -- NOT adopted here: single fp32 accumulator, measured by the G2/G3 gates instead). Ours: applying it
//    to a Lloyd-Max centroid LUT B operand (hi and lo LUTs of 256 x bf16x2 in shared memory) inside the SPQ4 GQA tile.
//  * Numerics: centroids rounded to BF16 (relative 2^-9) -> bulk scores differ from r2's fp32 FMA by a small
//    deterministic perturbation; selection flips are gated (SPQ6 G2), not bitwise (Fasi et al. 2021 for
//    tensor-core accumulation order as in SPQ4).
// Precision: bulk STORED packed 4-bit + fp32 norm per key (4.25 bit/elem @D128); bulk COMPUTED as bf16 mma:
// A = rotated query (fp32 -> bf16 hi [+ bf16 lo]), B = BF16(centroid[code]), fp32 accumulate, then
// fp32 (acc*norm)*softmax_scale.
#ifndef APA_SPQ6_MUTANT
#define APA_SPQ6_MUTANT 0   // DeMillo et al. 1978 compile-time mutants; 0 in production
#endif
#ifndef APA_SPQ6_QSPLIT
#define APA_SPQ6_QSPLIT 1   // LEGACY (SPQ6) knob, no longer read by the kernel; superseded by the FXTC1 variant below. Kept only because tests/test_apa_spq6_cpu.py pins the line.
#endif
// APA-SPQ-FXTC1 bulk precision variant (template parameter of the kernel; runtime-selectable through the env var
// APA_FXTC1_VARIANT for the A/B harness, compile-time default APA_FXTC1_VARIANT_DEFAULT; the shipped default is B):
//   0 = SPQ6 legacy: q->bf16, C->bf16, 1 mma                        (kept ONLY as the measured baseline)
//   1 = A: q split only:        q_hi*C_hi + q_lo*C_hi                (2 mma)
//   2 = B: q and C split:       q_hi*C_hi + q_hi*C_lo + q_lo*C_hi    (3 mma)   <- shipped candidate
//   3 = C: B plus q_lo*C_lo                                          (4 mma)
#ifndef APA_FXTC1_VARIANT_DEFAULT
#define APA_FXTC1_VARIANT_DEFAULT 2
#endif
// DeMillo et al. 1978 compile-time mutants of the split terms (0 in production). 1 drops the q_lo*C_hi term; 2 swaps q_hi/q_lo;
// 3 reads the centroid-lo LUT from the wrong index; 4 q_lo is the raw query (not the residual); 5 q_lo sign flipped;
// 6 centroid lo is the raw centroid (not the residual).
#ifndef APA_FXTC1_MUTANT
#define APA_FXTC1_MUTANT 0
#endif
namespace spq6 {
using bf16=__nv_bfloat16;
__device__ __forceinline__ unsigned pack_bf2(float a,float b){
  __nv_bfloat162 r=__floats2bfloat162_rn(a,b);return *reinterpret_cast<unsigned*>(&r);
}
__device__ __forceinline__ float bf16_val(float x){return __bfloat162float(__float2bfloat16_rn(x));}
__device__ __forceinline__ float bf2_lo(unsigned w){return __uint_as_float(w<<16);}
__device__ __forceinline__ float bf2_hi(unsigned w){return __uint_as_float(w&0xffff0000u);}
template<int D,int V> __global__ void __launch_bounds__(128) bulk_mma_tq(const float* qr,const uint8_t* packed,
 const float* scales,const float* cb,float* values,float* cm,int H,int KVH,int S,int NC,int capacity,float scale){
  constexpr int DPL=D/4,NS=DPL/4,NW=DPL/8,C=spq4::C,MAXG=spq4::MAXG;
  __shared__ float wmax[4][MAXG];
  constexpr bool QS=V>=1,CS=V>=2,LL=V>=3;
  __shared__ unsigned ctab[256];   // byte (two nibbles, even dim low) -> bf16x2 of the two centroids (hi part)
  __shared__ unsigned ctlo[CS?256:1];   // same, bf16x2 of the residual centroid - bf16(centroid) (lo part)
  int bh=blockIdx.x,c=blockIdx.y+gridDim.y*blockIdx.z;if(c>=NC)return;
  int G=H/KVH,b=bh/KVH,kvh=bh%KVH,h0=kvh*G,tid=threadIdx.x,lane=tid&31,warp=tid>>5,r=lane>>2,tig=lane&3;
  int start=c*C,count=min(C,S-start);
  for(int i=tid;i<256;i+=128){
#if APA_SPQ6_MUTANT==2
    int lo=i&15,hi=i>>4;if(lo==5)lo=6;if(hi==5)hi=6;   // wrong codebook entry (index 5 reads centroid 6)
    ctab[i]=pack_bf2(cb[lo],cb[hi]);
#elif APA_SPQ6_MUTANT==5
    ctab[i]=pack_bf2(cb[i>>4],cb[i&15]);               // nibble order swapped
#else
    ctab[i]=pack_bf2(cb[i&15],cb[i>>4]);
#endif
    if constexpr(CS){
      int ia=i&15,ib=i>>4;
#if APA_FXTC1_MUTANT==3
      ia=(ia+1)&15;ib=(ib+1)&15;                    // lo part from the wrong centroid index (hi LUT stays right)
#endif
      float c0=cb[ia],c1=cb[ib];
#if APA_FXTC1_MUTANT==6
      ctlo[i]=pack_bf2(c0,c1);                      // lo is the raw centroid, not the residual
#else
      ctlo[i]=pack_bf2(c0-bf16_val(c0),c1-bf16_val(c1));   // exact residual (fp32 subtraction of a bf16 prefix is exact)
#endif
    }
  }
  uint2 qh[NS];
  uint2 ql[NS];
  #pragma unroll
  for(int s=0;s<NS;++s){qh[s]=make_uint2(0u,0u);ql[s]=make_uint2(0u,0u);}
  if(r<G){
#if APA_SPQ6_MUTANT==4
    int qrow=b*H+h0+((r^1)<G?(r^1):r);   // head-row swap (pairs 0<->1, 2<->3, ...)
#else
    int qrow=b*H+h0+r;
#endif
    const float4* qp=reinterpret_cast<const float4*>(qr+(int64_t)qrow*D+tig*DPL);
    #pragma unroll
    for(int s=0;s<NS;++s){
      float4 f=qp[s];
      unsigned x=pack_bf2(f.x,f.y),y=pack_bf2(f.z,f.w);qh[s]=make_uint2(x,y);
      if constexpr(QS){
#if APA_FXTC1_MUTANT==4
        ql[s]=make_uint2(pack_bf2(f.x,f.y),pack_bf2(f.z,f.w));   // lo is the raw query, not the residual
#elif APA_FXTC1_MUTANT==5
        ql[s]=make_uint2(pack_bf2(bf2_lo(x)-f.x,bf2_hi(x)-f.y),pack_bf2(bf2_lo(y)-f.z,bf2_hi(y)-f.w));   // sign flipped
#else
        ql[s]=make_uint2(pack_bf2(f.x-bf2_lo(x),f.y-bf2_hi(x)),pack_bf2(f.z-bf2_lo(y),f.w-bf2_hi(y)));
#endif
      }
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
      if(j<count){
#if APA_SPQ6_MUTANT==3
        const float norm=scales[(int64_t)bh*capacity+start+j+1];   // norm from the neighbour key
#else
        const float norm=scales[(int64_t)bh*capacity+start+j];
#endif
        // PF2-F1 (Project-Tensor, 2026) and IEEE 754-2008: reuse the quiet-NaN norm guard on tc.
        ks[t][i]=isfinite(norm)?norm:__int_as_float(0x7fc00000);
      }
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
#if APA_FXTC1_MUTANT==2
      const uint2 qa=ql[s],qb=qh[s];                                              // swapped q_hi / q_lo roles
#else
      const uint2 qa=qh[s],qb=ql[s];
#endif
      spq4::mma_bf16(acc,qa.x,qa.y,b0,b1);                                        // q_hi * C_hi
      if constexpr(QS){
#if APA_FXTC1_MUTANT!=1
        spq4::mma_bf16(acc,qb.x,qb.y,b0,b1);                                      // q_lo * C_hi   (mutant 1 drops it)
#endif
      }
      if constexpr(CS){
        unsigned l0=ctlo[x&0xffu],l1=ctlo[x>>8];
        spq4::mma_bf16(acc,qa.x,qa.y,l0,l1);                                      // q_hi * C_lo
        if constexpr(LL)spq4::mma_bf16(acc,qb.x,qb.y,l0,l1);                      // q_lo * C_lo (variant C only)
      }
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
} // namespace spq6

// Launcher: structurally the SPQ4 launcher (apa_spq4_tc) with the bulk kernel replaced; select_pv and merge reused.
std::tuple<NDArray,NDArray,NDArray,NDArray> apa_spq6_tq4_tc(const NDArray& q,const NDArray& k,const NDArray& v,
  const NDArray& codes,const NDArray& sc,const NDArray& rot,const NDArray& cb,float scale,float delta,bool causal,const NDArray* sinks,bool diag,const std::string& carry_mode){
  // APA-P1-F3 R3' (P1R3-F03): q/k/v extents above INT_MAX raise ValueError naming this entry and the operand, first.
  apa_spo_require_attention_extents("apa_spq6 tc tq4",{{"q",&q},{"k",&k},{"v",&v}});
  if(carry_mode!="partition" && carry_mode!="row")throw std::runtime_error("carry must be partition or row");
  const char* flag=std::getenv("TC_APA_SP");
  if(!flag || std::strcmp(flag,"1")!=0)throw std::runtime_error("apa_sp: experimental entry requires TC_APA_SP=1 (default OFF)");
  if(!std::isfinite(scale) || scale<=0.f || !std::isfinite(delta) || delta<0.f)throw std::runtime_error("apa_sp: scale must be finite positive; delta finite nonnegative");
  if(q.ndim()!=4 || k.ndim()!=4 || v.ndim()!=4)throw std::runtime_error("apa_spq6: q/k/v must be rank four");
  if(q.dtype!=DType::BFloat16 || k.dtype!=DType::BFloat16 || v.dtype!=DType::BFloat16)throw std::runtime_error("apa_spq6: gqa_path='tc' requires bfloat16 q/k/v (BF16 mma)");
  for(const NDArray* a:{&q,&k,&v})if(!a->device.is_cuda() || a->device.index!=q.device.index)throw std::runtime_error("apa_spq6: inputs must share one CUDA device");
  apa_spo_require_extents("apa_spq6",{&q,&k,&v});   // R11: empty extents raise as the dense entry
  const int64_t B=q.shape[0],H=q.shape[1],L=q.shape[2],D=q.shape[3],KVH=k.shape[1],S=k.shape[2],VD=v.shape[3];
  if(L!=1)throw std::runtime_error("apa_spq6: requires L=1");
  if(KVH<=0 || H%KVH!=0 || k.shape[0]!=B || k.shape[3]!=D || v.shape[0]!=B || v.shape[1]!=KVH || v.shape[2]!=S)throw std::runtime_error("apa_spq6: shape/GQA mismatch");
  if(D!=64 && D!=128)throw std::runtime_error("apa_spq6: D must be 64 or 128");
  if(VD!=64 && VD!=128)throw std::runtime_error("apa_spq6: VD must be 64 or 128");
  int G=(int)(H/KVH);
  if(G<2 || G>spq4::MAXG)throw std::runtime_error("apa_spq6: tc path needs 2 <= H/KVH <= 8");
  if(S<1 || B*H>std::numeric_limits<int>::max()/4)throw std::runtime_error("apa_spq6: dimension range");
  apa_spo_require_key_range("apa_spq6 tc tq4",S);   // APA-P1-F2 R3 (P1R2-F04)
  if(cb.shape!=Shape({16}) || cb.dtype!=DType::Float32 || !cb.device.is_cuda() || cb.device.index!=q.device.index)throw std::runtime_error("apa_spq6: codebook shape/device/dtype");
  if(sinks && (sinks->ndim()!=1 || sinks->shape[0]!=H || sinks->dtype!=q.dtype || !sinks->device.is_cuda() || sinks->device.index!=q.device.index))throw std::runtime_error("apa_sp: sinks must be [H] with matching dtype/device");
  apa_spq1_buffer_check(k,codes,sc);
  int np,pk;apa_int4_fixed_partition_layout((int)S,np,pk);
  if(pk%spq4::C)throw std::runtime_error("apa_spq6: partition must align to chunk");
  // R2 (P1R-F02) census: q uint2 (spq4 select_pv :220), codes uint4 at D128 /
  // uint2 at D64 (spq6.cuh:130/:131), k uint4 (spq4.cuh:232), v uint4 (:168).
  // The rotated query (float4 at spq6.cuh:105) is this launcher's own buffer.
  apa_spo_require_aligned("apa_spq6 tc tq4","q",q,8);
  apa_spo_require_aligned("apa_spq6 tc tq4","codes",codes,D==128?16:8);
  apa_spo_require_aligned("apa_spq6 tc tq4","k",k,16);
  apa_spo_require_aligned("apa_spq6 tc tq4","v",v,16);
  // R7 (P1R-F07): APA_FXTC1_VARIANT (bulk precision A/B knob) is honoured only
  // in a hooks build; the default build runs the shipped variant and refuses a
  // non-default value here, before any launch.
  int var=APA_FXTC1_VARIANT_DEFAULT;
#if TC_APA_P1_TEST_HOOKS
  // APA-SPQ-FXTC1 variant: compile-time default, overridable per call by APA_FXTC1_VARIANT in {0,1,2,3} (A/B harness only).
  if(const char* ve=apa_spo_knob("APA_FXTC1_VARIANT",APA_SPO_STR(APA_FXTC1_VARIANT_DEFAULT))){
    if(ve[0]<'0' || ve[0]>'3' || ve[1]!=0)throw std::runtime_error("apa_fxtc1: APA_FXTC1_VARIANT must be one of 0,1,2,3");
    var=ve[0]-'0';
  }
#else
  (void)apa_spo_knob("APA_FXTC1_VARIANT",APA_SPO_STR(APA_FXTC1_VARIANT_DEFAULT));
#endif
  // R11 (P1R-F11): the query rotation is the first launch, after every check.
#if APA_SPQ6_MUTANT==1
  NDArray pq=apa_spq1_rotate(q,rot,(int)KVH,false);   // mutant: query NOT rotated
#else
  NDArray pq=apa_spq1_rotate(q,rot,(int)KVH,true);
#endif
  if(pq.dtype!=DType::Float32 || pq.shape!=q.shape || !pq.device.is_cuda() || pq.device.index!=q.device.index)throw std::runtime_error("apa_spq6: rotated query must be fp32, q-shaped, same device");
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
  if(cursor!=total)throw std::runtime_error("apa_spq6: workspace layout mismatch");
  auto Q=(spq4::bf16*)q.data_ptr();auto K=(spq4::bf16*)k.data_ptr();auto V=(spq4::bf16*)v.data_ptr();
  auto BS=(float*)bulk.data_ptr();auto CM=(float*)cm.data_ptr();auto M=(uint8_t*)mask.data_ptr();
  auto PX=prefix.defined()?(float*)prefix.data_ptr():nullptr;
  auto PM=(float*)pm.data_ptr();auto PL=(float*)pl.data_ptr();auto PA=(float*)pa.data_ptr();
  int capacity=(int)codes.shape[2];
  auto launch_bulk=[&](auto dd,auto vv){
    constexpr int DD=decltype(dd)::value,VV=decltype(vv)::value;
    spq6::bulk_mma_tq<DD,VV><<<grid,128>>>((float*)pq.data_ptr(),(uint8_t*)codes.data_ptr(),(float*)sc.data_ptr(),(float*)cb.data_ptr(),BS,CM,(int)H,(int)KVH,(int)S,nc,capacity,scale);};
  // R7: in the default build var is the compile-time default (no environment
  // read selects it), so only the shipped variant is ever launched. The other
  // variants stay instantiated as unreachable code: Build F r1 removed them and
  // that moved the static shared-memory placement of 96 unrelated kernels
  // (artifacts/apa_p1f1/SASS_COMPARISON.json), so the dispatch is kept as is.
  auto bulkv=[&](auto dd){
    if(var==0)launch_bulk(dd,std::integral_constant<int,0>{});else if(var==1)launch_bulk(dd,std::integral_constant<int,1>{});
    else if(var==2)launch_bulk(dd,std::integral_constant<int,2>{});else launch_bulk(dd,std::integral_constant<int,3>{});};
  if(D==128)bulkv(std::integral_constant<int,128>{});else bulkv(std::integral_constant<int,64>{});
  cuda_check_last("apa_spq6_bulk");
  auto sel=[&](auto dd,auto vv,auto dg){
    constexpr int DD=decltype(dd)::value,VV=decltype(vv)::value;constexpr bool DG=decltype(dg)::value;
    spq4::select_pv<DD,VV,DG><<<grid,128>>>(Q,K,V,BS,CM,M,PX,PM,PL,PA,(int)H,(int)KVH,(int)S,nc,pk,scale,delta);};
  auto pick=[&](auto dg){
    if(D==128){if(VD==128)sel(std::integral_constant<int,128>{},std::integral_constant<int,128>{},dg);else sel(std::integral_constant<int,128>{},std::integral_constant<int,64>{},dg);}
    else{if(VD==128)sel(std::integral_constant<int,64>{},std::integral_constant<int,128>{},dg);else sel(std::integral_constant<int,64>{},std::integral_constant<int,64>{},dg);}};
  if(diag)pick(std::true_type{});else pick(std::false_type{});
  cuda_check_last("apa_spq6_select_pv");
  spo2::merge_kernel<spq4::bf16><<<rows,128>>>(PM,PL,PA,sinks?(spq4::bf16*)sinks->data_ptr():nullptr,(spq4::bf16*)out.data_ptr(),(int)H,(int)VD,nc);
  cuda_check_last("apa_spq6_merge");
  (void)causal;
  return {out,mask,bulk,prefix};
}
// APA_SPQ6_END tq4_tc
