// APA-EB2b opt-in launchers. Included after SPQ6 inside namespace tc.
// Prior art: KIVI/Liu et al. (2024), per-token V; ThriftAttention/Sharratt
// (2026), selected high-precision blocks with low-precision remainder;
// Flash-Decoding/Dao et al. (2023), split-K; BLASST/Yuan et al. (2025),
// inherited running-max precision action, without sparsity. Taken: these
// ingredients and SPQ6 variant B; ours: the prescribed tk8 plumbing.
namespace eb2b {
using bf16=__nv_bfloat16;
// KIVI 2024 per-token V precedent. Exact E2a symmetric absmax/127,
// IEEE fp32 division, ties-to-even; store signed int8 bits in Uint8 NDArray.
__global__ void quantize(const bf16* v,int8_t* vi,float* scales,int tokens) {
  int row=blockIdx.x,tid=threadIdx.x;if(row>=tokens)return;
  float x=__bfloat162float(v[(int64_t)row*128+tid]),a=fabsf(x);
  #pragma unroll
  for(int o=16;o;o>>=1)a=fmaxf(a,__shfl_xor_sync(0xffffffffu,a,o));
  __shared__ float m[4];if((tid&31)==0)m[tid>>5]=a;__syncthreads();
  a=fmaxf(fmaxf(m[0],m[1]),fmaxf(m[2],m[3]));
  float scale=__fdiv_rn(a,127.f),safe=scale>0.f?scale:1.f;
  // APA-P1-F1 R1 (P1R-F01): fmaxf drops NaN (IEEE 754-2008 maxNum), so a NaN
  // value element was stored as a finite scale and an int8 code of 0, and the
  // NaN vanished. A token row with any non-finite element now stores a quiet
  // NaN scale, so every dequantized value of that token is NaN, as on the dense
  // path. Rows with only finite elements store the same scale and codes, bit
  // for bit. Prior art: IEEE 754 NaN propagation; the E2a quantizer unchanged.
  const bool nonfinite=__syncthreads_or(!isfinite(x))!=0;
  if(tid==0)scales[row]=nonfinite?__int_as_float(0x7fc00000):scale;
  int q=__float2int_rn(__fdiv_rn(x,safe));
  vi[(int64_t)row*128+tid]=int8_t(max(-127,min(127,q)));
}
// Milakov/Gimelshein (2018) stable normalizer; Flash-Decoding (2023)
// split-K statistics. Every key is present. Separate statistics leave the
// packed TQ4 bulk kernel and its reduction schedule completely unchanged.
__global__ void statistics(const float* bulk,const float* cm,float* cz,float* blog,
                           int S,int NC,int NB) {
  int row=blockIdx.x,c=blockIdx.y,tid=threadIdx.x,lane=tid&31,w=tid>>5;
  int start=c*256+w*64;float m=cm[(int64_t)row*NC+c],z=0.f;
  #pragma unroll
  for(int t=0;t<2;++t){int j=start+lane+32*t;
    if(j<S)z+=__expf(bulk[(int64_t)row*S+j]-m);
  }
  #pragma unroll
  for(int o=16;o;o>>=1)z+=__shfl_xor_sync(0xffffffffu,z,o);
  __shared__ float zs[4];if(lane==0){
    zs[w]=z;if(c*4+w<NB)blog[(int64_t)row*NB+c*4+w]=z>0.f?logf(z)+m:-INFINITY;
  }
  __syncthreads();if(tid==0)cz[(int64_t)row*NC+c]=(zs[0]+zs[1])+(zs[2]+zs[3]);
}
// ThriftAttention/Sharratt (2026): precision varies by block. Here the rule
// is exactly E2a topk_choices, group-max FINAL bulk mass, no refine input.
// Batcher (1968) bitonic sorting network, with (score descending,id ascending)
// comparator for exact stable tie semantics on the computed fp32 scores.
__global__ void choose(const float* cm,const float* cz,const float* blog,
                      uint8_t* flags,float* scores,float* head_logz,
                      int H,int KVH,int NC,int NB,float fraction,bool force_bf16) {
  __shared__ float lz[8],ss[1024];__shared__ int ids[1024];
  int bh=blockIdx.x,tid=threadIdx.x,G=H/KVH,h0=(bh/KVH)*H+(bh%KVH)*G;
  if(tid<G){
    int row=h0+tid;float m=-INFINITY,z=0.f;
    for(int c=0;c<NC;++c)m=fmaxf(m,cm[(int64_t)row*NC+c]);
    for(int c=0;c<NC;++c)z+=cz[(int64_t)row*NC+c]*__expf(cm[(int64_t)row*NC+c]-m);
    lz[tid]=logf(z)+m;if(head_logz)head_logz[row]=lz[tid];
  }
  __syncthreads();int nsort=1;while(nsort<NB)nsort<<=1;
  for(int block=tid;block<nsort;block+=256){
    float score=-INFINITY;
    if(block<NB){
#if APA_EB2B_MUTANT == 4
      int ng=1; // defect: head zero of the KV group only
#else
      int ng=G;
#endif
      for(int g=0;g<ng;++g)score=fmaxf(score,__expf(blog[(int64_t)(h0+g)*NB+block]-lz[g]));
      if(scores)scores[(int64_t)bh*NB+block]=score;
    }
    ss[block]=score;ids[block]=block;
  }
  __syncthreads();
  for(int k=2;k<=nsort;k<<=1)for(int j=k>>1;j;j>>=1){
    for(int i=tid;i<nsort;i+=256){int other=i^j;if(other>i){
      float a=ss[i],b=ss[other];int ia=ids[i],ib=ids[other];
      bool better=a>b || (a==b && ia<ib);
      bool swap=(i&k)?better:!better;
      if(swap){ss[i]=b;ss[other]=a;ids[i]=ib;ids[other]=ia;}
    }}
    __syncthreads();
  }
  int n=int(ceilf(fraction*float(NB)));
#if APA_EB2B_MUTANT == 5
  n=max(0,n-1);
#endif
  for(int rank=tid;rank<nsort;rank+=256)if(ids[rank]<NB)
    flags[(int64_t)bh*NB+ids[rank]]=force_bf16 || rank<n;
}
} // namespace eb2b

namespace eb2b {
// Operand parity diagnostic, no policy decisions. KIVI 2024 per-token V;
// same loader as PV, tested against the independent fp32 E2a arithmetic.
__global__ void values_dump(const bf16* v,const int8_t* vi,const float* vs,
                           const uint8_t* flags,bf16* out,int S,int total) {
  int i=blockIdx.x*128+threadIdx.x;if(i>=total)return;
  int token=i/16,dim=(i%16)*8,bh=token/S,j=token%S;
  *reinterpret_cast<uint4*>(out+(int64_t)i*8)=fetch_value8(v,vi,vs,flags,bh,j,dim,S,128);
}
}
NDArray apa_eb2b_values(const NDArray& v,const NDArray& vi,const NDArray& vs,const NDArray& flags) {
  if(v.ndim()!=4 || v.dtype!=DType::BFloat16 || v.shape[3]!=128 || vi.shape!=v.shape || vi.dtype!=DType::Uint8 || vs.dtype!=DType::Float32 || vs.shape!=Shape({v.shape[0],v.shape[1],v.shape[2]}) || flags.dtype!=DType::Uint8 || flags.shape!=Shape({v.shape[0],v.shape[1],(v.shape[2]+63)/64}))throw std::runtime_error("apa_eb2b_values: geometry/dtype");
  for(const NDArray* a:{&v,&vi,&vs,&flags})if(!a->device.is_cuda() || a->device.index!=v.device.index)throw std::runtime_error("apa_eb2b_values: device");
  // APA-P1-F2 R4 (P1R2-F05): positive extents as ValueError (raw helper);
  // R3 (P1R2-F04): total = numel/8 and the kernel's int index
  // i = blockIdx.x*128+threadIdx.x (up to total+127) must fit int.
  apa_spo_require_positive_extents("apa_eb2b_values","v",v);
  apa_spo_require_int_range("apa_eb2b_values","numel(v)/8+127 (int value-group index)",v.numel()/8+127);
  apa_spo_require_extents("apa_eb2b_values",{&v});   // R11
  // R2 (P1R-F02) census: v uint4 and vi uint2 (eb2b_values.cuh:43/:22).
  apa_spo_require_aligned("apa_eb2b_values","v",v,16);
  apa_spo_require_aligned("apa_eb2b_values","vi",vi,8);
  NDArray out(v.shape,v.dtype,v.device);int total=v.numel()/8;
  eb2b::values_dump<<<(total+127)/128,128>>>((eb2b::bf16*)v.data_ptr(),(int8_t*)vi.data_ptr(),(float*)vs.data_ptr(),(uint8_t*)flags.data_ptr(),(eb2b::bf16*)out.data_ptr(),v.shape[2],total);
  cuda_check_last("apa_eb2b_values_dump");return out;
}

// Standard native validation/buffer wrappers (Project-Tensor 2026) and
// pybind11/Jakob 2015. No new algorithm in the shape/device plumbing.
std::pair<NDArray,NDArray> apa_eb2b_quantize(const NDArray& v) {
  // APA-P1-F2 R4 (P1R2-F05): positive extents as ValueError (raw helper); the
  // numel()==0 refusal below stays as it was for any other rank.
  if(v.ndim()==3 || v.ndim()==4)apa_spo_require_positive_extents("apa_eb2b_quantize","v",v);
  if(!v.device.is_cuda() || v.dtype!=DType::BFloat16 || (v.ndim()!=3 && v.ndim()!=4) || v.shape.back()!=128 || v.numel()==0 || v.numel()/128>std::numeric_limits<int>::max())
    throw std::runtime_error("apa_eb2b_quantize: CUDA bf16 [...,128] rank3/4 required");
  NDArray vi(v.shape,DType::Uint8,v.device);Shape s=v.shape;s.pop_back();
  NDArray scales(s,DType::Float32,v.device);
  eb2b::quantize<<<v.numel()/128,128>>>((eb2b::bf16*)v.data_ptr(),(int8_t*)vi.data_ptr(),(float*)scales.data_ptr(),v.numel()/128);
  cuda_check_last("apa_eb2b_quantize");return {vi,scales};
}

// APA-P1-F1 R4 (P1R-F04): delta and carry are arguments, validated as EB3's
// (delta finite and >= 0; carry "partition" or "row"). The EB2b select kernel
// (spq4::select_pv<128,128,DG,true,PR>) takes the partition width PK and delta
// as arguments: carry "partition" uses the fixed SP1.1 partition layout
// (apa_int4_fixed_partition_layout), carry "row" uses one whole-row partition
// (NC*256), exactly the two modes the kernel implements; nothing else is
// accepted. No launch-site literal remains (the old 8.1f / NC*256 pair is now
// the explicit call delta=8.1, carry="row"). Prior art: BLASST (Yuan et al.
// 2025) running maximum and SP1 row prefix, inherited; no new technique.
std::vector<NDArray> apa_eb2b_run(const NDArray& q,const NDArray& k,const NDArray& v,
  const NDArray& vi,const NDArray& vs,const NDArray& codes,const NDArray& sc,
  const NDArray& rot,const NDArray& cb,float scale,float fraction,float delta,const std::string& carry_mode,
  bool force_bf16,bool diag,bool profile) {
  // APA-P1-F3 R3' (P1R3-F03): q/k/v extents above INT_MAX raise ValueError naming this entry and the operand, first.
  apa_spo_require_attention_extents("apa_eb2b",{{"q",&q},{"k",&k},{"v",&v}});
  diag=diag||profile;
  if(carry_mode!="partition" && carry_mode!="row")throw std::runtime_error("carry must be partition or row");
  const char* enabled=std::getenv("TC_APA_SP");
  if(!enabled || std::strcmp(enabled,"1"))throw std::runtime_error("apa_eb2b: TC_APA_SP=1 required");
  if(!std::isfinite(delta) || delta<0.f)throw std::runtime_error("apa_eb2b: delta must be finite nonnegative");
  if(q.ndim()!=4 || k.ndim()!=4 || v.ndim()!=4 || q.shape[2]!=1 || q.shape[3]!=128 || v.shape[3]!=128 || k.shape[3]!=128)
    throw std::runtime_error("apa_eb2b: rank4 D=VD=128 decode only");
  apa_spo_require_extents("apa_eb2b",{&q,&k,&v});   // R11: empty extents raise as the dense entry
  // APA-P1-F3 R3' (P1R3-F03): B*H was formed as a signed int product of the
  // narrowed extents (2^16 * 2^16 is undefined in ISO C++17 and wrapped to 0
  // on this toolchain, passing the check); it is now formed in int64 from the
  // int64 extents, before the check and before the narrowing below, and its
  // bound (rows*4 must fit int, as before) refuses with a ValueError naming
  // the operand. Census: artifacts/apa_p1f3/PRODUCT_CENSUS.json.
  apa_spo_require_at_most("apa_eb2b","B*H (q batch * q heads; the int row count, times 4, must fit int)",
    apa_spo_product({q.shape[0],q.shape[1]}),kApaSpoIntMax/4);
  int B=q.shape[0],H=q.shape[1],KVH=k.shape[1],S=k.shape[2];
  if(B<1 || H<1 || KVH<1 || H%KVH || H/KVH<2 || H/KVH>8 || S<1 || S>65536 || k.shape[0]!=B || v.shape!=k.shape)
    throw std::runtime_error("apa_eb2b: geometry/GQA/range mismatch");
  if(q.dtype!=DType::BFloat16 || k.dtype!=q.dtype || v.dtype!=q.dtype || vi.dtype!=DType::Uint8 || vi.shape!=v.shape || vs.dtype!=DType::Float32 || vs.shape!=Shape({B,KVH,S}))
    throw std::runtime_error("apa_eb2b: bf16 q/k/v, Uint8 int8 bits, fp32 per-token scales required");
  for(const NDArray* a:{&q,&k,&v,&vi,&vs,&codes,&sc,&rot,&cb})
    if(!a->device.is_cuda() || a->device.index!=q.device.index)throw std::runtime_error("apa_eb2b: one CUDA device required");
  if(!std::isfinite(scale) || scale<=0.f || !std::isfinite(fraction) || fraction<=0.f || fraction>1.f)
    throw std::runtime_error("apa_eb2b: scale/fraction range");
  if(cb.shape!=Shape({16}) || cb.dtype!=DType::Float32)throw std::runtime_error("apa_eb2b: codebook");
  apa_spq1_buffer_check(k,codes,sc);
  int np,pk;apa_int4_fixed_partition_layout(S,np,pk);
  if(pk%spq4::C)throw std::runtime_error("apa_eb2b: partition must align to chunk");
  // R2 (P1R-F02) census: codes uint4 (spq6.cuh:130, D=128), q uint2 / k uint4
  // (spq4 select_pv :220/:232), v uint4 and vi uint2 (fetch_value8,
  // eb2b_values.cuh:43/:22). The rotation is read by spq1::rotate (scalar).
  apa_spo_require_aligned("apa_eb2b","codes",codes,16);
  apa_spo_require_aligned("apa_eb2b","q",q,8);
  apa_spo_require_aligned("apa_eb2b","k",k,16);
  apa_spo_require_aligned("apa_eb2b","v",v,16);
  apa_spo_require_aligned("apa_eb2b","vi",vi,8);
  NDArray pq=apa_spq1_rotate(q,rot,KVH,true);
  int NC=(S+255)/256,NB=(S+63)/64,rows=B*H;
  if(carry_mode=="row")pk=NC*spq4::C;
  int64_t RS=(int64_t)rows*S,RC=(int64_t)rows*NC,RB=(int64_t)rows*NB,BF=(int64_t)B*KVH*NB;
  // Flash-Decoding 2023 split-K arena, inherited SPQ6 layout. Additional
  // statistics/flags are ephemeral and never substitute reconstructed K.
  // APA-DS1 amendment A2 (2026-11-03): align every arena view for 16B consumers.
  // Prior art: aligned bump allocation (ISO C++17, 2017) and NVIDIA CUDA
  // Programming Guide natural vector alignment (2024 version unverified;
  // lead to check: CUDA float4 natural alignment). Taken: cursor round-up;
  // ours: apply it to this arena. Device code and arithmetic are unchanged.
  constexpr int64_t kArenaAlignment = 16;
  std::vector<int64_t> sizes={RS,4*RS,diag?4*RS:0,4*RC,4*RC,4*RC,4*RC*128,4*RC,4*RB,BF,diag?4*BF:0,diag?4*rows:0,(diag||profile)?8LL*B*KVH*NC*6:0};
  int64_t total=0;for(auto n:sizes){if(n==0)continue;total=(total+kArenaAlignment-1)&~(kArenaAlignment-1);total+=n;} // A3: zero-byte slots; aligned bump allocation (ISO C++17, 2017).
  NDArray arena({total},DType::Uint8,q.device),out(q.shape,q.dtype,q.device);int64_t cursor=0;
  auto view=[&](Shape shape,DType dtype){cursor=(cursor+kArenaAlignment-1)&~(kArenaAlignment-1);NDArray x=arena;x.shape=shape;x.dtype=dtype;x.offset=cursor/dtype_size(dtype);cursor+=x.numel()*dtype_size(dtype);return x;};
  NDArray mask=view({B,H,1,S},DType::Uint8),bulk=view({B,H,1,S},DType::Float32),prefix;
  if(diag)prefix=view({B,H,1,S},DType::Float32);
  NDArray cm=view({B,H,NC},DType::Float32),pm=view({RC},DType::Float32),pl=view({RC},DType::Float32),pa=view({RC*128},DType::Float32);
  NDArray cz=view({B,H,NC},DType::Float32),blog=view({B,H,NB},DType::Float32),flags=view({B,KVH,NB},DType::Uint8),scores,logz,ticks;
  if(diag){scores=view({B,KVH,NB},DType::Float32);logz=view({B,H},DType::Float32);}
  if(diag||profile)ticks=view({B,KVH,NC,6},DType::Int64);
  if(cursor!=total)throw std::runtime_error("apa_eb2b: arena mismatch");   // R11: exact, as every other arena
  dim3 grid(B*KVH,NC);auto Q=(spq4::bf16*)q.data_ptr();auto K=(spq4::bf16*)k.data_ptr();auto V=(spq4::bf16*)v.data_ptr();
  auto BS=(float*)bulk.data_ptr();auto CM=(float*)cm.data_ptr();auto CZ=(float*)cz.data_ptr();auto BL=(float*)blog.data_ptr();auto F=(uint8_t*)flags.data_ptr();
  spq6::bulk_mma_tq<128,2><<<grid,128>>>((float*)pq.data_ptr(),(uint8_t*)codes.data_ptr(),(float*)sc.data_ptr(),(float*)cb.data_ptr(),BS,CM,H,KVH,S,NC,codes.shape[2],scale);
  cuda_check_last("apa_eb2b_bulk_variant_B");
  eb2b::statistics<<<dim3(rows,NC),128>>>(BS,CM,CZ,BL,S,NC,NB);
  cuda_check_last("apa_eb2b_statistics");
  eb2b::choose<<<B*KVH,256>>>(CM,CZ,BL,F,diag?(float*)scores.data_ptr():nullptr,diag?(float*)logz.data_ptr():nullptr,H,KVH,NC,NB,fraction,force_bf16);
  cuda_check_last("apa_eb2b_choose");
  auto sel=[&](auto dg,auto pr){constexpr bool DG=decltype(dg)::value,PR=decltype(pr)::value;
    spq4::select_pv<128,128,DG,true,PR><<<grid,128>>>(Q,K,V,BS,CM,(uint8_t*)mask.data_ptr(),diag?(float*)prefix.data_ptr():nullptr,(float*)pm.data_ptr(),(float*)pl.data_ptr(),(float*)pa.data_ptr(),H,KVH,S,NC,pk,scale,delta,(int8_t*)vi.data_ptr(),(float*)vs.data_ptr(),F,(diag||profile)?(unsigned long long*)ticks.data_ptr():nullptr);};
  if(profile)sel(std::true_type{},std::true_type{});
  else if(diag)sel(std::true_type{},std::false_type{});
  else sel(std::false_type{},std::false_type{});
  cuda_check_last("apa_eb2b_select_pv");
  spo2::merge_kernel<spq4::bf16><<<rows,128>>>((float*)pm.data_ptr(),(float*)pl.data_ptr(),(float*)pa.data_ptr(),nullptr,(spq4::bf16*)out.data_ptr(),H,128,NC);
  cuda_check_last("apa_eb2b_merge");
  if(!diag && !profile)return {out};
  return {out,mask,bulk,prefix,flags,scores,cm,cz,blog,logz,ticks};
}
// APA-P1-F2 R6 (P1R2-F07): tc/ops.h now declares this exact signature (with
// delta and carry), so a native caller that includes ops.h links to this one
// tc::apa_eb2b_run, and the spo2 binding calls it directly. F1's global
// bridge (apa_p1f1_eb2b_run), which existed only because ops.h was read-only
// in F1, is removed. ISO C++17 declarations/linkage; no prior art claimed.
