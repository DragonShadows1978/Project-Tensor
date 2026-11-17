// APA_SPQ1_BEGIN packed_launcher
// Prior art: D1/SPO2 validation and arena (2026); Python/pybind tensor tuples
// for caller-owned buffers. Existing public entry forwards unchanged defaults.
std::tuple<NDArray,NDArray,NDArray,NDArray> apa_spo2_run(
 const NDArray& q,const NDArray& k,const NDArray& kq,const NDArray& v,float scale,float delta,bool causal,const NDArray* sinks,bool diag,bool baseline,const std::string& carry,const std::string& dots){
 return apa_spq1_core(q,k,kq,v,scale,delta,causal,sinks,diag,baseline,carry,dots);
}
void apa_spq1_buffer_check(const NDArray& k,const NDArray& codes,const NDArray& sc){
 if(k.ndim()!=4 || !k.device.is_cuda() || (k.dtype!=DType::Float32 && k.dtype!=DType::BFloat16))throw std::runtime_error("packed K requires rank4 CUDA fp32/bf16");
 if(k.shape[3]!=64 && k.shape[3]!=128)throw std::runtime_error("packed D must be 64 or 128");
 if(codes.ndim()!=4 || codes.dtype!=DType::Uint8 || sc.dtype!=DType::Float32 || sc.ndim()!=3 || codes.shape[0]!=k.shape[0] || codes.shape[1]!=k.shape[1] || codes.shape[2]<k.shape[2] || codes.shape[3]!=k.shape[3]/2 || sc.shape!=Shape({codes.shape[0],codes.shape[1],codes.shape[2]}))throw std::runtime_error("packed codes/scales shape or dtype mismatch");
 for(const NDArray* a:{&codes,&sc})if(!a->device.is_cuda() || a->device.index!=k.device.index)throw std::runtime_error("packed device mismatch");
 // APA-P1-F2 R3 (P1R2-F04): every packed kernel takes the capacity as a native
 // int (spq1 bulk/bulk_shared/pack_tq `int capacity`, pack_int4_rows `int dst_s`,
 // spq4 bulk_mma, spq6 bulk_mma_tq, eb3 bulk_copy/bulk_swap, eb4 bulk_stats
 // `int capacity`); a capacity above INT_MAX would narrow (to a negative value at
 // 2^31) and index before the buffers. Refused here, the one check every packed
 // attention and append entry runs before its first allocation or launch.
 // Prior art: std::vector::reserve's max_size() refusal (ISO C++ 1998) as the
 // capacity analogue; ordinary range validation, ours.
 apa_spo_require_int_range("apa_spq1 packed buffers","capacity (codes.shape[2])",codes.shape[2]);
}
NDArray apa_spq1_rotate(const NDArray& q,const NDArray& r,int KVH,bool tq){
 // APA-P1-F2 R4 (P1R2-F05): positive extents before the launch (B=0 or H=0
 // used to reach a zero-grid launch); R3: the grid B*H is a native int row.
 if(q.ndim()==4){
   apa_spo_require_positive_extents("apa_spq1_rotate (packed_rotate)","q",q);
   apa_spo_require_int_range("apa_spq1_rotate (packed_rotate)","B*H (rotate grid rows)",apa_spo_product({q.shape[0],q.shape[1]}));
 }
 if(q.ndim()!=4 || q.shape[2]!=1 || !q.device.is_cuda() || (q.dtype!=DType::Float32 && q.dtype!=DType::BFloat16) || KVH<=0 || q.shape[1]%KVH || (q.shape[3]!=64 && q.shape[3]!=128))throw std::runtime_error("packed query shape/dtype");
 int D=q.shape[3];
 if(r.shape!=Shape({KVH,D,D}) || r.dtype!=DType::Float32 || !r.device.is_cuda() || r.device.index!=q.device.index)throw std::runtime_error("packed rotation shape/device/dtype");
 NDArray out(q.shape,DType::Float32,q.device);
 DISPATCH_FLOAT(q.dtype,T,{spq1::rotate<T><<<q.shape[0]*q.shape[1],128>>>((T*)q.data_ptr(),(float*)r.data_ptr(),(float*)out.data_ptr(),q.shape[1],KVH,D,tq);});
 cuda_check_last("apa_spq1_rotate");return out;
}
// APA_SPQ1_BEGIN prepare_launcher
// Prior art: native tensor wrappers (D1 2026); fixed-row kernel documented above.
std::pair<NDArray,NDArray> apa_spq1_prepare(const NDArray& k,const NDArray& r,int start){
 if(k.ndim()!=4 || !k.device.is_cuda() || (k.dtype!=DType::Float32 && k.dtype!=DType::BFloat16))throw std::runtime_error("prepare K shape/dtype/device");
 // APA-P1-F2 R4 (P1R2-F05): positive extents before the int narrowing below
 // and the launch (B=0 or KV=0 used to reach a zero-grid launch).
 apa_spo_require_positive_extents("apa_spq1_prepare (packed_prepare)","k",k);
 // APA-P1-F3 R3' (P1R3-F03, census row): N = S - start was formed in int before
 // the start < 0 refusal, so a negative start could overflow (undefined, ISO
 // C++17 [expr]/4) before the check; it is now formed in int64 from the int64
 // extent and narrowed only after every check. Same refusals for every
 // defined input. Ordinary checked arithmetic; no prior art claimed.
 const int64_t N64=k.shape[2]-(int64_t)start;
 int B=k.shape[0],H=k.shape[1],S=k.shape[2],D=k.shape[3];
 if((D!=64 && D!=128) || N64<=0 || start<0 || r.shape!=Shape({H,D,D}) || r.dtype!=DType::Float32 || !r.device.is_cuda() || r.device.index!=k.device.index)throw std::runtime_error("prepare geometry/rotation");
 // R3 (P1R2-F04): the prepare grid B*KV*N is an int (host grid and the
 // kernel's int row = blockIdx.x).
 apa_spo_require_int_range("apa_spq1_prepare (packed_prepare)","B*KV*N (prepare grid rows)",apa_spo_product({k.shape[0],k.shape[1],N64}));
 const int N=(int)N64;
 NDArray rot({B,H,N,D},DType::Float32,k.device),norm({B,H,N,1},DType::Float32,k.device);
 DISPATCH_FLOAT(k.dtype,T,{spq1::prepare<T><<<B*H*N,128>>>((T*)k.data_ptr(),(float*)r.data_ptr(),(float*)rot.data_ptr(),(float*)norm.data_ptr(),H,S,D,start);});
 cuda_check_last("apa_spq1_prepare");return {rot,norm};
}
// APA_SPQ1_END prepare_launcher
void apa_spq1_pack(const NDArray& k,NDArray& codes,NDArray& sc,const NDArray& rotated,const NDArray& norms,const NDArray& bnd,int start,bool tq){
 // APA-P1-F2 R4 (P1R2-F05): positive extents of k, codes and scales before
 // anything else (zero leading axes with N > 0 used to reach a zero-grid
 // launch); R3 (P1R2-F04): B*KV and the pack grid are native ints.
 if(k.ndim()==4)apa_spo_require_positive_extents("apa_spq1_pack (packed_append)","k",k);
 if(codes.ndim()==4)apa_spo_require_positive_extents("apa_spq1_pack (packed_append)","codes",codes);
 if(sc.ndim()==3)apa_spo_require_positive_extents("apa_spq1_pack (packed_append)","scales",sc);
 apa_spq1_buffer_check(k,codes,sc);
 apa_spo_require_int_range("apa_spq1_pack (packed_append)","B*KV",apa_spo_product({k.shape[0],k.shape[1]}));
 // APA-P1-F3 R3' (P1R3-F03, census row): N = S - start was formed in int before
 // the start < 0 refusal, so a negative start could overflow (undefined, ISO
 // C++17 [expr]/4) before the check; it is now formed in int64 from the int64
 // extent and narrowed only after every check. Same refusals for every
 // defined input. Ordinary checked arithmetic; no prior art claimed.
 const int64_t N64=k.shape[2]-(int64_t)start;
 int S=k.shape[2],D=k.shape[3],BH=k.shape[0]*k.shape[1];
 if(start<0 || start>S)throw std::runtime_error("packed append range");if(!N64)return;
 const int N=(int)N64;
 apa_spo_require_int_range("apa_spq1_pack (packed_append)",tq?"pack_tq grid blocks (B*KV*N*D/2+127)/128":"pack_int4_rows grid blocks (B*KV*N+3)/4",
   tq?apa_spo_ceil_div(apa_spo_product({(int64_t)BH,(int64_t)N,(int64_t)(D/2)}),128):apa_spo_ceil_div(apa_spo_product({(int64_t)BH,(int64_t)N}),4));
 if(tq){
   if(rotated.shape!=Shape({k.shape[0],k.shape[1],N,D}) || norms.numel()!=int64_t(BH)*N || bnd.numel()!=15)throw std::runtime_error("packed TQ preparation shape");
   for(const NDArray* a:{&rotated,&norms,&bnd})if(a->dtype!=DType::Float32 || !a->device.is_cuda() || a->device.index!=k.device.index)throw std::runtime_error("packed TQ preparation device/dtype");
   spq1::pack_tq<<<(int64_t(BH)*N*(D/2)+127)/128,128>>>((float*)rotated.data_ptr(),(float*)norms.data_ptr(),(float*)bnd.data_ptr(),(uint8_t*)codes.data_ptr(),(float*)sc.data_ptr(),BH,N,D,codes.shape[2],start);
 }else{
   // R1 (P1R-F01): packed-owned copy of apa_int4_pack_rows_kernel that also
   // carries a non-finite key element into the row's stored scale (packed.cuh).
   DISPATCH_FLOAT(k.dtype,T,{spq1::pack_int4_rows<T><<<(int64_t(BH)*N+3)/4,128>>>((T*)k.data_ptr(),(uint8_t*)codes.data_ptr(),(float*)sc.data_ptr(),BH,S,codes.shape[2],start,N,D);});
 }
 cuda_check_last("apa_spq1_pack_rows");
}
std::tuple<NDArray,NDArray,NDArray,NDArray> apa_spq1_run(const NDArray& q,const NDArray& k,const NDArray& v,const NDArray& codes,const NDArray& sc,const NDArray& r,const NDArray& cb,float scale,float delta,bool causal,const NDArray* sinks,bool diag,const std::string& fmt,const std::string& carry){
 // APA-P1-F3 R3' (P1R3-F03): q/k/v extents above INT_MAX raise ValueError naming this entry and the operand, first.
 apa_spo_require_attention_extents("apa_spq1 packed r2 (packed_run)",{{"q",&q},{"k",&k},{"v",&v}});
 if(fmt!="tq4" && fmt!="int4")throw std::runtime_error("fmt must be tq4 or int4");
 apa_spq1_buffer_check(k,codes,sc);
 if(cb.shape!=Shape({16}) || cb.dtype!=DType::Float32 || !cb.device.is_cuda() || cb.device.index!=q.device.index)throw std::runtime_error("packed codebook shape/device/dtype");
 // R11 (P1R-F11): the whole core validation (TC_APA_SP, scale, shapes,
 // alignment, knobs) runs before the query rotation launches.
 apa_spq1_core_check("apa_spq1 packed r2",q,k,k,v,scale,delta,causal,sinks,false,carry,"ordered",true);
 NDArray pq=apa_spq1_rotate(q,r,k.shape[1],fmt=="tq4");
 return apa_spq1_core(q,k,k,v,scale,delta,causal,sinks,diag,false,carry,"ordered",&codes,&sc,&pq,&cb,fmt=="tq4");
}
// APA_SPQ1_END packed_launcher
// APA_SPQ1_BEGIN z_diagnostic
// Prior art: existing symmetric INT4 stats kernel (Project-Tensor 2026).
// Reuse it with its optional bulk cache to expose the actual threshold/dots;
// this diagnostic is never inside the timed comparator. No new selector.
std::pair<NDArray,NDArray> apa_spq1_int4_stats(const NDArray& q,const NDArray& codes,const NDArray& scales,int S,float scale,float zthr){
 // R6 (P1R-F06): rank, dtype, device and every divisor are checked before any
 // shape is read or anything launches; refusals are ValueError. Plain
 // validation, no prior art claimed. (apa_spq1_buffer_check needs an exact-K
 // operand this diagnostic does not take, so its checks are restated here.)
 if(q.ndim()!=4 || codes.ndim()!=4 || scales.ndim()!=3)
   throw std::invalid_argument("apa_spq1_int4_stats: q must be rank 4 [B,H,1,D], codes rank 4 [B,KVH,capacity,D/2], scales rank 3 [B,KVH,capacity]");
 if((q.dtype!=DType::Float32 && q.dtype!=DType::Float16 && q.dtype!=DType::BFloat16) || codes.dtype!=DType::Uint8 || scales.dtype!=DType::Float32)
   throw std::invalid_argument("apa_spq1_int4_stats: q float32/float16/bfloat16, codes uint8, scales float32 required");
 for(const NDArray* a:{&q,&codes,&scales})
   if(!a->device.is_cuda() || a->device.index!=q.device.index)throw std::invalid_argument("apa_spq1_int4_stats: q, codes and scales must share one CUDA device");
 for(const NDArray* a:{&q,&codes,&scales})
   for(int64_t n:a->shape)if(n<=0 || n>std::numeric_limits<int>::max())throw std::invalid_argument("apa_spq1_int4_stats: dimensions must be positive and fit int (KVH > 0)");
 if(q.shape[1]%codes.shape[1]!=0)throw std::invalid_argument("apa_spq1_int4_stats: H must be a multiple of KVH");
 if(codes.shape[0]!=q.shape[0] || codes.shape[3]!=q.shape[3]/2 || scales.shape!=Shape({codes.shape[0],codes.shape[1],codes.shape[2]}))
   throw std::invalid_argument("apa_spq1_int4_stats: codes/scales shape does not match q");
 if(q.shape[0]*q.shape[1]>std::numeric_limits<int>::max())throw std::invalid_argument("apa_spq1_int4_stats: launch range");
 int B=q.shape[0],H=q.shape[1],D=q.shape[3],KVH=codes.shape[1];
 if(q.shape[2]!=1 || (D!=64 && D!=128) || codes.shape[2]<S || S<1)throw std::runtime_error("SPQ1 stats geometry");
 NDArray bulk({B,H,1,S},DType::Float32,q.device),thr({B*H},DType::Float32,q.device);
 DISPATCH_FLOAT(q.dtype,T,{
  auto launch=[&](auto cap,auto w){constexpr int CAP=decltype(cap)::value;constexpr bool W=decltype(w)::value;
   apa_selective_int4_stats_kernel<T,CAP,W><<<B*H,128>>>((T*)q.data_ptr(),(uint8_t*)codes.data_ptr(),(float*)scales.data_ptr(),(float*)thr.data_ptr(),(float*)bulk.data_ptr(),B,H,1,S,D,scale,zthr,0,KVH,H/KVH,D/2,codes.shape[2]);};
  if(D==64)launch(std::integral_constant<int,64>{},std::false_type{});else launch(std::integral_constant<int,128>{},std::true_type{});
 });
 cuda_check_last("apa_spq1_int4_stats");return {bulk,thr};
}
// APA_SPQ1_END z_diagnostic
// APA_SPQ4_BEGIN include
// gqa_path dispatch ("r2" default = unchanged apa_spq1_run; "tc" = tensor-core GQA path).
#include "spq4.cuh"
// APA_SPQ4_END include
