// APA_SPO2_D1_BEGIN launcher
#include "baseline_dump.cuh"
#include "merge.cuh"
// Prior art: validation copied from SP1, split merge from SP1.1 (2026).
std::tuple<NDArray,NDArray,NDArray,NDArray> apa_spq1_core(
    const NDArray& q,const NDArray& k,const NDArray& kq,const NDArray& v,
    float scale,float delta,bool is_causal,const NDArray* sinks,bool diagnostics,bool baseline,const std::string& carry_mode,const std::string& dots, const NDArray* packed=nullptr, const NDArray* pscale=nullptr, const NDArray* pq=nullptr, const NDArray* cb=nullptr, bool tq=false) {
  // APA_SPO5_BEGIN keywords
  // Prior art: SP1 whole-row prefix (2026), BLASST running maximum (2025).
  if(carry_mode!="partition" && carry_mode!="row")throw std::runtime_error("carry must be partition or row");
  if(dots!="ordered" && dots!="tensor")throw std::runtime_error("dots must be ordered or tensor");
  if(baseline && (carry_mode!="partition" || dots!="ordered"))throw std::runtime_error("baseline requires partition ordered");
  // APA_SPO5_END keywords
  const char* flag = std::getenv("TC_APA_SP");
  if (!flag || std::strcmp(flag, "1") != 0)
    throw std::runtime_error("apa_sp: experimental entry requires TC_APA_SP=1 (default OFF)");
  if (!std::isfinite(scale) || scale <= 0.f || !std::isfinite(delta) || delta < 0.f)
    throw std::runtime_error("apa_sp: scale must be finite positive; delta finite nonnegative");
  if (q.ndim() != 4 || k.ndim() != 4 || kq.ndim() != 4 || v.ndim() != 4)
    throw std::runtime_error("apa_sp: q/k/kq/v must be rank four");
  if (q.dtype != DType::Float32 && q.dtype != DType::Float16 && q.dtype != DType::BFloat16)
    throw std::runtime_error("apa_sp: supports float32, float16, bfloat16 only");
  for (const NDArray* a : {&q, &k, &kq, &v}) {
    if (a->dtype != q.dtype || !a->device.is_cuda() || a->device.index != q.device.index)
      throw std::runtime_error("apa_sp: inputs must share CUDA device and dtype");
    for (int64_t n : a->shape)
      if (n <= 0 || n > std::numeric_limits<int>::max())
        throw std::runtime_error("apa_sp: dimensions must be positive and fit int");
  }
  const int64_t B = q.shape[0], H = q.shape[1], L = q.shape[2], D = q.shape[3];
  const int64_t KVH = k.shape[1], S = k.shape[2], VD = v.shape[3];
  if (k.shape != kq.shape || k.shape[0] != B || k.shape[3] != D ||
      v.shape[0] != B || v.shape[1] != KVH || v.shape[2] != S || H % KVH != 0)
    throw std::runtime_error("apa_sp: shape/GQA mismatch");
  if (is_causal && S < L)
    throw std::runtime_error("apa_sp: bottom-right causal requires S >= L");
  const int cap = (int)(D > VD ? D : VD);
  if (cap > TC_APA_MAXD || B > std::numeric_limits<int>::max() / H / L)
    throw std::runtime_error("apa_sp: dimension cap or launch range exceeded");
  if (sinks && (sinks->ndim() != 1 || sinks->shape[0] != H ||
      sinks->dtype != q.dtype || !sinks->device.is_cuda() || sinks->device.index != q.device.index))
    throw std::runtime_error("apa_sp: sinks must be [H] with matching dtype/device");

  if(L!=1)throw std::runtime_error("apa_sp_d1: requires L=1");
  int np,pk;apa_int4_fixed_partition_layout(S,np,pk);
  if(pk % spo2::C)throw std::runtime_error("apa_sp_d1: partition must align to chunk");
  int nc=((int64_t)S+spo2::C-1)/spo2::C,rows=B*H;
  // APA_SPO5_BEGIN row_carry
  // Reuse unchanged Tier-0 kernel with a whole-row partition; no dot edit.
  if(carry_mode=="row" && APA_SPO5_MUTANT!=1)pk=nc*spo2::C;
  // APA_SPO5_END row_carry
  // NVIDIA CUDA grid folding: same logical (B*KVH,NC) grid, z covers y overflow.
  dim3 chunk_grid(B*KVH,std::min(nc,65535),(nc+65534)/65535);
  int parts=baseline?np:nc;
  // Prior art: caller-owned/arena workspaces, CUDA/cuBLAS workspace APIs
  // (NVIDIA); ordinary aligned bump allocation, no prior art claimed novel.
  // One stream-ordered allocation for all temporaries; output stays separate.
  // Every slice is 8-byte aligned for the wider FP32-output accumulator.
  int64_t RS=(int64_t)rows*S,RP=(int64_t)rows*parts,RC=(int64_t)rows*nc;
  std::vector<int64_t> sizes={RS,4*RS,(diagnostics||baseline)?4*RS:0,4*RC,4*RS,4*RP,4*RP,4*RP*VD,4*RP*VD*(q.dtype==DType::Float32?2:1)};
  int64_t total=0;for(auto n:sizes){total=(total+7)&~int64_t(7);total+=n;}
  NDArray arena({total},DType::Uint8,q.device),out({B,H,1,VD},q.dtype,q.device);
  int64_t cursor=0;
  auto view=[&](Shape shape,DType dtype)->NDArray {
    cursor=(cursor+7)&~int64_t(7);NDArray x=arena;x.shape=shape;x.dtype=dtype;
    x.offset=cursor/dtype_size(dtype);cursor+=x.numel()*dtype_size(dtype);return x;
  };
  NDArray mask=view({B,H,1,S},DType::Uint8),bulk=view({B,H,1,S},DType::Float32),prefix;
  if(diagnostics||baseline)prefix=view({B,H,1,S},DType::Float32);
  NDArray cm=view({RC},DType::Float32),scores=view({B,H,1,S},DType::Float32);
  NDArray pm=view({RP},DType::Float32),pl=view({RP},DType::Float32),pa=view({RP*VD},DType::Float32);
  NDArray wide=view({RP*VD*(q.dtype==DType::Float32?2:1)},DType::Float32);
  if(cursor!=total)throw std::runtime_error("apa_sp_d1: workspace layout mismatch");
  DISPATCH_FLOAT(q.dtype,T,{
    auto launch=[&](auto cap_tag,auto wc,auto dg) {
      constexpr int CAP=decltype(cap_tag)::value;constexpr bool W=decltype(wc)::value,DIAG=decltype(dg)::value;
      auto Q=(T*)q.data_ptr();auto K=(T*)k.data_ptr();auto KQ=(T*)kq.data_ptr();auto V=(T*)v.data_ptr();
      auto M=(uint8_t*)mask.data_ptr();auto BS=(float*)bulk.data_ptr();auto PS=prefix.defined()?(float*)prefix.data_ptr():nullptr;
      auto PM=(float*)pm.data_ptr();auto PL=(float*)pl.data_ptr();auto PA=(float*)pa.data_ptr();
      if(baseline) {
        apa_spo2_baseline_dump_kernel<T,CAP,W,true><<<dim3(rows,np),128>>>(Q,K,KQ,V,M,BS,PS,PM,PL,PA,H,S,D,VD,scale,delta,KVH,H/KVH,np,pk);
      } else {
        // APA_SPQ1_BEGIN packed_dispatch
        // Prior art: D1 staged bulk (2026); only packed path supplies these pointers.
        if(packed){
          // FlashInfer-style grouped reuse; preserve r1 as explicit reference.
          const char* ref=std::getenv("TC_APA_SPQ1_REFERENCE");
          bool shared=!(ref && std::strcmp(ref,"1")==0) && (H/KVH==4 || H/KVH==8);
          if(shared){
            auto run_shared=[&](auto gg,auto dd,auto tt){
              constexpr int G=decltype(gg)::value,DD=decltype(dd)::value;constexpr bool TQ=decltype(tt)::value;
              spq1::bulk_shared<DD,G,TQ><<<chunk_grid,128>>>((float*)pq->data_ptr(),(uint8_t*)packed->data_ptr(),(float*)pscale->data_ptr(),(float*)cb->data_ptr(),BS,(float*)cm.data_ptr(),H,KVH,S,nc,packed->shape[2],scale);
            };
            auto dispatch_group=[&](auto gg){
              if(D==64){if(tq)run_shared(gg,std::integral_constant<int,64>{},std::true_type{});else run_shared(gg,std::integral_constant<int,64>{},std::false_type{});}
              else {if(tq)run_shared(gg,std::integral_constant<int,128>{},std::true_type{});else run_shared(gg,std::integral_constant<int,128>{},std::false_type{});}
            };
            if(H/KVH==4)dispatch_group(std::integral_constant<int,4>{});else dispatch_group(std::integral_constant<int,8>{});
          }else spq1::bulk<CAP,W><<<chunk_grid,128>>>((float*)pq->data_ptr(),(uint8_t*)packed->data_ptr(),(float*)pscale->data_ptr(),(float*)cb->data_ptr(),BS,(float*)cm.data_ptr(),H,KVH,S,D,nc,packed->shape[2],scale,tq);
          cuda_check_last("apa_spq1_packed_bulk");
        }
        // APA_SPQ1_END packed_dispatch
        // APA_SPO5_BEGIN tensor_dispatch
        // FlashInfer (2024/25) GQA head packing; FP32 always stays Tier 0.
        bool tensor_done=false;
        if constexpr(!std::is_same<T,float>::value) {
          if(dots=="tensor") {
            auto tensor_launch=[&](auto mg_tag) {
              constexpr int MG=decltype(mg_tag)::value;
              dim3 tg(B*KVH*((H/KVH+MG-1)/MG),chunk_grid.y,chunk_grid.z);
              spo5::bulk_kernel<T,MG><<<tg,128>>>(Q,KQ,BS,(float*)cm.data_ptr(),H,KVH,S,D,nc,scale);
              cuda_check_last("apa_spo5_tensor_bulk");
              spo5::select_kernel<T,CAP,DIAG,MG><<<tg,128>>>(Q,APA_SPO5_MUTANT==5?KQ:K,V,BS,(float*)cm.data_ptr(),M,PS,PM,PL,PA,H,KVH,S,D,VD,nc,pk,scale,delta);
            };
            if(H/KVH<=16)tensor_launch(std::integral_constant<int,16>{});
            else tensor_launch(std::integral_constant<int,32>{});
            tensor_done=true;
          }
        }
        if(!tensor_done) {
        // APA_SPO5_END tensor_dispatch
        // APA_SPO4_BEGIN dispatch
        // Prior art: per-head split-K scheduling (Flash-Decoding 2023);
        // keep the original near-bandwidth-floor G=1 WCOOP path.
        if((!W && S>2048) || H/KVH>1) {
          dim3 heads_grid(rows,chunk_grid.y,chunk_grid.z);
          if(!packed) spo4::bulk_kernel<T,CAP,W><<<heads_grid,128>>>(Q,KQ,BS,(float*)cm.data_ptr(),H,KVH,S,D,nc,scale);
          cuda_check_last("apa_sp_d1_bulk_parallel");
          spo4::select_kernel<T,CAP,W,DIAG><<<heads_grid,128>>>(Q,K,V,BS,(float*)cm.data_ptr(),M,PS,PM,PL,PA,H,KVH,S,D,VD,nc,pk,scale,delta);
        } else {
        if(!packed) spo2::bulk_kernel<T,CAP,W><<<chunk_grid,128>>>(Q,KQ,BS,(float*)cm.data_ptr(),H,KVH,S,D,nc,scale);
        cuda_check_last("apa_sp_d1_bulk");
        spo2::select_kernel<T,CAP,W,DIAG><<<chunk_grid,128>>>(Q,K,V,BS,(float*)cm.data_ptr(),(float*)scores.data_ptr(),M,PS,PM,PL,PA,wide.data_ptr(),H,KVH,S,D,VD,nc,pk,scale,delta);
        }
        // APA_SPO4_END dispatch
        // APA_SPO5_BEGIN ordered_close
        }
        // APA_SPO5_END ordered_close
      }
      cuda_check_last("apa_sp_d1_select");
      if(baseline)apa_selective_merge_kernel<T,CAP><<<rows,128>>>(PM,PL,PA,sinks?(T*)sinks->data_ptr():nullptr,(T*)out.data_ptr(),H,VD,parts,sinks?1:0);
      else spo2::merge_kernel<T><<<rows,128>>>(PM,PL,PA,sinks?(T*)sinks->data_ptr():nullptr,(T*)out.data_ptr(),H,VD,parts);
    };
    auto caps=[&](auto diag) {
      if(cap<=64)launch(std::integral_constant<int,64>{},std::false_type{},diag);
      else if(cap<=128)launch(std::integral_constant<int,128>{},std::true_type{},diag);
      else if(cap<=256)launch(std::integral_constant<int,256>{},std::true_type{},diag);
      else launch(std::integral_constant<int,512>{},std::true_type{},diag);
    };
    if(diagnostics)caps(std::true_type{});else caps(std::false_type{});
  });
  cuda_check_last("apa_sp_d1_merge");
  return {out,mask,bulk,prefix};
}
// APA_SPO2_D1_END launcher

// APA_SPQ1_BEGIN launcher_include
#include "packed_launcher.cuh"
// APA_SPQ1_END launcher_include
