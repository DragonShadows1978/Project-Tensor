// APA_SPO2_D1_BEGIN launcher
#include "baseline_dump.cuh"
#include "merge.cuh"
// APA_P1F1_BEGIN guards
// Host-side guards shared by every packed launcher (APA-P1-F1, REV1 F02/F07/F11).
#ifndef TC_APA_P1_TEST_HOOKS
#define TC_APA_P1_TEST_HOOKS 0
#endif
#define APA_SPO_STR2(x) #x
#define APA_SPO_STR(x) APA_SPO_STR2(x)
// R2 (P1R-F02): a kernel that reads an operand through a vector type
// (uint2 = 8 B, uint4/float4 = 16 B) needs that operand's base address
// aligned to the vector width; a misaligned vector access faults and leaves
// the CUDA context unusable. Every launcher calls this, before any launch,
// for exactly the operands its dispatched kernels vector-load (census:
// artifacts/apa_p1f1/ALIGNMENT_CENSUS.json). Refusal only: no aligned copy.
// Prior art: CUDA C++ Programming Guide, "Device Memory Accesses" (natural
// alignment of vector types; NVIDIA). Taken: the alignment rule. Ours: the
// host refusal before launch. No new technique.
inline void apa_spo_require_aligned(const char* entry,const char* operand,const NDArray& a,int bytes){
  const uintptr_t address=reinterpret_cast<uintptr_t>(a.data_ptr());
  if(address%(uintptr_t)bytes!=0)
    throw std::invalid_argument(std::string(entry)+": operand '"+operand+"' must be "+std::to_string(bytes)+
      "-byte aligned (its kernel reads it with "+std::to_string(bytes)+"-byte vector loads); base address mod "+
      std::to_string(bytes)+" is "+std::to_string(address%(uintptr_t)bytes));
}
// R11 (P1R-F11): empty extents raise exactly as the dense entry does
// (kernels.cu apa_selective_attention_sp: RuntimeError "dimensions must be
// positive and fit int"). No prior art known to me; plain validation.
inline void apa_spo_require_extents(const char* entry,std::initializer_list<const NDArray*> arrays){
  for(const NDArray* a:arrays)
    for(int64_t n:a->shape)
      if(n<=0 || n>std::numeric_limits<int>::max())
        throw std::runtime_error(std::string(entry)+": dimensions must be positive and fit int");
}
// R7 (P1R-F07): A/B environment knobs. The hooks build (TC_APA_P1_TEST_HOOKS)
// keeps today's behaviour and returns the raw value. The default build has the
// knob compiled out: unset or the default value is accepted, any other value
// raises, naming the variable; it is never silently ignored.
// No prior art known to me beyond ordinary build-time feature flags.
inline const char* apa_spo_knob(const char* name,const char* default_value){
  const char* value=std::getenv(name);
#if TC_APA_P1_TEST_HOOKS
  (void)default_value;
  return value;
#else
  if(value && std::strcmp(value,default_value)!=0)
    throw std::invalid_argument(std::string(name)+"="+value+": this is an A/B test knob, compiled out of the default "
      "build (only a TC_APA_P1_TEST_HOOKS build honours it); unset "+name+" or set it to "+default_value);
  return nullptr;
#endif
}
// APA_P1F1_END guards
// APA_P1F2_BEGIN guards
// R3 (P1R2-F04) and R4 (P1R2-F05) of APA-P1-F2: host refusals raised before
// any allocation or launch, as ValueError (std::invalid_argument) naming the
// operand. R4: every axis of a raw helper's operand must be positive (a zero
// extent reaches a zero-grid launch, a CUDA "invalid configuration" error).
// R3: an extent, capacity or index product that a launcher hands to a native
// int (an int kernel argument, an int row/grid index) must fit int; census in
// artifacts/apa_p1f2/INDEX_CENSUS.json. Refusal only: nothing is clamped,
// widened or rerouted, so no finite result changes (L55).
// Prior art: ISO C++17 [conv.integral] (an out-of-range value converted to a
// signed int is implementation-defined; modular on this toolchain) and the
// CUDA Programming Guide grid-dimension limits; ordinary precondition
// validation. No prior art known to me for this census; ours.
constexpr int64_t kApaSpoIntMax=std::numeric_limits<int>::max();
inline void apa_spo_require_positive_extents(const char* entry,const char* operand,const NDArray& a){
  for(size_t i=0;i<a.shape.size();++i){
    const int64_t n=a.shape[i];
    if(n<=0)
      throw std::invalid_argument(std::string(entry)+": operand '"+operand+"' axis "+std::to_string(i)+" has extent "+
        std::to_string(n)+"; every extent must be positive (refused on the host before any launch)");
    if(n>kApaSpoIntMax)
      throw std::invalid_argument(std::string(entry)+": operand '"+operand+"' axis "+std::to_string(i)+" has extent "+
        std::to_string(n)+", which does not fit the native int index (max "+std::to_string(kApaSpoIntMax)+")");
  }
}
inline void apa_spo_require_int_range(const char* entry,const char* operand,int64_t value){
  if(value<0 || value>kApaSpoIntMax)
    throw std::invalid_argument(std::string(entry)+": "+operand+" = "+std::to_string(value)+
      " does not fit the native int index that receives it (max "+std::to_string(kApaSpoIntMax)+
      "); refused before any allocation or launch");
}
// Saturating int64 product of non-negative factors (INT64_MAX on overflow, -1
// for a negative factor), so a range check never sees a wrapped product.
inline int64_t apa_spo_product(std::initializer_list<int64_t> factors){
  int64_t p=1;
  for(int64_t x:factors){
    if(x<0)return -1;
    if(x!=0 && p>std::numeric_limits<int64_t>::max()/x)return std::numeric_limits<int64_t>::max();
    p*=x;
  }
  return p;
}
// Ceiling division without overflow for a non-negative (possibly saturated) n.
inline int64_t apa_spo_ceil_div(int64_t n,int64_t d){return n<0?n:n/d+(n%d!=0);}
// R3: the attention launchers compute S+part_keys-1 (fixed partition layout,
// int) and nc*256 (row-carry partition width, int) from the key count S; both
// fit int iff S + TC_APA_SPLITK_PART_KEYS - 1 does (2048 >= the 256-key chunk).
inline void apa_spo_require_key_range(const char* entry,int64_t S){
  apa_spo_require_int_range(entry,"S + 2047 (int partition and 256-key chunk arithmetic over the key count)",
    S+(int64_t)TC_APA_SPLITK_PART_KEYS-1);
}
// APA_P1F2_END guards
// APA_P1F3_BEGIN guards
// R3' (P1R3-F03) of APA-P1-F3, host checks only (no device code changes).
// (a) Attention extents: every packed attention entry (packed_run r2/tc, the
// spq4 and spq6 tc launchers, eb2b, eb3, eb4) calls this first, before any
// other check, allocation or launch. An extent of q, k or v above INT_MAX
// (which the launchers narrow to int) raises ValueError (std::invalid_argument)
// naming the entry, the operand, the axis and the extent. A zero or negative
// extent is left to the existing checks (apa_spo_require_extents and the core
// check), which keep F1's RuntimeError for dense parity (F2's D2, accepted).
// (b) A range-checked host product is formed in int64 from the int64 extents,
// before the check and before any narrowing to int (apa_spo_product
// saturates, so the check never sees a wrapped value); census:
// artifacts/apa_p1f3/PRODUCT_CENSUS.json.
// Prior art: ISO C++17 [expr]/4 (signed overflow is undefined) and
// [conv.integral] (narrowing); ordinary checked multiplication and
// precondition validation. No prior art known to me for the census; ours.
struct ApaSpoOperand{const char* name;const NDArray* array;};
inline void apa_spo_require_attention_extents(const char* entry,std::initializer_list<ApaSpoOperand> operands){
  for(const ApaSpoOperand& o:operands)
    for(size_t i=0;i<o.array->shape.size();++i){
      const int64_t n=o.array->shape[i];
      if(n>kApaSpoIntMax)
        throw std::invalid_argument(std::string(entry)+": operand '"+o.name+"' axis "+std::to_string(i)+" has extent "+
          std::to_string(n)+", which does not fit the native int index (max "+std::to_string(kApaSpoIntMax)+
          "); refused before any allocation or launch");
    }
}
inline void apa_spo_require_at_most(const char* entry,const char* operand,int64_t value,int64_t bound){
  if(value<0 || value>bound)
    throw std::invalid_argument(std::string(entry)+": "+operand+" = "+std::to_string(value)+
      " exceeds its bound "+std::to_string(bound)+" (computed in int64 before any narrowing); refused before any "
      "allocation or launch");
}
// APA_P1F3_END guards
// Prior art: validation copied from SP1, split merge from SP1.1 (2026).
// APA-P1-F1 R11 (P1R-F11): every check of apa_spq1_core runs here, before any
// launch; apa_spq1_run calls it before its query rotation. R2: the staged G=1
// path (spo2::stage, d1.cuh:26) reads kq (bulk, unpacked only), k and v as
// uint4 when the row width is a multiple of 16 bytes.
inline void apa_spq1_core_check(const char* entry,
    const NDArray& q,const NDArray& k,const NDArray& kq,const NDArray& v,
    float scale,float delta,bool is_causal,const NDArray* sinks,bool baseline,const std::string& carry_mode,const std::string& dots,bool packed) {
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
  apa_spo_require_key_range(entry,S);   // APA-P1-F2 R3 (P1R2-F04)
  const int cap = (int)(D > VD ? D : VD);
  if (cap > TC_APA_MAXD || B > std::numeric_limits<int>::max() / H / L)
    throw std::runtime_error("apa_sp: dimension cap or launch range exceeded");
  if (sinks && (sinks->ndim() != 1 || sinks->shape[0] != H ||
      sinks->dtype != q.dtype || !sinks->device.is_cuda() || sinks->device.index != q.device.index))
    throw std::runtime_error("apa_sp: sinks must be [H] with matching dtype/device");
  if(L!=1)throw std::runtime_error("apa_sp_d1: requires L=1");
  // R11: FP32 always takes ordered dots (the tensor-core branch exists for
  // 16-bit types only); an unsupported dots value for the dtype now raises.
  if(dots=="tensor" && q.dtype==DType::Float32)
    throw std::runtime_error("apa_sp_d1: dots='tensor' is not implemented for float32 (FP32 always uses ordered dots); pass dots='ordered'");
  int np,pk;apa_int4_fixed_partition_layout((int)S,np,pk);
  if(pk % spo2::C)throw std::runtime_error("apa_sp_d1: partition must align to chunk");
  // R7: TC_APA_SPQ1_REFERENCE (packed r2 grouped-reuse switch) is honoured
  // only in a hooks build; the default build refuses a non-default value.
  if(packed)(void)apa_spo_knob("TC_APA_SPQ1_REFERENCE","0");
  // R2 census, mirroring the dispatch in apa_spq1_core: the spo2 staged
  // kernels run iff not baseline, ordered dots, G == 1 and (W or S <= 2048),
  // with W = cap > 64 (the CAP64 instantiation is the only W=false one).
  const bool staged=!baseline && dots=="ordered" && H/KVH==1 && (cap>64 || S<=2048);
  if(staged){
    const int64_t V=16/dtype_size(q.dtype);
    if(!packed && D%V==0)apa_spo_require_aligned(entry,"kq",kq,16);   // d1.cuh:26 via bulk_kernel stage (d1.cuh:61)
    if(D%V==0)apa_spo_require_aligned(entry,"k",k,16);                 // d1.cuh:26 via select_kernel stage (d1.cuh:145)
    if(VD%V==0)apa_spo_require_aligned(entry,"v",v,16);                // d1.cuh:26 via select_kernel stage (d1.cuh:179)
  }
}
std::tuple<NDArray,NDArray,NDArray,NDArray> apa_spq1_core(
    const NDArray& q,const NDArray& k,const NDArray& kq,const NDArray& v,
    float scale,float delta,bool is_causal,const NDArray* sinks,bool diagnostics,bool baseline,const std::string& carry_mode,const std::string& dots, const NDArray* packed=nullptr, const NDArray* pscale=nullptr, const NDArray* pq=nullptr, const NDArray* cb=nullptr, bool tq=false) {
  apa_spq1_core_check(packed?"apa_spq1 packed r2":"apa_spo2 d1",q,k,kq,v,scale,delta,is_causal,sinks,baseline,carry_mode,dots,packed!=nullptr);
  // Every refusal of this entry ran in apa_spq1_core_check above (R11).
  const int64_t B = q.shape[0], H = q.shape[1], D = q.shape[3];
  const int64_t KVH = k.shape[1], S = k.shape[2], VD = v.shape[3];
  const int cap = (int)(D > VD ? D : VD);
  int np,pk;apa_int4_fixed_partition_layout(S,np,pk);
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
  // Every slice is 16-byte aligned, also covering the FP64 accumulator.
  int64_t RS=(int64_t)rows*S,RP=(int64_t)rows*parts,RC=(int64_t)rows*nc;
  // APA-DS1 amendment A2 (2026-11-03): align every arena view for 16B consumers.
  // Prior art: aligned bump allocation (ISO C++17, 2017) and NVIDIA CUDA
  // Programming Guide natural vector alignment (2024 version unverified;
  // lead to check: CUDA float4 natural alignment). Taken: cursor round-up;
  // ours: apply it to this arena. Device code and arithmetic are unchanged.
  constexpr int64_t kArenaAlignment = 16;
  std::vector<int64_t> sizes={RS,4*RS,(diagnostics||baseline)?4*RS:0,4*RC,4*RS,4*RP,4*RP,4*RP*VD,4*RP*VD*(q.dtype==DType::Float32?2:1)};
  int64_t total=0;for(auto n:sizes){if(n==0)continue;total=(total+kArenaAlignment-1)&~(kArenaAlignment-1);total+=n;} // A3: zero-byte slots; aligned bump allocation (ISO C++17, 2017).
  NDArray arena({total},DType::Uint8,q.device),out({B,H,1,VD},q.dtype,q.device);
  int64_t cursor=0;
  auto view=[&](Shape shape,DType dtype)->NDArray {
    cursor=(cursor+kArenaAlignment-1)&~(kArenaAlignment-1);NDArray x=arena;x.shape=shape;x.dtype=dtype;
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
          // R7: the r1 reference switch is read only in a hooks build; the
          // default build refused a non-default value in apa_spq1_core_check.
#if TC_APA_P1_TEST_HOOKS
          const char* ref=apa_spo_knob("TC_APA_SPQ1_REFERENCE","0");
          bool shared=!(ref && std::strcmp(ref,"1")==0) && (H/KVH==4 || H/KVH==8);
#else
          bool shared=(H/KVH==4 || H/KVH==8);
#endif
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
