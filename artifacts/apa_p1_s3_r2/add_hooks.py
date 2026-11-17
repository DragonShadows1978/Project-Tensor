from pathlib import Path
p=Path('tensor_cuda/src/apa_spo/eb3.cuh');s=p.read_text()
needle='namespace eb3 {\nusing bf16=__nv_bfloat16;'
assert s.count(needle)==1
s=s.replace(needle,'''#ifndef TC_APA_P1_TEST_HOOKS
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
struct EarlyCounterHook {unsigned* state=nullptr;unsigned long long ticks=0;};
__device__ __forceinline__ bool early_counter_wait(EarlyCounterHook hook,int c){
  if(!hook.state || c!=1)return true;
  __shared__ unsigned timed_out;
  if(threadIdx.x==0){
    timed_out=0;const auto start=clock64();
    while(atomicAdd(hook.state,0u)==0u){
      if(clock64()-start>=hook.ticks){timed_out=1;atomicExch(hook.state+1,1u);break;}
    }
  }
  __syncthreads();
  return timed_out==0;
}
__device__ __forceinline__ void early_counter_done(EarlyCounterHook hook,int c){
  if(!hook.state || c!=0)return;
  // Every writer has finished ALL possible merge/output stores before the
  // leader releases chunk 1. Also called on the non-merging early exit.
  __threadfence();__syncthreads();
  if(threadIdx.x==0)atomicExch(hook.state,1u);
}
// APA_P1_EARLY_DEVICE_END
inline bool early_counter_enabled(){
  const char* flag=std::getenv("APA_P1_EARLY_COUNTER");return flag && std::strcmp(flag,"1")==0;
}
inline EarlyCounterHook early_counter_setup(NDArray& state,const Device& device,
    int bh,int nc,bool eligible,float* pm,float* pl,float* pa,int rows,int vd){
  if(!early_counter_enabled())return {};
  if(bh!=1 || nc!=2 || !eligible)throw std::runtime_error("APA P1 early-counter requires one bh, NC=2, registered fused path");
  state=NDArray({8},DType::Uint8,device);int khz=0;
  if(cudaDeviceGetAttribute(&khz,cudaDevAttrClockRate,device.index)!=cudaSuccess || khz<=0)throw std::runtime_error("APA P1 clock rate");
  // Poison all partials: chunk 0 overwrites its own; chunk 1 stays poisoned
  // until the completion signal. 0xffffffff is an explicit fp32 NaN sentinel.
  if(cudaMemsetAsync(state.data_ptr(),0,8,0)!=cudaSuccess ||
     cudaMemsetAsync(pm,0xff,(size_t)rows*nc*4,0)!=cudaSuccess ||
     cudaMemsetAsync(pl,0xff,(size_t)rows*nc*4,0)!=cudaSuccess ||
     cudaMemsetAsync(pa,0xff,(size_t)rows*nc*vd*4,0)!=cudaSuccess)throw std::runtime_error("APA P1 early-counter poison");
  constexpr unsigned long long poll_milliseconds=2000;
  return {(unsigned*)state.data_ptr(),(unsigned long long)khz*poll_milliseconds};
}
inline void early_counter_check(EarlyCounterHook hook,const void* kernel){
  if(!hook.state)return;
  int blocks=0,sms=0;
  if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,128,0)!=cudaSuccess ||
     cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,0 /* replaced below by current device */)!=cudaSuccess)
    throw std::runtime_error("APA P1 early-counter occupancy");
  int device=0;
  if(cudaGetDevice(&device)!=cudaSuccess || cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device)!=cudaSuccess || blocks*sms<2)
    throw std::runtime_error("APA P1 early-counter needs two concurrently resident CTAs");
}
inline void early_counter_finish(EarlyCounterHook hook){
  if(!hook.state)return;
  unsigned state[2]={0,0};
  if(cudaDeviceSynchronize()!=cudaSuccess || cudaMemcpy(state,hook.state,sizeof(state),cudaMemcpyDeviceToHost)!=cudaSuccess)throw std::runtime_error("APA P1 early-counter completion");
  if(state[1])throw std::runtime_error("APA P1 early-counter poll timed out (FAIL, never a mutant kill)");
  if(state[0]!=1)throw std::runtime_error("APA P1 early-counter missing completion signal");
}
#endif
using bf16=__nv_bfloat16;''')
# eliminate unnecessary device-0 attribute query in helper
s=s.replace('if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,128,0)!=cudaSuccess ||\n     cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,0 /* replaced below by current device */)!=cudaSuccess)','if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,128,0)!=cudaSuccess)')
s=s.replace(' unsigned* ctr,const bf16* sinks,bf16* out,unsigned long long* tl){',''' unsigned* ctr,const bf16* sinks,bf16* out,unsigned long long* tl
#if TC_APA_P1_TEST_HOOKS
 ,EarlyCounterHook test_hook
#endif
 ){''')
needle='  Stamp st;tl_begin<TL>(st);\n  int b=bh/KVH,h0=(bh%KVH)*G'
# Must only select_pv, bulk has no matching second line
assert s.count(needle)==1
s=s.replace(needle,'''#if TC_APA_P1_TEST_HOOKS
  // Hold chunk 1 BEFORE any pm/pl/pa partial store (survey S3).
  if(!early_counter_wait(test_hook,c))return;
#endif
'''+needle)
s=s.replace('  if(!last_s){tl_end<TL>(tl,(int64_t)bh*NC+c,st);return;}','''  if(!last_s){
#if TC_APA_P1_TEST_HOOKS
    early_counter_done(test_hook,c);
#endif
    tl_end<TL>(tl,(int64_t)bh*NC+c,st);return;
  }''')
needle='  tl_end<TL>(tl,(int64_t)bh*NC+c,st);\n  }\n}\n\n// ---- persistent'
assert s.count(needle)==1
s=s.replace(needle,'''#if TC_APA_P1_TEST_HOOKS
  early_counter_done(test_hook,c);
#endif
'''+needle)
needle='  // L2-capacity budget for cross-kernel prefetch (tuning: L2_FRACTION).'
s=s.replace(needle,'''#if TC_APA_P1_TEST_HOOKS
  NDArray test_state;
  auto test_hook=eb3::early_counter_setup(test_state,q.device,(int)(B*KVH),nc,
    variant==0 && VA.sel==1 && VA.fuse,PM,PL,PA,rows,(int)VD);
#endif
'''+needle)
needle='      eb3::select_pv<128,128,GT,DG,MB,BG,VP,FU,ON,KP><<<grid,128>>>(Q,K,V,BS,CM,M,PX,PM,PL,PA,Hi,KVi,Si,nc,pk,scale,delta,CTR,SNK,OUT,ON?TLS:nullptr);};'
assert s.count(needle)==1
s=s.replace(needle,'''#if TC_APA_P1_TEST_HOOKS
      eb3::early_counter_check(test_hook,(const void*)eb3::select_pv<128,128,GT,DG,MB,BG,VP,FU,ON,KP>);
#endif
      eb3::select_pv<128,128,GT,DG,MB,BG,VP,FU,ON,KP><<<grid,128>>>(Q,K,V,BS,CM,M,PX,PM,PL,PA,Hi,KVi,Si,nc,pk,scale,delta,CTR,SNK,OUT,ON?TLS:nullptr
#if TC_APA_P1_TEST_HOOKS
        ,test_hook
#endif
      );};''')
needle='  operand_device.restore();\n  return {out,pq,mask,bulk,prefix,tlb,tls,foot};'
s=s.replace(needle,'''#if TC_APA_P1_TEST_HOOKS
  eb3::early_counter_finish(test_hook);
#endif
  operand_device.restore();
  return {out,pq,mask,bulk,prefix,tlb,tls,foot};''')
p.write_text(s)
p=Path('tensor_cuda/src/apa_spo/eb4.cuh');s=p.read_text()
needle=' int H,int KVH,int S,int NC,int PK,float scale,float delta,unsigned* ctr,const bf16* sinks,bf16* out){'
assert s.count(needle)==1
s=s.replace(needle,''' int H,int KVH,int S,int NC,int PK,float scale,float delta,unsigned* ctr,const bf16* sinks,bf16* out
#if TC_APA_P1_TEST_HOOKS
 ,eb3::EarlyCounterHook test_hook
#endif
 ){''')
needle='  int b=bh/KVH,h0=(bh%KVH)*G,tid=threadIdx.x,lane=tid&31,warp=tid>>5,r=lane>>2,tig=lane&3;\n  int start=c*C,count=min(C,S-start);'
assert s.count(needle)==1
s=s.replace(needle,'''#if TC_APA_P1_TEST_HOOKS
  // Survey S3: hold chunk 1 before ALL partial stores, not after pm/pl.
  if(!eb3::early_counter_wait(test_hook,c))return;
#endif
'''+needle)
s=s.replace('  if(!last_s)return;','''  if(!last_s){
#if TC_APA_P1_TEST_HOOKS
    eb3::early_counter_done(test_hook,c);
#endif
    return;
  }''')
needle='  }\n}\n\n// ---- persistent choose tickets'
assert s.count(needle)==1
s=s.replace(needle,'''  }
#if TC_APA_P1_TEST_HOOKS
  eb3::early_counter_done(test_hook,c);
#endif
}

// ---- persistent choose tickets''')
needle='  // L2-capacity budgets: EB3'
s=s.replace(needle,'''#if TC_APA_P1_TEST_HOOKS
  NDArray test_state;
  auto test_hook=eb3::early_counter_setup(test_state,q.device,(int)(B*KVH),nc,true,PM,PL,PA,rows,(int)VD);
#endif
'''+needle)
needle='      eb4::select_mixed<128,128,GT,DG,SCH><<<grid,128>>>'
s=s.replace(needle,'''#if TC_APA_P1_TEST_HOOKS
      eb3::early_counter_check(test_hook,(const void*)eb4::select_mixed<128,128,GT,DG,SCH>);
#endif
'''+needle)
s=s.replace('BS,CM,M,PX,PM,PL,PA,Hi,KVi,Si,nc,pk,scale,delta,CTR,SNK,OUT);};','''BS,CM,M,PX,PM,PL,PA,Hi,KVi,Si,nc,pk,scale,delta,CTR,SNK,OUT
#if TC_APA_P1_TEST_HOOKS
        ,test_hook
#endif
      );};''')
needle='  operand_device.restore();\n  return {out,flags,used,scores,logz,cm,cz,blog,mask,bulk,prefix,pq,foot};'
s=s.replace(needle,'''#if TC_APA_P1_TEST_HOOKS
  eb3::early_counter_finish(test_hook);
#endif
  operand_device.restore();
  return {out,flags,used,scores,logz,cm,cz,blog,mask,bulk,prefix,pq,foot};''')
p.write_text(s)
# Test binding releases the GIL so CPU host threads can overlap native entry.
p=Path('tensor_cuda/src/bindings.cpp');s=p.read_text();needle='PYBIND11_MODULE(_tensor_cuda_spo2,m) {';assert s.count(needle)==1
s=s.replace(needle,needle+'''
#if defined(TC_APA_P1_TEST_HOOKS) && TC_APA_P1_TEST_HOOKS
  // Test-only entry for survey S3's two-CTA handshake. pybind11 (Jakob 2015)
  // scoped GIL release is taken so tests can also exercise native host threads.
  // No default-build binding or public Python API change.
  m.def("_p1_test_run",[](bool mixed,Tensor& q,Tensor& k,Tensor& v,Tensor& vi,Tensor& vs,
      py::tuple packed,double scale,double delta,double fraction,bool force,int variant){
    if(packed.size()!=4)throw std::runtime_error("P1 test: packed tuple size");
    auto& codes=packed[0].cast<Tensor&>();auto& sc=packed[1].cast<Tensor&>();
    auto& rot=packed[2].cast<Tensor&>();auto& cb=packed[3].cast<Tensor&>();
    std::vector<NDArray> result;
    {
      py::gil_scoped_release release;
      if(mixed)result=tc::apa_eb4_run(q.data(),k.data(),v.data(),vi.data(),vs.data(),codes.data(),sc.data(),rot.data(),cb.data(),scale,delta,fraction,nullptr,"row",force,nullptr,true,true);
      else result=tc::apa_eb3_run(q.data(),k.data(),v.data(),codes.data(),sc.data(),rot.data(),cb.data(),scale,delta,nullptr,"row",true,true,false,variant);
    }
    py::tuple out(result.size());
    for(size_t i=0;i<result.size();++i)out[i]=result[i].defined()?py::cast(Tensor::make(result[i],false)):py::none();
    return out;
  });
#endif
''')
# Reference variant 8 must use diagnostics=False per existing contract, but
# still return its output and diagnostic placeholders for direct test readback.
s=s.replace('nullptr,"row",true,true,false,variant);','nullptr,"row",variant==0,true,false,variant);')
p.write_text(s)
print('S2 hook edits complete; only compiled with TC_APA_P1_TEST_HOOKS=1')
