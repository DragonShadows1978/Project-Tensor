// Prior art: ELF LD_PRELOAD/dlsym interposition (Unix dynamic loader practice),
// allocation high-water counters (standard profiling). SP3 adds a scoped
// observer, no allocation policy or numerical change. No polling thread.
// Tracks successful cudaMalloc/cudaFree only; cuBLAS driver allocations and
// context overhead are captured only in a pre-forward cudaMemGetInfo offset.
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <mutex>
#include <unordered_map>
#include <cstdint>
namespace {
struct State { std::mutex m; std::unordered_map<void*,size_t> sizes; size_t live=0,peak=0; };
State& state() { static State* s=new State; return *s; }
// Python extensions load dependencies RTLD_LOCAL. RTLD_NEXT from a preload
// need not see that scope, so resolve through the pinned toolkit handle.
void* runtime() {
 static void* h=dlopen("/usr/local/cuda-12.6/lib64/libcudart.so",RTLD_NOW|RTLD_LOCAL);
 return h;
}
}
extern "C" cudaError_t cudaMalloc(void** p,size_t n) {
 static auto real=reinterpret_cast<cudaError_t(*)(void**,size_t)>(dlsym(runtime(),"cudaMalloc"));
 if(!real) return cudaErrorUnknown;
 auto rc=real(p,n);
 if(rc==cudaSuccess) { auto& s=state(); std::lock_guard<std::mutex> g(s.m);
   s.sizes[*p]=n; s.live+=n; if(s.live>s.peak)s.peak=s.live; }
 return rc;
}
extern "C" cudaError_t cudaFree(void* p) {
 static auto real=reinterpret_cast<cudaError_t(*)(void*)>(dlsym(runtime(),"cudaFree"));
 if(!real) return cudaErrorUnknown;
 auto rc=real(p);
 if(rc==cudaSuccess) { auto& s=state(); std::lock_guard<std::mutex> g(s.m);
   auto it=s.sizes.find(p); if(it!=s.sizes.end()) {s.live-=it->second;s.sizes.erase(it);} }
 return rc;
}
extern "C" uint64_t apa_sp3_live_bytes() {auto& s=state();std::lock_guard<std::mutex> g(s.m);return s.live;}
extern "C" uint64_t apa_sp3_peak_bytes() {auto& s=state();std::lock_guard<std::mutex> g(s.m);return s.peak;}
extern "C" void apa_sp3_peak_reset() {auto& s=state();std::lock_guard<std::mutex> g(s.m);s.peak=s.live;}
