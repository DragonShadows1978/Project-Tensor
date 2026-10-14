// Prior art: ELF LD_PRELOAD interposition (FIX4/FIX7, 2026); CUDA 12.6 event return contracts (NVIDIA, 2024).
// Probe infrastructure only: fail one selected diagnostic event call against the actual built product.
#include <cuda_runtime_api.h>
#include <atomic>
#include <dlfcn.h>
static std::atomic<int> site{0};
extern "C" void rev8_arm(int n){site.store(n);}
static bool take(int n){int expected=n;return site.compare_exchange_strong(expected,0);}
extern "C" cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e,unsigned int flags){
 if(take(1))return cudaErrorMemoryAllocation;
 using F=cudaError_t(*)(cudaEvent_t*,unsigned int);static F f=(F)dlsym(RTLD_NEXT,"cudaEventCreateWithFlags");return f(e,flags);
}
extern "C" cudaError_t cudaEventRecord(cudaEvent_t e,cudaStream_t s){
 if(take(2))return cudaErrorInvalidValue;
 using F=cudaError_t(*)(cudaEvent_t,cudaStream_t);static F f=(F)dlsym(RTLD_NEXT,"cudaEventRecord");return f(e,s);
}
extern "C" cudaError_t cudaEventQuery(cudaEvent_t e){
 if(take(3))return cudaErrorInvalidValue;
 using F=cudaError_t(*)(cudaEvent_t);static F f=(F)dlsym(RTLD_NEXT,"cudaEventQuery");return f(e);
}
extern "C" cudaError_t cudaEventDestroy(cudaEvent_t e){
 if(take(4))return cudaErrorInvalidValue;
 using F=cudaError_t(*)(cudaEvent_t);static F f=(F)dlsym(RTLD_NEXT,"cudaEventDestroy");return f(e);
}
