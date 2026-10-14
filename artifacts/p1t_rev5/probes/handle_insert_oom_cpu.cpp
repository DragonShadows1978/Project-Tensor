// PASS iff successful cuBLAS creation is cleaned up when host cache insertion throws.
// CPU-only actual-source slices, cuBLAS call replacements; no CUDA/cuBLAS execution.
// Prior art: C++17 RAII/exception safety (ISO 2017); cuBLAS12.6 create/destroy ownership (NVIDIA 2024).
// Taken: resource lifetime across exceptions; ours: fault the first cache insertion after settings succeed.
#include <cublas_v2.h>
#include <unordered_map>
#include <sstream>
#include <iostream>
#include <new>
#include <cstdlib>
static bool fail_next=false;
static int creates=0,destroys=0;
void* operator new(std::size_t n){if(fail_next){fail_next=false;throw std::bad_alloc();}void* p=std::malloc(n);if(!p)throw std::bad_alloc();return p;}
void operator delete(void* p) noexcept {std::free(p);}
void operator delete(void* p,std::size_t) noexcept {std::free(p);}
extern "C" cublasStatus_t cublasCreate_v2(cublasHandle_t* h){*h=reinterpret_cast<cublasHandle_t>(static_cast<uintptr_t>(++creates));return CUBLAS_STATUS_SUCCESS;}
extern "C" cublasStatus_t cublasDestroy_v2(cublasHandle_t){++destroys;return CUBLAS_STATUS_SUCCESS;}
extern "C" cublasStatus_t cublasSetStream_v2(cublasHandle_t,cudaStream_t){return CUBLAS_STATUS_SUCCESS;}
extern "C" cublasStatus_t cublasSetMathMode(cublasHandle_t,cublasMath_t){return CUBLAS_STATUS_SUCCESS;}
extern "C" cublasStatus_t cublasSetAtomicsMode(cublasHandle_t,cublasAtomicsMode_t){return CUBLAS_STATUS_SUCCESS;}
extern "C" cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t,cublasPointerMode_t){fail_next=true;return CUBLAS_STATUS_SUCCESS;}
static constexpr cudaStream_t kLegacyStream=0;
static void check_cublas(cublasStatus_t s,const char*) {if(s!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cuBLAS error");}
namespace conv {
cublasHandle_t handle_for(int device) {
  static std::unordered_map<int, cublasHandle_t> handles;
  auto it = handles.find(device);
  if (it != handles.end()) return it->second;
  cublasHandle_t h = nullptr;
  check_cublas(cublasCreate(&h), "cublasCreate");
  // P1T-FIX4 (sweep): a setting that fails no longer leaks the new handle: it is destroyed (on this device, which the
  // caller's DeviceGuard keeps current), the destroy's own status is checked and named, and the call raises. Nothing is
  // cached, so the next call retries from cublasCreate.
  auto settle = [&h](cublasStatus_t st, const char* what) {
    if (st == CUBLAS_STATUS_SUCCESS) return;
    const cublasStatus_t ds = cublasDestroy(h);
    std::ostringstream os;
    os << "tc_signal: " << what << " failed: cuBLAS status " << static_cast<int>(st);
    if (ds != CUBLAS_STATUS_SUCCESS) os << "; destroying the new handle also failed: cuBLAS status " << static_cast<int>(ds);
    throw std::runtime_error(os.str());
  };
  settle(cublasSetStream(h, kLegacyStream), "cublasSetStream(0)");
  settle(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH), "cublasSetMathMode(CUBLAS_DEFAULT_MATH)");
  settle(cublasSetAtomicsMode(h, CUBLAS_ATOMICS_NOT_ALLOWED), "cublasSetAtomicsMode(NOT_ALLOWED)");
  settle(cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode(HOST)");
  handles.emplace(device, h);
  return h;
}
}
namespace lstm {
cublasHandle_t lstm_handle_for(int device) {
  static std::unordered_map<int, cublasHandle_t> handles;
  auto it = handles.find(device);
  if (it != handles.end()) return it->second;
  cublasHandle_t h = nullptr;
  check_cublas(cublasCreate(&h), "cublasCreate");
  // P1T-FIX4 (sweep): a setting that fails no longer leaks the new handle: it is destroyed (on this device, which the
  // caller's DeviceGuard keeps current), the destroy's own status is checked and named, and the call raises. Nothing is
  // cached, so the next call retries from cublasCreate.
  auto settle = [&h](cublasStatus_t st, const char* what) {
    if (st == CUBLAS_STATUS_SUCCESS) return;
    const cublasStatus_t ds = cublasDestroy(h);
    std::ostringstream os;
    os << "tc_signal.lstm: " << what << " failed: cuBLAS status " << static_cast<int>(st);
    if (ds != CUBLAS_STATUS_SUCCESS) os << "; destroying the new handle also failed: cuBLAS status " << static_cast<int>(ds);
    throw std::runtime_error(os.str());
  };
  settle(cublasSetStream(h, kLegacyStream), "cublasSetStream(0)");
  settle(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH), "cublasSetMathMode(CUBLAS_DEFAULT_MATH)");
  settle(cublasSetAtomicsMode(h, CUBLAS_ATOMICS_NOT_ALLOWED), "cublasSetAtomicsMode(NOT_ALLOWED)");
  settle(cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode(HOST)");
  handles.emplace(device, h);
  return h;
}
}
int main(){int failures=0;for(int i=0;i<2;++i){int c=creates,d=destroys;try{if(i==0)conv::handle_for(0);else lstm::lstm_handle_for(0);std::cout<<"missing expected OOM\n";++failures;}catch(const std::bad_alloc&){std::cout<<(i==0?"conv":"lstm")<<" creates="<<(creates-c)<<" destroys="<<(destroys-d)<<"\n";if(creates-c!=destroys-d)++failures;}}return failures?1:0;}
