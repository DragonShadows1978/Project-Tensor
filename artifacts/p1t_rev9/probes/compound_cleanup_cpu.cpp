
#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdlib>
#include <new>
#include <string>
#include <thread>
namespace py = pybind11;
static thread_local cudaError_t slot = cudaSuccess;
static int create_fail = 0, record_fail = 0, query_fail = 0, destroy_fail = 0, sync_fail = 0;
static int query_notready = 0, sync_delay_ms = 0, current_device = 0;
static bool oom_after_create_fail = false;
static bool fail_allocations = false;
static bool oom_after_destroy=false;
void* operator new(std::size_t n) { if (fail_allocations) throw std::bad_alloc(); void* p = std::malloc(n ? n : 1); if (!p) throw std::bad_alloc(); return p; }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
extern "C" cudaError_t cudaGetDevice(int* d) { *d = current_device; return cudaSuccess; }
extern "C" cudaError_t cudaSetDevice(int d) { current_device = d; return cudaSuccess; }
extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t) {
  if (sync_delay_ms) std::this_thread::sleep_for(std::chrono::milliseconds(sync_delay_ms));
  if (sync_fail) { --sync_fail; slot = cudaErrorInvalidValue; return slot; }
  return cudaSuccess;
}
extern "C" cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e, unsigned) {
  if (create_fail) {
    --create_fail; slot = cudaErrorMemoryAllocation;
    if (oom_after_create_fail) { oom_after_create_fail = false; fail_allocations = true; }
    return slot;
  }
  *e = reinterpret_cast<cudaEvent_t>(16); return cudaSuccess;
}
extern "C" cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t) { if (record_fail) { --record_fail; slot = cudaErrorInvalidValue; return slot; } return cudaSuccess; }
extern "C" cudaError_t cudaEventDestroy(cudaEvent_t) { if (destroy_fail) { --destroy_fail; slot = cudaErrorInvalidValue; if(oom_after_destroy) fail_allocations=true; return slot; } return cudaSuccess; }
extern "C" cudaError_t cudaEventQuery(cudaEvent_t) {
  if (query_fail) { --query_fail; slot = cudaErrorInvalidValue; return slot; }
  if (query_notready) { --query_notready; slot = cudaErrorNotReady; return cudaErrorNotReady; }
  return cudaSuccess;
}
extern "C" cudaError_t cudaGetLastError() { auto e = slot; slot = cudaSuccess; return e; }
extern "C" cudaError_t cudaPeekAtLastError() { return slot; }
extern "C" const char* cudaGetErrorName(cudaError_t e) {
  switch (e) {
    case cudaSuccess: return "cudaSuccess";
    case cudaErrorInvalidValue: return "cudaErrorInvalidValue";
    case cudaErrorInvalidDevice: return "cudaErrorInvalidDevice";
    case cudaErrorMemoryAllocation: return "cudaErrorMemoryAllocation";
    case cudaErrorNotReady: return "cudaErrorNotReady";
    case cudaErrorLaunchFailure: return "cudaErrorLaunchFailure";
    default: return "cudaErrorUnknown";
  }
}
extern "C" const char* cudaGetErrorString(cudaError_t e) { return e == cudaSuccess ? "no error" : "injected failure"; }
#include "dlpack_operand.cpp"
// A raising helper of cuda_guard.h under allocation failure: returns (outcome, slot after).
template <typename F> static py::tuple guarded(bool oom, F f) {
  const char* outcome = "returned";
  slot = cudaErrorInvalidValue;
  fail_allocations = oom;
  try { f(); } catch (const std::bad_alloc&) { outcome = "bad_alloc"; } catch (const std::runtime_error&) { outcome = "runtime_error"; }
  fail_allocations = false;
  const int left = static_cast<int>(slot);
  slot = cudaSuccess;
  return py::make_tuple(outcome, left);
}

// Prior art: CUDA12.6 per-thread error slot, SQLite OOM injection; taken: faults; ours: cleanup before report census.
PYBIND11_MODULE(_tc_signal_rev9_host, m) {
 tcs::init_pending_state();
 m.def("compound", [](bool oom) {
  slot=cudaErrorInvalidDevice; destroy_fail=1;oom_after_destroy=oom;
  tcs::test_hooks::arm(tcs::test_hooks::kAllocEventRecycle);
  const char* result="returned";std::string text;
  try { tcs::RetainUntilComplete keep(0, {}); }
  catch(const std::bad_alloc&) { result="bad_alloc"; }
  catch(const std::runtime_error& e) { result="runtime_error";text=e.what(); }
  fail_allocations=false; const int left=slot;
  return py::make_tuple(result,text,left,tcs::pending_stats(false));
 });
 m.def("synchronize",&tcs::synchronize_and_release_all);
 m.def("watchdog", [](bool oom) {
  tcs::exit_device_waits_diagnosed(5.0);create_fail=1;sync_delay_ms=400;
  fail_allocations=oom;
  auto r=tcs::sync_device_legacy(0);fail_allocations=false;
  tcs::exit_device_waits_undiagnosed();return int(r.err);
 });
 m.def("stats", [] {return tcs::pending_stats(false);});
}
