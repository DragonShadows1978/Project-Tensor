// CPU-only actual-source lifecycle/error trace; never links or executes real CUDA work.
// Prior art: CPython 3.12 finalization/GIL rules (Python Software Foundation, 2023);
// C++17 noexcept/RAII (ISO, 2017); CUDA 12.6 event query/error contracts (NVIDIA, 2024).
// Taken: those contracts. Ours: pending-entry callback race and one-shot query-error scenarios.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <cstdio>
namespace py=pybind11;
static cudaError_t slot=cudaSuccess;
static bool fail_query=false;
extern "C" cudaError_t cudaGetDevice(int* d) { *d=0; return cudaSuccess; }
extern "C" cudaError_t cudaSetDevice(int) { return cudaSuccess; }
extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
extern "C" cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e,unsigned) { *e=reinterpret_cast<cudaEvent_t>(16); return cudaSuccess; }
extern "C" cudaError_t cudaEventRecord(cudaEvent_t,cudaStream_t) { return cudaSuccess; }
extern "C" cudaError_t cudaEventDestroy(cudaEvent_t) { return cudaSuccess; }
extern "C" cudaError_t cudaEventQuery(cudaEvent_t) { if(fail_query){fail_query=false;slot=cudaErrorInvalidValue;return slot;}return cudaSuccess; }
extern "C" cudaError_t cudaGetLastError() { auto e=slot;slot=cudaSuccess;return e; }
extern "C" cudaError_t cudaPeekAtLastError() { return slot; }
extern "C" const char* cudaGetErrorName(cudaError_t e) { return e==cudaSuccess?"cudaSuccess":"cudaErrorInvalidValue"; }
extern "C" const char* cudaGetErrorString(cudaError_t e) { return e==cudaSuccess?"success":"invalid argument"; }
#include "dlpack_operand.cpp"
static PyObject* callback=nullptr;
static DLManagedTensor manager{};
static void producer_delete(DLManagedTensor*) { PyObject* r=PyObject_CallNoArgs(callback);if(!r)throw py::error_already_set();Py_DECREF(r); }
PYBIND11_MODULE(_p1t_rev5_trace,m) {
 tcs::init_pending_state();
 m.def("queue",[](py::object f){callback=f.ptr();Py_INCREF(callback);manager.deleter=producer_delete;tcs::PendingEntry e;e.event=reinterpret_cast<cudaEvent_t>(16);e.device=0;e.bytes=4;tcs::ManagedOwner o;o.legacy=&manager;e.owners.push_back(std::move(o));auto& st=tcs::pending();st.q.push_back(std::move(e));tcs::note_push(st,st.q.back());});
 m.def("drain",&tcs::drain_completed);
 m.def("synchronize",&tcs::synchronize_and_release_all);
 m.def("stats",[]{return tcs::pending_stats(false);});
 m.def("fail_query",[]{fail_query=true;});
 m.def("slot",[]{return int(slot);});
}
