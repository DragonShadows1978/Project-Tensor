// PASS rules in release_trace_cpu.py. Includes the sources under review; NEVER calls real CUDA.
// Prior art: CUDA12.6 checked-return/error-slot semantics (NVIDIA2024), DLPack (DMLC2017),
// CPython3.12 lifecycle (PSF2023), SQLite OOM injection. Taken: contracts/fault injection;
// ours: independent multi-device and transient-report/OOM scenarios. Replacements trace host logic only.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <memory>
namespace py=pybind11;
static thread_local cudaError_t slot=cudaSuccess;
static int query_fail=0,sync_fail=0,device_fail=0,restore_fail=0,record_fail=0;
static thread_local int current_device=0;
static std::vector<int> syncs;
extern "C" cudaError_t cudaGetDevice(int* d){if(device_fail){--device_fail;slot=cudaErrorInvalidValue;return slot;}*d=current_device;return cudaSuccess;}
extern "C" cudaError_t cudaSetDevice(int d){if(restore_fail&&d==1){--restore_fail;slot=cudaErrorInvalidDevice;return slot;}current_device=d;return cudaSuccess;}
extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t){syncs.push_back(current_device);if(sync_fail){--sync_fail;slot=cudaErrorInvalidValue;return slot;}return cudaSuccess;}
extern "C" cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e,unsigned){*e=reinterpret_cast<cudaEvent_t>(16);return cudaSuccess;}
extern "C" cudaError_t cudaEventRecord(cudaEvent_t,cudaStream_t){if(record_fail){--record_fail;slot=cudaErrorInvalidValue;return slot;}return cudaSuccess;}
extern "C" cudaError_t cudaEventDestroy(cudaEvent_t){return cudaSuccess;}
extern "C" cudaError_t cudaEventQuery(cudaEvent_t){if(query_fail){--query_fail;slot=cudaErrorInvalidValue;return slot;}return cudaSuccess;}
extern "C" cudaError_t cudaGetLastError(){auto e=slot;slot=cudaSuccess;return e;}
extern "C" cudaError_t cudaPeekAtLastError(){return slot;}
extern "C" const char* cudaGetErrorName(cudaError_t e){return e==cudaSuccess?"cudaSuccess":(e==cudaErrorInvalidDevice?"cudaErrorInvalidDevice":"cudaErrorInvalidValue");}
extern "C" const char* cudaGetErrorString(cudaError_t e){return e==cudaSuccess?"success":"injected failure";}
#include "dlpack_operand.cpp"
struct Held {DLManagedTensor m{};py::object callback;};
static auto& held = *new std::vector<std::unique_ptr<Held>>();
static void delete_manager(DLManagedTensor* m){auto* h=static_cast<Held*>(m->manager_ctx);if(!h->callback.is_none())h->callback();}
PYBIND11_MODULE(_p1t_rev6_trace,m){
 tcs::init_pending_state();
 m.def("queue",[](py::object f,int device){auto h=std::make_unique<Held>();h->callback=f;h->m.manager_ctx=h.get();h->m.deleter=delete_manager;tcs::PendingEntry e;e.event=reinterpret_cast<cudaEvent_t>(16);e.device=device;e.bytes=4;tcs::ManagedOwner o;o.legacy=&h->m;e.owners.push_back(std::move(o));held.push_back(std::move(h));auto& st=tcs::pending();st.q.push_back(std::move(e));tcs::note_push(st,st.q.back());},py::arg("callback")=py::none(),py::arg("device")=0);
 m.def("drain",&tcs::drain_completed);m.def("synchronize",&tcs::synchronize_and_release_all);
 m.def("close",&tcs::close_admission_and_wait);m.def("stats",[]{return tcs::pending_stats(false);});
 m.def("inject",[](const std::string& s){if(s=="query")query_fail=1;else if(s=="sync")sync_fail=1;else if(s=="device")device_fail=1;else if(s=="record")record_fail=1;else if(s=="restore")restore_fail=1;else if(s=="message")tcs::test_hooks::arm(tcs::test_hooks::kAllocErrorMessage);else if(s=="prior")slot=cudaErrorInvalidValue;else throw std::runtime_error("unknown injection");});
 m.def("set_device",[](int d){current_device=d;});m.def("syncs",[]{return syncs;});m.def("slot",[]{return int(slot);});
 // Deliberate keepalive: no static py::object destructor may run after interpreter finalization.
 held.reserve(16);
}
