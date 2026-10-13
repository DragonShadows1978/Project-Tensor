// LEAD helper, no GPU calls. Prior art: DLPack consumer deleter ABI (2017+),
// CPython PyGILState_Ensure lifecycle, std::thread ownership (C++11).
// Takes the REAL core export's manager; this file does not replace the core deleter.
#include <pybind11/pybind11.h>
#include <dlpack/dlpack.h>
#include <thread>
namespace py=pybind11;
struct Held {
 DLManagedTensorVersioned* manager=nullptr;
 explicit Held(py::object cap) {
  manager=static_cast<DLManagedTensorVersioned*>(PyCapsule_GetPointer(cap.ptr(),"dltensor_versioned"));
  if(!manager) throw py::error_already_set();
  if(PyCapsule_SetName(cap.ptr(),"used_dltensor_versioned")!=0) {manager=nullptr;throw py::error_already_set();}
 }
 ~Held(){ if(manager && manager->deleter) manager->deleter(manager); }
 void release_worker(){
  auto* moved=manager;manager=nullptr;
  bool completed=false;
  {
   py::gil_scoped_release unlock;
   std::thread worker([&]{if(moved && moved->deleter) moved->deleter(moved);completed=true;});
   worker.join();
  }
  if(!completed) throw std::runtime_error("foreign release did not complete");
 }
};
PYBIND11_MODULE(p1t_rev_release_worker,m){
 py::class_<Held>(m,"Held").def(py::init<py::object>()).def("release_worker",&Held::release_worker);
}
