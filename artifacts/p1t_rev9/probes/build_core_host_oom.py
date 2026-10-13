# Prior art: REV7 source-hashed core extraction; SQLite OOM fault injection. Product slices are generated anew.
"""PASS iff current-source core host slices compile; exact spans hashed. No GPU work.
Prior art: REV6 source-slice workflow; CPython3.12/C++17/DLPack contracts. Taken: source extraction;
ours: include actual FIX6 dispatch/raw shutdown/error reporting, rather than old typed test bindings.
"""
from pathlib import Path
import hashlib,json,subprocess,sysconfig
root=Path.cwd();out=Path('/tmp/p1t_rev9_build'); rows=[]
def span(path,a,b):
    s=(root/path).read_text();start=s.index(a);end=s.index(b,start);part=s[start:end]
    rows.append({'path':path,'start_line':s[:start].count('\n')+1,'end_line':s[:end].count('\n'),'sha256':hashlib.sha256(part.encode()).hexdigest()});return part
k=span('tensor_cuda/src/kernels.cu','namespace tc {','namespace {\ntemplate <typename T> __device__')
b=span('tensor_cuda/src/bindings.cpp','constexpr const char* kCapLegacy','const char* const kDlpackDoc')
c=span('tensor_cuda/src/bindings.cpp','struct CoreParam {','}  // namespace\n\nPYBIND11_MODULE')
a=span('tensor_cuda/src/autograd.cpp','Tensor Tensor::make(', '\nTensor Tensor::from_op(')
f=span('tensor_cuda/src/kernels.cu','constexpr int64_t kFillMaxGridX','}  // namespace\n\nvoid fill_')
method=span('tensor_cuda/src/bindings.cpp','      .def("__dlpack__",','      .def("data_ptr",')
defs=span('tensor_cuda/src/bindings.cpp','  m.def("_from_dlpack_capsule",','  m.def("empty_cache",')
# The doc constant does not change mechanics; needed by the exact dispatcher snippet above.
preamble='''#include <new>
#include <cstdlib>
static bool fail_allocations=false;
void* operator new(std::size_t n){if(fail_allocations)throw std::bad_alloc();void* p=std::malloc(n?n:1);if(!p)throw std::bad_alloc();return p;}
void operator delete(void* p) noexcept {std::free(p);}
void operator delete(void* p,std::size_t) noexcept {std::free(p);}
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <dlpack/dlpack.h>
#include <tc/core.h>
#include <tc/autograd.h>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <initializer_list>
namespace py=pybind11;
'''
module='''
PYBIND11_MODULE(_p1t_rev8_coreoom,m){
 (void)release_lifecycle();
 m.def("runner_oom",[](py::object hook){fail_allocations=true;PyObject* r=raw_dlpack_exit_hook_runner(nullptr,hook.ptr());fail_allocations=false;if(!r)throw py::error_already_set();return py::reinterpret_steal<py::object>(r);});
 py::class_<Tensor>(m,"Tensor")
METHOD;
DEFS
 m.def("zero_cuda_label",[]{Device d;d.type=DeviceType::CPU;NDArray a({0},DType::Float32,d);a.device.type=DeviceType::CUDA;return Tensor::make(a,false);});


 m.def("in_flight",[]{return release_lifecycle().in_flight;});
 m.def("release",[](py::object fn){DLManagedTensor m{};m.manager_ctx=fn.ptr();m.deleter=[](DLManagedTensor* p){py::reinterpret_borrow<py::object>(static_cast<PyObject*>(p->manager_ctx))();};import_release_impl(&m);});
 m.def("fill_blocks",&tc::fill_grid_blocks);
 m.def("record",[]{note_core_unraisable("host probe error",0,cudaErrorInvalidValue);});
}
'''.replace('METHOD',method).replace('DEFS',defs)
(out/'core_host_oom.cpp').write_text(preamble+k+'}\nnamespace tc {\n'+a+'\nconstexpr int kT=256;\n'+f+'\n}\nusing namespace tc;\n'+b+'\nconst char* kDlpackDoc="host dispatcher";\n'+c+module)
(root/'artifacts/p1t_rev9/core_oom_source_slices.json').write_text(json.dumps(rows,indent=2)+'\n')
pb='/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src/include';cu='/usr/local/cuda-12.6'
cmd=['timeout','180','g++','-std=c++17','-O1','-shared','-fPIC','-Wl,-Bsymbolic','-fsanitize=undefined','-fno-sanitize-recover=undefined',str(out/'core_host_oom.cpp'),'-I'+str(root/'tensor_cuda/include'),'-I'+pb,'-I'+cu+'/include','-I'+sysconfig.get_path('include'),'-L'+cu+'/lib64','-Wl,-rpath,'+cu+'/lib64','-lcudart','-pthread','-o',str(out/('_p1t_rev8_coreoom'+sysconfig.get_config_var('EXT_SUFFIX')))]
subprocess.run(cmd,check=True)
print('PASS core host compilation; '+str(len(rows))+' exact spans')
