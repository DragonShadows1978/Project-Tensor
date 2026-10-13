"""Generate exact current-source core host slices, never stale product copies. PASS iff slices compile.
Prior art: previous review source-slice workflows; C++17 and CPython3.12 contracts.
Taken: exact extraction/hash receipts; ours: zero-size CUDA-labelled export exercises argument callbacks without GPU.
"""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[3];out=Path('/tmp/p1t_rev6_build');rows=[]
def span(path,a,b):
    text=(root/path).read_text();start=text.index(a);end=text.index(b,start);s=text[start:end]
    rows.append({'path':path,'start_line':text[:start].count('\n')+1,'end_line':text[:end].count('\n'),'sha256':hashlib.sha256(s.encode()).hexdigest()});return s
k=span('tensor_cuda/src/kernels.cu','namespace tc {','namespace {\ntemplate <typename T> __device__')
b=span('tensor_cuda/src/bindings.cpp','constexpr const char* kCapLegacy','const char* const kDlpackDoc')
a=span('tensor_cuda/src/autograd.cpp','Tensor Tensor::make(', '\nTensor Tensor::from_op(')
preamble='''// Generated verbatim source spans; source_slices.json pins them. No GPU kernels compiled/executed.
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
#include <utility>
namespace py=pybind11;
'''
module='''
PYBIND11_MODULE(_p1t_rev6_core,m){
 (void)release_lifecycle();
 py::class_<Tensor>(m,"Tensor")
 .def("__dlpack__",&tensor_dlpack,py::kw_only(),py::arg("stream")=py::none(),py::arg("max_version")=py::none(),py::arg("dl_device")=py::none(),py::arg("copy")=py::none());
 m.def("empty",&tensor_empty);
 m.def("import_capsule",&tensor_from_dlpack_capsule);
 m.def("zero_cuda_label",[]{Device d;d.type=DeviceType::CPU;NDArray a({0},DType::Float32,d);a.device.type=DeviceType::CUDA;return Tensor::make(a,false);});
 m.def("close",&dlpack_begin_shutdown);
 m.def("in_flight",[]{return release_lifecycle().in_flight;});
 m.def("fill_blocks",&tc::fill_grid_blocks);
}
'''
f=span('tensor_cuda/src/kernels.cu','constexpr int64_t kFillMaxGridX','}  // namespace\n\nvoid fill_')
(out/'core_host.cpp').write_text(preamble+k+'}\nnamespace tc {\n'+a+'\nconstexpr int kT=256;\n'+f+'\n}\nusing namespace tc;\n'+b+module)
(root/'artifacts/p1t_rev6/source_slices.json').write_text(json.dumps(rows,indent=2)+'\n')
