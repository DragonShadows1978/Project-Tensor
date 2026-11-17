// APA-AB1-T1 harness-side interop helper (not engine source; engine headers are
// included read-only).  It exposes, for an existing tensor_cuda Tensor, the
// device address of its first element, its shape, dtype code, and the byte
// size of the Storage it views.  No allocation, no copy, no kernel launch, no
// write: the harness wraps the address as a torch view through the CUDA Array
// Interface so torch and the engine exchange device data on the legacy default
// stream without a host round trip.
//
// Prior art: pybind11 (Jakob et al. 2016) cross-module type casting of a class
// registered by another extension (shared pybind11 internals); the Numba CUDA
// Array Interface v2 (Numba developers, 2018) consumed by torch.as_tensor;
// DLPack (2017) is the standard alternative, which this engine build does not
// expose.  Taken: those mechanisms.  Ours: only these accessors.
//
// `alias(t, shape, dtype)` returns a second Tensor handle on the SAME Storage
// (shared ownership, offset 0) with another shape/dtype of identical byte size:
// the dtype analogue of the engine's metadata-only NDArray::reshape.  It is
// used only to hand the engine's uint8 int8-V operand a byte-identical view of
// a bf16-sliced copy (the engine's slice op has no uint8 dispatch).
// `int4_ws(ws)` returns Tensor handles sharing the codes/scales Storage of an
// ApaInt4Workspace (read-only use: refine statistics and G1 content checks).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
#include "tc/autograd.h"

namespace py = pybind11;

static size_t elsize(tc::DType d) {
  switch (d) {
    case tc::DType::Float32: return 4;
    case tc::DType::Float16: return 2;
    case tc::DType::Int64: return 8;
    case tc::DType::Bool: return 1;
    case tc::DType::Uint8: return 1;
    case tc::DType::BFloat16: return 2;
  }
  return 0;
}

static int64_t numel(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (auto x : s) n *= x;
  return n;
}

static tc::Tensor share(const tc::NDArray& a) {
  auto var = std::make_shared<tc::Variable>();
  var->data = a;
  var->requires_grad = false;
  return tc::Tensor(var);
}

PYBIND11_MODULE(_apa_ab1_interop, m) {
  m.doc() = "APA-AB1-T1 read-only device-address accessor for tensor_cuda Tensors";
  m.def("info", [](py::object obj) {
    tc::Tensor& t = obj.cast<tc::Tensor&>();
    const tc::NDArray& a = t.data();
    if (!a.storage) throw std::runtime_error("undefined tensor");
    const size_t es = elsize(a.dtype);
    const uintptr_t base = reinterpret_cast<uintptr_t>(a.storage->ptr);
    const uintptr_t addr = base + static_cast<uintptr_t>(a.offset) * es;
    py::tuple shape(a.shape.size());
    for (size_t i = 0; i < a.shape.size(); ++i) shape[i] = py::int_(a.shape[i]);
    return py::make_tuple(py::int_(addr), shape, static_cast<int>(a.dtype),
                          py::int_(a.storage->nbytes), py::int_(base),
                          a.storage->pooled, a.device.is_cuda(), a.device.index,
                          py::int_(a.offset));
  });
  m.def("alias", [](py::object obj, std::vector<int64_t> shape, int dtype) {
    tc::Tensor& t = obj.cast<tc::Tensor&>();
    tc::NDArray a = t.data();
    if (!a.storage || a.offset != 0) throw std::runtime_error("alias: offset-0 tensors only");
    const tc::DType nd = static_cast<tc::DType>(dtype);
    if (numel(a.shape) * (int64_t)elsize(a.dtype) != numel(shape) * (int64_t)elsize(nd) || elsize(nd) == 0)
      throw std::runtime_error("alias: byte size mismatch");
    a.shape = shape;
    a.dtype = nd;
    return share(a);
  });
  m.def("int4_ws", [](py::object obj) {
    tc::ApaInt4Workspace& w = obj.cast<tc::ApaInt4Workspace&>();
    return py::make_tuple(share(w.codes), share(w.scales), w.valid_rows, w.capacity);
  });
}
