// bindings.cpp: pybind11 module exposing the C++/CUDA engine to Python.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>

#include "tc/autograd.h"
#include "tc/ops.h"

namespace py = pybind11;
using namespace tc;

namespace {

DType numpy_dtype(const py::array& a) {
  auto dt = a.dtype();
  if (dt.kind() == 'f' && dt.itemsize() == 4) return DType::Float32;
  if (dt.kind() == 'f' && dt.itemsize() == 2) return DType::Float16;
  if (dt.kind() == 'i' && dt.itemsize() == 8) return DType::Int64;
  if (dt.kind() == 'u' && dt.itemsize() == 1) return DType::Uint8;  // packed int4
  if (dt.kind() == 'b') return DType::Bool;
  return DType::Float32;  // others are force-cast to float32 by the wrapper
}

Tensor tensor_from_numpy(py::array arr, const std::string& device, bool requires_grad) {
  DType dt = numpy_dtype(arr);
  // Force a C-contiguous buffer of the matching element size.
  py::array contig = py::array::ensure(arr, py::array::c_style);
  auto info = contig.request();
  Shape shape(info.shape.begin(), info.shape.end());
  Device dev = Device::from_string(device);
  NDArray nd = NDArray::from_host(info.ptr, shape, dt, dev);
  return Tensor::make(nd, requires_grad);
}

py::array tensor_to_numpy(Tensor& t) {
  NDArray a = t.data();
  std::vector<py::ssize_t> shape(a.shape.begin(), a.shape.end());
  py::array out;
  if (a.dtype == DType::Float16) out = py::array(py::dtype("float16"), shape);
  else if (a.dtype == DType::Int64) out = py::array(py::dtype("int64"), shape);
  else if (a.dtype == DType::Uint8) out = py::array(py::dtype("uint8"), shape);
  else out = py::array(py::dtype("float32"), shape);
  NDArray host = a.device.is_cuda() ? a : a;  // to_host handles D2H
  host.to_host(out.request().ptr);
  return out;
}

py::object grad_of(Tensor& t) {
  if (!t.v->grad.defined()) return py::none();
  return py::cast(Tensor::make(t.v->grad, false));
}

// __getitem__ for ints and unit-step slices (single key or per-dim tuple),
// composed from the differentiable slice/squeeze ops.
Tensor getitem(Tensor& t, py::object key) {
  std::vector<py::object> keys;
  if (py::isinstance<py::tuple>(key))
    for (auto k : key.cast<py::tuple>()) keys.push_back(py::reinterpret_borrow<py::object>(k));
  else
    keys.push_back(key);

  Tensor cur = t;
  int dim = 0;
  for (auto& k : keys) {
    if (py::isinstance<py::int_>(k)) {
      int64_t i = k.cast<int64_t>();
      int64_t sz = cur.shape()[dim];
      if (i < 0) i += sz;
      cur = ops::squeeze(ops::slice(cur, dim, i, 1), dim);  // dim removed; keep `dim`
    } else if (py::isinstance<py::slice>(k)) {
      size_t start, stop, step, len;
      k.cast<py::slice>().compute(cur.shape()[dim], &start, &stop, &step, &len);
      if (step != 1) throw std::runtime_error("getitem: only step==1 slices supported");
      cur = ops::slice(cur, dim, (int64_t)start, (int64_t)len);
      ++dim;
    } else {
      throw std::runtime_error("getitem: unsupported index type");
    }
  }
  return cur;
}

}  // namespace

PYBIND11_MODULE(_tensor_cuda, m) {
  m.doc() = "Project Tensor — standalone CUDA tensor engine (no PyTorch).";

  py::class_<Tensor>(m, "Tensor")
      .def_property_readonly("shape", [](Tensor& t) {
        return py::tuple(py::cast(t.shape()));
      })
      .def_property_readonly("ndim", [](Tensor& t) { return t.ndim(); })
      .def_property_readonly("dtype", [](Tensor& t) { return std::string(dtype_name(t.dtype())); })
      .def_property_readonly("device", [](Tensor& t) { return t.device().str(); })
      .def_property("requires_grad",
                    [](Tensor& t) { return t.requires_grad(); },
                    [](Tensor& t, bool r) { t.set_requires_grad(r); })
      .def_property_readonly("grad", &grad_of)
      .def("numpy", &tensor_to_numpy)
      .def("backward", [](Tensor& t) { t.backward(); })
      .def("zero_grad", &Tensor::zero_grad)
      .def("reshape", [](Tensor& t, std::vector<int64_t> s) { return ops::reshape(t, s); })
      .def("transpose_last", [](Tensor& t) { return ops::transpose_last(t); })
      .def("sum", [](Tensor& t, std::vector<int> axes, bool keepdim) { return ops::sum(t, axes, keepdim); },
           py::arg("axes") = std::vector<int>{}, py::arg("keepdim") = false)
      .def("mean", [](Tensor& t, std::vector<int> axes, bool keepdim) { return ops::mean(t, axes, keepdim); },
           py::arg("axes") = std::vector<int>{}, py::arg("keepdim") = false)
      .def("relu", [](Tensor& t) { return ops::relu(t); })
      .def("gelu", [](Tensor& t) { return ops::gelu(t); })
      .def("silu", [](Tensor& t) { return ops::silu(t); })
      .def("abs", [](Tensor& t) { return ops::abs(t); })
      .def("detach", [](Tensor& t) { return ops::detach(t); })
      .def("half", [](Tensor& t) { return ops::cast(t, DType::Float16); })
      .def("float", [](Tensor& t) { return ops::cast(t, DType::Float32); })
      .def("astype", [](Tensor& t, const std::string& dt) {
        return ops::cast(t, dtype_from_string(dt));
      })
      .def("sin", [](Tensor& t) { return ops::sin(t); })
      .def("cos", [](Tensor& t) { return ops::cos(t); })
      .def("tan", [](Tensor& t) { return ops::tan(t); })
      .def("asin", [](Tensor& t) { return ops::asin(t); })
      .def("acos", [](Tensor& t) { return ops::acos(t); })
      .def("atan", [](Tensor& t) { return ops::atan(t); })
      .def("sinh", [](Tensor& t) { return ops::sinh(t); })
      .def("cosh", [](Tensor& t) { return ops::cosh(t); })
      .def("log2", [](Tensor& t) { return ops::log2(t); })
      .def("log10", [](Tensor& t) { return ops::log10(t); })
      .def("sign", [](Tensor& t) { return ops::sign(t); })
      .def("floor", [](Tensor& t) { return ops::floor(t); })
      .def("ceil", [](Tensor& t) { return ops::ceil(t); })
      .def("round", [](Tensor& t) { return ops::round(t); })
      .def("isnan", [](Tensor& t) { return ops::isnan(t); })
      .def("isinf", [](Tensor& t) { return ops::isinf(t); })
      .def("isfinite", [](Tensor& t) { return ops::isfinite(t); })
      .def("nan_to_num", [](Tensor& t, double n, double p, double m) { return ops::nan_to_num(t, n, p, m); },
           py::arg("nan") = 0.0, py::arg("posinf") = 1e30, py::arg("neginf") = -1e30)
      .def("reciprocal", [](Tensor& t) { return ops::reciprocal(t); })
      .def("clamp", [](Tensor& t, double lo, double hi) { return ops::clamp(t, lo, hi); })
      .def("maximum", [](Tensor& a, Tensor& b) { return ops::maximum(a, b); })
      .def("minimum", [](Tensor& a, Tensor& b) { return ops::minimum(a, b); })
      .def("slice", [](Tensor& t, int dim, int64_t start, int64_t len) { return ops::slice(t, dim, start, len); })
      .def("__getitem__", &getitem)
      .def("sigmoid", [](Tensor& t) { return ops::sigmoid(t); })
      .def("tanh", [](Tensor& t) { return ops::tanh(t); })
      .def("exp", [](Tensor& t) { return ops::exp(t); })
      .def("log", [](Tensor& t) { return ops::log(t); })
      .def("sqrt", [](Tensor& t) { return ops::sqrt(t); })
      .def("softmax", [](Tensor& t, int axis) { return ops::softmax(t, axis); }, py::arg("axis") = -1)
      .def("log_softmax", [](Tensor& t, int axis) { return ops::log_softmax(t, axis); }, py::arg("axis") = -1)
      .def("pow", [](Tensor& t, double p) { return ops::pow_scalar(t, p); })
      .def("max", [](Tensor& t, std::vector<int> ax, bool kd) { return ops::max(t, ax, kd); }, py::arg("axes") = std::vector<int>{}, py::arg("keepdim") = false)
      .def("min", [](Tensor& t, std::vector<int> ax, bool kd) { return ops::min(t, ax, kd); }, py::arg("axes") = std::vector<int>{}, py::arg("keepdim") = false)
      .def("var", [](Tensor& t, std::vector<int> ax, bool kd) { return ops::var(t, ax, kd); }, py::arg("axes") = std::vector<int>{}, py::arg("keepdim") = false)
      .def("std", [](Tensor& t, std::vector<int> ax, bool kd) { return ops::std(t, ax, kd); }, py::arg("axes") = std::vector<int>{}, py::arg("keepdim") = false)
      .def("prod", [](Tensor& t, std::vector<int> ax, bool kd) { return ops::prod(t, ax, kd); }, py::arg("axes") = std::vector<int>{}, py::arg("keepdim") = false)
      .def("argmax", [](Tensor& t, int ax) { return ops::argmax(t, ax); }, py::arg("axis") = -1)
      .def("argmin", [](Tensor& t, int ax) { return ops::argmin(t, ax); }, py::arg("axis") = -1)
      .def("cumsum", [](Tensor& t, int ax) { return ops::cumsum(t, ax); }, py::arg("axis") = -1)
      .def("gather", [](Tensor& t, int dim, Tensor& idx) { return ops::gather(t, dim, idx); })
      .def("flip", [](Tensor& t, std::vector<int> dims) { return ops::flip(t, dims); })
      .def("topk", [](Tensor& t, int k, bool largest) {
        auto pr = ops::topk(t, k, largest);
        return py::make_tuple(std::get<0>(pr), std::get<1>(pr));
      }, py::arg("k"), py::arg("largest") = true)
      .def("sort", [](Tensor& t, bool descending) {
        auto pr = ops::topk(t, t.shape().back(), descending);
        return py::make_tuple(std::get<0>(pr), std::get<1>(pr));
      }, py::arg("descending") = false)
      .def("permute", [](Tensor& t, std::vector<int> d) { return ops::permute(t, d); })
      .def("transpose", [](Tensor& t, int a, int b) { return ops::transpose(t, a, b); })
      .def("squeeze", [](Tensor& t, int d) { return ops::squeeze(t, d); }, py::arg("dim") = 0)
      .def("unsqueeze", [](Tensor& t, int d) { return ops::unsqueeze(t, d); })
      .def("expand", [](Tensor& t, std::vector<int64_t> s) { return ops::expand(t, s); })
      .def("flatten", [](Tensor& t, int s, int e) { return ops::flatten(t, s, e); }, py::arg("start_dim") = 0, py::arg("end_dim") = -1)
      .def("masked_fill", [](Tensor& t, Tensor& m, double v) { return ops::masked_fill(t, m, v); })
      .def("gt", [](Tensor& a, Tensor& b) { return ops::compare(a, b, 0); })
      .def("ge", [](Tensor& a, Tensor& b) { return ops::compare(a, b, 1); })
      .def("lt", [](Tensor& a, Tensor& b) { return ops::compare(a, b, 2); })
      .def("le", [](Tensor& a, Tensor& b) { return ops::compare(a, b, 3); })
      .def("eq", [](Tensor& a, Tensor& b) { return ops::compare(a, b, 4); })
      .def("__gt__", [](Tensor& a, double s) { return ops::compare_scalar(a, s, 0); })
      .def("__ge__", [](Tensor& a, double s) { return ops::compare_scalar(a, s, 1); })
      .def("__lt__", [](Tensor& a, double s) { return ops::compare_scalar(a, s, 2); })
      .def("__le__", [](Tensor& a, double s) { return ops::compare_scalar(a, s, 3); })
      .def("__pow__", [](Tensor& a, double p) { return ops::pow_scalar(a, p); })
      // operator overloads (Tensor op Tensor and Tensor op scalar)
      .def("__add__", [](Tensor& a, Tensor& b) { return ops::add(a, b); })
      .def("__add__", [](Tensor& a, double s) { return ops::add_scalar(a, s); })
      .def("__radd__", [](Tensor& a, double s) { return ops::add_scalar(a, s); })
      .def("__sub__", [](Tensor& a, Tensor& b) { return ops::sub(a, b); })
      .def("__sub__", [](Tensor& a, double s) { return ops::add_scalar(a, -s); })
      .def("__mul__", [](Tensor& a, Tensor& b) { return ops::mul(a, b); })
      .def("__mul__", [](Tensor& a, double s) { return ops::mul_scalar(a, s); })
      .def("__rmul__", [](Tensor& a, double s) { return ops::mul_scalar(a, s); })
      .def("__truediv__", [](Tensor& a, Tensor& b) { return ops::div(a, b); })
      .def("__truediv__", [](Tensor& a, double s) { return ops::mul_scalar(a, 1.0 / s); })
      .def("__matmul__", [](Tensor& a, Tensor& b) { return ops::matmul(a, b); })
      .def("__neg__", [](Tensor& a) { return ops::neg(a); });

  // factory
  m.def("tensor", &tensor_from_numpy, py::arg("array"), py::arg("device") = "cuda",
        py::arg("requires_grad") = false);

  // free-function ops
  m.def("matmul", &ops::matmul);
  m.def("mse_loss", &ops::mse_loss);
  m.def("cross_entropy", &ops::cross_entropy);
  m.def("add", &ops::add);
  m.def("mul", &ops::mul);
  m.def("where", &ops::where);
  m.def("cat", [](std::vector<Tensor> ts, int dim) { return ops::cat(ts, dim); }, py::arg("tensors"), py::arg("dim") = 0);
  m.def("stack", [](std::vector<Tensor> ts, int dim) { return ops::stack(ts, dim); }, py::arg("tensors"), py::arg("dim") = 0);
  m.def("embedding", &ops::embedding);
  m.def("apa_quantize_gather", [](Tensor& r, Tensor& b, Tensor& c) {
    return Tensor::make(tc::apa_quantize_gather(r.data(), b.data(), c.data()), false);
  });
  // INT4 group-quantized linear (inference only, no autograd).
  m.def("int4_linear", [](Tensor& x, Tensor& packed, Tensor& scales,
                          Tensor& zeros, int group_size) {
    return Tensor::make(
        tc::int4_linear(x.data(), packed.data(), scales.data(), zeros.data(), group_size),
        false);
  }, py::arg("x"), py::arg("packed"), py::arg("scales"), py::arg("zeros"),
     py::arg("group_size") = 128);
  m.def("int4_dequant", [](Tensor& packed, Tensor& scales, Tensor& zeros,
                           int group_size, const std::string& out_dtype) {
    return Tensor::make(
        tc::int4_dequant(packed.data(), scales.data(), zeros.data(), group_size,
                         dtype_from_string(out_dtype)),
        false);
  }, py::arg("packed"), py::arg("scales"), py::arg("zeros"),
     py::arg("group_size") = 128, py::arg("out_dtype") = "float16");
  // Fused sparse selective APA attention (inference only, no autograd).
  m.def("apa_selective_attention", [](Tensor& q, Tensor& k, Tensor& kq, Tensor& v,
                                      double scale, double zthr, bool is_causal) {
    return Tensor::make(
        tc::apa_selective_attention(q.data(), k.data(), kq.data(), v.data(),
                                    (float)scale, (float)zthr, is_causal),
        false);
  }, py::arg("q"), py::arg("k"), py::arg("kq"), py::arg("v"),
     py::arg("scale"), py::arg("zthr"), py::arg("is_causal") = false);
  m.def("im2col", &ops::im2col);
  m.def("col2im", &ops::col2im);
  m.def("avg_pool2d", &ops::avg_pool2d);
  m.def("max_pool2d", &ops::max_pool2d);

  // in-place optimizer steps (param/state mutated on device)
  m.def("sgd_step", [](Tensor& p, Tensor& g, Tensor& buf, double lr, double mom, double wd) {
    tc::sgd_step(p.data(), g.data(), buf.data(), lr, mom, wd);
  });
  m.def("adam_step", [](Tensor& p, Tensor& g, Tensor& m_, Tensor& v_, double lr,
                        double b1, double b2, double eps, int64_t t, double wd, bool dec) {
    tc::adam_step(p.data(), g.data(), m_.data(), v_.data(), lr, b1, b2, eps, t, wd, dec);
  });
  m.def("rmsprop_step", [](Tensor& p, Tensor& g, Tensor& sq, double lr, double alpha, double eps, double wd) {
    tc::rmsprop_step(p.data(), g.data(), sq.data(), lr, alpha, eps, wd);
  });
  m.def("lion_step", [](Tensor& p, Tensor& g, Tensor& m_, double lr, double b1, double b2, double wd) {
    tc::lion_step(p.data(), g.data(), m_.data(), lr, b1, b2, wd);
  });
  m.def("radam_step", [](Tensor& p, Tensor& g, Tensor& m_, Tensor& v_, double lr, double b1,
                         double b2, double eps, double bc1, double bc2, double rect, bool rectified,
                         double wd, bool dec) {
    tc::radam_step(p.data(), g.data(), m_.data(), v_.data(), lr, b1, b2, eps, bc1, bc2, rect, rectified, wd, dec);
  });
  m.def("adagrad_step", [](Tensor& p, Tensor& g, Tensor& acc, double lr, double eps, double wd) {
    tc::adagrad_step(p.data(), g.data(), acc.data(), lr, eps, wd);
  });
  m.def("scale_", [](Tensor& t, double s) { tc::scale_(t.data(), s); });
  m.def("axpy_", [](Tensor& p, Tensor& o, double a) { tc::axpy_(p.data(), o.data(), a); });

  // grad mode
  m.def("is_grad_enabled", &grad_enabled);
  m.def("set_grad_enabled", &set_grad_enabled);
  m.def("synchronize", &cuda_sync);
}
