// bindings.cpp: pybind11 module exposing the C++/CUDA engine to Python.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <utility>
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
  // numpy has no bf16 — upcast to fp32 on device so the host buffer is fp32.
  if (a.dtype == DType::BFloat16) a = a.astype(DType::Float32);
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

py::tuple tensor_args(const std::vector<Tensor>& inputs) {
  py::tuple args(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) args[i] = py::cast(inputs[i]);
  return args;
}

Tensor checkpoint_py(py::function fn, std::vector<Tensor> inputs) {
  bool any_req = false;
  for (auto& t : inputs) any_req = any_req || t.requires_grad();
  bool req = grad_enabled() && any_req;
  if (!req) {
    py::object obj = fn(*tensor_args(inputs));
    return obj.cast<Tensor>();
  }

  py::object obj;
  {
    NoGradGuard guard;
    obj = fn(*tensor_args(inputs));
  }
  Tensor out0 = obj.cast<Tensor>();
  NDArray out_data = out0.data();

  return Tensor::from_op(out_data, inputs, "checkpoint",
      [fn, inputs](const NDArray& g) {
        py::gil_scoped_acquire gil;

        std::vector<Tensor> replay_inputs;
        replay_inputs.reserve(inputs.size());
        for (auto& t : inputs) {
          replay_inputs.push_back(Tensor::make(t.data(), t.requires_grad()));
        }

        bool prev = grad_enabled();
        set_grad_enabled(true);
        py::object replay_obj;
        try {
          replay_obj = fn(*tensor_args(replay_inputs));
        } catch (...) {
          set_grad_enabled(prev);
          throw;
        }
        Tensor replay_out = replay_obj.cast<Tensor>();
        try {
          replay_out.backward(g);
        } catch (...) {
          set_grad_enabled(prev);
          throw;
        }
        set_grad_enabled(prev);

        for (size_t i = 0; i < inputs.size(); ++i) {
          if (inputs[i].requires_grad() && replay_inputs[i].grad().defined()) {
            inputs[i].v->accumulate_grad(replay_inputs[i].grad());
          }
        }
      });
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

std::vector<TerrainRenderObject> terrain_objects_from_python(
    const py::object& objects) {
  std::vector<TerrainRenderObject> parsed;
  if (objects.is_none()) return parsed;
  if (!py::isinstance<py::list>(objects)) {
    throw std::runtime_error("terrain_render: objects must be a list or None");
  }
  const py::list entries = py::reinterpret_borrow<py::list>(objects);
  if (entries.size() > 16) {
    throw std::runtime_error("terrain_render: objects supports at most 16 entries");
  }
  parsed.reserve(entries.size());
  for (py::ssize_t index = 0; index < entries.size(); ++index) {
    const py::handle entry = entries[index];
    if ((!py::isinstance<py::tuple>(entry) &&
         !py::isinstance<py::list>(entry)) ||
        py::len(entry) != 3) {
      throw std::runtime_error(
          "terrain_render: each object must be (grid, origin, palette)");
    }
    const py::sequence terms = py::reinterpret_borrow<py::sequence>(entry);
    Tensor grid = py::cast<Tensor>(terms[0]);
    const std::vector<float> origin =
        py::cast<std::vector<float>>(terms[1]);
    Tensor palette = py::cast<Tensor>(terms[2]);
    if (origin.size() != 3 || !std::isfinite(origin[0]) ||
        !std::isfinite(origin[1]) || !std::isfinite(origin[2])) {
      throw std::runtime_error(
          "terrain_render: object origin must be a finite three-vector");
    }
    TerrainRenderObject object{};
    object.grid = grid.data();
    object.origin[0] = origin[0];
    object.origin[1] = origin[1];
    object.origin[2] = origin[2];
    object.palette = palette.data();
    parsed.push_back(std::move(object));
  }
  return parsed;
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
      .def("gelu_exact", [](Tensor& t) { return ops::gelu_exact(t); })
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
      .def("argmax_last_axis", [](Tensor& t) { return ops::argmax_last_axis(t); })
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
  m.def("matmul", &ops::matmul, py::arg("a"), py::arg("b"),
        py::arg("alpha") = 1.f, py::arg("trans_b") = false);
  m.def("rms_norm", &ops::rms_norm, py::arg("x"), py::arg("w"), py::arg("eps"));
  m.def("rope_apply", &ops::rope_apply, py::arg("x"), py::arg("cos"),
        py::arg("sin"), py::arg("pos0"), py::arg("inverse") = false,
        py::arg("pair_swap") = false);
  m.def("write_rows", &ops::write_rows, py::arg("buf"), py::arg("src"),
        py::arg("start"));
  m.def("export_rows", &ops::export_rows, py::arg("cache"),
        py::arg("dim"), py::arg("start"), py::arg("len"));
  m.def("export_rope_rows", &ops::export_rope_rows, py::arg("cache"),
        py::arg("cos"), py::arg("sin"), py::arg("dim"),
        py::arg("start"), py::arg("len"), py::arg("pos0"),
        py::arg("inverse") = false, py::arg("pair_swap") = false);
  m.def("export_row_pair", &ops::export_row_pair, py::arg("raw_cache"),
        py::arg("rope_cache"), py::arg("cos"), py::arg("sin"),
        py::arg("raw_dim"), py::arg("rope_dim"), py::arg("raw_start"),
        py::arg("rope_start"), py::arg("len"), py::arg("pos0"),
        py::arg("inverse") = false, py::arg("pair_swap") = false);
  m.def("export_row_pairs", &ops::export_row_pairs, py::arg("raw_caches"),
        py::arg("rope_caches"), py::arg("cos"), py::arg("sin"),
        py::arg("raw_dim"), py::arg("rope_dim"), py::arg("raw_starts"),
        py::arg("rope_starts"), py::arg("len"), py::arg("pos0"),
        py::arg("inverse") = false, py::arg("pair_swap") = false);
  m.def("swap_row_pairs_with_rope", &ops::swap_row_pairs_with_rope,
        py::arg("raw_caches"), py::arg("rope_caches"),
        py::arg("raw_inserts"), py::arg("rope_inserts"), py::arg("cos"),
        py::arg("sin"), py::arg("raw_dim"), py::arg("rope_dim"),
        py::arg("head_tokens"), py::arg("tail_start"), py::arg("pos0"),
        py::arg("pair_swap") = false);
  m.def("evict_row_pairs", &ops::evict_row_pairs, py::arg("raw_caches"),
        py::arg("rope_caches"), py::arg("raw_dim"), py::arg("rope_dim"),
        py::arg("head_tokens"), py::arg("drop_tokens"));
  m.def("arena_row_pair_transaction", &ops::arena_row_pair_transaction,
        py::arg("raw_caches"), py::arg("rope_caches"),
        py::arg("raw_inserts"), py::arg("rope_inserts"), py::arg("cos"),
        py::arg("sin"), py::arg("raw_dim"), py::arg("rope_dim"),
        py::arg("sink_tokens"), py::arg("current_mount_tokens"),
        py::arg("arena_width"), py::arg("pair_swap") = false);
  m.def("splice_rows", &ops::splice_rows, py::arg("old_cache"),
        py::arg("insert"), py::arg("dim"), py::arg("head_tokens"),
        py::arg("tail_start"));
  m.def("evict_rows", &ops::evict_rows, py::arg("old_cache"),
        py::arg("dim"), py::arg("head_tokens"), py::arg("drop_tokens"));
  m.def("dda_raycast",
        [](Tensor& grid, Tensor& origins, Tensor& directions, int max_steps) {
          auto out = ops::dda_raycast(grid, origins, directions, max_steps);
          return py::make_tuple(
              std::get<0>(out), std::get<1>(out), std::get<2>(out),
              std::get<3>(out), std::get<4>(out), std::get<5>(out));
        },
        py::arg("grid_u8"), py::arg("origins_f32"),
        py::arg("directions_f32"), py::arg("max_steps"));
  m.def("terrain_render",
        [](Tensor& materials, Tensor& palette,
           const std::vector<float>& position,
           const std::vector<float>& forward,
           const std::vector<float>& right, const std::vector<float>& up,
           float half_width, float half_height, int width, int height,
           const std::vector<float>& light_direction, int max_steps,
           const std::string& surface_mode, int density_filter, int detail,
           const py::object& objects, int grounding, float z_horizon,
           float fog_start, float fog_full) {
          if (position.size() != 3 || forward.size() != 3 ||
              right.size() != 3 || up.size() != 3 ||
              light_direction.size() != 3) {
            throw std::runtime_error(
                "terrain_render: camera vectors and light_direction must have length 3");
          }
          TerrainRenderCamera camera{};
          TerrainRenderLight light{};
          for (int axis = 0; axis < 3; ++axis) {
            camera.position[axis] = position[axis];
            camera.forward[axis] = forward[axis];
            camera.right[axis] = right[axis];
            camera.up[axis] = up[axis];
            light.direction[axis] = light_direction[axis];
          }
          camera.half_width = half_width;
          camera.half_height = half_height;
          camera.width = width;
          camera.height = height;
          if (max_steps <= 0) {
            throw std::runtime_error("terrain_render: max_steps must be positive");
          }
          if (surface_mode != "blocky" && surface_mode != "smooth") {
            throw std::runtime_error(
                "terrain_render: surface_mode must be 'blocky' or 'smooth'");
          }
          if (density_filter < 0 || density_filter > 2) {
            throw std::runtime_error(
                "terrain_render: density_filter must be 0, 1, or 2");
          }
          if (surface_mode == "blocky" && density_filter != 0) {
            throw std::runtime_error(
                "terrain_render: density_filter is only supported in smooth mode");
          }
          if (detail != 0 && detail != 1) {
            throw std::runtime_error(
                "terrain_render: detail must be 0 or 1");
          }
          if (grounding != 0 && grounding != 1) {
            throw std::runtime_error(
                "terrain_render: grounding must be 0 or 1");
          }
          if (!std::isfinite(z_horizon) || !std::isfinite(fog_start) ||
              !std::isfinite(fog_full)) {
            throw std::runtime_error(
                "terrain_render: grounding parameters must be finite");
          }
          if (fog_start < 0.0f || fog_full <= fog_start) {
            throw std::runtime_error(
                "terrain_render: fog range must satisfy 0 <= fog_start < fog_full");
          }
          // TerrainRenderConstants predates WO-8A and is used by the C++ op
          // ABI.  Reserve its otherwise-invalid negative range for the
          // pybind-only smooth selector; terrain.cu decodes it before launch.
          const int encoded_max_steps =
              surface_mode == "smooth" ? -max_steps : max_steps;
          TerrainRenderConstants constants{
              encoded_max_steps, density_filter, detail, grounding,
              z_horizon, fog_start, fog_full};
          const auto render_objects = terrain_objects_from_python(objects);
          auto out = render_objects.empty()
                         ? ops::terrain_render(materials, camera, light,
                                               palette, constants)
                         : ops::terrain_render(materials, camera, light,
                                               palette, constants,
                                               render_objects);
          return py::make_tuple(std::get<0>(out), std::get<1>(out));
        },
        py::arg("materials_u8"), py::arg("palette_u8"),
        py::arg("position"), py::arg("forward"), py::arg("right"),
        py::arg("up"), py::arg("half_width"), py::arg("half_height"),
        py::arg("width"), py::arg("height"), py::arg("light_direction"),
        py::arg("max_steps"), py::arg("surface_mode") = "blocky",
        py::arg("density_filter") = 0, py::arg("detail") = 0,
        py::arg("objects") = py::none(), py::arg("grounding") = 0,
        py::arg("z_horizon") = 0.0f, py::arg("fog_start") = 600.0f,
        py::arg("fog_full") = 2400.0f);
  m.def("causal_softmax", &ops::causal_softmax, py::arg("scores"));
  m.def("fused_sdpa_noncausal", &ops::fused_sdpa_noncausal,
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"));
  m.def("apa_int4_sdpa_noncausal", &ops::apa_int4_sdpa_noncausal,
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"),
        py::arg("zthr"), py::arg("refine_all") = false);
  m.def("argmax_last_axis", &ops::argmax_last_axis, py::arg("a"));
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
  // INT4 group-quantized linear. Weights are frozen (uint8/fp16 buffers,
  // never differentiable); the VJP covers x only — dx = g @ W^T with the
  // weight re-dequantized transiently at backward time (ops.cpp).
  m.def("int4_linear", &ops::int4_linear,
        py::arg("x"), py::arg("packed"), py::arg("scales"), py::arg("zeros"),
        py::arg("group_size") = 128);
  m.def("int4_linear_fused", &ops::int4_linear_fused,
        py::arg("x"), py::arg("packed"), py::arg("scales"), py::arg("zeros"),
        py::arg("group_size") = 128);
  m.def("intn_linear", &ops::intn_linear,
        py::arg("x"), py::arg("packed"), py::arg("scales"), py::arg("zeros"),
        py::arg("bits"), py::arg("in_features"),
        py::arg("group_size") = 128);
  m.def("intn_linear_fused", &ops::intn_linear_fused,
        py::arg("x"), py::arg("packed"), py::arg("scales"), py::arg("zeros"),
        py::arg("bits"), py::arg("in_features"),
        py::arg("group_size") = 128);
  m.def("mxfp4_linear", &ops::mxfp4_linear,
        py::arg("x"), py::arg("blocks"), py::arg("scales"));
  m.def("mxfp4_linear_expert", &ops::mxfp4_linear_expert,
        py::arg("x"), py::arg("blocks"), py::arg("scales"),
        py::arg("expert_idx"));
  // Differentiable O(L) selective attention (graft-native training path).
  m.def("apa_selective_train", [](Tensor& q, Tensor& k, Tensor& kq, Tensor& v,
                                  double scale, double zthr, bool is_causal) {
    return ops::apa_selective_train(q, k, kq, v, (float)scale, (float)zthr,
                                    is_causal);
  }, py::arg("q"), py::arg("k"), py::arg("kq"), py::arg("v"),
     py::arg("scale"), py::arg("zthr"), py::arg("is_causal") = false);
  // Fused GDN decode step (inference-only, functional: (out, new_state)).
  m.def("gated_delta_step", [](Tensor& q, Tensor& k, Tensor& v, Tensor& a,
                               Tensor& b, Tensor& A_neg, Tensor& dt_bias,
                               Tensor& state) {
    auto r = tc::gated_delta_step(q.data(), k.data(), v.data(), a.data(),
                                  b.data(), A_neg.data(), dt_bias.data(),
                                  state.data());
    return py::make_tuple(Tensor::make(r.first, false),
                          Tensor::make(r.second, false));
  }, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("a"), py::arg("b"),
     py::arg("A_neg"), py::arg("dt_bias"), py::arg("state"));
  m.def("int4_dequant", [](Tensor& packed, Tensor& scales, Tensor& zeros,
                           int group_size, const std::string& out_dtype) {
    return Tensor::make(
        tc::int4_dequant(packed.data(), scales.data(), zeros.data(), group_size,
                         dtype_from_string(out_dtype)),
        false);
  }, py::arg("packed"), py::arg("scales"), py::arg("zeros"),
     py::arg("group_size") = 128, py::arg("out_dtype") = "float16");
  m.def("intn_dequant", [](Tensor& packed, Tensor& scales, Tensor& zeros,
                           int bits, int64_t in_features, int group_size,
                           const std::string& out_dtype) {
    return Tensor::make(
        tc::intn_dequant(packed.data(), scales.data(), zeros.data(), bits,
                         in_features, group_size,
                         dtype_from_string(out_dtype)),
        false);
  }, py::arg("packed"), py::arg("scales"), py::arg("zeros"),
     py::arg("bits"), py::arg("in_features"),
     py::arg("group_size") = 128, py::arg("out_dtype") = "float16");
  // KV-cache INT4 storage (D-grouped symmetric-8). Distinct from int4_dequant
  // (weight path, K-grouped). pack -> (packed_u8, scales); unpack reads a
  // [lo:lo+n) slice on S. Inference-only (no autograd: the cache is frozen).
  m.def("kv_int4_pack", [](Tensor& x, int group) {
    NDArray scales;
    NDArray packed = tc::kv_int4_pack(x.data(), scales, group);
    return py::make_tuple(Tensor::make(packed, false),
                          Tensor::make(scales, false));
  }, py::arg("x"), py::arg("group") = 32);
  m.def("kv_int4_unpack", [](Tensor& packed, Tensor& scales, int group,
                             int64_t lo, int64_t n, const std::string& out_dtype) {
    return Tensor::make(
        tc::kv_int4_unpack(packed.data(), scales.data(), group, lo, n,
                           dtype_from_string(out_dtype)),
        false);
  }, py::arg("packed"), py::arg("scales"), py::arg("group") = 32,
     py::arg("lo") = 0, py::arg("n") = 0, py::arg("out_dtype") = "bfloat16");
  // Fused sparse selective APA attention (inference only, no autograd).
  m.def("apa_selective_attention", [](Tensor& q, Tensor& k, Tensor& kq, Tensor& v,
                                      double scale, double zthr, bool is_causal) {
    return Tensor::make(
        tc::apa_selective_attention(q.data(), k.data(), kq.data(), v.data(),
                                    (float)scale, (float)zthr, is_causal),
        false);
  }, py::arg("q"), py::arg("k"), py::arg("kq"), py::arg("v"),
     py::arg("scale"), py::arg("zthr"), py::arg("is_causal") = false);
  m.def("apa_selective_attention_sink",
      [](Tensor& q, Tensor& k, Tensor& kq, Tensor& v, Tensor& sinks,
         double scale, double zthr, bool is_causal) {
    return Tensor::make(
        tc::apa_selective_attention_sink(q.data(), k.data(), kq.data(), v.data(),
                                         sinks.data(), (float)scale,
                                         (float)zthr, is_causal),
        false);
  }, py::arg("q"), py::arg("k"), py::arg("kq"), py::arg("v"),
     py::arg("sinks"), py::arg("scale"), py::arg("zthr"),
     py::arg("is_causal") = false);
  // APA selective TRAINING forward: O(L) memory, saves (lse, thr) for backward.
  // Returns (out, lse, thr) as plain (non-grad) tensors; Python wires autograd.
  m.def("apa_selective_fwd_train", [](Tensor& q, Tensor& k, Tensor& kq, Tensor& v,
                                      double scale, double zthr, bool is_causal) {
    auto r = tc::apa_selective_fwd_train(q.data(), k.data(), kq.data(), v.data(),
                                         (float)scale, (float)zthr, is_causal);
    return py::make_tuple(Tensor::make(std::get<0>(r), false),
                          Tensor::make(std::get<1>(r), false),
                          Tensor::make(std::get<2>(r), false));
  }, py::arg("q"), py::arg("k"), py::arg("kq"), py::arg("v"),
     py::arg("scale"), py::arg("zthr"), py::arg("is_causal") = false);
  // APA selective backward: returns (dq, dk, dv).
  m.def("apa_selective_bwd", [](Tensor& q, Tensor& k, Tensor& kq, Tensor& v,
                                Tensor& dO, Tensor& lse, Tensor& thr,
                                double scale, bool is_causal) {
    auto r = tc::apa_selective_bwd(q.data(), k.data(), kq.data(), v.data(),
                                   dO.data(), lse.data(), thr.data(),
                                   (float)scale, is_causal);
    return py::make_tuple(Tensor::make(std::get<0>(r), false),
                          Tensor::make(std::get<1>(r), false),
                          Tensor::make(std::get<2>(r), false));
  }, py::arg("q"), py::arg("k"), py::arg("kq"), py::arg("v"), py::arg("dO"),
     py::arg("lse"), py::arg("thr"), py::arg("scale"), py::arg("is_causal") = false);
  // Fused APA blend+softmax over precomputed bulk/rank score matrices.
  // Phase 3.1 (board item 4a): Lq<=0 (default) is the legacy sentinel path —
  // causal/window masking must already be baked into bulk/rank as
  // large-negative bias. Lq>0 is the index-arithmetic path — no mask tensor
  // needed; row0 = absolute query-chunk start, Lq = full query length,
  // window<=0 = full causal else sliding width. See tc/core.h for the exact
  // convention.
  m.def("apa_blend_softmax", [](Tensor& bulk, Tensor& rank, double zthr,
                                int64_t Lq, int64_t row0, int64_t window) {
    return Tensor::make(tc::apa_blend_softmax(bulk.data(), rank.data(),
                                              (float)zthr, nullptr,
                                              (int)Lq, row0, (int)window),
                        false);
  }, py::arg("bulk"), py::arg("rank"), py::arg("zthr"),
     py::arg("Lq") = 0, py::arg("row0") = 0, py::arg("window") = 0);
  m.def("apa_blend_softmax_sink", [](Tensor& bulk, Tensor& rank, Tensor& sinks,
                                     double zthr, int64_t Lq, int64_t row0,
                                     int64_t window) {
    return Tensor::make(
        tc::apa_blend_softmax_sink(bulk.data(), rank.data(), sinks.data(),
                                   (float)zthr, (int)Lq, row0, (int)window),
        false);
  }, py::arg("bulk"), py::arg("rank"), py::arg("sinks"), py::arg("zthr"),
     py::arg("Lq") = 0, py::arg("row0") = 0, py::arg("window") = 0);
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
  m.def("checkpoint", &checkpoint_py, py::arg("fn"), py::arg("inputs"));
  m.def("synchronize", &cuda_sync);
  m.def("empty_cache", &empty_cache);
  m.def("set_alloc_pooling", &set_alloc_pooling);
}
