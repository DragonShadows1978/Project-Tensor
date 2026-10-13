// _tc_signal P1T-M4 entry points: T5 norms (instance_norm1d, layer_norm, group_norm) and T6 small ops
// (upsample_nearest1d, reflection_pad1d, rint). Registered into M2's module by `register_m4(m)` (one call in
// bindings.cpp), so the package stays one .so with one DLPack consumer contract.
//
// Same contract as M2 (src/bindings.cpp): every operand, the output included, is a DLPack producer or capsule; this file
// allocates no device memory. Two calls per op:
//   *_plan(...) -> acquires the input capsules (__dlpack__(stream=1)), PEEKS them (device/dtype/layout/rank/shape and
//                  the attributes are checked without consuming), returns (capsules..., out_shape);
//   *_run(...)  -> consumes every capsule, re-derives the geometry from the consumed shapes, checks the output's shape,
//                  the common device and that the output overlaps no input, launches on the legacy default stream, and
//                  calls every producer deleter when the operands go out of scope.
//
// Prior art: the DLPack consumer and the plan/run split are M2's (see dlpack_operand.h and bindings.cpp: DLPack v1.3,
// the Python array API stream protocol, the cuDNN/cuBLASLt "size, then execute" pattern), reused here unchanged. The
// ONNX operator specs define the attribute rules (LayerNormalization axis in [-r, r); GroupNormalization num_groups
// dividing C; Pad reflect needs pads < T; Resize nearest at an integer scale). Ours: the row-norm geometry mapping
// (rows x L plus an affine index rule) shared by the three norms; no prior art known to me for that mapping as such.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "dlpack_operand.h"
#include "m4_ops.h"

namespace py = pybind11;

namespace tcs {
namespace {

[[noreturn]] void bad_attr4(const std::string& msg) { throw py::value_error("tc_signal: " + msg); }

[[noreturn]] void bad_operand4(const char* role, const std::string& msg) {
  throw py::value_error(std::string("tc_signal: operand '") + role + "': " + msg);
}

// ------------------------------------------------------------------------------------------------ peek / consume
struct Peek4 {
  py::object capsule = py::none();  // None for an absent optional operand
  std::vector<int64_t> shape;
  int device = -1;
  bool present = false;
};

// rank >= 0: exactly that rank; rank < 0: any rank >= min_rank.
void check_rank(const char* role, int ndim, int rank, int min_rank) {
  if (rank < 0 && ndim < min_rank) {
    std::ostringstream os;
    os << "expected at least " << min_rank << " dimension" << (min_rank == 1 ? "" : "s") << ", got " << ndim;
    bad_operand4(role, os.str());
  }
}

Peek4 peek4(py::handle obj, const char* role, int rank, int min_rank) {
  Peek4 p;
  if (obj.is_none()) return p;
  p.capsule = to_capsule(obj, role, false);
  const DLTensor* t = peek(p.capsule, role);
  validate_tensor(t, role, rank >= 0 ? rank : t->ndim);  // device, dtype, (rank), non-empty, C-contiguous, aligned
  check_rank(role, t->ndim, rank, min_rank);
  p.shape = shape_of(t);
  p.device = t->device.device_id;
  p.present = true;
  return p;
}

void same_device4(const std::vector<std::pair<const Peek4*, const char*>>& ps) {
  int dev = -1;
  const char* first = "";
  for (const auto& pr : ps) {
    if (!pr.first->present) continue;
    if (dev < 0) {
      dev = pr.first->device;
      first = pr.second;
    } else if (pr.first->device != dev) {
      std::ostringstream os;
      os << "on CUDA device " << pr.first->device << " but operand '" << first << "' is on CUDA device " << dev
         << "; every operand must be on the same device";
      bad_operand4(pr.second, os.str());
    }
  }
}

// Consumes a capsule returned by a *_plan call (its rank is read from the capsule, then fully re-validated).
Operand consume4(py::handle cap, const char* role, int rank, int min_rank) {
  if (cap.is_none()) return Operand();
  const DLTensor* t = peek(cap, role);
  const int nd = t->ndim;
  Operand op(cap, role, rank >= 0 ? rank : nd, false);
  check_rank(role, op.ndim(), rank, min_rank);
  return op;
}

std::vector<int64_t> shape4(const Operand& op) {
  std::vector<int64_t> s(op.ndim());
  for (int i = 0; i < op.ndim(); ++i) s[i] = op.dim(i);
  return s;
}

py::tuple tuple4(const std::vector<int64_t>& v) {
  py::tuple t(v.size());
  for (size_t i = 0; i < v.size(); ++i) t[i] = py::int_(v[i]);
  return t;
}

int64_t prod(const std::vector<int64_t>& s, size_t a, size_t b) {
  int64_t p = 1;
  for (size_t i = a; i < b; ++i) p *= s[i];
  return p;
}

// ------------------------------------------------------------------------------------------------ norms: plan
enum class NormKind { kInstance, kLayer, kGroup };

const char* norm_name(NormKind k) {
  return k == NormKind::kInstance ? "instance_norm1d" : (k == NormKind::kLayer ? "layer_norm" : "group_norm");
}

struct NormPlan {
  RowNormGeom g;
  std::vector<int64_t> out_shape;
  std::vector<int64_t> param_shape;
};

float checked_eps(const char* op, double eps) {
  const float e = static_cast<float>(eps);
  if (!std::isfinite(eps) || eps < 0.0 || !std::isfinite(e)) {
    std::ostringstream os;
    os << op << ": eps must be a finite number >= 0 that fits fp32, got " << eps;
    bad_attr4(os.str());
  }
  return e;
}

// `arg` is the axis (layer_norm) or num_groups (group_norm); unused for instance_norm1d.
NormPlan plan_norm(NormKind kind, const std::vector<int64_t>& xs, const std::vector<int64_t>* ss,
                   const std::vector<int64_t>* bs, int64_t arg, double eps) {
  const char* op = norm_name(kind);
  NormPlan p{};
  RowNormGeom& g = p.g;
  g.eps = checked_eps(op, eps);
  g.channels = g.groups = g.ch_per_group = g.T = 1;
  const int64_t r = static_cast<int64_t>(xs.size());
  std::ostringstream os;
  if (kind == NormKind::kInstance || kind == NormKind::kGroup) {
    if (r != 3) {
      os << op << ": x must be [N, C, T], got " << r << " dimensions";
      bad_attr4(os.str());
    }
    const int64_t N = xs[0], C = xs[1], T = xs[2];
    p.param_shape = {C};
    if (kind == NormKind::kInstance) {
      g.rows = N * C;
      g.L = T;
      g.affine = NormAffine::kPerRow;
      g.channels = C;
    } else {
      if (arg < 1) {
        os << "group_norm: num_groups must be >= 1, got " << arg;
        bad_attr4(os.str());
      }
      if (C % arg != 0) {
        os << "group_norm: C (" << C << ") is not divisible by num_groups (" << arg << ")";
        bad_attr4(os.str());
      }
      g.rows = N * arg;
      g.L = (C / arg) * T;
      g.affine = NormAffine::kPerChannel;
      g.groups = arg;
      g.ch_per_group = C / arg;
      g.T = T;
    }
  } else {
    if (r < 1) {
      os << "layer_norm: x must have at least 1 dimension, got " << r;
      bad_attr4(os.str());
    }
    if (arg < -r || arg >= r) {
      os << "layer_norm: axis must be in [" << -r << ", " << r - 1 << "] for a " << r << "-D x, got " << arg;
      bad_attr4(os.str());
    }
    const int64_t a = arg < 0 ? arg + r : arg;
    g.rows = prod(xs, 0, static_cast<size_t>(a));
    g.L = prod(xs, static_cast<size_t>(a), xs.size());
    g.affine = NormAffine::kPerElem;
    p.param_shape.assign(xs.begin() + a, xs.end());
  }
  for (const auto& pr : {std::make_pair(ss, "scale"), std::make_pair(bs, "bias")}) {
    if (pr.first != nullptr && *pr.first != p.param_shape) {
      std::ostringstream o2;
      o2 << op << ": " << pr.second << " has shape " << shape_str(*pr.first) << ", expected "
         << shape_str(p.param_shape)
         << (kind == NormKind::kLayer ? " (x.shape[axis:])" : (kind == NormKind::kGroup ? " ([C], per channel)" : " ([C])"));
      bad_attr4(o2.str());
    }
  }
  p.out_shape = xs;
  return p;
}

int x_rank(NormKind kind) { return kind == NormKind::kLayer ? -1 : 3; }

py::tuple norm_plan(NormKind kind, py::object x, py::object scale, py::object bias, int64_t arg, double eps) {
  Peek4 X = peek4(x, "x", x_rank(kind), 1);
  Peek4 S = peek4(scale, "scale", -1, 1);
  Peek4 B = peek4(bias, "bias", -1, 1);
  same_device4({{&X, "x"}, {&S, "scale"}, {&B, "bias"}});
  NormPlan p = plan_norm(kind, X.shape, S.present ? &S.shape : nullptr, B.present ? &B.shape : nullptr, arg, eps);
  return py::make_tuple(X.capsule, S.capsule, B.capsule, tuple4(p.out_shape));
}

void norm_run(NormKind kind, py::object xc, py::object sc, py::object bc, py::object out, int64_t arg, double eps) {
  Operand X = consume4(xc, "x", x_rank(kind), 1);
  Operand S = consume4(sc, "scale", -1, 1);
  Operand B = consume4(bc, "bias", -1, 1);
  std::vector<int64_t> sshape, bshape;
  if (S.present()) sshape = shape4(S);
  if (B.present()) bshape = shape4(B);
  NormPlan p = plan_norm(kind, shape4(X), S.present() ? &sshape : nullptr, B.present() ? &bshape : nullptr, arg, eps);
  Operand Y(to_capsule(out, "out", true), "out", X.ndim(), true);
  expect_shape(Y, p.out_shape);
  const int device = common_device({&X, &S, &B, &Y});
  for (const Operand* in : {&X, &S, &B}) expect_disjoint(Y, *in);
  row_norm_forward(device, p.g, X.data(), S.present() ? S.data() : nullptr, B.present() ? B.data() : nullptr,
                   Y.data());
  // Leaving scope runs each Operand's destructor: every producer deleter is called after the launch.
}

// ------------------------------------------------------------------------------------------------ small ops: plan
std::vector<int64_t> upsample_shape(const std::vector<int64_t>& xs, int64_t scale) {
  if (scale < 1) {
    std::ostringstream os;
    os << "upsample_nearest1d: scale must be an integer >= 1, got " << scale;
    bad_attr4(os.str());
  }
  const int64_t T = xs.back();
  const int64_t lim = std::numeric_limits<int64_t>::max();
  if (scale > lim / T || prod(xs, 0, xs.size() - 1) > lim / (T * scale)) {
    std::ostringstream os;
    os << "upsample_nearest1d: output size overflows int64 (T " << T << " x scale " << scale << ")";
    bad_attr4(os.str());
  }
  std::vector<int64_t> o = xs;
  o.back() = T * scale;
  return o;
}

std::vector<int64_t> pad_shape(const std::vector<int64_t>& xs, int64_t pb, int64_t pe) {
  const int64_t T = xs.back();
  std::ostringstream os;
  if (pb < 0 || pe < 0) {
    os << "reflection_pad1d: pads must be >= 0, got (" << pb << ", " << pe << ")";
    bad_attr4(os.str());
  }
  if (pb >= T || pe >= T) {
    os << "reflection_pad1d: each pad must be < T (" << T << ") for reflect padding (ONNX/torch), got (" << pb << ", "
       << pe << ")";
    bad_attr4(os.str());
  }
  std::vector<int64_t> o = xs;
  o.back() = T + pb + pe;
  return o;
}

py::tuple upsample_plan(py::object x, int64_t scale) {
  Peek4 X = peek4(x, "x", -1, 1);
  return py::make_tuple(X.capsule, tuple4(upsample_shape(X.shape, scale)));
}

void upsample_run(py::object xc, py::object out, int64_t scale) {
  Operand X = consume4(xc, "x", -1, 1);
  const std::vector<int64_t> xs = shape4(X);
  const std::vector<int64_t> os = upsample_shape(xs, scale);
  Operand Y(to_capsule(out, "out", true), "out", X.ndim(), true);
  expect_shape(Y, os);
  const int device = common_device({&X, &Y});
  expect_disjoint(Y, X);
  upsample_nearest1d_forward(device, prod(xs, 0, xs.size() - 1), xs.back(), scale, X.data(), Y.data());
}

py::tuple pad_plan(py::object x, int64_t pb, int64_t pe) {
  Peek4 X = peek4(x, "x", -1, 1);
  return py::make_tuple(X.capsule, tuple4(pad_shape(X.shape, pb, pe)));
}

void pad_run(py::object xc, py::object out, int64_t pb, int64_t pe) {
  Operand X = consume4(xc, "x", -1, 1);
  const std::vector<int64_t> xs = shape4(X);
  const std::vector<int64_t> os = pad_shape(xs, pb, pe);
  Operand Y(to_capsule(out, "out", true), "out", X.ndim(), true);
  expect_shape(Y, os);
  const int device = common_device({&X, &Y});
  expect_disjoint(Y, X);
  reflection_pad1d_forward(device, prod(xs, 0, xs.size() - 1), xs.back(), pb, pe, X.data(), Y.data());
}

py::tuple rint_plan(py::object x) {
  Peek4 X = peek4(x, "x", -1, 0);
  return py::make_tuple(X.capsule, tuple4(X.shape));
}

void rint_run(py::object xc, py::object out) {
  Operand X = consume4(xc, "x", -1, 0);
  const std::vector<int64_t> xs = shape4(X);
  Operand Y(to_capsule(out, "out", true), "out", X.ndim(), true);
  expect_shape(Y, xs);
  const int device = common_device({&X, &Y});
  expect_disjoint(Y, X);
  rint_forward(device, X.numel(), X.data(), Y.data());
}

}  // namespace

void register_m4(py::module_& m) {
  m.def(
      "instance_norm1d_plan",
      [](py::object x, py::object s, py::object b, double eps) { return norm_plan(NormKind::kInstance, x, s, b, 0, eps); },
      py::arg("x"), py::arg("scale"), py::arg("bias"), py::arg("eps"),
      "Acquire and peek x/scale/bias; return (x_capsule, scale_capsule|None, bias_capsule|None, out_shape).");
  m.def(
      "instance_norm1d_run",
      [](py::object xc, py::object sc, py::object bc, py::object out, double eps) {
        norm_run(NormKind::kInstance, xc, sc, bc, out, 0, eps);
      },
      py::arg("x_capsule"), py::arg("scale_capsule"), py::arg("bias_capsule"), py::arg("out"), py::arg("eps"),
      "Consume every operand, validate, launch on the legacy stream, release.");
  m.def(
      "layer_norm_plan",
      [](py::object x, py::object s, py::object b, int64_t axis, double eps) {
        return norm_plan(NormKind::kLayer, x, s, b, axis, eps);
      },
      py::arg("x"), py::arg("scale"), py::arg("bias"), py::arg("axis"), py::arg("eps"),
      "Acquire and peek x/scale/bias; return (x_capsule, scale_capsule|None, bias_capsule|None, out_shape).");
  m.def(
      "layer_norm_run",
      [](py::object xc, py::object sc, py::object bc, py::object out, int64_t axis, double eps) {
        norm_run(NormKind::kLayer, xc, sc, bc, out, axis, eps);
      },
      py::arg("x_capsule"), py::arg("scale_capsule"), py::arg("bias_capsule"), py::arg("out"), py::arg("axis"),
      py::arg("eps"), "Consume every operand, validate, launch on the legacy stream, release.");
  m.def(
      "group_norm_plan",
      [](py::object x, int64_t groups, py::object s, py::object b, double eps) {
        return norm_plan(NormKind::kGroup, x, s, b, groups, eps);
      },
      py::arg("x"), py::arg("num_groups"), py::arg("scale"), py::arg("bias"), py::arg("eps"),
      "Acquire and peek x/scale/bias; return (x_capsule, scale_capsule|None, bias_capsule|None, out_shape).");
  m.def(
      "group_norm_run",
      [](py::object xc, int64_t groups, py::object sc, py::object bc, py::object out, double eps) {
        norm_run(NormKind::kGroup, xc, sc, bc, out, groups, eps);
      },
      py::arg("x_capsule"), py::arg("num_groups"), py::arg("scale_capsule"), py::arg("bias_capsule"), py::arg("out"),
      py::arg("eps"), "Consume every operand, validate, launch on the legacy stream, release.");
  m.def("upsample_nearest1d_plan", &upsample_plan, py::arg("x"), py::arg("scale"),
        "Acquire and peek x; return (x_capsule, out_shape).");
  m.def("upsample_nearest1d_run", &upsample_run, py::arg("x_capsule"), py::arg("out"), py::arg("scale"),
        "Consume, validate, launch on the legacy stream, release.");
  m.def("reflection_pad1d_plan", &pad_plan, py::arg("x"), py::arg("pad_begin"), py::arg("pad_end"),
        "Acquire and peek x; return (x_capsule, out_shape).");
  m.def("reflection_pad1d_run", &pad_run, py::arg("x_capsule"), py::arg("out"), py::arg("pad_begin"),
        py::arg("pad_end"), "Consume, validate, launch on the legacy stream, release.");
  m.def("rint_plan", &rint_plan, py::arg("x"), "Acquire and peek x; return (x_capsule, out_shape).");
  m.def("rint_run", &rint_run, py::arg("x_capsule"), py::arg("out"),
        "Consume, validate, launch on the legacy stream, release.");
  m.def("m4_tuning", []() {
    py::dict d;
    d["norm_block"] = m4tuning::kNormBlock;
    d["norm_chunk"] = m4tuning::kNormChunk;
    d["copy_block"] = m4tuning::kCopyBlock;
    return d;
  }, "The M4 kernels' tunables (the norm statistics' summation order depends on norm_block and norm_chunk).");
}

}  // namespace tcs
