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
//                  hands every consumed manager to the deferred-release list (P1T-FIX1 F01, dlpack_operand.h).
// P1T-FIX1: F02 (x is REQUIRED in all six ops: None -> TypeError before planning, in peek4 and consume4; scale and bias
// are the only optional operands), F03 (rows, L, T*scale, T + pads and every output size checked, checked_math.h).
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
#include <stdexcept>
#include <string>
#include <vector>

#include "checked_math.h"
#include "dlpack_operand.h"
#include "py_args.h"  // P1T-FIX6 sweep (g): C-level argument matching and conversion
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
  CapsuleRef capsule = CapsuleRef(py::none());  // None when absent; P1T-FIX5: admitted last-reference drop
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

enum class Need { kRequired, kOptional };

Peek4 peek4(py::handle obj, const char* role, int rank, int min_rank, Need need) {
  Peek4 p;
  if (need == Need::kRequired) require_operand(obj, role);  // F02: None is refused here, before any planning
  if (obj.is_none()) return p;
  p.capsule = to_capsule(obj, role, false);
  const DLTensor* t = peek(p.capsule.handle(), role);
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
Operand consume4(py::handle cap, const char* role, int rank, int min_rank, Need need) {
  if (need == Need::kRequired) require_operand(cap, role);  // F02
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

// F03: checked product (checked_math.h).
int64_t prod(const std::vector<int64_t>& s, size_t a, size_t b, const char* what) { return ck_prod(s, a, b, what); }

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
      g.rows = ck_mul(N, C, "instance_norm1d: rows N*C");
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
      g.rows = ck_mul(N, arg, "group_norm: rows N*num_groups");
      g.L = ck_mul(C / arg, T, "group_norm: row length (C/num_groups)*T");
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
    g.rows = prod(xs, 0, static_cast<size_t>(a), "layer_norm: rows");
    g.L = prod(xs, static_cast<size_t>(a), xs.size(), "layer_norm: row length");
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
  ck_shape_numel(p.out_shape, "norm: output size");
  ck_mul(g.rows, g.L, "norm: rows*L");
  return p;
}

int x_rank(NormKind kind) { return kind == NormKind::kLayer ? -1 : 3; }

py::tuple norm_plan(NormKind kind, py::object x, py::object scale, py::object bias, int64_t arg, double eps) {
  drain_completed();
  Peek4 X = peek4(x, "x", x_rank(kind), 1, Need::kRequired);
  Peek4 S = peek4(scale, "scale", -1, 1, Need::kOptional);
  Peek4 B = peek4(bias, "bias", -1, 1, Need::kOptional);
  same_device4({{&X, "x"}, {&S, "scale"}, {&B, "bias"}});
  NormPlan p = plan_norm(kind, X.shape, S.present ? &S.shape : nullptr, B.present ? &B.shape : nullptr, arg, eps);
  return py::make_tuple(X.capsule.object(), S.capsule.object(), B.capsule.object(), tuple4(p.out_shape));
}

void norm_run(NormKind kind, py::object xc, py::object sc, py::object bc, py::object out, int64_t arg, double eps) {
  drain_completed();
  require_operand(out, "out");
  Operand X = consume4(xc, "x", x_rank(kind), 1, Need::kRequired);
  Operand S = consume4(sc, "scale", -1, 1, Need::kOptional);
  Operand B = consume4(bc, "bias", -1, 1, Need::kOptional);
  std::vector<int64_t> sshape, bshape;
  if (S.present()) sshape = shape4(S);
  if (B.present()) bshape = shape4(B);
  NormPlan p = plan_norm(kind, shape4(X), S.present() ? &sshape : nullptr, B.present() ? &bshape : nullptr, arg, eps);
  Operand Y(to_capsule(out, "out", true).handle(), "out", X.ndim(), true);
  expect_shape(Y, p.out_shape);
  const int device = common_device({&X, &S, &B, &Y});
  for (const Operand* in : {&X, &S, &B}) expect_disjoint(Y, *in);
  RetainUntilComplete keep(device, {&X, &S, &B, &Y});  // F01: deleters run after the legacy stream passes the launch
  row_norm_forward(device, p.g, X.data(), S.present() ? S.data() : nullptr, B.present() ? B.data() : nullptr,
                   Y.data());
}

// ------------------------------------------------------------------------------------------------ small ops: plan
std::vector<int64_t> upsample_shape(const std::vector<int64_t>& xs, int64_t scale) {
  if (scale < 1) {
    std::ostringstream os;
    os << "upsample_nearest1d: scale must be an integer >= 1, got " << scale;
    bad_attr4(os.str());
  }
  const int64_t T = xs.back();
  std::vector<int64_t> o = xs;
  o.back() = ck_mul(T, scale, "upsample_nearest1d: output length T*scale");  // F03
  ck_shape_numel(o, "upsample_nearest1d: output size");
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
  // F03: T + pb + pe checked (the review's CPU reproduction returned a negative extent here), then the output size.
  o.back() = ck_add(ck_add(T, pb, "reflection_pad1d: output length"), pe, "reflection_pad1d: output length");
  ck_shape_numel(o, "reflection_pad1d: output size");
  return o;
}

py::tuple upsample_plan(py::object x, int64_t scale) {
  drain_completed();
  Peek4 X = peek4(x, "x", -1, 1, Need::kRequired);
  return py::make_tuple(X.capsule.object(), tuple4(upsample_shape(X.shape, scale)));
}

void upsample_run(py::object xc, py::object out, int64_t scale) {
  drain_completed();
  require_operand(out, "out");
  Operand X = consume4(xc, "x", -1, 1, Need::kRequired);
  const std::vector<int64_t> xs = shape4(X);
  const std::vector<int64_t> os = upsample_shape(xs, scale);
  Operand Y(to_capsule(out, "out", true).handle(), "out", X.ndim(), true);
  expect_shape(Y, os);
  const int device = common_device({&X, &Y});
  expect_disjoint(Y, X);
  const int64_t rows = prod(xs, 0, xs.size() - 1, "upsample_nearest1d: rows");
  RetainUntilComplete keep(device, {&X, &Y});  // F01
  upsample_nearest1d_forward(device, rows, xs.back(), scale, X.data(), Y.data());
}

py::tuple pad_plan(py::object x, int64_t pb, int64_t pe) {
  drain_completed();
  Peek4 X = peek4(x, "x", -1, 1, Need::kRequired);
  return py::make_tuple(X.capsule.object(), tuple4(pad_shape(X.shape, pb, pe)));
}

void pad_run(py::object xc, py::object out, int64_t pb, int64_t pe) {
  drain_completed();
  require_operand(out, "out");
  Operand X = consume4(xc, "x", -1, 1, Need::kRequired);
  const std::vector<int64_t> xs = shape4(X);
  const std::vector<int64_t> os = pad_shape(xs, pb, pe);
  Operand Y(to_capsule(out, "out", true).handle(), "out", X.ndim(), true);
  expect_shape(Y, os);
  const int device = common_device({&X, &Y});
  expect_disjoint(Y, X);
  const int64_t rows = prod(xs, 0, xs.size() - 1, "reflection_pad1d: rows");
  RetainUntilComplete keep(device, {&X, &Y});  // F01
  reflection_pad1d_forward(device, rows, xs.back(), pb, pe, X.data(), Y.data());
}

py::tuple rint_plan(py::object x) {
  drain_completed();
  Peek4 X = peek4(x, "x", -1, 0, Need::kRequired);
  return py::make_tuple(X.capsule.object(), tuple4(X.shape));
}

void rint_run(py::object xc, py::object out) {
  drain_completed();
  require_operand(out, "out");
  Operand X = consume4(xc, "x", -1, 0, Need::kRequired);
  const std::vector<int64_t> xs = shape4(X);
  Operand Y(to_capsule(out, "out", true).handle(), "out", X.ndim(), true);
  expect_shape(Y, xs);
  const int device = common_device({&X, &Y});
  expect_disjoint(Y, X);
  RetainUntilComplete keep(device, {&X, &Y});  // F01
  rint_forward(device, X.numel(), X.data(), Y.data());
}

// TEST HOOK (P1T-FIX1, F01 regression test only; not part of the API): rint_run whose call then throws AFTER its
// launch, so the test can witness that a partial-launch exception path keeps the consumed managers alive until the
// legacy stream has passed the launch (RetainUntilComplete's destructor runs during unwinding). No prior art claimed.
void test_rint_then_raise(py::object xc, py::object out) {
  drain_completed();
  require_operand(out, "out");
  Operand X = consume4(xc, "x", -1, 0, Need::kRequired);
  const std::vector<int64_t> xs = shape4(X);
  Operand Y(to_capsule(out, "out", true).handle(), "out", X.ndim(), true);
  expect_shape(Y, xs);
  const int device = common_device({&X, &Y});
  expect_disjoint(Y, X);
  RetainUntilComplete keep(device, {&X, &Y});
  rint_forward(device, X.numel(), X.data(), Y.data());
  throw std::runtime_error("tc_signal._test_rint_then_raise: injected failure after the launch (test hook)");
}

}  // namespace

void register_m4(py::module_& m) {
  // P1T-FIX6 (sweep (g)): (*args, **kwargs) + C-level conversion (py_args.h); first docstring line = the signature.
  m.def(
      "instance_norm1d_plan",
      [](py::args a, py::kwargs k) {
        const CallArgs c("instance_norm1d_plan", a, k, {{"x"}, {"scale"}, {"bias"}, {"eps"}});
        const double eps = arg_real(c, 3);
        return norm_plan(NormKind::kInstance, c.object(0), c.object(1), c.object(2), 0, eps);
      },
      "instance_norm1d_plan(x, scale, bias, eps)\n\n"
      "Acquire and peek x/scale/bias; return (x_capsule, scale_capsule|None, bias_capsule|None, out_shape).");
  m.def(
      "instance_norm1d_run",
      [](py::args a, py::kwargs k) {
        const CallArgs c("instance_norm1d_run", a, k,
                         {{"x_capsule"}, {"scale_capsule"}, {"bias_capsule"}, {"out"}, {"eps"}});
        const double eps = arg_real(c, 4);
        norm_run(NormKind::kInstance, c.object(0), c.object(1), c.object(2), c.object(3), 0, eps);
      },
      "instance_norm1d_run(x_capsule, scale_capsule, bias_capsule, out, eps)\n\n"
      "Consume every operand, validate, launch on the legacy stream, release.");
  m.def(
      "layer_norm_plan",
      [](py::args a, py::kwargs k) {
        const CallArgs c("layer_norm_plan", a, k, {{"x"}, {"scale"}, {"bias"}, {"axis"}, {"eps"}});
        const int64_t axis = arg_int64(c, 3);
        const double eps = arg_real(c, 4);
        return norm_plan(NormKind::kLayer, c.object(0), c.object(1), c.object(2), axis, eps);
      },
      "layer_norm_plan(x, scale, bias, axis, eps)\n\n"
      "Acquire and peek x/scale/bias; return (x_capsule, scale_capsule|None, bias_capsule|None, out_shape).");
  m.def(
      "layer_norm_run",
      [](py::args a, py::kwargs k) {
        const CallArgs c("layer_norm_run", a, k,
                         {{"x_capsule"}, {"scale_capsule"}, {"bias_capsule"}, {"out"}, {"axis"}, {"eps"}});
        const int64_t axis = arg_int64(c, 4);
        const double eps = arg_real(c, 5);
        norm_run(NormKind::kLayer, c.object(0), c.object(1), c.object(2), c.object(3), axis, eps);
      },
      "layer_norm_run(x_capsule, scale_capsule, bias_capsule, out, axis, eps)\n\n"
      "Consume every operand, validate, launch on the legacy stream, release.");
  m.def(
      "group_norm_plan",
      [](py::args a, py::kwargs k) {
        const CallArgs c("group_norm_plan", a, k, {{"x"}, {"num_groups"}, {"scale"}, {"bias"}, {"eps"}});
        const int64_t groups = arg_int64(c, 1);
        const double eps = arg_real(c, 4);
        return norm_plan(NormKind::kGroup, c.object(0), c.object(2), c.object(3), groups, eps);
      },
      "group_norm_plan(x, num_groups, scale, bias, eps)\n\n"
      "Acquire and peek x/scale/bias; return (x_capsule, scale_capsule|None, bias_capsule|None, out_shape).");
  m.def(
      "group_norm_run",
      [](py::args a, py::kwargs k) {
        const CallArgs c("group_norm_run", a, k,
                         {{"x_capsule"}, {"num_groups"}, {"scale_capsule"}, {"bias_capsule"}, {"out"}, {"eps"}});
        const int64_t groups = arg_int64(c, 1);
        const double eps = arg_real(c, 5);
        norm_run(NormKind::kGroup, c.object(0), c.object(2), c.object(3), c.object(4), groups, eps);
      },
      "group_norm_run(x_capsule, num_groups, scale_capsule, bias_capsule, out, eps)\n\n"
      "Consume every operand, validate, launch on the legacy stream, release.");
  m.def(
      "upsample_nearest1d_plan",
      [](py::args a, py::kwargs k) {
        const CallArgs c("upsample_nearest1d_plan", a, k, {{"x"}, {"scale"}});
        const int64_t scale = arg_int64(c, 1);
        return upsample_plan(c.object(0), scale);
      },
      "upsample_nearest1d_plan(x, scale)\n\nAcquire and peek x; return (x_capsule, out_shape).");
  m.def(
      "upsample_nearest1d_run",
      [](py::args a, py::kwargs k) {
        const CallArgs c("upsample_nearest1d_run", a, k, {{"x_capsule"}, {"out"}, {"scale"}});
        const int64_t scale = arg_int64(c, 2);
        upsample_run(c.object(0), c.object(1), scale);
      },
      "upsample_nearest1d_run(x_capsule, out, scale)\n\nConsume, validate, launch on the legacy stream, release.");
  m.def(
      "reflection_pad1d_plan",
      [](py::args a, py::kwargs k) {
        const CallArgs c("reflection_pad1d_plan", a, k, {{"x"}, {"pad_begin"}, {"pad_end"}});
        const int64_t pb = arg_int64(c, 1), pe = arg_int64(c, 2);
        return pad_plan(c.object(0), pb, pe);
      },
      "reflection_pad1d_plan(x, pad_begin, pad_end)\n\nAcquire and peek x; return (x_capsule, out_shape).");
  m.def(
      "reflection_pad1d_run",
      [](py::args a, py::kwargs k) {
        const CallArgs c("reflection_pad1d_run", a, k, {{"x_capsule"}, {"out"}, {"pad_begin"}, {"pad_end"}});
        const int64_t pb = arg_int64(c, 2), pe = arg_int64(c, 3);
        pad_run(c.object(0), c.object(1), pb, pe);
      },
      "reflection_pad1d_run(x_capsule, out, pad_begin, pad_end)\n\n"
      "Consume, validate, launch on the legacy stream, release.");
  m.def(
      "rint_plan",
      [](py::args a, py::kwargs k) {
        const CallArgs c("rint_plan", a, k, {{"x"}});
        return rint_plan(c.object(0));
      },
      "rint_plan(x)\n\nAcquire and peek x; return (x_capsule, out_shape).");
  m.def(
      "rint_run",
      [](py::args a, py::kwargs k) {
        const CallArgs c("rint_run", a, k, {{"x_capsule"}, {"out"}});
        rint_run(c.object(0), c.object(1));
      },
      "rint_run(x_capsule, out)\n\nConsume, validate, launch on the legacy stream, release.");
  m.def(
      "_test_rint_then_raise",
      [](py::args a, py::kwargs k) {
        const CallArgs c("_test_rint_then_raise", a, k, {{"x_capsule"}, {"out"}});
        test_rint_then_raise(c.object(0), c.object(1));
      },
      "_test_rint_then_raise(x_capsule, out)\n\n"
      "TEST HOOK (P1T-FIX1 F01): rint_run, then raise RuntimeError after the launch.");
  m.def(
      "m4_tuning",
      [](py::args a, py::kwargs k) {
        const CallArgs c("m4_tuning", a, k, {});
        (void)c;
        py::dict d;
        d["norm_block"] = m4tuning::kNormBlock;
        d["norm_chunk"] = m4tuning::kNormChunk;
        d["copy_block"] = m4tuning::kCopyBlock;
        return d;
      },
      "m4_tuning()\n\n"
      "The M4 kernels' tunables (the norm statistics' summation order depends on norm_block and norm_chunk).");
}

}  // namespace tcs
