// _tc_signal: the C++ half of the tc_signal add-on (P1T-M2). Every operand, outputs and scratch included, is a DLPack
// producer or capsule; this module allocates no device memory (the Python layer allocates outputs and the im2col
// scratch through tc_signal's allocator and passes them in).
//
// Two calls per op, so the Python layer can size and allocate before anything is consumed:
//   *_plan(x, w, b, attrs)  -> acquires the input capsules (__dlpack__(stream=1)), PEEKS their DLTensors (validates
//                              device/dtype/layout/shapes/attributes without consuming), and returns
//                              (x_capsule, w_capsule, b_capsule|None, out_shape[, scratch_numel]);
//   *_run(capsules, out[, scratch], attrs) -> consumes every capsule, re-validates, checks devices and overlap,
//                              launches on the legacy default stream, then hands every consumed manager to the
//                              deferred-release list (P1T-FIX1 F01, dlpack_operand.h RetainUntilComplete): the producer
//                              deleters run once the legacy stream has passed the launches, never before.
// P1T-FIX1 also adds: F02 (a required operand passed as None -> TypeError before planning), F03 (every size below is
// computed with checked arithmetic, checked_math.h), and drain_completed() first thing in every entry point.
//
// Prior art: see dlpack_operand.h (DLPack, Python array API stream protocol) and conv.cu (im2col, gather-form
// ConvTranspose, ONNX semantics). The plan/run split is ours (no prior art known to me for this exact split; it is the
// ordinary "query workspace size, then execute" pattern of cuDNN/cuBLASLt workspace APIs applied to DLPack operands).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <climits>
#include <sstream>
#include <string>
#include <vector>

#include "checked_math.h"
#include "conv_ops.h"
#include "dlpack_operand.h"
#include "py_args.h"  // P1T-FIX6 sweep (g): C-level argument matching and conversion
#include "tuning.h"

namespace py = pybind11;

namespace tcs {
namespace {

[[noreturn]] void bad_attr(const std::string& msg) { throw py::value_error("tc_signal: " + msg); }

void check_attr(const char* name, int64_t v, int64_t lo) {
  if (v < lo || v > INT_MAX) {
    std::ostringstream os;
    os << name << " must be in [" << lo << ", " << INT_MAX << "], got " << v;
    bad_attr(os.str());
  }
}

void check_int_range(const char* what, int64_t v) {
  if (v > INT_MAX) {
    std::ostringstream os;
    os << what << " (" << v << ") exceeds the 32-bit range cuBLAS takes";
    bad_attr(os.str());
  }
}

// ------------------------------------------------------------------------------------------------ Conv1d plan
struct Conv1dPlan {
  Conv1dGeom g;
  std::vector<int64_t> out_shape;
  int64_t scratch_numel;
};

Conv1dPlan plan_conv1d(const std::vector<int64_t>& xs, const std::vector<int64_t>& ws, const std::vector<int64_t>* bs,
                       int64_t stride, int64_t pb, int64_t pe, int64_t dilation, int64_t groups) {
  check_attr("stride", stride, 1);
  check_attr("dilation", dilation, 1);
  check_attr("groups", groups, 1);
  check_attr("pad_begin", pb, 0);
  check_attr("pad_end", pe, 0);
  Conv1dPlan p{};
  Conv1dGeom& g = p.g;
  g.N = xs[0];
  g.C_in = xs[1];
  g.T = xs[2];
  g.C_out = ws[0];
  g.K = ws[2];
  g.stride = stride;
  g.pad_begin = pb;
  g.pad_end = pe;
  g.dilation = dilation;
  g.groups = groups;
  std::ostringstream os;
  if (g.C_in % groups != 0) {
    os << "conv1d: C_in (" << g.C_in << ") is not divisible by groups (" << groups << ")";
    bad_attr(os.str());
  }
  if (ws[1] != g.C_in / groups) {
    os << "conv1d: w.shape[1] (" << ws[1] << ") != C_in/groups (" << g.C_in / groups << "); w is [C_out, C_in/groups, K]";
    bad_attr(os.str());
  }
  if (g.C_out % groups != 0) {
    os << "conv1d: C_out (" << g.C_out << ") is not divisible by groups (" << groups << ")";
    bad_attr(os.str());
  }
  if (bs != nullptr && (bs->size() != 1 || (*bs)[0] != g.C_out)) {
    os << "conv1d: b has shape " << shape_str(*bs) << ", expected [" << g.C_out << "]";
    bad_attr(os.str());
  }
  // Integer arithmetic in int64, every step checked (F03: dimensions may come from synthetic descriptors). Python's
  // floor division and C++'s truncation differ for a negative numerator, so the sign is checked first.
  const int64_t dk = ck_mul(dilation, g.K - 1, "conv1d: dilation*(K-1)");
  int64_t num = ck_add(g.T, pb, "conv1d: T + pad_begin");
  num = ck_add(num, pe, "conv1d: T + pad_begin + pad_end");
  num = ck_sub(num, dk, "conv1d: output length numerator");
  num = ck_sub(num, 1, "conv1d: output length numerator");
  if (num < 0) {
    os << "conv1d: output length (T + pb + pe - dilation*(K-1) - 1)//stride + 1 = (" << g.T << " + " << pb << " + "
       << pe << " - " << dilation << "*" << (g.K - 1) << " - 1)//" << stride << " + 1 is <= 0";
    bad_attr(os.str());
  }
  g.T_out = num / stride + 1;
  g.direct = (g.K == 1 && stride == 1 && pb == 0 && pe == 0);
  const int64_t Og = g.C_out / groups;
  const int64_t CgK = (g.C_in / groups) * g.K;
  check_int_range("T_out", g.T_out);
  check_int_range("C_out/groups", Og);
  check_int_range("C_in/groups*K", CgK);
  p.out_shape = {g.N, g.C_out, g.T_out};
  ck_shape_numel(p.out_shape, "conv1d: output size N*C_out*T_out");
  p.scratch_numel =
      g.direct ? 0 : ck_mul(ck_mul(g.C_in, g.K, "conv1d: im2col rows C_in*K"), g.T_out, "conv1d: im2col scratch size");
  ck_fp32_bytes(p.scratch_numel, "conv1d: im2col scratch bytes");
  return p;
}

// ------------------------------------------------------------------------------------------------ ConvTranspose1d plan
struct ConvTranspose1dPlan {
  ConvTranspose1dGeom g;
  std::vector<int64_t> out_shape;
};

ConvTranspose1dPlan plan_conv_transpose1d(const std::vector<int64_t>& xs, const std::vector<int64_t>& ws,
                                          const std::vector<int64_t>* bs, int64_t stride, int64_t pb, int64_t pe,
                                          int64_t output_padding, int64_t dilation, int64_t groups) {
  check_attr("stride", stride, 1);
  check_attr("dilation", dilation, 1);
  check_attr("groups", groups, 1);
  check_attr("pad_begin", pb, 0);
  check_attr("pad_end", pe, 0);
  check_attr("output_padding", output_padding, 0);
  std::ostringstream os;
  if (output_padding >= std::max(stride, dilation)) {
    os << "conv_transpose1d: output_padding (" << output_padding << ") must be < max(stride, dilation) ("
       << std::max(stride, dilation) << ")";
    bad_attr(os.str());
  }
  ConvTranspose1dPlan p{};
  ConvTranspose1dGeom& g = p.g;
  g.N = xs[0];
  g.C_in = xs[1];
  g.T = xs[2];
  g.K = ws[2];
  g.stride = stride;
  g.pad_begin = pb;
  g.pad_end = pe;
  g.output_padding = output_padding;
  g.dilation = dilation;
  g.groups = groups;
  if (ws[0] != g.C_in) {
    os << "conv_transpose1d: w.shape[0] (" << ws[0] << ") != C_in (" << g.C_in
       << "); w is [C_in, C_out/groups, K] (ONNX layout)";
    bad_attr(os.str());
  }
  if (g.C_in % groups != 0) {
    os << "conv_transpose1d: C_in (" << g.C_in << ") is not divisible by groups (" << groups << ")";
    bad_attr(os.str());
  }
  g.C_out = ck_mul(ws[1], groups, "conv_transpose1d: C_out = w.shape[1]*groups");
  if (bs != nullptr && (bs->size() != 1 || (*bs)[0] != g.C_out)) {
    os << "conv_transpose1d: b has shape " << shape_str(*bs) << ", expected [" << g.C_out << "]";
    bad_attr(os.str());
  }
  // F03: every step checked.
  int64_t t_out = ck_mul(stride, g.T - 1, "conv_transpose1d: stride*(T-1)");
  t_out = ck_add(t_out, output_padding, "conv_transpose1d: output length");
  t_out = ck_add(t_out, ck_mul(dilation, g.K - 1, "conv_transpose1d: dilation*(K-1)"), "conv_transpose1d: output length");
  t_out = ck_add(t_out, 1, "conv_transpose1d: output length");
  t_out = ck_sub(t_out, pb, "conv_transpose1d: output length");
  g.T_out = ck_sub(t_out, pe, "conv_transpose1d: output length");
  if (g.T_out <= 0) {
    os << "conv_transpose1d: output length stride*(T-1) + output_padding + dilation*(K-1) + 1 - pb - pe = "
       << g.T_out << " is <= 0";
    bad_attr(os.str());
  }
  p.out_shape = {g.N, g.C_out, g.T_out};
  ck_shape_numel(p.out_shape, "conv_transpose1d: output size N*C_out*T_out");
  return p;
}

// ------------------------------------------------------------------------------------------------ peek helpers
struct Peeked {
  CapsuleRef capsule;            // P1T-FIX5: admitted last-reference drop (dlpack_operand.h); empty when absent
  std::vector<int64_t> shape;
  int device = -1;
};

Peeked peek_operand(py::handle obj, const char* role, int ndim) {
  Peeked p;
  require_operand(obj, role);
  p.capsule = to_capsule(obj, role, false);
  const DLTensor* t = peek(p.capsule.handle(), role);
  validate_tensor(t, role, ndim);
  p.shape = shape_of(t);
  p.device = t->device.device_id;
  return p;
}

void same_device_peeked(const std::vector<const Peeked*>& ps, const std::vector<const char*>& roles) {
  int dev = -1;
  const char* first = "";
  for (size_t i = 0; i < ps.size(); ++i) {
    if (ps[i] == nullptr) continue;
    if (dev < 0) {
      dev = ps[i]->device;
      first = roles[i];
    } else if (ps[i]->device != dev) {
      std::ostringstream os;
      os << "tc_signal: operand '" << roles[i] << "': on CUDA device " << ps[i]->device << " but operand '" << first
         << "' is on CUDA device " << dev << "; every operand must be on the same device";
      throw py::value_error(os.str());
    }
  }
}

std::vector<int64_t> op_shape(const Operand& op) {
  std::vector<int64_t> s(op.ndim());
  for (int i = 0; i < op.ndim(); ++i) s[i] = op.dim(i);
  return s;
}

py::tuple to_tuple(const std::vector<int64_t>& v) {
  py::tuple t(v.size());
  for (size_t i = 0; i < v.size(); ++i) t[i] = py::int_(v[i]);
  return t;
}

// ------------------------------------------------------------------------------------------------ Conv1d entry points
py::tuple conv1d_plan(py::object x, py::object w, py::object b, int64_t stride, int64_t pb, int64_t pe,
                      int64_t dilation, int64_t groups) {
  drain_completed();
  require_operand(x, "x");
  require_operand(w, "w");
  Peeked X = peek_operand(x, "x", 3);
  Peeked W = peek_operand(w, "w", 3);
  Peeked B;
  const bool has_b = !b.is_none();
  if (has_b) B = peek_operand(b, "b", 1);
  same_device_peeked({&X, &W, has_b ? &B : nullptr}, {"x", "w", "b"});
  Conv1dPlan p = plan_conv1d(X.shape, W.shape, has_b ? &B.shape : nullptr, stride, pb, pe, dilation, groups);
  return py::make_tuple(X.capsule.object(), W.capsule.object(), has_b ? B.capsule.object() : py::none(),
                        to_tuple(p.out_shape), py::int_(p.scratch_numel));
}

void conv1d_run(py::object xc, py::object wc, py::object bc, py::object out, py::object scratch, int64_t stride,
                int64_t pb, int64_t pe, int64_t dilation, int64_t groups) {
  drain_completed();
  require_operand(xc, "x");
  require_operand(wc, "w");
  require_operand(out, "out");
  Operand X(xc, "x", 3, false);
  Operand W(wc, "w", 3, false);
  Operand B;
  if (!bc.is_none()) B = Operand(bc, "b", 1, false);
  std::vector<int64_t> bshape;
  if (B.present()) bshape = op_shape(B);
  const std::vector<int64_t> xshape = op_shape(X);
  const std::vector<int64_t> wshape = op_shape(W);
  Conv1dPlan p = plan_conv1d(xshape, wshape, B.present() ? &bshape : nullptr, stride, pb, pe, dilation, groups);
  Operand Y(to_capsule(out, "out", true).handle(), "out", 3, true);
  expect_shape(Y, p.out_shape);
  Operand S;
  if (p.scratch_numel > 0) {
    if (scratch.is_none()) bad_attr("conv1d: this call needs an im2col scratch operand and none was passed");
    S = Operand(to_capsule(scratch, "scratch", true).handle(), "scratch", 1, true);
    expect_shape(S, {p.scratch_numel});
  } else if (!scratch.is_none()) {
    bad_attr("conv1d: this call needs no scratch (K == 1, stride 1, no pads) but one was passed");
  }
  const int device = common_device({&X, &W, &B, &Y, &S});
  for (const Operand* in : {&X, &W, &B, &S}) expect_disjoint(Y, *in);
  for (const Operand* in : {&X, &W, &B}) expect_disjoint(S, *in);
  // F01: every consumed manager stays alive until the legacy stream has passed these launches (also when a launch
  // throws part-way); the producer deleters run later, from drain_completed / synchronize / atexit.
  RetainUntilComplete keep(device, {&X, &W, &B, &Y, &S});
  conv1d_forward(device, p.g, X.data(), W.data(), B.present() ? B.data() : nullptr, Y.data(),
                 S.present() ? S.data() : nullptr);
}

// ------------------------------------------------------------------------------------------------ ConvTranspose1d entry points
py::tuple conv_transpose1d_plan(py::object x, py::object w, py::object b, int64_t stride, int64_t pb, int64_t pe,
                                int64_t output_padding, int64_t dilation, int64_t groups) {
  drain_completed();
  require_operand(x, "x");
  require_operand(w, "w");
  Peeked X = peek_operand(x, "x", 3);
  Peeked W = peek_operand(w, "w", 3);
  Peeked B;
  const bool has_b = !b.is_none();
  if (has_b) B = peek_operand(b, "b", 1);
  same_device_peeked({&X, &W, has_b ? &B : nullptr}, {"x", "w", "b"});
  ConvTranspose1dPlan p = plan_conv_transpose1d(X.shape, W.shape, has_b ? &B.shape : nullptr, stride, pb, pe,
                                                output_padding, dilation, groups);
  return py::make_tuple(X.capsule.object(), W.capsule.object(), has_b ? B.capsule.object() : py::none(),
                        to_tuple(p.out_shape));
}

void conv_transpose1d_run(py::object xc, py::object wc, py::object bc, py::object out, int64_t stride, int64_t pb,
                          int64_t pe, int64_t output_padding, int64_t dilation, int64_t groups) {
  drain_completed();
  require_operand(xc, "x");
  require_operand(wc, "w");
  require_operand(out, "out");
  Operand X(xc, "x", 3, false);
  Operand W(wc, "w", 3, false);
  Operand B;
  if (!bc.is_none()) B = Operand(bc, "b", 1, false);
  std::vector<int64_t> bshape;
  if (B.present()) bshape = op_shape(B);
  const std::vector<int64_t> xshape = op_shape(X);
  const std::vector<int64_t> wshape = op_shape(W);
  ConvTranspose1dPlan p = plan_conv_transpose1d(xshape, wshape, B.present() ? &bshape : nullptr, stride, pb, pe,
                                                output_padding, dilation, groups);
  Operand Y(to_capsule(out, "out", true).handle(), "out", 3, true);
  expect_shape(Y, p.out_shape);
  const int device = common_device({&X, &W, &B, &Y});
  for (const Operand* in : {&X, &W, &B}) expect_disjoint(Y, *in);
  RetainUntilComplete keep(device, {&X, &W, &B, &Y});  // F01 (see conv1d_run)
  conv_transpose1d_forward(device, p.g, X.data(), W.data(), B.present() ? B.data() : nullptr, Y.data());
}

void divide_probe(py::object x, py::object y, py::object out) {
  drain_completed();
  require_operand(x, "x");
  require_operand(y, "y");
  require_operand(out, "out");
  Operand X(to_capsule(x, "x", false).handle(), "x", 1, false);
  Operand Y(to_capsule(y, "y", false).handle(), "y", 1, false);
  Operand O(to_capsule(out, "out", true).handle(), "out", 1, true);
  expect_shape(Y, {X.dim(0)});
  expect_shape(O, {X.dim(0)});
  const int device = common_device({&X, &Y, &O});
  expect_disjoint(O, X);
  expect_disjoint(O, Y);
  RetainUntilComplete keep(device, {&X, &Y, &O});  // F01
  fp32_divide_probe(device, X.data(), Y.data(), O.data(), X.dim(0));
}

// P1T-FIX6: tc_signal's exit wait as a raw CPython function (see py_args.h raw_one_real_arg for why). No catch-all: a
// forced unwind (not a std::exception) passes through the typed handlers. Prior art: CPython's METH_VARARGS |
// METH_KEYWORDS convention (caller-owned argument tuple and dict); ours: choosing it for the GIL-released wait.
PyObject* raw_exit_close_admission(PyObject*, PyObject* args, PyObject* kwargs) {
  double first_diag_s = 0.0;
  if (!raw_one_real_arg("_exit_close_admission", "first_diag_s", args, kwargs, &first_diag_s)) return nullptr;
  try {
    return close_admission_and_wait(first_diag_s).release().ptr();
  } catch (py::error_already_set& e) {
    e.restore();
  } catch (const std::bad_alloc&) {
    PyErr_NoMemory();
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
  }
  return nullptr;
}

}  // namespace

const char kExitCloseDoc[] =
    "_exit_close_admission(first_diag_s)\n\n"
    "P1T-FIX5 (V5-01), tc_signal's atexit hook: close admission for good (later producer deleter calls and capsule "
    "drops are deliberately leaked without entering Python, counted; later producer __dlpack__ callbacks are "
    "refused), then wait, GIL released and with NO time bound, until no admitted section of another thread is "
    "running (the calling thread's own are excluded). After first_diag_s seconds it writes a stderr line, then "
    "another each time the time waited doubles. Returns {in_flight_at_close, own_thread_in_flight, "
    "pending_entries_at_close, waited_us, diagnostics, overran}. P1T-FIX6: the only tc_signal call that waits for "
    "another thread; a raw CPython function (no pybind11 dispatcher frame on the stack while it waits).";
PyMethodDef kExitCloseDef = {"_exit_close_admission",
                             reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)(void)>(&raw_exit_close_admission)),
                             METH_VARARGS | METH_KEYWORDS, kExitCloseDoc};
}  // namespace tcs

namespace tcs { void register_lstm(py::module_& m); }  // P1T-M3: src/lstm_bindings.cpp
namespace tcs { void register_m4(py::module_& m); }  // P1T-M4 norms and small ops (src/m4_bindings.cpp)

PYBIND11_MODULE(_tc_signal, m) {
  m.doc() = "tc_signal C++ core: DLPack-consuming fp32 1-D convolutions on the legacy default stream (P1T-M2)";
  using namespace tcs;
  // P1T-FIX6 (sweep (g)): every binding takes (*args, **kwargs) and converts with py_args.h's C-level converters, so
  // no Python method of an argument runs inside pybind11's dispatcher or a binding frame (design and prior art:
  // py_args.h). The first docstring line is the real signature.
  m.def(
      "conv1d_plan",
      [](py::args a, py::kwargs k) {
        const CallArgs c("conv1d_plan", a, k,
                         {{"x"}, {"w"}, {"b"}, {"stride"}, {"pad_begin"}, {"pad_end"}, {"dilation"}, {"groups"}});
        const int64_t stride = arg_int64(c, 3), pb = arg_int64(c, 4), pe = arg_int64(c, 5), d = arg_int64(c, 6),
                      g = arg_int64(c, 7);
        return conv1d_plan(c.object(0), c.object(1), c.object(2), stride, pb, pe, d, g);
      },
      "conv1d_plan(x, w, b, stride, pad_begin, pad_end, dilation, groups)\n\n"
      "Acquire and peek x/w/b; return (x_capsule, w_capsule, b_capsule|None, out_shape, scratch_numel).");
  m.def(
      "conv1d_run",
      [](py::args a, py::kwargs k) {
        const CallArgs c("conv1d_run", a, k,
                         {{"x_capsule"}, {"w_capsule"}, {"b_capsule"}, {"out"}, {"scratch"}, {"stride"}, {"pad_begin"},
                          {"pad_end"}, {"dilation"}, {"groups"}});
        const int64_t stride = arg_int64(c, 5), pb = arg_int64(c, 6), pe = arg_int64(c, 7), d = arg_int64(c, 8),
                      g = arg_int64(c, 9);
        conv1d_run(c.object(0), c.object(1), c.object(2), c.object(3), c.object(4), stride, pb, pe, d, g);
      },
      "conv1d_run(x_capsule, w_capsule, b_capsule, out, scratch, stride, pad_begin, pad_end, dilation, groups)\n\n"
      "Consume every operand, validate, launch on the legacy stream, release.");
  m.def(
      "conv_transpose1d_plan",
      [](py::args a, py::kwargs k) {
        const CallArgs c("conv_transpose1d_plan", a, k,
                         {{"x"}, {"w"}, {"b"}, {"stride"}, {"pad_begin"}, {"pad_end"}, {"output_padding"},
                          {"dilation"}, {"groups"}});
        const int64_t stride = arg_int64(c, 3), pb = arg_int64(c, 4), pe = arg_int64(c, 5), op = arg_int64(c, 6),
                      d = arg_int64(c, 7), g = arg_int64(c, 8);
        return conv_transpose1d_plan(c.object(0), c.object(1), c.object(2), stride, pb, pe, op, d, g);
      },
      "conv_transpose1d_plan(x, w, b, stride, pad_begin, pad_end, output_padding, dilation, groups)\n\n"
      "Acquire and peek x/w/b; return (x_capsule, w_capsule, b_capsule|None, out_shape).");
  m.def(
      "conv_transpose1d_run",
      [](py::args a, py::kwargs k) {
        const CallArgs c("conv_transpose1d_run", a, k,
                         {{"x_capsule"}, {"w_capsule"}, {"b_capsule"}, {"out"}, {"stride"}, {"pad_begin"},
                          {"pad_end"}, {"output_padding"}, {"dilation"}, {"groups"}});
        const int64_t stride = arg_int64(c, 4), pb = arg_int64(c, 5), pe = arg_int64(c, 6), op = arg_int64(c, 7),
                      d = arg_int64(c, 8), g = arg_int64(c, 9);
        conv_transpose1d_run(c.object(0), c.object(1), c.object(2), c.object(3), stride, pb, pe, op, d, g);
      },
      "conv_transpose1d_run(x_capsule, w_capsule, b_capsule, out, stride, pad_begin, pad_end, output_padding, "
      "dilation, groups)\n\nConsume every operand, validate, launch on the legacy stream, release.");
  m.def(
      "fp32_divide_probe",
      [](py::args a, py::kwargs k) {
        const CallArgs c("fp32_divide_probe", a, k, {{"x"}, {"y"}, {"out"}});
        divide_probe(c.object(0), c.object(1), c.object(2));
      },
      "fp32_divide_probe(x, y, out)\n\n"
      "Fast-math canary: out = x / y in fp32 with the package's build flags (tests compare it to IEEE division).");
  m.def(
      "cublas_config",
      [](py::args a, py::kwargs k) {
        const CallArgs c("cublas_config", a, k, {{"device", false}});
        CublasConfig cfg = cublas_config(c.given(0) ? arg_int32(c, 0) : 0);
        py::dict d;
        d["math_mode"] = cfg.math_mode;
        d["atomics_mode"] = cfg.atomics_mode;
        d["pointer_mode"] = cfg.pointer_mode;
        d["legacy_stream"] = cfg.legacy_stream;
        d["default_math"] = cfg.default_math;
        d["atomics_not_allowed"] = cfg.atomics_not_allowed;
        return d;
      },
      "cublas_config(device=0)\n\nSettings of tc_signal's cuBLAS handle on `device` (creates it if needed).");
  m.def(
      "build_info",
      [](py::args a, py::kwargs k) {
        const CallArgs c("build_info", a, k, {});
        (void)c;
        BuildInfo b = build_info();
        py::dict d;
        d["cudart_version"] = b.cudart_version;
        d["cublas_version"] = py::make_tuple(b.cublas_major, b.cublas_minor, b.cublas_patch);
        d["current_device"] = b.current_device;
        d["dlpack_header_version"] = py::make_tuple(DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION);
        d["dlpack_max_version_requested"] = py::make_tuple(tuning::kDlpackMaxMajor, tuning::kDlpackMaxMinor);
        d["dlpack_consumer_stream"] = tuning::kDlpackLegacyStream;
        return d;
      },
      "build_info()\n\nCUDA runtime / cuBLAS / DLPack versions this module was built and runs with.");
  register_lstm(m);
  register_m4(m);
  // P1T-FIX1 F01: the deferred-release list (dlpack_operand.h). tc_signal.synchronize() and the package's atexit hook
  // call `synchronize`; tests and the G4 high-water receipt read `pending_stats`.
  m.def(
      "synchronize",
      [](py::args a, py::kwargs k) {
        const CallArgs c("synchronize", a, k, {});
        (void)c;
        synchronize_and_release_all();
      },
      "synchronize()\n\n"
      "Synchronize the legacy default stream (current device and every device with pending operands), then release "
      "every consumed DLPack operand tc_signal still holds. RuntimeError if a device cannot synchronize (its "
      "operands are then deliberately leaked, never released early). P1T-FIX6 (V6-01): it never waits for another "
      "thread (a release already running on another thread finishes there).");
  m.def(
      "pending_stats",
      [](py::args a, py::kwargs k) {
        const CallArgs c("pending_stats", a, k, {{"reset_high_water", false}});
        return pending_stats(c.given(0) ? arg_flag(c, 0) : false);
      },
      "pending_stats(reset_high_water=False)\n\n"
      "Counters of the deferred-release list: entries/managers/bytes now, high-water marks, releases, bound drains, "
      "fallback syncs, deliberate leaks, and the bounds; P1T-FIX2: deleter_exceptions, push_alloc_failures, "
      "event_recycle_failures, release_errors, unreported_release_errors.");
  // P1T-FIX5 (round-5 review V5-01): the exit side of the release lifecycle, called by tc_signal's atexit hook
  // (tc_signal/__init__.py _release_pending_at_exit) after its final synchronize(). Not an API for other callers.
  // P1T-FIX6: a raw CPython function (raw_exit_close_admission, above the module): nothing on this stack owns a Python object
  // across the GIL-released wait, whatever the calling style (py_args.h raw_one_real_arg).
  m.add_object("_exit_close_admission",
               py::reinterpret_steal<py::object>(PyCFunction_NewEx(&tcs::kExitCloseDef, nullptr, nullptr)));
  // P1T-FIX2 (N3): build the pending-list state now (it allocates), so no noexcept release path is the first to touch
  // it; a failure here fails the import (MemoryError), never a later release. P1T-FIX5: also the release lifecycle.
  init_pending_state();
  // TEST HOOK (P1T-FIX2 N1/N3 regression tests only; not part of the API): arm one-shot failures at named release-path
  // sites (dlpack_operand.h, test_hooks). `names` is a list of site names; "clear" disarms everything.
  m.def(
      "_test_inject",
      [](py::args a, py::kwargs k) {
        const CallArgs c("_test_inject", a, k, {{"names"}});
        const std::vector<std::string> names = arg_str_list(c, 0);
        static const std::pair<const char*, unsigned> kSites[] = {
            {"alloc:event_pool_reserve", test_hooks::kAllocEventPoolReserve},
            {"alloc:event_recycle", test_hooks::kAllocEventRecycle},
            {"alloc:pending_push", test_hooks::kAllocPendingPush},
            {"alloc:owner_reserve", test_hooks::kAllocOwnerReserve},
            {"alloc:error_message", test_hooks::kAllocErrorMessage},
            {"cuda:event_record", test_hooks::kCudaEventRecord},
            {"cuda:sync", test_hooks::kCudaSync},
            {"cuda:current_device", test_hooks::kCudaCurrentDevice},
            {"cuda:event_query", test_hooks::kCudaEventQuery},
        };
        unsigned bits = 0;
        for (const std::string& n : names) {
          if (n == "clear") {
            test_hooks::clear();
            continue;
          }
          bool found = false;
          for (const auto& site : kSites)
            if (n == site.first) {
              bits |= site.second;
              found = true;
            }
          if (!found) throw py::value_error("tc_signal._test_inject: unknown site '" + n + "'");
        }
        test_hooks::arm(bits);
        return test_hooks::armed();
      },
      "_test_inject(names)\n\n"
      "TEST HOOK (P1T-FIX2): arm one-shot failures at named release-path sites; returns the armed bits.");
  m.def(
      "_test_injection_armed",
      [](py::args a, py::kwargs k) {
        const CallArgs c("_test_injection_armed", a, k, {});
        (void)c;
        return test_hooks::armed();
      },
      "_test_injection_armed()\n\n"
      "TEST HOOK (P1T-FIX2): the release-path injection bits still armed (a hit site disarms itself).");
}
