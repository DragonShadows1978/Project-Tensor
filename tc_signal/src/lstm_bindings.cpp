// _tc_signal LSTM entry points (P1T-M3, T4). Registered into M2's module by one `register_lstm(m);` line in
// bindings.cpp. Same operand contract and plan/run split as M2's convolutions (src/bindings.cpp, dlpack_operand.h):
//   lstm_plan(x, W, R, B, initial_h, initial_c, hidden_size, direction)
//       -> acquires the input capsules (__dlpack__(stream=1)), PEEKS them (device, dtype, layout, shapes) without
//          consuming, and returns (x, W, R, B|None, initial_h|None, initial_c|None capsules, Y shape, Y_h/Y_c shape,
//          scratch numel, T, N) so the Python layer can size and allocate the outputs and the one scratch operand;
//   lstm_run(capsules, Y, Y_h, Y_c, scratch, hidden_size, direction)
//       -> consumes every capsule, re-validates, checks devices and that no written operand aliases any other operand,
//          runs the whole recurrence (lstm.cu: all T steps, every direction) on the legacy default stream inside this
//          one call, then every producer deleter runs at scope exit.
// direction codes: 0 forward, 1 reverse, 2 bidirectional (the Python layer maps the ONNX strings).
//
// Prior art: the operand contract and the plan/run split are M2's (see dlpack_operand.h and bindings.cpp for their
// prior art: DLPack, the Python array API stream protocol, cuDNN/cuBLASLt workspace-query APIs). M2's peek helpers in
// bindings.cpp have internal linkage, so the two small ones used here (peek + same-device check) are restated, not
// shared. The LSTM shape rules are the ONNX LSTM operator's (layout 0). Nothing here is new.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <climits>
#include <sstream>
#include <string>
#include <vector>

#include "dlpack_operand.h"
#include "lstm_ops.h"

namespace py = pybind11;

namespace tcs {
namespace {

constexpr int64_t kDirForward = 0;
constexpr int64_t kDirReverse = 1;
constexpr int64_t kDirBidirectional = 2;

[[noreturn]] void lstm_bad(const std::string& msg) { throw py::value_error("tc_signal.lstm: " + msg); }

void check_int_range(const char* what, int64_t v) {
  if (v > INT_MAX) {
    std::ostringstream os;
    os << what << " (" << v << ") exceeds the 32-bit range cuBLAS takes";
    lstm_bad(os.str());
  }
}

void expect_dims(const char* role, const std::vector<int64_t>& got, const std::vector<int64_t>& want,
                 const char* layout) {
  if (got != want) {
    std::ostringstream os;
    os << role << " has shape " << shape_str(got) << ", expected " << shape_str(want) << " (" << layout << ")";
    lstm_bad(os.str());
  }
}

struct LstmPlan {
  LstmGeom g;
  std::vector<int64_t> y_shape;       // [T, D, N, H]
  std::vector<int64_t> state_shape;   // [D, N, H]
  int64_t scratch_numel;
};

LstmPlan plan_lstm(const std::vector<int64_t>& xs, const std::vector<int64_t>& Ws, const std::vector<int64_t>& Rs,
                   const std::vector<int64_t>* Bs, const std::vector<int64_t>* h0s, const std::vector<int64_t>* c0s,
                   int64_t hidden_size, int64_t direction) {
  std::ostringstream os;
  if (direction != kDirForward && direction != kDirReverse && direction != kDirBidirectional) {
    os << "direction code " << direction << " is not 0 (forward), 1 (reverse) or 2 (bidirectional)";
    lstm_bad(os.str());
  }
  if (hidden_size < 1 || hidden_size > INT_MAX / (kLstmGates * kLstmBiasHalves)) {
    os << "hidden_size must be in [1, " << INT_MAX / (kLstmGates * kLstmBiasHalves) << "], got " << hidden_size;
    lstm_bad(os.str());
  }
  LstmPlan p{};
  LstmGeom& g = p.g;
  g.T = xs[0];
  g.N = xs[1];
  g.I = xs[2];
  g.H = hidden_size;
  g.D = (direction == kDirBidirectional) ? 2 : 1;
  g.first_reverse = (direction == kDirForward) ? g.D : (direction == kDirReverse ? 0 : 1);
  const int64_t H4 = kLstmGates * g.H;
  if (Ws[0] != g.D) {
    os << "W.shape[0] (" << Ws[0] << ") != num_directions (" << g.D << ") for this direction; W is [D, 4H, I]";
    lstm_bad(os.str());
  }
  if (Ws[1] != H4) {
    os << "W.shape[1] (" << Ws[1] << ") != 4*hidden_size (" << H4 << "); W is [D, 4H, I]";
    lstm_bad(os.str());
  }
  if (Ws[2] != g.I) {
    os << "W.shape[2] (" << Ws[2] << ") != input size x.shape[2] (" << g.I << "); W is [D, 4H, I]";
    lstm_bad(os.str());
  }
  expect_dims("R", Rs, {g.D, H4, g.H}, "R is [D, 4H, H]");
  if (Bs != nullptr) expect_dims("B", *Bs, {g.D, kLstmBiasHalves * H4}, "B is [D, 8H] = [Wb; Rb]");
  if (h0s != nullptr) expect_dims("initial_h", *h0s, {g.D, g.N, g.H}, "initial_h is [D, N, H]");
  if (c0s != nullptr) expect_dims("initial_c", *c0s, {g.D, g.N, g.H}, "initial_c is [D, N, H]");
  check_int_range("4*hidden_size", H4);
  check_int_range("T*N", g.T * g.N);
  check_int_range("input size", g.I);
  check_int_range("batch size", g.N);
  check_int_range("D*N*H", g.D * g.N * g.H);
  p.y_shape = {g.T, g.D, g.N, g.H};
  p.state_shape = {g.D, g.N, g.H};
  p.scratch_numel = lstm_scratch_numel(g);
  return p;
}

// ------------------------------------------------------------------------------------------------ peek helpers
struct Peeked {
  py::object capsule;            // None for an absent optional operand
  std::vector<int64_t> shape;
  int device = -1;
  bool present = false;
};

Peeked peek_operand(py::handle obj, const char* role, int ndim) {
  Peeked p;
  if (obj.is_none()) {
    p.capsule = py::none();
    return p;
  }
  p.capsule = to_capsule(obj, role, false);
  const DLTensor* t = peek(p.capsule, role);
  validate_tensor(t, role, ndim);
  p.shape = shape_of(t);
  p.device = t->device.device_id;
  p.present = true;
  return p;
}

void same_device_peeked(const std::vector<const Peeked*>& ps, const std::vector<const char*>& roles) {
  int dev = -1;
  const char* first = "";
  for (size_t i = 0; i < ps.size(); ++i) {
    if (!ps[i]->present) continue;
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

// ------------------------------------------------------------------------------------------------ entry points
py::tuple lstm_plan(py::object x, py::object W, py::object R, py::object B, py::object h0, py::object c0,
                    int64_t hidden_size, int64_t direction) {
  if (x.is_none()) throw py::type_error("tc_signal: operand 'x': expected a DLPack producer or capsule, got None");
  if (W.is_none()) throw py::type_error("tc_signal: operand 'W': expected a DLPack producer or capsule, got None");
  if (R.is_none()) throw py::type_error("tc_signal: operand 'R': expected a DLPack producer or capsule, got None");
  Peeked X = peek_operand(x, "x", 3);
  Peeked Wp = peek_operand(W, "W", 3);
  Peeked Rp = peek_operand(R, "R", 3);
  Peeked Bp = peek_operand(B, "B", 2);
  Peeked H0 = peek_operand(h0, "initial_h", 3);
  Peeked C0 = peek_operand(c0, "initial_c", 3);
  same_device_peeked({&X, &Wp, &Rp, &Bp, &H0, &C0}, {"x", "W", "R", "B", "initial_h", "initial_c"});
  LstmPlan p = plan_lstm(X.shape, Wp.shape, Rp.shape, Bp.present ? &Bp.shape : nullptr,
                         H0.present ? &H0.shape : nullptr, C0.present ? &C0.shape : nullptr, hidden_size, direction);
  return py::make_tuple(X.capsule, Wp.capsule, Rp.capsule, Bp.capsule, H0.capsule, C0.capsule, to_tuple(p.y_shape),
                        to_tuple(p.state_shape), py::int_(p.scratch_numel), py::int_(p.g.T), py::int_(p.g.N));
}

void lstm_run(py::object xc, py::object Wc, py::object Rc, py::object Bc, py::object h0c, py::object c0c, py::object Y,
              py::object Y_h, py::object Y_c, py::object scratch, int64_t hidden_size, int64_t direction) {
  Operand X(xc, "x", 3, false);
  Operand Wo(Wc, "W", 3, false);
  Operand Ro(Rc, "R", 3, false);
  Operand Bo;
  if (!Bc.is_none()) Bo = Operand(Bc, "B", 2, false);
  Operand H0;
  if (!h0c.is_none()) H0 = Operand(h0c, "initial_h", 3, false);
  Operand C0;
  if (!c0c.is_none()) C0 = Operand(c0c, "initial_c", 3, false);
  std::vector<int64_t> bshape, h0shape, c0shape;
  if (Bo.present()) bshape = op_shape(Bo);
  if (H0.present()) h0shape = op_shape(H0);
  if (C0.present()) c0shape = op_shape(C0);
  LstmPlan p = plan_lstm(op_shape(X), op_shape(Wo), op_shape(Ro), Bo.present() ? &bshape : nullptr,
                         H0.present() ? &h0shape : nullptr, C0.present() ? &c0shape : nullptr, hidden_size, direction);
  Operand Yo(to_capsule(Y, "Y", true), "Y", 4, true);
  expect_shape(Yo, p.y_shape);
  Operand Yh(to_capsule(Y_h, "Y_h", true), "Y_h", 3, true);
  expect_shape(Yh, p.state_shape);
  Operand Yc(to_capsule(Y_c, "Y_c", true), "Y_c", 3, true);
  expect_shape(Yc, p.state_shape);
  Operand S(to_capsule(scratch, "scratch", true), "scratch", 1, true);
  expect_shape(S, {p.scratch_numel});
  const int device = common_device({&X, &Wo, &Ro, &Bo, &H0, &C0, &Yo, &Yh, &Yc, &S});
  // Every written operand must not alias any input nor any other written operand.
  const std::vector<const Operand*> written = {&Yo, &Yh, &Yc, &S};
  const std::vector<const Operand*> inputs = {&X, &Wo, &Ro, &Bo, &H0, &C0};
  for (size_t a = 0; a < written.size(); ++a) {
    for (const Operand* in : inputs) expect_disjoint(*written[a], *in);
    for (size_t b = a + 1; b < written.size(); ++b) expect_disjoint(*written[a], *written[b]);
  }
  lstm_forward(device, p.g, X.data(), Wo.data(), Ro.data(), Bo.present() ? Bo.data() : nullptr,
               H0.present() ? H0.data() : nullptr, C0.present() ? C0.data() : nullptr, Yo.data(), Yh.data(), Yc.data(),
               S.data());
  // Leaving scope runs each Operand's destructor: every producer deleter is called after the launches.
}

}  // namespace

void register_lstm(py::module_& m) {
  m.def("lstm_plan", &lstm_plan, py::arg("x"), py::arg("W"), py::arg("R"), py::arg("B"), py::arg("initial_h"),
        py::arg("initial_c"), py::arg("hidden_size"), py::arg("direction"),
        "Acquire and peek the LSTM inputs; return (x, W, R, B|None, initial_h|None, initial_c|None capsules, Y shape, "
        "Y_h/Y_c shape, scratch numel, T, N).");
  m.def("lstm_run", &lstm_run, py::arg("x_capsule"), py::arg("W_capsule"), py::arg("R_capsule"), py::arg("B_capsule"),
        py::arg("initial_h_capsule"), py::arg("initial_c_capsule"), py::arg("Y"), py::arg("Y_h"), py::arg("Y_c"),
        py::arg("scratch"), py::arg("hidden_size"), py::arg("direction"),
        "Consume every operand, validate, run the whole LSTM recurrence on the legacy stream, release.");
  m.def(
      "lstm_cublas_config",
      [](int device) {
        LstmCublasConfig c = lstm_cublas_config(device);
        py::dict d;
        d["math_mode"] = c.math_mode;
        d["atomics_mode"] = c.atomics_mode;
        d["pointer_mode"] = c.pointer_mode;
        d["legacy_stream"] = c.legacy_stream;
        d["default_math"] = c.default_math;
        d["atomics_not_allowed"] = c.atomics_not_allowed;
        return d;
      },
      py::arg("device") = 0, "Settings of the LSTM's cuBLAS handle on `device` (creates it if needed).");
}

}  // namespace tcs
