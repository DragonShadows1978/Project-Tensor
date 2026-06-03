// ops.cpp: differentiable ops. Forward builds an NDArray result; the grad_fn
// implements the vector-Jacobian product, pushing gradients into the inputs.
//
// Closure capture rule (keeps the autograd graph acyclic): capture INPUT
// Tensors (to call accumulate_grad) and any saved-forward NDArrays by value —
// NEVER the output Tensor.

#include "tc/ops.h"

#include <vector>

namespace tc {
namespace ops {

// NDArray-level helpers used inside backward closures.
static NDArray nadd(const NDArray& a, const NDArray& b) { return ew_binary(a, b, 0); }
static NDArray nsub(const NDArray& a, const NDArray& b) { return ew_binary(a, b, 1); }
static NDArray nmul(const NDArray& a, const NDArray& b) { return ew_binary(a, b, 2); }
static NDArray ndiv(const NDArray& a, const NDArray& b) { return ew_binary(a, b, 3); }
static NDArray nmuls(const NDArray& a, double s) { return ew_scalar(a, s, 2, false); }
static NDArray nadds(const NDArray& a, double s) { return ew_scalar(a, s, 0, false); }
static NDArray nsubsl(double s, const NDArray& a) { return ew_scalar(a, s, 1, true); }  // s - a

// ------------------------------------------------------------- arithmetic
Tensor add(const Tensor& a, const Tensor& b) {
  NDArray out = ew_binary(a.data(), b.data(), 0);
  return Tensor::from_op(out, {a, b}, "add", [a, b](const NDArray& g) {
    a.v->accumulate_grad(g);
    b.v->accumulate_grad(g);
  });
}
Tensor sub(const Tensor& a, const Tensor& b) {
  NDArray out = ew_binary(a.data(), b.data(), 1);
  return Tensor::from_op(out, {a, b}, "sub", [a, b](const NDArray& g) {
    a.v->accumulate_grad(g);
    b.v->accumulate_grad(ew_unary(g, U_NEG));
  });
}
Tensor mul(const Tensor& a, const Tensor& b) {
  NDArray out = ew_binary(a.data(), b.data(), 2);
  return Tensor::from_op(out, {a, b}, "mul", [a, b](const NDArray& g) {
    a.v->accumulate_grad(nmul(g, b.data()));
    b.v->accumulate_grad(nmul(g, a.data()));
  });
}
Tensor div(const Tensor& a, const Tensor& b) {
  NDArray out = ew_binary(a.data(), b.data(), 3);
  return Tensor::from_op(out, {a, b}, "div", [a, b](const NDArray& g) {
    a.v->accumulate_grad(ndiv(g, b.data()));
    // db = -g * a / b^2
    NDArray b2 = nmul(b.data(), b.data());
    b.v->accumulate_grad(ew_unary(ndiv(nmul(g, a.data()), b2), U_NEG));
  });
}
Tensor add_scalar(const Tensor& a, double s) {
  NDArray out = ew_scalar(a.data(), s, 0, false);
  return Tensor::from_op(out, {a}, "add_scalar", [a](const NDArray& g) { a.v->accumulate_grad(g); });
}
Tensor mul_scalar(const Tensor& a, double s) {
  NDArray out = ew_scalar(a.data(), s, 2, false);
  return Tensor::from_op(out, {a}, "mul_scalar", [a, s](const NDArray& g) { a.v->accumulate_grad(nmuls(g, s)); });
}
Tensor neg(const Tensor& a) {
  NDArray out = ew_unary(a.data(), U_NEG);
  return Tensor::from_op(out, {a}, "neg", [a](const NDArray& g) { a.v->accumulate_grad(ew_unary(g, U_NEG)); });
}

// ------------------------------------------------------------- math / acts
Tensor exp(const Tensor& a) {
  NDArray out = ew_unary(a.data(), U_EXP);
  return Tensor::from_op(out, {a}, "exp", [a, out](const NDArray& g) { a.v->accumulate_grad(nmul(g, out)); });
}
Tensor log(const Tensor& a) {
  NDArray out = ew_unary(a.data(), U_LOG);
  return Tensor::from_op(out, {a}, "log", [a](const NDArray& g) { a.v->accumulate_grad(ndiv(g, a.data())); });
}
Tensor sqrt(const Tensor& a) {
  NDArray out = ew_unary(a.data(), U_SQRT);
  return Tensor::from_op(out, {a}, "sqrt", [a, out](const NDArray& g) {
    a.v->accumulate_grad(ndiv(g, nmuls(out, 2.0)));
  });
}
Tensor relu(const Tensor& a) {
  NDArray out = ew_unary(a.data(), U_RELU);
  return Tensor::from_op(out, {a}, "relu", [a](const NDArray& g) {
    a.v->accumulate_grad(nmul(g, ge_scalar(a.data(), 0.0)));
  });
}
Tensor sigmoid(const Tensor& a) {
  NDArray out = ew_unary(a.data(), U_SIGMOID);
  return Tensor::from_op(out, {a}, "sigmoid", [a, out](const NDArray& g) {
    a.v->accumulate_grad(nmul(g, nmul(out, nsubsl(1.0, out))));  // g*out*(1-out)
  });
}
Tensor tanh(const Tensor& a) {
  NDArray out = ew_unary(a.data(), U_TANH);
  return Tensor::from_op(out, {a}, "tanh", [a, out](const NDArray& g) {
    a.v->accumulate_grad(nmul(g, nsubsl(1.0, nmul(out, out))));  // g*(1-out^2)
  });
}
Tensor gelu(const Tensor& a) {
  NDArray out = ew_unary(a.data(), U_GELU);
  return Tensor::from_op(out, {a}, "gelu", [a](const NDArray& g) {
    const double c = 0.7978845608028654, a3 = 0.044715;
    NDArray x = a.data();
    NDArray x2 = nmul(x, x), x3 = nmul(x2, x);
    NDArray u = nmuls(nadd(x, nmuls(x3, a3)), c);
    NDArray t = ew_unary(u, U_TANH);
    NDArray left = nmuls(nadds(t, 1.0), 0.5);                       // 0.5(1+t)
    NDArray sech2 = nsubsl(1.0, nmul(t, t));                        // 1 - t^2
    NDArray inner = nmuls(nadds(nmuls(x2, 3.0 * a3), 1.0), c);      // c(1+3a3 x^2)
    NDArray right = nmuls(nmul(nmul(x, sech2), inner), 0.5);        // 0.5 x sech2 inner
    a.v->accumulate_grad(nmul(g, nadd(left, right)));
  });
}
Tensor silu(const Tensor& a) {
  NDArray out = ew_unary(a.data(), U_SILU);
  return Tensor::from_op(out, {a}, "silu", [a](const NDArray& g) {
    NDArray sig = ew_unary(a.data(), U_SIGMOID);
    // d = sig * (1 + x*(1-sig))
    NDArray d = nmul(sig, nadds(nmul(a.data(), nsubsl(1.0, sig)), 1.0));
    a.v->accumulate_grad(nmul(g, d));
  });
}

// ------------------------------------------------------------- linalg
Tensor matmul(const Tensor& a, const Tensor& b) {
  NDArray out = tc::matmul(a.data(), b.data());
  return Tensor::from_op(out, {a, b}, "matmul", [a, b](const NDArray& g) {
    a.v->accumulate_grad(tc::matmul(g, transpose2d_last(b.data())));
    b.v->accumulate_grad(tc::matmul(transpose2d_last(a.data()), g));
  });
}

// ------------------------------------------------------------- reductions
static Shape keepdim_shape(const Shape& in, const std::vector<int>& axes) {
  Shape s = in;
  int nd = (int)in.size();
  if (axes.empty()) { for (auto& x : s) x = 1; return s; }
  for (int ax : axes) { int d = ax < 0 ? ax + nd : ax; s[d] = 1; }
  return s;
}
Tensor sum(const Tensor& a, const std::vector<int>& axes, bool keepdim) {
  NDArray out = reduce_sum(a.data(), axes, keepdim);
  Shape in_shape = a.shape();
  Shape kshape = keepdim_shape(in_shape, axes);
  return Tensor::from_op(out, {a}, "sum", [a, in_shape, kshape](const NDArray& g) {
    NDArray gk = g.reshape(kshape);  // restore reduced dims as size 1
    NDArray bcast = nadd(NDArray::zeros(in_shape, a.dtype(), a.device()), gk);
    a.v->accumulate_grad(bcast);
  });
}
Tensor mean(const Tensor& a, const std::vector<int>& axes, bool keepdim) {
  int64_t cnt = 1;
  int nd = a.ndim();
  if (axes.empty()) cnt = a.data().numel();
  else for (int ax : axes) { int d = ax < 0 ? ax + nd : ax; cnt *= a.shape()[d]; }
  return mul_scalar(sum(a, axes, keepdim), 1.0 / (double)cnt);
}

// ------------------------------------------------------------- shape
Tensor reshape(const Tensor& a, const Shape& shape) {
  NDArray out = a.data().reshape(shape);
  Shape in_shape = a.shape();
  return Tensor::from_op(out, {a}, "reshape", [a, in_shape](const NDArray& g) {
    a.v->accumulate_grad(g.reshape(in_shape));
  });
}
Tensor transpose_last(const Tensor& a) {
  NDArray out = transpose2d_last(a.data());
  return Tensor::from_op(out, {a}, "transpose", [a](const NDArray& g) {
    a.v->accumulate_grad(transpose2d_last(g));
  });
}

// ------------------------------------------------------------- composite
Tensor softmax(const Tensor& a, int axis) {
  NDArray mx = reduce_max(a.data(), {axis}, /*keepdim=*/true);  // constant (stop-grad)
  Tensor shifted = sub(a, Tensor::make(mx, false));
  Tensor e = exp(shifted);
  Tensor s = sum(e, {axis}, /*keepdim=*/true);
  return div(e, s);
}
Tensor mse_loss(const Tensor& pred, const Tensor& target) {
  Tensor d = sub(pred, target);
  return mean(mul(d, d), {}, false);
}

}  // namespace ops
}  // namespace tc
