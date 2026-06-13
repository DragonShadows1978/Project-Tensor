// ops.cpp: differentiable ops. Forward builds an NDArray result; the grad_fn
// implements the vector-Jacobian product, pushing gradients into the inputs.
//
// Closure capture rule (keeps the autograd graph acyclic): capture INPUT
// Tensors (to call accumulate_grad) and any saved-forward NDArrays by value —
// NEVER the output Tensor.

#include "tc/ops.h"

#include <stdexcept>
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

Tensor abs(const Tensor& a) {
  NDArray out = ew_unary(a.data(), U_ABS);
  return Tensor::from_op(out, {a}, "abs", [a](const NDArray& g) {
    a.v->accumulate_grad(nmul(g, ew_unary(a.data(), U_SIGN)));
  });
}

Tensor sin(const Tensor& a) {
  NDArray out = ew_unary(a.data(), U_SIN);
  return Tensor::from_op(out, {a}, "sin", [a](const NDArray& g) {
    a.v->accumulate_grad(nmul(g, ew_unary(a.data(), U_COS)));
  });
}
Tensor cos(const Tensor& a) {
  NDArray out = ew_unary(a.data(), U_COS);
  return Tensor::from_op(out, {a}, "cos", [a](const NDArray& g) {
    a.v->accumulate_grad(nmul(g, ew_unary(ew_unary(a.data(), U_SIN), U_NEG)));
  });
}
Tensor tan(const Tensor& a) {
  NDArray out = ew_unary(a.data(), U_TAN);
  return Tensor::from_op(out, {a}, "tan", [a, out](const NDArray& g) {
    a.v->accumulate_grad(nmul(g, nadds(nmul(out, out), 1.0)));  // 1+tan^2
  });
}
Tensor asin(const Tensor& a) {
  NDArray out = ew_unary(a.data(), U_ASIN);
  return Tensor::from_op(out, {a}, "asin", [a](const NDArray& g) {
    NDArray denom = ew_unary(nsubsl(1.0, nmul(a.data(), a.data())), U_SQRT);
    a.v->accumulate_grad(ndiv(g, denom));
  });
}
Tensor acos(const Tensor& a) {
  NDArray out = ew_unary(a.data(), U_ACOS);
  return Tensor::from_op(out, {a}, "acos", [a](const NDArray& g) {
    NDArray denom = ew_unary(nsubsl(1.0, nmul(a.data(), a.data())), U_SQRT);
    a.v->accumulate_grad(ew_unary(ndiv(g, denom), U_NEG));
  });
}
Tensor atan(const Tensor& a) {
  NDArray out = ew_unary(a.data(), U_ATAN);
  return Tensor::from_op(out, {a}, "atan", [a](const NDArray& g) {
    a.v->accumulate_grad(ndiv(g, nadds(nmul(a.data(), a.data()), 1.0)));  // 1/(1+x^2)
  });
}
Tensor sinh(const Tensor& a) {
  NDArray out = ew_unary(a.data(), U_SINH);
  return Tensor::from_op(out, {a}, "sinh", [a](const NDArray& g) {
    a.v->accumulate_grad(nmul(g, ew_unary(a.data(), U_COSH)));
  });
}
Tensor cosh(const Tensor& a) {
  NDArray out = ew_unary(a.data(), U_COSH);
  return Tensor::from_op(out, {a}, "cosh", [a](const NDArray& g) {
    a.v->accumulate_grad(nmul(g, ew_unary(a.data(), U_SINH)));
  });
}
Tensor log2(const Tensor& a) {
  NDArray out = ew_unary(a.data(), U_LOG2);
  return Tensor::from_op(out, {a}, "log2", [a](const NDArray& g) {
    a.v->accumulate_grad(ndiv(g, nmuls(a.data(), 0.6931471805599453)));
  });
}
Tensor log10(const Tensor& a) {
  NDArray out = ew_unary(a.data(), U_LOG10);
  return Tensor::from_op(out, {a}, "log10", [a](const NDArray& g) {
    a.v->accumulate_grad(ndiv(g, nmuls(a.data(), 2.302585092994046)));
  });
}
// Non-differentiable (piecewise-constant / predicate) ops -> detached results.
Tensor sign(const Tensor& a) { return Tensor::make(ew_unary(a.data(), U_SIGN), false); }
Tensor floor(const Tensor& a) { return Tensor::make(ew_unary(a.data(), U_FLOOR), false); }
Tensor ceil(const Tensor& a) { return Tensor::make(ew_unary(a.data(), U_CEIL), false); }
Tensor round(const Tensor& a) { return Tensor::make(ew_unary(a.data(), U_ROUND), false); }
Tensor isnan(const Tensor& a) { return Tensor::make(ew_unary(a.data(), U_ISNAN), false); }
Tensor isinf(const Tensor& a) { return Tensor::make(ew_unary(a.data(), U_ISINF), false); }
Tensor isfinite(const Tensor& a) { return Tensor::make(ew_unary(a.data(), U_ISFINITE), false); }
Tensor nan_to_num(const Tensor& a, double nan, double pinf, double ninf) {
  NDArray out = ew_nan_to_num(a.data(), nan, pinf, ninf);
  return Tensor::from_op(out, {a}, "nan_to_num", [a](const NDArray& g) {
    a.v->accumulate_grad(g);  // identity grad where input was finite
  });
}
Tensor reciprocal(const Tensor& a) {
  NDArray out = ew_unary(a.data(), U_RECIP);
  return Tensor::from_op(out, {a}, "reciprocal", [a, out](const NDArray& g) {
    a.v->accumulate_grad(nmul(g, ew_unary(nmul(out, out), U_NEG)));  // -1/x^2
  });
}
Tensor clamp(const Tensor& a, double lo, double hi) {
  NDArray out = ew_clamp(a.data(), lo, hi);
  NDArray ad = a.data();
  return Tensor::from_op(out, {a}, "clamp", [a, ad, lo, hi](const NDArray& g) {
    // grad passes through where lo <= x <= hi
    NDArray inb = nmul(compare_scalar(ad, lo, 1), compare_scalar(ad, hi, 3));  // (x>=lo)*(x<=hi)
    a.v->accumulate_grad(nmul(g, inb));
  });
}
Tensor maximum(const Tensor& a, const Tensor& b) {
  NDArray out = ew_binary(a.data(), b.data(), 4);
  return Tensor::from_op(out, {a, b}, "maximum", [a, b](const NDArray& g) {
    NDArray amask = tc::compare(a.data(), b.data(), 1);  // a>=b -> a
    a.v->accumulate_grad(nmul(g, amask));
    b.v->accumulate_grad(nmul(g, tc::compare(b.data(), a.data(), 0)));  // b>a
  });
}
Tensor minimum(const Tensor& a, const Tensor& b) {
  NDArray out = ew_binary(a.data(), b.data(), 5);
  return Tensor::from_op(out, {a, b}, "minimum", [a, b](const NDArray& g) {
    a.v->accumulate_grad(nmul(g, tc::compare(a.data(), b.data(), 3)));   // a<=b
    b.v->accumulate_grad(nmul(g, tc::compare(b.data(), a.data(), 2)));   // b<a
  });
}
Tensor slice(const Tensor& a, int dim, int64_t start, int64_t len) {
  NDArray out = slice_nd(a.data(), dim, start, len);
  Shape in_shape = a.shape();
  int nd = a.ndim();
  int d = dim < 0 ? dim + nd : dim;
  return Tensor::from_op(out, {a}, "slice", [a, in_shape, d, start](const NDArray& g) {
    a.v->accumulate_grad(pad_into(g, in_shape, d, start));
  });
}

// ------------------------------------------------------------- linalg
Tensor causal_softmax(const Tensor& scores) {
  NDArray out = tc::causal_softmax(scores.data());
  return Tensor::from_op(out, {scores}, "causal_softmax", [](const NDArray&) -> void {
    throw std::runtime_error("causal_softmax: no backward — use the eager chain for training");
  });
}

Tensor rms_norm(const Tensor& x, const Tensor& w, double eps) {
  NDArray out = tc::rms_norm(x.data(), w.data(), eps, x.data().dtype);
  return Tensor::from_op(out, {x, w}, "rms_norm", [](const NDArray&) -> void {
    // Inference-only fusion: training paths must use the unfused chain
    // (RMSNormTC guards on is_grad_enabled). Loud failure > silent wrong grad.
    throw std::runtime_error("rms_norm: no backward — use the unfused chain for training");
  });
}

void write_rows(Tensor& buf, const Tensor& src, int64_t start) {
  // In-place mutation: forbidden under autograd (would silently corrupt
  // any graph that captured buf). Inference-only by construction.
  if (tc::grad_enabled())
    throw std::runtime_error("write_rows: in-place op is inference-only");
  NDArray b = buf.data();
  tc::write_rows(b, src.data(), start);
}

Tensor rope_apply(const Tensor& x, const Tensor& cs, const Tensor& sn,
                  int64_t pos0) {
  NDArray out = tc::rope_apply(x.data(), cs.data(), sn.data(), pos0);
  return Tensor::from_op(out, {x}, "rope_apply", [](const NDArray&) -> void {
    // Inference-only fusion (callers guard on is_grad_enabled and fall
    // back to the composed slice/cat/mul chain for training).
    throw std::runtime_error("rope_apply: no backward — use the composed chain");
  });
}

Tensor matmul(const Tensor& a, const Tensor& b, float alpha, bool trans_b) {
  NDArray out = tc::matmul(a.data(), b.data(), alpha, trans_b);
  return Tensor::from_op(out, {a, b}, "matmul", [a, b, alpha, trans_b](const NDArray& g) {
    // C = alpha * A @ op(B).  dA = alpha * g @ op(B)^T: trans_b flips, reusing
    // the no-copy OP_T read.  dB: non-trans alpha*A^T@g, trans alpha*g^T@A.
    a.v->accumulate_grad(tc::matmul(g, b.data(), alpha, !trans_b));
    if (trans_b)
      b.v->accumulate_grad(tc::matmul(transpose2d_last(g), a.data(), alpha, false));
    else
      b.v->accumulate_grad(tc::matmul(transpose2d_last(a.data()), g, alpha, false));
  });
}

// Shared VJP for both int4 linear variants: forward is y = x @ W_kn with
// W_kn = dequant(W) laid out (K, N), so dx = g @ W_kn^T. The dequantized
// weight is rebuilt inside the closure at backward time and freed on
// return — only the packed/scales/zeros handles (already resident for
// inference) are captured. Weights are frozen: no VJP for them exists.
static GradFn int4_grad(const Tensor& x, const Tensor& packed,
                        const Tensor& scales, const Tensor& zeros,
                        int group_size) {
  return [x, packed, scales, zeros, group_size](const NDArray& g) {
    NDArray w_kn = tc::int4_dequant(packed.data(), scales.data(),
                                    zeros.data(), group_size, g.dtype);
    x.v->accumulate_grad(tc::matmul(g, w_kn, 1.f, /*trans_b=*/true));
  };
}
Tensor int4_linear(const Tensor& x, const Tensor& packed, const Tensor& scales,
                   const Tensor& zeros, int group_size) {
  NDArray out = tc::int4_linear(x.data(), packed.data(), scales.data(),
                                zeros.data(), group_size);
  return Tensor::from_op(out, {x}, "int4_linear",
                         int4_grad(x, packed, scales, zeros, group_size));
}
Tensor int4_linear_fused(const Tensor& x, const Tensor& packed,
                         const Tensor& scales, const Tensor& zeros,
                         int group_size) {
  NDArray out = tc::int4_linear_fused(x.data(), packed.data(), scales.data(),
                                      zeros.data(), group_size);
  return Tensor::from_op(out, {x}, "int4_linear_fused",
                         int4_grad(x, packed, scales, zeros, group_size));
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

// ------------------------------------------------------------- Phase 2
Tensor pow_scalar(const Tensor& a, double p) {
  NDArray out = ew_pow(a.data(), p);
  return Tensor::from_op(out, {a}, "pow", [a, p](const NDArray& g) {
    // d/dx x^p = p * x^(p-1)
    a.v->accumulate_grad(nmul(g, nmuls(ew_pow(a.data(), p - 1.0), p)));
  });
}
Tensor sub_scalar(const Tensor& a, double s) { return add_scalar(a, -s); }

Tensor compare(const Tensor& a, const Tensor& b, int op) {
  return Tensor::make(tc::compare(a.data(), b.data(), op), false);
}
Tensor compare_scalar(const Tensor& a, double s, int op) {
  return Tensor::make(tc::compare_scalar(a.data(), s, op), false);
}

Tensor where(const Tensor& cond, const Tensor& x, const Tensor& y) {
  NDArray out = where_nd(cond.data(), x.data(), y.data());
  NDArray c = cond.data();  // constant
  return Tensor::from_op(out, {x, y}, "where", [x, y, c](const NDArray& g) {
    NDArray z = NDArray::zeros(g.shape, g.dtype, g.device);
    x.v->accumulate_grad(where_nd(c, g, z));
    y.v->accumulate_grad(where_nd(c, z, g));
  });
}
Tensor masked_fill(const Tensor& a, const Tensor& mask, double value) {
  NDArray fill = NDArray::full(a.shape(), value, a.dtype(), a.device());
  NDArray out = where_nd(mask.data(), fill, a.data());
  NDArray m = mask.data();
  return Tensor::from_op(out, {a}, "masked_fill", [a, m](const NDArray& g) {
    NDArray z = NDArray::zeros(g.shape, g.dtype, g.device);
    a.v->accumulate_grad(where_nd(m, z, g));  // grad only where not filled
  });
}

Tensor max(const Tensor& a, const std::vector<int>& axes, bool keepdim) {
  NDArray out = reduce_max(a.data(), axes, keepdim);
  NDArray mxk = reduce_max(a.data(), axes, true);  // keepdim for broadcasting
  Shape in_shape = a.shape();
  Shape kshape = keepdim_shape(in_shape, axes);
  std::vector<int> ax = axes;
  return Tensor::from_op(out, {a}, "max", [a, mxk, in_shape, kshape, ax](const NDArray& g) {
    NDArray mask = tc::compare(a.data(), broadcast_to(mxk, in_shape), 4);  // ==max
    NDArray cnt = broadcast_to(reduce_sum(mask, ax, true), in_shape);
    NDArray gk = broadcast_to(g.reshape(kshape), in_shape);
    a.v->accumulate_grad(ndiv(nmul(gk, mask), cnt));
  });
}
Tensor min(const Tensor& a, const std::vector<int>& axes, bool keepdim) {
  NDArray out = reduce_min(a.data(), axes, keepdim);
  NDArray mnk = reduce_min(a.data(), axes, true);
  Shape in_shape = a.shape();
  Shape kshape = keepdim_shape(in_shape, axes);
  std::vector<int> ax = axes;
  return Tensor::from_op(out, {a}, "min", [a, mnk, in_shape, kshape, ax](const NDArray& g) {
    NDArray mask = tc::compare(a.data(), broadcast_to(mnk, in_shape), 4);
    NDArray cnt = broadcast_to(reduce_sum(mask, ax, true), in_shape);
    NDArray gk = broadcast_to(g.reshape(kshape), in_shape);
    a.v->accumulate_grad(ndiv(nmul(gk, mask), cnt));
  });
}
Tensor var(const Tensor& a, const std::vector<int>& axes, bool keepdim) {
  Tensor m = mean(a, axes, /*keepdim=*/true);
  Tensor d = sub(a, m);
  return mean(mul(d, d), axes, keepdim);  // population variance (ddof=0)
}
Tensor std(const Tensor& a, const std::vector<int>& axes, bool keepdim) {
  return sqrt(var(a, axes, keepdim));
}
Tensor prod(const Tensor& a, const std::vector<int>& axes, bool keepdim) {
  NDArray out = reduce_prod(a.data(), axes, keepdim);
  NDArray pk = reduce_prod(a.data(), axes, true);
  Shape in_shape = a.shape();
  Shape kshape = keepdim_shape(in_shape, axes);
  return Tensor::from_op(out, {a}, "prod", [a, pk, in_shape, kshape](const NDArray& g) {
    NDArray gk = broadcast_to(g.reshape(kshape), in_shape);
    NDArray pb = broadcast_to(pk.reshape(kshape), in_shape);
    a.v->accumulate_grad(ndiv(nmul(gk, pb), a.data()));  // g * prod / x
  });
}
Tensor argmax(const Tensor& a, int axis) { return Tensor::make(reduce_arg(a.data(), axis, true), false); }
Tensor argmin(const Tensor& a, int axis) { return Tensor::make(reduce_arg(a.data(), axis, false), false); }
Tensor cumsum(const Tensor& a, int axis) {
  NDArray out = cumsum_nd(a.data(), axis);
  return Tensor::from_op(out, {a}, "cumsum", [a, axis](const NDArray& g) {
    // grad_i = sum_{j>=i} g_j = flip(cumsum(flip(g)))
    a.v->accumulate_grad(flip_nd(cumsum_nd(flip_nd(g, {axis}), axis), {axis}));
  });
}
Tensor gather(const Tensor& a, int dim, const Tensor& index) {
  NDArray out = gather_nd(a.data(), dim, index.data());
  NDArray idx = index.data();
  Shape in_shape = a.shape();
  DType dt = a.dtype();
  return Tensor::from_op(out, {a}, "gather", [a, idx, in_shape, dt, dim](const NDArray& g) {
    a.v->accumulate_grad(scatter_add_nd(in_shape, dt, dim, idx, g));
  });
}
Tensor flip(const Tensor& a, const std::vector<int>& dims) {
  NDArray out = flip_nd(a.data(), dims);
  return Tensor::from_op(out, {a}, "flip", [a, dims](const NDArray& g) {
    a.v->accumulate_grad(flip_nd(g, dims));
  });
}
std::tuple<Tensor, Tensor> topk(const Tensor& a, int k, bool largest) {
  NDArray vals_nd, idx_nd;
  std::tie(vals_nd, idx_nd) = tc::topk_nd(a.data(), k, largest);
  Shape in_shape = a.shape();
  NDArray idx = idx_nd;
  DType dt = a.dtype();
  int dim = a.ndim() - 1;
  Tensor values = Tensor::from_op(vals_nd, {a}, "topk", [a, idx, in_shape, dt, dim](const NDArray& g) {
    a.v->accumulate_grad(scatter_add_nd(in_shape, dt, dim, idx, g));
  });
  return {values, Tensor::make(idx_nd, false)};
}

Tensor permute(const Tensor& a, const std::vector<int>& dims) {
  NDArray out = tc::permute(a.data(), dims);
  std::vector<int> inv(dims.size());
  int nd = (int)dims.size();
  for (int d = 0; d < nd; ++d) { int sd = dims[d] < 0 ? dims[d] + nd : dims[d]; inv[sd] = d; }
  return Tensor::from_op(out, {a}, "permute", [a, inv](const NDArray& g) {
    a.v->accumulate_grad(tc::permute(g, inv));
  });
}
Tensor transpose(const Tensor& a, int dim0, int dim1) {
  int nd = a.ndim();
  std::vector<int> dims(nd);
  for (int d = 0; d < nd; ++d) dims[d] = d;
  int i = dim0 < 0 ? dim0 + nd : dim0, j = dim1 < 0 ? dim1 + nd : dim1;
  int tmp = dims[i]; dims[i] = dims[j]; dims[j] = tmp;
  return permute(a, dims);
}
Tensor squeeze(const Tensor& a, int dim) {
  int nd = a.ndim();
  int d = dim < 0 ? dim + nd : dim;
  Shape s;
  for (int i = 0; i < nd; ++i) if (i != d || a.shape()[i] != 1) s.push_back(a.shape()[i]);
  return reshape(a, s);
}
Tensor unsqueeze(const Tensor& a, int dim) {
  int nd = a.ndim();
  int d = dim < 0 ? dim + nd + 1 : dim;
  Shape s = a.shape();
  s.insert(s.begin() + d, 1);
  return reshape(a, s);
}
Tensor expand(const Tensor& a, const Shape& shape) {
  NDArray out = broadcast_to(a.data(), shape);
  Shape in_shape = a.shape();
  return Tensor::from_op(out, {a}, "expand", [a, in_shape](const NDArray& g) {
    a.v->accumulate_grad(reduce_to(g, in_shape));
  });
}
Tensor flatten(const Tensor& a, int start_dim, int end_dim) {
  int nd = a.ndim();
  int s = start_dim < 0 ? start_dim + nd : start_dim;
  int e = end_dim < 0 ? end_dim + nd : end_dim;
  Shape out;
  for (int d = 0; d < s; ++d) out.push_back(a.shape()[d]);
  int64_t merged = 1;
  for (int d = s; d <= e; ++d) merged *= a.shape()[d];
  out.push_back(merged);
  for (int d = e + 1; d < nd; ++d) out.push_back(a.shape()[d]);
  return reshape(a, out);
}
Tensor cat(const std::vector<Tensor>& ts, int dim) {
  std::vector<NDArray> datas;
  for (auto& t : ts) datas.push_back(t.data());
  NDArray out = cat_nd(datas, dim);
  int nd = ts[0].ndim();
  int d = dim < 0 ? dim + nd : dim;
  std::vector<int64_t> sizes;
  for (auto& t : ts) sizes.push_back(t.shape()[d]);
  return Tensor::from_op(out, ts, "cat", [ts, d, sizes](const NDArray& g) {
    int64_t start = 0;
    for (size_t i = 0; i < ts.size(); ++i) {
      ts[i].v->accumulate_grad(slice_nd(g, d, start, sizes[i]));
      start += sizes[i];
    }
  });
}
Tensor stack(const std::vector<Tensor>& ts, int dim) {
  std::vector<Tensor> expanded;
  for (auto& t : ts) expanded.push_back(unsqueeze(t, dim));
  return cat(expanded, dim);
}

Tensor log_softmax(const Tensor& a, int axis) {
  NDArray mx = reduce_max(a.data(), {axis}, true);  // constant
  Tensor shifted = sub(a, Tensor::make(mx, false));
  Tensor s = sum(exp(shifted), {axis}, true);
  return sub(shifted, log(s));
}
Tensor cross_entropy(const Tensor& logits, const Tensor& onehot) {
  Tensor ls = log_softmax(logits, -1);
  Tensor per = sum(mul(onehot, ls), {-1}, false);
  return mul_scalar(mean(per, {}, false), -1.0);
}

Tensor detach(const Tensor& a) { return Tensor::make(a.data(), false); }

Tensor cast(const Tensor& a, DType dt) {
  if (dt == a.dtype()) return a;
  NDArray out = a.data().astype(dt);
  DType src = a.dtype();
  return Tensor::from_op(out, {a}, "cast", [a, src](const NDArray& g) {
    a.v->accumulate_grad(g.astype(src));
  });
}

Tensor im2col(const Tensor& a, int kh, int kw, int sh, int sw, int ph, int pw) {
  NDArray out = tc::im2col(a.data(), kh, kw, sh, sw, ph, pw);
  Shape xs = a.shape();
  return Tensor::from_op(out, {a}, "im2col", [a, xs, kh, kw, sh, sw, ph, pw](const NDArray& g) {
    a.v->accumulate_grad(tc::col2im(g, xs, kh, kw, sh, sw, ph, pw));
  });
}
Tensor col2im(const Tensor& cols, int64_t N, int64_t C, int64_t OH, int64_t OW,
              int kh, int kw, int sh, int sw, int ph, int pw) {
  Shape xs = {N, C, OH, OW};
  NDArray out = tc::col2im(cols.data(), xs, kh, kw, sh, sw, ph, pw);
  return Tensor::from_op(out, {cols}, "col2im", [cols, kh, kw, sh, sw, ph, pw](const NDArray& g) {
    cols.v->accumulate_grad(tc::im2col(g, kh, kw, sh, sw, ph, pw));
  });
}
Tensor avg_pool2d(const Tensor& a, int kh, int kw, int sh, int sw, int ph, int pw) {
  NDArray out = tc::avgpool2d(a.data(), kh, kw, sh, sw, ph, pw);
  Shape xs = a.shape();
  return Tensor::from_op(out, {a}, "avg_pool2d", [a, xs, kh, kw, sh, sw, ph, pw](const NDArray& g) {
    a.v->accumulate_grad(tc::avgpool2d_bwd(g, xs, kh, kw, sh, sw, ph, pw));
  });
}
Tensor max_pool2d(const Tensor& a, int kh, int kw, int sh, int sw, int ph, int pw) {
  NDArray argmax;
  NDArray out = tc::maxpool2d(a.data(), kh, kw, sh, sw, ph, pw, argmax);
  Shape xs = a.shape();
  return Tensor::from_op(out, {a}, "max_pool2d", [a, argmax, xs](const NDArray& g) {
    a.v->accumulate_grad(tc::maxpool2d_bwd(g, argmax, xs));
  });
}

Tensor embedding(const Tensor& weight, const Tensor& idx) {
  NDArray out = embedding_forward(weight.data(), idx.data());
  NDArray idx_nd = idx.data();
  Shape wshape = weight.shape();
  DType wdt = weight.dtype();
  return Tensor::from_op(out, {weight}, "embedding", [weight, idx_nd, wshape, wdt](const NDArray& g) {
    weight.v->accumulate_grad(embedding_backward(g, idx_nd, wshape, wdt));
  });
}

}  // namespace ops
}  // namespace tc
