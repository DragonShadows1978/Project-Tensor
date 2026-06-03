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
