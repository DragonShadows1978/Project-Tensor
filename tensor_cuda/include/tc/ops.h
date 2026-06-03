// ops.h: differentiable operations on Tensor. Each builds the forward NDArray
// result and registers a grad_fn implementing the vector-Jacobian product.

#pragma once

#include "tc/autograd.h"

namespace tc {
namespace ops {

// Arithmetic (broadcasting). Scalar overloads avoid allocating a full tensor.
Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);
Tensor add_scalar(const Tensor& a, double s);
Tensor mul_scalar(const Tensor& a, double s);
Tensor neg(const Tensor& a);

// Math / activations.
Tensor exp(const Tensor& a);
Tensor log(const Tensor& a);
Tensor sqrt(const Tensor& a);
Tensor relu(const Tensor& a);
Tensor sigmoid(const Tensor& a);
Tensor tanh(const Tensor& a);
Tensor gelu(const Tensor& a);
Tensor silu(const Tensor& a);

// Linear algebra.
Tensor matmul(const Tensor& a, const Tensor& b);

// Reductions.
Tensor sum(const Tensor& a, const std::vector<int>& axes, bool keepdim);
Tensor mean(const Tensor& a, const std::vector<int>& axes, bool keepdim);

// Shape.
Tensor reshape(const Tensor& a, const Shape& shape);
Tensor transpose_last(const Tensor& a);

// Composite (built from primitives, autograd flows automatically).
Tensor softmax(const Tensor& a, int axis);
Tensor mse_loss(const Tensor& pred, const Tensor& target);

}  // namespace ops
}  // namespace tc
