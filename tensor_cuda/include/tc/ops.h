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
Tensor abs(const Tensor& a);
Tensor sin(const Tensor& a);
Tensor cos(const Tensor& a);
Tensor reciprocal(const Tensor& a);
Tensor clamp(const Tensor& a, double lo, double hi);
Tensor maximum(const Tensor& a, const Tensor& b);
Tensor minimum(const Tensor& a, const Tensor& b);
Tensor slice(const Tensor& a, int dim, int64_t start, int64_t len);

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

// ---- Phase 2: broadened op surface ----
Tensor pow_scalar(const Tensor& a, double exponent);
Tensor sub_scalar(const Tensor& a, double s);  // a - s

// Comparisons return detached (non-differentiable) 0/1 tensors.
Tensor compare(const Tensor& a, const Tensor& b, int op);
Tensor compare_scalar(const Tensor& a, double s, int op);

Tensor where(const Tensor& cond, const Tensor& x, const Tensor& y);
Tensor masked_fill(const Tensor& a, const Tensor& mask, double value);

// Reductions with gradients.
Tensor max(const Tensor& a, const std::vector<int>& axes, bool keepdim);
Tensor min(const Tensor& a, const std::vector<int>& axes, bool keepdim);
Tensor var(const Tensor& a, const std::vector<int>& axes, bool keepdim);
Tensor std(const Tensor& a, const std::vector<int>& axes, bool keepdim);

// Shape.
Tensor permute(const Tensor& a, const std::vector<int>& dims);
Tensor transpose(const Tensor& a, int dim0, int dim1);
Tensor squeeze(const Tensor& a, int dim);
Tensor unsqueeze(const Tensor& a, int dim);
Tensor expand(const Tensor& a, const Shape& shape);
Tensor flatten(const Tensor& a, int start_dim, int end_dim);
Tensor cat(const std::vector<Tensor>& ts, int dim);
Tensor stack(const std::vector<Tensor>& ts, int dim);

// Losses / log-domain.
Tensor log_softmax(const Tensor& a, int axis);
Tensor cross_entropy(const Tensor& logits, const Tensor& onehot_target);

// Indexing.
Tensor embedding(const Tensor& weight, const Tensor& idx);  // idx: int64 Tensor

// Stop-gradient: returns a constant view sharing storage (no autograd parents).
Tensor detach(const Tensor& a);

}  // namespace ops
}  // namespace tc
