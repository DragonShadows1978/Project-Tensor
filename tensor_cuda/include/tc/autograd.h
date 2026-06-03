// autograd.h: reverse-mode automatic differentiation.
//
// Mirrors the reference Python design (each op output stores a `_backward`
// closure and its parents), but with a cycle-free ownership model:
//
//   * A Variable owns its data/grad NDArrays, a list of parent Variables
//     (shared_ptr, forming a DAG: outputs -> inputs), and a grad_fn closure.
//   * grad_fn takes the output gradient as an argument (it does NOT capture the
//     output Variable), so closures only ever capture inputs / saved tensors.
//     That keeps the graph an acyclic shared_ptr DAG with no leaks.
//   * Tensor is a thin value-semantics handle around shared_ptr<Variable>.

#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "tc/core.h"

namespace tc {

struct Variable;
using VarPtr = std::shared_ptr<Variable>;

// grad_fn receives the gradient flowing into this node's output.
using GradFn = std::function<void(const NDArray& grad_out)>;

struct Variable {
  NDArray data;
  NDArray grad;                 // accumulated; undefined until first contribution
  bool requires_grad = false;
  GradFn grad_fn;               // null for leaves
  std::vector<VarPtr> parents;  // inputs that this node depends on
  const char* op = "leaf";

  void accumulate_grad(const NDArray& g);  // grad += unbroadcast(g, data.shape)
};

class Tensor {
 public:
  VarPtr v;

  Tensor() = default;
  explicit Tensor(VarPtr v) : v(std::move(v)) {}

  static Tensor make(NDArray data, bool requires_grad);
  // Build an op output with its backward closure and parents.
  static Tensor from_op(NDArray data, std::vector<Tensor> parents,
                        const char* op, GradFn grad_fn);

  // accessors
  const NDArray& data() const { return v->data; }
  NDArray& data() { return v->data; }
  const NDArray& grad() const { return v->grad; }
  bool requires_grad() const { return v->requires_grad; }
  void set_requires_grad(bool r) { v->requires_grad = r; }
  const Shape& shape() const { return v->data.shape; }
  int ndim() const { return v->data.ndim(); }
  DType dtype() const { return v->data.dtype; }
  Device device() const { return v->data.device; }
  bool defined() const { return v != nullptr; }

  void backward();                 // seed grad = ones, run engine
  void backward(const NDArray& seed);
  void zero_grad();
};

// Global no_grad flag (mirrors is_grad_enabled / no_grad).
bool grad_enabled();
void set_grad_enabled(bool enabled);

struct NoGradGuard {
  bool prev;
  NoGradGuard() : prev(grad_enabled()) { set_grad_enabled(false); }
  ~NoGradGuard() { set_grad_enabled(prev); }
};

}  // namespace tc
