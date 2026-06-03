// autograd.cpp: Tensor construction and the reverse-mode backward engine.

#include "tc/autograd.h"

#include <unordered_set>
#include <vector>

namespace tc {

static bool g_grad_enabled = true;
bool grad_enabled() { return g_grad_enabled; }
void set_grad_enabled(bool e) { g_grad_enabled = e; }

void Variable::accumulate_grad(const NDArray& g) {
  NDArray gr = reduce_to(g, data.shape);  // handle broadcasting (no-op if equal)
  if (!grad.defined()) grad = gr;
  else grad = ew_binary(grad, gr, /*add=*/0);
}

Tensor Tensor::make(NDArray data, bool requires_grad) {
  auto v = std::make_shared<Variable>();
  v->data = std::move(data);
  v->requires_grad = requires_grad;
  v->op = "leaf";
  return Tensor(v);
}

Tensor Tensor::from_op(NDArray data, std::vector<Tensor> parents,
                       const char* op, GradFn grad_fn) {
  bool req = false;
  if (grad_enabled())
    for (auto& p : parents) if (p.v && p.v->requires_grad) { req = true; break; }

  auto v = std::make_shared<Variable>();
  v->data = std::move(data);
  v->requires_grad = req;
  v->op = op;
  if (req) {
    for (auto& p : parents)
      if (p.v && p.v->requires_grad) v->parents.push_back(p.v);
    v->grad_fn = std::move(grad_fn);
  }
  return Tensor(v);
}

void Tensor::backward() {
  backward(NDArray::full(v->data.shape, 1.0, v->data.dtype, v->data.device));
}

void Tensor::backward(const NDArray& seed) {
  // Post-order DFS over the dependency DAG (parents), iterative to avoid deep
  // recursion. order ends with the root; we process it reversed (root first).
  std::vector<Variable*> order;
  std::unordered_set<Variable*> visited;
  std::vector<std::pair<Variable*, size_t>> stack;
  stack.push_back({v.get(), 0});
  visited.insert(v.get());
  while (!stack.empty()) {
    auto& top = stack.back();
    if (top.second < top.first->parents.size()) {
      Variable* p = top.first->parents[top.second].get();
      top.second++;
      if (p && p->requires_grad && !visited.count(p)) {
        visited.insert(p);
        stack.push_back({p, 0});
      }
    } else {
      order.push_back(top.first);
      stack.pop_back();
    }
  }

  v->accumulate_grad(seed);
  for (auto it = order.rbegin(); it != order.rend(); ++it) {
    Variable* node = *it;
    if (node->grad_fn && node->grad.defined()) node->grad_fn(node->grad);
  }
}

void Tensor::zero_grad() { v->grad = NDArray(); }

}  // namespace tc
