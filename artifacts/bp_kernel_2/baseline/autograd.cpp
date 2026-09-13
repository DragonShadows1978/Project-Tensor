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
    // An op node's grad is consumed exactly once — by the grad_fn call
    // above (post-order guarantees it was fully accumulated first). Freeing
    // it here keeps backward's live set to one grad wave instead of the
    // whole graph's, halving peak memory on deep graphs (62-layer reader
    // backward OOM'd an 8 GB card without this). Leaves (null grad_fn)
    // keep grads for the optimizer; re-backward still accumulates
    // correctly on leaves (and no longer double-counts op nodes).
    if (node->grad_fn) node->grad = NDArray();
  }
}

void Tensor::zero_grad() { v->grad = NDArray(); }

}  // namespace tc
