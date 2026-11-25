// autograd.cpp: Tensor construction and the reverse-mode backward engine.

#include "tc/autograd.h"
#include "tc/bp_op_timing.h"

#include <atomic>
#include <optional>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tc {

// Prior art: PyTorch GradMode (Paszke et al., 2019), thread-local mode.
// TensorCUDA retains the global setting and overrides it only in a VJP.
static std::atomic<bool> g_grad_enabled{true};
static thread_local std::optional<bool> callback_grad_mode;
bool grad_enabled() {
  return callback_grad_mode.value_or(g_grad_enabled.load());
}
void set_grad_enabled(bool e) {
  if (callback_grad_mode) callback_grad_mode = e;
  else g_grad_enabled.store(e);
}

namespace {
std::atomic_flag backward_lane = ATOMIC_FLAG_INIT;
thread_local bool backward_in_progress = false;

struct BackwardLane {
  const bool outermost = !backward_in_progress;
  BackwardLane() {
    if (callback_grad_mode)
      throw std::runtime_error("attach_vjp: re-entrant backward is not supported");
    // Checkpoint replay may traverse on the owning thread.
    if (outermost) {
      if (backward_lane.test_and_set())
        throw std::runtime_error("backward: only one host dispatch lane is supported");
      backward_in_progress = true;
    }
  }
  ~BackwardLane() {
    if (outermost) {
      backward_in_progress = false;
      backward_lane.clear();
    }
  }
};

struct CallbackNoGrad {
  CallbackNoGrad() { callback_grad_mode = false; }
  ~CallbackNoGrad() { callback_grad_mode.reset(); }
};

bool same_device(Device a, Device b) {
  return a.type == b.type && a.index == b.index;
}

// Compare half-open allocation ranges by subtraction to avoid end-address
// overflow. No specific prior art known to me for this TensorCUDA check.
bool storage_overlaps(const Storage& a, const Storage& b) {
  if (&a == &b) return true;
  if (!a.ptr || !b.ptr || !a.nbytes || !b.nbytes) return false;
  const auto ap = reinterpret_cast<uintptr_t>(a.ptr);
  const auto bp = reinterpret_cast<uintptr_t>(b.ptr);
  return ap <= bp ? bp - ap < a.nbytes : ap - bp < b.nbytes;
}
}  // namespace

struct AttachedVjp {
  std::string name;
  VjpFn callback;
  void* callback_gc_ref = nullptr;
  std::vector<bool> active;
  std::vector<uint64_t> revisions;
  uint64_t value_revision;

  std::string prefix() const { return "attach_vjp(" + name + "): "; }
  VjpFn take_callback() noexcept {
    callback_gc_ref = nullptr;
    return std::exchange(callback, {});
  }
};

void Variable::accumulate_grad(const NDArray& g) {
  // Prior art: CUDA event scopes / gprof exclusive accounting (see header).
  bp::Scope scope(bp::enabled ? (op == "leaf" ? "dense_leaf_gradient_accumulation"
                                                   : "intermediate_gradient_accumulation") : "");
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

// Prior art: Linnainmaa (1970) reverse accumulation, PyTorch autograd.Function
// (2019) and JAX custom_vjp (Bradbury et al., 2018) custom reverse rules and
// saved state. Ours: explicit saved slots, Storage revisions and one-shot
// callback ownership on the existing TensorCUDA DAG.
Tensor Tensor::attach_vjp(const Tensor& value, const std::vector<Tensor>& inputs,
                          VjpFn vjp, std::string name, void* callback_gc_ref) {
  auto state = std::make_shared<AttachedVjp>();
  state->name = std::move(name);
  if (!value.defined() || !value.data().defined() || value.requires_grad() ||
      value.v->grad_fn || value.grad().defined())
    throw std::invalid_argument(state->prefix() + "value must be a fresh detached output");

  bool any_active = false;
  for (const auto& input : inputs) {
    const auto slot = state->active.size();
    if (!input.defined() || !input.data().defined() || !same_device(input.device(), value.device()))
      throw std::invalid_argument(state->prefix() + "input slot " + std::to_string(slot) +
                                  " must be defined on the value's device");
    if (storage_overlaps(*input.data().storage, *value.data().storage))
      throw std::invalid_argument(state->prefix() + "value must have storage separate from its inputs");
    state->active.push_back(input.requires_grad());
    state->revisions.push_back(input.data().revision());
    any_active |= input.requires_grad();
  }
  if (!grad_enabled() || !any_active) return value;

  state->callback = std::move(vjp);
  state->callback_gc_ref = callback_gc_ref;
  state->value_revision = value.data().revision();
  auto result = from_op(value.data(), inputs, state->name.c_str(),
      [state, inputs, saved_value = value.data()](const NDArray& dy) {
        CallbackNoGrad no_grad;
        // Move the only callback owner into this invocation. Unwinding,
        // including a stale save or a Python exception, releases it too.
        auto callback = state->take_callback();
        if (!callback)
          throw std::runtime_error(state->prefix() + "callback already consumed");
        auto check_revisions = [&] {
          for (size_t i = 0; i < inputs.size(); ++i)
            if (inputs[i].data().revision() != state->revisions[i])
              throw std::runtime_error(state->prefix() + "saved input slot " + std::to_string(i) +
                                       " was modified after attachment");
          if (saved_value.revision() != state->value_revision)
            throw std::runtime_error(state->prefix() + "saved value was modified after attachment");
        };
        check_revisions();
        auto grads = callback(dy, state->active);
        if (grads.size() != inputs.size())
          throw std::invalid_argument(state->prefix() + "wrong gradient count");
        for (size_t i = 0; i < inputs.size(); ++i) {
          const auto& g = grads[i];
          if (!g.defined()) continue;
          const auto slot = state->prefix() + "gradient slot " + std::to_string(i) + ": ";
          if (!state->active[i]) throw std::invalid_argument(slot + "frozen input needs None");
          if (g.shape != inputs[i].shape()) throw std::invalid_argument(slot + "shape mismatch");
          if (g.dtype != inputs[i].dtype()) throw std::invalid_argument(slot + "dtype mismatch");
          if (!same_device(g.device, inputs[i].device())) throw std::invalid_argument(slot + "device mismatch");
        }
        check_revisions();
        for (size_t i = 0; i < inputs.size(); ++i)
          if (grads[i].defined()) inputs[i].v->accumulate_grad(grads[i]);
      });
  if (result.v->grad_fn) result.v->attached_vjp = std::move(state);
  return result;
}

void* Tensor::attached_vjp_gc_ref() const noexcept {
  // CPython GC must subtract precisely the reference owned by this node.
  // A native alias or descendant may still execute it, so treat the edge
  // as an external root until there is one owning wrapper.
  return v && v.use_count() == 1 && v->attached_vjp
      ? v->attached_vjp->callback_gc_ref : nullptr;
}

void Tensor::clear_attached_vjp_gc_ref() noexcept {
  if (attached_vjp_gc_ref()) v->attached_vjp->take_callback();
}

void Tensor::backward() {
  backward_impl(nullptr);
}

void Tensor::backward(const NDArray& seed) {
  backward_impl(&seed);
}

void Tensor::backward_impl(const NDArray* seed) {
  BackwardLane lane;
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

  // Prior art: Abrahams (1998), exception safety. Preflight one-shot nodes
  // before seeding, and roll back temporary gradients if execution throws.
  for (const auto* node : order)
    if (node->attached_vjp && !node->attached_vjp->callback)
      throw std::runtime_error(node->attached_vjp->prefix() + "callback already consumed");

  try {
    if (seed) v->accumulate_grad(*seed);
    else v->accumulate_grad(NDArray::full(v->data.shape, 1.0, v->data.dtype, v->data.device));
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
      Variable* node = *it;
      if (!node->grad_fn) continue;
      if (node->grad.defined()) {
        if (bp::enabled) {
          bp::Scope scope(std::string("vjp:") + node->op);
          node->grad_fn(node->grad);
        } else {
          node->grad_fn(node->grad);
        }
      }
      // Free the temporary after its last use. Leaves retain their gradients.
      node->grad = NDArray();
    }
  } catch (...) {
    for (auto* node : order)
      if (node->grad_fn) node->grad = NDArray();
    throw;
  }
}

void Tensor::zero_grad() { v->grad = NDArray(); }

}  // namespace tc
