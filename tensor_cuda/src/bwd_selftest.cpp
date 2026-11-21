// Prior art: PyTorch's private torch._C test entries (Paszke et al., 2019)
// and Kitware's CTest build integration (2000s). Both supply the pattern of
// building tests with their target. Here the r4 probes exercise the engine
// loaded by Python and expose their individual outcomes through one binding.
#include <pybind11/pybind11.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "tc/autograd.h"

namespace py = pybind11;
using namespace pybind11::literals;

// The implementation remains in ops.cpp and has extension-local visibility.
namespace tc::ops {
NDArray matmul_backward(const NDArray& a, const NDArray& b, float alpha, bool trans_b);
}

namespace {
using namespace tc;

// Prior art: PyTorch GradMode guards (Paszke et al., 2019). This guard
// enables recording for the self-test and restores the caller's mode.
struct SelfTestGradGuard {
  const bool previous = grad_enabled();
  SelfTestGradGuard() { set_grad_enabled(true); }
  ~SelfTestGradGuard() { set_grad_enabled(previous); }
};

void no_release(void*) {}

Tensor host_view(float* begin, bool active) {
  NDArray a;
  a.device = {DeviceType::CPU, 0};
  a.shape = {3};
  a.storage = std::make_shared<Storage>(begin, 3 * sizeof(float), a.device, no_release, nullptr);
  return Tensor::make(std::move(a), active);
}

std::vector<NDArray> identity(const NDArray& dy, const std::vector<bool>&) {
  return {dy};
}

template <class Error, class F>
std::string expect_error(F&& fn, std::string_view message) {
  try {
    fn();
  } catch (const Error& e) {
    if (std::string_view(e.what()).find(message) == std::string_view::npos) throw;
    return e.what();
  }
  throw std::runtime_error("expected error missing: " + std::string(message));
}

template <class F>
void check(py::dict& checks, const char* name, F&& fn) {
  try {
    checks[py::str(name)] = py::dict("passed"_a = true, "detail"_a = fn());
  } catch (const std::exception& e) {
    checks[py::str(name)] = py::dict("passed"_a = false, "detail"_a = e.what());
  }
}

void partial_overlap(py::dict& checks) {
  std::array<float, 7> owner{1, 2, -3, -4, 5, 6, 7};
  const auto x = host_view(owner.data(), true);
  const auto adjacent = host_view(owner.data() + 3, false);
  const auto right = host_view(owner.data() + 2, false);
  const auto left = host_view(owner.data() + 2, true);
  check(checks, "adjacent", [&] {
    if (!Tensor::attach_vjp(adjacent, {x}, identity, "adjacent").requires_grad())
      throw std::runtime_error("adjacent ranges should be accepted");
    return "adjacent ranges accepted";
  });
  check(checks, "distinct-storage", [&] {
    if (x.data().storage == right.data().storage || x.data().data_ptr() == right.data().data_ptr())
      throw std::runtime_error("fixture does not use distinct, offset Storages");
    return "distinct Storage objects with different starting addresses";
  });
  check(checks, "right-overlap", [&] {
    return expect_error<std::invalid_argument>(
        [&] { Tensor::attach_vjp(right, {x}, identity, "right-overlap"); }, "storage separate");
  });
  check(checks, "left-overlap", [&] {
    return expect_error<std::invalid_argument>(
        [&] { Tensor::attach_vjp(host_view(owner.data(), false), {left}, identity, "left-overlap"); },
        "storage separate");
  });
}

void batch_mismatch(py::dict& checks) {
  check(checks, "batch-mismatch", [] {
    const Device cuda{DeviceType::CUDA, 0};
    // Empty matrices construct without allocation. The shape guard must run
    // before the helper queries the current CUDA device.
    const NDArray a({2, 2, 0}, DType::Float32, cuda);
    const NDArray b({3, 0, 3}, DType::Float32, cuda);
    return expect_error<std::runtime_error>([&] { ops::matmul_backward(a, b, 1.f, false); },
                                           "batch size mismatch");
  });
}

int current_cuda_device() {
  int current;
  const auto status = cudaGetDevice(&current);
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("cudaGetDevice: ") + cudaGetErrorString(status));
  return current;
}

void current_device(py::dict& checks) {
  check(checks, "current-device", [] {
    // Metadata-only inputs exercise the guard on a one-GPU lease without
    // submitting work to the foreign device.
    const Device foreign{DeviceType::CUDA, current_cuda_device() + 1};
    const NDArray a({2, 0}, DType::Float32, foreign);
    const NDArray b({0, 3}, DType::Float32, foreign);
    return expect_error<std::runtime_error>([&] { ops::matmul_backward(a, b, 1.f, false); },
                                           "current CUDA device");
  });
}

void zero_k(py::dict& checks) {
  check(checks, "zero-k", [] {
    const Device cuda{DeviceType::CUDA, current_cuda_device()};
    const NDArray a({2, 0}, DType::Float32, cuda);
    const NDArray b({0, 3}, DType::Float32, cuda);
    auto out = ops::matmul_backward(a, b, 1.f, false);
    if (out.shape != Shape{2, 3}) throw std::runtime_error("wrong K=0 shape");
    std::array<uint32_t, 6> words;
    out.to_host(words.data());
    if (!std::all_of(words.begin(), words.end(), [](uint32_t x) { return x == 0; }))
      throw std::runtime_error("K=0 output must contain exact positive zeros");
    return "shape [2, 3] and six exact positive zeros";
  });
}

bool has_cuda_device() {
  const auto* visible = std::getenv("CUDA_VISIBLE_DEVICES");
  if (visible && *visible == '\0') return false;
  int count = 0;
  const auto status = cudaGetDeviceCount(&count);
  if (status == cudaSuccess) return count > 0;
  // CUDA Runtime (NVIDIA, 2007 onward): consume the failed probe's last error.
  cudaGetLastError();
  if (status == cudaErrorNoDevice) return false;
  if (status == cudaErrorInsufficientDriver) {
    int version = 0;
    if (cudaDriverGetVersion(&version) == cudaSuccess && version == 0) return false;
  }
  throw std::runtime_error(std::string("cudaGetDeviceCount: ") + cudaGetErrorString(status));
}

py::dict run_case(const std::string& name) {
  const SelfTestGradGuard grad_on;
  py::dict checks;
  if (name == "overlap") partial_overlap(checks);
  else if (name == "batch-mismatch") batch_mismatch(checks);
  else if (name == "current-device" || name == "zero-k") {
    if (!has_cuda_device())
      return py::dict("case"_a = name, "status"_a = "skipped", "checks"_a = checks,
                      "reason"_a = "no CUDA device present");
    if (name == "current-device") current_device(checks);
    else zero_k(checks);
  } else {
    throw std::invalid_argument("unknown backward native self-test case: " + name);
  }
  const bool passed = std::all_of(checks.begin(), checks.end(), [](const auto& item) {
    return py::cast<bool>(item.second["passed"]);
  });
  return py::dict("case"_a = name, "status"_a = (passed ? "passed" : "failed"), "checks"_a = checks);
}
}  // namespace

void bind_bwd_native_selftest(py::module_& module) {
  module.def("_bwd_native_selftest", &run_case, py::arg("case"),
             "Run a private native backward invariant check in the loaded engine.");
}
