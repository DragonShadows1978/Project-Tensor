// Author baseline, not GPU evidence. Prior art: PT-TF32 host contract (2026),
// take raw CPU NDArray construction to avoid CUDA calls in transfer helpers.
#include "tc/core.h"
#include <cstring>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  bool expected = std::string(argv[1]) == "1";
  if (tc::get_deterministic_embed_bwd() != expected)
    throw std::runtime_error("environment initial mode differs");
  tc::set_deterministic_embed_bwd(!expected);
  if (tc::get_deterministic_embed_bwd() == expected)
    throw std::runtime_error("setter did not override environment");
  tc::set_deterministic_embed_bwd(true);
  tc::Device cpu{tc::DeviceType::CPU, 0};
  tc::NDArray ids({4}, tc::DType::Int64, cpu), grad({4, 3}, tc::DType::Float32, cpu);
  const int64_t index[] = {2, 0, 2, 2};
  const float g[] = {1.e8f, 3, 7, 2, 4, 8, 1, 5, 9, -1.e8f, 6, 10};
  const float want[] = {2, 4, 8, 0, 0, 0, 1, 14, 26, 0, 0, 0};
  std::memcpy(ids.data_ptr(), index, sizeof(index));
  std::memcpy(grad.data_ptr(), g, sizeof(g));
  for (int i = 0; i < 5; ++i) {
    auto out = tc::embedding_backward(grad, ids, {4, 3}, tc::DType::Float32);
    if (std::memcmp(out.data_ptr(), want, sizeof(want)))
      throw std::runtime_error("CPU embedding_backward reference mismatch");
  }
  tc::set_deterministic_embed_bwd(false);
  if (tc::get_deterministic_embed_bwd()) return 3;
  std::cout << "PT_DET_1 HOST_CONTRACT: environment=" << expected
            << "; setter override; 5 exact CPU dispatches; no CUDA device operations\n";
}
