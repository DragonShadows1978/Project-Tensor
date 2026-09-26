// Prior art: PT-DET-1 (2026) raw CPU NDArray contract executable, reused.
// Ours: family/legacy alias equivalence, TLS isolation and actual gather dispatch.
#include "tc/core.h"
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>

void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  const bool expected = std::string(argv[1]) == "1";
  require(tc::get_deterministic() == expected, "family environment mismatch");
  require(tc::get_deterministic_embed_bwd() == expected, "alias environment mismatch");
  tc::set_deterministic(!expected);
  bool thread_ok = false;
  std::thread child([&] {
    thread_ok = tc::get_deterministic() == expected;
    tc::set_deterministic_embed_bwd(expected);
  });
  child.join();
  require(thread_ok && tc::get_deterministic() == !expected, "TLS isolation failed");
  tc::Device cpu{tc::DeviceType::CPU, 0};
  tc::NDArray ids({1, 2, 4}, tc::DType::Int64, cpu), grad({1, 2, 4}, tc::DType::Float32, cpu);
  const int64_t indices[] = {2, 2, 2, 2, 2, 2, 2, 2};
  const float values[] = {1.e8f, 1, -1.e8f, 2, 4, 5, 6, 7};
  const float want[] = {0, 0, 3, 0, 0, 0, 0, 0, 0, 22, 0, 0, 0, 0};
  std::memcpy(ids.data_ptr(), indices, sizeof(indices));
  std::memcpy(grad.data_ptr(), values, sizeof(values));
  for (int i = 0; i < 5; ++i) {
    // A legacy setter must enable the gather path too, not just embedding.
    if (i % 2) tc::set_deterministic_embed_bwd(true);
    else tc::set_deterministic(true);
    require(tc::get_deterministic() && tc::get_deterministic_embed_bwd(), "ON aliases disagree");
    auto out = tc::scatter_add_nd({1, 2, 7}, tc::DType::Float32, -1, ids, grad);
    require(std::memcmp(out.data_ptr(), want, sizeof(want)) == 0, "CPU gather dispatch differs");
    // One family also routes embedding, using its existing CPU-safe raw path.
    auto emb = tc::embedding_backward(grad.reshape({8, 1}), ids.reshape({8}), {7, 1}, tc::DType::Float32);
    const float emb_want[] = {0, 0, 25, 0, 0, 0, 0};
    require(std::memcmp(emb.data_ptr(), emb_want, sizeof(emb_want)) == 0, "CPU embedding alias differs");
  }
  bool rejected = false;
  try { tc::scatter_add_nd({1, 2, 7}, tc::DType::Float32, -1, ids, grad.reshape({8})); }
  catch (const std::runtime_error&) { rejected = true; }
  require(rejected, "mismatched source shape accepted");
  tc::set_deterministic_embed_bwd(false);
  require(!tc::get_deterministic(), "legacy OFF did not disable family");
  tc::set_deterministic(true);
  tc::set_deterministic(false);
  require(!tc::get_deterministic_embed_bwd(), "family OFF did not disable legacy");
  std::cout << "PT_DET_2 HOST_CONTRACT: environment=" << expected
            << "; shared aliases; TLS isolation; 5 exact CPU gather+embedding dispatches; no CUDA calls\n";
}
