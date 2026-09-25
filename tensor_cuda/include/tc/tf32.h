#pragma once
#include "tc/core.h"

namespace tc {
// PT-TF32-1. Prior art: NVIDIA Ampere TF32 / cuBLAS (2020), taken APIs;
// BP-KERNEL-2/4 (Project-Tensor 2026), taken explicit opt-in dispatch.
// Ours: thread-local FP32 mode, initialized once/thread by TC_TF32_GEMM=1.
void set_tf32_gemm(bool enabled);
bool get_tf32_gemm();
struct TF32GemmGuard {
  bool previous;
  explicit TF32GemmGuard(bool enabled) : previous(get_tf32_gemm()) {
    set_tf32_gemm(enabled);
  }
  ~TF32GemmGuard() { set_tf32_gemm(previous); }
  TF32GemmGuard(const TF32GemmGuard&) = delete;
  TF32GemmGuard& operator=(const TF32GemmGuard&) = delete;
};

std::tuple<NDArray,NDArray,NDArray,NDArray,NDArray> apa_selective_fwd_tf32(
    const NDArray&, const NDArray&, const NDArray&, const NDArray&,
    float, float, bool, const std::string&, bool diagnostic = false);
std::tuple<NDArray,NDArray,NDArray> apa_selective_bwd_tf32(
    const NDArray&, const NDArray&, const NDArray&, const NDArray&,
    const NDArray&, const NDArray&, const NDArray&, const NDArray&, float, bool);
} // namespace tc
