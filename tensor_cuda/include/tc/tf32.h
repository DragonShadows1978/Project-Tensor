#pragma once
#include "tc/core.h"
#include <map>

namespace tc {
// PT-TF32-1. Prior art: NVIDIA Ampere TF32 / cuBLAS (2020), taken APIs;
// BP-KERNEL-2/4 (Project-Tensor 2026), taken explicit opt-in dispatch.
// Ours: thread-local FP32 mode, initialized once/thread by TC_TF32_GEMM=1.
void set_tf32_gemm(bool enabled);
bool get_tf32_gemm();
// PT-TF32-2: NVIDIA cuBLASLt (2024) algorithms/capability flags, taken.
// Ours: most recent thread-local dispatch receipt: algo ID, flags, workspace.
std::tuple<int,uint64_t,size_t> get_tf32_gemm_info();
// PT-TF32-3. NVIDIA cuBLASLt (2024), taken host descriptor/capability API.
// Ours: CPU descriptor check; device heuristic selection is lead-slot-only.
std::map<std::string,int64_t> tf32_gemm_self_check(int64_t,int64_t,int64_t,bool,bool);
std::map<std::string,int64_t> get_tf32_gemm_dispatch();
NDArray tf32_gemm_output(const Shape&,Device);
void matmul_tf32(const NDArray&,const NDArray&,NDArray&,float,bool);
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
