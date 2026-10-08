// APA-MF2 stub harness ONLY (not shipped, not apa_cuda's kernels).
//
// apa_attention_cuda cannot be built on this machine: torch 2.11 is cu130 and only
// CUDA 12.x nvcc is installed, so torch.utils.cpp_extension refuses the CUDA build
// (major-version mismatch, logs/apa_mf2_ext_build_*.log). To EXECUTE the shipped
// csrc/apa.cpp (whose causal mask is item 3), this file supplies the two symbols
// apa.cpp declares from apa_kernels.cu, as an ATen transcription of those kernels:
//   quantize_gather: out = codebook[searchsorted_left(boundaries, (float)x)]
//                    (apa_kernels.cu searchsorted_left: first i with b[i] >= x)
//   mix_scores:      refine = fabsf((float)ranking) >= thr[row];
//                    out = refine ? ranking : bulk; mask = (uint8)refine
// Compiled with apa.cpp as a C++-only torch extension (no nvcc); every op runs on
// CUDA tensors through ATen. Prior art: apa_kernels.cu (Project-Tensor); numpy
// searchsorted side='left' semantics. Transcription: mine.
#include <torch/extension.h>

#include <tuple>

namespace apa {

at::Tensor quantize_gather(const at::Tensor& rotated, const at::Tensor& boundaries,
                           const at::Tensor& codebook) {
  TORCH_CHECK(codebook.numel() == boundaries.numel() + 1,
              "codebook length must be n_boundaries + 1");
  auto r = rotated.contiguous();
  auto idx = at::searchsorted(boundaries.to(at::kFloat).contiguous(), r.to(at::kFloat),
                              /*out_int32=*/false, /*right=*/false);
  auto out = codebook.contiguous().index_select(0, idx.reshape({-1})).view(r.sizes());
  return out.to(r.scalar_type());
}

std::tuple<at::Tensor, at::Tensor> mix_scores(const at::Tensor& ranking, const at::Tensor& bulk,
                                              const at::Tensor& thr) {
  TORCH_CHECK(ranking.sizes() == bulk.sizes(), "ranking/bulk shape mismatch");
  auto r = ranking.contiguous();
  auto b = bulk.contiguous();
  const int64_t S = r.size(-1);
  auto t = thr.contiguous().to(at::kFloat).view({-1, 1});
  TORCH_CHECK(t.size(0) * S == r.numel(), "thr must have numel == rows");
  auto refine = r.to(at::kFloat).view({-1, S}).abs().ge(t).view(r.sizes());
  return {at::where(refine, r, b), refine.to(at::kByte)};
}

}  // namespace apa
