// Prior art: CUB/PyTorch deterministic segmented scatter, cited in the shared
// header. Test-only C ABI exposes the actual CPU implementation to NumPy's
// independent FP64 scatter reference; no CUDA library or device operations.
#include "tc/deterministic_embed.h"
extern "C" int pt_det_cpu(const float* grad, const int64_t* ids, float* out,
                          int64_t n, int64_t vocab, int64_t width) {
  try { tc::det_embed::backward_cpu(grad, ids, out, n, vocab, width); }
  catch (...) { return 1; }
  return 0;
}
