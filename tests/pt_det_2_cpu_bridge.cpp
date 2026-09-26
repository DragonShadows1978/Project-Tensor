// Author baseline only. Prior art: PT-DET-1 (2026) shared CPU/CUDA arithmetic
// bridge, reused for independent NumPy scatter-reference testing.
#include "tc/deterministic_gather.h"
extern "C" int pt_det_gather_cpu(const float* grad, const int64_t* index, float* out,
                                 const int64_t* shape, const int64_t* index_shape,
                                 int ndim, int dim) {
  try {
    const auto spec = tc::det_gather::make_spec(tc::Shape(shape, shape + ndim),
                                               tc::Shape(index_shape, index_shape + ndim), dim);
    tc::det_gather::backward_cpu(grad, index, out, spec);
    return 0;
  } catch (const std::exception&) { return 1; }
}
