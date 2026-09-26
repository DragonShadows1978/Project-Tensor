// PT-DET-2: general gather/top-k VJP; the OFF path remains in kernels.cu.
#include "tc/deterministic_gather.h"
#include <cub/device/device_radix_sort.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tc {
namespace {
void check(cudaError_t status, const char* where) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(status));
}

// Prior art and taken/ours boundary: deterministic_gather.h. Flat destination
// keys include ALL non-gather coordinates: equal class IDs across distinct
// loss rows must not be combined. Stable sorting preserves original positions.
__global__ void make_keys(const int64_t* index, int64_t* keys, int64_t* positions,
                          det_gather::Spec s) {
  const int64_t p = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (p >= s.count) return;
  positions[p] = p;
  keys[p] = det_gather::destination(p, index[p], s);
  // The existing gather requires valid indices. Fail asynchronously on invalid
  // device indices instead of making an out-of-bounds access or reading IDs back.
  if (keys[p] < 0) asm("trap;");
}

// Taken: sorted segmented reduction / single ownership (CUB/PyTorch, above).
// Ours: one thread at each segment start, ascending-position FP64 sum shared
// with PT-DET-1, then one FP32 store. No floating atomic and no host ID readback.
template <class T>
__global__ void reduce_segments(const T* grad, const int64_t* keys,
                                const int64_t* positions, float* out, int64_t n) {
  const int64_t first = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (first >= n) return;
  const int64_t key = keys[first];
  if (key < 0 || (first && keys[first - 1] == key)) return;
  int64_t last = first + 1;
  while (last < n && keys[last] == key) ++last;
  out[key] = det_embed::segment_sum(grad, positions, first, last, 1, 0);
}

template <class T>
void launch(const NDArray& src, const NDArray& keys, const NDArray& positions,
            NDArray& out, int64_t n) {
  reduce_segments<T><<<static_cast<unsigned>((n + 255) / 256), 256>>>(
      static_cast<const T*>(src.data_ptr()), static_cast<const int64_t*>(keys.data_ptr()),
      static_cast<const int64_t*>(positions.data_ptr()), static_cast<float*>(out.data_ptr()), n);
  cuda_check_last("det gather segmented reduction");
}

bool floating(DType dt) {
  return dt == DType::Float32 || dt == DType::Float16 || dt == DType::BFloat16;
}
}  // namespace

NDArray scatter_add_deterministic(const Shape& shape, DType dtype, int dim,
                                   const NDArray& index, const NDArray& src) {
  const auto s = det_gather::make_spec(shape, index.shape, dim);
  if (src.shape != index.shape || index.dtype != DType::Int64 ||
      index.device.type != src.device.type || index.device.index != src.device.index ||
      !floating(src.dtype) || !floating(dtype))
    throw std::runtime_error("det gather: invalid source, index, dtype or device");
  if (!src.device.is_cuda()) {
    if (src.dtype != DType::Float32 || dtype != DType::Float32)
      throw std::runtime_error("det gather: CPU path supports float32 only");
    NDArray out(shape, dtype, src.device);
    det_gather::backward_cpu(static_cast<const float*>(src.data_ptr()),
                            static_cast<const int64_t*>(index.data_ptr()),
                            static_cast<float*>(out.data_ptr()), s);
    return out;
  }
  NDArray out = NDArray::zeros(shape, DType::Float32, src.device);
  if (!s.count) return out.astype(dtype);
  NDArray input_keys({s.count}, DType::Int64, src.device);
  NDArray input_positions({s.count}, DType::Int64, src.device);
  NDArray keys({s.count}, DType::Int64, src.device), positions({s.count}, DType::Int64, src.device);
  make_keys<<<static_cast<unsigned>((s.count + 255) / 256), 256>>>(
      static_cast<const int64_t*>(index.data_ptr()), static_cast<int64_t*>(input_keys.data_ptr()),
      static_cast<int64_t*>(input_positions.data_ptr()), s);
  cuda_check_last("det gather destination keys");
  auto in_keys = static_cast<const int64_t*>(input_keys.data_ptr());
  auto in_positions = static_cast<const int64_t*>(input_positions.data_ptr());
  auto out_keys = static_cast<int64_t*>(keys.data_ptr());
  auto out_positions = static_cast<int64_t*>(positions.data_ptr());
  size_t bytes = 0;
  check(cub::DeviceRadixSort::SortPairs(nullptr, bytes, in_keys, out_keys,
                                       in_positions, out_positions, int(s.count)),
        "det gather sort workspace");
  NDArray scratch({static_cast<int64_t>(bytes)}, DType::Uint8, src.device);
  check(cub::DeviceRadixSort::SortPairs(scratch.data_ptr(), bytes, in_keys, out_keys,
                                       in_positions, out_positions, int(s.count)),
        "det gather stable sort");
  if (src.dtype == DType::Float32) launch<float>(src, keys, positions, out, s.count);
  else if (src.dtype == DType::Float16) launch<__half>(src, keys, positions, out, s.count);
  else launch<__nv_bfloat16>(src, keys, positions, out, s.count);
  // Stream-ordered storage lifetime matches PT-DET-1 and the engine allocator.
  return out.astype(dtype);
}
}  // namespace tc
