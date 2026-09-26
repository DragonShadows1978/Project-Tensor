// PT-DET-1: opt-in deterministic embedding backward. Default path stays in kernels.cu.
#include "tc/core.h"
#include "tc/deterministic_embed.h"

#include <cub/device/device_radix_sort.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <climits>
#include <cstdlib>
#include <cstring>

namespace tc {
namespace {
// Prior art: PyTorch deterministic-algorithms mode (2021), taken as an opt-in
// policy. Ours: thread-local family for embedding and gather/top-k backward.
// PT-DET-2: canonical environment wins when present; legacy name is a fallback.
// Exactly "1" enables it. Either setter overrides this one shared state.
// No CUDA call in initialization/get/set; no arbitrary-op determinism promise.
thread_local bool deterministic = [] {
  const char* env = std::getenv("TC_DETERMINISTIC");
  if (!env) env = std::getenv("TC_DET_EMBED_BWD");
  return env && std::strcmp(env, "1") == 0;
}();

void check(cudaError_t status, const char* where) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(status));
}

__global__ void init_positions(int64_t* positions, int64_t n) {
  int64_t p = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (p < n) positions[p] = p;
}

// Prior art: CUB/NVIDIA stable radix sort (CUDA 12.6, 2024) guarantees equal
// keys preserve ascending original positions. Segmented reduction + single
// output ownership follows deterministic scatter-add practice (PyTorch 2021).
// See deterministic_embed.h for references and Demmel/Nguyen (2013) boundary.
// Ours: a CTA for each possible segment start and 256-feature tile; non-start
// CTAs exit immediately. No float atomics, no host transfer, no cached IDs.
template <class T>
__global__ void reduce_segments(const T* grad, const int64_t* keys,
                                const int64_t* positions, float* out,
                                int64_t n, int64_t vocab, int64_t width) {
  int64_t first = blockIdx.x, token = keys[first];
  if (first && keys[first - 1] == token) return;
  // Forward's valid-token contract still applies. Guard output addressing
  // here as well; invalid IDs are never used to address the output gradient.
  if (token < 0 || token >= vocab) return;
  int64_t feature = int64_t(blockIdx.y) * blockDim.x + threadIdx.x;
  if (feature >= width) return;
  // Ordinary upper_bound (binary search), identical for every active thread.
  int64_t lo = first + 1, hi = n;
  while (lo < hi) {
    int64_t mid = lo + (hi - lo) / 2;
    if (keys[mid] == token) lo = mid + 1;
    else hi = mid;
  }
  out[token * width + feature] = det_embed::segment_sum(grad, positions, first, lo, width, feature);
}

template <class T>
void launch(const NDArray& grad, const NDArray& keys, const NDArray& positions,
            NDArray& out, int64_t n, int64_t vocab, int64_t width) {
  reduce_segments<T><<<dim3(static_cast<unsigned>(n), static_cast<unsigned>((width + 255) / 256)), 256>>>(
      static_cast<const T*>(grad.data_ptr()), static_cast<const int64_t*>(keys.data_ptr()),
      static_cast<const int64_t*>(positions.data_ptr()), static_cast<float*>(out.data_ptr()),
      n, vocab, width);
  cuda_check_last("det embedding segmented reduction");
}
}  // namespace

void set_deterministic(bool enabled) { deterministic = enabled; }
bool get_deterministic() { return deterministic; }
void set_deterministic_embed_bwd(bool enabled) { set_deterministic(enabled); }
bool get_deterministic_embed_bwd() { return get_deterministic(); }

NDArray embedding_backward_deterministic(const NDArray& grad, const NDArray& idx,
                                         const Shape& weight_shape, DType weight_dtype) {
  if (weight_shape.empty() || weight_shape[0] <= 0 || idx.dtype != DType::Int64 ||
      grad.device.type != idx.device.type || grad.device.index != idx.device.index)
    throw std::runtime_error("det embedding: invalid weight shape, indices or device");
  for (auto size : weight_shape)
    if (size <= 0) throw std::runtime_error("det embedding: invalid weight shape");
  int64_t vocab = weight_shape[0], width = numel_of(weight_shape) / vocab, n = idx.numel();
  if (n > INT_MAX || (width + 255) / 256 > 65535 || grad.numel() != n * width)
    throw std::runtime_error("det embedding: unsupported dimensions");
  if (grad.dtype != DType::Float32 && grad.dtype != DType::Float16 && grad.dtype != DType::BFloat16)
    throw std::runtime_error("det embedding: grad must be floating point");
  if (weight_dtype != DType::Float32 && weight_dtype != DType::Float16 && weight_dtype != DType::BFloat16)
    throw std::runtime_error("det embedding: weight must be floating point");
  if (!grad.device.is_cuda()) {
    if (grad.dtype != DType::Float32 || weight_dtype != DType::Float32)
      throw std::runtime_error("det embedding: CPU path supports float32 only");
    NDArray out(weight_shape, DType::Float32, grad.device);
    det_embed::backward_cpu(static_cast<const float*>(grad.data_ptr()),
                          static_cast<const int64_t*>(idx.data_ptr()),
                          static_cast<float*>(out.data_ptr()), n, vocab, width);
    return out;
  }
  NDArray out = NDArray::zeros(weight_shape, DType::Float32, grad.device);
  if (!n) return out.astype(weight_dtype);
  NDArray input_positions({n}, DType::Int64, grad.device);
  NDArray keys({n}, DType::Int64, grad.device), positions({n}, DType::Int64, grad.device);
  init_positions<<<static_cast<unsigned>((n + 255) / 256), 256>>>(
      static_cast<int64_t*>(input_positions.data_ptr()), n);
  cuda_check_last("det embedding init positions");
  auto ids = static_cast<const int64_t*>(idx.data_ptr());
  auto in_pos = static_cast<const int64_t*>(input_positions.data_ptr());
  auto key_out = static_cast<int64_t*>(keys.data_ptr());
  auto pos_out = static_cast<int64_t*>(positions.data_ptr());
  size_t bytes = 0;
  check(cub::DeviceRadixSort::SortPairs(nullptr, bytes, ids, key_out, in_pos, pos_out, int(n)),
        "det embedding sort workspace");
  NDArray scratch({static_cast<int64_t>(bytes)}, DType::Uint8, grad.device);
  check(cub::DeviceRadixSort::SortPairs(scratch.data_ptr(), bytes, ids, key_out, in_pos, pos_out, int(n)),
        "det embedding stable sort");
  if (grad.dtype == DType::Float32) launch<float>(grad, keys, positions, out, n, vocab, width);
  else if (grad.dtype == DType::Float16) launch<__half>(grad, keys, positions, out, n, vocab, width);
  else launch<__nv_bfloat16>(grad, keys, positions, out, n, vocab, width);
  // Temporaries free on the engine's legacy default stream after their users.
  return out.astype(weight_dtype);
}
}  // namespace tc
