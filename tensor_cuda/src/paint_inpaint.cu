// paint_inpaint.cu: ordered CSR segmented executor for mesh-island smoothing.
//
// PAINT-CUDA-3 deliberately maps one island to one warp and one serial lane.
// Disconnected islands run concurrently, but lane zero preserves the oracle's
// original occurrence order, stable neighbor order, and in-place float32
// promotion semantics inside each island.  Convergence, deadlines, fallback,
// and receipt construction remain host policy.

#include "tc/core.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>

namespace tc {
namespace {

bool same_device(const Device& lhs, const Device& rhs) {
  return lhs.type == rhs.type && lhs.index == rhs.index;
}

void require_same_cuda_device(const NDArray& reference,
                              const NDArray& value,
                              const std::string& operation,
                              const char* argument) {
  if (!value.device.is_cuda()) {
    throw std::runtime_error(operation + ": " + argument +
                             " must be a CUDA tensor");
  }
  if (!same_device(reference.device, value.device)) {
    throw std::runtime_error(operation + ": device mismatch at " + argument);
  }
}

void validate_launch_threads(int threads, const char* argument) {
  if (threads < 32 || threads > 1024 || threads % 32 != 0) {
    throw std::runtime_error(
        std::string(argument) + " must be a warp multiple in [32, 1024]");
  }
}

__global__ void inpaint_island_passes_kernel(
    const float* __restrict__ positions,
    float* __restrict__ vertex_colors,
    float* __restrict__ vertex_mask,
    const int64_t* __restrict__ neighbor_offsets,
    const int64_t* __restrict__ neighbors,
    const int64_t* __restrict__ island_offsets,
    const int64_t* __restrict__ island_occurrences,
    int64_t island_count, int pass_count_cap,
    int64_t* __restrict__ island_uncolored) {
  const int warps_per_block = blockDim.x / 32;
  const int warp_in_block = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int64_t island =
      static_cast<int64_t>(blockIdx.x) * warps_per_block + warp_in_block;
  if (lane != 0 || island >= island_count) return;

  const int64_t occurrence_begin = island_offsets[island];
  const int64_t occurrence_end = island_offsets[island + 1];
  int64_t final_uncolored = 0;
  for (int pass = 0; pass < pass_count_cap; ++pass) {
    int64_t uncolored = 0;
    for (int64_t occurrence = occurrence_begin; occurrence < occurrence_end;
         ++occurrence) {
      const int64_t vertex = island_occurrences[occurrence];
      const float* origin = positions + vertex * 3;
      float sum_r = 0.0f;
      float sum_g = 0.0f;
      float sum_b = 0.0f;
      float total_weight = 0.0f;
      const int64_t neighbor_begin = neighbor_offsets[vertex];
      const int64_t neighbor_end = neighbor_offsets[vertex + 1];
      for (int64_t edge = neighbor_begin; edge < neighbor_end; ++edge) {
        const int64_t connected = neighbors[edge];
        if (vertex_mask[connected] > 0.0f) {
          const float* connected_position = positions + connected * 3;
          const float delta_x =
              __fsub_rn(origin[0], connected_position[0]);
          const float delta_y =
              __fsub_rn(origin[1], connected_position[1]);
          const float delta_z =
              __fsub_rn(origin[2], connected_position[2]);
          float squared_distance = __fmul_rn(delta_x, delta_x);
          squared_distance = __fadd_rn(
              squared_distance, __fmul_rn(delta_y, delta_y));
          squared_distance = __fadd_rn(
              squared_distance, __fmul_rn(delta_z, delta_z));
          float distance = __fsqrt_rn(squared_distance);
          if (distance < 1.0e-4f) distance = 1.0e-4f;
          const float inverse_distance = __fdiv_rn(1.0f, distance);
          const float weight =
              __fmul_rn(inverse_distance, inverse_distance);
          const float* connected_color = vertex_colors + connected * 3;
          sum_r = __fadd_rn(
              sum_r, __fmul_rn(connected_color[0], weight));
          sum_g = __fadd_rn(
              sum_g, __fmul_rn(connected_color[1], weight));
          sum_b = __fadd_rn(
              sum_b, __fmul_rn(connected_color[2], weight));
          total_weight = __fadd_rn(total_weight, weight);
        }
      }
      if (total_weight > 0.0f) {
        float* color = vertex_colors + vertex * 3;
        color[0] = __fdiv_rn(sum_r, total_weight);
        color[1] = __fdiv_rn(sum_g, total_weight);
        color[2] = __fdiv_rn(sum_b, total_weight);
        vertex_mask[vertex] = 1.0f;
      } else {
        ++uncolored;
      }
    }
    final_uncolored = uncolored;
  }
  island_uncolored[island] = final_uncolored;
}

}  // namespace

std::tuple<NDArray, NDArray, NDArray> inpaint_island_passes(
    const NDArray& positions, const NDArray& vertex_colors,
    const NDArray& vertex_mask, const NDArray& neighbor_offsets,
    const NDArray& neighbors, const NDArray& island_offsets,
    const NDArray& island_occurrences, int pass_count_cap, int threads) {
  const std::string operation = "inpaint_island_passes";
  if (positions.dtype != DType::Float32 || positions.ndim() != 2 ||
      positions.shape[1] != 3) {
    throw std::runtime_error(
        operation + ": positions must be float32 with shape (V, 3)");
  }
  if (!positions.device.is_cuda()) {
    throw std::runtime_error(operation + ": CUDA tensors required");
  }
  const int64_t vertex_count = positions.shape[0];
  if (vertex_colors.dtype != DType::Float32 ||
      vertex_colors.shape != Shape{vertex_count, 3}) {
    throw std::runtime_error(
        operation + ": vertex_colors must be float32 with shape (V, 3)");
  }
  if (vertex_mask.dtype != DType::Float32 ||
      vertex_mask.shape != Shape{vertex_count}) {
    throw std::runtime_error(
        operation + ": vertex_mask must be float32 with shape (V,)");
  }
  if (neighbor_offsets.dtype != DType::Int64 ||
      neighbor_offsets.shape != Shape{vertex_count + 1}) {
    throw std::runtime_error(
        operation +
        ": neighbor_offsets must be int64 with shape (V + 1,)");
  }
  if (neighbors.dtype != DType::Int64 || neighbors.ndim() != 1) {
    throw std::runtime_error(
        operation + ": neighbors must be int64 with shape (E,)");
  }
  if (island_offsets.dtype != DType::Int64 ||
      island_offsets.ndim() != 1 || island_offsets.shape[0] < 1) {
    throw std::runtime_error(
        operation + ": island_offsets must be int64 with shape (I + 1,)");
  }
  if (island_occurrences.dtype != DType::Int64 ||
      island_occurrences.ndim() != 1) {
    throw std::runtime_error(
        operation + ": island_occurrences must be int64 with shape (O,)");
  }
  require_same_cuda_device(positions, vertex_colors, operation,
                           "vertex_colors");
  require_same_cuda_device(positions, vertex_mask, operation, "vertex_mask");
  require_same_cuda_device(positions, neighbor_offsets, operation,
                           "neighbor_offsets");
  require_same_cuda_device(positions, neighbors, operation, "neighbors");
  require_same_cuda_device(positions, island_offsets, operation,
                           "island_offsets");
  require_same_cuda_device(positions, island_occurrences, operation,
                           "island_occurrences");
  if (pass_count_cap <= 0) {
    throw std::runtime_error(
        operation + ": pass_count_cap must be positive");
  }
  validate_launch_threads(threads, "inpaint_island_passes: threads");

  const int64_t island_count = island_offsets.shape[0] - 1;
  NDArray colors = vertex_colors.clone();
  NDArray mask = vertex_mask.clone();
  NDArray island_uncolored(
      {island_count}, DType::Int64, positions.device);
  if (island_count > 0) {
    const int warps_per_block = threads / 32;
    const int64_t blocks64 =
        (island_count + warps_per_block - 1) / warps_per_block;
    if (blocks64 > std::numeric_limits<unsigned>::max()) {
      throw std::runtime_error(operation + ": launch grid exceeds CUDA range");
    }
    inpaint_island_passes_kernel<<<static_cast<unsigned>(blocks64), threads>>>(
        static_cast<const float*>(positions.data_ptr()),
        static_cast<float*>(colors.data_ptr()),
        static_cast<float*>(mask.data_ptr()),
        static_cast<const int64_t*>(neighbor_offsets.data_ptr()),
        static_cast<const int64_t*>(neighbors.data_ptr()),
        static_cast<const int64_t*>(island_offsets.data_ptr()),
        static_cast<const int64_t*>(island_occurrences.data_ptr()),
        island_count, pass_count_cap,
        static_cast<int64_t*>(island_uncolored.data_ptr()));
    cuda_check_last("inpaint_island_passes");
  }
  return std::make_tuple(colors, mask, island_uncolored);
}

}  // namespace tc
