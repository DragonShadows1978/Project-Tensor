// paint_raster.cu: deterministic packed-key scatter-min and clip rasterizer.
//
// PAINT-CUDA-1 keeps the uint64 winner key internal because the public engine
// has no uint64 Tensor dtype.  The key is lexicographic
//   (uint32(depth) ^ 0x80000000, uint32(one_based_face_id))
// so unsigned atomicMin implements signed-int32 depth ordering followed by the
// lower-face-ID tie break.  UINT64_MAX is the background sentinel.

#include "tc/core.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>

namespace tc {
namespace {

constexpr uint64_t kEmptyWinner = std::numeric_limits<uint64_t>::max();
constexpr int32_t kInt32Min = (-2147483647 - 1);
constexpr int32_t kInt32Max = 2147483647;
constexpr int64_t kMaxFaceId = std::numeric_limits<int32_t>::max();

struct ScreenVertex {
  float x;
  float y;
  float z;
};

struct Barycentric {
  float alpha;
  float beta;
  float gamma;
};

bool same_device(const Device& lhs, const Device& rhs) {
  return lhs.type == rhs.type && lhs.index == rhs.index;
}

void validate_launch_threads(int threads, const char* argument) {
  if (threads < 32 || threads > 1024 || threads % 32 != 0) {
    throw std::runtime_error(
        std::string(argument) + " must be a warp multiple in [32, 1024]");
  }
}

void validate_cuda_memset(cudaError_t status, const char* where) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string("CUDA error at ") + where + ": " +
        cudaGetErrorString(status));
  }
}

void validate_raster_inputs(const NDArray& clip_positions,
                            const NDArray& faces, int height, int width,
                            const char* operation) {
  const std::string prefix = std::string(operation) + ": ";
  if (clip_positions.ndim() != 2 || clip_positions.shape[1] != 4 ||
      clip_positions.dtype != DType::Float32) {
    throw std::runtime_error(
        prefix + "clip_positions must be float32 with shape (V, 4)");
  }
  if (faces.ndim() != 2 || faces.shape[1] != 3 ||
      faces.dtype != DType::Int64) {
    throw std::runtime_error(
        prefix + "faces must be int64 with shape (F, 3)");
  }
  if (!clip_positions.device.is_cuda() || !faces.device.is_cuda()) {
    throw std::runtime_error(prefix + "CUDA tensors required");
  }
  if (!same_device(clip_positions.device, faces.device)) {
    throw std::runtime_error(prefix + "device mismatch");
  }
  if (height <= 0 || width <= 0) {
    throw std::runtime_error(prefix + "height and width must be positive");
  }
  if (faces.shape[0] > kMaxFaceId) {
    throw std::runtime_error(prefix + "face count exceeds INT32_MAX");
  }
  if (static_cast<int64_t>(height) >
      std::numeric_limits<int64_t>::max() / static_cast<int64_t>(width)) {
    throw std::runtime_error(prefix + "pixel count overflows int64");
  }
}

__device__ __forceinline__ uint64_t pack_winner(int32_t depth,
                                                 uint32_t face_id) {
  const uint32_t ordered_depth = static_cast<uint32_t>(depth) ^ 0x80000000u;
  return (static_cast<uint64_t>(ordered_depth) << 32) |
         static_cast<uint64_t>(face_id);
}

__device__ __forceinline__ ScreenVertex project_vertex(
    const float* __restrict__ clip_positions, int64_t vertex, int height,
    int width) {
  const float* clip = clip_positions + vertex * 4;
  const float reciprocal_w = __fdiv_rn(1.0f, clip[3]);
  const float ndc_x = __fmul_rn(clip[0], reciprocal_w);
  const float ndc_y = __fmul_rn(clip[1], reciprocal_w);
  const float ndc_z = __fmul_rn(clip[2], reciprocal_w);
  const float unit_x = __fmaf_rn(ndc_x, 0.5f, 0.5f);
  const float unit_y = __fmaf_rn(ndc_y, 0.5f, 0.5f);
  return ScreenVertex{
      __fmaf_rn(unit_x, static_cast<float>(width - 1), 0.5f),
      __fmaf_rn(unit_y, static_cast<float>(height - 1), 0.5f),
      __fmaf_rn(ndc_z, 0.49999f, 0.5f)};
}

__device__ __forceinline__ float signed_area2(const ScreenVertex& a,
                                               const ScreenVertex& b,
                                               const ScreenVertex& c) {
  const float first_x = __fsub_rn(c.x, a.x);
  const float first_y = __fsub_rn(b.y, a.y);
  const float second =
      __fmul_rn(__fsub_rn(b.x, a.x), __fsub_rn(c.y, a.y));
  return __fmaf_rn(first_x, first_y, -second);
}

__device__ __forceinline__ bool barycentric_at(
    const ScreenVertex& a, const ScreenVertex& b, const ScreenVertex& c,
    float px, float py, float area, Barycentric* result) {
  if (area == 0.0f) return false;
  const float inverse_area = __fdiv_rn(1.0f, area);
  const float px_from_a = __fsub_rn(px, a.x);
  const float py_from_a = __fsub_rn(py, a.y);

  const float beta_second =
      __fmul_rn(px_from_a, __fsub_rn(c.y, a.y));
  const float beta_tri =
      __fmaf_rn(__fsub_rn(c.x, a.x), py_from_a, -beta_second);
  const float beta = __fmul_rn(beta_tri, inverse_area);

  const float gamma_second =
      __fmul_rn(__fsub_rn(b.x, a.x), py_from_a);
  const float gamma_tri =
      __fmaf_rn(px_from_a, __fsub_rn(b.y, a.y), -gamma_second);
  const float gamma = __fmul_rn(gamma_tri, inverse_area);

  // The frozen source literal is 1.0 (double), not 1.0f.  Keep both binary64
  // subtractions explicit and round only the final alpha to binary32.
  const double alpha_double = __dsub_rn(
      __dsub_rn(1.0, static_cast<double>(beta)),
      static_cast<double>(gamma));
  const float alpha = __double2float_rn(alpha_double);
  result->alpha = alpha;
  result->beta = beta;
  result->gamma = gamma;
  return true;
}

__device__ __forceinline__ bool covered(const Barycentric& barycentric) {
  return barycentric.alpha >= 0.0f && barycentric.alpha <= 1.0f &&
         barycentric.beta >= 0.0f && barycentric.beta <= 1.0f &&
         barycentric.gamma >= 0.0f && barycentric.gamma <= 1.0f;
}

__device__ __forceinline__ float interpolate_depth(
    const Barycentric& barycentric, const ScreenVertex& a,
    const ScreenVertex& b, const ScreenVertex& c) {
  const float beta_b = __fmul_rn(barycentric.beta, b.z);
  const float alpha_beta =
      __fmaf_rn(barycentric.alpha, a.z, beta_b);
  return __fmaf_rn(barycentric.gamma, c.z, alpha_beta);
}

__device__ __forceinline__ int32_t quantize_depth(float depth) {
  return __float2int_rz(__fmul_rn(depth, 262144.0f));
}

__global__ void packed_winner_scatter_min_kernel(
    const int64_t* __restrict__ pixel_indices,
    const int64_t* __restrict__ depth_keys,
    const int64_t* __restrict__ face_ids, int64_t candidate_count,
    int64_t pixel_count, uint64_t* __restrict__ winners) {
  const int64_t candidate =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (candidate >= candidate_count) return;
  const int64_t pixel = pixel_indices[candidate];
  const int64_t depth = depth_keys[candidate];
  const int64_t face = face_ids[candidate];
  if (pixel < 0 || pixel >= pixel_count ||
      depth < kInt32Min || depth > kInt32Max ||
      face <= 0 || face > kMaxFaceId) {
    return;
  }
  const uint64_t key = pack_winner(static_cast<int32_t>(depth),
                                   static_cast<uint32_t>(face));
  atomicMin(reinterpret_cast<unsigned long long*>(winners + pixel),
            static_cast<unsigned long long>(key));
}

__global__ void triangle_winner_kernel(
    const float* __restrict__ clip_positions,
    const int64_t* __restrict__ faces, int64_t face_count, int height,
    int width, uint64_t* __restrict__ winners) {
  const int64_t face_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (face_index >= face_count) return;

  const int64_t* triangle = faces + face_index * 3;
  const ScreenVertex a =
      project_vertex(clip_positions, triangle[0], height, width);
  const ScreenVertex b =
      project_vertex(clip_positions, triangle[1], height, width);
  const ScreenVertex c =
      project_vertex(clip_positions, triangle[2], height, width);
  const float area = signed_area2(a, b, c);
  if (area == 0.0f) return;

  // Coverage alone uses a vertex-ID canonical order.  Exact arithmetic makes
  // barycentric inclusion permutation-invariant, but the frozen fp32/double
  // sequence can otherwise open/close lattice-edge pixels when winding is
  // reversed.  Canonical coverage gives both windings the same final support.
  // Depth and resolve still use the caller's original order below, preserving
  // the oracle's fp32 FMA association on every interior pixel.
  int64_t canonical_indices[3] = {
      triangle[0], triangle[1], triangle[2]};
  if (canonical_indices[1] < canonical_indices[0]) {
    const int64_t swap = canonical_indices[0];
    canonical_indices[0] = canonical_indices[1];
    canonical_indices[1] = swap;
  }
  if (canonical_indices[2] < canonical_indices[1]) {
    const int64_t swap = canonical_indices[1];
    canonical_indices[1] = canonical_indices[2];
    canonical_indices[2] = swap;
  }
  if (canonical_indices[1] < canonical_indices[0]) {
    const int64_t swap = canonical_indices[0];
    canonical_indices[0] = canonical_indices[1];
    canonical_indices[1] = swap;
  }
  const bool coverage_is_original =
      canonical_indices[0] == triangle[0] &&
      canonical_indices[1] == triangle[1] &&
      canonical_indices[2] == triangle[2];
  ScreenVertex coverage_a = a;
  ScreenVertex coverage_b = b;
  ScreenVertex coverage_c = c;
  float coverage_area = area;
  if (!coverage_is_original) {
    coverage_a = project_vertex(
        clip_positions, canonical_indices[0], height, width);
    coverage_b = project_vertex(
        clip_positions, canonical_indices[1], height, width);
    coverage_c = project_vertex(
        clip_positions, canonical_indices[2], height, width);
    coverage_area = signed_area2(coverage_a, coverage_b, coverage_c);
    if (coverage_area == 0.0f) return;
  }

  float x_min = a.x;
  if (b.x < x_min) x_min = b.x;
  if (c.x < x_min) x_min = c.x;
  float x_max = a.x;
  if (b.x > x_max) x_max = b.x;
  if (c.x > x_max) x_max = c.x;
  float y_min = a.y;
  if (b.y < y_min) y_min = b.y;
  if (c.y < y_min) y_min = c.y;
  float y_max = a.y;
  if (b.y > y_max) y_max = b.y;
  if (c.y > y_max) y_max = c.y;

  int x0 = __float2int_rz(x_min);
  int y0 = __float2int_rz(y_min);
  if (x0 < 0) x0 = 0;
  if (y0 < 0) y0 = 0;
  int x1 = static_cast<int>(ceil(static_cast<double>(x_max) + 1.0));
  int y1 = static_cast<int>(ceil(static_cast<double>(y_max) + 1.0));
  if (x1 > width) x1 = width;
  if (y1 > height) y1 = height;
  if (x1 <= x0 || y1 <= y0) return;

  const uint32_t one_based_face = static_cast<uint32_t>(face_index + 1);
  // Preserve the source producer's x-major/y-minor traversal.  Atomic winner
  // ordering makes the final result independent of face scheduling.
  for (int px_index = x0; px_index < x1; ++px_index) {
    const float px = __fadd_rn(static_cast<float>(px_index), 0.5f);
    for (int py_index = y0; py_index < y1; ++py_index) {
      const float py = __fadd_rn(static_cast<float>(py_index), 0.5f);
      Barycentric barycentric{};
      barycentric_at(a, b, c, px, py, area, &barycentric);
      if (coverage_is_original) {
        if (!covered(barycentric)) continue;
      } else {
        Barycentric coverage_barycentric{};
        barycentric_at(coverage_a, coverage_b, coverage_c, px, py,
                       coverage_area, &coverage_barycentric);
        if (!covered(coverage_barycentric)) continue;
      }
      const int32_t depth =
          quantize_depth(interpolate_depth(barycentric, a, b, c));
      const uint64_t key = pack_winner(depth, one_based_face);
      const int64_t pixel =
          static_cast<int64_t>(py_index) * width + px_index;
      atomicMin(reinterpret_cast<unsigned long long*>(winners + pixel),
                static_cast<unsigned long long>(key));
    }
  }
}

__global__ void unpack_winners_kernel(
    const uint64_t* __restrict__ winners, int64_t pixel_count,
    int64_t* __restrict__ depth_keys, int64_t* __restrict__ face_ids) {
  const int64_t pixel =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pixel >= pixel_count) return;
  const uint64_t key = winners[pixel];
  if (key == kEmptyWinner) {
    depth_keys[pixel] = kInt32Max;
    face_ids[pixel] = 0;
    return;
  }
  const uint32_t ordered_depth = static_cast<uint32_t>(key >> 32);
  const int32_t depth =
      static_cast<int32_t>(ordered_depth ^ 0x80000000u);
  depth_keys[pixel] = static_cast<int64_t>(depth);
  face_ids[pixel] = static_cast<int64_t>(static_cast<uint32_t>(key));
}

__global__ void winner_resolve_kernel(
    const float* __restrict__ clip_positions,
    const int64_t* __restrict__ faces, int64_t face_count,
    const int64_t* __restrict__ winner_faces, int height, int width,
    int64_t pixel_count, int64_t* __restrict__ resolved_depth,
    float* __restrict__ barycentric_output) {
  const int64_t pixel =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pixel >= pixel_count) return;
  const int64_t one_based_face = winner_faces[pixel];
  float* output = barycentric_output + pixel * 3;
  if (one_based_face <= 0 || one_based_face > face_count) {
    resolved_depth[pixel] = kInt32Max;
    output[0] = 0.0f;
    output[1] = 0.0f;
    output[2] = 0.0f;
    return;
  }

  const int64_t* triangle = faces + (one_based_face - 1) * 3;
  const ScreenVertex a =
      project_vertex(clip_positions, triangle[0], height, width);
  const ScreenVertex b =
      project_vertex(clip_positions, triangle[1], height, width);
  const ScreenVertex c =
      project_vertex(clip_positions, triangle[2], height, width);
  const float area = signed_area2(a, b, c);
  const int px_index = static_cast<int>(pixel % width);
  const int py_index = static_cast<int>(pixel / width);
  const float px = __fadd_rn(static_cast<float>(px_index), 0.5f);
  const float py = __fadd_rn(static_cast<float>(py_index), 0.5f);
  Barycentric barycentric{};
  if (!barycentric_at(a, b, c, px, py, area, &barycentric)) {
    resolved_depth[pixel] = kInt32Max;
    output[0] = 0.0f;
    output[1] = 0.0f;
    output[2] = 0.0f;
    return;
  }

  resolved_depth[pixel] = static_cast<int64_t>(
      quantize_depth(interpolate_depth(barycentric, a, b, c)));
  const float* clip_a = clip_positions + triangle[0] * 4;
  const float* clip_b = clip_positions + triangle[1] * 4;
  const float* clip_c = clip_positions + triangle[2] * 4;
  float interp_alpha = __fdiv_rn(barycentric.alpha, clip_a[3]);
  float interp_beta = __fdiv_rn(barycentric.beta, clip_b[3]);
  float interp_gamma = __fdiv_rn(barycentric.gamma, clip_c[3]);
  const float interp_sum =
      __fadd_rn(__fadd_rn(interp_alpha, interp_beta), interp_gamma);
  const float reciprocal = __fdiv_rn(1.0f, interp_sum);
  interp_alpha = __fmul_rn(interp_alpha, reciprocal);
  interp_beta = __fmul_rn(interp_beta, reciprocal);
  interp_gamma = __fmul_rn(interp_gamma, reciprocal);
  output[0] = interp_alpha;
  output[1] = interp_beta;
  output[2] = interp_gamma;
}

std::tuple<NDArray, NDArray> unpack_winners(const NDArray& packed,
                                            const Shape& output_shape,
                                            int threads) {
  const int64_t pixel_count = packed.numel();
  NDArray depth(output_shape, DType::Int64, packed.device);
  NDArray face(output_shape, DType::Int64, packed.device);
  if (pixel_count > 0) {
    const int64_t blocks64 = (pixel_count + threads - 1) / threads;
    unpack_winners_kernel<<<static_cast<unsigned>(blocks64), threads>>>(
        static_cast<const uint64_t*>(packed.data_ptr()), pixel_count,
        static_cast<int64_t*>(depth.data_ptr()),
        static_cast<int64_t*>(face.data_ptr()));
    cuda_check_last("paint_unpack_winners");
  }
  return std::make_tuple(depth, face);
}

}  // namespace

std::tuple<NDArray, NDArray> raster_winner_scatter_min(
    const NDArray& pixel_indices, const NDArray& depth_keys,
    const NDArray& face_ids, int64_t pixel_count, int threads) {
  if (pixel_indices.ndim() != 1 || depth_keys.shape != pixel_indices.shape ||
      face_ids.shape != pixel_indices.shape ||
      pixel_indices.dtype != DType::Int64 ||
      depth_keys.dtype != DType::Int64 || face_ids.dtype != DType::Int64) {
    throw std::runtime_error(
        "raster_winner_scatter_min: candidate inputs must be matching int64 "
        "vectors");
  }
  if (!pixel_indices.device.is_cuda() || !depth_keys.device.is_cuda() ||
      !face_ids.device.is_cuda()) {
    throw std::runtime_error(
        "raster_winner_scatter_min: CUDA tensors required");
  }
  if (!same_device(pixel_indices.device, depth_keys.device) ||
      !same_device(pixel_indices.device, face_ids.device)) {
    throw std::runtime_error(
        "raster_winner_scatter_min: device mismatch");
  }
  if (pixel_count < 0) {
    throw std::runtime_error(
        "raster_winner_scatter_min: pixel_count must be non-negative");
  }
  validate_launch_threads(threads, "raster_winner_scatter_min: threads");

  NDArray packed({pixel_count}, DType::Int64, pixel_indices.device);
  if (pixel_count > 0) {
    validate_cuda_memset(
        cudaMemset(packed.data_ptr(), 0xff,
                   static_cast<size_t>(pixel_count) * sizeof(uint64_t)),
        "raster_winner_scatter_min(init)");
  }
  const int64_t candidate_count = pixel_indices.numel();
  if (candidate_count > 0) {
    const int64_t blocks64 = (candidate_count + threads - 1) / threads;
    packed_winner_scatter_min_kernel<<<static_cast<unsigned>(blocks64),
                                       threads>>>(
        static_cast<const int64_t*>(pixel_indices.data_ptr()),
        static_cast<const int64_t*>(depth_keys.data_ptr()),
        static_cast<const int64_t*>(face_ids.data_ptr()), candidate_count,
        pixel_count, static_cast<uint64_t*>(packed.data_ptr()));
    cuda_check_last("raster_winner_scatter_min");
  }
  return unpack_winners(packed, {pixel_count}, threads);
}

std::tuple<NDArray, NDArray> raster_triangle_winners(
    const NDArray& clip_positions, const NDArray& faces, int height, int width,
    int face_threads) {
  validate_raster_inputs(clip_positions, faces, height, width,
                         "raster_triangle_winners");
  validate_launch_threads(face_threads,
                          "raster_triangle_winners: face_threads");
  const int64_t pixel_count = static_cast<int64_t>(height) * width;
  NDArray packed({pixel_count}, DType::Int64, clip_positions.device);
  validate_cuda_memset(
      cudaMemset(packed.data_ptr(), 0xff,
                 static_cast<size_t>(pixel_count) * sizeof(uint64_t)),
      "raster_triangle_winners(init)");
  const int64_t face_count = faces.shape[0];
  if (face_count > 0) {
    const int64_t blocks64 = (face_count + face_threads - 1) / face_threads;
    triangle_winner_kernel<<<static_cast<unsigned>(blocks64), face_threads>>>(
        static_cast<const float*>(clip_positions.data_ptr()),
        static_cast<const int64_t*>(faces.data_ptr()), face_count, height,
        width, static_cast<uint64_t*>(packed.data_ptr()));
    cuda_check_last("raster_triangle_winners");
  }
  return unpack_winners(packed, {height, width}, 256);
}

std::tuple<NDArray, NDArray> raster_winner_resolve(
    const NDArray& clip_positions, const NDArray& faces,
    const NDArray& winner_face_ids, int pixel_threads) {
  if (winner_face_ids.ndim() != 2 || winner_face_ids.shape[0] <= 0 ||
      winner_face_ids.shape[1] <= 0 ||
      winner_face_ids.dtype != DType::Int64) {
    throw std::runtime_error(
        "raster_winner_resolve: winner_face_ids must be a non-empty int64 "
        "matrix");
  }
  if (winner_face_ids.shape[0] > std::numeric_limits<int>::max() ||
      winner_face_ids.shape[1] > std::numeric_limits<int>::max()) {
    throw std::runtime_error(
        "raster_winner_resolve: winner dimensions exceed int range");
  }
  const int height = static_cast<int>(winner_face_ids.shape[0]);
  const int width = static_cast<int>(winner_face_ids.shape[1]);
  validate_raster_inputs(clip_positions, faces, height, width,
                         "raster_winner_resolve");
  if (!winner_face_ids.device.is_cuda() ||
      !same_device(clip_positions.device, winner_face_ids.device)) {
    throw std::runtime_error(
        "raster_winner_resolve: winner_face_ids device mismatch");
  }
  validate_launch_threads(pixel_threads,
                          "raster_winner_resolve: pixel_threads");

  const int64_t pixel_count = winner_face_ids.numel();
  NDArray depth({height, width}, DType::Int64, clip_positions.device);
  NDArray barycentric({height, width, 3}, DType::Float32,
                      clip_positions.device);
  const int64_t blocks64 = (pixel_count + pixel_threads - 1) / pixel_threads;
  winner_resolve_kernel<<<static_cast<unsigned>(blocks64), pixel_threads>>>(
      static_cast<const float*>(clip_positions.data_ptr()),
      static_cast<const int64_t*>(faces.data_ptr()), faces.shape[0],
      static_cast<const int64_t*>(winner_face_ids.data_ptr()), height, width,
      pixel_count, static_cast<int64_t*>(depth.data_ptr()),
      static_cast<float*>(barycentric.data_ptr()));
  cuda_check_last("raster_winner_resolve");
  return std::make_tuple(depth, barycentric);
}

std::tuple<NDArray, NDArray, NDArray> rasterize_clip(
    const NDArray& clip_positions, const NDArray& faces, int height, int width,
    int face_threads, int pixel_threads) {
  auto winners = raster_triangle_winners(clip_positions, faces, height, width,
                                         face_threads);
  NDArray winner_face = std::get<1>(winners);
  auto resolved = raster_winner_resolve(clip_positions, faces, winner_face,
                                        pixel_threads);
  return std::make_tuple(winner_face, std::get<1>(resolved),
                         std::get<0>(resolved));
}

}  // namespace tc
