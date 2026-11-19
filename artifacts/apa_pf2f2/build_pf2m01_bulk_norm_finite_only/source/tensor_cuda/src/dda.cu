// dda.cu: one-thread-per-ray Amanatides-Woo voxel traversal.

#include "tc/core.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <tuple>

namespace tc {
namespace {

__device__ __forceinline__ int dda_sign(float value) {
  return (value > 0.0f) - (value < 0.0f);
}

__device__ __forceinline__ float dda_nextafter(float value,
                                                float direction) {
  return nextafterf(value, value + direction);
}

__device__ __forceinline__ uint8_t dda_grid_load(
    const uint8_t* __restrict__ grid, int64_t dim_y, int64_t dim_z,
    int64_t x, int64_t y, int64_t z) {
  return grid[(x * dim_y + y) * dim_z + z];
}

__global__ void dda_raycast_kernel(
    const uint8_t* __restrict__ grid, int64_t dim_x, int64_t dim_y,
    int64_t dim_z, const float* __restrict__ origins,
    const float* __restrict__ directions, int64_t ray_count, int max_steps,
    uint8_t* __restrict__ hit, uint8_t* __restrict__ material,
    int64_t* __restrict__ voxel, int64_t* __restrict__ face_axis,
    int64_t* __restrict__ face_sign, float* __restrict__ distance) {
  const int64_t ray = static_cast<int64_t>(blockIdx.x) * blockDim.x
                    + threadIdx.x;
  if (ray >= ray_count) return;

  // Miss defaults mirror Project-Scorch render.dda._empty_result.
  hit[ray] = 0;
  material[ray] = 0;
  voxel[ray * 3 + 0] = -1;
  voxel[ray * 3 + 1] = -1;
  voxel[ray * 3 + 2] = -1;
  face_axis[ray] = -1;
  face_sign[ray] = 0;
  distance[ray] = INFINITY;

  const float origin[3] = {
      origins[ray * 3 + 0], origins[ray * 3 + 1], origins[ray * 3 + 2]};
  const float direction[3] = {
      directions[ray * 3 + 0], directions[ray * 3 + 1],
      directions[ray * 3 + 2]};
  const float dims[3] = {
      static_cast<float>(dim_x), static_cast<float>(dim_y),
      static_cast<float>(dim_z)};

  // Slab intersection avoids relying on 0 * infinity at box faces.
  float near_t[3] = {-INFINITY, -INFINITY, -INFINITY};
  float far_t[3] = {INFINITY, INFINITY, INFINITY};
  bool parallel_outside = false;
  bool has_direction = false;

#pragma unroll
  for (int axis = 0; axis < 3; ++axis) {
    const float component = direction[axis];
    if (component > 0.0f) {
      has_direction = true;
      near_t[axis] = (0.0f - origin[axis]) / component;
      far_t[axis] = (dims[axis] - origin[axis]) / component;
    } else if (component < 0.0f) {
      has_direction = true;
      near_t[axis] = (dims[axis] - origin[axis]) / component;
      far_t[axis] = (0.0f - origin[axis]) / component;
    } else if (origin[axis] < 0.0f || origin[axis] >= dims[axis]) {
      parallel_outside = true;
    }
  }

  float t_enter = near_t[0];
  if (near_t[1] > t_enter) t_enter = near_t[1];
  if (near_t[2] > t_enter) t_enter = near_t[2];

  float t_exit = far_t[0];
  if (far_t[1] < t_exit) t_exit = far_t[1];
  if (far_t[2] < t_exit) t_exit = far_t[2];

  const float start_t = fmaxf(t_enter, 0.0f);
  const bool origin_inside =
      origin[0] >= 0.0f && origin[0] < dims[0] &&
      origin[1] >= 0.0f && origin[1] < dims[1] &&
      origin[2] >= 0.0f && origin[2] < dims[2];
  const bool valid = has_direction && !parallel_outside &&
                     t_exit >= start_t && t_exit >= 0.0f;
  if (!valid) return;

  const int step[3] = {
      dda_sign(direction[0]), dda_sign(direction[1]),
      dda_sign(direction[2])};

  // NumPy performs the multiply and add as separate float32 ufunc steps.
  // Force the same rounding here: an FMA residual at an external slab face
  // can otherwise leave the entry coordinate just outside the grid, farther
  // than the subsequent one-ulp nextafter nudge can repair.
  float start_position[3] = {
      __fadd_rn(origin[0], __fmul_rn(direction[0], start_t)),
      __fadd_rn(origin[1], __fmul_rn(direction[1], start_t)),
      __fadd_rn(origin[2], __fmul_rn(direction[2], start_t))};
  if (!origin_inside) {
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
      start_position[axis] =
          dda_nextafter(start_position[axis], direction[axis]);
    }
  }

  int64_t current[3] = {
      static_cast<int64_t>(floorf(start_position[0])),
      static_cast<int64_t>(floorf(start_position[1])),
      static_cast<int64_t>(floorf(start_position[2]))};
  if (current[0] < 0 || current[1] < 0 || current[2] < 0 ||
      current[0] >= dim_x || current[1] >= dim_y ||
      current[2] >= dim_z) {
    return;
  }

  // NumPy argmax/argmin break ties at the lowest axis index.
  int entry_axis = 0;
  float entry_near = near_t[0];
  if (near_t[1] > entry_near) {
    entry_axis = 1;
    entry_near = near_t[1];
  }
  if (near_t[2] > entry_near) entry_axis = 2;

  const uint8_t initial_material = dda_grid_load(
      grid, dim_y, dim_z, current[0], current[1], current[2]);
  if (initial_material != 0) {
    hit[ray] = 1;
    material[ray] = initial_material;
    voxel[ray * 3 + 0] = current[0];
    voxel[ray * 3 + 1] = current[1];
    voxel[ray * 3 + 2] = current[2];
    distance[ray] = start_t;
    if (!origin_inside) {
      face_axis[ray] = entry_axis;
      face_sign[ray] = -step[entry_axis];
    }
    return;
  }

  float delta_t[3] = {INFINITY, INFINITY, INFINITY};
  float next_t[3] = {INFINITY, INFINITY, INFINITY};
#pragma unroll
  for (int axis = 0; axis < 3; ++axis) {
    if (step[axis] != 0) {
      delta_t[axis] = fabsf(1.0f / direction[axis]);
      const float boundary = step[axis] > 0
          ? static_cast<float>(current[axis] + 1)
          : static_cast<float>(current[axis]);
      float crossing = (boundary - origin[axis]) / direction[axis];
      if (crossing < start_t) crossing = start_t;
      next_t[axis] = crossing;
    }
  }

  for (int traversal_step = 0; traversal_step < max_steps;
       ++traversal_step) {
    int axis = 0;
    float crossing_t = next_t[0];
    if (next_t[1] < crossing_t) {
      axis = 1;
      crossing_t = next_t[1];
    }
    if (next_t[2] < crossing_t) {
      axis = 2;
      crossing_t = next_t[2];
    }

    current[axis] += step[axis];
    next_t[axis] = crossing_t + delta_t[axis];

    if (crossing_t > t_exit) return;
    if (current[0] < 0 || current[1] < 0 || current[2] < 0 ||
        current[0] >= dim_x || current[1] >= dim_y ||
        current[2] >= dim_z) {
      return;
    }

    const uint8_t current_material = dda_grid_load(
        grid, dim_y, dim_z, current[0], current[1], current[2]);
    if (current_material != 0) {
      hit[ray] = 1;
      material[ray] = current_material;
      voxel[ray * 3 + 0] = current[0];
      voxel[ray * 3 + 1] = current[1];
      voxel[ray * 3 + 2] = current[2];
      face_axis[ray] = axis;
      face_sign[ray] = -step[axis];
      distance[ray] = crossing_t;
      return;
    }
  }
}

bool same_device(const Device& lhs, const Device& rhs) {
  return lhs.type == rhs.type && lhs.index == rhs.index;
}

}  // namespace

std::tuple<NDArray, NDArray, NDArray, NDArray, NDArray, NDArray> dda_raycast(
    const NDArray& grid, const NDArray& origins, const NDArray& directions,
    int max_steps) {
  if (grid.ndim() != 3 || grid.dtype != DType::Uint8)
    throw std::runtime_error("dda_raycast: grid must be a 3D uint8 tensor");
  if (grid.shape[0] <= 0 || grid.shape[1] <= 0 || grid.shape[2] <= 0)
    throw std::runtime_error("dda_raycast: grid dimensions must be positive");
  if (origins.ndim() != 2 || origins.shape[1] != 3 ||
      origins.dtype != DType::Float32)
    throw std::runtime_error(
        "dda_raycast: origins must be a float32 tensor with shape (N, 3)");
  if (directions.ndim() != 2 || directions.shape != origins.shape ||
      directions.dtype != DType::Float32)
    throw std::runtime_error(
        "dda_raycast: directions must be float32 with the same (N, 3) shape");
  if (!grid.device.is_cuda() || !origins.device.is_cuda() ||
      !directions.device.is_cuda())
    throw std::runtime_error("dda_raycast: CUDA tensors required");
  if (!same_device(grid.device, origins.device) ||
      !same_device(grid.device, directions.device))
    throw std::runtime_error("dda_raycast: device mismatch");
  if (max_steps <= 0)
    throw std::runtime_error("dda_raycast: max_steps must be positive");

  const int64_t ray_count = origins.shape[0];
  NDArray hit({ray_count}, DType::Uint8, grid.device);
  NDArray material({ray_count}, DType::Uint8, grid.device);
  NDArray voxel({ray_count, 3}, DType::Int64, grid.device);
  NDArray face_axis({ray_count}, DType::Int64, grid.device);
  NDArray face_sign({ray_count}, DType::Int64, grid.device);
  NDArray distance({ray_count}, DType::Float32, grid.device);

  if (ray_count > 0) {
    constexpr int threads = 256;
    const int blocks = static_cast<int>((ray_count + threads - 1) / threads);
    dda_raycast_kernel<<<blocks, threads>>>(
        static_cast<const uint8_t*>(grid.data_ptr()), grid.shape[0],
        grid.shape[1], grid.shape[2],
        static_cast<const float*>(origins.data_ptr()),
        static_cast<const float*>(directions.data_ptr()), ray_count, max_steps,
        static_cast<uint8_t*>(hit.data_ptr()),
        static_cast<uint8_t*>(material.data_ptr()),
        static_cast<int64_t*>(voxel.data_ptr()),
        static_cast<int64_t*>(face_axis.data_ptr()),
        static_cast<int64_t*>(face_sign.data_ptr()),
        static_cast<float*>(distance.data_ptr()));
    cuda_check_last("dda_raycast");
  }

  return std::make_tuple(hit, material, voxel, face_axis, face_sign, distance);
}

}  // namespace tc
