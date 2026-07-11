// terrain_objects.cu: additive axis-aligned voxel-object overlay for
// terrain_render.  The established terrain kernels remain untouched; this
// launch only runs when the public object list is non-empty.

#include "tc/core.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace tc {
namespace {

constexpr int kMaxTerrainObjects = 16;
constexpr float kAmbientFloor = 0.35f;
constexpr float kAoStrength = 0.60f;
constexpr float kAoFloor = 0.40f;

struct TerrainObjectKernelView {
  const uint8_t* grid;
  const uint8_t* palette;
  int64_t dim_x;
  int64_t dim_y;
  int64_t dim_z;
  int64_t palette_rows;
  float origin[3];
};

struct TerrainObjectKernelList {
  TerrainObjectKernelView values[kMaxTerrainObjects];
  int count;
};

struct TerrainObjectHit {
  bool hit;
  uint8_t material;
  int64_t voxel[3];
  int face_axis;
  int face_sign;
  float distance;
};

__device__ __forceinline__ int object_sign(float value) {
  return (value > 0.0f) - (value < 0.0f);
}

__device__ __forceinline__ uint8_t object_grid_load(
    const uint8_t* __restrict__ grid, int64_t dim_y, int64_t dim_z,
    int64_t x, int64_t y, int64_t z) {
  return grid[(x * dim_y + y) * dim_z + z];
}

__device__ __forceinline__ uint8_t object_grid_load_or_air(
    const uint8_t* __restrict__ grid, int64_t dim_x, int64_t dim_y,
    int64_t dim_z, int64_t x, int64_t y, int64_t z) {
  if (x < 0 || y < 0 || z < 0 || x >= dim_x || y >= dim_y || z >= dim_z) {
    return 0;
  }
  return object_grid_load(grid, dim_y, dim_z, x, y, z);
}

// Literal WO-7B half-open AABB/Amanatides-Woo policy, translated into the
// object's local frame.  Pure translation leaves returned ray distance in
// world units.
__device__ __forceinline__ void object_dda_first_hit(
    const TerrainObjectKernelView& object, const float origin[3],
    const float direction[3], int max_steps, float distance_limit,
    TerrainObjectHit* result) {
  result->hit = false;
  result->material = 0;
  result->voxel[0] = -1;
  result->voxel[1] = -1;
  result->voxel[2] = -1;
  result->face_axis = -1;
  result->face_sign = 0;
  result->distance = INFINITY;

  const float dims[3] = {static_cast<float>(object.dim_x),
                         static_cast<float>(object.dim_y),
                         static_cast<float>(object.dim_z)};
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
  // Strict overlay comparison means an AABB entered at or behind the current
  // terrain/object winner cannot change this pixel.
  if (!valid || start_t >= distance_limit) return;

  const int step[3] = {object_sign(direction[0]), object_sign(direction[1]),
                       object_sign(direction[2])};
  float start_position[3] = {
      __fadd_rn(origin[0], __fmul_rn(direction[0], start_t)),
      __fadd_rn(origin[1], __fmul_rn(direction[1], start_t)),
      __fadd_rn(origin[2], __fmul_rn(direction[2], start_t))};
  if (!origin_inside) {
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
      start_position[axis] =
          nextafterf(start_position[axis], start_position[axis] + direction[axis]);
    }
  }

  int64_t current[3] = {static_cast<int64_t>(floorf(start_position[0])),
                        static_cast<int64_t>(floorf(start_position[1])),
                        static_cast<int64_t>(floorf(start_position[2]))};
  if (current[0] < 0 || current[1] < 0 || current[2] < 0 ||
      current[0] >= object.dim_x || current[1] >= object.dim_y ||
      current[2] >= object.dim_z) {
    return;
  }

  int entry_axis = 0;
  float entry_near = near_t[0];
  if (near_t[1] > entry_near) {
    entry_axis = 1;
    entry_near = near_t[1];
  }
  if (near_t[2] > entry_near) entry_axis = 2;

  const uint8_t initial_material = object_grid_load(
      object.grid, object.dim_y, object.dim_z, current[0], current[1],
      current[2]);
  if (initial_material != 0) {
    result->hit = true;
    result->material = initial_material;
    result->voxel[0] = current[0];
    result->voxel[1] = current[1];
    result->voxel[2] = current[2];
    result->distance = start_t;
    if (!origin_inside) {
      result->face_axis = entry_axis;
      result->face_sign = -step[entry_axis];
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

  for (int traversal_step = 0; traversal_step < max_steps; ++traversal_step) {
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
        current[0] >= object.dim_x || current[1] >= object.dim_y ||
        current[2] >= object.dim_z) {
      return;
    }

    const uint8_t current_material = object_grid_load(
        object.grid, object.dim_y, object.dim_z, current[0], current[1],
        current[2]);
    if (current_material != 0) {
      result->hit = true;
      result->material = current_material;
      result->voxel[0] = current[0];
      result->voxel[1] = current[1];
      result->voxel[2] = current[2];
      result->face_axis = axis;
      result->face_sign = -step[axis];
      result->distance = crossing_t;
      return;
    }
  }
}

__device__ __forceinline__ uint8_t object_to_u8(float value) {
  value = fminf(fmaxf(value, 0.0f), 255.0f);
  return static_cast<uint8_t>(floorf(value + 0.5f));
}

__global__ void terrain_render_objects_overlay_kernel(
    TerrainRenderCamera camera, TerrainRenderLight light,
    TerrainObjectKernelList objects, int max_steps,
    uint8_t* __restrict__ rgb, float* __restrict__ depth) {
  const int64_t pixel =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t pixel_count =
      static_cast<int64_t>(camera.width) * camera.height;
  if (pixel >= pixel_count) return;

  const int pixel_x = static_cast<int>(pixel % camera.width);
  const int pixel_y = static_cast<int>(pixel / camera.width);
  const float x_center = static_cast<float>(pixel_x) + 0.5f;
  const float y_center = static_cast<float>(pixel_y) + 0.5f;
  const float screen_x = __fmul_rn(
      __fadd_rn(__fmul_rn(__fdiv_rn(x_center, static_cast<float>(camera.width)),
                          2.0f),
                -1.0f),
      camera.half_width);
  const float screen_y = __fmul_rn(
      __fadd_rn(1.0f,
                -__fmul_rn(__fdiv_rn(y_center, static_cast<float>(camera.height)),
                            2.0f)),
      camera.half_height);

  float direction[3];
#pragma unroll
  for (int axis = 0; axis < 3; ++axis) {
    const float horizontal = __fmul_rn(screen_x, camera.right[axis]);
    const float vertical = __fmul_rn(screen_y, camera.up[axis]);
    direction[axis] =
        __fadd_rn(__fadd_rn(camera.forward[axis], horizontal), vertical);
  }
  const float length_sq = __fadd_rn(
      __fadd_rn(__fmul_rn(direction[0], direction[0]),
                __fmul_rn(direction[1], direction[1])),
      __fmul_rn(direction[2], direction[2]));
  const float length = sqrtf(length_sq);
#pragma unroll
  for (int axis = 0; axis < 3; ++axis) {
    direction[axis] = __fdiv_rn(direction[axis], length);
  }

  float best_distance = depth[pixel] >= 0.0f ? depth[pixel] : INFINITY;
  int best_object = -1;
  TerrainObjectHit best_hit{};
  for (int object_index = 0; object_index < objects.count; ++object_index) {
    const TerrainObjectKernelView& object = objects.values[object_index];
    const float local_origin[3] = {
        __fadd_rn(camera.position[0], -object.origin[0]),
        __fadd_rn(camera.position[1], -object.origin[1]),
        __fadd_rn(camera.position[2], -object.origin[2])};
    TerrainObjectHit hit;
    object_dda_first_hit(object, local_origin, direction, max_steps,
                         best_distance, &hit);
    if (hit.hit && hit.distance < best_distance) {
      best_distance = hit.distance;
      best_object = object_index;
      best_hit = hit;
    }
  }
  if (best_object < 0) return;

  const TerrainObjectKernelView& object = objects.values[best_object];
  float normal[3] = {0.0f, 0.0f, 1.0f};
  if (best_hit.face_axis >= 0) {
    normal[0] = 0.0f;
    normal[1] = 0.0f;
    normal[2] = 0.0f;
    normal[best_hit.face_axis] = static_cast<float>(best_hit.face_sign);
  }

  int occupied_neighbors = 0;
#pragma unroll
  for (int dz = -1; dz <= 1; ++dz) {
#pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
#pragma unroll
      for (int dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0 && dz == 0) continue;
        occupied_neighbors += object_grid_load_or_air(
                                  object.grid, object.dim_x, object.dim_y,
                                  object.dim_z, best_hit.voxel[0] + dx,
                                  best_hit.voxel[1] + dy,
                                  best_hit.voxel[2] + dz) != 0;
      }
    }
  }
  const float occlusion_fraction =
      static_cast<float>(occupied_neighbors) * (1.0f / 26.0f);
  const float ao = fminf(
      fmaxf(1.0f - kAoStrength * occlusion_fraction, kAoFloor), 1.0f);
  const float lambert =
      -(normal[0] * light.direction[0] + normal[1] * light.direction[1] +
        normal[2] * light.direction[2]);
  const float diffuse =
      fminf(fmaxf(lambert, kAmbientFloor), 1.0f);
  const int64_t palette_index =
      static_cast<int64_t>(best_hit.material) < object.palette_rows
          ? static_cast<int64_t>(best_hit.material)
          : object.palette_rows - 1;
  const float shading = diffuse * ao;
  rgb[pixel * 3 + 0] = object_to_u8(
      static_cast<float>(object.palette[palette_index * 3 + 0]) * shading);
  rgb[pixel * 3 + 1] = object_to_u8(
      static_cast<float>(object.palette[palette_index * 3 + 1]) * shading);
  rgb[pixel * 3 + 2] = object_to_u8(
      static_cast<float>(object.palette[palette_index * 3 + 2]) * shading);
  depth[pixel] = best_distance;
}

bool same_device(const Device& lhs, const Device& rhs) {
  return lhs.type == rhs.type && lhs.index == rhs.index;
}

}  // namespace

void terrain_render_objects_overlay(
    NDArray& rgb, NDArray& depth, const TerrainRenderCamera& camera,
    const TerrainRenderLight& light,
    const std::vector<TerrainRenderObject>& objects, int max_steps) {
  if (objects.empty()) return;
  if (objects.size() > kMaxTerrainObjects) {
    throw std::runtime_error(
        "terrain_render: objects supports at most 16 entries");
  }
  if (rgb.dtype != DType::Uint8 || rgb.shape != Shape{camera.height, camera.width, 3}) {
    throw std::runtime_error(
        "terrain_render: object overlay RGB output contract mismatch");
  }
  if (depth.dtype != DType::Float32 ||
      depth.shape != Shape{camera.height, camera.width}) {
    throw std::runtime_error(
        "terrain_render: object overlay depth output contract mismatch");
  }
  if (!rgb.device.is_cuda() || !same_device(rgb.device, depth.device)) {
    throw std::runtime_error(
        "terrain_render: object overlay requires matching CUDA outputs");
  }
  if (max_steps <= 0) {
    throw std::runtime_error("terrain_render: max_steps must be positive");
  }

  TerrainObjectKernelList kernel_objects{};
  kernel_objects.count = static_cast<int>(objects.size());
  for (size_t index = 0; index < objects.size(); ++index) {
    const TerrainRenderObject& object = objects[index];
    if (object.grid.ndim() != 3 || object.grid.dtype != DType::Uint8 ||
        object.grid.shape[0] <= 0 || object.grid.shape[1] <= 0 ||
        object.grid.shape[2] <= 0) {
      throw std::runtime_error(
          "terrain_render: object grid must be non-empty uint8 [X,Y,Z]");
    }
    if (object.palette.ndim() != 2 || object.palette.dtype != DType::Uint8 ||
        object.palette.shape[0] <= 0 || object.palette.shape[1] != 3) {
      throw std::runtime_error(
          "terrain_render: object palette must be non-empty uint8 [N,3]");
    }
    if (!object.grid.device.is_cuda() || !object.palette.device.is_cuda() ||
        !same_device(rgb.device, object.grid.device) ||
        !same_device(rgb.device, object.palette.device)) {
      throw std::runtime_error(
          "terrain_render: object grids and palettes must share the terrain device");
    }
    for (int axis = 0; axis < 3; ++axis) {
      if (!std::isfinite(object.origin[axis])) {
        throw std::runtime_error(
            "terrain_render: object origin must be a finite three-vector");
      }
    }

    TerrainObjectKernelView& view = kernel_objects.values[index];
    view.grid = static_cast<const uint8_t*>(object.grid.data_ptr());
    view.palette = static_cast<const uint8_t*>(object.palette.data_ptr());
    view.dim_x = object.grid.shape[0];
    view.dim_y = object.grid.shape[1];
    view.dim_z = object.grid.shape[2];
    view.palette_rows = object.palette.shape[0];
    view.origin[0] = object.origin[0];
    view.origin[1] = object.origin[1];
    view.origin[2] = object.origin[2];
  }

  const int64_t pixel_count =
      static_cast<int64_t>(camera.width) * camera.height;
  if (pixel_count > 0) {
    constexpr int threads = 256;
    const int blocks = static_cast<int>((pixel_count + threads - 1) / threads);
    terrain_render_objects_overlay_kernel<<<blocks, threads>>>(
        camera, light, kernel_objects, max_steps,
        static_cast<uint8_t*>(rgb.data_ptr()),
        static_cast<float*>(depth.data_ptr()));
    cuda_check_last("terrain_render(objects)");
  }
}

}  // namespace tc
