// terrain.cu: fused camera ray generation, voxel DDA, and terrain shading.
//
// The DDA path below deliberately mirrors src/dda.cu's traversal semantics.
// It remains separate so the established dda_raycast operation is untouched.

#include "tc/core.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace tc {
namespace {

constexpr float kAmbientFloor = 0.35f;
constexpr float kAoStrength = 0.60f;
constexpr float kAoFloor = 0.40f;
constexpr float kGradientEpsilon = 1.0e-6f;
constexpr float kValueJitter = 0.08f;
constexpr float kChannelMix = 0.04f;

__device__ __forceinline__ int terrain_sign(float value) {
  return (value > 0.0f) - (value < 0.0f);
}

__device__ __forceinline__ float terrain_nextafter(float value,
                                                    float direction) {
  return nextafterf(value, value + direction);
}

__device__ __forceinline__ uint8_t terrain_grid_load(
    const uint8_t* __restrict__ materials, int64_t dim_y, int64_t dim_z,
    int64_t x, int64_t y, int64_t z) {
  return materials[(x * dim_y + y) * dim_z + z];
}

__device__ __forceinline__ uint8_t terrain_grid_load_or_air(
    const uint8_t* __restrict__ materials, int64_t dim_x, int64_t dim_y,
    int64_t dim_z, int64_t x, int64_t y, int64_t z) {
  if (x < 0 || y < 0 || z < 0 || x >= dim_x || y >= dim_y || z >= dim_z) {
    return 0;
  }
  return terrain_grid_load(materials, dim_y, dim_z, x, y, z);
}

struct TerrainDdaHit {
  bool hit;
  uint8_t material;
  int64_t voxel[3];
  int face_axis;
  int face_sign;
  float distance;
};

// This is the same Amanatides-Woo policy as dda_raycast: half-open bounds,
// an external nextafter nudge, low-axis tie breaks, and explicit f32 FMA
// avoidance at the entry point.
__device__ __forceinline__ void terrain_dda_first_hit(
    const uint8_t* __restrict__ materials, int64_t dim_x, int64_t dim_y,
    int64_t dim_z, const float origin[3], const float direction[3],
    int max_steps, TerrainDdaHit* result) {
  result->hit = false;
  result->material = 0;
  result->voxel[0] = -1;
  result->voxel[1] = -1;
  result->voxel[2] = -1;
  result->face_axis = -1;
  result->face_sign = 0;
  result->distance = INFINITY;

  const float dims[3] = {static_cast<float>(dim_x),
                         static_cast<float>(dim_y),
                         static_cast<float>(dim_z)};
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
      origin[0] >= 0.0f && origin[0] < dims[0] && origin[1] >= 0.0f &&
      origin[1] < dims[1] && origin[2] >= 0.0f && origin[2] < dims[2];
  const bool valid = has_direction && !parallel_outside &&
                     t_exit >= start_t && t_exit >= 0.0f;
  if (!valid) return;

  const int step[3] = {terrain_sign(direction[0]), terrain_sign(direction[1]),
                       terrain_sign(direction[2])};
  float start_position[3] = {
      __fadd_rn(origin[0], __fmul_rn(direction[0], start_t)),
      __fadd_rn(origin[1], __fmul_rn(direction[1], start_t)),
      __fadd_rn(origin[2], __fmul_rn(direction[2], start_t))};
  if (!origin_inside) {
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
      start_position[axis] = terrain_nextafter(start_position[axis], direction[axis]);
    }
  }

  int64_t current[3] = {static_cast<int64_t>(floorf(start_position[0])),
                        static_cast<int64_t>(floorf(start_position[1])),
                        static_cast<int64_t>(floorf(start_position[2]))};
  if (current[0] < 0 || current[1] < 0 || current[2] < 0 ||
      current[0] >= dim_x || current[1] >= dim_y || current[2] >= dim_z) {
    return;
  }

  int entry_axis = 0;
  float entry_near = near_t[0];
  if (near_t[1] > entry_near) {
    entry_axis = 1;
    entry_near = near_t[1];
  }
  if (near_t[2] > entry_near) entry_axis = 2;

  const uint8_t initial_material = terrain_grid_load(
      materials, dim_y, dim_z, current[0], current[1], current[2]);
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
        current[0] >= dim_x || current[1] >= dim_y || current[2] >= dim_z) {
      return;
    }

    const uint8_t current_material = terrain_grid_load(
        materials, dim_y, dim_z, current[0], current[1], current[2]);
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

__device__ __forceinline__ uint64_t terrain_splitmix64(uint64_t value) {
  value += UINT64_C(0x9E3779B97F4A7C15);
  value = (value ^ (value >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
  value = (value ^ (value >> 27)) * UINT64_C(0x94D049BB133111EB);
  return value ^ (value >> 31);
}

__device__ __forceinline__ float terrain_signed_hash_byte(uint64_t hash,
                                                           int shift) {
  constexpr float kInv255 = 1.0f / 255.0f;
  const float value = static_cast<float>((hash >> shift) & UINT64_C(0xFF));
  return __fadd_rn(__fmul_rn(__fmul_rn(value, kInv255), 2.0f), -1.0f);
}

__device__ __forceinline__ uint8_t terrain_to_u8(float value) {
  value = fminf(fmaxf(value, 0.0f), 255.0f);
  return static_cast<uint8_t>(floorf(value + 0.5f));
}

__global__ void terrain_render_kernel(
    const uint8_t* __restrict__ materials, int64_t dim_x, int64_t dim_y,
    int64_t dim_z, TerrainRenderCamera camera, TerrainRenderLight light,
    const uint8_t* __restrict__ palette, int64_t palette_rows, int max_steps,
    uint8_t* __restrict__ rgb, float* __restrict__ depth) {
  const int64_t pixel = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t pixel_count = static_cast<int64_t>(camera.width) * camera.height;
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
    direction[axis] = __fadd_rn(__fadd_rn(camera.forward[axis], horizontal), vertical);
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

  const float origin[3] = {camera.position[0], camera.position[1], camera.position[2]};
  TerrainDdaHit hit;
  terrain_dda_first_hit(materials, dim_x, dim_y, dim_z, origin, direction,
                        max_steps, &hit);
  if (!hit.hit) {
    rgb[pixel * 3 + 0] = 0;
    rgb[pixel * 3 + 1] = 0;
    rgb[pixel * 3 + 2] = 0;
    depth[pixel] = -1.0f;
    return;
  }

  // Occupancy is 1 inside solid matter.  The outward normal is the negative
  // central density gradient, written as occ(-axis) - occ(+axis), so it has
  // the same orientation as the DDA entry face fallback.
  const int64_t x = hit.voxel[0];
  const int64_t y = hit.voxel[1];
  const int64_t z = hit.voxel[2];
  const float gradient[3] = {
      static_cast<float>(terrain_grid_load_or_air(materials, dim_x, dim_y, dim_z,
                                                   x - 1, y, z) != 0) -
          static_cast<float>(terrain_grid_load_or_air(materials, dim_x, dim_y, dim_z,
                                                       x + 1, y, z) != 0),
      static_cast<float>(terrain_grid_load_or_air(materials, dim_x, dim_y, dim_z,
                                                   x, y - 1, z) != 0) -
          static_cast<float>(terrain_grid_load_or_air(materials, dim_x, dim_y, dim_z,
                                                       x, y + 1, z) != 0),
      static_cast<float>(terrain_grid_load_or_air(materials, dim_x, dim_y, dim_z,
                                                   x, y, z - 1) != 0) -
          static_cast<float>(terrain_grid_load_or_air(materials, dim_x, dim_y, dim_z,
                                                       x, y, z + 1) != 0)};
  const float gradient_length = sqrtf(__fadd_rn(
      __fadd_rn(__fmul_rn(gradient[0], gradient[0]),
                __fmul_rn(gradient[1], gradient[1])),
      __fmul_rn(gradient[2], gradient[2])));
  float normal[3] = {0.0f, 0.0f, 1.0f};
  if (gradient_length >= kGradientEpsilon) {
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) normal[axis] = __fdiv_rn(gradient[axis], gradient_length);
  } else if (hit.face_axis >= 0) {
    normal[0] = 0.0f;
    normal[1] = 0.0f;
    normal[2] = 0.0f;
    normal[hit.face_axis] = static_cast<float>(hit.face_sign);
  }

  int occupied_neighbors = 0;
#pragma unroll
  for (int dz = -1; dz <= 1; ++dz) {
#pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
#pragma unroll
      for (int dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0 && dz == 0) continue;
        occupied_neighbors += terrain_grid_load_or_air(materials, dim_x, dim_y, dim_z,
                                                       x + dx, y + dy, z + dz) != 0;
      }
    }
  }
  const float occlusion_fraction = static_cast<float>(occupied_neighbors) * (1.0f / 26.0f);
  const float ao = fminf(fmaxf(1.0f - kAoStrength * occlusion_fraction, kAoFloor), 1.0f);
  const float lambert = -(normal[0] * light.direction[0] + normal[1] * light.direction[1] +
                          normal[2] * light.direction[2]);
  const float diffuse = fminf(fmaxf(lambert, kAmbientFloor), 1.0f);

  const int64_t palette_index = min(static_cast<int64_t>(hit.material), palette_rows - 1);
  float color[3] = {static_cast<float>(palette[palette_index * 3 + 0]),
                    static_cast<float>(palette[palette_index * 3 + 1]),
                    static_cast<float>(palette[palette_index * 3 + 2])};
  const uint64_t hash_input =
      static_cast<uint64_t>(x) * UINT64_C(0x9E3779B97F4A7C15) ^
      static_cast<uint64_t>(y) * UINT64_C(0xBF58476D1CE4E5B9) ^
      static_cast<uint64_t>(z) * UINT64_C(0x94D049BB133111EB);
  const uint64_t hash = terrain_splitmix64(hash_input);
  const float value_scale = 1.0f + terrain_signed_hash_byte(hash, 56) * kValueJitter;
  const float mix_r = terrain_signed_hash_byte(hash, 48) * kChannelMix;
  const float mix_g = terrain_signed_hash_byte(hash, 40) * kChannelMix;
  const float mix_b = terrain_signed_hash_byte(hash, 32) * kChannelMix;
  const float mixed_color[3] = {
      color[0] + mix_r * (color[1] - color[0]),
      color[1] + mix_g * (color[2] - color[1]),
      color[2] + mix_b * (color[0] - color[2])};
  const float shading = diffuse * ao * value_scale;
  rgb[pixel * 3 + 0] = terrain_to_u8(mixed_color[0] * shading);
  rgb[pixel * 3 + 1] = terrain_to_u8(mixed_color[1] * shading);
  rgb[pixel * 3 + 2] = terrain_to_u8(mixed_color[2] * shading);
  depth[pixel] = hit.distance;
}

bool same_device(const Device& lhs, const Device& rhs) {
  return lhs.type == rhs.type && lhs.index == rhs.index;
}

bool finite(float value) { return std::isfinite(value); }

}  // namespace

std::tuple<NDArray, NDArray> terrain_render(
    const NDArray& materials, const TerrainRenderCamera& camera,
    const TerrainRenderLight& light, const NDArray& palette,
    const TerrainRenderConstants& constants) {
  if (materials.ndim() != 3 || materials.dtype != DType::Uint8) {
    throw std::runtime_error(
        "terrain_render: materials must be a 3D uint8 tensor");
  }
  if (materials.shape[0] <= 0 || materials.shape[1] <= 0 || materials.shape[2] <= 0) {
    throw std::runtime_error("terrain_render: material dimensions must be positive");
  }
  if (palette.ndim() != 2 || palette.shape[0] <= 0 || palette.shape[1] != 3 ||
      palette.dtype != DType::Uint8) {
    throw std::runtime_error(
        "terrain_render: palette must be a non-empty uint8 tensor with shape (N, 3)");
  }
  if (!materials.device.is_cuda() || !palette.device.is_cuda()) {
    throw std::runtime_error("terrain_render: CUDA materials and palette tensors required");
  }
  if (!same_device(materials.device, palette.device)) {
    throw std::runtime_error("terrain_render: materials and palette device mismatch");
  }
  if (camera.width <= 0 || camera.height <= 0) {
    throw std::runtime_error("terrain_render: image dimensions must be positive");
  }
  if (constants.max_steps <= 0) {
    throw std::runtime_error("terrain_render: max_steps must be positive");
  }
  if (!finite(camera.half_width) || !finite(camera.half_height) ||
      camera.half_width <= 0.0f || camera.half_height <= 0.0f) {
    throw std::runtime_error("terrain_render: camera half extents must be finite and positive");
  }
  for (int axis = 0; axis < 3; ++axis) {
    if (!finite(camera.position[axis]) || !finite(camera.forward[axis]) ||
        !finite(camera.right[axis]) || !finite(camera.up[axis]) ||
        !finite(light.direction[axis])) {
      throw std::runtime_error("terrain_render: camera and light values must be finite");
    }
  }

  NDArray rgb({camera.height, camera.width, 3}, DType::Uint8, materials.device);
  NDArray depth({camera.height, camera.width}, DType::Float32, materials.device);
  const int64_t pixel_count = static_cast<int64_t>(camera.width) * camera.height;
  if (pixel_count > 0) {
    constexpr int threads = 256;
    const int blocks = static_cast<int>((pixel_count + threads - 1) / threads);
    terrain_render_kernel<<<blocks, threads>>>(
        static_cast<const uint8_t*>(materials.data_ptr()), materials.shape[0],
        materials.shape[1], materials.shape[2], camera, light,
        static_cast<const uint8_t*>(palette.data_ptr()), palette.shape[0],
        constants.max_steps, static_cast<uint8_t*>(rgb.data_ptr()),
        static_cast<float*>(depth.data_ptr()));
    cuda_check_last("terrain_render");
  }
  return std::make_tuple(rgb, depth);
}

}  // namespace tc
