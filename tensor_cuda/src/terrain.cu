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

// Smooth-mode hits use the same material/voxel and face-fallback contract as
// the blocky path, but their distance is the refined 0.5 isosurface crossing.
struct TerrainSmoothHit {
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

constexpr float kSmoothIsoLevel = 0.5f;
constexpr float kSmoothSampleStep = 0.5f;
constexpr int kSmoothRefineIterations = 6;

// Occupancy samples live at voxel centers.  Thus the trilinear lattice cell
// containing `position` starts at floor(position - 0.5), and out-of-grid
// lattice samples are air.  This is render-only: materials remain untouched.
__device__ __forceinline__ float terrain_trilinear_density(
    const uint8_t* __restrict__ materials, int64_t dim_x, int64_t dim_y,
    int64_t dim_z, const float position[3]) {
  int64_t base[3];
  float fraction[3];
#pragma unroll
  for (int axis = 0; axis < 3; ++axis) {
    const float shifted = position[axis] - 0.5f;
    const float lower = floorf(shifted);
    base[axis] = static_cast<int64_t>(lower);
    fraction[axis] = shifted - lower;
  }

  float density = 0.0f;
#pragma unroll
  for (int dz = 0; dz <= 1; ++dz) {
    const float wz = dz == 0 ? 1.0f - fraction[2] : fraction[2];
#pragma unroll
    for (int dy = 0; dy <= 1; ++dy) {
      const float wy = dy == 0 ? 1.0f - fraction[1] : fraction[1];
#pragma unroll
      for (int dx = 0; dx <= 1; ++dx) {
        const float wx = dx == 0 ? 1.0f - fraction[0] : fraction[0];
        const float occupancy = static_cast<float>(
            terrain_grid_load_or_air(materials, dim_x, dim_y, dim_z,
                                     base[0] + dx, base[1] + dy,
                                     base[2] + dz) != 0);
        density = __fadd_rn(density, __fmul_rn(__fmul_rn(wx, wy),
                                                __fmul_rn(wz, occupancy)));
      }
    }
  }
  return density;
}

__device__ __forceinline__ bool terrain_density_cell_is_mixed(
    const uint8_t* __restrict__ materials, int64_t dim_x, int64_t dim_y,
    int64_t dim_z, const int64_t base[3]) {
  bool has_solid = false;
  bool has_air = false;
#pragma unroll
  for (int dz = 0; dz <= 1; ++dz) {
#pragma unroll
    for (int dy = 0; dy <= 1; ++dy) {
#pragma unroll
      for (int dx = 0; dx <= 1; ++dx) {
        const bool occupied = terrain_grid_load_or_air(
                                  materials, dim_x, dim_y, dim_z,
                                  base[0] + dx, base[1] + dy,
                                  base[2] + dz) != 0;
        has_solid = has_solid || occupied;
        has_air = has_air || !occupied;
      }
    }
  }
  return has_solid && has_air;
}

__device__ __forceinline__ void terrain_ray_position(
    const float origin[3], const float direction[3], float distance,
    float position[3]) {
#pragma unroll
  for (int axis = 0; axis < 3; ++axis) {
    position[axis] = __fadd_rn(origin[axis],
                               __fmul_rn(direction[axis], distance));
  }
}

__device__ __forceinline__ float terrain_refine_iso_crossing(
    const uint8_t* __restrict__ materials, int64_t dim_x, int64_t dim_y,
    int64_t dim_z, const float origin[3], const float direction[3],
    float lower_t, float lower_density, float upper_t, float upper_density) {
  // Keep a bracket with density(lower) < iso <= density(upper).  The secant
  // proposal is used when it is safely interior; otherwise bisection keeps
  // the refinement robust at flat/quantized parts of the field.
  for (int iteration = 0; iteration < kSmoothRefineIterations; ++iteration) {
    const float span = upper_t - lower_t;
    float candidate_t = lower_t + 0.5f * span;
    const float denominator = upper_density - lower_density;
    if (fabsf(denominator) > 1.0e-7f) {
      const float secant_t = lower_t +
          (kSmoothIsoLevel - lower_density) * span / denominator;
      const float guard = 0.05f * span;
      if (secant_t > lower_t + guard && secant_t < upper_t - guard) {
        candidate_t = secant_t;
      }
    }
    float candidate_position[3];
    terrain_ray_position(origin, direction, candidate_t, candidate_position);
    const float candidate_density = terrain_trilinear_density(
        materials, dim_x, dim_y, dim_z, candidate_position);
    if (candidate_density >= kSmoothIsoLevel) {
      upper_t = candidate_t;
      upper_density = candidate_density;
    } else {
      lower_t = candidate_t;
      lower_density = candidate_density;
    }
  }
  return upper_t;
}

__device__ __forceinline__ bool terrain_trace_mixed_density_cell(
    const uint8_t* __restrict__ materials, int64_t dim_x, int64_t dim_y,
    int64_t dim_z, const float origin[3], const float direction[3],
    float segment_start, float segment_end, float* hit_distance) {
  float sample_position[3];
  terrain_ray_position(origin, direction, segment_start, sample_position);
  float previous_density = terrain_trilinear_density(
      materials, dim_x, dim_y, dim_z, sample_position);
  if (previous_density >= kSmoothIsoLevel) {
    *hit_distance = segment_start;
    return true;
  }

  float previous_t = segment_start;
  // A shifted DDA density cell is one voxel wide.  A normalized ray can cross
  // at most sqrt(3) voxels in it, so four <=0.5 samples cover the segment.
  for (int sample = 0; sample < 4 && previous_t < segment_end; ++sample) {
    const float current_t = fminf(previous_t + kSmoothSampleStep, segment_end);
    if (current_t <= previous_t) break;
    terrain_ray_position(origin, direction, current_t, sample_position);
    const float current_density = terrain_trilinear_density(
        materials, dim_x, dim_y, dim_z, sample_position);
    if (current_density >= kSmoothIsoLevel) {
      *hit_distance = terrain_refine_iso_crossing(
          materials, dim_x, dim_y, dim_z, origin, direction, previous_t,
          previous_density, current_t, current_density);
      return true;
    }
    previous_t = current_t;
    previous_density = current_density;
  }
  return false;
}

// A point in a trilinear cell always has a solid corner at a real crossing.
// Searching one extra center on each side makes this the actual nearest solid
// voxel (rather than merely the nearest occupied interpolation corner).
__device__ __forceinline__ void terrain_nearest_solid_voxel(
    const uint8_t* __restrict__ materials, int64_t dim_x, int64_t dim_y,
    int64_t dim_z, const float position[3], int64_t voxel[3],
    uint8_t* material) {
  int64_t base[3];
#pragma unroll
  for (int axis = 0; axis < 3; ++axis) {
    base[axis] = static_cast<int64_t>(floorf(position[axis] - 0.5f));
  }
  float best_distance_sq = INFINITY;
  voxel[0] = -1;
  voxel[1] = -1;
  voxel[2] = -1;
  *material = 0;
  for (int64_t x = base[0] - 1; x <= base[0] + 2; ++x) {
    for (int64_t y = base[1] - 1; y <= base[1] + 2; ++y) {
      for (int64_t z = base[2] - 1; z <= base[2] + 2; ++z) {
        const uint8_t candidate = terrain_grid_load_or_air(
            materials, dim_x, dim_y, dim_z, x, y, z);
        if (candidate == 0) continue;
        const float dx = position[0] - (static_cast<float>(x) + 0.5f);
        const float dy = position[1] - (static_cast<float>(y) + 0.5f);
        const float dz = position[2] - (static_cast<float>(z) + 0.5f);
        const float distance_sq = __fadd_rn(
            __fadd_rn(__fmul_rn(dx, dx), __fmul_rn(dy, dy)),
            __fmul_rn(dz, dz));
        if (distance_sq < best_distance_sq) {
          best_distance_sq = distance_sq;
          voxel[0] = x;
          voxel[1] = y;
          voxel[2] = z;
          *material = candidate;
        }
      }
    }
  }
}

__device__ __forceinline__ void terrain_smooth_normal(
    const uint8_t* __restrict__ materials, int64_t dim_x, int64_t dim_y,
    int64_t dim_z, const float position[3], const int64_t nearest_voxel[3],
    int face_axis, int face_sign, float normal[3]) {
  // rho is 1 inside matter.  The outward normal is -grad(rho), expressed as
  // the central difference rho(p-h) - rho(p+h), h=0.5 voxel.
  float outward_gradient[3];
#pragma unroll
  for (int axis = 0; axis < 3; ++axis) {
    float lower[3] = {position[0], position[1], position[2]};
    float upper[3] = {position[0], position[1], position[2]};
    lower[axis] -= 0.5f;
    upper[axis] += 0.5f;
    outward_gradient[axis] = terrain_trilinear_density(
        materials, dim_x, dim_y, dim_z, lower) - terrain_trilinear_density(
        materials, dim_x, dim_y, dim_z, upper);
  }
  const float smooth_length = sqrtf(__fadd_rn(
      __fadd_rn(__fmul_rn(outward_gradient[0], outward_gradient[0]),
                __fmul_rn(outward_gradient[1], outward_gradient[1])),
      __fmul_rn(outward_gradient[2], outward_gradient[2])));
  if (smooth_length >= kGradientEpsilon) {
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
      normal[axis] = __fdiv_rn(outward_gradient[axis], smooth_length);
    }
    return;
  }

  const int64_t x = nearest_voxel[0];
  const int64_t y = nearest_voxel[1];
  const int64_t z = nearest_voxel[2];
  const float blocky_gradient[3] = {
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
  const float blocky_length = sqrtf(__fadd_rn(
      __fadd_rn(__fmul_rn(blocky_gradient[0], blocky_gradient[0]),
                __fmul_rn(blocky_gradient[1], blocky_gradient[1])),
      __fmul_rn(blocky_gradient[2], blocky_gradient[2])));
  normal[0] = 0.0f;
  normal[1] = 0.0f;
  normal[2] = 1.0f;
  if (blocky_length >= kGradientEpsilon) {
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
      normal[axis] = __fdiv_rn(blocky_gradient[axis], blocky_length);
    }
  } else if (face_axis >= 0) {
    normal[0] = 0.0f;
    normal[1] = 0.0f;
    normal[2] = 0.0f;
    normal[face_axis] = static_cast<float>(face_sign);
  }
}

__device__ __forceinline__ void terrain_smooth_first_hit(
    const uint8_t* __restrict__ materials, int64_t dim_x, int64_t dim_y,
    int64_t dim_z, const float origin[3], const float direction[3],
    int max_steps, TerrainSmoothHit* result) {
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
  if (!(has_direction && !parallel_outside && t_exit >= start_t &&
        t_exit >= 0.0f)) {
    return;
  }

  const int step[3] = {terrain_sign(direction[0]), terrain_sign(direction[1]),
                       terrain_sign(direction[2])};
  float start_position[3];
  terrain_ray_position(origin, direction, start_t, start_position);
  if (!origin_inside) {
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
      start_position[axis] = terrain_nextafter(start_position[axis], direction[axis]);
    }
  }
  int64_t current[3] = {
      static_cast<int64_t>(floorf(start_position[0] - 0.5f)),
      static_cast<int64_t>(floorf(start_position[1] - 0.5f)),
      static_cast<int64_t>(floorf(start_position[2] - 0.5f))};
  // Shifted density cells that overlap the original half-open volume have
  // indices [-1, dim-1].  Their corners are exactly the 2x2x2 samples used
  // by the trilinear field above.
  if (current[0] < -1 || current[1] < -1 || current[2] < -1 ||
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
  int face_axis = origin_inside ? -1 : entry_axis;
  int face_sign = origin_inside ? 0 : -step[entry_axis];

  float exact_start_position[3];
  terrain_ray_position(origin, direction, start_t, exact_start_position);
  if (terrain_trilinear_density(materials, dim_x, dim_y, dim_z,
                                exact_start_position) >= kSmoothIsoLevel) {
    uint8_t material = 0;
    int64_t nearest_voxel[3];
    terrain_nearest_solid_voxel(materials, dim_x, dim_y, dim_z,
                                exact_start_position, nearest_voxel, &material);
    if (material != 0) {
      result->hit = true;
      result->material = material;
      result->voxel[0] = nearest_voxel[0];
      result->voxel[1] = nearest_voxel[1];
      result->voxel[2] = nearest_voxel[2];
      result->face_axis = face_axis;
      result->face_sign = face_sign;
      result->distance = start_t;
      return;
    }
  }

  float shifted_origin[3] = {origin[0] - 0.5f, origin[1] - 0.5f,
                             origin[2] - 0.5f};
  float delta_t[3] = {INFINITY, INFINITY, INFINITY};
  float next_t[3] = {INFINITY, INFINITY, INFINITY};
#pragma unroll
  for (int axis = 0; axis < 3; ++axis) {
    if (step[axis] != 0) {
      delta_t[axis] = fabsf(1.0f / direction[axis]);
      const float boundary = step[axis] > 0
                                 ? static_cast<float>(current[axis] + 1)
                                 : static_cast<float>(current[axis]);
      float crossing = (boundary - shifted_origin[axis]) / direction[axis];
      if (crossing < start_t) crossing = start_t;
      next_t[axis] = crossing;
    }
  }

  float segment_start = start_t;
  for (int traversal_step = 0; traversal_step <= max_steps; ++traversal_step) {
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
    const float segment_end = fminf(crossing_t, t_exit);
    if (terrain_density_cell_is_mixed(materials, dim_x, dim_y, dim_z, current)) {
      float hit_distance = INFINITY;
      if (terrain_trace_mixed_density_cell(
              materials, dim_x, dim_y, dim_z, origin, direction, segment_start,
              segment_end, &hit_distance)) {
        float hit_position[3];
        terrain_ray_position(origin, direction, hit_distance, hit_position);
        uint8_t material = 0;
        int64_t nearest_voxel[3];
        terrain_nearest_solid_voxel(materials, dim_x, dim_y, dim_z, hit_position,
                                    nearest_voxel, &material);
        if (material != 0) {
          result->hit = true;
          result->material = material;
          result->voxel[0] = nearest_voxel[0];
          result->voxel[1] = nearest_voxel[1];
          result->voxel[2] = nearest_voxel[2];
          result->face_axis = face_axis;
          result->face_sign = face_sign;
          result->distance = hit_distance;
          return;
        }
      }
    }
    if (crossing_t >= t_exit || traversal_step == max_steps) return;

    current[axis] += step[axis];
    next_t[axis] = crossing_t + delta_t[axis];
    if (current[0] < -1 || current[1] < -1 || current[2] < -1 ||
        current[0] >= dim_x || current[1] >= dim_y || current[2] >= dim_z) {
      return;
    }
    segment_start = crossing_t;
    face_axis = axis;
    face_sign = -step[axis];
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

// This kernel body is intentionally the original WO-7B blocky path.  Keep it
// separate from smooth mode so the default instruction/data path stays intact.
__global__ void terrain_render_blocky_kernel(
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

__global__ void terrain_render_smooth_kernel(
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
  TerrainSmoothHit hit;
  terrain_smooth_first_hit(materials, dim_x, dim_y, dim_z, origin, direction,
                           max_steps, &hit);
  if (!hit.hit) {
    rgb[pixel * 3 + 0] = 0;
    rgb[pixel * 3 + 1] = 0;
    rgb[pixel * 3 + 2] = 0;
    depth[pixel] = -1.0f;
    return;
  }

  float hit_position[3];
  terrain_ray_position(origin, direction, hit.distance, hit_position);
  float normal[3];
  terrain_smooth_normal(materials, dim_x, dim_y, dim_z, hit_position,
                        hit.voxel, hit.face_axis, hit.face_sign, normal);

  const int64_t x = hit.voxel[0];
  const int64_t y = hit.voxel[1];
  const int64_t z = hit.voxel[2];
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
  // The pybind entry point encodes its additive smooth-mode selector in the
  // sign of max_steps so the established C++ terrain-render ABI stays intact.
  // Positive values remain the original blocky contract byte-for-byte.
  if (constants.max_steps == std::numeric_limits<int>::min()) {
    throw std::runtime_error("terrain_render: max_steps is out of range");
  }
  const bool smooth_mode = constants.max_steps < 0;
  const int max_steps = smooth_mode ? -constants.max_steps : constants.max_steps;
  if (max_steps <= 0) {
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
    if (smooth_mode) {
      terrain_render_smooth_kernel<<<blocks, threads>>>(
          static_cast<const uint8_t*>(materials.data_ptr()), materials.shape[0],
          materials.shape[1], materials.shape[2], camera, light,
          static_cast<const uint8_t*>(palette.data_ptr()), palette.shape[0],
          max_steps, static_cast<uint8_t*>(rgb.data_ptr()),
          static_cast<float*>(depth.data_ptr()));
    } else {
      terrain_render_blocky_kernel<<<blocks, threads>>>(
          static_cast<const uint8_t*>(materials.data_ptr()), materials.shape[0],
          materials.shape[1], materials.shape[2], camera, light,
          static_cast<const uint8_t*>(palette.data_ptr()), palette.shape[0],
          max_steps, static_cast<uint8_t*>(rgb.data_ptr()),
          static_cast<float*>(depth.data_ptr()));
    }
    cuda_check_last("terrain_render");
  }
  return std::make_tuple(rgb, depth);
}

}  // namespace tc
