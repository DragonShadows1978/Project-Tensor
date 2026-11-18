// paint_bake.cu: fused texture-view back projection and ordered cosine blend.
//
// PAINT-CUDA-2 keeps every output aligned to the input atlas sample.  That is
// intentional: ColdCast's dense indexed += remains a later integration
// concern, while this engine leg can prove projection/gather/blend arithmetic
// without introducing duplicate-sensitive scatter or floating atomics.

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

bool same_device(const Device& lhs, const Device& rhs) {
  return lhs.type == rhs.type && lhs.index == rhs.index;
}

void validate_launch_threads(int threads, const char* argument) {
  if (threads < 32 || threads > 1024 || threads % 32 != 0) {
    throw std::runtime_error(
        std::string(argument) + " must be a warp multiple in [32, 1024]");
  }
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

void require_f32_matrix4(const NDArray& value,
                         const std::string& operation,
                         const char* argument) {
  if (value.dtype != DType::Float32 || value.ndim() != 2 ||
      value.shape[0] != 4 || value.shape[1] != 4) {
    throw std::runtime_error(operation + ": " + argument +
                             " must be float32 with shape (4, 4)");
  }
}

// NumPy/OpenBLAS evaluates each four-term float32 row dot used by the frozen
// ColdCast path as one rounded multiply followed by three ordered FFMA steps.
// Keep those instruction boundaries explicit; a compiler-selected dot or a
// reassociated expression changes projected coordinates at one ULP.
__device__ __forceinline__ float dot_world_to_camera(
    const float* __restrict__ position,
    const float* __restrict__ matrix, int output_column) {
  const float* row = matrix + output_column * 4;
  float value = __fmul_rn(position[0], row[0]);
  value = __fmaf_rn(position[1], row[1], value);
  value = __fmaf_rn(position[2], row[2], value);
  value = __fmaf_rn(position[3], row[3], value);
  return value;
}

__device__ __forceinline__ float dot_camera_to_projection(
    const float camera[4], const float* __restrict__ matrix,
    int output_column) {
  float value = __fmul_rn(camera[0], matrix[output_column]);
  value = __fmaf_rn(camera[1], matrix[4 + output_column], value);
  value = __fmaf_rn(camera[2], matrix[8 + output_column], value);
  value = __fmaf_rn(camera[3], matrix[12 + output_column], value);
  return value;
}

__device__ __forceinline__ float clip_unit(float value) {
  if (value < -1.0f) return -1.0f;
  if (value > 1.0f) return 1.0f;
  return value;
}

__global__ void bake_back_project_kernel(
    const float* __restrict__ atlas_positions_h, int64_t sample_count,
    const float* __restrict__ view, int side, int channels,
    const float* __restrict__ view_depth,
    const uint8_t* __restrict__ view_reliable,
    const float* __restrict__ view_cosine,
    const float* __restrict__ world_to_camera,
    const float* __restrict__ image_projection, float depth_threshold,
    uint8_t* __restrict__ output_valid,
    float* __restrict__ output_colors,
    float* __restrict__ output_cosine,
    float* __restrict__ output_depth_delta) {
  const int64_t sample =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (sample >= sample_count) return;

  const float* position = atlas_positions_h + sample * 4;
  float camera[4];
  camera[0] = dot_world_to_camera(position, world_to_camera, 0);
  camera[1] = dot_world_to_camera(position, world_to_camera, 1);
  camera[2] = dot_world_to_camera(position, world_to_camera, 2);
  camera[3] = dot_world_to_camera(position, world_to_camera, 3);
  const float projected_x =
      dot_camera_to_projection(camera, image_projection, 0);
  const float projected_y =
      dot_camera_to_projection(camera, image_projection, 1);
  const float projected_z =
      dot_camera_to_projection(camera, image_projection, 2);

  // Frozen ColdCast convention (square views):
  //   scaled_x uses height, scaled_y uses width, and flat index is y*height+x.
  // The public validator requires height == width rather than silently
  // changing those semantics for rectangular inputs.
  const float unit_x = __fadd_rn(
      __fmul_rn(clip_unit(projected_x), 0.5f), 0.5f);
  const float unit_y = __fadd_rn(
      __fmul_rn(clip_unit(projected_y), 0.5f), 0.5f);
  const float scaled_x = __fmul_rn(unit_x, static_cast<float>(side));
  const float scaled_y = __fmul_rn(unit_y, static_cast<float>(side));
  int x0 = __float2int_rz(scaled_x);
  int y0 = __float2int_rz(scaled_y);
  if (x0 < 0) x0 = 0;
  if (x0 >= side) x0 = side - 1;
  if (y0 < 0) y0 = 0;
  if (y0 >= side) y0 = side - 1;
  const int64_t base = static_cast<int64_t>(y0) * side + x0;

  const float depth_delta =
      fabsf(__fsub_rn(projected_z, view_depth[base]));
  output_depth_delta[sample] = depth_delta;
  const bool inner = projected_x <= 1.0f && projected_x >= -1.0f &&
                     projected_y <= 1.0f && projected_y >= -1.0f;
  const float cosine = view_cosine[base];
  const bool valid = inner && depth_delta < depth_threshold &&
                     view_reliable[base] != 0 && cosine > 0.0f;
  output_valid[sample] = valid ? uint8_t{1} : uint8_t{0};
  output_cosine[sample] = valid ? cosine : 0.0f;

  float* colors = output_colors + sample * channels;
  if (!valid) {
    for (int channel = 0; channel < channels; ++channel) {
      colors[channel] = 0.0f;
    }
    return;
  }

  int xr = x0 + 1;
  int yr = y0 + 1;
  if (xr >= side) xr = side - 1;
  if (yr >= side) yr = side - 1;
  const int64_t lower_right = static_cast<int64_t>(y0) * side + xr;
  const int64_t upper_left = static_cast<int64_t>(yr) * side + x0;
  const int64_t upper_right = static_cast<int64_t>(yr) * side + xr;
  const float wx = __fsub_rn(scaled_x, static_cast<float>(x0));
  const float wy = __fsub_rn(scaled_y, static_cast<float>(y0));
  const float one_minus_wx = __fsub_rn(1.0f, wx);
  const float one_minus_wy = __fsub_rn(1.0f, wy);
  for (int channel = 0; channel < channels; ++channel) {
    const float base_value = view[base * channels + channel];
    const float lower_right_value =
        view[lower_right * channels + channel];
    const float upper_left_value =
        view[upper_left * channels + channel];
    const float upper_right_value =
        view[upper_right * channels + channel];
    const float first_row = __fadd_rn(
        __fmul_rn(base_value, one_minus_wx),
        __fmul_rn(lower_right_value, wx));
    const float second_row = __fadd_rn(
        __fmul_rn(upper_left_value, one_minus_wx),
        __fmul_rn(upper_right_value, wx));
    colors[channel] = __fadd_rn(
        __fmul_rn(first_row, one_minus_wy),
        __fmul_rn(second_row, wy));
  }
}

__device__ __forceinline__ float cosine_weight(float cosine,
                                                float view_weight) {
  // Deliberately call the same native powf used by TensorCUDA's public pow.
  // G-TEXEL measures this exact instruction boundary against NumPy's host
  // powf and stops the bit-exact claim here if the device libm differs.
  const float powered = powf(cosine, 4.0f);
  return __fmul_rn(view_weight, powered);
}

template <int Channels>
__global__ void bake_cosine_blend_fixed_kernel(
    const float* __restrict__ view_colors,
    const float* __restrict__ view_cosine,
    const uint8_t* __restrict__ view_valid,
    const float* __restrict__ view_weights,
    const uint8_t* __restrict__ view_enabled, int views,
    int64_t sample_count, float* __restrict__ output_texture,
    float* __restrict__ output_trust,
    uint8_t* __restrict__ output_valid) {
  const int64_t sample =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (sample >= sample_count) return;

  float accumulated[Channels] = {};
  float trust = 0.0f;
  for (int view_index = 0; view_index < views; ++view_index) {
    if (view_enabled[view_index] == 0) continue;
    const int64_t offset = static_cast<int64_t>(view_index) * sample_count +
                           sample;
    if (view_valid[offset] == 0) continue;
    const float weighted =
        cosine_weight(view_cosine[offset], view_weights[view_index]);
    if (weighted > 0.0f) {
      const int64_t color_offset = offset * Channels;
#pragma unroll
      for (int channel = 0; channel < Channels; ++channel) {
        const float contribution =
            __fmul_rn(view_colors[color_offset + channel], weighted);
        accumulated[channel] =
            __fadd_rn(accumulated[channel], contribution);
      }
      trust = __fadd_rn(trust, weighted);
    }
  }
  output_trust[sample] = trust;
  output_valid[sample] = trust > 1.0e-8f ? uint8_t{1} : uint8_t{0};
  const float denominator = trust > 1.0e-8f ? trust : 1.0e-8f;

  for (int channel = 0; channel < Channels; ++channel) {
    output_texture[sample * Channels + channel] =
        __fdiv_rn(accumulated[channel], denominator);
  }
}

// Arbitrary-channel fallback.  The paint pipeline uses one or three channels,
// which dispatch to the fixed-register kernels above and compute powf once per
// sample/view.  Keeping this fallback avoids an artificial public channel cap;
// it recomputes the deterministic weight per channel rather than allocating a
// view-sized intermediate.
__global__ void bake_cosine_blend_dynamic_kernel(
    const float* __restrict__ view_colors,
    const float* __restrict__ view_cosine,
    const uint8_t* __restrict__ view_valid,
    const float* __restrict__ view_weights,
    const uint8_t* __restrict__ view_enabled, int views,
    int64_t sample_count, int channels, float* __restrict__ output_texture,
    float* __restrict__ output_trust,
    uint8_t* __restrict__ output_valid) {
  const int64_t sample =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (sample >= sample_count) return;

  float trust = 0.0f;
  for (int view_index = 0; view_index < views; ++view_index) {
    if (view_enabled[view_index] == 0) continue;
    const int64_t offset = static_cast<int64_t>(view_index) * sample_count +
                           sample;
    if (view_valid[offset] == 0) continue;
    const float weighted =
        cosine_weight(view_cosine[offset], view_weights[view_index]);
    if (weighted > 0.0f) trust = __fadd_rn(trust, weighted);
  }
  output_trust[sample] = trust;
  output_valid[sample] = trust > 1.0e-8f ? uint8_t{1} : uint8_t{0};
  const float denominator = trust > 1.0e-8f ? trust : 1.0e-8f;
  for (int channel = 0; channel < channels; ++channel) {
    float accumulated = 0.0f;
    for (int view_index = 0; view_index < views; ++view_index) {
      if (view_enabled[view_index] == 0) continue;
      const int64_t offset = static_cast<int64_t>(view_index) * sample_count +
                             sample;
      if (view_valid[offset] == 0) continue;
      const float weighted =
          cosine_weight(view_cosine[offset], view_weights[view_index]);
      if (weighted > 0.0f) {
        const int64_t color_offset = offset * channels + channel;
        const float contribution =
            __fmul_rn(view_colors[color_offset], weighted);
        accumulated = __fadd_rn(accumulated, contribution);
      }
    }
    output_texture[sample * channels + channel] =
        __fdiv_rn(accumulated, denominator);
  }
}

}  // namespace

std::tuple<NDArray, NDArray, NDArray, NDArray> bake_back_project(
    const NDArray& atlas_positions_h, const NDArray& view,
    const NDArray& view_depth, const NDArray& view_reliable,
    const NDArray& view_cosine, const NDArray& world_to_camera,
    const NDArray& image_projection, float depth_threshold, int threads) {
  const std::string operation = "bake_back_project";
  if (atlas_positions_h.dtype != DType::Float32 ||
      atlas_positions_h.ndim() != 2 || atlas_positions_h.shape[1] != 4) {
    throw std::runtime_error(
        operation +
        ": atlas_positions_h must be float32 with shape (N, 4)");
  }
  if (!atlas_positions_h.device.is_cuda()) {
    throw std::runtime_error(operation + ": CUDA tensors required");
  }
  if (view.dtype != DType::Float32 || view.ndim() != 3 ||
      view.shape[0] <= 0 || view.shape[1] <= 0 || view.shape[2] <= 0) {
    throw std::runtime_error(
        operation + ": view must be non-empty float32 with shape (H, W, C)");
  }
  if (view.shape[0] != view.shape[1]) {
    throw std::runtime_error(
        operation + ": the frozen paint convention requires a square view");
  }
  if (view.shape[0] > std::numeric_limits<int>::max() ||
      view.shape[2] > std::numeric_limits<int>::max()) {
    throw std::runtime_error(operation + ": view dimensions exceed int range");
  }
  const Shape image_shape{view.shape[0], view.shape[1]};
  if (view_depth.dtype != DType::Float32 ||
      view_depth.shape != image_shape) {
    throw std::runtime_error(
        operation + ": view_depth must be float32 with shape (H, W)");
  }
  if (view_reliable.dtype != DType::Uint8 ||
      view_reliable.shape != image_shape) {
    throw std::runtime_error(
        operation + ": view_reliable must be uint8 with shape (H, W)");
  }
  if (view_cosine.dtype != DType::Float32 ||
      view_cosine.shape != image_shape) {
    throw std::runtime_error(
        operation + ": view_cosine must be float32 with shape (H, W)");
  }
  require_f32_matrix4(world_to_camera, operation, "world_to_camera");
  require_f32_matrix4(image_projection, operation, "image_projection");
  require_same_cuda_device(atlas_positions_h, view, operation, "view");
  require_same_cuda_device(atlas_positions_h, view_depth, operation,
                           "view_depth");
  require_same_cuda_device(atlas_positions_h, view_reliable, operation,
                           "view_reliable");
  require_same_cuda_device(atlas_positions_h, view_cosine, operation,
                           "view_cosine");
  require_same_cuda_device(atlas_positions_h, world_to_camera, operation,
                           "world_to_camera");
  require_same_cuda_device(atlas_positions_h, image_projection, operation,
                           "image_projection");
  if (!std::isfinite(depth_threshold) || depth_threshold <= 0.0f) {
    throw std::runtime_error(
        operation + ": depth_threshold must be finite and positive");
  }
  validate_launch_threads(threads, "bake_back_project: threads");

  const int64_t sample_count = atlas_positions_h.shape[0];
  const int channels = static_cast<int>(view.shape[2]);
  NDArray valid({sample_count}, DType::Uint8, atlas_positions_h.device);
  NDArray colors({sample_count, channels}, DType::Float32,
                 atlas_positions_h.device);
  NDArray cosine({sample_count}, DType::Float32,
                 atlas_positions_h.device);
  NDArray depth_delta({sample_count}, DType::Float32,
                      atlas_positions_h.device);
  if (sample_count > 0) {
    const int64_t blocks64 = (sample_count + threads - 1) / threads;
    if (blocks64 > std::numeric_limits<unsigned>::max()) {
      throw std::runtime_error(operation + ": launch grid exceeds CUDA range");
    }
    bake_back_project_kernel<<<static_cast<unsigned>(blocks64), threads>>>(
        static_cast<const float*>(atlas_positions_h.data_ptr()), sample_count,
        static_cast<const float*>(view.data_ptr()),
        static_cast<int>(view.shape[0]), channels,
        static_cast<const float*>(view_depth.data_ptr()),
        static_cast<const uint8_t*>(view_reliable.data_ptr()),
        static_cast<const float*>(view_cosine.data_ptr()),
        static_cast<const float*>(world_to_camera.data_ptr()),
        static_cast<const float*>(image_projection.data_ptr()),
        depth_threshold, static_cast<uint8_t*>(valid.data_ptr()),
        static_cast<float*>(colors.data_ptr()),
        static_cast<float*>(cosine.data_ptr()),
        static_cast<float*>(depth_delta.data_ptr()));
    cuda_check_last("bake_back_project");
  }
  return std::make_tuple(valid, colors, cosine, depth_delta);
}

std::tuple<NDArray, NDArray, NDArray> bake_cosine_blend(
    const NDArray& view_colors, const NDArray& view_cosine,
    const NDArray& view_valid, const NDArray& view_weights,
    const NDArray& view_enabled, int threads) {
  const std::string operation = "bake_cosine_blend";
  if (view_colors.dtype != DType::Float32 || view_colors.ndim() != 3 ||
      view_colors.shape[0] <= 0 || view_colors.shape[2] <= 0) {
    throw std::runtime_error(
        operation +
        ": view_colors must be float32 with non-empty shape (V, N, C)");
  }
  if (!view_colors.device.is_cuda()) {
    throw std::runtime_error(operation + ": CUDA tensors required");
  }
  if (view_colors.shape[0] > std::numeric_limits<int>::max() ||
      view_colors.shape[2] > std::numeric_limits<int>::max()) {
    throw std::runtime_error(operation + ": dimensions exceed int range");
  }
  const int64_t views = view_colors.shape[0];
  const int64_t samples = view_colors.shape[1];
  const Shape sample_shape{views, samples};
  if (view_cosine.dtype != DType::Float32 ||
      view_cosine.shape != sample_shape) {
    throw std::runtime_error(
        operation + ": view_cosine must be float32 with shape (V, N)");
  }
  if (view_valid.dtype != DType::Uint8 ||
      view_valid.shape != sample_shape) {
    throw std::runtime_error(
        operation + ": view_valid must be uint8 with shape (V, N)");
  }
  if (view_weights.dtype != DType::Float32 || view_weights.ndim() != 1 ||
      view_weights.shape[0] != views) {
    throw std::runtime_error(
        operation + ": view_weights must be float32 with shape (V,)");
  }
  if (view_enabled.dtype != DType::Uint8 || view_enabled.ndim() != 1 ||
      view_enabled.shape[0] != views) {
    throw std::runtime_error(
        operation + ": view_enabled must be uint8 with shape (V,)");
  }
  require_same_cuda_device(view_colors, view_cosine, operation,
                           "view_cosine");
  require_same_cuda_device(view_colors, view_valid, operation, "view_valid");
  require_same_cuda_device(view_colors, view_weights, operation,
                           "view_weights");
  require_same_cuda_device(view_colors, view_enabled, operation,
                           "view_enabled");
  validate_launch_threads(threads, "bake_cosine_blend: threads");

  const int channels = static_cast<int>(view_colors.shape[2]);
  NDArray texture({samples, channels}, DType::Float32, view_colors.device);
  NDArray trust({samples}, DType::Float32, view_colors.device);
  NDArray valid({samples}, DType::Uint8, view_colors.device);
  if (samples > 0) {
    const int64_t blocks64 = (samples + threads - 1) / threads;
    if (blocks64 > std::numeric_limits<unsigned>::max()) {
      throw std::runtime_error(operation + ": launch grid exceeds CUDA range");
    }
    const float* colors_ptr =
        static_cast<const float*>(view_colors.data_ptr());
    const float* cosine_ptr =
        static_cast<const float*>(view_cosine.data_ptr());
    const uint8_t* valid_ptr =
        static_cast<const uint8_t*>(view_valid.data_ptr());
    const float* weights_ptr =
        static_cast<const float*>(view_weights.data_ptr());
    const uint8_t* enabled_ptr =
        static_cast<const uint8_t*>(view_enabled.data_ptr());
    float* texture_ptr = static_cast<float*>(texture.data_ptr());
    float* trust_ptr = static_cast<float*>(trust.data_ptr());
    uint8_t* output_valid_ptr = static_cast<uint8_t*>(valid.data_ptr());
    const unsigned blocks = static_cast<unsigned>(blocks64);
    switch (channels) {
      case 1:
        bake_cosine_blend_fixed_kernel<1><<<blocks, threads>>>(
            colors_ptr, cosine_ptr, valid_ptr, weights_ptr, enabled_ptr,
            static_cast<int>(views), samples, texture_ptr, trust_ptr,
            output_valid_ptr);
        break;
      case 2:
        bake_cosine_blend_fixed_kernel<2><<<blocks, threads>>>(
            colors_ptr, cosine_ptr, valid_ptr, weights_ptr, enabled_ptr,
            static_cast<int>(views), samples, texture_ptr, trust_ptr,
            output_valid_ptr);
        break;
      case 3:
        bake_cosine_blend_fixed_kernel<3><<<blocks, threads>>>(
            colors_ptr, cosine_ptr, valid_ptr, weights_ptr, enabled_ptr,
            static_cast<int>(views), samples, texture_ptr, trust_ptr,
            output_valid_ptr);
        break;
      case 4:
        bake_cosine_blend_fixed_kernel<4><<<blocks, threads>>>(
            colors_ptr, cosine_ptr, valid_ptr, weights_ptr, enabled_ptr,
            static_cast<int>(views), samples, texture_ptr, trust_ptr,
            output_valid_ptr);
        break;
      default:
        bake_cosine_blend_dynamic_kernel<<<blocks, threads>>>(
            colors_ptr, cosine_ptr, valid_ptr, weights_ptr, enabled_ptr,
            static_cast<int>(views), samples, channels, texture_ptr,
            trust_ptr, output_valid_ptr);
        break;
    }
    cuda_check_last("bake_cosine_blend");
  }
  return std::make_tuple(texture, trust, valid);
}

}  // namespace tc
