"""Registered WO-7B gates for the fused GPU-resident terrain renderer.

The reference below intentionally owns the full camera -> DDA -> shading path.
It does not call tensor_cuda's DDA operation, so it catches a traversal or
shading regression in the fused kernel instead of merely testing two paths that
share a bug.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc


WIDTH = 640
HEIGHT = 480
PALETTE = np.array(
    (
        (0, 0, 0),
        (126, 87, 50),
        (102, 108, 116),
        (201, 174, 108),
    ),
    dtype=np.uint8,
)
LIGHT_DIRECTION = np.array((0.46, -0.36, 0.81), dtype=np.float32)


def _require_cuda():
    """Skip only when the executing host cannot initialize the CUDA driver."""
    try:
        probe = tc.zeros((1,), dtype="uint8")
        probe.numpy()
    except Exception as exc:  # pragma: no cover - host dependent
        pytest.skip(f"CUDA unavailable for terrain_render gate: {exc}")


def _vec3(value, label):
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must be a finite three-vector")
    return vector


def _camera_terms(camera):
    """Standalone copy of Scorch's frozen camera convention."""
    position = _vec3(camera["position"], "position")
    look_at = _vec3(camera["look_at"], "look_at")
    world_up = _vec3(camera["world_up"], "world_up")
    fov = float(camera["vertical_fov_degrees"])
    width = int(camera["width"])
    height = int(camera["height"])
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if not 0.0 < fov < 180.0:
        raise ValueError("vertical_fov_degrees must lie between 0 and 180")

    forward = look_at - position
    forward_length = float(np.linalg.norm(forward))
    if forward_length == 0.0:
        raise ValueError("position and look_at must differ")
    forward /= np.float32(forward_length)
    right = np.cross(forward, world_up)
    right_length = float(np.linalg.norm(right))
    if right_length < 1.0e-7:
        raise ValueError("world_up must not be parallel to the view direction")
    right /= np.float32(right_length)
    camera_up = np.cross(right, forward).astype(np.float32)
    half_height = np.float32(np.tan(np.deg2rad(fov) * 0.5))
    half_width = half_height * np.float32(width / height)
    return position, forward, right, camera_up, half_width, half_height


def _reference_camera_rays(camera):
    """NumPy ray generation matching render/camera.py pixel-for-pixel."""
    position, forward, right, camera_up, half_width, half_height = _camera_terms(
        camera
    )
    width = int(camera["width"])
    height = int(camera["height"])
    screen_x = (
        (np.arange(width, dtype=np.float32) + np.float32(0.5))
        / np.float32(width)
        * np.float32(2.0)
        - np.float32(1.0)
    ) * half_width
    screen_y = (
        np.float32(1.0)
        - (np.arange(height, dtype=np.float32) + np.float32(0.5))
        / np.float32(height)
        * np.float32(2.0)
    ) * half_height
    directions = (
        forward[None, None, :]
        + screen_x[None, :, None] * right[None, None, :]
        + screen_y[:, None, None] * camera_up[None, None, :]
    )
    directions /= np.linalg.norm(directions, axis=2, keepdims=True)
    directions = np.ascontiguousarray(directions.reshape(-1, 3), dtype=np.float32)
    origins = np.broadcast_to(position, directions.shape).copy()
    return origins, directions


def _reference_raycast(grid, origins, directions, max_steps):
    """Vectorized float32 Amanatides-Woo reference with dda_raycast ties."""
    count = origins.shape[0]
    hit = np.zeros(count, dtype=np.bool_)
    material = np.zeros(count, dtype=np.uint8)
    voxel_out = np.full((count, 3), -1, dtype=np.int64)
    face_axis = np.full(count, -1, dtype=np.int64)
    face_sign = np.zeros(count, dtype=np.int64)
    distance = np.full(count, np.inf, dtype=np.float32)
    if count == 0:
        return hit, material, voxel_out, face_axis, face_sign, distance

    dims = np.asarray(grid.shape, dtype=np.float32)
    integer_dims = np.asarray(grid.shape, dtype=np.int64)
    near = np.full((count, 3), -np.inf, dtype=np.float32)
    far = np.full((count, 3), np.inf, dtype=np.float32)
    parallel_outside = np.zeros(count, dtype=np.bool_)
    for axis in range(3):
        component = directions[:, axis]
        positive = component > np.float32(0.0)
        negative = component < np.float32(0.0)
        stationary = ~(positive | negative)
        near[positive, axis] = (
            np.float32(0.0) - origins[positive, axis]
        ) / component[positive]
        far[positive, axis] = (
            dims[axis] - origins[positive, axis]
        ) / component[positive]
        near[negative, axis] = (
            dims[axis] - origins[negative, axis]
        ) / component[negative]
        far[negative, axis] = (
            np.float32(0.0) - origins[negative, axis]
        ) / component[negative]
        parallel_outside |= stationary & (
            (origins[:, axis] < np.float32(0.0))
            | (origins[:, axis] >= dims[axis])
        )

    t_enter = np.max(near, axis=1)
    t_exit = np.min(far, axis=1)
    start_t = np.maximum(t_enter, np.float32(0.0))
    origin_inside = np.all(
        (origins >= np.float32(0.0)) & (origins < dims[None, :]), axis=1
    )
    valid = (
        np.any(directions != np.float32(0.0), axis=1)
        & ~parallel_outside
        & (t_exit >= start_t)
        & (t_exit >= np.float32(0.0))
    )
    step = np.sign(directions).astype(np.int64)
    start_position = (origins + directions * start_t[:, None]).astype(np.float32)
    external = valid & ~origin_inside
    start_position[external] = np.nextafter(
        start_position[external], start_position[external] + directions[external]
    )
    voxel = np.floor(start_position).astype(np.int64)
    in_bounds = np.all(
        (voxel >= 0) & (voxel < integer_dims[None, :]), axis=1
    )
    eligible = valid & in_bounds

    initial_material = np.zeros(count, dtype=np.uint8)
    eligible_indices = np.flatnonzero(eligible)
    initial_material[eligible_indices] = grid[
        voxel[eligible_indices, 0],
        voxel[eligible_indices, 1],
        voxel[eligible_indices, 2],
    ]
    initial_hit = eligible & (initial_material != 0)
    initial_hit_indices = np.flatnonzero(initial_hit)
    hit[initial_hit_indices] = True
    material[initial_hit_indices] = initial_material[initial_hit_indices]
    voxel_out[initial_hit_indices] = voxel[initial_hit_indices]
    distance[initial_hit_indices] = start_t[initial_hit_indices]
    entry_axis = np.argmax(near, axis=1).astype(np.int64)
    entered_hit = initial_hit & ~origin_inside
    entered_indices = np.flatnonzero(entered_hit)
    face_axis[entered_indices] = entry_axis[entered_indices]
    face_sign[entered_indices] = -step[entered_indices, entry_axis[entered_indices]]

    delta_t = np.full((count, 3), np.inf, dtype=np.float32)
    next_t = np.full((count, 3), np.inf, dtype=np.float32)
    for axis in range(3):
        moving = step[:, axis] != 0
        delta_t[moving, axis] = np.abs(
            np.float32(1.0) / directions[moving, axis]
        )
        boundary = np.where(
            step[:, axis] > 0, voxel[:, axis] + 1, voxel[:, axis]
        ).astype(np.float32)
        crossing = (boundary[moving] - origins[moving, axis]) / directions[
            moving, axis
        ]
        next_t[moving, axis] = np.maximum(crossing, start_t[moving])

    active = eligible & ~initial_hit
    for _ in range(max_steps):
        active_indices = np.flatnonzero(active)
        if active_indices.size == 0:
            break
        active_next_t = next_t[active_indices]
        crossed_axis = np.argmin(active_next_t, axis=1).astype(np.int64)
        rows = np.arange(active_indices.size)
        crossed_t = active_next_t[rows, crossed_axis]
        voxel[active_indices, crossed_axis] += step[active_indices, crossed_axis]
        next_t[active_indices, crossed_axis] = (
            crossed_t + delta_t[active_indices, crossed_axis]
        )
        active[active_indices] = False
        crossed_voxel = voxel[active_indices]
        still_inside = (crossed_t <= t_exit[active_indices]) & np.all(
            (crossed_voxel >= 0) & (crossed_voxel < integer_dims[None, :]),
            axis=1,
        )
        candidate_indices = active_indices[still_inside]
        if candidate_indices.size == 0:
            continue
        candidate_axis = crossed_axis[still_inside]
        candidate_t = crossed_t[still_inside]
        candidate_material = grid[
            voxel[candidate_indices, 0],
            voxel[candidate_indices, 1],
            voxel[candidate_indices, 2],
        ]
        solid = candidate_material != 0
        solid_indices = candidate_indices[solid]
        if solid_indices.size:
            solid_axis = candidate_axis[solid]
            hit[solid_indices] = True
            material[solid_indices] = candidate_material[solid]
            voxel_out[solid_indices] = voxel[solid_indices]
            face_axis[solid_indices] = solid_axis
            face_sign[solid_indices] = -step[solid_indices, solid_axis]
            distance[solid_indices] = candidate_t[solid]
        active[candidate_indices[~solid]] = True

    return hit, material, voxel_out, face_axis, face_sign, distance


def _occupied_at(grid, x, y, z):
    """Vectorized, zero-padded occupancy lookup."""
    occupied = np.zeros(x.size, dtype=np.float32)
    valid = (
        (x >= 0)
        & (x < grid.shape[0])
        & (y >= 0)
        & (y < grid.shape[1])
        & (z >= 0)
        & (z < grid.shape[2])
    )
    occupied[valid] = (
        grid[x[valid], y[valid], z[valid]] != 0
    ).astype(np.float32)
    return occupied


def _splitmix64(value):
    value = value + np.uint64(0x9E3779B97F4A7C15)
    value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return value ^ (value >> np.uint64(31))


def _signed_hash_byte(hash_value, shift):
    byte = ((hash_value >> np.uint64(shift)) & np.uint64(0xFF)).astype(
        np.float32
    )
    return byte * np.float32(1.0 / 255.0) * np.float32(2.0) - np.float32(1.0)


def _reference_shade(grid, raycast, palette, light_direction):
    """Frozen WO-7B normal, AO, and deterministic palette shading."""
    hit, material, voxel, face_axis, face_sign, distance = raycast
    rgb = np.zeros((hit.size, 3), dtype=np.uint8)
    depth = np.full(hit.size, np.float32(-1.0), dtype=np.float32)
    indices = np.flatnonzero(hit)
    if not indices.size:
        return rgb, depth

    x, y, z = (voxel[indices, axis] for axis in range(3))
    # Outward central density gradient: occ(-axis) - occ(+axis), matching the
    # DDA entry-face fallback orientation.
    gradient = np.stack(
        (
            _occupied_at(grid, x - 1, y, z) - _occupied_at(grid, x + 1, y, z),
            _occupied_at(grid, x, y - 1, z) - _occupied_at(grid, x, y + 1, z),
            _occupied_at(grid, x, y, z - 1) - _occupied_at(grid, x, y, z + 1),
        ),
        axis=1,
    ).astype(np.float32)
    gradient_length = np.sqrt(np.sum(gradient * gradient, axis=1))
    normal = np.zeros((indices.size, 3), dtype=np.float32)
    normal[:, 2] = np.float32(1.0)
    defined = gradient_length >= np.float32(1.0e-6)
    normal[defined] = gradient[defined] / gradient_length[defined, None]
    fallback = ~defined & (face_axis[indices] >= 0)
    fallback_rows = np.flatnonzero(fallback)
    normal[fallback_rows] = np.float32(0.0)
    normal[
        fallback_rows, face_axis[indices][fallback_rows]
    ] = face_sign[indices][fallback_rows].astype(np.float32)

    occupied_neighbors = np.zeros(indices.size, dtype=np.float32)
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                occupied_neighbors += _occupied_at(grid, x + dx, y + dy, z + dz)
    ao = np.clip(
        np.float32(1.0)
        - np.float32(0.60) * occupied_neighbors * np.float32(1.0 / 26.0),
        np.float32(0.40),
        np.float32(1.0),
    )
    light = _vec3(light_direction, "light")
    light /= np.float32(np.linalg.norm(light))
    diffuse = np.clip(
        -np.sum(normal * light[None, :], axis=1),
        np.float32(0.35),
        np.float32(1.0),
    )

    palette_index = np.minimum(material[indices].astype(np.int64), palette.shape[0] - 1)
    base = palette[palette_index].astype(np.float32)
    hash_input = (
        x.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
        ^ y.astype(np.uint64) * np.uint64(0xBF58476D1CE4E5B9)
        ^ z.astype(np.uint64) * np.uint64(0x94D049BB133111EB)
    )
    hash_value = _splitmix64(hash_input)
    value_scale = np.float32(1.0) + _signed_hash_byte(hash_value, 56) * np.float32(0.08)
    mix_r = _signed_hash_byte(hash_value, 48) * np.float32(0.04)
    mix_g = _signed_hash_byte(hash_value, 40) * np.float32(0.04)
    mix_b = _signed_hash_byte(hash_value, 32) * np.float32(0.04)
    mixed = np.empty_like(base)
    mixed[:, 0] = base[:, 0] + mix_r * (base[:, 1] - base[:, 0])
    mixed[:, 1] = base[:, 1] + mix_g * (base[:, 2] - base[:, 1])
    mixed[:, 2] = base[:, 2] + mix_b * (base[:, 0] - base[:, 2])
    shaded = mixed * (diffuse * ao * value_scale)[:, None]
    rgb[indices] = np.clip(
        shaded + np.float32(0.5), np.float32(0.0), np.float32(255.0)
    ).astype(np.uint8)
    depth[indices] = distance[indices]
    return rgb, depth


def reference_terrain_render(grid, camera, light_direction, palette, max_steps=None):
    """Independent NumPy implementation of the full public terrain spec."""
    grid = np.asarray(grid, dtype=np.uint8)
    palette = np.asarray(palette, dtype=np.uint8)
    if max_steps is None:
        max_steps = int(sum(grid.shape) + 3)
    origins, directions = _reference_camera_rays(camera)
    raycast = _reference_raycast(grid, origins, directions, int(max_steps))
    rgb, depth = _reference_shade(grid, raycast, palette, light_direction)
    return rgb.reshape(camera["height"], camera["width"], 3), depth.reshape(
        camera["height"], camera["width"]
    )


def _camera_for_grid(grid_shape, *, width=WIDTH, height=HEIGHT):
    sx, sy, sz = grid_shape
    return {
        "position": np.array((sx * 0.50, -sy * 0.82, sz * 1.28), dtype=np.float32),
        "look_at": np.array((sx * 0.50, sy * 0.50, sz * 0.34), dtype=np.float32),
        "world_up": np.array((0.0, 0.0, 1.0), dtype=np.float32),
        "vertical_fov_degrees": 58.0,
        "width": width,
        "height": height,
    }


def _call_cuda(grid, camera, palette=PALETTE, light=LIGHT_DIRECTION, consts=None):
    grid_device = tc.tensor(np.ascontiguousarray(grid), dtype="uint8")
    palette_device = tc.tensor(np.ascontiguousarray(palette), dtype="uint8")
    rgb, depth = tc.terrain_render(grid_device, camera, light, palette_device, consts)
    return rgb.numpy(), depth.numpy()


def _assert_registered_parity(actual_rgb, actual_depth, expected_rgb, expected_depth, label):
    channel_exact_fraction = float(np.mean(actual_rgb == expected_rgb))
    max_channel_delta = int(
        np.max(np.abs(actual_rgb.astype(np.int16) - expected_rgb.astype(np.int16)))
    )
    np.testing.assert_array_equal(actual_depth == -1.0, expected_depth == -1.0)
    depth_relative_error = float(
        np.max(
            np.abs(actual_depth - expected_depth)
            / np.maximum(np.abs(expected_depth), np.float32(1.0))
        )
    )
    print(
        f"WO-7B G1 {label}: exact_channel_fraction={channel_exact_fraction:.9f} "
        f"max_abs_delta={max_channel_delta} depth_max_rel={depth_relative_error:.9g}"
    )
    assert channel_exact_fraction >= 0.999, label
    assert max_channel_delta <= 1, label
    assert depth_relative_error <= 1.0e-5, label


def _slab_grid():
    grid = np.zeros((48, 48, 32), dtype=np.uint8)
    grid[:, :, :10] = 1
    return grid


def _blocks_grid():
    grid = np.zeros((48, 48, 32), dtype=np.uint8)
    grid[:, :, :5] = 1
    grid[9:19, 14:27, 5:19] = 2
    grid[29:40, 8:17, 5:24] = 3
    grid[16:32, 31:42, 5:13] = 2
    return grid


def _terraced_grid():
    x, y, z = np.indices((48, 48, 32))
    heights = 7 + ((x * 5 + y * 3 + (x ^ y)) % 17)
    grid = np.where(z < heights, 1 + ((x // 7 + y // 9) % 3), 0).astype(np.uint8)
    return grid


def _noise_terrain(dims=(256, 256, 96)):
    """Deterministic noise-like mountains fixture, independent of Scorch sim."""
    sx, sy, sz = dims
    rng = np.random.default_rng(0x7B_256_96)
    x, y = np.indices((sx, sy), dtype=np.float32)
    noise = rng.integers(-8, 9, size=(sx, sy), dtype=np.int16).astype(np.float32)
    ridges = (
        np.sin(x * np.float32(0.083)) * np.float32(12.0)
        + np.cos(y * np.float32(0.071)) * np.float32(10.0)
        + np.sin((x + y) * np.float32(0.037)) * np.float32(8.0)
    )
    heights = np.clip(34.0 + ridges + noise, 4.0, float(sz - 3)).astype(np.int32)
    z = np.arange(sz, dtype=np.int32)[None, None, :]
    solid = z < heights[:, :, None]
    material = 1 + ((x.astype(np.int32) // 23 + y.astype(np.int32) // 19) % 3)
    return np.where(solid, material[:, :, None], 0).astype(np.uint8)


def test_terrain_render_output_contract_misses_and_frozen_constants():
    _require_cuda()
    grid = np.zeros((8, 8, 8), dtype=np.uint8)
    camera = _camera_for_grid(grid.shape, width=12, height=8)
    rgb, depth = _call_cuda(grid, camera)
    assert rgb.dtype == np.uint8
    assert depth.dtype == np.float32
    assert rgb.shape == (8, 12, 3)
    assert depth.shape == (8, 12)
    assert not np.any(rgb)
    np.testing.assert_array_equal(depth, np.full((8, 12), -1.0, dtype=np.float32))

    grid_device = tc.tensor(grid, dtype="uint8")
    palette_device = tc.tensor(PALETTE, dtype="uint8")
    with pytest.raises(ValueError, match="ambient_floor is frozen"):
        tc.terrain_render(
            grid_device,
            camera,
            LIGHT_DIRECTION,
            palette_device,
            {"ambient_floor": 0.30},
        )


def test_terrain_render_full_spec_numpy_parity_640x480():
    """G1: three synthetic grids plus 256x256x96 noise terrain at 640x480."""
    _require_cuda()
    cases = (
        ("slab", _slab_grid()),
        ("blocks", _blocks_grid()),
        ("terraces", _terraced_grid()),
        ("noise_256x256x96", _noise_terrain()),
    )
    for label, grid in cases:
        camera = _camera_for_grid(grid.shape)
        expected_rgb, expected_depth = reference_terrain_render(
            grid, camera, LIGHT_DIRECTION, PALETTE
        )
        actual_rgb, actual_depth = _call_cuda(grid, camera)
        _assert_registered_parity(
            actual_rgb, actual_depth, expected_rgb, expected_depth, label
        )


class _CudaEvents:
    """Minimal CUDA-runtime event wrapper; stdlib ctypes keeps tests framework-free."""

    def __init__(self):
        names = [ctypes.util.find_library("cudart"), "libcudart.so.12", "libcudart.so"]
        error = None
        self.lib = None
        for name in names:
            if not name:
                continue
            try:
                self.lib = ctypes.CDLL(name)
                break
            except OSError as exc:  # pragma: no cover - host dependent
                error = exc
        if self.lib is None:
            raise RuntimeError(f"cannot load CUDA runtime for event timings: {error}")
        self.lib.cudaEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cudaEventCreate.restype = ctypes.c_int
        self.lib.cudaEventDestroy.argtypes = [ctypes.c_void_p]
        self.lib.cudaEventDestroy.restype = ctypes.c_int
        self.lib.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.cudaEventRecord.restype = ctypes.c_int
        self.lib.cudaEventSynchronize.argtypes = [ctypes.c_void_p]
        self.lib.cudaEventSynchronize.restype = ctypes.c_int
        self.lib.cudaEventElapsedTime.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.cudaEventElapsedTime.restype = ctypes.c_int

    def _check(self, code, stage):
        if code != 0:
            raise RuntimeError(f"CUDA event {stage} failed with status {code}")

    def elapsed_ms(self, callback):
        start = ctypes.c_void_p()
        stop = ctypes.c_void_p()
        self._check(self.lib.cudaEventCreate(ctypes.byref(start)), "create(start)")
        self._check(self.lib.cudaEventCreate(ctypes.byref(stop)), "create(stop)")
        try:
            self._check(self.lib.cudaEventRecord(start, None), "record(start)")
            result = callback()
            self._check(self.lib.cudaEventRecord(stop, None), "record(stop)")
            self._check(self.lib.cudaEventSynchronize(stop), "synchronize(stop)")
            elapsed = ctypes.c_float()
            self._check(
                self.lib.cudaEventElapsedTime(ctypes.byref(elapsed), start, stop),
                "elapsed",
            )
            return float(elapsed.value), result
        finally:
            self.lib.cudaEventDestroy(start)
            self.lib.cudaEventDestroy(stop)


def terrain_render_timing_harness(grid, camera, *, frames=100, warmup=10):
    """G3 CUDA-event stage timing: kernel, D2H, and combined per-frame time."""
    _require_cuda()
    events = _CudaEvents()
    grid_device = tc.tensor(np.ascontiguousarray(grid), dtype="uint8")
    palette_device = tc.tensor(PALETTE, dtype="uint8")

    def render():
        return tc.terrain_render(
            grid_device, camera, LIGHT_DIRECTION, palette_device, None
        )

    for _ in range(warmup):
        render()
    tc.synchronize()

    kernel_ms = []
    d2h_ms = []
    total_ms = []
    for _ in range(frames):
        kernel_time, outputs = events.elapsed_ms(render)
        kernel_ms.append(kernel_time)
        d2h_time, _ = events.elapsed_ms(
            lambda: (outputs[0].numpy(), outputs[1].numpy())
        )
        d2h_ms.append(d2h_time)
        total_time, _ = events.elapsed_ms(
            lambda: tuple(output.numpy() for output in render())
        )
        total_ms.append(total_time)
    return {
        "frames": frames,
        "kernel_ms_mean": float(np.mean(kernel_ms)),
        "d2h_ms_mean": float(np.mean(d2h_ms)),
        "total_ms_mean": float(np.mean(total_ms)),
        "kernel_ms_p50": float(np.median(kernel_ms)),
        "d2h_ms_p50": float(np.median(d2h_ms)),
        "total_ms_p50": float(np.median(total_ms)),
    }


@pytest.mark.skipif(
    os.environ.get("TC_RUN_TERRAIN_TIMING") != "1",
    reason="set TC_RUN_TERRAIN_TIMING=1 to run the 100-frame WO-7B timing gate",
)
def test_terrain_render_stage_timing_100_frames():
    """G3 harness; a red <=16ms target is reported, not hidden as a test skip."""
    _require_cuda()
    for dims in ((256, 256, 96), (512, 512, 192)):
        grid = _noise_terrain(dims)
        camera = _camera_for_grid(dims)
        timings = terrain_render_timing_harness(grid, camera, frames=100)
        print(
            "WO-7B G3 "
            f"grid={dims} frames={timings['frames']} "
            f"kernel_ms_mean={timings['kernel_ms_mean']:.4f} "
            f"d2h_ms_mean={timings['d2h_ms_mean']:.4f} "
            f"total_ms_mean={timings['total_ms_mean']:.4f} "
            f"target_16ms={'GREEN' if timings['total_ms_mean'] <= 16.0 else 'RED'}"
        )
        assert timings["kernel_ms_mean"] >= 0.0
        assert timings["d2h_ms_mean"] >= 0.0
        assert timings["total_ms_mean"] >= 0.0
