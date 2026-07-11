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
from pathlib import Path
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


# WO-8A smooth reference -----------------------------------------------------
# This is deliberately independent from the CUDA implementation.  It walks
# the shifted lattice of voxel-center samples, not the blocky dda_raycast
# helper, and mirrors the registered field / bracket / refinement contract.
_SMOOTH_ISO = np.float32(0.5)
_SMOOTH_STEP = np.float32(0.5)
_SMOOTH_REFINEMENTS = 6


def _material_at(grid, x, y, z):
    """Vectorized zero-padded uint8 material lookup."""
    material = np.zeros(x.size, dtype=np.uint8)
    valid = (
        (x >= 0)
        & (x < grid.shape[0])
        & (y >= 0)
        & (y < grid.shape[1])
        & (z >= 0)
        & (z < grid.shape[2])
    )
    material[valid] = grid[x[valid], y[valid], z[valid]]
    return material


def _trilinear_density(grid, positions):
    """Occupancy interpolated from voxel centers; outside-grid samples are air."""
    positions = np.asarray(positions, dtype=np.float32)
    shifted = positions - np.float32(0.5)
    base = np.floor(shifted).astype(np.int64)
    fraction = (shifted - base).astype(np.float32)
    density = np.zeros(positions.shape[0], dtype=np.float32)
    for dz in (0, 1):
        wz = (np.float32(1.0) - fraction[:, 2]) if dz == 0 else fraction[:, 2]
        for dy in (0, 1):
            wy = (np.float32(1.0) - fraction[:, 1]) if dy == 0 else fraction[:, 1]
            for dx in (0, 1):
                wx = (
                    (np.float32(1.0) - fraction[:, 0])
                    if dx == 0
                    else fraction[:, 0]
                )
                occupancy = _occupied_at(
                    grid, base[:, 0] + dx, base[:, 1] + dy, base[:, 2] + dz
                )
                density += wx * wy * wz * occupancy
    return density


def _mixed_density_cells(grid, cells):
    """Whether each shifted trilinear cell's 2x2x2 corners contain both states."""
    has_solid = np.zeros(cells.shape[0], dtype=np.bool_)
    has_air = np.zeros(cells.shape[0], dtype=np.bool_)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                occupied = _material_at(
                    grid,
                    cells[:, 0] + dx,
                    cells[:, 1] + dy,
                    cells[:, 2] + dz,
                ) != 0
                has_solid |= occupied
                has_air |= ~occupied
    return has_solid & has_air


def _refine_smooth_crossings(
    grid, origins, directions, lower_t, lower_density, upper_t, upper_density
):
    """Six bracketed secant/bisection refinements of rho(t) == 0.5."""
    lower_t = lower_t.astype(np.float32, copy=True)
    lower_density = lower_density.astype(np.float32, copy=True)
    upper_t = upper_t.astype(np.float32, copy=True)
    upper_density = upper_density.astype(np.float32, copy=True)
    for _ in range(_SMOOTH_REFINEMENTS):
        span = upper_t - lower_t
        candidate_t = lower_t + np.float32(0.5) * span
        denominator = upper_density - lower_density
        secant_t = lower_t + (_SMOOTH_ISO - lower_density) * span / np.where(
            np.abs(denominator) > np.float32(1.0e-7), denominator, np.float32(1.0)
        )
        guard = np.float32(0.05) * span
        use_secant = (
            (np.abs(denominator) > np.float32(1.0e-7))
            & (secant_t > lower_t + guard)
            & (secant_t < upper_t - guard)
        )
        candidate_t = np.where(use_secant, secant_t, candidate_t).astype(np.float32)
        candidate_position = origins + directions * candidate_t[:, None]
        candidate_density = _trilinear_density(grid, candidate_position)
        above = candidate_density >= _SMOOTH_ISO
        upper_t[above] = candidate_t[above]
        upper_density[above] = candidate_density[above]
        lower_t[~above] = candidate_t[~above]
        lower_density[~above] = candidate_density[~above]
    return upper_t


def _trace_mixed_density_segments(grid, origins, directions, segment_start, segment_end):
    """Sample each mixed cell at <=0.5 voxel increments, then refine first entry."""
    count = segment_start.size
    hit = np.zeros(count, dtype=np.bool_)
    hit_distance = np.full(count, np.inf, dtype=np.float32)
    previous_t = segment_start.astype(np.float32, copy=True)
    previous_density = _trilinear_density(
        grid, origins + directions * previous_t[:, None]
    )
    already_inside = previous_density >= _SMOOTH_ISO
    hit[already_inside] = True
    hit_distance[already_inside] = previous_t[already_inside]

    # A unit shifted cell has a longest normalized-ray chord of sqrt(3), so
    # four fixed 0.5-voxel samples reach its end.
    for _ in range(4):
        current_t = np.minimum(previous_t + _SMOOTH_STEP, segment_end).astype(
            np.float32
        )
        sample_mask = (~hit) & (current_t > previous_t)
        if not np.any(sample_mask):
            continue
        rows = np.flatnonzero(sample_mask)
        current_density = _trilinear_density(
            grid, origins[rows] + directions[rows] * current_t[rows, None]
        )
        crossed = current_density >= _SMOOTH_ISO
        if np.any(crossed):
            crossed_rows = rows[crossed]
            hit[crossed_rows] = True
            hit_distance[crossed_rows] = _refine_smooth_crossings(
                grid,
                origins[crossed_rows],
                directions[crossed_rows],
                previous_t[crossed_rows],
                previous_density[crossed_rows],
                current_t[crossed_rows],
                current_density[crossed],
            )
        continuing_rows = rows[~crossed]
        previous_t[continuing_rows] = current_t[continuing_rows]
        previous_density[continuing_rows] = current_density[~crossed]
    return hit, hit_distance


def _nearest_solid_voxels(grid, positions):
    """Exact nearest solid center for crossed points, with lower xyz tie-breaks."""
    positions = np.asarray(positions, dtype=np.float32)
    count = positions.shape[0]
    base = np.floor(positions - np.float32(0.5)).astype(np.int64)
    voxel = np.full((count, 3), -1, dtype=np.int64)
    material = np.zeros(count, dtype=np.uint8)
    best_distance_sq = np.full(count, np.inf, dtype=np.float32)
    for dx in (-1, 0, 1, 2):
        x = base[:, 0] + dx
        for dy in (-1, 0, 1, 2):
            y = base[:, 1] + dy
            for dz in (-1, 0, 1, 2):
                z = base[:, 2] + dz
                candidate = _material_at(grid, x, y, z)
                delta_x = positions[:, 0] - (x.astype(np.float32) + np.float32(0.5))
                delta_y = positions[:, 1] - (y.astype(np.float32) + np.float32(0.5))
                delta_z = positions[:, 2] - (z.astype(np.float32) + np.float32(0.5))
                distance_sq = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z
                better = (candidate != 0) & (distance_sq < best_distance_sq)
                best_distance_sq[better] = distance_sq[better]
                material[better] = candidate[better]
                voxel[better, 0] = x[better]
                voxel[better, 1] = y[better]
                voxel[better, 2] = z[better]
    return material, voxel


def _reference_smooth_raycast(grid, origins, directions, max_steps):
    """Independent shifted-cell DDA plus mixed-band smooth crossing search."""
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
    start_t = np.maximum(t_enter, np.float32(0.0)).astype(np.float32)
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
    cell = np.floor(start_position - np.float32(0.5)).astype(np.int64)
    in_cells = np.all(
        (cell >= -1) & (cell < integer_dims[None, :]), axis=1
    )
    eligible = valid & in_cells
    entry_axis = np.argmax(near, axis=1).astype(np.int64)
    face_axis[eligible & ~origin_inside] = entry_axis[eligible & ~origin_inside]
    entered = eligible & ~origin_inside
    face_sign[entered] = -step[entered, entry_axis[entered]]

    exact_start = (origins + directions * start_t[:, None]).astype(np.float32)
    start_density = np.zeros(count, dtype=np.float32)
    eligible_rows = np.flatnonzero(eligible)
    if eligible_rows.size:
        start_density[eligible_rows] = _trilinear_density(
            grid, exact_start[eligible_rows]
        )
    initial_hit = eligible & (start_density >= _SMOOTH_ISO)
    hit[initial_hit] = True
    distance[initial_hit] = start_t[initial_hit]

    shifted_origin = origins - np.float32(0.5)
    delta_t = np.full((count, 3), np.inf, dtype=np.float32)
    next_t = np.full((count, 3), np.inf, dtype=np.float32)
    for axis in range(3):
        moving = step[:, axis] != 0
        delta_t[moving, axis] = np.abs(
            np.float32(1.0) / directions[moving, axis]
        )
        boundary = np.where(
            step[:, axis] > 0, cell[:, axis] + 1, cell[:, axis]
        ).astype(np.float32)
        crossing = (boundary[moving] - shifted_origin[moving, axis]) / directions[
            moving, axis
        ]
        next_t[moving, axis] = np.maximum(crossing, start_t[moving])

    segment_start = start_t.copy()
    active = eligible & ~initial_hit
    for _ in range(int(max_steps) + 1):
        rows = np.flatnonzero(active)
        if rows.size == 0:
            break
        row_next = next_t[rows]
        crossed_axis = np.argmin(row_next, axis=1).astype(np.int64)
        local = np.arange(rows.size)
        crossing_t = row_next[local, crossed_axis]
        segment_end = np.minimum(crossing_t, t_exit[rows]).astype(np.float32)
        mixed = _mixed_density_cells(grid, cell[rows])
        if np.any(mixed):
            mixed_rows = rows[mixed]
            local_hit, local_distance = _trace_mixed_density_segments(
                grid,
                origins[mixed_rows],
                directions[mixed_rows],
                segment_start[mixed_rows],
                segment_end[mixed],
            )
            hit_rows = mixed_rows[local_hit]
            hit[hit_rows] = True
            distance[hit_rows] = local_distance[local_hit]

        # Every current cell was considered.  Advance only rays whose next
        # shifted-cell boundary lies before the original grid exit and which
        # did not find their first iso crossing in this cell.
        active[rows] = False
        progressing = (crossing_t < t_exit[rows]) & ~hit[rows]
        if not np.any(progressing):
            continue
        progress_rows = rows[progressing]
        progress_axis = crossed_axis[progressing]
        progress_crossing = crossing_t[progressing]
        cell[progress_rows, progress_axis] += step[progress_rows, progress_axis]
        next_t[progress_rows, progress_axis] = (
            progress_crossing + delta_t[progress_rows, progress_axis]
        )
        segment_start[progress_rows] = progress_crossing
        face_axis[progress_rows] = progress_axis
        face_sign[progress_rows] = -step[progress_rows, progress_axis]
        still_in_cells = np.all(
            (cell[progress_rows] >= -1)
            & (cell[progress_rows] < integer_dims[None, :]),
            axis=1,
        )
        active[progress_rows[still_in_cells]] = True

    hit_rows = np.flatnonzero(hit)
    if hit_rows.size:
        hit_position = origins[hit_rows] + directions[hit_rows] * distance[hit_rows, None]
        hit_material, hit_voxel = _nearest_solid_voxels(grid, hit_position)
        # A valid 0.5 crossing necessarily has a solid trilinear corner.
        assert np.all(hit_material != 0)
        material[hit_rows] = hit_material
        voxel_out[hit_rows] = hit_voxel
    return hit, material, voxel_out, face_axis, face_sign, distance


def _reference_smooth_normals(grid, raycast, points):
    hit, _, voxel, face_axis, face_sign, _ = raycast
    normal = np.zeros((hit.size, 3), dtype=np.float32)
    normal[:, 2] = np.float32(1.0)
    rows = np.flatnonzero(hit)
    if not rows.size:
        return normal
    positions = points[rows]
    outward_gradient = np.empty((rows.size, 3), dtype=np.float32)
    for axis in range(3):
        lower = positions.copy()
        upper = positions.copy()
        lower[:, axis] -= np.float32(0.5)
        upper[:, axis] += np.float32(0.5)
        outward_gradient[:, axis] = _trilinear_density(
            grid, lower
        ) - _trilinear_density(grid, upper)
    gradient_length = np.sqrt(np.sum(outward_gradient * outward_gradient, axis=1))
    defined = gradient_length >= np.float32(1.0e-6)
    normal_rows = normal[rows]
    normal_rows[defined] = outward_gradient[defined] / gradient_length[defined, None]

    fallback = ~defined
    if np.any(fallback):
        fallback_rows = rows[fallback]
        x, y, z = (voxel[fallback_rows, axis] for axis in range(3))
        blocky_gradient = np.stack(
            (
                _occupied_at(grid, x - 1, y, z) - _occupied_at(grid, x + 1, y, z),
                _occupied_at(grid, x, y - 1, z) - _occupied_at(grid, x, y + 1, z),
                _occupied_at(grid, x, y, z - 1) - _occupied_at(grid, x, y, z + 1),
            ),
            axis=1,
        ).astype(np.float32)
        blocky_length = np.sqrt(np.sum(blocky_gradient * blocky_gradient, axis=1))
        blocky_defined = blocky_length >= np.float32(1.0e-6)
        fallback_normal = np.zeros((fallback_rows.size, 3), dtype=np.float32)
        fallback_normal[:, 2] = np.float32(1.0)
        fallback_normal[blocky_defined] = (
            blocky_gradient[blocky_defined] / blocky_length[blocky_defined, None]
        )
        face_fallback = ~blocky_defined & (face_axis[fallback_rows] >= 0)
        face_rows = np.flatnonzero(face_fallback)
        fallback_normal[face_rows] = np.float32(0.0)
        fallback_normal[
            face_rows, face_axis[fallback_rows][face_rows]
        ] = face_sign[fallback_rows][face_rows].astype(np.float32)
        normal_rows[fallback] = fallback_normal
    normal[rows] = normal_rows
    return normal


def _reference_smooth_shade(grid, raycast, palette, light_direction, points):
    """Frozen WO-7B shade constants using the smooth-mode surface normal."""
    hit, material, voxel, _, _, distance = raycast
    rgb = np.zeros((hit.size, 3), dtype=np.uint8)
    depth = np.full(hit.size, np.float32(-1.0), dtype=np.float32)
    normal = _reference_smooth_normals(grid, raycast, points)
    indices = np.flatnonzero(hit)
    if not indices.size:
        return rgb, depth, normal

    x, y, z = (voxel[indices, axis] for axis in range(3))
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
        -np.sum(normal[indices] * light[None, :], axis=1),
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
    return rgb, depth, normal


def reference_smooth_terrain_render(
    grid, camera, light_direction, palette, max_steps=None, *, return_surface=False
):
    """Independent NumPy reference for terrain_render(..., surface_mode="smooth")."""
    grid = np.asarray(grid, dtype=np.uint8)
    palette = np.asarray(palette, dtype=np.uint8)
    if max_steps is None:
        max_steps = int(sum(grid.shape) + 3)
    origins, directions = _reference_camera_rays(camera)
    raycast = _reference_smooth_raycast(grid, origins, directions, int(max_steps))
    points = origins + directions * raycast[5][:, None]
    rgb, depth, normal = _reference_smooth_shade(
        grid, raycast, palette, light_direction, points
    )
    rgb = rgb.reshape(camera["height"], camera["width"], 3)
    depth = depth.reshape(camera["height"], camera["width"])
    if not return_surface:
        return rgb, depth
    return rgb, depth, {
        "hit": raycast[0].reshape(camera["height"], camera["width"]),
        "point": points.reshape(camera["height"], camera["width"], 3),
        "normal": normal.reshape(camera["height"], camera["width"], 3),
    }


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


def _call_cuda(
    grid,
    camera,
    palette=PALETTE,
    light=LIGHT_DIRECTION,
    consts=None,
    surface_mode=None,
):
    grid_device = tc.tensor(np.ascontiguousarray(grid), dtype="uint8")
    palette_device = tc.tensor(np.ascontiguousarray(palette), dtype="uint8")
    if surface_mode is None:
        # Preserve a literal no-surface-mode call for the default-path receipt.
        rgb, depth = tc.terrain_render(grid_device, camera, light, palette_device, consts)
    else:
        rgb, depth = tc.terrain_render(
            grid_device,
            camera,
            light,
            palette_device,
            consts,
            surface_mode=surface_mode,
        )
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


def _assert_smooth_registered_parity(
    actual_rgb, actual_depth, expected_rgb, expected_depth, label
):
    """WO-8A G2(c): strict smooth image/depth comparison against NumPy."""
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
        f"WO-8A G2 {label}: exact_channel_fraction={channel_exact_fraction:.9f} "
        f"max_abs_delta={max_channel_delta} depth_max_rel={depth_relative_error:.9g}"
    )
    assert channel_exact_fraction >= 0.999, label
    assert max_channel_delta <= 1, label
    assert depth_relative_error <= 1.0e-4, label


def _surface_points_from_depth(camera, depth):
    origins, directions = _reference_camera_rays(camera)
    points = origins + directions * depth.reshape(-1, 1)
    return points.reshape(depth.shape + (3,))


def _angle_degrees(lhs, rhs):
    dot = np.sum(lhs * rhs, axis=-1)
    return np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0)))


def _flat_plane_grid():
    grid = np.zeros((64, 64, 48), dtype=np.uint8)
    grid[:, :, :20] = 1
    return grid


def _flat_plane_camera():
    return {
        "position": np.array((32.0, 32.0, 44.0), dtype=np.float32),
        "look_at": np.array((32.0, 32.0, 0.0), dtype=np.float32),
        "world_up": np.array((0.0, 1.0, 0.0), dtype=np.float32),
        "vertical_fov_degrees": 38.0,
        "width": 160,
        "height": 120,
    }


def _sphere_grid(radius=96.0):
    # A large binary sphere makes the trilinear, center-sampled approximation
    # precise enough to measure the actual smooth-surface and normal rails.
    side = int(radius * 3)
    dims = (side, side, side)
    center = np.array(tuple(np.float32(value) * 0.5 for value in dims), dtype=np.float32)
    coordinates = np.indices(dims, dtype=np.float32)
    distance_sq = sum(
        (coordinates[axis] + np.float32(0.5) - center[axis]) ** 2
        for axis in range(3)
    )
    return np.where(distance_sq <= np.float32(radius * radius), 2, 0).astype(
        np.uint8
    ), center


def _sphere_camera(center, radius):
    return {
        "position": center
        + np.array((0.0, -(radius + 14.0), 4.0), dtype=np.float32),
        "look_at": center,
        "world_up": np.array((0.0, 0.0, 1.0), dtype=np.float32),
        "vertical_fov_degrees": 24.0,
        "width": 80,
        "height": 64,
    }


def test_terrain_render_blocky_surface_mode_regression():
    """G1: omitted mode and explicit blocky mode are byte-identical WO-7B."""
    _require_cuda()
    grid = _noise_terrain((128, 128, 64))
    camera = _camera_for_grid(grid.shape, width=320, height=240)
    default_rgb, default_depth = _call_cuda(grid, camera)
    blocky_rgb, blocky_depth = _call_cuda(grid, camera, surface_mode="blocky")
    np.testing.assert_array_equal(blocky_rgb, default_rgb)
    np.testing.assert_array_equal(blocky_depth, default_depth)


def test_terrain_render_smooth_flat_plane_correctness():
    """G2(a): smooth crossing is the z=20 plane and its normal is vertical."""
    _require_cuda()
    grid = _flat_plane_grid()
    camera = _flat_plane_camera()
    expected_rgb, expected_depth, surface = reference_smooth_terrain_render(
        grid, camera, LIGHT_DIRECTION, PALETTE, return_surface=True
    )
    actual_rgb, actual_depth = _call_cuda(grid, camera, surface_mode="smooth")
    _assert_smooth_registered_parity(
        actual_rgb, actual_depth, expected_rgb, expected_depth, "flat_plane"
    )
    hit = actual_depth >= 0.0
    assert np.all(hit)
    hit_points = _surface_points_from_depth(camera, actual_depth)[hit]
    assert float(np.max(np.abs(hit_points[:, 2] - np.float32(20.0)))) <= 0.1
    expected_normal = surface["normal"][hit]
    vertical = np.zeros_like(expected_normal)
    vertical[:, 2] = np.float32(1.0)
    assert float(np.max(_angle_degrees(expected_normal, vertical))) <= 2.0


def test_terrain_render_smooth_sphere_correctness():
    """G2(b): radius-96 sphere surface/normal gate plus full-shader parity."""
    _require_cuda()
    radius = np.float32(96.0)
    grid, center = _sphere_grid(float(radius))
    camera = _sphere_camera(center, float(radius))
    expected_rgb, expected_depth, surface = reference_smooth_terrain_render(
        grid, camera, LIGHT_DIRECTION, PALETTE, return_surface=True
    )
    actual_rgb, actual_depth = _call_cuda(grid, camera, surface_mode="smooth")
    _assert_smooth_registered_parity(
        actual_rgb, actual_depth, expected_rgb, expected_depth, "sphere_r96"
    )
    hit = actual_depth >= 0.0
    assert np.any(hit)
    points = _surface_points_from_depth(camera, actual_depth)[hit]
    radial = points - center[None, :]
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    radius_error_rms = float(
        np.sqrt(np.mean((np.linalg.norm(points - center[None, :], axis=1) - radius) ** 2))
    )
    normal_angle_rms = float(
        np.sqrt(np.mean(_angle_degrees(surface["normal"][hit], radial) ** 2))
    )
    print(
        "WO-8A G2 sphere_r96: "
        f"surface_distance_rms={radius_error_rms:.6f} "
        f"normal_angle_rms_deg={normal_angle_rms:.6f}"
    )
    assert radius_error_rms <= 0.15
    assert normal_angle_rms <= 5.0


def test_terrain_render_smooth_numpy_fixture_geometry():
    """CPU-side G2 geometry oracle: flat-plane and radius-96 sphere rails."""
    flat_grid = _flat_plane_grid()
    flat_camera = _flat_plane_camera()
    _, flat_depth, flat_surface = reference_smooth_terrain_render(
        flat_grid, flat_camera, LIGHT_DIRECTION, PALETTE, return_surface=True
    )
    flat_hit = flat_depth >= 0.0
    flat_points = _surface_points_from_depth(flat_camera, flat_depth)[flat_hit]
    flat_normal = flat_surface["normal"][flat_hit]
    vertical = np.zeros_like(flat_normal)
    vertical[:, 2] = np.float32(1.0)
    assert float(np.max(np.abs(flat_points[:, 2] - np.float32(20.0)))) <= 0.1
    assert float(np.max(_angle_degrees(flat_normal, vertical))) <= 2.0

    radius = np.float32(96.0)
    sphere_grid, center = _sphere_grid(float(radius))
    sphere_camera = _sphere_camera(center, float(radius))
    _, sphere_depth, sphere_surface = reference_smooth_terrain_render(
        sphere_grid, sphere_camera, LIGHT_DIRECTION, PALETTE, return_surface=True
    )
    sphere_hit = sphere_depth >= 0.0
    sphere_points = _surface_points_from_depth(sphere_camera, sphere_depth)[sphere_hit]
    radial = sphere_points - center[None, :]
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    radius_error_rms = float(
        np.sqrt(
            np.mean(
                (np.linalg.norm(sphere_points - center[None, :], axis=1) - radius)
                ** 2
            )
        )
    )
    normal_angle_rms = float(
        np.sqrt(
            np.mean(_angle_degrees(sphere_surface["normal"][sphere_hit], radial) ** 2)
        )
    )
    print(
        "WO-8A NumPy geometry: "
        f"flat_z_max={float(np.max(np.abs(flat_points[:, 2] - 20.0))):.6f} "
        f"sphere_distance_rms={radius_error_rms:.6f} "
        f"sphere_normal_rms_deg={normal_angle_rms:.6f}"
    )
    assert radius_error_rms <= 0.15
    assert normal_angle_rms <= 5.0


def test_terrain_render_smooth_numpy_parity_640x480():
    """G2(c): 640x480 kernel result against independent smooth NumPy reference."""
    _require_cuda()
    grid = _noise_terrain()
    camera = _camera_for_grid(grid.shape)
    expected_rgb, expected_depth = reference_smooth_terrain_render(
        grid, camera, LIGHT_DIRECTION, PALETTE
    )
    actual_rgb, actual_depth = _call_cuda(grid, camera, surface_mode="smooth")
    _assert_smooth_registered_parity(
        actual_rgb, actual_depth, expected_rgb, expected_depth, "noise_256x256x96"
    )


def _write_ppm(path, rgb):
    """Binary P6 receipt with no image-library dependency."""
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width, channels = rgb.shape
    assert channels == 3
    with path.open("wb") as handle:
        handle.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        handle.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())


def test_terrain_render_wo8a_ppm_pair():
    """G5: operator-facing same-camera noise-terrain blocky/smooth PPM pair.

    The independent references deliberately generate this receipt so it is
    available even on a build host whose CUDA driver cannot initialize.  The
    GPU-side G1/G2 parity tests establish that these are the kernel images.
    """
    grid = _noise_terrain((256, 256, 96))
    camera = _camera_for_grid(grid.shape)
    blocky_rgb, _ = reference_terrain_render(
        grid, camera, LIGHT_DIRECTION, PALETTE
    )
    smooth_rgb, _ = reference_smooth_terrain_render(
        grid, camera, LIGHT_DIRECTION, PALETTE
    )
    repo_root = Path(__file__).resolve().parents[2]
    _write_ppm(repo_root / "artifacts" / "wo8a_blocky.ppm", blocky_rgb)
    _write_ppm(repo_root / "artifacts" / "wo8a_smooth.ppm", smooth_rgb)


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


def terrain_render_timing_harness(
    grid, camera, *, frames=100, warmup=10, surface_mode="blocky"
):
    """CUDA-event stages for the requested blocky or smooth terrain mode."""
    _require_cuda()
    events = _CudaEvents()
    grid_device = tc.tensor(np.ascontiguousarray(grid), dtype="uint8")
    palette_device = tc.tensor(PALETTE, dtype="uint8")

    def render():
        return tc.terrain_render(
            grid_device,
            camera,
            LIGHT_DIRECTION,
            palette_device,
            None,
            surface_mode=surface_mode,
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


@pytest.mark.skipif(
    os.environ.get("TC_RUN_TERRAIN_TIMING") != "1",
    reason="set TC_RUN_TERRAIN_TIMING=1 to run the 100-frame WO-8A timing gate",
)
def test_terrain_render_smooth_stage_timing_100_frames():
    """G3: 512x512x192 smooth render receipt; red is printed, not hidden."""
    _require_cuda()
    dims = (512, 512, 192)
    grid = _noise_terrain(dims)
    camera = _camera_for_grid(dims)
    timings = terrain_render_timing_harness(
        grid, camera, frames=100, surface_mode="smooth"
    )
    print(
        "WO-8A G3 "
        f"grid={dims} frames={timings['frames']} "
        f"kernel_ms_mean={timings['kernel_ms_mean']:.4f} "
        f"d2h_ms_mean={timings['d2h_ms_mean']:.4f} "
        f"total_ms_mean={timings['total_ms_mean']:.4f} "
        f"target_8ms={'GREEN' if timings['total_ms_mean'] <= 8.0 else 'RED'}"
    )
    assert timings["kernel_ms_mean"] >= 0.0
    assert timings["d2h_ms_mean"] >= 0.0
    assert timings["total_ms_mean"] >= 0.0
