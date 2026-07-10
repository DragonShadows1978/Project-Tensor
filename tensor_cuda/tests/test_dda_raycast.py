"""Correctness gates for the first-class CUDA voxel DDA operation."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc


FIELD_NAMES = (
    "hit",
    "material",
    "voxel",
    "face_axis",
    "face_sign",
    "distance",
)


def _normalized(vector):
    vector = np.asarray(vector, dtype=np.float32)
    return (vector / np.linalg.norm(vector, axis=-1, keepdims=True)).astype(
        np.float32
    )


def _call_cuda(grid, origins, directions, max_steps=None):
    if max_steps is None:
        max_steps = int(sum(grid.shape) + 3)
    outputs = tc.dda_raycast(
        tc.tensor(grid, dtype="uint8"),
        tc.tensor(origins, dtype="float32"),
        tc.tensor(directions, dtype="float32"),
        max_steps,
    )
    return outputs, tuple(output.numpy() for output in outputs)


def _reference_raycast(grid, origins, directions, max_steps):
    """Small scalar float32 marcher matching the Scorch DDA contract."""
    ray_count = origins.shape[0]
    hit = np.zeros(ray_count, dtype=np.uint8)
    material = np.zeros(ray_count, dtype=np.uint8)
    voxel_out = np.full((ray_count, 3), -1, dtype=np.int64)
    face_axis = np.full(ray_count, -1, dtype=np.int64)
    face_sign = np.zeros(ray_count, dtype=np.int64)
    distance = np.full(ray_count, np.inf, dtype=np.float32)
    dims = np.asarray(grid.shape, dtype=np.float32)
    integer_dims = np.asarray(grid.shape, dtype=np.int64)

    for ray in range(ray_count):
        origin = origins[ray]
        direction = directions[ray]
        near = np.full(3, -np.inf, dtype=np.float32)
        far = np.full(3, np.inf, dtype=np.float32)
        parallel_outside = False
        has_direction = False

        for axis in range(3):
            component = direction[axis]
            if component > np.float32(0.0):
                has_direction = True
                near[axis] = (np.float32(0.0) - origin[axis]) / component
                far[axis] = (dims[axis] - origin[axis]) / component
            elif component < np.float32(0.0):
                has_direction = True
                near[axis] = (dims[axis] - origin[axis]) / component
                far[axis] = (np.float32(0.0) - origin[axis]) / component
            elif origin[axis] < np.float32(0.0) or origin[axis] >= dims[axis]:
                parallel_outside = True

        t_enter = np.max(near)
        t_exit = np.min(far)
        start_t = np.maximum(t_enter, np.float32(0.0))
        origin_inside = bool(np.all((origin >= 0.0) & (origin < dims)))
        valid = (
            has_direction
            and not parallel_outside
            and t_exit >= start_t
            and t_exit >= np.float32(0.0)
        )
        if not valid:
            continue

        step = np.sign(direction).astype(np.int64)
        start_position = (origin + direction * start_t).astype(np.float32)
        if not origin_inside:
            start_position = np.nextafter(
                start_position, start_position + direction
            )
        current = np.floor(start_position).astype(np.int64)
        if np.any(current < 0) or np.any(current >= integer_dims):
            continue

        entry_axis = int(np.argmax(near))
        initial_material = grid[tuple(current)]
        if initial_material != 0:
            hit[ray] = 1
            material[ray] = initial_material
            voxel_out[ray] = current
            distance[ray] = start_t
            if not origin_inside:
                face_axis[ray] = entry_axis
                face_sign[ray] = -step[entry_axis]
            continue

        delta_t = np.full(3, np.inf, dtype=np.float32)
        next_t = np.full(3, np.inf, dtype=np.float32)
        for axis in range(3):
            if step[axis] != 0:
                delta_t[axis] = np.abs(
                    np.float32(1.0) / direction[axis]
                )
                boundary = np.float32(
                    current[axis] + (1 if step[axis] > 0 else 0)
                )
                next_t[axis] = np.maximum(
                    (boundary - origin[axis]) / direction[axis], start_t
                )

        for _ in range(max_steps):
            axis = int(np.argmin(next_t))
            crossing_t = next_t[axis]
            current[axis] += step[axis]
            next_t[axis] = crossing_t + delta_t[axis]

            if crossing_t > t_exit:
                break
            if np.any(current < 0) or np.any(current >= integer_dims):
                break

            current_material = grid[tuple(current)]
            if current_material != 0:
                hit[ray] = 1
                material[ray] = current_material
                voxel_out[ray] = current
                face_axis[ray] = axis
                face_sign[ray] = -step[axis]
                distance[ray] = crossing_t
                break

    return hit, material, voxel_out, face_axis, face_sign, distance


def test_axis_aligned_known_answer_and_output_contract():
    grid = np.zeros((8, 8, 8), dtype=np.uint8)
    grid[4, 3, 2] = 2
    origins = np.array([[0.5, 3.5, 2.5]], dtype=np.float32)
    directions = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    tensors, result = _call_cuda(grid, origins, directions)

    assert tuple(tensor.dtype for tensor in tensors) == (
        "uint8",
        "uint8",
        "int64",
        "int64",
        "int64",
        "float32",
    )
    assert tuple(tensor.shape for tensor in tensors) == (
        (1,),
        (1,),
        (1, 3),
        (1,),
        (1,),
        (1,),
    )
    assert int(result[0][0]) == 1
    assert int(result[1][0]) == 2
    np.testing.assert_array_equal(result[2][0], (4, 3, 2))
    assert int(result[3][0]) == 0
    assert int(result[4][0]) == -1
    assert result[5][0] == np.float32(3.5)


def test_diagonal_crosses_hand_computed_sequence():
    origins = np.array([[0.25, 0.25, 0.25]], dtype=np.float32)
    directions = _normalized(np.array([[1.0, 0.6, 0.3]], dtype=np.float32))
    expected_sequence = [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (2, 1, 0),
        (2, 1, 1),
        (3, 1, 1),
        (3, 2, 1),
    ]

    for expected_voxel in expected_sequence:
        grid = np.zeros((8, 8, 8), dtype=np.uint8)
        grid[expected_voxel] = 2
        _, result = _call_cuda(grid, origins, directions)
        assert int(result[0][0]) == 1
        np.testing.assert_array_equal(result[2][0], expected_voxel)

    grid = np.zeros((8, 8, 8), dtype=np.uint8)
    grid[0:4, 0:3, 0:2] = 1
    for expected_voxel in expected_sequence:
        grid[expected_voxel] = 0
    grid[expected_sequence[-1]] = 2
    _, result = _call_cuda(grid, origins, directions)
    assert int(result[0][0]) == 1
    assert int(result[1][0]) == 2
    np.testing.assert_array_equal(result[2][0], expected_sequence[-1])


def test_ray_starting_inside_solid():
    grid = np.zeros((8, 8, 8), dtype=np.uint8)
    grid[3, 4, 5] = 3
    origins = np.array([[3.25, 4.25, 5.25]], dtype=np.float32)
    directions = _normalized(
        np.array([[-0.4, 0.7, 0.2]], dtype=np.float32)
    )

    _, result = _call_cuda(grid, origins, directions)

    assert int(result[0][0]) == 1
    assert int(result[1][0]) == 3
    np.testing.assert_array_equal(result[2][0], (3, 4, 5))
    assert int(result[3][0]) == -1
    assert int(result[4][0]) == 0
    assert result[5][0] == np.float32(0.0)


def test_ray_missing_grid_entirely():
    grid = np.zeros((8, 8, 8), dtype=np.uint8)
    origins = np.array([[-2.0, 9.0, 4.0]], dtype=np.float32)
    directions = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    _, result = _call_cuda(grid, origins, directions)

    assert int(result[0][0]) == 0
    assert int(result[1][0]) == 0
    np.testing.assert_array_equal(result[2][0], (-1, -1, -1))
    assert int(result[3][0]) == -1
    assert int(result[4][0]) == 0
    assert bool(np.isinf(result[5][0]))


def test_200_random_rays_match_scalar_reference_exactly():
    rng = np.random.default_rng(7_731)
    grid = (rng.random((8, 8, 8)) < 0.18).astype(np.uint8)
    grid *= rng.integers(1, 4, size=grid.shape, dtype=np.uint8)
    origins = rng.uniform(-3.0, 11.0, size=(200, 3)).astype(np.float32)
    directions = _normalized(
        rng.normal(size=(200, 3)).astype(np.float32)
    )
    max_steps = int(sum(grid.shape) + 3)

    _, actual = _call_cuda(grid, origins, directions, max_steps)
    expected = _reference_raycast(grid, origins, directions, max_steps)

    for field_name, actual_field, expected_field in zip(
        FIELD_NAMES, actual, expected, strict=True
    ):
        np.testing.assert_array_equal(
            actual_field, expected_field, err_msg=field_name
        )
