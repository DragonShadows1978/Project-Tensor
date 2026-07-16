"""PAINT-CUDA-1 frozen engine gates for deterministic triangle rasterization.

The NumPy code in this file is intentionally vendored.  It transcribes the
frozen ColdCast ``SoftwareRasterizer.rasterize_clip`` oracle without importing
ColdCast or depending on its checkout at test time.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc


INT32_MAX = np.iinfo(np.int32).max
DEPTH_QUANTIZATION = np.float32(2 << 17)


def _fma32(first, second, addend):
    """Portable emulation of one binary32 CUDA FFMA."""

    return np.asarray(
        np.asarray(first, dtype=np.float64)
        * np.asarray(second, dtype=np.float64)
        + np.asarray(addend, dtype=np.float64),
        dtype=np.float32,
    )


def _screen_positions(clip, height, width):
    reciprocal_w = np.float32(1.0) / clip[:, 3]
    x = _fma32(
        _fma32(
            np.asarray(clip[:, 0] * reciprocal_w, dtype=np.float32),
            np.float32(0.5),
            np.float32(0.5),
        ),
        np.float32(width - 1),
        np.float32(0.5),
    )
    y = _fma32(
        _fma32(
            np.asarray(clip[:, 1] * reciprocal_w, dtype=np.float32),
            np.float32(0.5),
            np.float32(0.5),
        ),
        np.float32(height - 1),
        np.float32(0.5),
    )
    z = _fma32(
        np.asarray(clip[:, 2] * reciprocal_w, dtype=np.float32),
        np.float32(0.49999),
        np.float32(0.5),
    )
    return np.ascontiguousarray(np.stack([x, y, z], axis=1), dtype=np.float32)


def _signed_area2(a, b, c):
    first_x = np.float32(c[0] - a[0])
    first_y = np.float32(b[1] - a[1])
    second = np.float32(
        np.float32(b[0] - a[0]) * np.float32(c[1] - a[1])
    )
    return np.float32(
        np.float64(first_x) * np.float64(first_y) - np.float64(second)
    )


def _numpy_rasterize_clip(clip_positions, faces, height, width):
    """Frozen NumPy oracle transcribed from ColdCast rasterize_clip()."""

    clip = np.ascontiguousarray(np.asarray(clip_positions, dtype=np.float32))
    triangles = np.ascontiguousarray(np.asarray(faces, dtype=np.int64))
    screen = _screen_positions(clip, height, width)
    face_ids = np.zeros((height, width), dtype=np.int64)
    barycentric = np.zeros((height, width, 3), dtype=np.float32)
    depth_keys = np.full((height, width), INT32_MAX, dtype=np.int64)

    for face_index, triangle in enumerate(triangles):
        vertices = screen[triangle]
        a, b, c = vertices
        area = _signed_area2(a, b, c)
        if area == np.float32(0.0):
            continue

        x0 = max(0, int(float(np.min(vertices[:, 0]))))
        y0 = max(0, int(float(np.min(vertices[:, 1]))))
        x1 = min(width, int(math.ceil(float(np.max(vertices[:, 0])) + 1.0)))
        y1 = min(height, int(math.ceil(float(np.max(vertices[:, 1])) + 1.0)))
        if x1 <= x0 or y1 <= y0:
            continue

        px = np.arange(x0, x1, dtype=np.float32)[None, :] + np.float32(0.5)
        py = np.arange(y0, y1, dtype=np.float32)[:, None] + np.float32(0.5)
        inverse_area = np.float32(1.0) / area
        px_from_a = np.asarray(px - a[0], dtype=np.float32)
        py_from_a = np.asarray(py - a[1], dtype=np.float32)
        beta_second = np.asarray(
            px_from_a * np.float32(c[1] - a[1]), dtype=np.float32
        )
        beta_tri = _fma32(
            np.float32(c[0] - a[0]), py_from_a, -beta_second
        )
        beta = np.asarray(beta_tri * inverse_area, dtype=np.float32)
        gamma_second = np.asarray(
            np.float32(b[0] - a[0]) * py_from_a, dtype=np.float32
        )
        gamma_tri = _fma32(
            px_from_a, np.float32(b[1] - a[1]), -gamma_second
        )
        gamma = np.asarray(gamma_tri * inverse_area, dtype=np.float32)
        alpha = np.asarray(
            np.float64(1.0)
            - np.asarray(beta, dtype=np.float64)
            - np.asarray(gamma, dtype=np.float64),
            dtype=np.float32,
        )
        inside = (
            (alpha >= np.float32(0.0))
            & (alpha <= np.float32(1.0))
            & (beta >= np.float32(0.0))
            & (beta <= np.float32(1.0))
            & (gamma >= np.float32(0.0))
            & (gamma <= np.float32(1.0))
        )
        if not np.any(inside):
            continue

        depth = _fma32(
            gamma,
            np.float32(c[2]),
            _fma32(
                alpha,
                np.float32(a[2]),
                np.asarray(beta * np.float32(b[2]), dtype=np.float32),
            ),
        )
        interp_alpha = np.asarray(
            alpha / np.float32(clip[triangle[0], 3]), dtype=np.float32
        )
        interp_beta = np.asarray(
            beta / np.float32(clip[triangle[1], 3]), dtype=np.float32
        )
        interp_gamma = np.asarray(
            gamma / np.float32(clip[triangle[2], 3]), dtype=np.float32
        )
        interp_sum = np.asarray(
            np.asarray(interp_alpha + interp_beta, dtype=np.float32)
            + interp_gamma,
            dtype=np.float32,
        )
        interp_reciprocal = np.asarray(
            np.float32(1.0) / interp_sum, dtype=np.float32
        )
        interp_alpha = np.asarray(
            interp_alpha * interp_reciprocal, dtype=np.float32
        )
        interp_beta = np.asarray(
            interp_beta * interp_reciprocal, dtype=np.float32
        )
        interp_gamma = np.asarray(
            interp_gamma * interp_reciprocal, dtype=np.float32
        )
        quantized = np.asarray(
            depth * DEPTH_QUANTIZATION, dtype=np.int32
        ).astype(np.int64)
        current_depth = depth_keys[y0:y1, x0:x1]
        current_face = face_ids[y0:y1, x0:x1]
        one_based = face_index + 1
        equal_depth = quantized == current_depth
        replace = inside & (
            (quantized < current_depth)
            | (equal_depth & ((current_face == 0) | (one_based < current_face)))
        )
        if not np.any(replace):
            continue
        current_depth[replace] = quantized[replace]
        current_face[replace] = one_based
        bary_view = barycentric[y0:y1, x0:x1]
        bary_view[..., 0][replace] = interp_alpha[replace]
        bary_view[..., 1][replace] = interp_beta[replace]
        bary_view[..., 2][replace] = interp_gamma[replace]

    return face_ids, barycentric, depth_keys


def _clip_from_screen(points, height, width, *, clip_z=0.0, w=1.0):
    """Construct fp32 clip positions whose projected points are near `points`."""

    points = np.asarray(points, dtype=np.float64)
    count = len(points)
    ws = np.broadcast_to(np.asarray(w, dtype=np.float32), (count,)).copy()
    zs = np.broadcast_to(np.asarray(clip_z, dtype=np.float32), (count,)).copy()
    ndc_x = np.asarray(
        np.float64(2.0)
        * ((points[:, 0] - np.float64(0.5)) / np.float64(width - 1))
        - np.float64(1.0),
        dtype=np.float32,
    )
    ndc_y = np.asarray(
        np.float64(2.0)
        * ((points[:, 1] - np.float64(0.5)) / np.float64(height - 1))
        - np.float64(1.0),
        dtype=np.float32,
    )
    clip = np.stack(
        [
            np.asarray(ndc_x * ws, dtype=np.float32),
            np.asarray(ndc_y * ws, dtype=np.float32),
            zs,
            ws,
        ],
        axis=1,
    )
    return np.ascontiguousarray(clip, dtype=np.float32)


def _cuda_raster(clip, faces, height, width, *, face_threads=256, pixel_threads=256):
    face, barycentric, depth = tc.rasterize_clip(
        tc.tensor(clip, dtype="float32"),
        tc.tensor(faces, dtype="int64"),
        height,
        width,
        face_threads=face_threads,
        pixel_threads=pixel_threads,
    )
    return face.numpy(), barycentric.numpy(), depth.numpy()


def _shared_edge_fixture():
    height = width = 16
    points = np.asarray(
        [
            (2.0, 2.0),
            (13.0, 2.0),
            (13.0, 13.0),
            (2.0, 13.0),
        ],
        dtype=np.float64,
    )
    clip = _clip_from_screen(points, height, width, clip_z=-0.25)
    faces = np.asarray([(0, 1, 2), (0, 2, 3)], dtype=np.int64)
    return clip, faces, height, width


def _reversed_winding_fixture():
    clip, faces, height, width = _shared_edge_fixture()
    return clip, np.ascontiguousarray(faces[:, ::-1]), height, width


def _overlap_fixture():
    height = width = 16
    triangle = np.asarray([(2.0, 2.0), (13.0, 3.0), (4.0, 13.0)])
    clip_front = _clip_from_screen(triangle, height, width, clip_z=0.0)
    clip_signed_near = _clip_from_screen(triangle, height, width, clip_z=-4.0)
    clip = np.concatenate([clip_front, clip_signed_near, clip_signed_near])
    faces = np.arange(9, dtype=np.int64).reshape(3, 3)
    return np.ascontiguousarray(clip), faces, height, width


def _degeneracy_fixture():
    height = width = 16
    triangles = [
        [(2.0, 2.0), (5.0, 5.0), (8.0, 8.0)],
        [(9.0, 1.0), (9.0, 8.0), (9.0, 14.0)],
        [(-5.0, -4.0), (11.0, 2.0), (2.0, 11.0)],
        [(12.1, 12.1), (12.9, 12.1), (12.5, 12.9)],
        [(14.02, 14.02), (14.20, 14.02), (14.02, 14.20)],
        [(-5.0, -4.0), (11.0, 2.0), (2.0, 11.0)],
    ]
    clip_parts = [
        _clip_from_screen(points, height, width, clip_z=-0.5)
        for points in triangles
    ]
    clip = np.concatenate(clip_parts)
    faces = np.arange(len(triangles) * 3, dtype=np.int64).reshape(-1, 3)
    return np.ascontiguousarray(clip), faces, height, width


def _perspective_fixture():
    height = width = 32
    points = np.asarray(
        [
            (3.0, 4.0),
            (27.0, 6.0),
            (9.0, 28.0),
            (6.0, 8.0),
            (29.0, 17.0),
            (8.0, 25.0),
        ]
    )
    clip = _clip_from_screen(
        points,
        height,
        width,
        clip_z=np.asarray([-0.8, -0.3, 0.1, -0.2, -1.0, 0.4]),
        w=np.asarray([0.75, 1.25, 1.5, 1.1, 0.85, 1.4]),
    )
    faces = np.asarray([(0, 1, 2), (3, 4, 5)], dtype=np.int64)
    return clip, faces, height, width


def _determinism_fixture():
    rng = np.random.default_rng(20260715)
    height, width = 73, 67
    triangles = []
    clip_parts = []
    for _ in range(96):
        center = np.asarray(
            [rng.uniform(-2.0, width + 2.0), rng.uniform(-2.0, height + 2.0)]
        )
        radius = rng.uniform(0.35, 9.0)
        angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, size=3))
        points = center + np.stack([np.cos(angles), np.sin(angles)], axis=1) * radius
        ws = rng.uniform(0.7, 1.6, size=3).astype(np.float32)
        zs = rng.uniform(-4.0, 1.0, size=3).astype(np.float32)
        triangles.append(points)
        clip_parts.append(_clip_from_screen(points, height, width, clip_z=zs, w=ws))
    # Exact duplicate geometry/depth creates deterministic tie pressure.
    triangles.extend(triangles[:16])
    clip_parts.extend([part.copy() for part in clip_parts[:16]])
    clip = np.concatenate(clip_parts)
    faces = np.arange(len(triangles) * 3, dtype=np.int64).reshape(-1, 3)
    return np.ascontiguousarray(clip), faces, height, width


PARITY_FIXTURES = {
    "shared_edge": _shared_edge_fixture,
    "reversed_winding": _reversed_winding_fixture,
    "signed_overlap": _overlap_fixture,
    "degeneracies": _degeneracy_fixture,
    "perspective": _perspective_fixture,
    "random_overlap": _determinism_fixture,
}


def _interior_mask(face_ids):
    center = face_ids[1:-1, 1:-1]
    interior = np.ones_like(face_ids, dtype=bool)
    target = interior[1:-1, 1:-1]
    for row_offset in (-1, 0, 1):
        for col_offset in (-1, 0, 1):
            target &= center == face_ids[
                1 + row_offset : face_ids.shape[0] - 1 + row_offset,
                1 + col_offset : face_ids.shape[1] - 1 + col_offset,
            ]
    interior[[0, -1], :] = False
    interior[:, [0, -1]] = False
    return interior & (face_ids > 0)


def _pixel_divergence(actual, expected):
    actual_face, actual_bary, actual_depth = actual
    expected_face, expected_bary, expected_depth = expected
    face_diff = actual_face != expected_face
    depth_diff = actual_depth != expected_depth
    bary_diff = np.any(
        actual_bary.view(np.uint32) != expected_bary.view(np.uint32), axis=2
    )
    return face_diff | depth_diff | bary_diff


def test_public_contract_shapes_dtypes_and_background():
    pixel = tc.tensor(np.asarray([0, 0, 2], dtype=np.int64), dtype="int64")
    depth = tc.tensor(np.asarray([7, -3, 4], dtype=np.int64), dtype="int64")
    face = tc.tensor(np.asarray([8, 2, 9], dtype=np.int64), dtype="int64")
    winner_depth, winner_face = tc.raster_winner_scatter_min(
        pixel, depth, face, 4
    )
    assert winner_depth.shape == (4,)
    assert winner_face.shape == (4,)
    assert winner_depth.dtype == "int64"
    assert winner_face.dtype == "int64"
    assert not winner_depth.requires_grad
    assert not winner_face.requires_grad
    np.testing.assert_array_equal(
        winner_depth.numpy(), np.asarray([-3, INT32_MAX, 4, INT32_MAX])
    )
    np.testing.assert_array_equal(winner_face.numpy(), np.asarray([2, 0, 9, 0]))

    with pytest.raises(RuntimeError, match="warp multiple"):
        tc.raster_winner_scatter_min(pixel, depth, face, 4, threads=33)

    clip, faces, height, width = _shared_edge_fixture()
    clip_device = tc.tensor(clip, dtype="float32")
    faces_device = tc.tensor(faces, dtype="int64")
    produced_depth, produced_face = tc.raster_triangle_winners(
        clip_device, faces_device, height, width, face_threads=64
    )
    resolved_depth, resolved_bary = tc.raster_winner_resolve(
        clip_device, faces_device, produced_face, pixel_threads=128
    )
    final_face, final_bary, final_depth = tc.rasterize_clip(
        clip_device,
        faces_device,
        height,
        width,
        face_threads=64,
        pixel_threads=128,
    )
    np.testing.assert_array_equal(produced_depth.numpy(), resolved_depth.numpy())
    np.testing.assert_array_equal(produced_face.numpy(), final_face.numpy())
    np.testing.assert_array_equal(resolved_depth.numpy(), final_depth.numpy())
    np.testing.assert_array_equal(resolved_bary.numpy(), final_bary.numpy())
    assert produced_depth.dtype == produced_face.dtype == resolved_depth.dtype == "int64"
    assert resolved_bary.dtype == "float32"


def test_g1_winding_and_shared_edge_final_ownership():
    clip, faces, height, width = _shared_edge_fixture()
    expected = _numpy_rasterize_clip(clip, faces, height, width)
    actual = _cuda_raster(clip, faces, height, width)
    for actual_buffer, expected_buffer in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_buffer, expected_buffer)

    reversed_faces = np.ascontiguousarray(faces[:, ::-1])
    reversed_result = _cuda_raster(clip, reversed_faces, height, width)
    np.testing.assert_array_equal(reversed_result[0] > 0, actual[0] > 0)

    first_only = _numpy_rasterize_clip(clip, faces[:1], height, width)[0] > 0
    second_only = _numpy_rasterize_clip(clip, faces[1:], height, width)[0] > 0
    union = first_only | second_only
    shared = first_only & second_only
    assert np.any(shared), "fixture must exercise inclusive shared-edge candidates"
    assert not np.any(union & (actual[0] == 0)), "shared pair has a final-owner gap"
    assert np.count_nonzero(actual[0] > 0) == np.count_nonzero(union)
    assert np.all(actual[0][shared] == 1), "lower face ID must own exact ties"


def test_g2_scatter_signed_depth_and_lower_face_ties():
    pixel = np.asarray([0, 0, 0, 1, 1, 2, 2], dtype=np.int64)
    depth = np.asarray([7, -3, -3, 0, 0, INT32_MAX, -INT32_MAX], dtype=np.int64)
    face = np.asarray([2, 9, 4, 8, 1, 5, 7], dtype=np.int64)
    expected_depth = np.asarray([-3, 0, -INT32_MAX, INT32_MAX], dtype=np.int64)
    expected_face = np.asarray([4, 1, 7, 0], dtype=np.int64)

    for order, threads in ((np.arange(len(pixel)), 64), (np.arange(len(pixel))[::-1], 256)):
        got_depth, got_face = tc.raster_winner_scatter_min(
            tc.tensor(pixel[order], dtype="int64"),
            tc.tensor(depth[order], dtype="int64"),
            tc.tensor(face[order], dtype="int64"),
            4,
            threads=threads,
        )
        np.testing.assert_array_equal(got_depth.numpy(), expected_depth)
        np.testing.assert_array_equal(got_face.numpy(), expected_face)

    clip, faces, height, width = _overlap_fixture()
    actual_face, _, actual_depth = _cuda_raster(clip, faces, height, width)
    covered = actual_face > 0
    assert np.any(covered)
    assert np.all(actual_face[covered] == 2)
    assert np.all(actual_depth[covered] < 0), "signed negative depth must beat positive"

    coplanar_face, _, _ = _cuda_raster(clip[3:], faces[1:] - 3, height, width)
    assert np.all(coplanar_face[coplanar_face > 0] == 1)


def test_g3_synthetic_degeneracies_match_frozen_oracle():
    clip, faces, height, width = _degeneracy_fixture()
    expected = _numpy_rasterize_clip(clip, faces, height, width)
    actual = _cuda_raster(clip, faces, height, width)
    for actual_buffer, expected_buffer in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_buffer, expected_buffer)

    visible_faces = set(np.unique(actual[0]).tolist())
    assert 1 not in visible_faces  # zero-area diagonal
    assert 2 not in visible_faces  # edge-on vertical face
    assert 3 in visible_faces      # clipped bbox still emits visible pixels
    assert 4 in visible_faces      # covering subpixel triangle
    assert 5 not in visible_faces  # non-covering subpixel triangle
    assert 6 not in visible_faces  # exact tied duplicate loses to face 3


def test_g4_five_reruns_and_two_launch_configurations_are_byte_equal():
    clip, faces, height, width = _determinism_fixture()
    clip_device = tc.tensor(clip, dtype="float32")
    faces_device = tc.tensor(faces, dtype="int64")
    reference_bytes = None
    for face_threads, pixel_threads in ((64, 128), (256, 512)):
        configuration_bytes = None
        for _ in range(5):
            face, barycentric, depth = tc.rasterize_clip(
                clip_device,
                faces_device,
                height,
                width,
                face_threads=face_threads,
                pixel_threads=pixel_threads,
            )
            current = (
                face.numpy().tobytes(),
                barycentric.numpy().tobytes(),
                depth.numpy().tobytes(),
            )
            if configuration_bytes is None:
                configuration_bytes = current
            else:
                assert current == configuration_bytes
        if reference_bytes is None:
            reference_bytes = configuration_bytes
        else:
            assert configuration_bytes == reference_bytes


@pytest.mark.parametrize("fixture_name", tuple(PARITY_FIXTURES))
def test_g5_cpu_parity_interior_bit_exact_boundary_reported(fixture_name):
    clip, faces, height, width = PARITY_FIXTURES[fixture_name]()
    expected = _numpy_rasterize_clip(clip, faces, height, width)
    actual = _cuda_raster(clip, faces, height, width, face_threads=128, pixel_threads=256)
    divergent = _pixel_divergence(actual, expected)
    interior = _interior_mask(expected[0])
    assert not np.any(divergent & interior), (
        f"{fixture_name}: interior face/depth/barycentric buffer diverged"
    )

    boundary = ((expected[0] > 0) | (actual[0] > 0)) & ~interior
    boundary_pixels = int(np.count_nonzero(boundary))
    boundary_divergence = int(np.count_nonzero(divergent & boundary))
    boundary_percent = (
        100.0 * boundary_divergence / boundary_pixels if boundary_pixels else 0.0
    )
    print(
        f"PAINT-CUDA-1 G5 boundary fixture={fixture_name} "
        f"divergent={boundary_divergence} boundary_pixels={boundary_pixels} "
        f"percent={boundary_percent:.9f}%",
        flush=True,
    )
    # Boundary divergence is measurement-only in this leg: deliberately no
    # threshold or zero-divergence assertion here.
