"""PAINT-CUDA-2 frozen gates for back-project/gather/cosine blend.

The NumPy functions here are intentionally vendored transcriptions of
ColdCast ``hy3d_tc/paint/bake.py:649-704,757-781``.  The engine tests do not
import ColdCast or depend on its checkout at runtime.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc


DEPTH_THRESHOLD = np.float32(3.0e-3)


def _cpu_back_project(
    atlas_positions_h,
    view,
    view_depth,
    view_reliable,
    view_cosine,
    world_to_camera,
    image_projection,
    depth_threshold=DEPTH_THRESHOLD,
):
    """Fixed-row equivalent of the frozen compacting CPU back-project."""

    positions = np.ascontiguousarray(atlas_positions_h, dtype=np.float32)
    image = np.ascontiguousarray(view, dtype=np.float32)
    depth = np.ascontiguousarray(view_depth, dtype=np.float32)
    reliable = np.ascontiguousarray(view_reliable, dtype=np.uint8)
    cosine_map = np.ascontiguousarray(view_cosine, dtype=np.float32)
    w2c = np.ascontiguousarray(world_to_camera, dtype=np.float32)
    img_proj = np.ascontiguousarray(image_projection, dtype=np.float32)
    projected = np.ascontiguousarray(positions @ w2c.T @ img_proj, dtype=np.float32)

    height, width, channels = image.shape
    assert height == width  # public engine contract freezes this convention
    inner = (
        (projected[:, 0] <= np.float32(1.0))
        & (projected[:, 0] >= np.float32(-1.0))
        & (projected[:, 1] <= np.float32(1.0))
        & (projected[:, 1] >= np.float32(-1.0))
    )
    scaled_x = np.asarray(
        (np.clip(projected[:, 0], -1, 1) * np.float32(0.5) + np.float32(0.5))
        * np.float32(height),
        dtype=np.float32,
    )
    scaled_y = np.asarray(
        (np.clip(projected[:, 1], -1, 1) * np.float32(0.5) + np.float32(0.5))
        * np.float32(width),
        dtype=np.float32,
    )
    img_x = np.clip(scaled_x.astype(np.int64), 0, height - 1)
    img_y = np.clip(scaled_y.astype(np.int64), 0, width - 1)
    indices = img_y * height + img_x
    flat_depth = depth.reshape(-1)
    flat_reliable = reliable.reshape(-1)
    flat_cosine = cosine_map.reshape(-1)
    flat_rgb = image.reshape(-1, channels)
    depth_delta = np.asarray(
        np.abs(projected[:, 2] - flat_depth[indices]), dtype=np.float32
    )
    valid = (
        inner
        & (depth_delta < np.float32(depth_threshold))
        & (flat_reliable[indices] != 0)
        & (flat_cosine[indices] > np.float32(0.0))
    )

    output_colors = np.zeros((len(positions), channels), dtype=np.float32)
    output_cosine = np.zeros(len(positions), dtype=np.float32)
    local = np.flatnonzero(valid)
    if len(local):
        x0 = img_x[local]
        y0 = img_y[local]
        wx = np.asarray(scaled_x[local] - x0, dtype=np.float32)[:, None]
        wy = np.asarray(scaled_y[local] - y0, dtype=np.float32)[:, None]
        xr = np.clip(x0 + 1, 0, height - 1)
        yr = np.clip(y0 + 1, 0, width - 1)
        base = indices[local]
        indices_lr = y0 * height + xr
        indices_rl = yr * height + x0
        indices_rr = yr * height + xr
        sampled = (
            (
                flat_rgb[base] * (np.float32(1.0) - wx)
                + flat_rgb[indices_lr] * wx
            )
            * (np.float32(1.0) - wy)
            + (
                flat_rgb[indices_rl] * (np.float32(1.0) - wx)
                + flat_rgb[indices_rr] * wx
            )
            * wy
        )
        output_colors[local] = np.ascontiguousarray(sampled, dtype=np.float32)
        output_cosine[local] = flat_cosine[base]
    return (
        np.ascontiguousarray(valid, dtype=np.uint8),
        output_colors,
        output_cosine,
        depth_delta,
    )


def _cpu_cosine_blend(colors, cosine, valid, weights, enabled):
    """Frozen view-order accumulation with caller-supplied skip verdicts."""

    colors = np.ascontiguousarray(colors, dtype=np.float32)
    cosine = np.ascontiguousarray(cosine, dtype=np.float32)
    valid = np.ascontiguousarray(valid, dtype=np.uint8)
    weights = np.ascontiguousarray(weights, dtype=np.float32)
    enabled = np.ascontiguousarray(enabled, dtype=np.uint8)
    views, samples, channels = colors.shape
    texture_accum = np.zeros((samples, channels), dtype=np.float32)
    trust = np.zeros(samples, dtype=np.float32)
    for view_index in range(views):
        if not enabled[view_index]:
            continue
        weighted = np.asarray(
            np.float32(weights[view_index]) * np.power(cosine[view_index], 4),
            dtype=np.float32,
        )
        contributes = (valid[view_index] != 0) & (weighted > np.float32(0.0))
        texture_accum[contributes] += (
            colors[view_index, contributes] * weighted[contributes, None]
        )
        trust[contributes] += weighted[contributes]
    denominator = np.maximum(trust, np.float32(1.0e-8))
    texture_accum /= denominator[:, None]
    return (
        np.ascontiguousarray(texture_accum, dtype=np.float32),
        np.ascontiguousarray(trust, dtype=np.float32),
        np.ascontiguousarray(trust > np.float32(1.0e-8), dtype=np.uint8),
    )


def _device_back_project(fixture, threads=256):
    return tc.bake_back_project(
        tc.tensor(fixture["positions"], dtype="float32"),
        tc.tensor(fixture["view"], dtype="float32"),
        tc.tensor(fixture["depth"], dtype="float32"),
        tc.tensor(fixture["reliable"], dtype="uint8"),
        tc.tensor(fixture["cosine"], dtype="float32"),
        tc.tensor(fixture["w2c"], dtype="float32"),
        tc.tensor(fixture["img_proj"], dtype="float32"),
        fixture["threshold"],
        threads=threads,
    )


def _device_blend(colors, cosine, valid, weights, enabled, threads=256):
    return tc.bake_cosine_blend(
        tc.tensor(colors, dtype="float32"),
        tc.tensor(cosine, dtype="float32"),
        tc.tensor(valid, dtype="uint8"),
        tc.tensor(weights, dtype="float32"),
        tc.tensor(enabled, dtype="uint8"),
        threads=threads,
    )


def _host_tuple(device_tuple):
    return tuple(value.numpy() for value in device_tuple)


def _bytes_tuple(arrays):
    return tuple(np.ascontiguousarray(value).tobytes() for value in arrays)


def _assert_bit_exact(actual, expected):
    assert len(actual) == len(expected)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        assert actual_value.shape == expected_value.shape
        assert actual_value.dtype == expected_value.dtype
        assert actual_value.tobytes() == expected_value.tobytes()


def _consistent_quad_fixture():
    """Samples from two consistently wound triangles forming one UV quad."""

    rng = np.random.default_rng(20260716)
    side = 32
    rows, cols = np.meshgrid(
        np.arange(16, dtype=np.float32),
        np.arange(16, dtype=np.float32),
        indexing="ij",
    )
    # A 16x16 atlas sample grid over the consistently oriented quad.  Keep it
    # compact enough that the nontrivial camera transform remains in bounds.
    x = (cols.reshape(-1) + np.float32(0.3125)) / np.float32(16.0)
    y = (rows.reshape(-1) + np.float32(0.6875)) / np.float32(16.0)
    x = (x - np.float32(0.5)) * np.float32(0.7)
    y = (y - np.float32(0.5)) * np.float32(0.7)
    z = rng.uniform(-0.25, 0.25, len(x)).astype(np.float32)
    positions = np.ascontiguousarray(
        np.stack([x, y, z, np.ones_like(x)], axis=1), dtype=np.float32
    )
    w2c = np.asarray(
        [
            [-0.79863548, -0.60181504, 0.0, 0.0],
            [-0.17595369, 0.23349842, 0.95630473, 0.0],
            [-0.57551855, 0.76373893, -0.29237169, -0.45],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    img_proj = np.diag(
        np.asarray([1.35, 1.35, 1.0, 1.0], dtype=np.float32)
    )
    view = rng.uniform(0.0, 1.0, (side, side, 3)).astype(np.float32)
    depth = np.zeros((side, side), dtype=np.float32)
    reliable = np.ones((side, side), dtype=np.uint8)
    cosine = rng.uniform(0.1, 1.0, (side, side)).astype(np.float32)
    return {
        "positions": positions,
        "view": view,
        "depth": depth,
        "reliable": reliable,
        "cosine": cosine,
        "w2c": w2c,
        "img_proj": img_proj,
        # Visibility is deliberately open here; G-VIS owns threshold cases.
        "threshold": np.float32(4.0),
    }


def _visibility_fixture():
    side = 8
    count = 8
    columns = np.arange(count, dtype=np.float32)
    rows = np.arange(count, dtype=np.float32)[::-1]
    x = ((columns + np.float32(0.25)) / np.float32(side)) * np.float32(2.0) - 1
    y = ((rows + np.float32(0.625)) / np.float32(side)) * np.float32(2.0) - 1
    z = np.asarray([0.125, -0.25, 0.375, 0.0, -0.5, 0.25, -0.375, 0.5], np.float32)
    positions = np.ascontiguousarray(
        np.stack([x, y, z, np.ones(count, np.float32)], axis=1)
    )
    identity = np.eye(4, dtype=np.float32)
    projected = np.ascontiguousarray(positions @ identity.T @ identity, np.float32)
    scaled_x = np.asarray(
        (projected[:, 0] * np.float32(0.5) + np.float32(0.5))
        * np.float32(side),
        np.float32,
    )
    scaled_y = np.asarray(
        (projected[:, 1] * np.float32(0.5) + np.float32(0.5))
        * np.float32(side),
        np.float32,
    )
    img_x = np.clip(scaled_x.astype(np.int64), 0, side - 1)
    img_y = np.clip(scaled_y.astype(np.int64), 0, side - 1)
    flat = img_y * side + img_x
    assert len(np.unique(flat)) == count

    depth = np.full((side, side), np.float32(99.0), dtype=np.float32)
    # exact surface tie, front occluder, behind occluder, threshold boundary,
    # just-inside threshold, grazing positive cosine, zero cosine, unreliable.
    depth.reshape(-1)[flat] = np.asarray(
        [
            z[0],
            z[1] - np.float32(0.02),
            z[2] + np.float32(0.02),
            DEPTH_THRESHOLD,
            z[4] + np.nextafter(DEPTH_THRESHOLD, np.float32(0.0)),
            z[5],
            z[6],
            z[7],
        ],
        dtype=np.float32,
    )
    reliable = np.ones((side, side), dtype=np.uint8)
    reliable.reshape(-1)[flat[7]] = 0
    cosine = np.ones((side, side), dtype=np.float32)
    cosine.reshape(-1)[flat[5]] = np.nextafter(np.float32(0.0), np.float32(1.0))
    cosine.reshape(-1)[flat[6]] = 0.0
    view = np.arange(side * side * 3, dtype=np.float32).reshape(side, side, 3)
    view /= np.float32(side * side * 3)
    return {
        "positions": positions,
        "view": view,
        "depth": depth,
        "reliable": reliable,
        "cosine": cosine,
        "w2c": identity,
        "img_proj": identity,
        "threshold": DEPTH_THRESHOLD,
    }


def test_public_contract_shapes_dtypes_and_validation():
    fixture = _consistent_quad_fixture()
    valid, colors, cosine, delta = _device_back_project(fixture, threads=64)
    samples = len(fixture["positions"])
    assert valid.shape == (samples,)
    assert colors.shape == (samples, 3)
    assert cosine.shape == delta.shape == (samples,)
    assert valid.dtype == "uint8"
    assert colors.dtype == cosine.dtype == delta.dtype == "float32"
    assert not valid.requires_grad
    assert not colors.requires_grad

    views = 3
    stacked_colors = np.stack([colors.numpy()] * views)
    stacked_cosine = np.stack([cosine.numpy()] * views)
    stacked_valid = np.stack([valid.numpy()] * views)
    weights = np.asarray([1.0, 0.5, 0.25], dtype=np.float32)
    enabled = np.asarray([1, 0, 1], dtype=np.uint8)
    texture, trust, blend_valid = _device_blend(
        stacked_colors, stacked_cosine, stacked_valid, weights, enabled
    )
    assert texture.shape == (samples, 3)
    assert trust.shape == blend_valid.shape == (samples,)
    assert texture.dtype == trust.dtype == "float32"
    assert blend_valid.dtype == "uint8"

    with pytest.raises(RuntimeError, match="warp multiple"):
        _device_back_project(fixture, threads=33)
    rectangular = dict(fixture)
    rectangular["view"] = fixture["view"][:, :-1]
    rectangular["depth"] = fixture["depth"][:, :-1]
    rectangular["reliable"] = fixture["reliable"][:, :-1]
    rectangular["cosine"] = fixture["cosine"][:, :-1]
    with pytest.raises(RuntimeError, match="square view"):
        _device_back_project(rectangular)


def test_g_vis_synthetic_verdicts_and_ties_reported():
    fixture = _visibility_fixture()
    expected = _cpu_back_project(
        fixture["positions"],
        fixture["view"],
        fixture["depth"],
        fixture["reliable"],
        fixture["cosine"],
        fixture["w2c"],
        fixture["img_proj"],
        fixture["threshold"],
    )
    actual = _host_tuple(_device_back_project(fixture, threads=128))
    exact_depth_ties = expected[3] == np.float32(0.0)
    threshold_ties = expected[3].view(np.uint32) == np.asarray(
        fixture["threshold"], dtype=np.float32
    ).view(np.uint32)
    non_tie = ~(exact_depth_ties | threshold_ties)
    assert np.array_equal(actual[0][non_tie], expected[0][non_tie])
    divergent_all = int(np.count_nonzero(actual[0] != expected[0]))
    print(
        "PAINT-CUDA-2 G-VIS "
        f"exact_depth_ties={int(np.count_nonzero(exact_depth_ties))} "
        f"threshold_boundary_ties={int(np.count_nonzero(threshold_ties))} "
        f"non_tie={int(np.count_nonzero(non_tie))} "
        f"non_tie_divergent={int(np.count_nonzero((actual[0] != expected[0]) & non_tie))} "
        f"all_divergent={divergent_all}",
        flush=True,
    )


def test_g_texel_back_project_and_bilinear_are_bit_exact():
    fixture = _consistent_quad_fixture()
    expected = _cpu_back_project(
        fixture["positions"],
        fixture["view"],
        fixture["depth"],
        fixture["reliable"],
        fixture["cosine"],
        fixture["w2c"],
        fixture["img_proj"],
        fixture["threshold"],
    )
    actual = _host_tuple(_device_back_project(fixture, threads=256))
    _assert_bit_exact(actual, expected)


def test_g_texel_ordered_blend_is_bit_exact_when_power_is_exact():
    rng = np.random.default_rng(2026071601)
    views, samples, channels = 6, 257, 3
    colors = rng.uniform(0.0, 1.0, (views, samples, channels)).astype(np.float32)
    power_exact = np.asarray([1.0, 0.5, 0.25, 0.125], dtype=np.float32)
    cosine = np.ascontiguousarray(
        power_exact[(np.arange(views * samples) % len(power_exact))].reshape(views, samples)
    )
    valid = np.ones((views, samples), dtype=np.uint8)
    valid[2, ::7] = 0
    weights = np.asarray([1.0, 0.5, 0.25, 0.125, 1.0, 0.5], dtype=np.float32)
    enabled = np.asarray([1, 1, 0, 1, 1, 1], dtype=np.uint8)
    expected = _cpu_cosine_blend(colors, cosine, valid, weights, enabled)
    actual = _host_tuple(
        _device_blend(colors, cosine, valid, weights, enabled, threads=128)
    )
    _assert_bit_exact(actual, expected)


def _pow_break_fixture():
    rng = np.random.default_rng(2026071602)
    views, samples, channels = 6, 65537, 3
    colors = rng.uniform(0.0, 1.0, (views, samples, channels)).astype(np.float32)
    cosine = rng.uniform(0.01, 1.0, (views, samples)).astype(np.float32)
    valid = np.ones((views, samples), dtype=np.uint8)
    weights = np.asarray([1.0, 0.1, 0.5, 0.1, 0.05, 0.05], dtype=np.float32)
    enabled = np.ones(views, dtype=np.uint8)
    return colors, cosine, valid, weights, enabled


def _positive_ulp_distance(actual, expected):
    assert np.all(actual >= 0) and np.all(expected >= 0)
    return np.abs(
        actual.view(np.int32).astype(np.int64)
        - expected.view(np.int32).astype(np.int64)
    )


def test_g_texel_native_pow_spread_reported_without_threshold():
    colors, cosine, valid, weights, enabled = _pow_break_fixture()
    expected = _cpu_cosine_blend(colors, cosine, valid, weights, enabled)
    actual = _host_tuple(
        _device_blend(colors, cosine, valid, weights, enabled, threads=256)
    )
    trust_diff = actual[1].view(np.uint32) != expected[1].view(np.uint32)
    texture_diff = np.any(
        actual[0].view(np.uint32) != expected[0].view(np.uint32), axis=1
    )
    trust_ulp = _positive_ulp_distance(actual[1], expected[1])
    texture_ulp = _positive_ulp_distance(actual[0], expected[0])
    print(
        "PAINT-CUDA-2 G-TEXEL POW-STOP "
        f"trust_mismatch={int(np.count_nonzero(trust_diff))}/{len(trust_diff)} "
        f"texture_texel_mismatch={int(np.count_nonzero(texture_diff))}/{len(texture_diff)} "
        f"max_trust_ulp={int(trust_ulp.max(initial=0))} "
        f"max_texture_ulp={int(texture_ulp.max(initial=0))} "
        f"max_trust_abs={float(np.max(np.abs(actual[1].astype(np.float64) - expected[1].astype(np.float64))))} "
        f"max_texture_abs={float(np.max(np.abs(actual[0].astype(np.float64) - expected[0].astype(np.float64))))}",
        flush=True,
    )
    # This is an operation-identification assertion, not an acceptance bound:
    # the frozen exact gate must remain stopped while native device powf differs.
    assert np.any(trust_diff)


def test_g_texel_full_blend_lead_registered_gate():
    fixture = _pow_break_fixture()
    expected = _cpu_cosine_blend(*fixture)
    actual = _host_tuple(_device_blend(*fixture, threads=256))
    trust_ulp = _positive_ulp_distance(actual[1], expected[1])
    texture_ulp = _positive_ulp_distance(actual[0], expected[0])
    trust_abs = np.abs(actual[1].astype(np.float64) - expected[1].astype(np.float64))
    texture_abs = np.abs(
        actual[0].astype(np.float64) - expected[0].astype(np.float64)
    )
    # ColdCast LEDGER 2026-07-16 PAINT-CUDA-2 ACCEPTED, lead registration:
    # native powf composition is accepted at trust <=4 ULP, texture <=8 ULP,
    # and absolute spread <=5e-7.  Every non-pow output remains exact.
    assert int(trust_ulp.max(initial=0)) <= 4
    assert int(texture_ulp.max(initial=0)) <= 8
    assert float(trust_abs.max(initial=0.0)) <= 5.0e-7
    assert float(texture_abs.max(initial=0.0)) <= 5.0e-7
    assert actual[2].tobytes() == expected[2].tobytes()


def test_g_det_five_reruns_two_launch_configurations_byte_equal():
    fixture = _consistent_quad_fixture()
    reference_back_project = None
    reference_blend = None
    for threads in (64, 512):
        configuration_back_project = None
        configuration_blend = None
        for _ in range(5):
            projected = _host_tuple(_device_back_project(fixture, threads=threads))
            projected_bytes = _bytes_tuple(projected)
            colors = np.stack([projected[1]] * 6)
            cosine = np.stack([projected[2]] * 6)
            valid = np.stack([projected[0]] * 6)
            weights = np.asarray([1.0, 0.1, 0.5, 0.1, 0.05, 0.05], np.float32)
            enabled = np.asarray([1, 1, 1, 0, 1, 1], np.uint8)
            blended = _host_tuple(
                _device_blend(
                    colors, cosine, valid, weights, enabled, threads=threads
                )
            )
            blended_bytes = _bytes_tuple(blended)
            if configuration_back_project is None:
                configuration_back_project = projected_bytes
                configuration_blend = blended_bytes
            else:
                assert projected_bytes == configuration_back_project
                assert blended_bytes == configuration_blend
        if reference_back_project is None:
            reference_back_project = configuration_back_project
            reference_blend = configuration_blend
        else:
            assert configuration_back_project == reference_back_project
            assert configuration_blend == reference_blend
