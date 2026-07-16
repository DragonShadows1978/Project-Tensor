"""PAINT-CUDA-3 frozen gates for ordered CSR island smoothing.

The CPU arithmetic is a vendored transcription of ColdCast
``hy3d_tc/paint/bake.py:1029-1068``. Graph construction follows
``bake.py:812-841`` and stays entirely host-side.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc


def _positions(vertex_count, *, phase=0.0):
    index = np.arange(vertex_count, dtype=np.float32)
    return np.ascontiguousarray(
        np.stack(
            [
                index * np.float32(0.013) + np.float32(phase),
                np.sin(index * np.float32(0.17) + np.float32(phase)).astype(
                    np.float32
                ),
                np.cos(index * np.float32(0.11) - np.float32(phase)).astype(
                    np.float32
                ),
            ],
            axis=1,
        ),
        dtype=np.float32,
    )


def _fixture(name, faces, vertex_islands, seed_vertices, *, occurrences=None):
    faces = np.ascontiguousarray(faces, dtype=np.int64).reshape(-1, 3)
    vertex_islands = np.ascontiguousarray(vertex_islands, dtype=np.int64)
    vertex_count = len(vertex_islands)
    positions = _positions(vertex_count, phase=float(len(name)) * 0.03125)
    colors = np.zeros((vertex_count, 3), dtype=np.float32)
    mask = np.zeros(vertex_count, dtype=np.float32)
    seeds = np.ascontiguousarray(seed_vertices, dtype=np.int64)
    if len(seeds):
        seed_index = seeds.astype(np.float32)
        colors[seeds, 0] = np.float32(0.125) + seed_index * np.float32(0.00003125)
        colors[seeds, 1] = np.float32(0.375) + seed_index * np.float32(0.000015625)
        colors[seeds, 2] = np.float32(0.625) - seed_index * np.float32(0.0000078125)
        mask[seeds] = np.float32(1.0)
    if occurrences is None:
        flat = faces.reshape(-1)
        occurrences = flat[mask[flat] == np.float32(0.0)]
    occurrences = np.ascontiguousarray(occurrences, dtype=np.int64)
    metadata = tc.build_inpaint_island_csr(
        faces, vertex_islands, occurrences
    )
    return {
        "name": name,
        "faces": faces,
        "vertex_islands": vertex_islands,
        "positions": positions,
        "colors": np.ascontiguousarray(colors),
        "mask": np.ascontiguousarray(mask),
        "occurrences": occurrences,
        "neighbor_offsets": metadata[0],
        "neighbors": metadata[1],
        "active_islands": metadata[2],
        "island_offsets": metadata[3],
        "island_occurrences": metadata[4],
    }


def _chain_fixture(vertex_count=18):
    faces = np.asarray(
        [[index, index + 1, index + 2] for index in range(vertex_count - 2)],
        dtype=np.int64,
    )
    return _fixture("chain", faces, np.zeros(vertex_count, np.int64), [0])


def _star_fixture(leaves=18):
    faces = np.asarray(
        [[0, leaf, 1 + (leaf % leaves)] for leaf in range(1, leaves + 1)],
        dtype=np.int64,
    )
    return _fixture("star", faces, np.zeros(leaves + 1, np.int64), [1])


def _ring_fixture(ring_size=24):
    center = ring_size
    faces = np.asarray(
        [[index, (index + 1) % ring_size, center] for index in range(ring_size)],
        dtype=np.int64,
    )
    return _fixture(
        "ring", faces, np.zeros(ring_size + 1, np.int64), [center]
    )


def _single_texel_fixture():
    return _fixture(
        "single_texel",
        np.empty((0, 3), dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        [],
        occurrences=np.asarray([0], dtype=np.int64),
    )


def _swarm_fixture(islands=10_000):
    vertices = np.arange(islands * 3, dtype=np.int64).reshape(islands, 3)
    labels = np.repeat(np.arange(islands, dtype=np.int64), 3)
    return _fixture(
        f"{islands}_island_swarm", vertices, labels, vertices[:, 0]
    )


def _giant_fixture(vertex_count=8_192):
    faces = np.column_stack(
        (
            np.arange(vertex_count - 2, dtype=np.int64),
            np.arange(1, vertex_count - 1, dtype=np.int64),
            np.arange(2, vertex_count, dtype=np.int64),
        )
    )
    return _fixture(
        "one_giant_island", faces, np.zeros(vertex_count, np.int64), [0]
    )


def _mixed_cap_fixture():
    chain_vertices = 12
    chain = np.asarray(
        [[index, index + 1, index + 2] for index in range(chain_vertices - 2)],
        dtype=np.int64,
    )
    triangle = np.asarray([[12, 13, 14]], dtype=np.int64)
    faces = np.concatenate((chain, triangle), axis=0)
    labels = np.asarray([0] * 12 + [4] * 3 + [9], dtype=np.int64)
    mask_seed = [0, 12]
    flat = faces.reshape(-1)
    occurrences = np.concatenate(
        (flat[~np.isin(flat, mask_seed)], np.asarray([15], dtype=np.int64))
    )
    return _fixture(
        "mixed_cap", faces, labels, mask_seed, occurrences=occurrences
    )


def _cpu_passes(fixture, colors, mask, pass_count):
    colors = np.ascontiguousarray(colors, dtype=np.float32).copy()
    mask = np.ascontiguousarray(mask, dtype=np.float32).copy()
    active = fixture["active_islands"]
    active_lookup = {int(value): index for index, value in enumerate(active)}
    final_counts = np.zeros(len(active), dtype=np.int64)
    for _ in range(pass_count):
        counts = np.zeros(len(active), dtype=np.int64)
        for vertex_value in fixture["occurrences"]:
            vertex = int(vertex_value)
            sum_color = np.zeros(3, dtype=np.float32)
            total_weight = np.float32(0.0)
            origin = fixture["positions"][vertex]
            begin = int(fixture["neighbor_offsets"][vertex])
            end = int(fixture["neighbor_offsets"][vertex + 1])
            for connected_value in fixture["neighbors"][begin:end]:
                connected = int(connected_value)
                if mask[connected] > 0:
                    delta = np.asarray(
                        origin - fixture["positions"][connected], dtype=np.float32
                    )
                    distance = np.sqrt(
                        np.sum(delta * delta, dtype=np.float32), dtype=np.float32
                    )
                    distance = np.maximum(distance, np.float32(1.0e-4))
                    weight = np.float32(1.0) / distance
                    weight = np.float32(weight * weight)
                    sum_color += colors[connected] * weight
                    total_weight = np.float32(total_weight + weight)
            if total_weight > 0:
                colors[vertex] = sum_color / total_weight
                mask[vertex] = np.float32(1.0)
            else:
                island = int(fixture["vertex_islands"][vertex])
                counts[active_lookup[island]] += 1
        final_counts = counts
    return colors, mask, final_counts


def _device_passes(fixture, colors=None, mask=None, *, pass_count, threads):
    colors = fixture["colors"] if colors is None else colors
    mask = fixture["mask"] if mask is None else mask
    outputs = tc.inpaint_island_passes(
        tc.tensor(fixture["positions"], dtype="float32"),
        tc.tensor(colors, dtype="float32"),
        tc.tensor(mask, dtype="float32"),
        tc.tensor(fixture["neighbor_offsets"], dtype="int64"),
        tc.tensor(fixture["neighbors"], dtype="int64"),
        tc.tensor(fixture["island_offsets"], dtype="int64"),
        tc.tensor(fixture["island_occurrences"], dtype="int64"),
        pass_count,
        threads=threads,
    )
    return tuple(output.numpy() for output in outputs)


def _positive_ulp_distance(actual, expected):
    assert np.all(actual >= 0.0) and np.all(expected >= 0.0)
    return np.abs(
        actual.view(np.int32).astype(np.int64)
        - expected.view(np.int32).astype(np.int64)
    )


def _assert_order_report(name, actual, expected):
    color_bits_differ = actual[0].view(np.uint32) != expected[0].view(np.uint32)
    color_ulp = _positive_ulp_distance(actual[0], expected[0])
    color_abs = np.abs(
        actual[0].astype(np.float64) - expected[0].astype(np.float64)
    )
    mask_mismatch = int(np.count_nonzero(actual[1] != expected[1]))
    count_mismatch = int(np.count_nonzero(actual[2] != expected[2]))
    print(
        "PAINT-CUDA-3 G-ORDER "
        f"fixture={name} color_mismatch={int(np.count_nonzero(color_bits_differ))} "
        f"max_color_ulp={int(color_ulp.max(initial=0))} "
        f"max_color_abs={float(color_abs.max(initial=0.0))} "
        f"mask_mismatch={mask_mismatch} count_mismatch={count_mismatch}",
        flush=True,
    )
    assert actual[0].tobytes() == expected[0].tobytes()
    assert actual[1].tobytes() == expected[1].tobytes()
    assert actual[2].tobytes() == expected[2].tobytes()


@pytest.mark.parametrize(
    ("factory", "passes"),
    [
        (_chain_fixture, 5),
        (_star_fixture, 5),
        (_ring_fixture, 5),
        (_single_texel_fixture, 3),
        (_swarm_fixture, 4),
        (_giant_fixture, 4),
    ],
    ids=["chain", "star", "ring", "single-texel", "10k-swarm", "giant"],
)
def test_g_order_bit_exact_at_equal_global_pass_counts(factory, passes):
    fixture = factory()
    expected = _cpu_passes(
        fixture, fixture["colors"], fixture["mask"], passes
    )
    actual = _device_passes(fixture, pass_count=passes, threads=128)
    _assert_order_report(fixture["name"], actual, expected)


def _receipt_rows(fixture, reason, passes):
    if reason is None:
        return []
    occurrence_counts = np.diff(fixture["island_offsets"])
    return [
        {
            "island": int(island),
            "reason": reason,
            "passes": passes,
            "uncolored_occurrences": int(occurrence_counts[index]),
            "uncolored_occurrences_last_complete_pass": None,
        }
        for index, island in enumerate(fixture["active_islands"])
    ]


def _drive_cpu(fixture, *, island_iteration_cap, launch_budget=None):
    colors = fixture["colors"]
    mask = fixture["mask"]
    smooth_count = 2
    last_uncolored_count = 0
    passes = 0
    reason = None
    while smooth_count > 0:
        if launch_budget is not None and passes >= launch_budget:
            reason = "wall_clock"
            break
        if passes >= island_iteration_cap:
            reason = "island_iteration"
            break
        colors, mask, counts = _cpu_passes(fixture, colors, mask, 1)
        uncolored_count = int(counts.sum())
        smooth_count += -1 if last_uncolored_count == uncolored_count else 1
        last_uncolored_count = uncolored_count
        passes += 1
    return colors, mask, reason, passes, _receipt_rows(fixture, reason, passes)


def _drive_device(
    fixture, *, island_iteration_cap, launch_budget=None, threads=128
):
    colors = fixture["colors"]
    mask = fixture["mask"]
    smooth_count = 2
    last_uncolored_count = 0
    passes = 0
    reason = None
    while smooth_count > 0:
        # WO-30b policy checks the material soft deadline between bounded
        # launches. A deterministic launch budget stands in for the clock.
        if launch_budget is not None and passes >= launch_budget:
            reason = "wall_clock"
            break
        if passes >= island_iteration_cap:
            reason = "island_iteration"
            break
        colors, mask, counts = _device_passes(
            fixture, colors, mask, pass_count=1, threads=threads
        )
        uncolored_count = int(counts.sum())
        smooth_count += -1 if last_uncolored_count == uncolored_count else 1
        last_uncolored_count = uncolored_count
        passes += 1
    return colors, mask, reason, passes, _receipt_rows(fixture, reason, passes)


@pytest.mark.parametrize(
    ("island_iteration_cap", "launch_budget", "expected_reason"),
    [
        (2, None, "island_iteration"),
        (64, 1, "wall_clock"),
        (64, None, None),
    ],
    ids=["island-cap", "deadline-between-launches", "exact-completion"],
)
def test_g_cap_global_pass_and_receipt_semantics_identical(
    island_iteration_cap, launch_budget, expected_reason
):
    fixture = _mixed_cap_fixture()
    expected = _drive_cpu(
        fixture,
        island_iteration_cap=island_iteration_cap,
        launch_budget=launch_budget,
    )
    actual = _drive_device(
        fixture,
        island_iteration_cap=island_iteration_cap,
        launch_budget=launch_budget,
        threads=128,
    )
    assert actual[2] == expected[2] == expected_reason
    assert actual[3:] == expected[3:]
    assert actual[0].tobytes() == expected[0].tobytes()
    assert actual[1].tobytes() == expected[1].tobytes()
    if expected_reason is not None:
        assert [row["island"] for row in actual[4]] == [0, 4, 9]
        assert all(row["passes"] == actual[3] for row in actual[4])


def test_g_det_five_reruns_two_launch_configurations_byte_equal():
    fixture = _swarm_fixture(islands=257)
    reference = None
    for threads in (32, 256):
        configuration = None
        for _ in range(5):
            outputs = _device_passes(fixture, pass_count=7, threads=threads)
            output_bytes = tuple(value.tobytes() for value in outputs)
            if configuration is None:
                configuration = output_bytes
            else:
                assert output_bytes == configuration
        if reference is None:
            reference = configuration
        else:
            assert configuration == reference


def test_public_contract_host_builder_shapes_stability_and_validation():
    faces = np.asarray([[3, 4, 5], [0, 1, 2], [0, 2, 1]], dtype=np.int64)
    labels = np.asarray([2, 2, 2, 9, 9, 9], dtype=np.int64)
    occurrences = np.asarray([4, 2, 5, 1, 3, 2], dtype=np.int64)
    offsets, neighbors, active, island_offsets, grouped = (
        tc.build_inpaint_island_csr(faces, labels, occurrences)
    )
    assert offsets.dtype == neighbors.dtype == np.int64
    assert active.tolist() == [2, 9]
    assert island_offsets.tolist() == [0, 3, 6]
    assert grouped.tolist() == [2, 1, 2, 4, 5, 3]
    # Source 0 occurs in the second and third face. Stable source sorting keeps
    # its original target order (1 then 2).
    assert neighbors[offsets[0] : offsets[1]].tolist() == [1, 2]

    with pytest.raises(ValueError, match="shape"):
        tc.build_inpaint_island_csr(np.asarray([0, 1, 2]), labels, occurrences)
    bad_labels = labels.copy()
    bad_labels[1] = 7
    with pytest.raises(ValueError, match="splits an edge"):
        tc.build_inpaint_island_csr(faces, bad_labels, occurrences)

    fixture = _chain_fixture()
    args = (
        tc.tensor(fixture["positions"], dtype="float32"),
        tc.tensor(fixture["colors"], dtype="float32"),
        tc.tensor(fixture["mask"], dtype="float32"),
        tc.tensor(fixture["neighbor_offsets"], dtype="int64"),
        tc.tensor(fixture["neighbors"], dtype="int64"),
        tc.tensor(fixture["island_offsets"], dtype="int64"),
        tc.tensor(fixture["island_occurrences"], dtype="int64"),
    )
    with pytest.raises(RuntimeError, match="pass_count_cap must be positive"):
        tc.inpaint_island_passes(*args, 0)
    with pytest.raises(RuntimeError, match="warp multiple"):
        tc.inpaint_island_passes(*args, 1, threads=33)
