"""Synchronized PAINT-CUDA-3 ordered-island performance and VRAM report.

Run under the shared GPU flock, for example:

    flock /mnt/ForgeRealm/ColdCast/.gpu_lock \
      env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
      python3 tests/bench_paint_inpaint.py --mode production

Host fixture construction/upload is excluded. Production expands the exact
55,193-island albedo occurrence histogram receipted from the dragon and
duplicates it for two material instances: 110,386 islands and 2,895,698
occurrences. The synthetic CSR uses four stable chain neighbors per vertex.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np


TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS.parent))
import tensor_cuda as tc


DISTRIBUTION = TESTS / "paint_inpaint_dragon_distribution.json.txt"


def _gpu_memory_mib():
    overall_text = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    overall = int(overall_text.strip().splitlines()[0])
    process_text = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    process = 0
    for row in process_text.splitlines():
        fields = [field.strip() for field in row.split(",")]
        if len(fields) == 2 and fields[0] == str(os.getpid()):
            process = int(fields[1])
            break
    return {"gpu_total_used_mib": overall, "process_used_mib": process}


def _counts(mode):
    if mode == "fixture":
        # Existing compact inpaint receipt: 228 active islands and 2,487
        # uncolored occurrences. Keep a deterministic compact skew.
        counts = np.full(228, 2487 // 228, dtype=np.int64)
        counts[: 2487 % 228] += 1
        return counts, {
            "source": "ColdCast compact bake-parity inpaint receipt",
            "active_islands": 228,
            "uncolored_occurrences": 2487,
            "oracle_passes": 8,
        }

    receipt = json.loads(DISTRIBUTION.read_text())
    histogram = receipt["distribution"]["histogram"]
    values = np.asarray([int(value) for value in histogram], dtype=np.int64)
    frequencies = np.asarray(list(histogram.values()), dtype=np.int64)
    one_material = np.repeat(values, frequencies)
    assert len(one_material) == 55_193
    assert int(one_material.sum()) == 1_447_849
    counts = np.concatenate((one_material, one_material))
    # Island scheduling must not depend on size order. Shuffle deterministically
    # so the benchmark does not hand the GPU a conveniently sorted tail.
    np.random.default_rng(2026071603).shuffle(counts)
    return counts, {
        "source": str(DISTRIBUTION.name),
        "source_glb_sha256": receipt["provenance"]["glb_sha256"],
        "active_islands": len(counts),
        "uncolored_occurrences": int(counts.sum()),
        "oracle_passes": 11,
        "occurrence_percentiles": receipt["distribution"]["percentiles"],
    }


def _synthetic_state(counts):
    counts = np.ascontiguousarray(counts, dtype=np.int64)
    island_count = len(counts)
    unique_uncolored = (counts + np.int64(2)) // np.int64(3)
    vertex_lengths = unique_uncolored + np.int64(1)  # one colored seed/island
    vertex_offsets = np.empty(island_count + 1, dtype=np.int64)
    vertex_offsets[0] = 0
    np.cumsum(vertex_lengths, out=vertex_offsets[1:])
    vertex_count = int(vertex_offsets[-1])

    global_vertex = np.arange(vertex_count, dtype=np.int64)
    island_by_vertex = np.repeat(
        np.arange(island_count, dtype=np.int64), vertex_lengths
    )
    starts = np.repeat(vertex_offsets[:-1], vertex_lengths)
    ends = np.repeat(vertex_offsets[1:] - 1, vertex_lengths)
    local = global_vertex - starts
    previous = np.where(local > 0, global_vertex - 1, global_vertex + 1)
    following = np.where(global_vertex < ends, global_vertex + 1, global_vertex - 1)
    neighbors = np.ascontiguousarray(
        np.stack((previous, following, previous, following), axis=1).reshape(-1),
        dtype=np.int64,
    )
    neighbor_offsets = np.arange(
        0, 4 * vertex_count + 1, 4, dtype=np.int64
    )

    positions = np.empty((vertex_count, 3), dtype=np.float32)
    positions[:, 0] = local.astype(np.float32) * np.float32(0.00390625)
    positions[:, 1] = (
        island_by_vertex % np.int64(1024)
    ).astype(np.float32) * np.float32(0.015625)
    positions[:, 2] = (
        island_by_vertex // np.int64(1024)
    ).astype(np.float32) * np.float32(0.015625)
    colors = np.zeros((vertex_count, 3), dtype=np.float32)
    mask = np.zeros(vertex_count, dtype=np.float32)
    seeds = vertex_offsets[:-1]
    seed_value = np.arange(island_count, dtype=np.float32)
    colors[seeds, 0] = np.float32(0.25) + (
        seed_value % np.float32(97.0)
    ) * np.float32(0.0009765625)
    colors[seeds, 1] = np.float32(0.50) + (
        seed_value % np.float32(53.0)
    ) * np.float32(0.00048828125)
    colors[seeds, 2] = np.float32(0.75) - (
        seed_value % np.float32(31.0)
    ) * np.float32(0.00048828125)
    mask[seeds] = np.float32(1.0)

    island_offsets = np.empty(island_count + 1, dtype=np.int64)
    island_offsets[0] = 0
    np.cumsum(counts, out=island_offsets[1:])
    occurrence_count = int(island_offsets[-1])
    occurrence_starts = np.repeat(vertex_offsets[:-1], counts)
    occurrence_unique = np.repeat(unique_uncolored, counts)
    occurrence_ranks = np.arange(occurrence_count, dtype=np.int64)
    occurrence_ranks -= np.repeat(island_offsets[:-1], counts)
    island_occurrences = np.ascontiguousarray(
        occurrence_starts
        + np.int64(1)
        + occurrence_ranks % occurrence_unique,
        dtype=np.int64,
    )
    return {
        "positions": positions,
        "colors": colors,
        "mask": mask,
        "neighbor_offsets": neighbor_offsets,
        "neighbors": neighbors,
        "island_offsets": island_offsets,
        "island_occurrences": island_occurrences,
    }


def _upload_state(host_state):
    dtype = {
        "positions": "float32",
        "colors": "float32",
        "mask": "float32",
        "neighbor_offsets": "int64",
        "neighbors": "int64",
        "island_offsets": "int64",
        "island_occurrences": "int64",
    }
    theoretical_bytes = sum(value.nbytes for value in host_state.values())
    device = {}
    for name in tuple(host_state):
        device[name] = tc.tensor(host_state.pop(name), dtype=dtype[name])
        gc.collect()
    return device, theoretical_bytes


def _call(state, colors, mask, pass_count, threads):
    return tc.inpaint_island_passes(
        state["positions"],
        colors,
        mask,
        state["neighbor_offsets"],
        state["neighbors"],
        state["island_offsets"],
        state["island_occurrences"],
        pass_count,
        threads=threads,
    )


def _time_fixed(state, pass_count, threads, repeats, warmups):
    for _ in range(warmups):
        output = _call(
            state, state["colors"], state["mask"], pass_count, threads
        )
        tc.synchronize()
        del output
    walls = []
    for _ in range(repeats):
        started = time.perf_counter()
        output = _call(
            state, state["colors"], state["mask"], pass_count, threads
        )
        tc.synchronize()
        walls.append((time.perf_counter() - started) * 1000.0)
        del output
    return {
        "median_ms": statistics.median(walls),
        "mean_ms": statistics.fmean(walls),
        "min_ms": min(walls),
        "max_ms": max(walls),
        "repeats": repeats,
        "pass_count_cap_per_launch": pass_count,
    }


def _run_oracle_driver(state, passes, threads):
    colors = state["colors"]
    mask = state["mask"]
    host_uncolored = []
    outputs = None
    for _ in range(passes):
        outputs = _call(state, colors, mask, 1, threads)
        colors, mask, counts = outputs
        # The shipping oracle driver consumes this global count to update
        # smooth_count. numpy() is also the per-launch synchronization point
        # at which the host checks the material soft deadline.
        host_uncolored.append(int(counts.numpy().sum()))
    tc.synchronize()
    return outputs, host_uncolored


def _time_driver(state, passes, threads, repeats, warmups):
    for _ in range(warmups):
        output, counts = _run_oracle_driver(state, passes, threads)
        del output, counts
    walls = []
    final_counts = None
    for _ in range(repeats):
        started = time.perf_counter()
        output, final_counts = _run_oracle_driver(state, passes, threads)
        walls.append((time.perf_counter() - started) * 1000.0)
        del output
    return {
        "median_ms": statistics.median(walls),
        "mean_ms": statistics.fmean(walls),
        "min_ms": min(walls),
        "max_ms": max(walls),
        "repeats": repeats,
        "passes": passes,
        "pass_count_cap_per_launch": 1,
        "host_count_and_deadline_boundary_each_launch": True,
        "last_uncolored_occurrences_by_pass": final_counts,
    }


def run(mode, repeats=None, threads=128):
    counts, provenance = _counts(mode)
    passes = provenance["oracle_passes"]
    default_repeats = 20 if mode == "fixture" else 3
    repeats = default_repeats if repeats is None else repeats

    tc.set_alloc_pooling(False)
    tc.empty_cache()
    tc.synchronize()
    baseline = _gpu_memory_mib()
    host_state = _synthetic_state(counts)
    vertex_count = len(host_state["positions"])
    neighbor_entries = len(host_state["neighbors"])
    state, theoretical_input_bytes = _upload_state(host_state)
    tc.synchronize()
    resident = _gpu_memory_mib()
    tc.set_alloc_pooling(True)

    probe = _call(state, state["colors"], state["mask"], 1, threads)
    tc.synchronize()
    one_pass_peak = _gpu_memory_mib()
    del probe
    one_pass = _time_fixed(
        state, 1, threads, repeats, warmups=2 if mode == "fixture" else 1
    )
    fixed_pass_batch = _time_fixed(
        state, passes, threads, repeats, warmups=1
    )
    driver_probe, probe_counts = _run_oracle_driver(state, passes, threads)
    driver_peak = _gpu_memory_mib()
    del driver_probe, probe_counts
    oracle_driver = _time_driver(
        state, passes, threads, repeats, warmups=1
    )

    peak_process = max(
        one_pass_peak["process_used_mib"], driver_peak["process_used_mib"]
    )
    peak_gpu_over_baseline = max(
        one_pass_peak["gpu_total_used_mib"], driver_peak["gpu_total_used_mib"]
    ) - baseline["gpu_total_used_mib"]
    report = {
        "schema": "project-tensor-paint-cuda-3-perf-v1",
        "mode": mode,
        "mapping": "one island/warp; lane zero ordered serial executor",
        "launch_threads": threads,
        "provenance": provenance,
        "synthetic_state": {
            "active_islands": len(counts),
            "uncolored_occurrences": int(counts.sum()),
            "vertices": vertex_count,
            "neighbor_entries": neighbor_entries,
            "stable_neighbors_per_vertex": 4,
            "theoretical_resident_input_mib": theoretical_input_bytes / 2**20,
        },
        "timing_scope": (
            "resident device inputs; wrapper + output clones/allocation + "
            "ordered kernel + explicit synchronization; host build/upload excluded"
        ),
        "one_pass_launch": one_pass,
        "fixed_oracle_pass_count_one_launch": fixed_pass_batch,
        "oracle_driver_one_pass_launches": oracle_driver,
        "vram": {
            "baseline": baseline,
            "resident_inputs": resident,
            "one_pass_peak": one_pass_peak,
            "oracle_driver_peak": driver_peak,
            "peak_process_used_mib": peak_process,
            "peak_gpu_mib_over_baseline": peak_gpu_over_baseline,
            "registered_rail_mib": 3072,
            "rail_pass": peak_gpu_over_baseline <= 3072,
        },
    }

    del state
    gc.collect()
    tc.synchronize()
    tc.set_alloc_pooling(False)
    tc.empty_cache()
    tc.synchronize()
    report["post_release"] = _gpu_memory_mib()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fixture", "production"), required=True)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--threads", type=int, default=128)
    args = parser.parse_args()
    if args.repeats is not None and args.repeats <= 0:
        parser.error("--repeats must be positive")
    print(json.dumps(run(args.mode, args.repeats, args.threads), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
