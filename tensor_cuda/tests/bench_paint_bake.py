"""Synchronized PAINT-CUDA-2 kernel and VRAM report.

Run under the shared GPU flock, for example:

    flock /mnt/ForgeRealm/ColdCast/.gpu_lock \
      env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
      python3 tests/bench_paint_bake.py --mode production

Host construction/upload is excluded.  Inputs are resident, output allocations
use TensorCUDA's transient pool, and every timed call is explicitly synced.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc


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


def _upload(array, dtype):
    result = tc.tensor(array, dtype=dtype)
    del array
    gc.collect()
    return result


def _full(shape, value, np_dtype, tc_dtype):
    return _upload(np.full(shape, value, dtype=np_dtype), tc_dtype)


def _positions(sample_count, view_side):
    flat = np.arange(sample_count, dtype=np.uint32)
    positions = np.empty((sample_count, 4), dtype=np.float32)
    columns = (flat % np.uint32(view_side)).astype(np.float32)
    rows = ((flat // np.uint32(view_side)) % np.uint32(view_side)).astype(
        np.float32
    )
    positions[:, 0] = (
        (columns + np.float32(0.3125)) / np.float32(view_side)
    ) * np.float32(2.0) - np.float32(1.0)
    positions[:, 1] = (
        (rows + np.float32(0.6875)) / np.float32(view_side)
    ) * np.float32(2.0) - np.float32(1.0)
    positions[:, 2] = 0.0
    positions[:, 3] = 1.0
    del flat, columns, rows
    return positions


def _timings(call, repeats, warmups):
    for _ in range(warmups):
        output = call()
        tc.synchronize()
        del output
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        output = call()
        tc.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
        del output
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "repeats": repeats,
    }


def _release():
    gc.collect()
    tc.synchronize()
    tc.set_alloc_pooling(False)
    tc.empty_cache()
    tc.synchronize()


def run(mode, repeats=None):
    if mode == "fixture":
        atlas_side = 64
        view_side = 64
        default_repeats = 30
    else:
        atlas_side = 4096
        view_side = 2048
        default_repeats = 5
    repeats = default_repeats if repeats is None else repeats
    views = 6
    channels = 3
    sample_count = atlas_side * atlas_side

    tc.set_alloc_pooling(False)
    tc.empty_cache()
    tc.synchronize()
    baseline = _gpu_memory_mib()

    # Back-project stage: atlas positions persist while one view and its CPU
    # reliability/cosine products are streamed.
    positions = _upload(_positions(sample_count, view_side), "float32")
    view = _full((view_side, view_side, channels), 0.625, np.float32, "float32")
    depth = _full((view_side, view_side), 0.0, np.float32, "float32")
    reliable = _full((view_side, view_side), 1, np.uint8, "uint8")
    cosine_map = _full((view_side, view_side), 0.8125, np.float32, "float32")
    identity = _upload(np.eye(4, dtype=np.float32), "float32")
    image_projection = _upload(np.eye(4, dtype=np.float32), "float32")
    back_inputs = _gpu_memory_mib()
    tc.set_alloc_pooling(True)

    def call_back_project():
        return tc.bake_back_project(
            positions,
            view,
            depth,
            reliable,
            cosine_map,
            identity,
            image_projection,
            np.float32(3.0e-3),
            threads=256,
        )

    back_probe = call_back_project()
    tc.synchronize()
    back_peak = _gpu_memory_mib()
    del back_probe
    back_timing = _timings(
        call_back_project, repeats=repeats, warmups=2 if mode == "fixture" else 1
    )
    del positions, view, depth, reliable, cosine_map, identity, image_projection
    _release()

    # Blend stage: synthetic resident copies of the six aligned per-view
    # outputs.  No view-sized weight intermediate is allocated by the op.
    view_colors = _full(
        (views, sample_count, channels), 0.625, np.float32, "float32"
    )
    view_cosine = _full(
        (views, sample_count), 0.8125, np.float32, "float32"
    )
    view_valid = _full((views, sample_count), 1, np.uint8, "uint8")
    view_weights = _upload(
        np.asarray([1.0, 0.1, 0.5, 0.1, 0.05, 0.05], dtype=np.float32),
        "float32",
    )
    view_enabled = _upload(np.ones(views, dtype=np.uint8), "uint8")
    blend_inputs = _gpu_memory_mib()
    tc.set_alloc_pooling(True)

    def call_blend():
        return tc.bake_cosine_blend(
            view_colors,
            view_cosine,
            view_valid,
            view_weights,
            view_enabled,
            threads=256,
        )

    blend_probe = call_blend()
    tc.synchronize()
    blend_peak = _gpu_memory_mib()
    del blend_probe
    blend_timing = _timings(
        call_blend, repeats=repeats, warmups=2 if mode == "fixture" else 1
    )
    del view_colors, view_cosine, view_valid, view_weights, view_enabled
    _release()

    peak_process = max(
        back_peak["process_used_mib"], blend_peak["process_used_mib"]
    )
    peak_over_baseline = max(
        back_peak["gpu_total_used_mib"], blend_peak["gpu_total_used_mib"]
    ) - baseline["gpu_total_used_mib"]
    per_material_ms = back_timing["median_ms"] * views + blend_timing["median_ms"]
    return {
        "mode": mode,
        "provenance_faces": 1_230_000 if mode == "production" else 1_852,
        "views": views,
        "view_shape": [view_side, view_side, channels],
        "atlas_shape": [atlas_side, atlas_side],
        "atlas_samples": sample_count,
        "method": (
            "resident inputs; synchronized wrapper+allocation+kernel wall; "
            "per-material = 6 * median back-project/view + median blend"
        ),
        "back_project_per_view": back_timing,
        "cosine_blend": blend_timing,
        "per_material_bake_kernel_ms": per_material_ms,
        "vram": {
            "baseline": baseline,
            "back_inputs": back_inputs,
            "back_peak": back_peak,
            "blend_inputs": blend_inputs,
            "blend_peak": blend_peak,
            "peak_process_used_mib": peak_process,
            "peak_gpu_mib_over_baseline": peak_over_baseline,
            "registered_rail_mib": 3072,
            "rail_pass": peak_over_baseline <= 3072,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fixture", "production"), required=True)
    parser.add_argument("--repeats", type=int)
    args = parser.parse_args()
    if args.repeats is not None and args.repeats <= 0:
        parser.error("--repeats must be positive")
    print(json.dumps(run(args.mode, args.repeats), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
