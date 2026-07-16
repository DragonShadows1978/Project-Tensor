"""Synchronized PAINT-CUDA-1 performance receipt (not a pytest module).

Run under the repository GPU flock:

    PYTHONPATH=. python3 tests/bench_paint_raster.py

Input upload is outside the timed region.  Each timed call includes engine
output allocation, triangle winner production, winner unpack, attribute
resolve, and an explicit device synchronization.  No target is gated.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc


def _subpixel_mesh(face_count: int, height: int, width: int):
    """Build distributed one-pixel triangles with bounded atomic contention."""

    face_index = np.arange(face_count, dtype=np.int64)
    pixel_x = 1 + face_index % (width - 2)
    pixel_y = 1 + (face_index // (width - 2)) % (height - 2)
    center_x = pixel_x.astype(np.float32) + np.float32(0.5)
    center_y = pixel_y.astype(np.float32) + np.float32(0.5)

    screen_x = np.stack(
        [
            center_x - np.float32(0.40),
            center_x + np.float32(0.40),
            center_x,
        ],
        axis=1,
    )
    screen_y = np.stack(
        [
            center_y - np.float32(0.40),
            center_y - np.float32(0.40),
            center_y + np.float32(0.40),
        ],
        axis=1,
    )
    ndc_x = np.asarray(
        np.float32(2.0)
        * np.asarray(
            (screen_x - np.float32(0.5)) / np.float32(width - 1),
            dtype=np.float32,
        )
        - np.float32(1.0),
        dtype=np.float32,
    )
    ndc_y = np.asarray(
        np.float32(2.0)
        * np.asarray(
            (screen_y - np.float32(0.5)) / np.float32(height - 1),
            dtype=np.float32,
        )
        - np.float32(1.0),
        dtype=np.float32,
    )
    # Exercise signed depth order while staying far inside int32 quantization.
    face_z = np.asarray(
        ((face_index % 257) - 128).astype(np.float32) / np.float32(96.0),
        dtype=np.float32,
    )
    clip = np.empty((face_count, 3, 4), dtype=np.float32)
    clip[:, :, 0] = ndc_x
    clip[:, :, 1] = ndc_y
    clip[:, :, 2] = face_z[:, None]
    clip[:, :, 3] = np.float32(1.0)
    faces = np.arange(face_count * 3, dtype=np.int64).reshape(face_count, 3)
    return np.ascontiguousarray(clip.reshape(-1, 4)), np.ascontiguousarray(faces)


def _time_calls(clip, faces, height, width, count):
    clip_device = tc.tensor(clip, dtype="float32")
    faces_device = tc.tensor(faces, dtype="int64")
    tc.set_alloc_pooling(True)

    warm = tc.rasterize_clip(
        clip_device,
        faces_device,
        height,
        width,
        face_threads=256,
        pixel_threads=256,
    )
    tc.synchronize()
    del warm

    walls_ms = []
    covered_pixels = None
    for _ in range(count):
        started = time.perf_counter()
        face, barycentric, depth = tc.rasterize_clip(
            clip_device,
            faces_device,
            height,
            width,
            face_threads=256,
            pixel_threads=256,
        )
        tc.synchronize()
        walls_ms.append((time.perf_counter() - started) * 1000.0)
        if covered_pixels is None:
            covered_pixels = int(np.count_nonzero(face.numpy()))
        del face, barycentric, depth

    tc.set_alloc_pooling(False)
    tc.empty_cache()
    return {
        "walls_ms": walls_ms,
        "mean_ms": statistics.fmean(walls_ms),
        "median_ms": statistics.median(walls_ms),
        "min_ms": min(walls_ms),
        "max_ms": max(walls_ms),
        "covered_pixels": covered_pixels,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-repeats", type=int, default=10)
    parser.add_argument("--production-views", type=int, default=6)
    args = parser.parse_args()
    if args.fixture_repeats <= 0 or args.production_views <= 0:
        parser.error("repeat/view counts must be positive")

    height = width = 2048
    fixture_faces = 1_852
    production_faces = 1_230_000
    fixture_clip, fixture_indices = _subpixel_mesh(
        fixture_faces, height, width
    )
    fixture = _time_calls(
        fixture_clip,
        fixture_indices,
        height,
        width,
        args.fixture_repeats,
    )

    production_clip, production_indices = _subpixel_mesh(
        production_faces, height, width
    )
    production = _time_calls(
        production_clip,
        production_indices,
        height,
        width,
        args.production_views,
    )

    report = {
        "schema": "project-tensor-paint-cuda-1-perf-v1",
        "timing_scope": (
            "resident inputs; output allocation + winner producer/unpack + "
            "resolve + cuda synchronize"
        ),
        "launch": {"face_threads": 256, "pixel_threads": 256},
        "fixture": {
            "faces": fixture_faces,
            "repeats": args.fixture_repeats,
            "height": height,
            "width": width,
            **fixture,
        },
        "production_synthetic": {
            "faces": production_faces,
            "views": args.production_views,
            "height": height,
            "width": width,
            **production,
            "six_view_total_ms": sum(production["walls_ms"]),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
