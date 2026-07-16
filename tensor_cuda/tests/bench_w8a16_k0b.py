"""PAINT-Q2-K0b fixed-shape INT4 bar and W8A16 performance gate.

Run only under the shared ColdCast GPU flock. Registration mode writes the
INT4-fused/fp16 ratios that become the immutable Q-K0b-PERF bar. Passing that
JSON back with ``--bar`` evaluates both retained W8A16 launch selectors against
``W8/fp16 <= 1.2 * registered_INT4/fp16`` at all six frozen K0 families.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np

import tensor_cuda as tc


GROUP_SIZE = 32
WARMUP = 5
ITERATIONS = 21
LAUNCH_CONFIGS = ("m64n16", "m16n64")
CASES = (
    {"family": "conv_im2col_large_k8640", "shape": (512, 8640), "O": 1280},
    {"family": "conv_im2col_mid_k2880", "shape": (512, 2880), "O": 640},
    {"family": "attention_proj_320_batched", "shape": (2, 128, 320), "O": 320},
    {"family": "attention_proj_1280", "shape": (512, 1280), "O": 1280},
    {"family": "ff_expand_1280_to_5120", "shape": (512, 1280), "O": 5120},
    {"family": "ff_contract_5120_to_1280", "shape": (512, 5120), "O": 1280},
)


def _median_ms(fn):
    keep = None
    for _ in range(WARMUP):
        keep = fn()
    tc.synchronize()
    samples = []
    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        keep = fn()
        tc.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    assert keep is not None
    return float(statistics.median(samples)), samples


def _inputs(case, seed):
    rng = np.random.default_rng(seed)
    shape = tuple(case["shape"])
    k = shape[-1]
    n = int(case["O"])
    x = (rng.standard_normal(shape, dtype=np.float32) * 0.125).astype(np.float16)
    q = rng.integers(-8, 8, size=(n, k), dtype=np.int8)
    scales = rng.uniform(
        2.0e-4, 2.0e-3, size=(n, k // GROUP_SIZE)
    ).astype(np.float16)
    q_u4 = (q.astype(np.int16) + 8).astype(np.uint8)
    packed = q_u4[:, 0::2] | (q_u4[:, 1::2] << 4)
    codes = (q.astype(np.int16) + 128).astype(np.uint8)
    w = (
        q.reshape(n, k // GROUP_SIZE, GROUP_SIZE).astype(np.float32)
        * scales.astype(np.float32)[:, :, None]
    ).astype(np.float16).reshape(n, k)
    return (
        tc.tensor(np.ascontiguousarray(x), dtype="float16"),
        tc.tensor(np.ascontiguousarray(packed), dtype="uint8"),
        tc.tensor(np.ascontiguousarray(codes), dtype="uint8"),
        tc.tensor(np.ascontiguousarray(scales), dtype="float16"),
        tc.tensor(np.empty((0,), dtype=np.float16), dtype="float16"),
        tc.tensor(np.ascontiguousarray(w.T), dtype="float16"),
    )


def collect_rows():
    rows = []
    for index, case in enumerate(CASES):
        x, packed, codes, scales, zeros, w_kn = _inputs(case, 8100 + index)
        fp16_ms, fp16_samples = _median_ms(lambda: tc.matmul(x, w_kn))
        int4_ms, int4_samples = _median_ms(
            lambda: tc.int4_linear_fused(
                x, packed, scales, zeros, GROUP_SIZE
            )
        )
        w8 = {}
        for config in LAUNCH_CONFIGS:
            ms, samples = _median_ms(
                lambda config=config: tc.w8a16_matmul(x, codes, scales, config)
            )
            w8[config] = {
                "median_ms": ms,
                "over_fp16_ratio": ms / fp16_ms,
                "samples_ms": samples,
            }
        row = {
            "family": case["family"],
            "activation_shape": list(case["shape"]),
            "weight_shape": [case["O"], case["shape"][-1]],
            "group_size": GROUP_SIZE,
            "fp16_matmul_median_ms": fp16_ms,
            "int4_fused_median_ms": int4_ms,
            "int4_over_fp16_ratio": int4_ms / fp16_ms,
            "w8a16": w8,
            "warmup": WARMUP,
            "iterations": ITERATIONS,
            "fp16_samples_ms": fp16_samples,
            "int4_samples_ms": int4_samples,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bar", type=Path)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()

    rows = collect_rows()
    payload = {
        "registration": (
            "Q-K0b-PERF: W8A16/fp16 <= 1.2 * registered INT4-fused/fp16 "
            "at every frozen paint family"
        ),
        "rows": rows,
    }
    all_pass = True
    if args.bar is not None:
        bar_payload = json.loads(args.bar.read_text(encoding="utf-8"))
        bar_by_family = {row["family"]: row for row in bar_payload["rows"]}
        gates = []
        for row in rows:
            bar = float(bar_by_family[row["family"]]["int4_over_fp16_ratio"])
            for config in LAUNCH_CONFIGS:
                actual = float(row["w8a16"][config]["over_fp16_ratio"])
                passed = actual <= 1.2 * bar
                all_pass &= passed
                gates.append(
                    {
                        "family": row["family"],
                        "launch_config": config,
                        "registered_int4_over_fp16_ratio": bar,
                        "allowed_w8a16_over_fp16_ratio": 1.2 * bar,
                        "actual_w8a16_over_fp16_ratio": actual,
                        "w8a16_over_registered_int4_bar": actual / bar,
                        "pass": passed,
                    }
                )
        payload["registered_bar_source"] = str(args.bar)
        payload["gates"] = gates
        payload["all_pass"] = all_pass

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.require_pass and not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
