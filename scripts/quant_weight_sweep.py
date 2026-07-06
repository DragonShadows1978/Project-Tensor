#!/usr/bin/env python3
"""Run a deterministic low-bit weight sweep on TensorCUDA kernels.

This is a kernel-level sweep, not a model-PPL benchmark. It measures the
current affine group-quantized weight paths against a BF16/FP16/FP32 dense
linear reference on structured layer-like matrices.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tensor_cuda as tc
from tensor_cuda.quantization import (
    dequantize_affine_per_group,
    quantize_affine_per_group,
)


@dataclass(frozen=True)
class ShapeSpec:
    name: str
    m: int
    n: int
    k: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep BF16 dense vs INT4/INT3/INT2 TensorCUDA linear paths."
    )
    parser.add_argument("--quick", action="store_true", help="Run a small sweep.")
    parser.add_argument("--reps", type=int, default=20, help="Timing reps per mode.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup reps per mode.")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument(
        "--compute-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults under artifacts/quant_sweep.",
    )
    return parser.parse_args()


def shape_suite(quick: bool) -> list[ShapeSpec]:
    if quick:
        return [
            ShapeSpec("decode_mid", 1, 2048, 4096),
            ShapeSpec("prefill_mid", 16, 2048, 4096),
        ]
    return [
        ShapeSpec("decode_small", 1, 1024, 1024),
        ShapeSpec("prefill_small", 16, 1024, 1024),
        ShapeSpec("decode_mid", 1, 2048, 4096),
        ShapeSpec("prefill_mid", 16, 2048, 4096),
        ShapeSpec("decode_wide", 1, 4096, 4096),
        ShapeSpec("prefill_wide", 16, 4096, 4096),
    ]


def structured_weight(rng: np.random.Generator, n: int, k: int) -> np.ndarray:
    """Make deterministic nontrivial weights with channel outliers and drift."""
    base = rng.standard_normal((n, k), dtype=np.float32) * np.float32(0.075)
    row_gain = rng.lognormal(mean=0.0, sigma=0.18, size=(n, 1)).astype(np.float32)
    col_gain = rng.lognormal(mean=0.0, sigma=0.12, size=(1, k)).astype(np.float32)
    w = base * row_gain * col_gain

    outlier_cols = max(1, k // 128)
    outlier_rows = max(1, n // 128)
    cols = rng.choice(k, size=outlier_cols, replace=False)
    rows = rng.choice(n, size=outlier_rows, replace=False)
    w[:, cols] *= np.float32(3.5)
    w[rows, :] *= np.float32(1.75)

    # Add a weak low-frequency component so the distribution is not pure noise.
    x = np.linspace(0.0, 2.0 * np.pi, k, dtype=np.float32)
    phases = rng.random((n, 1), dtype=np.float32) * np.float32(2.0 * np.pi)
    w += np.sin(x[None, :] + phases) * np.float32(0.004)
    return w.astype(np.float32, copy=False)


def structured_input(rng: np.random.Generator, m: int, k: int) -> np.ndarray:
    x = rng.standard_normal((m, k), dtype=np.float32) * np.float32(0.35)
    if m > 1:
        trend = np.linspace(-0.1, 0.1, m, dtype=np.float32)[:, None]
        x += trend
    return x.astype(np.float32, copy=False)


def to_tc(array: np.ndarray, dtype: str):
    return tc.tensor(array, dtype=dtype)


def sync() -> None:
    tc.synchronize()


def time_call(fn, warmup: int, reps: int) -> tuple[float, object]:
    out = None
    for _ in range(max(0, warmup)):
        out = fn()
    sync()
    start = time.perf_counter()
    for _ in range(max(1, reps)):
        out = fn()
    sync()
    elapsed = time.perf_counter() - start
    return (elapsed * 1000.0) / max(1, reps), out


def output_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    cand = candidate.astype(np.float32, copy=False)
    ref = reference.astype(np.float32, copy=False)
    diff = cand - ref
    rmse = float(np.sqrt(np.mean(diff * diff)))
    ref_rms = float(np.sqrt(np.mean(ref * ref)))
    mae = float(np.mean(np.abs(diff)))
    max_abs = float(np.max(np.abs(diff)))
    denom = float(np.linalg.norm(cand.ravel()) * np.linalg.norm(ref.ravel()))
    cosine = float(np.dot(cand.ravel(), ref.ravel()) / denom) if denom else 1.0
    top1 = float(np.mean(np.argmax(cand, axis=-1) == np.argmax(ref, axis=-1)))
    return {
        "mae": mae,
        "max_abs": max_abs,
        "rmse": rmse,
        "relative_rmse": rmse / (ref_rms + 1e-12),
        "cosine": cosine,
        "top1_agreement": top1,
    }


def weight_metrics(dequant: np.ndarray, weight: np.ndarray) -> dict[str, float]:
    diff = dequant.astype(np.float32, copy=False) - weight.astype(np.float32, copy=False)
    rmse = float(np.sqrt(np.mean(diff * diff)))
    wrms = float(np.sqrt(np.mean(weight.astype(np.float32, copy=False) ** 2)))
    return {
        "weight_mae": float(np.mean(np.abs(diff))),
        "weight_rmse": rmse,
        "weight_relative_rmse": rmse / (wrms + 1e-12),
    }


def memory_bytes(n: int, k: int, packed: np.ndarray, scales: np.ndarray, zeros: np.ndarray) -> dict[str, float]:
    dense_bf16 = n * k * 2
    quant = int(packed.nbytes + scales.nbytes + zeros.nbytes)
    return {
        "dense_bf16_bytes": dense_bf16,
        "quant_bytes": quant,
        "compression_vs_bf16": dense_bf16 / quant if quant else math.inf,
    }


def run_shape(
    spec: ShapeSpec,
    rng: np.random.Generator,
    compute_dtype: str,
    group_size: int,
    warmup: int,
    reps: int,
) -> dict:
    if spec.k % group_size != 0:
        raise ValueError(f"{spec.name}: K={spec.k} must divide group_size={group_size}")

    print(
        f"running shape={spec.name} M={spec.m} N={spec.n} K={spec.k}",
        flush=True,
    )
    weight = structured_weight(rng, spec.n, spec.k)
    x = structured_input(rng, spec.m, spec.k)
    x_t = to_tc(x, compute_dtype)
    w_t = to_tc(weight, compute_dtype)

    dense_ms, dense_out = time_call(
        lambda: tc.matmul(x_t, w_t, trans_b=True), warmup, reps
    )
    dense_np = dense_out.numpy().astype(np.float32)

    results = {
        "shape": asdict(spec),
        "compute_dtype": compute_dtype,
        "group_size": group_size,
        "dense": {
            "mode": "bf16_dense" if compute_dtype == "bfloat16" else f"{compute_dtype}_dense",
            "latency_ms": dense_ms,
            "memory": {
                "dense_bf16_bytes": spec.n * spec.k * 2,
                "quant_bytes": spec.n * spec.k * 2,
                "compression_vs_bf16": 1.0,
            },
        },
        "quantized": [],
    }

    for bits in (4, 3, 2):
        quant_start = time.perf_counter()
        q = quantize_affine_per_group(weight, bits, group_size)
        quantize_ms = (time.perf_counter() - quant_start) * 1000.0
        print(
            f"  int{bits}: packed in {quantize_ms:.1f} ms; timing fused kernel",
            flush=True,
        )
        packed_t = tc.tensor(q.packed, dtype="uint8")
        scales_t = tc.tensor(q.scales, dtype="float16")
        zeros_t = tc.tensor(q.zeros, dtype="float16")

        if bits == 4:
            fn = lambda: tc.int4_linear_fused(x_t, packed_t, scales_t, zeros_t, group_size)
        else:
            fn = lambda: tc.intn_linear_fused(
                x_t, packed_t, scales_t, zeros_t, bits, spec.k, group_size
            )

        latency_ms, out = time_call(fn, warmup, reps)
        out_np = out.numpy().astype(np.float32)
        dequant = dequantize_affine_per_group(
            q.packed, q.scales, q.zeros, bits, spec.k, group_size
        )

        results["quantized"].append(
            {
                "bits": bits,
                "quantize_ms": quantize_ms,
                "latency_ms": latency_ms,
                "speedup_vs_dense": dense_ms / latency_ms if latency_ms else math.inf,
                "memory": memory_bytes(spec.n, spec.k, q.packed, q.scales, q.zeros),
                "output": output_metrics(out_np, dense_np),
                "weight": weight_metrics(dequant, weight),
            }
        )

    return results


def artifact_path(output: Path | None) -> Path:
    if output is not None:
        return output
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("artifacts") / "quant_sweep" / f"quant_weight_sweep_{stamp}.json"


def print_summary(payload: dict) -> None:
    print("Quant weight sweep")
    print(
        f"compute_dtype={payload['compute_dtype']} group_size={payload['group_size']} "
        f"reps={payload['reps']} wall_s={payload.get('total_wall_seconds', 0.0):.2f}"
    )
    print("")
    header = (
        "shape", "bits", "q_ms", "lat_ms", "speedup", "MiB", "xBF16",
        "rel_rmse", "cosine", "top1", "w_rel_rmse",
    )
    print(
        f"{header[0]:<14} {header[1]:>4} {header[2]:>9} {header[3]:>9} "
        f"{header[4]:>8} {header[5]:>8} {header[6]:>7} {header[7]:>10} "
        f"{header[8]:>9} {header[9]:>7} {header[10]:>11}"
    )
    for shape in payload["results"]:
        dense = shape["dense"]
        dense_mib = dense["memory"]["dense_bf16_bytes"] / (1024.0 * 1024.0)
        print(
            f"{shape['shape']['name']:<14} {'bf16':>4} {0.0:9.1f} "
            f"{dense['latency_ms']:9.4f} {1.0:8.2f} {dense_mib:8.2f} "
            f"{1.0:7.2f} {0.0:10.4f} {1.0:9.6f} {1.0:7.3f} "
            f"{0.0:11.4f}"
        )
        for item in shape["quantized"]:
            mem = item["memory"]
            out = item["output"]
            wt = item["weight"]
            mib = mem["quant_bytes"] / (1024.0 * 1024.0)
            print(
                f"{shape['shape']['name']:<14} {('int' + str(item['bits'])):>4} "
                f"{item['quantize_ms']:9.1f} {item['latency_ms']:9.4f} "
                f"{item['speedup_vs_dense']:8.2f} {mib:8.2f} "
                f"{mem['compression_vs_bf16']:7.2f} {out['relative_rmse']:10.4f} "
                f"{out['cosine']:9.6f} {out['top1_agreement']:7.3f} "
                f"{wt['weight_relative_rmse']:11.4f}"
            )


def main() -> int:
    args = parse_args()
    if args.reps < 1:
        raise ValueError("--reps must be >= 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")

    rng = np.random.default_rng(args.seed)
    tc.empty_cache()
    wall_start = time.perf_counter()
    payload = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "quick": args.quick,
        "compute_dtype": args.compute_dtype,
        "group_size": args.group_size,
        "warmup": args.warmup,
        "reps": args.reps,
        "results": [],
    }

    for spec in shape_suite(args.quick):
        payload["results"].append(
            run_shape(
                spec,
                rng,
                args.compute_dtype,
                args.group_size,
                args.warmup,
                args.reps,
            )
        )
        tc.empty_cache()

    out = artifact_path(args.output)
    payload["total_wall_seconds"] = time.perf_counter() - wall_start
    payload["artifact"] = str(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print_summary(payload)
    print("")
    print(f"artifact={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
