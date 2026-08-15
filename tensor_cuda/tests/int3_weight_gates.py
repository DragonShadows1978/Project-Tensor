"""ORDER INT3 acceptance gates with explicit CPU/GPU receipt modes.

Run from ``tensor_cuda`` with the repository package first on ``PYTHONPATH``:

    PYTHONPATH=. python tests/int3_weight_gates.py --cpu-only
    PYTHONPATH=. python tests/int3_weight_gates.py --gpu-only

The GPU gate allocates one Qwen-shaped weight at a time and reports both the
packed INT3 and dense FP16 memory shapes. It is kernel characterization and
dense-reference parity only; it makes no model-quality claim.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import tensor_cuda as tc
from tensor_cuda.quantization import (
    dequantize_symmetric_per_group,
    pack_lowbit,
    quantize_symmetric_per_group,
    unpack_lowbit,
)


GROUP = 128
BITS = 3
# Representative Qwen3.8-27B projection orientations. The mandated input
# widths are the gate key; each full N x K matrix fits comfortably on 12 GB.
QWEN_GEMV_SHAPES = (
    (5120, 1024),
    (5120, 2048),
    (6144, 5120),
    (5120, 6144),
    (5120, 17408),
)


def _empty_zeros():
    return tc.tensor(np.empty((0,), dtype=np.float16), dtype="float16")


def packing_gate():
    worked_codes = np.arange(8, dtype=np.uint8).reshape(1, 8)
    worked = pack_lowbit(worked_codes, BITS)
    expected = np.array([[0x88, 0xC6, 0xFA]], dtype=np.uint8)
    np.testing.assert_array_equal(worked, expected)

    codes = np.stack(
        (
            np.tile(np.arange(8, dtype=np.uint8), 32),
            np.tile(np.arange(7, -1, -1, dtype=np.int16).astype(np.uint8), 32),
        )
    )
    packed = pack_lowbit(codes, BITS)
    np.testing.assert_array_equal(unpack_lowbit(packed, BITS, 256), codes)
    assert packed.shape == (2, 96)
    print(
        "[PACK CPU] PASS worked=[0x88,0xC6,0xFA] "
        f"all_codes_roundtrip={codes.size} packed_bytes={packed.size}"
    )


def _roundtrip_case(name: str, weights: np.ndarray):
    quant = quantize_symmetric_per_group(weights, BITS, GROUP)
    dequant = dequantize_symmetric_per_group(
        quant.packed,
        quant.scales,
        quant.bits,
        quant.in_features,
        quant.group_size,
    )
    delta = dequant - weights
    max_abs = float(np.abs(delta).max())
    rel_fro = float(np.linalg.norm(delta) / np.linalg.norm(weights))
    assert quant.zeros.shape == (0,)
    assert np.isfinite(max_abs) and np.isfinite(rel_fro)
    print(
        f"[ROUNDTRIP {name} CPU] PASS characterization "
        f"max|delta|={max_abs:.8e} rel-fro={rel_fro:.8e}"
    )


def roundtrip_gate():
    rng = np.random.default_rng(3100)
    gaussian = rng.standard_normal((256, 2048)).astype(np.float32) * 0.1
    outlier_heavy = rng.standard_normal((256, 2048)).astype(np.float32) * 0.1
    mask = rng.random(outlier_heavy.shape) < 0.02
    outlier_heavy[mask] += (
        rng.standard_normal(int(mask.sum())).astype(np.float32) * 8.0
    )
    _roundtrip_case("gaussian", gaussian)
    _roundtrip_case("outlier-heavy", outlier_heavy)


def dequant_exactness_gate():
    base = np.tile(np.arange(8, dtype=np.uint8), 16)
    codes = np.stack((np.tile(base, 2), np.tile(base[::-1], 2)))
    scales = np.array(
        [[0.01973, -0.00731], [-0.04321, 0.01117]], dtype=np.float16
    )
    packed = pack_lowbit(codes, BITS)
    expected = dequantize_symmetric_per_group(
        packed, scales, BITS, 256, GROUP
    )
    actual = tc.int3_dequant(
        tc.tensor(packed, dtype="uint8"),
        tc.tensor(scales, dtype="float16"),
        _empty_zeros(),
        GROUP,
        out_dtype="float32",
    ).numpy().T
    diff = np.abs(actual - expected)
    max_abs = float(diff.max())
    boundary_max = float(diff[:, [127, 128]].max())
    assert max_abs == 0.0
    assert boundary_max == 0.0
    print(
        "[DEQUANT GPU] PASS max|delta|="
        f"{max_abs:.1f} boundary_max={boundary_max:.1f} "
        "codes=0..7 scale_signs=+/- groups=2"
    )


def _gemv_case(n: int, k: int):
    rng = np.random.default_rng(3300 + k)
    x = (rng.standard_normal((1, k)).astype(np.float32) * 0.125).astype(
        np.float16
    )
    codes = rng.integers(0, 8, size=(n, k), dtype=np.uint8)
    scales = (
        rng.random((n, k // GROUP), dtype=np.float32) * 0.02 + 2.0**-12
    ).astype(np.float16)
    packed = pack_lowbit(codes, BITS)
    packed_mib = packed.nbytes / (1024.0**2)
    dense_mib = n * k * np.dtype(np.float16).itemsize / (1024.0**2)

    x_t = tc.tensor(x, dtype="float16")
    packed_t = tc.tensor(packed, dtype="uint8")
    scales_t = tc.tensor(scales, dtype="float16")
    zeros_t = _empty_zeros()
    with tc.no_grad():
        started = time.monotonic()
        w_kn = tc.int3_dequant(
            packed_t, scales_t, zeros_t, GROUP, out_dtype="float16"
        )
        reference = tc.matmul(x_t, w_kn)
        tc.synchronize()
        reference_seconds = time.monotonic() - started

        started = time.monotonic()
        direct = tc.int3_linear(x_t, packed_t, scales_t, zeros_t, GROUP)
        tc.synchronize()
        gemv_seconds = time.monotonic() - started

        fused = tc.int3_linear_fused(x_t, packed_t, scales_t, zeros_t, GROUP)
        tc.synchronize()

    reference_np = reference.numpy().astype(np.float32)
    direct_np = direct.numpy().astype(np.float32)
    fused_np = fused.numpy().astype(np.float32)
    delta = direct_np - reference_np
    err_sq = float(np.dot(delta.ravel(), delta.ravel()))
    ref_sq = float(np.dot(reference_np.ravel(), reference_np.ravel()))
    max_abs = float(np.abs(delta).max())
    rel_fro = float(np.sqrt(err_sq / max(ref_sq, 1.0e-30)))
    max_rel = max_abs / max(float(np.abs(reference_np).max()), 1.0e-30)
    fused_max = float(np.abs(fused_np - direct_np).max())
    assert rel_fro < 3.0e-3, (n, k, rel_fro)
    assert max_rel < 3.0e-3, (n, k, max_rel)
    assert fused_max == 0.0, (n, k, fused_max)
    print(
        f"[GEMV N={n} K={k} GPU] PASS max|delta|={max_abs:.8e} "
        f"rel-fro={rel_fro:.8e} max-rel={max_rel:.8e} "
        f"fused-vs-direct-max={fused_max:.1f} packed-miB={packed_mib:.3f} "
        f"dense-fp16-miB={dense_mib:.3f} gemv-seconds={gemv_seconds:.6f} "
        f"dequant-matmul-seconds={reference_seconds:.6f}"
    )


def gemv_real_shape_gate():
    for n, k in QWEN_GEMV_SHAPES:
        _gemv_case(n, k)


def run_cpu():
    packing_gate()
    roundtrip_gate()
    print("INT3 CPU WEIGHT GATES: PASS")


def run_gpu():
    dequant_exactness_gate()
    gemv_real_shape_gate()
    print("INT3 GPU WEIGHT GATES: PASS")


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--cpu-only", action="store_true")
    mode.add_argument("--gpu-only", action="store_true")
    args = parser.parse_args()
    if not args.gpu_only:
        run_cpu()
    if not args.cpu_only:
        run_gpu()


if __name__ == "__main__":
    main()
