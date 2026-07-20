"""ORDER INT6 acceptance gates with compact, receipt-friendly output.

Run from ``tensor_cuda`` with the repository package first on ``PYTHONPATH``.
The 151936-output cases are streamed in row chunks so neither the host nor GPU
ever holds a full vocabulary-shaped dequantized weight.
"""

from __future__ import annotations

import math
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
BITS = 6
REAL_K = (2048, 6144)
REAL_M = (2048, 151936)
ROW_CHUNK = 4096


def _empty_zeros():
    return tc.tensor(np.empty((0,), dtype=np.float16), dtype="float16")


def packing_gate():
    worked_codes = np.array([[1, 2, 3, 4]], dtype=np.uint8)
    worked = pack_lowbit(worked_codes, BITS)
    expected = np.array([[0x81, 0x30, 0x10]], dtype=np.uint8)
    np.testing.assert_array_equal(worked, expected)

    codes = np.tile(np.arange(64, dtype=np.uint8), (2, 4))
    packed = pack_lowbit(codes, BITS)
    np.testing.assert_array_equal(unpack_lowbit(packed, BITS, 256), codes)
    print(
        "[PACK] PASS worked=[0x81,0x30,0x10] "
        f"all_codes_roundtrip={codes.size} packed_bytes={packed.size}"
    )


def dequant_exactness_gate():
    base = np.tile(np.arange(64, dtype=np.uint8), 4)
    codes = np.stack((base, base[::-1])).astype(np.uint8)
    scales = np.array(
        [[0.01973, -0.00731], [-0.04321, 0.01117]],
        dtype=np.float16,
    )
    expected = (
        (codes.reshape(2, 2, GROUP).astype(np.float32) - 32.0)
        * scales.astype(np.float32)[:, :, None]
    ).reshape(2, 256)
    actual = tc.int6_dequant(
        tc.tensor(pack_lowbit(codes, BITS), dtype="uint8"),
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
        "[DEQUANT] PASS max|delta|="
        f"{max_abs:.1f} boundary_max={boundary_max:.1f} "
        "codes=0..63 scale_signs=+/- groups=2"
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
    assert rel_fro < 0.08
    print(
        f"[ROUNDTRIP {name}] PASS max|delta|={max_abs:.8e} "
        f"rel-fro={rel_fro:.8e}"
    )


def roundtrip_gate():
    rng = np.random.default_rng(6100)
    gaussian = rng.standard_normal((256, 2048)).astype(np.float32) * 0.1
    outlier_heavy = rng.standard_normal((256, 2048)).astype(np.float32) * 0.1
    mask = rng.random(outlier_heavy.shape) < 0.02
    outlier_heavy[mask] += (
        rng.standard_normal(int(mask.sum())).astype(np.float32) * 8.0
    )
    _roundtrip_case("gaussian", gaussian)
    _roundtrip_case("outlier-heavy", outlier_heavy)


def _gemv_case(k: int, m: int):
    x_rng = np.random.default_rng(6200 + k)
    x = (x_rng.standard_normal((1, k)).astype(np.float32) * 0.125).astype(
        np.float16
    )
    x_t = tc.tensor(x, dtype="float16")
    zeros_t = _empty_zeros()
    err_sq = 0.0
    ref_sq = 0.0
    max_abs = 0.0
    max_ref = 0.0
    chunks = math.ceil(m / ROW_CHUNK)
    started = time.monotonic()

    for chunk_index, row0 in enumerate(range(0, m, ROW_CHUNK)):
        rows = min(ROW_CHUNK, m - row0)
        rng = np.random.default_rng(6300 + k * 1000003 + row0)
        codes = rng.integers(0, 64, size=(rows, k), dtype=np.uint8)
        scales = (
            rng.random((rows, k // GROUP), dtype=np.float32) * 0.02
            + 2.0**-12
        ).astype(np.float16)
        packed = pack_lowbit(codes, BITS)
        del codes

        with tc.no_grad():
            packed_t = tc.tensor(packed, dtype="uint8")
            scales_t = tc.tensor(scales, dtype="float16")
            w_kn = tc.int6_dequant(
                packed_t, scales_t, zeros_t, GROUP, out_dtype="float16"
            )
            reference = tc.matmul(x_t, w_kn).numpy().astype(np.float32)
            direct = tc.int6_linear_fused(
                x_t, packed_t, scales_t, zeros_t, GROUP
            ).numpy().astype(np.float32)

        delta = direct - reference
        err_sq += float(np.dot(delta.ravel(), delta.ravel()))
        ref_sq += float(np.dot(reference.ravel(), reference.ravel()))
        max_abs = max(max_abs, float(np.abs(delta).max()))
        max_ref = max(max_ref, float(np.abs(reference).max()))
        del packed_t, scales_t, w_kn, reference, direct, delta, packed, scales
        if (chunk_index % 8) == 7:
            tc.empty_cache()

    tc.synchronize()
    tc.empty_cache()
    rel_fro = math.sqrt(err_sq / max(ref_sq, 1.0e-30))
    max_rel = max_abs / max(max_ref, 1.0e-30)
    elapsed = time.monotonic() - started
    assert rel_fro < 2.0e-3, (k, m, rel_fro)
    assert max_rel < 2.0e-3, (k, m, max_rel)
    print(
        f"[GEMV K={k} M={m}] PASS chunks={chunks} "
        f"max|delta|={max_abs:.8e} rel-fro={rel_fro:.8e} "
        f"max-rel={max_rel:.8e} seconds={elapsed:.3f}"
    )


def gemv_real_shape_gate():
    for k in REAL_K:
        for m in REAL_M:
            _gemv_case(k, m)


def run():
    packing_gate()
    roundtrip_gate()
    dequant_exactness_gate()
    gemv_real_shape_gate()
    print("INT6 WEIGHT GATES: PASS")


if __name__ == "__main__":
    run()
