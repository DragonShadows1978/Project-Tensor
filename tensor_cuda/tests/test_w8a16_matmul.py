"""PAINT-Q2-K0 gates for group-32 symmetric W8A16 fused GEMM.

The registered accuracy gate is deliberately measurement-only: it emits
max-abs, relative Frobenius error, and cosine against the explicit two-stage
FP16-weight reference without inventing acceptance thresholds.  The lead owns
those thresholds after seeing this first spread.

Trinity-compatible storage contract:
  codes[O,K] uint8, q = code - 128
  scales[O,K/32] float16
  weight[o,k] = fp16(float32(q) * float32(scale[o,k//32]))

K must be positive and divisible by 32.  Leading activation dimensions are
flattened for the launch and restored on output, so a shared [O,K] weight works
for ordinary linears, batched linears, and im2col conv matrices alike.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import gc
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np
import pytest

import tensor_cuda as tc


GROUP_SIZE = 32
LAUNCH_CONFIGS = ("m64n16", "m16n64")

# Frozen before implementation: families called out by PAINT-Q2-SCOUT/K0.
# Shape is activation (..., K); O is the logical [O,K] weight row count.
ORACLE_SHAPES = (
    {"family": "conv_im2col_large_k8640", "shape": (512, 8640), "O": 1280},
    {"family": "conv_im2col_mid_k2880", "shape": (512, 2880), "O": 640},
    {"family": "attention_proj_320_batched", "shape": (2, 128, 320), "O": 320},
    {"family": "attention_proj_1280", "shape": (512, 1280), "O": 1280},
    {"family": "ff_expand_1280_to_5120", "shape": (512, 1280), "O": 5120},
    {"family": "ff_contract_5120_to_1280", "shape": (512, 5120), "O": 1280},
)
DETERMINISM_SHAPE = {"family": "attention_batched_det", "shape": (2, 17, 1280), "O": 1280}
VRAM_SHAPE = ORACLE_SHAPES[0]
DETERMINISM_RUNS = 5


def _case_arrays(case: dict, seed: int):
    rng = np.random.default_rng(seed)
    shape = tuple(case["shape"])
    k = shape[-1]
    o = int(case["O"])
    assert k % GROUP_SIZE == 0
    x = (rng.standard_normal(shape, dtype=np.float32) * 0.125).astype(np.float16)
    codes = rng.integers(0, 256, size=(o, k), dtype=np.uint8)
    scales = rng.uniform(2.0e-4, 2.0e-3, size=(o, k // GROUP_SIZE)).astype(np.float16)
    return x, codes, scales


def _device_inputs(case: dict, seed: int):
    x, codes, scales = _case_arrays(case, seed)
    return (
        tc.tensor(np.ascontiguousarray(x), dtype="float16"),
        tc.tensor(np.ascontiguousarray(codes), dtype="uint8"),
        tc.tensor(np.ascontiguousarray(scales), dtype="float16"),
        x,
        codes,
        scales,
    )


def _dequant_fp16(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    o, k = codes.shape
    q = codes.reshape(o, k // GROUP_SIZE, GROUP_SIZE).astype(np.int16) - 128
    w = q.astype(np.float32) * scales.astype(np.float32)[:, :, None]
    return w.astype(np.float16).reshape(o, k)


def _metrics(got: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    a = got.astype(np.float64, copy=False).reshape(-1)
    b = ref.astype(np.float64, copy=False).reshape(-1)
    d = a - b
    nb = float(np.linalg.norm(b))
    na = float(np.linalg.norm(a))
    return {
        "max_abs": float(np.max(np.abs(d), initial=0.0)),
        "rel_fro": float(np.linalg.norm(d) / nb) if nb else 0.0,
        "cosine": float(np.dot(a, b) / (na * nb)) if na and nb else 1.0,
    }


def collect_oracle_report() -> list[dict]:
    rows = []
    for index, case in enumerate(ORACLE_SHAPES):
        x_t, codes_t, scales_t, _x, codes, scales = _device_inputs(case, 7000 + index)
        w_nk = _dequant_fp16(codes, scales)
        w_kn_t = tc.tensor(np.ascontiguousarray(w_nk.T), dtype="float16")
        ref = tc.matmul(x_t, w_kn_t).numpy()
        for launch_config in LAUNCH_CONFIGS:
            got = tc.w8a16_matmul(x_t, codes_t, scales_t, launch_config).numpy()
            row = {
                "family": case["family"],
                "activation_shape": list(case["shape"]),
                "weight_shape": [case["O"], case["shape"][-1]],
                "launch_config": launch_config,
                **_metrics(got, ref),
            }
            assert all(math.isfinite(row[key]) for key in ("max_abs", "rel_fro", "cosine"))
            rows.append(row)
    return rows


def collect_determinism_report() -> list[dict]:
    x_t, codes_t, scales_t, *_ = _device_inputs(DETERMINISM_SHAPE, 7100)
    rows = []
    for launch_config in LAUNCH_CONFIGS:
        payloads = [
            tc.w8a16_matmul(x_t, codes_t, scales_t, launch_config).numpy().tobytes()
            for _ in range(DETERMINISM_RUNS)
        ]
        rows.append(
            {
                "family": DETERMINISM_SHAPE["family"],
                "launch_config": launch_config,
                "runs": DETERMINISM_RUNS,
                "byte_equal": all(p == payloads[0] for p in payloads[1:]),
                "bytes_per_output": len(payloads[0]),
            }
        )
    return rows


class _CudaPoolStats:
    """Read CUDA default-pool used-memory counters without another framework."""

    USED_CURRENT = 7
    USED_HIGH = 8

    def __init__(self):
        candidates = [ctypes.util.find_library("cudart"), "libcudart.so"]
        error = None
        for candidate in candidates:
            if not candidate:
                continue
            try:
                self.lib = ctypes.CDLL(candidate)
                break
            except OSError as exc:
                error = exc
        else:
            raise RuntimeError(f"could not load libcudart: {error}")
        self.lib.cudaDeviceGetDefaultMemPool.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
        self.lib.cudaMemPoolGetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        self.lib.cudaMemPoolSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        self.pool = ctypes.c_void_p()
        self._check(self.lib.cudaDeviceGetDefaultMemPool(ctypes.byref(self.pool), 0), "get default pool")

    @staticmethod
    def _check(code: int, what: str):
        if code != 0:
            raise RuntimeError(f"CUDA runtime error {code}: {what}")

    def get(self, attr: int) -> int:
        value = ctypes.c_uint64()
        self._check(
            self.lib.cudaMemPoolGetAttribute(self.pool, attr, ctypes.byref(value)),
            f"get mempool attribute {attr}",
        )
        return int(value.value)

    def reset_used_high(self):
        zero = ctypes.c_uint64(0)
        self._check(
            self.lib.cudaMemPoolSetAttribute(self.pool, self.USED_HIGH, ctypes.byref(zero)),
            "reset used-memory high watermark",
        )


def collect_vram_report() -> dict:
    # Pool high-water is allocator-level peak telemetry, not a post-call sample:
    # it catches allocations even when a transient is freed before return.
    gc.collect()
    tc.empty_cache()
    tc.set_alloc_pooling(True)
    x_t, codes_t, scales_t, _x, codes, scales = _device_inputs(VRAM_SHAPE, 7200)
    tc.synchronize()
    pool = _CudaPoolStats()
    baseline = pool.get(pool.USED_CURRENT)
    pool.reset_used_high()

    y = tc.w8a16_matmul(x_t, codes_t, scales_t, LAUNCH_CONFIGS[0])
    tc.synchronize()
    fused_peak = pool.get(pool.USED_HIGH)
    fused_current = pool.get(pool.USED_CURRENT)

    shape = tuple(VRAM_SHAPE["shape"])
    o, k = int(VRAM_SHAPE["O"]), int(shape[-1])
    m = math.prod(shape[:-1])
    code_bytes = o * k
    scale_bytes = o * (k // GROUP_SIZE) * 2
    activation_bytes = m * k * 2
    output_bytes = m * o * 2
    full_weight_fp16_bytes = o * k * 2
    # K0b's operand stage is 64x72 half activations + 64x72 half weights.
    # Its lifetime does not overlap the 8x16x36 FP32 output spill, so both
    # alias in one 18,432-byte static-shared union. It is on-chip, not a
    # CUDA-pool allocation.
    tile_shared_bytes = 18432
    allocator_slack = 64 * 1024
    fused_call_peak_delta = fused_peak - baseline
    assert fused_call_peak_delta <= output_bytes + allocator_slack
    assert fused_call_peak_delta < full_weight_fp16_bytes
    assert fused_current - baseline <= output_bytes + allocator_slack

    # Positive control: the explicit reference really does register a full
    # FP16 weight allocation in the same high-water sensor.
    del y
    tc.synchronize()
    tc.empty_cache()
    baseline_control = pool.get(pool.USED_CURRENT)
    pool.reset_used_high()
    w_nk = _dequant_fp16(codes, scales)
    w_kn_t = tc.tensor(np.ascontiguousarray(w_nk.T), dtype="float16")
    y_ref = tc.matmul(x_t, w_kn_t)
    tc.synchronize()
    reference_peak_delta = pool.get(pool.USED_HIGH) - baseline_control
    assert reference_peak_delta >= full_weight_fp16_bytes + output_bytes

    del y_ref, w_kn_t
    tc.synchronize()
    return {
        "family": VRAM_SHAPE["family"],
        "activation_shape": list(shape),
        "weight_shape": [o, k],
        "codes_bytes": code_bytes,
        "scales_bytes": scale_bytes,
        "activations_bytes": activation_bytes,
        "output_bytes": output_bytes,
        "resident_bound_bytes": code_bytes + scale_bytes + activation_bytes + output_bytes,
        "max_static_shared_tile_bytes_per_cta": tile_shared_bytes,
        "full_weight_fp16_transient_bytes": full_weight_fp16_bytes,
        "allocator_used_before_call_bytes": baseline,
        "allocator_peak_during_call_bytes": fused_peak,
        "allocator_used_after_call_bytes": fused_current,
        "fused_call_peak_delta_bytes": fused_call_peak_delta,
        "reference_control_peak_delta_bytes": reference_peak_delta,
        "allocator_slack_bound_bytes": allocator_slack,
        "full_weight_transient_observed": fused_call_peak_delta >= full_weight_fp16_bytes,
    }


def _median_ms(fn, warmup: int = 2, iterations: int = 7) -> float:
    keep = None
    for _ in range(warmup):
        keep = fn()
    tc.synchronize()
    samples = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        keep = fn()
        tc.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    assert keep is not None
    return float(statistics.median(samples))


def collect_perf_report() -> list[dict]:
    rows = []
    for index, case in enumerate(ORACLE_SHAPES):
        x_t, codes_t, scales_t, _x, codes, scales = _device_inputs(case, 7300 + index)
        w_nk = _dequant_fp16(codes, scales)
        w_kn_t = tc.tensor(np.ascontiguousarray(w_nk.T), dtype="float16")
        fp16_ms = _median_ms(lambda: tc.matmul(x_t, w_kn_t))
        for launch_config in LAUNCH_CONFIGS:
            fused_ms = _median_ms(
                lambda launch_config=launch_config: tc.w8a16_matmul(
                    x_t, codes_t, scales_t, launch_config
                )
            )
            rows.append(
                {
                    "family": case["family"],
                    "activation_shape": list(case["shape"]),
                    "weight_shape": [case["O"], case["shape"][-1]],
                    "launch_config": launch_config,
                    "fp16_matmul_median_ms": fp16_ms,
                    "w8a16_fused_median_ms": fused_ms,
                    "speedup_vs_fp16": fp16_ms / fused_ms,
                    "warmup": 2,
                    "iterations": 7,
                }
            )
    return rows


def collect_gate_report() -> dict:
    return {
        "gate_registration": {
            "Q-K0-ORACLE": "measurement only; no threshold registered by this seat",
            "Q-K0-DET": f"byte-equal reruns x{DETERMINISM_RUNS} for {list(LAUNCH_CONFIGS)}",
            "Q-K0-VRAM": "pool high-water bounded by resident tensors plus output; positive-control full dequant",
            "Q-K0-SUITE": "before/after recorded by the implementation run",
            "performance": "report only; no threshold",
        },
        "oracle": collect_oracle_report(),
        "determinism": collect_determinism_report(),
        "vram": collect_vram_report(),
        "performance": collect_perf_report(),
    }


def test_w8a16_trinity_layout_and_batched_shape():
    x = tc.tensor(np.ones((2, 3, 64), dtype=np.float16), dtype="float16")
    codes = tc.tensor(np.full((96, 64), 128, dtype=np.uint8), dtype="uint8")
    scales = tc.tensor(np.ones((96, 2), dtype=np.float16), dtype="float16")
    for launch_config in LAUNCH_CONFIGS:
        got = tc.w8a16_matmul(x, codes, scales, launch_config).numpy()
        assert got.shape == (2, 3, 96)
        assert got.dtype == np.float16
        assert np.array_equal(got, np.zeros_like(got))


def test_w8a16_single_group_tail_matches_two_stage():
    """K is contracted at group-32, not tile-64: exercise the zero-padded
    final half-stage plus M/N tails with nonzero values."""
    case = {"family": "k32_tail", "shape": (7, 96), "O": 70}
    x_t, codes_t, scales_t, _x, codes, scales = _device_inputs(case, 20260716)
    w_nk = _dequant_fp16(codes, scales)
    ref = tc.matmul(
        x_t, tc.tensor(np.ascontiguousarray(w_nk.T), dtype="float16")
    ).numpy()
    for launch_config in LAUNCH_CONFIGS:
        got = tc.w8a16_matmul(x_t, codes_t, scales_t, launch_config).numpy()
        assert np.array_equal(got, ref)


def test_w8a16_contract_errors():
    x_bad_k = tc.tensor(np.ones((2, 33), dtype=np.float16), dtype="float16")
    codes_bad_k = tc.tensor(np.ones((4, 33), dtype=np.uint8), dtype="uint8")
    scales_bad_k = tc.tensor(np.ones((4, 1), dtype=np.float16), dtype="float16")
    with pytest.raises(RuntimeError, match="divisible by 32"):
        tc.w8a16_matmul(x_bad_k, codes_bad_k, scales_bad_k)

    x = tc.tensor(np.ones((2, 64), dtype=np.float16), dtype="float16")
    codes = tc.tensor(np.ones((4, 64), dtype=np.uint8), dtype="uint8")
    scales_wrong = tc.tensor(np.ones((4, 1), dtype=np.float16), dtype="float16")
    with pytest.raises(RuntimeError, match="scales shape"):
        tc.w8a16_matmul(x, codes, scales_wrong)
    with pytest.raises(ValueError, match="launch_config"):
        tc.w8a16_matmul(x, codes, tc.tensor(np.ones((4, 2), dtype=np.float16)), "bad")


def test_q_k0_oracle_report_is_finite():
    assert len(collect_oracle_report()) == len(ORACLE_SHAPES) * len(LAUNCH_CONFIGS)


def test_q_k0_det_byte_equal_x5_two_launch_configs():
    rows = collect_determinism_report()
    assert len(rows) == 2
    assert all(row["byte_equal"] for row in rows)


def test_q_k0_vram_no_full_weight_transient():
    report = collect_vram_report()
    assert not report["full_weight_transient_observed"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "w8a16_gate_report.json",
    )
    args = parser.parse_args()
    report = collect_gate_report()
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
