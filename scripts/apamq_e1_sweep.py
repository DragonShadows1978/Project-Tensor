#!/usr/bin/env python3
"""APAMQ-E1: MQA/GQA/MHA attention transient sweep.

The coordinator starts exactly one worker process for every
(path, kv_heads, q_heads=16, D) row.  A worker walks S in ascending order
and runs the rectangular L=512 prefill cell followed by the L=1 decode cell.
The coordinator enforces the registered 120 second cap per cell.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import gc
import json
import math
import os
from pathlib import Path
import selectors
import statistics
import subprocess
import sys
import threading
import time
import traceback

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "apamq_e1"
RESULTS_JSON = OUT_DIR / "results.json"
RESULTS_MD = OUT_DIR / "RESULTS.md"

GEOMETRIES = ((1, 16), (4, 16), (8, 16), (16, 16))
DIMS = (128, 512)
CONTEXTS = (4096, 8192, 16384, 32768, 65536)
SHAPES = (("prefill", 512), ("decode", 1))
PATHS = ("standard", "fused_apa", "int4_apa", "gemm_apa")
DTYPE = "bfloat16"
BULK_BITS = 4
REFINE = 0.10
WARMUP = 1
REPS = 3
CELL_TIMEOUT_S = 120.0


def _utc_now() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _emit(payload: dict) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)


class CudaPoolStats:
    """CUDA default-pool used-memory telemetry."""

    USED_CURRENT = 7
    USED_HIGH = 8

    def __init__(self):
        candidates = [ctypes.util.find_library("cudart"), "libcudart.so"]
        errors = []
        for candidate in candidates:
            if not candidate:
                continue
            try:
                self.lib = ctypes.CDLL(candidate)
                break
            except OSError as exc:
                errors.append(str(exc))
        else:
            raise RuntimeError("could not load libcudart: " + "; ".join(errors))
        self.lib.cudaDeviceGetDefaultMemPool.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_int
        ]
        self.lib.cudaMemPoolGetAttribute.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p
        ]
        self.lib.cudaMemPoolSetAttribute.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p
        ]
        self.pool = ctypes.c_void_p()
        self._check(
            self.lib.cudaDeviceGetDefaultMemPool(ctypes.byref(self.pool), 0),
            "cudaDeviceGetDefaultMemPool",
        )

    @staticmethod
    def _check(code: int, action: str) -> None:
        if code:
            raise RuntimeError(f"CUDA runtime status {code}: {action}")

    def get(self, attr: int) -> int:
        value = ctypes.c_uint64()
        self._check(
            self.lib.cudaMemPoolGetAttribute(
                self.pool, attr, ctypes.byref(value)
            ),
            f"cudaMemPoolGetAttribute({attr})",
        )
        return int(value.value)

    def reset_high(self) -> None:
        zero = ctypes.c_uint64(0)
        self._check(
            self.lib.cudaMemPoolSetAttribute(
                self.pool, self.USED_HIGH, ctypes.byref(zero)
            ),
            "cudaMemPoolSetAttribute(USED_HIGH)",
        )


class NvmlMemory:
    """Minimal NVML binding, avoiding another CUDA tensor framework."""

    class MemoryInfo(ctypes.Structure):
        _fields_ = [
            ("total", ctypes.c_ulonglong),
            ("free", ctypes.c_ulonglong),
            ("used", ctypes.c_ulonglong),
        ]

    def __init__(self):
        candidate = ctypes.util.find_library("nvidia-ml") or "libnvidia-ml.so.1"
        self.lib = ctypes.CDLL(candidate)
        self.lib.nvmlInit_v2.restype = ctypes.c_int
        self.lib.nvmlShutdown.restype = ctypes.c_int
        self.lib.nvmlDeviceGetHandleByIndex_v2.argtypes = [
            ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)
        ]
        self.lib.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
        self.lib.nvmlDeviceGetMemoryInfo.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(self.MemoryInfo)
        ]
        self.lib.nvmlDeviceGetMemoryInfo.restype = ctypes.c_int
        self._check(self.lib.nvmlInit_v2(), "nvmlInit_v2")
        self.handle = ctypes.c_void_p()
        self._check(
            self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle)),
            "nvmlDeviceGetHandleByIndex_v2(0)",
        )

    @staticmethod
    def _check(code: int, action: str) -> None:
        if code:
            raise RuntimeError(f"NVML status {code}: {action}")

    def used_mib(self) -> float:
        info = self.MemoryInfo()
        self._check(
            self.lib.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(info)),
            "nvmlDeviceGetMemoryInfo",
        )
        return float(info.used) / (1024.0 * 1024.0)

    def close(self) -> None:
        self.lib.nvmlShutdown()


class NvmlSampler:
    def __init__(self, nvml: NvmlMemory, interval_s: float = 0.005):
        self.nvml = nvml
        self.interval_s = interval_s
        self.values: list[float] = []
        self.error: str | None = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.values.append(self.nvml.used_mib())
            except Exception as exc:  # retain the measurement failure
                self.error = f"{type(exc).__name__}: {exc}"
                return
            self.stop_event.wait(self.interval_s)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)


def _host_random_f16(shape: tuple[int, ...], seed: int) -> np.ndarray:
    """Generate a bounded random FP16 host array without a full FP64 temp."""
    arr = np.empty(shape, dtype=np.float16)
    flat = arr.reshape(-1)
    rng = np.random.default_rng(seed)
    chunk = 4 * 1024 * 1024
    for lo in range(0, flat.size, chunk):
        hi = min(flat.size, lo + chunk)
        vals = rng.standard_normal(hi - lo, dtype=np.float32)
        vals *= np.float32(0.1)
        flat[lo:hi] = vals.astype(np.float16)
    return arr


def _to_bf16(tc, arr: np.ndarray):
    """Stage through FP16 to avoid TensorCUDA's full-device FP32 BF16 input."""
    staged = tc.tensor(arr, dtype="float16")
    out = staged.astype("bfloat16")
    del staged
    tc.synchronize()
    return out


def _quantize_symmetric_int4_inplace(arr: np.ndarray) -> None:
    """Per-key-vector signed INT4 bulk quantize/dequantize (codes -7..7)."""
    rows = arr.reshape(-1, arr.shape[-1])
    chunk_rows = max(1, (4 * 1024 * 1024) // arr.shape[-1])
    for lo in range(0, rows.shape[0], chunk_rows):
        hi = min(rows.shape[0], lo + chunk_rows)
        block = rows[lo:hi].astype(np.float32)
        scale = np.max(np.abs(block), axis=1, keepdims=True) / np.float32(7.0)
        safe = np.where(scale > 0, scale, np.float32(1.0))
        codes = np.clip(np.rint(block / safe), -7.0, 7.0)
        rows[lo:hi] = (codes * scale).astype(np.float16)


def _make_k_kq(tc, kv: int, s: int, d: int, seed: int, need_kq: bool):
    host = _host_random_f16((1, kv, s, d), seed)
    k = _to_bf16(tc, host)
    kq = None
    if need_kq:
        _quantize_symmetric_int4_inplace(host)
        kq = _to_bf16(tc, host)
    del host
    gc.collect()
    tc.synchronize()
    return k, kq


def _make_tensor(tc, shape: tuple[int, ...], seed: int):
    host = _host_random_f16(shape, seed)
    out = _to_bf16(tc, host)
    del host
    gc.collect()
    return out


def _standard_call(tc, q, k, v, kv: int, qh: int, l: int, s: int, d: int):
    """Port-style de-expanded grouped causal attention."""
    rep = qh // kv
    qg = q.reshape([1, kv, rep * l, d])
    scores_g = tc.matmul(qg, k, alpha=1.0 / math.sqrt(d), trans_b=True)
    scores = scores_g.reshape([1, qh, l, s])
    weights = tc.causal_softmax(scores)
    weights_g = weights.reshape([1, kv, rep * l, s])
    return tc.matmul(weights_g, v).reshape([1, qh, l, d])


def _fused_call(tc, q, k, kq, v, d: int, zthr: float):
    return tc.apa_selective_attention(
        q, k, kq, v, 1.0 / math.sqrt(d), float(zthr), True
    )


def _int4_call(tc, q, k, v, d: int, zthr: float):
    """F-A1 call-local INT4 pack + causal selective attention."""
    return tc.apa_selective_attention_int4(
        q, k, v, 1.0 / math.sqrt(d), float(zthr), True
    )


def _gemm_apa_call(tc, q, k, v, d: int, zthr: float):
    """SB1 call-local INT8 Q/K + cuBLASLt bulk + selected BF16 refine."""
    return tc.apa_gemm_selective_attention(
        q, k, v, 1.0 / math.sqrt(d), float(zthr), True
    )


def _release_call_output(tc, out) -> None:
    del out
    gc.collect()
    tc.synchronize()
    tc.empty_cache()
    tc.synchronize()


def _measure_cell(tc, pool: CudaPoolStats, nvml: NvmlMemory, call):
    # One untimed warm-up.
    with tc.no_grad():
        warm = call()
    tc.synchronize()
    _release_call_output(tc, warm)

    walls = []
    pool_deltas = []
    nvml_runs = []
    for _ in range(REPS):
        gc.collect()
        tc.empty_cache()
        tc.synchronize()
        baseline_pool = pool.get(pool.USED_CURRENT)
        pool.reset_high()
        before = nvml.used_mib()
        sampler = NvmlSampler(nvml)
        sampler.start()
        start = time.perf_counter()
        try:
            with tc.no_grad():
                out = call()
            tc.synchronize()
            wall_ms = (time.perf_counter() - start) * 1000.0
        finally:
            sampler.stop()
        after_call = nvml.used_mib()
        peak_pool = pool.get(pool.USED_HIGH)
        pool_deltas.append(max(0, peak_pool - baseline_pool))
        walls.append(wall_ms)
        during_peak = max(sampler.values) if sampler.values else before
        nvml_runs.append(
            {
                "before_mib": before,
                "during_peak_mib": during_peak,
                "after_call_mib": after_call,
                "sample_count": len(sampler.values),
                "sampler_error": sampler.error,
            }
        )
        _release_call_output(tc, out)
        nvml_runs[-1]["after_release_mib"] = nvml.used_mib()

    nvml_peak = max(
        max(run["before_mib"], run["during_peak_mib"], run["after_call_mib"])
        for run in nvml_runs
    )
    return {
        "status": "ok",
        "pool_peak_bytes": max(pool_deltas),
        "pool_peak_samples_bytes": pool_deltas,
        "nvml_peak_mib": nvml_peak,
        "nvml_samples": nvml_runs,
        "wall_ms": statistics.median(walls),
        "wall_samples_ms": walls,
        "warmup": WARMUP,
        "repetitions": REPS,
    }


def _base_config(path: str, kv: int, d: int, shape: str, l: int, s: int) -> dict:
    return {
        "path": path,
        "geometry": "MQA" if kv == 1 else ("MHA" if kv == 16 else "GQA"),
        "B": 1,
        "kv_heads": kv,
        "q_heads": 16,
        "D": d,
        "shape": shape,
        "L": l,
        "S": s,
        "causal_alignment": "bottom-right",
        "dtype": DTYPE,
        "bulk_bits": (8 if path == "gemm_apa" else BULK_BITS)
        if path in ("fused_apa", "int4_apa", "gemm_apa") else None,
        "refine_percentile": REFINE
        if path in ("fused_apa", "int4_apa", "gemm_apa") else None,
    }


def worker(
    path: str,
    kv: int,
    d: int,
    *,
    only_shape: str | None = None,
    only_s: int | None = None,
) -> int:
    sys.path.insert(0, str(ROOT / "tensor_cuda"))
    import tensor_cuda as tc
    from tensor_cuda.quant import _norm_ppf

    if not hasattr(tc, "causal_softmax"):
        raise RuntimeError("missing required engine entry point: tensor_cuda.causal_softmax")
    if path == "fused_apa" and not hasattr(tc, "apa_selective_attention"):
        raise RuntimeError(
            "missing required engine entry point: tensor_cuda.apa_selective_attention"
        )
    if path == "int4_apa" and not hasattr(tc, "apa_selective_attention_int4"):
        raise RuntimeError(
            "missing required engine entry point: "
            "tensor_cuda.apa_selective_attention_int4"
        )
    if path == "gemm_apa" and not hasattr(tc, "apa_gemm_selective_attention"):
        raise RuntimeError(
            "missing required engine entry point: "
            "tensor_cuda.apa_gemm_selective_attention"
        )

    tc.set_alloc_pooling(False)
    tc.empty_cache()
    tc.synchronize()
    pool = CudaPoolStats()
    nvml = NvmlMemory()
    zthr = _norm_ppf(1.0 - REFINE)
    try:
        for s_index, s in enumerate(CONTEXTS):
            if only_s is not None and s != only_s:
                continue
            k = kq = v = None
            for shape_index, (shape, l) in enumerate(SHAPES):
                if only_shape is not None and shape != only_shape:
                    continue
                config = _base_config(path, kv, d, shape, l, s)
                _emit({"event": "cell_start", "config": config})
                try:
                    if k is None:
                        # Persistent operands are allocated with pooling disabled;
                        # only the attention call's outputs/transients enter the pool.
                        tc.set_alloc_pooling(False)
                        k, kq = _make_k_kq(
                            tc,
                            kv,
                            s,
                            d,
                            seed=100000 + kv * 1000 + d * 10 + s_index,
                            need_kq=(path == "fused_apa"),
                        )
                        v = _make_tensor(
                            tc,
                            (1, kv, s, d),
                            seed=200000 + kv * 1000 + d * 10 + s_index,
                        )
                    q = _make_tensor(
                        tc,
                        (1, 16, l, d),
                        seed=300000 + kv * 1000 + d * 10 + s_index * 2 + shape_index,
                    )
                    tc.set_alloc_pooling(True)
                    if path == "standard":
                        call = lambda: _standard_call(tc, q, k, v, kv, 16, l, s, d)
                    elif path == "fused_apa":
                        call = lambda: _fused_call(tc, q, k, kq, v, d, zthr)
                    elif path == "int4_apa":
                        call = lambda: _int4_call(tc, q, k, v, d, zthr)
                    else:
                        call = lambda: _gemm_apa_call(tc, q, k, v, d, zthr)
                    measurement = _measure_cell(tc, pool, nvml, call)
                    measurement["config"] = config
                    _emit({"event": "cell_result", "result": measurement})
                    tc.set_alloc_pooling(False)
                    del q
                    gc.collect()
                    tc.synchronize()
                    tc.empty_cache()
                except Exception as exc:
                    error_text = f"{type(exc).__name__}: {exc}"
                    low = error_text.lower()
                    status = "oom" if (
                        "out of memory" in low
                        or "memory allocation" in low
                        or "cudamalloc" in low
                    ) else "error"
                    _emit(
                        {
                            "event": "cell_result",
                            "result": {
                                "status": status,
                                "config": config,
                                "error": error_text,
                                "traceback": traceback.format_exc(),
                            },
                        }
                    )
                    return 0 if status == "oom" else 2
            tc.set_alloc_pooling(False)
            del k, kq, v
            gc.collect()
            tc.synchronize()
            tc.empty_cache()
        return 0
    finally:
        nvml.close()


def _cell_key(config: dict) -> tuple:
    return (
        config["path"], config["kv_heads"], config["q_heads"], config["D"],
        config["S"], config["shape"],
    )


def _all_configs() -> list[dict]:
    return [
        _base_config(path, kv, d, shape, l, s)
        for path in PATHS
        for kv, _qh in GEOMETRIES
        for d in DIMS
        for s in CONTEXTS
        for shape, l in SHAPES
    ]


def _document(results: list[dict], commands: list[str], started: str) -> dict:
    statuses = [r["status"] for r in results]
    completed_by_measurement_or_wall = all(
        status in ("ok", "oom") for status in statuses
    ) and len(results) == len(_all_configs())
    return {
        "experiment": "APAMQ-E1",
        "evidence_class": "KERNEL SWEEP",
        "started_utc": started,
        "updated_utc": _utc_now(),
        "complete": completed_by_measurement_or_wall,
        "sweep_status": (
            "COMPLETE" if completed_by_measurement_or_wall
            else ("PARTIAL" if any(status in ("ok", "oom") for status in statuses) else "RED")
        ),
        "matrix": {
            "geometries": [
                {"kv_heads": kv, "q_heads": qh} for kv, qh in GEOMETRIES
            ],
            "D": list(DIMS),
            "S": list(CONTEXTS),
            "shapes": [{"name": name, "L": l} for name, l in SHAPES],
            "paths": list(PATHS),
            "dtype": DTYPE,
        },
        "measurement": {
            "warmup": WARMUP,
            "timed_repetitions": REPS,
            "cell_timeout_s": CELL_TIMEOUT_S,
            "pool_peak_definition": "max CUDA default-pool USED_HIGH minus USED_CURRENT baseline across timed calls",
            "nvml_peak_definition": "absolute device used-memory max over before, 5ms during samples, and after-call samples",
        },
        "entry_points": {
            "standard": [
                "tensor_cuda.matmul(q_grouped, k, trans_b=True)",
                "tensor_cuda.causal_softmax(scores)",
                "tensor_cuda.matmul(weights_grouped, v)",
            ],
            "fused_apa": ["tensor_cuda.apa_selective_attention(q, k, kq, v, scale, zthr, True)"],
            "int4_apa": ["tensor_cuda.apa_selective_attention_int4(q, k, v, scale, zthr, True)"],
            "gemm_apa": ["tensor_cuda.apa_gemm_selective_attention(q, k, v, scale, zthr, True)"],
            "bulk_key_preparation": {
                "fused_apa": "host per-key-vector signed INT4 (-7..7) quantize/dequantize; excluded from call timing and pool delta",
                "int4_apa": "call-local per-key-vector signed INT4 (-7..7) pack; included in call timing and pool delta; no persistent kq ring",
                "gemm_apa": "call-local per-row Q and per-key K signed INT8 (-127..127) quantization; included in call timing and pool delta; optional cached K codes are deliberately not used",
            },
        },
        "commands": commands,
        "results": results,
    }


def _write_json(results: list[dict], commands: list[str], started: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = RESULTS_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(_document(results, commands, started), indent=2) + "\n")
    os.replace(tmp, RESULTS_JSON)


def _fmt_cell(result: dict | None) -> str:
    if result is None:
        return "MISSING"
    status = result["status"]
    if status == "ok":
        mib = result["pool_peak_bytes"] / (1024.0 * 1024.0)
        return f"{mib:.1f} MiB / {result['wall_ms']:.3f} ms"
    if status == "oom":
        return "OOM"
    if status == "error":
        return "ERROR"
    return "SKIPPED"


def _write_markdown(results: list[dict]) -> None:
    by_key = {_cell_key(r["config"]): r for r in results}
    lines = [
        "# APAMQ-E1 MQA-Geometry Kernel Transient Sweep",
        "",
        "Evidence class: **KERNEL SWEEP**. Values are raw measurements; no H-A/T1 verdict is made here.",
        "",
        "Each cell is `CUDA-pool peak delta MiB / warm median wall ms` (1 warm-up, 3 timed calls).",
        "",
    ]
    for d in DIMS:
        lines.extend([f"## D={d}", ""])
        for shape, l in SHAPES:
            lines.extend(
                [
                    f"### {shape.upper()} (L={l}, bottom-right causal)",
                    "",
                    "| Path | KV heads | " + " | ".join(f"S={s}" for s in CONTEXTS) + " |",
                    "|---|---:|" + "---:|" * len(CONTEXTS),
                ]
            )
            for path in PATHS:
                for kv, qh in GEOMETRIES:
                    cells = [
                        _fmt_cell(by_key.get((path, kv, qh, d, s, shape)))
                        for s in CONTEXTS
                    ]
                    lines.append(
                        f"| {path} | {kv} | " + " | ".join(cells) + " |"
                    )
            lines.append("")

    anomalies = [r for r in results if r["status"] != "ok"]
    any_ok = any(r["status"] == "ok" for r in results)
    lines.extend(
        [
            "## Factual notes",
            "",
            "- Inputs were configured as BF16. Fused APA was configured with a signed per-key-vector 4-bit bulk quantize/dequantize and `refine_percentile=0.10` (`z = NormalPPF(0.90)`). Bulk-key preparation was outside the measured attention call.",
            "- STANDARD is configured to invoke `tensor_cuda.matmul(q_grouped, k, trans_b=True)`, `tensor_cuda.causal_softmax(scores)`, then `tensor_cuda.matmul(weights_grouped, v)`. Q and weights are grouped as `(B, kv_heads, (q_heads/kv_heads)*L, ...)`; K/V are not expanded.",
            "- FUSED APA is configured to invoke `tensor_cuda.apa_selective_attention(q, k, kq, v, scale, zthr, True)` with native `(q_heads, kv_heads)` geometry.",
            "- INT4 APA is configured to invoke `tensor_cuda.apa_selective_attention_int4(q, k, v, scale, zthr, True)` with native `(q_heads, kv_heads)` geometry. Its call-local pack workspace and pack launch are included in both wall and pool measurements; it has no persistent kq operand.",
            "- GEMM APA is configured to invoke `tensor_cuda.apa_gemm_selective_attention(q, k, v, scale, zthr, True)` with native `(q_heads, kv_heads)` geometry. Call-local Q/K INT8 quantization, the INT32 bulk matrix, fp32 scores, selected-pair compaction/readback, bounded BF16 gather/GEMM refinement, and final P@V are all included in wall and pool measurements.",
            "- Pool values are call-local high-water deltas. NVML before/during/after absolute samples are retained per timed repetition in `results.json`.",
        ]
    )
    if not any_ok:
        lines.append("- No attention entry point was reached because CUDA initialization failed before the first cell.")
    if anomalies:
        groups: dict[tuple[str, str], list[dict]] = {}
        for result in anomalies:
            reason = result.get("error", result.get("reason", ""))
            groups.setdefault((result["status"], reason), []).append(result)
        lines.append("- Non-OK cells (grouped by exact outcome):")
        for (status, reason), grouped in groups.items():
            lines.append(f"  - {status.upper()} x{len(grouped)}: {reason}")
    else:
        lines.append("- No cells OOMed, errored, or were skipped.")
    RESULTS_MD.write_text("\n".join(lines) + "\n")


def coordinator() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = _utc_now()
    command = "flock -w 7200 /tmp/forge-gpu.lock python3 scripts/apamq_e1_sweep.py"
    commands = [command]
    results: list[dict] = []
    expected = {_cell_key(c): c for c in _all_configs()}
    abort_all = False

    for path in PATHS:
        for kv, _qh in GEOMETRIES:
            for d in DIMS:
                if abort_all:
                    break
                row_configs = [
                    c for c in _all_configs()
                    if c["path"] == path and c["kv_heads"] == kv and c["D"] == d
                ]
                row_keys = {_cell_key(c) for c in row_configs}
                seen: set[tuple] = set()
                proc = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), "--row", path, str(kv), str(d)],
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
                assert proc.stdout is not None
                selector = selectors.DefaultSelector()
                selector.register(proc.stdout, selectors.EVENT_READ)
                current_config = None
                deadline = None
                timed_out = False
                while proc.poll() is None:
                    events = selector.select(timeout=0.5)
                    if not events:
                        if deadline is not None and time.monotonic() > deadline:
                            timed_out = True
                            proc.kill()
                            proc.wait()
                            if current_config is not None:
                                result = {
                                    "status": "error",
                                    "config": current_config,
                                    "error": f"TimeoutError: cell exceeded {CELL_TIMEOUT_S:.0f}s cap",
                                }
                                results.append(result)
                                seen.add(_cell_key(current_config))
                            break
                        continue
                    line = proc.stdout.readline()
                    if not line:
                        continue
                    try:
                        message = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if message.get("event") == "cell_start":
                        current_config = message["config"]
                        deadline = time.monotonic() + CELL_TIMEOUT_S
                        print(
                            f"START path={path} kv={kv} D={d} "
                            f"shape={current_config['shape']} S={current_config['S']}",
                            flush=True,
                        )
                    elif message.get("event") == "cell_result":
                        result = message["result"]
                        results.append(result)
                        seen.add(_cell_key(result["config"]))
                        current_config = None
                        deadline = None
                        print(
                            f"DONE  path={path} kv={kv} D={d} "
                            f"shape={result['config']['shape']} S={result['config']['S']} "
                            f"status={result['status']}",
                            flush=True,
                        )
                        _write_json(results, commands, started)
                        _write_markdown(results)
                # Drain any final complete stdout lines after exit.
                for line in proc.stdout:
                    try:
                        message = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if message.get("event") == "cell_result":
                        result = message["result"]
                        key = _cell_key(result["config"])
                        if key not in seen:
                            results.append(result)
                            seen.add(key)
                stderr = proc.stderr.read() if proc.stderr is not None else ""
                exit_code = proc.returncode
                row_reason = None
                if timed_out:
                    row_reason = "row terminated after cell timeout"
                elif exit_code != 0:
                    row_reason = f"worker exited {exit_code}: {stderr.strip()}"
                    if not seen and row_configs:
                        first = row_configs[0]
                        results.append(
                            {
                                "status": "error",
                                "config": first,
                                "error": row_reason,
                            }
                        )
                        seen.add(_cell_key(first))
                    if (
                        "missing required engine entry point" in stderr
                        or "no CUDA-capable device is detected" in stderr
                        or "CUDA driver version is insufficient" in stderr
                    ):
                        abort_all = True
                elif len(seen) < len(row_keys):
                    last = next((r for r in reversed(results) if _cell_key(r["config"]) in row_keys), None)
                    if last and last["status"] == "oom":
                        row_reason = f"row ended at OOM: {last.get('error', '')}"
                    elif last and last["status"] == "error":
                        row_reason = f"row ended at error: {last.get('error', '')}"
                    else:
                        row_reason = "worker ended before all row cells"
                if row_reason:
                    for config in row_configs:
                        key = _cell_key(config)
                        if key not in seen:
                            results.append(
                                {
                                    "status": "skipped",
                                    "config": config,
                                    "reason": row_reason,
                                }
                            )
                            seen.add(key)
                _write_json(results, commands, started)
                _write_markdown(results)
            if abort_all:
                break
        if abort_all:
            break

    if abort_all:
        present = {_cell_key(r["config"]) for r in results}
        for key, config in expected.items():
            if key not in present:
                results.append(
                    {
                        "status": "skipped",
                        "config": config,
                        "reason": "sweep stopped because a global runtime blocker prevented GPU execution",
                    }
                )
    _write_json(results, commands, started)
    _write_markdown(results)
    counts = {status: sum(r["status"] == status for r in results) for status in ("ok", "oom", "error", "skipped")}
    print(json.dumps({"results_json": str(RESULTS_JSON), "results_md": str(RESULTS_MD), "counts": counts}, sort_keys=True))
    return 2 if abort_all else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row", nargs=3, metavar=("PATH", "KV", "D"))
    parser.add_argument(
        "--cell", nargs=5, metavar=("PATH", "KV", "D", "SHAPE", "S"),
        help="run exactly one registered sweep cell (worker JSON to stdout)",
    )
    args = parser.parse_args()
    if args.cell:
        path, kv, d, shape, s = args.cell
        if path not in PATHS:
            parser.error(f"PATH must be one of {PATHS}")
        if shape not in dict(SHAPES):
            parser.error(f"SHAPE must be one of {tuple(dict(SHAPES))}")
        if int(s) not in CONTEXTS:
            parser.error(f"S must be one of {CONTEXTS}")
        return worker(
            path, int(kv), int(d), only_shape=shape, only_s=int(s)
        )
    if args.row:
        path, kv, d = args.row
        return worker(path, int(kv), int(d))
    return coordinator()


if __name__ == "__main__":
    raise SystemExit(main())
