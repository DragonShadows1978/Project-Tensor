#!/usr/bin/env python3
"""APAMQ-DF1/DF2 stage-resolved decode benchmark and Nsight Compute driver.

CUDA events are created inside TensorCUDA immediately around each launch, so
the stage columns are pack/stats/split/merge kernel time rather than Python
wall time.  The pipeline column is a second event measurement around the whole
serving call.  All inputs are B=1, H=16, KVH=1, L=1, BF16.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import math
import os
from pathlib import Path
import statistics
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TC_ROOT = ROOT / "tensor_cuda"
if str(TC_ROOT) not in sys.path:
    sys.path.insert(0, str(TC_ROOT))

import tensor_cuda as tc  # noqa: E402
from tensor_cuda.quant import _norm_ppf  # noqa: E402


DIMS = (128, 512)
SEQUENCES = (8192, 16384, 32768, 65536)
VARIANTS = (
    "off", "v1", "v2", "v3", "v4",
    "v1_v2", "v1_v3", "v1_v4", "v2_v3", "v2_v4", "v3_v4",
    "v1_v2_v3", "v1_v2_v4", "v1_v3_v4", "v2_v3_v4",
    "v1_v2_v3_v4",
)
TARGET_MS = 1.0
STAGES = ("pack", "stats", "split", "merge")


class CudaEvents:
    def __init__(self):
        name = ctypes.util.find_library("cudart") or "libcudart.so"
        self.lib = ctypes.CDLL(name)
        self.lib.cudaEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cudaEventDestroy.argtypes = [ctypes.c_void_p]
        self.lib.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.cudaEventSynchronize.argtypes = [ctypes.c_void_p]
        self.lib.cudaEventElapsedTime.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p
        ]

    @staticmethod
    def _check(code, where):
        if code:
            raise RuntimeError(f"CUDA runtime error {code} at {where}")

    def elapsed_ms(self, callback):
        start, stop = ctypes.c_void_p(), ctypes.c_void_p()
        self._check(self.lib.cudaEventCreate(ctypes.byref(start)), "create start")
        self._check(self.lib.cudaEventCreate(ctypes.byref(stop)), "create stop")
        try:
            self._check(self.lib.cudaEventRecord(start, None), "record start")
            result = callback()
            self._check(self.lib.cudaEventRecord(stop, None), "record stop")
            self._check(self.lib.cudaEventSynchronize(stop), "sync stop")
            elapsed = ctypes.c_float()
            self._check(
                self.lib.cudaEventElapsedTime(ctypes.byref(elapsed), start, stop),
                "elapsed",
            )
            return float(elapsed.value), result
        finally:
            self.lib.cudaEventDestroy(start)
            self.lib.cudaEventDestroy(stop)


def set_variant(name: str) -> None:
    if name not in VARIANTS:
        raise ValueError(f"unknown variant {name!r}")
    os.environ["TC_APA_SELECTIVE_PATH"] = "2"
    for var in ("TC_APAMQ_DF_V2", "TC_APAMQ_DF_V3", "TC_APAMQ_DF_V4"):
        os.environ.pop(var, None)
    if "v2" in name:
        os.environ["TC_APAMQ_DF_V2"] = "1"
    if "v3" in name:
        os.environ["TC_APAMQ_DF_V3"] = "1"
    if "v4" in name:
        os.environ["TC_APAMQ_DF_V4"] = "1"


def make_inputs(d: int, sequence: int):
    # Deterministic bounded values keep setup reproducible without making RNG
    # generation dominate the script's host time at D=512,S=64K.
    q_np = np.linspace(-0.125, 0.125, 16 * d, dtype=np.float32).reshape(1, 16, 1, d)
    row = np.linspace(-0.2, 0.2, d, dtype=np.float32)
    offsets = ((np.arange(sequence, dtype=np.float32) % 31) - 15)[:, None] * 1e-4
    k_np = (row[None, :] + offsets).reshape(1, 1, sequence, d)
    v_np = (row[None, ::-1] - offsets).reshape(1, 1, sequence, d)
    return (
        tc.tensor(q_np, dtype="bfloat16"),
        tc.tensor(k_np, dtype="bfloat16"),
        tc.tensor(v_np, dtype="bfloat16"),
    )


def make_workspace(name: str, k, sequence: int):
    return tc.apa_int4_workspace(k, capacity=sequence) if "v1" in name else None


def call(q, k, v, workspace, d):
    return tc.apa_selective_attention_int4(
        q, k, v, 1.0 / math.sqrt(d), float(_norm_ppf(0.90)), True,
        workspace=workspace,
    )


def benchmark_cell(name, d, sequence, warmup, repeats):
    set_variant(name)
    q, k, v = make_inputs(d, sequence)
    workspace = make_workspace(name, k, sequence)
    for _ in range(warmup):
        call(q, k, v, workspace, d)
    tc.synchronize()
    stages = tc._apa_int4_profile_stages(
        q, k, v, 1.0 / math.sqrt(d), float(_norm_ppf(0.90)), True,
        warmup=warmup, repeats=repeats, workspace=workspace,
        profile_append_rows=(1 if workspace is not None else 0),
    )
    events = CudaEvents()
    pipeline = []
    for _ in range(repeats):
        if workspace is not None:
            workspace._rewind(sequence - 1)
        pipeline.append(events.elapsed_ms(lambda: call(q, k, v, workspace, d))[0])
    tc.synchronize()
    plan = tc._apa_int4_decode_plan(
        16, sequence, grid_fill=("v3" in name), cache_bulk=("v2" in name),
        split_stats=("v4" in name),
    )
    return stages, statistics.median(pipeline), plan, workspace


def print_design() -> None:
    print("# APAMQ-DF1/DF2 stage timing design")
    print("# CUDA-event means around individual launches; pipeline is median whole-call event time.")
    print("# shape: B=1 H=16 KVH=1 L=1 BF16 causal; target D=512 S=65536 <=1.000 ms.")
    print("variant,D,S,stats_P,stats_part_keys,split_P,split_part_keys,stats_partial_blocks,stats_reduce_blocks,split_blocks,merge_blocks,pack_ms,stats_ms,split_ms,merge_ms,stage_sum_ms,pipeline_ms,target")


def run_benchmark(args) -> int:
    variants = tuple(x.strip() for x in args.variants.split(",") if x.strip())
    print_design()
    rows = []
    for name in variants:
        for d in args.d:
            for sequence in args.s:
                stages, pipeline, plan, workspace = benchmark_cell(
                    name, d, sequence, args.warmup, args.repeats
                )
                stage_sum = sum(stages.values())
                target = "PASS" if d == 512 and sequence == 65536 and pipeline <= TARGET_MS else (
                    "MISS" if d == 512 and sequence == 65536 else "n/a"
                )
                values = [stages[s] for s in STAGES]
                print(
                    f"{name},{d},{sequence},{plan['stats_partitions']},"
                    f"{plan['stats_partition_keys']},"
                    f"{plan['split_partitions']},"
                    f"{plan['split_partition_keys']},"
                    f"{plan['stats_partial_blocks']},"
                    f"{plan['stats_reduce_blocks']},"
                    f"{plan['split_blocks']},{plan['merge_blocks']},"
                    + ",".join(f"{x:.6f}" for x in values)
                    + f",{stage_sum:.6f},{pipeline:.6f},{target}",
                    flush=True,
                )
                rows.append((name, d, sequence, pipeline))
                del workspace, plan, stages
                tc.empty_cache()
    target_rows = [r for r in rows if r[1:] and r[1] == 512 and r[2] == 65536]
    if target_rows:
        best = min(target_rows, key=lambda row: row[3])
        print(f"# BEST_TARGET variant={best[0]} pipeline_ms={best[3]:.6f} rail_ms={TARGET_MS:.3f}")
    return 0


def run_ncu_once(args) -> int:
    set_variant(args.variant)
    q, k, v = make_inputs(args.ncu_d, args.ncu_s)
    workspace = make_workspace(args.variant, k, args.ncu_s)
    if workspace is not None:
        workspace._rewind(args.ncu_s - 1)
    call(q, k, v, workspace, args.ncu_d)
    tc.synchronize()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--d", nargs="+", type=int, default=list(DIMS))
    parser.add_argument("--s", nargs="+", type=int, default=list(SEQUENCES))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--ncu-once", action="store_true")
    parser.add_argument("--variant", choices=VARIANTS, default="off")
    parser.add_argument("--ncu-d", type=int, default=512)
    parser.add_argument("--ncu-s", type=int, default=65536)
    args = parser.parse_args()
    try:
        if args.ncu_once:
            return run_ncu_once(args)
        return run_benchmark(args)
    except Exception as exc:
        message = str(exc)
        if "CUDA" in message or "driver" in message.lower():
            print(f"SKIP: CUDA unavailable: {exc}")
            return 0
        raise


if __name__ == "__main__":
    raise SystemExit(main())
