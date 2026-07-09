#!/usr/bin/env python3
"""Trinity Nano INT8 quant-path audit.

Receipts:
  - single-op BF16-rounded linear vs QuantLinearInt8TC.
  - optional resident INT8 P1 drift against an explicit HF reference dir.

The P1 comparison is intentionally reference-dir explicit because the resident
T1 harness runs fp32 compute after quantized load; fp32 compute must be gated
against the fp32 HF reference, not the older bf16 artifact.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tensor_cuda"))

import tensor_cuda as tc  # noqa: E402
import core.trinity_nano_tc as TN  # noqa: E402


DEFAULT_MODEL_DIR = Path("/mnt/ForgeRealm/models/trinity-nano")
DEFAULT_FP32_REF_DIR = ROOT / "artifacts" / "trinity_nano" / "reference_capture_fp32"
DEFAULT_OUT_ROOT = ROOT / "artifacts" / "trinity_nano" / "quant_audit"

DEFAULT_OPS = (
    ("attn_q_l3", "model.layers.3.self_attn.q_proj.weight"),
    ("expert_l2_e0_w1_gate", "model.layers.2.mlp.experts.0.gate_proj.weight"),
    ("lm_head", "lm_head.weight"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--reference-dir", type=Path, default=DEFAULT_FP32_REF_DIR)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--tokens", type=int, default=7)
    p.add_argument("--compute-dtype", choices=("float32", "bfloat16"), default="float32")
    p.add_argument("--int8-group-size", type=int, default=TN.INT8_GROUP_SIZE)
    p.add_argument("--rel-floor", type=float, default=1e-3)
    p.add_argument("--run-p1", action="store_true")
    p.add_argument("--p1-steps", type=int, default=8)
    p.add_argument("--drift-stop-max-abs", type=float, default=1.0)
    return p.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def nvidia_smi() -> str | None:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        return None


def bf16_round_np(arr: np.ndarray) -> np.ndarray:
    return torch.from_numpy(np.asarray(arr, dtype=np.float32)).to(torch.bfloat16).to(
        torch.float32
    ).numpy()


def compare_arrays(ref: np.ndarray, got: np.ndarray, *, rel_floor: float) -> dict[str, Any]:
    ref = np.asarray(ref, dtype=np.float32)
    got = np.asarray(got, dtype=np.float32)
    absdiff = np.abs(got - ref)
    ref_abs = np.abs(ref)
    rel = absdiff / np.maximum(ref_abs, float(rel_floor))
    peak = float(ref_abs.max())
    return {
        "shape": [int(x) for x in ref.shape],
        "ref_abs_max": peak,
        "max_abs": float(absdiff.max()),
        "mean_abs": float(absdiff.mean()),
        "peak_relative": float(absdiff.max() / max(peak, float(rel_floor))),
        "max_relative_floor": float(rel.max()),
        "median_relative_floor": float(np.median(rel)),
        "p99_relative_floor": float(np.quantile(rel, 0.99)),
        "rel_floor": float(rel_floor),
    }


def run_single_ops(args: argparse.Namespace) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(args.seed))
    rows: list[dict[str, Any]] = []
    TN.BlockTC.COMPUTE_DTYPE = args.compute_dtype
    TN.LinearTC.DTYPE = args.compute_dtype
    TN.RMSNormTC.USE_FUSED = False

    with tc.no_grad(), TN.SafeTensorSource(args.model_dir) as source:
        for label, key in DEFAULT_OPS:
            w = source.get_np(key)
            w_ref = bf16_round_np(w)
            x_np = rng.standard_normal(
                (int(args.batch), int(args.tokens), int(w.shape[1]))
            ).astype(np.float32)
            x = tc.tensor(np.ascontiguousarray(x_np), dtype=args.compute_dtype)
            ref = TN.LinearTC(w_ref)(x).float().numpy().astype(np.float32)
            q = TN.QuantLinearInt8TC(w, int(args.int8_group_size))(x).float().numpy().astype(
                np.float32
            )
            stats = compare_arrays(ref, q, rel_floor=float(args.rel_floor))
            rows.append(
                {
                    "label": label,
                    "weight": key,
                    "weight_shape": [int(x) for x in w.shape],
                    "int8_group_size": int(args.int8_group_size),
                    **stats,
                }
            )
            del x
            if hasattr(tc, "empty_cache"):
                tc.empty_cache()
    return rows


def set_compute_dtype(model: TN.TrinityNano_TC, dtype: str) -> dict[str, Any]:
    TN.BlockTC.COMPUTE_DTYPE = str(dtype)
    TN.LinearTC.DTYPE = str(dtype)
    rope_note = "rope_not_present"
    if getattr(model, "rope", None) is not None:
        prev_len = int(getattr(model.rope, "_rope_len", 0) or 0)
        prev_dtype = str(model.rope.cos.dtype) if model.rope.cos is not None else None
        model.rope._rope_len = 0
        model.rope.cos = None
        model.rope.sin = None
        if prev_len > 0:
            model.rope.extend(prev_len)
        rope_note = (
            f"invalidated_and_rebuild_prev_len={prev_len}"
            f"_prev_dtype={prev_dtype}_now={dtype}"
        )
    cast_count = 0
    for layer in model.layers:
        mlp = getattr(layer, "mlp", None)
        linears = [getattr(mlp, "router_gate", None)]
        for obj in (getattr(layer, "self_attn", None), mlp, getattr(mlp, "shared_experts", None)):
            if obj is None:
                continue
            for name in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
                linears.append(getattr(obj, name, None))
        for ex in getattr(mlp, "experts", None) or []:
            for name in ("gate_proj", "up_proj", "down_proj"):
                linears.append(getattr(ex, name, None))
        for lin in linears:
            if isinstance(lin, TN.LinearTC) and str(lin.wT.dtype) != str(dtype):
                lin.wT = lin.wT.astype(str(dtype))
                cast_count += 1
    if isinstance(model.lm_head, TN.LinearTC) and str(model.lm_head.wT.dtype) != str(dtype):
        model.lm_head.wT = model.lm_head.wT.astype(str(dtype))
        cast_count += 1
    return {"compute_dtype": str(dtype), "rope_rebuild": rope_note, "lineartc_cast_count": cast_count}


def load_reference(reference_dir: Path) -> dict[str, Any]:
    path = Path(reference_dir) / "probe_00_plain_short.npz"
    z = np.load(path)
    meta_path = Path(reference_dir) / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return {
        "path": str(path),
        "metadata_path": str(meta_path) if meta_path.exists() else None,
        "torch_dtype_requested": meta.get("torch_dtype_requested"),
        "input_ids": z["input_ids"].astype(np.int64),
        "generated_ids": z["generated_ids"].astype(np.int64),
        "logits": z["logits"].astype(np.float32),
        "top5_ids": z["top5_ids"].astype(np.int64),
    }


def top5_ids(row: np.ndarray) -> np.ndarray:
    return np.argsort(-row)[:5].astype(np.int64)


def run_p1_drift(args: argparse.Namespace) -> dict[str, Any]:
    ref = load_reference(args.reference_dir)
    TN.GROUP_SIZE = int(TN.GROUP_SIZE)
    TN.INT8_GROUP_SIZE = int(args.int8_group_size)
    TN.BlockTC.COMPUTE_DTYPE = "bfloat16"
    TN.LinearTC.DTYPE = "bfloat16"
    TN.RMSNormTC.USE_FUSED = False
    TN.QuantLinearTC.FUSED_DECODE = True

    t_load = time.perf_counter()
    with tc.no_grad():
        model, info = TN.TrinityNano_TC.from_pretrained(
            str(args.model_dir), weight_mode="int8", load_lm_head=True, progress=True
        )
        model.configure_moe_empty_cache(0)
        dtype_meta = set_compute_dtype(model, args.compute_dtype)
        if hasattr(tc, "synchronize"):
            tc.synchronize()
    load_s = time.perf_counter() - t_load

    input_ids = ref["input_ids"].reshape(1, -1)
    generated_ids = ref["generated_ids"].reshape(-1)
    prompt_len = int(input_ids.shape[1])
    current = input_ids
    caches = None
    rows = []
    t0 = time.perf_counter()
    with tc.no_grad():
        for step in range(min(int(args.p1_steps), int(ref["logits"].shape[0]))):
            logits, caches = model(
                current,
                kv_caches=caches,
                position_offset=0 if step == 0 else prompt_len + step - 1,
                last_token_only=True,
            )
            got = logits.float().numpy()[0, -1].astype(np.float32)
            want = ref["logits"][step].astype(np.float32, copy=False)
            got_top = top5_ids(got)
            ref_top = ref["top5_ids"][step].astype(np.int64, copy=False)
            overlap = sorted({int(x) for x in got_top.tolist()} & {int(x) for x in ref_top.tolist()})
            rows.append(
                {
                    "step": int(step),
                    "max_abs_delta_logit": float(np.max(np.abs(got - want))),
                    "mean_abs_delta_logit": float(np.mean(np.abs(got - want))),
                    "int8_top5_ids": [int(x) for x in got_top.tolist()],
                    "reference_top5_ids": [int(x) for x in ref_top.tolist()],
                    "top5_exact": bool(np.array_equal(got_top, ref_top)),
                    "top5_overlap_count": int(len(overlap)),
                    "top5_overlap_ids": overlap,
                    "argmax_id": int(got_top[0]),
                    "reference_argmax_id": int(ref_top[0]),
                }
            )
            current = np.asarray([[int(generated_ids[step])]], dtype=np.int64)
            del logits
            if hasattr(tc, "empty_cache"):
                tc.empty_cache()

    max_abs = max((r["max_abs_delta_logit"] for r in rows), default=None)
    exact = int(sum(1 for r in rows if r["top5_exact"]))
    return {
        "evidence_class": "teacher-forced INT8-resident logits vs saved HF P1 logits",
        "reference_path": ref["path"],
        "reference_torch_dtype_requested": ref["torch_dtype_requested"],
        "model_info": info,
        "load_seconds": round(load_s, 3),
        "dtype_meta": dtype_meta,
        "steps": rows,
        "step_count": len(rows),
        "max_abs_delta_logit": max_abs,
        "top5_exact_steps": exact,
        "top5_total_steps": len(rows),
        "top5_overlap_total": int(sum(r["top5_overlap_count"] for r in rows)),
        "stop_max_abs_delta_logit": float(args.drift_stop_max_abs),
        "multi_logit_drift_stop": bool(
            max_abs is not None and max_abs > float(args.drift_stop_max_abs)
        ),
        "wall_s": round(time.perf_counter() - t0, 3),
    }


def main() -> int:
    args = parse_args()
    out = (
        args.output
        if args.output is not None
        else DEFAULT_OUT_ROOT / f"quant_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    args.model_dir = args.model_dir.expanduser().resolve()
    args.reference_dir = args.reference_dir.expanduser().resolve()
    TN.INT8_GROUP_SIZE = int(args.int8_group_size)

    payload: dict[str, Any] = {
        "schema": "trinity_nano_quant_audit_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_dir": str(args.model_dir),
        "reference_dir": str(args.reference_dir),
        "compute_dtype": str(args.compute_dtype),
        "int8_group_size": int(args.int8_group_size),
        "seed": int(args.seed),
        "gpu_before": nvidia_smi(),
        "status": "running",
    }
    write_json(out, payload)
    t0 = time.perf_counter()
    payload["single_op_audit"] = run_single_ops(args)
    if args.run_p1:
        payload["p1_drift"] = run_p1_drift(args)
    payload.update(
        {
            "status": "ok",
            "wall_seconds": round(time.perf_counter() - t0, 3),
            "gpu_after": nvidia_smi(),
        }
    )
    write_json(out, payload)
    print(json.dumps({"status": "ok", "artifact": str(out)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
