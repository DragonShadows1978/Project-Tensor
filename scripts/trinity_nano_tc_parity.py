#!/usr/bin/env python3
"""TensorCUDA Trinity Nano parity and INT4 deviation receipt harness.

This script intentionally streams decoder blocks one at a time. That keeps the
bf16 parity path separate from the all-resident INT4 model class and mirrors the
GPT-OSS stream-forward precedent.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tensor_cuda"))

import tensor_cuda as tc  # noqa: E402
from core.trinity_nano_tc import (  # noqa: E402
    BlockTC,
    HostEmbedding,
    LinearTC,
    QuantLinearTC,
    RMSNormTC,
    RoPECache,
    SafeTensorSource,
    TrinityBlockTC,
    TrinityNanoConfig,
    _cast,
    _linear_from_weight,
)


DEFAULT_MODEL_DIR = "/mnt/ForgeRealm/models/trinity-nano"
DEFAULT_REF_DIR = "artifacts/trinity_nano/reference_capture"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--reference-dir", default=DEFAULT_REF_DIR)
    parser.add_argument("--output", default=None)
    parser.add_argument("--weight-mode", choices=("bf16", "int4", "int8"), default="bf16")
    parser.add_argument(
        "--compute-dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
        help=(
            "Harness-level engine compute dtype via BlockTC.COMPUTE_DTYPE and "
            "LinearTC.DTYPE (no product edits). float32 is the T4 A/B path."
        ),
    )
    parser.add_argument("--probe-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--decode-check-steps", type=int, default=16)
    parser.add_argument("--route-detail", choices=("summary", "full"), default="summary")
    parser.add_argument("--empty-cache-interval", type=int, default=1)
    parser.add_argument("--skip-reference-compare", action="store_true")
    return parser.parse_args()


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def probe_name(index: int) -> str:
    names = [
        "probe_00_plain_short",
        "probe_01_chat_short",
        "probe_02_long_sliding_boundary",
    ]
    return names[int(index)]


def load_reference(reference_dir: Path, probe_idx: int) -> dict[str, Any]:
    path = reference_dir / f"{probe_name(probe_idx)}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    z = np.load(path)
    return {
        "path": str(path),
        "input_ids": z["input_ids"].astype(np.int64),
        "generated_ids": z["generated_ids"].astype(np.int64),
        "logits": z["logits"].astype(np.float32),
        "top5_ids": z["top5_ids"].astype(np.int64),
        "top5_logits": z["top5_logits"].astype(np.float32),
    }


def bf16_bits_to_fp32(arr: np.ndarray) -> np.ndarray:
    return torch.from_numpy(arr).view(torch.bfloat16).float().numpy()


class TrinityStreamRunner:
    def __init__(
        self,
        cfg: TrinityNanoConfig,
        source: SafeTensorSource,
        *,
        weight_mode: str,
        route_detail: str,
        empty_cache_interval: int,
    ):
        self.cfg = cfg
        self.source = source
        self.weight_mode = weight_mode
        self.route_detail = route_detail
        self.empty_cache_interval = int(empty_cache_interval)
        self.embed_tokens = HostEmbedding()
        emb = source.get_np("model.embed_tokens.weight")
        self.embed_tokens.weight = np.ascontiguousarray(emb.astype(np.float32, copy=False))
        self.norm = RMSNormTC(cfg.hidden_size, cfg.rms_norm_eps)
        self.norm.weight = tc.tensor(source.get_np("model.norm.weight"), dtype="float32")
        self.lm_head = _linear_from_weight(source.get_np("lm_head.weight"), weight_mode)
        self.rope = RoPECache(cfg)

    def forward(
        self,
        input_ids_np: np.ndarray,
        *,
        kv_caches=None,
        position_offset: int = 0,
        last_token_only: bool = True,
        capture_kv: bool = False,
    ):
        input_ids_np = np.asarray(input_ids_np, dtype=np.int64)
        B, L = input_ids_np.shape
        self.rope.extend(position_offset + L)
        h = self.embed_tokens(input_ids_np)
        if self.cfg.mup_enabled:
            h = h * (float(self.cfg.hidden_size) ** 0.5)
        new_caches = []
        layer_receipts = []
        captures = {}
        for layer_idx in range(self.cfg.num_hidden_layers):
            t0 = time.perf_counter()
            block = TrinityBlockTC.from_safetensors(
                self.cfg, self.source, layer_idx, weight_mode=self.weight_mode
            )
            if block.moe_enabled:
                block.mlp.route_detail = self.route_detail
                block.mlp.empty_cache_interval = self.empty_cache_interval
            if capture_kv:
                block.self_attn._capture = True
            cache = kv_caches[layer_idx] if kv_caches is not None else None
            h, kv, route_info = block(h, self.rope.cos, self.rope.sin, position_offset, cache)
            if capture_kv:
                cap = block.self_attn._captured
                if cap is not None:
                    captures[int(layer_idx)] = (
                        cap[0].astype(np.float32, copy=False),
                        cap[1].astype(np.float32, copy=False),
                    )
            new_caches.append(kv)
            if kv_caches is not None:
                kv_caches[layer_idx] = None
            layer_receipts.append(
                {
                    "layer": int(layer_idx),
                    "layer_type": self.cfg.layer_types[layer_idx],
                    "attention_backend": block.self_attn.last_attention_backend,
                    "hidden_shape": [int(x) for x in h.shape],
                    "kv_shapes": [
                        [int(x) for x in kv[0].shape],
                        [int(x) for x in kv[1].shape],
                    ],
                    "route_info": route_info,
                    "wall_seconds": time.perf_counter() - t0,
                    "gpu_after_layer": nvidia_smi(),
                }
            )
            del block, kv
            gc.collect()
            if hasattr(tc, "empty_cache"):
                tc.empty_cache()
        h = _cast(self.norm(h))
        if last_token_only and h.shape[1] > 1:
            h = h.slice(1, h.shape[1] - 1, 1)
        logits = self.lm_head(h)
        return logits, new_caches, layer_receipts, captures


def compare_logits(tc_logits: np.ndarray, ref_logits: np.ndarray, ref_top5: np.ndarray):
    tc_top5 = np.argsort(-tc_logits)[:5].astype(np.int64)
    return {
        "max_abs_diff": float(np.max(np.abs(tc_logits - ref_logits))),
        "mean_abs_diff": float(np.mean(np.abs(tc_logits - ref_logits))),
        "top5_ids": [int(x) for x in tc_top5.tolist()],
        "reference_top5_ids": [int(x) for x in ref_top5.tolist()],
        "top5_exact": bool(np.array_equal(tc_top5, ref_top5)),
        "argmax_id": int(tc_top5[0]),
        "reference_argmax_id": int(ref_top5[0]),
    }


def compare_prerope_kv(
    reference_dir: Path,
    captures: dict[int, tuple[np.ndarray, np.ndarray]],
    *,
    compute_dtype: str = "bfloat16",
):
    """Compare TC pre-RoPE K/V against HF reference.

    Prefer dtype-matched artifacts:
      float32 -> probe_00_prerope_kv_fp32.npz
      bfloat16 -> probe_00_prerope_kv_bf16_bits.npz
    """
    if compute_dtype == "float32":
        path = reference_dir / "probe_00_prerope_kv_fp32.npz"
        if not path.exists():
            return {"status": "missing_reference_kv", "path": str(path)}
        z = np.load(path)
        load_k = lambda i: z[f"layer_{i:03d}_k_fp32"].astype(np.float32, copy=False)
        load_v = lambda i: z[f"layer_{i:03d}_v_fp32"].astype(np.float32, copy=False)
        ref_kind = "fp32"
    else:
        path = reference_dir / "probe_00_prerope_kv_bf16_bits.npz"
        if not path.exists():
            return {"status": "missing_reference_kv", "path": str(path)}
        z = np.load(path)
        load_k = lambda i: bf16_bits_to_fp32(z[f"layer_{i:03d}_k_bf16_u16"])
        load_v = lambda i: bf16_bits_to_fp32(z[f"layer_{i:03d}_v_bf16_u16"])
        ref_kind = "bf16_bits"
    rows = []
    for layer_idx in sorted(captures):
        k_tc, v_tc = captures[layer_idx]
        k_ref = load_k(layer_idx)
        v_ref = load_v(layer_idx)
        rows.append(
            {
                "layer": int(layer_idx),
                "k_shape": [int(x) for x in k_tc.shape],
                "v_shape": [int(x) for x in v_tc.shape],
                "k_max_abs_diff": float(np.max(np.abs(k_tc - k_ref))),
                "v_max_abs_diff": float(np.max(np.abs(v_tc - v_ref))),
                "k_mean_abs_diff": float(np.mean(np.abs(k_tc - k_ref))),
                "v_mean_abs_diff": float(np.mean(np.abs(v_tc - v_ref))),
            }
        )
    return {
        "status": "ok",
        "path": str(path),
        "reference_kind": ref_kind,
        "layers_compared": len(rows),
        "max_k_abs_diff": max((r["k_max_abs_diff"] for r in rows), default=None),
        "max_v_abs_diff": max((r["v_max_abs_diff"] for r in rows), default=None),
        "per_layer": rows,
    }


def run_reference_steps(runner: TrinityStreamRunner, reference: dict[str, Any], steps: int):
    ids = reference["input_ids"].reshape(1, -1)
    logits_receipts = []
    generated = []
    caches = None
    captures_first = None
    current = ids
    for step in range(int(steps)):
        logits_t, caches, layer_receipts, captures = runner.forward(
            current,
            kv_caches=caches,
            position_offset=0 if step == 0 else ids.shape[1] + step - 1,
            last_token_only=True,
            capture_kv=(step == 0),
        )
        logits_np = logits_t.float().numpy()[0, -1].astype(np.float32)
        ref_logits = reference["logits"][step]
        logits_receipts.append(
            {
                "step": int(step),
                **compare_logits(logits_np, ref_logits, reference["top5_ids"][step]),
                "layer_wall_seconds_sum": float(sum(x["wall_seconds"] for x in layer_receipts)),
            }
        )
        next_id = int(np.argmax(logits_np))
        generated.append(next_id)
        feed_id = int(reference["generated_ids"][step])
        current = np.asarray([[feed_id]], dtype=np.int64)
        if step == 0:
            captures_first = captures
        del logits_t
        if hasattr(tc, "empty_cache"):
            tc.empty_cache()
    return logits_receipts, generated, captures_first or {}


def run_incremental_refeed_check(
    runner: TrinityStreamRunner,
    input_ids: np.ndarray,
    steps: int,
):
    prefix = np.asarray(input_ids, dtype=np.int64).reshape(1, -1)
    cached_tokens = []
    refeed_tokens = []
    cache_logits = []
    refeed_logits = []
    caches = None
    current = prefix
    for step in range(int(steps)):
        logits_t, caches, _layers, _cap = runner.forward(
            current,
            kv_caches=caches,
            position_offset=0 if step == 0 else prefix.shape[1] + step - 1,
            last_token_only=True,
            capture_kv=False,
        )
        row = logits_t.float().numpy()[0, -1].astype(np.float32)
        tok = int(np.argmax(row))
        cached_tokens.append(tok)
        cache_logits.append(row)
        current = np.asarray([[tok]], dtype=np.int64)

        full = np.concatenate(
            [prefix, np.asarray(cached_tokens[:-1], dtype=np.int64).reshape(1, -1)],
            axis=1,
        )
        logits_r, _caches_r, _layers_r, _cap_r = runner.forward(
            full,
            kv_caches=None,
            position_offset=0,
            last_token_only=True,
            capture_kv=False,
        )
        row_r = logits_r.float().numpy()[0, -1].astype(np.float32)
        tok_r = int(np.argmax(row_r))
        refeed_tokens.append(tok_r)
        refeed_logits.append(row_r)
        if tok != tok_r:
            break
        del logits_t, logits_r
        if hasattr(tc, "empty_cache"):
            tc.empty_cache()
    diffs = [
        float(np.max(np.abs(a - b)))
        for a, b in zip(cache_logits, refeed_logits)
    ]
    return {
        "requested_steps": int(steps),
        "completed_steps": int(len(cached_tokens)),
        "cached_tokens": [int(x) for x in cached_tokens],
        "refeed_tokens": [int(x) for x in refeed_tokens],
        "token_for_token_equal": bool(cached_tokens == refeed_tokens),
        "max_abs_logit_diffs": diffs,
        "max_abs_logit_diff": max(diffs) if diffs else None,
    }


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    reference_dir = Path(args.reference_dir).expanduser().resolve()
    dtype_tag = "fp32" if args.compute_dtype == "float32" else args.weight_mode
    out = (
        Path(args.output).expanduser().resolve()
        if args.output
        else ROOT
        / "artifacts"
        / "trinity_nano"
        / "tc_parity"
        / f"{dtype_tag}_{probe_name(args.probe_index)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    # Harness-level dtype plumb only — BlockTC/LinearTC already expose class knobs.
    BlockTC.COMPUTE_DTYPE = args.compute_dtype
    LinearTC.DTYPE = args.compute_dtype
    # Afmoe RMSNorm is cast-before-weight; fused kernel is Llama-order. Keep off.
    RMSNormTC.USE_FUSED = False
    QuantLinearTC.FUSED_DECODE = True

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True
    )
    reference = load_reference(reference_dir, args.probe_index)
    cfg = TrinityNanoConfig.from_model_dir(model_dir)
    payload: dict[str, Any] = {
        "schema": "trinity_nano_tc_parity_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_dir": str(model_dir),
        "reference_dir": str(reference_dir),
        "reference_probe": reference["path"],
        "probe_index": int(args.probe_index),
        "weight_mode": args.weight_mode,
        "compute_dtype": args.compute_dtype,
        "steps": int(args.steps),
        "decode_check_steps": int(args.decode_check_steps),
        "gpu_before": nvidia_smi(),
        "status": "starting",
    }
    write_json(out, payload)

    started = time.perf_counter()
    with tc.no_grad(), SafeTensorSource(model_dir) as source:
        runner = TrinityStreamRunner(
            cfg,
            source,
            weight_mode=args.weight_mode,
            route_detail=args.route_detail,
            empty_cache_interval=args.empty_cache_interval,
        )
        if args.skip_reference_compare:
            logits_receipts = []
            generated = []
            captures = {}
        else:
            logits_receipts, generated, captures = run_reference_steps(
                runner, reference, min(int(args.steps), len(reference["generated_ids"]))
            )
        kv_compare = (
            compare_prerope_kv(
                reference_dir, captures, compute_dtype=args.compute_dtype
            )
            if int(args.probe_index) == 0 and captures
            else {"status": "not_run"}
        )
        decode_check = run_incremental_refeed_check(
            runner,
            reference["input_ids"],
            int(args.decode_check_steps),
        )

    decoded = tokenizer.decode(
        [int(x) for x in generated],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    payload.update(
        {
            "status": "ok",
            "wall_seconds": time.perf_counter() - started,
            "gpu_after": nvidia_smi(),
            "reference_compare": {
                "generated_ids": [int(x) for x in generated],
                "generated_text": decoded,
                "steps": logits_receipts,
                "top5_exact_steps": int(sum(1 for r in logits_receipts if r["top5_exact"])),
                "top5_total_steps": int(len(logits_receipts)),
                "max_abs_diff": max(
                    (float(r["max_abs_diff"]) for r in logits_receipts),
                    default=None,
                ),
            },
            "prerope_kv_compare": kv_compare,
            "incremental_vs_refeed": decode_check,
        }
    )
    write_json(out, payload)
    print(json.dumps({"status": "ok", "artifact": str(out)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
