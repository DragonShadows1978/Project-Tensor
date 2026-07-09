#!/usr/bin/env python3
"""P3 envelope receipts for Trinity Nano: cache-path logic, T3 residency, T2 APA.

Hard rails (order):
  - no git commit; plan untouched
  - each GPU run wall <= 10 min; stage long prefill and stop on projected rail
  - harness/driver edits here; product APA dialect hook only (flagged)

Tasks (in order):
  1. CACHE-PATH: INT4-resident incremental decode == full refeed, 16 greedy tokens
  2. T3 RESIDENCY: staged prefill S in {8k, 32k, 96k, 131072}; peak VRAM + wall
  3. T2 APA: full-attention layers APA-engaged vs STANDARD at mid-length (8k);
     zero-flip + cheap ppl-proxy delta
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tensor_cuda"))

import tensor_cuda as tc  # noqa: E402
from core.trinity_nano_tc import (  # noqa: E402
    BlockTC,
    LinearTC,
    QuantLinearTC,
    RMSNormTC,
    TrinityNano_TC,
)


DEFAULT_MODEL_DIR = "/mnt/ForgeRealm/models/trinity-nano"
DEFAULT_OUT_DIR = "artifacts/trinity_nano/p3_envelope"
GPU_WALL_RAIL_S = 600.0  # 10 min per run
PREFILL_CHUNK = 512
STAGES = (8192, 32768, 98304, 131072)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--output-dir", default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--task",
        choices=("all", "cache", "t3", "t2", "cache_severity"),
        default="all",
        help="Run one receipt or all three in order (default all).",
    )
    p.add_argument("--decode-steps", type=int, default=16)
    p.add_argument("--cache-prompt-len", type=int, default=16)
    p.add_argument(
        "--compute-dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
        help=(
            "Activation/KV compute dtype after INT4-resident load. "
            "float32 = fp32 COMPUTE + INT4-dequant (weights stay int4). "
            "Use for cache-path logic disambiguation."
        ),
    )
    p.add_argument(
        "--break-on-mismatch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop cache-path at first token mismatch (default True). "
        "Severity runs force False to collect all steps.",
    )
    p.add_argument("--apa-prompt-len", type=int, default=8192)
    p.add_argument("--apa-gen-steps", type=int, default=16)
    p.add_argument("--prefill-chunk", type=int, default=PREFILL_CHUNK)
    p.add_argument("--wall-rail-s", type=float, default=GPU_WALL_RAIL_S)
    p.add_argument(
        "--ms-per-token-estimate",
        type=float,
        default=None,
        help="Optional override for rail projection (ms/token). Else measured.",
    )
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def nvidia_smi() -> str | None:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        return None


def parse_used_mib(smi: str | None) -> int | None:
    if not smi:
        return None
    # "NAME, used, total, util, power"
    parts = [x.strip() for x in smi.split(",")]
    if len(parts) < 2:
        return None
    try:
        return int(float(parts[1]))
    except Exception:
        return None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _iter_lineartc(model):
    """Yield every plain LinearTC on a resident TrinityNano_TC (router gates)."""
    for layer in getattr(model, "layers", []) or []:
        mlp = getattr(layer, "mlp", None)
        rg = getattr(mlp, "router_gate", None)
        if isinstance(rg, LinearTC):
            yield ("router_gate", rg)
        # shared/dense plain linears if any slipped past int4
        for name in ("gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"):
            for obj in (getattr(layer, "self_attn", None), mlp, getattr(mlp, "shared_experts", None)):
                if obj is None:
                    continue
                lin = getattr(obj, name, None)
                if isinstance(lin, LinearTC):
                    yield (name, lin)
        experts = getattr(mlp, "experts", None) or []
        for ex in experts:
            for name in ("gate_proj", "up_proj", "down_proj"):
                lin = getattr(ex, name, None)
                if isinstance(lin, LinearTC):
                    yield (f"expert.{name}", lin)
    lm = getattr(model, "lm_head", None)
    if isinstance(lm, LinearTC):
        yield ("lm_head", lm)


def set_compute_dtype(model, compute_dtype: str) -> dict[str, Any]:
    """Harness-level compute dtype switch after INT4-resident load.

    from_pretrained forces COMPUTE_DTYPE=bfloat16 (and may have built bf16
    rope tables). Rebuild tables in the requested compute dtype so activations
    and RoPE agree. Product also casts tables→x.dtype at apply as a safety net.

    INT4 QuantLinearTC dequants into x.dtype natively. Plain LinearTC
    (router_gate) stores wT at construct-time DTYPE — cast those to match.
    """
    dt = str(compute_dtype)
    BlockTC.COMPUTE_DTYPE = dt
    LinearTC.DTYPE = dt
    rope_note = "rope_not_present"
    if hasattr(model, "rope") and model.rope is not None:
        # Force rebuild on next extend under the new COMPUTE_DTYPE.
        prev_len = int(getattr(model.rope, "_rope_len", 0) or 0)
        prev_dtype = None
        if model.rope.cos is not None:
            prev_dtype = str(model.rope.cos.dtype)
        model.rope._rope_len = 0
        model.rope.cos = None
        model.rope.sin = None
        if prev_len > 0:
            model.rope.extend(prev_len)
        rope_note = (
            f"invalidated_and_rebuild_prev_len={prev_len}"
            f"_prev_dtype={prev_dtype}_now={dt}"
        )
    cast_n = 0
    cast_names: list[str] = []
    for name, lin in _iter_lineartc(model):
        if str(lin.wT.dtype) != dt:
            lin.wT = lin.wT.astype(dt)
            cast_n += 1
            if len(cast_names) < 8:
                cast_names.append(f"{name}:{dt}")
    return {
        "compute_dtype": dt,
        "weight_mode": "int4",
        "mode_label": (
            "fp32_compute_int4_dequant" if dt == "float32" else "bf16_compute_int4_dequant"
        ),
        "rope_rebuild": rope_note,
        "lineartc_cast_count": cast_n,
        "lineartc_cast_sample": cast_names,
        "product_cast_seam": (
            "core/trinity_nano_tc.py TrinityAttentionTC: cast cos/sin to q.dtype "
            "before rope_apply (flagged cast-compatibility)"
        ),
    }


def load_model(
    model_dir: str, *, compute_dtype: str = "bfloat16"
) -> tuple[Any, dict[str, Any], float, dict[str, Any]]:
    BlockTC.COMPUTE_DTYPE = "bfloat16"
    LinearTC.DTYPE = "bfloat16"
    QuantLinearTC.FUSED_DECODE = True
    RMSNormTC.USE_FUSED = False
    t0 = time.perf_counter()
    with tc.no_grad():
        model, info = TrinityNano_TC.from_pretrained(
            model_dir, weight_mode="int4", load_lm_head=True, progress=False
        )
        model.configure_moe_empty_cache(0)
        dtype_meta = set_compute_dtype(model, compute_dtype)
        tc.synchronize()
    return model, info, time.perf_counter() - t0, dtype_meta


def free_model(model) -> None:
    del model
    if hasattr(tc, "empty_cache"):
        tc.empty_cache()
    try:
        tc.synchronize()
    except Exception:
        pass


def top2_margin(row: np.ndarray) -> dict[str, Any]:
    """Per-step near-tie metric: top1-top2 logit gap + ids."""
    r = np.asarray(row, dtype=np.float64).reshape(-1)
    top2 = np.argpartition(r, -2)[-2:]
    # sort those two descending
    order = top2[np.argsort(r[top2])[::-1]]
    t1, t2 = int(order[0]), int(order[1])
    v1, v2 = float(r[t1]), float(r[t2])
    return {
        "top1_id": t1,
        "top2_id": t2,
        "top1_logit": v1,
        "top2_logit": v2,
        "margin_top1_minus_top2": float(v1 - v2),
    }


def kv_cache_lengths(caches) -> list[int] | None:
    if caches is None:
        return None
    lengths = []
    for c in caches:
        if c is None:
            lengths.append(0)
            continue
        # cache is (k, v); k shape [B, H, S, D]
        try:
            lengths.append(int(c[0].shape[2]))
        except Exception:
            lengths.append(-1)
    return lengths


def make_ids(length: int, seed: int, vocab_cap: int = 200000) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    # Avoid special extremes; keep ids in a dense mid-vocab band.
    ids = rng.integers(32, min(vocab_cap, 50000), size=(1, int(length)), dtype=np.int64)
    return ids


def chunked_prefill(
    model,
    input_ids: np.ndarray,
    *,
    chunk: int,
    last_token_only: bool = True,
    sample_smi_every: int = 1,
) -> tuple[Any, list, dict[str, Any]]:
    """Harness-level chunked prefill (product auto-chunk not required)."""
    input_ids = np.asarray(input_ids, dtype=np.int64)
    B, L = input_ids.shape
    caches = None
    logits = None
    peak_mib = parse_used_mib(nvidia_smi()) or 0
    samples = []
    t0 = time.perf_counter()
    off = 0
    n_chunks = 0
    while off < L:
        # Bound score workspace ~ as Gemma: shrink as S grows.
        s_ctx = max(1, off + int(chunk))
        step = min(int(chunk), max(64, int(300 * 1024 * 1024 // (max(1, 8 * s_ctx * 2)))))
        step = max(64, (step // 64) * 64)
        step = min(step, L - off)
        seg = input_ids[:, off : off + step]
        logits, caches = model(
            seg,
            kv_caches=caches,
            position_offset=off,
            last_token_only=last_token_only,
        )
        tc.synchronize()
        off += step
        n_chunks += 1
        if sample_smi_every > 0 and (n_chunks % sample_smi_every == 0 or off >= L):
            smi = nvidia_smi()
            used = parse_used_mib(smi)
            if used is not None:
                peak_mib = max(peak_mib, used)
            samples.append({"offset": int(off), "smi": smi, "used_mib": used})
        if hasattr(tc, "empty_cache") and off < L:
            # free transient score buffers between chunks; keep KV caches
            pass
    wall = time.perf_counter() - t0
    return logits, caches, {
        "wall_seconds": float(wall),
        "n_chunks": int(n_chunks),
        "peak_used_mib": int(peak_mib),
        "smi_samples": samples[-8:],  # tail only
        "prompt_len": int(L),
        "chunk": int(chunk),
    }


def greedy_from_logits(logits) -> int:
    row = logits.float().numpy()[0, -1]
    return int(np.argmax(row))


def logprob_of_token(logits, token_id: int) -> float:
    row = logits.float().numpy()[0, -1].astype(np.float64)
    # stable log-softmax
    m = float(row.max())
    ex = np.exp(row - m)
    z = float(ex.sum())
    return float((row[int(token_id)] - m) - math.log(z))


# ---------------------------------------------------------------------------
# Task 1: cache-path logic
# ---------------------------------------------------------------------------
def run_cache_path(
    model,
    *,
    prompt_len: int,
    steps: int,
    seed: int,
    out: Path,
    compute_dtype: str = "bfloat16",
    break_on_mismatch: bool = True,
    collect_severity: bool = False,
    dtype_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "trinity_nano_p3_cache_path_v2",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "prompt_len": int(prompt_len),
        "steps": int(steps),
        "weight_mode": "int4",
        "compute_dtype": str(compute_dtype),
        "mode_label": (
            "fp32_compute_int4_dequant"
            if compute_dtype == "float32"
            else "bf16_compute_int4_dequant"
        ),
        "break_on_mismatch": bool(break_on_mismatch),
        "collect_severity": bool(collect_severity),
        "dtype_meta": dtype_meta or {},
        "gpu_before": nvidia_smi(),
        "status": "running",
    }
    write_json(out, payload)
    ids = make_ids(prompt_len, seed)
    cached_tokens: list[int] = []
    refeed_tokens: list[int] = []
    cache_logits: list[np.ndarray] = []
    refeed_logits: list[np.ndarray] = []
    step_receipts: list[dict[str, Any]] = []
    first_mismatch_dump: dict[str, Any] | None = None
    t0 = time.perf_counter()
    with tc.no_grad():
        # incremental path
        caches = None
        current = ids
        for step in range(int(steps)):
            pos_off = 0 if step == 0 else int(ids.shape[1] + step - 1)
            logits, caches = model(
                current,
                kv_caches=caches,
                position_offset=pos_off,
                last_token_only=True,
            )
            tc.synchronize()
            row = logits.float().numpy()[0, -1].astype(np.float32)
            tok = int(np.argmax(row))
            cached_tokens.append(tok)
            cache_logits.append(row)
            cache_lens = kv_cache_lengths(caches)

            # full refeed of prefix + generated-so-far-except-last:
            # refeed prefix + cached_tokens[:-1] -> predicts same next as cache step
            full = np.concatenate(
                [ids, np.asarray(cached_tokens[:-1], dtype=np.int64).reshape(1, -1)],
                axis=1,
            )
            logits_r, caches_r = model(
                full, kv_caches=None, position_offset=0, last_token_only=True
            )
            tc.synchronize()
            row_r = logits_r.float().numpy()[0, -1].astype(np.float32)
            tok_r = int(np.argmax(row_r))
            refeed_tokens.append(tok_r)
            refeed_logits.append(row_r)
            refeed_lens = kv_cache_lengths(caches_r)

            max_abs = float(np.max(np.abs(row.astype(np.float64) - row_r.astype(np.float64))))
            # cross-arm: logit of each arm's chosen id under the other arm
            cache_m = top2_margin(row)
            refeed_m = top2_margin(row_r)
            step_rec: dict[str, Any] = {
                "step": int(step),
                "position_offset_cache": int(pos_off),
                "input_len_cache": int(current.shape[1]),
                "input_len_refeed": int(full.shape[1]),
                "seq_len_implied": int(ids.shape[1] + step),  # tokens seen after this step
                "cached_tok": int(tok),
                "refeed_tok": int(tok_r),
                "match": bool(tok == tok_r),
                "max_abs_logit_diff": max_abs,
                "cache_kv_len_min": int(min(cache_lens)) if cache_lens else None,
                "cache_kv_len_max": int(max(cache_lens)) if cache_lens else None,
                "refeed_kv_len_min": int(min(refeed_lens)) if refeed_lens else None,
                "refeed_kv_len_max": int(max(refeed_lens)) if refeed_lens else None,
            }
            if collect_severity:
                step_rec["cache_top2"] = cache_m
                step_rec["refeed_top2"] = refeed_m
                step_rec["cache_logit_at_refeed_tok"] = float(row[tok_r])
                step_rec["refeed_logit_at_cache_tok"] = float(row_r[tok])
                step_rec["cross_gap_cache_pref_minus_refeed_pref"] = float(
                    row[tok] - row[tok_r]
                )
                step_rec["cross_gap_refeed_pref_minus_cache_pref"] = float(
                    row_r[tok_r] - row_r[tok]
                )
            step_receipts.append(step_rec)

            if tok != tok_r and first_mismatch_dump is None:
                first_mismatch_dump = {
                    "first_mismatch_step": int(step),
                    "position_offset_cache": int(pos_off),
                    "input_len_cache": int(current.shape[1]),
                    "input_len_refeed": int(full.shape[1]),
                    "cached_tok": int(tok),
                    "refeed_tok": int(tok_r),
                    "max_abs_logit_diff": max_abs,
                    "cache_kv_lengths_head8": (cache_lens or [])[:8],
                    "cache_kv_lengths_tail8": (cache_lens or [])[-8:],
                    "refeed_kv_lengths_head8": (refeed_lens or [])[:8],
                    "refeed_kv_lengths_tail8": (refeed_lens or [])[-8:],
                    "n_layers_cache_kv": len(cache_lens or []),
                    "cache_top2": cache_m,
                    "refeed_top2": refeed_m,
                    "prompt_ids_head16": [int(x) for x in ids.reshape(-1)[:16]],
                    "generated_prefix_cache": [int(x) for x in cached_tokens[:-1]],
                }
                if break_on_mismatch:
                    current = np.asarray([[tok]], dtype=np.int64)
                    break

            current = np.asarray([[tok]], dtype=np.int64)
            # drop refeed caches promptly
            del caches_r, logits_r
    diffs = [float(np.max(np.abs(a - b))) for a, b in zip(cache_logits, refeed_logits)]
    equal = cached_tokens == refeed_tokens and len(cached_tokens) == int(steps)
    # severity summary if collected
    severity = None
    if collect_severity and step_receipts:
        margins_c = [s["cache_top2"]["margin_top1_minus_top2"] for s in step_receipts]
        margins_r = [s["refeed_top2"]["margin_top1_minus_top2"] for s in step_receipts]
        flip_steps = [s["step"] for s in step_receipts if not s["match"]]
        severity = {
            "n_steps": len(step_receipts),
            "n_flips": len(flip_steps),
            "flip_steps": flip_steps,
            "min_margin_cache": float(min(margins_c)),
            "min_margin_refeed": float(min(margins_r)),
            "median_margin_cache": float(np.median(margins_c)),
            "median_margin_refeed": float(np.median(margins_r)),
            "margins_cache": margins_c,
            "margins_refeed": margins_r,
            "note": (
                "margin = top1_logit - top2_logit on that arm; "
                "small margin + flip ⇒ length-instability near-tie, not large semantic miss"
            ),
        }
    payload.update(
        {
            "status": "ok" if equal else "mismatch",
            "wall_seconds": float(time.perf_counter() - t0),
            "gpu_after": nvidia_smi(),
            "completed_steps": int(len(cached_tokens)),
            "cached_tokens": [int(x) for x in cached_tokens],
            "refeed_tokens": [int(x) for x in refeed_tokens],
            "token_for_token_equal": bool(equal),
            "max_abs_logit_diffs": diffs,
            "max_abs_logit_diff": max(diffs) if diffs else None,
            "first_mismatch_step": (
                next(
                    (i for i, (a, b) in enumerate(zip(cached_tokens, refeed_tokens)) if a != b),
                    None,
                )
            ),
            "first_mismatch_dump": first_mismatch_dump,
            "step_receipts": step_receipts if (collect_severity or first_mismatch_dump) else [
                {
                    "step": s["step"],
                    "cached_tok": s["cached_tok"],
                    "refeed_tok": s["refeed_tok"],
                    "match": s["match"],
                    "max_abs_logit_diff": s["max_abs_logit_diff"],
                    "position_offset_cache": s["position_offset_cache"],
                    "input_len_cache": s["input_len_cache"],
                    "input_len_refeed": s["input_len_refeed"],
                }
                for s in step_receipts
            ],
            "severity": severity,
        }
    )
    write_json(out, payload)
    return payload


# ---------------------------------------------------------------------------
# Task 2: T3 staged residency
# ---------------------------------------------------------------------------
def run_t3_residency(
    model,
    *,
    stages: tuple[int, ...],
    chunk: int,
    wall_rail_s: float,
    ms_per_token_estimate: float | None,
    seed: int,
    out: Path,
    compute_dtype: str = "float32",
    dtype_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # KV storage follows COMPUTE_DTYPE (no separate cast path). Plan T3
    # assumed 2-byte fp16 KV; fp32 compute stores 4-byte KV. Both theoretical
    # footprints + measured-peak linear extrapolations are ledgered.
    kv_bytes = 4 if str(compute_dtype) == "float32" else 2
    payload: dict[str, Any] = {
        "schema": "trinity_nano_p3_t3_residency_v2",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stages_requested": [int(s) for s in stages],
        "chunk": int(chunk),
        "wall_rail_s": float(wall_rail_s),
        "weight_mode": "int4",
        "compute_dtype": str(compute_dtype),
        "mode_label": (
            "fp32_compute_int4_dequant"
            if compute_dtype == "float32"
            else "bf16_compute_int4_dequant"
        ),
        "dtype_meta": dtype_meta or {},
        "kv_dtype_note": (
            f"activations/KV use BlockTC.COMPUTE_DTYPE={compute_dtype} "
            f"({kv_bytes}-byte storage). Plan T3 assumed fp16/bf16 (2-byte) KV; "
            f"fp16-equivalent footprint is half of measured KV delta under fp32."
        ),
        "kv_storage_bytes_per_elem": int(kv_bytes),
        "gpu_before": nvidia_smi(),
        "status": "running",
        "stage_receipts": [],
        "rail_stop": None,
        "projection_math": [],
    }
    write_json(out, payload)

    # Calibration micro-prefill for projection (if estimate not provided).
    ms_tok = ms_per_token_estimate
    if ms_tok is None:
        cal_L = 512
        ids = make_ids(cal_L, seed)
        with tc.no_grad():
            _, _, meta = chunked_prefill(model, ids, chunk=chunk, last_token_only=True)
        ms_tok = 1000.0 * float(meta["wall_seconds"]) / float(cal_L)
        payload["calibration"] = {
            "prompt_len": cal_L,
            "wall_seconds": meta["wall_seconds"],
            "ms_per_token": ms_tok,
            "peak_used_mib": meta["peak_used_mib"],
            "gpu_after": nvidia_smi(),
        }
        write_json(out, payload)

    stage_receipts = []
    last_passed = None
    last_peak_mib = None
    projection_math: list[dict[str, Any]] = []
    for S in stages:
        projected_s = (float(ms_tok) / 1000.0) * float(S)
        proj_entry = {
            "S": int(S),
            "ms_per_token_source": (
                "calibration"
                if last_passed is None and ms_per_token_estimate is None
                else (
                    "cli_override"
                    if last_passed is None and ms_per_token_estimate is not None
                    else f"measured_at_S={last_passed}"
                )
            ),
            "ms_per_token": float(ms_tok),
            "projected_wall_s": float(projected_s),
            "rail_s": float(wall_rail_s),
            "formula": f"projected_wall_s = (ms_per_token/1000)*S = ({ms_tok}/1000)*{S}",
            "under_rail": bool(projected_s <= float(wall_rail_s)),
        }
        projection_math.append(proj_entry)
        stage: dict[str, Any] = {
            "S": int(S),
            "projected_wall_s": float(projected_s),
            "ms_per_token_used": float(ms_tok),
            "projection": proj_entry,
        }
        if projected_s > float(wall_rail_s):
            stage["status"] = "skipped_projected_over_rail"
            stage_receipts.append(stage)
            payload["rail_stop"] = {
                "at_S": int(S),
                "reason": "projected_wall_exceeds_10min_rail",
                "projected_wall_s": float(projected_s),
                "ms_per_token_used": float(ms_tok),
                "last_passed_S": last_passed,
                "last_peak_used_mib": last_peak_mib,
                "projection_formula": proj_entry["formula"],
            }
            # Do not attempt remaining larger stages either.
            for S2 in stages[stages.index(S) + 1 :]:
                proj2 = (float(ms_tok) / 1000.0) * float(S2)
                stage_receipts.append(
                    {
                        "S": int(S2),
                        "projected_wall_s": float(proj2),
                        "ms_per_token_used": float(ms_tok),
                        "status": "skipped_after_rail_stop",
                        "projection": {
                            "S": int(S2),
                            "ms_per_token": float(ms_tok),
                            "projected_wall_s": float(proj2),
                            "formula": (
                                f"projected_wall_s = (ms_per_token/1000)*S = "
                                f"({ms_tok}/1000)*{S2}"
                            ),
                            "under_rail": bool(proj2 <= float(wall_rail_s)),
                        },
                    }
                )
            break

        ids = make_ids(S, seed + int(S))
        gpu_pre = nvidia_smi()
        with tc.no_grad():
            logits, caches, meta = chunked_prefill(
                model, ids, chunk=chunk, last_token_only=True, sample_smi_every=2
            )
            # touch logits to keep graph live until smi sample
            _ = int(np.argmax(logits.float().numpy()[0, -1]))
            gpu_peak = nvidia_smi()
            peak_from_smi = parse_used_mib(gpu_peak)
            peak_mib = int(
                max(
                    int(meta["peak_used_mib"] or 0),
                    int(peak_from_smi or 0),
                )
            )
            # drop caches before next stage
            del logits, caches
            if hasattr(tc, "empty_cache"):
                tc.empty_cache()
            tc.synchronize()
        gpu_after_free = nvidia_smi()
        stage.update(
            {
                "status": "ran",
                "wall_seconds": meta["wall_seconds"],
                "n_chunks": meta["n_chunks"],
                "peak_used_mib": peak_mib,
                "gpu_before": gpu_pre,
                "gpu_after_prefill": gpu_peak,
                "gpu_after_free": gpu_after_free,
                "ms_per_token_measured": 1000.0 * meta["wall_seconds"] / float(S),
                "tokens_per_sec": float(S) / float(meta["wall_seconds"]),
            }
        )
        # Refresh projection from measured stage (more accurate for next).
        ms_tok = float(stage["ms_per_token_measured"])
        last_passed = int(S)
        last_peak_mib = peak_mib
        stage_receipts.append(stage)
        payload["stage_receipts"] = stage_receipts
        payload["projection_math"] = projection_math
        write_json(out, payload)

    full_confirmed = any(
        r.get("S") == 131072 and r.get("status") == "ran" for r in stage_receipts
    )
    ceiling = last_passed
    extrap = _t3_extrapolation(
        stage_receipts, ms_tok, compute_dtype=str(compute_dtype), last_peak_mib=last_peak_mib
    )
    # Split / full / red verdict vs plan T3 (131k residency).
    vram_131 = extrap.get("vram_at_131072") or {}
    vram_fits = bool(vram_131.get("fits_12282_mib"))
    if full_confirmed:
        t3_verdict = "FULL_CONFIRMED_AT_131072"
    elif ceiling is not None and vram_fits:
        t3_verdict = (
            f"VRAM-CONFIRMED-BY-EXTRAPOLATION at S={ceiling}, wall-blocked"
        )
    elif ceiling is not None and not vram_fits:
        t3_verdict = (
            f"WALL_AND_OR_VRAM_OPEN at S={ceiling}; "
            f"extrap_peak_131k={vram_131.get('peak_used_mib_extrap')} MiB"
        )
    else:
        t3_verdict = "NO_STAGE_RAN (calibration-only or load failure)"

    payload.update(
        {
            "status": "ok",
            "stage_receipts": stage_receipts,
            "projection_math": projection_math,
            "t3_full_131k_confirmed": bool(full_confirmed),
            "ceiling_S_reached": ceiling,
            "t3_verdict": t3_verdict,
            "gpu_final": nvidia_smi(),
            "extrapolation": extrap,
        }
    )
    write_json(out, payload)
    return payload


def _kv_mib_theoretical(S: int, bytes_per: int) -> float:
    """Theoretical KV footprint: 14 full unbounded + 42 sliding@min(S,2048)."""
    full, slide, d, kvh = 14, 42, 128, 2
    full_bytes = full * 2 * kvh * int(S) * d * int(bytes_per)
    slide_bytes = slide * 2 * kvh * min(int(S), 2048) * d * int(bytes_per)
    return (full_bytes + slide_bytes) / (1024.0 * 1024.0)


def _t3_extrapolation(
    stage_receipts: list[dict],
    ms_tok: float,
    *,
    compute_dtype: str = "float32",
    last_peak_mib: int | None = None,
) -> dict[str, Any]:
    ran = [r for r in stage_receipts if r.get("status") == "ran"]
    bytes_per = 4 if compute_dtype == "float32" else 2
    out: dict[str, Any] = {
        "ms_per_token": float(ms_tok),
        "compute_dtype": str(compute_dtype),
        "kv_bytes_per_elem": int(bytes_per),
        "targets": {},
    }
    for S in (8192, 32768, 98304, 131072):
        hit = next((r for r in ran if int(r["S"]) == S), None)
        if hit is not None:
            out["targets"][str(S)] = {
                "source": "measured",
                "wall_s": hit["wall_seconds"],
                "peak_used_mib": hit.get("peak_used_mib"),
            }
        else:
            out["targets"][str(S)] = {
                "source": "extrapolated_linear_ms_tok",
                "wall_s": (float(ms_tok) / 1000.0) * float(S),
                "peak_used_mib": None,
            }

    out["kv_mib_model_compute_dtype"] = {
        str(S): _kv_mib_theoretical(S, bytes_per)
        for S in (8192, 32768, 98304, 131072)
    }
    out["kv_mib_model_fp16_plan"] = {
        str(S): _kv_mib_theoretical(S, 2) for S in (8192, 32768, 98304, 131072)
    }
    out["weight_resident_mib_prior"] = 3398  # from int4_all_resident_load

    # Linear peak VRAM from measured stages: peak ≈ a + b*S_full_effective.
    # Use S as x (sliding saturates at 2048; for S>>2k full layers dominate).
    vram_fit: dict[str, Any] = {"n_points": len(ran), "method": None}
    if len(ran) >= 2:
        xs = np.array([float(r["S"]) for r in ran], dtype=np.float64)
        ys = np.array([float(r["peak_used_mib"]) for r in ran], dtype=np.float64)
        # least-squares peak = a + b*S
        A = np.vstack([np.ones_like(xs), xs]).T
        coef, _, _, _ = np.linalg.lstsq(A, ys, rcond=None)
        a, b = float(coef[0]), float(coef[1])
        vram_fit.update(
            {
                "method": "lstsq_peak_a_plus_b_S",
                "a_mib": a,
                "b_mib_per_token": b,
                "formula": "peak_used_mib ≈ a + b*S",
                "points": [
                    {"S": int(r["S"]), "peak_used_mib": r.get("peak_used_mib")}
                    for r in ran
                ],
            }
        )
        for S in (8192, 32768, 98304, 131072):
            pred = a + b * float(S)
            tgt = out["targets"][str(S)]
            if tgt.get("peak_used_mib") is None:
                tgt["peak_used_mib"] = float(pred)
                tgt["peak_source"] = "linear_extrap_from_measured_stages"
            else:
                tgt["peak_source"] = "measured"
                tgt["peak_used_mib_linear_check"] = float(pred)
    elif len(ran) == 1:
        # Single-point: baseline = free-after-ish weight floor; grow with theoretical KV delta.
        r0 = ran[0]
        S0 = int(r0["S"])
        p0 = float(r0["peak_used_mib"])
        kv0 = _kv_mib_theoretical(S0, bytes_per)
        # residual non-KV (weights + workspace) at S0
        residual = p0 - kv0
        vram_fit.update(
            {
                "method": "single_point_plus_theoretical_kv_delta",
                "S0": S0,
                "peak0_mib": p0,
                "kv0_mib": kv0,
                "residual_non_kv_mib": residual,
                "formula": (
                    "peak(S) ≈ peak(S0) + (kv_theory(S)-kv_theory(S0)) "
                    f"[bytes_per={bytes_per}]"
                ),
            }
        )
        for S in (8192, 32768, 98304, 131072):
            pred = p0 + (_kv_mib_theoretical(S, bytes_per) - kv0)
            tgt = out["targets"][str(S)]
            if tgt.get("peak_used_mib") is None:
                tgt["peak_used_mib"] = float(pred)
                tgt["peak_source"] = "single_point_kv_delta"
            else:
                tgt["peak_source"] = "measured"
    out["vram_fit"] = vram_fit

    peak_131 = out["targets"]["131072"].get("peak_used_mib")
    # Also report fp16-equivalent peak: if we measured under fp32, halve KV portion.
    fp16_peak_131 = None
    if peak_131 is not None and ran:
        if bytes_per == 4:
            # peak_fp16 ≈ peak_fp32 - 0.5 * kv_fp32_theory(S_last_or_131)
            # Better: residual + kv_fp16(131)
            if len(ran) >= 1:
                r0 = ran[-1]
                S0 = int(r0["S"])
                p0 = float(r0["peak_used_mib"])
                residual = p0 - _kv_mib_theoretical(S0, 4)
                fp16_peak_131 = residual + _kv_mib_theoretical(131072, 2)
        else:
            fp16_peak_131 = float(peak_131)

    out["vram_at_131072"] = {
        "peak_used_mib_extrap": None if peak_131 is None else float(peak_131),
        "peak_source": out["targets"]["131072"].get("peak_source"),
        "budget_mib": 12282,
        "fits_12282_mib": (
            None if peak_131 is None else bool(float(peak_131) <= 12282.0)
        ),
        "fp16_equivalent_peak_mib_extrap": (
            None if fp16_peak_131 is None else float(fp16_peak_131)
        ),
        "fp16_equivalent_fits_12282_mib": (
            None
            if fp16_peak_131 is None
            else bool(float(fp16_peak_131) <= 12282.0)
        ),
        "wall_s_extrap": out["targets"]["131072"].get("wall_s"),
        "last_measured_S": int(ran[-1]["S"]) if ran else None,
        "last_measured_peak_mib": last_peak_mib,
    }
    return out


# ---------------------------------------------------------------------------
# Task 3: T2 APA zero-flip
# ---------------------------------------------------------------------------
def run_t2_apa(
    model,
    *,
    prompt_len: int,
    gen_steps: int,
    seed: int,
    out: Path,
    compute_dtype: str = "float32",
    dtype_meta: dict[str, Any] | None = None,
    chunk: int | None = None,
) -> dict[str, Any]:
    full_idx = [
        int(layer.layer_idx)
        for layer in model.layers
        if not layer.self_attn.is_local_attention
    ]
    slide_idx = [
        int(layer.layer_idx)
        for layer in model.layers
        if layer.self_attn.is_local_attention
    ]
    payload: dict[str, Any] = {
        "schema": "trinity_nano_p3_t2_apa_v2",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "prompt_len": int(prompt_len),
        "gen_steps": int(gen_steps),
        "weight_mode": "int4",
        "compute_dtype": str(compute_dtype),
        "mode_label": (
            "fp32_compute_int4_dequant"
            if compute_dtype == "float32"
            else "bf16_compute_int4_dequant"
        ),
        "dtype_meta": dtype_meta or {},
        "apa_min_context_default": 2048,
        "full_only": True,
        "geometry": {
            "full_layers": len(full_idx),
            "full_layer_indices": full_idx,
            "sliding_layers": len(slide_idx),
            "sliding_layer_indices": slide_idx,
            "kv_heads": 2,
            "head_dim": 128,
            "unbounded": True,
            "sliding_held_standard": True,
        },
        "gpu_before": nvidia_smi(),
        "status": "running",
    }
    write_json(out, payload)
    ids = make_ids(prompt_len, seed)
    prefill_chunk = int(chunk if chunk is not None else PREFILL_CHUNK)

    def _layer_backends() -> list[dict[str, Any]]:
        rows = []
        for layer in model.layers:
            att = layer.self_attn
            rows.append(
                {
                    "layer_idx": int(layer.layer_idx),
                    "is_sliding": bool(att.is_local_attention),
                    "attention_mode": str(att.attention_mode),
                    "last_attention_backend": str(att.last_attention_backend),
                    "apa_min_context": int(att.apa_min_context),
                }
            )
        return rows

    def _run(mode: str) -> dict[str, Any]:
        engage = model.set_attention_mode(
            mode,
            refine_percentile=0.15,
            bulk_bits=4,
            apa_min_context=2048,
            full_only=True,
        )
        tokens: list[int] = []
        logprobs: list[float] = []
        step_rows: list[np.ndarray] = []
        backends: dict[str, int] = {}
        t0 = time.perf_counter()
        with tc.no_grad():
            logits, caches, meta = chunked_prefill(
                model, ids, chunk=prefill_chunk, last_token_only=True
            )
            per_layer = _layer_backends()
            for row in per_layer:
                b = row["last_attention_backend"]
                backends[str(b)] = backends.get(str(b), 0) + 1
            for step in range(int(gen_steps)):
                row = logits.float().numpy()[0, -1].astype(np.float64)
                tok = int(np.argmax(row))
                tokens.append(tok)
                logprobs.append(logprob_of_token(logits, tok))
                step_rows.append(row)
                logits, caches = model(
                    np.asarray([[tok]], dtype=np.int64),
                    kv_caches=caches,
                    position_offset=ids.shape[1] + step,
                    last_token_only=True,
                )
                tc.synchronize()
            del caches
            if hasattr(tc, "empty_cache"):
                tc.empty_cache()
        return {
            "mode": mode,
            "engage": engage,
            "tokens": tokens,
            "step_logits": step_rows,  # kept in-memory only for flip gaps
            "mean_nll": float(-np.mean(logprobs)) if logprobs else None,
            "sum_logprob": float(sum(logprobs)),
            "wall_seconds": float(time.perf_counter() - t0),
            "prefill_wall_seconds": float(meta["wall_seconds"]),
            "prefill_peak_used_mib": meta["peak_used_mib"],
            "backends_after_prefill": backends,
            "per_layer_backends_after_prefill": per_layer,
            "gpu_after": nvidia_smi(),
        }

    # STANDARD first, then APA — same prompt/seed (A then B)
    std = _run("standard")
    apa = _run("apa_selective")

    flip_details: list[dict[str, Any]] = []
    for i, (a, b) in enumerate(zip(std["tokens"], apa["tokens"])):
        if a == b:
            continue
        row_s = std["step_logits"][i]
        row_a = apa["step_logits"][i]
        # logit gaps on both arms for the two chosen tokens
        flip_details.append(
            {
                "step": int(i),
                "std_tok": int(a),
                "apa_tok": int(b),
                "std_logit_at_std": float(row_s[a]),
                "std_logit_at_apa": float(row_s[b]),
                "apa_logit_at_apa": float(row_a[b]),
                "apa_logit_at_std": float(row_a[a]),
                "std_gap_pref_minus_other": float(row_s[a] - row_s[b]),
                "apa_gap_pref_minus_other": float(row_a[b] - row_a[a]),
                "cross_arm_max_abs_logit_diff": float(
                    np.max(np.abs(row_s - row_a))
                ),
                "std_top2": top2_margin(row_s),
                "apa_top2": top2_margin(row_a),
            }
        )
    flips = len(flip_details)
    zero_flip = flips == 0 and len(std["tokens"]) == len(apa["tokens"]) == gen_steps
    ppl_delta = None
    if std["mean_nll"] is not None and apa["mean_nll"] is not None:
        ppl_delta = float(apa["mean_nll"] - std["mean_nll"])

    # Per-layer engagement: APA must fire on exactly the 14 full layers and
    # MUST NOT fire on any sliding layer.
    apa_layers = apa["per_layer_backends_after_prefill"]
    full_apa = [
        r for r in apa_layers
        if (not r["is_sliding"]) and "apa_selective" in str(r["last_attention_backend"])
    ]
    full_non_apa = [
        r for r in apa_layers
        if (not r["is_sliding"]) and "apa_selective" not in str(r["last_attention_backend"])
    ]
    slide_apa_leak = [
        r for r in apa_layers
        if r["is_sliding"] and "apa_selective" in str(r["last_attention_backend"])
    ]
    slide_standard = [
        r for r in apa_layers
        if r["is_sliding"] and "apa_selective" not in str(r["last_attention_backend"])
    ]
    n_full = len(full_idx)
    apa_fired_exactly_14 = (
        len(full_apa) == n_full
        and len(full_non_apa) == 0
        and len(slide_apa_leak) == 0
        and n_full == 14
    )
    engagement = {
        "n_full_layers_expected": n_full,
        "n_full_apa_fired": len(full_apa),
        "full_apa_layer_indices": [r["layer_idx"] for r in full_apa],
        "full_non_apa_layer_indices": [r["layer_idx"] for r in full_non_apa],
        "n_sliding_standard": len(slide_standard),
        "n_sliding_apa_leak": len(slide_apa_leak),
        "sliding_apa_leak_indices": [r["layer_idx"] for r in slide_apa_leak],
        "apa_fired_exactly_14_full_not_sliding": bool(apa_fired_exactly_14),
        "backends_histogram_apa": apa["backends_after_prefill"],
        "backends_histogram_std": std["backends_after_prefill"],
    }

    if not engagement["apa_fired_exactly_14_full_not_sliding"]:
        status = "apa_engagement_fail"
    elif not zero_flip:
        # Flip under fp32 is a FINDING about APA-on-NoPE, not auto-failure.
        status = "token_flip_finding"
    else:
        status = "ok"

    # Drop heavy logit arrays from serialised arms
    def _arm_public(arm: dict[str, Any]) -> dict[str, Any]:
        return {
            k: arm[k]
            for k in (
                "tokens",
                "mean_nll",
                "sum_logprob",
                "wall_seconds",
                "prefill_wall_seconds",
                "prefill_peak_used_mib",
                "backends_after_prefill",
                "per_layer_backends_after_prefill",
                "engage",
            )
        }

    payload.update(
        {
            "status": status,
            "standard": _arm_public(std),
            "apa": _arm_public(apa),
            "zero_flip": bool(zero_flip),
            "n_flips": int(flips),
            "flip_details": flip_details,
            "ppl_proxy_mean_nll_delta_apa_minus_std": ppl_delta,
            "apa_fired_on_full_layers": bool(len(full_apa) == n_full and n_full > 0),
            "engagement": engagement,
            "wall_delta_apa_minus_std_s": float(
                apa["wall_seconds"] - std["wall_seconds"]
            ),
            "gpu_final": nvidia_smi(),
            "t2_verdict_note": (
                "Plan T2: APA engages clean on 14 full layers; sliding out of scope. "
                "zero-flip is the standard gate; under fp32 a flip is a finding "
                "(logit gaps ledgered), not an automatic failure — no knob tuning."
            ),
        }
    )
    write_json(out, payload)
    return payload


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # cache_severity is an INT4 bf16-compute full-step near-tie receipt
    run_severity = args.task == "cache_severity"
    compute_dtype = "bfloat16" if run_severity else args.compute_dtype
    if run_severity:
        # Force INT4+bf16 compute, no early break, severity margins on.
        compute_dtype = "bfloat16"
    summary: dict[str, Any] = {
        "schema": "trinity_nano_p3_envelope_summary_v2",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task": args.task,
        "model_dir": str(Path(args.model_dir).expanduser().resolve()),
        "wall_rail_s": float(args.wall_rail_s),
        "compute_dtype_requested": args.compute_dtype,
        "compute_dtype_effective": compute_dtype,
    }

    model, info, load_s, dtype_meta = load_model(
        args.model_dir, compute_dtype=compute_dtype
    )
    summary["load_wall_seconds"] = load_s
    summary["dtype_meta"] = dtype_meta
    summary["loader_info"] = {
        k: info[k]
        for k in (
            "weight_mode",
            "layers",
            "quantized_linear_bytes",
            "full_attention_layers",
            "num_key_value_heads",
        )
        if k in info
    }
    summary["gpu_after_load"] = nvidia_smi()
    write_json(out_dir / f"summary_{stamp}.json", summary)

    try:
        if args.task in ("all", "cache", "cache_severity"):
            tag = "cache_severity" if run_severity else (
                f"cache_path_{'fp32' if compute_dtype == 'float32' else 'bf16'}"
            )
            cache_out = out_dir / f"{tag}_{stamp}.json"
            print(
                f"[P3] cache-path mode={dtype_meta.get('mode_label')} -> {cache_out}",
                flush=True,
            )
            break_on = False if run_severity else bool(args.break_on_mismatch)
            # On float32 logic check: break on mismatch (real bug dump).
            # On severity: run all steps with margins.
            collect_sev = bool(run_severity)
            cache_res = run_cache_path(
                model,
                prompt_len=args.cache_prompt_len,
                steps=args.decode_steps,
                seed=args.seed,
                out=cache_out,
                compute_dtype=compute_dtype,
                break_on_mismatch=break_on,
                collect_severity=collect_sev or (compute_dtype == "float32"),
                dtype_meta=dtype_meta,
            )
            summary["cache_path"] = {
                "artifact": str(cache_out),
                "mode_label": cache_res.get("mode_label"),
                "compute_dtype": cache_res.get("compute_dtype"),
                "token_for_token_equal": cache_res.get("token_for_token_equal"),
                "completed_steps": cache_res.get("completed_steps"),
                "max_abs_logit_diff": cache_res.get("max_abs_logit_diff"),
                "first_mismatch_step": cache_res.get("first_mismatch_step"),
                "status": cache_res.get("status"),
                "severity": cache_res.get("severity"),
                "first_mismatch_dump": cache_res.get("first_mismatch_dump"),
            }
            write_json(out_dir / f"summary_{stamp}.json", summary)
            if args.task in ("cache", "cache_severity"):
                # single-task exit path (finally frees model)
                print(json.dumps(summary["cache_path"], indent=2), flush=True)
                if not cache_res.get("token_for_token_equal") and not run_severity:
                    print("[P3] CACHE-PATH MISMATCH under higher-precision compute", flush=True)
                    summary["gpu_final"] = nvidia_smi()
                    write_json(out_dir / f"summary_{stamp}.json", summary)
                    return 2
                summary["gpu_final"] = nvidia_smi()
                write_json(out_dir / f"summary_{stamp}.json", summary)
                return 0 if cache_res.get("token_for_token_equal") or run_severity else 2
            if not cache_res.get("token_for_token_equal"):
                print("[P3] CACHE-PATH MISMATCH — stop per order", flush=True)
                print(json.dumps(summary["cache_path"], indent=2), flush=True)
                return 2

        if args.task in ("all", "t3"):
            t3_out = out_dir / f"t3_residency_{stamp}.json"
            print(
                f"[P3] T3 residency mode={dtype_meta.get('mode_label')} -> {t3_out}",
                flush=True,
            )
            t3_res = run_t3_residency(
                model,
                stages=STAGES,
                chunk=args.prefill_chunk,
                wall_rail_s=args.wall_rail_s,
                ms_per_token_estimate=args.ms_per_token_estimate,
                seed=args.seed,
                out=t3_out,
                compute_dtype=compute_dtype,
                dtype_meta=dtype_meta,
            )
            summary["t3"] = {
                "artifact": str(t3_out),
                "mode_label": t3_res.get("mode_label"),
                "compute_dtype": t3_res.get("compute_dtype"),
                "t3_full_131k_confirmed": t3_res.get("t3_full_131k_confirmed"),
                "ceiling_S_reached": t3_res.get("ceiling_S_reached"),
                "t3_verdict": t3_res.get("t3_verdict"),
                "rail_stop": t3_res.get("rail_stop"),
                "projection_math": t3_res.get("projection_math"),
                "vram_at_131072": (t3_res.get("extrapolation") or {}).get(
                    "vram_at_131072"
                ),
                "stage_receipts": [
                    {
                        "S": r.get("S"),
                        "status": r.get("status"),
                        "wall_seconds": r.get("wall_seconds"),
                        "projected_wall_s": r.get("projected_wall_s"),
                        "peak_used_mib": r.get("peak_used_mib"),
                        "ms_per_token_measured": r.get("ms_per_token_measured"),
                        "tokens_per_sec": r.get("tokens_per_sec"),
                    }
                    for r in t3_res.get("stage_receipts", [])
                ],
            }
            write_json(out_dir / f"summary_{stamp}.json", summary)

        if args.task in ("all", "t2"):
            t2_out = out_dir / f"t2_apa_{stamp}.json"
            print(
                f"[P3] T2 APA mode={dtype_meta.get('mode_label')} "
                f"S={args.apa_prompt_len} -> {t2_out}",
                flush=True,
            )
            t2_res = run_t2_apa(
                model,
                prompt_len=args.apa_prompt_len,
                gen_steps=args.apa_gen_steps,
                seed=args.seed + 99,
                out=t2_out,
                compute_dtype=compute_dtype,
                dtype_meta=dtype_meta,
                chunk=args.prefill_chunk,
            )
            summary["t2"] = {
                "artifact": str(t2_out),
                "mode_label": t2_res.get("mode_label"),
                "compute_dtype": t2_res.get("compute_dtype"),
                "prompt_len": t2_res.get("prompt_len"),
                "zero_flip": t2_res.get("zero_flip"),
                "n_flips": t2_res.get("n_flips"),
                "flip_details": t2_res.get("flip_details"),
                "ppl_proxy_mean_nll_delta_apa_minus_std": t2_res.get(
                    "ppl_proxy_mean_nll_delta_apa_minus_std"
                ),
                "apa_fired_on_full_layers": t2_res.get("apa_fired_on_full_layers"),
                "engagement": t2_res.get("engagement"),
                "wall_delta_apa_minus_std_s": t2_res.get(
                    "wall_delta_apa_minus_std_s"
                ),
                "status": t2_res.get("status"),
            }
            write_json(out_dir / f"summary_{stamp}.json", summary)
    finally:
        free_model(model)

    summary["gpu_final"] = nvidia_smi()
    write_json(out_dir / f"summary_{stamp}.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
