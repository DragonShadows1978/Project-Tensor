#!/usr/bin/env python3
"""Teacher-forced STANDARD vs APA A/B at S=8192 (T2 prompt).

Order rails:
  - no git commit; harness edits scripts/trinity_nano_* only
  - INT4 weights + fp32 compute (quarantine holds)
  - each GPU arm wall <= 10 min
  - prefill only on IDENTICAL fixed token sequence (no generation)

Captures final logits for the last 4096 positions (streamed via memmap).
Reports argmax flip rate, flipped-position margin histogram, |Δlogit|
stats, and early/late (context-depth) buckets.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
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


def _load_p3():
    """Reuse P3 harness primitives without requiring scripts/ as a package."""
    path = ROOT / "scripts" / "trinity_nano_p3_envelope.py"
    spec = importlib.util.spec_from_file_location("trinity_nano_p3_envelope", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_p3 = _load_p3()
free_model = _p3.free_model
load_model = _p3.load_model
make_ids = _p3.make_ids
nvidia_smi = _p3.nvidia_smi
parse_used_mib = _p3.parse_used_mib
top2_margin = _p3.top2_margin
write_json = _p3.write_json

DEFAULT_MODEL_DIR = "/mnt/ForgeRealm/models/trinity-nano"
DEFAULT_OUT_DIR = "artifacts/trinity_nano/apa_teacher_forced_ab"
# T2 main() used seed=args.seed+99 with default seed=7 → 106.
T2_PROMPT_SEED = 106
T2_PROMPT_LEN = 8192
CAPTURE_LAST_N = 4096
GPU_WALL_RAIL_S = 600.0
PREFILL_CHUNK = 512
# Capture-region chunk: full logits [chunk, V] must fit in VRAM.
CAPTURE_CHUNK = 64


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--output-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--prompt-len", type=int, default=T2_PROMPT_LEN)
    p.add_argument(
        "--seed",
        type=int,
        default=T2_PROMPT_SEED,
        help="Prompt RNG seed. Default 106 matches T2 (main seed 7 + 99).",
    )
    p.add_argument("--capture-last-n", type=int, default=CAPTURE_LAST_N)
    p.add_argument("--prefill-chunk", type=int, default=PREFILL_CHUNK)
    p.add_argument("--capture-chunk", type=int, default=CAPTURE_CHUNK)
    p.add_argument("--wall-rail-s", type=float, default=GPU_WALL_RAIL_S)
    p.add_argument(
        "--compute-dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="Binding mode for this order: float32 (INT4+fp32).",
    )
    p.add_argument(
        "--keep-logit-memmap",
        action="store_true",
        help="Keep full per-arm logit memmaps after stats (large). Default: delete.",
    )
    return p.parse_args()


def _adaptive_step(off: int, chunk: int, remaining: int) -> int:
    """Mirror p3 chunked_prefill score-workspace bound."""
    s_ctx = max(1, off + int(chunk))
    step = min(int(chunk), max(64, int(300 * 1024 * 1024 // (max(1, 8 * s_ctx * 2)))))
    step = max(64, (step // 64) * 64)
    step = min(step, remaining)
    return max(1, step)


def teacher_forced_capture(
    model,
    input_ids: np.ndarray,
    *,
    mode: str,
    capture_start: int,
    capture_n: int,
    prefill_chunk: int,
    capture_chunk: int,
    memmap_path: Path,
    wall_rail_s: float,
    vocab_size: int,
) -> dict[str, Any]:
    """Prefill fixed sequence; stream logits for positions [capture_start, L).

    Positions are absolute sequence indices. Logits row j in the memmap is
    the final-layer vocabulary logits at absolute position capture_start + j
    (shape [capture_n, V]).
    """
    engage = model.set_attention_mode(
        mode,
        refine_percentile=0.15,
        bulk_bits=4,
        apa_min_context=2048,
        full_only=True,
    )
    input_ids = np.asarray(input_ids, dtype=np.int64)
    B, L = input_ids.shape
    assert B == 1
    if capture_start < 0 or capture_start + capture_n > L:
        raise ValueError(
            f"capture window [{capture_start}, {capture_start + capture_n}) "
            f"out of range for L={L}"
        )

    memmap_path.parent.mkdir(parents=True, exist_ok=True)
    # float32 rows; write-through so APA pass can reopen after free.
    mm = np.lib.format.open_memmap(
        str(memmap_path),
        mode="w+",
        dtype=np.float32,
        shape=(int(capture_n), int(vocab_size)),
    )

    caches = None
    peak_mib = parse_used_mib(nvidia_smi()) or 0
    off = 0
    n_chunks = 0
    wall_abort = False
    t0 = time.perf_counter()

    with tc.no_grad():
        # Phase 1: prefix before capture — last_token_only (no full V rows).
        while off < capture_start:
            remaining = capture_start - off
            step = _adaptive_step(off, prefill_chunk, remaining)
            # Prefer full prefill_chunk when score bound allows.
            step = min(step, remaining, int(prefill_chunk))
            if step < 1:
                step = remaining
            seg = input_ids[:, off : off + step]
            _logits, caches = model(
                seg,
                kv_caches=caches,
                position_offset=off,
                last_token_only=True,
            )
            tc.synchronize()
            del _logits
            off += step
            n_chunks += 1
            used = parse_used_mib(nvidia_smi())
            if used is not None:
                peak_mib = max(peak_mib, used)
            elapsed = time.perf_counter() - t0
            if elapsed > wall_rail_s:
                wall_abort = True
                break

        # Phase 2: capture window — full per-position logits, streamed to memmap.
        while off < L and not wall_abort:
            remaining = L - off
            step = min(int(capture_chunk), remaining)
            # Score workspace can force smaller steps at long S.
            step = min(step, _adaptive_step(off, capture_chunk, remaining))
            if step < 1:
                step = remaining
            seg = input_ids[:, off : off + step]
            logits, caches = model(
                seg,
                kv_caches=caches,
                position_offset=off,
                last_token_only=False,
            )
            tc.synchronize()
            # logits: [1, step, V]
            arr = logits.float().numpy()[0].astype(np.float32, copy=False)
            # Absolute positions off .. off+step-1 map into memmap rows.
            for i in range(step):
                abs_pos = off + i
                if abs_pos < capture_start:
                    continue
                if abs_pos >= capture_start + capture_n:
                    break
                row = abs_pos - capture_start
                mm[row, :] = arr[i]
            del logits, arr
            off += step
            n_chunks += 1
            used = parse_used_mib(nvidia_smi())
            if used is not None:
                peak_mib = max(peak_mib, used)
            elapsed = time.perf_counter() - t0
            if elapsed > wall_rail_s:
                wall_abort = True
                break
            if n_chunks % 8 == 0 and hasattr(tc, "empty_cache"):
                tc.empty_cache()

        del caches
        if hasattr(tc, "empty_cache"):
            tc.empty_cache()

    mm.flush()
    wall = time.perf_counter() - t0

    # Per-layer backend map after this arm's prefill.
    backends: dict[str, int] = {}
    per_layer: list[dict[str, Any]] = []
    for layer in model.layers:
        att = layer.self_attn
        b = str(att.last_attention_backend)
        backends[b] = backends.get(b, 0) + 1
        per_layer.append(
            {
                "layer_idx": int(layer.layer_idx),
                "is_sliding": bool(att.is_local_attention),
                "attention_mode": str(att.attention_mode),
                "last_attention_backend": b,
                "apa_min_context": int(att.apa_min_context),
            }
        )

    return {
        "mode": mode,
        "engage": engage,
        "wall_seconds": float(wall),
        "wall_abort": bool(wall_abort),
        "positions_filled": int(off - capture_start) if off >= capture_start else 0,
        "offset_reached": int(off),
        "n_chunks": int(n_chunks),
        "peak_used_mib": int(peak_mib),
        "memmap_path": str(memmap_path),
        "backends_after_prefill": backends,
        "per_layer_backends_after_prefill": per_layer,
        "gpu_after": nvidia_smi(),
    }


def _margin_bucket(m: float) -> str:
    if m < 0.05:
        return "<0.05"
    if m < 0.25:
        return "0.05-0.25"
    if m < 1.0:
        return "0.25-1.0"
    return ">1.0"


def compare_arms(
    std_mm: np.memmap,
    apa_mm: np.memmap,
    *,
    capture_start: int,
    capture_n: int,
    prompt_len: int,
) -> dict[str, Any]:
    """Streaming per-position compare. Margin = STANDARD top1−top2."""
    n = int(capture_n)
    assert std_mm.shape[0] >= n and apa_mm.shape[0] >= n

    flip_mask = np.zeros(n, dtype=np.bool_)
    std_argmax = np.empty(n, dtype=np.int64)
    apa_argmax = np.empty(n, dtype=np.int64)
    std_margins = np.empty(n, dtype=np.float64)
    max_abs_dlogit = np.empty(n, dtype=np.float64)
    # median |Δlogit| per position (over vocab) — for optional depth signal
    median_abs_dlogit = np.empty(n, dtype=np.float64)

    # Process in row batches to keep host RAM bounded.
    batch = 32
    for start in range(0, n, batch):
        end = min(n, start + batch)
        s = np.asarray(std_mm[start:end], dtype=np.float32)
        a = np.asarray(apa_mm[start:end], dtype=np.float32)
        diff = np.abs(s - a)
        max_abs_dlogit[start:end] = diff.max(axis=1)
        median_abs_dlogit[start:end] = np.median(diff, axis=1)
        for i in range(end - start):
            row_s = s[i]
            row_a = a[i]
            t2s = top2_margin(row_s)
            t2a = top2_margin(row_a)
            si = int(t2s["top1_id"])
            ai = int(t2a["top1_id"])
            std_argmax[start + i] = si
            apa_argmax[start + i] = ai
            std_margins[start + i] = float(t2s["margin_top1_minus_top2"])
            flip_mask[start + i] = si != ai

    n_flips = int(flip_mask.sum())
    flip_rate = float(n_flips / n) if n else 0.0

    # Margin histogram on STANDARD arm at flipped positions only.
    buckets = {"<0.05": 0, "0.05-0.25": 0, "0.25-1.0": 0, ">1.0": 0}
    flip_margins = std_margins[flip_mask]
    for m in flip_margins:
        buckets[_margin_bucket(float(m))] += 1

    # Global |Δlogit| (per-position max abs across vocab).
    dlogit_stats = {
        "max_of_pos_max_abs": float(max_abs_dlogit.max()) if n else None,
        "median_of_pos_max_abs": float(np.median(max_abs_dlogit)) if n else None,
        "mean_of_pos_max_abs": float(max_abs_dlogit.mean()) if n else None,
        "p95_of_pos_max_abs": float(np.percentile(max_abs_dlogit, 95)) if n else None,
        "max_of_pos_median_abs": float(median_abs_dlogit.max()) if n else None,
        "median_of_pos_median_abs": float(np.median(median_abs_dlogit)) if n else None,
    }

    def _bucket_stats(idx: np.ndarray) -> dict[str, Any]:
        if idx.size == 0:
            return {
                "n_pos": 0,
                "n_flips": 0,
                "flip_rate": None,
                "margin_hist_flipped": dict(buckets),  # empty-ish
                "dlogit": None,
            }
        fm = flip_mask[idx]
        n_f = int(fm.sum())
        b = {"<0.05": 0, "0.05-0.25": 0, "0.25-1.0": 0, ">1.0": 0}
        for m in std_margins[idx][fm]:
            b[_margin_bucket(float(m))] += 1
        mad = max_abs_dlogit[idx]
        return {
            "n_pos": int(idx.size),
            "n_flips": n_f,
            "flip_rate": float(n_f / idx.size),
            "margin_hist_flipped": b,
            "dlogit": {
                "max_of_pos_max_abs": float(mad.max()),
                "median_of_pos_max_abs": float(np.median(mad)),
                "mean_of_pos_max_abs": float(mad.mean()),
                "p95_of_pos_max_abs": float(np.percentile(mad, 95)),
            },
            "std_margin_all_median": float(np.median(std_margins[idx])),
            "std_margin_flipped_median": (
                float(np.median(std_margins[idx][fm])) if n_f else None
            ),
        }

    # Early vs late in capture window = shallower vs deeper context depth.
    # absolute pos p has causal context depth p+1.
    half = n // 2
    early_idx = np.arange(0, half)
    late_idx = np.arange(half, n)
    early_abs = [capture_start, capture_start + half - 1]
    late_abs = [capture_start + half, capture_start + n - 1]

    # Wide-margin flip population (signature test).
    wide = int(buckets["0.25-1.0"] + buckets[">1.0"])
    wide_rate = float(wide / n) if n else 0.0
    near_tie = int(buckets["<0.05"])
    mid = int(buckets["0.05-0.25"])

    # Sample a few wide-margin flips for the receipt (if any).
    sample_wide: list[dict[str, Any]] = []
    if n_flips:
        flip_pos = np.flatnonzero(flip_mask)
        order = flip_pos[np.argsort(-std_margins[flip_pos])]
        for j in order[:12]:
            sample_wide.append(
                {
                    "capture_row": int(j),
                    "abs_pos": int(capture_start + j),
                    "context_depth": int(capture_start + j + 1),
                    "std_argmax": int(std_argmax[j]),
                    "apa_argmax": int(apa_argmax[j]),
                    "std_margin": float(std_margins[j]),
                    "pos_max_abs_dlogit": float(max_abs_dlogit[j]),
                    "margin_bucket": _margin_bucket(float(std_margins[j])),
                }
            )

    return {
        "n_positions": n,
        "capture_start": int(capture_start),
        "capture_end_exclusive": int(capture_start + n),
        "prompt_len": int(prompt_len),
        "n_flips": n_flips,
        "flip_rate": flip_rate,
        "margin_hist_flipped_std_top1_minus_top2": buckets,
        "wide_margin_flips_gt_0_25": wide,
        "wide_margin_flip_rate": wide_rate,
        "near_tie_flips_lt_0_05": near_tie,
        "mid_margin_flips_0_05_0_25": mid,
        "dlogit_stats_all": dlogit_stats,
        "by_context_depth": {
            "early": {
                "label": "shallower half of capture window",
                "abs_pos_range": early_abs,
                "context_depth_range": [early_abs[0] + 1, early_abs[1] + 1],
                **_bucket_stats(early_idx),
            },
            "late": {
                "label": "deeper half of capture window",
                "abs_pos_range": late_abs,
                "context_depth_range": [late_abs[0] + 1, late_abs[1] + 1],
                **_bucket_stats(late_idx),
            },
        },
        "sample_flips_by_descending_std_margin": sample_wide,
        # Compact arrays for downstream (not full logits).
        "arrays": {
            "flip_mask": flip_mask,
            "std_argmax": std_argmax,
            "apa_argmax": apa_argmax,
            "std_margins": std_margins,
            "max_abs_dlogit": max_abs_dlogit,
            "median_abs_dlogit": median_abs_dlogit,
        },
    }


def engagement_ok(per_layer: list[dict[str, Any]], n_full_expected: int = 14) -> dict[str, Any]:
    full_apa = [
        r
        for r in per_layer
        if (not r["is_sliding"]) and "apa_selective" in str(r["last_attention_backend"])
    ]
    full_non = [
        r
        for r in per_layer
        if (not r["is_sliding"]) and "apa_selective" not in str(r["last_attention_backend"])
    ]
    slide_leak = [
        r
        for r in per_layer
        if r["is_sliding"] and "apa_selective" in str(r["last_attention_backend"])
    ]
    return {
        "n_full_apa": len(full_apa),
        "n_full_non_apa": len(full_non),
        "n_sliding_apa_leak": len(slide_leak),
        "full_apa_indices": [r["layer_idx"] for r in full_apa],
        "apa_fired_exactly_14_full_not_sliding": (
            len(full_apa) == n_full_expected
            and len(full_non) == 0
            and len(slide_leak) == 0
        ),
    }


def honest_read(cmp: dict[str, Any]) -> dict[str, Any]:
    """Registered classification before looking at numbers conceptually.

    Pre-registered:
      - near-tie-only: essentially all flips have STANDARD margin < 0.05
        (and wide-margin flip rate ≈ 0) → ordinary sensitivity
      - signature: nontrivial wide-margin (>0.25) flip population
    """
    n = int(cmp["n_positions"])
    n_flips = int(cmp["n_flips"])
    buckets = cmp["margin_hist_flipped_std_top1_minus_top2"]
    wide = int(cmp["wide_margin_flips_gt_0_25"])
    near = int(cmp["near_tie_flips_lt_0_05"])
    mid = int(cmp["mid_margin_flips_0_05_0_25"])
    wide_rate = float(cmp["wide_margin_flip_rate"])

    if n_flips == 0:
        verdict = "NO_FLIPS"
        note = "Zero argmax flips over capture window; no perturbation signature in argmax."
    elif wide > 0 and wide_rate >= (1.0 / n):  # any wide flip at nontrivial absolute count
        # "nontrivial rate": at least one wide flip is a finding population if
        # rate among flips is material, or absolute wide rate > 0.
        wide_among_flips = wide / n_flips
        if wide_among_flips >= 0.05 or wide >= 5:
            verdict = "SIGNATURE_WIDE_MARGIN_FLIPS"
            note = (
                f"Wide-margin (>0.25) flip population present: {wide}/{n_flips} flips "
                f"({wide_among_flips:.3%} of flips; {wide_rate:.5%} of positions). "
                "Not ordinary near-tie sensitivity alone."
            )
        else:
            verdict = "MOSTLY_NEAR_TIE_WITH_SPARSE_WIDE"
            note = (
                f"Dominant near/mid ties but sparse wide flips: wide={wide}, "
                f"near={near}, mid={mid} of {n_flips}."
            )
    elif near + mid == n_flips and wide == 0:
        if near >= 0.8 * n_flips:
            verdict = "ORDINARY_NEAR_TIE_SENSITIVITY"
            note = (
                f"All {n_flips} flips at margin <0.25; "
                f"{near}/{n_flips} at <0.05. Ordinary near-tie sensitivity."
            )
        else:
            verdict = "ORDINARY_NEAR_TIE_SENSITIVITY"
            note = (
                f"No wide-margin (>0.25) flips. mid={mid}, near={near} of {n_flips}. "
                "Ordinary sensitivity (no wide signature)."
            )
    else:
        verdict = "MIXED"
        note = f"buckets near={near} mid={mid} wide={wide} of {n_flips}."

    early = cmp["by_context_depth"]["early"]
    late = cmp["by_context_depth"]["late"]
    depth_note = (
        f"early flip_rate={early.get('flip_rate')} "
        f"med_max|Δ|={ (early.get('dlogit') or {}).get('median_of_pos_max_abs') }; "
        f"late flip_rate={late.get('flip_rate')} "
        f"med_max|Δ|={ (late.get('dlogit') or {}).get('median_of_pos_max_abs') }"
    )
    return {
        "verdict": verdict,
        "note": note,
        "depth_note": depth_note,
        "evidence_class": "e2e teacher-forced prefill A/B, INT4+fp32, fixed token sequence",
    }


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    prompt_len = int(args.prompt_len)
    capture_n = int(args.capture_last_n)
    if capture_n > prompt_len:
        raise SystemExit(f"capture_last_n={capture_n} > prompt_len={prompt_len}")
    capture_start = prompt_len - capture_n

    ids = make_ids(prompt_len, int(args.seed))
    ids_sha = hashlib.sha256(np.ascontiguousarray(ids).tobytes()).hexdigest()[:16]
    np.save(run_dir / "prompt_ids.npy", ids)

    receipt: dict[str, Any] = {
        "schema": "trinity_nano_apa_teacher_forced_ab_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "prompt_len": prompt_len,
        "seed": int(args.seed),
        "seed_note": "Default 106 matches T2 (p3 main seed 7 + 99)",
        "capture_last_n": capture_n,
        "capture_start": capture_start,
        "weight_mode": "int4",
        "compute_dtype": str(args.compute_dtype),
        "mode_label": (
            "fp32_compute_int4_dequant"
            if args.compute_dtype == "float32"
            else "bf16_compute_int4_dequant"
        ),
        "wall_rail_s": float(args.wall_rail_s),
        "prefill_chunk": int(args.prefill_chunk),
        "capture_chunk": int(args.capture_chunk),
        "prompt_ids_sha256_16": ids_sha,
        "prompt_ids_path": str(run_dir / "prompt_ids.npy"),
        "gpu_before": nvidia_smi(),
        "status": "running",
    }
    write_json(run_dir / "receipt.json", receipt)

    print(
        f"[TF-AB] load model compute={args.compute_dtype} "
        f"S={prompt_len} capture_last={capture_n} seed={args.seed}",
        flush=True,
    )
    model, info, load_s, dtype_meta = load_model(
        args.model_dir, compute_dtype=args.compute_dtype
    )
    receipt["load_wall_seconds"] = float(load_s)
    receipt["dtype_meta"] = dtype_meta
    receipt["loader_info"] = {
        k: info[k]
        for k in (
            "weight_mode",
            "layers",
            "quantized_linear_bytes",
            "full_attention_layers",
            "num_key_value_heads",
            "vocab_size" if "vocab_size" in info else "hidden_size",
        )
        if k in info
    }
    vocab_size = int(getattr(model.config, "vocab_size", 200192))
    receipt["vocab_size"] = vocab_size
    receipt["gpu_after_load"] = nvidia_smi()
    write_json(run_dir / "receipt.json", receipt)

    std_mm_path = run_dir / "logits_standard_f32.npy"
    apa_mm_path = run_dir / "logits_apa_f32.npy"

    try:
        print("[TF-AB] arm STANDARD ...", flush=True)
        t_std0 = time.perf_counter()
        std = teacher_forced_capture(
            model,
            ids,
            mode="standard",
            capture_start=capture_start,
            capture_n=capture_n,
            prefill_chunk=int(args.prefill_chunk),
            capture_chunk=int(args.capture_chunk),
            memmap_path=std_mm_path,
            wall_rail_s=float(args.wall_rail_s),
            vocab_size=vocab_size,
        )
        std["wall_seconds_outer"] = float(time.perf_counter() - t_std0)
        receipt["standard"] = {
            k: std[k]
            for k in (
                "mode",
                "engage",
                "wall_seconds",
                "wall_abort",
                "positions_filled",
                "offset_reached",
                "n_chunks",
                "peak_used_mib",
                "memmap_path",
                "backends_after_prefill",
                "per_layer_backends_after_prefill",
                "gpu_after",
                "wall_seconds_outer",
            )
        }
        write_json(run_dir / "receipt.json", receipt)
        print(
            f"[TF-AB] STANDARD done wall={std['wall_seconds']:.1f}s "
            f"abort={std['wall_abort']} peak={std['peak_used_mib']} MiB",
            flush=True,
        )
        if std["wall_abort"] or std["positions_filled"] < capture_n:
            receipt["status"] = "standard_wall_or_incomplete"
            write_json(run_dir / "receipt.json", receipt)
            print("[TF-AB] STANDARD incomplete — stop", flush=True)
            return 2

        print("[TF-AB] arm APA ...", flush=True)
        t_apa0 = time.perf_counter()
        apa = teacher_forced_capture(
            model,
            ids,
            mode="apa_selective",
            capture_start=capture_start,
            capture_n=capture_n,
            prefill_chunk=int(args.prefill_chunk),
            capture_chunk=int(args.capture_chunk),
            memmap_path=apa_mm_path,
            wall_rail_s=float(args.wall_rail_s),
            vocab_size=vocab_size,
        )
        apa["wall_seconds_outer"] = float(time.perf_counter() - t_apa0)
        receipt["apa"] = {
            k: apa[k]
            for k in (
                "mode",
                "engage",
                "wall_seconds",
                "wall_abort",
                "positions_filled",
                "offset_reached",
                "n_chunks",
                "peak_used_mib",
                "memmap_path",
                "backends_after_prefill",
                "per_layer_backends_after_prefill",
                "gpu_after",
                "wall_seconds_outer",
            )
        }
        eng = engagement_ok(apa["per_layer_backends_after_prefill"])
        receipt["engagement"] = eng
        write_json(run_dir / "receipt.json", receipt)
        print(
            f"[TF-AB] APA done wall={apa['wall_seconds']:.1f}s "
            f"abort={apa['wall_abort']} eng_ok={eng['apa_fired_exactly_14_full_not_sliding']}",
            flush=True,
        )
        if apa["wall_abort"] or apa["positions_filled"] < capture_n:
            receipt["status"] = "apa_wall_or_incomplete"
            write_json(run_dir / "receipt.json", receipt)
            return 2
    finally:
        free_model(model)

    # Compare on CPU from memmaps (GPU free).
    print("[TF-AB] compare memmaps ...", flush=True)
    t_cmp0 = time.perf_counter()
    std_mm = np.load(std_mm_path, mmap_mode="r")
    apa_mm = np.load(apa_mm_path, mmap_mode="r")
    cmp = compare_arms(
        std_mm,
        apa_mm,
        capture_start=capture_start,
        capture_n=capture_n,
        prompt_len=prompt_len,
    )
    arrays = cmp.pop("arrays")
    np.savez_compressed(
        run_dir / "per_position_summary.npz",
        flip_mask=arrays["flip_mask"],
        std_argmax=arrays["std_argmax"],
        apa_argmax=arrays["apa_argmax"],
        std_margins=arrays["std_margins"],
        max_abs_dlogit=arrays["max_abs_dlogit"],
        median_abs_dlogit=arrays["median_abs_dlogit"],
        capture_start=np.int64(capture_start),
        prompt_len=np.int64(prompt_len),
    )
    read = honest_read(cmp)
    cmp_wall = time.perf_counter() - t_cmp0
    del std_mm, apa_mm, arrays

    if not args.keep_logit_memmap:
        # Full logits ~3 GB each; summary npz retains the load-bearing series.
        for p in (std_mm_path, apa_mm_path):
            try:
                p.unlink()
            except OSError:
                pass
        logit_retention = "deleted_after_compare"
    else:
        logit_retention = "kept"

    receipt.update(
        {
            "status": "ok",
            "compare_wall_seconds": float(cmp_wall),
            "compare": cmp,
            "honest_read": read,
            "logit_memmap_retention": logit_retention,
            "artifacts": {
                "run_dir": str(run_dir),
                "receipt": str(run_dir / "receipt.json"),
                "per_position_summary": str(run_dir / "per_position_summary.npz"),
                "prompt_ids": str(run_dir / "prompt_ids.npy"),
            },
            "gpu_final": nvidia_smi(),
            "total_arm_wall_s": {
                "standard": receipt.get("standard", {}).get("wall_seconds"),
                "apa": receipt.get("apa", {}).get("wall_seconds"),
            },
        }
    )
    write_json(run_dir / "receipt.json", receipt)

    # Also write a dense tables-only summary for the operator.
    tables = {
        "1_argmax_flip": {
            "n_positions": cmp["n_positions"],
            "n_flips": cmp["n_flips"],
            "flip_rate": cmp["flip_rate"],
        },
        "2_margin_hist_flipped": cmp["margin_hist_flipped_std_top1_minus_top2"],
        "3_dlogit": cmp["dlogit_stats_all"],
        "4_by_context_depth": cmp["by_context_depth"],
        "honest_read": read,
        "walls": receipt["total_arm_wall_s"],
        "engagement": receipt.get("engagement"),
    }
    write_json(run_dir / "tables.json", tables)
    print(json.dumps(tables, indent=2), flush=True)
    print(f"[TF-AB] artifacts: {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
