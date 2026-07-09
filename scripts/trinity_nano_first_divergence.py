#!/usr/bin/env python3
"""HF-vs-TC first-divergence probe for Trinity Nano (probe 0 prefill).

Compares per-layer hidden states and, at the first MoE layer (2), MoE
router/expert intermediate tensors. HF runs on CPU; TC layer-streams on GPU.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

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
    _sigmoid_topk_route,
)

DEFAULT_MODEL_DIR = "/mnt/ForgeRealm/models/trinity-nano"
DEFAULT_REF_DIR = "artifacts/trinity_nano/reference_capture"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--reference-dir", default=DEFAULT_REF_DIR)
    p.add_argument("--output", default=None)
    p.add_argument("--max-layers", type=int, default=6)
    p.add_argument("--focus-layer", type=int, default=2)
    return p.parse_args()


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def mean_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def install_shims() -> dict[str, Any]:
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    from transformers.modeling_utils import PreTrainedModel

    receipt: dict[str, Any] = {}
    if "default" not in ROPE_INIT_FUNCTIONS:
        def _default_rope_init(config, device=None, seq_len=None, **kwargs):
            dim = getattr(config, "head_dim", None)
            if dim is None:
                dim = config.hidden_size // config.num_attention_heads
            base = getattr(config, "rope_theta", 10000.0)
            inv_freq = 1.0 / (
                base
                ** (
                    torch.arange(0, dim, 2, dtype=torch.float32, device=device)
                    / dim
                )
            )
            return inv_freq, 1.0

        ROPE_INIT_FUNCTIONS["default"] = _default_rope_init
        receipt["rope_default_shim"] = True
    else:
        receipt["rope_default_shim"] = False

    if not getattr(PreTrainedModel, "_trinity_nano_missing_key_shim", False):
        PreTrainedModel._initialize_missing_keys = lambda *a, **k: None  # type: ignore
        PreTrainedModel._trinity_nano_missing_key_shim = True  # type: ignore
        receipt["missing_key_shim"] = True
    else:
        receipt["missing_key_shim"] = False
    return receipt


def install_remote_masking_shim(model) -> str:
    import sys
    from transformers.masking_utils import (
        create_causal_mask as hf_create_causal_mask,
        create_sliding_window_causal_mask as hf_create_sliding_window_causal_mask,
    )

    remote_module = sys.modules[model.__class__.__module__]

    def adapt(fn):
        def wrapper(**kwargs):
            if "input_embeds" in kwargs and "inputs_embeds" not in kwargs:
                kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
            cache_position = kwargs.pop("cache_position", None)
            if cache_position is not None and kwargs.get("position_ids") is None:
                kwargs["position_ids"] = cache_position.unsqueeze(0)
            return fn(**kwargs)

        return wrapper

    remote_module.create_causal_mask = adapt(hf_create_causal_mask)
    remote_module.create_sliding_window_causal_mask = adapt(
        hf_create_sliding_window_causal_mask
    )
    return "mask_shim_installed"


def load_hf(model_dir: Path):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    install_shims()
    tok = AutoTokenizer.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True
    )
    config = AutoConfig.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True
    )
    config.bos_token_id = tok.bos_token_id
    config.eos_token_id = tok.eos_token_id
    config.pad_token_id = tok.pad_token_id
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        config=config,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        attn_implementation="eager",
    )
    model.eval()
    install_remote_masking_shim(model)
    return model, tok, config


def run_hf_capture(model, input_ids: np.ndarray, max_layers: int, focus_layer: int):
    """Partial forward through first max_layers only (CPU, cheap)."""
    from transformers.masking_utils import (
        create_causal_mask,
        create_sliding_window_causal_mask,
    )

    ids = torch.tensor(input_ids.reshape(1, -1), dtype=torch.long)
    captures: dict[str, Any] = {"layers": {}}
    moe_caps: dict[str, Any] = {}

    # instrument focus-layer MoE
    layer = model.model.layers[focus_layer]
    moe = layer.mlp
    orig_router = moe.router.forward
    orig_moe = moe.forward
    orig_shared = moe.shared_experts.forward if moe.shared_experts is not None else None

    def router_wrap(hidden_states, expert_bias=None):
        top_scores, selected = orig_router(hidden_states, expert_bias)
        hs = hidden_states.view(-1, hidden_states.shape[-1])
        scores = moe.router.gate(hs)
        scores_sig = torch.sigmoid(scores.float())
        moe_caps["router_logits"] = scores.detach().float().cpu().numpy()
        moe_caps["scores_sigmoid"] = scores_sig.detach().cpu().numpy()
        if expert_bias is not None:
            moe_caps["expert_bias"] = expert_bias.detach().float().cpu().numpy()
            moe_caps["scores_plus_bias"] = (
                (scores_sig + expert_bias.float()).detach().cpu().numpy()
            )
        moe_caps["selected_experts"] = selected.detach().cpu().numpy().astype(np.int64)
        moe_caps["top_scores"] = top_scores.detach().float().cpu().numpy()
        if expert_bias is not None:
            _, sel2 = torch.topk(scores_sig + expert_bias.float(), k=moe.router.top_k, dim=1)
            gathered = scores_sig.gather(1, sel2)
        else:
            gathered, sel2 = torch.topk(scores_sig, k=moe.router.top_k, dim=1)
        moe_caps["gathered_pre_norm"] = gathered.detach().cpu().numpy()
        if moe.router.score_func == "sigmoid" and moe.router.route_norm:
            denom = gathered.sum(dim=-1, keepdim=True) + 1e-20
            normed = gathered / denom
        else:
            normed = gathered
        moe_caps["top_scores_post_norm"] = normed.detach().cpu().numpy()
        moe_caps["top_scores_post_scale"] = (
            (normed * moe.router.route_scale).detach().cpu().numpy()
        )
        return top_scores, selected

    def shared_wrap(x):
        y = orig_shared(x)
        # shared may be flat or B,L,H depending on call site
        moe_caps["shared_out"] = y.detach().float().cpu().numpy()
        return y

    def moe_wrap(hidden_states):
        moe_caps["mlp_in"] = hidden_states.detach().float().cpu().numpy()
        out = orig_moe(hidden_states)
        moe_caps["mlp_out"] = out.detach().float().cpu().numpy()
        return out

    moe.router.forward = router_wrap
    moe.forward = moe_wrap
    if orig_shared is not None:
        moe.shared_experts.forward = shared_wrap

    with torch.inference_mode():
        embeds = model.model.embed_tokens(ids)
        captures["embed"] = embeds.float().cpu().numpy().astype(np.float32)
        hidden_states = embeds
        if model.config.mup_enabled:
            hidden_states = hidden_states * (model.config.hidden_size**0.5)
        captures["embed_mup"] = hidden_states.float().cpu().numpy().astype(np.float32)

        cache_position = torch.arange(0, ids.shape[1], device=ids.device)
        position_ids = cache_position.unsqueeze(0)
        mask_kwargs = {
            "config": model.config,
            "inputs_embeds": hidden_states,
            "attention_mask": None,
            "cache_position": cache_position,
            "past_key_values": None,
        }
        # prefer remote-shimmed functions if present
        remote = sys.modules[model.__class__.__module__]
        ccm = getattr(remote, "create_causal_mask", create_causal_mask)
        csw = getattr(remote, "create_sliding_window_causal_mask", create_sliding_window_causal_mask)
        try:
            causal_mask_mapping = {
                "full_attention": ccm(**{**mask_kwargs, "input_embeds": hidden_states}),
                "sliding_attention": csw(**{**mask_kwargs, "input_embeds": hidden_states}),
            }
        except TypeError:
            causal_mask_mapping = {
                "full_attention": ccm(**mask_kwargs),
                "sliding_attention": csw(**mask_kwargs),
            }
        position_embeddings = model.model.rotary_emb(hidden_states, position_ids)

        for i in range(max_layers):
            decoder_layer = model.model.layers[i]
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_value=None,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            captures["layers"][i] = {
                "out": hidden_states.detach().float().cpu().numpy().astype(np.float32)
            }

    # restore
    moe.router.forward = orig_router
    moe.forward = orig_moe
    if orig_shared is not None:
        moe.shared_experts.forward = orig_shared

    captures["moe"] = moe_caps
    return captures


def run_tc_capture(
    cfg: TrinityNanoConfig,
    source: SafeTensorSource,
    input_ids: np.ndarray,
    max_layers: int,
    focus_layer: int,
):
    BlockTC.COMPUTE_DTYPE = "bfloat16"
    LinearTC.DTYPE = "bfloat16"
    RMSNormTC.USE_FUSED = False
    QuantLinearTC.FUSED_DECODE = True

    embed = HostEmbedding()
    emb = source.get_np("model.embed_tokens.weight")
    embed.weight = np.ascontiguousarray(emb.astype(np.float32, copy=False))
    rope = RoPECache(cfg)
    L = int(input_ids.shape[-1])
    rope.extend(L)

    captures: dict[str, Any] = {"layers": {}}
    h = embed(input_ids.reshape(1, -1))
    captures["embed"] = h.float().numpy().astype(np.float32)
    if cfg.mup_enabled:
        h = h * (float(cfg.hidden_size) ** 0.5)
    captures["embed_mup"] = h.float().numpy().astype(np.float32)

    for layer_idx in range(max_layers):
        block = TrinityBlockTC.from_safetensors(cfg, source, layer_idx, weight_mode="bf16")
        if layer_idx == focus_layer and block.moe_enabled:
            # instrument MoE internals
            moe = block.mlp
            x = h
            residual = x
            attn_in = _cast(block.input_layernorm(x))
            attn, kv = block.self_attn(attn_in, rope.cos, rope.sin, 0, None)
            h_post_attn = residual + _cast(block.post_attention_layernorm(attn))
            residual = h_post_attn
            mlp_in = _cast(block.pre_mlp_layernorm(h_post_attn))
            moe_caps: dict[str, Any] = {
                "mlp_in": mlp_in.float().numpy().astype(np.float32),
                "pre_mlp_residual": h_post_attn.float().numpy().astype(np.float32),
            }
            B, T, H = mlp_in.shape
            x_flat = mlp_in.reshape([B * T, H])
            scores = moe.router_gate(x_flat)
            moe_caps["router_logits"] = scores.float().numpy().astype(np.float32)
            scores_sig = scores.float().sigmoid()
            moe_caps["scores_sigmoid"] = scores_sig.float().numpy().astype(np.float32)
            bias = moe.expert_bias
            moe_caps["expert_bias"] = bias.float().numpy().astype(np.float32)
            moe_caps["scores_plus_bias"] = (
                (scores_sig + bias.astype(scores_sig.dtype)).float().numpy().astype(np.float32)
            )
            topw, topi = _sigmoid_topk_route(
                scores,
                bias,
                cfg.num_experts_per_tok,
                cfg.route_norm,
                cfg.route_scale,
            )
            # manual intermediates
            selected = topi.numpy().astype(np.int64)
            gathered = scores_sig.gather(1, topi)
            moe_caps["selected_experts"] = selected
            moe_caps["gathered_pre_norm"] = gathered.float().numpy().astype(np.float32)
            denom = gathered.sum([-1], True) + 1e-20
            normed = gathered / denom
            moe_caps["top_scores_post_norm"] = normed.float().numpy().astype(np.float32)
            scaled = normed * float(cfg.route_scale)
            moe_caps["top_scores_post_scale"] = scaled.float().numpy().astype(np.float32)
            moe_caps["top_scores"] = topw.float().numpy().astype(np.float32)

            shared = moe.shared_experts(mlp_in) if moe.shared_experts is not None else None
            if shared is not None:
                moe_caps["shared_out"] = shared.float().numpy().astype(np.float32)

            topw_np = topw.float().numpy().astype(np.float32)
            topi_np = selected
            routed_rows = []
            for row in range(B * T):
                xt = x_flat.slice(0, row, 1)
                acc = None
                slot_order = np.argsort(topi_np[row], kind="stable")
                for slot in slot_order.tolist():
                    expert_id = int(topi_np[row, slot])
                    weight = float(topw_np[row, slot])
                    expert_out = moe.experts[expert_id](xt.reshape([1, 1, H])).reshape([1, H])
                    expert_out = _cast(expert_out.float() * weight)
                    acc = expert_out if acc is None else acc + expert_out
                routed_rows.append(acc)
            routed = tc.cat(routed_rows, dim=0).reshape([B, T, H])
            moe_caps["routed_out"] = routed.float().numpy().astype(np.float32)
            mlp_out = routed if shared is None else shared + routed
            moe_caps["mlp_out"] = mlp_out.float().numpy().astype(np.float32)
            h = residual + _cast(block.post_mlp_layernorm(mlp_out))
            captures["moe"] = moe_caps
            captures["layers"][layer_idx] = {
                "out": h.float().numpy().astype(np.float32)
            }
            del block
            gc.collect()
            if hasattr(tc, "empty_cache"):
                tc.empty_cache()
            continue

        h, _kv, _route = block(h, rope.cos, rope.sin, 0, None)
        captures["layers"][layer_idx] = {
            "out": h.float().numpy().astype(np.float32)
        }
        del block
        gc.collect()
        if hasattr(tc, "empty_cache"):
            tc.empty_cache()

    return captures


def compare_arrays(name: str, a: np.ndarray | None, b: np.ndarray | None) -> dict[str, Any]:
    if a is None or b is None:
        return {"name": name, "status": "missing", "a_is_none": a is None, "b_is_none": b is None}
    if a.shape != b.shape:
        return {
            "name": name,
            "status": "shape_mismatch",
            "a_shape": list(a.shape),
            "b_shape": list(b.shape),
        }
    return {
        "name": name,
        "status": "ok",
        "shape": list(a.shape),
        "max_abs_diff": max_abs(a, b),
        "mean_abs_diff": mean_abs(a, b),
        "a_max": float(np.max(np.abs(a))),
        "b_max": float(np.max(np.abs(b))),
    }


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    ref_dir = Path(args.reference_dir).expanduser().resolve()
    out = (
        Path(args.output).expanduser().resolve()
        if args.output
        else ROOT
        / "artifacts"
        / "trinity_nano"
        / "tc_parity"
        / f"first_divergence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    ref = np.load(ref_dir / "probe_00_plain_short.npz")
    input_ids = ref["input_ids"].astype(np.int64)
    payload: dict[str, Any] = {
        "schema": "trinity_nano_first_divergence_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_dir": str(model_dir),
        "input_ids": [int(x) for x in input_ids.tolist()],
        "max_layers": int(args.max_layers),
        "focus_layer": int(args.focus_layer),
        "status": "starting",
    }
    write_json(out, payload)

    t0 = time.perf_counter()
    print("[first-div] loading HF CPU...", flush=True)
    os.environ.setdefault(
        "HF_MODULES_CACHE",
        str(ref_dir / "hf_modules"),
    )
    model, _tok, _cfg = load_hf(model_dir)
    print("[first-div] HF capture...", flush=True)
    hf = run_hf_capture(model, input_ids, args.max_layers, args.focus_layer)
    del model
    gc.collect()

    print("[first-div] TC capture...", flush=True)
    cfg = TrinityNanoConfig.from_model_dir(model_dir)
    with tc.no_grad(), SafeTensorSource(model_dir) as source:
        tc_cap = run_tc_capture(cfg, source, input_ids, args.max_layers, args.focus_layer)

    rows = []
    rows.append(compare_arrays("embed", hf["embed"], tc_cap["embed"]))
    rows.append(compare_arrays("embed_mup", hf["embed_mup"], tc_cap["embed_mup"]))
    for i in range(args.max_layers):
        rows.append(
            compare_arrays(
                f"layer_{i}_out",
                hf["layers"][i]["out"],
                tc_cap["layers"][i]["out"],
            )
        )

    moe_rows = []
    keys = [
        "mlp_in",
        "router_logits",
        "scores_sigmoid",
        "expert_bias",
        "scores_plus_bias",
        "selected_experts",
        "gathered_pre_norm",
        "top_scores_post_norm",
        "top_scores_post_scale",
        "top_scores",
        "shared_out",
        "mlp_out",
    ]
    # HF shared_out is flat [T,H]; TC is [1,T,H]
    if hf["moe"].get("shared_out") is not None and hf["moe"]["shared_out"].ndim == 2:
        so = hf["moe"]["shared_out"]
        hf["moe"]["shared_out"] = so.reshape(1, so.shape[0], so.shape[1])
    if hf["moe"].get("mlp_out") is not None and tc_cap["moe"].get("mlp_out") is not None:
        # ensure both 3d
        pass

    for k in keys:
        a = hf["moe"].get(k)
        b = tc_cap["moe"].get(k)
        if k == "selected_experts" and a is not None and b is not None:
            # sort each row for set-compare too
            a_s = np.sort(a, axis=-1)
            b_s = np.sort(b, axis=-1)
            moe_rows.append(
                {
                    "name": k,
                    "exact_match": bool(np.array_equal(a, b)),
                    "sorted_exact_match": bool(np.array_equal(a_s, b_s)),
                    "hf": a.tolist(),
                    "tc": b.tolist(),
                }
            )
        else:
            moe_rows.append(compare_arrays(k, a, b if a is not None else None))

    # candidate checklist numerical verdicts
    checklist = {}
    # (a) expert_bias in selection only
    if "gathered_pre_norm" in hf["moe"] and "scores_plus_bias" in hf["moe"]:
        # TC gathered should match HF gathered (bias-free)
        checklist["a_expert_bias_selection_only"] = compare_arrays(
            "gathered_pre_norm",
            hf["moe"]["gathered_pre_norm"],
            tc_cap["moe"]["gathered_pre_norm"],
        )
    checklist["b_route_norm"] = compare_arrays(
        "top_scores_post_norm",
        hf["moe"]["top_scores_post_norm"],
        tc_cap["moe"]["top_scores_post_norm"],
    )
    checklist["c_route_scale"] = compare_arrays(
        "top_scores_post_scale",
        hf["moe"]["top_scores_post_scale"],
        tc_cap["moe"]["top_scores_post_scale"],
    )
    checklist["shared"] = compare_arrays(
        "shared_out", hf["moe"].get("shared_out"), tc_cap["moe"].get("shared_out")
    )
    checklist["mlp_out"] = compare_arrays(
        "mlp_out", hf["moe"].get("mlp_out"), tc_cap["moe"].get("mlp_out")
    )
    checklist["mlp_in"] = compare_arrays(
        "mlp_in", hf["moe"].get("mlp_in"), tc_cap["moe"].get("mlp_in")
    )

    # first layer where max_abs_diff exceeds thresholds
    first_div = None
    for r in rows:
        if r.get("status") != "ok":
            continue
        if r["name"].startswith("layer_") and r["max_abs_diff"] > 0.05:
            first_div = r
            break

    payload.update(
        {
            "status": "ok",
            "wall_seconds": time.perf_counter() - t0,
            "layer_compare": rows,
            "moe_compare": moe_rows,
            "checklist": checklist,
            "first_layer_out_diff_gt_0p05": first_div,
            "hf_selected": hf["moe"].get("selected_experts").tolist()
            if hf["moe"].get("selected_experts") is not None
            else None,
            "tc_selected": tc_cap["moe"].get("selected_experts").tolist()
            if tc_cap["moe"].get("selected_experts") is not None
            else None,
            "hf_top_scores": hf["moe"].get("top_scores").tolist()
            if hf["moe"].get("top_scores") is not None
            else None,
            "tc_top_scores": tc_cap["moe"].get("top_scores").tolist()
            if tc_cap["moe"].get("top_scores") is not None
            else None,
        }
    )
    write_json(out, payload)
    print(json.dumps({"status": "ok", "artifact": str(out), "first": first_div}, indent=2))
    # dense print of checklist
    for k, v in checklist.items():
        print(f"CHECK {k}: {json.dumps(v)}", flush=True)
    for r in rows:
        if r.get("status") == "ok":
            print(
                f"LAYER {r['name']}: max={r['max_abs_diff']:.6g} mean={r['mean_abs_diff']:.6g}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
