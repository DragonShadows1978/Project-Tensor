#!/usr/bin/env python3
"""Capture Trinity Nano HF reference logits and pre-RoPE K/V receipts.

CPU bf16 is intentional: the bf16 checkpoint does not fit the 12GB GPU. The
script writes one NPZ per probe under artifacts/trinity_nano/reference_capture
plus a tokenizer/model metadata JSON receipt.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_MODEL_DIR = "/mnt/ForgeRealm/models/trinity-nano"
DEFAULT_OUT_DIR = "artifacts/trinity_nano/reference_capture"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--long-min-tokens", type=int, default=2056)
    parser.add_argument("--max-probes", type=int, default=3)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument(
        "--torch-dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
        help="HF load/compute dtype. float32 is the T4 accumulation-vs-semantic disambiguation path.",
    )
    return parser.parse_args()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_ids(ids: list[int]) -> str:
    payload = ",".join(str(int(x)) for x in ids)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def bf16_bits(t: torch.Tensor) -> np.ndarray:
    return (
        t.detach()
        .to(torch.bfloat16)
        .contiguous()
        .cpu()
        .view(torch.uint16)
        .numpy()
        .copy()
    )


def fp32_arr(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().contiguous().cpu().numpy().astype(np.float32, copy=False)


def torch_dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported torch dtype {name!r}")


def tokenizer_receipts(tokenizer, model_dir: Path) -> dict[str, Any]:
    chat_template = getattr(tokenizer, "chat_template", None) or ""
    plain = "The capital of France is"
    plain_false = tokenizer(plain, add_special_tokens=False).input_ids
    plain_true = tokenizer(plain, add_special_tokens=True).input_ids
    if plain_false and isinstance(plain_false[0], list):
        plain_false = plain_false[0]
    if plain_true and isinstance(plain_true[0], list):
        plain_true = plain_true[0]
    chat_rendered = None
    chat_ids = []
    if chat_template:
        chat_rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Write one compact sentence."}],
            tokenize=False,
            add_generation_prompt=True,
        )
        chat_ids = tokenizer(chat_rendered, add_special_tokens=False).input_ids
        if chat_ids and isinstance(chat_ids[0], list):
            chat_ids = chat_ids[0]
    special_path = model_dir / "special_tokens_map.json"
    special_map = json.loads(special_path.read_text(encoding="utf-8")) if special_path.exists() else {}
    return {
        "tokenizer_class": tokenizer.__class__.__name__,
        "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "add_bos_token": getattr(tokenizer, "add_bos_token", None),
        "add_eos_token": getattr(tokenizer, "add_eos_token", None),
        "chat_template_present": bool(chat_template),
        "chat_template_sha256": sha256_text(chat_template) if chat_template else None,
        "chat_template_preview": chat_template[:400],
        "special_tokens_map": special_map,
        "plain_add_special_false_ids": [int(x) for x in plain_false],
        "plain_add_special_true_ids": [int(x) for x in plain_true],
        "plain_true_starts_with_bos": bool(plain_true and plain_true[0] == tokenizer.bos_token_id),
        "chat_probe_rendered": chat_rendered,
        "chat_probe_ids": [int(x) for x in chat_ids],
    }


def install_default_rope_shim() -> str | None:
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    def trinity_default_rope(config, device=None, seq_len=None, layer_type=None, **kwargs):
        del seq_len, layer_type, kwargs
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        base = float(getattr(config, "rope_theta", 10000.0))
        # Force CPU concrete storage. from_pretrained constructs modules under
        # a meta device first; building inv_freq on meta (or default-meta)
        # leaves a non-materialized table that is later observed as garbage
        # zeros/denorms after weight load (parity-killing for sliding RoPE).
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32, device="cpu")
                / float(head_dim)
            )
        )
        return inv_freq, 1.0

    # Always install/overwrite: transformers v5 no longer ships a "default"
    # entry, and a previously-registered broken one must not stick.
    ROPE_INIT_FUNCTIONS["default"] = trinity_default_rope
    return (
        "installed/overwrote ROPE_INIT_FUNCTIONS['default'] with CPU-concrete "
        "theta/head_dim inv_freq (avoids meta-device corruption)"
    )


def reinject_rotary_inv_freq(model) -> str:
    """Re-materialize AfmoeRotaryEmbedding.inv_freq after weight load.

    Observed 2026-07-08/09: even with a correct ROPE_INIT_FUNCTIONS['default'],
    after from_pretrained the buffer is zeros/denorms. Sliding-layer RoPE then
    only rotates dim-0 pairs. Call this after load, before any forward.
    """
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    rotary = model.model.rotary_emb
    fn = ROPE_INIT_FUNCTIONS.get(getattr(rotary, "rope_type", "default")) or ROPE_INIT_FUNCTIONS["default"]
    inv_freq, attention_scaling = fn(model.config, device="cpu")
    with torch.no_grad():
        rotary.inv_freq = inv_freq.to(dtype=torch.float32)
        rotary.original_inv_freq = inv_freq.to(dtype=torch.float32).clone()
        rotary.attention_scaling = float(attention_scaling)
    return (
        f"reinjected inv_freq len={int(inv_freq.numel())} "
        f"first={float(inv_freq[0]):.6g} last={float(inv_freq[-1]):.6g} "
        f"scale={float(attention_scaling)}"
    )


def install_missing_key_init_shim() -> str:
    from transformers.modeling_utils import PreTrainedModel

    if getattr(PreTrainedModel, "_trinity_missing_key_shim", False):
        return "already installed"

    def no_initialize_missing_keys(self, is_quantized):
        del self, is_quantized
        return None

    PreTrainedModel._initialize_missing_keys = no_initialize_missing_keys
    PreTrainedModel._trinity_missing_key_shim = True
    return (
        "installed PreTrainedModel._initialize_missing_keys no-op; "
        "remote AfmoeRotaryEmbedding already initializes inv_freq but lacks "
        "Transformers v5 compute_default_rope_parameters hook"
    )


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
    return (
        "installed remote modeling_afmoe mask wrappers for Transformers v5 "
        "inputs_embeds/cache_position API drift"
    )


def make_long_prompt(tokenizer, min_tokens: int) -> str:
    base = (
        "Sliding-window parity receipt line 0000: alpha beta gamma delta "
        "epsilon zeta eta theta. "
    )
    text = base
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    n = 1
    while len(ids) <= int(min_tokens):
        n *= 2
        text = base * n
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if ids and isinstance(ids[0], list):
            ids = ids[0]
    return text


def build_probes(tokenizer, long_min_tokens: int) -> list[dict[str, Any]]:
    chat_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Write one compact sentence about deterministic parity."}],
        tokenize=False,
        add_generation_prompt=True,
    )
    long_text = make_long_prompt(tokenizer, long_min_tokens)
    return [
        {
            "name": "probe_00_plain_short",
            "kind": "raw",
            "text": "The capital of France is",
            "add_special_tokens": False,
        },
        {
            "name": "probe_01_chat_short",
            "kind": "chat_template",
            "text": chat_text,
            "add_special_tokens": False,
        },
        {
            "name": "probe_02_long_sliding_boundary",
            "kind": "raw_long_gt_2048",
            "text": long_text,
            "add_special_tokens": False,
        },
    ]


def register_kv_hooks(
    model, *, store_dtype: str = "bfloat16"
) -> tuple[list[Any], dict[int, tuple[np.ndarray, np.ndarray]]]:
    captures: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook(module, args, kwargs):
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None:
                hidden_states = args[0]
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, module.head_dim)
            with torch.inference_mode():
                key_states = module.k_proj(hidden_states).view(hidden_shape)
                value_states = module.v_proj(hidden_states).view(hidden_shape)
                key_states = module.k_norm(key_states)
                key_states = key_states.transpose(1, 2).contiguous()
                value_states = value_states.transpose(1, 2).contiguous()
                if store_dtype == "float32":
                    captures[int(layer_idx)] = (fp32_arr(key_states), fp32_arr(value_states))
                else:
                    captures[int(layer_idx)] = (
                        bf16_bits(key_states),
                        bf16_bits(value_states),
                    )

        return hook

    for i, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.register_forward_pre_hook(make_hook(i), with_kwargs=True))
    return handles, captures


def run_probe(
    model,
    tokenizer,
    probe: dict[str, Any],
    steps: int,
    capture_kv: bool,
    *,
    kv_store_dtype: str = "bfloat16",
):
    ids = tokenizer(
        probe["text"],
        add_special_tokens=bool(probe.get("add_special_tokens", False)),
        return_tensors="pt",
    ).input_ids
    ids = ids.to(torch.long)
    input_ids_list = [int(x) for x in ids[0].tolist()]
    handles = []
    captures = {}
    if capture_kv:
        handles, captures = register_kv_hooks(model, store_dtype=kv_store_dtype)

    logits_rows = []
    top5_ids = []
    top5_logits = []
    generated = []
    current = ids
    past = None
    step_wall = []
    with torch.inference_mode():
        for step in range(int(steps)):
            t0 = time.perf_counter()
            out = model(
                input_ids=current,
                past_key_values=past,
                use_cache=True,
                logits_to_keep=1,
            )
            logits = out.logits[:, -1, :].float().cpu().numpy()[0].astype(np.float32)
            logits_rows.append(logits)
            idx = np.argsort(-logits)[:5].astype(np.int64)
            top5_ids.append(idx)
            top5_logits.append(logits[idx].astype(np.float32))
            next_id = int(np.argmax(logits))
            generated.append(next_id)
            current = torch.tensor([[next_id]], dtype=torch.long)
            past = out.past_key_values
            step_wall.append(time.perf_counter() - t0)
            if step == 0 and handles:
                for handle in handles:
                    handle.remove()
                handles = []
    for handle in handles:
        handle.remove()

    return {
        "input_ids": np.asarray(input_ids_list, dtype=np.int64),
        "generated_ids": np.asarray(generated, dtype=np.int64),
        "logits": np.stack(logits_rows).astype(np.float32),
        "top5_ids": np.stack(top5_ids).astype(np.int64),
        "top5_logits": np.stack(top5_logits).astype(np.float32),
        "step_wall_seconds": np.asarray(step_wall, dtype=np.float32),
        "captures": captures,
    }


def save_probe_npz(out_dir: Path, probe: dict[str, Any], result: dict[str, Any]) -> Path:
    path = out_dir / f"{probe['name']}.npz"
    np.savez_compressed(
        path,
        input_ids=result["input_ids"],
        generated_ids=result["generated_ids"],
        logits=result["logits"],
        top5_ids=result["top5_ids"],
        top5_logits=result["top5_logits"],
        step_wall_seconds=result["step_wall_seconds"],
    )
    return path


def save_kv_npz(
    out_dir: Path,
    captures: dict[int, tuple[np.ndarray, np.ndarray]],
    *,
    store_dtype: str = "bfloat16",
) -> Path:
    if store_dtype == "float32":
        path = out_dir / "probe_00_prerope_kv_fp32.npz"
        k_key, v_key = "k_fp32", "v_fp32"
    else:
        path = out_dir / "probe_00_prerope_kv_bf16_bits.npz"
        k_key, v_key = "k_bf16_u16", "v_bf16_u16"
    payload: dict[str, np.ndarray] = {
        "layer_indices": np.asarray(sorted(captures), dtype=np.int64),
        "store_dtype": np.asarray([store_dtype]),
    }
    for layer_idx in sorted(captures):
        k, v = captures[layer_idx]
        payload[f"layer_{layer_idx:03d}_{k_key}"] = k
        payload[f"layer_{layer_idx:03d}_{v_key}"] = v
        payload[f"layer_{layer_idx:03d}_k_shape"] = np.asarray(k.shape, dtype=np.int64)
        payload[f"layer_{layer_idx:03d}_v_shape"] = np.asarray(v.shape, dtype=np.int64)
    np.savez_compressed(path, **payload)
    return path


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_MODULES_CACHE", str(out_dir / "hf_modules"))

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True
    )
    rope_shim = install_default_rope_shim()
    missing_key_shim = install_missing_key_init_shim()
    config = AutoConfig.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True
    )
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.pad_token_id = tokenizer.pad_token_id
    probes = build_probes(tokenizer, args.long_min_tokens)[: int(args.max_probes)]
    torch_dtype = torch_dtype_from_name(args.torch_dtype)
    meta: dict[str, Any] = {
        "schema": "trinity_nano_hf_reference_capture_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_dir": str(model_dir),
        "out_dir": str(out_dir),
        "steps": int(args.steps),
        "torch_version": torch.__version__,
        "torch_dtype_requested": args.torch_dtype,
        "device_map": "cpu",
        "attn_implementation": args.attn_implementation,
        "hf_modules_cache": os.environ["HF_MODULES_CACHE"],
        "rope_default_shim": rope_shim,
        "missing_key_init_shim": missing_key_shim,
        "config_token_ids_injected": {
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "reason": "custom AfmoeConfig lacks these attributes but AfmoeModel reads pad_token_id",
        },
        "tokenizer": tokenizer_receipts(tokenizer, model_dir),
        "probes": [],
        "status": "loading_model",
    }
    meta_path = out_dir / "metadata.json"
    write_json(meta_path, meta)

    print(
        f"[trinity-capture] loading model from {model_dir} dtype={args.torch_dtype}",
        flush=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        config=config,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch_dtype,
        device_map="cpu",
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    )
    model.eval()
    masking_shim = install_remote_masking_shim(model)
    inv_freq_receipt = reinject_rotary_inv_freq(model)
    meta["status"] = "running_probes"
    meta["model_class"] = model.__class__.__name__
    meta["masking_api_shim"] = masking_shim
    meta["rotary_inv_freq_reinject"] = inv_freq_receipt
    write_json(meta_path, meta)
    print(f"[trinity-capture] model loaded; {inv_freq_receipt}", flush=True)
    print("[trinity-capture] running probes", flush=True)

    started = time.perf_counter()
    for i, probe in enumerate(probes):
        probe_start = time.perf_counter()
        ids = tokenizer(
            probe["text"],
            add_special_tokens=bool(probe.get("add_special_tokens", False)),
        ).input_ids
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        probe_meta = {
            "index": int(i),
            "name": probe["name"],
            "kind": probe["kind"],
            "add_special_tokens": bool(probe.get("add_special_tokens", False)),
            "text_sha256": sha256_text(probe["text"]),
            "text_preview": probe["text"][:240],
            "input_token_count": int(len(ids)),
            "input_ids_sha256": sha256_ids([int(x) for x in ids]),
            "status": "running",
        }
        meta["probes"].append(probe_meta)
        write_json(meta_path, meta)
        print(
            f"[trinity-capture] starting {probe['name']} tokens={len(ids)}",
            flush=True,
        )

        result = run_probe(
            model,
            tokenizer,
            probe,
            args.steps,
            capture_kv=(i == 0),
            kv_store_dtype=args.torch_dtype,
        )
        npz_path = save_probe_npz(out_dir, probe, result)
        decoded_generated = tokenizer.decode(
            [int(x) for x in result["generated_ids"].tolist()],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        probe_meta.update(
            {
                "status": "ok",
                "artifact": str(npz_path),
                "generated_ids": [int(x) for x in result["generated_ids"].tolist()],
                "generated_text": decoded_generated,
                "top5_step0_ids": [int(x) for x in result["top5_ids"][0].tolist()],
                "top5_step0_logits": [float(x) for x in result["top5_logits"][0].tolist()],
                "wall_seconds": time.perf_counter() - probe_start,
            }
        )
        if i == 0:
            kv_path = save_kv_npz(
                out_dir, result["captures"], store_dtype=args.torch_dtype
            )
            probe_meta["prerope_kv_artifact"] = str(kv_path)
            probe_meta["prerope_kv_layers"] = int(len(result["captures"]))
            probe_meta["prerope_kv_store_dtype"] = args.torch_dtype
        write_json(meta_path, meta)
        print(
            f"[trinity-capture] finished {probe['name']} "
            f"wall={probe_meta['wall_seconds']:.1f}s",
            flush=True,
        )
        del result
        gc.collect()

    meta["status"] = "ok"
    meta["wall_seconds"] = time.perf_counter() - started
    write_json(meta_path, meta)
    print(json.dumps({"status": "ok", "metadata": str(meta_path)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
