"""core/model_loader.py — Load model weights from disk into TensorCUDA tensors.

Supports two input formats:
  1. Safetensors (bf16 weights)  -> quantize to INT4 on load
  2. GGUF QAT (q4_0 pre-quantized)  -> exact import, no requantization

No PyTorch.  Uses safetensors and gguf libraries for file I/O only;
all tensor creation goes through tensor_cuda.
"""
from __future__ import annotations

import gc
import os
from typing import Callable

import numpy as np

import tensor_cuda as tc
from core.mistral7b_tc import QuantLinearTC


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _q40_repack(raw: np.ndarray, N: int, K: int) -> tuple:
    """GGUF q4_0 blocks -> engine packed layout, EXACT (no requant).

    q4_0 block = 18 bytes: fp16 scale d + 16 qs bytes where
    x[j] = (qs[j] & 0xF) - 8 and x[j+16] = (qs[j] >> 4) - 8. The raw
    nibble IS x+8, which is exactly the engine's q under the
    symmetric-8 convention (w = q*s - 8*s, empty zeros tensor).
    Only the nibble ORDER differs (GGUF j/j+16 interleave vs engine
    even/odd pairs)."""
    blk = np.ascontiguousarray(raw).reshape(N * (K // 32), 18)
    scales = blk[:, :2].copy().view(np.float16).reshape(N, K // 32)
    qs = blk[:, 2:]
    q = np.concatenate([qs & 0x0F, qs >> 4], axis=1)     # (B, 32) = x+8
    q = q.reshape(N, K)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    return packed, scales


class Q40LinearTC:
    """Exact q4_0 import: packed nibbles + fp16 scales at group 32,
    EMPTY zeros tensor (engine symmetric-8 path, z = -8*s in-register).
    Dispatch mirrors QuantLinearTC (GEMV/tile/two-stage)."""

    def __init__(self, packed_np: np.ndarray, scales_np: np.ndarray):
        self.out_features = packed_np.shape[0]
        self.in_features = packed_np.shape[1] * 2
        self.group_size = 32
        self.packed = tc.tensor(packed_np, dtype="uint8")
        self.scales = tc.tensor(np.ascontiguousarray(scales_np),
                                dtype="float16")
        self.zeros = tc.tensor(np.zeros((0,), np.float16), dtype="float16")

    def __call__(self, x):
        return tc.int4_linear_fused(x, self.packed, self.scales,
                                    self.zeros, self.group_size)


# ------------------------------------------------------------------
# Safetensors loader
# ------------------------------------------------------------------
def _has_safetensors():
    try:
        from safetensors import safe_open
        return True
    except ImportError:
        return False


def _has_gguf():
    try:
        from gguf import GGUFReader
        return True
    except ImportError:
        return False


def load_safetensors(
    model_dir: str,
    weight_callback: Callable[[str, np.ndarray], None],
    progress: bool = True,
):
    """Iterate over all safetensors files in model_dir, calling
    weight_callback(name, numpy_array) for each tensor.

    weight_callback receives the FULL key name (e.g.
    "model.language_model.layers.0.self_attn.q_proj.weight")
    and a float32 numpy array.
    """
    from safetensors import safe_open

    paths = sorted(
        [os.path.join(model_dir, f)
         for f in os.listdir(model_dir)
         if f.endswith(".safetensors")]
    )
    if not paths:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")

    for path in paths:
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                import torch
                arr = f.get_tensor(key).to(torch.float32).numpy()
                weight_callback(key, np.ascontiguousarray(arr))
                del arr
                gc.collect()


# ------------------------------------------------------------------
# GGUF QAT loader
# ------------------------------------------------------------------
def load_gguf(
    gguf_path: str,
    weight_callback: Callable[[str, np.ndarray], None],
    q40_callback: Callable[[str, np.ndarray, np.ndarray, int, int], None] | None = None,
    progress: bool = True,
):
    """Iterate over all tensors in a GGUF file.

    For fp32/fp16 tensors: weight_callback(name, array)
    For q4_0 tensors:       q40_callback(name, packed, scales, N, K)  if provided,
                            else weight_callback with dequantized array
    """
    from gguf import GGUFReader
    from gguf.quants import dequantize

    r = GGUFReader(gguf_path)
    for t in r.tensors:
        name = t.name
        data = np.asarray(t.data)
        tt = int(t.tensor_type)

        if tt == 0:   # F32
            weight_callback(name, data.astype(np.float32))
        elif tt == 1:  # F16
            weight_callback(name, data.astype(np.float32))
        elif tt == 2:  # Q4_0
            N = data.shape[0] if data.ndim > 0 else 1
            K = int(np.prod(data.shape[1:])) if data.ndim > 1 else 32
            packed, scales = _q40_repack(data, N, K)
            if q40_callback is not None:
                q40_callback(name, packed, scales, N, K)
            else:
                # Dequantize and return fp32
                dq = dequantize(t.data, tt).astype(np.float32)
                weight_callback(name, dq)
        else:
            # Fallback: dequantize to fp32
            dq = dequantize(t.data, tt).astype(np.float32)
            weight_callback(name, dq)


# ------------------------------------------------------------------
# Convenience: load a complete Gemma4 from safetensors (post-hoc INT4)
# ------------------------------------------------------------------
def load_gemma4_safetensors(
    model_dir: str,
    cfg,
    progress: bool = True,
):
    """Load Gemma 4 weights from safetensors, quantizing to INT4 on load.

    Returns a dict of loaded components ready to attach to a model:
      { "embed", "norm", "layers": [{"q", "k", "v", "o", "gate", "up",
         "down", "norms", "scalar"}, ...] }
    """
    result = {"layers": []}
    prefix = "model.language_model"

    def cb(key: str, arr: np.ndarray):
        # Embedding / head
        if key == f"{prefix}.embed_tokens.weight":
            result["embed"] = arr.copy()
            return
        if key == f"{prefix}.norm.weight":
            result["norm"] = arr.copy()
            return

        # Parse layer keys: model.language_model.layers.{i}.{component}
        parts = key.split(".")
        if len(parts) < 6 or parts[3] != "layers":
            return

        layer_idx = int(parts[4])
        while len(result["layers"]) <= layer_idx:
            result["layers"].append({})

        L = result["layers"][layer_idx]
        comp = ".".join(parts[5:])

        if comp == "self_attn.q_proj.weight":
            L["q"] = QuantLinearTC(arr, group_size=128)
        elif comp == "self_attn.k_proj.weight":
            L["k"] = QuantLinearTC(arr, group_size=128)
        elif comp == "self_attn.v_proj.weight":
            L["v"] = QuantLinearTC(arr, group_size=128)
        elif comp == "self_attn.o_proj.weight":
            L["o"] = QuantLinearTC(arr, group_size=128)
        elif comp == "mlp.gate_proj.weight":
            L["gate"] = QuantLinearTC(arr, group_size=128)
        elif comp == "mlp.up_proj.weight":
            L["up"] = QuantLinearTC(arr, group_size=128)
        elif comp == "mlp.down_proj.weight":
            L["down"] = QuantLinearTC(arr, group_size=128)
        elif comp.endswith("_norm.weight"):
            if "norms" not in L:
                L["norms"] = {}
            L["norms"][comp] = arr.copy()
        elif comp == "layer_scalar":
            L["scalar"] = float(arr[0])

    load_safetensors(model_dir, cb, progress=progress)
    return result


# ------------------------------------------------------------------
# Convenience: load a complete Gemma4 from GGUF QAT (exact, no requant)
# ------------------------------------------------------------------
def load_gemma4_gguf(
    gguf_path: str,
    cfg,
    progress: bool = True,
):
    """Load Gemma 4 weights from GGUF QAT (q4_0 exact import).

    Returns the same dict structure as load_gemma4_safetensors.
    """
    result = {"layers": []}

    def w_cb(name: str, arr: np.ndarray):
        # fp32 weights (norms, scales, etc.)
        if name == "token_embd.weight":
            result["embed"] = arr.copy()
            return
        if name == "output_norm.weight":
            result["norm"] = arr.copy()
            return

        # Layer components: blk.{i}.{component}
        if not name.startswith("blk."):
            return
        parts = name.split(".")
        layer_idx = int(parts[1])
        while len(result["layers"]) <= layer_idx:
            result["layers"].append({})
        L = result["layers"][layer_idx]
        comp = ".".join(parts[2:])

        if comp.endswith("_norm.weight") or comp == "layer_output_scale.weight":
            if "norms" not in L:
                L["norms"] = {}
            L["norms"][comp] = arr.copy()

    def q40_cb(name: str, packed: np.ndarray, scales: np.ndarray, N: int, K: int):
        if not name.startswith("blk."):
            return
        parts = name.split(".")
        layer_idx = int(parts[1])
        while len(result["layers"]) <= layer_idx:
            result["layers"].append({})
        L = result["layers"][layer_idx]
        comp = ".".join(parts[2:])

        q40 = Q40LinearTC(packed, scales)

        if comp == "attn_q.weight":
            L["q"] = q40
        elif comp == "attn_k.weight":
            L["k"] = q40
        elif comp == "attn_v.weight":
            L["v"] = q40
        elif comp == "attn_output.weight":
            L["o"] = q40
        elif comp == "ffn_gate.weight":
            L["gate"] = q40
        elif comp == "ffn_up.weight":
            L["up"] = q40
        elif comp == "ffn_down.weight":
            L["down"] = q40

    load_gguf(gguf_path, w_cb, q40_cb, progress=progress)
    return result
