"""Trinity Nano (afmoe) on the tensor_cuda engine.

The adapter follows the existing Project-Tensor model-object contract:
`__call__(input_ids_np, kv_caches=None, position_offset=0,
last_token_only=False, max_layers=None)` returns `(logits, new_caches)`.

Two weight modes are intentionally explicit:
  - bf16/plain linears for streamed parity probes.
  - affine INT4 linears for the resident inference target.

The MoE router implements Trinity's sigmoid + expert_bias selection semantics.
Expert recombination follows the GPT-OSS diagnostic precedent: selected experts
are run per token, avoiding an engine scatter-add dependency in this first port.
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tensor_cuda"))

import tensor_cuda as tc  # noqa: E402
from tensor_cuda import functional as F  # noqa: E402
from tensor_cuda.quantization import quantize_affine_per_group  # noqa: E402


GROUP_SIZE = 128


def _compute_dtype() -> str:
    return BlockTC.COMPUTE_DTYPE


def _cast(t):
    dt = _compute_dtype()
    return t.half() if dt == "float16" else t.astype(dt)


def _to_dtype(t, dtype: str):
    return t.half() if dtype == "float16" else t.astype(dtype)


def _nbytes_tensor(t) -> int:
    item = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
        "uint8": 1,
        "int64": 8,
        "int32": 4,
        "bool": 1,
    }.get(str(t.dtype), 4)
    n = 1
    for dim in t.shape:
        n *= int(dim)
    return int(n * item)


def _sigmoid_topk_route(
    scores,
    expert_bias,
    top_k: int,
    route_norm: bool,
    route_scale: float,
):
    scores = scores.float().sigmoid()
    if expert_bias is not None:
        bias = expert_bias if expert_bias.dtype == scores.dtype else expert_bias.astype(scores.dtype)
        _, selected = (scores + bias).topk(int(top_k), True)
        top_scores = scores.gather(1, selected)
    else:
        top_scores, selected = scores.topk(int(top_k), True)
    if route_norm:
        denom = top_scores.sum([-1], True) + 1e-20
        top_scores = top_scores / denom
    top_scores = top_scores * float(route_scale)
    return top_scores, selected


def _band_mask(L: int, S: int, window: int, device: str, dtype: str):
    key = (int(L), int(S), int(window), device, dtype)
    cached = _BAND_MASK_CACHE.get(key)
    if cached is not None:
        return cached
    i = np.arange(L, dtype=np.int64)[:, None] + (S - L)
    j = np.arange(S, dtype=np.int64)[None, :]
    visible = (j <= i) & (j > i - int(window))
    bias = np.where(visible, 0.0, -1e4).astype(np.float32)
    out = tc.tensor(np.ascontiguousarray(bias), device=device)
    if dtype in ("float16", "bfloat16"):
        out = out.astype(dtype)
    if len(_BAND_MASK_CACHE) >= 16:
        _BAND_MASK_CACHE.clear()
    _BAND_MASK_CACHE[key] = out
    return out


_BAND_MASK_CACHE: dict[tuple[int, int, int, str, str], Any] = {}


def _repeat_kv(x, n_rep: int):
    if int(n_rep) == 1:
        return x
    B, KV, S, D = x.shape
    return x.unsqueeze(2).expand([B, KV, int(n_rep), S, D]).reshape(
        [B, KV * int(n_rep), S, D]
    )


def _trinity_scaled_attention(q, k, v, *, scale: float, is_causal: bool, attn_mask=None):
    L, S = int(q.shape[-2]), int(k.shape[-2])
    scores = tc.matmul(q, k, alpha=float(scale), trans_b=True).float()
    if is_causal:
        scores = scores + F._causal_mask(L, S, q.device.split(":")[0], "float32")
    if attn_mask is not None:
        scores = scores + attn_mask.float()
    weights = scores.softmax(-1)
    if q.dtype != "float32":
        weights = weights.astype(q.dtype)
    return tc.matmul(weights, v)


def _quantize_int4_affine(w_fp32: np.ndarray, group_size: int = GROUP_SIZE):
    q = quantize_affine_per_group(w_fp32, 4, group_size)
    return q.packed, q.scales, q.zeros


class BlockTC:
    """Global compute dtype knob used by the Trinity adapter."""

    COMPUTE_DTYPE = "bfloat16"


class LinearTC:
    """Plain bf16/fp16 linear. Weight is stored transposed for direct matmul."""

    DTYPE = "bfloat16"

    def __init__(self, weight_fp32: np.ndarray):
        if weight_fp32.ndim != 2:
            raise ValueError(f"linear weight must be rank-2, got {weight_fp32.shape}")
        self.out_features, self.in_features = weight_fp32.shape
        self.wT = tc.tensor(
            np.ascontiguousarray(weight_fp32.T.astype(np.float32)), dtype="float32"
        ).astype(self.DTYPE)
        self._vram = int(self.out_features * self.in_features * 2)

    def __call__(self, x):
        return tc.matmul(x, self.wT)

    def vram_bytes(self) -> int:
        return self._vram


class QuantLinearTC:
    """Affine INT4 linear using Project-Tensor's validated int4 kernels."""

    USE_FUSED = False
    FUSED_DECODE = True
    FUSED_M_MAX = 8
    LOAD_CONTEXT: str | None = None

    def __init__(self, weight_fp32: np.ndarray, group_size: int = GROUP_SIZE):
        if weight_fp32.ndim != 2:
            raise ValueError(f"linear weight must be rank-2, got {weight_fp32.shape}")
        self.out_features, self.in_features = weight_fp32.shape
        self.group_size = int(group_size)
        try:
            packed, scales, zeros = _quantize_int4_affine(weight_fp32, self.group_size)
            self.packed = tc.tensor(np.ascontiguousarray(packed), dtype="uint8")
            self.scales = tc.tensor(np.ascontiguousarray(scales), dtype="float16")
            self.zeros = tc.tensor(np.ascontiguousarray(zeros), dtype="float16")
            self._vram = int(packed.nbytes + scales.nbytes + zeros.nbytes)
        except Exception as exc:
            ctx = f" tensor={self.LOAD_CONTEXT}" if self.LOAD_CONTEXT else ""
            raise RuntimeError(
                f"QuantLinearTC INT4 init failed{ctx} shape="
                f"({self.out_features},{self.in_features})"
            ) from exc

    def __call__(self, x):
        fused = bool(self.USE_FUSED)
        if not fused and self.FUSED_DECODE:
            rows = 1
            for dim in x.shape[:-1]:
                rows *= int(dim)
            fused = rows <= int(self.FUSED_M_MAX)
        if fused:
            return tc.int4_linear_fused(
                x, self.packed, self.scales, self.zeros, self.group_size
            )
        return tc.int4_linear(x, self.packed, self.scales, self.zeros, self.group_size)

    def vram_bytes(self) -> int:
        return self._vram


class RMSNormTC:
    """Afmoe-matching RMSNorm.

    modeling_afmoe.AfmoeRMSNorm is NOT the fused Llama order. It does:
      y = x.float(); y = y * rsqrt(mean(y^2)+eps); return weight * y.to(x.dtype)
    i.e. cast the normalized vector back to the input dtype BEFORE the affine
    multiply. The TensorCUDA fused rms_norm kernel multiplies weight in fp32
    and rounds once at the store (Llama-style). That is a real first-divergence
    against Trinity (layer-0 input_layernorm max_abs 0.125 on identical embeds).
    USE_FUSED is therefore off by default for this port.
    """

    USE_FUSED = False

    def __init__(self, dim: int, eps: float):
        self.weight = tc.tensor(np.ones(int(dim), dtype=np.float32), dtype="float32")
        self.eps = float(eps)

    def __call__(self, x):
        # Fused kernel is Llama-order (weight inside the fp32 chain). Only safe
        # when the caller explicitly opts in AND accepts that mismatch.
        if self.USE_FUSED and hasattr(tc, "rms_norm") and not tc.is_grad_enabled():
            return tc.rms_norm(x, self.weight, self.eps)
        xf = x.float()
        ms = (xf * xf).mean([-1], True)
        normed = xf * (ms + self.eps).pow(-0.5)
        # Afmoe: weight * normed.to(input_dtype)
        in_dtype = x.dtype
        if in_dtype != "float32":
            normed = _to_dtype(normed, in_dtype)
        w = self.weight
        if str(w.dtype) != str(normed.dtype):
            w = _to_dtype(w, normed.dtype)
        return normed * w


class HostEmbedding:
    """Host-resident embedding table, copied to the GPU only for requested rows."""

    def __init__(self):
        self.weight: np.ndarray | None = None

    def __call__(self, input_ids_np):
        if self.weight is None:
            raise RuntimeError("HostEmbedding weight is not loaded")
        rows = self.weight[np.asarray(input_ids_np, dtype=np.int64)]
        return tc.tensor(np.ascontiguousarray(rows), dtype=_compute_dtype())

    @property
    def shape(self):
        return (0, 0) if self.weight is None else self.weight.shape


class SafeTensorSource:
    """Open all checkpoint shards once and serve tensors by weight-map name."""

    def __init__(self, model_dir: str | os.PathLike[str]):
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.index_path = self.model_dir / "model.safetensors.index.json"
        if not self.index_path.exists():
            raise FileNotFoundError(self.index_path)
        idx = json.loads(self.index_path.read_text(encoding="utf-8"))
        self.weight_map: dict[str, str] = dict(idx["weight_map"])
        self.metadata = dict(idx.get("metadata", {}))
        self._stack: ExitStack | None = None
        self._handles: dict[str, Any] = {}

    def __enter__(self) -> "SafeTensorSource":
        from safetensors import safe_open

        self._stack = ExitStack()
        for shard in sorted(set(self.weight_map.values())):
            path = self.model_dir / shard
            self._handles[shard] = self._stack.enter_context(
                safe_open(str(path), framework="pt")
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._stack is not None:
            self._stack.close()
        self._stack = None
        self._handles.clear()

    def get_np(self, name: str, dtype=np.float32) -> np.ndarray:
        if name not in self.weight_map:
            raise KeyError(name)
        if not self._handles:
            raise RuntimeError("SafeTensorSource must be used as a context manager")
        tensor = self._handles[self.weight_map[name]].get_tensor(name)
        arr = tensor.cpu().float().numpy()
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return np.ascontiguousarray(arr)

    def has(self, name: str) -> bool:
        return name in self.weight_map


@dataclass
class TrinityNanoConfig:
    vocab_size: int = 200192
    hidden_size: int = 1024
    intermediate_size: int = 3072
    moe_intermediate_size: int = 256
    num_hidden_layers: int = 56
    num_dense_layers: int = 2
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    head_dim: int = 128
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    sliding_window: int = 2048
    num_experts: int = 128
    num_experts_per_tok: int = 8
    num_shared_experts: int = 1
    score_func: str = "sigmoid"
    route_norm: bool = True
    route_scale: float = 2.826
    hidden_act: str = "silu"
    mup_enabled: bool = True
    tie_word_embeddings: bool = False
    layer_types: tuple[str, ...] = ()
    group_size: int = GROUP_SIZE

    @classmethod
    def from_model_dir(cls, model_dir: str | os.PathLike[str]) -> "TrinityNanoConfig":
        cfg_path = Path(model_dir).expanduser().resolve() / "config.json"
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        layer_types = tuple(
            data.get("layer_types")
            or [
                "sliding_attention" if bool((i + 1) % data.get("global_attn_every_n_layers", 4))
                else "full_attention"
                for i in range(int(data["num_hidden_layers"]))
            ]
        )
        return cls(
            vocab_size=int(data["vocab_size"]),
            hidden_size=int(data["hidden_size"]),
            intermediate_size=int(data["intermediate_size"]),
            moe_intermediate_size=int(data["moe_intermediate_size"]),
            num_hidden_layers=int(data["num_hidden_layers"]),
            num_dense_layers=int(data["num_dense_layers"]),
            num_attention_heads=int(data["num_attention_heads"]),
            num_key_value_heads=int(data["num_key_value_heads"]),
            head_dim=int(data.get("head_dim", 128)),
            max_position_embeddings=int(data["max_position_embeddings"]),
            rms_norm_eps=float(data["rms_norm_eps"]),
            rope_theta=float(data["rope_theta"]),
            sliding_window=int(data["sliding_window"]),
            num_experts=int(data["num_experts"]),
            num_experts_per_tok=int(data["num_experts_per_tok"]),
            num_shared_experts=int(data["num_shared_experts"]),
            score_func=str(data["score_func"]),
            route_norm=bool(data["route_norm"]),
            route_scale=float(data["route_scale"]),
            hidden_act=str(data["hidden_act"]),
            mup_enabled=bool(data.get("mup_enabled", False)),
            tie_word_embeddings=bool(data.get("tie_word_embeddings", False)),
            layer_types=layer_types,
        )

    @property
    def num_heads_per_kv(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def is_sliding_attention(self, layer_idx: int) -> bool:
        return self.layer_types[int(layer_idx)] == "sliding_attention"

    def full_attention_indices(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == "full_attention"]

    def sliding_attention_indices(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == "sliding_attention"]


class RoPECache:
    def __init__(self, cfg: TrinityNanoConfig):
        self.cfg = cfg
        self.cos = None
        self.sin = None
        self._rope_len = 0

    def extend(self, seq_len: int) -> None:
        seq_len = int(seq_len)
        if seq_len <= self._rope_len:
            return
        cfg = self.cfg
        inv = 1.0 / (
            cfg.rope_theta
            ** (np.arange(0, cfg.head_dim, 2, dtype=np.float32) / cfg.head_dim)
        )
        pos = np.arange(seq_len, dtype=np.float32)[:, None] * inv[None, :]
        emb = np.concatenate([pos, pos], axis=-1)
        self.cos = _cast(tc.tensor(np.cos(emb).astype(np.float32)))
        self.sin = _cast(tc.tensor(np.sin(emb).astype(np.float32)))
        self._rope_len = seq_len


def _linear_from_weight(weight: np.ndarray, weight_mode: str):
    if weight_mode == "bf16":
        return LinearTC(weight)
    if weight_mode == "int4":
        return QuantLinearTC(weight, GROUP_SIZE)
    raise ValueError(f"unsupported weight_mode {weight_mode!r}")


class SwiGLUTC:
    FFN_CHUNK = 2048

    def __init__(self):
        self.gate_proj = None
        self.up_proj = None
        self.down_proj = None

    def __call__(self, x):
        if x.shape[1] <= self.FFN_CHUNK:
            return self.down_proj(self.gate_proj(x).silu() * self.up_proj(x))
        outs = []
        for start in range(0, int(x.shape[1]), self.FFN_CHUNK):
            size = min(self.FFN_CHUNK, int(x.shape[1]) - start)
            xs = x.slice(1, start, size)
            outs.append(self.down_proj(self.gate_proj(xs).silu() * self.up_proj(xs)))
        return tc.cat(outs, dim=1)


class TrinityMoETC:
    def __init__(self, cfg: TrinityNanoConfig):
        self.cfg = cfg
        self.router_gate = None
        self.expert_bias = None
        self.shared_experts: SwiGLUTC | None = None
        self.experts: list[SwiGLUTC] = []
        self.route_detail = "summary"
        self.empty_cache_interval = 1

    def _route(self, x_flat):
        if self.cfg.score_func != "sigmoid":
            raise NotImplementedError("TrinityNano_TC currently implements sigmoid routing")
        scores = self.router_gate(x_flat)
        return _sigmoid_topk_route(
            scores,
            self.expert_bias,
            self.cfg.num_experts_per_tok,
            self.cfg.route_norm,
            self.cfg.route_scale,
        )

    def __call__(self, x):
        B, L, H = x.shape
        x_flat = x.reshape([B * L, H])
        topw, topi = self._route(x_flat)
        topw_np = topw.float().numpy().astype(np.float32)
        topi_np = topi.numpy().astype(np.int64)

        shared = self.shared_experts(x) if self.shared_experts is not None else None
        routed_rows = []
        calls = 0
        empty_cache_interval = int(self.empty_cache_interval)
        for row in range(B * L):
            xt = x_flat.slice(0, row, 1)
            acc = None
            slot_order = np.argsort(topi_np[row], kind="stable")
            for slot in slot_order.tolist():
                expert_id = int(topi_np[row, slot])
                weight = float(topw_np[row, slot])
                expert_out = self.experts[expert_id](xt.reshape([1, 1, H])).reshape([1, H])
                expert_out = _cast(expert_out.float() * weight)
                acc = expert_out if acc is None else acc + expert_out
                calls += 1
                del expert_out
                if (
                    empty_cache_interval > 0
                    and hasattr(tc, "empty_cache")
                    and calls % empty_cache_interval == 0
                ):
                    tc.empty_cache()
            routed_rows.append(acc)
        routed = tc.cat(routed_rows, dim=0).reshape([B, L, H])
        out = routed if shared is None else shared + routed

        route_info = {
            "route_detail": self.route_detail,
            "token_count": int(B * L),
            "num_experts_per_tok": int(self.cfg.num_experts_per_tok),
            "unique_experts": sorted({int(x) for x in topi_np.reshape(-1)}),
            "empty_cache_interval": int(empty_cache_interval),
        }
        if self.route_detail == "summary":
            route_info["slot_weight_means"] = [
                float(topw_np[:, slot].mean())
                for slot in range(self.cfg.num_experts_per_tok)
            ]
            route_info["expert_histogram"] = {
                str(i): int(v)
                for i, v in enumerate(
                    np.bincount(topi_np.reshape(-1), minlength=self.cfg.num_experts)
                )
                if int(v)
            }
            return out, route_info
        if self.route_detail != "full":
            raise ValueError(f"unsupported route_detail {self.route_detail!r}")
        route_info["top_indices"] = topi_np.tolist()
        route_info["top_weights"] = topw_np.tolist()
        return out, route_info


class TrinityAttentionTC:
    def __init__(self, cfg: TrinityNanoConfig, layer_idx: int):
        self.cfg = cfg
        self.layer_idx = int(layer_idx)
        self.is_local_attention = cfg.is_sliding_attention(layer_idx)
        self.sliding_window = cfg.sliding_window if self.is_local_attention else None
        self.num_heads = cfg.num_attention_heads
        self.num_key_value_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.num_key_value_groups = cfg.num_heads_per_kv
        self.scaling = self.head_dim ** -0.5
        self.q_proj = self.k_proj = self.v_proj = self.o_proj = None
        self.gate_proj = None
        self.q_norm = RMSNormTC(cfg.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNormTC(cfg.head_dim, cfg.rms_norm_eps)
        self._capture = False
        self._captured = None
        self.last_attention_backend = None

    def __call__(self, x, cos, sin, position_offset: int = 0, kv_cache=None):
        B, L, _ = x.shape
        H, KV, D = self.num_heads, self.num_key_value_heads, self.head_dim
        q = self.q_proj(x).reshape([B, L, H, D])
        k = self.k_proj(x).reshape([B, L, KV, D])
        v = self.v_proj(x).reshape([B, L, KV, D])
        gate = self.gate_proj(x)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q = _cast(q).transpose(1, 2)
        k = _cast(k).transpose(1, 2)
        v = _cast(v).transpose(1, 2)

        if self._capture:
            self._captured = (k.numpy(), v.numpy())

        if self.is_local_attention:
            cseg = cos.slice(0, int(position_offset), int(L))
            sseg = sin.slice(0, int(position_offset), int(L))
            if hasattr(tc, "rope_apply") and not tc.is_grad_enabled():
                q = tc.rope_apply(q, cos, sin, int(position_offset))
                k = tc.rope_apply(k, cos, sin, int(position_offset))
            else:
                q = F.apply_rotary(q, cseg, sseg)
                k = F.apply_rotary(k, cseg, sseg)

        if kv_cache is not None:
            k = tc.cat([kv_cache[0], k], dim=2)
            v = tc.cat([kv_cache[1], v], dim=2)
        S = int(k.shape[2])
        k_rep = _repeat_kv(k, self.num_key_value_groups)
        v_rep = _repeat_kv(v, self.num_key_value_groups)
        if self.is_local_attention and S > int(self.sliding_window):
            attn_mask = _band_mask(
                int(L), S, int(self.sliding_window), q.device.split(":")[0], q.dtype
            )
            attn = _trinity_scaled_attention(
                q, k_rep, v_rep, attn_mask=attn_mask, is_causal=False, scale=self.scaling
            )
            self.last_attention_backend = "standard_sliding_band"
        else:
            attn = _trinity_scaled_attention(
                q,
                k_rep,
                v_rep,
                is_causal=(L > 1 or kv_cache is not None),
                scale=self.scaling,
            )
            self.last_attention_backend = (
                "standard_sliding_causal" if self.is_local_attention else "standard_full_nope"
            )

        keep = S
        if self.is_local_attention:
            keep = min(S, int(self.sliding_window))
        new_k = k if keep == S else k.slice(2, S - keep, keep)
        new_v = v if keep == S else v.slice(2, S - keep, keep)

        attn = attn.transpose(1, 2).reshape([B, L, H * D])
        gated = attn * gate.sigmoid()
        out = self.o_proj(_cast(gated))
        return out, (new_k, new_v)


class TrinityBlockTC:
    def __init__(self, cfg: TrinityNanoConfig, layer_idx: int):
        self.cfg = cfg
        self.layer_idx = int(layer_idx)
        e = cfg.rms_norm_eps
        self.input_layernorm = RMSNormTC(cfg.hidden_size, e)
        self.post_attention_layernorm = RMSNormTC(cfg.hidden_size, e)
        self.pre_mlp_layernorm = RMSNormTC(cfg.hidden_size, e)
        self.post_mlp_layernorm = RMSNormTC(cfg.hidden_size, e)
        self.self_attn = TrinityAttentionTC(cfg, layer_idx)
        self.moe_enabled = self.layer_idx >= cfg.num_dense_layers
        self.mlp = TrinityMoETC(cfg) if self.moe_enabled else SwiGLUTC()

    @classmethod
    def from_safetensors(
        cls,
        cfg: TrinityNanoConfig,
        source: SafeTensorSource,
        layer_idx: int,
        *,
        weight_mode: str = "int4",
    ) -> "TrinityBlockTC":
        obj = cls(cfg, layer_idx)
        p = f"model.layers.{layer_idx}"
        obj.input_layernorm.weight = tc.tensor(
            source.get_np(f"{p}.input_layernorm.weight"), dtype="float32"
        )
        obj.post_attention_layernorm.weight = tc.tensor(
            source.get_np(f"{p}.post_attention_layernorm.weight"), dtype="float32"
        )
        obj.pre_mlp_layernorm.weight = tc.tensor(
            source.get_np(f"{p}.pre_mlp_layernorm.weight"), dtype="float32"
        )
        obj.post_mlp_layernorm.weight = tc.tensor(
            source.get_np(f"{p}.post_mlp_layernorm.weight"), dtype="float32"
        )
        att = obj.self_attn
        ap = f"{p}.self_attn"
        for attr in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"):
            QuantLinearTC.LOAD_CONTEXT = f"{ap}.{attr}.weight"
            setattr(
                att,
                attr,
                _linear_from_weight(source.get_np(f"{ap}.{attr}.weight"), weight_mode),
            )
        QuantLinearTC.LOAD_CONTEXT = None
        att.q_norm.weight = tc.tensor(source.get_np(f"{ap}.q_norm.weight"), dtype="float32")
        att.k_norm.weight = tc.tensor(source.get_np(f"{ap}.k_norm.weight"), dtype="float32")

        mp = f"{p}.mlp"
        if obj.moe_enabled:
            moe: TrinityMoETC = obj.mlp
            moe.router_gate = LinearTC(source.get_np(f"{mp}.router.gate.weight"))
            moe.expert_bias = tc.tensor(source.get_np(f"{mp}.expert_bias"), dtype="float32")
            if cfg.num_shared_experts > 0:
                moe.shared_experts = SwiGLUTC()
                for attr in ("gate_proj", "up_proj", "down_proj"):
                    QuantLinearTC.LOAD_CONTEXT = f"{mp}.shared_experts.{attr}.weight"
                    setattr(
                        moe.shared_experts,
                        attr,
                        _linear_from_weight(
                            source.get_np(f"{mp}.shared_experts.{attr}.weight"),
                            weight_mode,
                        ),
                    )
            moe.experts = []
            for expert_idx in range(cfg.num_experts):
                ex = SwiGLUTC()
                ep = f"{mp}.experts.{expert_idx}"
                for attr in ("gate_proj", "up_proj", "down_proj"):
                    QuantLinearTC.LOAD_CONTEXT = f"{ep}.{attr}.weight"
                    setattr(
                        ex,
                        attr,
                        _linear_from_weight(source.get_np(f"{ep}.{attr}.weight"), weight_mode),
                    )
                moe.experts.append(ex)
            QuantLinearTC.LOAD_CONTEXT = None
        else:
            dense: SwiGLUTC = obj.mlp
            for attr in ("gate_proj", "up_proj", "down_proj"):
                QuantLinearTC.LOAD_CONTEXT = f"{mp}.{attr}.weight"
                setattr(
                    dense,
                    attr,
                    _linear_from_weight(source.get_np(f"{mp}.{attr}.weight"), weight_mode),
                )
            QuantLinearTC.LOAD_CONTEXT = None
        return obj

    def __call__(self, x, cos, sin, position_offset: int = 0, kv_cache=None):
        residual = x
        attn_in = _cast(self.input_layernorm(x))
        attn, kv = self.self_attn(attn_in, cos, sin, position_offset, kv_cache)
        h = residual + _cast(self.post_attention_layernorm(attn))

        residual = h
        mlp_in = _cast(self.pre_mlp_layernorm(h))
        if self.moe_enabled:
            mlp_out, route_info = self.mlp(mlp_in)
        else:
            mlp_out = self.mlp(mlp_in)
            route_info = {"dense_mlp": True, "layer": int(self.layer_idx)}
        h = residual + _cast(self.post_mlp_layernorm(mlp_out))
        return h, kv, route_info


class TrinityNano_TC:
    PREFILL_CHUNK = 512

    def __init__(self, cfg: TrinityNanoConfig | None = None, *, max_layers: int | None = None):
        self.config = cfg or TrinityNanoConfig()
        self.max_layers = max_layers
        n_layers = self.config.num_hidden_layers if max_layers is None else int(max_layers)
        self.embed_tokens = HostEmbedding()
        self.layers = [TrinityBlockTC(self.config, i) for i in range(n_layers)]
        self.norm = RMSNormTC(self.config.hidden_size, self.config.rms_norm_eps)
        self.lm_head = None
        self.rope = RoPECache(self.config)
        self.rope.extend(min(4096, self.config.max_position_embeddings))

    def extend_rope(self, seq_len: int) -> None:
        self.rope.extend(int(seq_len))

    def __call__(
        self,
        input_ids_np,
        kv_caches=None,
        position_offset: int = 0,
        last_token_only: bool = False,
        max_layers: int | None = None,
        caches=None,
    ):
        if caches is not None:
            if kv_caches is not None:
                raise ValueError("pass either kv_caches or caches, not both")
            kv_caches = caches
        input_ids_np = np.asarray(input_ids_np, dtype=np.int64)
        B, L = input_ids_np.shape
        self.extend_rope(position_offset + L)
        h = self.embed_tokens(input_ids_np)
        if self.config.mup_enabled:
            h = h * math.sqrt(float(self.config.hidden_size))

        run_layers = self.layers if max_layers is None else self.layers[: int(max_layers)]
        new_caches = []
        for i, layer in enumerate(run_layers):
            cache = kv_caches[i] if kv_caches is not None else None
            h, kv, _route_info = layer(h, self.rope.cos, self.rope.sin, position_offset, cache)
            new_caches.append(kv)
            if kv_caches is not None:
                kv_caches[i] = None
        if max_layers is not None:
            return None, new_caches, h

        h = _cast(self.norm(h))
        if last_token_only and h.shape[1] > 1:
            h = h.slice(1, h.shape[1] - 1, 1)
        if self.lm_head is None:
            raise RuntimeError("lm_head is not loaded")
        if h.shape[1] > 8:
            parts = []
            for start in range(0, int(h.shape[1]), 8):
                size = min(8, int(h.shape[1]) - start)
                parts.append(self.lm_head(h.slice(1, start, size)))
            logits = tc.cat(parts, dim=1)
        else:
            logits = self.lm_head(h)
        return logits, new_caches

    def generate(self, prompt_ids, max_new_tokens: int = 16, caches=None):
        prompt_ids = np.asarray(prompt_ids, dtype=np.int64)
        generated: list[int] = []
        logits, caches = self(prompt_ids, kv_caches=caches, last_token_only=True)
        for step in range(int(max_new_tokens)):
            next_id = int(np.argmax(logits.float().numpy()[0, -1]))
            generated.append(next_id)
            next_ids = np.array([[next_id]], dtype=np.int64)
            logits, caches = self(
                next_ids,
                kv_caches=caches,
                position_offset=prompt_ids.shape[1] + step,
                last_token_only=True,
            )
        return np.asarray(generated, dtype=np.int64).reshape(1, -1), caches

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | os.PathLike[str],
        *,
        weight_mode: str = "int4",
        max_layers: int | None = None,
        load_lm_head: bool = True,
        progress: bool = True,
    ) -> tuple["TrinityNano_TC", dict[str, Any]]:
        if weight_mode not in {"int4", "bf16"}:
            raise ValueError("weight_mode must be 'int4' or 'bf16'")
        BlockTC.COMPUTE_DTYPE = "bfloat16"
        LinearTC.DTYPE = "bfloat16"
        QuantLinearTC.FUSED_DECODE = True
        RMSNormTC.USE_FUSED = False  # Afmoe cast-before-weight order
        cfg = TrinityNanoConfig.from_model_dir(model_dir)
        with tc.no_grad(), SafeTensorSource(model_dir) as source:
            model = cls(cfg, max_layers=max_layers)
            emb = source.get_np("model.embed_tokens.weight")
            if emb.shape != (cfg.vocab_size, cfg.hidden_size):
                raise ValueError(f"unexpected embed shape {emb.shape}")
            model.embed_tokens.weight = np.ascontiguousarray(emb.astype(np.float32, copy=False))
            del emb

            n_layers = len(model.layers)
            quant_bytes = 0
            plain_bytes = 0
            for i in range(n_layers):
                model.layers[i] = TrinityBlockTC.from_safetensors(
                    cfg, source, i, weight_mode=weight_mode
                )
                quant_bytes += _estimate_layer_linear_bytes(model.layers[i])
                plain_bytes += _estimate_layer_plain_bytes(model.layers[i])
                gc.collect()
                if hasattr(tc, "empty_cache"):
                    tc.empty_cache()
                if progress:
                    print(f"    Trinity layer {i + 1}/{n_layers} loaded", flush=True)

            model.norm.weight = tc.tensor(source.get_np("model.norm.weight"), dtype="float32")
            if load_lm_head:
                QuantLinearTC.LOAD_CONTEXT = "lm_head.weight"
                lm_w = source.get_np("lm_head.weight")
                model.lm_head = _linear_from_weight(lm_w, weight_mode)
                if hasattr(model.lm_head, "vram_bytes"):
                    quant_bytes += model.lm_head.vram_bytes()
                del lm_w
                QuantLinearTC.LOAD_CONTEXT = None
        info = {
            "framework": "tensor_cuda TrinityNano_TC",
            "model_dir": str(Path(model_dir).expanduser().resolve()),
            "weight_mode": weight_mode,
            "layers": len(model.layers),
            "hidden_size": cfg.hidden_size,
            "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": cfg.num_key_value_heads,
            "full_attention_layers": cfg.full_attention_indices(),
            "sliding_attention_layers": cfg.sliding_attention_indices(),
            "quantized_linear_bytes": int(quant_bytes),
            "plain_linear_bytes": int(plain_bytes),
            "embedding_host_bytes": int(model.embed_tokens.weight.nbytes),
            "lm_head_loaded": bool(load_lm_head),
        }
        return model, info


def _estimate_layer_linear_bytes(layer: TrinityBlockTC) -> int:
    total = 0
    for lin in (
        layer.self_attn.q_proj,
        layer.self_attn.k_proj,
        layer.self_attn.v_proj,
        layer.self_attn.o_proj,
        layer.self_attn.gate_proj,
    ):
        if hasattr(lin, "vram_bytes"):
            total += int(lin.vram_bytes())
    if layer.moe_enabled:
        moe: TrinityMoETC = layer.mlp
        if moe.shared_experts is not None:
            for lin in (moe.shared_experts.gate_proj, moe.shared_experts.up_proj, moe.shared_experts.down_proj):
                if hasattr(lin, "vram_bytes"):
                    total += int(lin.vram_bytes())
        for ex in moe.experts:
            for lin in (ex.gate_proj, ex.up_proj, ex.down_proj):
                if hasattr(lin, "vram_bytes"):
                    total += int(lin.vram_bytes())
    else:
        dense: SwiGLUTC = layer.mlp
        for lin in (dense.gate_proj, dense.up_proj, dense.down_proj):
            if hasattr(lin, "vram_bytes"):
                total += int(lin.vram_bytes())
    return total


def _estimate_layer_plain_bytes(layer: TrinityBlockTC) -> int:
    total = 0
    for lin in (
        layer.self_attn.q_proj,
        layer.self_attn.k_proj,
        layer.self_attn.v_proj,
        layer.self_attn.o_proj,
        layer.self_attn.gate_proj,
    ):
        if isinstance(lin, LinearTC):
            total += int(lin.vram_bytes())
    if layer.moe_enabled:
        moe: TrinityMoETC = layer.mlp
        if moe.shared_experts is not None:
            for lin in (moe.shared_experts.gate_proj, moe.shared_experts.up_proj, moe.shared_experts.down_proj):
                if isinstance(lin, LinearTC):
                    total += int(lin.vram_bytes())
        for ex in moe.experts:
            for lin in (ex.gate_proj, ex.up_proj, ex.down_proj):
                if isinstance(lin, LinearTC):
                    total += int(lin.vram_bytes())
    else:
        dense: SwiGLUTC = layer.mlp
        for lin in (dense.gate_proj, dense.up_proj, dense.down_proj):
            if isinstance(lin, LinearTC):
                total += int(lin.vram_bytes())
    return total
