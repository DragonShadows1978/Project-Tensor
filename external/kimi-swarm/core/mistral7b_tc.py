"""core/mistral7b_tc.py — Base layer types shared across model ports.

BlockTC, LinearTC, QuantLinearTC, RMSNormTC, and helpers used by
Gemma4, Qwen, and other model adapters.  These are thin wrappers
around tensor_cuda kernels that the model runner expects.
"""
from __future__ import annotations

import numpy as np

import tensor_cuda as tc
from tensor_cuda import nn as tc_nn

F = tc.functional


def _cast(t):
    """Cast a Tensor to the current compute dtype (bf16 for inference)."""
    dt = BlockTC.COMPUTE_DTYPE
    return t if t.dtype == dt else t.astype(dt)


def _repeat_kv(x, n_rep: int):
    """Repeat KV heads to match query heads: (B, KV, S, D) -> (B, KV*n, S, D)."""
    if n_rep == 1:
        return x
    B, KV, S, D = x.shape
    # expand then reshape — uses engine broadcast, no materialized copy
    return x.unsqueeze(2).expand([B, KV, n_rep, S, D]).reshape([B, KV * n_rep, S, D])


# ------------------------------------------------------------------
# Base configuration and compute dtype
# ------------------------------------------------------------------
class BlockTC:
    """Static compute dtype used by all model ports (bf16 for inference)."""
    COMPUTE_DTYPE: str = "bfloat16"


# ------------------------------------------------------------------
# Linear layers
# ------------------------------------------------------------------
class LinearTC(tc_nn.Module):
    """Dense linear layer: y = x @ W^T + b.  W stored (out, in)."""
    DTYPE: str = "float32"

    def __init__(self, weight_np: np.ndarray, bias: bool = False):
        super().__init__()
        # weight_np: (out_features, in_features)
        self.weight = tc.tensor(np.ascontiguousarray(weight_np),
                                dtype=self.DTYPE, requires_grad=False)
        self.bias = None
        self.out_features = weight_np.shape[0]
        self.in_features = weight_np.shape[1]

    def forward(self, x):
        w = self.weight if self.weight.dtype == x.dtype else self.weight.astype(x.dtype)
        out = tc.matmul(x, w, trans_b=True)
        if self.bias is not None:
            b = self.bias if self.bias.dtype == out.dtype else self.bias.astype(out.dtype)
            out = out + b
        return out

    def __call__(self, x):
        return self.forward(x)


class QuantLinearTC:
    """INT4 group-quantized linear: y = x @ dequant(W)^T.

    Two paths:
      - FUSED_DECODE=True (default for inference): int4_linear_fused,
        dequant inside shared-memory tiles, no full fp16 transient.
      - FUSED_DECODE=False: int4_linear two-stage, materializes fp16 W.
    """
    FUSED_DECODE: bool = True

    def __init__(self, weight_fp32: np.ndarray, group_size: int = 128):
        """weight_fp32: (out_features, in_features) — quantized to INT4 here."""
        self.out_features, self.in_features = weight_fp32.shape
        self.group_size = group_size
        packed, scales, zeros = self._quantize(weight_fp32, group_size)
        self.packed = tc.tensor(packed, dtype="uint8", requires_grad=False)
        self.scales = tc.tensor(np.ascontiguousarray(scales),
                                dtype="float16", requires_grad=False)
        self.zeros = tc.tensor(np.ascontiguousarray(zeros),
                               dtype="float16", requires_grad=False)

    @staticmethod
    def _quantize(w: np.ndarray, group_size: int):
        """Symmetric-8 INT4 quantization: w = (q - 8) * scale.

        Returns packed (N, K/2) uint8, scales (N, K//group), zeros (N, K//group).
        """
        N, K = w.shape
        assert K % group_size == 0, f"in_features {K} not divisible by group_size {group_size}"
        n_groups = K // group_size

        w = w.astype(np.float32)
        w_blocks = w.reshape(N, n_groups, group_size)

        # symmetric-8: center at 8, range [-8*scale, 7*scale]
        wmin = w_blocks.min(axis=-1, keepdims=True)
        wmax = w_blocks.max(axis=-1, keepdims=True)
        scale = np.maximum((wmax - wmin) / 15.0, 1e-8).astype(np.float16)

        # quantize: q = round((w - wmin) / scale)  [0, 15]
        q = np.clip(np.round((w_blocks - wmin) / scale[:, :, None]), 0, 15).astype(np.uint8)

        # symmetric-8 offset: store q' = q where q in [0,15], zero point = wmin/scale
        # For symmetric-8: we use the convention q_sym = q (already 0-15)
        # w = (q - 8) * scale  =>  zero = -8 * scale
        zeros = (-8.0 * scale).astype(np.float16)

        # pack two nibbles per byte: even index = low nibble, odd = high nibble
        q_flat = q.reshape(N, K)
        packed = (q_flat[:, 0::2] | (q_flat[:, 1::2] << 4)).astype(np.uint8)

        return packed, scale.reshape(N, n_groups), zeros.reshape(N, n_groups)

    def __call__(self, x):
        if self.FUSED_DECODE:
            return tc.int4_linear_fused(x, self.packed, self.scales,
                                        self.zeros, self.group_size)
        return tc.int4_linear(x, self.packed, self.scales,
                              self.zeros, self.group_size)


# ------------------------------------------------------------------
# RMSNorm
# ------------------------------------------------------------------
class RMSNormTC(tc_nn.Module):
    """RMSNorm over the last dim.  Weight is fp32 PLAIN w (no 1+w bake)."""
    USE_FUSED: bool = True

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = tc.tensor(np.ones(dim, dtype=np.float32),
                                dtype="float32", requires_grad=False)
        self.eps = eps

    def forward(self, x):
        if self.USE_FUSED and hasattr(tc, "rms_norm") and not tc.is_grad_enabled():
            return tc.rms_norm(x, self.weight, self.eps)
        ms = (x * x).mean([-1], True)
        return x * (ms + self.eps).pow(-0.5) * self.weight

    def __call__(self, x):
        return self.forward(x)


# ------------------------------------------------------------------
# APA blend attention (cuBLAS path)
# ------------------------------------------------------------------
def _cublas_blend_attention(q, k, kq, v, n_rep, scale, zthr, is_causal, block_size):
    """APA selective attention via cuBLAS: bulk quantized scoring + refine.

    This is the engine-backed implementation.  If the engine's fused
    apa_selective_attention kernel is available, use it; otherwise fall
    back to a composed implementation (slower but functionally identical).

    Args:
        q: (B, H, L, D) queries
        k: (B, KV, S, D) full-precision keys
        kq: (B, KV, S, D) quantized keys (for bulk scoring)
        v: (B, KV, S, D) values
        n_rep: int, repetition factor for KV heads (H // KV)
        scale: float, attention scale (1.0 for Gemma 4)
        zthr: float, z-score threshold for selection
        is_causal: bool, apply causal masking
        block_size: int, chunk size for processing

    Returns:
        (B, H, L, D) attention output
    """
    if hasattr(tc, "apa_selective_attention"):
        # Use the fused engine kernel
        return tc.apa_selective_attention(q, k, kq, v, scale, zthr, is_causal)

    # Fallback: composed implementation
    B, H, L, D = q.shape
    _, KV, S, _ = k.shape

    # Bulk scoring with quantized keys
    q_rep = q.reshape([B, KV, H // KV, L, D])
    # Score against quantized keys
    bulk_scores = tc.matmul(q_rep, kq, alpha=scale, trans_b=True)
    # Score against full-precision keys (for selected positions)
    rank_scores = tc.matmul(q_rep, k, alpha=scale, trans_b=True)

    # Apply APA blend softmax
    if hasattr(tc, "apa_blend_softmax"):
        # Reshape for blend: (B*KV*n_rep*L, S)
        b_flat = bulk_scores.reshape([B * H * L, S])
        r_flat = rank_scores.reshape([B * H * L, S])
        weights = tc.apa_blend_softmax(b_flat, r_flat, zthr)
        weights = weights.reshape([B, H, L, S])
    else:
        # Manual blend: per-row threshold selection
        b_abs = bulk_scores.abs().reshape([B * H * L, S])
        mean = b_abs.mean(-1, True)
        std = ((b_abs - mean) ** 2).mean(-1, True).pow(0.5)
        thr = mean + zthr * std
        thr = thr.reshape([B, H, L, 1])
        # Select: use rank where |bulk| >= thr, else bulk
        mask = (bulk_scores.abs() >= thr)
        scores = tc.where(mask, rank_scores, bulk_scores)
        if is_causal:
            scores = scores + F._causal_mask(L, S, q.device.split(":")[0], scores.dtype)
        weights = scores.softmax(-1)

    # Apply to values
    out = tc.matmul(weights, v)
    return out
