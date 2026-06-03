"""Thin Python wrapper around the APA-Quant CUDA extension.

Exposes a drop-in replacement for ``torch.nn.functional.scaled_dot_product_attention``
backed by C++/CUDA kernels, plus the full-signature ``apa_quant_attention`` that
mirrors ``tensor_gpu_v2._core.apa_quant_attention``.

The hot path (quantization, refinement, tiling, adaptive budget, autograd) lives
in C++; this layer only validates arguments, normalizes tensor shapes, and caches
the quantization tables on-device.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

from .quant_tables import build_tables

_EXT = None
_TABLE_CACHE: Dict[tuple, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _ext():
    """Lazily import the compiled extension with a helpful error message."""
    global _EXT
    if _EXT is None:
        try:
            import apa_attention_cuda as _C  # built by setup.py
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "The APA CUDA extension is not built. From apa_cuda/ run:\n"
                "    pip install -e .\n"
                "(requires a CUDA toolkit + PyTorch built with CUDA)."
            ) from exc
        _EXT = _C
    return _EXT


def _get_tables(head_dim: int, bits: int, num_heads: int, apa_rotation: bool,
                device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (head_dim, bits, num_heads, apa_rotation, str(device))
    cached = _TABLE_CACHE.get(key)
    if cached is not None:
        return cached
    rotations_np, codebook_np, boundaries_np = build_tables(
        head_dim, bits, num_heads, apa_rotation)
    rotations = torch.from_numpy(rotations_np).to(device=device, dtype=torch.float32)
    codebook = torch.from_numpy(codebook_np).to(device=device, dtype=torch.float32)
    boundaries = torch.from_numpy(boundaries_np).to(device=device, dtype=torch.float32)
    tables = (rotations, codebook, boundaries)
    _TABLE_CACHE[key] = tables
    return tables


class _APAQuantAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, rotations, codebook, boundaries,
                refine_percentile, is_causal, attn_mask, scale, dropout_p,
                block_size, adaptive_heads, training):
        output, key_quant, refine_mask = _ext().forward(
            query, key, value, rotations, codebook, boundaries,
            refine_percentile, is_causal, attn_mask, scale, dropout_p,
            int(block_size), adaptive_heads, training)

        ctx.save_for_backward(query, key, value, key_quant, refine_mask)
        ctx.attn_mask = attn_mask
        ctx.scale = scale
        ctx.is_causal = is_causal
        ctx.block_size = int(block_size)
        B, H, L, _ = query.shape
        S = key.shape[2]
        ctx.use_tiled = (B * H * L * S * 4) > (256 * 1024 * 1024)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        query, key, value, key_quant, refine_mask = ctx.saved_tensors
        grad_q, grad_k, grad_v = _ext().backward(
            grad_output.contiguous(), query, key, value, key_quant, refine_mask,
            ctx.scale, ctx.is_causal, ctx.attn_mask, ctx.use_tiled, ctx.block_size)
        # Grads only for query/key/value; the rest are non-differentiable.
        return (grad_q, grad_k, grad_v, None, None, None, None, None, None,
                None, None, None, None, None)


def apa_quant_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    bulk_bits: int = 2,
    refine_percentile: float = 0.15,
    is_causal: bool = False,
    attn_mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    dropout_p: float = 0.0,
    block_size: int = 64,
    adaptive_heads: bool = False,
    apa_rotation: bool = True,
) -> torch.Tensor:
    """APA-Quant attention (CUDA), matching the reference Python signature.

    Accepts ``(B, H, L, D)`` or ``(B, L, D)`` query/key/value and returns a
    tensor of the same rank as ``query``.
    """
    if bulk_bits not in (1, 2, 4, 8):
        raise ValueError(f"bulk_bits must be one of (1, 2, 4, 8), got {bulk_bits}")
    if not math.isfinite(refine_percentile):
        raise ValueError(f"refine_percentile must be finite, got {refine_percentile}")
    if not (0.0 <= dropout_p < 1.0):
        raise ValueError(f"dropout_p must be in [0, 1), got {dropout_p}")
    if scale is not None and (not math.isfinite(scale) or scale <= 0):
        raise ValueError(f"scale must be a positive finite number, got {scale}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    refine_percentile = max(0.0, min(1.0, refine_percentile))

    squeeze_output = query.dim() == 3
    if squeeze_output:
        B, L, D = query.shape
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
    elif query.dim() != 4:
        raise ValueError(f"expected 3D or 4D query, got {query.dim()}D")

    B, H, L, D = query.shape
    scale_f = float(scale) if scale is not None else (D ** -0.5)

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    rotations, codebook, boundaries = _get_tables(
        D, bulk_bits, H, apa_rotation, query.device)

    mask = attn_mask
    if mask is not None and not torch.is_tensor(mask):
        raise TypeError("attn_mask must be a torch.Tensor or None")
    if mask is not None:
        mask = mask.to(device=query.device, dtype=query.dtype)

    training = torch.is_grad_enabled()
    output = _APAQuantAttention.apply(
        query, key, value, rotations, codebook, boundaries,
        float(refine_percentile), bool(is_causal), mask, scale_f,
        float(dropout_p), int(block_size), bool(adaptive_heads), bool(training))

    if squeeze_output:
        output = output.squeeze(1)
    return output


def apa_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    *,
    bulk_bits: int = 2,
    refine_percentile: float = 0.15,
    adaptive_heads: bool = False,
    apa_rotation: bool = True,
    block_size: int = 64,
) -> torch.Tensor:
    """Positional-signature alias matching ``F.scaled_dot_product_attention``.

    Swap ``F.scaled_dot_product_attention(q, k, v, ...)`` for this to benchmark
    APA-Quant attention head-to-head with PyTorch's kernels.
    """
    return apa_quant_attention(
        query, key, value, bulk_bits=bulk_bits,
        refine_percentile=refine_percentile, is_causal=is_causal,
        attn_mask=attn_mask, scale=scale, dropout_p=dropout_p,
        block_size=block_size, adaptive_heads=adaptive_heads,
        apa_rotation=apa_rotation)
