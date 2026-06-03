"""Functional ops composed from the C++ engine (attention, etc.).

All heavy compute (matmul, softmax) runs in C++ kernels; this module is the thin
orchestration layer. A single fused attention CUDA kernel is a later
optimization (see ROADMAP Phase 5b).
"""

from __future__ import annotations

import math

import numpy as np

import tensor_cuda as tc

_causal_cache = {}


def _causal_mask(L, S, device, dtype):
    key = (L, S, device, dtype)
    m = _causal_cache.get(key)
    if m is None:
        bias = np.triu(np.full((L, S), -1e9, dtype=np.float32), k=1)
        m = tc.tensor(bias, device=device, dtype=dtype)
        _causal_cache[key] = m
    return m


def scaled_dot_product_attention(query, key, value, attn_mask=None,
                                 is_causal=False, scale=None):
    """query/key/value: (B, H, L, D). Returns (B, H, L, D)."""
    D = query.shape[-1]
    L, S = query.shape[-2], key.shape[-2]
    scale = scale if scale is not None else 1.0 / math.sqrt(D)
    scores = tc.matmul(query, key.transpose(-2, -1)) * scale
    if is_causal:
        scores = scores + _causal_mask(L, S, query.device.split(":")[0], query.dtype)
    if attn_mask is not None:
        scores = scores + attn_mask
    weights = scores.softmax(-1)
    return tc.matmul(weights, value)


_rope_cache = {}


def rope_tables(seq_len, head_dim, device, base=10000.0):
    key = (seq_len, head_dim, device)
    t = _rope_cache.get(key)
    if t is None:
        inv = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
        pos = np.arange(seq_len, dtype=np.float32)[:, None] * inv[None, :]  # (L, D/2)
        emb = np.concatenate([pos, pos], axis=-1)                          # (L, D)
        t = (tc.tensor(np.cos(emb), device=device), tc.tensor(np.sin(emb), device=device))
        _rope_cache[key] = t
    return t


def apply_rotary(x, cos, sin):
    """RoPE on x (B, H, L, D). cos/sin are (L, D)."""
    D = x.shape[-1]
    half = D // 2
    x1 = x.slice(-1, 0, half)
    x2 = x.slice(-1, half, half)
    rot = tc.cat([-x2, x1], dim=-1)   # rotate_half
    return x * cos + rot * sin


def get_alibi_slopes(n_heads):
    import math

    def pow2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * (start ** i) for i in range(n)]

    if math.log2(n_heads).is_integer():
        return pow2(n_heads)
    closest = 2 ** math.floor(math.log2(n_heads))
    return pow2(closest) + get_alibi_slopes(2 * closest)[0::2][: n_heads - closest]


def build_alibi_bias(n_heads, seqlen, device="cuda"):
    """ALiBi additive bias (n_heads, seqlen, seqlen) = -slope * |i - j|."""
    slopes = np.array(get_alibi_slopes(n_heads), dtype=np.float32)
    ctx = np.arange(seqlen)
    dist = np.abs(ctx[None, :] - ctx[:, None]).astype(np.float32)  # (L, L)
    bias = -slopes[:, None, None] * dist[None, :, :]              # (H, L, L)
    return tc.tensor(bias, device=device)
