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
