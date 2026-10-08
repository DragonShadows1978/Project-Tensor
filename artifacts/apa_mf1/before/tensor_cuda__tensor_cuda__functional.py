"""Functional ops composed from the C++ engine (attention, etc.).

Most heavy compute (matmul, softmax) runs in C++ kernels; this module is the
thin orchestration layer. HY3D-sized unmasked non-causal inference can opt into
the native streaming SDPA primitive when it is safe to do so.
"""

from __future__ import annotations

import math
import os

import numpy as np

import tensor_cuda as tc

_causal_cache = {}


def _causal_mask(L, S, device, dtype):
    key = (L, S, device, dtype)
    m = _causal_cache.get(key)
    if m is None:
        # cap the cache: chunked long prefill creates one (L, S) entry
        # per chunk x context-length pair — at 24K context that is
        # ~500MB of cached masks that empty_cache() cannot reach
        if len(_causal_cache) >= 16:
            _causal_cache.clear()
        # bf16 has no numpy dtype — build fp32 then cast on device. -1e4 (not
        # -1e9) so it stays representable in fp16/bf16 without becoming -inf.
        # BOTTOM-RIGHT aligned (k = 1 + S - L): when S > L the L queries are the
        # LAST L positions of S keys (KV-cache decode, grafted/injected prefix),
        # so row i attends cols 0..(S-L)+i — prefix fully visible, causal among
        # the queries. Square (S == L) reduces to the standard k=1 mask. A plain
        # triu(k=1) on a rectangle is TOP-LEFT aligned and silently blinds the
        # queries to most of the prefix AND to each other (measured: predicted
        # ' briefing' from the first visible graft tokens instead of recall).
        bias = np.triu(np.full((L, S), -1e4, dtype=np.float32), k=1 + (S - L))
        m = tc.tensor(bias, device=device)
        if dtype in ("float16", "bfloat16"):
            m = m.astype(dtype)
        _causal_cache[key] = m
    return m


# Flag for the fused causal-softmax path in scaled_dot_product_attention
# (default off until ppl-gated; flip per-run or at merge).
USE_FUSED_SOFTMAX = False


def _fused_noncausal_sdpa_enabled():
    """Runtime opt-in for the inference-only streaming SDPA path.

    The environment is intentionally read for every call so a test or an
    operator can compare fused and composed paths in one process. Only the
    literal ``1`` enables it; absent or ``0`` retains the composed path.
    """
    # K4 receipt: composed is default (14.0 ms vs fused 77.4 ms at DiT); fused remains the 12 MiB vs 1236 MiB memory-lean opt-in.
    return os.environ.get("TC_FUSED_SDPA_NONCAUSAL", "0") == "1"


def scaled_dot_product_attention(query, key, value, attn_mask=None,
                                 is_causal=False, scale=None):
    """query/key/value: (B, H, L, D). Returns (B, H, L, D)."""
    D = query.shape[-1]
    L, S = query.shape[-2], key.shape[-2]
    scale = scale if scale is not None else 1.0 / math.sqrt(D)
    # HY3D DiT / VAE attention is non-causal, mask-free inference with D=64
    # or 128. Route it before the composed GEMM path so a [B,H,Lq,Lk] score
    # tensor is never allocated. Leave every training, masked, causal, bf16,
    # and larger-head case on the established composition for compatibility.
    if (not is_causal and attn_mask is None and not tc.is_grad_enabled()
            and _fused_noncausal_sdpa_enabled()
            and hasattr(tc, "fused_sdpa_noncausal")
            and query.dtype in ("float16", "float32")
            and key.dtype == query.dtype and value.dtype == query.dtype
            and 0 < D <= 128 and L >= 0 and S > 0):
        return tc.fused_sdpa_noncausal(query, key, value, scale)
    # OP_T GEMM: no materialized K^T copy; scale folded into the fp32
    # accumulator (one fewer 16-bit rounding + one fewer full pass over scores).
    scores = tc.matmul(query, key, alpha=scale, trans_b=True)
    if (USE_FUSED_SOFTMAX and is_causal and attn_mask is None and S >= L
            and not tc.is_grad_enabled()):
        # single-kernel bottom-right causal softmax: no mask tensor, masked
        # columns never read (inference-only; backward raises).
        return tc.matmul(tc.causal_softmax(scores), value)
    if is_causal:
        scores = scores + _causal_mask(L, S, query.device.split(":")[0], query.dtype)
    if attn_mask is not None:
        scores = scores + attn_mask
    # fused row softmax at inference: masks are already FOLDED INTO the
    # scores above (-1e4 entries exp to ~0 exactly as in the composed
    # path), so causal_softmax with L=1 rows == plain softmax(-1) in
    # one kernel. Measured 8x at sliding-window prefill shapes
    # (16,512,1535); the composed chain also paid the old pathological
    # trailing-axis reduce.
    if hasattr(tc, "causal_softmax") and not tc.is_grad_enabled():
        B_, H_ = scores.shape[0], scores.shape[1]
        weights = tc.causal_softmax(
            scores.reshape([B_, H_ * L, 1, S])).reshape([B_, H_, L, S])
    else:
        weights = scores.softmax(-1)
    return tc.matmul(weights, value)


def einsum(equation, *operands):
    """General einsum over engine ops (broadcast-multiply-sum).

    Supports single- and multi-operand contractions without repeated indices
    within a single term (no diagonals). Covers all standard attention/linear
    patterns. Differentiable end to end.
    """
    eq = equation.replace(" ", "")
    if "->" in eq:
        ins, out = eq.split("->")
    else:
        ins = eq
        from collections import Counter
        c = Counter(ins.replace(",", ""))
        out = "".join(sorted(k for k in c if c[k] == 1))
    terms = ins.split(",")
    assert len(terms) == len(operands), "einsum: term/operand count mismatch"

    all_idx = []
    for s in terms + [out]:
        for ch in s:
            if ch not in all_idx:
                all_idx.append(ch)

    factors = []
    for t, s in zip(operands, terms):
        order = [ch for ch in all_idx if ch in s]
        perm = [s.index(ch) for ch in order]
        tt = t.permute(perm) if perm != list(range(len(perm))) else t
        shape = [(t.shape[s.index(ch)] if ch in s else 1) for ch in all_idx]
        factors.append(tt.reshape(shape))

    prod = factors[0]
    for f in factors[1:]:
        prod = prod * f

    sum_axes = [i for i, ch in enumerate(all_idx) if ch not in out]
    if sum_axes:
        prod = prod.sum(sum_axes, False)
    remaining = [ch for ch in all_idx if ch in out]
    perm = [remaining.index(ch) for ch in out]
    if perm != list(range(len(perm))):
        prod = prod.permute(perm)
    return prod


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
