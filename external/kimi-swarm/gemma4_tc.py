"""Gemma 4 26B-A4B (MYTHOS) on tensor_cuda — INT4 weights, MQA/GQA attention, MoE.

Architecture (from GGUF ground truth):
  - 30 layers, hidden_dim=2816, vocab=262144
  - MoE: 128 experts, top-8 routing, fused 3D gate_up_exps/down_exps
  - Shared expert branch (dense GeGLU) + MoE expert branch
  - Dual attention: 25 sliding-window (head_dim=256, KV=8) +
    5 global (head_dim=512, KV=2)
  - V=K on global layers (no separate V projection)
  - Proportional RoPE with rope_freqs.weight as divisor
  - Tied embeddings (no separate output.weight)
  - GRM harvest/inject + APA KV quantization

Follows the tensor_cuda adapter pattern (see qwen3_tc.py, minicpm3_tc.py).
"""
from __future__ import annotations

import gc
import glob
import json
import os
from typing import List, Optional, Tuple

import numpy as np

from core.mistral7b_tc import (
    BlockTC, QuantLinearTC, RMSNormTC, _cublas_blend_attention, F, tc,
)
from core.qwen35_tc import HostEmbedding, _cast
from core.graft_arena import ArenaCache


# =============================================================================
# Configuration — all values verified against GGUF metadata
# =============================================================================
class Gemma4Config:
    """MYTHOS Gemma 4 26B-A4B — ground truth from GGUF metadata."""
    vocab_size = 262144
    hidden_dim = 2816
    num_layers = 30
    num_heads = 16
    # Attention: sliding layers use 8 KV heads, global use 2 KV heads
    num_kv_heads_sliding = 8      # 2048 / 256
    num_kv_heads_global = 2       # 1024 / 512
    head_dim_sliding = 256
    head_dim_global = 512
    sliding_window = 1024
    # RoPE
    rope_theta_global = 1000000.0
    rope_theta_swa = 10000.0
    p_rope_angles = 64            # partial RoPE dims for global layers
    rms_norm_eps = 1e-6
    logit_softcap = 30.0
    bos_token_id = 2
    eos_token_id = 1
    # MoE
    num_experts = 128
    num_experts_per_tok = 8       # top-8 routing
    expert_ffn_dim = 704          # per-expert intermediate
    shared_ffn_dim = 2112         # shared expert intermediate
    # Group size for INT4 quantization
    group_size = 128

    @staticmethod
    def is_global(i: int) -> bool:
        return i % 6 == 5           # layers 5, 11, 17, 23, 29

    @staticmethod
    def is_sliding(i: int) -> bool:
        return not Gemma4Config.is_global(i)


# =============================================================================
# Helpers
# =============================================================================

def _repeat_kv(x, n_rep: int):
    """Repeat KV heads to match query heads: (B, KV, S, D) -> (B, KV*n, S, D)."""
    if n_rep == 1:
        return x
    B, KV, S, D = x.shape
    return x.unsqueeze(2).expand([B, KV, n_rep, S, D]).reshape([B, KV * n_rep, S, D])


def _head_rmsnorm(x, w, eps, B, L, H, D):
    """RMSNorm per head: x (B, L, H*D) -> (B, L, H*D)."""
    if hasattr(tc, "rms_norm") and not tc.is_grad_enabled():
        wt = w if w is not None else _ones(D)
        return tc.rms_norm(x.reshape([B, L * H, D]), wt, eps).reshape([B, L, H * D])
    xf = x.reshape([B, L, H, D]).float()
    ms = (xf * xf).mean([-1], True)
    xf = xf * (ms + eps).pow(-0.5)
    if w is not None:
        xf = xf * w
    return _cast(xf)


def _zeros(*shape):
    """Allocate zeros in compute dtype."""
    cdt = BlockTC.COMPUTE_DTYPE
    return tc.zeros(*shape, dtype=cdt)


def _ones(D):
    """Cached ones for head norm fallback."""
    t = _ones_cache.get(D)
    if t is None:
        t = tc.tensor(np.ones(D, np.float32), dtype="float32")
        _ones_cache[D] = t
    return t


_ones_cache = {}


def _grow_cap(need: int) -> int:
    """Capacity growth policy."""
    blk = 2048
    if need <= blk:
        cap = 64
        while cap < need:
            cap *= 2
        return cap
    return ((need + blk - 1) // blk) * blk


def _zero_row():
    """Cached (1,1) zero for bias unmask writes."""
    cdt = BlockTC.COMPUTE_DTYPE
    t = _zero_cache.get(cdt)
    if t is None:
        t = _cast(tc.tensor(np.zeros((1, 1), np.float32)))
        _zero_cache[cdt] = t
    return t


_zero_cache = {}


# =============================================================================
# KV Ring (decode cache) — from existing gemma4_tc.py, proven
# =============================================================================
class KVRing:
    """Mutable decode cache with ring-buffer support.

    Sliding-window layers: ring_cap = 1024 (capped).
    Global layers: ring_cap = None (append-only growth).
    """
    __slots__ = ("kb", "vb", "bias", "count", "cap", "ring", "window",
                 "kqb", "kq_count")

    def __init__(self, k, v, ring_cap=None):
        B, KV, S, D = k.shape
        self.kqb = None
        self.kq_count = 0
        self.window = ring_cap
        self.ring = ring_cap is not None
        self.cap = _grow_cap(S + 1)
        if self.ring:
            self.cap = min(self.cap, ring_cap)
        self.kb = _zeros(B, KV, self.cap, D)
        self.vb = _zeros(B, KV, self.cap, D)
        bias = np.full((self.cap, 1), -1e4, np.float32)
        bias[:S] = 0.0
        self.bias = _cast(tc.tensor(bias))
        with tc.no_grad():
            tc.write_rows(self.kb, k, 0)
            tc.write_rows(self.vb, v, 0)
        self.count = S

    @property
    def full(self):
        return self.ring and self.cap == self.window and self.count >= self.cap

    def append(self, k1, v1, zero_row):
        at_cap = self.count == self.cap
        can_grow = (not self.ring) or self.cap < self.window
        if at_cap and can_grow:
            old_n = self.count
            B, KV, _, D = self.kb.shape
            self.cap = (min(_grow_cap(self.cap + 1), self.window)
                        if self.ring else _grow_cap(self.cap + 1))

            def _grow1(buf):
                nb = _zeros(B, KV, self.cap, D)
                with tc.no_grad():
                    tc.write_rows(nb, buf, 0)
                return nb

            self.kb = _grow1(self.kb)
            self.vb = _grow1(self.vb)
            if self.kqb is not None:
                self.kqb = _grow1(self.kqb)
            tc.empty_cache()
            bias = np.full((self.cap, 1), -1e4, np.float32)
            bias[:old_n] = 0.0
            self.bias = _cast(tc.tensor(bias))

        pos = (self.count % self.cap
               if self.ring and self.cap == self.window else self.count)
        with tc.no_grad():
            tc.write_rows(self.kb, k1, pos)
            tc.write_rows(self.vb, v1, pos)
            if not self.full:
                tc.write_rows(self.bias, zero_row, pos)
        self.count += 1

    def quantized_keys(self, quantize_fn):
        """Incremental APA quantized-key cache."""
        new = self.count - self.kq_count
        if new > 0:
            if self.kqb is None:
                B, KV, _, D = self.kb.shape
                self.kqb = _zeros(B, KV, self.cap, D)
            CHUNK = 512
            with tc.no_grad():
                for s0 in range(self.kq_count, self.count, CHUNK):
                    n = min(CHUNK, self.count - s0)
                    kq_s = quantize_fn(self.kb.slice(2, s0, n))
                    tc.write_rows(self.kqb, kq_s, s0)
                    if new > CHUNK:
                        tc.empty_cache()
            self.kq_count = self.count
        return self.kqb.slice(2, 0, self.count)

    def ordered(self):
        """Valid rows as (k, v) COPIES in logical order."""
        n = min(self.count, self.cap)
        if self.ring and self.count > self.cap:
            cut = self.count % self.cap
            if cut == 0:
                return (self.kb.slice(2, 0, self.cap),
                        self.vb.slice(2, 0, self.cap))
            k = tc.cat([self.kb.slice(2, cut, self.cap - cut),
                        self.kb.slice(2, 0, cut)], dim=2)
            v = tc.cat([self.vb.slice(2, cut, self.cap - cut),
                        self.vb.slice(2, 0, cut)], dim=2)
            return k, v
        return self.kb.slice(2, 0, n), self.vb.slice(2, 0, n)


# =============================================================================
# Band mask cache (sliding window)
# =============================================================================
_band_cache = {}


def _band_mask(L, S, window, device, dtype):
    key = (L, S, window, device, dtype)
    m = _band_cache.get(key)
    if m is None:
        i = np.arange(L, dtype=np.int64)[:, None] + (S - L)
        j = np.arange(S, dtype=np.int64)[None, :]
        vis = (j <= i) & (j > i - window)
        bias = np.where(vis, 0.0, -1e4).astype(np.float32)
        m = tc.tensor(np.ascontiguousarray(bias), device=device)
        if dtype in ("float16", "bfloat16"):
            m = m.astype(dtype)
        _band_cache[key] = m
    return m


# =============================================================================
# RoPE — proportional with rope_freqs.weight as divisor
# =============================================================================

class RoPECache:
    """Precomputed RoPE cos/sin tables with proportional scaling.

    Uses rope_freqs.weight from GGUF as divisor on computed inv_freq.
    The GGUF stores final inv_freq values, not raw theta/exponents.
    """

    def __init__(self, cfg: Gemma4Config):
        self.cfg = cfg
        self.freq_factors = None       # loaded from rope_freqs.weight
        self._rope_len = 0
        self.ropes = None              # (cos_sliding, sin_sliding, cos_global, sin_global)

    def load_freq_factors(self, arr: np.ndarray):
        """Load rope_freqs.weight (256,) as float32 divisor array."""
        self.freq_factors = arr.astype(np.float32)

    def extend(self, seq_len: int):
        """Extend RoPE tables to at least seq_len positions."""
        if seq_len <= self._rope_len:
            return
        cfg = self.cfg
        pos = np.arange(seq_len, dtype=np.float32)[:, None]

        # --- Sliding RoPE (head_dim=256, theta=10000) ---
        d_sl = cfg.head_dim_sliding
        # Compute base inv_freq
        inv_base_sl = 1.0 / (cfg.rope_theta_swa ** (
            np.arange(0, d_sl, 2, dtype=np.float32) / d_sl))
        # Apply freq_factors as divisor if available
        if self.freq_factors is not None:
            n_freqs = min(len(self.freq_factors), d_sl // 2)
            inv_sl = inv_base_sl.copy()
            inv_sl[:n_freqs] = inv_base_sl[:n_freqs] / self.freq_factors[:n_freqs]
            # Pad remaining with base values
            if n_freqs < d_sl // 2:
                inv_sl[n_freqs:] = inv_base_sl[n_freqs:]
        else:
            inv_sl = inv_base_sl
        emb_sl = np.concatenate([pos * inv_sl, pos * inv_sl], axis=-1)

        # --- Global RoPE (head_dim=512, theta=1000000) ---
        d_gl = cfg.head_dim_global
        inv_base_gl = 1.0 / (cfg.rope_theta_global ** (
            np.arange(0, d_gl, 2, dtype=np.float32) / d_gl))
        # Only first p_rope_angles dims get rotation; rest are zero
        # Per reality gate: "Proportional RoPE with model-specific freq_factors"
        if self.freq_factors is not None:
            n_freqs = min(len(self.freq_factors), cfg.p_rope_angles)
            inv_gl = inv_base_gl.copy()
            inv_gl[:n_freqs] = inv_base_gl[:n_freqs] / self.freq_factors[:n_freqs]
            # Zero out dims beyond p_rope_angles (partial RoPE)
            if cfg.p_rope_angles < d_gl // 2:
                inv_gl[cfg.p_rope_angles:] = 0.0
        else:
            inv_gl = inv_base_gl.copy()
            inv_gl[cfg.p_rope_angles:] = 0.0
        emb_gl = np.concatenate([pos * inv_gl, pos * inv_gl], axis=-1)

        self.ropes = (
            (_cast(tc.tensor(np.cos(emb_sl).astype(np.float32))),
             _cast(tc.tensor(np.sin(emb_sl).astype(np.float32)))),
            (_cast(tc.tensor(np.cos(emb_gl).astype(np.float32))),
             _cast(tc.tensor(np.sin(emb_gl).astype(np.float32)))),
        )
        self._rope_len = seq_len


# =============================================================================
# Attention — MQA/GQA dual sliding/global, V=K on global
# =============================================================================

class Gemma4AttentionTC:
    """Dual attention: sliding-window (local) + global (full context).

    Sliding layers: GQA 8 KV heads, head_dim=256, window=1024
    Global layers:  GQA 2 KV heads, head_dim=512, full context
    Global layers:  V=K (no separate V projection)

    GRM hooks: _capture (pre-RoPE K/V), _capture_q (pre-RoPE queries),
    inject_kv (graft prefix), graft_seats (position shift).
    """

    def __init__(self, cfg: Gemma4Config, layer_idx: int):
        self.cfg = cfg
        self.is_global = Gemma4Config.is_global(layer_idx)
        self.head_dim = cfg.head_dim_global if self.is_global else cfg.head_dim_sliding
        self.kv_heads = cfg.num_kv_heads_global if self.is_global else cfg.num_kv_heads_sliding
        self.num_heads = cfg.num_heads
        # Projections
        self.q_proj = None      # QuantLinearTC (out=H*D, in=hidden)
        self.k_proj = None      # QuantLinearTC (out=KV*D, in=hidden)
        self.v_proj = None      # QuantLinearTC — None on global layers (V=K)
        self.o_proj = None      # QuantLinearTC (out=hidden, in=H*D)
        # Head norms
        self.q_norm_w = None
        self.k_norm_w = None
        # APA dials
        self.attention_mode = "standard"
        self.refine_percentile = 0.15
        self.bulk_bits = 4
        self.attn_block = 1024
        self.apa_min_context = 2048
        # GRM hooks
        self.inject_kv = None
        self.graft_seats = 0
        self.live_shift = None
        # Capture hooks
        self._capture = False
        self._captured = None
        self._capture_q = False
        self._captured_q = None

    def __call__(self, x, cos, sin, position_offset=0, kv_cache=None):
        cfg = self.cfg
        B, L, _ = x.shape
        H, KV, D = cfg.num_heads, self.kv_heads, self.head_dim
        n_rep = H // KV

        # --- Projections ---
        q_lin = self.q_proj(x)          # (B, L, H*D)
        kraw = self.k_proj(x)           # (B, L, KV*D)

        # --- Head norms ---
        q = _head_rmsnorm(q_lin, self.q_norm_w, cfg.rms_norm_eps, B, L, H, D)
        q = q.reshape([B, L, H, D]).transpose(1, 2)   # (B, H, L, D)
        k = _head_rmsnorm(kraw, self.k_norm_w, cfg.rms_norm_eps, B, L, KV, D)
        k = k.reshape([B, L, KV, D]).transpose(1, 2)   # (B, KV, L, D)

        # --- RoPE ---
        if hasattr(tc, "rope_apply") and not tc.is_grad_enabled():
            q = tc.rope_apply(q, cos, sin, position_offset)
            k = tc.rope_apply(k, cos, sin, position_offset)
        else:
            cseg = cos.slice(0, position_offset, L)
            sseg = sin.slice(0, position_offset, L)
            q = F.apply_rotary(q, cseg, sseg)
            k = F.apply_rotary(k, cseg, sseg)

        # --- V projection ---
        if self.is_global:
            # Global layers: V = K (shared projection, no separate V)
            v = k
        else:
            # Sliding layers: separate V projection
            vsrc = self.v_proj(x) if self.v_proj is not None else kraw
            v = _head_rmsnorm(vsrc, None, cfg.rms_norm_eps, B, L, KV, D)
            v = v.reshape([B, L, KV, D]).transpose(1, 2)

        # --- GRM capture (pre-RoPE K/V for harvest) ---
        if self._capture:
            k_cap = kraw.reshape([B, L, KV, D]).transpose(1, 2).numpy()
            if self.is_global:
                v_cap = k_cap.copy()   # V=K on global
            else:
                vsrc_cap = (self.v_proj(x) if self.v_proj is not None else kraw)
                v_cap = vsrc_cap.reshape([B, L, KV, D]).transpose(1, 2).numpy()
            self._captured = (k_cap, v_cap)

        # --- GRM capture_q (pre-RoPE queries for router) ---
        if self._capture_q:
            # Capture post-norm, pre-RoPE queries
            q_cap = q_lin.reshape([B, L, H, D]).transpose(1, 2).numpy()
            self._captured_q = q_cap

        # --- Graft injection ---
        Sg = 0
        shift = 0
        if self.inject_kv is not None:
            if len(self.inject_kv) == 3:
                kg, vg, sc = self.inject_kv
                if sc != 1.0:
                    kg = kg * sc
            else:
                kg, vg = self.inject_kv[:2]
            k = tc.cat([kg, k], dim=2)
            v = tc.cat([vg, v], dim=2)
            Sg = int(kg.shape[2])

        shift = self.live_shift if self.live_shift is not None else Sg

        # --- DECODE path (L == 1): KVRing ---
        ring_cache = None
        if L == 1 and kv_cache is not None:
            if isinstance(kv_cache, tuple):
                ring_cap = None if self.is_global else cfg.sliding_window
                kv_cache = KVRing(kv_cache[0], kv_cache[1], ring_cap=ring_cap)

            kv_cache.append(k, v, _zero_row())
            S_all = min(kv_cache.count, kv_cache.cap)

            # APA selective attention (global layers only)
            apa_active = (self.is_global
                          and self.attention_mode == "apa_selective"
                          and S_all > self.apa_min_context)

            if apa_active:
                from tensor_cuda.quant import _norm_ppf, _quantize_keys, _tables
                dev = q.device.split(":")[0]
                R_t, C_t, B_t = _tables(D, self.bulk_bits, KV, True, dev)
                kq = kv_cache.quantized_keys(
                    lambda ks: _quantize_keys(ks, R_t, C_t, B_t))
                kk = kv_cache.kb.slice(2, 0, kv_cache.count)
                vv = kv_cache.vb.slice(2, 0, kv_cache.count)
                z_ = _norm_ppf(1.0 - max(0.0, min(1.0, self.refine_percentile)))
                blk = max(64, min(self.attn_block,
                                  int(300 * 1024 * 1024 // (H * S_all * 2))))
                attn = _cublas_blend_attention(
                    q, kk, kq, vv, n_rep, 1.0, float(z_), False, blk)
                attn = attn.transpose(1, 2).reshape([B, L, H * D])
                return self.o_proj(_cast(attn)), kv_cache
            else:
                # Standard decode attention via KVRing
                qg = q.reshape([B, KV, n_rep, D])
                sc = tc.matmul(qg, kv_cache.kb, alpha=1.0, trans_b=True)
                if not kv_cache.full:
                    sc = sc + kv_cache.bias.reshape([1, 1, 1, kv_cache.cap])
                if hasattr(tc, "causal_softmax"):
                    p = tc.causal_softmax(
                        sc.reshape([B, KV * n_rep, 1, kv_cache.cap]))
                    p = p.reshape([B, KV, n_rep, kv_cache.cap])
                else:
                    p = sc.softmax(-1)
                attn = tc.matmul(p, kv_cache.vb).reshape([B, L, H * D])
                return self.o_proj(_cast(attn)), kv_cache
        else:
            # Prefill path: concatenate with existing cache
            if isinstance(kv_cache, KVRing):
                kv_cache = kv_cache.ordered()
            if kv_cache is not None:
                k = tc.cat([kv_cache[0], k], dim=2)
                v = tc.cat([kv_cache[1], v], dim=2)

        S_all = k.shape[2]

        # --- PREFILL attention ---
        apa_active = (self.is_global
                      and self.attention_mode == "apa_selective"
                      and S_all > self.apa_min_context)

        if self.is_global:
            # Global: full context attention
            if apa_active:
                from tensor_cuda.quant import _norm_ppf, _quantize_keys, _tables
                dev = q.device.split(":")[0]
                R_t, C_t, B_t = _tables(D, self.bulk_bits, KV, True, dev)
                kq = _quantize_keys(k, R_t, C_t, B_t)
                z_ = _norm_ppf(1.0 - max(0.0, min(1.0, self.refine_percentile)))
                blk = max(64, min(self.attn_block,
                                  int(300 * 1024 * 1024 // (H * S_all * 2))))
                attn = _cublas_blend_attention(
                    q, k, kq, v, n_rep, 1.0, float(z_), L > 1, blk)
            else:
                qf = q.reshape([B, 1, H * L, D])
                sc = tc.matmul(qf, k, alpha=1.0, trans_b=True)
                sc = sc.reshape([B, H, L, S_all])
                if hasattr(tc, "causal_softmax") and not tc.is_grad_enabled():
                    p_ = tc.causal_softmax(sc)
                else:
                    p_ = (sc + F._causal_mask(L, S_all,
                                              q.device.split(":")[0], sc.dtype)).softmax(-1)
                attn = tc.matmul(p_.reshape([B, 1, H * L, S_all]),
                                 v).reshape([B, H, L, D])
            new_kv = ring_cache if ring_cache is not None else (k, v)
        else:
            # Sliding: windowed attention
            W = cfg.sliding_window
            if S_all <= W:
                attn = F.scaled_dot_product_attention(
                    q, _repeat_kv(k, n_rep), _repeat_kv(v, n_rep),
                    is_causal=(L > 1), scale=1.0)
            else:
                dev = q.device.split(":")[0]
                attn = F.scaled_dot_product_attention(
                    q, _repeat_kv(k, n_rep), _repeat_kv(v, n_rep),
                    attn_mask=_band_mask(L, S_all, W, dev, q.dtype),
                    scale=1.0)
            keep = min(W - 1, S_all)
            new_kv = ((k, v) if keep == S_all else
                      (k.slice(2, S_all - keep, keep),
                       v.slice(2, S_all - keep, keep)))

        attn = attn.transpose(1, 2).reshape([B, L, H * D])
        return self.o_proj(_cast(attn)), new_kv


# =============================================================================
# MoE Router
# =============================================================================

class MoERouter:
    """Top-k expert router for 128 experts, top-8 selection.

    GGUF layout (after reversal to [out, in]):
      ffn_gate_inp.weight: (128, 2816) — stored transposed from GGUF's (2816, 128)
      ffn_gate_inp.scale:  (2816,)      — input RMSNorm scale

    Flow: RMSNorm(x) * scale -> matmul(gate) -> softmax -> topk -> renorm
    """

    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 8):
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = None          # (128, 2816) — set during load
        self.gate_scale = None    # (2816,) — optional, from GGUF
        self.per_expert_scale = None  # (128,) — from ffn_down_exps.scale

    def __call__(self, x):
        """x: (B, L, hidden_dim) -> (B, L, k) weights, (B, L, k) indices."""
        # Apply gate_scale if present (input normalization from QAT)
        if self.gate_scale is not None:
            x = x * self.gate_scale

        # Gate logits: x (B, L, 2816) @ gate^T (2816, 128) = (B, L, 128)
        logits = tc.matmul(x, self.gate, trans_b=True)

        # Top-k via numpy (TensorCUDA may not have topk)
        logits_np = logits.float().numpy()
        B, L, _ = logits_np.shape
        k = min(self.top_k, self.num_experts)
        weights_np = np.zeros((B, L, k), dtype=np.float32)
        indices_np = np.zeros((B, L, k), dtype=np.int32)

        for b in range(B):
            for pos in range(L):
                row = logits_np[b, pos]
                top_idx = np.argpartition(row, -k)[-k:]
                top_vals = row[top_idx]
                top_vals -= np.max(top_vals)
                exp_vals = np.exp(top_vals)
                weights_np[b, pos] = exp_vals / exp_vals.sum()
                indices_np[b, pos] = top_idx

        weights = tc.tensor(np.ascontiguousarray(weights_np), dtype=x.dtype)
        indices = tc.tensor(np.ascontiguousarray(indices_np), dtype="int32")
        return weights, indices


# =============================================================================
# MoE GeGLU — Shared expert + Fused 3D routed experts
# =============================================================================

class MoEGeGLUTC:
    """Two-branch FFN: shared dense expert + MoE experts with fused 3D tensors.

    Fused expert tensors (GGUF shapes after reversal to PyTorch [out, in]):
      ffn_gate_up_exps: (128, 1408, 2816) — per-expert gate+up concatenated
      ffn_down_exps:    (128, 2816, 704)   — per-expert down proj
      ffn_down_exps.scale: (128,)           — per-expert down scale

    Splitting: expert e gets:
      gate_e = gate_up_exps[e, 0:704, :]      # (704, 2816)
      up_e   = gate_up_exps[e, 704:1408, :]   # (704, 2816)
      down_e = down_exps[e, :, :]             # (2816, 704)
    """

    def __init__(self, cfg: Gemma4Config):
        self.cfg = cfg
        self.router = MoERouter(cfg.hidden_dim, cfg.num_experts,
                                top_k=cfg.num_experts_per_tok)

        # Shared expert weights (dense, always runs)
        self.gate_shared = None   # QuantLinearTC (2112, 2816)
        self.up_shared = None     # QuantLinearTC (2112, 2816)
        self.down_shared = None   # QuantLinearTC (2816, 2112)

        # Fused MoE expert tensors — 3D, sliced per-expert at runtime
        self.gate_up_exps = None  # Tensor (128, 1408, 2816)
        self.down_exps = None     # Tensor (128, 2816, 704)
        self.down_exps_scale = None  # Tensor (128,)

        # Norms
        self.ffn_norm = None
        self.post_ffw_norm = None
        self.post_ffw_norm_1 = None
        self.pre_ffw_norm_2 = None
        self.post_ffw_norm_2 = None

    def _expert_forward(self, x, eid: int):
        """Run a single expert by slicing the fused 3D tensors.

        x: (B, L, 2816)
        eid: expert index 0..127
        """
        cfg = self.cfg

        # Slice expert eid from gate_up_exps: (128, 1408, 2816)
        # gate = first half [eid, 0:704, :], up = second half [eid, 704:1408, :]
        gate_w = self.gate_up_exps.slice(0, eid, 1).slice(1, 0, cfg.expert_ffn_dim)
        up_w = self.gate_up_exps.slice(0, eid, 1).slice(1, cfg.expert_ffn_dim,
                                                           cfg.expert_ffn_dim)
        # Reshape for matmul: remove expert dim -> (704, 2816)
        gate_w = gate_w.reshape([cfg.expert_ffn_dim, cfg.hidden_dim])
        up_w = up_w.reshape([cfg.expert_ffn_dim, cfg.hidden_dim])

        # Slice down_exps: (128, 2816, 704) -> (2816, 704)
        down_w = self.down_exps.slice(0, eid, 1).reshape([cfg.hidden_dim, cfg.expert_ffn_dim])

        # GeGLU: matmul with trans_b=True (weights stored [out, in])
        gate_out = tc.matmul(x, gate_w, trans_b=True).gelu()   # (B, L, 704)
        up_out = tc.matmul(x, up_w, trans_b=True)              # (B, L, 704)
        hidden = gate_out * up_out                              # (B, L, 704)

        # Apply per-expert down scale
        if self.down_exps_scale is not None:
            scale = self.down_exps_scale.slice(0, eid, 1)       # (1,)
            hidden = hidden * scale

        out = tc.matmul(hidden, down_w, trans_b=True)          # (B, L, 2816)
        return out

    def __call__(self, x):
        """Two-branch FFN forward.

        x: (B, L, hidden_dim)
        Returns: (B, L, hidden_dim)
        """
        cfg = self.cfg
        B, L, D = x.shape

        # Pre-FFN norm
        x_norm = self.ffn_norm(x) if self.ffn_norm is not None else x

        # ---- Shared expert branch (always runs) ----
        if self.gate_shared is not None:
            g = self.gate_shared(x_norm).gelu()
            u = self.up_shared(x_norm)
            shared_out = self.down_shared(g * u)
            shared_out = (self.post_ffw_norm_1(shared_out)
                          if self.post_ffw_norm_1 is not None else shared_out)
        else:
            shared_out = tc.zeros_like(x)

        # ---- MoE expert branch (top-8 routed) ----
        if self.gate_up_exps is not None:
            # Pre-norm for MoE path
            x_moe = (self.pre_ffw_norm_2(x_norm)
                     if self.pre_ffw_norm_2 is not None else x_norm)

            # Router scores
            weights, indices = self.router(x_moe)

            # Accumulate expert outputs
            moe_out = tc.zeros_like(x)
            idx_np = indices.numpy().astype(np.int32)
            w_np = weights.float().numpy()

            for ki in range(cfg.num_experts_per_tok):
                for eid in range(cfg.num_experts):
                    mask = (idx_np[:, :, ki] == eid)
                    if not mask.any():
                        continue

                    expert_out = self._expert_forward(x_moe, eid)
                    mask_t = tc.tensor(np.ascontiguousarray(mask.astype(np.float32)),
                                       dtype=x.dtype)
                    w_k = weights.slice(-1, ki, 1)    # (B, L, 1)
                    moe_out = moe_out + expert_out * mask_t * w_k

            moe_out = (self.post_ffw_norm_2(moe_out)
                       if self.post_ffw_norm_2 is not None else moe_out)
        else:
            moe_out = tc.zeros_like(x)

        # Combine and final norm
        out = shared_out + moe_out
        out = (self.post_ffw_norm(out)
               if self.post_ffw_norm is not None else out)
        return out


# =============================================================================
# Transformer Block
# =============================================================================

class Gemma4BlockTC:
    def __init__(self, cfg: Gemma4Config, layer_idx: int):
        self.is_global = Gemma4Config.is_global(layer_idx)
        e = cfg.rms_norm_eps
        self.input_layernorm = RMSNormTC(cfg.hidden_dim, e)
        self.post_attention_layernorm = RMSNormTC(cfg.hidden_dim, e)
        self.ffn_norm = RMSNormTC(cfg.hidden_dim, e)
        self.post_ffw_norm = RMSNormTC(cfg.hidden_dim, e)
        self.post_ffw_norm_1 = RMSNormTC(cfg.hidden_dim, e)
        self.pre_ffw_norm_2 = RMSNormTC(cfg.hidden_dim, e)
        self.post_ffw_norm_2 = RMSNormTC(cfg.hidden_dim, e)
        self.layer_scalar = 1.0
        self.mixer = Gemma4AttentionTC(cfg, layer_idx)
        self.mlp = MoEGeGLUTC(cfg)

    def __call__(self, x, ropes, position_offset=0, cache=None):
        cos, sin = ropes[1] if self.is_global else ropes[0]
        a, new_cache = self.mixer(_cast(self.input_layernorm(x)), cos, sin,
                                  position_offset, cache)
        h = x + _cast(self.post_attention_layernorm(a))
        mo = self.mlp(h)
        h = h + mo
        return h * self.layer_scalar, new_cache


# =============================================================================
# Full Model
# =============================================================================

class Gemma4_TC:
    """Gemma 4 26B-A4B (MYTHOS) — complete inference engine for tensor_cuda.

    Usage:
        model, info = Gemma4_TC.from_pretrained(
            "~/models/gemma-4-26b-a4b-it/",
            attention_mode="standard"
        )
        gen_ids, caches = model.generate(
            prompt_ids, max_new_tokens=128, temperature=0.7, top_p=0.9
        )
    """

    PREFILL_CHUNK = 512

    def __init__(self, cfg=None, attention_mode="standard", max_layers=None):
        self.config = cfg or Gemma4Config()
        cfg = self.config
        self.embed_tokens = HostEmbedding()
        n_layers = cfg.num_layers if max_layers is None else max_layers
        self.layers = [Gemma4BlockTC(cfg, i) for i in range(n_layers)]
        self.norm = RMSNormTC(cfg.hidden_dim, cfg.rms_norm_eps)
        self.lm_head = None
        self.rope = RoPECache(cfg)
        self.max_layers = max_layers
        self.set_attention_mode(attention_mode)

    def set_attention_mode(self, mode, refine_percentile=0.15, bulk_bits=4):
        for layer in self.layers:
            layer.mixer.attention_mode = mode
            layer.mixer.refine_percentile = refine_percentile
            layer.mixer.bulk_bits = bulk_bits

    def _get_ropes(self):
        """Return (sliding_ropes, global_ropes) where each is (cos, sin)."""
        return self.rope.ropes

    # ---- Forward pass ----------------------------------------------------

    def __call__(self, input_ids_np, kv_caches=None, position_offset=0,
                 last_token_only=False, max_layers=None):
        """Full forward: (B, L) token IDs -> (B, L, vocab) logits."""
        B, L = input_ids_np.shape
        C = self.PREFILL_CHUNK
        if max_layers is None and L > C:
            return self._chunked_forward(input_ids_np, kv_caches, position_offset,
                                         last_token_only)
        return self._forward(input_ids_np, kv_caches, position_offset,
                             last_token_only, max_layers)

    def _chunked_forward(self, input_ids_np, kv_caches, position_offset,
                         last_token_only):
        """Auto-chunked prefill for long sequences."""
        B, L = input_ids_np.shape
        C = self.PREFILL_CHUNK
        outs = []
        lg = None
        off = position_offset
        s0 = 0
        while s0 < L:
            S_ctx = off + C
            step = min(C, max(64, (32 * 1024 * 1024) // (32 * S_ctx)))
            step = max(64, (step // 64) * 64)
            seg = input_ids_np[:, s0:s0 + step]
            lg, kv_caches = self._forward(seg, kv_caches, off,
                                           last_token_only=last_token_only)
            off += seg.shape[1]
            s0 += seg.shape[1]
            if not last_token_only:
                outs.append(lg)
            tc.empty_cache()
        return (lg if last_token_only else tc.cat(outs, dim=1)), kv_caches

    def _forward(self, input_ids_np, kv_caches=None, position_offset=0,
                 last_token_only=False, max_layers=None):
        B, L = input_ids_np.shape
        self.rope.extend(position_offset + L)
        h = self.embed_tokens(input_ids_np)
        ropes = self._get_ropes()
        new_caches = []
        run = self.layers if max_layers is None else self.layers[:max_layers]
        for i, layer in enumerate(run):
            cache = kv_caches[i] if kv_caches is not None else None
            h, c = layer(h, ropes, position_offset, cache)
            new_caches.append(c)
            if kv_caches is not None:
                kv_caches[i] = None
        if max_layers is not None:
            return None, new_caches, h
        h = _cast(self.norm(h))
        if last_token_only and h.shape[1] > 1:
            h = h.slice(1, h.shape[1] - 1, 1)
        # lm_head in chunks to avoid OOM
        if h.shape[1] > 8:
            parts = []
            for s0 in range(0, h.shape[1], 8):
                seg = h.slice(1, s0, min(8, h.shape[1] - s0))
                parts.append(self.lm_head(seg))
            lg = tc.cat(parts, dim=1)
        else:
            lg = self.lm_head(h)
        c = self.config.logit_softcap
        logits = (lg.float() * (1.0 / c)).tanh() * c
        return logits, new_caches

    # ---- Generation ------------------------------------------------------

    def generate(self, prompt_ids, max_new_tokens=128, temperature=1.0,
                 top_p=1.0, top_k=0, stop_at_eos=True, caches=None):
        """Generate tokens autoregressively.

        Returns: (generated_ids, caches) — generated_ids shape (B, N).
        """
        cfg = self.config
        B = prompt_ids.shape[0]
        generated = []

        # Prefill
        logits, caches = self(prompt_ids, caches=caches, last_token_only=True)

        for step in range(max_new_tokens):
            next_token = self._sample_token(logits, temperature, top_p, top_k)
            tok_id = int(next_token[0, 0])
            generated.append(tok_id)

            if stop_at_eos and tok_id == cfg.eos_token_id:
                break

            next_ids = next_token
            pos = prompt_ids.shape[1] + step + 1
            logits, caches = self(next_ids, caches=caches, position_offset=pos,
                                  last_token_only=True)

        gen_np = np.array(generated, dtype=np.int64).reshape(1, -1)
        return gen_np, caches

    def _sample_token(self, logits, temperature, top_p, top_k):
        """Sample a single token from logits."""
        logits_np = logits.float().numpy()
        B = logits_np.shape[0]
        result = np.zeros((B, 1), dtype=np.int64)

        for b in range(B):
            row = logits_np[b, 0].copy()
            if temperature > 0 and temperature != 1.0:
                row /= temperature
            if top_k > 0:
                kth = np.partition(row, -top_k)[-top_k]
                row[row < kth] = -1e10
            if top_p < 1.0:
                sorted_logits = np.sort(row)[::-1]
                sorted_probs = np.exp(sorted_logits - np.max(sorted_logits))
                sorted_probs /= sorted_probs.sum()
                cumsum = np.cumsum(sorted_probs)
                cutoff_idx = np.searchsorted(cumsum, top_p) + 1
                cutoff = sorted_logits[cutoff_idx] if cutoff_idx < len(sorted_logits) else -1e10
                row[row < cutoff] = -1e10
            if temperature <= 0:
                result[b, 0] = int(np.argmax(row))
            else:
                probs = np.exp(row - np.max(row))
                probs /= probs.sum()
                result[b, 0] = int(np.random.choice(len(probs), p=probs))

        return result

    # ---- Weight loading: Safetensors ------------------------------------

    def _tensor_map(self, model_dir):
        """Build {tensor_name -> shard_path} mapping from index.json."""
        from safetensors import safe_open
        idx_path = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(idx_path):
            with open(idx_path) as f:
                idx = json.load(f)
            return {k: os.path.join(model_dir, v)
                    for k, v in idx["weight_map"].items()}
        # No index: scan all safetensors files
        shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
        if not shards:
            raise FileNotFoundError(f"No safetensors in {model_dir}")
        where = {}
        for shard in shards:
            with safe_open(shard, framework="pt") as f:
                for name in f.keys():
                    where[name] = shard
        return where

    def load_weights(self, model_dir, load_lm_head=True, progress=True):
        """Load weights from safetensors, quantizing to INT4 on load."""
        import torch
        from safetensors import safe_open

        where = self._tensor_map(model_dir)
        cfg = self.config
        group = cfg.group_size

        def gf(name):
            with safe_open(where[name], framework="pt") as f:
                return f.get_tensor(name).to(torch.float32).numpy()

        loaded = 0
        quant_bytes = 0
        orig_bytes = 0

        # ---- Embeddings (tied to output) ----
        emb = gf("model.embed_tokens.weight")
        # HostEmbedding expects [vocab_size, hidden_dim]
        if emb.shape[0] != cfg.vocab_size:
            emb = emb.T
        emb_scaled = emb * np.float32(cfg.hidden_dim ** 0.5)
        self.embed_tokens.weight = np.ascontiguousarray(emb_scaled)
        # lm_head tied to embeddings
        if load_lm_head:
            lm_w = emb if emb.shape == (cfg.vocab_size, cfg.hidden_dim) else emb.T
            self.lm_head = QuantLinearTC(np.ascontiguousarray(lm_w), group)
            quant_bytes += self.lm_head.vram_bytes()
        del emb, emb_scaled, lm_w
        gc.collect()
        loaded += 1

        # ---- RoPE freq_factors ----
        if "model.rotary_emb.inv_freq" in where:
            freq = gf("model.rotary_emb.inv_freq")
            self.rope.load_freq_factors(freq)
            loaded += 1
        elif "model.rope_freqs.weight" in where:
            freq = gf("model.rope_freqs.weight")
            self.rope.load_freq_factors(freq)
            loaded += 1

        # ---- Per-layer weights ----
        for i, layer in enumerate(self.layers):
            p = f"model.layers.{i}"
            Lr = layer
            mlp = Lr.mlp

            # Attention norms
            w = np.ascontiguousarray(gf(f"{p}.input_layernorm.weight"))
            Lr.input_layernorm.weight = tc.tensor(w, dtype="float32")
            w = np.ascontiguousarray(gf(f"{p}.post_attention_layernorm.weight"))
            Lr.post_attention_layernorm.weight = tc.tensor(w, dtype="float32")
            loaded += 2

            # Attention projections
            mx = Lr.mixer
            for attr, key in (
                ("q_proj", "self_attn.q_proj"),
                ("k_proj", "self_attn.k_proj"),
                ("o_proj", "self_attn.o_proj"),
            ):
                w = np.ascontiguousarray(gf(f"{p}.{key}.weight"))
                orig_bytes += w.nbytes
                ql = QuantLinearTC(w, group)
                setattr(mx, attr, ql)
                quant_bytes += ql.vram_bytes()
                loaded += 1
                del w

            # V projection (only on sliding layers)
            if not Lr.is_global:
                w = np.ascontiguousarray(gf(f"{p}.self_attn.v_proj.weight"))
                orig_bytes += w.nbytes
                mx.v_proj = QuantLinearTC(w, group)
                quant_bytes += mx.v_proj.vram_bytes()
                loaded += 1
                del w

            # Head norms
            w = np.ascontiguousarray(gf(f"{p}.self_attn.q_norm.weight"))
            mx.q_norm_w = tc.tensor(w, dtype="float32")
            w = np.ascontiguousarray(gf(f"{p}.self_attn.k_norm.weight"))
            mx.k_norm_w = tc.tensor(w, dtype="float32")
            loaded += 2

            # Layer output scale
            try:
                w = gf(f"{p}.layer_output_scale.weight")
                Lr.layer_scalar = float(w[0])
                loaded += 1
            except KeyError:
                pass

            # ---- FFN norms ----
            for norm_name, target_attr in [
                ("ffn_norm", "ffn_norm"),
                ("post_ffw_norm", "post_ffw_norm"),
                ("post_ffw_norm_1", "post_ffw_norm_1"),
                ("post_ffw_norm_2", "post_ffw_norm_2"),
                ("pre_ffw_norm_2", "pre_ffw_norm_2"),
            ]:
                try:
                    w = np.ascontiguousarray(gf(f"{p}.{norm_name}.weight"))
                    getattr(mlp, target_attr).weight = tc.tensor(w, dtype="float32")
                    loaded += 1
                except KeyError:
                    pass

            # ---- MoE Router ----
            try:
                w = np.ascontiguousarray(gf(f"{p}.ffn_gate_inp.weight"))
                # GGUF stores (in, out) = (2816, 128); we want (out, in) = (128, 2816)
                mlp.router.gate = tc.tensor(w.T, dtype="float32")
                loaded += 1
            except KeyError:
                pass

            try:
                w = gf(f"{p}.ffn_gate_inp.scale")
                mlp.router.gate_scale = tc.tensor(w, dtype="float32")
                loaded += 1
            except KeyError:
                pass

            # ---- Shared Expert ----
            for attr, key in (
                ("gate_shared", "ffn_gate"),
                ("up_shared", "ffn_up"),
                ("down_shared", "ffn_down"),
            ):
                try:
                    w = np.ascontiguousarray(gf(f"{p}.{key}.weight"))
                    orig_bytes += w.nbytes
                    ql = QuantLinearTC(w, group)
                    setattr(mlp, attr, ql)
                    quant_bytes += ql.vram_bytes()
                    loaded += 1
                    del w
                except KeyError:
                    pass

            # ---- Fused MoE Experts (3D tensors) ----
            try:
                # gate_up_exps: safetensors stores (128, 1408, 2816)
                w = gf(f"{p}.ffn_gate_up_exps.weight")
                mlp.gate_up_exps = tc.tensor(np.ascontiguousarray(w),
                                              dtype=BlockTC.COMPUTE_DTYPE)
                loaded += 1
            except KeyError:
                pass

            try:
                # down_exps: safetensors stores (128, 2816, 704)
                w = gf(f"{p}.ffn_down_exps.weight")
                mlp.down_exps = tc.tensor(np.ascontiguousarray(w),
                                          dtype=BlockTC.COMPUTE_DTYPE)
                loaded += 1
            except KeyError:
                pass

            try:
                w = gf(f"{p}.ffn_down_exps.scale")
                mlp.down_exps_scale = tc.tensor(w, dtype="float32")
                loaded += 1
            except KeyError:
                pass

            gc.collect()
            if progress:
                print(f"    layer {i + 1}/{len(self.layers)}", flush=True)

        # ---- Final norm ----
        w = np.ascontiguousarray(gf("model.norm.weight"))
        self.norm.weight = tc.tensor(w, dtype="float32")
        loaded += 1

        gc.collect()
        return {
            "loaded": loaded,
            "framework": f"tensor_cuda Gemma4-26B-A4B ({cfg.num_layers}L, {cfg.hidden_dim}H, {cfg.num_experts}E/{cfg.num_experts_per_tok}A)",
            "group_size": group,
            "original_bytes": orig_bytes,
            "quantized_bytes": quant_bytes,
            "compression_ratio": (orig_bytes / quant_bytes) if quant_bytes else 0.0,
        }

    @classmethod
    def from_pretrained(cls, model_dir=None, attention_mode="standard",
                        max_layers=None, load_lm_head=True, progress=True):
        BlockTC.COMPUTE_DTYPE = "bfloat16"
        QuantLinearTC.FUSED_DECODE = True
        RMSNormTC.USE_FUSED = True
        with tc.no_grad():
            model = cls(attention_mode=attention_mode, max_layers=max_layers)
            info = model.load_weights(model_dir, load_lm_head=load_lm_head,
                                      progress=progress)
        return model, info


# =============================================================================
# GRM ArenaCache dialect for Gemma 4 GQA
# =============================================================================

class Gemma4ArenaCache(ArenaCache):
    """GRM arena dialect for Gemma 4's GQA attention.

    Payload: per-layer (k, v) tensors — position-free for harvest,
    re-RoPE'd at arena seats on mount.
    """
    PAYLOAD = (("k", 2), ("v", 2))     # (key, seq_dim) pairs
    ROPE_KEYS = ("k",)                   # k needs RoPE re-seating
    ROPE_PAIR_SWAP = False
    VALS_PER_TOK_LAYER = 1024            # k 512 + v 512 (global) or k 256 + v 256 (sliding)
    # Average: global layers (1024) + sliding layers (512) blended

    def _set_inject(self, att, blk):
        att.inject_kv = (blk["k"], blk["v"])
        att.graft_seats = int(blk["k"].shape[2])

    def _set_injection_host(self, inj):
        from core import kv_graft
        kv_graft.set_injection(self.m, inj)


# =============================================================================
# KV cache save/load (for checkpointing / graft persistence)
# =============================================================================

def save_caches(caches, path, position_offset=0):
    """Save KVRing caches to disk as npz."""
    arrs = {"position_offset": np.array([position_offset], np.int64)}
    for i, c in enumerate(caches):
        if isinstance(c, tuple):
            arrs[f"l{i}_k"] = c[0].float().numpy()
            arrs[f"l{i}_v"] = c[1].float().numpy()
            arrs[f"l{i}_type"] = np.array([0], np.int64)
        else:
            k, v = c.ordered()
            arrs[f"l{i}_k"] = k.float().numpy()
            arrs[f"l{i}_v"] = v.float().numpy()
            arrs[f"l{i}_type"] = np.array([1 if c.ring else 0], np.int64)
            arrs[f"l{i}_count"] = np.array([c.count], np.int64)
    tmp = path + ".tmp"
    np.savez(tmp, **arrs)
    os.replace(tmp + ".npz", path)


def load_caches(path, cfg=None):
    """Load KVRing caches from disk."""
    cfg = cfg or Gemma4Config()
    z = np.load(path)
    caches = []
    for i in range(cfg.num_layers):
        k = _cast(tc.tensor(z[f"l{i}_k"]))
        v = _cast(tc.tensor(z[f"l{i}_v"]))
        t = int(z[f"l{i}_type"][0])
        if t == 0:
            caches.append((k, v))
        else:
            ring_cap = cfg.sliding_window if Gemma4Config.is_sliding(i) else None
            c = KVRing(k, v, ring_cap=ring_cap)
            c.count = int(z[f"l{i}_count"][0])
            caches.append(c)
    return caches, int(z["position_offset"][0])


# =============================================================================
# APA KV quantization helpers
# =============================================================================

def set_kv_int4(model, enabled=True, group=32):
    """Enable INT4 KV cache storage via APA's kv_int4_pack/unpack.

    When enabled, the global-layer KVRing quantizes new keys to INT4
    on append, reducing KV cache size by ~50%.
    """
    for layer in model.layers:
        layer.mixer.kv_int4_group = int(group) if enabled else 0


def get_kv_cache_size(model, caches):
    """Report total KV cache memory in bytes."""
    total = 0
    for c in caches:
        if isinstance(c, tuple):
            k, v = c
            total += k.nbytes + v.nbytes
        else:
            total += c.kb.nbytes + c.vb.nbytes
            if c.kqb is not None:
                total += c.kqb.nbytes
    return total


# =============================================================================
# Back-compat alias
# =============================================================================
Gemma4Runner = Gemma4_TC
