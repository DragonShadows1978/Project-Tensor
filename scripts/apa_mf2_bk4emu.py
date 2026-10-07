"""APA-MF2: independent numpy model of the BK4 selection rule (kernels.cu bk4::forward).

Two layers, both written from the kernel source / definition, never from MF1's or
the reviewer's helpers:

1. `rule_fp64(absb, vis)` -- the MATHEMATICAL rule in fp64 (the oracle):
   count_i = #visible keys of row i (S-L+i+1 causal, S non-causal);
   mean = sum|b|/count; var = max(sum|b|^2/count - mean^2, 0) (population, one-pass,
   clamped -- kernels.cu:8113-8114); th = mean + z*sqrt(var);
   sel = visible && |b| >= th (kernels.cu:8077).

2. `bk4_fp32(dot, vis, scale, z)` -- a bit-level TRANSCRIPTION of the kernel's fp32
   arithmetic as compiled (SASS of bk4::forward<false>, artifacts/apa_mf2/build):
     a      = |fl32(dot*scale)|                         FMUL then |.|
     sums  += a          (sequential j = 0..count-1)    FADD
     squares = fma(a, a, squares) (sequential)          FFMA R4,|R4|,|R4|,R21
     mean = sums / count; q = squares / count            IEEE div (MUFU.RCP+FCHK)
     var  = fma(-mean, mean, q); max(var, 0)            FFMA R3,-R32,R32,R3; FMNMX
     th   = fma(sqrt(var), z, mean)                     FFMA R3,R4,c[zthr],R32
   It is a diagnostic instrument (does BK4's arithmetic explain BK4's bits?), not
   the oracle.

fp32 emulation: +,-,*,/,sqrt computed in fp64 then rounded to fp32 are correctly
rounded (double rounding innocuous when p' >= 2p+2: 53 >= 50; Figueroa 1995).
fma is not covered by that theorem, so `fma32` adds an exact TwoSum tie repair.

Prior art: IEEE-754 fused multiply-add; Knuth TwoSum (TAOCP vol. 2, 1969);
Figueroa, "When is double rounding innocuous?" (SIGNUM 1995) -- unverified, lead to
check: "Figueroa double rounding innocuous 1995". Kernel rule: kernels.cu BK4
(Project-Tensor 2026). Transcription and harness: mine.
"""
from __future__ import annotations

import numpy as np

F32, F64 = np.float32, np.float64


def visible(L, S, causal=True):
    """(L, S) bool: row i sees j <= S-L+i (bottom-right, 63960a4 / Dao 2023)."""
    if not causal:
        return np.ones((L, S), bool)
    return np.arange(S)[None, :] <= (S - L + np.arange(L))[:, None]


def rule_fp64(absb, vis, z):
    """Mathematical BK4 rule in fp64. absb (..., L, S) >= 0; vis (L, S)."""
    a = np.where(vis, absb.astype(F64), 0.0)
    cnt = vis.sum(-1).astype(F64)[:, None]
    mean = a.sum(-1, keepdims=True) / cnt
    var = np.maximum((a * a).sum(-1, keepdims=True) / cnt - mean * mean, 0.0)
    th = mean + z * np.sqrt(var)
    return vis & (absb >= th), th


def fma32(a, b, c):
    """Correctly rounded fp32 fma(a, b, c) for fp32 arrays."""
    a, b, c = (np.asarray(x, F32) for x in (a, b, c))
    p = a.astype(F64) * b.astype(F64)          # exact (48 significant bits)
    cc = c.astype(F64)
    t = p + cc                                  # RN64
    bv = t - p
    err = (p - (t - bv)) + (cc - bv)            # TwoSum: t + err == p + c exactly
    r = t.astype(F32)
    d = t - r.astype(F64)
    toward = np.where(d > 0, F32(np.inf), F32(-np.inf))
    n = np.nextafter(r, toward)
    mid = (r.astype(F64) + n.astype(F64)) * 0.5
    fix = (d != 0) & (t == mid) & (err != 0) & (np.sign(err) == np.sign(d))
    return np.where(fix, n, r)


def bk4_fp32(dot, vis, scale, z):
    """Transcription of BK4's fp32 statistics. dot (R, L, S) fp32 raw dots."""
    dot = np.asarray(dot, F32)
    R, L, S = dot.shape
    a = np.abs(dot * F32(scale))                # FMUL (fp32 RN) then |.|
    a = np.where(vis[None], a, F32(0))          # invisible never summed (j < count)
    cnt = vis.sum(-1)                           # per row
    sums = np.zeros((R, L), F32)
    sq = np.zeros((R, L), F32)
    jmax = int(cnt.max())
    for j in range(jmax):
        live = (j < cnt)[None, :]
        aj = a[:, :, j]
        sums = np.where(live, sums + aj, sums)                 # FADD
        sq = np.where(live, fma32(aj, aj, sq), sq)             # FFMA
    c32 = cnt.astype(F32)[None, :]
    mean = (sums.astype(F64) / c32.astype(F64)).astype(F32)   # IEEE div
    q = (sq.astype(F64) / c32.astype(F64)).astype(F32)
    var = np.maximum(fma32(-mean, mean, q), F32(0))           # FFMA, FMNMX
    sd = np.sqrt(var.astype(F64)).astype(F32)                 # IEEE sqrt
    th = fma32(sd, np.full_like(sd, F32(z)), mean)            # FFMA
    sel = vis[None] & (np.abs(dot * F32(scale)) >= th[..., None])
    return sel, th


def bf16_bits_to_f32(bits):
    return (np.asarray(bits).astype(np.uint32) << 16).view(F32)


def to_bf16_f32(x):
    """Round fp32 -> bf16 (RNE) and return as fp32 values."""
    u = np.asarray(x, F32).view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
    return r.astype(np.uint32).view(F32)
