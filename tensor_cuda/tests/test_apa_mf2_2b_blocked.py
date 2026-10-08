"""APA-MF2.2 Amendment 2 (item 4): on the COMPOSED softmax paths, a row whose every score is -inf
(all visible keys blocked, or every key masked by attn_mask) returns 0 -- never NaN -- in forward,
with finite (0) gradients in backward. Rows with any non--inf score (NaN included) are untouched.

Before (MF2 66287b1 .. Amendment 1): the -inf predicate mask makes such a row softmax(all -inf) =
0/0 = NaN; in a left-padded batch that NaN reaches the next layer's K/V and 0 x NaN poisons every
row (census: artifacts/apa_mf2_2a/census_table.md, 72 causal cells; fd32b6d's finite value there was
a future key's V through its finite -1e4 mask, F4). Fix (functional._guarded_softmax): blocked =
every entry -inf; softmax(scores.masked_fill(blocked, 0)) * (1 - blocked).
Prior art: FlashAttention-2 (Dao 2023) returns 0 for fully masked rows (Check_inf / lse guard);
PyTorch SDPA returns NaN there (contrast). Unverified -- lead to check ("flash attention fully masked
row zero output", "pytorch sdpa all masked row nan"). Tests and oracle: mine.
"""
import contextlib
import os
import sys

import numpy as np
import pytest

import tensor_cuda as tc
from tensor_cuda import functional as F
from tensor_cuda import quant as Q

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
import apa_mf2_2_oracle as O  # noqa: E402

DTS = ("float32", "float16", "bfloat16")
TOL = {"float32": 1e-5, "float16": 2e-2, "bfloat16": 2e-2}


def mk(a, dt, grad=False):
    t = tc.tensor(np.asarray(a, np.float32), dtype=dt)
    if grad:
        t = t.detach(); t.requires_grad = True          # a real leaf in every dtype
    return t


@contextlib.contextmanager
def fused_softmax(flag):
    old = F.USE_FUSED_SOFTMAX; F.USE_FUSED_SOFTMAX = flag
    try:
        yield
    finally:
        F.USE_FUSED_SOFTMAX = old


def overflow_ops(dt, Lq, S, D, blocked_keys, seed):
    """Finite operands; blocked keys score -inf by overflow, the others score 0 (K = 0)."""
    rng = np.random.default_rng(seed)
    q0, k0, scale = (2.0, -60000.0, 1e34) if dt == "float16" else (1e21, -1e18, 1.0)
    q = np.zeros((1, 2, Lq, D), np.float32); q[..., 0] = q0
    k = np.zeros((1, 2, S, D), np.float32); k[..., blocked_keys, 0] = k0
    v = rng.normal(0, 1, (1, 2, S, D)).astype(np.float32)
    return q, k, v, scale


def oracle(v, allowed):
    """allowed (Lq, S) bool: uniform softmax over allowed keys (all scores 0); rows with none -> 0."""
    v64 = v.astype(np.float64)
    w = allowed.astype(np.float64)
    cnt = w.sum(-1, keepdims=True)
    w = np.divide(w, cnt, out=np.zeros_like(w), where=cnt > 0)
    return w @ v64                                       # (..., Lq, D) via broadcast over (B, H)


def check(o, ref, blocked_rows, dt, label):
    o = o.float().numpy()
    assert np.isfinite(o).all(), f"{label}: non-finite output"
    assert np.all(o[..., blocked_rows, :] == 0.0), f"{label}: blocked rows not 0: {o[0, 0, blocked_rows, :3]}"
    err = float(np.abs(o - ref).max())
    assert err <= TOL[dt], f"{label}: max err {err:.3e}"


# ---- causal rows whose visible keys are all blocked (Amendment-1 G2 class) ----
@pytest.mark.parametrize("path", ["infer_composed", "infer_fused", "grad"])
@pytest.mark.parametrize("dt", DTS)
def test_sdpa_causal_rows_with_all_visible_keys_blocked_are_zero(dt, path):
    for S in (3, 33, 1025):
        for D in (64, 128):
            Lq = 3
            q, k, v, scale = overflow_ops(dt, Lq, S, D, np.arange(S - 1), S + D)
            vis = O.visible(Lq, S, True)
            allowed = vis & (np.arange(S) == S - 1)[None]            # only key S-1 is unblocked
            ref = oracle(mk(v, dt).float().numpy(), allowed)
            xs = [mk(x, dt, path == "grad") for x in (q, k, v)]
            with fused_softmax(path == "infer_fused"), contextlib.nullcontext() if path == "grad" else tc.no_grad():
                o = F.scaled_dot_product_attention(*xs, is_causal=True, scale=scale)
                if path == "grad":
                    o.sum().backward()
            check(o, ref, [0, 1], dt, f"{dt} {path} S={S} D={D}")
            if path == "grad":
                for n, x in zip("qkv", xs):
                    g = x.grad.float().numpy()
                    assert np.isfinite(g).all(), f"d{n} non-finite"
                assert np.all(xs[0].grad.float().numpy()[..., :2, :] == 0.0), "blocked rows' dq != 0"


@pytest.mark.parametrize("grad", [False, True])
@pytest.mark.parametrize("dt", DTS)
def test_sdpa_all_masked_rows_are_zero(dt, grad):
    Lq, S, D = 4, 33, 64
    rng = np.random.default_rng(11)
    q, k, v = (rng.normal(0, 0.5, (1, 2, n, D)).astype(np.float32) for n in (Lq, S, S))
    mask = np.zeros((1, 2, Lq, S), np.float32); mask[..., 1, :] = -np.inf; mask[..., 3, 5:] = -np.inf
    xs = [mk(x, dt, grad) for x in (q, k, v)]
    with contextlib.nullcontext() if grad else tc.no_grad():
        o = F.scaled_dot_product_attention(*xs, attn_mask=mk(mask, dt), scale=D ** -0.5)
        if grad:
            o.sum().backward()
    qe, ke, ve = (x.float().numpy().astype(np.float64) for x in xs)
    s = np.where(np.isfinite(mask), qe @ np.swapaxes(ke, -1, -2) * D ** -0.5, -np.inf)
    ref = np.zeros((1, 2, Lq, D))
    for r in (0, 2, 3):
        ref[..., r:r + 1, :] = O.softmax64(s[..., r:r + 1, :], ve)
    check(o, ref, [1], dt, f"{dt} masked grad={grad}")
    if grad:
        assert all(np.isfinite(x.grad.float().numpy()).all() for x in xs)
        assert np.all(xs[0].grad.float().numpy()[..., 1, :] == 0.0)


@pytest.mark.parametrize("dt", ["float32", "bfloat16"])
def test_sdpa_noncausal_composed_all_blocked_rows_are_zero(dt):
    S, D = 33, 64
    q, k, v, scale = overflow_ops(dt, 3, S, D, np.arange(S), 7)
    q[..., 2, 0] = 0.0                                             # row 2 not blocked (all scores 0)
    with tc.no_grad():
        o = F.scaled_dot_product_attention(*[mk(x, dt) for x in (q, k, v)], scale=scale)
    allowed = np.zeros((3, S), bool); allowed[2] = True
    check(o, oracle(mk(v, dt).float().numpy(), allowed), [0, 1], dt, f"{dt} noncausal")


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("p", [1.0, 0.15])
@pytest.mark.parametrize("dt", DTS)
def test_apa_quant_rows_with_all_scores_blocked_are_zero(dt, p, causal):
    S, D, Lq = 33, 64, 3
    q, k, v, scale = overflow_ops(dt, Lq, S, D, np.arange(S - 1), 5)
    with tc.no_grad():
        o = Q.apa_quant_attention(*[mk(x, dt) for x in (q, k, v)], bulk_bits=4, refine_percentile=p,
                                  is_causal=causal, scale=scale).float().numpy()
    assert np.isfinite(o).all(), (dt, p, causal)
    if causal:                                                     # rows 0,1 see only blocked keys
        assert np.all(o[..., :2, :] == 0.0), o[0, 0, :2, :3]


def test_apa_quant_backward_blocked_rows_finite():
    S, D, Lq = 33, 64, 3
    q, k, v, scale = overflow_ops("float32", Lq, S, D, np.arange(S - 1), 6)
    xs = [mk(x, "float32", True) for x in (q, k, v)]
    o = Q.apa_quant_attention(*xs, bulk_bits=4, refine_percentile=1.0, is_causal=True, scale=scale)
    o.sum().backward()
    assert np.all(o.numpy()[..., :2, :] == 0.0)
    assert all(np.isfinite(x.grad.numpy()).all() for x in xs)


# ---- G4: left-padded batch, padding keys masked for every query, causal ----
@pytest.mark.parametrize("path", ["infer_composed", "infer_fused", "grad"])
@pytest.mark.parametrize("kpad", [1, 3])
@pytest.mark.parametrize("dt", DTS)
def test_left_padded_batch(dt, kpad, path):
    for S in (8, 33, 256):
        for D in (64, 128):
            rng = np.random.default_rng(S + kpad + D)
            q, k, v = (rng.normal(0, 0.5, (2, 2, S, D)).astype(np.float32) for _ in range(3))
            mask = np.zeros((2, 1, S, S), np.float32); mask[..., :kpad] = -np.inf
            xs = [mk(x, dt, path == "grad") for x in (q, k, v)]
            m = mk(np.broadcast_to(mask, (2, 2, S, S)), dt)
            with fused_softmax(path == "infer_fused"), contextlib.nullcontext() if path == "grad" else tc.no_grad():
                o = F.scaled_dot_product_attention(*xs, attn_mask=m, is_causal=True, scale=D ** -0.5)
                if path == "grad":
                    o.sum().backward()
            qe, ke, ve = (x.float().numpy().astype(np.float64) for x in xs)
            allowed = O.visible(S, S, True) & (np.arange(S) >= kpad)[None]
            s = np.where(allowed, qe @ np.swapaxes(ke, -1, -2) * D ** -0.5, -np.inf)
            ref = np.zeros_like(qe)
            for r in range(kpad, S):
                ref[..., r:r + 1, :] = O.softmax64(s[..., r:r + 1, :], ve)
            check(o, ref, list(range(kpad)), dt, f"{dt} {path} k={kpad} S={S} D={D}")
            if path == "grad":
                assert all(np.isfinite(x.grad.float().numpy()).all() for x in xs)
                assert np.all(xs[0].grad.float().numpy()[..., :kpad, :] == 0.0)


@pytest.mark.parametrize("dt", DTS)
def test_left_padded_two_layer_chain_stays_finite(dt):
    """Layer-1 output (padded rows) becomes layer 2's K and V: a NaN there would poison every row."""
    S, D, kpad = 33, 64, 3
    rng = np.random.default_rng(9)
    x = mk(rng.normal(0, 0.5, (1, 2, S, D)), dt)
    m = mk(np.broadcast_to(np.where(np.arange(S) < kpad, -np.inf, 0.0).astype(np.float32), (1, 2, S, S)), dt)
    with tc.no_grad():
        h1 = F.scaled_dot_product_attention(x, x, x, attn_mask=m, is_causal=True)
        h2 = F.scaled_dot_product_attention(h1, h1, h1, is_causal=True)
    a1, a2 = h1.float().numpy(), h2.float().numpy()
    assert np.all(a1[..., :kpad, :] == 0.0) and np.isfinite(a1).all() and np.isfinite(a2).all()


def test_left_padded_apa_quant_overflow_padding():
    S, D, kpad = 33, 64, 3
    q, k, v, scale = overflow_ops("float32", S, S, D, np.arange(kpad), 4)
    q[..., 1:] = np.random.default_rng(1).normal(0, 0.3, (S, D - 1)); k[..., kpad:, 1:] = np.random.default_rng(2).normal(0, 0.3, (S - kpad, D - 1))
    for p in (1.0, 0.15):
        with tc.no_grad():
            o = Q.apa_quant_attention(*[mk(x, "float32") for x in (q, k, v)], bulk_bits=4, refine_percentile=p,
                                      is_causal=True, scale=scale).numpy()
        assert np.isfinite(o).all() and np.all(o[..., :kpad, :] == 0.0), p


# ---- untouched rows ----
@pytest.mark.parametrize("path", ["infer_composed", "infer_fused", "grad"])
def test_partially_blocked_rows_match_oracle(path):
    """Rows with SOME -inf scores are not 'blocked': normal softmax over the rest (kills 'any' detection)."""
    S, D, Lq = 33, 64, 3
    q, k, v, scale = overflow_ops("float32", Lq, S, D, np.arange(5), 8)
    xs = [mk(x, "float32", path == "grad") for x in (q, k, v)]
    with fused_softmax(path == "infer_fused"), contextlib.nullcontext() if path == "grad" else tc.no_grad():
        o = F.scaled_dot_product_attention(*xs, is_causal=True, scale=scale)
    allowed = O.visible(Lq, S, True) & (np.arange(S) >= 5)[None]
    check(o, oracle(v, allowed), [], "float32", path)


def test_nan_score_rows_are_not_zeroed():
    """Rows containing a NaN score are not blocked rows: they stay NaN (order: may stay NaN)."""
    S, D = 9, 64
    q = np.zeros((1, 1, 3, D), np.float32); q[..., 0] = 1
    k = np.zeros((1, 1, S, D), np.float32); k[0, 0, 0, 0] = np.nan
    v = np.ones_like(k)
    with tc.no_grad():
        o = F.scaled_dot_product_attention(*[mk(x, "float32") for x in (q, k, v)], scale=1.0).numpy()
    assert np.isnan(o).all()
