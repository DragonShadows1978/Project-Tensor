"""APA-LD2: staggered frontier cuts (small leaks) and the zero-weight statistics tag (ZWS) -- detector logic on
numpy kernels (CPU) and the real engine at small shapes (GPU). Author-run tests: a baseline, not blind
verification.

Prior art: OpenAI (2026) openai/math #107 Prop 3.1 / #135 via the TX2 Opus M4a dossier and APA-LD1 (power-of-two
tags, bitwise invariance oracle, cut predicate); Goldberg (1991) exactness of power-of-two scaling; Chen et al.
(1998) metamorphic testing; DeMillo, Lipton & Sayward (1978) mutation testing; Dao (2023) FlashAttention-2
bottom-right causal. Unverified -- lead to check. Mine: the staggered per-slab cut packing and the ZWS ghost-key
statistics tag under test (scripts/apa_ld2_pdwl.py).

APA-REGT1 production port (from branch apa-ld2 @ cb2fac1): Attn2 (scripts/apa_ld2_entries.py) and zws_cell
(scripts/apa_ld2_run.py) now live in scripts/apa_ld2_pdwl.py, LD1's Attn in scripts/apa_ld1_pdwl.py; the engine
tests use the importable tensor_cuda and skip, never error, without a usable CUDA device. One cell added: B04, the
(B) square control L = S, on the corrected ghost block (zws_cuts' L = S branch; the B04 fix the port order allows).
B04 is a negative control: at L = S a top-left statistics set equals the bottom-right one, so a statistics-mask bug
(PYSTAT, CU7) is a no-op there and B04 must PASS on correct and on statistics-mutant engines alike.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
import apa_ld1_pdwl as P  # noqa: E402
import apa_ld2_pdwl as P2  # noqa: E402


def _attn_np(q, k, v, rep, lo, hi, scale, dt):
    kk, vv = np.repeat(k, rep, 1), np.repeat(v, rep, 1)
    s = np.einsum("bhld,bhsd->bhls", q, kk).astype(np.float32) * np.float32(scale)
    j = np.arange(k.shape[2])
    vis = (j >= lo[..., None]) & (j <= hi[..., None])
    s = np.where(vis, s, -np.inf).astype(np.float32)
    w = np.exp(s - s.max(-1, keepdims=True)).astype(np.float32)
    w = w / w.sum(-1, keepdims=True)
    return P.round_to(np.einsum("bhls,bhsd->bhld", w, vv).astype(np.float32), dt)


def test_frontier_cuts_cover_every_row_boundary_and_pack_per_slab():
    B, H, L, S = 2, 8, 64, 1025
    lo, hi = P.causal_intervals(B, H, L, S)
    cuts = P2.frontier_cuts(lo, hi, S)
    assert cuts == [(t, "ge") for t in range(S - L, S)]          # hi_0, then hi_r + 1 for r = 0..L-2
    plan = P2.stagger_plan(cuts, B, 8)
    assert len(plan) == math.ceil(len(cuts) / 16)
    assert sorted(c for a in plan for c in a.values()) == cuts
    lo, hi = P.causal_intervals(B, H, 4, 3000, window=1024)
    wc = [c for c in P2.frontier_cuts(lo, hi, 3000) if c[1] == "lt"]
    assert wc == [(int(x), "lt") for x in lo[0, 0]]


@pytest.mark.parametrize("dt", ["float32", "bfloat16", "float16"])
def test_stagger_cuts_catch_single_row_and_window_leaks(dt):
    rng = np.random.default_rng(5)
    B, H, KVH, L, S, D = 2, 4, 2, 8, 70, 16
    scale = 1 / math.sqrt(D)
    q, k = P.make_qk(rng, (B, H, L, D), (B, KVH, S, D), scale, dt)
    v = P.make_v(rng, (B, KVH, S, D), dt)
    sig = P.score_bound(scale, q, k)
    lo, hi = P.causal_intervals(B, H, L, S)
    plan = P2.stagger_plan(P2.frontier_cuts(lo, hi, S), B, KVH)
    good = lambda X: _attn_np(q, k, X, 2, lo, hi, scale, dt)  # noqa: E731
    r = P2.stagger_cut_check(good, v, lo, hi, hi - lo + 1, plan, 2, dt, sig)
    assert r["verdict"] == "PASS" and r["repeat_noise"] == 0
    for row in (0, 5, 6):                                      # one leaky row (a single tile edge)
        hi2 = hi.copy()
        hi2[..., row] += 1
        bad = lambda X, h2=hi2: _attn_np(q, k, X, 2, lo, h2, scale, dt)  # noqa: E731
        r = P2.stagger_cut_check(bad, v, lo, hi, hi - lo + 1, plan, 2, dt, sig)
        assert r["verdict"] == "ALARM" and r["leak_row_ids"] == [row]
    W = 20
    lo, hi = P.causal_intervals(B, H, L, S, window=W)
    lo2, _ = P.causal_intervals(B, H, L, S, window=W + 1)
    plan = P2.stagger_plan(P2.frontier_cuts(lo, hi, S), B, KVH)
    good = lambda X: _attn_np(q, k, X, 2, lo, hi, scale, dt)  # noqa: E731
    bad = lambda X: _attn_np(q, k, X, 2, lo2, hi, scale, dt)  # noqa: E731
    assert P2.stagger_cut_check(good, v, lo, hi, hi - lo + 1, plan, 2, dt, sig)["verdict"] == "PASS"
    assert P2.stagger_cut_check(bad, v, lo, hi, hi - lo + 1, plan, 2, dt, sig)["verdict"] == "ALARM"


def _apa_np(q, k, kq, V, lo, hi, scale, z, stats):
    """numpy APA refine definition (fp32): stats over the visible set ('br') or a top-left statistics set ('tl',
    fd32b6d quant.py:174 shape: excluded keys enter as 0, count stays the visible count)."""
    B, H, L, _ = q.shape
    S = k.shape[2]
    j = np.arange(S)
    vis = (j >= lo[..., None]) & (j <= hi[..., None])
    stat = vis & ((j[None, None, None, :] <= np.arange(L)[None, None, :, None]) if stats == "tl" else True)
    bulk = np.einsum("bhld,bhsd->bhls", q, kq).astype(np.float32) * np.float32(scale)
    rank = np.einsum("bhld,bhsd->bhls", q, k).astype(np.float32) * np.float32(scale)
    a = np.where(stat, np.abs(bulk), 0.0).astype(np.float32)
    cnt = vis.sum(-1, keepdims=True).astype(np.float32)
    mean = np.where(vis, a, 0).sum(-1, keepdims=True) / cnt
    var = (np.where(vis, a - mean, 0) ** 2).sum(-1, keepdims=True) / cnt
    sel = vis & (a >= mean + np.float32(z) * np.sqrt(var))
    s = np.where(vis, np.where(sel, rank, bulk), -np.inf).astype(np.float32)
    w = np.exp(s - s.max(-1, keepdims=True)).astype(np.float32)
    w = w / w.sum(-1, keepdims=True)
    return np.einsum("bhls,bhsd->bhld", w, V).astype(np.float32)


@pytest.mark.parametrize("L,S", [(1, 33), (3, 33), (16, 200)])
def test_zws_numpy_definition_passes_and_topleft_statistics_alarm(L, S):
    rng = np.random.default_rng(11)
    B, H, D, dt = 2, 2, 32, "float32"
    scale, z, E = 1 / math.sqrt(D), 1.0364333894937898, P2.ZWS["E"][dt]
    t_g, cuts = P2.zws_cuts(L, S)
    q, k = P2.zws_qk(rng, B, H, H, L, S, D, scale, dt, t_g)
    kq = (k + 0.3 * np.abs(k).mean() * rng.standard_normal(k.shape)).astype(np.float32)   # quantization stand-in
    lo, hi = P.causal_intervals(B, H, L, S)
    em0 = P2.zws_emulate(q, kq, k, np.zeros((B, H, S, D)), lo, hi, scale, z, 1, lambda x: x,
                         [(b, h) for b in range(B) for h in range(H)])
    V = P2.zws_sensitive_v(em0, B, H, S, D, 1, dt, rng)
    for key in em0:
        em0[key]["V"] = V[key[0], key[1]].astype(np.float64)
    verdicts = {}
    for stats in ("br", "tl"):
        base = _apa_np(q, k, kq, V, lo, hi, scale, z, stats)
        Vg = V.copy()
        Vg[:, :, t_g:] *= np.float32(2.0 ** 64)                 # ghost weights must be exactly 0
        assert np.array_equal(_apa_np(q, k, kq, Vg, lo, hi, scale, z, stats), base)
        nviol, nmust = 0, 0
        for t in cuts:
            kt, kqt = k.copy(), kq.copy()
            kt[:, :, t:] *= np.float32(2.0 ** E)
            kqt[:, :, t:] *= np.float32(2.0 ** E)
            Y = _apa_np(q, kt, kqt, V, lo, hi, scale, z, stats)
            changed = np.any(P.bits(Y) != P.bits(base), -1)
            em1 = P2.zws_emulate(q, kqt, kt, np.zeros((B, H, S, D)), lo, hi, scale, z, 1, lambda x: x, list(em0))
            for key in em1:
                em1[key]["V"] = em0[key]["V"]
            mc = P2.zws_must_change(em0, em1, dt)
            touched = hi >= t
            must = np.stack([np.stack([mc[(b, h)][0] for h in range(H)]) for b in range(B)]) & touched
            nmust += int(must.sum())
            nviol += int((changed & ~touched).sum() + (~changed & must).sum())
        verdicts[stats] = nviol
        assert nmust > 0
    assert verdicts["br"] == 0                 # correct definition: no false alarm
    assert verdicts["tl"] > 0                  # statistics-only top-left mask: blind rows


# ---------------------------------------------------------------- real engine (main), small shapes
def _require_cuda(tc):
    """APA-REGT1: tensor_cuda imports without a CUDA device and fails at the first allocation; skip, never error
    (the probe of test_terrain_render.py's _require_cuda)."""
    try:
        tc.zeros((1,), dtype="float32").numpy()
    except Exception as exc:  # pragma: no cover - host dependent
        pytest.skip(f"CUDA unavailable: {exc}")


def _engine():
    tc = pytest.importorskip("tensor_cuda")
    _require_cuda(tc)
    EN = P  # APA-REGT1: the Attn adapter (apa_ld1_entries @ cb2fac1) now lives in scripts/apa_ld1_pdwl.py
    from tensor_cuda import functional as F
    from tensor_cuda import quant as Q
    EN.tc, EN.F, EN.Q = tc, F, Q
    return EN


@pytest.mark.parametrize("dt", ["float32", "bfloat16"])
def test_engine_stagger_cuts_pass_on_main_and_catch_offby1(dt):
    EN = _engine()
    E2 = P2  # APA-REGT1: Attn2 (apa_ld2_entries @ cb2fac1) now lives in scripts/apa_ld2_pdwl.py
    for fam in ("sdpa", "apa_sel"):
        for mut, want in ((None, "PASS"), ("offby1", "ALARM")):
            e = E2.Attn2(fam, dt, 2, 4, 2, 8, 24, 32, mutant=mut)
            lo, hi, nden = e.intervals()
            plan = P2.stagger_plan(P2.frontier_cuts(lo, hi, e.S), e.B, e.KVH)
            r = P2.stagger_cut_check(e.run, e.v, lo, hi, nden, plan, e.rep, dt, e.sigma)
            assert r["verdict"] == want, (fam, mut, r)


def test_engine_zws_passes_on_main_apa():
    _engine()
    E2 = P2  # APA-REGT1: Attn2 (apa_ld2_entries @ cb2fac1) now lives in scripts/apa_ld2_pdwl.py
    R = P2  # APA-REGT1: zws_cell (apa_ld2_run @ cb2fac1) now lives in scripts/apa_ld2_pdwl.py
    assert P2.zws_cuts(3, 33) == (30, [30, 31])  # APA-REGT1: the B04 fix leaves the registered L < S cells alone
    for fam, dt in (("apa_quant", "float32"), ("apa_sel", "bfloat16")):
        e = E2.Attn2(fam, dt, 2, 4, 2, 3, 33, 32, None, causal=True,
                     inject=P2.zws_qk(np.random.default_rng(1), 2, 4, 2, 3, 33, 32, 1 / math.sqrt(32), dt, 30))
        r = R.zws_cell(e, 30, P2.zws_cuts(3, 33)[1], np.random.default_rng(2), None)
        assert r["verdict"] == "PASS" and r["must_change_rows"] > 0, r
    # APA-REGT1: B04, the square control (L = S; APA-LD2 registered it as apa_quant fp32), at this test's small
    # geometry, on the corrected ghost block (APA-LD2's exploratory rerun: t_g = 16, cuts {16, 24}). The registered
    # rule gave t_g = 0 here and the cell was REFUSED-operand; it must now PASS with must-change rows.
    L = S = 33
    t_g, cuts = P2.zws_cuts(L, S)
    assert (t_g, cuts) == (16, [16, 24])
    e = E2.Attn2("apa_quant", "float32", 2, 4, 2, L, S, 32, None, causal=True,
                 inject=P2.zws_qk(np.random.default_rng(1), 2, 4, 2, L, S, 32, 1 / math.sqrt(32), "float32", t_g))
    r = R.zws_cell(e, t_g, cuts, np.random.default_rng(2), None)
    assert r["verdict"] == "PASS" and r["must_change_rows"] > 0, r
