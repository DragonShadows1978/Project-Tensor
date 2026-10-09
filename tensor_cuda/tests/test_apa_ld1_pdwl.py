"""APA-LD1: the power-of-two tag leak detector (PDWL) -- detector logic on numpy kernels (CPU) and the
real engine at small shapes (GPU). Author-run tests: a baseline, not blind verification.

Prior art: OpenAI (2026) openai/math #107 Prop 3.1 / #135 via the TX2 Opus M4a and Astra M04 dossiers
(power-of-two degeneration tags, Walsh-bit runs, cut predicate, basis path probes); Goldberg (1991)
exactness of power-of-two scaling; Chen et al. (1998) metamorphic testing; DeMillo, Lipton & Sayward
(1978) mutation testing; Dao (2023) FlashAttention-2 bottom-right causal. Unverified -- lead to check.
Mine: the engine harness under test (scripts/apa_ld1_pdwl.py, scripts/apa_ld1_entries.py).

APA-REGT1 production port (from branch apa-ld2 @ cb2fac1): the engine adapter of scripts/apa_ld1_entries.py now
lives in scripts/apa_ld1_pdwl.py; the engine tests use the importable tensor_cuda (like every test in this
directory) and skip, never error, when no CUDA device is usable. Detector logic and cells unchanged.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
import apa_ld1_pdwl as P  # noqa: E402


def _attn_np(q, k, v, rep, lo, hi, scale, dt):
    kk, vv = np.repeat(k, rep, 1), np.repeat(v, rep, 1)
    s = np.einsum("bhld,bhsd->bhls", q, kk).astype(np.float32) * np.float32(scale)
    j = np.arange(k.shape[2])
    vis = (j >= lo[..., None]) & (j <= hi[..., None])
    s = np.where(vis, s, -np.inf).astype(np.float32)
    w = np.exp(s - s.max(-1, keepdims=True)).astype(np.float32)
    w = w / w.sum(-1, keepdims=True)
    return P.round_to(np.einsum("bhls,bhsd->bhld", w, vv).astype(np.float32), dt)


def test_walsh_schedule_separates_every_label_pair():
    for n in (2, 3, 5, 8, 13, 80):
        runs = P.walsh_schedule(n, 7)
        assert len(runs) == P.run_budget(n) == 2 * math.ceil(math.log2(n)) + 1
        assert not runs[0].any() and all(set(np.unique(r)) <= {0, 7} for r in runs)
        for a in range(n):
            for b in range(a + 1, n):
                assert any(r[a] != r[b] for r in runs[1:])


@pytest.mark.parametrize("dt", ["float32", "bfloat16", "float16"])
def test_numpy_kernels_correct_pass_mutants_alarm(dt):
    rng = np.random.default_rng(3)
    B, H, KVH, L, S, D = 2, 8, 4, 16, 40, 32
    scale = 1 / math.sqrt(D)
    q, k = P.make_qk(rng, (B, H, L, D), (B, KVH, S, D), scale, dt)
    v = P.make_v(rng, (B, KVH, S, D), dt)
    lo, hi = P.causal_intervals(B, H, L, S)
    sig = P.score_bound(scale, q, k)
    assert P.precheck(dt, sig, S)[0]
    li = np.ascontiguousarray(np.broadcast_to(np.arange(KVH)[None], (B, KVH)))
    lo_ = np.ascontiguousarray(np.broadcast_to((np.arange(H) // 2)[None], (B, H)))
    good = lambda X: _attn_np(q, k, X, 2, lo, hi, scale, dt)  # noqa: E731
    tile = lambda X: _attn_np(q, k[:, np.arange(H) % KVH], X[:, np.arange(H) % KVH], 1, lo, hi, scale, dt)  # noqa: E731
    tl_hi = np.ascontiguousarray(np.broadcast_to(np.arange(L), lo.shape))
    topleft = lambda X: _attn_np(q, k, X, 2, np.zeros_like(lo), tl_hi, scale, dt)  # noqa: E731
    assert P.label_check(good, v, li, lo_, dt)["verdict"] == "PASS"
    r = P.label_check(tile, v, li, lo_, dt)
    assert r["verdict"] == "ALARM" and r["runs"] == 5
    cuts = P.causal_cuts(L, S)
    assert P.cut_check(good, v, 2, lo, hi, cuts, dt, sig, hi - lo + 1)["verdict"] == "PASS"
    r = P.cut_check(topleft, v, 2, lo, hi, cuts, dt, sig, hi - lo + 1)
    assert r["verdict"] == "ALARM"
    pinned = [c for c in r["per_cut"] if "offset_discrepancy" in c]
    assert pinned and pinned[0]["offset_discrepancy"] == [S - L, S - L]
    assert pinned[0]["kernel_offset_estimate"] == [0, 0]


def test_operand_validity_refusals():
    assert not P.precheck("float32", 40.0, 4096)[0]          # weights below 2^-100
    assert P.precheck("float16", P.SIGMA, 98304)[0]
    tiny = np.full((2, 2), 2.0 ** -20, np.float32)              # fp16 subnormal-range output
    assert not P.postcheck("float16", tiny)[0]
    assert P.postcheck("float32", tiny)[0]


def test_basis_paths_have_no_cross_terms():
    paths = P.basis_paths(4, 64, 96, 96)
    A, Bm, C = P.basis_operands(4, 64, 96, 96, paths, trans_b=True)
    assert np.array_equal(np.einsum("bmk,bnk->bmn", A, Bm), C)
    assert {k for _, _, k, _ in paths} >= {0, 31, 32, 63, 64, 95}


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
def test_engine_sdpa_and_apa_selective_pass_and_mutants_alarm(dt):
    EN = _engine()
    for fam in ("sdpa", "apa_sel"):
        e = EN.Attn(fam, dt, 2, 4, 2, 8, 24, 32)
        li, lo_ = e.labels("head")
        assert P.label_check(e.run, e.v, li, lo_, dt)["verdict"] == "PASS"
        lo, hi, nden = e.intervals()
        assert P.cut_check(e.run, e.v, 2, lo, hi, e.cuts(), dt, e.sigma, nden)["verdict"] == "PASS"
        m = EN.Attn(fam, dt, 2, 4, 2, 8, 24, 32, mutant="tile")
        assert P.label_check(m.run, m.v, li, lo_, dt)["verdict"] == "ALARM"
        m = EN.Attn(fam, dt, 2, 4, 2, 8, 24, 32, mutant="topleft")
        r = P.cut_check(m.run, m.v, 2, lo, hi, m.cuts(), dt, m.sigma, nden)
        assert r["verdict"] == "ALARM"
        pinned = [c for c in r["per_cut"] if "offset_discrepancy" in c]
        assert pinned and pinned[0]["offset_discrepancy"] == [24 - 8, 24 - 8]
