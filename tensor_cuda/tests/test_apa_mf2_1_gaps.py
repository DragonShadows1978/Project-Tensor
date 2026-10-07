"""APA-MF2.1 item 4: the MF2 review's single-site test gaps (F4, F9).

F4a -- an additive reversion of the -inf predicate (scores + bias instead of
masked_fill) passes ordinary finite-score tests because finite + (-inf) = -inf. It is
discriminated only when a MASKED score is itself +inf: fp16 L3/S9/D64, Q0 = 2*e0 (row 0
only), K7 = 60000*e0, V7 = 1, scale 1 -> masked score 120000 overflows fp16 to +inf;
predicate -> -inf -> row 0 exactly 0, all finite; additive -> +inf + -inf = NaN.
Sites: SDPA (grad and no-grad), quant ranking (refine 1.0: scores = ranking) and quant
bulk (refine 0.15 with the supplied finite Kq = K, so the quantizer cannot overflow and
the bulk site is isolated from the ranking site: masked entries take the bulk value).
F4b (apa.cpp kNegInf-only reversion) and F4c (count divisor) live in
apa_cuda/tests/test_apa_mf2_1_extension.py and test_apa_mf2_1_definition.py.
F9 -- test_apa_phase6._ref masked SCORES top-left while its statistics were
bottom-right; its tests were square-only. Rectangular discriminator: L3/S9, Q=K=0,
V_j = j -> bottom-right rows see 7, 8, 9 keys -> [3, 3.5, 4] (top-left gives [0, .5, 1]).

Prior art: review operands (apa-mf2rev docs/APA_MF2_REVIEW.md F4, F9); FlashAttention-2
(Dao 2023) -inf predicate masking; DeMillo, Lipton & Sayward (1978) single-site
mutation adequacy -- unverified, lead to check. Tests: mine.
"""
import contextlib
import importlib.util
import os

import numpy as np
import pytest

import tensor_cuda as tc
from tensor_cuda import functional as F
from tensor_cuda import quant as Q

HERE = os.path.dirname(os.path.abspath(__file__))


def _f4_operands(dtype="float16"):
    q = np.zeros((1, 1, 3, 64), np.float32); q[0, 0, 0, 0] = 2
    k = np.zeros((1, 1, 9, 64), np.float32); k[0, 0, 7, 0] = 60000
    v = np.zeros_like(k); v[0, 0, 7] = 1
    return [tc.tensor(a, dtype=dtype) for a in (q, k, v)]


def _check_row0_zero(o):
    o = o.float().numpy()[0, 0]
    assert np.isfinite(o).all(), "non-finite output: additive -inf reversion (inf + -inf)"
    assert np.all(o[0] == 0.0), f"row 0 leaked {o[0, :4]}"


def test_f4_witness_masked_score_overflows_fp16():
    q, k, _ = _f4_operands()
    s = tc.matmul(q, k, alpha=1.0, trans_b=True).float().numpy()[0, 0]
    assert np.isposinf(s[0, 7]) and np.all(s[0, :7] == 0)


@pytest.mark.parametrize("grad", [False, True])
def test_f4_sdpa_predicate_site_fp16(grad):
    q, k, v = _f4_operands()
    ctx = contextlib.nullcontext() if grad else tc.no_grad()
    with ctx:
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0)
    _check_row0_zero(o)


def test_f4_quant_ranking_site_fp16():
    q, k, v = _f4_operands()
    _check_row0_zero(Q.apa_quant_attention(q, k, v, bulk_bits=8, refine_percentile=1.0,
                                           is_causal=True, scale=1.0))


def test_f4_quant_bulk_site_fp16_supplied_kq():
    q, k, v = _f4_operands()
    orig = Q._quantize_keys
    Q._quantize_keys = lambda *a, **kw: k
    try:
        o = Q.apa_quant_attention(q, k, v, bulk_bits=8, refine_percentile=0.15, is_causal=True, scale=1.0)
    finally:
        Q._quantize_keys = orig
    _check_row0_zero(o)


def _phase6():
    spec = importlib.util.spec_from_file_location("_mf2_1_phase6", os.path.join(HERE, "test_apa_phase6.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize("pct", [1.0, 0.15])
def test_f9_phase6_reference_is_bottom_right(pct):
    L, S, D = 3, 9, 64
    q = np.zeros((1, 1, L, D), np.float32); k = np.zeros((1, 1, S, D), np.float32)
    v = np.broadcast_to(np.arange(S, dtype=np.float32)[None, None, :, None], (1, 1, S, D)).copy()
    ref = _phase6()._ref(q, k, v, 2, pct, True)
    np.testing.assert_allclose(ref[0, 0, :, 0], [3.0, 3.5, 4.0], rtol=0, atol=1e-12)


@pytest.mark.parametrize("L,S", [(16, 40), (5, 23)])
def test_f9_phase6_reference_matches_engine_rectangular(L, S):
    rng = np.random.default_rng(L * 100 + S)
    q, k, v = (rng.standard_normal((1, 2, n, 32)).astype(np.float32) for n in (L, S, S))
    ref = _phase6()._ref(q, k, v, 2, 1.0, True)
    out = tc.apa_quant_attention(tc.tensor(q), tc.tensor(k), tc.tensor(v), bulk_bits=2,
                                 refine_percentile=1.0, is_causal=True).numpy().astype(np.float64)
    assert np.abs(out - ref).max() < 1e-4
