"""APA-MF2 item 1, re-based by APA-MF2.1 item 3: apa_quant_attention's refine selection
== the DEFINITION (not BK4's arithmetic).

The definition (orders/APA_MF2_1_REVIEW_FIXES.md, Measures 3): statistics over the
VISIBLE keys of |b|, b the fp32 bulk dot of the call's own (q, kq) (B1: never the
16-bit stored bulk); population mean and std (centered); th = mean + z*std;
sel = visible && |b| >= th (B3). Oracle: scripts/apa_mf2_1_defn.py (fp64, centered
two-pass) on the exact upcasts of the same operands; agreement inside the registered
tie band | |b| - th | <= 2e-5 |th| (relative, no floor).

MF2 asserted bit-equality with the BK4 'h' diagnostic mask. The MF2 review (F2) showed
BK4's one-pass FMA statistics deviate from the definition by up to 2.8e-3 relative on
near-ties; the lead retired that target ("the contract is the mathematical
definition"). The kernel deviation is recorded by the NON-gating observation test in
test_apa_mf2_1_definition.py; nothing here encodes BK4 arithmetic.

Review cases (docs/APA_MF1_REVIEW.md on apa-mf1rev): B1 stored-bf16 statistics,
B2 non-causal statistics, B3 the missing `visible &&`.

Prior art: centered two-pass population variance; Chan, Golub & LeVeque (1983) on
one-pass cancellation -- unverified, lead to check; FlashAttention-2 bottom-right
causal (Dao 2023); 63960a4. Harness and witnesses: mine (B1-B3 inputs from the MF1
review, independently re-derived).
"""
import contextlib
import os
import sys

import numpy as np
import pytest

import tensor_cuda as tc
from tensor_cuda import quant

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import apa_mf2_bk4emu as E  # noqa: E402  (bf16 rounding helpers only)
import apa_mf2_1_defn as DEF  # noqa: E402

P = 0.15
Z = quant._norm_ppf(1.0 - P)


@contextlib.contextmanager
def capture_selection():
    """Record the condition of every tc.where call; the selection is the last one."""
    rec = []
    orig = tc.where

    def wh(c, x, y):
        rec.append(c.float().numpy() != 0)
        return orig(c, x, y)

    tc.where = wh
    try:
        yield rec
    finally:
        tc.where = orig


@contextlib.contextmanager
def supplied_kq(kq):
    """Make apa_quant_attention use a supplied quantized key (isolates the rule)."""
    orig = quant._quantize_keys
    quant._quantize_keys = lambda *a, **k: kq
    try:
        yield
    finally:
        quant._quantize_keys = orig


def definition(qt, kq, scale, causal):
    """fp64 definition on the exact upcasts of the call's own operands."""
    b64 = DEF.bulk_fp64(qt.float().numpy(), kq.float().numpy(), scale)
    return DEF.definition_fp64(b64, causal, Z)


def bf16_operands(L, S, D, H=2, seed=0):
    rng = np.random.default_rng(20261007 + 1000 * L + 10 * S + D + seed)
    q, k, v = (E.to_bf16_f32(rng.normal(0, 0.5, (1, H, n, D)).astype(np.float32)) for n in (L, S, S))
    return q, k, v


def selection_of(qt, kt, vt, causal, **kw):
    with capture_selection() as rec:
        out = quant.apa_quant_attention(qt, kt, vt, bulk_bits=8, refine_percentile=P,
                                        is_causal=causal, **kw)
    assert rec, "no tc.where call recorded"
    return rec[-1], out


CELLS = [(L, S, D, c) for L in (1, 3, 64) for S in (L, L + 1, 4 * L + 5)
         for D in (64, 128) for c in (True, False)]


@pytest.mark.parametrize("L,S,D,causal", CELLS)
def test_bf16_selection_equals_definition(L, S, D, causal):
    """bf16 call on random operands: selection == fp64 definition outside the tie band
    (was: bit-equal to BK4 'h', retired by MF2.1)."""
    q, k, v = bf16_operands(L, S, D)
    qt, kt, vt = (tc.tensor(a, dtype="bfloat16") for a in (q, k, v))
    R, CB, BND = quant._tables(D, 8, 2, True, "cuda")
    kq = quant._quantize_keys(kt, R, CB, BND)
    ref, thr, a, vis = definition(qt, kq, D ** -0.5, causal)
    sel, out = selection_of(qt, kt, vt, causal)
    res = DEF.compare(sel, ref, a, thr, vis)
    assert res["out_of_band"] == 0 and res["invisible_selected"] == 0, f"L{L} S{S} D{D} causal={causal}: {res}"
    assert np.isfinite(out.float().numpy()).all()


def test_b1_statistics_on_unrounded_fp32_bulk():
    """B1: the definition thresholds the fp32 dot [.125, .12548828] and selects neither;
    the bf16-stored bulk [.125, .125] has std 0 and would select both."""
    D = 64
    q = np.zeros((1, 1, 1, D), np.float32); q[..., 0:2] = 1
    kqa = np.zeros((1, 1, 2, D), np.float32); kqa[..., 0] = 1; kqa[0, 0, 1, 1] = 1 / 256
    v = np.zeros_like(kqa); v[0, 0, 1, :] = 1
    qt, kqt, vt = (tc.tensor(a, dtype="bfloat16") for a in (q, kqa, v))
    ref, thr, a, vis = definition(qt, kqt, 0.125, True)
    assert ref.tolist() == [[[[False, False]]]], "definition witness changed"
    assert tc.tensor(np.float32(0.125 + 0.125 / 256), dtype="bfloat16").float().numpy() == 0.125  # stored bulk ties
    with supplied_kq(kqt):
        sel, _ = selection_of(qt, kqt, vt, True, scale=0.125)
    assert np.array_equal(np.broadcast_to(sel, ref.shape), ref), f"B1: selection {sel.tolist()} vs definition"


def test_b2_noncausal_population_statistics():
    """B2 operands (MF1 review), non-causal, fp32: bulk [100, 100.0009995]. The definition
    (centered population moments) gives th 100.0010177 and selects neither; key 1 is
    1.8e-7 relative below th -- inside the tie band, so only band agreement is asserted
    (MF2 asserted BK4's one-pass [F, T] here; retired)."""
    D = 64
    q = np.zeros((1, 1, 1, D), np.float32); q[..., 0:2] = 1
    kqa = np.zeros((1, 1, 2, D), np.float32); kqa[..., 0] = 800.0
    kqa[0, 0, 1, 1] = 0.00799560546875
    v = np.zeros_like(kqa); v[0, 0, 1, :] = 1
    qt, kqt, vt = (tc.tensor(x) for x in (q, kqa, v))
    ref, thr, a, vis = definition(qt, kqt, 0.125, False)
    assert ref.tolist() == [[[[False, False]]]]
    with supplied_kq(kqt):
        sel, _ = selection_of(qt, kqt, vt, False, scale=0.125)
    res = DEF.compare(sel, ref, a, thr, vis)
    assert res["out_of_band"] == 0, res


@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "float16"])
def test_b3_zero_threshold_selects_only_visible(dtype):
    """B3: Q=K=0 gives threshold 0; the definition selects exactly the 24 visible entries of L3 S9."""
    q = tc.tensor(np.zeros((1, 1, 3, 64), np.float32), dtype=dtype)
    k = tc.tensor(np.zeros((1, 1, 9, 64), np.float32), dtype=dtype)
    sel, out = selection_of(q, k, k, True)
    vis = E.visible(3, 9)
    assert int(vis.sum()) == 24
    assert np.array_equal(np.broadcast_to(sel, (1, 1, 3, 9))[0, 0], vis), f"selected {int(sel.sum())} bits, expected the 24 visible"
    assert np.isfinite(out.float().numpy()).all()


@pytest.mark.parametrize("L,S,D", [(3, 9, 64), (64, 261, 128), (1, 5, 96)])
def test_fp16_selection_within_tie_band_of_definition(L, S, D):
    """fp16 call: compare with the fp64 definition on the call's own operands."""
    q, k, v = bf16_operands(L, S, D)
    qt, kt, vt = (tc.tensor(a, dtype="float16") for a in (q, k, v))
    R, CB, BND = quant._tables(D, 8, 2, True, "cuda")
    kq = quant._quantize_keys(kt, R, CB, BND)
    ref, thr, a, vis = definition(qt, kq, D ** -0.5, True)
    sel, _ = selection_of(qt, kt, vt, True)
    res = DEF.compare(sel, ref, a, thr, vis)
    assert res["out_of_band"] == 0, f"fp16 mismatches outside the tie band: {res}"


def test_fp32_helper_is_the_selection_on_identical_operands():
    """The fp32 helper on fp32 copies of the bf16 operands equals apa_quant_attention's own
    bf16 selection (plumbing), and both agree with the definition outside the band."""
    L, S, D = 64, 261, 128
    q, k, v = bf16_operands(L, S, D, seed=7)
    qt, kt, vt = (tc.tensor(a, dtype="bfloat16") for a in (q, k, v))
    R, CB, BND = quant._tables(D, 8, 2, True, "cuda")
    kq = quant._quantize_keys(kt, R, CB, BND)
    helper = quant._bk4_selection(tc.tensor(q), kq.float(), D ** -0.5, Z, True).numpy() != 0
    sel, _ = selection_of(qt, kt, vt, True)
    assert np.array_equal(helper, np.broadcast_to(sel, helper.shape))
    ref, thr, a, vis = definition(qt, kq, D ** -0.5, True)
    assert DEF.compare(helper, ref, a, thr, vis)["out_of_band"] == 0
