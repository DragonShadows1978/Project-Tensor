"""APA-MF2 item 1: apa_quant_attention's refine selection == the fused BK4 rule.

The rule (kernels.cu bk4::forward, :8064-8077 and :8112-8114): statistics over the
VISIBLE keys of |fl32(q.kq * scale)| (the unrounded fp32 dot), population one-pass
moments with the clamp, th = mean + z*sqrt(var), sel = visible && |bulk| >= th.
The reference for the bit tests is the kernel's OWN diagnostic mask
(tc._C.bp_kernel_4_diagnostic, variant 'h') on identical operands: q, k, kq, v
are the same bf16 values on both sides (kq = quant._quantize_keys of the bf16 K,
i.e. exactly what apa_quant_attention computes). fp16 has no kernel, so it is
compared with the independent fp64 oracle (scripts/apa_mf2_bk4emu.rule_fp64)
inside the registered tie band |a - th| <= 2e-5*max(1, |th|), counted.

Review cases (docs/APA_MF1_REVIEW.md on apa-mf1rev): B1 stored-bf16 statistics,
B2 non-causal centered vs one-pass moments, B3 the missing `visible &&`.

Prior art: kernels.cu BK4 (Project-Tensor 2026); FlashAttention-2 bottom-right
causal (Dao 2023); 63960a4. Harness and witnesses: mine (B1-B3 inputs from the
MF1 review, independently re-derived).
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
import apa_mf2_bk4emu as E  # noqa: E402

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


def kernel_mask(q, k, kq, v, scale, causal, variant="h"):
    r = tc._C.bp_kernel_4_diagnostic(q, k, kq, v, float(scale), float(Z), bool(causal), variant)
    return r[3].numpy() != 0, r[2].numpy()


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
def test_bf16_selection_equals_bk4_kernel_mask(L, S, D, causal):
    """fp32 rule on identical bf16 operands must reproduce BK4 'h' bit for bit."""
    q, k, v = bf16_operands(L, S, D)
    qt, kt, vt = (tc.tensor(a, dtype="bfloat16") for a in (q, k, v))
    R, CB, BND = quant._tables(D, 8, 2, True, "cuda")
    kq = quant._quantize_keys(kt, R, CB, BND)
    km, _ = kernel_mask(qt, kt, kq, vt, D ** -0.5, causal)
    sel, out = selection_of(qt, kt, vt, causal)
    sel = np.broadcast_to(sel, km.shape)
    mism = int((sel != km).sum())
    assert mism == 0, f"{mism} selection bits differ from BK4 'h' (L{L} S{S} D{D} causal={causal})"
    assert np.isfinite(out.float().numpy()).all()


def test_b1_statistics_on_unrounded_fp32_bulk():
    """B1: BK4 thresholds the fp32 dot; bf16-stored bulk [.125,.125] would select both."""
    D = 64
    q = np.zeros((1, 1, 1, D), np.float32); q[..., 0:2] = 1
    kqa = np.zeros((1, 1, 2, D), np.float32); kqa[..., 0] = 1; kqa[0, 0, 1, 1] = 1 / 256
    v = np.zeros_like(kqa); v[0, 0, 1, :] = 1
    qt, kqt, vt = (tc.tensor(a, dtype="bfloat16") for a in (q, kqa, v))
    km, thr = kernel_mask(qt, kqt, kqt, vt, 0.125, True)
    assert km.tolist() == [[[[False, False]]]], "BK4 witness changed"
    with supplied_kq(kqt):
        sel, _ = selection_of(qt, kqt, vt, True, scale=0.125)
    assert np.array_equal(np.broadcast_to(sel, km.shape), km), f"B1: selection {sel.tolist()} vs BK4 {km.tolist()}"


def test_b2_noncausal_population_one_pass_moments():
    """B2: non-causal threshold uses one-pass moments (BK4), not centered mean/std."""
    D = 64
    q = np.zeros((1, 1, 1, D), np.float32); q[..., 0:2] = 1
    kqa = np.zeros((1, 1, 2, D), np.float32); kqa[..., 0] = 800.0
    kqa[0, 0, 1, 1] = 0.00799560546875                      # bf16-exact; bulk [100, 100.0009995]
    v = np.zeros_like(kqa); v[0, 0, 1, :] = 1
    qb, kqb, vb = (tc.tensor(a, dtype="bfloat16") for a in (q, kqa, v))
    km, thr = kernel_mask(qb, kqb, kqb, vb, 0.125, False)
    assert km.tolist() == [[[[False, True]]]], f"BK4 witness changed: {km.tolist()} thr {thr.tolist()}"
    a = np.array([100.0, 100.0009994506836], np.float32)
    centered = a >= np.float32(a.mean() + a.std() * np.float32(Z))
    assert centered.tolist() == [False, False]              # the witness is a real discriminator
    qt, kqt, vt = (tc.tensor(x) for x in (q, kqa, v))       # fp32 path isolates B2 from B1
    with supplied_kq(kqt):
        sel, _ = selection_of(qt, kqt, vt, False, scale=0.125)
    assert np.array_equal(np.broadcast_to(sel, km.shape), km), f"B2: selection {sel.tolist()} vs BK4 {km.tolist()}"


@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "float16"])
def test_b3_zero_threshold_selects_only_visible(dtype):
    """B3: Q=K=0 gives threshold 0; BK4 selects exactly the 24 visible entries of L3 S9."""
    q = tc.tensor(np.zeros((1, 1, 3, 64), np.float32), dtype=dtype)
    k = tc.tensor(np.zeros((1, 1, 9, 64), np.float32), dtype=dtype)
    sel, out = selection_of(q, k, k, True)
    vis = E.visible(3, 9)
    assert int(vis.sum()) == 24
    assert np.array_equal(np.broadcast_to(sel, (1, 1, 3, 9))[0, 0], vis), f"selected {int(sel.sum())} bits, expected the 24 visible"
    if dtype == "bfloat16":
        qb = tc.tensor(np.zeros((1, 1, 3, 64), np.float32), dtype="bfloat16")
        kb = tc.tensor(np.zeros((1, 1, 9, 64), np.float32), dtype="bfloat16")
        km, _ = kernel_mask(qb, kb, kb, kb, 0.125, True)
        assert np.array_equal(km[0, 0], vis)
    assert np.isfinite(out.float().numpy()).all()


@pytest.mark.parametrize("L,S,D", [(3, 9, 64), (64, 261, 128), (1, 5, 96)])
def test_fp16_selection_within_tie_band_of_fp64_rule(L, S, D):
    """fp16 has no BK4 entry: compare with the fp64 oracle on the call's own operands."""
    q, k, v = bf16_operands(L, S, D)
    qt, kt, vt = (tc.tensor(a, dtype="float16") for a in (q, k, v))
    R, CB, BND = quant._tables(D, 8, 2, True, "cuda")
    kq = quant._quantize_keys(kt, R, CB, BND).float().numpy().astype(np.float64)
    q64 = qt.float().numpy().astype(np.float64)
    absb = np.abs(np.einsum("bhld,bhsd->bhls", q64, kq) * (D ** -0.5))
    vis = E.visible(L, S)
    sel_ref, th = E.rule_fp64(absb, vis, Z)
    sel, _ = selection_of(qt, kt, vt, True)
    sel = np.broadcast_to(sel, sel_ref.shape)
    band = np.abs(absb - th) <= 2e-5 * np.maximum(1.0, np.abs(th))
    bad = (sel != sel_ref) & ~band
    assert not bad.any(), f"{int(bad.sum())} fp16 mismatches outside the tie band"


def test_fp32_helper_is_the_selection_on_identical_operands():
    """The fp32 item-1 cell: the selection helper on fp32 copies of the kernel's bf16
    operands equals both the kernel mask and apa_quant_attention's own bf16 selection."""
    L, S, D = 64, 261, 128
    q, k, v = bf16_operands(L, S, D, seed=7)
    qt, kt, vt = (tc.tensor(a, dtype="bfloat16") for a in (q, k, v))
    R, CB, BND = quant._tables(D, 8, 2, True, "cuda")
    kq = quant._quantize_keys(kt, R, CB, BND)
    km, _ = kernel_mask(qt, kt, kq, vt, D ** -0.5, True)
    helper = quant._bk4_selection(tc.tensor(q), kq.float(), D ** -0.5, Z, True).numpy() != 0
    sel, _ = selection_of(qt, kt, vt, True)
    assert np.array_equal(helper, np.broadcast_to(sel, helper.shape))
    assert int((helper != km).sum()) == 0
