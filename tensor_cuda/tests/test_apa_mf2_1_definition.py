"""APA-MF2.1 item 3 (review F2): apa_quant_attention's refine selection == the DEFINITION.

The contract is the mathematical definition, not BK4's arithmetic (order Measures 3):
population mean and std of the VISIBLE |bulk| (fp32 bulk dot), threshold mean + z*std,
select visible && |bulk| >= thr. The oracle is scripts/apa_mf2_1_defn.py (fp64,
centered two-pass, exact upcasts of the call's OWN operands: kq is captured from
quant._quantize_keys inside the call, or supplied). Agreement: every mismatch lies in
the registered tie band | |b| - thr | <= 2e-5 |thr| (relative, no floor).

BK4's one-pass FMA statistics deviate from the definition on near-ties (review F2: S2
rows 2/2 bits, S256 256/256, rel gap 2.5e-4 .. 2.8e-3). That is REPORTED by the
non-gating observation test at the bottom; no test here encodes BK4 arithmetic.

Prior art: centered two-pass variance; Chan, Golub & LeVeque (1983) on one-pass
cancellation; Welford (1962) -- unverified, lead to check. Bottom-right visibility:
FlashAttention-2 (Dao 2023) / 63960a4. Near-tie operands: the MF2 review (F2). Count-
divisor discriminator and harness: mine.
"""
import contextlib
import os
import sys

import numpy as np
import pytest

import tensor_cuda as tc
from tensor_cuda import quant as Q

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import apa_mf2_1_defn as DEF  # noqa: E402

P = 0.15
Z = Q._norm_ppf(1.0 - P)
DTYPES = ("float32", "bfloat16", "float16")


@contextlib.contextmanager
def observe():
    """Record the call's own quantized key and its selection (last tc.where condition)."""
    rec = {"where": []}
    ow, oq = tc.where, Q._quantize_keys

    def wh(c, x, y):
        rec["where"].append(c.float().numpy() != 0)
        return ow(c, x, y)

    def qk(*a, **kw):
        r = oq(*a, **kw)
        rec["kq"] = r
        return r

    tc.where, Q._quantize_keys = wh, qk
    try:
        yield rec
    finally:
        tc.where, Q._quantize_keys = ow, oq


@contextlib.contextmanager
def supplied_kq(kq):
    orig = Q._quantize_keys
    Q._quantize_keys = lambda *a, **kw: kq
    try:
        yield
    finally:
        Q._quantize_keys = orig


def run_and_compare(qt, kt, vt, causal, scale=None, kq=None):
    D = qt.shape[-1]
    sc = scale if scale is not None else D ** -0.5
    ctx = supplied_kq(kq) if kq is not None else contextlib.nullcontext()
    with ctx, observe() as rec:
        out = Q.apa_quant_attention(qt, kt, vt, bulk_bits=8, refine_percentile=P, is_causal=causal, scale=scale)
    kq_used = kq if kq is not None else rec["kq"]
    sel = rec["where"][-1]
    b64 = DEF.bulk_fp64(qt.float().numpy(), kq_used.float().numpy(), sc)
    ref, thr, a, vis = DEF.definition_fp64(b64, causal, Z)
    return DEF.compare(sel, ref, a, thr, vis), out, sel, ref


def operands(L, S, D, dtype, seed=0, H=2):
    rng = np.random.default_rng(20261007 + 1000 * L + 10 * S + D + seed)
    return [tc.tensor(rng.normal(0, 0.5, (1, H, n, D)).astype(np.float32), dtype=dtype) for n in (L, S, S)]


CELLS = [(L, S, D, dt, c) for L in (1, 3, 64) for S in (L, L + 1, 4 * L + 5)
         for D in (64, 96, 128) for dt in DTYPES for c in (True, False)]


@pytest.mark.parametrize("L,S,D,dtype,causal", CELLS)
def test_selection_equals_definition_synthetic(L, S, D, dtype, causal):
    res, out, _, _ = run_and_compare(*operands(L, S, D, dtype), causal)
    assert res["out_of_band"] == 0, res
    assert res["invisible_selected"] == 0, res
    assert np.isfinite(out.float().numpy()).all()


NEAR = [(D, L, S, c) for D in (64, 96, 128) for (L, S) in ((1, 1), (1, 2), (1, 256), (1, 2048), (3, 9))
        for c in (False, True)]


def near_tie(L, S, D):
    """Review F2: Q[:3]=1, Kq[:3]=[1,107/256,1/65536] (bf16-exact), every key equal, so
    every visible |b| = 1.4179840087890625 and the definition's variance is exactly 0."""
    q = np.zeros((1, 1, L, D), np.float32); q[..., :3] = 1
    k = np.zeros((1, 1, S, D), np.float32); k[..., :3] = [1, 107 / 256, 1 / 65536]
    return [tc.tensor(a, dtype="bfloat16") for a in (q, k, np.zeros_like(k))]


@pytest.mark.parametrize("D,L,S,causal", NEAR)
def test_review_near_tie_constructions_match_definition(D, L, S, causal):
    qt, kt, vt = near_tie(L, S, D)
    res, out, sel, ref = run_and_compare(qt, kt, vt, causal, scale=1.0, kq=kt)
    print("near-tie", D, L, S, causal, res)
    assert res["out_of_band"] == 0, res
    assert ref.sum() == ref.size if not causal else ref.sum() == DEF.visible(L, S).sum()


@pytest.mark.parametrize("D,L,S,causal", NEAR)
def test_zero_variance_rows_select_every_visible_key(D, L, S, causal):
    """Definition property: when every visible |b| of a row is equal, var = 0 exactly,
    thr = mean = |b| and every visible key is selected (any z). The review's near-tie
    rows have |b| = 1.4179840087890625 (17 significant bits); for these counts (1, 2,
    256, 2048 and 7, 8, 9) every fp32 partial sum of equal values is exact, so the
    composed fp32 path has no rounding excuse. The tie band cannot see this case (every
    entry sits exactly ON the fp64 threshold), hence this exact property test. MF2's
    one-pass E[a^2]-E[a]^2 selected 0 of 2048 at S=2048 (review F2 / rule cell)."""
    qt, kt, vt = near_tie(L, S, D)
    with supplied_kq(kt), observe() as rec:
        Q.apa_quant_attention(qt, kt, vt, bulk_bits=8, refine_percentile=P, is_causal=causal, scale=1.0)
    sel = np.broadcast_to(rec["where"][-1], (1, 1, L, S))[0, 0]
    vis = DEF.visible(L, S, causal)
    assert np.array_equal(sel, vis), f"selected {int(sel.sum())} of {int(vis.sum())} equal visible keys"


def test_b1_statistics_on_unrounded_fp32_bulk():
    """B1 (MF1 review): the 16-bit stored bulk [.125, .125] would give std 0 and select
    both; the definition on the fp32 dot [.125, .12548828] selects neither (gap 7e-5,
    far outside the band)."""
    D = 64
    q = np.zeros((1, 1, 1, D), np.float32); q[..., 0:2] = 1
    kq = np.zeros((1, 1, 2, D), np.float32); kq[..., 0] = 1; kq[0, 0, 1, 1] = 1 / 256
    v = np.zeros_like(kq); v[0, 0, 1, :] = 1
    qt, kqt, vt = (tc.tensor(a, dtype="bfloat16") for a in (q, kq, v))
    res, _, sel, ref = run_and_compare(qt, kqt, vt, True, scale=0.125, kq=kqt)
    assert ref.tolist() == [[[[False, False]]]]
    assert res["mismatch"] == 0, (sel.tolist(), res)


@pytest.mark.parametrize("dtype", DTYPES)
def test_b3_zero_threshold_selects_only_visible(dtype):
    """Q=K=0: thr 0, every visible |b| = 0 >= 0; the 24 visible entries of L3 S9 only."""
    q = tc.tensor(np.zeros((1, 1, 3, 64), np.float32), dtype=dtype)
    k = tc.tensor(np.zeros((1, 1, 9, 64), np.float32), dtype=dtype)
    res, out, sel, ref = run_and_compare(q, k, k, True)
    vis = DEF.visible(3, 9)
    assert np.array_equal(np.broadcast_to(sel, (1, 1, 3, 9))[0, 0], vis), f"selected {int(sel.sum())}"
    assert np.isfinite(out.float().numpy()).all()


def count_witness(dtype="float32"):
    """L3 S9 causal, Q = e0, Kq[:,0] = [1,1,1,1,1,1.2,1.5,1,1]: rows see 7, 8, 9 keys.
    Definition: each row selects only the 1.5 (thr 1.284 / 1.263 / 1.246). Dividing by
    S instead of the visible count gives thr 1.132 / 1.170 in rows 0 / 1 and selects the
    1.2 as well (gaps >= 0.06, no ties)."""
    q = np.zeros((1, 1, 3, 64), np.float32); q[..., 0] = 1
    k = np.zeros((1, 1, 9, 64), np.float32); k[0, 0, :, 0] = [1, 1, 1, 1, 1, 1.2, 1.5, 1, 1]
    return [tc.tensor(a, dtype=dtype) for a in (q, k, k)]


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_visible_count_divisor_discriminator(dtype):
    qt, kt, vt = count_witness(dtype)
    res, _, sel, ref = run_and_compare(qt, kt, vt, True, scale=1.0, kq=kt)
    want = np.zeros((3, 9), bool); want[:, 6] = True
    assert np.array_equal(ref[0, 0], want)
    assert res["mismatch"] == 0, (np.broadcast_to(sel, (1, 1, 3, 9))[0, 0].astype(int).tolist(), res)


def test_constant_visible_rows_select_every_visible_key():
    """Reviewer's constant-one rows (zero variance): every visible key is selected."""
    q = np.zeros((1, 1, 3, 64), np.float32); q[..., 0] = 1
    k = np.zeros((1, 1, 9, 64), np.float32); k[..., 0] = 1
    qt, kt = tc.tensor(q), tc.tensor(k)
    sel = Q._bk4_selection(qt, kt, 1.0, Z, True).numpy() != 0
    assert np.array_equal(sel[0, 0], DEF.visible(3, 9))


def test_fp32_helper_is_the_call_selection_on_identical_operands():
    L, S, D = 64, 261, 128
    qt, kt, vt = operands(L, S, D, "bfloat16", seed=7)
    R, CB, BND = Q._tables(D, 8, 2, True, "cuda")
    kq = Q._quantize_keys(kt, R, CB, BND)
    helper = Q._bk4_selection(qt.float(), kq.float(), D ** -0.5, Z, True).numpy() != 0
    with observe() as rec:
        Q.apa_quant_attention(qt, kt, vt, bulk_bits=8, refine_percentile=P, is_causal=True)
    assert np.array_equal(helper, np.broadcast_to(rec["where"][-1], helper.shape))


OBS = [(D, L, S, c) for D in (64, 128) for (L, S) in ((1, 1), (1, 2), (1, 256), (3, 9), (64, 261)) for c in (True, False)]


@pytest.mark.parametrize("D,L,S,causal", OBS)
def test_observe_kernel_deviation_from_definition_nongating(D, L, S, causal):
    """NON-GATING observation: BK4 'h' and 'a' masks vs the definition, printed for the
    report. Deviations are expected (one-pass FMA statistics, review F2) and never fail
    this test; it fails only if the kernel cannot be run on these operands."""
    if (L, S) in ((1, 1), (1, 2), (1, 256)):
        qt, kt, vt = near_tie(L, S, D); kq = kt; scale = 1.0
    else:
        qt, kt, vt = operands(L, S, D, "bfloat16", seed=3)
        R, CB, BND = Q._tables(D, 8, 2, True, "cuda")
        kq = Q._quantize_keys(kt, R, CB, BND); scale = D ** -0.5
    b64 = DEF.bulk_fp64(qt.float().numpy(), kq.float().numpy(), scale)
    ref, thr, a, vis = DEF.definition_fp64(b64, causal, Z)
    for var in ("h", "a"):
        r = tc._C.bp_kernel_4_diagnostic(qt, kt, kq, vt, float(scale), float(Z), bool(causal), var)
        km = r[3].numpy() != 0
        assert km.shape == ref.shape
        print(f"OBS kernel={var} D{D} L{L} S{S} causal={causal}", DEF.compare(km, ref, a, thr, vis))
