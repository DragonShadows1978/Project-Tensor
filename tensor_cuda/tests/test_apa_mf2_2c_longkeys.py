"""APA-MF2.2 Amendment 3 (review V7): the selection statistics match the fp64 definition for long key
axes (S up to 2^20 visible keys), including tightly clustered |bulk| rows.

Before: quant.py summed both moments with the engine's single-pass row sum (kernels.cu
reduce_lastdim_kernel: 256 threads x S/256 sequential fp32 adds + an 8-level tree). At S = 2^20 that is
4103 roundings deep; on the review witness b = sort(1 - (j mod 256) 2^-20) the mean drifts 4e-5 and
the threshold lands at 1.0000026 vs the definition's 0.99995145 (122880 keys outside the band; lead
GPU receipt apa-mf22rev logs/apa_mf22rev_gpu_long_keys_20261008T004356Z_377640.log).
After: quant._row_sum keeps the single-pass sum for S <= 8192 (depth <= 39) and sums 4096-key blocks
then their partials for longer rows (depth <= 32 at S <= 2^20); the definition is unchanged.
Bands: the registered |a - thr64| <= 2e-5 |thr64| AND the review's 2e-5 x visible row max.
Prior art: pairwise / blocked summation and its error bound -- Higham (1993, SIAM J. Sci. Comput.
14(4), "The accuracy of floating point summation"; 2002 book ch. 4); Chan, Golub & LeVeque (1983)
discuss pairwise accumulation for the variance. Unverified -- lead to check. Witness: the review's
(apa-mf22rev docs/APA_MF2_2_REVIEW.md V7); sweep and tests: mine.
"""
import contextlib
import os
import sys

import numpy as np
import pytest

import tensor_cuda as tc
from tensor_cuda import quant as Q

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
import apa_mf2_2_oracle as O  # noqa: E402

Z = Q._norm_ppf(1.0 - 0.15)


def review_band_errors(sel, b, causal):
    """The review oracle's band: 2e-5 x the visible row max of |b| (apa-mf22rev oracle.py)."""
    ref, thr, a, vis = O.definition64(b, causal, Z)
    rowmax = np.where(vis, a, 0.0).max(-1, keepdims=True)
    bad = ((np.asarray(sel) != 0) != ref) & vis
    return int((bad & (np.abs(a - thr) > 2e-5 * rowmax)).sum())


def check(sel, b, causal, label):
    r = O.mismatch(sel, b, causal, Z)
    rb = review_band_errors(sel, b, causal)
    assert r["out_of_band"] == 0 and rb == 0 and r["invisible_selected"] == 0, f"{label}: {r}, review-band {rb}"
    return r


def rows(S, width, pattern, L, seed):
    j = np.arange(S)
    if pattern == "sorted_steps":
        f = np.sort((j % 256) / 256.0)
    elif pattern == "steps":
        f = (j % 256) / 256.0
    else:
        f = np.random.default_rng(seed).uniform(0.0, 1.0, S)
    return np.broadcast_to((1.0 - width * f).astype(np.float32), (L, S))


def select(b32, causal):
    L, S = b32.shape[-2:]
    with tc.no_grad():
        return Q._definition_selection(tc.tensor(np.zeros((1, 1, L, 8), np.float32)),
                                       tc.tensor(np.zeros((1, 1, S, 8), np.float32)), 1.0, Z, causal,
                                       bulk32=tc.tensor(b32)).numpy()


def witness():
    S = 1 << 20
    return np.sort(1.0 - (np.arange(S) % 256) * 2.0 ** -20).astype(np.float32)


def test_review_v7_witness_exact_operand():
    b = witness(); S = b.size; D = 64
    q = np.zeros((1, 1, 1, D), np.float32); q[..., 0] = 1.0
    k = np.zeros((1, 1, S, D), np.float32); k[..., 0] = b
    with tc.no_grad():
        qt, kt = tc.tensor(q), tc.tensor(k)
        bulk = tc.matmul(qt, kt, trans_b=True).numpy()
        assert np.array_equal(bulk.ravel(), b)                     # the one-hot dot is exact
        sel = Q._definition_selection(qt, kt, 1.0, Z, False).numpy()
    r = check(sel, bulk, False, "V7 exact")
    assert r["selected64"] == 208896 and r["selected"] >= 208896 - r["in_band"]


def test_review_v7_witness_public_quantizer_operand():
    b = witness(); S = b.size; D = 64
    q = np.zeros((1, 1, 1, D), np.float32); q[..., 0] = 1.0
    k = np.zeros((1, 1, S, D), np.float32); k[..., 0] = b
    with tc.no_grad():
        qt, kt = tc.tensor(q), tc.tensor(k)
        R, CB, BND = Q._tables(D, 4, 1, True, "cuda")
        kq = Q._quantize_keys(kt, R, CB, BND)
        bulk = tc.matmul(qt, kq, trans_b=True)
        sel = Q._definition_selection(qt, kq, 1.0, Z, False, bulk32=bulk).numpy()
    check(sel, bulk.numpy(), False, "V7 public quantizer")


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("pattern", ["sorted_steps", "steps", "uniform"])
@pytest.mark.parametrize("width", [2.0 ** -20, 2.0 ** -16, 2.0 ** -12, 2.0 ** -10, 1.0])  # 2^-12: the witness width (added)
@pytest.mark.parametrize("S", [1 << 10, 1 << 14, 1 << 17, (1 << 17) + 3, 1 << 20])
def test_long_key_sweep(S, width, pattern, causal):
    b = rows(S, width, pattern, 2, S)[None, None]
    check(select(b, causal), b, causal, f"S={S} w={width:g} {pattern} causal={causal}")


@contextlib.contextmanager
def capture():
    rec = {}
    old = Q._definition_selection

    def spy(qf, kqf, scale, z, is_causal, bulk32=None):
        if bulk32 is None:
            bulk32 = tc.matmul(qf, kqf, alpha=scale, trans_b=True)
        rec["bulk32"] = bulk32.numpy()
        rec["sel"] = old(qf, kqf, scale, z, is_causal, bulk32=bulk32).numpy()
        return old(qf, kqf, scale, z, is_causal, bulk32=bulk32)

    Q._definition_selection = spy
    try:
        yield rec
    finally:
        Q._definition_selection = old


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("S", [1 << 14, 1 << 17, 1 << 20])
@pytest.mark.parametrize("dt", ["float32", "bfloat16"])
def test_full_api_long_keys(dt, S, causal):
    D, L = 64, 2
    b = rows(S, 2.0 ** -12, "sorted_steps", 1, S)[0]
    q = np.zeros((1, 1, L, D), np.float32); q[..., 0] = 1.0
    k = np.zeros((1, 1, S, D), np.float32); k[..., 0] = b
    v = np.zeros((1, 1, S, D), np.float32); v[..., 0] = 1.0
    with capture() as rec, tc.no_grad():
        o = Q.apa_quant_attention(*[tc.tensor(x, dtype=dt) for x in (q, k, v)], bulk_bits=4,
                                  refine_percentile=0.15, is_causal=causal, scale=1.0)
    check(rec["sel"], rec["bulk32"], causal, f"full API {dt} S={S} causal={causal}")
    assert np.isfinite(o.float().numpy()).all()


@pytest.mark.parametrize("S", [1, 2, 4097, 8192, 8193, 12288, (1 << 17) + 3, (1 << 20) + 5])
def test_row_sum_accuracy_and_coverage(S):
    """_row_sum adds every key exactly once (incl. the remainder) with error <= 40 * 2^-24 * sum|x|."""
    x = np.random.default_rng(S).uniform(0.0, 1.0, (2, 1, 3, S)).astype(np.float32)
    got = Q._row_sum(tc.tensor(x)).numpy().astype(np.float64)
    ref = x.astype(np.float64).sum(-1, keepdims=True)
    assert got.shape == ref.shape
    assert np.all(np.abs(got - ref) <= 40 * 2.0 ** -24 * ref + 1e-30), np.abs(got - ref).max()
