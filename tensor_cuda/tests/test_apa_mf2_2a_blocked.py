"""APA-MF2.2 Amendment 1, item 3: a FULLY BLOCKED row (every score -inf) of the two non-causal fused
kernels returns fd32b6d's value, 0, never NaN.

fd32b6d (start -3.0e38f): every warp keeps m = -3e38, l = 0, acc = 0; gmax = -3e38, rescale = 1,
denominator 0 -> inv_denom = 0 -> output 0. MF2.1/MF2.2 (start -inf, kept for F1/N1): gmax = -inf and
the merge's rescale = expf(-inf - -inf) = NaN poisons the row (census receipt
artifacts/apa_mf2_2a/census/compare_prefix_20261007T2327Z.json). Fix: the merge reference
g_ref = (gmax == -inf) ? 0 : gmax (FlashAttention-2's combine guard), so rescale = exp(-inf) = 0 and the
already-guarded normalizer (inv_denom = 0) gives 0.
Rows with any finite score keep MF2.2's bytes (identity harness); large-magnitude finite rows below
pin the merge reference to gmax (a constant reference overflows/underflows there).
Prior art: FlashAttention-2 (Dao 2023) split-KV combine / softmax Check_inf guards (max == -inf -> 0);
unverified -- lead to check. Tests: mine.
"""
import os
import sys

import numpy as np
import pytest

import tensor_cuda as tc

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
import apa_mf2_2_oracle as O  # noqa: E402

Z = 1.0364333894937898
OPS = ("sdpa", "int4_all", "int4_partial")


@pytest.fixture(autouse=True)
def _streaming(monkeypatch):
    monkeypatch.setenv("TC_ATTN_QTILE", "0")


def _run(op, q, k, v, scale):
    with tc.no_grad():
        if op == "sdpa":
            return tc.fused_sdpa_noncausal(q, k, v, scale)
        return tc.apa_int4_sdpa_noncausal(q, k, v, scale, Z, op == "int4_all")


def _blocked(dt, S, D, Lq, how, seed):
    rng = np.random.default_rng(seed)
    q = np.zeros((1, 2, Lq, D), np.float32); k = np.zeros((1, 2, S, D), np.float32)
    v = rng.normal(0, 1, (1, 2, S, D)).astype(np.float32)
    if how == "literal":                      # a -inf key component (fp32 only)
        q[..., 0] = 1.0; k[..., 0] = -np.inf
        return q, k, v, 1.0
    if dt == "float16":                       # finite operands, score overflows the fp32 product
        q[..., 0] = 2.0; k[..., 0] = -60000.0
        return q, k, v, 1e34
    q[..., 0] = 1e20; k[..., 0] = -1e20
    return q, k, v, 1.0


@pytest.mark.parametrize("how", ["overflow", "literal"])
@pytest.mark.parametrize("S", [1, 2, 33, 1025])
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("dt", ["float32", "float16"])
def test_fully_blocked_row_is_zero(dt, op, S, how):
    if how == "literal" and dt == "float16":
        pytest.skip("literal -inf operands: fp32 only (fp16 overflow construction covers fp16)")
    if how == "literal" and op == "int4_partial":
        # A -inf key component makes the INT4 packer's scale inf and its codes NaN: the bulk score is
        # NaN, not -inf -- a NaN row, not a blocked row. fd32b6d and MF2 return NaN there too
        # (logs/apa_mf2_2a_literal_inf_check.log); such outputs are recorded, not changed.
        pytest.skip("literal -inf K -> NaN INT4 codes (NaN on fd32b6d too); overflow construction covers int4_partial")
    for D in (64, 96, 128):
        for Lq in (1, 3):
            q, k, v, scale = _blocked(dt, S, D, Lq, how, S + D + Lq)
            o = _run(op, *[tc.tensor(x, dtype=dt) for x in (q, k, v)], scale).float().numpy()
            assert np.isfinite(o).all() and np.all(o == 0.0), f"{dt} {op} S={S} D={D} Lq={Lq} {how}: {o.reshape(-1)[:4]}"


@pytest.mark.parametrize("op", OPS)
def test_blocked_row_next_to_ordinary_rows(op):
    """One fully blocked query row among ordinary rows: the others stay correct (fp64 oracle)."""
    D, S = 64, 33
    rng = np.random.default_rng(3)
    q = rng.normal(0, 1, (1, 1, 3, D)).astype(np.float32); q[0, 0, 1] = 0; q[0, 0, 1, 0] = 1e20
    k = rng.normal(0, 0.3, (1, 1, S, D)).astype(np.float32); k[..., 0] = -1e20 if op != "int4_partial" else 0.0
    if op == "int4_partial":
        k = np.zeros_like(k); k[..., 0] = -1e20; q = np.zeros_like(q); q[0, 0, 0, 1] = 1; q[0, 0, 2, 1] = 1
        q[0, 0, 1, 0] = 1e20
    v = rng.normal(0, 1, (1, 1, S, D)).astype(np.float32)
    o = _run(op, *[tc.tensor(x) for x in (q, k, v)], 1.0).numpy()[0, 0]
    assert np.all(o[1] == 0.0), o[1, :4]
    for r in (0, 2):
        s64 = q[0, 0, r].astype(np.float64) @ k[0, 0].astype(np.float64).T
        if np.isfinite(s64).all() and np.abs(s64).max() < 1e30:
            ref = O.softmax64(s64[None], v[0, 0].astype(np.float64))[0]
            assert np.abs(o[r] - ref).max() <= 1e-5, (r, np.abs(o[r] - ref).max())


@pytest.mark.parametrize("shift", [-300.0, -150.0, 150.0, 300.0])
@pytest.mark.parametrize("op", ["sdpa", "int4_all"])
def test_large_magnitude_finite_rows_use_the_row_max(op, shift):
    """Scores ~ shift +- 3 (finite, far from 0): the merge must rescale by the row max; a fixed
    reference overflows (exp(+150) * ...) or underflows to 0/0 here."""
    D, S = 64, 33
    rng = np.random.default_rng(int(abs(shift)))
    q = np.zeros((1, 1, 3, D), np.float32); q[..., 0] = 1.0; q[..., 1:] = rng.normal(0, 1, (3, D - 1))
    k = np.zeros((1, 1, S, D), np.float32); k[..., 0] = shift; k[..., 1:] = rng.normal(0, 0.1, (S, D - 1))
    v = rng.normal(0, 1, (1, 1, S, D)).astype(np.float32)
    o = _run(op, *[tc.tensor(x) for x in (q, k, v)], 1.0).numpy()[0, 0]
    s64 = q[0, 0].astype(np.float64) @ k[0, 0].astype(np.float64).T
    ref = O.softmax64(s64, v[0, 0].astype(np.float64))
    assert np.isfinite(o).all() and np.abs(o - ref).max() <= 2e-5, np.abs(o - ref).max()
