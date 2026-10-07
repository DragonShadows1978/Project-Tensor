"""APA-MF2.2 item 1 (re-review N1): the two non-causal fused online-softmax recurrences
(fused_sdpa_noncausal_kernel, apa_int4_sdpa_noncausal_kernel) must not turn a -inf FIRST
score into NaN.

MF2.1 started the running max at m = -inf (closing F1). When a warp's first score is itself
-inf (finite operands overflowing: fp32 Q0=1e20 x K0=-1e20; fp16 scale 1e34 x dot -120000; the
fp16 INT4 bulk store of -120000), m_new = -inf and corr = expf(m - m_new) = expf(-inf + inf)
= NaN poisoned the row (lead GPU receipt: tensor_fork_apa_wt_mf21rev
logs/apa_mf21rev_gpu_fused_overflow_20261007T223910Z_248576.log, 18/18 N1 rows NaN).
Fix: FlashAttention-2's Check_inf guard -- reference m_ref = (m_new == -inf) ? 0 : m_new, so
corr = exp(-inf) = 0 and p = exp(-inf) = 0 -- with the -inf start KEPT (F1 stays closed; the
fused-kernel F1 rows below re-open if the start is reverted to -3.0e38f).

Witnesses: review N1 (apa-mf21rev docs/APA_MF2_1_REVIEW.md:20-27). Sweep, F1 rows and
oracle (scripts/apa_mf2_2_oracle.py softmax64): mine.
Prior art: FlashAttention-2 (Dao 2023), flash_attn softmax.h Check_inf (max == -inf -> 0);
Milakov & Gimelshein (2018) online normalizer. Unverified -- lead to check
("flash-attention softmax.h Check_inf", "Online normalizer calculation for softmax 2018").
"""
import os
import sys

import numpy as np
import pytest

import tensor_cuda as tc

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
import apa_mf2_2_oracle as O  # noqa: E402

Z = 1.0364333894937898          # Phi^-1(0.85), the registered p = 0.15
FLT_MAX = float(np.finfo(np.float32).max)
OPS = ("sdpa", "int4_all", "int4_partial")


@pytest.fixture(autouse=True)
def _streaming(monkeypatch):
    # fp16 D<=64 would route to the (unchanged) Q-tile kernel; force the streaming kernels.
    monkeypatch.setenv("TC_ATTN_QTILE", "0")


def _run(op, q, k, v, scale):
    with tc.no_grad():
        if op == "sdpa":
            return tc.fused_sdpa_noncausal(q, k, v, scale)
        return tc.apa_int4_sdpa_noncausal(q, k, v, scale, Z, op == "int4_all")


def _t(a, dt):
    return tc.tensor(np.asarray(a, np.float32), dtype=dt)


# ---------------------------------------------------------------- review witnesses
@pytest.mark.parametrize("S", [2, 5])
@pytest.mark.parametrize("D", [64, 96, 128])
@pytest.mark.parametrize("op", ["sdpa", "int4_all", "int4_partial_fp16"])
def test_review_n1_witness_is_exactly_one(op, D, S):
    dt = "float16" if op == "int4_partial_fp16" else "float32"
    q = np.zeros((1, 1, 1, D), np.float32); k = np.zeros((1, 1, S, D), np.float32)
    v = np.ones_like(k); v[0, 0, 0] = 0
    q[..., 0] = 2 if dt == "float16" else 1e20
    k[0, 0, 0, 0] = -60000 if dt == "float16" else -1e20
    xs = [_t(x, dt) for x in (q, k, v)]
    o = _run("int4_partial" if op == "int4_partial_fp16" else op, *xs, 1.0).float().numpy()
    assert np.isfinite(o).all() and np.all(o == 1.0), o[..., :4]


# ---------------------------------------------------------------- -inf-first sweep
def _pattern(name, S):
    if name == "key0":
        n = 1
    elif name == "warp_first":
        n = min(4, S - 1)
    else:
        n = S // 2
    return np.arange(S) < n


def _sweep_operands(dt, op, S, D, Lq, pattern, seed):
    """Finite operands whose first scores overflow to -inf in the kernel."""
    rng = np.random.default_rng(seed)
    neg = _pattern(pattern, S)
    q = np.zeros((1, 1, Lq, D), np.float32)
    k = np.zeros((1, 1, S, D), np.float32)
    v = rng.normal(0, 1, (1, 1, S, D)).astype(np.float32)
    if dt == "float32":
        q[..., 0] = 1e20
        if op != "int4_partial":
            q[..., 1:] = rng.normal(0, 1, (Lq, D - 1))
            k[0, 0, ~neg, 1:] = rng.normal(0, 0.3, (int((~neg).sum()), D - 1))
        k[0, 0, neg, 0] = -1e20
        scale = 1.0
    else:
        q[..., 0] = 2
        k[0, 0, neg, 0] = -60000
        scale = 1.0 if op == "int4_partial" else 1e34
    xs = [_t(x, dt) for x in (q, k, v)]
    q64, k64, v64 = (x.float().numpy().astype(np.float64)[0, 0] for x in xs)
    s64 = (q64 @ k64.T) * scale
    s64[:, neg] = -np.inf                     # the kernel's fp32 / fp16 score is -inf there
    return xs, scale, O.softmax64(s64, v64), neg


@pytest.mark.parametrize("pattern", ["key0", "warp_first", "half"])
@pytest.mark.parametrize("S", [2, 5, 33, 1025])
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("dt", ["float32", "float16"])
def test_neg_inf_first_rows_finite_and_correct(dt, op, S, pattern):
    for D in (64, 96, 128):
        for Lq in (1, 3):
            xs, scale, ref, neg = _sweep_operands(dt, op, S, D, Lq, pattern, 1000 * S + D + Lq)
            o = _run(op, *xs, scale).float().numpy()[0, 0]
            assert np.isfinite(o).all(), f"NaN/Inf: {dt} {op} S={S} D={D} Lq={Lq} {pattern}"
            tol = 1e-5 * max(1.0, float(np.abs(ref).max())) if dt == "float32" else 4e-3
            err = float(np.abs(o - ref).max())
            assert err <= tol, f"{dt} {op} S={S} D={D} Lq={Lq} {pattern}: max err {err:.3e} > {tol:.1e}"


def test_bf16_is_rejected_by_both_kernels():
    """Recorded domain boundary: both kernels take fp16/fp32 only (unchanged by this order)."""
    xs = [tc.tensor(np.zeros((1, 1, 1, 64), np.float32), dtype="bfloat16") for _ in range(3)]
    for op in ("sdpa", "int4_all"):
        with pytest.raises(Exception, match="float16/float32 only"):
            _run(op, *xs, 1.0)


# ---------------------------------------------------------------- F1 stays closed
@pytest.mark.parametrize("s", [-3.3e38, -3.402e38])
@pytest.mark.parametrize("S", [1, 5])
@pytest.mark.parametrize("op", ["sdpa", "int4_all"])
def test_fused_extreme_negative_visible_score_keeps_weight(op, S, s):
    """Row max below the old -3.0e38f start: with m = -inf it gets weight 1 (output exactly
    V0 = 1); a reverted -3.0e38f start gives l = 0 and output 0 (F1 re-opened)."""
    D = 64
    q = np.zeros((1, 1, 1, D), np.float32); q[..., 0] = 1
    k = np.zeros((1, 1, S, D), np.float32); k[0, 0, :, 0] = -FLT_MAX; k[0, 0, 0, 0] = s
    v = np.zeros_like(k); v[0, 0, 0] = 1
    o = _run(op, *[_t(x, "float32") for x in (q, k, v)], 1.0).numpy()
    assert np.isfinite(o).all() and np.all(o == 1.0), o[..., :4]


# ---------------------------------------------------------------- ordinary rows
@pytest.mark.parametrize("op", ["sdpa", "int4_all"])
@pytest.mark.parametrize("dt", ["float32", "float16"])
def test_ordinary_finite_rows_match_fp64(dt, op):
    for D in (64, 96, 128):
        for S in (1, 3, 9, 200):
            rng = np.random.default_rng(D + S)
            xs = [_t(rng.normal(0, 0.5, (1, 2, n, D)), dt) for n in (3, S, S)]
            q64, k64, v64 = (x.float().numpy().astype(np.float64)[0] for x in xs)
            ref = O.softmax64(np.einsum("hld,hsd->hls", q64, k64) * D ** -0.5, v64)
            o = _run(op, *xs, D ** -0.5).float().numpy()[0]
            tol = 1e-5 if dt == "float32" else 4e-3
            assert np.isfinite(o).all() and float(np.abs(o - ref).max()) <= tol
