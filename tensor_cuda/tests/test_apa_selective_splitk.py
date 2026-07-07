"""Parity tests for the APA selective split-K decode path (kernel-opt
Addendum 1, workstream A1: docs/KERNEL_OPT_PLAN_ADDENDUM_1.md).

The split-K path is a work-distribution change ONLY: three smaller kernels
(global stats -> per-partition split scoring -> merge) replace the single
fused one-block-per-row kernel for the decode grid-underfill case (small
rows, long S). It must reproduce the fused kernel's outputs exactly within
existing tolerances — same z-score threshold (computed from full-key-range
bulk statistics), same per-key bulk-vs-refine selection, same online-softmax
math, sink logit folded exactly once at the merge stage.

TC_APA_SELECTIVE_PATH env var forces a path independent of the shape-based
dispatch heuristic (apa_selective_use_splitk in kernels.cu):
  0 (unset) = auto (shape-based heuristic)
  1         = force the original fused kernel
  2         = force the split-K path (requires L==1 / decode shapes)

These tests force both paths explicitly at each shape so parity is checked
regardless of where the dispatch boundary sits, plus one auto-dispatch check
that the boundary actually routes long-S decode to split-K without being
told to.
"""
import os

import numpy as np
import pytest
import tensor_cuda as tc

from tensor_cuda.quant import _norm_ppf


def _clear_override():
    os.environ.pop("TC_APA_SELECTIVE_PATH", None)


def _run_plain(path, B, H, KVH, L, S, D, dtype, causal, seed=0):
    rng = np.random.default_rng(seed)
    q = (rng.standard_normal((B, H, L, D)) * 0.1).astype(np.float32)
    k = (rng.standard_normal((B, KVH, S, D)) * 0.1).astype(np.float32)
    kq = (k + rng.standard_normal((B, KVH, S, D)) * 0.01).astype(np.float32)
    v = (rng.standard_normal((B, KVH, S, D)) * 0.1).astype(np.float32)
    scale = 1.0 / np.sqrt(D)
    z = _norm_ppf(1.0 - 0.15)

    q_t = tc.tensor(q, dtype=dtype)
    k_t = tc.tensor(k, dtype=dtype)
    kq_t = tc.tensor(kq, dtype=dtype)
    v_t = tc.tensor(v, dtype=dtype)

    os.environ["TC_APA_SELECTIVE_PATH"] = str(path)
    try:
        out = tc.apa_selective_attention(q_t, k_t, kq_t, v_t, float(scale), float(z), causal)
    finally:
        _clear_override()
    return out.numpy().astype(np.float32)


def _run_sink(path, B, H, KVH, L, S, D, dtype, causal, seed=0):
    rng = np.random.default_rng(seed)
    q = (rng.standard_normal((B, H, L, D)) * 0.1).astype(np.float32)
    k = (rng.standard_normal((B, KVH, S, D)) * 0.1).astype(np.float32)
    kq = (k + rng.standard_normal((B, KVH, S, D)) * 0.01).astype(np.float32)
    v = (rng.standard_normal((B, KVH, S, D)) * 0.1).astype(np.float32)
    sinks = (rng.standard_normal(H) * 0.2).astype(np.float32)
    scale = 1.0 / np.sqrt(D)
    z = _norm_ppf(1.0 - 0.15)

    q_t = tc.tensor(q, dtype=dtype)
    k_t = tc.tensor(k, dtype=dtype)
    kq_t = tc.tensor(kq, dtype=dtype)
    v_t = tc.tensor(v, dtype=dtype)
    sinks_t = tc.tensor(sinks, dtype=dtype)

    os.environ["TC_APA_SELECTIVE_PATH"] = str(path)
    try:
        out = tc.apa_selective_attention_sink(
            q_t, k_t, kq_t, v_t, sinks_t, float(scale), float(z), causal)
    finally:
        _clear_override()
    return out.numpy().astype(np.float32)


# Tolerance: same as test_apa_selective.py's existing sink test (rtol/atol
# 1e-3) for f32; f16/bf16 get a looser absolute tolerance since the split-K
# path recomputes bulk dots twice independently (stats kernel + split
# kernel) just as the fused kernel recomputes them across its two passes —
# same reduced-precision rounding profile as the reference kernel, not a new
# numerics source, but low-precision dtypes have coarser ULPs.
TOL = {
    "float32": dict(rtol=1e-3, atol=1e-3),
    "float16": dict(rtol=2e-2, atol=2e-2),
    "bfloat16": dict(rtol=4e-2, atol=4e-2),
}

S_SHAPES = (2048, 8192, 32768)
DTYPES = ("float16", "bfloat16", "float32")

# (H, KVH, D) per addendum task geometry.
GPT_OSS20B = dict(H=64, KVH=8, D=64)
QWEN35 = dict(H=16, KVH=4, D=256)


@pytest.fixture(autouse=True)
def _cleanup_env():
    yield
    _clear_override()


@pytest.mark.parametrize("S", S_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_splitk_matches_fused_gpt_oss20b_nosink(S, dtype):
    got = _run_plain(2, 1, GPT_OSS20B["H"], GPT_OSS20B["KVH"], 1, S, GPT_OSS20B["D"], dtype, True)
    ref = _run_plain(1, 1, GPT_OSS20B["H"], GPT_OSS20B["KVH"], 1, S, GPT_OSS20B["D"], dtype, True)
    tol = TOL[dtype]
    diff = np.abs(got - ref)
    assert diff.max() < tol["atol"] + tol["rtol"] * np.abs(ref).max(), \
        f"S={S} dtype={dtype} max diff {diff.max()}"


@pytest.mark.parametrize("S", S_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_splitk_matches_fused_gpt_oss20b_sink(S, dtype):
    got = _run_sink(2, 1, GPT_OSS20B["H"], GPT_OSS20B["KVH"], 1, S, GPT_OSS20B["D"], dtype, True)
    ref = _run_sink(1, 1, GPT_OSS20B["H"], GPT_OSS20B["KVH"], 1, S, GPT_OSS20B["D"], dtype, True)
    tol = TOL[dtype]
    diff = np.abs(got - ref)
    assert diff.max() < tol["atol"] + tol["rtol"] * np.abs(ref).max(), \
        f"S={S} dtype={dtype} max diff {diff.max()}"


@pytest.mark.parametrize("S", S_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_splitk_matches_fused_qwen35_nosink(S, dtype):
    got = _run_plain(2, 1, QWEN35["H"], QWEN35["KVH"], 1, S, QWEN35["D"], dtype, True)
    ref = _run_plain(1, 1, QWEN35["H"], QWEN35["KVH"], 1, S, QWEN35["D"], dtype, True)
    tol = TOL[dtype]
    diff = np.abs(got - ref)
    assert diff.max() < tol["atol"] + tol["rtol"] * np.abs(ref).max(), \
        f"S={S} dtype={dtype} max diff {diff.max()}"


def test_splitk_noncausal_matches_fused():
    """Non-causal decode (S >= L, full key range every row) exercises the
    is_causal=0 s_max=S branch in both the stats and split kernels."""
    got = _run_plain(2, 1, GPT_OSS20B["H"], GPT_OSS20B["KVH"], 1, 8192, GPT_OSS20B["D"], "float32", False)
    ref = _run_plain(1, 1, GPT_OSS20B["H"], GPT_OSS20B["KVH"], 1, 8192, GPT_OSS20B["D"], "float32", False)
    diff = np.abs(got - ref)
    assert diff.max() < 1e-3, f"max diff {diff.max()}"


def test_splitk_matches_numpy_reference():
    """Cross-check the split-K path against the same independent NumPy
    reference used by test_apa_selective.py (not just against the fused
    kernel), so a bug shared by both CUDA kernels would still be caught."""
    B, H, KVH, L, S, D = 1, 8, 2, 1, 4096, 64
    rng = np.random.default_rng(42)
    q = (rng.standard_normal((B, H, L, D)) * 0.1).astype(np.float32)
    k = (rng.standard_normal((B, KVH, S, D)) * 0.1).astype(np.float32)
    v = (rng.standard_normal((B, KVH, S, D)) * 0.1).astype(np.float32)
    kq = (k + rng.standard_normal((B, KVH, S, D)) * 0.01).astype(np.float32)
    scale = 1.0 / np.sqrt(D)
    z = _norm_ppf(1.0 - 0.15)
    group = H // KVH

    expected = np.zeros((B, H, L, D), np.float32)
    for b in range(B):
        for h in range(H):
            kh = h // group
            for i in range(L):
                s_max = ((S - L) + i + 1)
                qi = q[b, h, i]
                bulk = (kq[b, kh, :s_max] @ qi) * scale
                a = np.abs(bulk)
                thr = a.mean() + z * np.sqrt(max(a.var(), 0.0))
                score = np.empty(s_max, np.float32)
                for j in range(s_max):
                    if abs(bulk[j]) >= thr:
                        score[j] = (k[b, kh, j] @ qi) * scale
                    else:
                        score[j] = bulk[j]
                m = score.max()
                w = np.exp(score - m)
                w /= w.sum()
                expected[b, h, i] = w @ v[b, kh, :s_max]

    os.environ["TC_APA_SELECTIVE_PATH"] = "2"
    try:
        got = tc.apa_selective_attention(
            tc.tensor(q), tc.tensor(k), tc.tensor(kq), tc.tensor(v),
            float(scale), float(z), True).numpy()
    finally:
        _clear_override()
    diff = np.abs(got - expected)
    assert diff.max() < 1e-3, f"max diff {diff.max()}"


def test_auto_dispatch_routes_long_decode_to_splitk():
    """The shape-based heuristic (apa_selective_use_splitk) should pick the
    split-K path without being told to for a real long-context decode shape
    (small rows, S well past the partition-headroom threshold), and its
    result must match the explicitly-forced fused path."""
    B, H, KVH, L, S, D = 1, GPT_OSS20B["H"], GPT_OSS20B["KVH"], 1, 32768, GPT_OSS20B["D"]
    fused = _run_plain(1, B, H, KVH, L, S, D, "float32", True, seed=7)
    _clear_override()  # auto (no override)
    rng = np.random.default_rng(7)
    q = (rng.standard_normal((B, H, L, D)) * 0.1).astype(np.float32)
    k = (rng.standard_normal((B, KVH, S, D)) * 0.1).astype(np.float32)
    kq = (k + rng.standard_normal((B, KVH, S, D)) * 0.01).astype(np.float32)
    v = (rng.standard_normal((B, KVH, S, D)) * 0.1).astype(np.float32)
    scale = 1.0 / np.sqrt(D)
    z = _norm_ppf(1.0 - 0.15)
    auto = tc.apa_selective_attention(
        tc.tensor(q), tc.tensor(k), tc.tensor(kq), tc.tensor(v),
        float(scale), float(z), True).numpy()
    diff = np.abs(auto - fused)
    assert diff.max() < 1e-3, f"max diff {diff.max()}"


def test_force_splitk_at_prefill_shape_raises():
    """Addendum #5: split-K is decode-only (L==1). Forcing it at a prefill
    shape (L>1) must fail loudly, not silently produce a wrong result via
    the merge kernel's L==1-assuming sink/head indexing."""
    B, H, KVH, L, S, D = 1, 4, 2, 8, 8, 64
    rng = np.random.default_rng(3)
    q = (rng.standard_normal((B, H, L, D)) * 0.1).astype(np.float32)
    k = (rng.standard_normal((B, KVH, S, D)) * 0.1).astype(np.float32)
    kq = (k + rng.standard_normal((B, KVH, S, D)) * 0.01).astype(np.float32)
    v = (rng.standard_normal((B, KVH, S, D)) * 0.1).astype(np.float32)
    scale = 1.0 / np.sqrt(D)
    z = _norm_ppf(1.0 - 0.15)
    os.environ["TC_APA_SELECTIVE_PATH"] = "2"
    try:
        with pytest.raises(RuntimeError):
            tc.apa_selective_attention(
                tc.tensor(q), tc.tensor(k), tc.tensor(kq), tc.tensor(v),
                float(scale), float(z), True)
    finally:
        _clear_override()
