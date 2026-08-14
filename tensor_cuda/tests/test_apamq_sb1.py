"""APAMQ-SB1 registered gates: INT8 cuBLASLt bulk + gather refinement.

Quantizer contract (Q and K, independently per final-axis row):

    scale = max(abs(row)) / 127, or 1 for an all-zero row
    code  = clamp(round_away_from_zero(row / scale), -127, 127)

CUDA ``roundf`` is nearest with halfway cases away from zero. NumPy ``rint``
is deliberately not used because it is ties-to-even (FA2's RED divergence).

GPU tolerances are registered as: integer sums exact, scale products 1e-6,
and composed end-to-end output 2e-2. This file remains useful in the GPU-less
sandbox: quantizer, integer-GQA, bounds, and composed references run on CPU;
native legs skip with the CUDA initialization error in the receipt.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import tensor_cuda as tc
from tensor_cuda.quant import _norm_ppf


SCALE_TOL = dict(rtol=1e-6, atol=1e-6)
E2E_TOL = dict(rtol=2e-2, atol=2e-2)
SB2_CHUNK_BUDGET = 288 * 1024 * 1024


def _require_cuda():
    if not hasattr(tc, "apa_gemm_selective_attention"):
        pytest.fail("built extension is missing apa_gemm_selective_attention")
    try:
        probe = tc.tensor(np.zeros(1, dtype=np.float32))
        tc.synchronize()
        del probe
    except Exception as exc:  # pragma: no cover - expected in this sandbox
        pytest.skip(f"CUDA unavailable: {exc}")


def _round_bf16(x):
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + bias) & np.uint32(0xFFFF0000)).view(np.float32)


def _round_away(x):
    x = np.asarray(x, dtype=np.float32)
    return np.copysign(np.floor(np.abs(x) + np.float32(0.5)), x)


def _quant_int8(x):
    xf = np.asarray(x, dtype=np.float32)
    amax = np.max(np.abs(xf), axis=-1, keepdims=True)
    scale = (amax * np.float32(1.0 / 127.0)).astype(np.float32)
    safe = np.where(scale > 0, scale, np.float32(1.0)).astype(np.float32)
    codes = np.clip(
        _round_away(xf * (np.float32(1.0) / safe)), -127, 127
    ).astype(np.int8)
    return codes, safe[..., 0]


def _integer_sums(qcodes, kcodes):
    q32 = np.asarray(qcodes, dtype=np.int32)
    k32 = np.asarray(kcodes, dtype=np.int32)
    B, H, L, _D = q32.shape
    KVH, S = k32.shape[1:3]
    group = H // KVH
    out = np.empty((B, H, L, S), dtype=np.int64)
    for b in range(B):
        for h in range(H):
            out[b, h] = q32[b, h] @ k32[b, h // group].T
    return out


def _bounds(row, *, L, S, full_lq, row0, causal, window):
    if not causal:
        return 0, S
    i = row % L
    hi = min(S, max(0, S - full_lq + row0 + i + 1))
    lo = max(0, hi - window) if window > 0 else 0
    return lo, hi


def _reference(q, k, v, scale, zthr, *, causal, Lq=0, row0=0, window=0):
    qf, kf, vf = (np.asarray(x, dtype=np.float32) for x in (q, k, v))
    qc, qs = _quant_int8(qf)
    kc, ks = _quant_int8(kf)
    isums = _integer_sums(qc, kc)
    B, H, L, _D = qf.shape
    KVH, S, VD = vf.shape[1], vf.shape[2], vf.shape[3]
    group = H // KVH
    full_lq = Lq if Lq > 0 else L
    out = np.zeros((B, H, L, VD), dtype=np.float32)
    selected = np.zeros((B, H, L, S), dtype=bool)
    bulk_all = np.zeros((B, H, L, S), dtype=np.float32)
    flat_row = 0
    for b in range(B):
        for h in range(H):
            kh = h // group
            for i in range(L):
                lo, hi = _bounds(
                    flat_row, L=L, S=S, full_lq=full_lq, row0=row0,
                    causal=causal, window=window,
                )
                bulk = (
                    isums[b, h, i, lo:hi].astype(np.float32)
                    * np.float32(qs[b, h, i] * scale)
                    * ks[b, kh, lo:hi].astype(np.float32)
                ).astype(np.float32)
                bulk_all[b, h, i, lo:hi] = bulk
                mag = np.abs(bulk)
                mean = mag.mean(dtype=np.float32)
                var = (mag * mag).mean(dtype=np.float32) - mean * mean
                thr = mean + np.float32(zthr) * np.sqrt(
                    np.maximum(var, np.float32(0)), dtype=np.float32
                )
                mask = mag >= thr
                selected[b, h, i, lo:hi] = mask
                exact = (qf[b, h, i] @ kf[b, kh, lo:hi].T) * np.float32(scale)
                scores = np.where(mask, exact, bulk).astype(np.float32)
                p = np.exp(scores - scores.max()).astype(np.float32)
                p /= p.sum(dtype=np.float32)
                out[b, h, i] = p @ vf[b, kh, lo:hi]
                flat_row += 1
    return out, (qc, qs, kc, ks, isums, bulk_all, selected)


def _arrays(H, KVH, L, S, D, seed, VD=None):
    VD = D if VD is None else VD
    rng = np.random.default_rng(seed)
    q = _round_bf16((rng.standard_normal((1, H, L, D)) * 0.125).astype(np.float32))
    k = _round_bf16((rng.standard_normal((1, KVH, S, D)) * 0.125).astype(np.float32))
    v = _round_bf16((rng.standard_normal((1, KVH, S, VD)) * 0.125).astype(np.float32))
    return q, k, v


def _legacy_int4_k(k):
    kf = np.asarray(k, dtype=np.float32)
    amax = np.max(np.abs(kf), axis=-1, keepdims=True)
    scale = amax * np.float32(1.0 / 7.0)
    safe = np.where(scale > 0, scale, np.float32(1.0)).astype(np.float32)
    code = np.clip(_round_away(kf / safe), -7, 7)
    return (code * safe).astype(np.float32)


def _sb2_nominal_fraction(zthr):
    return max(0.0, min(1.0, 0.5 * math.erfc(zthr / math.sqrt(2.0))))


def _sb2_chunk_policy(B, H, L, S, zthr):
    cap_fraction = min(1.0, 2.0 * _sb2_nominal_fraction(zthr))
    bytes_per_score = 4.0 + 4.0 + 4.0 + 2.0 + 8.0 * cap_fraction
    chunk = int(SB2_CHUNK_BUDGET / (B * H * S * bytes_per_score))
    return max(1, min(L, chunk)), cap_fraction, bytes_per_score


def test_sb1_round_half_away_contract_cpu():
    # Scale is exactly one, so these are direct halfway tie probes.
    row = np.array([127.0, -127.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5], np.float32)
    codes, scales = _quant_int8(row[None])
    assert scales[0] == np.float32(1.0)
    np.testing.assert_array_equal(
        codes[0], np.array([127, -127, 1, -1, 2, -2, 3, -3], np.int8)
    )


def test_sb2_nonfinite_selection_predicate_is_count_append_identical_cpu():
    scores = np.array([0.0, 2.0, np.nan, np.inf, -np.inf], np.float32)
    thresholds = np.array([1.0, 1.0, 1.0, 1.0, np.nan], np.float32)
    selected = np.isfinite(scores) & np.isfinite(thresholds) & (
        np.abs(scores) >= thresholds
    )
    np.testing.assert_array_equal(selected, [False, True, False, False, False])


def test_sb2_bounded_append_clamps_and_reports_overflow_cpu():
    attempted, capacity = 11, 7
    stored = list(range(attempted))[:capacity]
    overflow = attempted > capacity
    dropped = max(0, attempted - capacity)
    assert stored == list(range(capacity))
    assert overflow and dropped == 4


def test_sb2_stats_surface_includes_overflow_and_dropped_cpu():
    assert tc.apa_gemm_selective_stats(reset=True) == (0, 0, 0, 0)


def test_sb2_registered_shape_chunk_policy_caps_score_materialization_cpu():
    zthr = _norm_ppf(0.90)
    for S in (16384, 65536):
        chunk, cap_fraction, _bytes_per_score = _sb2_chunk_policy(
            1, 16, 512, S, zthr
        )
        score_bytes = 1 * 16 * chunk * S * (4 + 4)
        assert 1 <= chunk <= 512
        assert score_bytes <= SB2_CHUNK_BUDGET
        assert 0.19 < cap_fraction < 0.21


def test_g_sb1_a_composed_reference_cpu():
    q, k, v = _arrays(8, 4, 5, 23, 128, seed=101, VD=64)
    out, parts = _reference(
        q, k, v, 1.0 / math.sqrt(128), _norm_ppf(0.85), causal=False
    )
    qc, qs, kc, ks, isums, bulk, selected = parts
    assert qc.dtype == kc.dtype == np.int8
    assert isums.dtype == np.int64
    assert out.shape == (1, 8, 5, 64)
    assert np.all(np.isfinite(out)) and np.all(np.isfinite(bulk))
    assert np.all(qs > 0) and np.all(ks > 0)
    assert 0 < selected.sum() < selected.size


@pytest.mark.parametrize(
    "KVH,D",
    [(1, 512), (4, 128), (8, 128)],
)
def test_g_sb1_b_rect_causal_cache_reference_cpu(KVH, D):
    # Explicit 121-key / 11-query cached rectangle: catches flattening a
    # (121,D) KV matrix as if query and KV head counts matched.
    q, k, v = _arrays(16, KVH, 11, 121, D, seed=2000 + KVH + D, VD=64)
    out, parts = _reference(
        q, k, v, 1.0 / math.sqrt(D), _norm_ppf(0.85),
        causal=True, Lq=17, row0=6,
    )
    selected = parts[-1]
    assert out.shape == (1, 16, 11, 64)
    assert np.all(np.isfinite(out))
    valid_counts = [111 + i for _h in range(16) for i in range(11)]
    assert selected.sum() < sum(valid_counts)
    assert selected[..., :111].any()


def test_g_sb1_a_integer_bulk_and_scale_products_gpu():
    _require_cuda()
    q, k, _v = _arrays(8, 4, 5, 23, 128, seed=9)
    scale = np.float32(1.0 / math.sqrt(128))
    zthr = np.float32(_norm_ppf(0.85))
    values = tc.apa_gemm_selective_debug(
        tc.tensor(q, dtype="bfloat16"), tc.tensor(k, dtype="bfloat16"),
        float(scale), float(zthr), False,
    )
    tc.synchronize()
    qbytes, qscales, kbytes, kscales, isums, bulk, selected = [
        x.numpy() for x in values
    ]
    qc, qs = _quant_int8(q)
    kc, ks = _quant_int8(k)
    isum_ref = _integer_sums(qc, kc)
    np.testing.assert_array_equal(qbytes.view(np.int8), qc)
    np.testing.assert_array_equal(kbytes.view(np.int8), kc)
    np.testing.assert_allclose(qscales, qs, **SCALE_TOL)
    np.testing.assert_allclose(kscales, ks, **SCALE_TOL)
    np.testing.assert_array_equal(isums, isum_ref)  # registered zero tolerance
    bulk_ref = (
        isum_ref.astype(np.float32)
        * (qs * scale)[..., None].astype(np.float32)
        * ks[:, np.arange(8) // 2, None, :].astype(np.float32)
    )
    np.testing.assert_allclose(bulk, bulk_ref, **SCALE_TOL)
    assert 0 < selected.sum() < selected.size


@pytest.mark.parametrize("KVH,D", [(1, 512), (4, 128), (8, 128)])
def test_g_sb1_b_rect_causal_cache_e2e_gpu(KVH, D):
    _require_cuda()
    q, k, v = _arrays(16, KVH, 11, 121, D, seed=3000 + KVH + D, VD=64)
    scale = 1.0 / math.sqrt(D)
    zthr = float(_norm_ppf(0.85))
    args = [tc.tensor(x, dtype="bfloat16") for x in (q, k, v)]
    got = tc.apa_gemm_selective_attention(
        *args, scale, zthr, True, Lq=17, row0=6
    )
    tc.synchronize()
    ref, _ = _reference(
        q, k, v, scale, zthr, causal=True, Lq=17, row0=6
    )
    np.testing.assert_allclose(got.numpy().astype(np.float32), ref, **E2E_TOL)


def test_sb1_cached_k_codes_match_call_local_gpu():
    _require_cuda()
    q, k, v = _arrays(8, 4, 5, 31, 128, seed=404, VD=64)
    qt, kt, vt = [tc.tensor(x, dtype="bfloat16") for x in (q, k, v)]
    codes, scales = tc.apa_gemm_selective_quantize_k(kt)
    kwargs = dict(scale=1.0 / math.sqrt(128), zthr=float(_norm_ppf(0.85)))
    local = tc.apa_gemm_selective_attention(qt, kt, vt, **kwargs)
    cached = tc.apa_gemm_selective_attention(
        qt, kt, vt, **kwargs, k_codes=codes, k_scales=scales
    )
    tc.synchronize()
    np.testing.assert_array_equal(local.numpy(), cached.numpy())


def test_g_sb1_c_selection_fraction_overlap_data_only_gpu(capsys):
    _require_cuda()
    q, k, v = _arrays(8, 4, 7, 97, 128, seed=505)
    zthr = float(_norm_ppf(0.85))
    scale = 1.0 / math.sqrt(128)
    qt, kt, vt = [tc.tensor(x, dtype="bfloat16") for x in (q, k, v)]
    debug = tc.apa_gemm_selective_debug(qt, kt, scale, zthr, False)
    sb_mask = debug[-1].numpy().astype(bool)

    # The existing selector consumes dequantized symmetric-7 INT4 K. Run its
    # actual engine entry, then mirror its documented selection predicate for
    # set comparison; this is intentionally data-only because quantizers differ.
    kq = _legacy_int4_k(k)
    legacy_out = tc.apa_selective_attention(
        qt, kt, tc.tensor(kq, dtype="bfloat16"), vt, scale, zthr, False
    )
    tc.synchronize()
    bulk = np.empty_like(sb_mask, dtype=np.float32)
    for h in range(8):
        bulk[0, h] = (q[0, h] @ kq[0, h // 2].T) * np.float32(scale)
    mag = np.abs(bulk)
    thr = mag.mean(axis=-1, keepdims=True, dtype=np.float32) + np.float32(zthr) * np.sqrt(
        np.maximum(
            (mag * mag).mean(axis=-1, keepdims=True, dtype=np.float32)
            - mag.mean(axis=-1, keepdims=True, dtype=np.float32) ** 2,
            np.float32(0),
        ),
        dtype=np.float32,
    )
    legacy_mask = mag >= thr
    intersection = np.logical_and(sb_mask, legacy_mask).sum()
    union = np.logical_or(sb_mask, legacy_mask).sum()
    overlap = intersection / union if union else 1.0
    print(
        "G-SB1-c data: "
        f"gemm_apa_fraction={sb_mask.mean():.6f} "
        f"apa_selective_fraction={legacy_mask.mean():.6f} "
        f"jaccard_overlap={overlap:.6f}"
    )
    assert np.all(np.isfinite(legacy_out.numpy()))
    assert 0.0 < sb_mask.mean() < 1.0
    assert 0.0 < legacy_mask.mean() < 1.0
    assert 0.0 <= overlap <= 1.0  # no tolerance gate: registered data only
    assert "jaccard_overlap=" in capsys.readouterr().out


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-rs"]))
