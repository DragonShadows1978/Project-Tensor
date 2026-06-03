"""Numerical parity tests for the APA-Quant CUDA extension.

Compares the compiled extension (forward + the reference's approximate backward)
against a self-contained NumPy port of the algorithm. Run on a CUDA box after
``pip install -e .`` from ``apa_cuda/``::

    pytest apa_cuda/tests/test_parity.py        # or: python apa_cuda/tests/test_parity.py

Note on backward: APA's backward is deliberately an approximation -- it computes
gradients from the full-precision softmax weights, not from the quantized/mixed
scores used in the forward. So torch.autograd.gradcheck is *not* applicable; we
instead verify the extension reproduces the reference's backward exactly.
"""

import math
import sys

import numpy as np

try:
    import torch
    _HAVE_TORCH = True
except ImportError:
    _HAVE_TORCH = False

# Make the package importable when run as a script from the repo root.
sys.path.insert(0, __file__.rsplit("/", 2)[0])

from apa_attention.quant_tables import build_tables  # noqa: E402


# --------------------------------------------------------------------------- #
# NumPy reference (dense path), faithful to tensor_gpu_v2._core.apa_quant_attention
# --------------------------------------------------------------------------- #
def _norm_ppf(p):
    # scipy-free: SciPy is preferred for the true reference but is optional.
    try:
        from scipy.stats import norm
        return float(norm.ppf(p))
    except ImportError:
        # math.erfinv-based fallback (Python 3.8 lacks it; use a series).
        # Sufficient for tests when SciPy is unavailable.
        return math.sqrt(2.0) * _erfinv(2.0 * p - 1.0)


def _erfinv(x):
    a = 0.147
    ln = math.log(1 - x * x)
    t = 2.0 / (math.pi * a) + ln / 2.0
    return math.copysign(math.sqrt(math.sqrt(t * t - ln / a) - t), x)


def _quantize_keys(key, rotations, codebook, boundaries):
    kf = key.astype(np.float32)
    norms = np.sqrt(np.maximum(0.0, (kf * kf).sum(-1, keepdims=True)))
    safe = np.where(norms > 0, norms, 1.0)
    unit = np.where(norms > 0, kf / safe, 0.0)
    rotated = np.einsum("bhsd,hed->bhse", unit, rotations)
    idx = np.searchsorted(boundaries, rotated.ravel(), side="left").reshape(rotated.shape)
    centroids = codebook[idx]
    recon_unit = np.einsum("bhse,hed->bhsd", centroids, rotations)
    return recon_unit * norms


def _concentration(bulk):  # (B,H,L,S) -> (H,)
    m = bulk.max(-1, keepdims=True)
    e = np.exp(bulk - m)
    sm = e / e.sum(-1, keepdims=True)
    conc = sm.max(-1) / (sm.mean(-1) + 1e-10)
    return conc.mean(axis=(0, 2))


def _allocate_budget(conc, H, gb, minp=0.01, maxp=0.50):
    w = conc / (conc.sum() + 1e-30)
    alloc = np.clip(w * H * gb, minp, maxp)
    target = H * gb
    for _ in range(10):
        cur = float(alloc.sum())
        if abs(cur - target) < 1e-6:
            break
        free = ~((alloc <= minp + 1e-8) | (alloc >= maxp - 1e-8))
        fc = int(free.sum())
        if fc == 0:
            break
        adj = (target - cur) / fc
        alloc = np.clip(np.where(free, alloc + adj, alloc), minp, maxp)
    return alloc


def reference_forward(q, k, v, *, bulk_bits=2, refine_percentile=0.15,
                      is_causal=False, attn_mask=None, scale=None,
                      apa_rotation=True, adaptive_heads=False):
    B, H, L, D = q.shape
    S = k.shape[2]
    scale = scale if scale is not None else D ** -0.5
    refine_percentile = max(0.0, min(1.0, refine_percentile))
    refine_k = max(1, int(S * refine_percentile))
    rotations, codebook, boundaries = build_tables(D, bulk_bits, H, apa_rotation)

    full_precision = refine_percentile >= 1.0
    full_path = full_precision or (refine_k >= S)
    key_quant = k.astype(np.float32) if full_precision else _quantize_keys(
        k, rotations, codebook, boundaries)

    bulk = np.einsum("bhld,bhsd->bhls", q, key_quant) * scale
    ranking = np.einsum("bhld,bhsd->bhls", q, k) * scale
    cmask = None
    if is_causal:
        cmask = np.triu(np.ones((L, S), dtype=bool), k=1)
        bulk = np.where(cmask, -1e9, bulk)
        ranking = np.where(cmask, -1e9, ranking)
    if attn_mask is not None:
        bulk = bulk + attn_mask
        ranking = ranking + attn_mask

    adaptive = adaptive_heads and H > 1 and refine_k < S and not full_precision
    if full_path:
        mask = np.ones((B, H, L, S), dtype=bool)
    else:
        absr = np.abs(ranking)
        if cmask is not None:
            absr = np.where(cmask, 0.0, absr)
        if adaptive:
            pct = _allocate_budget(_concentration(bulk), H, refine_percentile)
            z = np.array([_norm_ppf(1.0 - p) for p in pct],
                         dtype=np.float32).reshape(1, H, 1, 1)
            thr = absr.mean(-1, keepdims=True) + z * absr.std(-1, keepdims=True)
        elif S > 256:
            z = _norm_ppf(1.0 - refine_percentile)
            thr = absr.mean(-1, keepdims=True) + z * absr.std(-1, keepdims=True)
        else:
            threshold_idx = S - refine_k
            part = np.partition(absr, threshold_idx, axis=-1)
            thr = part[:, :, :, threshold_idx:threshold_idx + 1]
        mask = absr >= thr

    scores = np.where(mask, ranking, bulk)
    sm = scores.max(-1, keepdims=True)
    es = np.exp(scores - sm)
    attn_w = es / es.sum(-1, keepdims=True)
    out = np.einsum("bhls,bhsd->bhld", attn_w, v)
    return out, key_quant, mask


def reference_backward(grad_out, q, k, v, key_quant, mask, *, scale=None,
                       is_causal=False, attn_mask=None):
    B, H, L, D = q.shape
    S = k.shape[2]
    scale = scale if scale is not None else D ** -0.5
    scores = np.einsum("bhld,bhsd->bhls", q, k) * scale
    cmask = None
    if is_causal:
        cmask = np.triu(np.ones((L, S), dtype=bool), k=1)
        scores = np.where(cmask, -1e9, scores)
    if attn_mask is not None:
        scores = scores + attn_mask
    sm = scores.max(-1, keepdims=True)
    es = np.exp(scores - sm)
    attn_w = es / es.sum(-1, keepdims=True)

    grad_v = np.einsum("bhls,bhld->bhsd", attn_w, grad_out)
    dAttn = np.einsum("bhld,bhsd->bhls", grad_out, v)
    sum_dA = (attn_w * dAttn).sum(-1, keepdims=True)
    dS = attn_w * (dAttn - sum_dA) * scale
    if cmask is not None:
        dS = np.where(cmask, 0.0, dS)
    dS_full = np.where(mask, dS, 0.0)
    dS_bulk = np.where(mask, 0.0, dS)
    grad_q = np.einsum("bhls,bhsd->bhld", dS_full, k) + \
        np.einsum("bhls,bhsd->bhld", dS_bulk, key_quant)
    grad_k = np.einsum("bhls,bhld->bhsd", dS, q)
    return grad_q, grad_k, grad_v


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def _run_case(B, H, L, S, D, *, is_causal=False, refine_percentile=0.15,
              bulk_bits=2, adaptive_heads=False, seed=0, tol=None):
    # cuBLAS vs NumPy matmul differ by ~1e-5; near the refinement threshold that
    # can flip a single mask bit, swapping one position between full-precision
    # and quantized scores. The output impact scales with quantization coarseness,
    # so the tolerance is bit-width aware (full-precision stays tight).
    if tol is None:
        tol = {1: 2e-2, 2: 1e-2, 4: 5e-3, 8: 3e-3}.get(bulk_bits, 1e-2)
        if refine_percentile >= 1.0:
            tol = 1e-4
    import apa_attention_cuda as _C  # noqa: F401  (ensures built)
    from apa_attention import apa_quant_attention

    rng = np.random.default_rng(seed)
    q_np = rng.standard_normal((B, H, L, D)).astype(np.float32)
    k_np = rng.standard_normal((B, H, S, D)).astype(np.float32)
    v_np = rng.standard_normal((B, H, S, D)).astype(np.float32)

    out_ref, key_quant, mask = reference_forward(
        q_np, k_np, v_np, bulk_bits=bulk_bits,
        refine_percentile=refine_percentile, is_causal=is_causal,
        adaptive_heads=adaptive_heads)

    dev = "cuda"
    q = torch.tensor(q_np, device=dev, requires_grad=True)
    k = torch.tensor(k_np, device=dev, requires_grad=True)
    v = torch.tensor(v_np, device=dev, requires_grad=True)
    out = apa_quant_attention(q, k, v, bulk_bits=bulk_bits,
                              refine_percentile=refine_percentile,
                              is_causal=is_causal,
                              adaptive_heads=adaptive_heads)
    out_np = out.detach().cpu().numpy()
    fwd_err = np.abs(out_np - out_ref).max()
    assert fwd_err < tol, f"forward mismatch {fwd_err:.2e} (L={L},S={S},causal={is_causal})"

    g_np = rng.standard_normal(out_np.shape).astype(np.float32)
    out.backward(torch.tensor(g_np, device=dev))
    gq_ref, gk_ref, gv_ref = reference_backward(
        g_np, q_np, k_np, v_np, key_quant, mask,
        is_causal=is_causal)
    for name, got, ref in (("grad_q", q.grad, gq_ref),
                           ("grad_k", k.grad, gk_ref),
                           ("grad_v", v.grad, gv_ref)):
        err = np.abs(got.detach().cpu().numpy() - ref).max()
        assert err < tol, f"{name} mismatch {err:.2e} (L={L},S={S},causal={is_causal})"
    return fwd_err


def test_dense_topk():
    _run_case(2, 4, 64, 64, 32)


def test_dense_topk_causal():
    _run_case(2, 4, 64, 64, 32, is_causal=True)


def test_dense_zscore_long():
    _run_case(1, 2, 320, 320, 32)  # S > 256 -> z-score path


def test_full_precision():
    _run_case(2, 4, 48, 48, 32, refine_percentile=1.0)


def test_bits_variants():
    for bits in (1, 2, 4, 8):
        _run_case(1, 2, 64, 64, 32, bulk_bits=bits)


def test_adaptive_heads():
    _run_case(2, 8, 64, 64, 32, adaptive_heads=True)


if __name__ == "__main__":
    if not _HAVE_TORCH or not torch.cuda.is_available():
        print("SKIP: CUDA-enabled PyTorch required.")
        sys.exit(0)
    for fn in (test_dense_topk, test_dense_topk_causal, test_dense_zscore_long,
               test_full_precision, test_bits_variants, test_adaptive_heads):
        fn()
        print(f"PASS {fn.__name__}")
    print("All parity tests passed.")
