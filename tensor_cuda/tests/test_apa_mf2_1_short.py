"""APA-MF2.1 item 1 (review F3): causal attention with S < L is REJECTED before any compute.

With bottom-right causal masking, query row i of L sees keys j <= S-L+i; for S < L rows
0..L-S-1 see nothing, softmax has no finite entry and the row is NaN (MF2 receipt:
reviewer `short` cell, artifacts/apa_mf2rev/gpu_short_20261007T210506Z_184542.json on
apa-mf2rev). The contract (review recommendation, matching kernels.cu causal_softmax
"S < L (queries outnumber keys)") is a ValueError naming S and L, raised before the
first tc.matmul, on every causal route in the writable files:
  functional.scaled_dot_product_attention (grad, no-grad, USE_FUSED_SOFTMAX),
  functional._causal_mask (the shared helper GRAPA/GRM add),
  quant.apa_quant_attention (refine 1.0 and 0.15).
Non-causal S < L is legal and still computes.

Prior art: FlashAttention-2 v2.1 bottom-right causal (Dao 2023) defines the empty-row
case (its kernel returns 0 there); this engine instead rejects, like its own CUDA
kernels (kernels.cu causal_softmax / APA throw on S < L). Test harness: mine.
"""
import contextlib

import numpy as np
import pytest

import tensor_cuda as tc
from tensor_cuda import functional as F
from tensor_cuda import quant as Q

DTYPES = ("float32", "bfloat16", "float16")
SHORT = [(8, 5), (3, 2), (2, 1)]
PATHS = ("sdpa_grad", "sdpa_infer", "sdpa_fused", "apa_p1", "apa_p015", "causal_mask")


@contextlib.contextmanager
def matmul_counter():
    calls = []
    orig = tc.matmul

    def mm(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    tc.matmul = mm
    try:
        yield calls
    finally:
        tc.matmul = orig


def _ops(L, S, dtype, D=64, seed=0):
    rng = np.random.default_rng(seed + 10 * L + S)
    return [tc.tensor(rng.normal(0, 1, (1, 2, n, D)).astype(np.float32), dtype=dtype) for n in (L, S, S)]


def _call(path, q, k, v):
    L, S = q.shape[-2], k.shape[-2]
    if path == "causal_mask":
        return F._causal_mask(L, S, "cuda", q.dtype)
    if path.startswith("sdpa"):
        saved = F.USE_FUSED_SOFTMAX
        F.USE_FUSED_SOFTMAX = path == "sdpa_fused"
        try:
            ctx = contextlib.nullcontext() if path == "sdpa_grad" else tc.no_grad()
            with ctx:
                return F.scaled_dot_product_attention(q, k, v, is_causal=True)
        finally:
            F.USE_FUSED_SOFTMAX = saved
    p = 1.0 if path == "apa_p1" else 0.15
    return Q.apa_quant_attention(q, k, v, bulk_bits=8, refine_percentile=p, is_causal=True)


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("L,S", SHORT)
def test_causal_s_less_than_l_raises_before_any_matmul(L, S, dtype, path):
    q, k, v = _ops(L, S, dtype)
    with matmul_counter() as calls:
        with pytest.raises(ValueError, match=r"S >= L"):
            _call(path, q, k, v)
    assert not calls, f"{len(calls)} tc.matmul call(s) before the S < L rejection"


@pytest.mark.parametrize("path", PATHS)
def test_message_names_both_lengths(path):
    q, k, v = _ops(8, 5, "float32")
    with pytest.raises(ValueError) as ei:
        _call(path, q, k, v)
    msg = str(ei.value)
    assert "S=5" in msg and "L=8" in msg, msg


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("p", [1.0, 0.15])
def test_noncausal_s_less_than_l_still_computes(dtype, p):
    q, k, v = _ops(8, 5, dtype)
    with tc.no_grad():
        o1 = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    o2 = Q.apa_quant_attention(q, k, v, bulk_bits=8, refine_percentile=p, is_causal=False)
    assert tuple(o1.shape) == (1, 2, 8, 64) and tuple(o2.shape) == (1, 2, 8, 64)
    assert np.isfinite(o1.float().numpy()).all() and np.isfinite(o2.float().numpy()).all()


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("L,S", [(5, 5), (3, 9), (1, 1)])
def test_causal_s_ge_l_unaffected(L, S, path):
    q, k, v = _ops(L, S, "float32")
    out = _call(path, q, k, v)
    assert np.isfinite(out.float().numpy()).all() if path != "causal_mask" else True
