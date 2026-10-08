"""APA-MF2 item 2: masked keys get exactly zero weight for every finite score.

The old bias was np.full(-1e4) (functional.py:38), added to the scores, so a masked
score s with s - 1e4 above the visible scores leaked (review N1: K7 = 80008 e0
gives row 0 = 0.2797 instead of 0). The contract under test:
  * F._causal_mask(L, S, ...) is 0 on visible (j <= S-L+i, bottom-right) and -inf
    on masked entries, in fp32/bf16/fp16;
  * SDPA (grad on and off) and apa_quant_attention (refine 1.0 and 0.15) give rows
    that cannot see a key EXACTLY 0.0 output when V is nonzero only on that key,
    for every finite score magnitude up to the dtype maximum (refine < 1 is
    restricted to |K| whose squared norm is finite in fp32: the quantizer computes
    |k|^2, so larger keys give a non-finite bulk score, outside "finite score");
  * realistic inputs: outputs (and SDPA grads) are byte-identical to the BEFORE
    code, loaded from files, never the new code: functional.py from
    artifacts/apa_mf2/before/mf1_AB; for apa_quant_attention at refine 1.0 the MF1
    quant.py (no selection runs), at refine 0.15 the item-1-only snapshot
    artifacts/apa_mf2/variants/quant_item1.py (MF2 selection, old additive mask
    application), so the arms differ only in the item-2 mask code.

Prior art: FlashAttention-2 (Dao 2023) predicate masking with -inf; PyTorch SDPA
boolean attn_mask semantics (masked -> -inf before softmax); 63960a4 bottom-right.
Harness and the sweep: mine.
"""
import contextlib
import importlib.util
import os

import numpy as np
import pytest

import tensor_cuda as tc
from tensor_cuda import functional as F
from tensor_cuda import quant

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BEFORE_FUNCTIONAL = os.path.join(ROOT, "artifacts/apa_mf2/before/mf1_AB/tensor_cuda__tensor_cuda__functional.py")
# APA-MF2.1: the refine-0.15 arm's selection moved to the definition (MF2.1 item 3), so
# its "before" is now the MF2.1 quant.py with only the mask application reverted to the
# additive form (artifacts/apa_mf2_1/variants/quant_defn_additive_mask.py); MF2's
# variants/quant_item1.py carried MF2's one-pass selection. Same isolation as before.
BEFORE_QUANT = {1.0: os.path.join(ROOT, "artifacts/apa_mf2/before/mf1_AB/tensor_cuda__tensor_cuda__quant.py"),
                0.15: os.path.join(ROOT, "artifacts/apa_mf2_1/variants/quant_defn_additive_mask.py")}
DTYPES = ("float32", "bfloat16", "float16")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _load_before():
    spec = importlib.util.spec_from_file_location("tensor_cuda._mf2_before_functional", BEFORE_FUNCTIONAL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@contextlib.contextmanager
def functional_arm(arm):
    """arm 'after' = the tree; 'before' = the before copy's SDPA and _causal_mask."""
    if arm == "after":
        yield
        return
    b = _load_before()
    saved = (F._causal_mask, F.scaled_dot_product_attention)
    F._causal_mask, F.scaled_dot_product_attention = b._causal_mask, b.scaled_dot_product_attention
    try:
        yield
    finally:
        F._causal_mask, F.scaled_dot_product_attention = saved


def visible(L, S):
    return np.arange(S)[None, :] <= (S - L + np.arange(L))[:, None]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("L,S", [(1, 1), (1, 7), (3, 3), (3, 4), (3, 17), (9, 9), (9, 41)])
def test_causal_mask_is_zero_or_minus_inf_bottom_right(L, S, dtype):
    m = F._causal_mask(L, S, "cuda", dtype).float().numpy()
    vis = visible(L, S)
    assert m.shape == (L, S)
    assert np.all(m[vis] == 0.0)
    assert np.all(np.isneginf(m[~vis]))


def _leak_inputs(L, S, D, key, M, dtype):
    q = np.zeros((1, 1, L, D), np.float32); q[..., 0] = 1.0
    k = np.zeros((1, 1, S, D), np.float32); k[0, 0, key, 0] = M
    v = np.zeros((1, 1, S, D), np.float32); v[0, 0, key, :] = 1.0
    return [tc.tensor(a, dtype=dtype) for a in (q, k, v)]


def _run(path, q, k, v):
    if path == "sdpa_nograd":
        with tc.no_grad():
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0)
    if path == "sdpa_grad":
        assert tc.is_grad_enabled()
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0)
    p = 1.0 if path == "apa_p1" else 0.15
    return quant.apa_quant_attention(q, k, v, bulk_bits=8, refine_percentile=p,
                                     is_causal=True, scale=1.0)


SWEEP = {"float32": (1e3, 9.9e3, 1e4, 1.0001e4, 2e4, 1e5, 1e6, 1e10, 1e20, 1e30),
         "bfloat16": (1e3, 1e4, 1e5, 1e10, 1e30, 3.3e38),
         "float16": (1e3, 1e4, 2e4, 6e4, 65504.0)}
LEAK_CASES = [(dt, M, path, shape) for dt, Ms in SWEEP.items() for M in Ms
              for path in ("sdpa_nograd", "sdpa_grad", "apa_p1", "apa_p015")
              for shape in ((3, 9), (9, 9))
              # refine < 1 quantizes K: keep |K|^2 finite in fp32 and leave 25% headroom
              # under the dtype max for the reconstructed key (else the bulk score is
              # itself non-finite -- outside "every finite score"; the gate script
              # still runs and reports those cells)
              if not (path == "apa_p015" and (M * M >= 3.4e38 or 1.25 * M >= {"float32": 3.4e38,
                      "bfloat16": 3.38e38, "float16": 65504.0}[dt]))]


@pytest.mark.parametrize("dtype,M,path,shape", LEAK_CASES)
def test_masked_key_gets_exactly_zero_weight(dtype, M, path, shape):
    L, S = shape
    key = 7
    q, k, v = _leak_inputs(L, S, 64, key, M, dtype)
    out = _run(path, q, k, v).float().numpy()[0, 0]
    blind = np.arange(L) < key - (S - L)           # rows that cannot see `key`
    assert blind.any()
    assert np.all(out[blind] == 0.0), f"leak: rows {np.where(blind)[0].tolist()} max {np.abs(out[blind]).max()}"
    assert np.isfinite(out).all(), "non-finite output on an S >= L case"


def test_reviewer_n1_case_row0_exactly_zero():
    """Review N1, verbatim: fp32 L3 S9 D64, refine 1, default scale 1/8."""
    q = np.zeros((1, 1, 3, 64), np.float32); q[..., 0] = 1
    k = np.zeros((1, 1, 9, 64), np.float32); k[0, 0, 7, 0] = 80008
    v = np.zeros_like(k); v[0, 0, 7, :] = 1
    out = quant.apa_quant_attention(*(tc.tensor(a) for a in (q, k, v)), bulk_bits=8,
                                    refine_percentile=1.0, is_causal=True).numpy()
    assert np.all(out[0, 0, 0] == 0.0), out[0, 0, 0, :4]
    # rows 1, 2 see key 7 with score 10001: weight 1/(1+7e^-10001) == 1.0 in fp32
    np.testing.assert_allclose(out[0, 0, 1:], 1.0, atol=2e-5)


def _realistic(L, S, D, dtype, seed):
    rng = np.random.default_rng(seed)
    return [rng.normal(0, 1, (1, 2, n, D)).astype(np.float32) for n in (L, S, S)]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("L,S", [(3, 3), (3, 17), (64, 64), (64, 261)])
def test_sdpa_byte_identical_to_before_on_realistic_inputs(L, S, dtype):
    arrs = _realistic(L, S, 64, dtype, 11 + L + S)
    res = {}
    for arm in ("before", "after"):
        with functional_arm(arm):
            with tc.no_grad():
                o_ng = F.scaled_dot_product_attention(*(tc.tensor(a, dtype=dtype) for a in arrs), is_causal=True)
            # fp32 leaves cast to the dtype (tc.tensor makes bf16 a non-leaf cast anyway)
            leaves = [tc.tensor(a, requires_grad=True) for a in arrs]
            ts = [x if dtype == "float32" else x.astype(dtype) for x in leaves]
            o = F.scaled_dot_product_attention(*ts, is_causal=True)
            o.sum().backward()
            res[arm] = [o_ng.float().numpy(), o.float().numpy()] + [x.grad.float().numpy() for x in leaves]
    for b, a in zip(res["before"], res["after"]):
        assert np.isfinite(a).all()
        assert b.tobytes() == a.tobytes()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("p", [1.0, 0.15])
@pytest.mark.parametrize("L,S", [(3, 17), (64, 64), (64, 261)])
def test_quant_byte_identical_to_before_mask(L, S, p, dtype):
    """Before = before functional.py + before quant file; after = the tree."""
    arrs = _realistic(L, S, 64, dtype, 5 + L + S)
    res = {}
    for arm in ("before", "after"):
        fn = quant.apa_quant_attention if arm == "after" else \
            _load("tensor_cuda._mf2_before_quant_" + ("p1" if p == 1.0 else "p015"), BEFORE_QUANT[p]).apa_quant_attention
        with functional_arm(arm):
            res[arm] = fn(*(tc.tensor(a, dtype=dtype) for a in arrs), bulk_bits=8,
                                                 refine_percentile=p, is_causal=True).float().numpy()
    assert np.isfinite(res["after"]).all()
    assert res["before"].tobytes() == res["after"].tobytes()
