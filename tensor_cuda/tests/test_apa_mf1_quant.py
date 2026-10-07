"""APA-MF1: tensor_cuda.quant.apa_quant_attention on rectangular causal shapes.

Fix A (alignment): the refine-STATISTICS mask must use the same bottom-right
convention as the score mask F._causal_mask (row i sees cols <= S-L+i).
Fix B (stats): the threshold statistics are the population mean/std over the
VISIBLE keys only (fused-kernel rule, kernels.cu bk4 `count = S-L+qi+tid+1`).

APA_MF1_RULE selects the stats rule the module is expected to implement:
  "A" -> diluted stats (masked zeros counted over all S), bottom-right aligned;
  "B" -> visible-count stats (default; the final A+B tree).
The fp64 reference (scripts/apa_mf1_ref.py) is written from the definition; it
never calls tensor_gpu_v2 or the code under test. The selection rule is applied
to the code's own |bulk| (captured as the code saw it; an operand of the rule,
sanity-checked against fp64 q.kq). Entries within 1e-5 relative of the fp64
threshold are fp ties: the code's choice is used for the output reference and
their count is printed; every non-tie visible entry must agree exactly.

Prior art: FlashAttention-2 bottom-right causal (Dao, 2023); Project-Tensor
63960a4 (2026-06-09) F._causal_mask; kernels.cu bk4 visible-count stats; GRAPA
attention.py _visible_counts. Test harness itself: no prior art known to me.
"""
from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import apa_mf1_ref as R  # noqa: E402

import tensor_cuda as tc  # noqa: E402
from tensor_cuda import quant as Q  # noqa: E402

RULE = os.environ.get("APA_MF1_RULE", "B")
BEFORE = ROOT / "artifacts/apa_mf1/before/tensor_cuda__tensor_cuda__quant.py"
H, D, P = 2, 32, 0.15
SHAPES = [(L, S) for L in (1, 7, 64, 512) for S in (L, L + 1, L + 100, 4 * L + 3)]
CASES = [("float32", 2), ("float32", 8), ("bfloat16", 8)]
TOL = {"float32": 1e-5, "bfloat16": 2e-2}
TIE_REL = 1e-5


def _load_before():
    spec = importlib.util.spec_from_file_location("tensor_cuda._mf1_quant_before", BEFORE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@contextlib.contextmanager
def capture():
    """Observe (never alter) the stats-mask fill and the refine selection."""
    rec = {}
    orig_mf, orig_where = tc.Tensor.masked_fill, tc.where

    def mf(self, m, v):
        rec["absr"] = self.float().numpy().astype(np.float64)
        rec["statmask"] = m.float().numpy()
        return orig_mf(self, m, v)

    def wh(c, x, y):
        rec["refine"] = c.float().numpy()
        return orig_where(c, x, y)

    tc.Tensor.masked_fill, tc.where = mf, wh
    try:
        yield rec
    finally:
        tc.Tensor.masked_fill, tc.where = orig_mf, orig_where


def operands(L, S, dtype, seed):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((1, H, L, D)).astype(np.float32)
    k = rng.standard_normal((1, H, S, D)).astype(np.float32)
    v = rng.standard_normal((1, H, S, D)).astype(np.float32)
    ts = [tc.tensor(a, dtype=dtype) for a in (q, k, v)]
    exact = [t.float().numpy().astype(np.float64) for t in ts]
    return ts, exact


def run(fn, L, S, dtype, bits):
    ts, (q, k, v) = operands(L, S, dtype, 1000 * L + S)
    with capture() as rec:
        out = fn(*ts, bulk_bits=bits, refine_percentile=P, is_causal=True)
    R_, CB, BND = Q._tables(D, bits, H, True, "cuda")
    kq = Q._quantize_keys(ts[1], R_, CB, BND).float().numpy().astype(np.float64)
    return out.float().numpy(), rec, (q, k, kq, v)


def check(out, rec, ops, L, S, dtype, rule):
    q, k, kq, v = ops
    scale = 1.0 / np.sqrt(D)
    vis = R.visible_mask(L, S)
    assert "absr" in rec and "refine" in rec, "stats mask / selection not observed"
    absr = rec["absr"]
    # operand sanity: the captured |bulk| is the code's own q.kq (within dtype rounding)
    bulk64 = np.abs(np.einsum("bhld,bhsd->bhls", q, kq) * scale)
    op_rel, _ = R.rel_err(np.where(vis, absr, 0), np.where(vis, bulk64, 0))
    assert op_rel <= (1e-5 if dtype == "float32" else 1e-2), f"operand drift {op_rel:.2e}"
    z = Q._norm_ppf(1.0 - P)
    if rule == "A":
        sel_ref, thr = R.diluted_rule_select(absr, vis, z)
    else:
        sel_ref, thr = R.fused_rule_select(absr, vis, z)
    code_sel = rec["refine"] != 0
    amb = R.ambiguous(absr, thr, TIE_REL) & vis
    dis = vis & ~amb & (code_sel != sel_ref)
    n_vis = int(vis.sum()) * q.shape[0] * q.shape[1]
    agree = 1.0 - float(((code_sel != sel_ref) & vis).sum()) / n_vis
    sel_use = np.where(amb, code_sel, sel_ref) & vis
    ref = R.apa_tc_reference(q, k, kq, v, scale, sel_use)
    rel, mae = R.rel_err(out, ref)
    elig = vis & np.broadcast_to(rec["statmask"] == 0, code_sel.shape)
    elig_frac = float(elig.sum()) / n_vis
    msg = (f"L={L} S={S} {dtype} rule={rule}: nontie_disagree={int(dis.sum())} ties={int(amb.sum())} "
           f"agree={agree:.6f} eligible_frac={elig_frac:.6f} rel={rel:.3e} maxabs={mae:.3e}")
    print(msg)
    assert int(dis.sum()) == 0 and rel <= TOL[dtype], msg


@pytest.mark.parametrize("dtype,bits", CASES)
@pytest.mark.parametrize("L,S", SHAPES)
def test_mf1_quant_causal_reference(L, S, dtype, bits):
    out, rec, ops = run(Q.apa_quant_attention, L, S, dtype, bits)
    check(out, rec, ops, L, S, dtype, RULE)


@pytest.mark.parametrize("L,S", SHAPES)
def test_mf1_quant_eligibility_is_bottom_right(L, S):
    """Every visible key enters the statistics; every masked key is zeroed."""
    _, rec, _ = run(Q.apa_quant_attention, L, S, "float32", 8)
    vis = R.visible_mask(L, S)
    stat = np.broadcast_to(rec["statmask"] != 0, (1, H, L, S))
    excluded_visible = int((stat & vis).sum())
    kept_masked = int((~stat & ~vis).sum())
    n_vis = int(vis.sum()) * H
    print(f"L={L} S={S}: eligible_frac={1 - excluded_visible / n_vis:.6f} "
          f"excluded_visible={excluded_visible} kept_masked={kept_masked}")
    assert excluded_visible == 0 and kept_masked == 0


@pytest.mark.parametrize("dtype,bits", CASES)
@pytest.mark.parametrize("L", (1, 7, 64, 512))
def test_mf1_quant_square_byte_identical_to_before(L, dtype, bits):
    """Fix A must not move the square (S == L) case by a single byte."""
    if RULE != "A":
        pytest.skip("fix B changes square selection by design; identity is receipted in the A-only set")
    before = _load_before()
    out_b, _, _ = run(before.apa_quant_attention, L, L, dtype, bits)
    out_t, _, _ = run(Q.apa_quant_attention, L, L, dtype, bits)
    assert out_b.tobytes() == out_t.tobytes(), f"L={L} {dtype} bits={bits}: square output moved"


def test_mf1_quant_s_less_than_l_raises():
    """S < L is out of contract. APA-MF2.1 item 1 (review F3): bottom-right causal rows
    0..L-S-1 see no key, so the call is REJECTED (ValueError naming S and L) before any
    compute -- was 'finite' under MF1's -1e4 bias, NaN under MF2's -inf mask."""
    ts, _ = operands(8, 5, "float32", 7)
    with pytest.raises(ValueError, match=r"S >= L"):
        Q.apa_quant_attention(*ts, bulk_bits=8, refine_percentile=P, is_causal=True)
