"""APA-MF1: the apa_cuda parity test's NumPy reference must be bottom-right causal.

apa_cuda/tests/test_parity.py::reference_forward / reference_backward are the
oracle the CUDA extension is checked against. If they are top-left on a
rectangle, parity passes while both sides are wrong. This checks them against
the fp64 definition (scripts/apa_mf1_ref.py; row i sees cols <= S-L+i) on the
MF1 rectangular grid, with tensor_gpu_v2's documented selection rule.
Pure NumPy: no extension, no GPU.

Prior art: FlashAttention-2 bottom-right causal (Dao, 2023); Project-Tensor
63960a4 (2026-06-09). Harness: no prior art known to me.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import apa_mf1_ref as R  # noqa: E402
import test_parity as TP  # noqa: E402
from scipy.stats import norm  # noqa: E402

SHAPES = [(L, S) for L in (1, 7, 64, 512) for S in (L, L + 1, L + 100, 4 * L + 3)]
P = 0.15


@pytest.mark.parametrize("L,S", SHAPES)
def test_mf1_parity_reference_is_bottom_right(L, S):
    rng = np.random.default_rng(1000 * L + S)
    H, D = 2, 16
    q = rng.standard_normal((1, H, L, D)).astype(np.float32)
    k = rng.standard_normal((1, H, S, D)).astype(np.float32)
    v = rng.standard_normal((1, H, S, D)).astype(np.float32)
    g = rng.standard_normal((1, H, L, D)).astype(np.float32)
    out, kq, mask = TP.reference_forward(q, k, v, bulk_bits=2, refine_percentile=P, is_causal=True)
    gq, gk, gv = TP.reference_backward(g, q, k, v, kq, mask, is_causal=True)
    scale = D ** -0.5
    vis = R.visible_mask(L, S)
    abs_rank = np.abs(np.einsum("bhld,bhsd->bhls", q.astype(np.float64), k.astype(np.float64)) * scale)
    sel_ref, thr = R.tgv2_apa_select(abs_rank, vis, S, P, float(norm.ppf(1.0 - P)))
    amb = R.ambiguous(abs_rank, thr, 1e-5) & vis
    dis = int((vis & ~amb & (mask != sel_ref)).sum())
    sel_use = np.where(amb, mask, sel_ref)
    ref_out = R.apa_tc_reference(q, k, kq, v, scale, sel_use)
    ref_g = R.tgv2_apa_backward(q, k, kq, v, g, scale, sel_use)
    errs = [R.rel_err(out, ref_out)] + [R.rel_err(a, b) for a, b in zip((gq, gk, gv), ref_g)]
    msg = (f"L={L} S={S} nontie_disagree={dis} ties={int(amb.sum())} "
           + " ".join(f"{n}:rel={e[0]:.3e},maxabs={e[1]:.3e}" for n, e in zip(("out", "dq", "dk", "dv"), errs)))
    print(msg)
    assert dis == 0 and all(e[0] <= 1e-5 for e in errs), msg
