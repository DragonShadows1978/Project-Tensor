"""APA-MF1: tensor_gpu_v2 causal attention paths on rectangular shapes.

Covers every causal branch in tensor_gpu_v2/_core.py: standard SDPA, the
FlashAttention-style tiled path, APA refine >= 1.0 (full-precision tiled loop),
APA refine < 1 non-tiled and use_tiled, forward AND backward. The fp64 reference
(scripts/apa_mf1_ref.py) is written from the definition (row i sees cols
<= S-L+i); the old tensor_gpu_v2 is never the reference. APA selection follows
tensor_gpu_v2's own documented rule (|ranking|, z-score for S > 256, exact top-k
otherwise, masked entries zeroed); the dequantized keys are an operand read from
the function's closure. Ties within 1e-5 relative of the fp64 threshold take the
code's choice (counted, printed); all other visible entries must agree.

Prior art: FlashAttention-2 bottom-right causal (Dao, 2023); Project-Tensor
63960a4 (2026-06-09). Test harness: no prior art known to me.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import apa_mf1_ref as R  # noqa: E402

cp = pytest.importorskip("cupy")
import tensor_gpu_v2 as tg  # noqa: E402
from tensor_gpu_v2 import _core as C  # noqa: E402
from scipy.stats import norm  # noqa: E402

BEFORE = ROOT / "artifacts/apa_mf1/before/tensor_gpu_v2___core.py"
SHAPES = [(L, S) for L in (1, 7, 64, 512) for S in (L, L + 1, L + 100, 4 * L + 3)]
PATHS = ["sdpa", "flash", "apa_full", "apa"]
TIE_REL = 1e-5
TOL = 1e-5
P = 0.15
_BEFORE = None


def _before():
    global _BEFORE
    if _BEFORE is None:
        spec = importlib.util.spec_from_file_location("tensor_gpu_v2._mf1_core_before", BEFORE)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _BEFORE = mod
    return _BEFORE


def _closure(fn):
    return {n: c.cell_contents for n, c in zip(fn.__code__.co_freevars, fn.__closure__ or ())}


def _np(a):
    return cp.asnumpy(a) if isinstance(a, cp.ndarray) else np.asarray(a)


def operands(L, S, H, D, seed):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((1, H, L, D)).astype(np.float32)
    k = rng.standard_normal((1, H, S, D)).astype(np.float32)
    v = rng.standard_normal((1, H, S, D)).astype(np.float32)
    dO = rng.standard_normal((1, H, L, D)).astype(np.float32)
    return q, k, v, dO


def run_path(M, path, q, k, v, dO):
    Qt, Kt, Vt = (M.Tensor(a, device="cuda", requires_grad=True) for a in (q, k, v))
    if path == "sdpa":
        out = M.scaled_dot_product_attention(Qt, Kt, Vt, is_causal=True, use_flash=False)
    elif path == "flash":
        out = M.flash_attention(Qt, Kt, Vt, block_size=32, is_causal=True)
    elif path == "apa_full":
        out = M.apa_quant_attention(Qt, Kt, Vt, refine_percentile=1.0, is_causal=True, block_size=32)
    else:
        out = M.apa_quant_attention(Qt, Kt, Vt, bulk_bits=2, refine_percentile=P, is_causal=True)
    info = _closure(out._backward) if path == "apa" else {}
    out.backward(cp.asarray(dO))
    res = {"out": _np(out.data), "dq": _np(Qt.grad), "dk": _np(Kt.grad), "dv": _np(Vt.grad)}
    if path == "apa":
        masks = info["stored_refine_masks"]
        res["sel"] = np.concatenate([_np(masks[i]) for i in sorted(masks)], axis=2)
        res["kq"] = _np(info["key_quant_data"]).astype(np.float64)
        res["tiled"] = bool(info["use_tiled"])
    return res


def check(path, res, q, k, v, dO, L, S):
    D = q.shape[-1]
    scale = D ** -0.5
    vis = R.visible_mask(L, S)
    extra = ""
    if path == "apa":
        z = float(norm.ppf(1.0 - P))
        abs_rank = np.abs(np.einsum("bhld,bhsd->bhls", q.astype(np.float64), k.astype(np.float64)) * scale)
        sel_ref, thr = R.tgv2_apa_select(abs_rank, vis, S, P, z)
        amb = R.ambiguous(abs_rank, thr, TIE_REL) & vis
        dis = vis & ~amb & (res["sel"] != sel_ref)
        sel_use = np.where(amb, res["sel"], sel_ref)
        ref_out = R.apa_tc_reference(q, k, res["kq"], v, scale, sel_use)
        ref_g = R.tgv2_apa_backward(q, k, res["kq"], v, dO, scale, sel_use)
        extra = f" tiled={res['tiled']} nontie_disagree={int(dis.sum())} ties={int(amb.sum())}"
        n_dis = int(dis.sum())
    else:
        ref_out, _ = R.exact_attention(q, k, v, scale)
        ref_g = R.exact_attention_grads(q, k, v, dO, scale)
        n_dis = 0
    errs = {"out": R.rel_err(res["out"], ref_out)}
    for name, g in zip(("dq", "dk", "dv"), ref_g):
        errs[name] = R.rel_err(res[name], g)
    msg = f"{path} L={L} S={S}{extra} " + " ".join(f"{n}:rel={e[0]:.3e},maxabs={e[1]:.3e}" for n, e in errs.items())
    print(msg)
    assert n_dis == 0 and all(e[0] <= TOL for e in errs.values()), msg


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("L,S", SHAPES)
def test_mf1_tgv2_causal_reference(L, S, path):
    q, k, v, dO = operands(L, S, 2, 16, 1000 * L + S)
    res = run_path(C, path, q, k, v, dO)
    if path == "apa":
        assert res["tiled"] is False
    check(path, res, q, k, v, dO, L, S)


def _tiled_heads(L, S):
    return (256 * 1024 * 1024) // (L * S * 4) + 1


@pytest.mark.parametrize("S", (512, 513, 612, 2051))
def test_mf1_tgv2_apa_use_tiled_reference(S):
    L = 512
    q, k, v, dO = operands(L, S, _tiled_heads(L, S), 8, 7000 + S)
    res = run_path(C, "apa", q, k, v, dO)
    assert res["tiled"] is True
    check("apa", res, q, k, v, dO, L, S)


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("L", (1, 7, 64, 512))
def test_mf1_tgv2_square_byte_identical_to_before(L, path):
    q, k, v, dO = operands(L, L, 2, 16, 1000 * L + L)
    a = run_path(_before(), path, q, k, v, dO)
    b = run_path(C, path, q, k, v, dO)
    for key in ("out", "dq", "dk", "dv"):
        assert a[key].tobytes() == b[key].tobytes(), f"{path} L={L}: square {key} moved"


def test_mf1_tgv2_square_byte_identical_tiled():
    L = 512
    q, k, v, dO = operands(L, L, _tiled_heads(L, L), 8, 7000 + L)
    a = run_path(_before(), "apa", q, k, v, dO)
    b = run_path(C, "apa", q, k, v, dO)
    assert a["tiled"] and b["tiled"]
    for key in ("out", "dq", "dk", "dv"):
        assert a[key].tobytes() == b[key].tobytes(), f"tiled L={L}: square {key} moved"
