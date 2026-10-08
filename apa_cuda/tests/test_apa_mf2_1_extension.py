"""APA-MF2.1: apa.cpp causal S < L rejection (item 1) and the kNegInf-only reversion
discriminator (item 4, review F4b), on the extension module.

Module under test: the in-place build of apa_cuda if importable; otherwise, with
APA_MF2_EXT_DIR set, the MF2 stand-in harness (the shipped csrc/apa.cpp compiled as a
C++ torch extension with apa_kernels.cu replaced by an ATen transcription,
scripts/apa_mf2_ext_stub_build.py). The harness runs apa.cpp's OWN mask / check code;
it is NOT the shipped extension (the official build needs a CUDA 13 toolkit).

F4b: apa.cpp REPLACES masked scores (masked_fill(cmask, kNegInf)); reverting kNegInf
alone to the old finite -1e9 is discriminated only by a VISIBLE score below the fill:
L=S=3, D64, Q=e0, K0=-2e9*e0, V0=1, other K/V 0, scale 1. Row 0 sees key 0 only ->
exactly 1. With a -1e9 fill, row 0's filled keys (-1e9) dominate -2e9 -> row 0 = 0.

Prior art: the review's operands (apa-mf2rev F4); FlashAttention-2 (Dao 2023) -inf
predicate masking. Harness reuse: MF2's. Tests: mine.
"""
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
if os.environ.get("APA_MF2_EXT_DIR"):
    sys.path.insert(0, os.environ["APA_MF2_EXT_DIR"])
sys.path.insert(0, PKG)
try:
    import apa_attention_cuda as ext  # noqa: F401
except ImportError as exc:  # environment, not a pass: the gate reports it as not-run
    pytest.skip(f"apa_attention_cuda not importable ({exc}); build it or set APA_MF2_EXT_DIR",
                allow_module_level=True)
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)
from apa_attention import apa_quant_attention  # noqa: E402


def _fwd(q, k, v, causal=True, p=1.0, scale=1.0):
    H, D = q.shape[1], q.shape[-1]
    rot = torch.eye(D, device="cuda").expand(H, D, D).contiguous()
    return ext.forward(q, k, v, rot, torch.zeros(1, device="cuda"), torch.empty(0, device="cuda"),
                       p, causal, None, scale, 0.0, 64, False, False)


@pytest.mark.parametrize("L,S", [(8, 5), (3, 2), (2, 1)])
def test_forward_causal_s_less_than_l_raises(L, S):
    q = torch.zeros((1, 1, L, 64), device="cuda"); k = torch.zeros((1, 1, S, 64), device="cuda")
    with pytest.raises(RuntimeError, match=r"S >= L"):
        _fwd(q, k, k)
    with pytest.raises(RuntimeError, match=r"S >= L"):
        apa_quant_attention(q, k, k, refine_percentile=1.0, is_causal=True)


def test_backward_causal_s_less_than_l_raises():
    L, S = 8, 5
    q = torch.zeros((1, 1, L, 64), device="cuda"); k = torch.zeros((1, 1, S, 64), device="cuda")
    mask = torch.zeros((1, 1, L, S), dtype=torch.uint8, device="cuda")
    with pytest.raises(RuntimeError, match=r"S >= L"):
        ext.backward(torch.ones_like(q), q, k, k, k, mask, 1.0, True, None, False, 64)


def test_noncausal_s_less_than_l_still_computes():
    q = torch.randn((1, 1, 8, 64), device="cuda"); k = torch.randn((1, 1, 5, 64), device="cuda")
    out = _fwd(q, k, k, causal=False)[0]
    assert out.shape == (1, 1, 8, 64) and torch.isfinite(out).all()


@pytest.mark.parametrize("p", [1.0, 0.15])
def test_f4b_negative_visible_score_below_finite_fill(p):
    L = S = 3; D = 64
    q = torch.zeros((1, 1, L, D), device="cuda"); q[..., 0] = 1
    k = torch.zeros((1, 1, S, D), device="cuda"); k[:, :, 0, 0] = -2e9
    v = torch.zeros_like(k); v[:, :, 0, :] = 1
    out = _fwd(q, k, v, p=p)[0]
    assert torch.isfinite(out).all()
    assert (out[0, 0, 0] == 1).all(), f"row 0 {out[0, 0, 0, :4].tolist()} (finite-fill reversion)"
