"""APA-MF2.1: source checks on apa_cuda/csrc/apa.cpp (no extension needed).

The official extension cannot be built here (torch cu130 vs nvcc 12.x, MF2
logs/apa_mf2_ext_build_*.log), so two contracts are also pinned at the source level:
  * the causal S < L TORCH_CHECK is the first statement of apa_forward / apa_backward
    that can fail, i.e. it precedes every at::matmul and quantize_keys call (item 1);
  * the causal fill is -INFINITY, applied with masked_fill (item 4, review F4b; the
    behavioural discriminator is test_apa_mf2_1_extension.py, run via the stand-in).
Prior art: none known to me for source-order assertions. Tests: mine.
"""
import os
import re

import pytest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "csrc", "apa.cpp")


def _body(src, name):
    i = src.index(f"std::vector<at::Tensor> {name}(")
    o = src.index("{", src.index(")", i))
    depth, j = 1, o + 1
    while depth:
        depth += (src[j] == "{") - (src[j] == "}")
        j += 1
    return src[o:j]


@pytest.mark.parametrize("fn", ["apa_forward", "apa_backward"])
def test_s_less_than_l_check_precedes_any_matmul(fn):
    body = _body(open(SRC).read(), fn)
    m = re.search(r"TORCH_CHECK\(\s*!is_causal\s*\|\|\s*S\s*>=\s*L", body)
    assert m, f"{fn}: no causal S >= L TORCH_CHECK"
    first = min(body.index(t) for t in ("at::matmul", "quantize_keys(", "block_grad(", "apply_masks(") if t in body)
    assert m.start() < first, f"{fn}: the S >= L check comes after compute"


def test_causal_fill_is_minus_inf_predicate():
    src = open(SRC).read()
    assert re.search(r"constexpr double kNegInf = -INFINITY;", src)
    assert "scores = scores.masked_fill(cmask, kNegInf);" in src
