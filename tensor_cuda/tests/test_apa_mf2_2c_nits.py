"""APA-MF2.2 Amendment 3 nits.

V5: test_apa_phase6._ref used a finite -1e9 causal fill and one-pass statistics. Review witness
(apa-mf22rev finding_witnesses.json V5): L=S=3, D64, fp32 Q=8e0, K0=-2e9e0, K1=K2=0, V0=1, others 0,
p=1, causal -> expected [1,0,0]; the old _ref gave [0,0,0] (row 0's only visible score -2e9 lost to
the masked keys' -1e9 fill). The reference now uses a -inf predicate, centered fp64 statistics and
the 0-weight policy for all -inf rows. V3: the guard docstring no longer claims PyTorch SDPA
returns NaN for a fully masked row (torch 2.11 SDPA returns 0; logs/apa_mf2_2c_v3_torch_check.log).
Witness: the review's; tests: mine.
"""
import importlib.util
import os

import numpy as np

import tensor_cuda as tc
from tensor_cuda import functional as F
from tensor_cuda import quant as Q

HERE = os.path.dirname(os.path.abspath(__file__))


def _phase6():
    spec = importlib.util.spec_from_file_location("apa_phase6_ref_under_test", os.path.join(HERE, "test_apa_phase6.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def _witness():
    q = np.zeros((1, 1, 3, 64), np.float32); q[..., 0] = 8.0
    k = np.zeros((1, 1, 3, 64), np.float32); k[0, 0, 0, 0] = -2e9
    v = np.zeros((1, 1, 3, 64), np.float32); v[0, 0, 0] = 1.0
    return q, k, v


def test_phase6_ref_v5_witness_matches_definition_and_engine():
    q, k, v = _witness()
    ref = _phase6()._ref(q, k, v, 4, 1.0, True)
    assert np.array_equal(ref[0, 0, :, 0], [1.0, 0.0, 0.0]), ref[0, 0, :, 0]
    with tc.no_grad():
        o = Q.apa_quant_attention(*[tc.tensor(x) for x in (q, k, v)], bulk_bits=4, refine_percentile=1.0,
                                  is_causal=True).numpy()
    assert np.array_equal(o[0, 0, :, 0], [1.0, 0.0, 0.0]), o[0, 0, :, 0]


def test_phase6_ref_partial_refine_on_witness_is_finite():
    q, k, v = _witness()
    ref = _phase6()._ref(q, k, v, 4, 0.15, True)
    assert np.isfinite(ref).all() and ref[0, 0, 0, 0] == 1.0


def test_v3_guard_docstring_states_measured_pytorch_behaviour():
    doc = F._guarded_softmax.__doc__
    assert "PyTorch SDPA returns NaN there" not in doc and "scaled_dot_product_attention" in doc
