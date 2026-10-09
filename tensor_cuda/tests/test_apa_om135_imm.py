"""APA-OM135: the IMM derivative self-reduction check of backward orientation (scripts/apa_om135_core.py).
CPU tests check the identity and the exactness machinery; GPU tests (main engine) check that the three methods
(IMM, integer textbook baseline, fast-mode gradcheck) pass on correct autograd and that a synthetic wrong-orientation
backward (Python straight-through wrapper: correct forward, transposed-operand gradient) is flagged.
Author-run tests: a baseline, not blind verification.

Prior art: openai/math #135 (2026) via TX2 Opus M4c (derivatives of IMM = shorter IMM); Nisan (1991), Nisan &
Wigderson (1997) partial-derivative measures; Baur & Strassen (1983); Griewank & Walther (2008) basis-seed reverse
mode; PyTorch gradcheck fast_mode (2021); DeMillo, Lipton & Sayward (1978). Unverified -- lead to check.
Mine: the harness under test.

APA-REGT1 production port (from branch apa-om135 @ 2bbee82): the hardcoded fork build path (MAIN_BUILD) and the
env-var engine pinning (C.load_engine) are gone -- the GPU tests use the importable tensor_cuda like every test in
this directory and skip, never error, without a usable CUDA device; prepare() (scripts/apa_om135_prep.py) now lives
in scripts/apa_om135_core.py. Methods, thresholds and cells unchanged.
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import apa_om135_core as C  # noqa: E402


# ------------------------------------------------------------------ CPU
@pytest.mark.parametrize("k,trans,alphas,shapes", [
    (2, [False, False], [1.0, 1.0], [(5, 4), (4, 6)]),
    (3, [False, True, False], [1.0, 2.0, 1.0], [(5, 4), (6, 4), (6, 3)]),
    (4, [False] * 4, [1.0] * 4, [(5, 4), (4, 3), (3, 6), (6, 2)]),
])
def test_imm_reference_equals_textbook_gradient_cpu(k, trans, alphas, shapes):
    """The derivative self-reduction: with the basis substitution and a one-hot seed, the slices of the middle chain
    equal the textbook gradient (fp64)."""
    cell = C.ChainCell("t", "chain", shapes, trans, alphas, 1, 1, 7)
    rng = np.random.default_rng(3)
    for _ in range(5):
        Xs, b, dY, seeds = cell.imm_operands(rng)
        want = cell.textbook(Xs, dY)
        got = _imm_reference_host(cell, Xs, seeds, C.chain_eff(Xs, trans))   # host stand-in for M_2 M_3
        for g, w in zip(got, want):
            assert np.array_equal(g, w)


def _imm_reference_host(cell, Xs, seeds, Ms):
    saved_up, saved_down, saved_tc = C.up, C.down, C.tc

    class _T:
        @staticmethod
        def matmul(a, b_):
            return np.matmul(a, b_)
    try:
        C.up, C.down, C.tc = (lambda x, dt, grad=False: x), (lambda t: np.asarray(t)), _T
        ref, _ = cell.imm_reference(Xs, seeds, "float32")
    finally:
        C.up, C.down, C.tc = saved_up, saved_down, saved_tc
    return ref


def test_imm_batched_and_broadcast_cpu():
    for kind, shapes, trans, shared in (("bbm", [(3, 5, 4), (4, 6)], [False, False], [False, True]),
                                        ("bmm", [(2, 3, 5, 4), (2, 3, 4, 6)], [False, False], None),
                                        ("linear", [(2, 5, 4), (6, 4)], [False, True], [False, True])):
        cell = C.ChainCell("t", kind, shapes, trans, [1.0, 1.0], 1, 1, 9, bias=(kind == "linear"), shared=shared)
        Xs, b, dY, seeds = cell.imm_operands(np.random.default_rng(5))
        want = cell.textbook(Xs, dY)
        got = _imm_reference_host(cell, Xs, seeds, C.chain_eff(Xs, trans))
        for g, w in zip(got, want):
            assert np.array_equal(g, w), kind


def test_round_to_bf16_and_certificate_cpu():
    assert C.round_to(np.array([257.0]), "bfloat16")[0] == 256.0      # 9 significant bits -> rounds
    assert C.round_to(np.array([256.0, 258.0, 0.5]), "bfloat16").tolist() == [256.0, 258.0, 0.5]
    Xs = [np.full((4, 4), 1.0), np.full((4, 4), 1.0)]
    ok, cert = C.chain_cert(Xs, [False, False], 1.0, np.full((4, 4), 1.0), "bfloat16")
    assert ok and cert["max_abs_bound"] == 4.0                         # every entry of ones(4,4) @ ones(4,4)
    Xs = [np.full((4, 300), 1.0), np.full((300, 4), 1.0)]
    ok, _ = C.chain_cert(Xs, [False, False], 1.0, np.ones((4, 4)), "bfloat16")
    assert not ok                                                       # 300 > 2^8: refused


def test_bf16_ladder_exact_and_distinct_cpu():
    lad = C.bf16_ladder(1025)
    assert len(np.unique(lad)) == 1025 and np.all(np.diff(lad) >= 1)
    for dt in C.DTYPES:
        assert np.array_equal(C.round_to(lad, dt), lad)


def test_sdpa_operands_two_hot_cpu():
    cell = C.SdpaCell("t", 2, 2, 9, 16, 4, 0.5, 0.5, 0.25, 4)
    ops = cell.operands(np.random.default_rng(1))
    P, _ = cell._np_fwd(ops["q"], ops["k"], ops["v"], ops["mask"])
    assert set(np.unique(P).tolist()) <= {0.0, 0.5, 1.0}
    assert np.all(P[:, :, 1:].max(-1) == 0.5) and np.all(P[:, :, 0, 0] == 1.0)


# ------------------------------------------------------------------ GPU (main engine)
def _engine():
    """APA-REGT1: the importable tensor_cuda (it imports without a CUDA device and fails at the first allocation:
    skip, never error -- the probe of test_terrain_render.py's _require_cuda)."""
    tc = pytest.importorskip("tensor_cuda")
    try:
        tc.zeros((1,), dtype="float32").numpy()
    except Exception as exc:  # pragma: no cover - host dependent
        pytest.skip(f"CUDA unavailable: {exc}")
    from tensor_cuda import functional as F
    from tensor_cuda import nn
    C.tc, C.F, C.nn = tc, F, nn


def _small_cells():
    return [
        C.ChainCell("CH3", "chain", [(17, 16), (16, 12), (12, 20)], [False] * 3, [1.0] * 3, 2, 2, 21),
        C.ChainCell("CH4", "chain", [(17, 16), (16, 8), (8, 24), (24, 10)], [False] * 4, [1.0] * 4, 2, 2, 22),
        C.ChainCell("SQ3T", "chain", [(16, 16)] * 3, [False, True, False], [1.0, 2.0, 1.0], 2, 2, 23),
        C.ChainCell("LIN", "linear", [(2, 17, 16), (12, 16)], [False, True], [1.0, 1.0], 2, 2, 24, bias=True,
                    shared=[False, True]),
        C.ChainCell("BBM", "bbm", [(2, 17, 16), (16, 12)], [False, False], [1.0, 1.0], 2, 2, 25, shared=[False, True]),
        C.ChainCell("BMM", "bmm", [(2, 4, 17, 16), (2, 4, 16, 12)], [False, False], [1.0, 1.0], 2, 2, 26),
        C.SdpaCell("SDPA", 2, 4, 33, 64, 27, 0.5, 0.5, 1 / 16, 4),
        C.ApaCell("APA", 2, 4, 2, 33, 64, 28, 0.5, 0.25),
    ]


@pytest.mark.parametrize("dtype", list(C.DTYPES))
def test_three_methods_pass_on_main_gpu(dtype):
    _engine()
    prepare = C.prepare  # APA-REGT1: apa_om135_prep.prepare @ 2bbee82 now lives in scripts/apa_om135_core.py
    for cell in _small_cells():
        p = prepare(cell, "bfloat16")
        assert p["ok"], (cell.cid, p["cert"])
        lins = {}
        if isinstance(cell, C.ChainCell) and cell.kind == "linear":
            lins = {d: cell.make_linear(d) for d in {dtype, "float32"}}
        prep = p["imm"]
        if not isinstance(cell, C.ChainCell):
            prep = (prep[0], prep[1], prep[2], cell.eye_host(dtype) if cell.kind == "sdpa" else None)
        a, w, _ = C.method_imm(cell, None, dtype, lin=lins.get(dtype), prepared=prep)
        assert not a, (cell.cid, w)
        a, w, _ = C.method_textbook(cell, dtype, lin=lins.get(dtype), prepared=p["textbook"])
        assert not a, (cell.cid, w)
        a, w, _ = C.method_gradcheck(cell, dtype, lin32=lins.get("float32"), lin=lins.get(dtype),
                                     prepared=p["textbook"], probe_seed=cell.seed + 2)
        assert not a, (cell.cid, w)


class _WrongOrientationChain(C.ChainCell):
    """Straight-through wrapper (Python-level fixture, not a registered mutant): forward value of X1 @ X2, gradient of
    X1 @ X2^T (the dropped-! class; square factors)."""

    def forward(self, ts, bt, lin=None):
        tc = C.tc
        good = tc.matmul(ts[0], ts[1])
        bad = tc.matmul(ts[0], ts[1], trans_b=True)
        return good.detach() + (bad - bad.detach())


@pytest.mark.parametrize("dtype", list(C.DTYPES))
def test_wrong_orientation_fixture_is_flagged_gpu(dtype):
    _engine()
    prepare = C.prepare  # APA-REGT1: apa_om135_prep.prepare @ 2bbee82 now lives in scripts/apa_om135_core.py
    cell = _WrongOrientationChain("WRONG", "chain", [(16, 16), (16, 16)], [False, False], [1.0, 1.0], 2, 2, 31)
    p = prepare(cell, "bfloat16")
    assert p["ok"]
    assert C.method_imm(cell, None, dtype, prepared=p["imm"])[0]
    assert C.method_textbook(cell, dtype, prepared=p["textbook"])[0]
    assert C.method_gradcheck(cell, dtype, prepared=p["textbook"], probe_seed=3)[0]
