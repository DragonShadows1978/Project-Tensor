"""The unchanged EB4 lead fixture is now a production suite test.

Prior art: EB4 lead_check.py (Project-Tensor, 2026), Batcher (1968) sorting
network/stable score-index total order inherited via EB2b. Taken: existing
blocks-5/20 fixture and lower-index tie order; no new selection rule.
Unverified literature -- lead to check: Batcher sorting networks 1968.
"""
import ast
import os
from pathlib import Path

import numpy as np
import pytest

from test_apa_p1_tickets import ROOT, ART, GPU, gpu_runtime, float_bits, assert_informative

LEAD = Path('/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eb4/artifacts/apa_eb4/lead_gate/lead_check.py')


def rnd(x):
    u = np.asarray(x, np.float32).view(np.uint32); u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.view(np.float32)


def tie_fixture():
    # Unchanged lead_check.py:60-64, including seed, rounding and draw order.
    rng = np.random.default_rng(20261019); H, KVH, S, D = 16, 8, 2048, 128
    q = rnd(rng.standard_normal((1, H, 1, D))); k = rnd(0.5 * rng.standard_normal((1, KVH, S, D))); v = rnd(rng.standard_normal((1, KVH, S, D)))
    for kh in range(KVH):
        blk = rnd(0.3 * q[0, 2 * kh, 0][None, :] + 0.05 * rng.standard_normal((64, D)))
        k[0, kh, 5 * 64:6 * 64] = blk; k[0, kh, 20 * 64:21 * 64] = blk   # bit-identical blocks 5 and 20
    return q, k, v


def test_fixture_matches_lead_ast():
    # The pinned faithful excerpt is inside this round's authorized artifacts;
    # its SHA and origin are recorded in FIXTURE_PROVENANCE.json.
    reference = ast.parse((ART / 'lead_tie_fixture.py.txt').read_text())
    own = ast.parse(Path(__file__).read_text())
    fn = next(x for x in own.body if isinstance(x, ast.FunctionDef) and x.name == 'tie_fixture')
    assert ast.dump(ast.Module(body=fn.body[:-1], type_ignores=[]), include_attributes=False) == ast.dump(reference, include_attributes=False)


def test_exact_tie_fixture_cpu():
    q, k, v = tie_fixture()
    assert q.shape == (1, 16, 1, 128) and k.shape == v.shape == (1, 8, 2048, 128)
    for a in (q, k, v):
        assert a.dtype == np.float32 and np.isfinite(a).all()
        assert np.all((a.view(np.uint32) & 0xFFFF) == 0), 'fixture must be bf16-exact'
        assert_informative(a.view(np.uint32))
    assert np.array_equal(k[:, :, 320:384].view(np.uint32), k[:, :, 1280:1344].view(np.uint32))
    assert int(np.ceil((1 / 32) * (2048 // 64))) == 1
    # CPU fixture sanity only. Actual fp32 packed scores are asserted on GPU.
    scores = np.einsum('bhd,bhsd->bhs', q[:, ::2, 0], k, dtype=np.float32)
    block_mass = np.exp(scores - scores.max(axis=-1, keepdims=True)).reshape(1, 8, 32, 64).sum(axis=-1)
    assert np.array_equal(block_mass[:, :, 5].view(np.uint32), block_mass[:, :, 20].view(np.uint32))
    assert np.all(block_mass[:, :, 5] == block_mass.max(axis=-1))


@pytest.mark.skipif(not GPU, reason='staged lead GPU test; CPU seat has no GPU')
def test_exact_tie_gpu(tmp_path):
    tc, _ = gpu_runtime()
    os.environ.pop('APA_P1_EARLY_COUNTER', None)
    q, k, v = tie_fixture()
    T = lambda x: tc.tensor(np.ascontiguousarray(x), dtype='bfloat16')
    qt, kt, vt = T(q), T(k), T(v)
    w = tc.ApaSpq1Workspace(kt, 2049, fmt='tq4')
    vi, vs = tc.apa_eb2b_quantize(vt)
    scale = float(np.float32(1 / np.sqrt(128)))
    ref = tc.apa_eb2b(qt, kt, vt, vi, vs, w.packed, scale, 1 / 32, delta=8.1, carry="row", diagnostics=True)
    got = tc.apa_eb4(qt, kt, vt, vi, vs, w.packed, scale, 8.1, 1 / 32, carry='row', diagnostics=True)
    flags_ref = ref[4].numpy().astype(np.uint8).reshape(1, 8, 32)
    flags_got = got[1].numpy().astype(np.uint8).reshape(1, 8, 32)
    out_ref, out_got = float_bits(ref[0]), float_bits(got[0])
    scores = got[3].numpy().astype(np.float32).reshape(1, 8, 32)
    for a in (flags_ref, out_ref): assert_informative(a)
    assert np.array_equal(scores[:, :, 5].view(np.uint32), scores[:, :, 20].view(np.uint32)), 'cutoff scores must tie in fp32 bits'
    assert np.all(scores[:, :, 5] == scores.max(axis=-1)), 'tie must straddle top-1 cutoff'
    assert np.all(flags_ref.sum(axis=-1) == 1) and np.all(flags_got.sum(axis=-1) == 1), 'one high block per KV head'
    # Save references at full bit width before the registered mutant fails.
    np.savez(tmp_path / 'tie_readback.npz', ref_out=out_ref, out=out_got, ref_flags=flags_ref, flags=flags_got, scores=scores.view(np.uint32))
    if os.environ.get('APA_P1_MUTANT') == 'eb4_reverse_tie':
        assert np.all(flags_got[:, :, 20] == 1) and np.all(flags_got[:, :, 5] == 0), 'reverse-tie mutant must select block 20'
    # Named negative control: APA_EB4_MUTANT=7 must fail selecting block 20.
    assert np.all(flags_got[:, :, 5] == 1) and np.all(flags_got[:, :, 20] == 0), 'exact-tie lower-index winner mismatch'
    assert np.array_equal(flags_ref, flags_got), 'EB2b/EB4 flags differ bitwise'
    assert np.array_equal(out_ref, out_got), 'EB2b/EB4 output differs bitwise'
