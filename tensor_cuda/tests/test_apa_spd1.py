"""G1 author-run CPU baseline, not blind verification.

Prior art: independent materialized SDPA oracle (Vaswani et al., 2017),
PyTorch math SDPA (2023-2026), SP1 bottom-right and GQA boundary testing (2026).
New work: adversarial plumbing inputs and receipt-integrity checks.
"""
from pathlib import Path
import copy
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
override = os.environ.get('APA_SPD1_COMMON_OVERRIDE')
if override:
    spec = importlib.util.spec_from_file_location('apa_spd1_common', override)
    common = importlib.util.module_from_spec(spec)
    sys.modules['apa_spd1_common'] = common
    spec.loader.exec_module(common)
    common.ROOT, common.ART = ROOT, ROOT / 'artifacts/apa_spd1'
else:
    import apa_spd1_common as common
from apa_spd1_report import assemble, render, score_predictions, gate_verdict


class NumpyTensor:
    """CPU test double for grouped engine composition, not a GPU emulation."""
    def __init__(self, a): self.a = np.asarray(a)
    def reshape(self, shape): return NumpyTensor(self.a.reshape(shape))
    def flip(self, axis): return NumpyTensor(np.flip(self.a, axis))
    def numpy(self): return self.a
    def softmax(self, axis):
        ex = np.exp(self.a - self.a.max(axis=axis, keepdims=True))
        return NumpyTensor(ex / ex.sum(axis=axis, keepdims=True))


class NumpyEngine:
    @staticmethod
    def matmul(a, b, alpha=1., trans_b=False):
        return NumpyTensor(np.matmul(a.a, b.a.swapaxes(-1, -2) if trans_b else b.a) * alpha)
    @staticmethod
    def causal_softmax(a):
        L, S = a.a.shape[-2:]
        mask = np.arange(S)[None, :] <= np.arange(L)[:, None] + S - L
        return NumpyTensor(np.where(mask, a.a, -np.inf)).softmax(-1)


def tiny(L=3, S=7, H=8, KVH=2, causal=True):
    return dict(id='tiny', B=1, H=H, KVH=KVH, L=L, S=S, D=4, VD=4,
                causal=causal, family='sp1', delta_status='test fixture')


def independent_dense(q, k, v, shape):
    out = np.zeros_like(q)
    # Explicit head/query loops independently check grouped reshape and mask.
    for b in range(shape['B']):
        for h in range(shape['H']):
            kh = h // (shape['H'] // shape['KVH'])
            for i in range(shape['L']):
                end = shape['S'] - shape['L'] + i + 1 if shape['causal'] else shape['S']
                score = k[b, kh, :end] @ q[b, h, i] / np.sqrt(shape['D'])
                prob = np.exp(score - max(score)); prob /= prob.sum()
                out[b, h, i] = prob @ v[b, kh, :end]
    return out


def test_grid_and_frozen_registration():
    reg = common.registration()
    previous = json.loads((ROOT / 'artifacts/apa_sp1/registration.json').read_text())
    assert len(reg['shapes']) == 50
    assert len({s['id'] for s in reg['shapes']}) == 50
    for now, old in zip(reg['shapes'], previous['shapes']):
        assert {k: now[k] for k in old} == old
    extras = reg['shapes'][48:]
    assert {s['S'] for s in extras} == {8192, 32768}
    assert all((s['B'], s['H'], s['KVH'], s['L'], s['D'], s['causal']) == (1,16,4,512,128,True) for s in extras)
    cal = json.loads((ROOT / 'artifacts/apa_sp1/calibration.json').read_text())['geometries']
    assert all(s['delta'] == cal[s['delta_source']]['delta'] for s in reg['shapes'])


def test_registry_is_explicit_and_unique():
    specs = common.registry()
    assert len(specs) == len({s['id'] for s in specs}) == 12
    assert {s.get('backend') for s in specs if 'backend' in s} == {'MATH', 'EFFICIENT_ATTENTION', 'FLASH_ATTENTION'}
    assert all(s['entry'] and s['dtype'] in ('float32', 'bfloat16') for s in specs)
    assert next(s for s in specs if s['id'] == 'engine_dense_fp32')['required']
    assert not next(s for s in specs if s['id'] == 'apa_sp2_fp32')['required']


@pytest.mark.parametrize('L,S', [(1,7), (3,7), (7,7)])
@pytest.mark.parametrize('causal', [False, True])
def test_bottom_right_contract(L, S, causal):
    mask = common.valid_mask(L, S, causal)
    expected = np.ones((L,S), bool)
    if causal:
        for i in range(L): expected[i, S-L+i+1:] = False
    np.testing.assert_array_equal(mask, expected)
    if L == 1: assert mask.all()


@pytest.mark.parametrize('L,S,H,KVH,causal', [(1,9,8,2,True), (3,9,8,2,True), (5,5,4,4,True), (3,9,8,2,False)])
def test_grouped_engine_and_real_torch_math_agree_with_independent_oracle(L,S,H,KVH,causal):
    shape = tiny(L,S,H,KVH,causal)
    q,k,v = common.inputs(shape, 7)
    # Different KV heads produce deliberately different outputs.
    v[:,1:] += 3
    want = independent_dense(q,k,v,shape)
    got = common.grouped_dense(NumpyEngine, *(NumpyTensor(x) for x in (q,k,v)), shape).numpy()
    np.testing.assert_allclose(got, want, atol=2e-6, rtol=2e-6)
    spec = next(s for s in common.registry() if s['id'] == 'torch_math_fp32')
    actual = common.sdpa_call(torch, spec, *(torch.from_numpy(x) for x in (q,k,v)), shape)
    np.testing.assert_allclose(actual.numpy(), want, atol=2e-6, rtol=2e-6)


def test_seed_and_dtype_rounding():
    a, b = common.inputs(tiny(), 4), common.inputs(tiny(), 4)
    assert all(np.array_equal(x,y) for x,y in zip(a,b))
    assert not np.array_equal(a[0], common.inputs(tiny(), 5)[0])
    values = np.array([0., 1.00390625, 1.01171875, -1.00390625, 0.0001, 17.32], np.float32)
    actual = common.rounded_host(values, 'bfloat16')
    np.testing.assert_array_equal(actual, torch.from_numpy(values).bfloat16().float().numpy())
    np.testing.assert_array_equal(common.rounded_host(values, 'float32'), values)
    with pytest.raises(ValueError): common.rounded_host(values, 'float16')


def test_accuracy_comparator_and_bad_shapes():
    want = np.array([3.,4.]); got = np.array([6.,8.])
    result = common.metrics(got, want)
    assert result['relative_frobenius'] == 1 and result['max_abs'] == 4
    assert common.metrics(np.zeros(2), np.zeros(2))['relative_frobenius'] == 0
    assert common.metrics(np.ones(2), np.zeros(2))['relative_undefined']
    assert common.metrics(np.ones(2), np.zeros(2))['relative_frobenius'] is None
    with pytest.raises(ValueError): common.metrics(np.ones(2), np.ones((1,2)))
    with pytest.raises(ValueError): common.metrics(np.array([]), np.array([]))


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_nonfinite_never_passes_or_writes_nan(bad):
    m = common.metrics(np.array([bad]), np.array([1.]))
    assert m['nonfinite_got'] == 1 and m['max_abs'] is None
    assert m['relative_frobenius'] is None
    json.dumps(m, allow_nan=False)


def test_timing_minimum_and_iqr():
    s = common.summarize_samples([1,2,3,4,5,6,7,8,9])
    assert s['median_ms'] == 5 and s['iqr_ms'] == 4
    for samples in ([1]*6, [], [0]*9, [float('nan')]*9, [-1]*9):
        with pytest.raises(ValueError): common.summarize_samples(samples)


def fixture_receipt(reg, pins):
    rows = []
    for spec in common.registry():
        row = dict(spec, status='OK', timing=common.summarize_samples([2.]*9),
                   peak_allocated_delta_bytes=1024, accuracy=common.metrics(np.ones(3),np.ones(3)),
                   fp32_parity=True)
        rows.append(row)
    return dict(shape=reg['shapes'][0], fingerprint=pins, status='COMPLETE', rows=rows)


def test_receipt_table_missing_stale_and_tampered():
    reg = common.registration(); pins = {'test': 'unit-test-only'}
    fixture = fixture_receipt(reg,pins)
    assembly = assemble(reg, [('unit-test-fixture.json', fixture)], pins)
    assert len(assembly['table']) == 600
    assert sum(r['status'] == 'OK' for r in assembly['table']) == 12
    assert sum(r['status'] == 'BLOCKED_NO_RECEIPT' for r in assembly['table']) == 588
    md, predictions = render(reg, assembly)
    assert common.SCOPE in md and '1/50' in md and 'B=2' in md
    assert predictions['P1_fp32_parity']['verdict'] == 'BLOCKED'
    assert len(assemble(reg, [('stale',fixture)], {'test':'different'})['rejected']) == 1
    for defect in ('short', 'dtype', 'duplicate', 'shape'):
        broken = copy.deepcopy(fixture)
        if defect == 'short': broken['rows'][0]['timing']['samples_ms'] = [1]*6
        if defect == 'dtype': broken['rows'][0]['dtype'] = 'bfloat16'
        if defect == 'duplicate': broken['rows'][-1] = broken['rows'][0]
        if defect == 'shape': broken['shape']['S'] = 999
        assert not common.receipt_valid(broken, reg, pins), defect


def test_prediction_empty_is_blocked_and_uncalibrated_not_matched():
    reg = common.registration()
    a = assemble(reg, [], {})
    preds = score_predictions(reg, a['table'])
    assert all(s['verdict'] == 'BLOCKED' for n,s in preds.items() if n != 'A5_paper_shape_mismatch')
    assert common.matching_fraction(reg['shapes'][-1], .15, .15)['within_002']
    assert not common.matching_fraction(reg['shapes'][-1], .15, .15)['matched_budget_eligible']
    assert not common.matching_fraction(reg['shapes'][0], .20, .15)['within_002']


def test_dry_run_no_device_import():
    # Block torch import entirely: dry-run must remain device/library independent.
    program = "import sys, runpy; sys.modules['torch']=None; sys.argv=['apa_spd1_bench.py','--dry-run']; sys.path.insert(0,'scripts'); runpy.run_path('scripts/apa_spd1_bench.py',run_name='__main__')"
    result = subprocess.run(['timeout','20s',sys.executable,'-B','-c',program], cwd=ROOT,
                            capture_output=True,text=True,timeout=25)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(rows) == 50 and len({r['id'] for r in rows}) == 50


def test_worker_without_lease_rejected():
    from apa_spd1_bench import worker
    old = os.environ.pop('APA_SPD1_LEASED', None)
    try:
        with pytest.raises(RuntimeError, match='leased runner'): worker('bad', '0')
    finally:
        if old is not None: os.environ['APA_SPD1_LEASED'] = old


def test_complete_collection_cannot_hide_red_numerics_or_blocked_sp2():
    reg = common.registration(); pins = {'fixture': 'test-only'}
    tiny_reg = dict(reg, shapes=reg['shapes'][:1])
    rec = fixture_receipt(reg, pins)
    rec['numeric_status'] = 'PASS'
    collection = {rec['shape']['id']: rec}
    assert gate_verdict(tiny_reg, collection).startswith('PASS')
    rec['numeric_status'] = 'RED'
    assert gate_verdict(tiny_reg, collection) == 'RED'
    rec['numeric_status'] = 'PASS'
    rec['rows'][-1]['status'] = 'BLOCKED_SP2_INTERFACE'
    assert gate_verdict(tiny_reg, collection).startswith('BLOCKED')
    rec['rows'][0]['status'] = 'UNAVAILABLE'
    assert not common.receipt_valid(rec, reg, pins)
