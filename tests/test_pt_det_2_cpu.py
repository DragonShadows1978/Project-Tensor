"""Author baseline, not blind verification or CUDA certification.

Prior art: PT-DET-1 (2026) host contracts, independent NumPy reference and
negative receipt fixtures; House Rules mutation-style counterexamples.
Ours: n-D destination collisions, family aliases and combined-slot boundaries.
"""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import pt_det_1 as d
import pt_det_2 as g
import pt_det_1_slot as slot
from test_pt_det_1_cpu import replay_fixture
from test_pt_det_1_cpu import run_record


@pytest.fixture(scope='session')
def cpu_gather(tmp_path_factory):
    return g.compile_cpu(tmp_path_factory.mktemp('gather_bridge'))


@pytest.mark.parametrize('shape,index_shape,dim', [
    ((7,), (0,), 0), ((7,), (27,), -1), ((5, 7), (3, 9), 1),
    ((5, 7), (13, 4), 0), ((2, 7, 3), (2, 9, 3), -2),
    ((2, 3, 7), (2, 3, 4), 2), ((0, 7), (0, 4), 1),
    ((2, 0), (2, 0), -1), ((2, 3, 2, 2, 2, 2, 2, 2), (2, 5, 2, 2, 2, 2, 2, 2), 1),
])
def test_actual_shared_address_and_sum_vs_fp64(cpu_gather, shape, index_shape, dim):
    rng = np.random.default_rng(97)
    ix = rng.integers(0, max(1, shape[dim]), size=index_shape, dtype=np.int64)
    grad = rng.standard_normal(index_shape, dtype=np.float32)
    ref = g.reference(shape, dim, ix, grad)
    outputs = [cpu_gather(shape, dim, ix, grad) for _ in range(5)]
    assert all(x.tobytes() == outputs[0].tobytes() for x in outputs)
    assert d.rel_l2(outputs[0], ref) <= 1e-6


def test_repeated_class_ids_are_not_duplicate_destinations(cpu_gather):
    ix = np.full((1, 4, 1), 2, np.int64)
    grad = np.array([1, 2, 3, 4], np.float32).reshape(ix.shape)
    got = cpu_gather((1, 4, 7), 2, ix, grad)
    assert np.array_equal(got[0, :, 2], [1, 2, 3, 4])
    assert np.count_nonzero(got) == 4


def test_true_collisions_cancellation_and_one_output_owner(cpu_gather):
    ix = np.full((2, 1024), 3, np.int64)
    grad = np.tile(np.array([1e8, 1, -1e8, 2], np.float32), (2, 256))
    got = cpu_gather((2, 7), -1, ix, grad)
    assert np.array_equal(got[:, 3], [768, 768])
    assert np.count_nonzero(got) == 2
    assert got.tobytes() == g.reference((2, 7), -1, ix, grad).astype(np.float32).tobytes()


@pytest.mark.parametrize('shape,ix,dim', [
    ((2, 7), np.zeros((2, 1), np.int64), 2),
    ((2, 7), np.zeros((2, 1), np.int64), -3),
    ((2, 7), np.zeros((3, 1), np.int64), 1),
    ((2, 7), np.full((2, 1), -1, np.int64), 1),
    ((2, 7), np.full((2, 1), 7, np.int64), 1),
    ((2, 0), np.zeros((2, 1), np.int64), 1),
    ((1,)*9, np.zeros((1,)*9, np.int64), 0),
    ((), np.array(0, np.int64), 0),
])
def test_invalid_gather_contract_rejected(cpu_gather, shape, ix, dim):
    with pytest.raises(ValueError):
        cpu_gather(shape, dim, ix, np.ones(ix.shape, np.float32))


@pytest.mark.parametrize('family,legacy,expected', [
    (None, None, False), (None, '1', True), (None, 'true', False),
    ('1', None, True), ('0', '1', False), ('', '1', False),
    ('true', '1', False), ('1', '0', True), ('1', '1', True), ('0', None, False),
])
def test_compiled_family_environment_aliases_and_tls(family, legacy, expected):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    for key, value in [('TC_DETERMINISTIC', family), ('TC_DET_EMBED_BWD', legacy)]:
        env.pop(key, None)
        if value is not None: env[key] = value
    result = subprocess.run([str(g.ART/'pt_det_2_host_contract'), str(int(expected))],
                            env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '5 exact CPU gather+embedding dispatches; no CUDA calls' in result.stdout


def test_python_family_api_and_legacy_aliases_without_cuda():
    code = '''import os
from pathlib import Path
import tensor_cuda as tc
assert Path(tc._C.__file__).resolve().is_relative_to(Path(os.environ['PYTHONPATH']))
assert tc.get_deterministic() and tc.get_deterministic_embed_bwd()
tc.set_deterministic_embed_bwd(False)
assert not tc.get_deterministic()
tc._C.set_deterministic(True)
assert tc._C.get_deterministic_embed_bwd()
tc.set_deterministic(False)
assert not tc.get_deterministic_embed_bwd()
assert callable(tc._C._gather_backward)
print('PT_DET_2 PYTHON_API: fork import and aliases only; no CUDA calls')
'''
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', TC_DETERMINISTIC='1', TC_DET_EMBED_BWD='0',
               PYTHONPATH=str(ROOT/'tensor_cuda'), PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, '-B', '-c', code], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr


def test_off_gather_atomic_and_dispatch_source_bytes_unchanged():
    old = (g.ART/'baseline/tensor_cuda/src/kernels.cu').read_bytes()
    new = (ROOT/'tensor_cuda/src/kernels.cu').read_bytes()
    def section(text, start, end): return text[text.index(start):text.index(end, text.index(start))]
    for start, end in [(b'__global__ void scatter_add_kernel', b'struct FlipSpec'),
                       (b'NDArray gather_nd(', b'NDArray scatter_add_nd('),
                       (b'__global__ void topk_kernel', b'NDArray flip_nd(')]:
        assert section(old, start, end) == section(new, start, end)
    old = section(old, b'NDArray scatter_add_nd(', b'\nnamespace {')
    new = section(new, b'NDArray scatter_add_nd(', b'\nnamespace {')
    assert old[old.index(b'  int nd ='):] == new[new.index(b'  int nd ='):]
    assert d.sha(ROOT/'tensor_cuda/src/ops.cpp') == g.registration()['baseline']['tensor_cuda/src/ops.cpp']


def test_registered_cases_keep_real_loss_and_collisions_separate():
    cases = list(g.cases())
    assert len(cases) == 6
    for case in cases:
        assert case['shape'] == [1, 4096, 8192] and case['source']['kind'] == 'real_CC46'
        k = case['index'].shape[-1]
        assert case['index'].shape == (1, 4096, k) and k in (1, 4)
        assert case['duplicate_destination_fraction'] == (0 if k == 1 else .75)
        if k == 4:
            assert np.all(case['index'] == case['index'][..., :1])
            assert np.array_equal(case['grad'][0, 0], [1e8, 1, -1e8, 2])


@pytest.mark.parametrize('arm', d.ARMS)
def test_family_replay_environment_and_missing_family_evidence_rejected(tmp_path, arm):
    env = d.family_environment(arm)
    assert env == dict(TC_DETERMINISTIC=str(int(arm.endswith('_on'))), TC_DET_EMBED_BWD=str(int(arm.endswith('_on'))))
    rec, manifest = replay_fixture(tmp_path/arm, arm)
    assert rec['checks']['family_modes']
    stdout = tmp_path/arm/'leg.stdout'
    stdout.write_text('\n'.join(line for line in stdout.read_text().splitlines() if 'PT_DETERMINISTIC' not in line)+'\n')
    bad = d.run_receipt(tmp_path/arm, dict(status='GREEN'), arm, d.tensor_schema(rec['checkpoint']), manifest)
    assert bad['status'] == 'RED' and not bad['checks']['family_modes']


def good_gather_rows():
    return [dict(name=f'{name}_k{k}', hashes=['fixture']*5, bitwise=True, relative_L2=0.,
                 atomic_ms=1., deterministic_ms=2., ratio=2., timing_samples_ms={'off': [1.]*10, 'on': [2.]*10})
            for name in ('x_32055', 'x_32083', 'x_32110') for k in (1, 4)]


@pytest.mark.parametrize('fault', ['missing', 'hash', 'one_call', 'error', 'nonfinite', 'ratio', 'untimed', 'functional'])
def test_gather_gate_rejects_missing_or_failed_evidence(fault):
    rows = good_gather_rows()
    functional = dict(gather_autograd=True, topk_autograd=True, dtype_conversions=True)
    assert g.assess(rows, gpu=True, functional=functional) == 'GREEN'
    if fault == 'missing': rows.pop()
    if fault == 'hash': rows[0]['hashes'][3] = 'changed'
    if fault == 'one_call': rows[0]['hashes'] = ['fixture']
    if fault == 'error': rows[0]['relative_L2'] = 1.000001e-6
    if fault == 'nonfinite': rows[0]['relative_L2'] = float('nan')
    if fault == 'ratio': rows[0].update(deterministic_ms=2.01, ratio=2.01, timing_samples_ms={'off': [1.]*10, 'on': [2.01]*10})
    if fault == 'untimed': rows[0]['timing_samples_ms']['on'] = []
    if fault == 'functional': functional['gather_autograd'] = False
    assert g.assess(rows, gpu=True, functional=functional) == 'RED'


def test_combined_slot_registered_budgets_and_full_replay():
    r = g.registration()['slot']; rows = slot.sequence(d.ART/'not_run', 9)
    assert [name for name, _, _ in rows] == r['sequence']
    assert [budget for _, budget, _ in rows] == [80, 80, 1320]
    assert sum(budget for _, budget, _ in rows) + r['finalize_reserve_seconds'] == r['total_seconds'] == 1500
    assert rows[-1][2][rows[-1][2].index('--steps')+1] == '30'
    assert rows[-1][2][rows[-1][2].index('--seconds')+1] == '1320'


def test_failed_first_slot_lane_blocks_remaining_lanes(tmp_path, monkeypatch):
    # Fictional host fixtures only: replace guard/manifest/process boundaries.
    # No device is made visible and no lock descriptor is inspected.
    monkeypatch.setattr(d, 'ART', tmp_path)
    monkeypatch.setattr(d, 'require_lead', lambda *args: {'fixture': True})
    monkeypatch.setattr(d, 'verify_manifest', lambda: {'binary': {'fixture': True}})
    (tmp_path/d.MANIFEST_NAME).write_text('fixture only')
    calls = []
    def fail_process(*args, **kwargs):
        calls.append(args[0]); return dict(status='RED', returncode=1)
    monkeypatch.setattr(d, 'run_process', fail_process)
    monkeypatch.setattr(sys, 'argv', ['pt_det_1_slot.py', '--out', str(tmp_path/'slot')])
    assert slot.main() == 1 and len(calls) == 1
    result = json.loads((tmp_path/'slot/SLOT_SUMMARY.json').read_text())
    assert result['lanes']['gather']['status'] == 'BLOCKED_PRIOR_GATE'
    assert result['lanes']['pt_det_1_repro']['status'] == 'BLOCKED_PRIOR_GATE'


def test_loss_probe_returns_original_value_and_records_all_bits(tmp_path):
    values = [np.float32(1), np.nextafter(np.float32(1), np.float32(2))]*15
    tensors = [SimpleNamespace(numpy=lambda x=x: np.asarray(x)) for x in values]
    iterator = iter(tensors)
    module = SimpleNamespace(nll_loss=lambda *args, **kwargs: next(iterator))
    path = tmp_path/'loss.jsonl'
    d.install_loss_probe(module, path)
    for expected in tensors:
        assert module.nll_loss('logits', 'targets', weights=None) is expected
    records = d.read_losses(path)
    assert len(records) == 30 and records[0]['sha256'] != records[1]['sha256']
    assert f'{float(values[0]):.4f}' == f'{float(values[1]):.4f}'


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'reorder', 'hash', 'nan', 'dtype', 'shape'])
def test_loss_records_reject_incomplete_or_malformed_evidence(tmp_path, fault):
    rows = [d.loss_record(np.asarray(1., np.float32), i) for i in range(1, 31)]
    if fault == 'missing': rows.pop()
    if fault == 'duplicate': rows[-1]['n'] = 1
    if fault == 'reorder': rows.reverse()
    if fault == 'hash': rows[0]['sha256'] = 'wrong'
    if fault == 'nan': rows[0]['value_hex'] = float('nan').hex()
    if fault == 'dtype': rows[0]['dtype'] = '<f8'
    if fault == 'shape': rows[0]['shape'] = [2]
    path = tmp_path/'loss.jsonl'; path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError): d.read_losses(path)


def test_full_precision_loss_change_cannot_hide_behind_rounded_text(tmp_path):
    a = run_record(tmp_path); b = copy.deepcopy(a)
    b['loss_steps'][4] = d.loss_record(np.asarray(np.nextafter(np.float32(1), np.float32(2))), 5)
    result = d.compare_pair(a, b)
    assert a['rows'] == b['rows'] and result['weights_bitwise']
    assert not result['bitwise'] and not result['losses_bitwise']


@pytest.mark.parametrize('own_group', [False, True])
def test_timeout_signals_only_owned_child_or_owned_group(tmp_path, monkeypatch, own_group):
    calls = []; pg_signals = []
    class FakeProcess:
        pid = 123456789; returncode = -15
        def communicate(self, *args, **kwargs): raise subprocess.TimeoutExpired('fixture', 1)
        def terminate(self): calls.append('terminate')
        def wait(self, *args, **kwargs): calls.append('wait')
    def popen(*args, **kwargs):
        calls.append(kwargs['start_new_session']); return FakeProcess()
    monkeypatch.setattr(d.subprocess, 'Popen', popen)
    monkeypatch.setattr(d.os, 'killpg', lambda pid, sig: pg_signals.append((pid, sig)))
    result = d.run_process(['fixture'], ROOT, {}, tmp_path/'timeout.log', 1, new_process_group=own_group)
    assert result['status'] == 'BLOCKED_TIMEOUT' and calls[0] == own_group
    assert pg_signals == ([(FakeProcess.pid, d.signal.SIGTERM)] if own_group else [])
    assert ('terminate' in calls) == (not own_group)
