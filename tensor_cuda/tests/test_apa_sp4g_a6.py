"""A6 author CPU gates. Prior art: SP4G A4/A5 (2026) receipt and
instrumentation negative tests; DeMillo/Lipton/Sayward (1978) mutation
testing, unverified — lead to check Hints on Test Data Selection.
No GPU numerics or blind-review claim.
"""
import copy, importlib.util, json, os, sys, time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R/'scripts'))
import apa_sp4g_a6_common as common
import apa_sp4g_a6_model as model
import apa_sp4g_a6_gpu as gpu
import apa_sp4g_a6_registry as registry
import apa_sp4g_a6_report as report
from test_apa_sp4g_a4 import T, fixture_tc, tensors, pin_records

if os.environ.get('APA_SP4G_A6_MUTANT'):
    name, path = os.environ['APA_SP4G_A6_MUTANT'].split(':', 1)
    spec = importlib.util.spec_from_file_location('a6_mutant', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    globals()[name] = module


def profile(first=.005, late=.1):
    return [dict(layer=str(l), relative_frobenius=first if l == 5 else late, max_abs=1., rows=2047)
            for l in list(range(5, 48, 6))+['final_norm']]


def test_a6_registration_preserved_and_literal_bound_failures_reported():
    before = common.preserved()
    assert len(before['receipt_sha256']) >= 82
    reg = common.read(registry.REGISTRATION)
    assert common.sha(registry.REGISTRATION) == registry.REGISTRATION_SHA
    assert reg['original_registration_sha256'] == common.REG_SHA
    assert reg['exactness']['lead_verdict'] == 'PASS'
    assert reg['exactness']['literal_bounds_verdict'] == 'RED'
    rows = reg['exactness']['per_call']
    assert len(rows) == 5
    assert [r['literal_bounds_pass'] for r in rows] == [True, False, True, False, True]
    for r in rows:
        assert common.sha(R/r['path']) == r['sha256']
        raw = common.read(R/r['path'])['result']['comparisons']['SP_vs_A_fp32']
        assert (r['max_abs'], r['relative_frobenius']) == (raw['max_abs'], raw['relative_frobenius'])
    assert reg['noise_floor']['measured_absolute_ppl'] == pytest.approx(2.5593081470667265)
    assert reg['noise_floor']['historical_gate']['tolerance'] == .005
    assert not common.a5.a4.exactness(common.a5.completed_d32()['result'])


def test_a6_floor_inclusive_and_direction():
    assert common.floor_comparison(3.56, 1.)['inside_floor']
    assert common.floor_comparison(1., 3.56)['inside_floor']
    assert common.floor_comparison(1., 1.)['verdict'] == 'not resolvable on this model'
    assert not common.floor_comparison(3.560001, 1.)['inside_floor']
    assert common.floor_comparison(49.41041563636506, 52.48048708520552)['difference'] < -2.56


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1., 0.])
def test_a6_floor_nonfinite_rejected(bad):
    with pytest.raises(common.Red, match='INVALID_PPL'):
        common.floor_comparison(bad, 50.)
    with pytest.raises(common.Red, match='INVALID_PPL'):
        common.floor_comparison(50., bad)


def test_a6_flat_and_intermediate_stop():
    assert common.propagation_verdict(profile())['C_E_unblocked']
    for late, outcome in ((.01, 'FLAT'), (.02, 'INCONCLUSIVE'), (.049, 'INCONCLUSIVE')):
        r = common.propagation_verdict(profile(late=late))
        assert r['outcome'] == outcome and not r['C_E_unblocked']
        assert 'stop' in r['stop_reason']
    assert common.propagation_verdict(profile(first=.01, late=.05))['C_E_unblocked']
    assert not common.propagation_verdict(profile(first=.02, late=.05))['C_E_unblocked']


def test_a6_profile_requires_full_population_and_finite():
    rows = profile(); rows[0]['rows'] = 2046
    with pytest.raises(common.Red, match='INCOMPLETE'):
        common.propagation_verdict(rows)
    for rows in (profile()[:-1], list(reversed(profile()))):
        with pytest.raises(common.Red, match='INCOMPLETE'):
            common.propagation_verdict(rows)
    rows = profile(); rows[4]['relative_frobenius'] = float('nan')
    with pytest.raises(common.Red, match='NONFINITE'):
        common.propagation_verdict(rows)


def test_a6_dag_unblocks_all_C_E_from_old_gate_but_requires_propagation():
    ancestors = {}; cs = registry.cells()
    for c in cs:
        assert c['id'] not in ancestors and c['worker_s'] == 285 and c['S'] <= 8192
        assert all(d in ancestors or d not in registry.by_id() for d in c['depends'])
        ancestors[c['id']] = set(c['depends']).union(*(ancestors.get(d, set()) for d in c['depends']))
        if c['arm'] in ('C', 'E'):
            assert registry.DIAG in ancestors[c['id']]
            assert 'ppl_a4_D32_2048_w0' not in ancestors[c['id']]
    assert len(cs) == 45
    assert sum(c['kind'] == 'trial' for c in cs) == 12
    assert sum(c['kind'] == 'margin' for c in cs) == 16
    assert sum(c['kind'] == 'decode' for c in cs) == 2
    for arm in 'CE':
        assert {c['window'] for c in cs if c['arm'] == arm and c['kind'] in ('ppl', 'ppl_capture') and c['S'] == 2048} == set(range(4))
        assert sum(c['arm'] == arm and c['S'] == 8192 and c['kind'] in ('ppl', 'ppl_capture') for c in cs) == 1
    assert {c['layer'] for c in cs if c['kind'] == 'margin'} == set(range(5, 48, 6))


def test_a6_preflight_flat_blocks_every_C_E_kind_before_execution(monkeypatch, tmp_path):
    monkeypatch.setattr(gpu, 'preserved', lambda: None)
    monkeypatch.setattr(gpu, 'tokens', lambda: None)
    monkeypatch.setattr(gpu, 'verify_weight', lambda: None)
    monkeypatch.setattr(gpu, 'read', lambda p: dict(status='PASS_CPU_ONLY', fingerprint_amendment_sha256='seal', fingerprint={}, execution_sha256={}))
    monkeypatch.setattr(gpu, 'sha', lambda p: 'seal')
    monkeypatch.setattr(gpu, 'fingerprint', lambda c: {})
    monkeypatch.setattr(gpu, 'path_a6', lambda n: tmp_path/(n+'.json'))
    def stop(n):
        raise common.Red('A6_RED_RECEIPT: propagation FLAT')
    monkeypatch.setattr(gpu, 'require_a6', stop)
    for c in registry.cells():
        if c['arm'] in ('C', 'E'):
            with pytest.raises(common.Red, match='FLAT'):
                gpu.preflight(c)


def receipt_fixture():
    c = registry.by_id()[registry.DIAG]
    pins = [dict(p, implementation='standard') for p in pin_records()]
    r = dict(per_layer=profile(), decision=common.propagation_verdict(profile()), dtype_pins=pins,
             dtype_pin_complete=True, same_schedule=True, reference_ppl_bitwise=True, arm='A32',
             native_call_count=144, targets=1024)
    return c, dict(cell=c, status='PASS', registration_sha256=common.REG_SHA,
                   a6_registration_sha256=registry.REGISTRATION_SHA, fingerprint={'source': 'good'},
                   dependencies={}, result=r, error=None)


def validate(c, j, **kw):
    return common.validate_receipt(j, c, {'source': 'good'}, {}, **kw)


def test_a6_fingerprint_red_payload_and_precision_pins_rejected(tmp_path):
    c, j = receipt_fixture(); validate(c, j)
    j['fingerprint'] = {'source': 'bad'}
    with pytest.raises(common.Red, match='STALE'): validate(c, j)
    c, j = receipt_fixture(); j['dependencies'] = {'unexpected': 'source'}
    with pytest.raises(common.Red, match='STALE'): validate(c, j)
    c, j = receipt_fixture(); j['status'] = 'RED'; j['result'] = {}
    with pytest.raises(common.Red, match='RED_RECEIPT'): validate(c, j)
    c, j = receipt_fixture(); j['result']['dtype_pins'][0]['implementation'] = 'SP'
    with pytest.raises(common.Red, match='PRECISION'): validate(c, j)
    c, j = receipt_fixture(); p = tmp_path/'payload'; p.write_bytes(b'ok')
    j['result']['files'] = [dict(path=str(p), sha256=common.sha(p))]; p.write_bytes(b'bad')
    with pytest.raises(common.Red, match='PAYLOAD'): validate(c, j)


def test_a6_completed_flat_is_reportable_but_cannot_unblock():
    c, j = receipt_fixture(); j['status'] = 'RED'
    j['result']['per_layer'] = profile(late=.005)
    decision = common.propagation_verdict(j['result']['per_layer'])
    j['result']['decision'] = decision; j['error'] = decision['stop_reason']
    with pytest.raises(common.Red, match='RED_RECEIPT'): validate(c, j)
    assert validate(c, j, allow_completed_red=True)['result']['decision']['outcome'] == 'FLAT'
    j['status'] = 'PASS'
    with pytest.raises(common.Red, match='STOP'): validate(c, j)


def test_a6_residual_sums_not_chunk_average_and_coverage(monkeypatch, tmp_path):
    monkeypatch.setattr(gpu, 'R', tmp_path); results = []
    for arm in ('A', 'A32'):
        residuals = {}
        for layer in [str(l) for l in range(5, 48, 6)]+['final_norm']:
            records = []
            for i, v in enumerate((1., 10.)):
                p = tmp_path/f'{arm}_{layer}_{i}.npy'
                np.save(p, np.full((1, 1, 3840), v+(1. if arm == 'A32' else 0.), np.float32))
                records.append(dict(lo=i, n=1, file=dict(path=str(p))))
            residuals[layer] = records
        p = tmp_path/(arm+'.json'); p.write_text(json.dumps(dict(arm=arm, S=3, residuals=residuals)))
        results.append(dict(manifest=str(p)))
    rows = gpu.compare_residuals(*results, S=3)
    assert len(rows) == 9
    assert all(r['relative_frobenius'] == pytest.approx(np.sqrt(2/101)) and r['max_abs'] == 1 for r in rows)
    p = tmp_path/'A32.json'; m = json.loads(p.read_text()); m['residuals']['5'][1]['lo'] = 2
    p.write_text(json.dumps(m))
    with pytest.raises(common.Red, match='COVERAGE'): gpu.compare_residuals(*results, S=3)


def test_a6_A32_capture_uses_standard_precision_and_restores(monkeypatch):
    calls = []
    class Block:
        def __call__(self, x, ropes, position_offset=0, cache=None):
            return T(11., shape=(1, 2, 3840)), cache
    original = Block.__call__; norm = lambda h: T(17., shape=h.shape)
    def setup(self, cell):
        calls.append(cell['arm']); self.cell = dict(cell, treatment=cell['arm'])
        self.gemma = NS(Gemma4BlockTC=Block, _cast=lambda t: t.astype('bfloat16'))
        self.model = NS(layers=[Block() for _ in range(48)], norm=norm)
        self.tc, self.seen = fixture_tc(); self.probes = []; self.active_layer = 5
    monkeypatch.setattr(model.PrecisionModel, '__init__', setup)
    closed = []; monkeypatch.setattr(model.PrecisionModel, 'close', lambda self: closed.append(True))
    owner = model.A32ResidualModel(dict(id='fixture', arm='A32'))
    saved = []; owner.save_residual = lambda l, t, off: saved.append((l, t.value, t.dtype, off))
    out = owner.dispatch(*tensors(), 1., 1., True)
    assert calls == ['A32'] and out.value == 7. and out.dtype == 'bfloat16' and not owner.seen
    assert owner.probes[0]['implementation'] == 'standard'
    assert owner.probes[0]['native_inputs'] == {n: 'float32' for n in ('q', 'k', 'kq', 'v')}
    h, cache = owner.model.layers[47](T(), None, 512, 'cache')
    owner.model.norm(h)
    assert saved == [('47', 11., 'bfloat16', 512), ('final_norm', 17., 'bfloat16', 512)]
    owner.close(); assert Block.__call__ is original and owner.model.norm is norm and closed == [True]


def test_a6_calibration_match_uses_real_B_fraction_and_tolerance():
    target = .15160508148834423
    r = gpu.freeze_match([('x', dict(fraction=target+.009, delta=3.))], target)
    assert r['delta'] == 3. and r['target'] == target
    for fraction in (target+.0101, target-.0101, float('nan')):
        with pytest.raises(common.Red, match='MATCH_FAILED'):
            gpu.freeze_match([('x', dict(fraction=fraction, delta=3.))], target)


def test_a6_clean_decode_never_installs_guard(monkeypatch):
    def forbidden(*args):
        raise AssertionError('timed path wrapped')
    monkeypatch.setattr(gpu, 'deadline_guard', forbidden)
    with gpu.measurement_context(NS(), 'decode', time.monotonic()+60):
        pass
    with pytest.raises(AssertionError, match='wrapped'):
        gpu.measurement_context(NS(), 'ppl', time.monotonic()+60)
    with pytest.raises(common.Red, match='RAIL'):
        gpu.check_deadline(time.monotonic()-1)


def test_a6_report_every_pair_has_floor_and_P2_sign(monkeypatch):
    def result(name):
        arm = name.split('_')[-3] if '_w' in name else name.split('_')[-2]
        return dict(ppl=dict(A=50., B=52., C=51., D=53., E=54.)[arm]), None
    monkeypatch.setattr(report, 'result_or_pending', result)
    monkeypatch.setattr(report, 'sha', lambda p: 'receipt')
    rows = report.table_rows()
    assert len(rows) == 6
    for r in rows:
        assert len(r['pairwise']) == 10
        assert r['P2']['difference'] == -1. and r['P2']['inside_floor']
        assert all(p['floor'] == 2.56 for p in r['pairwise'].values())


def test_a6_margin_namespace_and_bitwise_replay_rejection():
    import apa_sp4g_a6_metrics as metrics
    a = np.ones((2, 3)); raw = dict(out=a, mask=a.astype(bool))
    metrics.require_exact_replay(a, a.astype(bool), raw)
    with pytest.raises(common.Red, match='NATIVE_REPLAY'):
        metrics.require_exact_replay(a+1, a.astype(bool), raw)
    with pytest.raises(common.Red, match='NATIVE_SELECTION'):
        metrics.require_exact_replay(a, np.zeros_like(a, dtype=bool), raw)
    assert 'margin_errors_a6' in (R/'scripts/apa_sp4g_a6_metrics.py').read_text()


def test_a6_create_only_publish_and_no_signals_or_trace(tmp_path):
    p = tmp_path/'receipt.json'; common.publish(p, dict(status='RED'))
    with pytest.raises(FileExistsError): common.publish(p, dict(status='PASS'))
    assert json.loads(p.read_text())['status'] == 'RED'
    for p in list(R.glob('scripts/apa_sp4g_a6_*.py'))+[R/'scripts/apa_sp4g_a6_lead_gpu.sh']:
        if 'register' not in p.name and 'report' not in p.name and 'mutations' not in p.name:
            s = p.read_text()
            assert not any(x in s for x in ('os.kill(', 'signal.alarm(', 'timeout=', 'sys.settrace('))
