"""Registered A5 CPU author gates; no GPU performance/quality claim.

Prior art: constructed-input and mutation testing, DeMillo/Lipton/Sayward
1978 (unverified lead: Hints on Test Data Selection). New SP3 fixtures.
"""
import copy
import ctypes
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
import apa_sp3_common as common
import apa_sp3_gpu as driver
import apa_sp3_a4_provenance as provenance
import apa_sp3_a5_registry as registry
import apa_sp3_a5_decode as pool
import apa_sp3_a5_report as report
import apa_sp3_model as model_module


def test_a5_registration_preserves_all_prior_cells_and_rails():
    cells = [c for c in driver.cells() if c['kind'] not in ('decode_clean','decode_repro','decode_bisect')]
    before = common.read(common.ART/'a5_before.json')
    by = {c['id']:c for c in cells}
    assert len(cells) == len(by) == 877
    assert all(by[c['id']] == c for c in before['cells'])
    new = [c for c in cells if c['kind']=='decode_pool']
    assert {c['id']:c for c in new} == {c['id']:c for c in registry.manifest()['cells']}
    assert len(new) == 18
    seen = set()
    for c in cells:
        assert set(c['depends']) <= seen
        seen.add(c['id'])
    for c in new:
        assert c['id'] == f"decode_pool_b{c['bits']}_{c['arm']}_{c['S']}"
        assert c['bits'] in (4,8) and c['arm'] in 'ABC' and c['S'] in (2048,8192,32768)
        assert c['steps'] == 32 and c['alloc_pooling'] is True
        assert c['worker_timeout_s'] == 290 and c['job_ceiling_s'] == 590
        assert f"freeze_b{c['bits']}" in c['depends']
        assert c['estimate_s'] == ([60,280] if c['S']<=8192 else None)
        if c['S']==32768:
            assert c['estimate_from'] == f"decode_pool_b{c['bits']}_{c['arm']}_8192"
            assert c['estimate_from'] in c['depends']


def worker_fixture(monkeypatch, tmp_path, *, short=False):
    cell = next(c for c in driver.cells() if c['id']=='decode_pool_b4_C_2048')
    events = []
    tc = SimpleNamespace(state=None, synchronize=lambda:None)
    def enable(value):
        events.append(('pool', value))
        tc.state=value
    tc.set_alloc_pooling=enable
    from contextlib import nullcontext
    tc.no_grad=nullcontext
    class Peak:
        def reset(self):
            assert tc.state is True
        def result(self):
            return dict(peak_resident_mib=4.,pool_reserved_peak_mib=4.,pool_used_peak_mib=3.,
                        peak_status='POOL ONLY',peak_source='test counter')
    class Logits:
        def float(self):return self
        def numpy(self):return np.ones((1,1,4))
    class Engine:
        def extend_rope(self,n):
            assert n==2080 and tc.state is True
        def __call__(self, ids, **kw):
            # Load-bearing pin INSIDE the worker's actual Model.decode loop:
            # pool must be True for cache prefill and EVERY measured token.
            assert tc.state is True
            events.append(('forward', ids.copy(), kw))
            return Logits(), 'cache'
    class Model(model_module.Model):
        def __init__(self):
            self.tc=tc
            tc.set_alloc_pooling(False)  # replicate existing raw-load constructor
            self.model=Engine()
            self.peak=Peak()
        def set(self, arm, bits, delta):
            assert tc.state is True
            assert (arm,bits,delta)==('C',4,2.)
        def decode(self,*args):
            result=super().decode(*args)
            if short:
                result['steps']=31
                result['seconds_per_step']=result['seconds_per_step'][:31]
            return result
    def guard(model,ids):
        assert model.tc.state is True
        events.append(('guard',True))
        return {'test':'same-worker controls saw pool ON'}
    monkeypatch.setenv('APA_SP3_LEASE','1')
    monkeypatch.setattr(model_module,'Model',Model)
    monkeypatch.setattr(pool,'PoolPeak',Peak)
    monkeypatch.setattr(pool,'ART',tmp_path)
    monkeypatch.setattr(driver,'ART',tmp_path)
    monkeypatch.setattr(common,'ART',tmp_path)
    (tmp_path/'protocol_amendment.json').write_text('{}')
    monkeypatch.setattr(common,'protocol',lambda:({},np.arange(2080,dtype=np.int64)))
    monkeypatch.setattr(pool,'require_pass',lambda id:dict(result=dict(delta=2.)))
    monkeypatch.setattr(driver,'g0_guard',guard)
    monkeypatch.setattr(driver,'verify_sources',lambda:None)
    monkeypatch.setattr(driver,'cells',lambda:[cell])
    monkeypatch.setattr(common,'job_path',lambda id:tmp_path/(id+'.json'))
    monkeypatch.setattr(common,'cell_fingerprint',lambda c:dict(test='fingerprint'))
    monkeypatch.setattr(provenance,'bridge',lambda:dict(effective_sha256='test-bridge'))
    return cell,events


def test_decode_pool_worker_pool_state_pin(monkeypatch,tmp_path):
    cell,events=worker_fixture(monkeypatch,tmp_path)
    assert driver.work(cell['id'])==0
    result=common.read(tmp_path/(cell['id']+'.json'))['result']
    forwards=[e for e in events if e[0]=='forward']
    assert events[:3]==[('pool',False),('pool',True),('guard',True)]
    assert len(forwards)==33
    np.testing.assert_array_equal(forwards[0][1],np.arange(2048)[None])
    for pos,(_,ids,kwargs) in enumerate(forwards[1:],2048):
        assert ids.tolist()==[[pos]] and kwargs['position_offset']==pos
        assert kwargs['kv_caches']=='cache' and kwargs['last_token_only'] is True
    assert result['steps']==32 and result['alloc_pooling'] is True
    assert result['ms_token']==1000*sum(result['seconds_per_step'])/32
    assert result['tokens_s']==32/sum(result['seconds_per_step'])
    assert result['timing']=='per-token wall with CUDA sync'
    assert result['fit'] is True


def test_decode_pool_worker_rejects_short_measurement(monkeypatch,tmp_path):
    cell,_=worker_fixture(monkeypatch,tmp_path,short=True)
    assert driver.work(cell['id'])==1
    r=common.read(tmp_path/(cell['id']+'.json'))
    assert r['status']=='RED' and 'DECODE_POOL_STEP_CONTRACT' in r['error']


def planning_fixture(monkeypatch,tmp_path,estimate):
    cell=next(c for c in driver.cells() if c['id']=='decode_pool_b4_B_32768')
    path=tmp_path/'source.json'
    path.write_text('measured source fixture')
    receipt=dict(cell=dict(kind='decode_pool'),result=dict(alloc_pooling=True,steps=32,
                 bits=4,arm='B',S=8192,fit=True,setup_s=10.,guard_s=5.,
                 prefill_s=10.,decode_work_s=(estimate-190.)/4))
    monkeypatch.setattr(pool,'require_pass',lambda id:receipt)
    monkeypatch.setattr(pool,'job_path',lambda id:path)
    monkeypatch.setattr(pool,'ART',tmp_path)
    return cell,receipt,path


@pytest.mark.parametrize('estimate,nonfit',[(289.,False),(290.,True),(291.,True),(900.,True)])
def test_pool_32k_measured_plan_rail_boundary(monkeypatch,tmp_path,estimate,nonfit):
    cell,r,path=planning_fixture(monkeypatch,tmp_path,estimate)
    plan=pool.pin_plan(cell)
    assert plan['estimate_s']==estimate
    assert (plan['fit'] is False)==nonfit
    assert plan['source_sha256']==common.sha(path)
    assert pool.pin_plan(cell)==plan
    path.write_text('changed receipt')
    with pytest.raises(common.Red,match='IMMUTABLE_DECODE_POOL_PLAN_CHANGED'):
        pool.pin_plan(cell)


@pytest.mark.parametrize('field,value',[('alloc_pooling',False),('fit',False),('steps',31),
                                      ('guard_s',float('nan')),('setup_s',-1),('S',2048),('bits',8),('arm','C')])
def test_pool_32k_bad_source_blocks(monkeypatch,tmp_path,field,value):
    cell,r,_=planning_fixture(monkeypatch,tmp_path,289.)
    r['result'][field]=value
    with pytest.raises(common.Red):pool.planning(cell)


def test_pool_32k_missing_measurement_blocks(monkeypatch):
    cell=next(c for c in driver.cells() if c['id']=='decode_pool_b4_C_32768')
    def missing(id):raise common.Red('BLOCKED_DEPENDENCY: '+id)
    monkeypatch.setattr(pool,'require_pass',missing)
    with pytest.raises(common.Red,match='BLOCKED_DEPENDENCY'):pool.planning(cell)


def test_nonfit_preflight_terminal_receipt_before_gpu(monkeypatch,tmp_path):
    import apa_sp3_control as control
    cell,r,path=planning_fixture(monkeypatch,tmp_path,290.)
    monkeypatch.setattr(control,'validate',lambda id:cell)
    monkeypatch.setattr(control,'verify_sources',lambda:None)
    monkeypatch.setattr(control,'protocol',lambda:None)
    monkeypatch.setattr(control,'require_pass',lambda id:r)
    monkeypatch.setattr(control,'ART',tmp_path)
    dest=tmp_path/'result.json'
    monkeypatch.setattr(control,'job_path',lambda id:dest)
    # Distinct source/dependency path and terminal result path.
    monkeypatch.setattr(pool,'job_path',lambda id:dest if id==cell['id'] else path)
    monkeypatch.setattr(common,'cell_fingerprint',lambda c:dict(test='fingerprint'))
    monkeypatch.setattr(provenance,'bridge',lambda:dict(effective_sha256='test'))
    (tmp_path/'protocol_amendment.json').write_text('{}')
    monkeypatch.setattr(model_module,'Model',lambda:pytest.fail('nonfit must not construct model'))
    assert control.preflight(cell['id'])=='NON_FIT'
    result=common.read(dest)
    assert result['status']=='PASS' and result['result']['fit'] is False
    assert result['result']['steps']==0 and result['result']['tokens_s'] is None
    assert result['result']['outcome']=='NON_FIT_PLANNED_RAIL'
    with pytest.raises(common.Red,match='existing immutable'):
        control.preflight(cell['id'])
    shell=(ROOT/'scripts/apa_sp3_lead_gpu.sh').read_text()
    assert shell.index('"$decision" == NON_FIT') < shell.index('exec bash "$0" _leased')
    assert 'timeout --signal=TERM --kill-after=5s "${rail}s"' in shell


def test_dense_long_nonfit_without_model_or_timing(monkeypatch):
    monkeypatch.setattr(pool,'require_pass',lambda id:pytest.fail('dense fit does not need timing estimate'))
    for c in driver.cells():
        if c['kind']=='decode_pool' and c['arm']=='A' and c['S']>=8192:
            p=pool.planning(c)
            assert p['fit'] is False and p['outcome']=='NON_FIT_REGISTERED_DENSE'


def test_pool_counter_source_and_failure(monkeypatch):
    calls=[]
    class Call:
        def __init__(self,name):self.name=name
        def __call__(self,*args):
            calls.append((self.name,args))
            if self.name=='cudaMemPoolGetAttribute':
                ctypes.cast(args[2],ctypes.POINTER(ctypes.c_uint64))[0]=(6 if args[1]==6 else 3)*(1<<20)
            return 0
    cuda=SimpleNamespace(**{n:Call(n) for n in ('cudaGetDevice','cudaDeviceGetDefaultMemPool',
                         'cudaMemPoolGetAttribute','cudaMemPoolSetAttribute')})
    monkeypatch.setattr(pool.ctypes,'CDLL',lambda path:cuda)
    peak=pool.PoolPeak()
    peak.reset()
    result=peak.result()
    assert result['pool_reserved_peak_mib']==6. and result['pool_used_peak_mib']==3.
    assert result['peak_resident_mib']==6. and 'NOT device resident' in result['peak_status']
    assert [args[1] for name,args in calls if name=='cudaMemPoolSetAttribute']==[6,8]
    cuda.cudaMemPoolGetAttribute=lambda *args:1
    with pytest.raises(common.Red,match='CUDA_POOL_COUNTER_FAILED'):peak.result()


def test_default_capture_gate_and_commands_dependency_order(tmp_path):
    cells=[c for c in driver.cells() if c['kind'] not in ('decode_clean','decode_repro','decode_bisect')]
    default=registry.default_cells(cells)
    captures={c['id'] for c in cells if c['kind'].startswith('capture_') and c.get('S')==32768}
    assert len(captures)==126
    assert not captures & {c['id'] for c in default}
    assert captures <= {c['id'] for c in registry.default_cells(cells,True)}
    assert not any(c['kind']=='decode' for c in default)
    report.render(tmp_path,{},cells)
    commands=(tmp_path/'lead_commands.txt').read_text()
    assert not any(id in commands for id in captures)
    assert 'run decode_b' not in commands
    ids=[line.split()[-1] for line in commands.splitlines() if 'apa_sp3_lead_gpu.sh run ' in line]
    assert len(ids)==len(default)
    seen=set()
    by={c['id']:c for c in cells}
    for id in ids:
        assert set(by[id]['depends']) <= seen
        seen.add(id)
    report.render(tmp_path,{},cells,True)
    assert all('run '+id in (tmp_path/'lead_commands.txt').read_text() for id in captures)


def test_next_default_does_not_select_32k_captures_or_raw_decode(monkeypatch,tmp_path,capsys):
    raw=dict(id='decode_b4_C_2048',kind='decode',bits=4)
    capture=dict(id='capture_b4_C_32768',kind='capture_aggregate',bits=4,S=32768)
    pooled=dict(id='decode_pool_b4_C_2048',kind='decode_pool',bits=4)
    clean_cell=dict(id='decode_clean_b4_C_2048',kind='decode_clean',bits=4)
    monkeypatch.setattr(driver,'cells',lambda:[raw,capture,pooled,clean_cell])
    monkeypatch.setattr(common,'job_path',lambda id:tmp_path/id)
    monkeypatch.setattr(sys,'argv',['test','--next'])
    monkeypatch.delenv('APA_SP3_INCLUDE_32K_CAPTURES',raising=False)
    assert driver.main()==0
    assert capsys.readouterr().out.strip()==clean_cell['id']
    monkeypatch.setattr(sys,'argv',['test','--next','--include-32k-captures'])
    assert driver.main()==0
    assert capsys.readouterr().out.strip()==capture['id']


def test_p5_uses_only_pool_on_primary_valid_measurements():
    def receipt(speed,**kw):
        return dict(status='PASS',result=dict(fit=True,alloc_pooling=True,steps=32,tokens_s=speed,**kw))
    raw={f'decode_b4_{arm}_32768':receipt(speed) for arm,speed in [('B',1.),('C',9.)]}
    assert report.prediction(raw)['status']=='UNASSESSED'
    measurements={f'decode_pool_b4_{arm}_32768':receipt(speed) for arm,speed in [('B',10.),('C',20.)]}
    assert report.prediction({**raw,**measurements})['status']=='HIT'
    assert report.prediction(measurements)['ratio']==2.
    measurements['decode_pool_b4_C_32768']['result']['tokens_s']=19.99
    assert report.prediction(measurements)['status']=='MISSED'
    for field,value in [('fit',False),('steps',0),('alloc_pooling',False),('tokens_s',None)]:
        bad=copy.deepcopy(measurements)
        bad['decode_pool_b4_C_32768']['result'][field]=value
        assert report.prediction(bad)['status']=='UNASSESSED'
    measurements['decode_pool_b4_C_32768']['status']='STALE'
    assert report.prediction(measurements)['status']=='UNASSESSED'


def test_a5_bridge_accepts_every_a4_kind_and_rejects_future_edit():
    before=common.read(common.ART/'a5_before.json')
    amendment=provenance.bridge()
    for kind,paths in before['closures'].items():
        old={p:before['files'].get(p,common.sha(ROOT/p)) for p in paths}
        receipt=dict(cell=dict(kind=kind),registration_sha256=common.REG_SHA,
                     fingerprint=old,fingerprint_schema='apa_sp3_per_kind_v1',
                     fingerprint_amendment_sha256=before['effective_a4_sha256'])
        now=provenance.current_fingerprint(receipt['cell'])
        assert provenance.compatible(receipt,current=now,amendment=amendment),kind
        bad=dict(now)
        bad['scripts/apa_sp3_gpu.py']='unreviewed-future-source'
        assert not provenance.compatible(receipt,current=bad,amendment=amendment)
        wrong=dict(receipt,fingerprint_amendment_sha256='unknown-bridge')
        assert not provenance.compatible(wrong,current=now,amendment=amendment)
    pool_closure=provenance.closure(dict(kind='decode_pool'))
    assert 'scripts/apa_sp3_model.py' in pool_closure and 'scripts/apa_sp3_a5_decode.py' in pool_closure
    assert all(common.sha(ROOT/p)==v for p,v in before['receipts'].items())


def test_a5_new_receipt_rejects_changed_closure():
    cell=dict(kind='decode_pool')
    now=provenance.current_fingerprint(cell)
    m=provenance.bridge()
    r=dict(cell=cell,registration_sha256=common.REG_SHA,fingerprint=now,
           fingerprint_schema='apa_sp3_per_kind_v1',fingerprint_amendment_sha256=m['effective_sha256'])
    assert provenance.compatible(r,current=now,amendment=m)
    assert not provenance.compatible(r,current=dict(now,**{'scripts/apa_sp3_a5_decode.py':'changed'}),amendment=m)
