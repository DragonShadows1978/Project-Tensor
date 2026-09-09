"""A6 author CPU gates; not blind verification or GPU speed evidence.
Prior art: SP3 constructed-input/mutation testing (2026); DeMillo/Lipton/
Sayward 1978, unverified lead: Hints on Test Data Selection. New fixtures.
"""
import copy
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import apa_sp3_common as common
import apa_sp3_gpu as driver
import apa_sp3_a6_registry as registry
import apa_sp3_a6_decode as clean
import apa_sp3_a6_report as report
import apa_sp3_a4_provenance as provenance


def cell(name='decode_clean_b4_A_2048'):
    return copy.deepcopy(next(c for c in registry.manifest()['cells'] if c['id']==name))


def test_a6_registration_old_cells_ladder_and_rails():
    cs=driver.cells();old=common.read(common.ART/'a6_before.json')['cells']
    by={c['id']:c for c in cs}
    assert len(cs)==len(by)==908
    assert all(by[c['id']]==c for c in old)
    new=[c for c in cs if c['kind'] in registry.KINDS]
    assert new==registry.manifest()['cells'] and len(new)==31
    assert len([c for c in new if c['kind']=='decode_clean'])==18
    seen=set()
    for c in cs:
        assert set(c['depends'])<=seen
        seen.add(c['id'])
    base=cell('decode_bisect_00_reference')['config']
    rungs=[c for c in new if c['kind']=='decode_bisect']
    assert len(rungs)==12
    for c in rungs[1:11]:
        assert [k for k in base if base[k]!=c['config'][k]]==[c['change']]
        assert c['compare_to']==rungs[0]['id']
        assert not any(d.startswith('decode_bisect') for d in c['depends'])
    for c in new:
        assert c['worker_timeout_s']==290 and c['job_ceiling_s']==590
        assert c['steps']==32 and c['warmup_steps']==1
    selected=registry.default_cells(cs)
    assert all(c['kind'] not in ('decode','decode_pool') for c in selected)
    assert all(c in selected for c in new)


def fixture(monkeypatch,tmp_path,name='decode_clean_b4_A_2048'):
    c=cell(name);events=[];state=NS(pool=None)
    class Attention:
        def __call__(self,*a,**k):return None
    original=Attention.__call__
    class Logits:
        shape=[1,1,73448]
        def float(self):events.append(('full_float',));return self
        def numpy(self):events.append(('full_copy',));return np.ones((1,1,73448),np.float32)
    def cache(n):return [(NS(shape=[1,n,256]),NS(shape=[1,1,n,32])) for _ in range(62)]
    class Engine:
        def __init__(self):self.layers=[NS(self_attn=Attention()) for _ in range(62)]
        def extend_rope(self,n):assert n==2080
        def __call__(self,ids,**kw):
            assert state.pool is True
            assert base.QuantLinearTC.FUSED_DECODE is c['config']['fused_decode']
            assert base.RMSNormTC.USE_FUSED is c['config']['fused_rms_norm']
            assert base.F.USE_FUSED_SOFTMAX is c['config']['fused_softmax']
            if c['kind']=='decode_clean':
                assert Attention.__call__ is original
                assert base._cublas_blend_attention is blend
            if c['config']['attention_wrapper']:
                self.layers[0].self_attn()
            events.append(('forward',ids.copy(),kw))
            pos=kw.get('position_offset',0)
            return Logits(),cache(pos+ids.shape[1])
    def load(snapshot):
        events.append(('load',state.pool))
        assert state.pool is c['config']['pool_before_load']
        return Engine(),dict(weight_bits=4)
    def pooling(value):state.pool=value;events.append(('pool',value))
    def scalar(lg):
        events.append(('argmax',))
        return NS(numpy=lambda:np.array([[7]],dtype=np.int64))
    blend=lambda *a:None
    selective=lambda *a:None
    native=lambda *a:events.append(('native',a)) or 'SP'
    tc=NS(set_alloc_pooling=pooling,no_grad=nullcontext,synchronize=lambda:None,
          argmax_last_axis=scalar,apa_selective_attention=selective,
          _C=NS(apa_selective_attention_sp=native))
    mini=NS(MLAAttentionTC=Attention,MiniCPM3_TC=NS(from_pretrained=load))
    base=NS(QuantLinearTC=NS(),RMSNormTC=NS(),F=NS(),BlockTC=NS(COMPUTE_DTYPE='bfloat16'),
            LinearTC=NS(DTYPE='bfloat16'),GROUP_SIZE=128,_cublas_blend_attention=blend)
    class Peak:
        def reset(self):events.append(('peak_reset',))
        def result(self):return dict(pool_reserved_peak_mib=12.,pool_used_peak_mib=10.,peak_resident_mib=12.)
    monkeypatch.setattr(clean,'check_interposer',lambda enabled:dict(loaded=enabled))
    monkeypatch.setattr(clean,'verify_weight_stat',lambda:None)
    monkeypatch.setattr(clean,'load_runtime',lambda:tc)
    monkeypatch.setattr(clean,'load_adapter',lambda tc:(mini,base))
    monkeypatch.setattr(clean,'PoolPeak',Peak)
    monkeypatch.setattr(clean,'protocol',lambda:({},np.arange(2080,dtype='<i8')%73448))
    monkeypatch.setattr(clean,'require_pass',lambda n:dict(result=dict(delta=.25)))
    monkeypatch.setattr(clean,'ART',tmp_path)
    return c,events,tc,mini,base,original


@pytest.mark.parametrize('arm',['A','B','C'])
def test_decode_clean_no_attention_hook_installed(monkeypatch,tmp_path,arm):
    c,events,tc,mini,base,original=fixture(monkeypatch,tmp_path,f'decode_clean_b4_{arm}_2048')
    original_api=tc.apa_selective_attention
    owner=clean.CleanModel(c,.25 if arm=='C' else None)
    assert mini.MLAAttentionTC.__call__ is original
    assert not hasattr(owner,'layer_ids')
    assert not hasattr(owner,'diag')
    owner.assert_no_hook()
    if arm=='C':
        assert tc.apa_selective_attention('q','k','kq','v',.125,1.28,True)=='SP'
        assert events[-1]==('native',('q','k','kq','v',.125,.25,True,None,False))
    else:assert tc.apa_selective_attention is original_api
    result=clean.measure(owner,np.arange(2080,dtype='<i8'))
    assert result['steps']==32
    owner.close()
    assert mini.MLAAttentionTC.__call__ is original and tc.apa_selective_attention is original_api


@pytest.mark.parametrize('name',['decode_clean_b4_A_2048','decode_repro_b4_A_2048',
                               'decode_bisect_01_wrapper','decode_bisect_02_host_logits',
                               'decode_bisect_03_last_token_only','decode_bisect_04_cache_recompute',
                               'decode_bisect_05_pool_after','decode_bisect_07_int4_eager',
                               'decode_bisect_08_norm_eager','decode_bisect_09_expanded_mla',
                               'decode_bisect_10_softmax_eager','decode_bisect_11_legacy_stack'])
def test_clean_worker_load_cache_copies_and_timing(monkeypatch,tmp_path,name):
    c,events,tc,mini,base,original=fixture(monkeypatch,tmp_path,name)
    result=driver.execute(c)
    forwards=[e for e in events if e[0]=='forward']
    assert len(forwards)==34  # one prefill, one discarded warmup, 32 measured
    assert events[:2]==[('pool',c['config']['pool_before_load']),('load',c['config']['pool_before_load'])]
    initial_cache=forwards[1][2].get('kv_caches')
    for i,(_,ids,kw) in enumerate(forwards[2:]):
        pos=2048+i
        assert kw['last_token_only'] is c['config']['last_token_only']
        if c['config']['use_cache']:
            assert ids.tolist()==[[7 if c['feeding']=='greedy' else pos]]
            assert kw['position_offset']==pos
            assert kw['kv_caches'][0][0].shape[1]==pos
            if i==0:assert kw['kv_caches'] is initial_cache
        else:
            assert ids.tolist()==[list(range(pos+1))] and 'kv_caches' not in kw
    assert len([e for e in events if e[0]=='full_copy'])==(33 if c['config']['full_logits_host_copy'] else 1)
    assert len([e for e in events if e[0]=='argmax'])==(33 if c['feeding']=='greedy' else 32)
    assert result['ms_token']==1000*sum(result['seconds_per_step'])/32
    assert result['tokens_s']==32/sum(result['seconds_per_step'])
    assert all(s>=f for s,f in zip(result['seconds_per_step'],result['forward_seconds_per_step']))
    assert result['pool_reserved_peak_mib']==12. and result['fit'] is True
    assert mini.MLAAttentionTC.__call__ is original


def test_hook_detection_and_cache_shape_reject():
    with pytest.raises(common.Red,match='CACHE_GEOMETRY'):clean.cache_pin([],2048)
    with pytest.raises(common.Red,match='CACHE_GEOMETRY'):
        clean.cache_pin([(NS(shape=[1,2047,256]),NS(shape=[1,1,2048,32]))]*62,2048)


def test_clean_rejects_hook_added_in_constructor(monkeypatch,tmp_path):
    c,events,tc,mini,base,original=fixture(monkeypatch,tmp_path)
    load=mini.MiniCPM3_TC.from_pretrained
    def bad(snapshot):
        model,info=load(snapshot)
        mini.MLAAttentionTC.__call__=lambda *a:None
        return model,info
    mini.MiniCPM3_TC.from_pretrained=bad
    with pytest.raises(common.Red,match='DIAGNOSTIC_HOOK'):clean.CleanModel(c)
    assert mini.MLAAttentionTC.__call__ is original


@pytest.mark.parametrize('env,loaded,symbol,enabled,ok',[
    ('',False,False,False,True),('/foreign.so',False,False,False,False),
    ('',True,False,False,False),('',False,True,False,False),
    ('peak.so',True,True,True,True),('',False,False,True,False)])
def test_no_interposer_process_state(monkeypatch,env,loaded,symbol,enabled,ok):
    monkeypatch.setenv('LD_PRELOAD',env)
    monkeypatch.setattr(clean.Path,'read_text',lambda p:'libapa_sp3_peak.so' if loaded else '')
    monkeypatch.setattr(clean.ctypes,'CDLL',lambda p:NS(**({'apa_sp3_live_bytes':True} if symbol else {})))
    if ok:assert clean.check_interposer(enabled)['loaded']==loaded
    else:
        with pytest.raises(common.Red,match='INTERPOSER|PRELOAD'):clean.check_interposer(enabled)


def test_interposer_shell_new_kind_policy():
    for c in driver.cells():
        assert registry.needs_interposer(c)==(c['config']['interposer'] if c['kind'] in registry.KINDS else c['kind']!='torch_reference')
    shell=(ROOT/'scripts/apa_sp3_lead_gpu.sh').read_text()
    assert 'scripts/apa_sp3_control.py interposer "$job"' in shell
    assert 'if [[ "$observer" == 1 ]]; then' in shell
    start=shell.index('if [[ "$observer" == 1 ]]; then')
    assert 'unset LD_PRELOAD' in shell[start:shell.index('mkdir -p logs',start)]
    assert shell.index('preflight "$job"')<shell.index('flock --exclusive')


def planning_fixture(monkeypatch,tmp_path,estimate):
    c=cell('decode_clean_b4_B_32768');source=cell('decode_clean_b4_B_8192')
    r=dict(fit=True,steps=32,bits=4,arm='B',S=8192,config=source['config'],feeding='teacher_forced',
           setup_s=10.,prefill_s=10.,warmup_s=1.,decode_work_s=(estimate-189.)/4)
    path=tmp_path/'source.json';path.write_text('fixture')
    monkeypatch.setattr(clean,'require_pass',lambda n:dict(cell=source,result=r))
    monkeypatch.setattr(clean,'job_path',lambda n:path)
    monkeypatch.setattr(clean,'ART',tmp_path)
    return c,r,path


@pytest.mark.parametrize('estimate,nonfit',[(289,False),(290,True),(291,True),(800,True)])
def test_clean_32k_plan_rail(monkeypatch,tmp_path,estimate,nonfit):
    c,r,p=planning_fixture(monkeypatch,tmp_path,estimate)
    plan=clean.pin_plan(c)
    assert plan['estimate_s']==estimate
    assert (plan['fit'] is False)==nonfit
    assert clean.pin_plan(c)==plan
    p.write_text('changed')
    with pytest.raises(common.Red,match='IMMUTABLE_PLAN'):clean.pin_plan(c)


@pytest.mark.parametrize('key,value',[('steps',31),('S',2048),('arm','C'),('bits',8),('fit',False),
                                     ('feeding','greedy'),('config',{}),('setup_s',-1),('prefill_s',float('nan'))])
def test_clean_32k_bad_source(monkeypatch,tmp_path,key,value):
    c,r,_=planning_fixture(monkeypatch,tmp_path,289);r[key]=value
    with pytest.raises(common.Red):clean.planning(c)


def test_clean_nonfit_before_lease_and_no_model(monkeypatch,tmp_path):
    import apa_sp3_control as control
    c,r,source=planning_fixture(monkeypatch,tmp_path,290)
    dest=tmp_path/'nonfit.json'
    monkeypatch.setattr(control,'validate',lambda id:c)
    monkeypatch.setattr(control,'verify_sources',lambda:None)
    monkeypatch.setattr(control,'protocol',lambda:None)
    monkeypatch.setattr(control,'require_pass',lambda id:None)
    monkeypatch.setattr(control,'ART',tmp_path)
    monkeypatch.setattr(control,'job_path',lambda id:dest)
    monkeypatch.setattr(clean,'job_path',lambda id:dest if id==c['id'] else source)
    monkeypatch.setattr(common,'cell_fingerprint',lambda c:{})
    monkeypatch.setattr(provenance,'bridge',lambda:dict(effective_sha256='fixture'))
    monkeypatch.setattr(clean,'CleanModel',lambda *a:pytest.fail('nonfit must not load model'))
    (tmp_path/'protocol_amendment.json').write_text('{}')
    assert control.preflight(c['id'])=='NON_FIT'
    j=common.read(dest)
    assert j['result']['steps']==0 and j['result']['ms_token'] is None
    with pytest.raises(common.Red,match='existing immutable'):control.preflight(c['id'])


def test_clean_missing_source_blocks_and_dense_is_registered(monkeypatch):
    def missing(n):raise common.Red('BLOCKED_DEPENDENCY')
    monkeypatch.setattr(clean,'require_pass',missing)
    with pytest.raises(common.Red,match='BLOCKED_DEPENDENCY'):
        clean.planning(cell('decode_clean_b4_C_32768'))
    for name in ('decode_clean_b4_A_8192','decode_clean_b4_A_32768'):
        assert clean.planning(cell(name))['outcome']=='NON_FIT_REGISTERED_DENSE'


def measured(name,speed):
    c=cell(name)
    return dict(cell=c,status='PASS',result=dict(fit=True,steps=32,config=c['config'],feeding=c['feeding'],
                bits=c['bits'],arm=c['arm'],S=c['S'],tokens_s=speed,ms_token=1000/speed))


def test_clean_p5_no_legacy_or_8192_substitution():
    jobs={n:measured(n,v) for n,v in [('decode_clean_b4_B_8192',2),('decode_clean_b4_C_8192',5)]}
    r=report.prediction(jobs)
    assert r['status']=='UNASSESSABLE' and r['ratio'] is None and r['ratio_8192']==2.5
    for kind in ('decode','decode_pool'):
        for arm in 'BC':jobs[f'{kind}_b4_{arm}_32768']=measured(f'decode_clean_b4_{arm}_32768',100)
    assert report.prediction(jobs)['status']=='UNASSESSABLE'
    for arm,speed in [('B',3),('C',6)]:
        n=f'decode_clean_b4_{arm}_32768';jobs[n]=measured(n,speed)
    assert report.prediction(jobs)['status']=='HIT'
    jobs['decode_clean_b4_C_32768']['result']['tokens_s']=5.9
    assert report.prediction(jobs)['status']=='MISSED'
    jobs['decode_clean_b4_C_32768']['result']['config']['attention_wrapper']=True
    assert report.prediction(jobs)['status']=='UNASSESSABLE'


@pytest.mark.parametrize('ms,verdict',[(43.2,'WITHIN_2X_HARNESS_FINDING'),(43.21,'OUTSIDE_2X_ENGINE_ON_CARD_FINDING')])
def test_june_threshold_unchanged(ms,verdict):
    name='decode_repro_b4_A_2048';j=measured(name,1000/ms)
    assert report.reproduction({name:j})['comparison']==verdict
    assert report.reproduction({})['ms_token'] is None


def test_a6_bridge_all_old_kinds_future_edits_and_receipt_bytes():
    before=common.read(common.ART/'a6_before.json');m=provenance.bridge()
    for kind,paths in before['closures'].items():
        old={p:before['files'].get(p,common.sha(ROOT/p)) for p in paths}
        j=dict(cell=dict(kind=kind),registration_sha256=common.REG_SHA,fingerprint=old,
               fingerprint_schema='apa_sp3_per_kind_v1',fingerprint_amendment_sha256=before['effective_a5_sha256'])
        assert provenance.compatible(j,amendment=m),kind
        now=provenance.current_fingerprint(j['cell'])
        now['scripts/apa_sp3_gpu.py']='unreviewed'
        assert not provenance.compatible(j,current=now,amendment=m)
    assert all(common.sha(ROOT/p)==h for p,h in before['receipts'].items())
    common.verify_sources()
    assert common.sha(ROOT/'scripts/apa_sp3_model.py')==before['files']['scripts/apa_sp3_model.py']
    assert common.sha(ROOT/'scripts/apa_sp3_a5_decode.py')==before['files']['scripts/apa_sp3_a5_decode.py']
