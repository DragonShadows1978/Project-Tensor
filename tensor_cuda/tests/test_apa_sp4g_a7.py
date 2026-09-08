"""Prior art: SP4G (2026) negative harness gates; DeMillo/Lipton/Sayward
(1978) mutation testing, unverified lead search: Hints on Test Data Selection.
Author CPU evidence only, no GPU numerics or blind review claim.
"""
import copy, ctypes, importlib.util, json, os, sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'scripts'))
import apa_sp4g_a7_common as common
import apa_sp4g_a7_model as model
import apa_sp4g_a7_gpu as gpu
import apa_sp4g_model as base

if os.environ.get('APA_SP4G_A7_MUTANT'):
    name,path=os.environ['APA_SP4G_A7_MUTANT'].split(':',1)
    spec=importlib.util.spec_from_file_location('a7_mutant',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    globals()[name]=module


def cell(arm='A',S=16384):
    return common.by_id()[f'ceiling_long_{arm}_{S}']


def test_a7_registration_grid_rails_anchors_and_preservation():
    before=common.preserved(); assert len(before['receipt_sha256'])==127
    assert len(before['source_sha256'])==63 and len(before['amendment_sha256'])==20
    reg=common.registered(); cs=common.cells(); assert len(cs)==21
    assert [(c['arm'],c['S']) for c in cs]==[(arm,S) for arm in 'ABC' for S in common.SIZES]
    seen=[]
    for c in cs:
        assert c['kind']=='ceiling_long' and c['worker_s']==1500 and c['outer_s']==1560
        assert c['depends']==seen[-1:]; seen.append(c['id'])
        anchor=reg['anchors'][c['arm']];j=common.read(R/anchor['path'])
        assert common.sha(R/anchor['path'])==anchor['sha256']
        assert anchor['prefill_s']==j['result']['prefill_s']
        assert c['estimate_worker_s']==anchor['load_s']+anchor['prefill_s']*(c['S']/8192)**2
    assert len(common.tokens())>=131072
    assert reg['protocol']['C_delta']==3 and len(reg['predictions']['lead'])==len(reg['predictions']['seat'])==3
    for change in ({'kind':'ceiling'},{'worker_s':285},{'outer_s':1600},{'S':8192}):
        with pytest.raises(common.Red,match='UNREGISTERED'):
            common.valid_cell(dict(cs[0],**change))


def test_a7_deadline_inclusive_worker_and_outer():
    now=[1599.999];d=model.Deadline(100,100,lambda:now[0]);d.check()
    now[0]=1600
    with pytest.raises(model.Rail,match='WORKER_RAIL'):d.check()
    d=model.Deadline(150,100,lambda:now[0]);now[0]=1630
    with pytest.raises(model.Rail,match='OUTER_RAIL'):d.check()


@pytest.mark.parametrize('error',[
    RuntimeError('cudaMalloc failed: out of memory'),RuntimeError('cudaMallocAsync failed: out of memory'),
    RuntimeError('CUDA error at sync: out of memory'),RuntimeError('cudaErrorMemoryAllocation')])
def test_a7_cuda_oom_is_specific(error):
    assert model.classify(error)==('OOM','CUDA_OOM')


@pytest.mark.parametrize('error',[
    MemoryError('out of memory'),RuntimeError('CPU out of memory'),RuntimeError('worker exited 137'),
    RuntimeError('worker exited 124'),RuntimeError('OOM forecast'),RuntimeError('CUDA error at sync: invalid argument')])
def test_a7_other_errors_not_oom_or_rail(error):
    assert model.classify(error)[0]=='ERROR'
    assert model.classify(model.Rail('WORKER_RAIL'))==('RAIL','WORKER_RAIL')


def test_a7_cache_formula_and_actual_tuple_shape():
    for S in common.SIZES:
        k=common.kv_size(S)
        assert k['global_bytes']==S*8*2*1*512*2
        assert k['sliding_fixed_bytes']==335216640
        assert k['total_bytes']==S*16384+335216640
    S=32768
    cache=[tuple(NS(shape=(1,1,S,512) if i%6==5 else (1,8,1023,256),dtype='bfloat16') for _ in range(2)) for i in range(48)]
    assert model.cache_check(cache,S)==common.kv_size(S)
    cache[5][0].shape=(1,1,S-1,512)
    with pytest.raises(common.Red,match='SHAPE'):model.cache_check(cache,S)
    cache[5][0].shape=(1,1,S,512);cache[0][0].dtype='float32'
    with pytest.raises(common.Red,match='DTYPE'):model.cache_check(cache,S)


def prior(outcome,arm='A',S=16384):
    return dict(cell=cell(arm,S),result=dict(outcome=outcome))


def test_a7_only_same_arm_oom_stops_later_rungs():
    c=cell(S=24576)
    assert common.oom_source(c,prior('OOM'))=='ceiling_long_A_16384'
    for outcome in ('FIT','RAIL','ERROR'):
        assert common.oom_source(c,prior(outcome)) is None
    assert common.oom_source(cell('B'),prior('OOM')) is None
    p=prior('NON_FIT_AFTER_OOM');p['result']['oom_source']='first'
    assert common.oom_source(c,p)=='first'
    with pytest.raises(common.Red,match='REQUIRES_PREVIOUS_OOM'):
        gpu.inherited_nonfit(c,prior('RAIL'))


def receipt_fixture(outcome='FIT'):
    c=cell();r=dict(outcome=outcome,fit={'FIT':True,'RAIL':None,'OOM':False}[outcome],executed=True,
        completed_tokens=c['S'] if outcome=='FIT' else 512,cache_shape_verified=outcome=='FIT',
        kv_cache_at_S=common.kv_size(c['S']),extrapolated_worker_s=c['estimate_worker_s'],
        peak_resident_mib=None,sampled_peak_resident_mib=7400.,peak_status='RED_PEAK_UNAVAILABLE',
        failure_class={'FIT':None,'RAIL':'WORKER_RAIL','OOM':'CUDA_OOM'}[outcome])
    return c,dict(cell=c,status=common.outcome_status(outcome),result=r,worker_wall_s=1500.,
        a7_registration_sha256=common.REGISTRATION_SHA,registration_sha256=common.REG_SHA,
        fingerprint={'code':'good'},dependencies={})


def test_a7_receipt_rejects_stale_or_forged_evidence():
    c,j=receipt_fixture();common.validate_receipt(j,c,{'code':'good'},{})
    for update in ({'fingerprint':{'code':'bad'}},{'dependencies':{'ghost':'hash'}},
                   {'a7_registration_sha256':'bad'},{'cell':dict(c,S=8192)}):
        bad=dict(j,**update)
        with pytest.raises(common.Red,match='STALE'):common.validate_receipt(bad,c,{'code':'good'},{})
    for update in ({'completed_tokens':c['S']-1},{'fit':None},{'peak_status':'PASS'},
                   {'peak_resident_mib':float('nan')},{'executed':False}):
        bad=copy.deepcopy(j);bad['result'].update(update)
        with pytest.raises(common.Red):common.validate_receipt(bad,c,{'code':'good'},{})


def test_a7_rail_completes_without_becoming_fit_or_retry(monkeypatch,tmp_path):
    c,j=receipt_fixture('RAIL')
    common.validate_receipt(j,c,{'code':'good'},{})
    bad=copy.deepcopy(j);bad['result']['fit']=False
    with pytest.raises(common.Red,match='FIT_SEMANTICS'):common.validate_receipt(bad,c,{'code':'good'},{})
    p=tmp_path/'cell.json';common.publish(p,j);first=p.read_bytes()
    with pytest.raises(FileExistsError):common.publish(p,dict(j,status='PASS'))
    assert p.read_bytes()==first
    monkeypatch.setattr(gpu,'preflight',lambda c:'DONE')
    with pytest.raises(common.Red,match='CREATE_ONLY'):gpu.worker(c)


def fake_preflight(monkeypatch,tmp_path,previous=None):
    monkeypatch.setattr(gpu,'preserved',lambda:None)
    monkeypatch.setattr(gpu,'tokens',lambda:np.zeros(131072,dtype=np.int64))
    monkeypatch.setattr(gpu,'read',lambda p:dict(status='PASS_CPU_ONLY',fingerprint_amendment_sha256='seal',fingerprint={},execution_sha256={}))
    monkeypatch.setattr(gpu,'sha',lambda p:'seal')
    monkeypatch.setattr(gpu,'verify_fingerprint',lambda p:None)
    monkeypatch.setattr(gpu,'path_a7',lambda n:tmp_path/(n+'.json'))
    monkeypatch.setattr(gpu,'previous_receipt',lambda c:previous)


def test_a7_nonfits_use_no_model_or_lease(monkeypatch,tmp_path):
    previous=prior('OOM');fake_preflight(monkeypatch,tmp_path,previous)
    c=cell(S=24576);assert gpu.preflight(c)=='CPU_NON_FIT'
    monkeypatch.setattr(gpu,'measure',lambda *a:pytest.fail('model must not execute'))
    monkeypatch.setattr(gpu,'fingerprint',lambda:{})
    monkeypatch.delenv('APA_SP4G_A7_LEASE',raising=False)
    gpu.worker(c)
    j=json.loads((tmp_path/(c['id']+'.json')).read_text())
    assert j['result']['executed'] is False and j['result']['oom_source']==previous['cell']['id']
    common.validate_receipt(j,c,{}, {previous['cell']['id']:'seal'}, previous)


def test_a7_preflight_advances_rail_and_requires_lease(monkeypatch,tmp_path):
    fake_preflight(monkeypatch,tmp_path,prior('RAIL'))
    c=cell(S=24576);assert gpu.preflight(c)=='GPU'
    monkeypatch.delenv('APA_SP4G_A7_LEASE',raising=False)
    with pytest.raises(common.Red,match='LEASE_REQUIRED'):gpu.worker(c)
    monkeypatch.setattr(gpu,'tokens',lambda:np.zeros(8192,dtype=np.int64))
    with pytest.raises(common.Red,match='STREAM_TOO_SHORT'):gpu.preflight(c)


def test_a7_nvml_peak_unavailable_never_promotes_samples():
    m=model.ResidentPeak.__new__(model.ResidentPeak)
    m.device=ctypes.c_void_p(1);m.maximum=7500.;m.samples=3;m.last=7200.
    m.nvml=NS(nvmlDeviceGetAccountingStats=lambda *args:3)
    r=m.result();assert r['peak_resident_mib'] is None and r['sampled_peak_resident_mib']==7500.
    assert r['peak_status']=='RED_PEAK_UNAVAILABLE'
    def stats(dev,pid,ptr):ptr._obj.maxMemoryUsage=8000*(1<<20);return 0
    m.nvml.nvmlDeviceGetAccountingStats=stats
    r=m.result();assert r['peak_resident_mib']==8000 and r['peak_status']=='MEASURED_NVML_ACCOUNTING'
    assert ctypes.sizeof(model.ProcessInfo)==24 and ctypes.sizeof(model.AccountingStats)==56
    def missing(dev,pid,ptr):ptr._obj.maxMemoryUsage=(1<<64)-1;return 0
    m.nvml.nvmlDeviceGetAccountingStats=missing
    assert m.result()['peak_resident_mib'] is None


def test_a7_nvml_rejects_competing_pid():
    m=model.ResidentPeak.__new__(model.ResidentPeak)
    m.processes=lambda:[NS(pid=os.getpid(),usedGpuMemory=100),NS(pid=os.getpid()+1,usedGpuMemory=100)]
    with pytest.raises(common.Red,match='CONCURRENT'):m.sample()


def fake_measure(monkeypatch,error=None):
    events=[]
    class Monitor:
        sample=lambda self:7000.
        close=lambda self:None
        result=lambda self:dict(peak_resident_mib=None,sampled_peak_resident_mib=7500.,peak_status='RED_PEAK_UNAVAILABLE')
    class Owner:
        def __init__(self,c,delta,capture,observe):
            events.append((c,delta,capture,observe));self.delta=delta
            self.load_s=76.;self.resident_load_mib=6584.;self.pool=NS(result=lambda:dict(pool_reserved_high_mib=7200.))
        def close(self):events.append('closed')
    def run(owner,ids,c,deadline,monitor,progress):
        progress.update(completed_tokens=512,completed_chunks=1)
        if error:raise error
        progress['completed_tokens']=c['S']
        return dict(prefill_s=400.,cache_shape_verified=True,**progress)
    monkeypatch.setattr(base,'Model',Owner);monkeypatch.setattr(model,'ResidentPeak',Monitor)
    monkeypatch.setattr(model,'run_prefill',run)
    return events


def test_a7_frozen_delta_and_production_measurement(monkeypatch):
    events=fake_measure(monkeypatch);saved=base.resident
    for arm in 'ABC':
        c=cell(arm);r=model.measure(c,np.zeros(c['S']),NS(check=lambda:None))
        assert events[-2][1:]==(3. if arm=='C' else None,False,False)
        assert r['outcome']=='FIT' and r['completed_tokens']==c['S']
    assert base.resident is saved


@pytest.mark.parametrize('error,outcome',[(model.Rail('WORKER_RAIL'),'RAIL'),
    (RuntimeError('cudaMallocAsync failed: out of memory'),'OOM'),(MemoryError('host'),'ERROR')])
def test_a7_partial_failure_preserves_wall_cache_and_peak(monkeypatch,error,outcome):
    events=fake_measure(monkeypatch,error);c=cell()
    r=model.measure(c,np.zeros(c['S']),NS(check=lambda:None))
    assert r['outcome']==outcome and r['completed_tokens']==512 and events[-1]=='closed'
    assert r['kv_cache_at_S']['global_bytes']==c['S']*16384
    assert r['sampled_peak_resident_mib']==7500. and r['prefill_attempt_wall_s']>=0
    assert r['extrapolated_worker_s']==c['estimate_worker_s']


def test_a7_entire_prefix_uses_june_prefill(monkeypatch):
    c=cell();seen=[];ids=np.arange(c['S']+7)
    owner=NS(prefill=lambda x:(seen.append(x.copy()),[],123.))
    monkeypatch.setattr(model,'observe',lambda *args:nullcontext())
    monkeypatch.setattr(model,'cache_check',lambda cache,S:seen.append(S))
    r=model.run_prefill(owner,ids,c,NS(check=lambda:None),None,{'completed_tokens':c['S']})
    assert np.array_equal(seen[0],ids[:c['S']]) and len(seen[0])==c['S'] and seen[1]==c['S']
    assert r['cache_shape_verified'] and r['prefill_s']==123.


def test_a7_guard_restores_classes_after_rail():
    class Block:
        def __call__(self,*a,**kw):return 1
    class June:
        def _forward(self,*a,**kw):return 2
    gemma=NS(Gemma4BlockTC=Block,Gemma4_TC=June)
    owner=NS(gemma=gemma,model=NS(layers=[Block() for _ in range(48)]))
    b0=Block.__call__;f0=June._forward
    def stop():raise model.Rail('WORKER_RAIL')
    with pytest.raises(model.Rail):
        with model.observe(owner,NS(check=stop),NS(sample=lambda:7000.),{'completed_tokens':0,'completed_chunks':0}):
            owner.model.layers[0]()
    assert Block.__call__ is b0 and June._forward is f0


def test_a7_pool_on_before_load_and_june_flags(monkeypatch):
    events=[];original=lambda *a:None
    tc=NS(set_alloc_pooling=lambda v:events.append(('pool',v)),no_grad=nullcontext,
          synchronize=lambda:None,apa_selective_attention=original,_C=NS())
    layers=[NS(mixer=NS(is_global=i%6==5,head_dim=512,kv_heads=1)) for i in range(48)]
    def load(*a,**kw):events.append(('load',kw));return NS(layers=layers),{'loaded':'QAT q4_0 exact (symmetric-8 g32)'}
    gemma=NS(__file__='/mnt/ForgeRealm/GraftRepository/core/gemma4_tc.py',
             Gemma4AttentionTC=type('Attn',(),{'__call__':original,'KV_STORE_HOOK':None}),
             KVRing=NS(),Gemma4_TC=NS(from_pretrained=load))
    mistral=NS(QuantLinearTC=NS(),RMSNormTC=NS(),F=NS(),BlockTC=NS(COMPUTE_DTYPE='bfloat16'))
    monkeypatch.setitem(sys.modules,'core',NS(gemma4_tc=gemma,mistral7b_tc=mistral))
    monkeypatch.setattr(base,'load_runtime',lambda:tc);monkeypatch.setattr(base,'verify_weight',lambda:None)
    monkeypatch.setattr(base,'resident',lambda:6584.);monkeypatch.setattr(base,'PoolPeak',lambda:NS())
    with monkeypatch.context() as env:
        # Base Model deliberately rewrites its experiment flags.
        saved=dict(os.environ)
        try:
            owner=base.Model(cell('B'),capture=False,observe=False)
            assert events[0]==('pool',True) and events[1][0]=='load'
            assert all(x.mixer.apa_min_context==0 and x.mixer.bulk_bits==4 and x.mixer.refine_percentile==.15 for x in layers)
            assert gemma.KVRing.QUANT_V is False and gemma.KVRing.QUANT_KV4 is False
            assert tc.apa_selective_attention is original
            owner.close()
        finally:
            os.environ.clear();os.environ.update(saved)


def test_a7_foreground_shell_safety_and_rail_isolation():
    shell=(R/'scripts/apa_sp4g_a7_lead_gpu.sh').read_text()
    assert 'flock --exclusive --wait 20 9' in shell and 'sleep 30' in shell
    for forbidden in ('timeout ', 'kill ', 'pkill','killall','nohup','wait $',' &\n'):
        assert forbidden not in shell
    assert shell.count('gpu.py worker "$job"')==1
    from apa_sp4g_a6_registry import cells as old_cells
    assert all(c['worker_s']==285 for c in old_cells())


def test_a7_late_telemetry_fit_becomes_rail(monkeypatch,tmp_path):
    fake_preflight(monkeypatch,tmp_path)
    c=cell();_,j=receipt_fixture()
    monkeypatch.setattr(gpu,'fingerprint',lambda:{})
    monkeypatch.setenv('APA_SP4G_A7_LEASE','1')
    monkeypatch.setattr(gpu.os,'fstat',lambda fd:NS(st_dev=1,st_ino=2))
    original_stat=gpu.os.stat
    monkeypatch.setattr(gpu.os,'stat',lambda p,*a,**kw:NS(st_dev=1,st_ino=2) if str(p)=='/tmp/forge-gpu.lock' else original_stat(p,*a,**kw))
    monkeypatch.setattr(gpu.fcntl,'flock',lambda *a:None)
    def late():raise gpu.Rail('WORKER_RAIL')
    monkeypatch.setattr(gpu,'Deadline',lambda *a:NS(check=late))
    monkeypatch.setattr(gpu,'measure',lambda *a:j['result'])
    gpu.worker(c)
    result=json.loads((tmp_path/(c['id']+'.json')).read_text())
    assert result['status']=='RED' and result['result']['outcome']=='RAIL'
    assert result['result']['fit'] is None and result['result']['failure_class']=='WORKER_RAIL'


def test_a7_exact_peak_must_be_accounting_and_at_least_sampled():
    c,j=receipt_fixture()
    for peak,status in ((7399.,'MEASURED_NVML_ACCOUNTING'),(8000.,'SAMPLED')):
        bad=copy.deepcopy(j);bad['result'].update(peak_resident_mib=peak,peak_status=status)
        with pytest.raises(common.Red,match='ACCOUNTING_PEAK'):
            common.validate_receipt(bad,c,{'code':'good'},{})

    j['result'].update(peak_resident_mib=7400.,peak_status='MEASURED_NVML_ACCOUNTING')
    common.validate_receipt(j,c,{'code':'good'},{})
