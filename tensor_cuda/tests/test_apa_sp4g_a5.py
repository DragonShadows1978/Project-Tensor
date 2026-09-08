"""A5 author CPU gates. Prior art: SP4G A3/A4 (2026) seam/receipt
negative tests; DeMillo/Lipton/Sayward (1978) source-copy mutation tests
(unverified — lead to check: Hints on Test Data Selection). No GPU claim.
"""
import copy, importlib.util, json, os, sys, time
from pathlib import Path
from types import SimpleNamespace as NS
from contextlib import nullcontext
import numpy as np
import pytest
R = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'scripts'))
import apa_sp4g_a5_common as common
import apa_sp4g_a5_model as model
import apa_sp4g_a5_gpu as gpu
import apa_sp4g_a5_registry as registry
from apa_sp4g_model import Model, scoring_blocks
from apa_sp4g_a3_math import dense
from test_apa_sp4g_a3 import T

if os.environ.get('APA_SP4G_A5_MUTANT'):
    name,path = os.environ['APA_SP4G_A5_MUTANT'].split(':',1)
    spec = importlib.util.spec_from_file_location('a5_mutant',path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    globals()[name] = module


def test_a5_capture_shapes_include_actual_L64_cached_blocks():
    cs = registry.by_id()
    pins = common.read(common.path_a4(registry.GATE))['result']['dtype_pins']
    expected = [(5,0,1023,1087),(5,15,1983,2047),(47,15,1983,2047)]
    for name,(layer,block,lo,S) in zip(registry.CALLS,expected):
        c = cs[name]
        p = [p for p in pins if p['layer']==layer][2+block]
        assert (c['layer'],c['L'],c['S_all'],c['position_offset']) == (layer,64,S,lo)
        assert (p['n'],p['S_all'],p['lo']) == (64,S,lo)
        cache = (NS(shape=(1,1,lo,512)),NS(shape=(1,1,lo,512)))
        meta = model.geometry(c,2+block,64,S,lo,cache)
        assert meta['kq_count'] is None and meta['quantized_rows']==S
        with pytest.raises(common.Red,match='GEOMETRY'):
            model.geometry(c,2+block,64,S+1,lo,cache)
        with pytest.raises(common.Red,match='GEOMETRY'):
            model.geometry(c,2+block,64,S,lo,None)


def test_a5_dispatch_L64_is_prefill_even_with_cache():
    for S in (1087,2047,1088,2048,8192):
        for dtype in ('bfloat16','float32'):
            for diag in (False,True):
                r = model.launch_metadata(64,S,dtype,diag)
                assert r['path']=='prefill' and r['symbol']=='apa_selective_sp_kernel'
                assert (r['grid_blocks'],r['threads'],r['CAP'])==(1024,32,512)
    assert model.launch_metadata(1,4097,'float32',False)['path']=='split-K/decode'


def test_a5_dense_L64_cached_bottom_right_analytic_and_last_key():
    # Analytic mean of value indices under uniform attention; no copied kernel.
    for S in (1087,2047):
        q=np.zeros((1,2,64,2),np.float32);k=np.zeros((1,1,S,2),np.float32)
        v=np.broadcast_to(np.arange(S,dtype=np.float32)[None,None,:,None],(1,1,S,2)).copy()
        out=dense(q,k,v,position_offset=S-64)
        expected=(S-64+np.arange(64))/2
        np.testing.assert_allclose(out[0,0,:,0],expected,rtol=2e-6)
        np.testing.assert_array_equal(out[0,0],out[0,1])
        assert abs(out[0,0,-1,0]-(S-1)/2)<.01


def completed_fixture():
    j=common.read(common.path_a4(registry.GATE))
    ref=common.read(common.path_a4('diag_a2_fp32_A_2048_w0'))['result']
    return j,ref


def validate(j,ref):
    original,_=completed_fixture()
    return common.validate_completed_d32(j,original['cell'],original['fingerprint'],original['dependencies'],ref)


def test_a5_completed_RED_accepted_without_waiving_gate():
    j,ref=completed_fixture()
    assert validate(j,ref)['status']=='RED'
    assert not common.a4.exactness(j['result'])
    assert common.completed_d32()['status']=='RED'
    with pytest.raises(common.Red,match='STALE_OR_RED'):
        common.require_a4(registry.GATE)


def test_a5_worker_RED_is_not_completion():
    j,ref=completed_fixture()
    j['result']={};j['error']='RuntimeError: worker failed'
    with pytest.raises(common.Red,match='NOT_COMPLETED'):
        validate(j,ref)


@pytest.mark.parametrize('field',['fingerprint','dependencies','a4_registration_sha256','cell'])
def test_a5_completed_stale_receipt_rejected(field):
    j,ref=completed_fixture();j[field]={}
    with pytest.raises(common.Red,match='STALE'):
        validate(j,ref)


@pytest.mark.parametrize('field,value',[('native_call_count',143),('same_schedule',False),
    ('targets',1023),('ppl',float('nan')),('dtype_pin_complete',False),('arm','A32')])
def test_a5_partial_completion_rejected(field,value):
    j,ref=completed_fixture();j['result'][field]=value
    with pytest.raises(common.Red,match='NOT_COMPLETED'):
        validate(j,ref)


def test_a5_mislabeled_RED_or_bad_dtype_rejected():
    j,ref=completed_fixture();j['error']='worker failure'
    with pytest.raises(common.Red,match='INCONSISTENT'):
        validate(j,ref)
    j,ref=completed_fixture();j['result']['dtype_pins'][0]['native_output']['out']='bfloat16'
    with pytest.raises(common.Red,match='DTYPE_PIN'):
        validate(j,ref)


def test_a5_propagation_ungated_and_registration_preserved():
    before=common.preserved()
    assert len(before['receipt_sha256'])>=76
    cs=registry.by_id()
    for arm in 'AD':
        c=cs[f'diag_a4_propagation_{arm}_2048_w0']
        assert c['depends']==['diag_a2_fp32_A_2048_w0',registry.COMPLETED]
        for n in c['depends']:
            assert common.require_a5(n)['status'] in ('PASS','RED')
        assert not set(registry.CALLS)&set(c['depends'])
    assert cs['diag_a4_propagation_2048_w0']['depends']==[
        'diag_a4_propagation_A_2048_w0','diag_a4_propagation_D_2048_w0']
    assert common.path_a5(registry.GATE)!=common.path_a4(registry.GATE)
    assert len(cs)==7 and all(c['worker_s']==285 for c in cs.values())
    assert common.sha(registry.REGISTRATION)==registry.REGISTRATION_SHA


def test_a5_rerun_requires_all_three_and_strict_disagreement(monkeypatch):
    seen=[];relative=.001
    def require(n):
        seen.append(n)
        return dict(result=dict(comparisons=dict(SP_vs_A_fp32=dict(relative_frobenius=relative))))
    monkeypatch.setattr(common,'require_a5',require)
    assert not common.rerun_allowed()
    assert seen==registry.CALLS
    relative=.001000001;seen.clear()
    assert common.rerun_allowed() and seen==registry.CALLS
    def missing(n):
        if n==registry.CALLS[-1]:raise FileNotFoundError(n)
        return require(n)
    monkeypatch.setattr(common,'require_a5',missing)
    with pytest.raises(FileNotFoundError):common.rerun_allowed()


def test_a5_classification_does_not_blame_SP_when_standard_also_bad():
    def cs(sa,sd,ad):
        return {name:dict(relative_frobenius=x) for name,x in zip(
            ('SP_vs_A_fp32','sp_fp32_vs_dense_fp32','standard_fp32_vs_dense_fp32'),(sa,sd,ad))}
    meta={f'mask_{d}':dict(diagnostic_output_bitwise=True,all_refined=True) for d in ('bf16','fp32')}
    assert model.classify(cs(.002,.002,1e-6),meta)['outcome']=='SP_PATH_DEFECT_WITH_PINNED_ARGUMENTS_LEAD_DECISION'
    assert model.classify(cs(1e-6,1e-6,1e-6),meta)['outcome'].startswith('AGREES')
    assert model.classify(cs(.002,.002,.002),meta)['outcome'].startswith('UNRESOLVED')
    meta['mask_fp32']['all_refined']=False
    assert model.classify(cs(1e-6,1e-6,1e-6),meta)['outcome'].startswith('SP_DIAGNOSTIC')


@pytest.mark.parametrize('name',registry.CALLS)
def test_a5_runs_scoring_driver_reaches_target_layer_and_block(name):
    # Actual Model.perplexity method with a lightweight CPU schedule model;
    # reaches index17 only by feeding all16 cached blocks after true prefill.
    c=registry.by_id()[name];seen=[]
    owner=model.ScoredCallModel.__new__(model.ScoredCallModel)
    owner.cell=c;owner.deadline=time.monotonic()+30;owner.got=None
    owner.tc=NS(no_grad=nullcontext,synchronize=lambda:None,empty_cache=lambda:None)
    owner.install=lambda:None
    owner.load_s=0;owner.counts_result=lambda:{}
    owner.perplexity=lambda ids,n:Model.perplexity(owner,ids,n)
    def forward(ids,last_token_only=False,caches=None,position_offset=0):
        if caches is None:
            seen.extend([(0,512),(512,511)])
        else:
            seen.append((position_offset,ids.shape[1]))
            if len(seen)-1==c['call_index']:
                owner.got=dict(observed=list(seen))
                raise model.CapturedCall()
        return T(np.zeros((1,ids.shape[1],2),np.float32)),object()
    owner.model=forward
    r=owner.run_call(np.zeros(2048,np.int64))
    assert len(r['observed'])==c['call_index']+1
    assert r['observed'][-1]==(c['position_offset'],64)
    assert r['observed'][2:]==list(scoring_blocks(2048,1024))[:c['block']+1]


@pytest.mark.parametrize('name',registry.CALLS)
def test_a5_install_selects_registered_layer_and_restores(name,monkeypatch):
    c=registry.by_id()[name]
    class Mixer:pass
    mixers=[Mixer() for _ in range(48)]
    for m in mixers:m.attention_mode='standard'
    owner=model.ScoredCallModel.__new__(model.ScoredCallModel)
    owner.model=NS(layers=[NS(mixer=m) for m in mixers])
    owner.cell=c;owner.deadline=time.monotonic()+30;owner.index=0;owner.observed_calls=[]
    owner.gemma=NS(Gemma4AttentionTC=Mixer);owner.tc=NS()
    seen=[]
    def original(m,*args):seen.append(m);return 'output','cache'
    owner.original_attn=original
    owner.install()
    # All nontarget layers must retain exact original call behavior.
    for layer,m in enumerate(mixers):
        if layer!=c['layer']:
            assert Mixer.__call__(m,NS(shape=(1,64,3840)),None,None,0,None)==('output','cache')
    assert owner.index==0 and len(seen)==47
    target=mixers[c['layer']]
    owner.index=c['call_index']
    # Geometry rejection proves the selected layer intercepts this target.
    with pytest.raises(common.Red,match='GEOMETRY'):
        Mixer.__call__(target,NS(shape=(1,64,3840)),None,None,c['position_offset'],None)
    assert owner.index==c['call_index']+1


def test_a5_native_pin_and_mask_disagreement_are_receipted(monkeypatch):
    owner=model.ScoredCallModel.__new__(model.ScoredCallModel)
    owner.deadline=time.monotonic()+30;owner.save=lambda *args:None
    q=T(np.zeros((1,16,64,512)),'float32');k=T(np.zeros((1,1,1087,512)),'float32')
    calls=[]
    def sp(*args):
        calls.append(args)
        out=T(np.ones(q.shape),'float32')
        if args[-1]:return out,T(np.zeros((1,16,64,1087)))
        return out
    owner.tc=NS(_C=NS(apa_selective_attention_sp=sp))
    out,r=owner.sp_observed(q,k,k,k,1.,True,'fp32')
    assert len(calls)==2 and calls[0][4:]==(1.,model.MAX_DELTA,True,None,False)
    assert r['native_inputs']=={n:'float32' for n in ('q','k','kq','v')}
    assert not r['all_refined'] and r['missing_eligible']>0 and r['diagnostic_output_bitwise']
    with pytest.raises(common.Red,match='DTYPE_PIN'):
        owner.sp_observed(q.astype('bfloat16'),k,k,k,1.,True,'fp32')


def test_a5_cooperative_deadline_restores_method_without_signals():
    class Mixer:
        def __call__(self):return 7
    owner=NS(gemma=NS(Gemma4AttentionTC=Mixer));original=Mixer.__call__
    with gpu.deadline_guard(owner,time.monotonic()+30):assert Mixer()()==7
    assert Mixer.__call__ is original
    with pytest.raises(common.Red,match='COOPERATIVE'):
        with gpu.deadline_guard(owner,time.monotonic()-1):pass
    assert Mixer.__call__ is original


def test_a5_full_cached_fork_capture_saves_metadata_and_merge(monkeypatch,tmp_path):
    # SP4G A3 fixture, now exercising the new scored-call implementation at
    # the actual final-layer shape, including inherited on-disk finish().
    import apa_sp4g_common as base
    import apa_sp4g_a3_model as inherited
    from test_apa_sp4g_a3 import fake_tc
    monkeypatch.setattr(base,'R',tmp_path)
    monkeypatch.setattr(inherited,'R',tmp_path)
    c=registry.by_id()['diag_a5_call_l47_b15'];tc=fake_tc()
    class Mixer:pass
    mixers=[Mixer() for _ in range(48)]
    for m in mixers:
        m.attention_mode='standard';m.o_proj=lambda t:t
    owner=model.ScoredCallModel.__new__(model.ScoredCallModel)
    owner.tc=tc;owner.cell=c;owner.deadline=time.monotonic()+60
    owner.model=NS(layers=[NS(mixer=m) for m in mixers],PREFILL_CHUNK=512)
    owner.gemma=NS(Gemma4AttentionTC=Mixer,_cast=lambda t:t.astype('bfloat16'))
    owner.directory=tmp_path/'capture';owner.files=[];owner.observed_calls=[]
    owner.index=c['call_index'];owner.got=None;owner.load_s=0
    q=T(np.zeros((1,16,64,512)),'bfloat16')
    k=T(np.zeros((1,1,2047,512)),'bfloat16')
    v=T(np.ones(k.shape),'bfloat16')
    seen=[]
    def original(m,x,cos,sin,position_offset=0,kv_cache=None):
        seen.append((m,x,cos,sin,position_offset,kv_cache))
        out=model.standard(tc,q,k,v) if m.attention_mode=='standard' else owner.dispatch(q,k,k,v,1.,1.,True)
        return m.o_proj(owner.gemma._cast(out.transpose(1,2).reshape([1,64,8192]))),(k,v)
    owner.original_attn=original;owner.original=lambda *args:None
    owner.install();x=T(np.zeros((1,64,8)))
    cache=(T(np.zeros((1,1,1983,512))),T(np.zeros((1,1,1983,512))))
    cos,sin=object(),object()
    with pytest.raises(model.CapturedCall):
        Mixer.__call__(mixers[47],x,cos,sin,1983,cache)
    assert len(seen)==3 and all(s[0] is mixers[47] and s[1] is x and s[5] is cache for s in seen)
    manifest=json.loads((tmp_path/owner.got['manifest']).read_text());meta=manifest['metadata']
    assert (meta['layer'],meta['block'],meta['L'],meta['S_all'])==(47,15,64,2047)
    assert meta['kq_count'] is None and meta['position_offset']==1983
    assert meta['causal_last_key_per_query']==list(range(1983,2047))
    assert all(meta['parity'][n]['bitwise'] for n in ('q','k','v'))
    for precision in ('bf16','fp32'):
        assert meta['mask_'+precision]['all_refined']
        assert meta['mask_'+precision]['dispatch']['path']=='prefill'
    assert {'q','k','kq','v','native_scores','native_p','dense_fp32','sp_fp32_projected'} <= set(manifest['arrays'])
    assert owner.got['classification']['outcome'].startswith('AGREES')
    owner.close();assert Mixer.__call__ is original
