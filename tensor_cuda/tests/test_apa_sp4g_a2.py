"""A2 author baseline. Prior art: SP3 (2026) negative provenance/replay tests,
DeMillo/Lipton/Sayward1978 mutation testing (unverified lead: Hints on Test Data
Selection). Constructed seam inputs, not a GPU or blind-review claim.
"""
import ast,importlib.util,json,os,sys
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import numpy as np
import pytest
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'scripts'))
import apa_sp4g_a2_model as model
import apa_sp4g_a2_metrics as metrics
import apa_sp4g_a2_common as common
import apa_sp4g_a2_registry as registry
import apa_sp4g_a2_gpu as gpu
import apa_sp4g_a2_report as report
if os.environ.get('APA_SP4G_A2_MUTANT'):
    name,path=os.environ['APA_SP4G_A2_MUTANT'].split(':',1)
    spec=importlib.util.spec_from_file_location('a2_mutant',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);globals()[name]=module

class T:
    def __init__(self,x):self.x=np.asarray(x);self.shape=self.x.shape
    def reshape(self,s):return T(self.x.reshape(s))
    def float(self):return self
    def numpy(self):return self.x
    def astype(self,d):return self

def test_a2_registration_precedes_edits_and_a1_receipts_closures_preserved():
    before=common.preserved();h=common.read(common.A/'amendment_006_a2_hypotheses.json');e=common.read(common.A/'amendment_007_a2_execution.json')
    assert h['before_sha256']==common.sha(common.A/'a2_before.json')
    assert e['hypotheses_sha256']==common.sha(common.A/'amendment_006_a2_hypotheses.json')
    for p in ('scripts/apa_sp4g_model.py','scripts/apa_sp4g_metrics.py','scripts/apa_sp4g_gpu.py','scripts/apa_sp4g_common.py','scripts/apa_sp4g_registry.py','scripts/apa_sp4g_lead_gpu.sh'):
        assert common.sha(R/p)==before['source_sha256'][p]
    assert common.require_pass('kernel512')['status']=='PASS'
    assert common.require_pass('ppl_A_2048')['result']['targets']==4096
    assert common.require_pass('ppl_D_8192')['status']=='PASS'

def test_a2_cell_dag_diagnostics_order_and_actual_ppl_margins():
    seen=set();cs=registry.cells()
    for c in cs:
        assert c['id'] not in seen and c['S']<=8192 and c['worker_s']==285
        if 'population_rows' in c:assert c['population_rows']==c['S']-1
        assert all(d in seen or d in ('kernel512','capture_B_8192','freeze') for d in c['depends']);seen.add(c['id'])
    assert [c['focus'] for c in cs if c['kind']=='parity']==['source','scale','value','rope']
    margins=[c for c in cs if c['kind']=='margin_a2'];assert len(margins)==32
    for c in margins:
        assert c['depends']==[f'ppl_capture_{c["arm"]}_{c["S"]}_w0']
        assert c['population_rows']==c['S']-1
    assert len(cs)==43

@pytest.mark.parametrize('S',[16384,24576,32768])
def test_a2_long_rows_rejected_before_model_or_preflight(S):
    assert registry.rail_blocked(dict(S=S))
    with pytest.raises(common.Red,match='LONG_LEASE'):gpu.execute(dict(S=S))
    with pytest.raises(common.Red,match='LONG_LEASE'):gpu.preflight(dict(S=S))

def test_a2_rail_bookkeeping_distinguishes_observed_and_deferred():
    rows=report.rail_rows();assert len(rows)==20
    assert sum(r['receipt'] is not None for r in rows)==5
    assert all(r['outcome']=='RAIL' and r['fit'] is False and r['capacity_fit'] is None and not r['retry'] for r in rows)
    assert {r['cell'] for r in rows if r['receipt']}=={'ppl_A_16384','ppl_B_16384','ppl_D_16384','ceiling_A_16384','ceiling_A_24576'}

def test_a2_lossless_bf16_payload_keeps_sign_bits_and_subnormals():
    bits=np.array([0,0x8000,0x3f80,0xbf80,1,0x7f7f,0xff7f],np.uint16)
    x=(bits.astype(np.uint32)<<16).view(np.float32)
    np.testing.assert_array_equal(model.encode_bf16(x),bits)
    np.testing.assert_array_equal(model.decode_bf16(bits).view(np.uint32),x.view(np.uint32))
    with pytest.raises(common.Red,match='BF16'):model.encode_bf16(np.array([1.0001],np.float32))
    with pytest.raises(common.Red,match='BF16'):model.encode_bf16(np.array([np.nan],np.float32))

def test_a2_standard_same_tensors_bottom_right_mqa_and_scale():
    rng=np.random.default_rng(209);q=rng.normal(size=(1,16,3,512)).astype(np.float32)*.03
    k=rng.normal(size=(1,1,7,512)).astype(np.float32)*.03;v=rng.normal(size=k.shape).astype(np.float32)
    calls=[]
    def matmul(a,b,alpha=1.,trans_b=False):
        calls.append((a,b,alpha,trans_b));return T(alpha*(a.x@(b.x.swapaxes(-1,-2) if trans_b else b.x)))
    def softmax(t):
        x=t.x.astype(np.float64);L,S=x.shape[-2:];eligible=np.arange(S)<(S-L+np.arange(L)+1)[:,None]
        x=np.where(eligible,x,-np.inf);p=np.exp(x-x.max(axis=-1,keepdims=True));return T(p/p.sum(axis=-1,keepdims=True))
    kt,vt=T(k),T(v);tc=SimpleNamespace(matmul=matmul,causal_softmax=softmax)
    got=model.standard(tc,T(q),kt,vt,.75).numpy()
    scores=.75*(q.astype(np.float64)@k.astype(np.float64).swapaxes(-1,-2));scores=np.where(np.arange(7)<(5+np.arange(3))[:,None],scores,-np.inf)
    p=np.exp(scores-scores.max(-1,keepdims=True));want=(p/p.sum(-1,keepdims=True))@v
    np.testing.assert_allclose(got,want,rtol=1e-5,atol=1e-7)
    assert calls[0][1] is kt and calls[1][1] is vt and calls[0][2]==.75

def test_a2_replay_requires_output_and_selection_bitwise():
    raw=dict(out=np.zeros((1,16,2,512),np.float32),mask=np.zeros((1,16,2,3),bool))
    metrics.require_exact_replay(raw['out'].copy(),raw['mask'].copy(),raw)
    bad=raw['out'].copy();bad.flat[0]=np.nextafter(np.float32(0),np.float32(1))
    with pytest.raises(common.Red,match='REPLAY_NOT_BITWISE'):metrics.require_exact_replay(bad,raw['mask'],raw)
    mask=raw['mask'].copy();mask.flat[0]=True
    with pytest.raises(common.Red,match='SELECTION_NOT_BITWISE'):metrics.require_exact_replay(raw['out'],mask,raw)

def test_a2_record_restores_original_call_S_L_mask_and_lossless_tensors(tmp_path):
    rec=dict(lo=5,n=3,S=8,mask_shape=[1,16,3,8],files={})
    for name,shape in [('q',(1,16,3,512)),('out',(1,16,3,512)),('k',(1,1,8,512)),('kq',(1,1,8,512)),('v',(1,1,8,512))]:
        p=tmp_path/(name+'.npy');np.save(p,np.full(shape,0x3f80,np.uint16));rec['files'][name]={'path':str(p)}
    mask=np.broadcast_to(np.arange(8)<(6+np.arange(3))[:,None],(1,16,3,8))
    p=tmp_path/'mask.npy';np.save(p,np.packbits(mask.ravel(),bitorder='little'));rec['files']['mask']={'path':str(p)}
    raw=metrics.load_record(rec);assert raw['q'].shape[2]==3 and raw['kq'].shape[2]==8
    np.testing.assert_array_equal(raw['mask'],mask);assert np.all(raw['k']==1)
    with pytest.raises(common.Red,match='GEOMETRY'):metrics.load_record(dict(rec,S=7))

def test_a2_capture_finishes_only_full_actual_ppl_population(monkeypatch,tmp_path):
    owner=model.PPLCapture.__new__(model.PPLCapture);owner.cell=dict(id='ppl',S=129,arm='B');owner.delta=None
    owner.directory=tmp_path;owner.files=[];owner.counts_result=lambda:{}
    owner.records={l:[dict(lo=0,n=63,S=63),dict(lo=63,n=65,S=128)] for l in range(5,48,6)}
    monkeypatch.setattr(model,'R',tmp_path);r=owner.finish_capture(dict(ppl=7.,targets=64))
    assert r['population_rows']==128
    owner.records[5][1]['lo']=64
    with pytest.raises(common.Red,match='COVERAGE'):owner.finish_capture({})

def test_a2_capture_uses_original_output_and_saves_its_mask(monkeypatch,tmp_path):
    owner=model.PPLCapture.__new__(model.PPLCapture);owner.cell=dict(arm='B');owner.calls=0
    owner.counts={5:dict(pairs=0,selected=0)};owner.records={5:[]};owner.files=[];owner.directory=tmp_path
    owner.tc=SimpleNamespace();q=T(np.zeros((1,16,2,512),np.float32));k=T(np.ones((1,1,3,512),np.float32))
    original=T(np.zeros(q.shape,np.float32));mask=np.broadcast_to(np.arange(3)<np.array([2,3])[:,None],(1,16,2,3)).copy()
    owner.original=lambda *a:original;owner.diag=SimpleNamespace(selective=lambda *a:(T(original.x.copy()),T(mask),None))
    def save(p,x):
        p.parent.mkdir(parents=True,exist_ok=True);np.save(p,x);return dict(path=str(p))
    monkeypatch.setattr(model,'save_array',save)
    assert owner.diagnostic_dispatch(q,k,k,k,1.,1.,True) is original
    rec=owner.records[5][0]
    assert set(rec['files'])=={'q','k','kq','v','out','mask'}
    np.testing.assert_array_equal(np.unpackbits(np.load(rec['files']['mask']['path']),bitorder='little',count=mask.size).reshape(mask.shape),mask)
    assert owner.counts[5]==dict(pairs=80,selected=80)

def test_a2_precision_dispatch_reuses_exact_k_and_returns_registered_arm():
    owner=model.ParityModel.__new__(model.ParityModel);owner.cell=dict(kind='precision',treatment='D32');owner.probes=[];owner.active_layer=5
    q=T(np.zeros((1,16,2,512),np.float32));k=T(np.zeros((1,1,3,512),np.float32));seen=[]
    def mm(a,b,alpha=1.,trans_b=False):return T(a.x@(b.x.swapaxes(-1,-2) if trans_b else b.x))
    def sp(*args):seen.append(args);return T(np.full(q.shape,3.,np.float32))
    owner.tc=SimpleNamespace(matmul=mm,causal_softmax=lambda x:T(np.ones(x.shape,np.float32)/3),_C=SimpleNamespace(apa_selective_attention_sp=sp))
    out=owner.dispatch(q,k,k,k,1.,1.,True)
    assert np.all(out.x==3) and seen[0][1] is k and len(owner.probes)==1
    assert seen[0][5]==model.MAX_DELTA

def test_a2_compare_nonfinite_rejected_and_relative_is_descriptive():
    with pytest.raises(common.Red,match='COMPARISON'):model.compare([np.nan],[0])
    r=model.compare([1.,2.],[0.,2.]);assert r['max_abs']==1. and not r['bitwise'] and r['different']==1

def test_a2_same_state_attention_replays_identical_arguments_and_restores(monkeypatch):
    class Mixer:pass
    mix=Mixer();mix.is_global=True;mix.attention_mode='standard'
    owner=model.ParityModel.__new__(model.ParityModel);owner.layer_ids={id(mix):5};owner.cell={'kind':'parity'};owner.probed=set();owner.probes=[]
    owner.tc=SimpleNamespace(matmul='original');owner.original_matmul='original';owner.gemma=SimpleNamespace(Gemma4AttentionTC=Mixer)
    args_seen=[]
    def original(m,x,cos,sin,offset,cache):
        args_seen.append((m,x,cos,sin,offset,cache,m.attention_mode))
        if m.attention_mode=='apa_selective':owner.probes.append({})
        return T(np.zeros((1,2,8),np.float32)),cache
    owner.original_attn=original;owner.install();x=T(np.zeros((1,2,8)));cos=object();sin=object();cache=(object(),object())
    out,got=Mixer.__call__(mix,x,cos,sin,11,cache)
    assert got is cache and len(args_seen)==2
    assert all(a is b for a,b in zip(args_seen[0][:4],args_seen[1][:4])) and args_seen[0][4:6]==args_seen[1][4:6]
    assert [a[-1] for a in args_seen]==['standard','apa_selective']
    assert mix.attention_mode=='standard' and owner.tc.matmul=='original' and owner.probed=={5}

def test_a2_new_fingerprint_unknown_transition_and_red_rejected(monkeypatch,tmp_path):
    c=dict(id='unit',kind='parity',depends=[]);p=tmp_path/'unit.json'
    monkeypatch.setattr(common,'by_id',lambda:{'unit':c});monkeypatch.setattr(common,'path_a2',lambda n:p)
    monkeypatch.setattr(common,'fingerprint_a2',lambda c:{'code':'live'})
    j=dict(status='PASS',cell=c,registration_sha256=common.REG_SHA,fingerprint={'code':'live'},dependencies={},result={})
    p.write_text(json.dumps(j));assert common.require_a2('unit')==j
    for change in (dict(status='RED'),dict(fingerprint={'code':'unknown'}),dict(dependencies={'ghost':'hash'})):
        p.write_text(json.dumps(dict(j,**change)))
        with pytest.raises(common.Red,match='RED|DEPENDENCY'):common.require_a2('unit')

def test_a2_source_pins_pre_branch_norm_rope_v_and_scale():
    source=Path('/mnt/ForgeRealm/GraftRepository/core/gemma4_tc.py').read_text()
    tree=ast.parse(source);klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Gemma4AttentionTC')
    fn=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='__call__')
    text=ast.get_source_segment(source,fn)
    assert text.index('q = _head_rmsnorm')<text.index('q = tc.rope_apply')<text.index('apa_active =')
    assert 'vsrc = kraw if self.is_global else self.v_proj(x)' in text
    assert 'v = _head_rmsnorm(vsrc, None' in text and 'rope_apply(v' not in text
    assert 'softcap' not in text
    assert model.ENV['GEMMA4_QUANT_V']==model.ENV['GEMMA4_QUANT_KV4']=='0'

def test_a2_runner_preserves_bounds_and_uses_isolated_driver():
    s=(R/'scripts/apa_sp4g_a2_lead_gpu.sh').read_text()
    assert '--kill-after=5s 285s' in s and '--kill-after=3s 585s' in s and 'sleep 30' in s
    assert 'flock --exclusive --wait 20 9' in s and 'scripts/apa_sp4g_a2_gpu.py worker' in s
    assert 'pkill' not in s and 'killall' not in s

def test_a2_whole_layer_replays_original_calls_before_tiles_and_checks_ppl_counts(monkeypatch,tmp_path):
    recs=[dict(lo=0,n=129,S=129,scale=1.,zthr=1.,causal=True),dict(lo=129,n=64,S=193,scale=1.,zthr=1.,causal=True)]
    raw_by_S={};count=0
    for idx,rec in enumerate(recs):
        L,S=rec['n'],rec['S'];eligible=np.arange(S)<(S-L+np.arange(L)+1)[:,None]
        selected=np.broadcast_to(eligible & (np.arange(S)%2==0),(1,16,L,S)).copy();count+=int(selected.sum())
        raw_by_S[S]=dict(q=np.zeros((1,16,L,512),np.float32),k=np.zeros((1,1,S,512),np.float32),
            kq=np.full((1,1,S,512),idx+1,np.float32),v=np.zeros((1,1,S,512),np.float32),out=np.zeros((1,16,L,512),np.float32),mask=selected)
    manifest=dict(ppl_cell='ppl',arm='B',population_rows=193,records={'5':recs},per_layer=[dict(layer=5,selected=count,pairs=16*193*194//2)])
    p=tmp_path/'capture.json';p.write_text(json.dumps(manifest))
    monkeypatch.setattr(metrics,'require_a2',lambda n:dict(result=dict(manifest=str(p))))
    monkeypatch.setattr(metrics,'load_record',lambda rec:raw_by_S[rec['S']])
    monkeypatch.setattr(metrics,'A',tmp_path)
    def save(path,x):
        path.parent.mkdir(parents=True,exist_ok=True);np.save(path,x);return dict(path=str(path))
    monkeypatch.setattr(metrics,'save_array',save)
    seen=[]
    def selective(q,k,kq,v,scale,z,causal):
        L,S=q.shape[2],k.shape[2];seen.append((L,S));assert np.all(kq.x==(1 if S==129 else 2))
        raw=raw_by_S[S];return T(raw['out']),T(raw['mask']),T(np.zeros((1,16,L,S),np.float32))
    diag=SimpleNamespace(selective=selective,bulk_scores=lambda q,k,scale:T(np.zeros((1,16,q.shape[2],k.shape[2]),np.float32)))
    monkeypatch.setitem(sys.modules,'_apa_sp4g_diag',diag)
    tc=SimpleNamespace(tensor=lambda x,dtype:T(x),no_grad=nullcontext,empty_cache=lambda:None)
    monkeypatch.setattr(metrics,'load_runtime',lambda:tc)
    import apa_sp4g_metrics as historical_metrics
    monkeypatch.setattr(historical_metrics,'A',tmp_path)
    c=dict(id='margin',arm='B',S=194,population_rows=193,layer=5,depends=['ppl'])
    result=metrics.whole_layer(c)
    assert seen==[(129,129),(64,193)] and result['original_calls']==2
    assert result['pairs']==16*193*194//2 and result['selected']==count and result['selection_source']=='ppl'
    assert result['selection_bitwise'] and result['replay_bitwise'] and len(result['files'])==3
    assert result['error']==dict(mean=0.,p99=0.,p99_9=0.,max=0.)
