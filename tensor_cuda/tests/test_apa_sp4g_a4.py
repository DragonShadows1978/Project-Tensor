"""A4 author CPU gates. Prior art: SP3 (2026) seam negative tests,
DeMillo/Lipton/Sayward (1978) mutation tests; no GPU/blind-review claim.
"""
import importlib.util,os,sys,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'scripts'))
import apa_sp4g_a4_model as model
import apa_sp4g_a4_common as common
import apa_sp4g_a4_gpu as gpu
import apa_sp4g_a4_registry as registry
if os.environ.get('APA_SP4G_A4_MUTANT'):
    name,path=os.environ['APA_SP4G_A4_MUTANT'].split(':',1)
    spec=importlib.util.spec_from_file_location('a4_mutant',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);globals()[name]=module

class T:
    def __init__(self,value=0.,dtype='bfloat16',shape=(1,16,2,512),broken=False):
        self.value=value;self.dtype=dtype;self.shape=shape;self.broken=broken
    def astype(self,dtype):return self if self.broken else T(self.value,dtype,self.shape)
    def float(self):return self.astype('float32')
    def reshape(self,shape):return T(self.value,self.dtype,tuple(shape))
    def numpy(self):return np.full(self.shape,self.value,np.float32)

def fixture_tc(bad_output=False):
    seen=[]
    def sp(*args):
        seen.append(args)
        return T(3.,'bfloat16' if bad_output else 'float32')
    def mm(a,b,alpha=1.,trans_b=False):
        shape=tuple(a.shape[:-1])+(b.shape[-2] if trans_b else b.shape[-1],)
        return T(7.,a.dtype,shape)
    return SimpleNamespace(_C=SimpleNamespace(apa_selective_attention_sp=sp),matmul=mm,causal_softmax=lambda x:x),seen

def tensors():return [T(1.),T(2.,shape=(1,1,3,512)),T(4.,shape=(1,1,3,512)),T(8.,shape=(1,1,3,512))]

def test_a4_fp32_native_dispatch_dtype_pin_and_returned_arm():
    tc,seen=fixture_tc();inputs=tensors()
    out,pins=model.fp32_attention(tc,*inputs,1.,True,'D32')
    assert len(seen)==1 and all(t.dtype=='float32' for t in seen[0][:4])
    assert [t.value for t in seen[0][:4]]==[1.,2.,4.,8.]
    assert seen[0][4:]==(1.,model.MAX_DELTA,True,None,False)
    assert pins['native_output']=={'out':'float32'} and pins['implementation']=='SP'
    assert out.dtype=='bfloat16' and out.value==3.
    a,ap=model.fp32_attention(tc,*inputs,1.,True,'A32')
    assert a.value==7. and a.dtype=='bfloat16' and ap['implementation']=='standard' and len(seen)==1

def test_a4_dtype_pin_rejects_silent_failed_cast_before_sp():
    tc,seen=fixture_tc();inputs=tensors();inputs[2].broken=True
    with pytest.raises(common.Red,match='NATIVE_DTYPE_PIN'):model.fp32_attention(tc,*inputs,1.,True,'D32')
    assert not seen

def test_a4_dtype_pin_rejects_native_bf16_output():
    tc,_=fixture_tc(bad_output=True)
    with pytest.raises(common.Red,match='NATIVE_DTYPE_PIN'):model.fp32_attention(tc,*tensors(),1.,True,'D32')

def pin_records(S=2048):
    chunks=[(0,512),(512,511)]+[(1023+64*i,64) for i in range(16)]
    return [dict(layer=l,lo=lo,n=n,S_all=lo+n,source={k:'bfloat16' for k in ('q','k','kq','v')},
        native_inputs={k:'float32' for k in ('q','k','kq','v')},native_output={'out':'float32'},
        returned={'out':'bfloat16'},implementation='SP') for lo,n in chunks for l in model.LAYERS]

def test_a4_all_call_pins_require_full_population_all_layers():
    records=pin_records();assert len(records)==144
    model.validate_pins(records,2048)
    with pytest.raises(common.Red,match='COVERAGE'):model.validate_pins(records[:-1],2048)
    with pytest.raises(common.Red,match='MISSING_LAYER'):model.validate_pins([r for r in records if r['layer']!=5],2048)
    broken=[dict(r) for r in records];broken[0]['native_inputs']={'q':'float32'}
    with pytest.raises(common.Red,match='INCOMPLETE_DTYPE'):model.validate_pins(broken,2048)

def test_a4_coverage_rejects_gaps_and_empty():
    model.validate_coverage([dict(lo=0,n=2),dict(lo=2,n=3)],6)
    with pytest.raises(common.Red,match='COVERAGE_GAP'):model.validate_coverage([dict(lo=1,n=2),dict(lo=3,n=3)],6)
    with pytest.raises(common.Red,match='COVERAGE'):model.validate_coverage([],6)

def test_a4_gate_inclusive_tolerance_finite_and_dtype(monkeypatch):
    monkeypatch.setattr(common,'read',lambda p:dict(exactness=dict(reference_ppl=0.,tolerance=.005)))
    def r(ppl):return dict(ppl=ppl,dtype_pin_complete=True,dtype_pins=[dict(implementation='SP')])
    assert common.exactness(r(.005))
    assert not common.exactness(r(.00500001))
    assert not common.exactness(r(float('nan')))
    assert not common.exactness(r(float('inf')))
    assert not common.exactness(dict(r(0),dtype_pin_complete=False))
    assert not common.exactness(dict(r(0),dtype_pins=[]))
    assert not common.exactness(dict(r(0),dtype_pins=[dict(implementation='standard')]))

def test_a4_registration_preservation_and_transitive_gate():
    before=common.preserved();assert len(before['receipt_sha256'])>=75
    cs=registry.cells();seen=set();ancestors={}
    for c in cs:
        assert c['id'] not in seen and c['worker_s']==285 and c['S']<=8192
        assert all(d in seen or d not in registry.by_id() for d in c['depends'])
        ancestors[c['id']]=set(c['depends']).union(*(ancestors.get(d,set()) for d in c['depends']))
        if c['id']!=registry.GATE:assert registry.GATE in ancestors[c['id']]
        seen.add(c['id'])
    assert len([c for c in cs if c['kind']=='margin'])==16
    assert len([c for c in cs if c['kind']=='trial'])==12
    assert len([c for c in cs if c['kind']=='decode'])==2
    assert len([c for c in cs if c['kind']=='precision'])==5
    assert all(c['capture_cell'] in c['depends'] for c in cs if c['kind']=='margin')
    for name in ('ppl_capture_B_2048_w0','ppl_capture_B_8192_w0','diag_a2_fp32_A_2048_w0'):
        assert common.require_a4(name)['status']=='PASS'

def test_a4_receipt_fingerprint_dependency_payload_and_red_rejected(monkeypatch,tmp_path):
    c=dict(id='unit',kind='aggregate',depends=[])
    monkeypatch.setattr(common,'by_id',lambda:{'unit':c})
    monkeypatch.setattr(common,'path_a4',lambda n:tmp_path/'receipt.json')
    monkeypatch.setattr(common,'fingerprint',lambda c:dict(source='good'))
    j=dict(cell=c,status='PASS',registration_sha256=common.REG_SHA,a4_registration_sha256=common.REGISTRATION_SHA,
           fingerprint=dict(source='bad'),dependencies={},result={})
    def put():(tmp_path/'receipt.json').write_text(json.dumps(j))
    put()
    with pytest.raises(common.Red,match='STALE_OR_RED'):common.require_a4('unit')
    j['fingerprint']={'source':'good'};put();assert common.require_a4('unit')['status']=='PASS'
    j['dependencies']={'unregistered':'bad'};put()
    with pytest.raises(common.Red,match='DEPENDENCY'):common.require_a4('unit')
    j['dependencies']={};j['status']='RED';put()
    with pytest.raises(common.Red,match='STALE_OR_RED'):common.require_a4('unit')
    payload=tmp_path/'data';payload.write_bytes(b'good');j['status']='PASS'
    j['result']={'files':[dict(path=str(payload),sha256=common.sha(payload))]};put()
    payload.write_bytes(b'bad')
    with pytest.raises(common.Red,match='PAYLOAD'):common.require_a4('unit')

def test_a4_unregistered_long_cells_rejected_without_execution():
    for S in (16384,24576,32768):
        with pytest.raises(common.Red,match='UNREGISTERED_OR_LONG_RAIL'):gpu.execute(dict(id='unknown',S=S))
        with pytest.raises(common.Red,match='UNREGISTERED_OR_LONG_RAIL'):gpu.preflight(dict(id='unknown',S=S))

def test_a4_propagation_aggregates_squared_norms_and_pairs_schedule(monkeypatch,tmp_path):
    monkeypatch.setattr(gpu,'R',tmp_path);results={}
    for arm in ('A','D'):
        residuals={}
        for layer in [str(l) for l in model.LAYERS]+['final_norm']:
            recs=[]
            for i,val in enumerate((1.,10.)):
                p=tmp_path/f'{arm}_{layer}_{i}.npy';np.save(p,np.full((1,1,2),val+(1. if arm=='D' else 0.),np.float32))
                recs.append(dict(lo=i,n=1,file=dict(path=str(p))))
            residuals[layer]=recs
        p=tmp_path/f'{arm}.json';p.write_text(json.dumps(dict(residuals=residuals)))
        results[arm]=dict(result=dict(manifest=str(p),ppl=50 if arm=='A' else 51))
    monkeypatch.setattr(gpu,'require_a4',lambda n:results[n])
    c=dict(depends=['A','D'],S=3);r=gpu.propagation_result(c)
    assert len(r['per_layer'])==9
    for row in r['per_layer']:
        assert row['relative_frobenius']==pytest.approx(np.sqrt(4/202))
        assert row['relative_frobenius']!=pytest.approx((1+.1)/2)
    p=tmp_path/'D.json';m=json.loads(p.read_text());m['residuals']['5'][0]['lo']=1;p.write_text(json.dumps(m))
    with pytest.raises(common.Red,match='COVERAGE'):gpu.propagation_result(c)

def test_a4_residual_instrumentation_captures_full_block_and_norm_restores(monkeypatch):
    class Block:
        def __call__(self,x,ropes,position_offset=0,cache=None):return T(11.,shape=(1,2,3840)),cache
    original=Block.__call__;norm=lambda h:T(17.,shape=h.shape)
    def setup(self,cell,delta=None):
        self.cell=cell;self.gemma=SimpleNamespace(Gemma4BlockTC=Block,_cast=lambda x:x)
        self.model=SimpleNamespace(layers=[Block() for _ in range(48)],norm=norm)
        self.tc=SimpleNamespace();self.original='original_dispatch'
    monkeypatch.setattr(model.Model,'__init__',setup)
    owner=model.ResidualModel(dict(id='test',arm='D'));saved=[]
    owner.save_residual=lambda layer,t,offset:saved.append((layer,t.value,offset))
    block=owner.model.layers[47];h,cache=block(T(),None,512,'cache')
    out=owner.model.norm(h)
    assert saved==[('47',11.,512),('final_norm',17.,512)] and out.value==17. and cache=='cache'
    owner.close();assert Block.__call__ is original and owner.model.norm is norm

def test_a4_precision_install_routes_all_global_calls_and_restores(monkeypatch):
    # Exercise the inherited class interception, rather than a source-text pin.
    class Mixer:pass
    mix=Mixer();mix.is_global=True;mix.attention_mode='standard'
    owner=model.PrecisionModel.__new__(model.PrecisionModel)
    owner.cell=dict(kind='precision',treatment='D32');owner.layer_ids={id(mix):5};owner.probes=[]
    owner.gemma=SimpleNamespace(Gemma4AttentionTC=Mixer);owner.tc,seen=fixture_tc()
    owner.original_matmul=owner.tc.matmul;owner.original='dispatch_before';args=tensors()
    def original(m,*unused):return owner.dispatch(*args,1.,1.,True),'cache'
    owner.original_attn=original;owner.install()
    out,_=owner.gemma.Gemma4AttentionTC.__call__(mix,None,None,None,0,None)
    assert out.value==3. and len(seen)==1 and owner.probes[0]['native_inputs']['q']=='float32'
    assert mix.attention_mode=='standard';owner.close()
    assert owner.gemma.Gemma4AttentionTC.__call__ is original and owner.tc.apa_selective_attention=='dispatch_before'

def test_a4_calibration_uses_actual_ppl_fraction_receipt(monkeypatch):
    # SP4G A2 (2026): real PPL receipt schema differs from old prefill capture.
    # Trial carry lets this exercise execution without loading a GPU model.
    b=common.require_a4('ppl_capture_B_2048_w0')
    assert 'fraction' not in b['result']
    target=b['result']['global_fraction']['fraction']
    trial=dict(result=dict(fraction=target,delta=4.,global_fraction=dict(fraction=target)))
    monkeypatch.setattr(gpu,'require_a4',lambda n:b if n=='ppl_capture_B_2048_w0' else trial)
    r=gpu.execute(registry.by_id()['trial_a4_01'])
    assert r['fraction']==target and r['delta']==4. and r['no_model_execution']
    r=gpu.execute(registry.by_id()['freeze_a4'])
    assert r['target']==target and r['match_abs']==0 and r['delta']==4.
