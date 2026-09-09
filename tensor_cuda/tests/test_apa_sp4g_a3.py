"""A3 author CPU gates, not GPU or blind verification.
Prior art: SP3 (2026) negative seam/provenance tests; DeMillo/Lipton/Sayward
(1978) mutation testing (unverified lead: Hints on Test Data Selection).
"""
import importlib.util,json,os,sys
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import numpy as np
import pytest
R = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'scripts'))
import apa_sp4g_a3_math as math3
import apa_sp4g_a3_model as model
import apa_sp4g_a3_sweep as sweep
import apa_sp4g_a3_common as common
import apa_sp4g_a3_registry as registry
import apa_sp4g_a3_gpu as gpu
if os.environ.get('APA_SP4G_A3_MUTANT'):
    name,path = os.environ['APA_SP4G_A3_MUTANT'].split(':',1)
    spec = importlib.util.spec_from_file_location('a3_mutant',path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    globals()[name] = module

class T:
    def __init__(self,x,dtype='float32'):
        self.dtype = dtype
        x = np.asarray(x)
        self.x = math3.bf16_round(x) if dtype == 'bfloat16' else x.astype(np.float32)
        self.shape = self.x.shape
        self.device = 'cuda:0'
    def reshape(self,shape): return T(self.x.reshape(shape),self.dtype)
    def transpose(self,a,b): return T(self.x.swapaxes(a,b),self.dtype)
    def float(self): return T(self.x)
    def numpy(self): return self.x
    def astype(self,dtype): return T(self.x,dtype)
    def slice(self,axis,start,n):
        key = [slice(None)]*self.x.ndim; key[axis] = slice(start,start+n)
        return T(self.x[tuple(key)],self.dtype)

def fake_tc():
    def mm(a,b,alpha=1.,trans_b=False):
        return T(np.float32(alpha)*(a.x@(b.x.swapaxes(-1,-2) if trans_b else b.x)),a.dtype)
    def sm(t):
        L,S = t.shape[-2:]
        mask = np.arange(S) < (S-L+np.arange(L)+1)[:,None]
        x = np.where(mask,t.x,np.float32(-np.inf))
        p = np.exp(x-x.max(-1,keepdims=True))
        return T(p/p.sum(-1,keepdims=True),t.dtype)
    def sp(q,k,kq,v,scale,delta,causal,sinks,diagnostics):
        out = T(math3.dense(q.x,k.x,v.x,scale,'none' if sinks is None else 'zero',causal,
                           position_offset=k.shape[2]-q.shape[2]),q.dtype)
        if diagnostics:
            mask = np.broadcast_to(math3.eligible(q.shape[2],k.shape[2],causal,'native_bottom_right',0),
                                   (q.shape[0],q.shape[1],q.shape[2],k.shape[2]))
            return out,T(mask)
        return out
    return SimpleNamespace(matmul=mm,causal_softmax=sm,_C=SimpleNamespace(apa_selective_attention_sp=sp),
                           tensor=lambda x,dtype:T(x,dtype),cat=lambda xs,dim:T(np.concatenate([x.x for x in xs],axis=dim),xs[0].dtype),
                           no_grad=nullcontext,synchronize=lambda:None)

def test_a3_registration_preservation_and_real_prefix_geometry():
    before = common.preserved()
    assert len(before['receipt_sha256']) > 50
    assert common.sha(registry.REGISTRATION) == registry.REGISTRATION_SHA
    cs = registry.cells()
    assert [c['id'] for c in cs] == ['diag_a3_call_l05_c0','diag_a3_call_l05_c1','diag_a3_sweep_l05']
    assert [(c['L'],c['S_all'],c['position_offset']) for c in cs[:2]] == [(512,512,0),(511,1023,512)]
    assert all(c['worker_s']==285 and c['S']==2048 for c in cs)
    assert cs[2]['depends'] == [c['id'] for c in cs[:2]]
    assert all(c['kind'] in ('call','sweep') for c in cs)
    reg = common.read(registry.REGISTRATION)
    assert reg['thresholds']['ppl_abs']==.005
    assert reg['thresholds']['fp32_max_abs']==math3.MAX_ABS
    assert reg['thresholds']['fp32_relative_frobenius']==math3.REL_FROB

def test_a3_frobenius_uses_reference_norm_and_both_bounds():
    r = math3.compare(np.array([3.,4.]),np.array([0.,4.]))
    assert r['max_abs']==3. and r['relative_frobenius']==.75
    # Frobenius passes, absolute fails: AND is load-bearing.
    assert not math3.compare([100.0002],[100.])['fp32_agreement']
    # Absolute passes, Frobenius fails.
    assert not math3.compare([1e-6],[0.])['fp32_agreement']
    assert math3.compare([0.],[0.])['fp32_agreement']

@pytest.mark.parametrize('a,b',[([],[]),([np.nan],[0]),([np.inf],[0]),([1,2],[1])])
def test_a3_metrics_reject_empty_nonfinite_and_shape(a,b):
    with pytest.raises(common.Red,match='INVALID_COMPARISON'): math3.compare(a,b)

def test_a3_bitwise_includes_signed_zero_and_no_tolerance():
    assert not math3.compare(np.array([0.],np.float32),np.array([-0.],np.float32))['bitwise']
    with pytest.raises(common.Red,match='BITWISE'):
        model.exact(np.array([1.],np.float32),np.array([1.0000001],np.float32),'unit')

def test_a3_bf16_ties_even_and_negative_values():
    bits = np.array([0x3f808000,0x3f818000,0xbf808000,0xbf818000,0x80000000,0x00010000],np.uint32)
    got = math3.bf16_round(bits.view(np.float32)).view(np.uint32)
    np.testing.assert_array_equal(got,[0x3f800000,0x3f820000,0xbf800000,0xbf820000,0x80000000,0x00010000])

def test_a3_dense_analytic_zero_scores_mask_sink_and_shared_heads():
    q=np.zeros((1,4,2,2),np.float32);k=np.zeros((1,1,5,2),np.float32)
    v=np.arange(10,dtype=np.float32).reshape(1,1,5,2)
    base=math3.dense(q,k,v,position_offset=3)
    np.testing.assert_allclose(base[0,0],[[3,4],[4,5]],rtol=0,atol=1e-6)
    np.testing.assert_array_equal(base[:,0],base[:,3])
    zero=math3.dense(q,k,v,sink='zero',position_offset=3)
    np.testing.assert_allclose(zero[0,0],[[12/5,16/5],[20/6,25/6]],rtol=1e-6)
    left=math3.dense(q,k,v,offset='zero_rowwise',position_offset=3)
    np.testing.assert_allclose(left[0,0],[[0,1],[1,2]],rtol=0,atol=1e-6)
    full=math3.dense(q,k,v,causal=False,position_offset=3)
    np.testing.assert_allclose(full[0,0],[[4,5],[4,5]],rtol=0,atol=1e-6)

def test_a3_dense_nontrivial_scale_and_group_mapping_independent_scalar_reference():
    rng=np.random.default_rng(902)
    q=rng.normal(size=(1,4,3,6)).astype(np.float32)
    k=rng.normal(size=(1,2,5,6)).astype(np.float32)
    v=rng.normal(size=k.shape).astype(np.float32)
    got=math3.dense(q,k,v,scale=.2,position_offset=2)
    want=np.empty_like(got,dtype=np.float64)
    for h in range(4):
        for i in range(3):
            n=3+i
            scores=np.array([.2*sum(float(q[0,h,i,d])*float(k[0,h//2,j,d]) for d in range(6)) for j in range(n)])
            weights=np.exp(scores-scores.max());weights/=weights.sum()
            want[0,h,i]=sum(weights[j]*v[0,h//2,j].astype(np.float64) for j in range(n))
    np.testing.assert_allclose(got,want,rtol=2e-6,atol=2e-7)

def test_a3_offsets_absolute_bottom_right_and_invalid_offset():
    np.testing.assert_array_equal(math3.eligible(3,8,True,'absolute_rowwise',5),math3.eligible(3,8,True,'native_bottom_right',5))
    assert not np.array_equal(math3.eligible(3,8,True,'zero_rowwise',5),math3.eligible(3,8,True,'native_bottom_right',5))
    with pytest.raises(common.Red,match='OFFSET_OUT'):math3.eligible(3,8,True,'absolute_rowwise',6)

def test_a3_sweep_exercises_kernel_none_zero_causal_and_absolute_prefixes():
    tc=fake_tc();seen=[];original=tc._C.apa_selective_attention_sp
    def sp(*args):seen.append(args);return original(*args)
    tc._C.apa_selective_attention_sp=sp
    q=T(np.zeros((1,16,2,512),np.float32));k=T(np.zeros((1,1,5,512),np.float32))
    v=T(np.broadcast_to(np.arange(5)[None,None,:,None],k.shape))
    for variant in math3.variants():
        seen.clear()
        got=sweep.sp_variant(tc,q,k,k,v,variant,3).numpy()
        ref=math3.dense(q.x,k.x,v.x,**variant,position_offset=3)
        np.testing.assert_allclose(got,ref,atol=1e-6)
        assert all(a[5]==model.MAX_DELTA for a in seen)
        assert all((a[7] is None)==(variant['sink']=='none') for a in seen)
        if variant['causal'] and variant['offset']!='native_bottom_right':
            assert [a[1].shape[2] for a in seen]==([4,5] if variant['offset']=='absolute_rowwise' else [1,2])
            assert all(a[0].shape[2]==1 and a[6] is False for a in seen)
        else:assert len(seen)==1 and seen[0][6]==variant['causal']

def classifications(nominal_pass=False,winner=None,second_pass=True):
    calls=[]
    for index in range(2):
        rows=[]
        for variant in math3.variants():
            yes=(nominal_pass and variant==math3.NOMINAL) or (winner==variant and (index==0 or second_pass))
            rows.append(dict(variant=variant,vs_standard=dict(fp32_agreement=yes),vs_own_dense=dict(fp32_agreement=True),
                             standard_vs_nominal_dense=dict(fp32_agreement=True)))
        calls.append(rows)
    return calls

def test_a3_classifier_nominal_agreement_cannot_be_called_argument_rescue():
    calls=classifications(True,dict(math3.NOMINAL,sink='zero'))
    result=math3.classify(calls)
    assert result['outcome']=='NOMINAL_AGREES_NO_ARGUMENT_FIX' and not result['C_unblocked']
    assert result['single_change_matches']==[]

def test_a3_classifier_requires_standard_dense_directly_tolerance_not_transitive():
    calls=classifications(True)
    nominal=next(r for r in calls[1] if r['variant']==math3.NOMINAL)
    # Both candidate comparisons fit1e-4, while standard-reference does not.
    nominal['vs_standard']=math3.compare([10.00008],[10.00016])
    nominal['vs_own_dense']=math3.compare([10.00008],[10.])
    nominal['standard_vs_nominal_dense']=math3.compare([10.00016],[10.])
    assert nominal['vs_standard']['fp32_agreement'] and nominal['vs_own_dense']['fp32_agreement']
    assert math3.classify(calls)['outcome']!='NOMINAL_AGREES_NO_ARGUMENT_FIX'

def test_a3_classifier_requires_one_change_both_calls_and_dense_confirmation():
    winner=dict(math3.NOMINAL,sink='zero')
    assert math3.classify(classifications(False,winner))['single_change_matches']==[winner]
    assert math3.classify(classifications(False,winner,False))['single_change_matches']==[]
    two=dict(winner,causal=False)
    assert math3.classify(classifications(False,two))['single_change_matches']==[]
    calls=classifications(False,winner)
    next(r for r in calls[1] if r['variant']==winner)['vs_own_dense']['fp32_agreement']=False
    assert math3.classify(calls)['single_change_matches']==[]

def test_a3_native_trace_captures_exact_ops_and_restores_on_error():
    tc=fake_tc();mm,sm=tc.matmul,tc.causal_softmax
    q=T(np.zeros((1,16,2,512)));k=T(np.zeros((1,1,5,512)))
    with model.record_standard(tc,16,2,512) as trace:
        result=model.standard(tc,q,k,k)
    assert trace['ops']==['QK','softmax','PV'] and trace['k'] is k and trace['v'] is k
    assert trace['out'].shape==result.shape and tc.matmul is mm and tc.causal_softmax is sm
    with pytest.raises(ValueError):
        with model.record_standard(tc,16,2,512): raise ValueError('injected')
    assert tc.matmul is mm and tc.causal_softmax is sm

def test_a3_refine_all_requires_full_eligible_mask():
    tc=fake_tc();q=T(np.zeros((1,16,2,512)));k=T(np.zeros((1,1,5,512)))
    out,result=model.sp_checked(tc,q,k,k,k,1.,True)
    assert result['selected']==16*(4+5)
    original=tc._C.apa_selective_attention_sp
    def broken(*args):
        r=original(*args)
        if args[-1]:r[1].x.flat[0]=0
        return r
    tc._C.apa_selective_attention_sp=broken
    with pytest.raises(common.Red,match='MASK_MISMATCH'):model.sp_checked(tc,q,k,k,k,1.,True)

def test_a3_actual_fork_runs_A_then_APA_same_state_and_records_merge(monkeypatch,tmp_path):
    import apa_sp4g_common as base
    monkeypatch.setattr(model,'R',tmp_path);monkeypatch.setattr(base,'R',tmp_path)
    tc=fake_tc()
    class Mixer:pass
    mx=Mixer();mx.attention_mode='standard';mx.o_proj=lambda t:T(2*t.x,'bfloat16')
    owner=model.CallModel.__new__(model.CallModel)
    owner.tc=tc;owner.model=SimpleNamespace(layers=[None]*5+[SimpleNamespace(mixer=mx)],PREFILL_CHUNK=512)
    owner.gemma=SimpleNamespace(Gemma4AttentionTC=Mixer,_cast=lambda t:t.astype('bfloat16'))
    owner.cell=dict(id='unit',call_index=0,L=2,S_all=5,position_offset=3)
    owner.directory=tmp_path/'capture';owner.files=[];owner.observed_calls=[];owner.index=0;owner.got=None;owner.load_s=0
    q=T(np.ones((1,16,2,512))*.01,'bfloat16');k=T(np.ones((1,1,5,512))*.02,'bfloat16')
    v=T(np.broadcast_to(np.arange(5)[None,None,:,None],k.shape),'bfloat16')
    seen=[]
    def original(m,x,cos,sin,position_offset=0,kv_cache=None):
        seen.append((m.attention_mode,x,cos,sin,position_offset,kv_cache))
        out=model.standard(tc,q,k,v) if m.attention_mode=='standard' else tc.apa_selective_attention(q,k,k,v,1.,1.,True)
        return m.o_proj(owner.gemma._cast(out.transpose(1,2).reshape([1,2,8192]))),(k,v)
    owner.original_attn=original;owner.original=lambda *a:None;tc.apa_selective_attention=owner.dispatch
    owner.install();x=T(np.zeros((1,2,8)));cache=(T(np.zeros((1,1,3,512))),T(np.zeros((1,1,3,512))))
    cos,sin=object(),object()
    with pytest.raises(model.CapturedCall):Mixer.__call__(mx,x,cos,sin,3,cache)
    assert [s[0] for s in seen]==['standard','apa_selective','apa_selective']
    assert all(s[1] is x and s[2] is cos and s[3] is sin and s[5] is cache for s in seen)
    assert mx.attention_mode=='standard' and owner.got is not None
    manifest=json.loads((tmp_path/owner.got['manifest']).read_text())
    assert {'native_scores','native_p','native_out','q','k','kq','v','dense_fp32',
            'standard_fp32_cast_bf16','sp_fp32_projected','actual_D_projected'} <= set(manifest['arrays'])
    assert manifest['metadata']['implicit_query_offset']==3
    assert manifest['metadata']['mask_bf16']['all_refined']
    owner.close();assert Mixer.__call__ is original

def test_a3_fingerprint_dependency_and_payload_rejections(monkeypatch,tmp_path):
    c=dict(id='unit',kind='call',depends=[]);p=tmp_path/'receipt.json'
    monkeypatch.setattr(common,'by_id',lambda:{'unit':c});monkeypatch.setattr(common,'path_a3',lambda n:p)
    monkeypatch.setattr(common,'fingerprint',lambda c:{'source':'live'});monkeypatch.setattr(common,'R',tmp_path)
    payload=tmp_path/'q.npy';payload.write_bytes(b'abc')
    j=dict(cell=c,status='PASS',registration_sha256=common.REG_SHA,a3_registration_sha256=common.REGISTRATION_SHA,
           fingerprint={'source':'live'},dependencies={},result=dict(files=[dict(path='q.npy',sha256=common.sha(payload))]))
    p.write_text(json.dumps(j));assert common.require_a3('unit')==j
    for change in (dict(status='RED'),dict(fingerprint={'source':'other'}),dict(dependencies={'ghost':'x'})):
        p.write_text(json.dumps(dict(j,**change)))
        with pytest.raises(common.Red,match='RED|DEPENDENCY'):common.require_a3('unit')
    p.write_text(json.dumps(j));payload.write_bytes(b'abd')
    with pytest.raises(common.Red,match='PAYLOAD'):common.require_a3('unit')

def test_a3_unregistered_C_fix_AP_and_long_cells_rejected_before_model():
    for c in (dict(id='ppl_a3_AP_2048_w0'),dict(id='ppl_a3_D_fixed_2048_w0'),dict(id='C'),dict(id='long',S=16384)):
        with pytest.raises(common.Red,match='UNREGISTERED'):gpu.execute(c)
        with pytest.raises(common.Red,match='UNREGISTERED'):gpu.preflight(c)

def test_a3_runner_bounds_and_no_historical_source_changes():
    s=(R/'scripts/apa_sp4g_a3_lead_gpu.sh').read_text()
    assert '--kill-after=5s 285s' in s and '--kill-after=3s 585s' in s
    assert 'flock --exclusive --wait 20 9' in s and 'sleep 30' in s
    assert 'apa_sp4g_a3_gpu.py worker' in s and 'pkill' not in s and 'killall' not in s
