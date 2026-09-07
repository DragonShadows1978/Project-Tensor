"""Author baseline (not blind review). Prior art: SP3/SP1 independent NumPy
references, constructed-input/negative-path tests; DeMillo/Lipton/Sayward1978
mutation testing (unverified lead: Hints on Test Data Selection). New MQA seam
coverage only. See registration for fixed gates; no CUDA numerical claims here.
"""
import ast,hashlib,importlib.util,json,os,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'scripts'))
import apa_sp4g_common as common
import apa_sp4g_registry as registry
import apa_sp4g_model as model
import apa_sp4g_metrics as metrics
import apa_sp4g_gpu as gpu
import apa_sp1_reference as ref
import apa_sp1_1_reference as split
# Mutants are independent copies; tests import exactly one selected module copy.
if os.environ.get('APA_SP4G_MUTANT'):
    name,path=os.environ['APA_SP4G_MUTANT'].split(':',1)
    spec=importlib.util.spec_from_file_location('apa_sp4g_mutant',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    globals()[name]=module

def test_registration_and_all_preexisting_source_bytes_pinned():
    r=common.verify_sources();assert common.sha(common.A/'registration.json')==common.REG_SHA
    assert r['parent_registrations']['artifacts/apa_sp3/registration.json']==common.sha(R/'artifacts/apa_sp3/registration.json')
    p=Path('/mnt/ForgeRealm/Project-Tensor-wt-apa-sp2/artifacts/apa_sp2/registration.json')
    assert common.sha(p)==r['parent_registrations'][str(p)]
    assert r['safety']['worker_term_s']+r['safety']['own_child_grace_s']<=290

def test_protocol_offline_int64_stream_and_windows():
    ids=common.tokens();r=common.registration()['protocol']
    assert len(ids)==292282 and ids.dtype.str=='<i8'
    assert r['window_starts']==[0,2048,4096,6144]
    assert hashlib.sha256(ids.tobytes()).hexdigest()==r['token_sha256']

@pytest.mark.parametrize('S,scored',[(2048,1024),(8192,512),(16384,512),(32768,512)])
def test_exact_target_boundaries_and_no_final_target_outside_input(S,scored):
    blocks=list(model.scoring_blocks(S,scored));got=np.concatenate([np.arange(p+1,p+n+1) for p,n in blocks])
    np.testing.assert_array_equal(got,np.arange(S-scored,S))
    assert all(1<=n<=64 for p,n in blocks)

def test_nll_matches_independent_logaddexp_fp64_oracle():
    rng=np.random.default_rng(7);x=rng.normal(size=(64,17))*1000;y=rng.integers(0,17,64)
    expected=np.sum(np.logaddexp.reduce(x,axis=1)-x[np.arange(64),y],dtype=np.float64)
    assert model.nll(x,y)==pytest.approx(expected,rel=1e-14)
    assert model.nll(x+1234,y)==pytest.approx(expected,rel=1e-14)

def test_nll_targets_change_answer():
    x=np.array([[9.,0.,0.],[0.,9.,0.]])
    assert model.nll(x,[0,1])<.001
    assert model.nll(x,[1,2])>17

@pytest.mark.parametrize('x,y',[(np.zeros((0,4)),[]),(np.array([[np.nan,0]]),[0]),(np.ones((2,4)),[1])])
def test_invalid_logit_population_rejected(x,y):
    with pytest.raises(common.Red):model.nll(x,y)

def test_d512_mqa_prefill_contract_and_dense_pin():
    s=(R/'tensor_cuda/src/kernels.cu').read_text();entry=s[s.index('NDArray apa_selective_attention_sp('):]
    assert 'else launch(std::integral_constant<int,512>{},diag);' in entry
    assert 'H % KVH != 0' in entry and '(int)(H/KVH)' in entry
    rng=np.random.default_rng(451);q=rng.normal(size=(16,3,512))*.02;k=rng.normal(size=(9,512))*.02;v=rng.normal(size=(9,512))*.1
    lengths=np.arange(3)+7
    # One K/V head shared by all16 query heads, independent dense oracle.
    for h in range(16):
        exact=(q[h]@k.T).astype(np.float32);bulk=exact+.1
        got,mask=ref.online_batch(bulk,exact,np.broadcast_to(v.astype(np.float32),(3,9,512)),np.finfo(np.float32).max,lengths)
        np.testing.assert_allclose(got,ref.dense_scores(exact,v,lengths),atol=.001,rtol=.001)
        np.testing.assert_array_equal(mask,np.arange(9)<lengths[:,None])

def test_d512_mqa_splitk_contract_and_dense_pin():
    s=(R/'tensor_cuda/src/apa_sp1_1.cuh').read_text()
    assert 'else launch(std::integral_constant<int,512>{},std::true_type{},diag);' in s
    assert 'KVH,H/KVH,num_parts,part_keys' in s
    rng=np.random.default_rng(452);q=rng.normal(size=512).astype(np.float32)*.02;k=rng.normal(size=(4097,512)).astype(np.float32)*.02;v=rng.normal(size=(4097,512)).astype(np.float32)*.1
    exact=k@q;parts=[np.arange(i,min(i+2048,4097)) for i in range(0,4097,2048)]
    got,mask=split.partition_online(exact+.1,exact,v,np.finfo(np.float32).max,parts)
    np.testing.assert_allclose(got,ref.dense_scores(exact,v),atol=.001,rtol=.001);assert mask.all()

@pytest.mark.parametrize('L,S',[(7,19),(1,4097)])
def test_compiled_d512_mqa_flag_device_scalar_guards(monkeypatch,L,S):
    tc=common.load_runtime();q=tc.tensor(np.zeros((0,16,L,512),np.float32),device='cpu');k=tc.tensor(np.zeros((0,1,S,512),np.float32),device='cpu')
    monkeypatch.delenv('TC_APA_SP',raising=False)
    with pytest.raises(RuntimeError,match='default OFF'):tc._C.apa_selective_attention_sp(q,k,k,k,1.,1.)
    monkeypatch.setenv('TC_APA_SP','1')
    with pytest.raises(RuntimeError,match='share CUDA device'):tc._C.apa_selective_attention_sp(q,k,k,k,1.,1.)
    for delta in (float('inf'),float('nan'),-1.):
        with pytest.raises(RuntimeError,match='finite nonnegative'):tc._C.apa_selective_attention_sp(q,k,k,k,1.,delta)

def test_adapter_live_threshold_same_tensor_callsite_and_incremental_kq():
    s=Path('/mnt/ForgeRealm/GraftRepository/core/gemma4_tc.py').read_text();tree=ast.parse(s)
    klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Gemma4AttentionTC')
    call=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='__call__')
    fused=[n for n in ast.walk(call) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='apa_selective_attention']
    assert len(fused)==2
    assert [[ast.unparse(a) for a in n.args[:4]] for n in sorted(fused,key=lambda n:n.lineno)]==[['q','kk','kq','vv'],['q','k','kq','v']]
    assert s.count('and S_all > self.apa_min_context)')==2
    assert 'self.apa_min_context = 2048' in s and 'self.kq_count = self.count' in s
    assert 'for s0 in range(self.kq_count, self.count, CHUNK)' in s
    assert model.ENV['GEMMA4_APA_INT4']=='0' and model.ENV['GEMMA4_APA_GEMM']=='0'

def test_clean_decode_no_attention_class_wrapper_or_nonscalar_host_copy():
    tree=ast.parse((R/'scripts/apa_sp4g_model.py').read_text())
    klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Model')
    decode=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='decode')
    copies=[n for n in ast.walk(decode) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='numpy']
    assert len(copies)==1 and ast.unparse(copies[0].func.value)=='tc.argmax_last_axis(x)'
    calls=[ast.unparse(n.func) for n in ast.walk(decode) if isinstance(n,ast.Call)]
    assert 'resident' in calls
    for loop in [n for n in ast.walk(decode) if isinstance(n,ast.For) and ast.unparse(n.iter)=='range(S, S + 32)']:
        assert not any(isinstance(n,ast.Call) and ast.unparse(n.func)=='resident' for n in ast.walk(loop))
    assert 'self.capture' not in ast.unparse(decode)

def test_native_diag_generated_d512_mqa_and_actual_bulk_store():
    s=(common.BUILD/'diagnostics.cu').read_text();b=(common.BUILD/'diag_bindings.cpp').read_text()
    assert 'apa_selective_kernel<T,512,true>' in s and 'scale,z,causal,1,H,' in s
    assert 'float qr[16]' in s and '((int64_t)b*S+j)*D' in s and 'bscores[(int64_t)row*S+j]=bulk' in s
    assert 'qs[3]!=512' in b and 'qs[1]!=16 || ks[1]!=1' in b

def test_tail_all_key_normalization_and_max_ratio_distinct():
    mass,rel=metrics.tail(np.log([1.,2.,3.]),[True,False,True]);assert mass==pytest.approx(1/3) and rel==pytest.approx(2/3)
    assert metrics.tail(np.zeros(3),[1,1,1])==(0.,0.)
    assert metrics.tail(np.zeros(3),[0,0,0])==(1.,1.)

def test_population_percentiles_do_not_average_band_percentiles():
    x=np.arange(1,1001,dtype=np.float64);r=metrics.stats(x.copy())
    assert r==dict(mean=500.5,p99=990.,p99_9=999.,max=1000.)

def test_eq_uses_both_lengths_and_upward_margin(monkeypatch):
    rows={'a':dict(result=dict(eq_sp=.25)),'b':dict(result=dict(eq_sp=1.25))}
    monkeypatch.setattr(metrics,'require_pass',lambda d:rows[d]);r=metrics.eq_result(dict(depends=['a','b']))
    assert r['eq']>=1.25 and r['delta']>=np.log(100)+2*1.25
    assert r['source_count']==2

def test_match_selects_first_eligible_and_carries():
    p=[('first',dict(delta=4.,fraction=.14)),('second',dict(delta=3.,fraction=.15))]
    assert gpu.choose_trial(p,.145)==dict(carry='first',delta=4.)

def test_bisection_direction_and_global_delta():
    assert gpu.choose_trial([],.15)['delta']==4.
    r=gpu.choose_trial([('x',dict(delta=4.,fraction=.3))],.15);assert r['delta']==2. and r['carry'] is None
    r=gpu.choose_trial([('x',dict(delta=4.,fraction=.05))],.15);assert r['delta']==18.

def test_match_outside_tolerance_does_not_carry():
    assert gpu.choose_trial([('x',dict(delta=4.,fraction=.162))],.15)['carry'] is None

def test_registry_all_cells_ordered_and_required_grid_complete():
    seen=set();cs=registry.cells();d=registry.by_id()
    for c in cs:
        assert c['id'] not in seen and set(c['depends'])<=seen;seen.add(c['id'])
        assert c['worker_s']<=285 and c['apa_min_context']==0 and c['kind'] in common.CLOSURES
    for arm in 'ABCDE':
        for S in (4096,8192,16384,24576,32768):assert f'ceiling_{arm}_{S}' in d
        assert len(d[f'ppl_{arm}_2048']['depends'])==4
    for S in (2048,8192):
        for arm in 'BC':
            for l in registry.LAYERS:assert len(d[f'margin_{arm}_{S}_l{l:02d}']['depends'])==S//128
    for arm in 'ABC':assert f'decode_{arm}_8192' in d[f'decode_{arm}_32768']['depends']
    assert len(d['eq']['depends'])==32

def test_create_only_receipt_preserves_first_bytes(tmp_path):
    p=tmp_path/'r.json';common.publish(p,dict(value=1));first=p.read_bytes()
    with pytest.raises(FileExistsError):common.publish(p,dict(value=2))
    assert p.read_bytes()==first

def receipt_fixture(monkeypatch,tmp_path):
    c=dict(id='test',kind='kernel',depends=[])
    monkeypatch.setattr(registry,'by_id',lambda:{'test':c})
    monkeypatch.setattr(common,'job_path',lambda n:tmp_path/(n+'.json'))
    monkeypatch.setattr(common,'fingerprint',lambda c:{'code':'live'})
    j=dict(cell=c,status='PASS',registration_sha256=common.REG_SHA,fingerprint={'code':'live'},dependencies={},result={})
    return c,j,tmp_path/'test.json'

def test_receipt_stale_source_rejected(monkeypatch,tmp_path):
    c,j,p=receipt_fixture(monkeypatch,tmp_path);j['fingerprint']={'code':'stale'};common.publish(p,j)
    with pytest.raises(common.Red,match='STALE'):common.require_pass('test')

def test_receipt_forged_dependency_rejected(monkeypatch,tmp_path):
    c,j,p=receipt_fixture(monkeypatch,tmp_path);j['dependencies']={'ghost':'hash'};common.publish(p,j)
    with pytest.raises(common.Red,match='DEPENDENCY'):common.require_pass('test')

def test_receipt_red_not_reused(monkeypatch,tmp_path):
    c,j,p=receipt_fixture(monkeypatch,tmp_path);j['status']='RED';common.publish(p,j)
    with pytest.raises(common.Red,match='RED'):common.require_pass('test')

def test_payload_rewrite_rejected(monkeypatch,tmp_path):
    c,j,p=receipt_fixture(monkeypatch,tmp_path);payload=tmp_path/'arr.npy';payload.write_bytes(b'abcd')
    st=payload.stat();j['result']['files']=[dict(path=str(payload),sha256=common.sha(payload),stat={k:getattr(st,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')})]
    common.publish(p,j);payload.write_bytes(b'efgh')
    with pytest.raises(common.Red,match='PAYLOAD'):common.require_pass('test')

def test_runner_rails_lease_and_no_foreign_signal_commands():
    s=(R/'scripts/apa_sp4g_lead_gpu.sh').read_text()
    assert 'flock --exclusive --wait 20 9' in s
    assert '--kill-after=5s 285s' in s and '--kill-after=3s 585s' in s and 'sleep 30' in s
    assert 'pkill' not in s and 'killall' not in s and 'LD_PRELOAD=' not in s

def test_long_decode_planned_rail_is_unknown_fit(monkeypatch):
    monkeypatch.setattr(gpu,'require_pass',lambda d:dict(result=dict(steps=32,fit=True,setup_s=100,prefill_s=20,decode_s=5)))
    r=gpu.planning(dict(kind='decode',arm='C',S=32768))
    assert r['outcome']=='PLANNED_RAIL' and r['fit'] is None and r['estimate_s']==455

@pytest.mark.parametrize('S,scored',[(2048,1024),(8192,512)])
def test_perplexity_feeds_exact_context_then_every_required_target(S,scored):
    from contextlib import nullcontext
    seen=[]
    class Logits:
        def __init__(self,x):self.x=x
        def float(self):return self
        def numpy(self):return self.x
    def forward(ids,caches=None,position_offset=0,last_token_only=False):
        seen.append((position_offset,ids.shape[1],last_token_only))
        x=np.zeros((*ids.shape,7),np.float32)
        for i,t in enumerate(ids[0]):x[0,i,(int(t)+1)%7]=3.
        return Logits(x),['cache']
    owner=model.Model.__new__(model.Model);owner.tc=SimpleNamespace(no_grad=nullcontext,empty_cache=lambda:None,synchronize=lambda:None);owner.model=forward;owner.load_s=0.;owner.counts_result=lambda:{}
    ids=np.arange(S,dtype=np.int64)%7;r=owner.perplexity(ids,scored)
    assert seen[0]==(0,S-scored-1,True)
    assert r['targets']==scored and r['total_nll']==pytest.approx(scored*np.log1p(6*np.exp(-3)),rel=1e-13)
    assert seen[1][0]==S-scored-1 and sum(n for _,n,_ in seen[1:])==scored

def test_sp_binding_dispatch_preserves_all_four_tensor_objects():
    tree=ast.parse((R/'scripts/apa_sp4g_model.py').read_text());fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='native_sp')
    calls=[];tc=SimpleNamespace(_C=SimpleNamespace(apa_selective_attention_sp=lambda *args:calls.append(args) or 'out'))
    ns=dict(tc=tc,delta=7.5);exec(compile(ast.Module(body=[fn],type_ignores=[]),'dispatch','exec'),ns)
    q,k,kq,v=[object() for _ in range(4)]
    assert ns['native_sp'](q,k,kq,v,1.,1.03,True)=='out'
    assert all(a is b for a,b in zip(calls[0][:4],(q,k,kq,v)))
    assert calls[0][4:]==(1.,7.5,True,None,False)

def test_live_gemma_kq_cache_only_quantizes_new_rows(monkeypatch):
    from contextlib import nullcontext
    common.load_runtime();sys.path.insert(0,'/mnt/ForgeRealm/GraftRepository')
    from core import gemma4_tc as g
    class Tensor:
        shape=(1,1,4096,512)
        def slice(self,axis,lo,n):return ('view',lo,n)
    c=g.KVRing.__new__(g.KVRing);c.count=2048;c.kq_count=0;c.kqb=None;c.kb=Tensor();c.q4=False;c.cap=4096
    monkeypatch.setattr(g,'_zeros',lambda *shape:Tensor())
    monkeypatch.setattr(g,'tc',SimpleNamespace(no_grad=nullcontext,write_rows=lambda *args:None,empty_cache=lambda:None))
    monkeypatch.setattr(g.KVRing,'_k_get',lambda self,lo,n:(lo,n))
    calls=[]
    def quant(x):calls.append(x);return x
    assert c.quantized_keys(quant)==('view',0,2048)
    assert calls==[(0,512),(512,512),(1024,512),(1536,512)]
    c.count+=1;calls.clear();c.quantized_keys(quant)
    assert calls==[(2048,1)] and c.kq_count==2049
    calls.clear();c.quantized_keys(quant);assert calls==[]

def test_exactness_gate_rejects_model_ppl_miss(monkeypatch):
    monkeypatch.setattr(gpu,'require_pass',lambda name:dict(result=dict(ppl={'A':9.,'D':9.006}[name])))
    with pytest.raises(common.Red,match='D_A_EXACTNESS_FAILED'):gpu.execute(dict(kind='exactness',depends=['A','D']))

def test_exactness_gate_accepts_registered_tolerance(monkeypatch):
    monkeypatch.setattr(gpu,'require_pass',lambda name:dict(result=dict(ppl={'A':9.,'D':9.004}[name])))
    r=gpu.execute(dict(kind='exactness',depends=['A','D']));assert r['absolute_difference']<.005
