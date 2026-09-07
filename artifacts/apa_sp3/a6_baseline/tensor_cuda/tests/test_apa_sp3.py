"""Author CPU baseline, not blind verification or CUDA numerical evidence.

Prior art: independent NumPy SP1/SP1.1 specifications (2026), online softmax
(Milakov/Gimelshein 2018; Dao 2023), BLASST-style threshold and standard
constructed-input/negative-path testing. New: D96 model-seam coverage.
"""
import ctypes
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import apa_sp3_common as common
import apa_sp3_model as model
import apa_sp3_metrics as metrics
import apa_sp3_gpu as driver
import apa_sp1_reference as ref
import apa_sp1_1_reference as split


def test_every_preexisting_source_and_kernel_body_pinned():
    r=common.verify_sources()
    assert sum(map(len,r['kernel_body_pins'].values()))==107
    assert r['protocol']['token_sha256'] is None
    assert common.sha(common.ART/'registration.json')==common.REG_SHA


def test_d96_prefill_contract_and_dense_pin():
    s=(ROOT/'tensor_cuda/src/kernels.cu').read_text()
    entry=s[s.index('NDArray apa_selective_attention_sp('):]
    assert 'if (L == 1)' in entry
    assert 'else if (cap <= 128) launch(std::integral_constant<int,128>{},diag);' in entry
    rng=np.random.default_rng(901)
    # Composite D=96, V[64:96]=0, rectangle bottom-right causal.
    q=rng.normal(size=(4,96)).astype(np.float32)*.2
    k=rng.normal(size=(9,96)).astype(np.float32)*.2
    v=rng.normal(size=(9,96)).astype(np.float32);v[:,64:]=0
    exact=q@k.T*np.float32(96**-.5)
    bulk=exact+rng.normal(size=exact.shape).astype(np.float32)*.2
    lengths=np.arange(4)+6
    got,mask=ref.online_batch(bulk,exact,np.broadcast_to(v,(4,9,96)),np.finfo(np.float32).max,lengths)
    want=ref.dense_scores(exact,v,lengths)
    np.testing.assert_allclose(got,want,rtol=.001,atol=.001)
    np.testing.assert_array_equal(mask,np.arange(9)<lengths[:,None])
    assert np.all(got[:,64:]==0)


def test_d96_splitk_contract_and_dense_pin():
    s=(ROOT/'tensor_cuda/src/apa_sp1_1.cuh').read_text()
    assert 'else if (cap <= 128) launch(std::integral_constant<int,128>{},std::true_type{},diag);' in s
    assert 'apa_int4_fixed_partition_layout(S, num_parts, part_keys);' in s
    rng=np.random.default_rng(902)
    q=rng.normal(size=96).astype(np.float32)*.2
    k=rng.normal(size=(4097,96)).astype(np.float32)*.2
    v=rng.normal(size=(4097,96)).astype(np.float32);v[:,64:]=0
    exact=k@q*np.float32(96**-.5)
    bulk=exact+rng.normal(size=4097).astype(np.float32)*.2
    parts=[np.arange(i,min(4097,i+2048)) for i in range(0,4097,2048)]
    got,mask=split.partition_online(bulk,exact,v,np.finfo(np.float32).max,parts)
    np.testing.assert_allclose(got,ref.dense_scores(exact,v),rtol=.001,atol=.001)
    assert mask.all() and np.all(got[64:]==0)


@pytest.mark.parametrize('L,S',[(7,11),(1,4097)])
def test_compiled_d96_flag_device_and_scalar_guards(monkeypatch,L,S):
    tc=common.load_runtime()
    t=tc.tensor(np.zeros((0,40,L,96),np.float32),device='cpu')
    k=tc.tensor(np.zeros((0,40,S,96),np.float32),device='cpu')
    monkeypatch.delenv('TC_APA_SP',raising=False)
    with pytest.raises(RuntimeError,match='default OFF'):
        tc._C.apa_selective_attention_sp(t,k,k,k,96**-.5,1.)
    monkeypatch.setenv('TC_APA_SP','1')
    with pytest.raises(RuntimeError,match='share CUDA device'):
        tc._C.apa_selective_attention_sp(t,k,k,k,96**-.5,1.)
    with pytest.raises(RuntimeError,match='finite nonnegative'):
        tc._C.apa_selective_attention_sp(t,k,k,k,96**-.5,float('inf'))


def test_last512_scores_exactly_last512_targets():
    ids=np.arange(1024,dtype=np.int64)%5
    logits=np.zeros((1024,5),np.float64)
    for i in range(511,1023):logits[i,ids[i+1]]=3
    got=model.last512(logits,ids)
    assert got['targets']==512
    assert got['ppl']==pytest.approx(1+4*np.exp(-3),rel=1e-12)
    logits[:511]=100  # earlier query logits must not enter this window
    assert model.last512(logits,ids)['ppl']==got['ppl']


def test_mass_and_max_relative_weight_are_distinct_and_all_key_normalized():
    m,r=metrics.tail(np.log([1.,2.,3.]),[True,False,True])
    assert m==pytest.approx(1/3) and r==pytest.approx(2/3)
    assert metrics.tail(np.zeros(5),np.ones(5,bool))==(0.,0.)
    assert metrics.tail(np.zeros(5),np.zeros(5,bool))==(1.,1.)
    with pytest.raises(common.Red):metrics.tail(np.array([]),[])


def test_fraction_uses_pairs_and_reports_layer_spread():
    s=model.fraction_summary([dict(layer=0,pairs=10,selected=3),dict(layer=1,pairs=90,selected=9)])
    assert s['fraction']==.12 and s['per_layer_min']==.1 and s['per_layer_max']==.3
    with pytest.raises(common.Red):model.fraction_summary([])


def test_margin_rounds_up_and_never_uses_percentile():
    x=1.+2**-25
    assert common.upward_float32(x)>=x
    assert common.upward_float32(0)==0
    assert metrics.nearest_rank(np.arange(1000),.999)==998
    with pytest.raises(common.Red):common.upward_float32(float('nan'))


def test_initial_delta_ties_choose_smallest_and_measured_target():
    h=np.zeros(4098,dtype=int);h[0]=5;h[64]=10;h[128]=85
    d,f=metrics.initial_delta([dict(grid_gap_histogram=h.tolist())],.15)
    assert d==.5 and f==.15
    assert metrics.next_delta([dict(delta=.5,fraction=.05)],.15)==16.25


def test_sp_binding_parameter_order_and_no_wrong_arm_dispatch():
    calls=[]
    obj=model.Model.__new__(model.Model)
    obj.tc=SimpleNamespace(_C=SimpleNamespace(apa_selective_attention_sp=lambda *a: calls.append(a) or 'out'))
    obj.arm='C';obj.delta=.375;obj.observe=False
    t=SimpleNamespace(shape=(1,40,7,96))
    assert obj.selective(t,t,t,t,96**-.5,1.28155,True)=='out'
    assert calls[0][4:]==(96**-.5,.375,True,None,False)
    obj.arm='A'
    with pytest.raises(common.Red,match='wrong arm'):obj.selective(t,t,t,t,.1,1.,True)
    obj.arm='C';bad=SimpleNamespace(shape=(1,40,7,64))
    with pytest.raises(common.Red,match='padded'):obj.selective(bad,bad,bad,bad,.1,1.,True)


def test_missing_protocol_stops_before_any_model_or_gpu(monkeypatch,tmp_path):
    monkeypatch.setattr(common,'ART',tmp_path)
    with pytest.raises(common.Red,match='BLOCKED_PROTOCOL'):common.protocol()
    with pytest.raises(common.Red,match='BLOCKED_PROTOCOL'):
        driver.execute(dict(kind='parity',depends=[]))


def test_parity_stop_rail_preserves_numbers_and_never_calls_sp(monkeypatch,tmp_path):
    monkeypatch.setattr(driver,'ART',tmp_path)
    (tmp_path/'protocol_amendment.json').write_text('{}')
    seen=[]
    fake=SimpleNamespace(set=lambda arm,**kw:seen.append(arm))
    monkeypatch.setattr(model,'score_windows',lambda m,ids:dict(ppl=12.1 if seen[-1]=='A' else 12.,target_sha256='same'))
    monkeypatch.setattr(driver,'require_pass',lambda job:dict(result={a:dict(ppl=12.,target_sha256='same') for a in 'AB'}))
    with pytest.raises(common.Red,match='G0_DETERMINISM_MISS'):driver.g0_guard(fake,np.arange(6144))
    assert seen==['A','B']
    assert len(list((tmp_path/'progress').glob('*.json')))==1


# Prior art: constructed negative-path tests and independent scoring oracles,
# standard verification practice. Added here for the lead's PROTOCOL-2 (2026).
def test_protocol2_registered_complete_stream_and_bos():
    p,ids=common.protocol()
    assert p['status']==common.PROTOCOL2_STATUS
    assert p['token_count']==len(ids)==333337 and ids.dtype.str=='<i8'
    assert ids[0]==1 and p['tokenizer']['eos_token_id']==73440
    assert p['corpus']['rows']==4358 and p['corpus']['characters']==1289979
    assert p['tokens_sha256']==common.sha(common.ART/'protocol2_tokens.npy')
    assert p['token_sha256']==hashlib.sha256(ids.tobytes()).hexdigest()


def protocol_fixture(monkeypatch,tmp_path,allow_manifest_repin=False):
    original=common.ART
    j=common.read(original/'protocol_amendment.json')
    (tmp_path/'registration.json').write_bytes((original/'registration.json').read_bytes())
    manifest=tmp_path/'protocol_amendment.json'
    manifest.write_bytes((original/'protocol_amendment.json').read_bytes())
    monkeypatch.setattr(common,'ART',tmp_path)
    def save():
        manifest.write_text(json.dumps(j))
        if allow_manifest_repin:
            # Isolate deeper semantic pins after deliberately bypassing ONLY
            # the outer digest in this test fixture; production has no bypass.
            monkeypatch.setattr(common,'PROTOCOL2_SHA',common.sha(manifest))
    return j,save


@pytest.mark.parametrize('attack',['status','tolerance','corpus','legacy_downgrade'])
def test_protocol2_forged_amendment_red(monkeypatch,tmp_path,attack):
    j,save=protocol_fixture(monkeypatch,tmp_path)
    if attack=='status':j['status']='APPROVED_BY_ME'
    elif attack=='tolerance':j['g0']['absolute_tolerance']=10.
    elif attack=='corpus':j['corpus']['config']='other'
    else:j['status']='RECOVERED_ORIGINAL'
    save()
    with pytest.raises(common.Red):common.protocol()


@pytest.mark.parametrize('attack',['registration','order_digest','order_bytes'])
def test_protocol2_stale_amendment_red(monkeypatch,tmp_path,attack):
    j,save=protocol_fixture(monkeypatch,tmp_path,allow_manifest_repin=True)
    if attack=='registration':j['registration_sha256']='0'*64
    elif attack=='order_digest':j['amendment_order_sha256']='0'*64
    else:
        stale=tmp_path/'order.md';stale.write_text('stale order')
        j['amendment_order_path']=str(stale)
        monkeypatch.setattr(common,'PROTOCOL2_ORDER',stale)
    save()
    with pytest.raises(common.Red):common.protocol()


@pytest.mark.parametrize('attack',['file_bytes','canonical_bytes'])
def test_protocol2_wrong_stream_sha_red(monkeypatch,tmp_path,attack):
    j,save=protocol_fixture(monkeypatch,tmp_path,allow_manifest_repin=True)
    ids=np.load(common.local_path(j['tokens_path']));ids[10]=(ids[10]+1)%73448
    wrong=tmp_path/'wrong.npy';np.save(wrong,ids,allow_pickle=False)
    j['tokens_path']=str(wrong)
    if attack=='canonical_bytes':j['tokens_sha256']=common.sha(wrong)
    save()
    with pytest.raises(common.Red,match='protocol pin failed: tokens_path|canonical token SHA mismatch'):
        common.protocol()


@pytest.mark.parametrize('attack',['short','dtype','rank','vocabulary','scoring'])
def test_protocol2_semantic_stream_and_scoring_guards_red(monkeypatch,tmp_path,attack):
    j,save=protocol_fixture(monkeypatch,tmp_path,allow_manifest_repin=True)
    ids=np.load(common.local_path(j['tokens_path']))
    if attack=='short':ids=ids[:32800];j['token_count']=len(ids)
    elif attack=='dtype':ids=ids.astype('>i8')
    elif attack=='rank':ids=ids[None]
    elif attack=='vocabulary':ids[0]=73448
    else:j['scoring']='last_511'
    wrong=tmp_path/'wrong.npy';np.save(wrong,ids,allow_pickle=False)
    j.update(tokens_path=str(wrong),tokens_sha256=common.sha(wrong),token_sha256=hashlib.sha256(ids.tobytes()).hexdigest())
    save()
    with pytest.raises(common.Red):common.protocol()


def test_protocol2_six_disjoint_full_prefills_pool_nll_not_ppl():
    seen=[]
    def forward(ids):
        seen.append(ids.copy())
        n=len(seen)
        return dict(targets=512,total_nll=n*512.,wall_ms=n,peak_resident_mib=n)
    ids=np.arange(32801,dtype='<i8')
    got=model.score_windows(SimpleNamespace(forward=forward),ids)
    assert len(seen)==6 and all(len(w)==1024 for w in seen)
    np.testing.assert_array_equal(np.concatenate(seen),ids[:6144])
    assert got['targets']==3072 and got['ppl']==pytest.approx(np.exp(3.5))
    assert got['wall_ms']==21 and got['peak_resident_mib']==6
    assert got['feeding']==common.FEEDING
    expected=np.concatenate([ids[w*1024+512:(w+1)*1024] for w in range(6)])
    assert got['target_sha256']==hashlib.sha256(expected.tobytes()).hexdigest()


@pytest.mark.parametrize('S',[8192,32768])
def test_protocol2_long_rows_start_at_zero_last512(S):
    ids=np.arange(33337,dtype='<i8')%3;seen=[]
    def forward(w):
        seen.append(w.copy())
        return dict(model.last512(np.zeros((S,3)),w),wall_ms=1,peak_resident_mib=2)
    got=model.score_windows(SimpleNamespace(forward=forward),ids,S)
    assert len(seen)==1
    np.testing.assert_array_equal(seen[0],ids[:S])
    assert got['targets']==512 and got['ppl']==pytest.approx(3.)


def test_protocol2_last512_fp64_matches_independent_logaddexp_oracle():
    ids=np.arange(1024)%7
    logits=np.random.default_rng(23).normal(size=(1024,7)).astype(np.float32)*100
    got=model.last512(logits,ids)
    x=logits[511:1023].astype(np.float64)
    want=np.sum(np.logaddexp.reduce(x,axis=1)-x[np.arange(512),ids[512:1024]],dtype=np.float64)
    assert got['total_nll']==pytest.approx(want,rel=1e-14)
    assert got['targets']==512


def repeat_fixture(monkeypatch):
    p,ids=common.protocol()
    target=np.concatenate([ids[w*1024+512:(w+1)*1024] for w in range(6)])
    jobs={}
    for i,n in enumerate(p['g0']['order'][:-1]):
        a=n.split('_')[1];rep=int(n[-1])
        jobs[n]=dict(process_identity=dict(pid=100+i,start_ticks=str(i)),
                     result=dict(arm=a,repeat=rep,ppl=12. if a=='A' else 13.,targets=3072,
                                 target_sha256=hashlib.sha256(target.tobytes()).hexdigest()))
    monkeypatch.setattr(driver,'require_pass',lambda n:jobs[n])
    return jobs


def test_g0_fresh_repeats_accept_and_b_minus_a_is_prediction_only(monkeypatch):
    repeat_fixture(monkeypatch)
    result=driver.g0_repeats()
    assert result['fresh_processes'] and result['B_minus_A']==1.
    assert result['B_minus_A_prediction_within_0_3'] is False
    assert result['B_minus_A_is_gate'] is False


@pytest.mark.parametrize('attack',['same_process','A_miss','B_miss','wrong_targets','missing_targets','nan'])
def test_g0_determinism_rail_rejects_bad_repeats(monkeypatch,attack):
    rr=repeat_fixture(monkeypatch)
    if attack=='same_process':rr['g0_A_2']['process_identity']=rr['g0_A_1']['process_identity']
    elif attack in ('A_miss','B_miss'):rr[f'g0_{attack[0]}_2']['result']['ppl']+=.0011
    elif attack=='wrong_targets':rr['g0_B_2']['result']['target_sha256']='0'*64
    elif attack=='missing_targets':rr['g0_B_2']['result']['targets']=3066
    else:rr['g0_A_2']['result']['ppl']=float('nan')
    with pytest.raises(common.Red,match='G0_DETERMINISM_MISS'):driver.g0_repeats()


def test_g0_registry_runs_fresh_arm_jobs_before_sp():
    cells=driver.cells();byid={c['id']:c for c in cells}
    assert byid['g0']['depends']==['g0_A_1','g0_B_1','g0_A_2','g0_B_2']
    seen=set()
    for c in cells:
        assert set(c['depends'])<=seen
        seen.add(c['id'])
    for a in 'AB':
        for i in (1,2):assert byid[f'g0_{a}_{i}']['kind']=='baseline'


@pytest.mark.parametrize('arm',list('ABCDE'))
@pytest.mark.parametrize('S',[1024,8192])
def test_protocol2_every_ppl_arm_uses_identical_feeding(monkeypatch,arm,S):
    seen=[]
    class Fake:
        def set(self,a,bits=4,delta=None,observe=False,**kw):self.observe=observe
        def forward(self,ids,score=True):
            if self.observe:
                return dict(refinement=dict(fraction=1. if arm=='D' else .1))
            seen.append(ids.copy())
            return dict(targets=512,total_nll=512*np.log(12.),wall_ms=1,peak_resident_mib=2)
    monkeypatch.setattr(model,'Model',Fake)
    control={a:dict(ppl=12.) for a in 'AB'}
    monkeypatch.setattr(driver,'g0_guard',lambda m,ids:control)
    def dep(n):
        if n=='g0':return dict(result=control)
        if n.startswith('ppl_'):return dict(result=dict(ppl=12.))
        return dict(result=dict(delta=1.,target_fraction=.1))
    monkeypatch.setattr(driver,'require_pass',dep)
    result=driver.execute(dict(kind='ppl',arm=arm,S=S,bits=4,depends=[]))
    _,ids=common.protocol()
    count=6 if S==1024 else 1
    assert len(seen)==count and result['targets']==count*512
    np.testing.assert_array_equal(np.concatenate(seen),ids[:count*S])
    assert result['feeding']==common.FEEDING


def test_stale_or_red_receipt_is_not_resumable(monkeypatch,tmp_path):
    monkeypatch.setattr(common,'ART',tmp_path)
    monkeypatch.setattr(common,'fingerprint',lambda:{'hash':'current'})
    (tmp_path/'jobs').mkdir()
    p=tmp_path/'jobs/x.json'
    p.write_text(json.dumps(dict(status='PASS',fingerprint={'hash':'stale'})))
    with pytest.raises(common.Red):common.require_pass('x')
    p.write_text(json.dumps(dict(status='RED',fingerprint={'hash':'current'})))
    with pytest.raises(common.Red):common.require_pass('x')


def test_create_only_atomic_receipts(monkeypatch,tmp_path):
    p=tmp_path/'one.json'
    common.publish(p,{'status':'RED'})
    original=p.read_bytes()
    with pytest.raises(FileExistsError):common.publish(p,{'status':'PASS'})
    assert p.read_bytes()==original
    assert not list(tmp_path.glob('*.partial'))


def test_all_cells_unique_and_dependencies_registered():
    cells=driver.cells();ids={c['id'] for c in cells}
    assert len(cells)==len(ids)
    assert len([c for c in cells if c['kind']=='margin' and c['bits']==4 and c['arm'] in 'BC'])==248
    for c in cells:
        assert set(c['depends'])<=ids
        expected_rail=290 if c['kind'] in ('capture_range','ceiling','decode_pool') else 480
        assert c['worker_timeout_s']==expected_rail and c['job_ceiling_s']==590
    for S in (2048,8192,32768):
        for arm in 'ABC':assert f'decode_b4_{arm}_{S}' in ids


def test_capture_metrics_all_pairs_causal_not_sampled(tmp_path):
    p=tmp_path/'layer00';p.mkdir()
    q=np.zeros((1,2,3,96),np.float32);q[...,0]=1
    k=np.zeros((1,2,4,96),np.float32);k[...,0]=np.arange(4)
    kq=k.copy();kq[...,0]+=.2
    mask=np.zeros((1,2,3,4),np.uint8)
    # Rectangular lengths 2,3,4: refine first key only, excludes future keys.
    mask[...,0]=1
    for n,x in [('q',q),('k',k),('kq',kq),('selected.pack',np.packbits(mask,axis=-1,bitorder='little'))]:
        np.save(p/(n+'.npy'),x)
    meta=dict(layer=0,arm='B',bits=4,delta=None,causal=True,pairs=18,selected=6,scale=1.,
              files={f.name:common.sha(f) for f in p.glob('*.npy')})
    common.publish(p/'capture.json',meta)
    result=metrics.analyze_capture(p,tmp_path/'scratch.f32',
        bulk_provider=lambda h,lo,hi,scale:(q[0,h,lo:hi].astype(np.float64)@kq[0,h].astype(np.float64).T)*scale)
    assert result['pairs']==18 and result['queries']==6
    assert result['fraction']==pytest.approx(1/3)
    assert result['error']['max']==pytest.approx(.2,abs=1e-6)
    assert result['max_skipped_relative_weight']==1.
    assert not (tmp_path/'scratch.f32').exists()


@pytest.mark.parametrize('mask_dtype',[np.uint8,np.int32])
def test_capture_blend_nonfloat_cat_preserves_packed_masks(tmp_path,mask_dtype):
    # Prior art: ordinary dependency stubs and independent byte/count oracles;
    # no prior art known to me for this specific regression fixture. Reproduce
    # the lead's 2026 uint8-cat failure through the real blend/capture methods.
    class Tensor:
        def __init__(self,data):
            self.data=np.asarray(data)
            self.shape=self.data.shape
            self.dtype=str(self.data.dtype)
        def numpy(self):return self.data
        def float(self):return Tensor(self.data.astype(np.float32))
        def slice(self,dim,start,length):
            index=[slice(None)]*self.data.ndim
            index[dim]=slice(start,start+length)
            return Tensor(self.data[tuple(index)])
        def transpose(self,a,b):return Tensor(self.data.swapaxes(a,b))
        def __mul__(self,scale):return Tensor(self.data*scale)

    cats=[]
    def cat(tensors,dim):
        if any(t.dtype not in ('float32','float16','bfloat16') for t in tensors):
            raise RuntimeError('op supports float32/float16/bfloat16 only')
        cats.append([t.dtype for t in tensors])
        return Tensor(np.concatenate([t.data for t in tensors],axis=dim))

    B,H,L,S,D=1,40,5,9,96
    q=Tensor(np.zeros((B,H,L,D),np.float32))
    k=Tensor(np.zeros((B,H,S,D),np.float32))
    kq=Tensor(np.zeros((B,H,S,D),np.float32))
    v=Tensor(np.zeros((B,H,S,64),np.float32))
    selected=np.zeros((B,H,L,S),mask_dtype)
    packed=np.zeros((B,H,L,2),np.uint8)
    expected_selected=0
    for h in range(H):
        for row in range(L):
            for j in range(S-L+row+1):
                if j%3==(h+row)%3:
                    selected[0,h,row,j]=1
                    packed[0,h,row,j//8] |= 1 << (j%8)
                    expected_selected+=1
    with pytest.raises(RuntimeError,match='op supports float32/float16/bfloat16 only'):
        cat([Tensor(selected[:,:,:2]),Tensor(selected[:,:,2:])],dim=2)

    diag_calls=[]
    def blend(bulk,rank,z,Lq,row0):
        length=bulk.shape[2]
        diag_calls.append((Lq,row0,length))
        assert z==1.25 and rank.shape==bulk.shape
        return Tensor(np.zeros_like(bulk.data)),Tensor(selected[:,:,row0:row0+length])

    native=Tensor(np.zeros((B,H,L,64),np.float32))
    obj=model.Model.__new__(model.Model)
    obj.tc=SimpleNamespace(cat=cat,matmul=lambda a,b:Tensor(a.data@b.data))
    obj.diag=SimpleNamespace(blend=blend)
    native_args=[]
    obj.original_blend=lambda *args:native_args.append(args) or native
    obj.observe=True;obj.capture_dir=tmp_path;obj.rows=[]
    obj.layer=0;obj.arm='B';obj.bits=4;obj.delta=None
    assert obj.blend(q,k,kq,v,1,96**-.5,1.25,True,2) is native
    assert all(a is b for a,b in zip(native_args[0][:4],(q,k,kq,v)))
    assert diag_calls==[(L,0,2),(L,2,2),(L,4,1)]
    assert len(cats)==2 and all(t=='float32' for chunk in cats for t in chunk)
    pairs=H*sum(range(S-L+1,S+1))
    assert obj.rows==[dict(layer=0,selected=expected_selected,pairs=pairs,
                           fraction=expected_selected/pairs,path='B_native_blend')]
    p=tmp_path/'layer00'
    got=np.load(p/'selected.pack.npy',allow_pickle=False)
    np.testing.assert_array_equal(got,packed)
    np.testing.assert_array_equal(np.unpackbits(got,axis=-1,count=S,bitorder='little'),selected)
    meta=common.read(p/'capture.json')
    assert meta['selected']==expected_selected and meta['pairs']==pairs
    assert meta['path']=='B_native_blend' and meta['causal'] is True
    assert [(c['row0'],c['length']) for c in meta['native_bulk_chunks']]==[(0,2),(2,2),(4,1)]
    for name,digest in meta['files'].items():assert common.sha(p/name)==digest


def test_no_device_receipt_is_observation_only():
    lib=ctypes.CDLL('/usr/local/cuda-12.6/lib64/libcudart.so')
    count=ctypes.c_int()
    code=lib.cudaGetDeviceCount(ctypes.byref(count))
    # Do not require no GPU on the lead seat; just validate the host API call.
    assert isinstance(code,int) and count.value>=0


def test_reporter_preserves_completed_decode_and_never_fills_missing_ppl(monkeypatch,tmp_path):
    import apa_sp3_report as report
    monkeypatch.setattr(report,'ART',tmp_path)
    # Constructed fixture only: lives in pytest temporary directory, never a
    # production result. In particular duplicate bits/S/arm keys must render.
    fixture={'decode_b4_C_2048':dict(status='PASS',result=dict(bits=4,S=2048,arm='C',tokens_s=12.,steps=32,prefill_s=2.))}
    monkeypatch.setattr(report,'receipts',lambda:fixture)
    report.main()
    j=json.loads((tmp_path/'results.json').read_text())
    assert any(x.get('tokens_s')==12. for x in j['decode'])
    assert all(x['ppl'] is None for x in j['ppl'])
    assert 'G2/G3 rows establish nothing about model quality by themselves.' in (tmp_path/'RESULTS.md').read_text()


def test_shell_preflight_blocks_unknown_without_lease(tmp_path):
    import os,subprocess
    fake=tmp_path/'bin';fake.mkdir();marker=tmp_path/'touched'
    for name in ('flock','nvidia-smi'):
        p=fake/name
        p.write_text('#!/bin/sh\ntouch '+str(marker)+'\nexit 99\n');p.chmod(0o755)
    env=dict(os.environ,PATH=str(fake)+':'+os.environ['PATH'])
    for job in ('invalid_cell','g0; touch forbidden'):
        ran=subprocess.run(['bash',str(ROOT/'scripts/apa_sp3_lead_gpu.sh'),'run',job],
                           env=env,capture_output=True,text=True,timeout=30)
        assert ran.returncode!=0
        assert not marker.exists(),ran.stdout+ran.stderr
