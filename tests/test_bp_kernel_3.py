"""CPU author baseline, never a native receipt or a blind verification.
Prior art: pytest (Krekel 2004), NumPy (Harris et al. 2020), dense softmax VJP
(FlashAttention / Dao et al. 2022), taken. Ours: APA ownership boundary fixtures
and registered-verdict adversarial checks. Unverified — lead to check names.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pytest
import bp_kernel_3 as k
import bp_census_3 as c


def oracle(x,state,s):
    # Independent scalar pair traversal, output-dot identity (Dao et al. 2022).
    # Taken: dense VJP; ours: adversarial fixed APA state. Unverified lead.
    out={n:np.zeros_like(x[v],dtype=float) for n,v in zip(k.NAMES,('q','k','v'))}
    mask=np.zeros((s['B'],s['H'],s['L'],s['S']),bool)
    scale=float(np.float32(s['scale']))
    for b in range(s['B']):
        for h in range(s['H']):
            kh=h//(s['H']//s['KVH'])
            for i in range(s['L']):
                q=x['q'][b,h,i].astype(float);do=x['dO'][b,h,i].astype(float)
                rd=sum(do*state['out'][b,h,i])
                for j in range(s['S']):
                    if s['causal'] and j>s['S']-s['L']+i: continue
                    key=x['k'][b,kh,j].astype(float);kq=x['kq'][b,kh,j].astype(float)
                    bulk=sum(q*kq)*scale;sel=abs(bulk)>=state['thr'][b,h,i]
                    mask[b,h,i,j]=sel;key=key if sel else kq
                    p=np.exp(sum(q*key)*scale-float(state['lse'][b,h,i]))
                    ds=p*(sum(do*x['v'][b,kh,j])-rd)*scale
                    out['dQ'][b,h,i]+=ds*key
                    if sel: out['dK'][b,kh,j]+=ds*q
                    out['dV'][b,kh,j]+=p*do
    return out,mask


@pytest.mark.parametrize('causal',[False,True])
@pytest.mark.parametrize('lengths',[(1,1),(15,17),(16,16),(17,31),(33,35)])
@pytest.mark.parametrize('threshold',[0.,.25,100.])
def test_owner_tiles_dense_oracle(causal,lengths,threshold):
    s=dict(k.parent.old.TINY,H=2,KVH=1,L=lengths[0],S=lengths[1],D=19,VD=17,causal=causal)
    x=k.parent.old.seed_inputs(s)
    state=dict(lse=np.full((1,2,s['L']),3.),thr=np.full((1,2,s['L']),threshold),out=x['dO']*.1)
    expected,mask=oracle(x,state,s)
    for v in k.G:
        actual,_=k.tiled_backward(x,state,s,v)
        for n in k.NAMES: np.testing.assert_allclose(actual[n],expected[n],rtol=2e-12,atol=2e-12)
        np.testing.assert_array_equal(actual['selection'],mask)
    ref=k.reference(x,state,s,output_dot=True)
    for n in k.NAMES: np.testing.assert_allclose(expected[n],ref[n],rtol=2e-12,atol=2e-12)


def test_selection_equality_negative_causal_and_width_tail():
    s=dict(k.parent.old.TINY,B=2,H=4,KVH=2,L=17,S=19,D=1,VD=1,scale=1.)
    x={n:np.ones((2,4 if n in ('q','dO') else 2,17 if n in ('q','dO') else 19,1)) for n in ('q','k','kq','v','dO')}
    x['kq'][:]=-1.;x['k'][:]=2.
    state=dict(thr=np.ones((2,4,17)),lse=np.full((2,4,17),4.),out=np.full_like(x['dO'],.25))
    expected,mask=oracle(x,state,s)
    for v in k.G:
        actual,_=k.tiled_backward(x,state,s,v)
        assert actual['selection'][0,0,0,2] and not actual['selection'][0,0,0,3]
        assert mask.sum()>0
        for n in k.NAMES: np.testing.assert_allclose(actual[n],expected[n],rtol=1e-13,atol=1e-13)


@pytest.mark.parametrize('width',[16,96,128])
def test_width_fragments_and_bf16_route(width):
    s=dict(k.parent.old.TINY,H=1,KVH=1,L=3,S=5,D=width,VD=width)
    x=k.parent.old.seed_inputs(s);state=dict(thr=np.zeros((1,1,3)),lse=np.full((1,1,3),4.),out=x['dO']*.1)
    a,_=k.tiled_backward(x,state,s,'g1',round_mma=True)
    b,_=k.tiled_backward(x,state,s,'g2',round_mma=True)
    for n in k.NAMES: np.testing.assert_array_equal(a[n],b[n])


def test_gate_boundaries_zero_nonfinite_shape():
    ref=(np.ones(2),)*3;a=tuple(x+.25 for x in ref)
    g=k.gate(a,ref,dict(g1=tuple(x+.5 for x in ref),g2=tuple(x+.50001 for x in ref)))
    assert g['candidates']['g1']['verdict']=='GREEN';assert g['candidates']['g2']['verdict']=='RED'
    g=k.gate(ref,ref,dict(g1=ref,g2=tuple(x+1e-14 for x in ref)))
    assert len(g['zero_tolerances'])==6 and g['candidates']['g2']['verdict']=='RED'
    for bad in (np.nan,np.inf):
        assert k.gate(a,ref,dict(g1=tuple(x*bad for x in ref)))['candidates']['g1']['verdict']=='RED'
        assert not k.gate(tuple(x*bad for x in ref),ref,dict(g1=ref))['control_finite']
    with pytest.raises(ValueError): k.gate(a,ref,dict(g1=(np.ones(3),)*3))


def fabricated_gate(g1=True,g2=True,f=True):
    return dict(control_finite=True,candidates={v:dict(verdict='GREEN' if ok else 'RED') for v,ok in [('f',f),('g1',g1),('g2',g2)]})


def test_decision_registered_threshold_and_red():
    t={v:dict(mean_ms=x) for v,x in zip(k.VARIANTS,(410,100,35,36))}
    assert k.decision(fabricated_gate(),t,False)==('g1','g1 ≤ g2: selection pays in this implementation','CONFIRMED')
    t['g1']['mean_ms']=35.00001
    assert k.decision(fabricated_gate(),t,False)[2]=='PREDICTION_FALSIFIED'
    t['g2']['mean_ms']=30
    best,secondary,verdict=k.decision(fabricated_gate(),t,False)
    assert best=='g2' and secondary.startswith('g2 < g1') and verdict=='CONFIRMED'
    assert k.decision(fabricated_gate(False,False),t,False)==(None,None,'G_RED')
    assert k.decision(fabricated_gate(f=False),t,False)[2]=='F_RED'
    assert k.decision(fabricated_gate(),t,True)[2]=='DRY_RUN'


def test_variant_a_bytes_and_default_body_unchanged():
    r=json.loads((k.ROOT/'artifacts/bp_kernel_1/registration.json').read_text());p=r['variant_a_region']
    src=(k.ROOT/'tensor_cuda/src/kernels.cu').read_bytes();before=(k.ART/'baseline/kernels.cu').read_bytes()
    assert hashlib.sha256(src[p['start_byte']:p['end_byte_exclusive']]).hexdigest()==p['sha256']
    offset=before.index(b'// BP-KERNEL-1 opt-in variants;')
    assert src[:offset]==before[:offset]
    old=(k.ART/'baseline/ops.cpp').read_text();new=(k.ROOT/'tensor_cuda/src/ops.cpp').read_text()
    start=old.index('  return Tensor::from_op(out, {q, k, v}, "apa_selective_train",')
    end=old.index('// ------------------------------------------------------------- reductions',start)
    assert old[start:end] in new


def test_native_ownership_and_mma_sites():
    s=(k.ROOT/'tensor_cuda/src/kernels.cu').read_text();s=s[s.index('// BP-KERNEL-3 g1/g2.'):s.index('// ----------------------------------------------- APA blend+softmax')]
    assert 'atomicAdd' not in s and 'wmma::mma_sync' in s
    assert 'if constexpr(DENSE)' in s and 'if(selected)' in s
    assert 'scores(qs,kqs,bulk,D)' in s and 'scores(dos,vs,dov,VD)' in s
    assert 'outer(sel,KEY?qs:ks' in s and 'outer(prob,dos' in s
    assert 'if(q.dtype!=DType::BFloat16' in s


def test_create_only(tmp_path):
    p=tmp_path/'example.json';k.create_json(p,dict(value=1));old=p.read_bytes()
    with pytest.raises(FileExistsError): k.create_json(p,dict(value=2))
    assert p.read_bytes()==old
    p=tmp_path/'fixture.npz';k.parent.save_npz(p,dict(x=np.ones(2)))
    with pytest.raises(FileExistsError): k.parent.save_npz(p,dict(x=np.zeros(2)))


def test_registration_fail_closed_on_synthetic_pins(tmp_path,monkeypatch):
    # Synthetic registration proves mechanics without binding this test to a
    # particular local GPU binary or treating the test as campaign evidence.
    monkeypatch.setattr(k,'ROOT',tmp_path);reg=tmp_path/'registration.json';monkeypatch.setattr(k,'REG',reg)
    path=tmp_path/'tensor_cuda/src/kernels.cu';path.parent.mkdir(parents=True);path.write_bytes(b'fixture')
    binary=tmp_path/'binary';binary.write_bytes(b'engine')
    r=dict(k.protocol(),pins={'binary':k.sha(binary)},sources={},variant_a_region=dict(start_byte=0,end_byte_exclusive=7,sha256=k.sha(path)),engine_binary=dict(path='binary',sha256=k.sha(binary)))
    k.create_json(reg,r);reg.with_suffix('.sha256').write_text(k.sha(reg))
    assert k.verify_registration()['g2_built']
    binary.write_bytes(b'drift')
    with pytest.raises(ValueError,match='drift'): k.verify_registration()


def receipt_fixture(monkeypatch):
    monkeypatch.setattr(k,'sha',lambda p:'0'*64)
    monkeypatch.setattr(k,'registration_chain',lambda:({},'0'*64))
    r=k.empty_receipt(True);r['correctness']=fabricated_gate()
    for v in k.VARIANTS:
        r['timings'][v]=dict(samples_ms=[1.]*10,mean_ms=1.,gate_green=True,timing_counts=False)
    for v in ('f',*k.G):
        r['half_timings'][v]={h:dict(samples_ms=[.5]*10,mean_ms=.5,timing_counts=False) for h in ('query_owned','key_owned')}
    for key,vs in [('launch_order',k.VARIANTS),('half_launch_order',('f',*k.G))]:
        r[key]=[dict(phase=p,round=i,variant=v) for p,n in [('warmup',3),('measured',10)] for i in range(n) for v in vs]
    r['best_green'],r['secondary'],r['verdict']=k.decision(r['correctness'],r['timings'],True)
    return r


@pytest.mark.parametrize('defect',['schema','missing_half','cpu_claim','red_counts','mean','samples','order','best'])
def test_schema_rejects_mutated_receipts(monkeypatch,defect):
    r=receipt_fixture(monkeypatch);k.validate_receipt(r)
    if defect=='schema': r['schema_version']=2
    elif defect=='missing_half': del r['half_timings']['g1']
    elif defect=='cpu_claim': r['verdict']='CONFIRMED'
    elif defect=='red_counts': r['timings']['g1']['timing_counts']=True
    elif defect=='mean': r['timings']['g1']['mean_ms']=0
    elif defect=='samples': r['half_timings']['g2']['query_owned']['samples_ms'].pop()
    elif defect=='order': r['launch_order'].reverse()
    elif defect=='best': r['best_green']='g2'
    with pytest.raises(ValueError): k.validate_receipt(r)


def census_fixture(monkeypatch,fms=5000,gms=2500):
    monkeypatch.setattr(c,'sha',lambda p:'0'*64)
    reg=dict(checkpoint={},tokens_sha256='0'*64,off_control={})
    events=[dict(kind='step',arm=v,index=i,warmup=i<2,wall_ms=ms,components={n:dict(ms=1.) for n in c.parent.COMPONENTS}) for v,ms in [('f',fms),('g',gms)] for i in range(7)]
    return reg,events


@pytest.mark.parametrize('fms,gms,eligible,verdict',[(5000,2500,True,'CONFIRMED'),(5000,2500.01,True,'STEP_GT_2_5S'),(4500,2400,True,'CONFIRMED'),(5500,2400,True,'CONFIRMED'),(5500.1,2000,True,'F_REPRODUCTION_FAILED'),(5000,2000,False,'TIMING_ONLY_NOT_A_VALID_STEP')])
def test_census_decisions(monkeypatch,fms,gms,eligible,verdict):
    reg,events=census_fixture(monkeypatch,fms,gms);g=dict(eligible=eligible,route='g1')
    assert c.summarize(events,reg,g,False,1.)['verdict']==verdict
    assert c.summarize(events[:-1],reg,g,False,1.)['verdict']=='INCONCLUSIVE'
    assert c.summarize(events,reg,g,False,1.,'timeout')['verdict']=='INCONCLUSIVE'
    assert c.summarize(events,reg,g,True,1.)['verdict']=='DRY_RUN'


def test_canonical_paths_and_missing_gate(monkeypatch,tmp_path):
    assert c.GRAPA==Path('/mnt/ForgeRealm/GRAPA-Native-LLM')
    assert Path(c.parent.__file__)==c.GRAPA/'scripts/bp_census_1.py'
    assert c.CONFIG['arm_order']==['f','g']
    assert c.CONFIG['warmup_steps']==2 and c.CONFIG['measured_steps']==5
    monkeypatch.setattr(k,'ART',tmp_path)
    assert c.gate_source(False)['route']=='g1' and not c.gate_source(False)['eligible']
    assert 'TIMING-ONLY' in c.gate_source(False)['label']


def test_amendments_preserve_protocol_and_detect_broken_chain(tmp_path,monkeypatch):
    monkeypatch.setattr(k,'REG',tmp_path/'registration.json')
    base=dict(k.protocol(),pins={},sources={})
    k.create_json(k.REG,base);k.REG.with_suffix('.sha256').write_text(k.sha(k.REG))
    a=dict(number=1,previous_sha256=k.sha(k.REG),reason='synthetic',pins={'x':'1'*64},sources={},engine_binary=None)
    p=tmp_path/'registration_amendment_001.json';k.create_json(p,a);p.with_suffix('.sha256').write_text(k.sha(p))
    actual,chain=k.registration_chain()
    assert actual['prediction']==k.PREDICTION and actual['pins']['x']=='1'*64 and chain==k.sha(p)
    a['prediction']='weakened'
    p.write_text(json.dumps(a));p.with_suffix('.sha256').write_text(k.sha(p))
    with pytest.raises(ValueError,match='amendment chain'): k.registration_chain()


def test_census_keeps_best_green_when_f_is_red(monkeypatch,tmp_path):
    monkeypatch.setattr(k,'ART',tmp_path)
    monkeypatch.setattr(c,'sha',lambda p:'0'*64)
    monkeypatch.setattr(k,'registration_chain',lambda:({},'0'*64))
    monkeypatch.setattr(k,'validate_receipt',lambda r:r)
    r=dict(registration_sha256='0'*64,registration_chain_sha256='0'*64,
           reference_sha256='0'*64,mode='run',best_green='g2',verdict='F_RED')
    (tmp_path/'receipt.json').write_text(json.dumps(r))
    result=c.gate_source(False)
    assert result['route']=='g2' and not result['eligible']
    assert 'TIMING-ONLY' in result['label']
