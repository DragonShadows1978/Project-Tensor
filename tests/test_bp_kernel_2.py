"""CPU author baseline only, not a blind review or CUDA validation.
Prior art: pytest (Krekel 2004), NumPy (Harris et al. 2020), softmax Jacobian
and FlashAttention (Dao et al. 2022), taken; ours: adversarial frozen APA
fixtures and registered-gate checks. Unverified — lead to check these names.
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
import bp_kernel_2 as k
import bp_census_2 as c


def dense_oracle(x,state,s):
    # Independent scalar score/dov construction and explicit dense Jacobian.
    # Prior art: standard softmax Jacobian diag(p)-p p^T, taken (author/year
    # no prior art known to me for first derivation); ours: frozen APA fixture.
    out=[np.zeros_like(x[n],dtype=np.float64) for n in ('q','k','v')]
    selection=np.zeros((s['B'],s['H'],s['L'],s['S']),bool)
    scale=float(np.float32(s['scale']))
    for b in range(s['B']):
        for h in range(s['H']):
            kh=h//(s['H']//s['KVH'])
            for i in range(s['L']):
                p=np.zeros(s['S']); keys=[]; dov=[]
                for j in range(s['S']):
                    q=x['q'][b,h,i].astype(float); kj=x['k'][b,kh,j].astype(float); kq=x['kq'][b,kh,j].astype(float)
                    bulk=sum(float(q[d])*float(kq[d]) for d in range(s['D']))*scale
                    visible=not s['causal'] or j<=s['S']-s['L']+i
                    sel=visible and abs(bulk)>=state['thr'][b,h,i]
                    selection[b,h,i,j]=sel; key=kj if sel else kq; keys.append(key)
                    score=sum(q[d]*key[d] for d in range(s['D']))*scale
                    if visible: p[j]=np.exp(score-float(state['lse'][b,h,i]))
                    dov.append(sum(float(x['dO'][b,h,i,d])*float(x['v'][b,kh,j,d]) for d in range(s['VD'])))
                ds=(np.diag(p)-np.outer(p,p))@np.array(dov)
                for j in range(s['S']):
                    for d in range(s['D']):
                        out[0][b,h,i,d]+=ds[j]*scale*keys[j][d]
                        if selection[b,h,i,j]: out[1][b,kh,j,d]+=ds[j]*scale*float(x['q'][b,h,i,d])
                    for d in range(s['VD']): out[2][b,kh,j,d]+=p[j]*float(x['dO'][b,h,i,d])
    return out,selection


@pytest.mark.parametrize('causal',[False,True])
@pytest.mark.parametrize('lengths',[(3,3),(3,5)])
@pytest.mark.parametrize('threshold',[0.,.5,100.])
def test_fp64_reference_dense_oracle(causal,lengths,threshold):
    s=dict(k.old.TINY,B=2,H=4,KVH=2,L=lengths[0],S=lengths[1],causal=causal)
    x=k.old.seed_inputs(s)
    state=dict(lse=np.full((2,4,s['L']),2.3,np.float32),thr=np.full((2,4,s['L']),threshold,np.float32),out=np.zeros_like(x['dO']))
    actual=k.reference(x,state,s); expected,mask=dense_oracle(x,state,s)
    for name,value in zip(k.NAMES,expected): np.testing.assert_allclose(actual[name],value,rtol=2e-13,atol=2e-13)
    np.testing.assert_array_equal(actual['selection'],mask)
    assert all(actual[n].dtype==np.float64 for n in k.NAMES)


def test_selection_equality_negative_and_causal_boundary():
    s=dict(k.old.TINY,B=1,H=1,KVH=1,L=2,S=3,D=1,VD=1,scale=1.)
    x=dict(q=np.ones((1,1,2,1)),k=np.full((1,1,3,1),3.),kq=np.array([[[[-1.],[1.],[.5]]]]),v=np.ones((1,1,3,1)),dO=np.ones((1,1,2,1)))
    state=dict(lse=np.full((1,1,2),4.),thr=np.array([[[1.,np.nextafter(1.,2.)]]]),out=np.zeros_like(x['dO']))
    ref=k.reference(x,state,s)
    np.testing.assert_array_equal(ref['selection'][0,0],[[True,True,False],[False,False,False]])
    expected,_=dense_oracle(x,state,s)
    for n,v in zip(k.NAMES,expected): np.testing.assert_allclose(ref[n],v,rtol=1e-13,atol=1e-13)


def test_frozen_lse_not_renormalized_and_unselected_dk_zero():
    s=dict(k.old.TINY,H=1,KVH=1,L=1,S=1,D=1,VD=1,scale=1.)
    x={n:np.ones((1,1,1,1)) for n in ('q','k','kq','v','dO')}
    state=dict(lse=np.full((1,1,1),2.),thr=np.full((1,1,1),100.),out=np.zeros_like(x['dO']))
    r=k.reference(x,state,s);p=np.exp(-1.)
    assert r['dK'].item()==0
    assert r['dV'].item()==pytest.approx(p)
    assert r['dQ'].item()==pytest.approx(p*(1-p))


def test_gate_fabricated_boundary_and_better_than_a():
    ref=tuple(np.ones(2) for _ in range(3)); a=tuple(v+.25 for v in ref)
    result=k.gate(a,ref,dict(equal=tuple(v+.5 for v in ref),over=tuple(v+.50001 for v in ref),truth=ref))
    assert result['candidates']['equal']['verdict']=='GREEN'
    assert result['candidates']['over']['verdict']=='RED'
    assert result['candidates']['truth']['verdict']=='GREEN'
    assert result['tolerance']['dQ']['max_abs']==.5


def test_zero_tolerance_no_epsilon_and_nonfinite():
    a=tuple(np.ones(1) for _ in range(3))
    result=k.gate(a,a,dict(exact=a,tiny=tuple(v+1e-14 for v in a),bad=tuple(v*np.nan for v in a)))
    assert len(result['zero_tolerances'])==6
    assert result['candidates']['exact']['verdict']=='GREEN'
    assert result['candidates']['tiny']['verdict']=='RED'
    assert result['candidates']['bad']['verdict']=='RED'
    zero=tuple(np.zeros(1) for _ in range(3))
    assert not k.gate(a,zero,dict(x=a))['control_finite']
    assert k.gate(zero,zero,dict(x=zero))['candidates']['x']['verdict']=='GREEN'
    with pytest.raises(ValueError): k.gate(a,a,dict(x=(np.ones(2),)*3))


def test_variant_a_pinned_bytes_and_default_body():
    reg=json.loads((k.ROOT/'artifacts/bp_kernel_1/registration.json').read_text());pin=reg['variant_a_region']
    src=(k.ROOT/'tensor_cuda/src/kernels.cu').read_bytes()
    before=(k.ART/'baseline/kernels.cu').read_bytes()
    assert hashlib.sha256(src[pin['start_byte']:pin['end_byte_exclusive']]).hexdigest()==pin['sha256']
    # Full original kernel/launcher section, including bytes before pinned range.
    assert src[:before.index(b'// BP-KERNEL-1 opt-in variants;')]==before[:before.index(b'// BP-KERNEL-1 opt-in variants;')]
    ops=(k.ROOT/'tensor_cuda/src/ops.cpp').read_text();old=(k.ART/'baseline/ops.cpp').read_text()
    begin=old.index('  return Tensor::from_op(out, {q, k, v}, "apa_selective_train",')
    end=old.index('// ------------------------------------------------------------- reductions',begin)
    assert old[begin:end] in ops


def test_f_has_key_ownership_and_reuses_c_without_atomics():
    src=(k.ROOT/'tensor_cuda/src/kernels.cu').read_text()
    block=src[src.index('__global__ void apa_selective_bwd_bk2_kv_kernel'):src.index('// ----------------------------------------------- APA blend+softmax')]
    assert 'atomicAdd(' not in block
    assert 'apa_selective_bwd_bk1_kernel<T,DMAX,false,DOT>' in block
    assert '<<<B*KVH*S,128>>>' in block
    assert 'group*count' in block


def test_registration_and_fail_closed(monkeypatch):
    r=k.verify_registration()
    assert r['input']['sha256'].startswith('53c38919')
    assert r['budget']['cells']==2
    original=k.sha
    monkeypatch.setattr(k,'sha',lambda p:'0'*64 if Path(p)==k.ROOT/'scripts/bp_kernel_2.py' else original(p))
    with pytest.raises(ValueError,match='drift'): k.verify_registration()


def test_create_only(tmp_path):
    p=tmp_path/'receipt.json';k.create_json(p,dict(a=1));before=p.read_bytes()
    with pytest.raises(FileExistsError): k.create_json(p,dict(a=2))
    assert p.read_bytes()==before
    p=tmp_path/'reference.npz';k.save_npz(p,dict(a=np.ones(2)))
    with pytest.raises(FileExistsError): k.save_npz(p,dict(a=np.zeros(2)))


def test_schema_rejects_incomplete_or_cpu_claim():
    r=k.empty_receipt(True);k.validate_receipt(r)
    r['verdict']='CONFIRMED'
    with pytest.raises(ValueError): k.validate_receipt(r)
    r=k.empty_receipt(True);r['schema_version']=2
    with pytest.raises(ValueError): k.validate_receipt(r)


def test_census_registration_and_fixed_protocol():
    r=c.verify()
    assert r['config']['arm_order']==['a','f']
    assert r['config']['warmup_steps']==2 and r['config']['measured_steps']==5
    assert not r['off_control']['repeated']
    assert r['tokens_sha256']==k.sha(c.PARENT_ART/'tokens.npy')


def test_census_red_gate_timing_only_and_incomplete():
    reg=json.loads(c.REG.read_text())
    gate=dict(eligible=False,route='f_pass_a',label='TIMING-ONLY / NOT A VALID STEP')
    events=[]
    for v in ('a','f'):
        for i in range(7):
            events.append(dict(kind='step',arm=v,index=i,warmup=i<2,wall_ms=11800 if v=='a' else 4000,
                components={n:dict(ms=1.) for n in c.parent.COMPONENTS}))
    r=c.summarize(events,reg,gate,False,1.)
    assert r['verdict']=='TIMING_ONLY_NOT_A_VALID_STEP'
    gate['eligible']=True
    assert c.summarize(events,reg,gate,False,1.)['verdict']=='CONFIRMED'
    assert c.summarize(events[:-1],reg,gate,False,1.)['verdict']=='INCONCLUSIVE'
