"""Author CPU baselines, never blind or native GPU verification.
Prior art: finite differences, independent dense VJP, NumPy/pytest, standard
softmax identities, Higham rounding model; taken. Ours: defect ablation and
fail-closed registration/calibration tests for this order.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pytest
import pt_tf32_2 as t
import pt_tf32_2_numerics as n
import pt_tf32_2_grapa as g
import pt_tf32_2_slot as slot


def spec(L=17,S=19,D=19,VD=17,**kwargs):
    return dict(B=1,H=4,KVH=2,L=L,S=S,D=D,VD=VD,causal=True,
                scale=float(np.float32(1/np.sqrt(D))),zthr=float(np.float32(1.036433389)),**kwargs)


@pytest.mark.parametrize('width',[1,7,8,9,19,64,96,127,128])
def test_residual_product_recovers_rounding_loss_and_ragged_width(width):
    rng=np.random.default_rng(width);a=rng.normal(size=(13,width)).astype(np.float32);b=rng.normal(size=(width,11)).astype(np.float32)
    ref=a.astype(np.float64)@b.astype(np.float64)
    old=n.product(a,b,'tf32');new=n.product(a,b,'tf32x3')
    assert np.linalg.norm(new-ref)<np.linalg.norm(old-ref)/100
    assert np.linalg.norm(new-ref)/np.linalg.norm(ref)<1e-6


def test_predicate_ablation_kills_the_old_dk_defect():
    s=spec(512,512,96,64,rope_width=32);x=t.fixture(s)
    f=n.forward_reference(x,s);state={k:v.astype(np.float32) for k,v in f.items() if k!='selection'}
    ref=n.backward_reference(x,state,s)
    # Hold singleton handling fixed here: only the predicate differs.
    before=n.backward_reference(x,state,s,score_mode='tf32',dov_mode='tf32',outer_mode='tf32')
    predicate=n.backward_reference(x,state,s,score_mode='tf32',dov_mode='tf32',outer_mode='tf32',selection_mode='fp64')
    after=n.backward_model(x,state,s)
    errors=[t.back.metric(a['dK'],ref['dK'])['relative_L2'] for a in (before,predicate,after)]
    bound=n.bounds(t.registration(),s,'dK')['bound']
    assert errors[0]>bound and errors[1]<bound and errors[2]<bound
    assert np.count_nonzero(before['selection']!=ref['selection'])>0
    assert np.array_equal(after['selection'],ref['selection'])


@pytest.mark.parametrize('widths',[(1,1),(19,17),(96,64),(128,127)])
@pytest.mark.parametrize('causal',[False,True])
def test_singleton_has_exact_output_and_zero_score_gradients(widths,causal):
    s=spec(1,1,*widths);s['causal']=causal;x=t.fixture(s)
    f=n.forward_reference(x,s);actual=n.backward_model(x,f,s)
    np.testing.assert_array_equal(f['out'],np.repeat(x['v'],2,axis=1))
    assert f['selection'].all()
    for key in ('dQ','dK'):np.testing.assert_array_equal(actual[key],0.)
    expected=t.old.tf32_round(x['dO']).astype(np.float64).reshape(1,2,2,1,widths[1]).sum(2)
    np.testing.assert_array_equal(actual['dV'],expected)


def frozen_loss(x,s,mask):
    result=0.
    for h in range(s['H']):
        kh=h//(s['H']//s['KVH']);q=x['q'][0,h];k=x['k'][0,kh];kq=x['kq'][0,kh]
        bulk=q@kq.T*s['scale'];score=np.where(mask[0,h],q@k.T*s['scale'],bulk)
        visible=np.arange(s['S'])[None,:]<=s['S']-s['L']+np.arange(s['L'])[:,None]
        score=np.where(visible,score,-np.inf);p=np.exp(score-score.max(1)[:,None]);p/=p.sum(1)[:,None]
        result+=float(np.sum((p@x['v'][0,kh])*x['dO'][0,h]))
    return result


@pytest.mark.parametrize('shape',[(3,5,3,2),(17,19,7,9)])
def test_independent_vjp_against_finite_differences_with_frozen_mask(shape):
    s=spec(*shape);x={k:v.astype(np.float64) for k,v in t.fixture(s).items()}
    f=n.forward_reference(x,s);gradient=n.backward_reference(x,f,s)
    # Independent scalar loss, hold selection and detached KQ fixed.
    for key,grad in zip(('q','k','v'),t.NAMES):
        for flat in (0,x[key].size//3,x[key].size-1):
            at=np.unravel_index(flat,x[key].shape);value=x[key][at];eps=1e-5
            x[key][at]=value+eps;plus=frozen_loss(x,s,f['selection'])
            x[key][at]=value-eps;minus=frozen_loss(x,s,f['selection']);x[key][at]=value
            np.testing.assert_allclose((plus-minus)/(2*eps),gradient[grad][at],rtol=2e-5,atol=1e-8)


@pytest.mark.parametrize('shape',[(15,17,19,17),(16,16,96,64),(17,31,19,17),(33,35,96,64)])
@pytest.mark.parametrize('causal',[False,True])
def test_padding_grouped_reference_matches_independent_owner_traversal(shape,causal):
    s=spec(*shape);s['causal']=causal;s['zthr']=-100.;x=t.fixture(s);x['kq']=x['k'].copy()
    f=n.forward_reference(x,s)
    a=n.backward_reference(x,f,s)
    b=t.old.tiled_backward_model(x,f,s,rounded=False)
    for key in t.NAMES:np.testing.assert_allclose(a[key],b[key],rtol=1e-10,atol=2e-12)


def test_bounds_are_precision_derived_and_fail_closed():
    r=t.registration()
    for s in r['attention_shapes']:
        for name in ('out','lse',*t.NAMES):
            b=n.bounds(r,s,name)
            assert 1e-4<=b['expected']<=1e-3 and b['bound']==3*b['expected']
    assert n.metric_gate([0.,0.],[0.,0.],.003)['verdict']=='GREEN'
    for a in ([1e-40,0.],[np.inf,0.],[np.nan,0.]):
        assert n.metric_gate(a,[0.,0.],.003)['verdict']=='RED'
    assert n.metric_gate([1.,1.01],[1.,1.],.001)['verdict']=='RED'
    with pytest.raises(ValueError):n.metric_gate([],[],.001)
    with pytest.raises(ValueError):n.metric_gate([1.],[1.,2.],.001)


def test_noise_floor_includes_deterministic_policy_bias_and_six_pairs():
    base={'q':np.array([1.,0.])};policy={'q':np.array([np.cos(.03),np.sin(.03)])}
    floor=n.noise_floor([base,policy,policy,base])
    assert len(floor['pairs'])==6
    np.testing.assert_allclose(floor['observed_spread'],1-np.cos(.03),atol=2e-16)
    assert floor['bar']==1-3*floor['observed_spread']
    assert n.noise_floor([base]*4)['bar']==1.


@pytest.mark.parametrize('bad',[{}, {'q':np.array([0.,0.])},{'q':np.array([np.nan,1.])}, {'other':np.array([1.,0.])}])
def test_noise_floor_rejects_missing_zero_nonfinite_or_mismatched_gradients(bad):
    base={'q':np.array([1.,0.])}
    with pytest.raises(ValueError):n.noise_floor([base,base,bad,base])


def test_noise_floor_rejects_invalid_count_and_vacuous_bar():
    base={'q':np.array([1.,0.])};opposite={'q':np.array([-1.,0.])}
    with pytest.raises(ValueError):n.noise_floor([base]*3)
    with pytest.raises(ValueError):n.noise_floor([base,opposite,base,opposite])


def test_slot_sequence_is_one_slot_and_sanitizers_cannot_pass_vacuously():
    rows=slot.sequence(t.ART/'lead_slot')
    assert sum(r[1] for r in rows)<=1800
    assert [r[0] for r in rows]==t.registration()['slot']['sequence']
    for name in ('memcheck','racecheck','synccheck'):
        assert not slot.sanitizer_clean(name,'')
        assert not slot.sanitizer_clean(name,'ERROR SUMMARY: 1 errors')
        assert slot.sanitizer_clean(name,'ERROR SUMMARY: 0 errors')


def test_slot_deadline_stops_only_its_own_child(tmp_path):
    result=slot.run_lane([sys.executable,'-B','-c','while True: pass'],
                         tmp_path/'timeout.log',dict(os.environ,CUDA_VISIBLE_DEVICES=''),.1)
    assert result['status']=='BLOCKED_TIMEOUT' and result['returncode']!=0
    assert result['elapsed_seconds']<5


def test_frozen_noise_receipt_detects_input_and_bar_tampering(tmp_path,monkeypatch):
    monkeypatch.setattr(g,'ART',tmp_path)
    (tmp_path/'SOURCE_MANIFEST.json').write_text('{}')
    arrays=[{'q':np.array([1.,0.])},{'q':np.array([1.,.02])}]*2
    inputs=[]
    for i,arm in enumerate(('none','0-10','0-10','none')):
        path=tmp_path/f'grad_{i}.npz';np.savez(path,**arrays[i])
        inputs.append(dict(arm=arm,path=str(path),sha256=t.sha(path)))
    floor=dict(n.noise_floor(arrays),inputs=inputs,registration_sha256=t.sha(t.REG),
               manifest_sha256=t.sha(tmp_path/'SOURCE_MANIFEST.json'))
    path=tmp_path/'NOISE_FLOOR.json';t.create_json(path,floor)
    path.with_suffix('.sha256').write_text(t.sha(path)+'\n')
    assert g.checked_floor(path)['bar']==floor['bar']
    floor['bar']-=.01;path.write_text(json.dumps(floor))
    path.with_suffix('.sha256').write_text(t.sha(path)+'\n')
    with pytest.raises(ValueError,match='computation drift'):g.checked_floor(path)
    np.savez(inputs[0]['path'],q=np.array([1.,.01]))
    with pytest.raises(ValueError,match='input drift'):g.checked_floor(path)


def test_new_native_cpu_contract_and_getter():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    c=t.load_fork(False)
    assert c.get_tf32_gemm_info()==(-1,0,0)
    c.set_tf32_gemm(True);assert c.get_tf32_gemm() is True;c.set_tf32_gemm(False)
    p=subprocess.run([str(t.ART/'pt_tf32_host_contract')],cwd=t.ROOT,capture_output=True,text=True,timeout=20)
    assert p.returncode==0,p.stderr
    assert p.stdout.strip()=='HOST_CONTRACT: 6 expected rejections; no CUDA device operations'


def test_new_grapa_registration_preserves_state_and_uses_fork():
    r=g.registered();old=json.loads((t.old.ART/'GRAPA_REGISTRATION.json').read_text())
    for key in ('states','timing','batches_npz','trainer_sources_sha256'):assert r[key]==old[key]
    assert Path(r['engine']['so_path']).resolve().is_relative_to(t.ROOT)
    assert r['pt_tf32_2']['gates']['onset_cos_min']==.99
    assert r['pt_tf32_2']['gates']['bf16_ratio_max']==1.3
    assert r['pt_tf32_2']['gates']['healthy_control_cos_min'] is None


@pytest.mark.parametrize('script,args',[
    ('pt_tf32_2.py',['attention','--lead-gpu']),
    ('pt_tf32_2_grapa.py',['noise','--lead-gpu']),
    ('pt_tf32_2_slot.py',['--lead-gpu','--out','artifacts/pt_tf32_2/never_run'])])
def test_no_gpu_command_can_run_with_hidden_cuda(script,args):
    p=subprocess.run([sys.executable,'-B','scripts/'+script,*args],cwd=t.ROOT,
                     env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=20)
    assert p.returncode==2 and 'BLOCKED' in p.stderr
