"""Author checks, not blind verification or native GPU evidence.

Prior art: pytest/NumPy, Higham (2002) interval/product analysis, NVIDIA
TF32 (2020)/Lt (2024), Gumbel (1958)/David & Nagaraja (2003), taken.
Ours: reject injected tail/dispatch/edge defects and bind the new registration.
"""
import json
import math
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pytest
import pt_tf32_4 as t
import pt_tf32_4_grapa as g
import pt_tf32_4_slot as slot
from pt_tf32_4_numerics import edge_dv_budget, edge_dv_gate, probability_interval


def spec(L=17,S=31,D=96,VD=64,causal=True):
    return dict(B=1,H=4,KVH=2,L=L,S=S,D=D,VD=VD,causal=causal,
                scale=float(np.float32(1/np.sqrt(D))),zthr=-100.)


def test_registration_preserves_every_unamended_lane():
    r=t.registration();old=json.loads((t.ROOT/r['prior_registration']).read_text())
    for key in ('seed','gemm_shapes','attention_shapes','attention_gate','model_gate','storage_gate'):
        assert r[key]==old[key]
    assert r['rounding_model']['operand_rounding_events']==old['rounding_model']['operand_rounding_events']
    assert r['gemm_gate']['relative_L2_max']==old['gemm_gate']['relative_L2_max']==.001
    assert 'min_speedup' not in r['gemm_gate']
    assert r['edge_accumulator']['unit_roundoff']==2**-23
    for name in ('out','lse','dQ','dK','dV'):
        assert t.num.previous.bounds(r,r['attention_shapes'][0],name)==t.num.previous.bounds(old,r['attention_shapes'][0],name)


def test_registered_tail_family_risk_and_exact_shape_table():
    r=t.registration();expected_counts=[3145728,4194304,6291456,8388608]
    for s,row,N in zip(r['attention_shapes'],r['dk_tail']['shape_table'],expected_counts):
        tail=t.num.dk_tail(r,s)
        assert tail['elements']==row['elements']==N
        for key in row:
            if key!='case':assert row[key]==tail[key]
        eps=tail['expected_rel_L2'];z=tail['normalized_max_bound']/eps
        assert math.isclose(2*N*math.exp(-z*z/2),.001/12,rel_tol=2e-14)
        assert tail['normalized_max_margin']>0
    assert len(r['attention_shapes'])*3==r['dk_tail']['family_comparisons']


def tail_case():
    s=dict(B=1,H=1,KVH=1,L=1,S=4096,D=1,VD=1)
    return s,np.ones((1,1,4096,1),np.float64)


def test_tail_spike_and_l2_are_independent_mandatory_gates():
    s,ref=tail_case();r=t.registration();bar=t.num.dk_tail(r,s)
    allowed=ref.copy();allowed.flat[0]+=bar['normalized_max_bound']*.95
    assert t.num.precision_gate(allowed,ref,r,s,'dK')['verdict']=='GREEN'
    spike=ref.copy();spike.flat[0]+=bar['normalized_max_bound']*1.05
    a=t.num.precision_gate(spike,ref,r,s,'dK')
    assert a['verdict']=='RED' and a['relative_L2']<a['relative_L2_bound']
    biased=ref+bar['relative_L2_bound']*1.05
    a=t.num.precision_gate(biased,ref,r,s,'dK')
    assert a['verdict']=='RED' and a['normalized_max_abs']<a['normalized_max_bound']


@pytest.mark.parametrize('factor',[1e-4,1.,1e4])
def test_tail_normalization_scales_with_reference_peak(factor):
    s,ref=tail_case();ref*=factor;candidate=ref.copy();candidate.flat[0]+=.002*factor
    a=t.num.precision_gate(candidate,ref,t.registration(),s,'dK')
    assert a['verdict']=='GREEN'
    assert a['sigma_envelope_abs']==a['expected']*factor
    assert a['normalized_max_abs']==pytest.approx(.002)
    assert a['max_witness']['index']==[0,0,0,0]


@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf,1e-30])
def test_zero_reference_and_nonfinite_fail_closed(bad):
    s,ref=tail_case();ref.fill(0);candidate=ref.copy();candidate.flat[0]=bad
    assert t.num.precision_gate(candidate,ref,t.registration(),s,'dK')['verdict']=='RED'
    assert t.num.precision_gate(ref,ref,t.registration(),s,'dK')['verdict']=='GREEN'


def test_dk_wrong_shape_rejected_and_other_max_gates_unchanged():
    s,ref=tail_case()
    with pytest.raises(ValueError):t.num.precision_gate(ref[:,:,:-1],ref[:,:,:-1],t.registration(),s,'dK')
    candidate=ref.copy();candidate.flat[0]+=.004
    assert t.num.precision_gate(candidate,ref,t.registration(),s,'dQ')['verdict']=='RED'


@pytest.mark.parametrize('causal',[False,True])
@pytest.mark.parametrize('lengths',[(1,1),(15,17),(16,16),(17,31),(33,35)])
@pytest.mark.parametrize('widths',[(1,1),(19,17),(96,64),(128,127)])
def test_edge_interval_covers_fp32_sum_and_rejects_missing_head(causal,lengths,widths):
    s=spec(*lengths,*widths,causal=causal);x=t.fixture(s);x['kq']=x['k'].copy()
    f=t.num.forward_reference(x,s)
    state={n:f[n].astype(np.float32) for n in ('out','lse','thr')}
    model=edge_dv_budget(x,state,s,t.registration())
    ref=t.num.backward_model(x,state,s)['dV']
    np.testing.assert_allclose(model['reference'],ref,rtol=2e-14,atol=2e-14)
    assert edge_dv_gate(ref.astype(np.float32),model)['verdict']=='GREEN'
    # A missing grouped head cannot be justified by probability quantization.
    defect=ref*.5
    assert edge_dv_gate(defect,model)['verdict']=='RED'
    assert edge_dv_gate(np.full_like(ref,np.nan),model)['verdict']=='RED'


def test_midpoint_interval_encloses_one_bin_and_stays_tight_off_midpoint():
    visible=np.ones((1,2),bool);q=np.ones((1,1),np.float64)
    # For p=0.0625 + half a TF32 bin, opposite FP32 rounding directions
    # select adjacent bins. No historical observed error is used here.
    midpoint=.0625+2**-15
    k=np.array([[math.log(midpoint)],[math.log(.125)]],np.float64)
    center,low,high,delta,_=probability_interval(q,k,np.zeros(1),1.,visible,2**-24,1.173)
    assert high[0,0]-low[0,0]==2**-14
    assert delta[0,0]==2**-14 and delta[0,1]==0
    assert low[0,0]<=center[0,0]<=high[0,0]


def test_coordinatewise_edge_gate_does_not_use_large_neighbor_budget():
    s=spec();x=t.fixture(s);x['kq']=x['k'].copy()
    f=t.num.forward_reference(x,s);state={n:f[n].astype(np.float32) for n in ('out','lse','thr')}
    model=edge_dv_budget(x,state,s,t.registration());ref=model['reference'];bound=model['absolute_tolerance']
    assert model['crossing_pairs']>0
    allowed=ref+bound*.9
    assert edge_dv_gate(allowed,model)['verdict']=='GREEN'
    at=np.unravel_index(int(np.argmin(bound)),bound.shape)
    defect=ref.copy();defect[at]+=bound[at]*2+1e-12
    assert edge_dv_gate(defect,model)['verdict']=='RED'
    for factor in (1e-3,-1.,1e3):
        xx=dict(x,dO=(x['dO']*factor).astype(np.float32))
        m=edge_dv_budget(xx,state,s,t.registration())
        assert edge_dv_gate(m['reference'].astype(np.float32),m)['verdict']=='GREEN'


def test_edge_domain_is_fail_closed():
    s=spec();x=t.fixture(s);x['kq']=x['k'].copy();f=t.num.forward_reference(x,s)
    state={n:f[n].astype(np.float32) for n in ('out','lse','thr')}
    bad=dict(x,kq=x['kq']+1)
    with pytest.raises(ValueError):edge_dv_budget(bad,state,s,t.registration())
    with pytest.raises(ValueError):edge_dv_budget(x,dict(state,thr=np.ones_like(state['thr'])),s,t.registration())


def valid_dispatch():
    return dict(M=4096,N=1792,K=1024,batch=1,compute_type=77,fast_tf32=1,transa=1,transb=0,
        lda=1024,ldb=1024,ldc=1792,device_selected=1,algorithm_id=21,numerical_flags=0x40202,
        numerical_flags_query_status=0,numerical_flags_bytes=8,numerical_flags_attribute=15,
        mathmode_query_attempted=0,mathmode_query_supported=0,alignment_a=16,alignment_b=16,alignment_c=16,workspace_bytes=0)


@pytest.mark.parametrize('change',[
    {'numerical_flags':0x40201},{'numerical_flags':0x20202},{'device_selected':0},
    {'compute_type':68},{'M':1792},{'transa':0},{'transb':1},{'alignment_a':4},
    {'numerical_flags_query_status':7},{'numerical_flags_bytes':4},{'algorithm_id':-1},
    {'numerical_flags_attribute':8},{'mathmode_query_attempted':1}])
def test_actual_dispatch_rejects_bad_flags_status_geometry_and_alignment(change):
    row=valid_dispatch()
    assert t.dispatch_ok(row,4096,1792,1024,True)
    row.update(change)
    assert not t.dispatch_ok(row,4096,1792,1024,True)
    assert not t.dispatch_ok({},4096,1792,1024,True)


def test_gemm_gate_is_flags_and_accuracy_even_with_slow_timing():
    row=valid_dispatch();row['speedup']=.25
    assert t.gemm_row_ok(dict(finite=True,relative_L2=.001),row,4096,1792,1024,True)
    assert not t.gemm_row_ok(dict(finite=True,relative_L2=.001001),row,4096,1792,1024,True)
    assert not t.gemm_row_ok(dict(finite=True,relative_L2=float('nan')),row,4096,1792,1024,True)


def test_cpu_host_descriptor_query_and_contract():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    c=t.load_fork(False)
    row=c.tf32_gemm_self_check(4096,1792,1024,True,False)
    assert t.dispatch_ok(row,4096,1792,1024,True,False)
    assert row['mathmode_query_attempted']==0 and row['numerical_flags_query_status']==-1
    assert row['numerical_flags_attribute']==15 and row['numerical_flags_bytes']==0
    r=subprocess.run([str(t.ART/'pt_tf32_host_contract')],capture_output=True,text=True,timeout=10)
    assert r.returncode==0 and r.stdout.strip()=='HOST_CONTRACT: 6 expected rejections; no CUDA device operations'


def test_new_model_registration_exact_inputs_and_fork_binary():
    r=g.registered();old=json.loads((t.ROOT/'artifacts/pt_tf32_2/GRAPA_REGISTRATION_002.json').read_text())
    for key in ('states','timing','batches_npz','trainer_sources_sha256'):assert r[key]==old[key]
    assert Path(r['engine']['so_path']).resolve().is_relative_to(t.ROOT)


def test_slot_deadline_and_default_zero_dump_both_sequences():
    for model in (False,True):
        rows=slot.sequence(t.ART/'never_run',include_model=model)
        assert sum(r[1] for r in rows)==(1035 if model else 475)
        assert sum(r[1] for r in rows)<1200
        assert [r[0] for r in rows[:10]]==t.registration()['slot']['sequence']
        assert not any('--keep-grads' in r[2] for r in rows)
        for _,_,argv in rows:
            for a in argv:
                if a.startswith('scripts/') or a.startswith('tests/'):assert (t.ROOT/a).is_file()
    assert slot.dump_estimate(False)['total_dump_bytes_upper_estimate']==0


@pytest.mark.parametrize('script,args',[
    ('pt_tf32_4.py',['dispatch','--lead-gpu']),('pt_tf32_4.py',['gemm','--lead-gpu']),
    ('pt_tf32_4.py',['attention','--lead-gpu']),('pt_tf32_4_grapa.py',['noise','--lead-gpu']),
    ('pt_tf32_4_slot.py',['--lead-gpu','--out','artifacts/pt_tf32_4/never_run'])])
def test_hidden_cuda_blocks_all_gpu_entrypoints(script,args):
    r=subprocess.run([sys.executable,'-B','scripts/'+script,*args],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=10)
    assert r.returncode==2 and 'BLOCKED' in r.stderr


def test_sanitizer_verdict_is_independent_but_never_vacuous():
    for name in ('memcheck','racecheck','synccheck'):
        assert slot.sanitizer_result(name,'ERROR SUMMARY: 0 errors')['status']=='BLOCKED'
        assert slot.sanitizer_result(name,'2 failed, 58 passed\nERROR SUMMARY: 0 errors')['status']=='GREEN'
        assert slot.sanitizer_result(name,'60 passed\nERROR SUMMARY: 1 errors')['status']=='RED'


def test_final_manifest_matches_new_build():
    assert t.verify_manifest()['registration_sha256']==t.sha(t.REG)
