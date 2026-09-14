"""CPU author tests, not blind verification or CUDA parity.
Prior art: pytest (Krekel 2004), NumPy (Harris 2020), stable softmax
(standard numerical analysis), taken. Ours: APA boundary/oracle/gate cases.
Unverified — lead to check those titles/authors; no network.
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
import bp_kernel_4 as k
import bp_census_4 as c


def oracle(x,s):
    # Independent scalar dense oracle. Prior art: stable dense softmax,
    # taken; APA shipped population-variance z threshold, taken; fixtures ours.
    B,H,L,S,VD=[s[n] for n in ('B','H','L','S','VD')]
    out=np.zeros((B,H,L,VD));lse=np.zeros((B,H,L));thr=lse.copy();sel=np.zeros((B,H,L,S),bool)
    scale=float(np.float32(s['scale']));z=float(np.float32(s['zthr']))
    for b in range(B):
        for h in range(H):
            kh=h//(H//s['KVH'])
            for i in range(L):
                count=S-L+i+1 if s['causal'] else S
                q=x['q'][b,h,i].astype(float)
                bulk=np.array([sum(q*x['kq'][b,kh,j].astype(float))*scale for j in range(count)])
                ab=abs(bulk);mean=sum(ab)/count
                # Same population variance identity, independently scalar.
                th=mean+z*np.sqrt(max(sum(t*t for t in ab)/count-mean*mean,0.))
                selected=ab>=th
                scores=np.array([sum(q*x['k'][b,kh,j].astype(float))*scale if selected[j] else bulk[j] for j in range(count)])
                ex=np.exp(scores-max(scores));den=sum(ex)
                for j in range(count):out[b,h,i]+=ex[j]/den*x['v'][b,kh,j].astype(float)
                lse[b,h,i]=max(scores)+np.log(den);thr[b,h,i]=th;sel[b,h,i,:count]=selected
    return dict(out=out,lse=lse,thr=thr,selection=sel)


@pytest.mark.parametrize('causal',[True,False])
@pytest.mark.parametrize('lengths',[(1,1),(15,17),(16,16),(17,31),(33,35)])
@pytest.mark.parametrize('zthr',[-1.,0.,1.0364333894937898,20.])
def test_forward_dense_oracle(causal,lengths,zthr):
    s=dict(k.parent.old.TINY,B=1,H=2,KVH=1,L=lengths[0],S=lengths[1],D=19,VD=17,causal=causal,zthr=zthr)
    x=k.parent.old.seed_inputs(s);expected=oracle(x,s)
    for chunk in (16,128):
        actual=k.reference(x,s,chunk)
        for n in ('out','lse','thr'):np.testing.assert_allclose(actual[n],expected[n],rtol=2e-12,atol=2e-12)
        np.testing.assert_array_equal(actual['selection'],expected['selection'])


def test_percentile_contract_is_not_order_statistic():
    s=dict(k.parent.old.TINY,B=1,H=1,KVH=1,L=1,S=4,D=1,VD=1,scale=1.,zthr=1.,causal=False)
    x=dict(q=np.ones((1,1,1,1)),kq=np.array([0.,1.,2.,9.]).reshape(1,1,4,1),k=np.ones((1,1,4,1)),v=np.ones((1,1,4,1)))
    r=k.reference(x,s)
    assert r['thr'].item()==pytest.approx(3+np.sqrt(12.5))
    assert r['selection'].flatten().tolist()==[False,False,False,True]
    x['kq'][:]=-2.;r=k.reference(x,s)
    assert r['thr'].item()==2 and r['selection'].all() # abs and >=, zero variance


def fixture():
    ref=dict(out=np.ones(4),lse=np.ones(4),thr=np.ones(4),selection=np.zeros(200,bool))
    a=copy.deepcopy(ref);a['out']+=.25;a['lse']+=.25
    return ref,a


def test_forward_gate_and_flip_exact_boundaries():
    ref,a=fixture();h=copy.deepcopy(ref);h['out']+=.5;h['lse']+=.5;h['selection'][0]=True
    assert k.forward_gate(a,h,ref)['verdict']=='GREEN'
    h['selection'][1]=True
    assert k.forward_gate(a,h,ref)['verdict']=='RED'
    h=copy.deepcopy(ref);h['out']+=.500001
    assert k.forward_gate(a,h,ref)['verdict']=='RED'
    h=copy.deepcopy(ref);h['lse']+=1e-14
    assert k.forward_gate(ref,h,ref)['verdict']=='RED'
    for field in ('out','lse','thr'):
        for bad in (np.nan,np.inf):
            h=copy.deepcopy(ref);h[field][0]=bad
            assert k.forward_gate(a,h,ref)['verdict']=='RED'
    with pytest.raises(ValueError):k.flips(np.zeros(2,bool),np.zeros(3,bool))
    with pytest.raises(ValueError):k.flips(np.zeros(2),np.zeros(2))
    with pytest.raises(ValueError):k.flips(np.ones(2,bool),np.ones(2,bool),np.zeros(2,bool))


def test_downstream_red_stops_timing():
    b=k.CPU();b.downstream=lambda v:tuple(np.full_like(b.backref[n],1e10) for n in k.parent.NAMES)
    r=k.empty_receipt(True);k.experiment(b,r,lambda e:None)
    assert not r['correctness']['green'] and r['timings']=={} and r['phase_diagnostics'] is None
    k.validate_receipt(r)


def test_receipt_adversarial_schema():
    r=k.empty_receipt(True);k.experiment(k.CPU(),r,lambda e:None);k.validate_receipt(r)
    assert r['correctness']['green']
    for mutate in (lambda x:x.update(extra=1),lambda x:x['launch_order'].pop(),lambda x:x['timings']['h'].update(timing_counts=True),lambda x:x.update(verdict='CONFIRMED'),lambda x:x['timings']['a']['samples_ms'].pop()):
        bad=copy.deepcopy(r);mutate(bad)
        with pytest.raises(ValueError):k.validate_receipt(bad)


def test_original_regions_and_backward_switch():
    reg=json.loads(k.REG.read_text());data=(k.ROOT/'tensor_cuda/src/kernels.cu').read_bytes()
    for name,p in reg['regions'].items():assert hashlib.sha256(data[p['start_byte']:p['end_byte_exclusive']]).hexdigest()==p['sha256'],name
    baseline=(k.ART/'baseline/ops.cpp').read_text();current=(k.ROOT/'tensor_cuda/src/ops.cpp').read_text()
    start=baseline.index('static thread_local std::string bk2_variant')
    end=baseline.index('namespace ops {',start)
    assert baseline[start:end] in current
    start=baseline.index('  if (tc::bp_kernel_2_get_variant() != "a")')
    end=baseline.index('// ------------------------------------------------------------- reductions',start)
    assert baseline[start:end] in current
    src=(k.ROOT/'tensor_cuda/src/kernels.cu').read_text()
    assert src.count('BP-KERNEL-4 BEGIN')==1 and 'template<bool DIAG>' in src


def test_registration_pins_and_protocol():
    r=k.verify_registration()
    assert r['h2_built'] is False and r['flip_ceiling']==.005
    assert r['budget']==dict(cells=2,work_seconds=300,lease_seconds=590,lock='/tmp/forge-gpu.lock',cards=1)
    for key,value in k.protocol().items():assert r[key]==value
    assert c.verify()['config']['arm_order']==['a','h']


def test_create_only(tmp_path):
    p=tmp_path/'receipt.json';k.create_json(p,dict(a=1));before=p.read_bytes()
    with pytest.raises(FileExistsError):k.create_json(p,dict(a=2))
    assert p.read_bytes()==before
    p=tmp_path/'reference.npz';k.parent.save_npz(p,dict(x=np.ones(2)))
    with pytest.raises(FileExistsError):k.parent.save_npz(p,dict(x=np.ones(3)))


def test_census_maps_native_forward_without_double_counting():
    # Prior art: exclusive profiler accounting/gprof 1982, taken.
    rows=[dict(name='initial_forward',exclusive_ms=30.),dict(name='bk4_apa_initial',exclusive_ms=70.),dict(name='checkpoint_replay',exclusive_ms=20.),dict(name='bk4_apa_replay',exclusive_ms=80.)]
    mapped=[dict(r,name={'bk4_apa_initial':'initial_forward','bk4_apa_replay':'checkpoint_replay'}.get(r['name'],r['name'])) for r in rows]
    table,_,_=c.parent.component_table(mapped,200.)
    assert table['initial_forward']['ms']==100 and table['checkpoint_replay']['ms']==100
