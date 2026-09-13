"""CPU author-run tests, not blind review or CUDA validation.
Prior art: pytest (Krekel et al., 2004; date unverified — lead to check),
boundary/adversarial testing; ours: registered BP-KERNEL-1 invariants and fixtures.
"""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import os
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('bk1',ROOT/'scripts/bp_kernel_1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def triple(x): return tuple(np.asarray(x,dtype=np.float32) for _ in range(3))

def good_gate():
    return dict(control_finite=True,candidates={v:dict(verdict='GREEN') for v in 'bd'})

def times(a=10.,b=7.,c=5.,d=6.): return {v:[t]*10 for v,t in zip('abcd',(a,b,c,d))}


def test_registration_and_immutability(tmp_path):
    r=m.verify_registration()
    assert r['shape']==m.SHAPE
    assert r['shape']['H']==r['shape']['KVH']==16
    assert r['shape']['D']==96 and r['shape']['VD']==64
    assert r['warmups']==3 and r['samples']==10
    path=tmp_path/'registration.json';m.create_json(path,r)
    before=path.read_bytes()
    with pytest.raises(FileExistsError):m.create_json(path,{'changed':True})
    assert path.read_bytes()==before
    path.with_suffix('.sha256').write_text(m.sha(path))
    r['samples']=9;path.write_text(json.dumps(r))
    with pytest.raises(RuntimeError,match='registration bytes drift'):m.verify_registration(path)
    path.with_suffix('.sha256').write_text(m.sha(path))
    with pytest.raises(RuntimeError,match='protocol drift'):m.verify_registration(path)


def test_source_drift_closed(tmp_path):
    r=json.loads(m.REG.read_text());path=tmp_path/'registration.json'
    r['sources']['kernels.cu']['after_sha256']='0'*64
    m.create_json(path,r);path.with_suffix('.sha256').write_text(m.sha(path))
    with pytest.raises(RuntimeError,match='source drift'):m.verify_registration(path)


def test_default_regions_unchanged_vs_pinned_main():
    pins=json.loads((m.ART/'baseline_pins.json').read_text())
    pin=pins['variant_a_region']
    before=(m.ART/'baseline/kernels.cu').read_bytes()
    after=(ROOT/'tensor_cuda/src/kernels.cu').read_bytes()
    a,b=pin['start_byte'],pin['end_byte_exclusive']
    import hashlib
    assert hashlib.sha256(before[a:b]).hexdigest()==pin['sha256']
    assert after[a:b]==before[a:b]
    # Entire prefix through original forward + backward launchers unchanged.
    end=before.index(b'// ----------------------------------------------- APA blend+softmax (post-matmul)')
    assert after[:end]==before[:end]
    # Original full ops file is a byte-identical prefix (only appended wrapper).
    ops=(m.ART/'baseline/ops.cpp').read_bytes()
    assert (ROOT/'tensor_cuda/src/ops.cpp').read_bytes().startswith(ops)
    bind=(m.ART/'baseline/bindings.cpp').read_bytes()
    start=bind.index(b'  // APA selective backward: returns (dq, dk, dv).')
    end=bind.index(b'  // Fused APA blend+softmax',start)
    assert bind[start:end] in (ROOT/'tensor_cuda/src/bindings.cpp').read_bytes()
    for name in ('kernels.cu','ops.cpp','bindings.cpp'):
        assert pins[name]['main_live_equal']
        assert m.sha(m.ART/'baseline'/name)==pins[name]['sha256']


def test_tolerance_twice_spread_boundary():
    a1=triple([1.]);a2=triple([1.25])
    r=m.gate(a1,a2,dict(b=triple([1.5]),d=triple([1.5001])))
    assert r['control_finite']
    assert r['tolerance']['dQ']==dict(max_abs=.5,relative_L2=.5)
    assert r['candidates']['b']['verdict']=='GREEN'
    assert r['candidates']['d']['verdict']=='RED'


def test_one_bad_gradient_red_and_relative_error_reported():
    a1=triple([1.,2.]);a2=triple([1.1,2.])
    b=list(a1);b[2]=np.array([9.,2.])
    r=m.gate(a1,a2,dict(b=b))
    assert r['candidates']['b']['verdict']=='RED'
    assert r['spreads']['dQ']['relative_L2']>0


def test_zero_spread_has_no_epsilon_floor():
    a=triple([1.])
    r=m.gate(a,a,dict(b=triple([1.000001]),d=a))
    assert r['tolerance']['dQ']['max_abs']==0
    assert r['candidates']['b']['verdict']=='RED'
    assert r['candidates']['d']['verdict']=='GREEN'


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-float('inf')])
def test_nonfinite_control_and_candidate(bad):
    a=triple([1.]);x=triple([bad])
    r=m.gate(a,x,dict(b=a,d=a))
    assert not r['control_finite'] and r['candidates']['b']['verdict']=='RED'
    assert m.decide(times(),r)['primary']=='INCONCLUSIVE'
    assert m.gate(a,a,dict(b=x))['candidates']['b']['verdict']=='RED'
    json.dumps(r,allow_nan=False)


def test_zero_norm_relative_rule():
    assert m.metric([0.],[0.])['relative_L2']==0
    assert not m.metric([1.],[0.])['finite']


def test_verdict_exact_boundaries():
    assert m.decide(times(),good_gate())==dict(primary='ATOMICS_GE_50',atomic_fraction=.5,secondary='CONFIRMED')
    assert m.decide(times(c=5.0001),good_gate())['primary']=='SCALAR_DOTS_PROMOTED'
    assert m.decide(times(d=6.0001),good_gate())['secondary']=='FALSIFIED'
    r=good_gate();r['candidates']['d']['verdict']='RED'
    assert m.decide(times(),r)['secondary']=='NOT_APPLICABLE'


@pytest.mark.parametrize('kind',['short','nan','zero','negative','incomplete'])
def test_incomplete_or_bad_timing_inconclusive(kind):
    t=times()
    if kind=='short':t['b'].pop()
    elif kind=='nan':t['c'][0]=float('nan')
    elif kind=='zero':t['d'][0]=0.
    elif kind=='negative':t['a'][0]=-1.
    assert m.decide(t,good_gate(),complete=kind!='incomplete')['primary']=='INCONCLUSIVE'


def test_create_only_receipt_and_npz(tmp_path):
    p=tmp_path/'receipt.json';m.create_json(p,dict(verdict='INCONCLUSIVE'))
    with pytest.raises(FileExistsError):m.create_json(p,{})
    assert json.loads(p.read_text())==dict(verdict='INCONCLUSIVE')
    p=tmp_path/'inputs.npz';m.save_npz(p,dict(q=np.array([1.])))
    with pytest.raises(FileExistsError):m.save_npz(p,dict(q=np.array([2.])))


def test_cpu_shared_protocol_and_schema(tmp_path):
    arrays=m.seed_inputs(m.TINY)
    r=m.empty_receipt('dry-run',m.TINY,m.sha(m.REG))
    class GuardCPU(m.CPUBackend):
        calls=[]
        def backward(self,variant):
            assert (tmp_path/'forward_state_registration.json').exists()
            self.calls.append(variant)
            return super().backward(variant)
    backend=GuardCPU(arrays,m.TINY)
    import time
    m.experiment(backend,r,tmp_path,time.monotonic()+30)
    m.validate_receipt(r)
    assert backend.calls[:4]==list('aabd')
    assert backend.calls[4:]==list('abcd')*13
    assert [x['variant'] for x in r['launch_order']]==list('abcd')*13
    assert r['verdict']=='DRY_RUN'
    assert all(not t['timing_counts'] for t in r['timings'].values())
    assert r['forward_state']['sha256']==m.sha(tmp_path/'forward_state.npz')
    r['timings']['b']['timing_counts']=True
    with pytest.raises(ValueError):m.validate_receipt(r)


def test_dry_run_subprocess_precedence_and_schema(tmp_path):
    out=tmp_path/'dry'
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
    result=subprocess.run([sys.executable,str(ROOT/'scripts/bp_kernel_1.py'),'--run','--dry-run',
        '--output-dir',str(out)],env=env,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip().endswith(str(out/'receipt.json'))
    r=json.loads((out/'receipt.json').read_text());m.validate_receipt(r)
    assert r['mode']=='dry-run' and r['evidence_class']=='CPU simulation'
    result=subprocess.run([sys.executable,str(ROOT/'scripts/bp_kernel_1.py'),'--dry-run',
        '--output-dir',str(out)],env=env,capture_output=True,text=True,timeout=30)
    assert result.returncode!=0 and 'FileExistsError' in result.stderr


def test_seeded_bf16_fixture():
    a=m.seed_inputs(m.TINY);b=m.seed_inputs(m.TINY)
    for k in a:
        assert np.array_equal(a[k],b[k])
        assert ((a[k].view(np.uint32)&0xffff)==0).all()
    assert a['q'].shape[-1]!=a['v'].shape[-1]
    assert m.bf16(np.array([1.00390625],np.float32))[0]==1.0


def test_budget_expiry_before_forward(tmp_path):
    r=m.empty_receipt('dry-run',m.TINY,'x')
    with pytest.raises(TimeoutError):m.experiment(m.CPUBackend(m.seed_inputs(m.TINY),m.TINY),r,tmp_path,0)
    assert not (tmp_path/'forward_state_registration.json').exists()


def test_no_vacuous_gradients():
    with pytest.raises(ValueError,match='exactly dQ/dK/dV'):
        m.gate((),(),dict(b=(),d=()))
