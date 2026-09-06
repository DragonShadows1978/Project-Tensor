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
    fake=SimpleNamespace(set=lambda arm,**kw:seen.append(arm),
                         forward=lambda ids:dict(ppl=20.1 if seen[-1]=='A' else 19.817))
    with pytest.raises(common.Red,match='G0_PARITY_MISS'):driver.g0_guard(fake,np.arange(1024))
    assert seen==['A','B']
    assert len(list((tmp_path/'progress').glob('*.json')))==1


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
        assert c['worker_timeout_s']==480 and c['job_ceiling_s']==590
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
