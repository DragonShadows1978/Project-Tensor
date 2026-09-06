"""A4 author CPU gates; not GPU validation or independent blind review.

Prior art: constructed-input testing and mutation testing (DeMillo, Lipton,
Sayward 1978, unverified lead: Hints on Test Data Selection). New SP3 fixtures.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
import apa_sp3_common as common
import apa_sp3_gpu as driver
import apa_sp3_a4_registry as registry
import apa_sp3_a4_capture as capture
import apa_sp3_a4_jobs as jobs
import apa_sp3_a4_provenance as provenance
import apa_sp3_a4_torch as reference
from apa_sp3_model import last512, score_windows


def test_a4_registered_dag_ranges_bounds_and_grid():
    cells = [c for c in driver.cells() if c['kind'] != 'decode_pool']
    base = driver.base_cells()
    assert len(cells) == len(base)+167 == 859
    by = {c['id']: c for c in cells}
    assert len(by) == len(cells)
    seen = set()
    for c in cells:
        assert set(c['depends']) <= seen
        seen.add(c['id'])
        if c['kind'] in ('capture_range', 'ceiling'):
            assert c['estimate_s'][1] < 300
            assert c['worker_timeout_s']+5 < 300
    for c in cells:
        if c['kind'] != 'capture_aggregate':
            continue
        ranges = [by[d] for d in c['depends']]
        assert [i for r in ranges for i in range(r['layer_start'], r['layer_stop'])] == list(range(62))
        assert len(ranges) == (4 if c['S'] == 8192 else 62)
        for previous, current in zip(ranges, ranges[1:]):
            assert previous['id'] in current['depends']
        for r in ranges:
            assert r['estimate_s'][1] == 60+8*(r['S']//8192)**2*(r['layer_stop']-r['layer_start'])
    for arm in 'ABC':
        assert [c['S'] for c in cells if c['kind']=='ceiling' and c['arm']==arm] == [4096,8192,16384,24576,32768]
    assert 'ppl_T_32768' in by['ppl_b4_D_32768']['depends']
    assert 'ppl_b4_A_32768' not in by['ppl_b4_D_32768']['depends']
    for arm in 'BC':
        assert f'ceiling_b4_{arm}_32768' in by[f'ppl_b4_{arm}_32768']['depends']
        assert 'ppl_T_32768' not in by[f'ppl_b4_{arm}_32768']['depends']
    for old in base:
        if old['kind'] == 'margin':
            assert old == by[old['id']]


def test_compact_reference_oracle_six_windows_and_long_rows():
    rng = np.random.default_rng(49)
    ids = np.arange(33337, dtype='<i8')%7
    for S in (1024,8192,32768):
        seen = []
        def forward(window):
            seen.append(window.copy())
            full = rng.normal(size=(len(window),7)).astype(np.float32)
            compact = reference.compact_nll(full[-513:-1], window)
            assert compact == last512(full, window)
            return dict(compact,wall_ms=1,peak_resident_mib=2)
        r = score_windows(SimpleNamespace(forward=forward), ids, S)
        count = 6 if S==1024 else 1
        assert len(seen) == count and r['targets'] == count*512
        assert np.array_equal(np.concatenate(seen),ids[:S*count])
        targets = np.concatenate([w[-512:] for w in seen])
        assert r['target_sha256'] == hashlib.sha256(targets.astype('<i8').tobytes()).hexdigest()
    with pytest.raises(common.Red):
        reference.compact_nll(np.zeros((511,7)),ids)


def test_checkpoint_continuation_matches_full_layer_sequence(tmp_path):
    seen = []
    def layer(i):
        def call(h, cos, sin, offset, cache):
            assert offset == 0 and cache is None
            seen.append(i)
            return np.asarray(h*.75+i, dtype=np.float32), object()
        return call
    model = SimpleNamespace(layers=[layer(i) for i in range(62)],rope_cos='cos',rope_sin='sin')
    x = np.arange(51,dtype=np.float32).reshape(1,17,3)
    expected = capture.advance_layers(model,x,0,62)
    seen.clear()
    h = x
    for lo,hi in [(0,16),(16,32),(32,48),(48,62)]:
        h = capture.advance_layers(model,h,lo,hi)
        f = tmp_path/f'{hi}.npy'
        pin = capture.save_array(f,h)
        capture.check_file(f,pin)
        h = np.load(f,allow_pickle=False)
    assert seen == list(range(62))
    np.testing.assert_array_equal(h,expected)
    with f.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(common.Red,match='changed'):
        capture.check_file(f,pin)


class Tensor:
    def __init__(self,a):
        self.a=np.asarray(a)
        self.shape=self.a.shape
        self.dtype='float32'
    def float(self):return self
    def numpy(self):return self.a
    def slice(self,dim,lo,n):
        index=[slice(None)]*self.a.ndim;index[dim]=slice(lo,lo+n)
        return Tensor(self.a[tuple(index)])


def owner(tmp_path,arm='C',S=256):
    return SimpleNamespace(layer=0,arm=arm,bits=4,delta=2.,capture_token_sha='tokens',
                           capture_id='capture_test',capture_dir=tmp_path,rows=[])


def test_streamed_mask_roundtrip_and_gaps(tmp_path):
    S=257
    obj=owner(tmp_path,S=S)
    q=Tensor(np.zeros((1,40,S,96),np.float32))
    writer=capture.LayerWriter(obj,q,q,q,True,'test')
    expected=np.broadcast_to(np.tri(S,dtype=np.uint8),(1,40,S,S)).copy()
    for lo in range(0,S,128):
        hi=min(S,lo+128)
        writer.write(lo,expected[:,:,lo:hi,:hi])
    writer.finish()
    got=np.load(tmp_path/'layer00/selected.pack.npy')
    np.testing.assert_array_equal(np.unpackbits(got,axis=-1,count=S,bitorder='little'),expected)
    assert obj.rows[0]['selected']==40*S*(S+1)//2
    obj.layer=1
    writer=capture.LayerWriter(obj,q,q,q,True,'test')
    with pytest.raises(common.Red,match='incomplete'):
        writer.finish()
    with pytest.raises(common.Red,match='out-of-order'):
        writer.write(1,expected[:,:,:128,:128])
    invalid=expected[:,:,:128,:128].copy();invalid[0,0,0,1]=1
    with pytest.raises(common.Red,match='invalid'):
        writer.write(0,invalid)


@pytest.mark.parametrize('arm',['B','C'])
def test_native_ranged_replay_matches_and_rejects_output_change(tmp_path,arm):
    S=256
    obj=capture.RangeModel.__new__(capture.RangeModel)
    obj.__dict__.update(owner(tmp_path,arm).__dict__)
    obj.observe=True
    q=Tensor(np.arange(40*S*96,dtype=np.float32).reshape(1,40,S,96))
    def native(q,k,kq,v,*args):
        # Independent oracle has a distinctive row value; replay must preserve
        # absolute query positions even when key prefix lengths change.
        out=Tensor(q.a.copy())
        if not args[-1]:return out
        L,K=q.shape[2],k.shape[2]
        valid=np.arange(K)[None,:] <= np.arange(K-L,K)[:,None]
        return out,Tensor(np.broadcast_to(valid,(1,40,L,K)).astype(np.uint8))
    obj.tc=SimpleNamespace(_C=SimpleNamespace(apa_selective_attention_sp=native))
    obj.original_selective=lambda q,k,kq,v,*args:Tensor(q.a.copy())
    obj.diag=SimpleNamespace(selective=lambda q,k,kq,v,*args:native(q,k,kq,v,True))
    out=obj.selective(q,q,q,q,.1,1.,True)
    np.testing.assert_array_equal(out.a,q.a)
    assert obj.rows[0]['pairs']==40*S*(S+1)//2
    obj.layer=1
    if arm=='B':obj.original_selective=lambda q,k,kq,v,*args:Tensor(q.a+1)
    else:
        def bad(q,k,kq,v,*args):
            return native(q,k,kq,v,*args) if args[-1] else Tensor(q.a+1)
        obj.tc._C.apa_selective_attention_sp=bad
    with pytest.raises(common.Red,match='changes native output'):
        obj.selective(q,q,q,q,.1,1.,True)


def aggregation_fixture(monkeypatch,tmp_path):
    monkeypatch.setattr(capture,'ART',tmp_path)
    cell=dict(id='capture_b4_C_8192',arm='C',bits=4,S=8192,depends=['r0','r1'])
    dest=tmp_path/'captures/b4_C_8192';dest.mkdir(parents=True)
    rr={}
    for n,(lo,hi) in enumerate([(0,31),(31,62)]):
        layers={}
        for i in range(lo,hi):
            p=dest/f'layer{i:02d}';p.mkdir()
            pins={}
            for name in ['q.npy','k.npy','kq.npy','selected.pack.npy']:
                pins[name]=capture.save_array(p/name,np.array([i]))
            meta=dict(layer=i,capture_id=cell['id'],token_sha256='tokens',arm='C',bits=4,
                      shapes={'q':[1,40,8192,96],'k':[1,40,8192,96]},
                      files={n:v['sha256'] for n,v in pins.items()},file_pins=pins,
                      selected=1,pairs=2,fraction=.5,path='SP_prefill')
            common.publish(p/'capture.json',meta)
            layers[f'layer{i:02d}/capture.json']=common.sha(p/'capture.json')
        rr[f'r{n}']=dict(result=dict(layer_start=lo,layer_stop=hi,capture_id=cell['id'],
                                    arm='C',bits=4,S=8192,token_sha256='tokens',layers=layers))
    monkeypatch.setattr(capture,'require_pass',lambda id:rr[id])
    return cell,rr,dest


def test_aggregate_complete_sha_pinned_set(monkeypatch,tmp_path):
    cell,rr,dest=aggregation_fixture(monkeypatch,tmp_path)
    result=capture.aggregate(cell)
    assert len(result['layers'])==62 and result['refinement']['fraction']==.5
    assert result['set_sha256']==hashlib.sha256(json.dumps(result['layers'],sort_keys=True).encode()).hexdigest()
    assert (dest/'capture_set.json').exists()


@pytest.mark.parametrize('attack',['missing','overlap','token','foreign','bytes','metadata','stat','missing_pin'])
def test_aggregate_rejects_broken_capture(monkeypatch,tmp_path,attack):
    cell,rr,dest=aggregation_fixture(monkeypatch,tmp_path)
    if attack=='missing':rr['r1']['result']['layers'].pop('layer61/capture.json')
    elif attack=='overlap':rr['r1']['result']['layer_start']=30
    elif attack=='token':rr['r1']['result']['token_sha256']='wrong'
    elif attack=='foreign':rr['r1']['result']['arm']='B'
    elif attack=='bytes':(dest/'layer01/q.npy').write_bytes(b'wrong')
    elif attack=='metadata':(dest/'layer01/capture.json').write_text('{}')
    elif attack=='stat':os.utime(dest/'layer01/q.npy',ns=(1,1))
    else:
        p=dest/'layer01/capture.json';meta=common.read(p);meta['file_pins'].pop('q.npy');p.write_text(json.dumps(meta))
        rr['r0']['result']['layers']['layer01/capture.json']=common.sha(p)
    with pytest.raises((common.Red,FileNotFoundError)):
        capture.aggregate(cell)


def test_fit_dependency_blocks_before_model_load(monkeypatch):
    cell=dict(id='ppl_b4_C_32768',kind='ppl_long',arm='C',depends=['ceiling_b4_C_32768'])
    monkeypatch.setattr(jobs,'require_pass',lambda id:dict(result=dict(fit=False)))
    monkeypatch.setattr(jobs,'protocol',lambda:pytest.fail('must stop before protocol/model work'))
    with pytest.raises(common.Red,match='BLOCKED_NON_FIT'):
        jobs.execute(cell)
    assert jobs.is_oom(RuntimeError('cudaMalloc failed: out of memory'))
    assert not jobs.is_oom(RuntimeError('illegal memory access'))


@pytest.mark.parametrize('oom',[False,True])
def test_ceiling_executes_one_full_prefix_and_records_peak(monkeypatch,oom):
    import apa_sp3_model as legacy
    calls=[]
    class FakeModel:
        def __init__(self):self.peak=SimpleNamespace(result=lambda:dict(peak_resident_mib=123.))
        def set(self,*a):calls.append(('set',a))
        def forward(self,ids,score):
            calls.append(('forward',len(ids),int(ids[0]),score))
            if oom:raise RuntimeError('cudaMalloc failed: out of memory')
            return dict(wall_ms=1.,peak_resident_mib=123.)
    monkeypatch.setattr(legacy,'Model',FakeModel)
    monkeypatch.setattr(driver,'g0_guard',lambda *a:dict(guard='passed'))
    monkeypatch.setattr(jobs,'protocol',lambda:({},np.arange(33337,dtype='<i8')))
    r=jobs.execute(dict(kind='ceiling',arm='B',bits=4,S=32768,depends=[]))
    assert calls==[('set',('B',4,None)),('forward',32768,0,False)]
    assert r['fit'] is (not oom) and r['peak_resident_mib']==123.
    assert r['outcome']==('OOM' if oom else 'FIT') and r['quality_claim'] is False


def test_t_load_oom_receipt_has_no_quality_number(monkeypatch):
    import torch
    def fail():raise torch.OutOfMemoryError('registered mock load OOM')
    monkeypatch.setattr(reference,'TorchModel',fail)
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda:(100,1000))
    monkeypatch.setattr(torch.cuda,'memory_reserved',lambda:500)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda:600)
    with pytest.raises(common.Red,match='T_NON_FIT_OOM') as error:
        reference.reference(dict(S=32768),np.arange(33337))
    r=error.value.details
    assert r['fit'] is False and r['ppl'] is None and r['S']==32768
    assert r['peak_resident_mib']==1000/(1<<20)


def test_long_engine_head_only_slices_predictions_attention_keeps_full_input():
    from contextlib import nullcontext
    seen=[]
    ids=np.arange(32768,dtype='<i8')%7
    full=np.random.default_rng(9).normal(size=(1,32768,7)).astype(np.float32)
    class Engine:
        def __init__(self):self.lm_head=lambda h:h
        def __call__(self,tokens,last_token_only):
            seen.append((tokens.copy(),last_token_only))
            return self.lm_head(Tensor(full)),[]
    engine=Engine();head=engine.lm_head
    model=SimpleNamespace(model=engine,tc=SimpleNamespace(synchronize=lambda:None,no_grad=nullcontext),
                          peak=SimpleNamespace(reset=lambda:None,result=lambda:dict(peak_resident_mib=1)))
    got=jobs.long_forward(model,ids)
    want=last512(full[0],ids)
    assert all(got[k]==v for k,v in want.items())
    np.testing.assert_array_equal(seen[0][0],ids[None])
    assert seen[0][1] is False and engine.lm_head is head


def test_cross_weight_format_gap_reported_without_equality_gate():
    a=dict(ppl=20.,targets=512,target_sha256='same')
    t=dict(ppl=17.,targets=512,target_sha256='same')
    assert jobs.compare_reference(a,t)['engine_minus_T']==3.
    with pytest.raises(common.Red):jobs.compare_reference(a,dict(t,target_sha256='shifted'))
    with pytest.raises(common.Red):jobs.compare_reference(a,dict(t,targets=511))


def test_per_kind_compatibility_allowlist_and_unknown_changes():
    old={'model':'old','metrics':'same'}
    r=dict(cell=dict(kind='baseline'),registration_sha256=common.REG_SHA,fingerprint=old)
    m=dict(file_deltas={'model':dict(before_sha256=['old'],after_sha256='new',unchanged_kinds=['baseline'])})
    assert provenance.compatible(r,current={'model':'new','metrics':'same'},amendment=m)
    assert not provenance.compatible(r,current={'model':'future','metrics':'same'},amendment=m)
    assert not provenance.compatible(r,current={'model':'new','metrics':'changed'},amendment=m)
    r['cell']['kind']='capture'
    assert not provenance.compatible(r,current={'model':'new','metrics':'same'},amendment=m)
    assert 'scripts/apa_sp3_model.py' not in provenance.closure(dict(kind='calibration'))
    assert 'scripts/apa_sp3_metrics.py' in provenance.closure(dict(kind='calibration'))
    assert 'scripts/apa_sp3_model.py' in provenance.closure(dict(kind='ppl'))
    assert 'scripts/apa_sp3_a4_torch.py' in provenance.closure(dict(kind='ppl_long'))


@pytest.mark.parametrize('flash,efficient,expected',[(True,True,'flash'),(False,True,'efficient'),(False,False,None)])
def test_actual_tensor_fused_choice_never_math(flash,efficient,expected):
    events=[]
    class Context:
        def __enter__(self):events.append('enter')
        def __exit__(self,*a):events.append('exit')
    attention=SimpleNamespace(SDPBackend=SimpleNamespace(FLASH_ATTENTION='flash',EFFICIENT_ATTENTION='efficient'),
                              sdpa_kernel=lambda backends:events.append(backends) or Context())
    t=SimpleNamespace(bfloat16='bf16',nn=SimpleNamespace(attention=attention,functional=SimpleNamespace(
            scaled_dot_product_attention=lambda *a,**k:'out')),
            cuda=SimpleNamespace(get_device_capability=lambda:(8,9)),
            backends=SimpleNamespace(cuda=SimpleNamespace(SDPAParams=lambda *a:a,
                can_use_flash_attention=lambda *a,**kw:flash,can_use_efficient_attention=lambda *a,**kw:efficient)))
    q=SimpleNamespace(shape=(1,40,8192,96),dtype='bf16')
    v=SimpleNamespace(shape=(1,40,8192,64),dtype='bf16')
    f=reference.FusedSDPA(t);f.expected_S=8192
    if expected:
        assert f(q,q,v,is_causal=True)=='out'
        assert events==[[expected],'enter','exit']
        assert f.calls[0]['backend']==expected
    else:
        with pytest.raises(common.Red,match='NO_FUSED_BACKEND'):f(q,q,v,is_causal=True)
    with pytest.raises(common.Red,match='no attention chunking'):
        f(SimpleNamespace(shape=(1,40,128,96),dtype='bf16'),q,v,is_causal=True)


def test_hf_snapshot_cpu_import_load_and_sliced_head(tmp_path,monkeypatch):
    # CPU math attention is used ONLY for the toy oracle, not the T worker.
    monkeypatch.setenv('HF_MODULES_CACHE',str(tmp_path/'modules'))
    import torch
    from transformers import AutoConfig
    stage=tmp_path/'snapshot';stage.mkdir()
    reference.stage_snapshot(stage)
    cls,fixes=reference.reference_class(str(stage))
    cfg=AutoConfig.from_pretrained(stage,trust_remote_code=True,local_files_only=True)
    for k,v in dict(hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,
                    num_key_value_heads=2,q_lora_rank=8,kv_lora_rank=8,qk_nope_head_dim=4,
                    qk_rope_head_dim=4,v_head_dim=8,vocab_size=17,max_position_embeddings=1024,
                    rope_scaling=None,use_cache=False).items():setattr(cfg,k,v)
    cfg._attn_implementation='sdpa'
    torch.manual_seed(7)
    toy=cls(cfg).eval()
    weights=tmp_path/'tiny';weights.mkdir();toy.save_pretrained(weights,safe_serialization=False)
    loaded=cls.from_pretrained(weights,config=cfg,local_files_only=True,dtype=torch.bfloat16,
                               attn_implementation='sdpa').eval()
    reference.restore_rotary_buffers(loaded)
    for name, parameter in loaded.named_parameters():
        torch.testing.assert_close(parameter,toy.state_dict()[name].to(torch.bfloat16),rtol=0,atol=0)
    ids=(torch.arange(1024)[None]%17)
    with torch.inference_mode():
        full=loaded(ids,use_cache=False,return_dict=True).logits
        hook=loaded.lm_head.register_forward_pre_hook(reference.sliced_head_hook)
        small=loaded(ids,use_cache=False,return_dict=True).logits
        hook.remove()
    assert small.shape==(1,512,17)
    torch.testing.assert_close(small,full[:,-513:-1],rtol=0,atol=0)
    assert reference.compact_nll(small[0].numpy(),ids[0].numpy()) == last512(full[0].numpy(),ids[0].numpy())
    assert all(p.dtype==torch.bfloat16 for p in loaded.parameters())


def test_original_model_and_scorer_bytes_unchanged():
    before=common.read(common.ART/'a4_before.json')['files']
    assert common.sha(ROOT/'scripts/apa_sp3_model.py')==before['scripts/apa_sp3_model.py']
    assert common.sha(ROOT/'artifacts/apa_sp3/registration.json')==common.REG_SHA


def test_validation_cache_scope_rechecks_next_action(monkeypatch,tmp_path):
    monkeypatch.setattr(common,'job_path',lambda job:tmp_path/(job+'.json'))
    checked=[]
    monkeypatch.setattr(common,'receipt_valid',lambda r,cache=None:checked.append(r['job']) or True)
    common.publish(tmp_path/'parent.json',dict(job='parent',status='PASS',cell=dict(kind='kernel'),dependencies={}))
    common.publish(tmp_path/'child.json',dict(job='child',status='PASS',cell=dict(kind='kernel'),
                  dependencies={'parent':common.sha(tmp_path/'parent.json')}))
    with common.receipt_validation():
        common.require_pass('child')
        common.require_pass('parent')
    assert checked==['child','parent']
    changed=common.read(tmp_path/'parent.json');changed['status']='RED'
    (tmp_path/'parent.json').write_text(json.dumps(changed))
    with common.receipt_validation():
        with pytest.raises(common.Red,match='changed dependency'):
            common.require_pass('child')


def test_namespace_preserves_legacy_capture_timeout():
    assert common.job_path('capture_b4_B_8192').parent.name=='jobs_a4'
    assert common.job_path('ppl_T_32768').parent.name=='jobs_a4'
    assert common.job_path('g0').parent.name=='jobs'
    # The registered estimate uses int(float bytes * 1.15), which truncates
    # one byte below the exact rational estimate; this is a disk planning rail.
    assert capture.required_space(32768)==490783899647


def test_fingerprint_bridge_rejects_wrong_order(monkeypatch):
    original=provenance.read
    def wrong(path):
        value=original(path)
        if Path(path).name==provenance.BRIDGE:value['order_sha256']='0'*64
        return value
    monkeypatch.setattr(provenance,'read',wrong)
    with pytest.raises(common.Red,match='binding changed'):provenance.bridge()
