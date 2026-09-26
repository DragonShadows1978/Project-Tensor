"""Author baseline only. Prior art: independent NumPy scatter reference and
mutation-style counterexamples (House Rules 2026), taken; PT-DET-1 cases ours.
"""
import copy
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import pt_det_1 as d
import pt_det_1_slot as slot
import pt_tf32_4_slot as old_slot


@pytest.fixture(scope='session')
def cpu(tmp_path_factory):
    out=tmp_path_factory.mktemp('det_bridge')/'bridge.so'
    subprocess.run(['g++','-std=c++17','-O2','-shared','-fPIC','-I'+str(ROOT/'tensor_cuda/include'),
                    str(ROOT/'tests/pt_det_1_cpu_bridge.cpp'),'-o',str(out)],check=True,timeout=60,
                   env=dict(os.environ,TMPDIR=str(out.parent)))
    fn=ctypes.CDLL(str(out)).pt_det_cpu
    fn.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int64]*3;fn.restype=ctypes.c_int
    def run(ids,grad,vocab):
        ids=np.ascontiguousarray(ids,dtype=np.int64);grad=np.ascontiguousarray(grad,dtype=np.float32)
        result=np.full((vocab,grad.shape[1]),np.nan,np.float32)
        rc=fn(grad.ctypes.data,ids.ctypes.data,result.ctypes.data,len(ids),vocab,grad.shape[1])
        if rc:raise ValueError('invalid CPU input')
        return result
    return run


@pytest.mark.parametrize('n,width,kind',[(0,3,'empty'),(1,1,'unique'),(31,257,'unique'),
    (4096,33,'same'),(4096,129,'repeated'),(4096,1024,'repeated'),(27,7,'cancellation')])
def test_shared_segment_arithmetic_against_independent_fp64(cpu,n,width,kind):
    rng=np.random.default_rng(15)
    ids=np.arange(n,dtype=np.int64) if kind=='unique' else rng.integers(0,13,n,dtype=np.int64)
    if kind in ('same','cancellation'):ids[:]=3
    grad=rng.standard_normal((n,width),dtype=np.float32)
    if kind=='cancellation':
        grad[0::3]=1e8;grad[1::3]=1.;grad[2::3]=-1e8
    vocab=max(32,n if kind=='unique' else 13)
    ref=d.reference(ids,grad,vocab);results=[cpu(ids,grad,vocab) for _ in range(5)]
    assert all(a.tobytes()==results[0].tobytes() for a in results)
    assert d.rel_l2(results[0],ref)<=1e-6
    if kind=='cancellation':assert np.all(results[0][3]==9.)


@pytest.mark.parametrize('ids',[[-1],[8]])
def test_cpu_rejects_out_of_range(cpu,ids):
    with pytest.raises(ValueError):cpu(ids,np.ones((1,3),np.float32),8)


@pytest.mark.parametrize('value,expected',[(None,False),('0',False),('1',True),('true',False),('',False)])
def test_compiled_engine_host_contract_environment_and_setter(value,expected):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
    env.pop('TC_DET_EMBED_BWD',None)
    if value is not None:env['TC_DET_EMBED_BWD']=value
    r=subprocess.run([str(d.ART/'pt_det_1_host_contract'),str(int(expected))],env=env,
                     capture_output=True,text=True,timeout=10)
    assert r.returncode==0,r.stdout+r.stderr
    assert '5 exact CPU dispatches; no CUDA device operations' in r.stdout


def test_default_off_kernel_and_body_are_byte_preserved():
    base=(d.ART/'baseline/tensor_cuda/src/kernels.cu').read_text()
    now=(ROOT/'tensor_cuda/src/kernels.cu').read_text()
    start='__global__ void embed_bwd_kernel';end='template <typename T>\n__global__ void sgd_kernel'
    assert base[base.index(start):base.index(end)]==now[now.index(start):now.index(end)]
    start='NDArray embedding_backward(';end='\nvoid axpy_('
    old=base[base.index(start):base.index(end)]
    new=now[now.index(start):now.index(end)]
    assert new[new.index('  int64_t V ='):]==old[old.index('  int64_t V ='):]


def logs():
    return '\n'.join(f'2026-09-26 08:00:00 | step {s} | loss 1.00000 | refine 0.15 | lr 0.0003 | 100 tok/s '
                     f'| 5.0s/step | sample_i {s+1} | foo x | gnorm 0.25000 | clip_coef 1 | clip 1'
                     for s in range(32056,32086))


@pytest.mark.parametrize('mutate',[lambda s:'',lambda s:'\n'.join(s.splitlines()[:-1]),
    lambda s:s+'\n'+s.splitlines()[0],lambda s:'\n'.join(reversed(s.splitlines())),
    lambda s:s.replace('loss 1.00000','loss nan',1),lambda s:s.replace('gnorm 0.25000','gnorm inf',1),
    lambda s:s.replace('| clip_coef 1','| clipcoef 1',1)])
def test_log_gate_rejects_missing_duplicate_malformed_nonfinite(mutate):
    with pytest.raises(ValueError):d.parse_rows(mutate(logs()))


def ckpt():
    a=np.array([1.,0.,-2.],np.float32)
    return dict(model={'emb.weight':a,'buffer':np.ones((2,3),np.float32)},
                adam_m=[a.copy()],adam_v=[a.copy()],adam_t=27450,data_index=32086,
                extra=dict(step=32085,loader=dict(sample_i=32086)))


def digest(tmp_path,name,b):
    path=tmp_path/(name+'.ckpt');path.write_bytes(pickle.dumps(b));return d.checkpoint_digest(path)


def run_record(tmp_path,name='a',blob=None):
    return dict(status='GREEN',rows=d.parse_rows(logs()),checkpoint=digest(tmp_path,name,ckpt() if blob is None else blob),
                probe_steps=[dict(n=i,gnorm_hex=(.25).hex(),coef_hex=(1.).hex(),**({'grad_sha256':['a']} if i==1 else {}))
                             for i in range(1,31)])


def test_exact_pair_and_text_diff(tmp_path):
    a=run_record(tmp_path);b=copy.deepcopy(a)
    assert d.compare_pair(a,b)['bitwise']
    b['rows'][8]['loss']='1.0000'  # Same numeric value, different logged text.
    result=d.compare_pair(a,b)
    assert not result['bitwise'] and result['differing_log_steps']==[32064]


@pytest.mark.parametrize('change',['ulp','signed_zero','dtype','shape','missing','extra','adam'])
def test_bitwise_saved_weights_including_dtype_shape_names(tmp_path,change):
    a=run_record(tmp_path);blob=ckpt()
    if change=='ulp':blob['model']['emb.weight'][0]=np.nextafter(np.float32(1),np.float32(2))
    elif change=='signed_zero':blob['model']['emb.weight'][1]=-0.
    elif change=='dtype':blob['model']['emb.weight']=blob['model']['emb.weight'].astype(np.float64)
    elif change=='shape':blob['model']['emb.weight']=blob['model']['emb.weight'].reshape(1,3)
    elif change=='missing':del blob['model']['buffer']
    elif change=='extra':blob['model']['extra']=np.ones(1,np.float32)
    elif change=='adam':blob['adam_m'][0][0]=2.
    b=run_record(tmp_path,'b',blob)
    assert not d.compare_pair(a,b)['bitwise']


@pytest.mark.parametrize('change',['nan','empty_model','empty_adam','moments_length'])
def test_checkpoint_rejects_nonfinite_and_empty(tmp_path,change):
    b=ckpt()
    if change=='nan':b['model']['emb.weight'][0]=np.nan
    if change=='empty_model':b['model']={}
    if change=='empty_adam':b['adam_m']=[]
    if change=='moments_length':b['adam_v']=[]
    with pytest.raises(ValueError):digest(tmp_path,'bad',b)


def test_pair_cannot_pass_failed_or_incomplete_runs(tmp_path):
    a=run_record(tmp_path);b=copy.deepcopy(a);b['rows'].pop()
    assert not d.compare_pair(a,b)['measured']
    b['status']='RED'
    assert d.compare_pair(a,b)['bitwise'] is None


def test_control_prediction_is_separate_from_treatment_pass():
    pairs={a:dict(measured=True,bitwise=a.endswith('_on')) for a in d.ARMS}
    assert d.assess_pairs(pairs)=='GREEN'
    pairs['v3_off']['bitwise']=True;assert d.assess_pairs(pairs)=='NOT_RECURRED'
    pairs['v3_on']['bitwise']=False;assert d.assess_pairs(pairs)=='RED'
    pairs['bf16_on']['measured']=False;assert d.assess_pairs(pairs)=='BLOCKED'
    assert d.assess_pairs({})=='BLOCKED'


def test_replay_argv_differences_are_only_registered_paths_and_policy(tmp_path):
    for arm in d.ARMS:
        argv=d.trainer_argv(arm,tmp_path/arm)
        assert argv[argv.index('--steps')+1]=='32085'
        assert argv[argv.index('--ckpt')+1]==d.registration()['repro']['checkpoint']['path']
        assert argv[argv.index('--save-ckpt')+1]==str(tmp_path/arm/'replay.ckpt')
        assert argv[argv.index('--corpus-phase')+1]=='consolidated_v3'  # bf16 is same corpus.
        assert ('--fwd-fp32-blocks' in argv)==arm.startswith('v3')
    assert d.trainer_argv('v3_on',tmp_path)==d.trainer_argv('v3_off',tmp_path)
    assert d.trainer_argv('bf16_on',tmp_path)==d.trainer_argv('bf16_off',tmp_path)


def test_gpu_guard_runs_before_descriptor_access(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    def fail(*a,**kw):raise AssertionError('lock inspected on CPU seat')
    monkeypatch.setattr(os,'readlink',fail)
    with pytest.raises(d.Blocked,match='CUDA_VISIBLE_DEVICES'):d.require_lead(True,9)


def test_slot_budgets_and_mandatory_certification_lane():
    rows=slot.sequence(d.ART/'not_run',9)
    assert [r[0] for r in rows]==['embedding','pt_det_1_repro']
    assert sum(r[1] for r in rows)+20==1500
    assert 'pt_det_1_repro' in [r[0] for r in old_slot.sequence(ROOT/'not_run')]
    assert old_slot.sequence(ROOT/'not_run')[-1][1]==0


def test_actual_batch_selection_and_digest():
    rows,source=d.batches()
    assert len(rows)==3 and source['kind']=='real_CC46'
    assert rows[0][0]=='x_32055'
    assert all(ids.shape==(4096,) for _,ids in rows)


def test_registration_unchanged_bars():
    r=d.registration()
    assert r['embedding_gate']['relative_L2_max']==1e-6
    assert r['embedding_gate']['det_over_atomic_ms_max']==2
    assert r['repro']['steps']==30 and r['repro']['runs_per_arm']==2


def test_cruise_adapter_only_allows_the_verified_engine():
    old=dict(spec='0-10',kernels='tf32',nested=dict(engine=dict(so_path='old',so_sha256='a')))
    new_engine=dict(so_path='new',so_sha256='b')
    actual=copy.deepcopy(old);actual['nested']['engine']=new_engine
    assert d.historical_precision_record(actual,old,new_engine)==old
    assert old['nested']['engine']['so_path']=='old'
    for key,value in [('spec','0-9'),('kernels','fp32')]:
        bad=copy.deepcopy(actual);bad[key]=value
        with pytest.raises(ValueError):d.historical_precision_record(bad,old,new_engine)
    bad=copy.deepcopy(actual);bad['nested']['engine']['so_sha256']='wrong'
    with pytest.raises(ValueError):d.historical_precision_record(bad,old,new_engine)


def test_cruise_adapter_passes_original_nonprecision_arguments():
    from types import SimpleNamespace
    received=[]
    cruise=SimpleNamespace(run_boundary=lambda *a,**kw:received.append((a,kw)))
    historical=dict(fwd_fp32_blocks={'engine':{'so_path':'old','so_sha256':'a'}},fwd_precision_kernels=None)
    engine=dict(so_path='new',so_sha256='b')
    d.install_engine_replay_adapter(cruise,historical,engine)
    cruise.run_boundary('loader','model',step=32055,grad_clip=1.,fwd_fp32_blocks={'engine':engine},fwd_precision_kernels=None)
    assert received==[(('loader','model'),dict(step=32055,grad_clip=1.,**historical))]


@pytest.mark.parametrize('mode',['0','1'])
def test_python_api_opt_in_without_cuda_calls(mode):
    code='''from pathlib import Path
import os, tensor_cuda as tc
assert Path(tc._C.__file__).resolve().is_relative_to(Path(os.environ['PYTHONPATH']).resolve())
assert tc._C.get_deterministic_embed_bwd() == (os.environ['TC_DET_EMBED_BWD'] == '1')
tc._C.set_deterministic_embed_bwd(True)
assert tc._C.get_deterministic_embed_bwd()
tc._C.set_deterministic_embed_bwd(False)
assert not tc._C.get_deterministic_embed_bwd()
assert callable(tc._C._embedding_backward)
print('PT_DET_1 PYTHON_API: fork import and switches only; no CUDA calls')
'''
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',TC_DET_EMBED_BWD=mode,
             PYTHONPATH=str(ROOT/'tensor_cuda'),PYTHONDONTWRITEBYTECODE='1')
    r=subprocess.run([sys.executable,'-B','-c',code],env=env,capture_output=True,text=True,timeout=10)
    assert r.returncode==0,r.stdout+r.stderr


def test_legacy_slot_cannot_reach_gpu_without_required_repro(tmp_path,monkeypatch):
    # Simulated environment mapping in the unit under test only. The process's
    # real CUDA_VISIBLE_DEVICES remains empty; no child or GPU call occurs.
    from types import SimpleNamespace
    registered=old_slot.t.registration()
    monkeypatch.setattr(old_slot.t,'registration',lambda:registered)
    out=tmp_path/'pt4'/'slot';monkeypatch.setattr(old_slot.t,'ART',tmp_path/'pt4')
    monkeypatch.setattr(old_slot,'os',SimpleNamespace(environ={'CUDA_VISIBLE_DEVICES':'fixture-only'}))
    monkeypatch.setattr(sys,'argv',['pt_tf32_4_slot.py','--lead-gpu','--out',str(out)])
    assert old_slot.main()==2
    receipt=json.loads((out/'SLOT_SUMMARY.json').read_text())
    assert receipt['gpu_executed'] is False and receipt['lanes']['pt_det_1_repro']['status']=='BLOCKED'


def replay_fixture(path,arm,changed=False):
    """Fictional tiny checkpoints + registered structural fields; never a GPU receipt."""
    path.mkdir();(path/'logs').mkdir()
    reg=d.registration()['repro'];cr=json.loads(Path(reg['cc46_registration']['path']).read_text())
    live={x[0]:dict(zip(d.ROW_FIELDS,x)) for x in cr['live']['train_log']['rows']}
    rows=[live[s] for s in range(32056,32086)]
    text='\n'.join(f"| step {r['step']} | loss {r['loss']} | refine {r['refine']} | lr {r['lr']} | 1 tok/s "
                   f"| 1s/step | sample_i {r['sample_i']} | foo x | gnorm {r['gnorm']} | clip_coef {r['clip_coef']} | clip {r['clip']}"
                   for r in rows)
    (path/'logs/train.log').write_text(text+'\n')
    mode=int(arm.endswith('_on'));cc_arm='a' if arm.startswith('v3') else 'd'
    m=dict(binary=dict(path='tensor_cuda/tensor_cuda/fake_test_only.so',sha256='fake_test_only'))
    (path/'leg.stdout').write_text(
        f"ENGINE SO {d.ROOT/m['binary']['path']} SHA256 fake_test_only\n"
        f"PT_DET_1 MODE {mode}\nPT_DET_1 MODE_END {mode}\n"
        '| RESUMED from fixture: step 32055, sample_i 11584,\n'
        'CRUISE REPLAY step=32055 rule=rail_binding_cap fixture\n')
    b=ckpt();b['data_index']=rows[-1]['sample_i'];b['extra']['loader']['sample_i']=rows[-1]['sample_i']
    b['extra']['grad_clip']={'max_norm':1.}
    if cc_arm=='a':
        b['extra']['fwd_fp32_blocks']={'spec':'0-10'};b['extra']['fwd_precision_kernels']={'kernels':'tf32'}
    if changed:b['model']['emb.weight'][0]=np.nextafter(np.float32(1),np.float32(2))
    (path/'replay.ckpt').write_bytes(pickle.dumps(b))
    records=[dict(kind='step',n=i,gnorm_hex=(.25).hex(),coef_hex=(1.).hex(),
                  **({'grad_sha256':['fixture_grad']} if i==1 else {})) for i in range(1,31)]
    records.append(dict(kind='end',steps=30,arm=cc_arm,variants_end=['g1','h'],lt_info_end=[1 if cc_arm=='a' else -1]))
    (path/'probe.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records))
    schema=d.tensor_schema(d.checkpoint_digest(path/'replay.ckpt'))
    receipt=d.run_receipt(path,dict(status='GREEN',returncode=0),arm,schema,m)
    assert receipt['status']=='GREEN',receipt['checks']
    return receipt,m


@pytest.mark.parametrize('arm',d.ARMS)
def test_structural_replay_receipts_all_four_arms(tmp_path,arm):
    r,m=replay_fixture(tmp_path/arm,arm)
    assert len(r['rows'])==len(r['probe_steps'])==30
    path=tmp_path/arm
    records=[json.loads(x) for x in (path/'probe.jsonl').read_text().splitlines()]
    records[0]['grad_sha256']=[]
    (path/'probe.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records))
    bad=d.run_receipt(path,dict(status='GREEN',returncode=0),arm,d.tensor_schema(r['checkpoint']),m)
    assert bad['status']=='RED' and not bad['checks']['gradients']


def test_certificate_verifier_rederives_pairs_and_rejects_log_tampering(tmp_path,monkeypatch):
    pairs={};hashes={}
    for arm in d.ARMS:
        pair=[]
        for i in (1,2):
            path=tmp_path/f'{arm}_{i}'
            rec,m=replay_fixture(path,arm,changed=arm.endswith('_off') and i==2)
            d.create_json(path/'receipt.json',rec);pair.append(rec)
            hashes[f'{arm}_{i}/receipt.json']=d.sha(path/'receipt.json')
        pairs[arm]=d.compare_pair(*pair)
    monkeypatch.setattr(d,'ART',tmp_path)
    monkeypatch.setattr(d,'verify_manifest',lambda:m)
    (tmp_path/d.MANIFEST_NAME).write_text('fictional CPU test manifest')
    result=dict(verdict='GREEN',registration_sha256=d.REG_SHA,binary=m['binary'],steps=30,
                manifest_sha256=d.sha(tmp_path/d.MANIFEST_NAME),
                source_checkpoint_sha256=d.registration()['repro']['checkpoint']['sha256'],
                pairs=pairs,receipt_sha256=hashes)
    d.create_json(tmp_path/'summary.json',result)
    assert d.verify_repro(tmp_path/'summary.json')['verdict']=='GREEN'
    log=tmp_path/'v3_on_1/logs/train.log';log.write_text(log.read_text()+'tampered\n')
    with pytest.raises(ValueError,match='log drift'):d.verify_repro(tmp_path/'summary.json')
