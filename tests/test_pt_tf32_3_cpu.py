"""Author CPU regressions, not blind review or GPU certification.
Prior art: pytest (2004), NumPy (2020), POSIX pipes, Higham (2002), taken.
Ours: storage-failure containment, exact transient gradient provenance,
independent sanitizer verdicts and real host Lt descriptor checks.
"""
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pytest
import pt_tf32_3 as t
import pt_tf32_3_grapa as g
import pt_tf32_3_storage as s
import pt_tf32_3_slot as slot


def arrays():return {'q':np.arange(12,dtype=np.float32).reshape(3,4),'v':np.array([1.,-2.,3.],np.float32)}


def test_pipe_exact_bytes_and_content_provenance():
    a=arrays();stream=io.BytesIO();s.send_grads(stream,a);stream.seek(0)
    b=s.receive_grads(stream)
    assert s.canonical_sha(a)==s.canonical_sha(b)
    for k in a:np.testing.assert_array_equal(a[k].view(np.uint32),b[k].view(np.uint32))
    b['q'][0,0]=-0.
    assert s.canonical_sha(a)!=s.canonical_sha(b)


def test_real_pipe_cross_process_larger_than_pipe_buffer():
    read_fd,write_fd=os.pipe()
    code="import os,sys,numpy as np;sys.path.insert(0,'scripts');import pt_tf32_3_storage as s\nwith os.fdopen(int(sys.argv[1]),'wb') as f:s.send_grads(f,{'q':np.arange(300000,dtype=np.float32)})"
    p=subprocess.Popen([sys.executable,'-B','-c',code,str(write_fd)],pass_fds=(write_fd,),
                       env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    os.close(write_fd)
    with os.fdopen(read_fd,'rb') as f:a=s.receive_grads(f)
    assert p.wait(timeout=5)==0
    np.testing.assert_array_equal(a['q'],np.arange(300000,dtype=np.float32))


@pytest.mark.parametrize('bad',[b'',struct.pack('<Q',2**32),struct.pack('<Q',1)+b'{'])
def test_pipe_rejects_truncated_or_invalid_headers(bad):
    with pytest.raises((EOFError,ValueError)):s.receive_grads(io.BytesIO(bad))


@pytest.mark.parametrize('shape,nbytes',[([2**40],2**42),([-1],4),([2],4),([1.5],6)])
def test_pipe_validates_metadata_before_allocating(shape,nbytes):
    header=json.dumps([dict(name='q',shape=shape,bytes=nbytes)]).encode()
    with pytest.raises(ValueError):s.receive_grads(io.BytesIO(struct.pack('<Q',len(header))+header))


def test_pipe_detects_truncated_array_and_trailing_bytes():
    stream=io.BytesIO();s.send_grads(stream,arrays());raw=stream.getvalue()
    with pytest.raises(EOFError):s.receive_grads(io.BytesIO(raw[:-1]))
    with pytest.raises(ValueError):s.receive_grads(io.BytesIO(raw+b'x'))


@pytest.mark.parametrize('free,expected',[(s.MIN_FREE_BYTES-1,'BLOCKED'),(s.MIN_FREE_BYTES,'GREEN')])
def test_space_rail_at_exact_boundary(tmp_path,monkeypatch,free,expected):
    monkeypatch.setattr(s.shutil,'disk_usage',lambda path:SimpleNamespace(free=free))
    assert s.space_status(tmp_path/'missing/subdir')['status']==expected


def test_low_disk_prevents_process_launch_and_can_write_receipt(tmp_path,monkeypatch):
    monkeypatch.setattr(s.shutil,'disk_usage',lambda path:SimpleNamespace(free=7*1024**3))
    monkeypatch.setattr(slot.subprocess,'Popen',lambda *a,**k:pytest.fail('low-space lane launched'))
    row=slot.run_lane([sys.executable,'-c','raise Exception'],tmp_path/'lane.log',dict(os.environ,CUDA_VISIBLE_DEVICES=''),1.)
    t.create_json(tmp_path/'lane_receipt.json',row)
    assert row['status']=='BLOCKED' and row['returncode'] is None
    assert not (tmp_path/'lane.log').exists()


@pytest.mark.parametrize('name',['memcheck','racecheck','synccheck'])
def test_real_slot10_sanitizer_zero_is_green_despite_unit_rc_one(name):
    root=t.ROOT/'artifacts/pt_tf32_2/lead_slot_01'
    old=json.loads((root/(name+'_receipt.json')).read_text())
    assert old['returncode']==1
    row=slot.sanitizer_result(name,(root/(name+'.log')).read_text())
    assert row['status']=='GREEN' and row['errors']==0 and row['executed_tests']>=56


@pytest.mark.parametrize('name',['memcheck','racecheck','synccheck'])
def test_sanitizers_do_not_pass_without_completion_or_with_any_errors(name):
    assert slot.sanitizer_result(name,'ERROR SUMMARY: 0 errors')['status']=='BLOCKED'
    assert slot.sanitizer_result(name,'56 passed\n')['status']=='BLOCKED'
    assert slot.sanitizer_result(name,'56 passed\nERROR SUMMARY: 0 errors\nERROR SUMMARY: 2 errors')['status']=='RED'


def test_own_lane_timeout_and_enospc_are_blocked(tmp_path):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
    row=slot.run_lane([sys.executable,'-B','-c','while True: pass'],tmp_path/'timeout.log',env,.1)
    assert row['status']=='BLOCKED_TIMEOUT' and row['elapsed_seconds']<5
    row=slot.run_lane([sys.executable,'-B','-c','raise OSError(28,"No space left on device")'],tmp_path/'space.log',env,5)
    assert row['status']=='BLOCKED'


@pytest.mark.parametrize('keep',[False,True])
def test_gradient_sink_default_off_opt_in_and_module_restored(tmp_path,keep):
    class Buffer(io.BytesIO):
        def close(self):self.saved=self.getvalue();super().close()
    stream=Buffer();dest=tmp_path/'child';dest.mkdir();a=arrays()
    module=SimpleNamespace(np=np,sha256_file=t.sha,write_exclusive=t.create_json)
    old_savez=np.savez
    with g.gradient_output(module,dest,stream,keep):
        module.np.savez(dest/'grad.npz',**a)
        digest=module.sha256_file(dest/'grad.npz')
        module.write_exclusive(dest/'pair.json',dict(grad_file=str(dest/'grad.npz'),grad_sha256=digest))
    assert module.np is np and np.savez is old_savez
    assert (dest/'grad.npz').exists()==keep
    receipt=json.loads((dest/'pair.json').read_text())
    assert receipt['gradient_content_sha256']==s.canonical_sha(a)
    assert receipt['grad_file'] is not None if keep else receipt['grad_file'] is None
    assert s.canonical_sha(s.receive_grads(io.BytesIO(stream.saved)))==s.canonical_sha(a)


@pytest.mark.parametrize('keep',[False,True])
def test_checkpoint_sink_default_off_opt_in_and_restoration(tmp_path,keep):
    calls=[]
    def save(path,*a,**kw):calls.append(path);Path(path).write_bytes(b'checkpoint')
    module=SimpleNamespace(save=save)
    with g.checkpoint_output(module,tmp_path,keep):
        module.save(tmp_path/'timing.ckpt',None,None,data_index=1)
    assert module.save is save and bool(calls)==keep
    assert (tmp_path/'timing.ckpt').exists()==keep
    assert json.loads((tmp_path/'checkpoint_output.json').read_text())['attempts']==[dict(path=str(tmp_path/'timing.ckpt'),written=keep)]


def test_checkpoint_sink_refuses_path_outside_lane(tmp_path):
    module=SimpleNamespace(save=lambda *a,**k:pytest.fail('unexpected save'))
    with g.checkpoint_output(module,tmp_path,False):
        with pytest.raises(ValueError,match='outside lane'):module.save(tmp_path.parent/'wrong.ckpt')


def test_frozen_noise_floor_without_archives_and_tamper_detection(tmp_path,monkeypatch):
    monkeypatch.setattr(g,'ART',tmp_path)
    (tmp_path/'SOURCE_MANIFEST.json').write_text('{}')
    def run(command,state,arm,dest,keep=False):
        dest.mkdir(parents=True)
        a={'q':np.array([1.,.02 if arm=='0-10' else 0.],np.float32)}
        t.create_json(dest/'pair.json',dict(gradient_content_sha256=s.canonical_sha(a),grad_file=None,grad_sha256=None))
        return a
    monkeypatch.setattr(g,'run_child',run)
    assert g.calibrate(tmp_path)['verdict']=='GREEN'
    floor=g.checked_floor(tmp_path/'NOISE_FLOOR.json')
    assert len(floor['pairs'])==6 and floor['bar']<1 and s.dump_bytes(tmp_path)==0
    floor['pairs'][0]['dot']+=.1
    path=tmp_path/'NOISE_FLOOR.json';path.write_text(json.dumps(floor));path.with_suffix('.sha256').write_text(t.sha(path)+'\n')
    with pytest.raises(ValueError):g.checked_floor(path)


def test_slot_budget_sequence_dump_size_and_opt_in():
    rows=slot.sequence(t.ART/'never_run')
    assert sum(r[1] for r in rows)<=1800
    assert [r[0] for r in rows]==t.registration()['slot']['sequence']
    assert all('--keep-grads' not in argv for _,_,argv in rows)
    assert all('--keep-grads' in argv for n,_,argv in slot.sequence(t.ART/'never_run',True) if n in ('noise_floor','onset','healthy','control','step_time'))
    assert slot.dump_estimate(False)['total_dump_bytes_upper_estimate']==0
    assert slot.dump_estimate(True)['total_dump_bytes_upper_estimate']>8*1024**3


@pytest.mark.parametrize('tb',[False,True])
def test_real_cpu_lt_descriptors_and_no_selected_algorithm(tb):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    c=t.load_fork(False);r=c.tf32_gemm_self_check(4096,1792,1024,tb,False)
    assert r['fast_tf32']==1 and r['transa']==int(tb) and r['transb']==0
    assert r['lda']==(1024 if tb else 1792) and r['ldb']==1024 and r['ldc']==1792
    assert r['algorithm_id']==-1 and r['device_selected']==0 and r['heuristic_count']==0
    assert c.get_tf32_gemm() is False
    with pytest.raises(RuntimeError):c.tf32_gemm_self_check(-1,16,8,tb,False)


def test_new_registration_exact_states_and_fork_binary():
    r=g.registered();old=json.loads((t.ROOT/'artifacts/pt_tf32_2/GRAPA_REGISTRATION_002.json').read_text())
    for key in ('states','timing','batches_npz','trainer_sources_sha256'):assert r[key]==old[key]
    assert Path(r['engine']['so_path']).resolve().is_relative_to(t.ROOT)


def test_host_contract_executable():
    r=subprocess.run([str(t.ART/'pt_tf32_host_contract')],capture_output=True,text=True,timeout=10)
    assert r.returncode==0,r.stderr
    assert r.stdout.strip()=='HOST_CONTRACT: 6 expected rejections; no CUDA device operations'


@pytest.mark.parametrize('script,args',[
    ('pt_tf32_3.py',['dispatch','--lead-gpu']),
    ('pt_tf32_3.py',['gemm','--lead-gpu']),
    ('pt_tf32_3_grapa.py',['noise','--lead-gpu']),
    ('pt_tf32_3_slot.py',['--lead-gpu','--out','artifacts/pt_tf32_3/never_run'])])
def test_hidden_cuda_blocks_all_gpu_entrypoints(script,args):
    r=subprocess.run([sys.executable,'-B','scripts/'+script,*args],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=10)
    assert r.returncode==2 and 'BLOCKED' in r.stderr


def test_dv_midpoint_witnesses_explain_recorded_coordinates():
    # Prior art: TF32 adjacent-bin spacing 2^(e-10). A permitted FP32
    # perturbation at the midpoint can switch RN_TF32(p) by one full bin.
    # Fixture is independent of the native implementation; this is a
    # quantization justification, not a suppressed native test failure.
    rows=json.loads((t.ART/'CPU_DIAGNOSIS.json').read_text())['edges']
    expected=[(17,3,2,1,30,2.4646520614624023e-05),
              (17,3,2,1,42,3.319978713989258e-05),
              (33,3,18,0,22,2.3052096366882324e-05)]
    observed=[2.4646520614624e-5,3.32072377e-5,2.30306759e-5]
    for (L,h,i,j,d,delta),actual in zip(expected,observed):
        row=next(r for r in rows if r['shape']['L']==L);x=t.fixture(row['shape'])
        w=next(w for w in row['witnesses'] if (w['h'],w['i'],w['j'])==(h,i,j))
        assert w['fp32_ulps_to_midpoint']==0
        jump=abs(w['tf32_bin']*float(t.old.tf32_round(x['dO'][0,h,i,d])))
        assert jump==delta and abs(jump-actual)<2.2e-8
