"""Author CPU tests, never native GPU parity or blind verification.
Prior art: NumPy/pytest; BP-KERNEL-2/3/4 FP64 references and gate rules, taken.
Ours: TF32 conversion/owner fixtures, native host guards and fork containment.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pytest
import pt_tf32_1 as t


def spec(L=17,S=19,D=19,VD=17,**kw):
    return dict(B=1,H=4,KVH=2,L=L,S=S,D=D,VD=VD,causal=True,
                scale=float(np.float32(1/np.sqrt(D))),zthr=1.0364333894937898,**kw)


@pytest.mark.parametrize('sign',[1,-1])
def test_tf32_rna_halfway_and_precision(sign):
    x=np.array([1.,1+2**-12,1+2**-11,1+3*2**-11,1+2**-9],np.float32)*sign
    want=np.array([1.,1.,1+2**-10,1+2**-9,1+2**-9],np.float32)*sign
    np.testing.assert_array_equal(t.tf32_round(x),want)
    # p/dS value that BF16 would erase survives TF32; no BF16 coefficient buffer.
    assert t.tf32_round(x)[-1] != sign
    special=t.tf32_round(np.array([np.inf,-np.inf,np.nan,0.,-0.],np.float32))
    assert np.isposinf(special[0]) and np.isneginf(special[1]) and np.isnan(special[2])
    assert np.signbit(special[-1])


@pytest.mark.parametrize('width',[1,7,8,9,19,64,96,127,128])
def test_k8_fragment_partition_and_zero_padding(width):
    rng=np.random.default_rng(width)
    a=np.zeros((16,128),np.float32);b=a.copy()
    a[:13,:width]=rng.normal(size=(13,width));b[:11,:width]=rng.normal(size=(11,width))
    total=np.zeros((16,16),np.float64)
    for d in range(0,width,8):
        total+=t.mma_model(a[:,d:d+8],b[:,d:d+8].T)
    np.testing.assert_allclose(total,t.mma_model(a,b.T),atol=1e-12,rtol=1e-12)
    np.testing.assert_array_equal(total[13:],0)
    np.testing.assert_array_equal(total[:,11:],0)


@pytest.mark.parametrize('causal',[False,True])
@pytest.mark.parametrize('lengths',[(1,1),(15,17),(16,16),(17,31),(33,35)])
@pytest.mark.parametrize('widths',[(1,1),(19,17),(96,64),(128,65)])
def test_tiled_forward_and_both_backward_owners(causal,lengths,widths):
    s=spec(*lengths,*widths);s['causal']=causal
    x=t.fixture(s)
    expected=t.front.reference(x,s)
    actual=t.tiled_forward_model(x,s,rounded=False)
    for n in ('out','lse','thr'):
        np.testing.assert_allclose(actual[n],expected[n],rtol=2e-12,atol=2e-12)
    np.testing.assert_array_equal(actual['selection'],expected['selection'])
    grads=t.tiled_backward_model(x,expected,s,rounded=False)
    # Independent dense VJP reference, not a second owner traversal.
    ref=t.back.reference(x,expected,s,output_dot=True)
    for n in t.NAMES:
        np.testing.assert_allclose(grads[n],ref[n],rtol=3e-12,atol=3e-12)


def test_selected_full_fp32_exact_scores_and_detached_kq():
    s=spec(2,3,1,1);s.update(H=1,KVH=1,scale=1.,zthr=0.)
    x={n:np.ones((1,1,2 if n in ('q','dO') else 3,1),np.float32) for n in ('q','k','kq','v','dO')}
    x['q']*=1.0002;x['k']*=2.0003;x['kq'][:]=-1.
    f=t.tiled_forward_model(x,s)
    assert f['selection'][0,0].tolist()==[[True,True,False],[True,True,True]]
    # Identical logits: LSE exposes full FP32 Q.K, not truncated operands.
    score=float(x['q'].flat[0])*float(x['k'].flat[0])
    np.testing.assert_allclose(f['lse'][0,0],[score+np.log(2),score+np.log(3)],rtol=1e-12)
    frozen=dict(out=np.zeros_like(x['dO']),lse=f['lse'],thr=np.full_like(f['thr'],1e10))
    grads=t.tiled_backward_model(x,frozen,s)
    np.testing.assert_array_equal(grads['dK'],0)
    assert np.linalg.norm(grads['dQ'])>0


def test_rope_suffix_and_registered_geometries():
    r=t.registration()
    assert {(s['D'],s['rope_width']) for s in r['attention_shapes']}=={(96,32),(128,64)}
    assert {s['L'] for s in r['attention_shapes']}=={2048,4096}
    for d,rope in [(96,32),(128,64)]:
        x=t.fixture(spec(2,3,d,64,rope_width=rope))
        np.testing.assert_array_equal(x['k'][...,d-rope:],x['kq'][...,d-rope:])
        assert np.any(x['k'][...,:d-rope]!=x['kq'][...,:d-rope])
    assert {1792,4096,768,256,8192} <= {s[n] for s in r['gemm_shapes'] for n in ('K','N')}


def test_spread_zero_and_nonfinite_fail_closed():
    ref=(np.ones(2),)*3;a=tuple(x+.25 for x in ref)
    g=t.back.gate(a,ref,dict(edge=tuple(x+.5 for x in ref),over=tuple(x+.50001 for x in ref)))
    assert g['candidates']['edge']['verdict']=='GREEN'
    assert g['candidates']['over']['verdict']=='RED'
    for bad in (tuple(x+1e-14 for x in ref),(np.full(2,np.nan),)*3,(np.full(2,np.inf),)*3):
        assert t.back.gate(ref,ref,dict(bad=bad))['candidates']['bad']['verdict']=='RED'
    assert not t.front.flips(np.zeros(200,bool),np.arange(200)<2)['passes']
    assert t.front.flips(np.zeros(200,bool),np.arange(200)<1)['passes']


@pytest.mark.parametrize('trans_b',[False,True])
def test_fp64_gemm_oracle(trans_b):
    a=np.arange(21,dtype=np.float32).reshape(3,7)/13
    b=np.arange(35,dtype=np.float32).reshape(7,5)/11
    bb=b.T if trans_b else b
    expected=np.array([[sum(float(a[i,k])*float(b[k,j]) for k in range(7)) for j in range(5)] for i in range(3)])
    np.testing.assert_allclose(t.gemm_reference(a,bb,trans_b),expected,rtol=1e-15)


def test_original_kernel_file_and_bf16_gemm_branch_unchanged():
    r=t.registration()
    assert t.sha(t.ROOT/'tensor_cuda/src/kernels.cu')==r['baseline']['tensor_cuda/src/kernels.cu']
    old=(t.ART/'baseline/matmul.cu').read_text();new=(t.ROOT/'tensor_cuda/src/matmul.cu').read_text()
    start=old.index('  } else if (a.dtype == DType::Float16)')
    assert old[start:]==new[new.index('  } else if (a.dtype == DType::Float16)'):]


@pytest.fixture(scope='module')
def native():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    return t.load_fork(require_manifest=False)


def test_native_cpu_setters_defaults_and_thread_local(native):
    native.set_tf32_gemm(False)
    assert native.get_tf32_gemm() is False
    assert native.bp_kernel_2_get_variant()=='a' and native.bp_kernel_4_get_variant()=='a'
    native.set_tf32_gemm(True)
    def worker():
        return native.get_tf32_gemm(),native.bp_kernel_2_get_variant(),native.bp_kernel_4_get_variant()
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(worker).result()==(False,'a','a')
    assert native.get_tf32_gemm() is True
    native.set_tf32_gemm(False)
    for setter,getter,variant in [(native.bp_kernel_2_set_variant,native.bp_kernel_2_get_variant,'g1_tf32'),
                                  (native.bp_kernel_4_set_variant,native.bp_kernel_4_get_variant,'h_tf32')]:
        setter(variant);assert getter()==variant
        with pytest.raises(RuntimeError):setter('unknown')
        assert getter()==variant
        setter('a')


def test_fork_loader_rejects_an_already_imported_external_engine(monkeypatch):
    monkeypatch.setitem(sys.modules,'tensor_cuda._tensor_cuda',SimpleNamespace(__file__='/outside/_tensor_cuda.so'))
    with pytest.raises(ValueError,match='different engine'):
        t.load_fork(require_manifest=False)


@pytest.mark.parametrize('env_value,expected',[('',False),('0',False),('1',True),('true',False)])
def test_native_env_initialization(env_value,expected):
    code="import sys;sys.path.insert(0,'scripts');import pt_tf32_1 as t;c=t.load_fork(False);print(c.get_tf32_gemm());c.set_tf32_gemm(False);print(c.get_tf32_gemm())"
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',TC_TF32_GEMM=env_value,PYTHONDONTWRITEBYTECODE='1')
    p=subprocess.run([sys.executable,'-B','-c',code],cwd=t.ROOT,env=env,capture_output=True,text=True,timeout=20)
    assert p.returncode==0,p.stderr
    assert p.stdout.strip().splitlines()==[str(expected),'False']


def test_native_contract_rejects_cpu_before_gpu_access(native):
    # Actual native guards, reached via CPU buffers that do not call the
    # engine's GPU-dependent from_host. Same assertions live in the C++ test.
    p=subprocess.run([str(t.ART/'pt_tf32_host_contract')],cwd=t.ROOT,
                     env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=20)
    assert p.returncode==0,p.stderr
    assert p.stdout.strip()=='HOST_CONTRACT: 6 expected rejections; no CUDA device operations'


def test_receipts_create_only_and_gpu_command_blocked(tmp_path):
    path=tmp_path/'receipt.json';t.create_json(path,dict(x=1))
    with pytest.raises(FileExistsError):t.create_json(path,dict(x=2))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    p=subprocess.run([sys.executable,'-B','scripts/pt_tf32_1.py','attention','--lead-gpu'],cwd=t.ROOT,
                     env=env,capture_output=True,text=True,timeout=20)
    assert p.returncode==2 and 'BLOCKED' in p.stderr
