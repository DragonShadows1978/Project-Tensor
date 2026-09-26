"""Lead-only GPU unit suite; collection is CPU-only, execution requires a slot.
Prior art: BP-KERNEL-2/3/4 reference tests (2026), NumPy/pytest, taken.
Ours: TF32 dtype, precision, saved-mode, checkpoint and boundary regressions.
PT-TF32-2 retains all tolerances. Its corrected CPU operand model and new
source seal are used; independent FP64 certification is a separate gate.
"""
import os
from pathlib import Path
import sys

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pytest
# PT-TF32-3 uses the same assertions/tolerances against its own immutable seal.
# Prior art: PT-TF32-2 versioned registration (2026), taken; no gate suppression.
if os.environ.get('PT_TF32_GENERATION')=='4':
    import pt_tf32_4 as t
elif os.environ.get('PT_TF32_GENERATION')=='3':
    import pt_tf32_3 as t
else:
    import pt_tf32_2 as t


@pytest.fixture(scope='module')
def c():
    if os.environ.get('PT_TF32_LEAD_GPU')!='1' or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        pytest.fail('BLOCKED: lead-only suite; explicit idle GPU slot required')
    c=t.load_fork()
    yield c
    c.set_tf32_gemm(False);c.bp_kernel_2_set_variant('a');c.bp_kernel_4_set_variant('a')


def tensors(c,x,dtype='float32',grad=False):
    return {n:c.tensor(np.ascontiguousarray(v),'cuda',grad and n in ('q','k','v')).astype(dtype) for n,v in x.items()}


@pytest.mark.parametrize('causal',[False,True])
@pytest.mark.parametrize('lengths',[(1,1),(15,17),(16,16),(17,31),(33,35)])
@pytest.mark.parametrize('widths',[(1,1),(19,17),(96,64),(128,127)])
def test_native_padding_grouped_heads_and_fp32_storage(c,causal,lengths,widths):
    D,VD=widths;L,S=lengths
    s=dict(B=1,H=4,KVH=2,L=L,S=S,D=D,VD=VD,causal=causal,scale=float(np.float32(1/np.sqrt(D))),zthr=-100.)
    x=t.fixture(s);x['kq']=x['k'].copy() # all-selected: separate arithmetic from selection flips
    d=tensors(c,x)
    args=[d[n] for n in ('q','k','kq','v')]
    f=c.apa_selective_fwd_train_variant(*args,s['scale'],s['zthr'],causal,'h_tf32')
    actual=dict(zip(('out','lse','thr'),[v.numpy() for v in f]))
    ref=t.front.reference(x,s)
    assert all(v.dtype=='float32' for v in f)
    for n in ('out','lse'):
        assert t.back.metric(actual[n],ref[n])['relative_L2']<=3e-3
    g=c.apa_selective_bwd_variant(*args,d['dO'],f[1],f[2],f[0],s['scale'],causal,'g1_tf32')
    expected=t.tiled_backward_model(x,actual,s)
    # PT-TF32-4 prior art: NVIDIA TF32 (2020), CUDA 12.6 __expf (2024),
    # Higham (2002) interval/accumulation bounds. Taken rounding model;
    # ours: input-only per-element dV allowance at probability midpoints.
    # Applies to every padding case, never to chosen failing coordinates.
    # All older generation invocations retain their historical assertions.
    dv_model=t.num.edge_dv_budget(x,actual,s,t.registration()) if os.environ.get('PT_TF32_GENERATION')=='4' else None
    for n,v in zip(t.NAMES,g):
        assert v.dtype=='float32'
        if n=='dV' and dv_model is not None:
            # Also bind the interval center to the independent legacy model.
            np.testing.assert_allclose(dv_model['reference'],expected[n],rtol=2e-14,atol=2e-14)
            result=t.num.edge_dv_gate(v.numpy(),dv_model)
            assert result['verdict']=='GREEN',result
        else:
            np.testing.assert_allclose(v.numpy(),expected[n],rtol=3e-3,atol=2e-5)


def test_native_rejects_dtype_mixes_and_bad_geometry(c):
    s=dict(B=1,H=2,KVH=1,L=2,S=3,D=19,VD=17,causal=True,scale=.2,zthr=0.)
    x=t.fixture(s);d=tensors(c,x);b=tensors(c,x,'bfloat16')
    for vals in ([b[n] for n in ('q','k','kq','v')],
                 [d['q'],d['k'],b['kq'],d['v']]):
        with pytest.raises(RuntimeError,match='same CUDA FP32'):
            c.apa_selective_fwd_train_variant(*vals,.2,0.,True,'h_tf32')
    vals=[d[n] for n in ('q','k','kq','v')]
    for scale,zthr in [(float('nan'),0.),(.2,float('inf'))]:
        with pytest.raises(RuntimeError):
            c.apa_selective_fwd_train_variant(*vals,scale,zthr,True,'h_tf32')
    f=c.apa_selective_fwd_train_variant(*vals,.2,0.,True,'h_tf32')
    with pytest.raises(RuntimeError):
        c.apa_selective_bwd_variant(*vals,b['dO'],f[1],f[2],f[0],.2,True,'g1_tf32')


@pytest.mark.parametrize('trans_b',[False,True])
def test_matmul_mode_is_captured_for_backward_and_bf16_unchanged(c,trans_b):
    rng=np.random.default_rng(42)
    a=rng.normal(size=(32,64)).astype(np.float32)
    b=rng.normal(size=(48,64) if trans_b else (64,48)).astype(np.float32)
    do=c.tensor(rng.normal(size=(32,48)).astype(np.float32),'cuda',False)
    results=[]
    for reset in (False,True):
        A=c.tensor(a,'cuda',True);B=c.tensor(b,'cuda',True)
        c.set_tf32_gemm(True);y=c.matmul(A,B,1.,trans_b)
        if reset:c.set_tf32_gemm(False)
        (y*do).sum().backward()
        assert c.get_tf32_gemm()==(not reset)
        results.append((A.grad.numpy(),B.grad.numpy()))
    for a1,a2 in zip(*results):np.testing.assert_array_equal(a1,a2)
    A=c.tensor(a,'cuda',False).astype('bfloat16');B=c.tensor(b,'cuda',False).astype('bfloat16')
    c.set_tf32_gemm(False);before=c.matmul(A,B,1.,trans_b).numpy()
    c.set_tf32_gemm(True);after=c.matmul(A,B,1.,trans_b).numpy()
    np.testing.assert_array_equal(before,after)
    c.set_tf32_gemm(False)


def test_checkpoint_initial_inference_and_replay_match_tf32_route(c):
    s=dict(B=1,H=2,KVH=1,L=17,S=19,D=96,VD=64,causal=True,scale=.1,zthr=1.)
    x=t.fixture(s);d=tensors(c,x,grad=True)
    c.bp_kernel_2_set_variant('g1_tf32');c.bp_kernel_4_set_variant('h_tf32')
    args=[d[n] for n in ('q','k','kq','v')]
    try:
        a=c.apa_selective_attention(*args,.1,1.,True)
        b=c.apa_selective_train(*args,.1,1.,True)
        np.testing.assert_array_equal(a.numpy(),b.numpy())
        def block(q,k,v):
            fn=c.apa_selective_train if c.is_grad_enabled() else c.apa_selective_attention
            return fn(q,k,d['kq'],v,.1,1.,True)
        result=c.checkpoint(block,[d['q'],d['k'],d['v']])
        np.testing.assert_array_equal(a.numpy(),result.numpy())
        (result*d['dO']).sum().backward()
        for n in ('q','k','v'):
            assert d[n].grad is not None and np.isfinite(d[n].grad.numpy()).all()
    finally:
        c.bp_kernel_2_set_variant('a');c.bp_kernel_4_set_variant('a')


def test_backward_variant_capture_survives_scope_restoration(c):
    s=dict(B=1,H=2,KVH=1,L=3,S=5,D=19,VD=17,causal=True,scale=.2,zthr=0.)
    x=t.fixture(s);d=tensors(c,x,grad=True)
    c.bp_kernel_2_set_variant('g1_tf32');c.bp_kernel_4_set_variant('h_tf32')
    y=c.apa_selective_train(*[d[n] for n in ('q','k','kq','v')],.2,0.,True)
    c.bp_kernel_2_set_variant('g1');c.bp_kernel_4_set_variant('h')
    try:
        # If the closure rereads the restored BF16-only mode, this throws.
        (y*d['dO']).sum().backward()
        assert d['q'].grad.dtype=='float32'
    finally:
        c.bp_kernel_2_set_variant('a');c.bp_kernel_4_set_variant('a')


@pytest.mark.parametrize('broadcast',[False,True])
@pytest.mark.parametrize('trans_b',[False,True])
def test_tf32_batched_gemm_strides_broadcast_alpha(c,broadcast,trans_b):
    rng=np.random.default_rng(123)
    a=rng.normal(size=(2,17,19)).astype(np.float32)
    bs=(13,19) if trans_b else (19,13)
    b=rng.normal(size=bs if broadcast else (2,*bs)).astype(np.float32)
    A=c.tensor(a,'cuda',False);B=c.tensor(b,'cuda',False)
    alpha=float(np.float32(.3))
    c.set_tf32_gemm(True)
    try:
        got=c.matmul(A,B,alpha,trans_b).numpy()
        # PT-TF32-2 ragged WMMA dispatch (NVIDIA WMMA, 2020); numeric
        # thresholds below are unchanged from the original regression.
        assert c.get_tf32_gemm_info()==(-2,0x40202,0)
        b64=b.astype(np.float64)
        ref=alpha*(a.astype(np.float64)@(b64.swapaxes(-2,-1) if trans_b else b64))
        assert got.dtype==np.float32 and got.shape==(2,17,13)
        assert t.back.metric(got,ref)['relative_L2']<=1e-3
    finally:
        c.set_tf32_gemm(False)
