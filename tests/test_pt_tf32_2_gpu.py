"""Lead-only regressions. Prior art: standard softmax/VJP identities,
NumPy/pytest and PT-TF32-1 (2026), taken. Ours: exact singleton and selected
predicate regressions, plus actual cuBLASLt HMMA capability receipts.
"""
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pytest
import pt_tf32_2 as t
from test_pt_tf32_gpu import c,tensors


@pytest.mark.parametrize('widths',[(19,17),(96,64),(128,127)])
def test_single_visible_key_exact_zero_score_gradients(c,widths):
    D,VD=widths
    s=dict(B=2,H=4,KVH=2,L=1,S=1,D=D,VD=VD,causal=True,scale=float(np.float32(1/np.sqrt(D))),zthr=1.)
    x=t.fixture(s);d=tensors(c,x);args=[d[n] for n in ('q','k','kq','v')]
    f=c.apa_selective_fwd_train_variant(*args,s['scale'],1.,True,'h_tf32')
    np.testing.assert_array_equal(f[0].numpy(),np.repeat(x['v'],2,axis=1))
    g=c.apa_selective_bwd_variant(*args,d['dO'],f[1],f[2],f[0],s['scale'],True,'g1_tf32')
    np.testing.assert_array_equal(g[0].numpy(),0);np.testing.assert_array_equal(g[1].numpy(),0)
    # dV remains a TF32 product. Compare its rounded operands exactly enough
    # to catch grouped-head omission without pretending full FP32 products.
    rounded=t.old.tf32_round(x['dO']).astype(np.float64).reshape(2,2,2,1,VD).sum(2)
    np.testing.assert_allclose(g[2].numpy(),rounded,rtol=2e-6,atol=2e-7)


def test_no_selected_pairs_have_exact_zero_dk(c):
    s=dict(B=1,H=4,KVH=2,L=15,S=17,D=19,VD=17,causal=True,scale=.2,zthr=100.)
    x=t.fixture(s);d=tensors(c,x);args=[d[n] for n in ('q','k','kq','v')]
    f=c.apa_selective_fwd_train_variant(*args,.2,100.,True,'h_tf32')
    thr=c.tensor(np.full((1,4,15),1e10,np.float32),'cuda',False)
    g=c.apa_selective_bwd_variant(*args,d['dO'],f[1],thr,f[0],.2,True,'g1_tf32')
    np.testing.assert_array_equal(g[1].numpy(),0)
    assert np.isfinite(g[0].numpy()).all() and np.linalg.norm(g[0].numpy())>0


def test_dk_selection_discontinuity_is_below_tf32_budget(c):
    s=dict(B=1,H=4,KVH=2,L=512,S=512,D=96,VD=64,rope_width=32,causal=True,
           scale=float(np.float32(1/np.sqrt(96))),zthr=float(np.float32(1.036433389)))
    x=t.fixture(s);ref=t.num.forward_reference(x,s)
    state={n:ref[n].astype(np.float32) for n in ('out','lse','thr')};d=tensors(c,x)
    f={n:c.tensor(state[n],'cuda',False) for n in state}
    actual=c.apa_selective_bwd_variant(*[d[n] for n in ('q','k','kq','v','dO')],f['lse'],f['thr'],f['out'],s['scale'],True,'g1_tf32')
    expected=t.num.backward_reference(x,state,s)
    for n,v in zip(t.NAMES,actual):
        result=t.num.metric_gate(v.numpy(),expected[n],t.num.bounds(t.registration(),s,n)['bound'])
        assert result['verdict']=='GREEN', (n,result)


@pytest.mark.parametrize('trans_b',[False,True])
def test_model_geometry_reports_tf32_tensor_core_algorithm(c,trans_b):
    rng=np.random.default_rng(9)
    a=rng.normal(size=(128,256)).astype(np.float32);b=rng.normal(size=(64,256) if trans_b else (256,64)).astype(np.float32)
    c.set_tf32_gemm(True)
    try:
        value=c.matmul(c.tensor(a,'cuda',False),c.tensor(b,'cuda',False),1.,trans_b).numpy()
        algo,flags,workspace=c.get_tf32_gemm_info()
        assert algo>=0 and flags & 0x40202==0x40202 and 0<=workspace<=32*1024*1024
        assert t.back.metric(value,a.astype(np.float64)@(b.T if trans_b else b).astype(np.float64))['relative_L2']<=1e-3
    finally:c.set_tf32_gemm(False)
