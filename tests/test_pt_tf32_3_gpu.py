"""Lead-only new checks; inherited legacy assertions remain unchanged.
Prior art: FP64 dense VJP, Higham (2002) TF32 rounding budget, NVIDIA Lt
(2024), taken. Ours: complete saved-state edge and actual dispatch receipts.
"""
from pathlib import Path
import os
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pytest
import pt_tf32_3 as t
from test_pt_tf32_gpu import c,tensors


@pytest.mark.parametrize('L,S',[(17,31),(33,35)])
def test_legacy_edge_independent_fp64_budget(c,L,S):
    s=dict(B=1,H=4,KVH=2,L=L,S=S,D=96,VD=64,causal=True,scale=float(np.float32(1/np.sqrt(96))),zthr=-100.)
    x=t.fixture(s);x['kq']=x['k'].copy();d=tensors(c,x)
    f=c.apa_selective_fwd_train_variant(*[d[n] for n in ('q','k','kq','v')],s['scale'],s['zthr'],True,'h_tf32')
    gradients=c.apa_selective_bwd_variant(*[d[n] for n in ('q','k','kq','v','dO')],f[1],f[2],f[0],s['scale'],True,'g1_tf32')
    ref=t.num.backward_reference(x,t.num.forward_reference(x,s),s)
    for name,v in zip(t.NAMES,gradients):
        result=t.num.metric_gate(v.numpy(),ref[name],t.num.bounds(t.registration(),s,name)['bound'])
        assert result['verdict']=='GREEN',(name,result)


@pytest.mark.parametrize('tb',[False,True])
def test_dispatch_records_actual_pointer_alignment_and_stream_ordered_output(c,tb):
    a=c.tensor(np.ones((128,256),np.float32),'cuda',False)
    b=c.tensor(np.ones((64,256) if tb else (256,64),np.float32),'cuda',False)
    c.set_tf32_gemm(True)
    try:
        y=c.matmul(a,b,1.,tb);r=c.get_tf32_gemm_dispatch()
        np.testing.assert_array_equal(y.numpy(),256.)
        assert r['output_stream_ordered']==1 and r['device_selected']==1 and r['algorithm_id']>=0
        assert r['fast_tf32']==1 and r['transa']==int(tb) and r['transb']==0
        assert r['numerical_flags']&0x40202==0x40202
        assert min(r['alignment_a'],r['alignment_b'],r['alignment_c'])>=16
        if r['mathmode_query_status']==0:assert r['mathmode_impl']==1
        if os.environ.get('PT_TF32_GENERATION')=='4':
            import pt_tf32_4
            assert pt_tf32_4.dispatch_ok(r,128,64,256,tb)
            assert r['mathmode_query_status']==-1 and r['mathmode_query_attempted']==0
    finally:c.set_tf32_gemm(False)
