"""Known existing failures: all asserts express the mathematical pass criterion.
Prior art: bottom-right convention (63960a4, Dao 2023 v2.1); independent
counterexample construction is this review's contribution. Expected RED on
current sources; never xfail/skip/suppress. Run suites independently.
"""
import argparse,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tensor_cuda'),str(ROOT/'apa_cuda')]
from apa_mf1rev_oracle import attention
from apa_mf1rev_runtime import preflight


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',required=True,choices=['extension','finite_bias'])
    args=p.parse_args();L,S,D=3,9,64
    preflight(ROOT)
    if args.case=='extension':
        import torch
        from apa_attention import apa_quant_attention
        import apa_attention.ops as ops
        assert Path(ops.__file__).resolve()==ROOT/'apa_cuda/apa_attention/ops.py'
        q=np.zeros((1,1,L,D),np.float32);k=np.zeros((1,1,S,D),np.float32)
        v=np.broadcast_to(np.arange(S)[None,None,:,None],(1,1,S,D)).astype(np.float32).copy()
        tensors=[torch.tensor(a,device='cuda') for a in (q,k,v)]
        out=apa_quant_attention(*tensors,refine_percentile=1.,is_causal=True).detach().cpu().numpy()
    else:
        from probe_quant import engine
        tc,Q=engine()
        q=np.zeros((1,1,L,D),np.float32);q[...,0]=1
        k=np.zeros((1,1,S,D),np.float32);k[0,0,7,0]=80008
        v=np.zeros_like(k);v[0,0,7,:]=1
        out=Q.apa_quant_attention(*(tc.tensor(a) for a in (q,k,v)),bulk_bits=8,
                                  refine_percentile=1.,is_causal=True).numpy()
    expected=attention(q,k,v,.125)
    print(args.case,'expected row0',expected[0,0,0,:4],'actual',out[0,0,0,:4],flush=True)
    assert np.max(np.abs(out-expected)) <= 2e-5,'known existing mathematical-definition failure'

if __name__=='__main__':main()
