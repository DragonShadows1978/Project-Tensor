"""CPU adversarial checks of the GPU report's full-reference/matching inputs."""
from pathlib import Path
import json
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from apa_sp1_gpu import arrays,all_references,metrics
from apa_sp1_cpu import registration
import apa_sp1_reference as ref


def test_gpu_harness_reference_counts_all_causal_gqa_rows_and_values():
    bad=metrics(np.array([np.nan,np.inf],np.float32),np.ones(2,np.float32))
    assert bad['nonfinite_got']==2 and bad['relative_frobenius'] is None
    json.dumps(bad,allow_nan=False)  # failed numerics must remain reportable
    reg=registration()
    for causal in [False,True]:
        shape=dict(B=2,H=4,KVH=2,L=3,S=11,D=33,VD=7,causal=causal)
        q,k,kq,v=arrays(shape,119)
        masks=np.zeros((2,4,3,11),dtype=np.uint8)
        # Deliberately force a fully-refined diagnostic mask to ensure the
        # reporter uses actual GPU diagnostics, not its own predicted mask.
        for i in range(3):masks[:,:,i,:11-3+i+1 if causal else 11]=1
        sinks=np.array([-10,0,3,8],np.float32)
        sp,old,dense,counts=all_references(q,k,kq,v,shape,0.5,reg['zthr'],masks,sinks)
        for got,rule,threshold in [(sp,'prefix',0.5),(old,'zscore',reg['zthr']),(dense,'dense',0.5)]:
            expected=ref.tensor_reference(q,k,kq,v,1/np.sqrt(33),threshold,causal,sinks,rule)
            np.testing.assert_allclose(got,expected,atol=1e-6,rtol=1e-6)
        assert counts['valid']==2*4*(9+10+11 if causal else 3*11)
        assert counts['sp_fraction']==1
        assert counts['cpu_gpu_mask_disagreements']>0
        assert not counts['matched']
