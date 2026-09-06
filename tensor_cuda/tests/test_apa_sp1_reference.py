import hashlib
import json
import ast
import math
from pathlib import Path
import re

import numpy as np
import pytest

import apa_sp1_reference as ref

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'artifacts/apa_sp1'
REG = json.loads((ART / 'registration.json').read_text())


def test_registration_and_original_bytes_pinned():
    expected = (ART / 'registration.sha256').read_text().split()[0]
    assert hashlib.sha256((ART / 'registration.json').read_bytes()).hexdigest() == expected
    for path, pin in REG['pins'].items():
        original = (ART / pin['snapshot']).read_bytes()
        assert hashlib.sha256(original).hexdigest() == pin['sha256']
        current = (ROOT / path).read_bytes()
        stripped = re.sub(rb'// APA_SP1_ADDITION_BEGIN[^\n]*\n.*?// APA_SP1_ADDITION_END[^\n]*\n',
                          b'', current, flags=re.S)
        assert stripped == original, path


def test_registered_zthr_has_the_original_host_launcher_fp32_bits():
    path=ROOT/'tensor_cuda/tensor_cuda/quant.py'
    fn=next(n for n in ast.parse(path.read_text()).body
            if isinstance(n,ast.FunctionDef) and n.name=='_norm_ppf')
    scope={'math':math}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),scope)
    expected=np.float32(scope['_norm_ppf'](1-REG['refine_percentile']))
    assert expected.tobytes()==np.float32(REG['zthr']).tobytes()


def test_q1_counterexample_and_buffer_loophole():
    result = ref.counterexample(REG['zthr'])
    assert result[0]['mask'][0] != result[1]['mask'][0]
    assert abs(result[0]['output'][0] - result[1]['output'][0]) > 1e-3


def test_q1_wrong_early_commit_is_observable_for_either_suffix():
    # Pin the cost of the wrong first-key decision, holding each suffix fixed.
    exact=np.array([2,0,0,-10],np.float32)
    v=np.array([[1],[0],[0],[0]],np.float32)
    for last in [0,10]:
        bulk=np.array([1,0,0,last],np.float32)
        mask,_=ref.zmask(bulk,REG['zthr'])
        correct=ref.dense_scores(np.where(mask,exact,bulk),v)
        wrong_mask=mask.copy();wrong_mask[0]=~wrong_mask[0]
        wrong=ref.dense_scores(np.where(wrong_mask,exact,bulk),v)
        assert float(np.abs(correct-wrong).max())>1e-3


def test_q2_output_is_not_zscore_equivalent_on_the_registered_witness():
    q=np.array([1],np.float32)
    k=np.array([[2],[0],[0],[-10]],np.float32)
    kq=np.array([[1],[0],[0],[10]],np.float32)
    v=np.array([[1],[0],[0],[0]],np.float32)
    sp,sm=ref.row_single_visit(q,k,kq,v,1.,0.)
    old,zm,_=ref.row_two_pass(q,k,kq,v,1.,REG['zthr'])
    assert sm[0] and not zm[0]
    assert float(np.abs(sp-old).max())>1e-3


def test_zero_query_ties_refine_all_and_sink_is_denominator_only():
    q=np.zeros(33,np.float32);k=np.ones((7,33),np.float32)
    kq=-k;v=np.ones((7,5),np.float32)
    got,mask=ref.row_single_visit(q,k,kq,v,1.,0.,0.)
    np.testing.assert_array_equal(mask,np.ones(7,bool))
    np.testing.assert_allclose(got,np.full(5,7/8,dtype=np.float32),atol=1e-7,rtol=0)


@pytest.mark.parametrize('causal', [False, True])
@pytest.mark.parametrize('D,VD,H,KVH,L,S', [(64,64,4,4,9,9), (128,32,8,2,7,29),
                                        (512,17,16,1,3,19), (33,7,4,2,2,11)])
def test_zscore_pinned_against_original_reference(causal, D, VD, H, KVH, L, S):
    rng = np.random.default_rng(701 + D + S)
    q = rng.standard_normal((1,H,L,D), dtype=np.float32) * 0.1
    k = rng.standard_normal((1,KVH,S,D), dtype=np.float32) * 0.1
    kq = k + rng.standard_normal(k.shape, dtype=np.float32) * 0.01
    v = rng.standard_normal((1,KVH,S,VD), dtype=np.float32) * 0.1
    sinks = rng.standard_normal(H, dtype=np.float32) * 0.2
    got = ref.tensor_reference(q,k,kq,v,1/np.sqrt(D),REG['zthr'],causal,sinks)
    old = ref.original_numpy_reference('ref_selective_sink')
    expected = old(q,k,kq,v,sinks,1/np.sqrt(D),REG['zthr'],causal)
    np.testing.assert_allclose(got,expected,atol=1e-3,rtol=1e-3)


@pytest.mark.parametrize('delta', [0, 0.5, 2, 4])
def test_prefix_single_visit_equivalence_and_buffered_zscore(delta):
    rng = np.random.default_rng(91)
    for D in [1,33,64,128,512]:
        q = rng.standard_normal(D,dtype=np.float32)
        k = rng.standard_normal((37,D),dtype=np.float32)
        kq = k + 0.1*rng.standard_normal(k.shape,dtype=np.float32)
        v = rng.standard_normal((37,7),dtype=np.float32)
        bulk, exact = kq@q/np.sqrt(D), k@q/np.sqrt(D)
        mask = ref.prefix_mask(bulk,delta)
        want = ref.dense_scores(np.where(mask,exact,bulk),v,sinks=0.3)
        got, sm = ref.row_single_visit(q,k,kq,v,1/np.sqrt(D),delta,0.3)
        np.testing.assert_array_equal(mask,sm)
        np.testing.assert_allclose(got,want,atol=1e-3,rtol=1e-3)
        buffered,bm=ref.row_buffered_one_read(q,k,kq,v,1/np.sqrt(D),REG['zthr'])
        original,om,_=ref.row_two_pass(q,k,kq,v,1/np.sqrt(D),REG['zthr'])
        np.testing.assert_array_equal(bm,om)
        np.testing.assert_allclose(buffered,original,atol=1e-3,rtol=1e-3)


def test_suffix_perturbation_never_invalidates_skipped_prefix():
    rng = np.random.default_rng(REG['data']['cpu_seed'])
    # Independent concrete suffix attacks, not only a test of identical prefixes.
    prefix = rng.standard_normal((10000,17),dtype=np.float32)
    for delta in [0,0.5,2,4]:
        pm = ref.prefix_mask(prefix,delta)
        for amplitude in [0,1,100]:
            suffix = rng.standard_normal((10000,13),dtype=np.float32)*amplitude
            whole = np.concatenate([prefix,suffix],axis=1)
            mask = ref.prefix_mask(whole,delta)
            np.testing.assert_array_equal(mask[:,:17],pm)
            final_required = prefix >= whole.max(-1,keepdims=True)-np.float32(delta)
            assert not np.any(final_required & ~pm)


def test_ties_signed_rule_order_dependence_and_dense_non_guarantee():
    bulk=np.array([[-5,-5,0,-1,2,2]],dtype=np.float32)
    np.testing.assert_array_equal(ref.prefix_mask(bulk,0), [[True,True,True,False,True,True]])
    a=np.array([[1,0,10]],dtype=np.float32)
    perm=np.array([2,1,0])
    assert not np.array_equal(ref.prefix_mask(a,0)[:,perm],ref.prefix_mask(a[:,perm],0))
    # A skipped bulk key can have an arbitrarily large EXACT logit.
    b=np.array([[2,0]],dtype=np.float32)
    e=np.array([[2,100]],dtype=np.float32)
    assert not ref.prefix_mask(b,0)[0,1] and e[0,1] == e.max()


def test_causal_gqa_sink_tensor_matches_literal_visits():
    rng=np.random.default_rng(142)
    B,H,KVH,L,S,D,VD=2,4,2,3,11,33,7
    q=rng.standard_normal((B,H,L,D),dtype=np.float32)
    k=rng.standard_normal((B,KVH,S,D),dtype=np.float32)
    kq=k+0.1*rng.standard_normal(k.shape,dtype=np.float32)
    v=rng.standard_normal((B,KVH,S,VD),dtype=np.float32)
    sinks=np.array([-10,0,3,10],dtype=np.float32)
    for causal in [False,True]:
        got=ref.tensor_reference(q,k,kq,v,1/np.sqrt(D),0.5,causal,sinks,'prefix')
        for b in range(B):
            for h in range(H):
                for i in range(L):
                    n=S-L+i+1 if causal else S
                    kh=h//(H//KVH)
                    want,_=ref.row_single_visit(q[b,h,i],k[b,kh,:n],kq[b,kh,:n],v[b,kh,:n],1/np.sqrt(D),0.5,sinks[h])
                    np.testing.assert_allclose(got[b,h,i],want,atol=1e-3,rtol=1e-3)
