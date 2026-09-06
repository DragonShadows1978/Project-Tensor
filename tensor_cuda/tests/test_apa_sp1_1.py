import ctypes
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pytest
import apa_sp1_reference as ref
import apa_sp1_1_reference as sp

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT/'artifacts/apa_sp1'
sys.path.insert(0, str(ROOT/'scripts'))
from apa_sp1_gpu import arrays


def test_registration_and_every_existing_kernel_body_pinned():
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest(ART/'registration_sp1_1.json') == (ART/'registration_sp1_1.sha256').read_text().split()[0]
    reg = json.loads((ART/'registration_sp1_1.json').read_text())
    assert reg['parent_registration_sha256'] == digest(ART/'registration.json')
    for name, pin in reg['pins'].items():
        assert digest(ROOT/pin['snapshot']) == pin['sha256']
    for name, sha in reg['sp1_gpu_receipts'].items():
        assert digest(ROOT/name) == sha
    before = (ROOT/reg['pins']['tensor_cuda/src/kernels.cu']['snapshot']).read_text()
    now = (ROOT/'tensor_cuda/src/kernels.cu').read_text()
    # Include full signature/body, not only baseline attention kernels.
    for start in (m.start() for m in re.finditer(r'__global__\s+void\s+', before)):
        opening = before.index('{', start)
        depth, end = 1, opening+1
        while depth:
            depth += (before[end]=='{') - (before[end]=='}')
            end += 1
        assert before[start:end] in now, before[start:opening]
    assert 'TC_APA_SPLITK_PART_KEYS = 128 * 16;' in now
    assert sp.PART_KEYS == 128*16


def test_random_partition_monotonicity_and_online_merge():
    reg = json.loads((ART/'registration_sp1_1.json').read_text())['part_A']['cpu_pin']
    rng = np.random.default_rng(reg['seed'])
    for trial in range(reg['random_trials']):
        S = int(rng.integers(1,reg['max_keys']+1))
        P = int(rng.integers(1,reg['max_parts']+1))
        bulk = rng.standard_normal(S,dtype=np.float32)
        exact = bulk + rng.standard_normal(S,dtype=np.float32)*np.float32(.1)
        values = rng.standard_normal((S,reg['value_dim']),dtype=np.float32)
        if trial % 7 == 0: bulk[:] = 0  # inclusive ties
        delta = reg['deltas'][trial%len(reg['deltas'])]
        # Noncontiguous random partitions preserve original order in each subset.
        labels = rng.integers(P,size=S)
        parts = [np.flatnonzero(labels==p) for p in range(P)]
        if trial % 2:
            parts = np.split(np.arange(S),np.sort(rng.integers(0,S+1,size=P-1)))
        sink = np.float32(rng.normal()*10) if trial%3 else None
        got,mask = sp.partition_online(bulk,exact,values,delta,parts,sink)
        global_mask = ref.prefix_mask(bulk,delta)
        assert not np.any(global_mask & ~mask)
        assert not np.any((bulk>=bulk.max()-np.float32(delta)) & ~mask)
        want = ref.dense_scores(np.where(mask,exact,bulk),values,sinks=sink)
        np.testing.assert_allclose(got,want,atol=reg['atol'],rtol=reg['rtol'])


def test_extra_refinement_changes_output_and_cannot_be_called_work_only():
    bulk=np.array([2,0],np.float32);exact=np.array([2,10],np.float32)
    values=np.array([[0],[1]],np.float32)
    got,mask=sp.partition_online(bulk,exact,values,0,[[0],[1]],sink=0)
    global_mask=ref.prefix_mask(bulk,0)
    assert mask.tolist()==[True,True] and global_mask.tolist()==[True,False]
    global_out=ref.dense_scores(np.where(global_mask,exact,bulk),values,sinks=0)
    assert float(abs(got-global_out).max())>.8


@pytest.mark.parametrize('S',[1,127,128,129,2047,2048,2049,4097])
def test_fixed_partition_mask_matches_literal_with_ragged_tail(S):
    rng=np.random.default_rng(S)
    b=rng.standard_normal(S,dtype=np.float32);v=rng.standard_normal((S,3),dtype=np.float32)
    parts=[np.arange(j,min(S,j+sp.PART_KEYS)) for j in range(0,S,sp.PART_KEYS)]
    _,mask=sp.partition_online(b,b,v,.5,parts)
    np.testing.assert_array_equal(mask,sp.partition_mask(b,.5))


def test_baseline_failure_specific_keys_and_output_magnitude():
    reg=json.loads((ART/'registration.json').read_text())
    shape=reg['shapes'][10]
    assert shape['id']=='prefill_s2048_d64_c1_h4_kv4'
    q,k,kq,v=arrays(shape,np.random.SeedSequence([20260907,10]))
    # Use exactly the original chunk sizes to preserve BLAS reference order.
    for h,row,key,expected_thr_bits in [(1,1271,1199,0x3f9db1d7),(3,1137,737,0x3fc7c232)]:
        first=row//32*32;ri=row-first;qr=q[0,h,first:first+32]
        lengths=np.arange(first,first+32)+1
        bulk=qr@kq[0,h].T*np.float32(.125)
        cb=sp.cuda_bulk(qr,kq[0,h],.125)
        oldmask,oldthr=ref.zmask(bulk,reg['zthr'],lengths)
        mask,thr=sp.cuda_zmask(cb,reg['zthr'],lengths)
        assert oldmask[ri,key] and not mask[ri,key]
        assert int(thr[ri].view(np.uint32))==expected_thr_bits
        assert key < lengths[ri] and shape['H']==shape['KVH']
        if h==3:
            e=qr@k[0,h].T*np.float32(.125);ce=sp.cuda_bulk(qr,k[0,h],.125)
            before=ref.dense_scores(np.where(oldmask[ri],e[ri],bulk[ri]),v[0,h],lengths[ri])
            after=ref.dense_scores(np.where(mask[ri],ce[ri],cb[ri]),v[0,h],lengths[ri])
            receipt=next((ART/'gpu').glob(shape['id']+'.*.json'))
            observed=json.loads(receipt.read_text())['G2']['baseline_vs_own_emulator']['max_abs']
            assert abs(float(abs(before-after).max())-observed)<1e-7
            assert observed>reg['data']['gpu_fp32_tolerance']['atol']


def test_fma_witness_arithmetic_matches_libm():
    libm=ctypes.CDLL('libm.so.6');libm.fmaf.argtypes=[ctypes.c_float]*3;libm.fmaf.restype=ctypes.c_float
    rng=np.random.default_rng(20260908)
    for a,b,c in rng.standard_normal((1000,3),dtype=np.float32):
        assert sp.fma32(a,b,c).tobytes()==np.float32(libm.fmaf(float(a),float(b),float(c))).tobytes()


def test_splitk_harness_checks_partition_rule_and_real_diagnostic_counts():
    from apa_sp1_gpu import all_references
    rng=np.random.default_rng(771)
    shape=dict(B=2,H=4,KVH=2,L=1,S=4097,D=33,VD=7,causal=True)
    q,k,kq,v=arrays(shape,771)
    masks=np.empty((2,4,1,4097),np.uint8)
    want=np.empty((2,4,1,7),np.float32)
    sinks=np.array([-10,0,3,10],np.float32)
    for b in range(2):
        for h in range(4):
            bulk=q[b,h,0]@kq[b,h//2].T*np.float32(1/np.sqrt(33))
            exact=q[b,h,0]@k[b,h//2].T*np.float32(1/np.sqrt(33))
            parts=[np.arange(j,min(j+2048,4097)) for j in range(0,4097,2048)]
            out,mask=sp.partition_online(bulk,exact,v[b,h//2],.5,parts,sinks[h])
            masks[b,h,0]=mask;want[b,h,0]=out
    got,_,_,counts=all_references(q,k,kq,v,shape,.5,1.0364333894937898,masks,sinks,splitk=True)
    np.testing.assert_allclose(got,want,atol=.001,rtol=.001)
    assert counts['cpu_gpu_mask_disagreements']==0
    assert counts['valid']==2*4*4097
    wrong=masks.copy();wrong[0,0,0,2048]=0
    _,_,_,bad=all_references(q,k,kq,v,shape,.5,1.0364333894937898,wrong,sinks,splitk=True)
    assert bad['cpu_gpu_mask_disagreements']==1


def test_final_scoring_uses_complete_pinned_history_and_rejects_unmatched_hits():
    from apa_sp1_1_scoring import registered_receipts,score
    rows=registered_receipts();scores=score(rows)
    assert len(rows)==48 and all(s['verdict'] in ['HIT','MISS'] for s in scores.values())
    assert all(s['receipts'] for s in scores.values())
    assert sum(not r['tail']['matched'] for r in rows)==14
    assert scores['P3_half_shapes_deviation']['verdict']=='HIT'
    assert scores['P2_prefill_overlap_ge_0_8']['verdict']=='MISS'
    assert scores['P3_all_prefill_speed']['verdict']=='MISS'
    assert scores['A4_decode_speed_le_1']['verdict']=='MISS'  # coverage, not a made-up speed failure
    # A forged favorable number cannot turn unmatched coverage into a hit.
    altered=json.loads(json.dumps(rows))
    for row in altered:
        if not row['tail']['matched']:
            row['G3']['speedup']=.01;row['G3']['deviation_ratio']=.01
    assert score(altered)['A4_decode_speed_le_1']['verdict']=='MISS'
