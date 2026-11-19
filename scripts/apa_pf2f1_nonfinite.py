"""G2 fixtures faithfully ported from PF2-REV1 reproductions (2026).
Prior art: IEEE 754-2008, P1-F1/F2 denominator guard and P1-F3 sentinel;
REV1 sink/raw-norm witnesses reused. No new numerical algorithm.
"""
import itertools
import numpy as np


def sink_cases():
    return [dict(D=D, dt=dt, diag=diag, kind=kind) for D,dt,diag,kind in
            itertools.product((64,128), ('float32','bfloat16'), (False,True), ('nan','inf'))]


def norm_cases():
    return [dict(D=D, dt=dt, G=G, diag=diag) for D,dt,G,diag in
            itertools.product((64,128), ('float32','bfloat16'), (1,4), (False,True))]


def assert_finite_control(a):
    assert a.size and np.isfinite(a).all() and (a!=0).any() and (a!=a.flat[0]).any()


def sink_fixture(D, dt):
    # Preserve REV1's RNG seed AND draw order, including preceding dtype/D draws.
    rng = np.random.default_rng(20261117)
    for d in (64,128):
        for dtype in ('float32','bfloat16'):
            q = rng.standard_normal((1,4,3,d),dtype=np.float32)
            k = rng.standard_normal((1,2,17,d),dtype=np.float32)
            v = rng.standard_normal((1,2,17,d),dtype=np.float32)
            s = rng.standard_normal((4,),dtype=np.float32)
            if (d,dtype)==(D,dt):
                return q,k,v,s
    raise AssertionError('unknown fixture')


def _out(z, diag):
    return (z[0] if diag else z).float().numpy()


def _decode(tc, qn, k, v, packed, scale, delta, sink, dt):
    outs=[]
    for i in range(qn.shape[2]):
        q = tc.tensor(np.ascontiguousarray(qn[:,:,i:i+1]), device='cuda', dtype=dt)
        z = tc._spo2.packed_run(q,k,v,packed,scale,delta,False,sink,False,fmt='tq4',carry='row',gqa_path='r2')
        outs.append(z.float().numpy())
    return np.concatenate(outs,axis=2)


def run_sink(tc, c):
    D, dt, diag = c['D'], c['dt'], c['diag']
    qn,kn,vn,sn = sink_fixture(D,dt)
    for a in (qn,kn,vn,sn):
        assert a.size and (a!=0).any() and (a!=a.flat[0]).any()
    T=lambda a: tc.tensor(np.ascontiguousarray(a),device='cuda',dtype=dt)
    q,k,v=T(qn),T(kn),T(vn)
    ws=tc.ApaSpq1Workspace(k,17,fmt='tq4')
    scale=float(np.float32(1/np.sqrt(D)))
    delta=tc.apa_packed.derive_delta(.01,tc.apa_packed.PACKED_PINNED_ENTRIES[f'4:{D}'])
    clean=_out(tc._spo2.packed_prefill(q,k,v,ws.packed,scale,delta,False,T(sn),diag,fmt='tq4'),diag)
    assert_finite_control(clean)
    sink=T(np.full(4,float(c['kind']),np.float32))
    pf=_out(tc._spo2.packed_prefill(q,k,v,ws.packed,scale,delta,False,sink,diag,fmt='tq4'),diag)
    dec=_decode(tc,qn,k,v,ws.packed,scale,delta,sink,dt)
    assert pf.shape==dec.shape==(1,4,3,D) and pf.dtype==dec.dtype==np.float32
    bad=[]
    if not (np.isnan(pf).all() and np.isnan(dec).all()):
        bad.append('sink_nonfinite')
    return dict(case=c, failed=bad, finite_control=True, rows=12, structurally_constant_nan_rows=int(np.isnan(pf).all(-1).sum()),
                prefill_finite_elements=int(np.isfinite(pf).sum()),decode_nan_elements=int(np.isnan(dec).sum()))


def norm_fixture(tc, D):
    cb=tc.quant.build_codebook(D,4)
    bnd=np.float32(.5)*(cb[:-1]+cb[1:])
    base=np.linspace(.5,1.5,D,dtype=np.float32)
    k=np.stack([base,-base])[None,None]
    q=np.stack([base,np.float32(1.25)*base])[None,None]
    norm=np.sqrt(np.sum(base.astype(np.float64)**2)).astype(np.float32)
    unit=k/norm
    indices=np.sum(unit[...,None]>bnd,axis=-1).astype(np.uint8)
    codes=indices[...,0::2] | (indices[...,1::2]<<4)
    rot=np.eye(D,dtype=np.float32)[None]
    v=np.stack([np.linspace(1,2,D,dtype=np.float32),np.linspace(-2,-1,D,dtype=np.float32)])[None,None]
    assert np.all(cb[indices[0,0,0]]>0) and np.all(cb[indices[0,0,1]]<0)
    for a in (k,q,v,codes):
        assert a.size and (a!=0).any() and (a!=a.flat[0]).any()
    return q,k,v,codes,rot,cb,norm


def run_norm(tc, c):
    D,dt,G,diag=c['D'],c['dt'],c['G'],c['diag']
    qn,kn,vn,cn,rn,cb,norm=norm_fixture(tc,D)
    qn=np.repeat(qn,G,axis=1)
    T=lambda a: tc.tensor(np.ascontiguousarray(a),device='cuda',dtype=dt)
    q,k,v=T(qn),T(kn),T(vn)
    codes=tc.tensor(cn,device='cuda',dtype='uint8')
    rot=tc.tensor(rn,device='cuda'); table=tc.tensor(cb,device='cuda')
    scale=float(np.float32(1/np.sqrt(D)))
    outs={}; decs={}
    for kind in ('finite','inf','nan'):
        n=np.array([[[norm,norm if kind=='finite' else float(kind)]]],np.float32)
        norms=tc.tensor(n,device='cuda')
        packed=(codes,norms,rot,table)
        outs[kind]=_out(tc._spo2.packed_prefill(q,k,v,packed,scale,0.,False,None,diag,fmt='tq4'),diag)
        if kind!='finite':
            decs[kind]=_decode(tc,qn,k,v,packed,scale,0.,None,dt)
        assert outs[kind].shape==(1,G,2,D) and outs[kind].dtype==np.float32
    assert_finite_control(outs['finite'])
    bad=[]
    if not np.isnan(outs['inf']).all(): bad.append('norm_nonfinite')
    if not np.isnan(outs['nan']).all(): bad.append('nan_norm_control')
    if not np.isnan(decs['nan']).all(): bad.append('nan_norm_decode_control')
    # Registered G2 wording requires the raw-Inf decode comparison too. Keep
    # this explicit: decode is read-only; an oracle boundary mismatch is RED.
    if not np.array_equal(np.any(~np.isfinite(outs['inf']),axis=-1),
                          np.any(~np.isfinite(decs['inf']),axis=-1)):
        bad.append('raw_inf_decode_row_sets')
    return dict(case=c, failed=bad, finite_control=True, nan_control_all_nan=bool(np.isnan(outs['nan']).all()),
                rows=2*G, structurally_constant_nan_rows=int(np.isnan(outs['inf']).all(-1).sum()),
                inf_prefill_nan_elements=int(np.isnan(outs['inf']).sum()),
                inf_decode_nan_elements=int(np.isnan(decs['inf']).sum()),
                nan_decode_all_nan=bool(np.isnan(decs['nan']).all()))
