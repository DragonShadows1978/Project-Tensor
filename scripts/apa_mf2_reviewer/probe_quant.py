"""Deferred GPU probe for actual tensor_cuda.quant, independent fp64 oracle.
Prior art: Dao 2023 FlashAttention-2 v2.1 and 63960a4 bottom-right; GRAPA
_visible_counts / BK4 population moments; DeMillo et al. 1978 mutations.
Own contribution: full dtype/dimension census and exact eligibility probes.
Criteria registered before execution. Requires pinned built _tensor_cuda on
PYTHONPATH; no build, GPU run or engine write is performed by the static seat.
"""
import argparse, ast, contextlib, hashlib, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tensor_cuda'),str(ROOT/'scripts')]
from apa_mf1rev_oracle import attention, population_select, visibility
from apa_mf1rev_runtime import preflight
DTYPES=('float32','bfloat16','float16')
DIMS=(64,96,128)
P=.15
TOL={'float32':(2e-5,2e-5),'bfloat16':(.03,.03),'float16':(.003,.003)}


def engine():
    preflight(ROOT)
    import tensor_cuda as tc
    from tensor_cuda import quant as Q
    assert Path(tc.__file__).resolve()==ROOT/'tensor_cuda/tensor_cuda/__init__.py'
    assert Path(Q.__file__).resolve()==ROOT/'tensor_cuda/tensor_cuda/quant.py'
    binary=Path(tc._C.__file__).resolve()
    sha=hashlib.sha256(binary.read_bytes()).hexdigest()
    expected='8ca4de6c9d8aecfd5129a45794ba309254fa3bcaa35ba5eb54b69a69b6c0908a'  # APA-MF2 copy: registered MF2 build (device code == MF1's, logs/apa_mf2_build_provenance.log)
    assert sha==expected, f'engine binary not pinned: {sha}; register rebuilt engine separately before running gates'
    print('ENGINE',str(binary),sha,flush=True)
    return tc,Q


@contextlib.contextmanager
def capture(tc):
    rec={};orig_fill,orig_where=tc.Tensor.masked_fill,tc.where
    def fill(self,condition,value):
        rec['bulk_abs']=self.float().numpy().astype(np.float64)
        rec['eligibility']=condition.float().numpy()==0
        return orig_fill(self,condition,value)
    def where(condition,x,y):
        rec['selection']=condition.float().numpy()!=0
        return orig_where(condition,x,y)
    tc.Tensor.masked_fill=fill;tc.where=where
    try:yield rec
    finally:tc.Tensor.masked_fill=orig_fill;tc.where=orig_where


def case(tc,Q,L,S,D,dtype,p=P,fn=None):
    rng=np.random.default_rng(20261007+100*L+S+D)
    arrays=[rng.normal(0,.5,(1,2,n,D)).astype(np.float32) for n in (L,S,S)]
    tensors=[tc.tensor(a,dtype=dtype) for a in arrays]
    q,k,v=[a.float().numpy().astype(np.float64) for a in tensors]
    with capture(tc) as rec:
        out=(fn or Q.apa_quant_attention)(*tensors,bulk_bits=8,refine_percentile=p,is_causal=True)
    actual=out.float().numpy()
    vis=visibility(L,S)
    if p<1:
        assert np.array_equal(rec['eligibility'],vis),f'eligibility failed {L} {S} {dtype}'
        R,CB,BND=Q._tables(D,8,2,True,'cuda')
        kq=Q._quantize_keys(tensors[1],R,CB,BND).float().numpy().astype(np.float64)
        # Code-produced Kq is an operand; oracle never uses its masks/helpers.
        # Stats oracle takes stored bulk abs, independently sanity-checked vs q.kq.
        bulk64=np.abs(q@kq.swapaxes(-1,-2)*D**-.5)
        operand_error=np.max(np.abs(np.where(vis,bulk64-rec['bulk_abs'],0.)))
        atol,rtol=TOL[dtype]
        assert operand_error<=atol+rtol*np.max(bulk64),f'bulk operand drift {operand_error}'
        sel,th=population_select(rec['bulk_abs'],L,S,P)
        boundary=vis & (np.abs(rec['bulk_abs']-th)<=2e-5*np.maximum(1.,np.abs(th)))
        considered=np.broadcast_to(vis & ~boundary,sel.shape)
        assert np.array_equal(rec['selection'][considered],sel[considered]),'selection rule mismatch'
        # Never replace oracle selection with implementation choices at boundaries.
        ref=attention(q,k,v,D**-.5,kq=kq,selection=sel)
        excluded=int(boundary.sum())
    else:
        ref=attention(q,k,v,D**-.5);excluded=0
    atol,rtol=TOL[dtype]
    e=float(np.max(np.abs(actual-ref)));limit=atol+rtol*float(np.max(np.abs(ref)))
    assert np.isfinite(actual).all() and e<=limit, (L,S,D,dtype,p,e,limit)
    print(json.dumps({'L':L,'S':S,'D':D,'dtype':dtype,'p':p,'maxabs':e,
                      'boundary_entries':excluded,'pass':True}),flush=True)


def variant(Q,kind):
    # Prior art: DeMillo et al. single-site mutation; regression A-only isolate.
    # In-memory copies preserve engine source as read-only.
    tree=ast.parse((ROOT/'tensor_cuda/tensor_cuda/quant.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='apa_quant_attention')
    if kind in ('A_only','square_definition'):
        statement=next(n for n in ast.walk(fn) if isinstance(n,ast.If) and n.lineno==183)
        statement.body=statement.orelse;statement.orelse=[]
    if kind in ('top_left','square_definition'):
        replacement=ast.parse('absr_f = absr_f.masked_fill(tc.tensor(np.array([[j > i for j in range(S)] for i in range(L)], dtype=np.float32), device=dev), 0.0)').body[0]
        class Replace(ast.NodeTransformer):
            def visit_Assign(self,n):return ast.copy_location(replacement,n) if n.lineno==181 else self.generic_visit(n)
        fn=Replace().visit(fn)
    ast.fix_missing_locations(fn);ns=dict(Q.__dict__)
    exec(compile(ast.Module([fn],type_ignores=[]),'in-memory quant variant','exec'),ns)
    return ns['apa_quant_attention']


def square(tc,Q):
    # Prior art: 63960a4 square reduction j<=i. Current in-memory function
    # copies only; no old code is loaded, and no square comparator is an oracle
    # for rectangles. B is removed from both copies to isolate fix A.
    square_definition=variant(Q,'square_definition')
    A=variant(Q,'A_only')
    for L in (1,3,64,513):
        for D in DIMS:
            for dtype in DTYPES:
                rng=np.random.default_rng(L+D)
                tensors=[tc.tensor(rng.normal(0,.5,(1,2,L,D)).astype(np.float32),dtype=dtype) for _ in range(3)]
                old=square_definition(*tensors,bulk_bits=8,refine_percentile=P,is_causal=True).float().numpy()
                new=A(*tensors,bulk_bits=8,refine_percentile=P,is_causal=True).float().numpy()
                assert old.tobytes()==new.tobytes(),f'A-only square bytes differ {L} {D} {dtype}'
                print('A-ONLY SQUARE BYTE PASS',L,D,dtype,flush=True)


def rule_diagnostics(tc,Q):
    # Diagnostic only: supplied Kq isolates statistics, not a quantizer witness.
    q=np.zeros((1,1,1,64),np.float32);q[...,0:2]=1
    kq=np.zeros((1,1,2,64),np.float32);kq[...,0]=1;kq[0,0,1,1]=1/256
    v=np.zeros_like(kq);v[0,0,1,:]=1
    qt,kqt,vt=[tc.tensor(a,dtype='bfloat16') for a in (q,kq,v)]
    out,lse,thr,mask,cycles=tc._C.bp_kernel_4_diagnostic(qt,kqt,kqt,vt,.125,Q._norm_ppf(.85),True,'h')
    b=tc.matmul(qt,kqt,alpha=.125,trans_b=True).abs().float()
    mu=b.sum([-1],True)*.5
    var=((b*b).sum([-1],True)*.5-mu*mu).relu()
    composed=b.ge(mu+var.pow(.5)*Q._norm_ppf(.85)).numpy()
    print('RULE DIAGNOSTIC BF16 stored bulk',b.numpy().tolist(),'BK4 threshold',thr.numpy().tolist(),
          'BK4 selection',mask.numpy().tolist(),'composed',composed.tolist(),flush=True)
    assert np.array_equal(mask.numpy()!=0,[[[[False,False]]]])
    assert np.array_equal(composed!=0,[[[[True,True]]]])
    # Positive/negative equality boundary: correct operator is >=, not >.
    t=tc.tensor(np.array([1.,2.,2.],np.float32))
    assert np.array_equal(t.ge(2.).numpy(),[0,1,1])
    # Population, not sample, variance in actual engine.
    t=tc.tensor(np.array([[0.,2.]],np.float32))
    assert float(t.std([-1],True).numpy()[0,0])==1.
    # Prior art: BK4 moments versus centered population variance, existing
    # std implementation. Stored-bulk diagnostic, not quantizer-output proof.
    t=tc.tensor(np.array([[100.,100.001]],np.float32))
    mu=t.mean([-1],True);z=Q._norm_ppf(.85)
    centered=mu+t.std([-1],True)*z
    moment=mu+((t*t).mean([-1],True)-mu*mu).relu().pow(.5)*z
    sc=t.ge(centered).numpy()!=0;sm=t.ge(moment).numpy()!=0
    print('NONCAUSAL RULE DIAGNOSTIC centered',centered.numpy().tolist(),
          'moments',moment.numpy().tolist(),'selected',sc.tolist(),sm.tolist(),flush=True)
    assert np.array_equal(sc,[[False,False]])
    assert np.array_equal(sm,[[False,True]])


def zero_visibility(tc,Q):
    # Prior art: BK4 visible && >= and 63960a4. Own degenerate witness tests
    # the mask itself, rather than claiming a Q=0 attention-output failure.
    for dtype in DTYPES:
        q=tc.tensor(np.zeros((1,1,3,64),np.float32),dtype=dtype)
        k=tc.tensor(np.zeros((1,1,9,64),np.float32),dtype=dtype)
        with capture(tc) as rec:
            out=Q.apa_quant_attention(q,k,k,bulk_bits=8,refine_percentile=P,is_causal=True)
        expected=np.broadcast_to(visibility(3,9),rec['selection'].shape)
        print('ZERO VISIBILITY',dtype,'expected',int(expected.sum()),
              'actual',int(rec['selection'].sum()),flush=True)
        assert np.array_equal(rec['eligibility'],visibility(3,9))
        assert np.isfinite(out.float().numpy()).all()
        assert np.array_equal(rec['selection'],expected),'BK4 rule requires visible && >=; zero-threshold invisible entries selected'


def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',choices=['baseline','square','mutations','rule','zero_visibility'],default='baseline')
    args=p.parse_args();tc,Q=engine()
    if args.suite=='square':square(tc,Q);return
    if args.suite=='rule':rule_diagnostics(tc,Q);return
    if args.suite=='zero_visibility':zero_visibility(tc,Q);return
    if args.suite=='mutations':
        case(tc,Q,3,17,64,'float32')
        mutant=variant(Q,'top_left')
        try:case(tc,Q,3,17,64,'float32',fn=mutant)
        except AssertionError as exc:print('QUANT TOP-LEFT MUTANT KILLED',str(exc),flush=True)
        else:raise AssertionError('quant stats mask mutant survived')
        return
    for L in (1,3,64,513):
        for S in (L,L+1,4*L+5):
            for D in DIMS:
                for dtype in DTYPES:
                    for pct in (P,1.):case(tc,Q,L,S,D,dtype,p=pct)

if __name__=='__main__':main()
