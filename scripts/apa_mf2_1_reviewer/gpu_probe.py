"""Lead-run, actual-engine probes. Each cell is synchronous and independent.

APA-MF2.1 COPY (scripts/apa_mf2_1_reviewer/, originals pinned in
SHA256SUMS.orig_mf2rev_8100409 == the review's verdict.json evidence pins). Edits, all
marked "MF2.1 edit": (1) the before-arm pin file is read from the apa-mf2rev worktree
(read-only); (2) receipts carry run_by; (3) NEW cell `rule_definition`: the rule cell's
operands checked against the fp64 DEFINITION (centered two-pass, scripts/apa_mf2_1_defn.py)
instead of BK4 bit-exactness, which the MF2.1 order retired; kernel 'h'/'a' threshold
errors vs the definition are recorded (non-gating). The original cells are unchanged.

Prior art: FlashAttention-2 (Dao 2023), PyTorch boolean mask semantics, local
63960a4 bottom-right mask, DeMillo et al. (1978) mutation tests. Taken: predicate
masks, fp64 attention/Jacobian oracle and isolated source mutations. Ours:
adversarial operands, source/binary/capture pins, cell partition. No new algorithm.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback
import numpy as np
from oracle import attention, visible, selection

ROOT = Path(__file__).resolve().parents[2]
CAP = Path(os.environ.get("APA_MF2REV_CAPTURE_DIR",
    "/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_spq3/artifacts/apa_spq3/captures"))
DTYPES = ("float32", "bfloat16", "float16")
TOL = {"float32": 1e-4, "float16": .003, "bfloat16": .03}
ROWS = []
GPU_ATTEMPTED = False


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def emit(**kw):
    ROWS.append(kw)
    print(json.dumps(kw, allow_nan=False), flush=True)


def check(name, fn, **fields):
    try:
        detail = fn() or {}
        emit(name=name, passed=True, **fields, **detail)
    except Exception as exc:
        emit(name=name, passed=False, error=f"{type(exc).__name__}: {exc}", **fields)


def init_engine():
    import tensor_cuda as engine
    from tensor_cuda import functional, quant
    global tc, F, Q
    tc, F, Q = engine, functional, quant
    assert Path(F.__file__).resolve() == ROOT/"tensor_cuda/tensor_cuda/functional.py"
    assert Path(Q.__file__).resolve() == ROOT/"tensor_cuda/tensor_cuda/quant.py"
    manifest = Path(os.environ["APA_MF2REV_ENGINE_DIR"])/"manifest.json"
    m = json.loads(manifest.read_text())
    binary = Path(tc._C.__file__).resolve()
    assert sha(binary) == m["modules"][binary.name], "engine binary drift"
    for rel, pin in m["sources"].items():
        assert sha(ROOT/rel) == pin, f"engine source drift: {rel}"
    # Source receipt states scope. No new GPU training, builds, or driver overrides.
    emit(name="provenance", passed=True, binary=str(binary), binary_sha256=sha(binary),
         functional_sha256=sha(F.__file__), quant_sha256=sha(Q.__file__), manifest_sha256=sha(manifest))


def ts(arrs, dtype="float32", grad=False):
    global GPU_ATTEMPTED
    GPU_ATTEMPTED = True
    leaves = [tc.tensor(np.asarray(a, np.float32), requires_grad=grad) for a in arrs]
    return leaves, [a if dtype == "float32" else a.astype(dtype) for a in leaves]


def host(t):
    return t.float().numpy()


def finite_equal(got, ref, dtype):
    assert np.isfinite(got).all(), "nonfinite actual output/gradient"
    np.testing.assert_allclose(got, ref, atol=TOL[dtype], rtol=TOL[dtype])


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


@contextlib.contextmanager
def old_arm(folder):
    # MF2.1 edit (1): the pin file lives on the review branch (read-only)
    pins = json.loads(Path("/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2rev/artifacts/apa_mf2rev/cpu_receipt.json").read_text())["baseline_provenance"]["pins"]
    base = ROOT/"artifacts/apa_mf2/before"/folder
    fs = base/"tensor_cuda__tensor_cuda__functional.py"
    qs = base/"tensor_cuda__tensor_cuda__quant.py"
    for p in (fs, qs):
        assert sha(p) == pins[str(p.relative_to(ROOT))], "before arm drift"
    bf = load("tensor_cuda._mf2rev_"+folder+"_F", fs)
    bq = load("tensor_cuda._mf2rev_"+folder+"_Q", qs)
    assert sha(fs) != sha(F.__file__) and sha(qs) != sha(Q.__file__), "arms are same source"
    saved = (F._causal_mask, F.scaled_dot_product_attention)
    F._causal_mask, F.scaled_dot_product_attention = bf._causal_mask, bf.scaled_dot_product_attention
    try:
        yield bq
    finally:
        F._causal_mask, F.scaled_dot_product_attention = saved


def leak():
    sweep = {"float32": [10001., 1e6, 1e30], "bfloat16": [10048., 1e6, 1e30],
             "float16": [10008., 30000., 65504.]}
    for dt in DTYPES:
        for L, S in ((3,9), (9,9), (16,23)):
            def mask_values():
                m=host(F._causal_mask(L,S,"cuda",dt));vis=visible(L,S)
                assert (m[vis]==0).all() and np.isneginf(m[~vis]).all()
            check("gpu_mask_values_and_cast",mask_values,dtype=dt,L=L,S=S)
            for D in (64,96,128):
                for magnitude in sweep[dt]:
                    q = np.zeros((1,1,L,D), np.float32); q[...,0] = 1
                    k = np.zeros((1,1,S,D), np.float32); k[0,0,-1,0] = magnitude
                    v = np.zeros_like(k); v[0,0,-1,:] = 1
                    for path in ("sdpa_grad", "sdpa_infer", "sdpa_fused", "apa_full", "apa_partial"):
                        def run():
                            _, xx = ts((q,k,v), dt)
                            saved = F.USE_FUSED_SOFTMAX
                            try:
                                F.USE_FUSED_SOFTMAX = path == "sdpa_fused"
                                if path.startswith("sdpa"):
                                    ctx = contextlib.nullcontext() if path == "sdpa_grad" else tc.no_grad()
                                    with ctx:
                                        out = F.scaled_dot_product_attention(*xx, is_causal=True, scale=1.)
                                else:
                                    # Ensure finite quantized operands rather than silently excluding a failed cell.
                                    if path == "apa_partial":
                                        R,CB,BND = Q._tables(D,8,1,True,"cuda")
                                        kq = Q._quantize_keys(xx[1],R,CB,BND)
                                        assert np.isfinite(host(kq)).all(), "quantizer overflow (separate prerequisite failure)"
                                    out = Q.apa_quant_attention(*xx, bulk_bits=8,
                                        refine_percentile=1. if path=="apa_full" else .15, is_causal=True, scale=1.)
                                o = host(out)
                                assert np.isfinite(o).all()
                                assert (o[0,0,:-1] == 0).all(), "future key leaked"
                            finally:
                                F.USE_FUSED_SOFTMAX = saved
                        check("leak", run, dtype=dt,L=L,S=S,D=D,magnitude=magnitude,path=path)
    # Masked score may overflow before predicate; other rows remain finite.
    for dt in DTYPES:
        def masked_overflow():
            D,L,S = 64,3,9
            q = np.zeros((1,1,L,D),np.float32); q[0,0,0,0]=2
            k = np.zeros((1,1,S,D),np.float32); k[0,0,7,0]=60000 if dt=="float16" else 3e38
            v = np.zeros_like(k); v[0,0,7]=1
            _, x = ts((q,k,v),dt)
            with tc.no_grad():
                o=host(F.scaled_dot_product_attention(*x,is_causal=True,scale=1.))
            assert np.isfinite(o).all() and (o[0,0,0]==0).all()
        check("masked_overflow_predicate",masked_overflow,dtype=dt)
        # Isolate the two quant.py masked_fill sites on identical actual engine
        # operands; supplied Kq bypasses quantizer overflow, not masking or GEMM.
        for pct in (1.,.15):
            def quant_overflow():
                q=np.zeros((1,1,3,64),np.float32);q[0,0,0,0]=2
                k=np.zeros((1,1,9,64),np.float32);k[0,0,7,0]=60000 if dt=="float16" else 3e38
                v=np.zeros_like(k);v[0,0,7]=1
                _,x=ts((q,k,v),dt)
                save=Q._quantize_keys;Q._quantize_keys=lambda *a,**kw:x[1].detach()
                try:
                    out=Q.apa_quant_attention(*x,bulk_bits=8,refine_percentile=pct,is_causal=True,scale=1.)
                    o=host(out)
                    assert np.isfinite(o).all() and (o[0,0,0]==0).all(),"quant predicate site failed"
                finally:
                    Q._quantize_keys=save
            check("quant_masked_overflow_predicate",quant_overflow,dtype=dt,percentile=pct,
                  scope="supplied Kq isolates ranking/bulk masked_fill sites")


def negative():
    for path, score in (("infer",-3.3e38),("fused",-3.3e38),("grad",-3.402e38)):
        def run():
            q=np.zeros((1,1,2,64),np.float32);q[...,0]=1
            k=np.zeros_like(q);k[0,0,0,0]=score
            v=np.zeros_like(q);v[0,0,0]=1
            _, x = ts((q,k,v))
            save=F.USE_FUSED_SOFTMAX;F.USE_FUSED_SOFTMAX=path=="fused"
            try:
                with contextlib.nullcontext() if path=="grad" else tc.no_grad():
                    o=host(F.scaled_dot_product_attention(*x,is_causal=True,scale=1.))
                assert np.isfinite(o).all(), "finite negative visible score produced NaN"
                assert (o[0,0,0]==1).all(), "one visible key must have weight one"
            finally:
                F.USE_FUSED_SOFTMAX=save
        check("finite_negative_sentinel",run,path=path,score=score)
    for arr in ([[-np.inf,0.]], [[-np.inf,-np.inf]]):
        def weights():
            a=tc.tensor(np.array(arr,np.float32))
            with tc.no_grad():
                w=host(tc.causal_softmax(a))
            if np.isfinite(arr).any():
                np.testing.assert_array_equal(w,[[0.,1.]])
            else:
                assert np.isfinite(w).all(), "fully blocked additive row emits NaN; policy unresolved"
        check("fused_inf_entries",weights,scores=[str(x) for x in arr[0]])


def captures(model, layer, L, S, H=2):
    d=CAP/model;man=json.loads((d/"manifest.json").read_text())
    total=man["S"];group=man["config"]["num_attention_heads"]//man["config"]["num_key_value_heads"]
    assert L<=S<=total
    arrs=[]
    for kind in "qkv":
        p=d/f"L{layer:02d}_{kind}.npy"
        assert sha(p)==man["files"][p.name]["sha256"],"capture pin drift"
        a=np.load(p,mmap_mode="r")
        hs=np.arange(H) if kind=="q" else np.arange(H)//group
        sl=np.asarray(a[hs,total-L:total] if kind=="q" else a[hs,total-S:total])
        b=(sl.astype(np.uint32)<<16).view(np.float32)[None]
        arrs.append(b)
    return arrs


def identity():
    for model,layer in (("Qwen3-1.7B",16),("Qwen3-4B",21)):
        for L,S in ((1,8192),(16,16),(64,8192),(512,8192)):
            arrs=captures(model,layer,L,S)
            for dt in DTYPES:
                for folder in ("mf1_AB","prod_fd32b6d"):
                    for path in ("sdpa_infer","sdpa_grad","apa_full","apa_partial_diagnostic"):
                        def run():
                            def arm(quantmod):
                                leaves,x=ts(arrs,dt,grad=path=="sdpa_grad")
                                ctx=contextlib.nullcontext() if path=="sdpa_grad" else tc.no_grad()
                                with ctx:
                                    if path.startswith("sdpa"):
                                        out=F.scaled_dot_product_attention(*x,is_causal=L>1)
                                    else:
                                        out=quantmod.apa_quant_attention(*x,bulk_bits=8,
                                            refine_percentile=.15 if path.endswith("diagnostic") else 1.,is_causal=L>1)
                                result=[host(out)]
                                if path=="sdpa_grad":
                                    out.sum().backward();result.extend(host(a.grad) for a in leaves)
                                assert all(np.isfinite(a).all() for a in result)
                                return result
                            with old_arm(folder) as bq:
                                before=arm(bq)
                            after=arm(Q)
                            equal=all(b.tobytes()==a.tobytes() for b,a in zip(before,after))
                            if not path.endswith("diagnostic"):
                                assert equal,"different output/gradient bytes across verified arms"
                            return {"byte_identical":equal,"full_MF1_partial_selection_diagnostic_only":path.endswith("diagnostic")}
                        check("real_capture_identity",run,model=model,layer=layer,L=L,S=S,dtype=dt,before=folder,path=path)


def sliding():
    # Load REAL GRM _band_mask function body without importing the whole port.
    # AST extraction executes the shipped host code, no stand-in mask generator.
    import ast
    p=Path("/mnt/ForgeRealm/GraftRepository/core/gemma4_tc.py")
    tree=ast.parse(p.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="_band_mask")
    ns={"np":np,"tc":tc,"_band_cache":{}}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(p),"exec"),ns)
    for dt in DTYPES:
        for L,S,W in ((3,9,3),(16,31,8)):
            for mode in ("predicate_plus_band","additive_causal_plus_band","gemma_actual_band_only"):
                def run():
                    rng=np.random.default_rng(740+L)
                    arr=[rng.normal(0,.3,(1,2,n,64)).astype(np.float32) for n in (L,S,S)]
                    leaves,x=ts(arr,dt,grad=True)
                    band=ns["_band_mask"](L,S,W,"cuda",dt)
                    bh=host(band)
                    if mode=="additive_causal_plus_band":
                        sc=tc.matmul(x[0],x[1],alpha=1.,trans_b=True)
                        w=(sc+F._causal_mask(L,S,"cuda",dt)+band).softmax(-1)
                        out=tc.matmul(w,x[2])
                    else:
                        out=F.scaled_dot_product_attention(*x,attn_mask=band,
                            is_causal=mode=="predicate_plus_band",scale=1.)
                    real=[host(a) for a in x]
                    ref,wr,gr=attention(*real,1.,causal=mode!="gemma_actual_band_only",additive=bh,
                        upstream=np.ones(tuple(out.shape)))
                    finite_equal(host(out),ref,dt)
                    out.sum().backward()
                    for leaf,g in zip(leaves,gr):finite_equal(host(leaf.grad),g,dt)
                check("gemma_additive",run,dtype=dt,L=L,S=S,window=W,mode=mode)


def grapa():
    sys.path.insert(0,"/mnt/ForgeRealm/GRAPA-Native-LLM")
    from grapa.attention import apa_selective
    for dt in DTYPES:
        for L,S in ((3,3),(3,9)):
            arr=[np.random.default_rng(n+L).normal(0,.3,(1,2,n,64)).astype(np.float32) for n in (L,S,S)]
            def full():
                leaves,x=ts(arr,dt,True)
                out=apa_selective(*x,None,1.,.125,True)
                real=[host(a) for a in x]
                ref,w,gr=attention(*real,.125,upstream=np.ones(tuple(out.shape)))
                finite_equal(host(out),ref,dt);out.sum().backward()
                after=[host(out)]+[host(a.grad) for a in leaves]
                for a,g in zip(after[1:],gr):finite_equal(a,g,dt)
                with old_arm("mf1_AB"):
                    oldleaves,oldx=ts(arr,dt,True)
                    oldout=apa_selective(*oldx,None,1.,.125,True);oldout.sum().backward()
                    before=[host(oldout)]+[host(a.grad) for a in oldleaves]
                assert all(a.tobytes()==b.tobytes() for a,b in zip(after,before)),"GRAPA output/gradient byte drift"
            check("grapa_actual_full_helper_forward_backward",full,dtype=dt,L=L,S=S)
        def fused_smoke():
            L=S=4;rng=np.random.default_rng(54)
            arr=[rng.normal(0,.3,(1,1,n,64)).astype(np.float32) for n in (L,S,S)]
            leaves,x=ts(arr,dt,True)
            out=apa_selective(*x,x[1].detach(),.15,.125,True)
            out.sum().backward()
            assert all(np.isfinite(host(a)).all() for a in [out]+[l.grad for l in leaves])
            return {"scope":"finite small fused training smoke; fused APA masking unchanged; no parity claim"}
        check("grapa_fused_training_smoke",fused_smoke,dtype=dt)


def short():
    # Registered recommendation: causal S<L raises BEFORE matmul; current code expected RED.
    for dt in DTYPES:
        for path in ("sdpa_grad","sdpa_infer","apa_full","apa_partial","grapa_full"):
            def run():
                arr=[np.zeros((1,1,n,64),np.float32) for n in (8,5,5)]
                _,x=ts(arr,dt)
                called=[];orig=tc.matmul
                def mm(*a,**kw):called.append(True);return orig(*a,**kw)
                tc.matmul=mm
                rejected=False;status=None
                try:
                    try:
                        if path.startswith("sdpa"):
                            with contextlib.nullcontext() if path.endswith("grad") else tc.no_grad():
                                o=F.scaled_dot_product_attention(*x,is_causal=True)
                        elif path=="grapa_full":
                            sys.path.insert(0,"/mnt/ForgeRealm/GRAPA-Native-LLM")
                            from grapa.attention import apa_selective
                            o=apa_selective(*x,None,1.,.125,True)
                        else:
                            o=Q.apa_quant_attention(*x,refine_percentile=1. if path=="apa_full" else .15,is_causal=True)
                        status="finite" if np.isfinite(host(o)).all() else "nonfinite"
                    except (RuntimeError,ValueError) as exc:
                        rejected=True;status=str(exc)
                finally:
                    tc.matmul=orig
                emit(name="short_observation",passed=rejected and not called,dtype=dt,path=path,
                     status=status,matmul_calls=len(called))
                assert rejected and not called,"S<L must reject before matmul under recommended contract"
            check("short_geometry",run,dtype=dt,path=path)


def rule():
    from statistics import NormalDist
    z=NormalDist().inv_cdf(.85)
    for D in (64,96,128):
        for L,S in ((1,1),(1,2),(1,256),(1,2048),(3,9)):
            for causal in (False,True):
                def run():
                    q=np.zeros((1,1,L,D),np.float32);q[...,:3]=1
                    k=np.zeros((1,1,S,D),np.float32);k[...,:3]=[1,107/256,1/65536]
                    v=np.zeros_like(k)
                    _,x=ts((q,k,v),"bfloat16")
                    r=tc._C.bp_kernel_4_diagnostic(x[0],x[1],x[1],x[2],1.,float(z),causal,"h")
                    km=r[3].numpy()!=0;thr=r[2].numpy()[...,None]
                    composed=host(Q._bk4_selection(x[0].float(),x[1].float(),1.,z,causal))!=0
                    b=host(x[0]).astype(np.float64)@host(x[1]).astype(np.float64).swapaxes(-1,-2)
                    oracle,th=selection(b,causal,z)
                    band=np.abs(np.abs(b)-thr)<=2e-5*np.maximum(1.,np.abs(thr))
                    mism=km!=composed
                    emit(name="rule_observation",passed=not mism.any(),D=D,L=L,S=S,causal=causal,
                         bit_mismatches=int(mism.sum()),outside_tie_band=int((mism&~band).sum()),
                         composed_vs_fp64=int((oracle!=composed).sum()),max_threshold=float(thr.max()))
                    assert not mism.any(),"exact BK4 selection contract failed"
                check("rule_exact_adversarial",run,D=D,L=L,S=S,causal=causal)
    # Regression discriminators for >= and visibility; use actual helper.
    for dt in DTYPES:
        def zeros():
            _,x=ts((np.zeros((1,1,3,64)),np.zeros((1,1,9,64))),dt)
            s=host(Q._bk4_selection(x[0].float(),x[1].float(),.125,z,True))!=0
            np.testing.assert_array_equal(s[0,0],visible(3,9))
        check("rule_zero_tie_visible",zeros,dtype=dt)
    def counts():
        # Constant visible scores: count=7,8,9; using S for earlier rows
        # spuriously raises the threshold and drops all visible keys.
        q=np.zeros((1,1,3,64),np.float32);q[...,0]=1
        k=np.zeros((1,1,9,64),np.float32);k[...,0]=1
        _,x=ts((q,k))
        s=host(Q._bk4_selection(x[0],x[1],1.,z,True))!=0
        np.testing.assert_array_equal(s[0,0],visible(3,9))
    check("rule_visible_count_discriminator",counts)


def census():
    for dt in DTYPES:
        for L in (1,2,3,16,64,513,2048):
            for S in (L,L+7):
                for D in (64,96,128):
                    for causal in (False,True):
                        def run():
                            rng=np.random.default_rng(L+S+D)
                            arr=[rng.normal(0,.2,(1,1,n,D)).astype(np.float32) for n in (L,S,S)]
                            _,x=ts(arr,dt)
                            with tc.no_grad():out=F.scaled_dot_product_attention(*x,is_causal=causal)
                            real=[host(a) for a in x]
                            ref,w=attention(*real,D**-.5,causal=causal)
                            finite_equal(host(out),ref,dt)
                        check("boundary_census",run,dtype=dt,L=L,S=S,D=D,causal=causal)


def extension():
    global GPU_ATTEMPTED
    import torch
    assert torch.cuda.is_available(),"CUDA unavailable"
    GPU_ATTEMPTED = True
    edir=Path(os.environ.get("APA_MF2REV_EXTENSION_DIR",str(ROOT/"apa_cuda")))
    sys.path.insert(0,str(edir))
    import apa_attention_cuda as ext
    manifest=json.loads((edir/"apa_mf2rev_manifest.json").read_text())
    assert manifest["official_cuda_kernels"] is True,"stand-in extension forbidden"
    assert sha(ext.__file__)==manifest["binary_sha256"]
    for rel in ("apa_cuda/csrc/apa.cpp","apa_cuda/csrc/apa_kernels.cu"):
        assert sha(ROOT/rel)==manifest["sources"][rel]
    emit(name="extension_provenance",passed=True,path=ext.__file__,sha256=sha(ext.__file__))
    def call(q,k,v,L,S,H,D):
        return ext.forward(q,k,v,torch.eye(D,device="cuda").expand(H,D,D).contiguous(),
            torch.zeros(1,device="cuda"),torch.empty(0,device="cuda"),1.,True,None,1.,0.,64,False,False)
    for L,S,H in ((1,7,1),(3,9,1),(257,4097,64)):
        def closed():
            D=64;q=torch.zeros((1,H,L,D),device="cuda",requires_grad=True)
            k=torch.zeros((1,H,S,D),device="cuda");v=torch.arange(S,device="cuda",dtype=torch.float32).view(1,1,S,1).expand(1,H,S,D).contiguous()/S
            r=call(q,k,v,L,S,H,D);out,kq,mask=r
            expected=(S-L+np.arange(L))/(2.*S)
            o=out.detach().cpu().numpy()
            np.testing.assert_allclose(o[0,:, :,0],np.broadcast_to(expected,(H,L)),atol=1e-5,rtol=1e-5)
            grads=ext.backward(torch.ones_like(out),q,k,v,kq,mask,1.,True,None,
                H*L*S*4>256*1024*1024,64)
            gv=np.array([sum(1/(S-L+i+1) for i in range(L) if j<=S-L+i) for j in range(S)])
            np.testing.assert_allclose(grads[2].cpu().numpy()[0,0,:,0],gv,atol=1e-5,rtol=1e-4)
            assert all(torch.isfinite(g).all() for g in grads)
        check("official_extension_rectangular_forward_backward",closed,L=L,S=S,H=H,tiled=H*L*S*4>256*1024*1024)
    def negfill():
        L=S=3;H=1;D=64
        q=torch.zeros((1,H,L,D),device="cuda");q[...,0]=1
        k=torch.zeros((1,H,S,D),device="cuda");k[:,:,0,0]=-2e9
        v=torch.zeros_like(k);v[:,:,0,:]=1
        out=call(q,k,v,L,S,H,D)[0]
        assert torch.isfinite(out).all() and (out[0,0,0]==1).all(),"kNegInf reversion leak"
    check("official_extension_negative_fill",negfill)
    def short_extension():
        L,S,H,D=8,5,1,64
        q=torch.zeros((1,H,L,D),device="cuda");k=torch.zeros((1,H,S,D),device="cuda");v=torch.zeros_like(k)
        rejected=False
        try:
            out=call(q,k,v,L,S,H,D)[0]
            emit(name="extension_short_observation",passed=False,
                 status="finite" if torch.isfinite(out).all() else "nonfinite",L=L,S=S)
        except (RuntimeError,ValueError) as exc:
            rejected=True
            emit(name="extension_short_observation",passed=True,status=str(exc),L=L,S=S)
        assert rejected,"official extension must reject causal S<L under recommended contract"
    check("official_extension_short_geometry",short_extension)


def rule_definition():
    # MF2.1 edit (3): definition-based variant of `rule` (same operands). Gate: composed vs
    # the fp64 definition, 0 mismatches outside |a-thr|<=2e-5|thr|, and (exact property)
    # every equal visible |b| selected. Kernel rows are observations (never asserted).
    sys.path.insert(0, str(ROOT/"scripts"))
    import apa_mf2_1_defn as DEF
    z = Q._norm_ppf(1.0 - 0.15)
    for D in (64,96,128):
        for L,S in ((1,1),(1,2),(1,256),(1,2048),(3,9)):
            for causal in (False,True):
                def run():
                    q=np.zeros((1,1,L,D),np.float32);q[...,:3]=1
                    k=np.zeros((1,1,S,D),np.float32);k[...,:3]=[1,107/256,1/65536]
                    _,x=ts((q,k,np.zeros_like(k)),"bfloat16")
                    composed=host(Q._bk4_selection(x[0].float(),x[1].float(),1.,z,causal))!=0
                    b=host(x[0]).astype(np.float64)@host(x[1]).astype(np.float64).swapaxes(-1,-2)
                    ref,thr,a,vis=DEF.definition_fp64(b,causal,z)
                    c=DEF.compare(composed,ref,a,thr,vis)
                    obs={}
                    for var in ("h","a"):
                        r=tc._C.bp_kernel_4_diagnostic(x[0],x[1],x[1],x[2],1.,float(z),causal,var)
                        kt=r[2].numpy()[...,None].astype(np.float64)
                        obs[var]={**{k_:v for k_,v in DEF.compare(r[3].numpy()!=0,ref,a,thr,vis).items() if k_ in ("mismatch","out_of_band")},
                                  "thr_max_rel_err":float((np.abs(kt-thr)/np.abs(thr)).max())}
                    emit(name="rule_definition_observation",passed=c["out_of_band"]==0,D=D,L=L,S=S,causal=causal,
                         composed_vs_def=c,kernel_h=obs["h"],kernel_a=obs["a"])
                    assert c["out_of_band"]==0,"composed outside the definition tie band"
                    assert np.array_equal(composed[0,0],vis),"equal visible |b| must all be selected"
                check("rule_definition",run,D=D,L=L,S=S,causal=causal)


CELLS={"leak":leak,"negative":negative,"identity":identity,"sliding":sliding,
       "grapa":grapa,"short":short,"rule":rule,"census":census,"extension":extension,
       "rule_definition":rule_definition}


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("cell",choices=CELLS);parser.add_argument("--receipt",required=True)
    args=parser.parse_args()
    status=1
    try:
        if args.cell!="extension":init_engine()
        CELLS[args.cell]()
        status=0 if ROWS and all(r["passed"] for r in ROWS) else 1
    except Exception as exc:
        emit(name="cell_error",passed=False,error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        receipt={"cell":args.cell,"status":"PASS" if status==0 else "RED",
                 "gpu_execution_attempted":GPU_ATTEMPTED,"rows":ROWS,"model":"gpt-6.1-sol","reasoning_effort":"high",
                 "run_by":"apa-mf2-1 seat (Claude Opus 5.5), copy of the apa-mf2rev probe"}  # MF2.1 edit (2)
        with open(args.receipt,"x") as f:json.dump(receipt,f,indent=2,allow_nan=False);f.write("\n")
    sys.exit(status)
