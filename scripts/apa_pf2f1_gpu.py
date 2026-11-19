"""GPU payloads; run only through the registered launcher. No timing gate.
Prior art: PF2 (Project-Tensor 2026) grid, input generator, row-decode oracle,
spy/memory fixtures and unsigned bit comparisons, reused. G1 uses a binary
pipe to compare isolated imports without retaining full-grid device readbacks
on disk. This is ordinary IPC/test bookkeeping, no new numerical algorithm.
"""
from pathlib import Path
import argparse
import gc
import json
import math
import os
import struct
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import apa_pf2f1_common as C
import apa_pf2f1_reference as R
import apa_pf2f1_nonfinite as N


def coverage_cases():
    cases=[]
    for dt in ('float32','bfloat16'):
        for diag in (False,True):
            for cap,seq,D,VD,G in ((64,True,64,64,1), (128,True,64,65,4),
                                  (128,False,128,65,1), (256,True,64,129,4),
                                  (256,False,128,129,1), (512,True,64,257,4),
                                  (512,False,128,257,1)):
                cases.append(dict(dt=dt,diag=diag,CAP=cap,SEQ=seq,D=D,VD=VD,G=G,B=1,
                                  S=257,L=3,causal=True,sinks=True,cap=294,sub='coverage'))
    assert len(cases)==28
    return cases


def emit_arrays(label, idx, wire=None, tc=None, torch=None):
    # Preserve a binary stdout channel; module/import chatter goes to stderr.
    if wire is None:
        wire=sys.stdout.buffer
        sys.stdout=sys.stderr
        tc,_=C.load_arm(label)
        torch=C.torch_setup()
    E=R.Env(tc,torch)
    c=R.g1_cases()[idx]
    with tc.no_grad():
        q,k,v,s=R.case_tensors(torch,c,idx)
        scale=float(np.float32(1/math.sqrt(c['D'])))
        delta=tc.apa_packed.derive_delta(.01,tc.apa_packed.PACKED_PINNED_ENTRIES[f"4:{c['D']}"])
        ws,packed=E.store(k,c['cap'])
        out,mask=E.prefill(q,k,v,packed,scale,delta,c['causal'],s)[:2]
        assert out.shape==(c['B'],2*c['G'],c['L'],c['VD'])
        assert mask.shape==(c['B'],2*c['G'],c['L'],c['S']) and mask.dtype==torch.uint8
        assert bool(torch.isfinite(out).all())
        arrays=[out.contiguous().view(torch.int32 if out.element_size()==4 else torch.int16).cpu().numpy()
                .view(np.uint32 if out.element_size()==4 else np.uint16),mask.cpu().numpy()]
        for a in arrays:
            C.informative(a)  # both arms checked before emitting comparison data (L19)
        meta=dict(index=idx,id=R.case_id(c),module_sha256=C.build.receipt(label)['module_sha256'],
                  arrays=[dict(name=name,shape=list(a.shape),source_dtype=str(t.dtype),
                               source_width=t.element_size(),dtype=a.dtype.str,nbytes=a.nbytes,
                               nonempty=True,nonconstant=True)
                          for name,a,t in zip(('out','mask'),arrays,(out,mask))])
        b=json.dumps(meta).encode()
        wire.write(struct.pack('<Q',len(b)));wire.write(b)
        for a in arrays:
            wire.write(memoryview(a).cast('B'))
        wire.flush()


def read_exact(pipe,n):
    out=bytearray(n)
    view=memoryview(out)
    pos=0
    while pos<n:
        r=pipe.readinto(view[pos:])
        assert r, f'truncated binary comparison stream at {pos}/{n}'
        pos+=r
    return out


def worker(args):
    # Prior art: persistent process isolation / pipe request-response, ordinary
    # IPC. PF2's separate arm imports are retained. Ours: explicit cache-release
    # acknowledgement before the other worker is allowed to compute.
    wire=sys.stdout.buffer
    sys.stdout=sys.stderr
    tc,_=C.load_arm(args.arm)
    torch=C.torch_setup()
    while True:
        b=sys.stdin.buffer.read(4)
        if not b:break
        assert len(b)==4
        idx=struct.unpack('<I',b)[0]
        emit_arrays(args.arm,idx,wire,tc,torch)
        gc.collect();tc.empty_cache();torch.cuda.empty_cache();torch.cuda.synchronize()
        wire.write(b'R');wire.flush()
    return 0


def child(label,log):
    f=log.open('a')
    p=subprocess.Popen(['timeout','460s',sys.executable,'-B','-P',__file__,'worker','--arm',label],
                       cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=f)
    return p,f


def identity(args):
    spec=json.loads((C.ART/'LINES.json').read_text())[args.line]
    dest=C.ART/'identity'/args.line
    dest.mkdir(parents=True)
    rows=[]
    pa,fa=child('PF',dest/'PF.log')
    pb,fb=child('F1',dest/'F1.log')
    for idx in spec['indices']:
        C.build.rails()
        t0=time.monotonic()
        p=pa
        p.stdin.write(struct.pack('<I',idx));p.stdin.flush()
        assert p.stdout is not None
        n=struct.unpack('<Q',read_exact(p.stdout,8))[0]
        ah=json.loads(read_exact(p.stdout,n))
        baseline=[]
        for m in ah['arrays']:
            assert m['nbytes']>0
            baseline.append(read_exact(p.stdout,m['nbytes']))
        assert read_exact(p.stdout,1)==b'R'  # reference allocations released
        p=pb
        p.stdin.write(struct.pack('<I',idx));p.stdin.flush()
        n=struct.unpack('<Q',read_exact(p.stdout,8))[0]
        bh=json.loads(read_exact(p.stdout,n))
        assert ah['index']==bh['index']==idx and ah['id']==bh['id']
        assert len(ah['arrays'])==len(bh['arrays'])==2
        counts={}
        for m,want,buf in zip(bh['arrays'],ah['arrays'],baseline):
            # All metadata checks precede construction of the same-width views.
            assert m==want and m['source_width']==np.dtype(m['dtype']).itemsize
            assert np.dtype(m['dtype']).kind=='u'
            assert m['nonempty'] and m['nonconstant']
            a=np.frombuffer(buf,dtype=np.dtype(m['dtype'])).reshape(m['shape'])
            C.informative(a)
            flat=a.reshape(-1);offset=0;unequal=0;nonconstant=False;first=None
            while offset<flat.size:
                count=min(flat.size-offset,(16<<20)//flat.dtype.itemsize)
                got=np.frombuffer(read_exact(p.stdout,count*flat.dtype.itemsize),dtype=flat.dtype)
                if first is None:first=got[0]
                nonconstant=nonconstant or bool((got!=first).any())
                unequal+=int(np.count_nonzero(got!=flat[offset:offset+count]))
                offset+=count
            assert offset>0 and nonconstant, ('noninformative',idx,m['name'])
            counts[m['name']]=dict(elements=offset,unequal=unequal,dtype=m['dtype'],shape=m['shape'])
        assert read_exact(p.stdout,1)==b'R'
        r=dict(index=idx,id=ah['id'],comparisons=counts,seconds=time.monotonic()-t0,
               ok=all(v['unequal']==0 for v in counts.values()))
        rows.append(r)
        C.save(dest/f'case_{idx:04d}.json',r)
        assert r['ok'],r
        del a,flat,got,buf
        baseline.clear();gc.collect()
        print('G1_CASE',idx,'PASS',flush=True)
    for p,f in ((pa,fa),(pb,fb)):
        p.stdin.close()
        assert not p.stdout.read(1) and p.wait(timeout=30)==0
        f.close()
    C.save(dest/'RESULT.json',dict(status='PASS',cases=len(rows),rows=rows,indices=spec['indices']))
    return 0


def nonfinite(args):
    tc,_=C.load_arm(args.arm)
    torch=C.torch_setup()
    cases=N.sink_cases() if args.kind=='sink' else N.norm_cases()
    fn=N.run_sink if args.kind=='sink' else N.run_norm
    dest=Path(args.out)
    rows=[]
    with tc.no_grad():
        for c in cases:
            r=fn(tc,c);rows.append(r)
            print('G2_CASE',json.dumps(r),flush=True)
    torch.cuda.synchronize()
    C.save(dest/'RESULT.json',dict(arm=args.arm,kind=args.kind,cases=len(rows),
           passed=sum(not r['failed'] for r in rows),status='PASS' if all(not r['failed'] for r in rows) else 'FAIL',rows=rows))
    return 0 if all(not r['failed'] for r in rows) else 1


def coverage(args):
    c=coverage_cases()[args.index]
    tc,arm=C.load_arm('F1')
    torch=C.torch_setup();E=R.Env(tc,torch)
    q,k,v,s=R.case_tensors(torch,c,100000+args.index)
    scale=float(np.float32(1/math.sqrt(c['D'])))
    delta=tc.apa_packed.derive_delta(.01,tc.apa_packed.PACKED_PINNED_ENTRIES[f"4:{c['D']}"])
    tau=json.loads((ROOT/'artifacts/apa_p1pf2/a1/PF_I3_A1.json').read_text())[c['dt']]['tau']
    ws,packed=E.store(k,c['cap'])
    def pf(diag):
        z=tc._spo2.packed_prefill(E.to_tc(q),E.to_tc(k),E.to_tc(v),packed,scale,delta,c['causal'],E.to_tc(s),diag,fmt='tq4')
        return tuple(E.tt(x) for x in z) if diag else (E.tt(z),)
    with tc.no_grad():
        got=pf(c['diag'])[0]
        paired=pf(True)
        mask=paired[1]
        assert got.dtype==paired[0].dtype and got.shape==paired[0].shape
        assert torch.equal(got.contiguous().view(torch.int32 if got.element_size()==4 else torch.int16),
                           paired[0].contiguous().view(torch.int32 if got.element_size()==4 else torch.int16))
        C.informative(got.float().cpu().numpy());C.informative(mask.cpu().numpy())
        rows=[]
        for i in range(c['L']):
            smax=c['S']-c['L']+i+1
            dec=E.decode(q[:,:,i:i+1].contiguous(),k[:,:,:smax].contiguous(),v[:,:,:smax].contiguous(),packed,scale,delta,s)
            assert mask[:,:,i,:smax].dtype==dec[1][:,:,0,:smax].dtype
            eq=bool(torch.equal(mask[:,:,i,:smax],dec[1][:,:,0,:smax]))
            tail=bool((mask[:,:,i,smax:]==0).all())
            num,den=R.rel_rows(torch,got[:,:,i].float(),dec[0][:,:,0].float())
            rel=num/den.clamp_min(1e-300)
            rows.append(dict(row=i,mask_equal=eq,tail_zero=tail,rel=rel.cpu().tolist(),
                             ok=eq and tail and bool((rel<=tau).all())))
    result=dict(index=args.index,combination=c,arm=arm,executed=True,tau=tau,rows=rows,
                mask_method='paired diagnostics on the identical input; no-mask output also bit-equal to diagnostic output',
                status='PASS' if all(r['ok'] for r in rows) else 'FAIL')
    C.save(C.ART/'coverage'/f'case_{args.index:02d}.json',result)
    print('G4',json.dumps(result),flush=True)
    return 0 if result['status']=='PASS' else 1


def regression(args):
    import apa_pf2f1_g45 as F
    C.torch_setup()
    # Use the copied PF2 fixtures; capture their JSON and assert all rows.
    prefix='G4_JSON ' if args.kind=='refusals' else 'G5_JSON '
    cmd=[sys.executable,'-B','-P',str(ROOT/'scripts/apa_pf2f1_g45.py'),
         'g4-arm' if args.kind=='refusals' else 'g5-case','--arm','F1']
    env=dict(os.environ)
    if args.kind!='refusals':
        # A subprocess entry in this file binds the selected S/L before calling
        # PF2's unchanged memory fixture. Every original row is retained.
        cmd=[sys.executable,'-B','-P',__file__,'memory-child','--model',args.model,'--index',str(args.index)]
    dest=C.ART/'regression'/args.line;dest.mkdir(parents=True)
    log=dest/'payload.log'
    with log.open('x') as f:
        code=subprocess.run(['timeout','460s',*cmd],env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode
    texts=[s[len(prefix):] for s in log.read_text().splitlines() if s.startswith(prefix)]
    assert code==0 and len(texts)==1, (code,log)
    r=json.loads(texts[0]);rows=r['rows']
    if args.kind=='refusals':
        refs=[x for x in rows if x['kind']=='refusal']
        assert len(refs)==26 and len(rows)==29
        assert all(x['ok'] for x in rows)
        visible=next(x for x in refs if x['id']=='PF_visible_rows_not_k_rows')
        assert visible['spied_calls']==visible['launches']==visible['allocations']==0
        assert 'packed visible rows differ' in visible['message']
        assert any(x['id']=='PF_check_source_mismatch' and 'exact-K source' in x['message'] for x in refs)
    else:
        assert len(rows)==4 and all(x['within_bound'] and x['no_key_shaped_transient'] and
                x['route_backend']=='spq_packed_prefill_tq4_row' for x in rows)
    C.save(dest/'RESULT.json',dict(status='PASS',rows=rows,payload_exit=code,log_sha256=C.sha(log)))
    print('G6',args.line,'PASS',len(rows),flush=True)
    return 0


def memory_child(args):
    import apa_pf2f1_g45 as F
    F.G5_SL=[F.G5_SL[args.index]]
    # ensure_shim execs into its own script, losing the list override; preload
    # is set before process entry by the launcher, so that branch cannot run.
    assert os.environ.get('LD_PRELOAD')==str(F.SHIM)
    args.arm='F1'
    return F.g5_case(args)


def mutant(args):
    label='pf1m01_sink_guard_positive' if args.kind=='sink' else 'pf1m02_norm_finite_only'
    sides={}
    for side,arm in (('control','F1'),('mutant',label)):
        dest=C.ART/'nonfinite'/f'mutant_{args.kind}_{side}'
        log=C.ART/'gpu'/f'mutant_{args.kind}_{side}.log'
        with log.open('x') as f:
            code=subprocess.run(['timeout','220s',sys.executable,'-B','-P',__file__,'nonfinite',
                 '--kind',args.kind,'--arm',arm,'--out',str(dest)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode
        result=json.loads((dest/'RESULT.json').read_text())
        sides[side]=dict(exit=code,result=result,log_sha256=C.sha(log))
        if side=='control':
            assert code==0 and result['passed']==result['cases']==16, sides
    expected='sink_nonfinite' if args.kind=='sink' else 'norm_nonfinite'
    killed=(sides['mutant']['exit']==1 and sides['mutant']['result']['cases']==16 and
            all(expected in r['failed'] for r in sides['mutant']['result']['rows']))
    C.save(C.ART/f'MUTANT_{args.kind}.json',dict(status='PASS' if killed else 'FAIL',killed=killed,
            clean_control=True,expected_failure=expected,sides=sides))
    assert killed,sides
    return 0


def coverage_all(args):
    for idx in range(28):
        code=subprocess.run(['timeout','30s',sys.executable,'-B','-P',__file__,'coverage','--index',str(idx)],cwd=ROOT).returncode
        assert code==0, ('coverage case',idx,code)
    results=[json.loads((C.ART/'coverage'/f'case_{idx:02d}.json').read_text()) for idx in range(28)]
    assert all(r['executed'] and r['status']=='PASS' for r in results)
    C.save(C.ART/'COVERAGE_LEAD.json',dict(registered=28,executed=28,passed=28,status='PASS',combinations=results))
    return 0


def main():
    p=argparse.ArgumentParser();p.add_argument('mode');p.add_argument('--arm');p.add_argument('--kind')
    p.add_argument('--index',type=int);p.add_argument('--line');p.add_argument('--out');p.add_argument('--model')
    a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')!='', 'GPU payload requires a lease-visible device'
    assert os.environ.get('APA_PF2F1_LEASE')=='1', 'use the registered launcher'
    os.environ['TC_APA_SP']='1'
    return {'emit':lambda a:emit_arrays(a.arm,a.index),'worker':worker,'identity':identity,'nonfinite':nonfinite,
            'coverage':coverage,'coverage-all':coverage_all,'regression':regression,
            'memory-child':memory_child,'mutant':mutant}[a.mode](a)


if __name__=='__main__':
    raise SystemExit(main())
