"""Staged numerical/byte worker; never executed by this CPU-only seat.
Prior art: AB-1/SPQ3/EB4 (2026) real captures, append history and bitwise
reference-pair checks; IEEE-754 integer views; pybind11 read-only accessor.
No new attention algorithm; only frozen inputs and exact comparisons.
"""
from gpu_common import *
import argparse,numpy as np,gc

def bits_to_float(a):return (np.ascontiguousarray(a).astype(np.uint32)<<16).view(np.float32)
def inputs(case):
    arr={}
    for k,r in case['files'].items():
        assert sha(r['path'])==r['sha256'],r['path']
        a=np.load(r['path'],mmap_mode='r');assert a.dtype==np.uint16 and list(a.shape)==r['shape']
        assert np.isfinite(bits_to_float(a)).all();assert np.any(a!=0) and np.any(a!=a.flat[0])
        arr[k]=a
    return arr

def numerical(args):
    tc,I,provenance=runtime(args.arm);os.environ['TC_APA_SP']='1';os.environ['APA_FXTC1_VARIANT']='2'
    case=next(r for r in load(ART/'INPUT_CENSUS.json') if r['model']==args.model and r['S']==args.S)
    host=inputs(case); S=args.S;outdir=Path(args.out);outdir.mkdir(parents=True,exist_ok=True)
    def tensor(bits):
        bits=np.ascontiguousarray(bits);t=tc.tensor(bits_to_float(bits),dtype='bfloat16')
        assert np.array_equal(raw(tc,I,t),bits),'bf16 input upload changed bits'
        return t
    def query(n):
        # 32K export holds the last 256 query rows; all exercised decode rows lie there.
        j=n-1-(32768-256) if S==32768 else n-1
        return tensor(host['q'][:,j:j+1,:][None])
    arrays={}; history=[]
    for n in case['append_lengths']:
        disk();k=tensor(host['k'][:,:n,:][None]);v=None
        if not history:w=tc.ApaSpq1Workspace(k,S,fmt='tq4')
        else:w.append(k)
        assert w.valid_rows==n and w.capacity==S
        history.append(n);arrays[f'append_{n}_codes']=raw(tc,I,w.codes);arrays[f'append_{n}_norms']=raw(tc,I,w.scales)
        if n>=S-3:
            v=tensor(host['v'][:,:n,:][None]);q=query(n);vi,vs=tc.apa_eb2b_quantize(v)
            scale=float(np.float32(1/np.sqrt(128)));delta=float(np.float32(8.1))
            r3=tc.apa_eb3(q,k,v,w,scale,delta,carry='row',diagnostics=True,variant=0)
            r4=tc.apa_eb4(q,k,v,vi,vs,w,scale,delta,case['fraction'],carry='row',diagnostics=True)
            for name,t in [('eb3_out',r3[0]),('eb3_mask',r3[2]),('eb4_out',r4[0]),('eb4_flags',r4[1]),('eb4_used',r4[2]),('eb4_mask',r4[8]),('int8_v',vi),('v_scales',vs)]:arrays[f'step_{n}_{name}']=raw(tc,I,t)
            if n==S:
                for prefix,rs,items in [('eb3',r3,{'qr':1,'bulk':3,'prefix':4}),('eb4',r4,{'scores':3,'logz':4,'cm':5,'cz':6,'blog':7,'bulk':9,'prefix':10,'qr':11})]:
                    for name,j in items.items():arrays[f'{prefix}_{name}']=raw(tc,I,rs[j])
                for name,t in [('eb3_nondiag',tc.apa_eb3(q,k,v,w,scale,delta,carry='row')),('eb4_nondiag',tc.apa_eb4(q,k,v,vi,vs,w,scale,delta,case['fraction'],carry='row'))]:arrays[name]=raw(tc,I,t)
                assert np.array_equal(arrays['eb3_nondiag'],arrays[f'step_{S}_eb3_out'])
                assert np.array_equal(arrays['eb4_nondiag'],arrays[f'step_{S}_eb4_out'])
        # All visible K/V operands are compact prefix allocations (DS1 pitch ABI is out of scope).
    arrays['valid_rows_history']=np.asarray(history,dtype=np.uint64)
    arrays['rotation']=raw(tc,I,w.rotation);arrays['codebook']=raw(tc,I,w.codebook)
    metadata={name:dict(dtype=str(a.dtype),shape=list(a.shape),sha256=hashlib.sha256(a.tobytes()).hexdigest(),nonzero=bool(np.any(a!=0)),nonconstant=bool(a.size>1 and np.any(a!=a.flat[0]))) for name,a in arrays.items()}
    # Every numerical reference must carry information; a constant field is a RED result, never waived.
    guards=all(r['nonzero'] and r['nonconstant'] for r in metadata.values())
    with (outdir/'integer_views.npz').open('xb') as f:np.savez(f,**arrays)
    save(outdir/'result.json',dict(status='PASS' if guards else 'FAIL_NONVacuous',provenance=provenance,case=case,arrays=metadata,nonvacuity_pass=guards,input_sha256={k:r['sha256'] for k,r in case['files'].items()},input_kind='real model post-norm/post-RoPE bf16 captures'))
    assert guards,'constant/zero numerical reference: see result.json'

def allocated(args):
    tc,I,provenance=runtime('port');S=args.S; model=args.model; NL=28 if model=='Qwen3-1.7B' else 36
    tc.set_alloc_pooling(False)
    owners=[];layers=[];bases=set()
    for layer in range(NL):
        disk();k=tc.zeros([1,8,S,128],dtype='bfloat16');v=tc.zeros([1,8,S,128],dtype='bfloat16')
        fields={'K_bf16':k,'V_bf16':v};w=None
        if layer:
            seed=tc.zeros([1,8,1,128],dtype='bfloat16');w=tc.ApaSpq1Workspace(seed,S,fmt='tq4')
            fields.update(tq4_codes=w.codes,tq4_norms=w.scales)
            del seed
        measured={}
        for key,t in fields.items():
            address,shape,dt,nbytes,base,pooled,is_cuda,device,offset=I.info(t)
            assert is_cuda and offset==0 and not pooled and base not in bases
            bases.add(base);measured[key]=int(nbytes)
        expected=8*S*(512 if layer==0 else 580)
        assert sum(measured.values())==expected
        layers.append(dict(layer=layer,measured_fields=measured,measured=sum(measured.values()),expected=expected))
        owners.append((k,v,w))
    total=sum(x['measured'] for x in layers);expected=8*S*(512+(NL-1)*580)
    assert total==expected
    receipt=load(ART/'reference_g2.json')['models'][model]['SP']['alloc']
    for capacity,r in receipt.items():assert r['measured']==r['arithmetic']==8*int(capacity)*(512+(NL-1)*580)
    save(Path(args.out)/'result.json',dict(status='PASS',provenance=provenance,model=model,S=S,KVH=8,layers=layers,measured=total,expected=expected,packed_per_key_per_kv_head=580,exact_layer0_per_key_per_kv_head=512,reference_receipt='reference_g2.json',evidence='Actual Storage.nbytes on live distinct allocations for all model layers, not NVML slope; shared rotation/codebook tables and transient seed/decode scratch excluded as in AB-1 G2.'))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['numerical','bytes']);ap.add_argument('--arm',choices=['lab','port'],default='port');ap.add_argument('--model',required=True);ap.add_argument('--S',type=int,required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
    lead_gpu();(numerical if args.mode=='numerical' else allocated)(args)
