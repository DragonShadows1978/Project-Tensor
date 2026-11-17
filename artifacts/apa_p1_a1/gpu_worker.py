# Prior art: round-1 worker allocated() retained verbatim (Project-Tensor 2026).
from worker_core import *

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
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['numerical','bytes']);ap.add_argument('--arm',choices=['lab','port'],default='port');ap.add_argument('--case');ap.add_argument('--model');ap.add_argument('--S',type=int);ap.add_argument('--out',required=True);args=ap.parse_args()
    lead_gpu();(numerical if args.mode=='numerical' else allocated)(args)
