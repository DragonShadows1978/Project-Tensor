"""Controlled same-state attention probes and actual-PPL capture.
Prior art: June Gemma (2026) exact MQA standard implementation; SP3 (2026)
same-tensor reference, native-mask replay. Standard precision ablation and
IEEE754 bit payload storage; no novel attention/selection algorithm.
"""
import numpy as np
from apa_sp4g_model import Model,ENV
from apa_sp4g_common import *

MAX_DELTA=float(np.finfo(np.float32).max)

def compare(a,b):
    a=np.asarray(a);b=np.asarray(b)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise Red('A2_INVALID_COMPARISON')
    diff=np.abs(a.astype(np.float64)-b.astype(np.float64))
    return dict(bitwise=bool(np.array_equal(a,b)),max_abs=float(diff.max()),
                max_relative=float((diff/np.maximum(np.abs(b),1e-30)).max()),
                relative_denominator='max(abs(reference),1e-30); descriptive only',different=int(np.count_nonzero(diff)))

def array(t):return t.float().numpy()

def encode_bf16(x):
    # IEEE754 / NumPy (system; year unverified lead to check): lossless upper
    # 16 bits of BF16-representable FP32; no numerical rounding/compression.
    x=np.ascontiguousarray(x,dtype=np.float32);u=x.view(np.uint32)
    if not np.isfinite(x).all():raise Red('A2_CAPTURE_NOT_BF16_EXACT')
    return (u>>16).astype(np.uint16)

def decode_bf16(x):return (np.asarray(x,dtype=np.uint32)<<16).view(np.float32)

def standard(tc,q,k,v,scale=1.):
    # Literal June Gemma4 global standard branch (2026), :729-745; exact K,
    # shared MQA, bottom-right causal. Kq is deliberately not an input.
    B,H,L,D=q.shape;S=k.shape[2]
    sc=tc.matmul(q.reshape([B,1,H*L,D]),k,alpha=scale,trans_b=True).reshape([B,H,L,S])
    p=tc.causal_softmax(sc)
    return tc.matmul(p.reshape([B,1,H*L,S]),v).reshape([B,H,L,D])

class ParityModel(Model):
    def __init__(self,cell):
        super().__init__(dict(cell,arm='A'))
        self.cell=cell;self.probes=[];self.probed=set();self.original_matmul=self.tc.matmul
        self.layer_ids={id(x.mixer):i for i,x in enumerate(self.model.layers)}
        self.tc.apa_selective_attention=self.dispatch
        os.environ['TC_APA_SP']='1'

    def install(self):
        owner=self
        def attention(mixer,x,cos,sin,position_offset=0,kv_cache=None):
            idx=owner.layer_ids[id(mixer)]
            if not mixer.is_global:return owner.original_attn(mixer,x,cos,sin,position_offset,kv_cache)
            if owner.cell['kind']=='precision':
                old=mixer.attention_mode;mixer.attention_mode='apa_selective'
                owner.active_layer=idx
                try:return owner.original_attn(mixer,x,cos,sin,position_offset,kv_cache)
                finally:mixer.attention_mode=old
            if idx in owner.probed:return owner.original_attn(mixer,x,cos,sin,position_offset,kv_cache)
            if x.shape[1]<=1 or (kv_cache is not None and not isinstance(kv_cache,tuple)):raise Red('A2_SAME_STATE_REQUIRES_IMMUTABLE_PREFILL_CACHE')
            owner.active_layer=idx;owner.standard_inputs={}
            def matmul(a,b,*args,**kw):
                # Record only the two standard global attention GEMMs.
                if len(a.shape)==4 and len(b.shape)==4 and a.shape[1]==1 and b.shape[1]==1:
                    if kw.get('trans_b') and a.shape[-1]==512:
                        owner.standard_inputs['q']=array(a).reshape(1,16,x.shape[1],512)
                        owner.standard_inputs['k']=array(b)
                        owner.standard_inputs['scale']=kw.get('alpha',1.)
                    elif 'q' in owner.standard_inputs and b.shape[-1]==512:
                        owner.standard_inputs['v']=array(b)
                return owner.original_matmul(a,b,*args,**kw)
            owner.tc.matmul=matmul
            try:out,cache=owner.original_attn(mixer,x,cos,sin,position_offset,kv_cache)
            finally:owner.tc.matmul=owner.original_matmul
            old=mixer.attention_mode;mixer.attention_mode='apa_selective'
            try:alt,_=owner.original_attn(mixer,x,cos,sin,position_offset,kv_cache)
            finally:mixer.attention_mode=old
            owner.probes[-1]['post_o_projection']=compare(array(alt),array(out))
            owner.probes[-1]['position_offset']=position_offset
            owner.probed.add(idx)
            return out,cache
        self.gemma.Gemma4AttentionTC.__call__=attention

    def dispatch(self,q,k,kq,v,scale,zthr,is_causal=False):
        tc=self.tc
        if not is_causal or q.shape[2]<=1 or scale!=1.:raise Red('A2_GLOBAL_PREFILL_PIN')
        if self.cell['kind']=='precision':
            qf,kf,kqf,vf=[t.float() for t in (q,k,kq,v)]
            a=standard(tc,qf,kf,vf,scale)
            d=tc._C.apa_selective_attention_sp(qf,kf,kqf,vf,scale,MAX_DELTA,True,None,False)
            self.probes.append(dict(layer=self.active_layer,L=q.shape[2],S=k.shape[2],fp32_D_vs_standard=compare(array(d),array(a))))
            return (a if self.cell['treatment']=='A32' else d).astype('bfloat16')
        refs=self.standard_inputs
        parity={n:compare(array(t),refs[n]) for n,t in [('q',q),('k',k),('v',v)]}
        a=standard(tc,q,k,v,scale)
        d=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,MAX_DELTA,True,None,False)
        a32=standard(tc,q.float(),k.float(),v.float(),scale)
        d32=tc._C.apa_selective_attention_sp(q.float(),k.float(),kq.float(),v.float(),scale,MAX_DELTA,True,None,False)
        self.probes.append(dict(layer=self.active_layer,tensors=parity,scale_APA=scale,scale_A=refs['scale'],
            storage_flags=ENV,exact_K_vs_bulk_Kq=compare(array(k),array(kq)),
            bf16_D_vs_standard=compare(array(d),array(a)),fp32_D_vs_standard=compare(array(d32),array(a32)),
            standard_bf16_vs_fp32=compare(array(a),array(a32))))
        return a

    def close(self):
        self.gemma.Gemma4AttentionTC.__call__=self.original_attn
        self.tc.matmul=self.original_matmul
        super().close()

class PPLCapture(Model):
    def __init__(self,cell,delta=None):
        super().__init__(cell,delta,capture=False,observe=True)
        self.capture=True;self.directory=A/'captures_a2'/cell['id']

    def diagnostic_dispatch(self,q,k,kq,v,scale,zthr,is_causal=False):
        # SP3 native mask observation: returned output is the original native
        # call; instrumentation must reproduce it BITWISE before any save.
        tc=self.tc;layer=5+6*(self.calls%8);self.calls+=1
        B,H,L,D=q.shape;S=k.shape[2]
        if (B,H,D)!=(1,16,512) or tuple(k.shape)!=(1,1,S,512) or k.shape!=kq.shape or k.shape!=v.shape or scale!=1. or L<=1 or not is_causal:raise Red('CALLSITE_PARITY_PIN')
        if self.cell['arm']=='B':
            out=self.original(q,k,kq,v,scale,zthr,is_causal)
            check,mask,_=self.diag.selective(q,k,kq,v,scale,zthr,is_causal)
        else:
            out=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,self.delta,is_causal,None,False)
            check,mask=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,self.delta,is_causal,None,True)
        if not np.array_equal(array(out),array(check)):raise Red('A2_CAPTURE_NATIVE_NOT_BITWISE')
        m=mask.numpy().astype(bool);lengths=S-L+np.arange(L)+1
        eligible=np.arange(S)<lengths[:,None]
        if np.any(m & ~eligible[None,None]):raise Red('MASK_OUTSIDE_CAUSAL')
        self.counts[layer]['pairs']+=int(H*lengths.sum());self.counts[layer]['selected']+=int(m.sum())
        idx=len(self.records[layer]);directory=self.directory/f'l{layer:02d}'
        rec=dict(lo=S-L,n=L,S=S,causal=is_causal,scale=scale,zthr=zthr,files={},mask_shape=list(m.shape),storage='bf16_uint16_and_mask_packbits_little')
        for name,t in [('q',q),('k',k),('kq',kq),('v',v),('out',out)]:
            f=save_array(directory/f'{idx:04d}_{name}.npy',encode_bf16(array(t)));self.files.append(f);rec['files'][name]=f
        f=save_array(directory/f'{idx:04d}_mask.npy',np.packbits(m.ravel(),bitorder='little'));self.files.append(f);rec['files']['mask']=f
        self.records[layer].append(rec)
        return out

    def finish_capture(self,ppl):
        rows=self.cell['S']-1
        for layer,recs in self.records.items():
            at=0
            for rec in recs:
                if rec['lo']!=at or rec['S']!=rec['lo']+rec['n']:raise Red('A2_CAPTURE_COVERAGE_GAP')
                at+=rec['n']
            if at!=rows:raise Red('A2_PPL_QUERY_COVERAGE')
        manifest=dict(arm=self.cell['arm'],S=self.cell['S'],population_rows=rows,delta=self.delta,
            records=self.records,files=self.files,ppl=ppl,ppl_cell=self.cell['id'],
            coverage='All queries actually executed by PROTOCOL-G PPL, 0..S-2; same native selections; final input token is a target only',**self.counts_result())
        p=self.directory/'capture.json';publish(p,manifest)
        return dict(ppl,manifest=str(p.relative_to(R)),population_rows=rows,
            files=self.files+[dict(path=str(p.relative_to(R)),sha256=sha(p))])
