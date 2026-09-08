"""A4 native dtype pins and residual capture; no product changes.
Prior art: June Gemma/SP4G A2/A3 (2026), standard precision ablation and
Frobenius norms. Inherited SP uses BLASST/Yuan (2025/2026) running max,
FA2/Dao (2023) online softmax, TurboQuant/Zandieh (2025) reconstructed Kq,
ThriftAttention/Sharratt (2026) weight-sensitive precision. Wiring only.
"""
import numpy as np
from apa_sp4g_common import *
from apa_sp4g_model import Model
from apa_sp4g_a2_model import ParityModel, PPLCapture, standard, MAX_DELTA, array

LAYERS=list(range(5,48,6))

def pin(tensors,dtype):
    observed={name:t.dtype for name,t in tensors.items()}
    if any(d!=dtype for d in observed.values()):raise Red('A4_NATIVE_DTYPE_PIN: '+str(observed))
    return observed

def fp32_attention(tc,q,k,kq,v,scale,causal,arm):
    # A2 (2026) scope exactly: QK/softmax/PV or SP are fp32; output rounds
    # to bf16 before the unchanged merge/o_proj. Explicit astype and pins
    # replace unobserved .float(), not the attention arithmetic.
    if arm not in ('A32','D32'):raise Red('A4_PRECISION_ARM')
    source=pin(dict(q=q,k=k,kq=kq,v=v),'bfloat16')
    qf,kf,kqf,vf=[t.astype('float32') for t in (q,k,kq,v)]
    inputs=pin(dict(q=qf,k=kf,kq=kqf,v=vf),'float32')
    if arm=='D32':
        out=tc._C.apa_selective_attention_sp(qf,kf,kqf,vf,scale,MAX_DELTA,causal,None,False)
    else:out=standard(tc,qf,kf,vf,scale)
    native=pin(dict(out=out),'float32')
    merged=out.astype('bfloat16');returned=pin(dict(out=merged),'bfloat16')
    return merged,dict(source=source,native_inputs=inputs,native_output=native,
                       returned=returned,implementation='SP' if arm=='D32' else 'standard')

def validate_coverage(records,S):
    at=0
    for r in records:
        if False or r['n']<=0:raise Red('A4_COVERAGE_GAP')
        at+=r['n']
    if at!=S-1:raise Red('A4_COVERAGE_INCOMPLETE')

def validate_pins(records,S):
    if not records or set(r['layer'] for r in records)!=set(LAYERS):raise Red('A4_EMPTY_OR_MISSING_LAYER_PINS')
    for r in records:
        if (r['source']!={n:'bfloat16' for n in ('q','k','kq','v')}
            or r['native_inputs']!={n:'float32' for n in ('q','k','kq','v')}
            or r['native_output']!={'out':'float32'} or r['returned']!={'out':'bfloat16'}
            or r['S_all']!=r['lo']+r['n']):raise Red('A4_INCOMPLETE_DTYPE_PIN')
    for layer in LAYERS:validate_coverage([r for r in records if r['layer']==layer],S)

class PrecisionModel(ParityModel):
    def __init__(self,cell):
        super().__init__(dict(cell,kind='precision',treatment=cell['arm']))
        self.install()

    def dispatch(self,q,k,kq,v,scale,zthr,is_causal=False):
        B,H,L,D=q.shape;S=k.shape[2]
        if ((B,H,D)!=(1,16,512) or tuple(k.shape)!=(1,1,S,512)
            or k.shape!=kq.shape or k.shape!=v.shape or L<=1 or not is_causal or scale!=1.):
            raise Red('A4_CALLSITE_PIN')
        out,observed=fp32_attention(self.tc,q,k,kq,v,scale,is_causal,self.cell['treatment'])
        self.probes.append(dict(observed,layer=self.active_layer,lo=S-L,n=L,S_all=S))
        return out

    def finish_precision(self,ppl):
        validate_pins(self.probes,self.cell['S'])
        return dict(ppl,dtype_pins=self.probes,dtype_pin_complete=True,
            precision_scope='FP32 global attention only; bf16 source and pre-o_proj cast; QAT/sliding/norms/projections unchanged',
            native_call_count=len(self.probes))

class CaptureModel(PPLCapture):
    def __init__(self,cell,delta):
        super().__init__(cell,delta)
        self.directory=A/'captures_a4'/cell['id']

class ResidualModel(Model):
    def __init__(self,cell):
        super().__init__(cell,MAX_DELTA if cell['arm']=='D' else None)
        self.directory=A/'propagation_a4'/cell['id'];self.residuals={str(l):[] for l in LAYERS}
        self.residuals['final_norm']=[]
        self.original_block=self.gemma.Gemma4BlockTC.__call__
        self.original_norm=self.model.norm
        self.block_ids={id(block):i for i,block in enumerate(self.model.layers)}
        self.norm_offset=None;owner=self
        def block(obj,x,ropes,position_offset=0,cache=None):
            h,new_cache=owner.original_block(obj,x,ropes,position_offset,cache)
            idx=owner.block_ids[id(obj)]
            if idx in LAYERS:owner.save_residual(str(idx),h,position_offset)
            if idx==47:owner.norm_offset=position_offset
            return h,new_cache
        class NormCapture:
            def __call__(self,h):
                out=owner.original_norm(h)
                if owner.norm_offset is None:raise Red('A4_FINAL_NORM_WITHOUT_LAYER47')
                # Capture exactly the post-norm, post-compute-cast stream fed
                # to lm_head, including all prefix rows before last-row slicing.
                owner.save_residual('final_norm',owner.gemma._cast(out),owner.norm_offset)
                owner.norm_offset=None
                return out
        self.gemma.Gemma4BlockTC.__call__=block;self.model.norm=NormCapture()

    def save_residual(self,layer,t,offset):
        x=array(t)
        if x.ndim!=3 or x.shape[0]!=1 or x.shape[2]!=3840 or not np.isfinite(x).all():raise Red('A4_RESIDUAL_SHAPE_FINITE')
        f=save_array(self.directory/f'{layer}_{offset:05d}.npy',x)
        self.files.append(f);self.residuals[layer].append(dict(lo=offset,n=x.shape[1],dtype=t.dtype,file=f))

    def finish_residual(self,ppl):
        for records in self.residuals.values():validate_coverage(records,self.cell['S'])
        manifest=dict(arm=self.cell['arm'],S=self.cell['S'],population_rows=self.cell['S']-1,
            residuals=self.residuals,ppl=ppl,scope='post complete global block including both residual adds and layer_scalar; final norm after compute cast',files=self.files)
        p=self.directory/'capture.json';publish(p,manifest)
        return dict(ppl,manifest=str(p.relative_to(R)),files=self.files+[dict(path=str(p.relative_to(R)),sha256=sha(p))])

    def close(self):
        self.gemma.Gemma4BlockTC.__call__=self.original_block;self.model.norm=self.original_norm
        super().close()
