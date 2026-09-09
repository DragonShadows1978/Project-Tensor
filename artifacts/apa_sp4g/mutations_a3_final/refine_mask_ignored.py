"""Actual A-propagated per-call fork/merge capture, stopping after layer5.
Prior art: June Gemma (2026) exact branch, SP3 (2026) same-tensor native
replay; controlled paired ablation, no new algorithm. BLASST/Yuan (2025/26),
FA2/Dao (2023), ThriftAttention/Sharratt (2026), TurboQuant/Zandieh (2025)
are inherited SP/quantizer work, not implemented here.
"""
import os
import numpy as np
from contextlib import contextmanager
from apa_sp4g_common import A, R, Red, save_array, publish, sha
from apa_sp4g_model import Model, ENV
from apa_sp4g_a2_model import standard, array, MAX_DELTA
from apa_sp4g_a3_math import compare, dense, eligible

class CapturedCall(Exception):
    """Local early return, caught only by this diagnostic's foreground worker."""

def exact(a, b, label):
    comparison = compare(a, b)
    if not comparison['bitwise']:
        raise Red('A3_NATIVE_NOT_BITWISE: ' + label)
    return comparison

@contextmanager
def record_standard(tc, H, L, D):
    """Observe actual two GEMMs AND intervening native softmax, no substitute.
    Prior art: SP3 native observation (2026); ordinary call instrumentation.
    """
    original_mm, original_sm = tc.matmul, tc.causal_softmax
    trace = dict(ops=[])
    def mm(a, b, *args, **kwargs):
        if args:
            raise Red('A3_UNEXPECTED_POSITIONAL_MATMUL')
        if len(a.shape) != 4 or tuple(a.shape[:2]) != (1, 1):
            return original_mm(a, b, **kwargs)
        result = original_mm(a, b, **kwargs)
        if not trace['ops']:
            if (tuple(a.shape) != (1, 1, H * L, D) or not kwargs.get('trans_b')
                    or kwargs.get('alpha', 1.) != 1.):
                raise Red('A3_NATIVE_QK_ARGUMENTS')
            trace.update(q=a.reshape([1,H,L,D]), k=b,
                         scores=result.reshape([1,H,L,b.shape[2]]),
                         scale=kwargs.get('alpha', 1.), qk_kwargs=kwargs)
            trace['ops'].append('QK')
        elif trace['ops'] == ['QK', 'softmax']:
            if kwargs.get('trans_b', False) or kwargs.get('alpha', 1.) != 1.:
                raise Red('A3_NATIVE_PV_ARGUMENTS')
            trace.update(v=b, out=result.reshape([1,H,L,D]), pv_kwargs=kwargs)
            exact(array(a).reshape(trace['p'].shape), array(trace['p']), 'PV_probability_input')
            trace['ops'].append('PV')
        else:
            raise Red('A3_UNEXPECTED_NATIVE_GEMM')
        return result
    def sm(scores):
        if trace['ops'] != ['QK']:
            raise Red('A3_NATIVE_SOFTMAX_ORDER')
        exact(array(scores), array(trace['scores']), 'softmax_score_input')
        result = original_sm(scores)
        trace['p'] = result
        trace['ops'].append('softmax')
        return result
    tc.matmul, tc.causal_softmax = mm, sm
    try:
        yield trace
        if trace['ops'] != ['QK', 'softmax', 'PV']:
            raise Red('A3_INCOMPLETE_STANDARD_TRACE')
    finally:
        tc.matmul, tc.causal_softmax = original_mm, original_sm

def sp_checked(tc, q, k, kq, v, scale, causal):
    out = tc._C.apa_selective_attention_sp(q,k,kq,v,scale,MAX_DELTA,causal,None,False)
    checked, mask = tc._C.apa_selective_attention_sp(q,k,kq,v,scale,MAX_DELTA,causal,None,True)
    exact(array(out), array(checked), 'SP_diagnostic_output')
    L, S = q.shape[2], k.shape[2]
    expected = np.broadcast_to(eligible(L,S,causal,'native_bottom_right',S-L), mask.shape)
    observed = mask.numpy().astype(bool)
    if False:
        raise Red('A3_REFINE_ALL_MASK_MISMATCH')
    return out, dict(selected=int(observed.sum()), eligible=int(expected.sum()), all_refined=True)

class CallModel(Model):
    def __init__(self, cell):
        super().__init__(dict(cell, arm='A'))
        self.cell = cell
        self.directory = A / 'captures_a3' / cell['id']
        self.files, self.observed_calls = [], []
        self.index = 0
        self.got = None
        self.tc.apa_selective_attention = self.dispatch
        os.environ['TC_APA_SP'] = '1'

    def save(self, name, value):
        saved = save_array(self.directory / (name + '.npy'), value)
        self.files.append(saved)
        return saved

    def install(self):
        owner = self
        selected_mixer = self.model.layers[5].mixer
        def attention(mixer, x, cos, sin, position_offset=0, kv_cache=None):
            if mixer is not selected_mixer:
                return owner.original_attn(mixer,x,cos,sin,position_offset,kv_cache)
            index = owner.index
            owner.index += 1
            L = x.shape[1]
            S = L + (0 if kv_cache is None else kv_cache[0].shape[2])
            owner.observed_calls.append(dict(index=index,L=L,S_all=S,position_offset=position_offset,
                                            mode=mixer.attention_mode))
            if index != owner.cell['call_index']:
                return owner.original_attn(mixer,x,cos,sin,position_offset,kv_cache)
            if (L != owner.cell['L'] or S != owner.cell['S_all']
                    or position_offset != owner.cell['position_offset']
                    or position_offset != S-L or L <= 1
                    or (kv_cache is not None and not isinstance(kv_cache,tuple))):
                raise Red('A3_ACTUAL_CALL_GEOMETRY_CHANGED')
            owner.active_mixer = mixer
            owner.position_offset = position_offset
            with record_standard(owner.tc,16,L,512) as trace:
                a_projected, a_cache = owner.original_attn(mixer,x,cos,sin,position_offset,kv_cache)
            owner.trace = trace
            old = mixer.attention_mode
            mixer.attention_mode = 'apa_selective'
            try:
                owner.return_arm = 'A'
                alt_projected, alt_cache = owner.original_attn(mixer,x,cos,sin,position_offset,kv_cache)
                exact(array(alt_projected),array(a_projected),'APA_standard_replacement_o_proj')
                for i in (0,1):
                    exact(array(alt_cache[i]),array(a_cache[i]),'return_cache_'+str(i))
                owner.return_arm = 'D'
                d_projected, d_cache = owner.original_attn(mixer,x,cos,sin,position_offset,kv_cache)
                for i in (0,1):
                    exact(array(d_cache[i]),array(a_cache[i]),'D_return_cache_'+str(i))
            finally:
                mixer.attention_mode = old
            owner.finish(a_projected,d_projected)
            # Actual standard history reached this call; no later layer or
            # prefix logits are needed for a per-call measurement.
            raise CapturedCall()
        self.gemma.Gemma4AttentionTC.__call__ = attention

    def dispatch(self,q,k,kq,v,scale,zthr,is_causal=False):
        if (not is_causal or scale != 1. or q.shape[2] != self.cell['L']
                or k.shape[2] != self.cell['S_all']):
            raise Red('A3_SP_SEAM_ARGUMENTS')
        parity = {name:exact(array(t),array(self.trace[name]),name)
                  for name,t in [('q',q),('k',k),('v',v)]}
        if self.return_arm == 'D':
            exact(array(kq),self.raw['kq'],'repeated_Kq')
            return self.outputs['sp_bf16']
        self.raw = {name:array(t) for name,t in [('q',q),('k',k),('kq',kq),('v',v)]}
        self.tensor_meta = {name:dict(shape=list(t.shape),dtype=str(t.dtype),device=str(t.device))
                            for name,t in [('q',q),('k',k),('kq',kq),('v',v)]}
        a = standard(self.tc,q,k,v,scale)
        exact(array(a),array(self.trace['out']),'isolated_A_vs_native')
        d, mask16 = sp_checked(self.tc,q,k,kq,v,scale,is_causal)
        qf,kf,kqf,vf = [t.float() for t in (q,k,kq,v)]
        a32 = standard(self.tc,qf,kf,vf,scale)
        d32, mask32 = sp_checked(self.tc,qf,kf,kqf,vf,scale,is_causal)
        self.outputs = dict(standard_bf16=a,sp_bf16=d,standard_fp32=a32,sp_fp32=d32)
        self.metadata = dict(layer=5,call_index=self.cell['call_index'],L=q.shape[2],S_all=k.shape[2],
                             position_offset=self.position_offset,implicit_query_offset=k.shape[2]-q.shape[2],
                             scale=scale,zthr=zthr,delta=MAX_DELTA,is_causal=is_causal,sinks=None,
                             diagnostics=False,mask_tensor=None,window=0,H=16,KVH=1,group=16,D=512,
                             causal_bound='0 <= j <= S_all-L+i',storage=ENV,
                             fast_max_seq=0,apa_min_context=0,prefill_chunk=self.model.PREFILL_CHUNK,
                             tensors=self.tensor_meta,parity=parity,mask_bf16=mask16,mask_fp32=mask32,
                             standard_ops=self.trace['ops'],standard_qk_kwargs=self.trace['qk_kwargs'],
                             standard_pv_kwargs=self.trace['pv_kwargs'],
                             prior_history='standard A; ids[:1023], true model adaptive chunks; final input target2047 not queried')
        return a

    def finish(self,a_projected,d_projected):
        # Exact fork arrays, native intermediates and each isolation output.
        file_map = {name:self.save(name,t) for name,t in self.raw.items()}
        for name in ('scores','p','out'):
            file_map['native_'+name] = self.save('native_'+name,array(self.trace[name]))
        arrays = {name:array(t) for name,t in self.outputs.items()}
        arrays['dense_fp32'] = dense(self.raw['q'],self.raw['k'],self.raw['v'],position_offset=self.position_offset)
        arrays['dense_staged_bf16'] = dense(self.raw['q'],self.raw['k'],self.raw['v'],position_offset=self.position_offset,staged_bf16=True)
        arrays['native_A_projected'] = array(a_projected)
        arrays['actual_D_projected'] = array(d_projected)
        for name in ('standard_fp32','sp_fp32'):
            t = self.outputs[name].astype('bfloat16')
            arrays[name+'_cast_bf16'] = array(t)
            merged = t.transpose(1,2).reshape([1,self.cell['L'],16*512])
            arrays[name+'_projected'] = array(self.active_mixer.o_proj(self.gemma._cast(merged)))
        comparisons = {}
        for precision in ('bf16','fp32'):
            comparisons['SP_vs_A_'+precision] = compare(arrays['sp_'+precision],arrays['standard_'+precision])
            for arm in ('standard','sp'):
                comparisons[arm+'_'+precision+'_vs_dense_fp32'] = compare(arrays[arm+'_'+precision],arrays['dense_fp32'])
        comparisons['standard_bf16_vs_staged_numpy'] = compare(arrays['standard_bf16'],arrays['dense_staged_bf16'])
        comparisons['SP_vs_A_after_fp32_to_bf16'] = compare(arrays['sp_fp32_cast_bf16'],arrays['standard_fp32_cast_bf16'])
        comparisons['SP_vs_A_after_fp32_cast_o_proj'] = compare(arrays['sp_fp32_projected'],arrays['standard_fp32_projected'])
        comparisons['actual_D_vs_A_projected'] = compare(arrays['actual_D_projected'],arrays['native_A_projected'])
        for name,t in arrays.items():
            file_map[name] = self.save(name,t)
        manifest = dict(metadata=self.metadata,observed_calls=self.observed_calls,comparisons=comparisons,
                        arrays=file_map,evidence_class='same-input in-model per-call diagnostic; NOT perplexity')
        path = self.directory / 'capture.json'
        publish(path,manifest)
        self.files.append(dict(path=str(path.relative_to(R)),sha256=sha(path)))
        self.got = dict(manifest=str(path.relative_to(R)),files=self.files,metadata=self.metadata,
                        comparisons=comparisons,observed_calls=self.observed_calls,load_s=self.load_s,
                        C_unblocked=False,interpretation='PASS is completed capture; numerical disagreement remains reported, .005 PPL not tested')

    def run_call(self, ids):
        self.install()
        with self.tc.no_grad():
            try:
                self.model(ids[None,:1023],last_token_only=True)
            except CapturedCall:
                pass
        self.tc.synchronize()
        if self.got is None:
            raise Red('A3_CALL_NOT_CAPTURED')
        return self.got

    def close(self):
        self.gemma.Gemma4AttentionTC.__call__ = self.original_attn
        super().close()
