"""Scored-block same-input capture. Prior art: SP4G A3/A4 (2026) native
observation, replay, precision pins and fork/merge; Vaswani et al. (2017)
dense attention, ordinary relative Frobenius norm. New work selects the
actual scored calls and receipts dispatch/cache metadata; no kernel changes.
Inherited SP: BLASST/Yuan et al. (2025/26), FA2/Dao (2023),
ThriftAttention/Sharratt (2026), TurboQuant/Zandieh et al. (2025).
"""
import time
import numpy as np
from apa_sp4g_a3_model import CallModel, CapturedCall, record_standard, exact
from apa_sp4g_a3_math import compare, eligible
from apa_sp4g_a2_model import standard, array, MAX_DELTA
from apa_sp4g_a4_model import pin
from apa_sp4g_common import A, R, Red, sha
from apa_sp4g_model import ENV


def geometry(cell, index, L, S, offset, cache):
    if (index != cell['call_index'] or L != cell['L'] or L != 64
            or S != cell['S_all'] or offset != cell['position_offset']
            or offset != S-L or not isinstance(cache,tuple)
            or len(cache) != 2 or cache[0].shape[2] != offset):
        raise Red('A5_ACTUAL_SCORED_CALL_GEOMETRY_CHANGED')
    return dict(type='tuple', count_before=offset, count_after=S,
                kq_count=None, kq_count_reason='KVRing-only field; L>1 uses tuple and whole-span reconstructed Kq',
                quantized_rows=S, quantization='whole-span _quantize_keys(k); S_all<=4096')


def launch_metadata(L, S, dtype, diagnostics):
    # Direct transcription of pinned TensorCUDA kernels.cu:7477-7503.
    # Source-determined dispatch from observed arguments; NOT profiler data.
    split = L < S
    return dict(entry='tc._C.apa_selective_attention_sp',
                path='split-K/decode' if split else 'prefill',
                symbol='apa_selective_sp_splitk_dispatch' if split else 'apa_selective_sp_kernel',
                rule='L == 1 selects split-K; every L > 1 selects prefill',
                L=L, S_all=S, dtype=dtype, CAP=512, diagnostics=diagnostics,
                grid_blocks=None if split else 16*L, threads=None if split else 32,
                evidence_class='pinned launcher source evaluated on observed native arguments; not GPU trace',
                source='tensor_cuda/src/kernels.cu', source_sha256=sha(R/'tensor_cuda/src/kernels.cu'))


def classify(comparisons, metadata):
    sp_a = comparisons['SP_vs_A_fp32']['relative_frobenius']
    sp_dense = comparisons['sp_fp32_vs_dense_fp32']['relative_frobenius']
    a_dense = comparisons['standard_fp32_vs_dense_fp32']['relative_frobenius']
    masks = [metadata['mask_'+d] for d in ('bf16','fp32')]
    valid = all(m['diagnostic_output_bitwise'] and m['all_refined'] for m in masks)
    if not valid:
        outcome = 'SP_DIAGNOSTIC_OR_REFINE_ALL_DEFECT_LEAD_DECISION'
    elif sp_a <= .001 and sp_dense <= .001 and a_dense <= .001:
        outcome = 'AGREES_AT_REGISTERED_THRESHOLD_PROPAGATION_REMAINS'
    elif sp_a > .001 and sp_dense > .001 and a_dense <= .001:
        outcome = 'SP_PATH_DEFECT_WITH_PINNED_ARGUMENTS_LEAD_DECISION'
    else:
        outcome = 'UNRESOLVED_STANDARD_OR_REFERENCE_ALSO_DISAGREES'
    return dict(outcome=outcome, prediction_disagreement_confirmed=sp_a > .001,
                threshold=.001, arguments='parity, causal offset, scale, tuple cache and dtype pinned',
                mechanism='No internal kernel root cause inferred from error magnitude; use recorded native arguments and source path.',
                kernel_fix=False, C_E_unblocked=False)


class ScoredCallModel(CallModel):
    def __init__(self, cell):
        super().__init__(cell)
        self.directory = A/'captures_a5'/cell['id']
        self.deadline = time.monotonic()+cell['worker_s']

    def check_deadline(self):
        if time.monotonic() >= self.deadline:
            raise Red('A5_COOPERATIVE_WORKER_RAIL')

    def install(self):
        owner = self
        selected = self.model.layers[self.cell['layer']].mixer
        def attention(mixer,x,cos,sin,position_offset=0,kv_cache=None):
            owner.check_deadline()
            if mixer is not selected:
                return owner.original_attn(mixer,x,cos,sin,position_offset,kv_cache)
            index = owner.index
            owner.index += 1
            L = x.shape[1]
            S = L+(0 if kv_cache is None else kv_cache[0].shape[2])
            owner.observed_calls.append(dict(layer=owner.cell['layer'],index=index,L=L,S_all=S,
                                            position_offset=position_offset,mode=mixer.attention_mode))
            if index != owner.cell['call_index']:
                return owner.original_attn(mixer,x,cos,sin,position_offset,kv_cache)
            owner.cache_metadata = geometry(owner.cell,index,L,S,position_offset,kv_cache)
            owner.active_mixer, owner.position_offset = mixer, position_offset
            with record_standard(owner.tc,16,L,512) as trace:
                a_projected,a_cache = owner.original_attn(mixer,x,cos,sin,position_offset,kv_cache)
            owner.trace = trace
            old = mixer.attention_mode
            mixer.attention_mode = 'apa_selective'
            try:
                owner.return_arm = 'A'
                alt_projected,alt_cache = owner.original_attn(mixer,x,cos,sin,position_offset,kv_cache)
                exact(array(alt_projected),array(a_projected),'A5_standard_replacement_o_proj')
                for i in (0,1):
                    exact(array(alt_cache[i]),array(a_cache[i]),'A5_return_cache_'+str(i))
                owner.return_arm = 'D'
                d_projected,d_cache = owner.original_attn(mixer,x,cos,sin,position_offset,kv_cache)
                for i in (0,1):
                    exact(array(d_cache[i]),array(a_cache[i]),'A5_D_return_cache_'+str(i))
            finally:
                mixer.attention_mode = old
            owner.finish(a_projected,d_projected)
            owner.got['classification'] = classify(owner.got['comparisons'],owner.metadata)
            raise CapturedCall()
        self.gemma.Gemma4AttentionTC.__call__ = attention

    def sp_observed(self,q,k,kq,v,scale,causal,label):
        self.check_deadline()
        native_inputs = pin(dict(q=q,k=k,kq=kq,v=v),'float32' if label=='fp32' else 'bfloat16')
        out = self.tc._C.apa_selective_attention_sp(q,k,kq,v,scale,MAX_DELTA,causal,None,False)
        checked,mask = self.tc._C.apa_selective_attention_sp(q,k,kq,v,scale,MAX_DELTA,causal,None,True)
        pin(dict(out=out,checked=checked),q.dtype)
        observed = mask.numpy().astype(bool)
        L,S = q.shape[2],k.shape[2]
        expected = np.broadcast_to(eligible(L,S,causal,'native_bottom_right',S-L),observed.shape)
        consistency = compare(array(out),array(checked))
        self.save('SP_selected_'+label,observed)
        return out, dict(selected=int(observed.sum()),eligible=int(expected.sum()),
            all_refined=bool(np.array_equal(observed,expected)),
            missing_eligible=int(np.count_nonzero(expected & ~observed)),
            selected_outside_causal=int(np.count_nonzero(observed & ~expected)),
            diagnostic_output_bitwise=consistency['bitwise'],diagnostic_comparison=consistency,
            native_inputs=native_inputs,native_output=out.dtype,
            dispatch=launch_metadata(L,S,q.dtype,False),diagnostic_dispatch=launch_metadata(L,S,q.dtype,True))

    def dispatch(self,q,k,kq,v,scale,zthr,is_causal=False):
        self.check_deadline()
        if (not is_causal or scale != 1. or tuple(q.shape) != (1,16,64,512)
                or tuple(k.shape) != (1,1,self.cell['S_all'],512)
                or k.shape != kq.shape or k.shape != v.shape):
            raise Red('A5_SP_SEAM_ARGUMENTS')
        parity = {name:exact(array(t),array(self.trace[name]),name)
                  for name,t in [('q',q),('k',k),('v',v)]}
        if self.return_arm == 'D':
            exact(array(kq),self.raw['kq'],'A5_repeated_Kq')
            return self.outputs['sp_bf16']
        pin(dict(q=q,k=k,kq=kq,v=v),'bfloat16')
        self.raw = {name:array(t) for name,t in [('q',q),('k',k),('kq',kq),('v',v)]}
        a = standard(self.tc,q,k,v,scale)
        exact(array(a),array(self.trace['out']),'A5_isolated_A_vs_native')
        d,mask16 = self.sp_observed(q,k,kq,v,scale,is_causal,'bf16')
        qf,kf,kqf,vf = [t.astype('float32') for t in (q,k,kq,v)]
        a32 = standard(self.tc,qf,kf,vf,scale)
        pin(dict(out=a32),'float32')
        d32,mask32 = self.sp_observed(qf,kf,kqf,vf,scale,is_causal,'fp32')
        self.outputs = dict(standard_bf16=a,sp_bf16=d,standard_fp32=a32,sp_fp32=d32)
        offset,S = self.position_offset,k.shape[2]
        self.metadata = dict(layer=self.cell['layer'],block=self.cell['block'],call_index=self.cell['call_index'],
            L=64,S_all=S,order_stated_S_all=self.cell['order_stated_S_all'],
            position_offset=offset,implicit_query_offset=S-64,
            causal_bound='0 <= j <= S_all-L+i',causal_last_key_per_query=(offset+np.arange(64)).tolist(),
            visible_keys_per_query=(offset+np.arange(64)+1).tolist(),
            scale=scale,zthr=zthr,delta=MAX_DELTA,is_causal=is_causal,sinks=None,mask_tensor=None,
            H=16,KVH=1,group=16,D=512,window=0,storage=ENV,fast_max_seq=0,apa_min_context=0,
            prefill_chunk=self.model.PREFILL_CHUNK,cache=self.cache_metadata,
            kq_count=self.cache_metadata['kq_count'],parity=parity,mask_bf16=mask16,mask_fp32=mask32,
            tensors={name:dict(shape=list(t.shape),dtype=t.dtype,device=str(t.device))
                     for name,t in [('q',q),('k',k),('kq',kq),('v',v)]},
            standard_ops=self.trace['ops'],standard_qk_kwargs=self.trace['qk_kwargs'],standard_pv_kwargs=self.trace['pv_kwargs'],
            prior_history='standard A, exact Model.perplexity schedule; two prefill chunks then scored L64 cached blocks; not D-propagated inputs',
            prior_art='SP4G A3/A4 (2026) replay/reference; Vaswani et al. (2017); observation wiring only')
        return a

    def run_call(self, ids):
        self.install()
        try:
            # Reuse the exact scoring driver: no fabricated 1024-token prefix.
            self.perplexity(ids,1024)
        except CapturedCall:
            pass
        self.tc.synchronize()
        if self.got is None:
            raise Red('A5_SCORED_CALL_NOT_CAPTURED')
        self.check_deadline()
        return self.got
