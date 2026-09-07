"""June-stack decode without diagnostic layer hooks; no product edits.

Prior art: GraftRepository MiniCPM3/TensorCUDA June 2026 results/adapter fast
stack, DeepSeek-V2 (DeepSeek-AI 2024) absorbed MLA as credited in local sources
(external reference unverified; lead search: DeepSeek V2 MLA absorption).
Reuse existing fused kernels, cache and native SP; new experiment wiring only.
NVIDIA CUDA 12.6 default pool high-water counters are reused from A5.
"""
import ctypes
import hashlib
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
from apa_sp3_common import (ART, BUILD, Red, load_runtime, registration, protocol,
                           verify_weight_stat, require_pass, publish, read, sha, job_path)
from apa_sp3_a5_decode import PoolPeak


def check_interposer(enabled):
    # Prior art: process-map and symbol inspection, ordinary loader diagnostics.
    # Check loaded state, not just the env: unsetting LD_PRELOAD cannot unload it.
    loaded='libapa_sp3_peak.so' in Path('/proc/self/maps').read_text()
    symbol=hasattr(ctypes.CDLL(None),'apa_sp3_live_bytes')
    env=os.environ.get('LD_PRELOAD','')
    if enabled:
        if not loaded or not symbol:
            raise Red('A6_INTERPOSER_NOT_LOADED')
    elif loaded or symbol or env:
        raise Red('A6_UNEXPECTED_PRELOAD_OR_INTERPOSER')
    return dict(loaded=loaded,symbol=symbol,LD_PRELOAD=env)


def load_adapter(tc):
    sys.dont_write_bytecode=True
    sys.path.insert(0,'/mnt/ForgeRealm/GraftRepository')
    from core import minicpm3_tc as mini, mistral7b_tc as base
    if (Path(tc._C.__file__).resolve().parent!=BUILD or
        Path(mini.__file__).resolve()!=Path('/mnt/ForgeRealm/GraftRepository/core/minicpm3_tc.py')):
        raise Red('A6_WRONG_RUNTIME_ADAPTER')
    return mini,base


class CleanModel:
    def __init__(self, cell, delta=None):
        self.cell,self.cfg,self.delta=cell,cell['config'],delta
        self.interposer=check_interposer(self.cfg['interposer'])
        verify_weight_stat()
        self.tc=tc=load_runtime()
        for k in list(os.environ):
            if k.startswith('TC_'):del os.environ[k]
        os.environ.update(TC_APA_SP='1' if cell['arm']=='C' else '0',
                          TC_APA_SELECTIVE_PATH='0',TC_ATTN_QTILE='0')
        tc.set_alloc_pooling(self.cfg['pool_before_load'])
        self.mini,self.base=mini,base=load_adapter(tc)
        self.original_attn=mini.MLAAttentionTC.__call__
        self.original_blend=base._cublas_blend_attention
        self.original_selective=tc.apa_selective_attention
        base.QuantLinearTC.WEIGHT_BITS=4
        base.QuantLinearTC.USE_FUSED=False
        base.QuantLinearTC.FUSED_DECODE=self.cfg['fused_decode']
        base.RMSNormTC.USE_FUSED=self.cfg['fused_rms_norm']
        base.F.USE_FUSED_SOFTMAX=self.cfg['fused_softmax']
        try:
            with tc.no_grad():
                self.model,self.info=mini.MiniCPM3_TC.from_pretrained(registration()['model']['snapshot'])
            if not self.cfg['pool_before_load']:tc.set_alloc_pooling(True)
            if (self.info['weight_bits']!=4 or base.BlockTC.COMPUTE_DTYPE!='bfloat16'
                or base.LinearTC.DTYPE!='bfloat16' or base.GROUP_SIZE!=128):
                raise Red('A6_WEIGHT_DTYPE_GROUP_PIN')
            if len(self.model.layers)!=62:raise Red('A6_LAYER_COUNT')
            for layer in self.model.layers:
                a=layer.self_attn
                a.attention_mode='standard' if cell['arm']=='A' else 'apa_selective'
                a.bulk_bits,a.refine_percentile=cell['bits'],.10
                a.absorbed_decode=self.cfg['absorbed_decode']
                a.fast_max_seq=4096 if cell['arm']=='B' else 0
                a.telemetry=False
                a._capture=False
            if cell['arm']=='C':
                if delta is None or not math.isfinite(delta) or delta<0:raise Red('A6_DELTA_PIN')
                # Prior art: SP3 C native binding routing, unchanged kernels.
                # This is the required algorithm dispatch, not a diagnostic
                # layer hook. Never captures, counts, or copies attention.
                def native_sp(q,k,kq,v,scale,zthr,is_causal=False):
                    return tc._C.apa_selective_attention_sp(q,k,kq,v,scale,delta,
                                                            is_causal,None,False)
                tc.apa_selective_attention=native_sp
            if self.cfg['attention_wrapper']:
                # Prior art: exact SP3 owner.original_attn diagnostic wrapper
                # (2026), observation OFF; one-factor instrumentation rung.
                owner=self
                self.layer_ids={id(l.self_attn):i for i,l in enumerate(self.model.layers)}
                def call(attention,*args,**kwargs):
                    owner.layer=owner.layer_ids[id(attention)]
                    return owner.original_attn(attention,*args,**kwargs)
                mini.MLAAttentionTC.__call__=call
            if cell['kind']=='decode_clean' or not self.cfg['attention_wrapper']:
                self.assert_no_hook()
            self.peak=PoolPeak()
        except BaseException:
            self.close()
            raise

    def assert_no_hook(self):
        if (self.mini.MLAAttentionTC.__call__ is not self.original_attn or
            self.base._cublas_blend_attention is not self.original_blend):
            raise Red('A6_DIAGNOSTIC_HOOK_INSTALLED')
        if self.cell['arm']!='C' and self.tc.apa_selective_attention is not self.original_selective:
            raise Red('A6_UNEXPECTED_SELECTIVE_WRAPPER')

    def close(self):
        self.mini.MLAAttentionTC.__call__=self.original_attn
        self.base._cublas_blend_attention=self.original_blend
        self.tc.apa_selective_attention=self.original_selective


def scalar_argmax(tc, logits):
    # Prior art: existing TensorCUDA device argmax (2026), not a new reduction.
    # Copy one int64, never the 73448-vocabulary vector for token selection.
    last=logits.slice(1,logits.shape[1]-1,1) if logits.shape[1]!=1 else logits
    a=tc.argmax_last_axis(last).numpy()
    if a.size!=1:raise Red('A6_ARGMAX_NOT_SCALAR')
    token=int(a.reshape(-1)[0])
    if not 0<=token<73448:raise Red('A6_ARGMAX_OUT_OF_BOUNDS')
    return token


def cache_pin(cache, length):
    # Prior art: MLA adapter latent cache shape invariant (GraftRepository 2026).
    # Host metadata only; no per-layer attention wrapper or tensor host copy.
    if len(cache)!=62 or any(tuple(c[0].shape)!=(1,length,256) or tuple(c[1].shape)!=(1,1,length,32) for c in cache):
        raise Red('A6_CACHE_GEOMETRY')


def measure(owner, ids):
    c,cfg,tc,model=owner.cell,owner.cfg,owner.tc,owner.model
    S,steps=c['S'],c['steps']
    if steps!=32 or c['warmup_steps']!=1 or len(ids)<S+steps:raise Red('A6_STEP_STREAM_PIN')
    model.extend_rope(S+steps)
    tc.synchronize()
    owner.peak.reset()
    t=time.perf_counter()
    with tc.no_grad():
        lg,cache=model(ids[:S][None],last_token_only=cfg['last_token_only'])
    tc.synchronize()
    prefill_s=time.perf_counter()-t
    cache_pin(cache,S)
    seed=scalar_argmax(tc,lg) if c['feeding']=='greedy' else int(ids[S])
    del lg
    def call(pos, token, cache):
        if cfg['use_cache']:
            return model(np.array([[token]],dtype=np.int64),kv_caches=cache,
                         position_offset=pos,last_token_only=cfg['last_token_only'])
        return model(ids[:pos+1][None],last_token_only=cfg['last_token_only'])
    # Prior art: standard steady-state benchmark warmup; adapter lazily
    # materializes absorbed matrices. Discard new cache, reuse original prefix.
    t=time.perf_counter()
    with tc.no_grad():warm_lg,warm_cache=call(S,seed,cache)
    tc.synchronize()
    cache_pin(warm_cache,S+1)
    del warm_lg,warm_cache
    warmup_s=time.perf_counter()-t
    times,forward_times,chosen,fed=[],[],[],[]
    token=seed
    loop=time.perf_counter()
    with tc.no_grad():
        for pos in range(S,S+steps):
            if c['feeding']=='teacher_forced':token=int(ids[pos])
            tc.synchronize()
            t=time.perf_counter()
            lg,cache=call(pos,token,cache)
            tc.synchronize()
            forward_times.append(time.perf_counter()-t)
            if cfg['full_logits_host_copy'] and not np.isfinite(lg.float().numpy()).all():
                raise Red('nonfinite decode logits')
            next_token=scalar_argmax(tc,lg)
            times.append(time.perf_counter()-t)
            chosen.append(next_token);fed.append(token);token=next_token
            # Free old logits before next timer, as in the existing SP3 loop.
            if pos!=S+steps-1:del lg
    tc.synchronize()
    decode_work_s=time.perf_counter()-loop
    cache_pin(cache,S+steps)
    # Single final validation is outside ALL timed decode work, not per step.
    final_finite=bool(np.isfinite(lg.float().numpy()).all())
    if not final_finite:raise Red('A6_FINAL_LOGITS_NONFINITE')
    if len(times)!=32 or any(not math.isfinite(t) or t<=0 for t in times+forward_times):
        raise Red('A6_TIMING_PIN')
    peak=owner.peak.result()
    peak['peak_status']='POOL ONLY, reserved/used default pool high water since prefill; may include pooled weights enabled before load; excludes raw allocations/context; NOT device resident peak'
    return dict(tokens_s=32/sum(times),ms_token=1000*sum(times)/32,steps=32,
                seconds_per_step=times,forward_seconds_per_step=forward_times,
                forward_ms_token=1000*sum(forward_times)/32,
                timing='synchronized step wall including scalar argmax; forward-only wall also recorded',
                prefill_s=prefill_s,warmup_s=warmup_s,decode_work_s=decode_work_s,
                feeding=c['feeding'],fed_tokens=fed,argmax_tokens=chosen,
                fed_sha256=hashlib.sha256(np.asarray(fed,dtype='<i8').tobytes()).hexdigest(),
                final_logits_finite=final_finite,attention_contexts=[S+1,S+32],
                trained_window_exceeded=S+32>32768,**peak)


def planning(cell):
    if cell.get('nonfit'):
        return dict(fit=False,outcome=cell['nonfit'],estimate_s=None)
    if cell['S']!=32768:return dict(fit=None,outcome='UNMEASURED_PLANNING',estimate_s=cell['estimate_s'])
    source=cell['estimate_from']
    j=require_pass(source);r=j['result']
    if (j['cell']['kind']!='decode_clean' or r.get('fit') is not True or r.get('steps')!=32
        or r.get('bits')!=cell['bits'] or r.get('arm')!=cell['arm'] or r.get('S')!=8192
        or r.get('config')!=cell['config'] or r.get('feeding')!='teacher_forced'):
        raise Red('A6_INVALID_CLEAN8192_SOURCE: '+source)
    fields=['setup_s','prefill_s','warmup_s','decode_work_s']
    if any(not isinstance(r.get(k),(int,float)) or not math.isfinite(r[k]) or r[k]<0 for k in fields):
        raise Red('A6_INVALID_PLANNING_TIMES')
    # Prior art: A5 conservative quadratic prefill/linear cached-decode
    # extrapolation (2026). No prior art known to me for this exact formula.
    estimate=r['setup_s']+16*r['prefill_s']+4*(r['warmup_s']+r['decode_work_s'])+15
    return dict(fit=False if estimate>cell['worker_timeout_s'] else None,
                outcome='NON_FIT_PLANNED_RAIL' if estimate>cell['worker_timeout_s'] else 'WITHIN_PLANNING_RAIL',
                estimate_s=estimate,source=source,source_sha256=sha(job_path(source)),
                source_times={k:r[k] for k in fields},worker_timeout_s=cell['worker_timeout_s'],
                formula='setup + 16*prefill + 4*(warmup+decode_work) + 15',
                evidence_class='planning extrapolation; no measured 32K throughput')


def pin_plan(cell):
    plan=planning(cell);p=ART/'plans_a6'/(cell['id']+'.json')
    if p.exists():
        if read(p)!=plan:raise Red('A6_IMMUTABLE_PLAN_CHANGED')
    else:publish(p,plan)
    return plan


def nonfit_result(cell,plan):
    return dict(plan,arm=cell['arm'],bits=cell['bits'],S=cell['S'],steps=0,
                config=cell['config'],tokens_s=None,ms_token=None,
                pool_reserved_peak_mib=None,pool_used_peak_mib=None,
                evidence_class='registered/planned non-fit; no GPU execution')


def preflight(cell):
    from apa_sp3_common import REG_SHA,cell_fingerprint
    from apa_sp3_a4_provenance import bridge
    plan=pin_plan(cell)
    if plan['fit'] is not False:return None
    publish(job_path(cell['id']),dict(job=cell['id'],cell=cell,status='PASS',
            registration_sha256=REG_SHA,fingerprint=cell_fingerprint(cell),
            fingerprint_schema='apa_sp3_per_kind_v1',
            fingerprint_amendment_sha256=bridge()['effective_sha256'],
            protocol_sha256=sha(ART/'protocol_amendment.json'),
            dependencies={d:sha(job_path(d)) for d in cell['depends']},
            result=nonfit_result(cell,plan),wall_s=0,
            evidence_class='planning receipt before lease; no model execution'))
    return 'NON_FIT'


def execute(cell):
    for d in cell['depends']:require_pass(d)
    plan=pin_plan(cell)
    if plan['fit'] is False:return nonfit_result(cell,plan)
    _,ids=protocol()
    delta=require_pass(f"freeze_b{cell['bits']}")['result']['delta'] if cell['arm']=='C' else None
    t=time.perf_counter();owner=CleanModel(cell,delta);setup_s=time.perf_counter()-t
    try:
        result=measure(owner,ids)
        return dict(result,arm=cell['arm'],bits=cell['bits'],S=cell['S'],delta=delta,
                    config=cell['config'],alloc_pooling=True,setup_s=setup_s,
                    pool_enable_stage='before weight load' if cell['config']['pool_before_load'] else 'after weight load',
                    diagnostic_attention_hook=cell['config']['attention_wrapper'],
                    interposer=owner.interposer,fit=True,outcome='MEASURED',planning=plan,
                    evidence_class='in-model decode timing; no new PPL/quality validation')
    finally:owner.close()
