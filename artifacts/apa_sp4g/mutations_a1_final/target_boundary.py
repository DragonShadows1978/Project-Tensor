"""Gemma QAT experiment adapter. Prior art: June Gemma port/floor (2026),
SP3 A6 (2026) clean decode; BLASST/Yuan2025 running-max precision promotion,
ThriftAttention/Sharratt2026 weight-sensitive error, FA2/Dao2023 online softmax,
TurboQuant/Zandieh2025 existing reconstructed-kq quantizer. Only binding routing
and measurement wiring are new; no product kernels or adapter methods changed.
"""
import ctypes,math,os,sys,time,subprocess
from pathlib import Path
import numpy as np
from apa_sp4g_common import *

ENV={'GEMMA4_APA_INT4':'0','GEMMA4_APA_GEMM':'0','GEMMA4_APA_DECODE_FUSED':'1','GEMMA4_QUANT_V':'0','GEMMA4_QUANT_KV4':'0','TC_APA_SELECTIVE_PATH':'0','TC_ATTN_QTILE':'0'}

# Prior art: NVIDIA CUDA Runtime 12.6 (2024), cuda_runtime_api.h; SP3 A5
# (2026) pool counters. Resolve the complete ABI on import, before model load.
# Symbol lookup does not call CUDA or require a GPU. No new algorithm.
CUDART_PATH='/usr/local/cuda-12.6/lib64/libcudart.so.12'
CUDART_SIGNATURES={
    'cudaGetDevice': [ctypes.POINTER(ctypes.c_int)],
    'cudaDeviceGetDefaultMemPool': [ctypes.POINTER(ctypes.c_void_p),ctypes.c_int],
    'cudaMemPoolGetAttribute': [ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p],
    'cudaMemPoolSetAttribute': [ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p],
}
CUDART=ctypes.CDLL(CUDART_PATH)
for _name,_args in CUDART_SIGNATURES.items():
    _symbol=getattr(CUDART,_name)
    _symbol.argtypes=_args
    _symbol.restype=ctypes.c_int

def nll(logits,targets):
    # Prior art: next-token cross entropy (Shannon1948), June floor; fp64
    # max-shift logsumexp, standard numerical analysis, no novel scoring rule.
    x=np.asarray(logits,np.float64);y=np.asarray(targets,np.int64)
    if x.ndim!=2 or x.shape[0]!=len(y) or len(y)==0 or not np.isfinite(x).all():raise Red('INVALID_LOGITS')
    mx=x.max(axis=1);loss=mx+np.log(np.exp(x-mx[:,None]).sum(axis=1))-x[np.arange(len(y)),y]
    return float(loss.sum(dtype=np.float64))

def scoring_blocks(S,scored):
    pos=S-scored
    if not 0<scored<S:raise Red('INVALID_SCORING_WINDOW')
    while pos<S-1:
        n=min(64,S-1-pos);yield pos,n;pos+=n

class PoolPeak:
    # Prior art: NVIDIA CUDA12.6 pool attributes; copied scope of SP3 A5.
    def __init__(self):
        self.cuda=CUDART;self.pool=ctypes.c_void_p();d=ctypes.c_int()
        self.check(self.cuda.cudaGetDevice(ctypes.byref(d)));self.check(self.cuda.cudaDeviceGetDefaultMemPool(ctypes.byref(self.pool),d))
    @staticmethod
    def check(rc):
        if rc:raise Red('POOL_COUNTER_FAILED '+str(rc))
    def reset(self):
        for a in (6,8):
            x=ctypes.c_uint64(0);self.check(self.cuda.cudaMemPoolSetAttribute(self.pool,a,ctypes.byref(x)))
    def result(self):
        out={}
        for a,k in [(6,'pool_reserved_high_mib'),(8,'pool_used_high_mib')]:
            x=ctypes.c_uint64();self.check(self.cuda.cudaMemPoolGetAttribute(self.pool,a,ctypes.byref(x)));out[k]=x.value/(1<<20)
        return dict(out,pool_scope='default-pool high water since reset, not whole-device resident peak')

def resident():
    # Prior art: NVIDIA nvidia-smi own-PID observation, no interposition/thread.
    r=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
    if r.returncode:raise Red('RESIDENT_QUERY_FAILED: '+r.stderr.strip())
    rows=[x.split(',') for x in r.stdout.splitlines() if x.strip()]
    mine=[float(x[1]) for x in rows if int(x[0])==os.getpid()]
    if len(mine)!=1:raise Red('OWN_PID_RESIDENCY_UNAVAILABLE')
    return mine[0]

class Model:
    def __init__(self,cell,delta=None,capture=False,observe=False):
        self.cell=cell;self.delta=delta;self.capture=capture;self.observe=observe or capture
        if os.environ.get('LD_PRELOAD') or any(x in Path('/proc/self/maps').read_text() for x in ('libapa_sp3_peak','libapa_sp4g_peak')):raise Red('INTERPOSER_FORBIDDEN')
        for k in list(os.environ):
            if k.startswith(('TC_','GEMMA4_')):del os.environ[k]
        os.environ.update(ENV,TC_APA_SP='1' if cell['arm'] in 'CDE' else '0')
        verify_weight();self.tc=tc=load_runtime();tc.set_alloc_pooling(True)
        sys.path.insert(0,'/mnt/ForgeRealm/GraftRepository')
        from core import gemma4_tc as gemma, mistral7b_tc as base
        if Path(gemma.__file__).resolve()!=Path('/mnt/ForgeRealm/GraftRepository/core/gemma4_tc.py'):raise Red('WRONG_GEMMA_ADAPTER')
        self.gemma=gemma;self.original_attn=gemma.Gemma4AttentionTC.__call__
        self.original=tc.apa_selective_attention
        base.QuantLinearTC.FUSED_DECODE=True;base.RMSNormTC.USE_FUSED=True;base.F.USE_FUSED_SOFTMAX=True
        gemma.KVRing.QUANT_V=False;gemma.KVRing.QUANT_KV4=False
        if gemma.Gemma4AttentionTC.KV_STORE_HOOK is not None:raise Red('UNEXPECTED_KV_HOOK')
        t=time.perf_counter()
        with tc.no_grad():self.model,self.info=gemma.Gemma4_TC.from_pretrained(registration()['weight']['path'],qat=True)
        tc.synchronize();self.load_s=time.perf_counter()-t
        if self.info['loaded']!='QAT q4_0 exact (symmetric-8 g32)' or base.BlockTC.COMPUTE_DTYPE!='bfloat16':raise Red('QAT_DTYPE_PIN')
        layers=self.model.layers
        if len(layers)!=48 or [i for i,l in enumerate(layers) if l.mixer.is_global]!=list(range(5,48,6)):raise Red('GLOBAL_LAYER_PIN')
        for layer in layers:
            mx=layer.mixer
            mx.attention_mode='apa_selective' if mx.is_global and cell['arm']!='A' else 'standard'
            mx.apa_min_context=0;mx.fast_max_seq=0;mx.bulk_bits=4;mx.refine_percentile=.15
            if mx.is_global and (mx.head_dim!=512 or mx.kv_heads!=1):raise Red('MQA_DIM_PIN')
        self.calls=0;self.counts={i:dict(selected=0,pairs=0) for i in range(5,48,6)};self.records={i:[] for i in self.counts};self.files=[]
        self.directory=A/'captures'/cell['id']
        self.diag=None
        if self.observe:
            import _apa_sp4g_diag
            self.diag=_apa_sp4g_diag
            tc.apa_selective_attention=self.diagnostic_dispatch
        elif cell['arm'] in 'CDE':
            if delta is None or not math.isfinite(delta) or delta<0:raise Red('INVALID_DELTA')
            # Required algorithm dispatch only: no counters/attention wrappers,
            # no tensor host copies. SP1 binding/kernel is reused verbatim.
            def native_sp(q,k,kq,v,scale,zthr,is_causal=False):
                return tc._C.apa_selective_attention_sp(q,k,kq,v,scale,delta,is_causal,None,False)
            tc.apa_selective_attention=native_sp
        self.pool=PoolPeak();self.resident_load_mib=resident()

    def diagnostic_dispatch(self,q,k,kq,v,scale,zthr,is_causal=False):
        tc=self.tc;arm=self.cell['arm'];layer=5+6*(self.calls%8);self.calls+=1
        B,H,L,D=q.shape;S=k.shape[2]
        if (B,H,D)!=(1,16,512) or tuple(k.shape)!=(1,1,S,512) or k.shape!=kq.shape or k.shape!=v.shape or scale!=1.0 or L<=1:raise Red('CALLSITE_PARITY_PIN')
        if arm=='B':
            out=self.original(q,k,kq,v,scale,zthr,is_causal)
            check,mask,_=self.diag.selective(q,k,kq,v,scale,zthr,is_causal)
            if not np.array_equal(out.float().numpy(),check.float().numpy()):raise Red('B_DIAGNOSTIC_OUTPUT_NOT_BITWISE')
        else:
            check,mask=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,self.delta,is_causal,None,True)
            out=tc._C.apa_selective_attention_sp(q,k,kq,v,scale,self.delta,is_causal,None,False)
            if not np.array_equal(out.float().numpy(),check.float().numpy()):raise Red('SP_DIAGNOSTIC_OUTPUT_NOT_BITWISE')
        m=mask.numpy().astype(bool);lengths=S-L+np.arange(L)+1 if is_causal else np.full(L,S)
        eligible=np.arange(S)<lengths[:,None]
        if np.any(m & ~eligible[None,None]):raise Red('MASK_OUTSIDE_CAUSAL')
        pairs=int(H*lengths.sum());sel=int(m.sum())
        self.counts[layer]['pairs']+=pairs;self.counts[layer]['selected']+=sel
        if arm=='D' and sel!=pairs:raise Red('REFINE_ALL_DID_NOT_REFINE_ALL')
        if self.capture:
            idx=len(self.records[layer]);d=self.directory/f'l{layer:02d}';d.mkdir(parents=True,exist_ok=True)
            rec=dict(lo=S-L,n=L,S=S,causal=is_causal,scale=scale,zthr=zthr,files={})
            for name,t in [('q',q),('out',out)]:
                f=save_array(d/f'{idx:04d}_{name}.npy',t.float().numpy());self.files.append(f);rec['files'][name]=f
            self.records[layer].append(rec)
            if S==self.cell['S']:
                for name,t in [('k',k),('kq',kq),('v',v)]:
                    f=save_array(d/f'{name}.npy',t.float().numpy());self.files.append(f)
        return out

    def counts_result(self):
        rows=[dict(layer=l,**c) for l,c in self.counts.items()];pairs=sum(c['pairs'] for c in rows);selected=sum(c['selected'] for c in rows)
        return dict(pairs=pairs,selected=selected,fraction=selected/pairs if pairs else None,per_layer=rows)

    def prefill(self,ids):
        self.pool.reset();self.tc.synchronize();t=time.perf_counter()
        with self.tc.no_grad():lg,cache=self.model(ids[None],last_token_only=True)
        self.tc.synchronize()
        return lg,cache,time.perf_counter()-t

    def capture_run(self,ids):
        lg,cache,prefill_s=self.prefill(ids[:self.cell['S']])
        if not np.isfinite(lg.float().numpy()).all():raise Red('CAPTURE_LOGITS_NONFINITE')
        for layer,recs in self.records.items():
            at=0
            for r in recs:
                if r['lo']!=at or not r['causal']:raise Red('CAPTURE_COVERAGE_GAP')
                at+=r['n']
            if at!=self.cell['S']:raise Red('CAPTURE_LAYER_INCOMPLETE')
        manifest=dict(arm=self.cell['arm'],S=self.cell['S'],delta=self.delta,records=self.records,files=self.files,**self.counts_result())
        p=self.directory/'capture.json';publish(p,manifest)
        return dict(manifest= str(p.relative_to(R)),delta=self.delta,files=self.files+[dict(path=str(p.relative_to(R)),sha256=sha(p))],load_s=self.load_s,prefill_s=prefill_s,**self.counts_result())

    def perplexity(self,ids,scored):
        ctx=len(ids)-scored-1;total=0.;count=0
        with self.tc.no_grad():
            _,cache=self.model(ids[None,:ctx],last_token_only=True)
            for pos,n in scoring_blocks(len(ids),scored):
                lg,cache=self.model(ids[None,pos:pos+n],caches=cache,position_offset=pos)
                total+=nll(lg.float().numpy()[0],ids[pos+1:pos+n+1]);count+=n
                del lg;self.tc.empty_cache()
        self.tc.synchronize()
        if count!=scored:raise Red('SCORED_TARGET_COUNT')
        return dict(total_nll=total,targets=count,ppl=math.exp(total/count),load_s=self.load_s,global_fraction=self.counts_result())

    def decode(self,ids):
        # Prior art: June greedy decode + SP3 A6 synchronized wall timing.
        # Gemma KVRing mutates caches: no throwaway warmup that corrupts prefix.
        if self.observe or self.gemma.Gemma4AttentionTC.__call__ is not self.original_attn:raise Red('CLEAN_DECODE_HAS_WRAPPER')
        tc=self.tc;S=self.cell['S'];lg,cache,prefill_s=self.prefill(ids[:S])
        def argmax(x):
            a=tc.argmax_last_axis(x).numpy()
            if a.size!=1:raise Red('ARGMAX_NOT_SCALAR')
            token=int(a.reshape(-1)[0])
            if not 0<=token<262144:raise Red('ARGMAX_RANGE')
            return token
        token=argmax(lg);del lg;times=[];forward=[];chosen=[];start=time.perf_counter()
        with tc.no_grad():
            for pos in range(S,S+32):
                tc.synchronize();t=time.perf_counter()
                lg,cache=self.model(np.array([[token]],np.int64),caches=cache,position_offset=pos,last_token_only=True)
                tc.synchronize();forward.append(time.perf_counter()-t)
                token=argmax(lg);tc.synchronize();times.append(time.perf_counter()-t);chosen.append(token);del lg
        elapsed=time.perf_counter()-start
        for i in range(5,48,6):
            c=cache[i]
            if c.count!=S+32:raise Red('DECODE_CACHE_LENGTH')
            if self.cell['arm']!='A' and c.kq_count!=c.count:raise Red('DECODE_KQ_COUNT')
        return dict(fit=True,outcome='MEASURED',steps=32,ms_token=1000*sum(times)/32,seconds_per_step=times,forward_ms_token=1000*sum(forward)/32,argmax_tokens=chosen,prefill_s=prefill_s,decode_s=elapsed,setup_s=self.load_s,load_s=self.load_s,apa_min_context=0,config=ENV,pool_before_load=True,attention_class_wrapper=False,interposer=False,per_step_host_copy='one device argmax int64 only',kq_incremental=True,resident_after_mib=resident(),resident_load_mib=self.resident_load_mib,resident_scope='own-PID post-load/post-decode snapshots; not peak',**self.pool.result())
    def close(self):self.tc.apa_selective_attention=self.original
