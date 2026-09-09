"""A3 bounded same-input argument sweep.
Prior art: ordinary factorial/single-factor ablation; SP3 (2026) replay;
June Gemma (2026) bottom-right causal reference. Rowwise prefix replay is
diagnostic wiring, not a kernel optimization or a proposed product fix.
"""
import os
import numpy as np
from apa_sp4g_a3_common import *
from apa_sp4g_a3_math import variants, dense, compare, classify, eligible
from apa_sp4g_a3_model import exact
from apa_sp4g_a2_model import standard, array, MAX_DELTA

def sp_variant(tc,q,k,kq,v,variant,position_offset):
    L,S = q.shape[2],k.shape[2]
    mask = eligible(L,S,variant['causal'],variant['offset'],position_offset)
    sink = None if variant['sink'] == 'none' else tc.tensor(np.zeros(q.shape[1],np.float32),dtype='float32')
    entry = tc._C.apa_selective_attention_sp
    if variant['offset'] == 'native_bottom_right' or not variant['causal']:
        return entry(q,k,kq,v,variant['scale'],MAX_DELTA,variant['causal'],sink,False)
    # No explicit offset exists in the SP ABI. Expose exactly the requested
    # per-row prefix with L=1/noncausal. This exercises existing split-K and
    # MUST be labelled a call-geometry intervention, not merely an argument.
    outs = []
    for i,n in enumerate(mask.sum(axis=-1)):
        n = int(n)
        outs.append(entry(q.slice(2,i,1),k.slice(2,0,n),kq.slice(2,0,n),v.slice(2,0,n),
                          variant['scale'],MAX_DELTA,False,sink,False))
    return tc.cat(outs,dim=2)

def run_sweep(c):
    os.environ['TC_APA_SP'] = '1'
    tc = load_runtime()
    tc.set_alloc_pooling(True)
    directory = A / 'captures_a3' / c['id']
    files, calls = [], []
    with tc.no_grad():
        for name in c['depends']:
            receipt = require_a3(name)
            manifest = read(R / receipt['result']['manifest'])
            meta = manifest['metadata']
            raw = {n:np.load(R/manifest['arrays'][n]['path'],allow_pickle=False)
                   for n in ('q','k','kq','v','standard_fp32','dense_fp32')}
            q,k,kq,v = [tc.tensor(raw[n],dtype='float32') for n in ('q','k','kq','v')]
            a = array(standard(tc,q,k,v,meta['scale']))
            exact(a,raw['standard_fp32'],'sweep_standard_replay_'+name)
            rows, cache = [], {}
            for index,variant in enumerate(variants()):
                # Only causal-off offset labels are literal ABI no-ops.
                key = (variant['scale'],variant['sink'],variant['causal'],
                       variant['offset'] if variant['causal'] else 'offset_ignored_noncausal')
                reused = key in cache
                if not reused:
                    output = array(sp_variant(tc,q,k,kq,v,variant,meta['position_offset']))
                    reference = dense(raw['q'],raw['k'],raw['v'],**variant,position_offset=meta['position_offset'])
                    f = save_array(directory/name/(f'variant_{index:02d}.npy'),output)
                    files.append(f)
                    cache[key] = (output,reference,f)
                output,reference,f = cache[key]
                rows.append(dict(variant=variant,output_file=f,
                                 effective_call='native' if variant['offset']=='native_bottom_right' or not variant['causal'] else 'L1_prefix_splitK',
                                 reused_identical_noncausal_call=reused,
                                 vs_standard=compare(output,a),
                                 standard_vs_nominal_dense=compare(a,raw['dense_fp32']),
                                 vs_own_dense=compare(output,reference),
                                 vs_nominal_dense=compare(output,raw['dense_fp32'])))
            calls.append(dict(source_cell=name,source_receipt_sha256=sha(path_a3(name)),
                              L=meta['L'],S_all=meta['S_all'],position_offset=meta['position_offset'],
                              standard_vs_dense=compare(a,raw['dense_fp32']),rows=rows))
            del q,k,kq,v,cache,raw
            tc.empty_cache()
    tc.synchronize()
    decision = classify([r['rows'] for r in calls])
    result = dict(calls=calls,decision=decision,files=files,dtype='float32',
                  combinations_per_call=24,model_load=False,
                  bf16_controls='original-argument bf16 measured in each dependency call receipt',
                  evidence_class='same-input real-model tensors kernel diagnostic; no PPL / quality gate',C_unblocked=False)
    return result
