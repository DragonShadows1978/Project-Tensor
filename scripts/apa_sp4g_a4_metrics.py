# A4 copy of the A2 bitwise native replay and population metrics (2026).
# Only receipt lookup/capture id/output namespace change. Prior art is below;
# no replay geometry, selection, statistic or tolerance is changed.
"""Exact original-call replay. Prior art: SP3 (2026) native masks/replay,
June Gemma quantizer (2026), BLASST max-relative weights (Yuan2025),
ThriftAttention (Sharratt2026), population statistics. No tolerance/new selector.
"""
import os
import numpy as np
from apa_sp4g_a4_common import *
from apa_sp4g_a2_model import array,compare,decode_bf16
from apa_sp4g_metrics import tail,aggregate_rows

def replay_probe(c):
    os.environ['TC_APA_SP']='1';tc=load_runtime()
    import _apa_sp4g_diag as diag
    from tensor_cuda.quant import _quantize_keys,_tables
    src=require_pass('capture_B_8192')['result'];m=read(R/src['manifest']);directory=(R/src['manifest']).parent
    rec=m['records'][str(c['layer'])][0];S,L=rec['S'],rec['n']
    raw={n:np.load(directory/f'l{c["layer"]:02d}/{n}.npy',mmap_mode='r')[:,:,:S,:] for n in ('k','kq','v')}
    qraw=np.load(R/rec['files']['q']['path']);expected=np.load(R/rec['files']['out']['path'])
    q=tc.tensor(qraw,dtype='bfloat16');k,kq,v=[tc.tensor(raw[n],dtype='bfloat16') for n in ('k','kq','v')]
    with tc.no_grad():
        R_t,C_t,B_t=_tables(512,4,1,True,q.device.split(':')[0])
        regenerated=_quantize_keys(k,R_t,C_t,B_t)
        def run(qt,kt,kqt,vt):return diag.selective(qt,kt,kqt,vt,rec['scale'],rec['zthr'],rec['causal'])
        final1,m1,_=run(q,k,kq,v);final2,m2,_=run(q,k,kq,v)
        regen1,r1,_=run(q,k,regenerated,v);regen2,r2,_=run(q,k,regenerated,v)
        # First old128 band; exact original tensor prefixes but changed geometry.
        n=128;qt=tc.tensor(qraw[:,:,:n,:],dtype='bfloat16')
        bt=[tc.tensor(array(t)[:,:,:n,:],dtype='bfloat16') for t in (k,regenerated,v)]
        band,mb,_=run(qt,*bt)
    same_final=compare(array(final1),array(final2));same_regen=compare(array(regen1),array(regen2))
    fit_final=compare(array(final1),expected);fit_regen=compare(array(regen1),expected)
    geometry=compare(array(band),array(regen1)[:,:,:128,:])
    stable=same_final['bitwise'] and same_regen['bitwise'] and np.array_equal(m1.numpy(),m2.numpy()) and np.array_equal(r1.numpy(),r2.numpy())
    if not stable:finding='NATIVE_REPEAT_VARIATION_OBSERVED; tolerance requires separate registration, NOT accepted'
    elif fit_regen['bitwise'] and not fit_final['bitwise']:finding='FINAL_PREFIX_KQ_SUBSTITUTION_CONTEXT_BUG_CONFIRMED_FOR_THIS_CALL'
    elif fit_regen['bitwise'] and not geometry['bitwise']:finding='BAND_GEOMETRY_CONTEXT_BUG_CONFIRMED_FOR_THIS_CALL'
    elif fit_final['bitwise'] and geometry['bitwise']:finding='FIRST_CALL_REPRODUCES; original failure not reproduced here'
    else:finding='UNRESOLVED_CAPTURE_CONTEXT; no nondeterminism or tolerance conclusion'
    return dict(finding=finding,layer=c['layer'],L=L,S=S,original_record=rec,
        final_Kq_vs_regenerated=compare(raw['kq'],array(regenerated)),
        final_repeat=same_final,regenerated_repeat=same_regen,
        original_shape_final_Kq_vs_capture=fit_final,original_shape_regenerated_Kq_vs_capture=fit_regen,
        band_vs_original_shape=geometry,mask_repeat_bitwise=bool(np.array_equal(m1.numpy(),m2.numpy()) and np.array_equal(r1.numpy(),r2.numpy())),
        limitations='Old capture lacks original per-call Kq and masks; regeneration is diagnostic evidence, not certified original selection. New G2 requires actual PPL capture.')

def load_record(rec):
    raw={n:decode_bf16(np.load(R/rec['files'][n]['path'],mmap_mode='r')) for n in ('q','k','kq','v','out')}
    shape=tuple(rec['mask_shape']);bits=np.load(R/rec['files']['mask']['path'],mmap_mode='r')
    raw['mask']=np.unpackbits(bits,bitorder='little',count=int(np.prod(shape))).reshape(shape).astype(bool)
    if raw['q'].shape!=(1,16,rec['n'],512) or raw['k'].shape!=(1,1,rec['S'],512) or rec['S']!=rec['lo']+rec['n']:raise Red('A2_RECORD_GEOMETRY')
    return raw

def require_exact_replay(out,mask,raw):
    if not np.array_equal(out,raw['out']):raise Red('NATIVE_REPLAY_NOT_BITWISE')
    if not np.array_equal(mask,raw['mask']):raise Red('NATIVE_SELECTION_NOT_BITWISE')

def metric_tiles(c,rec,raw,native,sp,selected):
    # SP3/June (2026): tile AFTER native replay; tiling never recomputes a
    # selection. FP64 exact dots and all-key mass; all 16 query heads included.
    rows=[];lo=rec['lo'];S=rec['S']
    for offset in range(0,rec['n'],128):
        n=min(128,rec['n']-offset);sl=slice(offset,offset+n)
        q=raw['q'][0,:,sl,:].astype(np.float64);exact=q@raw['k'][0,0].astype(np.float64).T
        lengths=lo+offset+np.arange(n)+1;eligible=np.arange(S)<lengths[:,None]
        sel=selected[0,:,sl,:]
        if np.any(sel & ~eligible):raise Red('MASK_CAUSAL')
        err=np.abs(native[0,:,sl,:].astype(np.float64)-exact)[:,eligible]
        sperr=np.abs(sp[0,:,sl,:].astype(np.float64)-exact)[:,eligible]
        masses=[];relative=[]
        for h in range(16):
            for i,count in enumerate(lengths):
                mass,rel=tail(exact[h,i,:count],sel[h,i,:count]);masses.append(mass);relative.append(rel)
        f=save_array(A/'margin_errors_a4'/f'{c["id"]}_q{lo+offset:05d}.npy',err.ravel())
        rows.append(dict(files=[f],error_file=f['path'],pairs=err.size,selected=int(sel.sum()),queries=16*n,
            mass_sum=float(sum(masses)),mass_max=max(masses),max_skipped_relative_weight=max(relative),eq_sp=float(sperr.max()),replay_bitwise=True))
    return rows

def whole_layer(c):
    os.environ['TC_APA_SP']='1';tc=load_runtime()
    import _apa_sp4g_diag as diag
    src=require_a4(c['capture_cell'])['result'];m=read(R/src['manifest'])
    if m['ppl_cell']!=c['capture_cell'] or m['arm']!=c['arm'] or m['population_rows']!=c['population_rows']:raise Red('A2_PPL_CAPTURE_SOURCE')
    rows=[];recs=m['records'][str(c['layer'])];at=0
    for rec in recs:
        if rec['lo']!=at:raise Red('A2_REPLAY_COVERAGE')
        at+=rec['n'];raw=load_record(rec)
        q,k,kq,v=[tc.tensor(raw[n],dtype='bfloat16') for n in ('q','k','kq','v')]
        with tc.no_grad():
            if c['arm']=='B':out,mask,bulk=diag.selective(q,k,kq,v,rec['scale'],rec['zthr'],rec['causal'])
            else:
                out,mask=tc._C.apa_selective_attention_sp(q,k,kq,v,rec['scale'],m['delta'],rec['causal'],None,True)
                bulk=diag.bulk_scores(q,kq,rec['scale'])
            sp=diag.bulk_scores(q,kq,rec['scale'])
        selected=mask.numpy().astype(bool)
        require_exact_replay(array(out),selected,raw)
        rows.extend(metric_tiles(c,rec,raw,bulk.numpy(),sp.numpy(),selected))
        del q,k,kq,v,out,mask,bulk,sp,raw;tc.empty_cache()
    if at!=c['population_rows']:raise Red('A2_REPLAY_POPULATION')
    result=aggregate_rows(dict(c,S=c['population_rows']),rows)
    source=next(r for r in m['per_layer'] if r['layer']==c['layer'])
    if (source['selected'],source['pairs'])!=(result['selected'],result['pairs']):raise Red('A2_PPL_SELECTION_COUNTS_CHANGED')
    return dict(result,files=[f for row in rows for f in row['files']],original_calls=len(recs),
        selection_source=c['capture_cell'],selection_bitwise=True,population_rows=at,
        coverage='All executed PPL causal pairs, query positions0..S-2, all16 heads; full-population nearest-rank percentiles')
