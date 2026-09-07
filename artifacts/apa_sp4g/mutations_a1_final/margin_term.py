"""Full-population real-key margins. Prior art: SP3(2026) captured native
masks/replay, BLASST/Yuan2025 max-relative weights, ThriftAttention/Sharratt2026
weight-sensitive error, SP2 delta bound. Nearest-rank statistics and memory maps
are standard; no new algorithm or universal quantization certificate claimed.
"""
import math,os
from pathlib import Path
import numpy as np
from apa_sp4g_common import *

def upward(x):return float(np.nextafter(np.float32(x),np.float32(np.inf)))
def tail(exact,mask):
    x=np.asarray(exact,np.float64);m=np.asarray(mask,bool)
    if x.size==0 or not np.isfinite(x).all() or x.shape!=m.shape:raise Red('INVALID_TAIL')
    w=np.exp(x-x.max());return float(w[~m].sum()/w.sum()),float(w[~m].max()) if (~m).any() else 0.
def stats(errors):
    x=np.asarray(errors)
    if x.size==0 or not np.isfinite(x).all():raise Red('INVALID_ERROR_POPULATION')
    indexes=[math.ceil(p*x.size)-1 for p in (.99,.999)]
    mean=float(x.sum(dtype=np.float64)/x.size);maximum=float(x.max());x.partition(indexes)
    return dict(mean=mean,p99=float(x[indexes[0]]),p99_9=float(x[indexes[1]]),max=maximum)
def capture_for(cell):
    source=require_pass(cell['depends'][0])['result']
    if cell['S']==2048 and cell['arm']=='C':source=require_pass(source['trial'])['result']
    p=R/source['manifest'];m=read(p)
    if m['arm']!=cell['arm'] or m['S']!=cell['S']:raise Red('CAPTURE_CELL_MISMATCH')
    return p.parent,m

def replay_context(cell):
    os.environ['TC_APA_SP']='1';directory,m=capture_for(cell);tc=load_runtime()
    import _apa_sp4g_diag as diag
    return directory,m,tc,diag

def band(cell,context=None):
    directory,m,tc,diag=replay_context(cell) if context is None else context
    layer=cell['layer'];lo=cell['lo'];hi=lo+cell['n'];records=m['records'][str(layer)]
    # Restore exact BF16-representable floats; intersect original adaptive query
    # chunks, preserve absolute bottom-right alignment by truncating K at stop.
    pieces=[];outs=[];zs=[]
    for rec in records:
        a=max(lo,rec['lo']);b=min(hi,rec['lo']+rec['n'])
        if a<b:
            sl=slice(a-rec['lo'],b-rec['lo'])
            pieces.append(np.load(R/rec['files']['q']['path'],mmap_mode='r')[:,:,sl,:])
            outs.append(np.load(R/rec['files']['out']['path'],mmap_mode='r')[:,:,sl,:]);zs.append(rec['zthr'])
    q=np.concatenate(pieces,axis=2);expected=np.concatenate(outs,axis=2)
    if q.shape!=(1,16,cell['n'],512) or len(set(zs))!=1:raise Red('BAND_COVERAGE')
    raw={n:np.load(directory/f'l{layer:02d}/{n}.npy',mmap_mode='r')[:,:,:hi,:] for n in ('k','kq','v')}
    qt=tc.tensor(q,dtype='bfloat16');kt,kqt,vt=[tc.tensor(raw[n],dtype='bfloat16') for n in ('k','kq','v')]
    with tc.no_grad():
        if cell['arm']=='B':out,mask,bulk=diag.selective(qt,kt,kqt,vt,1.,zs[0],True)
        else:
            out,mask=tc._C.apa_selective_attention_sp(qt,kt,kqt,vt,1.,m['delta'],True,None,True)
            bulk=diag.bulk_scores(qt,kqt,1.)
        sp_bulk=diag.bulk_scores(qt,kqt,1.)
    tc.synchronize()
    if not np.array_equal(out.float().numpy(),expected):raise Red('NATIVE_REPLAY_NOT_BITWISE')
    native=bulk.numpy()[0];sp=sp_bulk.numpy()[0];selected=mask.numpy()[0].astype(bool)
    exact=q[0].astype(np.float64)@raw['k'][0,0].astype(np.float64).T
    lengths=lo+np.arange(cell['n'])+1;eligible=np.arange(hi)<lengths[:,None]
    if np.any(selected & ~eligible):raise Red('MASK_CAUSAL')
    err=np.abs(native.astype(np.float64)-exact)[:,eligible]
    sperr=np.abs(sp.astype(np.float64)-exact)[:,eligible]
    masses=[];relative=[]
    for h in range(16):
        for i,n in enumerate(lengths):
            mass,rel=tail(exact[h,i,:n],selected[h,i,:n]);masses.append(mass);relative.append(rel)
    path=A/'margin_errors'/f'{cell["id"]}.npy';f=save_array(path,err.ravel())
    pairs=err.size;count=int(selected.sum())
    return dict(files=[f],error_file=f['path'],pairs=pairs,selected=count,fraction=count/pairs,queries=len(masses),mass_sum=float(sum(masses)),mass_max=max(masses),max_skipped_relative_weight=max(relative),eq_sp=float(sperr.max()),replay_bitwise=True)

def aggregate_rows(cell,rows):
    pairs=sum(r['pairs'] for r in rows);expected=16*cell['S']*(cell['S']+1)//2
    if pairs!=expected or sum(r['queries'] for r in rows)!=16*cell['S']:raise Red('MARGIN_ALL_PAIR_COVERAGE')
    if not all(r['replay_bitwise'] for r in rows):raise Red('NATIVE_REPLAY_NOT_BITWISE')
    # Scratch is unique and owned by this worker; durable errors remain hashed.
    scratch=A/'scratch'/f'{cell["id"]}.{os.getpid()}.errors';scratch.parent.mkdir(exist_ok=True)
    if scratch.exists():raise Red('SCRATCH_EXISTS')
    x=np.memmap(scratch,dtype='float64',mode='w+',shape=(pairs,));at=0
    try:
        for row in rows:
            y=np.load(R/row['error_file'],mmap_mode='r');x[at:at+len(y)]=y;at+=len(y)
        er=stats(x);count=sum(r['selected'] for r in rows)
        return dict(error=er,eq_sp=max(r['eq_sp'] for r in rows),pairs=pairs,selected=count,fraction=count/pairs,unrefined_mass_mean=sum(r['mass_sum'] for r in rows)/(16*cell['S']),unrefined_mass_max=max(r['mass_max'] for r in rows),max_skipped_relative_weight=max(r['max_skipped_relative_weight'] for r in rows),coverage='all causal pairs; nearest-rank exact population',replay_bitwise=all(r['replay_bitwise'] for r in rows))
    finally:del x;scratch.unlink()

def whole_layer(cell):
    # Prior art: existing SP4G/SP3 (2026) exact capture replay, tiled dots,
    # memory maps and population order statistics. A1 coalesces dispatch only.
    # All 16 query heads remain in the population despite ONE shared KV head.
    context=replay_context(cell);rows=[]
    for lo in range(0,cell['S'],128):
        tile=dict(cell,id=f'{cell["id"]}_tile_q{lo:05d}',lo=lo,n=min(128,cell['S']-lo))
        rows.append(band(tile,context))
    result=aggregate_rows(cell,rows)
    return dict(result,files=[f for r in rows for f in r['files']],
                replay_tiles=len(rows),execution='one whole-layer worker')

def summarize(cell):
    rows=[require_pass(d)['result'] for d in cell['depends']]
    return aggregate_rows(cell,rows)

def eq_result(cell):
    rows=[require_pass(d)['result'] for d in cell['depends']];eq=upward(max(r['eq_sp'] for r in rows))
    return dict(eq=eq,epsilon=.01,delta=upward(math.log(100)+0*eq),calibration='all B/C global layers at2048/8192; finite empirical real-key maximum; conditional bound only, not universal',source_count=len(rows))
