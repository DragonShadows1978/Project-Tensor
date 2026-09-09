"""Registered cell DAG. Prior art: SP3 a4 split work/immutable fingerprints,
Kahn1962 topological dependencies (unverified lead: Topological sorting of large
networks). Bisection is classical; no new calibration algorithm claimed.
"""
from apa_sp4g_common import registration
from functools import lru_cache
LAYERS=list(range(5,48,6))
@lru_cache(maxsize=1)
def cells():
    out=[]
    def add(name,kind,depends=(),**kw):
        est={'kernel':[2,45],'margin':[5,120],'margin_summary':[1,120],'freeze':[1,5],'eq':[1,5],'ppl_summary':[1,5],'exactness':[1,5]}.get(kind,[60,285])
        c=dict(id=name,kind=kind,depends=list(depends),bits=4,apa_min_context=0,worker_s=285,estimate_s=est,**kw);out.append(c);return name
    k=add('kernel512','kernel')
    b=add('capture_B_2048','capture',[k],arm='B',S=2048)
    trials=[]
    for i in range(12):trials.append(add(f'trial_{i:02d}','trial',[b]+trials.copy(),arm='C',S=2048,index=i))
    freeze=add('freeze','freeze',[b]+trials)
    caps={(2048,'B'):b,(2048,'C'):freeze}
    for arm in 'BC':caps[8192,arm]=add(f'capture_{arm}_8192','capture',[k,freeze] if arm=='C' else [k],arm=arm,S=8192)
    sums=[]
    for S in (2048,8192):
        for arm in 'BC':
            for layer in LAYERS:
                bands=[]
                for lo in range(0,S,128):bands.append(add(f'margin_{arm}_{S}_l{layer:02d}_q{lo:05d}','margin',[caps[S,arm]],arm=arm,S=S,layer=layer,lo=lo,n=128))
                sums.append(add(f'margin_{arm}_{S}_l{layer:02d}','margin_summary',bands,arm=arm,S=S,layer=layer))
    eq=add('eq','eq',sums)
    for arm in 'ABDCE':
        deps=[k]+([freeze] if arm=='C' else [eq] if arm=='E' else [])
        windows=[]
        for w in range(4):windows.append(add(f'ppl_{arm}_2048_w{w}','ppl',deps,arm=arm,S=2048,window=w))
        add(f'ppl_{arm}_2048','ppl_summary',windows,arm=arm,S=2048)
        for S in (8192,16384,32768):add(f'ppl_{arm}_{S}','ppl',deps,arm=arm,S=S,window=0)
        for S in (4096,8192,16384,24576,32768):add(f'ceiling_{arm}_{S}','ceiling',deps,arm=arm,S=S)
        if arm in 'ABC':
            for S in (2048,8192,32768):add(f'decode_{arm}_{S}','decode',deps+([f'decode_{arm}_8192'] if S==32768 else []),arm=arm,S=S)
    exact=add('exactness_2048','exactness',['ppl_A_2048','ppl_D_2048'],S=2048)
    for S in (8192,16384,32768):add(f'exactness_{S}','exactness',[f'ppl_A_{S}',f'ppl_D_{S}'],S=S)
    for c in out:
        if c['id']=='capture_B_2048':c['depends'].append(exact)
    # Independence permits earlier quality cells; stable topological priority
    # keeps first model comparisons ahead of the large margin sweep.
    priority={'kernel':0,'ppl':1,'ppl_summary':1,'exactness':1,'capture':2,'trial':3,'freeze':4,'ceiling':5,'decode':6,'margin':7,'margin_summary':8,'eq':9}
    ordered=[];seen=set()
    while len(ordered)<len(out):
        ready=[c for c in out if c['id'] not in seen and set(c['depends'])<=seen]
        if not ready:raise RuntimeError('cycle/unknown dependency')
        def rank(c):
            if c.get('S')==32768:return 90
            if c['kind']=='ppl' and c['S']>2048:return 11
            if c['kind']=='capture' and c['S']==8192:return 12
            if c['kind']=='ceiling' and c['S']>8192:return 13
            return priority[c['kind']]
        c=min(ready,key=rank);ordered.append(c);seen.add(c['id'])
    return ordered
@lru_cache(maxsize=1)
def by_id():return {c['id']:c for c in cells()}
