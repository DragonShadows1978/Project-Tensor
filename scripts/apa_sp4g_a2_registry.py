"""A2 isolated diagnostic schedule. Prior art: SP3 (2026) bounded DAG;
Make/Feldman1979 dependency invalidation. No new scheduling algorithm.
"""
from apa_sp4g_registry import LAYERS

def cells():
    out=[]
    def add(name,kind,depends=(),**kw):
        estimates={'replay_probe':[5,90], 'margin_a2': [10,90] if kw.get('S')==2048 else [60,270]}
        out.append(dict(id=name,kind=kind,depends=list(depends),arm=kw.pop('arm','A'),S=kw.pop('S',2048),bits=4,apa_min_context=0,worker_s=285,estimate_s=estimates.get(kind,[60,285]),**kw))
    previous='kernel512'
    for focus in ('source','scale','value','rope'):
        name=f'diag_a2_{focus}_2048'
        add(name,'parity',[previous],focus=focus,window=0);previous=name
    add('diag_a2_fp32_A_2048_w0','precision',[previous],treatment='A32',window=0)
    add('diag_a2_fp32_2048_w0','precision',['diag_a2_fp32_A_2048_w0'],treatment='D32',window=0)
    add('diag_a2_replay_B_8192_l05','replay_probe',['capture_B_8192'],arm='B',S=8192,layer=5)
    for arm in 'BC':
        for S in (2048,8192):
            cap=f'ppl_capture_{arm}_{S}_w0'
            deps=['kernel512']+(['freeze'] if arm=='C' else [])
            add(cap,'ppl_capture',deps,arm=arm,S=S,window=0,population_rows=S-1)
            for layer in LAYERS:
                add(f'margin_a2_{arm}_{S}_l{layer:02d}','margin_a2',[cap],arm=arm,S=S,layer=layer,population_rows=S-1)
    return out

def by_id():return {c['id']:c for c in cells()}

def rail_blocked(c):
    # A2 lead directive (2026): no model retry at >=16K without long lease.
    return c.get('S',0)>=16384
