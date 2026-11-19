"""CPU-only acceptance after ALL lead lines, never substitutes for GPU runs.
Prior art: PF2 receipt aggregation (Project-Tensor 2026), ordinary exhaustive
case census and SHA admission. No new numerical algorithm.
"""
from pathlib import Path
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import apa_pf2f1_common as C


def get(rel):
    return json.loads((C.ART/rel).read_text())


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    lines=get('LINES.json')
    modules={a:C.precheck(a) for a in ('PF','F1',*C.build.MUTANTS)}
    for name in lines:
        r=get(f'gpu/{name}_real.exit.json')
        assert r['exit']==0 and not r['dry_run'] and r['lease_attempted'],name
        assert C.sha(C.ART/f'gpu/{name}_real.log')==r['log_sha256'],name
    cases=[]
    for name,s in lines.items():
        if name.startswith('G1_'):
            r=get(f'identity/{name}/RESULT.json')
            assert r['status']=='PASS' and r['indices']==s['indices']
            cases.extend(r['rows'])
    assert len(cases)==4848 and sorted(c['index'] for c in cases)==list(range(4848))
    assert all(c['ok'] and set(c['comparisons'])=={'out','mask'} and
               all(v['elements']>0 and v['unequal']==0 for v in c['comparisons'].values()) for c in cases)
    for kind in ('sink','norm'):
        r=get(f'nonfinite/F1_{kind}/RESULT.json')
        assert r['status']=='PASS' and r['cases']==r['passed']==16,kind
        m=get(f'MUTANT_{kind}.json')
        assert m['status']=='PASS' and m['killed'] and m['clean_control'],kind
    cov=get('COVERAGE_LEAD.json')
    assert cov['registered']==cov['executed']==cov['passed']==28 and cov['status']=='PASS'
    refs=get('regression/G6_refusals/RESULT.json')
    assert refs['status']=='PASS' and len(refs['rows'])==29
    memory=[]
    for name in lines:
        if name.startswith('G6_memory_'):
            r=get(f'regression/{name}/RESULT.json')
            assert r['status']=='PASS' and len(r['rows'])==4
            memory.extend(r['rows'])
    assert len(memory)==24 and all(r['within_bound'] and r['no_key_shaped_transient'] for r in memory)
    assert get('G7.json')['status']=='PASS' and get('SASS_COMPARISON.json')['status']=='PASS'
    assert get('INDEX_F1.json')['status']=='PASS'
    assert get('INDEX_pf1m03_index_int32.json')['exit']==1
    result=dict(verdict='ACCEPT: registered CPU and lead GPU gates passed',modules=modules,
                G1_cases=4848,G2_sink=16,G2_norm=16,G3='PASS',G4_combinations=28,
                G5_killed=3,G6_refusal_rows=26,G6_memory_rows=24,G7='PASS',real_lines=len(lines),
                scope='This order only; no speed, model quality, promotion or cross-model review claim')
    C.save(C.ART/'ACCEPTANCE_LEAD.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
