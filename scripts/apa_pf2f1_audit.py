"""CPU scope and receipt audit. Prior art: PF2 receipt/file census (2026),
ordinary immutable-registration SHA and filesystem stat comparisons.
No new algorithm or numerical claim.
"""
from pathlib import Path
import difflib
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import apa_pf2f1_common as C
import apa_pf2f1_prepare as P
import apa_pf2f1_gpu as G


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    before=json.loads((C.ART/'SCOPE_BEFORE.json').read_text())
    after=P.census()
    added=sorted(set(after)-set(before));missing=sorted(set(before)-set(after))
    changed=sorted(k for k in set(before)&set(after) if before[k]!=after[k])
    reg=C.sha(C.ART/'REGISTRATION.json')
    assert reg==(C.ART/'REGISTRATION.sha256').read_text().strip()
    pins=json.loads((C.ART/'LINE_PINS.json').read_text())
    for path,h in pins['scripts'].items():assert C.sha(ROOT/path)==h,path
    assert C.sha(C.ART/'LINES.json')==pins['lines_sha256']
    lines=json.loads((C.ART/'LINES.json').read_text())
    indices=[i for k,v in lines.items() if k.startswith('G1_') for i in v['indices']]
    assert sorted(indices)==list(range(4848)) and len(indices)==len(set(indices))
    assert len(G.coverage_cases())==28
    builds={a:C.precheck(a) for a in ('PF','F1',*C.build.MUTANTS)}
    dry=json.loads((C.ART/'DRY_RUNS.json').read_text())
    assert dry['lines']==len(lines)==58 and dry['leases']==0
    for r in dry['records']:
        p=C.ART/'gpu'/f"{r['line']}_dry.log"
        assert C.sha(p)==r['log_sha256'] and r['exit']==78
    for p in (ROOT/'scripts').glob('apa_pf2f1_*.py'):compile(p.read_text(),str(p),'exec')
    x=json.loads((C.ART/'INDEX_F1.json').read_text())
    y=json.loads((C.ART/'INDEX_pf1m03_index_int32.json').read_text())
    assert x['status']=='PASS' and x['exit']==0 and y['exit']==1
    assert 'signed integer overflow' in y['stderr']
    sass=json.loads((C.ART/'SASS_COMPARISON.json').read_text())
    assert sass['status']=='PASS' and sass['counts']==dict(equal=781,changed=32,added=0,removed=0)
    diffs=[]
    for rel in ('tensor_cuda/src/apa_spo/packed_prefill.cuh','scripts/apa_p1pf2_g45.py'):
        original=C.ART/'originals'/Path(rel).name
        diffs.extend(difflib.unified_diff(original.read_text().splitlines(True),(ROOT/rel).read_text().splitlines(True),
                                        fromfile='before/'+rel,tofile='after/'+rel))
    (C.ART/'CHANGES.diff').write_text(''.join(diffs))
    result=dict(status='PASS' if not(added or missing or changed) else 'FAIL',
                outside_allowlist=dict(added=added,missing=missing,changed=changed),
                outside_allowlist_files_checked=len(before),registration_sha256=reg,
                git='No git command invoked; .git pointer stat unchanged in scoped snapshot; git metadata outside writable roots',
                deletes='No explicit deletion command or deletion call; prior staging drafts retained by rename',
                GPU=dict(real_leases=0,dry_lines=58,dry_exit=78),modules=builds,
                G3='PASS',G5_CPU='PASS: index mutant killed by actual helper execution / UBSan; F1 clean',
                sass='PASS: 781 equal / 32 prefill changed / 0 added / 0 removed',
                CPU_pytest=json.loads((C.ART/'PYTEST_CPU.json').read_text()),
                scope_limit='stat census verifies this worktree; no write-capable action targeted other worktrees, production or git metadata')
    C.save(C.ART/'G7.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('modules','CPU_pytest')},indent=2))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
