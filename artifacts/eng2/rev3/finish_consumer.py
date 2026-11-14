"""Finish independent inventory using the retained receiver candidates.

Prior art: same Python ast/data-flow tradition as consumer_scan.py (Python
project; Kildall 1973 unverified -- lead to check); SHA-256 (NIST FIPS 180-4,
2015, unverified -- lead to check). Standard syntax and hashing reused;
review-specific source binding, conservative classification and census diff.
"""
import ast
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import consumer_scan as scan

OUT=Path('artifacts/eng2/rev3')
SCRATCH=Path(os.environ['TMPDIR'])/'consumer_sources'


def main():
    data=json.loads((OUT/'Q3_consumer_scan.json').read_text())
    a3=json.loads(Path('artifacts/eng2/a3/CONSUMER_CENSUS.json').read_text())
    rows=[r for r in data['sites'] if r['argument']!='<default float32>']
    candidates=[r for r in data['candidate_calls_unbound'] if r['function'].startswith('tc.') or '.tc.' in r['function']]
    binding={
        'Project-Voice': 'KokoroTC imports tensor_cuda at model.py:375; self.tc=tc at :378; test_gpu.py:35 and test_gpu_residency.py:41 copy model.tc; unit_dose.py:76 receives the model from model.py:1007.',
        'GRAPA-Native-LLM': 'native.tc is returned by scripts/cc10_ablation.py:1992-2004 after import tensor_cuda as tc; grapa/tripwire.py:630 saves constructor tc, passed the real engine by grapa/train.py:758,1195. Other local tc arguments are retained conservatively; their explicit supported strings do not depend on receiver provenance.',
        'GraftRepository': 'scripts/gpt_oss20b_expert_e1.py:585 delegates runtime loading to gpt_oss20b_router_telemetry.py:327-345, which imports tensor_cuda and exports tc. gpt.tc in RS2/RS4 reaches core.gpt_oss20b_tc, importing tc from core.mistral7b_tc.',
    }
    for row in candidates:
        tree=ast.parse((SCRATCH/row['repo']/row['file']).read_bytes())
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and n.lineno==row['line'] and n.col_offset==row['col'])
        dtype=next((k.value for k in node.keywords if k.arg=='dtype'),None)
        aliases,_=scan.imports_and_aliases(tree,row['file'][:-3].replace('/','.'))
        ctx=scan.gather_context(tree)
        values=scan.classify(dtype,ctx[3][id(node)],ctx,aliases)
        row.update(binding_evidence=binding[row['repo']], values=sorted(values),
                   classification='potentially-newly-refused' if any(v.startswith(('unsupported:','non-string:')) for v in values)
                   else 'unresolved' if any(v.startswith('unresolved:') for v in values) else 'supported',
                   resolved='tensor_cuda.'+row['entry'],receiver_resolved_by='source review')
        rows.append(row)
    a3keys={(r['repo'],r['file'],r['line']) for r in a3['rows'] if r['changed_by_eng2']}
    ourkeys={(r['repo'],r['file'],r['line']) for r in rows}
    assert a3keys <= ourkeys, sorted(a3keys-ourkeys)
    extras=[r for r in rows if (r['repo'],r['file'],r['line']) not in a3keys]
    assert len(rows)==498 and len(extras)==16
    assert not any(r['classification']=='potentially-newly-refused' for r in rows)
    unresolved=[r for r in rows if r['classification']=='unresolved']
    assert len(unresolved)==49
    # Voice HEAD moved during A3/review; prove both extra Voice operands already
    # existed byte-for-byte at A3's recorded revision, instead of treating drift
    # as a census defect.
    historical=[]
    for row in extras:
        if row['repo']!='Project-Voice':
            continue
        oldhead=next(r['head'] for r in a3['repos'] if r['repo'].endswith('/Project-Voice'))
        argv=['git','-C','/mnt/ForgeRealm/Project-Voice','show',oldhead+':'+row['file']]
        r=subprocess.run(argv,capture_output=True,timeout=30)
        current=(SCRATCH/row['repo']/row['file']).read_bytes()
        assert r.returncode==0 and r.stdout==current
        historical.append(dict(argv=argv,exit=r.returncode,source_sha256=hashlib.sha256(current).hexdigest(),same_at_a3=True))
    # Check live tracked working-tree text as well as pinned HEAD contents,
    # using git grep only. Ignore newline encoding/final-newline differences.
    live=[]
    for repo in data['repos']:
        name=repo['repo']
        argv=['git','-C','/mnt/ForgeRealm/'+name,'grep','-n','-I','-e','^','--','*.py']
        r=subprocess.run(argv,capture_output=True,timeout=60)
        assert r.returncode==0
        lines=defaultdict(list)
        for line in r.stdout.decode(errors='replace').splitlines():
            file,number,text=line.split(':',2)
            lines[file].append(text)
        different=[]
        for source in repo['source_files']:
            file=source['path']
            expected=(SCRATCH/name/file).read_text(errors='replace').splitlines()
            if lines.get(file,[])!=expected:
                different.append(file)
        live.append(dict(repo=name,argv=argv,exit=r.returncode,stdout_sha256=hashlib.sha256(r.stdout).hexdigest(),
                         changed_source_files=different))
    summary=[]
    for repo in data['repos']:
        rr=[r for r in rows if r['repo']==repo['repo']]
        counts={role:dict(Counter(r['classification'] for r in rr if r['role']==role)) for role in ['product','script','test','receipt']}
        entries={entry:sum(r['entry']==entry for r in rr) for entry in sorted(scan.ENTRIES)}
        summary.append(dict(repo=repo['repo'],head=repo['head'],tracked_python=repo['tracked_python'],
                            dtype_sites=len(rr),counts=dict(Counter(r['classification'] for r in rr)),roles=counts,entries=entries))
    result=dict(summary=summary,sites=sorted(rows,key=lambda r:(r['repo'],r['file'],r['line'],r['col'])),
                newly_refused=[],unresolved=unresolved,omitted_by_a3=extras,voice_historical_checks=historical,
                live_worktree_checks=live,original_a3_sites=len(a3keys),
                limits='Static calls and tracked Python; arbitrary reflection/eval, untracked files and external callers are not proven. All other receiver candidates remain in Q3_consumer_scan.json. Default-only calls were inspected in the initial scan and cannot pass an unsupported dtype.')
    (OUT/'Q3_consumer_final.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    print('confirmed newly refused=0; unresolved=49; A3 missing=16; live source differences=',[(x['repo'],x['changed_source_files']) for x in live])


if __name__=='__main__':
    main()
