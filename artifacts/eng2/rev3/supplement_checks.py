"""Source spot checks, staging ID deltas, and original dry-run corroboration.

Prior art: standard source inspection and pytest ID set comparisons (pytest
project); SHA-256 (NIST FIPS 180-4 2015, unverified -- lead to check).
No new algorithm: only review-specific source/receipt bindings.
"""
import ast
import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path.cwd()
OUT=ROOT/'artifacts/eng2/rev3'
S=Path(os.environ['TMPDIR'])/'consumer_sources'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    a3=json.loads((ROOT/'artifacts/eng2/a3/CONSUMER_CENSUS.json').read_text())
    spots=[
        ('Project-Voice','engine/tensor_requests/p1t_tests/adapters/tensorcuda_stub.py',205,
         'holds for recorded harness callers; dtype is an unguarded external parameter, so retained unresolved',
         [('engine/tensor_requests/p1t_tests/run_p1t.py',605),('engine/tensor_requests/p1t_tests/adapters/tensorcuda_stub.py',201)]),
        ('GRAPA-Native-LLM','grapa/attention.py',82,
         'holds for tensor_cuda q: q.dtype is an engine dtype string; call uses q.device and tensor_cuda matmul',
         [('grapa/attention.py',80),('grapa/attention.py',81)]),
        ('GRAPA-Native-LLM','artifacts/cc5/mutation/answer_span_shift/grapa/attention.py',82,
         'holds on the same tensor_cuda query path in the tracked mutant receipt copy',
         [('artifacts/cc5/mutation/answer_span_shift/grapa/attention.py',80)]),
        ('GRAPA-Native-LLM','scripts/cc10_ablation.py',2048,
         'holds: scores is produced by tensor_cuda matmul and tensor arithmetic, so scores.dtype is an engine string',
         [('scripts/cc10_ablation.py',2042),('scripts/cc10_ablation.py',2046)]),
        ('GraftRepository','artifacts/grm_xm3/evidence/trinity_nano_tc.py',316,
         'holds: _compute_dtype returns BlockTC.COMPUTE_DTYPE, with in-file bfloat16 default and loader assignment',
         [('artifacts/grm_xm3/evidence/trinity_nano_tc.py',42),('artifacts/grm_xm3/evidence/trinity_nano_tc.py',165),('artifacts/grm_xm3/evidence/trinity_nano_tc.py',951)]),
        ('GraftRepository','core/deepseek_v2_lite_tc.py',115,
         'qualified reading holds: dtype or BlockTC.COMPUTE_DTYPE; registered callers use supported values; external override remains unresolved',
         [('core/deepseek_v2_lite_tc.py',105),('core/deepseek_v2_lite_tc.py',108),('core/deepseek_v2_lite_tc.py',829)]),
        ('GraftRepository','core/gemma4_tc.py',139,
         'qualified reading holds: cdt is shared compute dtype, assigned bfloat16 by loader; mutable global remains unresolved',
         [('core/gemma4_tc.py',138),('core/gemma4_tc.py',1086),('core/mistral7b_tc.py',558)]),
        ('GraftRepository','core/gemma4_tc.py',740,
         'holds: sc is tensor_cuda matmul result reshaped without changing dtype; sc.dtype is an engine string',
         [('core/gemma4_tc.py',730),('core/gemma4_tc.py',731),('core/gemma4_tc.py',741)]),
        ('GraftRepository','core/minicpm3_tc.py',292,
         'qualified reading holds: shared compute dtype is float16 by default and bfloat16 in loader; mutable global retained unresolved',
         [('core/minicpm3_tc.py',293),('core/minicpm3_tc.py',437),('core/mistral7b_tc.py',558)]),
        ('GraftRepository','scripts/gpt_oss20b_e1_diag.py',2525,
         'qualified reading holds: runtime BlockTC.COMPUTE_DTYPE feeds compute_dtype; runtime loader imports tensor_cuda; mutable setting retained unresolved',
         [('scripts/gpt_oss20b_e1_diag.py',2513),('scripts/gpt_oss20b_e1_diag.py',2514),('scripts/gpt_oss20b_e1_diag.py',2520),('scripts/gpt_oss20b_router_telemetry.py',330)]),
        ('GraftRepository','scripts/gpt_oss20b_expert_e1.py',4030,
         'qualified reading holds: runtime dictionary supplies tc and shared BlockTC dtype; no claim about arbitrary setters',
         [('scripts/gpt_oss20b_expert_e1.py',4022),('scripts/gpt_oss20b_expert_e1.py',4023),('scripts/gpt_oss20b_expert_e1.py',4028),('scripts/gpt_oss20b_expert_e1.py',585)]),
    ]
    records=[]
    a3sites={(r['repo'],r['file'],r['line']) for r in a3['unresolved_changed']}
    for repo,file,line,reading,refs in spots:
        assert (repo,file,line) in a3sites
        snippets=[]
        for path,n in [(file,line)]+refs:
            raw=(S/repo/path).read_bytes()
            lines=raw.decode().splitlines()
            snippets.append(dict(file=path,line=n,sha256=sha(raw),excerpt='\n'.join(f'{i}: {lines[i-1]}' for i in range(max(1,n-2),min(len(lines),n+3)+1))))
        records.append(dict(repo=repo,file=file,line=line,seat_reading=reading,reading_supported=True,evidence_class='reasoning from source',snippets=snippets))
    (OUT/'Q3_spot_checks.json').write_text(json.dumps(records,indent=2)+'\n')
    # Bind the engine dtype-name reading to the reviewed commit's code, not to
    # later live edits. This reads source only; no engine function executes.
    dtype_sources=[]
    for path,needle in [('tensor_cuda/src/kernels.cu','const char* dtype_name'),('tensor_cuda/src/bindings.cpp','.def_property_readonly("dtype"')]:
        argv=['git','show','10738fe3:'+path]
        r=subprocess.run(argv,capture_output=True,timeout=30)
        assert r.returncode==0
        lines=r.stdout.decode().splitlines()
        hits=[i for i,l in enumerate(lines) if needle in l]
        assert hits,(path,needle)
        dtype_sources.append(dict(argv=argv,sha256=sha(r.stdout),excerpts=['\n'.join(f'{j+1}: {lines[j]}' for j in range(max(0,i-1),min(len(lines),i+14))) for i in hits]))
    (OUT/'Q3_engine_dtype_source.json').write_text(json.dumps(dtype_sources,indent=2)+'\n')
    q1=json.loads((OUT/'Q1_ids.json').read_text())
    removed={x.split('::')[-1] for x in q1['baseline_diff']['removed']}
    census=[]
    for name in ['ids_regnorm.txt','ids_regnorm_candidate.txt','ids_regdtype.txt','ids_eng2.txt','ids_audit.txt']:
        values=[]
        for commit in ['e2f166b8','10738fe3']:
            argv=['git','show',commit+':artifacts/eng2/tool/'+name]
            r=subprocess.run(argv,capture_output=True,timeout=30)
            assert r.returncode==0
            values.append(r.stdout)
        before=set(values[0].decode().splitlines()); after=set(values[1].decode().splitlines())
        lost=sorted(before-after); gained=sorted(after-before)
        assert not gained
        if name.startswith('ids_regnorm'):
            assert {x.split('::')[-1] for x in lost}==removed and len(lost)==10
        else:
            assert values[0]==values[1]
        census.append(dict(file=name,before=len(before),after=len(after),removed=lost,added=gained,
                           before_sha256=sha(values[0]),after_sha256=sha(values[1])))
    newsha=q1['candidate_10738fe3']['source_sha256']
    seal=json.loads((ROOT/'artifacts/eng2/tool/staging_seal.json').read_text())
    assert all(arm['tensor_cuda/tests/test_eng1.py']==newsha for arm in seal['arms'].values())
    pins=json.loads((ROOT/'artifacts/eng2/L38_PINS.json').read_text())
    assert pins['expected_changes']=={}
    (OUT/'Q1_staged_censuses.json').write_text(json.dumps(dict(censuses=census,all_17_arm_tests_equal_reviewed_commit=True,pins_empty=True),indent=2)+'\n')
    tool=ROOT/'artifacts/eng2/tool'
    dry=json.loads((ROOT/'artifacts/eng2/a3/G4_DRY_dry_4.json').read_text())
    registered={x['label']:x for x in json.loads((tool/'gpu_lines.json').read_text())}
    corroborated=[]
    for row in dry['rows']:
        line=registered[row['label']]
        drycmd=row['s1_dry_run']
        assert drycmd['min']==line['minimum'] and drycmd['max']==line['maximum'] and drycmd['margin']==30
        assert drycmd['label']=='eng2-'+line['label']
        assert drycmd['argv']==['timeout','--kill-after=10s',str(line['inner'])+'s','python3','run_gpu_tests.py',line['arm'],line['suite']]
        if 'dependency_refusal' in row:
            assert not row['attempts'] and row['staged_exit']==1
            expected='mutations require a passing candidate_eng2' if line['role']=='mutant' else 'candidate regression requires a passing paired baseline'
            # Exact refusal text is retained; check the registered dependency's
            # role and name without assuming an undocumented error spelling.
            assert ('passing candidate_eng2' in row['dependency_refusal'] if line['role']=='mutant' else 'baseline' in row['dependency_refusal'])
            corroborated.append(dict(label=line['label'],kind='dependent',dependency_refusal=row['dependency_refusal'],s1_exit=row['s1_rc'],registered_argv=True))
        else:
            assert row['governor_exit']==78 and row['attempts']==['attempt-0001'] and not row['child_started']
            base=tool/'gpu_runs/dry_4'/line['label']/'attempt-0001'
            parentraw=(base/'parent_result.json').read_bytes(); lograw=(base/'stdout.log').read_bytes()
            parent=json.loads(parentraw)
            assert parent['exit']==78 and not parent['passed']
            assert 'DRY RUN' in lograw.decode()
            corroborated.append(dict(label=line['label'],kind='root',parent=parent,
                                      parent_sha256=sha(parentraw),stdout_sha256=sha(lograw),registered_argv=True))
    assert CounterLike(corroborated)=={'root':4,'dependent':17}
    (OUT/'Q4_dry_corroborated.json').write_text(json.dumps(corroborated,indent=2)+'\n')
    print('11 original unresolved changed-entry spot checks support the seat readings with stated qualifications.')
    print('Staged ID censuses and empty pins verified; 17 arm test files match reviewed after commit.')
    print('Dry receipt: 4 roots at governor exit 78, 17 dependency refusals; all 21 S1 argv/bounds registered.')


def CounterLike(rows):
    return {kind:sum(r['kind']==kind for r in rows) for kind in {'root','dependent'}}


if __name__=='__main__':
    main()
