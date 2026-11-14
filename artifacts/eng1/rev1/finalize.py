# Prior art: SHA-256 NIST FIPS180-4 (2015), exact-map comparisons; Python JSON.
# Taken: byte identity. Ours: final rev1 evidence inventory and completeness checks.
import json,hashlib,datetime,ast
from pathlib import Path
root=Path.cwd(); rev=root/'artifacts/eng1/rev1'
before=json.loads((rev/'worktree_before.json').read_text())
after=json.loads((rev/'worktree_after.json').read_text())
diff=sorted(p for p in set(before)|set(after) if before.get(p)!=after.get(p))
result=dict(passed=not diff,before_count=len(before),after_count=len(after),differences=diff,
            comparison='sha256 of every regular file and symlink target in this worktree, excluding rev1',
            recorded=datetime.datetime.now(datetime.timezone.utc).isoformat())
(rev/'worktree_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
assert diff == ['logs/eng1_rev1.log'], diff  # Preserve the RED gate; finalize the review, not the gate.
log_probe=json.loads((rev/'scope_log_probe.json').read_text())
assert log_probe['append_only_proved'] and log_probe['all_other_worktree_paths_unchanged']
f=json.loads((rev/'findings.json').read_text())
assert len(f['findings'])==5 and len({x['id'] for x in f['findings']})==5
for row in f['findings']:
    assert row['severity'] in ('BLOCKER','MAJOR','MINOR','NIT')
    assert (root/row['file']).is_file()
    assert 0<row['line']<=len((root/row['file']).read_text().splitlines())
    for key in ('reproduction','suggested_fix'): assert row[key]
    for path in row['receipts']:assert (rev/path).is_file(),path
f['status']='complete';f['verdict']='REJECT';f['worktree_unchanged']=False
f['scope_check']=result
f['scope_attribution']=log_probe['attribution']
f['all_other_paths_unchanged']=True
f['review_scope']='CPU-only source/test/tool review; GPU G5 not executed'
(rev/'findings.json').write_text(json.dumps(f,indent=2,allow_nan=False)+'\n')
# Validate all authored JSON and Python scripts; no compile caches emitted.
strict=lambda val:(_ for _ in ()).throw(ValueError('non-JSON token '+val))
json_count=0;script_count=0
for p in rev.glob('*.json'):
    if p.name=='review_files.json':continue
    json.loads(p.read_text(),parse_constant=strict);json_count+=1
for p in rev.glob('*.py'):
    ast.parse(p.read_text());script_count+=1
# Check own gate outcome assertions, including expected negative results.
expect={'scorer':1,'arm_baseline':0,'arm_candidate':0,'numerics':0,'semantics':0,'error_bounds':0,
        'coverage_gaps':0,'census_scope':0,'backward_bound':0,'e6_cpu':0,'m14_cpu':1,
        'cpu_gates':0,'mutant_bounds':0,'caller_tool_audit':0,'report_build':0,'snapshot_final':0}
for label,rc in expect.items():
    d=json.loads((rev/(label+'.exit.json')).read_text())
    assert d['exit']==rc and not d['timeout'],(label,d)
control=[]
seal=json.loads((root/'artifacts/eng1/tool/staging_seal.json').read_text())
for arm,folder in (('C_COPY','_unmutated_copy'),('C_UBSAN','C_UBSAN')):
    receipt=json.loads((root/'artifacts/eng1/mutants'/folder/'receipt.json').read_text())
    mod=next((root/'artifacts/eng1/tool/arms'/arm/'tensor_cuda/tensor_cuda').glob('_tensor_cuda*.so'))
    h=hashlib.sha256(mod.read_bytes()).hexdigest()
    assert h==receipt['module_sha256']==seal['engine_sha256'][arm]
    control.append(dict(arm=arm,sha256=h,receipt_and_pin_match=True))
(rev/'control_pins.json').write_text(json.dumps(control,indent=2)+'\n')
with (rev/'REPORT.md').open('a') as out:
    out.write('\n## Final scope-preservation check\n\n')
    out.write(f'**RED:** before and after maps contain {len(before):,} files each, with exactly one changed path outside rev1: logs/eng1_rev1.log. The initial log hash matches its first 119,943 bytes, proving an append-only transcript change; the enclosing session logger is inferred from the recorded tool/commentary stream, not independently attested. Every other path, including source, tests, frozen receipts, modules and builds, is byte-identical. This does NOT satisfy immutable L46. No exception, restoration, deletion or write to that log was applied. See worktree_comparison.json, scope_log_probe.json and worktree_before/after.json. All 5 substantive findings remain open; verdict REJECT.\n')
with (rev/'LEDGER.md').open('a') as out:
    out.write(f'- Final scope check RED: {len(before)} paths in each map, one append-only external transcript log change (logs/eng1_rev1.log); every other path unchanged. Initial finalize exited 1 on this gate and its failure receipt is preserved. No immutable exclusion was added. Control hashes match receipts/seal. Validated {json_count} JSON files and parsed {script_count} scripts without caches; expected test/audit exits matched, no timeout. Review complete, REJECT with 2 MAJOR, 2 MINOR and 1 NIT plus the failed whole-worktree identity condition.\n')
qa=dict(passed=False,evidence_checks_passed=True,scope_gate_passed=False,initial_scope_gate_exit=1,json_files_validated=json_count,python_scripts_parsed=script_count,
        expected_receipt_exits=expect,findings=len(f['findings']),scope_unchanged=False,all_other_paths_unchanged=True,
        no_gpu_run=True,no_deletes=True,outputs='rev1 plus /tmp/eng1_rev1_scratch')
(rev/'review_validation.json').write_text(json.dumps(qa,indent=2)+'\n')
files=[]
for p in sorted(rev.rglob('*')):
    if p.is_file() and p.name not in ('review_files.json','finalize_report.stdout','finalize_report.stderr','finalize_report.exit.json'):
        data=p.read_bytes();files.append(dict(path=str(p.relative_to(rev)),size=len(data),sha256=hashlib.sha256(data).hexdigest()))
(rev/'review_files.json').write_text(json.dumps(dict(files=files,excluded_self=True,
      publication_metadata=['finalize_report.stdout','finalize_report.stderr','finalize_report.exit.json'], publication_note='Supervisor publication metadata is written after this inventory'),indent=2)+'\n')
print(json.dumps(result))
print('Validated JSON',json_count,'Python scripts',script_count,'inventoried files',len(files))
