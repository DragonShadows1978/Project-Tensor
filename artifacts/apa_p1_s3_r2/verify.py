"""Final receipt consistency/scope audit.
Prior art: SHA-256 (NIST, 2015), exact set/record comparison (P1, 2026).
No new selection, reduction, optimization, or numerical claim.
"""
from pathlib import Path
import ast,datetime,hashlib,json,re
A=Path('artifacts/apa_p1_s3_r2')
def read(name):return json.loads((A/name).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def all_passed(receipt,ids):
    return all(any(r['when']=='call' and r['outcome']=='passed' for r in receipt['reports'].get(i,[])) for i in ids)
reg=read('REGISTRATION.json');cpu=read('CPU_TESTS_verified.json');expected=read('EXPECTED.json');dry=read('DRYRUN.json');build=read('BUILD_COMPLETED.json')
current={p:sha(p) for p in read('SCOPE_BEFORE.json') if Path(p).exists()}
changed=sorted(p for p,h in read('SCOPE_BEFORE.json').items() if current.get(p)!=h)
allowed=['tensor_cuda/src/apa_spo/eb3.cuh','tensor_cuda/src/apa_spo/eb4.cuh','tensor_cuda/src/bindings.cpp']
core='tensor_cuda/tests/'
struct=ast.literal_eval((A/'cpu_compiled/structural_results.txt').read_text())
fp=read('FIXTURE_PROVENANCE.json');design=(A/'DEFAULT_SWITCH_DESIGN.md').read_text()
checks={
'G1':all_passed(cpu,[core+'test_apa_p1_tickets.py::test_per_device_ticket_ownership']) and expected['lines'][0]['case']=='threads',
'G2':all_passed(cpu,[core+'test_apa_p1_early_counter.py::test_early_counter_compiled_structure',core+'test_apa_p1_early_counter.py::test_two_cta_operand_census_cpu']) and [(x[0],x[1],x[2]) for x in struct]==[('eb3',False,0),('eb3',True,1),('eb4',False,0),('eb4',True,1)],
'G3':all_passed(cpu,[core+'test_apa_p1_exact_tie.py::test_fixture_matches_lead_ast',core+'test_apa_p1_exact_tie.py::test_exact_tie_fixture_cpu']) and sha(fp['source'])==fp['source_sha256'],
'G4':all('## ('+x+')' in design for x in 'abcd') and (A/'CALLER_ARGUMENTS.md').read_text() in design and (A/'CALLER_CENSUS_FINAL_COMMANDS.md').read_text() in design,
'G5':dry['status']=='PASS' and len(dry['rows'])==6 and all(x['exit']==78 for x in dry['rows']) and dry['expected_sha256']==sha(A/'EXPECTED.json') and all(sha(p)==h for p,h in expected['code_sha256'].items()) and (A/'EXPECTED.json').stat().st_mtime_ns<(A/'STAGED_GPU_LINES.sh').stat().st_mtime_ns,
'G6':read('OUTCOMES_DIFF.json')['status']=='PASS' and read('SASS_DIFF.json')['status']=='PASS' and read('SASS_DIFF.json')['outside_select_differences']==0 and read('SASS_BASELINE_PROVENANCE.json')['status']=='PASS' and sha(build['module'])==build['module_sha256'] and cpu['module']['sha256']==build['module_sha256'],
'G7':changed==sorted(allowed) and sha(reg['order'])==reg['order_sha256'] and all(read(x)['cuda_visible_devices']=='' for x in ('OUTCOMES_before.json','OUTCOMES_after.json','CPU_TESTS_verified.json','DRYRUN.json','BUILD_COMPLETED.json')) and datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(reg['deadline_utc'].replace('Z','+00:00')),
}
manual=dict(no_gpu='No device jobs were executed; offline CUDA compilation/cuobjdump only. Every actual command had CUDA_VISIBLE_DEVICES empty. Lead GPU children were stopped by real forge-gpu-run dry-run 78.',no_git='No git commands/writes; only direct read of the GRM checkout config to identify the named census root.',no_deletes='No seat deletion or retirement/move; build and test-generated products remain. /tmp scratch retained.',single_stream='Legacy default stream only; no multi-stream/capture support.',source_scope='591 initial source/config/doc hashes; exactly three authorized existing source files changed. Three new authorized test files. Artifacts and scratch only for all other seat writes.',blind_review='Not run; lead follows this and default-switch round with independent cross-model red team.')
r=dict(status='ACCEPT_CPU_STAGING' if all(checks.values()) else 'RED',gates={k:'PASS' if v else 'FAIL' for k,v in checks.items()},changed_existing=changed,source_files_audited=len(current),cpu_new_calls=5,gpu_tests_skipped=4,manual_audit=manual,verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(A/'GATES.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));raise SystemExit(0 if all(checks.values()) else 1)
