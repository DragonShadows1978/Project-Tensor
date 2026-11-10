#!/usr/bin/env python3
"""Acceptance receipts, without changing any threshold.
Prior art: A1 verify_gates.py, REV2 L46 maps and ENG1 evidence.py (2026);
SHA256 (NIST FIPS 180-4, 2015), Python AST. Taken: byte identity, exact-ID
checks and existing parent recomputation; new: A2 registered conjunction.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

A = Path(__file__).resolve().parent
R = A.parents[2]
T = R / 'artifacts/eng1/tool'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
write = lambda name, data: (A / name).write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
def file_map(directory):
    return {str(p.relative_to(R)): sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}
def changes(before, after):
    return sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p))
def git(*args):
    return subprocess.run(['git', *args], cwd=R, check=True, capture_output=True, text=True).stdout

before = read(A / 'tensor_cuda_before.json')
after = file_map(R / 'tensor_cuda')
write('tensor_cuda_after.json', after)
protected_before = read(A / 'protected_before.json')
protected_after = {p: sha(R / p) for p in protected_before if (R / p).is_file()}
write('protected_after.json', protected_after)
arms_before = read(A / 'arms_before.json')
arms_after = file_map(T / 'arms')
write('arms_after.json', arms_after)
tracked_before = read(A / 'tracked_before.json')
tracked_after = {p: sha(R / p) for p in tracked_before if (R / p).is_file()}
tool_before = read(A / 'tool_before.json')
tracked_changed = changes(tracked_before, tracked_after)
arm_changed = changes(arms_before, arms_after)
diff_args = ['diff', '7cf46bd5', '--', '.', ':!artifacts/eng1', ':!orders', ':!logs']
diff = git(*diff_args)
(A / 'test_eng1.diff').write_text(diff)
paths = git('diff', '--name-only', '7cf46bd5', '--', '.', ':!artifacts/eng1', ':!orders', ':!logs').splitlines()
old_test = git('show', '7cf46bd5:tensor_cuda/tests/test_eng1.py')
def defs(text):
    return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(text).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
d0, d1 = defs(old_test), defs((R / 'tensor_cuda/tests/test_eng1.py').read_text())
changed_defs = sorted(n for n in d0.keys() | d1.keys() if d0.get(n) != d1.get(n))
stage_names = ['collect_a1_baseline_eng1.json', 'collect_a1_candidate_eng1.json',
    'collect_a1_candidate_eng1_cpu.json', 'collect_a1_candidate_eng1_noe6.json',
    'collect_a1_baseline_regression.json', 'collect_a1_candidate_regression.json',
    'collect_a1_baseline_terrain.json', 'collect_a1_candidate_terrain.json']
ids_names = ['ids_a1_' + n + '.txt' for n in ('eng1', 'eng1_candidate', 'audit', 'eng1_noe6',
    'regression', 'regression_candidate', 'terrain', 'terrain_candidate')]
allowed_tracked = {'tensor_cuda/tests/test_eng1.py', 'artifacts/eng1/tool/source_reading_tests.json'}
allowed_tracked |= {'artifacts/eng1/tool/' + n for n in stage_names + ids_names}
bad_tracked = sorted(set(tracked_changed) - allowed_tracked)
tool_changed = sorted(p for p, h in tool_before.items() if not (R / p).is_file() or sha(R / p) != h)
bad_tool = sorted(set(tool_changed) - allowed_tracked)
scope = dict(diff_base='7cf46bd5', dispatch=read(A / 'dispatch.json')['dispatch'],
    diff_command='git diff 7cf46bd5 -- . \':!artifacts/eng1\' \':!orders\' \':!logs\'',
    diff_paths=paths, changed_definitions=changed_defs, tensor_cuda_changed=changes(before, after),
    src_byte_identical={p:h for p,h in before.items() if p.startswith('tensor_cuda/src/')} ==
                       {p:h for p,h in after.items() if p.startswith('tensor_cuda/src/')},
    protected_changed=changes(protected_before, protected_after), staged_arm_changed=arm_changed,
    unrelated_tracked_changed=bad_tracked, unrelated_tool_changed=bad_tool,
    module_pins={arm: sha(next((T / 'arms' / arm / 'tensor_cuda/tensor_cuda').glob('_tensor_cuda*.so')))
                 for arm in ('baseline', 'candidate')})
scope['passed'] = (paths == ['tensor_cuda/tests/test_eng1.py']
    and changed_defs == ['e3_distance', 'e3_measure']
    and scope['tensor_cuda_changed'] == ['tensor_cuda/tests/test_eng1.py']
    and scope['src_byte_identical'] and not scope['protected_changed'] and not bad_tracked and not bad_tool
    and len(arm_changed) == 20 and all(p.endswith('/tensor_cuda/tests/test_eng1.py') for p in arm_changed)
    and scope['module_pins']['candidate'] == '3e6d5805fb8b89ebfeb2b32c173fd097ef1cfe4a01b3349dc177d4b0c8c0ad53'
    and scope['module_pins']['baseline'] == '73ededa0e73357c8578e58e5eadd87df9f2f3b2d64d631eba32499f2fdb9523e')
write('scope.json', scope)

selectors = read(T / 'source_reading_tests.json')
base_ids = (T / 'ids_a1_regression.txt').read_text().splitlines()
cand_ids = (T / 'ids_a1_regression_candidate.txt').read_text().splitlines()
exclude = {r['test_id'] for r in selectors}
excluded_ids = [i for i in cand_ids if i.split('[', 1)[0] in exclude]
census = dict(selectors=len(selectors), candidate_ids=len(cand_ids), comparison_ids=len(base_ids),
    excluded_ids=excluded_ids, collections={n: read(T / n) for n in stage_names})
census['passed'] = (len(exclude) == 22 and len(cand_ids) == 2133 and len(base_ids) == 2110
    and len(excluded_ids) == 23 and base_ids == [i for i in cand_ids if i not in excluded_ids]
    and all(x['rc'] == 0 and not x['errors'] for x in census['collections'].values()))
write('census.json', census)

trace = read(A / 'q2_classify_reads.json')
groups = {g: read(A / ('q2_readers_' + g + '.json')) for g in 'ABC'}
files = [f for d in groups.values() for f in d['files']]
trace_ids = {k.split('|', 1)[0] for d in groups.values() for k in d['outcomes']}
by_name = {
    'tensor_cuda/tests/test_apa_sp1_1.py::test_registration_and_every_existing_kernel_body_pinned':
        'Fails first on missing artifacts/apa_sp1/sp1_1_before/tensor_cuda__src__kernels.cu (FileNotFoundError); the later live-source read is not reached. See q2_readers_A_retry.stdout.',
    'tensor_cuda/tests/test_apa_sp3_a6.py::test_a6_bridge_all_old_kinds_future_edits_and_receipt_bytes':
        'Fails first on missing artifacts/apa_sp3/build/manifest.json (FileNotFoundError); the later source closure checks are not reached. See q2_readers_A_retry.stdout.',
    'tensor_cuda/tests/test_apa_sp4g_a2.py::test_a2_source_pins_pre_branch_norm_rope_v_and_scale':
        'Passes; directly reads /mnt/ForgeRealm/GraftRepository/core/gemma4_tc.py, outside the worktree-only audit trace. See test_apa_sp4g_a2.py and q2_readers_B.json.'}
completeness = dict(observed_readers=trace['observed_readers'], listed=trace['a1_count'],
    observed_listed=trace['in_both'], missing=trace['observed_not_in_a1'],
    missing_count=len(trace['observed_not_in_a1']), listed_only_by_name=by_name,
    files=len(files), exact_test_ids=len(trace_ids), blocked_events=trace['blocked_events'],
    scripts_byte_identical=all((A / n).read_bytes() == (R / 'artifacts/eng1/rev2' / n).read_bytes()
        for n in ('q2_source_readers.py', 'q2_classify_reads.py')),
    trace_pytest_exits={g: d['pytest_exit'] for g,d in groups.items()})
completeness['passed'] = (not completeness['missing'] and completeness['observed_readers'] == 19
    and completeness['observed_listed'] == 19 and completeness['listed'] == 22
    and set(trace['a1_not_observed']) == set(by_name) and len(files) == len(set(files)) == 49
    and trace_ids == set(cand_ids) and completeness['scripts_byte_identical'])
write('completeness.json', completeness)
lines = ['# CPU trace coverage — A2', '',
    '19 observed source-reader selectors; all 19 listed; **0 missing**. All 49 regression files and 2133 exact IDs ran under the unchanged REV2 attribution/classification scripts. The 36 blocked operations and trace pytest failures are retained instrument outcomes; this is a read census, not a passing regression suite.', '',
    '| Listed only by name | Reason |', '|---|---|']
lines += ['| `' + t + '` | ' + why + ' |' for t,why in by_name.items()]
lines += ['', 'Residual: CPU reads after an initial failing CUDA call are not observed. Imports, traceback rendering, tree walks and artifact/order receipt reads are excluded exactly as in REV2. The pyproject.toml collection read is retained under collection_reads and is not attributed to a test.']
(A / 'TRACE_COVERAGE.md').write_text('\n'.join(lines) + '\n')

cpu = read(A / 'cpu_result.json')
cpu_parent = read(A / 'cpu.exit.json')
cpu_pass = (cpu_parent['exit'] == cpu['pytest_exit'] == 0 and cpu['passed'] == 47
    and cpu['exact_ids_match'] and not cpu['non_pass_reports'] and cpu['CUDA_VISIBLE_DEVICES'] == '')
e3 = read(A / 'e3_rows_cpu.json')
e3_pass = (read(A / 'e3_rows.exit.json')['exit'] == 0 and e3['row_count'] == 40
    and e3['nonfinite_row_count'] == 4 and len(e3['branch_checks']) == 7
    and all(c['passed'] for c in e3['branch_checks']))
sys.path.insert(0, str(T))
os.environ.update(ENG1_A2_AUDIT_ROUTING='1', ENG1_AUDIT='1')
runpy.run_path(str(A / 'sitecustomize.py'))
import evidence
seal, start = evidence.verify_start(A / 'audit')
audit = evidence.recompute_line(A / 'audit', 'candidate_eng1', seal, start)
parent = read(Path(audit['selected_attempt']) / 'parent_result.json')
child = read(Path(audit['selected_attempt']) / 'child_result.json')
audit_pass = (audit['passed'] and audit['mode'] == 'cpu' and parent['exit'] == child['pytest_exit'] == 0
    and not parent['timeout'] and parent['child_exited'] <= parent['recorded']
    and len(child['collected_ids']) == 47 and read(A / 'audit.exit.json')['exit'] == 0)
gates = dict(scope=scope['passed'], census=census['passed'], completeness=completeness['passed'],
    E3_CPU_instrument=e3_pass, CPU_47=cpu_pass, audit=audit_pass,
    seal_file='artifacts/eng1/tool/staging_seal_a2.json', seal_sha256=sha(T / 'staging_seal_a2.json'),
    sealed_arms=len(seal['arms']), arms_pending=seal['arms_pending'], audit_mode=audit['mode'],
    audit_parent='artifacts/eng1/a2/audit/candidate_eng1/attempt-0001/parent_result.json',
    evidence_class='CPU unit execution, CPU measurement-instrument checks, runtime read trace, byte identity and completed-child CPU audit')
gates['accepted'] = all(gates[k] for k in ('scope', 'census', 'completeness', 'E3_CPU_instrument', 'CPU_47', 'audit'))
write('gates.json', gates)
print(json.dumps(gates, indent=2))
assert gates['accepted']
