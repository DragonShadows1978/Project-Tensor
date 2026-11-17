"""F4 closure audit. Prior art: F3 r2 scope census (2026), Python AST,
NIST SHA-256 (2001). Taken: metadata/hash comparisons; ours: explicit F4
AST boundaries and file inventory. External leads unverified: Python ast,
NIST SHA-256. No prior art known to me for this specific report assembly.
"""
import ast
import difflib
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/eng3f4'


def dump(name, obj):
    (ART / name).write_text(json.dumps(obj, indent=2))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def census(base):
    rows = {}
    for path, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in ('__pycache__', '.pytest_cache') and not Path(path, d).is_symlink()]
        for name in files:
            p = Path(path, name)
            st = p.lstat()
            row = {'size': st.st_size, 'mtime_ns': st.st_mtime_ns, 'mode': st.st_mode}
            if p.is_symlink():
                row['symlink'] = os.readlink(p)
            rows[str(p.relative_to(base))] = row
    return rows


def main():
    before = json.loads((ART / 'SCOPE_BEFORE.json').read_text())
    after = census(ROOT)
    changed = sorted(i for i in before.keys() & after.keys() if before[i] != after[i])
    new = sorted(after.keys() - before.keys())
    missing = sorted(before.keys() - after.keys())

    def allowed(path):
        return (path in ('tensor_cuda/tests/test_eng3_f3.py', 'tensor_cuda/tests/test_eng3_host.py')
                or path.startswith('artifacts/eng3f4/') or path.startswith('logs/eng3f4_')
                or (path.startswith('scripts/eng3f4_') and path.endswith('.py') and path not in before))

    unexpected = [p for p in changed + new if not allowed(p)]
    old_git = json.loads((ART / 'GIT_BEFORE.json').read_text())
    git_now = {base: census(Path(base)) for base in old_git}
    git_diffs = {base: sorted(k for k in set(rows) | set(git_now[base]) if rows.get(k) != git_now[base].get(k))
                 for base, rows in old_git.items()}
    dump('GIT_AFTER.json', git_now)
    product = {}
    for arm, name in [('H', 'artifacts/eng3f3/build/BUILD_H.json'), ('G', 'artifacts/eng3f2/build/BUILD_G.json')]:
        rec = json.loads((ROOT / name).read_text())
        product[arm+'_module'] = sha(ROOT / rec['module']) == rec['module_sha256']
        if arm == 'H':
            product['H_live_sources'] = {p: sha(ROOT / 'tensor_cuda' / p) == expected for p, expected in rec['sources'].items()}
    all_product = product['H_module'] and product['G_module'] and all(product['H_live_sources'].values())

    host_old = (ART / 'before/test_eng3_host.py').read_text()
    host_new = (ROOT / 'tensor_cuda/tests/test_eng3_host.py').read_text()

    def outside_cuda(text):
        node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'test_f2_cpu_overflow_message')
        branch = next(n for n in node.body if isinstance(n, ast.If) and ast.unparse(n.test) == "device == 'cuda'")
        lines = text.splitlines(keepends=True)
        return ''.join(lines[:branch.lineno-1] + ['<bare-cuda branch>\n'] + lines[branch.end_lineno:])

    host_confined = outside_cuda(host_old) == outside_cuda(host_new)
    f3_old = (ART / 'before/test_eng3_f3.py').read_text()
    f3_new = (ROOT / 'tensor_cuda/tests/test_eng3_f3.py').read_text()
    old_ast, new_ast = ast.parse(f3_old), ast.parse(f3_new)
    permitted = {'child', 'run_child', 'test_f1_gpu_device_index', 'test_f3_cpu_no_device_query'}
    old_other = [ast.dump(n) for n in old_ast.body if not isinstance(n, ast.FunctionDef) or n.name not in permitted]
    new_other = [ast.dump(n) for n in new_ast.body if not isinstance(n, ast.FunctionDef) or n.name not in permitted]
    defs_old = {n.name: n for n in old_ast.body if isinstance(n, ast.FunctionDef)}
    defs_new = {n.name: n for n in new_ast.body if isinstance(n, ast.FunctionDef)}
    memory_unchanged = ast.dump(defs_old['child'].body[1].orelse[0]) == ast.dump(defs_new['child'].body[1].orelse[0])
    original_query = [ast.dump(n) for n in defs_old['test_f3_cpu_no_device_query'].body]
    child_query = [ast.dump(n) for n in defs_new['child'].body[1].body[1:-1]]
    query_assertions_preserved = original_query == child_query
    dead_overflow_absent = not any(isinstance(n, ast.Compare) and isinstance(n.left, ast.Name) and n.left.id == 'mode'
                                   and any(isinstance(c, ast.Constant) and c.value == 'overflow' for c in n.comparators)
                                   for n in ast.walk(new_ast))
    diff = ''
    for name, old, current in [('test_eng3_f3.py', f3_old, f3_new), ('test_eng3_host.py', host_old, host_new)]:
        diff += ''.join(difflib.unified_diff(old.splitlines(keepends=True), current.splitlines(keepends=True),
                                           fromfile='before/'+name, tofile='tensor_cuda/tests/'+name))
    (ART / 'tests.diff').write_text(diff)
    pins = json.loads((ART / 'INPUT_PINS.json').read_text())
    input_pins = {p: sha(ROOT / p) == expected for p, expected in pins.items()}
    stage = {}
    for line in ('gpu-tests', 'g3-g1', 'g3-g2'):
        out = ART / 'staged' / line / 'dry'
        d = json.loads((out / 'RESULT.json').read_text())
        prefix = json.loads((out / 'CPU_PREFIX.json').read_text())
        stage[line] = {'lead_command': 'timeout 590s env FORGE_GPU_DRY_RUN=0 bash artifacts/eng3f4/staged/launch-'+line+'.sh',
                       'expected_forge_exit': 0, 'expected_inner_exit': 0, 'expected_launcher_exit': 0,
                       'dry_result': d, 'prefix_pass': prefix['pass'],
                       'real_result': 'artifacts/eng3f4/staged/'+line+'/real/RESULT.json',
                       'full_environment_receipt': 'artifacts/eng3f4/staged/'+line+'/dry/PRELEASE.json'}
        assert d['pass'] and d['forge_exit'] == d['launcher_exit'] == 78 and d['inner_exit'] is None
        assert not d['gpu_run'] and prefix['pass']
        assert sha(out / 'GPU.log') == d['gpu_log_sha256']
    dump('STAGED_LINES.json', stage)
    tests_ok = host_confined and old_other == new_other and memory_unchanged and query_assertions_preserved and dead_overflow_absent
    result = {'pass': not unexpected and not missing and not any(git_diffs.values()) and all_product and tests_ok and all(input_pins.values()),
              'changed_existing': changed, 'new_files_at_audit': new, 'missing': missing, 'unexpected_writes': unexpected,
              'git_admin_metadata_differences': git_diffs, 'product_pins': product,
              'host_only_bare_cuda_branch_changed': host_confined,
              'F3_other_statements_unchanged': old_other == new_other,
              'memory_child_unchanged': memory_unchanged, 'original_no_device_assertions_preserved': query_assertions_preserved,
              'dead_overflow_mode_absent': dead_overflow_absent, 'input_pins': input_pins,
              'registration_sha256': sha(ART / 'REGISTRATION.md'),
              'registration_unchanged': sha(ART / 'REGISTRATION.md') == '701a401b5c40d93279c7f6337b73f0b07e06324cf242588d41c7ce4ec00d1a35',
              'no_git_commands': True, 'no_rebuild': True, 'no_gpu_run': True, 'no_deletes_or_DELETE_ME_moves': True,
              'limits': 'Workspace metadata, G/H module and H source hashes, AST boundaries, and git administrative metadata; not a global filesystem audit.'}
    result['pass'] = result['pass'] and result['registration_unchanged']
    dump('G5.json', result)
    dump('SCOPE_AFTER.json', after)
    print(json.dumps({k: result[k] for k in ('pass', 'unexpected_writes', 'missing', 'host_only_bare_cuda_branch_changed',
                                           'F3_other_statements_unchanged', 'memory_child_unchanged', 'original_no_device_assertions_preserved')}))
    return 0 if result['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
