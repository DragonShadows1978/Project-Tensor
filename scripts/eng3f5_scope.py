"""F5 scope receipts. Prior art: F4 scope census (2026), SHA-256 (NIST,
2001), Python difflib (PSF). Reuse metadata/hash manifests and unified diff;
ours: the F5 allowed set. Unverified -- lead to check: NIST SHA-256, difflib.
"""
import ast
import base64
import difflib
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/eng3f5'
HOST = 'tensor_cuda/tests/test_eng3_host.py'
SUPPORT = 'tensor_cuda/tests/support/cudart_count.c'
SOURCE = 'artifacts/eng3/tools/cudart_count.c'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot():
    files = {}
    for base, dirs, names in os.walk(ROOT, followlinks=False):
        dirs[:] = [d for d in dirs if not Path(base, d).is_symlink()]
        for name in names:
            p = Path(base, name)
            st = p.lstat()
            files[str(p.relative_to(ROOT))] = [st.st_size, st.st_mtime_ns, st.st_mode,
                                              os.readlink(p) if p.is_symlink() else None]
    protected = {}
    for directory in ('tensor_cuda/src', 'tensor_cuda/tensor_cuda',
                      'tensor_cuda/build-eng3f3/source_H/tensor_cuda', 'tensor_cuda/tests'):
        for p in (ROOT / directory).rglob('*'):
            if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts:
                protected[str(p.relative_to(ROOT))] = sha(p)
    protected[SOURCE] = sha(ROOT / SOURCE)
    protected['orders/ENG3_F5_SHIM_SELF_CONTAINED.md'] = sha(ROOT / 'orders/ENG3_F5_SHIM_SELF_CONTAINED.md')
    admin = ROOT / '.git'
    if admin.is_file():
        admin = (ROOT / admin.read_text().strip().removeprefix('gitdir: ')).resolve()
    git_admin = {str(p): sha(p) for p in admin.rglob('*') if p.is_file() and not p.is_symlink()}
    return {'files': files, 'protected_sha256': protected, 'git_admin_sha256': git_admin}


def allowed(path):
    return (path in (HOST, SUPPORT) or path.startswith('artifacts/eng3f5/')
            or (path.startswith('scripts/eng3f5_') and path.endswith('.py'))
            or path.startswith('logs/eng3f5_'))


def outside_test(text):
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    ranges = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'test_g1_cpu_no_cuda_calls':
            ranges.append((node.lineno, node.end_lineno))
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SHIM' for t in node.targets):
            ranges.append((node.lineno, node.end_lineno))
    assert len(ranges) == 2
    return ''.join(line for n, line in enumerate(lines, 1) if not any(a <= n <= b for a, b in ranges))


def main():
    if sys.argv[1] == 'before':
        data = snapshot()
        data['host_bytes_base64'] = base64.b64encode((ROOT / HOST).read_bytes()).decode()
        with (ART / 'SCOPE_BEFORE.json').open('x') as f:
            json.dump(data, f, indent=2)
        print('Before census:', len(data['files']), 'files; protected:', len(data['protected_sha256']))
        return
    before = json.loads((ART / 'SCOPE_BEFORE.json').read_text())
    after = snapshot()
    (ART / 'SCOPE_AFTER.json').write_text(json.dumps(after, indent=2))
    removed = sorted(set(before['files']) - set(after['files']))
    changed = sorted(p for p in set(before['files']) | set(after['files'])
                     if before['files'].get(p) != after['files'].get(p))
    old = base64.b64decode(before['host_bytes_base64']).decode()
    new = (ROOT / HOST).read_text()
    diff = ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                      fromfile='before/' + HOST, tofile='after/' + HOST))
    diff += ''.join(difflib.unified_diff([], (ROOT / SUPPORT).read_text().splitlines(True),
                                       fromfile='/dev/null', tofile='after/' + SUPPORT))
    (ART / 'source.diff').write_text(diff)
    protected_changes = [p for p, h in before['protected_sha256'].items()
                         if p != HOST and after['protected_sha256'].get(p) != h]
    result = {'changed_metadata_paths': changed, 'removed': removed,
              'outside_allowed_set': [p for p in changed if not allowed(p)],
              'protected_content_changes': protected_changes,
              'git_admin_identical': before['git_admin_sha256'] == after['git_admin_sha256'],
              'host_outside_edit_sites_identical': outside_test(old) == outside_test(new),
              'source_sha256': {p: sha(ROOT / p) for p in (SOURCE, SUPPORT)},
              'scope_evidence_limit': 'Worktree metadata plus protected content and seat git-admin hashes; no system-wide write audit.'}
    result['pass'] = (not removed and not result['outside_allowed_set'] and not protected_changes
                      and result['git_admin_identical'] and result['host_outside_edit_sites_identical']
                      and result['source_sha256'][SOURCE] == result['source_sha256'][SUPPORT])
    (ART / 'G4.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
