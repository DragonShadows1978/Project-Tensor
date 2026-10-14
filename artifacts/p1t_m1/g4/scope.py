"""P1T-M1 G3 emulation without git: every file in the worktree vs production's checkout (tracked files at 9df5bef,
clean per the session's git status snapshot). Skips gitignored build products (tensor_cuda/build/, *.so,
__pycache__). Prints a git-diff-stat-like table: insertions/deletions from difflib."""
import difflib, os, sys
W, P = sys.argv[1], sys.argv[2]
skip_dirs = {'.git', '__pycache__', '.pytest_cache'}
rows = []
for root, dirs, files in os.walk(W):
    rel_root = os.path.relpath(root, W)
    dirs[:] = [d for d in dirs if d not in skip_dirs and not (rel_root == 'tensor_cuda' and d == 'build')]
    for f in files:
        rel = os.path.normpath(os.path.join(rel_root, f))
        if rel == '.git' or f.endswith('.so') or f.endswith('.pyc'):
            continue
        a, b = os.path.join(P, rel), os.path.join(W, rel)
        nb = open(b, 'rb').read()
        if os.path.exists(a):
            na = open(a, 'rb').read()
            if na == nb:
                continue
            kind = 'M'
            al = na.decode('utf-8', 'replace').splitlines()
        else:
            kind = 'A'
            al = []
        bl = nb.decode('utf-8', 'replace').splitlines()
        ins = dele = 0
        for l in difflib.unified_diff(al, bl, lineterm='', n=0):
            if l.startswith('+') and not l.startswith('+++'): ins += 1
            elif l.startswith('-') and not l.startswith('---'): dele += 1
        rows.append((rel, kind, ins, dele))
rows.sort()
for rel, kind, ins, dele in rows:
    print(f' {kind} {rel:<70} | +{ins} -{dele}')
print(f' {len(rows)} files changed, {sum(r[2] for r in rows)} insertions(+), {sum(r[3] for r in rows)} deletions(-)')
