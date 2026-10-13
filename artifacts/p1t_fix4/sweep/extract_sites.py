"""P1T-FIX4 G2: extract every CUDA runtime / cuBLAS call site and kernel launch (<<<) on ADDED lines of the P1-T diff,
`git diff 9df5bef -- tensor_cuda tc_signal` (working tree = the FIX4 final code), plus the new untracked files there.
Comment-only lines are skipped; a token inside a string literal is skipped (error messages name calls); each remaining
token is one site: file, line (in the working-tree file), enclosing function (nearest preceding definition), call and
the source line. Output: sites.json and a count summary. The annotation (device, how established, return handling,
verdict) is joined by annotate.py. No prior art claimed (a source scanner).
"""
import json
import re
import subprocess
from pathlib import Path

W = Path('/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix4')


def git(*a):
    return subprocess.run(['git', '-C', str(W), *a], capture_output=True, text=True, check=True).stdout


CALL = re.compile(r'\b(cuda[A-Z]\w*|cublas[A-Z]\w*)\s*\(|<<<')
SRC_EXT = ('.cpp', '.cu', '.h', '.c', '.py')


def strip_strings(line):
    # remove "..." and '...' literals (no escaped-quote handling needed for these sources' messages)
    return re.sub(r'"(?:[^"\\]|\\.)*"', '""', re.sub(r"'(?:[^'\\]|\\.)*'", "''", line))


def added_lines():
    out = {}
    diff = git('diff', '-U0', '9df5bef', '--', 'tensor_cuda', 'tc_signal')
    cur, ln = None, 0
    for line in diff.splitlines():
        if line.startswith('+++ '):
            cur = line[6:] if line != '+++ /dev/null' else None
        elif line.startswith('@@'):
            ln = int(re.search(r'\+(\d+)', line).group(1))
        elif line.startswith('+') and cur:
            out.setdefault(cur, []).append((ln, line[1:]))
            ln += 1
    for f in git('ls-files', '--others', '--exclude-standard', '--', 'tensor_cuda', 'tc_signal').split():
        out[f] = list(enumerate(Path(W, f).read_text().splitlines(), 1))
    return {f: rows for f, rows in out.items() if f.endswith(SRC_EXT)}   # code only (README/docs mention calls)


FUNC = re.compile(r'^(?:[A-Za-z_][\w:<>,\*&\s~]*?\s+)?(~?[A-Za-z_][\w:~]*)\s*\([^;]*\)\s*(?:const\s*)?(?:noexcept\s*)?(?:override\s*)?\{?\s*$')
PYFUNC = re.compile(r'^\s*def\s+(\w+)\s*\(')


def enclosing(path, lineno):
    lines = Path(W, path).read_text().splitlines()
    for i in range(lineno - 1, -1, -1):
        s = lines[i]
        if path.endswith('.py'):
            m = PYFUNC.match(s)
            if m:
                return m.group(1)
            if s.startswith(('_', 'r"""', "r'''")) and '=' in s and not s.startswith(' '):
                return s.split('=')[0].strip() + ' (embedded script)'
            continue
        if s and not s[0].isspace() and not s.startswith(('//', '#', '}', '/*', ' *')):
            m = FUNC.match(s)
            if m and not s.startswith(('if ', 'for ', 'while ', 'return', 'switch')):
                return m.group(1)
        m = re.match(r'^\s{2,4}(?:explicit\s+)?(~?\w+)\s*\([^;]*\)\s*(?:noexcept\s*)?\{\s*$', s)
        if m and m.group(1) not in ('if', 'for', 'while', 'switch', 'catch'):
            return m.group(1)
        m = re.match(r'^\s*(?:void|bool|int|cudaError_t|static\s+\w+)\s+(\w+)\s*\(.*\)\s*(?:noexcept\s*)?\{', s)
        if m:
            return m.group(1)
    return '?'


sites = []
for f, rows in sorted(added_lines().items()):
    for ln, text in rows:
        st = text.strip()
        if not st or st.startswith(('//', '*', '/*', '#')) or (f.endswith('.py') and st.startswith('#')):
            continue
        code = strip_strings(text)
        code = re.sub(r'/\*.*?\*/', '', code)          # inline C block comments (the shims' annotations)
        if '//' in code and not f.endswith('.py'):
            code = code[:code.index('//')]
        for m in CALL.finditer(code):
            call = m.group(1) or '<<< kernel launch'
            if call.startswith('cudaError'):              # an enum value followed by "(", not a call
                continue
            definition = bool(re.match(r'\s*(?:int|void|cudaError_t)\s+' + re.escape(call) + r'\s*\(', code))
            sites.append({'file': f, 'line': ln, 'function': enclosing(f, ln), 'call': call, 'text': st[:200],
                          'kind': 'test' if '/tests/' in f else 'product', 'definition': definition})
json.dump(sites, open('sites.json', 'w'), indent=1)
from collections import Counter
print('sites', len(sites), Counter(s['kind'] for s in sites))
print(Counter((s['kind'], s['file']) for s in sites))
