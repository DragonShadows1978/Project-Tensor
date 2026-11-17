"""APA-P1-D S3 (closing S3's open item): Build A vs A1's build from the retained dumps.

The two builds come from different source paths (this worktree vs
tensor_fork_apa_wt_apa2pprod), so nvcc's path-derived hashes in internal symbol
names differ (`__nv_static_<n>__<8 hex>_`, `_GLOBAL__N__<8 hex>_`,
`_cu_<8 hex>`). Those hashes are mapped, in names and in body text (call
targets), and nothing else: instructions, encodings and control words are
compared exactly. Prior art: P1 A1 SASS_COMPARE.py symbol correspondence (2026);
ours: the path-hash classes above. CPU only.
"""
from pathlib import Path
import hashlib
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/apa_p1d'
# Attempt 2 (repair 1): the per-build counter after the hash (`_cu_<hex>_<n>`)
# and the mangled length prefixes that change with it are mapped too.
PATTERNS = [(r'__nv_static_\d+__[0-9a-f]{8}_', r'__nv_static_N__H_'),
            (r'tc\d+_GLOBAL__N__[0-9a-f]{8}_', r'tcN_GLOBAL__N__H_'),
            (r'_cu_[0-9a-f]{8}_\d+', r'_cu_H_N'),
            (r'_cu_[0-9a-f]{8}', r'_cu_H')]


def norm(text):
    for pat, rep in PATTERNS:
        text = re.sub(pat, rep, text)
    return text


def read(path):
    blocks, names, key, lines = {}, {}, None, []
    for line in path.open():
        m = re.search(r'Function : (\S+)', line)
        if m:
            key = norm(m[1])
            assert key not in blocks, ('non-bijective symbol map', key)
            names[key] = m[1]
            lines = []
        elif key and line.strip() == '..........':
            blocks[key] = norm(''.join(lines))
            key = None
        elif key:
            lines.append(line)
    return blocks, names


def main():
    a, an = read(ART / 'sass/A.sass')
    b, bn = read(ART / 'sass/A1.sass')
    rows = []
    for k in sorted(set(a) | set(b)):
        rows.append(dict(kernel=k, A_symbol=an.get(k), A1_symbol=bn.get(k), exact_body_equal=a.get(k) == b.get(k),
                         only_in=None if (k in a and k in b) else ('A' if k in a else 'A1')))
    differing = [r for r in rows if not r['exact_body_equal']]
    changed = subprocess.run(['git', '-C', str(ROOT), 'diff', '--stat', 'f7e6f49b', 'e2fb85a1', '--',
                              'tensor_cuda/src', 'tensor_cuda/include', 'tensor_cuda/CMakeLists.txt'],
                             capture_output=True, text=True).stdout
    result = dict(method='exact body text after mapping only path-derived symbol hashes', patterns=PATTERNS,
                  kernels=len(rows), differences=len(differing),
                  differing=[dict(kernel=r['kernel'], only_in=r['only_in']) for r in differing],
                  source_changes_f7e6f49b_to_e2fb85a1=changed, rows=rows)
    with (ART / 'SASS_A_VS_A1_r2.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(len(rows), 'kernels', len(differing), 'differ')
    for r in differing:
        print(r['only_in'], r['kernel'])


if __name__ == '__main__':
    main()
