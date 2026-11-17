"""APA-P1-D S3: per-kernel SASS comparison, Build B vs Build A and Build A vs A1.

Prior art: NVIDIA cuobjdump; P1 A1 SASS.py/SASS_COMPARE.py (Project-Tensor,
2026): every Function block compared as exact text (instructions, encodings,
control words); only nvcc's per-build anonymous-namespace symbol suffix
(`_cu_<8 hex>_<n>`) is mapped, bijectively, as A1 did. No instruction is
normalized. CPU only.
"""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/apa_p1d'
A1 = Path('/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod/artifacts/apa_p1_a1')
CUOBJDUMP = '/usr/local/cuda-12.6/bin/cuobjdump'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    blocks, names, key, lines = {}, {}, None, []
    for line in path.open():
        m = re.search(r'Function : (\S+)', line)
        if m:
            name = m[1]
            key = re.sub(r'_cu_[0-9a-f]{8}_[0-9]+', '_cu_BUILD', name)
            assert key not in blocks, ('non-bijective symbol map', key)
            names[key] = name
            lines = []
        elif key and line.strip() == '..........':
            blocks[key] = ''.join(lines)
            key = None
        elif key:
            lines.append(line)
    return blocks, names


def dump(label, module):
    out = ART / 'sass' / f'{label}.sass'
    out.parent.mkdir(exist_ok=True)
    cmd = [CUOBJDUMP, '-sass', '--gpu-architecture', 'sm_89', str(module)]
    with out.open('x') as f:
        subprocess.run(['timeout', '1500s', *cmd], stdout=f, stderr=subprocess.PIPE, check=True)
    return dict(module=str(module), module_sha256=sha(module), dump=str(out.relative_to(ROOT)),
                dump_sha256=sha(out), argv=cmd)


def compare(x, y, xl, yl):
    a, an = read(ROOT / x['dump'])
    b, bn = read(ROOT / y['dump'])
    rows = []
    for k in sorted(set(a) | set(b)):
        rows.append({'kernel': k, f'{xl}_symbol': an.get(k), f'{yl}_symbol': bn.get(k),
                     'exact_body_equal': a.get(k) == b.get(k),
                     f'{xl}_sha256': hashlib.sha256(a[k].encode()).hexdigest() if k in a else None,
                     f'{yl}_sha256': hashlib.sha256(b[k].encode()).hexdigest() if k in b else None})
    return dict(status='PASS' if a and a == b else 'FAIL', kernels=len(rows),
                differences=sum(not r['exact_body_equal'] for r in rows),
                differing=[r['kernel'] for r in rows if not r['exact_body_equal']],
                symbol_suffix_renames=sum(r[f'{xl}_symbol'] != r[f'{yl}_symbol'] for r in rows),
                rows=rows)


def main():
    a = json.loads((ART / 'build_A_r2/BUILD.json').read_text())
    b = json.loads((ART / 'build_B/BUILD.json').read_text())
    a1 = json.loads((A1 / 'BUILD.json').read_text())
    a1_module = A1.parent.parent / a1['module']
    assert sha(a1_module) == a1['module_sha256']
    dumps = {'A': dump('A', ROOT / a['module']), 'B': dump('B', ROOT / b['module']), 'A1': dump('A1', a1_module)}
    result = dict(method='cuobjdump -sass sm_89; every Function block exact text; only the _cu_<hex>_<n> '
                         'anonymous-namespace suffix mapped (bijective)', dumps=dumps,
                  B_vs_A=compare(dumps['A'], dumps['B'], 'A', 'B'),
                  A_vs_A1=compare(dumps['A1'], dumps['A'], 'A1', 'A'))
    path = ART / 'SASS_COMPARISON.json'
    with path.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    for key in ('B_vs_A', 'A_vs_A1'):
        r = result[key]
        print(key, r['status'], r['kernels'], 'kernels', r['differences'], 'differences', r['differing'][:20])
    sys.exit(0 if result['B_vs_A']['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
