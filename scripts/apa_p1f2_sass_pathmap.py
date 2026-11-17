"""APA-P1-F2 SUPPLEMENTARY (not a gate): Build F2 vs Build F with the path-derived symbol hashes mapped.

G6 as registered (scripts/apa_p1f2_sass.py: only the `_cu_<8 hex>[_<n>]` suffix mapped) FAILS because
Build F was compiled from tensor_fork_apa_wt_p1d and Build F2 from tensor_fork_apa_wt_p1f2: nvcc's
path-derived hashes (`__nv_static_<n>__<8 hex>_`, `tc<n>_GLOBAL__N__<8 hex>_`) differ for every kernel
with internal linkage. This script re-reads the SAME two dumps with P1-D's A1 reader
(scripts/apa_p1d_sass_a1.py read(), imported unchanged), which maps those path hashes in names AND in
body text (call targets) and nothing else, and lists what still differs. It is evidence for the lead's
ruling on G6, never a re-registration (L35).
Prior art: P1 A1 SASS_COMPARE.py symbol correspondence / P1-D apa_p1d_sass_a1.py (Project-Tensor 2026),
imported. CPU only.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import apa_p1d_sass_a1 as a1  # noqa: E402

ART = ROOT / 'artifacts/apa_p1f2'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    reg = json.loads((ART / 'SASS_COMPARISON_F2.json').read_text())
    fdump, f2dump = ROOT / reg['dumps']['F']['dump'], ROOT / reg['dumps']['F2']['dump']
    assert sha(fdump) == reg['dumps']['F']['dump_sha256'] and sha(f2dump) == reg['dumps']['F2']['dump_sha256']
    a, an = a1.read(fdump)
    b, bn = a1.read(f2dump)
    keys = sorted(set(a) | set(b))
    differing = [k for k in keys if a.get(k) != b.get(k)]
    dem = subprocess.run(['c++filt'], input='\n'.join(bn.get(k) or an.get(k) for k in differing) + '\n',
                         capture_output=True, text=True, check=True).stdout.splitlines()
    rows = [dict(kernel=d.split('(')[0], in_F=k in a, in_F2=k in b) for k, d in zip(differing, dem)]
    expected = json.loads((ART / 'EXPECTED.json').read_text())['sass']['expected_changed']
    exp = sorted(n for names in expected.values() for n in names)
    result = dict(supplementary=True, gate=False,
                  method='P1-D apa_p1d_sass_a1.read (imported): path-derived hashes mapped in names and bodies',
                  dumps=dict(F=reg['dumps']['F'], F2=reg['dumps']['F2']),
                  kernels=len(keys), only_in_F=sum(k not in b for k in keys), only_in_F2=sum(k not in a for k in keys),
                  differing=rows, differing_equals_registered_changed=sorted(r['kernel'] for r in rows) == exp,
                  registered_changed=exp)
    out = ART / 'SASS_PATHMAP_SUPPLEMENT.json'
    with out.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'dumps'}, indent=1))


if __name__ == '__main__':
    main()
