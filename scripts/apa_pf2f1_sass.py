"""CPU G6 SASS, PF2 mappings only. Prior art: PF2 SASS parser (2026),
NVIDIA cuobjdump, exact Function-block comparison; inherited without extra normalization.
"""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import apa_pf2f1_common as C
import apa_pf2f1_sass_reference as R


def main():
    dumps = {}
    for label in ('PF', 'F1'):
        C.precheck(label)
        module = ROOT / C.build.receipt(label)['module']
        dest = C.ART / 'sass' / f'{label}.sass'
        dest.parent.mkdir(exist_ok=True)
        cmd = ['timeout', '300s', 'nice', '-n', '19', '/usr/local/cuda-12.6/bin/cuobjdump',
               '-sass', '--gpu-architecture', 'sm_89', str(module)]
        with dest.open('x') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, check=True)
        dumps[label] = dict(path=str(dest.relative_to(ROOT)), sha256=C.sha(dest),
                            module_sha256=C.sha(module), argv=cmd)
    a, an = R.read(ROOT / dumps['PF']['path'])
    b, bn = R.read(ROOT / dumps['F1']['path'])
    keys = sorted(set(a) | set(b))
    p = subprocess.run(['timeout', '30s', 'c++filt'], input='\n'.join(an.get(k, bn.get(k)) for k in keys)+'\n',
                       capture_output=True, text=True, check=True)
    dem = p.stdout.splitlines()
    rows = [dict(symbol=k, demangled=d, kind='added' if k not in a else 'removed' if k not in b else
                 'equal' if a[k]==b[k] else 'changed') for k, d in zip(keys, dem)]
    changed = [r for r in rows if r['kind']!='equal']
    ok = bool(a) and set(a)==set(b) and bool(changed) and all(
        r['kind']=='changed' and 'tc::spq_pf::prefill<' in r['demangled'] for r in changed)
    result = dict(status='PASS' if ok else 'FAIL', kernels=len(rows),
                  counts={kind:sum(r['kind']==kind for r in rows) for kind in ('equal','changed','added','removed')},
                  mappings=[[p.pattern, r] for p,r in R.PATTERNS], dumps=dumps,
                  only_prefill_changes=ok, rows=rows)
    C.save(C.ART / 'SASS_COMPARISON.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','dumps','mappings')}, indent=2))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
