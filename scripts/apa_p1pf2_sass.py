"""APA-P1-PF2 G6 (object code): per-kernel SASS, Build PF vs Build F3, against artifacts/apa_p1pf2/EXPECTED.json.

Dumps come from P1-D's dump() (scripts/apa_p1d_sass.py, imported; cuobjdump -sass --gpu-architecture sm_89), with
its output directory pointed at artifacts/apa_p1pf2. Build F3 (tensor_fork_apa_wt_p1f2 artifacts/apa_p1f3/arms/F3,
sha256 2e557a23...) is re-dumped here; the fresh dump's sha256 is compared with F3's own dump (e1b49ad9...) as a
receipt check. Each Function block is compared as exact text after the registered mappings and nothing else
(EXPECTED.json non_code_sources_of_difference): path-derived names (<digits>_GLOBAL__N__<8 hex>_,
_GLOBAL__N__<8 hex>_, __nv_static_<digits>__<8 hex>_) and the build-directory suffix _cu_<8 hex>[_<n>], in names and
bodies; the mapped key must be unique per build (asserted). Kernel keys are demangled with c++filt (parameter list
dropped) to match the expected lists.
Prior art: NVIDIA cuobjdump; P1-D apa_p1d_sass_a1.py path-hash classes and the lead's lead_f3_sass_check.py widening
(Project-Tensor 2026), restated as one pattern list; F1's compare_builds() expectation bookkeeping (restated). No
prior art known to me beyond that. CPU only.
"""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import apa_p1d_sass as p1d  # noqa: E402
import apa_p1pf2_build as pfb  # noqa: E402

ART = ROOT / 'artifacts/apa_p1pf2'
F3_MODULE = ROOT.parent / 'tensor_fork_apa_wt_p1f2/artifacts/apa_p1f3/arms/F3/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so'
F3_SHA = '2e557a23eaf228512a2f1a2f0629401d91d02d05705087661be5c48849c0f3e9'
F3_OWN_DUMP_SHA = 'e1b49ad97118210b9949679b1af3d697a5e3043826a9a642751892a52f8524af'
PATTERNS = [(re.compile(r'([A-Za-z]*)\d+_GLOBAL__N__[0-9a-f]{8}_'), r'\1N_GLOBAL__N__H_'),
            (re.compile(r'_GLOBAL__N__[0-9a-f]{8}_'), '_GLOBAL__N__H_'),
            (re.compile(r'__nv_static_\d+__[0-9a-f]{8}_'), '__nv_static_N__H_'),
            (re.compile(r'_cu_[0-9a-f]{8}(?:_\d+)?'), '_cu_H')]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def norm(text):
    for pat, rep in PATTERNS:
        text = pat.sub(rep, text)
    return text


def read(path):
    blocks, names, key, lines = {}, {}, None, []
    for line in Path(path).open():
        m = re.search(r'Function : (\S+)', line)
        if m:
            key = norm(m[1])
            assert key not in names, ('non-bijective symbol map', key)
            names[key] = m[1]
            lines = []
        elif key and line.strip() == '..........':
            blocks[key] = norm(''.join(lines))
            key = None
        elif key:
            lines.append(line)
    return blocks, names


def demangle(names):
    p = subprocess.run(['c++filt'], input='\n'.join(names) + '\n', capture_output=True, text=True, check=True)
    return [re.sub(r'\(.*$', '', line) for line in p.stdout.splitlines()]


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else 'PF'
    rec = pfb.receipt(label)
    module = ROOT / rec['module']
    assert sha(module) == rec['module_sha256']
    assert sha(F3_MODULE) == F3_SHA
    p1d.ART = ART
    dumps = {}
    for name, mod in (('F3', F3_MODULE), (label, module)):
        path = ART / 'sass' / f'{name}.sass'
        dumps[name] = (dict(module=str(mod), module_sha256=sha(mod), dump=str(path.relative_to(ROOT)),
                            dump_sha256=sha(path), reused=True) if path.exists() else p1d.dump(name, mod))
    a, an = read(ROOT / dumps['F3']['dump'])
    b, bn = read(ROOT / dumps[label]['dump'])
    keys = sorted(set(a) | set(b))
    dem = dict(zip(keys, demangle([(an.get(k) or bn.get(k)) for k in keys])))
    expected = json.loads((ART / 'EXPECTED.json').read_text())['sass']
    exp = {kind: {n: why for why, ns in expected[f'expected_{kind}'].items() for n in ns}
           for kind in ('changed', 'removed', 'added')}
    rows, unexpected, seen = [], [], dict(changed=set(), removed=set(), added=set())
    for k in keys:
        if k in a and k in b:
            kind = 'equal' if a[k] == b[k] else 'changed'
        else:
            kind = 'added' if k in b else 'removed'
        d = dem[k]
        row = dict(kernel=d, kind=kind, F3_symbol=an.get(k), new_symbol=bn.get(k),
                   F3_sha256=hashlib.sha256(a[k].encode()).hexdigest() if k in a else None,
                   new_sha256=hashlib.sha256(b[k].encode()).hexdigest() if k in b else None)
        if kind != 'equal':
            if d in exp[kind]:
                seen[kind].add(d)
                row['expected_as'] = exp[kind][d]
            else:
                unexpected.append(dict(kernel=d, kind=kind))
        rows.append(row)
    missing = {kind: sorted(set(exp[kind]) - seen[kind]) for kind in exp}
    counts = {kind: sum(r['kind'] == kind for r in rows) for kind in ('equal', 'changed', 'removed', 'added')}
    paired = [r for r in rows if r['F3_symbol'] and r['new_symbol']]
    status = 'PASS' if not unexpected and not any(missing.values()) else 'FAIL'
    f3rec = json.loads(pfb.F3_BUILD.read_text())
    norm_reps = [[x.replace('tensor_fork_apa_wt_p1f2', 'tensor_fork_apa_wt_p1pf2') for x in r]
                 for r in f3rec['p1d_adapter_replacements']]
    result = dict(status=status, base='F3', new=label, method=__doc__.strip().splitlines()[0],
                  patterns=[[p.pattern, r] for p, r in PATTERNS],
                  expected_json=str((ART / 'EXPECTED.json').relative_to(ROOT)),
                  expected_json_sha256=sha(ART / 'EXPECTED.json'), dumps=dumps,
                  f3_fresh_dump_equals_f3_own_dump=dumps['F3']['dump_sha256'] == F3_OWN_DUMP_SHA,
                  flags_equal=rec['p1d_flags'] == f3rec['p1d_flags'],
                  adapter_replacements_equal_after_worktree_name=rec['p1d_adapter_replacements'] == norm_reps,
                  kernels=len(rows), counts=counts, unexpected_differences=unexpected, expected_but_not_seen=missing,
                  symbol_renames_among_paired=sum(r['F3_symbol'] != r['new_symbol'] for r in paired),
                  differing=[dict(kernel=r['kernel'], kind=r['kind'], expected=r.get('expected_as') is not None)
                             for r in rows if r['kind'] != 'equal'],
                  rows=rows)
    out = ART / f'SASS_COMPARISON_{label}.json'
    with out.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'dumps')}, indent=1)[:6000])
    print('sha256', sha(out))
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
