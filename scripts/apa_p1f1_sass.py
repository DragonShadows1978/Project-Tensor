"""APA-P1-F1 S3: per-kernel SASS, Build F vs Build B, against artifacts/apa_p1f1/EXPECTED.json.

P1-D's comparison (scripts/apa_p1d_sass.py) is imported: its dump() (cuobjdump
-sass sm_89) and its compare() (every Function block compared as exact text).
Only its output directory is pointed at artifacts/apa_p1f1. Kernel keys are
demangled with c++filt to match the expected list (names without the parameter
list).

APA-P1-F2 R7 (P1R2-F08), NIT, fixed once:
- the result records the expectation file it actually compared against
  (`expected_active`: path and sha256) next to the original registration
  (`expected_json_sha256`, unchanged meaning);
- `symbol_suffix_renames` now counts only paired kernels present in both builds
  whose symbol changed; kernels present in one build only are counted apart
  (`symbol_only_in_base`, `symbol_only_in_new`), so an added function is no
  longer counted as a rename;
- symbol keys map nvcc's per-build `_cu_<8 hex>` suffix with an optional
  `_<n>` (bijective; read_blocks below), the only normalization;
- compare_builds() takes the base/new builds and the artifacts directory as
  arguments, so APA-P1-F2 runs this same comparison for Build F2 against Build
  F (scripts/apa_p1f2_sass.py). F1's filed receipts are not rewritten.
Prior art: NVIDIA cuobjdump; P1 A1 / P1-D SASS comparison (Project-Tensor 2026),
imported. No prior art known to me beyond that. CPU only.
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

ART = ROOT / 'artifacts/apa_p1f1'
BUILD_B_SHA = '8728e67e26027dbafe8a61592decb38e68171fa7b5634239ce0af7897c4e8c4e'
SUFFIX = re.compile(r'_cu_[0-9a-f]{8}(?:_[0-9]+)?')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def demangle(names):
    p = subprocess.run(['c++filt'], input='\n'.join(names) + '\n', capture_output=True, text=True, check=True)
    return [re.sub(r'\(.*$', '', line) for line in p.stdout.splitlines()]


def read_blocks(path):
    """Function blocks keyed by symbol with `_cu_<8 hex>[_<n>]` mapped to `_cu_BUILD` (R7: optional `_<n>`)."""
    blocks, names, key, lines = {}, {}, None, []
    for line in Path(path).open():
        m = re.search(r'Function : (\S+)', line)
        if m:
            name = m[1]
            key = SUFFIX.sub('_cu_BUILD', name)
            assert key not in blocks and key not in names, ('non-bijective symbol map', key)
            names[key] = name
            lines = []
        elif key and line.strip() == '..........':
            blocks[key] = ''.join(lines)
            key = None
        elif key:
            lines.append(line)
    return blocks, names


def compare_dumps(x, y, xl, yl, root=ROOT):
    """Exact text per Function block (P1-D's compare(), with read_blocks and R7's split counts)."""
    a, an = read_blocks(Path(root) / x['dump'])
    b, bn = read_blocks(Path(root) / y['dump'])
    rows = []
    for k in sorted(set(a) | set(b)):
        rows.append({'kernel': k, f'{xl}_symbol': an.get(k), f'{yl}_symbol': bn.get(k),
                     'exact_body_equal': a.get(k) == b.get(k),
                     f'{xl}_sha256': hashlib.sha256(a[k].encode()).hexdigest() if k in a else None,
                     f'{yl}_sha256': hashlib.sha256(b[k].encode()).hexdigest() if k in b else None})
    paired = [r for r in rows if r[f'{xl}_symbol'] is not None and r[f'{yl}_symbol'] is not None]
    return dict(kernels=len(rows), rows=rows,
                symbol_suffix_renames=sum(r[f'{xl}_symbol'] != r[f'{yl}_symbol'] for r in paired),
                symbol_only_in_base=sum(r[f'{yl}_symbol'] is None for r in rows),
                symbol_only_in_new=sum(r[f'{xl}_symbol'] is None for r in rows))


def compare_builds(base_label, base_module, base_sha, new_label, new_module, new_sha, art, expected_path,
                   override_path=None, out_name=None, root=ROOT):
    """Compare new against base per kernel and check every difference against the expected lists."""
    art = Path(art)
    assert sha(base_module) == base_sha, f'{base_label} module sha mismatch'
    assert sha(new_module) == new_sha, f'{new_label} module sha mismatch'
    registered = json.loads(Path(expected_path).read_text())['sass']
    expected = registered
    active = Path(expected_path)
    if override_path is not None and Path(override_path).exists():
        # Attempt-specific registration (written before that build): same changed
        # and added lists, its own removed list.
        reg = json.loads(Path(override_path).read_text())
        assert isinstance(reg['expected_removed'], list)
        expected = dict(registered, expected_removed={'registered in ' + Path(override_path).name:
                                                      reg['expected_removed']})
        active = Path(override_path)
    p1d.ART = art   # dumps land in <art>/sass
    sass_dir = art / 'sass'
    dumps = {}
    for name, module in ((base_label, base_module), (new_label, new_module)):
        if (sass_dir / f'{name}.sass').exists():
            dumps[name] = dict(module=str(module), module_sha256=sha(module),
                               dump=str((sass_dir / f'{name}.sass').relative_to(root)),
                               dump_sha256=sha(sass_dir / f'{name}.sass'), reused=True)
        else:
            dumps[name] = p1d.dump(name, module)
    cmp = compare_dumps(dumps[base_label], dumps[new_label], base_label, new_label, root)
    keys = [r['kernel'] for r in cmp['rows']]
    # Map each comparison key (mapped mangled name) to its demangled name.
    dem = dict(zip(keys, demangle([k.replace('_cu_BUILD', '_cu_00000000_0') for k in keys])))
    exp_changed = {n: why for why, names in expected['expected_changed'].items() for n in names}
    exp_removed = {n: why for why, names in expected['expected_removed'].items() for n in names}
    exp_added = {n: why for why, names in expected['expected_added'].items() for n in names}
    rows, unexpected = [], []
    seen = dict(changed=set(), removed=set(), added=set())
    for r in cmp['rows']:
        d = dem[r['kernel']]
        if r['exact_body_equal']:
            kind = 'equal'
        elif r[f'{base_label}_sha256'] is None:
            kind = 'added'
        elif r[f'{new_label}_sha256'] is None:
            kind = 'removed'
        else:
            kind = 'changed'
        r['demangled'], r['kind'] = d, kind
        if kind != 'equal':
            table = dict(changed=exp_changed, removed=exp_removed, added=exp_added)[kind]
            if d in table:
                r['expected_as'] = table[d]
                seen[kind].add(d)
            else:
                unexpected.append(dict(kernel=d, kind=kind))
        rows.append(r)
    missing = dict(changed=sorted(set(exp_changed) - seen['changed']),
                   removed=sorted(set(exp_removed) - seen['removed']),
                   added=sorted(set(exp_added) - seen['added']))
    status = 'PASS' if not unexpected and not any(missing.values()) else 'FAIL'
    result = dict(status=status, method=p1d.__doc__.strip().splitlines()[0] +
                  ' (imported dump; exact Function-block text; _cu_<8 hex>[_<n>] mapped, bijective)',
                  base=base_label, new=new_label,
                  expected_json_sha256=sha(expected_path), expected_json=str(Path(expected_path).relative_to(root)),
                  expected_active=dict(path=str(active.relative_to(root)), sha256=sha(active)),
                  dumps=dumps, kernels=len(rows),
                  counts={k: sum(r['kind'] == k for r in rows) for k in ('equal', 'changed', 'removed', 'added')},
                  unexpected_differences=unexpected, expected_but_not_seen=missing,
                  symbol_suffix_renames=cmp['symbol_suffix_renames'],
                  symbol_only_in_base=cmp['symbol_only_in_base'], symbol_only_in_new=cmp['symbol_only_in_new'],
                  differing=[dict(kernel=r['demangled'], kind=r['kind'], expected_as=r.get('expected_as'))
                             for r in rows if r['kind'] != 'equal'],
                  rows=rows)
    out = art / (out_name or f'SASS_COMPARISON_{new_label}.json')
    with out.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(status, result['counts'], 'renames', result['symbol_suffix_renames'], 'unexpected', unexpected,
          'missing', missing)
    return result


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else 'F'
    b_module = ROOT / 'artifacts/apa_p1d/build_B/lib/_tensor_cuda.cpython-312-x86_64-linux-gnu.so'
    rec = json.loads((ART / f'build_{label}/BUILD.json').read_text())
    result = compare_builds('B', b_module, BUILD_B_SHA, label, ROOT / rec['module'], rec['module_sha256'], ART,
                            ART / 'EXPECTED.json', ART / f'EXPECTED_SASS_{label}.json')
    sys.exit(0 if result['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
