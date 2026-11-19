"""CPU arm, arithmetic and code-object gates.

Prior art: IEEE 754-2008 quiet NaNs and PF2-F1 (Project-Tensor, 2026).
The CPU extracts each real load assignment for a host bit-pattern check.
SASS reuses PF2-F1's exact blocks and path-only normalization.
"""
import argparse
from collections import Counter
import ctypes
import difflib
import hashlib
import itertools
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import apa_pf2f2_build as B
import apa_pf2f2_sass_reference as S

LABELS = ('F1', 'F2', *B.MUTANTS)


def arm_dir(label):
    return B.ART / 'arms' / label


def precheck(label):
    rec = B.receipt(label)
    data = B.read(arm_dir(label) / 'ARM.json')
    assert B.sha(B.ROOT / data['module']) == B.sha(B.ROOT / rec['module']) == rec['module_sha256']
    for filename, digest in data['python_files'].items():
        assert B.sha(B.ROOT / filename) == digest, filename
    if label != 'F1':
        assert B.input_map(B.receipt('F1'), Path(rec['source_root'])) == rec['source_sha256_after']
    return data


def stage(label):
    rec = B.receipt(label)
    target = arm_dir(label) / 'tensor_cuda'
    target.mkdir(parents=True)
    files = {}
    root = B.ROOT / 'tensor_cuda/tensor_cuda'
    for source in sorted(root.rglob('*.py')):
        name = source.relative_to(root)
        assert B.sha(source) == rec['source_sha256_after'][f'tensor_cuda/tensor_cuda/{name}']
        dest = target / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        files[str(dest.relative_to(B.ROOT))] = B.sha(dest)
    module = B.ROOT / rec['module']
    final = target / module.name
    shutil.copyfile(module, final)
    B.save(target.parent / 'ARM.json', dict(label=label, module=str(final.relative_to(B.ROOT)),
                                           module_sha256=rec['module_sha256'], python_files=files))
    precheck(label)


def load_arm(label):
    import importlib
    data = precheck(label)
    sys.path.insert(0, str(arm_dir(label)))
    tc = importlib.import_module('tensor_cuda')
    assert Path(tc.__file__).resolve().parent == arm_dir(label) / 'tensor_cuda'
    assert B.sha(tc._C.__file__) == data['module_sha256']
    return tc


def guard_expressions(text):
    shared = re.search(r'^   n=(isfinite\(n\).+);$', text, re.MULTILINE)
    bulk = re.search(r'^      norms\[tid\]=(isfinite\(norm\).+);$', text, re.MULTILINE)
    return (shared[1] if shared else 'n', bulk[1] if bulk else 'norm')


def isolated_mutation_diff(label):
    path = Path(B.receipt(label)['source_root']) / B.HEADER
    before = (B.ROOT / B.HEADER).read_text()
    old, new = B.REVERT[label]
    assert before.count(old) == 1
    after = path.read_text()
    assert after == before.replace(old, new, 1)
    name = f'{label}.diff'
    with (B.ART / name).open('x') as file:
        file.writelines(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                            fromfile=B.HEADER, tofile=B.HEADER))
    return name


def host_guard():
    tmp = Path(os.environ['TMPDIR']) / 'guard'
    tmp.mkdir()
    labels = ('F2', *B.MUTANTS)
    raw = (0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000,
           0xbf800000, 0x3f800000, 0x7f7fffff, 0xff7fffff,
           0x7f800000, 0xff800000, 0x7fc00123, 0xffc00000, 0x7f800001)
    src = tmp / 'guard.cpp'
    signatures = {}
    with src.open('x') as f:
        f.write("""#include <cmath>
#include <cstdint>
#include <cstring>
using std::isfinite;
static float __int_as_float(uint32_t b) {float n; std::memcpy(&n,&b,4); return n;}
""")
        for i, label in enumerate(labels):
            path = Path(B.receipt(label)['source_root']) / B.HEADER
            shared, bulk = guard_expressions(path.read_text())
            for j, (arg, expr) in enumerate((('n', shared), ('norm', bulk))):
                symbol = f'guard_{i}_{j}'
                signatures[label, j] = (symbol, expr)
                f.write(f'extern "C" uint32_t {symbol}(uint32_t b) {{float {arg}=__int_as_float(b); '
                        f'float result={expr}; uint32_t bits; std::memcpy(&bits,&result,4); return bits;}}\n')
    lib = tmp / 'guard.so'
    argv = ['timeout', '60s', 'nice', '-n', '19', 'c++', '-std=c++17', '-O2',
            '-shared', '-fPIC', str(src), '-o', str(lib)]
    subprocess.run(argv, check=True)
    loaded = ctypes.CDLL(str(lib))
    rows = []
    for (label, j), (symbol, expr) in signatures.items():
        name = ('bulk_shared', 'bulk')[j]
        reverted = B.MUTANTS.get(label) == name
        guard = getattr(loaded, symbol)
        guard.argtypes, guard.restype = (ctypes.c_uint32,), ctypes.c_uint32
        for v in raw:
            want = 0x7fc00000 if v & 0x7f800000 == 0x7f800000 and not reverted else v
            got = guard(v)
            rows.append(dict(label=label, kernel=name, expr=expr, raw=f'{v:08x}', want=f'{want:08x}', got=f'{got:08x}',
                             ok=got == want))
    good = all(r['ok'] for r in rows)
    data = dict(status='PASS' if good else 'FAIL', argv=argv, checks=len(rows), rows=rows)
    B.save(B.ART / 'HOST_GUARD.json', data)
    assert good
    print('HOST_GUARD', len(rows), 'PASS')


def sass():
    dumps = {}
    for label in LABELS:
        mod = B.ROOT / precheck(label)['module']
        path = B.ART / 'sass' / f'{label}.sass'
        path.parent.mkdir(exist_ok=True)
        argv = ['timeout','240s','nice','-n','19','/usr/local/cuda-12.6/bin/cuobjdump',
                '-sass','--gpu-architecture','sm_89',str(mod)]
        with path.open('x') as stream:
            subprocess.run(argv, stdout=stream, check=True)
        dumps[label] = dict(argv=argv, path=str(path.relative_to(B.ROOT)), module_sha256=B.sha(mod),
                            sass_sha256=B.sha(path))

    bulk = {f'void tc::spq1::bulk<{c}, {w}>' for c,w in
            ((64,'false'),(128,'true'),(256,'true'),(512,'true'))}
    shared = {f'void tc::spq1::bulk_shared<{d}, {g}, {t}>' for d,g,t in
              itertools.product((64,128),(4,8),('false','true'))}
    comparisons = []
    for base, new, expected in (('F1','F2',bulk | shared), ('F2',B.BULK,bulk), ('F2',B.SHARED,shared)):
        before, before_names = S.read(B.ROOT / dumps[base]['path'])
        after, after_names = S.read(B.ROOT / dumps[new]['path'])
        keys = sorted(set(before) | set(after))
        names = S.demangle([before_names.get(k) or after_names[k] for k in keys])
        assert len(keys) == len(names) and len(keys) > 0
        rows = []
        for k, demangled in zip(keys, names):
            change = ('added' if k not in before else 'removed' if k not in after else
                      'same' if before[k] == after[k] else 'changed')
            rows.append(dict(symbol=k, demangled=demangled, change=change,
                             before=hashlib.sha256(before[k].encode()).hexdigest() if k in before else None,
                             after=hashlib.sha256(after[k].encode()).hexdigest() if k in after else None))
        counts = dict(Counter(r['change'] for r in rows))
        changed = {r['demangled'] for r in rows if r['change'] == 'changed'}
        missing, extra = expected - changed, changed - expected
        good = not (missing or extra or counts.get('added') or counts.get('removed'))
        comparisons.append(dict(base=base, new=new, status='PASS' if good else 'FAIL', counts=counts,
                                expected=sorted(expected), missing=sorted(missing), extra=sorted(extra), rows=rows))
        print('SASS', base, new, counts, 'missing', sorted(missing),'extra',sorted(extra), flush=True)
    B.save(B.ART / 'SASS_COMPARISON.json', dict(dumps=dumps, comparisons=comparisons))
    assert all(r['status'] == 'PASS' for r in comparisons)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('stage','import','host','sass'))
    p.add_argument('label', nargs='?', choices=LABELS)
    a = p.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    if a.mode == 'stage':
        for label in LABELS:
            stage(label)
        for label in B.MUTANTS:
            isolated_mutation_diff(label)
    elif a.mode == 'import':
        tc = load_arm(a.label)
        assert all(hasattr(tc._spo2, name) for name in ('packed_run','packed_prefill'))
        assert B.sha(tc._C.__file__) == B.receipt(a.label)['module_sha256']
        print('CPU_IMPORT', a.label, tc._C.__file__, B.sha(tc._C.__file__))
    elif a.mode == 'host':
        host_guard()
    else:
        sass()


if __name__ == '__main__':
    main()
