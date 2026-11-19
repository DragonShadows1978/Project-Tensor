"""CPU arm, arithmetic and code-object gates.

Prior art: IEEE 754-2008 quiet NaNs and PF2-F1/F2 (Project-Tensor,
2026). Reuse PF2-F2's host extraction and normalized SASS comparison.
"""
from collections import Counter
import ctypes
import hashlib
import itertools
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import apa_pf2f3_build as B
import apa_pf2f3_sass_reference as S

LABELS = ('F2', 'F3', *B.MUTANTS)


def arm_dir(label):
    return B.ART / 'arms' / label


def precheck(label):
    row = B.read(arm_dir(label) / 'ARM.json')
    assert row['label'] == label
    assert B.sha(B.ROOT / row['module']) == row['module_sha256']
    for filename, digest in row['python_files'].items():
        assert B.sha(B.ROOT / filename) == digest, filename
    return row


def stage(label):
    receipt = B.receipt(label)
    target = arm_dir(label) / 'tensor_cuda'
    target.mkdir(parents=True)
    files = {}
    source = (B.F2 / 'arms/F2/tensor_cuda' if label == 'F2'
              else B.ROOT / 'tensor_cuda/tensor_cuda')
    for path in sorted(source.rglob('*.py')):
        name = path.relative_to(source)
        want = receipt['source_sha256_after'][f'tensor_cuda/tensor_cuda/{name}']
        final = target / name
        final.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, final)
        assert B.sha(final) == want
        files[str(final.relative_to(B.ROOT))] = want
    expected = {name.removeprefix('tensor_cuda/tensor_cuda/') for name in receipt['source_sha256_after']
                if name.startswith('tensor_cuda/tensor_cuda/') and name.endswith('.py')}
    assert {str((B.ROOT / name).relative_to(target)) for name in files} == expected
    module = B.ROOT / receipt['module']
    final = target / module.name
    shutil.copyfile(module, final)
    B.save(target.parent / 'ARM.json', dict(label=label, module=str(final.relative_to(B.ROOT)),
                                           module_sha256=receipt['module_sha256'], python_files=files))
    precheck(label)


def load_arm(label):
    import importlib

    row = precheck(label)
    sys.path.insert(0, str(arm_dir(label)))
    tc = importlib.import_module('tensor_cuda')
    assert Path(tc.__file__).resolve().parent == arm_dir(label) / 'tensor_cuda'
    assert Path(tc._C.__file__).resolve() == B.ROOT / row['module']
    assert B.sha(tc._C.__file__) == row['module_sha256']
    return tc


def source_file(label, name):
    root = Path(B.receipt(label)['source_root'])
    return (root / name).read_text()


def guard_expressions(label):
    pre = source_file(label, B.PRE)
    seq, = re.findall(r'const float norm = kn\[jk\];\n          x = __fmul_rn\(x, (.*)\);$', pre, re.MULTILINE)
    lane, = re.findall(r'^      bulk = __fmul_rn\(__fmul_rn\(__shfl_sync\(FULL, x, 0\), (.*)\), scale\);$',
                       pre, re.MULTILINE)
    values = [('prefill_seq', seq, '', None), ('prefill_lane', lane, '', None)]
    for name in B.TC:
        text = source_file(label, name)
        hook, default = re.findall(r'const float norm=(scales\[.*?\]);', text)
        expr, = re.findall(r'^        ks\[t\]\[i\]=(.*?);$', text, re.MULTILINE)
        key = Path(name).stem
        values += [(f'{key}_{branch}', expr, load, index)
                   for branch, load, index in (('hook', hook, 3), ('default', default, 2))]
    return values


def host_guard():
    scratch = Path(tempfile.mkdtemp(prefix='guard_', dir=os.environ['TMPDIR']))
    path = scratch / 'guard.cpp'
    lib = scratch / 'guard.so'
    patterns = (0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000,
                0xbf800000, 0x3f800000, 0x7f7fffff, 0xff7fffff,
                0x7f800000, 0xff800000, 0x7fc00123, 0xffc00000, 0x7f800001)
    signatures = {}
    with path.open('x') as stream:
        stream.write("""#include <cmath>
#include <cstdint>
#include <cstring>
using std::isfinite;
static float __int_as_float(uint32_t b) {float n; std::memcpy(&n,&b,4); return n;}
""")
        for i, label in enumerate(LABELS[1:]):
            for j, (site, expr, load, index) in enumerate(guard_expressions(label)):
                name = f'guard_{i}_{j}'
                signatures[label, site] = (name, expr, load)
                stream.write(f'extern "C" uint32_t {name}(uint32_t b) {{\n')
                if index is not None:
                    stream.write('int bh=1, capacity=2, start=0, j=0; float scales[4]={1,1,1,1};\n')
                    stream.write(f'scales[{index}]=__int_as_float(b); float norm={load};\n')
                else:
                    stream.write('float norm=__int_as_float(b);\n')
                stream.write(f'float result={expr}; uint32_t bits; std::memcpy(&bits,&result,4); return bits;}}\n')
    argv = ['timeout', '60s', 'nice', '-n', '19', 'c++', '-std=c++17', '-O2', '-shared', '-fPIC',
            str(path), '-o', str(lib)]
    subprocess.run(argv, check=True)
    loaded = ctypes.CDLL(str(lib))
    rows = []
    for (label, site), (symbol, expr, load) in signatures.items():
        prefill = site.startswith('prefill')
        guard = getattr(loaded, symbol)
        guard.argtypes, guard.restype = (ctypes.c_uint32,), ctypes.c_uint32
        for bits in patterns:
            convert = bits & 0x7f800000 == 0x7f800000
            if (prefill and label == B.MUTANTS[0]) or (not prefill and label == B.MUTANTS[2]):
                convert = False
            elif prefill and label == B.MUTANTS[1]:
                convert = bits == 0x7f800000
            want = 0x7fc00000 if convert else bits
            got = guard(bits)
            rows.append(dict(arm=label, site=site, load=load, expression=expr, raw=f'{bits:08x}',
                             wanted=f'{want:08x}', actual=f'{got:08x}', ok=got == want))
    good = len(rows) == 360 and all(row['ok'] for row in rows)
    B.save(B.ART / 'HOST_GUARD.json', dict(status='PASS' if good else 'FAIL', argv=argv, rows=rows))
    print('HOST_GUARD', len(rows), 'PASS' if good else 'FAIL')
    assert good


def sass():
    scratch = Path(os.environ['TMPDIR']) / 'sass'
    scratch.mkdir()
    dumps = {}
    for label in LABELS:
        arm = precheck(label)
        path = scratch / f'{label}.sass'
        argv = ['timeout', '240s', 'nice', '-n', '19', '/usr/local/cuda-12.6/bin/cuobjdump',
                '-sass', '--gpu-architecture', 'sm_89', str(B.ROOT / arm['module'])]
        with path.open('x') as stream:
            subprocess.run(argv, stdout=stream, check=True)
        dumps[label] = dict(argv=argv, path=str(path), module_sha256=arm['module_sha256'],
                            sass_sha256=B.sha(path))

    tc = {f'void tc::spq4::bulk_mma<{d}>' for d in (64, 128)}
    tc |= {f'void tc::spq6::bulk_mma_tq<{d}, {v}>' for d, v in itertools.product((64, 128), range(4))}
    pf = {f'void tc::spq_pf::prefill<{dt}, {cap}, {diag}, {seq}>' for dt, cap, diag, seq in
          itertools.product(('float', '__nv_bfloat16'), (64, 128, 256, 512), ('false', 'true'), ('false', 'true'))}
    comparisons = []
    for base, new, expected in (('F2', 'F3', tc), ('F3', B.MUTANTS[0], pf),
                                ('F3', B.MUTANTS[1], pf), ('F3', B.MUTANTS[2], tc)):
        before, before_names = S.read(dumps[base]['path'])
        after, after_names = S.read(dumps[new]['path'])
        keys = sorted(set(before) | set(after))
        names = S.demangle([before_names.get(k) or after_names[k] for k in keys])
        assert len(keys) == len(names) > 0
        rows = []
        for key, demangled in zip(keys, names):
            change = ('added' if key not in before else 'removed' if key not in after else
                      'same' if before[key] == after[key] else 'changed')
            rows.append(dict(symbol=key, demangled=demangled, change=change,
                             before=hashlib.sha256(before[key].encode()).hexdigest() if key in before else None,
                             after=hashlib.sha256(after[key].encode()).hexdigest() if key in after else None))
        counts = dict(Counter(row['change'] for row in rows))
        changed = {row['demangled'] for row in rows if row['change'] == 'changed'}
        missing, extra = sorted(expected - changed), sorted(changed - expected)
        good = not (missing or extra or counts.get('added') or counts.get('removed'))
        comparisons.append(dict(base=base, new=new, status='PASS' if good else 'FAIL', counts=counts,
                                expected=sorted(expected), missing=missing, extra=extra, rows=rows))
        print('SASS', base, new, counts, 'missing', missing, 'extra', extra, flush=True)
    B.save(B.ART / 'SASS_COMPARISON.json', dict(dumps=dumps, comparisons=comparisons))
    assert all(row['status'] == 'PASS' for row in comparisons)


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    mode = sys.argv[1]
    if mode == 'stage':
        for label in LABELS:
            stage(label)
        assert len({precheck(label)['module_sha256'] for label in LABELS}) == len(LABELS)
    elif mode == 'import':
        label = sys.argv[2]
        tc = load_arm(label)
        assert all(hasattr(tc._spo2, name) for name in ('packed_run', 'packed_prefill'))
        print('CPU_IMPORT', label, tc._C.__file__, B.sha(tc._C.__file__))
    elif mode == 'host':
        host_guard()
    else:
        assert mode == 'sass'
        sass()


if __name__ == '__main__':
    main()
