"""F3 release build and isolated incremental mutants.

Prior art: CMake (Kitware, 2000), PF2-F2's cloned CMake trees (2026) and
mutation analysis (DeMillo, Lipton and Sayward, 1978). Reuse the tree
relocation and rebuild only the translation unit owning the edited guards.
"""
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/apa_pf2f3'
F2 = ROOT.parent / 'tensor_fork_apa_wt_pf2f2/artifacts/apa_pf2f2'
F2_MODULE_SHA = 'aefe2513a4d33d7310421f65e6619f0706de2889410836bda42193053f8ec97a'
F2_RECEIPT_SHA = 'b23fa3e93bd5866a34e9e61eca2933990ee3bacefadac9001ced0410c58df827'
MUTANTS = ('pf3m01_prefill_norm_revert', 'pf3m02_prefill_norm_pinf_only', 'pf3m03_tc_norm_revert')
PRE = 'tensor_cuda/src/apa_spo/packed_prefill.cuh'
TC = tuple(f'tensor_cuda/src/apa_spo/{name}.cuh' for name in ('spq4', 'spq6'))
TC_GUARD = 'isfinite(norm)?norm:__int_as_float(0x7fc00000)'
PRE_GUARD = 'isfinite(norm) ? norm : __int_as_float(0x7fc00000)'
MUTATIONS = {
    MUTANTS[0]: {PRE: (PRE_GUARD, 'norm', 2)},
    MUTANTS[1]: {PRE: (PRE_GUARD, '(norm == INFINITY ? __int_as_float(0x7fc00000) : norm)', 2)},
    MUTANTS[2]: {name: (TC_GUARD, 'norm', 1) for name in TC},
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def receipt(label):
    if label == 'F2':
        path = F2 / 'build_F2/BUILD.json'
        assert sha(path) == F2_RECEIPT_SHA
        row = read(path)
        row['module'] = str(F2.parents[1] / row['module'])
        assert row['module_sha256'] == F2_MODULE_SHA and not row['hooks']
    else:
        row = read(ART / f'build_{label}' / 'BUILD.json')
    assert row['status'] == 'PASS', row
    return row


def disk_free():
    free = shutil.disk_usage('/mnt/ForgeRealm').free
    assert free >= 40_000_000_000, f'RAIL: disk_free_bytes={free}'
    return free


def run(out, name, command, limit):
    argv = ['timeout', f'{limit}s', 'nice', '-n', '19', *command]
    log = out / f'{name}.log'
    start = time.monotonic()
    minimum = disk_free()
    # One foreground build, with the disk rail checked while it runs.
    with log.open('x') as stream:
        process = subprocess.Popen(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        while True:
            try:
                code = process.wait(timeout=10)
                break
            except subprocess.TimeoutExpired:
                try:
                    minimum = min(minimum, disk_free())
                except AssertionError:
                    process.terminate()
                    process.wait(timeout=20)
                    code = 6
                    break
    return dict(argv=argv, exit=code, seconds=time.monotonic() - start,
                log=str(log.relative_to(ROOT)), log_sha256=sha(log), min_disk_free_bytes=minimum)


def configure(out, flags):
    cmake_source = out / 'cmake_source'
    cmake_source.mkdir()
    source = ROOT / 'tensor_cuda'
    cmake = (source / 'CMakeLists.txt').read_text()
    cmake = cmake.replace('  src/', f'  {source}/src/')
    cmake = cmake.replace('${CMAKE_CURRENT_SOURCE_DIR}/include', str(source / 'include'))
    cmake = cmake.replace('${CMAKE_CURRENT_SOURCE_DIR}/tensor_cuda', str(out / 'lib'))
    (cmake_source / 'CMakeLists.txt').write_text(cmake)
    return ['cmake', '-S', str(cmake_source), '-B', str(out / 'tree'), *flags]


def mutant_inputs(out, label, parent):
    base = ART / 'build_F3'
    for name in ('tree', 'cmake_source', 'lib'):
        shutil.copytree(base / name, out / name, symlinks=True)
    source = out / 'source' / 'tensor_cuda'
    for sub in ('src', 'include'):
        shutil.copytree(ROOT / 'tensor_cuda' / sub, source / sub)
    substitutions = [(str(base), str(out))]
    for sub in ('src', 'include'):
        substitutions.append((str(ROOT / 'tensor_cuda' / sub), str(source / sub)))
    for folder in (out / 'tree', out / 'cmake_source'):
        for path in folder.rglob('*'):
            if not path.is_file() or path.is_symlink() or path.suffix in ('.o', '.so', '.a', '.bin'):
                continue
            try:
                before = path.read_text()
            except UnicodeDecodeError:
                continue
            after = before
            for old, new in substitutions:
                after = after.replace(old, new)
            if after != before:
                info = path.stat()
                path.write_text(after)
                os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))
    obj = out / 'tree/CMakeFiles/_tensor_cuda.dir'
    old = obj / str(ROOT / 'tensor_cuda/src').lstrip('/')
    new = obj / str(source / 'src').lstrip('/')
    new.parent.mkdir(parents=True, exist_ok=True)
    old.rename(new)
    edits = {}
    with (out / 'MUTATION.diff').open('x') as output:
        for name, (old, new, count) in MUTATIONS[label].items():
            path = source.parent / name
            before = path.read_text()
            assert before.count(old) == count, name
            after = before.replace(old, new)
            path.write_text(after)
            output.writelines(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                                    fromfile=f'a/{name}', tofile=f'b/{name}'))
            edits[name] = dict(count=count, before=old, after=new)
    module, = (out / 'lib').glob('_tensor_cuda*.so')
    assert sha(module) == parent['module_sha256']
    return source.parent, edits


def input_map(parent, source):
    return {name: sha(source / name if (source / name).exists() else ROOT / name)
            for name in parent['source_sha256_after']}


def build(label):
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert os.environ['TMPDIR'].startswith('/tmp/pf2f3_')
    assert label == 'F3' or label in MUTANTS
    disk_free()
    f2 = receipt('F2')
    assert sha(f2['module']) == F2_MODULE_SHA
    parent = f2 if label == 'F3' else receipt('F3')
    out = ART / f'build_{label}'
    out.mkdir()
    rec = dict(label=label, status='FAIL', hooks=False, commands=[], p1d_flags=f2['p1d_flags'],
               registration_sha256=sha(ART / 'REGISTRATION.md'), base_module_sha256=parent['module_sha256'])
    try:
        if label == 'F3':
            source = ROOT
            changed = set(TC)
        else:
            source, rec['mutation'] = mutant_inputs(out, label, parent)
            changed = set(MUTATIONS[label])
        rec['source_root'] = str(source)
        rec['source_sha256'] = input_map(f2, source)
        actual = {k for k, v in rec['source_sha256'].items() if v != parent['source_sha256_after'][k]}
        assert actual == changed, (actual, changed)
        if label == 'F3':
            rec['commands'].append(run(out, 'configure', configure(out, f2['p1d_flags']), 300))
            assert rec['commands'][-1]['exit'] == 0, rec['commands'][-1]
        incremental = label in MUTANTS
        rec['commands'].append(run(out, 'build_incremental' if incremental else 'build',
                                   ['cmake', '--build', str(out / 'tree'), '-j1' if incremental else '-j2'], 3200))
        result = rec['commands'][-1]
        assert result['exit'] == 0, result
        rec['rebuilt_objects'] = [s.strip() for s in (ROOT / result['log']).read_text().splitlines()
                                  if 'Building ' in s]
        if incremental:
            assert len(rec['rebuilt_objects']) == 1 and 'kernels.cu.o' in rec['rebuilt_objects'][0], rec['rebuilt_objects']
        rec['source_sha256_after'] = input_map(f2, source)
        assert rec['source_sha256'] == rec['source_sha256_after']
        module, = (out / 'lib').glob('_tensor_cuda*.so')
        rec['module'] = str(module.relative_to(ROOT))
        rec['module_sha256'] = sha(module)
        assert rec['module_sha256'] != parent['module_sha256']
        rec['status'] = 'PASS'
    except BaseException as error:
        rec['error'] = repr(error)
        raise
    finally:
        save(out / 'BUILD.json', rec)
        with (ART / 'LEDGER.md').open('a') as ledger:
            ledger.write(f"- Build {label} {rec['status']}, {out.relative_to(ROOT)}/BUILD.json. "
                         f"Module {rec.get('module_sha256')}.\n")
    print(label, rec['status'], rec['module_sha256'], flush=True)


if __name__ == '__main__':
    build(sys.argv[1])
