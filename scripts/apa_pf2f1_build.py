"""CPU-only F1 and incremental mutants. Prior art: PF2/F2 build relocation
(Project-Tensor 2026), CMake (Kitware 2000), make (Feldman 1979), reused.
No new build algorithm. All output paths are under this round's build_*.
"""
from pathlib import Path
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/apa_pf2f1'
HEADER = 'tensor_cuda/src/apa_spo/packed_prefill.cuh'
MUTANTS = {
    'pf1m01_sink_guard_positive': [('denom != 0.f ? 1.f / denom', 'denom > 0.f ? 1.f / denom')],
    'pf1m02_norm_finite_only': [('isfinite(norm) ? norm : __int_as_float(0x7fc00000)', 'norm')],
    'pf1m03_index_int32': [('diagnostic_tail_index(int64_t base, int step)', 'diagnostic_tail_index(int base, int step)')],
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=2) + '\n')


def rails():
    mem = next(int(s.split()[1]) * 1024 for s in Path('/proc/meminfo').read_text().splitlines()
               if s.startswith('MemAvailable:'))
    disk = shutil.disk_usage('/mnt/ForgeRealm').free
    assert mem >= 12e9 and disk >= 40e9, f'RAIL: memory={mem}, disk={disk}'
    return mem, disk


def source_map(root):
    files = []
    for part in ('src', 'include', 'tensor_cuda'):
        files += [p for p in (root / 'tensor_cuda' / part).rglob('*')
                  if p.is_file() and p.suffix != '.pyc' and '__pycache__' not in p.parts and p.suffix != '.so']
    files += [root / 'tensor_cuda/CMakeLists.txt']
    return {str(p.relative_to(root)): sha(p) for p in sorted(files)}


def receipt(label):
    if label == 'PF':
        return json.loads((ROOT / 'artifacts/apa_p1pf2/build_PF/BUILD.json').read_text())
    r = json.loads((ART / f'build_{label}/BUILD.json').read_text())
    assert r['status'] == 'PASS'
    return r


def run(rec, cmd, log, limit):
    argv = ['timeout', f'{limit}s', 'nice', '-n', '19', *cmd]
    started = time.monotonic()
    mem_min, disk_min = rails()
    with log.open('x') as f:
        p = subprocess.Popen(argv, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        # Foreground supervision; no detached command or background wait.
        while True:
            try:
                code = p.wait(timeout=5)
                break
            except subprocess.TimeoutExpired:
                try:
                    m, d = rails()
                    mem_min, disk_min = min(mem_min, m), min(disk_min, d)
                except AssertionError as e:
                    # Only our own foreground process group may be stopped.
                    p.terminate()
                    p.wait(timeout=30)
                    code = 6
                    rec['rail_stop'] = str(e)
                    break
    rec['commands'].append(dict(argv=argv, exit=code, seconds=time.monotonic()-started,
                                log=str(log.relative_to(ROOT)), log_sha256=sha(log),
                                min_mem_available_bytes=mem_min, min_disk_free_bytes=disk_min))
    save(log.parent / 'BUILD.json', rec)
    assert code == 0, f'build command exit {code}: {log}'


def relocate(tree, pairs):
    count = 0
    for p in tree.rglob('*'):
        if not p.is_file() or p.is_symlink() or p.suffix in ('.o', '.so', '.a', '.bin', '.cubin', '.fatbin'):
            continue
        try:
            s = p.read_text()
        except UnicodeDecodeError:
            continue
        t = s
        for old, new in pairs:
            t = t.replace(str(old), str(new))
        if t != s:
            st = p.stat()
            p.write_text(t)
            os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
            count += 1
    return count


def build(label):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert os.environ['TMPDIR'].startswith('/tmp/pf2f1_')
    Path(os.environ['TMPDIR']).mkdir(exist_ok=True)
    rails()
    out = ART / f'build_{label}'
    assert not out.exists()
    rec = dict(label=label, status='RUNNING', started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               commands=[], registration_sha256=sha(ART / 'REGISTRATION.json'), hooks=False)
    if label == 'F1':
        source = ROOT
        cmake_source = out / 'cmake_source'
        cmake_source.mkdir(parents=True)
        original = (ROOT / 'tensor_cuda/CMakeLists.txt').read_text()
        adapted = original.replace('  src/', '  ' + str(ROOT / 'tensor_cuda/src') + '/')
        adapted = adapted.replace('${CMAKE_CURRENT_SOURCE_DIR}/include', str(ROOT / 'tensor_cuda/include'))
        adapted = adapted.replace('${CMAKE_CURRENT_SOURCE_DIR}/tensor_cuda', str(out / 'lib'))
        (cmake_source / 'CMakeLists.txt').write_text(adapted)
        base = receipt('PF')
        rec['p1d_flags'] = base['p1d_flags']
        rec['base_module_sha256'] = base['module_sha256']
        rec['source_sha256'] = source_map(source)
        save(out / 'BUILD.json', rec)
        run(rec, ['cmake', '-S', str(cmake_source), '-B', str(out / 'tree'), *base['p1d_flags']], out / 'configure.log', 300)
    else:
        assert label in MUTANTS
        base = receipt('F1')
        base_out = ART / 'build_F1'
        subprocess.run(['timeout', '120s', 'cp', '-a', str(base_out), str(out)], check=True)
        source = out / 'source'
        (source / 'tensor_cuda').mkdir(parents=True)
        for name in ('src', 'include', 'CMakeLists.txt'):
            subprocess.run(['timeout', '120s', 'cp', '-a', str(ROOT / 'tensor_cuda' / name), str(source / 'tensor_cuda' / name)], check=True)
        pairs = [(base_out, out), (ROOT / 'tensor_cuda/src', source / 'tensor_cuda/src'),
                 (ROOT / 'tensor_cuda/include', source / 'tensor_cuda/include')]
        rec['relocated_files'] = relocate(out / 'tree', pairs) + relocate(out / 'cmake_source', pairs)
        for objdir in (out / 'tree/CMakeFiles').glob('*.dir'):
            old = objdir / str(ROOT / 'tensor_cuda/src').lstrip('/')
            new = objdir / str(source / 'tensor_cuda/src').lstrip('/')
            if old.is_dir():
                new.parent.mkdir(parents=True, exist_ok=True)
                old.rename(new)
        p = source / HEADER
        text = p.read_text()
        for a, b in MUTANTS[label]:
            assert text.count(a) == (2 if label == 'pf1m02_norm_finite_only' else 1)
            text = text.replace(a, b)
        p.write_text(text)
        rec.update(base_module_sha256=base['module_sha256'], incremental=True,
                   module_before_sha256=sha(next((out / 'lib').glob('_tensor_cuda*.so'))),
                   mutations=MUTANTS[label], source_sha256=source_map(source))
        # cp -a preserved the baseline log: retain it, write a fresh incremental log.
        save(out / 'BUILD.json', rec)
    rec['source_root'] = str(source)
    try:
        run(rec, ['cmake', '--build', str(out / 'tree'), '-j3'],
            out / ('build.log' if label == 'F1' else 'build_incremental.log'), 3300)
        module = next((out / 'lib').glob('_tensor_cuda*.so'))
        rec.update(status='PASS', module=str(module.relative_to(ROOT)), module_sha256=sha(module),
                   source_sha256_after=source_map(source))
        assert rec['source_sha256'] == rec['source_sha256_after']
        log = out / ('build.log' if label == 'F1' else 'build_incremental.log')
        rec['rebuilt_objects'] = [s.strip() for s in log.read_text().splitlines() if 'Building' in s]
    except BaseException as e:
        rec.update(status='FAIL', error=repr(e))
        raise
    finally:
        rec['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save(out / 'BUILD.json', rec)
        with (ART / 'LEDGER.md').open('a') as f:
            f.write(f"- Build {label}: {rec['status']}; receipt {out.relative_to(ROOT)}/BUILD.json; module SHA {rec.get('module_sha256')}.\n")
    print(json.dumps({k: rec[k] for k in ('label', 'status', 'module_sha256', 'rebuilt_objects')}, indent=2))


def resume_f1():
    """Preserve the failed build receipt/log; resume the existing object tree."""
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    out = ART / 'build_F1'
    (out / 'BUILD.json').rename(out / 'BUILD.failed1.json')
    rec = json.loads((out / 'BUILD.failed1.json').read_text())
    rec.update(status='RUNNING', source_sha256=source_map(ROOT), commands=[], attempt=2)
    run(rec, ['cmake', '--build', str(out / 'tree'), '-j3'], out / 'build_resume.log', 3300)
    module = next((out / 'lib').glob('_tensor_cuda*.so'))
    rec.update(status='PASS', module=str(module.relative_to(ROOT)), module_sha256=sha(module),
               source_sha256_after=source_map(ROOT))
    assert rec['source_sha256'] == rec['source_sha256_after']
    rec['rebuilt_objects'] = [s.strip() for s in (out / 'build_resume.log').read_text().splitlines() if 'Building' in s]
    save(out / 'BUILD.json', rec)
    print('Build F1 PASS', rec['module_sha256'])


if __name__ == '__main__':
    resume_f1() if sys.argv[1] == 'resume-F1' else build(sys.argv[1])
