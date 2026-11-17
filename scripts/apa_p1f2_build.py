"""APA-P1-F2 S2/S5 builds: Build F2 (this tree, hooks off) and the incremental device/binding mutant builds.

P1-D's build script (scripts/apa_p1d_build.py) is IMPORTED, not copied: its path adapter (the
three CMakeLists replacements) and configure flags are read from its main() source by F1's
p1d_adapter_and_flags() (scripts/apa_p1f1_build.py, imported), exactly as F1 built. F1's build()
itself cannot be called: it asserts TMPDIR under /tmp/apap1f1_ and writes into artifacts/apa_p1f1.

  F2                 : this tree -> artifacts/apa_p1f2/build_F2[_r<n>]
  mutant <name>      : incremental rebuild in a COPY of Build F2's tree. The F2 build tree and the
                       F2 source tree are copied with timestamps kept (cp -a), the copied build
                       tree's absolute paths are rewritten (F2 source root -> mutant source root,
                       F2 build dir -> mutant build dir), the mutant patch is applied to the copied
                       source (newer mtime), and `cmake --build` recompiles only what depends on it.
                       The receipt records which objects were rebuilt (build.log).

Every receipt records the module sha256 and the sha256 map of every engine source file
(tensor_cuda/src, tensor_cuda/include, tensor_cuda/tensor_cuda) at build time (L103).
Prior art: CMake (Kitware, 2000), make's timestamp-driven incremental rebuild (Feldman, 1979),
nvcc (NVIDIA); P1-D/A1 path adapter (Project-Tensor 2026), imported. No prior art known to me
for the build-tree relocation beyond that; plumbing. CPU only (CUDA_VISIBLE_DEVICES empty).
"""
from pathlib import Path
import datetime
import difflib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import apa_p1f1_build as f1b  # noqa: E402  (imports apa_p1d_build)

ART = ROOT / 'artifacts/apa_p1f2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def env_check():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert os.environ.get('TMPDIR', '').startswith('/tmp/apap1f2_')
    Path(os.environ['TMPDIR']).mkdir(exist_ok=True)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1')


def run_cmd(record, cmd, log, limit):
    start = time.monotonic()
    with log.open('x') as f:
        code = subprocess.run(['timeout', f'{limit}s', *cmd], stdout=f, stderr=subprocess.STDOUT).returncode
    record['commands'].append(dict(argv=[str(c) for c in cmd], exit=code, seconds=time.monotonic() - start,
                                   log=str(log)))
    return code


def finish(record, out, src_root):
    record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out / 'BUILD.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: v for k, v in record.items() if not k.startswith('source_sha256')}, indent=2))
    return 0 if record['status'] == 'PASS' else 1


def build_f2(attempt):
    env_check()
    out = ART / ('build_F2' if attempt == 1 else f'build_F2_r{attempt}')
    source = out / 'cmake_source'
    source.mkdir(parents=True)
    reps, flags, p1d_sha = f1b.p1d_adapter_and_flags()
    original = (ROOT / 'tensor_cuda/CMakeLists.txt').read_text()
    adapted = original
    scope = dict(ROOT=ROOT, build=out, str=str)
    for old, new in reps:
        adapted = adapted.replace(eval(old, scope), eval(new, scope))
    (source / 'CMakeLists.txt').write_text(adapted)
    (out / 'BUILD_PATH_ADAPTER.diff').write_text(''.join(difflib.unified_diff(
        original.splitlines(True), adapted.splitlines(True), fromfile='tree/CMakeLists.txt',
        tofile='build_F2/cmake_source/CMakeLists.txt')))
    record = dict(label='F2', attempt=attempt, hooks=False,
                  started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), source_root=str(ROOT),
                  p1d_build_script_sha256=p1d_sha, p1d_adapter_replacements=reps, p1d_flags=flags,
                  source_sha256=f1b.engine_sources(ROOT), status='RUNNING', commands=[])
    (out / 'BUILD.json').write_text(json.dumps(record, indent=2) + '\n')
    for i, cmd in enumerate([['cmake', '-S', str(source), '-B', str(out / 'tree'), *flags],
                             ['cmake', '--build', str(out / 'tree'), '-j3']]):
        if run_cmd(record, cmd, out / ('configure.log' if i == 0 else 'build.log'), 900 if i == 0 else 3300):
            record['status'] = 'FAIL'
            return finish(record, out, ROOT)
    module = next((out / 'lib').glob('_tensor_cuda*.so'))
    record.update(status='PASS', module=str(module.relative_to(ROOT)), module_sha256=sha(module))
    cache = (out / 'tree/CMakeCache.txt').read_text()
    record['cmake_cache_TC_APA_PACKED'] = [l for l in cache.splitlines() if l.startswith('TC_APA_PACKED')]
    record['source_sha256_after'] = f1b.engine_sources(ROOT)
    record['source_unchanged_during_build'] = record['source_sha256_after'] == record['source_sha256']
    return finish(record, out, ROOT)


def f2_receipt():
    recs = [json.loads(p.read_text()) for p in sorted(ART.glob('build_F2*/BUILD.json'))]
    recs = [r for r in recs if r.get('status') == 'PASS' and r['label'] == 'F2']
    assert recs, 'no passing Build F2 receipt'
    return recs[-1]


TEXT_SUFFIXES = {'.make', '.txt', '.cmake', '.json', '.d', '.marks', '.ts', '.includecache', '.internal', ''}


def relocate(tree, pairs):
    """Rewrite absolute paths in the copied build tree's text files (never object files)."""
    changed = []
    for p in sorted(tree.rglob('*')):
        if not p.is_file() or p.is_symlink() or p.suffix in ('.o', '.so', '.a', '.bin', '.out', '.cubin', '.fatbin'):
            continue
        # Repair 2 (L56b): every text file is rewritten (attempt 2 skipped deviceObjects1.rsp); binaries are
        # skipped by suffix above and by the UTF-8 decode below.
        try:
            text = p.read_text()
        except UnicodeDecodeError:
            continue
        new = text
        for a, b in pairs:
            new = new.replace(a, b)
        if new != text:
            st = p.stat()
            p.write_text(new)
            os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))   # keep make's view of the tree
            changed.append(str(p.relative_to(tree)))
    return changed


def mutant_build_dir(mdir, attempt):
    return mdir / ('build' if attempt == 1 else f'build_r{attempt}')


def build_mutant(name, attempt=1):
    """Incremental rebuild of mutants/<name>: its patched source tree is mutants/<name>/tree (staged by
    scripts/apa_p1f2_mutants.py stage, a cp -a copy of this worktree's tensor_cuda/{src,include,
    CMakeLists.txt} with the one-change patch applied)."""
    env_check()
    base = f2_receipt()
    base_out = ROOT / Path(base['module']).parents[1]
    mdir = ART / 'mutants' / name
    mroot = mdir / 'tree'
    assert (mroot / 'tensor_cuda/src').is_dir(), 'stage the mutant source first (apa_p1f2_mutants.py stage)'
    out = mutant_build_dir(mdir, attempt)
    assert not out.exists()
    # Copy F2's build directory with timestamps (cp -a), then relocate its absolute paths.
    subprocess.run(['timeout', '600s', 'cp', '-a', str(base_out), str(out)], check=True)
    pairs = [(str(base_out), str(out)), (str(ROOT / 'tensor_cuda/src'), str(mroot / 'tensor_cuda/src')),
             (str(ROOT / 'tensor_cuda/include'), str(mroot / 'tensor_cuda/include'))]
    rewritten = relocate(out / 'tree', pairs) + ['cmake_source/' + x for x in relocate(out / 'cmake_source', pairs)]
    # Repair 1 (L56b): CMake keys each object by its absolute source path
    # (CMakeFiles/<target>.dir/<abs src path>.o); move the copied object directory to the rewritten
    # source path so make finds the F2 objects and rebuilds only what depends on the patched file.
    moved = []
    for objdir in sorted((out / 'tree/CMakeFiles').glob('*.dir')):
        old = objdir / str(ROOT / 'tensor_cuda/src').lstrip('/')
        new = objdir / str(mroot / 'tensor_cuda/src').lstrip('/')
        if old.is_dir():
            new.parent.mkdir(parents=True, exist_ok=True)
            os.rename(old, new)
            moved.append(str(new.relative_to(out)))
    record = dict(label=f'mutant_{name}', base_build=base.get('label'), base_module_sha256=base['module_sha256'],
                  started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), source_root=str(mroot),
                  kind='incremental rebuild in a copy of Build F2 (cp -a, absolute paths rewritten)',
                  attempt=attempt, path_rewrites=pairs, rewritten_files=len(rewritten), object_dirs_moved=moved,
                  source_sha256=f1b.engine_sources(mroot),
                  status='RUNNING', commands=[])
    module = next((out / 'lib').glob('_tensor_cuda*.so'))
    record['module_before_sha256'] = sha(module)
    (out / 'BUILD.json').write_text(json.dumps(record, indent=2) + '\n')
    if run_cmd(record, ['cmake', '--build', str(out / 'tree'), '-j3'], out / 'build_incremental.log', 3300):
        record['status'] = 'FAIL'
        return finish(record, out, mroot)
    log = (out / 'build_incremental.log').read_text()
    record['rebuilt_objects'] = [l.strip() for l in log.splitlines() if 'Building' in l]
    record.update(status='PASS', module=str(module.relative_to(ROOT)), module_sha256=sha(module))
    record['module_changed_from_F2'] = record['module_sha256'] != base['module_sha256']
    record['source_sha256_after'] = f1b.engine_sources(mroot)
    return finish(record, out, mroot)


def refresh(attempt):
    """Receipt-only refresh after a Python-only change (as F1's build_F_r3): `cmake --build` again in
    build_F2/tree (make recompiles nothing when no C++/CUDA source changed), check the module is
    byte-identical, and record the current engine source map (L103) in build_F2_r<attempt>/BUILD.json."""
    env_check()
    base = json.loads((ART / 'build_F2/BUILD.json').read_text())
    assert base['status'] == 'PASS'
    out = ART / f'build_F2_r{attempt}'
    out.mkdir()
    before = sha(ROOT / base['module'])
    assert before == base['module_sha256']
    record = dict(label='F2', attempt=attempt, hooks=False, kind='incremental refresh of build_F2 (receipt only)',
                  started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), source_root=str(ROOT),
                  source_sha256=f1b.engine_sources(ROOT), status='RUNNING', commands=[])
    code = run_cmd(record, ['cmake', '--build', str(ART / 'build_F2/tree'), '-j3'], out / 'build.log', 3300)
    after = sha(ROOT / base['module'])
    log = (out / 'build.log').read_text()
    record.update(status='PASS' if code == 0 and after == before else 'FAIL', module=base['module'],
                  module_sha256=after, module_unchanged=after == before,
                  rebuilt_objects=[l.strip() for l in log.splitlines() if 'Building' in l],
                  source_sha256_after=f1b.engine_sources(ROOT))
    record['source_unchanged_during_build'] = record['source_sha256_after'] == record['source_sha256']
    return finish(record, out, ROOT)


def main():
    what = sys.argv[1]
    if what == 'refresh':
        sys.exit(refresh(int(sys.argv[2])))
    if what == 'F2':
        sys.exit(build_f2(int(sys.argv[2]) if len(sys.argv) > 2 else 1))
    if what == 'mutant':
        sys.exit(build_mutant(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1))
    raise SystemExit('usage: apa_p1f2_build.py F2 [attempt] | mutant <name>')


if __name__ == '__main__':
    main()
