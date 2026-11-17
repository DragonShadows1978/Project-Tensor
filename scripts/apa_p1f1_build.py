"""APA-P1-F1 S2/S5 builds: Build F, Build FH and the mutant builds.

P1-D's build script (scripts/apa_p1d_build.py) is IMPORTED, not copied:
its path adapter (the three CMakeLists replacements) and its configure flag
list are taken at run time from the imported module's main() source (AST), so
the adapter and flags are byte-for-byte P1-D's. Its main() itself cannot be
called: it asserts arm in (A, B) and TMPDIR under /tmp/apap1d_, and this order
puts scratch under /tmp/apap1f1_*. The build directory, receipt and the
optional hooks definition are this script's.

  F    : this tree, default configuration (test hooks off)
  FH   : this tree + target_compile_definitions(_tensor_cuda PRIVATE
         TC_APA_P1_TEST_HOOKS=1), appended exactly as S3's BUILD.py did
         (tensor_fork_apa_wt_apa2pprod/artifacts/apa_p1_s3_r2/BUILD.py)
  mutant <name> [hooks]: artifacts/apa_p1f1/mutants/<name>/src (a patched copy
         of tensor_cuda/src + include) built the same way.

Every receipt records the module sha256 and the sha256 map of every engine
source file (tensor_cuda/src, tensor_cuda/include, tensor_cuda/tensor_cuda).

Prior art: CMake (Kitware, 2000), nvcc (NVIDIA); P1-D/A1 path adapter
(Project-Tensor 2026), imported. No prior art known to me beyond that; plumbing.
CPU only (CUDA_VISIBLE_DEVICES empty).
"""
from pathlib import Path
import ast
import datetime
import difflib
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import apa_p1d_build as p1d  # noqa: E402

ART = ROOT / 'artifacts/apa_p1f1'
HOOKS_LINE = ('\n# Test-only hook; never enabled by the shipped CMake default.\n'
              'target_compile_definitions(_tensor_cuda PRIVATE TC_APA_P1_TEST_HOOKS=1)\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def engine_sources(root):
    out = {}
    for d in ('tensor_cuda/src', 'tensor_cuda/include', 'tensor_cuda/tensor_cuda'):
        for p in sorted((root / d).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.so':
                out[str(p.relative_to(root))] = sha(p)
    out['tensor_cuda/CMakeLists.txt'] = sha(root / 'tensor_cuda/CMakeLists.txt')
    return out


def p1d_adapter_and_flags():
    """Return (replacement pairs, flags) as written in apa_p1d_build.main."""
    src = inspect.getsource(p1d.main)
    tree = ast.parse(src)
    reps, flags = [], None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and \
                isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'replace' and \
                isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'adapted':
            reps.append((ast.get_source_segment(src, node.value.args[0]), ast.get_source_segment(src, node.value.args[1])))
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'flags' \
                and isinstance(node.value, ast.List):
            flags = [ast.literal_eval(e) for e in node.value.elts]
    assert len(reps) == 3 and flags, (reps, flags)
    return reps, flags, hashlib.sha256(Path(p1d.__file__).read_bytes()).hexdigest()


def build(label, src_root, out, hooks):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert os.environ.get('TMPDIR', '').startswith('/tmp/apap1f1_')
    Path(os.environ['TMPDIR']).mkdir(exist_ok=True)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1')
    source = out / 'cmake_source'
    source.mkdir(parents=True)
    reps, flags, p1d_sha = p1d_adapter_and_flags()
    original = (src_root / 'tensor_cuda/CMakeLists.txt').read_text()
    adapted = original
    # Evaluate P1-D's replacement expressions with this build's ROOT/build bound
    # to the names its main() uses (ROOT, build).
    scope = dict(ROOT=src_root, build=out, str=str)
    for old, new in reps:
        adapted = adapted.replace(eval(old, scope), eval(new, scope))
    if hooks:
        adapted += HOOKS_LINE
    (source / 'CMakeLists.txt').write_text(adapted)
    (out / 'BUILD_PATH_ADAPTER.diff').write_text(''.join(difflib.unified_diff(
        original.splitlines(True), adapted.splitlines(True),
        fromfile='tree/CMakeLists.txt', tofile=f'{label}/cmake_source/CMakeLists.txt')))
    commands = [['cmake', '-S', str(source), '-B', str(out / 'tree'), *flags],
                ['cmake', '--build', str(out / 'tree'), '-j3']]
    record = dict(label=label, hooks=hooks, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  source_root=str(src_root), p1d_build_script_sha256=p1d_sha, p1d_adapter_replacements=reps,
                  p1d_flags=flags, hooks_definition=HOOKS_LINE.strip() if hooks else None,
                  source_sha256=engine_sources(src_root), status='RUNNING', commands=[])
    path = out / 'BUILD.json'
    path.write_text(json.dumps(record, indent=2) + '\n')
    for i, cmd in enumerate(commands):
        start = time.monotonic()
        log = out / ('configure.log' if i == 0 else 'build.log')
        with log.open('x') as f:
            code = subprocess.run(['timeout', '900s' if i == 0 else '3300s', *cmd],
                                  stdout=f, stderr=subprocess.STDOUT).returncode
        record['commands'].append(dict(argv=cmd, exit=code, seconds=time.monotonic() - start, log=str(log)))
        if code:
            record['status'] = 'FAIL'
            break
    else:
        module = next((out / 'lib').glob('_tensor_cuda*.so'))
        record.update(status='PASS', module=str(module.relative_to(ROOT)), module_sha256=sha(module))
        cache = (out / 'tree/CMakeCache.txt').read_text()
        record['cmake_cache_TC_APA_PACKED'] = [l for l in cache.splitlines() if l.startswith('TC_APA_PACKED')]
        record['source_sha256_after'] = engine_sources(src_root)
        record['source_unchanged_during_build'] = record['source_sha256_after'] == record['source_sha256']
    record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    path.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: v for k, v in record.items() if not k.startswith('source_sha256')}, indent=2))
    return 0 if record['status'] == 'PASS' else 1


def refresh(label, base, attempt):
    """Incremental rebuild of an existing build tree after a Python-only change.

    Runs `cmake --build` again in <base>/tree (make recompiles nothing when no
    C++/CUDA source changed), checks the module is byte-identical, and writes a
    new receipt build_<label>_r<attempt>/BUILD.json with the source map at this
    build time (L103). The module path stays the base build's.
    """
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert os.environ.get('TMPDIR', '').startswith('/tmp/apap1f1_')
    base_rec = json.loads((ART / base / 'BUILD.json').read_text())
    assert base_rec['status'] == 'PASS' and base_rec['label'] == label
    out = ART / f'build_{label}_r{attempt}'
    out.mkdir()
    before = sha(ROOT / base_rec['module'])
    assert before == base_rec['module_sha256']
    record = dict(label=label, hooks=base_rec['hooks'], kind='incremental refresh of ' + base,
                  started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  source_root=str(ROOT), source_sha256=engine_sources(ROOT), status='RUNNING', commands=[])
    cmd = ['cmake', '--build', str(ART / base / 'tree'), '-j3']
    start = time.monotonic()
    with (out / 'build.log').open('x') as f:
        code = subprocess.run(['timeout', '3300s', *cmd], stdout=f, stderr=subprocess.STDOUT).returncode
    record['commands'].append(dict(argv=cmd, exit=code, seconds=time.monotonic() - start, log=str(out / 'build.log')))
    after = sha(ROOT / base_rec['module'])
    record.update(status='PASS' if code == 0 and after == before else 'FAIL', module=base_rec['module'],
                  module_sha256=after, module_unchanged=after == before,
                  source_sha256_after=engine_sources(ROOT))
    record['source_unchanged_during_build'] = record['source_sha256_after'] == record['source_sha256']
    record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out / 'BUILD.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: v for k, v in record.items() if not k.startswith('source_sha256')}, indent=2))
    return 0 if record['status'] == 'PASS' else 1


def main():
    what = sys.argv[1]
    if what == 'refresh':
        sys.exit(refresh(sys.argv[2], sys.argv[3], int(sys.argv[4])))
    if what in ('F', 'FH'):
        attempt = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        name = f'build_{what}' if attempt == 1 else f'build_{what}_r{attempt}'
        sys.exit(build(what, ROOT, ART / name, hooks=what == 'FH'))
    if what == 'mutant':
        name = sys.argv[2]
        hooks = len(sys.argv) > 3 and sys.argv[3] == 'hooks'
        mroot = ART / 'mutants' / name / 'tree'
        assert (mroot / 'tensor_cuda/src').is_dir(), 'stage the mutant source first (apa_p1f1_mutants.py stage)'
        sys.exit(build(f'mutant_{name}', mroot, ART / 'mutants' / name / 'build', hooks=hooks))
    raise SystemExit('usage: apa_p1f1_build.py F|FH [attempt] | mutant <name> [hooks]')


if __name__ == '__main__':
    main()
