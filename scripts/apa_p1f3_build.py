"""APA-P1-F3 S2/S5 builds: Build F3 (this tree, hooks off) and the incremental device/host mutant builds.

F2's build script (scripts/apa_p1f2_build.py) is IMPORTED, not copied or edited:
- Build F3 runs F2's steps with F2's own helpers: the P1-D path adapter and configure flags
  (f2b.f1b.p1d_adapter_and_flags, which reads P1-D's script), f2b.run_cmd, f2b.finish and the engine source
  map (f2b.f1b.engine_sources). Only the output directory (artifacts/apa_p1f3/build_F3[_r<n>]) and the label
  "F3" differ; F2's build_f2() hard-codes artifacts/apa_p1f2 and the label "F2", so it is not called.
- Mutant builds call F2's build_mutant() itself, with its module globals pointed at this round: ART ->
  artifacts/apa_p1f3, f2_receipt -> the Build F3 receipt, env_check -> TMPDIR under /tmp/apap1f3_.

Every receipt records the module sha256 and the sha256 map of every engine source file (L103).
Prior art: CMake (Kitware, 2000), make's timestamp-driven incremental rebuild (Feldman, 1979), nvcc (NVIDIA);
P1-D/F1/F2 build plumbing (Project-Tensor 2026), imported. No prior art known to me beyond that. CPU only.
"""
from pathlib import Path
import datetime
import difflib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import apa_p1f2_build as f2b  # noqa: E402

ART = ROOT / 'artifacts/apa_p1f3'
F2_BUILD = ROOT / 'artifacts/apa_p1f2/build_F2/BUILD.json'


def env_check():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert os.environ.get('TMPDIR', '').startswith('/tmp/apap1f3_')
    Path(os.environ['TMPDIR']).mkdir(exist_ok=True)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1')


def f3_receipt():
    recs = [json.loads(p.read_text()) for p in sorted(ART.glob('build_F3*/BUILD.json'))]
    recs = [r for r in recs if r.get('status') == 'PASS' and r['label'] == 'F3']
    assert recs, 'no passing Build F3 receipt'
    return max(recs, key=lambda r: r.get('attempt', 1))


def build_f3(attempt):
    env_check()
    assert (ART / 'EXPECTED.json').exists(), 'S3: EXPECTED.json must exist before the first Build F3'
    out = ART / ('build_F3' if attempt == 1 else f'build_F3_r{attempt}')
    source = out / 'cmake_source'
    source.mkdir(parents=True)
    reps, flags, p1d_sha = f2b.f1b.p1d_adapter_and_flags()
    original = (ROOT / 'tensor_cuda/CMakeLists.txt').read_text()
    adapted = original
    scope = dict(ROOT=ROOT, build=out, str=str)
    for old, new in reps:
        adapted = adapted.replace(eval(old, scope), eval(new, scope))
    (source / 'CMakeLists.txt').write_text(adapted)
    (out / 'BUILD_PATH_ADAPTER.diff').write_text(''.join(difflib.unified_diff(
        original.splitlines(True), adapted.splitlines(True), fromfile='tree/CMakeLists.txt',
        tofile=f'{out.name}/cmake_source/CMakeLists.txt')))
    f2rec = json.loads(F2_BUILD.read_text())
    record = dict(label='F3', attempt=attempt, hooks=False,
                  started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), source_root=str(ROOT),
                  p1d_build_script_sha256=p1d_sha, p1d_adapter_replacements=reps, p1d_flags=flags,
                  same_flags_as_F2=flags == f2rec['p1d_flags'],
                  same_adapter_replacements_as_F2=reps == f2rec['p1d_adapter_replacements'],
                  same_source_root_as_F2=str(ROOT) == f2rec['source_root'],
                  expected_json_sha256=f2b.sha(ART / 'EXPECTED.json'),
                  source_sha256=f2b.f1b.engine_sources(ROOT), status='RUNNING', commands=[])
    (out / 'BUILD.json').write_text(json.dumps(record, indent=2) + '\n')
    for i, cmd in enumerate([['cmake', '-S', str(source), '-B', str(out / 'tree'), *flags],
                             ['cmake', '--build', str(out / 'tree'), '-j3']]):
        if f2b.run_cmd(record, cmd, out / ('configure.log' if i == 0 else 'build.log'), 900 if i == 0 else 3300):
            record['status'] = 'FAIL'
            return f2b.finish(record, out, ROOT)
    module = next((out / 'lib').glob('_tensor_cuda*.so'))
    record.update(status='PASS', module=str(module.relative_to(ROOT)), module_sha256=f2b.sha(module))
    cache = (out / 'tree/CMakeCache.txt').read_text()
    record['cmake_cache_TC_APA_PACKED'] = [l for l in cache.splitlines() if l.startswith('TC_APA_PACKED')]
    record['source_sha256_after'] = f2b.f1b.engine_sources(ROOT)
    record['source_unchanged_during_build'] = record['source_sha256_after'] == record['source_sha256']
    return f2b.finish(record, out, ROOT)


def build_mutant(name, attempt=1):
    env_check()
    f2b.ART = ART
    f2b.f2_receipt = f3_receipt
    f2b.env_check = env_check
    return f2b.build_mutant(name, attempt)


def main():
    what = sys.argv[1]
    if what == 'F3':
        sys.exit(build_f3(int(sys.argv[2]) if len(sys.argv) > 2 else 1))
    if what == 'mutant':
        sys.exit(build_mutant(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1))
    raise SystemExit('usage: apa_p1f3_build.py F3 [attempt] | mutant <name> [attempt]')


if __name__ == '__main__':
    main()
