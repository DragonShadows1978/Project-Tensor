"""APA-P1-F1 S5: stage the registered mutants, run the CPU ones, collect results.

  stage <name>     copy tensor_cuda/{src,include,CMakeLists.txt,tensor_cuda} into
                   artifacts/apa_p1f1/mutants/<name>/tree and apply the ONE change
                   of artifacts/apa_p1f1/mutants/MUTANTS.json (anchor must occur once)
  cpu              baseline-pass then mutant run of each CPU mutant's named test
  collect          MUTANT_RESULTS.json from the CPU runs and the p1f1-MUT GPU line

Prior art: mutation testing (DeMillo, Lipton and Sayward, 1978); house rule 8;
P1-D's mutant runner shape (scripts/apa_p1d_mutants.py). CPU only.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/apa_p1f1/mutants'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spec(name):
    return {m['name']: m for m in json.loads((ART / 'MUTANTS.json').read_text())['mutants']}[name]


def stage(name):
    m = spec(name)
    tree = ART / name / 'tree'
    for rel in ('tensor_cuda/src', 'tensor_cuda/include', 'tensor_cuda/tensor_cuda'):
        shutil.copytree(ROOT / rel, tree / rel, ignore=shutil.ignore_patterns('__pycache__', '*.so'))
    shutil.copyfile(ROOT / 'tensor_cuda/CMakeLists.txt', tree / 'tensor_cuda/CMakeLists.txt')
    target = tree / m['file']
    text = target.read_text()
    if 'old' in m:
        assert text.count(m['old']) == 1, ('anchor must occur once', name)
        new = text.replace(m['old'], m['new'])
    else:
        a, b = text.index(m['old_block_from']), text.index(m['old_block_to'])
        assert text.count(m['old_block_from']) == 1 and text.count(m['old_block_to']) == 1 and a < b
        new = text[:a] + text[b:]
    target.write_text(new)
    rec = dict(name=name, file=m['file'], original_sha256=sha(ROOT / m['file']), mutated_sha256=sha(target),
               changed_bytes=len(text) - len(new))
    (ART / name / 'STAGED.json').write_text(json.dumps(rec, indent=2) + '\n')
    print(json.dumps(rec))


def pytest(node, src, log):
    env = os.environ.copy()
    env.update(APA_P1F1_SRC=str(src), CUDA_VISIBLE_DEVICES='', TMPDIR='/tmp/apap1f1_mut', PYTHONDONTWRITEBYTECODE='1')
    Path(env['TMPDIR']).mkdir(exist_ok=True)
    cmd = [sys.executable, '-B', '-m', 'pytest', '-q', '--tb=short', '-p', 'no:cacheprovider',
           '--basetemp', f'/tmp/apap1f1_mut/bt_{log.parent.name}_{log.stem}', node]
    with log.open('x') as f:
        code = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=300).returncode
    return code, log.read_text()


def cpu():
    rows = []
    for m in json.loads((ART / 'MUTANTS.json').read_text())['mutants']:
        if m['runs_on'] != 'cpu':
            continue
        base = ART / m['name'] / 'baseline.log'
        code0, _ = pytest(m['test'], ROOT / 'tensor_cuda/tensor_cuda', base)
        assert code0 == 0, ('baseline must pass', m['name'])
        code, out = pytest(m['test'], ART / m['name'] / 'tree/tensor_cuda/tensor_cuda', ART / m['name'] / 'run.log')
        killed = code == 1 and m['expected_failure'] in out
        rows.append(dict(name=m['name'], test=m['test'], expected_failure=m['expected_failure'], baseline_exit=code0,
                         exit=code, expected_text_found=m['expected_failure'] in out, killed=killed,
                         log=str((ART / m['name'] / 'run.log').relative_to(ROOT))))
        print(m['name'], 'KILLED' if killed else 'SURVIVED', code, flush=True)
    (ART / 'CPU_MUTANT_RESULTS.json').write_text(json.dumps(rows, indent=2) + '\n')


def collect():
    specs = json.loads((ART / 'MUTANTS.json').read_text())['mutants']
    cpu_rows = {r['name']: r for r in json.loads((ART / 'CPU_MUTANT_RESULTS.json').read_text())}
    gpu = sorted((ROOT / 'artifacts/apa_p1f1/gpu/MUT').glob('RESULT.*.json'))
    gpu_rows = {}
    for p in gpu:   # later results supersede earlier attempts per mutant
        for r in json.loads(p.read_text())['results']:
            gpu_rows[r['name']] = dict(r, result_file=str(p.relative_to(ROOT)))
    rows = []
    for m in specs:
        r = cpu_rows.get(m['name']) if m['runs_on'] == 'cpu' else gpu_rows.get(m['name'])
        build = ART / m['name'] / 'build/BUILD.json'
        b = json.loads(build.read_text()) if build.exists() else None
        rows.append(dict(name=m['name'], finding=m['finding'], line=m['line'], expected_failure=m['expected_failure'],
                         killed=bool(r and r['killed']), module_sha256=b and b.get('module_sha256'),
                         failed_assertions=r and r.get('failed_assertions'), source=r and (r.get('log') or r.get('result_file'))))
    result = dict(mutants_json_sha256=sha(ART / 'MUTANTS.json'), rows=rows, killed=sum(r['killed'] for r in rows),
                  total=len(rows), status='PASS' if all(r['killed'] for r in rows) else 'FAIL')
    out = ART / 'MUTANT_RESULTS.json'
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(result['status'], result['killed'], '/', result['total'])


if __name__ == '__main__':
    {'stage': lambda: stage(sys.argv[2]), 'cpu': cpu, 'collect': collect}[sys.argv[1]]()
