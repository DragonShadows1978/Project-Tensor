"""APA-P1-D S5: run the nine registered mutants (artifacts/apa_p1d/mutants/MUTANTS.json).

Prior art: mutation testing (DeMillo, Lipton and Sayward, 1978); house rule 8
(seeded registered defects, named expected failures). Ours: the nine policy
mutants of this order. CPU only; each mutant is a patched copy, never the tree.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/apa_p1d/mutants'
SRC = ROOT / 'tensor_cuda/tensor_cuda'
TEST = 'tensor_cuda/tests/test_apa_p1d_policy.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pytest(node, src, log):
    env = os.environ.copy()
    env.update(APA_P1D_SRC=str(src), CUDA_VISIBLE_DEVICES='', TMPDIR='/tmp/apap1d_mut', PYTHONDONTWRITEBYTECODE='1')
    Path(env['TMPDIR']).mkdir(exist_ok=True)
    cmd = [sys.executable, '-B', '-m', 'pytest', '-q', '--tb=short', '-p', 'no:cacheprovider',
           '--basetemp', f'/tmp/apap1d_mut/bt_{log.parent.name}_{log.stem}', f'{TEST}::{node}']
    with log.open('x') as f:
        code = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=300).returncode
    return code, log.read_text()


def main():
    spec = json.loads((ART / 'MUTANTS.json').read_text())
    sources = {name: sha(SRC / name) for name in ('apa_sp2.py', 'apa_packed.py')}
    baseline_dir = ART / '_baseline'
    baseline_dir.mkdir()
    rows = []
    for m in spec['mutants']:
        code, out = pytest(m['test'], SRC, baseline_dir / f"{m['name']}.log")
        assert code == 0, ('baseline must pass', m['test'])
        d = ART / m['name']
        d.mkdir()
        for name in sources:
            shutil.copyfile(SRC / name, d / name)
        text = (d / m['file']).read_text()
        for old, new in [(m['old'], m['new'])] + ([(m['also']['old'], m['also']['new'])] if 'also' in m else []):
            assert text.count(old) == 1, ('patch anchor must occur once', m['name'], old)
            text = text.replace(old, new)
        (d / m['file']).write_text(text)
        code, out = pytest(m['test'], d, d / 'run.log')
        killed = code == 1 and m['expected_failure'] in out
        rows.append(dict(name=m['name'], test=m['test'], expected_failure=m['expected_failure'], exit=code,
                         expected_text_found=m['expected_failure'] in out, killed=killed,
                         mutated_file_sha256=sha(d / m['file']), log=str((d / 'run.log').relative_to(ROOT))))
        print(m['name'], 'KILLED' if killed else 'SURVIVED', code, flush=True)
    result = dict(source_sha256=sources, mutants_json_sha256=sha(ART / 'MUTANTS.json'), rows=rows,
                  killed=sum(r['killed'] for r in rows), total=len(rows),
                  status='PASS' if all(r['killed'] for r in rows) else 'FAIL')
    with (ART / 'MUTANT_RESULTS.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(result['status'], result['killed'], '/', result['total'])
    sys.exit(0 if result['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
