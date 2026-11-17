"""APA-P1-F2 S3: per-kernel SASS, Build F2 vs Build F, against artifacts/apa_p1f2/EXPECTED.json.

Runs scripts/apa_p1f1_sass.py's compare_builds() (after R7: optional `_<n>` suffix mapping, active
expectation recorded, renames counted apart from additions), imported, with Build F (p1d worktree,
module sha256 5c171477...) as the base and the given F2 build receipt as the new side. No override
file exists for F2: EXPECTED.json is the only registration (written once, before the first F2 build).
Prior art: NVIDIA cuobjdump; P1-D/F1 SASS comparison (Project-Tensor 2026), imported. CPU only.
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import apa_p1f1_sass as f1s  # noqa: E402

ART = ROOT / 'artifacts/apa_p1f2'
P1D_WT = Path('/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1d')
F_MODULE = P1D_WT / 'artifacts/apa_p1f1/build_F_r2/lib/_tensor_cuda.cpython-312-x86_64-linux-gnu.so'
F_SHA = '5c17147715ffc422b0aa7ba8fae826c554a4caff63b1231c22ce580867efc243'


def main():
    build = sys.argv[1] if len(sys.argv) > 1 else 'build_F2'
    rec = json.loads((ART / build / 'BUILD.json').read_text())
    assert rec['status'] == 'PASS'
    label = build.replace('build_', '')
    result = f1s.compare_builds('F', F_MODULE, F_SHA, label, ROOT / rec['module'], rec['module_sha256'], ART,
                                ART / 'EXPECTED.json', None, out_name=f'SASS_COMPARISON_{label}.json')
    sys.exit(0 if result['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
