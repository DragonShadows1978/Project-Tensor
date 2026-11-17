"""APA-P1-F3 S3/G4: per-kernel SASS, Build F3 vs Build F2, against artifacts/apa_p1f3/EXPECTED.json.

Runs scripts/apa_p1f1_sass.py's compare_builds() (after F2's R7: bijective _cu_<8 hex>[_<n>] mapping only,
active expectation recorded, renames counted apart from additions), imported unchanged, with Build F2
(artifacts/apa_p1f2/arms/F2, module sha256 6f03b2cd...) as the base and the given Build F3 receipt as the new
side. EXPECTED.json is the only registration (no override file). After the comparison, two receipt checks the
registration names (not gates of their own): the fresh dump of Build F2's module against F2's own dump sha256
(artifacts/apa_p1f2/HELD_BACK.txt / REV3: 3b7efd2a...), and whether any path-derived name (_GLOBAL__N__,
__nv_static_) differs between the two dumps (the registered rule: they must be equal, unmapped).
Prior art: NVIDIA cuobjdump; P1-D/F1/F2 SASS comparison (Project-Tensor 2026), imported. CPU only.
"""
from pathlib import Path
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import apa_p1f1_sass as f1s  # noqa: E402

ART = ROOT / 'artifacts/apa_p1f3'
F2_MODULE = ROOT / 'artifacts/apa_p1f2/arms/F2/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so'
F2_SHA = '6f03b2cdd5d7f9bee4456ed34759c6c710cccfab9a50f9ac0fc6a678c076f579'
F2_DUMP_SHA = '3b7efd2a7b6608a3f482374bcecf7273b3586ee5444a96166d2bf2eb6df1de3b'
PATH_NAME = re.compile(r'_GLOBAL__N__[0-9a-f]{8}_|__nv_static_[0-9]+__[0-9a-f]{8}_')


def main():
    build = sys.argv[1] if len(sys.argv) > 1 else 'build_F3'
    rec = json.loads((ART / build / 'BUILD.json').read_text())
    assert rec['status'] == 'PASS' and rec['label'] == 'F3'
    label = build.replace('build_', '')
    result = f1s.compare_builds('F2', F2_MODULE, F2_SHA, label, ROOT / rec['module'], rec['module_sha256'], ART,
                                ART / 'EXPECTED.json', None, out_name=f'SASS_COMPARISON_{label}.json')
    names = {}
    for side in ('F2', label):
        found = set()
        for line in (ART / 'sass' / f'{side}.sass').open():
            if 'Function : ' in line:
                found.update(PATH_NAME.findall(line))
        names[side] = found
    checks = dict(f2_dump_sha256=result['dumps']['F2']['dump_sha256'],
                  f2_dump_equals_f2_own_dump=result['dumps']['F2']['dump_sha256'] == F2_DUMP_SHA,
                  path_derived_names=dict(F2=len(names['F2']), new=len(names[label]),
                                          equal_sets=names['F2'] == names[label],
                                          only_F2=sorted(names['F2'] - names[label])[:20],
                                          only_new=sorted(names[label] - names['F2'])[:20]),
                  comparison=f'SASS_COMPARISON_{label}.json',
                  comparison_sha256=hashlib.sha256((ART / f'SASS_COMPARISON_{label}.json').read_bytes()).hexdigest(),
                  status=result['status'], counts=result['counts'],
                  symbol_suffix_renames=result['symbol_suffix_renames'],
                  symbol_only_in_base=result['symbol_only_in_base'], symbol_only_in_new=result['symbol_only_in_new'],
                  differing=result['differing'], unexpected=result['unexpected_differences'],
                  expected_but_not_seen=result['expected_but_not_seen'])
    with (ART / f'SASS_CHECKS_{label}.json').open('x') as f:
        json.dump(checks, f, indent=2)
        f.write('\n')
    print(json.dumps(checks, indent=2))
    sys.exit(0 if result['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
