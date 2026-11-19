"""Copy reused harnesses and rebind writers before execution (L40).
Prior art: Project-Tensor PF2 (2026) harness, copied unchanged except paths/imports.
"""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/apa_pf2f1'


def main():
    rows = []
    for old, new, extra in [
        ('apa_p1pf2_gpu.py', 'apa_pf2f1_reference.py', [('import apa_p1pf2_build as pfb', 'import apa_pf2f1_build as pfb')]),
        ('apa_p1pf2_g45.py', 'apa_pf2f1_g45.py', [('import apa_p1pf2_gpu as G', 'import apa_pf2f1_common as G')]),
        ('apa_p1pf2_sass.py', 'apa_pf2f1_sass_reference.py', [('import apa_p1pf2_build as pfb', 'import apa_pf2f1_build as pfb')]),
    ]:
        src = ROOT / 'scripts' / old
        text = src.read_text()
        pairs = [('artifacts/apa_p1pf2', 'artifacts/apa_pf2f1'), ('/tmp/apap1pf2_', '/tmp/pf2f1_'), *extra]
        for a, b in pairs:
            text = text.replace(a, b)
        dest = ROOT / 'scripts' / new
        with dest.open('x') as f:
            f.write(text)
        rows.append(dict(source=str(src.relative_to(ROOT)), source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                         copy=str(dest.relative_to(ROOT)), copy_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                         substitutions=pairs))
    result = dict(copies=rows, outputs=[
        'artifacts/apa_pf2f1/build_*/**', 'artifacts/apa_pf2f1/arms/**',
        'artifacts/apa_pf2f1/gpu/**', 'artifacts/apa_pf2f1/identity/**',
        'artifacts/apa_pf2f1/coverage/**', 'artifacts/apa_pf2f1/nonfinite/**',
        'artifacts/apa_pf2f1/regression/**', 'artifacts/apa_pf2f1/sass/**',
        'artifacts/apa_pf2f1/*.json', 'artifacts/apa_pf2f1/*.md',
        '/tmp/pf2f1_shim/**', '/tmp/pf2f1_index*/**', 'logs/apa_pf2f1_*'],
        imported_read_only=['scripts/apa_p1f1_gpu.py (fake extent fixture only)',
                           'scripts/apa_p1d_sass.py (not invoked; parser reference import only)',
                           'artifacts/apa_p1pf2/arms/PF/** (finite baseline, imported -B)',
                           'artifacts/apa_p1pf2/a1/PF_I3_A1.json (exact per-dtype tau)'],
        unused_writers='Copied PF2 reference modules retain unused CLI functions; only pure grid/input/oracle/parser helpers are called. No original PF2 writer is called.')
    with (ART / 'COPY_DEPS.json').open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print('COPIED', [r['copy'] for r in rows])


if __name__ == '__main__':
    main()
