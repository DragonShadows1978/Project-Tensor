"""Register concrete GPU invocations from the sealed F2 case split.

Prior art: PF2-F1/F2 (Project-Tensor, 2026), same case indices and SHA-256
launch manifests. Add the tc cases, suite smoke and isolated mutant lines.
"""
import hashlib
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import apa_pf2f3_build as B
import apa_pf2f3_cpu as C
import apa_pf2f3_gpu as G
import apa_pf2f3_identity_reference as I
import apa_pf2f3_nonfinite as N
import apa_pf2f3_p1_regression as P

F2_LINES = B.ROOT / 'artifacts/apa_pf2f2/LINES.json'
F2_LINES_SHA = '6207c4c7122a4e5eec5bb4660249eea2725634f0ac08ad473dea7d5dac093d04'


def stage():
    assert B.sha(F2_LINES) == F2_LINES_SHA
    assert B.sha(B.ART / 'REGISTRATION.md') == (B.ART / 'REGISTRATION.sha256').read_text().split()[0]
    inherited = B.read(F2_LINES)
    counts = dict(public=len(I.l1_cases()), raw=len(G.raw_cases()), extra=len(G.extra_cases()),
                  prefill=len(I.g1_cases()), tc=len(G.tc_finite_cases()))
    assert counts == dict(public=512, raw=112, extra=128, prefill=4848, tc=224)
    suites = dict(sink=len(N.sink_cases()), norm=len(N.norm_cases()),
                  partition=len(N.partition_norm_cases()), tc=len(G.tc_norm_cases()))
    assert suites == dict(sink=16, norm=16, partition=44, tc=8)
    assert len(P.lf2_cases()) == 32
    modules = {label: C.precheck(label)['module_sha256'] for label in C.LABELS}
    assert modules['F2'] == B.F2_MODULE_SHA
    rows = {}

    def add(name, mode, min_s, labels=('F3',), **data):
        assert name not in rows and 30 <= min_s <= 510
        rows[name] = dict(mode=mode, min_s=min_s, max_s=540,
                          modules={label: modules[label] for label in labels}, **data)

    for suite, min_s in (('sink', 30), ('norm', 60), ('partition', 90), ('tc', 60)):
        add(f'G2_{suite}', 'nonfinite', min_s, suite=suite, cases=suites[suite])
    for name, info in inherited.items():
        if name.startswith('G1_'):
            add(name, 'identity', info['min_s'], ('F2', 'F3'),
                suite=info['suite'], indices=info['indices'])
    for index, start in enumerate(range(0, counts['tc'], 56)):
        add(f'G1_tc_{index:02d}', 'identity', 120, ('F2', 'F3'),
            suite='tc', indices=list(range(start, start + 56)))
    for suite, count in counts.items():
        actual = [i for row in rows.values() if row['mode'] == 'identity' and row['suite'] == suite
                  for i in row['indices']]
        assert sorted(actual) == list(range(count)), suite

    for i, (mutant, suite, failures) in enumerate(zip(
        B.MUTANTS, ('norm', 'norm', 'tc'),
        (('inf:prefill', 'inf:parity', '-inf:prefill', '-inf:parity'),
         ('-inf:prefill', '-inf:parity'), ('inf:output', '-inf:output')),
        strict=True,
    ), 1):
        add(f'G3_pf3m{i:02d}', 'mutation', 120, ('F3', mutant), mutant=mutant,
            suite=suite, cases=suites[suite], expected_failures=list(failures))
    add('G4_pytest', 'pytest', 120)
    add('G5_regression', 'regression', inherited['G4_regression']['min_s'])
    assert len(rows) == 69, len(rows)

    B.save(B.ART / 'LINES.json', rows)
    scripts = sorted((B.ROOT / 'scripts').glob('apa_pf2f3_*.py'))
    evidence = [B.ART / name for name in ('LINES.json', 'REGISTRATION.md', 'REGISTRATION.sha256', 'COPY_DEPS.json')]
    local = [F2_LINES, P.PACKED_TABLE, B.ROOT / 'tensor_cuda/tests/test_apa_pf2_nonfinite.py',
             *scripts, *evidence, *(C.arm_dir(label) / 'ARM.json' for label in C.LABELS)]
    pins = {str(path.relative_to(B.ROOT)): B.sha(path) for path in local}
    pins[str(P.DENSE_TABLE)] = B.sha(P.DENSE_TABLE)
    assert pins[str(P.DENSE_TABLE)] == P.DENSE_TABLE_SHA
    forge = shutil.which('forge-gpu-run')
    assert forge
    B.save(B.ART / 'LINE_PINS.json', dict(files=pins, forge_sha256=B.sha(forge)))
    print('STAGED', len(rows), 'first G2_sink', counts, suites)


def dry_readback():
    rows = {}
    for name, spec in B.read(B.ART / 'LINES.json').items():
        prefix = B.ART / 'gpu' / f'{name}_dry'
        row = B.read(prefix.with_suffix('.exit.json'))
        start = B.read(prefix.with_suffix('.start.json'))
        admission, = row['admission']
        assert row['exit'] == 78 and row['dry'] and admission['would_start'] is True
        assert not admission['self_leasing'] and not row['lease_attempted']
        assert admission['min'] == spec['min_s'] <= 510 and admission['max'] == spec['max_s'] == 540
        assert admission['margin'] == 30
        assert start['module_sha256'] == spec['modules']
        assert B.sha(prefix.with_suffix('.log')) == row['log_sha256']
        rows[name] = row
    B.save(B.ART / 'DRY_RUNS.json', rows)
    print('DRY RUN READBACK', len(rows), 'exit 78, would_start true, zero leases')


if __name__ == '__main__':
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    {'stage': stage, 'check-dry': dry_readback}[sys.argv[1]]()
