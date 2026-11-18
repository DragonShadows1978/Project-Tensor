"""Prepare the fixed PF2-F2 GPU line split and read back dry-run receipts.

Prior art: F1's max-540 bounded line plan and manifests (2026).
Retain its full prefill splits, and split decode over independent case IDs.
"""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import apa_pf2f2_build as B
import apa_pf2f2_gpu as G
import apa_pf2f2_identity_reference as I
import apa_pf2f2_nonfinite as N
import apa_pf2f2_p1_regression as P


def stage():
    assert len(I.l1_cases()) == 512 and len(I.g1_cases()) == 4848
    assert len(G.raw_cases()) == 112 and len(G.extra_cases()) == 128
    assert len(P.lf2_cases()) == 32
    assert len(N.sink_cases()) == len(N.norm_cases()) == 16
    assert len(N.partition_norm_cases()) == 44
    bases = ('F1','F2')
    rows = {}

    def line(name, mode, min_s, labels=('F2',), **data):
        assert name not in rows and 30 <= min_s <= 510
        rows[name] = dict(mode=mode, min_s=min_s, max_s=540, **data,
                          modules={label:B.receipt(label)['module_sha256'] for label in labels})

    line('G2_sink','nonfinite',30,suite='sink',cases=16)
    line('G2_norm','nonfinite',30,suite='norm',cases=16)
    line('G2_partition','nonfinite',60,suite='partition',cases=44)
    for suite, count, batch in [('public',512,128),('raw',112,28),('extra',128,64)]:
        for j,start in enumerate(range(0,count,batch)):
            line(f'G1_{suite}_{j:02d}','identity',120,bases,suite=suite,indices=list(range(start,min(start+batch,count))))
    reference = B.ROOT / 'artifacts/apa_pf2f1/LINES.json'
    split = [info['indices'] for name,info in B.read(reference).items() if name.startswith('G1_')]
    assert len(split) == 46 and sorted(i for group in split for i in group) == list(range(4848))
    for j, indices in enumerate(split):
        line(f'G1_prefill_{j:03d}','identity',500,bases,suite='prefill',indices=indices)
    for label, g in ((B.BULK, 1), (B.SHARED, 4)):
        assert sum(c['G'] == g for c in N.norm_cases()) == 8
        line(f'G3_{B.MUTANTS[label]}','mutation',90,('F2',label),mutant=label,group=g)
    line('G4_regression','regression',120)
    B.save(B.ART / 'LINES.json', rows)
    scripts = sorted((B.ROOT / 'scripts').glob('apa_pf2f2_*.py'))
    artifacts = [B.ART / name for name in ('LINES.json','REGISTRATION.md','REGISTRATION.sha256',
                 'DISPATCH_TRACE.json','COPY_DEPS.json')]
    external = [reference, P.DENSE_TABLE]
    local = [P.PACKED_TABLE, B.ROOT / 'tensor_cuda/tests/test_apa_pf2_nonfinite.py',
             *scripts, *artifacts, *(B.ART / 'arms' / label / 'ARM.json' for label in ('F1','F2',*B.MUTANTS))]
    files = {str(path.relative_to(B.ROOT)):B.sha(path) for path in local}
    files.update({str(path):B.sha(path) for path in external})
    B.save(B.ART / 'LINE_PINS.json', dict(files=files, forge_sha256=B.sha(shutil.which('forge-gpu-run'))))
    print('STAGED',len(rows),'lines, first',next(iter(rows)), 'prefill 4848, decode 752, G2 76')


def dry_readback():
    results = {}
    for name in B.read(B.ART / 'LINES.json'):
        prefix = B.ART / 'gpu' / f'{name}_dry'
        row = B.read(prefix.with_suffix('.exit.json'))
        assert row['exit'] == 78 and row['dry'] and row['admission'][0]['would_start']
        assert not row['lease_attempted']
        assert B.sha(prefix.with_suffix('.log')) == row['log_sha256']
        results[name] = row
    B.save(B.ART / 'DRY_RUNS.json', results)
    print('DRY RUN READBACK', len(results), 'exit 78, would_start true, zero leases')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('stage','check-dry'))
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    {'stage':stage, 'check-dry':dry_readback}[p.parse_args().action]()
