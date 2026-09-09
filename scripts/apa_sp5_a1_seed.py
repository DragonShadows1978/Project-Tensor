#!/usr/bin/env python3
"""Seed amendment-1 per-window cells 0..3 for arms A/B/C/D/E from the r1
N=4 aggregate receipts, and the A32 window-0..3 partners from the r1 floor
receipts. CPU only; no GPU, no lease.

This is legitimate ONLY because the per-window runner was verified to
reproduce the r1 numbers to the last digit on two independent checks
(A/w04 path, and C/w01 = 2341.776919703325 exactly). Each seeded cell
records provenance naming its r1 source file and sha256.
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np

A = Path('/mnt/ForgeRealm/Project-Tensor-wt-apa-sp5/artifacts/apa_sp5')
OUT = A / 'windows_a1'
REG = '3cc3b3e112479ad71077a98a221fd5af521013d16ce423d52afe308e9ae58159'
AMD = '06a5a1e742ae2474469f8d5177bc5b9d35b4d89e9553d928c9cac1d7a6753e33'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def emit(cell, rec):
    p = OUT / f'{cell}.json'
    if p.exists():
        return 'exists'
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(rec, f, indent=2, allow_nan=False); f.write('\n')
    return 'written'


def main():
    made = {}
    for arm in 'ABCDE':
        src = A / f'ppl_{arm}_W1024_N4_b4.json'
        d = json.loads(src.read_text())
        for w in d['windows']:
            cell = f"ppl_{arm}_W1024_w{w['window']:02d}"
            rec = dict(cell=cell, arm=arm, window=w['window'], lo=w['lo'],
                       W=1024, half=512, delta=d['delta'],
                       bulk_bits=d['bulk_bits'],
                       refine_percentile=d['refine_percentile'],
                       registration_sha256=REG, amendment='APA_SP5_AMENDMENT_1',
                       amendment_sha256=AMD, token_sha256=d['token_sha256'],
                       targets=w['targets'], mean_nll=w['mean_nll'],
                       ppl=w['ppl'], nll_sum=w['mean_nll'] * w['targets'],
                       backend_full=d['backend_full'],
                       backend_sliding=d['backend_sliding'],
                       forward_wall_s=w['wall_s'], gpu_mib=w['gpu_mib'],
                       status='PASS',
                       provenance=dict(
                           kind='seeded_from_r1_aggregate',
                           source=str(src.relative_to(A.parent.parent)),
                           source_sha256=sha(src),
                           justification='the amendment-1 per-window runner was '
                                         'verified to reproduce this exact code '
                                         'path to the last digit (C/w01 = '
                                         '2341.776919703325 on both); reusing '
                                         'the r1 receipt avoids recomputing an '
                                         'identical number'))
            made[cell] = emit(cell, rec)
    # A32 partners for windows 0..3 from the r1 N=4 floor receipt
    src = A / 'item1_sensitivity_W1024_win0_N4.json'
    d = json.loads(src.read_text())
    for w, ppl in enumerate(d['A32_fp32_full']['per_window_ppl']):
        cell = f'ppl_A32_W1024_w{w:02d}'
        rec = dict(cell=cell, arm='A32', window=w, lo=w * 1024, W=1024,
                   half=512, delta=None, bulk_bits=4, refine_percentile=0.15,
                   registration_sha256=REG, amendment='APA_SP5_AMENDMENT_1',
                   amendment_sha256=AMD, token_sha256=d['token_sha256'],
                   targets=512, mean_nll=float(np.log(ppl)), ppl=float(ppl),
                   nll_sum=float(np.log(ppl)) * 512,
                   backend_full='standard_sink_fp32',
                   backend_sliding='standard_sink_sliding_chunked',
                   status='PASS',
                   provenance=dict(
                       kind='seeded_from_r1_floor_N4',
                       source=str(src.relative_to(A.parent.parent)),
                       source_sha256=sha(src),
                       justification='same construction and same code path as '
                                     'the amendment-1 A32 cell'))
        made[cell] = emit(cell, rec)
    print(json.dumps(dict(written=sum(1 for v in made.values() if v == 'written'),
                          existed=sum(1 for v in made.values() if v == 'exists'),
                          cells=sorted(made)), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
