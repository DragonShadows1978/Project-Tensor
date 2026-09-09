#!/usr/bin/env python3
"""HOUSE_RULES §8 registered mutation baseline (Project-Tensor, 2026).

Prior art: mutation testing (DeMillo, Lipton & Sayward, 1978; unverified — lead
to check those search terms). Five deliberate defects in temporary source copies;
no production edits or blind-verification claim. Original source is hash checked.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

from apa_spd1_common import ROOT, ART, registration, sha, write_json


def main():
    registration()
    path = ROOT / 'scripts/apa_spd1_common.py'
    before = sha(path); source = path.read_text()
    output = ART / 'mutations' / str(time.time_ns())
    output.mkdir(parents=True)
    mutations = {
        'top_left_causal': ('np.arange(S)[None, :] <= S - L + np.arange(L)[:, None]',
                            'np.arange(S)[None, :] <= np.arange(L)[:, None]'),
        'wrong_gqa_mapping': ('tc.matmul(wg, v).reshape', 'tc.matmul(wg, v.flip(1)).reshape'),
        'zero_reference_false_pass': ('(0.0 if num == 0 else None)', '(0.0 if num == 0 else 0.0)'),
        'nonfinite_hidden': ('nonfinite_got=bad_got,', 'nonfinite_got=0,'),
        'seven_samples_not_enforced': ('len(arr) < 7', 'len(arr) < 1'),
    }
    rows = []
    with tempfile.TemporaryDirectory(prefix='apa_spd1_mutations_') as temp:
        for name, (old, new) in mutations.items():
            if source.count(old) != 1:
                raise RuntimeError(f'mutant anchor count is not one: {name}')
            candidate = Path(temp) / f'{name}.py'; candidate.write_text(source.replace(old,new))
            xml = Path(temp) / f'{name}.xml'
            env = dict(os.environ, APA_SPD1_COMMON_OVERRIDE=str(candidate), PYTHONDONTWRITEBYTECODE='1',
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            proc = subprocess.run(['timeout','40s',sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',
                'tensor_cuda/tests/test_apa_spd1.py',f'--junitxml={xml}'], cwd=ROOT,env=env,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=45)
            log = output / f'{name}.log'; log.write_text(proc.stdout)
            failures, errors = 0, 1
            if xml.exists():
                tree = ET.parse(xml)
                failures = len(tree.findall('.//failure')); errors = len(tree.findall('.//error'))
            killed = proc.returncode == 1 and failures > 0 and errors == 0
            rows.append(dict(name=name, exit_code=proc.returncode, assertion_failures=failures,
                             errors=errors, killed=killed, log=str(log.relative_to(ROOT))))
    nonerror = [r for r in rows if r['errors'] == 0 and r['exit_code'] in (0,1)]
    fraction = sum(r['killed'] for r in nonerror) / len(nonerror) if nonerror else 0
    unchanged = sha(path) == before
    result = dict(evidence_class='author-run CPU mutation tests; no blind verification', mutants=rows,
                  nonerror=len(nonerror), kill_fraction=fraction, threshold=.80,
                  source_unchanged=unchanged, source_sha256=before,
                  status='PASS' if unchanged and len(nonerror) == 5 and fraction >= .80 else 'FAIL')
    write_json(output / 'result.json', result, exclusive=True)
    result['receipt'] = str((output / 'result.json').relative_to(ROOT))
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
