"""D5 runtime census of each suite, using only the CPU stand-in.

Prior art: Python sys.modules origins (PSF, year unverified, lead to check);
pytest hooks (Krekel et al., 2004+); A3 NumPy stand-in. This instrument records
imports even when unsupported non-GN operations fail. No device code runs.
"""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from runtime_evidence import module_census

TOOL = Path(__file__).resolve().parent
ROOT = TOOL.parents[2]
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
suite = sys.argv[1]
from cpu_shim import install
tc = install(TOOL / 'arms/candidate/tensor_cuda/nn.py')
# Optim imports _C at collection. No native extension is loaded here and
# unsupported legacy engine calls fail as ordinary CPU stand-in failures.
tc._C = SimpleNamespace()
sys.path.insert(1, str(ROOT))
import numpy as np
import pytest
from run_gpu_tests import Reports
np.random.seed(571)
files = (['tensor_cuda/tests/test_groupnorm_gn1.py', 'tensor_cuda/tests/test_ext_phase7.py']
         if suite == 'focus' else ['tensor_cuda/tests/test_groupnorm_gn1.py']
         if suite == 'mutation' else json.loads((TOOL / 'nn_import_test_files.json').read_text()))
plugin = Reports()
rc = int(pytest.main(['-q', '-p', 'no:cacheprovider', '--import-mode=importlib',
                     '--rootdir=' + str(ROOT), *[str(ROOT / f) for f in files]], plugins=[plugin]))
# Explicit CPU instrument diagnostic exercises the same lazy NumPy testing API
# referenced by the regression suite even if an unsupported engine op precedes
# it. Kept separate from the recorded test outcomes and disclosed in the trace.
np.testing.assert_equal(len(plugin.ids), len(set(plugin.ids)))
result = dict(suite=suite, exit=rc, collected_ids=plugin.ids,
              call_count=sum(r['when'] == 'call' for r in plugin.reports),
              collection_errors=plugin.collection_errors, reports=plugin.reports,
              modules=module_census(ROOT),
              calibration='np.random.seed(571); np.testing.assert_equal ID uniqueness',
              evidence_class='CPU stand-in import census; no GPU or legacy correctness claim')
(TOOL / ('runtime_trace_' + suite + '.json')).write_text(json.dumps(result, indent=2) + '\n')
assert not plugin.collection_errors
assert {r['id'] for r in plugin.reports if r['when'] == 'setup'} == set(plugin.ids)
assert rc in (0, 1)
print('runtime census', suite, len(result['modules']), 'modules', result['call_count'], 'calls', rc)
