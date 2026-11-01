"""D6 full real-test CPU executions, separate from synthetic GPU staging.
Prior art: PEP 357 (2006); equivalent mutation analysis Offutt/Voas
(year unverified, lead to check); pytest (Krekel et al., 2004+).
"""
import contextlib
import json
import os
from pathlib import Path
import sys
import pytest
from cpu_shim import install
from run_gpu_tests import Reports
from runtime_evidence import npbool_index_check

TOOL=Path(__file__).resolve().parent
ROOT=TOOL.parents[2]
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
label=sys.argv[1]
assert label in ('E_npbool_g','E_npbool_c')
install(TOOL/'arms'/label/'tensor_cuda/nn.py')
plugin=Reports()
live=npbool_index_check()
with (TOOL/(label+'_full_cpu.log')).open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
    rc=int(pytest.main(['-q','-p','no:cacheprovider','--import-mode=importlib','--rootdir='+str(ROOT),
                       str(ROOT/'tensor_cuda/tests/test_groupnorm_gn1.py')],plugins=[plugin]))
result=dict(label=label,exit=rc,live=live,collected_ids=plugin.ids,reports=plugin.reports,
            collection_errors=plugin.collection_errors,
            passed=rc==0 and not plugin.collection_errors and len(plugin.ids)==405
                   and all(r['outcome']=='passed' for r in plugin.reports)
                   and live['accepts_numpy_bool'] is False and live['numpy_version']=='2.4.2',
            evidence_class='real tests with CPU stand-in; no GPU claim')
(TOOL/(label+'_full_cpu.json')).write_text(json.dumps(result,indent=2)+'\n')
print(label,rc,len(plugin.ids),live)
sys.exit(0 if result['passed'] else 1)
