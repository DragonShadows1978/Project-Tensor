#!/usr/bin/env python3
"""Candidate CPU subset, same binding and IDs as REV2/A1 (2026).
Prior art: pytest hooks/JUnit and ENG1 arm_origin.py. Taken: real pinned-arm
execution, preflight and exact-ID census; A2 changes receipt/scratch locations.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

A = Path(__file__).resolve().parent
R = A.parents[2]
T = R / 'artifacts/eng1/tool'
pkg = T / 'arms/candidate/tensor_cuda'
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
sys.path.insert(0, str(T))
from arm_origin import bound_environment, preflight
os.environ.update(bound_environment(pkg))
pin = '3e6d5805fb8b89ebfeb2b32c173fd097ef1cfe4a01b3349dc177d4b0c8c0ad53'
module = '_tensor_cuda.cpython-312-x86_64-linux-gnu.so'
preflight(pkg, module, pin, os.getcwd(), A / 'cpu_origin.json')
sys.path.insert(0, str(pkg))
import tensor_cuda as tc
import pytest
from evidence import norm_id

class Reports:
    ids = []
    reports = []
    def pytest_collection_finish(self, session):
        self.ids = [norm_id(i.nodeid) for i in session.items]
    def pytest_runtest_logreport(self, report):
        self.reports.append(dict(id=norm_id(report.nodeid), when=report.when, outcome=report.outcome))

test = pkg / 'tests/test_eng1.py'
assert test.read_bytes() == (R / 'tensor_cuda/tests/test_eng1.py').read_bytes()
p = Reports()
rc = int(pytest.main(['-q', '-p', 'no:cacheprovider', '--import-mode=importlib',
    '--rootdir=' + str(R), '--basetemp=/tmp/eng1_a2_work/cpu_bt',
    '--junitxml=' + str(A / 'cpu.xml'), '-k', 'cpu', str(test)], plugins=[p]))
passed = [r['id'] for r in p.reports if r['when'] == 'call' and r['outcome'] == 'passed']
failures = [r for r in p.reports if r['outcome'] != 'passed']
expected = (T / 'ids_a1_audit.txt').read_text().splitlines()
result = dict(pytest_exit=rc, package=tc.__file__, engine=tc._C.__file__,
    engine_sha256=hashlib.sha256(Path(tc._C.__file__).read_bytes()).hexdigest(),
    collected_ids=p.ids, reports=p.reports, passed=len(passed), non_pass_reports=failures,
    exact_ids_match=sorted(p.ids) == sorted(expected), CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'])
(A / 'cpu_result.json').write_text(json.dumps(result, indent=2) + '\n')
assert rc == 0 and len(passed) == 47 and not failures and result['exact_ids_match']
assert result['engine_sha256'] == pin
print('CPU subset: 47 passed / 47 exact IDs; candidate pin verified')
