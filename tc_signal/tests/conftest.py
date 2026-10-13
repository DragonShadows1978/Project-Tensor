"""tc_signal test session hooks: an optional JSON receipt of every census cell's N and determinism.

Set TC_SIGNAL_TEST_RECEIPT=<path> to write it (the gate itself is the assertions; the receipt is the evidence table).
This file imports neither torch nor tc_signal: the narrow skip guard lives in the test modules.
"""
import json
import os

import pytest

_RECORDS = []


@pytest.fixture(scope='session')
def receipt():
    return _RECORDS


def pytest_sessionfinish(session, exitstatus):
    path = os.environ.get('TC_SIGNAL_TEST_RECEIPT')
    if path and _RECORDS:
        with open(path, 'w') as f:
            json.dump({'exitstatus': int(exitstatus), 'cells': _RECORDS}, f, indent=1)
            f.write('\n')
